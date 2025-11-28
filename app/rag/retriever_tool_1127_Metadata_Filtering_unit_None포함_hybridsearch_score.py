import os
import sys
import requests
import json
import time
import re
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import defaultdict

# LangChain Imports
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.tools import tool
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 설정 및 상수 정의
# ============================================================

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
INDEX_NAME = "card_benefit_bgem3_v2"

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct:novita" 
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large" # API 안정성 위해 e5 사용

# ============================================================
# 0. 카테고리 사전 로드
# ============================================================

try:
    from category_dictionary import KNOWN_CATEGORIES
    print(f"✅ 카테고리 사전을 로드했습니다. ({len(KNOWN_CATEGORIES)}개 항목)")
except ImportError:
    print("⚠️ 'category_dictionary.py'를 찾을 수 없습니다. 빈 리스트를 사용합니다.")
    KNOWN_CATEGORIES = []

# ============================================================
# 1. 유틸리티
# ============================================================

def safe_print(text):
    try: print(text)
    except: pass

def sanitize_text(text: str) -> str:
    if not isinstance(text, str): return str(text)
    try: return text.encode('utf-8', 'ignore').decode('utf-8')
    except: return ""

def query_hf_chat_api(payload):
    chat_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.post(chat_url, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        print(f"❌ Chat API 요청 실패: {e}")
        return None

# ============================================================
# 2. Custom Embedding Class
# ============================================================

class CustomHFEmbeddings(Embeddings):
    def __init__(self, api_key, model_id):
        self.api_key = api_key
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.model_id = model_id

    def _query(self, texts: List[str]) -> Any:
        clean_texts = [sanitize_text(t) for t in texts]
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json={"inputs": clean_texts, "options": {"wait_for_model": True}}
            )
            return response.json()
        except Exception as e:
            print(f"❌ [API 통신 에러] {e}")
            return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._query(texts)
        if isinstance(result, list): return result
        return []

    @lru_cache(maxsize=1000)
    def embed_query(self, text: str) -> List[float]:
        if "e5" in self.model_id: text = f"query: {sanitize_text(text)}"
        else: text = sanitize_text(text)
            
        result = self._query([text])
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list): return result[0]
            elif isinstance(result[0], float): return result
        return []

# ============================================================
# 3. 통합 분석 함수 (LLM)
# ============================================================

@traceable(name="0. Analyze Query Unified", run_type="llm")
def analyze_query_unified(original_query: str) -> Dict:
    found_categories = []
    query_lower = original_query.lower()
    for cat in KNOWN_CATEGORIES:
        if cat.lower() in query_lower:
            found_categories.append(cat)
    
    if not HF_API_KEY:
        return {"brands": [], "categories": list(set(found_categories)), "expanded_queries": [original_query]}

    categories_str = ", ".join(KNOWN_CATEGORIES)
    messages = [
        {
            "role": "system",
            "content": "You are an expert in search query analysis. Extract information from the user's query and output it in JSON format."
        },
        {
            "role": "user",
            "content": f"""
사용자의 검색어를 분석하여 아래 3가지 정보를 JSON 포맷으로 추출해줘.

1. **brands**: 질문에 언급된 브랜드나 서비스명 (예: "스벅"->"스타벅스", "톡톡O"). 일반 명사 제외.
2. **categories**: 아래 [카테고리 목록] 중 질문과 가장 관련 있는 것들.
3. **expanded_queries**: 검색 정확도를 높이기 위한 확장 검색어 **최대 3개**.

[카테고리 목록]
{categories_str}

[사용자 질문]
"{original_query}"

[출력 예시]
{{
  "brands": ["스타벅스"],
  "categories": ["카페"],
  "expanded_queries": ["스타벅스 할인 카드", "카페 혜택"]
}}

반드시 JSON 형식만 출력해.
"""
        }
    ]

    payload = {"model": MODEL_NAME, "messages": messages, "max_tokens": 500, "temperature": 0.1, "response_format": {"type": "json_object"}}
    response_data = query_hf_chat_api(payload)
    
    result = {"brands": [], "categories": list(set(found_categories)), "expanded_queries": [original_query]}

    if response_data and "choices" in response_data:
        try:
            content = response_data["choices"][0]["message"]["content"]
            clean_text = content.replace("```json", "").replace("```", "").strip()
            clean_text = sanitize_text(clean_text)
            parsed = json.loads(clean_text)
            
            if "brands" in parsed: result["brands"] = parsed["brands"]
            if "categories" in parsed: 
                valid_cats = [c for c in parsed["categories"] if c in KNOWN_CATEGORIES]
                result["categories"].extend(valid_cats)
            if "expanded_queries" in parsed:
                cleaned_queries = [sanitize_text(q) for q in parsed["expanded_queries"]]
                cleaned_queries.insert(0, sanitize_text(original_query))
                result["expanded_queries"] = cleaned_queries
        except: pass

    result["brands"] = list(set(result["brands"]))
    result["categories"] = list(set(result["categories"]))
    result["expanded_queries"] = list(dict.fromkeys(result["expanded_queries"]))

    return result

# ============================================================
# 4. 하이브리드 검색 구현 (Vector + Keyword + RRF)
# ============================================================

def perform_vector_search(vector_store, query: str, k: int, categories: List[str]) -> List[Any]:
    query_vector = vector_store.embedding.embed_query(query)
    if not query_vector: 
        print(f"⚠️ [Vector Search Skip] 임베딩 실패")
        return []
    
    es_filter = []
    if categories:
        es_filter.append({"terms": {"metadata.category.keyword": categories}})

    knn_query = {
        "field": "vector", 
        "query_vector": query_vector,
        "k": k,
        "num_candidates": max(100, k * 2),
        "filter": es_filter
    }
    
    try:
        # size 파라미터 추가!
        response = vector_store.client.search(index=INDEX_NAME, body={
            "knn": knn_query, "size": k, "_source": ["text", "metadata", "page_content"]
        })
        results = []
        for hit in response["hits"]["hits"]:
            score = hit["_score"]
            source = hit["_source"]
            content = source.get("text") or source.get("page_content") or ""
            doc = Document(page_content=content, metadata=source.get("metadata", {}))
            results.append((doc, score))
        
        print(f"✅ [Vector] '{query}' -> {len(results)}건")
        return results
    except Exception as e:
        print(f"❌ [Vector Error] {e}")
        return []

def perform_keyword_search(vector_store, query: str, k: int, categories: List[str]) -> List[Any]:
    es_filter = []
    if categories:
        es_filter.append({"terms": {"metadata.category.keyword": categories}})

    query_nospace = query.replace(" ", "")
    
    match_query = {
        "bool": {
            "should": [
                {"match": {"text": query}},          
                {"match": {"text": query_nospace}}   
            ],
            "minimum_should_match": 1,
            "filter": es_filter
        }
    }

    try:
        response = vector_store.client.search(index=INDEX_NAME, body={
            "query": match_query, "size": k, "_source": ["text", "metadata", "page_content"]
        })
        results = []
        for hit in response["hits"]["hits"]:
            score = hit["_score"]
            source = hit["_source"]
            content = source.get("text") or source.get("page_content") or ""
            doc = Document(page_content=content, metadata=source.get("metadata", {}))
            results.append((doc, score))
            
        print(f"✅ [Keyword] '{query}' -> {len(results)}건")
        return results
    except: return []

def apply_rrf(vector_results, keyword_results, k=60):
    fusion_scores = defaultdict(float)
    doc_map = {}
    c = 60

    for rank, (doc, score) in enumerate(vector_results):
        doc_id = doc.page_content
        if doc_id not in doc_map:
            doc.metadata["vec_score"] = score 
            doc.metadata["key_score"] = 0.0   
            doc_map[doc_id] = doc
        else:
            doc_map[doc_id].metadata["vec_score"] = score
        fusion_scores[doc_id] += 1 / (rank + c)

    for rank, (doc, score) in enumerate(keyword_results):
        doc_id = doc.page_content
        if doc_id not in doc_map:
            doc.metadata["key_score"] = score 
            doc.metadata["vec_score"] = 0.0   
            doc_map[doc_id] = doc
        else:
            doc_map[doc_id].metadata["key_score"] = score
        fusion_scores[doc_id] += 1 / (rank + c)

    sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for doc_id, rrf_score in sorted_docs[:k]:
        doc = doc_map[doc_id]
        doc.metadata["rrf_score"] = rrf_score 
        final_results.append(doc) 
    return final_results

@traceable(name="1. Retrieve Candidates (Hybrid+Parallel)", run_type="retriever")
def retrieve_candidates(vector_store, queries: List[str], k_per_query: int, categories: List[str]):
    all_docs = []
    unique_queries = list(set([sanitize_text(q) for q in queries if q.strip()]))

    def _run_vector(q): return perform_vector_search(vector_store, q, k_per_query, categories)
    def _run_keyword(q): return perform_keyword_search(vector_store, q, k_per_query, categories)

    with ThreadPoolExecutor(max_workers=10) as executor:
        vec_futures = {executor.submit(_run_vector, q): q for q in unique_queries}
        key_futures = {executor.submit(_run_keyword, q): q for q in unique_queries}
        
        vec_results = {}
        key_results = {}
        
        for f in as_completed(vec_futures): vec_results[vec_futures[f]] = f.result()
        for f in as_completed(key_futures): key_results[key_futures[f]] = f.result()

    for q in unique_queries:
        v_res = vec_results.get(q, [])
        k_res = key_results.get(q, [])
        hybrid_docs = apply_rrf(v_res, k_res, k=k_per_query)
        all_docs.extend(hybrid_docs)
    
    unique_docs_map = {}
    for doc in all_docs:
        if doc.page_content not in unique_docs_map:
            unique_docs_map[doc.page_content] = doc
        else:
            if doc.metadata.get("rrf_score", 0) > unique_docs_map[doc.page_content].metadata.get("rrf_score", 0):
                unique_docs_map[doc.page_content] = doc
                
    return list(unique_docs_map.values())

# ============================================================
# 5. 점수 계산 및 랭킹 (채점표 생성 기능 추가 🚀)
# ============================================================

def calculate_benefit_score(doc, user_brands, user_categories) -> Tuple[float, str]:
    """
    점수와 함께 '계산 내역(Breakdown)'을 반환합니다.
    """
    try:
        base = float(doc.metadata.get("benefit_value", 0))
        unit = doc.metadata.get("benefit_unit", "").strip()
        doc_cat = doc.metadata.get("category", "")
        summary = doc.metadata.get("benefit_summary", "")
        text = (doc.metadata.get("card_name", "") + " " + summary + " " + (doc.page_content or ""))

        score_log = [] # 채점 기록용 리스트

        # 1. 텍스트 마이닝
        extracted_score = 0
        if base < 10:
            pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
            if pct_matches: 
                val = max([float(x) for x in pct_matches]) * 100
                if val > extracted_score: extracted_score = val
                
            krw_matches = re.findall(r'(\d+(?:,\d+)*)\s*원', text)
            if krw_matches: 
                val = max([float(x.replace(',', '')) for x in krw_matches])
                if val > extracted_score: extracted_score = val
            
            man_matches = re.findall(r'(\d+(?:,\d+)*)\s*만원', text)
            if man_matches: 
                val = max([float(x.replace(',', '')) for x in man_matches]) * 10000
                if val > extracted_score: extracted_score = val
            
            if extracted_score > 0:
                score_log.append(f"텍스트추출({extracted_score:.0f})")
        
        # 2. 메타데이터 점수
        meta_score = base * 100 if unit == "%" else base
        if meta_score > 0:
            score_log.append(f"메타데이터({meta_score:.0f})")
        
        # 최종 기본 점수 선정
        score = max(meta_score, extracted_score)
        if score <= 0: 
            score = 1.0
            score_log.append("기본(1.0)")

        # 3. 가중치
        if user_brands and any(b in text for b in user_brands): 
            score *= 2.0
            score_log.append("브랜드(x2.0)")
            
        if user_categories:
            if any(c in doc_cat or doc_cat in c for c in user_categories): 
                score *= 1.5
                score_log.append("카테고리일치(x1.5)")
            elif any(c in text for c in user_categories): 
                score *= 1.2
                score_log.append("카테고리포함(x1.2)")

        # 계산 내역 문자열 생성 (예: "메타데이터(10000) + 브랜드(x2.0)")
        breakdown_str = " + ".join(score_log)
        return score, breakdown_str

    except: return 0.0, "Error"

@traceable(name="2. Rerank Candidates", run_type="parser")
def rerank_candidates(docs, user_brands, user_categories):
    docs_scored = []
    for doc in docs:
        score, breakdown = calculate_benefit_score(doc, user_brands, user_categories)
        # doc 객체에 breakdown 정보 심기 (임시)
        doc.metadata["score_breakdown"] = breakdown
        docs_scored.append((doc, score))
    return sorted(docs_scored, key=lambda x: x[1], reverse=True)

def select_final_results_with_merge(sorted_docs, top_k, user_brands, user_categories):
    merged_results = {} 
    for doc, score in sorted_docs:
        card_name = doc.metadata.get("card_name", "")
        if not card_name: continue

        summary = doc.metadata.get("benefit_summary", "")
        cat = doc.metadata.get("category", "")
        breakdown = doc.metadata.get("score_breakdown", "") # 채점표 가져오기
        
        # 검색 점수
        vec_score = doc.metadata.get("vec_score", 0.0)
        key_score = doc.metadata.get("key_score", 0.0)
        rrf_score = doc.metadata.get("rrf_score", 0.0)
        
        if card_name not in merged_results:
            merged_results[card_name] = {
                "card_name": card_name,
                "origin_id": doc.metadata.get("origin_id", ""),
                "summaries": [summary],
                "categories": [cat],
                "total_score": score,
                "match_reason": f"브랜드:{user_brands}, 카테고리:{user_categories}",
                "max_vec": vec_score,
                "max_key": key_score,
                "max_rrf": rrf_score,
                "breakdown_log": [breakdown] # 채점표 로그 리스트
            }
        else:
            if summary not in merged_results[card_name]["summaries"]:
                merged_results[card_name]["summaries"].append(summary)
            if cat not in merged_results[card_name]["categories"]:
                merged_results[card_name]["categories"].append(cat)
            
            merged_results[card_name]["total_score"] += (score * 0.2)
            merged_results[card_name]["breakdown_log"].append(f"추가매칭(+{score*0.2:.0f})") # 가산점 로그
            
            merged_results[card_name]["max_vec"] = max(merged_results[card_name]["max_vec"], vec_score)
            merged_results[card_name]["max_key"] = max(merged_results[card_name]["max_key"], key_score)
            merged_results[card_name]["max_rrf"] = max(merged_results[card_name]["max_rrf"], rrf_score)

    final_list = []
    for info in merged_results.values():
        combined_summary = " / ".join(info["summaries"])
        combined_cat = ", ".join(list(set(info["categories"])))
        # 채점표도 하나로 합치기
        combined_breakdown = " | ".join(list(set(info["breakdown_log"])))
        
        final_list.append({
            "card_name": info["card_name"],
            "score": info["total_score"],
            "benefit_summary": combined_summary,
            "category": combined_cat,
            "match_reason": info["match_reason"],
            "vec_score": info["max_vec"],
            "key_score": info["max_key"],
            "rrf_score": info["max_rrf"],
            "score_breakdown": combined_breakdown # 최종 채점표
        })

    final_list = sorted(final_list, key=lambda x: x["score"], reverse=True)
    return final_list[:top_k]

# ============================================================
# 6. Main Tool
# ============================================================

@tool
def retriever_tool(query: str) -> List[Dict]:
    """Search for credit card benefits using Hybrid Search & Metadata Filtering."""
    try:
        if not HF_API_KEY: return []

        embeddings = CustomHFEmbeddings(api_key=HF_API_KEY, model_id=EMBEDDING_MODEL_ID)
        vector_store = ElasticsearchStore(
            es_url=ELASTICSEARCH_URL, index_name=INDEX_NAME, embedding=embeddings
        )

        analysis = analyze_query_unified(query)
        user_brands = analysis["brands"]
        user_cats = analysis["categories"]
        queries = analysis["expanded_queries"]
        
        safe_print(f"\n👀 [분석] 브랜드:{user_brands}, 카테고리:{user_cats}")
        safe_print(f"   [확장검색어]: {queries}")

        candidates = retrieve_candidates(vector_store, queries, 60, user_cats)
        safe_print(f"   👉 후보군: {len(candidates)}개 문서 검색됨")

        ranked = rerank_candidates(candidates, user_brands, user_cats)
        return select_final_results_with_merge(ranked, 3, user_brands, user_cats)

    except Exception as e:
        safe_print(f"❌ 오류: {e}")
        return []

# ============================================================
# 7. 실행
# ============================================================

if __name__ == "__main__":
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = "CardBenefit RAG Debug"
        safe_print(f"✅ LangSmith Tracing Enabled")
    else:
        safe_print("⚠️ LangSmith API Key not found.")
    safe_print(f"🔍 최종 고도화 검색 테스트 (Score Breakdown Added)")
    safe_print("💡 종료하려면 'q' 입력 또는 Ctrl+C를 누르세요.\n")
    
    while True:
        try:
            q = input("\n💬 입력: ").strip()
        except KeyboardInterrupt:
            safe_print("\n\n👋 프로그램을 강제 종료합니다.")
            break
        except Exception: continue

        if q.lower() in ["q", "quit", "exit"]: break
        if not q: continue

        start = time.perf_counter()
        results = retriever_tool.invoke(q)
        elapsed = time.perf_counter() - start

        safe_print(f"\n⏱️ Time: {elapsed:.4f}s")

        if results:
            safe_print(f"🏆 [추천 결과 Top {len(results)}]")
            for i, res in enumerate(results):
                safe_print(f"{i+1}. {res['card_name']} (점수: {res['score']:.0f})")
                safe_print(f"   - 혜택: {res['benefit_summary']}")
                # 상세 채점표 출력 🚀
                safe_print(f"   - 📝 채점표: {res['score_breakdown']}")
                safe_print(f"   - [검색] V:{res['vec_score']:.4f} / K:{res['key_score']:.4f} / RRF:{res['rrf_score']:.4f}")
        else:
            safe_print("⚠️ 결과 없음")