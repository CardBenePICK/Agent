import os
import sys
import requests
import json
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

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

# 1. LLM 모델 선택
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct:novita" 

# 2. 임베딩 모델 교체 (API 오류 해결을 위한 조치)
# BAAI/bge-m3 API가 현재 불안정하여, 동일한 1024차원 모델인 e5-large로 교체합니다.
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large"

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
# 1. Custom Embedding Class (URL 수정됨 🚀)
# ============================================================

class CustomHFEmbeddings(Embeddings):
    def __init__(self, api_key, model_id):
        self.api_key = api_key
        # 🚨 [수정됨] Router URL 형식 적용
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _query(self, texts: List[str]) -> Any:
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json={"inputs": texts, "options": {"wait_for_model": True}}
            )
            return response.json()
        except Exception as e:
            print(f"❌ HuggingFace API Connection Error: {e}")
            return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._query(texts)
        if isinstance(result, list):
            return result
        return []

    @lru_cache(maxsize=1000)
    def embed_query(self, text: str) -> List[float]:
        # e5 모델은 query 앞에 'query: ' 접두어를 붙이는 것이 성능에 좋음
        if "e5" in EMBEDDING_MODEL_ID:
            text = f"query: {text}"
            
        result = self._query([text])
        
        # 결과 검증 및 에러 핸들링
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list): 
                return result[0]
            elif isinstance(result[0], float): 
                return result
        
        # 에러 로그 출력 (디버깅용)
        if isinstance(result, dict) and 'error' in result:
            print(f"\n❌ [Embedding Failed] API Error: {result['error']}")
        
        return []

# ============================================================
# 2. 유틸리티
# ============================================================

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            print(text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
        except:
            pass

def query_hf_chat_api(payload):
    # LLM용 Router URL
    chat_url = "https://router.huggingface.co/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(chat_url, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        print(f"❌ Chat API 요청 실패: {e}")
        return None

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

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    response_data = query_hf_chat_api(payload)
    
    result = {
        "brands": [],
        "categories": list(set(found_categories)),
        "expanded_queries": [original_query]
    }

    if response_data and "choices" in response_data:
        try:
            content = response_data["choices"][0]["message"]["content"]
            clean_text = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_text)
            
            if "brands" in parsed: result["brands"] = parsed["brands"]
            if "categories" in parsed: 
                valid_cats = [c for c in parsed["categories"] if c in KNOWN_CATEGORIES]
                result["categories"].extend(valid_cats)
            if "expanded_queries" in parsed:
                parsed["expanded_queries"].insert(0, original_query)
                result["expanded_queries"] = parsed["expanded_queries"]
        except:
            pass

    result["brands"] = list(set(result["brands"]))
    result["categories"] = list(set(result["categories"]))
    result["expanded_queries"] = list(dict.fromkeys(result["expanded_queries"]))

    return result

# ============================================================
# 4. 하이브리드 검색 (ES Client 직접 호출)
# ============================================================

def perform_hybrid_search(vector_store, query: str, k: int, categories: List[str]) -> List[Any]:
    """
    Elasticsearch Client를 직접 호출하여 num_candidates 파라미터를 확실하게 전달
    """
    
    # 1. 쿼리 임베딩
    query_vector = vector_store.embedding.embed_query(query)
    
    # 임베딩 실패 시 빈 리스트 반환
    if not query_vector:
        return []
    
    # 2. 메타데이터 필터 구성
    es_filter = []
    if categories:
        es_filter.append({"terms": {"metadata.category.keyword": categories}})

    # 3. 안전한 후보군 수 설정 (k보다 커야 함)
    safe_num_candidates = max(100, k * 2)

    # 4. Raw Query Body 생성
    knn_query = {
        "field": "vector", 
        "query_vector": query_vector,
        "k": k,
        "num_candidates": safe_num_candidates,
        "filter": es_filter
    }
    
    body = {
        "knn": knn_query,
        "_source": ["text", "metadata", "page_content"] 
    }

    try:
        # 5. Client 직접 호출
        response = vector_store.client.search(index=INDEX_NAME, body=body)
        
        # 6. 결과 파싱
        results = []
        for hit in response["hits"]["hits"]:
            score = hit["_score"]
            source = hit["_source"]
            
            content = source.get("text") or source.get("page_content") or ""
            metadata = source.get("metadata", {})
            
            doc = Document(page_content=content, metadata=metadata)
            results.append((doc, score))
            
        return results

    except Exception as e:
        # print(f"⚠️ 검색 실패: {e}")
        return []

# ============================================================
# 5. 점수 계산 및 랭킹 (Parent Document Merge)
# ============================================================

def calculate_benefit_score(doc, user_brands, user_categories):
    try:
        base = float(doc.metadata.get("benefit_value", 0))
        unit = doc.metadata.get("benefit_unit", "").strip()
        doc_cat = doc.metadata.get("category", "")
        text = (doc.page_content or "") + " " + doc.metadata.get("benefit_summary", "")

        if unit == "%": score = base * 100
        else: score = base
        if score <= 0: score = 1.0

        if user_brands and any(b in text for b in user_brands): score *= 2.0
        
        if user_categories:
            if any(c in doc_cat or doc_cat in c for c in user_categories): score *= 1.5
            elif any(c in text for c in user_categories): score *= 1.2

        return score
    except:
        return 0.0

@traceable(name="1. Retrieve Candidates (Parallel)", run_type="retriever")
def retrieve_candidates(vector_store, queries: List[str], k_per_query: int, categories: List[str]):
    all_docs = []
    unique_queries = list(set([q for q in queries if q.strip()]))

    def _single_search(query):
        return perform_hybrid_search(vector_store, query, k_per_query, categories)

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_query = {executor.submit(_single_search, q): q for q in unique_queries}
        
        for future in as_completed(future_to_query):
            results = future.result()
            for doc, score in results:
                if score < 0.1: continue 
                all_docs.append(doc)
    
    return list({doc.page_content: doc for doc in all_docs}.values())

@traceable(name="2. Rerank Candidates", run_type="parser")
def rerank_candidates(docs, user_brands, user_categories):
    docs_scored = []
    for doc in docs:
        score = calculate_benefit_score(doc, user_brands, user_categories)
        docs_scored.append((doc, score))
    return sorted(docs_scored, key=lambda x: x[1], reverse=True)

def select_final_results_with_merge(sorted_docs, top_k, user_brands, user_categories):
    merged_results = {} 

    for doc, score in sorted_docs:
        card_name = doc.metadata.get("card_name", "")
        if not card_name: continue

        summary = doc.metadata.get("benefit_summary", "")
        cat = doc.metadata.get("category", "")
        
        if card_name not in merged_results:
            merged_results[card_name] = {
                "card_name": card_name,
                "origin_id": doc.metadata.get("origin_id", ""),
                "summaries": [summary],
                "categories": [cat],
                "total_score": score,
                "match_reason": f"브랜드:{user_brands}, 카테고리:{user_categories}"
            }
        else:
            if summary not in merged_results[card_name]["summaries"]:
                merged_results[card_name]["summaries"].append(summary)
            if cat not in merged_results[card_name]["categories"]:
                merged_results[card_name]["categories"].append(cat)
            
            merged_results[card_name]["total_score"] += (score * 0.1)

    final_list = []
    for info in merged_results.values():
        combined_summary = " / ".join(info["summaries"])
        combined_cat = ", ".join(list(set(info["categories"])))
        
        final_list.append({
            "card_name": info["card_name"],
            "score": info["total_score"],
            "benefit_summary": combined_summary,
            "category": combined_cat,
            "match_reason": info["match_reason"]
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
        if not HF_API_KEY:
            safe_print("❌ HF_API_KEY 없음")
            return []

        embeddings = CustomHFEmbeddings(api_key=HF_API_KEY, model_id=EMBEDDING_MODEL_ID)

        vector_store = ElasticsearchStore(
            es_url=ELASTICSEARCH_URL,
            index_name=INDEX_NAME,
            embedding=embeddings
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
    # langsmith 추적 활성화
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = "CardBenefit RAG Debug"
        safe_print(f"✅ LangSmith Tracing Enabled")
    else:
        safe_print("⚠️ LangSmith API Key not found.")
    # safe print
    safe_print(f"🔍 최종 고도화 검색 테스트 (Model: {MODEL_NAME})")
    safe_print(f"   - Embed Model: {EMBEDDING_MODEL_ID}")
    safe_print("💡 종료하려면 'q' 입력 또는 Ctrl+C를 누르세요.\n")
    
    while True:
        try:
            q = input("\n💬 입력: ").strip()
        except KeyboardInterrupt:
            safe_print("\n\n👋 프로그램을 강제 종료합니다.")
            break
        except EOFError:
            break
        except Exception:
            continue

        if q.lower() in ["q", "quit", "exit"]:
            safe_print("👋 종료합니다.")
            break
        
        if not q:
            continue

        start = time.perf_counter()
        results = retriever_tool.invoke(q)
        elapsed = time.perf_counter() - start

        safe_print(f"\n⏱️ Time: {elapsed:.4f}s")

        if results:
            for i, res in enumerate(results):
                safe_print(f"{i+1}. {res['card_name']} ({res['score']:.0f}점)")
                cat_str = res.get('category', '카테고리 없음')
                safe_print(f"   - 카테고리: {cat_str}")
                safe_print(f"   - 혜택: {res['benefit_summary']}")
        else:
            safe_print("⚠️ 결과 없음")