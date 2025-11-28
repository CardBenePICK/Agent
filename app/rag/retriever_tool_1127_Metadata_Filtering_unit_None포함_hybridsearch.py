import os
import sys
import requests
import json
import ast
import time
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import re # 정규표현식 모듈 추가

# LangChain & LangSmith Imports
from langchain_huggingface import HuggingFaceEndpointEmbeddings
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

# 1. Llama-3.1 (추천: JSON 구조화 우수)
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct:novita" 

# 2. Gemma-2 (한국어 문맥 이해 우수)
MODEL_NAME = "google/gemma-2-9b-it:nebius"

API_URL = "https://router.huggingface.co/v1/chat/completions"

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
# 1. HF API Wrapper
# ============================================================

def query_hf_api(payload):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        print(f"❌ API 요청 실패: {e}")
        return None

# ============================================================
# 2. 유틸리티 (Safe Print)
# ============================================================

def safe_print(text):
    """윈도우 인코딩 오류 방지 출력 함수"""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            print(text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
        except:
            pass

# ============================================================
# 3. 통합 분석 함수 (One-Shot Extraction + 대소문자 해결)
# ============================================================

@traceable(name="0. Analyze Query Unified", run_type="llm")
def analyze_query_unified(original_query: str) -> Dict:
    """
    LLM 호출 한 번으로 [브랜드, 카테고리, 확장검색어]를 모두 추출합니다.
    """
    
    # [수정] 대소문자 구분 없이 1차 매칭 (ott -> OTT 인식 해결)
    found_categories = []
    query_lower = original_query.lower()
    for cat in KNOWN_CATEGORIES:
        if cat.lower() in query_lower:
            found_categories.append(cat)
    
    if not HF_API_KEY:
        return {
            "brands": [],
            "categories": list(set(found_categories)),
            "expanded_queries": [original_query]
        }

    # 프롬프트 구성
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

1. **brands**: 질문에 언급된 브랜드나 서비스명 (예: "스벅"->"스타벅스", "넷플"->"넷플릭스"). 일반 명사는 제외.
2. **categories**: 아래 [카테고리 목록] 중 질문과 가장 관련 있는 것들. (목록에 없는 단어 사용 금지)
3. **expanded_queries**: 검색 정확도를 높이기 위한 확장 검색어 **최대 3개**. (동의어, 오탈자 교정 등)

[카테고리 목록]
{categories_str}

[사용자 질문]
"{original_query}"

[출력 예시]
{{
  "brands": ["스타벅스", "넷플릭스"],
  "categories": ["카페", "OTT/영화/문화"],
  "expanded_queries": ["스타벅스 할인 카드", "넷플릭스 혜택", "커피빈 할인"]
}}

반드시 JSON 형식만 출력해. 설명은 생략해.
"""
        }
    ]

    # API 호출
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.1,
        "response_format": {"type": "json_object"} 
    }

    response_data = query_hf_api(payload)
    
    result = {
        "brands": [],
        "categories": list(set(found_categories)),
        "expanded_queries": [original_query]
    }

    if response_data and "choices" in response_data:
        content = response_data["choices"][0]["message"]["content"]
        
        try:
            clean_text = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_text)
            
            if "brands" in parsed and isinstance(parsed["brands"], list):
                result["brands"] = parsed["brands"]
            
            if "categories" in parsed and isinstance(parsed["categories"], list):
                valid_cats = [c for c in parsed["categories"] if c in KNOWN_CATEGORIES]
                result["categories"].extend(valid_cats)
                
            if "expanded_queries" in parsed and isinstance(parsed["expanded_queries"], list):
                # 원본 쿼리를 맨 앞에 추가
                parsed["expanded_queries"].insert(0, original_query)
                result["expanded_queries"] = parsed["expanded_queries"]
                
        except json.JSONDecodeError:
            safe_print(f"⚠️ JSON 파싱 실패. 원본 응답: {content}")
        except Exception as e:
            safe_print(f"⚠️ 분석 중 오류 발생: {e}")

    # 중복 제거
    result["brands"] = list(set(result["brands"]))
    result["categories"] = list(set(result["categories"]))
    result["expanded_queries"] = list(dict.fromkeys(result["expanded_queries"]))

    return result

# ============================================================
# 4. 검색 및 랭킹 헬퍼 함수 (가중치 & 병렬 처리 적용)
# ============================================================


def calculate_benefit_score(doc, user_brands: List[str], user_categories: List[str]) -> float:
    """
    혜택 점수 계산 (메타데이터가 0일 경우 텍스트에서 자동 추출)
    """
    try:
        # 1. 메타데이터에서 기본값 가져오기
        metadata_val = float(doc.metadata.get("benefit_value", 0))
        unit = doc.metadata.get("benefit_unit", "").strip()
        
        doc_cat = doc.metadata.get("category", "")
        summary = doc.metadata.get("benefit_summary", "")
        # 검색 대상 텍스트 (제목 + 요약 + 본문)
        text_all = (doc.metadata.get("card_name", "") + " " + 
                   summary + " " + 
                   (doc.page_content or ""))

        # 2. 점수 1차 산정 (메타데이터 기준)
        score = 0.0
        if unit == "%":
            score = metadata_val * 100 # 1% = 100점
        else:
            score = metadata_val # 원화는 그대로

        # 🚀 [핵심 개선] 메타데이터 점수가 0이거나 너무 낮으면 텍스트에서 직접 채굴
        if score < 10: 
            # (1) % 패턴 찾기 (예: 10%, 5.5%)
            # \d+(?:\.\d+)? : 정수 또는 소수
            pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', text_all)
            if pct_matches:
                # 추출된 % 값 중 최대값 * 100
                max_pct = max([float(x) for x in pct_matches])
                score = max(score, max_pct * 100)

            # (2) 원화 패턴 찾기 (예: 1,500원, 1만원, 20000 원)
            # 콤마(,) 제거 및 '만' 단위 처리 로직 필요
            # 간단하게 숫자+원 패턴만 우선 처리
            krw_matches = re.findall(r'(\d+(?:,\d+)*)\s*원', text_all)
            if krw_matches:
                # 콤마 제거 후 숫자로 변환
                amounts = [float(x.replace(',', '')) for x in krw_matches]
                if amounts:
                    score = max(score, max(amounts))
            
            # '만원' 단위 처리 (예: 1만원 -> 10000)
            man_matches = re.findall(r'(\d+(?:,\d+)*)\s*만원', text_all)
            if man_matches:
                amounts_man = [float(x.replace(',', '')) * 10000 for x in man_matches]
                if amounts_man:
                    score = max(score, max(amounts_man))

        # 기본 점수 보정 (여전히 0이면 1.0)
        if score <= 0: score = 1.0

        # 3. 가중치 적용 (브랜드 & 카테고리)
        if user_brands and any(b in text_all for b in user_brands): 
            score *= 2.0
        
        if user_categories:
            # 카테고리명 일치 시
            if any(c in doc_cat or doc_cat in c for c in user_categories): 
                score *= 1.5
            # 텍스트 내 카테고리 키워드 발견 시
            elif any(c in text_all for c in user_categories): 
                score *= 1.2

        return score
    except Exception as e:
        # print(f"점수 계산 에러: {e}")
        return 0.0

@traceable(name="1. Retrieve Candidates (Parallel)", run_type="retriever")
def retrieve_candidates(vector_store, queries: List[str], k_per_query: int):
    """
    [속도 개선] 병렬 처리(Parallel Processing)를 적용하여 검색 속도를 높임
    """
    all_docs = []
    # 중복/빈 쿼리 제거
    unique_queries = list(set([q for q in queries if q.strip()]))

    def _single_search(query):
        try:
            return vector_store.similarity_search_with_score(query, k=k_per_query)
        except Exception as e:
            # print(f"검색 실패: {e}")
            return []

    # 최대 5개의 쓰레드로 동시 검색
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_query = {executor.submit(_single_search, q): q for q in unique_queries}
        
        for future in as_completed(future_to_query):
            docs_with_scores = future.result()
            for doc, score in docs_with_scores:
                if score < 0.15: continue 
                all_docs.append(doc)
    
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    return list(unique_docs)

@traceable(name="2. Rerank Candidates", run_type="parser")
def rerank_candidates(docs: List, user_brands: List[str], user_categories: List[str]) -> List:
    docs_scored = []
    for doc in docs:
        score = calculate_benefit_score(doc, user_brands, user_categories)
        docs_scored.append((doc, score))
    
    sorted_docs = sorted(docs_scored, key=lambda x: x[1], reverse=True)
    return [d for d, s in sorted_docs]

@traceable(name="3. Select Top Results", run_type="parser")
def select_final_results(sorted_docs: List, top_k: int, user_brands: List[str], user_categories: List[str]) -> List[Dict]:
    top_results = []
    seen = set()

    for doc in sorted_docs:
        card_name = doc.metadata.get("card_name", "")
        if card_name in seen: continue
        seen.add(card_name)

        final_score = calculate_benefit_score(doc, user_brands, user_categories)
        top_results.append({
            "card_name": card_name,
            "origin_id": doc.metadata.get("origin_id", ""),
            "benefit_summary": doc.metadata.get("benefit_summary", ""),
            "score": final_score,
            "category": doc.metadata.get("category", ""),
            "match_reason": f"브랜드:{user_brands}, 카테고리:{user_categories}"
        })
        if len(top_results) >= top_k: break
    return top_results

# ============================================================
# 5. Main Tool Definition
# ============================================================

@tool
def retriever_tool_unit_none(query: str) -> List[Dict]:
    """Search for credit card benefits using optimized unified analysis."""
    try:
        if not HF_API_KEY:
            safe_print("❌ HF_API_KEY가 설정되지 않았습니다.")
            return []

        embeddings = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-m3",
            task="feature-extraction",
            huggingfacehub_api_token=HF_API_KEY,
        )

        vector_store = ElasticsearchStore(
            es_url=ELASTICSEARCH_URL,
            index_name=INDEX_NAME,
            embedding=embeddings
        )

        # 1. 통합 분석
        analysis_result = analyze_query_unified(query)
        
        user_brands = analysis_result["brands"]
        user_categories = analysis_result["categories"]
        expanded_queries = analysis_result["expanded_queries"]
        
        safe_print(f"\n👀 [분석 결과 ({MODEL_NAME})]")
        safe_print(f"   - 브랜드: {user_brands}")
        safe_print(f"   - 카테고리: {user_categories}")
        safe_print(f"   - 확장검색어: {expanded_queries}")

        # 2. 병렬 검색
        candidate_docs = retrieve_candidates(vector_store, expanded_queries, k_per_query=20)

        # 3. 리랭킹
        ranked_docs = rerank_candidates(candidate_docs, user_brands, user_categories)

        # 4. 결과 선택
        final_results = select_final_results(ranked_docs, top_k=3, user_brands=user_brands, user_categories=user_categories)

        return final_results

    except Exception as e:
        safe_print(f"❌ 검색 프로세스 오류: {e}")
        return []

# ============================================================
# 6. 실행 블록
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

    safe_print(f"🔍 검색 테스트 모드 (Model: {MODEL_NAME})")
    safe_print("💡 종료하려면 'q' 입력\n")
    
    while True:
        try:
            user_query = input("\n💬 검색어 입력: ").strip()
        except UnicodeDecodeError:
            continue
            
        if user_query.lower() in ["q", "quit"]:
            break
        if not user_query:
            continue

        # 🕒 시간 측정 시작
        start_time = time.perf_counter()

        results = retriever_tool_unit_none.invoke(user_query)

        # 🕒 시간 측정 종료
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # ⏱️ 소요 시간 출력 (소수점 4자리까지)
        safe_print(f"\n⏱️ [Total Time]: {elapsed_time:.4f} sec")

        if results:
            safe_print(f"🏆 [추천 결과 Top {len(results)}]")
            for i, res in enumerate(results):
                safe_print(f"{i+1}. {res['card_name']} (점수: {res['score']:.1f})")
                safe_print(f"   - 혜택: {res['benefit_summary']}")
                safe_print(f"   - 매칭: {res['match_reason']}")
        else:
            safe_print("⚠️ 결과 없음")