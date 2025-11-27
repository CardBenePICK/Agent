import os
import sys
import requests
import json
from typing import List, Dict

# LangChain & LangSmith Imports
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.tools import tool
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
HF_API_KEY = os.getenv("HF_API_KEY")
INDEX_NAME = "card_benefit_bgem3_v1"

# ============================================================
# HF Router Chat Completion API Wrapper
# ============================================================

ROUTER_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")

def hf_chat_completion(messages, model, max_tokens=120, temperature=0.2):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    resp = requests.post(ROUTER_API_URL, headers=headers, json=payload)

    try:
        data = resp.json()
    except Exception as err:
        print("❌ JSON 파싱 오류:", err)
        print(resp.text)
        return None

    if "choices" not in data:
        print("⚠️ LLM 응답 오류:", data)
        return None

    return data["choices"][0]["message"]["content"]


# ============================================================
# Utility Safe Print
# ============================================================

def safe_print(text):
    try:
        print(text.encode("utf-8", "ignore").decode("utf-8"))
    except:
        print(text)


# ============================================================
# 브랜드 사전 & 사용자 입력 브랜드 추출
# ============================================================

# 전체 브랜드 사전 (여기서는 주요 예시만 넣었지만, 나중에 더 추가하면 됨)
ALL_BRANDS = [
    "스타벅스", "스벅",
    "투썸", "투썸플레이스",
    "이디야",
    "커피빈", "할리스", "폴바셋",
    "빽다방", "메가커피", "엔제리너스",
]

def extract_user_brands(original_query: str) -> List[str]:
    """사용자 입력 문장에서 등장한 브랜드만 추출"""
    found = []
    for b in ALL_BRANDS:
        if b in original_query:
            found.append(b)
    # 중복 제거 순서 유지
    return list(dict.fromkeys(found))


# ============================================================
# Query Expansion using HF ChatCompletion API
# ============================================================

@traceable(name="0. Expand Queries (HF ChatCompletion API)", run_type="llm")
def generate_expanded_queries(original_query: str) -> List[str]:
    """사용자 입력을 기반으로 검색용 확장 키워드 생성"""

    if not HF_TOKEN:
        safe_print("❌ HF Token 없음 — 원본 사용")
        return [original_query]

    MODEL_NAME = "google/gemma-2-9b-it:nebius"
    # MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct:novita"

    messages = [
        {
            "role": "system",
            "content": "너는 한국어 검색 질의 확장 전문가야."
        },
        {
            "role": "user",
            "content": f"""
다음 문장을 기반으로 검색용 확장 키워드 3개를 만들어줘.

규칙:
- 카페/커피 브랜드 줄임말 확장 (스벅→스타벅스 등)
- 오탈자 보정
- 명사구 형태로
- 출력은 "키워드1, 키워드2, 키워드3"

문장: {original_query}
"""
        }
    ]

    response_text = hf_chat_completion(messages, model=MODEL_NAME)
    if not response_text:
        return [original_query]

    keywords = [k.strip() for k in response_text.split(",") if k.strip()]

    # 카페 관련 일반 키워드는 "검색 보조용"으로만 추가
    keywords += ["카페 할인", "커피 할인", "브랜드 카페 혜택"]

    safe_print(f"🔀 확장된 키워드: {keywords}")
    return keywords


# ============================================================
# Retrieval Helpers (유사도 스코어 필터링 + 가중치)
# ============================================================

def calculate_benefit_score(doc, user_brands: List[str]) -> float:
    """혜택 값 + 사용자 브랜드 + 카페 관련 여부를 종합해서 점수 계산"""
    try:
        base = float(doc.metadata.get("benefit_value", 0))
        unit = doc.metadata.get("benefit_unit", "NONE")
        category = doc.metadata.get("category", "")
        summary = doc.metadata.get("benefit_summary", "")

        # 기본 점수: %면 그대로, 숫자면 그대로 (나중에 필요하면 더 튜닝)
        if unit == "%":
            score = base
        else:
            score = base

        text_all = (doc.page_content or "") + " " + summary

        # ① 사용자가 입력한 브랜드가 문서에 등장하면 강한 가중치
        if user_brands and any(b in text_all for b in user_brands):
            score *= 3

        # ② 일반 카페/커피 관련 키워드가 문서에 있으면 보조 가중치
        if any(x in text_all for x in ["커피", "카페"]):
            score *= 2

        # ③ 금융/환전/ATM 위주의 혜택은 이 시나리오에서는 중요도 낮게
        if "금융" in category or "환전" in summary or "ATM" in summary:
            score *= 0.2

        return score

    except:
        return 0.0


@traceable(name="1. Retrieve Candidates (ES)", run_type="retriever")
def retrieve_candidates(vector_store, queries: List[str], k_per_query: int):

    all_docs = []

    for q in queries:
        if not q.strip():
            continue

        # 유사도 점수 포함 검색
        docs_with_scores = vector_store.similarity_search_with_score(q, k=k_per_query)

        for doc, score in docs_with_scores:
            # 💡 유사도 Threshold 적용 (너 상황에 맞게 튜닝 가능)
            if score < 0.15:
                continue
            all_docs.append(doc)

    # page_content 기준 dedup
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    return list(unique_docs)


# ============================================================
# Rerank Candidates
# ============================================================

@traceable(name="2. Rerank Candidates", run_type="parser")
def rerank_candidates(docs: List, user_brands: List[str]) -> List:
    """사용자 브랜드 가중치를 적용해 후보군 정렬"""
    docs_scored = []
    for doc in docs:
        score = calculate_benefit_score(doc, user_brands)
        docs_scored.append((doc, score))

    # 점수 기준 내림차순
    sorted_docs = sorted(docs_scored, key=lambda x: x[1], reverse=True)
    return [d for d, s in sorted_docs]


# ============================================================
# Select Final Results
# ============================================================

@traceable(name="3. Select Top Results", run_type="parser")
def select_final_results(sorted_docs: List, top_k: int, user_brands: List[str]) -> List[Dict]:

    top_results = []
    seen = set()

    for doc in sorted_docs:
        card_name = doc.metadata.get("card_name", "")
        if card_name in seen:
            continue

        seen.add(card_name)

        final_score = calculate_benefit_score(doc, user_brands)

        top_results.append({
            "card_name": card_name,
            "origin_id": doc.metadata.get("origin_id", ""),
            "benefit_summary": doc.metadata.get("benefit_summary", ""),
            "benefit_value": doc.metadata.get("benefit_value", ""),
            "benefit_unit": doc.metadata.get("benefit_unit", ""),
            "score": final_score,
            "category": doc.metadata.get("category", ""),
            "detail": doc.page_content
        })

        if len(top_results) >= top_k:
            break

    return top_results


# ============================================================
# Main Tool
# ============================================================

@tool
def retriever_tool(query: str) -> List[Dict]:
    """Search for credit card benefits related to the query using Hugging Face API."""
    try:
        if not HF_API_KEY:
            safe_print("❌ HF_API_KEY가 없습니다.")
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

        # 0) 사용자 입력 브랜드 추출 (여기에만 가중치 적용)
        user_brands = extract_user_brands(query)
        safe_print(f"👀 사용자 입력 브랜드: {user_brands}")

        # 1) 쿼리 확장
        expanded_queries = generate_expanded_queries(query)

        # 2) 검색
        candidate_docs = retrieve_candidates(vector_store, expanded_queries, k_per_query=20)

        # 3) 리랭킹 (사용자 브랜드 가중치 반영)
        ranked_docs = rerank_candidates(candidate_docs, user_brands)

        # 4) 최종 Top-k 선택
        final_results = select_final_results(ranked_docs, top_k=3, user_brands=user_brands)

        return final_results

    except Exception as e:
        safe_print(f"❌ 검색 중 오류 발생: {e}")
        return []


# ============================================================
# Test Block (LangSmith 포함)
# ============================================================

if __name__ == "__main__":
    # LangSmith Tracing 설정
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = "CardBenefit RAG Debug"
        safe_print(f"✅ LangSmith Tracing Enabled (Project: {os.environ['LANGCHAIN_PROJECT']})")
    else:
        safe_print("⚠️ LangSmith API Key not found. Tracing disabled.")

    safe_print("🔍 API 검색 테스트 모드 (Query Expansion: HF Router ChatCompletion)")
    safe_print("💡 종료하려면 'q'를 입력하세요.\n")

    while True:
        try:
            try:
                user_query = input("💬 검색어 입력: ").strip()
            except UnicodeDecodeError:
                continue

            if user_query.lower() in ["q", "quit", "exit"]:
                safe_print("👋 종료합니다.")
                break

            if not user_query:
                continue

            safe_print(f"\n🚀 '{user_query}' 처리 중...\n")

            results = retriever_tool.invoke(user_query)

            if results:
                safe_print(f"\n🏆 [최종 추천 Top {len(results)}]")
                safe_print("=" * 50)
                for i, res in enumerate(results):
                    score_info = f"(점수: {res.get('score', 0):.1f})"
                    card_info = f"✅ {i+1}위 [ID:{res.get('origin_id')}]: {res.get('card_name')}"
                    benefit_info = f"   혜택: {res.get('benefit_summary')}"
                    detail_info = f"   상세: {score_info}"

                    safe_print(card_info)
                    safe_print(benefit_info)
                    safe_print(detail_info)
                    safe_print("-" * 50)
            else:
                safe_print("⚠️ 결과 없음\n")

        except KeyboardInterrupt:
            safe_print("\n👋 강제 종료")
            break
        except Exception as e:
            safe_print(f"❌ 오류: {e}\n")
