import sys
import io
import os
import json
import requests
import time
from typing import List, Dict, Any
from collections import defaultdict
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# [필수] Docker/Linux 환경에서 한글 출력 에러 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# [LangSmith] 추적용 (없으면 패스)
try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(func): return func
        return decorator

load_dotenv()

# ============================================================
# ⚙️ 설정 (Configuration)
# ============================================================
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct:novita"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
INDEX_NAME = "credit_cards_nested_top100"

# 임베딩 객체 설정
embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBEDDING_MODEL_ID,
    task="feature-extraction",
    huggingfacehub_api_token=HF_API_KEY,
)

def safe_print(title, data):
    print(f"\n🔹 [{title}]")
    if isinstance(data, list):
        for i, item in enumerate(data[:3]): # 상위 3개만 로그
            print(f"   {i+1}. {item}")
        if len(data) > 3: print(f"   ... (총 {len(data)}개)")
    else:
        print(f"   {data}")

# ============================================================
# 1. LLM Client
# ============================================================
class LLMClient:
    """[공통] LLM API 호출기"""
    
    @staticmethod
    @traceable(run_type="llm", name="HF_Inference_API")
    def call_api(messages: List[Dict], temperature=0.1) -> Dict:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {
            "model": MODEL_NAME, "messages": messages, 
            "max_tokens": 1000, "temperature": temperature,
            "response_format": {"type": "json_object"}
        }
        try:
            resp = requests.post("https://router.huggingface.co/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            content = resp.json()['choices'][0]['message']['content']
            
            # --- [파싱 로직] ---
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            content = content.strip()
            
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx : end_idx + 1]
                return json.loads(json_str)
            else:
                return json.loads(content)

        except Exception as e:
            print(f"❌ [LLM Error] {e}")
            return {}

# ============================================================
# 2. Query Analysis Logic
# ============================================================
class QueryAnalyzer:
    def rewrite_and_extract(self, query: str) -> List[str]:
        """Step 1: 사용자 쿼리에서 핵심 브랜드/카테고리 추출"""
        system_prompt = (
            "Extract potential brand or category keywords from the query. "
            "Output JSON ONLY: {\"keywords\": [\"k1\", \"k2\"]}. No notes."
        )
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        resp = LLMClient.call_api(msg)
        extracted = resp.get("keywords", [])
        safe_print("Step 1: Extracted Keywords", extracted)
        return extracted

    def group_and_weight(self, query: str, keywords: List[str]) -> List[Dict]:
        """Step 2: 추출된 키워드를 바탕으로 그룹핑 (분리 강화 버전)"""
        system_prompt = (
            "You are a strict JSON generator. "
            "Your Goal: Split keywords into DISTINCT semantic categories. "
            "Do NOT merge different concepts (e.g., Cafe and Gas must be separate). "
            "Output ONLY the JSON object."
        )

        prompt = f"""
        User Query: "{query}"
        Extracted Keywords: {keywords}

        Rules:
        1. **Separate Strictly**: If keywords belong to different industries (e.g., 'Cafe', 'Gas', 'Convenience', 'Shopping'), create SEPARATE groups for each.
        2. Assign 'weight' (0.5 to 1.0).
        3. 'is_must': true if the user implies it's mandatory (꼭, 필수, must).
        4. 'search_query': Create a specific search query for that single category.

        Example Input: "카페랑 주유소 필수고 편의점"
        Example Output:
        {{
            "groups": [
                {{"name": "Cafe", "keywords": ["Cafe"], "weight": 1.0, "is_must": true, "search_query": "카페 스타벅스 할인"}},
                {{"name": "Gas", "keywords": ["Gas Station"], "weight": 1.0, "is_must": true, "search_query": "주유소 리터당 할인"}},
                {{"name": "Convenience", "keywords": ["Convenience Store"], "weight": 0.8, "is_must": false, "search_query": "편의점 할인"}}
            ]
        }}
        """
        
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        resp = LLMClient.call_api(msg)
        groups = resp.get("groups", [])
        safe_print("Step 2: Groups & Weights (with Must)", groups)
        return groups

# ============================================================
# 3. Hybrid Searcher (ES Boost & Nested Sum 적용)
# ============================================================
class HybridSearcher:
    def __init__(self):
        self.es = Elasticsearch(ELASTICSEARCH_URL)
        self.embedder = embeddings 
    @traceable(run_type="retriever", name="Step3_Group_Search")
    def search_group(self, group: Dict) -> List[Dict]:
        """Step 3: 각 그룹별 하이브리드 검색"""
        query_text = group["search_query"]
        group_keywords = group["keywords"]
        
        # 🌟 [변경 1] 가져올 필드에 'previous_month_performance', 'domestic_year_cost' 추가
        target_fields = ["card_name", "card_id", "benefits.summary", "benefits.category", "previous_month_performance", "domestic_year_cost"]

        # --- 1. Vector Search ---
        try:
            vector = self.embedder.embed_query(query_text)
        except Exception as e:
            print(f"⚠️ Embedding API Error: {e}")
            vector = []

        vec_hits = []
        if vector and len(vector) > 0:
            try:
                vec_res = self.es.search(index=INDEX_NAME, knn={
                    "field": "benefits.vector",
                    "query_vector": vector,
                    "k": 50,
                    "num_candidates": 100
                }, _source=target_fields) # 🌟 target_fields 사용
                vec_hits = vec_res["hits"]["hits"]
            except Exception as e:
                print(f"⚠️ Vector Search Error: {e}")

        # --- 2. Keyword Search ---
        should_clauses = []
        for kw in group_keywords:
            should_clauses.append({"match": {"benefits.summary": {"query": kw, "boost": 2.0}}})
            should_clauses.append({"match": {"benefits.category": {"query": kw, "boost": 1.0}}})

        key_hits = []
        try:
            key_res = self.es.search(index=INDEX_NAME, query={
                "nested": {
                    "path": "benefits",
                    "score_mode": "sum",
                    "query": {
                        "bool": {
                            "should": should_clauses,
                            "minimum_should_match": 1
                        }
                    },
                    "inner_hits": {
                        "_source": ["benefits.summary", "benefits.category", "benefits.value"],
                        "size": 3
                    }
                }
            }, size=50, _source=target_fields) # 🌟 target_fields 사용
            key_hits = key_res["hits"]["hits"]
        except Exception as e:
            print(f"⚠️ Keyword Search Error: {e}")

        # --- 3. RRF Merge ---
        rrf_results = self._apply_rrf(vec_hits, key_hits)
        
        # 메타데이터 주입
        for item in rrf_results:
            item["matched_group"] = group["name"]
            item["group_weight"] = group["weight"]
            item["is_must"] = group.get("is_must", False)
            item["search_keywords"] = group_keywords
        
        safe_print(f"Step 3: Search Results for '{group['name']}'", 
                   [f"{doc['card_name']} (RRF: {doc['rrf_score']:.4f})" for doc in rrf_results])
        
        return rrf_results

    def _apply_rrf(self, vec_hits, key_hits, k=60):
        scores = defaultdict(float)
        docs = {}
        
        # Vector Rank
        for rank, hit in enumerate(vec_hits):
            cid = hit["_id"]
            scores[cid] += 1 / (rank + k)
            docs[cid] = hit["_source"]
            docs[cid]["_id"] = cid
            docs[cid]["vec_score"] = hit["_score"]
            docs[cid]["key_score"] = 0.0
            docs[cid].setdefault("inner_hits_info", []) # 벡터 검색은 inner_hits가 기본적으로 없음

        # Keyword Rank
        for rank, hit in enumerate(key_hits):
            cid = hit["_id"]
            scores[cid] += 1 / (rank + k)
            if cid not in docs:
                docs[cid] = hit["_source"]
                docs[cid]["_id"] = cid
                docs[cid]["vec_score"] = 0.0
            docs[cid]["key_score"] = hit["_score"]
            
            # Inner Hits 저장 (가장 관련성 높은 혜택 추출용)
            if "inner_hits" in hit:
                ih = hit["inner_hits"]["benefits"]["hits"]["hits"]
                docs[cid]["inner_hits_info"] = [h["_source"] for h in ih]
        
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        final_docs = []
        for cid in sorted_ids:
            d = docs[cid]
            d["rrf_score"] = scores[cid]
            final_docs.append(d)
            
        return final_docs

# ============================================================
# 4. Reranker (모든 관련 혜택 수집 버전)
# ============================================================
class Reranker:
    # 기존 rerank_cards 함수를 아래 코드로 통째로 교체하세요.

    @traceable(run_type="chain", name="Step4_Reranking")
    def rerank_cards(self, candidates: List[Dict], groups: List[Dict]) -> List[Dict]:
        """
        Step 4: Must Boost + Cross-Check + Diversity Bonus
        + [변경] JSON 출력을 위한 필드(ID, 연회비, 전월실적) 보존
        """
        unique_cards = {}
        
        MUST_BOOST_FACTOR = 3.0       
        
        safe_print("Step 4", f"Reranking & Collecting ALL benefits...")

        # 1. 기본 점수 합산 및 초기화
        for cand in candidates:
            cid = cand.get("card_id", cand.get("card_name"))
            
            if cid not in unique_cards:
                unique_cards[cid] = {
                    "card_name": cand["card_name"],
                    "card_id": cand.get("card_id"), # 🌟 [변경] ID 저장
                    "previous_month_performance": cand.get("previous_month_performance", "정보없음"), # 🌟 [변경] 전월실적 저장
                    "domestic_year_cost": cand.get("domestic_year_cost", "정보없음"), # 🌟 [변경] 연회비 저장
                    "benefits": cand.get("benefits", []),
                    "inner_hits_info": cand.get("inner_hits_info", []),
                    "total_score": 0.0,
                    "matched_reasons": [],
                    "matched_summaries": set(),
                    "search_keywords": cand.get("search_keywords", []),
                    "matched_group_names": set()
                }
            
            g_weight = cand.get("group_weight", 1.0)
            is_must = cand.get("is_must", False)
            rrf_score = cand.get("rrf_score", 0.0)
            group_name = cand.get("matched_group", "General")
            
            if group_name in unique_cards[cid]["matched_group_names"]:
                continue

            base_score = rrf_score * 1000
            final_multiplier = g_weight * (MUST_BOOST_FACTOR if is_must else 1.0)
            score_contribution = base_score * final_multiplier
            
            unique_cards[cid]["total_score"] += score_contribution
            unique_cards[cid]["matched_group_names"].add(group_name)
            
            mark = "🔥" if is_must else ""
            unique_cards[cid]["matched_reasons"].append(f"{mark}{group_name}({score_contribution:.1f})")
            
            if cand.get("inner_hits_info"):
                for hit in cand["inner_hits_info"]:
                    unique_cards[cid]["matched_summaries"].add(hit.get("summary", ""))

        # 2. Cross-Check (놓친 그룹 찾기) & 모든 혜택 스캔
        for cid, card in unique_cards.items():
            for group in groups:
                g_name = group["name"]
                if g_name in card["matched_group_names"]: continue
                
                g_keywords = group["keywords"]
                g_weight = group["weight"]
                g_is_must = group.get("is_must", False)
                
                for ben in card["benefits"]:
                    text = (ben.get("summary", "") + " " + ben.get("category", "")).lower()
                    if any(k.lower() in text for k in g_keywords):
                        card["matched_group_names"].add(g_name)
                        bonus_score = 10.0 * g_weight * (MUST_BOOST_FACTOR if g_is_must else 1.0)
                        card["total_score"] += bonus_score
                        mark = "🔥" if g_is_must else ""
                        card["matched_reasons"].append(f"{mark}{g_name}(Found! {bonus_score:.1f})")
                        break 
            
            active_keywords = []
            for group in groups:
                if group["name"] in card["matched_group_names"]:
                    active_keywords.extend(group["keywords"])
            
            for ben in card["benefits"]:
                text = (ben.get("summary", "") + " " + ben.get("category", "")).lower()
                if any(k.lower() in text for k in active_keywords):
                    card["matched_summaries"].add(ben.get("summary", ""))

        # 3. Diversity Bonus & 정리
        final_list = []
        for cid, card in unique_cards.items():
            match_count = len(card["matched_group_names"])
            diversity_multiplier = 1.0

            if match_count == 2: diversity_multiplier = 1.3
            elif match_count == 3: diversity_multiplier = 2.0
            elif match_count == 4: diversity_multiplier = 4.0
            elif match_count >= 5: diversity_multiplier = 10.0
            
            if diversity_multiplier > 1.0:
                card["total_score"] *= diversity_multiplier
                card["matched_reasons"].append(f"🎁Variety(x{diversity_multiplier})")

            breakdown_text = " + ".join(card["matched_reasons"])
            
            all_benefits = sorted(list(card["matched_summaries"]))
            if not all_benefits and card["benefits"]:
                 all_benefits = [card["benefits"][0].get("summary", "")]

            # 🌟 [변경] 최종 반환 포맷을 요청하신 키값으로 맞춤
            final_list.append({
                "card_id": card["card_id"],
                "card_name": card["card_name"],
                "previous_month_performance": card["previous_month_performance"],
                "domestic_year_cost": card["domestic_year_cost"],
                "benefit_list": all_benefits,
                "match_reason": breakdown_text, # score_breakdown -> match_reason으로 매핑
                "score": card["total_score"]
            })

        final_list.sort(key=lambda x: x["score"], reverse=True)
        return final_list[:3]

# ============================================================
# 🚀 메인 실행부
# ============================================================
def run_pipeline(user_query: str):
    analyzer = QueryAnalyzer()
    searcher = HybridSearcher()
    reranker = Reranker()
    
    # 1. 키워드 추출
    keywords = analyzer.rewrite_and_extract(user_query)
    
    # 2. 그룹핑 및 가중치
    groups = analyzer.group_and_weight(user_query, keywords)
    
    # 3. 그룹별 검색 (ES Boost & Nested Sum)
    all_candidates = []
    for group in groups:
        group_results = searcher.search_group(group)
        all_candidates.extend(group_results)
        
    # 4. 리랭킹 (단위 계산 없이 점수 합산)
    return reranker.rerank_cards(all_candidates, groups)

# ============================================================
# 🚀 메인 실행부 (출력 부분 수정)
# ============================================================
if __name__ == "__main__":
    # ... (기존 LangSmith 설정 코드 동일) ...
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = "CardBenefit ES_Score_Based"
    
    safe_print("🔍", "Card Benefit Search (All Benefits Display)")
    print("💡 종료하려면 'q' 입력")
    
    while True:
        try:
            q = input("\n💬 입력: ").strip()
        except KeyboardInterrupt: break
        if q.lower() in ["q", "exit"]: break
        if not q: continue

        start = time.perf_counter()
        results = run_pipeline(q)
        elapsed = time.perf_counter() - start

        print(f"\n⏱️ Time: {elapsed:.4f}s")
        if results:
            print(f"🏆 [추천 결과 Top {len(results)}]")
            for i, res in enumerate(results):
                print(f"\n{i+1}. {res['card_name']} (Score: {res['score']:.1f})")
                print(f"   📊 {res['score_breakdown']}")
                print(f"   💡 관련 혜택 모음:")
                # 🌟 [변경] 리스트를 돌면서 전부 출력
                for ben in res['benefit_list']:
                    print(f"      - {ben}")
        else:
            print("⚠️ 결과 없음")