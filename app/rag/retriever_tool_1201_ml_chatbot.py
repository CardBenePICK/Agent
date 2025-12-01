import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import requests

# 데이터 모듈 (ML용)
from data.mappings import map_codes_to_korean
from data.persona import PERSONA_CLUSTERS

load_dotenv()

# 설정
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = "credit_cards_nested_v1"
HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct:novita"

logger = logging.getLogger("CardPipeline")
logger.setLevel(logging.INFO)

# ============================================================
# [Module 1] 공통 유틸리티 (계산기 & ES 검색기 & LLM)
# ============================================================

class BenefitCalculator:
    """[공통] 카드 혜택 금액 계산기"""
    @staticmethod
    def calculate(card_source: Dict, user_spend: int, preferred_keywords: List[str]) -> Dict:
        total_benefit_krw = 0
        matched_details = []

        for ben in card_source.get("benefits", []):
            summary = ben.get("summary", "")
            category = ben.get("category", "")
            tiers = ben.get("tiers", [])

            # 1. Tiers에서 실적 조건 충족하는 가장 높은 구간 찾기
            sorted_tiers = sorted(tiers, key=lambda x: x.get("previous_min_spend", 0))
            applied_tier = None
            for tier in sorted_tiers:
                if user_spend >= tier.get("previous_min_spend", 0):
                    applied_tier = tier
                else:
                    break 
            
            if not applied_tier: continue

            # 2. 혜택 금액 산정
            val = 0
            if applied_tier.get("unit") == "%":
                # 가정: 해당 영역 소비 비중 10%
                val = (user_spend * 0.1) * applied_tier.get("rate", 0)
            elif applied_tier.get("unit") == "KRW":
                val = applied_tier.get("rate", 0)

            # 3. 선호 키워드 가중치 (1.5배)
            is_preferred = any(pk in category or pk in summary for pk in preferred_keywords)
            if is_preferred:
                val *= 1.5
                matched_details.append(f"{category}(★)")
            else:
                matched_details.append(category)
            
            total_benefit_krw += val

        return {"score": total_benefit_krw, "matched_info": ", ".join(list(set(matched_details)))}

class ESMultiRetriever:
    """[공통] Elasticsearch 검색기"""
    def __init__(self):
        self.client = Elasticsearch(ELASTICSEARCH_URL)

    def search_by_coverage(self, keywords: List[str], size: int = 150) -> List[Dict]:
        should_clauses = []
        for kw in keywords:
            should_clauses.append({
                "nested": {
                    "path": "benefits",
                    "score_mode": "sum",
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"benefits.category": kw}},
                                {"match": {"benefits.summary": kw}}
                            ]
                        }
                    }
                }
            })
        
        body = {
            "size": size,
            "_source": True,
            "query": {"bool": {"should": should_clauses, "minimum_should_match": 1}}
        }
        
        try:
            resp = self.client.search(index=INDEX_NAME, body=body)
            return [{"_id": h["_id"], "_source": h["_source"]} for h in resp["hits"]["hits"]]
        except Exception as e:
            logger.error(f"ES Error: {e}")
            return []

class LLMClient:
    """[공통] LLM API 호출기"""
    @staticmethod
    def call_api(messages: List[Dict], temperature=0.1) -> str:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {
            "model": MODEL_NAME, "messages": messages, 
            "max_tokens": 1000, "temperature": temperature,
            "response_format": {"type": "json_object"}
        }
        resp = requests.post("https://router.huggingface.co/v1/chat/completions", json=payload, headers=headers)
        content = resp.json()['choices'][0]['message']['content']
        # JSON 문자열 정제
        return content.replace("```json", "").replace("```", "").strip()

# ============================================================
# [Module 2] 챗봇용 파이프라인 (Interactive)
# ============================================================

class ChatbotPipeline:
    def __init__(self):
        self.retriever = ESMultiRetriever()

    def _analyze_query(self, user_query: str) -> Dict:
        """LLM을 사용해 사용자 질문에서 의도(키워드, 금액) 추출"""
        prompt = f"""
        사용자 질문을 분석하여 JSON으로 반환하세요.
        질문: "{user_query}"
        
        출력 필드:
        - keywords: 혜택 관련 핵심 단어 리스트 (예: ["통신", "스타벅스"])
        - spend: 사용자가 언급한 월 사용 금액(숫자). 언급 없으면 0.
        
        예시:
        {{"keywords": ["편의점", "교통"], "spend": 300000}}
        """
        try:
            res_str = LLMClient.call_api([{"role": "user", "content": prompt}])
            return json.loads(res_str)
        except:
            return {"keywords": [user_query], "spend": 0}

    def run(self, user_query: str) -> Dict:
        """
        [챗봇 진입점]
        1. 질문 분석 -> 2. 검색 -> 3. 계산 -> 4. 결과 리턴
        """
        # 1. 분석
        intent = self._analyze_query(user_query)
        keywords = intent.get("keywords", [])
        user_spend = intent.get("spend") or 500000 # 기본값 50만원
        
        logger.info(f"[Chatbot] Query: {user_query} -> Intent: {intent}")

        # 2. 검색
        candidates = self.retriever.search_by_coverage(keywords, size=100)

        # 3. 계산 및 정렬
        scored = []
        for cand in candidates:
            src = cand["_source"]
            res = BenefitCalculator.calculate(src, user_spend, keywords)
            scored.append({
                "card_name": src["card_name"],
                "score": res["score"],
                "info": res["matched_info"],
                "summary": src["benefits"][0]["summary"]
            })
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_3 = scored[:3]

        # 4. 챗봇용 답변 생성 (LLM에게 자연어 요약 요청)
        final_prompt = f"""
        사용자 질문: "{user_query}"
        추천 결과: {json.dumps(top_3, ensure_ascii=False)}
        
        위 결과를 바탕으로 사용자에게 친절하게 Top 3 카드를 추천하는 답변을 작성해줘.
        JSON 형식이 아니라 대화체 텍스트로 작성해.
        """
        # 여기서는 JSON이 아닌 일반 텍스트 모드로 호출한다고 가정 (코드 생략)
        return {"top_3": top_3, "raw_intent": intent}


# ============================================================
# [Module 3] ML/Persona 파이프라인 (Batch/Survey)
# ============================================================

class MLPersonaPipeline:
    def __init__(self):
        self.retriever = ESMultiRetriever()

    def run(self, cluster_id: int, category_codes: List[str]) -> Dict:
        """
        [ML/설문 진입점]
        1. 페르소나 Lookup -> 2. 검색 -> 3. 계산 -> 4. JSON 포맷 응답
        """
        # 1. 설정 로드
        if cluster_id not in PERSONA_CLUSTERS:
            raise ValueError("Invalid Cluster ID")
        
        persona = PERSONA_CLUSTERS[cluster_id]
        user_spend = persona["avg_spend"]
        
        # 키워드 결합 (Cluster Keywords + User Categories)
        user_cats_kr = map_codes_to_korean(category_codes)
        search_keywords = list(set(persona["keywords"] + user_cats_kr))

        logger.info(f"[ML] Cluster: {cluster_id}, Spend: {user_spend}, Keys: {search_keywords}")

        # 2. 검색
        candidates = self.retriever.search_by_coverage(search_keywords, size=150)

        # 3. 계산 및 정렬
        scored = []
        for cand in candidates:
            src = cand["_source"]
            # user_cats_kr(사용자가 직접 찍은 카테고리)에만 가중치 부여
            res = BenefitCalculator.calculate(src, user_spend, user_cats_kr)
            
            scored.append({
                "card_id": src["card_id"],
                "card_name": src["card_name"],
                "final_score": res["score"],
                "matched_info": res["matched_info"],
                "summary_text": src["benefits"][0]["summary"]
            })
        
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top_3 = scored[:3]

        # 4. JSON 생성 요청 (구조화된 출력)
        prompt = f"""
        [프로필] {persona['name']}, 월 {user_spend}원
        [후보] {json.dumps(top_3, ensure_ascii=False)}
        
        위 정보를 바탕으로 recommendation_summary, card_comparison_list 형태의 JSON을 생성해.
        """
        final_json_str = LLMClient.call_api([{"role": "user", "content": prompt}])
        return json.loads(final_json_str)

# ============================================================
# 인스턴스 생성
# ============================================================
chatbot_pipeline = ChatbotPipeline()
ml_pipeline = MLPersonaPipeline()