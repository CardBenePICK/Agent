import json
import sys
import os
from langchain_core.tools import tool
import logging

# 🌟 [여기 수정] 'rag' 패키지부터 명시적으로 경로를 적어줍니다.
# main.py가 /app에서 실행되므로 'rag'는 최상위 패키지로 인식됩니다.
from rag.chatbot_pipeline import run_pipeline

# 🟢 [수정] 상위 폴더(rag)에 있는 모듈을 불러오기 위한 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool(return_direct=True)
def search_credit_cards(query: str) -> str:
    """
    이 도구는 여러 카테고리(예: 카페, 주유, 편의점 등)를 동시에 분석하고 종합 순위를 매길 수 있습니다.
    사용자가 여러 조건을 말하더라도, 쿼리를 분리하지 말고 반드시 전체 문장을 한 번에 입력해야 합니다.
    '카페, 주유, 편의점 혜택을 가진 카드를 추천해줘'와 같은 쿼리가 들어오면 혜택을 가진 카드와 같은 맥락은
    카드 추천을 위한 조건이라고 생각하고 카페, 주유, 편의점만을 넘겨야 합니다.
    Do NOT split the query into multiple calls. Pass the full user query into this tool ONCE.

    Search for credit card recommendations based on user's lifestyle or specific needs (e.g., gas, coffee, shopping).
    
    Args:
        query (str): User's natural language query describing their needs.
        
    Returns:
        str: A JSON string containing a list of recommended cards with details like:
             - card_id, card_name
             - previous_month_performance (전월실적)
             - domestic_year_cost (연회비)
             - match_reason (Why this card was recommended)
             - benefit_list (List of relevant benefits)
    """
    try:
        # 1. 파이프라인 실행 (List[Dict] 반환됨)
        results = run_pipeline(query)

        if not results:
            return json.dumps({
                "message": "검색 결과가 없습니다.",
                "results": []
            }, ensure_ascii=False)

        # 2. 결과 구성 (파이프라인이 이미 키를 잘 맞춰주므로 포장만 하면 됨)
        final_output = {
            "user_query": query,
            "recommended_cards_NEW": results  # 이미 card_id, year_cost 등이 다 들어있음
        }

        # 🔍 [핵심] JSON으로 변환하여 로그 찍기 (한글 깨짐 방지: ensure_ascii=False)
        log_json = json.dumps(final_output, ensure_ascii=False, indent=2)
        logger.info(f"🚀 [FINAL RESPONSE LOG]:\n{log_json}")

        # 3. JSON String으로 변환하여 반환
        return json.dumps(final_output, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "error": f"카드 검색 중 오류 발생: {str(e)}"
        }, ensure_ascii=False)