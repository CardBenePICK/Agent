from fastapi import APIRouter, HTTPException
from typing import Any, List
import logging

# 1. 스키마 Import
from app.schemas.user_preference import UserPreferenceCreate

# 2. Retriever Tool 파이프라인 Import (경로 주의)
# app/rag/retriever_tool_1201_ml_chatbot.py 파일에서 ml_pipeline 객체를 가져옵니다.
from app.rag.retriever_tool_1201_ml_chatbot import ml_pipeline

router = APIRouter()
logger = logging.getLogger("ML_API")

@router.post("/preferences", response_model=dict)
async def save_user_preferences(preference_data: UserPreferenceCreate) -> Any:
    """
    [통합 데이터 수신 및 추천 실행 API]
    1. 프론트엔드로부터 클러스터 ID와 선호 카테고리를 수신합니다.
    2. RAG 파이프라인(ml_pipeline)을 실행하여 추천 카드를 생성합니다.
    3. 결과를 반환합니다.
    """
    print(f"===== [Backend] 데이터 수신 =====")
    print(f"Cluster ID: {preference_data.cluster_id}")
    print(f"Categories: {preference_data.preferred_categories}")
    print(f"=================================")

    try:
        # 3. 파이프라인 실행 (핵심 로직 연결)
        # cluster_id와 카테고리 리스트를 넘겨주면 -> 추천 결과 JSON이 반환됩니다.
        recommendation_result = ml_pipeline.run(
            cluster_id=preference_data.cluster_id,
            category_codes=preference_data.preferred_categories
        )

        print("✅ 추천 생성 완료")

        # 4. 결과 반환
        return {
            "status": "success",
            "message": "Recommendation generated successfully",
            "received_data": preference_data,     # 요청 데이터 (확인용)
            "recommendation": recommendation_result # 생성된 추천 결과
        }

    except ValueError as e:
        logger.error(f"Input Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")