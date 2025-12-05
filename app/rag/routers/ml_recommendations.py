from fastapi import APIRouter, HTTPException
import logging
from typing import Any
import traceback
import json
# [수정 1] 스키마 Import (app.schemas -> schemas)
try:
    from rag.schemas.user_preference import UserPreferenceCreate
except ImportError:
    from app.schemas.user_preference import UserPreferenceCreate

# [수정 2] 파이프라인 Import (app.rag... -> rag...)
try:
    from rag.retriever_tool_1201_ml_chatbot import ml_pipeline
except ImportError:
    # 로컬/도커 환경 차이로 인한 Fallback 처리
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from rag.retriever_tool_1201_ml_chatbot import ml_pipeline

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
    print("\n" + "="*60)
    print(f"🚀 [API Endpoint] 요청 수신 (Time: {preference_data.timestamp})")
    print(f"   - Cluster ID: {preference_data.cluster_id}")
    print(f"   - Categories: {preference_data.preferred_categories}")
    print("="*60 + "\n")

    try:
        # 3. 파이프라인 실행 (핵심 로직 연결)
        # cluster_id와 카테고리 리스트를 넘겨주면 -> 추천 결과 JSON이 반환됩니다.
        recommendation_result = ml_pipeline.run(
            cluster_id=preference_data.cluster_id,
            category_codes=preference_data.preferred_categories
        )

        print("✅ [API] ML Pipeline 실행 완료!")
        print(f"   - 결과 요약: {json.dumps(recommendation_result.get('recommendation_summary', {}), ensure_ascii=False)[:100]}...")

        # 4. 결과 반환
        return {
            "status": "success",
            "message": "Recommendation generated successfully",
            "received_data": preference_data,     # 요청 데이터 (확인용)
            "recommendation": recommendation_result # 생성된 추천 결과
        }

    except ValueError as e:
        print(f"❌ [API Error] 값 오류: {e}")
        logger.error(f"Input Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # [체크 3] 진짜 에러 원인을 찾기 위해 traceback 출력
        error_msg = f"Internal Server Error: {str(e)}"
        print(f"❌ [CRITICAL ERROR] {error_msg}")
        traceback.print_exc() # 터미널에 상세 에러 위치 출력
        
        logger.error(f"Pipeline Error: {e}")
        
        # 여기서 NameError가 났던 것입니다. 상단 import 확인 필수!
        raise HTTPException(status_code=500, detail=error_msg)