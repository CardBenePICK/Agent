from fastapi import APIRouter, HTTPException
from typing import Any, List
import logging

# 데이터 스키마 및 정의 임포트
from app.schemas.user_preference import UserPreferenceCreate
from data.persona_definitions import PERSONA_CLUSTERS
from data.mappings import map_codes_to_korean

# (이전 단계에서 만든) 검색 엔진 도구 임포트
# 실제 파일 경로에 맞게 수정 필요 (예: app.services.card_retriever)
from app.services.retriever_tool import retriever_tool, CardRetrieverPipeline 

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/preferences", response_model=dict)
async def generate_recommendation(preference_data: UserPreferenceCreate) -> Any:
    """
    [통합 추천 실행 API]
    1. Cluster ID로 '기본 검색어'와 '평균 소비 금액'을 조회합니다.
    2. 사용자가 선택한 Preferred Categories를 한글로 변환하여 검색어에 추가(가중치용)합니다.
    3. Retriever Pipeline을 실행하여 맞춤형 카드를 추천합니다.
    """
    
    # 1. 클러스터 정보 조회
    cluster_id = preference_data.cluster_id
    if cluster_id not in PERSONA_CLUSTERS:
        raise HTTPException(status_code=400, detail="Invalid Cluster ID")
        
    persona = PERSONA_CLUSTERS[cluster_id]
    
    # 2. 키워드 & 금액 설정
    # 검색어 = (클러스터 기본 키워드) + (사용자 선택 카테고리 한글명)
    user_selected_keywords = map_codes_to_korean(preference_data.preferred_categories)
    
    # 중복 제거 및 리스트 병합
    search_keywords = list(set(persona["base_keywords"] + user_selected_keywords))
    
    # 금액은 클러스터 평균 금액 사용 (추후 사용자 입력값이 있다면 그걸 우선시하는 로직 추가 가능)
    target_spend = persona["avg_spend"]

    print(f"===== [Recommendation Engine Start] =====")
    print(f"🎯 Persona: {persona['name_kr']} ({persona['name_en']})")
    print(f"💰 Target Spend: {target_spend:,}원")
    print(f"🔑 Final Search Keywords: {search_keywords}")
    print(f"🎨 User Preferences (Boost): {user_selected_keywords}")
    print(f"=========================================")

    try:
        # 3. Retriever 파이프라인 실행
        # (주의: retriever_tool 내부 구조를 함수 인자 형태로 호출할 수 있게 약간 수정했다고 가정)
        # 예: retriever_pipeline.run(keywords, spend, preferred_categories_for_boost)
        
        # 여기서 'user_selected_keywords'는 점수 계산 시 가중치(1.5배)를 줄 대상이 됩니다.
        results = CardRetrieverPipeline.run(
            query_keywords=search_keywords,
            user_spend=target_spend,
            preferred_categories=user_selected_keywords 
        )

        # 4. 결과 반환 (프론트엔드 포맷에 맞춤)
        return {
            "status": "success",
            "meta": {
                "persona_applied": persona['name_kr'],
                "search_keywords": search_keywords
            },
            "result": results  # recommendation_summary, card_comparison_list 포함
        }

    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))