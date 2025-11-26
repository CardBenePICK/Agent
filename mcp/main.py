import os
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi_mcp import FastApiMCP
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
import json
import jwt
import pandas as pd
from tool_extra.recommend_llm import invoke_question
import time
from datetime import datetime, timezone, timedelta
from db_tools.repo import get_mcc_code_by_merchant, get_benefits_by_user_assets_and_mcc,get_user_benefit_limit_in_benefit_sum

# JWT 설정 (검증용 비밀키가 없으면 서명 검증을 건너뜁니다)
load_dotenv()

JWT_SECRET = os.getenv("SECRET_KEY")
JWT_ALGORITHM = "HS256"
app = FastAPI(title="Card Benefit Recommendation MCP Server")

# OpenWeather API 설정
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
HF_API_KEY = os.getenv("HF_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LLM_MODEL = "openai/gpt-oss-120b" # gpt-4o-mini, openai/gpt-oss-120b

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
 
rate_limiter = InMemoryRateLimiter(requests_per_second=10)
# llm = ChatOpenAI(
#     model="gpt-4o-mini", 
#     openai_api_key=OPENAI_API_KEY,
#     rate_limiter=rate_limiter
# )
chat = ChatOpenAI( # get_sale에서 사용할 llm
    # model="Qwen/Qwen3-8B",
    model="openai/gpt-oss-120b",  # Hugging Face Router의 모델
    openai_api_key=HF_API_KEY,
    openai_api_base="https://router.huggingface.co/v1",  # base_url 대신 사용
    temperature=0
)

with open('prompt/prompt.json', 'r', encoding='utf-8') as f:
    prompt_data = json.load(f)

    print("prompt_json을 불러왔습니다." + prompt_data["get_sale"][:20])

def merge_context(items):
    """
    JSON 리스트의 각 항목을 merge_context 방식으로 텍스트로 변환
    """
    parts = []
    for item in items:
        for key, value in item.items():
            if isinstance(value, list):
                # 리스트일 경우 각 항목을 줄바꿈으로 합침
                if value:
                    parts.append('\n'.join(value))
            elif value:  # 문자열 등 일반 값
                parts.append(str(value))
    return '\n'.join(parts)

def format_benefits_to_markdown(benefits_df: pd.DataFrame) -> str:
    """
    혜택 DataFrame의 모든 정보를 그대로 유지하며 benefit별로만 구분선을 추가
    """
    if benefits_df.empty:
        return "사용 가능한 혜택이 없습니다."
    
    result = ""
    
    # 각 benefit별로 모든 컬럼 정보를 그대로 출력
    for idx, row in benefits_df.iterrows():
        result += f"BENEFIT #{idx + 1}\n"
        result += "=" * 50 + "\n"
        
        # 혜택 적용 내역 컬럼들을 한글로 설명
        benefit_usage_cols = {
            'day_amount': '일간 혜택 적용 금액',
            'day_count': '일간 혜택 적용 횟수', 
            'week_amount': '주간 혜택 적용 금액',
            'week_count': '주간 혜택 적용 횟수',
            'month_amount': '월간 혜택 적용 금액',
            'month_count': '월간 혜택 적용 횟수',
            'year_amount': '연간 혜택 적용 금액',
            'year_count': '연간 혜택 적용 횟수',
            'prev_month_total': '전월 카드 사용 실적'
        }
        
        # 혜택 적용 내역이 아닌 컬럼들 먼저 출력
        for col_name in benefits_df.columns:
            if col_name not in benefit_usage_cols:
                value = row[col_name]
                
                # json_rawdata와 json_notice 컬럼은 특별히 처리
                if col_name in ['json_rawdata', 'json_notice'] and value:
                    result += f"{col_name}:\n"
                    try:
                        # JSON 문자열을 파싱
                        json_data = json.loads(value) if isinstance(value, str) else value
                        
                        # merge_context 방식으로 처리
                        if isinstance(json_data, list):
                            merged_text = merge_context(json_data)
                            result += f"  {merged_text}\n"
                        # 딕셔너리인 경우 그대로 출력
                        elif isinstance(json_data, dict):
                            result += f"  {json.dumps(json_data, ensure_ascii=False, indent=4)}\n"
                        # 그 외의 경우 원본 그대로
                        else:
                            result += f"  {json_data}\n"
                    except (json.JSONDecodeError, TypeError):
                        # JSON 파싱 실패시 원본 그대로 출력
                        result += f"  {value}\n"
                else:
                    result += f"{col_name}: {value}\n"
        
        # 혜택 적용 내역 섹션
        result += "\n사용자가 기간별 적용받은 혜택 내역 및 카드 사용 실적:\n"
        result += "-" * 50 + "\n"
        for col_name, korean_name in benefit_usage_cols.items():
            if col_name in benefits_df.columns:
                value = row[col_name]
                # 금액 관련 컬럼은 천단위 구분자 추가
                if 'amount' in col_name or 'total' in col_name:
                    result += f"{korean_name}: {value:,}원\n"
                else:
                    result += f"{korean_name}: {value}\n"
        
        result += "\n" + "-" * 80 + "\n\n"
    
    return result

@app.get("/sale", operation_id ="get_sale_value")
def get_sale(user_id :int, merchant: str, mcc_code : int, amount: int = None) -> Dict[str, Any]:
    """
    가맹점 이름과 결제금액, 사용자 보유 카드 혜택을 이용하여 가장 결제 금액이 저렴한 카드와 결제 정보를 반환합니다.

    이 함수를 실행하기 전 필수 정보 수집 과정:
    1. user_id을 모르면 get_user_id() 도구를 먼저 사용하세요
    2. merchant의 MCC 코드가 필요하면 get_mcc_code() 도구를 사용하세요
    3. 모든 정보가 수집되면 이 함수를 호출하여 최종 카드를 추천받으세요
    """
    start_time = time.perf_counter()
    print(f"get_sale func start time {datetime.now(timezone(timedelta(hours=9)))}" )

    # 현재 시각을 결제 시각으로 사용 (한국시간 UTC+9)
    kst = timezone(timedelta(hours=9))
    current_time = datetime.now(kst)
    payment_time = current_time.strftime("%Y년 %m월 %d일 %H시 %M분 %S초")
    weekday = ["월", "화", "수", "목", "금", "토", "일"][current_time.weekday()]



    question = f"{payment_time}, {weekday}요일 {merchant}에서 {amount}원 사용 예정. \n 내 사용 내역과 혜택 적용 가능한 시간, 요일인지, 한도를 고려해서 가장 결제 금액이 저렴한 카드 계산해줘. \n 한번에 하나 카드만 사용 가능하니까 모든 혜택 정보를 합산하지 말고 가장 좋은 카드 하나만 추천해줘. \n"

    # DB 연결해서 데이터 가져오기 및 context 정리
    try:
        # MCC 코드 조회
        mcc_code = get_mcc_code_by_merchant(merchant)
        
        # 혜택 리스트 조회 (benefit_sum과 조인된 데이터)
        benefits_df = get_benefits_by_user_assets_and_mcc(user_id, mcc_code)
        
        print(f"🔍 조회된 혜택 데이터:\n{benefits_df}")
        # 마크다운 형식으로 변환
        benefits_markdown = format_benefits_to_markdown(benefits_df)
        print(f"혜택 마크다운:\n{benefits_markdown}")

        
        # context 구성 (마크다운 형식)
        context = f"""
# 🛒 카드 추천 요청 정보

**사용자 ID:** {user_id}
**가맹점:** {merchant}
**MCC 코드:** {mcc_code}
**결제 예정 금액:** {amount:,}원 (예상)
**결제 시각:** {payment_time} ({weekday}요일)

{benefits_markdown}

# 📈 분석 요청
Benefit별 json_rawdata 정보를 복합적으로 이해하여 혜택이 적용된 최종 결제 금액을 계산하고, 가장 혜택이 높은 카드를 추천해주세요.
오후 9시 ~ 오전 9시까지 Night 할인서비스 10% 할인이라고 써있는 경우 Night에만 적용 가능한 혜택입니다. 그 아래에 일반 혜택은 없는 것입니다.
혜택을 복합적으로 잘 이해하여 계산하세요.
        """
        
        print(f"📊 완전한 마크다운 context:")
        print("="*80)
        print(context)
        print("="*80)
        
    except Exception as e:
        print(f"❌ DB 데이터 조회 실패: {e}")
        context = f"""
# ⚠️ 데이터 조회 실패

**사용자 ID:** {user_id}
**가맹점:** {merchant}
**금액:** {amount:,}원

오류: {str(e)} 
        """
    
    # 카드 혜택 비교하고 카드 추천하기
    answer = invoke_question(llm=chat, prompt=prompt_data["get_sale"], context=context, question=question)
    
    
    # answer에는 딕셔너리 모양의 str type이 반환됨.
    data_dict = json.loads(answer)
    data_dict["user_id"] = user_id
    print("llm 대답", data_dict)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"get_sale 함수 실행 시간: {elapsed_time:.4f}초")

    try:
        return data_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

def _extract_token(auth_header: str | None) -> str:
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Authorization header must be a Bearer token")
    return token


def _decode_user_id_from_token(token: str) -> int:
    try:
        if JWT_SECRET:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        else:
            payload = jwt.decode(token, options={"verify_signature": False})
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="JWT token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid JWT token: {exc}") from exc

    user_id = payload.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is missing in JWT payload")

    try:
        return int(user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="user_id in JWT payload must be an integer") from exc


@app.get("/get_user_id", operation_id ="get_user_id")
def get_user_id(request: Request) -> Dict[str, int]:
    """JWT 토큰을 디코딩해 사용자 ID를 반환합니다."""
    print("🔍 get_user_id() 호출됨")
    token = _extract_token(request.headers.get("Authorization"))
    user_id = _decode_user_id_from_token(token)
    return {"user_id": user_id}

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    GET 요청에 대해 200 OK와 함께 상태를 반환합니다.
    """
    return {"status": "ok"}

@app.head("/health", status_code=status.HTTP_200_OK)
def head_health_check():
    """
    HEAD 요청에 대해 200 OK를 반환합니다. (Docker healthcheck용)
    FastAPI는 HEAD 요청에 대해 자동으로 body 없는 응답을 처리합니다.
    """
    return Response() # 빈 응답을 보내면 FastAPI가 알아서 처리해줍니다.


@app.get("/get_mcc_code", operation_id="get_mcc_code")
def get_mcc_code(merchant_name: str):
    """
    주어진 가맹점 이름으로 DB에서 MCC 코드를 조회합니다.
    
    카드 혜택 계산을 위해 필요한 가맹점 분류 코드를 반환합니다.
    예: "GS25" → 5411 (편의점), "스타벅스" → 5814 (카페)
    """
    print(f"🔍 get_mcc_code() 호출됨 - 가맹점: {merchant_name}")
    try:
        mcc_code = get_mcc_code_by_merchant(merchant_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query error: {str(e)}")

    if mcc_code is None:
        raise HTTPException(status_code=404, detail=f"MCC code not found for merchant: {merchant_name}")

    return {"merchant_name": merchant_name, "mcc_code": int(mcc_code)}

@app.get("/get_location", operation_id="get_location")
def get_location():
    """
    현재 사용자의 위치 정보를 반환합니다.
    """
    print("🔍 get_location() 호출됨")
    return {"location": "Busan, South Korea"}

# @app.get("/get_benefits_by_mcc", operation_id="get_benefits_by_mcc")
# def get_benefits_by_mcc(user_id : int, mcc : int):
#     """
#     주어진 user_id와 mcc 코드를 사용하여 관련된 혜택 정보를 DB에서 조회하고 반환합니다.
#     사용자가 가진 카드와 결제 상황에 매칭되는 모든 혜택을 조회합니다.
#     """
#     try:
#         benefits_df = get_benefits_by_user_assets_and_mcc(user_id, mcc)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"DB query error: {str(e)}")

#     if benefits_df.empty:
#         raise HTTPException(status_code=404, detail=f"No benefits found for user_id: {user_id} and mcc: {mcc}")

#     return benefits_df.to_dict(orient="records")

# @app.get("/get_user_benefit_limit", operation_id="get_user_benefit_limit")
# def get_user_benefit_limit(user_id: int):
#     """
#     해당 user가 이번 기간에 적용받은 모든 혜택의 금액을 조회해서 반환합니다.
#     """
#     try:
#         user_benefits_df = get_user_benefit_limit_in_benefit_sum(user_id)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"DB query error: {str(e)}")
#     if user_benefits_df.empty:
#         raise HTTPException(status_code=404, detail=f"No benefit limits found for user_id: {user_id}")
    
#     return user_benefits_df.to_dict(orient="records")


mcp = FastApiMCP(
    app,
    name="Weather & Stock API MCP"
  
)

# /mcp 경로에 MCP 서버를 마운트합니다.
mcp.mount_http(mount_path="/mcp") 
if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8001)
    get_sale("bangbang", "GS25", 128000)