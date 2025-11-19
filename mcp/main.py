import os
from fastapi import FastAPI, HTTPException, Response, status
from fastapi_mcp import FastApiMCP
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
import json
from tool_extra.recommend_llm import invoke_question
import time
from datetime import datetime
from db_tools.repo import get_mcc_code_by_merchant, get_benefits_by_user_assets_and_mcc,get_user_benefit_limit_in_benefit_sum

load_dotenv()
app = FastAPI(title="Weather & Stock MCP Server")

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

@app.get("/sale", operation_id ="get_sale_value")
def get_sale(user_id :int, merchant: str, mcc_code : int, amount: int = None) -> Dict[str, Any]:
    """
    가맹점 이름과 결제금액, 사용자 보유 카드 혜택을 이용하여 가장 결제 금액이 저렴한 카드와 결제 정보를 반환합니다.
    이 함수를 실행하는 입력 예시는 'user_id = 1, merchant = '버거킹', 'amount = 5000'입니다.

    이 함수를 실행하기 전 필수 정보 수집 과정:
    1. user_id을 모르면 get_user_id() 도구를 먼저 사용하세요
    2. merchant의 MCC 코드가 필요하면 get_mcc_code() 도구를 사용하세요 
    3. 해당 MCC에 적용 가능한 혜택 리스트(benefit_list_by_mcc)가 필요하면 get_benefits_by_mcc() 도구를 사용하세요
    4. 사용자의 혜택 적용 한도 정보(user_benefit_limit_list)가 필요하면 get_user_benefit_limit() 도구를 사용하세요
    5. 모든 정보가 수집되면 이 함수를 호출하여 최종 카드를 추천받으세요
    """
    start_time = time.perf_counter()
    print(f"get_sale func start time {datetime.now()}" )

    print("디버깅을 한번 해봅시다~")
    print("user_id:", user_id)
    print("merchant:", merchant)
    print("mcc_code:", mcc_code)
    print("amount:", amount)

    question = merchant+ "에서 " + str(amount) + "원 사용 예정. 내 카드 중 가장 유리한 카드 추천해줘."

    # DB 연결해서 데이터 가져오기 및 context 정리
    try:
        # MCC 코드 조회
        mcc_code = get_mcc_code_by_merchant(merchant)
        
        # 혜택 리스트 조회
        benefits_df = get_benefits_by_user_assets_and_mcc(user_id, mcc_code)
        
        # 사용자 혜택 한도 조회
        user_benefits_df = get_user_benefit_limit_in_benefit_sum(user_id)
        
        # context 구성 (실제 DB 데이터 기반)
        context = f"""
사용자 ID: {user_id}
가맹점: {merchant}
MCC 코드: {mcc_code}

=== 사용 가능한 카드 혜택 ===
{benefits_df.to_string() if not benefits_df.empty else "혜택 정보 없음"}

=== 사용자 혜택 한도 현황 ===
{user_benefits_df.to_string() if not user_benefits_df.empty else "한도 정보 없음"}
        """
        
        print(f"📊 구성된 context:")
        print(context)
        
    except Exception as e:
        print(f"❌ DB 데이터 조회 실패: {e}")
        context = f"사용자 ID: {user_id}, 가맹점: {merchant}, 금액: {amount}원"
    
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

@app.get("/get_user_id", operation_id ="get_user_id")
def get_user_id() -> int:
    """
        사용자의 user_id를 알아냅니다.
    """

    # 나중에 DB에서 사용자 이름으로 user_id를 조회하는 로직으로 변경 필요
    return 1

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

@app.get("/get_benefits_by_mcc", operation_id="get_benefits_by_mcc")
def get_benefits_by_mcc(user_id : int, mcc : int):
    """
    주어진 user_id와 mcc 코드를 사용하여 관련된 혜택 정보를 DB에서 조회하고 반환합니다.
    사용자가 가진 카드와 결제 상황에 매칭되는 모든 혜택을 조회합니다.
    """
    try:
        benefits_df = get_benefits_by_user_assets_and_mcc(user_id, mcc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query error: {str(e)}")

    if benefits_df.empty:
        raise HTTPException(status_code=404, detail=f"No benefits found for user_id: {user_id} and mcc: {mcc}")

    return benefits_df.to_dict(orient="records")

@app.get("/get_user_benefit_limit", operation_id="get_user_benefit_limit")
def get_user_benefit_limit(user_id: int):
    """
    해당 user가 이번 기간에 적용받은 모든 혜택의 금액을 조회해서 반환합니다.
    """
    try:
        user_benefits_df = get_user_benefit_limit_in_benefit_sum(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query error: {str(e)}")
    if user_benefits_df.empty:
        raise HTTPException(status_code=404, detail=f"No benefit limits found for user_id: {user_id}")
    
    return user_benefits_df.to_dict(orient="records")


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