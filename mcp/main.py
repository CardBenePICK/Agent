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
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    openai_api_key=OPENAI_API_KEY,
    rate_limiter=rate_limiter
)
chat = ChatOpenAI( # get_sale에서 사용할 llm
    model="openai/gpt-oss-120b",  # Hugging Face Router의 모델
    openai_api_key=HF_API_KEY,
    openai_api_base="https://router.huggingface.co/v1",  # base_url 대신 사용
    temperature=0
)

with open('prompt/prompt.json', 'r', encoding='utf-8') as f:
    prompt_data = json.load(f)

    print("prompt_json을 불러왔습니다." + prompt_data["get_sale"][:20])

@app.get("/sale", operation_id ="get_sale_value")
def get_sale(username :str, merchant: str, amount: int = None) -> Dict[str, Any]:
    """
    사용처와 사용금액을 이용해서 가장 혜택이 높은 카드를 추천합니다.

    만약 사용자의 이름을 모르겠으면 get_user_name tool을 사용합니다.
    """
    start_time = time.perf_counter()
    print(f"get_sale func start time {datetime.now()}" )

    question = merchant+"에서 " + str(amount) + " 사용 예정. 카드 추천 해줘."

    # DB 연결해서 데이터 가져오기
    # get_db_info()

    #context 정리하기

    
    
    
    # 카드 혜택 비교하고 카드 추천하기
    answer = invoke_question(llm=chat, prompt=prompt_data["get_sale"], context="", question=question)

    
    # answer에는 딕셔너리 모양의 str type이 반환됨.
    data_dict = json.loads(answer)
    data_dict["username"] = username
    print("llm 대답", data_dict) 

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"get_sale 함수 실행 시간: {elapsed_time:.4f}초")

    try:
        return data_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

@app.get("/user_name", operation_id ="get_user_name")
def get_user_name() -> str:
    """
        사용자의 이름을 알아냅니다.
    """

    return "오방일"

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

mcp = FastApiMCP(
    app,
    name="Weather & Stock API MCP"
  
)

# /mcp 경로에 MCP 서버를 마운트합니다.
mcp.mount_http(mount_path="/mcp") 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)