import os
import uuid
from typing import List, Optional
import time
from datetime import datetime

from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# LangChain 및 Redis 관련 모듈 import
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_redis import RedisChatMessageHistory  # Redis 연동을 위한 클래스

from agent.agent_graph import create_agent_app

load_dotenv()

# LangSmith 설정 (기존과 동일)
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key and langsmith_api_key.strip():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LLM Agent with LangGraph")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

app = FastAPI(title="LLM Agent API")
templates = Jinja2Templates(directory="templates")

# --- 전역 변수 변경 ---
# 1. 전역 chat_memory 객체를 제거합니다. 이 역할은 각 요청마다 생성되는 Redis 기반 메모리가 대체합니다.
# chat_memory = ConversationBufferWindowMemory(...) # 이 줄을 삭제합니다.
 
# 2. Redis 접속 정보 설정 (환경 변수 사용 권장)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 3. agent_app은 전역으로 유지. 에이전트 자체는 상태가 없으므로 재사용 가능
agent_app = None 

# 시스템 프롬프트 (기존과 동일)
SYSTEM_PROMPT = (
    "당신은 카드 정보를 활용한 개인화된 AI Agent 입니다..\n"
    # ... (중략) ...
    "- 반드시 사실 기반의 답변만 제공합니다.\n"
)

# --- 헬퍼 함수 수정 ---

def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    """
    주어진 세션 ID에 대한 Redis 기반 대화 메모리 객체를 생성하고 반환합니다.
    """
    redis_history = RedisChatMessageHistory(
        session_id, 
        REDIS_URL,
        ttl=3600  # 1시간 후 대화 기록 자동 삭제 (선택 사항)
    )
    
    # 기존에 사용하시던 ConversationBufferWindowMemory를 그대로 사용하되,
    # chat_memory 백엔드로 Redis를 지정합니다.
    return ConversationBufferWindowMemory(
        k=10,  # 기존과 동일하게 최근 10개 대화 유지
        chat_memory=redis_history,
        return_messages=True,
        memory_key="chat_history",
    )

def pick_last_ai_text(messages: List[BaseMessage]) -> Optional[str]:
    # 이 함수는 기존과 동일하게 사용 가능
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai":
            return getattr(m, "content", "")
    return None

# --- FastAPI 이벤트 및 라우트 수정 ---

@app.on_event("startup")
async def _startup():
    # 앱 시작 시 에이전트 그래프는 한 번만 생성 (기존과 동일, 효율적)
    global agent_app
    if agent_app is None:
        agent_app = await create_agent_app()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # 쿠키에서 세션 ID를 가져와 해당 세션의 기록을 표시
    session_id = request.cookies.get("session_id")
    chat_history = []
    if session_id:
        memory = get_session_memory(session_id)
        chat_history = memory.chat_memory.messages

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_history": chat_history,
            "success": True,
            "error": "",
        },
    )

@app.post("/chat", response_class=HTMLResponse) 
async def chat(request: Request, query: str = Form(...)):
    start_time = time.perf_counter()
    
    # 1. 세션 ID 가져오기 (없으면 새로 생성)
    session_id = request.cookies.get("session_id") or str(uuid.uuid4())

    # 2. 현재 세션에 맞는 메모리 객체 가져오기
    session_memory = get_session_memory(session_id)
    
    # 3. 현재 세션의 대화 기록 불러오기
    history = session_memory.load_memory_variables({})["chat_history"]
    
    # messages: System + '현재 세션'의 history + 현재 사용자 질문
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=query)]
    
    success = True
    error = ""
    try:
        # 전역 agent_app은 그대로 사용, 상태를 가지는 messages만 세션에 맞게 전달
        print(f"agent_app invoke time {datetime.now()} for session: {session_id}")
        result = await agent_app.ainvoke({"messages": messages})
        state_messages: List[BaseMessage] = result.get("messages", [])
        ai_text = pick_last_ai_text(state_messages) or ""

        # 4. '현재 세션'의 메모리에 대화 내용 저장 (Redis에 저장됨)
        # LangChain Memory의 공식적인 방법인 save_context 사용
        session_memory.save_context({"input": query}, {"output": ai_text})
        
    except Exception as e:
        success = False
        error = str(e)
        print(f"Error for session {session_id}: {e}") # 에러 로그에 세션 ID 추가

    end_time = time.perf_counter()
    print(f"app chat 함수 실행 시간: {end_time - start_time:.4f}초")

    # 응답 생성
    response = templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            # 5. 템플릿에 전달할 기록도 현재 세션의 것만 전달
            "chat_history": session_memory.chat_memory.messages,
            "success": success,
            "error": error,
        },
    )
    
    # 6. 클라이언트에게 세션 ID를 쿠키로 설정/갱신
    response.set_cookie(
        key="session_id", 
        value=session_id, 
        max_age=30 * 24 * 60 * 60,  # 30일 (초 단위)
        httponly=True
        )

    return response

@app.post("/clear-chat")
async def clear_chat(request: Request):
    # 특정 세션의 기록만 삭제
    session_id = request.cookies.get("session_id")
    if session_id:
        memory = get_session_memory(session_id)
        memory.clear()  # RedisChatMessageHistory의 clear()가 호출됨
        return JSONResponse({"ok": True, "message": f"Session {session_id} cleared."})
    return JSONResponse({"ok": False, "message": "No session to clear."})

@app.get("/chat-history")
async def get_chat_history(request: Request):
    # 특정 세션의 기록만 JSON으로 반환
    session_id = request.cookies.get("session_id")
    if not session_id:
        return {"chat_history": []}
        
    memory = get_session_memory(session_id)
    chat_history_list = []
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            chat_history_list.append({"type": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            chat_history_list.append({"type": "ai", "content": msg.content})
    return {"chat_history": chat_history_list}

@app.get("/api")
async def root():
    return {"message": "LLM Agent API is running"}
