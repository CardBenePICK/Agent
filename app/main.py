import os
import json
from typing import Any, Dict, List, Optional, Union


from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
# 👇 새로 만든 라우터 import
from rag.routers import ml_recommendations


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage


from agent.agent_graph import create_agent_app
import time
from datetime import datetime
import jwt


load_dotenv()


# LangSmith 설정 (선택)
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key and langsmith_api_key.strip():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LLM Agent with LangGraph")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


import cProfile
import pstats
import io
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request



app = FastAPI(
    title="LLM Agent API",
    root_path="/agent"  # 👈 "나는 /agent 경로 뒤에 살고 있어"라고 알려줌
)
# app.add_middleware(CProfileMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중이므로 모든 출처 허용 (보안상 나중에 프론트 주소로 특정하는 것이 좋음)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, OPTIONS 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

templates = Jinja2Templates(directory="templates")

# JWT 설정
JWT_SECRET = os.getenv("SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# 사용자별 세션 저장소
SessionKey = Union[int, str]
DEFAULT_SESSION_KEY: SessionKey = "guest"
_MAX_HISTORY_LENGTH = 20
user_sessions: Dict[SessionKey, Dict[str, Any]] = {}

# 전역 에이전트
agent_app = None


# RAG/도구 사용 지침 프롬프트
SYSTEM_PROMPT = (
    "당신은 카드 정보를 활용한 개인화된 AI Agent 입니다..\n"
    "\n"
    "절대 원칙:\n"
    "- 근거 없는 추측이나 임의의 지식 생성(할루시네이션)을 절대 하지 않습니다.\n"
    "- MCP 서버의 도구 결과에 반드시 근거해야 합니다.\n"
    "- MCP 도구를 사용한다면 어떤 도구(함수 이름)를 사용했는지도 표현합니다.\n"
    "- 한국어로 대답합니다." 
    "- 도구 결과나 문서에서 확인할 수 없는 정보는 반드시 '모르겠습니다'고 명시합니다.\n"
    "\n"
    "행동 지침:\n"
    "1. MCP 서버에 등록된 MCP 도구를 이용해 답변을 해야 합니다.\n"
    "\n"
    "추가 원칙:\n"
    "- 자신의 내부 지식을 근거로 확장하거나 추측하지 않습니다.\n"
    "- 문서나 도구 결과가 모호할 경우, 모호하다고 명시합니다.\n"
    "- 답변을 할 때 답변 이유가 존재하면 보기 좋게 이유를 정리해서 답변해야 합니다\n"
    "- 적절한 개행 문자를 사용해서 답변을 보기 좋게 구분해야 합니다.\n"
    "- 반드시 사실 기반의 답변만 제공합니다.\n"
)



def _get_session(session_key: SessionKey) -> Dict[str, Any]:
    if session_key not in user_sessions:
        user_sessions[session_key] = {
            "history": [],
            "auth": None,
            "user_id": session_key if isinstance(session_key, int) else None
        }
    return user_sessions[session_key]


def _trim_history(history: List[BaseMessage]) -> None:
    if len(history) > _MAX_HISTORY_LENGTH:
        del history[:len(history) - _MAX_HISTORY_LENGTH]


def _build_system_prompt(user_id: Optional[int] = None) -> str:
    base_prompt = SYSTEM_PROMPT
    if user_id is not None:
        base_prompt += f"\n\n현재 대화 중인 사용자 ID: {user_id}"
    return base_prompt


def _extract_token(auth_header: Optional[str]) -> str:
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authorization header is missing")
    
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Authorization header must be a Bearer token")
    return token


def _decode_user_id(token: str) -> int:
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


def get_history(session_key: SessionKey = DEFAULT_SESSION_KEY) -> List[BaseMessage]:
    return _get_session(session_key)["history"]


def pick_last_ai_text(messages: List[BaseMessage]) -> Optional[str]:
    last_message = messages[-1]
    final_response = ""

    # 1. 만약 마지막 메시지가 '도구의 결과(ToolMessage)'라면 (return_direct=True 설정 시)
    if isinstance(last_message, ToolMessage):
        # 도구의 결과(JSON 문자열)를 그대로 사용
        final_response = last_message.content
        return final_response
    
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai":
            return getattr(m, "content", "")
    return None




# 앱 시작 시 1회 초기화를 위해 on_event -> startup을 명시해둔 것
@app.on_event("startup")
async def _startup():
    global agent_app
    agent_app = await create_agent_app() # agent_graph.py에 create_agent_app() 있음. llm 및 tools 설정 가능
    user_sessions.clear()


# FastApi와 연결해서 루트 경로 "/"로 들어오면 templates의 index.html을 열도록 한다.
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_history": get_history(),
            "success": True,
            "error": "",
        },
    )


# templates의 <form method="post" action="/chat" id="chatForm"> 에서 값 입력
@app.post("/chat", response_class=HTMLResponse) 
async def chat(request: Request, query: str = Form(...)):
    start_time = time.perf_counter()
    global agent_app
    if agent_app is None:
        agent_app = await create_agent_app()


    # 게스트 세션 사용
    session = _get_session(DEFAULT_SESSION_KEY)
    history = session["history"]
    system_prompt = _build_system_prompt()
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=query)]
    input_len = len(messages)


    try:
        # create_react_agent는 {"messages": [...]} 입력을 받습니다.
        print(f"agent_app invoke time {datetime.now()}" )
        result = await agent_app.ainvoke({"messages": messages})
        state_messages: List[BaseMessage] = result.get("messages", [])
        ai_text = pick_last_ai_text(state_messages) or ""

        # 히스토리에 새 메시지들 저장 (Human + 새로 추가된 AI/Tool 메시지들)
        history.append(HumanMessage(content=query))
        if len(state_messages) > input_len:
            history.extend(state_messages[input_len:])
        _trim_history(history)


        success = True
        error = ""
    except Exception as e:
        success = False
        error = str(e)


    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"app chat 함수 실행 시간: {elapsed_time:.4f}초")


    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_history": get_history(DEFAULT_SESSION_KEY),
            "success": success,
            "error": error,
        },
    )

@app.post("/chat_react") # response_class=HTMLResponse 제거
async def chat_react(request: Request, query: str = Form(...)):
    start_time = time.perf_counter()
    global agent_app
    if agent_app is None:
        agent_app = await create_agent_app()

    # JWT 토큰에서 user_id 추출
    auth_header = request.headers.get("Authorization")
    try:
        token = _extract_token(auth_header)
        user_id = _decode_user_id(token)
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "response": "", "error": exc.detail, "cards": []}
        )

    # 사용자별 세션 가져오기
    session = _get_session(user_id)
    session["auth"] = auth_header
    history = session["history"]
    
    system_prompt = _build_system_prompt(user_id)
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=query)]
    input_len = len(messages)

    # 변수 초기화
    ai_text = ""
    success = False
    error = ""

    try:
        # create_react_agent는 {"messages": [...]} 입력을 받습니다.
        print(f"agent_app invoke time {datetime.now()}" )
        result = await agent_app.ainvoke({"messages": messages})
        state_messages: List[BaseMessage] = result.get("messages", [])
        ai_text = pick_last_ai_text(state_messages) or ""

        # 히스토리에 새 메시지들 저장 (Human + 새로 추가된 AI/Tool 메시지들)
        history.append(HumanMessage(content=query))
        if len(state_messages) > input_len:
            history.extend(state_messages[input_len:])
        _trim_history(history)

        success = True
        
    except Exception as e:
        success = False
        error = str(e)
        print(f"Error during chat processing: {e}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"app chat 함수 실행 시간: {elapsed_time:.4f}초")

    print("chat_react : 답변",{
        "success": success,
        "response": ai_text,
        "error": error,
        "cards": [] # 필요하다면 나중에 카드 데이터를 여기에 추가
    })
    # React 프론트엔드를 위한 JSON 응답 반환
    return {
        "success": success,
        "response": ai_text,
        "error": error,
        "cards": [] # 필요하다면 나중에 카드 데이터를 여기에 추가
    }


@app.post("/clear-chat")
async def clear_chat():
    user_sessions[DEFAULT_SESSION_KEY] = {"history": [], "auth": None, "user_id": None}
    return JSONResponse({"ok": True})


@app.get("/chat-history")
async def get_chat_history():
    chat_history = []
    for msg in get_history(DEFAULT_SESSION_KEY):
        if isinstance(msg, HumanMessage):
            chat_history.append({"type": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            chat_history.append({"type": "ai", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            chat_history.append({"type": "tool", "content": msg.content})
    return {"chat_history": chat_history}


# 잘 되고 있다는 것을 표현하기 위해.
@app.get("/api")
async def root():
    return {"message": "LLM Agent API is running"}




app.include_router(ml_recommendations.router, prefix="/api/ml", tags=["ML Recommendation"])