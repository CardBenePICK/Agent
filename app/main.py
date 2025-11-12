import os
from typing import List, Optional


from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.memory import ConversationBufferWindowMemory


from agent.agent_graph import create_agent_app
import time
from datetime import datetime


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



app = FastAPI(title="LLM Agent API")
# app.add_middleware(CProfileMiddleware)


templates = Jinja2Templates(directory="templates")


# 대화 메모리
chat_memory = ConversationBufferWindowMemory(
    k=10,
    return_messages=True,
    memory_key="chat_history",
)


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



def get_history() -> List[BaseMessage]:
    return chat_memory.load_memory_variables({}).get("chat_history", [])


def pick_last_ai_text(messages: List[BaseMessage]) -> Optional[str]:
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai":
            return getattr(m, "content", "")
    return None




# 앱 시작 시 1회 초기화를 위해 on_event -> startup을 명시해둔 것
@app.on_event("startup")
async def _startup():
    global agent_app
    agent_app = await create_agent_app() # agent_graph.py에 create_agent_app() 있음. llm 및 tools 설정 가능


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


    # messages: System + history + 현재 사용자 질문
    history = get_history()
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=query)]


    try:
        # create_react_agent는 {"messages": [...]} 입력을 받습니다.
        print(f"agent_app invoke time {datetime.now()}" )
        result = await agent_app.ainvoke({"messages": messages})
        state_messages: List[BaseMessage] = result.get("messages", [])
        ai_text = pick_last_ai_text(state_messages) or ""


        # 메모리에 저장
        chat_memory.chat_memory.add_user_message(query)
        chat_memory.chat_memory.add_ai_message(ai_text)


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
            "chat_history": get_history(),
            "success": success,
            "error": error,
        },
    )


@app.post("/clear-chat")
async def clear_chat():
    chat_memory.clear()
    return JSONResponse({"ok": True})


@app.get("/chat-history")
async def get_chat_history():
    chat_history = []
    for msg in chat_memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            chat_history.append({"type": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            chat_history.append({"type": "ai", "content": msg.content})
    return {"chat_history": chat_history}


# 잘 되고 있다는 것을 표현하기 위해.
@app.get("/api")
async def root():
    return {"message": "LLM Agent API is running"}