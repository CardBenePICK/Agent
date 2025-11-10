import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# LangSmith 설정
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_api_key and langsmith_api_key.strip():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LLM Agent with LangGraph")
    print("✅ LangSmith 추적이 활성화되었습니다.")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("⚠️  LangSmith API 키가 없습니다. 추적이 비활성화됩니다.")

# MCP 클라이언트 초기화
try:
    client = MultiServerMCPClient({
        "fisa-mcp": {
            # "url": "http://host.docker.internal:8001/mcp", # docker 안에서 돌고 있기 때문에 밖에 있는 8001 번의 mcp를 찾기 위해서
            "url": "http://mcp_server_final:8001/mcp", # docker 안에서 돌고 있기 때문에 다른 container의 8001 번의 mcp를 찾기 위해서
            # fastapi_llm_agent_final: 
            "transport": "streamable_http"
        }
    })
    print("✅ MCP 클라이언트가 초기화되었습니다.")
except Exception as e:
    print(f"⚠️ MCP 클라이언트 초기화 실패: {e}")
    client = None


    
async def create_agent_app():
    """LangGraph create_react_agent + retriever_tool + MCP 도구 구성 (messages 기반 호출과 호환)"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 1) tools 설정
    tools = []

    # 2) MCP 서버 도구를 로드하여 합치기 (이름 중복 제거)
    if client:
        try:
            loaded = await client.get_tools()
            loaded = loaded or []
            existing = {getattr(t, "name", None) for t in tools}
            for t in loaded:
                if getattr(t, "name", None) not in existing:
                    tools.append(t)
            if tools:
                print(f"🔧 사용 도구: {[t.name for t in tools]}")
            else:
                print("⚠️ 사용할 도구가 없습니다.")
        except Exception as e:
            print(f"❌ MCP 서버 도구 로드 실패: {e}")
    else:
        print("⚠️ MCP 클라이언트가 없습니다.")

    # 주의: 설치된 langgraph 버전에 따라 state_modifier 인자를 지원하지 않을 수 있음
    # 해당 경우, SYSTEM_PROMPT를 호출부(main.py)에서 SystemMessage로 prepend 하세요.
    agent = create_react_agent(llm, tools)
    return agent

# 전역 변수 (필요 시 외부에서 await create_agent_app() 호출 후 할당)
agent_app = None