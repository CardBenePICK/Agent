import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI API 키를 환경 변수로 설정
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

@tool
def retriever_tool(query: str) -> List:
    """
    Retrieves relevant documents from the Elasticsearch vector store based on a query.
    
    This tool should be used to find information within the pre-indexed knowledge base.
    Input should be a single string query.
    """
    # API 키를 직접 전달하지 않고 환경 변수에서 자동으로 읽도록 함
    embeddings = OpenAIEmbeddings()

    vector_store = ElasticsearchStore(
        es_url=ELASTICSEARCH_URL,
        index_name="llm_rag_index",
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    results = retriever.invoke(query)

    formatted_results = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
    return formatted_results

if __name__ == "__main__":
    search_query = "랭그래프가 뭐야?"
    retrieved_docs = retriever_tool(search_query)
    print("Retrieved documents:", retrieved_docs)