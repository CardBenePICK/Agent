import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_documents(file_path: str):
    """텍스트 파일을 로드하는 함수"""
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        print(f"✅ Successfully loaded {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        raise

def split_documents(documents):
    """문서를 청크로 분할하는 함수"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(documents)
        print(f"✅ Successfully split documents into {len(docs)} chunks")
        return docs
    except Exception as e:
        print(f"❌ Error splitting documents: {e}")
        raise

def ingest_documents(docs, index_name: str):
    """문서를 Elasticsearch에 인덱싱하는 함수"""
    try:
        # API 키 확인
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        print("🔑 Creating OpenAI embeddings...")
        # OpenAI 임베딩 생성 - 환경 변수에서 자동으로 API 키를 읽음
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"  # 최신 임베딩 모델 사용
        )
        
        print("📊 Testing embeddings...")
        # 임베딩 테스트
        test_embed = embeddings.embed_query("test")
        print(f"✅ Embeddings working! Dimension: {len(test_embed)}")

        print("🔍 Connecting to Elasticsearch...")
        db = ElasticsearchStore.from_documents(
            docs,
            embeddings,
            es_url=ELASTICSEARCH_URL,
            index_name=index_name
        )
        
        print(f"✅ Documents indexed to Elasticsearch index '{index_name}' successfully.")
        return db
        
    except Exception as e:
        print(f"❌ Error ingesting documents: {e}")
        raise

def main():
    """메인 함수"""
    try:
        # 샘플 텍스트 생성
        sample_text = """
LangChain is an open-source framework designed to help developers build applications with Large Language Models (LLMs).
It provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.

LangGraph is a library for building stateful, multi-actor applications with LLMs, modeling steps as nodes and edges in a graph.
It extends the LangChain Expression Language with the ability to coordinate multiple chains across multiple steps of computation.

FastAPI is a modern, high-performance web framework for building APIs with Python.
It is based on standard Python type hints and provides automatic API documentation.

Elasticsearch is a distributed search and analytics engine, which can also serve as a powerful vector store for RAG.
It provides scalable full-text search and supports vector similarity search for semantic retrieval.

LangSmith is a platform for debugging, testing, and monitoring AI applications.
It helps developers trace, evaluate, and monitor their LangChain applications in production.
"""

        print("📝 Creating sample document...")
        with open("example_docs.txt", "w", encoding='utf-8') as f:
            f.write(sample_text)

        print("📚 Loading documents...")
        documents = load_documents("example_docs.txt")
        
        print("✂️  Splitting documents...")
        chunks = split_documents(documents)
        
        print("🚀 Ingesting documents to Elasticsearch...")
        ingest_documents(chunks, "llm_rag_index")
        
        print("🎉 All done! Documents have been successfully indexed.")
        
        print(ELASTICSEARCH_URL)
    except Exception as e:
        print(f"💥 Error in main process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()