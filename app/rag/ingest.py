import os
import json
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from dotenv import load_dotenv

load_dotenv()

# 환경 변수 설정
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "card_benefit2080_index_v1"  # 인덱스 이름 변경 권장

def load_processed_docs(json_path: str) -> List[Document]:
    """전처리된 JSON 파일을 읽어 LangChain Document 객체로 변환"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        documents = []
        for item in data:
            # page_content와 metadata가 확실히 분리되어 있어야 함
            doc = Document(
                page_content=item["page_content"],
                metadata=item["metadata"]
            )
            documents.append(doc)
            
        print(f"✅ JSON에서 {len(documents)}개의 문서를 로드했습니다.")
        return documents
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        raise

def ingest_documents(docs: List[Document]):
    """Elasticsearch에 문서와 메타데이터 적재 (배치 처리 적용)"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        print(f"🚀 Elasticsearch({ELASTICSEARCH_URL})에 적재 시작...")
        
        # 1. Vector Store 인스턴스 초기화 (데이터 없이 연결만 설정)
        vector_store = ElasticsearchStore(
            es_url=ELASTICSEARCH_URL,
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        
        # 2. 배치 처리 설정 (한 번에 1000개씩)
        batch_size = 200
        total_docs = len(docs)
        
        # 3. 반복문으로 나누어 적재
        for i in range(0, total_docs, batch_size):
            batch = docs[i : i + batch_size]
            print(f"📦 배치 적재 중... ({i + 1}/{total_docs}) - {len(batch)}개 문서")
            
            # add_documents 함수를 사용하여 데이터 추가
            vector_store.add_documents(batch)
            
        print(f"🎉 모든 적재 완료! 총 {total_docs}개 문서가 '{INDEX_NAME}' 인덱스에 저장되었습니다.")
        return vector_store
        
    except Exception as e:
        print(f"❌ 적재 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    # 1. 현재 이 파이썬 파일(ingest.py)이 있는 폴더 경로를 구함
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 그 폴더 안에 있는 processed_card_chunks.json 파일을 지정
    JSON_FILE_PATH = os.path.join(current_dir, "processed_card_chunks.json")
    
    print(f"📂 파일 찾는 경로: {JSON_FILE_PATH}")  # 경로 확인용 출력

    if os.path.exists(JSON_FILE_PATH):
        docs = load_processed_docs(JSON_FILE_PATH)
        ingest_documents(docs)
    else:
        print(f"⚠️ '{JSON_FILE_PATH}' 파일이 없습니다. 전처리 코드를 먼저 실행해서 파일을 만들어주세요.")