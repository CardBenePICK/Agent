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
INDEX_NAME = "card_benefit_index_v1"  # 인덱스 이름 변경 권장

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
    """Elasticsearch에 문서와 메타데이터 적재"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        print(f"🚀 Elasticsearch({ELASTICSEARCH_URL})에 적재 시작...")
        
        # from_documents를 사용하면 metadata도 자동으로 ES에 매핑되어 저장됩니다.
        vector_store = ElasticsearchStore.from_documents(
            documents=docs,
            embedding=embeddings,
            es_url=ELASTICSEARCH_URL,
            index_name=INDEX_NAME,
            # 이미 청킹이 되어 있으므로 여기서 또 자르지 않습니다.
        )
        
        print(f"🎉 적재 완료! 총 {len(docs)}개 문서가 '{INDEX_NAME}' 인덱스에 저장되었습니다.")
        return vector_store
        
    except Exception as e:
        print(f"❌ 적재 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    # 전처리 단계에서 만든 파일 경로를 지정하세요.
    JSON_FILE_PATH = "processed_card_chunks.json" 
    
    if os.path.exists(JSON_FILE_PATH):
        docs = load_processed_docs(JSON_FILE_PATH)
        ingest_documents(docs)
    else:
        print(f"⚠️ '{JSON_FILE_PATH}' 파일이 없습니다. 전처리 코드를 먼저 실행해서 파일을 만들어주세요.")