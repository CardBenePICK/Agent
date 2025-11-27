import os
import json
import time  # ✅ 필수 추가!
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from dotenv import load_dotenv

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
HF_API_KEY = os.getenv("HF_API_KEY")
INDEX_NAME = "card_benefit_bgem3_v1"

def load_processed_docs(json_path: str) -> List[Document]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = []
        for item in data:
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
    try:
        if not HF_API_KEY:
            raise ValueError("HF_API_KEY가 설정되지 않았습니다.")

        print(f"🚀 Hugging Face API 연결 중... (Model: BAAI/bge-m3)")
        
        embeddings = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-m3",
            task="feature-extraction",
            huggingfacehub_api_token=HF_API_KEY,
        )
        
        print(f"🚀 Elasticsearch({ELASTICSEARCH_URL})에 적재 시작...")
        
        vector_store = ElasticsearchStore(
            es_url=ELASTICSEARCH_URL,
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        
        # ✅ 배치 사이즈 축소 (200 -> 32)
        # API 타임아웃(504) 방지를 위해 아주 작게 쪼갭니다.
        batch_size = 32  
        total_docs = len(docs)
        
        for i in range(0, total_docs, batch_size):
            batch = docs[i : i + batch_size]
            print(f"📦 API 전송 중... ({i + 1}/{total_docs}) - {len(batch)}개")
            
            # 재시도 로직 (최대 5회로 증가)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    vector_store.add_documents(batch)
                    break # 성공하면 탈출
                except Exception as e:
                    print(f"⚠️ 전송 실패 (시도 {attempt+1}/{max_retries}): {e}")
                    # 대기 시간 점진적 증가 (5초, 10초, 15초...)
                    wait_time = (attempt + 1) * 5
                    print(f"⏳ {wait_time}초 대기 후 재시도합니다...")
                    time.sleep(wait_time)
            else:
                # break 없이 반복문이 끝난 경우 (모든 시도 실패)
                print(f"❌ 배치 {i} 적재 최종 실패. 건너뜁니다.")

        print(f"🎉 모든 작업 완료!")
        return vector_store
        
    except Exception as e:
        print(f"❌ 치명적 오류 발생: {e}")
        raise

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    JSON_FILE_PATH = os.path.join(current_dir, "processed_card_chunks.json")
    
    if os.path.exists(JSON_FILE_PATH):
        docs = load_processed_docs(JSON_FILE_PATH)
        ingest_documents(docs)
    else:
        print(f"⚠️ '{JSON_FILE_PATH}' 파일이 없습니다.")