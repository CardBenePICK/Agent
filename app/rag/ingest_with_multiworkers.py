import os
import json
import time
import threading
from typing import List, Set
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

load_dotenv()

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
HF_API_KEY = os.getenv("HF_API_KEY")
INDEX_NAME = "card_benefit_bgem3_v2"
CHECKPOINT_FILE = "ingest_checkpoint.json"
checkpoint_lock = threading.Lock()

# --- 체크포인트 관리 함수들 ---
def load_completed_batches() -> Set[int]:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    try:
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get("completed_batches", []))
    except Exception as e:
        print(f"⚠️ 체크포인트 로드 실패 (새로 시작): {e}")
        return set()

def mark_batch_complete(batch_idx: int):
    with checkpoint_lock:
        completed = load_completed_batches()
        completed.add(batch_idx)
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump({"completed_batches": list(completed)}, f)

def load_processed_docs(json_path: str) -> List[Document]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        documents = []
        for item in data:
            doc = Document(page_content=item["page_content"], metadata=item["metadata"])
            documents.append(doc)
        print(f"✅ JSON에서 {len(documents)}개의 문서를 로드했습니다.")
        return documents
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        raise

def ingest_batch(vector_store, batch, batch_idx):
    max_retries = 5
    last_error = None

    for attempt in range(max_retries):
        try:
            vector_store.add_documents(batch)
            mark_batch_complete(batch_idx)
            return True, batch_idx, None
            
        except Exception as e:
            last_error = e
            wait_time = (attempt + 1) * 5
            time.sleep(wait_time)
    
    return False, batch_idx, last_error

# 🛠️ 수정됨: batch_size와 max_workers를 인자로 받도록 변경
def ingest_documents_parallel(docs: List[Document], batch_size: int = 8, max_workers: int = 2):
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY가 설정되지 않았습니다.")

    print(f"🚀 Hugging Face API 연결 중... (Model: BAAI/bge-m3)")
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        task="feature-extraction",
        huggingfacehub_api_token=HF_API_KEY,
    )
    
    vector_store = ElasticsearchStore(
        es_url=ELASTICSEARCH_URL,
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    
    # 인자로 받은 batch_size 사용
    all_batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
    total_batches_count = len(all_batches)
    
    completed_batches = load_completed_batches()
    
    batches_to_process = []
    for i, batch in enumerate(all_batches):
        if i not in completed_batches:
            batches_to_process.append((i, batch))
    
    skipped_count = total_batches_count - len(batches_to_process)
    
    print("-" * 50)
    print(f"📊 작업 요약")
    print(f"   - 설정: Batch Size={batch_size}, Max Workers={max_workers}")
    print(f"   - 총 배치 수: {total_batches_count}")
    print(f"   - 완료된 배치: {skipped_count} (건너뜀 ✅)")
    print(f"   - 남은 배치  : {len(batches_to_process)} (작업 예정 🚀)")
    print("-" * 50)

    if not batches_to_process:
        print("🎉 모든 작업이 이미 완료되어 있습니다!")
        return

    print(f"🚀 병렬 처리를 시작합니다... (Max Workers: {max_workers})")

    # 🛠️ 수정됨: 인자로 받은 max_workers 사용
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(ingest_batch, vector_store, batch, idx) for idx, batch in batches_to_process]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="⚡ 임베딩 적재 중", unit="batch"):
            success, idx, err_msg = future.result()
            
            if not success:
                print(f"\n❌ 배치 {idx} 최종 실패! 원인: {err_msg}")

    print(f"\n🎉 모든 작업 완료!")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    JSON_FILE_PATH = os.path.join(current_dir, "processed_card_chunks_only_credit_1126.json")
    
    if os.path.exists(JSON_FILE_PATH):
        all_docs = load_processed_docs(JSON_FILE_PATH)
        target_docs = all_docs 
        
        # 🛠️ 수정됨: 여기서 안전한 설정값(8, 2)을 전달합니다.
        ingest_documents_parallel(target_docs, batch_size=8, max_workers=4)
    else:
        print(f"⚠️ '{JSON_FILE_PATH}' 파일이 없습니다.")