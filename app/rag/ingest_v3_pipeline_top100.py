import os
import json
import time
# import re  # <-- 정규식 모듈 불필요
import threading
from typing import List, Dict, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

# [주의] 이 모듈이 설치되어 있어야 합니다 (HuggingFace API 키 필요)
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from elasticsearch import Elasticsearch, helpers

load_dotenv()

# ============================================================
# 1. 설정 (Configuration)
# ============================================================
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
HF_API_KEY = os.getenv("HF_API_KEY")
INDEX_NAME = "credit_cards_nested_top100"  # 구조가 바꼈으니 인덱스 이름 변경 권장
INPUT_FILE = "processed_card_chunks_only_credit_1129.json"
CHECKPOINT_FILE = "ingest_nested_checkpoint_simple.json"
CARD_ID_TOP100_FILE = "card_ids_top100.json" # CARD_ID_TOP100 변수명 변경 및 파일 정의
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

BATCH_SIZE = 8
MAX_WORKERS = 4

# ============================================================
# 1-1. CARD_ID_TOP100 목록 로드 함수 (가정: 파일에 ID 리스트가 있다고 가정)
# ============================================================
def get_top_100_ids() -> List[str]:
    """
    실제 환경에서 CARD_ID_TOP100_FILE 또는 DB/API에서 TOP 100 ID를 로드하는 함수.
    
    요청하신 ID 목록이 card_ids.json에 있으므로, 이 파일을 읽는 것으로 대체합니다.
    (파일명은 CARD_ID_TOP100_FILE 변수로 설정)
    """
    if os.path.exists(CARD_ID_TOP100_FILE):
        with open(CARD_ID_TOP100_FILE, 'r', encoding='utf-8') as f:
            try:
                # 카드 ID는 문자열 또는 정수 형태일 수 있으므로 모두 문자열로 변환하여 반환
                ids = json.load(f)
                return [str(i) for i in ids]
            except Exception as e:
                print(f"⚠️ {CARD_ID_TOP100_FILE} 로드 중 에러: {e}")
                return []
    else:
        # [주의] 실제 사용 시에는 이 부분이 파일 또는 DB에서 ID를 로드해야 함
        print(f"⚠️ {CARD_ID_TOP100_FILE} 파일이 존재하지 않습니다. 인덱싱 대상을 제한하지 않습니다.")
        return []

# ============================================================
# 2. Elasticsearch 매핑 (Schema)
# ... (이 부분은 동일)
# ============================================================
def create_index_with_mapping(es: Elasticsearch):
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "korean_analyzer": {
                        "type": "nori",
                        "tokenizer": "nori_tokenizer"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                # --- Root Level ---
                "card_id": {"type": "keyword"},
                "card_name": {"type": "text", "analyzer": "korean_analyzer"},
                "card_company": {"type": "keyword"},
                "domestic_year_cost": {"type": "integer"},
                "abroad_year_cost": {"type": "integer"},
                "previous_month_performance": {"type": "integer"},
                
                # --- Nested Level ---
                "benefits": {
                    "type": "nested",
                    "properties": {
                        "category": {"type": "keyword"},
                        "summary": {"type": "text", "analyzer": "korean_analyzer"},
                        "description": {"type": "text", "analyzer": "korean_analyzer"},
                        
                        "vector": {
                            "type": "dense_vector",
                            "dims": EMBEDDING_DIM,
                            "index": True,
                            "similarity": "cosine"
                        },
                        
                        # [수정] 단순화된 Tiers 구조
                        "tiers": {
                            "type": "nested",
                            "properties": {
                                "previous_min_spend": {"type": "long"},  # 카드의 전월실적
                                "rate": {"type": "float"},      # benefit_value
                                "unit": {"type": "keyword"},      # benefit_unit (%)
                                "type": {"type": "keyword"}      # benefit_type (saving/discount 등) [추가됨]
                            }
                        }
                    }
                }
            }
        }
    }

    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=mapping)
        print(f"✅ 인덱스 생성 완료: {INDEX_NAME}")
    else:
        print(f"ℹ️ 인덱스가 이미 존재합니다: {INDEX_NAME}")

# ============================================================
# 3. 데이터 구조화 로직 (Logic Changed)
# ============================================================
def create_tier_from_metadata(meta: Dict, card_min_spend: int) -> List[Dict]:
    """
    [수정됨] 복잡한 텍스트 파싱 없이 메타데이터 값을 그대로 할당합니다.
    """
    
    # 메타데이터 추출
    val = meta.get("benefit_value", 0)
    unit = meta.get("benefit_unit", "UNKNOWN")
    b_type = meta.get("benefit_type", "UNKNOWN")
    
    # --- [안전장치 추가] ---
    try:
        rate = float(val)
    except (ValueError, TypeError):
        rate = 0.0
    # ----------------------

    # rate 계산 (% 단위 처리)
    if unit == "%" and rate > 0:
        rate = rate / 100.0

    return [{
        "previous_min_spend": card_min_spend,
        "rate": rate,
        "unit": unit,
        "type": b_type
    }]


# ============================================================
# 4. 데이터 변환 및 임베딩 처리 (Worker)
# ============================================================
def process_single_card(card_id: str, chunks: List[Dict], embedding_model) -> Dict:
    if not chunks: return None

    try:
        base_meta = chunks[0]["metadata"]
        
        # Root 필드 값
        domestic_cost = int(base_meta.get("domestic_year_cost", 0))
        abroad_cost = int(base_meta.get("abroad_year_cost", 0))
        prev_perf = int(base_meta.get("previous_month_performance", 0))
        
        card_doc = {
            "_id": card_id,
            "_index": INDEX_NAME,
            "_source": {
                "card_id": card_id,
                "card_name": base_meta.get("card_name", "Unknown"),
                "card_company": base_meta.get("card_company", ""),
                "domestic_year_cost": domestic_cost,
                "abroad_year_cost": abroad_cost,
                "previous_month_performance": prev_perf,
                "benefits": []
            }
        }

        for chunk in chunks:
            meta = chunk["metadata"]
            content = chunk["page_content"]
            
            # (A) [변경] 파싱 대신 메타데이터 매핑 함수 호출
            # 인자로 카드의 전월실적(prev_perf)을 넘겨서 previous_min_spend로 사용
            tiers = create_tier_from_metadata(meta, prev_perf)
            
            # (B) 임베딩 생성
            text_to_embed = f"{meta.get('benefit_summary', '')} {content}"
            vector = embedding_model.embed_query(text_to_embed)
            
            # (C) 혜택 객체 조립
            benefit_obj = {
                "category": meta.get("category", "기타"),
                "summary": meta.get("benefit_summary", ""),
                "description": content,
                "vector": vector,
                "tiers": tiers
            }
            
            card_doc["_source"]["benefits"].append(benefit_obj)
            
        return card_doc

    except Exception as e:
        print(f"⚠️ 카드 {card_id} 처리 중 에러: {e}")
        return None

# ============================================================
# 5. 메인 실행 (Main Execution)
# ============================================================
def main():
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY 환경변수가 필요합니다.")

    es = Elasticsearch(ELASTICSEARCH_URL)
    if not es.ping():
        raise ConnectionError("Elasticsearch에 연결할 수 없습니다.")
    
    create_index_with_mapping(es)

    print(f"🚀 임베딩 모델 로드 중: {EMBEDDING_MODEL_ID}")
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_ID,
        task="feature-extraction",
        huggingfacehub_api_token=HF_API_KEY,
    )

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} 파일이 없습니다.")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 전체 카드 데이터를 card_id별로 그룹화
    card_groups = defaultdict(list)
    for item in raw_data:
        cid = item["metadata"].get("card_id")
        if cid:
            card_groups[cid].append(item)
    
    # --- [추가/변경] 인덱싱 대상 ID 목록 로드 ---
    top_100_ids = get_top_100_ids()
    if top_100_ids:
        print(f"✅ TOP 100 ID 목록 로드 완료. ({len(top_100_ids)}개)")
        top_100_set = set(top_100_ids)
    else:
        # 파일 로드에 실패하면 모든 카드를 대상으로 함 (또는 에러 처리)
        top_100_set = set(card_groups.keys())
        print("⚠️ TOP 100 ID 목록 로드 실패. 모든 카드를 대상으로 인덱싱을 진행합니다.")

    # --- 체크포인트 로드 및 작업 대상 필터링 ---
    processed_ids = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            processed_ids = set(json.load(f))
            
    # [변경] top_100_set에 포함되어 있고, 아직 처리되지 않은 ID만 선택
    target_card_ids = [
        cid for cid in card_groups.keys() 
        if cid in top_100_set and cid not in processed_ids
    ]
    
    # 작업 대상 그룹 생성
    target_groups = [(cid, card_groups[cid]) for cid in target_card_ids]
    
    # 테스트용 10개 제한 (필요시 주석 처리)
    # target_groups = target_groups[:10]

    print(f"📊 전체 카드: {len(card_groups)}개 | TOP 100 대상: {len(top_100_set)}개 | 완료: {len(processed_ids)}개 | 최종 작업 예정: {len(target_groups)}개")

    if not target_groups:
        print("🎉 모든 작업이 완료되었거나 인덱싱할 대상 카드가 없습니다.")
        return

    batches = [target_groups[i:i + BATCH_SIZE] for i in range(0, len(target_groups), BATCH_SIZE)]
    total_indexed = 0
    
    print(f"🚀 병렬 처리 시작 (Workers: {MAX_WORKERS}, Batch Size: {BATCH_SIZE})")
    
    # ... (Bulk 인덱싱 로직은 동일)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for batch in tqdm(batches, desc="Total Progress"):
            futures = []
            for cid, chunks in batch:
                futures.append(executor.submit(process_single_card, cid, chunks, embeddings))
            
            bulk_docs = []
            completed_ids = []
            
            for future in as_completed(futures):
                result_doc = future.result()
                if result_doc:
                    bulk_docs.append(result_doc)
                    completed_ids.append(result_doc["_source"]["card_id"])
            
            if bulk_docs:
                try:
                    success, failed = helpers.bulk(es, bulk_docs, stats_only=True)
                    if failed:
                        print(f"\n⚠️ {failed}건 인덱싱 실패")
                    
                    processed_ids.update(completed_ids)
                    with open(CHECKPOINT_FILE, 'w') as f:
                        json.dump(list(processed_ids), f)
                        
                    total_indexed += len(completed_ids)

                except Exception as e:
                    print(f"\n❌ ES Bulk Error: {e}")

    print(f"\n🎉 작업 종료! 총 {total_indexed}개의 카드가 인덱싱되었습니다.")

if __name__ == "__main__":
    main()
    # [주의] 이 코드를 실행하기 전에 'card_ids_top100.json' 파일에 인덱싱할 ID 목록이 
    # JSON 리스트 형태로 존재해야 합니다. (예: ["13", "51", ...])
    pass