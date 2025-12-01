import os
import json
import time
import re
import threading
from typing import List, Dict, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

# LangChain & Elasticsearch imports
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from elasticsearch import Elasticsearch, helpers

# 환경 변수 로드
load_dotenv()

# ============================================================
# 1. 설정 (Configuration)
# ============================================================
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
HF_API_KEY = os.getenv("HF_API_KEY")
INDEX_NAME = "credit_cards_nested_v1"  # 인덱스 버전 업
INPUT_FILE = "processed_card_chunks_only_credit_1129.json"
CHECKPOINT_FILE = "ingest_nested_checkpoint_v2.json"

# 임베딩 설정
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# 배치 및 병렬 처리 설정
BATCH_SIZE = 8       # 한 번에 처리할 '카드' 개수
MAX_WORKERS = 4      # 병렬 스레드 수

# ============================================================
# 2. Elasticsearch 매핑 (Nested Schema)
# ============================================================
def create_index_with_mapping(es: Elasticsearch):
    """
    Card(Root) -> Benefits(Nested) 구조의 인덱스 생성
    """
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
                # --- Root Level: 카드 기본 정보 ---
                "card_id": {"type": "keyword"},
                "card_name": {"type": "text", "analyzer": "korean_analyzer"},
                "card_company": {"type": "keyword"},
                "annual_fee": {"type": "integer"},
                
                # --- Nested Level: 혜택 정보 ---
                "benefits": {
                    "type": "nested",  # 중요: Nested 타입
                    "properties": {
                        "category": {"type": "keyword"},
                        "summary": {"type": "text", "analyzer": "korean_analyzer"},
                        "description": {"type": "text", "analyzer": "korean_analyzer"},
                        
                        # [Vector] 혜택별 임베딩
                        "vector": {
                            "type": "dense_vector",
                            "dims": EMBEDDING_DIM,
                            "index": True,
                            "similarity": "cosine"
                        },
                        
                        # [Struct] 파싱된 조건(Tier) 정보
                        "tiers": {
                            "type": "nested",
                            "properties": {
                                "min_spend": {"type": "long"},
                                "max_cap": {"type": "long"},
                                "rate": {"type": "float"},
                                "unit": {"type": "keyword"}
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
# 3. 텍스트 파싱 로직 (Parsing Logic)
# ============================================================
def extract_tiers_from_text(text: str, default_value: float) -> List[Dict]:
    """
    텍스트에서 정규표현식을 사용해 '전월실적'과 '한도/적립률' 구조 추출
    """
    tiers = []
    
    # 예: "40만원 이상... 1만 포인트 적립" 패턴 매칭
    pattern_tier = re.compile(r'(\d+)만원.*?이상.*?(\d+)(만|천)?\s*(원|점|포인트)')
    
    matches = pattern_tier.findall(text)
    for match in matches:
        min_spend_str, cap_str, unit_big, unit_type = match
        
        # 금액 계산
        min_spend = int(min_spend_str) * 10000
        
        cap = int(cap_str)
        if unit_big == '만': cap *= 10000
        elif unit_big == '천': cap *= 1000
        
        tiers.append({
            "min_spend": min_spend,
            "max_cap": cap,
            "rate": default_value / 100.0 if default_value else 0.0,
            "unit": "KRW" if unit_type == "원" else "POINT"
        })
    
    # 매칭되는 패턴이 없으면 기본값 처리
    if not tiers:
        tiers.append({
            "min_spend": 0,
            "max_cap": -1,  # 한도 없음 식별용
            "rate": default_value / 100.0 if default_value else 0.0,
            "unit": "UNKNOWN"
        })
        
    return tiers

# ============================================================
# 4. 데이터 변환 및 임베딩 처리 (Worker)
# ============================================================
def process_single_card(card_id: str, chunks: List[Dict], embedding_model) -> Dict:
    """
    [Worker 함수]
    하나의 카드 ID에 속한 여러 청크를 모아서
    1. 텍스트 파싱 (Tiers)
    2. 임베딩 생성 (LangChain 이용)
    3. Nested 문서 구조로 변환
    """
    if not chunks:
        return None

    try:
        # 1. 카드 기본 정보 (첫 번째 청크 메타데이터 활용)
        base_meta = chunks[0]["metadata"]
        card_doc = {
            "_id": card_id,
            "_index": INDEX_NAME,
            "_source": {
                "card_id": card_id,
                "card_name": base_meta.get("card_name", "Unknown"),
                "card_company": base_meta.get("card_company", ""),
                "annual_fee": base_meta.get("domestic_year_cost", 0),
                "benefits": []  # 여기에 혜택들이 쌓임
            }
        }

        # 2. 각 혜택(Chunk) 처리
        for chunk in chunks:
            meta = chunk["metadata"]
            content = chunk["page_content"]
            
            # (A) 텍스트 파싱 (Tiers 구조화)
            default_val = meta.get("benefit_value", 0)
            tiers = extract_tiers_from_text(content, default_val)
            
            # (B) 임베딩 생성 (LangChain 사용)
            # 검색 품질 향상을 위해 '요약 + 상세내용'을 합쳐서 벡터화
            text_to_embed = f"{meta.get('benefit_summary', '')} {content}"
            
            # LangChain의 embed_query 사용 (동기 호출)
            vector = embedding_model.embed_query(text_to_embed)
            
            # (C) 혜택 객체 조립
            benefit_obj = {
                "category": meta.get("category", "기타"),
                "summary": meta.get("benefit_summary", ""),
                "description": content,
                "vector": vector,  # Dense Vector
                "tiers": tiers     # Nested Structure
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

    # 1. Elasticsearch 연결
    es = Elasticsearch(ELASTICSEARCH_URL)
    if not es.ping():
        raise ConnectionError("Elasticsearch에 연결할 수 없습니다.")
    
    create_index_with_mapping(es)

    # 2. 임베딩 모델 초기화 (LangChain)
    print(f"🚀 임베딩 모델 로드 중: {EMBEDDING_MODEL_ID}")
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_ID,
        task="feature-extraction",
        huggingfacehub_api_token=HF_API_KEY,
    )

    # 3. 데이터 로드 및 그룹화
    # JSON 파일은 혜택(청크) 단위로 되어있으므로, 카드 ID로 묶어야 함
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} 파일이 없습니다.")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    card_groups = defaultdict(list)
    for item in raw_data:
        cid = item["metadata"].get("card_id")
        if cid:
            card_groups[cid].append(item)

    # 4. 체크포인트 확인
    processed_ids = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            processed_ids = set(json.load(f))
            
    # 처리할 대상 필터링
    target_card_ids = [cid for cid in card_groups.keys() if cid not in processed_ids]
    target_groups = [(cid, card_groups[cid]) for cid in target_card_ids]
    
    # ==========================================
    # 🛑 [수정] 테스트를 위해 10개만 자르기
    # ==========================================
    target_groups = target_groups[:10]
    # ==========================================

    print(f"📊 전체 카드: {len(card_groups)}개 | 완료: {len(processed_ids)}개 | 작업 예정: {len(target_groups)}개")

    if not target_groups:
        print("🎉 모든 작업이 완료되었습니다.")
        return

    # 5. 배치 처리 및 병렬 실행
    # target_groups 리스트를 BATCH_SIZE만큼 쪼개서 처리
    batches = [target_groups[i:i + BATCH_SIZE] for i in range(0, len(target_groups), BATCH_SIZE)]
    
    total_indexed = 0
    
    print(f"🚀 병렬 처리 시작 (Workers: {MAX_WORKERS}, Batch Size: {BATCH_SIZE})")
    
    # ThreadPoolExecutor 시작
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for batch in tqdm(batches, desc="Total Progress"):
            # 배치 내 각 카드를 병렬로 변환/임베딩
            futures = []
            for cid, chunks in batch:
                # embeddings 객체를 인자로 전달 (Thread-safe 가정)
                futures.append(executor.submit(process_single_card, cid, chunks, embeddings))
            
            bulk_docs = []
            completed_ids = []
            
            # 결과 수집
            for future in as_completed(futures):
                result_doc = future.result()
                if result_doc:
                    bulk_docs.append(result_doc)
                    completed_ids.append(result_doc["_source"]["card_id"])
            
            # Elasticsearch Bulk 적재
            if bulk_docs:
                try:
                    success, failed = helpers.bulk(es, bulk_docs, stats_only=True)
                    if failed:
                        print(f"\n⚠️ {failed}건 인덱싱 실패")
                    
                    # 성공적으로 적재된 ID만 체크포인트에 기록
                    processed_ids.update(completed_ids)
                    with open(CHECKPOINT_FILE, 'w') as f:
                        json.dump(list(processed_ids), f)
                        
                    total_indexed += len(completed_ids)

                except Exception as e:
                    print(f"\n❌ ES Bulk Error: {e}")
            
            # API Rate Limit 조절용 딜레이 (선택사항)
            # time.sleep(0.5)

    print(f"\n🎉 작업 종료! 총 {total_indexed}개의 카드가 인덱싱되었습니다.")

if __name__ == "__main__":
    main()