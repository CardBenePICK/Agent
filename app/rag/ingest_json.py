import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers

# .env 파일 로드
load_dotenv()

# ============================================================
# 1. 설정 (Configuration)
# ============================================================
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

# [설정] 데이터를 저장할 새로운 인덱스 이름
INDEX_NAME = "credit_cards_nested_v2"

# [설정] 이미 벡터가 포함된 원본 파일 경로 (업로드해주신 파일명)
INPUT_FILE = "credit_cards_backup.json" 

BATCH_SIZE = 100  # 벡터 생성이 없으므로 배치를 크게 잡음

# ============================================================
# 2. Elasticsearch 인덱스 및 맵핑 생성
# ============================================================
def create_index_if_not_exists(es: Elasticsearch):
    """
    인덱스가 없을 경우, Nested 구조와 Vector 설정이 포함된 맵핑으로 생성합니다.
    """
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "tokenizer": {
                    "nori_tokenizer_mixed": {
                        "type": "nori_tokenizer",
                        "decompound_mode": "mixed"
                    }
                },
                "analyzer": {
                    "korean_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer_mixed",
                        "filter": ["lowercase", "nori_part_of_speech"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                # --- Root Fields ---
                "card_id": {"type": "keyword"},
                "card_name": {"type": "text", "analyzer": "korean_analyzer"},
                "card_company": {"type": "keyword"},
                "domestic_year_cost": {"type": "integer"},
                "abroad_year_cost": {"type": "integer"},
                "previous_month_performance": {"type": "integer"},
                
                # --- Nested Fields (Benefits) ---
                "benefits": {
                    "type": "nested", 
                    "properties": {
                        "category": {"type": "keyword"},
                        "summary": {"type": "text", "analyzer": "korean_analyzer"},
                        "description": {"type": "text", "analyzer": "korean_analyzer"},
                        
                        # [중요] 이미 있는 벡터를 담을 필드 정의 (차원수 1024 확인)
                        "vector": {
                            "type": "dense_vector",
                            "dims": 1024,  
                            "index": True,
                            "similarity": "cosine"
                        },
                        
                        # --- Nested Fields (Tiers) ---
                        "tiers": {
                            "type": "nested",
                            "properties": {
                                "previous_min_spend": {"type": "long"},
                                "rate": {"type": "float"},
                                "unit": {"type": "keyword"},
                                "type": {"type": "keyword"}
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
        # 맵핑 충돌 방지를 위해 기존 인덱스가 있다면 삭제 후 재생성 추천 (선택사항)
        print(f"ℹ️ 인덱스가 이미 존재합니다: {INDEX_NAME}")
        # es.indices.delete(index=INDEX_NAME)
        # es.indices.create(index=INDEX_NAME, body=mapping)
        # print(f"♻️ 인덱스 재생성 완료")

# ============================================================
# 3. 데이터 로딩 및 변환 (Generator)
# ============================================================
def generate_actions(filename):
    """
    NDJSON 파일을 한 줄씩 읽어서 ES Bulk Action 형태로 변환 (메모리 효율적)
    """
    with open(filename, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 1. JSON 파싱
                doc = json.loads(line)
                
                # 2. _source 데이터 추출 (백업 파일 구조에 따름)
                source_data = doc.get('_source')
                if not source_data:
                    continue

                # 3. 데이터 정제 (필요시)
                # 원본 파일에 이미 벡터가 있으므로 그대로 사용
                
                # 4. Bulk Action 생성
                action = {
                    "_index": INDEX_NAME,
                    "_id": source_data.get("card_id"),  # ID 지정
                    "_source": source_data
                }
                yield action

            except json.JSONDecodeError:
                print(f"⚠️ JSON 파싱 에러 (Line {line_number})")
                continue

# ============================================================
# 4. 메인 실행
# ============================================================
def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ 입력 파일이 없습니다: {INPUT_FILE}")

    # 1. ES 연결
    es = Elasticsearch(ELASTICSEARCH_URL)
    if not es.ping():
        raise ConnectionError("❌ Elasticsearch에 연결할 수 없습니다.")
    
    # 2. 인덱스 준비
    create_index_if_not_exists(es)

    # 3. 데이터 입력 (Bulk Insert)
    print(f"🚀 데이터 입력을 시작합니다... (File: {INPUT_FILE})")
    
    try:
        # helpers.streaming_bulk는 제너레이터를 사용하여 대용량 데이터도 끊어서 전송함
        success_count = 0
        failed_count = 0
        
        # 진행률 표시줄(tqdm)과 함께 실행
        for ok, info in tqdm(helpers.streaming_bulk(es, generate_actions(INPUT_FILE), chunk_size=BATCH_SIZE)):
            if ok:
                success_count += 1
            else:
                failed_count += 1
                print(f"실패: {info}")

        print(f"\n🎉 작업 완료!")
        print(f"✅ 성공: {success_count} 건")
        if failed_count > 0:
            print(f"❌ 실패: {failed_count} 건")

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")

if __name__ == "__main__":
    main()