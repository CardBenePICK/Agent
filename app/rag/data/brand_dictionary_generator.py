import os
import json
import time
import requests
from elasticsearch import Elasticsearch

# ================================
# 설정
# ================================
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = "card_benefit_bgem3_v1"
HF_TOKEN = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
ROUTER_API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "google/gemma-2-9b-it:nebius"

OUTPUT_JSON = "brand_dictionary.json"
CHECKPOINT_JSON = "brand_progress.json"

CHUNK_SIZE = 1          # 문서당 1 chunk
SAVE_EVERY = 20         # N chunks 처리할 때마다 자동 저장
RETRY_LIMIT = 3         # HF API 실패 시 재시도 횟수


# ================================
# HF Router ChatCompletion wrapper
# ================================
def hf_chat(messages):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 120,
    }

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = requests.post(ROUTER_API_URL, json=payload, headers=headers, timeout=30)
            data = resp.json()
            return data
        except Exception as e:
            print(f"⚠️ HF API 오류 (시도 {attempt}/{RETRY_LIMIT}) → {e}")
            time.sleep(2)

    print("❌ HF API 재시도 실패 → None 반환")
    return None


# ================================
# 체크포인트 로드
# ================================
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_JSON):
        return {"processed": 0}

    with open(CHECKPOINT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(idx):
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump({"processed": idx}, f, ensure_ascii=False, indent=2)


# ================================
# 기존 브랜드 사전 누적 로드
# ================================
def load_brand_dict():
    if not os.path.exists(OUTPUT_JSON):
        return set()

    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        old_list = json.load(f)
        return set(old_list)


def save_brand_dict(brand_set):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(list(brand_set)), f, ensure_ascii=False, indent=2)


# ================================
# Elasticsearch 전체 문서 로드
# ================================
def load_all_docs():
    print("🔍 Elasticsearch 전체 문서 읽는 중…")

    es = Elasticsearch(ELASTICSEARCH_URL, verify_certs=False)
    query = {"query": {"match_all": {}}}

    docs = []
    res = es.search(index=INDEX_NAME, body=query, scroll="2m", size=500)
    sid = res["_scroll_id"]
    hits = res["hits"]["hits"]
    docs.extend(hits)

    while len(hits) > 0:
        res = es.scroll(scroll_id=sid, scroll="2m")
        sid = res["_scroll_id"]
        hits = res["hits"]["hits"]
        docs.extend(hits)

    print(f"📄 총 {len(docs)}개 문서 로드 완료")
    return docs


# ================================
# 브랜드 추출
# ================================
def extract_brands(text):
    messages = [
        {
            "role": "system",
            "content": "텍스트에서 브랜드명만 JSON 배열로 추출해줘. 예: ['스타벅스','이마트']"
        },
        {
            "role": "user",
            "content": text[:2000]  
        }
    ]

    res = hf_chat(messages)
    if not res or "choices" not in res:
        return []

    try:
        output = res["choices"][0]["message"]["content"]
        data = json.loads(output)
        return [x.strip() for x in data]
    except Exception:
        return []


# ================================
# 메인 실행
# ================================
if __name__ == "__main__":

    # 1) 데이터 로드
    docs = load_all_docs()
    total_chunks = len(docs)

    # 2) checkpoint & 기존 사전 로드
    checkpoint = load_checkpoint()
    start_idx = checkpoint["processed"]

    print(f"⏳ 이전 처리 지점: chunk {start_idx}/{total_chunks}")
    brand_set = load_brand_dict()

    # 3) chunk 처리 loop
    print("\n🚀 브랜드 자동 추출 시작 (재시작 가능)…\n")

    for idx in range(start_idx, total_chunks):

        if idx % SAVE_EVERY == 0:
            print(f"💾 자동 저장 — 진행률 {idx}/{total_chunks}")
            save_checkpoint(idx)
            save_brand_dict(brand_set)

        hit = docs[idx]["_source"]
        raw_text = hit.get("text", "")
        meta = hit.get("metadata", {})
        benefit = meta.get("benefit_summary", "")
        chunk_text = f"{benefit}\n{raw_text}"

        print(f"→ Chunk {idx}/{total_chunks} 처리 중…")

        try:
            brands = extract_brands(chunk_text)
            brand_set.update(brands)
        except Exception as e:
            print(f"⚠️ Chunk {idx} 처리 오류: {e}")

    # 4) 최종 저장
    print("\n🏁 전체 처리 완료 → 최종 저장 중…")
    save_checkpoint(total_chunks)
    save_brand_dict(brand_set)

    print(f"📦 브랜드 사전 저장 완료 → {OUTPUT_JSON}")
    print("✨ 작업 종료")
