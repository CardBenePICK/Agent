import os
import json
import re
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------
# 파일 경로 설정 (절대경로, 덮어쓰기 방지)
# -------------------------------------------------------
RAW_INPUT = os.path.join(BASE_DIR, "brand_dictionary_with_card.json")

CLEAN_OUTPUT = os.path.join(
    BASE_DIR, f"brand_dictionary_cleaned_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
)

CATEGORY_OUTPUT = os.path.join(
    BASE_DIR, f"brand_dictionary_categorized_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
)

FINAL_BRAND_LIST = os.path.join(BASE_DIR, "brand_list_final.json")
FINAL_CATEGORY_MAP = os.path.join(BASE_DIR, "brand_category_map.json")


# -------------------------------------------------------
# 1) 브랜드 정제 함수
# -------------------------------------------------------
def is_valid_brand(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False

    t = text.strip()

    # 길이 제한
    if len(t) < 2 or len(t) > 20:
        return False

    # JSON 깨짐 문구 제거
    if t in ["']", "]}", "'}", "}]", "''", '"', "'"]:
        return False

    # 숫자/기호 시작 제거
    if re.match(r"^[0-9\W]", t):
        return False

    # 혜택 문구 제거
    if any(kw in t for kw in ["할인", "캐시백", "적립", "면제", "%"]):
        return False

    # 회사/기관명 제거
    if any(kw in t for kw in ["(주)", "유한회사", "협회", "공단", "공사", "은행"]):
        return False

    # 카드 관련 제거
    if "카드" in t.lower() or "card" in t.lower():
        return False

    # 단일 한글 제거
    if re.fullmatch(r"[가-힣]", t):
        return False

    return True


def clean_brands(brands):
    return sorted([b for b in brands if is_valid_brand(b)])


# -------------------------------------------------------
# 2) 브랜드 카테고리 자동 분류 (HF Router)
# -------------------------------------------------------

API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")

MODEL = "google/gemma-2-9b-it:nebius"

CATEGORY_INSTRUCTION = """
다음 브랜드의 카테고리를 지정해줘.

카테고리 리스트:
- 카페
- 편의점
- 백화점/마트
- 온라인쇼핑
- 배달앱
- 주유/차량
- 패션/의류
- 문화/영화
- 통신
- 금융
- 기타

출력 형식(JSON):
{"brand": "<브랜드>", "category": "<카테고리>"}
"""


def hf_chat(messages):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 120,
        "temperature": 0.1
    }

    resp = requests.post(API_URL, headers=headers, json=payload)

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ HF Router Error:", e)
        return None


def categorize_brands(cleaned):
    results = []

    for idx, brand in enumerate(cleaned, 1):
        print(f"🗂 [{idx}/{len(cleaned)}] 분류 중 → {brand}")

        messages = [
            {"role": "system", "content": "너는 한국 브랜드 카테고리 분류 전문가다."},
            {"role": "user", "content": CATEGORY_INSTRUCTION + f"\n브랜드: {brand}"}
        ]

        res = hf_chat(messages)
        if not res:
            results.append({"brand": brand, "category": "기타"})
            continue

        try:
            parsed = json.loads(res)
            results.append(parsed)
        except:
            results.append({"brand": brand, "category": "기타"})

    return results


# -------------------------------------------------------
# 3) retriever_tool 최적화용 자원 생성
# -------------------------------------------------------
def build_final_resources(categorized):
    brand_list = sorted([item["brand"] for item in categorized])
    category_map = {item["brand"]: item["category"] for item in categorized}

    with open(FINAL_BRAND_LIST, "w", encoding="utf-8") as f:
        json.dump(brand_list, f, ensure_ascii=False, indent=2)

    with open(FINAL_CATEGORY_MAP, "w", encoding="utf-8") as f:
        json.dump(category_map, f, ensure_ascii=False, indent=2)

    print("📦 retriever_tool 통합용 파일 생성 완료!")
    print(" → brand_list_final.json")
    print(" → brand_category_map.json")


# -------------------------------------------------------
# MAIN 실행
# -------------------------------------------------------
if __name__ == "__main__":

    if not os.path.exists(RAW_INPUT):
        print("❌ 입력 파일이 없습니다:", RAW_INPUT)
        exit(1)

    # 1) CLEAN
    print("🔍 1) 브랜드 정제 중…")
    with open(RAW_INPUT, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = clean_brands(raw)
    with open(CLEAN_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"✅ 정제 완료 → {CLEAN_OUTPUT}")

    # 2) CATEGORIZE
    print("\n🧠 2) 브랜드 카테고리 분류 중…")
    categorized = categorize_brands(cleaned)
    with open(CATEGORY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(categorized, f, ensure_ascii=False, indent=2)

    print(f"✅ 카테고리 분류 완료 → {CATEGORY_OUTPUT}")

    # 3) FINAL RAG RESOURCE
    print("\n📦 3) retriever_tool 최종 리소스 생성 중…")
    build_final_resources(categorized)

    print("\n🎉 모든 단계 완료!")
