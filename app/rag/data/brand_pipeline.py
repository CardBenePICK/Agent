import os
import json
import re
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ======================================================
# 기본 설정
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_INPUT = os.path.join(BASE_DIR, "brand_dictionary_with_card.json")

ts = datetime.now().strftime("%Y%m%d_%H%M")

CLEAN_OUTPUT = os.path.join(BASE_DIR, f"brand_dictionary_cleaned_{ts}.json")
CATEGORY_OUTPUT = os.path.join(BASE_DIR, f"brand_dictionary_categorized_{ts}.json")

FINAL_BRAND_LIST = os.path.join(BASE_DIR, "brand_list_final.json")
FINAL_CATEGORY_MAP = os.path.join(BASE_DIR, "brand_category_map.json")

CATEGORY_CACHE_PATH = os.path.join(BASE_DIR, "brand_category_cache.json")

# ======================================================
# HF 모델 설정 (Zero-shot Classification)
# ======================================================
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("❌ HF_TOKEN 환경변수가 필요합니다.")

API_URL = "https://router.huggingface.co/hf-inference/models/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ======================================================
# 브랜드 정제
# ======================================================
def is_valid_brand(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    t = text.strip()

    if len(t) < 2 or len(t) > 20:
        return False
    if t in ["']", "]}", "'}", "}]", "''", '"', "'"]:
        return False
    if re.match(r"^[0-9\W]", t):
        return False

    for kw in ["할인", "캐시백", "적립", "면제", "%"]:
        if kw in t:
            return False

    for kw in ["(주)", "유한회사", "협회", "공단", "공사", "은행"]:
        if kw in t:
            return False

    if "카드" in t.lower() or "card" in t.lower():
        return False

    if re.fullmatch(r"[가-힣]", t):
        return False

    return True


def clean_brands(brands):
    cleaned = [b for b in brands if is_valid_brand(b)]
    return sorted(set(cleaned))


# ======================================================
# 카테고리 정의
# ======================================================
CATEGORIES = [
    "카페","편의점","백화점/마트","온라인쇼핑","배달앱","주유/차량",
    "교통/대중교통/택시","여행/항공/숙박","패션/의류","문화/영화/도서",
    "통신","생활요금/공과금","금융/보험/렌터카","외식/음식점","뷰티/미용",
    "교육/학원","유아/키즈","반려동물","레저/스포츠/테마파크","기타"
]

# ======================================================
# 그룹 / 법인 / 계열사 → 자동 기타
# ======================================================
NO_CATEGORY_KEYWORDS = [
    "계열", "계열사", "관계사", "지주", "홀딩스", "파트너스",
    "산업", "인터내셔널", "엔터프라이즈", "그룹", "법인",
    "유한회사", "(주)", "llc", "inc", "co", "corporation"
]

def is_group_or_corporate(name):
    n = name.lower()
    return any(kw in n for kw in NO_CATEGORY_KEYWORDS)


# ======================================================
# Zero-shot Classification
# ======================================================
def classify_brand(brand):
    # 1) 전처리: 기업/그룹명은 자동 기타
    if is_group_or_corporate(brand):
        return "기타"

    payload = {
        "inputs": brand,
        "parameters": {
            "candidate_labels": CATEGORIES
        }
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        result = response.json()

        # 정상 응답
        if "labels" in result and "scores" in result:
            best_label = result["labels"][0]
            best_score = result["scores"][0]

            # 2) 후처리: 신뢰도 낮으면 기타 처리
            if best_score < 0.35:
                return "기타"

            return best_label

        # 에러 처리
        if "error" in result:
            print("❌ API Error:", result)
            return "기타"

        print("⚠️ Unexpected response:", result)
        return "기타"

    except Exception as e:
        print("❌ API 호출 오류:", e)
        return "기타"


# ======================================================
# 체크포인트
# ======================================================
def load_category_cache():
    if os.path.exists(CATEGORY_CACHE_PATH):
        return json.load(open(CATEGORY_CACHE_PATH, "r", encoding="utf-8"))
    return {}

def save_category_cache(cache):
    json.dump(cache, open(CATEGORY_CACHE_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ======================================================
# 메인 분류 로직 (Resume 지원)
# ======================================================
def categorize_brands(cleaned):
    cache = load_category_cache()

    pending = [b for b in cleaned if b not in cache]
    print(f"🔍 이미 분류된 브랜드: {len(cache)}개")
    print(f"🚀 새로 분류해야 할 브랜드: {len(pending)}개\n")

    for idx, brand in enumerate(pending, 1):
        print(f" ⏳ [{idx}/{len(pending)}] 분류 중: {brand}")
        category = classify_brand(brand)

        cache[brand] = category
        save_category_cache(cache)

        print(f" ✔ {brand} → {category}\n")

    return cache


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    if not os.path.exists(RAW_INPUT):
        raise SystemExit("❌ 입력 파일 없음")

    print("🔍 1) 브랜드 정제 중…")
    raw = json.load(open(RAW_INPUT, "r", encoding="utf-8"))
    cleaned = clean_brands(raw)
    json.dump(cleaned, open(CLEAN_OUTPUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✔ 정제 완료: {len(cleaned)}개\n")

    print("🧠 2) Zero-shot 카테고리 분류 시작…")
    category_map = categorize_brands(cleaned)

    # 전체 결과 저장
    json.dump(
        [{"brand": b, "category": category_map[b]} for b in cleaned],
        open(CATEGORY_OUTPUT, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    json.dump(cleaned, open(FINAL_BRAND_LIST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(category_map, open(FINAL_CATEGORY_MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n🎉 전체 파이프라인 완료!")
    print("📦 최종 결과:", FINAL_CATEGORY_MAP)
