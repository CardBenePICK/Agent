import os
import json
import re
import requests
from datetime import datetime

# ======================================================
# 기본 경로 & 파일 설정
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 입력 파일 (이미 어느 정도 정제된 버전 사용 권장)
#   예: clean_brand_dictionary.py 로 만든 brand_dictionary.json
RAW_INPUT = os.path.join(BASE_DIR, "brand_dictionary_with_card.json")

# 🔹 출력 파일들 (절대 기존 파일 덮어쓰지 않게 타임스탬프 포함)
ts = datetime.now().strftime("%Y%m%d_%H%M")

CLEAN_OUTPUT = os.path.join(BASE_DIR, f"brand_dictionary_cleaned_{ts}.json")
CATEGORY_OUTPUT = os.path.join(BASE_DIR, f"brand_dictionary_categorized_{ts}.json")

FINAL_BRAND_LIST = os.path.join(BASE_DIR, "brand_list_final.json")
FINAL_CATEGORY_MAP = os.path.join(BASE_DIR, "brand_category_map.json")


# ======================================================
# 1) 브랜드 정제 (노이즈 제거, 길이/패턴 필터링)
# ======================================================
def is_valid_brand(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False

    t = text.strip()

    # 길이 제한 (너무 짧거나, 너무 문장 같은 것 제거)
    if len(t) < 2 or len(t) > 20:
        return False

    # JSON 깨짐 조각 제거
    if t in ["']", "]}", "'}", "}]", "''", '"', "'"]:
        return False

    # 숫자/기호로 시작하는 경우 제거 (예: "10% 캐시백")
    if re.match(r"^[0-9\W]", t):
        return False

    # 혜택 문구 제거
    benefit_keywords = ["할인", "캐시백", "적립", "면제", "%"]
    if any(kw in t for kw in benefit_keywords):
        return False

    # 회사/기관명 제거 (브랜드보단 발급사/기관인 경우)
    company_keywords = ["(주)", "유한회사", "협회", "공단", "공사", "은행"]
    if any(kw in t for kw in company_keywords):
        return False

    # 카드 관련 명칭 제거
    if "카드" in t.lower() or "card" in t.lower():
        return False

    # 단일 한글 글자 제거 (노이즈)
    if re.fullmatch(r"[가-힣]", t):
        return False

    return True


def clean_brands(brands):
    """브랜드 후보 리스트에서 유효한 브랜드만 남기기"""
    cleaned = [b for b in brands if is_valid_brand(b)]
    return sorted(set(cleaned))


# ======================================================
# 2) 브랜드 카테고리 자동 분류 (HF Router + Gemma)
# ======================================================

API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")

MODEL = "google/gemma-2-9b-it:nebius"

CATEGORY_INSTRUCTION = """
다음 브랜드를 아래 카테고리 중 하나로 분류해줘.

카테고리:
카페, 편의점, 백화점/마트, 온라인쇼핑, 배달앱, 주유/차량,
교통/대중교통/택시, 여행/항공/숙박, 패션/의류, 문화/영화/도서,
통신, 생활요금/공과금, 금융/보험/렌터카, 기타

브랜드가 어디에 속하는지 가장 적절한 하나만 선택해줘.

출력은 JSON 한 줄만:
{"brand": "<브랜드>", "category": "<카테고리>"}
"""


def hf_chat(messages):
    """HF Router ChatCompletion 호출 (에러 발생 시 None 반환)"""
    if not HF_TOKEN:
        print("❌ HF_TOKEN(HF_API_KEY/HF_TOKEN)이 설정되지 않았습니다.")
        return None

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

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ HF Router Error:", e)
        return None


def extract_category_fallback(text: str) -> str:
    """JSON 파싱 실패 시, 응답 텍스트에서 카테고리 단어를 직접 추출"""
    categories = [
        "카페",
        "편의점",
        "백화점/마트",
        "온라인쇼핑",
        "배달앱",
        "주유/차량",
        "교통/대중교통/택시",
        "여행/항공/숙박",
        "패션/의류",
        "문화/영화/도서",
        "통신",
        "생활요금/공과금",
        "금융/보험/렌터카",
        "기타",
    ]

    for cat in categories:
        if cat in text:
            return cat

    return "기타"


def categorize_brands(cleaned):
    """정제된 브랜드 리스트를 카테고리별로 분류"""
    results = []

    total = len(cleaned)

    for idx, brand in enumerate(cleaned, 1):
        print(f"🗂 [{idx}/{total}] 분류 중 → {brand}")

        messages = [
            {"role": "system", "content": "너는 한국 브랜드 카테고리 분류 전문가다."},
            {"role": "user", "content": CATEGORY_INSTRUCTION + f"\n브랜드: {brand}"}
        ]

        res = hf_chat(messages)
        if not res:
            category = "기타"
        else:
            parsed = None
            try:
                parsed = json.loads(res)
            except Exception:
                parsed = None

            if isinstance(parsed, dict) and "category" in parsed:
                category = parsed["category"]
            else:
                # JSON 실패 시, 텍스트에서 카테고리 단어만 추출
                category = extract_category_fallback(res)

        results.append({"brand": brand, "category": category})
        print(f"✔ [{idx}/{total}] 분류 완료 → 브랜드: {brand}, 카테고리: {category}\n")

    return results


# ======================================================
# 3) retriever_tool 통합용 최종 리소스 생성
# ======================================================
def build_final_resources(categorized):
    """최종적으로 retriever에서 쓸 브랜드 리스트 & 카테고리 맵 생성"""
    brand_list = sorted({item["brand"] for item in categorized})
    category_map = {item["brand"]: item["category"] for item in categorized}

    with open(FINAL_BRAND_LIST, "w", encoding="utf-8") as f:
        json.dump(brand_list, f, ensure_ascii=False, indent=2)

    with open(FINAL_CATEGORY_MAP, "w", encoding="utf-8") as f:
        json.dump(category_map, f, ensure_ascii=False, indent=2)

    print("📦 retriever_tool 통합용 파일 생성 완료!")
    print(" →", FINAL_BRAND_LIST)
    print(" →", FINAL_CATEGORY_MAP)


# ======================================================
# MAIN: 전체 파이프라인 실행
# ======================================================
if __name__ == "__main__":

    if not os.path.exists(RAW_INPUT):
        print("❌ 입력 파일이 없습니다:", RAW_INPUT)
        raise SystemExit(1)

    # -------------------------
    # 1) 브랜드 정제 단계
    # -------------------------
    print("🔍 1) 브랜드 정제 중…")
    with open(RAW_INPUT, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = clean_brands(raw)
    with open(CLEAN_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"✅ 정제 완료 → {CLEAN_OUTPUT} (총 {len(cleaned)}개)")

    # -------------------------
    # 2) 카테고리 분류 단계
    # -------------------------
    print("\n🧠 2) 브랜드 카테고리 분류 중…")
    categorized = categorize_brands(cleaned)

    with open(CATEGORY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(categorized, f, ensure_ascii=False, indent=2)

    print(f"✅ 카테고리 분류 완료 → {CATEGORY_OUTPUT}")

    # -------------------------
    # 3) retriever_tool용 최종 리소스 생성
    # -------------------------
    print("\n📦 3) retriever_tool 최종 리소스 생성 중…")
    build_final_resources(categorized)

    print("\n🎉 모든 단계 완료!")
    print("   - 정제 파일:", CLEAN_OUTPUT)
    print("   - 카테고리 파일:", CATEGORY_OUTPUT)
    print("   - 최종 브랜드 리스트:", FINAL_BRAND_LIST)
    print("   - 카테고리 맵:", FINAL_CATEGORY_MAP)

"""
# 전체 파이프라인 실행 방법 (예시)

# 1) 컨테이너 진입
docker exec -it fastapi_llm_agent_final /bin/bash

# 2) 작업 디렉토리 이동
cd /app/rag

# 3) brand_dictionary.json 이 입력으로 사용됨 (변경하고 싶으면 RAW_INPUT 수정)
ls brand_dictionary*.json

# 4) 전체 파이프라인 실행
python brand_pipeline.py

# 5) 생성 파일 확인
ls brand_dictionary_cleaned_*.json
ls brand_dictionary_categorized_*.json
ls brand_list_final.json brand_category_map.json
"""
