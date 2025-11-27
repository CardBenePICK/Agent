import os
import json
import re

# ⛳️ 현재 스크립트 위치 기준 경로 자동 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT = os.path.join(BASE_DIR, "brand_dictionary_with_card.json")
OUTPUT = os.path.join(BASE_DIR, "brand_dictionary.json")

print("📄 INPUT:", INPUT)
print("📄 OUTPUT:", OUTPUT)

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

    # 불용어 제거
    if t in ["서비스", "포인트", "혜택"]:
        return False

    # 단일 한글 제거
    if re.fullmatch(r"[가-힣]", t):
        return False

    return True


def clean(brands):
    return [b for b in brands if is_valid_brand(b)]


if __name__ == "__main__":
    # 파일 존재 검증
    if not os.path.exists(INPUT):
        print("❌ ERROR: 파일을 찾을 수 없습니다:", INPUT)
        exit(1)

    with open(INPUT, "r", encoding="utf-8") as f:
        brands = json.load(f)

    print("🔍 원본 개수:", len(brands))

    cleaned = clean(brands)

    print("🧹 정제 후 개수:", len(cleaned))

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(sorted(cleaned), f, ensure_ascii=False, indent=2)

    print("✅ 저장 완료!")
