# data/mappings.py

CATEGORY_CODE_MAP = {
    # [1] 고정비/필수
    "TRANSPORT": "대중교통",
    "FUEL": "주유",
    "TELECOM": "통신",
    "UTILITIES": "공과금",

    # [2] 식생활
    "CAFE": "카페",
    "CONVENIENCE": "편의점",
    "DELIVERY": "배달앱",
    "DINING": "일반음식점",

    # [3] 쇼핑
    "ONLINE_SHOP": "온라인쇼핑",
    "MART": "대형마트",
    "DEPT_STORE": "백화점",
    "BAKERY": "베이커리",

    # [4] 라이프/여가
    "OTT": "OTT",
    "MOVIE": "영화",
    "HOSPITAL": "병원",
    "ACADEMY": "학원",

    # [5] 트렌드/특화
    "PAY": "간편결제",
    "OVERSEAS": "해외이용",
    "AIRLINE": "항공마일리지",
    "LOUNGE": "공항라운지"
}

def map_codes_to_korean(codes: list[str]) -> list[str]:
    return [CATEGORY_CODE_MAP.get(code, code) for code in codes]