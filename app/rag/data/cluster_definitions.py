# data/persona_definitions.py

PERSONA_CLUSTERS = {
    0: {
        "name_kr": "실속 미식가",
        "name_en": "Value Diner",
        "avg_spend": 400000,  # 40만원
        "base_keywords": ["외식", "병원", "약국", "대중교통", "식비", "의료비"],
        "description": "식비와 의료비 지출 비중이 높은 실속형 소비자. 외식 및 병원비 할인 중점."
    },
    1: {
        "name_kr": "알뜰 소액족",
        "name_en": "Minimalist",
        "avg_spend": 200000,  # 20만원
        "base_keywords": ["주유", "통신", "공과금", "보험료", "무실적"],
        "description": "소비가 적지만 고정비(주유, 통신, 공과금) 방어가 필요한 사용자. 전월 실적 없는 카드 선호."
    },
    2: {
        "name_kr": "에듀 맘/대디",
        "name_en": "Edu-Focus",
        "avg_spend": 500000,  # 50만원
        "base_keywords": ["학원", "서점", "가족외식", "교육", "자녀"],
        "description": "중고등학생 자녀 학원비 및 교육비 지출이 큰 학부모."
    },
    3: {
        "name_kr": "럭셔리 VIP",
        "name_en": "Affluent Lifestyle",
        "avg_spend": 1000000, # 100만원
        "base_keywords": ["항공마일리지", "공항라운지", "골프", "호텔", "백화점", "여행"],
        "description": "삶의 질을 높이는 프리미엄 혜택(여행, 레저) 및 마일리지 적립 선호."
    },
    4: {
        "name_kr": "마이카 중산층",
        "name_en": "Car Owner",
        "avg_spend": 600000,  # 60만원
        "base_keywords": ["주유", "정비", "차량", "병원", "외식"],
        "description": "자차 운행이 많아 주유비 지출이 크고, 중장년층 생활 패턴(병원/외식)을 가진 사용자."
    }
}