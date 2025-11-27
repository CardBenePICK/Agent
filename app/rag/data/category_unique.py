import json
import os

# 원본 데이터 파일 경로
SOURCE_FILE = 'processed_card_chunks_only_credit_1126.json'
# 생성할 사전 파일 경로
OUTPUT_FILE = 'category_dictionary.py'

def generate_dictionary_file():
    print(f"📂 '{SOURCE_FILE}' 로딩 중...")
    
    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 중복 제거를 위한 Set 생성
        unique_categories = set()
        
        for item in data:
            if 'metadata' in item and 'category' in item['metadata']:
                raw_cat = item['metadata']['category']
                
                if not raw_cat:
                    continue
                    
                # 문자열이면 처리
                if isinstance(raw_cat, str):
                    # 1. 슬래시(/)가 포함된 경우 분리
                    if '/' in raw_cat:
                        parts = raw_cat.split('/')
                        for part in parts:
                            clean_part = part.strip() # 공백 제거
                            if clean_part:
                                unique_categories.add(clean_part)
                    else:
                        # 2. 슬래시가 없으면 그대로 추가
                        unique_categories.add(raw_cat.strip())

        # 가나다순 정렬
        sorted_categories = sorted(list(unique_categories))
        
        # 파이썬 파일로 저장
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("# 이 파일은 update_category_dict.py에 의해 자동 생성되었습니다.\n")
            f.write("# 슬래시(/)로 구분된 항목은 개별 키워드로 분리되었습니다.\n\n")
            
            f.write("KNOWN_CATEGORIES = [\n")
            for cat in sorted_categories:
                f.write(f"    '{cat}',\n")
            f.write("]\n")
            
        print(f"✅ '{OUTPUT_FILE}' 생성 완료!")
        print(f"📊 총 {len(sorted_categories)}개의 고유 카테고리가 등록되었습니다.")
        
        # 확인용 출력 (상위 10개)
        print("👀 미리보기 (앞 10개):", sorted_categories[:10])

    except FileNotFoundError:
        print(f"❌ 오류: 원본 파일 '{SOURCE_FILE}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    generate_dictionary_file()