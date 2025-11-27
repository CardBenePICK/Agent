import json
import os

def remove_specific_key(input_path, output_path):
    # 1. 파일 읽기
    if not os.path.exists(input_path):
        print(f"⚠️ 파일을 찾을 수 없습니다: {input_path}")
        return

    print(f"📂 파일 로딩 중: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 키 삭제 작업
    removed_count = 0
    for item in data:
        # metadata가 딕셔너리로 존재하는지 확인
        if "metadata" in item and isinstance(item["metadata"], dict):
            # pop(키, None)을 사용하면 키가 없어도 에러가 나지 않습니다.
            # 리턴값이 None이 아니면 삭제가 수행된 것입니다.
            if item["metadata"].pop("previous_month_performance", None) is not None:
                removed_count += 1

    # 3. 변경된 내용 저장
    print(f"💾 저장 중: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False로 해야 한글이 깨지지 않고 저장됩니다.
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("-" * 30)
    print(f"✅ 작업 완료!")
    print(f"🗑️  총 {removed_count}개의 항목에서 'previous_month_performance' 필드를 삭제했습니다.")
    print(f"👉 결과 파일: {output_path}")

if __name__ == "__main__":
    # 현재 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 입력 파일명 (기존 파일명)
    input_filename = "processed_card_chunks_only_credit_1126.json"
    
    # 출력 파일명 (뒤에 _cleaned 붙임)
    output_filename = "processed_card_chunks_only_credit_1126.json"

    input_path = os.path.join(current_dir, input_filename)
    output_path = os.path.join(current_dir, output_filename)

    remove_specific_key(input_path, output_path)