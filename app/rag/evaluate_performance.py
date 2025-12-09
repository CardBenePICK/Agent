import json
import os
import sys
import time
from tqdm import tqdm

# [설정] rag 모듈 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.chatbot_pipeline import run_pipeline

# 평가 데이터셋 경로
# DATASET_PATH = "data/evaluation_dataset.json"
DATASET_PATH ="data/evaluation_dataset_complex_top100_generated.json"

def calculate_metrics(k=3):
    if not os.path.exists(DATASET_PATH):
        print("❌ 평가 데이터셋이 없습니다. generate_dataset.py를 먼저 실행하세요.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"🚀 RAG 성능 평가를 시작합니다. (Top-K: {k}, Total Queries: {len(eval_data)})")
    print("-" * 60)
    print(f"{'Type':<10} | {'Query':<40} | {'Result':<10} | {'Rank'}")
    print("-" * 60)

    total_count = 0
    hit_count = 0
    mrr_sum = 0
    
    # 타입별 통계
    type_stats = {"Simple": {"hit": 0, "total": 0}, "Complex": {"hit": 0, "total": 0}}

    for item in tqdm(eval_data):
        query = item['query']
        ground_truth_ids = set(map(str, item['ground_truth_ids'])) # 문자열로 통일
        q_type = item['type']
        
        try:
            # RAG 파이프라인 실행
            results = run_pipeline(query) 
            
            # 결과에서 ID 추출 (Top-K)
            # results는 dict list라고 가정 [{'card_id': ...}, ...]
            recommended_ids = [str(r.get('card_id') or r.get('id')) for r in results[:k]]
            
            # 정답 확인
            is_hit = False
            rank = 0
            
            for idx, rec_id in enumerate(recommended_ids):
                if rec_id in ground_truth_ids:
                    is_hit = True
                    rank = idx + 1
                    break
            
            # 지표 업데이트
            total_count += 1
            type_stats[q_type]["total"] += 1
            
            if is_hit:
                hit_count += 1
                mrr_sum += 1 / rank
                type_stats[q_type]["hit"] += 1
                print(f"{q_type:<10} | {query[:38]:<40} | ✅ HIT     | {rank}")
            else:
                print(f"{q_type:<10} | {query[:38]:<40} | ❌ MISS    | -")
                
        except Exception as e:
            print(f"Error processing query '{query}': {e}")

    # --- 결과 리포트 출력 ---
    hit_rate = (hit_count / total_count) * 100 if total_count > 0 else 0
    mrr = mrr_sum / total_count if total_count > 0 else 0
    
    print("\n" + "=" * 50)
    print("📊 [Final Evaluation Report]")
    print("=" * 50)
    print(f"🎯 Overall Hit Rate @ {k} : {hit_rate:.2f}%")
    print(f"🥇 Overall MRR          : {mrr:.4f}")
    print("-" * 50)
    
    # 타입별 상세 결과
    for t, stat in type_stats.items():
        t_hit_rate = (stat['hit'] / stat['total'] * 100) if stat['total'] > 0 else 0
        print(f"🔹 {t:<10} Hit Rate    : {t_hit_rate:.2f}% ({stat['hit']}/{stat['total']})")
    print("=" * 50)

    # 발표 자료용 텍스트 생성
    print("\n[📢 발표 자료용 요약 멘트]")
    print(f"\"총 {total_count}개의 테스트 쿼리(단순/복합 혼합)에 대해 평가를 진행한 결과,")
    print(f"상위 {k}개 추천 내 정답 포함 비율인 Hit Rate는 {hit_rate:.1f}%를 기록했으며,")
    print(f"평균적으로 정답 카드가 {1/mrr:.1f}번째 순위에 노출되는 {mrr:.2f}의 MRR 점수를 달성했습니다.\"")

if __name__ == "__main__":
    calculate_metrics(k=3)