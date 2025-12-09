import json
import os
import sys
import time
import argparse
from tqdm import tqdm

# ------------------------------------------------------------
# [설정] rag 모듈 경로 추가
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../rag
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # .../app
sys.path.append(PROJECT_ROOT)

from rag.chatbot_pipeline import run_pipeline


# ------------------------------------------------------------
# [기본 데이터셋 경로]
# 1) 발표용 통합셋이 있으면 우선 사용
# 2) 없으면 기존 evaluation_dataset.json 사용
# ------------------------------------------------------------
DEFAULT_MIXED = os.path.join(BASE_DIR, "data", "evaluation_mixed_for_presentation.json")
DEFAULT_LEGACY = os.path.join(BASE_DIR, "data", "evaluation_dataset.json")

def resolve_default_dataset():
    if os.path.exists(DEFAULT_MIXED):
        return DEFAULT_MIXED
    return DEFAULT_LEGACY


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def calculate_metrics(dataset_path: str, k: int = 3, verbose: bool = False):
    if not os.path.exists(dataset_path):
        print(f"❌ 평가 데이터셋이 없습니다: {dataset_path}")
        print("   rag/data/ 아래에 JSON을 넣어주세요.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    if not isinstance(eval_data, list) or len(eval_data) == 0:
        print("❌ 평가 데이터가 비어있거나 형식이 올바르지 않습니다. (list of dict 필요)")
        return

    print(f"🚀 RAG 성능 평가를 시작합니다.")
    print(f"   Dataset: {dataset_path}")
    print(f"   Top-K  : {k}")
    print(f"   Total  : {len(eval_data)}")
    print("-" * 70)

    if verbose:
        print(f"{'Type':<10} | {'Query':<45} | {'Result':<10} | {'Rank'}")
        print("-" * 70)

    total_count = 0
    hit_count = 0
    mrr_sum = 0.0

    # 타입별 통계
    type_stats = {}

    # source_set이 있으면 세트별 통계도
    has_source = any(isinstance(item, dict) and "source_set" in item for item in eval_data)
    source_stats = {}

    start_all = time.perf_counter()

    for item in tqdm(eval_data):
        if not isinstance(item, dict):
            continue

        query = item.get("query", "").strip()
        if not query:
            continue

        q_type = item.get("type", "Complex")
        ground_truth_ids = item.get("ground_truth_ids") or item.get("ground_truth") or []
        ground_truth_ids = set(map(str, ground_truth_ids))

        source_set = item.get("source_set", "Unknown") if has_source else None

        # init stats buckets
        if q_type not in type_stats:
            type_stats[q_type] = {"hit": 0, "total": 0}

        if has_source:
            if source_set not in source_stats:
                source_stats[source_set] = {"hit": 0, "total": 0}

        try:
            results = run_pipeline(query) or []
            recommended_ids = [str(r.get("card_id") or r.get("id")) for r in results[:k]]

            is_hit = False
            rank = 0

            for idx, rec_id in enumerate(recommended_ids):
                if rec_id in ground_truth_ids:
                    is_hit = True
                    rank = idx + 1
                    break

            # update global
            total_count += 1
            type_stats[q_type]["total"] += 1
            if has_source:
                source_stats[source_set]["total"] += 1

            if is_hit:
                hit_count += 1
                mrr_sum += 1.0 / rank
                type_stats[q_type]["hit"] += 1
                if has_source:
                    source_stats[source_set]["hit"] += 1

                if verbose:
                    print(f"{q_type:<10} | {query[:43]:<45} | ✅ HIT     | {rank}")
            else:
                if verbose:
                    print(f"{q_type:<10} | {query[:43]:<45} | ❌ MISS    | -")

        except Exception as e:
            # 에러도 total로 포함할지 여부는 정책에 따라 다르지만
            # 지금은 "평가 실패"로만 로그 남기고 스킵
            if verbose:
                print(f"{q_type:<10} | {query[:43]:<45} | ⚠️ ERROR  | -")
            print(f"Error processing query '{query}': {e}")

    elapsed_all = time.perf_counter() - start_all

    # --- Final Report ---
    hit_rate = (hit_count / total_count) * 100 if total_count > 0 else 0.0
    mrr = (mrr_sum / total_count) if total_count > 0 else 0.0

    print("\n" + "=" * 50)
    print("📊 [Final Evaluation Report]")
    print("=" * 50)
    print(f"🎯 Overall Hit Rate @ {k} : {hit_rate:.2f}% ({hit_count}/{total_count})")
    print(f"🥇 Overall MRR            : {mrr:.4f}")
    print(f"⏱️ Total Time             : {elapsed_all:.2f}s")
    print("-" * 50)

    # 타입별 상세 결과
    for t, stat in type_stats.items():
        t_hit_rate = (stat["hit"] / stat["total"] * 100) if stat["total"] > 0 else 0.0
        print(f"🔹 {t:<10} Hit Rate    : {t_hit_rate:.2f}% ({stat['hit']}/{stat['total']})")

    # source_set이 있으면 세트별도 출력
    if has_source:
        print("-" * 50)
        print("📚 [By Source Set]")
        for s, stat in source_stats.items():
            s_hit_rate = (stat["hit"] / stat["total"] * 100) if stat["total"] > 0 else 0.0
            print(f"🔸 {s:<18} Hit Rate : {s_hit_rate:.2f}% ({stat['hit']}/{stat['total']})")

    print("=" * 50)

    # 발표 자료용 텍스트
    print("\n[📢 발표 자료용 요약 멘트]")
    if total_count == 0:
        print("\"평가 데이터가 없어 지표를 계산할 수 없습니다.\"")
        return

    avg_rank = (1 / mrr) if mrr > 0 else float("inf")
    avg_rank_text = f"{avg_rank:.1f}" if mrr > 0 else "N/A"

    print(
        f"\"총 {total_count}개의 테스트 쿼리(단순/복합 혼합)에 대해 평가를 진행한 결과,\n"
        f"상위 {k}개 추천 내 정답 포함 비율인 Hit Rate는 {hit_rate:.1f}%를 기록했으며,\n"
        f"평균적으로 정답 카드가 {avg_rank_text}번째 순위에 노출되는 {mrr:.2f}의 MRR 점수를 달성했습니다.\""
    )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG performance for card recommendation.")
    parser.add_argument("-k", type=int, default=3, help="Top-K for Hit Rate and MRR.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=resolve_default_dataset(),
        help="Path to evaluation dataset JSON."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query results."
    )

    args = parser.parse_args()
    calculate_metrics(dataset_path=args.dataset, k=args.k, verbose=args.verbose)


if __name__ == "__main__":
    main()

# 발표용 통합셋을 이미 rag/data/에 넣었다면:

# cd /app/rag
# python evaluate_performance.py


# 특정 파일로 돌리고 싶으면:

# python evaluate_performance.py --dataset data/evaluation_humanstyle_complex.json
# python evaluate_performance.py --dataset data/evaluation_synonym_variations.json
# python evaluate_performance.py --dataset data/evaluation_hard_negative.json


# 로그까지 보고 싶으면:

# python evaluate_performance.py --verbose