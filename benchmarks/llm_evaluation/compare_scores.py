#!/usr/bin/env python3
"""Compare llm_score (from score_answers.py) vs answer_correctness (from benchmark eval).

Writes items where the two scores differ by more than --threshold to an output file.

Usage:
  python benchmarks/llm_evaluation/compare_scores.py \\
      --llm-scores  benchmarks/results/bench_app_simple/novel_llm_scores.json \\
      --bench-scores GraphRAG-Benchmark/expirements/bench_app_simple/novel_scores.json \\
      --output      benchmarks/results/bench_app_simple/novel_scores_diff.json \\
      [--threshold 0.3]

Output fields per item:
  id, question, source, question_type, ground_truth, generated_answer,
  llm_score, llm_explanation, answer_correctness
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_llm_scored(path: Path) -> dict[str, dict]:
    """Load flat array from score_answers.py output; key by id."""
    with open(path) as f:
        items: list[dict] = json.load(f)
    return {item["id"]: item for item in items}


def load_bench_scored(path: Path) -> dict[str, dict]:
    """Load grouped benchmark scores file (e.g. novel_scores.json); flatten to dict keyed by id."""
    with open(path) as f:
        data: dict = json.load(f)

    by_id: dict[str, dict] = {}
    for question_type, group in data.items():
        for item in group.get("detailed", []):
            item_id = item["id"]
            metrics = item.get("metrics", {})
            by_id[item_id] = {
                "id": item_id,
                "question_type": question_type,
                "answer_correctness": metrics.get("answer_correctness"),
            }
    return by_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--llm-scores",
        required=True,
        help="Flat JSON from score_answers.py (has llm_score, llm_explanation)",
    )
    parser.add_argument(
        "--bench-scores",
        required=True,
        help="Grouped benchmark JSON (has metrics.answer_correctness per item)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for differing items",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Minimum absolute score difference to include (default: 0.3)",
    )
    args = parser.parse_args()

    llm_by_id = load_llm_scored(Path(args.llm_scores))
    bench_by_id = load_bench_scored(Path(args.bench_scores))

    common_ids = set(llm_by_id) & set(bench_by_id)
    only_llm = set(llm_by_id) - set(bench_by_id)
    only_bench = set(bench_by_id) - set(llm_by_id)

    if only_llm:
        print(f"Warning: {len(only_llm)} items only in llm-scores file (no bench score)")
    if only_bench:
        print(f"Warning: {len(only_bench)} items only in bench-scores file (no llm score)")

    diffs: list[dict] = []
    for item_id in common_ids:
        llm_item = llm_by_id[item_id]
        bench_item = bench_by_id[item_id]

        llm_score = llm_item.get("llm_score")
        answer_correctness = bench_item.get("answer_correctness")

        if llm_score is None or answer_correctness is None:
            continue

        if abs(llm_score - answer_correctness) >= args.threshold:
            diffs.append(
                {
                    "id": item_id,
                    "question": llm_item.get("question", ""),
                    "source": llm_item.get("source", ""),
                    "question_type": llm_item.get("question_type") or bench_item.get("question_type", ""),
                    "ground_truth": llm_item.get("ground_truth", ""),
                    "generated_answer": llm_item.get("generated_answer", ""),
                    "llm_score": llm_score,
                    "llm_explanation": llm_item.get("llm_explanation", ""),
                    "answer_correctness": answer_correctness,
                    "diff": round(llm_score - answer_correctness, 4),
                }
            )

    # Sort by absolute diff descending
    diffs.sort(key=lambda x: abs(x["diff"]), reverse=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(diffs, f, indent=2)

    if not diffs:
        print(f"No items differ by more than {args.threshold}. Output file is empty array.")
        sys.exit(0)

    llm_higher = sum(1 for d in diffs if d["diff"] > 0)
    bench_higher = len(diffs) - llm_higher
    print(f"Found {len(diffs)} items differing by >= {args.threshold} (of {len(common_ids)} common)")
    print(f"  llm_score higher: {llm_higher}  |  answer_correctness higher: {bench_higher}")
    print(f"Wrote to {output_path}")


if __name__ == "__main__":
    main()
