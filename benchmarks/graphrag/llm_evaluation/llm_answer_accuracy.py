#!/usr/bin/env python3
"""Score generated answers against ground truth using an LLM judge.

Modes:
  score  — read input JSON, send each item to LLM, write scored output JSON
  stats  — read scored JSON, print average score per question_type

Usage:
  python benchmarks/graphrag/llm_evaluation/llm_answer_accuracy.py score \
      --input  benchmarks/graphrag/results/bench_app_simple_5novels_gpt54mini/novel_all.json \
      --output benchmarks/graphrag/results/bench_app_simple_5novels_gpt54mini/novel_llm_scores.json \
      [--model gpt-5.4-mini] [--concurrency 8]

  python benchmarks/graphrag/llm_evaluation/llm_answer_accuracy.py stats \
      --input benchmarks/graphrag/results/bench_app_simple_5novels_gpt54mini/novel_llm_scores.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import openai


SCORE_PROMPT = """\
You are an answer correctness judge. Given a question, a ground-truth answer, \
and a generated answer, score the generated answer on a scale from 0.0 to 1.0 \
based on factual correctness relative to the ground truth.

Scoring guide:
  1.0 — fully correct, same facts as ground truth
  0.7 — mostly correct with minor omissions or extra detail
  0.5 — partially correct, captures some key facts but misses others
  0.3 — slightly related but largely incorrect or incomplete
  0.0 — wrong or irrelevant

Respond with JSON only, no extra text:
{{"score": <float 0.0–1.0>, "explanation": "<one sentence>"}}

Question: {question}
Ground truth: {ground_truth}
Generated answer: {generated_answer}"""


async def judge_item(
    client: openai.AsyncOpenAI,
    model: str,
    item: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    prompt = SCORE_PROMPT.format(
        question=item["question"],
        ground_truth=item["ground_truth"],
        generated_answer=item.get("generated_answer", ""),
    )
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            service_tier="flex",
        )
    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    score = float(parsed.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    explanation = str(parsed.get("explanation", ""))
    return {**item, "llm_score": score, "llm_explanation": explanation}


async def run_score(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path) as f:
        items: list[dict] = json.load(f)

    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [judge_item(client, args.model, item, semaphore) for item in items]

    scored: list[dict] = []
    total = len(tasks)
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        scored.append(result)
        print(f"\r  scored {i}/{total}", end="", flush=True)

    print()  # newline after progress

    # Preserve original order
    id_to_scored = {r["id"]: r for r in scored}
    ordered = [id_to_scored.get(item["id"], item) for item in items]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"Wrote {len(ordered)} scored items to {output_path}")


def run_stats(args: argparse.Namespace) -> None:
    with open(args.input) as f:
        items: list[dict] = json.load(f)

    by_type: dict[str, list[float]] = {}
    missing = 0
    for item in items:
        if "llm_score" not in item:
            missing += 1
            continue
        qt = item.get("question_type", "Unknown")
        by_type.setdefault(qt, []).append(float(item["llm_score"]))

    if missing:
        print(f"Warning: {missing} items have no llm_score (run 'score' mode first)")

    if not by_type:
        print("No scored items found.")
        sys.exit(1)

    all_scores: list[float] = []
    all_type_avg_scores: list[float] = []
    print(f"\n{'Question type':<35} {'avg score':>9}  {'n':>5}")
    print("-" * 55)
    for qt in sorted(by_type):
        scores = by_type[qt]
        avg = sum(scores) / len(scores)
        all_scores.extend(scores)
        all_type_avg_scores.append(avg)
        print(f"  {qt:<33} {avg:>9.4f}  {len(scores):>5}")

    overall = sum(all_scores) / len(all_scores)
    overtype = sum(all_type_avg_scores) / len(all_type_avg_scores)
    print("-" * 55)
    print(f"  {'Overall':<33} {overall:>9.4f}  {len(all_scores):>5}")
    print(f"  {'Overtype':<33} {overtype:>9.4f}  {len(all_type_avg_scores):>5}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    score_p = sub.add_parser("score", help="Judge answers with an LLM and write scored JSON")
    score_p.add_argument("--input", required=True, help="Path to input JSON file")
    score_p.add_argument("--output", required=True, help="Path for output scored JSON file")
    score_p.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model to use (default: gpt-5.4-mini)")
    score_p.add_argument("--concurrency", type=int, default=8, help="Max concurrent LLM calls (default: 8)")

    stats_p = sub.add_parser("stats", help="Print average score per question_type from a scored JSON file")
    stats_p.add_argument("--input", required=True, help="Path to scored JSON file")

    args = parser.parse_args()

    if args.mode == "score":
        asyncio.run(run_score(args))
    else:
        run_stats(args)


if __name__ == "__main__":
    main()
