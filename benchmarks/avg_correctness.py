#!/usr/bin/env python3
import json
import sys


# example benchmarks/results/bench_app_simple/sample_gen_scores.json
path = sys.argv[1]

with open(path) as f:
    scores = json.load(f)

values = [v["answer_correctness"] for v in scores.values() if "answer_correctness" in v]

if not values:
    print("No answer_correctness scores found.")
    sys.exit(1)

for category, v in scores.items():
    if "answer_correctness" in v:
        print(f"  {category}: {v['answer_correctness']:.4f}")

print(f"\nAverage answer_correctness: {sum(values) / len(values):.4f}  (n={len(values)})")
