#!/usr/bin/env python3
import json
import sys


# example benchmarks/results/bench_app_simple/sample_gen_scores.json
path = sys.argv[1]

with open(path) as f:
    scores = json.load(f)

category_scores = []
print("Results:")
for category, v in scores.items():
    formatted = {k: round(val, 4) if isinstance(val, float) else val for k, val in v['average_scores'].items()}
    print(f"  {category}:  {json.dumps(formatted)}")
    score = v['average_scores']['answer_correctness']
    category_scores.append(score)

print(f"\nAverage Answer Correctness: {sum(category_scores) / len(category_scores):.4f}")
