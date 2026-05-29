"""
Merge two scores JSON files (e.g. novel_scores.json into novel_scores_processed.json).
Used when you expirement with 2 corporas to get the scores, then add 2 more corporas and
don't want to run Evaluation.generation_eval again for the first 2 corporas.

Both files share the same schema:
  { "<category>": { "average_scores": {...}, "detailed": [ {..."metrics": {...}} ] } }

The script combines the detailed entries across both files (no deduplication by id),
then recomputes average_scores from the merged set.

Usage:
  python benchmarks/expirements/merge_phases_scores.py --source <source.json> --target <target.json> [--out merged.json]

  --source  : file whose entries are merged INTO target
  --target  : base file (updated in place unless --out is given)
  --out     : write result here instead of overwriting target
"""

import argparse
import json
import pathlib
from collections import defaultdict


def recompute_averages(detailed: list[dict]) -> dict:
    if not detailed:
        return {}
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for entry in detailed:
        for metric, value in entry.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                totals[metric] += value
                counts[metric] += 1
    return {m: totals[m] / counts[m] for m in totals}


def merge_scores(source: dict, target: dict) -> dict:
    result = {}
    all_cats = set(target) | set(source)
    for cat in all_cats:
        t_cat = target.get(cat, {"average_scores": {}, "detailed": []})
        s_cat = source.get(cat, {"average_scores": {}, "detailed": []})

        t_ids = {e["id"] for e in t_cat["detailed"] if "id" in e}
        new_entries = [e for e in s_cat["detailed"] if e.get("id") not in t_ids]
        merged_detailed = t_cat["detailed"] + new_entries

        result[cat] = {
            "average_scores": recompute_averages(merged_detailed),
            "detailed": merged_detailed,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two scores JSON files")
    parser.add_argument("--source", help="file whose entries are merged into target")
    parser.add_argument("--target", help="base file (modified in place unless --out given)")
    parser.add_argument("--out", help="write result here instead of overwriting target")
    args = parser.parse_args()

    source = json.loads(pathlib.Path(args.source).read_text())
    target = json.loads(pathlib.Path(args.target).read_text())

    merged = merge_scores(source, target)

    out_path = pathlib.Path(args.out) if args.out else pathlib.Path(args.target)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")

    for cat in merged:
        t_n = len(target.get(cat, {}).get("detailed", []))
        s_n = len(source.get(cat, {}).get("detailed", []))
        m_n = len(merged[cat]["detailed"])
        added = m_n - t_n
        print(f"{cat}: {t_n} + {added} new (of {s_n}) = {m_n} total")
    print(f"\nWrote → {out_path}")
