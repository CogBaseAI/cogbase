"""
Merge benchmarks results across corporas.
Used when you expirement with 2 corporas to get the results, then add 2 more corporas and
don't want to run Evaluation.generation_eval again for the first 2 corporas.

Usage:
  python benchmarks/expirements/merge_phases_results.py --dir <benchmark_results_dir> --existing <existing_all.json>

  --dir       : the benchmark results dir for all corporas
  --existing  : the benchmark results of the already evaluated corporas
"""


import argparse
import json, glob, pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge benchmarks results across corporas")
    parser.add_argument("--dir", required=True, help="example: ./benchmarks/results/bench_app_simple/novel")
    parser.add_argument("--existing", help="path to an existing merged file; source files already represented in it are skipped")
    args = parser.parse_args()

    prefix = args.dir.rstrip("/")
    files = glob.glob(f"{prefix}/*/predictions_*.json")

    seen_sources: set[str] = set()
    if args.existing:
        existing_path = pathlib.Path(args.existing)
        if existing_path.exists():
            for rec in json.loads(existing_path.read_text()):
                src = rec.get("source")
                if src:
                    seen_sources.add(src)

    merged = []
    skipped = 0
    for f in files:
        records = json.load(open(f))
        if seen_sources and records and records[0].get("source") in seen_sources:
            skipped += 1
            continue
        merged.extend(records)

    pathlib.Path(f"{prefix}_all.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False)
    )

    if skipped:
        print(f"Skipped {skipped} already-seen corpora")
    print(f"Merged {len(files) - skipped} corpora → {len(merged)} questions total")
