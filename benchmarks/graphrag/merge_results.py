import argparse
import json, glob, pathlib
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge benchmarks results across corporas")
    parser.add_argument("--dir", required=True, help="example: ./benchmarks/results/bench_app_simple/novel")
    args = parser.parse_args()

    prefix = args.dir.rstrip("/")
    files = glob.glob(f"{prefix}/*/predictions_*.json")

    merged = []
    for f in files:
        records = json.load(open(f))
        merged.extend(records)
        pathlib.Path(f"{prefix}_all.json").write_text(
            json.dumps(merged, indent=2, ensure_ascii=False)
        )

    print(f"Merged {len(files)} corpora → {len(merged)} questions total")
