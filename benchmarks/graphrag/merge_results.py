import argparse
import json, glob, pathlib

# Mirror run_cogbase.py SUBSET_FILES so --corpora selects the same set of corpora.
SUBSET_FILES = {
    "novel": "Corpus/novel.json",
    "medical": "Corpus/medical.json",
}


def _corpus_order(dataset_dir: str, subset: str) -> list[str]:
    """Return corpus_names in the dataset file's order (matches run_cogbase.py)."""
    corpus_path = pathlib.Path(dataset_dir) / SUBSET_FILES[subset]
    raw = json.load(open(corpus_path))
    corpora = [raw] if isinstance(raw, dict) else raw
    return [c["corpus_name"] for c in corpora]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge benchmarks results across corporas")
    parser.add_argument("--dir", required=True, help="example: ./benchmarks/results/bench_app_simple/novel")
    parser.add_argument("--corpora", type=int, default=None,
                        help="Merge only the first N corpora, selected the same way as "
                             "run_cogbase.py --corpora (dataset corpus-file order). Requires "
                             "--dataset_dir and --subset.")
    parser.add_argument("--dataset_dir", default="./GraphRAG-Benchmark/Datasets",
                        help="Path to the Datasets directory (used with --corpora to match order)")
    parser.add_argument("--subset", choices=list(SUBSET_FILES), default=None,
                        help="Subset name (used with --corpora to match order)")
    args = parser.parse_args()

    prefix = args.dir.rstrip("/")

    if args.corpora is not None:
        if args.subset is None:
            parser.error("--corpora requires --subset (and --dataset_dir) to match run_cogbase.py selection")
        # Same selection as run_cogbase.py: first N corpora in dataset-file order.
        names = _corpus_order(args.dataset_dir, args.subset)[: args.corpora]
        files = [f"{prefix}/{name}/predictions_{name}.json" for name in names]
        files = [f for f in files if pathlib.Path(f).exists()]
    else:
        files = sorted(glob.glob(f"{prefix}/*/predictions_*.json"))

    merged = []
    for f in files:
        records = json.load(open(f))
        merged.extend(records)

    pathlib.Path(f"{prefix}_all.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False)
    )

    print(f"Merged {len(files)} corpora → {len(merged)} questions total")
