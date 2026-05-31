"""Score CogBase predictions on the LoCoMo QA benchmark.

Computes token-level F1 (with Porter stemming) for all five question categories
and recall@retrieved-sessions for questions that have evidence annotations.

Usage:
    python benchmarks-locomo/token_f1_score.py \\
        --data_file locomo/data/locomo10.json \\
        --pred_file benchmarks/locomo/results/locomo10_cogbase.json

Categories:
    1  Multi-hop      — partial F1 across comma-separated sub-answers
    2  Temporal       — token F1 on date strings
    3  Open-domain    — token F1
    4  Single-hop     — token F1
    5  Adversarial    — 1 if model says "not mentioned" / "no information", else 0
"""

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

from nltk.stem import PorterStemmer

ps = PorterStemmer()
PREDICTION_KEY = "cogbase_prediction"

CAT_NAMES = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
}


# ---------------------------------------------------------------------------
# Scoring (mirrors locomo/task_eval/evaluation.py logic)
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.lower().split())


def _f1(pred: str, gt: str) -> float:
    pred_tokens = [ps.stem(w) for w in _normalize(pred).split()]
    gt_tokens = [ps.stem(w) for w in _normalize(gt).split()]
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def _f1_multi(pred: str, gt: str) -> float:
    """F1 for multi-hop answers with comma-separated sub-answers."""
    preds = [p.strip() for p in pred.split(",")]
    gts = [g.strip() for g in gt.split(",")]
    return float(
        sum(max(_f1(p, g) for p in preds) for g in gts) / len(gts)
    )


def score_qa(qa: dict) -> float | None:
    """Return the per-QA score, or None if no prediction is present."""
    if PREDICTION_KEY not in qa:
        return None
    pred = qa[PREDICTION_KEY]
    answer = str(qa.get("answer", ""))
    cat = qa["category"]

    if cat == 5:
        lower = pred.strip().lower()
        return 1.0 if ("not mentioned" in lower or "no information" in lower) else 0.0
    elif cat == 1:
        return _f1_multi(pred, answer)
    else:
        return _f1(pred, answer)


def recall_score(qa: dict) -> float | None:
    """Fraction of evidence sessions present in retrieved context. None if unavailable."""
    context_ids = qa.get(PREDICTION_KEY + "_context", [])
    evidence = qa.get("evidence", [])
    if not evidence or not context_ids:
        return None
    # context_ids are "S{N}" strings; evidence are "D{N}:{T}" strings
    if context_ids and context_ids[0].startswith("S"):
        sessions = {e[1:] for e in context_ids}  # "S3" → "3"
        hits = sum(1 for ev in evidence if ev.split(":")[0][1:] in sessions)
        return hits / len(evidence)
    return None


# ---------------------------------------------------------------------------
# Aggregation and display
# ---------------------------------------------------------------------------

def compute_and_print(data_file: Path, pred_file: Path) -> None:
    ann = {d["sample_id"]: d for d in json.loads(data_file.read_text())}
    preds = json.loads(pred_file.read_text())

    cat_total: dict[int, int] = defaultdict(int)
    cat_score: dict[int, float] = defaultdict(float)
    cat_recall: dict[int, list[float]] = defaultdict(list)
    skipped = 0

    for sample in preds:
        sample_id = sample["sample_id"]
        if sample_id not in ann:
            continue
        for qa in sample.get("qa", []):
            s = score_qa(qa)
            if s is None:
                skipped += 1
                continue
            cat = qa["category"]
            cat_total[cat] += 1
            cat_score[cat] += s
            rec = recall_score(qa)
            if rec is not None:
                cat_recall[cat].append(rec)

    if skipped:
        print(f"Note: {skipped} question(s) skipped (no prediction).")

    n_convs = len(preds)
    total_q = sum(cat_total.values())
    print(f"\nCogBase on LoCoMo  ({n_convs} conversations, {total_q} questions)")
    print(f"{'Category':<22} {'N':>6}  {'F1':>6}  {'Recall':>8}")
    print("─" * 50)

    total_n = 0
    total_score = 0.0
    for cat in [4, 1, 2, 3, 5]:
        n = cat_total[cat]
        if n == 0:
            continue
        avg_f1 = cat_score[cat] / n
        recs = cat_recall[cat]
        recall_str = f"{sum(recs)/len(recs):.3f}" if recs else "   n/a"
        print(f"  {CAT_NAMES[cat]:<20} {n:>6}  {avg_f1:.3f}  {recall_str:>8}")
        total_n += n
        total_score += cat_score[cat]

    print("─" * 50)
    overall = total_score / total_n if total_n else 0.0
    print(f"  {'Overall':<20} {total_n:>6}  {overall:.3f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score CogBase predictions on LoCoMo")
    parser.add_argument(
        "--data_file", default="locomo/data/locomo10.json",
        help="Path to locomo10.json (ground truth)",
    )
    parser.add_argument(
        "--pred_file", default="benchmarks-locomo/results/locomo10_cogbase.json",
        help="Path to predictions file written by run_cogbase.py",
    )
    args = parser.parse_args()
    compute_and_print(Path(args.data_file), Path(args.pred_file))
