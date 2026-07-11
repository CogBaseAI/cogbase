#!/usr/bin/env python3
"""Finalize a contract review and bridge it to the edit-docx apply helper.

Two deterministic steps sit between the agent's per-clause *analysis* and a
redlined ``.docx``:

  finalize     raw analysis (+ the clauses.json from segment_clauses.py)  ->  review ops.json
  to-edit-ops  review ops.json                                            ->  edit-docx operations JSON

**finalize** resolves each suggestion's ``para_id`` back to that paragraph's
*verbatim* text and bakes it in as ``anchor_text`` — so the review file is
self-contained (apply never needs the segmentation again) and every anchor is
guaranteed to locate its paragraph in the base. It defaults every clause's
``verdict`` to ``pending`` and validates that referenced ``para_id``s exist.

**to-edit-ops** filters the review to the suggestions that should be applied
(``verdict == "accepted"`` by default, or all of them with ``--all`` for a
preview redline) and emits the ``{"operations": [...]}`` shape that
``edit-docx/apply_operations.py`` consumes. edit-docx ignores the extra review
fields, but we strip to the bare op to keep its input clean.

Review ops.json shape::

    {"base_doc_id": "...",
     "meta": {"parties": [...], "representative_party": "...",
              "review_position": "dominant|neutral|disadvantaged",
              "contract_type": "...", "governing_law": "..."},
     "clauses": [
        {"clause_id": "c1", "heading": "1. PAYMENT TERMS",
         "risk": {"level": "high|medium|low|none", "rationale": "..."},
         "contradicts": ["c7"],
         "suggestion": {"op": "replace", "anchor_text": "...", "new_text": "..."},
         "verdict": "pending|accepted|rejected"}
     ]}

A clause with no suggested change carries ``"suggestion": null``.
"""

from __future__ import annotations

import argparse
import json
import sys

_OPS_NEEDING_ANCHOR = {"replace", "delete", "insert_after"}
_OPS_NEEDING_TEXT = {"replace", "insert_after", "append"}
_ALL_OPS = _OPS_NEEDING_ANCHOR | _OPS_NEEDING_TEXT
_REVIEW_POSITIONS = {"dominant", "neutral", "disadvantaged"}


def _anchor_index(clauses: list[dict]) -> dict[str, str]:
    """Map every para_id from the segmentation to its verbatim paragraph text."""
    index: dict[str, str] = {}
    for clause in clauses:
        for para in clause.get("paragraphs", []):
            index[para["para_id"]] = para["text"]
    return index


def _heading_index(clauses: list[dict]) -> dict[str, str]:
    return {c["clause_id"]: c.get("heading", "") for c in clauses}


def finalize(analysis: dict, clauses: list[dict]) -> dict:
    """Resolve para_id anchors and default verdicts; raise ValueError on a bad reference."""
    anchors = _anchor_index(clauses)
    headings = _heading_index(clauses)

    meta = analysis.get("meta", {})
    position = meta.get("review_position")
    if position is not None and position not in _REVIEW_POSITIONS:
        raise ValueError(
            f"review_position must be one of {sorted(_REVIEW_POSITIONS)}, got {position!r}"
        )

    out_clauses: list[dict] = []
    for item in analysis.get("analyses", []):
        clause_id = item.get("clause_id")
        suggestion = item.get("suggestion")
        resolved = None
        if suggestion:
            op = suggestion.get("op")
            if op not in _ALL_OPS:
                raise ValueError(f"clause {clause_id}: unknown op {op!r}")
            resolved = {"op": op}
            if op in _OPS_NEEDING_ANCHOR:
                para_id = suggestion.get("para_id")
                if para_id not in anchors:
                    raise ValueError(
                        f"clause {clause_id}: para_id {para_id!r} not found in segmentation"
                    )
                resolved["anchor_text"] = anchors[para_id]
            if op in _OPS_NEEDING_TEXT:
                new_text = suggestion.get("new_text")
                if not new_text:
                    raise ValueError(f"clause {clause_id}: op {op!r} requires new_text")
                resolved["new_text"] = new_text

        out_clauses.append(
            {
                "clause_id": clause_id,
                "heading": headings.get(clause_id, item.get("heading", "")),
                "risk": item.get("risk", {"level": "none", "rationale": ""}),
                "contradicts": item.get("contradicts", []),
                "suggestion": resolved,
                "verdict": "pending",
            }
        )

    return {
        "base_doc_id": analysis.get("base_doc_id", ""),
        "meta": meta,
        "clauses": out_clauses,
    }


def to_edit_ops(review: dict, accepted_only: bool = True) -> dict:
    """Project a review file to edit-docx operations, filtered by verdict."""
    operations: list[dict] = []
    for clause in review.get("clauses", []):
        suggestion = clause.get("suggestion")
        if not suggestion:
            continue
        if accepted_only and clause.get("verdict") != "accepted":
            continue
        op = {"op": suggestion["op"]}
        if "anchor_text" in suggestion:
            op["anchor_text"] = suggestion["anchor_text"]
        if "new_text" in suggestion:
            op["new_text"] = suggestion["new_text"]
        operations.append(op)
    return {"operations": operations}


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write(payload: dict, output: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fin = sub.add_parser("finalize", help="raw analysis + clauses -> review ops.json")
    p_fin.add_argument("--analysis", required=True, help="raw per-clause analysis JSON")
    p_fin.add_argument("--clauses", required=True, help="clauses.json from segment_clauses.py")
    p_fin.add_argument("--output", help="write review ops.json (default: stdout)")

    p_edit = sub.add_parser("to-edit-ops", help="review ops.json -> edit-docx operations")
    p_edit.add_argument("--review", required=True, help="review ops.json")
    p_edit.add_argument(
        "--all",
        action="store_true",
        help="include every suggestion (preview), not just accepted verdicts",
    )
    p_edit.add_argument("--output", help="write operations JSON (default: stdout)")

    args = parser.parse_args()

    if args.cmd == "finalize":
        review = finalize(_load(args.analysis), _load(args.clauses).get("clauses", []))
        _write(review, args.output)
    elif args.cmd == "to-edit-ops":
        ops = to_edit_ops(_load(args.review), accepted_only=not args.all)
        _write(ops, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
