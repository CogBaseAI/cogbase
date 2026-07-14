#!/usr/bin/env python3
"""Finalize a contract review and bridge it to the edit-docx apply helper.

Three deterministic steps sit between the agent's per-clause *analysis* and a
redlined ``.docx``:

  finalize     raw analysis (+ the clauses.json from segment_clauses.py)  ->  review ops.json
  patch        review ops.json + verdict / suggestion changes             ->  updated review ops.json
  to-edit-ops  review ops.json                                            ->  edit-docx operations JSON

**finalize** resolves each suggestion's ``para_id`` back to that paragraph's
*verbatim* text and bakes it in as ``anchor_text`` — so the review file is
self-contained (apply never needs the segmentation again) and every anchor is
guaranteed to locate its paragraph in the base. It defaults every clause's
``verdict`` to ``pending`` and validates that referenced ``para_id``s exist.

**patch** applies the user's accept / reject / refine decisions to an existing
review file deterministically, so the agent never hand-edits the JSON: set a
clause's ``verdict`` (``--accept`` / ``--reject`` / ``--pending``), and reword or
drop a suggestion via a small ``--patch`` file. It validates every referenced
``clause_id``, verdict value, and resulting op — a bad reference fails loudly here
rather than silently corrupting the working state.

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
_VERDICTS = {"accepted", "rejected", "pending"}


def _validate_op(clause_id: str, suggestion: dict) -> None:
    """Raise ValueError unless *suggestion* is a well-formed edit op with its required fields."""
    op = suggestion.get("op")
    if op not in _ALL_OPS:
        raise ValueError(f"clause {clause_id}: unknown op {op!r}")
    if op in _OPS_NEEDING_ANCHOR and not suggestion.get("anchor_text"):
        raise ValueError(f"clause {clause_id}: op {op!r} requires anchor_text")
    if op in _OPS_NEEDING_TEXT and not suggestion.get("new_text"):
        raise ValueError(f"clause {clause_id}: op {op!r} requires new_text")


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


def patch(
    review: dict,
    verdicts: dict[str, str] | None = None,
    suggestions: dict[str, dict | None] | None = None,
) -> dict:
    """Apply verdict and suggestion changes to a review file in place.

    *verdicts* maps ``clause_id`` -> one of ``accepted`` / ``rejected`` /
    ``pending``. *suggestions* maps ``clause_id`` -> either a dict of fields
    merged into that clause's existing suggestion (e.g. ``{"new_text": "..."}``
    to soften wording, keeping the baked ``anchor_text``) or ``None`` to drop the
    suggestion entirely. Every ``clause_id`` must exist; a merged suggestion must
    remain a well-formed op. Raises ValueError on any bad reference or value.
    """
    by_id = {c["clause_id"]: c for c in review.get("clauses", [])}

    for clause_id, verdict in (verdicts or {}).items():
        if clause_id not in by_id:
            raise ValueError(f"patch: unknown clause_id {clause_id!r}")
        if verdict not in _VERDICTS:
            raise ValueError(
                f"patch: verdict for {clause_id} must be one of {sorted(_VERDICTS)}, got {verdict!r}"
            )
        by_id[clause_id]["verdict"] = verdict

    for clause_id, change in (suggestions or {}).items():
        if clause_id not in by_id:
            raise ValueError(f"patch: unknown clause_id {clause_id!r}")
        clause = by_id[clause_id]
        if change is None:
            clause["suggestion"] = None
            continue
        merged = {**(clause.get("suggestion") or {}), **change}
        _validate_op(clause_id, merged)
        clause["suggestion"] = merged

    return review


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

    p_patch = sub.add_parser(
        "patch", help="apply accept/reject/refine changes to a review ops.json"
    )
    p_patch.add_argument("--review", required=True, help="review ops.json to patch")
    p_patch.add_argument(
        "--accept", nargs="*", default=[], metavar="CLAUSE_ID",
        help="clause_ids to mark accepted",
    )
    p_patch.add_argument(
        "--reject", nargs="*", default=[], metavar="CLAUSE_ID",
        help="clause_ids to mark rejected",
    )
    p_patch.add_argument(
        "--pending", nargs="*", default=[], metavar="CLAUSE_ID",
        help="clause_ids to reset to pending",
    )
    p_patch.add_argument(
        "--patch",
        dest="patch_file",
        help='JSON file with {"verdicts": {...}, "suggestions": {...}} for '
        "reword/drop changes the flags can't express",
    )
    p_patch.add_argument("--output", help="write updated review ops.json (default: stdout)")

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
    elif args.cmd == "patch":
        review = _load(args.review)
        patch_data = _load(args.patch_file) if args.patch_file else {}
        verdicts = dict(patch_data.get("verdicts", {}))
        for clause_id in args.accept:
            verdicts[clause_id] = "accepted"
        for clause_id in args.reject:
            verdicts[clause_id] = "rejected"
        for clause_id in args.pending:
            verdicts[clause_id] = "pending"
        review = patch(review, verdicts=verdicts, suggestions=patch_data.get("suggestions"))
        _write(review, args.output)
    elif args.cmd == "to-edit-ops":
        ops = to_edit_ops(_load(args.review), accepted_only=not args.all)
        _write(ops, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
