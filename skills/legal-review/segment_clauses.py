#!/usr/bin/env python3
"""Segment a contract .docx into clauses with verbatim, matchable anchors.

The deterministic *segment* helper for the legal-review skill. The agent does
the *analyze* step (judging risk and drafting suggested changes per clause); this
script does the mechanical part first: walk the base OOXML, group paragraphs into
clauses at section/heading boundaries, and emit each paragraph with a stable id
and its **verbatim** text.

Why this matters: downstream edits are applied by the ``edit-docx`` helper, which
locates a paragraph by an ``anchor_text`` snippet (a tolerant normalized-contains
match — see ``edit-docx/apply_operations.py``). If the agent invented anchor text
freely it might not match the base. By handing the agent paragraph text taken
*directly from the document*, every anchor the agent reuses is guaranteed to
locate its paragraph. The agent references a ``para_id``; ``build_ops.py`` resolves
it back to that paragraph's verbatim text as the anchor.

Usage::

    python segment_clauses.py --original in.docx [--output clauses.json]

Emits JSON to stdout (or ``--output``)::

    {"clauses": [
        {"clause_id": "c1", "heading": "1. PAYMENT TERMS", "paragraphs": [
            {"para_id": "c1.p0", "text": "1. PAYMENT TERMS"},
            {"para_id": "c1.p1", "text": "Payment shall be due within 30 days of invoice."}
        ]}
    ]}

Body paragraphs before the first heading form a preamble clause ``c0``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from docx import Document
from docx.text.paragraph import Paragraph


# A paragraph opens a new clause when it looks like a section boundary:
#   "ARTICLE V", "Section 4", "4.", "4.2", "4.2.1)"  — optionally followed by a title.
_SECTION_LEAD = re.compile(
    r"^\s*(?:article|section)\b"        # ARTICLE / SECTION keyword
    r"|^\s*\d+(?:\.\d+)*[.)]?\s+\S"     # 4  /  4.2  /  4.2.1)  followed by text
    r"|^\s*[IVXLCDM]+[.)]\s+\S",        # roman-numeral outline: "IV. ..."
    re.IGNORECASE,
)


def _is_heading(para: Paragraph) -> bool:
    """True when *para* starts a new clause (heading style, section number, or ALL-CAPS title)."""
    text = (para.text or "").strip()
    if not text:
        return False
    try:
        style = (para.style.name or "").lower()
    except Exception:
        style = ""
    if style.startswith("heading") or style == "title":
        return True
    if _SECTION_LEAD.match(text):
        return True
    # Short ALL-CAPS line with no sentence punctuation — a bare section title.
    if text.isupper() and len(text) <= 60 and not text.endswith((".", ";", ",")):
        return True
    return False


def segment(doc) -> list[dict]:
    """Group the document's non-empty paragraphs into clauses at heading boundaries."""
    clauses: list[dict] = []
    current: dict | None = None

    def _open(heading_text: str) -> dict:
        clause = {"clause_id": f"c{len(clauses)}", "heading": heading_text, "paragraphs": []}
        clauses.append(clause)
        return clause

    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        if current is None or _is_heading(para):
            current = _open(text if _is_heading(para) else "")
        pid = f"{current['clause_id']}.p{len(current['paragraphs'])}"
        current["paragraphs"].append({"para_id": pid, "text": text})

    return clauses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="path to the contract .docx")
    parser.add_argument("--output", help="path to write clauses JSON (default: stdout)")
    args = parser.parse_args()

    doc = Document(args.original)
    clauses = segment(doc)
    payload = json.dumps({"clauses": clauses}, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
