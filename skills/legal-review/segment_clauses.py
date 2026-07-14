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

# CJK contracts mark sections with enumerators rather than the English
# ARTICLE/SECTION keywords or ALL-CAPS titles, and often omit the space the
# ASCII outline regex above requires. Examples:
#   一、  十二．  （三）  第一条  第4章  1.评估目的  2.1、范围
_CJK_SECTION_LEAD = re.compile(
    r"^\s*[一二三四五六七八九十百千]+\s*[、.．)）]"            # 一、  十二．  三)
    r"|^\s*[（(]\s*[一二三四五六七八九十\d]+\s*[)）]"          # （一）  (3)
    r"|^\s*第\s*[一二三四五六七八九十百千\d]+\s*[条章节部编款项]"  # 第一条  第4章
    r"|^\s*\d+(?:[.．]\d+)*[.．)、]\s*[一-鿿]",         # 1.评估目的  2.1、范围
)

# Han ideographs (incl. extension A and compatibility blocks).
_CJK_CHAR = re.compile(r"[㐀-䶿一-鿿豈-﫿]")

# Sentence-final / list punctuation that disqualifies a CJK line from being a
# bare section title (prose or an inline enumeration, not a heading).
_CJK_NONTITLE_END = ("。", "！", "？", "；", "，", "、", ".", ",", ";")


def _is_heading(para: Paragraph) -> bool:
    """True when *para* starts a new clause (heading style, section number, or bare title)."""
    text = (para.text or "").strip()
    if not text:
        return False
    try:
        style = (para.style.name or "").lower()
    except Exception:
        style = ""
    if style.startswith("heading") or style == "title":
        return True
    if _SECTION_LEAD.match(text) or _CJK_SECTION_LEAD.match(text):
        return True
    # Short ALL-CAPS line with no sentence punctuation — a bare section title.
    # Exclude "label：value" fields, whose CJK label leaves an incidental
    # uppercase code (e.g. "文档编号：AZ-1") passing isupper().
    if (
        text.isupper()
        and len(text) <= 60
        and "：" not in text
        and not text.endswith((".", ";", ","))
    ):
        return True
    # CJK bare title (no case to test): a short Han line that is neither a
    # "label：value" field nor sentence-like prose. Full-width colon U+FF1A
    # marks field labels ("姓名：陈福根") and lead-ins, so it disqualifies.
    if (
        _CJK_CHAR.search(text)
        and len(text) <= 30
        and "：" not in text
        and ":" not in text
        and not text.endswith(_CJK_NONTITLE_END)
    ):
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
