#!/usr/bin/env python3
"""Apply a list of edit operations to a base .docx, as a redline or a clean edit.

The deterministic *apply* helper for the edit-docx skill. The agent does
the *understand* step (reading the change source and deriving the operation list);
this script applies them to the base OOXML in place, matching each operation to a
paragraph by an anchor snippet. Two output modes:

- **tracked** (default) — record each change as Word tracked changes
  (``<w:ins>`` / ``<w:del>``) rather than a clean overwrite, so a reviewer opening
  the result in Word sees every insertion and deletion and can accept or reject each.
- **clean** (``--clean``) — apply each change directly, producing a *final*
  document with the edits baked in and no tracked-change markup. Use this once the
  edits are settled (e.g. after the user has accepted/rejected a redline's
  suggestions) to hand back the final contract.

Either way styles, numbering, and fonts survive.

Usage::

    python apply_operations.py --original in.docx --ops ops.json --output out.docx \
        [--author "Name"] [--clean]

``ops.json`` shape::

    {"operations": [
        {"op": "replace",      "anchor_text": "...", "new_text": "..."},
        {"op": "delete",       "anchor_text": "..."},
        {"op": "insert_after", "anchor_text": "...", "new_text": "..."},
        {"op": "append",                             "new_text": "..."}
    ]}

Prints a JSON report to stdout: per-operation ``matched`` flag plus an
``unmatched`` count, so the caller can surface operations that found no anchor
(a common sign the change referenced text the base doesn't contain).
"""

from __future__ import annotations

import argparse
import copy
import datetime
import itertools
import json
import re
import sys

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph


# Inline markdown emphasis/code markers, and leading block markers (heading,
# ordered/unordered list, blockquote). The text the agent reads is markdown
# (the base is extracted to markdown at ingest), but python-docx paragraph text
# is raw — no `**`, no `3. ` list prefix. Stripping both lets an anchor copied
# verbatim from the markdown match the raw paragraph on the first try.
_MD_INLINE = re.compile(r"[*_`~]")
_MD_LEADING = re.compile(r"^\s*(?:#{1,6}\s+|\d+[.)]\s+|[-*+]\s+|>\s+)")

# Monotonic revision ids. Word requires every <w:ins>/<w:del> to carry a unique
# w:id within the document.
_rev_ids = itertools.count(1)


def _norm(text: str) -> str:
    """Normalize for anchor matching: strip markdown, collapse whitespace, lowercase.

    Tolerates reflow (whitespace), case, and the markdown-vs-raw gap between the
    text the agent reads and the raw paragraph text this script matches against.
    """
    text = _MD_LEADING.sub("", text or "")
    text = _MD_INLINE.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _find_paragraph(doc, anchor_text: str):
    """Return the first paragraph whose normalized text contains the anchor, or None."""
    needle = _norm(anchor_text)
    if not needle:
        return None
    for para in doc.paragraphs:
        if needle in _norm(para.text):
            return para
    return None


def _revision(tag: str, author: str, date: str):
    """Build an empty ``<w:ins>`` or ``<w:del>`` wrapper with a unique id/author/date."""
    el = OxmlElement(tag)
    el.set(qn("w:id"), str(next(_rev_ids)))
    el.set(qn("w:author"), author)
    el.set(qn("w:date"), date)
    return el


def _make_text_run(new_text: str, rpr=None):
    """Build a ``<w:r>`` carrying *new_text*, optionally cloning run properties."""
    run = OxmlElement("w:r")
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = new_text
    run.append(t)
    return run


def _first_run_rpr(para: Paragraph):
    """Return a copy target of the first run's ``<w:rPr>`` (or None), for format reuse."""
    if para.runs:
        return para.runs[0]._element.find(qn("w:rPr"))
    return None


def _mark_run_deleted(run, wrapper) -> None:
    """Move *run* into a ``<w:del>`` wrapper and convert its text to ``<w:delText>``.

    Deleted text lives in ``<w:delText>`` rather than ``<w:t>`` so Word renders it
    as struck-through tracked deletion instead of live text.
    """
    r = run._element
    for t in r.findall(qn("w:t")):
        t.tag = qn("w:delText")
    parent = r.getparent()
    parent.insert(parent.index(r), wrapper)
    wrapper.append(r)


def _mark_para_mark(para: Paragraph, tag: str, author: str, date: str) -> None:
    """Mark the paragraph mark itself as inserted/deleted (``<w:pPr><w:rPr><tag/>``).

    Needed so a whole-paragraph insert or delete tracks cleanly — accepting the
    revision also adds/removes the paragraph break, not just the visible text.
    """
    p = para._p
    pPr = p.find(qn("w:pPr"))
    if pPr is None:
        pPr = OxmlElement("w:pPr")
        p.insert(0, pPr)
    rPr = pPr.find(qn("w:rPr"))
    if rPr is None:
        rPr = OxmlElement("w:rPr")
        pPr.append(rPr)
    mark = OxmlElement(tag)
    mark.set(qn("w:id"), str(next(_rev_ids)))
    mark.set(qn("w:author"), author)
    mark.set(qn("w:date"), date)
    rPr.insert(0, mark)


def _replace_tracked(para: Paragraph, new_text: str, author: str, date: str) -> None:
    """Redline a paragraph: strike its runs as deleted, insert *new_text* after them."""
    rpr = _first_run_rpr(para)
    for run in list(para.runs):
        _mark_run_deleted(run, _revision("w:del", author, date))
    ins = _revision("w:ins", author, date)
    ins.append(_make_text_run(new_text, rpr))
    para._p.append(ins)


def _delete_tracked(para: Paragraph, author: str, date: str) -> None:
    """Redline a whole-paragraph deletion: strike every run and the paragraph mark."""
    for run in list(para.runs):
        _mark_run_deleted(run, _revision("w:del", author, date))
    _mark_para_mark(para, "w:del", author, date)


def _fill_inserted_paragraph(para: Paragraph, new_text: str, author: str, date: str) -> None:
    """Populate an empty new paragraph as a tracked insertion (runs + paragraph mark)."""
    ins = _revision("w:ins", author, date)
    ins.append(_make_text_run(new_text))
    para._p.append(ins)
    _mark_para_mark(para, "w:ins", author, date)


def _insert_after_tracked(para: Paragraph, new_text: str, author: str, date: str) -> Paragraph:
    """Insert a new tracked-insertion paragraph immediately after *para*."""
    new_p = OxmlElement("w:p")
    para._p.addnext(new_p)
    new_para = Paragraph(new_p, para._parent)
    try:
        new_para.style = para.style
    except Exception:
        pass  # style may not be assignable on some documents; leave default
    _fill_inserted_paragraph(new_para, new_text, author, date)
    return new_para


def _replace_clean(para: Paragraph, new_text: str) -> None:
    """Overwrite a paragraph's text directly, reusing the first run's formatting."""
    rpr = _first_run_rpr(para)
    for run in list(para.runs):
        run._element.getparent().remove(run._element)
    para._p.append(_make_text_run(new_text, rpr))


def _delete_clean(para: Paragraph) -> None:
    """Remove a whole paragraph outright."""
    para._p.getparent().remove(para._p)


def _insert_after_clean(para: Paragraph, new_text: str) -> Paragraph:
    """Insert a new plain paragraph immediately after *para*."""
    new_p = OxmlElement("w:p")
    para._p.addnext(new_p)
    new_para = Paragraph(new_p, para._parent)
    try:
        new_para.style = para.style
    except Exception:
        pass  # style may not be assignable on some documents; leave default
    new_para._p.append(_make_text_run(new_text))
    return new_para


def apply_operations(
    doc, operations: list[dict], author: str, date: str, tracked: bool = True
) -> list[dict]:
    """Apply operations in order; return a per-op match report.

    With *tracked* (the default) each change is recorded as Word tracked changes;
    otherwise the edits are applied cleanly, producing a final document with no
    tracked-change markup.
    """
    report: list[dict] = []
    for op in operations:
        kind = op.get("op")
        anchor = op.get("anchor_text", "")
        new_text = op.get("new_text", "")
        matched = True

        if kind == "append":
            if tracked:
                _fill_inserted_paragraph(doc.add_paragraph(), new_text, author, date)
            else:
                doc.add_paragraph(new_text)
        else:
            para = _find_paragraph(doc, anchor)
            if para is None:
                matched = False
            elif kind == "replace":
                _replace_tracked(para, new_text, author, date) if tracked else _replace_clean(para, new_text)
            elif kind == "insert_after":
                _insert_after_tracked(para, new_text, author, date) if tracked else _insert_after_clean(para, new_text)
            elif kind == "delete":
                _delete_tracked(para, author, date) if tracked else _delete_clean(para)
            else:
                matched = False  # unknown op type

        report.append({"op": kind, "anchor_text": anchor, "matched": matched})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="path to the original .docx")
    parser.add_argument("--ops", required=True, help="path to the operations JSON file")
    parser.add_argument("--output", required=True, help="path to write the edited .docx")
    parser.add_argument(
        "--author",
        default="edit-docx",
        help="author name recorded on each tracked change (default: edit-docx)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="apply edits directly (final document) instead of as tracked changes",
    )
    args = parser.parse_args()

    with open(args.ops, encoding="utf-8") as f:
        operations = json.load(f).get("operations", [])

    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    doc = Document(args.original)
    report = apply_operations(doc, operations, args.author, date, tracked=not args.clean)
    doc.save(args.output)

    unmatched = sum(1 for r in report if not r["matched"])
    json.dump({"operations": report, "unmatched": unmatched}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
