"""Unit tests for the edit-docx skill's apply_operations helper.

The helper is a standalone CLI shipped inside the skill bundle (not an importable
package), so it is loaded by path. Tests cover the paragraph-level apply logic
(replace / delete / insert_after / append), formatting preservation, the unmatched
report, and the CLI entry point end-to-end.

The helper produces a **redline**: edits are recorded as Word tracked changes
(``<w:ins>`` / ``<w:del>``), not clean overwrites. python-docx's ``paragraph.text``
and ``.runs`` do not see runs nested inside those wrappers, so the assertions here
inspect the tracked-change markup directly — inserted text from ``<w:ins>//<w:t>``
and deleted text from ``<w:del>//<w:delText>`` — rather than the visible text.

Skipped entirely when python-docx is not installed (it is a skill-declared
dependency, installed into the skill's venv at load time, not a test dependency).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("docx")

from docx import Document  # noqa: E402  (import after importorskip)
from docx.oxml.ns import qn  # noqa: E402

SKILL_DIR = pathlib.Path(__file__).resolve().parents[2] / "skills" / "edit-docx"
SCRIPT = SKILL_DIR / "apply_operations.py"

AUTHOR = "tester"
DATE = "2026-01-01T00:00:00Z"


def _load_helper():
    spec = importlib.util.spec_from_file_location("apply_operations", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


helper = _load_helper()


def _apply(doc, operations):
    """apply_operations with fixed author/date so tests don't repeat them."""
    return helper.apply_operations(doc, operations, AUTHOR, DATE)


def _doc(paragraphs: list[str]) -> Document:
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    return doc


# --- tracked-change readers -------------------------------------------------
# python-docx can't see runs inside <w:ins>/<w:del>, so read the markup directly.


def _inserted_text(scope) -> str:
    """Concatenated text of every ``<w:ins>//<w:t>`` under *scope* (doc or paragraph)."""
    return "".join(
        t.text or "" for ins in scope.iter(qn("w:ins")) for t in ins.iter(qn("w:t"))
    )


def _deleted_text(scope) -> str:
    """Concatenated text of every ``<w:del>//<w:delText>`` under *scope*."""
    return "".join(
        t.text or "" for d in scope.iter(qn("w:del")) for t in d.iter(qn("w:delText"))
    )


def _el(doc_or_para):
    """The lxml element to iterate: a paragraph's ``<w:p>`` or the document body."""
    return doc_or_para._p if hasattr(doc_or_para, "_p") else doc_or_para.element.body


def _ins_text(scope):
    return _inserted_text(_el(scope))


def _del_text(scope):
    return _deleted_text(_el(scope))


def _para_mark_tag(para) -> str | None:
    """Return 'w:ins'/'w:del' if the paragraph mark itself is tracked, else None."""
    pPr = para._p.find(qn("w:pPr"))
    if pPr is None:
        return None
    rPr = pPr.find(qn("w:rPr"))
    if rPr is None:
        return None
    for tag in ("w:ins", "w:del"):
        if rPr.find(qn(tag)) is not None:
            return tag
    return None


# ---------------------------------------------------------------------------
# apply_operations — per-op behavior (as tracked changes)
# ---------------------------------------------------------------------------


def test_replace_redlines_matched_paragraph():
    doc = _doc(["Payment shall be due within 30 days.", "Other clause."])
    report = _apply(
        doc,
        [{"op": "replace", "anchor_text": "Payment shall be due within 30 days",
          "new_text": "Payment shall be due within 45 days."}],
    )
    para = doc.paragraphs[0]
    # old text struck as a deletion, new text recorded as an insertion — both present
    assert "Payment shall be due within 30 days." in _del_text(para)
    assert "Payment shall be due within 45 days." in _ins_text(para)
    # the second (untouched) paragraph keeps its plain visible text
    assert doc.paragraphs[1].text == "Other clause."
    assert report == [{"op": "replace", "anchor_text": "Payment shall be due within 30 days", "matched": True}]


def test_replace_records_author_and_date():
    doc = _doc(["Payment shall be due within 30 days."])
    _apply(doc, [{"op": "replace", "anchor_text": "Payment shall be due within 30 days", "new_text": "x"}])
    ins = doc.paragraphs[0]._p.find(qn("w:ins"))
    assert ins.get(qn("w:author")) == AUTHOR
    assert ins.get(qn("w:date")) == DATE


def test_replace_preserves_run_formatting():
    doc = Document()
    para = doc.add_paragraph()
    run = para.add_run("Payment due in 30 days")
    run.bold = True

    _apply(doc, [{"op": "replace", "anchor_text": "Payment due in 30 days", "new_text": "Payment due in 45 days"}])

    # the inserted run clones the original run's <w:rPr>, so bold survives the redline
    ins = para._p.find(qn("w:ins"))
    ins_run = ins.find(qn("w:r"))
    assert ins_run.find(qn("w:rPr")).find(qn("w:b")) is not None
    assert _ins_text(para) == "Payment due in 45 days"


def test_delete_strikes_paragraph_and_mark():
    doc = _doc(["Keep this.", "Either party may terminate with 60 days notice.", "Keep that."])
    report = _apply(
        doc, [{"op": "delete", "anchor_text": "Either party may terminate with 60 days notice"}]
    )
    # paragraph is not removed — it stays as a tracked deletion (text + paragraph mark)
    assert len(doc.paragraphs) == 3
    target = doc.paragraphs[1]
    assert "Either party may terminate with 60 days notice." in _del_text(target)
    assert _para_mark_tag(target) == "w:del"
    # surrounding paragraphs untouched
    assert doc.paragraphs[0].text == "Keep this."
    assert doc.paragraphs[2].text == "Keep that."
    assert report[0]["matched"] is True


def test_insert_after_adds_tracked_paragraph_in_position():
    doc = _doc(["Section 8 Term", "Section 10 Misc"])
    _apply(
        doc,
        [{"op": "insert_after", "anchor_text": "Section 8 Term",
          "new_text": "Section 9 Governing Law: State of Delaware."}],
    )
    assert len(doc.paragraphs) == 3
    inserted = doc.paragraphs[1]
    assert _ins_text(inserted) == "Section 9 Governing Law: State of Delaware."
    assert _para_mark_tag(inserted) == "w:ins"
    # placed between the anchor and the following paragraph
    assert doc.paragraphs[0].text == "Section 8 Term"
    assert doc.paragraphs[2].text == "Section 10 Misc"


def test_append_adds_tracked_paragraph_at_end():
    doc = _doc(["First.", "Second."])
    _apply(doc, [{"op": "append", "new_text": "Appended clause."}])
    last = doc.paragraphs[-1]
    assert _ins_text(last) == "Appended clause."
    assert _para_mark_tag(last) == "w:ins"


def test_anchor_matching_is_whitespace_and_case_insensitive():
    doc = _doc(["Section   4.2   PAYMENT terms apply."])
    report = _apply(
        doc, [{"op": "replace", "anchor_text": "section 4.2 payment terms", "new_text": "Replaced."}]
    )
    assert report[0]["matched"] is True
    assert _ins_text(doc) == "Replaced."


# ---------------------------------------------------------------------------
# markdown-tolerant anchor matching
#
# The agent reads the base as markdown (docx is extracted to markdown at ingest),
# so it copies anchors containing `**bold**`, `3.` list prefixes, and `#` headings.
# The docx paragraph text is raw — no markdown. Matching must bridge that gap so a
# verbatim-from-markdown anchor still locates the raw paragraph on the first pass.
# ---------------------------------------------------------------------------


def test_norm_strips_inline_emphasis_and_leading_markers():
    # leading list number + inline bold in the anchor; neither in the raw paragraph
    assert helper._norm("3. **Security Deposit:** pay **$4500** now") == \
        helper._norm("Security Deposit: pay $4500 now")


def test_replace_matches_markdown_anchor_against_raw_paragraph():
    # Reproduces the log.1 failure: the agent's anchor carried markdown, the docx
    # paragraph did not, so the anchor never matched and the agent burned its budget.
    doc = _doc(["Security Deposit: Tenant shall pay a security deposit of $4500 to Landlord."])
    report = _apply(
        doc,
        [{"op": "replace",
          "anchor_text": "3. **Security Deposit:** Tenant shall pay a security deposit of **$4500** to Landlord.",
          "new_text": "Security Deposit: Tenant shall pay a security deposit of $5000 to Landlord."}],
    )
    assert report[0]["matched"] is True
    assert "$4500" in _del_text(doc)
    assert "$5000" in _ins_text(doc)


def test_delete_matches_markdown_heading_anchor():
    doc = _doc(["Confidentiality", "Keep this."])
    report = _apply(
        doc, [{"op": "delete", "anchor_text": "## Confidentiality"}]
    )
    assert report[0]["matched"] is True
    assert "Confidentiality" in _del_text(doc)
    assert _para_mark_tag(doc.paragraphs[0]) == "w:del"
    assert doc.paragraphs[1].text == "Keep this."


def test_insert_after_matches_bulleted_markdown_anchor():
    doc = _doc(["Provider shall notify Customer within 48 hours.", "Section 6."])
    _apply(
        doc,
        [{"op": "insert_after",
          "anchor_text": "- Provider shall notify Customer within **48 hours**.",
          "new_text": "5.5 Subprocessors clause."}],
    )
    assert len(doc.paragraphs) == 3
    assert _ins_text(doc.paragraphs[1]) == "5.5 Subprocessors clause."
    assert doc.paragraphs[0].text == "Provider shall notify Customer within 48 hours."
    assert doc.paragraphs[2].text == "Section 6."


# ---------------------------------------------------------------------------
# unmatched / unknown operations
# ---------------------------------------------------------------------------


def test_unmatched_anchor_is_reported_not_applied():
    doc = _doc(["Only clause."])
    report = _apply(
        doc, [{"op": "replace", "anchor_text": "nonexistent section", "new_text": "x"}]
    )
    assert report[0]["matched"] is False
    assert doc.paragraphs[0].text == "Only clause."  # unchanged
    assert _ins_text(doc) == "" and _del_text(doc) == ""  # no redline produced


def test_unknown_op_reported_unmatched():
    doc = _doc(["Clause."])
    report = _apply(doc, [{"op": "frobnicate", "anchor_text": "Clause"}])
    assert report[0]["matched"] is False


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


def test_cli_applies_ops_and_reports(tmp_path):
    original = tmp_path / "in.docx"
    _doc(["Payment shall be due within 30 days.", "Termination clause here."]).save(str(original))

    ops = {"operations": [
        {"op": "replace", "anchor_text": "Payment shall be due within 30 days",
         "new_text": "Payment shall be due within 45 days."},
        {"op": "delete", "anchor_text": "Termination clause here"},
        {"op": "append", "new_text": "Governing law: Delaware."},
        {"op": "insert_after", "anchor_text": "does not exist", "new_text": "y"},
    ]}
    ops_path = tmp_path / "ops.json"
    ops_path.write_text(json.dumps(ops))
    output = tmp_path / "out.docx"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--original", str(original), "--ops", str(ops_path), "--output", str(output),
         "--author", "Jun Luo"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    report = json.loads(proc.stdout)
    assert report["unmatched"] == 1
    assert [r["matched"] for r in report["operations"]] == [True, True, True, False]

    redlined = Document(str(output))
    inserted = _inserted_text(redlined.element.body)
    deleted = _deleted_text(redlined.element.body)
    # replace + append landed as insertions; replace + delete landed as deletions
    assert "Payment shall be due within 45 days." in inserted
    assert "Governing law: Delaware." in inserted
    assert "Payment shall be due within 30 days." in deleted
    assert "Termination clause here." in deleted
    # author propagated from the CLI flag onto the tracked changes
    assert redlined.element.body.iter(qn("w:ins")).__next__().get(qn("w:author")) == "Jun Luo"
