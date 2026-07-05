"""Unit tests for the edit-docx skill's apply_operations helper.

The helper is a standalone CLI shipped inside the skill bundle (not an importable
package), so it is loaded by path. Tests cover the paragraph-level apply logic
(replace / delete / insert_after / append), formatting preservation, the unmatched
report, and the CLI entry point end-to-end.

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

SKILL_DIR = pathlib.Path(__file__).resolve().parents[2] / "skills" / "edit-docx"
SCRIPT = SKILL_DIR / "apply_operations.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("apply_operations", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


helper = _load_helper()


def _doc(paragraphs: list[str]) -> Document:
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    return doc


def _texts(doc: Document) -> list[str]:
    return [p.text for p in doc.paragraphs if p.text.strip()]


# ---------------------------------------------------------------------------
# apply_operations — per-op behavior
# ---------------------------------------------------------------------------


def test_replace_updates_matched_paragraph():
    doc = _doc(["Payment shall be due within 30 days.", "Other clause."])
    report = helper.apply_operations(
        doc,
        [{"op": "replace", "anchor_text": "Payment shall be due within 30 days",
          "new_text": "Payment shall be due within 45 days."}],
    )
    assert _texts(doc)[0] == "Payment shall be due within 45 days."
    assert report == [{"op": "replace", "anchor_text": "Payment shall be due within 30 days", "matched": True}]


def test_replace_preserves_run_formatting():
    doc = Document()
    para = doc.add_paragraph()
    run = para.add_run("Payment due in 30 days")
    run.bold = True

    helper.apply_operations(
        doc, [{"op": "replace", "anchor_text": "Payment due in 30 days", "new_text": "Payment due in 45 days"}]
    )

    assert para.runs[0].text == "Payment due in 45 days"
    assert para.runs[0].bold is True  # first run's formatting is retained


def test_delete_removes_paragraph():
    doc = _doc(["Keep this.", "Either party may terminate with 60 days notice.", "Keep that."])
    report = helper.apply_operations(
        doc, [{"op": "delete", "anchor_text": "Either party may terminate with 60 days notice"}]
    )
    assert _texts(doc) == ["Keep this.", "Keep that."]
    assert report[0]["matched"] is True


def test_insert_after_places_paragraph_in_position():
    doc = _doc(["Section 8 Term", "Section 10 Misc"])
    helper.apply_operations(
        doc,
        [{"op": "insert_after", "anchor_text": "Section 8 Term",
          "new_text": "Section 9 Governing Law: State of Delaware."}],
    )
    assert _texts(doc) == [
        "Section 8 Term",
        "Section 9 Governing Law: State of Delaware.",
        "Section 10 Misc",
    ]


def test_append_adds_at_end():
    doc = _doc(["First.", "Second."])
    helper.apply_operations(doc, [{"op": "append", "new_text": "Appended clause."}])
    assert _texts(doc)[-1] == "Appended clause."


def test_anchor_matching_is_whitespace_and_case_insensitive():
    doc = _doc(["Section   4.2   PAYMENT terms apply."])
    report = helper.apply_operations(
        doc, [{"op": "replace", "anchor_text": "section 4.2 payment terms", "new_text": "Replaced."}]
    )
    assert report[0]["matched"] is True
    assert _texts(doc)[0] == "Replaced."


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
    report = helper.apply_operations(
        doc,
        [{"op": "replace",
          "anchor_text": "3. **Security Deposit:** Tenant shall pay a security deposit of **$4500** to Landlord.",
          "new_text": "Security Deposit: Tenant shall pay a security deposit of $5000 to Landlord."}],
    )
    assert report[0]["matched"] is True
    assert _texts(doc)[0] == "Security Deposit: Tenant shall pay a security deposit of $5000 to Landlord."


def test_delete_matches_markdown_heading_anchor():
    doc = _doc(["Confidentiality", "Keep this."])
    report = helper.apply_operations(
        doc, [{"op": "delete", "anchor_text": "## Confidentiality"}]
    )
    assert report[0]["matched"] is True
    assert _texts(doc) == ["Keep this."]


def test_insert_after_matches_bulleted_markdown_anchor():
    doc = _doc(["Provider shall notify Customer within 48 hours.", "Section 6."])
    helper.apply_operations(
        doc,
        [{"op": "insert_after",
          "anchor_text": "- Provider shall notify Customer within **48 hours**.",
          "new_text": "5.5 Subprocessors clause."}],
    )
    assert _texts(doc) == [
        "Provider shall notify Customer within 48 hours.",
        "5.5 Subprocessors clause.",
        "Section 6.",
    ]


# ---------------------------------------------------------------------------
# unmatched / unknown operations
# ---------------------------------------------------------------------------


def test_unmatched_anchor_is_reported_not_applied():
    doc = _doc(["Only clause."])
    report = helper.apply_operations(
        doc, [{"op": "replace", "anchor_text": "nonexistent section", "new_text": "x"}]
    )
    assert report[0]["matched"] is False
    assert _texts(doc) == ["Only clause."]  # unchanged


def test_unknown_op_reported_unmatched():
    doc = _doc(["Clause."])
    report = helper.apply_operations(doc, [{"op": "frobnicate", "anchor_text": "Clause"}])
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
         "--original", str(original), "--ops", str(ops_path), "--output", str(output)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    report = json.loads(proc.stdout)
    assert report["unmatched"] == 1
    assert [r["matched"] for r in report["operations"]] == [True, True, True, False]

    merged = _texts(Document(str(output)))
    assert "Payment shall be due within 45 days." in merged
    assert "Governing law: Delaware." in merged
    assert all("Termination clause" not in t for t in merged)
