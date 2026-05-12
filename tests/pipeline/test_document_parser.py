"""Unit tests for cogbase.pipeline.document_parser."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from cogbase.pipeline.document_parser import parse_to_markdown


class TestParseToMarkdown:
    def test_parses_plain_text(self):
        content = b"Hello, World!\nThis is a test document."
        result = parse_to_markdown(content, "note.txt")
        assert "Hello, World!" in result
        assert "test document" in result

    def test_filename_stem_does_not_appear_as_doc_id_in_output(self):
        content = b"Just some text."
        result = parse_to_markdown(content, "my_contract.txt")
        # markitdown should return the text content, not inject the filename
        assert result.strip() == "Just some text."

    def test_multiline_text_preserved(self):
        content = b"Line one.\nLine two.\nLine three."
        result = parse_to_markdown(content, "multi.txt")
        assert "Line one" in result
        assert "Line two" in result
        assert "Line three" in result

    def test_uses_extension_for_type_inference(self):
        # markitdown uses the extension from StreamInfo to route converters;
        # passing .txt should work even if the bytes look like a generic stream.
        content = b"Plain text content."
        result = parse_to_markdown(content, "document.TXT")  # uppercase ext
        assert "Plain text content" in result

    def test_raises_import_error_when_markitdown_missing(self):
        with patch.dict(sys.modules, {"markitdown": None}):
            with pytest.raises(ImportError, match="markitdown"):
                parse_to_markdown(b"text", "file.txt")

    def test_propagates_markitdown_exception(self):
        fake_result = MagicMock()
        fake_result.text_content = "ok"
        fake_md_instance = MagicMock()
        fake_md_instance.convert.side_effect = RuntimeError("parse boom")

        fake_markitdown_module = MagicMock()
        fake_markitdown_module.MarkItDown.return_value = fake_md_instance

        fake_stream_info_module = MagicMock()
        fake_stream_info_module.StreamInfo = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {
            "markitdown": fake_markitdown_module,
            "markitdown._stream_info": fake_stream_info_module,
        }):
            with pytest.raises(RuntimeError, match="parse boom"):
                parse_to_markdown(b"bytes", "file.pdf")

    def test_returns_string(self):
        result = parse_to_markdown(b"Some content.", "test.txt")
        assert isinstance(result, str)
