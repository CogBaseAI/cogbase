"""Parse uploaded files to markdown text using markitdown."""

from __future__ import annotations

import io
import pathlib


def parse_to_markdown(content: bytes, filename: str) -> str:
    """Convert file bytes to markdown text via markitdown.

    Supports all formats handled by markitdown[all]: PDF, DOCX, PPTX, XLSX,
    XLS, HTML, XML, JSON, CSV, plain text, Outlook MSG, and audio transcription.

    Args:
        content:  Raw bytes of the uploaded file.
        filename: Original filename (used to infer the file type).

    Returns:
        Extracted text as a markdown string.

    Raises:
        ImportError: If markitdown is not installed.
        Exception:   If markitdown cannot parse the file.
    """
    try:
        from markitdown import MarkItDown
        from markitdown._stream_info import StreamInfo
    except ImportError as exc:
        raise ImportError("markitdown is required: pip install 'markitdown[all]'") from exc

    suffix = pathlib.Path(filename).suffix.lower()
    md = MarkItDown()
    result = md.convert(
        io.BytesIO(content),
        stream_info=StreamInfo(extension=suffix, filename=filename),
    )
    return result.text_content
