"""Template rendering for workflow step parameters.

The implementation moved to :mod:`cogbase.templating` when the pipeline layer
grew the same need (the ``extract-structured`` ``fields:`` overlay). Re-exported
here because every workflow tool imports it from this module.
"""

from __future__ import annotations

from cogbase.templating import jinja_available, render_value

__all__ = ["jinja_available", "render_value"]
