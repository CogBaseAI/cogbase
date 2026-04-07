"""Session primitive threaded through every skill call."""

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class Session(BaseModel):
    """Identifies a query or agent session.

    Not frozen — metadata can accumulate during a session's lifetime.
    Memory state belongs in the memory layer, not here.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
