"""In-memory implementation of StructuredStoreBase.

Useful for tests, prototyping, and single-process workloads that don't need persistence.
All data is lost when the process exits.
"""

from cogbase.core.models import Contradiction, Event, Fact
from cogbase.stores.base import StructuredStoreBase


class InMemoryStructuredStore(StructuredStoreBase):
    """Thread-unsafe in-memory store backed by plain dicts.

    Upserts by primary ID on every save — saving the same fact_id twice
    overwrites the previous value.
    """

    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}
        self._events: dict[str, Event] = {}
        self._contradictions: dict[str, Contradiction] = {}

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def save_facts(self, facts: list[Fact]) -> None:
        for fact in facts:
            self._facts[fact.fact_id] = fact

    def query_facts(self, filters: dict) -> list[Fact]:
        return [f for f in self._facts.values() if _fact_matches(f, filters)]

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def save_timeline(self, events: list[Event]) -> None:
        for event in events:
            self._events[event.event_id] = event

    def query_timeline(self, session_id: str) -> list[Event]:
        return sorted(
            (e for e in self._events.values() if e.session_id == session_id),
            key=lambda e: e.timestamp,
        )

    # ------------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------------

    def save_contradiction(self, c: Contradiction) -> None:
        self._contradictions[c.contradiction_id] = c

    def query_contradictions(self, filters: dict) -> list[Contradiction]:
        return [c for c in self._contradictions.values() if _contradiction_matches(c, filters)]


# ------------------------------------------------------------------
# Filter helpers
# ------------------------------------------------------------------

def _fact_matches(fact: Fact, filters: dict) -> bool:
    for key, value in filters.items():
        if getattr(fact, key, None) != value:
            return False
    return True


def _contradiction_matches(c: Contradiction, filters: dict) -> bool:
    for key, value in filters.items():
        if key == "doc_id":
            # Match if either nested fact belongs to the given doc
            if c.fact_a.doc_id != value and c.fact_b.doc_id != value:
                return False
        elif getattr(c, key, None) != value:
            return False
    return True
