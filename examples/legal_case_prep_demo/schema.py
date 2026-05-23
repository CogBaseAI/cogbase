"""Pydantic models for the legal case preparation demo.

Five per-document extractions populate the core collections:

- ``case_documents``   — one record per ingested document (inventory & summary)
- ``timeline_events``  — many records: every dated event, communication, action
- ``entities``         — many records: every named person, company, organisation
- ``facts``            — many records: factual assertions with party + source
- ``structured_data``  — many records: key dates, amounts, obligations, breaches

Two workflows produce derived collections:

- ``contradictions``   — incompatible factual claims across documents
- ``evidence_gaps``    — assertions unsupported by any other source
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model


# ---------------------------------------------------------------------------
# case_documents — document inventory (one record per document)
# ---------------------------------------------------------------------------

DocType = Literal[
    "correspondence",
    "contract",
    "invoice",
    "witness_statement",
    "expert_report",
    "court_order",
    "pleading",
    "disclosed_evidence",
    "other",
]


class CaseDocument(BaseModel):
    """Inventory record for one ingested case document.

    ``doc_id`` is injected by the extractor.
    """

    doc_type: DocType = Field(
        description=(
            "Document category. Choose the closest match: correspondence (letters, emails, "
            "memos), contract (agreements, amendments), invoice (bills, statements of "
            "account), witness_statement (signed statements of fact), expert_report "
            "(technical or expert opinion), court_order (judgments, directions, orders), "
            "pleading (claim form, defence, particulars), disclosed_evidence (any other "
            "document produced on disclosure), other."
        )
    )
    title: str = Field(
        description="Short human-readable title or subject line for the document",
    )
    summary: str = Field(
        description="Two to four sentence factual summary of what the document says",
    )
    relevance_tag: str = Field(
        description=(
            "Short tag for the issue or topic this document relates to, "
            "e.g. 'breach-notice', 'delivery-dispute', 'liability-allocation'"
        ),
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Named authors, signatories, or senders of the document",
    )
    recipients: list[str] = Field(
        default_factory=list,
        description="Named recipients or addressees of the document",
    )
    document_date: str | None = Field(
        default=None,
        description="Date the document itself bears (sent / signed / dated) in YYYY-MM-DD",
    )
    date_range_start: str | None = Field(
        default=None,
        description="Earliest date discussed inside the document in YYYY-MM-DD",
    )
    date_range_end: str | None = Field(
        default=None,
        description="Latest date discussed inside the document in YYYY-MM-DD",
    )


CaseDocumentRecord = create_model(
    "CaseDocumentRecord",
    doc_id=(str, ...),
    __base__=CaseDocument,
)


# ---------------------------------------------------------------------------
# timeline_events — every dated event, communication, obligation, action
# ---------------------------------------------------------------------------

EventType = Literal[
    "communication",
    "meeting",
    "obligation",
    "action",
    "notice",
    "payment",
    "delivery",
    "breach",
    "other",
]


class TimelineEvent(BaseModel):
    """One dated event extracted from the document.

    ``event_id`` and ``doc_id`` are injected by the extractor.
    """

    date_start: str = Field(
        description="Date the event occurred or began, in YYYY-MM-DD format",
    )
    date_end: str | None = Field(
        default=None,
        description="End date for multi-day events; null for single-day events",
    )
    event_type: EventType = Field(
        description="Event category"
    )
    description: str = Field(
        description="One sentence describing what happened",
    )
    actors: list[str] = Field(
        default_factory=list,
        description="Names of the people or entities involved",
    )
    issue: str | None = Field(
        default=None,
        description=(
            "Short tag for the disputed issue this event relates to, when relevant. "
            "Use the same tag across events for the same issue (e.g. 'delivery-dispute')."
        ),
    )
    source_quote: str = Field(
        description="Verbatim sentence or short passage from the document evidencing the event",
    )
    section_or_page: str | None = Field(
        default=None,
        description="Section heading, paragraph number, or page reference for the quote, when available",
    )


TimelineEventRecord = create_model(
    "TimelineEventRecord",
    event_id=(str, ...),
    doc_id=(str, ...),
    __base__=TimelineEvent,
)


# ---------------------------------------------------------------------------
# entities — every named individual, company, entity in the case bundle
# ---------------------------------------------------------------------------

EntityType = Literal["individual", "company", "organisation", "government_body", "other"]
EntityRole = Literal[
    "claimant",
    "defendant",
    "director",
    "counterparty",
    "witness",
    "expert",
    "insurer",
    "solicitor",
    "judge",
    "third_party",
    "other",
]


class Entity(BaseModel):
    """One named entity (person / company / organisation) appearing in the document.

    ``entity_id`` and ``doc_id`` are injected by the extractor.
    """

    name: str = Field(description="Full name as it appears in the document")
    entity_type: EntityType = Field(description="Entity category")
    role: EntityRole = Field(
        description="Role this entity plays in the case at the time of the document",
    )
    title_at_time: str | None = Field(
        default=None,
        description=(
            "Title or position the individual held at the time the document refers to, "
            "e.g. 'Managing Director', 'Project Manager', 'Head of Procurement'. "
            "Null for non-individuals or when not stated."
        ),
    )
    related_to: list[str] = Field(
        default_factory=list,
        description="Names of other entities this one is connected to (employer, counterparty, etc.)",
    )
    source_quote: str = Field(
        description="Verbatim passage where the entity appears in the document",
    )


EntityRecord = create_model(
    "EntityRecord",
    entity_id=(str, ...),
    doc_id=(str, ...),
    __base__=Entity,
)


# ---------------------------------------------------------------------------
# facts — factual assertions with attribution
# ---------------------------------------------------------------------------

FactCategory = Literal[
    "event",
    "amount",
    "date",
    "obligation",
    "state_of_mind",
    "breach",
    "performance",
    "communication",
    "other",
]


class Fact(BaseModel):
    """One factual assertion made in the document.

    ``fact_id`` and ``doc_id`` are injected by the extractor.
    """

    issue: str = Field(
        description=(
            "Short tag identifying which disputed issue this fact bears on, "
            "e.g. 'delivery-dispute', 'breach-notice', 'liability-allocation'. "
            "Use the same tag across documents for the same issue."
        ),
    )
    assertion: str = Field(
        description=(
            "One sentence stating the fact in neutral language as the document presents it, "
            "e.g. 'The goods were delivered on 14 March 2025'."
        ),
    )
    asserting_party: str = Field(
        description=(
            "Name of the party or person who asserts this fact in the document. "
            "Use the document author when the document itself makes the assertion."
        ),
    )
    fact_category: FactCategory = Field(description="Type of fact being asserted")
    source_quote: str = Field(
        description="Verbatim sentence from the document supporting the assertion",
    )
    section_or_page: str | None = Field(
        default=None,
        description="Section, paragraph number, or page reference for the quote",
    )


FactRecord = create_model(
    "FactRecord",
    fact_id=(str, ...),
    doc_id=(str, ...),
    __base__=Fact,
)


# ---------------------------------------------------------------------------
# structured_data — key dates, amounts, obligations, notices, breaches
# ---------------------------------------------------------------------------

StructuredKind = Literal[
    "date_deadline",
    "monetary_amount",
    "payment",
    "obligation",
    "notice_served",
    "alleged_breach",
    "clause_reference",
    "other",
]


class StructuredDataItem(BaseModel):
    """One discrete structured datum extracted from the document.

    Populates the cross-document reference table used for submissions,
    affidavits, and cross-examination outlines.

    ``item_id`` and ``doc_id`` are injected by the extractor.
    """

    kind: StructuredKind = Field(description="Kind of structured datum")
    description: str = Field(
        description="One sentence describing the item in plain language",
    )
    amount: float | None = Field(
        default=None,
        description="Monetary value as a number (no currency symbol); null for non-monetary items",
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 currency code (e.g. 'GBP', 'USD') when amount is set",
    )
    date: str | None = Field(
        default=None,
        description="Associated date or deadline in YYYY-MM-DD; null when not applicable",
    )
    party_responsible: str | None = Field(
        default=None,
        description="Party owing the obligation, making the payment, or alleged to be in breach",
    )
    clause_reference: str | None = Field(
        default=None,
        description=(
            "Contract clause, statute, or document section the item is grounded in, "
            "e.g. 'Clause 7.2', 'Schedule 3', 'paragraph 14'"
        ),
    )
    issue: str | None = Field(
        default=None,
        description="Short tag for the disputed issue this item relates to, when relevant",
    )
    source_quote: str = Field(
        description="Verbatim sentence or short passage supporting the item",
    )


StructuredDataItemRecord = create_model(
    "StructuredDataItemRecord",
    item_id=(str, ...),
    doc_id=(str, ...),
    __base__=StructuredDataItem,
)


# ---------------------------------------------------------------------------
# contradictions — workflow output
# ---------------------------------------------------------------------------

Significance = Literal["low", "medium", "high", "critical"]


class Contradiction(BaseModel):
    """One contradiction between two facts in the case bundle.

    Written by the detect-contradictions workflow. ``contradiction_id`` is the
    primary key so repeated runs overwrite earlier findings.
    """

    contradiction_id: str = Field(
        description=(
            "Stable identifier for the contradiction. "
            "Use the format '{issue}__{fact_a_id}__{fact_b_id}'."
        )
    )
    issue: str = Field(description="Short tag for the disputed issue both facts bear on")
    fact_a_id: str = Field(description="fact_id of the first conflicting fact")
    fact_b_id: str = Field(description="fact_id of the second conflicting fact")
    doc_a_id: str = Field(description="doc_id of the source document for fact A")
    doc_b_id: str = Field(description="doc_id of the source document for fact B")
    asserting_party_a: str = Field(description="Party that asserts fact A")
    asserting_party_b: str = Field(description="Party that asserts fact B")
    quote_a: str = Field(description="Verbatim source quote supporting fact A")
    quote_b: str = Field(description="Verbatim source quote supporting fact B")
    explanation: str = Field(
        description="One sentence describing why these two facts are mutually inconsistent",
    )
    significance: Significance = Field(
        description=(
            "Likely legal significance: critical — a witness statement directly conflicts "
            "with a contemporaneous document; high — material factual conflict on a "
            "central issue; medium — meaningful but peripheral conflict; "
            "low — minor date or wording inconsistency."
        )
    )
    reasoning: str = Field(
        description="Brief explanation of the significance ranking grounded in the quotes",
    )


class ContradictionList(BaseModel):
    """Wrapper used as the llm-structured output schema for the contradiction judge."""

    contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="All contradictions identified between the supplied facts. May be empty.",
    )


# ---------------------------------------------------------------------------
# evidence_gaps — workflow output
# ---------------------------------------------------------------------------

GapImpact = Literal["low", "medium", "high", "critical"]


class EvidenceGap(BaseModel):
    """One asserted fact that is not corroborated by any other source.

    Written by the identify-evidence-gaps workflow. ``gap_id`` is the primary key.
    """

    gap_id: str = Field(
        description=(
            "Stable identifier for the gap. Use the format '{issue}__{fact_id}'."
        )
    )
    fact_id: str = Field(description="fact_id of the uncorroborated assertion")
    doc_id: str = Field(description="doc_id of the document making the assertion")
    issue: str = Field(description="Short tag for the disputed issue")
    asserting_party: str = Field(description="Party that asserts the uncorroborated fact")
    gap_description: str = Field(
        description=(
            "One sentence describing what corroboration is missing, e.g. "
            "'no signed delivery confirmation to support the alleged acceptance of goods'"
        )
    )
    potential_impact: GapImpact = Field(
        description=(
            "How damaging the gap is to the asserting party's case: "
            "critical — pivotal fact with no support at all; high — material fact relying "
            "on a single weak source; medium — supporting fact lacking corroboration; "
            "low — peripheral detail."
        )
    )
    suggested_action: str = Field(
        description=(
            "Concrete next step, e.g. 'request further disclosure of delivery records', "
            "'commission an expert report on x', 'take an additional witness statement from y'."
        )
    )


class EvidenceGapList(BaseModel):
    """Wrapper used as the llm-structured output schema for the gap judge."""

    gaps: list[EvidenceGap] = Field(
        default_factory=list,
        description="All evidence gaps identified for the supplied facts. May be empty.",
    )
