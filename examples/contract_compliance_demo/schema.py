"""Pydantic models and CollectionSchema definitions for the contract compliance demo.

Exports
-------
Models (used for extraction, query results, and skill I/O):
  Party                          — named party in a contract (name + role)
  ContractClause                 — one stored clause record (contract_clauses)
  ContractClausesExtractionResult — LLM output wrapper used by the clause extractor
  ContractMetadata               — contract-level facts (contract_metadata)
  ClauseComplianceFinding        — one compliance finding (clause_compliance_findings)

CollectionSchema objects (passed to structured_store.create_collection):
  CONTRACT_CLAUSES_SCHEMA
  CONTRACT_METADATA_SCHEMA
  CLAUSE_COMPLIANCE_FINDINGS_SCHEMA
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model

from cogbase.stores import CollectionSchema, FieldSchema, FieldType


# ---------------------------------------------------------------------------
# Supporting model
# ---------------------------------------------------------------------------

class Party(BaseModel):
    """A named party in a contract and its role in the agreement."""

    name: str = Field(description="Legal name of the party")
    role: str = Field(description="Role in the agreement, e.g. vendor, customer, licensor, licensee")


# ---------------------------------------------------------------------------
# contract_clauses
# ---------------------------------------------------------------------------

class ContractClause(BaseModel):
    """One clause extracted by the LLM (extraction schema).

    ``clause_id`` and ``doc_id`` must NOT appear here — they are injected by the
    extractor and declared in ``ContractClauseRecord`` (the record schema).
    """

    clause_type: str | None = Field(
        default=None,
        description=(
            "Clause category. Use one of: liability, indemnification, termination, "
            "payment, privacy, confidentiality, ip, governing_law, other."
        ),
    )
    text: str = Field(description="Verbatim clause text copied from the contract without paraphrasing")


ContractClauseRecord = create_model(
    "ContractClauseRecord",
    clause_id=(str, ...),
    doc_id=(str, ...),
    __base__=ContractClause,
)


# ---------------------------------------------------------------------------
# contract_metadata
# ---------------------------------------------------------------------------

class ContractMetadata(BaseModel):
    """Contract-level facts extracted by the LLM from a contract document (extraction schema).

    ``doc_id`` must NOT appear here — it is injected by the extractor and declared
    in ``ContractMetadataRecord`` (the record schema stored in the collection).
    """

    contract_type: str | None = Field(
        default=None,
        description="Contract category, e.g. SaaS subscription, professional services, data processing agreement",
    )
    parties: list[Party] = Field(
        default_factory=list,
        description="All named parties and their roles in the agreement",
    )
    effective_date: str | None = Field(
        default=None,
        description="Contract start date in YYYY-MM-DD format",
    )
    expiry_date: str | None = Field(
        default=None,
        description="Contract end or expiry date in YYYY-MM-DD format",
    )
    contract_value: float | None = Field(
        default=None,
        description="Total monetary value of the contract when explicitly stated",
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 currency code for contract_value, e.g. USD, EUR",
    )
    governing_law: str | None = Field(
        default=None,
        description="Governing law jurisdiction as stated in the contract, e.g. 'State of New York'",
    )
    termination_notice_days: int | None = Field(
        default=None,
        description="Number of days of written notice required to terminate for convenience",
    )


ContractMetadataRecord = create_model(
    "ContractMetadataRecord",
    doc_id=(str, ...),
    __base__=ContractMetadata,
)


# ---------------------------------------------------------------------------
# clause_compliance_findings
# ---------------------------------------------------------------------------

class ClauseComplianceFinding(BaseModel):
    """One compliance finding produced by the compliance-check.

    ``clause_id`` is the primary key for stable, idempotent upserts.
    Re-running the check overwrites prior findings for the same key.
    """

    clause_id: str = Field(description="ID of the reviewed clause from contract_clauses")
    doc_id: str = Field(description="Source contract document ID")
    clause_type: str | None = Field(
        default=None,
        description="Category of the reviewed clause, e.g. liability, payment",
    )
    status: Literal["compliant", "non_compliant", "needs_review", "not_applicable"] = Field(
        description=(
            "compliant — clause satisfies company policy; "
            "non_compliant — clause violates company policy; "
            "needs_review — retrieved rules are insufficient to make a determination; "
            "not_applicable — the company rules do not cover this clause type."
        )
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description=(
            "low — minor deviation with minimal business impact; "
            "medium — notable deviation requiring negotiation; "
            "high — significant policy violation that must be corrected; "
            "critical — clause poses serious legal, financial, or security risk."
        )
    )
    summary: str = Field(description="One-sentence human-readable finding")
    contract_clause_text: str = Field(description="Verbatim text of the reviewed clause")
    matched_rule_ids: list[str] = Field(
        default_factory=list,
        description="IDs of the rule chunks used as evidence for this finding",
    )
    matched_rule_quotes: list[str] = Field(
        default_factory=list,
        description="Verbatim excerpts from the matched rule chunks that support the finding",
    )
    reasoning: str = Field(
        description="Explanation of the finding grounded exclusively in the matched rule excerpts"
    )
    recommended_redline: str | None = Field(
        default=None,
        description=(
            "Suggested replacement clause language that would bring the clause into compliance. "
            "Null when status is compliant or not_applicable."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Judge confidence in the finding on a scale from 0.0 (uncertain) to 1.0 (certain)",
    )


CLAUSE_COMPLIANCE_FINDINGS_SCHEMA = CollectionSchema(
    name="clause_compliance_findings",
    description=(
        "Clause-level compliance findings produced by the compliance-check skill. "
        "Each record links a contract clause to the company rules it was checked against "
        "and records whether the clause is compliant, non-compliant, needs review, or "
        "not applicable. Filter by doc_id to get all findings for a contract, or filter "
        "by status and severity to find high-priority issues across all contracts."
    ),
    primary_fields=["clause_id"],
    fields={
        "doc_id":               FieldSchema(type=FieldType.STRING, index=True),
        "clause_id":            FieldSchema(type=FieldType.STRING, index=True),
        "clause_type":          FieldSchema(type=FieldType.STRING, nullable=True, index=True),
        "status":               FieldSchema(type=FieldType.STRING, index=True),
        "severity":             FieldSchema(type=FieldType.STRING, index=True),
        "summary":              FieldSchema(type=FieldType.STRING),
        "contract_clause_text": FieldSchema(type=FieldType.STRING),
        "matched_rule_ids":     FieldSchema(type=FieldType.JSON),
        "matched_rule_quotes":  FieldSchema(type=FieldType.JSON),
        "reasoning":            FieldSchema(type=FieldType.STRING),
        "recommended_redline":  FieldSchema(type=FieldType.STRING, nullable=True),
        "confidence":           FieldSchema(type=FieldType.FLOAT),
    },
)
