"""Schema and Pydantic models for the legal contract review pack."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType
from cogbase.stores.schema_util import cls_generate_schema

# Fields that ContractExtractor._parse always populates — they cannot be
# excluded from a customised schema without breaking extraction.
_CORE_FIELDS = frozenset({"contract_id", "doc_id"})

CONTRACTS_COLLECTION = "contracts"


class Party(BaseModel):
    """A named party to the contract."""

    name: str = Field(description="full legal name of the party")
    role: str | None = Field(
        default=None,
        description='role in the contract, e.g. "buyer", "seller", "licensor", "licensee", "employer", "employee"',
    )
    jurisdiction: str | None = Field(
        default=None,
        description="state or country of incorporation / governing jurisdiction",
    )


class PaymentTerms(BaseModel):
    """Structured payment terms extracted from the contract."""

    schedule: str | None = Field(
        default=None,
        description='payment schedule, e.g. "net-30", "monthly", "upfront", "milestone-based"',
    )
    due_date: str | None = Field(
        default=None,
        description="specific payment due date in YYYY-MM-DD format, if stated",
    )
    late_penalty: str | None = Field(
        default=None,
        description="penalty or interest rate for late payment, verbatim if present",
    )
    verbatim: str | None = Field(
        default=None,
        description="verbatim payment terms clause from the contract",
    )


class ContractExtraction(BaseModel):
    """Structured information extracted by the LLM from a contract document.

    Every field is optional (null if not found).  This model drives both the
    LLM prompt schema (via ``cls_json_schema_for_llm``) and the store schema
    (via ``cls_generate_schema`` applied to ``ContractRecord``).
    """

    # contract basics
    contract_type: str | None = Field(
        default=None,
        description='type of contract, e.g. "NDA", "SaaS", "employment", "vendor", "lease"',
    )
    purpose: str | None = Field(
        default=None,
        description="one sentence describing what the contract is for",
    )
    effective_date: str | None = Field(
        default=None,
        description="start date in YYYY-MM-DD format",
    )
    expiry_date: str | None = Field(
        default=None,
        description="end/expiry date in YYYY-MM-DD format",
    )
    parties: list[Party] = Field(
        default_factory=list,
        description="all named parties to the contract; use [] if none identified",
    )
    contract_value: float | None = Field(
        default=None,
        description="total monetary value as a number (no currency symbol)",
    )
    currency: str | None = Field(
        default=None,
        description='ISO 4217 currency code (e.g. "USD")',
    )
    # common clause text (verbatim)
    payment_terms: PaymentTerms | None = Field(
        default=None,
        description="structured payment terms extracted from the contract",
    )
    termination: str | None = Field(
        default=None,
        description="verbatim termination clause",
    )
    liability: str | None = Field(
        default=None,
        description="verbatim limitation of liability clause",
    )
    governing_law: str | None = Field(
        default=None,
        description="verbatim governing law clause",
    )
    confidentiality: str | None = Field(
        default=None,
        description="verbatim confidentiality clause",
    )
    indemnification: str | None = Field(
        default=None,
        description="verbatim indemnification clause",
    )
    dispute_resolution: str | None = Field(
        default=None,
        description="verbatim dispute resolution clause",
    )
    # clause-level numeric
    notice_period_days: int | None = Field(
        default=None,
        description="integer days required for termination notice",
    )
    liability_cap: float | None = Field(
        default=None,
        description="liability cap amount as a number",
    )
    # flexible extraction
    key_terms: list[str] = Field(
        default_factory=list,
        description="significant defined terms, unusual provisions, or contract-type-specific clauses not covered above; use [] if none",
    )
    special_conditions: list[str] = Field(
        default_factory=list,
        description="verbatim conditions precedent, carve-outs, custom provisions, or anything unusual; use [] if none",
    )


class ContractRecord(ContractExtraction):
    """Full stored record: ``ContractExtraction`` fields plus identity fields."""

    contract_id: str
    doc_id: str


CONTRACTS_SCHEMA = CollectionSchema(
    name=CONTRACTS_COLLECTION,
    id_field="contract_id",
    fields=cls_generate_schema(ContractRecord),
)


def build_contracts_schema(
    extra_fields: dict[str, FieldSchema] | None = None,
    exclude: set[str] | None = None,
) -> CollectionSchema:
    """Build a customised contracts schema from the default.

    Start from ``CONTRACTS_SCHEMA`` and apply additions and/or removals to
    produce a new ``CollectionSchema`` without touching the shared default.

    Args:
        extra_fields: Additional fields to append after the defaults.  Field
                      names must not already exist in ``CONTRACTS_SCHEMA``.
        exclude:      Field names to remove from the default schema.
                      ``contract_id`` and ``doc_id`` cannot be excluded —
                      ``ContractExtractor`` always populates them.

    Returns:
        A new ``CollectionSchema`` with the requested customisations applied.

    Raises:
        ValueError: If *exclude* contains a core field, or if *extra_fields*
                    duplicates an existing field name.

    Example — remove fields your company does not need::

        schema = build_contracts_schema(exclude={"indemnification", "dispute_resolution"})

    Example — add a company-specific field::

        from cogbase.stores.schema import FieldSchema, FieldType
        schema = build_contracts_schema(
            extra_fields={"risk_score": FieldSchema(type=FieldType.FLOAT, nullable=True)}
        )

    Example — both at once::

        schema = build_contracts_schema(
            extra_fields={"jurisdiction": FieldSchema(type=FieldType.STRING, nullable=True, index=True)},
            exclude={"dispute_resolution"},
        )
    """
    excluded = exclude or set()
    extras = extra_fields or {}

    protected = excluded & _CORE_FIELDS
    if protected:
        raise ValueError(
            f"Cannot exclude core fields required by ContractExtractor: {sorted(protected)}"
        )

    duplicates = extras.keys() & CONTRACTS_SCHEMA.fields.keys()
    if duplicates:
        raise ValueError(
            f"extra_fields duplicates existing field names: {sorted(duplicates)}. "
            "To change an existing field, exclude it first then re-add it."
        )

    fields = {
        name: field
        for name, field in CONTRACTS_SCHEMA.fields.items()
        if name not in excluded
    }
    fields.update(extras)

    return CollectionSchema(
        name=CONTRACTS_SCHEMA.name,
        id_field=CONTRACTS_SCHEMA.id_field,
        fields=fields,
    )
