"""Schema and Pydantic model for the legal contract review pack."""

from __future__ import annotations

from pydantic import BaseModel, Field

from cogbase.stores.schema import CollectionSchema, FieldSchema, FieldType

# Fields that ContractExtractor._parse always populates — they cannot be
# excluded from a customised schema without breaking extraction.
_CORE_FIELDS = frozenset({"contract_id", "doc_id"})

CONTRACTS_COLLECTION = "contracts"

CONTRACTS_SCHEMA = CollectionSchema(
    name=CONTRACTS_COLLECTION,
    id_field="contract_id",
    fields={
        # --- identity ---
        "contract_id":        FieldSchema(type=FieldType.STRING),
        "doc_id":             FieldSchema(type=FieldType.STRING, index=True),
        # --- contract basics ---
        "contract_type":      FieldSchema(type=FieldType.STRING, nullable=True, index=True),
        "purpose":            FieldSchema(type=FieldType.STRING, nullable=True),
        "effective_date":     FieldSchema(type=FieldType.STRING, nullable=True, index=True),
        "expiry_date":        FieldSchema(type=FieldType.STRING, nullable=True, index=True),
        "party_a":            FieldSchema(type=FieldType.STRING, nullable=True, index=True),
        "party_b":            FieldSchema(type=FieldType.STRING, nullable=True, index=True),
        "contract_value":     FieldSchema(type=FieldType.FLOAT,  nullable=True),
        "currency":           FieldSchema(type=FieldType.STRING, nullable=True),
        # --- common clause text (verbatim) ---
        "payment_terms":      FieldSchema(type=FieldType.STRING, nullable=True),
        "termination":        FieldSchema(type=FieldType.STRING, nullable=True),
        "liability":          FieldSchema(type=FieldType.STRING, nullable=True),
        "governing_law":      FieldSchema(type=FieldType.STRING, nullable=True),
        "confidentiality":    FieldSchema(type=FieldType.STRING, nullable=True),
        "indemnification":    FieldSchema(type=FieldType.STRING, nullable=True),
        "dispute_resolution": FieldSchema(type=FieldType.STRING, nullable=True),
        # --- clause-level numeric ---
        "notice_period_days": FieldSchema(type=FieldType.INTEGER, nullable=True),
        "liability_cap":      FieldSchema(type=FieldType.FLOAT,   nullable=True),
        # --- flexible extraction ---
        "key_terms":          FieldSchema(type=FieldType.JSON),
        "special_conditions": FieldSchema(type=FieldType.JSON),
    },
)


class ContractRecord(BaseModel):
    """Structured summary extracted from a single contract document.

    One ``ContractRecord`` is produced per ingested document.

    Attributes:
        contract_id:        Stable unique ID: ``{doc_id}_{uuid}``.
        doc_id:             Source document identifier.
        contract_type:      Contract category (e.g. ``"NDA"``, ``"SaaS"``,
                            ``"employment"``, ``"vendor"``, ``"lease"``).
        purpose:            One-sentence description of what the contract is for.
        effective_date:     Contract start date in ``YYYY-MM-DD`` format.
        expiry_date:        Contract end/expiry date in ``YYYY-MM-DD`` format.
        party_a:            Primary party name (client or buyer).
        party_b:            Counterparty name (vendor or seller).
        contract_value:     Total monetary value in ``currency`` units.
        currency:           ISO 4217 currency code (e.g. ``"USD"``).
        payment_terms:      Verbatim payment terms clause text.
        termination:        Verbatim termination clause text.
        liability:          Verbatim limitation-of-liability clause text.
        governing_law:      Verbatim governing law clause text.
        confidentiality:    Verbatim confidentiality clause text.
        indemnification:    Verbatim indemnification clause text.
        dispute_resolution: Verbatim dispute resolution clause text.
        notice_period_days: Notice period for termination in days.
        liability_cap:      Liability cap amount in ``currency`` units.
        key_terms:          Significant defined terms, unusual provisions, or
                            contract-type-specific clauses not covered by the
                            named fields above. Each entry is a dict with
                            ``"term"`` and ``"description"`` keys.
        special_conditions: Verbatim text of conditions precedent, carve-outs,
                            custom provisions, or anything unusual in the contract.
    """

    contract_id: str
    doc_id: str
    # contract basics
    contract_type: str | None = None
    purpose: str | None = None
    effective_date: str | None = None
    expiry_date: str | None = None
    party_a: str | None = None
    party_b: str | None = None
    contract_value: float | None = None
    currency: str | None = None
    # common clause text
    payment_terms: str | None = None
    termination: str | None = None
    liability: str | None = None
    governing_law: str | None = None
    confidentiality: str | None = None
    indemnification: str | None = None
    dispute_resolution: str | None = None
    # clause-level numeric
    notice_period_days: int | None = None
    liability_cap: float | None = None
    # flexible extraction
    key_terms: list[dict] = Field(default_factory=list)
    special_conditions: list[str] = Field(default_factory=list)


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
