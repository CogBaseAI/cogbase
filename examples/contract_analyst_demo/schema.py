"""Schema and Pydantic models for the legal contract review pack."""

from pydantic import BaseModel, Field, create_model


CONTRACTS_SYSTEM_PROMPT_PREFIX = (
    "You are a legal contract analyst.  Extract structured information from the\n"
    "contract provided by the user.\n\n"
    "Rules:\n"
    "- Copy all clause text verbatim — do not paraphrase or summarise.\n"
    "- Do not invent information not present in the contract.\n"
    "- Use null for any field not found in the contract.\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
    "Return a single JSON object with these fields:\n\n"
)


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
    (via ``cls_generate_schema``).

    ``doc_id`` MUST NOT appear here - it is injected by the extractor and declared
    in ``ContractExtractionRecord`` (the record schema).
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


ContractExtractionRecord = create_model(
    "ContractExtractionRecord",
    doc_id=(str, ...),
    __base__=ContractExtraction,
)

