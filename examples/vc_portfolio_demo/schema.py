"""Pydantic models for VC portfolio KPI extraction."""

from pydantic import BaseModel, Field, create_model


class PortfolioKPIExtraction(BaseModel):
    """Fields the LLM extracts from a board deck or LP update.

    ``doc_id`` MUST NOT appear here — it is injected by the extractor and
    declared in ``PortfolioKPIRecord``.
    """

    company_name: str | None = Field(
        default=None,
        description="Portfolio company name, e.g. 'Acme Corp'.",
    )
    reporting_period: str | None = Field(
        default=None,
        description="Quarter and year of the report, e.g. 'Q3 2024'.",
    )
    doc_type: str | None = Field(
        default=None,
        description="Document type inferred from content: 'board_update' or 'deal_memo'.",
    )
    arr_usd: float | None = Field(
        default=None,
        description="Annual Recurring Revenue in USD as a plain number, e.g. 2400000 for $2.4M.",
    )
    mrr_usd: float | None = Field(
        default=None,
        description="Monthly Recurring Revenue in USD as a plain number.",
    )
    arr_growth_yoy_pct: float | None = Field(
        default=None,
        description="ARR year-over-year growth percentage, e.g. 85.0 for 85%.",
    )
    burn_rate_monthly_usd: float | None = Field(
        default=None,
        description="Net monthly cash burn in USD as a plain number.",
    )
    runway_months: float | None = Field(
        default=None,
        description="Cash runway in months as of the reporting date.",
    )
    headcount: int | None = Field(
        default=None,
        description="Total full-time employee count as an integer.",
    )
    customer_count: int | None = Field(
        default=None,
        description="Total paying customer count as an integer.",
    )
    ndr_pct: float | None = Field(
        default=None,
        description="Net Dollar Retention percentage, e.g. 118.0 for 118%.",
    )
    key_milestones: list[str] = Field(
        default_factory=list,
        description="Key milestones achieved this period. Use [] if none mentioned.",
    )
    notable_risks: list[str] = Field(
        default_factory=list,
        description="Risks or concerns raised. Use [] if none mentioned.",
    )


PortfolioKPIRecord = create_model(
    "PortfolioKPIRecord",
    doc_id=(str, ...),
    __base__=PortfolioKPIExtraction,
)
