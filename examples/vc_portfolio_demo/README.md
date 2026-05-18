# VC Portfolio Intelligence Demo

Ask natural-language questions across a portfolio of board decks, LP updates, and investment memos: track ARR and burn rate over time, compare runway across companies, surface which deals mention hiring risk, or spot contradictions between what a company reported to the board and what it told LPs. Structured KPI lookups return exact records; open-ended questions stream a synthesized answer.

The demo ships with synthetic data for three portfolio companies — Nova Analytics, Helix Biotech, and Lumina Energy — across multiple quarters, plus investment memos for two of them.

## Quick start

```bash
# 1. Start the API server
uvicorn api.main:app --reload

# 2. Run the demo (from repo root)
python examples/vc_portfolio_demo/demo.py
```

Requires `OPENAI_API_KEY` in a `.env` file at the repo root (or in the environment). Set `COGBASE_API_URL` to override the default `http://localhost:8000`.

## Interactive commands

| Command | Description |
|---------|-------------|
| `/ingest_all` | Ingest all built-in board updates, LP updates, and investment memos |
| `/ingest_board` | Ingest only board updates and LP updates |
| `/ingest_memos` | Ingest only investment memos |

Any other input is sent as a natural-language query. Structured results (e.g. KPI lookups) are printed as JSON; open-ended answers stream as SSE tokens.

## Example queries

```
> Which companies are burning more than $500K per month?
> What was Nova Analytics' ARR growth across all quarters?
> Are there any contradictions in how Helix reported ARR in Q3 2024?
> Which portfolio companies have runway below 12 months?
> What are the biggest risks across the portfolio?
> What was the investment thesis for Helix Biotech?
```

## What the demo creates

One CogBase application named `vc-portfolio` with two pipelines routed by `metadata.doc_type`:

| Pipeline | `doc_type` | Steps |
|----------|------------|-------|
| `board-update` | `board_update` | chunk-embed-upsert → extract-structured → document-embed-upsert |
| `deal-memo` | `deal_memo` | chunk-embed-upsert → document-embed-upsert |

Investment memos are indexed for search but do not produce KPI records (there are no quarterly metrics to extract from a pre-investment document).

### Collections

| Collection | Type | Purpose |
|------------|------|---------|
| `portfolio_chunks` | Vector | Full-text passage index across all documents |
| `portfolio_summaries` | Vector | One LLM-generated summary per document |
| `portfolio_kpis` | Structured | Extracted KPIs from board decks and LP updates |

### Extracted KPI record

Each board update or LP update produces one record in `portfolio_kpis`.

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | Source document identifier (injected by pipeline) |
| `company_name` | `str \| None` | Portfolio company name |
| `reporting_period` | `str \| None` | Quarter and year, e.g. `"Q3 2024"` |
| `doc_type` | `str \| None` | `"board_update"` or `"deal_memo"` |
| `arr_usd` | `float \| None` | Annual Recurring Revenue in USD |
| `mrr_usd` | `float \| None` | Monthly Recurring Revenue in USD |
| `arr_growth_yoy_pct` | `float \| None` | ARR year-over-year growth percentage |
| `burn_rate_monthly_usd` | `float \| None` | Net monthly cash burn in USD |
| `runway_months` | `float \| None` | Cash runway in months |
| `headcount` | `int \| None` | Total full-time employee count |
| `customer_count` | `int \| None` | Total paying customer count |
| `ndr_pct` | `float \| None` | Net Dollar Retention percentage |
| `key_milestones` | `list[str]` | Milestones achieved this period. `[]` if none. |
| `notable_risks` | `list[str]` | Risks or concerns raised. `[]` if none. |

## Project structure

```text
vc_portfolio_demo/
├── README.md
├── config.yaml             # pipeline and collection definitions
├── demo.py                 # interactive demo script
├── schema.py               # PortfolioKPIExtraction, PortfolioKPIRecord
├── kpi_extraction_prompt.txt
└── portfolio_data.py       # synthetic board updates and deal memos
```
