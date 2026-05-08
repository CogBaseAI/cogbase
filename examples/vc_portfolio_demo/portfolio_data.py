"""Synthetic board updates and deal memos for the VC portfolio demo."""

# doc_id convention: {company}_{period}_{doc_type}
# metadata.doc_type is used to route to the correct pipeline

BOARD_UPDATES: dict[str, dict] = {
    "nova_q1_2024": {
        "text": """
NOVA ANALYTICS — Q1 2024 BOARD UPDATE

Company: Nova Analytics
Reporting Period: Q1 2024
Document Type: Board Update

FINANCIAL HIGHLIGHTS
- ARR: $4,200,000 (up from $2,800,000 in Q1 2023 — 50% YoY growth)
- MRR: $350,000
- Net Dollar Retention: 118%
- Monthly Burn: $420,000
- Cash Runway: 18 months
- Headcount: 42 FTEs

OPERATIONAL HIGHLIGHTS
- Customers: 87 paying accounts
- Closed largest enterprise deal to date: $240,000 ACV with Meridian Financial
- Launched new data connector for Salesforce (15 integrations total)
- Promoted VP of Sales from within; head of marketing search underway

KEY MILESTONES THIS QUARTER
- Closed Meridian Financial at $240K ACV — largest deal in company history
- Shipped Salesforce connector ahead of schedule
- Achieved SOC 2 Type II certification

RISKS & CONCERNS
- Pipeline coverage at 2.1x — below the 3x target entering Q2
- Two senior engineers departed; engineering capacity constrained
- Sales cycle lengthening in mid-market segment (avg 67 days, up from 52)
""",
        "metadata": {"doc_type": "board_update", "company": "nova_analytics"},
    },
    "nova_q2_2024": {
        "text": """
NOVA ANALYTICS — Q2 2024 BOARD UPDATE

Company: Nova Analytics
Reporting Period: Q2 2024
Document Type: Board Update

FINANCIAL HIGHLIGHTS
- ARR: $5,100,000 (up from $3,200,000 in Q2 2023 — 59% YoY growth)
- MRR: $425,000
- Net Dollar Retention: 121%
- Monthly Burn: $390,000
- Cash Runway: 14 months
- Headcount: 47 FTEs

OPERATIONAL HIGHLIGHTS
- Customers: 104 paying accounts
- Hired new VP of Marketing (started June 1)
- Launched enterprise tier; 8 accounts upgraded in first 6 weeks
- Backfilled both departed engineers; two more offers out

KEY MILESTONES THIS QUARTER
- Crossed 100 customer milestone
- Enterprise tier launched with 8 immediate upgrades
- VP Marketing onboarded; demand gen program underway

RISKS & CONCERNS
- Runway down to 14 months — Series B process should begin no later than Q4
- Churn in SMB segment elevated: 3 logos churned (largest was $18K ARR)
- Infrastructure costs growing faster than revenue; need to optimize
""",
        "metadata": {"doc_type": "board_update", "company": "nova_analytics"},
    },
    "nova_q3_2024": {
        "text": """
NOVA ANALYTICS — Q3 2024 BOARD UPDATE

Company: Nova Analytics
Reporting Period: Q3 2024
Document Type: Board Update

FINANCIAL HIGHLIGHTS
- ARR: $6,800,000 (up from $3,900,000 in Q3 2023 — 74% YoY growth)
- MRR: $566,667
- Net Dollar Retention: 124%
- Monthly Burn: $510,000
- Cash Runway: 10 months
- Headcount: 61 FTEs

OPERATIONAL HIGHLIGHTS
- Customers: 128 paying accounts
- Series B process launched; 4 term sheets received, final diligence underway
- Signed two Fortune 500 pilots (Crestwood Industries, Apex Group)
- Expanded to EMEA: 3 European customers signed

KEY MILESTONES THIS QUARTER
- 4 term sheets received in Series B process
- Crestwood Industries and Apex Group pilots signed
- First EMEA customers onboarded

RISKS & CONCERNS
- Runway critically low at 10 months; Series B close is urgent
- Sales headcount ramp slower than modeled — two AE hires 45 days behind
- EMEA expansion costs ahead of revenue: €120K spend, €0 ARR so far
""",
        "metadata": {"doc_type": "board_update", "company": "nova_analytics"},
    },
    "helix_q2_2024": {
        "text": """
HELIX BIOTECH — Q2 2024 BOARD UPDATE

Company: Helix Biotech
Reporting Period: Q2 2024
Document Type: Board Update

FINANCIAL HIGHLIGHTS
- ARR: $1,800,000
- MRR: $150,000
- Net Dollar Retention: 108%
- Monthly Burn: $680,000
- Cash Runway: 22 months
- Headcount: 38 FTEs

OPERATIONAL HIGHLIGHTS
- Customers: 14 biopharma accounts (all enterprise)
- Phase II clinical trial data integration module launched
- Partnership signed with LabCore for data sharing
- FDA digital health pilot approved for genomics workflow module

KEY MILESTONES THIS QUARTER
- Phase II clinical trial integration shipped — 6 months ahead of roadmap
- FDA digital health pilot approval received
- LabCore partnership signed

RISKS & CONCERNS
- ARR growth slower than projected ($1.8M vs $2.4M target for H1)
- Long sales cycles in biopharma (avg 9 months); pipeline visibility limited
- Headcount below plan — 3 open scientific roles unfilled for >90 days
""",
        "metadata": {"doc_type": "board_update", "company": "helix_biotech"},
    },
    "helix_q3_2024": {
        "text": """
HELIX BIOTECH — Q3 2024 BOARD UPDATE

Company: Helix Biotech
Reporting Period: Q3 2024
Document Type: Board Update

FINANCIAL HIGHLIGHTS
- ARR: $2,600,000 (44% YoY growth)
- MRR: $216,667
- Net Dollar Retention: 112%
- Monthly Burn: $720,000
- Cash Runway: 18 months
- Headcount: 44 FTEs

OPERATIONAL HIGHLIGHTS
- Customers: 19 biopharma accounts
- Closed Pfizer pilot expansion to full contract ($420,000 ACV)
- Genomics workflow module GA released
- Filed 2 provisional patents on ML-based variant classification

KEY MILESTONES THIS QUARTER
- Pfizer pilot converted to $420K ACV full contract
- Genomics workflow module generally available
- 2 provisional patents filed

RISKS & CONCERNS
- Burn rate increasing ($720K/mo) as we scale clinical team
- Competitor (GenomeSight) raised $50M Series C — accelerating their roadmap
- Two enterprise RFPs lost to incumbent vendors on procurement relationship
""",
        "metadata": {"doc_type": "board_update", "company": "helix_biotech"},
    },
    # Intentional contradiction: Helix LP update reports different ARR than board deck for Q3
    "helix_lp_q3_2024": {
        "text": """
HELIX BIOTECH — Q3 2024 LP UPDATE

Company: Helix Biotech
Reporting Period: Q3 2024
Document Type: LP Update

Dear Limited Partners,

We are pleased to share our Q3 2024 update for Helix Biotech, one of our portfolio companies in the precision medicine infrastructure space.

FINANCIAL SUMMARY
Helix ended Q3 2024 with ARR of $2,200,000, representing 22% growth year-over-year.
Monthly burn is approximately $700,000 against a cash runway of 18 months.
The company employs 44 people.

HIGHLIGHTS
- The Pfizer relationship continues to develop positively.
- The genomics workflow product launched to general availability in September.
- Two provisional patents have been filed for ML-based variant classification.

PORTFOLIO CONTEXT
Helix remains on track for a Series B raise in mid-2025. The biopharma software market
continues to show strong demand for compliant, AI-enabled data platforms.

Regards,
Fund Team
""",
        "metadata": {"doc_type": "board_update", "company": "helix_biotech"},
    },
    "lumina_q3_2024": {
        "text": """
LUMINA ENERGY — Q3 2024 BOARD UPDATE

Company: Lumina Energy
Reporting Period: Q3 2024
Document Type: Board Update

FINANCIAL HIGHLIGHTS
- ARR: $9,400,000 (91% YoY growth)
- MRR: $783,333
- Net Dollar Retention: 132%
- Monthly Burn: $850,000
- Cash Runway: 26 months
- Headcount: 79 FTEs

OPERATIONAL HIGHLIGHTS
- Customers: 31 utility and grid operator accounts
- Series B ($22M) closed in August; led by Greenfield Capital
- DOE grant ($3.2M) awarded for grid resilience modeling project
- Expanded into Texas market; ERCOT pilot live

KEY MILESTONES THIS QUARTER
- $22M Series B closed
- DOE grant awarded
- ERCOT pilot live — largest grid operator in North America

RISKS & CONCERNS
- Regulatory timeline risk for utility procurement: 2 deals pushed to Q4
- Burn increasing post-Series B as we invest in go-to-market
- Key account (Pacific Grid, 12% of ARR) in M&A process — outcome uncertain
""",
        "metadata": {"doc_type": "board_update", "company": "lumina_energy"},
    },
}

DEAL_MEMOS: dict[str, dict] = {
    "nova_investment_memo": {
        "text": """
INVESTMENT MEMO — NOVA ANALYTICS (SERIES A)

Date: January 2023
Stage: Series A
Check Size: $4,000,000
Lead: [Fund Name]
Round Total: $12,000,000

COMPANY OVERVIEW
Nova Analytics provides a modern data operations platform for mid-market financial
services companies. Their product automates data pipeline management, monitoring,
and compliance reporting — replacing fragmented spreadsheet workflows that are
common in the $50M–$500M AUM segment.

INVESTMENT THESIS
1. Large underserved market: ~18,000 mid-market financial firms in the US lack
   modern data infrastructure; legacy tools (Excel, Tableau) don't meet growing
   regulatory demands.
2. Product-led growth flywheel: bottom-up adoption through the data team, followed
   by enterprise expansion into compliance and operations.
3. Strong founder-market fit: CEO previously led data engineering at a $2B AUM RIA;
   CTO built Palantir's financial services integration layer.

KEY RISKS
- Sales motion still being validated: only 12 paying customers at time of investment
- Highly regulated vertical; compliance requirements may slow adoption
- Several well-funded competitors (Fivetran, dbt Cloud) adjacent to the space

DEAL TERMS
- Pre-money valuation: $28,000,000
- Fully diluted ownership post-round: 14.3%
- Board seat: Yes (2 founders, 1 lead investor, 1 independent)
""",
        "metadata": {"doc_type": "deal_memo", "company": "nova_analytics"},
    },
    "helix_investment_memo": {
        "text": """
INVESTMENT MEMO — HELIX BIOTECH (SEED)

Date: September 2022
Stage: Seed
Check Size: $1,500,000
Lead: [Fund Name]
Round Total: $5,000,000

COMPANY OVERVIEW
Helix Biotech is building the data infrastructure layer for clinical genomics workflows.
Their platform enables biopharma and clinical research organizations to ingest, harmonize,
and analyze genomic data across disparate lab systems with full HIPAA and GxP compliance.

INVESTMENT THESIS
1. Genomics data volumes growing 40% annually; existing tools (LIMS, EHR integrations)
   not designed for the scale or complexity of modern multi-omics datasets.
2. Regulatory moat: GxP-compliant data handling is technically and legally complex;
   high switching costs once integrated into clinical workflows.
3. Strong early traction: 4 biopharma pilots at seed stage, including one with a top-10
   pharmaceutical company.

KEY RISKS
- Long enterprise sales cycles (9–12 months) create cash flow risk
- Regulatory landscape still evolving (FDA guidance on AI/ML in genomics)
- Small founding team (3 people) — execution risk as they scale

DEAL TERMS
- Pre-money valuation: $9,000,000
- Fully diluted ownership post-round: 13.6%
- Board seat: Observer rights only at seed; full seat triggers at Series A
""",
        "metadata": {"doc_type": "deal_memo", "company": "helix_biotech"},
    },
}
