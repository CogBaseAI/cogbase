"""Generate examples/demos.json from the demo Python data and schema files.

Run from the repo root:
    python examples/gen_demos_json.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cogbase.config.config import AppConfig  # noqa: E402
from api.routers.applications import _resolve_file_refs, _serialize_config  # noqa: E402
from examples.contract_analyst_demo.saas_contracts import CONTRACTS as CONTRACT_ANALYST_DOCS  # noqa: E402
from examples.contract_analyst_demo.schema import ContractExtraction, ContractExtractionRecord  # noqa: E402
from examples.contract_compliance_demo.contracts_data import CONTRACTS_DOCUMENTS as COMPLIANCE_CONTRACT_DOCS  # noqa: E402
from examples.contract_compliance_demo.rules_data import RULES_DOCUMENTS as COMPLIANCE_RULE_DOCS  # noqa: E402
from examples.contract_compliance_demo.schema import (  # noqa: E402
    ClauseComplianceFinding,
    ContractClause,
    ContractClauseRecord,
    ContractMetadata,
    ContractMetadataRecord,
)
from examples.vc_portfolio_demo.portfolio_data import BOARD_UPDATES as VC_BOARD_UPDATES  # noqa: E402
from examples.vc_portfolio_demo.portfolio_data import DEAL_MEMOS as VC_DEAL_MEMOS  # noqa: E402
from examples.vc_portfolio_demo.schema import PortfolioKPIExtraction, PortfolioKPIRecord  # noqa: E402

_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent


def _read_config(demo_dir: str, file_refs: dict[str, str]) -> str:
    config_path = _EXAMPLES_DIR / demo_dir / "config.yaml"
    data = yaml.safe_load(config_path.read_text())
    _resolve_file_refs(data, file_refs)
    config = AppConfig.model_validate(data)
    return _serialize_config(config)


def _file_refs_contract_analyst() -> dict[str, str]:
    demo_dir = _EXAMPLES_DIR / "contract_analyst_demo"
    return {
        "contracts_record_schema.json": json.dumps(ContractExtractionRecord.model_json_schema()),
        "contracts_extraction_schema.json": json.dumps(ContractExtraction.model_json_schema()),
        "contracts_prompt.txt": (demo_dir / "contracts_prompt.txt").read_text(),
    }


def _file_refs_contract_compliance() -> dict[str, str]:
    demo_dir = _EXAMPLES_DIR / "contract_compliance_demo"
    return {
        "contract_metadata_record_schema.json": json.dumps(ContractMetadataRecord.model_json_schema()),
        "contract_metadata_extraction_schema.json": json.dumps(ContractMetadata.model_json_schema()),
        "contract_metadata_prompt.txt": (demo_dir / "contract_metadata_prompt.txt").read_text(),
        "contract_clause_record_schema.json": json.dumps(ContractClauseRecord.model_json_schema()),
        "contract_clause_extraction_schema.json": json.dumps(ContractClause.model_json_schema()),
        "contract_clauses_prompt.txt": (demo_dir / "contract_clauses_prompt.txt").read_text(),
        "clause_compliance_findings_schema.json": json.dumps(ClauseComplianceFinding.model_json_schema()),
        "compliance_judge_prompt.txt": (demo_dir / "compliance_judge_prompt.txt").read_text(),
    }


def _file_refs_vc_portfolio() -> dict[str, str]:
    demo_dir = _EXAMPLES_DIR / "vc_portfolio_demo"
    return {
        "kpi_record_schema.json": json.dumps(PortfolioKPIRecord.model_json_schema()),
        "kpi_extraction_schema.json": json.dumps(PortfolioKPIExtraction.model_json_schema()),
        "kpi_extraction_prompt.txt": (demo_dir / "kpi_extraction_prompt.txt").read_text(),
    }


def _docs_from_pairs(items: dict[str, str], metadata: dict) -> list[dict]:
    return [{"doc_id": doc_id, "text": text, "metadata": dict(metadata)} for doc_id, text in items.items()]


def _docs_from_documents(items) -> list[dict]:
    return [{"doc_id": doc.doc_id, "text": doc.text, "metadata": dict(doc.metadata)} for doc in items]


def _docs_from_mapping(items: dict[str, dict]) -> list[dict]:
    return [{"doc_id": doc_id, "text": data["text"], "metadata": dict(data["metadata"])} for doc_id, data in items.items()]


def build_catalog() -> dict:
    config_yaml_ca = _read_config("contract_analyst_demo", _file_refs_contract_analyst())
    config_yaml_cc = _read_config("contract_compliance_demo", _file_refs_contract_compliance())
    config_yaml_vc = _read_config("vc_portfolio_demo", _file_refs_vc_portfolio())

    return {
        "demos": [
            {
                "key": "contract-analyst",
                "name": "contract-analyst",
                "title": "Contract Analyst",
                "description": (
                    "Extract structured facts from SaaS contracts, then query the stored records "
                    "for dates, liability caps, payment terms, and clause text."
                ),
                "config_yaml": config_yaml_ca,
                "docs": _docs_from_pairs(CONTRACT_ANALYST_DOCS, {"doc_type": "contract"}),
                "query_examples": [
                    "Which contracts expire before 2026-01-01?",
                    "Which contracts mention New York law?",
                    "Show the payment terms for the Acme contracts.",
                ],
                "notes": "Deploys the contract-analyst app and ingests five built-in SaaS agreements.",
            },
            {
                "key": "contract-compliance",
                "name": "contract-compliance",
                "title": "Contract Compliance",
                "description": (
                    "Compare incoming contracts against company policy documents and review "
                    "clause-level compliance findings."
                ),
                "config_yaml": config_yaml_cc,
                "docs": _docs_from_documents(COMPLIANCE_RULE_DOCS + COMPLIANCE_CONTRACT_DOCS),
                "query_examples": [
                    "Which clauses are non-compliant on liability?",
                    "Show all findings for contract-002.",
                    "What rules govern breach notification?",
                ],
                "notes": "Ingests five policy documents and three example vendor contracts.",
            },
            {
                "key": "vc-portfolio",
                "name": "vc-portfolio",
                "title": "VC Portfolio Intelligence",
                "description": (
                    "Track board-deck KPIs and compare them against investment memos and LP "
                    "updates across the portfolio."
                ),
                "config_yaml": config_yaml_vc,
                "docs": [*_docs_from_mapping(VC_BOARD_UPDATES), *_docs_from_mapping(VC_DEAL_MEMOS)],
                "query_examples": [
                    "Which companies are burning more than $500K per month?",
                    "What was Nova Analytics' ARR in Q3 2024?",
                    "What are the key risks across the portfolio?",
                ],
                "notes": "Ingests board updates, LP updates, and investment memos for the portfolio demo.",
            },
        ]
    }


if __name__ == "__main__":
    import sys as _sys
    catalog = build_catalog()
    _sys.stdout.write(json.dumps(catalog, indent=2, ensure_ascii=False))
    _sys.stdout.write("\n")
