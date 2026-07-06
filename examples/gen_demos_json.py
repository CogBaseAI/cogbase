"""Generate examples/demos.json from the demo Python data and schema files.

Run from the repo root:
    python examples/gen_demos_json.py
"""

from __future__ import annotations

import base64
import json
import pathlib
import sys

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cogbase.config.config import AppConfig, _iter_save_steps  # noqa: E402
from api.routers.applications import _resolve_file_refs  # noqa: E402
from examples.contract_analyst_demo.saas_contracts import CONTRACTS as CONTRACT_ANALYST_DOCS  # noqa: E402
from examples.contract_analyst_demo.schema import ContractExtraction, ContractExtractionRecord  # noqa: E402
from examples.docx_render import DOCX_CONTENT_TYPE, to_docx_bytes  # noqa: E402
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
from examples.legal_case_prep_demo.case_data import CASE_DOCUMENTS as LEGAL_CASE_DOCUMENTS  # noqa: E402
from examples.legal_case_prep_demo.schema import (  # noqa: E402
    CaseDocument,
    CaseDocumentRecord,
    Contradiction,
    ContradictionList,
    Entity,
    EntityRecord,
    EvidenceGap,
    EvidenceGapList,
    Fact,
    FactRecord,
    StructuredDataItem,
    StructuredDataItemRecord,
    TimelineEvent,
    TimelineEventRecord,
)

_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent


def _read_config(demo_dir: str, file_refs: dict[str, str]) -> str:
    config_path = _EXAMPLES_DIR / demo_dir / "config.yaml"
    data = yaml.safe_load(config_path.read_text())
    _resolve_file_refs(data, file_refs)
    config = AppConfig.model_validate(data)
    return config.to_yaml()


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


def _file_refs_legal_case_prep() -> dict[str, str]:
    demo_dir = _EXAMPLES_DIR / "legal_case_prep_demo"
    return {
        "case_document_record_schema.json": json.dumps(CaseDocumentRecord.model_json_schema()),
        "case_document_extraction_schema.json": json.dumps(CaseDocument.model_json_schema()),
        "case_document_prompt.txt": (demo_dir / "case_document_prompt.txt").read_text(),
        "timeline_event_record_schema.json": json.dumps(TimelineEventRecord.model_json_schema()),
        "timeline_event_extraction_schema.json": json.dumps(TimelineEvent.model_json_schema()),
        "timeline_event_prompt.txt": (demo_dir / "timeline_event_prompt.txt").read_text(),
        "entity_record_schema.json": json.dumps(EntityRecord.model_json_schema()),
        "entity_extraction_schema.json": json.dumps(Entity.model_json_schema()),
        "entity_prompt.txt": (demo_dir / "entity_prompt.txt").read_text(),
        "fact_record_schema.json": json.dumps(FactRecord.model_json_schema()),
        "fact_extraction_schema.json": json.dumps(Fact.model_json_schema()),
        "fact_prompt.txt": (demo_dir / "fact_prompt.txt").read_text(),
        "structured_data_record_schema.json": json.dumps(StructuredDataItemRecord.model_json_schema()),
        "structured_data_extraction_schema.json": json.dumps(StructuredDataItem.model_json_schema()),
        "structured_data_prompt.txt": (demo_dir / "structured_data_prompt.txt").read_text(),
        "contradiction_record_schema.json": json.dumps(Contradiction.model_json_schema()),
        "contradiction_list_schema.json": json.dumps(ContradictionList.model_json_schema()),
        "contradiction_judge_prompt.txt": (demo_dir / "contradiction_judge_prompt.txt").read_text(),
        "evidence_gap_record_schema.json": json.dumps(EvidenceGap.model_json_schema()),
        "evidence_gap_list_schema.json": json.dumps(EvidenceGapList.model_json_schema()),
        "evidence_gap_judge_prompt.txt": (demo_dir / "evidence_gap_judge_prompt.txt").read_text(),
    }


def _workflow_save_targets(config_yaml: str, workflow_actions: list[dict]) -> list[dict]:
    """Return metadata linking each structured-save collection to its source params collection.

    Used by the demo UI to detect which docs in params_from_collection are missing from
    the save target, so it can prompt the user to run the workflow for those docs.
    """
    config = AppConfig.from_yaml(config_yaml)
    targets = []
    wf_name_to_action_idx = {wa["name"]: i for i, wa in enumerate(workflow_actions)}
    for wf in config.workflows:
        params_col = wf.params_from_collection.collection
        param_key = next(iter(wf.params_from_collection.params.keys()), None)
        if not param_key:
            continue
        action_idx = wf_name_to_action_idx.get(wf.name, 0)
        for save_step in _iter_save_steps(wf.steps):
            targets.append({
                "save_collection": save_step.collection,
                "params_collection": params_col,
                "param_key": param_key,
                "workflow_action_index": action_idx,
            })
    return targets


def _docs_from_pairs(items: dict[str, str], metadata: dict) -> list[dict]:
    return [{"doc_id": doc_id, "text": text, "metadata": dict(metadata)} for doc_id, text in items.items()]


def _with_docx(docs: list[dict]) -> list[dict]:
    """Attach a rendered .docx upload to each ``{doc_id, text, metadata}`` doc.

    ``text`` is kept for the UI preview; ``upload`` carries the base64-encoded
    Word file the Demos tab uploads (parsed to markdown server-side) so the
    fixtures are ingested as .docx rather than plain text.
    """
    for doc in docs:
        doc["upload"] = {
            "filename": f"{doc['doc_id']}.docx",
            "content_type": DOCX_CONTENT_TYPE,
            "content_b64": base64.b64encode(to_docx_bytes(doc["text"])).decode("ascii"),
        }
    return docs


def _docs_from_documents(items) -> list[dict]:
    return [{"doc_id": doc.doc_id, "text": doc.text, "metadata": dict(doc.metadata)} for doc in items]


def _docs_from_mapping(items: dict[str, dict]) -> list[dict]:
    return [{"doc_id": doc_id, "text": data["text"], "metadata": dict(data["metadata"])} for doc_id, data in items.items()]


def build_catalog() -> dict:
    config_yaml_ca = _read_config("contract_analyst_demo", _file_refs_contract_analyst())
    config_yaml_vc = _read_config("vc_portfolio_demo", _file_refs_vc_portfolio())
    config_yaml_cc = _read_config("contract_compliance_demo", _file_refs_contract_compliance())
    config_yaml_lcp = _read_config("legal_case_prep_demo", _file_refs_legal_case_prep())

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
                "docs": _with_docx(_docs_from_pairs(CONTRACT_ANALYST_DOCS, {"doc_type": "contract"})),
                "query_examples": [
                    "Which contracts expire before 2026-01-01?",
                    "Which contracts mention New York law?",
                    "Show the payment terms for the Acme contracts.",
                ],
                "notes": "Deploys the contract-analyst app and ingests the built-in SaaS agreements as Word .docx files.",
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
                "docs": _with_docx([*_docs_from_mapping(VC_BOARD_UPDATES), *_docs_from_mapping(VC_DEAL_MEMOS)]),
                "query_examples": [
                    "Which companies are burning more than $500K per month?",
                    "What was Nova Analytics' ARR in Q3 2024?",
                    "What are the key risks across the portfolio?",
                ],
                "notes": "Ingests board updates, LP updates, and investment memos as Word .docx files for the portfolio demo.",
            },
            {
                "key": "contract-compliance",
                "name": "contract-compliance",
                "title": "Contract Compliance - Workflow Demo",
                "description": (
                    "Compare incoming contracts against company policy documents and review "
                    "clause-level compliance findings."
                ),
                "config_yaml": config_yaml_cc,
                "docs": _with_docx(_docs_from_documents(COMPLIANCE_RULE_DOCS + COMPLIANCE_CONTRACT_DOCS)),
                "query_examples": [
                    "Which clauses are non-compliant on liability?",
                    "Show all findings for contract-002.",
                    "What rules govern breach notification?",
                ],
                "notes": (
                    "Ingests five policy documents and three example vendor contracts as Word .docx files. "
                    "Check compliance for every clause in each contract against policies."
                ),
                "workflow_actions": [
                    {
                        "name": "check-contract-compliance",
                        "label": "Check Compliance",
                        "param_key": "doc_id",
                        "param_label": "Contract",
                        "param_values": [
                            doc.doc_id
                            for doc in COMPLIANCE_CONTRACT_DOCS
                        ],
                    }
                ],
                "workflow_save_targets": _workflow_save_targets(
                    config_yaml_cc,
                    [{"name": "check-contract-compliance"}],
                ),
            },
            {
                "key": "legal-case-prep",
                "name": "legal-case-prep",
                "title": "Legal Case Preparation - Workflow Demo",
                "description": (
                    "Upload a litigation case bundle — correspondence, contracts, witness "
                    "statements, expert reports, and pleadings — and let the system build "
                    "the working artefacts a lawyer needs: a document inventory, "
                    "chronological timeline, cast of characters, fact matrix, contradiction "
                    "detection, and evidence-gap identification."
                ),
                "config_yaml": config_yaml_lcp,
                "docs": _with_docx(_docs_from_documents(LEGAL_CASE_DOCUMENTS)),
                "query_examples": [
                    "Which documents discuss the delivery on 14 March 2025?",
                    "Who is Sarah Patel and which documents mention her?",
                    "What does Beacon allege about the condition of the valves?",
                    "Summarise the case against Acme based on the witness statement.",
                ],
                "notes": (
                    "Ingests a nine-document fictional commercial dispute (Acme v Beacon) as Word .docx files. "
                    "After ingestion, run the workflows to detect contradictions and "
                    "identify evidence gaps per issue."
                ),
                "workflow_actions": [
                    {
                        "name": "detect-contradictions",
                        "label": "Detect Contradictions",
                        "param_key": "issue",
                        "param_label": "Issue",
                    },
                    {
                        "name": "identify-evidence-gaps",
                        "label": "Identify Evidence Gaps",
                        "param_key": "issue",
                        "param_label": "Issue",
                    },
                ],
                "workflow_save_targets": _workflow_save_targets(
                    config_yaml_lcp,
                    [
                        {"name": "detect-contradictions"},
                        {"name": "identify-evidence-gaps"},
                    ],
                ),
            },
        ]
    }


if __name__ == "__main__":
    import sys as _sys
    catalog = build_catalog()
    _sys.stdout.write(json.dumps(catalog, indent=2, ensure_ascii=False))
    _sys.stdout.write("\n")
