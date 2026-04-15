"""Contract analyst pack for legal workflows."""

from packs.legal.contract_analyst.app import IngestResult, LegalContractApp
from packs.legal.contract_analyst.extractor import ContractExtractor
from packs.legal.contract_analyst.schema import (
    CONTRACTS_COLLECTION,
    CONTRACTS_SCHEMA,
    ContractRecord,
    build_contracts_schema,
)

__all__ = [
    "LegalContractApp",
    "IngestResult",
    "ContractExtractor",
    "ContractRecord",
    "CONTRACTS_SCHEMA",
    "CONTRACTS_COLLECTION",
    "build_contracts_schema",
]
