"""Legal contract analyst pack for CogBase.

Quick start::

    from packs.legal import LegalContractApp, IngestResult
    from cogbase.core.models import Document

    app = LegalContractApp(client=client, model="claude-sonnet-4-6", structured_store=store)
    await app.setup()

    results = await app.ingest_many([
        Document(doc_id="vendor-001", text=vendor_text),
        Document(doc_id="nda-002",    text=nda_text),
    ])
    result = await app.query("which contracts expire before 2026-01-01?")
"""

from packs.legal.app import IngestResult, LegalContractApp
from packs.legal.extractor import ContractExtractor
from packs.legal.schema import (
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
