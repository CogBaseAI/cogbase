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
    result = await app.query("what are the termination clauses?")
"""

from packs.legal.app import IngestResult, LegalContractApp
from packs.legal.extractor import ClauseExtractor
from packs.legal.schema import CLAUSES_COLLECTION, CLAUSES_SCHEMA, Clause

__all__ = [
    "LegalContractApp",
    "IngestResult",
    "ClauseExtractor",
    "Clause",
    "CLAUSES_SCHEMA",
    "CLAUSES_COLLECTION",
]
