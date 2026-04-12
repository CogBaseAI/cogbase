"""Legal contract analyst pack for CogBase.

Quick start::

    from packs.legal import LegalContractApp

    app = LegalContractApp(client=client, model="claude-sonnet-4-6", structured_store=store)
    await app.setup()
    await app.ingest(contract_text, doc_id="contract-001")
    result = await app.query("what are the termination clauses?")
"""

from packs.legal.app import LegalContractApp
from packs.legal.extractor import ClauseExtractor
from packs.legal.schema import CLAUSES_COLLECTION, CLAUSES_SCHEMA, Clause

__all__ = [
    "LegalContractApp",
    "ClauseExtractor",
    "Clause",
    "CLAUSES_SCHEMA",
    "CLAUSES_COLLECTION",
]
