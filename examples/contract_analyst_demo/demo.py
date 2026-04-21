"""Contract Analyst Demo — interactive ingestion and Q&A over SaaS contracts.

Usage
-----
    cd /path/to/cogbase
    python examples/contract_analyst_demo/demo.py

Requires OPENAI_API_KEY in a .env file at the repo root (or in the environment).

Commands (interactive loop)
---------------------------
    ingest saas          Ingest the built-in 5 SaaS contract fixtures
    ingest <path>        Ingest a plain-text contract file from disk
    list                 Show ingested contract IDs
    reset                Delete all stored data and start fresh
    q / quit / exit      Exit

On startup the demo looks for previously persisted data under ./data/ (relative
to this script).  If found, it loads the stores and skips ingestion so you go
straight to Q&A.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys

format = '%(asctime)s [%(levelname)s] %(threadName)s %(filename)s:%(lineno)d - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=format)


# ---------------------------------------------------------------------------
# Repo root on the Python path so we can import cogbase and packs directly.
# ---------------------------------------------------------------------------

_DEMO_DIR = pathlib.Path(__file__).parent.resolve()
_REPO_ROOT = _DEMO_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Load .env — try python-dotenv, fall back to manual parse
# ---------------------------------------------------------------------------

_ENV_FILE = _REPO_ROOT / ".env"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv(_ENV_FILE, override=False)
    except ImportError:
        if _ENV_FILE.exists():
            for line in _ENV_FILE.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

# ---------------------------------------------------------------------------
# Validate API key early
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not _API_KEY:
    print("ERROR: OPENAI_API_KEY not found.")
    print(f"  Add it to {_ENV_FILE} or set it in your environment.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Imports (after sys.path is configured)
# ---------------------------------------------------------------------------

import openai  # noqa: E402

from cogbase.core.models import Document  # noqa: E402
from cogbase.embeddings import OpenAIEmbedding  # noqa: E402
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker  # noqa: E402
from cogbase.stores.structured.memory import InMemoryStructuredStore  # noqa: E402
from cogbase.stores.vector.faiss_store import FAISSVectorStore  # noqa: E402
from cogbase.core.app import CogBaseApp  # noqa: E402
from cogbase.pipeline.extraction.llm import LLMExtractor  # noqa: E402
from cogbase.core.basemodel_to_schema import cls_json_schema_for_llm  # noqa: E402
from examples.contract_analyst_demo.schema import (  # noqa: E402
    CONTRACTS_COLLECTION,
    CONTRACTS_SYSTEM_PROMPT_PREFIX,
    ContractExtraction,
)
from saas_contracts import CONTRACTS  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CHAT_MODEL = "gpt-5-mini"
_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIM = 1536

_DATA_DIR = _DEMO_DIR / "data"
_STRUCTURED_DIR = _DATA_DIR / "structured"
_VECTOR_DIR = _DATA_DIR / "vector"

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_saved_data() -> bool:
    return (
        (_STRUCTURED_DIR / "_schemas.json").exists()
        and (_VECTOR_DIR / "index.faiss").exists()
    )


def _build_app(
    client: openai.AsyncOpenAI,
    store: InMemoryStructuredStore,
    vector_store: FAISSVectorStore,
) -> CogBaseApp:
    embedder = OpenAIEmbedding(client, model=_EMBED_MODEL)
    chunker = FixedSizeChunker(chunk_size=512, overlap=64)
    extractor = LLMExtractor(
        client,
        model=_CHAT_MODEL,
        extraction_model=ContractExtraction,
        collection_name=CONTRACTS_COLLECTION,
        id_field="contract_id",
        system_prompt=CONTRACTS_SYSTEM_PROMPT_PREFIX + cls_json_schema_for_llm(ContractExtraction),
    )
    return CogBaseApp(
        client=client,
        model=_CHAT_MODEL,
        extractors=[extractor],
        structured_store=store,
        vector_store=vector_store,
        embedder=embedder,
        chunker=chunker,
    )


async def _save(store: InMemoryStructuredStore, vector_store: FAISSVectorStore) -> None:
    await store.persist(_STRUCTURED_DIR)
    await vector_store.save(_VECTOR_DIR)
    print(f"  [saved to {_DATA_DIR.relative_to(_REPO_ROOT)}]")


async def _list_contracts(store: InMemoryStructuredStore) -> None:
    try:
        rows = await store.query("contracts")
    except Exception:
        print("  (no contracts collection yet)")
        return
    if not rows:
        print("  (no contracts ingested yet)")
        return
    for row in rows:
        print(f"{row}")
        #doc_id = row.get("doc_id", "?")
        #parties = row.get("parties", "")
        #expires = row.get("expiry_date", "")
        #print(f"  {doc_id:<12}  parties: {parties}  expires: {expires}")


async def _ingest_text(
    app: LegalContractApp,
    store: InMemoryStructuredStore,
    vector_store: FAISSVectorStore,
    doc_id: str,
    text: str,
) -> bool:
    doc = Document(doc_id=doc_id, text=text)
    try:
        await app.ingest(doc)
    except Exception as exc:
        print(f"  ERROR ingesting {doc_id}: {exc}")
        return False
    print(f"  {doc_id}  OK")
    await _save(store, vector_store)
    return True


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------


async def main() -> None:
    print()
    print("Contract Analyst Demo")
    print("=" * 40)
    print(f"  model:    {_CHAT_MODEL}")
    print(f"  embed:    {_EMBED_MODEL}")
    print(f"  data dir: {_DATA_DIR.relative_to(_REPO_ROOT)}")
    print()

    client = openai.AsyncOpenAI(api_key=_API_KEY)
    store = InMemoryStructuredStore()
    vector_store = FAISSVectorStore(dim=_EMBED_DIM)

    # ------------------------------------------------------------------
    # Load persisted state if available — skip ingestion on warm start
    # ------------------------------------------------------------------
    if _has_saved_data():
        print("Loading saved data...")
        await store.load(_STRUCTURED_DIR)
        await vector_store.load(_VECTOR_DIR)
        print(f"  {vector_store.ntotal} chunks in vector store")
        print()
    else:
        print("No saved data found. Use 'ingest saas' to ingest the built-in contracts.")
        print()

    app = _build_app(client, store, vector_store)
    await app.setup()

    # ------------------------------------------------------------------
    # Interactive loop
    # ------------------------------------------------------------------
    print("Commands: ingest saas | ingest <file> | list | reset | q")
    print()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        lower = raw.lower()

        # ---- exit -------------------------------------------------------
        if lower in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        # ---- list -------------------------------------------------------
        if lower == "list":
            await _list_contracts(store)
            continue

        # ---- reset ------------------------------------------------------
        if lower == "reset":
            confirm = input("  Delete all stored data? [y/N] ").strip().lower()
            if confirm == "y":
                import shutil
                if _DATA_DIR.exists():
                    shutil.rmtree(_DATA_DIR)
                print("  Data deleted. Restart the demo to start fresh.")
                break
            continue

        # ---- ingest saas ------------------------------------------------
        if lower == "ingest saas":
            print(f"Ingesting {len(CONTRACTS)} built-in SaaS contracts...")
            documents = [
                Document(doc_id=doc_id, text=text)
                for doc_id, text in CONTRACTS.items()
            ]
            results = await app.ingest_documents(documents, concurrency=3)
            for r in results:
                if r.success:
                    print(f"  {r.doc_id:<12}  OK  ({r.records_extracted} record extracted)")
                else:
                    print(f"  {r.doc_id:<12}  FAILED: {r.error}")
            await _save(store, vector_store)
            continue

        # ---- ingest <file> ----------------------------------------------
        if lower.startswith("ingest "):
            rest = raw[len("ingest "):].strip()
            file_path = pathlib.Path(rest).expanduser()
            if not file_path.is_absolute():
                file_path = pathlib.Path.cwd() / file_path
            if not file_path.exists():
                print(f"  File not found: {file_path}")
                continue
            text = file_path.read_text(errors="replace")
            doc_id = file_path.stem
            print(f"Ingesting {file_path.name} as doc_id={doc_id!r}...")
            await _ingest_text(app, store, vector_store, doc_id, text)
            continue

        # ---- question / anything else -----------------------------------
        print("Thinking...")
        try:
            result = await app.query(raw)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        print("Answer:\n")
        print(result.answer)
        print()


if __name__ == "__main__":
    asyncio.run(main())
