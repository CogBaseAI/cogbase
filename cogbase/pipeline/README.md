# cogbase.pipeline

The pipeline module handles ingest-time processing: chunking documents into vector-searchable passages, extracting structured facts via LLM, and storing document-level summaries. It is asynchronous throughout and designed around pluggable chunkers, extractors, and step-level prompts.

## Structure

```
pipeline/
├── ingestion_pipeline.py   # orchestration — IngestionPipeline, PipelineStep, VectorCollection, StructuredCollection
├── extraction/
│   ├── base.py             # ExtractorBase abstract class
│   └── llm.py              # LLMExtractor — LLM-backed structured extraction
└── ingestion/
    ├── base.py             # ChunkerBase abstract class
    ├── fixed.py            # FixedSizeChunker — sliding-window character splitter
    └── langchain.py        # LangChainChunker — adapter for any LangChain TextSplitter
```

## Concepts

### Step types

`IngestionPipeline` runs three step types in declaration order for each document:

| Step | What it does |
|------|-------------|
| `chunk-embed-upsert` | Splits text into overlapping chunks, embeds each, upserts to a vector collection |
| `extract-structured` | Calls `LLMExtractor` to pull typed records, writes them to a structured collection |
| `document-embed-upsert` | Produces one vector record per document — either the full text or an LLM summary |

Steps can be gated by document metadata predicates (`when` dict). A step executes only if all predicate keys match the document's metadata.

### Collections

Two collection wrapper types carry the stores and components each step needs:

- **`VectorCollection`** — `VectorCollectionSchema` + `VectorStoreBase` + `EmbeddingBase`
- **`StructuredCollection`** — `CollectionSchema` + `StructuredStoreBase`

`PipelineStep` binds a tool to one of those collections and supplies any tool-specific component needed at runtime, such as a chunker, extractor, or document-summary prompt.

### Extraction modes

`LLMExtractor` supports two modes:

- **Single-record** (`extract_as_list=False`): the LLM returns one JSON object per document. The model fields are extracted verbatim plus a `doc_id` column.
- **List-record** (`extract_as_list=True`): the LLM returns a JSON array of items (e.g., one record per clause in a contract). Each item gets a `doc_id` and a generated `item_id` of the form `"{doc_id}__{i:04d}"`.

Both modes auto-derive the `CollectionSchema` from the Pydantic model's field types and descriptions.

`ExtractorBase` wraps every extraction attempt in exponential-backoff retry logic (configurable via `max_retries`). A return value of `None` from `_extract_once` signals a retryable failure; an empty list means no records were found and is not retried.

## Usage

```python
from cogbase.pipeline.ingestion_pipeline import (
    IngestionPipeline, PipelineStep, StructuredCollection, VectorCollection,
)
from cogbase.pipeline.ingestion.fixed import FixedSizeChunker

pipeline = IngestionPipeline(
    name="legal",
    steps=[
        PipelineStep(tool="chunk-embed-upsert", collection="document_chunks", chunker=FixedSizeChunker(chunk_size=1000, overlap=200)),
        PipelineStep(tool="extract-structured", collection="contracts"),
        PipelineStep(tool="document-embed-upsert", collection="document_summary", doc_prompt="Summarize this document in a concise way."),
    ],
    vector_collections=[
        VectorCollection(schema=chunk_schema, store=vector_store, embedder=embedder),
        VectorCollection(schema=summary_schema, store=vector_store, embedder=embedder),
    ],
    structured_collections=[
        StructuredCollection(schema=contracts_schema, store=structured_store),
    ],
)

results = await pipeline.ingest_documents(documents, concurrency=5)
```

`ingest_documents` processes documents in parallel (bounded by `concurrency`), returns `IngestResult` objects in input order, and does not abort remaining documents on a single failure.

## Extension points

**Custom chunker** — subclass `ChunkerBase` and implement `chunk(doc) -> list[Chunk]`.

**LangChain splitter** — wrap any `TextSplitter` with `LangChainChunker` (requires `pip install "cogbase[langchain]"`).

**Custom extractor** — subclass `ExtractorBase` and implement `_extract_once(doc) -> list[BaseModel] | None`. Return `None` to trigger a retry, an empty list for no results, or a list of Pydantic records on success.
