# cogbase.pipeline

The pipeline module handles ingest-time processing: chunking documents into vector-searchable passages, extracting structured facts via LLM, and storing document-level summaries. It is asynchronous throughout and designed around pluggable chunkers, extractors, and step-level prompts.

## Structure

```
pipeline/
├── ingestion_pipeline.py   # orchestration — IngestionPipeline, PipelineStep, VectorCollection, StructuredCollection
├── document_parser.py      # parse_to_markdown — convert uploaded files to markdown text via markitdown
├── extraction/
│   ├── base.py             # ExtractorBase abstract class
│   └── llm.py              # LLMExtractor — LLM-backed structured extraction
└── chunking/
    ├── base.py             # ChunkerBase abstract class + _make_chunk helper
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

Only doc metadata keys listed in `VectorCollectionSchema.metadata_fields` are copied onto chunks and document-level vectors — all other metadata keys are dropped at ingest time.

### Pipeline routing

`IngestionPipeline` accepts a `match: dict[str, str] | None` parameter. When set, the pipeline only processes documents whose metadata contains every key/value pair in `match`. This is the mechanism for routing documents to the right pipeline in a multi-pipeline app (e.g. `match={"doc_type": "contract"}`). `None` (the default) matches all documents.

### Parallel steps

By default steps run sequentially. Set `parallel=True` on the pipeline to run all steps for each document concurrently via `asyncio.gather`. Use this when steps are independent and latency matters more than throughput fairness.

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
from cogbase.pipeline.chunking.fixed import FixedSizeChunker

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

## Document parsing

`document_parser.parse_to_markdown(content: bytes, filename: str) -> str` converts uploaded file bytes to a markdown string using `markitdown`. Supported formats: PDF, DOCX, PPTX, XLSX, XLS, HTML, XML, JSON, CSV, plain text, Outlook MSG, and audio transcription. The extension is inferred from `filename`. Requires `pip install 'markitdown[all]'`.

Call this at the API boundary to turn raw uploads into `Document.text` before passing documents to the pipeline.

## Extension points

**Custom chunker** — subclass `ChunkerBase` and implement `chunk(doc) -> list[Chunk]`. Use the provided `_make_chunk(doc, index, text, char_offset, char_length)` helper to construct each `Chunk` — it generates the `chunk_id` (`"{doc_id}_{index}"`) and leaves `metadata={}` for the pipeline to populate from `doc.metadata`.

**LangChain splitter** — wrap any `TextSplitter` with `LangChainChunker`. `LangChainChunker` configures the splitter with sentence-boundary separators (`.`, `?`, `!`, `。`) so splits occur between sentences rather than mid-word or mid-sentence. Chinese text is supported via the `。` separator.

**Custom extractor** — subclass `ExtractorBase` and implement `_extract_once(doc) -> list[BaseModel] | None`. Return `None` to trigger a retry, an empty list for no results, or a list of Pydantic records on success.
