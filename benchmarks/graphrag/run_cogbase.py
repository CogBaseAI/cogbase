"""GraphRAG-Benchmark adapter for CogBase.

Ingests each corpus into a dedicated CogBase application, runs all QA questions
through the query endpoint, and writes results in the format expected by the
benchmark's generation_eval and retrieval_eval scripts.

Usage:
    python benchmarks/run_cogbase.py \
        --config benchmarks/bench_app_simple.yaml \
        --subset novel \
        --base_url http://localhost:8000 \
        --dataset_dir ./GraphRAG-Benchmark/Datasets \
        --output_dir ./benchmarks/results \
        [--sample 20]

Output: benchmarks/results/{config_stem}/{subset}/{corpus_name}/predictions_{corpus_name}.json

How it works:
  1. For each corpus, creates a CogBase app named bench-{slug} via POST /applications (skips if already exists)
  2. Uploads the corpus text via POST /applications/{name}/upload_documents
  3. Polls GET /applications/{name}/tasks until ingestion completes
  4. Queries each QA pair via POST /applications/{name}/query
  5. Writes results/{subset}/{corpus_name}/predictions_{corpus_name}.json in the benchmark's required format

See benchmarks/README.md for the full evaluation workflow.
"""

import argparse
import asyncio
import io
import json
import logging
import os
import re
import time
import zipfile
from pathlib import Path

import httpx
import yaml

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

SUBSET_FILES = {
    "novel": {
        "corpus": "Corpus/novel.json",
        "questions": "Questions/novel_questions.json",
    },
    "medical": {
        "corpus": "Corpus/medical.json",
        "questions": "Questions/medical_questions.json",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _app_name(config_path: Path, corpus_name: str) -> str:
    """Build a valid CogBase app name from the config file stem and corpus name.

    Example: bench_app_simple.yaml + 'Novel-30752' → 'bench-app-simple-novel-30752'
    """
    config_slug = re.sub(r"[^a-z0-9\-]", "-", config_path.stem.lower()).strip("-")
    corpus_slug = re.sub(r"[^a-z0-9\-]", "-", corpus_name.lower()).strip("-")
    return f"{config_slug}-{corpus_slug}"


def _build_bundle(config_path: Path, app_name: str) -> bytes:
    """Return a ZIP bundle bytes with config.yaml set for the given app name."""
    template = config_path.read_text()
    config_yaml = template.replace("name: PLACEHOLDER", f"name: {app_name}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("config.yaml", config_yaml)
    return buf.getvalue()


_CITATION_RE = re.compile(r"\[[A-Za-z0-9][A-Za-z0-9_\-:]*(?:,\s*[A-Za-z0-9][A-Za-z0-9_\-:]*)*\]")


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


def _group_by_source(questions: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for q in questions:
        grouped.setdefault(q["source"], []).append(q)
    return grouped


# ---------------------------------------------------------------------------
# CogBase API calls
# ---------------------------------------------------------------------------

async def ensure_app(client: httpx.AsyncClient, config_path: Path, app_name: str) -> None:
    """Create the CogBase application if it does not exist yet."""
    resp = await client.get(f"/applications/{app_name}")
    if resp.status_code == 200:
        log.info("App '%s' already exists, reusing.", app_name)
        return

    bundle = _build_bundle(config_path, app_name)
    resp = await client.post(
        "/applications",
        files={"bundle": ("bundle.zip", bundle, "application/zip")},
        timeout=60,
    )
    resp.raise_for_status()
    log.info("Created app '%s' (status=%s)", app_name, resp.json().get("status"))


async def upload_corpus(client: httpx.AsyncClient, app_name: str, corpus_name: str, text: str) -> list[str]:
    """Upload corpus text and return the list of task IDs."""
    resp = await client.get(f"/applications/{app_name}/docs")
    if resp.status_code == 200 and resp.json().get("total", 0) > 0:
        log.info("Corpus '%s' already ingested, skipping upload.", corpus_name)
        return []

    filename = f"{corpus_name}.txt"
    resp = await client.post(
        f"/applications/{app_name}/upload_documents",
        files={"files": (filename, text.encode(), "text/plain")},
        data={"metadata": "{}"},
        timeout=120,
    )
    resp.raise_for_status()
    task_ids: list[str] = resp.json().get("task_ids", [])
    log.info("Uploaded '%s' → %d ingest task(s)", corpus_name, len(task_ids))
    return task_ids


async def wait_for_ingestion(client: httpx.AsyncClient, app_name: str, poll_interval: float = 3.0, max_retries: int = 5) -> None:
    """Poll until all ingest tasks for the app are done or failed."""
    log.info("Waiting for ingestion to complete for '%s' …", app_name)
    consecutive_errors = 0
    while True:
        try:
            resp = await client.get(
                f"/applications/{app_name}/tasks",
                params={"task_type": "ingest"},
                timeout=30,
            )
            resp.raise_for_status()
            consecutive_errors = 0
        except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as exc:
            consecutive_errors += 1
            if consecutive_errors > max_retries:
                raise
            log.warning("Transient error polling tasks (%d/%d): %s", consecutive_errors, max_retries, exc)
            await asyncio.sleep(poll_interval * consecutive_errors)
            continue
        tasks = resp.json().get("tasks", [])
        pending = [t for t in tasks if t["status"] in ("pending", "running")]
        failed = [t for t in tasks if t["status"] == "failed"]
        if failed:
            log.warning("%d ingest task(s) failed for '%s'.", len(failed), app_name)
        if not pending:
            log.info("Ingestion complete for '%s'.", app_name)
            return
        log.info("  %d task(s) still running …", len(pending))
        await asyncio.sleep(poll_interval)


async def query(client: httpx.AsyncClient, app_name: str, question: str, question_type: str, system_prompt: str) -> tuple[str, list[str]]:
    """Run a question through the CogBase query endpoint.

    Returns (answer, context) where context combines chunk texts, document
    slice texts, and JSON-serialized structured records.
    """
    resp = await client.post(
        f"/applications/{app_name}/query",
        json={"text": question, "system_prompt": system_prompt},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    answer: str = _strip_citations(data.get("answer", ""))
    context: list[str] = (
        [c["text"] for c in data.get("chunks", [])]
        + [s["text"] for s in data.get("document_slices", [])]
        + [json.dumps(r, ensure_ascii=False) for r in data.get("structured_records", [])]
    )
    return answer, context


# ---------------------------------------------------------------------------
# Per-corpus processing
# ---------------------------------------------------------------------------

async def process_corpus(
    client: httpx.AsyncClient,
    config_path: Path,
    system_prompt: str,
    corpus_name: str,
    corpus_text: str,
    questions: list[dict],
    output_dir: Path,
    sample: int | None,
    question_type: str | None = None,
) -> None:
    app_name = _app_name(config_path, corpus_name)
    log.info("=== %s → app '%s' ===", corpus_name, app_name)

    out_dir = output_dir / corpus_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"predictions_{corpus_name}.json"

    existing: list[dict] = []
    done_ids: set = set()
    if out_file.exists():
        try:
            existing = json.loads(out_file.read_text())
            done_ids = {r["id"] for r in existing}
            log.info("Resuming '%s': %d question(s) already answered.", corpus_name, len(done_ids))
        except Exception:
            log.warning("Could not parse existing %s — starting fresh.", out_file)

    await ensure_app(client, config_path, app_name)
    task_ids = await upload_corpus(client, app_name, corpus_name, corpus_text)
    if task_ids:
        await wait_for_ingestion(client, app_name)

    if question_type:
        questions = [q for q in questions if q.get("question_type") == question_type]
        log.info("Filtered to question_type='%s': %d question(s)", question_type, len(questions))

    if sample:
        questions = questions[:sample]

    questions = [q for q in questions if q["id"] not in done_ids]
    log.info("%d question(s) remaining to process.", len(questions))

    def _save(results: list[dict]) -> None:
        merged = existing + results
        out_file.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        log.info("Checkpoint: saved %d total predictions → %s", len(merged), out_file)

    results = []
    for i, q in enumerate(questions, 1):
        log.info("  [%d/%d] %s", i, len(questions), q["question"][:80])
        try:
            answer, context = await query(client, app_name, q["question"], q["question_type"], system_prompt)
        except Exception as exc:
            log.warning("  Query failed: %s", exc)
            answer, context = "", []

        results.append({
            "id": q["id"],
            "question": q["question"],
            "source": corpus_name,
            "context": context,
            "evidence": q.get("evidence", ""),
            "question_type": q["question_type"],
            "generated_answer": answer,
            "ground_truth": q.get("answer", ""),
        })

        if i % 50 == 0:
            _save(results)

    _save(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    system_prompt = config.get("query_prompt", "")

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) / config_path.stem / args.subset
    output_dir.mkdir(parents=True, exist_ok=True)

    files = SUBSET_FILES[args.subset]
    corpus_path = dataset_dir / files["corpus"]
    questions_path = dataset_dir / files["questions"]

    with corpus_path.open() as f:
        # GraphRAG-Benchmark/Datasets/Corpus/medical.json has only 1 json record
        _corpora_raw = json.load(f)
        corpora: list[dict] = [_corpora_raw] if isinstance(_corpora_raw, dict) else _corpora_raw
    with questions_path.open() as f:
        questions_raw: list[dict] = json.load(f)

    if args.corpora:
        corpora = corpora[: args.corpora]

    grouped = _group_by_source(questions_raw)

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        for item in corpora:
            corpus_name: str = item["corpus_name"]
            corpus_text: str = item["context"]
            qs = grouped.get(corpus_name, [])
            if not qs:
                log.warning("No questions for corpus '%s', skipping.", corpus_name)
                continue
            out_file = output_dir / corpus_name / f"predictions_{corpus_name}.json"
            if args.skip_existing and out_file.exists():
                log.info("Skipping '%s' — predictions already exist at %s", corpus_name, out_file)
                continue
            await process_corpus(
                client=client,
                config_path=config_path,
                system_prompt=system_prompt,
                corpus_name=corpus_name,
                corpus_text=corpus_text,
                questions=qs,
                output_dir=output_dir,
                sample=args.sample,
                question_type=args.question_type,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG-Benchmark adapter for CogBase")
    parser.add_argument("--config", required=True,
                        help="Path to the app config YAML (e.g. benchmarks/bench_app_simple.yaml)")
    parser.add_argument("--subset", required=True, choices=["novel", "medical"])
    parser.add_argument("--base_url", default="http://localhost:8000",
                        help="CogBase API base URL")
    parser.add_argument("--dataset_dir", default="./GraphRAG-Benchmark/Datasets",
                        help="Path to the Datasets directory from the benchmark repo")
    parser.add_argument("--output_dir", default="./benchmarks/results",
                        help="Directory to write prediction JSON files")
    parser.add_argument("--corpora", type=int, default=None,
                        help="Process only the first N corpora (e.g. --corpora 3)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only the first N questions per corpus (for quick testing)")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip a corpus if its predictions JSON already exists")
    parser.add_argument("--question_type", default=None,
                        help="Filter to a single question_type (e.g. 'Fact Retrieval'); tests all types if omitted")
    args = parser.parse_args()

    asyncio.run(main(args))
