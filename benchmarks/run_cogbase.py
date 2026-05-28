"""GraphRAG-Benchmark adapter for CogBase.

Ingests each corpus into a dedicated CogBase application, runs all QA questions
through the query endpoint, and writes results in the format expected by the
benchmark's generation_eval and retrieval_eval scripts.

Usage:
    python benchmarks/run_cogbase.py \\
        --config benchmarks/bench_app_simple.yaml \\
        --subset novel \\
        --base_url http://localhost:8000 \\
        --dataset_dir ./GraphRAG-Benchmark/Datasets \\
        --output_dir ./benchmarks/results \\
        [--sample 20]

Output: benchmarks/results/{config_stem}/{subset}/{corpus_name}/predictions_{corpus_name}.json

How it works:
  1. For each corpus, creates a CogBase app named bench-{slug} via POST /applications (skips if already exists)
  2. Uploads the corpus text via POST /applications/{name}/upload_documents
  3. Polls GET /applications/{name}/tasks until ingestion completes
  4. Queries each QA pair via POST /applications/{name}/query
  5. Writes results/{subset}/{corpus_name}/predictions_{corpus_name}.json in the benchmark's required format

To run:
  # Start CogBase first, then:
  python benchmarks/run_cogbase.py --config benchmarks/bench_app_simple.yaml --subset novel --corpora 1 --sample 5
  python benchmarks/run_cogbase.py --config benchmarks/bench_app_extraction.yaml --subset novel

  # Score with the benchmark's eval scripts:
  python -m Evaluation.generation_eval --data_file benchmarks/results/{config_stem}/novel/Novel-30752/predictions_Novel-30752.json ...


Cross-corpus final score:
  The eval script takes one flat merged JSON file, not per-corpus files. The intended flow is:
  
  Step 1: Merge all per-corpus predictions into one file
  python benchmarks/merge_results.py --app bench_app_simple --subset novel
 
  Step 2: Run eval once on the merged file
  python -m Evaluation.generation_eval \
    --mode API \
    --model gpt-5.4-mini \
    --data_file /your-path/benchmarks/results/bench_app_simple/novel_all.json \
    --output_file /your-path/benchmarks/results/bench_app_simple/novel_gen_scores.json

  Step 3: What the output looks like
  {
    "Fact Retrieval":         {"rouge_score": 0.42, "answer_correctness": 0.61},
    "Complex Reasoning":      {"rouge_score": 0.31, "answer_correctness": 0.54},
    "Contextual Summarize":   {"answer_correctness": 0.58, "coverage_score": 0.63},
    "Creative Generation":    {"answer_correctness": 0.55, "coverage_score": 0.60, "faithfulness": 0.71}
  }
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

QUERY_PROMPT = """
You are a document Q&A assistant. Generate concise answers based strictly on the retrieved evidence.
No preamble ("Based on the documents…", "According to the text…").
No inline citations or source references in the answer text.
"""

"""
  query_prompt: |
    You are a document Q&A assistant. Answer each question using retrieved evidence only.

    Match your response to the question type:
    - Fact retrieval / reasoning: 1–3 sentences, plain and direct.
    - Contextual summarize: a cohesive prose paragraph (2–4 sentences); synthesize the
      argument or evidence structure rather than listing facts; capture nuance and
      qualifications (e.g., "although X, the text argues Y").
    - Creative generation (diary, letter, story): write in the requested form and voice;
      blend evidence naturally; stay focused on the specific angle the question asks for;
      aim for a short, complete paragraph.

    All answers:
    - No preamble ("Based on the documents…", "According to the text…").
    - Do not invent facts absent from the retrieved evidence.
    - No inline citations or source references in the answer text.
    - No trailing summaries or meta-commentary.
"""

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


async def wait_for_ingestion(client: httpx.AsyncClient, app_name: str, poll_interval: float = 3.0) -> None:
    """Poll until all ingest tasks for the app are done or failed."""
    log.info("Waiting for ingestion to complete for '%s' …", app_name)
    while True:
        resp = await client.get(f"/applications/{app_name}/tasks", params={"task_type": "ingest"})
        resp.raise_for_status()
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


async def query(client: httpx.AsyncClient, app_name: str, question: str, system_prompt: str) -> tuple[str, list[str]]:
    """Run a question through the CogBase query endpoint.

    Returns (answer, context) where context combines chunk texts, document
    slice texts, and JSON-serialized structured records.
    """
    resp = await client.post(
        f"/applications/{app_name}/query",
        json={"text": question, "system_prompt": QUERY_PROMPT},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    answer: str = data.get("answer", "")
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

    await ensure_app(client, config_path, app_name)
    task_ids = await upload_corpus(client, app_name, corpus_name, corpus_text)
    if task_ids:
        await wait_for_ingestion(client, app_name)

    if question_type:
        questions = [q for q in questions if q.get("question_type") == question_type]
        log.info("Filtered to question_type='%s': %d question(s)", question_type, len(questions))

    if sample:
        questions = questions[:sample]

    results = []
    for i, q in enumerate(questions, 1):
        log.info("  [%d/%d] %s", i, len(questions), q["question"][:80])
        try:
            answer, context = await query(client, app_name, q["question"], system_prompt)
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

    out_dir = output_dir / corpus_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"predictions_{corpus_name}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log.info("Saved %d predictions → %s", len(results), out_file)


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
        corpora: list[dict] = json.load(f)
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
