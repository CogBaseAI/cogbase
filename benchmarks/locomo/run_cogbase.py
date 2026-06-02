"""LoCoMo benchmark adapter for CogBase.

For each of the 10 conversations in locomo10.json, creates a dedicated CogBase
application, ingests each session as a document (preserving timestamps and turn
IDs), then queries every QA pair through the CogBase query endpoint.

Usage:
    # Quick test — first conversation, first 5 questions:
    python benchmarks/locomo/run_cogbase.py \
        --data_file locomo/data/locomo10.json \
        --out_file benchmarks/locomo/results/locomo10_cogbase.json \
        --base_url http://localhost:8000 \
        --conversations 1 --sample 5

    # Full run with LLM judge (categories 1-4, adversarial skipped by default):
    python benchmarks/locomo/run_cogbase.py \
        --data_file locomo/data/locomo10.json \
        --out_file benchmarks/locomo/results/locomo10_cogbase.json \
        --base_url http://localhost:8000 \
        --judge_model gpt-4o-mini

    # Include category 5 adversarial questions (note: judge does not score them):
    python benchmarks/locomo/run_cogbase.py \
        --data_file locomo/data/locomo10.json \
        --out_file benchmarks/locomo/results/locomo10_cogbase.json \
        --base_url http://localhost:8000 \
        --include_adversarial

Resumable: re-running with the same --out_file skips already-answered questions.
When --judge_model is added on a resume run, previously answered questions that
lack a judge verdict are judged automatically.
"""

import argparse
import asyncio
import io
import json
import logging
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import httpx

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

PREDICTION_KEY = "cogbase_prediction"
JUDGE_LABEL_KEY = "cogbase_judge_label"
JUDGE_SCORE_KEY = "cogbase_judge_score"
JUDGE_REASONING_KEY = "cogbase_judge_reasoning"
INPUT_TOKENS_KEY = "cogbase_input_tokens"
OUTPUT_TOKENS_KEY = "cogbase_output_tokens"
QUERY_TIME_KEY = "cogbase_query_time"
CHUNKS_KEY = "cogbase_chunks"
DOCUMENT_SLICES_KEY = "cogbase_document_slices"
APP_CONFIG_PATH = Path(__file__).parent / "locomo_app.yaml"

# ---------------------------------------------------------------------------
# LLM judge (adapted from mem0-memory-benchmarks/benchmarks/locomo/prompts.py)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = (
    "You are evaluating conversational AI memory recall. "
    "Return JSON only with the format requested."
)

_JUDGE_PROMPT_TEMPLATE = """Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. Emotions in the same positive/negative family count as paraphrases: "proud" = "fulfilled" = "accomplished"; "huge success" = "relieved" (all express positive achievement). Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT (e.g., "5 months" matches "six months"). Relative dates match specific dates in the same window.

5. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording or level of detail should not result in WRONG if the underlying concept matches.

6. **SAME REFERENT**: If the generated answer mentions the same named entity as the gold answer, mark CORRECT even if it includes additional details.

7. **FOCUS ON KNOWLEDGE, NOT WORDING**: Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {answer}
Generated answer: {response}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels."""


def _preprocess_answer(category: int, answer: str) -> str:
    """Category 3 (open-domain): use only the first part before semicolon."""
    if category == 3 and ";" in answer:
        return answer.split(";")[0].strip()
    return answer


class _JudgeClient:
    """Minimal async LLM client for binary CORRECT/WRONG judgments."""

    def __init__(self, model: str, provider: str = "openai") -> None:
        self.model = model
        self.provider = provider.lower()
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.AsyncAnthropic()
        else:
            import openai
            self._client = openai.AsyncOpenAI()

    async def judge(
        self, question: str, answer: str, response: str
    ) -> tuple[float, str, str]:
        """Return (score, label, reasoning). score is 1.0 for CORRECT, 0.0 for WRONG."""
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            question=question, answer=answer, response=response
        )
        try:
            if self.provider == "anthropic":
                raw = await self._judge_anthropic(prompt)
            else:
                raw = await self._judge_openai(prompt)
            label = raw.get("label", "").upper()
            correct = label == "CORRECT"
            return (1.0 if correct else 0.0), ("CORRECT" if correct else "WRONG"), raw.get("reasoning", "")
        except Exception as exc:
            log.warning("Judge call failed: %s", exc)
            return 0.0, "ERROR", str(exc)

    async def _judge_openai(self, prompt: str) -> dict:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=256,
        )
        return json.loads(resp.choices[0].message.content or "{}")

    async def _judge_anthropic(self, prompt: str) -> dict:
        resp = await self._client.messages.create(
            model=self.model,
            system=_JUDGE_SYSTEM_PROMPT + "\n\nIMPORTANT: Respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        text = resp.content[0].text if resp.content else "{}"
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0) if m else "{}")


# System prompt injected per-query (overrides the app-level query_prompt).
# This prompt mirrors the ANSWER_GENERATION_PROMPT in mem0 locomo test,
# https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/locomo/prompts.py
SYSTEM_PROMPT = """\
You are answering questions about a long-term personal conversation between two people.
Use your retrieval tools to fetch relevant chunks, then follow these reasoning steps IN ORDER.

## Step 1: SCAN ALL RETRIEVED CHUNKS
Read EVERY retrieved chunk from first to last. For each one that contains information
relevant to the question, note it. Do NOT stop after finding the first relevant chunk —
important details are often scattered across multiple chunks, including ones retrieved
later. Give equal weight to ALL chunks regardless of retrieval rank.

## Step 2: ENTITY VERIFICATION
Confirm each relevant chunk is about the correct person or entity. If the question asks
"What does Person A like?" and a chunk says "Person B likes X", do NOT use that chunk to
answer about Person A. In two-person conversations, both speakers' actions are relevant —
always verify that attribution is correct before using a chunk.

## Step 3: COMBINE AND CROSS-REFERENCE
- COMBINE facts from multiple chunks about the same topic. If one chunk says "won first
  place" and another says "performed a piece titled X", those describe the same event.
- For listing or counting questions, extract EVERY distinct item from ALL chunks. Think
  about what categories of answers are possible, then re-scan for each category.
- For counting questions ("how many times", "how many X"), enumerate each distinct
  instance explicitly with its date or context BEFORE giving a final count. Do not
  estimate — list them out, then count the list.
- Connect related facts across chunks: if one says "nearby lake" and another says
  "Lake Tahoe is great for kayaking", the nearby lake IS Lake Tahoe.

## Step 4: SELECT THE BEST ANSWER
- Do NOT assume the highest-ranked chunk is correct. Compare each candidate's relevance
  to the SPECIFIC question asked.
- ALWAYS choose the MOST SPECIFIC detail available. A proper name, title, or number beats
  a generic description.
- Report what someone actually DID, not what was offered or available to them. "Has not
  tried X yet" means X was NOT done. "Joined X" or "has done X" means it WAS done.
- Re-read the question carefully before answering. If it asks "what aspect/type/kind",
  answer with the specific aspect, not the setting.

## Step 5: TEMPORAL GROUNDING
These conversations took place in 2022–2024. Each chunk includes a session header with
an explicit date, e.g. "Session N (YYYY-MM-DD HH:MM:SS):".
- Use dates explicitly stated in chunk text. Do not invent or estimate dates.
- For "how long" questions, find the start and end dates explicitly, then compute the
  duration. Do not guess.
- When you find MULTIPLE instances of similar events at different dates, enumerate them
  all with their dates BEFORE picking the one the question refers to. Never default to
  the first-mentioned instance — the date context determines the answer.

## Step 6: INCLUSION CHECK
For lists and counts: include all items found unless you have STRONG evidence they are
wrong. The most common mistake is finding relevant items but dropping them due to overly
strict filtering. After enumerating, re-verify each item — check for duplicates (same
event described differently) and ensure you haven't missed items from later chunks.

## Step 7: COMMIT AND ANSWER
Give a direct, specific answer using exact words from the conversation whenever possible.
NEVER return an empty answer when relevant chunks exist — if ANY chunk contains relevant
information, give the best answer from available evidence.
If the information is genuinely not present in any retrieved chunk, say:
"Not mentioned in the conversation."
Do not invent facts not present in the retrieved context.
"""

# Simple prompt like below works, but achieves lower scores.
"""
  You are answering questions about a long-term personal conversation between two people.
  Answer with a concise phrase using exact words from the conversation whenever possible.
  For temporal questions, include the date shown in the conversation.
  If the information is not found in any retrieved conversation turn, respond with:
  "Not mentioned in the conversation."
  Do not invent facts not present in the retrieved context.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _app_name(sample_id: str) -> str:
    slug = re.sub(r"[^a-z0-9\-]", "-", sample_id.lower()).strip("-")
    return f"locomo-{slug}"


def _build_bundle(app_name: str) -> bytes:
    template = APP_CONFIG_PATH.read_text()
    config_yaml = template.replace("name: PLACEHOLDER", f"name: {app_name}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("config.yaml", config_yaml)
    return buf.getvalue()


def _session_nums(conv: dict) -> list[int]:
    nums = []
    for k in conv:
        if k.startswith("session_") and not k.endswith("_date_time"):
            try:
                nums.append(int(k.split("_")[-1]))
            except ValueError:
                pass
    return sorted(nums)


def _format_session(conv: dict, n: int) -> str:
    """Format one session as plain text with embedded turn IDs and timestamp."""
    date_time = conv.get(f"session_{n}_date_time", "")
    turns = conv.get(f"session_{n}", [])
    lines = [f"Session {n} ({date_time}):"]
    for turn in turns:
        dia_id = turn.get("dia_id", "")
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        line = f"[{dia_id}] {speaker}: {text}"
        if turn.get("blip_caption"):
            line += f" [shares image: {turn['blip_caption']}]"
        lines.append(line)
    return "\n".join(lines)


_DIA_ID_RE = re.compile(r"\[D(\d+):\d+\]")
_SESSION_HEADER_RE = re.compile(r"^Session (\d+) \(", re.MULTILINE)


def _extract_session_ids(chunks: list) -> list[str]:
    """Return sorted 'S{N}' session IDs inferred from retrieved chunk texts."""
    ids: set[str] = set()
    for chunk in chunks:
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        for m in _DIA_ID_RE.finditer(text):
            ids.add(f"S{m.group(1)}")
        for m in _SESSION_HEADER_RE.finditer(text):
            ids.add(f"S{m.group(1)}")
    return sorted(ids)


# ---------------------------------------------------------------------------
# CogBase API calls
# ---------------------------------------------------------------------------

async def ensure_app(client: httpx.AsyncClient, app_name: str) -> None:
    resp = await client.get(f"/applications/{app_name}")
    if resp.status_code == 200:
        log.info("App '%s' already exists, reusing.", app_name)
        return
    bundle = _build_bundle(app_name)
    resp = await client.post(
        "/applications",
        files={"bundle": ("bundle.zip", bundle, "application/zip")},
        timeout=60,
    )
    resp.raise_for_status()
    log.info("Created app '%s'.", app_name)


async def ingest_conversation(
    client: httpx.AsyncClient, app_name: str, sample_id: str, conv: dict
) -> list[str]:
    resp = await client.get(f"/applications/{app_name}/docs")
    if resp.status_code == 200:
        body = resp.json()
        total = body.get("total", 0) if isinstance(body, dict) else len(body)
        if total > 0:
            log.info("'%s' already has %d doc(s), skipping upload.", sample_id, total)
            return []

    session_nums = _session_nums(conv)
    files = [
        ("files", (f"session_{n:02d}.txt", _format_session(conv, n).encode(), "text/plain"))
        for n in session_nums
    ]
    resp = await client.post(
        f"/applications/{app_name}/upload_documents",
        files=files,
        data={"metadata": "{}"},
        timeout=300,
    )
    resp.raise_for_status()
    task_ids: list[str] = resp.json().get("task_ids", [])
    log.info("Uploaded %d session(s) for '%s' → %d ingest task(s).", len(session_nums), sample_id, len(task_ids))
    return task_ids


async def wait_for_ingestion(
    client: httpx.AsyncClient, app_name: str, poll_interval: float = 3.0, max_retries: int = 5
) -> None:
    log.info("Waiting for ingestion of '%s'…", app_name)
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
        log.info("  %d task(s) still running…", len(pending))
        await asyncio.sleep(poll_interval)


async def query_cogbase(
    client: httpx.AsyncClient, app_name: str, question: str
) -> tuple[str, list[str], list[dict], list[dict], int, int]:
    resp = await client.post(
        f"/applications/{app_name}/query",
        json={"text": question, "system_prompt": SYSTEM_PROMPT},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    answer = data.get("answer", "").strip()
    chunks = data.get("chunks", [])
    document_slices = data.get("document_slices", [])
    context_ids = _extract_session_ids(chunks)
    input_tokens = data.get("input_tokens", 0)
    output_tokens = data.get("output_tokens", 0)
    return answer, context_ids, chunks, document_slices, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Per-conversation processing
# ---------------------------------------------------------------------------

def _save(out_file: Path, all_results: dict) -> None:
    out_file.write_text(json.dumps(list(all_results.values()), indent=2, ensure_ascii=False))


async def process_conversation(
    client: httpx.AsyncClient,
    sample: dict,
    out_file: Path,
    all_results: dict,
    sample_n: int | None,
    judge: "_JudgeClient | None" = None,
    judge_categories: set[int] | None = None,
    include_adversarial: bool = False,
) -> dict:
    sample_id = sample["sample_id"]
    app_name = _app_name(sample_id)
    log.info("=== %s → app '%s' ===", sample_id, app_name)

    await ensure_app(client, app_name)
    task_ids = await ingest_conversation(client, app_name, sample_id, sample["conversation"])
    if task_ids:
        await wait_for_ingestion(client, app_name)

    qas = sample["qa"]
    if not include_adversarial:
        qas = [qa for qa in qas if qa.get("category") != 5]
    if sample_n is not None:
        qas = qas[:sample_n]

    # Index already-answered items from a previous run
    prev_by_key: dict[tuple, dict] = {}
    if sample_id in all_results:
        for qa in all_results[sample_id].get("qa", []):
            if qa.get(PREDICTION_KEY):
                prev_by_key[(qa["question"], qa.get("category"))] = qa

    result_qas: list[dict] = []
    todo = [qa for qa in qas if (qa["question"], qa.get("category")) not in prev_by_key]
    skipped = len(qas) - len(todo)
    if skipped:
        log.info("Resuming '%s': skipping %d already-answered question(s).", sample_id, skipped)

    # Prepend previously answered items in original order
    for qa in qas:
        key = (qa["question"], qa.get("category"))
        if key in prev_by_key:
            result_qas.append(prev_by_key[key])

    new_results: list[dict] = []
    for i, qa in enumerate(todo, 1):
        log.info("  [%d/%d] %s", i, len(todo), qa["question"][:80])
        question = qa["question"]
        if qa.get("category") == 2:
            question += " Use the date shown in the conversation to answer."

        t0 = time.perf_counter()
        try:
            answer, context_ids, chunks, document_slices, input_tokens, output_tokens = await query_cogbase(client, app_name, question)
        except Exception as exc:
            log.warning("  Query failed: %s", exc)
            answer, context_ids, chunks, document_slices, input_tokens, output_tokens = "", [], [], [], 0, 0
        query_time = time.perf_counter() - t0

        result_qa = qa.copy()
        result_qa[PREDICTION_KEY] = answer
        result_qa[PREDICTION_KEY + "_context"] = context_ids
        result_qa[CHUNKS_KEY] = chunks
        result_qa[DOCUMENT_SLICES_KEY] = document_slices
        result_qa[INPUT_TOKENS_KEY] = input_tokens
        result_qa[OUTPUT_TOKENS_KEY] = output_tokens
        result_qa[QUERY_TIME_KEY] = round(query_time, 3)

        if judge and (judge_categories is None or qa.get("category") in judge_categories):
            gt = _preprocess_answer(qa.get("category", 0), str(qa.get("answer", "")))
            score, label, reasoning = await judge.judge(qa["question"], gt, answer)
            result_qa[JUDGE_SCORE_KEY] = score
            result_qa[JUDGE_LABEL_KEY] = label
            result_qa[JUDGE_REASONING_KEY] = reasoning

        new_results.append(result_qa)

        if i % 20 == 0:
            all_results[sample_id] = {"sample_id": sample_id, "qa": result_qas + new_results}
            _save(out_file, all_results)
            log.info("Checkpoint: %d/%d questions saved.", i, len(todo))

    # On resume with a judge model, backfill verdicts for previously answered items.
    if judge and result_qas:
        pending = [
            qa for qa in result_qas
            if PREDICTION_KEY in qa
            and JUDGE_LABEL_KEY not in qa
            and (judge_categories is None or qa.get("category") in judge_categories)
        ]
        if pending:
            log.info("Backfilling judge verdicts for %d previously answered question(s).", len(pending))
            for qa in pending:
                gt = _preprocess_answer(qa.get("category", 0), str(qa.get("answer", "")))
                score, label, reasoning = await judge.judge(
                    qa["question"], gt, qa[PREDICTION_KEY]
                )
                qa[JUDGE_SCORE_KEY] = score
                qa[JUDGE_LABEL_KEY] = label
                qa[JUDGE_REASONING_KEY] = reasoning

    result = {"sample_id": sample_id, "qa": result_qas + new_results}
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_CAT_NAMES = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial"}


def _print_token_summary(all_results: dict) -> None:
    cat_input: dict[int, list[int]] = defaultdict(list)
    cat_output: dict[int, list[int]] = defaultdict(list)
    for d in all_results.values():
        for qa in d.get("qa", []):
            if INPUT_TOKENS_KEY not in qa:
                continue
            cat = qa.get("category", 0)
            cat_input[cat].append(qa[INPUT_TOKENS_KEY])
            cat_output[cat].append(qa[OUTPUT_TOKENS_KEY])

    all_input = [t for v in cat_input.values() for t in v]
    all_output = [t for v in cat_output.values() for t in v]
    if not all_input:
        return

    print(f"\nToken usage  ({len(all_input)} questions)")
    print(f"{'Category':<22} {'N':>6}  {'Avg Input':>10}  {'Avg Output':>11}")
    print("─" * 56)
    for cat in [4, 1, 2, 3, 5]:
        inputs = cat_input.get(cat, [])
        if not inputs:
            continue
        outputs = cat_output.get(cat, [])
        print(
            f"  {_CAT_NAMES.get(cat, str(cat)):<20} {len(inputs):>6}"
            f"  {sum(inputs)/len(inputs):>10.0f}  {sum(outputs)/len(outputs):>11.0f}"
        )
    print("─" * 56)
    print(
        f"  {'Overall':<20} {len(all_input):>6}"
        f"  {sum(all_input)/len(all_input):>10.0f}  {sum(all_output)/len(all_output):>11.0f}"
    )
    print()


def _print_query_time_summary(all_results: dict) -> None:
    cat_times: dict[int, list[float]] = defaultdict(list)
    for d in all_results.values():
        for qa in d.get("qa", []):
            if QUERY_TIME_KEY not in qa:
                continue
            cat = qa.get("category", 0)
            cat_times[cat].append(qa[QUERY_TIME_KEY])

    all_times = [t for v in cat_times.values() for t in v]
    if not all_times:
        return

    print(f"\nQuery time  ({len(all_times)} questions)")
    print(f"{'Category':<22} {'N':>6}  {'Total (s)':>10}  {'Avg (s)':>8}")
    print("─" * 54)
    for cat in [4, 1, 2, 3, 5]:
        times = cat_times.get(cat, [])
        if not times:
            continue
        print(
            f"  {_CAT_NAMES.get(cat, str(cat)):<20} {len(times):>6}"
            f"  {sum(times):>10.1f}  {sum(times)/len(times):>8.2f}"
        )
    print("─" * 54)
    print(
        f"  {'Overall':<20} {len(all_times):>6}"
        f"  {sum(all_times):>10.1f}  {sum(all_times)/len(all_times):>8.2f}"
    )
    print()


def _print_chunks_summary(all_results: dict) -> None:
    cat_chunks: dict[int, list[int]] = defaultdict(list)
    cat_slices: dict[int, list[int]] = defaultdict(list)
    for d in all_results.values():
        for qa in d.get("qa", []):
            if CHUNKS_KEY not in qa:
                continue
            cat = qa.get("category", 0)
            cat_chunks[cat].append(len(qa[CHUNKS_KEY]))
            cat_slices[cat].append(len(qa.get(DOCUMENT_SLICES_KEY, [])))

    all_chunks = [n for v in cat_chunks.values() for n in v]
    if not all_chunks:
        return

    all_slices = [n for v in cat_slices.values() for n in v]
    print(f"\nChunks & document slices  ({len(all_chunks)} questions)")
    print(f"{'Category':<22} {'N':>6}  {'Chunks tot':>10}  {'Chunks avg':>10}  {'Slices tot':>10}  {'Slices avg':>10}")
    print("─" * 76)
    for cat in [4, 1, 2, 3, 5]:
        chunks = cat_chunks.get(cat, [])
        if not chunks:
            continue
        slices = cat_slices.get(cat, [])
        print(
            f"  {_CAT_NAMES.get(cat, str(cat)):<20} {len(chunks):>6}"
            f"  {sum(chunks):>10}  {sum(chunks)/len(chunks):>10.1f}"
            f"  {sum(slices):>10}  {sum(slices)/len(slices) if slices else 0:>10.1f}"
        )
    print("─" * 76)
    print(
        f"  {'Overall':<20} {len(all_chunks):>6}"
        f"  {sum(all_chunks):>10}  {sum(all_chunks)/len(all_chunks):>10.1f}"
        f"  {sum(all_slices):>10}  {sum(all_slices)/len(all_slices) if all_slices else 0:>10.1f}"
    )
    print()


def _print_judge_summary(all_results: dict) -> None:
    cat_total: dict[int, int] = defaultdict(int)
    cat_correct: dict[int, int] = defaultdict(int)
    for d in all_results.values():
        for qa in d.get("qa", []):
            if JUDGE_LABEL_KEY not in qa:
                continue
            cat = qa.get("category", 0)
            cat_total[cat] += 1
            if qa.get(JUDGE_SCORE_KEY, 0.0) >= 0.5:
                cat_correct[cat] += 1

    total_n = sum(cat_total.values())
    if not total_n:
        return

    total_correct = sum(cat_correct.values())
    print(f"\nLLM Judge results  ({total_n} questions judged)")
    print(f"{'Category':<22} {'N':>6}  {'Correct':>8}  {'Accuracy':>9}")
    print("─" * 52)
    for cat in [4, 1, 2, 3, 5]:
        n = cat_total[cat]
        if n == 0:
            continue
        c = cat_correct[cat]
        print(f"  {_CAT_NAMES.get(cat, str(cat)):<20} {n:>6}  {c:>8}  {c/n*100:>8.1f}%")
    print("─" * 52)
    print(f"  {'Overall':<20} {total_n:>6}  {total_correct:>8}  {total_correct/total_n*100:>8.1f}%")
    print()


async def main(args: argparse.Namespace) -> None:
    data_file = Path(args.data_file)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    if out_file.exists():
        try:
            for d in json.loads(out_file.read_text()):
                all_results[d["sample_id"]] = d
            log.info("Loaded existing output: %d conversation(s) have partial predictions.", len(all_results))
        except Exception:
            log.warning("Could not parse existing output — starting fresh.")

    if args.summary_only:
        _print_token_summary(all_results)
        #_print_query_time_summary(all_results)
        _print_chunks_summary(all_results)
        _print_judge_summary(all_results)
        return

    samples: list[dict] = json.loads(data_file.read_text())

    judge: _JudgeClient | None = None
    judge_categories: set[int] | None = None
    if args.judge_model:
        judge = _JudgeClient(model=args.judge_model, provider=args.judge_provider)
        judge_categories = {int(c) for c in args.categories.split(",")}
        log.info(
            "LLM judge enabled: model=%s provider=%s categories=%s",
            args.judge_model, args.judge_provider, args.categories,
        )

    if args.conversations is not None:
        samples = samples[: args.conversations]

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        for sample in samples:
            result = await process_conversation(
                client=client,
                sample=sample,
                out_file=out_file,
                all_results=all_results,
                sample_n=args.sample,
                judge=judge,
                judge_categories=judge_categories,
                include_adversarial=args.include_adversarial,
            )
            all_results[result["sample_id"]] = result
            _save(out_file, all_results)
            log.info("Saved predictions for '%s' → %s", result["sample_id"], out_file)

    answered = sum(
        sum(1 for qa in d.get("qa", []) if PREDICTION_KEY in qa)
        for d in all_results.values()
    )
    log.info("Done. %d total predictions across %d conversation(s).", answered, len(all_results))

    _print_token_summary(all_results)
    #_print_query_time_summary(all_results)
    _print_chunks_summary(all_results)
    if judge:
        _print_judge_summary(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoCoMo benchmark adapter for CogBase")
    parser.add_argument(
        "--data_file", default="locomo/data/locomo10.json",
        help="Path to locomo10.json (default: locomo/data/locomo10.json)",
    )
    parser.add_argument(
        "--out_file", default="benchmarks/locomo/results/locomo10_cogbase.json",
        help="Output predictions file",
    )
    parser.add_argument(
        "--base_url", default="http://localhost:8000",
        help="CogBase API base URL",
    )
    parser.add_argument(
        "--conversations", type=int, default=None,
        help="Process only the first N conversations (default: all 10)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Process only the first N questions per conversation (for quick testing)",
    )
    parser.add_argument(
        "--judge_model", default=None,
        help="LLM model for binary CORRECT/WRONG judgment (e.g. gpt-4o-mini). "
             "If omitted, no judging is performed.",
    )
    parser.add_argument(
        "--judge_provider", default="openai",
        help="LLM provider for judge: openai or anthropic (default: openai)",
    )
    parser.add_argument(
        "--summary_only", action="store_true",
        help="Load --out_file and print the LLM judge summary without running any queries.",
    )
    parser.add_argument(
        "--include_adversarial", action="store_true",
        help="Include category 5 adversarial questions in queries (skipped by default "
             "because the LLM judge does not score them).",
    )
    parser.add_argument(
        "--categories", default="1,2,3,4",
        help="Comma-separated question categories to judge (default: 1,2,3,4; "
             "category 5 adversarial is excluded by default)",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
