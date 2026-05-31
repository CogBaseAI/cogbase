# LoCoMo Benchmark for CogBase

Evaluates CogBase as a long-term conversational memory system using the
[LoCoMo](https://github.com/snap-research/locomo) benchmark (ACL 2024).

LoCoMo contains 10 long-running conversations between two people (19–32 sessions,
370–690 turns each, ~1986 QA pairs total). Questions span five categories:

| Category | Description |
|---|---|
| 1 – Multi-hop | Requires reasoning across multiple sessions |
| 2 – Temporal | Dates and time-ordering questions |
| 3 – Open-domain | General factual questions |
| 4 – Single-hop | Direct lookup in one session |
| 5 – Adversarial | Answer is NOT in the conversation; model should say "not mentioned" |

## How it works

For each conversation:
1. Creates a CogBase app named `locomo-{conv-id}` (one app = one person's memory)
2. Uploads each session as a separate document with turn IDs embedded: `[D3:7] Speaker: text`
3. Waits for chunk-embed-upsert ingestion to complete
4. Queries each QA pair via `POST /applications/{name}/query`
5. Tracks which sessions appear in retrieved chunks for recall scoring

## Setup

Start CogBase first:
```
./server/docker_hub_demo.sh run $version
```

Dependencies (already in the project requirements):
```
pip install httpx pyyaml nltk
```

## Run

Quick test — first conversation, first 5 questions:
```
python benchmarks-locomo/run_cogbase.py \
    --data_file locomo/data/locomo10.json \
    --out_file benchmarks-locomo/results/locomo10_cogbase.json \
    --base_url http://localhost:8000 \
    --conversations 1 --sample 5
```

Full run (all 10 conversations, ~1986 questions):
```
python benchmarks-locomo/run_cogbase.py \
    --data_file locomo/data/locomo10.json \
    --out_file benchmarks-locomo/results/locomo10_cogbase.json \
    --base_url http://localhost:8000
```

With LLM judge (adds `cogbase_judge_label`/`cogbase_judge_score` to each QA entry):
```
python benchmarks-locomo/run_cogbase.py \
    --data_file locomo/data/locomo10.json \
    --out_file benchmarks-locomo/results/locomo10_cogbase.json \
    --base_url http://localhost:8000 \
    --judge_model gpt-4o-mini
```

Judge options:
- `--judge_model MODEL` — LLM for binary CORRECT/WRONG judgment (e.g. `gpt-4o-mini`, `gpt-4o`)
- `--judge_provider openai|anthropic` — provider for the judge (default: `openai`)
- `--categories 1,2,3,4` — categories to judge (default: `1,2,3,4`; category 5 adversarial excluded)
- `--summary_only` — load `--out_file` and print the judge summary table without running any queries

Print judge summary from an existing output file:
```
python benchmarks-locomo/run_cogbase.py \
    --out_file benchmarks-locomo/results/locomo10_cogbase.json \
    --summary_only
```

Each judged QA entry gains three extra fields: `cogbase_judge_label` (CORRECT/WRONG),
`cogbase_judge_score` (1.0/0.0), and `cogbase_judge_reasoning` (one-sentence explanation).

The run is **resumable**: re-running with the same `--out_file` skips already-answered
questions. Adding `--judge_model` on a resume run backfills verdicts for previously answered
questions that lack them. Predictions are checkpointed every 20 questions.

## Score

The recommended scoring method is the built-in LLM judge — it produces binary CORRECT/WRONG
verdicts on the same scale as Mem0 and other systems. Pass `--judge_model` during the run
(see [Run](#run)), then print the summary:

```
python benchmarks/locomo/run_cogbase.py \
    --out_file benchmarks/locomo/results/locomo10_cogbase.json \
    --summary_only
```

Example output:
```
LLM Judge results  (1540 questions judged)
Category               N       Correct   Accuracy
────────────────────────────────────────────────────
  Single-hop          841        xxx      xx.x%
  Multi-hop           282        xxx      xx.x%
  Temporal             96        xxx      xx.x%
  Open-domain         321        xxx      xx.x%
────────────────────────────────────────────────────
  Overall            1540        xxx      xx.x%
```

Category 5 (adversarial) is excluded from judge scoring by default.

## Comparison with other memory systems

Other memory systems that publish LoCoMo results also isolate each conversation
— there is no cross-conversation testing in the benchmark.

| System | Isolation | Scoring | Category 5 |
|---|---|---|---|
| **CogBase** (this) | One app per conversation (`locomo-conv-26`, …) | LLM-as-judge (`--judge_model`, default) or token F1 (`token_f1_score.py`) | Excluded by default (judge) / Included (F1) |
| **Mem0** ([memory-benchmarks](https://github.com/mem0ai/memory-benchmarks)) | One `user_id` per conversation (`locomo_0_<run_id>`) in a shared server | LLM-as-judge (binary CORRECT/WRONG) | Excluded by default (`--categories 1,2,3,4`) |
| **Memobase** ([locomo-benchmark](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md)) | Per-user scoping | LLM judge score | — |

LLM judge scores are comparable across systems **when run without evidence**. If Mem0 is run
with `--with-evidence`, its scores will be higher than CogBase's and should not be compared
directly (see below).

### Judge prompt comparison with Mem0

The CogBase judge (`_JudgeClient` in `run_cogbase.py`) uses the same prompt structure and
system prompt as [mem0-memory-benchmarks](https://github.com/mem0ai/memory-benchmarks), with
the same JSON output format (`reasoning` + `label`). The rule text diverged slightly; CogBase
carries an earlier version that is marginally stricter on edge cases:

| Rule | CogBase | Mem0 |
|---|---|---|
| PARAPHRASES COUNT | Core rule only | Adds food/volunteer-work synonym examples |
| EXTRA DETAIL IS FINE | Core rule only | Adds: same core entity with extra descriptive detail → CORRECT |
| DATE TOLERANCE | 14-day window, 50% duration | Also: `"19 days"` ≈ `"two weeks"`; converting `"last year"` to the actual year is CORRECT |
| SEMANTIC OVERLAP | Core rule only | Adds emotions/valence clause: same positive/negative family → CORRECT |
| SAME REFERENT | Named entities | Extends to characters and concepts; adds "does it identify the same core entity?" framing |
| Evidence support | Not supported | `--with-evidence` injects gold conversation turns; see below |

The rule differences have small practical impact — CogBase's judge is marginally stricter on
paraphrase, emotion, and date edge cases.

**Evidence support is the most significant scoring gap.** Mem0's `--with-evidence` injects
the actual source conversation turns into the judge prompt and adds a rule that is strictly
additive: evidence can only flip WRONG→CORRECT (when the generated answer matches the source
but diverges from the gold label), never CORRECT→WRONG. This means any Mem0 score produced
with `--with-evidence` is inflated relative to CogBase's judge and the two are not directly
comparable. For a fair comparison, ensure Mem0 is run without `--with-evidence`.


## Token F1 scoring (paper methodology)
Token F1 (`token_f1_score.py`) matches the original LoCoMo paper but is not directly comparable to the others — it is
stricter because paraphrased correct answers are penalised.

To score with token-level F1 + Porter stemming, matching the original LoCoMo paper:

```
python benchmarks/locomo/token_f1_score.py \
    --data_file locomo/data/locomo10.json \
    --pred_file benchmarks/locomo/results/locomo10_cogbase.json
```

Example output:
```
CogBase on LoCoMo  (10 conversations, 1986 questions)
Category               N       F1    Recall
──────────────────────────────────────────────────
  Single-hop         841   0.xxx     x.xxx
  Multi-hop          282   0.xxx     x.xxx
  Temporal            96   0.xxx     x.xxx
  Open-domain        321   0.xxx     x.xxx
  Adversarial        446   0.xxx       n/a
──────────────────────────────────────────────────
  Overall           1986   0.xxx
```

Scoring details:
- Categories 1–4: token-level F1 with Porter stemming (same as the LoCoMo paper)
- Category 1 (multi-hop): partial F1 across comma-separated sub-answers
- Category 5 (adversarial): 1.0 if the answer contains "not mentioned" or "no information", else 0.0
- Recall: fraction of evidence sessions present in the retrieved chunks (when context IDs are available)

Token F1 is stricter than an LLM judge — paraphrased correct answers are penalised. Results
are not directly comparable to Mem0's reported ~92% LLM-judge accuracy.
