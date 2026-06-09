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

Using gpt-4o-mini as the answering model, CogBase scores **92.8%** overall on the LLM-judge
metric (categories 1–4), compared to Mem0's **91.6%** as of April 2026. CogBase leads on
single-hop, multi-hop, and open-domain; Mem0 leads on temporal reasoning (92.8% vs 87.5%).
See [Token usage](#token-usage-and-model-trade-off) for the accuracy/cost trade-off across
answering models.

## How it works

For each conversation:
1. Creates a CogBase app named `locomo-{conv-id}` (one app = one person's memory)
2. Uploads each session as a separate document with turn IDs embedded: `[D3:7] Speaker: text`
3. Waits for chunk-embed-upsert ingestion to complete
4. Queries each QA pair via `POST /applications/{name}/query`
5. Tracks which sessions appear in retrieved chunks for recall scoring

## Score

The recommended scoring method is the built-in LLM judge — it produces binary CORRECT/WRONG
verdicts on the same scale as Mem0 and other systems. Pass `--judge_model` during the run
(see [Run](#run)), then print the summary:

```
python benchmarks/locomo/run_cogbase.py \
    --out_file benchmarks/locomo/results/locomo10_cogbase.json \
    --summary_only
```

Example output with gpt-4o-mini as both answering and judge models:
```
LLM Judge results  (1540 questions judged)
Category                    N   Correct   Accuracy
────────────────────────────────────────────────────
  Single-hop              841       793      94.3%
  Multi-hop               282       268      95.0%
  Temporal                321       281      87.5%
  Open-domain              96        87      90.6%
────────────────────────────────────────────────────
  Overall                1540      1429      92.8%
```

Category 5 (adversarial) is excluded from judge scoring by default.

### Head-to-head comparison (LLM judge, categories 1–4)

| System | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
|---|---|---|---|---|---|
| **CogBase** | **94.3%** | **95.0%** | 87.5% | **90.6%** | **92.8%** |
| **Mem0** (Apr 2026) | 92.3% | 93.3% | **92.8%** | 76.0% | 91.6% |

Source: [mem0 memory evaluation docs](https://docs.mem0.ai/core-concepts/memory-evaluation).
Mem0 reports a mean of 6,956 tokens per query. See [Token usage](#token-usage-and-model-trade-off)
for CogBase's per-query token cost.

CogBase leads on single-hop, multi-hop, and open-domain recall. Mem0 leads on temporal
reasoning — likely due to its ADD-only memory model preserving chronological ordering.
Scores are comparable only when Mem0 is run **without** `--with-evidence` (see [Judge
prompt comparison](#judge-prompt-comparison-with-mem0) below).

## Token usage and model trade-off

The query runner counts the input and output tokens for each query and returns them in
`QueryResponse`, so the benchmark can report per-query token cost alongside accuracy.

The answering model is **not** a CLI flag — it is configured server-side via
system config file or `POST /system/config` (`--judge_model` only selects the model that grades answers).
The two runs below use the same command and the same `gpt-4o-mini` judge; only the server's answering
model differs. Both cover the first 2 conversations (233 questions):

| Answering model | Tokens / query | Accuracy | vs Mem0 (6,956 tok) |
|---|---|---|---|
| `gpt-4o-mini`  | 11,066 | **95.3%** | +59% tokens |
| `gpt-5.4-mini` |  5,831 | 92.7% | −16% tokens |

`gpt-4o-mini` spends more tokens — the runner pulls in extra retrieved context — and reaches
the highest accuracy. `gpt-5.4-mini` reaches comparable accuracy at roughly half the token
cost, coming in under Mem0's mean.

```
python benchmarks/locomo/run_cogbase.py \
    --data_file locomo/data/locomo10.json \
    --out_file benchmarks/locomo/results/gpt4omini_2conv_tokens.json \
    --judge_model gpt-4o-mini --conversations 2
```

<details>
<summary><code>gpt-4o-mini</code> — full breakdown</summary>

```
Token usage  (233 questions)
Category                    N   Avg Input   Avg Output
────────────────────────────────────────────────────────
  Single-hop              114        9073          157
  Multi-hop                43       13496          259
  Temporal                 63       12066          127
  Open-domain              13       15654          260
────────────────────────────────────────────────────────
  Overall                 233       11066          173

LLM Judge results  (233 questions judged)
Category                    N   Correct   Accuracy
────────────────────────────────────────────────────
  Single-hop              114       109      95.6%
  Multi-hop                43        42      97.7%
  Temporal                 63        58      92.1%
  Open-domain              13        13     100.0%
────────────────────────────────────────────────────
  Overall                 233       222      95.3%
```
<summary><code>gpt-4o-mini</code> — round 2</summary>
```
Token usage  (233 questions)
Category                    N   Avg Input   Avg Output
────────────────────────────────────────────────────────
  Single-hop              114        9231          159
  Multi-hop                43       12053          264
  Temporal                 63       11946          128
  Open-domain              13       10902          202
────────────────────────────────────────────────────────
  Overall                 233       10579          172

LLM Judge results  (233 questions judged)
Category                    N   Correct   Accuracy
────────────────────────────────────────────────────
  Single-hop              114       112      98.2%
  Multi-hop                43        40      93.0%
  Temporal                 63        55      87.3%
  Open-domain              13        13     100.0%
────────────────────────────────────────────────────
  Overall                 233       220      94.4%
```
</details>

<details>
<summary><code>gpt-5.4-mini</code> — full breakdown</summary>

```
Token usage  (233 questions)
Category                    N   Avg Input   Avg Output
────────────────────────────────────────────────────────
  Single-hop              114        5189           93
  Multi-hop                43        5938          148
  Temporal                 63        7011           94
  Open-domain              13        5400          133
────────────────────────────────────────────────────────
  Overall                 233        5831          106

LLM Judge results  (233 questions judged)
Category                    N   Correct   Accuracy
────────────────────────────────────────────────────
  Single-hop              114       107      93.9%
  Multi-hop                43        38      88.4%
  Temporal                 63        59      93.7%
  Open-domain              13        12      92.3%
────────────────────────────────────────────────────
  Overall                 233       216      92.7%
```
</details>

## Comparison with other memory systems

All systems isolate each conversation — there is no cross-conversation testing in the benchmark.

| System | Isolation | Scoring | Category 5 |
|---|---|---|---|
| **CogBase** (this) | One app per conversation (`locomo-conv-26`, …) | LLM-as-judge (`--judge_model`) | Excluded by default |
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

## Setup

Start CogBase first:
```
./server/docker_hub_demo.sh run $version
```

Dependencies (already in the project requirements):
```
pip install httpx pyyaml nltk
```

The answering model is set on the server, not by this script. Configure it at runtime via
`POST /system/config` (no restart required) before running the benchmark.

## Run

Quick test — first conversation, first 5 questions:
```
python benchmarks/locomo/run_cogbase.py \
    --data_file locomo/data/locomo10.json \
    --out_file benchmarks/locomo/results/locomo10_cogbase.json \
    --base_url http://localhost:8000 \
    --conversations 1 --sample 5
```

Full run (all 10 conversations, ~1986 questions):
```
python benchmarks/locomo/run_cogbase.py \
    --data_file locomo/data/locomo10.json \
    --out_file benchmarks/locomo/results/locomo10_cogbase.json \
    --base_url http://localhost:8000
```

With LLM judge (adds `cogbase_judge_label`/`cogbase_judge_score` to each QA entry):
```
python benchmarks/locomo/run_cogbase.py \
    --data_file locomo/data/locomo10.json \
    --out_file benchmarks/locomo/results/locomo10_cogbase.json \
    --base_url http://localhost:8000 \
    --judge_model gpt-4o-mini
```

Judge options:
- `--judge_model MODEL` — LLM for binary CORRECT/WRONG judgment (e.g. `gpt-4o-mini`)
- `--judge_provider openai|anthropic` — provider for the judge (default: `openai`)
- `--categories 1,2,3,4` — categories to judge (default: `1,2,3,4`; category 5 adversarial excluded)
- `--summary_only` — load `--out_file` and print the judge summary table without running any queries

Print judge summary from an existing output file:
```
python benchmarks/locomo/run_cogbase.py \
    --out_file benchmarks/locomo/results/locomo10_cogbase.json \
    --summary_only
```

Each judged QA entry gains three extra fields: `cogbase_judge_label` (CORRECT/WRONG),
`cogbase_judge_score` (1.0/0.0), and `cogbase_judge_reasoning` (one-sentence explanation).

The run is **resumable**: re-running with the same `--out_file` skips already-answered
questions. Adding `--judge_model` on a resume run backfills verdicts for previously answered
questions that lack them. Predictions are checkpointed every 20 questions.
