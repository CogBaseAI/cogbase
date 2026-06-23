# GraphRAG-Benchmark Runner for CogBase

Ingests each corpus into a dedicated CogBase application, runs all QA questions through the query endpoint, and writes results in the format expected by the benchmark's `generation_eval` script.

CogBase places **3rd on the Novel leaderboard** and **2nd on the Medical leaderboard** of the public [GraphRAG-Bench](https://graphrag-bench.github.io/), using only simple chunking plus LLM-driven inference — no knowledge graph. These results align with [docs/knowledge-graph-decision.md](docs/knowledge-graph-decision.md).

Full result files are on [Google Drive](https://drive.google.com/drive/u/0/folders/1KPUk6nMQUeyPMkM5prkVLgEQJv-K6CuB) — download and place under `benchmarks/graphrag/results/`.

## Results

CogBase uses simple chunking (`bench_app_simple`) and an LLM agent loop (with `vector_search` and `read_document` tools). Scores are average answer correctness, scored by the benchmark's `generation_eval` script.

| Subset  | Config              | Ingest model  | Query model   | Avg. correctness | Leaderboard            |
|---------|---------------------|---------------|---------------|------------------|------------------------|
| Novel   | `bench_app_simple`  | gpt-4o-mini   | gpt-4o-mini   | **58.62**        | 3rd (leaders 63.72 / 58.94) |
| Medical | `bench_app_simple`  | gpt-4o-mini   | gpt-4o-mini   | **72.94**        | 2nd (leader 73.30)     |
| Medical | `bench_app_extraction` | gpt-5.4-mini | gpt-5.4-mini | **74.46**        | best CogBase result    |

The leaderboard gaps (Novel: 0.32 below 2nd place; Medical: 0.36 below 1st) are close enough to be within test deviation.

### Novel

`bench_app_simple` with gpt-4o-mini for both ingest and query achieves **58.62**:

```
python benchmarks/graphrag/print_scores.py benchmarks/graphrag/results/bench_app_simple_novels_gpt4omini/novel_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.2794, "answer_correctness": 0.5456}
  Complex Reasoning:  {"rouge_score": 0.1836, "answer_correctness": 0.5128}
  Contextual Summarize:  {"answer_correctness": 0.7061, "coverage_score": 0.5295}
  Creative Generation:  {"answer_correctness": 0.5804, "coverage_score": 0.354, "faithfulness": 0.3601}

Average Answer Correctness: 0.5862
```

On a 5-corpus subset, gpt-5.4-mini at query raises this to **61.79**. LLM-as-judge evaluation on the same 5 corpora reaches the highest correctness, **69.75** — see [llm_evaluation/README.md](llm_evaluation/README.md).

### Medical

`bench_app_simple` with gpt-4o-mini for both ingest and query achieves **72.94**:

```
python benchmarks/graphrag/print_scores.py benchmarks/graphrag/results/bench_app_simple_medical_gpt4omini/medical_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.285, "answer_correctness": 0.6648}
  Complex Reasoning:  {"rouge_score": 0.2095, "answer_correctness": 0.7263}
  Contextual Summarize:  {"answer_correctness": 0.798, "coverage_score": 0.6394}
  Creative Generation:  {"answer_correctness": 0.7286, "coverage_score": 0.4582, "faithfulness": 0.1605}

Average Answer Correctness: 0.7294
```

The best Medical result, **74.46**, comes from `bench_app_extraction` with gpt-5.4-mini for both ingest and query (over all 300 questions). We use gpt-5.4-mini for extraction at ingest because the medical corpora exceed gpt-4o-mini's max context. **Creating this extraction app takes just a natural-language description** — the CogBase app generator (or Claude Code) writes the full config from sample corpus and questions, no hand-authored schema required (see [Build the app config](#2-build-the-app-config-extraction-only)).

```
python benchmarks/graphrag/print_scores.py benchmarks/graphrag/results/bench_app_extraction_medical_gpt54mini/medical_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.4238, "answer_correctness": 0.7207}
  Complex Reasoning:  {"rouge_score": 0.2644, "answer_correctness": 0.7105}
  Contextual Summarize:  {"answer_correctness": 0.8484, "coverage_score": 0.6806}
  Creative Generation:  {"answer_correctness": 0.6987, "coverage_score": 0.4773, "faithfulness": 0.6413}

Average Answer Correctness: 0.7446
```

The query model matters more than the ingest model for this subset:

| Ingest model | Query model | Config                 | Avg. correctness |
|--------------|-------------|------------------------|------------------|
| gpt-5.4-mini | gpt-5.4-mini | `bench_app_extraction` | 74.46           |
| gpt-4o-mini  | gpt-4o-mini | `bench_app_simple`      | 72.94           |
| gpt-5.4-mini | gpt-4o-mini | `bench_app_extraction` | 71.53           |
| gpt-5.4-mini | gpt-5.4-mini | `bench_app_simple`      | 69.63           |

Notably, on `bench_app_simple` gpt-5.4-mini scores *lower* than gpt-4o-mini (69.63 vs 72.94). And swapping the query model from gpt-5.4-mini down to gpt-4o-mini on `bench_app_extraction` drops the score from 74.46 to 71.53.

<details>
<summary>Detailed scores for the other Medical runs</summary>

`bench_app_simple`, gpt-5.4-mini ingest + query (69.63):
```
python benchmarks/graphrag/print_scores.py benchmarks/graphrag/results/bench_app_simple_medical_gpt54mini/medical_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.4314, "answer_correctness": 0.724}
  Complex Reasoning:  {"rouge_score": 0.2539, "answer_correctness": 0.6422}
  Contextual Summarize:  {"answer_correctness": 0.7101, "coverage_score": 0.7759}
  Creative Generation:  {"answer_correctness": 0.7088, "coverage_score": 0.5869, "faithfulness": 0.4509}

Average Answer Correctness: 0.6963
```

`bench_app_extraction`, gpt-5.4-mini ingest + gpt-4o-mini query (71.53):
```
python benchmarks/graphrag/print_scores.py benchmarks/graphrag/results/bench_app_extraction_medical_ingestgpt54mini_querygpt4omini/medical_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.2556, "answer_correctness": 0.6581}
  Complex Reasoning:  {"rouge_score": 0.136, "answer_correctness": 0.6648}
  Contextual Summarize:  {"answer_correctness": 0.82, "coverage_score": 0.6369}
  Creative Generation:  {"answer_correctness": 0.7183, "coverage_score": 0.5672, "faithfulness": 0.255}

Average Answer Correctness: 0.7153
```

</details>

## Reproducing Results

### 1. Setup

Start CogBase first (see `server/README.md` for details):
```
./server/docker_hub_demo.sh run $version
```

### 2. Build the app config (extraction only)

The `bench_app_simple` config needs no setup. The `bench_app_extraction` config adds an `extract-structured` step, generated per corpus.

Open the CogBase UI and build an app by asking it to extract structured information that helps answer the questions, providing the first 5000 chars of `GraphRAG-Benchmark/Datasets/Corpus/medical.json` and the first 50 Q&A pairs of `GraphRAG-Benchmark/Datasets/Questions/medical_questions.json`.

Or ask Claude Code directly, referencing `benchmarks/graphrag/bench_app_extraction.yaml`. This produces `benchmarks/graphrag/bench_app_extraction_medical.yaml`.

> The generated config may include a `document-embed-upsert` step that summarizes the document into its own vector collection. This is useful for real applications with many documents, but the benchmark has only one corpus per app — you can remove it.

### 3. Run

Validate the setup with a small sample (`--corpora 1 --sample 5` limits to 1 corpus, 5 questions):
```
python benchmarks/graphrag/run_cogbase.py \
    --config benchmarks/graphrag/bench_app_simple.yaml \
    --subset novel \
    --base_url http://localhost:8000 \
    --dataset_dir /your-path/GraphRAG-Benchmark/Datasets \
    --output_dir benchmarks/graphrag/results \
    --corpora 1 \
    --sample 5
```

Full run — omit `--corpora` and `--sample`:
```
python benchmarks/graphrag/run_cogbase.py \
    --config benchmarks/graphrag/bench_app_simple.yaml \
    --subset novel \
    --base_url http://localhost:8000 \
    --dataset_dir /your-path/GraphRAG-Benchmark/Datasets \
    --output_dir benchmarks/graphrag/results
```

### 4. Evaluate

**Merge per-corpus predictions into one file:**
```
python benchmarks/graphrag/merge_results.py --dir benchmarks/graphrag/results/bench_app_simple/novel
```

**Score with the benchmark's eval script.** Pass `--detailed_output` to save per-question scores (required for the print step below); without it, aggregate scores print to stdout.
```
export LLM_API_KEY=sk-xxx

cd /your-path/GraphRAG-Benchmark
python -m Evaluation.generation_eval \
  --mode API \
  --model gpt-5.4-mini \
  --data_file benchmarks/graphrag/results/bench_app_simple/novel_all.json \
  --output_file benchmarks/graphrag/results/bench_app_simple/novel_scores.json \
  --detailed_output
```

**Print scores from the detailed output:**
```
python benchmarks/graphrag/print_scores.py benchmarks/graphrag/results/bench_app_simple/novel_scores.json
```

## Experiments

### Impact of the `read_document` tool (Novel-30752)

Tested `bench_app_simple` against Novel-30752 with and without the `read_document` tool.

| Variant              | Avg. correctness |
|----------------------|------------------|
| With `read_document` | 0.6246           |
| Without              | 0.6159           |

<details>
<summary>Detailed scores</summary>

**With `read_document`** (0.6246):
```
python benchmarks/print_scores.py benchmarks/graphrag/results/bench_app_simple_5novels_gpt54mini/novel_30752_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.4362, "answer_correctness": 0.834}
  Creative Generation:  {"answer_correctness": 0.6374, "coverage_score": 0.25, "faithfulness": 0.0}
  Contextual Summarize:  {"answer_correctness": 0.5369, "coverage_score": 0.5008}
  Complex Reasoning:  {"rouge_score": 0.2224, "answer_correctness": 0.4902}

Average Answer Correctness: 0.6246
```

**Without `read_document`** (0.6159):
```
python benchmarks/print_scores.py benchmarks/graphrag/results/bench_app_simple_5novels_gpt54mini/novel_30752_no_readdoctool_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.4462, "answer_correctness": 0.7496}
  Creative Generation:  {"answer_correctness": 0.5687, "coverage_score": 0.3333, "faithfulness": NaN}
  Contextual Summarize:  {"answer_correctness": 0.6002, "coverage_score": 0.6555}
  Complex Reasoning:  {"rouge_score": 0.1666, "answer_correctness": 0.545}

Average Answer Correctness: 0.6159
```

</details>

**Finding:** Disabling `read_document` has negligible impact on average correctness (0.6246 → 0.6159), suggesting vector search alone is sufficient for this corpus. Contextual summarization and complex reasoning improve slightly without it, while fact retrieval drops. We did observe the LLM calling `read_document` during the test, and will run more experiments to understand its impact.

### Impact of the memory layer (Novel)

Tested whether **long-term memory** lifts answer correctness, under two memory sources:

- **Gold memory** (upper bound) — distill each corpus's **ground-truth answers** into long-term memory before querying. This is the ceiling: it shows how much memory can help when what it stores is correct.
- **Self-distilled memory** (realistic) — distill each corpus's **own generated answers** from a prior `bench_app_simple` run (gpt-4o-mini ingest + query) into memory, then re-run. This mirrors the system distilling memory from its own closed sessions and reusing it on the next run — no gold answers involved.

In both, queries run with memory `recall` alongside `vector_search`.

```
# Gold memory (upper bound) — distill ground-truth answers:
python benchmarks/graphrag/run_cogbase.py \
    --config benchmarks/graphrag/bench_app_simple.yaml \
    --subset novel \
    --dataset_dir /your-path/GraphRAG-Benchmark/Datasets \
    --output_dir benchmarks/graphrag/results \
    --build_memory

# Self-distilled memory — distill a prior run's own generated answers:
python benchmarks/graphrag/run_cogbase.py \
    --config benchmarks/graphrag/bench_app_simple.yaml \
    --subset novel \
    --dataset_dir /your-path/GraphRAG-Benchmark/Datasets \
    --output_dir benchmarks/graphrag/results \
    --build_memory \
    --memory_from_results benchmarks/graphrag/results/bench_app_simple_novels/novel_all.json
```

| Variant                              | Scope        | Avg. correctness |
|--------------------------------------|--------------|------------------|
| Gold memory (upper bound)            | 5 corpora    | **66.56**        |
| Self-distilled memory (realistic)    | full Novel   | **60.18**        |
| Baseline (RAG-only)                  | full Novel   | 58.62            |

> The gold and self-distilled runs cover different corpus sets (5 corpora vs. the full Novel subset), so their absolute scores aren't directly comparable; gold marks the ceiling, self-distilled the realistic lift over its own full-Novel baseline.

<details>
<summary>Detailed scores</summary>

**Gold memory** (66.56, 5 corpora):
```
python benchmarks/graphrag/print_scores.py benchmarks/graphrag/results/bench_app_simple_memory_gold/novel_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.4805, "answer_correctness": 0.719}
  Complex Reasoning:  {"rouge_score": 0.2208, "answer_correctness": 0.5591}
  Contextual Summarize:  {"answer_correctness": 0.7408, "coverage_score": 0.6205}
  Creative Generation:  {"answer_correctness": 0.6437, "coverage_score": 0.5773, "faithfulness": 0.0}

Average Answer Correctness: 0.6656
```

**Self-distilled memory** (60.18, full Novel):
```
python benchmarks/graphrag/print_scores.py benchmarks/graphrag/results/bench_app_simple/novel_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.344, "answer_correctness": 0.5766}
  Complex Reasoning:  {"rouge_score": 0.2032, "answer_correctness": 0.5363}
  Contextual Summarize:  {"answer_correctness": 0.7195, "coverage_score": 0.567}
  Creative Generation:  {"answer_correctness": 0.5748, "coverage_score": 0.3718, "faithfulness": 0.1234}

Average Answer Correctness: 0.6018
```

**Baseline** (58.62, full Novel) — see [Novel](#novel) above.

</details>

**Finding:** Even self-distilled memory — built from the system's *own* (imperfect) prior answers — lifts average correctness by **+1.56 points** (58.62 → 60.18) on the full Novel subset, with Fact Retrieval (+3.10), Complex Reasoning (+2.35), and Contextual Summarize (+1.34) all improving and Creative Generation flat (−0.56). Gold memory (distilling ground-truth answers) reaches **66.56** on the 5-corpus set, marking the ceiling memory can reach when what it stores is correct. The gain comes from memory surfacing previously-derived facts that vector search alone re-retrieves less reliably.

## Future Work

- **Test full corpora in one app** — each corpus is currently tested in its own isolated app, which the real world won't be. Testing all corpora together in a single application would better reflect cross-document reasoning and reveal how CogBase handles retrieval across a larger, mixed collection.
- **Investigate the `bench_app_extraction` gap** — extraction-based scoring (0.5990) lags `bench_app_simple` (0.6179); worth understanding whether this is a prompt-quality issue, schema design, or a fundamental tradeoff of structured extraction vs. chunk-level retrieval.
- **Memory and Adaptive Engine** — the memory layer is now implemented and an initial Novel run shows a +1.56-point lift (see [Impact of the memory layer](#impact-of-the-memory-layer-novel)); extend this to the Medical subset and, once the adaptive evolution engine lands, measure its impact on answer correctness, latency, and token usage.
- **Stronger model** — current scores use gpt-4o-mini or gpt-5.4-mini; running with a stronger model such as gpt-5.4 would establish an upper bound and is expected to push the leaderboard score higher.
