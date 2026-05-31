# GraphRAG-Benchmark Runner for CogBase

Ingests each corpus into a dedicated CogBase application, runs all QA questions through the query endpoint, and writes results in the format expected by the benchmark's `generation_eval` script.

These results are promising and align with the discussion in [docs/knowledge-graph-decision.md](docs/knowledge-graph-decision.md).

Download the test results from [GoogleDrive](https://drive.google.com/drive/u/0/folders/1KPUk6nMQUeyPMkM5prkVLgEQJv-K6CuB), and put
under like `benchmarks/graphrag/results/`

Using only simple chunking (bench_app_simple) and LLM-driven inference (gpt-4o-mini with vector_search and read_document tools), CogBase achieves an average answer correctness of **58.62**, placing 3rd on the [GraphRAG-Bench (Novel) Leaderboard](https://graphrag-bench.github.io/), just below the current leader at **63.72** and **58.94**. 58.62 is very close to 58.94 and could be within test deviation. Experimenting with gpt-5.4-mini on 5 corpora gets **61.79**.

```
python benchmarks/graphrag/print_scores.py benchmarks/graphrag/results/bench_app_simple_novels_gpt4omini/novel_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.2794, "answer_correctness": 0.5456}
  Complex Reasoning:  {"rouge_score": 0.1836, "answer_correctness": 0.5128}
  Contextual Summarize:  {"answer_correctness": 0.7061, "coverage_score": 0.5295}
  Creative Generation:  {"answer_correctness": 0.5804, "coverage_score": 0.354, "faithfulness": 0.3601}

Average Answer Correctness: 0.5862
```

Experimenting with LLM-as-judge on 5 corpora gets higher correctness, **69.75**. See details in [benchmarks/graphrag/llm_evaluation/README.md](benchmarks/graphrag/llm_evaluation/README.md)

## Setup

Start CogBase first:
```
./server/docker_hub_demo.sh run $version
```
See `server/README.md` for details.

## Run

A few samples to validate the setup:
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

Full run (add `--corpora 5` to limit to 5 corpora):
```
python benchmarks/graphrag/run_cogbase.py \
    --config benchmarks/graphrag/bench_app_simple.yaml \
    --subset novel \
    --base_url http://localhost:8000 \
    --dataset_dir /your-path/GraphRAG-Benchmark/Datasets \
    --output_dir benchmarks/graphrag/results
```

## Evaluate

**Step 1: Merge per-corpus predictions into one file**
```
python benchmarks/graphrag/merge_results.py --dir benchmarks/graphrag/results/bench_app_simple/novel
```

**Step 2: Score with the benchmark's eval script**

Add `--detailed_output` to save per-question scores (required for Step 3):
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

Without `--detailed_output`, `generation_eval` prints aggregate scores directly to stdout and Step 3 is not needed.

**Step 3: Print scores from detailed output**

If you ran Step 2 with `--detailed_output`, use `print_scores.py` to summarize:
```
python benchmarks/graphrag/print_scores.py benchmarks/graphrag/results/bench_app_simple/novel_scores.json
```

# Future Work

- **Test full corpora in one app** — each corpus is currently tested in its own isolated app. The real world application won't be this simple. Testing all corpora together in a single application would better reflect cross-document reasoning and reveal how CogBase handles retrieval across a larger, mixed collection.
- **Investigate bench_app_extraction gap** — extraction-based scoring (0.5990) lags bench_app_simple (0.6179); worth understanding whether this is a prompt quality issue, schema design, or a fundamental tradeoff of structured extraction vs. chunk-level retrieval.
- **Memory and Adaptive Engine** — once the memory layer and adaptive evolution engine are implemented, re-run benchmarks to measure the impact on answer correctness, latency, and token usage.
- **Stronger model** — current scores use gpt-4o-mini or gpt-5.4-mini; running with a stronger model such as gpt-5.4 would establish an upper bound and is expected to push the leaderboard score higher.

# Experiments

## Novel-30752: impact of the `read_document` tool

Tested bench_app_simple against Novel-30752 with and without the `read_document` tool.

**With `read_document`** (average correctness: 0.6246):
```
python benchmarks/print_scores.py benchmarks/example_results/bench_app_simple/novel_30752_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.4362, "answer_correctness": 0.834}
  Creative Generation:  {"answer_correctness": 0.6374, "coverage_score": 0.25, "faithfulness": 0.0}
  Contextual Summarize:  {"answer_correctness": 0.5369, "coverage_score": 0.5008}
  Complex Reasoning:  {"rouge_score": 0.2224, "answer_correctness": 0.4902}

Average Answer Correctness: 0.6246
```

**Without `read_document`** (average correctness: 0.6159):
```
python benchmarks/print_scores.py benchmarks/example_results/bench_app_simple/novel_30752_no_readdoctool_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.4462, "answer_correctness": 0.7496}
  Creative Generation:  {"answer_correctness": 0.5687, "coverage_score": 0.3333, "faithfulness": NaN}
  Contextual Summarize:  {"answer_correctness": 0.6002, "coverage_score": 0.6555}
  Complex Reasoning:  {"rouge_score": 0.1666, "answer_correctness": 0.545}

Average Answer Correctness: 0.6159
```

**Finding:** Disabling `read_document` has negligible impact on average correctness (0.6246 → 0.6159), suggesting vector search alone are sufficient for this corpus. Contextual summarization and complex reasoning improve slightly without it, while fact retrieval drops. We did see read_document being called by llm in the test. We will do more tests to understand the impact of read_document tool in the future.
