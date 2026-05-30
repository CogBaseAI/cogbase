# GraphRAG-Benchmark Runner for CogBase

Ingests each corpus into a dedicated CogBase application, runs all QA questions through the query endpoint, and writes results in the format expected by the benchmark's `generation_eval` script.

These results are promising and align with the discussion in [docs/knowledge-graph-decision.md](docs/knowledge-graph-decision.md).
Using only simple chunking (bench_app_simple) and LLM-driven inference, CogBase (use gpt-5.4-mini) achieves an average answer correctness of **61.79**, placing 2nd on the GraphRAG-Bench (Novel) Leaderboard (https://graphrag-bench.github.io/), just below the current leader at **63.72**.

For an apple-to-apple comparison with other leaderboard entries (which use gpt-4o-mini), the gpt-4o-mini run scores **59.57** — still 2nd place. See more details below.

> Note: 61.79 and 59.57 scores are results across 5 corpora only. Full corpora testing will be run later.

## Setup

Start CogBase first:
```
./server/docker_hub_demo.sh run $version
```
See `server/README.md` for details.

## Run

A few samples to validate the setup:
```
python benchmarks/run_cogbase.py \
    --config benchmarks/bench_app_simple.yaml \
    --subset novel \
    --base_url http://localhost:8000 \
    --dataset_dir /your-path/GraphRAG-Benchmark/Datasets \
    --output_dir benchmarks/results \
    --corpora 1 \
    --sample 5
```

Full run (add `--corpora 5` to test against fewer corporas):
```
python benchmarks/run_cogbase.py \
    --config benchmarks/bench_app_simple.yaml \
    --subset novel \
    --base_url http://localhost:8000 \
    --dataset_dir /your-path/GraphRAG-Benchmark/Datasets \
    --output_dir benchmarks/results
```

## Evaluate

**Step 1: Merge per-corpus predictions into one file**
```
python benchmarks/merge_results.py --dir benchmarks/results/bench_app_simple/novel
```

**Step 2: Score with the benchmark's eval script**
```
export LLM_API_KEY=sk-xxx

cd /your-path/GraphRAG-Benchmark
python -m Evaluation.generation_eval \
  --mode API \
  --model gpt-5.4-mini \
  --data_file /your-path/benchmarks/results/bench_app_simple/novel_all.json \
  --output_file /your-path/benchmarks/results/bench_app_simple/novel_scores.json \
  --detailed_output
```

Add `--detailed_output` to see per-question scores.

## Get Evaluation Scores

If Evaluation runs with --detailed_output, run
```
python benchmarks/print_scores.py /your-path/GraphRAG-Benchmark/bench_app_simple/novel_scores.json
```

Example Output
```
Results:
  Fact Retrieval:  {"rouge_score": 0.406, "answer_correctness": 0.7763}
  Creative Generation:  {"answer_correctness": 0.5716, "coverage_score": 0.4393, "faithfulness": NaN}
  Contextual Summarize:  {"answer_correctness": 0.5834, "coverage_score": 0.6469}
  Complex Reasoning:  {"rouge_score": 0.255, "answer_correctness": 0.5401}

Average Answer Correctness: 0.6179
```

## Test with gpt-4o-mini

Most other leaderboard entries use gpt-4o-mini. To compare on equal footing, CogBase was restarted with gpt-4o-mini (no reranking support) and the evaluation was also scored with gpt-4o-mini. The average answer correctness over 5 corpora is **59.57** — still 2nd on the current leaderboard.

```
python benchmarks/print_scores.py benchmarks/example_results/bench_app_simple_gpt4omini/novel_scores.json
Results:
  Fact Retrieval:  {"rouge_score": 0.3052, "answer_correctness": 0.5595}
  Complex Reasoning:  {"rouge_score": 0.1737, "answer_correctness": 0.5285}
  Contextual Summarize:  {"answer_correctness": 0.6883, "coverage_score": 0.5821}
  Creative Generation:  {"answer_correctness": 0.6065, "coverage_score": 0.4853, "faithfulness": 0.4286}

Average Answer Correctness: 0.5957
```

**Tool-call behavior:** gpt-4o-mini issues only a single `vector_search` call per query, unlike gpt-5.4-mini which may chain multiple calls. A likely explanation is that gpt-4o-mini condenses the question into a tighter search query on the first pass and returns immediately rather than iterating. The example below illustrates this — the full question is distilled down to `"Osiris roles ancient Egypt"` in one shot:

```
2026-05-29 23:05:19,583 [INFO] 54993 MainThread app.py:326 - app.query_stream.start query=Within the narrative's discussion of ancient Egypt, what roles did Osiris fulfill?
2026-05-29 23:05:20,090 [INFO] 54993 MainThread _client.py:1740 - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-05-29 23:05:20,225 [INFO] 54993 MainThread query_runner.py:614 - [runner] tool_calls (skill=none): vector_search
2026-05-29 23:05:20,226 [INFO] 54993 MainThread query_runner.py:627 - [runner] execute_tool vector_search({"collection": "bench_chunks", "query": "Osiris roles ancient Egypt", "top_k": 5})
```

# Future Work

- **Test full corpora** - only 5 corpora are tested currently.
- **Test full corpora in one app** — each corpus is currently tested in its own isolated app. Testing all corpora together in a single application would better reflect cross-document reasoning and reveal how CogBase handles retrieval across a larger, mixed collection.
- **Investigate bench_app_extraction gap** — extraction-based scoring (0.5990) lags bench_app_simple (0.6179); worth understanding whether this is a prompt quality issue, schema design, or a fundamental tradeoff of structured extraction vs. chunk-level retrieval.
- **Memory and Adaptive Engine** — once the memory layer and adaptive evolution engine are implemented, re-run benchmarks to measure the impact on answer correctness, latency, and token usage.
- **Stronger model** — current scores use gpt-5.4-mini; running with a stronger model (e.g., gpt-5.4) would establish an upper bound and is expected to push the leaderboard score higher.

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

**Finding:** Disabling `read_document` has negligible impact on average correctness (0.6246 → 0.6159), suggesting vector search alone are sufficient for this corpus. Contextual summarization and complex reasoning improve slightly without it, while fact retrieval drops.

We can test more corpus to check the impact.
