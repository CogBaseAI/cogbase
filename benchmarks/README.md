# GraphRAG-Benchmark Runner for CogBase

Ingests each corpus into a dedicated CogBase application, runs all QA questions through the query endpoint, and writes results in the format expected by the benchmark's `generation_eval` and `retrieval_eval` scripts.

These results are promising and align with the discussion in [docs/knowledge-graph-decision.md](docs/knowledge-graph-decision.md).
Using only simple chunking (bench_app_simple) and LLM-driven inference, CogBase achieves an average answer correctness of 61.79, placing 2nd on the GraphRAG-Bench (Novel) Leaderboard (https://graphrag-bench.github.io/), just below the current leader at 63.72.

> Note: this score used gpt-5.4-mini across 5 corpora only. A stronger model (e.g. gpt-5.4) would push the score higher. Full corpora testing is planned.

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

Note: bench_app_extraction scores slightly lower on average than bench_app_simple (0.5990 vs 0.6179), which is worth investigating further.
