# LLM Evaluation

The GraphRAG-Benchmark evaluation can produce low scores for answers that are factually correct but phrased differently from the ground truth. This directory explores using an LLM judge as an alternative scoring method. **Note:** this is not a claim that LLM evaluation is strictly better — it is an experiment to understand the gaps between the two approaches.

## Running LLM Evaluation

Score answers using the LLM judge:
```bash
python benchmarks/graphrag/llm_evaluation/llm_answer_accuracy.py scores \
    --input benchmarks/graphrag/results/bench_app_simple_5novels_gpt54mini/novel_all.json \
    --output benchmarks/graphrag/results/bench_app_simple_5novels_gpt54mini/novel_llm_scores.json
```

View aggregate stats by question type:
```bash
python benchmarks/graphrag/llm_evaluation/llm_answer_accuracy.py stats \
    --input benchmarks/graphrag/results/bench_app_simple_5novels_gpt54mini/novel_llm_scores.json
```

Example output — LLM scores are generally higher than benchmark scores:
```
Question type                       avg score      n
-------------------------------------------------------
  Complex Reasoning                    0.6153    137
  Contextual Summarize                 0.7727     86
  Creative Generation                  0.6167     12
  Fact Retrieval                       0.7852    216
-------------------------------------------------------
  Overall                              0.7267    451
  Overtype                             0.6975      4
```

## Comparing Scores

Find answers where the benchmark and LLM scores diverge significantly:
```bash
python benchmarks/graphrag/llm_evaluation/compare_scores.py \
    --llm-scores   benchmarks/graphrag/results/bench_app_simple_5novels_gpt54mini/novel_llm_scores.json \
    --bench-scores benchmarks/graphrag/results/bench_app_simple_5novels_gpt54mini/novel_scores.json \
    --output       benchmarks/graphrag/results/bench_app_simple_5novels_gpt54mini/novel_scores_diff.json \
    [--threshold 0.3]
```

## Example Cases

### Case 1: Benchmark under-scores a correct answer

The benchmark gives 0.48 and 0.22 for the two answers below. The LLM judge scores them 1.0 and 0.7 — a better reflection of actual correctness.

```json
{
  "id": "Novel-219da1ba",
  "question": "How does the text describe the relationship between the human organism and the machine in the chapter where Trost discusses technological advancement?",
  "source": "Novel-10762",
  "question_type": "Complex Reasoning",
  "ground_truth": "The machine is described as an extension of the human organism.",
  "generated_answer": "The text says machines are “simply extensions of the human organism,” like “limbs immeasurably more powerful” and “subtle finger-tips,” and that they depend on human skill and oversight rather than replacing humanity.",
  "llm_score": 1.0,
  "llm_explanation": "The generated answer matches the ground truth by stating that the machine is an extension of the human organism.",
  "answer_correctness": 0.4797062086680114,
  "diff": 0.5203
}
```

```json
{
  "id": "Novel-01abc791",
  "question": "What is the relationship between Asaph Hall and Angeline Stickney?",
  "source": "Novel-58553",
  "question_type": "Complex Reasoning",
  "ground_truth": "Angeline was loved by Asaph Hall (A. HALL).",
  "generated_answer": "They were husband and wife; Angeline Stickney married Asaph Hall in 1856.",
  "llm_score": 0.7,
  "llm_explanation": "The answer correctly identifies their marital relationship, but the ground truth only states that Angeline was loved by Asaph Hall and does not explicitly confirm marriage.",
  "answer_correctness": 0.21666245881188773,
  "diff": 0.4833
}
```

### Case 2: LLM correctly catches a wrong answer

The benchmark gives 0.68 (high similarity to the question text) for a factually incorrect answer. The LLM judge correctly gives 0.0.

**Note:** CogBase runs with gpt-5.4-mini in the test. Running CogBase with a stronger model (gpt-5.4) returns the correct answer: "She visited the Deanery at Westminster Abbey on her way back from an operation call with Mrs. Garrett Anderson."

```json
{
  "id": "Novel-cfc34fd6",
  "question": "During her time in London for medical experience, which notable location did Elsie visit that is mentioned in her correspondence?",
  "source": "Novel-51410",
  "question_type": "Fact Retrieval",
  "ground_truth": "Elsie visited the Deanery.",
  "generated_answer": "Victoria Station",
  "llm_score": 0.0,
  "llm_explanation": "The generated answer names Victoria Station, but the ground truth says Elsie visited the Deanery, so it is incorrect.",
  "answer_correctness": 0.6852822696968655,
  "diff": -0.6853
}
```

### Case 3: LLM over-penalizes a correct answer

The generated answer is detailed and accurate, but it contradicts a poorly written ground truth ("rites unknown to Egyptians" vs. the text actually listing shared rites). The LLM judge gives 0.0. This case suggests the evaluation prompt may need tuning to handle ground truth that is an oversimplification.

```json
{
  "id": "Novel-77457615",
  "question": "What does the narrative state about the religious rites and habits of the Mayas in comparison to those of the Egyptians, highlighting any unique practices?",
  "source": "Novel-30752",
  "question_type": "Fact Retrieval",
  "ground_truth": "The Mayas had religious rites and habits unknown to Egyptians.",
  "generated_answer": "The narrative says the Mayas and Egyptians shared many religious rites and habits, but the Mayas also had some distinctive practices.\n\nShared practices included:\n- a similar sign of reverence in worship, with the arm across the chest and hand on the opposite shoulder\n- use of the sistrum in religious rites\n- belief in the immortality of the soul and a return to earthly life after death\n- funeral customs involving vessels or urns placed with the dead\n\nUnique Maya practices mentioned are:\n- destroying the body by fire and then making statues in the likeness of the deceased so the spirit could return and animate them\n- preparing yearly food offerings for the spirits, the _hanal pixan_, placed in forests or fields in November\n- painting sacrificial victims blue, after which they were venerated as sanctified\n\nIt also says the Mayas worshiped the mastodon head in the scene described, but doubts that this should be called idol worship.",
  "llm_score": 0.0,
  "llm_explanation": "The generated answer directly contradicts the ground truth by claiming the Mayas shared many rites with the Egyptians, whereas the ground truth says their rites and habits were unknown to Egyptians.",
  "answer_correctness": 0.6951286955457796,
  "diff": -0.6951
}
```
