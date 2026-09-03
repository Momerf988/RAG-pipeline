# Corrective RAG Tutor -- System A vs System B

Two Python tutoring RAG pipelines built and evaluated head-to-head over the same corpus,
retriever, generator, and benchmark:

- **System A** -- plain retrieve-rerank-generate RAG. No quality gate on retrieval.
- **System B** -- corrective RAG. A separate evaluator model judges retrieved context
  before generation (CORRECT / AMBIGUOUS / INCORRECT), rewrites and re-retrieves once on a
  weak result, and abstains if the second attempt is still insufficient.

Both are grounded in a syllabus-bounded corpus (the official Python tutorial, split into an
eight-week syllabus), evaluated on a 150-item benchmark stratified by Bloom's taxonomy
level, difficulty, retrieval depth, and question type, and scored with RAGAS (faithfulness,
answer relevancy, context precision, context recall).

## Layout

```
src/
  config.py                             shared settings, loaded from .env
  app_init.py                           client and model initialisation
  document_processor.py                 PDF -> chunks -> Pinecone
  user_query_processor.py               query embedding, interactive input
  RAG_response_processor.py             retrieval, reranking, prompts, generation
  crag_evaluator.py                     System B's retrieval evaluator + query rewriter

  01_system_a_main.py                   interactive CLI, System A, one question at a time
  02_system_b_main.py                   interactive CLI, System B, one question at a time
  03_generate_benchmark_questions.py    writes benchmark questions into the spec matrix
  04_validate_benchmark.py              sanity-checks the spec matrix, executes embedded code
  05_spanify_gold_answers.py            converts answer keys to chunking-independent spans
  06_tune_embedding_model.py            embedder ablation (MiniLM vs bge-base)
  07_tune_retrieval_depth.py            keep_top recall sweep, no LLM calls
  08_tune_keep_top_quality.py           keep_top quality ablation, real generation + RAGAS
  09_generate_system_a_dataset.py       full 150-row batch run, System A
  10_generate_system_b_dataset.py       full 150-row batch run, System B
  11_run_ragas_evaluation.py            RAGAS-scores both systems' output
  12_measure_latency.py                 stage-by-stage timing, both systems, same sample
  13_analyse_results.py                 exploratory breakdown workbook
  14_verify_statistics.py               reproduces every statistic with significance tests

benchmark_datasets/   benchmark spec matrix
pdf_files/             the corpus (7 PDFs, one per syllabus topic)
results/               final generation output, scorecards, ablation results, latency runs
```

The six files at the top of `src/` are library code -- other scripts `import` them, so
their filenames can't start with a digit (Python doesn't allow that). Everything numbered
`01` to `14` is a script you run directly, in that order, top to bottom: each step's output
feeds the next. `01`/`02` are standalone interactive demos and nothing downstream depends
on them. Steps `06` and `07` are pure local computation (no LLM calls, no cost); everything
from `08` onward calls a paid or rate-limited API.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your own keys
```

Needs a Pinecone index, an OpenAI-compatible chat endpoint for the tutor and evaluator
(this project used Groq and OpenRouter), and an OpenAI key for RAGAS judging.

## Running

Every numbered script is written to be run from the repo root, e.g.:

```
python src/09_generate_system_a_dataset.py
```

not from inside `src/` itself -- the data paths (`benchmark_datasets/...`, `pdf_files/...`,
`results/...`) are relative to the repo root.

## Reproducing the results

`results/` already holds the final output this project reports -- the CSVs and scorecards
are the actual data behind every number in the write-up. Running `14_verify_statistics.py`
regenerates every reported statistic from those files with no API calls and no cost.

To regenerate from scratch: build the corpus with `document_processor.py`'s
`upsert_to_db()` (called from `01_system_a_main.py`, commented out by default so it isn't
run by accident), then work through `03` to `14` in order. `09` and `10` are resumable --
safe to interrupt and rerun, they pick up from whichever spec_ids are already in their
output file.

## What each script does, and what it was called before this cleanup

| File now | What it does | Original filename |
|---|---|---|
| `config.py` | Loads `.env`, holds shared constants (temperature, chunk count, model names) | `config.py` |
| `app_init.py` | Creates the LLM client, Pinecone connection, embedder, and reranker once | `app_init.py` |
| `document_processor.py` | Reads PDFs, chunks them, embeds and upserts into Pinecone | `document_processor.py` |
| `user_query_processor.py` | Takes user input, embeds the query | `user_query_processor.py` |
| `RAG_response_processor.py` | Retrieval, reranking, both generation prompts, the actual LLM call | `RAG_response_processor.py` |
| `crag_evaluator.py` | System B's three-way retrieval judge and the query rewriter | `crag_evaluator.py` |
| `01_system_a_main.py` | Interactive terminal chat with System A | `main.py` |
| `02_system_b_main.py` | Interactive terminal chat with System B | `system_b_main.py` |
| `03_generate_benchmark_questions.py` | Writes exam-style questions into the spec matrix, topic by topic | `benchmark_question_generator.py` |
| `04_validate_benchmark.py` | Checks the generated questions for length, leakage, duplicates, and actually runs the embedded code | `validate_benchmark.py` |
| `05_spanify_gold_answers.py` | Converts each question's answer key from chunk IDs to raw character positions | `spanify_gold.py` |
| `06_tune_embedding_model.py` | Compares MiniLM against a stronger embedder on retrieval recall | `02_diag_embeddings.py` |
| `07_tune_retrieval_depth.py` | Sweeps keep_top from 3 to 60 and measures recall, no generation involved | `09_tune_retrieval.py` |
| `08_tune_keep_top_quality.py` | Takes the keep_top candidates from step 07 and checks which one actually produces the best generated answers (real generation + RAGAS) | `10_tune_keep_top_quality.py` |
| `09_generate_system_a_dataset.py` | Runs System A over all 150 benchmark questions | `generate_evaluation_dataset.py` |
| `10_generate_system_b_dataset.py` | Runs System B over all 150 benchmark questions | `generate_evaluation_dataset_B.py` |
| `11_run_ragas_evaluation.py` | Scores both systems' output with RAGAS | `run_ragas_eval.py` |
| `12_measure_latency.py` | Times both systems stage-by-stage on the same 20-row sample | `measure_latency.py` |
| `13_analyse_results.py` | Builds a browsable Excel breakdown of the results | `analyse_results.py` |
| `14_verify_statistics.py` | Reruns every statistic reported in the write-up, with significance tests | `verify_statistics.py` |

