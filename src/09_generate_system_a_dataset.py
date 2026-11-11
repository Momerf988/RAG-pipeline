import pandas as pd
import time
import os
import openpyxl
from openai import RateLimitError, APIStatusError
from app_init import llm_client, db_index, embedder_model, reranker_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor

# Runs System A (plain RAG) over every "Generated" row in the benchmark and saves the
# question/context/answer/ground-truth quadruple RAGAS needs, plus a per-row response time
# so this run doubles as latency data across the whole benchmark, not just the 20-row
# sample 12_measure_latency.py uses for its stage-by-stage breakdown.
#
# Resumable: writes after every row and skips spec_ids already present in the output file,
# so an interruption (rate limit, closed laptop, crash) just means rerunning this picks up
# exactly where it left off instead of starting over.

BENCHMARK_FILE = "benchmark_datasets/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"
OUTPUT_FILE = "results/System_A_final_eval_results.csv"
SLEEP_BETWEEN_ROWS = 8    # seconds -- llama-3.1-8b-instant is TPM-bottlenecked

user_query_processing_init = UserQueryProcessor(embedder_model)
LLM_response_processing_init = LLMResponseProcessor(llm_client, db_index, reranker_model)

if os.path.exists(OUTPUT_FILE):
    done_df = pd.read_csv(OUTPUT_FILE)
    done_spec_ids = set(done_df['spec_id'].tolist())
    print(f"Resuming — {len(done_spec_ids)} rows already done.")
else:
    done_df = pd.DataFrame(columns=['spec_id', 'topic_id', 'question', 'contexts', 'answer',
                                     'ground_truth', 'response_time_sec'])
    done_spec_ids = set()
    print("Starting fresh.")

workbook = openpyxl.load_workbook(BENCHMARK_FILE)
sheet = workbook[SHEET_NAME]

print("Starting evaluation loop...")

for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, values_only=True):
    spec_id, topic_id = row[0], row[1]
    status = row[9]
    question = row[10]
    ground_truth = row[11]

    if status != "Generated" or not question:
        continue
    if spec_id in done_spec_ids:
        continue

    retries = 0
    truncated = False
    while retries < 3:
        try:
            embedded_query = user_query_processing_init.vectorize_query(question)
            # Covers retrieval + rerank + generation combined (respond_to_user is one
            # bundled call), not generation alone. For a stage-by-stage breakdown, see
            # 12_measure_latency.py, which times each step separately on its own sample.
            t0 = time.perf_counter()
            final_answer, context_list, chunk_id_list = LLM_response_processing_init.respond_to_user(
                embedded_query,
                question,
                'elaborate'
            )
            response_time_sec = time.perf_counter() - t0
            break
        except RateLimitError as e:
            wait_seconds = 30 * (retries + 1)   # 30s, 60s, 90s
            print(f"{spec_id}: rate limit hit, sleeping {wait_seconds}s...")
            time.sleep(wait_seconds)
            retries += 1
        except AssertionError as e:
            # A truncated answer (finish_reason == "length") gets logged and skipped rather
            # than crashing the whole run -- spec_id stays out of done_spec_ids, so it's
            # picked up automatically next time this script is run.
            print(f"{spec_id}: SKIPPED -- {e}")
            truncated = True
            break
        except APIStatusError as e:
            # A 413 "request too large" error -- this account's real ceiling is 6000 TPM,
            # prompt + completion combined. Retrying the same request won't help, it's not a
            # timing issue, so skip and move on.
            print(f"{spec_id}: SKIPPED -- request too large for this account's TPM limit: {e}")
            truncated = True
            break
    else:
        print(f"{spec_id}: giving up after 3 retries. Saved progress so far; rerun script to continue.")
        break

    if truncated:
        continue

    new_row = pd.DataFrame([{
        'spec_id': spec_id,
        'topic_id': topic_id,
        'question': question,
        'contexts': context_list,
        'answer': final_answer,
        'ground_truth': ground_truth,
        'response_time_sec': response_time_sec
    }])
    done_df = pd.concat([done_df, new_row], ignore_index=True)
    done_df.to_csv(OUTPUT_FILE, index=False)

    done_spec_ids.add(spec_id)
    print(f"Processed {spec_id}. Total done: {len(done_spec_ids)}")
    time.sleep(SLEEP_BETWEEN_ROWS)

print(f"\nDone. {len(done_spec_ids)} rows saved to {OUTPUT_FILE}.")
