'''
import pandas as pd
import time
from app_init import llm_client, db_index, embedder_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor

user_query_processing_init = UserQueryProcessor(embedder_model)
LLM_response_processing_init = LLMResponseProcessor(llm_client, db_index)
 
df = pd.read_excel("data/benchmark_specification_matrix.xlsx")
df = df.dropna(subset = ['Question'])
eval_data = []

print("Starting evaluation loop...")

for index, row in df.iterrows(): 
    question = row['Question']
    ground_truth = row['Answer']
    
    embedded_query = user_query_processing_init.vectorize_query(question)
    
    final_answer, context_list = LLM_response_processing_init.respond_to_user(
        embedded_query,
        question,
        'elaborate'
    )
    
    eval_data.append({
        'question': question,
        'contexts': context_list,
        'answer': final_answer,
        'ground_truth': ground_truth
    })
    
    print(f"Processed question {index + 1}")
    time.sleep(2)

eval_results_df = pd.DataFrame(eval_data)
eval_results_df.to_csv("System_A_eval_results.csv", index=False)
'''
'''
import pandas as pd
import time
import openpyxl
from app_init import llm_client, db_index, embedder_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor

BENCHMARK_FILE = "data/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"

user_query_processing_init = UserQueryProcessor(embedder_model)
LLM_response_processing_init = LLMResponseProcessor(llm_client, db_index)

workbook = openpyxl.load_workbook(BENCHMARK_FILE)
sheet = workbook[SHEET_NAME]

eval_data = []
print("Starting evaluation loop...")

for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, values_only=True):
    spec_id, topic_id = row[0], row[1]
    status = row[9]
    question = row[10]
    ground_truth = row[11]

    if status != "Generated" or not question:
        continue

    embedded_query = user_query_processing_init.vectorize_query(question)
    final_answer, context_list = LLM_response_processing_init.respond_to_user(
        embedded_query,
        question,
        'elaborate'
    )

    eval_data.append({
        'spec_id': spec_id,
        'topic_id': topic_id,
        'question': question,
        'contexts': context_list,
        'answer': final_answer,
        'ground_truth': ground_truth
    })

    print(f"Processed {spec_id}")
    time.sleep(2)

eval_results_df = pd.DataFrame(eval_data)
eval_results_df.to_csv("System_A_eval_results.csv", index=False)
'''
# V2.6: fixed a crash bug found before the overnight run -- respond_to_user() has returned
# THREE values (final_answer, context_list, chunk_id_list) since the V2 stich_context fix,
# but this script still unpacked only two. Would have raised "ValueError: too many values to
# unpack" on the very first row and crashed the whole run with zero rows saved. Also added
# per-row response_time_sec so the full 150-row run doubles as latency data across the
# whole benchmark, not just the 20-row measure_latency.py sample.
import pandas as pd
import time
import os
import openpyxl
from openai import RateLimitError, APIStatusError
from app_init import llm_client, db_index, embedder_model, reranker_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor

# V2.4: OUTPUT_FILE renamed from "System_A_eval_results.csv" -- that file was generated
# 03/08 under the OLD config (keep_top=7, pre-recalibration). Reusing that filename here
# would trigger this script's own resume-from-existing-file logic, which would see every
# spec_id already present and silently skip the entire run -- producing zero new rows while
# looking like it completed successfully. The old file is left in place, untouched, as the
# keep_top=7 baseline (comparable to your original V7 numbers if you ever want that
# reference point); this run gets its own filename instead of overwriting or aliasing it.
BENCHMARK_FILE = "data/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"
OUTPUT_FILE = "System_A_final_eval_results.csv"
SLEEP_BETWEEN_ROWS = 8    # seconds — llama-3.1-8b-instant is TPM-bottlenecked

user_query_processing_init = UserQueryProcessor(embedder_model)
LLM_response_processing_init = LLMResponseProcessor(llm_client, db_index, reranker_model)

# Resume: load already-processed spec_ids if the file exists
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

    # rate-limit-safe call with one retry
    retries = 0
    truncated = False
    while retries < 3:
        try:
            embedded_query = user_query_processing_init.vectorize_query(question)
            # covers retrieval + rerank + generation combined (respond_to_user is one bundled
            # call) -- NOT generation alone. For a stage-by-stage breakdown, see
            # measure_latency.py, which times each step separately on its own 20-row sample.
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
            # V2.7: this used to crash the whole overnight run on a single truncated answer
            # (finish_reason == "length", from respond_to_user's own hard-abort check). Now
            # logged loudly and SKIPPED (not marked done) instead of halting everything --
            # spec_id stays out of done_spec_ids, so it's automatically picked up the next
            # time this script is run, same as any other unfinished row. Nothing is silently
            # lost: this line is your visible flag that TUTOR_MAX_TOKENS may need raising
            # further if this keeps happening.
            print(f"{spec_id}: SKIPPED -- {e}")
            truncated = True
            break
        except APIStatusError as e:
            # V2.10: this is what actually crashed the overnight run -- a 413 "request too
            # large" error (this account's real ceiling: 6000 TPM, prompt + completion
            # combined). Unlike RateLimitError, waiting and retrying the SAME request would
            # fail again identically every time -- it's not a "too fast" problem, it's a
            # "this exact request is structurally too big" problem. So: log it, skip the row
            # (not marked done, picked up automatically next run), and move on immediately
            # rather than wasting retries on something retrying can't fix.
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
    done_df.to_csv(OUTPUT_FILE, index=False)      # save after every row

    done_spec_ids.add(spec_id)
    print(f"Processed {spec_id}. Total done: {len(done_spec_ids)}")
    time.sleep(SLEEP_BETWEEN_ROWS)

print(f"\nDone. {len(done_spec_ids)} rows saved to {OUTPUT_FILE}.")