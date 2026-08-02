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
import pandas as pd
import time
import os
import openpyxl
from openai import RateLimitError
from app_init import llm_client, db_index, embedder_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor

BENCHMARK_FILE = "data/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"
OUTPUT_FILE = "System_A_eval_results.csv"
SLEEP_BETWEEN_ROWS = 8    # seconds — llama-3.1-8b-instant is TPM-bottlenecked

user_query_processing_init = UserQueryProcessor(embedder_model)
LLM_response_processing_init = LLMResponseProcessor(llm_client, db_index)

# Resume: load already-processed spec_ids if the file exists
if os.path.exists(OUTPUT_FILE):
    done_df = pd.read_csv(OUTPUT_FILE)
    done_spec_ids = set(done_df['spec_id'].tolist())
    print(f"Resuming — {len(done_spec_ids)} rows already done.")
else:
    done_df = pd.DataFrame(columns=['spec_id', 'topic_id', 'question', 'contexts', 'answer', 'ground_truth'])
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
    while retries < 3:
        try:
            embedded_query = user_query_processing_init.vectorize_query(question)
            final_answer, context_list = LLM_response_processing_init.respond_to_user(
                embedded_query,
                question,
                'elaborate'
            )
            break
        except RateLimitError as e:
            wait_seconds = 30 * (retries + 1)   # 30s, 60s, 90s
            print(f"{spec_id}: rate limit hit, sleeping {wait_seconds}s...")
            time.sleep(wait_seconds)
            retries += 1
    else:
        print(f"{spec_id}: giving up after 3 retries. Saved progress so far; rerun script to continue.")
        break

    new_row = pd.DataFrame([{
        'spec_id': spec_id,
        'topic_id': topic_id,
        'question': question,
        'contexts': context_list,
        'answer': final_answer,
        'ground_truth': ground_truth
    }])
    done_df = pd.concat([done_df, new_row], ignore_index=True)
    done_df.to_csv(OUTPUT_FILE, index=False)      # save after every row

    done_spec_ids.add(spec_id)
    print(f"Processed {spec_id}. Total done: {len(done_spec_ids)}")
    time.sleep(SLEEP_BETWEEN_ROWS)

print(f"\nDone. {len(done_spec_ids)} rows saved to {OUTPUT_FILE}.")