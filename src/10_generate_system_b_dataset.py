import pandas as pd
import time
import os
import openpyxl
from openai import RateLimitError, APIStatusError
from app_init import llm_client, db_index, embedder_model, reranker_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor
from crag_evaluator import CRAGEvaluator
import config

# Runs System B (corrective RAG, three-way verdict) over every "Generated" row in the
# benchmark.
#
# Routing:
#   decision_1 == CORRECT              -> answer directly. path = "direct"
#   decision_1 in (AMBIGUOUS,INCORRECT) -> rewrite + retry once (bounded, no further loops)
#       decision_2 in (CORRECT, AMBIGUOUS) -> answer from the second context. path = "rewritten"
#       decision_2 == INCORRECT             -> abstain. path = "fallback"
#
# response_time_sec covers the entire call (all retrieval + evaluator + rewrite +
# generation steps combined, whatever the path taken), not directly comparable row-by-row
# to System A's timing since B can take 1-4 LLM calls depending on path while A always
# takes exactly one. That asymmetry is the whole point of 12_measure_latency.py's
# stage-by-stage breakdown.
#
# Resumable, same pattern as 09_generate_system_a_dataset.py: writes after every row and
# skips spec_ids already present, so an interruption just means rerunning to continue.

BENCHMARK_FILE = "benchmark_datasets/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"
OUTPUT_FILE = "results/System_B_3way_eval_results.csv"
SLEEP_BETWEEN_ROWS = 12    # System B makes 2-4 LLM calls per row, give TPM more room

query_processor = UserQueryProcessor(embedder_model)
response_processor = LLMResponseProcessor(llm_client, db_index, reranker_model)
crag_evaluator = CRAGEvaluator(llm_client)

FALLBACK_MSG = "I'm sorry, that concept is outside the scope of our current syllabus or I couldn't find reliable information in the database to answer you accurately."

if os.path.exists(OUTPUT_FILE):
    done_df = pd.read_csv(OUTPUT_FILE)
    done_spec_ids = set(done_df['spec_id'].tolist())
    print(f"Resuming — {len(done_spec_ids)} rows already done.")
else:
    done_df = pd.DataFrame(columns=[
        'spec_id', 'topic_id', 'question', 'contexts', 'answer', 'ground_truth',
        'crag_decision_1', 'rewritten_query', 'crag_decision_2', 'path_taken',
        'response_time_sec'
    ])
    done_spec_ids = set()
    print("Starting fresh.")

workbook = openpyxl.load_workbook(BENCHMARK_FILE)
sheet = workbook[SHEET_NAME]


def process_one_row(question):
    """Runs System B's bounded three-way routing for one question.
    Returns (answer, context_list, crag_decision_1, rewritten_query, crag_decision_2, path_taken, response_time_sec)."""
    t0 = time.perf_counter()
    embedded_query = query_processor.vectorize_query(question)
    matches = response_processor.search_db(embedded_query)
    reranked_matches = response_processor.rerank(question, matches)
    context_string, context_list, chunk_ids = response_processor.stich_context(reranked_matches)

    decision_1 = crag_evaluator.evaluate_context(question, context_string)

    if decision_1 == "CORRECT":
        system_persona, user_instruction = response_processor.LLM_prompt_crag(question, 'elaborate', context_string)
        answer, finish_reason = response_processor.generate_response(system_persona, user_instruction)
        assert finish_reason == "stop", f"HARD ABORT: finish_reason={finish_reason}, answer may be truncated"
        # CRAG judged this CORRECT and routed to generation, but the tutor's own separate
        # grounding check can still independently refuse -- relabel the path if that
        # happened so it isn't counted as a real answer downstream.
        path = "direct_but_refused" if answer.strip() == config.REFUSAL_STRING else "direct"
        return answer, context_list, decision_1, "", "", path, time.perf_counter() - t0

    # Corrective path: rewrite + second retrieval, for either AMBIGUOUS or INCORRECT.
    rewritten = crag_evaluator.rewrite_query(question)
    embedded_rewritten = query_processor.vectorize_query(rewritten)
    matches_2 = response_processor.search_db(embedded_rewritten)
    reranked_matches_2 = response_processor.rerank(rewritten, matches_2)
    context_string_2, context_list_2, chunk_ids_2 = response_processor.stich_context(reranked_matches_2)

    decision_2 = crag_evaluator.evaluate_context(rewritten, context_string_2)

    if decision_2 in ("CORRECT", "AMBIGUOUS"):
        system_persona, user_instruction = response_processor.LLM_prompt_crag(question, 'elaborate', context_string_2)
        answer, finish_reason = response_processor.generate_response(system_persona, user_instruction)
        assert finish_reason == "stop", f"HARD ABORT: finish_reason={finish_reason}, answer may be truncated"
        path = "rewritten_but_refused" if answer.strip() == config.REFUSAL_STRING else "rewritten"
        return answer, context_list_2, decision_1, rewritten, decision_2, path, time.perf_counter() - t0

    return FALLBACK_MSG, context_list_2, decision_1, rewritten, decision_2, "fallback", time.perf_counter() - t0


print("Starting System B (three-way) evaluation loop...")

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
    result = None
    truncated = False
    while retries < 3:
        try:
            result = process_one_row(question)
            break
        except RateLimitError as e:
            wait_seconds = 30 * (retries + 1)
            print(f"{spec_id}: rate limit hit, sleeping {wait_seconds}s...")
            time.sleep(wait_seconds)
            retries += 1
        except AssertionError as e:
            print(f"{spec_id}: SKIPPED -- {e}")
            truncated = True
            break
        except APIStatusError as e:
            # This account's real ceiling is 6000 TPM (prompt + completion combined).
            # System B stacks retrieval + evaluator + rewrite + a second retrieval on top of
            # generation, so it's more exposed to this than System A. Retrying won't help,
            # so skip and move on.
            print(f"{spec_id}: SKIPPED -- request too large for this account's TPM limit: {e}")
            truncated = True
            break
    if truncated:
        continue
    if result is None:
        print(f"{spec_id}: giving up after 3 retries. Progress saved; rerun to continue.")
        break

    answer, context_list, decision_1, rewritten, decision_2, path, response_time_sec = result

    new_row = pd.DataFrame([{
        'spec_id': spec_id,
        'topic_id': topic_id,
        'question': question,
        'contexts': context_list,
        'answer': answer,
        'ground_truth': ground_truth,
        'crag_decision_1': decision_1,
        'rewritten_query': rewritten,
        'crag_decision_2': decision_2,
        'path_taken': path,
        'response_time_sec': response_time_sec
    }])
    done_df = pd.concat([done_df, new_row], ignore_index=True)
    done_df.to_csv(OUTPUT_FILE, index=False)

    done_spec_ids.add(spec_id)
    print(f"Processed {spec_id} — path: {path}, decision_1: {decision_1}. Total done: {len(done_spec_ids)}")
    time.sleep(SLEEP_BETWEEN_ROWS)

print(f"\nDone. {len(done_spec_ids)} rows saved to {OUTPUT_FILE}.")

if len(done_df) > 0:
    print("\nPath breakdown:")
    print(done_df['path_taken'].value_counts())
    print("\ncrag_decision_1 breakdown (first-pass evaluator verdicts):")
    print(done_df['crag_decision_1'].value_counts())
