# V2 work -- bounded three-way CRAG batch evaluation. Updated to match the V2-fixed
# RAG_response_processor.py signatures (stich_context now returns chunk_id_list too;
# generate_response now returns finish_reason too) and to route on three evaluator verdicts
# (CORRECT / AMBIGUOUS / INCORRECT) instead of the old binary CORRECT / else.
#
# IMPORTANT: OUTPUT_FILE below is deliberately a NEW filename, not the original
# System_B_eval_results.csv. That file holds the binary-evaluator results already reported
# in the write-up (V7) -- overwriting it would destroy that baseline and this script's own
# resume-from-existing-file logic would silently skip every row if pointed at it. Keep both
# CSVs; System_B_eval_results.csv = binary baseline, System_B_3way_eval_results.csv = this run.
import pandas as pd
import time
import os
import openpyxl
from openai import RateLimitError
from app_init import llm_client, db_index, embedder_model, reranker_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor
from crag_evaluator import CRAGEvaluator
import config

BENCHMARK_FILE = "data/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"
OUTPUT_FILE = "System_B_3way_eval_results.csv"
SLEEP_BETWEEN_ROWS = 12    # System B makes 2-4 LLM calls per row; give TPM more room

query_processor = UserQueryProcessor(embedder_model)
response_processor = LLMResponseProcessor(llm_client, db_index, reranker_model)
crag_evaluator = CRAGEvaluator(llm_client)

FALLBACK_MSG = "I'm sorry, that concept is outside the scope of our current syllabus or I couldn't find reliable information in the database to answer you accurately."

# Resume: load already-processed spec_ids if the file exists
if os.path.exists(OUTPUT_FILE):
    done_df = pd.read_csv(OUTPUT_FILE)
    done_spec_ids = set(done_df['spec_id'].tolist())
    print(f"Resuming — {len(done_spec_ids)} rows already done.")
else:
    done_df = pd.DataFrame(columns=[
        'spec_id', 'topic_id', 'question', 'contexts', 'answer', 'ground_truth',
        'crag_decision_1', 'rewritten_query', 'crag_decision_2', 'path_taken'
    ])
    done_spec_ids = set()
    print("Starting fresh.")

workbook = openpyxl.load_workbook(BENCHMARK_FILE)
sheet = workbook[SHEET_NAME]

def process_one_row(question):
    """Run System B's bounded three-way routing for one question.
    Returns (answer, context_list, crag_decision_1, rewritten_query, crag_decision_2, path_taken).

    Routing (v2, three-way):
      - decision_1 == CORRECT             -> answer directly. path = "direct"
      - decision_1 in (AMBIGUOUS,INCORRECT) -> rewrite + retry ONCE (bounded, no further loops)
          - decision_2 in (CORRECT, AMBIGUOUS) -> answer from the second context. path = "rewritten"
          - decision_2 == INCORRECT             -> abstain. path = "fallback"
    """
    # 1. Standard retrieval
    embedded_query = query_processor.vectorize_query(question)
    matches = response_processor.search_db(embedded_query)
    reranked_matches = response_processor.rerank(question, matches)
    context_string, context_list, chunk_ids = response_processor.stich_context(reranked_matches)

    # 2. First CRAG eval
    decision_1 = crag_evaluator.evaluate_context(question, context_string)

    if decision_1 == "CORRECT":
        # Direct answer path
        # v2 work: LLM_prompt_crag, not LLM_prompt -- CRAG already judged sufficiency, the
        # tutor should not independently re-judge and refuse (see RAG_response_processor.py).
        system_persona, user_instruction = response_processor.LLM_prompt_crag(question, 'elaborate', context_string)
        answer, finish_reason = response_processor.generate_response(system_persona, user_instruction)
        assert finish_reason == "stop", f"HARD ABORT: finish_reason={finish_reason}, answer may be truncated"
        # v2 work -- disguised-refusal detection. CRAG judged this CORRECT and routed to
        # generation, but the tutor's own separate critical_rule check (in LLM_prompt) can
        # still independently refuse. If that happened, relabel the path so it isn't counted
        # as a real answer downstream -- found live during manual testing (list/tuple queries).
        path = "direct_but_refused" if answer.strip() == config.REFUSAL_STRING else "direct"
        return answer, context_list, decision_1, "", "", path

    # 3. Corrective path: rewrite + second retrieval (AMBIGUOUS or INCORRECT both land here)
    rewritten = crag_evaluator.rewrite_query(question)
    embedded_rewritten = query_processor.vectorize_query(rewritten)
    matches_2 = response_processor.search_db(embedded_rewritten)
    reranked_matches_2 = response_processor.rerank(rewritten, matches_2)
    context_string_2, context_list_2, chunk_ids_2 = response_processor.stich_context(reranked_matches_2)

    # 4. Second CRAG eval
    decision_2 = crag_evaluator.evaluate_context(rewritten, context_string_2)

    # v2 change: CORRECT and AMBIGUOUS both generate here -- only a second, independent
    # INCORRECT verdict triggers the fallback. See module docstring for rationale.
    if decision_2 in ("CORRECT", "AMBIGUOUS"):
        # v2 work: LLM_prompt_crag, not LLM_prompt -- same reasoning as the direct path above.
        system_persona, user_instruction = response_processor.LLM_prompt_crag(question, 'elaborate', context_string_2)
        answer, finish_reason = response_processor.generate_response(system_persona, user_instruction)
        assert finish_reason == "stop", f"HARD ABORT: finish_reason={finish_reason}, answer may be truncated"
        # v2 work -- same disguised-refusal check as the direct path above.
        path = "rewritten_but_refused" if answer.strip() == config.REFUSAL_STRING else "rewritten"
        return answer, context_list_2, decision_1, rewritten, decision_2, path

    # 5. Graceful fallback
    return FALLBACK_MSG, context_list_2, decision_1, rewritten, decision_2, "fallback"


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

    # rate-limit-safe call with retry
    retries = 0
    result = None
    while retries < 3:
        try:
            result = process_one_row(question)
            break
        except RateLimitError as e:
            wait_seconds = 30 * (retries + 1)
            print(f"{spec_id}: rate limit hit, sleeping {wait_seconds}s...")
            time.sleep(wait_seconds)
            retries += 1
    if result is None:
        print(f"{spec_id}: giving up after 3 retries. Progress saved; rerun to continue.")
        break

    answer, context_list, decision_1, rewritten, decision_2, path = result

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
        'path_taken': path
    }])
    done_df = pd.concat([done_df, new_row], ignore_index=True)
    done_df.to_csv(OUTPUT_FILE, index=False)

    done_spec_ids.add(spec_id)
    print(f"Processed {spec_id} — path: {path}, decision_1: {decision_1}. Total done: {len(done_spec_ids)}")
    time.sleep(SLEEP_BETWEEN_ROWS)

print(f"\nDone. {len(done_spec_ids)} rows saved to {OUTPUT_FILE}.")

# Path breakdown for quick sanity
if len(done_df) > 0:
    print("\nPath breakdown:")
    print(done_df['path_taken'].value_counts())
    print("\ncrag_decision_1 breakdown (first-pass evaluator verdicts):")
    print(done_df['crag_decision_1'].value_counts())
