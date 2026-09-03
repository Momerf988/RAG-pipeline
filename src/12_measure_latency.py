import time
import random
import os
import openpyxl
import pandas as pd
from openai import RateLimitError, APIStatusError
from app_init import llm_client, db_index, embedder_model, reranker_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor
from crag_evaluator import CRAGEvaluator

# Times System A and System B stage-by-stage (retrieval, evaluator, rewrite, generation) on
# the same 20-row stratified sample, both systems timed on the same items in the same
# session. Every pacing sleep runs after every timestamp for a row has already been
# captured, right before the row returns, so it can never leak into a measured duration --
# total_ms is the sum of the already-clean component timings, not a fresh wall-clock read,
# which is what actually guarantees that.
#
# T (inter-row pacing delay) is a request-spacing parameter against the provider, not an
# architecture parameter. This project ran it at T=1, 30 and 70s across three separate
# executions, all targeting the identical 20-row sample (random.seed(42) below). Heavier
# pacing reduces contention but doesn't eliminate it, so report median alongside mean when
# writing this up, and don't describe any T value as "guaranteeing" a clean run.
T = 70
BENCHMARK_FILE = "benchmark_datasets/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"
OUTPUT_FILE = f"results/Latency_Results_t{T}.csv"
SAMPLE_SIZE_PER_RETRIEVAL_LEVEL = 5   # 5 rows x 4 levels (R1-R4) = 20 rows total

random.seed(42)

query_processor = UserQueryProcessor(embedder_model)
response_processor = LLMResponseProcessor(llm_client, db_index, reranker_model)
crag_evaluator = CRAGEvaluator(llm_client)


def select_latency_sample():
    workbook = openpyxl.load_workbook(BENCHMARK_FILE)
    sheet = workbook[SHEET_NAME]

    rows_by_retrieval = {}
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, values_only=True):
        spec_id, topic_id, status, question, retrieval_level = row[0], row[1], row[9], row[10], row[6]
        if status != "Generated" or not question:
            continue
        rows_by_retrieval.setdefault(retrieval_level, []).append((spec_id, topic_id, question))

    sample = []
    for level, rows in rows_by_retrieval.items():
        n = min(SAMPLE_SIZE_PER_RETRIEVAL_LEVEL, len(rows))
        sample.extend([(*r, level) for r in random.sample(rows, n)])
    return sample


def time_system_a(question):
    t0 = time.perf_counter()
    embedded_query = query_processor.vectorize_query(question)
    matches = response_processor.search_db(embedded_query)
    reranked = response_processor.rerank(question, matches)
    context_string, context_list, chunk_ids = response_processor.stich_context(reranked)
    t1 = time.perf_counter()

    if not context_string:
        return {"retrieval_ms": (t1 - t0) * 1000, "evaluator_ms": 0, "generation_ms": 0, "total_ms": (t1 - t0) * 1000}

    system_persona, user_instruction = response_processor.LLM_prompt(question, 'elaborate', context_string)
    final_answer, finish_reason = response_processor.generate_response(system_persona, user_instruction)
    t2 = time.perf_counter()
    time.sleep(5)  # rate-limit pacing, runs after the last timestamp is already captured

    return {
        "retrieval_ms": (t1 - t0) * 1000,
        "evaluator_ms": 0,
        "generation_ms": (t2 - t1) * 1000,
        "total_ms": (t2 - t0) * 1000,
    }


def time_system_b(question):
    t0 = time.perf_counter()
    embedded_query = query_processor.vectorize_query(question)
    matches = response_processor.search_db(embedded_query)
    reranked = response_processor.rerank(question, matches)
    context_string, context_list, chunk_ids = response_processor.stich_context(reranked)
    t1 = time.perf_counter()

    decision_1 = crag_evaluator.evaluate_context(question, context_string)
    t2 = time.perf_counter()

    retrieval_ms = (t1 - t0) * 1000
    evaluator_ms = (t2 - t1) * 1000
    generation_ms = 0
    rewrite_ms = 0
    path = "direct"

    if decision_1 == "CORRECT":
        # matches production: System B generates with LLM_prompt_crag, not LLM_prompt
        system_persona, user_instruction = response_processor.LLM_prompt_crag(question, 'elaborate', context_string)
        response_processor.generate_response(system_persona, user_instruction)
        t3 = time.perf_counter()
        generation_ms = (t3 - t2) * 1000
    else:
        path = "rewritten_or_fallback"
        rewritten = crag_evaluator.rewrite_query(question)
        embedded_rewritten = query_processor.vectorize_query(rewritten)
        matches_2 = response_processor.search_db(embedded_rewritten)
        reranked_2 = response_processor.rerank(rewritten, matches_2)
        context_string_2, context_list_2, chunk_ids_2 = response_processor.stich_context(reranked_2)
        decision_2 = crag_evaluator.evaluate_context(rewritten, context_string_2)
        t3 = time.perf_counter()
        # rewrite_ms spans the rewrite call, second retrieval, and second evaluator call --
        # no sleeps anywhere inside this span.
        rewrite_ms = (t3 - t2) * 1000

        if decision_2 in ("CORRECT", "AMBIGUOUS"):
            # matches production routing: CORRECT and AMBIGUOUS both generate on the second
            # pass, only a second INCORRECT falls back. Also matches production: uses
            # LLM_prompt_crag, not LLM_prompt.
            system_persona, user_instruction = response_processor.LLM_prompt_crag(question, 'elaborate', context_string_2)
            response_processor.generate_response(system_persona, user_instruction)
            t4 = time.perf_counter()
            generation_ms = (t4 - t3) * 1000
        else:
            t4 = t3

    total_ms = retrieval_ms + evaluator_ms + rewrite_ms + generation_ms
    result = {
        "retrieval_ms": retrieval_ms,
        "evaluator_ms": evaluator_ms,
        "rewrite_cycle_ms": rewrite_ms,
        "generation_ms": generation_ms,
        "total_ms": total_ms,
        "path": path,
    }
    time.sleep(5)  # rate-limit pacing before the next row -- every timestamp for this row
                   # has already been captured, so this can't contaminate anything above
    return result


def main():
    sample = select_latency_sample()

    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        done_spec_ids = set(existing_df['spec_id'].tolist())
        results = existing_df.to_dict('records')
        print(f"Resuming -- {len(done_spec_ids)} rows already timed.")
    else:
        done_spec_ids = set()
        results = []
        print("Starting fresh.")

    remaining = [row for row in sample if row[0] not in done_spec_ids]
    print(f"Timing {len(sample)} rows total across both systems ({len(remaining)} remaining)...\n")

    for spec_id, topic_id, question, retrieval_level in remaining:
        print(f"Timing {spec_id} ({retrieval_level})...")

        retries = 0
        row_result = None
        skipped = False
        while retries < 3:
            try:
                a_timing = time_system_a(question)
                b_timing = time_system_b(question)
                row_result = {
                    "spec_id": spec_id,
                    "topic_id": topic_id,
                    "retrieval_level": retrieval_level,
                    "A_retrieval_ms": a_timing["retrieval_ms"],
                    "A_generation_ms": a_timing["generation_ms"],
                    "A_total_ms": a_timing["total_ms"],
                    "B_retrieval_ms": b_timing["retrieval_ms"],
                    "B_evaluator_ms": b_timing["evaluator_ms"],
                    "B_rewrite_cycle_ms": b_timing.get("rewrite_cycle_ms", 0),
                    "B_generation_ms": b_timing["generation_ms"],
                    "B_total_ms": b_timing["total_ms"],
                    "B_path": b_timing["path"],
                }
                break
            except RateLimitError as e:
                wait_seconds = 30 * (retries + 1)
                print(f"{spec_id}: rate limit hit, sleeping {wait_seconds}s...")
                time.sleep(wait_seconds)
                retries += 1
            except AssertionError as e:
                print(f"{spec_id}: SKIPPED -- {e}")
                skipped = True
                break
            except APIStatusError as e:
                print(f"{spec_id}: SKIPPED -- request too large for this account's TPM limit: {e}")
                skipped = True
                break

        if skipped:
            continue
        if row_result is None:
            print(f"{spec_id}: giving up after 3 retries (likely daily quota). "
                  f"Progress saved -- rerun this script to continue.")
            break

        results.append(row_result)
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        done_spec_ids.add(spec_id)
        time.sleep(T)  # long pause -- must fully clear the per-minute token budget between
                        # rows, since each row can issue up to 5 LLM calls with large context

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("\nNo rows timed yet -- nothing to summarize.")
        return
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n=== Latency summary (n={len(df)} of {len(sample)} planned) ===\n")
    print("System A:")
    print(f"  retrieval:  mean={df['A_retrieval_ms'].mean():.0f}ms")
    print(f"  generation: mean={df['A_generation_ms'].mean():.0f}ms")
    print(f"  total:      mean={df['A_total_ms'].mean():.0f}ms")

    print("\nSystem B:")
    print(f"  retrieval:      mean={df['B_retrieval_ms'].mean():.0f}ms")
    print(f"  evaluator:      mean={df['B_evaluator_ms'].mean():.0f}ms")
    print(f"  rewrite cycle:  mean={df['B_rewrite_cycle_ms'].mean():.0f}ms (0 for direct-path rows)")
    print(f"  generation:     mean={df['B_generation_ms'].mean():.0f}ms")
    print(f"  total:          mean={df['B_total_ms'].mean():.0f}ms")

    print(f"\nOverhead: System B is {df['B_total_ms'].mean() - df['A_total_ms'].mean():.0f}ms slower than System A on average (T = {T})")
    print(f"\nPath breakdown:")
    print(df['B_path'].value_counts())
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
