import time
import random
import openpyxl
import pandas as pd
from app_init import llm_client, db_index, embedder_model, reranker_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor
from crag_evaluator import CRAGEvaluator
T = 1
BENCHMARK_FILE = "data/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"
OUTPUT_FILE = f"Latency_Results_t{{T}}.csv"
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
    context_string, context_list = response_processor.stich_context(reranked)
    t1 = time.perf_counter()

    if not context_string:
        return {"retrieval_ms": (t1 - t0) * 1000, "evaluator_ms": 0, "generation_ms": 0, "total_ms": (t1 - t0) * 1000}

    system_persona, user_instruction = response_processor.LLM_prompt(question, 'elaborate', context_string)
    final_answer = response_processor.generate_response(system_persona, user_instruction)
    t2 = time.perf_counter()
    time.sleep(5)  # clear rate-limit headroom before the next LLM call anywhere in the script

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
    context_string, context_list = response_processor.stich_context(reranked)
    t1 = time.perf_counter()

    decision_1 = crag_evaluator.evaluate_context(question, context_string)
    t2 = time.perf_counter()
    time.sleep(5)

    retrieval_ms = (t1 - t0) * 1000
    evaluator_ms = (t2 - t1) * 1000
    generation_ms = 0
    rewrite_ms = 0
    path = "direct"

    if decision_1 == "CORRECT":
        system_persona, user_instruction = response_processor.LLM_prompt(question, 'elaborate', context_string)
        response_processor.generate_response(system_persona, user_instruction)
        t3 = time.perf_counter()
        time.sleep(5)
        generation_ms = (t3 - t2) * 1000
    else:
        path = "rewritten_or_fallback"
        t2b = time.perf_counter()
        rewritten = crag_evaluator.rewrite_query(question)
        time.sleep(5)
        embedded_rewritten = query_processor.vectorize_query(rewritten)
        matches_2 = response_processor.search_db(embedded_rewritten)
        reranked_2 = response_processor.rerank(rewritten, matches_2)
        context_string_2, context_list_2 = response_processor.stich_context(reranked_2)
        decision_2 = crag_evaluator.evaluate_context(rewritten, context_string_2)
        time.sleep(5)
        t3 = time.perf_counter()
        rewrite_ms = (t3 - t2b) * 1000

        if decision_2 == "CORRECT":
            system_persona, user_instruction = response_processor.LLM_prompt(question, 'elaborate', context_string_2)
            response_processor.generate_response(system_persona, user_instruction)
            t4 = time.perf_counter()
            time.sleep(5)
            generation_ms = (t4 - t3) * 1000
        else:
            t4 = t3

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "retrieval_ms": retrieval_ms,
        "evaluator_ms": evaluator_ms,
        "rewrite_cycle_ms": rewrite_ms,
        "generation_ms": generation_ms,
        "total_ms": total_ms,
        "path": path,
    }


def main():
    sample = select_latency_sample()
    print(f"Timing {len(sample)} rows across both systems...\n")

    results = []
    for spec_id, topic_id, question, retrieval_level in sample:
        print(f"Timing {spec_id} ({retrieval_level})...")

        a_timing = time_system_a(question)
        b_timing = time_system_b(question)

        results.append({
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
        })
        time.sleep(T)  # long pause -- must fully clear the per-minute token budget between rows,
                         # since each row can issue up to 5 LLM calls with large context payloads
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n=== Latency summary (n={len(df)}) ===\n")
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