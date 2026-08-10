# V2 work -- bounded three-way CRAG routing (System B). Updated to match the V2-fixed
# RAG_response_processor.py signatures (stich_context now returns chunk_id_list too;
# generate_response now returns finish_reason too) and to route on three evaluator verdicts
# (CORRECT / AMBIGUOUS / INCORRECT) instead of the old binary CORRECT / else.
from app_init import llm_client, db_index, embedder_model, reranker_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor
from crag_evaluator import CRAGEvaluator
import config

# v2 debug addition: print the exact chunk IDs and a preview of the context text that goes
# into evaluate_context, so an AMBIGUOUS/INCORRECT verdict can actually be inspected instead
# of trusted blindly -- added after a manual test session where "list" vs "tuple" queries got
# different verdicts and there was no way to see WHY without this.
CONTEXT_PREVIEW_CHARS = 400

def print_context_preview(label, context_string, chunk_ids):
    print(f">> [System B] {label} chunk IDs: {chunk_ids}")
    preview = context_string[:CONTEXT_PREVIEW_CHARS]
    suffix = "..." if len(context_string) > CONTEXT_PREVIEW_CHARS else ""
    print(f">> [System B] {label} context preview ({len(context_string)} chars total):\n{preview}{suffix}")

# --- THE SYSTEM B MAIN LOOP ---
if __name__ == "__main__":
    print("Starting System B (Corrective RAG, three-way) Tutor...")

    # Spin up all your modular processors
    query_processor = UserQueryProcessor(embedder_model)
    response_processor = LLMResponseProcessor(llm_client, db_index, reranker_model)
    evaluator = CRAGEvaluator(llm_client)

    while True:
        # 1. Get User Input
        user_prompt, detail_level = query_processor.get_user_preferences()
        if user_prompt.lower() == 'quit':
            break

        # 2. Standard Retrieval
        embedded_query = query_processor.vectorize_query(user_prompt)
        search_result = response_processor.search_db(embedded_query)
        reranked_result = response_processor.rerank(user_prompt, search_result)
        context_string, raw_chunks, chunk_ids = response_processor.stich_context(reranked_result)
        print_context_preview("First-pass", context_string, chunk_ids)

        # 3. The Interlock (Phase 1 Evaluation)
        print("\n>> [System B] Evaluating retrieved context...")
        eval_result = evaluator.evaluate_context(user_prompt, context_string)
        print(f">> [System B] Evaluator Decision: {eval_result}")

        # 4. Routing Logic -- bounded three-way
        if eval_result == "CORRECT":
            # Context is good. Generate answer directly. Path: "direct"
            system_persona, user_instruction = response_processor.LLM_prompt(user_prompt, detail_level, context_string)
            final_answer, finish_reason = response_processor.generate_response(system_persona, user_instruction)
            assert finish_reason == "stop", f"HARD ABORT: finish_reason={finish_reason}, answer may be truncated"
            if final_answer.strip() == config.REFUSAL_STRING:
                print(">> [System B] NOTE: CRAG judged the context CORRECT, but the tutor's own "
                      "internal check overrode that and refused anyway (disguised refusal).")
            print("\n--- TUTOR ---")
            print(final_answer)

        else:
            # AMBIGUOUS or INCORRECT both trigger ONE corrective attempt (bounded -- no loops,
            # no further retries after this). v2 change: AMBIGUOUS used to fall into the same
            # bucket as INCORRECT with no distinct treatment; it still does here at this first
            # branch (both rewrite), but the two verdicts are now treated differently on the
            # SECOND evaluation below, which is the actual point of the three-way upgrade.
            print(f">> [System B] Context {eval_result}. Initiating Corrective Action (Query Rewriting)...")
            rewritten_query = evaluator.rewrite_query(user_prompt)
            print(f">> [System B] Rewritten Query: {rewritten_query}")

            # Second Retrieval
            embedded_rewritten = query_processor.vectorize_query(rewritten_query)
            search_result_2 = response_processor.search_db(embedded_rewritten)
            reranked_result_2 = response_processor.rerank(rewritten_query, search_result_2)
            context_string_2, raw_chunks_2, chunk_ids_2 = response_processor.stich_context(reranked_result_2)
            print_context_preview("Second-pass", context_string_2, chunk_ids_2)

            # Second Evaluation
            print(">> [System B] Evaluating new context...")
            eval_result_2 = evaluator.evaluate_context(rewritten_query, context_string_2)
            print(f">> [System B] Second Evaluator Decision: {eval_result_2}")

            # v2 change: CORRECT and AMBIGUOUS on the second pass both generate an answer.
            # Rationale: two independent retrievals producing at least partial, on-topic
            # context is good enough to answer from -- only two independent INCORRECT
            # verdicts (nothing relevant found either time) trigger abstention. This is the
            # fix for the over-rejection V7 documented (items abstained on despite recall
            # of 1.000): AMBIGUOUS-after-retry no longer gets thrown away.
            if eval_result_2 in ("CORRECT", "AMBIGUOUS"):
                print(">> [System B] Second retrieval usable. Generating answer...")
                system_persona, user_instruction = response_processor.LLM_prompt(user_prompt, detail_level, context_string_2)
                final_answer, finish_reason = response_processor.generate_response(system_persona, user_instruction)
                assert finish_reason == "stop", f"HARD ABORT: finish_reason={finish_reason}, answer may be truncated"
                if final_answer.strip() == config.REFUSAL_STRING:
                    print(">> [System B] NOTE: CRAG judged the second-pass context usable, but the "
                          "tutor's own internal check overrode that and refused anyway (disguised refusal).")
                print("\n--- TUTOR ---")
                print(final_answer)

            else:
                # Both attempts independently judged INCORRECT -- graceful fallback.
                print(">> [System B] Second retrieval also INCORRECT. Triggering Graceful Fallback.")
                print("\n--- TUTOR ---")
                print("I'm sorry, that concept is outside the scope of our current syllabus or I couldn't find reliable information in the database to answer you accurately.")
