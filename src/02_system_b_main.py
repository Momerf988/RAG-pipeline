from app_init import llm_client, db_index, embedder_model, reranker_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor
from crag_evaluator import CRAGEvaluator
import config

# System B -- corrective RAG. Retrieval is checked by CRAGEvaluator before generation; a
# weak first retrieval gets one rewrite-and-retry, and only two independent INCORRECT
# verdicts in a row cause the tutor to abstain. Interactive loop for manual testing; the
# batch benchmark run lives in 10_generate_system_b_dataset.py.

CONTEXT_PREVIEW_CHARS = 400


def print_context_preview(label, context_string, chunk_ids):
    print(f">> [System B] {label} chunk IDs: {chunk_ids}")
    preview = context_string[:CONTEXT_PREVIEW_CHARS]
    suffix = "..." if len(context_string) > CONTEXT_PREVIEW_CHARS else ""
    print(f">> [System B] {label} context preview ({len(context_string)} chars total):\n{preview}{suffix}")


if __name__ == "__main__":
    print("Starting Intelligent Tutor (System B, Corrective RAG)...")

    query_processor = UserQueryProcessor(embedder_model)
    response_processor = LLMResponseProcessor(llm_client, db_index, reranker_model)
    evaluator = CRAGEvaluator(llm_client)

    while True:
        user_prompt, detail_level = query_processor.get_user_preferences()
        if user_prompt.lower() == 'quit':
            break

        embedded_query = query_processor.vectorize_query(user_prompt)
        search_result = response_processor.search_db(embedded_query)
        reranked_result = response_processor.rerank(user_prompt, search_result)
        context_string, raw_chunks, chunk_ids = response_processor.stich_context(reranked_result)
        print_context_preview("First-pass", context_string, chunk_ids)

        print("\n>> [System B] Evaluating retrieved context...")
        eval_result = evaluator.evaluate_context(user_prompt, context_string)
        print(f">> [System B] Evaluator Decision: {eval_result}")

        if eval_result == "CORRECT":
            system_persona, user_instruction = response_processor.LLM_prompt_crag(user_prompt, detail_level, context_string)
            final_answer, finish_reason = response_processor.generate_response(system_persona, user_instruction)
            assert finish_reason == "stop", f"HARD ABORT: finish_reason={finish_reason}, answer may be truncated"
            if final_answer.strip() == config.REFUSAL_STRING:
                print(">> [System B] NOTE: CRAG judged the context CORRECT, but the tutor's own "
                      "internal check overrode that and refused anyway (disguised refusal).")
            print("\n--- TUTOR ---")
            print(final_answer)

        else:
            # AMBIGUOUS and INCORRECT both trigger one bounded corrective attempt -- rewrite,
            # retrieve again, evaluate again. No further retries after this.
            print(f">> [System B] Context {eval_result}. Initiating Corrective Action (Query Rewriting)...")
            rewritten_query = evaluator.rewrite_query(user_prompt)
            print(f">> [System B] Rewritten Query: {rewritten_query}")

            embedded_rewritten = query_processor.vectorize_query(rewritten_query)
            search_result_2 = response_processor.search_db(embedded_rewritten)
            reranked_result_2 = response_processor.rerank(rewritten_query, search_result_2)
            context_string_2, raw_chunks_2, chunk_ids_2 = response_processor.stich_context(reranked_result_2)
            print_context_preview("Second-pass", context_string_2, chunk_ids_2)

            print(">> [System B] Evaluating new context...")
            eval_result_2 = evaluator.evaluate_context(rewritten_query, context_string_2)
            print(f">> [System B] Second Evaluator Decision: {eval_result_2}")

            # CORRECT or AMBIGUOUS on the second pass both generate -- two independent
            # retrievals that are at least partially on-topic is treated as good enough to
            # answer from. Only two INCORRECT verdicts in a row falls back to abstention.
            if eval_result_2 in ("CORRECT", "AMBIGUOUS"):
                print(">> [System B] Second retrieval usable. Generating answer...")
                system_persona, user_instruction = response_processor.LLM_prompt_crag(user_prompt, detail_level, context_string_2)
                final_answer, finish_reason = response_processor.generate_response(system_persona, user_instruction)
                assert finish_reason == "stop", f"HARD ABORT: finish_reason={finish_reason}, answer may be truncated"
                if final_answer.strip() == config.REFUSAL_STRING:
                    print(">> [System B] NOTE: CRAG judged the second-pass context usable, but the "
                          "tutor's own internal check overrode that and refused anyway (disguised refusal).")
                print("\n--- TUTOR ---")
                print(final_answer)

            else:
                print(">> [System B] Second retrieval also INCORRECT. Triggering Graceful Fallback.")
                print("\n--- TUTOR ---")
                print("I'm sorry, that concept is outside the scope of our current syllabus or I couldn't find reliable information in the database to answer you accurately.")
