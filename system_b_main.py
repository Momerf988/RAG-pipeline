from app_init import llm_client, db_index, embedder_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor
from crag_evaluator import CRAGEvaluator
import config

# --- THE CORRECTIVE ACTION ENGINE ---
def rewrite_query(user_query):
    """Rewrites a failed query into betteri w domain terminology."""
    prompt = f"Rewrite the following user query to use more precise academic terminology suitable for searching a University AI and Python syllabus. Output ONLY the rewritten query, nothing else.\nQuery: {user_query}"
    
    response = llm_client.chat.completions.create(
        model=config.LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2 # Slight creativity to allow good rewriting
    )
    return response.choices[0].message.content.strip()

# --- THE SYSTEM B MAIN LOOP ---
if __name__ == "__main__":
    print("Starting System B (Corrective RAG) Tutor...")
    
    # Spin up all your modular processors
    query_processor = UserQueryProcessor(embedder_model)
    response_processor = LLMResponseProcessor(llm_client, db_index)
    evaluator = CRAGEvaluator(llm_client)

    while True:
        # 1. Get User Input
        user_prompt, detail_level = query_processor.get_user_preferences()
        if user_prompt.lower() == 'quit':
            break

        # 2. Standard Retrieval
        embedded_query = query_processor.vectorize_query(user_prompt)
        search_result = response_processor.search_db(embedded_query)
        context_string, raw_chunks = response_processor.stich_context(search_result)

        # 3. The Interlock (Phase 1 Evaluation)
        print("\n>> [System B] Evaluating retrieved context...")
        eval_result = evaluator.evaluate_context(user_prompt, context_string)
        print(f">> [System B] Evaluator Decision: {eval_result}")

        # 4. Routing Logic
        if eval_result == "CORRECT":
            # Context is good. Generate answer normally.
            system_persona, user_instruction = response_processor.LLM_prompt(user_prompt, detail_level, context_string)
            final_answer = response_processor.generate_response(system_persona, user_instruction)
            print("\n--- TUTOR ---")
            print(final_answer)

        elif eval_result == "INCORRECT":
            # Context is bad. Trip the interlock and rewrite!
            print(">> [System B] Context rejected. Initiating Corrective Action (Query Rewriting)...")
            rewritten_query = rewrite_query(user_prompt)
            print(f">> [System B] Rewritten Query: {rewritten_query}")

            # Second Retrieval
            embedded_rewritten = query_processor.vectorize_query(rewritten_query)
            search_result_2 = response_processor.search_db(embedded_rewritten)
            context_string_2, raw_chunks_2 = response_processor.stich_context(search_result_2)

            # Second Evaluation
            print(">> [System B] Evaluating new context...")
            eval_result_2 = evaluator.evaluate_context(rewritten_query, context_string_2)
            print(f">> [System B] Second Evaluator Decision: {eval_result_2}")

            if eval_result_2 == "CORRECT":
                print(">> [System B] Second retrieval successful. Generating answer...")
                system_persona, user_instruction = response_processor.LLM_prompt(user_prompt, detail_level, context_string_2)
                final_answer = response_processor.generate_response(system_persona, user_instruction)
                print("\n--- TUTOR ---")
                print(final_answer)
                
            else:
                # The True Cure to Hallucination
                print(">> [System B] Second retrieval failed. Triggering Graceful Fallback.")
                print("\n--- TUTOR ---")
                print("I'm sorry, that concept is outside the scope of our current syllabus or I couldn't find reliable information in the database to answer you accurately.")