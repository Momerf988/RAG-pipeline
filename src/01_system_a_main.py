from app_init import llm_client, db_index, embedder_model, reranker_model, pdf_dir
from document_processor import PDFDocumentProcessor
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor

# System A -- plain retrieve-rerank-generate RAG, no correction step. Interactive loop for
# manual testing; the batch benchmark run lives in 09_generate_system_a_dataset.py.

if __name__ == "__main__":
    print("Starting Intelligent Tutor (System A)...")
    document_processing_init = PDFDocumentProcessor(pdf_dir, embedder_model, db_index)
    # document_processing_init.upsert_to_db()   # uncomment to (re)build the Pinecone index

    user_query_processing_init = UserQueryProcessor(embedder_model)
    LLM_response_processing_init = LLMResponseProcessor(llm_client, db_index, reranker_model)

    while True:
        user_prompt, detail_level = user_query_processing_init.get_user_preferences()
        if user_prompt.lower() == 'quit':
            break
        embedded_user_query = user_query_processing_init.vectorize_query(user_prompt)
        final_answer, context_list, chunk_id_list = LLM_response_processing_init.respond_to_user(
            embedded_user_query, user_prompt, detail_level)
        print(final_answer)
