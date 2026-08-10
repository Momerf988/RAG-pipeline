#ORIGINAL V1 FILE (10-08-2026)
from app_init import llm_client, db_index, embedder_model, reranker_model,pdf_dir
from document_processor import PDFDocumentProcessor
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor

#Execution: main application loop

if __name__ == "__main__":
    print("Starting Intelligent Tutor...")
    document_processing_init = PDFDocumentProcessor(pdf_dir, embedder_model, db_index)
#    document_processing_init.upsert_to_db()
    user_query_processing_init = UserQueryProcessor(embedder_model)
    LLM_response_processing_init = LLMResponseProcessor(llm_client, db_index, reranker_model)
    while True:
        user_prompt, detail_level = user_query_processing_init.get_user_preferences()
        if user_prompt.lower() == 'quit':
            break
        embedded_user_query = user_query_processing_init.vectorize_query(user_prompt)
        final_answer, context_list = LLM_response_processing_init.respond_to_user(embedded_user_query, user_prompt, detail_level)
        print(final_answer)