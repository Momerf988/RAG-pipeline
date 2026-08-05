from app_init import llm_client, db_index, embedder_model, reranker_model
from RAG_response_processor import LLMResponseProcessor
from user_query_processor import UserQueryProcessor

query_processor = UserQueryProcessor(embedder_model)
response_processor = LLMResponseProcessor(llm_client, db_index, reranker_model)

test_query = "What is an f-string in Python?"
embedded = query_processor.vectorize_query(test_query)

raw_matches = response_processor.search_db(embedded)
print(f"Raw matches from Pinecone (top_k=20): {len(raw_matches)}")

reranked = response_processor.rerank(test_query, raw_matches)
print(f"After reranking, kept: {len(reranked)}")

print("\nTop 3 reranked chunks:")
for i, match in enumerate(reranked[:3]):
    print(f"\n[{i+1}] score not shown here, but content:")
    print(f"  {match['metadata']['line'][:150]}")