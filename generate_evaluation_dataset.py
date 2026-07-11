import pandas as pd
import time
from app_init import llm_client, db_index, embedder_model
from user_query_processor import UserQueryProcessor
from RAG_response_processor import LLMResponseProcessor

user_query_processing_init = UserQueryProcessor(embedder_model)
LLM_response_processing_init = LLMResponseProcessor(llm_client, db_index)
 
df = pd.read_csv("data/BeginnerResponse.csv")
df = df.dropna(subset = ['Question'])
eval_data = []

print("Starting evaluation loop...")

for index, row in df.iterrows(): 
    question = row['Question']
    ground_truth = row['Answer']
    
    embedded_query = user_query_processing_init.vectorize_query(question)
    
    final_answer, context_list = LLM_response_processing_init.respond_to_user(
        embedded_query,
        question,
        'elaborate'
    )
    
    eval_data.append({
        'question': question,
        'contexts': context_list,
        'answer': final_answer,
        'ground_truth': ground_truth
    })
    
    print(f"Processed question {index + 1}")
    time.sleep(2)

eval_results_df = pd.DataFrame(eval_data)
eval_results_df.to_csv("System_A_eval_results.csv", index=False)
