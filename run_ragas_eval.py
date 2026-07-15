'''
import pandas as pd
import ast
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import config
from ragas.run_config import RunConfig

def run_evaluation():
    df = pd.read_csv("System_A_eval_results.csv")
    df['contexts'] = df['contexts'].apply(ast.literal_eval)
    df = df.rename(columns = {
        "question": "user_input",
        "answer": "response",
        "contexts": "retrieved_contexts",
        "ground_truth": "reference"
    })
    eval_dataset = Dataset.from_pandas(df)
    judge_llm = ChatGroq(
        api_key = config.LLM_API_KEY,
        model_name = config.LLM_MODEL_NAME
    )

    judge_embeddings = HuggingFaceEmbeddings(
        model_name = config.TRANSFORMER_MODEL
    )
    all_results = []
    for index in range(len(df)):
        single_row_df = df.iloc[[index]]
        single_eval_dataset = Dataset.from_pandas(single_row_df)
        try:
            result = evaluate(
                eval_dataset,
                metrics = [faithfulness, answer_relevancy, context_precision, context_recall],
                llm = judge_llm,
                embeddings = judge_embeddings,
                raise_exceptions=False
                #run_config=RunConfig(max_workers=1, timeout=60)
            )
            score_dict = result.to_pandas().iloc[0].to_dict()
            all_results.append(score_dict)
        except Exception as e:
            print(f"failed to grade question {index + 1}: {e}")
        time.sleep(15)
    final_result_df = result.to_pandas()
    final_result_df.to_csv("System_A_Ragas_Scorecard.csv", index = False)
if __name__ == "__main__":
    run_evaluation()
'''

import pandas as pd
import ast
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import config
import warnings

# Suppress annoying background warnings
warnings.filterwarnings("ignore")

def run_evaluation():
    print("1. Loading evidence file...")
    df = pd.read_csv("System_A_eval_results.csv")
    
    print("2. Cleaning context data formats...")
    df['contexts'] = df['contexts'].apply(ast.literal_eval)
    
    df = df.rename(columns={
        "question": "user_input",
        "answer": "response",
        "contexts": "retrieved_contexts",
        "ground_truth": "reference"
    })
    
    print("3. Initializing Judges...")
    judge_llm = ChatGroq(
        api_key=config.LLM_API_KEY,
        model_name=config.LLM_MODEL_NAME,
        max_retries=3,
        max_tokens=4096
    )
    judge_embeddings = HuggingFaceEmbeddings(
        model_name=config.TRANSFORMER_MODEL
    )
    
    all_results = []
    
    print(f"\n4. Starting Drip-Feed Evaluation for {len(df)} questions...")
    print("This will take approximately 5-7 minutes. Please let it run...\n")
    
    for index in range(len(df)):
        print(f"--- Grading Question {index + 1}/{len(df)} ---")
        
        # Isolate exactly ONE row
        single_row_df = df.iloc[[index]]
        single_eval_dataset = Dataset.from_pandas(single_row_df)
        
        try:
            # Grade just this one question (Progress bar will only show 4 jobs instead of 80)
            result = evaluate(
                single_eval_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=judge_llm,
                embeddings=judge_embeddings,
                raise_exceptions=False 
            )
            
            score_dict = result.to_pandas().iloc[0].to_dict()
            all_results.append(score_dict)
            print(f"Score captured for Question {index + 1}.")
            
        except Exception as e:
            print(f"Failed to grade question {index + 1}: {e}")
            
        # 15-second cooldown to respect Groq's free-tier rate limits
        print("Cooling down for 15 seconds to prevent API blocks...")
        time.sleep(15)

    print("\n5. Assembling final scorecard...")
    final_results_df = pd.DataFrame(all_results)
    
    print("\n--- FINAL SYSTEM A AVERAGES ---")
    print(final_results_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean())
    
    final_results_df.to_csv("System_A_Ragas_Scorecard.csv", index=False)
    print("\nSaved to System_A_Ragas_Scorecard.csv!")

if __name__ == "__main__":
    run_evaluation()