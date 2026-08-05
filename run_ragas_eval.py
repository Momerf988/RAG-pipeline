import pandas as pd
import ast
import time
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import config
import warnings

warnings.filterwarnings("ignore")

# --- EDIT THIS BEFORE EACH RUN ---
SYSTEM_TO_SCORE = "B"    # then rerun with "B"

INPUT_FILE = f"System_{SYSTEM_TO_SCORE}_eval_results.csv"
OUTPUT_FILE = f"System_{SYSTEM_TO_SCORE}_scorecard.csv"

def run_evaluation():
    print(f"1. Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   {len(df)} rows loaded")

    print("2. Parsing contexts...")
    df['contexts'] = df['contexts'].apply(ast.literal_eval)
    df = df.rename(columns={
        "question": "user_input",
        "answer": "response",
        "contexts": "retrieved_contexts",
        "ground_truth": "reference"
    })

    print("3. Initializing OpenAI judge (gpt-4o-mini)...")
    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        api_key=config.OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0,
        max_retries=3
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=config.TRANSFORMER_MODEL
    ))

    if os.path.exists(OUTPUT_FILE):
        done_df = pd.read_csv(OUTPUT_FILE)
        done_spec_ids = set(done_df['spec_id'].tolist())
        print(f"Resuming — {len(done_spec_ids)} rows already scored.")
    else:
        done_df = pd.DataFrame()
        done_spec_ids = set()
        print("Starting fresh.")

    print(f"\n4. Scoring {len(df)} rows on 4 metrics each...\n")

    for index in range(len(df)):
        spec_id = df.iloc[index]['spec_id']
        if spec_id in done_spec_ids:
            continue

        print(f"--- Scoring {spec_id} ({index + 1}/{len(df)}) ---")
        single_row_df = df.iloc[[index]][['user_input', 'response', 'retrieved_contexts', 'reference']]
        single_eval_dataset = Dataset.from_pandas(single_row_df)

        try:
            result = evaluate(
                single_eval_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=judge_llm,
                embeddings=judge_embeddings,
                raise_exceptions=False
            )
            score_dict = result.to_pandas().iloc[0].to_dict()
            score_dict['spec_id'] = spec_id
            score_dict['topic_id'] = df.iloc[index]['topic_id']

            new_row = pd.DataFrame([score_dict])
            done_df = pd.concat([done_df, new_row], ignore_index=True)
            done_df.to_csv(OUTPUT_FILE, index=False)
            done_spec_ids.add(spec_id)
            print(f"  saved.")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")

        time.sleep(2)

    print(f"\n5. Final averages for System {SYSTEM_TO_SCORE}:\n")
    numeric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    available = [c for c in numeric_cols if c in done_df.columns]
    print(done_df[available].mean())
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()