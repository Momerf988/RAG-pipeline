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

# Scores both systems' generation output with RAGAS: faithfulness and answer_relevancy
# characterise the generator, context_precision and context_recall characterise the
# retriever, judged throughout by gpt-4o-mini.
#
# One execution scores both systems, each with its own independent resume logic (separate
# scorecard files), so if this is interrupted partway through B, rerunning skips the
# now-complete A instantly and resumes B exactly where it left off. A missing input file
# (that system's generation hasn't finished yet) is reported and skipped rather than
# crashing the whole script.

SYSTEMS = ["A", "B"]

INPUT_FILES = {
    "A": "results/System_A_final_eval_results.csv",   # from 09_generate_system_a_dataset.py
    "B": "results/System_B_3way_eval_results.csv",    # from 10_generate_system_b_dataset.py
}
OUTPUT_FILES = {
    "A": "results/System_A_final_scorecard.csv",
    "B": "results/System_B_3way_scorecard.csv",
}


def run_evaluation(system_to_score):
    input_file = INPUT_FILES[system_to_score]
    output_file = OUTPUT_FILES[system_to_score]

    if not os.path.exists(input_file):
        gen_script = "09_generate_system_a_dataset.py" if system_to_score == "A" else "10_generate_system_b_dataset.py"
        print(f"SKIPPING System {system_to_score}: {input_file} not found yet "
              f"(generation for this system hasn't produced output). Run {gen_script} "
              f"first, then rerun this script.")
        return

    print(f"\n{'=' * 70}\nSCORING SYSTEM {system_to_score}\n{'=' * 70}")
    print(f"1. Loading {input_file}...")
    df = pd.read_csv(input_file)
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

    if os.path.exists(output_file):
        done_df = pd.read_csv(output_file)
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
            done_df.to_csv(output_file, index=False)
            done_spec_ids.add(spec_id)
            print(f"  saved.")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")

        time.sleep(2)

    print(f"\n5. Final averages for System {system_to_score}:\n")
    numeric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    available = [c for c in numeric_cols if c in done_df.columns]
    if len(done_df) > 0:
        print(done_df[available].mean())
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    for system in SYSTEMS:
        run_evaluation(system)
    print(f"\n{'=' * 70}\nALL SYSTEMS SCORED\n{'=' * 70}")
