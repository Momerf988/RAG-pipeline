"""Re-scores RAGAS faithfulness for both systems with a second judge from a different model
family (Google Gemini), so the article can show that its findings do not depend on one judge.

It reads the saved answers that 09_generate_system_a_dataset.py and
10_generate_system_b_dataset.py produced and that 11_run_ragas_evaluation.py scored with
gpt-4o-mini. Nothing is regenerated, and no existing file is changed: the scores go to two
new files in results/. Everything else matches script 11: RAGAS 0.4.3, the same faithfulness
metric and prompts, the same LangChain wrapper and the same row-by-row saving.

Run from the top folder of the archive (the folder that contains src/ and results/):

    python src/16_second_judge_faithfulness.py --dry-run     # 5 rows of System A, as a check
    python src/16_second_judge_faithfulness.py               # all 150 rows of each system

Needs GOOGLE_API_KEY in .env (create a key at https://aistudio.google.com). The free tier has
daily request limits. If the script stops because a limit was reached, run the same command
again later: it skips every row already saved and carries on from where it stopped.
"""
import argparse
import ast
import os
import time
import warnings
from datetime import datetime, timezone

import pandas as pd

warnings.filterwarnings("ignore")  # before the RAGAS imports, which warn on import

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness
from ragas.run_config import RunConfig

import config  # loads .env, exactly as the other scripts do

# Second judge. A different model family from the first judge (gpt-4o-mini), the router and
# the benchmark generator (OpenAI) and from the tutor's generator (Llama), and in the same
# small, fast class as gpt-4o-mini. Called through Google's OpenAI-compatible endpoint, so it
# uses the same ChatOpenAI wrapper as script 11 and needs no extra packages.
JUDGE2_MODEL = "gemini-3.5-flash-lite"
JUDGE2_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

SYSTEMS = ["A", "B"]
INPUT_FILES = {   # the same inputs script 11 scored
    "A": "results/System_A_final_eval_results.csv",
    "B": "results/System_B_3way_eval_results.csv",
}
JUDGE1_FILES = {  # gpt-4o-mini scores, read only to print a side-by-side check
    "A": "results/System_A_final_scorecard.csv",
    "B": "results/System_B_3way_scorecard.csv",
}
OUTPUT_FILES = {
    "A": "results/System_A_judge2_faithfulness.csv",
    "B": "results/System_B_judge2_faithfulness.csv",
}
DRY_RUN_FILE = "results/judge2_dry_run.csv"
DRY_RUN_ROWS = 5
STOP_AFTER_FAILURES = 3   # consecutive API failures before stopping (usually a quota limit)
# Fewer, shorter RAGAS-level retries than its defaults, so a spent daily quota stops the run
# in minutes instead of retrying for an hour. Retries never change a score.
RUN_CONFIG = RunConfig(max_retries=3, max_wait=20)


def make_judge():
    """The second judge, set up the same way as gpt-4o-mini in script 11."""
    return LangchainLLMWrapper(ChatOpenAI(
        api_key=GOOGLE_API_KEY,
        base_url=JUDGE2_BASE_URL,
        model=JUDGE2_MODEL,
        temperature=0,
        max_retries=3
    ))


def preflight():
    """One tiny request, so a wrong key or model name fails here with a clear message."""
    if not GOOGLE_API_KEY:
        raise SystemExit("GOOGLE_API_KEY is not set. Add the line GOOGLE_API_KEY=your-key to "
                         "the .env file in the top folder, then run this again.")
    from openai import OpenAI
    client = OpenAI(api_key=GOOGLE_API_KEY, base_url=JUDGE2_BASE_URL)
    try:
        reply = client.chat.completions.create(
            model=JUDGE2_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            temperature=0,
        )
    except Exception as e:
        raise SystemExit(f"Could not reach {JUDGE2_MODEL}: {type(e).__name__}: {e}\n"
                         f"Check the key in .env and that the model name is still offered "
                         f"by the Gemini API.")
    print(f"Connected to {JUDGE2_MODEL} (reply: {reply.choices[0].message.content!r})")


def load_rows(system):
    """The saved answers, prepared exactly as script 11 prepared them."""
    df = pd.read_csv(INPUT_FILES[system])
    df['contexts'] = df['contexts'].apply(ast.literal_eval)
    return df.rename(columns={
        "question": "user_input",
        "answer": "response",
        "contexts": "retrieved_contexts",
        "ground_truth": "reference"
    })


def score_one(row_df, judge):
    """RAGAS faithfulness for one row. API errors are raised, so they are never saved as
    scores; a NaN from RAGAS itself (no checkable statements) is returned as NaN."""
    result = evaluate(
        Dataset.from_pandas(row_df[['user_input', 'response', 'retrieved_contexts', 'reference']]
                            .reset_index(drop=True)),
        metrics=[faithfulness],
        llm=judge,
        raise_exceptions=True,
        run_config=RUN_CONFIG,
        show_progress=False,
    )
    return result.to_pandas().iloc[0]['faithfulness']


def run_system(system, judge, pause, dry_run=False):
    """Scores one system's rows, saving after every row. Returns False if it had to stop."""
    out_file = DRY_RUN_FILE if dry_run else OUTPUT_FILES[system]
    df = load_rows(system)
    if dry_run:
        df = df.head(DRY_RUN_ROWS)

    if os.path.exists(out_file) and not dry_run:
        done = pd.read_csv(out_file)
        print(f"\nSystem {system}: resuming, {len(done)} of {len(df)} rows already saved.")
    else:
        done = pd.DataFrame()
        print(f"\nSystem {system}: starting{' (dry run)' if dry_run else ''}.")
    done_ids = set(done['spec_id']) if len(done) else set()

    failures = 0
    for i in range(len(df)):
        spec_id = df.iloc[i]['spec_id']
        if spec_id in done_ids:
            continue
        print(f"--- {system} {spec_id} ({i + 1}/{len(df)}) ", end="", flush=True)
        started = time.time()
        try:
            score = score_one(df.iloc[[i]], judge)
        except Exception as e:
            failures += 1
            print(f"failed: {type(e).__name__}: {str(e)[:200]}")
            if failures >= STOP_AFTER_FAILURES:
                print(f"\nStopped after {failures} failures in a row. This usually means the "
                      f"free tier's limit was reached. Run the same command again later; "
                      f"everything saved so far is kept.")
                return False
            time.sleep(pause * 3)
            continue
        failures = 0
        new_row = pd.DataFrame([{
            'spec_id': spec_id,
            'topic_id': df.iloc[i]['topic_id'],
            'faithfulness_judge2': score,
            'judge2_model': JUDGE2_MODEL,
            'scored_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        }])
        done = pd.concat([done, new_row], ignore_index=True)
        done.to_csv(out_file + '.tmp', index=False)
        os.replace(out_file + '.tmp', out_file)   # never leaves a half-written file
        done_ids.add(spec_id)
        shown = 'NaN (no checkable statements)' if pd.isna(score) else f"{score:.3f}"
        print(f"faithfulness {shown}  ({time.time() - started:.1f} s) saved")
        time.sleep(pause)

    print(f"System {system}: {len(done)} of {len(df)} rows saved to {out_file}")
    return True


def dry_run_report():
    """Side-by-side view of the dry-run rows against gpt-4o-mini."""
    d = pd.read_csv(DRY_RUN_FILE)
    j1 = pd.read_csv(JUDGE1_FILES['A'])[['spec_id', 'faithfulness']]
    m = d.merge(j1, on='spec_id', how='left').rename(columns={'faithfulness': 'gpt_4o_mini'})
    print("\nDry run: System A faithfulness, gpt-4o-mini vs the second judge")
    print(m[['spec_id', 'gpt_4o_mini', 'faithfulness_judge2']].round(3).to_string(index=False))
    print(f"\nIf these look sensible, run the full set: python src/16_second_judge_faithfulness.py"
          f"\n(The dry-run file {DRY_RUN_FILE} is only a check and is not used later.)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dry-run', action='store_true',
                        help=f'score the first {DRY_RUN_ROWS} rows of System A only, as a check')
    parser.add_argument('--system', choices=['A', 'B', 'both'], default='both')
    parser.add_argument('--pause', type=float, default=8.0,
                        help='seconds to wait between rows, to stay under the per-minute limit')
    args = parser.parse_args(argv)

    for f in list(INPUT_FILES.values()):
        if not os.path.exists(f):
            raise SystemExit(f"{f} not found. Run this from the top folder of the archive "
                             f"(the folder that contains src/ and results/).")

    preflight()
    judge = make_judge()

    if args.dry_run:
        if os.path.exists(DRY_RUN_FILE):
            os.remove(DRY_RUN_FILE)
        if run_system('A', judge, args.pause, dry_run=True):
            dry_run_report()
        return

    for system in (SYSTEMS if args.system == 'both' else [args.system]):
        if not run_system(system, judge, args.pause):
            return
    print("\nAll rows scored. Next: python src/17_compare_judges.py")


if __name__ == "__main__":
    main()
