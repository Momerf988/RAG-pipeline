"""Re-scores the article's RAGAS metrics with a second judge from a different model family
(Google Gemini), so the article can show that its findings do not depend on one judge.

It reads the saved answers that 09_generate_system_a_dataset.py and
10_generate_system_b_dataset.py produced and that 11_run_ragas_evaluation.py scored with
gpt-4o-mini. Nothing is regenerated, and no existing file is changed: each set of scores goes
to its own new file in results/. Everything else matches script 11: RAGAS 0.4.3, the same
metrics and prompts, the same LangChain wrapper, the same embedding model for answer
relevancy, and the same row-by-row saving.

The work is split into four jobs, run in this order. A finished job is skipped.

    main       faithfulness, both systems                         about 600 judge calls
    recall     context recall, both systems                       about 300
    repeat     faithfulness of System A a second time, to         about 300
               measure the second judge against itself
    relevancy  answer relevancy, both systems                     about 900

Context precision is not re-scored. RAGAS makes one judge call per retrieved passage, which
is 15 per answer and 4,500 in all, and none of the article's conclusions rests on it.

Run from the top folder of the archive (the folder that contains src/ and results/):

    python src/16_second_judge.py --dry-run      # 5 rows of System A faithfulness, as a check
    python src/16_second_judge.py                # every job, in order
    python src/16_second_judge.py --job recall   # one job only

Needs GOOGLE_API_KEY in .env (create a key at https://aistudio.google.com). Answer relevancy
also needs sentence-transformers (in requirements-scoring.txt). The free tier has a daily
request limit. When it is reached the script stops; run the same command again the next day.
It skips every row already saved and carries on from where it stopped.
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
from ragas.metrics import answer_relevancy, context_recall, faithfulness
from ragas.run_config import RunConfig

import config  # loads .env, exactly as the other scripts do

# Second judge. A different model family from the first judge (gpt-4o-mini), the router and
# the benchmark generator (OpenAI) and from the tutor's generator (Llama), and in the same
# small, fast class as gpt-4o-mini. Called through Google's OpenAI-compatible endpoint, so it
# uses the same ChatOpenAI wrapper as script 11.
JUDGE2_MODEL = "gemini-3.5-flash-lite"
JUDGE2_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # the embedding model script 11 used for answer relevancy

INPUT_FILES = {   # the same inputs script 11 scored
    "A": "results/System_A_final_eval_results.csv",
    "B": "results/System_B_3way_eval_results.csv",
}
JUDGE1_FILES = {  # gpt-4o-mini scores, read only to print a side-by-side check
    "A": "results/System_A_final_scorecard.csv",
    "B": "results/System_B_3way_scorecard.csv",
}
JOBS = {   # run in this order
    "main": {
        "metric": faithfulness, "label": "faithfulness",
        "files": {"A": "results/System_A_judge2_faithfulness.csv",
                  "B": "results/System_B_judge2_faithfulness.csv"}},
    "recall": {
        "metric": context_recall, "label": "context recall",
        "files": {"A": "results/System_A_judge2_context_recall.csv",
                  "B": "results/System_B_judge2_context_recall.csv"}},
    "repeat": {
        "metric": faithfulness, "label": "faithfulness, second pass",
        "files": {"A": "results/System_A_judge2_faithfulness_repeat.csv"}},
    "relevancy": {
        "metric": answer_relevancy, "label": "answer relevancy",
        "files": {"A": "results/System_A_judge2_answer_relevancy.csv",
                  "B": "results/System_B_judge2_answer_relevancy.csv"}},
}
DRY_RUN_FILE = "results/judge2_dry_run.csv"
DRY_RUN_ROWS = 5
STOP_AFTER_FAILURES = 3   # consecutive API failures before stopping (usually a quota limit)
# Fewer, shorter RAGAS-level retries than its defaults, so a spent daily quota stops the run
# in minutes instead of retrying for an hour. Retries never change a score.
RUN_CONFIG = RunConfig(max_retries=3, max_wait=20)


def make_judge(separate_calls=False):
    """The second judge, set up the same way as gpt-4o-mini in script 11.

    Answer relevancy asks the judge for three questions per answer, at the temperature RAGAS
    sets for that step. With gpt-4o-mini they came from one call with n=3. With
    separate_calls=True they come from three separate calls instead, so that three questions
    are guaranteed whether or not the Gemini endpoint honours n. The other metrics make
    single-completion calls, for which the two settings send the same request."""
    return LangchainLLMWrapper(ChatOpenAI(
        api_key=GOOGLE_API_KEY,
        base_url=JUDGE2_BASE_URL,
        model=JUDGE2_MODEL,
        temperature=0,
        max_retries=3
    ), bypass_n=separate_calls)


def make_embeddings():
    """The embedding model script 11 used for answer relevancy."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    except ImportError:
        raise SystemExit("Answer relevancy needs the sentence-transformers package. Install it "
                         "with: pip install -r requirements-scoring.txt\nThen run the same "
                         "command again; everything saved so far is kept.")


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


def rows_saved(path):
    return len(pd.read_csv(path)) if os.path.exists(path) else 0


def show_progress(jobs):
    print("\nProgress (rows saved of 150):")
    for job in jobs:
        for system, path in JOBS[job]["files"].items():
            print(f"  {JOBS[job]['label'] + ', System ' + system:40s} {rows_saved(path):3d}")


def score_one(row_df, judge, metric, embeddings=None):
    """One RAGAS metric for one row. API errors are raised, so they are never saved as
    scores; a NaN from RAGAS itself (for example, no checkable statements) is returned."""
    result = evaluate(
        Dataset.from_pandas(row_df[['user_input', 'response', 'retrieved_contexts', 'reference']]
                            .reset_index(drop=True)),
        metrics=[metric],
        llm=judge,
        embeddings=embeddings,
        raise_exceptions=True,
        run_config=RUN_CONFIG,
        show_progress=False,
    )
    return result.to_pandas().iloc[0][metric.name]


def run_system(system, job, judge, pause, embeddings=None, dry_run=False):
    """Scores one system's rows for one job, saving after every row. Returns 'complete',
    'incomplete' (some rows failed and need another run) or 'stopped' (repeated failures,
    usually the daily limit)."""
    metric, label = JOBS[job]["metric"], JOBS[job]["label"]
    column = f"{metric.name}_judge2"
    out_file = DRY_RUN_FILE if dry_run else JOBS[job]["files"][system]
    df = load_rows(system)
    if dry_run:
        df = df.head(DRY_RUN_ROWS)

    if os.path.exists(out_file) and not dry_run:
        done = pd.read_csv(out_file)
        print(f"\nSystem {system}, {label}: resuming, {len(done)} of {len(df)} rows already saved.")
    else:
        done = pd.DataFrame()
        print(f"\nSystem {system}, {label}: starting{' (dry run)' if dry_run else ''}.")
    done_ids = set(done['spec_id']) if len(done) else set()

    failures = 0
    for i in range(len(df)):
        spec_id = df.iloc[i]['spec_id']
        if spec_id in done_ids:
            continue
        print(f"--- {system} {spec_id} ({i + 1}/{len(df)}) ", end="", flush=True)
        started = time.time()
        try:
            score = score_one(df.iloc[[i]], judge, metric, embeddings)
        except Exception as e:
            failures += 1
            print(f"failed: {type(e).__name__}: {str(e)[:200]}")
            if failures >= STOP_AFTER_FAILURES:
                print(f"\nStopped after {failures} failures in a row. This usually means the "
                      f"free tier's daily limit was reached. Run the same command again "
                      f"tomorrow; everything saved so far is kept.")
                return 'stopped'
            time.sleep(pause * 3)
            continue
        failures = 0
        new_row = pd.DataFrame([{
            'spec_id': spec_id,
            'topic_id': df.iloc[i]['topic_id'],
            column: score,
            'judge2_model': JUDGE2_MODEL,
            'scored_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        }])
        done = pd.concat([done, new_row], ignore_index=True)
        done.to_csv(out_file + '.tmp', index=False)
        os.replace(out_file + '.tmp', out_file)   # never leaves a half-written file
        done_ids.add(spec_id)
        shown = f"{score:.3f}" if not pd.isna(score) else (
            'NaN (no checkable statements)' if metric is faithfulness else 'NaN')
        print(f"{metric.name} {shown}  ({time.time() - started:.1f} s) saved")
        time.sleep(pause)

    print(f"System {system}, {label}: {len(done)} of {len(df)} rows saved to {out_file}")
    return 'complete' if len(done) == len(df) else 'incomplete'


def dry_run_report():
    """Side-by-side view of the dry-run rows against gpt-4o-mini."""
    d = pd.read_csv(DRY_RUN_FILE)
    j1 = pd.read_csv(JUDGE1_FILES['A'])[['spec_id', 'faithfulness']]
    m = d.merge(j1, on='spec_id', how='left').rename(columns={'faithfulness': 'gpt_4o_mini'})
    print("\nDry run: System A faithfulness, gpt-4o-mini vs the second judge")
    print(m[['spec_id', 'gpt_4o_mini', 'faithfulness_judge2']].round(3).to_string(index=False))
    print(f"\nIf these look sensible, run the full set: python src/16_second_judge.py"
          f"\n(The dry-run file {DRY_RUN_FILE} is only a check and is not used later.)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dry-run', action='store_true',
                        help=f'score the first {DRY_RUN_ROWS} rows of System A faithfulness only')
    parser.add_argument('--job', choices=['all'] + list(JOBS), default='all',
                        help='which job to run (default: all of them, in order)')
    parser.add_argument('--system', choices=['A', 'B', 'both'], default='both')
    parser.add_argument('--pause', type=float, default=8.0,
                        help='seconds to wait between rows, to stay under the per-minute limit')
    args = parser.parse_args(argv)

    for f in list(INPUT_FILES.values()):
        if not os.path.exists(f):
            raise SystemExit(f"{f} not found. Run this from the top folder of the archive "
                             f"(the folder that contains src/ and results/).")

    preflight()

    if args.dry_run:
        if os.path.exists(DRY_RUN_FILE):
            os.remove(DRY_RUN_FILE)
        if run_system('A', 'main', make_judge(), args.pause, dry_run=True) == 'complete':
            dry_run_report()
        return

    jobs = list(JOBS) if args.job == 'all' else [args.job]
    show_progress(jobs)
    judge, judge_separate_calls, embeddings = make_judge(), None, None
    complete = True
    for job in jobs:
        systems = [s for s in JOBS[job]["files"] if args.system in ('both', s)]
        todo = [s for s in systems if rows_saved(JOBS[job]["files"][s]) < len(load_rows(s))]
        if not todo:
            continue
        job_judge = judge
        if job == 'relevancy':
            if embeddings is None:
                embeddings = make_embeddings()
            if judge_separate_calls is None:
                judge_separate_calls = make_judge(separate_calls=True)
            job_judge = judge_separate_calls
        for system in todo:
            outcome = run_system(system, job, job_judge, args.pause,
                                 embeddings=embeddings if job == 'relevancy' else None)
            if outcome == 'stopped':
                show_progress(jobs)
                return
            complete = complete and outcome == 'complete'

    show_progress(jobs)
    if complete:
        print("\nEverything is scored. Next: python src/17_compare_judges.py")
    else:
        print("\nSome rows are not scored yet. Run the same command again; it scores only "
              "the rows that are missing.")


if __name__ == "__main__":
    main()
