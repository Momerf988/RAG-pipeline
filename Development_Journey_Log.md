# Development Journey Log — System A (RAG) vs System B (CRAG) Evaluation Pipeline

This log reconstructs, in order, every problem encountered, the diagnosis behind it, the fix implemented, and the reasoning that justified it, across the full build-out of the automated System A vs System B evaluation pipeline. It is written to be lifted directly into dissertation sections (methodology, implementation, limitations) — each entry states what was wrong, why, what changed, and what evidence justified the decision.

Final verified state at time of writing (checked directly against the CSVs, not recalled from memory):

| File | Rows | Status |
|---|---|---|
| `System_A_final_eval_results.csv` | 150 / 150 | complete |
| `System_B_3way_eval_results.csv` | 150 / 150 | complete |
| `System_A_final_scorecard.csv` (RAGAS) | 150 / 150 | complete |
| `System_B_3way_scorecard.csv` (RAGAS) | 150 / 150 | complete |
| `Latency_Results_t1.csv` | 20 / 20 | complete |
| `keep_top_ablation_summary.csv` | 3 rows (k=7/15/20) | complete |

---

## 1. Retrieval depth (`keep_top`) ablation

**Problem:** the original implementation reranked and kept the top 7 chunks (`keep_top=7`) with no empirical justification on record.

**Decision process:** ran a targeted ablation at k=7 (matches the original V7 report, kept as a comparison anchor), k=15, and k=20 (k=10 excluded as not a meaningfully distinct practical contender). Deliberately run on **System A only** — duplicating it on System B was judged to weaken the study's central A-vs-B comparison rather than strengthen it, since System B's own retrieval-quality gate is the variable under test, not retrieval depth.

**Method:** `10_tune_keep_top_quality.py`, RAGAS quality metrics (faithfulness, answer_relevancy, context_precision, context_recall) on an n=24 sample, cross-checked against a separate span-recall ablation (`09_tune_retrieval.py`) that had already set a pre-declared +0.05 recall materiality bar.

**Result:** k=15 cleared the recall materiality bar by a wide margin, and *also* won on faithfulness (0.476) and answer_relevancy (0.668) versus both k=7 and k=20. k=20 had higher raw recall but measurably worse generated-answer quality (context dilution) — rejected despite the bigger recall number, since recall alone isn't the target, answer quality is.

**Implemented:** `RAG_response_processor.py`, `rerank()` default `keep_top` changed 7 → 15.

**Side effect disclosed later:** raising keep_top increases typical retrieved-context size, which increases typical "elaborate" answer length — this became directly relevant to a truncation issue encountered later (Section 6).

---

## 2. CRAG evaluator model selection

**Problem:** the CRAG evaluator was initially using the *same* model as the tutor generator (`llama-3.1-8b-instant`) to judge its own retrieved context — a same-model self-judging bias risk that weakens the independence of the CRAG evaluator's verdict.

**Research:** the original CRAG paper (Yan et al., 2024) uses a separately fine-tuned, lightweight T5-Large (770M) classifier — not a general-purpose prompted LLM, and not the same model as generation. Confirmed via an independent 2026 open-reproduction paper (arXiv:2603.16169) that the original T5-Large checkpoint's weights were never released ("the original implementation relies on proprietary components including ... closed model weights, limiting reproducibility"). Training an equivalent classifier from scratch was ruled out as outside MSc scope. Using a separate, prompted general LLM as the evaluator is confirmed as the standard adaptation used across CRAG reproductions.

**Candidates considered and rejected:**
- `gpt-oss-120b` — ~15× larger than the 8B tutor, inverting the paper's "lightweight evaluator" design intent regardless of cost.
- `qwen/qwen3.6-27b` — Preview-tier on Groq at the time ("may be discontinued at short notice" per Groq's own docs) — too risky against a hard deadline.
- Below 20B on Groq's catalog, the only smaller models were `llama-prompt-guard-2` variants (22M/86M) — specialised prompt-injection classifiers, wrong tool for relevance judgment.

**Decision:** `openai/gpt-oss-20b` — practical size floor (20B vs 120B), a different model family from the Llama tutor (reduces correlated same-family failure modes), Production-tier (not Preview) on Groq.

**Implemented:** `config.py` — added `EVALUATOR_MODEL_NAME = "openai/gpt-oss-20b"` with full rationale in comments. `crag_evaluator.py` — `evaluate_context()` switched to `config.EVALUATOR_MODEL_NAME`.

**Follow-on bug found and fixed:** gpt-oss-20b is a *reasoning* model — it spends part of its token budget on internal reasoning before producing the final verdict word, which was silently consuming the whole response budget. Fixed via `reasoning_effort="low"`, `include_reasoning=False`, and raising `max_tokens` 10 → 200, passed through the OpenAI SDK's `extra_body={}` (these aren't part of the standard typed API). Added a defensive empty-response warning as a safety net.

---

## 3. Evaluator recalibration

**Problem:** user smoke-tested the new evaluator with 3 prepared cases. Test 2 (context that mentions applications/tools but not a definition — a deliberately partial/ambiguous case) was scored **INCORRECT** instead of the intended **AMBIGUOUS**.

**Fix:** strengthened the evaluator prompt with an explicit "do NOT collapse AMBIGUOUS into INCORRECT" instruction, plus a second worked example matching the exact failure pattern (ML-applications-without-definition).

**Verification:** user reran the 3-case smoke test 3 times; confirmed all as expected.

---

## 4. Reproducibility audit (temperature = 0)

**Problem:** user noticed non-deterministic behaviour in System B — the same input query produced a different second-pass CRAG verdict across separate runs.

**Root cause:** `crag_evaluator.py`'s `rewrite_query()` was still running at `temperature=0.2`, while the project's standing rule (established earlier for the tutor generator) was temperature=0 wherever reproducibility is required. This method had been added *after* the original reproducibility sweep, so it was missed.

**Fix:** `rewrite_query()` temperature 0.2 → 0.0. Deliberately left on `config.LLM_MODEL_NAME` (the tutor model), not moved to the evaluator model — same-model bias is a self-*judging* concern, not a query-rewriting one.

**Full audit performed:** checked every other temperature setting in the active codebase. Confirmed `benchmark_question_generator.py`'s `temperature=0.3` is fine as-is — it's a one-time, already-frozen dataset-generation step, not part of any live or repeated experiment, so reproducibility doesn't apply to it in the same way.

---

## 5. System A "textbook RAG" fidelity fix

**Problem (caught by the user):** System A's generation prompt (`LLM_prompt` in `RAG_response_processor.py`) had its own independent sufficiency judgment bundled into the same call that wrote the answer — if it decided the context was insufficient, it returned a canned refusal string. This was a deliberate, previously-documented anti-hallucination safety addition, but it meant System A (meant to represent plain vanilla RAG) had its *own* retrieval-quality gate — which is specifically CRAG's (Yan et al., 2024) contribution over plain RAG (Lewis et al., 2020), which has no such gate and generates from whatever is retrieved, good or bad. This blurred the exact A-vs-B contrast the whole study exists to measure, since System A shouldn't have had any refusal capability of its own at all.

**Verification before changing anything:** checked whether this fix could invalidate the keep_top ablation (Section 1) already run under the old prompt. Queried the actual 72 keep_top-ablation rows (k=7/15/20) directly via pandas — confirmed **zero** of them ever triggered the old refusal string. The fix was therefore safe to apply without rerunning the ablation.

**Fix:** `LLM_prompt()` rewritten to drop the sufficiency judgment and refusal message entirely, keeping only the grounding rule (answer only from the given context, no outside facts — standard RAG hygiene, not CRAG's contribution). The hard `if not stiched_context` guard in `respond_to_user()` was left untouched, since that's a code-level empty-input guard, not a sufficiency judgment.

**Related fix, System B side:** a separate prompt, `LLM_prompt_crag()`, was added and used only by System B. Rationale: by the time System B calls this, CRAG's evaluator has already judged the context CORRECT/usable — that *is* System B's sufficiency check. The old shared prompt let the tutor independently re-judge and silently override CRAG's own routing decision (caught live during manual testing: CRAG said "usable", the tutor refused anyway — a "disguised refusal"). `generate_evaluation_dataset_B.py` now detects this case explicitly and relabels the path (`direct_but_refused` / `rewritten_but_refused`) so it's never miscounted as a genuine answer downstream.

**Verified outcome:** user reran System A live and confirmed it now attempts an answer from whatever context it has (the textbook vanilla-RAG failure mode CRAG exists to correct), rather than refusing — restoring a clean, demonstrable A-vs-B contrast.

---

## 6. Building the resumable, automated batch pipeline

Requirement (explicit, from the user): a fully automated pipeline that generates System A and System B answers for the full ~150-row benchmark, RAGAS-scores both without manual system-toggling, times both systems separately, and is safe to interrupt and resume at any point without losing or duplicating work — because it needed to run unattended overnight.

**`generate_evaluation_dataset.py` (System A batch):** fixed a crash bug — it still unpacked `respond_to_user()`'s return as 2 values after an earlier fix had made it return 3 (`final_answer, context_list, chunk_id_list`); would have crashed on row 1. Added `response_time_sec` (covers retrieval + rerank + generation combined, not generation alone — documented explicitly to avoid confusion with the stage-by-stage latency script). Renamed output to `System_A_final_eval_results.csv` — the old `System_A_eval_results.csv` was a stale keep_top=7-era file; reusing that name would have triggered the resume logic to see every row as already done and silently produce zero new rows. Both files were kept: the old one as a keep_top=7 baseline, the new one as the current run.

**`generate_evaluation_dataset_B.py` (System B batch):** confirmed its unpacking was already correct. Added the same `response_time_sec` timing, threaded through all three return paths (direct / rewritten / fallback). Output file was already correctly distinct.

**`run_ragas_eval.py`:** refactored from a manual "edit `SYSTEM_TO_SCORE = 'B'` and rerun" pattern into a `run_evaluation(system_to_score)` function called in a loop over `["A", "B"]`, so a single execution scores both systems automatically. Missing input files are reported and skipped, not crashed on — so one system's incomplete generation never blocks scoring the other. Filenames corrected to point at the current-config outputs (`System_A_final_*`, `System_B_3way_*`), not the stale 03/08 keep_top=7 / binary-evaluator files that were still sitting on disk. This script already had full per-row resume logic (reads the scorecard CSV, tracks done spec_ids, skips them) from the start — confirmed on rereading, after mistakenly telling the user otherwise mid-session (see Section 9).

**`measure_latency.py`:** fixed four stale bugs found before ever running it: `stich_context()` unpacked as 2 values instead of 3 in three call sites; a `generate_response()` return value was mishandled; System B's timing used `LLM_prompt` (System A's prompt) instead of the actual production `LLM_prompt_crag`; the routing condition only checked `decision_2 == "CORRECT"` instead of matching production's `in ("CORRECT", "AMBIGUOUS")`; and `OUTPUT_FILE = f"Latency_Results_t{{T}}.csv"` had a double-brace bug producing a literal `{T}` instead of interpolating the variable.

**`analyse_results.py`, `verify_all.py`, `verify_statistics.py`:** all three "regenerate every number in the report" scripts were found to still hardcode the stale filenames — updated to point at the new `_final` / `_3way` files. `verify_statistics.py` also had an unrelated bug (missing `data/` prefix on the benchmark spec matrix path).

**`app_init.py`:** added print statements around each initialisation step (LLM client, Pinecone, embedder load, reranker load), since loading the embedder/reranker models silently takes several seconds with zero output — a contributing cause of the "frozen screen" symptom investigated in Section 7.

**`run_overnight.sh`:** new orchestrator running, in order, System A generation → System B generation → RAGAS scoring (both) → latency measurement. Order is deliberate: the two RAGAS-critical steps run first since they're what the dissertation actually reports; latency runs last since it's supplementary and shouldn't risk the more important data.

**Resume-safety pattern used throughout:** every batch script checks its own output CSV on startup, builds a set of already-completed spec_ids, and skips them — combined with saving after every single row (not batching writes), this means interruption at any point (crash, quota exhaustion, closed terminal, deliberate Ctrl+C) loses at most the row in progress, never anything already completed. Rows are fully isolated from each other — no cross-row dependencies anywhere in the pipeline, confirmed explicitly when the user asked.

---

## 7. Real-world execution failures during the actual overnight runs

These were only discoverable by actually running the full pipeline against the live Groq account — each surfaced a real constraint that no amount of code review alone would have caught.

### 7.1 Truncation crash (Run 1 — stopped after 33 rows)

**Symptom:** crash on row 34/150 with `AssertionError: HARD ABORT: finish_reason=length` — an answer had been truncated by the token limit.

**Fix:** raised `TUTOR_MAX_TOKENS` 1024 → 2048 in `config.py` (commented rationale: keep_top=15 gives the tutor more context to work with on "elaborate" answers, increasing typical answer length). Added a self-healing retry directly in `generate_response()` (`RAG_response_processor.py`): on `finish_reason == "length"`, retry once at a higher token budget before ever returning a truncated result. Kept the batch scripts' `AssertionError`-catch-and-skip as a last-resort safety net so a row that still fails after the retry is logged and skipped (not marked done, auto-picked-up on the next run) rather than crashing the whole batch.

### 7.2 413 "request too large" (Run 2)

**Symptom:** crash with `openai.APIStatusError` — a live 413 error: `Limit 6000, Requested 11076`.

**Root cause:** the retry fix from 7.1 was originally set to retry at 4× the configured budget (2048 × 4 = 8192). Combined with keep_top=15's larger retrieved context (~2,600–3,000 tokens), this blew straight through the account's real per-minute ceiling — discovered live to be **6,000 TPM** (tokens per minute, prompt + completion combined), not the ~250K figure shown on Groq's general docs page (which applies to a paid "Developer Plan" tier this account doesn't have — confirmed via Groq's own billing page: "Developer tier upgrades are temporarily unavailable due to high demand").

**Fix:** reduced the retry bump to a modest `+512` (2,560 total) — comfortably under 6,000 TPM even with the largest realistic context. Added `except APIStatusError` handling to both generation scripts: this class of error is *not* retried, since retrying an identically-sized request fails identically every time (it's a structural size problem, not a timing one) — logged and skipped instead.

**Methodological note directly relevant to reproducibility framing:** this confirmed the project's standing practice of trusting *live evidence* (actual error payloads) over documentation once ambiguity arises, since the account's real tier didn't match the publicly documented figures.

### 7.3 "Frozen screen" / stdout buffering

**Symptom:** user reported the terminal appeared to freeze after starting an overnight run, with no new output for extended periods, especially after a Ctrl+C-and-restart.

**Root cause:** piping script output through `tee` (for logging) changes Python's stdout buffering from line-buffered (interactive terminal) to block-buffered — `print()` output was being held in a buffer rather than appearing immediately, even though the script was working correctly underneath.

**Fix:** added `python -u` (unbuffered) to every script invocation in `run_overnight.sh`. Combined with the `app_init.py` progress prints (Section 6), this ensures visible, continuous output from the first few seconds of any run.

**Related shell fix:** `run_overnight.sh` uses `set -eo pipefail`, not `set -e` alone — `pipefail` is necessary because output is piped through `tee`, and `tee` itself almost always exits 0 regardless of whether the Python step it's wrapping actually failed; without `pipefail`, a real crash could be silently masked and the script would continue on broken/missing data.

### 7.4 429 "tokens per day" (Run 3) — the daily quota ceiling

**Symptom:** the most complete run attempted. System A reached 82/150 rows (one row, `T3_004`, skipped even after the 7.2 retry also truncated; the run then stopped gracefully at `T6_014` after 3 rate-limit retries with 30/60/90s backoff, saving progress cleanly — not a crash). System B got **0/150** rows — hit a rate limit immediately on the first row, retried 3×, gave up. RAGAS scoring correctly and successfully scored all 82 available System A rows, and correctly skipped System B (its input file didn't exist yet, since zero rows had been written). `measure_latency.py` then crashed uncaught with a previously unseen error: `openai.RateLimitError` citing **tokens per day (TPD): Limit 500000, Used 499959, Requested 3164. Please try again in 8m59s.`

**Diagnosis:** this revealed a third, previously unknown Groq constraint — a **500,000 tokens/day** cap, entirely separate from the 6,000 TPM per-minute cap in 7.2, shared cumulatively across *every* call made that day (System A generation, System B generation, the CRAG evaluator, the latency script). By the time the run reached System B, the day's budget was already nearly exhausted, which is why System B failed immediately (it needs up to 4 Groq calls per row versus System A's 1, so it hit empty first) and why `measure_latency.py` crashed shortly after.

**Also explained:** why `T3_004` behaved differently from an earlier informal comparison run the user recalled ("V1") — that earlier run used the old keep_top=7 configuration, which produces smaller retrieved contexts and therefore a lower truncation risk on the same question. Not a regression or a new bug — a disclosed, direct side effect of the deliberate keep_top=15 decision (Section 1) interacting with a specific question's answer length.

**Resolution at the time:** no code fix exists for an exhausted daily account quota — this is a genuine external resource ceiling, not a bug. Advised stopping for the night; the resume-safe architecture meant the 82 completed and already-RAGAS-scored System A rows were real, valid, retained progress, not wasted work.

### 7.5 `measure_latency.py` resume gap (found the next morning)

**Problem:** unlike the two generation scripts, `measure_latency.py` had no resume or incremental-save logic at all — results were only written to disk once, at the very end of `main()`. This meant the 7.4 crash had discarded every row it had already timed, unlike System A/B's generation, which had preserved all progress.

**Fix:** brought it in line with the same pattern already used elsewhere — reads its own `OUTPUT_FILE` on startup, builds a set of already-timed spec_ids and skips them, saves after every row (not just at the end), retries `RateLimitError` 3× with backoff then stops cleanly (keeping partial progress) instead of crashing, and skips (rather than crashes on) `AssertionError`/`APIStatusError` the same way the generation scripts do.

---

## 8. Migration from Groq to OpenRouter

**Trigger:** the 500K TPD ceiling (7.4) meant a single Groq free-tier account could not realistically finish the remaining ~68 System A rows plus all 150 System B rows in one sitting, and Groq's paid tier was confirmed unavailable ("Developer tier upgrades are temporarily unavailable due to high demand"). User asked whether the same two models could be served by a different provider to remove the daily cap entirely, given no time to spare.

**Research (live, verified against OpenRouter's current model pages, not assumed):** OpenRouter hosts both models needed, via the same OpenAI-compatible client interface already used throughout the codebase:
- `openai/gpt-oss-20b` — identical model slug to the one already in `config.py`, $0.03 / $0.13 per 1M input/output tokens.
- `meta-llama/llama-3.1-8b-instruct` — same underlying weights as Groq's `llama-3.1-8b-instant` (Groq's "-instant" suffix denotes its own speed-optimised serving of the same base model, not a different model), $0.02 / $0.04 per 1M tokens.

At these rates, the full remaining dataset (both systems) was estimated at well under a few dollars total — not a meaningful cost concern. Paid OpenRouter models are governed by account balance, not a fixed daily token cap, which directly solves the 7.4 constraint.

**Change required:** config-only, no code changes — three `.env` values (`LLM_API_KEY`, `LLM_SERVER_URL`, `LLM_MODEL_NAME`). `EVALUATOR_MODEL_NAME` in `config.py` needed no change at all, since its slug was already identical on OpenRouter.

**Bug hit during the switch:** user set `LLM_MODEL_NAME = "llama/llama-3.1-8b-instruct"` (missing the `meta-` org prefix) → `openai.BadRequestError: 400 - "llama/llama-3.1-8b-instruct is not a valid model ID"`. Diagnosed as a model-ID validation error, unrelated to the unpaid OpenRouter balance at the time (a billing/credit issue would surface as a different error class, not an invalid-model-ID 400). Fixed by correcting the slug to `meta-llama/llama-3.1-8b-instruct`.

**Disclosed methodological caveat (belongs in limitations):** the 82 System A rows generated before this point ran on Groq's `llama-3.1-8b-instant`; all rows generated after this point ran on OpenRouter's `meta-llama/llama-3.1-8b-instruct`, which itself load-balances across roughly five backend hosting providers. Same base model weights, different serving stack/quantization path — a disclosed, minor provider-level inconsistency introduced by a real infrastructure constraint (Groq's daily quota), not a methodological oversight.

**Outcome:** full run completed successfully. Verified directly against the CSVs (not recalled): System A 150/150, System B 150/150, both RAGAS scorecards 150/150, latency 20/20 — see the table at the top of this document.

---

## 9. Process notes worth including in a methodology/reflection section

- **Live evidence over documentation:** every real rate-limit constraint on this account (6,000 TPM, then 500,000 TPD) turned out to differ from what Groq's public docs implied, because those docs describe a paid tier this account never had. Each was only discovered by reading the actual error payload from a live failed request, not by re-reading documentation more carefully. This is a fair thing to note in a methodology section on working with third-party LLM APIs under free-tier constraints.
- **A self-correction, disclosed honestly:** at one point `run_ragas_eval.py` was incorrectly described as not being row-resumable, causing understandable alarm about wasted OpenAI spend. On rereading the actual code, it already had complete per-row resume logic from when it was first built — the claim was simply wrong, not a reflection of an actual gap. No code change was needed; only the earlier (incorrect) statement needed correcting.
- **Verification-before-change discipline:** every fix that could plausibly have invalidated already-completed work (the System A textbook-RAG fix, most notably) was checked against the actual saved data (via pandas, not assumption) *before* being applied, specifically to confirm zero reruns of prior ablations were required.

---

## 10. Suggested mapping to write-up sections

- **Methodology — evaluator design:** Sections 2–4 (evaluator model selection and its literature justification, recalibration, temperature/reproducibility controls).
- **Methodology — retrieval configuration:** Section 1 (keep_top ablation, decision criteria, materiality bar).
- **Methodology — system fidelity / experimental design:** Section 5 (why System A needed the refusal-logic removal to make the A-vs-B contrast valid, and the verification that this didn't invalidate other completed work).
- **Implementation:** Section 6 (resumable pipeline architecture, error-handling design, why each script is structured the way it is).
- **Limitations:** Sections 7 and 8 — free-tier API constraints (per-minute and per-day token caps), and the disclosed cross-provider serving inconsistency for the tail of the System A dataset.
- **Reflection / process:** Section 9.
