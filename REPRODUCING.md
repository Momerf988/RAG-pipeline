# Code and data for: Localising failure in retrieval-augmented tutoring

This archive accompanies the article "Localising failure in retrieval-augmented
tutoring: Standard RAG versus a corrective retrieval interlock in an introductory Python
tutor" by Muhammad Omer Farooq (Ulster University).

It contains the code for both tutoring systems (System A, standard RAG; System B, with the
corrective retrieval interlock), the 150-item benchmark, every system output and RAGAS
score reported in the article, and the ablation and latency results. README.md explains
what each script does.

## Check the reported statistics (no API keys, no cost)

Open a terminal in the top folder of this archive (the folder that contains `src/` and
`results/`) and run:

```
pip install pandas numpy scipy openpyxl
python src/15_reproduce_article_statistics.py
```

The script reads `results/` and `benchmark_datasets/` and prints, section by section,
every statistic reported in the article except the token totals and charges in Section
4.5, which were read from the provider dashboards. It takes a few seconds. Run it from the
top folder, not from inside `src/`, because the file paths are relative to the top folder.

`src/14_verify_statistics.py` is an earlier, partial check kept for reference. It prints
`nan` for the faithfulness tests because one System A faithfulness score is missing.

## Rerun the pipeline

- Python 3.9 was used.
- `requirements.txt` lists the packages for the tutor pipeline (both systems and the
  benchmark tools).
- RAGAS scoring (`src/11_run_ragas_evaluation.py`) and the keep-top quality ablation
  (`src/08_tune_keep_top_quality.py`) also need `ragas` (version 0.4.3 was used),
  `langchain-openai`, `langchain-community`, `datasets`, `pandas` and `openpyxl`.
- You need your own API access: an OpenAI-compatible endpoint for the tutor and evaluator
  models (Groq and OpenRouter were used), a Pinecone index, and an OpenAI key for the
  RAGAS judge (gpt-4o-mini). Copy `.env.example` to `.env` and fill in your keys.
- Hosted models change over time, so a rerun will not reproduce the outputs word for
  word. The files in `results/` are the outputs the article reports.

## How this was made

- The code was written by the author with help from AI coding assistants (Claude and
  ChatGPT), as Section 3.7 of the article states.
- The 150 benchmark items were written by a model (gpt-oss-120b). The author then checked
  every item with an AI assistant (Claude), which ran the code in each item that contained
  code and flagged answers that did not match the actual output; the author made each
  correction by hand and re-ran a sample of items in a Python interpreter. Corrections are
  recorded in the `Validation_Notes` column of the benchmark matrix. The items were not
  checked by independent experts, so some errors may remain.
- `src/04_validate_benchmark.py` is a screening tool that flags items for review, not a
  pass or fail test. Run with Python 3.13, it flags 41 of the 117 items that contain code,
  nearly all for expected reasons: blocks that show output rather than code, the
  tutorial's example modules and files (such as `fibo` and `workfile`), fragments that use
  variables defined in the question, and errors that items demonstrate on purpose. Its
  wording check also flags the 55 questions that refer to "the context" (see Section 5.4
  of the article).
- If you find an error, please report it to the author (momerf988@gmail.com). Corrections
  will be released as a new version of this record.

## Folders

- `src/`: code (see README.md)
- `benchmark_datasets/`: the benchmark specification matrix (150 items with Bloom level,
  difficulty, retrieval level, question type, reference answers and audit notes). The
  items used in the article are in the `Benchmark Spec Matrix` sheet. The
  `Topic Allocation (Locked)` sheet is left over from an earlier 108-item version of the
  benchmark and is not used by any script.
- `results/`: system outputs, RAGAS scorecards, ablation and latency results
- `pdf_files/`: the corpus, seven chapters of the official Python tutorial, as indexed by
  the tutor. Topic 6 (Errors and Exceptions) is saved as `topic_7_errors_exceptions.pdf`.
  This is a naming slip made before the experiment, kept so that the archive matches the
  run exactly. It had no effect: chunk IDs use the full file name, and the `Source_PDF`
  column of the benchmark matrix maps T6 to `topic_7_errors_exceptions.pdf` and T7 to
  `topic_7_oop.pdf`, so the two topics were never mixed.
- `Syllabus/`: the same seven PDFs under their original chapter titles

## Licences

- Code and data: MIT License (see LICENSE), copyright 2026 Muhammad Omer Farooq, except
  for the Python tutorial text below.
- The PDFs in `pdf_files/` and `Syllabus/` are chapters of the Python tutorial
  (https://docs.python.org/3/tutorial/), copyright 2001-2026 Python Software Foundation,
  and are redistributed under the Python Software Foundation License Version 2
  (https://docs.python.org/3/license.html). The same applies to the tutorial passages
  quoted in `results/` and `benchmark_datasets/`. This text is not covered by the MIT
  License.
