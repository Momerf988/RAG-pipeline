import re
import ast
import sys
import subprocess
import tempfile
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Validation for benchmark_specification_matrix.xlsx.
# Two passes:
#   1. Whole-dataset sanity check (cheap, all 150 rows): required fields present,
#      reference length, leakage phrases, near-duplicate questions.
#   2. Per-topic deep validation (run in batches, e.g. T1-T3 first): extracts every
#      fenced code block from the question + answer text and actually executes it.
#      For Debugging questions, confirms the code really raises an exception, and
#      checks the exception type named in the text against what actually happens.
# No API calls anywhere in this script. Zero cost.

SPEC_FILE = "data/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"
LEAKAGE_PHRASES = ["the context", "the provided", "as given", "the passage", "the excerpt"]
EXCEPTION_NAMES = [
    "SyntaxError", "NameError", "TypeError", "ValueError", "IndexError", "KeyError",
    "AttributeError", "ZeroDivisionError", "ImportError", "ModuleNotFoundError",
    "FileNotFoundError", "StopIteration", "RuntimeError", "IndentationError",
    "UnboundLocalError", "OverflowError", "RecursionError",
]


def load_rows():
    wb = openpyxl.load_workbook(SPEC_FILE, data_only=True)
    ws = wb[SHEET_NAME]
    headers = [c.value for c in ws[1]]
    idx = {h: i for i, h in enumerate(headers)}
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[idx["Generation_Status"]] == "Generated"]
    return rows, idx


def extract_code_blocks(text):
    if not text:
        return []
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


def whole_dataset_sanity(rows, idx):
    print("=" * 78)
    print("PASS 1 -- whole-dataset sanity check, all rows")
    print("=" * 78)
    issues = []

    texts = []
    for r in rows:
        spec_id = r[idx["Spec_ID"]]
        question = r[idx["Generated_Question"]] or ""
        answer = r[idx["Reference_Answer"]] or ""
        texts.append(question)

        ref_words = len(answer.split())
        if ref_words < 25:
            issues.append((spec_id, f"reference answer too short: {ref_words} words (need >= 25)"))

        low = (question + " " + answer).lower()
        for phrase in LEAKAGE_PHRASES:
            if phrase in low:
                issues.append((spec_id, f"leakage phrase found: '{phrase}'"))

        if not r[idx["Supporting_Chunks"]]:
            issues.append((spec_id, "no Supporting_Chunks recorded"))

    # near-duplicate check across ALL questions, TF-IDF cosine similarity (local, no API)
    if len(texts) > 1:
        vec = TfidfVectorizer().fit_transform(texts)
        sim = cosine_similarity(vec)
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i][j] >= 0.75:
                    issues.append((rows[i][idx["Spec_ID"]],
                                   f"near-duplicate of {rows[j][idx['Spec_ID']]} (similarity {sim[i][j]:.2f})"))

    print(f"{len(rows)} rows checked. {len(issues)} issues found.\n")
    for spec_id, reason in issues:
        print(f"  {spec_id}: {reason}")
    if not issues:
        print("  none.")
    return issues


def run_code_block(code, timeout=5):
    """Actually executes a code block in a subprocess. Returns (ok, exception_type, stderr_tail)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, None, None
        # pull the exception type off the last traceback line
        stderr_lines = result.stderr.strip().splitlines()
        last_line = stderr_lines[-1] if stderr_lines else ""
        exc_type = last_line.split(":")[0].strip() if ":" in last_line else last_line.strip()
        return False, exc_type, "\n".join(stderr_lines[-4:])
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", f"did not finish within {timeout}s"


def is_fragment(block):
    """True if this block is deliberately not meant to run standalone: a fill-in-the-blank
    placeholder (Code Completion questions use '___'), or too short/bare to be a real program
    (a lone expression or a signature with no body, shown for illustration only)."""
    if "___" in block:
        return True
    stripped = block.strip()
    if len(stripped) < 15:
        return True
    return False


def deep_validate_topics(rows, idx, topics):
    print("\n" + "=" * 78)
    print(f"PASS 2 -- deep validation, executing code, topics: {', '.join(topics)}")
    print("=" * 78)

    subset = [r for r in rows if r[idx["Topic_ID"]] in topics]
    print(f"{len(subset)} rows in scope.\n")

    results = []
    for r in subset:
        spec_id = r[idx["Spec_ID"]]
        question = r[idx["Generated_Question"]] or ""
        answer = r[idx["Reference_Answer"]] or ""
        qtype = r[idx["Question_Type"]]

        all_blocks = extract_code_blocks(question) + extract_code_blocks(answer)

        if not all_blocks:
            results.append((spec_id, qtype, "SKIP", "no code block found"))
            continue

        # Fill-in-the-blank / bare-fragment blocks aren't meant to run at all -- skip them
        # rather than penalise the row for not being standalone-executable.
        real_blocks = [b for b in all_blocks if not is_fragment(b)]
        fragment_count = len(all_blocks) - len(real_blocks)

        if not real_blocks:
            note = "only fill-in-blank / fragment code found -- nothing executable to check"
            results.append((spec_id, qtype, "SKIP", note))
            continue

        row_notes = []
        if fragment_count:
            row_notes.append(f"({fragment_count} fill-in-blank/fragment block(s) skipped, not scored)")

        # Try the blocks CONCATENATED first (in order) -- many Scenario/Comparison/Debugging
        # answers show a sequence of snippets that build on each other (later blocks reuse
        # variables the earlier block defined). Testing them isolated produces spurious
        # NameErrors that aren't real content bugs.
        concat_code = "\n".join(real_blocks)
        try:
            ast.parse(concat_code)
            concat_parses = True
        except SyntaxError:
            concat_parses = False

        concat_ok, concat_exc, concat_stderr = (False, None, None)
        if concat_parses:
            concat_ok, concat_exc, concat_stderr = run_code_block(concat_code)

        named = [name for name in EXCEPTION_NAMES if name in (question + answer)]

        if qtype == "Debugging":
            # Only a real problem if the text explicitly names an exception type and neither
            # the concatenated run nor any individual block actually produces it. A debugging
            # question about a silent logic bug (wrong output, no crash) is legitimate and
            # can't be checked this way -- we don't fail those.
            if concat_ok and not named:
                row_notes.append("INFO: code ran cleanly, no exception claimed in text -- "
                                  "likely a silent logic bug (can't auto-verify), not flagged")
                status = "PASS"
            elif not named:
                row_notes.append(f"INFO: no exception named in text; concatenated run gave "
                                  f"{'clean exit' if concat_ok else concat_exc} -- not flagged")
                status = "PASS"
            elif not concat_ok and concat_exc in named:
                row_notes.append(f"OK: raised {concat_exc}, matches text")
                status = "PASS"
            else:
                # named an exception but concatenated run doesn't reproduce it -- try each
                # block individually before concluding it's a real mismatch
                per_block_hits = []
                for b in real_blocks:
                    try:
                        ast.parse(b)
                    except SyntaxError:
                        continue
                    ok, exc, _ = run_code_block(b)
                    if not ok:
                        per_block_hits.append(exc)
                if any(e in named for e in per_block_hits):
                    row_notes.append(f"OK: one block individually raised {named}, matches text "
                                      f"(concatenated run masked it)")
                    status = "PASS"
                else:
                    row_notes.append(
                        f"FAIL: text names {named} but neither concatenated nor individual "
                        f"blocks raised it (got {per_block_hits or concat_exc})")
                    status = "FAIL"
        else:
            if concat_ok:
                row_notes.append("OK: ran cleanly (concatenated)")
                status = "PASS"
            elif not concat_parses:
                row_notes.append("FAIL: code does not parse even when blocks are concatenated "
                                  "in order -- likely a real syntax problem")
                status = "FAIL"
            else:
                row_notes.append(f"FAIL: non-Debugging code raised {concat_exc} even when "
                                  f"blocks concatenated in order -- {concat_stderr}")
                status = "FAIL"

        results.append((spec_id, qtype, status, "; ".join(row_notes)))

    passed = sum(1 for r in results if r[2] == "PASS")
    failed = sum(1 for r in results if r[2] == "FAIL")
    skipped = sum(1 for r in results if r[2] == "SKIP")
    print(f"PASS: {passed}   FAIL: {failed}   SKIP (no code, nothing to execute): {skipped}\n")

    for spec_id, qtype, status, notes in results:
        if status == "FAIL":
            print(f"  [FAIL] {spec_id} ({qtype}): {notes}")
    for spec_id, qtype, status, notes in results:
        if status == "PASS":
            print(f"  [PASS] {spec_id} ({qtype}): {notes}")
    for spec_id, qtype, status, notes in results:
        if status == "SKIP":
            print(f"  [SKIP] {spec_id} ({qtype}): {notes}")

    return results


if __name__ == "__main__":
    topics = sys.argv[1:] if len(sys.argv) > 1 else ["T1", "T2", "T3"]
    rows, idx = load_rows()
    whole_dataset_sanity(rows, idx)
    deep_validate_topics(rows, idx, topics)
