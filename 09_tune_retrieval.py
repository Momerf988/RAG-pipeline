# NEW FILE -- retrieval keep_top comparison (REBUILD_SPEC Step 09, adapted to your scope).
#
# Purpose: answer "would keeping more chunks after reranking (keep_top) actually retrieve
# more of the evidence a question needs?" -- directly motivated by the list-vs-tuple case,
# where the real definition chunk may simply not have survived the top-7 cut.
#
# Method: for each of the 150 benchmark questions, retrieve the same top-20 candidates
# (unchanged), rerank ALL 20 ONCE (not once per k), then compute span-recall at several
# prefix lengths of that single reranked list: k = 3, 5, 7, 10, 15, 20. This gives a full
# recall-vs-k curve from one pass, not six separate runs.
#
# Uses the embedding model already settled by 02_diag_embeddings.py (MiniLM, kept -- this
# script does not re-open that question). No LLM calls, no cost, no Pinecone writes.
#
# Decision rule, declared before looking at the numbers (same materiality bar used for the
# embedder ablation, applied to this knob instead): raising keep_top above 7 is only worth
# adopting if some k > 7 improves overall recall by >= +0.05 versus the current k = 7.
# Otherwise keep keep_top = 7 -- also a valid, reportable outcome.

import os
import json
import openpyxl
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import config

PDF_DIR = "pdf_files"
SPEC_FILE = "data/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"
GOLD_SPANS_FILE = "data/gold_spans.json"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # unchanged, same as production
TOP_K_CANDIDATES = 60   # pre-rerank pool size, matches RAG_response_processor.search_db
K_VALUES = [3, 5, 7, 10, 15, 20, 60]   # prefixes of the single reranked list
CURRENT_KEEP_TOP = 7    # what's live in RAG_response_processor.rerank right now


# ---- chunking: copied verbatim from document_processor.py. Do not edit. ----
def read_pdf(file_path):
    import pdfplumber
    raw_text = ""
    with pdfplumber.open(file_path) as pdf_file:
        for page in pdf_file.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                raw_text += extracted_text + "\n"
    return raw_text


def chunking(raw_text, chunk_size=700, overlap=100):
    raw_text = raw_text.replace("(cid:415)", "ti")
    chunks = []
    start = 0
    text_length = len(raw_text)
    while start < text_length:
        end = start + chunk_size
        if end < text_length:
            newline_pos = raw_text.rfind("\n", start, end)
            if newline_pos > start:
                end = newline_pos
            else:
                sentence_pos = raw_text.rfind(". ", start, end)
                if sentence_pos > start:
                    end = sentence_pos + 1
                else:
                    space_pos = raw_text.rfind(" ", start, end)
                    if space_pos > start:
                        end = space_pos
        chunk_group = raw_text[start:end].strip("\n").strip()
        if chunk_group:
            chunks.append(chunk_group)
        next_start = end - overlap if end - overlap > start else end
        if next_start < text_length:
            space_pos = raw_text.find(" ", next_start)
            newline_pos = raw_text.find("\n", next_start)
            candidates = [p for p in (space_pos, newline_pos) if p != -1]
            if candidates:
                boundary = min(candidates)
                if boundary - next_start < 50:
                    next_start = boundary + 1
        start = next_start
    return chunks
# ---- end verbatim copy ----


def build_corpus():
    pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    all_chunk_ids, all_chunk_texts = [], []
    offsets_by_chunk_id = {}
    total = 0
    for pdf_file in pdf_files:
        raw_text = read_pdf(os.path.join(PDF_DIR, pdf_file))
        chunks = chunking(raw_text, chunk_size=700, overlap=100)
        cursor = 0
        for i, chunk_text in enumerate(chunks):
            cid = f"{pdf_file}_{i}"
            start = raw_text.index(chunk_text, cursor)
            end = start + len(chunk_text)
            cursor = start + 1
            offsets_by_chunk_id[cid] = (start, end)
            all_chunk_ids.append(cid)
            all_chunk_texts.append(chunk_text)
        total += len(chunks)
    assert total == config.EXPECTED_CHUNKS, \
        f"HARD ABORT: rebuilt {total} chunks, expected {config.EXPECTED_CHUNKS}"
    print(f"Corpus OK: {total} chunks across {len(pdf_files)} files.")
    return all_chunk_ids, all_chunk_texts, offsets_by_chunk_id


def load_benchmark_rows():
    wb = openpyxl.load_workbook(SPEC_FILE, data_only=True)
    ws = wb[SHEET_NAME]
    headers = [c.value for c in ws[1]]
    idx = {h: i for i, h in enumerate(headers)}
    rows = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[idx["Generation_Status"]] != "Generated":
            continue
        rows[r[idx["Spec_ID"]]] = {
            "question": r[idx["Generated_Question"]],
            "r_level": r[idx["Retrieval_Level"]],
            "source_pdf": r[idx["Source_PDF"]],
        }
    return rows


def merge_intervals(intervals):
    if not intervals:
        return []
    s = sorted(intervals)
    merged = [list(s[0])]
    for cur_s, cur_e in s[1:]:
        if cur_s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], cur_e)
        else:
            merged.append([cur_s, cur_e])
    return [tuple(m) for m in merged]


def overlap_len(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def span_recall(gold_spans_list, retrieved_spans_list):
    total_gold_chars = sum(e - s for s, e in gold_spans_list)
    if total_gold_chars == 0:
        return None
    merged_retrieved = merge_intervals(retrieved_spans_list)
    covered = 0
    for g in gold_spans_list:
        for r in merged_retrieved:
            covered += overlap_len(g, r)
    return covered / total_gold_chars


if __name__ == "__main__":
    print("Loading benchmark and gold spans...")
    bench_rows = load_benchmark_rows()
    with open(GOLD_SPANS_FILE) as f:
        gold_spans = json.load(f)
    print(f"{len(bench_rows)} generated rows, {len(gold_spans)} gold spans loaded.")

    chunk_ids, chunk_texts, offsets_by_chunk_id = build_corpus()

    print(f"\nEncoding {len(chunk_texts)} chunks with {config.TRANSFORMER_MODEL} "
          f"(already-settled embedder, not re-tested here)...")
    embedder = SentenceTransformer(config.TRANSFORMER_MODEL)
    chunk_vecs = np.array(embedder.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False))

    print(f"Loading reranker ({RERANKER_MODEL})...")
    reranker = CrossEncoder(RERANKER_MODEL)

    # results[k] = list of (spec_id, r_level, recall)
    results_by_k = {k: [] for k in K_VALUES}

    print(f"\nScoring {len(gold_spans)} questions (one rerank pass each, sliced at every k)...")
    for spec_id, gold in gold_spans.items():
        if spec_id not in bench_rows:
            continue
        question = bench_rows[spec_id]["question"]
        r_level = bench_rows[spec_id]["r_level"]
        source_pdf = gold["source"]
        gold_spans_list = [(s["start"], s["end"]) for s in gold["spans"]]

        q_vec = embedder.encode([question], normalize_embeddings=True)[0]
        sims = chunk_vecs @ q_vec
        top20_idx = np.argsort(-sims)[:TOP_K_CANDIDATES]
        top20_ids = [chunk_ids[i] for i in top20_idx]
        top20_texts = [chunk_texts[i] for i in top20_idx]

        pairs = [(question, t) for t in top20_texts]
        scores = reranker.predict(pairs)
        reranked_ids = [cid for _, cid in sorted(zip(scores, top20_ids), key=lambda x: x[0], reverse=True)]

        for k in K_VALUES:
            top_k_ids = reranked_ids[:k]
            retrieved_spans = [offsets_by_chunk_id[cid] for cid in top_k_ids if cid.startswith(source_pdf)]
            recall = span_recall(gold_spans_list, retrieved_spans)
            if recall is not None:
                results_by_k[k].append((spec_id, r_level, recall))

    print("\n" + "=" * 70)
    print(f"{'k':<6}{'Overall':<10}{'R1':<10}{'R2':<10}{'R3':<10}{'R4':<10}")
    print("-" * 70)
    overall_by_k = {}
    for k in K_VALUES:
        rows = results_by_k[k]
        overall = np.mean([r for _, _, r in rows])
        overall_by_k[k] = overall
        by_level = {}
        for _, lvl, r in rows:
            by_level.setdefault(lvl, []).append(r)
        level_means = {lvl: np.mean(v) for lvl, v in by_level.items()}
        marker = "  <- current" if k == CURRENT_KEEP_TOP else ""
        print(f"{k:<6}{overall:<10.4f}"
              f"{level_means.get('R1', float('nan')):<10.4f}"
              f"{level_means.get('R2', float('nan')):<10.4f}"
              f"{level_means.get('R3', float('nan')):<10.4f}"
              f"{level_means.get('R4', float('nan')):<10.4f}{marker}")
    print("=" * 70)

    current_recall = overall_by_k[CURRENT_KEEP_TOP]
    best_k = max(K_VALUES, key=lambda k: overall_by_k[k])
    best_gain = overall_by_k[best_k] - current_recall

    print(f"\nDECISION (pre-declared rule):")
    print(f"  Current (k={CURRENT_KEEP_TOP}) overall recall: {current_recall:.4f}")
    print(f"  Best k found: k={best_k}, overall recall: {overall_by_k[best_k]:.4f}, gain: {best_gain:+.4f}")
    print(f"  Switch threshold: >= +0.05 overall gain")
    if best_gain >= 0.05 and best_k != CURRENT_KEEP_TOP:
        print(f"  -> RAISE keep_top to {best_k}. Threshold met.")
    else:
        print(f"  -> KEEP keep_top = {CURRENT_KEEP_TOP}. Threshold not met -- valid, reportable outcome.")
