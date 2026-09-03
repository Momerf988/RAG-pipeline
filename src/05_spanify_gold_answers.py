import os
import json
import openpyxl
from document_processor import PDFDocumentProcessor

# Converts each benchmark question's "correct answer location" from chunk IDs into plain
# character positions in the raw PDF text.
#
# Chunk IDs like "topic_1_basics.pdf_9, topic_1_basics.pdf_10" only make sense under the
# current 700/100 chunking -- if chunking is ever changed later, those IDs would point at
# the wrong text and every answer key would silently become wrong. Character spans don't
# have that problem: they stay correct no matter how the text gets split later.
#
# No API calls. No cost. Reads the benchmark spreadsheet and PDFs, writes
# benchmark_datasets/gold_spans.json.

SPEC_FILE = "benchmark_datasets/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"
PDF_DIR = "pdf_files"
OUTPUT_FILE = "benchmark_datasets/gold_spans.json"


def build_chunk_offsets(raw_text, chunks):
    """For each chunk, find exactly where it sits in the raw text. Returns a list of
    (start, end) character positions, one per chunk, in order."""
    offsets = []
    cursor = 0
    for chunk_text in chunks:
        start = raw_text.index(chunk_text, cursor)
        end = start + len(chunk_text)
        offsets.append((start, end))
        cursor = start + 1
    return offsets


def merge_adjacent_spans(spans):
    """Supporting chunks for one question are often consecutive, and consecutive chunks
    overlap by 100 characters (the chunking overlap), so merge any spans that touch or
    overlap into one span."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def main():
    print("Loading benchmark spreadsheet (read-only)...")
    workbook = openpyxl.load_workbook(SPEC_FILE)
    sheet = workbook[SHEET_NAME]
    rows = list(sheet.iter_rows(min_row=2, max_row=sheet.max_row, values_only=True))
    generated_rows = [r for r in rows if r[9] == "Generated"]  # col 10 = Generation_Status
    print(f"{len(generated_rows)} generated questions found (expect 150).")

    print("\nRebuilding corpus and locating every chunk's character position...")
    processor = PDFDocumentProcessor(pdf_dir=PDF_DIR, embedder_model=None, db_index=None)
    raw_by_file = {}
    offsets_by_chunk_id = {}

    pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    total_chunks = 0
    for pdf_file in pdf_files:
        raw_text = processor.read_pdf(os.path.join(PDF_DIR, pdf_file))
        chunks = processor.chunking(raw_text, chunk_size=700, overlap=100)
        raw_by_file[pdf_file] = raw_text
        offsets = build_chunk_offsets(raw_text, chunks)
        for i, (start, end) in enumerate(offsets):
            offsets_by_chunk_id[f"{pdf_file}_{i}"] = (start, end)
        total_chunks += len(chunks)
        print(f"  {pdf_file}: {len(chunks)} chunks located")

    assert total_chunks == 317, f"HARD ABORT: rebuilt {total_chunks} chunks, expected 317"
    print(f"Total: {total_chunks} chunks (expected 317) -- OK")

    print("\nConverting each question's supporting chunks to character spans...")
    gold_spans = {}
    failures = []

    for row in generated_rows:
        spec_id = row[0]                  # col 1
        source_pdf = row[3]               # col 4
        supporting_chunks_raw = row[12]   # col 13 = Supporting_Chunks

        if not supporting_chunks_raw:
            failures.append((spec_id, "no Supporting_Chunks value"))
            continue

        chunk_ids = [c.strip() for c in str(supporting_chunks_raw).split(",") if c.strip()]
        spans = []
        missing = []
        for chunk_id in chunk_ids:
            if chunk_id not in offsets_by_chunk_id:
                missing.append(chunk_id)
                continue
            spans.append(offsets_by_chunk_id[chunk_id])

        if missing:
            failures.append((spec_id, f"chunk IDs not found: {missing}"))
            continue

        merged = merge_adjacent_spans(spans)

        # Sanity check: every merged span must be non-empty text that really exists in the
        # raw source file.
        raw_text = raw_by_file[source_pdf]
        for start, end in merged:
            span_text = raw_text[start:end]
            if not span_text.strip():
                failures.append((spec_id, f"empty span at {start}-{end}"))
                break
        else:
            gold_spans[spec_id] = {
                "source": source_pdf,
                "spans": [{"start": s, "end": e} for s, e in merged],
                "chunk_ids": chunk_ids,
            }

    print(f"\nResolved: {len(gold_spans)} / {len(generated_rows)} questions")
    if failures:
        print(f"FAILED: {len(failures)} questions could not be converted:")
        for spec_id, reason in failures:
            print(f"  {spec_id}: {reason}")
    else:
        print("No failures.")

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(gold_spans, f, indent=2)
    print(f"\nWritten to {OUTPUT_FILE}")

    if len(gold_spans) != 150:
        print(f"\nWARNING: expected 150 resolved questions, got {len(gold_spans)}. "
              f"Do not treat this as done until every failure above is understood.")
    else:
        print("\nAll 150 questions converted successfully.")


if __name__ == "__main__":
    main()
