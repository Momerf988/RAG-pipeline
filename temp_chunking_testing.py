from document_processor import PDFDocumentProcessor
from app_init import pdf_dir, embedder_model, db_index
import os

processor = PDFDocumentProcessor(pdf_dir, embedder_model, db_index)

def is_word_char(c):
    return c.isalnum()

def find_boundary_breaks(raw_text, chunk_size=700, overlap=100):
    """Mirrors document_processor.chunking()'s exact logic, but tracks
    start/end positions so we can check every boundary against raw_text,
    not just eyeball a few printed chunks."""
    text = raw_text.replace("(cid:415)", "ti")
    text_length = len(text)
    start = 0
    breaks = []
    chunk_count = 0

    while start < text_length:
        end = start + chunk_size
        if end < text_length:
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start:
                end = newline_pos
            else:
                sentence_pos = text.rfind(". ", start, end)
                if sentence_pos > start:
                    end = sentence_pos + 1
                else:
                    space_pos = text.rfind(" ", start, end)
                    if space_pos > start:
                        end = space_pos

        chunk_count += 1

        # check START boundary: is the char right before `start` a word char
        # AND the char at `start` also a word char? if so, we split a word.
        if start > 0:
            before = text[start - 1]
            at_start = text[start] if start < text_length else ""
            if is_word_char(before) and is_word_char(at_start):
                breaks.append(("START", chunk_count, text[max(0, start-15):start+15]))

        # check END boundary: is the last char of the chunk a word char
        # AND the char right after `end` also a word char?
        if end < text_length:
            last_char = text[end - 1] if end > 0 else ""
            after = text[end]
            if is_word_char(last_char) and is_word_char(after):
                breaks.append(("END", chunk_count, text[max(0, end-15):end+15]))

        next_start = end - overlap if end - overlap > start else end
        if next_start < text_length:
            space_pos = text.find(" ", next_start)
            newline_pos = text.find("\n", next_start)
            candidates = [p for p in (space_pos, newline_pos) if p != -1]
            if candidates:
                boundary = min(candidates)
                if boundary - next_start < 50:
                    next_start = boundary + 1
        start = next_start

    return chunk_count, breaks


pdfs = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])
total_chunks_all = 0
total_breaks_all = 0

for pdf_file in pdfs:
    text = processor.read_pdf(os.path.join(pdf_dir, pdf_file))
    chunk_count, breaks = find_boundary_breaks(text)
    total_chunks_all += chunk_count
    total_breaks_all += len(breaks)
    status = "CLEAN" if not breaks else f"{len(breaks)} BREAKS FOUND"
    print(f"{pdf_file}: {chunk_count} chunks — {status}")
    for kind, idx, snippet in breaks[:5]:  # show up to 5 examples per file
        print(f"    [{kind} chunk#{idx}] ...{snippet!r}...")

print(f"\n{'='*50}")
print(f"TOTAL: {total_chunks_all} chunks across {len(pdfs)} PDFs, {total_breaks_all} word-boundary breaks found")