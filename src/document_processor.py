import os
import pdfplumber


class PDFDocumentProcessor:
    """Reads the PDF corpus, chunks it, embeds it, and upserts it into Pinecone."""

    def __init__(self, pdf_dir, embedder_model, db_index):
        self.pdf_dir = pdf_dir
        self.embedder = embedder_model
        self.index = db_index

    def read_pdf(self, file_path):
        raw_text = ""
        with pdfplumber.open(file_path) as pdf_file:
            for page in pdf_file.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    raw_text += extracted_text + "\n"
        return raw_text

    def chunking(self, raw_text, chunk_size=700, overlap=100):
        # Snap chunk boundaries to a newline, then a sentence end, then whitespace -- in
        # that order of preference -- so a chunk almost never ends mid-word or mid-code-line.
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
                        end = sentence_pos + 1  # keep the period, break after the space
                    else:
                        space_pos = raw_text.rfind(" ", start, end)
                        if space_pos > start:
                            end = space_pos
            chunk_group = raw_text[start:end].strip("\n").strip()
            if chunk_group:
                chunks.append(chunk_group)

            next_start = end - overlap if end - overlap > start else end
            # also snap the START of the next chunk forward to a word boundary, so the
            # overlap doesn't reintroduce a mid-word cut at the front of the next chunk
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

    def embeddings(self, chunks):
        return self.embedder.encode(chunks)

    def upsert_to_db(self):
        # Rebuild the corpus and check the chunk count BEFORE touching the index at all --
        # if this doesn't match, something upstream changed (a PDF, the chunking params) and
        # the index should not be wiped and reloaded with the wrong data.
        print("Verifying chunk count before touching the index...")
        total_chunks = 0
        for file in os.listdir(self.pdf_dir):
            if file.endswith(".pdf"):
                raw_text = self.read_pdf(os.path.join(self.pdf_dir, file))
                total_chunks += len(self.chunking(raw_text, chunk_size=700, overlap=100))
        if total_chunks != 317:
            raise AssertionError(
                f"HARD ABORT: corpus rebuilds to {total_chunks} chunks, expected exactly 317. "
                f"Do not proceed -- this would silently index the wrong data."
            )
        print(f"Chunk count OK ({total_chunks}).")

        print("Wiping old Pinecone database...")
        self.index.delete(delete_all=True)

        print("Starting fast, batched upload...")
        for file in os.listdir(self.pdf_dir):
            if file.endswith(".pdf"):
                print(f"Processing {file}...")
                full_path = os.path.join(self.pdf_dir, file)
                raw_text = self.read_pdf(full_path)
                chunks = self.chunking(raw_text, chunk_size=700, overlap=100)
                embeddings = self.embeddings(chunks)

                vectors_to_upsert = []
                for i, embedding in enumerate(embeddings):
                    vector_id = f"{file}_{i}"
                    vector_math = embedding.tolist()
                    metadata = {"line": chunks[i]}
                    vectors_to_upsert.append((vector_id, vector_math, metadata))

                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i: i + batch_size]
                    self.index.upsert(batch)

                print(f"Uploaded {len(vectors_to_upsert)} chunks for {file}")

        stats = self.index.describe_index_stats()
        assert stats.total_vector_count == 317, \
            f"HARD ABORT: index reports {stats.total_vector_count} vectors, expected 317"
        print("Database upsert complete.")


# Zero-cost sanity check -- rebuilds the chunk count with no API calls and no Pinecone
# writes, so the corpus can be verified before running anything that costs money.
if __name__ == "__main__":
    processor = PDFDocumentProcessor(pdf_dir="pdf_files", embedder_model=None, db_index=None)
    total_chunks = 0
    for file in os.listdir(processor.pdf_dir):
        if file.endswith(".pdf"):
            raw_text = processor.read_pdf(os.path.join(processor.pdf_dir, file))
            n = len(processor.chunking(raw_text, chunk_size=700, overlap=100))
            total_chunks += n
            print(f"{file}: {n} chunks")
    print(f"TOTAL: {total_chunks} (expected 317)")
    assert total_chunks == 317, f"HARD ABORT: got {total_chunks}, expected 317"
    print("PASS")
