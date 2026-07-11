import os
import pdfplumber

# PHASE 1: Load and embed PDF documents into the vector database
class PDFDocumentProcessor:
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
    
    def chunking(self, raw_text, chunk_size = 3):
        raw_text = raw_text.replace("(cid:415)", "ti")
        raw_lines = [line.strip() for line in raw_text.split("\n") if line.strip()] 
        chunks, chunk_group = [], ""
        for i in range(0, len(raw_lines), chunk_size):
            chunk_group = " ".join(raw_lines[i:i + chunk_size])
            chunks.append(chunk_group)
        return chunks 
    
    def embeddings(self, chunks):
        embeddings = self.embedder.encode(chunks)
        return embeddings
    
    def upsert_to_db(self): 
        for file in os.listdir(self.pdf_dir):
            if file.endswith(".pdf"):
                pdf_file = file
                full_path = os.path.join(self.pdf_dir, pdf_file)
                raw_text = self.read_pdf(full_path)
                chunks = self.chunking(raw_text, chunk_size = 3)
                embeddings = self.embeddings(chunks)
                for i, embedding in enumerate(embeddings):
                    vector_id = f"{pdf_file}_{i}" 
                    vector_math = embedding.tolist()
                    metadata = {"line": chunks[i]} 
                    self.index.upsert([(vector_id, vector_math, metadata)])