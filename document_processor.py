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


#    def chunking(self, raw_text, chunk_size = 3):
#        raw_text = raw_text.replace("(cid:415)", "ti")
#        raw_lines = [line.strip() for line in raw_text.split("\n") if line.strip()] 
#        chunks, chunk_group = [], ""
#        for i in range(0, len(raw_lines), chunk_size):
#            chunk_group = " ".join(raw_lines[i:i + chunk_size])
#            chunks.append(chunk_group)
#        return chunks 
   
    def chunking(self, raw_text, chunk_size = 700, overlap = 100):
        raw_text = raw_text.replace("(cid:415)", "ti")
        chunks = []
        start = 0
        text_length = len(raw_text)
        while start < text_length:
            end = start + chunk_size
            if end < text_length:
                newline_pos = raw_text.rfind("\n", start, end)
                if newline_pos > start:
                    end = newline_pos  # snap to a line break so we don't cut a code line in half
            chunk_group = raw_text[start:end].strip("\n")
            if chunk_group.strip():
                chunks.append(chunk_group)
            start = end - overlap if end - overlap > start else end
        return chunks

    def embeddings(self, chunks):
        embeddings = self.embedder.encode(chunks)
        return embeddings
    
 #   def upsert_to_db(self): 
 #       for file in os.listdir(self.pdf_dir):
 #           if file.endswith(".pdf"):
 #               pdf_file = file
 #               full_path = os.path.join(self.pdf_dir, pdf_file)
 #               raw_text = self.read_pdf(full_path)
 #               chunks = self.chunking(raw_text, chunk_size = 3)
 #               embeddings = self.embeddings(chunks)
 #               for i, embedding in enumerate(embeddings):
 #                   vector_id = f"{pdf_file}_{i}" 
 #                   vector_math = embedding.tolist()
 #                   metadata = {"line": chunks[i]} 
 #                   self.index.upsert([(vector_id, vector_math, metadata)])

    def upsert_to_db(self): 
            print("Wiping old Pinecone database...")
#            self.index.delete(delete_all=True)
            
            print("Starting fast, batched upload...")
            for file in os.listdir(self.pdf_dir):
                if file.endswith(".pdf"):
                    print(f"Processing {file}...")
                    pdf_file = file
                    full_path = os.path.join(self.pdf_dir, pdf_file)
                    raw_text = self.read_pdf(full_path)
                    
                    # Chunk and embed
                    chunks = self.chunking(raw_text, chunk_size=700, overlap=100)
                    embeddings = self.embeddings(chunks)
                    
                    # 1. Gather all vectors for this file into a list
                    vectors_to_upsert = []
                    for i, embedding in enumerate(embeddings):
                        vector_id = f"{pdf_file}_{i}" 
                        vector_math = embedding.tolist()
                        metadata = {"line": chunks[i]} 
                        vectors_to_upsert.append((vector_id, vector_math, metadata))
                    
                    # 2. Upload in batches of 100
                    batch_size = 100
                    for i in range(0, len(vectors_to_upsert), batch_size):
                        batch = vectors_to_upsert[i : i + batch_size]
                        self.index.upsert(batch)
                        
                    print(f"✓ Uploaded {len(vectors_to_upsert)} chunks for {file}")
                    
            print("Database Upsert Complete!")