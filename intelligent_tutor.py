import os
import config
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import pdfplumber

# PHASE 0: Initialize the application configurations
llm_client = OpenAI(
    api_key = config.LLM_API_KEY,
    base_url = config.LLM_SERVER_URL)

db_client = Pinecone(
    api_key = config.DB_API_KEY)

db_index = db_client.Index(
    config.DB_INDEX_NAME)

embedder_model = SentenceTransformer(config.TRANSFORMER_MODEL)
pdf_dir = config.PDF_DIRECTORY

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
        # NOTE: I have dropped this so develop intuition for this: cleaned_text = raw_text.replace("(cid:415)", "ti")
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

# PHASE 2: User prompt input
class UserQueryProcessor:
    def __init__(self, embedder_model):
        self.embedder = embedder_model

    def get_user_preferences(self):
        user_prompt = input("\nWhat are we here to learn today? :)").strip()
        while not user_prompt:
            print("Please enter a question or topic to learn about.")
            user_prompt = input("\nWhat are we here to learn today?\n> ").strip() #why not worked here?
        detail_level = input("\nHow detailed would you like the answer to be (e.g., brief, elaborate, in-depth)?\n").strip()
        valid_detail_levels = ["brief", "elaborate", "in-depth"]
        if detail_level not in valid_detail_levels:     #Q: why replaced if with while? A: to avoid infinite loop if user keeps entering invalid input
            detail_level = 'brief'
        return user_prompt, detail_level
    
    def vectorize_query(self, user_prompt):
        embedded_user_query = self.embedder.encode(user_prompt)
        return embedded_user_query.tolist()

# PHASE 3: LLM response generation
class LLMResponseProcessor:
    def __init__(self, llm_client, db_index):
        self.llm_client = llm_client
        self.index = db_index

    def search_db(self, embedded_user_query, top_k = 3):
        search_result = self.index.query(
            vector = embedded_user_query, 
            top_k = 3, 
            include_metadata = True)
        matches = search_result['matches']
        if not matches:
            matches = []          # why not ''?
        return search_result['matches']
    
    def stich_context(self, matches):
        context_list = [match['metadata']['line'] for match in matches]
        stiched_context = " ".join(context_list)
        return stiched_context
    
    def LLM_prompt(self, user_prompt, detail_level, context_string):
        system_persona = "You are a highly intelligent, helpful university teaching assistant."
        critical_rule = '''
        CRITICAL RULE: You must base your answer EXCLUSIVELY on the Context provided below. 
        If the Context does not contain the necessary information to answer the question, 
        do NOT guess or make up an analogy. Reply EXACTLY with: 
        "I do not have enough information in the syllabus to answer that.
        '''
        user_instruction = f'''
        Please answer the user's question. Provide the answer at {detail_level} level.
        NOTE: Remember to strictly follow this: {critical_rule}
        Context: {context_string}
        User prompt: {user_prompt} 
        '''
        return system_persona, user_instruction
    
    def generate_response(self, system_persona, user_instruction):
        response = self.llm_client.chat.completions.create(
            model = config.LLM_MODEL_NAME,
            messages = [
                {"role": "system", "content": system_persona},
                {"role": "user", "content": user_instruction}
            ],
            temperature = 0.1
        )
        return response.choices[0].message.content
    
    def respond_to_user(self, embedded_user_query, user_prompt, detail_level):
        db_search_result = self.search_db(embedded_user_query)
        context = self.stich_context(db_search_result)
        if not context:
            return "Sorry, there are no relevant documents in the Database to answer your query :(."
        system_persona, user_instruction = self.LLM_prompt(user_prompt, detail_level, context)
        final_answer = self.generate_response(system_persona, user_instruction)
        return final_answer

#Execution: main application loop

if __name__ == "__main__":
    print("Starting Intelligent Tutor...")
    document_processing_init = PDFDocumentProcessor(pdf_dir, embedder_model, db_index)
    #document_processing_init.upsert_to_db()
    user_query_processing_init = UserQueryProcessor(embedder_model)
    LLM_response_processing_init = LLMResponseProcessor(llm_client, db_index)
    while True:
        user_prompt, detail_level = user_query_processing_init.get_user_preferences()
        if user_prompt.lower() == 'quit':
            break
        embedded_user_query = user_query_processing_init.vectorize_query(user_prompt)
        final_answer = LLM_response_processing_init.respond_to_user(embedded_user_query, user_prompt, detail_level)
        print(final_answer)


    

