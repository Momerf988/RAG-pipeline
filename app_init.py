import config
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

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