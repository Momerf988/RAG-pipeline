import config
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder

# Every script in this project imports from here, and loading the embedder + reranker
# models takes several seconds with no output otherwise -- easy to mistake for a frozen
# terminal on first run, so this prints as it goes.
print("Initializing clients and models (this can take up to ~30s, especially the first time)...")

llm_client = OpenAI(
    api_key=config.LLM_API_KEY,
    base_url=config.LLM_SERVER_URL)
print("  LLM client ready.")

db_client = Pinecone(api_key=config.DB_API_KEY)
db_index = db_client.Index(config.DB_INDEX_NAME)
print("  Pinecone index connected.")

print("  Loading embedder model...")
embedder_model = SentenceTransformer(config.TRANSFORMER_MODEL)
print("  Loading reranker model...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pdf_dir = config.PDF_DIRECTORY
print("Initialization complete.\n")
