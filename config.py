import os
from dotenv import load_dotenv

# Load the environment variables once
load_dotenv()

# Client Keys & URLs
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL")
DB_API_KEY = os.getenv("DB_API_KEY")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Application Settings
DB_INDEX_NAME = os.getenv("DB_INDEX_NAME")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
TRANSFORMER_MODEL = os.getenv("TRANSFORMER_MODEL")
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY")
#RERANKER_MODEL = os.getenv("RERANKER_MODEL")