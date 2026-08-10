# V2 UPDATED (10-08-2026)
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

# --- v2 additions below this line. Nothing above this line was touched. ---

# HANDOFF limitation: "Tutor T = 0.1, noise sources conflated" -- v2 sets T = 0 for the
# tutor (REBUILD_SPEC section 2: "T = 0.0 -- Removes generator sampling as a noise source").
TEMP_TUTOR = 0.0

# HANDOFF limitation: "No max_tokens; 6 items lost" -- V1 never set this, so long answers
# were silently truncated with no error. v2 sets it explicitly.
TUTOR_MAX_TOKENS = 1024

# Rule 2 (HANDOFF + your instructions): "Add a hard abort on any count != 317."
EXPECTED_CHUNKS = 317
