import os
from dotenv import load_dotenv

load_dotenv()

# Client keys and URLs
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL")
DB_API_KEY = os.getenv("DB_API_KEY")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Application settings
DB_INDEX_NAME = os.getenv("DB_INDEX_NAME")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
TRANSFORMER_MODEL = os.getenv("TRANSFORMER_MODEL")
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY")

# Generation settings for the tutor. Temperature 0 everywhere in this project so runs are
# comparable across systems and repeatable across sessions -- not a stylistic choice.
TEMP_TUTOR = 0.0
TUTOR_MAX_TOKENS = 2048

# The corpus should always chunk down to exactly this many chunks. Every script that
# rebuilds or re-indexes the corpus hard-aborts if it doesn't match, rather than silently
# indexing something different from what the benchmark was built against.
EXPECTED_CHUNKS = 317

# Shared refusal string. System B checks its own output against this to catch a "disguised
# refusal" -- CRAG's evaluator can approve a context for generation, but the tutor prompt
# still has its own grounding rule and could bail out anyway. Kept as one constant here so
# every script that needs to recognise a refusal (system_b_main.py, the System B batch
# generator) can't drift out of sync with each other.
REFUSAL_STRING = "Sorry, there are no relevant documents in the Database to answer your query. :("

# Retrieval evaluator model for System B, deliberately a different model family from the
# tutor (LLM_MODEL_NAME above). An evaluator judging context for the same model that then
# generates from it shares that model's blind spots by construction, so it needs to be
# something else. The original CRAG paper uses a purpose-trained T5-Large classifier for
# this, but those weights were never released, so this project uses a general-purpose model
# prompted as a classifier instead -- the standard workaround other CRAG reproductions use
# too. Picked the smallest general-purpose (non-specialised) model available on Groq's
# production tier that isn't the same family as the tutor.
EVALUATOR_MODEL_NAME = "openai/gpt-oss-20b"
