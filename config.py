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

# V2 work -- shared constant so System B can detect a "disguised refusal": CRAG's evaluator
# routes to generation (CORRECT or AMBIGUOUS-but-usable), but the tutor's own separate
# critical_rule check in RAG_response_processor.LLM_prompt independently decides the context
# wasn't good enough and returns this exact string anyway. Defined once here so
# system_b_main.py and generate_evaluation_dataset_B.py can't drift out of sync with the
# literal string in RAG_response_processor.py.
REFUSAL_STRING = "Sorry, there are no relevant documents in the Database to answer your query. :("

# V2 work -- separate CRAG evaluator model, distinct from the tutor's LLM_MODEL_NAME
# (llama-3.1-8b-instant, unchanged). Rationale: the original CRAG paper's evaluator is a
# separately fine-tuned, lightweight T5-Large (770M params) classifier -- NOT the same model
# used for generation, and NOT a general-purpose prompted chat LLM. We can't replicate that
# exactly: the paper's own fine-tuned checkpoint is closed-weight and was never released
# (confirmed via an independent 2026 open-reproduction paper that hit the same wall --
# arXiv:2603.16169, "the original implementation relies on proprietary components including
# ... closed model weights, limiting reproducibility"), and training our own from scratch is
# out of MSc scope regardless of time. This is the standard adaptation used across CRAG
# reproductions: a general LLM prompted as a classifier, matching what we'd already built.
#
# gpt-oss-120b was considered and rejected -- at 120B params it is ~15x larger than the 8B
# tutor it's meant to sanity-check, the inverse of the paper's "lightweight" evaluator intent.
# Checked Groq's live model catalog (console.groq.com/docs/models) for the smallest available
# general-purpose option: below 20B, the only smaller models on Groq are llama-prompt-guard-2
# (22M/86M) -- specialized prompt-injection classifiers, wrong tool for relevance judgment.
# openai/gpt-oss-20b is the practical floor: 20B (vs 120B), different model family from the
# Llama tutor (reduces correlated same-model failure modes), and listed as a Production model
# on Groq (qwen/qwen3.6-27b was the other different-family candidate but is Preview-tier,
# i.e. Groq's own docs say it "may be discontinued at short notice" -- not worth the risk on
# a tight deadline).
EVALUATOR_MODEL_NAME = "openai/gpt-oss-20b"
