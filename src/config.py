"""Central configuration. Everything tunable lives here, nothing is hard-coded elsewhere."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # allows tests to run before full install
    def load_dotenv(*a, **k):
        return False

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --- paths ---------------------------------------------------------------
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
INDEX_DIR = DATA_DIR / "index"
MEMORY_FILE = DATA_DIR / "memory.json"
RUNS_FILE = DATA_DIR / "runs.jsonl"
for _d in (DATA_DIR, CORPUS_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- LLM -----------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TOOL_MODEL = os.getenv("TOOL_MODEL", "llama-3.3-70b-versatile")     # fast, T=0, tool selection
SYNTH_MODEL = os.getenv("SYNTH_MODEL", "llama-3.3-70b-versatile")   # final synthesis
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "llama-3.3-70b-versatile")  # critic agent

# USD per 1M tokens (input, output). Used by TokenBudget.
PRICING = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "deepseek-r1-distill-llama-70b": (0.75, 0.99),
    "llama-3.1-8b-instant": (0.05, 0.08),
}
DEFAULT_PRICE = (0.59, 0.79)

# --- retrieval -----------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
PARENT_WORDS = int(os.getenv("PARENT_WORDS", "800"))   # returned to the LLM
CHILD_WORDS = int(os.getenv("CHILD_WORDS", "200"))     # indexed for retrieval
CHILD_OVERLAP = int(os.getenv("CHILD_OVERLAP", "40"))
K_CANDIDATES = 15   # hybrid retrieve this many children
TOP_K = 5           # cross-encoder keeps this many parents
RRF_K = 60          # the 60 in 1/(60+rank)

# --- agent ---------------------------------------------------------------
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))
SELF_CONSISTENCY_K = int(os.getenv("SELF_CONSISTENCY_K", "3"))
SC_TEMPERATURE = 0.8
TOOL_TEMPERATURE = 0.0

# --- budget (Block 2) ----------------------------------------------------
MAX_USD = float(os.getenv("MAX_USD", "2.0"))
WARN_USD = float(os.getenv("WARN_USD", "0.5"))

# --- observability -------------------------------------------------------
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- optional web search -------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- versioning (Block 4: hash the system prompt) ------------------------
AGENT_NAME = "climate-displacement-agent"
AGENT_VERSION = "1.0.0"
