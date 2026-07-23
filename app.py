"""Web interface for the climate displacement agent.

Deliberately a separate entry point. `python src/agent.py` — the command the
grading gate runs — does not import this file, and FastAPI is not in
requirements.txt, so a broken web layer cannot break the CLI.

    pip install -r requirements-web.txt
    uvicorn app:app --port 7860
    open http://localhost:7860

A run takes 30-60 s, so the client does not block on it: POST /api/ask starts a
background job and returns an id, and the client polls /api/job/{id} for stage
events. Polling rather than SSE because SSE is the first thing a proxy buffers,
and this has to work behind whatever Hugging Face Spaces puts in front of it.
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src import config
from src.agent import DEFAULT_QUESTION, run
from src.ingest import index_exists, load_documents

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

app = FastAPI(title="Climate Displacement Research Agent", version=config.AGENT_VERSION)

_pool = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, dict] = {}
_last_success: dict | None = None
_lock = threading.Lock()
MAX_JOBS = 40


class AskRequest(BaseModel):
    question: str


def _set(job_id: str, **fields) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _append_event(job_id: str, stage: str, payload: dict) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["events"].append({"stage": stage, **payload})


def _execute(job_id: str, question: str) -> None:
    global _last_success
    try:
        result = run(question, on_event=lambda s, p: _append_event(job_id, s, p))
        _set(job_id, status="done", result=result)
        if result.get("status") == "ok":
            with _lock:
                _last_success = {**result, "completed_at": time.strftime("%H:%M:%S")}
    except Exception as e:  # never leave a job hanging in "running"
        detail = f"{type(e).__name__}: {e}"
        low = detail.lower()
        if "model" in low and ("not found" in low or "decommission" in low or "404" in low):
            detail += ("  ·  The configured model no longer exists at the provider. "
                       "List current models and update TOOL_MODEL / SYNTH_MODEL / "
                       "CRITIC_MODEL in .env.")
        elif "rate" in low and "limit" in low:
            detail += "  ·  Provider rate limit. Wait a minute and run it again."
        elif "api_key" in low or "authentication" in low:
            detail += "  ·  GROQ_API_KEY is missing or rejected."
        _set(job_id, status="error", error=detail,
             result={"status": "error", "answer": "", "passages": []})


@app.post("/api/ask")
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "Enter a question.")
    if len(question) > 2000:
        raise HTTPException(400, "Question is too long — keep it under 2000 characters.")

    job_id = uuid.uuid4().hex[:12]
    with _lock:
        if len(_jobs) > MAX_JOBS:  # bounded memory; oldest first
            for old in list(_jobs)[: len(_jobs) - MAX_JOBS]:
                _jobs.pop(old, None)
        _jobs[job_id] = {"status": "running", "events": [], "result": None,
                         "question": question}
    _pool.submit(_execute, job_id, question)
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def job(job_id: str):
    with _lock:
        j = _jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "No such run. It may have expired — ask again.")
        return JSONResponse(dict(j))


@app.get("/api/meta")
def meta():
    """Masthead data: what the operator is looking at."""
    try:
        docs = load_documents()
        corpus = {"documents": len(docs),
                  "words": sum(len(d["text"].split()) for d in docs)}
    except Exception:
        corpus = {"documents": 0, "words": 0}
    from src import observability
    from src.reasoning import SYNTHESIS_SYSTEM_PROMPT
    return {
        "name": config.AGENT_NAME,
        "version": config.AGENT_VERSION,
        "prompt_hash": observability.prompt_version(SYNTHESIS_SYSTEM_PROMPT),
        "models": {"planner": config.TOOL_MODEL, "synthesis": config.SYNTH_MODEL,
                   "critic": config.CRITIC_MODEL},
        "self_consistency_k": config.SELF_CONSISTENCY_K,
        "max_usd": config.MAX_USD,
        "corpus": corpus,
        "index_built": index_exists(),
        "llm_configured": bool(config.GROQ_API_KEY),
        "default_question": DEFAULT_QUESTION,
    }


@app.get("/api/last")
def last():
    """The most recent successful run, kept in memory.

    This exists for one reason: if a live run fails in front of an audience —
    provider outage, rate limit, network — there is still something real to show,
    explicitly labelled as an earlier run rather than passed off as live."""
    with _lock:
        if _last_success is None:
            raise HTTPException(404, "No completed run yet in this process.")
        return JSONResponse(_last_success)


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": config.AGENT_VERSION}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.on_event("startup")
def warm() -> None:
    """Build the index at boot rather than inside the first request, so the first
    user does not wait through a model download with no explanation."""
    def _warm():
        try:
            from src.retrieval import get_reranker, hybrid_retrieve
            hybrid_retrieve("climate displacement", top_k=1)
            get_reranker()
            print("[startup] index and rerankers warm")
        except Exception as e:
            print(f"[startup] warm-up failed ({e}); first request will be slower")
    threading.Thread(target=_warm, daemon=True).start()
