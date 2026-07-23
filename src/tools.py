"""Tool implementations, shared by the in-process agent loop and the MCP server.

Hard rule from Block 1: every tool returns a STRING and never raises. A tool
that raises kills the server process, which disconnects the agent.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

try:
    from . import config
    from .retrieval import format_context, hybrid_retrieve
except ImportError:
    import config  # type: ignore
    from retrieval import format_context, hybrid_retrieve  # type: ignore


# --------------------------------------------------------------- memory I/O
def _load_memory() -> list[dict]:
    if not config.MEMORY_FILE.exists():
        return []
    try:
        return json.loads(config.MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_memory(items: list[dict]) -> None:
    config.MEMORY_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=1),
                                  encoding="utf-8")


# ------------------------------------------------------------------- tool 1
def search_displacement_corpus(query: str, top_k: int | None = None) -> str:
    """Search the indexed climate-displacement document corpus (IDMC, World Bank
    Groundswell, IPCC AR6, UNHCR, regional case studies) using hybrid retrieval
    (BM25 + dense + RRF) followed by cross-encoder reranking.

    Use when: the question concerns displacement figures, projections, legal
    status, regional case studies, adaptation or response finance. This is the
    PRIMARY tool — always try it before web_search.
    Do NOT use for: arithmetic, events after the corpus cut-off, or anything
    already returned by recall_memory in this session.
    Returns: up to top_k numbered passages [S1]..[Sn], each with its source
    filename, ordered by cross-encoder relevance.
    Example: query="annual number of people displaced by weather-related disasters"
    """
    try:
        # Cap at config.TOP_K: the model may request more, but the
        # provider's per-minute token limit is the real constraint.
        passages = hybrid_retrieve(query, top_k=min(top_k or config.TOP_K, config.TOP_K))
        if not passages:
            return "No matching passages in the corpus. Consider web_search."
        return format_context(passages)
    except Exception as e:
        return f"ERROR in search_displacement_corpus: {type(e).__name__}: {e}"


# ------------------------------------------------------------------- tool 2
def recall_memory(query: str, source: str = "", limit: int = 5) -> str:
    """Retrieve findings already verified and stored earlier in this session or
    a previous one.

    Use FIRST, before search_displacement_corpus or web_search — it avoids
    redundant retrieval and repeated API cost.
    Do NOT use for: storing anything (use store_finding), or as a substitute for
    the corpus when nothing has been stored yet.
    Returns: stored findings with their source organisation, url and timestamp,
    or an explicit message telling you to search instead.
    Example: query="Tuvalu mobility quota", source="Falepili Union"
    """
    try:
        items = _load_memory()
        terms = {t for t in query.lower().split() if len(t) > 3}
        scored = []
        for it in items:
            if source and source.lower() not in it.get("source", "").lower():
                continue
            blob = f"{it.get('finding','')} {it.get('topic','')} {it.get('source','')}".lower()
            overlap = sum(1 for t in terms if t in blob)
            if overlap:
                scored.append((overlap, it))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return ("No relevant memories found. Use search_displacement_corpus "
                    "to retrieve from the document corpus.")
        return "\n\n".join(
            f"[MEM {i}] source={it.get('source','?')} url={it.get('url','n/a')} "
            f"topic={it.get('topic','?')} stored={it.get('timestamp','?')}\n{it['finding']}"
            for i, (_s, it) in enumerate(scored[:limit], 1))
    except Exception as e:
        return f"ERROR in recall_memory: {type(e).__name__}: {e}"


# ------------------------------------------------------------------- tool 3
def store_finding(finding: str, source: str, url: str = "", topic: str = "") -> str:
    """Persist a single verified finding so later runs can recall it.

    Use after a retrieval step has produced a fact you have actually seen in the
    returned context.
    Do NOT store: speculation, your own inference, or anything not traceable to a
    named source. One finding per call.
    Returns: a storage confirmation with the assigned memory id.
    Example: finding="Up to 216 million internal climate migrants by 2050",
             source="World Bank Groundswell Part 2", url="https://...", topic="projections"
    """
    try:
        if not finding.strip():
            return "ERROR: empty finding, nothing stored."
        if not source.strip():
            return "ERROR: a finding must carry a named source. Nothing stored."
        items = _load_memory()
        entry = {"id": f"M{len(items)+1:04d}", "finding": finding.strip(),
                 "source": source.strip(), "url": url.strip(), "topic": topic.strip(),
                 "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        items.append(entry)
        _save_memory(items)
        return f"Stored as {entry['id']} (source={entry['source']}, at {entry['timestamp']})."
    except Exception as e:
        return f"ERROR in store_finding: {type(e).__name__}: {e}"


# ------------------------------------------------------------------- tool 4
def web_search(query: str, max_results: int = 3) -> str:
    """Search the live web for facts more recent than the local corpus.

    Use when: the question needs data published after the corpus was built, or
    the corpus search returned nothing relevant.
    Do NOT use for: anything search_displacement_corpus or recall_memory already
    answered, arithmetic, or opinion.
    Returns: title, url and summary per result, or an explanatory message if the
    provider is not configured.
    Example: query="IDMC global report internal displacement 2026"
    """
    if not config.TAVILY_API_KEY:
        return ("web_search is not configured (no TAVILY_API_KEY). "
                "Use search_displacement_corpus instead and state in your "
                "CONFIDENCE line that live data was unavailable.")
    try:
        import requests
        r = requests.post("https://api.tavily.com/search",
                          json={"api_key": config.TAVILY_API_KEY, "query": query,
                                "max_results": max_results},
                          timeout=20)
        if r.status_code != 200:
            return f"ERROR: web_search HTTP {r.status_code}. Fall back to the corpus."
        data = r.json().get("results", [])
        if not data:
            return "No web results found for that query. Try different terms."
        return "\n\n".join(
            f"[W{i}] {d.get('title','')}\n{d.get('url','')}\n{d.get('content','')[:600]}"
            for i, d in enumerate(data, 1))
    except Exception as e:
        return f"ERROR in web_search: {type(e).__name__}: {e}. Fall back to the corpus."


# ------------------------------------------------------- registry + schemas
REGISTRY = {
    "recall_memory": recall_memory,
    "search_displacement_corpus": search_displacement_corpus,
    "store_finding": store_finding,
    "web_search": web_search,
}


def openai_schemas() -> list[dict]:
    """Tool schemas for Groq function calling. The description IS the docstring:
    the model chooses tools by reading it, so it is written for the model."""
    def spec(fn, props, required):
        return {"type": "function", "function": {
            "name": fn.__name__, "description": (fn.__doc__ or "").strip(),
            "parameters": {"type": "object", "properties": props, "required": required}}}

    s = {"type": "string"}
    i = {"type": "integer"}
    return [
        spec(recall_memory,
             {"query": s, "source": s, "limit": i}, ["query"]),
        spec(search_displacement_corpus,
             {"query": s, "top_k": i}, ["query"]),
        spec(store_finding,
             {"finding": s, "source": s, "url": s, "topic": s}, ["finding", "source"]),
        spec(web_search, {"query": s, "max_results": i}, ["query"]),
    ]


def call_tool(name: str, args: dict) -> tuple[str, float]:
    """Execute a tool by name. Returns (result_string, latency_seconds)."""
    t0 = time.time()
    fn = REGISTRY.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'.", 0.0
    try:
        out = fn(**args)
    except TypeError as e:
        out = f"ERROR: bad arguments for {name}: {e}"
    except Exception as e:
        out = f"ERROR in {name}: {type(e).__name__}: {e}"
    return str(out), round(time.time() - t0, 3)
