"""Block 1 retrieval pipeline.

    hybrid(k=15 children)  ->  cross-encoder rerank  ->  top 5 PARENTS  -> LLM

Why each piece exists:
  * dense  — matches paraphrase ('coastal flooding' ~ 'sea level rise') but
             under-represents numbers and acronyms ('IDMC', '21.5 million').
  * BM25   — exact term matching, so acronyms and figures always hit.
  * RRF    — fuses the two RANKINGS, so no score normalisation is needed:
             score(d) = sum over lists of 1 / (60 + rank(d)).
  * rerank — a bi-encoder embeds query and document separately, with zero
             interaction. A cross-encoder attends over the pair jointly, which
             is what 'relevance of THIS doc for THIS query' actually requires.
             Too slow for the whole corpus, hence step 2 on 15 candidates.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

try:
    from . import config
    from .ingest import build_index, get_embedder, index_exists, tokenise
except ImportError:  # script mode
    import config  # type: ignore
    from ingest import build_index, get_embedder, index_exists, tokenise  # type: ignore

_state: dict[str, Any] = {}
_reranker = None
_RERANK_AVAILABLE = True


def _load():
    if _state:
        return _state
    if not index_exists():
        print("[retrieval] no index found — building it now")
        build_index()
    blob = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    _state["parents"] = {p["id"]: p for p in blob["parents"]}
    _state["children"] = blob["children"]
    _state["vectors"] = np.load(config.INDEX_DIR / "child_vectors.npy")

    from rank_bm25 import BM25Okapi
    _state["bm25"] = BM25Okapi([tokenise(c["text"]) for c in _state["children"]])
    return _state


# --------------------------------------------------------------- retrievers
def dense_search(query: str, k: int) -> list[int]:
    s = _load()
    q = np.asarray(get_embedder().encode(query), dtype=np.float32).ravel()
    q /= max(float(np.linalg.norm(q)), 1e-9)
    scores = s["vectors"] @ q
    return list(np.argsort(-scores)[:k])


def bm25_search(query: str, k: int) -> list[int]:
    s = _load()
    scores = s["bm25"].get_scores(tokenise(query))
    return list(np.argsort(-scores)[:k])


def rrf_fuse(rankings: list[list[int]], k: int, rrf_k: int | None = None) -> list[int]:
    """Reciprocal Rank Fusion. Operates on ranks, not scores."""
    rrf_k = config.RRF_K if rrf_k is None else rrf_k
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:k]]


def get_reranker():
    global _reranker, _RERANK_AVAILABLE
    if _reranker is not None or not _RERANK_AVAILABLE:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(config.RERANK_MODEL)
    except Exception as e:  # pragma: no cover
        print(f"[retrieval] WARNING: cross-encoder unavailable ({e}); "
              f"falling back to RRF order.")
        _RERANK_AVAILABLE = False
    return _reranker


# ------------------------------------------------------------ full pipeline
def hybrid_retrieve(query: str,
                    k_candidates: int | None = None,
                    top_k: int | None = None,
                    use_rerank: bool = True,
                    use_hybrid: bool = True,
                    use_parents: bool = True) -> list[dict]:
    """Returns up to top_k context passages, best first.

    The three flags exist so eval/run_ragas.py can measure the BASELINE
    (dense-only, no rerank, child chunks) against the FINAL pipeline using
    exactly the same code path. That is what makes the RAGAS table honest.
    """
    s = _load()
    k_candidates = k_candidates or config.K_CANDIDATES
    top_k = top_k or config.TOP_K

    if use_hybrid:
        cand = rrf_fuse([dense_search(query, k_candidates),
                         bm25_search(query, k_candidates)], k_candidates)
    else:
        cand = dense_search(query, k_candidates)

    children = [s["children"][i] for i in cand]

    if use_rerank and children:
        ce = get_reranker()
        if ce is not None:
            scores = ce.predict([(query, c["text"]) for c in children])
            order = np.argsort(-np.asarray(scores))
            children = [children[i] for i in order]
            for rank, c in enumerate(children):
                c["rerank_score"] = float(scores[order[rank]])

    # Retrieve small -> return large. Deduplicate parents, preserve rank order.
    out, seen = [], set()
    for c in children:
        key = c["parent_id"] if use_parents else c["id"]
        if key in seen:
            continue
        seen.add(key)
        payload = s["parents"][c["parent_id"]] if use_parents else c
        out.append({"id": key, "source": payload["source"], "text": payload["text"],
                    "matched_child": c["text"][:180],
                    "rerank_score": c.get("rerank_score")})
        if len(out) >= top_k:
            break
    return out


def format_context(passages: list[dict]) -> str:
    """Numbered, source-attributed context. Numbering matters: the synthesis
    prompt requires every EVIDENCE line to cite [S<n>], which is what the
    critic verifies."""
    return "\n\n".join(
        f"[S{i}] source={p['source']}\n{p['text']}" for i, p in enumerate(passages, 1))


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How many people are displaced by disasters each year?"
    for p in hybrid_retrieve(q):
        print(f"--- {p['id']} ({p['source']}) score={p['rerank_score']}")
        print(p["text"][:300], "\n")
