"""Corpus loading + parent-child chunking + index build (Block 1, Fix 1).

Parent-child rationale: a 150-word fixed chunk cuts facts in half, so neither
half is a complete answer and context_recall collapses. We index SMALL children
(precise matching) but return LARGE parents (full context) to the LLM.
Retrieve small, return large — precision and recall at the same time.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

try:
    from . import config
except ImportError:
    import config  # type: ignore

_embedder = None


def get_embedder():
    """Load the sentence-transformer once. Falls back to a deterministic hashing
    embedder if the model cannot be downloaded, so the pipeline still runs
    offline (documented as a limitation in REPORT.md)."""
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBED_MODEL)
        print(f"[ingest] dense encoder: {config.EMBED_MODEL}")
    except Exception as e:  # pragma: no cover
        print(f"[ingest] WARNING: {config.EMBED_MODEL} unavailable ({e}). "
              f"Falling back to hashing embeddings — semantic quality degraded.")
        _embedder = _HashingEmbedder()
    return _embedder


class _HashingEmbedder:
    """Offline fallback only. Bag-of-words hashed into a fixed vector."""
    dim = 512

    def encode(self, texts, **kw):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenise(t):
                out[i, hash(tok) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.clip(norms, 1e-9, None)
        return out[0] if single else out


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ------------------------------------------------------------------ loading
def load_documents() -> list[dict]:
    """Read every .md/.txt/.pdf in data/corpus/."""
    docs = []
    for path in sorted(config.CORPUS_DIR.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"} or path.name == "README.md":
            continue
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        if text and text.strip():
            docs.append({"source": path.name, "text": text})
    return docs


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    except Exception as e:  # pragma: no cover
        print(f"[ingest] could not read {path.name}: {e}")
        return ""


# ----------------------------------------------------------------- chunking
def _windows(words: list[str], size: int, overlap: int) -> list[tuple[int, int]]:
    step = max(1, size - overlap)
    spans, i = [], 0
    while i < len(words):
        spans.append((i, min(i + size, len(words))))
        if i + size >= len(words):
            break
        i += step
    return spans


def parent_child_chunk(docs: list[dict]) -> tuple[list[dict], list[dict]]:
    parents, children = [], []
    for doc in docs:
        words = doc["text"].split()
        if not words:
            continue
        for (ps, pe) in _windows(words, config.PARENT_WORDS, config.PARENT_WORDS // 8):
            pid = f"P{len(parents):05d}"
            ptext = " ".join(words[ps:pe])
            parents.append({"id": pid, "source": doc["source"], "text": ptext})
            pwords = ptext.split()
            for (cs, ce) in _windows(pwords, config.CHILD_WORDS, config.CHILD_OVERLAP):
                children.append({
                    "id": f"C{len(children):05d}",
                    "parent_id": pid,
                    "source": doc["source"],
                    "text": " ".join(pwords[cs:ce]),
                })
    return parents, children


# -------------------------------------------------------------- index build
def build_index(verbose: bool = True) -> dict:
    docs = load_documents()
    if not docs:
        raise RuntimeError(
            f"No documents found in {config.CORPUS_DIR}. See data/README.md.")
    parents, children = parent_child_chunk(docs)
    if verbose:
        print(f"[ingest] {len(docs)} documents -> {len(parents)} parents, "
              f"{len(children)} children")

    emb = get_embedder()
    vectors = np.asarray(emb.encode([c["text"] for c in children],
                                    show_progress_bar=False), dtype=np.float32)
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.INDEX_DIR / "child_vectors.npy", vectors)
    (config.INDEX_DIR / "chunks.json").write_text(
        json.dumps({"parents": parents, "children": children,
                    "embed_model": getattr(emb, "_model_name", config.EMBED_MODEL)},
                   ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"[ingest] index written to {config.INDEX_DIR}")
    return {"parents": len(parents), "children": len(children), "documents": len(docs)}


def index_exists() -> bool:
    return ((config.INDEX_DIR / "chunks.json").exists()
            and (config.INDEX_DIR / "child_vectors.npy").exists())


if __name__ == "__main__":
    build_index()
