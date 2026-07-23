"""Langfuse instrumentation (Block 4).

Every LLM call and every tool call gets its own span. If Langfuse is not
configured — or the installed SDK has a different API — this degrades to a
no-op so the agent never crashes because of telemetry. Telemetry that can take
down the system it observes is worse than no telemetry.
"""
from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from typing import Any

try:
    from . import config
except ImportError:
    import config  # type: ignore

_client = None
_ENABLED = False


def init() -> bool:
    global _client, _ENABLED
    if _client is not None:
        return _ENABLED
    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        print("[langfuse] not configured — tracing disabled (set LANGFUSE_* in .env)")
        _client = False
        return False
    try:
        from langfuse import Langfuse
        _client = Langfuse(public_key=config.LANGFUSE_PUBLIC_KEY,
                           secret_key=config.LANGFUSE_SECRET_KEY,
                           host=config.LANGFUSE_HOST)
        _ENABLED = True
        print(f"[langfuse] tracing enabled -> {config.LANGFUSE_HOST}")
    except Exception as e:  # pragma: no cover
        print(f"[langfuse] disabled ({e})")
        _client = False
    return _ENABLED


def prompt_version(system_prompt: str) -> str:
    """Block 4: hash the system prompt so any behaviour change is traceable to
    a prompt change rather than being invisible."""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]


class _NoSpan:
    def update(self, **kw): return self
    def end(self, **kw): return self


@contextmanager
def span(name: str, kind: str = "span", **payload: Any):
    """One span per agent run / LLM call / tool call."""
    init()
    t0 = time.time()
    handle = _NoSpan()
    if _ENABLED:
        try:
            starter = (getattr(_client, "start_as_current_span", None)
                       or getattr(_client, "start_span", None))
            if starter:
                cm = starter(name=name, input=payload or None)
                handle = cm.__enter__() if hasattr(cm, "__enter__") else cm
        except Exception:
            handle = _NoSpan()
    try:
        yield handle
    finally:
        elapsed = round(time.time() - t0, 3)
        try:
            handle.update(metadata={"kind": kind, "latency_s": elapsed,
                                    "agent_version": config.AGENT_VERSION})
            handle.end()
        except Exception:
            pass


def flush():
    if _ENABLED and _client:
        try:
            _client.flush()
        except Exception:
            pass
