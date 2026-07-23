# Architecture

## Diagram

```
                       ┌──────────────────────────────────────┐
   user question ─────►│  L1 INPUT FILTER  (guardrails.py)    │
                       │  NFKC normalise → strip zero-width   │
                       │  → 15 injection patterns             │
                       │  CLEAN / FLAGGED / BLOCKED           │
                       └──────────────┬───────────────────────┘
                          BLOCKED ◄───┤ refuse, log, return
                                      ▼ CLEAN
                       ┌──────────────────────────────────────┐
                       │  PLANNER LOOP  (agent.py)            │
                       │  llama-3.3-70b · T=0 · max 6 steps   │
                       └──────────────┬───────────────────────┘
                                      ▼ tool call
                       ┌──────────────────────────────────────┐
                       │  L4 ACTION GATE  (guardrails.py)     │
                       │  ACTION_RISK_MATRIX lookup           │
                       │  SAFE│MONITOR│CONFIRM│BLOCK          │
                       └──────────────┬───────────────────────┘
                     refused ◄────────┤ CONFIRM / BLOCK / unknown tool
                                      ▼ allowed
        ┌─────────────────────────────────────────────────────────┐
        │  TOOLS (tools.py — shared with mcp_server.py)           │
        │                                                         │
        │  recall_memory ──────► data/memory.json                 │
        │  search_displacement_corpus ──► RETRIEVAL (below)       │
        │  store_finding ──────► data/memory.json    [MONITOR]    │
        │  web_search ─────────► Tavily (optional)                │
        └──────────────┬──────────────────────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────────────────────┐
        │  sanitise_tool_result()  ← indirect-injection defence    │
        │  strip HTML → neutralise patterns → [EXTERNAL DATA] wrap │
        │  → truncate 3 000 chars                                  │
        └──────────────┬──────────────────────────────────────────┘
                       ▼ context
        ┌─────────────────────────────────────────────────────────┐
        │  SYNTHESIS  (reasoning.py)                              │
        │  few-shot CoT: EVIDENCE / ANALYSIS / CONCLUSION /       │
        │  CONFIDENCE                                             │
        │  Self-Consistency k=3 @ T=0.8 → embed conclusions →     │
        │  medoid = majority answer, agreement score reported     │
        └──────────────┬──────────────────────────────────────────┘
                       ▼ candidate answer
        ┌─────────────────────────────────────────────────────────┐
        │  CRITIC  (critic.py) — second agent role                │
        │  (a) deterministic: ghost [S<n>], invented figures,     │
        │      missing sections, HIGH-on-one-source               │
        │  (b) LLM critic @ T=0 against the same context          │
        │  VERDICT: PASS | REVISE  →  one revision pass on REVISE │
        └──────────────┬──────────────────────────────────────────┘
                       ▼
              answer + visible verdict + cost / latency / tool distribution

   RETRIEVAL (ingest.py + retrieval.py)
   corpus ─► parent-child chunk ─► children embedded (MiniLM-L6-v2)
        query ─┬─► dense top-15 ─┐
               └─► BM25  top-15 ─┴─► RRF  1/(60+rank) ─► 15 candidates
                     ─► cross-encoder (ms-marco-MiniLM-L-6) ─► top 5
                     ─► dedup to PARENT passages ─► [S1]..[S5]

   CROSS-CUTTING
   TokenBudget      — enforced inside reasoning.chat(); no LLM call escapes it
   observability    — one Langfuse span per agent run, LLM call and tool call;
                      system prompt SHA-256 (12 chars) logged as prompt_hash
```

## Components

| File | Responsibility |
|---|---|
| `src/config.py` | Single source of truth for models, paths, thresholds, pricing. No magic numbers elsewhere. |
| `src/ingest.py` | Corpus loading (md/txt/pdf), parent-child chunking, embedding, index persistence. |
| `src/retrieval.py` | `dense_search`, `bm25_search`, `rrf_fuse`, cross-encoder rerank, parent dedup, context formatting. |
| `src/tools.py` | The four tool implementations. Every tool returns a string and never raises. |
| `src/mcp_server.py` | FastMCP wrapper over `tools.py`, stdio transport. Docstrings written for the LLM. |
| `src/guardrails.py` | L1 filter, L4 gate, TokenBudget. Dependency-free so the security tests run standalone. |
| `src/reasoning.py` | Prompts, the single budgeted `chat()` entry point, Self-Consistency. |
| `src/critic.py` | Deterministic grounding checks + LLM critic, verdict rendering. |
| `src/observability.py` | Langfuse spans with a no-op fallback; prompt hashing for versioning. |
| `src/agent.py` | Orchestration, run logging, CLI. |

## Data flow invariants

1. **No LLM call bypasses the budget.** Every call goes through `reasoning.chat()`,
   which records usage and raises `BudgetExceeded` at the cap.
2. **No tool result reaches the model unsanitised.** `agent.py` calls
   `sanitise_tool_result()` on every return value before appending it to messages.
3. **No tool executes without a gate decision.** Unknown tools default to CONFIRM
   (fail closed), so adding a tool without classifying it cannot silently widen the
   attack surface.
4. **No answer is returned without a critic verdict**, and the verdict is printed,
   not swallowed.

## Design decisions worth defending

**Majority vote over free text.** Classic Self-Consistency assumes an extractable
answer token to count. Analytical prose has none. We embed the CONCLUSION section
of each of the k paths and select the *medoid* — the conclusion closest to all the
others — and report the mean pairwise agreement as a diagnostic. Low agreement is
itself a signal that the retrieved context underdetermines the answer.
Trade-off: the medoid selects a representative answer rather than synthesising a
consensus one, so a k=3 run where all three paths differ returns the least-odd of
three weak answers rather than flagging the disagreement as a failure. The
agreement score is reported so the reader can see when that happened.

**Deterministic checks override the LLM critic.** A regex that finds a citation to
`[S9]` when the context stops at `[S5]` is more reliable than a model's opinion
about grounding. The LLM critic catches semantic mismatch that regex cannot; the
regex catches fabrication that the LLM critic sometimes waves through. The
verdict is REVISE if *either* fires.

**Tools shared between the MCP server and the agent loop.** `mcp_server.py` is a
thin wrapper over `tools.py` rather than a reimplementation, so the MCP surface
and the in-process loop cannot drift apart and be tested separately into
disagreement.
