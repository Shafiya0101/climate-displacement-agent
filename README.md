# Climate Displacement Research Agent

A production-oriented research agent for humanitarian programme officers who must
justify climate-displacement funding decisions in writing. It retrieves from a
document corpus (IDMC, World Bank Groundswell, IPCC AR6, UNHCR, regional case
studies), reasons in an auditable format, and passes its own answer through a
critic before returning it.

**AIVANCITY PGE5 · Agentic AI · Topic 1 — Climate displacement**

---

## Live demo

**Deployed app:** https://huggingface.co/spaces/Shafiya1234/climate-displacement-agent

The interface shows each run as a chain of custody (L1 filter -> L4 gate + tools
-> reasoning -> critic), renders the answer as a document with clickable `[S<n>]`
citations, and carries a persistent AI-disclosure notice (EU AI Act Art. 50).

Try the **Security probe** example: it is refused by the L1 input filter in under
a second, at zero token cost, before the model is ever called.

To run the same interface locally:

```bash
pip install -r requirements-web.txt
uvicorn app:app --port 7860
```

---

## Quick start

```bash
git clone <your-repo-url>
cd <your-repo>

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # then add your GROQ_API_KEY (free: console.groq.com)

python src/agent.py           # runs the default scenario
python src/agent.py "What does the Falepili Union commit Australia to?"
```

First run downloads two small models (~180 MB total): the sentence-transformer
encoder and the cross-encoder reranker. The vector index is built automatically
from `data/corpus/` on first use.

### Verify the install

```bash
python -m pytest tests/test_security.py -v     # 10 tests, incl. the 5 injection tests
python src/ingest.py                            # rebuild the index explicitly
python src/retrieval.py "sea level rise atolls" # inspect retrieval only, no LLM
```

---

## What it does

| Requirement | Where it lives |
|---|---|
| Hybrid search (BM25 + dense + RRF) | `src/retrieval.py` — `bm25_search`, `dense_search`, `rrf_fuse` |
| Cross-encoder reranking | `src/retrieval.py` — `get_reranker`, applied in `hybrid_retrieve` |
| Parent-child chunking | `src/ingest.py` — `parent_child_chunk` (200-word children indexed, 800-word parents returned) |
| MCP server, 4 tools | `src/mcp_server.py` (implementations shared with the agent via `src/tools.py`) |
| L1 input filter | `src/guardrails.py` — `normalise`, `check_input`, `sanitise_tool_result` |
| L4 action gate | `src/guardrails.py` — `ACTION_RISK_MATRIX`, `ActionGate` |
| Token budget | `src/guardrails.py` — `TokenBudget`, enforced inside `reasoning.chat` |
| Few-shot CoT (EVIDENCE/ANALYSIS/CONCLUSION/CONFIDENCE) | `src/reasoning.py` — `SYNTHESIS_SYSTEM_PROMPT` |
| Self-Consistency k=3 | `src/reasoning.py` — `self_consistency_synthesis` |
| Critic agent | `src/critic.py` — deterministic checks + LLM critic, verdict printed every run |
| Langfuse instrumentation | `src/observability.py` — one span per agent run, LLM call and tool call |

Full component descriptions and the diagram: [`docs/architecture.md`](docs/architecture.md).
Evaluation, security results and the EU AI Act assessment: [`REPORT.md`](REPORT.md).

---

## Run pipeline

```
question
  ├─ L1 input filter ............... normalise (NFKC + strip zero-width) → injection patterns
  ├─ planner loop (max 6 steps) .... llama-3.3-70b @ T=0, function calling
  │    ├─ L4 action gate ........... SAFE / MONITOR / CONFIRM / BLOCK per tool
  │    ├─ tool execution ........... recall_memory · search_displacement_corpus · store_finding · web_search
  │    └─ sanitise_tool_result ..... wrap as [EXTERNAL DATA], neutralise injections, truncate to 3k
  ├─ synthesis ..................... few-shot CoT × Self-Consistency k=3 @ T=0.8, medoid vote
  ├─ critic ........................ deterministic grounding checks + LLM critic → PASS / REVISE
  └─ output ........................ answer + visible verdict + cost, latency, tool distribution
```

## The MCP server

```bash
npx @modelcontextprotocol/inspector python src/mcp_server.py
```

Four tools, each returning a string and never raising:

| Tool | L4 level | Purpose |
|---|---|---|
| `recall_memory` | SAFE | check what has already been verified, before paying for retrieval |
| `search_displacement_corpus` | SAFE | hybrid + reranked retrieval over the local corpus |
| `store_finding` | MONITOR | persist one sourced finding, logged prominently |
| `web_search` | SAFE | live search (Tavily); returns an explanatory string if unconfigured |

## Web interface (optional)

```bash
pip install -r requirements-web.txt
uvicorn app:app --port 7860
# http://localhost:7860
```

`app.py` and `web/index.html` are a separate entry point with separate
dependencies: `python src/agent.py` does not import them, so the web layer cannot
break the CLI. The interface shows the run as a chain of custody (L1 → L4/tools →
reason → verify), renders the answer as a document, and makes every `[S<n>]`
citation clickable — clicking one opens the retrieved passage that backs the
claim. Citations with no matching passage are marked in red.

It carries a persistent AI-disclosure notice, which is how this project
implements the EU AI Act Article 50 transparency obligation (REPORT.md §5).

Deployment to Hugging Face Spaces: [`deploy/README.md`](deploy/README.md).

## Evaluation

```bash
pip install -r requirements-eval.txt
python eval/run_ragas.py --config all          # baseline vs final, prints the report table
python eval/report_metrics.py --runs 10 --budget-demo
```

`eval/run_ragas.py` runs both configurations through the same `hybrid_retrieve`
code path with three flags flipped (`use_hybrid`, `use_rerank`, `use_parents`),
so nothing but the technique under test changes between the two columns.

## Configuration

All tunables are in `src/config.py` and overridable from `.env`: models, chunk
sizes, `K_CANDIDATES` (15), `TOP_K` (5), `RRF_K` (60), `MAX_STEPS`,
`SELF_CONSISTENCY_K`, `MAX_USD`, `WARN_USD`.

## Adding your own documents

Drop `.md`, `.txt` or `.pdf` files into `data/corpus/` and run
`python src/ingest.py`. See [`data/README.md`](data/README.md) for the source
list this corpus is built from.

## Group

Group — Shafiya Kausar, Clara Mapessi, Ketsia Talotsing and Viany Arnold Mbouyom Leumale — AIVANCITY PGE5, 2026.
