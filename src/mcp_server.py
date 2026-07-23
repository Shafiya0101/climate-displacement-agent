"""MCP server for the climate displacement agent (Block 1, part 2).

Four tools over the stdio transport — zero network configuration.
Test visually with:
    npx @modelcontextprotocol/inspector python src/mcp_server.py

Every tool wraps the shared implementation in src/tools.py, so the MCP surface
and the in-process agent loop can never drift apart. Every tool returns a
string and never raises: an uncaught exception here disconnects the agent.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from src import tools  # noqa: E402

mcp = FastMCP("climate-displacement")


@mcp.tool()
def search_displacement_corpus(query: str, top_k: int = 5) -> str:
    """Search the indexed climate-displacement corpus (IDMC, World Bank Groundswell,
    IPCC AR6, UNHCR, regional case studies) with hybrid retrieval (BM25 + dense +
    RRF) and cross-encoder reranking.

    Use when: the question concerns displacement figures, projections, legal status,
    regional case studies, adaptation or response finance. PRIMARY tool — try it
    before web_search.
    Do NOT use for: arithmetic, events after the corpus cut-off, or facts already
    returned by recall_memory in this session.
    Returns: numbered passages [S1]..[Sn] with source filename, ranked by
    cross-encoder relevance.
    Example: query="annual number of people displaced by weather-related disasters"
    """
    return tools.search_displacement_corpus(query, top_k)


@mcp.tool()
def recall_memory(query: str, source: str = "", limit: int = 5) -> str:
    """Retrieve findings verified and stored in an earlier step or session.

    Use FIRST, before search_displacement_corpus or web_search — avoids redundant
    retrieval and repeated cost.
    Do NOT use for: storing (use store_finding), or as a corpus substitute when
    nothing has been stored yet.
    Returns: stored findings with source, url and timestamp, or an explicit
    instruction to search instead.
    Example: query="Tuvalu mobility quota", source="Falepili Union"
    """
    return tools.recall_memory(query, source, limit)


@mcp.tool()
def store_finding(finding: str, source: str, url: str = "", topic: str = "") -> str:
    """Persist one verified finding for later recall.

    Use after a retrieval step produced a fact you actually saw in the context.
    Do NOT store: speculation, your own inference, or anything without a named
    source. One finding per call.
    Returns: storage confirmation with the assigned memory id.
    Example: finding="Up to 216 million internal climate migrants by 2050",
             source="World Bank Groundswell Part 2", topic="projections"
    """
    return tools.store_finding(finding, source, url, topic)


@mcp.tool()
def web_search(query: str, max_results: int = 3) -> str:
    """Search the live web for facts newer than the local corpus.

    Use when: the question needs post-corpus data, or corpus search returned
    nothing relevant.
    Do NOT use for: anything the corpus or memory already answered, arithmetic,
    or opinion.
    Returns: title, url and summary per result, or an explanatory message when the
    provider is unconfigured.
    Example: query="IDMC global report internal displacement 2026"
    """
    return tools.web_search(query, max_results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
