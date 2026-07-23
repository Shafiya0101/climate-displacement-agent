"""Block 3: few-shot CoT + Self-Consistency.

The synthesis prompt fixes the reasoning FORMAT
(EVIDENCE / ANALYSIS / CONCLUSION / CONFIDENCE) because:
  * EVIDENCE makes the reasoning auditable â€” the critic can check each line
    against the retrieved context independently;
  * ANALYSIS forces decomposition before the model commits to an answer;
  * CONFIDENCE is a control signal, not decoration: LOW triggers another
    retrieval pass, and HIGH on a single source is treated as suspicious.

The performance measure below is a reward function, so it names the failure
modes explicitly. "Answer the question efficiently" is maximised by returning
nothing â€” zero tokens, zero latency, task nominally complete.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

import numpy as np

try:
    from . import config, observability
    from .guardrails import TokenBudget
except ImportError:
    import config, observability  # type: ignore
    from guardrails import TokenBudget  # type: ignore

# =========================================================================
# PROMPTS
# =========================================================================
SYNTHESIS_SYSTEM_PROMPT = """You are a climate displacement research analyst supporting humanitarian \
programme officers who must justify allocation decisions in writing.

PERFORMANCE MEASURE â€” you are judged on all four, not on speed:
1. ACCURACY â€” every factual claim traces to a specific [S<n>] passage in the provided context.
2. COMPLETENESS â€” address every part of the question, including the parts the context cannot answer.
3. CALIBRATION â€” state confidence and justify it in one sentence.
4. TRACEABILITY â€” the analyst must be able to check each claim without re-reading the corpus.

EXPLICITLY FORBIDDEN (these are graded as failures, not as efficiency):
- Returning an empty or near-empty answer when the context contains relevant passages.
- Citing an [S<n>] identifier that does not appear in the context.
- Stating a number that is not in the context.
- Reporting CONFIDENCE: HIGH when the claim rests on a single source.
- Silently omitting a sub-question you could not answer â€” say "not covered by the retrieved context".

OUTPUT FORMAT â€” use exactly these four sections, in this order:

EVIDENCE
- [S<n>] <source name>: <the specific fact, with its figure and year if present>
- (one line per fact actually used; omit passages you did not use)

ANALYSIS
Step 1: <what the evidence establishes>
Step 2: <what it does not establish, or where sources disagree>
Step 3: <reconcile the conflict, or state that it cannot be reconciled from this context>

CONCLUSION
<3-6 sentences answering the question directly, every claim carrying its [S<n>]>

CONFIDENCE: HIGH | MEDIUM | LOW
<one sentence: how many independent sources support the conclusion, and what is missing>
"""

FEW_SHOT_EXAMPLE = """Here is one worked example of the required format.

QUESTION: Is annual disaster displacement rising because disasters are worsening?

EVIDENCE
- [S1] idmc_global_report: weather-related hazards account for the large majority of new internal displacements; floods and storms dominate.
- [S1] idmc_global_report: the count measures movements, not people â€” one person displaced three times counts three times.
- [S4] bangladesh_delta: pre-emptive mass evacuation is a major driver of totals in Bangladesh and the Philippines.

ANALYSIS
Step 1: The evidence establishes that weather-related hazards dominate the totals [S1].
Step 2: It does not establish that a rising count means worsening outcomes: the metric counts movements rather than people [S1], and evacuation inflates it [S4].
Step 3: These reconcile if the count is read as an exposure-and-response indicator rather than a harm indicator. Bangladesh shows displacement rising while deaths fell [S4].

CONCLUSION
Rising disaster displacement counts cannot be read directly as worsening outcomes [S1][S4]. The metric counts movement events rather than individuals, so repeat displacement inflates it [S1]. A large share of movement is pre-emptive evacuation, which indicates a functioning warning system [S4]. Bangladesh is the clearest case: displacement figures rose while cyclone mortality fell by orders of magnitude [S4]. Attribution of a trend to hazard intensity alone is not supported by the retrieved context.

CONFIDENCE: MEDIUM
Two independent sources support the metric caveat, but the context contains no year-on-year time series, so the direction of the underlying trend cannot be verified here.
"""

CRITIC_SYSTEM_PROMPT = """You are a verification critic. You do not answer the question; you audit \
another agent's answer against the context it was given.

Check, in order:
1. GROUNDING â€” does every [S<n>] cited actually exist in the context, and does the cited passage
   support the claim attached to it?
2. NUMBERS â€” is every figure in the answer present in the context? Invented or altered figures are
   the most serious failure.
3. FORMAT â€” are all four sections present (EVIDENCE, ANALYSIS, CONCLUSION, CONFIDENCE)?
4. CALIBRATION â€” is CONFIDENCE: HIGH claimed on the basis of a single source? That is an automatic
   downgrade.
5. COMPLETENESS â€” is any part of the question left unaddressed without being flagged as uncovered?

Reply in exactly this format:

VERDICT: PASS | REVISE
ISSUES:
- <at most 4 lines. Report ONLY unsupported claims, invented figures, missing sections of the four-section schema, or miscalibrated confidence. Do NOT list topics the answer could have covered. Write "none" if there are none.>
RECOMMENDED_CONFIDENCE: HIGH | MEDIUM | LOW
"""


def build_synthesis_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLE},
        {"role": "user", "content":
            f"CONTEXT (the only admissible evidence):\n{context}\n\n"
            f"QUESTION: {question}\n\nAnswer in the required four-section format."},
    ]


# =========================================================================
# GROQ CLIENT
# =========================================================================
_client = None


def get_client():
    global _client
    if _client is None:
        from groq import Groq
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "(free at https://console.groq.com).")
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.0,
         tools_schema: list[dict] | None = None, budget: TokenBudget | None = None,
         span_name: str = "llm_call", max_tokens: int = 2600) -> Any:
    """One LLM call: traced, budgeted, and cost-recorded. Every call goes
    through here so no call can escape the token budget."""
    model = model or config.SYNTH_MODEL
    kwargs: dict[str, Any] = {"model": model, "messages": messages,
                              "temperature": temperature, "max_tokens": max_tokens}
    if tools_schema:
        kwargs["tools"] = tools_schema
        kwargs["tool_choice"] = "auto"

    with observability.span(span_name, kind="llm", model=model,
                            temperature=temperature) as sp:
        resp = get_client().chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        if budget is not None:
            budget.record(model, pt, ct)          # raises BudgetExceeded at the cap
        try:
            sp.update(usage_details={"input": pt, "output": ct},
                      metadata={"model": model,
                                "spent_usd": round(budget.spent_usd, 6) if budget else None})
        except Exception:
            pass
    return resp.choices[0].message


# =========================================================================
# SELF-CONSISTENCY
# =========================================================================
def extract_section(text: str, name: str) -> str:
    m = re.search(rf"^{name}\s*:?\s*(.*?)(?=^\s*(?:EVIDENCE|ANALYSIS|CONCLUSION|CONFIDENCE)\b|\Z)",
                  text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_confidence(text: str) -> str:
    m = re.search(r"CONFIDENCE\s*:?\s*\**\s*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def _medoid(texts: list[str]) -> tuple[int, float]:
    """Majority vote over free text: embed each candidate conclusion and pick the
    one closest to all the others (the medoid). A wrong answer only wins if the
    same bias appears in a majority of independent paths â€” idiosyncratic errors
    do not survive. Returns (winning_index, mean_pairwise_agreement)."""
    try:
        from .ingest import get_embedder
    except ImportError:
        from ingest import get_embedder  # type: ignore
    vecs = np.asarray(get_embedder().encode(texts), dtype=np.float32)
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, 0.0)
    mean_sim = sim.sum(axis=1) / max(len(texts) - 1, 1)
    return int(np.argmax(mean_sim)), float(mean_sim.mean())


def self_consistency_synthesis(question: str, context: str, budget: TokenBudget,
                               k: int | None = None) -> dict:
    """Generate k independent reasoning paths at high temperature, then take the
    majority answer. Cost is k x tokens, so this runs ONLY on the final synthesis
    step â€” never inside the tool loop."""
    k = k or config.SELF_CONSISTENCY_K
    messages = build_synthesis_messages(question, context)
    paths: list[str] = []
    for i in range(k):
        temp = 0.1 if k == 1 else config.SC_TEMPERATURE
        if i:
            # Self-consistency fires k calls back to back. On a low
            # tokens-per-minute tier that alone can breach the window, so we
            # space the paths out rather than relying on SDK retries.
            time.sleep(float(os.getenv("SC_PATH_DELAY", "8")))
        msg = chat(messages, model=config.SYNTH_MODEL, temperature=temp,
                   budget=budget, span_name=f"synthesis_path_{i+1}")
        paths.append(msg.content or "")

    conclusions = [extract_section(p, "CONCLUSION") or p for p in paths]
    winner, agreement = (0, 1.0) if k == 1 else _medoid(conclusions)
    confidences = [extract_confidence(p) for p in paths]

    # A path count that disagrees is itself a calibration signal.
    majority_conf = max(set(confidences), key=confidences.count)
    final = paths[winner]
    if len(set(confidences)) > 1 and majority_conf != extract_confidence(final):
        final = re.sub(r"CONFIDENCE\s*:?\s*\**\s*(HIGH|MEDIUM|LOW)",
                       f"CONFIDENCE: {majority_conf}", final, flags=re.IGNORECASE)

    return {"answer": final, "k": k, "paths": paths,
            "agreement": round(agreement, 3),
            "confidences": confidences,
            "confidence": extract_confidence(final)}
