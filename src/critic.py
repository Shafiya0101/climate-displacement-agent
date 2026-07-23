"""The critic agent — the second agent role (Generator-Critic pattern).

This is RLAIF at small scale: an AI evaluator applying a written constitution to
another model's output. The critic never answers the question; it audits the
answer against the context and emits a visible verdict.

It runs two layers:
  1. deterministic checks (citation existence, invented figures, format) — free,
     cannot hallucinate;
  2. an LLM critic for semantic support of each claim.
Deterministic findings override the LLM verdict, because a regex that finds a
citation to a passage that does not exist is more reliable than a model's opinion.
"""
from __future__ import annotations

import re

try:
    from . import config
    from .guardrails import TokenBudget
    from .reasoning import CRITIC_SYSTEM_PROMPT, chat, extract_confidence, extract_section
except ImportError:
    import config  # type: ignore
    from guardrails import TokenBudget  # type: ignore
    from reasoning import CRITIC_SYSTEM_PROMPT, chat, extract_confidence, extract_section  # type: ignore

REQUIRED_SECTIONS = ("EVIDENCE", "ANALYSIS", "CONCLUSION", "CONFIDENCE")


def deterministic_checks(answer: str, context: str) -> list[str]:
    issues: list[str] = []

    for sec in REQUIRED_SECTIONS:
        if not re.search(rf"^\s*\**\s*{sec}\b", answer, re.MULTILINE | re.IGNORECASE):
            issues.append(f"FORMAT: section {sec} is missing.")

    available = set(re.findall(r"\[S(\d+)\]", context))
    cited = set(re.findall(r"\[S(\d+)\]", answer))
    ghosts = sorted(cited - available, key=int)
    if ghosts:
        issues.append("GROUNDING: cites " + ", ".join(f"[S{g}]" for g in ghosts)
                      + " which do not exist in the context.")
    if not cited:
        issues.append("GROUNDING: no [S<n>] citation anywhere in the answer.")

    # invented numbers: any figure of 3+ digits or a decimal not present in context
    ctx_nums = set(re.findall(r"\d[\d,\.]*", context))
    ctx_flat = {n.replace(",", "") for n in ctx_nums}
    for n in re.findall(r"\d[\d,\.]*", answer):
        flat = n.replace(",", "")
        if len(flat.replace(".", "")) >= 3 and flat not in ctx_flat and n not in ctx_nums:
            issues.append(f"NUMBERS: figure '{n}' does not appear in the retrieved context.")
            break

    conf = extract_confidence(answer)
    sources = set(re.findall(r"source=([^\s\]]+)", context))
    if conf == "HIGH" and len(cited) <= 1:
        issues.append("CALIBRATION: CONFIDENCE HIGH on a single cited passage.")
    if conf == "HIGH" and len(sources) <= 1:
        issues.append("CALIBRATION: CONFIDENCE HIGH but the context has one source document.")
    if conf == "UNKNOWN":
        issues.append("CALIBRATION: no parseable CONFIDENCE value.")

    if len(extract_section(answer, "CONCLUSION")) < 80:
        issues.append("COMPLETENESS: CONCLUSION is too short to answer the question.")

    return issues


def review(question: str, answer: str, context: str, budget: TokenBudget) -> dict:
    """Returns a visible verdict dict. Never raises — a critic failure must not
    take down the run, it downgrades to the deterministic verdict."""
    hard = deterministic_checks(answer, context)

    llm_verdict, llm_issues, rec_conf = "PASS", [], extract_confidence(answer)
    try:
        msg = chat(
            [{"role": "system", "content": CRITIC_SYSTEM_PROMPT},
             {"role": "user", "content":
                 f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
                 f"ANSWER UNDER REVIEW:\n{answer}"}],
            model=config.CRITIC_MODEL, temperature=0.0, budget=budget,
            span_name="critic_review", max_tokens=700)
        text = msg.content or ""
        m = re.search(r"VERDICT\s*:?\s*\**\s*(PASS|REVISE)", text, re.IGNORECASE)
        llm_verdict = m.group(1).upper() if m else "PASS"
        block = extract_section(text, "ISSUES")
        llm_issues = [ln.strip("- ").strip() for ln in block.splitlines()
                      if ln.strip().startswith("-") and "none" not in ln.lower()]
        rc = re.search(r"RECOMMENDED_CONFIDENCE\s*:?\s*\**\s*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
        if rc:
            rec_conf = rc.group(1).upper()
    except Exception as e:
        llm_issues = [f"LLM critic unavailable ({type(e).__name__}); "
                      f"verdict rests on deterministic checks only."]

    issues = hard + llm_issues
    verdict = "REVISE" if (hard or llm_verdict == "REVISE") else "PASS"
    return {"verdict": verdict, "issues": issues or ["none"],
            "deterministic_issues": hard,
            "recommended_confidence": rec_conf}


def render(v: dict) -> str:
    lines = [f"CRITIC VERDICT: {v['verdict']}",
             f"RECOMMENDED_CONFIDENCE: {v['recommended_confidence']}",
             "ISSUES:"]
    lines += [f"  - {i}" for i in v["issues"]]
    return "\n".join(lines)
