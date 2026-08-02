"""S4: type-aware critic ablation.

The baseline critic asks: is the stated VALUE present in the retrieved context?
  -> blind to type errors (the value IS present; only the type is wrong)

The type-aware critic additionally asks: does the answer's PREDICATION
(flow / stock / projection wording) match the TYPE of the context claim that
carries that value?
  -> catches type errors deterministically.

This module provides both checks and a demo that verifies the logic on the
probe's known-type answers. When S3 has produced generator answers, the same
`type_aware_verdict` is run over them to produce the S4 numbers (reduction in
type-error rate).

Run:  python S1_probe/s4_type_critic.py        # self-test on probe answers
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
items = json.loads((HERE / "probe_v2_clean.json").read_text(encoding="utf-8"))

# --- type detection from wording ---
FLOW_CUES  = ["new displacements", "were recorded", "were triggered", "movements", "during"]
STOCK_CUES = ["living in displacement", "were living", "at the end of", "remained displaced", "idps"]
PROJ_CUES  = ["projected", "by 2050", "projection"]

def detect_type(text):
    t = text.lower()
    if any(c in t for c in PROJ_CUES):  return "PROJ"
    if any(c in t for c in STOCK_CUES): return "STOCK"
    if any(c in t for c in FLOW_CUES):  return "FLOW"
    return "UNKNOWN"

def valnum(v):
    m = re.search(r"\d+\.?\d*", v)
    return float(m.group()) if m else None

def context_claims(it):
    # the real claims available in retrieved context: gold + both distractors
    return [it["gold"], it["near_miss"], it["far_miss"]]

def value_only_verdict(answer_text, it):
    """Baseline critic: is the stated value present in context? (type-blind)"""
    nums = [float(n) for n in re.findall(r"\d+\.?\d*", answer_text) if re.search(r"\d", n)]
    ctx_vals = [valnum(c["value"]) for c in context_claims(it)]
    for s in nums:
        if any(abs(s-cv) < 0.05*max(cv,1) for cv in ctx_vals if cv):
            return "PASS"   # value found -> baseline critic is satisfied
    return "REVISE"

def type_aware_verdict(answer_text, it):
    """Type-aware critic: value present AND predication type matches the
    context claim that carries that value."""
    nums = [float(n) for n in re.findall(r"\d+\.?\d*", answer_text) if re.search(r"\d", n)]
    ans_type = detect_type(answer_text)
    for s in nums:
        for c in context_claims(it):
            cv = valnum(c["value"])
            if cv and abs(s-cv) < 0.05*max(cv,1):
                # value matches THIS context claim; now check type agreement
                if ans_type == c["temporal"]:
                    return "PASS"       # value present AND type matches
                else:
                    return "REVISE"     # value present but TYPE mismatched -> type error caught
    return "REVISE"  # value not in context at all

def self_test():
    """Verify the two critics on the probe's own framed answers, where we KNOW
    the ground truth. The framed near-miss is a TYPE ERROR by construction."""
    base_catch = type_catch = total = 0
    for it in items:
        # the framed near-miss answer: correct value-in-context, WRONG type
        ans = it["answers"]["near_framed"]
        total += 1
        # baseline should mostly PASS it (blind); type-aware should REVISE it (caught)
        if value_only_verdict(ans, it) == "REVISE":
            base_catch += 1
        if type_aware_verdict(ans, it) == "REVISE":
            type_catch += 1
    print(f"On {total} framed near-miss answers (each a TRUE type error):")
    print(f"  baseline value-only critic caught: {base_catch}/{total}  ({base_catch/total*100:.0f}%)")
    print(f"  type-aware critic caught:          {type_catch}/{total}  ({type_catch/total*100:.0f}%)")
    print(f"\n  => type-aware check catches {type_catch-base_catch} more type errors "
          f"that the value-only critic misses.")
    print("\n  (Final S4 numbers come from running type_aware_verdict over the")
    print("   S3 generator answers once S3 has run. This self-test verifies the")
    print("   critic logic is correct on known-type items.)")

if __name__ == "__main__":
    self_test()
