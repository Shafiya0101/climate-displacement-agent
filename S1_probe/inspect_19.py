"""URGENT (professor's point 3): why did the type-aware check catch 19/57
framed near-miss items when it should catch 0?

Framed items are type-consistent by construction, so a predication check should
find nothing. If 19 are caught, either the framing is contaminated (old wording
left in) or the check is doing something else. This prints the 19 items so we
can see which.
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
items = json.loads((HERE/"probe_v2_clean.json").read_text(encoding="utf-8"))

FLOW_CUES  = ["new displacements","were recorded","were triggered","movements","during"]
STOCK_CUES = ["living in displacement","were living","at the end of","remained displaced","idps"]
PROJ_CUES  = ["projected","by 2050","projection"]

def detect_type(text):
    t=text.lower()
    if any(c in t for c in PROJ_CUES): return "PROJ"
    if any(c in t for c in STOCK_CUES): return "STOCK"
    if any(c in t for c in FLOW_CUES): return "FLOW"
    return "UNKNOWN"

print("Inspecting framed near-miss answers.")
print("For a CLEAN framed item: the answer's wording type should MATCH the")
print("near-miss claim's own type (that's what 'framed' means).\n")

caught, clean = [], []
for it in items:
    ans = it["answers"]["near_framed"]
    near_type = it["near_miss"]["temporal"]
    ans_type = detect_type(ans)
    if ans_type != near_type:
        caught.append((it["id"], near_type, ans_type, ans))
    else:
        clean.append(it["id"])

print(f"{len(caught)} framed items where answer wording type != near-miss type (CONTAMINATED):")
for cid, nt, at, ans in caught:
    print(f"\n  [{cid}] near-miss type={nt}  but answer reads as={at}")
    print(f"        answer: {ans}")

print(f"\n\n{len(clean)} items are clean (answer wording matches near-miss type).")
print(f"\nVERDICT: {len(caught)} items are contaminated framings -> explanation (b).")
print("These need their framing repaired so the wording fully matches the")
print("near-miss's own type, with no leftover gold-type words.")
