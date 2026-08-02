"""S1: build the typed-claim probe from real report figures.

For each gold figure, find:
  - NEAR-miss: incommensurable (different type) but numerically CLOSEST
  - FAR-miss:  incommensurable and numerically REMOTE
Both are real figures from the same report. Then emit 6 answer versions.
"""
import json
from pathlib import Path

figs = json.load(open("figures_from_report.json"))["figures"]

def num(v):
    return float(v.split()[0])

def commensurable(a, b):
    return a["entity"] == b["entity"] and a["temporal"] == b["temporal"]

def frame_words(temporal):
    return {
        "FLOW":  ("new displacements were recorded", "during"),
        "STOCK": ("people were living in displacement", "at the end of"),
        "PROJ":  ("people are projected to be displaced", "by"),
    }[temporal]

def year_for(temporal):
    return "2050" if temporal == "PROJ" else "2023"

items = []
for gold in figs:
    g = num(gold["value"])
    cands = [f for f in figs if f["value"] != gold["value"]
             and not commensurable(gold, f)]
    if len(cands) < 2:
        continue
    cands.sort(key=lambda f: abs(num(f["value"]) - g))
    near = cands[0]
    # far: incommensurable AND at least 3x away, else take the farthest
    far_cands = [f for f in cands if abs(num(f["value"]) - g) / g >= 2.0]
    far = far_cands[-1] if far_cands else cands[-1]
    if far["value"] == near["value"]:
        continue

    def ans(claim, gamma):
        # bare  -> gold's own frame wording (near/far become internally inconsistent)
        # framed-> the claim's own-type wording (internally consistent, wrong vs question)
        t = claim["temporal"] if gamma == "framed" else gold["temporal"]
        verb, prep = frame_words(t)
        return f"{claim['value']} {verb} {prep} {year_for(t)}."

    items.append({
        "id": f"t{len(items)+1:02d}",
        "question": f"What is the reported figure for '{gold['label']}'?",
        "gold": {**gold, "delta": 0.0},
        "near_miss": {**near, "delta": round(abs(num(near['value'])-g)/g, 3)},
        "far_miss":  {**far,  "delta": round(abs(num(far['value']) -g)/g, 3)},
        "answers": {
            "correct_bare":   ans(gold, "bare"),
            "correct_framed": ans(gold, "framed"),
            "near_bare":      ans(near, "bare"),
            "near_framed":    ans(near, "framed"),
            "far_bare":       ans(far,  "bare"),
            "far_framed":     ans(far,  "framed"),
        },
        "gold_evidence": gold["label"],
    })

Path("probe_v2.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))

# stats for the paper (S1: delta ranges)
import statistics as st
dn = [it["near_miss"]["delta"] for it in items]
df = [it["far_miss"]["delta"] for it in items]
print(f"BUILT {len(items)} probe items, 6 answer versions each")
print(f"near-miss delta range: {min(dn):.3f} to {max(dn):.3f} (median {st.median(dn):.3f})")
print(f"far-miss  delta range: {min(df):.3f} to {max(df):.3f} (median {st.median(df):.3f})")
print("\n--- sample item ---")
print(json.dumps(items[2], indent=1, ensure_ascii=False))
