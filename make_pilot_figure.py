"""Generate the pilot-results bar chart from the numeric-trap data.
Reads numeric_trap_results.jsonl and _hard.jsonl, plots mean faithfulness
per answer version. Produces pilot_results.pdf for the paper.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def means(fn):
    rows = [json.loads(l) for l in Path(fn).read_text().splitlines() if l.strip()]
    out = {}
    for v in ("correct", "near", "far"):
        xs = [r["faithfulness"] for r in rows
              if r["version"] == v and r["faithfulness"] is not None]
        out[v] = np.mean(xs) if xs else 0.0
    return out

clean = means("numeric_trap_results.jsonl")
hard  = means("numeric_trap_results_hard.jsonl")

correct = [clean["correct"], hard["correct"]]
near    = [clean["near"],    hard["near"]]
far     = [clean["far"],     hard["far"]]

x = np.arange(2); w = 0.25
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.bar(x - w, correct, w, label="correct",   color="#2c7fb8")
ax.bar(x,      near,    w, label="near-miss", color="#d95f0e")
ax.bar(x + w,  far,     w, label="far-miss",  color="#999999")
ax.set_ylabel("mean faithfulness")
ax.set_xticks(x); ax.set_xticklabels(["Clean", "Hard"])
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("pilot_results.pdf", bbox_inches="tight")
print("saved pilot_results.pdf from your data")
