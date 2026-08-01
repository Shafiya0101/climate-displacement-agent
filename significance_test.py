"""Significance test for the numeric-trap results (professor request #3).

Tests whether near-miss faithfulness scores differ significantly from correct
and from far-miss, using Mann-Whitney U (non-parametric, correct for scores
that clump at 0 and 1). Reports per-condition p-values and effect sizes.
"""
import json, sys
from pathlib import Path
from scipy.stats import mannwhitneyu
import numpy as np

def load(fn):
    rows = [json.loads(l) for l in Path(fn).read_text().splitlines() if l.strip()]
    return [r for r in rows if r["faithfulness"] is not None]

def scores(rows, version):
    return [r["faithfulness"] for r in rows if r["version"] == version]

for label, fn in [("CLEAN", "numeric_trap_results.jsonl"),
                  ("HARD",  "numeric_trap_results_hard.jsonl")]:
    rows = load(fn)
    correct = scores(rows, "correct")
    near    = scores(rows, "near")
    far     = scores(rows, "far")
    print(f"\n=== {label} condition (n={len(near)} per group) ===")
    print(f"  correct  mean={np.mean(correct):.3f}")
    print(f"  near     mean={np.mean(near):.3f}")
    print(f"  far      mean={np.mean(far):.3f}")

    # near vs correct: is near-miss significantly LOWER than correct?
    u1, p1 = mannwhitneyu(near, correct, alternative="less")
    # near vs far: is near-miss significantly HIGHER than far? (tests if judge
    # treats near-miss as different from an obvious error)
    u2, p2 = mannwhitneyu(near, far, alternative="greater")

    print(f"  near < correct : U={u1:.1f}, p={p1:.2e}  "
          f"({'significant' if p1 < 0.05 else 'n.s.'})")
    print(f"  near > far     : U={u2:.1f}, p={p2:.2e}  "
          f"({'significant' if p2 < 0.05 else 'n.s.'})")
