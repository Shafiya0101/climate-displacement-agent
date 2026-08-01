"""Analyse numeric_trap_results.jsonl.

The single question: does the judge treat a near-miss (wrong meaning, close
number) more like the correct answer or more like the far-miss?

    python numeric_trap_analyse.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path("numeric_trap_results.jsonl")


def main():
    if not RES.exists():
        print("No results yet. Run numeric_trap_score.py first.")
        return
    rows = [json.loads(l) for l in RES.read_text().splitlines() if l.strip()]
    vals = [r for r in rows if r["faithfulness"] is not None]

    def mean_for(version, judge=None):
        xs = [r["faithfulness"] for r in vals
              if r["version"] == version and (judge is None or r["judge"] == judge)]
        return (np.mean(xs), np.std(xs), len(xs)) if xs else (None, None, 0)

    print("=" * 60)
    print("OVERALL  (all judges pooled)")
    print("=" * 60)
    for v, label in [("correct", "correct number"),
                     ("near", "NEAR-miss (wrong meaning, close number)"),
                     ("far", "far-miss (obviously wrong)")]:
        m, s, n = mean_for(v)
        if m is not None:
            print(f"  {label:44s} mean faithfulness = {m:.3f}  (sd {s:.3f}, n={n})")

    c, _, _ = mean_for("correct")
    nr, _, _ = mean_for("near")
    fr, _, _ = mean_for("far")
    print("\n" + "-" * 60)
    if None in (c, nr, fr):
        print("  not enough data yet")
        return

    # The key quantity: how much closer is near-miss to correct than to far?
    gap_correct = c - nr          # how much the judge drops for a near-miss
    gap_far = c - fr              # how much it drops for an obvious wrong answer
    print("INTERPRETATION")
    print("-" * 60)
    print(f"  drop for near-miss : {gap_correct:+.3f}")
    print(f"  drop for far-miss  : {gap_far:+.3f}")
    if gap_far <= 0.05:
        print("\n  far-miss was NOT penalised either -> the judge is not")
        print("  discriminating at all. Finding inconclusive; check the prompt.")
    elif gap_correct < gap_far * 0.4:
        print("\n  *** FINDING HOLDS ***")
        print("  The judge penalises the obviously-wrong number but barely")
        print("  penalises the near-miss with the wrong MEANING. The metric")
        print("  has a numeric-proximity blind spot. This is the paper.")
    else:
        print("\n  Finding does NOT hold: the judge penalises near-miss and")
        print("  far-miss similarly. No special blind spot. Honest null result.")

    print("\n" + "=" * 60)
    print("PER JUDGE")
    print("=" * 60)
    for judge in sorted({r["judge"] for r in vals}):
        c2, _, _ = mean_for("correct", judge)
        n2, _, _ = mean_for("near", judge)
        f2, _, _ = mean_for("far", judge)
        if None not in (c2, n2, f2):
            holds = "HOLDS" if (c2 - n2) < (c2 - f2) * 0.4 and (c2 - f2) > 0.05 else "no"
            print(f"  {judge:32s} correct={c2:.2f} near={n2:.2f} far={f2:.2f}  [{holds}]")


if __name__ == "__main__":
    main()
