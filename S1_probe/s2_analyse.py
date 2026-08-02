"""Analyse S2: build the 4x3 table with bootstrap 95% CIs (professor's Table 1).

  python S1_probe/s2_analyse.py
"""
import json, statistics as st
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (HERE/"s2_results.jsonl").read_text().splitlines() if l.strip()]

def boot_ci(xs, n=10000):
    if len(xs) < 2: return (xs[0] if xs else 0, xs[0] if xs else 0)
    xs = np.array(xs)
    means = [np.mean(np.random.choice(xs, len(xs), replace=True)) for _ in range(n)]
    return (np.percentile(means, 2.5), np.percentile(means, 97.5))

def cell(evidence, gamma, value):
    xs = [r["faith"] for r in rows
          if r["evidence"]==evidence and r["gamma"]==gamma and r["value"]==value]
    if not xs: return None
    lo, hi = boot_ci(xs)
    return np.mean(xs), lo, hi, len(xs)

print("="*70)
print("TABLE 1 — Judge faithfulness by evidence x camouflage x value")
print("="*70)
print(f"{'Evidence':<11}{'camo':<8}{'correct':<18}{'near':<18}{'far':<18}")
print("-"*70)
for ev in ("SENTENCE","PARAGRAPH"):
    for g in ("bare","framed"):
        line=f"{ev:<11}{g:<8}"
        for val in ("correct","near","far"):
            c = cell(ev,g,val)
            if c: line += f"{c[0]:.2f} [{c[1]:.2f},{c[2]:.2f}]  "
            else: line += f"{'--':<18}"
        print(line)
print("-"*70)
print("DIAGNOSTIC CELL = PARAGRAPH x framed. If judge catches type errors,")
print("its 'near' score there should be LOW (like far), not high (like correct).")

# the key numbers for the paper text (S2)
d = cell("PARAGRAPH","framed","near")
c = cell("PARAGRAPH","framed","correct")
f = cell("PARAGRAPH","framed","far")
if d and c and f:
    print(f"\nPARAGRAPH x framed:  correct={c[0]:.2f}  near={d[0]:.2f}  far={f[0]:.2f}")
    if d[0] < (c[0]+f[0])/2:
        print("=> judge scores framed near-miss LOW: it DETECTS camouflaged type errors.")
    else:
        print("=> judge scores framed near-miss HIGH: it is FOOLED by camouflage.")
