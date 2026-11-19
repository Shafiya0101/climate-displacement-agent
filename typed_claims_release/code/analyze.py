"""
STEP 5 - Produce every number the paper needs, already formatted as LaTeX.

Usage:
    python 05_analyze.py --judge judge_gpt-4o-mini.jsonl \
                         --replication judge_openai_gpt-oss-20b.jsonl \
                         [--pilot pilot_tb_results.json]

Prints:
  * Table 1 rows (proportions per class, both criteria)
  * Wilson 95% intervals on the faithfulness proportions
  * Mann-Whitney AUC against correct answers (correct for binary scores)
  * the cross-domain comparison table (TB vs displacement)
  * Fisher's exact on the dose-response, if a pilot file is given
"""
import argparse, json
from math import comb, sqrt
from pathlib import Path

Z = 1.959964
ROWS = ["correct", "ATT_near", "ATT_far", "SUB_near", "SUB_far"]

# August cycle, displacement domain, primary judge gpt-4o-mini, n=57
DISPLACEMENT = {
    "correct":  (57, 55), "ATT_near": (0, 55), "ATT_far": (0, 11),
    "SUB_near": (57, 36), "SUB_far":  (57, 3),
}
DISP_N = 57


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h) * 100, min(1.0, c + h) * 100


def auc_binary(p1, p2):
    """Mann-Whitney AUC when both samples are Bernoulli."""
    return p1 * (1 - p2) + 0.5 * (p1 * p2 + (1 - p1) * (1 - p2))


def fisher(a, b, c, d):
    obs = comb(a + b, a) * comb(c + d, c) / comb(a + b + c + d, a + c)
    tot = 0.0
    for i in range(0, min(a + b, a + c) + 1):
        j, k = a + b - i, a + c - i
        l = c + d - k
        if j < 0 or k < 0 or l < 0:
            continue
        p = comb(a + b, i) * comb(c + d, k) / comb(a + b + c + d, a + c)
        if p <= obs + 1e-12:
            tot += p
    return tot


def load(path):
    agg = {r: {"faith": [0, 0], "rel": [0, 0]} for r in ROWS}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["row"] not in agg:
            continue
        for crit in ("faith", "rel"):
            agg[r["row"]][crit][1] += 1
            agg[r["row"]][crit][0] += int(r[crit] == 1.0)
    return agg


def report(name, agg):
    print(f"\n===== {name} =====")
    print(f"{'class':10s} {'faithful':>12s} {'relevant':>12s}")
    for r in ROWS:
        fk, fn = agg[r]["faith"]
        rk, rn = agg[r]["rel"]
        fp = fk / fn if fn else 0
        rp = rk / rn if rn else 0
        print(f"{r:10s} {fk:3d}/{fn:<3d}={fp:.3f} {rk:3d}/{rn:<3d}={rp:.3f}")

    print("\nWilson 95% CI on faithfulness:")
    for r in ROWS:
        fk, fn = agg[r]["faith"]
        lo, hi = wilson(fk, fn)
        print(f"  {r:10s} {100*fk/fn if fn else 0:5.1f}%  [{lo:.1f}, {hi:.1f}]")

    ck, cn = agg["correct"]["rel"]
    pc = ck / cn if cn else 0
    print("\nRelevance AUC vs correct:")
    for r in ROWS[1:]:
        rk, rn = agg[r]["rel"]
        print(f"  {r:10s} {auc_binary(pc, rk/rn if rn else 0):.3f}")

    print("\nLaTeX table body:")
    for r in ROWS:
        fk, fn = agg[r]["faith"]
        rk, rn = agg[r]["rel"]
        lbl = {"correct": "correct & ---", "ATT_near": r"\attS          & near",
               "ATT_far": r"\attS          & far ", "SUB_near": r"\subS          & near",
               "SUB_far": r"\subS          & far "}[r]
        print(f"  {lbl} & {fk/fn if fn else 0:.3f} & {rk/rn if rn else 0:.3f} \\\\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--replication")
    ap.add_argument("--pilot")
    args = ap.parse_args()

    agg = load(args.judge)
    report(f"TB - primary judge ({Path(args.judge).stem})", agg)

    if args.replication and Path(args.replication).exists():
        report(f"TB - replication ({Path(args.replication).stem})",
               load(args.replication))

    print("\n===== CROSS-DOMAIN (the generality claim) =====")
    print(f"{'class':10s} {'displacement':>14s} {'TB':>14s}")
    for r in ROWS:
        dfk, drk = DISPLACEMENT[r]
        fk, fn = agg[r]["faith"]
        print(f"{r:10s} {dfk:3d}/{DISP_N} ={dfk/DISP_N:.3f} {fk:3d}/{fn:<3d}="
              f"{fk/fn if fn else 0:.3f}")
    print("\nThe claim holds across domains iff SUB rows are ~1.000 and ATT rows")
    print("~0.000 in BOTH columns.")

    if args.pilot and Path(args.pilot).exists():
        rows = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
        nocue = [r for r in rows if r.get("condition") == "nocue"]
        cued = [r for r in rows if r.get("condition") == "cued"]
        a = sum(1 for r in nocue if r.get("is_substitution_error"))
        b = len(nocue) - a
        c = sum(1 for r in cued if r.get("is_substitution_error"))
        d = len(cued) - c
        print("\n===== DOSE-RESPONSE =====")
        print(f"  nocue: {a} err / {len(nocue)}   cued: {c} err / {len(cued)}")
        if (a + b) and (c + d):
            print(f"  Fisher's exact p = {fisher(a, b, c, d):.4f}")


if __name__ == "__main__":
    main()
