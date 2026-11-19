"""
Build the TB probe from the WHO Global TB Report 2024 data repository
(github.com/GTB-TME/gtbreport2024), file data/ps/prev_notif_2023-07-03.csv.

Why this source rather than mining the PDF: the type of each quantity comes
from the data schema, not from cue words. The `prev` column is a prevalence
(a STOCK: cases existing at survey time) and `notified` is a notification
count (a FLOW: cases reported during the year). Both are counts of TB cases
in the same country, the same year and the same population, and they differ
by a factor of roughly 0.6 to 6. That is exactly the configuration the paper
needs: numerically comparable, semantically incommensurable.

Usage:
    python 02b_build_probe_from_who_data.py \
        --csv gtb/data/ps/prev_notif_2023-07-03.csv --out probe_tb.json
"""
import argparse, csv, json, random
from pathlib import Path

STOCK_T = ("A national TB prevalence survey in {country} estimated "
           "{v} prevalent cases of TB in {year}.")
FLOW_T = ("{v} TB cases were notified to national authorities in "
          "{country} during {year}.")
Q_STOCK = ("How many prevalent cases of TB were estimated by the national "
           "prevalence survey in {country} in {year}?")
Q_FLOW = ("How many TB cases were notified to national authorities in "
          "{country} during {year}?")


def fmt(n):
    n = round(n)
    return f"{n:,}"


def delta(a, b):
    return abs(b - a) / a if a else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="probe_tb.json")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    recs = []
    for r in csv.DictReader(open(args.csv, encoding="utf-8")):
        try:
            prev_rate, pop, notif = float(r["prev"]), float(r["pop"]), float(r["notified"])
        except (ValueError, KeyError):
            continue
        prev_abs = prev_rate / 1e5 * pop
        if prev_abs <= 0 or notif <= 0:
            continue
        country = r["country"].split("(")[0].strip()
        recs.append({"country": country, "year": r["year"],
                     "stock": prev_abs, "flow": notif})
    print(f"{len(recs)} country-surveys with both a stock and a flow")

    random.seed(args.seed)
    items = []
    for i, rec in enumerate(recs):
        others = [o for o in recs if o is not rec]
        for gold_type in ("STOCK", "FLOW"):
            gv = rec[gold_type.lower()]
            nv = rec["flow" if gold_type == "STOCK" else "stock"]
            d_near = delta(gv, nv)
            if not (0.03 <= d_near <= 1.0):
                continue
            # far: another country's figure of the OTHER type, 5x-500x away
            cands = []
            for o in others:
                ov = o["flow" if gold_type == "STOCK" else "stock"]
                d = delta(gv, ov)
                if 5.0 <= d <= 500.0:
                    cands.append((d, o, ov))
            if not cands:
                continue
            cands.sort(key=lambda x: x[0])
            d_far, orec, ov = cands[len(cands) // 2]

            g_t = STOCK_T if gold_type == "STOCK" else FLOW_T
            n_t = FLOW_T if gold_type == "STOCK" else STOCK_T
            f_t = n_t
            gold_s = g_t.format(v=fmt(gv), country=rec["country"], year=rec["year"])
            near_s = n_t.format(v=fmt(nv), country=rec["country"], year=rec["year"])
            far_s = f_t.format(v=fmt(ov), country=orec["country"], year=orec["year"])

            att_near = gold_s.replace(fmt(gv), fmt(nv), 1)
            att_far = gold_s.replace(fmt(gv), fmt(ov), 1)

            items.append({
                "id": f"tb{len(items)+1:02d}",
                "question": (Q_STOCK if gold_type == "STOCK" else Q_FLOW).format(
                    country=rec["country"], year=rec["year"]),
                "gold": {"value": fmt(gv), "label": gold_s, "entity": "cases",
                         "temporal": gold_type, "unit": "count", "delta": 0.0},
                "near_miss": {"value": fmt(nv), "label": near_s, "entity": "cases",
                              "temporal": "FLOW" if gold_type == "STOCK" else "STOCK",
                              "unit": "count", "delta": round(d_near, 3)},
                "far_miss": {"value": fmt(ov), "label": far_s, "entity": "cases",
                             "temporal": "FLOW" if gold_type == "STOCK" else "STOCK",
                             "unit": "count", "delta": round(d_far, 1)},
                "evidence": " ".join([gold_s, near_s, far_s]),
                "answers": {
                    "correct":  gold_s,
                    "ATT_near": att_near,
                    "ATT_far":  att_far,
                    "SUB_near": near_s,
                    "SUB_far":  far_s,
                },
            })

    Path(args.out).write_text(json.dumps(items, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    if items:
        med = lambda xs: sorted(xs)[len(xs) // 2]
        print(f"\nbuilt {len(items)} items -> {args.out}")
        print(f"  median near delta = {med([i['near_miss']['delta'] for i in items]):.3f}")
        print(f"  median far  delta = {med([i['far_miss']['delta'] for i in items]):.1f}")
        from collections import Counter
        print("  gold types:", dict(Counter(i["gold"]["temporal"] for i in items)))
    else:
        print("no items built")


if __name__ == "__main__":
    main()
