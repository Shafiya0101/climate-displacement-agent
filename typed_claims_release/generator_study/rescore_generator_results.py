"""Re-score generator_results.json with a corrected number extractor.

Why: the first Colab run used an extractor that could not read numbers below
1,000 (e.g. Gambia 2012, 871 prevalent cases), so those answers were scored
NEITHER. The evidence itself was not saved, so each survey's stock and flow are
reconstructed from the model answers to the two CUED questions (majority vote
over all generations; the script stops if they do not agree).

Run (same folder as generator_results.json):
    python rescore_generator_results.py
Writes generator_results_rescored.json with both the original and corrected label.
"""
import collections, json, re

R = json.load(open("generator_results.json", encoding="utf-8"))
PAT = re.compile(r"^(.*)_(\d{4})_(cued_stock|cued_flow|nocue)$")

def numbers(text):
    return [int(x.replace(",", "")) for x in re.findall(r"\d{1,3}(?:,\d{3})+|\d+", text)]

votes = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for r in R:
    country, year, cond = PAT.match(r["id"]).groups()
    if cond in ("cued_stock", "cued_flow"):
        for n in set(numbers(r["answer"])):
            if n != int(year):
                votes[(country, year)][cond][n] += 1

truth = {}
for key, d in votes.items():
    (s, s_n), (f, f_n) = d["cued_stock"].most_common(1)[0], d["cued_flow"].most_common(1)[0]
    assert s_n == f_n == 4, f"stock/flow reconstruction not unanimous for {key}"
    truth[key] = {"stock": s, "flow": f}

out = []
for r in R:
    country, year, cond = PAT.match(r["id"]).groups()
    t = truth[(country, year)]
    ns = [n for n in numbers(r["answer"]) if n != int(year)]
    has_s, has_f = t["stock"] in ns, t["flow"] in ns
    new = "BOTH" if has_s and has_f else "STOCK" if has_s else "FLOW" if has_f else "NEITHER"
    row = {k: v for k, v in r.items() if k != "type_number"}
    row.update(stock=t["stock"], flow=t["flow"],
               type_number_original=r["type_number"], type_number_corrected=new)
    out.append(row)

json.dump(out, open("generator_results_rescored.json", "w", encoding="utf-8"), indent=1)
changed = [o for o in out if o["type_number_original"] != o["type_number_corrected"]]
print(f"{len(out)} rows; {len(changed)} labels changed:")
for o in changed:
    print("  ", o["id"], o["model"], f"run{o['run']}", o["type_number_original"], "->", o["type_number_corrected"])
