# ===================================================================
#  Generator study - TB domain.
#  Paste this whole block into ONE cell of a blank Colab notebook,
#  put your two keys below, then Run.
#  ~15 min. Groq is free and runs first; OpenAI costs about $0.05.
#  The PILOT section stops the run if any component misbehaves.
# ===================================================================
!pip -q install openai

GROQ_KEY   = ""   # gsk_...  free
OPENAI_KEY = ""   # sk-...   ~$0.05 for the full run

N_ITEMS   = 50    # target number of questions
N_RUNS    = 2     # independent repetitions (needed for stability)

assert GROQ_KEY or OPENAI_KEY, "Set at least one key."
print("Groq:  ", "set" if GROQ_KEY else "not set")
print("OpenAI:", "set" if OPENAI_KEY else "not set")

import csv, io, json, re, urllib.request

CSV_URL = ("https://raw.githubusercontent.com/GTB-TME/gtbreport2024/"
           "main/data/ps/prev_notif_2023-07-03.csv")

STOCK_CUES = ["prevalent", "prevalence", "living with", "at the time of",
              "existing", "point prevalence"]
FLOW_CUES  = ["notified", "notification", "reported during", "new cases",
              "newly", "incidence", "during"]
ALL_CUES   = sorted(set(STOCK_CUES + FLOW_CUES), key=len, reverse=True)

def cues_in(s):
    low = s.lower()
    return [c for c in ALL_CUES if c in low]

raw = urllib.request.urlopen(CSV_URL).read().decode("utf-8")
recs = []
for r in csv.DictReader(io.StringIO(raw)):
    try:
        pr, pop, nt = float(r["prev"]), float(r["pop"]), float(r["notified"])
    except (ValueError, KeyError):
        continue
    stock = pr / 1e5 * pop
    if stock > 0 and nt > 0:
        recs.append({"country": r["country"].split("(")[0].strip(),
                     "year": r["year"],
                     "stock": round(stock), "flow": round(nt)})
print(f"{len(recs)} country-surveys loaded")

fmt = lambda n: f"{n:,}"

def evidence(rec):
    return (f"A national TB prevalence survey in {rec['country']} estimated "
            f"{fmt(rec['stock'])} prevalent cases of TB in {rec['year']}. "
            f"{fmt(rec['flow'])} TB cases were notified to national "
            f"authorities in {rec['country']} during {rec['year']}.")

ITEMS = []
for rec in recs:
    c, y = rec["country"], rec["year"]
    trio = [
        ("cued_stock", f"How many prevalent cases of TB were estimated in {c} in {y}?", "STOCK"),
        ("cued_flow",  f"How many TB cases were notified in {c} during {y}?",           "FLOW"),
        ("nocue",      f"How many TB cases does the report give for {c} in {y}?",       None),
    ]
    for cond, q, gold in trio:
        leaks = cues_in(q)
        if cond == "nocue" and leaks:
            print(f"  [drop] nocue for {c} {y} leaks {leaks}")
            continue
        if cond.startswith("cued") and not leaks:
            print(f"  [warn] {cond} for {c} {y} carries no cue")
        ITEMS.append({
            "id": f"{c[:12].replace(' ','_')}_{y}_{cond}",
            "condition": cond, "question": q, "gold_type": gold,
            "cues": leaks, "evidence": evidence(rec),
            "stock": rec["stock"], "flow": rec["flow"],
        })

ITEMS = ITEMS[:N_ITEMS]
from collections import Counter
print(f"\n{len(ITEMS)} questions: {dict(Counter(i['condition'] for i in ITEMS))}")
print("\nexample nocue item:")
ex = [i for i in ITEMS if i["condition"] == "nocue"][0]
print("  Q:", ex["question"])
print("  E:", ex["evidence"][:120], "...")
print("  stock =", fmt(ex["stock"]), "| flow =", fmt(ex["flow"]))

import time
from openai import OpenAI

GEN_PROMPT = ("Answer the question using only the evidence provided. "
              "Give one short sentence.\n\n"
              "EVIDENCE:\n{evidence}\n\nQUESTION:\n{question}\n")

def client_for(model):
    if "oss" in model:
        return OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")
    return OpenAI(api_key=OPENAI_KEY)

def generate(model, item, temperature=0.0):
    cl = client_for(model)
    kw = dict(model=model, temperature=temperature, max_tokens=200,
              messages=[{"role": "user",
                         "content": GEN_PROMPT.format(**item)}])
    if "oss" in model:
        kw["reasoning_effort"] = "low"
    r = cl.chat.completions.create(**kw)
    m = r.choices[0].message
    return ((m.content or "").strip() or
            (getattr(m, "reasoning", None) or "").strip())

def numbers_in(text):
    return [int(x.replace(",", "").replace(" ", ""))
            for x in re.findall(r"\d[\d, ]{2,}\d", text)]

def type_from_number(ans, item):
    nums = numbers_in(ans)
    hit = {"STOCK": item["stock"] in nums, "FLOW": item["flow"] in nums}
    if hit["STOCK"] and not hit["FLOW"]: return "STOCK"
    if hit["FLOW"] and not hit["STOCK"]: return "FLOW"
    if hit["STOCK"] and hit["FLOW"]:     return "BOTH"
    return "NEITHER"

def type_from_words(ans):
    s = cues_in(ans)
    st = [c for c in s if c in STOCK_CUES]
    fl = [c for c in s if c in FLOW_CUES]
    if st and not fl: return "STOCK"
    if fl and not st: return "FLOW"
    if st and fl:     return "BOTH"
    return "UNKNOWN"

MODEL = "gpt-4o-mini" if OPENAI_KEY else "openai/gpt-oss-20b"
item = [i for i in ITEMS if i["condition"] == "nocue"][0]

print(f"PILOT  model={MODEL}")
print(f"  item     : {item['id']}")
print(f"  question : {item['question']}")
print(f"  stock    : {fmt(item['stock'])}")
print(f"  flow     : {fmt(item['flow'])}\n")

ans = generate(MODEL, item)
print(f"  raw answer: {ans!r}\n")

checks = []
checks.append(("API returned text", bool(ans)))
nums = numbers_in(ans)
checks.append(("answer contains a number", bool(nums)))
tn = type_from_number(ans, item)
checks.append(("number matches stock or flow", tn in ("STOCK", "FLOW")))
tw = type_from_words(ans)
checks.append(("word classifier returns a type", tw in ("STOCK", "FLOW")))
checks.append(("number and words agree", tn == tw))

print(f"  numbers found      : {nums}")
print(f"  type (from number) : {tn}")
print(f"  type (from wording): {tw}\n")
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

hard = [n for n, ok in checks[:3] if not ok]
if hard:
    raise SystemExit(f"\nSTOP - fix before the full run: {hard}")
if not checks[4][1]:
    print("\n  NOTE: number and wording disagree on this item. Not fatal - the"
          "\n  full run records both and the analysis reports disagreement.")
print("\nPilot OK. Proceed to cell 5.")

RESULTS = []
MODELS = ([("gpt-4o-mini", OPENAI_KEY)] if OPENAI_KEY else []) + \
         ([("openai/gpt-oss-20b", GROQ_KEY)] if GROQ_KEY else [])

for model, _ in MODELS:
    for run_i in range(1, N_RUNS + 1):
        print(f"\n=== {model}  run {run_i}/{N_RUNS} ===")
        for n, it in enumerate(ITEMS, 1):
            try:
                a = generate(model, it, temperature=0.7)
            except Exception as e:
                print(f"  [{n:3d}] ERROR {str(e)[:60]}"); time.sleep(10); continue
            tn, tw = type_from_number(a, it), type_from_words(a)
            RESULTS.append({"model": model, "run": run_i, "id": it["id"],
                            "condition": it["condition"],
                            "gold_type": it["gold_type"],
                            "type_number": tn, "type_words": tw,
                            "answer": a})
            ok = "" if it["gold_type"] is None else \
                 (" OK" if tn == it["gold_type"] else " <-- TYPE ERROR")
            print(f"  [{n:3d}] {it['condition']:10s} num={tn:7s} words={tw:7s}{ok}")
            time.sleep(0.4)

json.dump(RESULTS, open("generator_results.json", "w"), indent=1)
print(f"\n{len(RESULTS)} generations saved to generator_results.json")

from collections import defaultdict
from math import comb

def fisher(a, b, c, d):
    obs = comb(a+b, a) * comb(c+d, c) / comb(a+b+c+d, a+c); tot = 0.0
    for i in range(0, min(a+b, a+c) + 1):
        j, k = a+b-i, a+c-i; l = c+d-k
        if j < 0 or k < 0 or l < 0: continue
        p = comb(a+b, i) * comb(c+d, k) / comb(a+b+c+d, a+c)
        if p <= obs + 1e-12: tot += p
    return tot

for model, _ in MODELS:
    rs = [r for r in RESULTS if r["model"] == model]
    if not rs: continue
    print(f"\n===== {model} =====")

    print("\n-- 1. TYPE-RESPONSIVENESS (cued items, gold known)")
    tot = defaultdict(lambda: [0, 0])
    for r in rs:
        if r["gold_type"]:
            tot[r["condition"]][1] += 1
            tot[r["condition"]][0] += int(r["type_number"] == r["gold_type"])
    for cond, (ok, n) in sorted(tot.items()):
        print(f"   {cond:12s} {ok}/{n} correct = {ok/n if n else 0:.3f}")

    print("\n-- 2. TYPE-STABILITY across runs (all items, no gold needed)")
    by = defaultdict(list)
    for r in rs: by[r["id"]].append(r["type_number"])
    st = defaultdict(lambda: [0, 0])
    for iid, types in by.items():
        cond = next(r["condition"] for r in rs if r["id"] == iid)
        st[cond][1] += 1
        st[cond][0] += int(len(set(types)) == 1)
    for cond, (same, n) in sorted(st.items()):
        print(f"   {cond:12s} {same}/{n} stable = {same/n if n else 0:.3f}")

    print("\n-- 3. WHICH TYPE DOES IT PICK WHEN THE QUESTION IS AMBIGUOUS?")
    pick = defaultdict(int)
    for r in rs:
        if r["condition"] == "nocue": pick[r["type_number"]] += 1
    for t, c in sorted(pick.items(), key=lambda x: -x[1]):
        print(f"   {t:8s} {c}")

    print("\n-- 4. CUED vs NOCUE instability (Fisher)")
    cu = [v for k, v in st.items() if k.startswith("cued")]
    a = sum(n - s for s, n in cu); b = sum(s for s, n in cu)
    c_, d_ = (st["nocue"][1] - st["nocue"][0], st["nocue"][0]) if "nocue" in st else (0, 0)
    print(f"   cued unstable {a}/{a+b} | nocue unstable {c_}/{c_+d_}")
    if (a+b) and (c_+d_):
        print(f"   Fisher exact p = {fisher(c_, d_, a, b):.4f}")

print("\nDone. Copy everything above and send it.")