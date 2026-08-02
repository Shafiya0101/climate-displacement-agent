"""S2: judge-side experiment on the 4x3 design (professor's Table 1).

For each probe item, score the 6 answer versions (correct/near/far x bare/framed)
with a faithfulness judge, under two evidence conditions:
  SENTENCE  : judge sees only the gold evidence sentence
  PARAGRAPH : judge sees a passage containing gold + both distractor figures

Produces the 4 (evidence x camouflage) x 3 (value) cells with bootstrap 95% CIs.
The diagnostic cell is PARAGRAPH x FRAMED.

Run from repo root:
  python S1_probe/s2_judge.py --judges openai/gpt-oss-20b llama-3.1-8b-instant --reps 1
Out: s2_results.jsonl
"""
import argparse, json, re, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reasoning import chat
from src.guardrails import TokenBudget

HERE = Path(__file__).resolve().parent
items = json.loads((HERE / "probe_v2_clean.json").read_text(encoding="utf-8"))

JUDGE = ("You are evaluating the FAITHFULNESS of an ANSWER against EVIDENCE.\n"
         "Faithfulness = are the claims in the ANSWER supported by the EVIDENCE?\n"
         "Score 0.0 (unsupported) to 1.0 (fully supported).\n"
         'Reply with ONLY this JSON: {"faithfulness": 0.0}')

def sent(claim):
    t = claim["temporal"]
    if t == "FLOW":  return f"{claim['value']} new displacements were recorded during 2023 ({claim['label']})."
    if t == "STOCK": return f"{claim['value']} people were living in displacement at the end of 2023 ({claim['label']})."
    return f"{claim['value']} people are projected to be displaced by 2050 ({claim['label']})."

def evidence(it, cond):
    if cond == "SENTENCE":
        return sent(it["gold"])
    return "\n".join("- " + sent(c) for c in (it["gold"], it["near_miss"], it["far_miss"]))

def extract(txt):
    txt = re.sub(r"<think>.*?</think>", " ", txt or "", flags=re.DOTALL)
    for b in reversed(re.findall(r"\{[^{}]*\}", txt)):
        try:
            d = json.loads(b)
            if "faithfulness" in d: return max(0.0, min(1.0, float(d["faithfulness"])))
        except: pass
    return None

def score(ev, ans, judge, budget):
    for _ in range(3):
        try:
            m = chat([{"role":"system","content":JUDGE},
                {"role":"user","content":f"EVIDENCE:\n{ev}\n\nANSWER:\n{ans}"}],
                model=judge, temperature=0.0, budget=budget,
                span_name="s2_judge", max_tokens=300)
            v = extract(m.content)
            if v is not None: return v
        except Exception as e:
            if "per day" in str(e).lower(): raise
            time.sleep(10)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    budget = TokenBudget(max_usd=25.0)
    out = HERE / "s2_results.jsonl"
    n_cells = len(items)*len(a.judges)*a.reps*2*3
    print(f"scoring up to {n_cells} judge calls...")
    done=0
    with open(out, "a", encoding="utf-8") as fh:
        for judge in a.judges:
            for it in items:
                for cond in ("SENTENCE","PARAGRAPH"):
                    ev = evidence(it, cond)
                    for value in ("correct","near","far"):
                        for gamma in ("bare","framed"):
                            ans = it["answers"][f"{value}_{gamma}"]
                            for rep in range(a.reps):
                                try: v = score(ev, ans, judge, budget)
                                except Exception: print("  quota hit, stopping"); print(f"wrote {done}"); return
                                if v is None: continue
                                fh.write(json.dumps({"id":it["id"],"judge":judge,
                                    "evidence":cond,"value":value,"gamma":gamma,
                                    "rep":rep,"faith":v})+"\n"); fh.flush()
                                done+=1
                print(f"  {judge.split('/')[-1]} {it['id']} done (${budget.spent_usd:.3f})")
    print(f"\nwrote {done} judge scores to {out}")
    print("Next: python S1_probe/s2_analyse.py")

if __name__=="__main__":
    main()
