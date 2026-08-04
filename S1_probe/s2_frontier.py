"""Frontier judge for the S2 table, using OpenAI (gpt-4o-mini by default).

Re-scores the 5 classes x 2 metrics with a frontier judge, so the paper can
report a strong-judge column beside gpt-oss-20b. If faithfulness stays 1.000 on
substitution here too, the 'weak judge' objection is answered.

SECURITY: reads OPENAI_API_KEY from .env. Do NOT hardcode the key.
COST: gpt-4o-mini is cheap (~$0.15/1M input tokens). ~570 calls ~= a few cents.

Run from repo root:
  python S1_probe/s2_frontier.py --model gpt-4o-mini
Out: s2_frontier_results.jsonl
"""
import argparse, json, re, sys, time, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    print("Need the openai package: pip install openai"); sys.exit(1)

key=os.getenv("OPENAI_API_KEY")
if not key:
    print("OPENAI_API_KEY not found in .env. Add a line: OPENAI_API_KEY=sk-...")
    sys.exit(1)
client=OpenAI(api_key=key)

HERE=Path(__file__).resolve().parent
items=json.loads((HERE/"probe_v3.json").read_text(encoding="utf-8"))
RESULTS=HERE/"s2_frontier_results.jsonl"

def sent(c):
    t=c["temporal"]
    if t=="FLOW":  return f"{c['value']} new displacements were recorded during 2023 ({c['label']})."
    if t=="STOCK": return f"{c['value']} people were living in displacement at the end of 2023 ({c['label']})."
    return f"{c['value']} people are projected to be displaced by 2050 ({c['label']})."

def paragraph(it):
    return "\n".join("- "+sent(c) for c in (it["gold"],it["near_miss"],it["far_miss"]))

FAITH=('Evaluate FAITHFULNESS: are the ANSWER\'s claims supported by the EVIDENCE? '
       'Score 0.0-1.0. Reply ONLY with JSON: {"faithfulness":0.0}')
RELEV=('Evaluate ANSWER_RELEVANCY: does the ANSWER address the QUESTION asked? '
       'Ignore factual support; judge only whether it answers THIS question. '
       'Score 0.0-1.0. Reply ONLY with JSON: {"answer_relevancy":0.0}')

def extract(txt,key):
    for b in reversed(re.findall(r"\{[^{}]*\}",txt or "")):
        try:
            d=json.loads(b)
            if key in d: return max(0.0,min(1.0,float(d[key])))
        except: pass
    return None

def score(model, sysp, user, key):
    for _ in range(3):
        try:
            r=client.chat.completions.create(model=model,temperature=0.0,max_tokens=60,
                messages=[{"role":"system","content":sysp},{"role":"user","content":user}])
            v=extract(r.choices[0].message.content,key)
            if v is not None: return v
        except Exception as e:
            print(f"    api err: {str(e)[:80]}"); time.sleep(5)
    return None

ROWS=[("correct","correct_bare"),("ATT_near","near_bare"),("ATT_far","far_bare"),
      ("SUB_near","near_framed"),("SUB_far","far_framed")]

def load_done():
    if not RESULTS.exists(): return set()
    return {(json.loads(l)["id"],json.loads(l)["row"]) for l in RESULTS.read_text().splitlines() if l.strip()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="gpt-4o-mini")
    ap.add_argument("--limit",type=int,default=57)
    a=ap.parse_args()
    done=load_done()
    print(f"  judge={a.model}, {len(done)} cells already done")
    with open(RESULTS,"a",encoding="utf-8") as fh:
        for it in items[:a.limit]:
            ev=paragraph(it); q=it["question"]
            for rowname,akey in ROWS:
                if (it["id"],rowname) in done: continue
                ans=it["answers"][akey]
                f=score(a.model,FAITH,f"EVIDENCE:\n{ev}\n\nANSWER:\n{ans}","faithfulness")
                r=score(a.model,RELEV,f"QUESTION: {q}\n\nANSWER:\n{ans}","answer_relevancy")
                fh.write(json.dumps({"id":it["id"],"row":rowname,"faith":f,"rel":r})+"\n"); fh.flush()
            print(f"  {it['id']} done")
    # summary
    import statistics as st
    from collections import defaultdict
    faith=defaultdict(list); rel=defaultdict(list)
    for l in RESULTS.read_text().splitlines():
        if not l.strip(): continue
        d=json.loads(l)
        if d["faith"] is not None: faith[d["row"]].append(d["faith"])
        if d["rel"] is not None: rel[d["row"]].append(d["rel"])
    print(f"\n=== FRONTIER JUDGE ({a.model}) TABLE ===")
    print(f"{'Class':10s} {'faith':>8s} {'ans_rel':>8s}")
    for r in ["correct","ATT_near","ATT_far","SUB_near","SUB_far"]:
        fm=f"{st.mean(faith[r]):.3f}" if faith[r] else "--"
        rm=f"{st.mean(rel[r]):.3f}" if rel[r] else "--"
        print(f"{r:10s} {fm:>8s} {rm:>8s}  (n={len(faith[r])})")

if __name__=="__main__":
    main()
