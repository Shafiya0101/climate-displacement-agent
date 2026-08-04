"""Fill Table 2 (the S2 judge table): 5 classes x 2 metrics, paragraph evidence.

Rows: correct, ATT-near, ATT-far, SUB-near, SUB-far
Cols: faithfulness, answer_relevancy
All under PARAGRAPH evidence (the design condition).

Uses probe_v3.json. Run from repo root:
  python S1_probe/s2_table.py --model openai/gpt-oss-20b
Out: s2_table_results.jsonl
"""
import argparse, json, re, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reasoning import chat
from src.guardrails import TokenBudget

HERE=Path(__file__).resolve().parent
items=json.loads((HERE/"probe_v3.json").read_text(encoding="utf-8"))

def sent(c):
    t=c["temporal"]
    if t=="FLOW":  return f"{c['value']} new displacements were recorded during 2023 ({c['label']})."
    if t=="STOCK": return f"{c['value']} people were living in displacement at the end of 2023 ({c['label']})."
    return f"{c['value']} people are projected to be displaced by 2050 ({c['label']})."

def paragraph(it):
    return "\n".join("- "+sent(c) for c in (it["gold"],it["near_miss"],it["far_miss"]))

FAITH=('Evaluate FAITHFULNESS: are the ANSWER\'s claims supported by the EVIDENCE? '
       'Score 0.0-1.0. Reply ONLY: {"faithfulness":0.0}')
RELEV=('Evaluate ANSWER_RELEVANCY: does the ANSWER address the QUESTION asked? '
       'Ignore factual support; judge only whether it answers THIS question. '
       'Score 0.0-1.0. Reply ONLY: {"answer_relevancy":0.0}')

def extract(txt,key):
    txt=re.sub(r"<think>.*?</think>"," ",txt or "",flags=re.DOTALL)
    for b in reversed(re.findall(r"\{[^{}]*\}",txt)):
        try:
            d=json.loads(b)
            if key in d: return max(0.0,min(1.0,float(d[key])))
        except: pass
    return None

def score(sysp,user,key,model,budget):
    for _ in range(3):
        try:
            m=chat([{"role":"system","content":sysp},{"role":"user","content":user}],
                model=model,temperature=0.0,budget=budget,span_name="s2tab",max_tokens=250)
            v=extract(m.content,key)
            if v is not None: return v
        except Exception as e:
            if "per day" in str(e).lower(): raise
            time.sleep(8)
    return None

# the five answer versions -> row keys
ROWS=[("correct","correct_bare",None),
      ("ATT_near","near_bare","near"),
      ("ATT_far","far_bare","far"),
      ("SUB_near","near_framed","near"),
      ("SUB_far","far_framed","far")]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="openai/gpt-oss-20b")
    a=ap.parse_args()
    budget=TokenBudget(max_usd=25.0)
    out=HERE/"s2_table_results.jsonl"
    agg={r[0]:{"faith":[],"rel":[]} for r in ROWS}
    with open(out,"a",encoding="utf-8") as fh:
        for it in items:
            ev=paragraph(it); q=it["question"]
            for rowname,akey,_ in ROWS:
                ans=it["answers"][akey]
                try:
                    f=score(FAITH,f"EVIDENCE:\n{ev}\n\nANSWER:\n{ans}","faithfulness",a.model,budget)
                    r=score(RELEV,f"QUESTION: {q}\n\nANSWER:\n{ans}","answer_relevancy",a.model,budget)
                except Exception:
                    print("quota hit — partial saved"); dump(agg); return
                if f is not None: agg[rowname]["faith"].append(f)
                if r is not None: agg[rowname]["rel"].append(r)
                fh.write(json.dumps({"id":it["id"],"row":rowname,"faith":f,"rel":r})+"\n"); fh.flush()
            print(f"  {it['id']} done (${budget.spent_usd:.3f})")
    dump(agg)

def dump(agg):
    import statistics as st
    print("\n=== TABLE 2 (paragraph evidence) ===")
    print(f"{'Class':10s} {'faithf.':>10s} {'ans_rel.':>10s}")
    for r in ["correct","ATT_near","ATT_far","SUB_near","SUB_far"]:
        fa=agg[r]["faith"]; re_=agg[r]["rel"]
        fm=f"{st.mean(fa):.3f}" if fa else "--"
        rm=f"{st.mean(re_):.3f}" if re_ else "--"
        print(f"{r:10s} {fm:>10s} {rm:>10s}  (n={len(fa)})")

if __name__=="__main__":
    main()
