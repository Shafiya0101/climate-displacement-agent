"""Task 3 (headline): faithfulness vs answer_relevancy on substitution errors.

For each framed near-miss (a SUBSTITUTION error: true sentence, wrong tau_q),
in the PARAGRAPH condition, compute BOTH:
  - faithfulness    (question-blind; expected to MISS -> score high)
  - answer_relevancy(question-aware; professor predicts it ALSO misses)

If BOTH miss the substitution error, that's the paper's headline: the
question-aware metric is fooled too, because the two questions sit close in
embedding space.

Uses GROQ_API_KEY from .env. Run from repo root:
  python S1_probe/s3b_relevancy.py --model openai/gpt-oss-20b
Out: s3b_results.jsonl
"""
import argparse, json, re, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reasoning import chat
from src.guardrails import TokenBudget

HERE = Path(__file__).resolve().parent
items = json.loads((HERE/"probe_v3.json").read_text(encoding="utf-8"))

def sent(c):
    t=c["temporal"]
    if t=="FLOW":  return f"{c['value']} new displacements were recorded during 2023 ({c['label']})."
    if t=="STOCK": return f"{c['value']} people were living in displacement at the end of 2023 ({c['label']})."
    return f"{c['value']} people are projected to be displaced by 2050 ({c['label']})."

def paragraph(it):
    return "\n".join("- "+sent(c) for c in (it["gold"],it["near_miss"],it["far_miss"]))

FAITH = ('Evaluate FAITHFULNESS: are the ANSWER\'s claims supported by the EVIDENCE? '
         'Score 0.0-1.0. Reply ONLY: {"faithfulness":0.0}')
RELEV = ('Evaluate ANSWER_RELEVANCY: does the ANSWER address the QUESTION that was asked? '
         'Ignore whether it is factually supported; judge only whether it answers THIS question. '
         'Score 0.0-1.0. Reply ONLY: {"answer_relevancy":0.0}')

def extract(txt,key):
    txt=re.sub(r"<think>.*?</think>"," ",txt or "",flags=re.DOTALL)
    for b in reversed(re.findall(r"\{[^{}]*\}",txt)):
        try:
            d=json.loads(b)
            if key in d: return max(0.0,min(1.0,float(d[key])))
        except: pass
    return None

def score(sys_p, user, key, model, budget):
    for _ in range(3):
        try:
            m=chat([{"role":"system","content":sys_p},{"role":"user","content":user}],
                model=model,temperature=0.0,budget=budget,span_name="s3b",max_tokens=250)
            v=extract(m.content,key)
            if v is not None: return v
        except Exception as e:
            if "per day" in str(e).lower(): raise
            time.sleep(10)
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="openai/gpt-oss-20b")
    a=ap.parse_args()
    budget=TokenBudget(max_usd=25.0)
    out=HERE/"s3b_results.jsonl"
    fa,re_=[],[]
    with open(out,"a",encoding="utf-8") as fh:
        for it in items:
            ev=paragraph(it)
            ans=it["answers"]["near_framed"]   # the substitution error
            q=it["question"]
            try:
                f=score(FAITH,f"EVIDENCE:\n{ev}\n\nANSWER:\n{ans}","faithfulness",a.model,budget)
                r=score(RELEV,f"QUESTION: {q}\n\nANSWER:\n{ans}","answer_relevancy",a.model,budget)
            except Exception:
                print("quota hit, stopping"); break
            if f is not None: fa.append(f)
            if r is not None: re_.append(r)
            fh.write(json.dumps({"id":it["id"],"tau_q":it["tau_q"],
                "faithfulness":f,"answer_relevancy":r})+"\n"); fh.flush()
            print(f"  {it['id']}: faith={f}  relevancy={r}  (${budget.spent_usd:.3f})")
    import statistics as st
    if fa and re_:
        print(f"\n=== HEADLINE: metrics on {len(fa)} SUBSTITUTION errors (framed near-miss) ===")
        print(f"  mean faithfulness    = {st.mean(fa):.3f}  (high = metric MISSES the error)")
        print(f"  mean answer_relevancy= {st.mean(re_):.3f}  (high = ALSO misses it)")
        if st.mean(fa)>0.6 and st.mean(re_)>0.6:
            print("\n  *** BOTH metrics miss the substitution error. This is the headline. ***")
        elif st.mean(re_)<0.4:
            print("\n  answer_relevancy CATCHES it -> the question-aware metric is the fix.")

if __name__=="__main__":
    main()
