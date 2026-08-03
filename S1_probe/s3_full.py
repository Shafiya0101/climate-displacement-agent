"""S3 (full retrieval, per professor's decision).

Runs the real agent over the full report for each question. Because full
retrieval surfaces whatever figures the report contains (not necessarily the
probe's designed distractors), we classify by TYPE-correctness relative to the
question's demanded type tau_q, using the answer's CONCLUSION:

  correct            -> answer states a figure whose predication type == tau_q
  substitution_error -> answer states a figure of a DIFFERENT type than tau_q,
                        with internally-consistent predication (true sentence,
                        wrong type for the question) -> faithfulness-invisible
  attribution_error  -> answer's predication is internally inconsistent
                        (mixed type wording) -> faithfulness CAN catch
  abstain            -> answer says the figure is not in context (good behaviour)
  other              -> no clear numeric claim

Run from repo root:
  python S1_probe/s3_full.py --topk 3 --reps 1
Out: s3_full_results.jsonl
"""
import argparse, json, re, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval import hybrid_retrieve, format_context
from src.reasoning import self_consistency_synthesis, TokenBudget

HERE=Path(__file__).resolve().parent
items=json.loads((HERE/"probe_v3.json").read_text(encoding="utf-8"))

FLOW=["new displacement","recorded during","triggered","movements","during 2023","displacements"]
STOCK=["living in displacement","were living","remained displaced","at the end of","idps","end of 2023","as of 31 december"]
PROJ=["projected","by 2050","projection"]
ABSTAIN=["not contain","no figure","does not provide","not present","cannot","no evidence","not available","not specify"]

def conclusion(ans):
    # take the CONCLUSION section if present, else whole answer
    m=re.search(r"CONCLUSION(.*?)(CONFIDENCE|$)",ans,re.DOTALL|re.IGNORECASE)
    return m.group(1) if m else ans

def detect_type(text):
    t=text.lower()
    if any(c in t for c in PROJ):  return "PROJ"
    # count stock vs flow cues; internal inconsistency if BOTH strongly present
    s=sum(c in t for c in STOCK); f=sum(c in t for c in FLOW)
    if s and f and abs(s-f)<=0: return "MIXED"
    if s>f: return "STOCK"
    if f>s: return "FLOW"
    return "UNKNOWN"

def classify(ans, it):
    concl=conclusion(ans)
    low=ans.lower()
    if any(c in low for c in ABSTAIN) and not re.search(r"\d+\.?\d*\s*(million|m\b)",concl.lower()):
        return "abstain"
    if not re.search(r"\d",concl): return "other"
    atype=detect_type(concl)
    tau=it["tau_q"]
    if atype=="MIXED": return "attribution_error"
    if atype=="UNKNOWN": return "other"
    if atype==tau: return "correct"
    return "substitution_error"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reps",type=int,default=1)
    ap.add_argument("--topk",type=int,default=3)
    a=ap.parse_args()
    budget=TokenBudget(max_usd=25.0)
    out=HERE/"s3_full_results.jsonl"
    counts={"correct":0,"substitution_error":0,"attribution_error":0,"abstain":0,"other":0}
    n=0
    with open(out,"a",encoding="utf-8") as fh:
        for it in items:
            try:
                hits=hybrid_retrieve(it["question"],top_k=a.topk)
                ctx=format_context(hits)
                res=self_consistency_synthesis(it["question"],ctx,budget,k=3)
                ans=res["answer"] if isinstance(res,dict) else str(res)
            except Exception as e:
                s=str(e).lower()
                if "per day" in s or "tpd" in s or "quota" in s or "rate" in s:
                    print("  quota hit — partial saved"); summ(counts,n); return
                if "413" in s or "too large" in s:
                    print(f"  {it['id']}: SKIP (too large)"); continue
                print(f"  ERR {it['id']}: {str(e)[:80]}"); time.sleep(3); continue
            cls=classify(ans,it); counts[cls]+=1; n+=1
            fh.write(json.dumps({"id":it["id"],"tau_q":it["tau_q"],"class":cls,
                "answer":ans[:500]})+"\n"); fh.flush()
            print(f"  {it['id']} [{it['tau_q']}]: {cls}  (${budget.spent_usd:.3f})")
    summ(counts,n)

def summ(counts,n):
    print(f"\n=== S3 FULL results (n={n}) ===")
    if n==0: print("none"); return
    for k,v in counts.items(): print(f"  {k:20s} {v:3d}  ({v/n*100:.1f}%)")
    scored=counts['correct']+counts['substitution_error']+counts['attribution_error']
    if scored:
        print(f"\n  Of {scored} numeric answers:")
        print(f"  SUBSTITUTION error rate: {counts['substitution_error']/scored*100:.1f}%  (the headline)")
        print(f"  ATTRIBUTION error rate:  {counts['attribution_error']/scored*100:.1f}%")

if __name__=="__main__":
    main()
