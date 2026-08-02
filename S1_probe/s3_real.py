"""S3 REAL (headline) — splits the two error classes per the professor's framework.

For each probe question, uses the agent's OWN pipeline:
  1. hybrid_retrieve() over the full report -> real messy context
  2. self_consistency_synthesis() -> the agent's real answer
Then classifies the stated figure into the professor's classes:

  correct              -> stated the gold value (right value, right type)
  attribution_error    -> stated a distractor value with the WRONG predication
                          for that value (internally inconsistent; a citation /
                          faithfulness check CAN catch this)
  substitution_error   -> stated a distractor value with predication matching
                          THAT value's own type, but the wrong type for the
                          QUESTION (tau_q). Internally consistent, cite-able,
                          faithfulness-invisible. THIS is the paper's target.
  value_error          -> stated a value not in the report (hallucination)
  other                -> no clear numeric answer

Run from repo root (needs quota):
  python S1_probe/s3_real.py --reps 1
Out: s3_real_results.jsonl
"""
import argparse, json, re, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval import hybrid_retrieve, format_context
from src.reasoning import self_consistency_synthesis, TokenBudget

HERE = Path(__file__).resolve().parent
items = json.loads((HERE/"probe_v3.json").read_text(encoding="utf-8"))

FLOW_CUES  = ["new displacement","recorded during","triggered","movements","during 2023"]
STOCK_CUES = ["living in displacement","were living","at the end of","remained displaced","idps","end 2023","end of 2023"]
PROJ_CUES  = ["projected","by 2050","projection"]

def detect_type(text):
    t=text.lower()
    if any(c in t for c in PROJ_CUES):  return "PROJ"
    if any(c in t for c in STOCK_CUES): return "STOCK"
    if any(c in t for c in FLOW_CUES):  return "FLOW"
    return "UNKNOWN"

def valnum(v):
    m=re.search(r"\d+\.?\d*",v); return float(m.group()) if m else None

def classify(answer, it):
    """Classify the agent's answer into the professor's error taxonomy."""
    nums=[float(n) for n in re.findall(r"\d+\.?\d*",answer) if re.search(r"\d",n)]
    if not nums: return "other"
    ans_type = detect_type(answer)
    tau_q = it["tau_q"]                      # type the QUESTION demands
    g=valnum(it["gold"]["value"])
    for dist in ("near_miss","far_miss"):
        dv=valnum(it[dist]["value"]); dtype=it[dist]["temporal"]
        if any(abs(s-dv)<0.03*max(dv,1) for s in nums):
            # answer used a distractor VALUE. Which error class?
            if ans_type==dtype and ans_type!=tau_q:
                return "substitution_error"  # predication matches value's type, wrong for question
            elif ans_type==tau_q:
                return "attribution_error"   # question-type wording on wrong value (inconsistent)
            else:
                return "attribution_error"
    if any(abs(s-g)<0.03*max(g,1) for s in nums):
        return "correct"
    return "value_error"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reps",type=int,default=1)
    ap.add_argument("--topk",type=int,default=5)
    a=ap.parse_args()
    budget=TokenBudget(max_usd=25.0)
    out=HERE/"s3_real_results.jsonl"
    counts={"correct":0,"attribution_error":0,"substitution_error":0,"value_error":0,"other":0}
    n=0
    with open(out,"a",encoding="utf-8") as fh:
        for it in items:
            for rep in range(a.reps):
                try:
                    hits=hybrid_retrieve(it["question"],top_k=a.topk)
                    ctx=format_context(hits)
                    res=self_consistency_synthesis(it["question"],ctx,budget,k=3)
                    ans=res["answer"] if isinstance(res,dict) else str(res)
                except Exception as e:
                    s=str(e).lower()
                    if "per day" in s or "tpd" in s or "quota" in s:
                        print("  quota hit — partial saved"); summarise(counts,n); return
                    print(f"  ERROR {it['id']}: {type(e).__name__}: {str(e)[:120]}")
                    time.sleep(3); continue
                cls=classify(ans,it); counts[cls]+=1; n+=1
                fh.write(json.dumps({"id":it["id"],"tau_q":it["tau_q"],
                    "class":cls,"answer":ans[:400]})+"\n"); fh.flush()
                print(f"  {it['id']} [{it['tau_q']}]: {cls}  (${budget.spent_usd:.3f})")
    summarise(counts,n)

def summarise(counts,n):
    print(f"\n=== S3 results (n={n}) — two error classes separated ===")
    if n==0: print("  none completed"); return
    for k,v in counts.items():
        print(f"  {k:20s} {v:3d}  ({v/n*100:.1f}%)")
    print(f"\n  ATTRIBUTION error rate: {counts['attribution_error']/n*100:.1f}%  (faithfulness CAN catch)")
    print(f"  SUBSTITUTION error rate: {counts['substitution_error']/n*100:.1f}%  (faithfulness BLIND — the headline)")

if __name__=="__main__":
    main()
