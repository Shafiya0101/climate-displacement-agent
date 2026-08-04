"""S3 cheap: single direct call per question (no 3-path self-consistency).

Same test as full S3 -- does the generator commit substitution errors when the
correct figure AND a commensurable distractor are both in the retrieved
context -- but ~10x cheaper, so it fits in the daily token budget.

For each item we build the PARAGRAPH context (gold + near + far, all real) and
ask the model the question in ONE call. Classify by type vs tau_q.

Resumes automatically; skips done items. Run repeatedly until all 57 done:
  python S1_probe/s3_cheap.py --limit 57
Out: s3_cheap_results.jsonl
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
RESULTS=HERE/"s3_cheap_results.jsonl"

def sent(c):
    t=c["temporal"]
    if t=="FLOW":  return f"{c['value']} new displacements were recorded during 2023 ({c['label']})."
    if t=="STOCK": return f"{c['value']} people were living in displacement at the end of 2023 ({c['label']})."
    return f"{c['value']} people are projected to be displaced by 2050 ({c['label']})."

def paragraph(it):
    return "\n".join("- "+sent(c) for c in (it["gold"],it["near_miss"],it["far_miss"]))

SYS=("Answer the QUESTION using ONLY the CONTEXT. Give the single best figure in "
     "one short sentence, stating the number and what it measures.")

FLOW=["new displacement","recorded during","triggered","movements","during 2023","displacements"]
STOCK=["living in displacement","were living","remained displaced","at the end of","idps","end of 2023"]
PROJ=["projected","by 2050"]
ABSTAIN=["not contain","no figure","does not","not present","cannot","no evidence","not available"]

def detect_type(t):
    t=t.lower()
    if any(c in t for c in PROJ): return "PROJ"
    s=sum(c in t for c in STOCK); f=sum(c in t for c in FLOW)
    if s and f and s==f: return "MIXED"
    if s>f: return "STOCK"
    if f>s: return "FLOW"
    return "UNKNOWN"

def classify(ans, it):
    low=ans.lower()
    if any(c in low for c in ABSTAIN) and not re.search(r"\d",ans): return "abstain"
    if not re.search(r"\d",ans): return "other"
    at=detect_type(ans); tau=it["tau_q"]
    if at=="MIXED": return "attribution_error"
    if at=="UNKNOWN": return "other"
    if at==tau: return "correct"
    return "substitution_error"

def load_done():
    if not RESULTS.exists(): return set()
    return {json.loads(l)["id"] for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=57)
    ap.add_argument("--model",default="openai/gpt-oss-20b")
    a=ap.parse_args()
    budget=TokenBudget(max_usd=25.0)
    done=load_done()
    todo=[it for it in items if it["id"] not in done][:a.limit]
    print(f"  ({len(done)} done, {len(todo)} to do)")
    with open(RESULTS,"a",encoding="utf-8") as fh:
        for it in todo:
            ev=paragraph(it)
            try:
                ans=""
                for _try in range(3):
                    m=chat([{"role":"system","content":SYS},
                        {"role":"user","content":f"CONTEXT:\n{ev}\n\nQUESTION: {it['question']}"}],
                        model=a.model,temperature=0.0,budget=budget,span_name="s3cheap",max_tokens=150)
                    ans=(m.content or "").strip()
                    if ans: break
                    time.sleep(1)
            except Exception as e:
                if any(x in str(e).lower() for x in ["per day","tpd","quota","rate","429"]):
                    print("  quota hit — partial saved"); break
                print(f"  ERR {it['id']}: {str(e)[:70]}"); time.sleep(3); continue
            cls=classify(ans,it)
            fh.write(json.dumps({"id":it["id"],"tau_q":it["tau_q"],"class":cls,"answer":ans[:200]})+"\n"); fh.flush()
            print(f"  {it['id']} [{it['tau_q']}]: {cls}  (${budget.spent_usd:.3f})")
    # accumulated summary
    from collections import Counter
    rows=[json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()] if RESULTS.exists() else []
    c=Counter(r["class"] for r in rows); n=len(rows)
    print(f"\n=== S3 ACCUMULATED (n={n}) ===")
    for k in ["correct","substitution_error","attribution_error","abstain","other"]:
        print(f"  {k:20s} {c.get(k,0):3d}  ({c.get(k,0)/n*100:.1f}%)" if n else k)
    scored=c.get("correct",0)+c.get("substitution_error",0)+c.get("attribution_error",0)
    if scored:
        print(f"\n  SUBSTITUTION rate: {c.get('substitution_error',0)/scored*100:.1f}%  (of {scored} numeric answers)")
        print(f"  ATTRIBUTION rate:  {c.get('attribution_error',0)/scored*100:.1f}%")

if __name__=="__main__":
    main()
