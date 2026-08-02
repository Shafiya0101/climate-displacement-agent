"""S3 REAL (headline): does the actual AGENT commit type errors?

For each probe question, this uses the agent's OWN pipeline:
  1. hybrid_retrieve() over the full 358-chunk report -> real messy context
     (distractor figures co-occur naturally, as in the report)
  2. self_consistency_synthesis() -> the agent's real answer
Then classifies the stated figure as correct / type_error / value_error.

This is the fair test the professor specified (retrieval over the full report).

Run from repo root:
  python S1_probe/s3_real.py --reps 1
Out: s3_real_results.jsonl  (saves as it goes; safe to hit quota)
"""
import argparse, json, re, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval import hybrid_retrieve, format_context
from src.reasoning import self_consistency_synthesis, TokenBudget

HERE = Path(__file__).resolve().parent
items = json.loads((HERE / "probe_v2_clean.json").read_text(encoding="utf-8"))

def valnum(v):
    m = re.search(r"\d+\.?\d*", v)
    return float(m.group()) if m else None

def classify(answer_text, it):
    # pull the conclusion's numbers
    nums = [float(n) for n in re.findall(r"\d+\.?\d*", answer_text) if re.search(r"\d", n)]
    if not nums:
        return "other"
    g = valnum(it["gold"]["value"]); n = valnum(it["near_miss"]["value"]); f = valnum(it["far_miss"]["value"])
    def close(x, y): return y is not None and abs(x - y) < 0.05 * max(y, 1)
    if any(close(s, g) for s in nums):
        return "correct"
    if any(close(s, n) for s in nums) or any(close(s, f) for s in nums):
        return "type_error"     # stated a real distractor figure (in corpus, wrong type)
    return "value_error"         # stated something not among the probe figures

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()
    budget = TokenBudget(max_usd=25.0)
    out = HERE / "s3_real_results.jsonl"
    counts = {"correct":0,"type_error":0,"value_error":0,"other":0}
    n=0
    with open(out, "a", encoding="utf-8") as fh:
        for it in items:
            for rep in range(a.reps):
                try:
                    hits = hybrid_retrieve(it["question"], top_k=a.topk)
                    ctx = format_context(hits)
                    res = self_consistency_synthesis(it["question"], ctx, budget, k=3)
                    ans = res["answer"] if isinstance(res, dict) else str(res)
                except Exception as e:
                    s=str(e).lower()
                    if "per day" in s or "tpd" in s or "quota" in s:
                        print("  quota hit — stopping (partial results saved)")
                        summarise(counts, n); return
                    print(f"  ERROR {it['id']}: {type(e).__name__}: {str(e)[:150]}")
                    time.sleep(3); continue
                cls = classify(ans, it)
                counts[cls]+=1; n+=1
                fh.write(json.dumps({"id":it["id"],"rep":rep,"class":cls,
                    "answer":ans[:400]})+"\n"); fh.flush()
                print(f"  {it['id']}: {cls}  (${budget.spent_usd:.3f})")
    summarise(counts, n)

def summarise(counts, n):
    print(f"\n=== S3 REAL results (n={n}) ===")
    if n==0:
        print("  no items completed"); return
    for k,v in counts.items():
        print(f"  {k:12s} {v:3d}  ({v/n*100:.1f}%)")
    print(f"\nTYPE-ERROR RATE (headline): {counts['type_error']/n*100:.1f}%")
    print(f"VALUE-ERROR RATE:           {counts['value_error']/n*100:.1f}%")

if __name__=="__main__":
    main()
