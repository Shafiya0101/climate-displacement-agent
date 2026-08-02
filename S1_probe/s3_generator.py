"""S3 (headline): does the GENERATOR commit type errors?

For each probe item we build a realistic retrieved context containing the gold
figure AND both distractor figures (all real, all co-occurring, as they do in
the report). We ask the model the question and classify its answer:

  correct     -> stated the gold value
  type_error  -> stated a distractor value that IS in context but wrong TYPE
                 (value-faithful, type-unfaithful -> invisible to citation check)
  value_error -> stated a value NOT in the provided context (classic hallucination)
  other       -> no clear numeric answer

Run from repo root (needs src.reasoning.chat):
  python S1_probe/s3_generator.py --model openai/gpt-oss-20b --reps 1
Out: s3_results.jsonl
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

SYS = ("You answer questions about the IDMC displacement report using ONLY the "
       "CONTEXT provided. Give the single most accurate figure. Answer in one "
       "short sentence stating the number.")

def val_num(v):  # "7.7 million" -> 7.7
    m = re.search(r"[\d.]+", v)
    return float(m.group()) if m else None

def build_context(it):
    # gold + both distractors, each as a real typed sentence, shuffled order
    lines = []
    for c in (it["gold"], it["near_miss"], it["far_miss"]):
        t = c["temporal"]
        if t == "FLOW":
            lines.append(f"- {c['value']} new displacements were recorded during 2023 ({c['label']}).")
        elif t == "STOCK":
            lines.append(f"- {c['value']} people were living in displacement at the end of 2023 ({c['label']}).")
        else:
            lines.append(f"- {c['value']} people are projected to be displaced by 2050 ({c['label']}).")
    return "\n".join(lines)

def classify(ans, it):
    nums = re.findall(r"\d+\.?\d*", ans)
    stated = [float(n) for n in nums if re.search(r"\d", n)]
    if not stated:
        return "other"
    g = val_num(it["gold"]["value"])
    n = val_num(it["near_miss"]["value"])
    f = val_num(it["far_miss"]["value"])
    def close(x, y): return y is not None and abs(x - y) < 0.05 * max(y, 1)
    if any(close(s, g) for s in stated):
        return "correct"
    if any(close(s, n) for s in stated) or any(close(s, f) for s in stated):
        return "type_error"   # gave a distractor that's IN context but wrong type
    return "value_error"       # gave something not in the provided context

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    budget = TokenBudget(max_usd=25.0)
    out = HERE / "s3_results.jsonl"
    counts = {"correct":0,"type_error":0,"value_error":0,"other":0}
    with open(out, "a", encoding="utf-8") as fh:
        for it in items:
            ctx = build_context(it)
            for rep in range(a.reps):
                try:
                    msg = chat([{"role":"system","content":SYS},
                        {"role":"user","content":f"CONTEXT:\n{ctx}\n\nQUESTION: {it['question']}"}],
                        model=a.model, temperature=0.0, budget=budget,
                        span_name="s3_gen", max_tokens=120)
                    ans = msg.content.strip()
                except Exception as e:
                    s=str(e).lower()
                    if "per day" in s or "tpd" in s:
                        print("STOP: daily quota exhausted"); 
                        print(counts); return
                    print(f"  ERROR on {it["id"]}: {type(e).__name__}: {str(e)[:200]}")
                    time.sleep(3); continue
                cls = classify(ans, it)
                counts[cls]+=1
                fh.write(json.dumps({"id":it["id"],"model":a.model,"rep":rep,
                    "class":cls,"answer":ans[:200]})+"\n"); fh.flush()
                print(f"  {it['id']}: {cls}")
    n=sum(counts.values())
    print(f"\n=== S3 results (n={n}) ===")
    for k,v in counts.items():
        print(f"  {k:12s} {v:3d}  ({v/n*100:.1f}%)")
    print(f"\nTYPE-ERROR RATE (the headline): {counts['type_error']/n*100:.1f}%")
    print(f"VALUE-ERROR RATE:               {counts['value_error']/n*100:.1f}%")

if __name__=="__main__":
    main()
