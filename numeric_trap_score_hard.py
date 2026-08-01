"""Numeric-trap experiment - HARD condition.

Same probe set, but the judge sees a realistic paragraph where BOTH the correct
figure and the trap figure appear in context. The judge must reason about which
meaning the question asks for, instead of matching the handed-over correct answer.

This mirrors the condition where the agent originally confused stock vs flow:
reasoning over a messy passage, not scoring against a clean evidence sentence.

    python numeric_trap_score_hard.py --judges openai/gpt-oss-20b llama-3.1-8b-instant --reps 3

Writes: numeric_trap_results_hard.jsonl
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.reasoning import chat
from src.guardrails import TokenBudget

JUDGE_PROMPT = """You are evaluating the FAITHFULNESS of an ANSWER against EVIDENCE.
Faithfulness = are the claims in the ANSWER supported by the EVIDENCE?
Score 0.0 (unsupported) to 1.0 (fully supported).
Reply with ONLY this JSON: {"faithfulness": 0.0}"""

# A single realistic paragraph containing ALL the key 2023 figures, so every
# question's correct AND trap number appear together. The judge cannot pattern-
# match; it must pick the figure that matches the question's concept.
CONTEXT = ("In 2023, disasters triggered 26.4 million new internal displacements "
"(movements) across 148 countries, while 7.7 million people were still living in "
"displacement due to disasters at year-end. Conflict and violence triggered 20.5 "
"million new displacements during the year, and 68.3 million people remained "
"displaced by conflict at the end of 2023. In total, 46.9 million new displacements "
"were recorded across all causes during 2023, and 75.9 million people were living "
"in internal displacement globally at year-end. Among disasters, floods caused 9.8 "
"million displacements and storms 9.5 million, while earthquakes triggered 6.1 "
"million - of which the Turkiye-Syria earthquakes alone accounted for 4.7 million.")

def build_answer(row, version):
    num = {"correct":row["correct_number"],"near":row["near_miss"],"far":row["far_miss"]}[version]
    q = row["question"].rstrip("?")
    return f"Based on the report, the answer to '{q}' is {num}."

def extract(text):
    text = re.sub(r"<think>.*?</think>"," ",text or "",flags=re.DOTALL)
    for blob in reversed(re.findall(r"\{[^{}]*\}",text)):
        try:
            d=json.loads(blob)
            if "faithfulness" in d: return max(0.0,min(1.0,float(d["faithfulness"])))
        except: continue
    return None

def score(row,version,judge,budget):
    ans=build_answer(row,version)
    for attempt in range(3):
        try:
            msg=chat([{"role":"system","content":JUDGE_PROMPT},
                {"role":"user","content":f"EVIDENCE:\n{CONTEXT}\n\nANSWER:\n{ans}"}],
                model=judge,temperature=0.0,budget=budget,
                span_name="numeric_trap_hard",max_tokens=400)
            v=extract(msg.content)
            if v is not None: return v
        except Exception as e:
            s=str(e).lower()
            if "per day" in s or "tpd" in s: raise RuntimeError(f"quota exhausted {judge}")
            time.sleep(15*(attempt+1))
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--judges",nargs="+",required=True)
    ap.add_argument("--reps",type=int,default=3)
    ap.add_argument("--probe",default="probe_set.json")
    a=ap.parse_args()
    rows=json.loads(Path(a.probe).read_text(encoding="utf-8"))
    print(f"HARD condition: {len(rows)} rows x 3 x {len(a.judges)} judges x {a.reps} reps")
    budget=TokenBudget(max_usd=25.0)
    out=Path("numeric_trap_results_hard.jsonl")
    with open(out,"a",encoding="utf-8") as f:
        for judge in a.judges:
            for row in rows:
                for version in ("correct","near","far"):
                    for rep in range(a.reps):
                        try: v=score(row,version,judge,budget)
                        except RuntimeError as e: print(f"  STOP: {e}"); return
                        f.write(json.dumps({"id":row["id"],"judge":judge,
                            "version":version,"rep":rep,"faithfulness":v})+"\n")
                        f.flush()
                    time.sleep(0.5)
            print(f"  done judge={judge} (${budget.spent_usd:.3f})")
    print(f"written to {out}\nNext: python numeric_trap_analyse_hard.py")

if __name__=="__main__": main()
