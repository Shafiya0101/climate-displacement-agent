"""Numeric-trap experiment.

Takes YOUR hand-built probe_set.json, builds three answer-versions per row
(correct / near_miss / far_miss), and scores each with a faithfulness-style LLM
judge. The question the paper asks:

    Does the judge penalise the near_miss (wrong meaning, close number)
    as much as the far_miss (obviously wrong number)?

If near_miss scores as high as correct while far_miss scores low, the metric has
a numeric blind spot -> the finding holds.

Run from inside the climate-displacement-agent repo (needs src.reasoning.chat):

    python numeric_trap_score.py --judges openai/gpt-oss-20b llama-3.3-70b-versatile --reps 3

Reads:  probe_set.json   (that YOU built by hand)
Writes: numeric_trap_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from src.reasoning import chat
    from src.guardrails import TokenBudget
except Exception as e:
    print("Run from the climate-displacement-agent repo root.")
    print(f"(import error: {e})")
    sys.exit(1)

# A deliberately standard faithfulness judge prompt — we are testing the STANDARD
# metric, so we must not hand-tune it to catch numeric errors.
JUDGE_PROMPT = """You are evaluating the FAITHFULNESS of an ANSWER against EVIDENCE.
Faithfulness = are the claims in the ANSWER supported by the EVIDENCE?
Score 0.0 (unsupported) to 1.0 (fully supported).
Reply with ONLY this JSON: {"faithfulness": 0.0}"""


def build_answer(row: dict, version: str) -> str:
    """Construct a one-sentence answer using the correct / near / far number."""
    num = {"correct": row["correct_number"],
           "near": row["near_miss"],
           "far": row["far_miss"]}[version]
    # Same sentence frame for all three; only the number changes. This isolates
    # the number as the sole variable.
    q = row["question"].rstrip("?")
    return f"Based on the report, the answer to '{q}' is {num}."


def extract(text: str) -> float | None:
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL)
    for blob in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            d = json.loads(blob)
            if "faithfulness" in d:
                return max(0.0, min(1.0, float(d["faithfulness"])))
        except Exception:
            continue
    return None


def score(row: dict, version: str, judge: str, budget: TokenBudget) -> float | None:
    ans = build_answer(row, version)
    for attempt in range(3):
        try:
            msg = chat(
                [{"role": "system", "content": JUDGE_PROMPT},
                 {"role": "user", "content":
                  f"EVIDENCE:\n{row['evidence']}\n\nANSWER:\n{ans}"}],
                model=judge, temperature=0.0, budget=budget,
                span_name="numeric_trap_judge", max_tokens=400)
            v = extract(msg.content)
            if v is not None:
                return v
        except Exception as e:
            s = str(e).lower()
            if "per day" in s or "tpd" in s:
                raise RuntimeError(f"daily quota exhausted on {judge}")
            time.sleep(15 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--probe", default="probe_set.json")
    a = ap.parse_args()

    rows = json.loads(Path(a.probe).read_text(encoding="utf-8"))
    print(f"{len(rows)} probe rows x 3 versions x {len(a.judges)} judges "
          f"x {a.reps} reps = {len(rows)*3*len(a.judges)*a.reps} judge calls\n")

    budget = TokenBudget(max_usd=25.0)
    out = Path("numeric_trap_results.jsonl")
    with open(out, "a", encoding="utf-8") as f:
        for judge in a.judges:
            for row in rows:
                for version in ("correct", "near", "far"):
                    for rep in range(a.reps):
                        try:
                            v = score(row, version, judge, budget)
                        except RuntimeError as e:
                            print(f"  STOP: {e}")
                            return
                        rec = {"id": row["id"], "judge": judge,
                               "version": version, "rep": rep, "faithfulness": v}
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                    time.sleep(0.5)
            print(f"  done judge={judge}  (${budget.spent_usd:.3f})")

    print(f"\nwritten to {out}\nNext: python numeric_trap_analyse.py")


if __name__ == "__main__":
    main()
