"""Re-score saved eval rows with a configurable judge model."""
import json, os, re, sys, time, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.guardrails import TokenBudget
from src.reasoning import chat

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")
KEYS = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
JUDGE = """You are a strict RAG evaluator. Score each metric 0.0 to 1.0.

context_recall     : fraction of the claims in REFERENCE supported by CONTEXT.
context_precision  : fraction of the CONTEXT passages relevant to the question.
faithfulness       : fraction of the claims in ANSWER entailed by CONTEXT.
answer_relevancy   : how directly ANSWER addresses QUESTION, ignoring correctness.

Reply with ONLY this JSON and nothing else, no explanation:
{"context_recall":0.0,"context_precision":0.0,"faithfulness":0.0,"answer_relevancy":0.0}"""


class DailyLimit(Exception):
    pass


def extract(text):
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL)
    blobs = re.findall(r"\{[^{}]*\}", text)
    for b in reversed(blobs):
        try:
            d = json.loads(b)
            if any(k in d for k in KEYS):
                return d
        except Exception:
            continue
    return None


def judge_one(row, budget, attempts=3):
    ctx = "\n\n".join(f"[P{j}] {c[:6000]}"
                      for j, c in enumerate(row["retrieved_contexts"], 1))
    for a in range(attempts):
        try:
            msg = chat([{"role": "system", "content": JUDGE},
                        {"role": "user", "content":
                         f"QUESTION:\n{row['user_input']}\n\nREFERENCE:\n{row['reference']}\n\n"
                         f"CONTEXT:\n{ctx}\n\nANSWER:\n{row['response'][:8000]}"}],
                       model=JUDGE_MODEL, temperature=0.0, budget=budget,
                       span_name="rescore_judge", max_tokens=1500)
            d = extract(msg.content)
            if d:
                return d
            print(f"    no JSON (reply started: {(msg.content or '')[:60]!r})")
        except Exception as e:
            s = str(e)
            if "per day" in s or "TPD" in s:
                raise DailyLimit(f"{JUDGE_MODEL} daily quota exhausted")
            wait = 20 * (a + 1)
            print(f"    retry {a+1}/{attempts} in {wait}s ({type(e).__name__})")
            time.sleep(wait)
    return {}


def score(rows):
    budget, totals = TokenBudget(max_usd=10.0), {k: [] for k in KEYS}
    for i, row in enumerate(rows, 1):
        d = judge_one(row, budget)
        got = [k for k in KEYS if isinstance(d.get(k), (int, float))]
        for k in got:
            totals[k].append(max(0.0, min(1.0, float(d[k]))))
        print(f"  {i}/{len(rows)}  {'ok' if got else 'FAILED'}")
        time.sleep(2)
    return {"engine": f"llm_judge:{JUDGE_MODEL}",
            **{k: (round(sum(v)/len(v), 3) if v else None) for k, v in totals.items()},
            "scored_n": {k: len(v) for k, v in totals.items()}}


for name in (sys.argv[1:] or ["baseline", "final"]):
    p = HERE / f"results_{name}.json"
    if not p.exists():
        print(f"skip {name}"); continue
    blob = json.loads(p.read_text(encoding="utf-8"))
    print(f"--- {name}: {len(blob['rows'])} questions, judge={JUDGE_MODEL} ---")
    try:
        blob["scores_judge"] = score(blob["rows"])
    except DailyLimit as e:
        print(f"\nSTOPPED: {e}\nTry: $env:JUDGE_MODEL=\"openai/gpt-oss-20b\"")
        break
    p.write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {blob['scores_judge']}\n")
