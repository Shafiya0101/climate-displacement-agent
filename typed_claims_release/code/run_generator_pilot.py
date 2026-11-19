"""Surface-ambiguous pilot: does the real agent commit substitution errors
when the question does not leak the demanded type (FLOW / STOCK / PROJ)?

Saves progress after every single question, and skips questions it already
has an answer for — so if the daily quota runs out partway through, you can
just wait and run this exact same command again later; it will pick up
where it left off instead of starting over.
"""
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import run_ragas as rr  # noqa: E402

AMBIGUOUS_ITEMS = json.loads(
    (HERE / "questions_ambiguous.json").read_text(encoding="utf-8")
)

OUT_PATH = HERE / "pilot_ambiguous_results.json"

CUES = {
    "FLOW": ["new displacement", "recorded", "triggered", " new "],
    "STOCK": ["remained", "living in displacement", "at the end of", "at year-end"],
    "PROJ": ["projected", "could reach", "by 2050", re.compile(r"\bby 20\d\d\b")],
}


def guess_type(text: str) -> str:
    text_low = text.lower()
    scores = {}
    for t, cues in CUES.items():
        n = 0
        for cue in cues:
            if hasattr(cue, "search"):
                n += 1 if cue.search(text_low) else 0
            else:
                n += 1 if cue in text_low else 0
        scores[t] = n
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "UNKNOWN"


def load_existing():
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    return []


def save(results):
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_one_with_retry(item, max_retries=3):
    rr.QUESTIONS = [{"question": item["question"], "reference": item["reference"]}]
    for attempt in range(1, max_retries + 1):
        try:
            rows = rr.generate("final", k_sc=3, limit=1)
            return rows[0]
        except Exception as e:
            print(f"  [rate limit or error] attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                wait = 30 * attempt
                print(f"  waiting {wait} seconds before retrying...")
                time.sleep(wait)
    return None  # give up on this one for now, but don't crash


def main():
    results = load_existing()
    done_questions = {r["question"] for r in results}
    remaining = [it for it in AMBIGUOUS_ITEMS if it["question"] not in done_questions]

    print(f"{len(results)} already done, {len(remaining)} remaining.\n")
    if not remaining:
        print("All done! See summary below.")
    for idx, item in enumerate(remaining, 1):
        print(f"--- {idx}/{len(remaining)}: {item['question']}")
        row = generate_one_with_retry(item)

        if row is None:
            print("  Could not get an answer right now (quota likely still exhausted).")
            print("  Stopping here. Your progress so far is saved.")
            print("  Wait a while, then run this exact same command again.")
            break

        detected = guess_type(row["response"])
        gold = item["gold_type"]
        is_sub = detected != "UNKNOWN" and detected != gold

        results.append({
            "question": item["question"],
            "gold_type": gold,
            "detected_type": detected,
            "is_substitution_error": is_sub,
            "agent_answer": row["response"],
            "note": item.get("note", ""),
        })
        save(results)  # save immediately, every time

        verdict = "SUBSTITUTION ERROR" if is_sub else (
            "unclear - check by hand" if detected == "UNKNOWN" else "ok"
        )
        print(f"  [{verdict}] gold={gold} detected={detected}")
        print(f"  A: {row['response'][:200]}{'...' if len(row['response']) > 200 else ''}\n")

        if idx < len(remaining):
            print("  pausing 20s before next question...\n")
            time.sleep(20)

    n = len(results)
    n_sub = sum(r["is_substitution_error"] for r in results)
    n_unknown = sum(r["detected_type"] == "UNKNOWN" for r in results)

    print("=" * 60)
    print(f"Items done so far:       {n} / {len(AMBIGUOUS_ITEMS)}")
    print(f"Substitution errors:     {n_sub}")
    print(f"Unclear (check by hand): {n_unknown}")
    if n:
        print(f"Substitution rate:       {n_sub}/{n} = {n_sub/n:.1%}")
    print(f"Saved to:                {OUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()