"""
STEP 4 - Run the two judges over the TB probe.

Uses the SAME prompts as the August cycle so the two domains are comparable.
Resumable: saves after every call, skips anything already done.

Settings per judge:
    gpt-4o-mini          max_tokens=8
    openai/gpt-oss-20b   max_tokens=512, reasoning_effort="low"
                         (reasoning model: returns empty content under a
                         small max_tokens; verdict = last 0/1 digit emitted)

Usage:
    export OPENAI_API_KEY=...
    export GROQ_API_KEY=...
    python run_judges.py --probe probe_tb.json --judge gpt-4o-mini
    python run_judges.py --probe probe_tb.json --judge openai/gpt-oss-20b

Output: judge_<model>.jsonl   one line per (item, row):
    {"id":..., "row":"SUB_near", "faith":1.0, "rel":0.0}
"""
import argparse, json, os, re, sys, time
from pathlib import Path

FAITH_PROMPT = (
    "Evaluate FAITHFULNESS: are the ANSWER's claims supported by the EVIDENCE?\n"
    "Reply with only 1 (supported) or 0 (not supported).\n\n"
    "EVIDENCE:\n{evidence}\n\nANSWER:\n{answer}\n"
)
REL_PROMPT = (
    "Evaluate ANSWER_RELEVANCY: does the ANSWER address the QUESTION asked?\n"
    "Reply with only 1 (addresses it) or 0 (does not).\n\n"
    "QUESTION:\n{question}\n\nANSWER:\n{answer}\n"
)
ROWS = ["correct", "ATT_near", "ATT_far", "SUB_near", "SUB_far"]


def is_reasoning(model):
    return model.startswith("openai/") or "oss" in model


def make_client(model):
    if is_reasoning(model):
        from openai import OpenAI
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            sys.exit("set GROQ_API_KEY")
        return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("set OPENAI_API_KEY")
    return OpenAI(api_key=key)


def parse_verdict(txt, reasoning):
    txt = (txt or "").strip()
    if reasoning:
        # reasoning model: take the last 0/1 digit emitted
        digits = re.findall(r"[01]", txt)
        return float(digits[-1]) if digits else None
    if "1" in txt and "0" not in txt:
        return 1.0
    if "0" in txt and "1" not in txt:
        return 0.0
    return 1.0 if txt.startswith("1") else 0.0 if txt.startswith("0") else None


def ask(client, model, prompt, retries=3):
    reasoning = is_reasoning(model)
    kwargs = dict(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    if reasoning:
        kwargs["max_tokens"] = 512
        kwargs["extra_body"] = {"reasoning_effort": "low"}
    else:
        kwargs["max_tokens"] = 8
    for a in range(1, retries + 1):
        try:
            r = client.chat.completions.create(**kwargs)
            return parse_verdict(r.choices[0].message.content, reasoning)
        except Exception as e:
            print(f"    retry {a}/{retries}: {e}")
            time.sleep(20 * a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="probe_tb.json")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    out = Path(f"judge_{args.judge.replace('/', '_')}.jsonl")

    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["id"], r["row"]))
    print(f"{len(done)} verdicts already saved")

    client = make_client(args.judge)
    n_new = 0
    with out.open("a", encoding="utf-8") as fh:
        for it in probe:
            for row in ROWS:
                if (it["id"], row) in done:
                    continue
                ans = it["answers"][row]
                faith = ask(client, args.judge,
                            FAITH_PROMPT.format(evidence=it["evidence"], answer=ans))
                rel = ask(client, args.judge,
                          REL_PROMPT.format(question=it["question"], answer=ans))
                if faith is None or rel is None:
                    print("  unparseable - skipping (counts as dropout)")
                    continue
                fh.write(json.dumps({"id": it["id"], "row": row,
                                     "faith": faith, "rel": rel}) + "\n")
                fh.flush()
                n_new += 1
                print(f"  {it['id']:6s} {row:9s} faith={faith} rel={rel}")
                time.sleep(args.sleep)

    print(f"\n{n_new} new verdicts -> {out}")
    print("NEXT: run the other judge, then analyze.py")


if __name__ == "__main__":
    main()
