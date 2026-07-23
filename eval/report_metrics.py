"""Produces the cost / latency / tool-distribution numbers for REPORT.md §3.

    python eval/report_metrics.py --runs 10        # execute 10 runs then report
    python eval/report_metrics.py --report-only    # report from data/runs.jsonl

Also runs one deliberate budget-exhaustion run so you can document that the
TokenBudget actually triggered (rubric G asks for this explicitly).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.agent import run  # noqa: E402

QUESTIONS = [q["question"] for q in
             json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))]


def execute(n: int) -> None:
    for i, q in enumerate(QUESTIONS[:n], 1):
        print(f"\n########## RUN {i}/{n} ##########\n{q}")
        r = run(q)
        print(f"-> verdict={r['critic']['verdict']} cost=${r['cost']['usd']:.5f} "
              f"latency={r['latency_s']}s tools={r['tool_calls']}")


def budget_demo() -> None:
    """Force the budget to trigger, with the cap temporarily set very low."""
    original = config.MAX_USD
    config.MAX_USD = 0.0005
    print("\n########## BUDGET TRIGGER DEMO (max_usd=$0.0005) ##########")
    try:
        r = run(QUESTIONS[0])
        print(f"status={r['status']} budget_triggered={r['cost']['budget_triggered']}")
        print(r["answer"][:300])
    finally:
        config.MAX_USD = original


def report() -> None:
    if not config.RUNS_FILE.exists():
        print("No runs logged yet. Run with --runs 10 first.")
        return
    rows = [json.loads(l) for l in config.RUNS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print("No successful runs logged.")
        return
    costs = [r["cost"]["usd"] for r in ok]
    lat = [r["latency_s"] for r in ok]
    tools: Counter = Counter()
    for r in ok:
        tools.update(r.get("tool_calls", {}))
    verdicts = Counter(r.get("verdict") for r in ok)

    print("\n" + "=" * 60)
    print(f"runs analysed        : {len(ok)}")
    print(f"average cost / run   : ${statistics.mean(costs):.5f}  "
          f"(min ${min(costs):.5f} / max ${max(costs):.5f})")
    print(f"average latency      : {statistics.mean(lat):.1f} s  "
          f"(median {statistics.median(lat):.1f} s)")
    print(f"total spend          : ${sum(costs):.4f}")
    print(f"critic verdicts      : {dict(verdicts)}")
    print("\ntool call distribution (paste into REPORT.md):")
    print("| Tool | Calls | Calls per run |")
    print("|------|-------|---------------|")
    for name, c in tools.most_common():
        print(f"| {name} | {c} | {c/len(ok):.1f} |")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=0)
    ap.add_argument("--budget-demo", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.runs:
        execute(a.runs)
    if a.budget_demo:
        budget_demo()
    report()
