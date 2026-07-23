"""Regenerate eval rows only — no scoring. Scoring is done separately by
ragas_complete.py (sequential) and rescore.py (independent judge), because
ragas' default parallel dispatch times out under free-tier rate limits."""
import json, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_ragas as rr

for name in (sys.argv[1:] or ["baseline", "final"]):
    print(f"=== generating {name}: {rr.CONFIGS[name]} ===")
    rows = rr.generate(name, 3, 10)
    p = HERE / f"results_{name}.json"
    p.write_text(json.dumps({"config": rr.CONFIGS[name], "rows": rows,
                             "corpus": "141 parents / 676 children"},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  saved {p.name}\n")
