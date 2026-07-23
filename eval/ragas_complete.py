"""Complete the two ragas metrics that NaN'd out under concurrent rate limiting.

The first attempt failed because ragas dispatches every (question x metric) job
in parallel; under a free-tier limiter the queued jobs exceeded their timeout and
returned NaN. This runs them strictly sequentially with a long timeout and
aggressive retry, and evaluates only the two reference-based metrics that are
still missing.

    python eval/ragas_complete.py baseline
    python eval/ragas_complete.py final
"""
import json, os, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src import config

JUDGE = os.getenv("RAGAS_MODEL", "openai/gpt-oss-safeguard-20b")

from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
try:
    from ragas.run_config import RunConfig
except ImportError:
    from ragas import RunConfig

name = (sys.argv[1:] or ["baseline"])[0]
p = HERE / f"results_{name}.json"
blob = json.loads(p.read_text(encoding="utf-8"))
rows = blob["rows"]
print(f"--- ragas {name}: {len(rows)} questions, judge={JUDGE} ---")
print("Sequential (max_workers=1). Expect 15-30 minutes.")

llm = LangchainLLMWrapper(ChatGroq(model=JUDGE, api_key=config.GROQ_API_KEY,
                                   temperature=0, max_retries=5))
rc = RunConfig(max_workers=1, timeout=600, max_retries=15, max_wait=120)

res = evaluate(dataset=EvaluationDataset.from_list(rows),
               metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()],
               llm=llm, run_config=rc, raise_exceptions=False)

df = res.to_pandas()
out = {"engine": f"ragas:{JUDGE}"}
for want, cols in {
        "context_recall": ["context_recall"],
        "context_precision": ["llm_context_precision_with_reference",
                              "context_precision"]}.items():
    for c in cols:
        if c in df.columns:
            series = df[c]
            out[want] = round(float(series.mean(skipna=True)), 3)
            out[want + "_n"] = int(series.notna().sum())
            break

blob["scores_ragas_reference"] = out
p.write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {out}")
print(f"  scored {out.get('context_recall_n', 0)}/{len(rows)} questions")
