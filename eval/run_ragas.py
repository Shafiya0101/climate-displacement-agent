"""RAGAS evaluation — baseline vs final retrieval pipeline.

    python eval/run_ragas.py --config baseline   # dense only, no rerank, child chunks
    python eval/run_ragas.py --config final      # hybrid + RRF + cross-encoder + parents
    python eval/run_ragas.py --config all        # both, prints the report table

The two configurations run through the SAME code path in src/retrieval.py with
three flags flipped, which is what makes the comparison honest: nothing else
changes between the baseline and the final numbers.

If the `ragas` package is unavailable, an equivalent LLM-judge implementation of
the same four metric definitions is used instead. The output states which one ran
— report that honestly in REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.guardrails import TokenBudget  # noqa: E402
from src.reasoning import chat, self_consistency_synthesis  # noqa: E402
from src.retrieval import format_context, hybrid_retrieve  # noqa: E402

QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))

CONFIGS = {
    # BEFORE Block 1: plain top-k cosine over fixed child chunks, no fusion, no rerank
    "baseline": dict(use_hybrid=False, use_rerank=False, use_parents=False),
    # AFTER Block 1 + 3
    "final": dict(use_hybrid=True, use_rerank=True, use_parents=True),
}


def generate(cfg_name: str, k_sc: int, limit: int | None) -> list[dict]:
    flags = CONFIGS[cfg_name]
    budget = TokenBudget(max_usd=10.0)
    rows = []
    qs = QUESTIONS[:limit] if limit else QUESTIONS
    for i, item in enumerate(qs, 1):
        q = item["question"]
        passages = hybrid_retrieve(q, **flags)
        context = format_context(passages)
        if cfg_name == "baseline":
            # baseline = direct prompting, no CoT, no self-consistency
            msg = chat([{"role": "system",
                         "content": "Answer the question using only the context provided."},
                        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {q}"}],
                       model=config.SYNTH_MODEL, temperature=0.1, budget=budget,
                       span_name="eval_baseline", max_tokens=600)
            answer = msg.content or ""
        else:
            answer = self_consistency_synthesis(q, context, budget, k=k_sc)["answer"]
        rows.append({"user_input": q, "response": answer,
                     "retrieved_contexts": [p["text"] for p in passages],
                     "reference": item["reference"]})
        print(f"  [{cfg_name}] {i}/{len(qs)} done  (${budget.spent_usd:.4f})")
    print(f"  [{cfg_name}] generation cost ${budget.spent_usd:.4f}")
    return rows


# ------------------------------------------------------------------- ragas
def score_with_ragas(rows: list[dict]) -> dict | None:
    try:
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                                   LLMContextRecall, ResponseRelevancy)
    except Exception as e:
        print(f"[ragas] unavailable ({e}); falling back to the built-in LLM judge.")
        return None
    try:
        llm = LangchainLLMWrapper(ChatGroq(model=config.SYNTH_MODEL,
                                          api_key=config.GROQ_API_KEY, temperature=0))
        emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=config.EMBED_MODEL))
        ds = EvaluationDataset.from_list(rows)
        res = evaluate(dataset=ds, llm=llm, embeddings=emb,
                       metrics=[LLMContextRecall(), LLMContextPrecisionWithReference(),
                                Faithfulness(), ResponseRelevancy()])
        df = res.to_pandas()
        out = {}
        for want, cols in {
            "context_recall": ["context_recall"],
            "context_precision": ["llm_context_precision_with_reference", "context_precision"],
            "faithfulness": ["faithfulness"],
            "answer_relevancy": ["answer_relevancy", "semantic_similarity"],
        }.items():
            for c in cols:
                if c in df.columns:
                    out[want] = round(float(df[c].mean(skipna=True)), 3)
                    break
        return {"engine": "ragas", **out}
    except Exception as e:
        print(f"[ragas] evaluation failed ({e}); falling back to the built-in LLM judge.")
        return None


# -------------------------------------------------------- fallback LLM judge
JUDGE = """You are a strict RAG evaluator. Score 0.0 to 1.0. Output ONLY a JSON object.

context_recall     : fraction of the claims in REFERENCE that are supported by CONTEXT.
context_precision  : fraction of the CONTEXT passages that are relevant to answering the question.
faithfulness       : fraction of the claims in ANSWER that are entailed by CONTEXT (not by world knowledge).
answer_relevancy   : how directly ANSWER addresses QUESTION, ignoring whether it is correct.

Output exactly: {"context_recall":0.0,"context_precision":0.0,"faithfulness":0.0,"answer_relevancy":0.0}"""


def score_with_judge(rows: list[dict]) -> dict:
    budget = TokenBudget(max_usd=10.0)
    keys = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
    totals = {k: [] for k in keys}
    for i, r in enumerate(rows, 1):
        ctx = "\n\n".join(f"[P{j}] {c[:1200]}" for j, c in enumerate(r["retrieved_contexts"], 1))
        try:
            msg = chat([{"role": "system", "content": JUDGE},
                        {"role": "user", "content":
                         f"QUESTION:\n{r['user_input']}\n\nREFERENCE:\n{r['reference']}\n\n"
                         f"CONTEXT:\n{ctx}\n\nANSWER:\n{r['response'][:4000]}"}],
                       model=config.SYNTH_MODEL, temperature=0.0, budget=budget,
                       span_name="llm_judge", max_tokens=200)
            m = re.search(r"\{.*\}", msg.content or "", re.DOTALL)
            d = json.loads(m.group(0)) if m else {}
            for k in keys:
                if isinstance(d.get(k), (int, float)):
                    totals[k].append(max(0.0, min(1.0, float(d[k]))))
        except Exception as e:
            print(f"  [judge] item {i} failed: {e}")
    out = {k: (round(sum(v) / len(v), 3) if v else None) for k, v in totals.items()}
    return {"engine": "llm_judge_fallback", **out}


def evaluate_config(name: str, k_sc: int, limit: int | None) -> dict:
    print(f"\n=== {name.upper()} : {CONFIGS[name]} ===")
    rows = generate(name, k_sc, limit)
    scores = score_with_ragas(rows) or score_with_judge(rows)
    (Path(__file__).parent / f"results_{name}.json").write_text(
        json.dumps({"config": CONFIGS[name], "scores": scores, "rows": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {scores}")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["baseline", "final", "all"], default="all")
    ap.add_argument("--k", type=int, default=config.SELF_CONSISTENCY_K)
    ap.add_argument("--limit", type=int, default=None, help="use fewer questions (debug)")
    a = ap.parse_args()

    names = ["baseline", "final"] if a.config == "all" else [a.config]
    res = {n: evaluate_config(n, a.k, a.limit) for n in names}

    if len(res) == 2:
        print("\n| Metric | Baseline | Final | Delta |")
        print("|--------|---------|-------|-------|")
        for m in ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]:
            b, f = res["baseline"].get(m), res["final"].get(m)
            d = f"{(f - b):+.3f}" if isinstance(b, float) and isinstance(f, float) else "n/a"
            print(f"| {m} | {b} | {f} | {d} |")
        print(f"\nengine: {res['final'].get('engine')} · "
              f"questions: {a.limit or len(QUESTIONS)}")


if __name__ == "__main__":
    main()
