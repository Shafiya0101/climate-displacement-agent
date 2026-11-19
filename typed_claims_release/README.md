# Typed numeric claims — data and code release

Artifacts for *Right Number, Wrong Question: A Typed-Claim Account of Numeric
Faithfulness in Retrieval-Augmented Generation*.

The paper argues that reference-based faithfulness is a function `f(a, R)` of
the answer and the retrieved context only. It never receives the question, so
it cannot distinguish a correct answer from a **substitution error** — a true,
correctly predicated, correctly citable figure that answers a *different*
measurement question. This release contains the probes used to demonstrate
that, the judge verdicts, and everything needed to rebuild both from source.

---

## Contents

```
displacement/   probe and generator questions for the IDMC domain
tuberculosis/   probe and source data for the WHO TB domain
results/        raw per-item judge verdicts (both judges, TB domain)
code/           build, run and analysis scripts + a Colab notebook
```

---

## The three error classes

For a question `q` whose correct answer has type `τ_q`, retrieved context `R`,
and an answer asserting a numeric claim `c`:

| Check | Test | Relates |
|---|---|---|
| (1) value fidelity | is `v(c)` present in `R`? | answer ↔ evidence |
| (2) attribution consistency | does `c`'s predication match the source claim supplying `v(c)`? | answer ↔ evidence |
| (3) type responsiveness | does `type(c)` equal `τ_q`? | answer ↔ **question** |

- **fabrication** — fails (1). Ordinary ungrounded generation.
- **attribution error (ATT)** — passes (1), fails (2). A *false*, internally
  inconsistent sentence: a flow value under a stock predication.
- **substitution error (SUB)** — passes (1) and (2), fails (3). A *true*,
  internally consistent, citable sentence that answers a different question.

Faithfulness can express ATT. It cannot express SUB, because detecting SUB
requires `q` and faithfulness never receives it.

---

## Probe file format

Both probes use the same schema. One JSON array; each element:

```jsonc
{
  "id": "tb01",
  "question": "How many prevalent cases of TB were estimated ... in Philippines in 2007?",
  "gold":      { "value": "159,638", "label": "<full sentence>",
                 "temporal": "STOCK", "entity": "cases",
                 "unit": "count", "delta": 0.0 },
  "near_miss": { "value": "85,720",  "temporal": "FLOW", "delta": 0.463, ... },
  "far_miss":  { "value": "1,531,762", "temporal": "FLOW", "delta": 8.6,  ... },
  "evidence":  "<gold sentence> <near sentence> <far sentence>",
  "answers": {
    "correct":  "<gold sentence, verbatim>",
    "ATT_near": "<gold sentence, numeral swapped for the near value>",
    "ATT_far":  "<gold sentence, numeral swapped for the far value>",
    "SUB_near": "<the near distractor's OWN sentence, verbatim>",
    "SUB_far":  "<the far distractor's OWN sentence, verbatim>"
  }
}
```

`delta` is `|v' − v| / v`, the relative numeric distance from the gold value.

### Why the answers are assembled rather than written

This is the methodological point of the construction. `ATT` answers are the
gold sentence with **only the numeral changed**, so they are false and
internally inconsistent *by construction*. `SUB` answers are the distractor's
**own** sentence, so they are true and internally type-consistent *by
construction*.

No answer is hand-authored, so "framing purity" — whether a SUB answer really
is internally consistent — is a structural property of the generator rather
than a judgement needing annotator validation.

---

## Domain 1: internal displacement

`displacement/probe_displacement.json` — 57 items built from the IDMC
*Global Report on Internal Displacement 2024*. Gold claims and distractors are
sentences occurring in that report. Median near δ = 0.13, median far δ = 75.

Temporal operators were assigned from a cue-word scheme fixed before any item
was written (`new`/`during`/`triggered` → FLOW; `living in`/`remained`/`at
year-end` → STOCK; `projected`/`could reach`/`by 2050` → PROJ) and validated by
a second annotator on 20 items (gold type κ = 1.00, distractor type κ = 1.00,
framing purity 20/20 agreement).

`displacement/questions_surface_ambiguous.json` — the generator study. Each
question is labelled `cued` (wording discloses the demanded type) or `nocue`
(it does not). Cue membership is checked mechanically, so "surface-ambiguous"
is a property of the string rather than a judgement.

## Domain 2: tuberculosis

`tuberculosis/probe_tb.json` — 38 items. Median near δ = 0.57, median far
δ = 19.9. 25 items have a STOCK gold, 13 a FLOW gold.

`tuberculosis/who_prev_notif_source.csv` — the source, taken unmodified from
WHO's own report data repository (`GTB-TME/gtbreport2024`, file
`data/ps/prev_notif_2023-07-03.csv`). For each national prevalence survey it
gives a prevalence rate, a population, and a notification count.

**Types here are schema-given, not inferred.** The `prev` column is a
prevalence — a count of cases *existing* at survey time, a STOCK. The
`notified` column counts cases *reported during* the year, a FLOW. Both cover
the same country, year and population, and they differ by factors of 0.6 to
5.8. No cue words, no annotator.

Absolute prevalent cases are computed as `prev / 100000 × pop`.

Near distractors hold country and year fixed and swap only the temporal
operator. Far distractors come from a *different* country, since each country
has only one survey — so the far condition differs in geography as well as
temporal operator. **The near condition is the clean one.**

---

## Results

`results/tb_judge_gpt-4o-mini.jsonl` (primary) and
`results/tb_judge_gpt-oss-20b.jsonl` (replication). One line per
item × answer-version, 190 lines each:

```jsonc
{"id": "tb01", "row": "correct", "faith": 1.0, "rel": 1.0}
```

`faith` and `rel` are binary verdicts from single-call LLM judges at
temperature 0, following the faithfulness and answer-relevance *definitions* of
Es et al. (2024) but **not** computed with that library — so these numbers are
not comparable to scores reported with it. The two prompts:

```
FAITHFULNESS: are the ANSWER's claims supported by the EVIDENCE?
              inputs: EVIDENCE, ANSWER          <- note: no QUESTION
RELEVANCE:    does the ANSWER address the QUESTION asked?
              inputs: QUESTION, ANSWER
```

The faithfulness prompt is the paper's argument in concrete form: the question
is not among its inputs.

### Faithfulness (proportion judged faithful)

| class | displacement (n=57) | TB (n=38) |
|---|---|---|
| correct | 1.000 | 1.000 |
| ATT near | 0.000 | 0.000 |
| ATT far | 0.000 | 0.000 |
| **SUB near** | **1.000** | **1.000** |
| **SUB far** | **1.000** | **1.000** |

Identical across two unrelated corpora and two model families. Wilson 95% CIs:
[93.7, 100] and [90.8, 100] for the SUB rows; [0, 6.3] and [0, 9.2] for ATT.

### Relevance (proportion judged relevant), TB

| class | gpt-4o-mini | gpt-oss-20b |
|---|---|---|
| correct | 38/38 | 38/38 |
| ATT near | 38/38 | 38/38 |
| ATT far | 38/38 | 33/38 |
| SUB near | 0/38 | 0/38 |
| SUB far | 0/38 | 0/38 |

The only cell where the two judges disagree is ATT far.

### Caveat on the TB relevance column

TB relevance separates more sharply (AUC 1.000 for SUB) than displacement
(0.667 near). **Treat this as an upper bound, not a stronger result.** The TB
items come from two fixed sentence templates, one per temporal operator, so
stock and flow readings differ in stereotyped surface wording a judge can
separate without representing measurement type at all. The displacement probe,
built from naturally-occurring report sentences, carries no such regularity.

The faithfulness column is unaffected: a stereotyped sentence is still entailed
by its evidence, which is precisely why faithfulness scores every SUB item 1.0.

---

## Reproducing

**Easiest.** Open `code/reproduce_tb_colab.ipynb` in Colab, add a Groq key
(free) and/or an OpenAI key, Run all. It rebuilds the TB probe from the WHO
source, runs both judges and prints the analysis. ~15 min, ~$0.008 on OpenAI.

**Locally.**

```bash
pip install openai

python code/build_probe_tb.py \
    --csv tuberculosis/who_prev_notif_source.csv --out probe_tb.json

export OPENAI_API_KEY=...   GROQ_API_KEY=...
python code/run_judges.py --probe probe_tb.json --judge gpt-4o-mini
python code/run_judges.py --probe probe_tb.json --judge openai/gpt-oss-20b

python code/analyze.py --judge judge_gpt-4o-mini.jsonl \
                       --replication judge_openai_gpt-oss-20b.jsonl
```

`code/run_judges.py` saves after every call and skips completed work, so it can
be interrupted and resumed. `code/analyze.py` prints the proportions, Wilson
intervals, Mann–Whitney AUCs, the cross-domain table and a LaTeX table body.

**Note on `gpt-oss-20b`:** it is a reasoning model and will return empty
content under a small `max_tokens`. Use `max_tokens=512` with
`reasoning_effort="low"`, and read the last `0`/`1` digit emitted. The Colab
notebook already does this.

### Known variance

Re-running the displacement judges at temperature 0 moved 1–4 items per cell in
the *relevance* column (e.g. SUB near 36/57 → 32/57) while faithfulness was
identical every time. LLM judges are not fully deterministic at temperature 0;
expect small drift in relevance and none in faithfulness.

---

## Licence and sources

Probe items are derived from two public sources, used under the terms of their
release and reproduced only to the extent needed for reproducibility:

- IDMC, *Global Report on Internal Displacement 2024*, Geneva.
- WHO, *Global Tuberculosis Report 2024*, Geneva; survey data via
  `github.com/GTB-TME/gtbreport2024`.

The probes deliberately contain false and misleading statements paired with
real published figures. Every item is released with its gold type and
error-class labels. **Do not detach generated incorrect answers from those
labels**, and do not treat any `ATT_*` or `SUB_*` string as a factual claim
about displacement or tuberculosis.

Code is released for research use.
