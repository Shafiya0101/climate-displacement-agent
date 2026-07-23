# Corpus

## What goes here

Any `.md`, `.txt` or `.pdf` file placed in `data/corpus/` is indexed. Rebuild with:

```bash
python src/ingest.py
```

Chunking is parent-child: 200-word children are indexed for retrieval, 800-word
parents are returned to the LLM. Sizes are configurable via `PARENT_WORDS`,
`CHILD_WORDS` and `CHILD_OVERLAP` in `.env`.

## What is currently here

`data/corpus/` ships with **eight summary notes** covering the domain. They are
clearly headed as course demo notes, each naming the report it summarises and its
URL. They exist so the repository runs from a clean clone without a large download.

**They are a smoke-test corpus, not a research corpus.** The summaries were written
for this project, not extracted from the source PDFs, so any RAGAS figure obtained
against them measures the pipeline, not the documents. Replace them before drawing
substantive conclusions.

## Recommended real sources

Download these PDFs into `data/corpus/` and delete the demo notes:

| File | Source | URL |
|---|---|---|
| GRID | IDMC Global Report on Internal Displacement | https://www.internal-displacement.org/global-report/ |
| Groundswell Part 2 | World Bank, 2021 | https://openknowledge.worldbank.org/handle/10986/36248 |
| AR6 WGII SPM | IPCC, 2022 | https://www.ipcc.ch/report/ar6/wg2/ |
| Climate change and disaster displacement | UNHCR | https://www.unhcr.org/what-we-do/build-better-futures/environment-disasters-and-climate-change |
| Protection Agenda | Platform on Disaster Displacement / Nansen Initiative | https://disasterdisplacement.org/ |
| Kampala Convention | African Union, 2009 | https://au.int/en/treaties/ |

After adding PDFs, re-run `python src/ingest.py` and then
`python eval/run_ragas.py --config all` to regenerate the evaluation table.

## Generated files (git-ignored)

- `data/index/` — `chunks.json` and `child_vectors.npy`, rebuilt on demand
- `data/memory.json` — findings written by `store_finding`
- `data/runs.jsonl` — one line per run; read by `eval/report_metrics.py`
