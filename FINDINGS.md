PART A — the result, written honestly

Hypothesis



We hypothesised that reference-based RAG faithfulness metrics (LLM-as-judge) have a numeric-proximity blind spot: that they fail to penalise an answer which substitutes a semantically-wrong but numerically-close figure — e.g. a year-end "stock" count (7.7 M people still displaced by disasters) swapped for a within-year "flow" count (26.4 M new disaster displacements). This distinction is the single most common error in displacement statistics, flagged by IDMC itself.



Method



We hand-built a 10-item probe set from the IDMC GRID 2024 report. Each item had a correct number, a near-miss (a real figure from the same report meaning something different but numerically comparable), and a far-miss (an obviously wrong number). We generated three answer-versions per item and scored each with a standard faithfulness judge, across two judge models (openai/gpt-oss-20b, llama-3.1-8b-instant), 3 repetitions — 180 judgements per condition. We ran two conditions:



Clean: judge sees only the single correct evidence sentence.

Hard: judge sees a realistic paragraph containing all the key figures, so the correct and trap numbers appear together and the judge must reason about which concept the question asks for.

Result



The hypothesis did not hold, in either condition.



Condition	correct	near-miss	far-miss

Clean	0.95	0.15	0.00

Hard	1.00	0.10	0.00



Both judges reliably penalised the near-miss almost as hard as the far-miss, even when the trap figure appeared in the same paragraph as the correct one. There is no numeric-proximity blind spot under these conditions.



Interpretation



The faithfulness evaluator is not the weak point: it distinguishes stock from flow correctly. This relocates the original problem. When our agent confused 9.7 M and 7.6 M in an earlier run, that was a generation failure — the model picking the wrong figure while reasoning over a long, messy 26 MB report — not an evaluation failure. The judge, given the answer to check, catches the error; the generator, having to produce the answer from a large context, does not always.



Honest limitations



Small pilot (10 items, one domain, two small open judges, 3 reps). A frontier judge and a larger, multi-domain probe set would be needed before drawing a general conclusion. This is a scoping pilot, not a finished study.

