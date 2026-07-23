# Viva preparation

The rubric says you will be asked to explain any function in the codebase, and
that a group who cannot explain it scores 0 on the transparency category — the
one place where an honest answer beats a polished one every time.

Read this **out loud** once. If a sentence doesn't make sense to you, open the
file next to it and read the code until it does. Six of these you will almost
certainly be asked.

---

## Retrieval

**Why both BM25 and dense embeddings? Isn't dense strictly better?**
No — they fail differently. Dense embeddings match paraphrase: "coastal flooding"
finds "sea level rise". They're weak on exact tokens, because proper nouns,
acronyms and figures are rare in embedding training data, so "Falepili Union" or
"216 million" doesn't reliably land near itself in vector space. BM25 is the
opposite: exact term frequency, so acronyms and numbers always match, but
"forced migration" won't find "displacement". Our questions contain a lot of
proper nouns and figures, which is exactly the case dense retrieval is worst at.
→ `retrieval.py::dense_search`, `bm25_search`

**Why RRF and not just averaging the scores?**
Because the scores aren't comparable. BM25 returns unbounded term-frequency
scores; cosine similarity is bounded in [-1, 1]. Averaging them means inventing a
normalisation, and the result depends on your invention. RRF ignores scores and
uses **ranks**: each list contributes `1/(60+rank)` to each document, and you sum.
A document ranked well by either list gets promoted. Nothing to tune, nothing to
normalise. The 60 is a damping constant from the original paper — it keeps rank 1
from dominating rank 2 too heavily.
→ `retrieval.py::rrf_fuse`

**What does the cross-encoder do that the embeddings didn't?**
A bi-encoder embeds the query and the document *separately* — they never see each
other, so the score is a distance between two independent summaries. A
cross-encoder takes the pair as one input and attends across both, which is what
"is THIS document relevant to THIS query" actually requires. It's far slower —
one full forward pass per pair instead of one vector comparison — so you can't run
it over the corpus. That's why it's stage two: retrieve 15 cheaply, rerank those
15 expensively, keep 5.
→ `retrieval.py::hybrid_retrieve`, `get_reranker`

**Explain parent-child chunking. Why not just use bigger chunks?**
Because chunk size trades precision against recall, and this decouples them. Big
chunks match queries badly — the signal is diluted across 800 words. Small chunks
match well but get cut mid-fact: "21.5 million people are displaced every year" in
one chunk, "by floods and droughts" in the next, and neither is a complete answer.
So we index 200-word children for matching and return their 800-word parents for
generation. Retrieve small, return large. `parent_child_chunk` builds both and
keeps a `parent_id` on every child; `hybrid_retrieve` deduplicates by parent so
three matching children from one parent return that parent once.
→ `ingest.py::parent_child_chunk`, `retrieval.py::hybrid_retrieve`

---

## Security

**Why normalise before matching patterns?**
Because the patterns are the thing being evaded. `Ｉｇｎｏｒｅ` in fullwidth
characters is visually identical to `Ignore` and has completely different bytes,
so a regex for "ignore" misses it. NFKC folds it back. Separately, zero-width
characters (U+200B and friends) render as nothing but split words, so
`Ig\u200bnore` defeats the same regex while looking normal on screen — we strip
those first. Normalise, then match. The reverse order is security theatre.
→ `guardrails.py::normalise`, and the test `test_unicode_evasion_is_normalised`

**Why is indirect injection worse than direct injection?**
Direct injection is a user attacking their own session — they already had whatever
access they had. Indirect injection is hostile text hidden in content the agent
*reads* — a web page, a document, a ticket — and the attacker doesn't need any
permissions, because the attack executes with **the agent's** permissions. An
agent with a send-email tool becomes an exfiltration channel for anyone who can
get text in front of it. That's why every tool result goes through
`sanitise_tool_result` before it touches the context.
→ `guardrails.py::sanitise_tool_result`

**Why is `delete_records` BLOCK rather than CONFIRM?**
CONFIRM means the agent may ask a human. For actions where the downside is
catastrophic and irreversible, letting the agent ask is already too much surface —
humans approve things under time pressure, and a confirmation prompt is one more
thing an attacker can try to shape. BLOCK means there is no path from the agent to
that action at all. The general rule in the matrix: reversible and read-only is
SAFE, reversible with side effects is MONITOR, irreversible or leaving the system
is CONFIRM, catastrophic is BLOCK.
→ `guardrails.py::ACTION_RISK_MATRIX`

**What happens if you add a tool and forget to classify it?**
It resolves to CONFIRM, not SAFE — `DEFAULT_RISK`. Fail closed. Adding a tool
without thinking about it cannot silently widen the attack surface; it just won't
run until someone classifies it. There's a test for this.
→ `guardrails.py::ActionGate.check`, `test_unknown_tool_defaults_to_confirm`

**Why do your security tests not use the LLM?**
Because a test whose outcome depends on sampling temperature isn't a test. If we
asserted "the model refuses this prompt," the suite would pass on Tuesday and fail
on Wednesday with no code change. We test the deterministic components — does the
filter match, does the gate refuse, does the sanitiser neutralise — which are the
things we actually control. The report is explicit that this measures the
guardrails, not the model.

---

## Reasoning

**Why does chain-of-thought help at all?**
Because the intermediate tokens are in the context when the final answer token is
generated, so they constrain it. The model isn't reasoning in some hidden way and
then reporting — the written reasoning *is* the computation, and a wrong final
answer becomes less likely when three correct steps precede it. It reduces the
error rate; it doesn't eliminate it.

**Why Self-Consistency, and why does the majority vote work?**
One reasoning path has no check on itself — the first error propagates to the
conclusion. Three independent paths at high temperature explore different routes,
and for a wrong answer to win the vote, the *same* bias has to appear in a
majority. Idiosyncratic errors don't survive. Cost is k× tokens, which is why it
runs only on the final synthesis and never inside the tool loop.

**How do you take a majority vote over paragraphs of prose?**
This is our design decision, and it's the one worth being able to defend. Classic
Self-Consistency counts identical extracted answers — that needs a discrete answer,
a number or a label. Our output is four sections of analytical prose, and no two
paths are ever textually identical, so the mode doesn't exist and naive voting
degenerates to picking path 1. Instead we embed each path's CONCLUSION and pick
the **medoid** — the one with the highest mean similarity to the others. That's the
answer the paths semantically agree on. We also publish the mean agreement score,
because a low one means the paths disagreed and the reader should know.
The trade-off, which is in the report: the medoid *selects* rather than
*synthesises*, so three mutually different weak answers return the least-odd of
three, not a failure signal.
→ `reasoning.py::_medoid`, `self_consistency_synthesis`

**Why does the critic run deterministic checks as well as an LLM?**
They catch different things and fail differently. A regex that finds a citation to
`[S9]` when the context stops at `[S5]` cannot be talked out of it — that's
fabrication detection you can trust. The LLM critic catches semantic mismatch: a
citation that exists but doesn't support the claim attached to it, which no regex
can see. The verdict is REVISE if either fires, and the deterministic result wins
on conflict.
→ `critic.py::deterministic_checks`, `review`

**How is your system prompt a reward function?**
Whatever the prompt says to optimise, the model optimises — including in ways you
didn't intend. "Complete the research task efficiently" is maximised by returning
nothing: zero tokens, zero latency, task nominally complete, usefulness zero. So
our performance measure names the failure modes explicitly — no empty answer when
sources exist, no citation to something not in context, no HIGH confidence on one
source — which closes off the favourable interpretations. The token budget is the
same idea one level up: a hard cap that makes cost explosion through looping
impossible rather than merely discouraged.
→ `reasoning.py::SYNTHESIS_SYSTEM_PROMPT`, `guardrails.py::TokenBudget`

---

## Questions with an uncomfortable honest answer

**"How much of this did AI write?"**
Answer it straight, matching your disclosure table exactly. The version that
scores is: "The scaffold and first implementations were AI-generated. We wrote
X and Y ourselves, we changed Z after testing showed the original was wrong, and
every measured number in section 3 is from our own runs." Then be able to explain
whatever they point at. A group that says "AI-generated, and here's how it works"
outscores a group that says "we wrote it" and then stalls.

**"Are these RAGAS numbers meaningful on eight documents?"**
No, and the report says so. Top-5 retrieval over eight parent chunks returns most
of the corpus regardless of ranking quality, which flatters recall and depresses
precision. The numbers measure that the pipeline works, not that the retrieval is
good. Limitation 1.

**"What would break first in production?"**
`web_search`. Corpus documents are trusted by construction; web results are
attacker-influenceable and pass through pattern matching, and patterns are
enumerable. A payload phrased outside our fifteen patterns reaches the model
inside a wrapper whose authority depends on the model choosing to respect it.
That's the next thing to replace with a classifier.

**"Why didn't you build L2 and L3?"**
Time. L3 is the one we'd build first and it's specified in the report: promote
the critic's citation check from advisory to a hard gate that blocks the response
if any `[S<n>]` doesn't resolve or any figure isn't verbatim in its cited passage.
The check already exists in `critic.py`; making it blocking is a small change we
chose not to make without measuring the false-positive rate first.
