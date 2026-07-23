# RUNBOOK — from zip to submitted, in one pass

Everything below is copy-paste. **PowerShell** commands are the default (Windows).
macOS/Linux equivalents are given where they differ.

> **Deadline: today, 23:59 Paris.** Late = 0.
> **Push a working version by mid-afternoon**, then keep improving and re-pushing.
> The instructor clones at 23:59, so the last commit before then is what counts —
> but a repo that already works at 15:00 is insurance against everything going
> wrong at 22:00.

**Time budget** (start ~10:15):

| Time | Step | Blocking? |
|---|---|---|
| 10:15–11:00 | 1–6 · install and first run | **yes, everything depends on it** |
| 11:00–11:15 | 7 · Langfuse | no |
| 11:15–12:30 | 8 · RAGAS (runs unattended — start it, then go to step 11) | no |
| 12:30–13:00 | 9–10 · metrics + security baseline | no |
| 13:00–15:00 | 11 · write REPORT.md | no |
| 15:00–15:45 | 12–14 · GitHub, clean-clone test, **email** | **yes** |
| after | 15 · deploy the UI, read demo/VIVA.md | no |

---

## 1 · Extract the zip

Download `climate-displacement-agent.zip` from the chat. It lands in `Downloads`.

**PowerShell**
```powershell
cd $HOME\Downloads
Expand-Archive -Path .\climate-displacement-agent.zip -DestinationPath $HOME\Projects -Force
cd $HOME\Projects\climate-displacement-agent
dir
```

**macOS / Linux**
```bash
mkdir -p ~/Projects && unzip ~/Downloads/climate-displacement-agent.zip -d ~/Projects
cd ~/Projects/climate-displacement-agent && ls
```

You should see `src`, `tests`, `eval`, `data`, `docs`, `demo`, `web`, `README.md`,
`REPORT.md`. If you see a *nested* `climate-displacement-agent` folder, `cd` into it.

---

## 2 · Check Python

```powershell
python --version
```

Need **3.10 or newer**. If the command isn't found or the version is older:

```powershell
winget install Python.Python.3.12
```
Then **close and reopen PowerShell** and check again.

---

## 3 · Virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**If you get `running scripts is disabled on this system`** — this is the single
most common Windows blocker. Fix it for this window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. It must say that for every command
from here on. If you open a new terminal, re-run the activate line.

**macOS / Linux:** `python3 -m venv .venv && source .venv/bin/activate`

---

## 4 · Install

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Takes 3–8 minutes; `torch` is a large download. Wait for it to finish cleanly.
If it errors, read the last 5 lines — it is almost always a Python version
mismatch, which step 2 fixes.

---

## 5 · API key

1. Go to **https://console.groq.com** → sign up (free) → **API Keys** → **Create API Key**.
2. Copy it immediately — it is shown once. It starts with `gsk_`.

```powershell
Copy-Item .env.example .env
notepad .env
```

In Notepad, put your key on the `GROQ_API_KEY=` line with **no spaces and no quotes**:

```
GROQ_API_KEY=gsk_yourkeyhere
```

Save, close.

**Now verify the model in `.env` still exists at the provider.** Models get
retired, and a retired ID is a 404 that will waste an hour of your evening:

```powershell
$k = (Get-Content .env | Select-String '^GROQ_API_KEY=').ToString().Split('=',2)[1]
(Invoke-RestMethod -Uri https://api.groq.com/openai/v1/models -Headers @{Authorization="Bearer $k"}).data.id
```

**macOS / Linux:**
```bash
source .env && curl -s https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY" | grep '"id"'
```

Look for `llama-3.3-70b-versatile` in the output. **If it isn't there**, pick a
similar large model from the list and set all three lines in `.env`:
`TOOL_MODEL=`, `SYNTH_MODEL=`, `CRITIC_MODEL=`.

---

## 6 · Prove it works — the graded gate

```powershell
python -m pytest tests/test_security.py -v
python src/agent.py
```

The first `agent.py` run downloads two models (~180 MB) — one to two minutes,
once. Then it prints an EVIDENCE / ANALYSIS / CONCLUSION / CONFIDENCE answer, a
CRITIC VERDICT, and a cost line.

**If both of those work, the hardest part is done.** If `agent.py` fails, fix it
before anything else — a repo that doesn't run caps Technical Implementation at
10/50 no matter how good the report is.

---

## 7 · Langfuse (5 points, 10 minutes)

1. **https://cloud.langfuse.com** → sign up → **New project**.
2. **Settings → API Keys → Create** → copy the public and secret keys.
3. `notepad .env` → fill `LANGFUSE_PUBLIC_KEY=` and `LANGFUSE_SECRET_KEY=` → save.

```powershell
python src/agent.py "What is the Falepili Union?"
```

It should print `[langfuse] tracing enabled`. Open the Langfuse dashboard →
**Tracing → Traces** → open the newest one. You should see nested spans:
`agent_run` → `planner_step_1` → `tool:...` → `synthesis_path_1..3` → `critic_review`.

**Screenshot it and copy the trace URL** — both go in REPORT.md §3.2.

---

## 8 · RAGAS — start this now, it runs unattended (12 points)

```powershell
pip install -r requirements-eval.txt
python eval/run_ragas.py --config all
```

**This takes 20–45 minutes and costs a few cents.** Leave it running in this
window, open a **second** PowerShell window, `cd` to the project, run
`.\.venv\Scripts\Activate.ps1` there, and get on with step 11 while it works.

It prints a finished markdown table at the end:

```
| Metric | Baseline | Final | Delta |
|--------|---------|-------|-------|
| context_recall | 0.61 | 0.83 | +0.220 |
...
engine: ragas · questions: 12
```

**Copy that whole block.** Also note the `engine:` line — if it says
`llm_judge_fallback`, ragas didn't install cleanly and the built-in judge ran
instead. That is fine; say so in the report.

Quick smoke test first if you're nervous: `python eval/run_ragas.py --config final --limit 2`

---

## 9 · Cost, latency, tool distribution, budget trigger (8 points)

```powershell
python eval/report_metrics.py --runs 10 --budget-demo
```

~10 minutes. Prints average cost, median latency, a tool-distribution table, and
the budget-trigger demo output. **Copy all of it.**

---

## 10 · Security baseline — the "before" column (part of 10 points)

```powershell
python eval/security_before.py
```

Runs the five payloads with the guardrails switched off and tells you which ones
succeeded, then prints the table rows to paste. Expect 1–3 of 5 to get through —
that's the normal, publishable result.

Then the "after" column, which is just:

```powershell
python -m pytest tests/test_security.py -v
```

All 10 pass → all five "after" cells are ✓.

---

## 11 · Write REPORT.md (20 + 10 points — do not rush this)

```powershell
notepad REPORT.md
```

Search for `<<` — every one is a hole to fill. Nothing else needs writing; the
prose is done. Work through in this order:

1. **§7 AI disclosure table.** Do it first, while you're honest and unhurried.
   Mark AI-generated where it's true. An accurate table plus "here's how it works"
   scores 10/10; a false one scores 0 and takes the whole category with it.
2. **§3 Evaluation** — paste the RAGAS table from step 8 and the metrics from step 9.
   Then write the two paragraphs: which techniques moved which metric, and
   **which metric didn't improve and why**. The second is graded and most groups
   skip it.
3. **§4 Security** — the before column from step 10.
4. **§5 EU AI Act** — mostly written. Delete disclosure items 1 and 2 if you don't
   deploy the UI today.
5. **§1, §2, §6** — already written. Read them, change anything that isn't true of
   your build, add your names at the top.

Delete the instruction blockquote at the top when you're finished.

---

## 12 · GitHub

### Create the repo (browser)

1. **https://github.com/new**
2. **Repository name:** `climate-displacement-agent`
3. **Description:**
   `Production AI research agent for climate displacement analysis — hybrid RAG (BM25+dense+RRF), cross-encoder reranking, custom MCP server, L1/L4 guardrails, self-consistency CoT, critic agent. AIVANCITY PGE5.`
4. **Public** ← required, the rubric checks accessibility.
5. **Do not** add a README, .gitignore, or licence — you already have them.
6. **Create repository**, then copy the URL.

### Push

```powershell
git init
git add -A
git status
```

**Before committing, confirm your key is not in the staging list:**

```powershell
git status --short | Select-String "\.env"
```

This must show **only** `.env.example`. If plain `.env` appears, stop:
`git rm --cached .env` and check `.gitignore` contains `.env`.

```powershell
git commit -m "Production climate displacement agent: hybrid RAG, MCP server, guardrails, CoT, critic"
git branch -M main
git remote add origin https://github.com/YOURUSERNAME/climate-displacement-agent.git
git push -u origin main
```

First push opens a browser sign-in. If it asks for a password on the command
line, that's a personal access token, not your account password — the browser
flow is easier.

**If `git` isn't installed:** `winget install Git.Git`, then reopen PowerShell.

**Set your identity if git complains:**
```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## 13 · The clean-clone test — do not skip this

This is literally what the instructor will do. Everything above can be perfect and
still fail here.

```powershell
cd $HOME\Desktop
git clone https://github.com/YOURUSERNAME/climate-displacement-agent.git gate-test
cd gate-test
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```
(paste your key, save, close)
```powershell
python src/agent.py
python -m pytest tests/test_security.py
```

Both must work. If they don't, fix it in your **real** project folder, commit,
push, delete `gate-test`, and do this again.

```powershell
cd $HOME\Desktop
Remove-Item -Recurse -Force gate-test
```

---

## 14 · Submit

Email the instructor:

- **Subject:** `[PGE5 HW] Group N — Climate displacement`
- **Body:** the repository URL, your group number, and the names of everyone in
  the group. If you deployed the app, add the Space URL on its own line.

Then push any final commits before 23:59. **You are done and graded.**

---

## 15 · After submission — the demo

Only now:

```powershell
pip install -r requirements-web.txt
uvicorn app:app --port 7860
```

Open **http://localhost:7860**. Try a research question, then the **Security
probe** button. Screenshot the disclosure bar, the red refusal card, and the run
record — those three go into REPORT.md §5, §4 and §3.2, and you can still commit
them before the deadline.

Deployment steps: `deploy/README.md`
Demo walkthrough: `demo/SCRIPT.md`
Questions you'll be asked: `demo/VIVA.md` ← **read this out loud before you present**

---

## When something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `running scripts is disabled` | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `python` not recognised | not on PATH | reinstall via `winget install Python.Python.3.12`, reopen the terminal |
| `ModuleNotFoundError` | venv not activated | `.\.venv\Scripts\Activate.ps1` — prompt must show `(.venv)` |
| `model ... does not exist` / 404 | model retired | re-run the model list in step 5, update all three model lines in `.env` |
| `AuthenticationError` | key wrong | check `.env` for stray quotes or spaces |
| `rate limit` during RAGAS | free tier throttle | wait a minute, re-run — results are written per config |
| ragas install fails | dependency conflict | skip it — the script falls back to a built-in judge; say so in the report |
| agent answers "not covered by the retrieved context" | corpus is only 8 docs | correct behaviour, not a bug — it's Limitation 1 in the report |
| first run very slow | downloading encoders | one-off, ~180 MB |
