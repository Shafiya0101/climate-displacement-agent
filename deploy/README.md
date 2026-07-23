# Deploying the web interface

The CLI is what gets graded. This is optional and lives behind a separate entry
point (`app.py`) and a separate requirements file, so nothing here can break
`python src/agent.py`.

## Run it locally first

```bash
pip install -r requirements-web.txt
uvicorn app:app --port 7860 --reload
# http://localhost:7860
```

Check before deploying: the masthead fills in, a research question returns a
paper sheet with clickable [S] citations, and the security probe button produces
the red refusal card.

## Deploy to Hugging Face Spaces (free, recommended)

Spaces gives 16 GB RAM on the free CPU tier. That matters: torch plus two
transformer models will OOM on a 512 MB free tier elsewhere.

1. Create an account at huggingface.co, then **New Space** → name it →
   **SDK: Docker** → **Blank** → Public.
2. Clone the Space and copy this project into it:

   ```bash
   git clone https://huggingface.co/spaces/<user>/<space-name>
   cd <space-name>
   rsync -a --exclude '.git' --exclude '.venv' --exclude 'data/index' \
         --exclude '__pycache__' /path/to/your-repo/ .
   cp deploy/SPACE_README.md README.md   # the YAML frontmatter is required
   git add -A && git commit -m "Deploy climate displacement agent" && git push
   ```

   Note `README.md` must be the Space one — the YAML frontmatter at the top is
   how Spaces learns to use Docker and port 7860.

3. In the Space: **Settings → Variables and secrets → New secret**
   - `GROQ_API_KEY` = your key  (required)
   - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (optional, enables tracing)
   - `TAVILY_API_KEY` (optional, enables live web search)

4. Wait for the build. The image bakes both encoders, so it takes 5–10 minutes
   the first time and starts instantly afterwards.

**Never commit `.env`.** Secrets go in the Space settings, where they arrive as
environment variables and `src/config.py` picks them up unchanged.

## Screenshots worth taking for REPORT.md

- The persistent AI-disclosure bar → §5, evidence the Art. 50 obligation is
  implemented rather than just described.
- The red refusal card after pressing **Security probe** → §4.
- A run record panel showing cost, latency and agreement → §3.2.
