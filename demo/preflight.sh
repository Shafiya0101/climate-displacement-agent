#!/usr/bin/env bash
# Run this 20 minutes before you present. Every check is one that has failed for
# somebody in front of an audience.
set -uo pipefail
SPACE_URL="${SPACE_URL:-}"          # export SPACE_URL=https://<user>-<space>.hf.space
PASS=0; FAIL=0
ok(){ echo "  ok    $1"; PASS=$((PASS+1)); }
no(){ echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

echo "== 1. local repo =="
python -m pytest tests/test_security.py -q >/dev/null 2>&1 && ok "10 security tests pass" || no "security tests"
[ -f .env ] && grep -q "GROQ_API_KEY=." .env && ok "GROQ_API_KEY present locally" || no "GROQ_API_KEY missing in .env"
git status --porcelain 2>/dev/null | grep -q . && echo "  warn  uncommitted changes" || ok "working tree clean"

echo "== 2. provider is alive and the model still exists =="
if [ -f .env ]; then set -a; . ./.env; set +a; fi
MODEL="${SYNTH_MODEL:-llama-3.3-70b-versatile}"
LIST=$(curl -s -m 15 https://api.groq.com/openai/v1/models -H "Authorization: Bearer ${GROQ_API_KEY:-}")
echo "$LIST" | grep -q '"id"' && ok "provider reachable, key accepted" || no "provider unreachable or key rejected"
echo "$LIST" | grep -q "\"$MODEL\"" && ok "model $MODEL is current" || no "model $MODEL NOT in provider list — update .env"

echo "== 3. one real end-to-end run =="
OUT=$(timeout 180 python src/agent.py "What is the Falepili Union?" 2>&1)
echo "$OUT" | grep -q "CRITIC VERDICT" && ok "CLI run produced an answer and a verdict" || no "CLI run did not complete"
echo "$OUT" | tail -5 | sed 's/^/        /'

echo "== 4. deployed Space =="
if [ -z "$SPACE_URL" ]; then
  echo "  skip  set SPACE_URL to check the deployment"
else
  code=$(curl -s -o /dev/null -m 90 -w "%{http_code}" "$SPACE_URL/healthz")
  [ "$code" = "200" ] && ok "Space awake ($SPACE_URL)" || no "Space returned $code — open it in a browser and wait for boot"
  curl -s -m 30 "$SPACE_URL/api/meta" | grep -q '"llm_configured":true' \
    && ok "Space has its API key secret set" || no "Space is missing GROQ_API_KEY in Settings → Secrets"
  echo "  ... sending one warm-up question so the first live demo run is not the first run"
  JOB=$(curl -s -m 30 -X POST "$SPACE_URL/api/ask" -H 'Content-Type: application/json' \
        -d '{"question":"What is the Falepili Union?"}' | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
  [ -n "$JOB" ] && ok "warm-up run started (job $JOB)" || no "could not start a warm-up run"
fi

echo
echo "passed $PASS, failed $FAIL"
[ "$FAIL" -eq 0 ] && echo "Ready." || echo "Fix the failures above before you present."
