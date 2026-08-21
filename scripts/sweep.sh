#!/usr/bin/env bash
# Boot a recipe, wait for health, benchmark c1/c3/c5 warm, append one TSV row.
#
# Deliberately minimal. Earlier versions grew "smart" liveness checks (screen
# alive, container present) and every one of them produced false negatives,
# because sparkrun exits after launching and its containers take ~2 min to
# appear. Only four outcomes are decidable without racing the launcher:
#   health 200            -> measure
#   explicit vLLM error   -> FAILED
#   repeated shm stalls   -> WORKER_KILLED (worker died; head waits forever)
#   deadline              -> TIMEOUT
set -uo pipefail

RECIPE="${1:?usage: sweep3.sh <recipe.yaml> <port> <label>}"
PORT="${2:?}"
LABEL="${3:?}"
OUT="${OUT:-$HOME/sweep3_results.tsv}"
LOG="$HOME/sweep3_${LABEL}.log"
DEADLINE=$(( $(date +%s) + 1500 ))

health() { curl -s -m 4 -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null; }
kvtok()  { grep -ohE 'GPU KV cache size: [0-9,]+ tokens' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9,]+'; }

bash "$HOME/spark-launch.sh" "$RECIPE" "$LOG" >/dev/null 2>&1

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  [ "$(health)" = "200" ] && break

  if grep -qaE 'ValueError:|AssertionError:|RuntimeError:' "$LOG" 2>/dev/null; then
    err=$(grep -aoE 'ValueError:.*|AssertionError:.*|RuntimeError:.*' "$LOG" | tail -1 | cut -c1-90)
    printf '%s\t%s\tFAILED\t\t\t%s\n' "$LABEL" "$(kvtok)" "$err" >> "$OUT"; exit 1
  fi

  if [ "$(grep -ac 'No available shared memory broadcast block' "$LOG" 2>/dev/null)" -ge 3 ]; then
    printf '%s\t%s\tWORKER_KILLED\t\t\tallocated then worker died\n' "$LABEL" "$(kvtok)" >> "$OUT"; exit 1
  fi

  sleep 30
done

if [ "$(health)" != "200" ]; then
  printf '%s\t%s\tTIMEOUT\t\t\t\n' "$LABEL" "$(kvtok)" >> "$OUT"; exit 1
fi

MODEL=$(curl -s -m 8 "http://localhost:$PORT/v1/models" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
export PATH="$HOME/.local/bin:$PATH"
run() { BASE="http://localhost:$PORT/v1" MODEL="$MODEL" C="$1" \
  timeout 120 uv run --quiet --with aiohttp python "$HOME/c5.py" 2>/dev/null \
  | grep -oE 'agg= *[0-9.]+' | grep -oE '[0-9.]+'; }

for c in 3 5; do run "$c" >/dev/null; done   # warm
printf '%s\t%s\t%s\t%s\t%s\t\n' "$LABEL" "$(kvtok)" "$(run 1)" "$(run 3)" "$(run 5)" >> "$OUT"
