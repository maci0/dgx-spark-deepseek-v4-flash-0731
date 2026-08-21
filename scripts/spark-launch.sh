#!/usr/bin/env bash
# Launch a sparkrun recipe from a clean slate, in a detached screen session.
#
# Always tears down the previous run on BOTH nodes first. Stale /dev/shm
# segments matter more here than on a normal box: GB10 is unified memory, so
# orphaned segments consume the same DRAM the GPU allocates its KV pool from,
# and a leftover 40+ GiB silently caps gpu_memory_utilization on the next boot.
#
#   ./spark-launch.sh eugr-prod.yaml [logfile]
set -euo pipefail

RECIPE="${1:?usage: spark-launch.sh <recipe.yaml> [logfile]}"
LOG="${2:-$HOME/$(basename "$RECIPE" .yaml).log}"
WORKER="${WORKER:-10.0.1.2}"
RECIPE_DIR="${RECIPE_DIR:-$HOME/tonyd2wild/sparkrun}"
SESSION="${SESSION:-vllm}"

shm_sweep() {
  find /dev/shm -maxdepth 1 \
    \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) \
    -delete 2>/dev/null || true
}

echo "==> stopping previous run"
screen -S "$SESSION" -X quit 2>/dev/null || true
pkill -9 -f '[s]parkrun' 2>/dev/null || true

# Remove by explicit name: a piped `docker ps -q | xargs docker rm` has proved
# unreliable over nested ssh here.
docker ps -a --format '{{.Names}}' | grep '^sparkrun' | while read -r n; do
  docker rm -f "$n" >/dev/null 2>&1 && echo "    removed $n (head)"
done
ssh -o StrictHostKeyChecking=no "$WORKER" '
  docker ps -a --format "{{.Names}}" | grep "^sparkrun" | while read -r n; do
    docker rm -f "$n" >/dev/null 2>&1 && echo "    removed $n (worker)"
  done' 2>/dev/null || true

sleep 5

echo "==> sweeping /dev/shm on both nodes"
shm_sweep
ssh -o StrictHostKeyChecking=no "$WORKER" "$(declare -f shm_sweep); shm_sweep" 2>/dev/null || true

printf '    head   free: %s GiB\n' "$(free -g | awk '/Mem:/{print $7}')"
printf '    worker free: %s GiB\n' \
  "$(ssh -o StrictHostKeyChecking=no "$WORKER" "free -g | awk '/Mem:/{print \$7}'" 2>/dev/null)"

echo "==> launching $RECIPE (log: $LOG)"
screen -dmS "$SESSION" bash -lc \
  "cd '$RECIPE_DIR' && PATH=\$HOME/.local/bin:\$PATH uvx sparkrun@0.3.5 run '$RECIPE' --cluster spark --trust > '$LOG' 2>&1"
sleep 3
screen -ls | grep -q "$SESSION" && echo "    screen '$SESSION' running" || {
  echo "    ERROR: screen session did not start" >&2; exit 1; }
