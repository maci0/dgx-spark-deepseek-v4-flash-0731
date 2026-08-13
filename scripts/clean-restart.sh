#!/usr/bin/env bash
# Clean restart for the 2-node mp serve. Avoids the restart-loop/deadlock:
#  - down both nodes, reap orphaned vllm/EngineCore procs (they survive `down`),
#  - free the other node's GPU (stop competing tenants), re-apply the clock cap,
#  - single start. A reboot is only needed if state is truly wedged.
set -euo pipefail
KEY="${SSH_KEY:-$HOME/.config/NVIDIA/Sync/config/nvsync.key}"
HEAD="${HEAD:-192.168.0.211}"; WORKER="${WORKER:-192.168.0.212}"
DIR="${REPO_DIR:-/home/maci/tonyd2wild}"
CLK="${CLOCK_CAP:-2200}"
for n in "$HEAD" "$WORKER"; do
  ssh -i "$KEY" "maci@$n" "cd '$DIR' && COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file .env.dspark -f docker-compose.dspark.yml down --remove-orphans >/dev/null 2>&1 || true
    pkill -9 -f 'vllm serve|EngineCore|multiproc_executor' 2>/dev/null || true
    docker rm -f tonyd2wild-vllm-dspark-1 2>/dev/null || true
    docker stop \$(docker ps -q --filter name=llama) gpustack-worker 2>/dev/null || true
    sudo -n nvidia-smi -lgc 0,$CLK 2>/dev/null || true"
  echo "cleaned $n"
done
ssh -i "$KEY" "maci@$HEAD" "cd '$DIR' && ./start-deepseek-v4-flash-dspark.sh"
