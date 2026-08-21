# Recipes — runbook

sparkrun recipes for the 2× DGX Spark pair. Copy to your sparkrun recipe dir and run
from the head node.

## `deepseek-v4-flash-0731-dspark-arena-threshold.yaml`

The **arena / low-concurrency** config. This is the recipe behind the leaderboard
submission (44.75 decode) and the one to run when you want a stable server with fast
startup rather than maximum context.

| | |
|---|---|
| Image | `ghcr.io/bjk110/vllm-spark@sha256:d8492e76…` (stage-c, vLLM 0.21.1rc1) — digest-pinned |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| KV | `fp8`, `block_size 256` |
| Spec | DSpark **k=3**, `draft_sample_method: probabilistic` |
| MoE | b12x (`VLLM_USE_B12X_MOE=1`, `VLLM_USE_B12X_WO_PROJECTION=1`) |
| Scheduling | `max_num_seqs 12`, `long_prefill_token_threshold 1024`, `max_num_partial_prefills 1` |
| Memory | `gpu_memory_utilization 0.78` |
| Port | **8888** |

### Run

```bash
cd <your sparkrun recipe dir>
uvx sparkrun@0.3.5 run deepseek-v4-flash-0731-dspark-arena-threshold.yaml \
    --cluster spark --trust
```

Boot is ~300 s (weights + CUDA-graph capture). Verify:

```bash
curl -s localhost:8888/health                      # expect 200
curl -s localhost:8888/v1/models | jq -r .data[0].id  # deepseek-v4-flash-0731
```

In the log you want to see:

```
Using 'B12X' Mxfp4 MoE backend        # missing = half-speed fallback
num_spec_tokens=3                     # DSpark active
GPU KV cache size: ~1.45M tokens
```

### Measured (2× GB10, TP=2, warm, `ignore_eos`, 128 tok/req, 2026-08-21)

| concurrency | aggregate tok/s | per-request |
|---|---:|---:|
| 1 | 58.3 | 58.3 |
| 5 | **162.5** | ~32.5 |

KV pool: **12.27 GiB = 1,448,712 tokens** (1.38× a full 1M-token context).

### When *not* to use this one

For long concurrent coding sessions use the **1M recipe** (NVFP4 KV, `max-num-seqs 6`,
k=5, util 0.82) — it trades startup time and low-concurrency throughput for a
2.1M-token KV pool. See [TUNING.md](../TUNING.md).

Do **not** simply set `kv_cache_dtype: nvfp4_ds_mla` on *this* recipe: spec-decode
buffers scale with `max_num_seqs × (k+1)`, so at `seqs 12` they consume the memory that
should become KV and you end up with *fewer* tokens (1.35M) and slower decode. KV dtype
and seqs/k must be changed together.

### Gotchas

- `k` must be ≤5 or a multiple of `n_predict=5`. **k=7 boots then crashes on the first
  generation; k=10 crashes every generation.** On this recipe k=5 measured −6.7% vs k=3.
- `gpu_memory_utilization` above ~0.80 with `max_num_seqs 12` risks the spec-decode
  buffer OOM that appears only under real traffic, not at boot.
- Both nodes need `loginctl enable-linger $USER`, or systemd deletes the worker's POSIX
  semaphores when the SSH session closes and the head hangs forever on a collective.
  See [TROUBLESHOOTING.md](../TROUBLESHOOTING.md).
