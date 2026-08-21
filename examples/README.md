# Recipes: runbook

sparkrun recipes for the 2× DGX Spark pair. Copy to your sparkrun recipe dir and run
from the head node.

## `deepseek-v4-flash-0731-dspark-arena-threshold.yaml`

The **arena / low-concurrency** config. This is the recipe behind the leaderboard
submission (44.75 decode) and the one to run when you want a stable server with fast
startup rather than maximum context.

| | |
|---|---|
| Image | `ghcr.io/bjk110/vllm-spark@sha256:d8492e76…` (stage-c, vLLM 0.21.1rc1), digest-pinned |
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
k=5, util 0.82), it trades startup time and low-concurrency throughput for a
2.1M-token KV pool (allocated; see its serving caveat below). See [TUNING.md](../TUNING.md).

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

## `prod-c5-ssd.yaml`

The **production** config: ~5 concurrent clients, KV spill to NVMe.

> **Does not serve yet.** The NVFP4 KV pool allocates (2.23M tokens at util 0.82) but the worker
> node dies during sparse-MLA warmup before the server is healthy. The SSD offload path itself is
> verified. Use the arena-threshold recipe above for a config that actually serves.
Serves on **port 8890** (so it can run alongside the arena config on 8888).

| | |
|---|---|
| Image | stage-c, digest-pinned (only build with the padded `nvfp4_ds_mla` writer) |
| Model | `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` (gated) |
| KV | `nvfp4_ds_mla`, util **0.82** → 2,233,845 tokens allocated (2.13×), *never reached serving* |
| Spec | DSpark **k=5** |
| Clients | `max_num_seqs 6` (5 + headroom) |
| Offload | `OffloadingConnector` + stock `fs_python` disk tier → `/kvspill` |

### Run

```bash
mkdir -p ~/kvspill                      # on BOTH nodes
uvx sparkrun@0.3.5 run prod-c5-ssd.yaml --cluster spark --trust
curl -s localhost:8890/health           # expect 200
```

### Prerequisites that are easy to miss

1. **`loginctl enable-linger $USER` on both nodes**: otherwise systemd deletes the worker's POSIX
   semaphores when the SSH session ends and the head hangs forever.
2. **A worker-node death during warmup is the open blocker**: the headless worker logs
   `Parent process exited, terminating worker queues` mid-`Warming up DeepSeek V4 sparse MLA
   attention` and the boot never completes. Ruled out so far: systemd `RemoveIPC` (linger is
   enabled), SSH session teardown (reproduces under `screen`), and a `start_worker_monitor`
   liveness patch (loaded and confirmed active; the parent still exited). Kernel OOM is *not*
   implicated either: `dmesg` shows zero kills on both nodes. Cause still unknown.
3. **Clear stale `/dev/shm`** before a restart, orphaned segments from crashed runs eat the same
   DRAM the GPU allocates from and will cap `gpu_memory_utilization`:
   ```bash
   find /dev/shm -maxdepth 1 \
     \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete
   ```

### Tuning notes

- `gpu_memory_utilization` **0.84** allocates 2,575,356 tokens (2.46×) but startup runs 25+ minutes and
  becomes fragile. 0.82 is the shipped default for restart sanity.
- `cpu_bytes_to_use` is **2 GiB deliberately**. GB10 is unified memory, so the CPU staging tier
  competes with the GPU KV pool, 32 GiB OOM-killed the worker.
- Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; any offload connector rejects it.

Full rationale, measurements and upstream evidence: [PROD_C5_SSD.md](../PROD_C5_SSD.md).


## `eugr-prod.yaml` (SHIPPED production config)

The **production** config for 3-5 concurrent sessions. This is the one that
serves. Full rationale: [EUGR_B12X_PROD.md](../EUGR_B12X_PROD.md).

| | |
|---|---|
| Image | `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:2026081903` |
| Model | `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` (gated) |
| KV | `fp8_ds_mla`, util 0.89 -> **1,652,056 tokens** |
| Spec | DSpark **k=5**, probabilistic |
| Backends | b12x MoE + linear + `B12X_MLA_SPARSE` attention |
| Sessions | `max_num_seqs 6` (5 + headroom) |
| Port | **8000** |

### Run

```bash
cd ~/tonyd2wild/sparkrun
screen -dmS vllm bash -lc "uvx sparkrun@0.3.5 run eugr-prod.yaml --cluster spark --trust > ~/eugr.log 2>&1"
curl -s localhost:8000/health          # expect 200
```

Confirm in the log:

```
Using 'B12X' Mxfp4 MoE backend     # missing = half-speed fallback
num_spec_tokens=5                  # DSpark active
GPU KV cache size: 1,652,056 tokens
```

### Measured (warm, 128 tok/req)

| concurrency | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| aggregate tok/s | 53.2 | 75.7 | 113.2 | 124.9 | **140.3** | 109.0 |

c6 falls below c3: `max_num_seqs 6` is the saturation point.

### Always launch via `scripts/spark-launch.sh`

```bash
./spark-launch.sh eugr-prod.yaml ~/eugr-prod.log
```

It tears down both nodes, sweeps `/dev/shm` on both, prints free memory, then
launches into a detached `screen`. The sweep is not optional: GB10 is unified
memory, so orphaned segments from a killed run eat the DRAM the GPU allocates its
KV pool from and silently lower the next boot's ceiling.

### Utilization ceiling

| util | tokens | outcome |
|---|---:|---|
| 0.85 | ~1.00M | fails, below the 11.04 GiB one 1M request needs |
| **0.89** | **1,652,056** | **serves** |
| 0.91 | 1,941,101 | allocates, worker then SIGKILLed with no error |
| 0.92 | n/a | fails: `Free memory on device cuda:0 (111.46/121.69 GiB)` |

At 0.91 the head logs the full pool and continues while the worker dies silently,
then the head hangs in `shm_broadcast`. The only evidence is in the worker's own
log inside its container (`/tmp/sparkrun_serve.log`), not the head log.

### Do not set `kv_cache_dtype: nvfp4_ds_mla` here

It is rejected twice, and the second rejection is real. See §4 of
[EUGR_B12X_PROD.md](../EUGR_B12X_PROD.md). `eugr-prod-nvfp4.yaml` is kept only as
a record of that attempt.


### Tuning: leave `max_num_seqs` at 6

| config | KV tokens | c3 | c5 |
|---|---:|---:|---:|
| **seqs 6, k=5 (shipped)** | **1,663,439** | **105.6** | **140.2** |
| seqs 8, k=5 | 1,638,922 | 102.7 | 129.4 |
| seqs 12, k=5 | 1,635,809 | 92.9 | 124.0 |

Raising it costs throughput *and* KV: 5 live streams never fill the extra slots.
k=3 is rejected on this image (`DSpark requires num_speculative_tokens >=
dspark_block_size (5)`), so k must be 5 or 10.

### Do not enable SSD KV offload on this image

`examples/eugr-prod-ssd.yaml` reproduces `CUDA error: an illegal memory access
was encountered` during startup, with and without the b12x attention backend. It
is kept only as a record. See EUGR_B12X_PROD.md §8.
