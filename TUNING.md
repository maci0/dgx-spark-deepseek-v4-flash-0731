# Tuning — what moves the needle (and what doesn't)

Measured on 2× GB10, tonyd2wild `dspark-nvfp4-stage-c`, `deepseek-ai`/`apetersson` FP8, 1M ctx,
NVFP4 KV, DSpark k=5. See [scripts/bench.py](scripts/bench.py) to reproduce.

## The one real lever: `gpu-memory-utilization` → KV pool size

Weights are fixed (~167 GB / ~83.5 GB per node). Everything above that becomes KV pool. Since the
weights sit just under the budget, small util bumps produce large pool changes — and this is the lever
that matters for **concurrent large coding sessions** (200-500K tokens each).

| util | KV pool (tokens) | concurrency @ 1M | startup | verdict |
|------|-----------------:|-----------------:|---------|---------|
| 0.78 | 1,181,262 | 1.13× | fast (~3 min) | too tight for 3 large sessions |
| 0.80 | ~1.65–1.87M | ~1.6–1.8× | fast | good |
| **0.82** | **~2M (measuring)** | **~2×** | **fast** | **recommended — big pool, fast startup** |
| 0.85 | 2,769,487 | 2.64× | **~11+ min, stalls** | ❌ pool quantization/graph setup pathologically slow at startup |

**The 0.85 cliff:** the NVFP4 KV pool at 2.77M tokens takes 11+ min to quantize/capture at startup
(one rank pinned at GPU 96% on "kv cache quantization", the other waiting on `shm_broadcast`), which
makes every restart painful. **0.82 is the practical ceiling** — nearly the capacity, none of the
startup pain. Push toward 0.85 only if you (a) rarely restart and (b) can eat the ~11-min cold start.

Pool math for your workload: at util 0.82 (~2M tokens), 2-3 concurrent coding sessions of ~500-650K
each fit; sessions approaching a full 1M each will contend past ~2 concurrent (that's the `nvfp4_ds_mla`
+ `gpu-memory-utilization` ceiling — the only way past it on this hardware is expert-pruned weights,
see MODEL_VARIANTS REAP, or LMCache disk-spill, see UPSTREAM_GAPS #7).

## Single-stream decode: already at the ceiling (don't bother tuning)

The recipe author's exhaustive sweep + our re-check: **zero config wins.** Proven negatives — do not
re-test:

| lever | result |
|---|---|
| `num_speculative_tokens` (k) | **locked at 5** — 7 rejected at boot (must be multiple of n_predict=5), 10 crashes at runtime |
| `max-model-len` 1M → 200K | no gain |
| `max-num-seqs` 6 → 2 | no gain |
| `--max-cudagraph-capture-size 36` | no gain |
| util (for *speed*) | no effect (only changes pool size) |

Decode is **acceptance-driven**, and acceptance is **content-driven** — the same server does ~83 tok/s
on counting and ~64 on a BST implementation. Any single number without the workload is meaningless.

## What content does to DSpark (measured)

| content | mean accepted length | draft acceptance | effect |
|---|---|---|---|
| code (BST, functions) | **4.26** | ~65% | fast — coding agents win big |
| math (primality) | ~2.4 | ~28% | slower |

Coding is predictable → speculation flies. This is why an agentic coding client is the best-case
workload for this serve.

## Concurrency (your 2-3 chat load), util 0.78 baseline, coding prompt

| concurrency | per-stream tok/s | aggregate |
|---|---|---|
| c1 | 41.2 | 41 |
| c2 | 39.5 | 79 |
| **c3** | **37.1** | 111 |
| c6 | 14.8 | 89 |

Per-stream barely drops c1→c3 (41→37): your 2-3 concurrent coding chats each stay near single-stream
speed. It only collapses once you fill the batch (c6 with seqs 6). Re-measure at util 0.82 — speed is
unchanged by util, only capacity grows.

## Load time (subsequent restarts)

VRAM does not survive process exit (CUDA context destroyed) — every restart reloads weights. But:
- **Warm reload weight read ≈ 36 s** (OS page cache hot), vs ~4 min cold first load.
- The **compile/capture cache persists** (`VLLM_CACHE_ROOT`) → "Directly load AOT compilation from
  cache"; the torch.compile phase is already fast on restart.
- `fastsafetensors` (0.3.2, present) could parallelize the FP8 unpack — the recipe uses plain
  `safetensors`; marginal on warm reloads, bigger on cold.
- Net: warm restarts are dominated by warmup + KV-pool setup, not the weight read — and KV-pool setup
  is exactly what balloons at util 0.85 (the cliff above).

## Recommended config

```
--gpu-memory-utilization 0.82        # big KV pool, fast startup (not 0.85 — startup cliff)
--max-num-seqs 6                     # >=3 for your load; higher doesn't help single/low-c
--kv-cache-dtype nvfp4_ds_mla        # the 1M enabler
--max-model-len 1048576
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
```

Clock capped 2200 MHz (free — bandwidth-bound). For pure many-user throughput at ≤512K instead of 1M,
use the eugr-b12x FP8 no-spec path (326 tok/s @ c48) — different image, different tradeoff.
