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
| **0.82** | **2,137,521** | **2.04×** | **fast (~4 min)** | **recommended — big pool, fast startup** |
| 0.85 | 2,769,487 | 2.64× | **~11+ min, stalls** | ❌ pool quantization/graph setup pathologically slow at startup |

**The 0.85 cliff:** the NVFP4 KV pool at 2.77M tokens takes 11+ min to quantize/capture at startup
(one rank pinned at GPU 96% on "kv cache quantization", the other waiting on `shm_broadcast`), which
makes every restart painful. **0.82 is the practical ceiling** — nearly the capacity, none of the
startup pain. Push toward 0.85 only if you (a) rarely restart and (b) can eat the ~11-min cold start.

Pool math for your workload: at util 0.82 (~2M tokens), 2-3 concurrent coding sessions of ~500-650K
each fit; sessions approaching a full 1M each will contend past ~2 concurrent (that's the `nvfp4_ds_mla`
+ `gpu-memory-utilization` ceiling — the only way past it on this hardware is expert-pruned weights,
see MODEL_VARIANTS REAP, or LMCache disk-spill, see UPSTREAM_GAPS #7).

## `max-num-seqs` → aggregate throughput (the real throughput lever, measured 2026-08)

The recipe ships `max-num-seqs 6`, tuned for a few large coding sessions. That value **caps aggregate
throughput** — the batch fills at c6 and queues beyond it. Raising it (util **0.82**, nvfp4 1M KV,
DSpark, coding prompt) scales aggregate ~linearly and **costs nothing at low concurrency** (single-stream
c1 is unchanged, ~53-62 tok/s):

| max-num-seqs | peak aggregate | at | single-stream (c1) | boots? |
|---|---:|---|---:|---|
| 6 | 159 tok/s | c6 | ~55 | ✅ |
| 16 | 293 tok/s | c16 | ~58 | ✅ |
| **32** | **421 tok/s** | **c32** | ~53 | ✅ (cudagraph capture ~55s, ~7 min warm start) |
| 48 | — | — | — | ❌ **hangs** at API-server handoff (`shm_broadcast` 60s stall; 48-seq cudagraph capture wedges the multi-node frontend) |

**`max-num-seqs 32` is the recommended setting** and strictly dominates 6: identical behavior when only
2-3 requests are active (KV is allocated per-*active*-request from the shared pool, not pre-reserved per
slot), but a **2.6× higher aggregate ceiling (159 → 421 tok/s)** under load. This **beats the old FP8
throughput-mode number (326 tok/s @ c48 on eugr-b12x)** while keeping nvfp4 **1M context + DSpark + full
per-stream speed** — so there is no longer a reason to switch to the FP8/no-spec image just for aggregate
throughput. **48 is the boot ceiling** on this 2× GB10 multi-node setup (frontend hang), so stay at 32.

## Single-stream decode: already at the ceiling (don't bother tuning)

The recipe author's exhaustive sweep + our re-check: **zero config wins.** Proven negatives — do not
re-test:

| lever | result |
|---|---|
| `num_speculative_tokens` (k) | **locked at 5** — 7 rejected at boot (must be multiple of n_predict=5), 10 crashes at runtime |
| `max-model-len` 1M → 200K | no gain |
| `max-num-seqs` 6 → 2 | no gain *for single-stream* (but it IS the aggregate-throughput lever — see next section) |
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

## Concurrency (your 2-3 chat load), coding prompt

| concurrency | per-stream tok/s (util 0.78) | per-stream tok/s (util 0.82, settled) |
|---|---|---|
| c1 | 41.2 | 53.7 |
| c2 | 39.5 | 44.5 |
| **c3** | **37.1** | **40.5** |
| c6 | 14.8 | 25.9 |

Per-stream barely drops c1→c3: your 2-3 concurrent coding chats each stay near single-stream speed.
util changes KV pool size, not decode speed — the spread between the two columns is run-to-run
acceptance variance (content-driven), not a util effect. It only collapses once you fill the batch
(c6 with seqs 6).

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
--max-num-seqs 32                    # aggregate-throughput lever: 6→32 = 159→421 tok/s peak, free at low-c (48 hangs)
--kv-cache-dtype nvfp4_ds_mla        # the 1M enabler
--max-model-len 1048576
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
```

Clock capped 2200 MHz (free — bandwidth-bound). For pure many-user throughput at ≤512K instead of 1M,
use the eugr-b12x FP8 no-spec path (326 tok/s @ c48) — different image, different tradeoff.

## fp8 vs nvfp4_ds_mla KV — measured head-to-head (2026-08-21)

Both booted on the **same** recipe (`deepseek-v4-flash-0731-dspark-arena-threshold`,
util **0.78**, `max-num-seqs 12`, DSpark **k=3**), changing only `kv_cache_dtype`.
Measured warm via `/v1/completions` with `ignore_eos:true`, 128 tokens/request.

| `kv_cache_dtype` | KV pool | tokens | bytes/token | c1 tok/s | c5 aggregate |
|---|---:|---:|---:|---:|---:|
| `fp8` | 12.27 GiB | **1,448,712** | ~9,094 | **58.3** | **162.5** |
| `nvfp4_ds_mla` | 11.46 GiB | 1,354,614 | ~9,083 | 51.1 | 159.2 |

**At this util/seqs combination NVFP4 KV is not a win** — near-identical bytes/token
(~9.1 KB), *fewer* total tokens (less memory left available), and ~12% slower
single-stream.

**Do not read this as "NVFP4 KV is worse."** The large NVFP4 pools documented above
(2.14M @ util 0.82, 2.77M @ util 0.85, allocated but not served) come from the **1M recipe**, which pairs
NVFP4 KV with `max-num-seqs 6` and `k=5`. Spec-decode buffers scale with
`max_num_seqs × (k+1)`, so at `max-num-seqs 12` those buffers consume the memory that
should have become KV. The KV dtype and the seqs/k choice have to be tuned together —
changing one alone gives a misleading result (this table is exactly that mistake,
kept here as the control).

## Aggregate throughput at low concurrency (2026-08-21)

Production config (`arena-threshold`: fp8 KV, util 0.78, `max-num-seqs 12`, k=3),
warm, `ignore_eos:true`, 128 tokens/request:

| concurrency | aggregate tok/s | per-request tok/s |
|---|---:|---:|
| 1 | 58.3 | 58.3 |
| 5 | **162.5** | ~32.5 |

Consistent with the `max-num-seqs` table above (seqs 6 → 159 tok/s at c6), and with an
independent community GB10 measurement of 61.5 tok/s single-stream text-only. For
higher aggregate, raise `max-num-seqs` — it scales roughly linearly and costs nothing
at c1.

## b12x sub-feature flags (2026-08-20) — all measured, none adopted

The stage-c image exposes more b12x toggles than the recipe enables. Tested on the
arena-threshold config; baseline 28-cell grid mean **53.57**, official arena decode
raw mean **44.75**.

| Flag | Result | Verdict |
|---|---|---|
| `VLLM_USE_B12X_MOE=1` | already on in the shipped recipe | **keep** |
| `VLLM_USE_B12X_WO_PROJECTION=1` | already on | **keep** |
| `VLLM_USE_B12X_MHC=1` | +1.9% local (inside noise); **official arena raw 37.95 vs 44.75** | reject — officially worse |
| `VLLM_USE_B12X_SPARSE_INDEXER=1` | −2.6%; collapses deep-context cells (the Lightning Indexer does most of its work at long context) | reject |
| `VLLM_USE_B12X_FP8_GEMM=1` | **crash**: `DeepGEMM/csrc/.../utils/layout.hpp:39: t.dim() == N` | reject |
| `VLLM_DSV4_B12X_COMPRESSED_MLA` | left at `0` in the recipe; same kernel family that hangs under the eugr b12x image | leave off |

Also measured and rejected:

| Change | Result |
|---|---|
| `num_speculative_tokens` 3 → 5 *(on the arena-threshold config)* | **−6.7%** (49.99 vs 53.57). Note the 1M recipe uses k=5 deliberately — k must be a multiple of `n_predict=5`, and it pairs with `max-num-seqs 6`. |
| `long_prefill_token_threshold` 1024 → 4096 | neutral |
| `--load-format fastsafetensors` | load ~3× faster (60 s vs 3m17s) but **OOM-kills the worker** on unified memory — unusable |
