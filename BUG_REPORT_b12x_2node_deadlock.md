# b12x DSpark 2-node deadlock: `shm_broadcast` stall during cross-node collectives (Aug18 and Aug19 nightlies)

## Summary

On a 2-node DGX Spark (GB10 / sm_121a) cluster with TP=2 across nodes, the b12x
fast-kernel DSpark speculative-decode path deadlocks on cross-node collectives.
The worker engine stalls and the head logs a repeating
`shm_broadcast.py:801 No available shared memory broadcast block found in 60 seconds`,
GPU busy-spins at ~96% with zero forward progress. This reproduces identically on
two consecutive nightlies (`2026081802` and `2026081902`), so it is not a
per-build regression but a fundamental incompatibility of the b12x path with the
2-node collective on this setup.

Because DeepSeek-V4-Flash-0731 is a 153-164 GB checkpoint (every variant) and a
single GB10 has 121 GB, the model **cannot** be served on one node; 2-node TP=2 is
mandatory. That makes this deadlock a hard blocker: the fast path cannot be used
for this model on this hardware at all.

## Environment

- Hardware: 2x DGX Spark, GB10, unified memory ~121 GB usable each, sm_121a.
- Interconnect: RoCE fabric (enp1s0f1np1, HCA rocep1s0f1); rendezvous over mgmt
  interface, master_port 25000.
- Image: `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:2026081902` (also
  reproduced on `:2026081802`).
- vLLM: 0.21.1rc1.dev339 line for the working stage-c baseline; the eugr nightly
  ships the b12x build (VLLM_USE_V2_MODEL_RUNNER=1 required to boot).
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731` (43 layers, 256 routed experts,
  6/tok, expert_dtype fp4, quantization fp8), TP=2.
- Launcher: sparkrun 0.3.5, 2-node cluster.

## Serve config (the fast path)

```
--tensor-parallel-size 2
--kv-cache-dtype fp8
--load-format instanttensor
--moe-backend b12x --linear-backend b12x
--attention-backend B12X_MLA_SPARSE
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","attention_backend":"B12X_MLA_SPARSE"}'
env: VLLM_USE_V2_MODEL_RUNNER=1, VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
```

## Two distinct manifestations of the same deadlock

### Aug18 (`2026081802`): hangs at inference on MIXED prefill+decode+spec batches
- Boots fully, becomes healthy.
- Single-stream and up-to-4 concurrent long-context requests: fast (~3 s,
  ~58 tok/s single stream).
- The moment the workload QUEUES (more requests than `max_num_seqs`, so finished
  decodes are replaced by fresh prefills -> a batch containing prefill + decode +
  spec simultaneously), the 2-node spec-decode step hangs. `sample_tokens` RPC /
  cross-node `shm_broadcast` stalls 3+ minutes, `kv_cache_usage` stays ~0.01 (NOT
  OOM), then EngineCore dumps its input and dies.
- Isolation: exactly-4-concurrent with no queue = OK; 10-concurrent (which queues
  down to 4 running) = hang. The variable is queuing -> mixed batches, not the
  concurrency number itself. No `max_num_seqs` value (4/8/10), timeout, chunked-
  prefill toggle, or thinking on/off avoids it, because the arena workload always
  queues.

### Aug19 (`2026081902`): hangs earlier, during FULL cudagraph capture
- Boots further than a no-spec variant (clears PIECEWISE profiling 11/11 and the
  DSpark draft load) but then deadlocks in FULL cudagraph capture, freezing at
  `Capturing CUDA graphs (FULL): 20%|██ | 2/10`.
- Head engine log, repeating every 60 s:
  ```
  (EngineCore) INFO shm_broadcast.py:801 No available shared memory broadcast block
  found in 60 seconds. This typically happens when some processes are hanging or
  doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
  ```
- GPU pinned at 96% util, zero progress. Same cross-node shm_broadcast stall as
  Aug18, just triggered during capture instead of inference.

## Why it's the same root cause

Both are a cross-node collective (`shm_broadcast`) failing to make progress when
the b12x path drives it: Aug18 during a mixed spec-decode step, Aug19 during FULL
cudagraph capture of that same path. The stage-c (non-b12x, vLLM 0.21) DSpark path
with identical 2-node TP=2 does NOT exhibit this: it captures FULL cudagraphs and
runs the full arena grid (concurrency 1/2/5/10, depths to 100k) without stalling.
So the deadlock is specific to the b12x kernels' interaction with the 2-node
collective, not to 2-node TP=2 in general.

## Reproduction

1. 2x GB10, TP=2, image `:2026081902`, config above.
2. Serve `deepseek-ai/DeepSeek-V4-Flash-0731`.
3. Aug19: boot alone reproduces (hangs at FULL cudagraph capture 2/10).
   Aug18: boot succeeds; drive >4 concurrent long-context (e.g. depth 8k-32k,
   concurrency 10) so requests queue -> hang within the first mixed batch.

## Impact / ask

- The b12x fast path is unusable for DeepSeek-V4-Flash-0731 on 2-node GB10, which
  is the only viable topology for this model (153-164 GB > 121 GB/node).
- Request: investigate the `shm_broadcast` progress stall in the b12x path's
  cross-node collective (both the spec-decode `sample_tokens` step and FULL
  cudagraph capture). A working 2-node b12x path would materially raise achievable
  throughput vs the stage-c fallback.

## Contrast: the config that DOES work on this hardware

vLLM 0.21 stage-c (bjk110 unholy-fusion base + DSpark overlay), 2-node TP=2,
`kv_cache_dtype fp8`, DSpark `num_speculative_tokens=3`,
`long_prefill_token_threshold=1024`, non-b12x backends. Boots clean, survives the
full arena concurrency/depth grid. This is the current production/record config.
