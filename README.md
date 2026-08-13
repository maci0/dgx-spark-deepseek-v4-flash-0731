# DeepSeek-V4-Flash-0731 on 2× DGX Spark (GB10)

Serving **DeepSeek-V4-Flash-0731** on **two NVIDIA DGX Spark (GB10, sm_121a)** nodes over RoCE,
with everything actually measured on the hardware. This repo is the field notes: the config that
works, the full matrix of what was tried (and why most of it failed), the performance envelope, and
the open gaps that upstream vLLM/SGLang still need to close for this model on this hardware.

Scope is intentionally narrow: **this one checkpoint, this one hardware**. Not a general serving guide.

- **[TEST_LOG.md](TEST_LOG.md)** — the full quant × framework × image sweep, every result, verbatim errors.
- **[UPSTREAM_GAPS.md](UPSTREAM_GAPS.md)** — what's still broken/missing upstream, filed for maintainers.

---

## TL;DR — two configs that work

| Goal | Framework / image | Quant | Ctx | Spec | Measured |
|------|-------------------|-------|-----|------|----------|
| **Max context (1M)** | vLLM, tonyd2wild `dspark-nvfp4-stage-c` (bjk110 base) | FP8 weights + **NVFP4 KV** | **1,048,576** | DSpark k5 | ~37-41 tok/s/stream @ c1-3; KV pool up to 2.77M tokens |
| **Max throughput (≤512K)** | vLLM, eugr `spark-vllm-b12x` | FP8 (UE8M0) | 512K | off | **~326 tok/s @ c48** |

Everything else is worse or broken on this hardware — see the matrix.

**Hardware:** 2× GB10 (sm_121a, ~122 GB unified memory/node), 200 Gb/s RoCE (CX7) between nodes, TP=2.
GPU clock capped at **2200 MHz** (proven zero throughput loss, prevents thermal shutdown — the box is
bandwidth-bound, not clock-bound).

---

## The 1M recipe (NVFP4 KV + DSpark)

The only path to 1M context is **NVFP4 KV cache** (`--kv-cache-dtype nvfp4_ds_mla`), which needs a
DeepSeek-V4-specific padded-NVFP4 KV *writer* that **only exists in the tonyd2wild custom image** —
stock vLLM, eugr, and even the newer eugr-b12x all lack it (they have a GLM-only NVFP4-KV writer and
a 432-byte envelope that mismatches DeepSeek's sparse-MLA page; see UPSTREAM_GAPS).

Build (both nodes) from [tonyd2wild's recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark):

```
./build-dspark-vllm-runtime.sh          # base ghcr.io/bjk110/vllm-spark:unholy-fusion + overlay + stage-a/b/c
```

Key serve flags (via their `docker-compose.dspark.yml` + `.env.dspark`):

```
--kv-cache-dtype nvfp4_ds_mla --block-size 256
--max-model-len 1048576
--max-num-seqs 6
--max-num-batched-tokens 8192
--gpu-memory-utilization 0.85          # << see "Tuning" — biggest lever for concurrent large sessions
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
--distributed-executor-backend mp --nnodes 2
--tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice
```

Model: `deepseek-ai/DeepSeek-V4-Flash-0731` (official) or an FP8 abliterated variant
(e.g. `apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8`) — **abliteration is speed-neutral**.

RoCE env (per node): `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_GID_INDEX=3` (RoCE v2),
`VLLM_HOST_IP=<this node's fabric IP>` (must be per-node — see gotchas).

---

## Performance findings (all measured)

- **Single-stream decode is acceptance-driven, not config-tunable.** The recipe author's exhaustive
  sweep found **zero tuning wins**; `k` is locked at 5 (7 rejected at boot, 10 crashes at runtime).
  Decode ranges **~64-83 tok/s** on the *same server* purely by content type.
- **Coding content = high DSpark acceptance.** Measured **mean accepted length 4.26 / ~65% draft
  acceptance** on code (vs ~2.4 on math) → coding agents get ~37-41 tok/s/stream. Code is predictable,
  so speculation flies.
- **Concurrency barely degrades at low load:** c1=41, c2=40, c3=37 tok/s/stream (per stream). Your
  2-3 concurrent chats each stay near single-stream.
- **The one real lever = `gpu-memory-utilization` → KV pool size** (for concurrent *large* sessions):

  | util | KV pool (tokens) | concurrency @ 1M |
  |------|-----------------:|-----------------:|
  | 0.78 | 1,181,262 | 1.13× |
  | 0.85 | **2,769,487** | **2.64×** |

  2.3× more KV capacity for a small util bump — the thing that matters for coding agents whose
  sessions grow to 200-500K+ tokens. (Push higher with care: less capture/runtime headroom.)
- **FP8 512K throughput mode** (eugr-b12x, spec off, seqs 48): **~326 tok/s @ c48**, saturates ~48
  concurrent. Clock cap 2200 costs nothing (bandwidth-bound).

---

## Client integration (Kimi Code and other OpenAI-compatible harnesses)

This vLLM build returns reasoning under the **`reasoning`** field, **not** `reasoning_content`
(DeepSeek's hosted API uses `reasoning_content`). Harnesses that assume `reasoning_content` will
**leak `</think>` into displayed content**. Fix on the client:

- **Kimi Code** (`~/.kimi-code/config.toml`): set `reasoning_key = "reasoning"` on the local model
  entry (and `max_context_size = 1000000`). Without it, the think block bleeds into content.

(Server-side, tonyd2wild patch 0005 additionally guards against stop-strings decapitating reasoning
mid-`<think>` when harnesses send `stop` sequences — a separate but related null-content bug.)

---

## What does NOT work here (short list; full detail + errors in TEST_LOG)

- **SGLang:** loads DeepSeek-V4 on sm_121 but **cross-node TP2 NCCL collective #2 always drops** —
  every transport (RDMA/TCP), every knob. Not the fabric (vLLM runs cross-node fine). Model is 152GB
  so single-node isn't an option. Blocked.
- **NVFP4 *weights* on vLLM (neko/sakamakismile/RedHatAI/nvidia):** all fail — swiglu-clamp/cutlass-
  eager/`block_tables` for all-NVFP4, and compressed-tensors ≠ B12X native-FP8 kernels for RedHatAI.
  And NVFP4 weights don't even shrink the footprint (all ~156-168GB). The NVFP4 win is **KV**, not weights.
- **Stock vLLM images (latest/nightly/NGC):** run no-spec eager only (PR #41834 unmerged), ~+38% slower.
- **NVFP4 KV on eugr-b12x:** architecturally incomplete for DeepSeek-V4 (GLM-only writer; see gaps).

---

## Gotchas that cost real time

- **`restart: unless-stopped` causes a restart *loop*** on this multi-node mp setup: engine deaths that
  exit 0 + a capture-time cross-node collective wedge → docker auto-restarts into a deadlock. Use
  `restart: "no"` and a clean-restart procedure (down both → kill stray `vllm`/`EngineCore` procs →
  free the other node's GPU → single start). A reboot is only needed if state is truly wedged.
- **Orphaned `vllm`/`EngineCore` procs survive `docker compose down`** and hold the GPU + dist port
  25000 → next deploy deadlocks. `pkill -9` them before restart.
- **Other GPU tenants** (e.g. a `llama-server`/gpustack container auto-restarting on boot) silently
  contend for GB10's shared memory. Stop them before serving.
- **Baked-in per-node values** in some images (`VLLM_HOST_IP`, `--node-rank`, `NCCL_IB_HCA`) must be
  overridden per node or the cluster hangs silently at distributed init.

---

## Credits

- **[tonyd2wild](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)**
  — the 1M NVFP4-KV + DSpark runtime (the DeepSeek-V4 NVFP4-KV writer that makes 1M possible), and the
  cold-prefill garble (Patch 3) + shared-expert + stop-in-reasoning fixes.
- **eugr** `spark-vllm-b12x` — the B12X/sparkinfer sm_121 kernels + the proven FP8 512K throughput path.
- **bjk110** `vllm-spark:unholy-fusion` — the base image the 1M runtime builds on.

This repo just measures and documents; the hard runtime work is theirs.
