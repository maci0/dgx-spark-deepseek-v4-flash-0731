# DeepSeek-V4-Flash-0731 on 2× DGX Spark (GB10)

Serving **DeepSeek-V4-Flash-0731** on **two NVIDIA DGX Spark (GB10, sm_121a)** nodes over RoCE,
with everything actually measured on the hardware. This repo is the field notes: the config that
works, the full matrix of what was tried (and why most of it failed), the performance envelope, and
the open gaps that upstream vLLM/SGLang still need to close for this model on this hardware.

Scope is intentionally narrow: **this one checkpoint, this one hardware**. Not a general serving guide.

- **[TEST_LOG.md](TEST_LOG.md)**: the full quant × framework × image sweep, every result, verbatim errors.
- **[UPSTREAM_GAPS.md](UPSTREAM_GAPS.md)**: what's still broken/missing upstream, filed for maintainers.
- **[CLIENT_INTEGRATION.md](CLIENT_INTEGRATION.md)**: OpenAI-compat harness setup (Kimi Code, the `reasoning` field gotcha).
- **[MODEL_VARIANTS.md](MODEL_VARIANTS.md)**: which HF checkpoints fit this setup (abliterated FP8, REAP-pruned) + what to try next.
- **[TUNING.md](TUNING.md)**: the util→KV-pool lever (and the 0.85 startup cliff), single-stream ceiling, content-driven DSpark.
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: symptom → cause → fix table for every failure hit here.
- **[PRODUCTION.md](PRODUCTION.md)** ← **START HERE**: the shipped production config, every value justified by measurement, plus what this hardware cannot do (SSD offload, ~2.5M KV) and why.
- **[EUGR_B12X_PROD.md](EUGR_B12X_PROD.md)**: **the production path**: eugr b12x image, fp8_ds_mla KV, DSpark, 1.65M tokens, measured c1-c6. Includes why `nvfp4_ds_mla` is closed on this image.
- **[KV_CEILING.md](KV_CEILING.md)**: why ~2.5M KV is unreachable here. The full measured ladder (17.8 → 26.3 GiB arenas, all of which allocate, none above 17.8 of which run), the `--kv-cache-memory-bytes` semantics that make it counterintuitive, and the `docker rm -f` teardown trap that invalidated four runs.
- **[PROD_C5_SSD.md](PROD_C5_SSD.md)**: production config: 5 clients, param minimization, node housekeeping. **Two claims in it are now disproven by testing:** SSD KV offload does *not* work (it faults under every KV dtype, see [KV_OFFLOAD_MLA.md](https://github.com/maci0/vllm-spark-nvfp4/blob/main/KV_OFFLOAD_MLA.md) §"Tested end to end"), and its NVFP4 KV pool dies in warmup.
- **[examples/.env.dspark.example](examples/.env.dspark.example)** · **[scripts/clean-restart.sh](scripts/clean-restart.sh)** · **[scripts/bench.py](scripts/bench.py)**

## Topology

```
        ┌─────────────────────────┐   200 Gb/s RoCE (CX7)   ┌─────────────────────────┐
        │  spark1 (HEAD, rank 0)  │ ══════════════════════ │  spark2 (WORKER, rank 1)│
        │  GB10 sm_121a, ~122 GB  │   NCCL_IB_HCA / GID 3   │  GB10 sm_121a, ~122 GB  │
        │  fabric 10.0.1.1        │   dist init :25000      │  fabric 10.0.1.2        │
        │  serves :8000  ◄────────┼─ clients (Kimi, curl)   │  headless               │
        └─────────────────────────┘                         └─────────────────────────┘
                 TP=2, --distributed-executor-backend mp --nnodes 2
        152 GB model split ~76 GB/node · largest KV pool that has served: 1.46M tokens · clock capped 2200 MHz
```

---

## TL;DR: the shipped prod config, plus history

| Goal | Framework / image | Quant | Ctx | Spec | Measured |
|------|-------------------|-------|-----|------|----------|
| **Max context (1M)** | vLLM, tonyd2wild `dspark-nvfp4-stage-c` (bjk110 base) | FP8 weights + **NVFP4 KV** | **1,048,576** | DSpark k5 | ~37-41 tok/s/stream @ c1-3; KV pools >1.5M allocate but have not reached serving |
| **Max throughput (≤512K)** | vLLM, eugr `spark-vllm-b12x` | FP8 (UE8M0) | 512K | off | **~326 tok/s @ c48** |
| **Prod, 3-5 sessions (fastest)** | vLLM, eugr `dgx-vllm-eugr-nightly-b12x:2026081903` | FP8 weights + **fp8_ds_mla KV** | 1M | DSpark k=5 | **1,674,044-token KV, 135.0 tok/s agg @ c5**, b12x MoE, zero patches, 73-line recipe |
| **Prod, 3-5 sessions (max context)** | vLLM, tonyd2wild `dspark-nvfp4-stage-c` | FP8 weights + **NVFP4 KV** | 1M | DSpark k=5 | **2,198,373-token KV** (+31%), 109.7 tok/s agg @ c5 (-19%). Only route past ~1.7M |
| **Prod, 5 clients + SSD KV spill** | vLLM, tonyd2wild `dspark-nvfp4-stage-c` | FP8 weights + **NVFP4 KV** | 1M | DSpark k=5 | superseded by the two rows above; kept for the SSD-offload findings. See [PROD_C5_SSD.md](PROD_C5_SSD.md) |

Everything else is worse or broken on this hardware, see the matrix.

**Hardware:** 2× GB10 (sm_121a, ~122 GB unified memory/node), 200 Gb/s RoCE (CX7) between nodes, TP=2.
GPU clock capped at **2200 MHz** (proven zero throughput loss, prevents thermal shutdown, the box is
bandwidth-bound, not clock-bound).

---

## The 1M recipe (NVFP4 KV + DSpark)

The only path to 1M context is **NVFP4 KV cache** (`--kv-cache-dtype nvfp4_ds_mla`), which needs a
DeepSeek-V4-specific padded-NVFP4 KV *writer* that **only exists in the tonyd2wild custom image** , 
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
--max-num-seqs 32                      # << aggregate-throughput lever: 6→32 lifts peak 159→421 tok/s, free at low concurrency (48 hangs on 2-node)
--max-num-batched-tokens 8192
--gpu-memory-utilization 0.85          # << see "Tuning", biggest lever for concurrent large sessions
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
--distributed-executor-backend mp --nnodes 2
--tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice
```

Model: `deepseek-ai/DeepSeek-V4-Flash-0731` (official) or an FP8 abliterated variant
(e.g. `apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8`), **abliteration is speed-neutral**.

RoCE env (per node): `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_GID_INDEX=3` (RoCE v2),
`VLLM_HOST_IP=<this node's fabric IP>` (must be per-node, see gotchas).

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

  2.3× more KV capacity for a small util bump, the thing that matters for coding agents whose
  sessions grow to 200-500K+ tokens. (Push higher with care: less capture/runtime headroom.)
- **FP8 512K throughput mode** (eugr-b12x, spec off, seqs 48): **~326 tok/s @ c48**, saturates ~48
  concurrent. Clock cap 2200 costs nothing (bandwidth-bound).
- **Low-concurrency aggregate, arena-threshold config** (fp8 KV, util 0.78, seqs 12, k=3; warm,
  `ignore_eos`, 128 tok/req): **c1 = 58.3 tok/s, c5 = 162.5 tok/s aggregate** (~32.5/stream).
  Matches an independent community GB10 figure of 61.5 tok/s single-stream text-only.
- **KV dtype must be tuned *together with* `max-num-seqs`/`k`.** Swapping `fp8` → `nvfp4_ds_mla` while
  holding util 0.78 / seqs 12 / k=3 makes things *worse* (1.35M vs 1.45M tokens, c1 51.1 vs 58.3):
  spec-decode buffers scale with `max_num_seqs × (k+1)` and eat the memory that should become KV. The
  big NVFP4 pools (2.14M @ 0.82, 2.77M @ 0.85) come from the 1M recipe's **seqs 6 + k=5** pairing.
  Full head-to-head in [TUNING.md](TUNING.md).
- **b12x sub-flags beyond the shipped two are all rejected:** `MHC` is *officially worse* (arena raw
  37.95 vs 44.75), `SPARSE_INDEXER` −2.6%, `FP8_GEMM` crashes (`DeepGEMM layout.hpp:39`). See
  [TUNING.md](TUNING.md).
- **SGLang + b12x is impossible on GB10 today**: the community b12x SGLang images are `linux/amd64`
  and GB10 is `arm64`; the arm64 SGLang build has no b12x, and b12x 1.2.3 removed the generic
  integration API a port would need. SGLang *without* b12x also fails 2-node (worker dies during
  FlashInfer autotune). Detail in [TEST_LOG.md](TEST_LOG.md).

---

## Client integration (Kimi Code and other OpenAI-compatible harnesses)

This vLLM build returns reasoning under the **`reasoning`** field, **not** `reasoning_content`
(DeepSeek's hosted API uses `reasoning_content`). Harnesses that assume `reasoning_content` will
**leak `</think>` into displayed content**. Fix on the client:

- **Kimi Code** (`~/.kimi-code/config.toml`): set `reasoning_key = "reasoning"` on the local model
  entry (and `max_context_size = 1000000`). Without it, the think block bleeds into content.

(Server-side, tonyd2wild patch 0005 additionally guards against stop-strings decapitating reasoning
mid-`<think>` when harnesses send `stop` sequences, a separate but related null-content bug.)

---

## What does NOT work here (short list; full detail + errors in TEST_LOG)

- **SGLang:** older builds dropped the cross-node collective at boot. **0.5.17 (Aug 2026) now boots the
  2-node path** (2.04M KV pool, server ready, trivial gen works), but **real generation hangs the worker**
  (`Scheduler watchdog timeout 300s` on TP1 → death). The instability moved from boot to decode; still
  not viable. Only `lmsysorg/sglang:latest` has DeepSeek-V4 at all. See UPSTREAM_GAPS #5.
- **NVFP4 *weights* on vLLM (neko/sakamakismile/RedHatAI/nvidia):** all fail, swiglu-clamp/cutlass-
  eager/`block_tables` for all-NVFP4, and compressed-tensors ≠ B12X native-FP8 kernels for RedHatAI.
  And NVFP4 weights don't even shrink the footprint (all ~156-168GB). The NVFP4 win is **KV**, not weights.
- **Stock vLLM images (latest/nightly/NGC):** run no-spec eager only (PR #41834 unmerged), ~+38% slower.
- **NVFP4 KV on eugr-b12x:** architecturally incomplete for DeepSeek-V4 (GLM-only writer; see gaps).
- **TokenSpeed (LightSeek) engine:** **builds + boots on GB10** (strip the Kimi-K3 `attn_res` tcgen05
  kernel from setup.py → compiles for `12.1a`; runs portable `--attention-backend triton --moe-backend
  flashinfer_cutlass`; clears distributed init + MoE-select on 2× GB10). But it **wedges at weight-load**:
  the loader puts the ~80 GB/node skeleton on the GPU then reads 156 GB of shards, and on GB10's shared
  122 GB the GPU skeleton + shard page cache collide (~160 GB) → OOM/hard-reboot. Fixing it needs
  root-level page-cache control (`drop_caches` during load); util/cgroup/watchdog levers don't bound it.
  tcgen05 is only the Kimi/MiniMax kernels, **not** DeepSeek-V4. See UPSTREAM_GAPS #9.

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

## Verify + benchmark

```bash
# health + confirm 1M context
curl -s http://HEAD_IP:8000/v1/models | python3 -c 'import sys,json;m=json.load(sys.stdin)["data"][0];print(m["id"],m.get("max_model_len"))'
# smoke
curl -s http://HEAD_IP:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":32}'
# concurrency benchmark (per-stream + aggregate at c1/c2/c3/c6)
BASE=http://HEAD_IP:8000/v1 uv run --with aiohttp python3 scripts/bench.py 1 2 3 6
```

`scripts/bench.py` defaults to a coding prompt (high DSpark acceptance). Change `PROMPT=` to see the
content-driven spread, the same server does ~83 tok/s on counting and ~64 on a BST implementation.

## Versions pinned (what these numbers were measured on)

| component | value |
|---|---|
| Hardware | 2× NVIDIA DGX Spark (GB10, sm_121a), 200 Gb/s CX7 RoCE, TP=2 |
| Runtime image | `vllm-dspark-runtime:dspark-nvfp4-stage-c` (tonyd2wild), base `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` |
| vLLM | `0.21.1rc1.dev339+g1967a5627bc3` |
| Throughput image | `eugr/spark-vllm-b12x:latest` (vLLM main + B12X sm_121 kernels) |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` / `apetersson/...-Abliterated-FP8` (FP8 e4m3, 256 experts, 167 GB) |
| KV / spec | `nvfp4_ds_mla` KV · DSpark k=5 (locked; multiple of n_predict=5) |
| Measured | 2026-08 |

## Credits

- **[tonyd2wild](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)**
 , the 1M NVFP4-KV + DSpark runtime (the DeepSeek-V4 NVFP4-KV writer that makes 1M possible), and the
  cold-prefill garble (Patch 3) + shared-expert + stop-in-reasoning fixes.
- **eugr** `spark-vllm-b12x`: the B12X/sparkinfer sm_121 kernels + the proven FP8 512K throughput path.
- **bjk110** `vllm-spark:unholy-fusion`: the base image the 1M runtime builds on.

This repo just measures and documents; the hard runtime work is theirs.
