# DeepSeek-V4-Flash-0731 on 2× DGX Spark (GB10 / sm_121): Quant × Framework × Image Sweep

Goal: find the best-performing way to serve DeepSeek-V4-Flash-0731 on 2× GB10, across
model variants (official / abliterated / NVFP4 / FP8-repack), frameworks (vLLM / SGLang),
container images (eugr B12X, eugr, official vLLM, vLLM nightly, NVIDIA NGC, SGLang), and
strategies (DSpark spec on/off, latency vs throughput).

Hardware: spark1 (192.168.0.211, RoCE 10.0.1.1, head) + spark2 (192.168.0.212, RoCE 10.0.1.2,
worker). GB10, sm_121, ~122 GB unified mem/node. TP2 over RoCE (RDMA HCAs rocep1s0f1,
roceP2p1s0f1 on enp1s0f1np1). Clock: **capped 2200 MHz** (proven zero throughput loss,
thermal safety). Common sweep ctx = 262144 (256K) unless noted.

Bench: fixed 256-token completions, aggregate decode tok/s at concurrency c1/c8/c24/c48
(`sweep.py`), thinking-off where model supports it.

---

## ⭐ 1M CONTEXT — ACHIEVED (final live serve)

`apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8` (uncensored) + **NVFP4 KV cache** + DSpark k5,
**1,048,576 context** on 2× GB10, via the tonyd2wild `dspark-nvfp4-stage-c` image (bjk110 base).
Endpoint `http://192.168.0.211:8000/v1`, model `deepseek-v4-flash`. KV pool 1.69M tokens. 46-47°C.

Perf (DSpark, seqs 12): **c1=36, c4=73, c8=123, c12=162 tok/s**; DSpark mean accept-len ~2.4-2.7,
first-pos ~68%. Official vs abliterated identical (abliteration speed-neutral). Recipe above (E2).

Trade vs the FP8-512K eugr serve: 1M ctx + DSpark latency, but on a custom image (bjk110 base +
tonyd2wild overlays). For pure throughput/many-clients at ≤512K, eugr FP8 no-spec (326 @c48) is faster.

## TL;DR VERDICT

**Best way to serve DeepSeek-V4-Flash-0731 on 2× GB10: vLLM on `eugr/spark-vllm-b12x`, FP8
(official `deepseek-ai/...` or abliterated `apetersson/...`, identical speed), TP2 over RoCE.**
Pick the mode:
- **Throughput / many clients** — spec OFF, `--max-num-seqs 48` → **~326 tok/s @ c48** (256K ctx).
- **Latency / interactive** — DSpark ON (`num_spec 5`), `--max-num-seqs 8` → **37 tok/s single, 106 @c16**.

Everything else is worse or broken on this hardware:
- **SGLang: ❌ blocked** — supports DeepSeek-V4 on sm_121 (loads, kernels, KV all fine) but its
  **cross-node TP2 NCCL collective #2 always drops the inter-node connection** (RDMA + TCP, all knobs
  tried). vLLM runs cross-node on the same fabric, so it's SGLang-specific. Model too big (152GB) for
  single-node. Deep NCCL/SGLang-TP issue; not cracked. (Full diagnosis in the SGLang section.)
- **NVFP4 on vLLM: ❌** — 3 independent walls (swiglu-clamp backend rejection, cutlass eager-only,
  `block_tables` init-order); non-viable. (Qwen3.6 NVFP4 works via Marlin — DeepSeek-specific bug.)
- **Stock vLLM images (latest/nightly/NGC): ⚠️** — run no-spec eager only (PR #41834 unmerged),
  ~+38% slower single-stream, no DSpark. eugr-b12x is the only image with the sm_121 spec+cudagraph patch.
- **Abliteration: speed-neutral** (official 326 = abliterated 317-324 @ c48).
- **Clock cap 2200 MHz: free** (bandwidth-bound; capped ≈ uncapped).

---

## Inventory (present on nodes)

Models cached (both nodes, ~156 GB each unless noted):
- `deepseek-ai/DeepSeek-V4-Flash-0731` — **official**, UE8M0 FP8 mixed, bundled DSpark head
- `apetersson/DeepSeek-V4-Flash-0731-Abliterated-FP8` — abliterated FP8 (proven)
- `neko-legends/DeepSeek-V4-Flash-0731-Abliterated-NVFP4` — abliterated NVFP4 (real GB10 calib)
- `sakamakismile/DeepSeek-V4-Flash-0731-Abliterated-NVFP4` — abliterated NVFP4 (synthetic scales)
- `RedHatAI/DeepSeek-V4-Flash-speculator.dflash` — standalone DFlash draft/speculator

Not yet pulled (for SGLang branch, pull only if SGLang runs on sm_121):
- `nvidia/DeepSeek-V4-Flash-NVFP4` — official NVFP4 (FP4 MoE + FP8 attn); SGLang needs
  `flashinfer_trtllm_routed`, **doc says Blackwell SM100+** (GB10 = sm_121, uncertain)
- `sgl-project/DeepSeek-V4-Flash-FP8` — SGLang FP8 repack (DP-attn + DeepEP)

Images present:
- `eugr/spark-vllm-b12x:latest` — vLLM main + B12X/sparkinfer sm_121 kernels (PROVEN)
- `eugr/spark-vllm:latest` — eugr non-B12X base
- `vllm/vllm-openai:latest` — official stable
- `vllm/vllm-openai:v0.23.0 / v0.20.x / ...` — pinned
- `lmsysorg/sglang:v0.5.10.post1-cu130-runtime` — SGLang
- `nvcr.io/nvidia/vllm:26.03-py3`, `nvcr.io/nvidia/sglang:26.03-py3` — NGC builds

Disk: 1.7 TB free.

---

## Established results (prior work this session, vLLM eugr-b12x)

All on `eugr/spark-vllm-b12x`, TP2, fp8 KV, block 256, B12X_MLA_SPARSE attn, b12x MoE+linear,
FULL_AND_PIECEWISE cudagraphs. Clock effect: **capped 2200 ≈ uncapped** (bandwidth-bound;
317 vs 324 @ c48, within variance) → cap is free.

| ID | Model | Quant | Spec | ctx | seqs | c1 | c8 | c24 | c48 | Notes |
|----|-------|-------|------|-----|------|----|----|-----|-----|-------|
| A1 | apetersson abl | FP8 | DSpark | 512K | 8 | 37 | — | — | — | latency mode; first abliterated+spec working on GB10 |
| A2 | apetersson abl | FP8 | off | 512K | 48 | 28 | 109/114 | 190/216 | **317/324** | throughput champ (uncapped/capped) |

NVFP4 on vLLM eugr — **all paths blocked** (documented in `DEEPSEEK_V4_NVFP4_B12X_PATCH.md`):
- `flashinfer_b12x` MoE → swiglu clamp only for SWIGLUOAI_UNINTERLEAVE; DeepSeek SILU rejected
  at kernel (`FlashInferB12xExperts only applies swiglu_limit with swigluoai_uninterleave`).
- `flashinfer_cutlass` (sm120 kernel exists) → only correct clamp backend, but **eager-only**,
  and every warmup dummy-run (cudagraph capture / flashinfer autotune / mem-profile-with-attn)
  hits `AttributeError: GPUModelRunner has no attribute block_tables` — NVFP4 runner sets
  `block_tables` too late (init-ordering, neko's undocumented patch). Non-viable on this image.
- `flashinfer_trtllm` fp4 = `gen_trtllm_gen_fused_moe_sm100_module` → **sm100-only**, not GB10.
- Verdict: **NVFP4 DeepSeek-V4 does not run on vLLM eugr-b12x.** (Contrast: Qwen3.6 NVFP4
  runs fine via Marlin — the block_tables bug is DeepSeek-B12X-runner-specific.)

Newer eugr image (35h build): my swiglu-clamp adapter patch is partly upstreamed
(`NVFP4_BACKENDS_WITH_CLAMP.add(FLASHINFER_B12X)` gated on `activation==SWIGLUOAI_UNINTERLEAVE`
+ `has_flashinfer_b12x_moe_activation()`), but the gate excludes DeepSeek's SILU → no change
for this model.

---

## Test matrix (this sweep)

Legend: ✅ works+measured · ⚠️ works with caveat · ❌ fails (root cause logged) · ⏳ pending · ⛔ skipped (infeasible)

### vLLM — eugr/spark-vllm-b12x
| ID | Model | Quant | Spec | Status | Peak tok/s | Notes |
|----|-------|-------|------|--------|-----------|-------|
| B1 | deepseek-ai official | FP8 | off | ✅ | **326 @c48** | non-abliterated baseline; c1=28 c8=114 c24=196. = abliterated → abliteration speed-neutral |
| B2 | deepseek-ai official | FP8 | DSpark | ✅ | 37 c1 / 106 @c16 | latency mode seqs 8; accept len 2.80; from prior verified work (`DEEPSEEK_V4_SPECULATIVE_B12X.md`). Config: `method:dspark num_spec:5 probabilistic B12X_MLA_SPARSE`. Spec+seqs48 = 9.8 c1 ❌ (worst of both) |
| B3 | apetersson abl | FP8 | off | ✅ | 317-324 @c48 | = A2 (reference) |

### vLLM — other images
| ID | Image | Model | Status | Notes |
|----|-------|-------|--------|-------|
| C1 | vllm/vllm-openai:latest | deepseek-ai FP8 | ⚠️ | **works no-spec only** (from `DEEPSEEK_V4_2NODE_SERVE.md`): MLA `fp8_ds_mla` auto, needs `VLLM_USE_DEEP_GEMM=1` (UE8M0), `--all2all-backend deepep_low_latency`, **`--enforce-eager`** (DSA/MLA custom kernels), needs `ray` via beam mount. **No spec-decode** (PR #41834 unmerged → dspark dies `sparse_mla_sm120 num_tokens>64`). B12X gives **+38% single-stream** + spec + cudagraphs over this. |
| C2 | vllm/vllm-openai:nightly | deepseek-ai FP8 | ⚠️ | same as C1 — PR #41834 (SM12x sparse-MLA+DSpark) unmerged in main → absent from nightly too. No spec. Pulled (image ready) but not separately swept (identical stock limitation). |
| C3 | nvcr.io/nvidia/vllm:26.03 | deepseek-ai FP8 | ⏳ | NGC build; not swept (deprioritized after SGLang ate the session; expected stock-like, no B12X kernels) |
| C4 | eugr/spark-vllm (non-b12x) | deepseek-ai FP8 | ⏳ | base eugr; the recipe container. Not separately swept (b12x is a superset used for all B-runs) |

**vLLM image conclusion:** only **eugr/spark-vllm-b12x** carries the SM12x sparse-MLA-decode +
DSpark patch (PR #41834, unmerged upstream) → the only image with spec-decode + cudagraphs on GB10.
Stock `vllm-openai` (latest/nightly) runs the model **no-spec, eager only** (~+38% slower single-stream,
no DSpark). B12X is the definitive vLLM image for this model on GB10.

### SGLang — lmsysorg/sglang + NGC
| ID | Image | Model | Quant | Spec | Status | Notes |
|----|-------|-------|-------|------|--------|-------|
| S0 | lmsysorg/sglang v0.5.10 | deepseek-ai official | FP8 | — | ❌ | **`deepseek_v4` unrecognized** — image's Transformers too old (`ValueError: Transformers does not recognize deepseek_v4`). Need SGLang v0.5.16+. Pulling latest. |
| S0b | lmsysorg/sglang:latest | deepseek-ai official | mxfp4 | — | ⏳ | v0.5.10.post→latest (28.4GB, 2d old). **deepseek_v4 recognized** (`DeepseekV4ForCausalLM`, dsv4 attn backend, KV fp8_e4m3, `is_fp4_experts=True`). Reached `Init torch distributed` (2-node NCCL). Awaiting load/kernel/generate verdict on sm_121. |

**SGLang latest KEY facts (from S0b boot):** official model MoE experts are **FP4** (`is_fp4_experts=True`)
→ cookbook's `flashinfer_mxfp4` is correct. dsv4 attention backend auto-selected, page_size 256, KV
fp8_e4m3, hybrid SWA, `max_running_requests` auto=256. Breakable CUDA graph disabled for DSv4 (capture
memory). `sglang serve`/`launch_server` both present. DSpark: check `--speculative-algorithm` on latest.

**S0b attempt 1 (RDMA):** loaded weights + kernels OK, reached CUDA graph capture, then died at
cross-node NCCL allreduce: `IBV_WC_RETRY_EXC_ERR(12)` on HCAs rocep1s0f1/roceP2p1s0f1 (NCCL 2.28.9,
inside `capture_cuda_graphs`). RDMA transport failure, NOT a model/sm_121 kernel problem — DeepSeek-V4
kernels ran fine. vLLM TP2 ran over the same RoCE fabric OK, so SGLang's NCCL+cudagraph-capture path
trips RDMA where vLLM didn't. **Retry with `NCCL_IB_DISABLE=1` (TCP)** to isolate feasibility.

**S0b attempt 2 (TCP, IB disabled):** distributed init OK (5.33s); abandoned — cross-node TCP
allreduce per-layer is unusably slow for real TP2 serving (not a valid config, feasibility-only).

**S0b attempt 3 (RDMA + fix):** root-caused attempt-1: NCCL CUDA-graph capture over IB needs
`NCCL_CUMEM_ENABLE=1` (absent in attempt 1 → RETRY_EXC at capture). Relaunch with
`NCCL_CUMEM_ENABLE=1` + `NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=10 NCCL_IB_QPS_PER_CONNECTION=2`.
GID index 3 confirmed = **RoCE v2** (gid1/gid3 are v2; matches vLLM's working GID). Status: ⏳.

**S0b attempt 3 result:** same `IBV_WC_RETRY_EXC_ERR` at capture — cumem did NOT fix. Failure is
precise + reproducible: **71680-byte `RDMA_WITH_IMM` recvs fail; small collectives (init) succeed**,
on BOTH RoCE HCAs.

**RoCE topology:** each node has 2 HCAs (rocep1s0f1 = 10.0.1.x link/enp1s0f1np1; roceP2p1s0f1 = 2nd
link), both ACTIVE/LinkUp **200 Gb/s (2X NDR)**, correctly same-subnet paired. So not cabling.
Small-ok / large-fail on a healthy 200Gb link = **RoCEv2 lossless/PFC not configured** (large bursts
dropped without priority flow control). Attempt 4 = `NCCL_NET_GDR_LEVEL=0` (host-staged) — if PFC is
the cause this won't help either.

**Reframe:** interconnect is **200 Gb/s ethernet**, so NCCL-over-TCP (`NCCL_IB_DISABLE=1`) is NOT
slow for TP2 (1 GPU/node → small per-token allreduce). Plan: if RDMA can't be made lossless, measure
TCP-on-200Gb as the real SGLang config. (vLLM's 320 tok/s may itself have used TCP-fallback here.)

**S0b attempt 5 (TCP full run) — ROOT CAUSE FOUND.** rank1 (spark2) log shows it loaded fully
(weights 105s/77.6GB, DSV4 KV pool 2.45M tokens, dsv4 attn backend) then hit
**`Running FlashInfer autotune`** → hang → both ranks killed. rank0's error: "collective timeout ...
wrong sizes/order across ranks". → **FlashInfer autotune is NOT collective-synchronized across the
2 nodes** — one rank races ahead, the other's allreduce (SeqNum=2) never matches → deadlock →
RDMA RETRY_EXC / TCP timeout are both *symptoms*, not the cause. The RoCE fabric is likely fine.
Cookbook's verified command includes **`--disable-flashinfer-autotune`** (I'd omitted it).

**S0b attempt 6 (RDMA + `--disable-flashinfer-autotune`):** cleared autotune (fix confirmed), loaded
weights + KV pool, entered **cudagraph capture** — then hung 21+ min on the first graph (bs=256), no
progress, no RETRY_EXC. The capture-time cross-node allreduce (large, ~71680B) can't complete over
RDMA on this fabric (same large-transfer wall, now silently hanging inside capture instead of erroring).
→ RDMA is unusable for SGLang's large cross-node collectives on this GB10 RoCE setup, period.

**S0b attempt 7 (TCP + autotune-off + eager):** loaded fully both ranks (weights, KV pool 2.45M,
dsv4 backend, tree cache) — then died at the **first model-forward allreduce (SeqNum=2, 24576 elems)**
with rank0 "remote process exited or network error". rank1 (spark2) log shows **no error of its own**
— it reached "Tree cache initialized", was healthy, and got SIGTERM'd by orchestration after rank0's
watchdog fired. So the **2nd cross-node NCCL collective drops the inter-node connection** — SeqNum=1
completes, SeqNum=2 always fails, on RDMA AND TCP, small AND large messages.

### SGLang VERDICT (S0/S1/S2/S3/S4/S5): ❌ cross-node TP2 blocked on 2×GB10
- SGLang **latest DOES support DeepSeek-V4 on sm_121**: recognizes `DeepseekV4ForCausalLM`, dsv4
  attention backend, mxfp4 FP4 experts, fp8 KV, loads weights (~340s), builds 2.45M-token KV pool,
  initializes — all fine. The model + kernels work on GB10.
- **Blocker: cross-node TP2 NCCL.** The 2nd TP-group collective (`PG ID 2`, first real forward
  allreduce) drops the inter-node connection every time. Independent of: transport (RDMA `RETRY_EXC`
  / TCP "remote exited"), flashinfer autotune (disabled), cudagraph capture (disabled/eager), message
  size (24K–71K), NCCL_CUMEM, GDR level, IB timeout/retry/QPS. rank1 never crashes on its own.
- **Not the hardware:** vLLM TP2 runs cross-node on the identical RoCE fabric (326 tok/s). SGLang's
  NCCL 2.28.9 cross-node TP path is the differentiator. Cookbook only ever verified **single-node**
  (TP4 on 1×GB300, or TP2 on 1×RTX-PRO-6000 with NVLink/PCIe) — never 2-node-1-GPU-each.
- **Can't use single-node:** DeepSeek-V4-Flash ≈ 152GB (73GB/rank at TP2) → does not fit one 122GB
  GB10. So TP2-cross-node is mandatory here, and that's exactly the broken path.
- **Left untested (gated behind this):** S1/S2 DSpark+mxfp4, S3 nvidia-NVFP4, S4 sgl-FP8 MegaMoE,
  S5 NGC. All require the same cross-node TP2 collective → all blocked until it's fixed.
- **Fix candidates for later** (not pursued now): newer NCCL, `--enable-p2p-check`, mscclpp,
  `--moe-a2a-backend deepep` (proper EP dispatch vs pure-TP), or an SGLang cross-node-TP bugfix. Would
  need SGLang-team input; not worth more serial hours vs the working vLLM path.

**Bottom line: on 2×GB10, vLLM (eugr-b12x) is the only working DeepSeek-V4 framework; SGLang is
blocked at cross-node TP2.**

**SGLang image versions (both lack DSpark):** lmsysorg = **v0.5.10.post1**, NGC = **v0.5.9**.
`--speculative-algorithm` = {EAGLE,EAGLE3,NEXTN,STANDALONE,NGRAM} on both — **no DSPARK** (cookbook
DSpark verified on v0.5.16). NGC has `sglang serve` CLI; both have `python3 -m sglang.launch_server`.
→ DSpark-on-SGLang needs a v0.5.16+ pull; gate S0 first (sm_121 feasibility) before pulling.
| S1 | lmsysorg/sglang | deepseek-ai official | mxfp4 | DSpark | ⏳ | cookbook low-latency recipe (flashinfer_mxfp4) |
| S2 | lmsysorg/sglang | deepseek-ai official | mxfp4 | off | ⏳ | high-throughput |
| S3 | lmsysorg/sglang | nvidia NVFP4 | nvfp4 | — | ⏳ | needs flashinfer_trtllm_routed (SM100+ — may fail on sm_121) |
| S4 | lmsysorg/sglang | sgl-project FP8 | FP8 | megamoe | ⏳ | DP-attn + DeepEP + MegaMoE |
| S5 | nvcr.io/nvidia/sglang:26.03 | deepseek-ai official | mxfp4 | DSpark | ⏳ | NGC SGLang |

SGLang cookbook reference (GB200/GB300, not GB10 — for shape only):
- Low-latency verified 4×GB300 FP4: `--tp 4 --moe-runner-backend flashinfer_mxfp4
  --speculative-algorithm DSPARK --mem-fraction-static 0.90 --chunked-prefill-size 4096
  --swa-full-tokens-ratio 0.1`. DSpark shape read from checkpoint (no num-steps/topk/draft-tokens).
- DSpark constraints: CUDA, pp_size==1, DP-Attn disabled, not PD-disagg compatible.
- Concurrency invariant: `max-running-requests × MTP_draft_tokens ≤
  SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`.
- RTX PRO 6000 (SM120, closest to GB10): Flash only, FlashInfer MXFP4 runner, no HiCache/MegaMoE.

---

## Context-window ceiling + NVFP4-KV (2× GB10)

**FP8 practical ceiling ≈ 512K.** Weights ~156GB (78GB/node) dominate; KV pool holds only
~500-560K tokens after that. Model native = 1M (YaRN, 65536×16). 1M `--max-model-len` SIGKILLs on
first forward (runtime activation spike, not KV-pool). MLA already compresses KV to a low-rank latent
(~7.7KB/token, fp8_ds_mla = UE8M0 packed uint8, 576B/token slot) — so KV-quant tricks (TurboQuant,
asymmetric K/V) give ~nothing (asymmetric K/V is moot — MLA has no separate K/V, just a joint latent).
Host offload is moot — GB10 is **UMA** (memory is memory, no tier to spill to).

**The real 1M lever = NVFP4 KV cache** (4-bit KV → ~2× ctx tokens). Confirmed by tonyd2wild's recipe
`DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`:
- `--kv-cache-dtype nvfp4_ds_mla` + `VLLM_NVFP4_MLA_DYNAMIC_SCALE=1` (+ optional
  `VLLM_NVFP4_MLA_SCALES_FILE`), official FP8 weights, `--max-model-len 1048576`, seqs 12, batched
  8192, util 0.85, cudagraph-cap 36, DSpark k=5, `--distributed-executor-backend mp --nnodes 2`.
- **Needs their CUSTOM image** `vllm-dspark-runtime:dspark-nvfp4-stage-c` (Stage A/B/C Dockerfiles add
  the `nvfp4_ds_mla` KV plumbing + 584B padded sparse-MLA envelope + DSpark patches: Patch 4 fixes
  draft shared-expert `w1`/`w3` in `gate_up_proj` — else acceptance 60%→26%; Patch 3 cold-start garble).
- **Our eugr-b12x image can't do it:** `_resolve_dsv4_kv_cache_dtype()` **asserts `kv_cache_dtype`
  starts with `fp8`** for the ds_mla layout → `nvfp4_ds_mla` rejected. The image HAS the env hooks
  (`VLLM_NVFP4_MLA_DYNAMIC_SCALE/SCALES_FILE`) but not the validated Stage-C KV path. → 1M requires
  building tonyd2wild's image (separate effort).

**NVFP4 *weights* don't help ctx** (checked configs): `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` = **164GB**
(mixed: FP8 block attn + nvfp4-pack ffn experts group16), `nvidia/DeepSeek-V4-Flash-NVFP4` = **168GB**.
Both ≈ FP8 (~156GB), NOT smaller — attn stays FP8 + FP4 scale overhead. So no memory freed for ctx.
NVFP4-weight value is only "does NVFP4 finally serve on vLLM" — being tested (RedHatAI, D-rows below).

### Available NVFP4/FP4 DeepSeek-V4-Flash checkpoints (HF)
official: `nvidia/DeepSeek-V4-Flash-NVFP4` (168GB, fp8 method, FP4 Linear g16, ignores attn/shared/head/mtp).
RedHat: `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` (164GB, compressed-tensors mixed-precision, **vLLM-canonical**).
also: `amd/...-NVFP4`, `RedHatAI/...-BF16`, many community (`MJPansa`,`auroter`,`utarn`,`Rarri`,...),
abliterated (`neko-legends`,`sakamakismile` — both failed the 3 vLLM walls).

### D — NVFP4 weights on vLLM eugr-b12x
| ID | Model | Quant | Status | Peak tok/s | Notes |
|----|-------|-------|--------|-----------|-------|
| D1 | RedHatAI NVFP4-FP8 | FP4 MoE + FP8 attn | ❌ | | **compressed-tensors incompatible with B12X kernels.** Progressed further than neko (NVFP4 MoE backend picked = FLASHINFER_CUTLASS, no swiglu-clamp rejection — RedHatAI sets no clamp), but B12X FP8 kernels hard-require DeepSeek-native UE8M0 `weight_scale_inv` which RedHatAI's **block-FP8** attn/dense lacks: `VLLM_USE_B12X_WO_PROJECTION requires FP8 wo_a.weight_scale_inv`, then `AttributeError: 'ColumnParallelLinear' object has no attribute 'weight_scale_inv'` (from `VLLM_USE_B12X_FP8_GEMM` at compile). Would need ALL B12X opts off → generic paths → slower than apetersson FP8. Per decision rule → **abliterated wins.** 164GB anyway = no ctx gain. |

**NVFP4 on vLLM — final verdict:** no NVFP4 DeepSeek-V4-Flash checkpoint runs *well* on eugr-b12x.
Abliterated all-NVFP4 (neko/sakamakismile): 3 walls (swiglu-clamp, cutlass-eager, block_tables).
RedHatAI NVFP4-FP8: compressed-tensors scale format ≠ B12X native-FP8 kernels (`weight_scale_inv`).
nvidia NVFP4 (168GB) untested but same compressed-tensors/modelopt family → same mismatch expected.
The B12X speed kernels are hard-wired to DeepSeek's UE8M0 native FP8 → **FP8 (apetersson/official) is
the only quant that gets the full B12X path.** NVFP4 weights give no ctx benefit either (all ~164-168GB).
The only real NVFP4 win on GB10 = tonyd2wild's NVFP4 **KV** (1M ctx).

### E — 1M context via NVFP4 KV cache (the real prize)
tonyd2wild recipe `DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark` builds a custom image
(base `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` + Python overlays + Stage-A/B/C patches
that set the DeepSeek-V4 NVFP4 cache envelope to **584 bytes/token**). PORTABILITY.md = clean-room
bring-up on **2× DGX Spark GB10 sm_121a / 200G CX7 (our exact hardware)**, verified 1M + DSpark k=5.

**BUT — eugr-b12x already has the NVFP4-KV kernels natively** (no custom image needed):
- `--kv-cache-dtype nvfp4_ds_mla` is a **valid choice** (`config/cache.py:28`).
- `v1/attention/backends/mla/b12x_mla_sparse.py` fully implements it (scale-format-2, store/decode,
  KV_FP8_ROPE — lines 178-1747); `kv_cache_interface.py:478` + `torch_utils.nvfp4_kv_cache_full_dim`.
- KV_FP8_ROPE is GLM-only gated (`_is_glm_moe_dsa_model()`) → DeepSeek uses the plain NVFP4-KV branch.
- (The fp8-only assert in `models/deepseek_v4/attention.py:_resolve_dsv4_kv_cache_dtype` is a separate
  model-level path — may or may not fire; patch if it does.)

| ID | Config | Status | Notes |
|----|--------|--------|-------|
| E1 | eugr-b12x + nvfp4_ds_mla KV + 3 patches, 1M | ❌ | eugr NVFP4 writer architecturally incomplete (see verdict below) |
| **E2** | **tonyd2wild bjk110-based image**: official 0731 FP8 wts + nvfp4_ds_mla KV + DSpark k5, **1M ctx**, seqs 12, batched 8192, util 0.80, `mp` executor `--nnodes 2` over RoCE | ✅ | **1M ACHIEVED.** KV pool **1,650,115 tokens**, `max_model_len=1048576` (API-confirmed), coherent, 46-47°C. Perf (DSpark): **c1=36, c8=104, c12=160 tok/s**. Built `dspark-nvfp4-stage-c` both nodes (base `ghcr.io/bjk110/vllm-spark:unholy-fusion`, vLLM 0.21.1rc1) via `build-dspark-vllm-runtime.sh`; deployed via their compose. **The DeepSeek padded-nvfp4 writer eugr lacks is what makes 1M work.** |

**1M recipe (working, this is the deliverable for 1M):**
- Image: build `vllm-dspark-runtime:dspark-nvfp4-stage-c` from tonyd2wild repo on **both** nodes
  (`./build-dspark-vllm-runtime.sh` — base `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready`,
  overlay + stage-a/b/c). Lightweight (COPY + Python patches, no compile).
- `.env.dspark`: our RoCE (WORKER_HOST=192.168.0.212, MASTER_ADDR/VLLM_HOST_IP=10.0.1.1,
  WORKER_VLLM_HOST_IP=10.0.1.2, NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1, NCCL_SOCKET_IFNAME=enp1s0f1np1,
  NCCL_IB_GID_INDEX=3), `--kv-cache-dtype nvfp4_ds_mla`, MAX_MODEL_LEN=1048576, seqs 12, batched 8192,
  DSpark MTP_NUM_TOKENS=5, `--distributed-executor-backend mp --nnodes 2`. VLLM_PORT=8000.
- Deploy: `./start-deepseek-v4-flash-dspark.sh` (starts worker rank1 + head rank0 via compose). NOTE:
  the launcher polls port 8888 hardcoded; with VLLM_PORT=8000 the server still comes up on 8000 —
  ignore the launcher's connect-timeout messages, check `docker logs tonyd2wild-vllm-dspark-1`.
- Model: currently official `deepseek-ai/DeepSeek-V4-Flash-0731`; swap `DSPARK_MODEL` to
  `apetersson/...-Abliterated-FP8` for uncensored (same arch). Patch 4 (draft shared-expert) raises
  DSpark acceptance ~60% vs ~26% — apply if acceptance matters.

**E1 walls peeled on eugr-b12x (the tonyd2wild Stage-C fixes, hand-applied):**
1. arg-parse: `nvfp4_ds_mla` is a valid `--kv-cache-dtype` (native). ✅
2. `models/deepseek_v4/attention.py` `_resolve_dsv4_kv_cache_dtype` asserted fp8-only → **patch:
   allow nvfp4_ds_mla through** (return uint8, no rewrite). ✅
3. `kv_cache_utils.py:1886 assert max(sm_page_sizes) <= max(all_page_sizes)` — eugr sizes DeepSeek-V4
   NVFP4 KV at **432 B/token** but its fp8_ds_mla page is 584 → sparse-MLA page > full page → assert.
   **patch: pad NVFP4 envelope 432→584** in `v1/kv_cache_interface.py` (deepseek_v4 storage) +
   `v1/attention/backends/mla/b12x_mla_sparse.py` (get_kv_cache_shape stock). = tonyd2wild Stage-C. ✅
   (These `pyc`-purged + re-applied after every container start.)
4. `RuntimeError: setStorage: ... requiring 12878016 out of bounds for 12877824` (over by 192B) at
   `_init_minimal_kv_cache_for_profiling`. Decoded: view wants **1728 = 3×576** (q_head_dim) but
   buffer gives **1536 = 3×512** (kv_lora_rank) — a **512-vs-576 geometry mismatch in eugr's own
   NVFP4 store** (`_borrow_workspaces`/DCP, `b12x_mla_sparse.py`). ❌ **Not fixable by a minimal patch.**

**E1 VERDICT — minimal-changeset-on-eugr NOT feasible.** eugr's `b12x_mla_sparse.py` (**2890 lines**)
is a totally different vLLM generation from tonyd2wild's overlay (**501 lines**, 2653 diff lines) — can't
port their store fix. eugr has NVFP4-KV *plumbing* (valid dtype, backend refs, envelope) but its
**DeepSeek-V4 sparse NVFP4 store path is incomplete** (the 512/576 workspace overrun). 3 hand-patches
cleared resolver + page-grouping; the store geometry is a genuine eugr gap needing internal debugging,
not a small delta. **Real 1M path = tonyd2wild's image on the `ghcr.io/bjk110/vllm-spark:unholy-fusion`
base** (their patches match it exactly; PORTABILITY.md verifies 1M + DSpark on 2× GB10 sm_121a — our HW).

**Deep store debug (root, definitive).** `b12x_mla_sparse.do_kv_cache_update` has two NVFP4 writers:
- `if not self._kv_fp8_rope: super().do_kv_cache_update(...)` = **stock 432-byte** writer (DeepSeek),
  fixed-geometry, **cannot pad**.
- else `_concat_and_cache_nvfp4_mla_fp8_rope(...)` = **GLM 576** fp8-rope writer (`_is_glm_moe_dsa_model`).

DeepSeek-V4 is hybrid-SWA + a **DSA sparse-indexer** cache whose page > 432, so the MLA nvfp4 (432)
must pad up to it (`_get_kv_cache_groups_uniform_groups`) — but the stock 432 writer can't write into
a padded buffer (→ setStorage +192B when forced to 584). **eugr has no DeepSeek padded-nvfp4 writer**
(only GLM's). That writer is precisely what tonyd2wild implements on their base. **Conclusion: eugr
DeepSeek-V4 NVFP4-KV is architecturally incomplete; not fixable by a Python patch. 1M requires the
bjk110-based tonyd2wild image.** (FP8 512K remains the eugr ceiling.)

## Findings log (chronological, verbatim errors)

(Filled as tests run.)

### B1 — deepseek-ai official FP8, eugr-b12x, no-spec, 256K, seqs 48
- Status: ⏳ loading

---

## SGLang + b12x on GB10 — architecture blocker (2026-08-20/21)

Goal was aggregate throughput at c5 under SGLang with b12x kernels. **Not achievable
today**, for a hardware reason rather than a config one.

### b12x availability by architecture

| Component | arm64 / DGX Spark? | Note |
|---|---|---|
| b12x kernels **v1.2.3** | ✅ | ships inside `eugr/spark-vllm-b12x` — "DGX Spark and RTX 6000-focused inference kernel library" |
| vLLM + b12x | ✅ | what this repo serves with |
| SGLang, arm64 | ✅ | `scitrera/dgx-spark-sglang:0.5.17` (digest `sha256:cc1cec4d…`), SGLang 0.5.17 / torch 2.11.0 / FlashInfer 0.6.15.post1 — **no b12x** |
| SGLang + b12x | ❌ | `voipmonitor/sglang:cu130`, `lukealonso/sglang-cuda13-b12x` are **linux/amd64** → `exec format error` on GB10 |
| b12x upstreamed into SGLang | ❌ | one doc mention in `sgl-project/sglang`, no integration code |

The community b12x SGLang images target RTX PRO 6000 x86 workstations (SM120), not
GB10 (sm_121a, aarch64).

### Why a port was scoped and rejected

The b12x fork of SGLang is ~0.5.12-era (May 2026) and written against b12x **0.8/0.10**.
The arm64 package is **1.2.3**, whose `integration/` layer was refactored to be
vLLM-only — `attention.py`, `mla.py`, `nsa_indexer.py`, `tp_moe.py` were all removed.
Of ten symbols the fork imports, five no longer exist:

| Symbol | In b12x 1.2.3 |
|---|---|
| `dense_gemm` | `b12x/_lib/dense_gemm.py` |
| `b12x_moe_fp4` | `b12x/moe/fused_moe/_impl.py` |
| `PagedAttentionWorkspace` | `b12x/attention/paged/workspace.py` |
| `build_decode_chunk_pages_lut` | `b12x/attention/paged/planner.py` |
| `get_decode_graph_policy` | `b12x/attention/paged/tuning/registry.py` |
| `get_paged_mqa_logits_metadata` | **missing** |
| `make_nsa_indexer_contract_phantoms` | **missing** |
| `_as_grouped_scale_view` | **missing** |
| `B12XExecutionLane` | **missing** |
| `get_b12x_moe_workspace_pool` | **missing** |

Also, `nsa_backend.py` carries 147 b12x-referencing lines and diverges heavily between
0.5.12 and 0.5.17. A working port means writing a **new** SGLang↔b12x bridge against
undocumented internals, using vLLM's b12x call sites as the only reference. Days of
work, unproven payoff — rejected.

### SGLang without b12x — also blocked

`scitrera/dgx-spark-sglang:0.5.17`, recipe `sglang-c5.yaml`, model
`drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` (gated, 156 GB,
48 shards, `num_nextn_predict_layers=1` so it ships the MTP/DSpark draft head).

**What worked:** DSPARK is accepted and auto-detects the bundled drafter —
*"DSpark draft weights are bundled in the target checkpoint; defaulting
`--speculative-draft-model-path`"* — and KV dtype auto-selects `fp8_e4m3`.
MoE backends available: `flashinfer_mxfp4` (DeepSeek's own recommendation), plus
`deep_gemm`, `flashinfer_cutlass`, `cutlass`, `triton`, `marlin`, `hpc_ops`.

**Blocker:** on 2-node bring-up the worker rank dies during FlashInfer autotune:

```
[TP1] Running FlashInfer autotune with cache: /cache/runtime/sglang/flashinfer/autotune/0.6.15.post1/sm121/...
Terminated
```

then the head fails with
`RuntimeError: gloo/transport/tcp/pair.cc:547 Connection closed by peer [192.168.0.212]`.

`--dist-timeout 3600` does **not** help — the worker is terminated outright, it is not
timing out. The same log also warns *"Breakable CUDA graph is incompatible with
DeepSeek-V4 (heavy capture-pool memory pressure); disabling prefill CUDA graph"*.
Unresolved; abandoned in favour of the working vLLM path.

## b12x hangs under the eugr image — root-caused with py-spy (2026-08-20)

Two distinct hangs, both located with `py-spy` after granting `CAP_SYS_PTRACE`
(see TROUBLESHOOTING → Tooling).

**1. GEMM launched on a non-default CUDA stream (workaround found).**

```
run_compiled_program (cutlass/base_dsl/jit_executor.py:1074)
dense_gemm_fused_quant_a (b12x/_lib/dense_gemm.py:7057)
_block_fp8_linear_mxfp8_fused_op (b12x/gemm/_shared/block_fp8.py:568)
...
apply_weights (vllm/model_executor/kernels/linear/scaled_mm/b12x.py:280)
<lambda> (vllm/models/deepseek_v4/attention.py:718)
execute_in_parallel (vllm/utils/multi_stream_utils.py:212)
_run_parallel_input_projections (vllm/models/deepseek_v4/attention.py:717)
```

`execute_in_parallel` dispatches the attention input projections onto auxiliary CUDA
streams. Forcing its built-in serial fallback (`enable=False`) clears the hang and the
server serves normally.

**2. Sparse-MLA prefill (no workaround).**

```
_sparse_mla_prefill_mg_flat_launch (b12x/attention/_shared/mla/prefill_mg.py:3888)
run_unified_prefill_mg → _run_partitioned_mg → run_unified_prefill
_run_sm120_compressed_prefill (b12x/attention/_shared/mla/compressed_api.py:366)
compressed_mla_decode_forward (b12x/attention/_shared/mla/compressed_api.py:233)
```

Note `_run_sm120_compressed_prefill` being selected on **sm_121a** — possibly an
arch-gate treating 12.1 as 12.0-compatible. `EngineCore` sits in
`multiproc_executor.get_response`→`shm_broadcast.dequeue` waiting on that worker, which
is why the process looks alive and `/health` keeps returning 200.

Reported upstream with full stacks:
<https://github.com/eugr/spark-vllm-docker/issues/352>

Also confirmed: `/v1/chat/completions` hangs on that image (requests never reach the
engine — the hang is in the API-server chat/reasoning-parser path) and wedges every
subsequent `/v1/completions` request. Benchmark through `/v1/completions` only.
