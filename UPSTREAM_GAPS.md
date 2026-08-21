# Open gaps, DeepSeek-V4-Flash-0731 on 2× DGX Spark (GB10 / sm_121a)

What still has to be re-implemented or patched to serve this checkpoint on this hardware, filed for
upstream maintainers. Every item below was hit and diagnosed on real 2× GB10 (see TEST_LOG.md for
verbatim errors). "Works only in tonyd2wild's custom image" means: not in stock vLLM, not in eugr,
not in eugr-b12x, it needs the overlay + stage-a/b/c patches on the bjk110 base.

## 1. vLLM: DeepSeek-V4 NVFP4 KV (`nvfp4_ds_mla`) writer is GLM-only / incomplete
**The 1M blocker.** eugr-b12x *has* the `nvfp4_ds_mla` dtype, backend refs, and env hooks
(`VLLM_NVFP4_MLA_DYNAMIC_SCALE`), but its DeepSeek-V4 NVFP4-KV path is unfinished:
- `b12x_mla_sparse.do_kv_cache_update` routes to `_concat_and_cache_nvfp4_mla_fp8_rope`: a **GLM
  576-geometry** writer, or a **stock 432-byte** writer that **cannot pad**.
- DeepSeek-V4 is hybrid-SWA + a **DSA sparse-indexer** cache whose page > 432, so the MLA NVFP4 page
  must pad up to it (`_get_kv_cache_groups_uniform_groups: assert max(sm_page_sizes) <= max(all_page_sizes)`).
  The stock 432 writer can't write into a padded buffer → `setStorage ... out of bounds` (512-vs-576).
- tonyd2wild's Stage-C fixes it with a **584-byte padded DeepSeek-V4 NVFP4 envelope** + a real
  padded-NVFP4 writer. **Upstream should land the DeepSeek-V4 padded-NVFP4 KV writer** so 1M works on
  stock/eugr without the custom image.
- **Status @ vLLM v0.27.1 (checked in source):** STILL the gap, and now *actively rejected*. There is
  **no `nvfp4_ds_mla`** dtype (`config/cache.py` lists `fp8_ds_mla` + a generic `nvfp4`, not the packed
  DS-MLA nvfp4), and a new validator **`VllmConfig.validate_nvfp4_kv_cache_with_mla` (config/vllm.py)
  raises** `"nvfp4 KV cache is not supported with MLA ... use 'fp8' or 'auto'"`. So 0.27.1 caps
  DeepSeek-V4 at **fp8 KV (~512K ctx)**; the 1M nvfp4 path remains tonyd2wild-only.

## 2. vLLM: SM12x sparse-MLA decode + DSpark, MERGED as of v0.27.1
**Was** unmerged (PR #41834 sparse-MLA + DSpark; the model ran no-spec/eager-only on stock ≤0.24).
**Now, in v0.27.1 source, the whole spec+sparse stack is upstream** (verified by grep):
- **DeepSeek-V4 model** lives at `vllm/models/deepseek_v4/` (new plugin-model dir alongside
  `deepseek_v32`, `kimi_k3`, `minimax_m3`): registry maps `DeepseekV4ForCausalLM`,
  `DSparkDeepseekV4ForCausalLM` (draft), and `DeepSeekV4MTP`.
- **DSpark / DFlash / MTP** speculators: `vllm/v1/worker/gpu/spec_decode/{dspark,dflash}/`.
- **SM120 sparse-MLA decode**: `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`
  (`FLASHINFER_MLA_SPARSE_SM120`), which **requires the packed `fp8_ds_mla` KV** and FlashInfer's
  `trtllm_batch_decode_sparse_mla_dsv4` (gated by `has_flashinfer_sparse_mla_sm120()`).
- Plus deepseek_v4 `parser/renderer/tokenizer/transformers_utils config`.

**What this means:** stock **v0.27.1 can serve DeepSeek-V4-Flash + DSpark + sparse-MLA on sm120 with
fp8 KV, no tonyd2wild patches**: i.e. the ≤512K path is (near) upstream. The **only** remaining custom
piece is the `nvfp4_ds_mla` writer for 1M (gap #1). **Two GB10 caveats to verify before trusting it on
sm_121a:** (a) FlashInfer must ship `sparse_mla_sm120` / `trtllm_batch_decode_sparse_mla_dsv4` **built
for sm_121a** (stock flashinfer wheels historically target sm100/sm120 datacenter/desktop, not GB10 , 
this is what eugr's sparkinfer/B12X supplied); (b) the MoE + attention kernels must have sm_121a SASS.
Current live serve still runs the tonyd2wild image (vLLM `0.21.1rc1`) for the 1M nvfp4 path.

## 3. vLLM: NVFP4 *weight* MoE path broken on sm_121 for DeepSeek-V4
- `flashinfer_b12x` MoE: swiglu-clamp gate admits only `SWIGLUOAI_UNINTERLEAVE`; DeepSeek's plain
  SILU-swiglu is rejected at the kernel (`FlashInferB12xExperts only applies swiglu_limit with
  swigluoai_uninterleave`).
- `flashinfer_cutlass` (sm120 kernel exists): **eager-only**, and every warmup dummy-run
  (cudagraph capture / flashinfer autotune / mem-profile-with-attn) hits
  `AttributeError: GPUModelRunner has no attribute 'block_tables'`: NVFP4 runner sets `block_tables`
  after warmup (init-ordering). `flashinfer_trtllm` fp4 = sm100-only.
- `RedHatAI/...-NVFP4-FP8` (compressed-tensors, block-FP8 attn): incompatible with B12X native-FP8
  kernels, `VLLM_USE_B12X_WO_PROJECTION requires FP8 wo_a.weight_scale_inv`, then
  `'ColumnParallelLinear' object has no attribute 'weight_scale_inv'` at compile.
- **Note:** NVFP4 *weights* give no memory benefit here anyway (all checkpoints ~156-168GB, mixed
  precision + FP4 scale overhead). The only NVFP4 win on GB10 is **KV** (gap #1).

## 4. vLLM: reasoning field name diverges from DeepSeek's hosted API
This runtime returns reasoning under **`reasoning`**; DeepSeek's hosted API uses **`reasoning_content`**.
OpenAI-compat harnesses that assume `reasoning_content` (Kimi Code, lm-eval, ...) **leak `</think>`
into content**. Please align the field name, or document it prominently. (Client workaround: point the
harness's reasoning key at `reasoning`.)

## 5. SGLang: cross-node TP2 NCCL fails on 2× GB10 (1 GPU/node)
SGLang latest **supports DeepSeek-V4 on sm_121** (recognizes `DeepseekV4ForCausalLM`, dsv4 attn,
FP4 experts, builds a 2.45M-token KV pool), but the **2nd TP-group collective** (first real forward
allreduce, `PG ID 2`) **always drops the inter-node connection**: RDMA (`IBV_WC_RETRY_EXC_ERR`) and
TCP ("remote process exited") alike, independent of flashinfer-autotune, cudagraph capture, message
size (24K-71K), `NCCL_CUMEM`, GDR level, IB timeout/retry/QPS. rank1 never crashes on its own, it's
killed by orchestration after rank0's watchdog. **Not the fabric** (vLLM TP2 runs cross-node on the
same RoCE). Cookbook only ever verified single-node (TP4 on 1×GB300, or TP2 on 1×RTX-PRO-6000 with
NVLink). The 2-node-1-GPU-each layout is broken. Model is 152GB so single-node isn't an option.

**Update, SGLang 0.5.17 (lmsysorg/sglang:latest, built 2026-08-07) fixes the BOOT collective, but hangs
at DECODE.** Re-tested on 2× GB10 (fp8_e4m3 KV, 524K ctx, `--tp-size 2 --nnodes 2`, same RoCE env):
- ✅ **Boots fully now**: the forward-profiling allreduce that used to drop at `PG ID 2` **succeeds**:
  builds a **2,043,648-token** KV pool, `max_running_requests=256`, captures decode cudagraphs (bs up to
  256), reaches `Application startup complete` / `The server is fired up and ready`, health 200, and a
  trivial completion ("hi") returns correctly. So the boot-time cross-node NCCL bug is **gone** in 0.5.17.
- ❌ **Real generation hangs the worker.** On a normal coding prompt (128-256 tokens) the **TP1 (worker)
  scheduler stalls and hits `Scheduler watchdog timeout (300s)`** → worker exits → head follows with a
  `TimeoutError` and `kill_process_tree`. The instability didn't disappear, it **moved from boot to
  decode**: a cross-node decode-time stall. Still **not production-viable** on this 2-node-1-GPU layout.
- Note: 0.5.17 uses `DeepseekV4AttnBackend` (CUDA) + FlashInfer autotune on `sm121`, and only
  `lmsysorg/sglang:latest` has DeepSeek-V4 at all (nvcr `sglang:26.03`=0.5.9 and `v0.5.10.post1` do not).
- **Verdict:** meaningful progress (boot fixed), but **vLLM remains the only viable serve** on 2× GB10.

## 6. vLLM: multi-node `mp` executor restart wedge
- `restart: unless-stopped` + engine deaths that **exit 0** + a **capture-time cross-node collective
  wedge** → docker auto-restarts straight into a deadlock (GPU idle at KV-alloc, no error). A boot loop.
- **Orphaned `vllm`/`EngineCore`/`multiproc_executor` procs survive `docker compose down`**, holding
  the GPU and dist port 25000 → the next deploy deadlocks at distributed init.
- Needs a clean teardown that reliably reaps the mp children + releases the RoCE/NCCL state.

## 7. LMCache / disk KV offload, TESTED, blocked by HMA vs sparse-MLA + DSpark
On UMA (GB10), CPU/RAM offload is moot (shared memory). Disk-tier (LRU-spill-to-NVMe) is the useful
lever for concurrent large coding sessions that exceed the RAM KV pool. **We wired it end-to-end**
(baked `lmcache==0.5.3` into the image, `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1",
"kv_role":"kv_both"}'`, `LMCACHE_LOCAL_DISK=file:///lmcache_disk`, 150 GB NVMe). Result, 3 walls
cleared, 4th blocks it:
1. ✅ `lmcache` installs + imports cleanly (no dep conflict with vLLM 0.21rc); `LMCacheConnectorV1`
   registered; LMCache is MLA-aware.
2. ✅ Passes config validation **after** dropping `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   (LMCache's VMM remap would invalidate registered KV; needs expandable-segments off or the cumem
   allocator). Overriding to `garbage_collection_threshold:0.9` clears it.
3. ✅ **Accepts the `nvfp4_ds_mla` packed KV**: the format was never the problem.
4. ❌ `--kv-transfer-config` **turns off the hybrid KV-cache manager (HMA)**; DeepSeek-V4 is hybrid-SWA
   + sparse-MLA, and the DSpark verify batch (`num_tokens = seqs × (k+1)`, e.g. 36) then mis-routes:
   `sparse_mla_sm120_decode_dsv4: Check failed num_tokens>64 (36 vs 64)`. Fatal at startup.

**Root cause:** `LMCacheConnectorV1` does **not** implement `SupportsHMA`, so vLLM disables the hybrid
KV manager, which the DeepSeek-V4 sparse-MLA sm120 decode path requires (it also carries the DSpark k=5
verify). **The disk-spill blocker is HMA support in the connector, not the NVFP4 KV format.**

> ### ⚠️ CORRECTION (2026-08-21), this gap is narrower than originally written
>
> The original claim that **all** offload connectors lack `SupportsHMA` is **wrong**. Only LMCache was
> tested. Checking the stage-c image directly
> (`vllm/distributed/kv_transfer/kv_connector/v1/`):
>
> | Connector | `SupportsHMA` |
> |---|---|
> | `LMCacheConnectorV1` | ❌, the one that was tested |
> | `OffloadingConnector` | ✅ `class OffloadingConnector(KVConnectorBase_V1, SupportsHMA)` |
> | `SimpleCPUOffloadConnector` | ✅ |
>
> Stock vLLM also already ships a **disk-backed tier**, so no custom image is needed:
> `vllm/v1/kv_offload/tiering/` registers `TieringOffloadingSpec` (a CPU primary tier plus
> configurable secondary tiers) and `SecondaryTierFactory.register_tier("fs_python", …,
> "FileSystemTierManager")`: *"pure-Python disk-backed secondary tier"*, taking `root_dir`,
> `n_read_threads`, `n_write_threads`.
>
> So SSD KV-spill **with** DSpark + sparse-MLA is worth pursuing via `OffloadingConnector`, not
> LMCache. See `examples/prod-c5-ssd.yaml`. Note wall #2 still applies to *any* of these connectors:
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is rejected outright , 
> `Value error, KV connector OffloadingConnector is incompatible with
> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unless enable_cumem_alloc`: so set
> `garbage_collection_threshold:0.9` instead.

## 9. TokenSpeed (LightSeek) engine, builds + runs on GB10, blocked at weight-load by UMA page-cache OOM
TokenSpeed ships a `deepseek_v4_dspark` model + a DeepSeek-V4-Flash recipe (4×B200 SM100). We got it
**building and booting on 2× GB10** with portable backends; the wall is memory, not the ISA. Full trace:

**tcgen05 is a red herring for DeepSeek-V4.** The initial build fails at ptxas , 
`attn_res_fwd_tma.compute_121a.ptx: Instruction 'tcgen05.st/.ld' not supported on .target 'sm_121a'`
(`tcgen05` = 5th-gen Tensor Memory ISA, **SM100/SM103 datacenter-Blackwell only**; GB10 `sm_121a` has no
TMEM). But `attn_res_fwd_tma.cu` is the **Kimi-K3/KDA** kernel, imported only by `kimi_k3.py`: **not on
the DeepSeek-V4 path**. `tokenspeed-mla` (the tcgen05 CuTe MLA) is imported by **nobody**. Deleting the
`attn_res` entry from `tokenspeed-kernel/python/setup.py` → the kernel package **builds clean for
`CUDA_ARCH_LIST="12.1a"`** (all of `deepseek_v4_attention.cu`, `deepseek_v4_topk.cu`, trtllm comms, etc.).

**Portable backends exist and are selected.** DeepSeek-V4 attention registers a **`triton_dsa_decode` +
`triton_dsa_prefill`** path with `CapabilityRequirement(vendors={nvidia,amd})`: **no min-arch gate** →
runs on sm_121a (Triton JITs). MoE has **`flashinfer_cutlass_fp8_moe_apply`** (`weight_dtype: fp8`,
`fp8_scale_block_shape:(128,128)`, **min_arch 9.0** → GB10 qualifies; the `flashinfer_trtllm_fp8` variant
is min_arch 10.0 = sm100-only). Serve flags: `--attention-backend triton --moe-backend flashinfer_cutlass
--kv-cache-dtype fp8_e4m3`. The B200 fast path (`tokenspeed-flashmla` DSA + `tokenspeed-deepgemm` mega_moe)
is **not shipped in the image and not needed**: those are the tcgen05/sm100 kernels. **DSpark drafter has
no hard sm100 import** (shares the main attention backend).

**What actually works on GB10 (verified):** image builds; `import tokenspeed` + a Triton kernel compile +
`deepseek_v4_dspark` module all succeed on `CAP (12,1) NVIDIA GB10`; 2-node launch **clears distributed
init with no NCCL wedge** (unlike SGLang, gap #5, `trtllm one-shot all-reduce unavailable → NCCL
fallback`, fine); **clears MoE kernel selection** (`tp=2 ep=1 dp=1`, flashinfer_cutlass).

**The blocker, weight-load OOM on unified memory.** `DefaultModelLoader` allocates the full ~80 GB/node
model skeleton **on the GPU** (`with torch.device("cuda")`, off-book to cgroup on GB10), then reads the
156 GB of fp8 safetensors shards. On a discrete GPU the shard bytes land in host page cache separate from
VRAM; on GB10's **shared 122 GB** the 80 GB GPU skeleton + up to 80 GB of shard page cache coexist in the
*same* pool → **~160 GB attempted → the node wedges** (sshd starves during swap-thrash; hard-reboot to
recover). This is arch-agnostic to the kernels, purely a UMA memory-management problem. Levers tried:
- `--disable-weight-loader-prefetch-checkpoints` (default prefetches `min(80 GiB, 25% host RAM)` ahead) , 
  necessary but **insufficient**: demand-faulted reads still cache the shards.
- Docker `--memory` cgroup cap, **ineffective**: the GPU skeleton is off-book, and the bind-mounted
  shard page cache is charged to the **root** cgroup (files already resident from prior loads), not the
  container. Neither a low (32 GB) nor loose (100 GB) cap bounded the real memory.
- A host watchdog on `MemAvailable`: **too slow**: it starves (can't fork) before it can `docker kill`.
- `--gpu-memory-utilization`: **wrong axis**: KV is sized from free-mem *after* weights load, so util
  only sets the post-load KV/reserve split, not the load-time peak.

**The clean fix needs root** (which the loader/engine can't reach on its own): a `drop_caches` loop during
load (`sync; echo 1 > /proc/sys/vm/drop_caches`) to keep the shard page cache from stacking on the GPU
skeleton, or `O_DIRECT`/`madvise(DONTNEED)` reads in the loader (not exposed). **Upstream fix for UMA
GPUs (GB10/GH200-class):** either page-cache-bypass reads in `safetensors_weights_iterator`, or a
"unified-memory" load mode that streams shard→GPU while evicting each shard's pages. Until then TokenSpeed
is **build- and boot-OK on GB10 but not servable without root-level page-cache control**; stick with the
vLLM tonyd2wild runtime (which loads the same 156 GB fine at util 0.82). Kimi-K3 / MiniMax paths remain
genuinely tcgen05/sm100-only.

## 8. Minor
- `fastsafetensors` (0.3.2) is present but the recipe uses `--load-format safetensors`; could
  parallelize cold loads. Warm loads are already fast (page cache; ~36s weight read).
- No prebuilt ghcr image published for the tonyd2wild runtime yet → every consumer must build locally.


---

## Gap: `nvfp4_ds_mla` is unreachable for DeepSeek-V4 on the eugr b12x lineage

Two independent gates, only one of which is a genuine bug.

**1. Over-broad config guard (`vllm/config/vllm.py`, bug):**

```python
if (self.cache_config.cache_dtype.startswith("nvfp4")
        and self.model_config.use_mla):
    raise ValueError("nvfp4 KV cache is not supported with MLA ...")
```

`startswith("nvfp4")` sweeps up `nvfp4_ds_mla`, the NVFP4 layout built
specifically for MLA. The check should be an exact match on `"nvfp4"`. This one
is worth an upstream report.

**2. Missing DeepSeek-V4 writer (`vllm/models/deepseek_v4/attention*.py`, real):**

```python
assert kv_cache_dtype.startswith("fp8"), (
    "DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got nvfp4_ds_mla")
```

The DeepSeek-V4 path hardcodes the fp8 packed layout. Generic MLA layers
(`mla.py`, `mla_cache_format.py`, `b12x_mla_sparse.py`) do reference
`nvfp4_ds_mla`, so grepping the image overstates support. Patching gate 1 only
moves the failure to gate 2. **Do not patch gate 2**: it would mismatch the cache
layout and corrupt results silently rather than fail cleanly. Providing that
writer is what the stage-c overlay did.

Consequence: NVFP4 KV capacity (~6.6 KB/token) is unavailable on the eugr images;
they are limited to `fp8_ds_mla` at ~11.0 KB/token.
