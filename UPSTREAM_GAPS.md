# Open gaps — DeepSeek-V4-Flash-0731 on 2× DGX Spark (GB10 / sm_121a)

What still has to be re-implemented or patched to serve this checkpoint on this hardware, filed for
upstream maintainers. Every item below was hit and diagnosed on real 2× GB10 (see TEST_LOG.md for
verbatim errors). "Works only in tonyd2wild's custom image" means: not in stock vLLM, not in eugr,
not in eugr-b12x — it needs the overlay + stage-a/b/c patches on the bjk110 base.

## 1. vLLM: DeepSeek-V4 NVFP4 KV (`nvfp4_ds_mla`) writer is GLM-only / incomplete
**The 1M blocker.** eugr-b12x *has* the `nvfp4_ds_mla` dtype, backend refs, and env hooks
(`VLLM_NVFP4_MLA_DYNAMIC_SCALE`), but its DeepSeek-V4 NVFP4-KV path is unfinished:
- `b12x_mla_sparse.do_kv_cache_update` routes to `_concat_and_cache_nvfp4_mla_fp8_rope` — a **GLM
  576-geometry** writer — or a **stock 432-byte** writer that **cannot pad**.
- DeepSeek-V4 is hybrid-SWA + a **DSA sparse-indexer** cache whose page > 432, so the MLA NVFP4 page
  must pad up to it (`_get_kv_cache_groups_uniform_groups: assert max(sm_page_sizes) <= max(all_page_sizes)`).
  The stock 432 writer can't write into a padded buffer → `setStorage ... out of bounds` (512-vs-576).
- tonyd2wild's Stage-C fixes it with a **584-byte padded DeepSeek-V4 NVFP4 envelope** + a real
  padded-NVFP4 writer. **Upstream should land the DeepSeek-V4 padded-NVFP4 KV writer** so 1M works on
  stock/eugr without the custom image.

## 2. vLLM: SM12x sparse-MLA decode + DSpark not in stock
- PR **#41834** (SM12x sparse-MLA decode + DSpark) is **unmerged into main** → stock `vllm-openai`
  (latest/nightly) and NGC run this model **no-spec, eager only** (`sparse_mla_sm120: num_tokens>64`
  assert on spec). ~+38% slower single-stream, no cudagraphs.
- DSpark PR **#46995** merged (helps garble/concurrency) but is **insufficient** — v0.24 still can't
  boot the 1M/NVFP4 path without re-porting the GB10/SM120 survival overlays and the `nvfp4_ds_mla`
  writer above. Ref: tonyd2wild `UPSTREAM_V024_STATUS.md`.

## 3. vLLM: NVFP4 *weight* MoE path broken on sm_121 for DeepSeek-V4
- `flashinfer_b12x` MoE: swiglu-clamp gate admits only `SWIGLUOAI_UNINTERLEAVE`; DeepSeek's plain
  SILU-swiglu is rejected at the kernel (`FlashInferB12xExperts only applies swiglu_limit with
  swigluoai_uninterleave`).
- `flashinfer_cutlass` (sm120 kernel exists): **eager-only**, and every warmup dummy-run
  (cudagraph capture / flashinfer autotune / mem-profile-with-attn) hits
  `AttributeError: GPUModelRunner has no attribute 'block_tables'` — NVFP4 runner sets `block_tables`
  after warmup (init-ordering). `flashinfer_trtllm` fp4 = sm100-only.
- `RedHatAI/...-NVFP4-FP8` (compressed-tensors, block-FP8 attn): incompatible with B12X native-FP8
  kernels — `VLLM_USE_B12X_WO_PROJECTION requires FP8 wo_a.weight_scale_inv`, then
  `'ColumnParallelLinear' object has no attribute 'weight_scale_inv'` at compile.
- **Note:** NVFP4 *weights* give no memory benefit here anyway (all checkpoints ~156-168GB — mixed
  precision + FP4 scale overhead). The only NVFP4 win on GB10 is **KV** (gap #1).

## 4. vLLM: reasoning field name diverges from DeepSeek's hosted API
This runtime returns reasoning under **`reasoning`**; DeepSeek's hosted API uses **`reasoning_content`**.
OpenAI-compat harnesses that assume `reasoning_content` (Kimi Code, lm-eval, ...) **leak `</think>`
into content**. Please align the field name, or document it prominently. (Client workaround: point the
harness's reasoning key at `reasoning`.)

## 5. SGLang: cross-node TP2 NCCL fails on 2× GB10 (1 GPU/node)
SGLang latest **supports DeepSeek-V4 on sm_121** (recognizes `DeepseekV4ForCausalLM`, dsv4 attn,
FP4 experts, builds a 2.45M-token KV pool) — but the **2nd TP-group collective** (first real forward
allreduce, `PG ID 2`) **always drops the inter-node connection**: RDMA (`IBV_WC_RETRY_EXC_ERR`) and
TCP ("remote process exited") alike, independent of flashinfer-autotune, cudagraph capture, message
size (24K-71K), `NCCL_CUMEM`, GDR level, IB timeout/retry/QPS. rank1 never crashes on its own — it's
killed by orchestration after rank0's watchdog. **Not the fabric** (vLLM TP2 runs cross-node on the
same RoCE). Cookbook only ever verified single-node (TP4 on 1×GB300, or TP2 on 1×RTX-PRO-6000 with
NVLink). The 2-node-1-GPU-each layout is broken. Model is 152GB so single-node isn't an option.

## 6. vLLM: multi-node `mp` executor restart wedge
- `restart: unless-stopped` + engine deaths that **exit 0** + a **capture-time cross-node collective
  wedge** → docker auto-restarts straight into a deadlock (GPU idle at KV-alloc, no error). A boot loop.
- **Orphaned `vllm`/`EngineCore`/`multiproc_executor` procs survive `docker compose down`**, holding
  the GPU and dist port 25000 → the next deploy deadlocks at distributed init.
- Needs a clean teardown that reliably reaps the mp children + releases the RoCE/NCCL state.

## 7. LMCache / disk KV offload — TESTED, blocked by HMA vs sparse-MLA + DSpark
On UMA (GB10), CPU/RAM offload is moot (shared memory). Disk-tier (LRU-spill-to-NVMe) is the useful
lever for concurrent large coding sessions that exceed the RAM KV pool. **We wired it end-to-end**
(baked `lmcache==0.5.3` into the image, `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1",
"kv_role":"kv_both"}'`, `LMCACHE_LOCAL_DISK=file:///lmcache_disk`, 150 GB NVMe). Result — 3 walls
cleared, 4th blocks it:
1. ✅ `lmcache` installs + imports cleanly (no dep conflict with vLLM 0.21rc); `LMCacheConnectorV1`
   registered; LMCache is MLA-aware.
2. ✅ Passes config validation **after** dropping `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   (LMCache's VMM remap would invalidate registered KV; needs expandable-segments off or the cumem
   allocator). Overriding to `garbage_collection_threshold:0.9` clears it.
3. ✅ **Accepts the `nvfp4_ds_mla` packed KV** — the format was never the problem.
4. ❌ `--kv-transfer-config` **turns off the hybrid KV-cache manager (HMA)**; DeepSeek-V4 is hybrid-SWA
   + sparse-MLA, and the DSpark verify batch (`num_tokens = seqs × (k+1)`, e.g. 36) then mis-routes:
   `sparse_mla_sm120_decode_dsv4: Check failed num_tokens>64 (36 vs 64)`. Fatal at startup.

**Root cause:** `LMCacheConnectorV1` (and all offload connectors — `OffloadingConnector`,
`FlexKVConnectorV1`, `SimpleCPUOffloadConnector`) do **not** implement `SupportsHMA`, so vLLM disables
the hybrid KV manager, which the DeepSeek-V4 sparse-MLA sm120 decode path requires (it also carries the
DSpark k=5 verify). **The disk-spill blocker is HMA support in the connector, not the NVFP4 KV format.**
Fixes: land `SupportsHMA` on `LMCacheConnectorV1`, or run without DSpark/sparse-MLA (defeats the purpose).

## 8. Minor
- `fastsafetensors` (0.3.2) is present but the recipe uses `--load-format safetensors`; could
  parallelize cold loads. Warm loads are already fast (page cache; ~36s weight read).
- No prebuilt ghcr image published for the tonyd2wild runtime yet → every consumer must build locally.
