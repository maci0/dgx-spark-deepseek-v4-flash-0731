# Production config: 5 concurrent clients, SSD KV offload (NVFP4 pool NOT yet serving)

Target: a **production** serve (not a leaderboard run) for ~5 concurrent clients with the
largest practical KV pool and KV spill to NVMe, using as few non-stock knobs as possible.

**Status: not shipped.** The SSD offload path and the parameter set below are settled, but the
NVFP4 KV pool that makes ~2.2M tokens possible has never survived startup (§4). For a config
that actually serves today, use
[`examples/deepseek-v4-flash-0731-dspark-arena-threshold.yaml`](examples/deepseek-v4-flash-0731-dspark-arena-threshold.yaml)
(fp8, 1.45M tokens, 162.5 tok/s aggregate at c5).

Recipe: [`examples/prod-c5-ssd.yaml`](examples/prod-c5-ssd.yaml). Serves on **port 8890**
so it never collides with the arena-threshold serve on 8888.

---

## 1. The config

| Setting | Value | Why this value |
|---|---|---|
| image | `ghcr.io/bjk110/vllm-spark@sha256:d8492e76…` (stage-c, vLLM 0.21.1rc1) | only build with the padded `nvfp4_ds_mla` writer — see §5 |
| model | `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` | gated; 156 GB, 48 shards; ships an MTP/DSpark draft head (`num_nextn_predict_layers=1`) |
| `kv_cache_dtype` | `nvfp4_ds_mla` | the only route past ~1.5M tokens |
| `max_num_seqs` | **6** | 5 clients + 1 slot headroom. Spec-decode buffers scale with `max_num_seqs × (k+1)`, so anything larger silently steals memory from the KV pool |
| `num_speculative_tokens` | **5** | locked: must be ≤5 or a multiple of `n_predict=5`. k=7 boots then crashes on first generation |
| `gpu_memory_utilization` | **0.82** | 2,233,845 tokens. 0.84 gives 2,575,356 but startup becomes pathological — see §3 |
| `max_model_len` | 1,048,576 | full 1M context |
| KV offload | `OffloadingConnector` → `fs_python` tier at `/kvspill` (1.1 TB NVMe) | stock vLLM, no patches — see §2 |
| `cpu_bytes_to_use` | **2 GiB** | staging buffer only. GB10 is unified memory, so every GiB here is a GiB the GPU KV pool cannot use — 32 GiB OOM-killed the worker |
| `PYTORCH_CUDA_ALLOC_CONF` | `garbage_collection_threshold:0.9` | mandatory with any offload connector — see §2 |

KV pools **allocated** (abliterated model, `max_num_seqs 6`, k=5):

| util | KV pool | tokens | concurrency @ 1M ctx | startup | reached serving? |
|---|---:|---:|---:|---|---|
| 0.82 | 17.1 GiB | 2,233,845 | 2.13× | ~10 min | **no**, worker dies in warmup (§4) |
| 0.84 | 19.9 GiB | 2,575,356 | 2.46× | 25+ min | **no**, same, plus the §4 race |

> **Read these numbers correctly.** They are the pool size vLLM *reports at allocation*, taken
> from a boot that then died during `Warming up DeepSeek V4 sparse MLA attention`. **No NVFP4
> configuration has ever reached a serving state on this pair.** The largest KV pool that has
> actually served traffic is **1,458,744 tokens** (fp8, arena-threshold recipe); every log here
> that reaches a healthy `/health` shows at most 1.46M. Treat 2.2M/2.6M as *evidence NVFP4
> allocates what the arithmetic predicts*, not as a shipped capability. §4 is the open blocker.

---

## 2. SSD KV offload — this works, and UPSTREAM_GAPS #7 was too pessimistic

The earlier finding "disk KV offload is blocked" was based on testing **only LMCache**. Checking
the image directly:

| Connector | implements `SupportsHMA` |
|---|---|
| `LMCacheConnectorV1` | ❌ — the one previously tested |
| **`OffloadingConnector`** | ✅ `class OffloadingConnector(KVConnectorBase_V1, SupportsHMA)` |
| `SimpleCPUOffloadConnector` | ✅ |

This matters because `--kv-transfer-config` disables vLLM's hybrid KV manager (HMA) **only for
connectors that don't declare support**, and DeepSeek-V4's sparse-MLA sm120 decode requires HMA.
That is what produced the old `sparse_mla_sm120_decode_dsv4: num_tokens>64 (36 vs 64)` failure.
With `OffloadingConnector` that error does not occur.

Stock vLLM also already ships a disk tier — no custom image needed:

- `vllm/v1/kv_offload/factory.py` registers `TieringOffloadingSpec` (CPU primary tier +
  configurable secondary tiers)
- `vllm/v1/kv_offload/tiering/factory.py` registers
  `register_tier("fs_python", "vllm.v1.kv_offload.tiering.fs.manager", "FileSystemTierManager")`
- `FileSystemTierManager` — *"pure-Python disk-backed secondary tier"* — takes `root_dir`,
  `n_read_threads`, `n_write_threads`

Config used:

```yaml
--kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 2147483648,
    "eviction_policy": "lru",
    "secondary_tiers": [
      {"type": "fs_python", "root_dir": "/kvspill",
       "n_read_threads": 16, "n_write_threads": 16}
    ]
  }
}'
```

**Mandatory companion setting.** Any offload connector registers KV buffers, and an
expandable-segments VMM remap would invalidate them, so vLLM refuses outright:

```
ValueError: KV connector OffloadingConnector is incompatible with
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unless enable_cumem_alloc
```

Set `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9` instead.

**Unified-memory caveat.** On GB10 the CPU primary tier is not a separate pool — it is the same
DRAM the GPU allocates from. `cpu_bytes_to_use: 32 GiB` on top of util 0.84 OOM-killed the worker
(`Killed`, and the SSH session dropped from memory pressure). 8 GiB still did. 2 GiB is fine and
is all a staging buffer to NVMe needs.

---

## 3. Why util 0.82 and not 0.84

0.84 genuinely allocates **2,575,356 tokens (2.46×)** — it meets the ~2.5M target *at allocation
time*, though like 0.82 it never reached serving. But NVFP4 pool
quantization at that size runs 25+ minutes, during which the §4 race reliably kills the worker.
A production server that takes 25 minutes to restart *and* loses a coin flip on the way up is not
production-grade, so 0.82 is the shipped default. Changing it back is one line if the capacity
matters more than restart time.

Ceiling is bounded by real free memory, not by the util number: vLLM's profiler compares
`util × total` against **actual free memory**, and the worker node is the binding side.

```
ValueError: Free memory on device cuda:0 (96.75/121.69 GiB) on startup is less than ...
```

---

## 4. The open blocker: worker dies during sparse-MLA warmup

Every NVFP4 boot, util 0.84 **and** 0.82, dies with this on the worker node:

```
[Worker_TP1] Warming up DeepSeek V4 sparse MLA attention for mixed tokens=16 ...
[multiproc_executor] Parent process exited, terminating worker queues
```

and the head then hangs forever in `shm_broadcast`. **This is unresolved.** The child is reporting
that its parent (the headless `vllm serve` on the worker node) is gone; the question is why.

### Hypotheses tested and ruled out

| Hypothesis | Test | Result |
|---|---|---|
| systemd `RemoveIPC` reaps the worker's POSIX semaphores at session end | `loginctl enable-linger maci` on both nodes | Ruled out. Linger is enabled and this still reproduces. (It *was* a real, separate bug: it had corrupted several earlier "b12x deadlock" conclusions.) |
| The launcher is bound to the SSH session, so the parent dies when the session churns | Relaunched sparkrun inside a detached `screen` session | Ruled out. Reproduces identically under `screen`. |
| `run_headless` returns early because `start_worker_monitor(inline=True)` wakes spuriously | `sitecustomize.py` replacing the inline path with a real liveness poll | **Ruled out, and the hypothesis was wrong.** The patch loaded and was confirmed active, yet the parent still exited, which proves the parent is being killed *externally* rather than returning from the monitor. Patch reverted and deleted. |
| Kernel OOM-kills the parent during the NVFP4 quantization spike (GB10 unified memory) | `dmesg` oom-kill counters on both nodes, before and after | Ruled out. Zero kills on either node. |

### What is still consistent with the evidence

The parent is killed by something external that is not the OOM killer. Not yet examined: the
container runtime's own supervision (exit of PID 1 or a healthcheck in the sparkrun-generated
container), and whatever sparkrun does to the remote worker when the head's own startup exceeds an
internal timeout. That is the next thing to instrument: capture the worker parent's PID and its
exit status/signal at the moment of death, rather than inferring from the child's message.

### Why compose is not the workaround

The `docker-compose.dspark.yml` path in the tonyd2wild repo cannot substitute here: with this image
it fails multi-node TP outright, with each node computing `local_world_size=2` on its single GPU:

```
AssertionError: local_world_size (2) must be less than or equal to ...
AssertionError: DP adjusted local rank 1 is out of bounds.
```

despite correct `--nnodes 2 --node-rank 0/1 --headless` argv on both sides (verified with
`docker inspect`). Every boot that has ever served did so through sparkrun's `mp` backend.

---

## 5. Why not just use a newer image

vLLM 0.27 is better maintained (sparse-MLA + DSpark are upstream there, no overlay needed) but it
**cannot** reach this KV capacity. Verified directly in
`ghcr.io/bjk110/vllm-spark:v027-…` (`0.27.1.dev0+g4bdc8a788`):

- `config/cache.py` dtype list is `auto, fp8, fp8_ds_mla, fp8_e4m3, fp8_e5m2, nvfp4` —
  **no `nvfp4_ds_mla`**
- generic `nvfp4` is hard-rejected for MLA models, `config/vllm.py:2315`:

```python
if self.cache_config.cache_dtype == "nvfp4" and self.model_config.use_mla:
    raise ValueError("nvfp4 KV cache is not supported with MLA (Multi-head Latent "
                     "Attention) backends. Please use a different --kv-cache-dtype ...")
```

The reason is page geometry. MLA stores a packed composite page, not a flat tensor —
`mla/common.py`:

```python
if cache_dtype_str in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
    # fp8_ds_mla packed layout: 512 NoPE + 16 scales + 128 RoPE.
    return (num_blocks, block_size, 656)
return (num_blocks, block_size, head_size)
```

The generic `nvfp4` path returns a flat `head_size` with nowhere to put the scales or the
unquantized RoPE half, so upstream banned the combination rather than special-case it.

Their alternative, `fp8_ds_mla`, is a real improvement over naive fp8 but is still **656
bytes/token/layer** vs roughly **400** for an NVFP4 latent (RoPE stays bf16 either way, so NVFP4 KV
can never be a true 2× — the ceiling is ~40%). That ~40% is precisely what makes 2.2-2.6M reachable
instead of ~1.8M. The digest pin is therefore a deliberate capacity trade, revisitable if upstream
ever lands a padded NVFP4 MLA writer.

---

## 6. Parameter minimization

Started at 43 env vars, now **28**, every removal provable rather than guessed.

Audit method matters here: the base image is **misleading**, because the recipe's `pre_exec`
rebuilds the stage-c runtime and that overlay adds code reading several `VLLM_DSPARK_*` vars that
do not exist in the base image. All checks below were run against the **patched runtime inside the
running container**.

**Removed — never read** (name appears nowhere in `vllm/` or `b12x/`, or vLLM logs
`Unknown vLLM environment variable detected`):

`PYTHONUNBUFFERED`, `HF_HUB_DISABLE_XET`, `TORCH_EXTENSIONS_DIR`,
`VLLM_SKIP_INIT_MEMORY_CHECK`, `VLLM_TRITON_MLA_SPARSE`

**Removed — redundant** (recipe sets exactly the code default, e.g.
`getenv("DSPARK_SLOT_CLAMP", "1")` with the recipe also `"1"`):

| var | value = default |
|---|---|
| `DSPARK_SLOT_CLAMP` | 1 |
| `VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M` | 16 |
| `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM` | 0 |
| `VLLM_DSPARK_CONFIDENCE_THRESHOLD` | 0.0 |
| `VLLM_DSPARK_FUSED_MARKOV_ARGMAX` | 0 |
| `VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP` | 1 |
| `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT` | 0 |
| `VLLM_DSV4_B12X_COMPRESSED_MLA` | 0 |
| `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE` | 0 |
| `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT` | 0 |

**Kept — genuinely differ from defaults**, so they are real tuning decisions:
`VLLM_DSPARK_CONFIDENCE_SCHEDULER=off` (default `auto`),
`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` (default 0),
`VLLM_DSPARK_LOCAL_ARGMAX=1` (default 0),
`VLLM_DSPARK_REPLICATE_MARKOV_W1=1` (default 0),
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` (default 512),
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (default 1),
plus `VLLM_USE_B12X_MOE=1` / `VLLM_USE_B12X_WO_PROJECTION=1` (b12x MoE is the single biggest speed
contributor), the JIT cache dirs, the arch lists, and the NCCL/alloc settings.

Remaining candidate: `--pipeline-parallel-size 1` is the vLLM default and can go.

---

## 7. Housekeeping done on the nodes

Roughly **317 GB of disk** and **46 GB of RAM** reclaimed:

| Item | Reclaimed |
|---|---|
| `gpustack-data` volume | 183 GB |
| gpustack images (`latest`, `v2.2.1`, `v2.1.1`) + worker container | ~16 GB |
| dangling images (n0 54.6 GB + n1 33.7 GB) | 88 GB |
| exited `docker-llama-gen-1`, `docker-llama-embed-1` | — |
| **stale `/dev/shm` segments** (n0 28 GB, n1 18 GB) | **46 GB RAM** |

The `/dev/shm` sweep is the one to remember. Dozens of orphaned `psm_*` / `nccl-*` / `sem.mp-*`
segments accumulate from crashed runs, and on unified memory tmpfs consumes the same DRAM the GPU
allocates from — so they were directly capping `gpu_memory_utilization` via the
`Free memory on device` guard. Both nodes went from ~99 GB to **117 GB available**.

Add to a clean-restart routine, once all engine processes are stopped:

```bash
find /dev/shm -maxdepth 1 \
  \( -name 'psm_*' -o -name 'nccl-*' -o -name 'sem.mp-*' -o -name 'mp-*' \) -delete
```
