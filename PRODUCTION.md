# Production setup: 3-5 concurrent sessions on 2x DGX Spark

The shipped configuration, and the measurements behind every value in it.

**Recipe:** [`examples/eugr-prod.yaml`](examples/eugr-prod.yaml) · **Port:** 8000

```bash
bash ~/spark-launch.sh eugr-prod.yaml ~/PROD.log   # teardown + /dev/shm sweep + launch
curl -s localhost:8000/health                      # expect 200
```

---

## 1. The configuration

| Setting | Value | Why this value |
|---|---|---|
| container | `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:2026081903` | newest eugr build; ships b12x, DSpark and MLA KV quants together |
| model | `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` | gated, 156 GB, 48 shards, ships the DSpark draft head |
| `kv_cache_dtype` | **`fp8_ds_mla`** | the only MLA KV quant DeepSeek-V4 can use on this image (§4) |
| `gpu_memory_utilization` | **0.89** | measured ceiling: 0.91 kills the worker, 0.92 fails at startup (§3) |
| `max_num_seqs` | **6** | swept: 8 and 12 both lose throughput *and* KV (§2) |
| `num_speculative_tokens` | **5** | k=3 is rejected by this image; only 5 and 10 are legal (§2) |
| `max_model_len` | 1,048,576 | full 1M context; the KV pool must hold one max-length request |
| backends | `--moe-backend b12x --linear-backend b12x --attention-backend B12X_MLA_SPARSE` | missing b12x is roughly half speed |
| patches / hooks | **none** | stock image, 73-line recipe, no overlay build |

## 2. Measured performance

Warm, 128 tok/req, aggregate tok/s across concurrent streams.

| concurrency | 1 | 3 | 5 | 6 |
|---|---:|---:|---:|---:|
| aggregate tok/s | ~51-60 | ~105 | **~140** | ~109 |

KV pool: **~1.66M tokens** (1.58x a full 1M context).

Reproducibility: c5 measured 140.3 / 135.0 / 140.2 across three independent
boots, so treat anything under ~5% as noise. c1 swings more (51-60); weight c3
and c5 when comparing configs.

### `max_num_seqs` sweep

| config | slots (`seqs x (k+1)`) | KV tokens | c3 | c5 |
|---|---:|---:|---:|---:|
| **6, k=5 (shipped)** | 36 | **1,663,439** | **105.6** | **140.2** |
| 8, k=5 | 48 | 1,638,922 | 102.7 | 129.4 |
| 12, k=5 | 72 | 1,635,809 | 92.9 | 124.0 |

Monotonic. With only 5 live streams the extra slots are never filled, so they
buy nothing while their spec-decode buffers take memory from the KV pool. The
c6 rolloff is therefore a **compute** limit, not a slot limit.

### k is not tunable

```
DSpark requires num_speculative_tokens >= dspark_block_size (5); got 3
```

k must be **>= 5 and a multiple of 5**. Only 5 and 10 are legal. (On stage-c,
k=3 was legal and measured 6.7% faster, which is why it was worth testing.)

## 3. The utilization ceiling

| util | KV tokens | outcome |
|---|---:|---|
| 0.85 | ~1.00M | **fails**: under the 11.04 GiB one 1M request needs |
| **0.89** | **~1.66M** | **serves** |
| 0.91 | 1,941,101 | allocates, then the worker is **SIGKILLed** mid-allocation |
| 0.92 | n/a | **fails at startup**: `Free memory on device cuda:0 (111.46/121.69 GiB)` |

`fp8_ds_mla` costs **~11.0 KB/token**, derived from vLLM's own sizing error
(11.04 GiB for 1,048,576 tokens).

At 0.91 the head logs the full pool and continues while the worker dies
**silently**: no error, no traceback, no CUDA OOM, and not the OOM killer either
(`/proc/vmstat oom_kill` is 0 on both nodes, container `OOMKilled=false`). The
only evidence is bash reporting the signal in the worker's own in-container log:

```
/tmp/sparkrun_serve.sh: line 26:   120 Killed   vllm serve drowzeys/...
```

Always clear `/dev/shm` on **both** nodes between runs. Two boots of the same
config differed by 22k KV tokens purely from leftover segments;
[`scripts/spark-launch.sh`](scripts/spark-launch.sh) does this automatically.

## 4. What this setup does NOT do

Both of these were required by the original goal and neither is achievable on
this hardware and software. They are recorded so nobody re-spends the time.

### SSD KV offload: closed

`OffloadingConnector` with an `fs_python` disk tier fails on **both** stacks:

| stack | result |
|---|---|
| eugr, b12x attention | `CUDA error: an illegal memory access was encountered` |
| eugr, default attention | same illegal memory access (so not a b12x interaction) |
| stage-c + NVFP4 | no IMA; allocates 2,216,035 tokens, then **deadlocks** in CUDA-graph capture (worker finishes, head stalls at 20%, both GPUs 0%) |

The configuration itself is correct: the flag renders properly, `/kvspill` is
bound from NVMe (ext4, 1.6 TB free), and `PYTORCH_CUDA_ALLOC_CONF` is set to
`garbage_collection_threshold:0.9` because offload connectors reject
`expandable_segments` outright.

**Deliberately not worked around.** An illegal memory access means the KV buffers
the connector registers are not laid out the way the kernels read them, and that
class of bug can silently produce wrong tokens instead of crashing. Shipping it
would be worse than shipping without offload.
[`examples/eugr-prod-ssd.yaml`](examples/eugr-prod-ssd.yaml) is kept as a record.

Context: the 1.66M pool already holds ~1.6 full 1M-context sessions, so offload
would buy cross-session reuse rather than capacity that is currently short.

### ~2.5M KV: not on this config

NVFP4 KV is the only way to reach it. `nvfp4_ds_mla` **can** be enabled on the
eugr image with an 89-line patch (see
[vllm-spark-nvfp4](https://github.com/maci0/vllm-spark-nvfp4)
`Dockerfile.eugr-nvfp4`), correcting the earlier claim here that the capability
was absent. It allocates, passes the forward pass, and is measurably cheaper per
token. It is still **not competitive**, and the reason is worth recording.

| | eugr `fp8_ds_mla` (shipped) | eugr + NVFP4 patch |
|---|---:|---:|
| bytes/token | 11,315 | **8,864 (-22%)** |
| best KV pool | **1,674,044** | 1,507,777 |
| serves | **yes** | no |
| c5 tok/s | **140.2** | never reached |
| cudagraphs | `FULL_AND_PIECEWISE` | OOMs in PIECEWISE capture |

**Correction: the per-token saving is not real.** An earlier version of this
section called it "reproduced three ways". Those three readings were not
independent, they all derive from the same page-size computation. Direct
measurement in `TUNING.md` puts fp8 at 9,094 B/token and nvfp4_ds_mla at 9,083,
i.e. identical, and both known implementations of the format use the same
584-byte envelope. The 8,864 figure is one KV group being sized with a
432-byte generic-NVFP4 page while the writer emits 584, a 26% **under**
allocation. See [EUGR_NVFP4.md](https://github.com/maci0/vllm-spark-nvfp4/blob/main/EUGR_NVFP4.md) §4.

Nor is non-KV memory the obstacle: inverting vLLM's own sizing formula
(`available_kv = util x total - non_kv - cudagraph_estimate`) gives 90.90 GiB
non-KV for the fp8 run and 91.89 GiB for the NVFP4 run, a difference of
**+0.98 GiB**, which is the pre-profiling dequant staging buffer in
`flashinfer.py`. The apparent ~6 GiB gap was simply the 0.04 utilization given
back. **NVFP4 is not the route past 1.7M.**

The patch itself is correct and independently useful (it is a legitimate eugr
contribution, and unlike open PR
[#311](https://github.com/eugr/spark-vllm-docker/pull/311) it makes the stock
b12x image NVFP4-capable in 89 lines rather than wrapping a third-party runtime
in 25,000). It is simply not a better production config today.

The alternative, [`examples/stagec-nvfp4-prod.yaml`](examples/stagec-nvfp4-prod.yaml),
**does** serve 2,198,373 tokens, and is the right choice only if total context is
the binding constraint:

| | eugr (shipped) | stage-c |
|---|---:|---:|
| KV pool | 1,663,439 | **2,198,373** (+31%) |
| c5 aggregate | **140.2** | 109.7 (-22%) |
| vLLM | Aug 15 source | **0.21.1rc1** on a 2-month-old base |
| recipe | **73 lines, 0 hooks** | 344 lines, 2 `pre_exec` hooks |

The newer image is both newer and faster; stage-c's only advantage is capacity.

### The two routes that looked like they reach ~2.5M

> **Tested 2026-08-22. Route 1 does not work here.** Pinning the pool was run at
> 19, 22, 24 and 26.3 GiB. Every size **allocates** (up to 2,740,813 tokens,
> matching nepenth's figure exactly) and every size **fails to run**, dying in
> workspace allocation or swapping the node to a standstill. The blocker is the
> b12x workspace footprint, and `deep_gemm` is not available here because the
> module is not installed in the eugr image at all. Route 2 (PP=2) remains
> untested. Full ladder and teardown traps in [KV_CEILING.md](KV_CEILING.md).

Both are on `fp8_ds_mla`, the dtype already shipped here. Neither needs NVFP4.

**1. Pin the KV pool instead of inferring it from utilization.**
[`nepenth/deepseek-v4-flash-gb10`](https://github.com/nepenth/deepseek-v4-flash-gb10)
(2026-08-21) runs the same checkpoint, TP=2, `max_num_seqs 6`, DSpark k=5, and
reports:

```
Available KV cache memory: 26.3 GiB
GPU KV cache size: 2,740,813 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.61x
```

with a 1.04M three-needle retrieval PASS, so it is validated beyond the boot
line. The lever is `--kv-cache-memory 28235618304` at GMU **0.84**, which is
*lower* than the 0.89 used here. Pinning the pool sidesteps the utilization
cliff entirely (0.89 serves, 0.91 gets SIGKILLed) rather than searching for it.
At the measured 11,161 B/token, ~26.0 GiB clears 2.5M.

**2. Pipeline parallel instead of tensor parallel.**
vLLM replicates the MLA latent KV on every rank and never head-shards it, so
under TP=2 **both nodes hold the same KV** and half the pool is a duplicate.
Under PP=2 each rank builds only its half of the layers, so the per-worker
`num_blocks = available_memory // page_size // num_layers` doubles:

| config | B/token/node | cluster tokens |
|---|---:|---:|
| TP=2 (current) | 11,315 | 1,651,180 |
| **PP=2** | ~5,658 | **~3,302,000** |

Costs: pipeline bubbles (mitigate with `--async-scheduling`), the DSpark drafter
pinned to the last stage clamping `num_blocks` via `min()` (rebalance with
`VLLM_PP_LAYER_PARTITION=31,30`), and b12x backends under PP are unverified.
The usual "TP beats PP" result assumes fast intra-node fabric; here TP=2 is
inherently cross-node with no GPUDirect RDMA, so PP may also be faster.

### Known trap in the current config

`cudagraph_mode: FULL_AND_PIECEWISE` plus chunked prefill is reported to hang
DeepSeek-V4-Flash on sm_12x after 5-6 requests
([vllm#40969](https://github.com/vllm-project/vllm/issues/40969), open); the
confirmed workaround is `PIECEWISE`. Also watch for a **negative** "Estimated
CUDA graph memory" line on unified memory
([vllm#46932](https://github.com/vllm-project/vllm/issues/46932)): being
subtracted, it inflates the KV budget into an OOM.

### Images evaluated and rejected

| source | why not |
|---|---|
| [blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker) (`voipmonitor/vllm`, `voipmonitor/sglang`) | **amd64 only**, every tag in all three repos. Targets RTX PRO 6000 Blackwell (SM120, 96 GB discrete VRAM), not GB10 (sm_121a, arm64, unified). Its DSv4 compose is single-node TP=2 across two local GPUs at `util 0.975` and `max_model_len 131072`; none of those assumptions hold when the OS shares the same DRAM. |
| TensorRT-LLM | `get_sm_version()` returns **121** on GB10, which matches no branch: `== 120` misses it, `is_sm_100f()` is `>=100 and <110`. Its DSA sparse-attention guards are `>= 100`, so they *admit* SM121 and then run SM100-built kernels that abort at launch. No published 2-node DGX Spark reproduction exists. |

## 5. Still untested

`draft_sample_method: greedy` (bjk110 ships greedy), k=10,
`max_num_batched_tokens`, `cudagraph_mode: FULL_DECODE_ONLY`, and the b12x env
flags individually (`MHC`, `SPARSE_INDEXER`, `FP8_GEMM` each measured harmful on
stage-c; removing all three at once failed to boot, so they need one at a time).

## 6. Operating notes

- **Launch only via [`scripts/spark-launch.sh`](scripts/spark-launch.sh).** It tears down both nodes, sweeps `/dev/shm` on both, prints free memory, then launches into a detached `screen`.
- **sparkrun is not a daemon.** It exits after `[6/6] Post-launch hooks` while the server is still coming up. A gone `screen` session is not a failure.
- **The worker's real log is inside its container**, not `docker logs`:
  ```bash
  C=$(ssh worker 'docker ps --format "{{.Names}}" | grep sparkrun')
  ssh worker "docker exec $C tail -50 /tmp/sparkrun_serve.log"
  ```
- Boot takes ~10-12 minutes (weight load ~250 s, then `torch.compile` and CUDA-graph capture).
