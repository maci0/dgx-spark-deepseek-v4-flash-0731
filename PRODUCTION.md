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

NVFP4 KV (~6.6 KB/token) would reach it, but DeepSeek-V4 cannot use `nvfp4_ds_mla`
on the eugr image: the dtype is generic MLA support (none of its 10 files are
under `models/deepseek_v4/`), while DeepSeek-V4 has its own attention path whose
sm12x class is fp8-only. Details in [EUGR_B12X_PROD.md](EUGR_B12X_PROD.md) §4.

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
