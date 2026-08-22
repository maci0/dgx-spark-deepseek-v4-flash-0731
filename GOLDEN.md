# Golden deployment: anemll NVFP4, 2.0M KV, 2x DGX Spark

The shipped configuration as of 2026-08-22. Every number below is measured on
this cluster with one harness, not quoted from upstream.

**Recipe:** [`examples/anemll-nvfp4-golden.yaml`](examples/anemll-nvfp4-golden.yaml)
· **Endpoint:** `http://192.168.0.211:8000/v1`
· **Model names:** `deepseek-v4-flash`, `dsv4`, or the full HF path

```bash
curl -s http://192.168.0.211:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

The aliases are declared with `--served-model-name` **alongside** the full HF
path, not instead of it, so existing clients keep working and swapping the
underlying checkpoint later does not force every caller to change.

```bash
bash ~/spark-launch.sh anemll-nvfp4.yaml ~/anemll.log
curl -s localhost:8000/health                       # 200
```

---

## 1. What it is

| Setting | Value | Why |
|---|---|---|
| container | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | the only image tested that delivers **real** NVFP4 KV compression |
| model | `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` | abliterated, DSpark draft head bundled |
| `kv_cache_dtype` | **`nvfp4_ds_mla`** | 7,650 B/token, 32% cheaper than fp8_ds_mla (§3) |
| `gpu_memory_utilization` | **0.82** | measured ceiling: 0.835 allocates 2,227,486 then dies in FlashInfer autotune (§4) |
| `max_num_seqs` | 6 | 5 concurrent clients plus headroom |
| `num_speculative_tokens` | 5 (DSpark) | k=5 is the floor this family accepts |
| `max_cudagraph_capture_size` | 36 | exactly `max_num_seqs x (k+1)`; nothing reachable is dropped |
| `max_model_len` | 1,048,576 | full 1M context |
| patches | **none** | stock image |

## 2. Measured, three lineages, one harness

Warm, 128 tok/req, single shared coding prompt at temperature 0.7, aggregate
tok/s. Prompt choice matters enormously here: see §5.

| | **anemll (shipped)** | eugr + PIECEWISE | stage-c (tonyd2wild) |
|---|---:|---:|---:|
| KV pool | **2,002,497** | 1,768,024 | 1,438,916 |
| bytes/token | **7,650** | 11,317 | ~11,900 |
| max concurrency @ 1M | **1.91x** | 1.58x | 1.37x |
| c1 | 51.4 | 54.3 | **56.1** |
| c3 | **112.7** | 90.3 | 93.5 |
| c5 | 126.2 | **127.4** | 116.0 |
| c6 | **157.9** | ~109 | 141.1 |
| vLLM | 0.25.2 (Jul) | 0.27.x (Aug) | 0.21.1rc1 |

anemll wins on capacity (+13% over eugr, +39% over stage-c) and on multi-client
throughput (c3 +25%, c6 +45%). It gives up ~5% at c1, which is the least
relevant case for a 5-client workload.

## 3. The NVFP4 saving is real here, and only here

**7,650 bytes/token against fp8_ds_mla's 11,317, a 32% reduction.** This is the
only authentic NVFP4 KV saving found across three lineages:

- Our own NVFP4 patch on the eugr image appeared to give 22%. It did not. One KV
  group was sized with a 432-byte page while the writer emitted 584, a 26%
  **under**-allocation. Direct measurement puts fp8 at 9,094 B/token and
  nvfp4_ds_mla at 9,083, i.e. identical. See
  [vllm-spark-nvfp4/EUGR_NVFP4.md](https://github.com/maci0/vllm-spark-nvfp4/blob/main/EUGR_NVFP4.md).
- stage-c is labelled NVFP4 and measures ~11,900 B/token, barely different from
  fp8.

anemll's image was built around the format rather than retrofitted with it,
which is the difference.

## 4. The utilization ceiling is 0.82, and 0.835 fails specifically

| util | KV pool | outcome |
|---|---:|---|
| 0.835 (MiaAI-Lab's value) | 2,227,486 | allocates, worker SIGKILLed during **FlashInfer sparse-MLA autotune** |
| **0.82** | **2,002,497** | **serves** |

The failure is not graph capture and not the arena. It is FlashInfer's
autotuning step allocating workspaces on top of a 16.25 GiB arena. The eugr
image skips that step entirely (`Skipping FlashInfer autotune because no
FlashInfer...`), which is part of why it tolerates a higher utilization.

There is no env var to disable it in this image; only
`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`, which would need a pre-warmed cache.
Anyone wanting MiaAI-Lab's 2.49M should start by warming that cache.

## 5. Throughput is workload-dependent, by a factor of ~1.7

Measured on the same server, same config, differing only in prompts:

| harness | c5 aggregate |
|---|---:|
| one shared coding prompt, natural EOS, temp 0.7 | **141 tok/s** |
| unique generic prose prompts, `ignore_eos` forced | **81 tok/s** |

Two effects compound: unique prompts defeat prefix-cache sharing across the five
streams, and forced continuation past natural EOS collapses DSpark draft
acceptance (code is where the draft head predicts well). Quote a tok/s number
for this stack only alongside the prompt shape that produced it.

## 6. Operating notes

- **`/health` is not a liveness check.** It only proves the API server is up. A
  hung worker leaves it returning 200 while generation blocks indefinitely. Use a
  tiny generation request instead. The early warning in the log is
  `No available shared memory broadcast block`, repeated.
- **`docker rm -f` does not release memory from a wedged vLLM container.**
  `pkill -9 -f 'VLLM::'` does, and recovers a node from 118-123 GiB used back to
  ~4 GiB without a reboot. Always confirm both nodes are under ~10 GiB used
  before launching; four runs were once wasted on a node that never released
  memory from the previous failure.
- **Launch only via `spark-launch.sh`**, which tears down both nodes, sweeps
  `/dev/shm`, drops page cache and refuses configs that exceed physical memory.
- Boot takes ~8-10 minutes. `stall: 1` during startup is usually transient;
  three or more means the worker is gone.

## 7. What this does not do

**~2.5M KV: not reached.** 2.0M is 80% of the target. The ceiling is FlashInfer
autotune at util 0.835, and above that the constraint is 81.34 GiB of weights
leaving ~30 GiB for arena, workspaces and graphs. Details in
[KV_CEILING.md](KV_CEILING.md).

**SSD offload: does not work for this model.** Built and tested end to end on the
eugr image with the `fs` tier on NVMe. Faults with `cudaErrorIllegalAddress`
under both `fp8_ds_mla` and flat `fp8`, because vLLM's offload transfer path
assumes a single flat layout while DeepSeek-V4 builds a hybrid multi-group cache.
Full test in
[vllm-spark-nvfp4/KV_OFFLOAD_MLA.md](https://github.com/maci0/vllm-spark-nvfp4/blob/main/KV_OFFLOAD_MLA.md).

**Rollback:** `bash ~/spark-launch.sh eugr-prod.yaml ~/PROD.log` gives 1,768,024
tokens on vLLM 0.27.x, five weeks newer, at the cost of 13% capacity and 45% of
c6 throughput.
