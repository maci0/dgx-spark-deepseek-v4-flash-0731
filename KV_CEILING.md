# The KV capacity ceiling on 2x DGX Spark, and why ~2.5M is out of reach

Measured 2026-08-22 on 2x DGX Spark (GB10, 121.69 GiB unified each, TP=2),
serving `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` on the
stock eugr b12x image.

**Short version:** arenas up to 2,740,813 tokens *allocate* successfully. None
above the utilization-derived ~17.8 GiB will *run*. The wall is not the arena,
it is what has to fit alongside it.

---

## 1. The budget

vLLM reports its own numbers at boot, and they close exactly:

```
Free memory on device        111.31 / 121.69 GiB
Model loading took            81.34 GiB
Available KV cache memory     17.81 GiB   ->  1,692,431 tokens (11,317 B/token)
```

`81.34 + 17.81 + 9.15 (activations, graphs, workspaces) = 108.30`, which is
`util 0.89 x 121.69`. Host free is 117 GiB, so the container and slack account
for the remaining ~8.7 GiB.

Note the model figure: **81.34 GiB, not the 77.7 GiB the raw weights suggest**
(155.4 / TP 2). The extra ~3.6 GiB is the DSpark draft head and load overhead.
Sizing arithmetic that starts from the checkpoint size will be optimistic by
exactly that much.

## 2. The ladder, measured

Every row was run; none is projected. Rows below the line were re-run on
verified-clean nodes after discovering the contamination described in §4.

| arena | tokens | outcome |
|---:|---:|---|
| **17.81 GiB** (derived) | **1,692,431** | **serves**, c5 127-141 tok/s |
| 19.0 GiB | 1,980,308 | worker SIGKILLed at `kernel_warmup` |
| 22.0 GiB | 2,292,993 | worker died, 2.6 GiB swapped |
| 24.0 GiB | 2,501,483 | allocates, then swaps: worker CPU falls to 27%, 4.3 GiB in swap |
| 26.3 GiB | 2,740,813 | allocates, dies during CUDA graph capture |

The 24 GiB row is the instructive one. It reaches the ~2.5M target exactly
(projected 2,501,458 at the measured 10,302 B/token) and then crawls rather than
failing, because CUDA memory is unswappable: the kernel pushes everything else
to swap instead, and the worker blocks on IO.

## 3. Two things that make this counterintuitive

**`--kv-cache-memory-bytes` ignores `gpu_memory_utilization`.** From vLLM's own
docstring: *"kv_cache_memory_bytes (when not-None) ignores
gpu_memory_utilization"*. Treating util as a floor the pinned arena must sit
under is wrong, and juggling it wastes boots. Util still matters for one thing:
vLLM refuses to start if `util x total` exceeds device-free memory.

**Freeing host memory can destabilise a working config.** With a *derived*
arena, vLLM sizes the pool from measured free memory. Trimming services freed
~1.5 GiB, the derived pool grew 17.81 -> 18.45 GiB, and the same recipe that had
served then died in capture. The utilization value is not portable across
environment changes.

## 4. A trap that invalidated four runs

`docker rm -f` does **not** release memory from a wedged vLLM container. After
an OOM the container reports `running`, and spark1 sat at 118-123 GiB used
across subsequent launches, so every "failure" was really a launch onto a node
with ~6 GiB free.

The reliable teardown is to kill the processes directly:

```bash
pkill -9 -f 'VLLM::'          # then docker rm -f
```

That recovered the node from 123 GiB used to ~4 GiB twice, without the reboots
the earlier attempts cost. `spark-launch.sh` should assert both nodes are under
~10 GiB used before launching; a polluted node otherwise produces a confident,
meaningless result.

## 5. Why nepenth fits 26.3 GiB and this config cannot

[`nepenth/deepseek-v4-flash-gb10`](https://github.com/nepenth/deepseek-v4-flash-gb10)
reports 2,740,813 tokens with a 26.3 GiB arena on the same hardware, checkpoint,
TP, `max_num_seqs` and k. Their own notes say it *"leaves ~0.6 GiB"*, so they run
it to the wall. The difference is backend footprint, not tuning:

| | nepenth | here |
|---|---|---|
| MoE backend | `deep_gemm` | `b12x` |
| attention backend | default (FlashInfer SM120 via their patch 0027) | `B12X_MLA_SPARSE` |
| linear backend | default | `b12x` |
| image | vLLM 0.27.1 from source + DeepGEMM `2fd6732` + 28 patches | stock eugr b12x |

b12x carries persistent workspaces for MoE, linear and attention that their
configuration does not. **`deep_gemm` is not an option on the eugr image: the
module is not installed at all** (`ModuleNotFoundError: No module named
'deep_gemm'`), which is precisely why b12x exists in this lineage.

Reaching ~2.5M therefore requires building their image, not changing flags here.

## 6. What this config delivers

| | |
|---|---|
| KV pool | 1,659,937 - 1,692,431 tokens across four boots |
| max concurrency at 1M context | 1.58x |
| throughput | c1 ~54, c3 ~90-110, c5 **127-141 tok/s** aggregate |
| patches | none, stock image |

Treat ~1.67M and ~135 tok/s at c5 as the steady-state figures; single boots vary
by a few percent.

Disk offload is not an alternative route past this ceiling: see
[KV_OFFLOAD_MLA.md](https://github.com/maci0/vllm-spark-nvfp4/blob/main/KV_OFFLOAD_MLA.md)
for the end-to-end test showing it faults under every KV dtype on this model.
