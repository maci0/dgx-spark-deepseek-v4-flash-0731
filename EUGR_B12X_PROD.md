# Production on the eugr b12x image: fp8_ds_mla KV + DSpark + b12x

Target: 3-5 concurrent sessions, a 1.5-2.5M token KV pool, DSpark speculative
decoding, b12x kernels, and as few patches as possible.

This replaces the stage-c path ([PROD_C5_SSD.md](PROD_C5_SSD.md)), which never
reached a serving state. The eugr image serves, and it does so with a **73-line
recipe and zero `pre_exec` hooks**, against 344 lines plus an overlay rebuild.

Recipe: [`examples/eugr-prod.yaml`](examples/eugr-prod.yaml). Stock image, no patches.
[`examples/eugr-prod-nvfp4.yaml`](examples/eugr-prod-nvfp4.yaml) is kept only as a
record of the NVFP4 attempt that **does not work**, see §4.

---

## 1. Why this image

Every candidate was inspected directly rather than trusted from its name
(`docker run --entrypoint bash <img>` + grep of the installed `vllm` tree):

| Image | vLLM build | `nvfp4_ds_mla` | `fp8_ds_mla` | b12x | DSpark |
|---|---|---|---|---|---|
| **eugr nightly `2026081903`** | `dev20003` (Aug 15) | ✅ 10 files | ✅ 27 | ✅ 74 refs | ✅ 48 |
| eugr nightly `latest` | `dev19043` (Aug 14) | ✅ 10 | ✅ 23 | ✅ 68 | ✅ 36 |
| `arena-b12x-modbaked` | `dev19043` (Aug 13) | ✅ 10 | ✅ 23 | ✅ 68 | ✅ 36 |
| `eugr/spark-vllm-b12x:latest` | `dev19024` (Aug 9) | ✅ 10 | ✅ 23 | ✅ 67 | ✅ 34 |
| stage-c / `tonyd2wild-arena` | `0.21.1rc1` | ✅ 6 | ✅ 25 | ✅ 45 | ✅ 21 |

`2026081903` is newest on every axis, so it is the one shipped here.

**bjk110 v027 is not a candidate for this workload.** Its own production preset
(`presets/deepseek-v4-flash-0731-dspark-k7-256k-v027-candidate-tp2.env`) runs
`--moe-backend marlin`, i.e. no b12x, and targets 256K context with a 10 GiB FP8
KV pool at `MAX_NUM_SEQS=1`. Useful ideas were still taken from it: see
`--kv-cache-memory-bytes` in §5.

---

## 2. The config

| Setting | Value | Why |
|---|---|---|
| container | `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:2026081903` | newest build with b12x + both MLA KV quants + DSpark |
| model | `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` | gated, 156 GB, ships the DSpark draft head |
| `kv_cache_dtype` | **`fp8_ds_mla`** | the only MLA KV quant DeepSeek-V4 supports on this image, see §4 |
| `max_num_seqs` | **6** | 5 sessions + 1 headroom. Spec-decode buffers scale with `max_num_seqs x (k+1)`, so raising this silently takes memory from the KV pool |
| `num_speculative_tokens` | **5** | must be <=5 or a multiple of `n_predict=5`; k=7 boots then crashes on first generation |
| `gpu_memory_utilization` | **0.89** | 0.85 leaves too little KV for a single 1M request, see §3 |
| `max_model_len` | 1,048,576 | vLLM requires the KV pool to hold at least one max-length request |
| backends | `--moe-backend b12x --linear-backend b12x --attention-backend B12X_MLA_SPARSE` | confirmed live via `Using 'B12X' Mxfp4 MoE backend` |

No `pre_exec` hooks, no overlay rebuild, no offload connector, no `kill` shim.

---

## 3. KV capacity, measured

The per-token cost was measured on this image, not assumed. vLLM's own sizing
error states the arithmetic exactly:

```
To serve at least one request with the model's max seq len (1048576),
(11.04 GiB KV cache is needed, which is larger than the available KV cache
memory (10.85 GiB) ... estimated maximum model length is 1001216
```

11.04 GiB / 1,048,576 tokens = **~11.0 KB/token for `fp8_ds_mla`**.

| dtype | KB/token | 1.5M needs | 2.5M needs | available here |
|---|---:|---:|---:|---|
| `fp8_ds_mla` | ~11.0 (measured) | 15.8 GiB (util ~0.89) | 26.3 GiB (util ~0.98) | **yes** |
| `nvfp4_ds_mla` | ~6.6 (projected) | 9.4 GiB | 15.7 GiB | **no**, see §4 |

**Consequence: ~2.5M is out of reach on this image.** It would require NVFP4 KV,
which DeepSeek-V4 does not support here, and util ~0.98 on fp8 leaves nothing for
activations. Realistic range on the eugr lineage is **1.65M at util 0.89**, with
higher utilization traded against stability.

Measured at util 0.89: **`fp8_ds_mla` = 1,652,056 tokens** (1.58x a full 1M
context), serving.

---

## 4. `nvfp4_ds_mla` does NOT work on this image. Do not chase it.

**Tested and closed.** DeepSeek-V4 cannot use NVFP4 KV on the eugr image. The
capability is genuinely absent, not merely gated, and there are two separate
gates that must both be understood before anyone retries this.

**Gate 1, `vllm/config/vllm.py` (over-broad, patchable):**

```python
if (self.cache_config.cache_dtype.startswith("nvfp4")
        and self.model_config.use_mla):
    raise ValueError("nvfp4 KV cache is not supported with MLA ...")
```

`startswith("nvfp4")` also matches `nvfp4_ds_mla`. Narrowing it to exact
`"nvfp4"` does let config validation pass, and
[`scripts/sitecustomize-nvfp4-mla-guard.py`](scripts/sitecustomize-nvfp4-mla-guard.py)
does exactly that. **It gets you to gate 2 and no further.**

**Gate 2, `vllm/models/deepseek_v4/attention*.py` (real, do NOT patch):**

```python
assert kv_cache_dtype.startswith("fp8"), (
    "DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got nvfp4_ds_mla")
```

The DeepSeek-V4 model path hardcodes the fp8 packed layout. The generic MLA
layers (`mla.py`, `mla_cache_format.py`, `b12x_mla_sparse.py`) do carry
`nvfp4_ds_mla` support, which is why grepping the image suggests it works, but
the DeepSeek-V4 writer is fp8-only. Forcing this assert would mismatch the cache
layout and produce silent numerical corruption instead of a clean failure, so it
is deliberately left alone. Supplying that writer is exactly what the stage-c
overlay's "padded `nvfp4_ds_mla` writer" existed for.

**Practical consequence:** on the eugr lineage the KV pool is fp8_ds_mla at
~11.0 KB/token, and capacity is bought only with `gpu_memory_utilization`. The
~2.5M target needs the stage-c stack, which has never reached a serving state
(see [PROD_C5_SSD.md](PROD_C5_SSD.md)).

The patch script is kept only as documentation of gate 1 and is **not** part of
the shipped recipe. If revisiting: `VllmConfig` is a pydantic *dataclass* whose
schema is built at class creation, so mutating
`__pydantic_decorators__[...].func` does nothing and `rebuild_dataclass(force=True)`
re-collects the original. Only rewriting the module source before class creation
works, and it must be verified on the compiled object rather than with
`inspect.getsource`, which re-reads the file and shows unpatched text:

```python
f = VllmConfig.validate_nvfp4_kv_cache_with_mla
assert "startswith" not in f.__code__.co_names
```

---

## 5. Ideas taken from the bjk110 v027 preset

`--kv-cache-memory-bytes 10737418240` pins the KV pool to an exact byte count
instead of reverse-engineering it from `gpu_memory_utilization`. Given the
measured KB/token in §3 this is the more deterministic dial:

```
1.5M tokens  fp8_ds_mla -> --kv-cache-memory-bytes 17716740096   (~16.5 GiB)
2.0M tokens  fp8_ds_mla -> --kv-cache-memory-bytes 23622320128   (~22.0 GiB)
```

Their DSpark also runs **k=7 greedy**, against the k<=5 limit measured on the old
stack. Worth re-testing here, since native DSpark in this lineage may lift it.

---

## 6. Measured throughput (A, `fp8_ds_mla`, util 0.89, warm, 128 tok/req)

| concurrency | aggregate tok/s | per-stream |
|---|---:|---:|
| 1 | 53.2 | 53.2 |
| 2 | 75.7 | 37.8 |
| 3 | 113.2 | 37.7 |
| 4 | 124.9 | 31.2 |
| **5** | **140.3** | 28.1 |
| 6 | 109.0 | 18.2 |

c6 aggregate falls *below* c3, so `max_num_seqs 6` is the saturation point. That
is acceptable for a 3-5 session target but marks the ceiling.

**This is below the old stage-c arena result (162.5 agg at c5).** The variables
differ (`max_num_seqs` 6 vs 12, k=5 vs k=3, abliterated vs base model, 1.65M vs
1.45M pool), so the gap is not yet attributed. Do not treat A as the throughput
optimum until the ablations in §7 are run.

---

## 7. Open optimization work

Flags currently enabled only because they are eugr's defaults for this image,
and which measured **harmful on the stage-c stack**:

| Flag | Stage-c result | Status here |
|---|---|---|
| `VLLM_USE_B12X_MHC` | clearly worse (arena raw 37.95 vs 44.75) | untested |
| `VLLM_USE_B12X_SPARSE_INDEXER` | -2.6% | untested |
| `VLLM_USE_B12X_FP8_GEMM` | crashed (`DeepGEMM/utils/layout.hpp:39: t.dim() == N`) | untested, does not crash here |

Also untested here: k=3 vs k=5, `max_num_seqs` 6 vs 8, and `--kv-cache-memory-bytes`
in place of `gpu_memory_utilization`.
