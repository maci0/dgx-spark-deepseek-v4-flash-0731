# NVFP4 MLA KV cache: porting it onto an official vLLM image

Goal: `FROM vllm/vllm-openai:<release>` + `pip install b12x` + a source patch,
instead of depending on a third-party fork image.

Reference extract: [`nvfp4-ds-mla-anemll-reference.txt`](nvfp4-ds-mla-anemll-reference.txt)
(every `nvfp4` code site in the anemll image, with context).

---

## 1. Why a patch is needed at all

`nvfp4_ds_mla` is **not upstream**. Verified against `vllm/vllm-openai:nightly`
built 6 hours before writing (`0.26.1rc1.dev1046`):

| | upstream nightly | anemll `0.1.1` | stage-c |
|---|---|---|---|
| `nvfp4_ds_mla` sites | **0** | 17 | 6 |
| nvfp4+MLA guard | **present** | absent | absent |
| b12x wheel | absent | 0.15.3 | 1.2.3 |

Upstream ships the **guard but not the kernels**, so patching only the guard
achieves nothing. No upstream PR exists for `nvfp4_ds_mla` (searched
vllm-project/vllm PRs); both known implementations are out-of-tree.

## 2. It is pure Python

No `.so` contains the symbol in either implementation, so **no CUDA
compilation is required**. The format reuses the existing `fp8_ds_mla` machinery
with a different packing constant:

```python
# models/deepseek_v4/attention.py
if kv_cache_dtype in ("nvfp4", "nvfp4_ds_mla"):
    assert use_fp8_ds_mla_layout, (
        "DeepseekV4 nvfp4 KV cache requires the sparse MLA padded layout")
    cache_config.cache_dtype = "nvfp4_ds_mla"
    return "nvfp4_ds_mla", torch.uint8
...
alignment = 584 if self.kv_cache_dtype == "nvfp4_ds_mla" else ...
```

A **584-byte padded uint8 envelope**. Both independent implementations agree on
584, which is a useful cross-check. Note this repo's older docs cite a 656-byte
MLA page: that is the **fp8** figure, and the ~11% difference is part of why
NVFP4 reaches a larger pool.

## 3. Base the port on anemll, not stage-c

Both work; anemll is the better donor.

| Aspect | anemll (0.25.2) | stage-c (0.21.1rc1) |
|---|---|---|
| Abstraction | **`KVQuantMode` enum**, `is_nvfp4`, `get_kv_quant_mode()` | inline `head_bytes = 584` |
| Helper | **`nvfp4_kv_cache_full_dim()`** in `utils/torch_utils.py` | none |
| Alignment | **584 nvfp4 / 576 fp8_ds_mla / 512 else** | fixed 576 |
| Backends | `flashmla_sparse` **and** `sparse_swa` | `nvidia/flashmla.py` only |
| Extras | int4/int8/fp8 `per_token_head` modes; `MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP` | none |
| Distance to current upstream | 0.25.2 | 0.21.1rc1 (much further) |

**Take from stage-c:** not code, but a constraint to verify. It pins
`alignment=576` with the comment *"FlashMLA requires 576B alignment"*. anemll
uses 584 for NVFP4, so any FlashMLA-backed path must be re-checked rather than
assumed.

## 4. Patch surface (7 files)

| File | Change |
|---|---|
| `config/cache.py` | add `nvfp4_ds_mla` to the `CacheDType` literal |
| `config/vllm.py` | narrow the guard from `startswith("nvfp4")` to `== "nvfp4"` |
| `utils/torch_utils.py` | dtype→`torch.uint8` mapping, `nvfp4_kv_cache_full_dim()`, ModelOpt map |
| `v1/kv_cache_interface.py` | `KVQuantMode.NVFP4`, `get_kv_quant_mode()`, page-size accounting |
| `models/deepseek_v4/attention.py` | layout selection, 584-byte envelope |
| `models/deepseek_v4/sparse_mla.py` | sparse-MLA path |
| `v1/attention/backends/mla/{flashmla_sparse,sparse_swa}.py` | backend support |

## 5. Dockerfile shape

```dockerfile
FROM vllm/vllm-openai:<pinned release>
RUN pip install --no-cache-dir b12x==1.2.6      # public on PyPI
COPY nvfp4-ds-mla.patch /tmp/
RUN patch -p1 -d /usr/local/lib/python3.12/dist-packages < /tmp/nvfp4-ds-mla.patch
```

`b12x` is a normal public PyPI package (latest **1.2.6**; anemll ships 0.15.3,
eugr 1.2.3), described as *"Unapologetically SM120-only CuTe DSL kernels for
NVFP4 GEMM and MoE"*. Upstream vLLM already carries the integration code
(`fused_moe/experts/flashinfer_b12x_moe.py`), so only the wheel is missing.

## 6. Known porting risk

The donor is 0.25.2 and upstream is 0.26.1+. The DeepSeek-V4 tree still lives at
`vllm/models/deepseek_v4/` in both, so the layout matches, but the surrounding
APIs have drifted: a raw `diff -ruN` between the two images is ~3,900 lines,
almost all unrelated version churn. **Port the ~27 nvfp4 code sites by hand**
using the reference extract rather than applying that diff.
