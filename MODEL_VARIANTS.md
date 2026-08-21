# DeepSeek-V4-Flash-0731 checkpoint variants, what fits our setup

Survey of HF checkpoints (147 `*V4-Flash*` variants as of 2026-08-13), scored against **this** stack:
2× GB10, vLLM tonyd2wild/eugr images, FP8 weights + `nvfp4_ds_mla` KV + DSpark. "Fits" means the
proven path: `DeepseekV4ForCausalLM`, **FP8 (e4m3) weights**, 256 experts, DSpark/MTP head present , 
so you swap `DSPARK_MODEL` and nothing else. NVFP4-KV is independent of the weight checkpoint.

## Already tried (in TEST_LOG)

| checkpoint | quant | result |
|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash-0731` | FP8 e4m3 (official) | ✅ works (baseline, 326@c48 FP8-512K; 1M via NVFP4-KV) |
| `apetersson/...-Abliterated-FP8` | FP8 e4m3, abliterated | ✅ works, **speed-neutral vs official**: current live model |
| `neko-legends/...-Abliterated-NVFP4` | NVFP4 weights | ❌ 3 walls (swiglu-clamp / cutlass-eager / block_tables) |
| `sakamakismile/...-Abliterated-NVFP4` | NVFP4 weights (synthetic scales) | ❌ same walls |
| `RedHatAI/...-NVFP4-FP8` | NVFP4 MoE + FP8 attn (compressed-tensors) | ❌ incompatible with B12X native-FP8 kernels |
| `nvidia/DeepSeek-V4-Flash-NVFP4` | NVFP4 (modelopt) | ❌ same family mismatch (untested to boot, 168GB, no ctx benefit) |

Takeaway: **NVFP4/MXFP4/AWQ/INT4 *weights* don't work on the vLLM path** (and don't shrink the model , 
all ~156-168GB). Only **FP8 weights** get the full B12X/DSpark path. Abliteration is free (no speed cost).

## Worth trying, fits the setup, not yet tested

### Abliterated FP8 (drop-in; same size/perf, different alignment)
| checkpoint | experts | size | notes |
|---|---|---|---|
| **[`squanchyzx/…HERETIC-Abliterated-FP8`](https://huggingface.co/squanchyzx/DeepSeek-V4-Flash-0731-HERETIC-Abliterated-FP8)** | 256 | 167 GB | fp8 e4m3, DSpark in config. **Direct drop-in**: HERETIC abliteration (heretic = automated refusal-direction ablation). Try next if you want a different/stronger abliteration than apetersson. Expect identical speed. |
| `pocharlies/...-uncensored-abliterated-refusal-directions` |, |, | listed 2026-08-13; config not resolvable yet (no config.json), watch. |

### REAP-pruned FP8 (the real perf / capacity lever)
"REAP/REAM" = expert-pruned MoE (fewer routed experts). Same `DeepseekV4` arch + DSpark head, FP8 , 
**should load on our path**. Smaller weights → faster MoE + **frees memory for a much larger KV pool**
(more concurrent 1M sessions). Cost: pruning can degrade quality + may lower DSpark acceptance;
untested here. Ranked balanced→aggressive:

| checkpoint | experts | size | freed vs 167GB | risk |
|---|---|---|---|---|
| [`WaveCut/…REAM160-180B`](https://huggingface.co/WaveCut/DeepSeek-V4-Flash-0731-REAM160-180B) | 160 | **100.8 GB** | ~66 GB | moderate, best balance |
| [`WaveCut/…REAM128-146B-exp`](https://huggingface.co/WaveCut/DeepSeek-V4-Flash-0731-REAM128-146B-exp) | 128 | ~ | more | higher |
| [`WaveCut/…REAM96-111B-exp`](https://huggingface.co/WaveCut/DeepSeek-V4-Flash-0731-REAM96-111B-exp) | 96 | **64 GB** | ~103 GB | aggressive, biggest capacity/speed, most quality risk |

> **REAM160-180B** is the one to A/B first on the **2-node (TP2)** setup: ~66 GB freed roughly
> **doubles the KV pool again** on top of the util win, and fewer experts should speed decode, if
> quality + DSpark acceptance hold. Measure both tok/s **and** answer quality on real coding tasks.

**Single-Spark (1× GB10) option.** GB10 has ~122 GB usable; weights must fit alongside KV/activations:
- **REAM96-111B (64 GB) fits one node comfortably**: util 0.85 → ~40 GB KV → ~4M-token pool → full 1M
  ctx + 2-3 concurrent sessions on a single GB10. Big win: **no cross-node NCCL** (kills the entire
  restart-hang / distributed-init failure class) and **frees the second node**. Cost: ~half the decode
  compute (1 GPU) + aggressive-pruning quality risk (96/256 experts, verify).
- **REAM160-180B (100.8 GB) does NOT fit one node well**: ~3 GB left for KV → ~256K ctx, no headroom,
  high-util catch-22. Use TP2 for it.
- The full 167 GB model and REAM128-146B need TP2 (don't fit one node).

### Other FP8 with DSpark (not abliterated)
| checkpoint | notes |
|---|---|
| `Sn1waR/DeepSeek-V4-Flash-0731-CRACK-DSpark` | 256 experts, 167GB, DSpark, a "CRACK" repack; no clear advantage over official unless it fixes a DSpark quirk. |
| `patrickbdevaney/...-REAP-calibrated-draft-head` | FP8, REAP + re-calibrated **draft head**: relevant if pruning hurts DSpark acceptance. |

## Does NOT fit our vLLM path (skip)

- **GGUF** (dozens: Lucebox, Jared, rbinrs, BahamutRU, batiai, ...), llama.cpp/ROCm, not vLLM.
- **EXL3** (bullerwins 3.48–5.04bpw, wrldsuksgo2mars), ExLlamaV3.
- **MLX / MXFP8-MLX** (philipjohnbasile, cnrai), Apple Silicon.
- **AWQ** (True2456), **INT4 / W4A16 / WNA16** (Intel AutoRound, yiminyuan, hampsonw), weight-only
  quant on non-B12X paths; unproven here and no ctx benefit.
- **MXFP4** (Kanposer speedy-colibri, scouzi DwarfStar), MXFP4 *weights*; the SGLang mxfp4 runner
  wants them but SGLang cross-node is broken on 2×GB10 (UPSTREAM_GAPS #5); vLLM path is FP8+NVFP4-KV.
- **Vision** (umans-ai, webbrain, mumitrol), multimodal, different serving profile.

## Recommendation

1. Stay on **apetersson abliterated FP8** (proven, live) or swap to **squanchyzx HERETIC** if you want
   a different abliteration, zero-risk, speed-neutral.
2. For a genuine perf/capacity jump, **A/B `WaveCut REAM160-180B`** (pruned FP8): measure decode tok/s,
   DSpark acceptance, KV-pool size, and coding-answer quality vs the full model. If quality holds, it's
   a strictly better serve on this hardware (smaller, faster, more concurrent 1M sessions).
