# Troubleshooting — symptom → cause → fix

Every row below was hit and fixed on real 2× GB10. Verbatim errors + deeper diagnosis in
[TEST_LOG.md](TEST_LOG.md) and [UPSTREAM_GAPS.md](UPSTREAM_GAPS.md).

## Startup / distributed

| Symptom | Cause | Fix |
|---|---|---|
| Both nodes hang silently at distributed init, **GPU idle (0%)**, no error | Stale NCCL/mp state — **orphaned `vllm`/`EngineCore` procs survive `docker compose down`** and hold dist port 25000 + the GPU | `pkill -9 -f 'vllm serve\|EngineCore\|multiproc_executor'` on **both** nodes before restart. Use [scripts/clean-restart.sh](scripts/clean-restart.sh). |
| Container boots, loads, then **restarts into a deadlock repeatedly** (weights re-load from 0%) | `restart: unless-stopped` + engine deaths that exit 0 + a capture-time collective wedge → docker auto-restarts into the hang | Set `restart: "no"` in the compose. Clean-restart manually. |
| Worker dies with `zmq.error.ZMQError: Cannot assign requested address (tcp://<ip>:...)` | Image bakes `VLLM_HOST_IP` = the *author's* address; on your node that IP isn't local | Set `-e VLLM_HOST_IP=<this node's fabric IP>` **per node** (head=10.0.1.1, worker=10.0.1.2). |
| Both nodes come up as **rank 1**, cluster hangs at init | Baked `CMD` carries a fixed `--node-rank`; launcher inherited the wrong identity | Pass `--node-rank`/`--headless` explicitly per node; don't trust baked values. |
| `ProcessGroupGloo ... gloo/transport/tcp/device.cc` at rank init | `GLOO_SOCKET_IFNAME`/`TP_SOCKET_IFNAME` baked to a NIC that doesn't exist on your host | Default both to `NCCL_SOCKET_IFNAME` (one value covers all three). |
| Startup **stalls ~11+ min** at "kv cache quantization", GPU 96%, `shm_broadcast` repeating | `gpu-memory-utilization` too high (0.85 → 2.77M-token NVFP4 pool is pathologically slow to quantize/capture) | Use **util 0.82** (2.14M pool, fast startup). See [TUNING.md](TUNING.md). |
| Another container (`llama-server`/gpustack) silently steals GB10 memory | Auto-restarted on boot, contends for shared unified memory | Stop it before serving: `docker stop $(docker ps -q --filter name=llama) gpustack-worker`. |

## NVFP4 / KV cache

| Symptom | Cause | Fix |
|---|---|---|
| `--kv-cache-dtype nvfp4_ds_mla: invalid choice` | Overlay-only image; `nvfp4_ds_mla` lives in the Stage-A/B/C chain | Build the full `dspark-nvfp4-stage-c` image, not just the overlay. |
| `assert kv_cache_dtype.startswith("fp8") ... got nvfp4_ds_mla` | eugr resolver blocks NVFP4 KV for DeepSeek-V4 | Use tonyd2wild's image (has the DeepSeek NVFP4-KV writer); eugr can't do it (UPSTREAM_GAPS #1). |
| `setStorage ... out of bounds` (512-vs-576) at profiling | eugr's stock 432-byte NVFP4 writer can't pad to the DSA sparse-indexer page | Not patchable client-side — needs the 584-byte padded DeepSeek writer (tonyd2wild). |
| NVFP4 **weight** model won't serve (swiglu-clamp / cutlass-eager / `block_tables`) | NVFP4 *weight* MoE path broken on sm_121 for DeepSeek-V4 | Use **FP8 weights** + NVFP4 **KV**. NVFP4 weights give no memory benefit anyway. |

## Output quality / client

| Symptom | Cause | Fix |
|---|---|---|
| `</think>` leaks into displayed content | Client reads reasoning from `reasoning_content`; this serve uses **`reasoning`** | Set the harness's reasoning key to `reasoning`. Kimi Code: `reasoning_key = "reasoning"`. See [CLIENT_INTEGRATION.md](CLIENT_INTEGRATION.md). |
| Empty content, tokens billed (`content=null`) | Client `stop` strings decapitate reasoning mid-`<think>`; `</think>` never arrives | tonyd2wild patch 0005 (baked into `dspark-nvfp4-stage-c`) scopes stop-strings to content. Verify the image has it. |
| Garble / CJK drift / prompt echo / repetition, **only on cold requests** | Missing Patch 3 (cold-prefill spec-placeholder bug), or greedy draft | Confirm Patch 3: `docker exec <c> grep -c is_prefill_chunk .../v1/core/sched/scheduler.py` → **5**. Ensure `draft_sample_method:"probabilistic"`, drop `--override-generation-config`. |
| Client silently capped at 100K context | Client model entry `max_context_size` too low | Set `max_context_size = 1000000` (Kimi Code and similar). |
| `num_speculative_tokens` rejected (7) or crashes (10) | DSpark `k` must be a multiple of `n_predict=5` | **Keep k=5.** |

## Thermal

| Symptom | Cause | Fix |
|---|---|---|
| Node overheats / powers off under sustained load | GB10 firmware cooling limits under 140W sustained | Cap clock: `sudo nvidia-smi -lgc 0,2200`. **Zero throughput loss** (bandwidth-bound). Do NOT rely on a firmware "fix" — some UEFI/EC updates *cause* fan-curve regressions (see NVIDIA forums). |
| GPU pinned ~611 MHz / ~13W / ~50°C under load | USB-C PD controller firmware wedge | Cold-drain reset of the power brick (community-confirmed). |
