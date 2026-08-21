# Troubleshooting, symptom → cause → fix

Every row below was hit and fixed on real 2× GB10. Verbatim errors + deeper diagnosis in
[TEST_LOG.md](TEST_LOG.md) and [UPSTREAM_GAPS.md](UPSTREAM_GAPS.md).

## Startup / distributed

| Symptom | Cause | Fix |
|---|---|---|
| Both nodes hang silently at distributed init, **GPU idle (0%)**, no error | Stale NCCL/mp state, **orphaned `vllm`/`EngineCore` procs survive `docker compose down`** and hold dist port 25000 + the GPU | `pkill -9 -f 'vllm serve\|EngineCore\|multiproc_executor'` on **both** nodes before restart. Use [scripts/clean-restart.sh](scripts/clean-restart.sh). |
| Container boots, loads, then **restarts into a deadlock repeatedly** (weights re-load from 0%) | `restart: unless-stopped` + engine deaths that exit 0 + a capture-time collective wedge → docker auto-restarts into the hang | Set `restart: "no"` in the compose. Clean-restart manually. |
| Worker dies with `zmq.error.ZMQError: Cannot assign requested address (tcp://<ip>:...)` | Image bakes `VLLM_HOST_IP` = the *author's* address; on your node that IP isn't local | Set `-e VLLM_HOST_IP=<this node's fabric IP>` **per node** (head=10.0.1.1, worker=10.0.1.2). |
| Both nodes come up as **rank 1**, cluster hangs at init | Baked `CMD` carries a fixed `--node-rank`; launcher inherited the wrong identity | Pass `--node-rank`/`--headless` explicitly per node; don't trust baked values. |
| `ProcessGroupGloo ... gloo/transport/tcp/device.cc` at rank init | `GLOO_SOCKET_IFNAME`/`TP_SOCKET_IFNAME` baked to a NIC that doesn't exist on your host | Default both to `NCCL_SOCKET_IFNAME` (one value covers all three). |
| Startup **stalls ~11+ min** at "kv cache quantization", GPU 96%, `shm_broadcast` repeating | `gpu-memory-utilization` too high (0.85 → 2.77M-token NVFP4 pool is pathologically slow to quantize/capture) | Use **util 0.82** (2.14M pool allocated, faster startup, though NVFP4 pools still fail later in warmup). See [TUNING.md](TUNING.md). |
| Another container (`llama-server`/gpustack) silently steals GB10 memory | Auto-restarted on boot, contends for shared unified memory | Stop it before serving: `docker stop $(docker ps -q --filter name=llama) gpustack-worker`. |

### systemd `RemoveIPC` silently kills the worker rank (2026-08-20)

The single highest-impact bug found so far. `logind.conf` ships `RemoveIPC=yes`, and the worker node is
started over SSH, when that login session closes, systemd deletes **every POSIX semaphore owned by the
user**, including the ones vLLM's `MultiprocExecutor` just created:

```
File ".../multiprocessing/synchronize.py", line 115, in __setstate__
    self._semlock = _multiprocessing.SemLock._rebuild(*state)
FileNotFoundError: [Errno 2] No such file or directory
```

The head node does **not** error. It blocks forever on the next collective. Presentations seen:
`shm_broadcast ... no block found in 60 seconds` repeating, GPU pinned at 96% with zero progress,
CUDA-graph capture frozen mid-percentage, and `DistStoreError: Timed out ... 1/2 clients joined`.

**Diagnostic tell:** the worker container is up but has **no `Worker_TP` process at all**
(`docker exec <cid> ps -eo comm`). "Worker idle" actually means "worker dead".

**Fix**: persists across reboots, needs no root:

```bash
loginctl enable-linger $USER            # on BOTH nodes
loginctl show-user $USER | grep Linger  # must print Linger=yes
```

Several failures previously attributed to b12x kernel deadlocks were really this.

## NVFP4 / KV cache

| Symptom | Cause | Fix |
|---|---|---|
| `--kv-cache-dtype nvfp4_ds_mla: invalid choice` | Overlay-only image; `nvfp4_ds_mla` lives in the Stage-A/B/C chain | Build the full `dspark-nvfp4-stage-c` image, not just the overlay. |
| `assert kv_cache_dtype.startswith("fp8") ... got nvfp4_ds_mla` | eugr resolver blocks NVFP4 KV for DeepSeek-V4 | Use tonyd2wild's image (has the DeepSeek NVFP4-KV writer); eugr can't do it (UPSTREAM_GAPS #1). |
| `setStorage ... out of bounds` (512-vs-576) at profiling | eugr's stock 432-byte NVFP4 writer can't pad to the DSA sparse-indexer page | Not patchable client-side, needs the 584-byte padded DeepSeek writer (tonyd2wild). |
| NVFP4 **weight** model won't serve (swiglu-clamp / cutlass-eager / `block_tables`) | NVFP4 *weight* MoE path broken on sm_121 for DeepSeek-V4 | Use **FP8 weights** + NVFP4 **KV**. NVFP4 weights give no memory benefit anyway. |
| LMCache: `LMCacheConnectorV1 incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | LMCache's VMM allocator would remap registered KV pages | Set `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9` (drop expandable_segments) or enable the cumem allocator. |
| LMCache: `sparse_mla_sm120_decode_dsv4: num_tokens>64 (36 vs 64)` at startup | `--kv-transfer-config` disables the hybrid KV manager (HMA); the DeepSeek-V4 sparse-MLA + DSpark verify (36 = seqs×(k+1)) then mis-routes | **No fix** without `SupportsHMA` on the connector. LMCache/FlexKV/Offloading all lack it → disk KV-offload is blocked with DSpark+sparse-MLA. See UPSTREAM_GAPS #7. |

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
| Node overheats / powers off under sustained load | GB10 firmware cooling limits under 140W sustained | Cap clock: `sudo nvidia-smi -lgc 0,2200`. **Zero throughput loss** (bandwidth-bound). Do NOT rely on a firmware "fix", some UEFI/EC updates *cause* fan-curve regressions (see NVIDIA forums). |
| GPU pinned ~611 MHz / ~13W / ~50°C under load | USB-C PD controller firmware wedge | Cold-drain reset of the power brick (community-confirmed). |

## Tooling / measurement traps (2026-08-20)

These cost hours and produced several wrong conclusions before being identified.

| Symptom | Cause | Fix |
|---|---|---|
| Server "never healthy" but is actually serving | eugr/SGLang recipes serve on **8000**, the stage-c record recipe on **8888** | Health-check the port the recipe declares, not the one you used last |
| `Starting vLLM server` never appears in the launcher log | That banner goes to the container's own log, not sparkrun's | `docker exec <cid> tail /tmp/sparkrun_serve.log` |
| Engine looks deadlocked: `EngineCore` in `run_busy_loop`→`queue.get()`, workers in `shm_broadcast.dequeue` | That is the **normal ready-and-waiting state** | Confirm with a real request before declaring a hang |
| Teardown/launch commands silently do nothing | `pkill -f sparkrun_serve` matches **your own SSH command line** and kills the shell mid-script | Use `pkill -f "[s]parkrun_serve"` |
| Detached launch dies as soon as the SSH call returns | `nohup ... &` inside a compound remote command | `setsid nohup … </dev/null &` as its own invocation |
| HF download fails `PermissionError: .../hub/.locks/...` | Root-owned cache dirs left by an earlier docker-compose-as-root run | Parent `hub/` is user-owned, so **no sudo needed**: `mv hub/.locks hub/.locks.root && mkdir hub/.locks` |
| sparkrun re-downloads the full 156 GB checkpoint every launch | sparkrun distributes models itself into the standard `hub/` cache and ignores `HF_HUB_CACHE` | Stage checkpoints in `~/.cache/huggingface/hub/`, not a custom root |
| `py-spy` fails with `Permission denied` inside the container | sparkrun rootless mode hardcodes `cap_add = []`; a recipe-level `cap_add:` does **not** override it. Host also has `kernel.yama.ptrace_scope=1` | Patch `sparkrun/orchestration/executors/docker.py`: `adjustments["cap_add"] = ["SYS_PTRACE"]` (every copy under `~/.cache/uv/archive-v0/*/`), then `docker exec -u 0 <cid> py-spy dump --pid <pid>` |

### Benchmark validity

| Trap | Effect | Guard |
|---|---|---|
| Cold vs warm cache | Up to **2.5×** swing, the same config measured `c1 = 23.7` then `59.4` tok/s back to back | Always warm first, then take ≥2 measurements |
| Run-to-run variance | ~4% (same config gridded 56.95 and 54.60 on different days) | Treat deltas under ~5% as noise |
| `ignore_eos` off | Early EOS shortens requests and inflates per-request overhead | Set `"ignore_eos": true` so every request emits exactly `max_tokens` |
| `--enable-prefix-caching` + repeated prompts | First cell of a grid row pays cold prefill, later cells reuse the prefix, makes c1-at-depth look catastrophic vs c2/c5/c10 | Compare only like-for-like cache states |

## KV offload to SSD (2026-08-21)

Disk-spill **is** reachable on the stage-c image without patches, but only via a connector that
implements `SupportsHMA`. DeepSeek-V4's sparse-MLA sm120 decode requires the hybrid KV manager, and
`--kv-transfer-config` disables HMA for any connector that doesn't declare support.

| Symptom | Cause | Fix |
|---|---|---|
| `sparse_mla_sm120_decode_dsv4: Check failed num_tokens>64 (36 vs 64)` at startup | Connector lacks `SupportsHMA` → vLLM turned off the hybrid KV manager. `LMCacheConnectorV1` is in this category | Use **`OffloadingConnector`** (or `SimpleCPUOffloadConnector`), both declare `SupportsHMA` |
| `ValidationError ... KV connector OffloadingConnector is incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unless enable_cumem_alloc` | The connector registers KV buffers; an expandable-segments VMM remap would invalidate them | Set `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9` (drop `expandable_segments`), or enable the cumem allocator |

Disk tier is stock: `TieringOffloadingSpec` + secondary tier `"fs_python"`
(`vllm/v1/kv_offload/tiering/fs/manager.py`, *"pure-Python disk-backed secondary tier"*), parameters
`root_dir`, `n_read_threads`, `n_write_threads`. Working config in
[examples/prod-c5-ssd.yaml](examples/prod-c5-ssd.yaml).


---

## Multi-node debugging: where the worker's log actually is

`docker logs <worker container>` is **empty by design**. sparkrun runs the worker
container as `sleep infinity` and `docker exec`s the serve process into it,
redirecting to a file *inside* the container:

```bash
C=$(ssh worker 'docker ps --format "{{.Names}}" | grep sparkrun')
ssh worker "docker exec $C tail -50 /tmp/sparkrun_serve.log"
ssh worker "docker exec $C cat /tmp/sparkrun_serve.sh"     # exact serve command
ssh worker "docker exec $C cat /tmp/sparkrun_serve.pid"
```

Diagnosing a multi-node boot from the head log alone will mislead: the head only
ever reports the *symptoms* (`shm_broadcast` timeouts, NCCL `RETRY_EXC_ERR`).

## `Parent process exited` on the worker is not what it looks like

The child's `Parent process exited, terminating worker queues` sits several
layers below the cause. `vllm/entrypoints/cli/serve.py` installs its own handlers:

```python
def signal_handler(signum, frame):
    if not shutdown_requested:
        shutdown_requested = True
        raise SystemExit
signal.signal(signal.SIGTERM, signal_handler)
```

`SystemExit` prints **no traceback and exits 0**, so an externally killed worker
is indistinguishable from a clean success in every log. To tell them apart, chain
the handler rather than replacing it (vLLM's registration overwrites any handler
installed earlier by `sitecustomize`).

Ruled out as causes during one such investigation, each with evidence: systemd
`RemoveIPC` (linger enabled, still reproduced), SSH teardown (reproduced under
`screen`), `run_headless` returning early (probe showed the monitor was never
reached), and kernel OOM (`oom_kill 0` in the container cgroup, `OOMKilled=false`).
Note that `dmesg` is restricted on these nodes and returns **0 lines**, so it is
not evidence of absence; use the cgroup counter:

```bash
docker inspect $C --format '{{.State.OOMKilled}}'
cat /sys/fs/cgroup/system.slice/docker-$(docker inspect $C --format '{{.Id}}').scope/memory.events
```

## `To serve at least one request with the model's max seq len`

vLLM requires the KV pool to hold at least one `max_model_len` request. The error
gives the exact arithmetic, which is the cheapest way to measure bytes/token:

```
(11.04 GiB KV cache is needed, ... available KV cache memory (10.85 GiB)
```

11.04 GiB / 1,048,576 = ~11.0 KB/token for `fp8_ds_mla`. Raise
`gpu_memory_utilization`, lower `max_model_len`, or pin the pool with
`--kv-cache-memory-bytes`.
