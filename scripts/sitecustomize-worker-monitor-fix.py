"""Keep the headless worker-node parent alive until its workers actually exit.

Upstream bug. vllm/entrypoints/cli/serve.py::run_headless does:

    executor = MultiprocExecutor(vllm_config, monitor_workers=False)
    executor.start_worker_monitor(inline=True)
    return                      # <- process exits the moment the monitor returns

and start_worker_monitor(inline=True) blocks on
multiprocessing.connection.wait(sentinels), whose return is supposed to mean
"a worker died". On this 2-node setup it returns while the worker is still
initialising, so the parent returns, exits, and SIGKILLs a healthy worker. The
head then blocks forever on the next collective, and the server never serves.

Observed on the worker node as:
    [Worker_TP1] Warming up DeepSeek V4 sparse MLA attention ...
    [multiproc_executor] Parent process exited, terminating worker queues

The window scales with startup time, so NVFP4 KV pools (long quantization +
kernel warmup) hit it every time, while short fp8 boots usually get away with it.

Fix: for the inline case, poll real process liveness and only return once every
worker has genuinely exited. Everything else is left untouched.
"""

import importlib
import importlib.abc
import sys
import time

TARGET = "vllm.v1.executor.multiproc_executor"


def _patch(module):
    cls = getattr(module, "MultiprocExecutor", None)
    if cls is None:
        return
    orig = cls.start_worker_monitor
    if getattr(orig, "_liveness_patched", False):
        return

    def start_worker_monitor(self, inline=False):
        if not inline:
            return orig(self, inline=inline)
        while any(h.proc.is_alive() for h in self.workers):
            time.sleep(5)
        print("[sitecustomize] all workers exited; releasing parent", file=sys.stderr)

    start_worker_monitor._liveness_patched = True
    cls.start_worker_monitor = start_worker_monitor
    print("[sitecustomize] headless worker-monitor liveness patch active", file=sys.stderr)


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            _patch(importlib.import_module(TARGET))
        except Exception as exc:  # never break startup over this
            print("[sitecustomize] patch skipped: %r" % (exc,), file=sys.stderr)
        finally:
            sys.meta_path.insert(0, self)
        return None


sys.meta_path.insert(0, _PatchOnImport())
