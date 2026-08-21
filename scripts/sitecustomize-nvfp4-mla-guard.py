"""Allow `nvfp4_ds_mla` KV cache on MLA models.

Upstream guard, vllm/config/vllm.py::VllmConfig.validate_nvfp4_kv_cache_with_mla:

    if (self.cache_config.cache_dtype.startswith("nvfp4")
            and self.model_config.use_mla):
        raise ValueError("nvfp4 KV cache is not supported with MLA ...")

`startswith("nvfp4")` is too broad: it also matches `nvfp4_ds_mla`, the NVFP4
layout built specifically FOR DeepSeek MLA (512 NoPE + 16 scales + 128 RoPE =
656 B/token), which this image ships kernels for. The guard is meant for the
generic `nvfp4` layout, which genuinely is incompatible. Plain `nvfp4` + MLA
still raises after this patch.

Patching the compiled pydantic validator after import does not work: VllmConfig
is a pydantic dataclass whose core schema is built at class creation, and
rebuilding it re-collects the original function. So rewrite the one expression
in the module source *before* the class is created.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys

TARGET = "vllm.config.vllm"
OLD = 'self.cache_config.cache_dtype.startswith("nvfp4")'
NEW = 'self.cache_config.cache_dtype == "nvfp4"'


def _say(msg):
    print("[nvfp4-mla-patch] %s" % msg, file=sys.stderr, flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, origin):
        self._origin = origin

    def create_module(self, spec):
        return None  # default module creation

    def exec_module(self, module):
        with open(self._origin, "r") as fh:
            source = fh.read()
        if OLD in source:
            source = source.replace(OLD, NEW, 1)
            _say("narrowed nvfp4 MLA guard to exact 'nvfp4' in %s" % self._origin)
        else:
            _say("guard expression not found in %s; loading unmodified" % self._origin)
        code = compile(source, self._origin, "exec")
        exec(code, module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        # Find where it would normally come from, without recursing into self.
        sys.meta_path.remove(self)
        try:
            real = importlib.util.find_spec(fullname)
        except Exception as exc:
            _say("could not locate %s: %r" % (fullname, exc))
            real = None
        finally:
            sys.meta_path.insert(0, self)
        if real is None or not real.origin or not real.origin.endswith(".py"):
            return None
        return importlib.machinery.ModuleSpec(
            fullname, _Loader(real.origin), origin=real.origin
        )


sys.meta_path.insert(0, _Finder())
