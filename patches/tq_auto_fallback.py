import sys

p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/turboquant/config.py"
src = open(p).read()

if "VLLM_TQ_PRESET" in src:
    print("already patched")
    sys.exit(0)

old = "        if cache_dtype not in TQ_PRESETS:"
new = """        # Draft/MTP model runners do not inherit the main cache_config's kv
        # dtype (upstream gap: the drafter's cache_config.cache_dtype stays
        # "auto"), so TQ layers on the draft path arrive here with "auto".
        # Every TQ layer in a run shares one preset, so fall back to it.
        if cache_dtype in ("auto", None, ""):
            import os

            _fallback = os.environ.get("VLLM_TQ_PRESET")
            if _fallback in TQ_PRESETS:
                cache_dtype = _fallback
        if cache_dtype not in TQ_PRESETS:"""

assert old in src, "anchor not found"
open(p, "w").write(src.replace(old, new, 1))
print("PATCHED_AUTO_FALLBACK")
