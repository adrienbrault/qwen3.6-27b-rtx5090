import os
import sys

# Make TurboQuant's fixed decode KV-split count settable at runtime.
# Upstream freezes it at 32 (AttentionConfig.tq_max_kv_splits_for_cuda_graph) so the
# cudagraph grid stays constant. 32 is tuned to saturate the GPU for a SINGLE request;
# at batch>1 it launches batch x 32 splits plus the cross-split reduction, which costs
# batched decode throughput. Expose it via $VLLM_TQ_KV_SPLITS so we can sweep it.
p = "/usr/local/lib/python3.12/dist-packages/vllm/config/attention.py"
src = open(p).read()

if "VLLM_TQ_KV_SPLITS" in src:
    print("already patched")
    sys.exit(0)

old = "    tq_max_kv_splits_for_cuda_graph: int = 32"
new = (
    "    tq_max_kv_splits_for_cuda_graph: int = int(\n"
    '        os.environ.get("VLLM_TQ_KV_SPLITS", "32")\n'
    "    )"
)
assert old in src, "anchor not found"
src = src.replace(old, new, 1)

if "\nimport os\n" not in src:
    # dataclass module always has a future/import block at the top; prepend safely.
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            lines.insert(i, "import os")
            break
    src = "\n".join(lines)

open(p, "w").write(src)
print("PATCHED_TQ_KV_SPLITS")
