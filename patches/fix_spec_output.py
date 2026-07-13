import sys

p = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py"
src = open(p).read()

if "spec-verify: honor the out-param contract" in src:
    print("already patched")
    sys.exit(0)

# PR #40914 returns the raw kernel tensor from forward()'s spec-verify branch.
# forward() is invoked as a mutated-out-param custom op (unified_attention_with_output):
# under FULL_AND_PIECEWISE cudagraphs the return value is DISCARDED and the caller
# reads `output`. Returning attn_out therefore leaves `output` stale/zeroed -> the
# model decodes a constant garbage token ("!!!!"). Write into `output` like every
# other branch of forward() does.
old = """                    max_num_kv_splits=self.max_num_kv_splits,
                )
                return attn_out
"""

new = """                    max_num_kv_splits=self.max_num_kv_splits,
                )
                # spec-verify: honor the out-param contract. forward() is called
                # as a mutated-out-param custom op; under FULL_AND_PIECEWISE
                # cudagraph capture the return value is discarded and the caller
                # reads `output`. Mirror the normal tail of forward().
                if output.ndim == 3:
                    output[:N] = attn_out.to(output.dtype)
                else:
                    output[:N] = attn_out.reshape(N, -1).to(output.dtype)
                return output
"""

assert old in src, "anchor not found (PR #40914 block missing or changed)"
open(p, "w").write(src.replace(old, new, 1))
print("PATCHED_SPEC_OUTPUT_CONTRACT")
