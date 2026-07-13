# Third-party code and attribution

This repository is MIT-licensed (see [LICENSE](LICENSE)), **except** for the parts below, which
are derived from [vLLM](https://github.com/vllm-project/vllm) and remain under the
**Apache License 2.0**.

## `patches/vllm-only.diff`

**Verbatim redistribution** of the vLLM-source portion of
[vllm-project/vllm#40914](https://github.com/vllm-project/vllm/pull/40914) —
*"[Bugfix][Spec-Decode] TurboQuant K+1 spec-verify routing (fixes #40880)"* — authored by
**@Sandermage** and contributed to vLLM under Apache-2.0. Redistributed here unmodified, with
attribution, so this setup is reproducible while the PR remains unmerged.

- Upstream PR: https://github.com/vllm-project/vllm/pull/40914
- vLLM license: https://github.com/vllm-project/vllm/blob/main/LICENSE (Apache-2.0)

## `patches/fix_spec_output.py`, `patches/tq_auto_fallback.py`, `patches/tq_splits.py`

Original work (MIT), but they **modify Apache-2.0 vLLM source files** at build time and quote
short anchor snippets of that source in order to locate the edit sites. The resulting patched
files inside the Docker image are derivative works of vLLM and are governed by Apache-2.0.

Files modified in the image:
- `vllm/v1/attention/backends/turboquant_attn.py`
- `vllm/model_executor/layers/quantization/turboquant/config.py`
- `vllm/config/attention.py`

## Base image

`vllm/vllm-openai:nightly` — Apache-2.0. This repo ships no vLLM binaries; the Dockerfile pulls
the official image and patches it locally.

## Models and benchmarks (not redistributed here)

- [`unsloth/Qwen3.6-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) — Apache-2.0
  (Qwen3.6 base license).
- [Terminal-Bench / Harbor](https://www.tbench.ai/) — Apache-2.0.
- [aider](https://github.com/Aider-AI/aider) — Apache-2.0.
- [llama-benchy](https://github.com/eugr/llama-benchy).

## If you upstream this

The fix in `fix_spec_output.py` belongs in vLLM, not in a patch repo. It is offered to the vLLM
project and to the author of #40914 under Apache-2.0, on the same terms as any other contribution.
