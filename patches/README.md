# Patches

Applied on top of `vllm/vllm-openai:nightly`. All pure-Python — no CUDA recompile, ~1 min build.

```bash
docker build -t vllm-turboquant:patched .
```

| file | purpose |
|---|---|
| `vllm-only.diff` | Upstream [PR #40914](https://github.com/vllm-project/vllm/pull/40914) (open) — TurboQuant K+1 spec-verify routing. |
| `fix_spec_output.py` | **Makes #40914 actually work on Blackwell.** It returned the kernel tensor instead of writing the out-param buffer; under full CUDA-graph capture the return value is discarded → constant-token garbage. |
| `tq_auto_fallback.py` | MTP draft runner never inherits `cache_config.cache_dtype` (arrives `"auto"`, crashes). Falls back to `$VLLM_TQ_PRESET`. |
| `tq_splits.py` | Exposes TurboQuant's fixed decode KV-split count as `$VLLM_TQ_KV_SPLITS`. Default 32 is optimal — this exists so you can verify that, not change it. |

**After any vLLM nightly bump**: re-verify. These patch a file (`vllm/v1/attention/backends/turboquant_attn.py`)
that upstream actively moves. A failed `patch` fails the build loudly; a *shifted* apply could
silently produce garbage. Test with raw `/v1/completions` — the chat parser hides degeneration
as empty content.
