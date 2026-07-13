# Benchmarks

- **[RESULTS.md](RESULTS.md)** — all numbers.
- `template_probe.py <base_url> <label>` — behavioural probe: single tool call, **parallel** tool
  calls, chat→tool-result→chat continuation, plain chat turn after tools. This is how you evaluate
  a chat template; a speed benchmark can't.
- `prefix_probe.py <base_url> <label>` — prefix-cache hit rate. **Note:** vLLM's counter reports 0%
  on this hybrid model even when the cache works. Trust the TTFT timing, not the metric.

Throughput: [llama-benchy](https://github.com/eugr/llama-benchy).

```bash
llama-benchy --base-url http://localhost:8020/v1 --model qwen3.6-27b \
  --tokenizer /path/to/model --pp 512 4096 --tg 256 --concurrency 1 2 4 8 \
  --runs 2 --skip-coherence --format md
```

Agentic: [Harbor / Terminal-Bench](https://www.tbench.ai/).

```bash
harbor run -d terminal-bench/terminal-bench-2 -a terminus-2 -m openai/qwen3.6-27b -n 4 \
  --ae OPENAI_API_KEY=sk-local --ae OPENAI_BASE_URL=http://172.17.0.1:8020/v1 \
  --allow-agent-host 172.17.0.1
```
