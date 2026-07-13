# Benchmark results

Hardware: RTX 5090 32 GB (`sm_120`), Ryzen 9 5900X, 62 GB RAM, Ubuntu 24.04, driver 595.71.05.
Model: `unsloth/Qwen3.6-27B-NVFP4` + MTP `ns=3` + vision, unless noted.
Tool: [llama-benchy](https://github.com/eugr/llama-benchy) — *not* ad-hoc curl loops, which are noisy enough to produce wrong conclusions.

## KV cache: TurboQuant 4-bit vs fp8

Identical image, flags, and tool; same day.

| | fp8_e4m3 | **turboquant_4bit_nc** |
|---|---|---|
| KV pool | 172,000 | **261,333** |
| KV density | 26.8K tok/GiB | **47.8K tok/GiB** |
| decode c1 | 129 | **143** |
| decode c2 | 253 | 251 |
| decode c4 | 492 | **552** |
| decode c8 | **868** | 540 |
| prefill @512 | 4,466 | 4,466 |
| prefill @4096 | 9,607 | **10,222** |
| MTP acceptance | ~60% | **75.8%** |

TurboQuant's Triton decode kernel saturates around 4 concurrent and plateaus; its 4-bit dequant
is ALU work that flash-attn's fp8 path gets free in hardware. Single-user → TurboQuant.
8+ concurrent → fp8.

## Quant shoot-out (all NVFP4, fresh nightly, same flags)

| | Unsloth | natfii (modelopt) | NVIDIA official |
|---|---|---|---|
| decode c1 / c8 | 131 / 894 | 126 / 881 | 93 / 757 |
| prefill @4K | 9,592 | **13,348** | 4,921 |
| max ctx @ util 0.95 | 144K | **200K** | OOM (→150K @0.92) |
| **Terminal-Bench 2.1** (8 tasks ×2) | **15/16, 8/8 pass@2** | 12/16, 7/8 | — |

natfii is faster; **Unsloth is smarter**, and quality won. NVIDIA's official quant loses on
every axis — its slow prefill path is the killer.

## Quality

| eval | config | result |
|---|---|---|
| **Aider polyglot** (225 exercises) | diff format, 4 threads | **72.3% pass@2**, 34.4% pass@1, **97.3% well-formed** |
| **Terminal-Bench 2.1** (8-task subset ×2) | Harbor + Terminus-2 | **7/8 pass@2** (12/16 trials; 2 of the 4 misses were agent *timeouts*, not wrong answers) |
| Terminal-Bench 2.0 (89 tasks) | Harbor + Terminus-2, temp 1.0 | *running — vs Qwen's published **59.3*** |

Coherence: needle-in-haystack at 10K recalled exactly; factual list clean; MTP per-position
acceptance 0.945 / 0.764 / 0.564 (healthy decay — flat 100% would mean degenerate lock-step).

### On comparability

The **aider polyglot leaderboard is frozen** (last data commit 2025-10-04) — no 2026 models on
it, so 72.3% isn't comparable to modern peers. It remains an excellent *quant-regression* test.
The nearest published Qwen reference is Qwen3-32B at 41.3% (diff, May 2025).

**Terminal-Bench 2.0 is the comparable one**: Qwen publishes **59.3** for Qwen3.6-27B with a fully
documented config (Harbor + Terminus-2, temp 1.0, top_p 0.95, top_k 20, 256K ctx, avg of 5 runs).
That is the number to beat — or to lose to, by however much 4-bit weights + 4-bit KV cost.

## Rejected (with numbers, so nobody redoes them)

| | result |
|---|---|
| `--async-scheduling` | c4 552 → 526. Rejected. |
| `--max-num-batched-tokens 4096` | prefill 9,607 → 2,556 (**−73%**) for +28K ctx. Rejected. |
| `VLLM_TQ_KV_SPLITS=8` | c1 143 → 132, c8 unchanged. Rejected (default 32 is right). |
| froggeric chat template | 4/4 vs bundled 4/4 on a behavioural tool-call probe. No gain. |
| DFlash | 3.3 GB draft model → 1,616-token context. Fatal on 32 GB. |
