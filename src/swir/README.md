# SwiR Switch-Thinking — Engine Primitive

> **Plan:** [275](../../.plans/275_swir_switch_thinking.md) · **Research:** [241](../../.research/241_SwiReasoning_Explicit_Latent_Switch.md) · **Paper:** [SwiReasoning (ICLR 2026)](https://arxiv.org/abs/2510.05069) · **Feature:** `swir_switch_thinking`

A modelless, MIT-licensed Rust port of SwiReasoning's explicit↔latent reasoning
mode controller. Three training-free primitives that switch a transformer
decoder between token-space (`Explicit`) and continuous-embedding-space
(`Latent`) reasoning at inference time, driven by block-relative entropy trends.

## Module structure

| File | Purpose |
|------|---------|
| `types.rs` | `SwiRConfig`, `ThinkMode`, `StepAction`, `ControlToken`, `SwiRStats` |
| `controller.rs` | `SwiRController` — the 2-mode state machine (paper Algorithm 1) |
| `soft_embedding.rs` | `soft_embedding()` — SIMD `ẽ_t = Σ_v p_t[v]·e(v)` for Latent mode |
| `signal_mix.rs` | `mix_thinking_signal()` — control-token blending at switch instants (Eq. 4) |
| `convex_hull_check.rs` | G4 invariant: soft embeddings lie in vocab convex hull |
| `entropy.rs` | `entropy_from_logits()` / `shannon_entropy()` — vendored max-shift stable kernel |
| `strategy_adapter.rs` | `SwiRStrategyAdapter` — `impl ThinkingStrategy for SwiRController` |
| `bench.rs` | Benchmark harness — traits for real-model swap-in + synthetic reference |

## Target model (Plan 275 T3.1)

**Qwen3-1.7B** is the recommended validation target for this primitive.

### Why Qwen3-1.7B

1. **`<think>` token native.** Qwen3 ships with the `<think>`/`</think>` control
   tokens that `ControlToken::CloseThink` maps to — no prompt-engineering hack
   needed to inject a synthetic thinking boundary.
2. **Smallest in the Qwen3 family.** SwiR is inference-time, so the validation
   cost is per-token decode; a 1.7B model fits the paper's Qwen3-8B
   architecture family at ~5× lower compute per gate run.
3. **Paper defaults are Qwen3-tuned.** Paper Tab. 6 reports best-practice
   hyperparameters (`w_e_to_l=512`, `c_max=20`, `α_0=0.6`, `β_0=0.7`) on
   Qwen3-8B. Qwen3-1.7B shares the tokenizer and the thinking-token protocol,
   so the defaults transfer with minimal tuning (paper §5.2 confirms the
   family shares the same hyperparameter plateau).
4. **Locally available.** `riir-train/data/` holds `gemma-2-2b-it-f16.gguf` and
   `MiniCPM5-1B-F16.gguf`; the Qwen3-1.7B GGUF is the natural sibling for
   SwiR validation. katgpt-rs cannot load any of them (no model loader — see
   below), but riir-ai Plan 299 can.

### Fallbacks (if Qwen3-1.7B unavailable)

- **Qwen3-4B** — same family, larger but still small; paper's mid-scale data
  point. Same `<think>` token, same defaults.
- **Gemma-2-2B-it** (available locally as `gemma-2-2b-it-f16.gguf`) — no
  native `<think>` token, requires prompt-engineering a synthetic
  `<think>...</think>` wrapper. Use only if no Qwen3 variant is available;
  document the wrapper in the riir-ai benchmark harness.

### Why not the paper's Qwen3-8B for the first gate

The paper's headline numbers (+1.8–3.1pp accuracy, 1.36–6.8× efficiency) are
on Qwen3-8B. Reproducing on 1.7B first is the standard "smallest viable scale"
discipline — if SwiR can't beat `thinking_cot` baseline on 1.7B at all, the
algorithm has a transferability issue worth catching before burning 8B-scale
compute. Once 1.7B validates, scale to 8B for the final GOAT proof.

## The modelless constraint — why the gate is split

katgpt-rs is an **engine-primitives library** (the "engine" half of the
engine/fuel split). It has no model loader, no tokenizer, no KV cache, no
inference loop — by design. grep for `gguf|candle|burn|tch|llm|model_loader`
in `Cargo.toml` returns zero matches.

Therefore the GOAT gate is split:

| Gate | Scope | Where it runs |
|------|-------|---------------|
| G3 step perf < 200ns | Algorithmic | **katgpt-rs** (this repo) — `bench_275_swir_goat.rs::g3_*` ✅ 3.1ns |
| G4 convex hull | Algorithmic | **katgpt-rs** — `g4_*` ✅ 1000/1000 |
| G5 feature isolation | Algorithmic | **katgpt-rs** — `g5_*` ✅ clean both ways |
| G6 kurtosis auto-fallback | Algorithmic | **katgpt-rs** — `g6_*` ✅ forces Explicit |
| G7 zero-alloc step() | Algorithmic | **katgpt-rs** — `g7_*` ✅ 0 allocs/1023 steps |
| G8 α_t/β_t schedule | Algorithmic | **katgpt-rs** — `g8_*` ✅ monotonic |
| G9 hyperparameter sweeps | Algorithmic | **katgpt-rs** — `g9a/g9b/g9c` ✅ |
| G1 accuracy ≥ +1.5pp on MATH500 | Empirical | **riir-ai Plan 299** — needs Qwen3-1.7B + MATH500 |
| G2 token efficiency ≥ 1.3× | Empirical | **riir-ai Plan 299** — needs decode loop |
| T3.9 accuracy ablations | Empirical | **riir-ai Plan 299** — needs accuracy signal |

The katgpt-rs half is **complete** (8/8 synthetic gates pass, plus the G9
ablation sweeps). The riir-ai half is the real-model proof.

## Public API (frozen)

```rust
use katgpt_rs::swir::{SwiRConfig, SwiRController, StepAction, soft_embedding};

let mut ctrl = SwiRController::new(SwiRConfig::default());
match ctrl.step(entropy, step_index) {
    StepAction::EmitToken(_id) => { /* sample concrete token */ }
    StepAction::EmitSoftEmbedding => { /* compute ẽ_t into scratch */ }
    StepAction::InjectControlToken(token) => { /* resolve + feed */ }
    StepAction::Terminate => { /* stop */ }
}
```

Hosts that already plug into `thinking_cot` (Plan 194) should prefer
`SwiRStrategyAdapter` over driving the controller directly — see
`tests/swir_strategy_integration.rs`.

## References

- **Paper:** [SwiReasoning: Switching between Explicit and Latent Reasoning](https://arxiv.org/abs/2510.05069) — Shi et al., ICLR 2026
- **Plan:** [`katgpt-rs/.plans/275_swir_switch_thinking.md`](../../.plans/275_swir_switch_thinking.md)
- **Research:** [`katgpt-rs/.research/241_SwiReasoning_Explicit_Latent_Switch.md`](../../.research/241_SwiReasoning_Explicit_Latent_Switch.md)
- **GOAT results:** [`katgpt-rs/.benchmarks/275_swir_switch_thinking_goat.md`](../../.benchmarks/275_swir_switch_thinking_goat.md)
- **Precedent:** Plan 271 (`attn_match`) — same synthetic-only GOAT pattern, same engine/fuel split
