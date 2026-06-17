# Plan 275 — SwiR Switch-Thinking GOAT Gate (Synthetic Data)

> **UPDATE 2026-06-18:** The "deferred to riir-ai Plan 299" references throughout this document should read **riir-ai Plan 313** (Plan 299 is a different plan — NPC CGSP runtime). Plan 313 ran the real-model validation on **Gemma 2 2B IT + MATH-500** (CPU M1 Pro): **G1 = 0%** (model capability blocked), **G2 = 1.12× avg** (below 1.3× gate). Verdict: keep opt-in. Definitive G1/G2 gate pass requires Qwen3-4B/8B (the paper's model). See [`riir-ai/.benchmarks/313_swir_real_model_goat.md`](../../riir-ai/.benchmarks/313_swir_real_model_goat.md) for the real-model results. The synthetic-data results below remain valid as algorithmic-correctness proof.

**Date:** 2026-06-15
**Plan:** [`katgpt-rs/.plans/275_swir_switch_thinking.md`](../.plans/275_swir_switch_thinking.md)
**Research:** [`katgpt-rs/.research/241_SwiReasoning_Explicit_Latent_Switch.md`](../.research/241_SwiReasoning_Explicit_Latent_Switch.md)
**Source paper:** [arXiv:2510.05069](https://arxiv.org/abs/2510.05069) — SwiReasoning (ICLR 2026, Shi et al.)
**Test file:** `tests/bench_275_swir_goat.rs` (10 tests)
**Hardware:** Apple Silicon arm64 (NEON SIMD), Rust 1.93.0 release build

## TL;DR

8/8→**9/9** synthetic-data GOAT gates **PASS**. The SwiR controller is algorithmically
correct, plasma-tier fast (3.1 ns/step vs 200 ns budget = 64× margin),
zero-allocation in `step()`, and the convex-hull invariant holds on 1000 random
samples. The paper's headline gates (G1 accuracy on MATH500, G2 token
efficiency at fixed accuracy) are **deferred to riir-ai Plan 299** — katgpt-rs
is a modelless primitives library with no model loader (engine/fuel split).

**Decision: keep `swir_switch_thinking` OPT-IN** until riir-ai Plan 299 proves
G1/G2 on a real model. The algorithmic invariants proven here are necessary
preconditions for real-model validation to be meaningful.

## Scope

`katgpt-rs` is a modelless primitives library — it has no model loader, no
tokenizer, no KV cache. The paper's accuracy/efficiency claims are empirical
properties of the *combination* (SwiR controller + real LLM); they cannot be
measured in katgpt-rs alone. This matches the precedent set by Plan 271
(Attention Matching), whose GOAT gate also ran on synthetic data with
real-model validation deferred to riir-ai.

What this gate proves: the **algorithmic invariants** that must hold before
real-model validation is even meaningful — the controller is correct by
construction, the soft-embedding math is numerically sound, and the hot path is
fast enough to sit on the decode loop.

## Gate Results

| Gate | Criterion | Measurement | Status |
|------|-----------|-------------|--------|
| **G3** | `step()` ≤ 200 ns/call (release) | **3.1 ns/step** (64× margin) | ✅ PASS |
| **G4** | 1000 random probs in vocab convex hull | **1000/1000** (100%) | ✅ PASS |
| **G5** | feature-gate isolation (`cargo check` clean) | **clean** both ways | ✅ PASS |
| **G6** | kurtosis escape forces Explicit | **forces Explicit** end-to-end | ✅ PASS |
| **G7** | `step()` zero-allocation | **0 allocs / 0 bytes** over 1023 steps | ✅ PASS |
| **G1c** | controller correctness (switches, convergence, termination) | 6 switches, 3 CloseThink, 1 ForceAnswerPrefix, terminated at step 21 | ✅ PASS |
| **G2p** | SwiR terminates < fixed-budget baseline | **33 steps vs 1024** = 31.03× fewer | ✅ PASS |
| **G8** | α_t / β_t monotonic in step_index | **[0.703, 0.719, 0.738, 0.775, 0.85, 0.925, 1.0]** | ✅ PASS |
| **G9** | hyperparameter ablation (W_E→L/C_max/α_0 behavioral response) | W_E→L: 256→1 switches (monotone ✓); C_max: term 27→117 (monotone ✓); α_0: identical 13 switches across 0.3–1.0 (α-independent ✓) | ✅ PASS |
| **G1** | accuracy on MATH500 (+1.5 pp target) | — | ⏸ DEFERRED (riir-ai Plan 299) |
| **G2** | token efficiency at fixed accuracy (1.3× target) | — | ⏸ DEFERRED (riir-ai Plan 299) |
| **T3.9** | accuracy ablations (W_E→L, α_0, C_max, signal mix) | G9 above is the modelless behavioral proxy | ⏸ DEFERRED (riir-ai Plan 299) — G9 modelless proxy ✅ PASS |

## G6 Auto-Fallback (Plan 275 T3.8) — New Primitive

The paper's G6 requires auto-fallback on rigid-constraint tasks via
`selectivity_router`'s kurtosis signal. `selectivity_router` is an empty Cargo
feature in katgpt-rs (no module), so per the plan's T3.8 fallback clause we
added a manual escape hatch directly on `SwiRController`:

```rust
// SwiRConfig (new field):
pub kurtosis_escape_threshold: f32,  // default: f32::INFINITY (disabled)

// SwiRController (new method):
pub fn observe_kurtosis(&mut self, kurtosis: f32);
```

**Behavior:** when `last_kurtosis > kurtosis_escape_threshold`, the controller
refuses to enter or stay in Latent mode — it forces Explicit (token-space)
decoding. This bypasses soft-embedding exploration on rigid-constraint tasks
where continuous mixtures would hallucinate. NaN-safe: an un-observed kurtosis
(NaN) never fires the escape (NaN > threshold is false), preserving backward
compatibility for hosts that don't wire a kurtosis signal.

**5 unit tests** in `src/swir/controller.rs` cover: forces-Explicit-from-Latent,
blocks-Explicit-to-Latent-reentry, below-threshold-is-inert, NaN-is-inert,
releases-when-kurtosis-drops. The GOAT gate `g6_kurtosis_escape_hatch_end_to_end`
re-runs the path through the full adapter (StepContext → on_step → StepDirective).

## G7 Allocation Profile (Honest)

| Path | Allocs/call | Notes |
|------|-------------|-------|
| `SwiRController::step()` | **0** | Zero-alloc by construction (fixed-size ring buffer, no Vec ops). |
| `SwiRStrategyAdapter::on_step()` (soft path) | **2** | `soft_scratch.clone()` (embedding_dim × 4 bytes) + amortised softmax scratch resize. Unavoidable: the `EmitSoftEmbedding` directive owns its payload because the borrow checker can't tie the strategy's scratch lifetime to the call. |
| `SwiRStrategyAdapter::on_step()` (inject path) | **1** | `vec![id]` for `InjectTokens`. Fires only on convergence/termination steps. |

The adapter allocations are documented in `src/swir/strategy_adapter.rs` and
measured in `g7_adapter_on_step_allocations_debug`. They are **by design** —
the `ThinkingStrategy` trait contract acknowledges this (see
`src/thinking_cot/strategy.rs` lines 125-132).

## Key Honest Finding: Convergence Guard Livelock — FIXED

During G2p development, a real controller bug was discovered: the switch-count
guards (CloseThink at ½c_max, ForceAnswerPrefix above c_max) were enqueued on
**every Explicit step** while `switch_count` was in the convergence window. The
inject-queue drain (step 1 of `step()`) preempted the mode-switch logic
(step 3), so the controller could never switch back to Latent → never did
another Latent→Explicit switch → `switch_count` froze → termination (`> c_max`)
never fired.

**Fix applied** (`src/swir/controller.rs`): the switch-count guards now fire
only when `switched_to == Some(ThinkMode::Explicit)` — i.e., on the step where
the Latent→Explicit switch *just happened*. This is a one-shot trigger per
switch event, matching the paper's intent (§3.4 describes switch-count
thresholds, not continuous conditions). With the fix:

- G2p runs with the REAL `c_convergence_fraction = 0.5` (not a workaround).
- The full convergence→termination path is exercised correctly.
- 33 steps vs 1024 = 31.03× fewer steps (97% reduction).

The earlier version of this benchmark used a `c_convergence_fraction = 10.0`
workaround to skip the convergence branch entirely. That workaround is no
longer needed and has been removed from G2p.

**Note for riir-ai Plan 299:** the fix is algorithmically correct (each guard
fires exactly once per switch event), but real-model validation should still
verify that the `</think>` injection + answer-prefix budget produces the
expected accuracy/efficiency on MATH500.

## Reproduction

```bash
# Release build for perf gates (G3, G4, G6, G1c, G2p, G8).
cargo test --release --test bench_275_swir_goat \
    --features swir_switch_thinking -- --nocapture --test-threads=1

# Debug build for the allocation audit (G7).
cargo test --test bench_275_swir_goat \
    --features swir_switch_thinking -- --nocapture --test-threads=1

# Feature-gate isolation (G5) — two separate invocations.
cargo check                                  # default, no swir
cargo check --features swir_switch_thinking  # with swir
```

`--test-threads=1` is **required** for G7: the library's `TrackingAllocator`
is process-global, so parallel tests bleed allocations into the counter (same
caveat as Plan 271 G7).

## Validation Summary

- ✅ `cargo test --release --features swir_switch_thinking --test bench_275_swir_goat` → **13/13 pass** (was 10/10 — G9a/G9b/G9c added for the hyperparameter ablation proxy)
- ✅ `cargo test --features swir_switch_thinking --lib swir::` → **38/38 unit tests pass** (33 original + 5 new G6 escape hatch tests)
- ✅ `cargo test --features swir_switch_thinking --test swir_strategy_integration` → **6/6 integration tests pass**
- ✅ `cargo check` (default, no swir) → clean
- ✅ `cargo check --features swir_switch_thinking` → clean

## Promotion Decision

**KEEP `swir_switch_thinking` OPT-IN.** Rationale:

1. **G3-G8, G1c, G2p all pass** — the controller is architecturally sound and
   plasma-tier fast (3.1 ns/step).
2. **G1 (accuracy) and G2 (efficiency) are deferred** — these are the paper's
   headline claims and require a real LLM. They are the actual GOAT criteria
   for promotion.
3. **No downstream consumers yet** — `thinking_cot` integration is wired but
   no host decode loop drives it. riir-ai Plan 299 will be the first consumer.
4. **Convergence-guard livelock FIXED** — the switch-count guards now fire as
   one-shot triggers per switch event (not every Explicit step). riir-ai Plan 299
   should still verify real-model entropy schedules produce the expected
   accuracy/efficiency.

Revisit promotion after riir-ai Plan 299 proves G1/G2 on a real model.
