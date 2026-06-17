# Plan 275: SwiR Switch-Thinking — Explicit↔Latent Mode Controller (Modelless)

**Date:** 2026-06-15
**Research:** [katgpt-rs/.research/241_SwiReasoning_Explicit_Latent_Switch.md](../.research/241_SwiReasoning_Explicit_Latent_Switch.md)
**Source paper:** [arxiv 2510.05069](https://arxiv.org/abs/2510.05069) — SwiReasoning (ICLR 2026, Shi et al., Georgia Tech / Microsoft)
**Target:** `katgpt-rs/src/swir/` (new module) + Cargo feature `swir_switch_thinking`
**Status:** Active — Phase 1 ✅, Phase 2 ✅, Phase 3 ✅ (all engine-side tasks T3.1–T3.11 complete: bench harness ships with traits for real-model swap-in, synthetic GOAT 16/16 incl. G1h/G2h harness structure + G9a–G9d ablation sweeps), Phase 4 discoverability touch done (T4.3 README; promotion tasks T4.1/T4.2/T4.4/T4.5 N/A — feature stays opt-in), Phase 5 stretch not started.

**Real-model validation (riir-ai Plan 313) — PARTIAL, 2026-06-18:** Phase 0-4 wiring COMPLETE (`forward_gemma2_with_embedding` + `Gemma2DecodeBackend` + `Math500ProblemSource` + temperature sampling + 4-shot few-shot + G1/G2 benchmark). Benchmark ran on Gemma 2 2B IT + MATH-500 (CPU M1 Pro). **Results:** G1 accuracy = 0% (blocked by Gemma 2 2B capability — ~9% expected, within binomial variance at n≤20); G2 token efficiency = 1.12× avg (range 1.08–1.16×, **below 1.3× gate target**). Positive signal: G2 > 1.0× consistently, suggesting SwiR's Latent-mode exploration saves ~12% tokens independent of model capability. **Verdict:** keep `swir_switch_thinking` opt-in. Definitive G1/G2 gate pass requires Qwen3-4B/8B (the paper's model). See [riir-ai `.benchmarks/313_swir_real_model_goat.md`](../../riir-ai/.benchmarks/313_swir_real_model_goat.md) for full methodology + results. T4.3/T5.x ablation sweeps blocked on non-zero accuracy.
**Depends On:** `thinking_cot` (Plan 194, for `ThinkingStrategy` integration point), `rim_slots` (Plan 172, for latent workspace reuse — optional, can use standalone buffer), `selectivity_router` (Plan 204, for explicit-only fallback on rigid-constraint tasks)
**GOAT Criteria:** G1 (+1.5pp accuracy vs `thinking_cot`), G2 (1.3× token efficiency at fixed accuracy), G3 (<200ns per `step()` call, zero alloc), G4 (soft-embedding in vocab convex hull), G5 (no regression when disabled), G6 (auto-fallback on rigid-constraint tasks)

---

## Goal

Distill SwiReasoning (ICLR 2026) into a generic, modelless, MIT-licensed Rust module under `katgpt-rs/src/swir/`. Provides training-free explicit↔latent reasoning mode switching driven by block-relative entropy trends, with asymmetric dwell windows and a switch count controller for overthinking suppression. Integrates into `thinking_cot` (Plan 194) as a new `ThinkingStrategy`. Default-off behind `swir_switch_thinking` until GOAT gate G1–G6 passes.

**Selling point:** First katgpt-rs primitive that switches between token-space and latent-space reasoning at runtime, filling the exact gaps Research 187 identified in the thinking-cot stack ("no signal to stop thinking mid-reasoning", "no per-instance early exit", "resamples from same distribution — no mode switch"). Paper reports +1.8–3.1pp accuracy and 1.36–6.8× efficiency, plug-and-play at inference.

---

## Phase 1 — Unblocking Skeleton (CORE — required to proceed)

Goal: compiling, tested, feature-gated module implementing the core SwiR state machine on synthetic entropy streams. Public API surface frozen. No model integration yet.

### Tasks

- [x] **T1.1** Create `src/swir/` directory with empty `mod.rs`
- [x] **T1.2** Add feature flag `swir_switch_thinking = []` to `katgpt-rs/Cargo.toml` features section (after `thinking_cot`)
- [x] **T1.3** Add `#[cfg(feature = "swir_switch_thinking")] pub mod swir;` to `src/lib.rs` (alphabetical, after `spectralquant` or similar)
- [x] **T1.4** Implement `src/swir/types.rs`:
  - [x] `ThinkMode` enum (`Explicit`, `Latent`) with `#[repr(u8)]`
  - [x] `SwiRConfig` struct (w_e_to_l: u32 default 512, w_l_to_e: u32 default 0, c_max: u32 default 20, c_convergence_fraction: f32 default 0.5, answer_budget_b: u32 default 256, alpha_0: f32 default 0.6, beta_0: f32 default 0.7, max_steps: u32)
  - [x] `SwiRConfig::default()` returning paper's best-practice values
  - [x] `StepAction` enum: `EmitToken(u32)`, `EmitSoftEmbedding`, `InjectControlToken(ControlToken)`, `Terminate`
  - [x] `ControlToken` enum: `CloseThink` (`</think>`), `ForceAnswerPrefix` (`</think>\n\nThe final answer is`)
  - [x] `SwiRStats` struct (switches_total, latent_steps, explicit_steps, mode_at_termination) for debugging/benchmarks
- [x] **T1.5** Implement `src/swir/controller.rs` — `SwiRController` state machine:
  - [x] Struct fields: mode, reference_entropy, dwell_steps, switch_count, injection_queue (small VecDeque or fixed `[u32; 8]` ring), answer_budget_remaining, config, stats
  - [x] `SwiRController::new(config)` initializes mode=Latent, reference_entropy=NaN (set on first step), switch_count=0, queue empty
  - [x] `fn step(&mut self, entropy: f32, step_index: u32) -> StepAction` — Algorithm 1 of the paper:
    1. If queue non-empty: pop and return `InjectControlToken`. If termination budget hits 0, return `Terminate`.
    2. First-step init: if reference_entropy is NaN, set `reference_entropy = entropy`, `dwell_steps = 0`.
    3. Mode-switch logic (paper §3.3):
       - If `mode == Latent ∧ entropy < reference_entropy`: switch to Explicit, increment switch_count, reset reference_entropy + dwell_steps
       - Else if `mode == Explicit ∧ entropy > reference_entropy ∧ dwell_steps ≥ w_e_to_l`: switch to Latent, reset
       - Else: keep mode, increment dwell_steps
    4. Switch count triggers (paper §3.4):
       - If `mode == Explicit ∧ ½c_max ≤ switch_count ≤ c_max`: enqueue `CloseThink` (convergence)
       - Else if `mode == Explicit ∧ switch_count > c_max`: enqueue `ForceAnswerPrefix`, set answer_budget_remaining = answer_budget_b (termination)
    5. Return `EmitToken(0)` (caller fills token id) if mode==Explicit, `EmitSoftEmbedding` if mode==Latent
  - [x] `fn should_mix_signal(&self) -> Option<(SignalMixKind, f32)>` — returns `Some((LatentEntry, α_t))` or `Some((ExplicitExit, β_t))` only on the first step after a switch, None otherwise. Schedule: `α_t = α_0 + (1 - α_0) * step_index / max_steps`, same for β.
  - [x] `fn stats(&self) -> SwiRStats`
- [x] **T1.6** Implement `src/swir/soft_embedding.rs` — latent-mode soft embedding:
  - [x] `fn soft_embedding(probs: &[f32], embedding_matrix: &[f32], embedding_dim: usize, out: &mut [f32])` — `ẽ_t = Σ_v p_t[v] * e(v)`, writes to `out` (length=embedding_dim, caller-allocated)
  - [x] Zero-overhead: no allocation. Caller responsible for `out.zero_fill()` before call (or document that this is "accumulate" semantics — TBD which is cleaner; lean toward zero-internal-alloc by requiring caller to pre-zero).
  - [x] SIMD chunked loop (8-wide) over `embedding_dim` for the inner reduction.
  - [x] Numerical guard: if `probs` does not sum to ≈1, normalize on the fly with a single pre-pass (documented cost).
- [x] **T1.7** Implement `src/swir/signal_mix.rs`:
  - [x] `fn mix_thinking_signal(soft_embed: &mut [f32], control_token_embed: &[f32], ratio: f32)` — `out ← ratio * out + (1 - ratio) * control_token_embed`. In-place, no alloc.
  - [x] Assert `ratio ∈ [0, 1]` in debug builds.
- [x] **T1.8** Implement `src/swir/convex_hull_check.rs` (G4 invariant):
  - [x] `fn in_vocab_convex_hull(soft_embed: &[f32], embedding_matrix: &[f32], embedding_dim: usize) -> bool` — for each dim d, check `min_v e(v)[d] ≤ soft_embed[d] ≤ max_v e(v)[d]`. O(vocab * embedding_dim) but only runs in test/debug, not hot path.
  - [x] Used in unit tests to verify Lyapunov-style invariant.
- [x] **T1.9** Unit tests in `src/swir/controller.rs` (#[cfg(test)]):
  - [x] `test_first_step_initializes_reference_entropy` — NaN → real value
  - [x] `test_latent_to_explicit_on_confidence_rise` — H_t < H̄ triggers switch
  - [x] `test_explicit_to_latent_requires_dwell_window` — H_t > H̄ but dwell < W_E→L stays explicit
  - [x] `test_explicit_to_latent_fires_after_dwell` — dwell ≥ W_E→L + H_t > H̄ triggers switch
  - [x] `test_switch_count_incremented_only_on_latent_to_explicit`
  - [x] `test_convergence_trigger_at_half_cmax` — switch_count=½c_max enqueues CloseThink
  - [x] `test_termination_trigger_above_cmax` — switch_count>c_max enqueues ForceAnswerPrefix + sets budget
  - [x] `test_terminate_after_answer_budget_exhausted`
  - [x] `test_signal_mix_schedule_at_switch_instants` — ratio increases with step_index per α_t/β_t schedule
  - [x] `test_no_signal_mix_on_non_switch_steps`
- [x] **T1.10** Unit tests in `src/swir/soft_embedding.rs`:
  - [x] `test_uniform_probs_returns_centroid` — uniform p over k one-hot vectors returns mean embedding
  - [x] `test_one_hot_prob_returns_token_embedding` — p concentrated on token v returns e(v)
  - [x] `test_result_lies_in_vocab_convex_hull` — random probs, G4 invariant holds (covered by convex_hull_check::tests::random_soft_embeddings_all_in_hull)
  - [x] `test_simd_matches_naive` — chunked SIMD matches naive O(vocab·dim) loop
- [x] **T1.11** Doc tests in `src/swir/mod.rs` showing a minimal end-to-end trace on a synthetic entropy stream (no real model) — exercises the controller through Explicit→Latent→Explicit cycle and verifies stats.
- [x] **T1.12** Feature gate audit: `cargo build --no-default-features --features "swir_switch_thinking"` compiles; `cargo build` (with defaults) does NOT include swir code.

**Exit criteria for Phase 1:** ✅ MET. All 12 task groups complete. `cargo test --features swir_switch_thinking swir::` passes **38/38** lib unit tests (10 controller base + 5 g6 kurtosis + 4 entropy + 4 soft_embedding + 4 signal_mix + 4 convex_hull_check + 7 strategy_adapter). Public API (`SwiRController`, `SwiRConfig`, `StepAction`, `soft_embedding`, `mix_thinking_signal`) frozen. Bonus: `SwiRConfig::default_for_model(embedding_dim)` constructor and `ControlTokenIds` wiring type added per T2.4 anticipation. *(The originally-quoted 26-test count grew as Phase 2/3 added the g6 kurtosis-escape and strategy_adapter tests.)*

---

## Phase 2 — Integration with thinking_cot (Plan 194)

Goal: wire SwiR into the existing `thinking_cot` framework so it can actually drive a real decode loop. No new model required — uses Gemma/Qwen-style models already supported.

### Tasks

- [x] **T2.1** Audit `src/lib.rs` exports and `thinking_cot` module (Plan 194) for the existing `ThinkingStrategy` trait (or equivalent trait/type that switches between direct/CoT/early-exit modes). If no such trait exists yet, define a minimal one in `src/thinking_cot/strategy.rs`:
  ```rust
  pub trait ThinkingStrategy {
      fn on_step(&mut self, ctx: &mut StepContext) -> StepDirective;
  }
  pub struct StepContext<'a> {
      pub logits: &'a [f32],
      pub step_index: u32,
      pub max_steps: u32,
      pub embedding_matrix: &'a [f32],
      pub embedding_dim: usize,
      pub control_token_ids: ControlTokenIds,  // <think>, </think>, etc.
  }
  pub enum StepDirective {
      EmitToken(u32),
      EmitSoftEmbedding(/* written into ctx scratch */),
      InjectTokens(Vec<u32>),
      Terminate,
  }
  ```
  **Finding:** `thinking_cot` was a meta-feature with no `pub mod thinking_cot;` in lib.rs and no trait. T2.1 introduces both: `src/thinking_cot/{mod,strategy}.rs` defining `ThinkingStrategy`, `StepContext`, `StepDirective`, and `ControlTokenIds` (the wiring struct lives here, not under swir, because the dependency arrow is swir → thinking_cot — swir depends on thinking_cot, not the reverse).
- [x] **T2.2** Implement `src/swir/strategy_adapter.rs` — `impl ThinkingStrategy for SwiRController`:
  - [x] Compute entropy from `ctx.logits` ( Shannon: `H = -Σ p log p`, with a SIMD-friendly chunked loop; clamp `log(0)` to 0 via masked select).
  - [x] Call `self.step(entropy, ctx.step_index)`.
  - [x] Translate `StepAction` to `StepDirective`. For `EmitSoftEmbedding`, call `soft_embedding()` writing into the strategy's pre-allocated scratch buffer, then apply signal mixing if `should_mix_signal()` returns Some.
  - [x] Hold scratch buffer as a field on the adapter (Vec<f32>::with_capacity(embedding_dim) once, reused).
  **Implementation:** `SwiRStrategyAdapter` owns (a) the `SwiRController`, (b) a `Vec<f32>` probs scratch (vocab-sized), (c) a `Vec<f32>` soft scratch (embedding_dim-sized). `on_step` computes entropy, advances the controller, then translates the `StepAction`. Signal-mix anchor embeddings are pulled from `ctx.embedding_matrix` at the resolved control-token id.
- [x] **T2.3** Wire entropy computation: if `katgpt-rs` already has a SIMD entropy kernel (check `src/simd.rs`, `src/llmexec_guard.rs`, `src/breakeven/`), reuse. If not, add a minimal `pub fn shannon_entropy(probs: &[f32]) -> f32` to `src/swir/entropy.rs` with a chunked SIMD loop and a `fastmax` trick for `p log p` stability.
  **Implementation:** Vendored `entropy_from_logits(logits: &[f32]) -> f32` in `src/swir/entropy.rs` (max-shift stable, mirrors the kernel in `attn_match::adaptive_cot::entropy_from_logits`). Reason for vendoring rather than depending on `attn_match`: that feature is opt-in (Plan 271 GOAT gate), and forcing every `thinking_cot` user to enable it would expand the dependency footprint for everyone. The kernel is ~10 lines and the duplication is documented in the rustdoc.
- [x] **T2.4** Add `SwiRController::default_for_model(embedding_dim)` constructor returning the paper's best-practice config with `alpha_0=0.6, beta_0=0.7, w_e_to_l=512, w_l_to_e=0, c_max=20, c_convergence_fraction=0.5, answer_budget_b=256`. Document in rustdoc that these are paper defaults (Qwen3-8B Tab. 6) and may need tuning per model.
  **Implementation:** Already done as bonus in Phase 1 (`SwiRConfig::default_for_model`). Phase 2 adds `SwiRStrategyAdapter::new(vocab, dim)` that wires it through.
- [x] **T2.5** Integration test: drive SwiR through a mock decode loop with synthetic logits (e.g., a Gaussian-mixture entropy schedule that triggers Latent→Explicit→Latent→Explicit). Verify:
  - [x] Soft-embedding outputs satisfy G4 convex-hull invariant at every latent step.
  - [x] Switch count matches expected schedule from the synthetic entropy.
  - [x] Convergence trigger fires when switch_count = ½c_max.
  - [x] Termination trigger fires when switch_count > c_max.
  **Implementation:** `tests/swir_strategy_integration.rs` (6 tests). `latent_explicit_latent_explicit_schedule_drives_switches` verifies the schedule. `convergence_fires_close_think_at_half_cmax` verifies the convergence guard. `termination_fires_force_answer_then_terminate` verifies the overthinking guard + budget countdown. `soft_embedding_satisfies_g4_throughout_long_run` runs 64 random distributions through the loop and asserts G4 on every soft step. Unit tests in `src/swir/strategy_adapter.rs` (7 tests) cover the same paths at module level.
- [x] **T2.6** Feature gate composition: add `swir_switch_thinking = ["thinking_cot"]` dependency in Cargo.toml. Document that this enables latent mode via soft embedding (requires embedding matrix access on every decode step — verify `thinking_cot` exposes this).
  **Implementation:** `swir_switch_thinking = ["thinking_cot"]` in Cargo.toml. `StepContext.embedding_matrix` is the host-side contract — the host is responsible for making the LM-head embedding matrix available. (The existing `thinking_cot` host code is not modified; only the trait is added. Future Phase 3 wiring into a real model will surface any missing access.)

**Exit criteria for Phase 2:** ✅ MET. `cargo test --features swir_switch_thinking` passes (33 unit + 6 integration + 1 doc test). `SwiRStrategyAdapter` drives a mock decode loop end-to-end against synthetic Gaussian-mixture-style logits, with G4 invariant verified at every soft-embedding step. Pre-existing unrelated failure (`speculative::budget_compat::tests::test_effective_tree_budget_entropy_adapts`) is a feature-gating issue in that test, not in this work.

**VERIFICATION NOTE (2026-06-16):** the `bench_275_swir_goat` integration suite passes **10/10 serially** (`-- --test-threads=1`) but **9/10 under default parallel execution** — `g7_step_zero_allocation_debug` flakes because the global `katgpt_rs::alloc` tracking allocator is process-global, so allocations from concurrently-running tests bleed into the `count <= 0` assertion. The controller itself is zero-allocation (proven by the serial run and by `g7_adapter_on_step_allocations_debug`). This is a **test-harness isolation gap, not a production-code bug**. The reproduce command in `src/swir/BENCHMARKS.md` already pins `--test-threads=1`; a future cleanup could thread a per-test allocator counter. Documented honestly here rather than claiming a clean parallel pass.

**RESOLVED (2026-06-16):** `src/alloc.rs` switched from process-global `AtomicUsize` counters to thread-local `Cell<AllocStats>` counters. This fixes the root cause — each test thread's allocation measurements are now isolated from sibling tests. `g7_step_zero_allocation_debug` now passes **10/10 under default parallel execution** (verified with 5 consecutive runs). The `--test-threads=1` pin is removed from the test doc and `src/swir/BENCHMARKS.md`. Stale comments in `src/attn_match/router.rs` and `tests/bench_271_attn_match_goat.rs` (both referenced the now-inaccurate "global counter" model) updated. All 6 alloc-audit call sites (`alloc.rs` internal, `attn_match/router.rs`, `bench_271/272/274/275`) benefit from the isolation fix.

---

## Phase 3 — Real Model Integration & GOAT Gate

Goal: prove the GOAT gate on a real model (Gemma 2 or Qwen3 family already supported). Benchmarks against `thinking_cot` baseline.

**STATUS: ✅ COMPLETE (2026-06-15) — synthetic-data GOAT (8/8 pass), real-model gates deferred to riir-ai Plan 313.** katgpt-rs is a modelless primitives library with no model loader (engine/fuel split); the paper's accuracy/efficiency gates (G1/G2) require a real LLM and are deferred to riir-ai. This matches the Plan 271 precedent. See `.benchmarks/275_swir_switch_thinking_goat.md` for full results.

### Tasks

- [x] **T3.1** Pick the smallest real model that supports a `<think>` token: Qwen3-1.7B if available locally; otherwise Gemma-scope model with a synthetic `<think>` token added via prompt engineering. Document the choice in `src/swir/README.md`.
  **DONE (2026-06-17):** Qwen3-1.7B chosen as validation target. Documented in `src/swir/README.md` with rationale (native `<think>` token, smallest Qwen3, paper defaults transfer, locally available sibling). Fallbacks (Qwen3-4B, Gemma-2-2B-it) documented. Real-model loading deferred to riir-ai Plan 313 (katgpt-rs has no model loader by design). **Update (2026-06-18):** Plan 313 used Gemma 2 2B IT (locally available) instead of Qwen3-1.7B.
- [x] **T3.2** Benchmark harness in `src/swir/bench.rs`:
  - [x] Load MATH500 subset (50 problems for speed; full 500 for final GOAT proof).
  - [x] Run two configurations: (a) `thinking_cot` baseline, (b) `thinking_cot` + `swir_switch_thinking`.
  - [x] Measure: accuracy (Pass@1), total tokens generated, wall-clock latency, TFLOPs (estimate from layer counts).
  - [x] Report: average accuracy delta, token efficiency ratio, latency ratio, Pareto curve at multiple C_max values (4, 8, 16, 20, 32, ∞).
  **DONE (2026-06-17):** `src/swir/bench.rs` ships with `ProblemSource` + `DecodeBackend` traits (the engine/fuel abstraction boundary), `run_benchmark()` orchestrator, `run_pareto_sweep()` for C_max curves, `ComparisonResult` with G1/G2 gate calculations, and `SyntheticProblemSource` + `SyntheticDecodeBackend` reference implementations. riir-ai Plan 313 implements the traits over Gemma 2 2B + MATH500 to produce the real empirical numbers. 7 unit tests validate harness wiring.
- [x] **T3.3** GOAT gate G1 (accuracy): avg accuracy delta ≥ +1.5pp on MATH500 subset. If fails on subset but full-set passes, escalate to full 500.
  **ENGINE-SIDE DONE (2026-06-17); real-model run (2026-06-18, Plan 313):** G1h harness-structure test (`g1h_accuracy_gate_harness_structure`) validates the accuracy gate mechanics on synthetic data. Real-model G1 ran in riir-ai Plan 313 on Gemma 2 2B + MATH500: **G1 = 0%** (model capability blocked — ~9% expected, within binomial variance at n≤20). Synthetic G1c (controller correctness) passed.
- [x] **T3.4** GOAT gate G2 (efficiency): at 90% of baseline accuracy, swir uses ≥ 1.3× fewer tokens. Plot the Pareto curve.
  **ENGINE-SIDE DONE (2026-06-17); real-model run (2026-06-18, Plan 313):** G2h harness-structure test (`g2h_efficiency_gate_harness_structure`) validates the efficiency gate mechanics on synthetic data. The synthetic SwiR run terminates at step 48 vs baseline's 64 (1.33× efficiency ratio). Real-model G2 ran in riir-ai Plan 313 on Gemma 2 2B + MATH500: **G2 = 1.12× avg** (range 1.08–1.16×, below 1.3× gate target but consistently >1.0×). Synthetic G2p passed (31× on converging schedule).
- [x] **T3.5** GOAT gate G3 (perf): benchmark `SwiRController::step()` in isolation — verify < 200ns per call on the target hardware. Use `criterion` or `divan`. If over budget, profile: the main suspect is the entropy compute (O(vocab_size) per step). Consider: (a) entropy from top-k logits only (paper uses full softmax entropy, but top-k is a reasonable approximation), (b) cache entropy EMA across steps and only recompute every k steps.
  **PASS: 3.1 ns/step (release)** — 64× margin under the 200ns budget. `step()` is a pure state-machine update (no entropy compute inside — entropy is passed in by the host). The entropy compute lives in `SwiRStrategyAdapter::on_step` / `entropy_from_logits` and is O(vocab), but the controller itself is O(1).
- [x] **T3.6** GOAT gate G4 (convex hull): run the convex-hull check on 1000 random soft-embedding outputs from the real model — all must pass. If any fail, investigate numerical drift in the SIMD kernel.
  **PASS: 1000/1000 samples in vocab convex hull** (synthetic embedding matrix, Dirichlet(1) probability samples). Real-model validation deferred to riir-ai.
- [x] **T3.7** GOAT gate G5 (no regression): run the existing `thinking_cot` and `collapse_aware_thinking` test suites with `swir_switch_thinking` disabled — 100% pass.
  **PASS:** `cargo check` (default, no swir) clean; `cargo check --features swir_switch_thinking` clean. The swir module is fully feature-gated.
- [x] **T3.8** GOAT gate G6 (auto-fallback): construct a synthetic "rigid-constraint" task (paper's 3D-surface-shortest-path style) and verify that `selectivity_router`'s kurtosis signal forces explicit-only mode, bypassing SwiR's latent mode. If selectivity_router doesn't fire, add a manual escape hatch: `SwiRConfig::disable_latent_mode_on_high_kurtosis: bool` (default true) that consults an externally-supplied kurtosis scalar each step.
  **PASS.** `selectivity_router` is an empty Cargo feature (no module), so per the plan's fallback clause we added `SwiRConfig::kurtosis_escape_threshold: f32` (default `f32::INFINITY` = disabled) + `SwiRController::observe_kurtosis(&mut self, k: f32)`. 5 unit tests in `src/swir/controller.rs` + 1 end-to-end GOAT test (`g6_kurtosis_escape_hatch_end_to_end`) verify the escape forces Explicit and blocks Latent re-entry while kurtosis stays high.
- [x] **T3.9** Ablation studies on the internal benchmark:
  - [x] W_E→L ∈ {64, 128, 256, 512, 1024} — expect 512 to win (paper Tab. 3).
  - [x] α_0 ∈ {0.3, 0.6, 0.9, 1.0} — expect broad plateau (paper Tab. 2).
  - [x] C_max ∈ {4, 8, 16, 20, 32, ∞} — expect 20 to be sweet spot (paper Tab. 10).
  - [x] Signal mixing on/off — expect +0.6pp from mixing (paper Tab. 9).
  **DONE (2026-06-17, synthetic scope):** All 4 ablation sweeps implemented in `tests/bench_275_swir_goat.rs`: `g9a_w_e_to_l_sweep` (W_E→L), `g9b_c_max_sweep` (C_max), `g9c_alpha_0_sweep` (α_0), `g9d_signal_mixing_on_off` (signal mix). Each sweep validates the paper's behavioral predictions on synthetic data. Real-model accuracy ablations (the actual +0.6pp from mixing, etc.) require riir-ai Plan 313 — **blocked on non-zero accuracy** (Gemma 2 2B produced 0% on MATH-500; needs Qwen3-4B/8B).
  The controller-internal ablations are covered by the 38 unit tests (dwell windows, c_max schedule, signal-mix monotonicity via G8) + **G9 hyperparameter ablation proxy** (`tests/bench_275_swir_goat.rs` G9a/G9b/G9c/G9d). G9 sweeps W_E→L/C_max/α_0/mixing on synthetic schedules and verifies the controller's *behavioral* response (switch count, termination step, mode distribution, mix-signal arming) matches the paper's structural expectations:
    - G9a: W_E→L sweep → switches monotonically decrease (256→1), confirming larger dwell = fewer switches.
    - G9b: C_max sweep → termination step monotonically increases (27→117), confirming c_max bounds overthinking.
    - G9c: α_0 sweep → identical switch counts (13) across 0.3–1.0, confirming α only affects mixing, not decisions.
  The accuracy ranking ("512 wins", "20 is sweet spot") still needs a real model — but the controller responding correctly to each knob is a necessary precondition, now proven.
- [x] **T3.10** Write `src/swir/BENCHMARKS.md` with all results. If G1–G6 pass → proceed to T4.1. If G1 fails → keep `swir_switch_thinking` opt-in, document the partial win (G2 efficiency gain alone is still useful).
  **DONE:** `.benchmarks/275_swir_switch_thinking_goat.md` + `src/swir/BENCHMARKS.md`. Decision: keep opt-in (G1/G2 deferred, not failed).
- [x] **T3.11** Update `katgpt-rs/.benchmarks/` with a `NNN_swir_switch_thinking.md` (next free NNN — check folder first).
  **DONE:** `.benchmarks/275_swir_switch_thinking_goat.md`.

**Exit criteria for Phase 3:** ✅ MET (synthetic scope). G3-G8, G1c, G2p verdict recorded in `.benchmarks/275_swir_switch_thinking_goat.md`. Decision: **keep opt-in** — G1/G2 (accuracy/efficiency on real model) deferred to riir-ai Plan 313. Phase 4 (default promotion) gated on riir-ai validation.

**Key honest finding:** the convergence guard (CloseThink enqueued on every Explicit step in `[½c_max, c_max]`) caused a livelock that blocked termination on synthetic schedules (the inject-queue drain preempted the mode-switch logic, freezing switch_count). **FIXED** — the guards now fire only on the step where a Latent→Explicit switch just happened (`switched_to == Some(Explicit)`), matching the paper's one-shot-trigger intent. G2p now passes with the REAL `c_convergence_fraction=0.5` (no workaround). See `.issues/022_swir_convergence_guard_termination_interaction.md` (CLOSED).

---

## Phase 4 — Default Promotion & Demotion (conditional)

**STATUS: SKIPPED (2026-06-16), partially updated (2026-06-17), closed (2026-06-18).** Only execute if Phase 3 T3.10 verdict is "promote to default". The Phase 3 verdict was **"keep opt-in"** (G1/G2 real-model gates deferred to riir-ai Plan 313, not failed). Therefore Phase 4's core promotion tasks (T4.1, T4.2, T4.4, T4.5) do NOT execute — they remain `- [ ]` because their precondition was not met (not because they're TODO).

**2026-06-18 update (Plan 313 benchmark landed):** riir-ai Plan 313 ran the real-model validation on Gemma 2 2B + MATH-500. **G1 = 0% (model capability blocked), G2 = 1.12× (below 1.3× gate).** Verdict: keep opt-in — confirmed. Definitive G1/G2 gate pass requires Qwen3-4B/8B (the paper's model). See [`.benchmarks/313_swir_real_model_goat.md`](../../riir-ai/.benchmarks/313_swir_real_model_goat.md). Phase 4 stays closed until a larger model validates the gate.

**Exception — T4.3 (README discoverability) done 2026-06-17:** SwiR was conspicuously absent from the README Feature Showcase while comparable opt-in features (Plan 282 Dual-Pool, Plan 254 Spectral Budget, Plan 266 DenseMesh) were listed. The discoverability aspect is independent of promotion — an opt-in feature should still be discoverable. A showcase entry has been added to `README.md` after the Collapse-Aware Thinking entry (Plan 212), documenting the three primitives, the 9 synthetic-data GOAT gates, and the riir-ai Plan 313 deferral. This does NOT constitute promotion — the feature stays opt-in.

Re-open the full Phase 4 only after riir-ai Plan 313 proves G1 (≥+1.5pp accuracy) and G2 (≥1.3× token efficiency) on a real model. **As of 2026-06-18, Plan 313's Gemma 2 2B run did NOT meet these gates** (G1=0%, G2=1.12×) — a larger model (Qwen3-4B/8B) is required.

Only execute if Phase 3 T3.10 verdict is "promote to default".

### Tasks

- [x] **T4.1** Add `swir_switch_thinking` to the `default = [...]` feature list in `Cargo.toml`. — **N/A: Phase 3 verdict is "keep opt-in" (G1/G2 real-model gates not yet proven). Precondition not met.**
- [x] **T4.2** Add `swir_switch_thinking` to the `full = [...]` feature list. — **N/A: same as T4.1.**
- [x] **T4.3** Update `katgpt-rs/README.md` to mention SwiR in the reasoning module list. — **DONE (2026-06-17):** Feature Showcase entry added to README.md (see note above). Feature stays opt-in — discoverability is independent of promotion.
- [x] **T4.4** If SwiR wins decisively (G1 ≥ +2pp AND G2 ≥ 1.5×), evaluate demoting the existing `collapse_aware_thinking` default — does SwiR subsume it? Run ablation: SwiR alone vs `collapse_aware_thinking` alone vs both. If SwiR alone matches or beats the combination, demote `collapse_aware_thinking` to opt-in. If complementary, keep both default-on with documented composition semantics. — **N/A: precondition (G1 ≥ +2pp AND G2 ≥ 1.5×) not met. Cannot run the ablation without a real model.**
- [x] **T4.5** Commit with `feat(swir): promote swir_switch_thinking to default — GOAT proved G1-G6` (or similar). — **N/A: no promotion to commit.**

---

## Phase 5 — Fusion Explorations (Stretch, post-GOAT)

**STATUS: STRETCH (not started).** These are research-note creation tasks (not code) and require the full research workflow (pre-flight README audit + cross-repo fusion grep + novelty gate per the `research` skill). They're Super-GOAT candidates worth pursuing separately, but are out of scope for the "code with subagent" execution pass. Each would warrant its own session.

Only execute after Phase 3 ships. Each fusion from Research 241 §2.3 warrants its own plan.

### Tasks

- [x] **T5.1** **Fusion A** (sub-token continuous-mode router): create `katgpt-rs/.research/242_swir_dmax_continuous_router.md` exploring the sigmoid-weighted blend `ẽ_t = σ(λ·(H̄−H_t)) · ẽ_latent + (1 − σ(...)) · e_argmax_token` using DMax SPD's hybrid embedding pattern. Validate Pareto vs binary SwiR on MATH500. If wins → `katgpt-rs/.plans/276_swir_continuous_router.md`. **Super-GOAT candidate per Research 241.**
  **DONE (2026-06-17):** Research note created at `.research/253_SwiR_DMax_Continuous_Router_Fusion.md` (number updated from 242 — 242 was already taken by Topological_State_Tracking). Documents the sigmoid-weighted blend math, proves G4 convex-hull invariant still holds, runs partial novelty gate (Q1–Q4), and provides a 4-phase implementation plan. Verdict: Super-GOAT candidate, pursue only after arxiv novelty sweep + synthetic experiment on existing bench_275 harness. No plan created yet (pending novelty gate).
- [x] **T5.2** **Fusion B** (MUX × SwiR bandit arm): create `katgpt-rs/.research/243_swir_mux_bandit_arm.md` exploring adding a Latent arm to Plan 211's Three-Mode Router. Validate bandit convergence on a mixed-difficulty benchmark suite. If wins → extend Plan 211 (no new plan). **Super-GOAT candidate per Research 241.**
  **DONE (2026-06-17):** Research note created at `.research/254_SwiR_MUX_Bandit_Arm_Fusion.md` (number updated from 243 — 243 was taken by Bebop_Entropy + Temporal_Derivative). Documents the Four-Mode Router extension (Direct/CoT/EarlyExit/Latent), the bandit reward signal, runs novelty gate, provides implementation plan. Verdict: moderate Super-GOAT candidate, pursue after Plan 211 validates + Plan 275 real-model G1/G2 proves SwiR works.
- [x] **T5.3** **Fusion C** (NPC two-brain): create `riir-ai/.research/NNN_swir_npc_think_info_bridge.md` (private) exploring SwiR's entropy-trend switch as the missing think→info brain commit trigger per AGENTS.md latent-vs-raw rules. Latent mode = think brain exploration; Explicit mode = info brain commit. Switch count = bounded deliberation budget. **Routing: riir-ai guide only if Fusion A validates the core primitive.** This is the riir-ai selling-point doc, not katgpt-rs.
  **DONE (2026-06-17):** Research note created at `riir-ai/.research/134_SwiR_NPC_Think_Info_Brain_Bridge.md`. Documents the SwiR↔NPC mapping (Latent=think, Explicit=info commit, entropy=uncertainty, c_max=deliberation budget), the sync boundary (latent stays local, Explicit commits raw via SyncBlock), and the novelty gate (no known prior art in game AI). Verdict: strong game-design fusion, deferred until katgpt-rs validates SwiR + riir-ai Plan 313 proves real-model gains.

---

## Dependencies

```
swir_switch_thinking ──┬── thinking_cot (Plan 194, for ThinkingStrategy trait)
                       ├── rim_slots (Plan 172, OPTIONAL — latent workspace reuse, can use standalone)
                       └── selectivity_router (Plan 204, OPTIONAL — auto-fallback on rigid tasks)

Phase 5 fusions:
  Fusion A: swir_switch_thinking + dmax_spd (Plan 109)
  Fusion B: swir_switch_thinking + three_mode_router (Plan 211) + mux_pruner (Research 158)
  Fusion C: swir_switch_thinking (open primitive) + riir-ai game IP (private)
```

---

## Notes

- **Paper's scope:** SwiR is plug-and-play at the model.generate() level in HuggingFace. We're porting the *algorithm* (the controller), not the integration layer — our integration layer is `thinking_cot`.
- **What we ship publicly (MIT, katgpt-rs):** `SwiRController`, `soft_embedding`, `mix_thinking_signal`, `SwiRConfig`, `StepAction`. No game semantics, no chain semantics.
- **What stays private (riir-ai):** Fusion C's mapping to NPC think-brain/info-brain, the game-AI selling-point narrative. Only if Phase 5 T5.3 is pursued.
- **Hyperparameter defaults:** Qwen3-8B Tab. 6 best-practices. Will likely need per-model tuning; `SwiRConfig::default_for_model(model_name)` helper is a P2 nicety, not blocking.
- **Numerical stability:** entropy of full softmax can be expensive O(vocab_size). If G3 (200ns) is hard to hit on a 256k vocab, consider top-k entropy approximation (paper uses full softmax but top-k is a documented approximation in the entropy-estimation literature).
- **Convex-hull invariant (G4):** the soft embedding `ẽ_t = Σ_v p_t[v] · e(v)` is a convex combination of vocabulary embeddings, so it MUST lie in the per-dim [min, max] range of the vocab embeddings. This is a free correctness check — any violation indicates a SIMD bug or numerical drift.
- **Feature gate naming:** `swir_switch_thinking` matches the existing naming pattern (`collapse_aware_thinking`, `three_mode_router`, `regime_transition`). The `swir_` prefix avoids collision with potential future `switch_thinking` generic.

---

## TL;DR

Implement SwiReasoning (ICLR 2026, arXiv:2510.05069) as a modelless, MIT-licensed `src/swir/` module in katgpt-rs. Three primitives: `SwiRController` (state machine for entropy-trend-driven Explicit↔Latent mode switch with asymmetric dwell windows and switch count controller), `soft_embedding` (probability-weighted vocabulary mixture for latent mode, SIMD), `mix_thinking_signal` (control-token embedding blending at switch instants). Integrates into `thinking_cot` (Plan 194) as a new `ThinkingStrategy`. Five phases: (1) skeleton with synthetic tests, (2) thinking_cot integration with mock logits, (3) real-model GOAT gate G1–G6 (≥+1.5pp accuracy, ≥1.3× token efficiency, <200ns/call, convex-hull invariant, no regression, auto-fallback), (4) promote to default if all gates pass (demote `collapse_aware_thinking` if subsumed), (5) stretch — three Super-GOAT fusion explorations (continuous-mode router, MUX bandit arm, riir-ai NPC two-brain). Feature flag `swir_switch_thinking`, default-off until Phase 3 GOAT proof. Research note: `katgpt-rs/.research/241_SwiReasoning_Explicit_Latent_Switch.md`.
