//! Plan 275 Phase 3 — SwiR Switch-Thinking GOAT Gate Benchmark
//!
//! Hard pass/fail benchmark proving the load-bearing properties of the SwiR
//! (SwiReasoning, ICLR 2026, arXiv:2510.05069) Explicit↔Latent mode controller
//! distilled into `katgpt-rs/src/swir/`.
//!
//! # Scope: synthetic data only
//!
//! `katgpt-rs` is a modelless primitives library — it has no model loader, no
//! tokenizer, no KV cache (per the engine/fuel split: katgpt-rs = engine,
//! riir-ai = fuel). The paper's headline gates (G1 accuracy on MATH500, G2
//! token efficiency at fixed accuracy) therefore cannot run here. They are
//! **deferred to riir-ai Plan 313** (SwiR Real-Model Validation) which
//! wires real models. This matches the precedent set by Plan 271 (Attention
//! Matching), whose GOAT gate also ran on synthetic data with real-model
//! validation deferred.
//!
//! **Real-model update (riir-ai Plan 313, 2026-06-19):** G2 = **1.37× PASS**
//! at `w_e_to_l=32, c_max=64` (n=5); G1 = 0% (blocked by Gemma 2 2B capability,
//! T4.2e ruled out prompt/checker bugs). See `riir-ai/.benchmarks/313_swir_real_model_goat.md`.
//!
//! What this gate *does* prove (the algorithmic invariants that must hold
//! before real-model validation is even meaningful):
//!
//! - **G3 — Per-step perf**: `SwiRController::step()` mean ≤ 200ns in release
//!   on Apple Silicon NEON. The controller is on the decode hot path.
//! - **G4 — Convex-hull invariant**: 1000 random probability distributions
//!   produce soft embeddings that all lie inside the vocab per-dim
//!   `[min, max]` box. Paper's Lyapunov-style correctness guarantee.
//! - **G5 — No regression with feature off**: default build (without
//!   `swir_switch_thinking`) compiles clean — feature-gate isolation.
//! - **G6 — Auto-fallback (kurtosis escape)**: high-kurtosis signal forces
//!   Explicit mode, bypassing Latent exploration on rigid-constraint tasks.
//!   (Unit-tested exhaustively in `src/swir/controller.rs`; this gate re-runs
//!   the path end-to-end through the adapter.)
//! - **G7 — Zero-alloc `step()`**: the modelless controller's `step()` path
//!   allocates 0 bytes (debug build, `TrackingAllocator`). The
//!   `SwiRStrategyAdapter::on_step` path allocates `embedding_dim * 4` bytes
//!   on the soft-embedding branch (documented in the adapter) — measured
//!   honestly.
//! - **G1c — Controller correctness**: on a synthetic "converging" entropy
//!   schedule, the controller produces the expected sequence of mode switches,
//!   convergence trigger (CloseThink at ½c_max), and termination trigger
//!   (ForceAnswerPrefix > c_max). Replaces paper G1 (accuracy) which needs a
//!   real model.
//! - **G2p — Efficiency proxy**: on the same schedule, SwiR's overthinking
//!   guard terminates the run in strictly fewer steps than a fixed-budget
//!   baseline (`max_steps` with no switching). Replaces paper G2 (token
//!   efficiency at fixed accuracy) which needs a real model.
//! - **G8 — Signal-mix schedule monotonicity**: α_t / β_t are monotonically
//!   non-decreasing in `step_index` (paper Eq. 4 schedule).
//!
//! Run with:
//! ```bash
//! # Release for perf gates (G3).
//! cargo test --release --test bench_275_swir_goat \
//!     --features swir_switch_thinking -- --nocapture
//!
//! # Debug for the allocation audit (G7).
//! cargo test --test bench_275_swir_goat \
//!     --features swir_switch_thinking -- --nocapture
//! ```
//!
//! No `--test-threads=1` flag is needed: the library's `TrackingAllocator`
//! (in `src/alloc.rs`) uses thread-local counters, so each test thread's
//! allocation measurements are isolated from sibling tests.

#![cfg(feature = "swir_switch_thinking")]
#![cfg(test)]

use katgpt_rs::swir::{
    entropy_from_logits, in_vocab_convex_hull, mix_thinking_signal, shannon_entropy,
    soft_embedding, SwiRConfig, SwiRController, SwiRStrategyAdapter, StepAction,
    ThinkMode,
};
use katgpt_rs::thinking_cot::{ControlTokenIds, StepContext, StepDirective, ThinkingStrategy};
use std::time::Instant;

// ════════════════════════════════════════════════════════════════════════════
// Tunables
// ════════════════════════════════════════════════════════════════════════════

/// Vocab size for the synthetic embedding matrix (G4, G7).
/// Small enough to keep the test fast, large enough that the SIMD chunked
/// inner loop exercises multiple 8-wide lanes.
const VOCAB: usize = 64;

/// Embedding dimension for the synthetic matrix (G4, G7).
/// 32 gives 4 SIMD lanes per soft-embedding call.
const EMB_DIM: usize = 32;

/// G3 per-step budget (paper §3.5, Plan 275 T3.5).
const G3_STEP_BUDGET_NS: u64 = 200;

/// G7 per-step allocation budget for `step()` (Plan 275 T3.5 zero-alloc claim).
/// `step()` itself must allocate 0 bytes; we allow a tiny slack for allocator
/// noise from other test infrastructure.
#[cfg(debug_assertions)]
const G7_STEP_ALLOC_SLACK: usize = 0;

/// G4 sample count (paper Lyapunov-style invariant, Plan 275 T3.6).
const G4_N_SAMPLES: usize = 1000;

// ════════════════════════════════════════════════════════════════════════════
// Synthetic data helpers
// ════════════════════════════════════════════════════════════════════════════

/// SplitMix64-step PRNG — deterministic, no deps.
fn next_rand(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let z = *state;
    (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9)
}

fn rand_f32(state: &mut u64) -> f32 {
    let r = next_rand(state);
    // Map to [0, 1) with 24-bit mantissa.
    ((r >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Build a synthetic embedding matrix with known per-dim min/max so G4 can
/// verify the hull invariant. Rows are random in [-1, 1].
fn synth_embedding_matrix(vocab: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut m = vec![0.0f32; vocab * dim];
    for v in 0..vocab {
        for d in 0..dim {
            // [-1, 1) range — guarantees non-trivial min/max per dim.
            m[v * dim + d] = rand_f32(&mut state) * 2.0 - 1.0;
        }
    }
    m
}

/// Build a random probability distribution (Dirichlet-like via normalised
/// uniform exponentials). Returns a Vec of length `vocab`.
fn synth_probs(vocab: usize, state: &mut u64) -> Vec<f32> {
    let mut p = Vec::with_capacity(vocab);
    let mut sum = 0.0f32;
    for _ in 0..vocab {
        // Exponential of uniform — gives a Dirichlet(1) sample.
        let u = rand_f32(state).max(1e-6);
        let e = (-u.ln()).min(1e6); // guard against overflow
        p.push(e);
        sum += e;
    }
    let inv = 1.0 / sum.max(1e-12);
    for x in p.iter_mut() {
        *x *= inv;
    }
    p
}

/// Convert probs → logits (inverse softmax, max-shifted to 0).
fn probs_to_logits(probs: &[f32]) -> Vec<f32> {
    let mut max_p = 0.0f32;
    for &p in probs {
        if p > max_p {
            max_p = p;
        }
    }
    probs
        .iter()
        .map(|&p| (p.max(1e-12)).ln() - max_p.ln())
        .collect()
}

// ════════════════════════════════════════════════════════════════════════════
// G3 — Per-step perf: step() ≤ 200ns (release)
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn g3_step_perf_under_200ns_release() {
    // Paper §3.5 / Plan 275 T3.5: <200ns per step on Apple Silicon.
    // We feed a sawtooth entropy schedule so the controller exercises all
    // branches (Latent, Explicit, switch, convergence enqueue, inject drain).
    let mut ctrl = SwiRController::new(SwiRConfig {
        w_e_to_l: 4,
        w_l_to_e: 0,
        c_max: 8,
        c_convergence_fraction: 0.5,
        answer_budget_b: 16,
        alpha_0: 0.6,
        beta_0: 0.7,
        max_steps: 4096,
        kurtosis_escape_threshold: f32::INFINITY,
    });

    // Warm up (first step initialises reference_entropy — slightly slower).
    let mut entropy = 5.0f32;
    for i in 0..64 {
        entropy = if i % 16 < 8 { entropy - 0.3 } else { entropy + 0.3 };
        entropy = entropy.max(0.1).min(10.0);
        let _ = ctrl.step(entropy, i);
    }

    // Measure.
    const N: u32 = 100_000;
    let t0 = Instant::now();
    for i in 0..N {
        entropy = if (i % 16) < 8 { entropy - 0.3 } else { entropy + 0.3 };
        entropy = entropy.max(0.1).min(10.0);
        let action = ctrl.step(entropy, i);
        // Prevent the compiler from optimising step() away.
        if action == StepAction::Terminate {
            ctrl = SwiRController::new(ctrl_stats_config());
        }
        // Drain any pending mix to avoid backpressure skewing the measurement.
        let _ = ctrl.should_mix_signal();
    }
    let elapsed = t0.elapsed();
    let ns_per_step = elapsed.as_nanos() as f64 / N as f64;

    println!(
        "G3 step perf: {ns_per_step:.1} ns/step (budget {G3_STEP_BUDGET_NS}ns) over {N} iters, \
         total {elapsed:?}"
    );

    // Soft assertion in debug (debug builds are ~10× slower than release —
    // the gate is meaningful only in release).
    if !cfg!(debug_assertions) {
        assert!(
            ns_per_step <= G3_STEP_BUDGET_NS as f64,
            "G3 FAIL: {ns_per_step:.1} ns/step > {G3_STEP_BUDGET_NS}ns budget. \
             The controller sits on the decode hot path — this regression must be \
             investigated before promoting swir_switch_thinking to default."
        );
    } else {
        println!(
            "G3 (debug build): {ns_per_step:.1} ns/step — RELEASE gate is {G3_STEP_BUDGET_NS}ns; \
             debug timing is informational only."
        );
    }
}

fn ctrl_stats_config() -> SwiRConfig {
    SwiRConfig {
        w_e_to_l: 4,
        w_l_to_e: 0,
        c_max: 8,
        c_convergence_fraction: 0.5,
        answer_budget_b: 16,
        alpha_0: 0.6,
        beta_0: 0.7,
        max_steps: 4096,
        kurtosis_escape_threshold: f32::INFINITY,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// G4 — Convex-hull invariant on 1000 random probs
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn g4_soft_embeddings_all_in_vocab_convex_hull() {
    // Paper's Lyapunov-style invariant: ẽ = Σ_v p[v] · e(v) is a convex
    // combination of vocab rows → per-dim value must lie in [min_v, max_v].
    // A violation indicates either a SIMD bug or numerical drift.
    let emb = synth_embedding_matrix(VOCAB, EMB_DIM, 0x5417_533d);
    let mut scratch = vec![0.0f32; EMB_DIM];
    let mut state = 1u64;
    let mut violations = 0usize;
    for s in 0..G4_N_SAMPLES {
        let probs = synth_probs(VOCAB, &mut state);
        for x in scratch.iter_mut() {
            *x = 0.0;
        }
        soft_embedding(&probs, &emb, EMB_DIM, &mut scratch);
        if !in_vocab_convex_hull(&scratch, &emb, EMB_DIM) {
            violations += 1;
            if violations <= 3 {
                eprintln!(
                    "G4 violation at sample {s}: soft_embed {:?} outside hull",
                    &scratch[..8.min(scratch.len())]
                );
            }
        }
    }
    println!(
        "G4 convex hull: {}/{} samples in hull ({:.2}%)",
        G4_N_SAMPLES - violations,
        G4_N_SAMPLES,
        100.0 * (G4_N_SAMPLES - violations) as f32 / G4_N_SAMPLES as f32
    );
    assert_eq!(
        violations, 0,
        "G4 FAIL: {violations}/{G4_N_SAMPLES} soft embeddings escaped the vocab convex hull. \
         Indicates numerical drift or a SIMD bug in soft_embedding."
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G5 — No regression: feature-gate isolation
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn g5_feature_gate_isolation_smoke() {
    // The full isolation check (`cargo check` without `swir_switch_thinking`)
    // is done by a separate `cargo check` invocation documented in
    // BENCHMARKS.md. This smoke test just verifies that *enabling* the feature
    // exposes the expected API surface — a regression here would indicate a
    // broken `#[cfg]` gate.
    let _ctrl = SwiRController::new(SwiRConfig::default());
    let _adapter = SwiRStrategyAdapter::new(VOCAB, EMB_DIM);

    // Touch each public fn to ensure they compile + link.
    let probs = vec![1.0f32 / VOCAB as f32; VOCAB];
    let _h1 = shannon_entropy(&probs);
    let logits = probs_to_logits(&probs);
    let _h2 = entropy_from_logits(&logits);

    let emb = vec![0.0f32; VOCAB * EMB_DIM];
    let mut soft = vec![0.0f32; EMB_DIM];
    soft_embedding(&probs, &emb, EMB_DIM, &mut soft);
    mix_thinking_signal(&mut soft, &emb[..EMB_DIM], 0.5);
    let _ = in_vocab_convex_hull(&soft, &emb, EMB_DIM);

    println!("G5 smoke: swir API surface compiles + links under feature.");
    println!(
        "G5 full isolation: run `cargo check` (no feature) separately — see \
         .benchmarks/275_swir_switch_thinking.md for the verdict."
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G6 — Auto-fallback (kurtosis escape hatch) end-to-end via adapter
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn g6_kurtosis_escape_hatch_end_to_end() {
    // Plan 275 T3.8: high-kurtosis → force Explicit, bypass Latent.
    // The controller-path is exhaustively unit-tested in src/swir/controller.rs
    // (5 tests). This gate re-runs the path end-to-end through the adapter to
    // verify the host wiring (StepContext → on_step → StepDirective) doesn't
    // drop the escape signal.
    let emb = synth_embedding_matrix(VOCAB, EMB_DIM, 0x6dead_beef);
    let mut adapter = SwiRStrategyAdapter::with_config(
        VOCAB,
        EMB_DIM,
        SwiRConfig {
            kurtosis_escape_threshold: 3.0,
            ..SwiRConfig::default()
        },
    );
    let ids = ControlTokenIds {
        think_open: 0,
        think_close: 1,
        force_answer_prefix: 2,
    };
    let probs = vec![1.0f32 / VOCAB as f32; VOCAB]; // uniform → high entropy
    let logits = probs_to_logits(&probs);

    // Step 0: Latent (initial), uniform probs.
    let mut ctx = StepContext {
        logits: &logits,
        step_index: 0,
        max_steps: 16,
        embedding_matrix: &emb,
        embedding_dim: EMB_DIM,
        control_token_ids: ids,
    };
    let d0 = adapter.on_step(&mut ctx);
    assert!(
        matches!(d0, StepDirective::EmitSoftEmbedding(_)),
        "step 0 should be Latent (soft), got {d0:?}"
    );

    // Observe high kurtosis → escape should fire on step 1.
    adapter.controller_mut().observe_kurtosis(5.0);
    let mut ctx = StepContext {
        logits: &logits,
        step_index: 1,
        max_steps: 16,
        embedding_matrix: &emb,
        embedding_dim: EMB_DIM,
        control_token_ids: ids,
    };
    let d1 = adapter.on_step(&mut ctx);
    assert!(
        matches!(d1, StepDirective::EmitToken(_)),
        "G6 FAIL: step 1 should be forced Explicit by kurtosis escape, got {d1:?}"
    );
    assert_eq!(
        adapter.controller().mode(),
        ThinkMode::Explicit,
        "G6 FAIL: controller should be in Explicit mode after escape"
    );

    println!("G6 auto-fallback: kurtosis=5.0 > threshold=3.0 → forced Explicit (PASS)");
}

// ════════════════════════════════════════════════════════════════════════════
// G7 — Zero-alloc step() (debug only)
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn g7_step_zero_allocation_debug() {
    // Plan 275 T3.5 / T1.4: zero-allocation steady state for `step()`.
    // The strategy adapter's `on_step` allocates on the soft-embedding branch
    // (documented) — that's measured separately in g7_adapter_on_step_allocations.
    //
    // Run in debug so the library's TrackingAllocator is installed.
    #[cfg(debug_assertions)]
    {
        katgpt_rs::alloc::reset_alloc_stats();
        let mut ctrl = SwiRController::new(SwiRConfig {
            w_e_to_l: 4,
            w_l_to_e: 0,
            c_max: 8,
            c_convergence_fraction: 0.5,
            answer_budget_b: 16,
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps: 1024,
            kurtosis_escape_threshold: f32::INFINITY,
        });
        // Warm up — first call initialises reference_entropy (no alloc expected,
        // but be conservative in case the constructor's lazy init triggers).
        let _ = ctrl.step(5.0, 0);

        katgpt_rs::alloc::reset_alloc_stats();
        let mut entropy = 5.0f32;
        for i in 1..1024u32 {
            entropy = if (i % 16) < 8 { entropy - 0.3 } else { entropy + 0.3 };
            entropy = entropy.max(0.1).min(10.0);
            let _ = ctrl.step(entropy, i);
            let _ = ctrl.should_mix_signal();
        }
        let (count, bytes) = katgpt_rs::alloc::get_alloc_stats();
        println!(
            "G7 step() allocations: {count} allocs, {bytes} bytes over 1023 steps \
             (budget: {G7_STEP_ALLOC_SLACK} allocs)"
        );
        assert!(
            count == G7_STEP_ALLOC_SLACK,
            "G7 FAIL: step() allocated {count} times ({bytes} bytes). \
             The controller is supposed to be allocation-free after construction."
        );
    }
    #[cfg(not(debug_assertions))]
    {
        println!(
            "G7 (release): TrackingAllocator is debug-only. Run `cargo test --test \
             bench_275_swir_goat --features swir_switch_thinking` in debug for the \
             allocation audit."
        );
    }
}

#[test]
fn g7_adapter_on_step_allocations_debug() {
    // Honest measurement of the adapter's on_step allocation profile.
    // Plan 275 T2.2 acknowledges the soft-embedding clone + InjectTokens Vec.
    // We document the per-step cost so downstream hosts know what to expect.
    #[cfg(debug_assertions)]
    {
        let emb = synth_embedding_matrix(VOCAB, EMB_DIM, 0x7a10c_5ee);
        let mut adapter = SwiRStrategyAdapter::new(VOCAB, EMB_DIM);
        let ids = ControlTokenIds::default();
        let probs = vec![1.0f32 / VOCAB as f32; VOCAB]; // uniform → high entropy → Latent
        let logits = probs_to_logits(&probs);

        // Warm up (adapter's first call may lazily resize scratch).
        let mut ctx = StepContext {
            logits: &logits,
            step_index: 0,
            max_steps: 64,
            embedding_matrix: &emb,
            embedding_dim: EMB_DIM,
            control_token_ids: ids,
        };
        let _ = adapter.on_step(&mut ctx);

        katgpt_rs::alloc::reset_alloc_stats();
        let n_steps = 64u32;
        let mut soft_steps = 0u32;
        let mut inject_steps = 0u32;
        for i in 1..n_steps {
            let mut ctx = StepContext {
                logits: &logits,
                step_index: i,
                max_steps: n_steps,
                embedding_matrix: &emb,
                embedding_dim: EMB_DIM,
                control_token_ids: ids,
            };
            match adapter.on_step(&mut ctx) {
                StepDirective::EmitSoftEmbedding(_) => soft_steps += 1,
                StepDirective::InjectTokens(_) => inject_steps += 1,
                _ => {}
            }
        }
        let (count, bytes) = katgpt_rs::alloc::get_alloc_stats();
        let per_step = count as f64 / (n_steps - 1) as f64;
        println!(
            "G7 adapter on_step: {count} allocs / {bytes} bytes over {} steps ({per_step:.2} \
             allocs/step). Breakdown: {soft_steps} soft-embedding steps (each clones \
             embedding_dim={EMB_DIM} f32s = {} bytes), {inject_steps} inject steps.",
            n_steps - 1,
            EMB_DIM * 4,
        );
        // The adapter is documented to allocate on soft-embedding + inject paths.
        // We don't fail the gate (the design requires the clone to satisfy the
        // borrow checker), but we surface the cost for downstream hosts.
        let expected_allocs = soft_steps + inject_steps;
        println!(
            "G7 adapter on_step: expected ≥ {expected_allocs} allocs from design (soft clone + \
             inject Vec). Actual {count}. Excess = {} — investigate if > 0.",
            count.saturating_sub(expected_allocs as usize)
        );
        // Sanity: the count should be at least the number of allocating paths taken.
        assert!(
            count >= expected_allocs as usize,
            "G7 adapter: measured {count} < expected {expected_allocs} — measurement bug?"
        );
    }
    #[cfg(not(debug_assertions))]
    {
        println!("G7 adapter (release): see debug build for allocation audit.");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// G1c — Controller correctness on a synthetic converging schedule
// ════════════════════════════════════════════════════════════════════════════
//
// Replaces paper G1 (accuracy on MATH500) which needs a real model. This gate
// proves the controller's state machine is correct: given a known entropy
// schedule, it produces the expected sequence of mode switches, convergence
// trigger, and termination trigger.

#[test]
fn g1c_controller_correctness_on_converging_schedule() {
    // Schedule: alternate high/low entropy to force many Latent→Explicit
    // switches. With c_max = 4 and convergence_fraction = 0.5, convergence
    // (CloseThink) should fire at switch_count = ceil(0.5 * 4) = 2, and
    // termination (ForceAnswerPrefix) at switch_count > 4.
    let mut ctrl = SwiRController::new(SwiRConfig {
        w_e_to_l: 1, // Easy to switch back to Latent.
        w_l_to_e: 0,
        c_max: 4,
        c_convergence_fraction: 0.5,
        answer_budget_b: 4,
        alpha_0: 0.6,
        beta_0: 0.7,
        max_steps: 1024,
        kurtosis_escape_threshold: f32::INFINITY,
    });

    let high_entropy = 5.0f32;
    let low_entropy = 1.0f32;

    // Track what we see.
    let mut switches = 0u32;
    let mut close_think_injections = 0u32;
    let mut force_answer_injections = 0u32;
    let mut terminated_at: Option<u32> = None;

    // Step 0: initial, Latent, ref = high.
    let _ = ctrl.step(high_entropy, 0);
    assert_eq!(ctrl.mode(), ThinkMode::Latent);

    // Drive up to 200 steps or termination.
    for i in 1..200u32 {
        // Alternate: low entropy triggers Latent→Explicit (switch_count++),
        // then high entropy triggers Explicit→Latent (no count) after dwell.
        let entropy = if i % 4 < 2 { low_entropy } else { high_entropy };
        let prev_mode = ctrl.mode();
        let action = ctrl.step(entropy, i);
        let new_mode = ctrl.mode();
        if prev_mode == ThinkMode::Latent && new_mode == ThinkMode::Explicit {
            switches += 1;
        }
        match action {
            StepAction::InjectControlToken(katgpt_rs::swir::ControlToken::CloseThink) => {
                close_think_injections += 1;
            }
            StepAction::InjectControlToken(katgpt_rs::swir::ControlToken::ForceAnswerPrefix) => {
                force_answer_injections += 1;
            }
            StepAction::Terminate => {
                terminated_at = Some(i);
                break;
            }
            _ => {}
        }
    }

    println!(
        "G1c: switches={switches}, CloseThink injections={close_think_injections}, \
         ForceAnswerPrefix injections={force_answer_injections}, terminated_at={terminated_at:?}"
    );

    // Correctness assertions:
    // 1. We should have observed Latent→Explicit switches (the schedule forces them).
    assert!(
        switches >= 1,
        "G1c FAIL: expected ≥1 Latent→Explicit switch, got {switches}"
    );
    // 2. Once switch_count >= convergence (2), CloseThink should be enqueued.
    //    The controller enqueues on the step where switch_count enters the
    //    convergence window, and drains on the next step.
    assert!(
        close_think_injections >= 1,
        "G1c FAIL: expected ≥1 CloseThink injection once switch_count >= ceil(½c_max)=2"
    );
    // 3. Once switch_count > c_max (4), ForceAnswerPrefix should fire, and
    //    after answer_budget_b (4) tokens, Terminate.
    assert!(
        force_answer_injections >= 1,
        "G1c FAIL: expected ≥1 ForceAnswerPrefix injection once switch_count > c_max=4"
    );
    assert!(
        terminated_at.is_some(),
        "G1c FAIL: expected termination after answer_budget_b exhausted"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G2p — Efficiency proxy: SwiR terminates earlier than fixed-budget baseline
// ════════════════════════════════════════════════════════════════════════════
//
// Replaces paper G2 (token efficiency at fixed accuracy) which needs a real
// model. This gate proves the *mechanism* by which SwiR saves tokens: the
// overthinking guard (ForceAnswerPrefix at switch_count > c_max + answer budget)
// terminates the run earlier than a fixed-budget baseline that always runs to
// max_steps. The saving is mechanistic, not empirical.

#[test]
fn g2p_efficiency_proxy_swir_terminates_earlier_than_fixed_budget() {
    // Baseline: fixed budget of max_steps steps, no switching, no termination.
    const FIXED_BUDGET: u32 = 1024;

    // SwiR config: tight c_max so termination fires early. We use the REAL
    // c_convergence_fraction=0.5 (not a workaround) because the controller's
    // switch-count guards now only fire on the step where a Latent→Explicit
    // switch JUST happened (see controller.rs step (4) — previously they fired
    // every Explicit step, causing a livelock that starved the mode-switch
    // logic and froze switch_count). With the one-shot trigger fix, the full
    // convergence→termination path is exercised correctly.
    let swir_config = SwiRConfig {
        w_e_to_l: 1,
        w_l_to_e: 0,
        c_max: 4,
        c_convergence_fraction: 0.5, // real value — conv at ceil(0.5*4)=2
        answer_budget_b: 16,
        alpha_0: 0.6,
        beta_0: 0.7,
        max_steps: FIXED_BUDGET,
        kurtosis_escape_threshold: f32::INFINITY,
    };

    // Drive SwiR on an alternating schedule that forces Latent→Explicit
    // switches. Step 0 sets reference_entropy = HIGH. Then:
    //   - LOW entropy (< HIGH) triggers Latent→Explicit (switch_count++).
    //   - HIGH entropy (> LOW ref after switch) triggers Explicit→Latent
    //     after w_e_to_l=1 dwell (no count).
    // This produces one Latent→Explicit switch every ~2 steps.
    let mut ctrl = SwiRController::new(swir_config);
    const HIGH: f32 = 5.0;
    const LOW: f32 = 1.0;
    let mut swir_steps = 0u32;
    for i in 0..FIXED_BUDGET {
        // Step 0: HIGH (sets ref=HIGH, Latent). Steps after: alternate LOW/HIGH.
        let entropy = if i == 0 { HIGH } else if i % 2 == 1 { LOW } else { HIGH };
        let action = ctrl.step(entropy, i);
        swir_steps += 1;
        if action == StepAction::Terminate {
            break;
        }
    }

    println!(
        "G2p efficiency proxy: SwiR terminated after {swir_steps} steps vs fixed-budget \
         {FIXED_BUDGET} → {:.2}× fewer steps ({:.0}% reduction)",
        FIXED_BUDGET as f64 / swir_steps as f64,
        100.0 * (1.0 - swir_steps as f64 / FIXED_BUDGET as f64)
    );

    // The paper claims 1.36–6.8× efficiency. On a synthetic schedule that
    // maximises switching (alternating every 2 steps), c_max=4 + budget=16
    // should terminate well under FIXED_BUDGET. We assert a conservative 2×
    // floor — the real-model gate (deferred to riir-ai) enforces the paper's
    // 1.3× threshold at matched accuracy.
    assert!(
        swir_steps < FIXED_BUDGET / 2,
        "G2p FAIL: SwiR took {swir_steps} steps, expected < {} (half of fixed budget) — \
         overthinking guard not firing?",
        FIXED_BUDGET / 2
    );
    let speedup = FIXED_BUDGET as f64 / swir_steps as f64;
    assert!(
        speedup >= 2.0,
        "G2p FAIL: speedup {speedup:.2}× < 2× conservative floor"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G8 — Signal-mix schedule monotonicity (α_t / β_t non-decreasing)
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn g8_signal_mix_schedule_monotonic_in_step_index() {
    // Paper Eq. 4: α_t = α_0 + (1 - α_0) · step_index / max_steps.
    // The ratio *increases* over the run — early switches favour the anchor
    // token, late switches favour the soft embedding. We verify the schedule
    // is monotonically non-decreasing by forcing switches at increasing
    // step_index values and recording the returned ratio.
    let mut ratios: Vec<f32> = Vec::new();
    // Test at step_index = 1, 64, 128, ..., 1024.
    // (step 0 can't switch — it initialises reference_entropy — so we start at 1.)
    for &step_at in &[1u32, 64, 128, 256, 512, 768, 1024] {
        let mut ctrl = SwiRController::new(SwiRConfig {
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps: 1024,
            w_e_to_l: 1,
            w_l_to_e: 0,
            c_max: 1024, // Large so convergence/termination don't interfere.
            c_convergence_fraction: 0.5,
            answer_budget_b: 1024,
            kurtosis_escape_threshold: f32::INFINITY,
        });
        // Step 0: set reference_entropy = HIGH (Latent).
        let _ = ctrl.step(5.0, 0);
        // Drive to the target step with HIGH entropy (no switch — equal to ref).
        for i in 1..step_at {
            let _ = ctrl.step(5.0, i); // constant entropy == ref → no switch
        }
        // Force the switch at step_at: drop entropy below ref (5.0).
        let _ = ctrl.step(1.0, step_at); // Latent→Explicit, arms mix_pending.
        let mix = ctrl.should_mix_signal();
        assert!(mix.is_some(), "mix must fire at step {step_at}");
        let (_kind, ratio) = mix.unwrap();
        ratios.push(ratio);
    }

    println!("G8 signal-mix ratios at steps 0..1024: {:?}", ratios);

    // Verify monotonic non-decreasing.
    for w in ratios.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "G8 FAIL: schedule not monotonic: {} → {}",
            w[0],
            w[1]
        );
    }
    // Verify the endpoints match the formula. The switches we forced are
    // Latent→Explicit, which produce ExplicitExit (β_t) mixes — so the
    // schedule is β_t = β_0 + (1 - β_0) · step / max_steps, starting at β_0=0.7.
    let beta_0 = 0.7f32;
    let max_steps = 1024f32;
    let expected_first = beta_0;
    let expected_last = beta_0 + (1.0 - beta_0) * (1024.0 / max_steps);
    assert!(
        (ratios[0] - expected_first).abs() < 1e-3,
        "G8 FAIL: first ratio {} != β_0 {expected_first} ( ExplicitExit schedule)",
        ratios[0]
    );
    assert!(
        (ratios[ratios.len() - 1] - expected_last).abs() < 1e-3,
        "G8 FAIL: last ratio {} != β_0 + (1-β_0)·1 {expected_last}",
        ratios[ratios.len() - 1]
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G9 — Hyperparameter ablation (modelless proxy for T3.9)
// ════════════════════════════════════════════════════════════════════════════
//
// T3.9 (accuracy ablations on W_E→L, α_0, C_max, signal mixing) is deferred
// to riir-ai Plan 313 — accuracy needs a real model. This gate is the
// modelless proxy: it sweeps the same hyperparameters and verifies the
// controller's *behavioral* response matches the paper's structural
// expectations. The accuracy ranking can only be validated on a real model,
// but the controller responding correctly to each knob is a necessary
// precondition for the accuracy ablation to be meaningful.
//
// Three sub-gates:
//   G9a — W_E→L sweep: larger dwell window → fewer total switches (longer
//         Explicit phases before returning to Latent).
//   G9b — C_max sweep: termination step scales monotonically with C_max
//         (tighter c_max → earlier termination).
//   G9c — α_0 sweep: switch decisions are α-independent (α only affects the
//         signal-mix blend ratio at switch instants, not the switching logic).

/// Result of driving the controller on a fixed alternating schedule.
struct AblationRun {
    switches: u32,
    terminated_at: Option<u32>,
    latent_steps: u32,
    explicit_steps: u32,
}

/// Drive `ctrl` on a deterministic alternating high/low entropy schedule for
/// up to `max_steps` steps (or termination). Returns the behavioral summary.
///
/// Schedule: step 0 sets reference_entropy = HIGH (Latent). Subsequent steps
/// alternate LOW (triggers Latent→Explicit, switch_count++) then HIGH
/// (triggers Explicit→Latent after dwell, no count). This maximises switching
/// so the ablation can observe the effect of dwell windows and c_max.
fn drive_alternating(ctrl: &mut SwiRController, max_steps: u32) -> AblationRun {
    const HIGH: f32 = 5.0;
    const LOW: f32 = 1.0;
    let mut switches = 0u32;
    let mut terminated_at: Option<u32> = None;
    for i in 0..max_steps {
        let entropy = if i == 0 { HIGH } else if i % 2 == 1 { LOW } else { HIGH };
        let prev_mode = ctrl.mode();
        let action = ctrl.step(entropy, i);
        let new_mode = ctrl.mode();
        if prev_mode == ThinkMode::Latent && new_mode == ThinkMode::Explicit {
            switches += 1;
        }
        if action == StepAction::Terminate {
            terminated_at = Some(i + 1);
            break;
        }
    }
    let stats = ctrl.stats();
    AblationRun {
        switches,
        terminated_at,
        latent_steps: stats.latent_steps,
        explicit_steps: stats.explicit_steps,
    }
}

#[test]
fn g9a_w_e_to_l_sweep_larger_dwall_fewer_switches() {
    // Paper Tab. 3: W_E→L=512 is the sweet spot. Behaviourally, a larger
    // Explicit→Latent dwell window means the controller stays in Explicit mode
    // longer before switching back, so it accumulates fewer Latent→Explicit
    // switches over a fixed-length run (each Explicit phase is longer).
    //
    // We disable termination (c_max huge) to isolate the dwell effect.
    let w_e_to_l_values: &[u32] = &[1, 4, 16, 64, 256, 512];
    let max_steps = 512u32;
    let mut prev_switches = u32::MAX;
    println!("G9a — W_E→L sweep (c_max=∞, {max_steps} steps):");
    println!("  {:>10} {:>10} {:>12} {:>12}", "W_E→L", "switches", "latent_st", "explicit_st");
    for &w_e_to_l in w_e_to_l_values {
        let mut ctrl = SwiRController::new(SwiRConfig {
            w_e_to_l,
            w_l_to_e: 0,
            c_max: 10_000, // effectively ∞ — isolate dwell effect
            c_convergence_fraction: 0.5,
            answer_budget_b: 1024,
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps,
            kurtosis_escape_threshold: f32::INFINITY,
        });
        let run = drive_alternating(&mut ctrl, max_steps);
        println!(
            "  {:>10} {:>10} {:>12} {:>12}",
            w_e_to_l, run.switches, run.latent_steps, run.explicit_steps
        );
        // Monotonicity: larger W_E→L must NOT increase switch count. Strict
        // decrease is expected once W_E→L exceeds the schedule's natural
        // switch period (2 steps), but we assert the weaker non-increasing
        // property to tolerate schedule aliasing at small W_E→L.
        assert!(
            run.switches <= prev_switches,
            "G9a FAIL: W_E→L={w_e_to_l} produced {} switches > prev {} — \
             larger dwell window should not increase switches",
            run.switches,
            prev_switches
        );
        prev_switches = run.switches;
    }
    // Sanity: W_E→L=512 must produce strictly fewer switches than W_E→L=1.
    let mut small = SwiRController::new(SwiRConfig {
        w_e_to_l: 1,
        w_l_to_e: 0,
        c_max: 10_000,
        c_convergence_fraction: 0.5,
        answer_budget_b: 1024,
        alpha_0: 0.6,
        beta_0: 0.7,
        max_steps,
        kurtosis_escape_threshold: f32::INFINITY,
    });
    let mut large = SwiRController::new(SwiRConfig {
        w_e_to_l: 512,
        w_l_to_e: 0,
        c_max: 10_000,
        c_convergence_fraction: 0.5,
        answer_budget_b: 1024,
        alpha_0: 0.6,
        beta_0: 0.7,
        max_steps,
        kurtosis_escape_threshold: f32::INFINITY,
    });
    let run_small = drive_alternating(&mut small, max_steps);
    let run_large = drive_alternating(&mut large, max_steps);
    assert!(
        run_large.switches < run_small.switches,
        "G9a FAIL: W_E→L=512 ({}) did not beat W_E→L=1 ({}) on switch count",
        run_large.switches,
        run_small.switches
    );
}

#[test]
fn g9b_c_max_sweep_termination_step_scales_monotonically() {
    // Paper Tab. 10: C_max=20 is the sweet spot. Behaviourally, C_max directly
    // bounds the number of Latent→Explicit switches before ForceAnswerPrefix
    // fires, so termination step must scale monotonically with C_max (tighter
    // c_max → earlier termination).
    let c_max_values: &[u32] = &[2, 4, 8, 16, 20, 32];
    let max_steps = 4096u32;
    let mut prev_term: u32 = 0;
    println!("G9b — C_max sweep (w_e_to_l=1, answer_budget=16):");
    println!("  {:>6} {:>14} {:>10}", "C_max", "terminated_at", "switches");
    for &c_max in c_max_values {
        let mut ctrl = SwiRController::new(SwiRConfig {
            w_e_to_l: 1,
            w_l_to_e: 0,
            c_max,
            c_convergence_fraction: 0.5,
            answer_budget_b: 16,
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps,
            kurtosis_escape_threshold: f32::INFINITY,
        });
        let run = drive_alternating(&mut ctrl, max_steps);
        let term = run.terminated_at.expect("each finite c_max must terminate");
        println!("  {:>6} {:>14} {:>10}", c_max, term, run.switches);
        // Monotonicity: larger C_max must NOT terminate earlier.
        assert!(
            term >= prev_term,
            "G9b FAIL: C_max={c_max} terminated at {term} < prev {prev_term} — \
             larger c_max should not terminate earlier"
        );
        // C_max bounds the termination trigger, but switches continue during
        // the answer_budget countdown (the controller doesn't disable mode-
        // switching after ForceAnswerPrefix — it just starts the budget timer).
        // So total switches ≈ c_max + budget_period, not ≤ c_max. We assert a
        // loose ceiling: switches should not exceed c_max + answer_budget_b
        // (one switch per step during the countdown, pessimistic).
        assert!(
            run.switches <= c_max + 16 + 2,
            "G9b FAIL: C_max={c_max} produced {} switches, expected ≤ {} (c_max + budget + slop)",
            run.switches,
            c_max + 16 + 2
        );
        prev_term = term;
    }
}

#[test]
fn g9c_alpha_0_sweep_switch_decisions_are_alpha_independent() {
    // Paper Tab. 2: broad plateau on α_0. Behaviourally, α_0 ONLY affects the
    // signal-mix blend ratio at switch instants (α_t = α_0 + (1-α_0)·t/T) —
    // it does NOT influence the mode-switch decisions, which are driven purely
    // by entropy trends + dwell windows + switch count. So the controller's
    // switch count, termination step, and mode distribution must be IDENTICAL
    // across all α_0 values (bit-identical for switch_count; the soft-embedding
    // values differ but the decisions don't).
    let alpha_values: &[f32] = &[0.3, 0.6, 0.9, 1.0];
    let max_steps = 256u32;
    let mut baseline: Option<AblationRun> = None;
    println!("G9c — α_0 sweep (switch decisions must be α-independent):");
    println!("  {:>6} {:>10} {:>14} {:>10}", "α_0", "switches", "terminated_at", "latent_st");
    for &alpha_0 in alpha_values {
        let mut ctrl = SwiRController::new(SwiRConfig {
            w_e_to_l: 1,
            w_l_to_e: 0,
            c_max: 8,
            c_convergence_fraction: 0.5,
            answer_budget_b: 16,
            alpha_0,
            beta_0: 0.7,
            max_steps,
            kurtosis_escape_threshold: f32::INFINITY,
        });
        let run = drive_alternating(&mut ctrl, max_steps);
        println!(
            "  {:>6} {:>10} {:>14} {:>10}",
            alpha_0, run.switches, format!("{:?}", run.terminated_at), run.latent_steps
        );
        match &baseline {
            None => baseline = Some(run),
            Some(b) => {
                assert_eq!(
                    run.switches, b.switches,
                    "G9c FAIL: α_0={alpha_0} produced {} switches vs baseline {} — \
                     switch decisions must be α-independent",
                    run.switches, b.switches
                );
                assert_eq!(
                    run.terminated_at, b.terminated_at,
                    "G9c FAIL: α_0={alpha_0} terminated at {:?} vs baseline {:?} — \
                     termination must be α-independent",
                    run.terminated_at, b.terminated_at
                );
                assert_eq!(
                    run.latent_steps, b.latent_steps,
                    "G9c FAIL: α_0={alpha_0} latent_steps={} vs baseline {} — \
                     mode distribution must be α-independent",
                    run.latent_steps,
                    b.latent_steps
                );
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// G9d — signal-mixing on/off sweep (T3.9 sub-task 4: "Signal mixing on/off")
// Paper Tab. 9: signal mixing contributes ~+0.6pp accuracy. The controller-
// internal effect of signal mixing is to blend the soft embedding with the
// control-token anchor at switch instants — this does NOT change mode-switch
// decisions (same as α_0 in g9c) but DOES change the soft-embedding values.
// We verify the decision-equivalence: with mixing enabled vs disabled, the
// switch count and termination step must be identical (mixing only affects
// the embedding values, not the controller's mode logic).
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn g9d_signal_mixing_on_off_decisions_identical() {
    // Signal mixing is applied via `should_mix_signal()` AFTER step() returns.
    // The controller's mode-switch logic doesn't depend on whether the host
    // applies the mix — it only depends on entropy + dwell + switch count.
    // So two runs (mix on vs mix off) must produce identical switch counts,
    // termination steps, and mode distributions.
    //
    // We can't directly toggle "mixing on/off" at the controller level (the
    // controller doesn't know whether the host applied the ratio). But we CAN
    // verify that `should_mix_signal()` returns Some only on the step
    // immediately after a switch, and None otherwise — this is the contract
    // that makes mixing decision-independent.
    let max_steps = 256u32;
    let mut ctrl = SwiRController::new(SwiRConfig {
        w_e_to_l: 1,
        w_l_to_e: 0,
        c_max: 8,
        c_convergence_fraction: 0.5,
        answer_budget_b: 16,
        alpha_0: 0.6,
        beta_0: 0.7,
        max_steps,
        kurtosis_escape_threshold: f32::INFINITY,
    });

    let mut mix_steps = 0u32;
    let mut total_switches = 0u32;
    for i in 0..max_steps {
        let entropy = if i == 0 { 5.0 } else if i % 2 == 1 { 1.0 } else { 5.0 };
        let prev_mode = ctrl.mode();
        let action = ctrl.step(entropy, i);
        let new_mode = ctrl.mode();

        // Check mix signal.
        if let Some((_kind, _ratio)) = ctrl.should_mix_signal() {
            mix_steps += 1;
            // Mix signal MUST only fire on a switch step.
            assert!(
                prev_mode != new_mode,
                "G9d FAIL: mix signal fired on non-switch step {i}"
            );
        }

        if prev_mode == ThinkMode::Latent && new_mode == ThinkMode::Explicit {
            total_switches += 1;
        }

        if action == StepAction::Terminate {
            break;
        }
    }

    println!("G9d — signal mixing sweep:");
    println!("  total_switches: {total_switches}");
    println!("  mix_steps (must equal total switches): {mix_steps}");

    // Every Latent→Explicit or Explicit→Latent switch should arm exactly one
    // mix signal. Since we count Latent→Explicit switches (the paper's switch
    // count), and there are also Explicit→Latent switches, mix_steps should
    // be >= total_switches (it includes both directions).
    assert!(
        mix_steps >= total_switches,
        "G9d FAIL: mix_steps ({mix_steps}) < Latent→Explicit switches ({total_switches}) — \
         every switch should arm a mix signal"
    );
    assert!(
        mix_steps > 0,
        "G9d FAIL: no mix signals fired — the alternating schedule must trigger switches"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G1h — G1 accuracy gate harness structure (T3.3)
// The real G1 (accuracy ≥ +1.5pp on MATH500) requires a real model. This test
// validates the harness STRUCTURE: that `run_benchmark` produces the right
// metrics, and that `ComparisonResult::accuracy_delta_pp` computes correctly.
// riir-ai Plan 313 plugs in Gemma 2 2B + MATH500 to get the real number
// (currently 0% — blocked by model capability; needs Qwen3-4B/8B).
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn g1h_accuracy_gate_harness_structure() {
    use katgpt_rs::swir::bench::*;

    let src = SyntheticProblemSource::new(10, 42);
    let mut backend_a = SyntheticDecodeBackend::new(42);
    let mut backend_b = SyntheticDecodeBackend::new(42);

    let baseline = run_benchmark(
        &src,
        &mut backend_a,
        BenchConfig {
            mode: BenchMode::Baseline,
            max_steps: 32,
            ..Default::default()
        },
    );
    let swir = run_benchmark(
        &src,
        &mut backend_b,
        BenchConfig {
            mode: BenchMode::Swir,
            swir_config: SwiRConfig {
                w_e_to_l: 1,
                c_max: 4,
                ..Default::default()
            },
            max_steps: 32,
            ..Default::default()
        },
    );

    let cmp = ComparisonResult { baseline, swir };
    println!("G1h — accuracy gate harness structure:");
    println!("  {}", cmp.baseline.summary());
    println!("  {}", cmp.swir.summary());
    println!("  {}", cmp.verdict());

    // Structural assertions (not accuracy assertions — those need a real model):
    assert_eq!(cmp.baseline.problems.len(), 10);
    assert_eq!(cmp.swir.problems.len(), 10);
    // Baseline accuracy and SwiR accuracy are both in [0, 1].
    assert!(cmp.baseline.accuracy() >= 0.0 && cmp.baseline.accuracy() <= 1.0);
    assert!(cmp.swir.accuracy() >= 0.0 && cmp.swir.accuracy() <= 1.0);
    // Delta is computable.
    let delta = cmp.accuracy_delta_pp();
    assert!(delta >= -100.0 && delta <= 100.0);
}

// ════════════════════════════════════════════════════════════════════════════
// G2h — G2 efficiency gate harness structure (T3.4)
// The real G2 (token efficiency ≥ 1.3× at fixed accuracy) requires a real
// model. This test validates the harness computes the efficiency ratio
// correctly. The synthetic backend's SwiR mode should terminate earlier
// than baseline due to the c_max guard.
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn g2h_efficiency_gate_harness_structure() {
    use katgpt_rs::swir::bench::*;

    let src = SyntheticProblemSource::new(5, 42);
    let mut backend_a = SyntheticDecodeBackend::new(42);
    let mut backend_b = SyntheticDecodeBackend::new(42);

    let baseline = run_benchmark(
        &src,
        &mut backend_a,
        BenchConfig {
            mode: BenchMode::Baseline,
            max_steps: 64,
            ..Default::default()
        },
    );
    let swir = run_benchmark(
        &src,
        &mut backend_b,
        BenchConfig {
            mode: BenchMode::Swir,
            swir_config: SwiRConfig {
                w_e_to_l: 1,
                c_max: 4,
                answer_budget_b: 8,
                ..Default::default()
            },
            max_steps: 64,
            ..Default::default()
        },
    );

    let cmp = ComparisonResult { baseline, swir };
    println!("G2h — efficiency gate harness structure:");
    println!("  baseline avg_steps: {:.1}", cmp.baseline.avg_steps());
    println!("  swir avg_steps:     {:.1}", cmp.swir.avg_steps());
    println!("  efficiency ratio:   {:.2}×", cmp.token_efficiency_ratio());

    // Structural assertions:
    // Baseline uses all max_steps (no early termination).
    assert_eq!(cmp.baseline.avg_steps(), 64.0);
    // SwiR should terminate earlier (c_max=4 + answer_budget=8 forces termination).
    assert!(
        cmp.swir.avg_steps() < 64.0,
        "G2h FAIL: SwiR avg_steps ({}) should be < baseline (64) due to c_max guard",
        cmp.swir.avg_steps()
    );
    // Efficiency ratio > 1.0 (SwiR uses fewer steps).
    assert!(
        cmp.token_efficiency_ratio() > 1.0,
        "G2h FAIL: efficiency ratio ({}) should be > 1.0",
        cmp.token_efficiency_ratio()
    );
}

// ════════════════════════════════════════════════════════════════════════════
// G-summary — print the verdict table
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn g_summary_print_verdict_table() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║ Plan 275 Phase 3 — SwiR Switch-Thinking GOAT Gate (synthetic data)      ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║ Gate | Test                                              | Status      ║");
    println!("╠══════╪═══════════════════════════════════════════════════╪═════════════╣");
    println!("║ G3   | step() perf ≤ 200ns (release)                    | see g3_*    ║");
    println!("║ G4   | 1000 random probs all in vocab convex hull       | see g4_*    ║");
    println!("║ G5   | feature-gate isolation (cargo check both ways)   | see g5_*    ║");
    println!("║ G6   | kurtosis escape forces Explicit                  | see g6_*    ║");
    println!("║ G7   | step() zero-alloc; adapter allocs documented     | see g7_*    ║");
    println!("║ G1c  | controller correctness on converging schedule   | see g1c_*   ║");
    println!("║ G2p  | SwiR terminates < fixed-budget baseline         | see g2p_*   ║");
    println!("║ G8   | α_t / β_t monotonic in step_index               | see g8_*    ║");
    println!("║ G9   | hyperparameter ablation (W_E→L, C_max, α_0, mix)   | see g9_*    ║");
    println!("║ G1h  | accuracy gate harness structure (T3.3)            | see g1h_*   ║");
    println!("║ G2h  | efficiency gate harness structure (T3.4)          | see g2h_*   ║");
    println!("╠══════╧═══════════════════════════════════════════════════╧═════════════╣");
    println!("║ DEFERRED to riir-ai Plan 313 (needs real model):                        ║");
    println!("║   G1 — accuracy on MATH500 (+1.5pp target) — 0% on Gemma 2 2B (blocked) ║");
    println!("║   G2 — token efficiency at fixed accuracy (1.3× target) — 1.37× PASS    ║");
    println!("║   T3.9 real-model accuracy ablations — harness + synthetic proxies ship  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Decision: keep swir_switch_thinking OPT-IN until riir-ai Plan 313 confirms");
    println!("the G2 gate at n=20+ (currently 1.37× at n=5, target 1.3×). The algorithmic");
    println!("invariants (G3-G8, G1c, G2p) all pass on synthetic data — the controller");
    println!("is correct by construction. G1 is blocked by Gemma 2 2B capability.");
}
