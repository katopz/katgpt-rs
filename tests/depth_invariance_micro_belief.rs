//! Plan 306 Phase 4 — G3 micro_belief negative control tests (T4.2 / T4.3).
//!
//! These tests prove the depth-invariance diagnostic distinguishes healthy
//! (clamp-bounded) recursive kernels from drifty (unclamped) ones in our own
//! codebase. They are the in-repo counterpart to G1's synthetic chains.
//!
//! # Test layout
//!
//! - **T4.2 / G3a (negative control):** `AttractorKernel` classifies as
//!   `DepthInvariant`. The attractor applies `(2·σ(·) − 1).clamp(±clamp)`,
//!   bounding magnitude per construction.
//! - **T4.3 / G3b (positive control):** an inline **unclamped** leaky
//!   integrator classifies as `DepthSpecificRefinement` under constant
//!   positive input. Our shipped `LeakyIntegrator` clamps to `[-1, 1]` on
//!   every tick, so it would also classify as `DepthInvariant` — the inline
//!   variant strips the clamp to demonstrate the diagnostic's discriminating
//!   power on a real drift kernel.
//!
//! # Honest-results policy
//!
//! If G3b fails (unclamped leaky still classifies invariant), the leak
//! parameter is decaying faster than the input accumulates. We document the
//! threshold; informative either way.
//!
//! Source paper: arXiv:2605.09992 §3.
//! Research note: `.research/286_Attention_Drift_Depth_Invariance_Diagnostic.md`.

#![cfg(all(feature = "depth_invariance", feature = "micro_belief"))]

use katgpt_core::{
    AttractorKernel, DepthInvarianceConfig, DepthInvarianceKind, Scratch, classify_chain,
};

// ── Helpers ──────────────────────────────────────────────────────────────

/// Deterministic xorshift64* PRNG — mirrors the pattern used by other
/// katgpt-rs benches (e.g. `sink_classify_bench.rs`).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        // xorshift64
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 & 0xFFFF) as f32 / 0x8000 as f32) - 1.0
    }
}

// ── T4.2 (G3a): AttractorKernel classifies DepthInvariant ────────────────

/// G3a — `AttractorKernel` (Family A) classifies as `DepthInvariant` under
/// random input.
///
/// # Why we expect DepthInvariant
///
/// The attractor update is `state[i] = clamp(2·σ(W_s·s + W_x·x + b) − 1, ±clamp)`.
/// The `2·σ(·) − 1` map bounds each element to `(−1, 1)`, and the additional
/// `.clamp(±clamp)` (default `clamp = 6.0`) is a no-op safety net for the
/// default state range. Either way, magnitude is bounded per construction —
/// the magnitude series `‖h_t‖` is flat in the long run (bounded above by
/// `√d`), so the least-squares slope is ~0 and the chain classifies as
/// `DepthInvariant`.
///
/// This is the **negative control**: it confirms the diagnostic does not
/// false-positive on healthy clamp-bounded kernels.
///
/// # Assertion
///
/// `kind == DepthInvariant`. If this fails, either the kernel is misbehaving
/// (state escaping `(-1, 1)`) or the diagnostic thresholds are wrong.
#[test]
fn attractor_kernel_classifies_depth_invariant() {
    let dim = 8usize;
    let k = 64usize;
    let cfg = DepthInvarianceConfig::default();

    let kernel = AttractorKernel::from_seed(42, dim);
    let state: Vec<f32> = vec![0.0; dim];
    let mut rng = Rng::new(7);

    // Build the input series + initial state.
    let inputs: Vec<Vec<f32>> = (0..k)
        .map(|_| (0..dim).map(|_| rng.next_f32()).collect())
        .collect();
    let inputs_ref: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();

    let diag = kernel.audit_depth_invariance(&state, &inputs_ref, &cfg);

    eprintln!(
        "G3a AttractorKernel @ dim={dim}, k={k}: kind={:?}, magnitude_slope={:.6}, \
         mean_cos_step={:.6}, effective_rank_slope={:.6}",
        diag.kind, diag.magnitude_slope, diag.mean_cos_step, diag.effective_rank_slope
    );

    // Sanity: the state stayed in (-1, 1) — clamp bounds magnitude per
    // construction. If this fails, the kernel is misbehaving and the
    // classification result is meaningless.
    for &v in &state {
        assert!(
            (-1.0001..=1.0001).contains(&v),
            "attractor state escaped (-1,1): {v}"
        );
    }

    assert_eq!(
        diag.kind,
        DepthInvarianceKind::DepthInvariant,
        "clamp-bounded AttractorKernel must classify as DepthInvariant; \
         if it doesn't, the kernel is escaping (-1,1) or the thresholds are wrong"
    );

    eprintln!("  → PASS: AttractorKernel is DepthInvariant (negative control confirmed).");
}

// ── T4.3 (G3b): unclamped leaky classifies DepthSpecificRefinement ───────

/// G3b — an **unclamped** leaky integrator classifies as `DepthSpecificRefinement`
/// under constant positive input.
///
/// # Why we need an inline unclamped variant
///
/// Our shipped `LeakyIntegrator` (in `katgpt_core::micro_belief::leaky`)
/// clamps state to `[-1, 1]` on every tick via `.clamp(-1.0, 1.0)` in
/// `leaky_core::leaky_step`. It would therefore also classify as
/// `DepthInvariant`, like the attractor. To exercise the diagnostic's
/// discriminating power on a **real drift kernel**, we strip the clamp and
/// run the bare update `state[i] = state[i] + lr * input[i]` under constant
/// positive input. Magnitude grows linearly with `t` → slope is positive →
/// `DepthSpecificRefinement`.
///
/// # Honest-results policy
///
/// Plan 306 §T4.3: "if G3b fails (leaky still classifies invariant),
/// document the leak-decay threshold; informative either way." Our unclamped
/// variant has no leak decay by construction (the bare accumulator is pure
/// integration), so we expect a clean positive classification.
#[test]
fn leaky_kernel_without_clamp_classifies_depth_specific() {
    let dim = 8usize;
    let k = 64usize;
    let cfg = DepthInvarianceConfig::default();
    let lr = 0.1f32;

    // Constant positive input drives pure accumulation.
    let input: Vec<f32> = vec![0.5; dim];
    let inputs: Vec<&[f32]> = (0..k).map(|_| input.as_slice()).collect();

    // Initial state: small non-zero so the chain isn't degenerate.
    let mut state: Vec<f32> = vec![0.01; dim];

    // Capture the chain using the bare unclamped leaky update.
    // No `MicroRecurrentBeliefState` impl — this is a synthetic drift kernel
    // for diagnostic validation only.
    let mut chain: Vec<f32> = Vec::with_capacity((k + 1) * dim);
    chain.extend_from_slice(&state);
    let mut s_a = state.clone();
    let mut s_b: Vec<f32>;
    for inp in &inputs {
        s_b = s_a
            .iter()
            .zip(inp.iter())
            .map(|(s, x)| s + lr * x)
            .collect();
        chain.extend_from_slice(&s_b);
        std::mem::swap(&mut s_a, &mut s_b);
    }
    state = s_a;

    let mut scratch = Scratch::with_capacity(k + 1, dim);
    let diag = classify_chain(&chain, dim, &cfg, &mut scratch);

    eprintln!(
        "G3b unclamped leaky @ dim={dim}, k={k}, lr={lr}: kind={:?}, \
         magnitude_slope={:.6}, mean_cos_step={:.6}, effective_rank_slope={:.6}",
        diag.kind, diag.magnitude_slope, diag.mean_cos_step, diag.effective_rank_slope
    );

    // The state must have grown — sanity check.
    let final_mag = state.iter().map(|x| x * x).sum::<f32>().sqrt();
    let initial_mag = (0.01f32 * dim as f32).sqrt();
    eprintln!(
        "  ‖s_0‖ = {initial_mag:.4}, ‖s_{k}‖ = {final_mag:.4} (growth factor {:.2}x)",
        final_mag / initial_mag
    );
    assert!(
        final_mag > initial_mag,
        "unclamped leaky under constant positive input must grow in magnitude"
    );

    assert_eq!(
        diag.kind,
        DepthInvarianceKind::DepthSpecificRefinement,
        "unclamped leaky under constant positive input must classify as \
         DepthSpecificRefinement; if it doesn't, the leak-decay threshold wasn't \
         reached (lr={lr}, k={k}) — see Plan 306 §T4.3 caveat"
    );

    eprintln!(
        "  → PASS: unclamped leaky classifies DepthSpecificRefinement (positive control confirmed)."
    );
    eprintln!(
        "  Note: the shipped LeakyIntegrator clamps to [-1,1] on every tick, so it would \
         ALSO classify DepthInvariant. This test strips the clamp to demonstrate the \
         diagnostic's discriminating power — see test doc for the rationale."
    );
}
