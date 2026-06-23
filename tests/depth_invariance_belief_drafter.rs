//! Plan 306 Phase 3 — G2 BeliefDrafter audit tests (T3.2 / T3.3 / T3.4).
//!
//! Reproduce the central empirical finding of arXiv:2605.09992 (Eldenk et al.,
//! *Attention Drift*) on our own `BeliefDrafter`: pre-norm EAGLE-3-style
//! drafters classify as `DepthSpecificRefinement` beyond their TTT horizon
//! because the unnormalized residual `h_{t+1} = h_t + FC3(...)` accumulates
//! magnitude monotonically.
//!
//! # Honest-results policy
//!
//! Plan 306 explicitly says G2 outcomes are "informative either way" —
//! random-init weights may differ from trained. Each test below documents
//! the expected outcome AND the failure-mode we'd report if the diagnostic
//! disagrees. **Do not fudge these tests to pass** — a documented negative
//! result is more valuable than a fake pass.
//!
//! Source paper: arXiv:2605.09992 — §3 (depth-invariance finding),
//! Table 1 (Llama 3.1 8B magnitude series 3.92 → 4.87 → 5.86 → 14.02),
//! §4.4 Table 4 (-56% acceptance for inference-time pin on pre-norm).
//!
//! Research note: `.research/286_Attention_Drift_Depth_Invariance_Diagnostic.md`.

#![cfg(all(feature = "depth_invariance", feature = "belief_drafter"))]

use katgpt_core::{
    DepthInvarianceConfig, DepthInvarianceKind, MagnitudeRegularization, Scratch,
    apply_magnitude_regularization, classify_chain,
};
use katgpt_rs::speculative::belief_drafter::{BeliefDrafter, LatentDynamicsMLP};

// ── Helpers ──────────────────────────────────────────────────────────────

/// Fixed-seed verifier-style starting hidden state.
///
/// `n_embd` values drawn from a small LCG so the test is deterministic. The
/// exact distribution isn't important — what matters is that `‖h_0‖` is
/// non-trivial so the magnitude slope has signal to fit.
fn make_h0(n_embd: usize) -> Vec<f32> {
    let mut state: u32 = 0xC0FFEE;
    (0..n_embd)
        .map(|_| {
            state = state.wrapping_mul(1_106_351_524).wrapping_add(12_345);
            let bits = state >> 1;
            (bits as f32) / (u32::MAX as f32 * 0.5) - 1.0
        })
        .collect()
}

/// Build a random-init BeliefDrafter (no trained weights available in-tree).
///
/// Plan 306 §T3.2 caveat: random init may differ from trained. We seed the
/// MLP via `LatentDynamicsMLP::random_init`'s built-in LCG (deterministic).
fn make_drafter(n_embd: usize, vocab_size: usize) -> BeliefDrafter {
    let mlp = LatentDynamicsMLP::random_init(n_embd);
    // Dummy output_head + wte: we never read drafted tokens in these tests,
    // only the hidden-state chain. Zero weights → all-zero embeddings → still
    // a valid `forward_into` input.
    let output_head = vec![0.0f32; vocab_size * n_embd];
    let wte = vec![0.0f32; vocab_size * n_embd];
    BeliefDrafter::new(mlp, output_head, wte).expect("valid drafter construction")
}

/// Token sequence — uses modulo to stay in-vocab. With all-zero `wte` every
/// token maps to the same (zero) embedding, so the drafter is driven purely
/// by the structural `h_t + FC3(GELU(...))` recursion — the cleanest test of
/// the paper's depth-specific refinement mechanism.
fn token_seq(vocab_size: usize, k: usize) -> Vec<usize> {
    (0..k).map(|i| i % vocab_size.max(1)).collect()
}

/// Compute the magnitude series `‖h_t‖` for `t ∈ [0, k]` from a flattened
/// chain. Used by T3.3 to assert monotonicity without re-running the audit.
fn magnitude_series(chain: &[f32], n: usize) -> Vec<f32> {
    let k_plus_1 = chain.len() / n;
    (0..k_plus_1)
        .map(|t| {
            let h_t = &chain[t * n..(t + 1) * n];
            h_t.iter().map(|x| x * x).sum::<f32>().sqrt()
        })
        .collect()
}

// ── T3.2 (G2a): BeliefDrafter classifies as DepthSpecificRefinement? ──────

/// G2a — does a random-init `BeliefDrafter` classify as `DepthSpecificRefinement`
/// beyond its TTT horizon?
///
/// # Expected (paper §3)
///
/// Pre-norm EAGLE-3 drafters do — the unnormalized residual accumulates
/// magnitude monotonically with depth. Our `LatentDynamicsMLP` has the same
/// structural shape (input LayerNorm + unnormalized residual write).
///
/// # Random-init caveat (Plan 306 §T3.2)
///
/// **Random init may differ from trained.** Xavier init for FC3 scales weights
/// by `sqrt(2 / fan_in)` — at `n_embd = 64`, fan_in is 64, so FC3 weights are
/// ~0.18. Multiplied by the GELU-bounded FC2 output (≤ ~0.84 element-wise),
/// FC3 output magnitudes are small, and `h_t + small_residual` may saturate
/// slowly enough that 16 steps isn't enough to clear the magnitude-slope drift
/// threshold (`cfg.magnitude_slope_drift = 0.05`).
///
/// If this test classifies as `DepthInvariant`, that is informative either way:
/// it documents a regime where the structural drift mechanism doesn't trigger
/// within the audit horizon. The plan does NOT require us to fudge the test —
/// we report the honest classification.
///
/// The assertion below is intentionally a `println!` + assert on
/// `kind != Insufficient` (we have enough samples) — the specific kind is
/// reported, not asserted. The companion bench `depth_invariance_bench.rs`
/// repeats this at deeper `k` and larger `n_embd` to characterize the regime.
#[test]
fn belief_drafter_classifies_depth_specific_beyond_ttt() {
    let n_embd = 64usize;
    let vocab_size = 32usize;
    let k = 16usize;
    let cfg = DepthInvarianceConfig::default();

    let drafter = make_drafter(n_embd, vocab_size);
    let h_0 = make_h0(n_embd);
    let tokens = token_seq(vocab_size, k);

    let diag = drafter.audit_depth_invariance(&h_0, &tokens, k, &cfg);

    eprintln!(
        "G2a BeliefDrafter @ n_embd={n_embd}, k={k}: kind={:?}, magnitude_slope={:.6}, \
         mean_cos_step={:.6}, effective_rank_slope={:.6}",
        diag.kind, diag.magnitude_slope, diag.mean_cos_step, diag.effective_rank_slope
    );

    // We must have enough samples (k+1 = 17 ≥ min_samples = 4).
    assert_ne!(
        diag.kind,
        DepthInvarianceKind::Insufficient,
        "audit must gather enough samples"
    );

    // Paper expectation is DepthSpecificRefinement. We log the kind but do NOT
    // hard-assert it — random init may show DepthInvariant if Xavier bounds
    // FC3 output enough that 16 steps don't clear the slope threshold. See the
    // test doc above for the full caveat. The honest verdict is reported.
    match diag.kind {
        DepthInvarianceKind::DepthSpecificRefinement => {
            eprintln!("  → PASS: paper finding reproduced on random-init BeliefDrafter.");
            eprintln!(
                "  locked-drift sub-case (mean_cos_step > {:.2})? {}",
                cfg.cos_step_drift_lock,
                diag.mean_cos_step > cfg.cos_step_drift_lock
            );
        }
        DepthInvarianceKind::DepthInvariant => {
            eprintln!(
                "  → INFORMATIVE-NEGATIVE: random-init BeliefDrafter classified DepthInvariant."
            );
            eprintln!(
                "  Likely cause: Xavier init scales FC3 weights by sqrt(2/{n_embd}) ≈ {:.4}, so \
                 FC3 output magnitudes are small and the magnitude slope ({:.6}) doesn't clear \
                 cfg.magnitude_slope_drift ({}) within k={k} steps.",
                (2.0 / n_embd as f32).sqrt(),
                diag.magnitude_slope,
                cfg.magnitude_slope_drift
            );
            eprintln!(
                "  Re-run with trained nextlat.bin weights (T3.2 fallback) or larger k to \
                 reproduce the paper's drift signature."
            );
        }
        DepthInvarianceKind::Collapsed => {
            eprintln!(
                "  → INFORMATIVE: random-init produced Collapsed (rank collapse dominant). \
                 magnitude_slope={:.6}, effective_rank_slope={:.6}",
                diag.magnitude_slope, diag.effective_rank_slope
            );
        }
        DepthInvarianceKind::Insufficient => unreachable!("checked above"),
    }
}

// ── T3.3 (G2b): magnitude series monotonic non-decreasing for k > 1 ───────

/// G2b — the magnitude series `‖h_t‖` should be monotonic non-decreasing
/// for `t > 1` if the paper's mechanism is in effect.
///
/// # Paper reference (Table 1, Llama 3.1 8B)
///
/// `‖h_t‖`: 3.92 → 4.87 → 5.86 → 14.02 (monotonic non-decreasing).
///
/// # Assertion
///
/// We assert the **trend** (monotonic non-decreasing), not the paper's specific
/// numbers — random-init absolute values differ. Equality (within `1e-6`) is
/// allowed; only strict decreases count as violations.
///
/// # Honest-results policy
///
/// If the series is NOT monotonic, this is informative either way (random init
/// may produce cancellations). The plan asks us to document, not fudge.
#[test]
fn belief_drafter_magnitude_series_monotonic() {
    let n_embd = 64usize;
    let vocab_size = 32usize;
    let k = 16usize;
    let cfg = DepthInvarianceConfig::default();

    let drafter = make_drafter(n_embd, vocab_size);
    let h_0 = make_h0(n_embd);
    let tokens = token_seq(vocab_size, k);

    // Capture the chain via the public `capture_chain` API (no regularization)
    // so we can extract the magnitude series directly without re-running the
    // audit's classify pass.
    let chain =
        drafter.capture_chain(&h_0, &tokens, k, MagnitudeRegularization::None);
    let n = n_embd;

    let mags = magnitude_series(&chain, n);

    eprintln!("G2b magnitude series (k+1 = {}):", mags.len());
    for (t, m) in mags.iter().enumerate() {
        eprintln!("  ‖h_{t}‖ = {m:.6}");
    }

    // Monotonic non-decreasing for t > 0. Equality (1e-6 tolerance) is OK.
    let mut violations = 0usize;
    for t in 1..mags.len() {
        let delta = mags[t] - mags[t - 1];
        if delta < -1e-6 {
            violations += 1;
        }
    }

    eprintln!(
        "  monotonic-violations = {violations} (out of {} steps)",
        mags.len() - 1
    );

    // The paper's mechanism predicts monotonic non-decreasing. If violations
    // > 0 we report honestly — random init may produce local cancellations
    // even when the long-run trend is growth.
    if violations > 0 {
        eprintln!(
            "  → INFORMATIVE-NEGATIVE: magnitude series is NOT strictly monotonic ({violations} \
             violations). Random-init weights can produce local cancellations; the long-run \
             trend is captured by `audit_depth_invariance`'s least-squares slope instead. \
             Re-run with trained weights for paper-fidelity."
        );
    }
    assert_eq!(
        violations, 0,
        "magnitude series must be monotonic non-decreasing for paper mechanism to hold; \
         see log for the violation pattern"
    );

    // Cross-check: a non-trivial slope is the paper's primary signal.
    let mut scratch = Scratch::with_capacity(k + 1, n);
    let diag = classify_chain(&chain, n, &cfg, &mut scratch);
    eprintln!(
        "  cross-check classify_chain: kind={:?}, magnitude_slope={:.6}",
        diag.kind, diag.magnitude_slope
    );
}

// ── T3.4 (G2c): RmsNorm post-hoc flips classification — diagnostic only ──

/// G2c — apply `MagnitudeRegularization::RmsNorm` post-hoc to the drafter's
/// output, re-run the audit, expect `DepthInvariant`.
///
/// # Diagnostic intent (NOT a shipped feature)
///
/// This is the inference-time pin the paper §4.4 Table 4 reports **drops
/// acceptance by 56% on pre-norm models**. We demonstrate it classifies the
/// chain as `DepthInvariant` — proving the regularization *does* kill the
/// magnitude accumulation — but the fix is **diagnostic-only** here. The
/// shipped fix requires retraining the MLP with the regularization baked in
/// (→ riir-train).
///
/// # Assertion
///
/// `DepthInvariant` is the only acceptable outcome after RmsNorm — by
/// construction `rmsnorm(h)` has RMS = 1 for every `h`, so the magnitude
/// series is exactly flat. If this fails, something is wrong with the
/// regularization primitive (Phase 5).
#[test]
fn belief_drafter_rmsnorm_post_hoc_is_depth_invariant() {
    let n_embd = 64usize;
    let vocab_size = 32usize;
    let k = 16usize;
    let cfg = DepthInvarianceConfig::default();

    let drafter = make_drafter(n_embd, vocab_size);
    let h_0 = make_h0(n_embd);
    let tokens = token_seq(vocab_size, k);

    // Baseline (no regularization) — for comparison logging.
    let diag_baseline = drafter.audit_depth_invariance(&h_0, &tokens, k, &cfg);
    eprintln!(
        "G2c baseline (no reg): kind={:?}, magnitude_slope={:.6}",
        diag_baseline.kind, diag_baseline.magnitude_slope
    );

    // Re-capture with RmsNorm applied at each step — the public capture_chain
    // API handles the interleaving.
    //
    // NOTE: we also RmsNorm `h_0` itself before capture — the paper's
    // prescription applies RmsNorm to the residual stream uniformly, not
    // just to the deltas. If we leave `h_0` unnormalized, the first step
    // from `h_0` (random-magnitude) to `h_1` (RMS=1) creates a one-time
    // magnitude jump that the least-squares slope picks up, falsely
    // classifying the chain as DepthSpecificRefinement. With `h_0` also
    // regularized, the magnitude series is exactly flat (every `‖h_t‖ = √d`)
    // and the chain classifies as DepthInvariant — the paper's intended effect.
    let n = n_embd;
    let mut h_0_reg = h_0.clone();
    let mut reg_init_scratch = vec![0.0f32; n];
    apply_magnitude_regularization(
        &mut h_0_reg,
        MagnitudeRegularization::RmsNorm,
        &mut reg_init_scratch,
    );
    let chain = drafter.capture_chain(&h_0_reg, &tokens, k, MagnitudeRegularization::RmsNorm);

    let mut scratch = Scratch::with_capacity(k + 1, n);
    let diag_reg = classify_chain(&chain, n, &cfg, &mut scratch);
    eprintln!(
        "G2c with RmsNorm post-hoc: kind={:?}, magnitude_slope={:.6}, mean_cos_step={:.6}",
        diag_reg.kind, diag_reg.magnitude_slope, diag_reg.mean_cos_step
    );

    // RmsNorm bounds RMS(h) = 1 by construction → ‖h_t‖ = sqrt(d) exactly,
    // magnitude_slope = 0 → DepthInvariant (unless rank collapses, which it
    // can't for a stable RmsNorm).
    assert_eq!(
        diag_reg.kind,
        DepthInvarianceKind::DepthInvariant,
        "RmsNorm post-hoc MUST classify as DepthInvariant (RMS=1 by construction); \
         if this fails, the regularization primitive is broken"
    );

    eprintln!(
        "  → PASS: RmsNorm pin classifies DepthInvariant. Note (paper §4.4 Table 4): \
         inference-time pin drops acceptance -56% on pre-norm models. The fix \
         requires MLP retraining → riir-train; this test is diagnostic-only."
    );
}
