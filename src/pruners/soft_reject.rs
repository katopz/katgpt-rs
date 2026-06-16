//! SoftReject — Sigmoid-Graded Relax-and-Retry Caller (Plan 310 T1.4).
//!
//! Distillation of HarnessBridge Table 7: tolerant rejection strictly beats
//! strict rejection because **false-reject cost > false-pass cost**. This module
//! implements the opt-in soft-reject caller path that consumes a pruner's
//! [`reject_confidence`](katgpt_core::traits::ConstraintPruner::reject_confidence)
//! and routes borderline candidates through a relaxed retry pass instead of
//! hard-failing them outright.
//!
//! # Decision Rule
//!
//! Given a candidate's `reject_confidence ∈ [0.0, 1.0]` and thresholds
//! `τ_low < τ_high` (defaults 0.4 / 0.8):
//!
//! | Confidence range | Verdict | Action |
//! |------------------|---------|--------|
//! | `≤ τ_low`        | `Accept`| take the candidate (as today) |
//! | `τ_low < c < τ_high` | `SoftReject` | retry against a relaxed constraint set; hard-reject only if the relaxed pass also fails |
//! | `≥ τ_high`       | `Reject`| hard-fail (identical to today's `is_valid() == false`) |
//!
//! # Sigmoid Discipline (AGENTS.md)
//!
//! The *caller* does not impose sigmoid on the pruner — the pruner's
//! `reject_confidence` impl is responsible for using `sigmoid(β · evidence)`
//! (never softmax). This helper is purely threshold + retry plumbing. A pruner
//! using the default binary `reject_confidence` will only ever emit `Accept` /
//! `Reject` (the `SoftReject` band is unreachable because the default only
//! produces 0.0 / 1.0). This is intentional: opting into soft reject is a
//! two-step choice (graded pruner + feature flag).
//!
//! # Zero-Allocation Hot Path
//!
//! The caller passes a pre-allocated scratch buffer for the retry pass. The
//! retry loop never grows a `Vec`; it reuses the caller-owned scratch slice.

use katgpt_core::traits::ConstraintPruner;

/// Verdict returned by [`soft_reject_decide`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftRejectVerdict {
    /// `reject_confidence ≤ τ_low` — take the candidate (as today).
    Accept,
    /// `τ_low < reject_confidence < τ_high` — caller should retry against a
    /// relaxed constraint set; hard-reject only if the relaxed pass also fails.
    SoftReject,
    /// `reject_confidence ≥ τ_high` — hard-fail (identical to `is_valid() == false`).
    Reject,
}

/// Configuration for the soft-reject caller.
///
/// Threshold defaults follow the Plan 310 T1.4 spec (τ_low = 0.4, τ_high = 0.8).
/// The band `[τ_low, τ_high)` is where relaxation kicks in.
#[derive(Debug, Clone, Copy)]
pub struct SoftRejectConfig {
    /// Below this confidence the candidate is accepted outright.
    pub tau_low: f32,
    /// At or above this confidence the candidate is hard-rejected.
    pub tau_high: f32,
}

impl Default for SoftRejectConfig {
    #[inline]
    fn default() -> Self {
        Self {
            tau_low: 0.4,
            tau_high: 0.8,
        }
    }
}

impl SoftRejectConfig {
    /// Construct a config, clamping `tau_low < tau_high` and both to `[0, 1]`.
    pub fn new(tau_low: f32, tau_high: f32) -> Self {
        let lo = tau_low.clamp(0.0, 1.0);
        let hi = tau_high.clamp(0.0, 1.0);
        // Preserve the invariant tau_low < tau_high; if invalid, fall back to defaults.
        if lo < hi {
            Self {
                tau_low: lo,
                tau_high: hi,
            }
        } else {
            Self::default()
        }
    }
}

/// Decide how to handle a candidate given its reject confidence (Plan 310 T1.4).
///
/// This is the pure threshold decision — no retry side-effect. Callers compose
/// it with their own relaxation logic via [`soft_reject_with_relax`].
#[inline]
pub fn soft_reject_decide(reject_confidence: f32, cfg: &SoftRejectConfig) -> SoftRejectVerdict {
    // Clamp defensively; the trait contract says [0,1] but a misbehaving impl
    // could violate it. NaN compares false to everything → falls to Reject
    // (safest — a NaN confidence should not silently accept).
    let c = if reject_confidence.is_nan() {
        1.0
    } else {
        reject_confidence.clamp(0.0, 1.0)
    };
    if c <= cfg.tau_low {
        SoftRejectVerdict::Accept
    } else if c >= cfg.tau_high {
        SoftRejectVerdict::Reject
    } else {
        SoftRejectVerdict::SoftReject
    }
}

/// Trait abstracting the "relax-and-retry" step.
///
/// When a candidate lands in the `SoftReject` band, the caller hands control
/// to a `RelaxationStrategy` which decides whether the candidate survives under
/// a relaxed constraint set (e.g. widen tolerance, drop the lowest-weight
/// predicate). This decouples the threshold plumbing from any specific
/// relaxation recipe — different pruners can plug in their own.
///
/// # Zero-allocation contract
///
/// Implementations MUST NOT allocate inside [`retry`]. Use the `scratch` slice
/// for any temporary state — it is owned by the caller and reused across calls.
pub trait RelaxationStrategy {
    /// Re-evaluate `token_idx` at `depth` against a *relaxed* constraint set.
    ///
    /// Returns `true` if the candidate survives relaxation (becomes accepted),
    /// `false` if it still fails (escalates to hard-reject).
    ///
    /// `scratch` is a caller-owned scratch buffer for intermediate work;
    /// implementations may `clear()` and reuse it but must not grow it.
    fn retry(
        &mut self,
        depth: usize,
        token_idx: usize,
        parent_tokens: &[usize],
        scratch: &mut [u8],
    ) -> bool;
}

/// No-op relaxation: never relaxes. The `SoftReject` band escalates directly
/// to hard-reject. Useful as a baseline (equivalent to today's binary behavior
/// even when a graded pruner produces mid-band confidences).
pub struct NoRelaxation;

impl RelaxationStrategy for NoRelaxation {
    #[inline]
    fn retry(
        &mut self,
        _depth: usize,
        _token_idx: usize,
        _parent_tokens: &[usize],
        _scratch: &mut [u8],
    ) -> bool {
        false
    }
}

/// Full soft-reject pipeline for a single candidate (Plan 310 T1.4).
///
/// 1. Query `pruner.reject_confidence(depth, token_idx, parent_tokens)`.
/// 2. Apply the threshold decision ([`soft_reject_decide`]).
/// 3. If `SoftReject`, hand off to `relaxer.retry(...)` using the caller's
///    `scratch` buffer. If the relaxer accepts, the candidate is accepted; if
///    it still rejects, the candidate is hard-rejected.
///
/// Returns `true` if the candidate is ultimately accepted (either outright or
/// after relaxation), `false` if hard-rejected. This mirrors the return shape
/// of [`ConstraintPruner::is_valid`] so callers can swap one for the other.
///
/// # Zero-allocation
///
/// No allocations occur inside this function. The `scratch` buffer is owned by
/// the caller and passed through to the `RelaxationStrategy` verbatim.
#[inline]
pub fn soft_reject_with_relax<P: ConstraintPruner, R: RelaxationStrategy>(
    pruner: &P,
    relaxer: &mut R,
    cfg: &SoftRejectConfig,
    depth: usize,
    token_idx: usize,
    parent_tokens: &[usize],
    scratch: &mut [u8],
) -> bool {
    let conf = pruner.reject_confidence(depth, token_idx, parent_tokens);
    match soft_reject_decide(conf, cfg) {
        SoftRejectVerdict::Accept => true,
        SoftRejectVerdict::Reject => false,
        SoftRejectVerdict::SoftReject => {
            // Relax-and-retry; escalation to hard-reject if the relaxed pass also fails.
            relaxer.retry(depth, token_idx, parent_tokens, scratch)
        }
    }
}

/// Batched soft-reject pipeline (Plan 310 T1.4 zero-alloc batch path).
///
/// For each `candidates[i]`, runs the full [`soft_reject_with_relax`] pipeline
/// and writes the final accept/reject bit into `results[i]`. Uses
/// [`ConstraintPruner::batch_reject_confidence`] to amortize any setup cost
/// in the pruner, then dispatches relaxations one-by-one (relaxation is
/// inherently stateful per-candidate).
///
/// # Zero-allocation
///
/// Reuses the caller-supplied `conf_scratch` for the confidence values and
/// `byte_scratch` for the relaxation step. No `Vec` growth.
pub fn batch_soft_reject_with_relax<P: ConstraintPruner, R: RelaxationStrategy>(
    pruner: &P,
    relaxer: &mut R,
    cfg: &SoftRejectConfig,
    depth: usize,
    candidates: &[usize],
    parent_tokens: &[usize],
    conf_scratch: &mut [f32],
    byte_scratch: &mut [u8],
    results: &mut [bool],
) {
    let len = candidates.len().min(results.len()).min(conf_scratch.len());
    // Phase 1: batched confidence query (amortizes pruner setup).
    pruner.batch_reject_confidence(depth, &candidates[..len], parent_tokens, &mut conf_scratch[..len]);
    // Phase 2: per-candidate threshold + relaxation dispatch.
    for i in 0..len {
        results[i] = match soft_reject_decide(conf_scratch[i], cfg) {
            SoftRejectVerdict::Accept => true,
            SoftRejectVerdict::Reject => false,
            SoftRejectVerdict::SoftReject => {
                relaxer.retry(depth, candidates[i], parent_tokens, byte_scratch)
            }
        };
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock binary pruner — produces only 0.0 / 1.0 confidence (default impl).
    struct BinaryThresholdPruner {
        threshold: usize,
    }

    impl ConstraintPruner for BinaryThresholdPruner {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx < self.threshold
        }
    }

    /// Mock *graded* pruner: emits a synthetic sigmoid-shaped confidence
    /// increasing in `token_idx`, so we can exercise the SoftReject band.
    /// Confidence = sigmoid(β · (token_idx - center)) where center sits between
    /// valid and invalid regions.
    struct GradedThresholdPruner {
        center: f32,
        beta: f32,
    }

    impl ConstraintPruner for GradedThresholdPruner {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            // Hard boundary at `center` (integers compare cleanly).
            (token_idx as f32) < self.center
        }

        fn reject_confidence(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            // sigmoid(β · (token_idx - center)). Below center → <0.5 → low reject.
            // Above center → >0.5 → high reject. At center → exactly 0.5.
            let x = self.beta * ((token_idx as f32) - self.center);
            1.0 / (1.0 + (-x).exp())
        }
    }

    /// Relaxation strategy that always accepts (simulates a permissive retry).
    struct AlwaysAcceptRelax;

    impl RelaxationStrategy for AlwaysAcceptRelax {
        fn retry(
            &mut self,
            _depth: usize,
            _token_idx: usize,
            _parent_tokens: &[usize],
            _scratch: &mut [u8],
        ) -> bool {
            true
        }
    }

    // -- Decision logic --

    #[test]
    fn decide_accept_below_tau_low() {
        let cfg = SoftRejectConfig::default();
        assert_eq!(soft_reject_decide(0.0, &cfg), SoftRejectVerdict::Accept);
        assert_eq!(soft_reject_decide(0.39, &cfg), SoftRejectVerdict::Accept);
        assert_eq!(soft_reject_decide(0.4, &cfg), SoftRejectVerdict::Accept); // boundary inclusive
    }

    #[test]
    fn decide_soft_reject_in_band() {
        let cfg = SoftRejectConfig::default();
        assert_eq!(soft_reject_decide(0.41, &cfg), SoftRejectVerdict::SoftReject);
        assert_eq!(soft_reject_decide(0.5, &cfg), SoftRejectVerdict::SoftReject);
        assert_eq!(soft_reject_decide(0.79, &cfg), SoftRejectVerdict::SoftReject);
    }

    #[test]
    fn decide_reject_at_or_above_tau_high() {
        let cfg = SoftRejectConfig::default();
        assert_eq!(soft_reject_decide(0.8, &cfg), SoftRejectVerdict::Reject); // boundary inclusive
        assert_eq!(soft_reject_decide(0.95, &cfg), SoftRejectVerdict::Reject);
        assert_eq!(soft_reject_decide(1.0, &cfg), SoftRejectVerdict::Reject);
    }

    #[test]
    fn decide_nan_falls_to_reject_safely() {
        // NaN must not silently accept — pick the safe (reject) side.
        let cfg = SoftRejectConfig::default();
        assert_eq!(soft_reject_decide(f32::NAN, &cfg), SoftRejectVerdict::Reject);
    }

    #[test]
    fn decide_out_of_range_clamps() {
        let cfg = SoftRejectConfig::default();
        assert_eq!(soft_reject_decide(-1.0, &cfg), SoftRejectVerdict::Accept);
        assert_eq!(soft_reject_decide(2.0, &cfg), SoftRejectVerdict::Reject);
    }

    // -- Config invariants --

    #[test]
    fn config_new_clamps_and_orders() {
        let valid = SoftRejectConfig::new(0.3, 0.7);
        assert!((valid.tau_low - 0.3).abs() < 1e-6);
        assert!((valid.tau_high - 0.7).abs() < 1e-6);

        let swapped = SoftRejectConfig::new(0.7, 0.3); // invalid → defaults
        assert!(swapped.tau_low < swapped.tau_high);
        assert_eq!(swapped.tau_low, 0.4);
        assert_eq!(swapped.tau_high, 0.8);

        let oob = SoftRejectConfig::new(-1.0, 2.0);
        assert_eq!(oob.tau_low, 0.0);
        assert_eq!(oob.tau_high, 1.0);
    }

    // -- Soft-reject pipeline: binary pruner (no SoftReject band) --

    #[test]
    fn binary_pruner_never_enters_soft_reject_band() {
        // Default binary reject_confidence emits only 0.0 / 1.0, so the
        // SoftReject band is unreachable — the caller behaves exactly like is_valid.
        let p = BinaryThresholdPruner { threshold: 5 };
        let cfg = SoftRejectConfig::default();
        let mut relaxer = NoRelaxation;
        let mut scratch = [0u8; 16];

        for tok in 0..10 {
            let accepted =
                soft_reject_with_relax(&p, &mut relaxer, &cfg, 0, tok, &[], &mut scratch);
            assert_eq!(
                accepted,
                tok < 5,
                "token {tok}: soft_reject_with_relax must agree with is_valid for binary pruners"
            );
        }
    }

    // -- Soft-reject pipeline: graded pruner + relaxation --

    #[test]
    fn graded_pruner_soft_rejects_then_relaxes() {
        // center=5, beta=3.0: token 5 lands at exactly 0.5 (in the band).
        let p = GradedThresholdPruner { center: 5.0, beta: 3.0 };
        let cfg = SoftRejectConfig::default();
        let mut scratch = [0u8; 16];

        // NoRelaxation: SoftReject escalates to Reject.
        let mut no_relax = NoRelaxation;
        // token 5 → conf 0.5 → SoftReject → NoRelaxation rejects → false
        let v_no_relax = soft_reject_with_relax(&p, &mut no_relax, &cfg, 0, 5, &[], &mut scratch);
        assert!(!v_no_relax, "SoftReject + NoRelaxation must escalate to hard-reject");

        // AlwaysAcceptRelax: SoftReject becomes accept.
        let mut yes_relax = AlwaysAcceptRelax;
        let v_yes_relax = soft_reject_with_relax(&p, &mut yes_relax, &cfg, 0, 5, &[], &mut scratch);
        assert!(v_yes_relax, "SoftReject + AlwaysAcceptRelax must accept");

        // Far-below token → Accept regardless of relaxation.
        let mut any_relax = AlwaysAcceptRelax;
        let v_below = soft_reject_with_relax(&p, &mut any_relax, &cfg, 0, 0, &[], &mut scratch);
        assert!(v_below, "token 0 (conf~0.05) must be outright accepted");

        // Far-above token → Reject regardless of relaxation.
        let v_above = soft_reject_with_relax(&p, &mut any_relax, &cfg, 0, 20, &[], &mut scratch);
        assert!(!v_above, "token 20 (conf~1.0) must be hard-rejected");
    }

    // -- Sigmoid monotonicity (GOAT G2-T1) --

    #[test]
    fn graded_confidence_is_monotone_in_token_idx() {
        // Sigmoid(β·(x-center)) is monotone increasing in x by construction;
        // verify the impl actually exhibits it (no softmax crossover artifacts).
        let p = GradedThresholdPruner { center: 5.0, beta: 3.0 };
        let mut prev = -f32::INFINITY;
        for tok in 0..20 {
            let c = p.reject_confidence(0, tok, &[]);
            assert!(
                c >= prev,
                "non-monotone at token {tok}: prev={prev}, cur={c}"
            );
            prev = c;
        }
    }

    // -- Batch pipeline --

    #[test]
    fn batch_soft_reject_matches_single_calls() {
        let p = GradedThresholdPruner { center: 5.0, beta: 3.0 };
        let cfg = SoftRejectConfig::default();
        let candidates: Vec<usize> = (0..12).collect();
        let mut relaxer = NoRelaxation;
        let mut conf_scratch = vec![0.0f32; 12];
        let mut byte_scratch = [0u8; 16];
        let mut batch_results = vec![false; 12];

        batch_soft_reject_with_relax(
            &p,
            &mut relaxer,
            &cfg,
            0,
            &candidates,
            &[],
            &mut conf_scratch,
            &mut byte_scratch,
            &mut batch_results,
        );

        let mut scratch = [0u8; 16];
        let mut no_relax = NoRelaxation;
        for (i, &tok) in candidates.iter().enumerate() {
            let single = soft_reject_with_relax(&p, &mut no_relax, &cfg, 0, tok, &[], &mut scratch);
            assert_eq!(
                batch_results[i], single,
                "token {tok}: batch={} single={}",
                batch_results[i], single
            );
        }
    }

    // -- Backward compatibility: NoPruner + binary pruners reproduce is_valid --

    #[test]
    fn backward_compat_binary_pruner_matches_is_valid() {
        // GOAT G1-T1 invariant at the *caller* level: for any pruner using the
        // default binary reject_confidence, soft_reject_with_relax must agree
        // with is_valid (the SoftReject band is unreachable).
        let p = BinaryThresholdPruner { threshold: 7 };
        let cfg = SoftRejectConfig::default();
        let mut relaxer = AlwaysAcceptRelax; // even with a permissive relaxer
        let mut scratch = [0u8; 16];

        for tok in 0..15 {
            let via_soft =
                soft_reject_with_relax(&p, &mut relaxer, &cfg, 0, tok, &[], &mut scratch);
            let via_is_valid = p.is_valid(0, tok, &[]);
            assert_eq!(
                via_soft, via_is_valid,
                "token {tok}: soft_reject disagrees with is_valid for binary pruner"
            );
        }
    }
}
