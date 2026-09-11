//! Margin-gated verification escalation (Issue 745, Plan 595; Research 548 —
//! TriSpec distill, arXiv:2601.23180).
//!
//! The transferable primitive from TriSpec is NOT "use a small model as
//! verifier" — it is the **margin-gated escalation rule**: a rank-gap
//! statistic `gap = top1 − top2` computed in O(1) from scores already
//! materialized at the accept-decision point, gating whether a cheap
//! evaluator's verdict stands or an expensive one is invoked.
//!
//! Substrate consumed (all pre-existing — this module adds no numeric core):
//! - [`IndicatorProbeBank`] + [`IndicatorCascade`] (Plan 320) — the two-stage
//!   cascade this gate fuses into as a *graded* stage-1 firing predicate.
//! - `crate::speculative::sample_residual_distribution_into` (Leviathan Eq. 3)
//!   — the escalation re-decide the d2f draft-accept mapping reuses
//!   (`katgpt-forward::d2f_verifier` + `step.rs` are the shipped consumers);
//!   the reuse seam is proven by [`tests::margin_gate_escalation_reuses_residual_substitution`].
//! - `katgpt-forward::diffusion_sampler::SamplerFeatures::margin` already
//!   carries the same top1−top2 statistic for the D2F softmax path (Research
//!   548's "absent workspace-wide" grep missed it by vocabulary — the operator
//!   here is the cascade-side shape, not a duplicate of that field).
//!
//! # Trust polarity is a workload-geometry property (the PoC's defended axis)
//!
//! TriSpec's domain (token acceptance) trusts a DECISIVE top verdict:
//! `gap ≥ λ` → the cheap verdict stands. The cascade's domain (correlated
//! multi-indicator evidence, Zhou et al. 2026 + Bench 320's planted-pair
//! world) has the opposite geometry: a genuine event activates MULTIPLE
//! correlated indicators (small gap), while a lone decisive spike is the
//! spurious-fire shape the stage-2 verifier exists to reject — there,
//! `gap ≤ λ` is the trusted arm. Both polarities ship explicitly
//! ([`MarginPolarity`]); the Bench 711 PoC measures both against both world
//! geometries and reports the defend-wrong matrix.

use std::sync::Arc;

use crate::pruners::indicator_cascade::IndicatorVerifier;
use crate::pruners::indicator_probe_bank::{IndicatorLabel, IndicatorProbeBank};

/// Result of the zero-alloc two-running-maxima pass over a score slice.
///
/// `scores` are expected FINITE and normalized to a common scale (softmax
/// probabilities, sigmoid probe scores, …). The slice is interpreted as a
/// score distribution: a single-element slice is a degenerate one-hot
/// (`top2 = 0.0`, so `gap = top1`); an empty slice returns the zero split.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MarginSplit {
    /// Highest score.
    pub top1: f32,
    /// Second-highest score (`0.0` for a single-element slice).
    pub top2: f32,
    /// `top1 − top2` — the TriSpec rank gap g(p) numerator.
    pub gap: f32,
    /// Index of the top-1 element (lowest index wins ties, matching
    /// [`IndicatorProbeBank::or_fused_fire`]'s stable tie-break).
    pub argmax: usize,
}

/// Two-running-maxima pass: `(top1, top2, gap, argmax)` in O(n) time, O(1)
/// space, zero allocation. No sort, no partial sort, no scratch.
///
/// Branchless body: the runner-up absorbs `min(s, top1)` BEFORE the champion
/// absorbs `s`, so the previous champion demotes exactly when `s` wins. The
/// only conditional left is the argmax select (`f32::max`/`min` compile to
/// `maxss`/`minss`, the index update to a `csel`/`cmov`).
///
/// Ties: the first occurrence takes `top1`/`argmax` (stable tie-break matching
/// [`IndicatorProbeBank::or_fused_fire`]); an equal later value fills `top2`
/// (so `[a, a]` → `gap = 0`, the uniform-two shape).
#[inline]
pub fn margin_split(scores: &[f32]) -> MarginSplit {
    if scores.is_empty() {
        return MarginSplit::default();
    }
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    for (i, &s) in scores.iter().enumerate() {
        top2 = top2.max(top1.min(s));
        argmax = if s > top1 { i } else { argmax };
        top1 = top1.max(s);
    }
    // A slice shorter than 2 leaves top2 at −∞; the degenerate reading is
    // one-hot (second mass = 0), never −∞.
    let top2 = if top2.is_finite() { top2 } else { 0.0 };
    MarginSplit {
        top1,
        top2,
        gap: top1 - top2,
        argmax,
    }
}

/// Which side of the rank gap is the TRUSTED shape.
///
/// The sign is workload geometry, not a universal constant — see the module
/// doc. Shipping both arms explicitly is the defend-wrong lesson of Bench 711:
/// TriSpec's polarity on a cluster-geometry world auto-confirms exactly the
/// lone-spike false positives the stage-2 verifier exists to reject.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MarginPolarity {
    /// TriSpec g(p) = 1[top1 − top2 ≥ λ]: a decisive top verdict is trusted.
    /// Correct on decisive-verdict worlds (token acceptance, single-label
    /// routing).
    #[default]
    DecisiveTrusted,
    /// Correlated-evidence worlds (Bench 320 cluster geometry): coherent
    /// multi-indicator activation (`gap ≤ λ`) is the trusted shape; a lone
    /// decisive spike escalates to the expensive verifier.
    CoherentTrusted,
}

/// The margin gate predicate: `trusted(gap)` decides whether the cheap
/// evaluator's verdict stands (skip the expensive path) or escalates.
#[derive(Clone, Copy, Debug, Default)]
pub struct MarginGate {
    /// Rank-gap threshold λ (TriSpec hand-pins 0.5; Bandit-λ is the T4 arm).
    pub lambda: f32,
    /// Trust polarity — see [`MarginPolarity`].
    pub polarity: MarginPolarity,
}

impl MarginGate {
    /// `true` → the cheap verdict stands WITHOUT the expensive evaluator;
    /// `false` → escalate (stage-2 verifier / residual re-decide).
    #[inline]
    pub fn trusted(&self, gap: f32) -> bool {
        match self.polarity {
            MarginPolarity::DecisiveTrusted => gap >= self.lambda,
            MarginPolarity::CoherentTrusted => gap <= self.lambda,
        }
    }
}

/// Outcome of one margin-gated cascade pass.
///
/// `Clean` preserves the flagged-only contract (an unflagged candidate never
/// reaches the verifier — identical to [`crate::pruners::indicator_cascade::IndicatorCascade`]).
/// `Trusted` is the invocation cut: the cheap verdict stands, the expensive
/// path is skipped. `Escalated` re-decides through the expensive path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MarginDecision<L> {
    /// No probe fired — not flagged, verifier never eligible.
    Clean,
    /// Fired; the margin gate TRUSTS the bank's verdict — verifier skipped.
    Trusted(L),
    /// Fired; the margin gate DISTRUSTS — escalated to the expensive verifier.
    Escalated(L),
}

/// Two-stage cascade with a graded margin-gated stage-1 firing predicate
/// (Issue 745 T2).
///
/// Flow: project → OR-fuse fire (unchanged from Plan 320) → margin gate on
/// the score vector's rank gap → trusted flags confirm WITHOUT the verifier
/// (the invocation cut) → distrusted flags escalate to the verifier
/// (flagged-only semantics preserved: an unflagged candidate still never
/// reaches the verifier).
///
/// Escalation-path reuse (issue T2): in the draft-accept mapping the
/// "expensive verifier" is the target-model re-decide via
/// `crate::speculative::sample_residual_distribution_into` (Leviathan Eq. 3 —
/// already shipped and consumed by `katgpt-forward::d2f_verifier` +
/// `step.rs`); in the cascade mapping it is the opaque stage-2 verifier
/// below. One gate, two consumers, no new machinery.
pub struct MarginGatedCascade<L: IndicatorLabel, const D: usize> {
    /// Stage-1 bank (unchanged substrate).
    pub bank: Arc<IndicatorProbeBank<L, D>>,
    /// Stage-2 verifier: adjudicates ESCALATED flags only.
    pub verifier: Arc<dyn IndicatorVerifier<L>>,
    /// OR-fusion firing threshold (unchanged from [`crate::pruners::indicator_cascade::IndicatorCascade`]).
    pub tau_fire: f32,
    /// The margin gate (λ + polarity).
    pub gate: MarginGate,
}

impl<L: IndicatorLabel, const D: usize> MarginGatedCascade<L, D> {
    /// Construct from the same parts as [`crate::pruners::indicator_cascade::IndicatorCascade::new`]
    /// plus the margin gate.
    pub fn new(
        bank: Arc<IndicatorProbeBank<L, D>>,
        verifier: Arc<dyn IndicatorVerifier<L>>,
        tau_fire: f32,
        gate: MarginGate,
    ) -> Self {
        Self {
            bank,
            verifier,
            tau_fire,
            gate,
        }
    }

    /// Graded stage-1: project → fire → margin-gate the flag.
    ///
    /// Zero-allocation (caller-owned `scores_scratch`, length `L::COUNT`).
    /// The scores remain in `scores_scratch` after the call — the escalation
    /// consumer ([`Self::confirmed`], or a residual-substitution re-decide)
    /// reads them from there, matching [`IndicatorVerifier::verify`]'s shape.
    #[inline]
    pub fn run(&self, state: &[f32; D], scores_scratch: &mut [f32]) -> MarginDecision<L> {
        self.bank.project_all_into(state, scores_scratch);
        let firing = match self.bank.or_fused_fire(scores_scratch, self.tau_fire) {
            Some(l) => l,
            None => return MarginDecision::Clean,
        };
        let split = margin_split(scores_scratch);
        if self.gate.trusted(split.gap) {
            MarginDecision::Trusted(firing)
        } else {
            MarginDecision::Escalated(firing)
        }
    }

    /// Confirmed-misaligned verdict under the gate: `Trusted` stands without
    /// the verifier; `Escalated` is adjudicated by the verifier; `Clean` is
    /// `None` (flagged-only preserved). Same return contract as
    /// [`crate::pruners::indicator_cascade::IndicatorCascade::run`].
    #[inline]
    pub fn confirmed(&self, state: &[f32; D], scores_scratch: &mut [f32]) -> Option<L> {
        match self.run(state, scores_scratch) {
            MarginDecision::Clean => None,
            MarginDecision::Trusted(l) => Some(l),
            MarginDecision::Escalated(l) => {
                if self.verifier.verify(l, scores_scratch) {
                    Some(l)
                } else {
                    None
                }
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::indicator_probe_bank::DemoIndicatorLabel;
    use crate::speculative::sample_residual_distribution_into;
    use katgpt_types::Rng;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ── T1 property tests ────────────────────────────────────────────────

    #[test]
    fn margin_split_gap_zero_on_uniform() {
        // Softmax of equal logits → uniform distribution → top1 == top2 → gap 0.
        let n = 8;
        let p = 1.0 / n as f32;
        let uniform = vec![p; n];
        let split = margin_split(&uniform);
        assert!(
            split.gap.abs() < 1e-6,
            "uniform gap must be 0, got {}",
            split.gap
        );
        assert_eq!(
            split.argmax, 0,
            "uniform argmax is the first index (stable)"
        );
    }

    #[test]
    fn margin_split_gap_one_minus_p2_on_one_hot() {
        // One-hot [1, 0, 0]: top1 = 1, top2 = p2 = 0 → gap = 1 − p2 = 1.
        let one_hot = [1.0f32, 0.0, 0.0];
        let split = margin_split(&one_hot);
        assert_eq!(split.top1, 1.0);
        assert_eq!(split.top2, 0.0);
        assert!((split.gap - (1.0 - split.top2)).abs() < 1e-6);
        assert!((split.gap - 1.0).abs() < 1e-6);
        assert_eq!(split.argmax, 0);
    }

    #[test]
    fn margin_split_matches_sort_reference_with_ties() {
        // Cross-check the two-running-maxima pass against a sort-based
        // reference over deterministic pseudo-random slices, ties included.
        let mut seed = 0x745u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for len in 2..=9usize {
            for _ in 0..64 {
                // Values in a small discrete grid so ties occur frequently.
                let scores: Vec<f32> = (0..len).map(|_| ((next() % 5) as f32) * 0.25).collect();
                let mut sorted = scores.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let split = margin_split(&scores);
                assert!((split.top1 - sorted[0]).abs() < 1e-6, "top1 {scores:?}");
                assert!((split.top2 - sorted[1]).abs() < 1e-6, "top2 {scores:?}");
                assert!((split.gap - (sorted[0] - sorted[1])).abs() < 1e-6);
                assert_eq!(scores[split.argmax], sorted[0]);
            }
        }
    }

    #[test]
    fn margin_split_tie_takes_lowest_index_and_zero_gap() {
        let split = margin_split(&[5.0f32, 5.0, 5.0]);
        assert_eq!(split.top1, 5.0);
        assert_eq!(split.top2, 5.0);
        assert_eq!(split.gap, 0.0);
        assert_eq!(split.argmax, 0, "stable tie-break: lowest index");
    }

    #[test]
    fn margin_split_single_element_is_degenerate_one_hot() {
        let split = margin_split(&[0.7f32]);
        assert_eq!(split.top1, 0.7);
        assert_eq!(split.top2, 0.0, "single candidate: second mass is 0");
        assert!((split.gap - 0.7).abs() < 1e-6);
    }

    #[test]
    fn margin_split_empty_is_zero() {
        assert_eq!(margin_split(&[]), MarginSplit::default());
    }

    // ── T2 gate semantics + flagged-only preservation ────────────────────

    /// Verifier stub that counts invocations and confirms iff the label is A.
    struct CountingVerifier {
        calls: AtomicUsize,
    }
    impl CountingVerifier {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
            })
        }
    }
    impl IndicatorVerifier<DemoIndicatorLabel> for CountingVerifier {
        fn verify(&self, label: DemoIndicatorLabel, _scores: &[f32]) -> bool {
            self.calls.fetch_add(1, Ordering::Relaxed);
            label == DemoIndicatorLabel::A
        }
    }

    fn demo_bank() -> Arc<IndicatorProbeBank<DemoIndicatorLabel, 4>> {
        let directions = vec![
            1.0, 0.0, 0.0, 0.0, // A
            0.0, 1.0, 0.0, 0.0, // B
            0.0, 0.0, 1.0, 0.0, // C
        ];
        Arc::new(IndicatorProbeBank::new(directions, vec![0.0f32; 3]).unwrap())
    }

    #[test]
    fn clean_candidate_never_invokes_verifier() {
        // Flagged-only preservation: zero state → no probe exceeds tau=0.5
        // → Clean, and the verifier is NEVER called (either polarity).
        for polarity in [
            MarginPolarity::DecisiveTrusted,
            MarginPolarity::CoherentTrusted,
        ] {
            let verifier = CountingVerifier::new();
            let cascade = MarginGatedCascade::new(
                demo_bank(),
                verifier.clone(),
                0.5,
                MarginGate {
                    lambda: 0.5,
                    polarity,
                },
            );
            let state = [0.0f32; 4];
            let mut scratch = [0.0f32; DemoIndicatorLabel::COUNT];
            assert_eq!(cascade.run(&state, &mut scratch), MarginDecision::Clean);
            assert_eq!(cascade.confirmed(&state, &mut scratch), None);
            assert_eq!(
                verifier.calls.load(Ordering::Relaxed),
                0,
                "verifier must not be called on Clean"
            );
        }
    }

    /// State planting a lone spike on indicator A (sigmoid(4) ≈ 0.982) with
    /// the runner-up at sigmoid(0) = 0.5 → gap ≈ 0.482 (decisive shape).
    fn lone_spike_state() -> [f32; 4] {
        [4.0, 0.0, 0.0, 0.0]
    }

    /// State planting a two-indicator cluster A+B (both ≈ 0.982, C at 0.5)
    /// → gap ≈ 0.0 (coherent shape).
    fn cluster_state() -> [f32; 4] {
        [4.0, 4.0, 0.0, 0.0]
    }

    #[test]
    fn decisive_polarity_trusts_lone_spike_and_skips_verifier() {
        let verifier = CountingVerifier::new();
        let cascade = MarginGatedCascade::new(
            demo_bank(),
            verifier.clone(),
            0.5,
            MarginGate {
                lambda: 0.2,
                polarity: MarginPolarity::DecisiveTrusted,
            },
        );
        let state = lone_spike_state();
        let mut scratch = [0.0f32; DemoIndicatorLabel::COUNT];
        let decision = cascade.run(&state, &mut scratch);
        assert_eq!(decision, MarginDecision::Trusted(DemoIndicatorLabel::A));
        assert_eq!(
            cascade.confirmed(&state, &mut scratch),
            Some(DemoIndicatorLabel::A),
            "trusted verdict stands WITHOUT the verifier"
        );
        assert_eq!(
            verifier.calls.load(Ordering::Relaxed),
            0,
            "the invocation cut"
        );
    }

    #[test]
    fn decisive_polarity_escalates_cluster_to_verifier() {
        let verifier = CountingVerifier::new();
        let cascade = MarginGatedCascade::new(
            demo_bank(),
            verifier.clone(),
            0.5,
            MarginGate {
                lambda: 0.2,
                polarity: MarginPolarity::DecisiveTrusted,
            },
        );
        let state = cluster_state();
        let mut scratch = [0.0f32; DemoIndicatorLabel::COUNT];
        assert_eq!(
            cascade.run(&state, &mut scratch),
            MarginDecision::Escalated(DemoIndicatorLabel::A)
        );
        assert_eq!(
            cascade.confirmed(&state, &mut scratch),
            Some(DemoIndicatorLabel::A),
            "escalated flag adjudicated by the verifier (A confirms)"
        );
        assert_eq!(
            verifier.calls.load(Ordering::Relaxed),
            1,
            "escalation consumed one verifier call"
        );
    }

    #[test]
    fn coherent_polarity_mirrors_the_decisions() {
        let verifier = CountingVerifier::new();
        let cascade = MarginGatedCascade::new(
            demo_bank(),
            verifier.clone(),
            0.5,
            MarginGate {
                lambda: 0.2,
                polarity: MarginPolarity::CoherentTrusted,
            },
        );
        let mut scratch = [0.0f32; DemoIndicatorLabel::COUNT];
        // Cluster (gap ≈ 0 ≤ λ) → trusted, no verifier call.
        let cluster = cluster_state();
        assert_eq!(
            cascade.run(&cluster, &mut scratch),
            MarginDecision::Trusted(DemoIndicatorLabel::A)
        );
        // Lone spike (gap ≈ 0.482 > λ) → escalated → verifier adjudicates.
        let spike = lone_spike_state();
        assert_eq!(
            cascade.run(&spike, &mut scratch),
            MarginDecision::Escalated(DemoIndicatorLabel::A)
        );
        assert_eq!(
            cascade.confirmed(&spike, &mut scratch),
            Some(DemoIndicatorLabel::A),
            "escalated flag adjudicated by the verifier (A confirms)"
        );
        assert_eq!(
            verifier.calls.load(Ordering::Relaxed),
            1,
            "only the escalated flag reached the verifier"
        );
    }

    #[test]
    fn lambda_at_zero_trusts_everything_decisive_nothing_coherent() {
        // Boundary semantics: gap >= 0 is always true; gap <= 0 only on exact ties.
        let gate = MarginGate {
            lambda: 0.0,
            polarity: MarginPolarity::DecisiveTrusted,
        };
        assert!(gate.trusted(0.0));
        assert!(gate.trusted(0.9));
        let gate = MarginGate {
            lambda: 0.0,
            polarity: MarginPolarity::CoherentTrusted,
        };
        assert!(gate.trusted(0.0));
        assert!(!gate.trusted(0.1));
    }

    // ── T2 escalation seam: residual substitution reuse ───────────────────

    #[test]
    fn margin_gate_escalation_reuses_residual_substitution() {
        // The d2f draft-accept mapping: on DISTRUST (low margin under the
        // gate), the expensive re-decide is Leviathan Eq. 3 residual
        // substitution — the SAME `sample_residual_distribution_into` the
        // shipped accept loops consume (d2f_verifier.rs, step.rs). This test
        // proves the seam: the escalated arm's replacement token comes from
        // the residual distribution max(p − q, 0), deterministically.
        let gate = MarginGate {
            lambda: 0.2,
            polarity: MarginPolarity::DecisiveTrusted,
        };
        // Draft distribution: ambiguous top-2 (gap 0.05 < λ) → DISTRUSTED.
        let p = [0.45f32, 0.40, 0.15];
        // Target distribution: mass moved to token 2 (p[2] < q[2] → no
        // residual mass there).
        let q = [0.10f32, 0.10, 0.80];
        let mut scratch = [0.0f32; 3];
        let split = margin_split(&p);
        assert!(
            !gate.trusted(split.gap),
            "fixture must DISTRUST (that is the arm under test)"
        );
        let mut rng_a = Rng::new(745);
        let tok_a = sample_residual_distribution_into(&p, &q, &mut scratch, &mut rng_a);
        // Residual mass lives only where p exceeds q (tokens 0/1): the
        // re-decide never lands where the target dominates.
        assert_ne!(
            tok_a, 2,
            "residual substitution must not sample where p <= q"
        );
        // Determinism: same seed → same replacement token (raw sync contract).
        let mut rng_b = Rng::new(745);
        let mut scratch_b = [0.0f32; 3];
        let tok_b = sample_residual_distribution_into(&p, &q, &mut scratch_b, &mut rng_b);
        assert_eq!(tok_a, tok_b, "same seed → identical escalation outcome");
    }
}
