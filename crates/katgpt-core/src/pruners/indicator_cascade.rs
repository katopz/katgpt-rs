//! Indicator Cascade — two-stage verifier escalation
//! (Plan 320 Phase 3, Research 301 §2.2 end + §2.3).
//!
//! The paper's two-stage cascade: probes online (stage-1 bank) → verifier
//! offline (stage-2). The bank OR-fuses; the verifier adjudicates flagged
//! candidates only. The cascade's payoff: stage-2 reduces FPR ~15× at modest
//! TPR cost (paper Table 10: 4.5% → 0.3% neutral FPR at -10pp transcript-TPR).
//!
//! The verifier impl is opaque (trait object); the katgpt-rs side ships the
//! trait + stubs. Consumer crates (riir-ai) supply the real LLM-judge impl.

use std::sync::Arc;

use crate::pruners::indicator_probe_bank::{IndicatorLabel, IndicatorProbeBank};

/// Stage-2 verifier for the indicator cascade. The bank (stage-1) flags
/// candidates; the verifier (stage-2) adjudicates flagged candidates only.
///
/// This is the paper's two-stage cascade: probes online → LLM judge offline.
/// The verifier impl is opaque (trait object); the katgpt-rs side ships
/// the trait + stubs. Consumer crates (riir-ai) supply the LLM-judge impl.
pub trait IndicatorVerifier<L: IndicatorLabel>: Send + Sync {
    /// Adjudicate a flagged candidate. Returns `true` if confirmed misaligned
    /// (the verdict that crosses the sync boundary as raw), `false` if the flag
    /// is a false positive.
    ///
    /// The verifier has access to: the firing label, the per-indicator scores,
    /// and any opaque context the impl carries (e.g., full NPC reasoning trace
    /// for an LLM judge).
    fn verify(&self, label: L, scores: &[f32]) -> bool;
}

/// Stub verifier that confirms every flag. For tests + benchmarks only.
pub struct AlwaysConfirmVerifier;

impl<L: IndicatorLabel> IndicatorVerifier<L> for AlwaysConfirmVerifier {
    #[inline]
    fn verify(&self, _label: L, _scores: &[f32]) -> bool {
        true
    }
}

/// Stub verifier that rejects every flag. For tests + benchmarks only.
pub struct AlwaysRejectVerifier;

impl<L: IndicatorLabel> IndicatorVerifier<L> for AlwaysRejectVerifier {
    #[inline]
    fn verify(&self, _label: L, _scores: &[f32]) -> bool {
        false
    }
}

/// Two-stage cascade: bank (stage-1) OR-fuses; verifier (stage-2) adjudicates
/// flagged candidates only.
///
/// Paper §2.2 end: "a turn is reported as misaligned only if both stages agree".
/// The cascade's payoff: stage-2 reduces FPR ~15× at modest TPR cost
/// (paper Table 10: 4.5% → 0.3% neutral FPR at -10pp transcript-TPR).
pub struct IndicatorCascade<L: IndicatorLabel, const D: usize> {
    /// Stage-1 bank: holds the N pre-computed direction vectors + thresholds.
    pub bank: Arc<IndicatorProbeBank<L, D>>,
    /// Stage-2 verifier: adjudicates flagged candidates only. Opaque impl.
    pub verifier: Arc<dyn IndicatorVerifier<L>>,
    /// OR-fusion firing threshold. A label fires iff its sigmoid score strictly
    /// exceeds `tau_fire`.
    pub tau_fire: f32,
}

impl<L: IndicatorLabel, const D: usize> IndicatorCascade<L, D> {
    /// Construct a cascade from a bank + verifier + firing threshold.
    pub fn new(
        bank: Arc<IndicatorProbeBank<L, D>>,
        verifier: Arc<dyn IndicatorVerifier<L>>,
        tau_fire: f32,
    ) -> Self {
        Self {
            bank,
            verifier,
            tau_fire,
        }
    }

    /// Full pipeline: project → OR-fuse → verify.
    ///
    /// Returns the firing label if the cascade confirms (both stages agree);
    /// `None` otherwise (either no label fired, or the verifier rejected).
    ///
    /// Zero-allocation: the caller provides `scores_scratch` (length `L::COUNT`).
    /// No allocation occurs in `run()` itself.
    #[inline]
    pub fn run(&self, state: &[f32; D], scores_scratch: &mut [f32]) -> Option<L> {
        self.bank.project_all_into(state, scores_scratch);
        let firing = self.bank.or_fused_fire(scores_scratch, self.tau_fire)?;
        if self.verifier.verify(firing, scores_scratch) {
            Some(firing)
        } else {
            None
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::indicator_probe_bank::{DemoIndicatorLabel, IndicatorProbeBank};

    const D: usize = 4;

    fn demo_bank() -> IndicatorProbeBank<DemoIndicatorLabel, D> {
        let directions: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // A
            0.0, 1.0, 0.0, 0.0, // B
            0.0, 0.0, 1.0, 0.0, // C
        ];
        let thresholds = vec![0.0f32; 3];
        IndicatorProbeBank::new(directions, thresholds).unwrap()
    }

    #[test]
    fn test_cascade_confirms_when_verifier_confirms() {
        let bank = Arc::new(demo_bank());
        let verifier: Arc<dyn IndicatorVerifier<DemoIndicatorLabel>> =
            Arc::new(AlwaysConfirmVerifier);
        let cascade = IndicatorCascade::new(bank, verifier, 0.5);
        // State [0, 2, 0, 0] → B fires (raw=2, sigmoid≈0.88 > 0.5).
        let state = [0.0f32, 2.0, 0.0, 0.0];
        let mut scratch = [0.0f32; DemoIndicatorLabel::COUNT];
        assert_eq!(
            cascade.run(&state, &mut scratch),
            Some(DemoIndicatorLabel::B),
            "confirming verifier must return the firing label"
        );
    }

    #[test]
    fn test_cascade_rejects_when_verifier_rejects() {
        let bank = Arc::new(demo_bank());
        let verifier: Arc<dyn IndicatorVerifier<DemoIndicatorLabel>> =
            Arc::new(AlwaysRejectVerifier);
        let cascade = IndicatorCascade::new(bank, verifier, 0.5);
        let state = [0.0f32, 2.0, 0.0, 0.0];
        let mut scratch = [0.0f32; DemoIndicatorLabel::COUNT];
        assert_eq!(
            cascade.run(&state, &mut scratch),
            None,
            "rejecting verifier must suppress the firing label"
        );
    }

    #[test]
    fn test_cascade_no_fire_returns_none() {
        let bank = Arc::new(demo_bank());
        let verifier: Arc<dyn IndicatorVerifier<DemoIndicatorLabel>> =
            Arc::new(AlwaysConfirmVerifier);
        let cascade = IndicatorCascade::new(bank, verifier, 0.5);
        // Zero state → all scores = sigmoid(0) = 0.5, which does NOT strictly
        // exceed tau=0.5 → no fire → verifier never called.
        let state = [0.0f32; D];
        let mut scratch = [0.0f32; DemoIndicatorLabel::COUNT];
        assert_eq!(
            cascade.run(&state, &mut scratch),
            None,
            "no fire above tau must return None"
        );
    }
}

// The zero-alloc G4 assertion is verified in the Phase 4 bench
// (`bench_320_indicator_probe_bank_goat.rs`) using a CountingAllocator, matching
// the Plan 327 G4 pattern. It cannot live in a unit test because
// `#[global_allocator]` is crate-binary-unique and would collide with other
// test modules in the same lib test binary. The hot-path contract:
// `IndicatorCascade::run` must allocate 0 bytes over 100 calls after warmup
// (caller-owned `scores_scratch`, no logging, no trait-object boxing).
