//! Corpus-distance abstain gate — Proposal 014 T1.6 / Issue 863, the
//! Research 576 §2.1 extraction (`SalesRLAgent`, arXiv:2503.23303).
//!
//! The engine's ABSTAIN is score-threshold-based
//! ([`crate::bridge::calibrated::CalibratedActionBridge::should_abstain`]):
//! abstain when the *reported confidence* is low. That gate is blind to one
//! failure mode the comparison-arena workload guarantees — a decision state
//! far from every registered corpus region, where a confident score is
//! exactly the overconfidence that produces hallucinated answers. This gate
//! adds the orthogonal signal: **how far is this state from the corpus?**
//!
//! Shape: K registered exemplar vectors (unit-normalized once at
//! construction) + a sigmoid projection of the max cosine similarity onto
//! (0, 1). Zero-allocation query path (one stack scratch row, one fold);
//! sigmoid, never softmax; no learned state — the corpus rows ARE the input,
//! so there is nothing to freeze beyond them (no `commitment()`: the
//! caller's corpus is already content-addressed upstream).
//!
//! Calibration-from-outcomes (feeding distance-derived pseudo-probabilities
//! through [`crate::sigmoid_calibration::SigmoidGateCalibrator`]) is the
//! production wiring a live consumer adds; this module ships the raw
//! monotone signal plus the abstain predicate. Opt-in
//! (`distance_abstain`) per the no-default-consumer rule — the candidate
//! consumer is the Proposal 014 Phase-1 engine's abstain arm.
//!
//! Validation: `benches/bench_845_distance_abstain_goat.rs` (T1.6 — two
//! error worlds on one geometry, risk–coverage vs the score-threshold
//! baseline).
//!
//! Corpus-distance abstain gate (Proposal 014 T1.6 / Issue 863) — the
//! Research 576 §2.1 extraction: max-cosine similarity to registered
//! corpus exemplars → sigmoid → abstain when the decision state is far
//! from every corpus region (the OOD failure mode a score-threshold
//! ABSTAIN is blind to). Opt-in (`distance_abstain`).

/// Unit-normalize `v` in place-on-stack; a zero vector passes through
/// (its dots are 0 — "no direction" reads as mid-corpus distance, never NaN).
///
/// `pub(crate)` since Plan 607 T1: `state_option_scoring` consumes THIS
/// implementation (feature implication, not a fork) so the two normalize
/// bit-identically — a duplicated numeric helper is a drift seam.
pub(crate) fn unit<const D: usize>(v: [f32; D]) -> [f32; D] {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        let mut out = v;
        for x in out.iter_mut() {
            *x *= inv;
        }
        out
    } else {
        v
    }
}

/// Sigmoid projection for the gate — delegates to [`crate::exact_sigmoid`]
/// (the libm two-branch reference form, Issue 870). Bit-identical to the
/// original single-branch body for x ≥ 0 (same expression); ≤ 3 ULP for
/// x < 0 across [−20, 20] (MEASURED at 0.001 step: max 3 ULPs — at
/// x = −4.851 inside the gate's validated domain and x = −16.743 in the
/// margin; a relative error ≲5e-7 on values at or below the gate's
/// decision region). The envelope is pinned by
/// `sigmoid_delegation_matches_frozen_legacy_body` so a future form change
/// cannot silently move gate outputs; the bench_845 GOAT gates were re-run
/// at this form (Issue 870 closeout).
#[inline]
fn sigmoid(x: f32) -> f32 {
    crate::exact_sigmoid(x)
}

/// Abstain gate over corpus distance. `D` is the latent width; `K` is
/// carried by the row vector's length, not a const parameter (corpora grow).
pub struct CorpusDistanceGate<const D: usize> {
    /// Unit-normalized exemplar rows, flat `[K][D]`.
    rows: Vec<[f32; D]>,
    /// Sigmoid midpoint on the max-similarity axis.
    mid: f32,
    /// Sigmoid scale (> 0).
    scale: f32,
}

impl<const D: usize> CorpusDistanceGate<D> {
    /// Register `exemplars` (normalized internally — callers may pass raw
    /// latents). `abstain_confidence = sigmoid(scale · (max_sim − mid))`.
    ///
    /// An EMPTY corpus is legal and means "no coverage": every query reads
    /// max-similarity −1 → confidence ≈ 0 → abstains at any positive
    /// threshold. Refusing the empty corpus would force every caller to
    /// carry its own no-corpus special case; abstain-always is the honest
    /// degradation.
    pub fn new(exemplars: &[[f32; D]], mid: f32, scale: f32) -> Self {
        assert!(mid.is_finite(), "sigmoid midpoint must be finite");
        assert!(
            scale.is_finite() && scale > 0.0,
            "sigmoid scale must be finite and positive"
        );
        Self {
            rows: exemplars.iter().map(|r| unit(*r)).collect(),
            mid,
            scale,
        }
    }

    /// Max cosine similarity against the registered rows (−1.0 for an empty
    /// corpus). Zero-allocation: one stack scratch row, one fold.
    pub fn max_similarity(&self, q: &[f32; D]) -> f32 {
        let nq = unit(*q);
        self.rows.iter().fold(-1.0f32, |m, r| {
            m.max(r.iter().zip(nq.iter()).map(|(a, b)| a * b).sum())
        })
    }

    /// Sigmoid projection of [`Self::max_similarity`] onto (0, 1).
    pub fn abstain_confidence(&self, q: &[f32; D]) -> f32 {
        sigmoid(self.scale * (self.max_similarity(q) - self.mid))
    }

    /// Abstain when the distance-derived confidence falls below `threshold`
    /// (`<`, mirroring `CalibratedActionBridge`'s score-threshold shape).
    pub fn should_abstain(&self, q: &[f32; D], threshold: f32) -> bool {
        self.abstain_confidence(q) < threshold
    }

    /// Fused abstain: the OR of the two gates. The deployable arm — abstain
    /// when EITHER the score is untrusted OR the state is off-corpus.
    /// Composed here (not by the caller) so the union semantics live in one
    /// place and the T1.6 bench's `min(p, d_conf)` sweep matches production.
    pub fn fused_should_abstain(
        &self,
        q: &[f32; D],
        score_confidence: f32,
        score_threshold: f32,
        distance_threshold: f32,
    ) -> bool {
        score_confidence < score_threshold || self.should_abstain(q, distance_threshold)
    }

    /// Registered exemplar count.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// True when no corpus is registered (abstains at every threshold).
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 16;

    fn seed_row(seed: u64) -> [f32; D] {
        // Deterministic SplitMix64 (the workspace-bench idiom — no rand dep).
        let mut z = seed.wrapping_mul(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        [0u64; D].map(|_| {
            z ^= z >> 13;
            z = z.wrapping_mul(0xFF51AFD7ED558CCD);
            z ^= z >> 33;
            (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
    }

    #[test]
    fn confidence_monotone_decreasing_in_corpus_distance() {
        // K=1 so max-similarity IS the cosine to the single row — strict
        // monotonicity of the gate follows from strict monotonicity of the
        // blend, with no row-crossover jitter.
        let e0 = seed_row(0x100);
        let gate = CorpusDistanceGate::<D>::new(std::slice::from_ref(&e0), 0.2, 8.0);
        // Blend the in-corpus query toward a fresh direction — similarity
        // must fall, confidence with it.
        let far = seed_row(0x9999);
        let mut prev = gate.abstain_confidence(&e0);
        for step in 1..=10u32 {
            let t = step as f32 / 10.0;
            let q = std::array::from_fn(|i| e0[i] * (1.0 - t) + far[i] * t);
            let c = gate.abstain_confidence(&q);
            assert!(
                c < prev,
                "confidence must decrease as the query leaves the corpus ({prev} -> {c})"
            );
            prev = c;
        }
    }

    #[test]
    fn confidence_is_scale_invariant() {
        let exemplars: Vec<[f32; D]> = (0..4).map(|i| seed_row(0x200 + i)).collect();
        let gate = CorpusDistanceGate::<D>::new(&exemplars, 0.2, 8.0);
        let q = exemplars[1];
        let scaled = q.map(|x| x * 7.5);
        assert_eq!(
            gate.abstain_confidence(&q).to_bits(),
            gate.abstain_confidence(&scaled).to_bits(),
            "cosine geometry: a rescaled query must read bit-identically"
        );
    }

    #[test]
    fn empty_corpus_abstains_at_any_positive_threshold() {
        let gate = CorpusDistanceGate::<D>::new(&[], 0.2, 8.0);
        assert!(gate.is_empty());
        let q = seed_row(0x300);
        assert!(gate.should_abstain(&q, 0.01));
        assert_eq!(gate.max_similarity(&q), -1.0);
    }

    #[test]
    fn zero_vector_query_is_nan_free() {
        let exemplars: Vec<[f32; D]> = (0..4).map(|i| seed_row(0x400 + i)).collect();
        let gate = CorpusDistanceGate::<D>::new(&exemplars, 0.2, 8.0);
        let q = [0.0f32; D];
        let c = gate.abstain_confidence(&q);
        assert!(c.is_finite(), "zero-vector query must not produce NaN");
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn threshold_semantics_mirror_score_bridge() {
        let exemplars: Vec<[f32; D]> = (0..4).map(|i| seed_row(0x500 + i)).collect();
        let gate = CorpusDistanceGate::<D>::new(&exemplars, 0.2, 8.0);
        let q = exemplars[2];
        let c = gate.abstain_confidence(&q);
        // should_abstain is `< threshold` — the confidence itself does not
        // abstain against itself; one ulp ABOVE the confidence does.
        assert!(!gate.should_abstain(&q, c));
        let one_ulp_up = f32::from_bits(c.to_bits() + 1);
        assert!(gate.should_abstain(&q, one_ulp_up));
    }

    #[test]
    fn fused_gate_is_the_exact_or_of_its_halves() {
        let exemplars: Vec<[f32; D]> = (0..4).map(|i| seed_row(0x600 + i)).collect();
        let gate = CorpusDistanceGate::<D>::new(&exemplars, 0.2, 8.0);
        let q = seed_row(0x601);
        for &st in &[0.3f32, 0.9] {
            for &dt in &[0.3f32, 0.9] {
                let fused = gate.fused_should_abstain(&q, 0.5, st, dt);
                let expect = 0.5 < st || gate.should_abstain(&q, dt);
                assert_eq!(fused, expect, "fused must be the OR at ({st}, {dt})");
            }
        }
    }

    #[test]
    fn sigmoid_delegation_matches_frozen_legacy_body() {
        // Issue 870: the module originally carried a local single-branch
        // `1.0 / (1.0 + (-x).exp())`. The delegation to `exact_sigmoid` must
        // not move gate outputs: bit-identical for x ≥ 0, ≤ 3 ULP for x < 0
        // across the gate's reachable domain plus margin (the bench geometry
        // is scale 8 / mid 0.35 ⇒ arg ∈ [−10.8, 5.2]; this sweep covers
        // [−20, 20], all finite-exp territory for both forms; measured max
        // 3 ULPs — x = −4.851 in-domain, x = −16.743 in-margin, 0.001 step).
        // The legacy body is frozen HERE so any future form drift reds this
        // pin.
        let legacy = |x: f32| 1.0f32 / (1.0 + (-x).exp());
        for i in 0..=4_000i32 {
            let x = -20.0f32 + (i as f32) * 0.01;
            let got = sigmoid(x);
            let want = legacy(x);
            if x >= 0.0 {
                assert_eq!(got.to_bits(), want.to_bits(), "x={x} must be bit-identical");
            } else {
                let ulps = (got.to_bits() as i64 - want.to_bits() as i64).abs();
                assert!(
                    ulps <= 3,
                    "x={x} drifted {ulps} ULPs (got {got}, want {want})"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "scale must be finite and positive")]
    fn non_positive_scale_refuses() {
        CorpusDistanceGate::<D>::new(&[], 0.2, 0.0);
    }
}
