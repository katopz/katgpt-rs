//! InfiLoop gated N/D running summary — modelless extraction (Issue 950).
//!
//! Research 626 (arXiv:2610.11570, InfiLoop), stripped of training: a
//! two-counter normalized running summary whose writes are gated by a
//! content score against the decayed accumulated mass.
//!
//! ```text
//! N_t = β·N_{t−1} + e_t·z_t      e_t = exp(clip(s_t, ±8))
//! D_t = β·D_{t−1} + e_t          β ∈ (0,1) asserted at construction
//! h_t = N_t / D_t                (mean view; convex hull of candidates)
//! γ_t = σ(s_t − ln(β·D_{t−1}))   (step view; sigmoid of log-ratio — never softmax)
//! ```
//!
//! O(D) memory + O(D) work per write, zero-alloc, f32, deterministic.
//! β ∈ (0,1) is load-bearing (paper ablation: β=1.82 → 2.0%, β=1 → 38.4%,
//! β=0.55 → 74.0%) — refused loud outside the range; β=1 exists ONLY as the
//! named [`GatedSummary::new_oracle`] posture (the [`PrecisionVector`]
//! compatibility proof, P4).
//!
//! # Storage law (P4 — the oracle requirement)
//!
//! The state is stored the way [`PrecisionVector`] updates it — a mean
//! NUDGED toward each new value, `h += (z−h)·e/(βD+e)`, same in-order
//! floating-point ops — never as a separate weighted sum and total that get
//! divided. The N/D form is the exact-arithmetic identity the nudge
//! implements (`h_new = (β·h·D + e·z)/(β·D + e)`); storing the ratio form
//! would break the bit-for-bit oracle with a tolerance-shaped ordering
//! divergence, which is exactly what the oracle exists to catch.
//!
//! # Brain+FSM law
//!
//! The gate is DATA feeding existing deciders (cadence, FSM, frustration
//! drives) — never a bypass, never its own decision layer. The kernel is
//! generic (no game vocabulary); game wiring is a downstream consumer
//! posture, not this module.

/// Score clip bound. `exp(±8)` keeps `e_t` in `[3.4e-4, 2.98e3]` — far from
/// both the f32 underflow and overflow edges. The clip is load-bearing
/// (P2): unclipped large scores overflow `exp` and poison the mass.
pub const SCORE_CLIP: f32 = 8.0;

/// Paper default decay (β₀ = 0.55; the ablation's best retention arm).
pub const DEFAULT_BETA: f32 = 0.55;

/// A gated, β-decayed running summary over `D` dimensions.
///
/// See the [module docs](self) for the recurrence and the storage law.
#[derive(Debug, Clone)]
pub struct GatedSummary<const D: usize> {
    /// `h_t` — the summary mean, stored in nudged form (P4 storage law).
    mean: [f32; D],
    /// `D_t` — decayed accumulated mass.
    mass: f32,
    /// Decay factor β ∈ (0,1]; 1.0 only in the oracle posture.
    beta: f32,
}

/// The step-view gate: `γ = σ(s − ln(β·D_{t−1}))`, with the t=1 edge pinned
/// (`D = 0 ⇒ γ = 1` exactly — the first solid write is accepted fully;
/// never left to log-domain luck). `s` is clipped to ±`SCORE_CLIP` first,
/// matching the writer.
#[inline]
pub fn gamma_step_view(s: f32, mass: f32, beta: f32) -> f32 {
    let s = s.clamp(-SCORE_CLIP, SCORE_CLIP);
    if mass <= 0.0 {
        return 1.0;
    }
    let x = s - (beta * mass).ln();
    // σ(x) = 1/(1+e^(−x)); for x ≥ 0 the reciprocal form stays finite.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// The ratio-form gate: `e/(β·D + e)` — algebraically the same quantity as
/// [`gamma_step_view`] (the P3 identity), and the exact gain the writer
/// applies. `e` must be the clipped-score weight `exp(clip(s, ±8))`.
#[inline]
pub fn gamma_ratio(e: f32, mass: f32, beta: f32) -> f32 {
    e / (beta * mass + e)
}

/// Clip a raw content score and raise it to the write weight `e = exp(s)`.
#[inline]
pub fn score_weight(s: f32) -> f32 {
    s.clamp(-SCORE_CLIP, SCORE_CLIP).exp()
}

impl<const D: usize> GatedSummary<D> {
    /// Fresh summary: zero mean, zero mass. The first write is accepted
    /// fully (mass 0 ⇒ gain `e/e = 1` exactly).
    ///
    /// # Panics
    /// Refuses loud unless `0 < beta < 1` (β=1 is the named oracle posture
    /// only; β outside (0,1) is the paper's broken-decay regime), or `D = 0`.
    pub fn new(beta: f32) -> Self {
        Self::with_prior(beta, [0.0; D], 0.0)
    }

    /// Summary with an explicit prior mean and prior mass. The oracle
    /// posture is `with_prior(1.0, [0.5; D], 1.0)` — [`GatedSummary::new_oracle`].
    ///
    /// # Panics
    /// Same β law as [`GatedSummary::new`]; also refuses a negative or
    /// non-finite prior mass.
    pub fn with_prior(beta: f32, prior_mean: [f32; D], prior_mass: f32) -> Self {
        assert!(D > 0, "GatedSummary: D must be > 0");
        assert!(
            beta > 0.0 && beta < 1.0,
            "GatedSummary: beta {beta} outside (0,1) — the paper's broken-decay \
             regime (β=1.82 → 2.0%, β=1 → 38.4%, β=0.55 → 74.0%); β=1 is the \
             named new_oracle() posture only"
        );
        assert!(
            prior_mass.is_finite() && prior_mass >= 0.0,
            "GatedSummary: prior mass {prior_mass} must be finite and >= 0"
        );
        Self {
            mean: prior_mean,
            mass: prior_mass,
            beta,
        }
    }

    /// The ONLY β=1 posture: byte-compatible with
    /// [`PrecisionVector`](crate::posterior::precision::PrecisionVector)'s
    /// `new()` (prior precision 1.0, prior mean 0.5) under `step_fixed(z, 1.0)`
    /// — the P4 oracle. Named so the compatibility posture is never confused
    /// with a decayed serving posture.
    #[must_use]
    pub fn new_oracle() -> Self {
        Self {
            mean: [0.5; D],
            mass: 1.0,
            beta: 1.0,
        }
    }

    /// Content-scored write: `e = exp(clip(s, ±8))`, then the gated update.
    #[inline]
    pub fn step(&mut self, z: &[f32; D], s: f32) {
        self.write(z, score_weight(s));
    }

    /// Fixed-weight write — the shipped
    /// [`PrecisionVector`](crate::posterior::precision::PrecisionVector)
    /// posture (`w` plays the shipped `obs_weight`; 1.0 for the oracle).
    /// Skips `exp` entirely so the oracle op order is exact.
    #[inline]
    pub fn step_fixed(&mut self, z: &[f32; D], w: f32) {
        self.write(z, w);
    }

    /// The gated nudge (P4 storage law). Op order is load-bearing:
    /// `d_new = β·D + e` first, then per dim `h += (z−h)·(e/d_new)`.
    /// With β=1, e=1.0 this is bit-for-bit `PrecisionVector::update`'s
    /// mean/precision trajectory.
    #[inline]
    fn write(&mut self, z: &[f32; D], e: f32) {
        debug_assert!(
            e.is_finite() && e >= 0.0,
            "write weight must be finite >= 0"
        );
        debug_assert!(z.iter().all(|v| v.is_finite()), "candidate must be finite");
        let d_new = self.beta * self.mass + e;
        let gain = e / d_new;
        for (h, &v) in self.mean.iter_mut().zip(z.iter()) {
            let innovation = v - *h;
            *h += innovation * gain;
        }
        self.mass = d_new;
    }

    /// The current summary mean `h_t`.
    #[inline]
    pub fn read(&self) -> &[f32; D] {
        &self.mean
    }

    /// The current accumulated mass `D_t`.
    #[inline]
    pub fn mass(&self) -> f32 {
        self.mass
    }

    /// The decay factor β (1.0 only in the oracle posture).
    #[inline]
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// The step-view gate the NEXT write with score `s` would receive —
    /// a read-only consumer signal (brain+FSM law: data, never a decision).
    #[inline]
    pub fn gate(&self, s: f32) -> f32 {
        gamma_step_view(s, self.mass, self.beta)
    }
}

/// P4 oracle: β=1 + fixed w=1 reproduces [`PrecisionVector`]'s mean AND
/// precision trajectory bit-for-bit (same initial state, same in-order ops).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::precision::PrecisionVector;
    use crate::posterior::types::EvidenceOutcome;

    /// xorshift64* (bench_684 convention — platform-independent, seeded).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    const D: usize = 8;
    const N_STEPS: usize = 512;

    fn candidate(rng: &mut Rng) -> [f32; D] {
        let mut z = [0.0f32; D];
        for v in &mut z {
            *v = rng.next_f32();
        }
        z
    }

    /// P1/P4 — the compatibility proof. Bit-for-bit on BOTH state halves
    /// (mean and mass↔precision), every step, over a seeded random stream.
    /// A tolerance match is NOT acceptable here: a tolerance oracle hides
    /// an ordering divergence, which is what this oracle exists to catch.
    #[test]
    fn oracle_beta1_fixed_w_reproduces_precision_vector_bit_for_bit() {
        let mut rng = Rng::new(0x950);
        let mut pv = PrecisionVector::new();
        let mut gs = GatedSummary::<D>::new_oracle();

        assert_eq!(gs.mass().to_bits(), pv.precision_at(0).to_bits());
        for step in 0..N_STEPS {
            let z = candidate(&mut rng);
            pv.update(EvidenceOutcome::Success, &z, None);
            gs.step_fixed(&z, 1.0);
            for d in 0..D {
                assert_eq!(
                    gs.read()[d].to_bits(),
                    pv.mean_at(d).to_bits(),
                    "mean diverged at step {step}, dim {d}: {:e} vs {:e}",
                    gs.read()[d],
                    pv.mean_at(d)
                );
            }
            assert_eq!(
                gs.mass().to_bits(),
                pv.precision_at(0).to_bits(),
                "mass↔precision diverged at step {step}"
            );
        }
    }

    /// P3a — the sigmoid/ratio identity within f32 tolerance.
    #[test]
    fn step_view_matches_ratio_form_within_f32() {
        let mut rng = Rng::new(0x951);
        for _ in 0..4096 {
            let s = (rng.next_f32() - 0.5) * 40.0; // beyond the clip on purpose
            let mass = rng.next_f32() * 2000.0;
            let beta = 0.05f32 + 0.9 * rng.next_f32();
            let e = score_weight(s);
            let a = gamma_step_view(s, mass, beta);
            let b = gamma_ratio(e, mass, beta);
            let tol = 1e-5f32.max(b.abs() * 1e-4);
            assert!(
                (a - b).abs() <= tol,
                "gamma identity: step {a:.7} vs ratio {b:.7} (s={s}, m={mass}, β={beta})"
            );
        }
    }

    /// P3b — where both views consume the same state, the gate readout
    /// agrees with the ratio form the writer applies.
    #[test]
    fn gate_readout_agrees_with_writer_gain() {
        let mut rng = Rng::new(0x952);
        let mut gs = GatedSummary::<D>::new(DEFAULT_BETA);
        for _ in 0..256 {
            let z = candidate(&mut rng);
            let s = (rng.next_f32() - 0.5) * 12.0;
            let g_read = gs.gate(s);
            let g_apply = gamma_ratio(score_weight(s), gs.mass(), gs.beta());
            assert!((g_read - g_apply).abs() <= 1e-5);
            gs.step(&z, s);
        }
    }

    /// P3c — the t=1 edge: `D = 0 ⇒ γ = 1` exactly, and the first write is
    /// accepted fully (h₁ = z₁ bit-exact, mass 0 ⇒ gain e/e = 1).
    #[test]
    fn t1_edge_zero_mass_gate_is_exactly_one() {
        assert_eq!(
            gamma_step_view(3.0, 0.0, DEFAULT_BETA).to_bits(),
            1.0f32.to_bits()
        );
        assert_eq!(
            gamma_step_view(-3.0, 0.0, DEFAULT_BETA).to_bits(),
            1.0f32.to_bits()
        );
        let mut gs = GatedSummary::<4>::new(DEFAULT_BETA);
        let z = [0.25f32, -0.5, 2.0, 7.0];
        gs.step(&z, 0.0);
        assert_eq!(gs.read(), &z, "first write must be accepted fully");
        assert_eq!(
            gs.mass().to_bits(),
            1.0f32.to_bits(),
            "mass after first write is e=e^(0)=1"
        );
    }

    /// P2 — the clip is load-bearing: extreme scores must leave the state
    /// finite. (With the clip removed, exp overflows → mass inf → mean NaN
    /// and this arm fails.)
    #[test]
    fn extreme_scores_stay_finite_through_the_clip() {
        let mut gs = GatedSummary::<4>::new(DEFAULT_BETA);
        let z = [1.0f32; 4];
        gs.step(&z, 1e30);
        gs.step(&[-1.0f32; 4], -1e30);
        assert!(gs.mass().is_finite());
        assert!(gs.read().iter().all(|v| v.is_finite()));
        // The clip pins the weights: e(1e30) == e(+8), e(-1e30) == e(-8),
        // and the mass is exactly β·e₊ + e₋ (both steps took the clipped
        // weights — an unclipped exp would have overflowed to inf).
        assert_eq!(score_weight(1e30).to_bits(), SCORE_CLIP.exp().to_bits());
        assert_eq!(score_weight(-1e30).to_bits(), (-SCORE_CLIP).exp().to_bits());
        let expected = DEFAULT_BETA * SCORE_CLIP.exp() + (-SCORE_CLIP).exp();
        assert_eq!(gs.mass().to_bits(), expected.to_bits());
    }

    /// P2 — f32 accumulation order pinned: the same write stream is
    /// byte-deterministic (same bits on every run; the seed fixes the stream).
    #[test]
    fn reference_stream_is_byte_deterministic() {
        let run = |seed: u64| {
            let mut rng = Rng::new(seed);
            let mut gs = GatedSummary::<D>::new(DEFAULT_BETA);
            for _ in 0..N_STEPS {
                let z = candidate(&mut rng);
                let s = (rng.next_f32() - 0.5) * 16.0;
                gs.step(&z, s);
            }
            (gs.read().map(|v| v.to_bits()), gs.mass().to_bits())
        };
        assert_eq!(
            run(0x953),
            run(0x953),
            "same seed must reproduce bit-exact state"
        );
    }

    /// P5 — convex-hull invariant: `‖h‖ ≤ max‖z_i‖` across randomized
    /// streams (f32 tolerance), including the adversarial single-dominant
    /// candidate shape.
    #[test]
    fn summary_stays_inside_the_convex_hull_of_candidates() {
        for seed in [0x954u64, 0x955, 0x956] {
            let mut rng = Rng::new(seed);
            let mut gs = GatedSummary::<D>::new(DEFAULT_BETA);
            let mut max_norm = 0.0f32;
            for step in 0..2000 {
                let mut z = candidate(&mut rng);
                if step % 97 == 0 {
                    // Adversarial single-dominant candidate: one huge spike.
                    z[0] = 40.0 * (rng.next_f32() - 0.5);
                }
                let norm = z.iter().map(|v| v * v).sum::<f32>().sqrt();
                max_norm = max_norm.max(norm);
                gs.step(&z, (rng.next_f32() - 0.5) * 12.0);
                let h_norm = gs.read().iter().map(|v| v * v).sum::<f32>().sqrt();
                let bound = max_norm * (1.0 + 1e-5) + 1e-6;
                assert!(
                    h_norm <= bound,
                    "hull violated at step {step}: ‖h‖ {h_norm} > max‖z‖ {bound}"
                );
            }
        }
    }

    /// P5 — never decays to zero: on a stopped (rejected-write) stream the
    /// summary holds bit-still, where a leaky EMA of the same writes decays.
    #[test]
    fn rejected_writes_never_decay_the_summary() {
        let mut rng = Rng::new(0x957);
        let mut gs = GatedSummary::<D>::new(DEFAULT_BETA);
        for _ in 0..100 {
            let z = candidate(&mut rng);
            gs.step(&z, 2.0); // solid writes establish the summary
        }
        let before = *gs.read();
        for _ in 0..1000 {
            gs.step_fixed(&[0.0; D], 0.0); // fully rejected writes (w=0)
        }
        assert_eq!(gs.read(), &before, "rejected writes must leave h bit-still");
        assert!(
            gs.mass() > 0.0,
            "mass decays geometrically but never reaches zero"
        );
    }

    /// P6 — config law: β outside (0,1) refused loud; the oracle posture is
    /// the only named β=1 path.
    #[test]
    #[should_panic(expected = "outside (0,1)")]
    fn beta_one_is_refused_outside_the_oracle_posture() {
        let _ = GatedSummary::<4>::new(1.0);
    }

    #[test]
    #[should_panic(expected = "outside (0,1)")]
    fn beta_above_one_is_refused() {
        let _ = GatedSummary::<4>::new(1.82);
    }

    #[test]
    #[should_panic(expected = "outside (0,1)")]
    fn beta_zero_is_refused() {
        let _ = GatedSummary::<4>::new(0.0);
    }

    #[test]
    #[should_panic(expected = "outside (0,1)")]
    fn beta_negative_is_refused() {
        let _ = GatedSummary::<4>::new(-0.5);
    }

    #[test]
    #[should_panic(expected = "finite and >= 0")]
    fn negative_prior_mass_is_refused() {
        let _ = GatedSummary::<4>::with_prior(DEFAULT_BETA, [0.0; 4], -1.0);
    }

    #[test]
    fn oracle_posture_carries_the_exact_prior() {
        let gs = GatedSummary::<D>::new_oracle();
        assert_eq!(gs.mass().to_bits(), 1.0f32.to_bits());
        assert_eq!(gs.read(), &[0.5; D]);
        assert_eq!(gs.beta().to_bits(), 1.0f32.to_bits());
    }
}
