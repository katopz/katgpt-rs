//! Input-perturbation ensemble — the missing third member of the UQ
//! provenance family (Plan: reflex 008 / Issue 055; arXiv:2609.33803
//! "Diffusion Reward Models", the decision-layer findings only).
//!
//! # The family census (why this module exists)
//!
//! | Provenance | Substrate home |
//! |---|---|
//! | inter-member disagreement | `velocity_field_disagreement` / `velocity_field_ensemble` |
//! | intra-model sampling | the paper's own diffusion head (external; NOT modelless) |
//! | **input perturbation** | **this module** |
//!
//! A deterministic pipeline answered once gives a point estimate. Run it N
//! times over seeded perturbations of its INPUT features and the spread of
//! the answers is a per-instance uncertainty signal — no training, no
//! generative head. For a hashed-bag embedding (the reflex engine's
//! `[f32; 256]` unigram+bigram bag), the perturbation is Bernoulli bucket
//! dropout: each bucket survives with probability `1 − p_drop`, survivors are
//! re-L2-normalized (cosine geometry must not see a shrunk vector — the
//! `distance_abstain::unit` law), and because corpus routing is an argmax
//! over the perturbed vector, mask draws that reroute produce **natively
//! multimodal answer histograms** — the paper's headline property without a
//! generative head.
//!
//! # The three DRM decision rules (pure modelless math over the samples)
//!
//! - **U_pair** `= 1 − |2·p_majority − 1|` (0 = every sample agrees, 1 = coin
//!   flip) — rank decisions by this and reject the most uncertain first
//!   (paper: +2.81 avg at 70% coverage, PPE Correctness).
//! - **U_BoN** `= runnerup_share` (P̂(a single rival outranks the majority)) —
//!   the Best-of-N flip probability.
//! - **LCB_λ** `= μ_i − λ·σ_i` per option (paper: λ = 0.4) — risk-sensitive
//!   ranking of options by their per-option score moments.
//!
//! # Laws (consumed from the family that preceded this module)
//!
//! - **Determinism**: every draw is BLAKE3-keyed (the [`blake3_uniform_fill`]
//!   helper in this module — the same per-block hash-stream shape as the
//!   guided-width ε source `diversity::temp::blake3_noise_fill`, kept HERE
//!   so the feature stays `= []`-clean: `diversity` lives behind
//!   `temp_loss_fingerprint`, which narrow-feature consumers do not
//!   enable). Same `(seed, len)` ⇒ bit-identical output, every platform,
//!   every run. Cross-seed AGGREGATE stability is a separate assertion
//!   class owned by the consumer's harness — never conflated with the
//!   per-seed determinism pin.
//! - **`p_drop == 0` is bit-identical** (a plain copy — no re-normalization,
//!   which would perturb an already-normalized vector's bits).
//! - **Alloc-free observe (G4)**: `EnsembleHistogram` allocates at
//!   `prepare`/construction only; `observe` touches pre-sized slices.
//! - **Poison control**: NaN scores are silently rejected by the `WelfordVariance`
//!   moments and never win an argmax (`NaN > x` is false) — the
//!   `act_channel_moments` refusal class.
//! - **sigmoid, never softmax**: the fused-gate projection
//!   (`instability_gate`) is a single sigmoid on the U statistic; LCB ranking
//!   is arithmetic on moments with no normalization competition.
//!
//! Input-perturbation ensemble UQ — the third provenance member beside VFD
//! (inter-member) and the DRM head (intra-model, external). Opt-in (Plan:
//! reflex 008 / Issue 055).

use crate::welford::WelfordVariance;

/// The uniform [0, 1) draw source — the sibling of `diversity::temp::
/// blake3_noise_fill`'s per-block hash stream (`seed.to_le_bytes()` for
/// block 0, `seed ‖ b` after), mapped to `[0, 1)` instead of `[-1, 1)·σ`.
/// Kept HERE (not in `diversity::temp`) so this feature carries zero
/// feature deps — `diversity` sits behind `temp_loss_fingerprint`, which
/// narrow-feature consumers (riir-reflex) do not enable.
///
/// `(u >> 8) as f32 / 2^24` — a 24-bit uniform in `[0, 1)`, exactly
/// representable in f32 (a full `u32 as f32 / 2^32` map would ROUND
/// `u32::MAX` to exactly `1.0` — the 24-bit mantissa closes the top of the
/// range; 24 bits of entropy is far beyond any threshold draw's needs).
///
/// Zero-allocation. Same `(seed, out.len())` ⇒ bit-identical output on
/// every platform.
#[inline]
pub fn blake3_uniform_fill(seed: u64, out: &mut [f32]) {
    for (block, chunk) in out.chunks_mut(8).enumerate() {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed.to_le_bytes());
        if block > 0 {
            hasher.update(&(block as u64).to_le_bytes());
        }
        let hash = hasher.finalize();
        let bytes = hash.as_bytes();
        for (i, slot) in chunk.iter_mut().enumerate() {
            let o = i * 4;
            let u = u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            *slot = ((u >> 8) as f32) / 16_777_216.0;
        }
    }
}

/// Bernoulli bucket-dropout perturbation of a feature vector.
///
/// Each of `q.len()` buckets independently survives with probability
/// `1 − p_drop` (draw order = bucket order from [`blake3_uniform_fill`]);
/// dropped buckets are zeroed and the survivor set is re-L2-normalized so
/// cosine consumers see a unit vector of unchanged scale semantics.
///
/// - `p_drop <= 0` ⇒ `out` is a bit-identical copy of `q` (no hashing, no
///   re-normalization — re-normalizing an already-normalized vector can move
///   bits, and the unarmed posture must not).
/// - `p_drop >= 1` ⇒ every bucket drops; `out` is the zero vector (the
///   embed law: "no direction reads as no signal, never NaN").
/// - A survivor set with zero norm (all-zero survivors) also yields the zero
///   vector by the same law.
///
/// Deterministic in `(q, seed, p_drop)`. Allocation-free. `q.len()` must
/// equal `out.len()` (debug-asserted; the caller owns scratch shapes).
pub fn bucket_dropout_into(q: &[f32], out: &mut [f32], seed: u64, p_drop: f32) {
    debug_assert_eq!(q.len(), out.len(), "bucket_dropout_into: scratch shape");
    if p_drop <= 0.0 {
        out.copy_from_slice(q);
        return;
    }
    let mut uniforms = [0f32; 8];
    let mut norm_sq = 0f32;
    for (block, (q_chunk, o_chunk)) in q.chunks(8).zip(out.chunks_mut(8)).enumerate() {
        blake3_uniform_fill(
            seed.wrapping_add(block as u64),
            &mut uniforms[..q_chunk.len()],
        );
        for (i, (src, dst)) in q_chunk.iter().zip(o_chunk.iter_mut()).enumerate() {
            if uniforms[i] < p_drop {
                *dst = 0.0;
            } else {
                *dst = *src;
                norm_sq += src * src;
            }
        }
    }
    if norm_sq > 0.0 {
        let inv = 1.0 / norm_sq.sqrt();
        for slot in out.iter_mut() {
            *slot *= inv;
        }
    }
}

/// Per-question sample accumulator: pick counts + per-option score moments
/// over an ensemble's samples, plus the DRM decision-rule readouts.
///
/// Allocates at [`prepare`](Self::prepare) (construction/config time, the
/// `ActChannelMoments` law); `observe` is allocation-free. One histogram per
/// question — the consumer owns the per-question set.
#[derive(Clone, Debug, Default)]
pub struct EnsembleHistogram {
    pick_counts: Vec<usize>,
    score_moments: Vec<WelfordVariance>,
    n_samples: usize,
    n_abstains: usize,
}

impl EnsembleHistogram {
    /// Size the accumulator for `n_options` options and reset all state.
    pub fn prepare(&mut self, n_options: usize) {
        self.pick_counts.clear();
        self.pick_counts.resize(n_options, 0);
        self.score_moments.clear();
        self.score_moments.resize(n_options, WelfordVariance::new());
        self.n_samples = 0;
        self.n_abstains = 0;
    }

    /// Observe one ensemble sample: `scores[i]` is option `i`'s score, and
    /// `pick` is the pipeline's pick for this sample (`None` = the sample
    /// abstained — moments still record, no pick count does).
    ///
    /// Pick rule when the caller passes `Some` computed elsewhere is the
    /// caller's; [`observe_argmax`](Self::observe_argmax) is the crate's own
    /// deterministic rule (strictly-greater fold from `-inf`: NaN never wins,
    /// ties keep the lower index).
    pub fn observe(&mut self, pick: Option<usize>, scores: &[f32]) {
        debug_assert_eq!(scores.len(), self.pick_counts.len());
        self.n_samples += 1;
        for (i, s) in scores.iter().enumerate() {
            self.score_moments[i].observe(*s);
        }
        match pick {
            Some(p) => {
                if p < self.pick_counts.len() {
                    self.pick_counts[p] += 1;
                }
            }
            None => self.n_abstains += 1,
        }
    }

    /// Observe one sample with the crate's deterministic argmax pick rule
    /// (NaN never wins, ties keep the lower index).
    pub fn observe_argmax(&mut self, scores: &[f32]) {
        let pick = argmax_scores(scores);
        self.observe(pick, scores);
    }

    /// Total observed samples (picks + abstains).
    #[inline]
    pub const fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Samples that abstained (no pick counted).
    #[inline]
    pub const fn n_abstains(&self) -> usize {
        self.n_abstains
    }

    /// The majority pick: argmax over counts, ties to the lower index.
    /// `None` while no sample has picked (all-zero counts are NOT a
    /// majority for option 0).
    #[must_use]
    pub fn majority_pick(&self) -> Option<usize> {
        if self.pick_counts.is_empty() || self.pick_counts.iter().all(|c| *c == 0) {
            return None;
        }
        argmax_lower_index(&self.pick_counts)
    }

    /// Majority pick's share of PICKED samples (abstains excluded from the
    /// denominator — an abstain is not evidence for a rival).
    #[must_use]
    pub fn majority_share(&self) -> Option<f32> {
        let n_picked: usize = self.pick_counts.iter().sum();
        if n_picked == 0 {
            return None;
        }
        let m = self.majority_pick()?;
        Some(self.pick_counts[m] as f32 / n_picked as f32)
    }

    /// The largest non-majority share (P̂(a single rival beats the majority)
    /// — conservative in the k > 2 case, exact for k = 2 where it equals
    /// `1 − majority_share`).
    #[must_use]
    pub fn runnerup_share(&self) -> Option<f32> {
        let n_picked: usize = self.pick_counts.iter().sum();
        if n_picked == 0 {
            return None;
        }
        let m = self.majority_pick()?;
        let mut best = 0usize;
        let mut best_i = None;
        for (i, c) in self.pick_counts.iter().enumerate() {
            if i != m && *c > best {
                best = *c;
                best_i = Some(i);
            }
        }
        match best_i {
            Some(_) => Some(best as f32 / n_picked as f32),
            None => Some(0.0), // unanimous — no rival exists
        }
    }

    /// **U_pair** `= 1 − |2·p_majority − 1|` over picked samples
    /// (0 = unanimous, → 1 = even split). The DRM uncertainty-aware
    /// rejection key; `None` while no sample picked.
    #[must_use]
    pub fn u_pair(&self) -> Option<f32> {
        let p = self.majority_share()?;
        Some(1.0 - (2.0 * p - 1.0).abs())
    }

    /// **U_BoN** = the runner-up's share of picked samples (P̂(the strongest
    /// single rival flips the pick)). `None` while no sample picked.
    #[must_use]
    pub fn u_bon(&self) -> Option<f32> {
        self.runnerup_share()
    }

    /// Option `i`'s mean score across ALL samples (f64 accumulator width).
    #[must_use]
    pub fn option_mean(&self, i: usize) -> f64 {
        self.score_moments[i].mean()
    }

    /// Option `i`'s sample sigma (sqrt of sample variance); 0.0 before two
    /// observations.
    #[must_use]
    pub fn option_sigma(&self, i: usize) -> f32 {
        self.score_moments[i].variance().map_or(0.0, f32::sqrt)
    }

    /// **LCB_λ ranking** — `out[i] = μ_i − λ·σ_i` over the per-option
    /// moments (the paper's risk-sensitive Best-of-N key at λ ≈ 0.4).
    /// Alloc-free; `out.len()` must cover the option set.
    pub fn lcb_into(&self, lambda: f32, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.score_moments.len());
        let lambda = lambda as f64;
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = (self.option_mean(i) - lambda * self.option_sigma(i) as f64) as f32;
        }
    }

    /// The option index maximizing the LCB_λ key (ties → lower index).
    #[must_use]
    pub fn top_by_lcb(&self, lambda: f32) -> Option<usize> {
        let lambda = lambda as f64;
        let mut best = f64::NEG_INFINITY;
        let mut best_i = None;
        for i in 0..self.score_moments.len() {
            let lcb = self.option_mean(i) - lambda * self.option_sigma(i) as f64;
            if lcb > best {
                best = lcb;
                best_i = Some(i);
            }
        }
        best_i
    }
}

/// Deterministic argmax over COUNTS: strictly-greater fold, ties keep the
/// lower index. Empty ⇒ `None`.
fn argmax_lower_index(xs: &[usize]) -> Option<usize> {
    let mut best = 0usize;
    let mut best_i = None;
    for (i, x) in xs.iter().enumerate() {
        if best_i.is_none() || *x > best {
            best = *x;
            best_i = Some(i);
        }
    }
    best_i
}

/// Deterministic argmax over SCORES: fold from `f32::NEG_INFINITY` with a
/// strict `>` — NaN never seeds and never wins (`NaN > x` is false), ties
/// keep the lower index, an all-NaN slice yields `None`.
fn argmax_scores(xs: &[f32]) -> Option<usize> {
    let mut best = f32::NEG_INFINITY;
    let mut best_i = None;
    for (i, x) in xs.iter().enumerate() {
        if *x > best {
            best = *x;
            best_i = Some(i);
        }
    }
    best_i
}

/// The fused-gate third-signal projection: sigmoid of the U statistic at a
/// calibrated temperature. The house bridge (sigmoid, never softmax — a
/// bounded monotone map needs no normalization competition). `u == 0` ⇒ 0.5
/// by construction of the substrate sigmoid.
#[inline]
pub fn instability_gate(u: f32, u_temperature: f32) -> f32 {
    crate::exact_sigmoid(u / u_temperature)
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: [f32; 16] = [
        0.25, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.25, 0.5, 0.0, 0.0, 0.5, 0.0, 0.25, 0.0, 0.0,
    ];

    #[test]
    fn p_zero_is_bit_identical_copy() {
        let mut out = [0f32; 16];
        bucket_dropout_into(&Q, &mut out, 42, 0.0);
        for (a, b) in Q.iter().zip(out.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn seed_determinism_bit_identity() {
        let mut a = [0f32; 16];
        let mut b = [0f32; 16];
        bucket_dropout_into(&Q, &mut a, 7, 0.3);
        bucket_dropout_into(&Q, &mut b, 7, 0.3);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        // A different seed must differ somewhere (16 buckets, p=0.3 — a
        // collision would need every draw to coincide; pinned seed pair).
        let mut c = [0f32; 16];
        bucket_dropout_into(&Q, &mut c, 8, 0.3);
        assert!(
            a.iter()
                .zip(c.iter())
                .any(|(x, y)| x.to_bits() != y.to_bits())
        );
    }

    #[test]
    fn survivors_are_unit_norm_and_zeros_stay_zero() {
        let mut out = [0f32; 16];
        bucket_dropout_into(&Q, &mut out, 3, 0.25);
        let kept = out.iter().filter(|x| **x != 0.0).count();
        assert!(
            kept > 0 && kept < 16,
            "p=0.25 on 16 buckets must keep a strict subset"
        );
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "survivor set must be re-normalized: {norm}"
        );
        // Kept slots scale by ONE common factor: dst/src is constant across kept slots.
        let ratios: Vec<f32> = out
            .iter()
            .zip(Q.iter())
            .filter(|(d, _)| **d != 0.0)
            .map(|(d, s)| d / s)
            .collect();
        assert!(ratios.len() == kept);
        assert!(ratios.iter().all(|r| (r - ratios[0]).abs() < 1e-5));
        // A dropped slot is exactly zero; Q's own zero slots stay zero either way.
        assert_eq!(out.iter().filter(|x| **x == 0.0).count(), 16 - kept);
    }

    #[test]
    fn p_one_yields_the_zero_vector() {
        let mut out = [0f32; 16];
        bucket_dropout_into(&Q, &mut out, 5, 1.0);
        assert!(out.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn histogram_definition_fixtures() {
        let mut h = EnsembleHistogram::default();
        h.prepare(2);
        // 3 samples pick option 0 (scores 0.9/0.1), 1 picks option 1
        // (scores 0.45/0.55), 1 abstains (moments only, no pick).
        h.observe(Some(0), &[0.9, 0.1]);
        h.observe(Some(0), &[0.9, 0.1]);
        h.observe_argmax(&[0.9, 0.1]);
        h.observe_argmax(&[0.45, 0.55]);
        h.observe(None, &[0.5, 0.5]);
        assert_eq!(h.n_samples(), 5);
        assert_eq!(h.n_abstains(), 1);
        assert_eq!(h.majority_pick(), Some(0));
        // majority share over PICKED samples: 3/4 — abstain excluded.
        assert!((h.majority_share().unwrap() - 0.75).abs() < 1e-6);
        assert!((h.runnerup_share().unwrap() - 0.25).abs() < 1e-6);
        // u_pair = 1 − |2·0.75 − 1| = 0.5
        assert!((h.u_pair().unwrap() - 0.5).abs() < 1e-6);
        assert!((h.u_bon().unwrap() - 0.25).abs() < 1e-6);
        // Per-option means over ALL 5 samples (abstain included in moments).
        assert!((h.option_mean(0) - (0.9 + 0.9 + 0.9 + 0.45 + 0.5) / 5.0).abs() < 1e-6);
        assert!((h.option_mean(1) - (0.1 + 0.1 + 0.1 + 0.55 + 0.5) / 5.0).abs() < 1e-6);
        // LCB ranking with these moments: option 0 wins at any sane λ.
        assert_eq!(h.top_by_lcb(0.4), Some(0));
        let mut lcb = [0f32; 2];
        h.lcb_into(0.4, &mut lcb);
        assert!(lcb[0] > lcb[1]);
    }

    #[test]
    fn nan_never_wins_argmax_and_is_rejected_by_moments() {
        let mut h = EnsembleHistogram::default();
        h.prepare(2);
        h.observe_argmax(&[0.3, f32::NAN]);
        assert_eq!(h.majority_pick(), Some(0), "NaN must never win the argmax");
        // NaN was silently rejected from option 1's moments (n stays 0),
        // while option 0's clean 0.3 recorded.
        assert_eq!(h.score_moments[1].n(), 0, "NaN rejected from moments");
        assert_eq!(h.score_moments[0].n(), 1);
        assert!((h.option_mean(1) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn empty_histogram_reads_are_none() {
        let h = EnsembleHistogram::default();
        assert_eq!(h.majority_pick(), None);
        assert_eq!(h.majority_share(), None);
        assert_eq!(h.u_pair(), None);
        // Prepared but unobserved: counts all zero — still None, never Some(0).
        let mut h2 = EnsembleHistogram::default();
        h2.prepare(3);
        assert_eq!(h2.majority_pick(), None);
        assert_eq!(h2.top_by_lcb(0.4), Some(0)); // degenerate: all-equal LCB, lower index
    }

    #[test]
    fn unanimous_histogram_has_zero_uncertainty() {
        let mut h = EnsembleHistogram::default();
        h.prepare(3);
        for _ in 0..8 {
            h.observe_argmax(&[0.2, 0.7, 0.1]);
        }
        assert_eq!(h.u_pair(), Some(0.0));
        assert_eq!(h.u_bon(), Some(0.0));
        assert_eq!(h.runnerup_share(), Some(0.0));
    }

    #[test]
    fn instability_gate_is_sigmoid_never_softmax() {
        assert!((instability_gate(0.0, 1.0) - 0.5).abs() < 1e-6);
        let high = instability_gate(2.0, 1.0);
        let higher = instability_gate(10.0, 1.0);
        assert!(high > 0.5 && higher > high && higher < 1.0);
    }
}
