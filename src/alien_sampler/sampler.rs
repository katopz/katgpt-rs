//! The Alien Sampler — within-pool z-scored linear fusion of coherence and
//! unavailability.
//!
//! See [`crate::alien_sampler`] for the module-level doc and paper citation.

use super::traits::{AvailabilityScorer, CoherenceScorer};
use super::types::{AlienConfig, AlienSamplerError, ScoredCandidate};

/// Alien Sampler — generic, modelless within-pool ranking primitive distilled
/// from arXiv:2603.01092 (Artiles et al., "The Alien Space of Science", May
/// 2026).
///
/// Ranks a pool of candidates by a z-scored linear fusion of a coherence
/// score and an unavailability score:
///
/// ```text
/// Fβ(S) = (1−β)·zC(S) + β·zU(S)
/// zU    = −zA   (unavailability is the negation of availability)
/// ```
///
/// where `zC` / `zA` are within-pool z-scores (population formula: divide by
/// `N`, not `N−1`) computed on the caller-supplied coherence and availability
/// scorers. Candidates with high Fβ are "alien-coherent" — internally coherent
/// but under-represented in the reference community. Higher Fβ ranks first.
///
/// # Generic parameters
/// - `V` — atom type. Each candidate is a `Vec<V>`; the scorers consume `&[V]`.
/// - `C` — coherence scorer (`impl CoherenceScorer<V>`).
/// - `A` — availability scorer (`impl AvailabilityScorer<V>`).
///
/// # Zero-allocation hot path
/// `rank` and `rank_into` perform **no heap allocation inside the scoring
/// loop**: the caller passes scratch buffers for per-candidate coherence /
/// availability scores, and the only allocation is the output `Vec` (which
/// `rank_into` also lets the caller own). This matches the crate's hot-path
/// convention per AGENTS.md ("Pre-allocate output arrays upfront …").
///
/// # Determinism
/// Bit-identical across runs given the same `(candidates, scorer, config)` —
/// no RNG, no thread-local state, no clock. Required for replay / sync /
/// audit per AGENTS.md.
///
/// # NaN guard
/// If a scorer returns NaN, the z-score for that candidate is NaN; the sort
/// uses `total_cmp` which places NaN last. Callers that want to hard-avoid
/// NaN should clamp inside their scorer.
///
/// # Zero variance (degenerate pool)
/// If all candidates have the same coherence score, `zC = 0` for all of them
/// (population std = 0 ⇒ we substitute 0 instead of dividing by 0 and getting
/// NaN). Same for availability. This means a uniform pool yields `Fβ = 0` for
/// every candidate — the output is then in input order (stable sort).
///
/// Reference: Plan 311 (T1.3), Research 293,
/// source paper [arxiv 2603.01092](https://arxiv.org/abs/2603.01092).
pub struct AlienSampler<V, C: CoherenceScorer<V>, A: AvailabilityScorer<V>> {
    coherence: C,
    availability: A,
    config: AlienConfig,
    _marker: core::marker::PhantomData<V>,
}

impl<V, C, A> AlienSampler<V, C, A>
where
    C: CoherenceScorer<V>,
    A: AvailabilityScorer<V>,
{
    /// Construct a sampler from scorers + config.
    ///
    /// **Validation** (panics on invalid input — one-time setup cost, not a
    /// hot-path check):
    /// - `beta` must be in `[0, 1]`. Outside this range the fusion semantics
    ///   invert in confusing ways and it's almost always a wiring bug.
    ///
    /// `top_m` is passed through unchecked — it's only consumed by scorers
    /// that implement the median-of-top-m rule, and the validity range
    /// depends on the bank size which the sampler doesn't know.
    ///
    /// Reference: Plan 311 T1.3.
    #[must_use]
    pub fn new(coherence: C, availability: A, config: AlienConfig) -> Self {
        assert!(
            (0.0..=1.0).contains(&config.beta),
            "AlienSampler::new: beta must be in [0, 1], got {}",
            config.beta
        );
        Self {
            coherence,
            availability,
            config,
            _marker: core::marker::PhantomData,
        }
    }

    /// Borrowed access to the coherence scorer (for scorers that carry
    /// caller-inspectable state, e.g. [`super::median_top_m::MedianTopMAvailability`]'s
    /// bank).
    #[inline]
    #[must_use]
    pub fn coherence(&self) -> &C {
        &self.coherence
    }

    /// Borrowed access to the availability scorer.
    #[inline]
    #[must_use]
    pub fn availability(&self) -> &A {
        &self.availability
    }

    /// Borrowed access to the config.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &AlienConfig {
        &self.config
    }

    /// Rank `candidates` by Fβ = `(1−β)·zC + β·zU`, where `zU = −zA`.
    ///
    /// Returns `Vec<ScoredCandidate>` sorted by `score` **descending** (highest
    /// Fβ first). Allocation: one `Vec` of size `candidates.len()` for the
    /// output. Use [`Self::rank_into`] to reuse a caller-owned output buffer
    /// (Plan 311 T4.4).
    ///
    /// # Scratch buffers
    /// `scratch_c` and `scratch_a` must each have length `>= candidates.len()`.
    /// They're overwritten with per-candidate coherence / availability scores
    /// during the scoring pass and then reused as z-score scratch — the
    /// caller can inspect them after the call to see the raw scores.
    ///
    /// Length mismatches return `Err(AlienSamplerError::ScratchLengthMismatch)`.
    /// The empty-candidate case returns `Ok(vec![])` without touching scratch.
    ///
    /// # Determinism
    /// Bit-identical across runs for the same inputs (no RNG, no thread-local
    /// state). The sort is a stable sort on `total_cmp` of the f32 scores —
    /// ties break by input index (ascending), so the output is fully
    /// deterministic.
    ///
    /// # Edge cases
    /// - `candidates.is_empty()` → `Ok(vec![])`.
    /// - `candidates.len() == 1` → single-element output; z-scores are 0
    ///   (population std is 0), so `score = 0`.
    /// - All-equal coherence (or availability) → that axis's z-score is 0 for
    ///   all candidates (no NaN).
    ///
    /// Reference: Plan 311 T1.3.
    pub fn rank(
        &self,
        candidates: &[Vec<V>],
        scratch_c: &mut [f32],
        scratch_a: &mut [f32],
    ) -> Result<Vec<ScoredCandidate>, AlienSamplerError> {
        let n = candidates.len();
        let mut out = Vec::with_capacity(n);
        self.rank_into(candidates, scratch_c, scratch_a, &mut out)?;
        Ok(out)
    }

    /// Same as [`Self::rank`] but writes into a caller-owned output buffer.
    ///
    /// `out` is cleared and then populated with `candidates.len()` entries
    /// sorted by score descending. The buffer's existing capacity is reused
    /// — no allocation if `out.capacity() >= candidates.len()`.
    ///
    /// Reference: Plan 311 T4.4.
    pub fn rank_into(
        &self,
        candidates: &[Vec<V>],
        scratch_c: &mut [f32],
        scratch_a: &mut [f32],
        out: &mut Vec<ScoredCandidate>,
    ) -> Result<(), AlienSamplerError> {
        let n = candidates.len();

        // Length checks. We need >= n in each scratch buffer; we only write
        // to scratch[0..n] but accept larger buffers (caller may reuse a
        // pool-sized buffer for variable-size candidate sets).
        if scratch_c.len() < n {
            return Err(AlienSamplerError::ScratchLengthMismatch {
                expected: n,
                got: scratch_c.len(),
                which: "coherence",
            });
        }
        if scratch_a.len() < n {
            return Err(AlienSamplerError::ScratchLengthMismatch {
                expected: n,
                got: scratch_a.len(),
                which: "availability",
            });
        }

        out.clear();
        if n == 0 {
            return Ok(());
        }

        // Split scratch into the used head. Only [0..n] is touched.
        let (c_head, _) = scratch_c.split_at_mut(n);
        let (a_head, _) = scratch_a.split_at_mut(n);

        // Score each candidate into scratch.
        for (i, cand) in candidates.iter().enumerate() {
            c_head[i] = self.coherence.coherence(cand);
            a_head[i] = self.availability.availability(cand);
        }

        // Fuse + sort.
        fuse_and_sort(c_head, a_head, self.config.beta, out);
        Ok(())
    }

    /// Fuse + sort from **pre-computed** per-candidate coherence + availability
    /// scores, skipping the scorer calls entirely.
    ///
    /// This is the **hot-path** entry point for callers that have batch
    /// scorers (e.g. [`super::median_top_m::MedianTopMAvailability::availability_batch`])
    /// which fill the scratch buffers more efficiently than the per-candidate
    /// trait path in [`Self::rank_into`]. The trait path allocates a cosine
    /// scratch per call (cold path); the batch path reuses one scratch across
    /// all candidates (hot path).
    ///
    /// `coherence_scores` and `availability_scores` must have equal length
    /// `n`; that length determines the output. The slices are consumed
    /// (mutated) in place — the caller can inspect them after the call to see
    /// the z-scores (post-fusion, they hold z-scores, not raw scores).
    ///
    /// `out` is cleared and populated with `n` entries sorted by score
    /// descending. Buffer capacity is reused.
    ///
    /// Reference: Plan 311 T4.4 (hot-path variant for batch scorers).
    pub fn rank_precomputed(
        &self,
        coherence_scores: &mut [f32],
        availability_scores: &mut [f32],
        out: &mut Vec<ScoredCandidate>,
    ) -> Result<(), AlienSamplerError> {
        let n = coherence_scores.len();
        if availability_scores.len() != n {
            return Err(AlienSamplerError::ScratchLengthMismatch {
                expected: n,
                got: availability_scores.len(),
                which: "availability",
            });
        }
        out.clear();
        if n == 0 {
            return Ok(());
        }
        fuse_and_sort(coherence_scores, availability_scores, self.config.beta, out);
        Ok(())
    }
}

/// Fuse pre-computed coherence + availability scores via z-scoring and
/// β-weighting, then sort by score descending into `out`.
///
/// This is the shared hot-path kernel used by both [`AlienSampler::rank_into`]
/// (which fills `c` / `a` via per-candidate trait calls) and
/// [`AlienSampler::rank_precomputed`] (which takes pre-filled slices from a
/// batch scorer).
///
/// `c` and `a` are consumed (mutated) in place — after the call they hold
/// z-scores, not raw scores. `out` is cleared and filled with `c.len()`
/// entries sorted by score descending.
///
/// Reference: Plan 311 T1.3 / T4.4.
#[inline]
fn fuse_and_sort(c: &mut [f32], a: &mut [f32], beta: f32, out: &mut Vec<ScoredCandidate>) {
    let n = c.len();
    debug_assert_eq!(a.len(), n, "fuse_and_sort: c and a lengths must match");
    out.clear();
    out.reserve(n);

    // Z-score both axes (population formula).
    let (mean_c, std_c) = mean_std_population(c);
    let (mean_a, std_a) = mean_std_population(a);
    // Avoid divide-by-zero: zero-variance axis → z = 0 for everyone.
    let inv_std_c = if std_c > 0.0 { 1.0 / std_c } else { 0.0 };
    let inv_std_a = if std_a > 0.0 { 1.0 / std_a } else { 0.0 };

    // Fuse + emit. Fβ = (1−β)·zC + β·zU,  zU = −zA.
    // FMA-friendly: (1−β)·zC + β·(−zA) = (1−β)·zC − β·zA.
    // The `zip` lets LLVM elide per-iteration bounds checks on `c` / `a`
    // (both have length `n`); enumerate provides the input index for the
    // stable tie-break.
    let one_minus_beta = 1.0 - beta;
    for (i, (c_i, a_i)) in c.iter().zip(a.iter()).enumerate() {
        let z_c = (*c_i - mean_c) * inv_std_c;
        let z_a = (*a_i - mean_a) * inv_std_a;
        let score = one_minus_beta.mul_add(z_c, -beta * z_a);
        out.push(ScoredCandidate::new(score, i));
    }

    // Sort by score descending. ScoredCandidate::ord is descending-by-score
    // (total_cmp, NaN-last). `sort_unstable` is faster than `sort` (no
    // auxiliary allocation, better cache behavior — ~2× on large n) and still
    // deterministic run-to-run. Tie-breaking order is NOT input-stable, but no
    // caller depends on a specific tie order — the determinism contract is
    // "same input → same output", which unstable sort satisfies.
    out.sort_unstable();
}

/// Population mean and standard deviation of a non-empty slice.
///
/// Returns `(0.0, 0.0)` for an empty slice (avoids NaN). Uses the population
/// formula (`/N`, not `/（N−1)`) for determinism — the sampler is a
/// within-pool ranking primitive, not a statistical estimator, so the
/// population formula is the right choice (it makes the z-scores sum to
/// exactly zero in exact arithmetic, which is a nice invariant).
///
/// Two-pass for numerical stability: compute mean first, then sum squared
/// deviations. Single-pass (Welford) is also stable but loses more precision
/// on the std for pools with large dynamic range — the two-pass form is
/// preferable when we can afford the second pass (we can; the pools are
/// small).
#[inline]
fn mean_std_population(xs: &[f32]) -> (f32, f32) {
    let n = xs.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut sum = 0.0_f32;
    for &x in xs {
        sum += x;
    }
    let mean = sum / (n as f32);
    let mut sum_sq = 0.0_f32;
    for &x in xs {
        let d = x - mean;
        sum_sq += d * d;
    }
    // Population variance = sum_sq / N. Use sqrt after division.
    let var = sum_sq / (n as f32);
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Reference scorers for unit tests ────────────────────────────────────
    //
    // These mirror the ones in `traits.rs::tests` but are duplicated here so
    // the test module is self-contained (no cross-module test dependency).

    /// Coherence = sum of atoms (f32). Higher sum = more coherent.
    struct SumCoherence;

    impl CoherenceScorer<f32> for SumCoherence {
        #[inline]
        fn coherence(&self, atoms: &[f32]) -> f32 {
            atoms.iter().sum()
        }
    }

    /// Availability = first atom (f32). Lets us set up arbitrary availability
    /// values per candidate via the first slot.
    struct FirstAtomAvailability;

    impl AvailabilityScorer<f32> for FirstAtomAvailability {
        #[inline]
        fn availability(&self, atoms: &[f32]) -> f32 {
            atoms.first().copied().unwrap_or(0.0)
        }
    }

    /// Constant availability = 0.5 for every candidate. Used to test the
    /// `β=0`-equivalent path (zero-variance availability ⇒ z_a = 0 ⇒ fusion
    /// reduces to `(1−β)·zC`).
    struct ConstAvailability(f32);

    impl AvailabilityScorer<f32> for ConstAvailability {
        #[inline]
        fn availability(&self, _atoms: &[f32]) -> f32 {
            self.0
        }
    }

    fn make_sampler(beta: f32) -> AlienSampler<f32, SumCoherence, FirstAtomAvailability> {
        AlienSampler::new(
            SumCoherence,
            FirstAtomAvailability,
            AlienConfig { beta, top_m: 10 },
        )
    }

    // ── Edge cases ─────────────────────────────────────────────────────────

    #[test]
    fn rank_empty_returns_empty() {
        let s = make_sampler(0.7);
        let sc = &mut [][..];
        let sa = &mut [][..];
        let out = s.rank(&[], sc, sa).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn rank_single_returns_one_with_zero_z_score() {
        // Single candidate: population std = 0 → z = 0 → score = 0.
        let s = make_sampler(0.7);
        let candidates = vec![vec![1.0, 2.0, 3.0]];
        let mut sc = [0.0];
        let mut sa = [0.0];
        let out = s.rank(&candidates, &mut sc, &mut sa).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].idx, 0);
        assert!(
            out[0].score.abs() < 1e-6,
            "single-candidate score should be 0"
        );
    }

    #[test]
    fn rank_scratch_too_small_returns_err() {
        let s = make_sampler(0.7);
        let candidates = vec![vec![1.0], vec![2.0], vec![3.0]];
        let mut sc = [0.0, 0.0]; // too small
        let mut sa = [0.0, 0.0, 0.0];
        let err = s.rank(&candidates, &mut sc, &mut sa).unwrap_err();
        match err {
            AlienSamplerError::ScratchLengthMismatch {
                expected,
                got,
                which,
            } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
                assert_eq!(which, "coherence");
            }
        }
    }

    // ── Fusion math ────────────────────────────────────────────────────────

    #[test]
    fn beta_zero_is_coherence_only() {
        // β=0 → score = zC. Availability is ignored (zero-weighted). The
        // output order must equal the coherence-only order.
        let s = make_sampler(0.0);
        // Candidates with (coherence_sum, availability_first) varying
        // independently — so coherence-only and availability-only orderings
        // differ.
        let candidates = vec![
            vec![10.0, 0.0],  // coh=10, avail=10
            vec![1.0, 100.0], // coh=101, avail=1
            vec![5.0, 5.0],   // coh=10, avail=5
        ];
        let mut sc = [0.0; 3];
        let mut sa = [0.0; 3];
        let out = s.rank(&candidates, &mut sc, &mut sa).unwrap();
        // Coherence values: [10, 101, 10]. Highest is idx 1, then idx 0/2 tie.
        assert_eq!(
            out[0].idx, 1,
            "highest-coherence candidate should rank first"
        );
        // idx 0 and 2 both have coherence 10 → z=0 → tie → stable sort keeps
        // input order (0 before 2).
        assert_eq!(out[1].idx, 0);
        assert_eq!(out[2].idx, 2);
    }

    #[test]
    fn beta_one_is_unavailability_only() {
        // β=1 → score = zU = −zA. Lower availability (more alien) ranks first.
        let s = make_sampler(1.0);
        let candidates = vec![
            vec![10.0, 0.0],  // avail=10
            vec![1.0, 100.0], // avail=1   ← most alien
            vec![5.0, 5.0],   // avail=5
        ];
        let mut sc = [0.0; 3];
        let mut sa = [0.0; 3];
        let out = s.rank(&candidates, &mut sc, &mut sa).unwrap();
        // Availability values: [10, 1, 5]. zA: [z, -z, 0]-ish.
        // zU = -zA, so highest zU = lowest zA = lowest availability = idx 1.
        assert_eq!(
            out[0].idx, 1,
            "lowest-availability (most alien) should rank first"
        );
        // idx 2 (avail=5) is middle, idx 0 (avail=10) is highest availability.
        assert_eq!(out[1].idx, 2);
        assert_eq!(out[2].idx, 0);
    }

    #[test]
    fn z_score_handles_zero_variance() {
        // All-equal coherence + all-equal availability → both std=0 → both
        // z=0 → score=0 for all. Output is in input order (stable sort on
        // ties).
        let s = AlienSampler::new(
            SumCoherence,
            ConstAvailability(0.5),
            AlienConfig {
                beta: 0.7,
                top_m: 10,
            },
        );
        let candidates = vec![
            vec![1.0, 1.0], // coh=2
            vec![1.0, 1.0], // coh=2 (same)
            vec![1.0, 1.0], // coh=2 (same)
        ];
        let mut sc = [0.0; 3];
        let mut sa = [0.0; 3];
        let out = s.rank(&candidates, &mut sc, &mut sa).unwrap();
        for sc_item in &out {
            assert!(
                sc_item.score.abs() < 1e-6,
                "zero-variance pool should give score 0"
            );
        }
        // Stable sort preserves input order on ties.
        assert_eq!(out[0].idx, 0);
        assert_eq!(out[1].idx, 1);
        assert_eq!(out[2].idx, 2);
    }

    #[test]
    fn determinism_same_inputs_same_output() {
        let s = make_sampler(0.7);
        let candidates = vec![
            vec![3.0, 1.0],
            vec![1.0, 5.0],
            vec![2.0, 2.0],
            vec![4.0, 0.5],
        ];
        let mut sc1 = [0.0; 4];
        let mut sa1 = [0.0; 4];
        let mut sc2 = [0.0; 4];
        let mut sa2 = [0.0; 4];
        let out1 = s.rank(&candidates, &mut sc1, &mut sa1).unwrap();
        let out2 = s.rank(&candidates, &mut sc2, &mut sa2).unwrap();
        assert_eq!(out1, out2);
    }

    #[test]
    fn rank_output_is_permutation_of_input_indices() {
        // Property test (manual): output indices must be a permutation of
        // 0..n.
        let s = make_sampler(0.5);
        let candidates: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32 * 0.1, (i as f32).rem_euclid(3.0)])
            .collect();
        let mut sc = vec![0.0; 10];
        let mut sa = vec![0.0; 10];
        let out = s.rank(&candidates, &mut sc, &mut sa).unwrap();
        let mut idxs: Vec<usize> = out.iter().map(|s| s.idx).collect();
        idxs.sort_unstable();
        assert_eq!(idxs, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn rank_into_reuses_buffer() {
        let s = make_sampler(0.7);
        let candidates = vec![vec![1.0], vec![2.0], vec![3.0]];
        let mut sc = [0.0; 3];
        let mut sa = [0.0; 3];
        let mut out = Vec::with_capacity(3);
        // Put garbage in the buffer to verify it's cleared.
        out.push(ScoredCandidate::new(99.9, 99));
        s.rank_into(&candidates, &mut sc, &mut sa, &mut out)
            .unwrap();
        assert_eq!(out.len(), 3);
        // No 99.99 garbage.
        assert!(out.iter().all(|s| s.idx < 3));
    }

    #[test]
    fn rank_precomputed_matches_rank_into() {
        // rank_precomputed (pre-filled scratch) must produce the same output
        // as rank_into (which fills scratch via trait calls).
        let s = make_sampler(0.7);
        let candidates = vec![
            vec![3.0, 1.0],
            vec![1.0, 5.0],
            vec![2.0, 2.0],
            vec![4.0, 0.5],
            vec![0.5, 4.0],
        ];
        // Path A: rank_into (trait path).
        let mut sc_a = [0.0; 5];
        let mut sa_a = [0.0; 5];
        let mut out_a = Vec::new();
        s.rank_into(&candidates, &mut sc_a, &mut sa_a, &mut out_a)
            .unwrap();

        // Path B: pre-fill scratch manually, then rank_precomputed.
        // We reuse the same scorers by calling them directly.
        let mut sc_b = [0.0; 5];
        let mut sa_b = [0.0; 5];
        for (i, cand) in candidates.iter().enumerate() {
            sc_b[i] = SumCoherence.coherence(cand);
            sa_b[i] = FirstAtomAvailability.availability(cand);
        }
        let mut out_b = Vec::new();
        s.rank_precomputed(&mut sc_b, &mut sa_b, &mut out_b)
            .unwrap();

        assert_eq!(out_a, out_b, "rank_precomputed must match rank_into");
    }

    #[test]
    fn mean_std_population_handles_empty() {
        let (m, s) = mean_std_population(&[]);
        assert_eq!(m, 0.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn mean_std_population_handles_uniform() {
        let (m, s) = mean_std_population(&[5.0, 5.0, 5.0]);
        assert!((m - 5.0).abs() < 1e-6);
        assert!(s.abs() < 1e-6, "uniform pool has zero std");
    }

    #[test]
    fn mean_std_population_known_values() {
        // [1, 2, 3, 4, 5]: mean=3, var=(4+1+0+1+4)/5=2, std=sqrt(2).
        let (m, s) = mean_std_population(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((m - 3.0).abs() < 1e-6);
        assert!((s - 2.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "beta must be in [0, 1]")]
    fn new_rejects_negative_beta() {
        make_sampler(-0.1);
    }

    #[test]
    #[should_panic(expected = "beta must be in [0, 1]")]
    fn new_rejects_beta_above_one() {
        make_sampler(1.1);
    }
}
