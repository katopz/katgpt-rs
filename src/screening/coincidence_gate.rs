//! Coincidence Gate — theorem-backed cross-task transfer (Plan 305, Research 284).
//!
//! Given a found optimum `x*` for one simple objective `f1`, probe `x*` against
//! all other simple objectives `f2_k`. Theorem-backed hit rate: `r / |X_O(1)|`
//! per probe vs `r / |X|` from random candidates (exponential lift).
//!
//! # Mechanism
//!
//! Dingle–Hutter 2026 prove that low-Kolmogorov-complexity optima "coincide"
//! across objectives far more often than random pairs: if `x*` is optimal for a
//! simple `f1`, the probability that `x*` is also good for another simple `f2`
//! is `Θ(r / |X_O(1)|)` where `X_O(1)` is the set of "simple" candidates. This
//! is exponentially larger than the `Θ(r / |X|)` hit rate from probing with a
//! random candidate. The gate therefore:
//!
//! 1. **Filters objectives** via [`CoincidenceGate::should_probe`] — only probe
//!    `f2_k` whose own complexity is below `simple_set_size_estimate` (high-K
//!    reward functions have large `X_O(1)` and the theorem doesn't help).
//! 2. **Probes** `x*` against each surviving `f2_k` via
//!    [`CoincidenceGate::probe_transfer`], returning indices where `x*` ranks
//!    in the top-`r` of a small random comparison sample.
//!
//! # Latent-vs-raw
//!
//! This gate operates on `&[u8]` candidates and closure objectives `Fn(&[u8]) ->
//! f32` — it is agnostic to whether those bytes are raw game state or quantised
//! latents. riir-ai Plan 331 wires it to KG triple emission in the private
//! runtime.

use fastrand::Rng;

// ── CoincidenceGate ──────────────────────────────────────────────────────────

/// Theorem-backed cross-task transfer gate.
///
/// Holds a single scalar `simple_set_size_estimate` (the `|X_O(1)|` proxy from
/// Dingle–Hutter 2026). Objectives whose own complexity exceeds this threshold
/// are not probed — the theorem's exponential-lift guarantee fails for them.
#[derive(Debug, Clone, Copy)]
pub struct CoincidenceGate {
    /// Threshold τ on `|X_O(1)|`. Objectives with `K̃(f2) < τ` are eligible
    /// for transfer probing; above τ the candidate set is too large for the
    /// theorem to bite.
    pub simple_set_size_estimate: f32,
}

impl CoincidenceGate {
    /// Construct with a given simple-set-size estimate `τ`.
    ///
    /// Calibrate from domain knowledge: a small `τ` (e.g. 0.3) is conservative
    /// (only probe very simple objectives); a large `τ` (e.g. 0.9) is
    /// permissive but risks wasted probes on near-random objectives.
    #[inline]
    #[must_use]
    pub const fn new(simple_set_size_estimate: f32) -> Self {
        Self {
            simple_set_size_estimate,
        }
    }

    /// Should we probe objective `f2` whose estimated complexity is `k_tilde_of_f2`?
    ///
    /// Returns `true` when `k_tilde_of_f2 < simple_set_size_estimate` — i.e.
    /// the objective is "simple enough" for the theorem's exponential lift to
    /// apply. Complex `f2` (high `K̃`) are skipped.
    ///
    /// # Match over `if`
    ///
    /// Uses `match` per the project style guide, with a guard-free form that
    /// LLVM will compile to a single `ucomiss + jb`.
    #[inline]
    #[must_use]
    pub fn should_probe(&self, k_tilde_of_f2: f32) -> bool {
        match k_tilde_of_f2 < self.simple_set_size_estimate {
            true => true,
            false => false,
        }
    }

    /// Probe `x*` against each objective `f2_k`, returning indices where `x*`
    /// ranks in the top-`r` of a random comparison sample.
    ///
    /// For each objective `f2_k`:
    /// 1. Score `x*` under `f2_k`: `score_star = f2_k(x_star)`.
    /// 2. Draw `sample_size = max(r · 2, 8)` random candidates by uniform
    ///    byte-slice synthesis (we use a caller-provided byte buffer of length
    ///    `x_star.len()` and fill it from `rng`).
    /// 3. Count how many random candidates score strictly higher than `x*`.
    /// 4. If at most `r - 1` random candidates beat `x*`, then `x*` is in the
    ///    top-`r` of the sample → include `k` in the result.
    ///
    /// **Allocation policy:** this is NOT the per-tick hot path (it runs at
    /// objective-evaluation cadence, much slower than the sampler). It returns
    /// `Vec<usize>` for ergonomic chaining. Per the project rule, SmallVec
    /// optimisation is deferred — Phase 1 returns `Vec<usize>`; a future plan
    /// may swap to `SmallVec` once that dep is added. The sampler's zero-alloc
    /// contract is preserved on its own hot path.
    ///
    /// # Determinism
    ///
    /// Pass `Rng::with_seed(N)` for deterministic test runs. Prod callers should
    /// use `Rng::new()` (thread-local entropy).
    ///
    /// # Arguments
    ///
    /// - `x_star`: the known optimum of objective `f1`.
    /// - `objectives`: slice of closures `f2_k(x) -> score` (higher = better).
    /// - `rank_threshold_r`: include `k` if `x*` is in the top-`r` of the random
    ///   comparison sample. `r >= 1`.
    /// - `rng`: caller-owned RNG.
    pub fn probe_transfer<F>(
        &self,
        x_star: &[u8],
        objectives: &[F],
        rank_threshold_r: usize,
        rng: &mut Rng,
    ) -> Vec<usize>
    where
        F: Fn(&[u8]) -> f32,
    {
        debug_assert!(
            rank_threshold_r >= 1,
            "rank_threshold_r must be >= 1, got {}",
            rank_threshold_r
        );
        let mut hits: Vec<usize> = Vec::with_capacity(objectives.len().min(8));
        if x_star.is_empty() {
            // Without state bytes, random synthesis is degenerate; return empty.
            return hits;
        }
        // Sample size: large enough to give the rank test statistical power,
        // small enough to bound per-objective cost. `max(2·r, 8)`.
        let sample_size = (rank_threshold_r * 2).max(8);
        // Reusable scratch buffer for random candidates.
        let mut scratch_candidate = vec![0u8; x_star.len()];
        for (k, f2_k) in objectives.iter().enumerate() {
            let score_star = f2_k(x_star);
            // Count how many random candidates BEAT x*.
            let mut beat_count: usize = 0;
            for _ in 0..sample_size {
                // Synthesise a random candidate of the same shape as x_star.
                rng.fill(&mut scratch_candidate);
                let score_rand = f2_k(&scratch_candidate);
                if score_rand > score_star {
                    beat_count += 1;
                }
            }
            // x* is in the top-r of the sample iff at most r-1 random candidates
            // beat it. Equivalent: beat_count < r.
            if beat_count < rank_threshold_r {
                hits.push(k);
            }
        }
        hits
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── should_probe ──────────────────────────────────────────────────────────

    #[test]
    fn test_should_probe_true_for_simple_f2() {
        let gate = CoincidenceGate::new(0.5);
        assert!(
            gate.should_probe(0.1),
            "low-K f2 (0.1 < 0.5) should be probeable"
        );
    }

    #[test]
    fn test_should_probe_false_for_complex_f2() {
        let gate = CoincidenceGate::new(0.5);
        assert!(
            !gate.should_probe(0.9),
            "high-K f2 (0.9 > 0.5) should NOT be probeable"
        );
    }

    #[test]
    fn test_should_probe_boundary_excludes_equal() {
        // Strict `<`: equality is treated as complex (excluded).
        let gate = CoincidenceGate::new(0.5);
        assert!(
            !gate.should_probe(0.5),
            "boundary case: k_tilde == τ should be excluded (strict <)"
        );
    }

    // ── probe_transfer ────────────────────────────────────────────────────────

    #[test]
    fn test_probe_transfer_finds_top_ranked() {
        // x_star = [10; 4]. Three objectives:
        //   f2_0: prefers high-magnitude bytes → x_star scores 40.
        //         Random [0..255]^4 has E[sum] = 4·127.5 = 510 > 40 → x_star
        //         NOT top-ranked. Should be excluded.
        //   f2_1: prefers LOW-magnitude bytes → x_star (sum 40) is good.
        //         Random [0..255]^4 has E[sum] = 510 > 40, so x_star ranks high.
        //   f2_2: prefers CONSTANT bytes → x_star (all 10) is perfect; random
        //         rarely constant.
        //
        // To make f2_1 deterministic-ish in the test, we craft a deterministic
        // objective: f2_1(x) = -|sum(x) - 40| (peaks at sum=40, x_star hits it).
        // Random candidates rarely sum to ~40, so x_star wins most draws.
        // f2_2 peaks when stddev(x) == 0.
        let x_star = [10u8; 4];

        let f_high_sum = |x: &[u8]| x.iter().map(|&b| b as f32).sum::<f32>();
        let f_near_40 = |x: &[u8]| {
            let s: f32 = x.iter().map(|&b| b as f32).sum();
            -(s - 40.0).abs()
        };
        let f_constant = |x: &[u8]| {
            // Peaks when all bytes equal: -variance.
            let n = x.len() as f32;
            let mean: f32 = x.iter().map(|&b| b as f32).sum::<f32>() / n;
            let var: f32 = x.iter().map(|&b| (b as f32 - mean).powi(2)).sum::<f32>() / n;
            -var
        };
        let objectives = [f_high_sum, f_near_40, f_constant];

        let gate = CoincidenceGate::new(0.5);
        let mut rng = Rng::with_seed(42);
        let hits = gate.probe_transfer(&x_star, &objectives, 1, &mut rng);

        // x_star should NOT win f_high_sum (random sum > 40 in expectation).
        // x_star SHOULD win f_near_40 (it's the unique optimum).
        // x_star SHOULD win f_constant (variance = 0, random rarely matches).
        assert!(
            !hits.contains(&0),
            "x_star (sum=40) should NOT beat random (E[sum]=510) on f_high_sum; hits={hits:?}"
        );
        assert!(
            hits.contains(&1),
            "x_star is optimum of f_near_40, should be in top-1; hits={hits:?}"
        );
        assert!(
            hits.contains(&2),
            "x_star is optimum of f_constant, should be in top-1; hits={hits:?}"
        );
        // Exactly the two expected hits (no spurious extras).
        assert_eq!(
            hits.len(),
            2,
            "expected exactly 2 hits (f_near_40, f_constant); got {hits:?}"
        );
    }

    #[test]
    fn test_probe_transfer_empty_x_star() {
        // Empty x_star → degenerate, return empty hits without panic.
        let gate = CoincidenceGate::new(0.5);
        let mut rng = Rng::with_seed(1);
        let objectives = [|x: &[u8]| x.iter().map(|&b| b as f32).sum::<f32>()];
        let hits = gate.probe_transfer(&[], &objectives, 1, &mut rng);
        assert!(hits.is_empty(), "empty x_star should yield no hits");
    }

    #[test]
    fn test_probe_transfer_no_objectives() {
        let gate = CoincidenceGate::new(0.5);
        let mut rng = Rng::with_seed(1);
        let x_star = [1u8, 2, 3, 4];
        let objectives: [fn(&[u8]) -> f32; 0] = [];
        let hits = gate.probe_transfer(&x_star, &objectives, 1, &mut rng);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_probe_transfer_rank_threshold_direction() {
        // Sanity check: the hit set MONOTONICALLY grows as `r` increases
        // (a looser rank threshold admits weakly more objectives).
        let x_star = [128u8; 4]; // mid-range — mediocre on sum objective.
        let objectives = [|x: &[u8]| x.iter().map(|&b| b as f32).sum::<f32>()];
        let gate = CoincidenceGate::new(0.5);
        let mut rng1 = Rng::with_seed(7);
        let mut rng2 = Rng::with_seed(7);
        let hits_strict = gate.probe_transfer(&x_star, &objectives, 1, &mut rng1);
        let hits_loose = gate.probe_transfer(&x_star, &objectives, 4, &mut rng2);
        assert!(
            hits_loose.len() >= hits_strict.len(),
            "looser r should admit weakly more hits: strict={hits_strict:?} loose={hits_loose:?}"
        );
    }
}
