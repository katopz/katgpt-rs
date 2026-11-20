//! Lepski agreement-cascade k budget — the adaptive_k family's third arm.
//!
//! Feature gate: `msa_adaptive_k_lepski` (Issue 1229, Research 681). A
//! CONSUMER of `katgpt_core::lepski_cascade` — the selection rule, `τ̂` and
//! the σ̂ estimator are defined there; this file only maps block routing onto
//! them (substrate-first: wire, never fork).
//!
//! # The three arms
//!
//! | arm | signal | guarantee |
//! |---|---|---|
//! | [`super::adaptive_k::compute_adaptive_k`] | dispersion of the score vector, `sigmoid(w·var+b)` | none (fitted heuristic) |
//! | `compute_derived_k` (`asentmax_schedule`) | Lemma-2 level gap, `4/Δ̂²` | concentration bound |
//! | [`compute_lepski_k`] (this) | **cross-k agreement** of nested prefix means vs `A·τ̂` | oracle inequality (paper §4) |
//!
//! # Mapping (attention → kNN regression)
//!
//! The inner router's top-`k_max` decision is ONE ranked retrieval; prefixes
//! of it are the nested neighbourhoods (no re-scoring). The estimate at size
//! `k` is the mean score of the top-`k` blocks (`f̂_k`), folded once in
//! ascending rank. The noise scale is the bulk MAD of ALL `n_blocks` scores
//! (`sigma_hat_bulk_mad_f32` — the covariate-free null form; differencing
//! ranked scores would measure spacing, not noise). `n = n_blocks`.
//!
//! - A **plateau** of near-tied blocks keeps the prefix means within `A·τ̂`
//!   → the cascade widens to the grid top (broad attention).
//! - A **cliff** (a few needles over the bulk) moves the prefix mean by more
//!   than `A·τ̂` as soon as the bulk enters → the cascade stops at the needle
//!   count (concentrated attention).
//!
//! # Determinism (Issue-940 law)
//!
//! Prefix sums are a sequential f64 fold in ascending rank; the grid is
//! scanned ascending; ties pass (`≤`); NaN rejects. The selected k is a
//! LOCAL per-query routing value — it never crosses the sync boundary.
//!
//! # Side effect
//!
//! [`LepskiKRouter::forward_indexer`] permutes `scratch.scores[..n_blocks]`
//! (in-place median selection, zero allocation). The scores are per-call
//! scratch; read them from the [`RoutingDecision`] or recompute — never
//! from `scratch.scores` after this router returns.

use katgpt_core::lepski_cascade::{
    A_DEFAULT, geometric_grid_into, lepski_cascade, sigma_hat_bulk_mad_f32, tau_hat_into,
};

use super::adaptive_k::AdaptiveKConfig;
use super::vortex_flow::{RoutingDecision, VortexFlow, VortexScratch};

/// Grid capacity: a doubling grid from `k₀ ≥ 1` reaches `2^63` in 64 points,
/// so this never truncates a `usize` range.
const GRID_CAP: usize = 64;

/// Configuration for the Lepski arm: the shared `[k_min, k_max]` band plus the
/// agreement constant `A`. `k_min` is the grid base `k₀` (always admissible).
#[derive(Debug, Clone, Copy)]
pub struct LepskiKConfig {
    /// `k_min` = grid base `k₀`; `k_max` = grid ceiling. `w`/`b` are unused.
    pub band: AdaptiveKConfig,
    /// Agreement constant `A` (`A_DEFAULT = 1` — the G1 cell-2 serving
    /// constant; `A_THEORY = 4.5` is for reading the oracle bound only).
    pub a: f64,
}

impl Default for LepskiKConfig {
    fn default() -> Self {
        Self {
            band: AdaptiveKConfig::default(),
            a: A_DEFAULT,
        }
    }
}

impl LepskiKConfig {
    /// Band `[k_min, k_max]` with the serving constant `A_DEFAULT`.
    pub fn new(k_min: usize, k_max: usize) -> Self {
        Self {
            band: AdaptiveKConfig::new(k_min, k_max),
            a: A_DEFAULT,
        }
    }

    /// Override the agreement constant (builder pattern).
    pub fn with_a(mut self, a: f64) -> Self {
        self.a = a;
        self
    }
}

/// Grid + prefix-mean estimates of one ranked retrieval (stack-resident).
struct PrefixGrid {
    grid: [usize; GRID_CAP],
    est: [f64; GRID_CAP],
    len: usize,
}

impl PrefixGrid {
    /// Build the doubling grid `k₀·2^j ≤ len` and the prefix means
    /// `f̂_k = mean(rank_score(0..k))` — one sequential f64 fold in ascending
    /// rank (Issue-940 order). `None` when the grid has fewer than two points
    /// (nothing to compare: the caller keeps `min(k₀, len)`).
    fn build(k0: usize, len: usize, rank_score: impl Fn(usize) -> f32) -> Option<Self> {
        let mut grid = [0usize; GRID_CAP];
        let j = geometric_grid_into(k0, len, &mut grid);
        if j <= 1 {
            return None;
        }
        let mut est = [0.0f64; GRID_CAP];
        let mut sum = 0.0f64;
        let mut consumed = 0usize;
        for (&k, slot) in grid[..j].iter().zip(est.iter_mut()) {
            for r in consumed..k {
                sum += rank_score(r) as f64;
            }
            consumed = k;
            *slot = sum / k as f64;
        }
        Some(Self { grid, est, len: j })
    }

    /// Run the agreement cascade at noise scale `σ̂` over population `n`.
    fn select(&self, sigma_hat: f64, n: usize, a: f64) -> usize {
        let grid = &self.grid[..self.len];
        let mut tau = [0.0f64; GRID_CAP];
        tau_hat_into(sigma_hat, n.max(1), grid, &mut tau);
        match lepski_cascade(&self.est[..self.len], &tau[..self.len], a) {
            Some(sel) => grid[sel.index],
            None => grid[0],
        }
    }
}

/// Lepski k from scores already in RANK order (descending relevance) and a
/// precomputed noise scale.
///
/// `ranked` is the retrieval (only its first `min(len, k_max)` entries are
/// read); `n_blocks` is the population size in `τ̂ = σ̂·√(log n / k)`.
/// Returns a k in `[min(k_min, len), min(k_max, len)]`; an empty retrieval
/// returns 0. Zero allocation (fixed stack grid).
pub fn compute_lepski_k(
    ranked: &[f32],
    sigma_hat: f64,
    n_blocks: usize,
    config: &LepskiKConfig,
) -> usize {
    let len = ranked.len().min(config.band.k_max);
    let k0 = config.band.k_min.max(1);
    if len <= k0 {
        return len;
    }
    match PrefixGrid::build(k0, len, |r| ranked[r]) {
        Some(pg) => pg.select(sigma_hat, n_blocks, config.a),
        None => k0,
    }
}

/// Lepski-k wrapper around any [`VortexFlow`] router — the third
/// [`super::adaptive_k::AdaptiveKRouter`] sibling.
///
/// 1. Calls the inner router at `top_k = k_max` (one ranked retrieval).
/// 2. Reads the ranked scores through the decision's block indices.
/// 3. Estimates σ̂ as the bulk MAD of every block score (in place).
/// 4. Runs the agreement cascade over the prefix grid; truncates in place.
pub struct LepskiKRouter<R: VortexFlow> {
    /// Inner router to delegate scoring to.
    pub inner: R,
    /// Band + agreement constant.
    pub config: LepskiKConfig,
}

impl<R: VortexFlow> LepskiKRouter<R> {
    /// Create a new Lepski-k router wrapping `inner`.
    pub fn new(inner: R, config: LepskiKConfig) -> Self {
        Self { inner, config }
    }
}

impl<R: VortexFlow> VortexFlow for LepskiKRouter<R> {
    type Cache = R::Cache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        self.inner
            .forward_cache(cache, keys, values, block_idx, head_dim);
    }

    fn forward_indexer(
        &self,
        query: &[f32],
        cache: &Self::Cache,
        n_blocks: usize,
        _top_k: usize,
        scratch: &mut VortexScratch,
    ) -> RoutingDecision {
        if n_blocks == 0 {
            return RoutingDecision::new();
        }
        let band = &self.config.band;
        let k_max = band.k_max.min(n_blocks);
        let mut decision = self
            .inner
            .forward_indexer(query, cache, n_blocks, k_max, scratch);
        let len = decision.len();
        let k0 = band.k_min.max(1);
        if len <= k0 {
            return decision;
        }

        // Prefix estimates read the ranked scores through the decision's
        // block indices BEFORE the in-place σ̂ selection permutes them.
        let k = match PrefixGrid::build(k0, len, |r| scratch.scores[decision.blocks[r]]) {
            Some(pg) => {
                let sigma_hat =
                    sigma_hat_bulk_mad_f32(&mut scratch.scores[..n_blocks]).unwrap_or(0.0);
                pg.select(sigma_hat, n_blocks, self.config.a)
            }
            None => k0,
        };
        truncate(&mut decision, k);
        decision
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        self.inner.cache_new(n_blocks_capacity, head_dim)
    }
}

/// Truncate a decision in place (no fresh allocation).
#[inline]
fn truncate(decision: &mut RoutingDecision, k: usize) {
    decision.blocks.truncate(k);
    decision.weights.truncate(k);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dash_attn::block_topk::{BlockTopKCache, BlockTopKRouter};

    fn sigma(scores: &[f32]) -> f64 {
        let mut s = scores.to_vec();
        sigma_hat_bulk_mad_f32(&mut s).unwrap()
    }

    /// Deterministic bulk: level 0 + splitmix-ish ±0.1 noise, sorted desc.
    fn bulk(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed;
        let mut v: Vec<f32> = (0..n)
            .map(|_| {
                x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = x;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 / (1u64 << 53) as f64 * 0.2 - 0.1) as f32
            })
            .collect();
        v.sort_by(|a, b| b.total_cmp(a));
        v
    }

    #[test]
    fn plateau_widens_to_grid_top() {
        let v = bulk(256, 1);
        let cfg = LepskiKConfig::new(4, 32);
        assert_eq!(compute_lepski_k(&v, sigma(&v), 256, &cfg), 32);
    }

    #[test]
    fn cliff_stops_at_needle_count() {
        let mut v = bulk(256, 2);
        for s in v.iter_mut().take(4) {
            *s = 5.0;
        }
        let cfg = LepskiKConfig::new(4, 32);
        assert_eq!(compute_lepski_k(&v, sigma(&v), 256, &cfg), 4);
        // Eight needles: the cascade keeps the 8-plateau, rejects at 16.
        let mut v8 = bulk(256, 3);
        for s in v8.iter_mut().take(8) {
            *s = 5.0;
        }
        assert_eq!(compute_lepski_k(&v8, sigma(&v8), 256, &cfg), 8);
    }

    #[test]
    fn bounds_and_degenerate_inputs() {
        let cfg = LepskiKConfig::new(4, 32);
        assert_eq!(compute_lepski_k(&[], 0.0, 0, &cfg), 0);
        assert_eq!(compute_lepski_k(&[1.0, 0.5], 0.1, 2, &cfg), 2);
        // σ̂ = 0 with a flat retrieval: every pair ties → passes → grid top.
        assert_eq!(compute_lepski_k(&[1.0; 64], 0.0, 64, &cfg), 32);
        // NaN estimates reject at the first comparison → k₀.
        let mut v = [1.0f32; 64];
        v[5] = f32::NAN;
        assert_eq!(compute_lepski_k(&v, 0.1, 64, &cfg), 4);
    }

    #[test]
    fn router_matches_pure_fn_and_stays_in_band() {
        const HD: usize = 16;
        let n_blocks = 128;
        let inner = BlockTopKRouter::new(true);
        let router = LepskiKRouter::new(inner, LepskiKConfig::new(4, 32));
        let mut cache: BlockTopKCache = router.cache_new(n_blocks, HD);
        let values = vec![0.0f32; 16 * HD];
        for b in 0..n_blocks {
            let keys: Vec<f32> = (0..16 * HD)
                .map(|i| (((b * 31 + i * 7) % 97) as f32 / 97.0) - 0.5)
                .collect();
            router.forward_cache(&mut cache, &keys, &values, b, HD);
        }
        let mut scratch = VortexScratch::new(n_blocks);
        for q in 0..8 {
            let query: Vec<f32> = (0..HD).map(|d| ((q * 13 + d) % 11) as f32 - 5.0).collect();
            // Reference: the inner retrieval + the pure fn over its ranked scores.
            let full = router
                .inner
                .forward_indexer(&query, &cache, n_blocks, 32, &mut scratch);
            let ranked: Vec<f32> = full.blocks.iter().map(|&b| scratch.scores[b]).collect();
            let s = sigma(&scratch.scores[..n_blocks]);
            let want = compute_lepski_k(&ranked, s, n_blocks, &router.config);
            let got = router.forward_indexer(&query, &cache, n_blocks, 32, &mut scratch);
            assert_eq!(got.len(), want, "query {q}");
            assert!((4..=32).contains(&got.len()));
            assert_eq!(
                &got.blocks[..],
                &full.blocks[..got.len()],
                "prefix of ONE retrieval"
            );
        }
    }
}
