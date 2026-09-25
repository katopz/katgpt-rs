//! Differential KV eviction — Issue 882 P3 (Research 586, Diff Transformer
//! arXiv:2410.05258's subtract arm moved from the attention map to the KV
//! eviction score).
//!
//! # The score
//!
//! ```text
//! d_j(t)         = a_j(t) − λ·μ_j(t−1)          // specificity of one query
//! μ_j(t)         = μ_j(t−1) + β·(a_j(t) − μ_j(t−1))   // EMA of key j's mass
//! specificity_j  = max over recent queries of d_j
//! evict argmin specificity (sinks exempt)
//! ```
//!
//! `a_j(t)` is the attention mass query `t` puts on key `j` (one head's
//! softmax row, caller-supplied — the `kv_eviction` house pattern; no kernel
//! change). `μ_j` is key `j`'s common-mode level: a hub key (BOS-like,
//! punctuation, "the") that every query attends has `a ≈ μ`, so its
//! specificity collapses to `(1−λ)·a` plus its fluctuation; a key that ONE
//! query attends keeps nearly its whole spike. The reference uses `μ(t−1)`
//! — the mass BEFORE this query — so a spike cannot cancel itself.
//!
//! # "Recent" — a two-bucket window, O(1) per key per step
//!
//! Queries are grouped into buckets of `window` queries. Each key carries the
//! max over the CURRENT bucket and the max over the PREVIOUS one; the
//! specificity is their max. So a key's score covers between `window` and
//! `2·window − 1` recent queries (never fewer than `window`), at 3 `f32` per
//! key and no ring buffer. Bucket rotation is a whole-table event decided
//! once per query, so the per-key inner loop is branch-free on the common
//! path.
//!
//! # The two reductions (both pinned in the tests and in Bench 894)
//!
//! - **λ = 0** is the plain max-recent-attention baseline (TOVA / H2O class)
//!   **bit-identically**: `a − 0·μ` is exactly `a` for finite `μ ≥ 0`, and `μ`
//!   is finite by construction (only finite, non-negative masses enter it).
//!   [`DiffEvictConfig::max_recent`] names that baseline.
//! - **Budget ≥ live** evicts nothing ([`evictions_for_budget`] returns 0),
//!   so the "no eviction" configuration is a no-op on the cache.
//!
//! # Sinks are exempt — the shipped machinery, not a second rule
//!
//! A sink is the ultimate hub: every query attends it, so `μ ≈ a` and its
//! specificity is `≈ (1−λ)·a` plus its fluctuation. At `λ ≤ 1` a sink's sheer
//! magnitude usually still out-scores the context (Bench 894 G1c: 0/32 sinks
//! lost unpinned at λ = 1 — the pre-registered bar expected otherwise and
//! failed). At `λ > 1` — over-cancellation, where the G1a λ grid PEAKS — the
//! sink's specificity goes negative and it is evicted FIRST (32/32 at
//! λ = 1.5): the P2 trap-3 failure one task over. Selection therefore
//! goes through [`crate::kv_eviction::select_evict_into`]'s pin mask, and
//! [`DifferentialEvictTable::select_evict_sink_exempt`] builds that mask with
//! [`crate::kv_sink_window::sink_pin_mask_into`].
//!
//! # ⛔ Trap 4 — "generically attended" can mean "consistently relevant"
//!
//! The score assumes the queries that come AFTER eviction look like the ones
//! that produced the specificity. When the recent window is full of one-off
//! spikes on content the future will not ask about, and the future queries
//! are generic, this policy keeps the spikes and drops the hubs the future
//! needs — worse than the λ = 0 baseline, and silently so. Bench 894 pins
//! that negative. A fresh key starts at `μ = 0`, which is a built-in recency
//! prior (its first spike is fully "specific"); that covers part of the trap,
//! not all of it.
//!
//! λ is a prior, not a law (trap 5): the caller picks it on its own fixtures.
//!
//! # Sync boundary
//!
//! None. Latent-side bookkeeping over caller-owned attention masses; nothing
//! here is synced or committed.

use crate::kv_sink_window::{SinkWindowPolicy, sink_pin_mask_into};

/// Differential eviction configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffEvictConfig {
    /// Common-mode rejection weight. `0.0` is the max-recent baseline.
    pub lambda: f32,
    /// EMA weight on the NEW sample (`μ ← μ + β(a − μ)`), in `(0, 1]`.
    pub beta: f32,
    /// Queries per recency bucket (≥ 1; `0` is read as `1`).
    pub window: u32,
}

impl DiffEvictConfig {
    pub const fn new(lambda: f32, beta: f32, window: u32) -> Self {
        Self {
            lambda,
            beta,
            window,
        }
    }

    /// The plain max-recent-attention baseline: the same table at `λ = 0`.
    /// Bit-identical to a direct max over the same window (test-pinned).
    pub const fn max_recent(window: u32) -> Self {
        Self::new(0.0, 0.1, window)
    }
}

/// Number of rows to evict so that `live` fits in `budget`. `budget >= live`
/// → 0, which is the "no eviction" no-op.
#[inline]
pub const fn evictions_for_budget(live: usize, budget: usize) -> usize {
    live.saturating_sub(budget)
}

/// Per-slot differential bookkeeping for ONE head, SoA so the per-query
/// update loop vectorises. Allocates once at construction; every per-step
/// path is allocation-free once caller buffers reach capacity (G4).
///
/// Slot indexing mirrors the caller's cache, exactly as
/// [`crate::kv_eviction::UsageScoreTable`] does: when a slot is reused, call
/// [`Self::reset_row`].
pub struct DifferentialEvictTable {
    cfg: DiffEvictConfig,
    mu: Vec<f32>,
    cur: Vec<f32>,
    prev: Vec<f32>,
    len: usize,
    /// Queries observed so far (drives bucket rotation).
    queries: u64,
}

impl DifferentialEvictTable {
    /// Allocate once for `cap` slots. The live prefix starts empty.
    pub fn with_capacity(cap: usize, cfg: DiffEvictConfig) -> Self {
        Self {
            cfg,
            mu: vec![0.0; cap],
            cur: vec![f32::NEG_INFINITY; cap],
            prev: vec![f32::NEG_INFINITY; cap],
            len: 0,
            queries: 0,
        }
    }

    pub fn config(&self) -> DiffEvictConfig {
        self.cfg
    }

    /// Live row count.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Queries observed so far.
    pub fn queries(&self) -> u64 {
        self.queries
    }

    /// (Re)admit slot `idx`: `μ = 0`, no specificity evidence yet
    /// (`−∞`, which orders FIRST for eviction — observe the admitting query
    /// before selecting). Grows the live prefix if `idx >= len`. O(1).
    pub fn reset_row(&mut self, idx: usize) {
        debug_assert!(idx < self.mu.len(), "slot {idx} out of capacity");
        self.mu[idx] = 0.0;
        self.cur[idx] = f32::NEG_INFINITY;
        self.prev[idx] = f32::NEG_INFINITY;
        if idx >= self.len {
            self.len = idx + 1;
        }
    }

    /// Admit slots `0..n` at once (a prefilled context).
    pub fn admit_prefix(&mut self, n: usize) {
        debug_assert!(n <= self.mu.len(), "prefix {n} out of capacity");
        for i in 0..n {
            self.reset_row(i);
        }
    }

    /// Observe one query's attention row over the live prefix. `masses[j]`
    /// is the mass on slot `j`; a shorter slice records no sample for the
    /// missing slots (their buckets still rotate, so the recency window stays
    /// exact for every live slot). A non-finite or negative mass is IGNORED for that slot (the
    /// softmax contract says finite and `>= 0`; dropping it keeps `μ` finite,
    /// which is what makes the λ = 0 reduction exact). O(1) per key.
    pub fn observe_query(&mut self, masses: &[f32]) {
        let w = self.cfg.window.max(1) as u64;
        let rotate = self.queries > 0 && self.queries.is_multiple_of(w);
        let live = self.len;
        let n = live.min(masses.len());
        let lam = self.cfg.lambda;
        let beta = self.cfg.beta;
        let (mu, cur, prev) = (
            &mut self.mu[..live],
            &mut self.cur[..live],
            &mut self.prev[..live],
        );
        match rotate {
            true => {
                for j in n..live {
                    prev[j] = cur[j];
                    cur[j] = f32::NEG_INFINITY;
                }
                for j in 0..n {
                    let a = masses[j];
                    if !(a.is_finite() && a >= 0.0) {
                        // Still rotate: the previous bucket ages out even for
                        // a slot whose sample this step was rejected.
                        prev[j] = cur[j];
                        cur[j] = f32::NEG_INFINITY;
                        continue;
                    }
                    let m = mu[j];
                    let d = a - lam * m;
                    prev[j] = cur[j];
                    cur[j] = d;
                    mu[j] = m + beta * (a - m);
                }
            }
            false => {
                for j in 0..n {
                    let a = masses[j];
                    if !(a.is_finite() && a >= 0.0) {
                        continue;
                    }
                    let m = mu[j];
                    let d = a - lam * m;
                    cur[j] = cur[j].max(d);
                    mu[j] = m + beta * (a - m);
                }
            }
        }
        self.queries += 1;
    }

    /// Specificity of one slot (`max(cur, prev)`; `−∞` = no evidence yet).
    #[inline]
    pub fn specificity(&self, idx: usize) -> f32 {
        self.cur[idx].max(self.prev[idx])
    }

    /// Common-mode level `μ` of one slot.
    #[inline]
    pub fn mass_ema(&self, idx: usize) -> f32 {
        self.mu[idx]
    }

    /// Write every live slot's specificity into `out` (reused buffer).
    pub fn specificity_into(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(
            self.cur[..self.len]
                .iter()
                .zip(&self.prev[..self.len])
                .map(|(c, p)| c.max(*p)),
        );
    }

    /// Lowest-`k` specificity among unpinned slots, in eviction-priority
    /// order — [`crate::kv_eviction::select_evict_into`] over
    /// [`Self::specificity_into`], so ties break by ascending index and the
    /// ordering is NaN-safe (`float_order::cmp_for_min`). Zero allocation
    /// once `scores` / `out` reach capacity.
    pub fn select_evict_into(
        &self,
        k: usize,
        pinned: &[bool],
        scores: &mut Vec<f32>,
        out: &mut Vec<usize>,
    ) {
        self.specificity_into(scores);
        super::select_evict_into(scores, k, pinned, out);
    }

    /// [`Self::select_evict_into`] with the pin mask built by the shipped
    /// sink rule ([`sink_pin_mask_into`]): `positions[i]` is slot `i`'s
    /// absolute position; every [`crate::kv_sink_window::SlotClass::Sink`]
    /// slot is exempt at any specificity.
    #[allow(clippy::too_many_arguments)]
    pub fn select_evict_sink_exempt(
        &self,
        policy: &SinkWindowPolicy,
        positions: &[u64],
        current_pos: u64,
        k: usize,
        pin_buf: &mut Vec<bool>,
        scores: &mut Vec<f32>,
        out: &mut Vec<usize>,
    ) {
        sink_pin_mask_into(policy, positions, current_pos, pin_buf);
        self.select_evict_into(k, pin_buf, scores, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(n: usize, cfg: DiffEvictConfig) -> DifferentialEvictTable {
        let mut t = DifferentialEvictTable::with_capacity(n, cfg);
        t.admit_prefix(n);
        t
    }

    #[test]
    fn hub_loses_specificity_spike_keeps_it() {
        // slot 0 = hub (0.4 every query), slot 1 = one spike of 0.3 at the
        // last query, slot 2 = flat filler. Window 4 × 21 queries: the
        // admission bucket (μ = 0, the recency prior) has aged out.
        let mut t = table(3, DiffEvictConfig::new(1.0, 0.5, 4));
        for _ in 0..20 {
            t.observe_query(&[0.4, 0.01, 0.01]);
        }
        t.observe_query(&[0.4, 0.3, 0.01]);
        assert!(t.specificity(1) > t.specificity(0));
        // Baseline (λ=0) ranks the hub first.
        let mut b = table(3, DiffEvictConfig::max_recent(4));
        for _ in 0..20 {
            b.observe_query(&[0.4, 0.01, 0.01]);
        }
        b.observe_query(&[0.4, 0.3, 0.01]);
        assert!(b.specificity(0) > b.specificity(1));
    }

    #[test]
    fn spike_does_not_cancel_itself() {
        // μ(t−1) reference: a first-query spike keeps its full mass.
        let mut t = table(1, DiffEvictConfig::new(1.0, 1.0, 8));
        t.observe_query(&[0.7]);
        assert_eq!(t.specificity(0), 0.7);
        assert_eq!(t.mass_ema(0), 0.7);
    }

    #[test]
    fn lambda_zero_is_exact_max() {
        let mut t = table(2, DiffEvictConfig::max_recent(4));
        let rows = [[0.1f32, 0.2], [0.5, 0.05], [0.3, 0.3]];
        for r in &rows {
            t.observe_query(r);
        }
        assert_eq!(t.specificity(0).to_bits(), 0.5f32.to_bits());
        assert_eq!(t.specificity(1).to_bits(), 0.3f32.to_bits());
    }

    #[test]
    fn two_bucket_window_ages_out() {
        let mut t = table(1, DiffEvictConfig::max_recent(2));
        t.observe_query(&[0.9]); // q0, bucket 0
        t.observe_query(&[0.1]); // q1, bucket 0
        t.observe_query(&[0.2]); // q2, bucket 1 (rotate: prev = 0.9)
        assert_eq!(t.specificity(0), 0.9);
        t.observe_query(&[0.2]); // q3, bucket 1
        t.observe_query(&[0.3]); // q4, bucket 2 (rotate: prev = 0.2) — 0.9 gone
        assert_eq!(t.specificity(0), 0.3);
    }

    #[test]
    fn short_mass_row_still_rotates_missing_slots() {
        let mut t = table(2, DiffEvictConfig::max_recent(1));
        t.observe_query(&[0.9, 0.9]); // bucket 0
        t.observe_query(&[0.1]); // bucket 1: slot 1 unsampled, still rotates
        t.observe_query(&[0.1]); // bucket 2: slot 1's 0.9 ages out
        assert_eq!(t.specificity(0), 0.1);
        assert_eq!(t.specificity(1), f32::NEG_INFINITY);
    }

    #[test]
    fn bad_mass_ignored_mu_stays_finite() {
        let mut t = table(2, DiffEvictConfig::new(0.8, 0.5, 4));
        t.observe_query(&[0.2, 0.2]);
        t.observe_query(&[f32::NAN, -1.0]);
        t.observe_query(&[f32::INFINITY, 0.2]);
        assert!(t.mass_ema(0).is_finite() && t.mass_ema(1).is_finite());
        // 0 + 0.5·(0.2 − 0) after the first query; the bad samples left it.
        assert_eq!(t.mass_ema(0), 0.1);
    }

    #[test]
    fn unobserved_slot_orders_first_pinned_never() {
        let mut t = table(3, DiffEvictConfig::new(0.8, 0.2, 8));
        t.observe_query(&[0.3, 0.3, 0.3]);
        t.reset_row(1); // re-admitted, no evidence
        let (mut s, mut out) = (Vec::new(), Vec::new());
        t.select_evict_into(1, &[], &mut s, &mut out);
        assert_eq!(out, vec![1]);
        t.select_evict_into(3, &[false, true, false], &mut s, &mut out);
        assert!(!out.contains(&1));
    }

    #[test]
    fn sink_exempt_selection_uses_shipped_rule() {
        // λ > 1 (over-cancellation) — where a sink's specificity goes
        // negative (Bench 894 G1c). Window 4 so the admission bucket ages out.
        let mut t = table(6, DiffEvictConfig::new(1.5, 0.5, 4));
        // slots 0,1 are sinks: the biggest hubs, lowest specificity.
        for _ in 0..11 {
            t.observe_query(&[0.4, 0.4, 0.05, 0.05, 0.05, 0.05]);
        }
        t.observe_query(&[0.4, 0.4, 0.05, 0.2, 0.05, 0.05]);
        let positions: Vec<u64> = (0..6).collect();
        let pol = SinkWindowPolicy::new(2, 1024);
        let (mut pins, mut s, mut out) = (Vec::new(), Vec::new(), Vec::new());
        t.select_evict_sink_exempt(&pol, &positions, 6, 3, &mut pins, &mut s, &mut out);
        assert_eq!(out.len(), 3);
        assert!(!out.contains(&0) && !out.contains(&1) && !out.contains(&3));
        // Without the pin the sinks go first.
        t.select_evict_into(2, &[], &mut s, &mut out);
        assert!(out.contains(&0) && out.contains(&1));
    }

    #[test]
    fn no_eviction_budget_is_zero_k() {
        assert_eq!(evictions_for_budget(100, 100), 0);
        assert_eq!(evictions_for_budget(100, 1_000), 0);
        assert_eq!(evictions_for_budget(100, 75), 25);
        let t = table(4, DiffEvictConfig::new(0.8, 0.1, 8));
        let (mut s, mut out) = (Vec::new(), vec![7usize]);
        t.select_evict_into(0, &[], &mut s, &mut out);
        assert!(out.is_empty());
    }
}
