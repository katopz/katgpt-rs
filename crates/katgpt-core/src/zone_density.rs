//! Zone Density Routing — modelless physical compute scheduler.
//!
//! Turns a per-zone population count into a compute-scheduling decision via
//! three pure primitives:
//!
//! 1. [`zone_density_classify`] — per-zone `(mobility, tier, cache_key)` from
//!    raw population. Mobility `m(ρ) = fast_sigmoid(-β·(ρ−ρ₀))` is monotone in
//!    density (sparse → high mobility → full compute; dense → low mobility →
//!    cached). The cache key is a composite `(tier << 32 | density_bucket)`.
//! 2. [`schedule_outer_first`] — stable ascending-density sort of zone indices.
//!    Outer (sparse, high-entropy) zones compute first so their entropy
//!    contributes to the mean before dense zones are batched.
//! 3. [`ZoneDensityCache`] — `papaya`-backed lock-free per-zone LRU with three
//!    invalidation rules: tier transition, density drift > δ, TTL expiry.
//!
//! # Source
//!
//! Distilled from Treuille, Cooper, Popović (2006) *"Continuum Crowds"* (SIGGRAPH),
//! van Toll et al. density-aware navigation meshes, and the Fokker-Planck /
//! continuity equation on cochains already shipped in `katgpt_dec::stokes_calculus`
//! (Plan 314). See `.research/350_density_aware_compute_scheduling.md` and
//! `.plans/351_density_aware_zone_routing.md` for the full derivation.
//!
//! # Sibling, not replacement
//!
//! This primitive gates **physical** compute (mobility/tier/cache). The existing
//! `zone_manifold` module (Plan 305 cognitive gating) gates **cognitive**
//! compute (tau/beta/budget). The two compose orthogonally as siblings; they do
//! not overlap. Wire both in `NpcFunctorRuntime` (Plan 351 Phase 4).
//!
//! # Latent vs raw boundary
//!
//! - **Raw / synced**: `population: &[f32]` (per-zone head count). Syncs via
//!   `SyncBlock`, deterministic replay, anti-cheat. Bit-identical across nodes.
//! - **Latent / local**: `mobility: f32`, `tier: DensityTier`, `cache_key: u64`.
//!   These are deterministic *derivations* of raw population; they never cross
//!   the sync boundary. The 5 synced affect scalars (valence/arousal/...) stay
//!   in their existing sync envelope — this primitive does not extend it.
//!
//! # Determinism
//!
//! All arithmetic is IEEE-754 `f32` with a fixed operation order:
//! `fast_sigmoid(-beta * (rho - rho0))`. The same `population + config` pair
//! yields bit-identical `mobility / tier / cache_key` on every call across
//! `x86_64` / `aarch64` / `wasm32`. Cache ordering is **not** deterministic
//! (lock-free map), but cache *contents* are (same inserts → same entries).
//!
//! # Zero-alloc hot path
//!
//! [`zone_density_classify`] and [`schedule_outer_first`] perform **no heap
//! allocation after warmup** — all output lives in caller-owned slices, all sort
//! scratch lives in a caller-owned [`Vec`] that is `clear()`ed + reused.
//! [`ZoneDensityCache`] is the only allocator on this path (papaya's lock-free
//! table), and only on `insert` / `invalidate` (the `get` path is read-only).
//!
//! # No UQ claim
//!
//! Mobility is a deterministic weight in `[0, 1]`, **not** a probability,
//! predictive interval, quantile, coverage guarantee, or calibrated uncertainty.
//! The Plan 340 "Report the Floor" conformal-naive baseline requirement does
//! **not** apply. Documented explicitly to prevent future reviewers from
//! re-introducing the floor requirement by mistake.
//!
//! Feature-gated behind `#[cfg(feature = "zone_density_routing")]`.

use crate::simd;
use papaya::HashMap;

// ── Tier enum ──────────────────────────────────────────────────

/// Per-zone physical compute tier. **Distinct from `ZoneGatingTier`** (Plan 305)
/// which is cognitive. This is physical: dense = cached (NPCs can't move),
/// sparse = full compute (high movement freedom).
///
/// Field-less + `#[repr(u8)]` per AGENTS.md (1-byte size, sync-friendly).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DensityTier {
    /// Sparse periphery — full compute every tick. High mobility, high entropy.
    Sparse = 0,
    /// Transitional — moderate compute, cached with short TTL.
    Transitional = 1,
    /// Dense core — LRU-cached, low compute. NPCs physically constrained.
    Dense = 2,
}

impl DensityTier {
    /// Reconstruct from the high 32 bits of a [`zone_density_classify`]-emitted
    /// cache key. Returns `None` for an out-of-range discriminant (defensive —
    /// should never happen for keys produced by this module).
    #[inline]
    pub fn from_cache_key_high(key: u64) -> Option<Self> {
        match (key >> 32) as u8 {
            0 => Some(Self::Sparse),
            1 => Some(Self::Transitional),
            2 => Some(Self::Dense),
            _ => None,
        }
    }
}

// ── Config ─────────────────────────────────────────────────────

/// Caller-supplied parameters for [`zone_density_classify`].
///
/// All fields have closed-form derivations from Plan 305's tier thresholds
/// (see Research 350 §2.2); no training is required to tune them.
#[derive(Clone, Copy, Debug)]
pub struct DensityClassifyConfig {
    /// Sigmoid midpoint density. Default `5.0` (Plan 305 midpoint between
    /// transitional=1.0 and dense=10.0). At `ρ = rho0`, mobility = 0.5 exactly.
    pub rho0: f32,
    /// Sigmoid slope. Default `0.5` — puts the 0.1→0.9 mobility transition
    /// across roughly one Plan-305 tier step.
    pub beta: f32,
    /// Mobility threshold above which a zone is [`DensityTier::Sparse`].
    /// Default `0.7`. Strict `>` (a zone with mobility exactly `tier_high`
    /// classifies as [`DensityTier::Transitional`]).
    pub tier_high: f32,
    /// Mobility threshold below which a zone is [`DensityTier::Dense`].
    /// Default `0.3`. Strict `<` (between `tier_low` and `tier_high`
    /// inclusive = [`DensityTier::Transitional`]).
    pub tier_low: f32,
    /// Density drift beyond which a cached entry is invalidated even if the
    /// tier hasn't changed. Default `2.0` (one Plan-305 tier step). Only read
    /// by [`ZoneDensityCache::get_or_invalidate`].
    pub cache_invalidation_delta: f32,
}

impl Default for DensityClassifyConfig {
    #[inline]
    fn default() -> Self {
        Self {
            rho0: 5.0,
            beta: 0.5,
            tier_high: 0.7,
            tier_low: 0.3,
            cache_invalidation_delta: 2.0,
        }
    }
}

// ── Report ─────────────────────────────────────────────────────

/// Per-tick summary returned by [`zone_density_classify`]. All counts are over
/// the input `population` slice.
#[derive(Debug, Default, Clone, Copy)]
pub struct DensityClassifyReport {
    /// Number of zones classified [`DensityTier::Sparse`].
    pub n_sparse: usize,
    /// Number of zones classified [`DensityTier::Transitional`].
    pub n_transitional: usize,
    /// Number of zones classified [`DensityTier::Dense`].
    pub n_dense: usize,
    /// Arithmetic mean of per-zone mobilities. `0.0` for empty input.
    pub mean_mobility: f32,
}

// ── classify ───────────────────────────────────────────────────

/// Per-zone density → `(mobility, tier, cache_key)`. Deterministic, zero-alloc.
///
/// Single pass over `population`. For each zone `i`:
/// 1. `mobility[i] = fast_sigmoid(-beta * (rho - rho0))` — monotone decreasing.
/// 2. `tier[i]` via strict-threshold `match` (see [`DensityClassifyConfig`]).
/// 3. `cache_key[i] = ((tier as u64) << 32) | density_bucket` where
///    `density_bucket = floor(rho * 0.5)` (buckets of size 2.0, matching
///    Plan 305's tier-step granularity and the default
///    `cache_invalidation_delta`).
///
/// # Arguments
///
/// - `population` — per-zone head counts (raw, synced). Read-only.
/// - `config` — caller-supplied thresholds. Read-only.
/// - `out_mobility / out_tier / out_cache_key` — caller-owned output slices.
///   Only the first `population.len()` entries are written.
///
/// # Panics
///
/// Debug-asserts that all output slices are `>= population.len()`. Empty input
/// returns [`DensityClassifyReport::default`] and writes nothing.
///
/// # Determinism
///
/// Bit-identical across calls for the same `population + config` (IEEE-754 f32
/// with fixed operation order; `fast_sigmoid` is branch-bounded but order-stable).
pub fn zone_density_classify(
    population: &[f32],
    config: &DensityClassifyConfig,
    out_mobility: &mut [f32],
    out_tier: &mut [DensityTier],
    out_cache_key: &mut [u64],
) -> DensityClassifyReport {
    let n = population.len();
    debug_assert!(
        out_mobility.len() >= n,
        "out_mobility.len() = {} < population.len() = {}",
        out_mobility.len(),
        n
    );
    debug_assert!(
        out_tier.len() >= n,
        "out_tier.len() = {} < population.len() = {}",
        out_tier.len(),
        n
    );
    debug_assert!(
        out_cache_key.len() >= n,
        "out_cache_key.len() = {} < population.len() = {}",
        out_cache_key.len(),
        n
    );

    let mut report = DensityClassifyReport::default();
    if n == 0 {
        return report;
    }

    let mut mobility_sum: f32 = 0.0;
    for (i, &rho) in population.iter().enumerate() {
        let m = simd::fast_sigmoid(-config.beta * (rho - config.rho0));
        let tier = match m {
            x if x > config.tier_high => DensityTier::Sparse,
            x if x < config.tier_low => DensityTier::Dense,
            _ => DensityTier::Transitional,
        };
        // Buckets of size 2.0 (matches Plan 305 tier-step granularity). The
        // `.max(0.0)` is defensive — population is non-negative by domain
        // invariant, but a stray negative would otherwise saturate-cast to 0
        // anyway (Rust 1.45+), so we make the intent explicit.
        let density_bucket = ((rho * 0.5_f32).floor().max(0.0)) as u64;
        let cache_key = ((tier as u64) << 32) | density_bucket;

        out_mobility[i] = m;
        out_tier[i] = tier;
        out_cache_key[i] = cache_key;
        mobility_sum += m;
        match tier {
            DensityTier::Sparse => report.n_sparse += 1,
            DensityTier::Transitional => report.n_transitional += 1,
            DensityTier::Dense => report.n_dense += 1,
        }
    }
    report.mean_mobility = mobility_sum / (n as f32);
    report
}

/// Decode a [`zone_density_classify`]-emitted cache key back into its
/// `(tier, density_bucket)` components. Inverse of the `cache_key` formula.
///
/// Returns `None` if the tier discriminant is out of range (defensive — should
/// never happen for keys produced by this module, but safe for arbitrary u64).
#[inline]
pub fn decode_cache_key(key: u64) -> Option<(DensityTier, u64)> {
    let tier = DensityTier::from_cache_key_high(key)?;
    let density_bucket = key & 0xFFFF_FFFF;
    Some((tier, density_bucket))
}

// ── scheduler ──────────────────────────────────────────────────

/// Sort zone indices ascending by density. Outer (sparse, high-mobility,
/// high-entropy) zones come first. `O(Z log Z)`, **stable**.
///
/// Stable sort preserves within-tier ordering for determinism: two zones with
/// the same density retain their original index order in `out_order`. This is
/// the contract that lets callers compose this with other ordering signals
/// (e.g., the Plan 305 cognitive tier) without breaking reproducibility.
///
/// # Arguments
///
/// - `population` — per-zone head counts. Read-only.
/// - `out_order` — caller-owned output slice; receives the sorted zone indices.
///   Only the first `population.len()` entries are written.
/// - `scratch` — caller-owned `(zone_idx, density)` buffer, `clear()`ed and
///   reused. Pass the same `Vec` across ticks to avoid reallocation.
///
/// # Panics
///
/// Debug-asserts `out_order.len() >= population.len()`. Empty input writes
/// nothing and does not panic.
pub fn schedule_outer_first(
    population: &[f32],
    out_order: &mut [u32],
    scratch: &mut Vec<(u32, f32)>,
) {
    let n = population.len();
    debug_assert!(
        out_order.len() >= n,
        "out_order.len() = {} < population.len() = {}",
        out_order.len(),
        n
    );

    scratch.clear();
    scratch.reserve(n);
    for (i, &rho) in population.iter().enumerate() {
        scratch.push((i as u32, rho));
    }
    // `sort_by` is stable by contract — equal-density zones keep input order.
    // `partial_cmp(...).unwrap_or(Equal)` defends against NaN (shouldn't occur
    // for population counts, but the FFI boundary doesn't enforce it).
    scratch.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    for (i, &(z, _)) in scratch.iter().enumerate() {
        out_order[i] = z;
    }
}

// ── cache ──────────────────────────────────────────────────────

/// Per-zone LRU cache for transitional/dense-tier values. Lock-free via papaya.
///
/// # Invariant: sparse-tier zones are NEVER cached
///
/// [`Self::insert`] silently drops any value whose tier is
/// [`DensityTier::Sparse`]; [`Self::get_or_invalidate`] short-circuits to
/// `None` for sparse-tier lookups. Sparse zones always recompute from scratch
/// (their high mobility makes cached values stale almost immediately).
///
/// # Three invalidation rules (checked in order)
///
/// For a `get_or_invalidate` to return `Some(value)`, ALL of:
/// 1. **Tier stability**: `cached_tier == current_tier`. A tier transition
///    (e.g., zone crossed the 0.3 or 0.7 mobility threshold) invalidates.
/// 2. **Density drift**: `|cached_density - current_density| <= delta`. Large
///    intra-tier drift invalidates even without a tier flip.
/// 3. **TTL**: `current_tick <= cached_at_tick + ttl_ticks`. Stale entries
///    expire even if the zone is otherwise stable.
///
/// A failed rule **removes** the entry (lazy eviction on read), not just
/// returns `None` — so the next `insert` starts clean.
///
/// # Concurrency
///
/// papaya is lock-free and `Send + Sync`. Reads and writes may interleave
/// across threads; an `insert` racing a `get_or_invalidate` on the same zone
/// yields either the old or new value (last-writer-wins), never a corrupt
/// entry. The bulk [`Self::invalidate_all`] (called on stampede detection) is
/// linearizable with respect to concurrent operations.
pub struct ZoneDensityCache<V: Clone> {
    map: HashMap<u32, CacheEntry<V>>,
    ttl_ticks: u64,
}

#[derive(Clone)]
struct CacheEntry<V> {
    value: V,
    cached_density: f32,
    cached_tier: DensityTier,
    cached_at_tick: u64,
}

impl<V: Clone> ZoneDensityCache<V> {
    /// Construct an empty cache with the given TTL window (in ticks).
    #[inline]
    pub fn new(ttl_ticks: u64) -> Self {
        Self {
            map: HashMap::new(),
            ttl_ticks,
        }
    }

    /// TTL window (ticks) configured at construction. Read-only.
    #[inline]
    pub fn ttl_ticks(&self) -> u64 {
        self.ttl_ticks
    }

    /// Returns the cached value for `zone_id` iff (1) `current_tier` is
    /// Transitional/Dense (sparse always misses), (2) the cached tier matches,
    /// (3) density drift is within `invalidation_delta`, and (4) the entry is
    /// within TTL. Any failed rule evicts the entry and returns `None`.
    ///
    /// `invalidation_delta` is taken per-call (not stored) so the caller can
    /// pass [`DensityClassifyConfig::cache_invalidation_delta`] or override it
    /// per workload (e.g., tighten during stampede recovery).
    pub fn get_or_invalidate(
        &self,
        zone_id: u32,
        current_density: f32,
        current_tier: DensityTier,
        current_tick: u64,
        invalidation_delta: f32,
    ) -> Option<V> {
        // Sparse-tier zones are NEVER cached — short-circuit before touching the map.
        if current_tier == DensityTier::Sparse {
            return None;
        }

        // Decide under the pin; act (remove) within the same pin. papaya 0.2's
        // Guard is `&self`-borrowable for both `get` and `remove`, so we can
        // chain them without dropping the guard. The entry reference is valid
        // for the lifetime of the guard (epoch-based reclamation).
        let pin = self.map.pin();
        let entry = pin.get(&zone_id)?;

        let tier_changed = entry.cached_tier != current_tier;
        let drift = (entry.cached_density - current_density).abs();
        let drift_too_large = drift > invalidation_delta;
        let expired = current_tick > entry.cached_at_tick.saturating_add(self.ttl_ticks);

        if tier_changed || drift_too_large || expired {
            // Evict the stale entry so the next insert starts clean. We do not
            // read `entry` after this point (it may be reclaimed post-remove),
            // so the borrow is sound.
            pin.remove(&zone_id);
            return None;
        }

        Some(entry.value.clone())
    }

    /// Insert a value. **Sparse-tier zones are silently skipped** (never cached)
    /// — see the type-level invariant. Transitional/Dense tiers overwrite any
    /// prior entry for the same `zone_id`.
    pub fn insert(
        &self,
        zone_id: u32,
        density: f32,
        tier: DensityTier,
        tick: u64,
        value: V,
    ) {
        // Sparse-tier zones are NEVER cached.
        if tier == DensityTier::Sparse {
            return;
        }
        let entry = CacheEntry {
            value,
            cached_density: density,
            cached_tier: tier,
            cached_at_tick: tick,
        };
        self.map.pin().insert(zone_id, entry);
    }

    /// Bulk-invalidate all entries. Called on stampede detection (caller
    /// decides the trigger — typically `belief_mass_divergence > τ` from Plan
    /// 314, but this primitive is agnostic).
    ///
    /// Linearizable with concurrent `get`/`insert`. After this returns, the
    /// cache is empty.
    pub fn invalidate_all(&self) {
        self.map.pin().clear();
    }

    /// Current number of cached entries. For diagnostics / G5b benchmark.
    /// Lock-free; the count is a snapshot at the time of the call.
    pub fn len(&self) -> usize {
        self.map.pin().len()
    }

    /// `true` iff the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.pin().is_empty()
    }
}

// ── Phase 2 full test suite (T2.1–T2.4) ─────────────────────────
//
// ≥18 tests covering: monotonicity (20-point sweep), midpoint, tier
// boundaries (resolved from sigmoid inverse), cache key decode, determinism,
// empty input, single zone, NaN defense, stable sort, scratch reuse, sparse
// bypass, all three invalidation rules (tier / drift / TTL), TTL boundary,
// invalidate_all, and a concurrent-access smoke test (T2.3.8).

#[cfg(test)]
mod tests {
    use super::*;

    // ── zone_density_classify ──

    // ── T2.1.1 Monotonicity (20-point sweep) ──

    #[test]
    fn classify_mobility_monotone_decreasing_over_20_points() {
        // Plan T2.1.1: as ρ increases from 0 to 50, mobility must be
        // monotonically non-increasing. Sample 20 evenly-spaced points.
        let cfg = DensityClassifyConfig::default();
        let n = 20;
        let pop: Vec<f32> = (0..n).map(|i| (i as f32) * (50.0 / (n - 1) as f32)).collect();
        let mut mob = vec![0.0f32; n];
        let mut tier = vec![DensityTier::Transitional; n];
        let mut key = vec![0u64; n];
        let _ = zone_density_classify(&pop, &cfg, &mut mob, &mut tier, &mut key);
        for i in 1..n {
            assert!(
                mob[i] <= mob[i - 1] + 1e-6,
                "monotonicity broken at i={}: mob[{}]={} > mob[{}]={}",
                i, i - 1, mob[i - 1], i, mob[i]
            );
        }
        // Sanity: endpoints actually span the dynamic range.
        assert!(mob[0] > 0.9, "ρ=0 should give mobility near 1.0, got {}", mob[0]);
        assert!(mob[n - 1] < 0.05, "ρ=50 should give mobility near 0, got {}", mob[n - 1]);
    }

    // ── T2.1.2 Midpoint (mobility ≈ 0.5 at ρ = rho0) ──

    #[test]
    fn classify_midpoint_mobility_is_half() {
        // At ρ = rho0, sigmoid arg = 0 → sigmoid(0) = 0.5 exactly.
        let cfg = DensityClassifyConfig::default();
        let pop = [5.0f32];
        let mut mob = [0.0f32];
        let mut tier = [DensityTier::Dense];
        let mut key = [0u64];
        let _report = zone_density_classify(&pop, &cfg, &mut mob, &mut tier, &mut key);
        assert!(
            (mob[0] - 0.5).abs() < 1e-5,
            "midpoint mobility = {}, expected ~0.5",
            mob[0]
        );
        assert_eq!(tier[0], DensityTier::Transitional);
    }

    // ── T2.1.3 Tier boundaries resolved from sigmoid inverse ──

    #[test]
    fn classify_tier_boundaries_resolved_from_sigmoid_inverse() {
        // Plan T2.1.3: at mobility = tier_high (0.7) the zone is Transitional
        // (strict >), and just above tier_high it's Sparse. At mobility =
        // tier_low (0.3) the zone is Transitional (strict <), and just below
        // it's Dense. At mobility = 0.0 (saturated dense) it's Dense.
        //
        // Sigmoid inverse: ρ = rho0 - (1/β) · logit(m). logit(0.7) ≈ 0.8473.
        let cfg = DensityClassifyConfig::default();
        let logit = |m: f32| (m / (1.0 - m)).ln();
        let rho_at_high = cfg.rho0 - logit(cfg.tier_high) / cfg.beta;
        let rho_at_low = cfg.rho0 - logit(cfg.tier_low) / cfg.beta;

        // 4 cases: just-above-high, exactly-high, exactly-low, saturated-zero.
        let pop = [
            rho_at_high - 0.001, // mobility just above tier_high → Sparse
            rho_at_high,         // mobility exactly tier_high → Transitional (strict >)
            rho_at_low,          // mobility exactly tier_low → Transitional (strict <)
            100.0,               // saturated dense → mobility ~ 0 → Dense
        ];
        let mut mob = [0.0f32; 4];
        let mut tier = [DensityTier::Transitional; 4];
        let mut key = [0u64; 4];
        let _ = zone_density_classify(&pop, &cfg, &mut mob, &mut tier, &mut key);

        assert_eq!(tier[0], DensityTier::Sparse, "just-above-high must be Sparse (mob={})", mob[0]);
        assert_eq!(
            tier[1], DensityTier::Transitional,
            "mobility == tier_high must be Transitional (strict >), mob={}", mob[1]
        );
        assert_eq!(
            tier[2], DensityTier::Transitional,
            "mobility == tier_low must be Transitional (strict <), mob={}", mob[2]
        );
        assert_eq!(tier[3], DensityTier::Dense, "saturated dense (mob~0) must be Dense, mob={}", mob[3]);
        // Sanity: the resolved boundaries actually produce the expected mobilities.
        assert!((mob[1] - cfg.tier_high).abs() < 1e-4, "mob[1]={} ~= tier_high", mob[1]);
        assert!((mob[2] - cfg.tier_low).abs() < 1e-4, "mob[2]={} ~= tier_low", mob[2]);
    }

    // ── T2.1.4 Cache key round-trip ──

    #[test]
    fn classify_cache_key_round_trips() {
        let cfg = DensityClassifyConfig::default();
        let pop = [0.0f32, 5.0, 20.0];
        let mut mob = [0.0f32; 3];
        let mut tier = [DensityTier::Transitional; 3];
        let mut key = [0u64; 3];
        let _report = zone_density_classify(&pop, &cfg, &mut mob, &mut tier, &mut key);
        for i in 0..3 {
            let (decoded_tier, decoded_bucket) =
                decode_cache_key(key[i]).expect("in-range tier discriminant");
            assert_eq!(decoded_tier, tier[i], "tier mismatch at index {}", i);
            // Bucket = floor(ρ * 0.5).
            let expected_bucket = ((pop[i] * 0.5_f32).floor().max(0.0)) as u64;
            assert_eq!(
                decoded_bucket, expected_bucket,
                "bucket mismatch at index {}",
                i
            );
        }
    }

    // ── T2.1.5 Determinism ──

    #[test]
    fn classify_is_deterministic() {
        let cfg = DensityClassifyConfig::default();
        let pop = [0.5f32, 1.0, 2.0, 5.0, 10.0, 50.0];
        let mut mob_a = [0.0f32; 6];
        let mut tier_a = [DensityTier::Transitional; 6];
        let mut key_a = [0u64; 6];
        let mut mob_b = [0.0f32; 6];
        let mut tier_b = [DensityTier::Transitional; 6];
        let mut key_b = [0u64; 6];
        let _ra = zone_density_classify(&pop, &cfg, &mut mob_a, &mut tier_a, &mut key_a);
        let _rb = zone_density_classify(&pop, &cfg, &mut mob_b, &mut tier_b, &mut key_b);
        assert_eq!(mob_a, mob_b, "mobility must be bit-identical");
        assert_eq!(tier_a, tier_b, "tier must be identical");
        assert_eq!(key_a, key_b, "cache_key must be identical");
    }

    // ── T2.1.6 Empty input ──

    #[test]
    fn classify_empty_input_returns_default_no_panic() {
        let cfg = DensityClassifyConfig::default();
        let mut mob = [];
        let mut tier = [];
        let mut key = [];
        let report = zone_density_classify(&[], &cfg, &mut mob, &mut tier, &mut key);
        assert_eq!(report.n_sparse, 0);
        assert_eq!(report.n_transitional, 0);
        assert_eq!(report.n_dense, 0);
        assert_eq!(report.mean_mobility, 0.0);
    }

    // ── T2.1.7 NaN defense (FFI boundary) ──
    //
    // A stray NaN in `population` (shouldn't happen by domain invariant, but
    // the FFI boundary doesn't enforce it) must not panic. mobility will be
    // NaN-propagated, tier will be Transitional (no match arm fires for NaN),
    // and the cache key bucket is computed via `as u64` which saturates NaN
    // to 0 (Rust 1.45+ cast semantics).

    #[test]
    fn classify_nan_population_does_not_panic() {
        let cfg = DensityClassifyConfig::default();
        let pop = [f32::NAN, 5.0, f32::INFINITY];
        let mut mob = [0.0f32; 3];
        let mut tier = [DensityTier::Dense; 3];
        let mut key = [0u64; 3];
        // Must not panic.
        let _ = zone_density_classify(&pop, &cfg, &mut mob, &mut tier, &mut key);
        // NaN propagates into mobility (sigmoid(NaN) = NaN).
        assert!(mob[0].is_nan(), "NaN population → NaN mobility, got {}", mob[0]);
        // NaN matches no `>` / `<` arm → falls through to Transitional.
        assert_eq!(tier[0], DensityTier::Transitional, "NaN → Transitional fallback");
        // Infinity: -β·(∞ - ρ0) = -∞ → sigmoid(-∞) = 0.0 < 0.3 → Dense.
        assert_eq!(tier[2], DensityTier::Dense, "INFINITY → Dense");
        // Cache key bucket for NaN: `(NaN * 0.5).floor().max(0.0) as u64` → 0.
        let (decoded_tier, decoded_bucket) = decode_cache_key(key[0]).unwrap();
        assert_eq!(decoded_tier, DensityTier::Transitional);
        assert_eq!(decoded_bucket, 0, "NaN bucket saturates to 0");
    }

    // ── Endpoint sanity (complements T2.1.1 monotonicity + T2.1.3 boundaries) ──

    #[test]
    fn classify_extremes_land_in_correct_tier() {
        let cfg = DensityClassifyConfig::default();
        // ρ = 0 → sigmoid(+2.5) ≈ 0.924 > 0.7 → Sparse
        // ρ = 20 → sigmoid(-7.5) ≈ 5.5e-4 < 0.3 → Dense
        let pop = [0.0f32, 20.0];
        let mut mob = [0.0f32; 2];
        let mut tier = [DensityTier::Transitional; 2];
        let mut key = [0u64; 2];
        let report = zone_density_classify(&pop, &cfg, &mut mob, &mut tier, &mut key);
        assert_eq!(tier[0], DensityTier::Sparse, "ρ=0 should be Sparse");
        assert_eq!(tier[1], DensityTier::Dense, "ρ=20 should be Dense");
        assert_eq!(report.n_sparse, 1);
        assert_eq!(report.n_dense, 1);
        assert_eq!(report.n_transitional, 0);
        // Mobility is monotone decreasing in ρ.
        assert!(mob[0] > mob[1]);
    }

    // ── T2.2.1 Ascending order ──

    #[test]
    fn schedule_sorts_ascending_by_density() {
        let pop = [10.0f32, 1.0, 5.0, 0.5];
        let mut order = [0u32; 4];
        let mut scratch = Vec::new();
        schedule_outer_first(&pop, &mut order, &mut scratch);
        // Densities: idx3=0.5, idx1=1.0, idx2=5.0, idx0=10.0 → ascending.
        assert_eq!(order, [3, 1, 2, 0]);
    }

    // ── T2.2.2 Stable within ties ──

    #[test]
    fn schedule_is_stable_within_ties() {
        let pop = [5.0f32, 5.0, 5.0];
        let mut order = [0u32; 3];
        let mut scratch = Vec::new();
        schedule_outer_first(&pop, &mut order, &mut scratch);
        // All equal density → original order preserved (stable sort).
        assert_eq!(order, [0, 1, 2]);
    }

    // ── T2.2.3 Single zone ──

    #[test]
    fn schedule_single_zone_returns_identity() {
        let pop = [3.0f32];
        let mut order = [0u32; 1];
        let mut scratch = Vec::new();
        schedule_outer_first(&pop, &mut order, &mut scratch);
        assert_eq!(order, [0], "single zone → out_order = [0]");
    }

    // ── T2.2.4 Empty input ──

    #[test]
    fn schedule_empty_input_writes_nothing_no_panic() {
        let pop: [f32; 0] = [];
        let mut order = [99u32; 4]; // sentinel — must remain untouched
        let mut scratch = Vec::new();
        schedule_outer_first(&pop, &mut order, &mut scratch);
        assert_eq!(order, [99, 99, 99, 99], "empty input must not write anything");
        assert!(scratch.is_empty(), "scratch cleared after empty input");
    }

    // ── T2.2.5 Scratch reuse ──

    #[test]
    fn schedule_reuses_scratch_across_calls() {
        let pop = [3.0f32, 1.0, 2.0];
        let mut order = [0u32; 3];
        let mut scratch = Vec::new();
        // First call.
        schedule_outer_first(&pop, &mut order, &mut scratch);
        let first = order;
        // Second call with same scratch — must produce identical result.
        schedule_outer_first(&pop, &mut order, &mut scratch);
        assert_eq!(order, first);
        // Third call with different population, same scratch — no stale state.
        let pop2 = [9.0f32, 0.0, 4.0];
        schedule_outer_first(&pop2, &mut order, &mut scratch);
        assert_eq!(order, [1, 2, 0], "densities idx1=0, idx2=4, idx0=9 → [1,2,0]");
    }

    // ── ZoneDensityCache ──
    //
    // T2.3.1 sparse bypass · T2.3.2 transitional hit · T2.3.3 dense hit ·
    // T2.3.4 tier transition · T2.3.5 density drift · T2.3.6 TTL expiry ·
    // T2.3.7 invalidate_all · T2.3.8 concurrent access.

    #[test]
    fn cache_never_stores_sparse_tier() {
        let cache: ZoneDensityCache<String> = ZoneDensityCache::new(100);
        cache.insert(1, 0.5, DensityTier::Sparse, 0, "v".to_string());
        assert_eq!(cache.len(), 0, "Sparse insert must be dropped");
        // And get on Sparse tier short-circuits to None.
        let hit = cache.get_or_invalidate(1, 0.5, DensityTier::Sparse, 0, 2.0);
        assert!(hit.is_none());
    }

    #[test]
    fn cache_transitional_hit_within_all_rules() {
        let cache: ZoneDensityCache<u32> = ZoneDensityCache::new(100);
        cache.insert(7, 4.0, DensityTier::Transitional, 0, 42);
        let hit = cache.get_or_invalidate(7, 4.0, DensityTier::Transitional, 0, 2.0);
        assert_eq!(hit, Some(42));
    }

    #[test]
    fn cache_dense_hit_within_all_rules() {
        let cache: ZoneDensityCache<u32> = ZoneDensityCache::new(100);
        cache.insert(9, 15.0, DensityTier::Dense, 10, 99);
        let hit = cache.get_or_invalidate(9, 15.0, DensityTier::Dense, 10, 2.0);
        assert_eq!(hit, Some(99));
    }

    #[test]
    fn cache_tier_transition_invalidates() {
        let cache: ZoneDensityCache<u32> = ZoneDensityCache::new(100);
        cache.insert(3, 15.0, DensityTier::Dense, 0, 1);
        // Same density, different tier → miss + eviction.
        let hit = cache.get_or_invalidate(3, 15.0, DensityTier::Transitional, 0, 2.0);
        assert!(hit.is_none());
        assert_eq!(cache.len(), 0, "tier transition must evict");
    }

    #[test]
    fn cache_density_drift_invalidates() {
        let cache: ZoneDensityCache<u32> = ZoneDensityCache::new(100);
        cache.insert(5, 10.0, DensityTier::Dense, 0, 7);
        // Drift = |10 - 15| = 5 > delta=2 → miss + eviction.
        let hit = cache.get_or_invalidate(5, 15.0, DensityTier::Dense, 0, 2.0);
        assert!(hit.is_none());
        assert_eq!(cache.len(), 0, "drift > delta must evict");
    }

    #[test]
    fn cache_ttl_expiry_invalidates() {
        let cache: ZoneDensityCache<u32> = ZoneDensityCache::new(5);
        cache.insert(2, 10.0, DensityTier::Dense, 0, 1);
        // tick=0 + ttl=5 = expiry boundary at tick=5; tick=6 > 5 → miss.
        let hit = cache.get_or_invalidate(2, 10.0, DensityTier::Dense, 6, 2.0);
        assert!(hit.is_none());
        assert_eq!(cache.len(), 0, "TTL expiry must evict");
    }

    #[test]
    fn cache_ttl_boundary_is_inclusive() {
        let cache: ZoneDensityCache<u32> = ZoneDensityCache::new(5);
        cache.insert(2, 10.0, DensityTier::Dense, 0, 1);
        // tick=5 is exactly cached_at + ttl → NOT expired (strict >).
        let hit = cache.get_or_invalidate(2, 10.0, DensityTier::Dense, 5, 2.0);
        assert_eq!(hit, Some(1), "tick == cached_at + ttl must still hit");
    }

    #[test]
    fn cache_invalidate_all_clears_everything() {
        let cache: ZoneDensityCache<u32> = ZoneDensityCache::new(100);
        for z in 0..5u32 {
            cache.insert(z, 10.0, DensityTier::Dense, 0, z);
        }
        assert_eq!(cache.len(), 5);
        cache.invalidate_all();
        assert_eq!(cache.len(), 0);
        // Subsequent gets all miss.
        for z in 0..5u32 {
            let hit = cache.get_or_invalidate(z, 10.0, DensityTier::Dense, 0, 2.0);
            assert!(hit.is_none(), "zone {} should miss after invalidate_all", z);
        }
    }

    // ── T2.3.8 Concurrent access smoke test ──
    //
    // papaya's lock-free HashMap is `Send + Sync`. We exercise this with two
    // std::threads — one inserting, one getting — over 1000 iterations each
    // on a shared cache. No deadlock, no panic, no data race. We do NOT assert
    // specific values (interleaving is nondeterministic) — only that both
    // threads complete cleanly and the cache is non-empty at the end.

    #[test]
    fn cache_concurrent_access_no_deadlock_no_panic() {
        let cache: std::sync::Arc<ZoneDensityCache<u32>> =
            std::sync::Arc::new(ZoneDensityCache::new(1000));
        let cache_writer = cache.clone();
        let cache_reader = cache.clone();

        let writer = std::thread::spawn(move || {
            for i in 0..1000u32 {
                // Mix tiers; Sparse inserts are dropped by design (T2.3.1).
                let tier = match i % 3 {
                    0 => DensityTier::Dense,
                    1 => DensityTier::Transitional,
                    _ => DensityTier::Sparse,
                };
                cache_writer.insert(i, (i % 20) as f32, tier, 0, i);
            }
        });

        let reader = std::thread::spawn(move || {
            let mut total_hits = 0u32;
            let mut total_misses = 0u32;
            for i in 0..1000u32 {
                let tier = if i % 2 == 0 { DensityTier::Dense } else { DensityTier::Transitional };
                match cache_reader.get_or_invalidate(i, (i % 20) as f32, tier, 0, 2.0) {
                    Some(_) => total_hits += 1,
                    None => total_misses += 1,
                }
            }
            (total_hits, total_misses)
        });

        writer.join().expect("writer thread panicked");
        let (hits, misses) = reader.join().expect("reader thread panicked");
        // Both threads completed without panic. We don't assert on hit/miss
        // counts (interleaving-dependent) — only that the totals add up and
        // the cache is in a consistent state.
        assert_eq!(hits + misses, 1000, "reader must observe all 1000 zones");
        // Cache should have *some* non-sparse entries from the writer (666 of
        // 1000, modulo eviction races). At minimum it shouldn't be empty.
        assert!(!cache.is_empty() || hits > 0, "cache should be non-empty or hits observed");
    }
}
