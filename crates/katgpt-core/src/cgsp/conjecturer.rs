//! `PoolConjecturer` — reference `CuriosityConjecturer` impl (Plan 274 T1.2).
//!
//! Samples k candidates from a fixed direction pool using
//! priority-weighted roulette. Zero-allocation: writes into caller-provided
//! `out` slice and uses `ScratchBuffers::cdf_scratch` for the CDF.

use crate::cgsp::traits::CuriosityConjecturer;
use crate::cgsp::types::{Candidate, Direction, Priority, Target};

// ── PoolConjecturer ───────────────────────────────────────────────────────

/// Frozen conjecturer backed by a fixed pool of direction vectors.
///
/// Samples k candidates per cycle weighted by the current priority table.
/// Implements priority-weighted roulette with a self-contained splitmix64
/// RNG so behaviour is reproducible across runs given a seed.
///
/// `Debug` is derived so that wrappers (e.g. `DerivativeCuriosity` in the
/// `temporal_deriv` fusion) can also derive `Debug` without a manual impl.
#[derive(Debug)]
pub struct PoolConjecturer {
    /// Frozen pool of direction vectors (immutable after construction).
    pool: Vec<Direction>,
    /// RNG seed (splitmix64 state).
    rng_state: u64,
    /// Optional perturbation magnitude applied to each sampled direction
    /// (0.0 = sample verbatim from pool; default 0.0).
    perturbation: f32,
}

impl PoolConjecturer {
    /// Build a new pool conjecturer with the given RNG seed.
    pub fn new(pool: Vec<Direction>, seed: u64) -> Self {
        Self {
            pool,
            rng_state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            perturbation: 0.0,
        }
    }

    /// Set the perturbation magnitude applied to each sampled direction.
    ///
    /// `0.0` = sample verbatim (default). Small positive values perturb each
    /// coordinate by a uniform random value in `[-mag, +mag]`. Used by the
    /// collapse-aware exploration path to widen the sampling distribution.
    pub fn with_perturbation(mut self, magnitude: f32) -> Self {
        self.perturbation = magnitude.clamp(0.0, 1.0);
        self
    }

    /// Pool size.
    #[inline]
    pub fn pool_len(&self) -> usize {
        self.pool.len()
    }

    /// Borrow the underlying pool.
    #[inline]
    pub fn pool(&self) -> &[Direction] {
        &self.pool
    }

    /// Advance the internal RNG by one step and return the next u64.
    fn next_u64(&mut self) -> u64 {
        // splitmix64 — fast, deterministic, good enough for sampling.
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Sample a uniform f32 in `[0, 1)`.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        // Use the top 24 bits for a uniform value (better statistical
        // properties than using all 53 bits of an f64 from u64 halves).
        let u = self.next_u64() >> 40; // top 24 bits
        (u as f32) / ((1u64 << 24) as f32)
    }

    /// Build the priority-weighted CDF into `cdf_scratch`.
    ///
    /// Returns the total weight (also stored as the last CDF entry).
    fn build_cdf(priorities: &[Priority], cdf_scratch: &mut Vec<f32>) -> f32 {
        cdf_scratch.clear();
        cdf_scratch.reserve(priorities.len());
        let mut acc = 0.0f32;
        for &p in priorities {
            // Floor at tiny epsilon so a fully-degenerate table still samples.
            let w = if p.is_finite() && p > 0.0 { p } else { 1e-6 };
            acc += w;
            cdf_scratch.push(acc);
        }
        acc
    }

    /// Inverse-CDF sample: pick the smallest index `i` such that
    /// `cdf[i] >= u * total`. Binary search for speed when pool is large.
    #[inline]
    fn sample_index(cdf: &[f32], total: f32, u: f32) -> usize {
        let target = u * total;
        // Linear scan is faster than binary search for small pools (< 32).
        // The conjecturer pool is typically 8–64 arms.
        for (i, &c) in cdf.iter().enumerate() {
            if c >= target {
                return i;
            }
        }
        cdf.len().saturating_sub(1)
    }
}

impl CuriosityConjecturer for PoolConjecturer {
    fn sample_candidates(
        &mut self,
        _target: &Target,
        priorities: &[Priority],
        out: &mut [Candidate],
        cdf_scratch: &mut Vec<f32>,
    ) {
        let total = Self::build_cdf(priorities, cdf_scratch);
        // Defensive: if priority table is empty, write zeros.
        if cdf_scratch.is_empty() || total <= 0.0 || self.pool.is_empty() {
            let dim = self.pool.first().map(|d| d.dim()).unwrap_or(0);
            for slot in out.iter_mut() {
                // Resize in place to avoid per-slot allocation.
                slot.direction.coords.resize(dim, 0.0);
                slot.direction.coords.fill(0.0);
                slot.pool_index = usize::MAX;
            }
            return;
        }
        let dim = self.pool[0].dim();
        let mag = self.perturbation;
        for slot in out.iter_mut() {
            let u = self.next_f32();
            let idx = Self::sample_index(cdf_scratch, total, u);
            let idx = idx.min(self.pool.len() - 1);
            let dir = &self.pool[idx];
            if mag > 0.0 {
                // Perturb each coordinate by uniform noise in [-mag, +mag].
                // Use clone_from to reuse the slot's existing Vec capacity
                // rather than allocating a fresh Vec each call.
                slot.direction.coords.clone_from(&dir.coords);
                for c in slot.direction.coords.iter_mut() {
                    let n = (self.next_f32() - 0.5) * 2.0 * mag;
                    *c += n;
                }
            } else {
                // Reuse the slot's existing Vec capacity. clone_from resizes
                // in place and only reallocates if capacity is insufficient —
                // in steady state (same dim across cycles) this is zero-alloc.
                slot.direction.coords.clone_from(&dir.coords);
            }
            slot.pool_index = idx;
            // Suppress unused-warning for `dim` (kept for future shape checks).
            let _ = dim;
        }
    }

    #[inline]
    fn pool_size(&self) -> usize {
        self.pool.len()
    }

    #[inline]
    fn pool_directions(&self) -> &[Direction] {
        &self.pool
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cgsp::types::{Priority, ScratchBuffers};

    fn unit_direction(dim: usize, axis: usize) -> Direction {
        let mut coords = vec![0.0f32; dim];
        coords[axis.min(dim.saturating_sub(1))] = 1.0;
        Direction { coords }
    }

    #[test]
    fn samples_match_priority_weights() {
        // Build a 4-arm pool where arm 0 has 90% of the priority mass.
        let pool = vec![
            unit_direction(4, 0),
            unit_direction(4, 1),
            unit_direction(4, 2),
            unit_direction(4, 3),
        ];
        let mut conj = PoolConjecturer::new(pool, 42);
        let prios: Vec<Priority> = vec![0.9, 0.04, 0.03, 0.03];
        let target = Target::new(unit_direction(4, 0));
        let mut scratch = ScratchBuffers::new(4, 4);

        let mut counts = [0u32; 4];
        let trials = 4000u32;
        let mut buf = vec![
            Candidate::new(Direction::zeros(4), usize::MAX),
            Candidate::new(Direction::zeros(4), usize::MAX),
            Candidate::new(Direction::zeros(4), usize::MAX),
            Candidate::new(Direction::zeros(4), usize::MAX),
        ];
        for _ in 0..trials {
            conj.sample_candidates(&target, &prios, &mut buf, &mut scratch.cdf_scratch);
            for c in &buf {
                if c.pool_index < 4 {
                    counts[c.pool_index] += 1;
                }
            }
        }

        let total_samples = counts.iter().sum::<u32>() as f32;
        let p0 = counts[0] as f32 / total_samples;
        // Loose χ²-style check: arm 0 should dominate (~90%) with some slack.
        assert!(
            p0 > 0.85 && p0 < 0.95,
            "arm 0 frequency out of expected range: {p0}"
        );
    }

    #[test]
    fn zero_perturbation_is_verbatim() {
        let pool = vec![unit_direction(2, 0), unit_direction(2, 1)];
        let mut conj = PoolConjecturer::new(pool.clone(), 1);
        let prios = vec![1.0, 0.0]; // arm 0 only
        let target = Target::new(unit_direction(2, 0));
        let mut scratch = ScratchBuffers::new(2, 2);
        let mut buf = vec![
            Candidate::new(Direction::zeros(2), usize::MAX),
            Candidate::new(Direction::zeros(2), usize::MAX),
        ];
        conj.sample_candidates(&target, &prios, &mut buf, &mut scratch.cdf_scratch);
        for c in &buf {
            assert_eq!(c.pool_index, 0);
            assert_eq!(c.direction.coords, pool[0].coords);
        }
    }

    #[test]
    fn degenerate_priorities_still_sample() {
        // All-zero priorities should still produce samples (via epsilon floor).
        let pool = vec![unit_direction(2, 0), unit_direction(2, 1)];
        let mut conj = PoolConjecturer::new(pool, 1);
        let prios = vec![0.0, 0.0];
        let target = Target::new(unit_direction(2, 0));
        let mut scratch = ScratchBuffers::new(2, 2);
        let mut buf = vec![
            Candidate::new(Direction::zeros(2), usize::MAX),
            Candidate::new(Direction::zeros(2), usize::MAX),
        ];
        conj.sample_candidates(&target, &prios, &mut buf, &mut scratch.cdf_scratch);
        for c in &buf {
            assert!(c.pool_index < 2, "should always sample a valid arm");
        }
    }
}
