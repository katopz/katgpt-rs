//! Continuous E2LSH per-axis addressing (Plan 619 T1.2; Issue 938 T2.1).
//!
//! The REJECTED form is recorded in the plan: integer-split SimHash (sign-bit
//! split → two integers) — a 1-bit high-order change moves |Δx| ≫ 1 (lattice
//! neighbors ≠ embedding neighbors) and yields no fractional part, killing
//! the bump. The adopted form: per axis ℓ, the coordinate of embedding `x`
//! is `(a_ℓ·x + b_ℓ) / w_ℓ` with a seeded p-stable (Gaussian) `a_ℓ`, uniform
//! `b_ℓ ∈ [0, w_ℓ)` — the integer part is the cell coordinate, the
//! **fractional part is the bump input** (real, non-zero — the property the
//! rejected form could not give).
//!
//! The axis count is a **const generic** `A`, not a constant (Issue 938
//! T2.1): `A = 2` is the shipped Plan 619 lattice, `A = 3` is the second
//! capacity remedy behind `lattice_memory_3d`. The draw and accumulation
//! orders below are written so the `A = 2` monomorphization is
//! bit-identical to the pre-generic form (same `(u1, u2)` pairs in the same
//! order, per-axis adds in the same row order).
//!
//! Seeds follow the shared `crate::lsh_seed::config_seed` discipline (the
//! Issue 809 law, moved down from `katgpt-pruners/lsh_cache.rs`): same
//! configuration → same projection stream, every process.

use crate::lsh_seed::config_seed;

/// Seeded per-axis continuous-LSH projector, generic over the axis count.
///
/// One Gaussian direction `a_ℓ ∈ R^d` per axis, one offset `b_ℓ ∈ [0, w_ℓ)`,
/// one width per axis. Draws come from a single `fastrand::Rng` seeded by
/// FNV-1a over the configuration scalars; construction is O(A · d) and
/// happens once. `A` is inferred at 2D call sites from the `[f32; 2]`
/// width argument — existing call sites compile unchanged.
#[derive(Clone, Debug)]
pub struct LshAxes<const A: usize> {
    /// Per-embedding-dim projection entries, all axes at once (row-major:
    /// one row per `x` element, `[axis0, .., axisA−1]` per row) — one pass
    /// over `x` computes every coordinate.
    a: Vec<[f32; A]>,
    b: [f32; A],
    w: [f32; A],
    d: usize,
}

impl<const A: usize> LshAxes<A> {
    /// Build the seeded axes for a `d`-dim embedding and per-axis widths.
    ///
    /// `parts` extends the seed feed — callers fold THEIR distinguishing
    /// scalars in (the lattice folds grid + widths; a second consumer folds
    /// its own) so two configs never share a projection stream by accident.
    #[must_use]
    pub fn new(d: usize, w: [f32; A], seed: u64, parts: &[u64]) -> Self {
        let mut feed = Vec::with_capacity(parts.len() + 2 + A);
        feed.push(d as u64);
        feed.push(seed);
        for axis_w in w {
            feed.push(u64::from(axis_w.to_bits()));
        }
        feed.extend_from_slice(parts);
        let mut rng = fastrand::Rng::with_seed(config_seed(&feed));
        // Gaussian directions via Box–Muller from the seeded uniform stream —
        // deterministic given the seed, zero new deps (fastrand is already a
        // katgpt-core dep). One (u1, u2) pair yields TWO independent
        // Gaussians (the cos and sin arms); rows fill pairwise and truncate
        // at A, so the A = 2 stream is exactly the pre-generic one.
        let mut a = Vec::with_capacity(d);
        for _ in 0..d {
            let mut row = [0.0_f32; A];
            let mut filled = 0;
            while filled < A {
                let u1 = rng.f32().max(f32::EPSILON);
                let u2 = rng.f32();
                let mag = (-2.0 * u1.ln()).sqrt();
                let angle = core::f32::consts::TAU * u2;
                row[filled] = mag * angle.cos();
                filled += 1;
                if filled < A {
                    row[filled] = mag * angle.sin();
                    filled += 1;
                }
            }
            a.push(row);
        }
        let mut b = [0.0_f32; A];
        for (axis, bw) in b.iter_mut().enumerate() {
            *bw = rng.f32() * w[axis];
        }
        Self { a, b, w, d }
    }

    /// Embedding dimension.
    #[must_use]
    pub const fn d(&self) -> usize {
        self.d
    }

    /// Every axis coordinate for `x` — ONE pass over the projection rows
    /// computes all axes (the hot path never walks `x` twice).
    ///
    /// The inner loop is branch-free and A-wide per row (LLVM unrolls it at
    /// every monomorphization); `x` shorter than `d` is treated as
    /// zero-padded, longer is truncated — the caller's config `d` is the
    /// contract.
    #[must_use]
    pub fn coordinates(&self, x: &[f32]) -> [f32; A] {
        let n = self.d.min(x.len());
        let mut acc = [0.0_f32; A];
        for (i, row) in self.a[..n].iter().enumerate() {
            let x_i = x[i];
            for (axis, acc_axis) in acc.iter_mut().enumerate() {
                *acc_axis += row[axis] * x_i;
            }
        }
        let mut out = [0.0_f32; A];
        for axis in 0..A {
            out[axis] = (acc[axis] + self.b[axis]) / self.w[axis];
        }
        out
    }
}

/// Integer + fractional split of a continuous coordinate.
///
/// `floor` is the cell coordinate (clamped into the grid by the caller),
/// `frac ∈ [0, 1)` is the bump input — the real fractional part the rejected
/// SimHash form could not produce.
#[must_use]
pub fn split_coordinate(c: f32) -> (i64, f32) {
    let floored = c.floor();
    let frac = c - floored;
    (floored as i64, frac)
}

#[cfg(test)]
mod tests {
    use super::{LshAxes, split_coordinate};

    fn axes_for<const A: usize>(d: usize, w: [f32; A], seed: u64) -> LshAxes<A> {
        LshAxes::new(d, w, seed, &[7, 15])
    }

    #[test]
    fn same_config_same_axes() {
        // The Issue 809 law at the lattice: identical configs → identical
        // projection streams → identical coordinates, bit-for-bit.
        let a = axes_for(16, [4.0, 4.0], 42);
        let b = axes_for(16, [4.0, 4.0], 42);
        let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 1.5).collect();
        assert_eq!(a.coordinates(&x), b.coordinates(&x));
    }

    #[test]
    fn different_seed_different_axes() {
        let a = axes_for(16, [4.0, 4.0], 42);
        let b = axes_for(16, [4.0, 4.0], 43);
        let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 1.5).collect();
        assert_ne!(a.coordinates(&x), b.coordinates(&x));
    }

    /// The generic refactor's A = 3 arm: the same laws hold on the wider
    /// lattice (Issue 938 T2.1 — the addressing layer is arity-blind).
    #[test]
    fn three_axis_laws_hold() {
        let a = axes_for(16, [4.0, 4.0, 4.0], 42);
        let b = axes_for(16, [4.0, 4.0, 4.0], 42);
        let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 1.5).collect();
        assert_eq!(a.coordinates(&x), b.coordinates(&x));
        let c = axes_for(16, [4.0, 4.0, 4.0], 43);
        assert_ne!(a.coordinates(&x), c.coordinates(&x));
        // Locality per axis at A = 3 (the T1.2 premise, spot-checked — the
        // full 95% sweep stays the A = 2 gate below).
        let mut close = 0_u32;
        let trials = 200;
        let mut rng = fastrand::Rng::with_seed(0x938_0001);
        for _ in 0..trials {
            let x: Vec<f32> = (0..16).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let mut x2 = x.clone();
            for v in &mut x2 {
                *v += 0.05 * (rng.f32() * 2.0 - 1.0);
            }
            for axis in 0..3 {
                let dax = (a.coordinates(&x)[axis] - a.coordinates(&x2)[axis]).abs();
                if dax <= 1.0 + f32::EPSILON {
                    close += 1;
                }
            }
        }
        assert!(
            close as f32 / (3 * trials) as f32 >= 0.95,
            "3-axis locality broke: {close} of {} axis-pairs within ±1 cell",
            3 * trials
        );
    }

    #[test]
    fn locality_small_perturbation_stays_close() {
        // T1.2's load-bearing property: a small embedding perturbation lands
        // in the same or an adjacent cell with high probability — per axis,
        // across seeds. Measured over 200 trials × 8 seeds: |Δcell| ≤ 1 must
        // dominate (≥ 95%); a failure here is the plan's STOP gate.
        let d = 32;
        let trials = 200;
        let eps = 0.05_f32;
        let mut close = 0_u32;
        let mut total = 0_u32;
        for seed in 0..8_u64 {
            let axes = axes_for(d, [2.0, 2.0], seed);
            let mut rng = fastrand::Rng::with_seed(seed ^ 0x9e37_79b9);
            for _ in 0..trials {
                let x: Vec<f32> = (0..d).map(|_| rng.f32() * 2.0 - 1.0).collect();
                let mut x2 = x.clone();
                for v in &mut x2 {
                    *v += eps * (rng.f32() * 2.0 - 1.0);
                }
                let (c0, c1) = { (axes.coordinates(&x), axes.coordinates(&x2)) };
                for axis in 0..2 {
                    let delta = (c0[axis] - c1[axis]).abs();
                    if delta <= 1.0 + f32::EPSILON {
                        close += 1;
                    }
                    total += 1;
                }
            }
        }
        let rate = f64::from(close) / f64::from(total);
        assert!(
            rate >= 0.95,
            "locality FAILED the plan's premise: same/adjacent-cell rate {rate:.4} < 0.95 \
             — per Plan 619 T1.2 this STOPS the lattice claim (restate as Engram+delta-cells)"
        );
    }

    #[test]
    fn fractional_part_not_degenerate() {
        // The bump's input must be a real, spread distribution — not
        // pinned at 0/½, not collapsed. Chi-square-lite over 16 bins.
        let d = 32;
        let axes = axes_for(d, [2.0, 2.0], 7);
        let mut rng = fastrand::Rng::with_seed(0xDECA_FBAD);
        let bins = 16;
        let mut hist = [0_u32; 16];
        let n = 4000;
        for _ in 0..n {
            let x: Vec<f32> = (0..d).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let c = axes.coordinates(&x);
            let (_, frac) = split_coordinate(c[0]);
            let bin = ((frac * bins as f32) as usize).min(bins - 1);
            hist[bin] += 1;
        }
        let expected = n / bins as u32;
        for (i, &count) in hist.iter().enumerate() {
            let share = f64::from(count) / f64::from(expected);
            // A degenerate distribution pins ≥ 2× expected in one bin; a
            // healthy uniform-ish spread stays within ±25% per bin.
            assert!(
                (0.75..=1.25).contains(&share),
                "fractional-part bin {i} at {share:.2}× expected — distribution degenerate"
            );
        }
    }

    #[test]
    fn split_is_exact_at_integers_and_fraction_in_range() {
        assert_eq!(split_coordinate(3.0), (3, 0.0));
        assert_eq!(split_coordinate(-0.25), (-1, 0.75));
        let (i, f) = split_coordinate(-2.5);
        assert_eq!((i, f), (-3, 0.5));
        let (i, f) = split_coordinate(17.125);
        assert_eq!((i, f), (17, 0.125));
    }
}
