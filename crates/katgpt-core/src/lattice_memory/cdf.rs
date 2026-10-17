//! Gaussian-CDF occupancy warp — Plan 619 T1.5.
//!
//! E2LSH projections of real (roughly Gaussian) embeddings are NOT uniform
//! per axis: with a fixed width the central cells overload to capacity long
//! before the outer ones, and a uniform Poisson sizing law under-predicts
//! collisions exactly where recall collapses. The fix is design-time (cheaper
//! than the histogram catch): map each projected axis through a **monotone
//! Gaussian CDF** fitted to the corpus's projected mean/spread before
//! quantizing. Monotone ⇒ 1-D locality is preserved (nearby projections map
//! to nearby coordinates, and the fractional part the bump consumes stays
//! real and non-degenerate away from the clamp plateaus); CDF ⇒ post-warp
//! marginals are ≈ uniform ⇒ the Poisson occupancy law the sizer
//! ([`super::sizer`]) solves against actually holds.
//!
//! Numerics: `erf` via Abramowitz–Stegun 7.1.26 (max abs error 1.5e-7) —
//! std-only, zero new deps. `f32::exp` is platform-libm in exactly the way
//! the Box–Muller draws in `address.rs` already are: same-platform
//! determinism is the contract (same-process, same-config ⇒ same lattice —
//! pinned by the tests), and this is not a cross-platform committed-value
//! path.

/// The clamped domain in sigmas. Beyond ±6 the CDF is within ~1e-8 of its
/// asymptote; clamping keeps `apply` total AND monotone (the clamp plateaus
/// are non-decreasing), which is the property locality needs — exactness of
/// the tail shape is irrelevant to occupancy shaping.
const Z_CLAMP: f32 = 6.0;

/// A fitted per-axis monotone Gaussian-CDF remap: raw projected coordinate →
/// uniform-in-`(0, cells)` coordinate.
///
/// `apply` is monotone non-decreasing (strictly increasing on the interior),
/// maps the fitted mean to `cells/2`, and clamps the tails. Fitted by
/// [`CdfWarp::fit`] from projected per-axis samples; carried in
/// [`LatticeConfig`](super::types::LatticeConfig) so a config fully
/// describes its lattice (the freeze/replay story).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CdfWarp {
    /// Fitted mean of the projected samples. Public: a hand-tuned warp is
    /// legitimate geometry (the sizer is the intended producer, not the
    /// only one).
    pub mean: f32,
    /// Fitted inverse population standard deviation (the `fit` producer
    /// guards it against a degenerate sample — `1 / max(std, 1e-12)`).
    pub inv_std: f32,
    /// The grid side this warp quantizes into; `apply` scales `[0, 1)` CDF
    /// mass to `[0, cells)`.
    pub cells: f32,
}

impl CdfWarp {
    /// Fit the warp to projected per-axis samples for a grid of `cells`
    /// cells on this axis.
    ///
    /// Deterministic: the fit is the plain mean + population standard
    /// deviation of the sample — same samples ⇒ same warp, every process.
    #[must_use]
    pub fn fit(projected: &[f32], cells: u32) -> Self {
        assert!(!projected.is_empty(), "CdfWarp::fit needs samples");
        assert!(cells > 0, "CdfWarp::fit needs a positive grid side");
        let n = projected.len() as f32;
        let mean = projected.iter().sum::<f32>() / n;
        let var = projected
            .iter()
            .map(|&s| (s - mean) * (s - mean))
            .sum::<f32>()
            / n;
        let std = var.max(0.0).sqrt().max(1e-12);
        Self {
            mean,
            inv_std: 1.0 / std,
            cells: cells as f32,
        }
    }

    /// The warped continuous coordinate in `[0, cells)` — the value the
    /// lattice splits into integer cell + fractional bump input.
    ///
    /// Monotone non-decreasing in `c` (strict on the interior; the ±`Z_CLAMP`
    /// plateaus are flat), total for every finite input.
    #[must_use]
    pub fn apply(&self, c: f32) -> f32 {
        let z = ((c - self.mean) * self.inv_std).clamp(-Z_CLAMP, Z_CLAMP);
        phi(z) * self.cells
    }

    /// The fitted mean (diagnostic readout).
    #[must_use]
    pub const fn mean(&self) -> f32 {
        self.mean
    }

    /// The fitted standard deviation (diagnostic readout).
    #[must_use]
    pub fn std(&self) -> f32 {
        1.0 / self.inv_std
    }

    /// The grid side this warp quantizes into.
    #[must_use]
    pub const fn cells(&self) -> f32 {
        self.cells
    }
}

/// Standard normal CDF `Φ(z)` from the A–S erf approximation.
#[must_use]
fn phi(z: f32) -> f32 {
    0.5 * (1.0 + erf(z * core::f32::consts::FRAC_1_SQRT_2))
}

/// Abramowitz–Stegun 7.1.26 `erf` (max abs error 1.5e-7), odd-symmetric.
/// The constants are a PUBLISHED TUPLE — their f32 sum at t = 1 (x = 0)
/// must be exactly 1 for `erf(0) = 0`; clippy's per-literal truncation
/// suggestion was tested and BREAKS that (erf(0) read 1.19e-7), so the
/// full-precision literals stay under a reasoned allow.
#[allow(clippy::excessive_precision)]
#[must_use]
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-ax * ax).exp())
}

#[cfg(test)]
mod tests {
    use super::{CdfWarp, erf, phi};

    #[test]
    fn erf_known_values_pinned() {
        assert_eq!(erf(0.0), 0.0);
        // A–S 7.1.26's documented accuracy class: |err| ≤ 1.5e-7.
        assert!(
            (erf(1.0) - 0.842_700_8).abs() < 2e-6,
            "erf(1) = {}",
            erf(1.0)
        );
        assert!((erf(-1.0) + 0.842_700_8).abs() < 2e-6);
        assert!((erf(2.0) - 0.995_322_3).abs() < 2e-6);
    }

    #[test]
    fn phi_known_values_pinned() {
        assert!((phi(0.0) - 0.5).abs() < 1e-7);
        // Φ(1.96) ≈ 0.975 — the classic two-sided 95% point.
        assert!((phi(1.96) - 0.975).abs() < 1e-3);
        assert!(phi(-8.0) < 1e-6 && phi(8.0) > 1.0 - 1e-6);
    }

    #[test]
    fn phi_monotone_across_domain() {
        // The load-bearing property: monotone ⇒ 1-D locality preserved under
        // the warp. Finite differences must never decrease. The interior is
        // strictly increasing; the ±Z_CLAMP plateaus are flat (allowed).
        let mut prev = phi(-7.0);
        let mut steps = 0_u32;
        let mut z = -7.0;
        while z <= 7.0 {
            let v = phi(z);
            assert!(v >= prev, "phi decreased at z={z}: {v} < {prev}");
            // Strict interior ends at |z| = 4: the A-S approximation's
            // absolute error (≤ 1.5e-7 on erf) dwarfs the true Φ past the
            // ~1e-6 tail, so strictness there would pin float noise, not
            // mathematics. Monotonicity (≥) still holds everywhere.
            if z.abs() < 4.0 {
                assert!(v > prev, "phi not strictly increasing at z={z}");
            }
            prev = v;
            z += 0.01;
            steps += 1;
        }
        assert!(steps > 1000, "sweep too coarse: {steps} steps");
    }

    #[test]
    fn fit_recovers_sample_moments_and_apply_maps_mean_to_half() {
        // A seeded sample with known moments (Irwin–Hall-ish sum of uniforms
        // → approximately Gaussian by CLT, deterministic, no libm draw).
        let mut rng = fastrand::Rng::with_seed(0x6196_1900);
        let mut samples = Vec::with_capacity(4000);
        for _ in 0..4000 {
            let s: f32 = (0..12).map(|_| rng.f32() - 0.5).sum();
            samples.push(s);
        }
        let warp = CdfWarp::fit(&samples, 64);
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!((warp.mean() - mean).abs() < 1e-6);
        // apply(mean) = Φ(0)·cells = cells/2.
        assert!((warp.apply(mean) - 32.0).abs() < 1e-4);
        // One sigma up → Φ(1)·cells ≈ 0.8413·cells.
        let one_up = mean + warp.std();
        assert!((warp.apply(one_up) - 0.841_344_7 * 64.0).abs() < 0.05);
        // Tails clamp inside [0, cells] — the deep-tail CDF saturates to
        // exactly its asymptote in f32, and `locate` clamps the boundary
        // cell anyway (the sizer gate relies on this, not on tail shape).
        let lo = warp.apply(mean - 100.0);
        let hi = warp.apply(mean + 100.0);
        assert!(
            (0.0..=64.0).contains(&lo) && (0.0..=64.0).contains(&hi),
            "{lo} {hi}"
        );
    }

    #[test]
    fn apply_monotone_and_locality_safe() {
        // Monotone across the whole operating range at warp resolution.
        let warp = CdfWarp {
            mean: 3.0,
            inv_std: 0.25,
            cells: 128.0,
        };
        let mut prev = warp.apply(-100.0);
        let mut c = -99.9;
        while c <= 100.0 {
            let v = warp.apply(c);
            assert!(v >= prev, "warp decreased at c={c}");
            prev = v;
            c += 0.1;
        }
    }
}
