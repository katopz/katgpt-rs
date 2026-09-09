//! T4/M4 — Gardner capacity LUT (Issue 740, Research 541 M4; paper eq 26,
//! Appx E, Fig 6).
//!
//! The paper's capacity relation for a binary associative memory with
//! loading `γ` (patterns per neuron) and required margin `κ`:
//!
//! ```text
//! 1 / γ_c(κ) = (1 + κ²)·Φ(κ) + κ·φ(κ)
//! ```
//!
//! where `Φ` / `φ` are the standard normal CDF / PDF. `γ_c` decreases
//! monotonically from `γ_c(0) = 2` toward `0`, so the inverse `κ_max(γ)`
//! exists for `γ ∈ (0, 2]`. The basin-radius corollary (paper eq 26 /
//! CLT argument): flipping a fraction `ρ` of the inputs perturbs a
//! unit-variance pre-activation by `≈ 2√ρ`, so a stored pattern with margin
//! `κ` survives corruption when
//!
//! ```text
//! κ > 2√ρ   ⇔   ρ < (κ/2)²
//! ```
//!
//! # API shape (build-time LUT, O(1) queries)
//!
//! - [`gamma_capacity`] / [`phi_cdf`] / [`phi_pdf`] — the closed forms (f64).
//! - [`kappa_max_bisection`] — direct numeric inversion (the golden
//!   reference; f64 bisection to ~1e-12).
//! - [`kappa_max`] — the production path: a monotone grid of `γ_c(κ)`
//!   computed **once** into a `std::sync::OnceLock` LUT
//!   ([`KAPPA_GRID_POINTS`] points over `κ ∈ [0, KAPPA_MAX]`), queried by
//!   binary search + linear interpolation. Zero per-call allocation.
//! - [`basin_radius_bound`] / [`basin_radius_from_kappa`] — the `ρ` bound,
//!   clamped to `[0, 1]`.
//!
//! # Φ precision
//!
//! `erf` is evaluated with Abramowitz & Stegun 7.1.5 — the all-positive
//! confluent series `erf(x) = (2/√π)·x·e^{−x²}·Σ_{n≥0} (2x²)ⁿ/(1·3⋯(2n+1))`.
//! Every term is positive, so there is **no cancellation**; in f64 the
//! relative error is ≲1e-13 over the domain used here (κ ≤ 8). The golden
//! test compares the LUT against bisection of the *same* closed forms, so
//! the Φ approximation error cancels identically on both sides — the
//! measured <1e-6 relative error is pure interpolation error.

/// LUT resolution: uniform log-γ grid points over
/// `[ln γ_c(KAPPA_MAX), ln 2]`. κ(γ) is nearly linear in ln γ on BOTH ends
/// (κ ≈ γ^(−1/2) in the tail; κ ≈ (π/4)(2−γ) near γ = 2), so linear
/// interpolation on this axis keeps the relative error ≲1e-6 at 2048
/// points keeps the relative error ≲1e-6 (measured by the golden test;
/// 2048 points measured 1.0042e-6 at the κ→0 end — the one region with
/// visible curvature — so the grid doubles to clear the bar with margin).
/// A uniform-κ grid misses by ~1e-4 at small loads (measured: 9.7e-5 at
/// γ=0.031) because γ_c is flat there.
pub const KAPPA_GRID_POINTS: usize = 4096;

/// Largest κ in the LUT (`γ_c(8) ≈ 0.0294`; loads below that clamp at the
/// cap and the `ρ` bound saturates at 1).
pub const KAPPA_MAX: f64 = 8.0;

const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1/√(2π)

/// Standard normal PDF `φ(κ)`.
#[inline]
pub fn phi_pdf(kappa: f64) -> f64 {
    (-0.5 * kappa * kappa).exp() * FRAC_1_SQRT_2PI
}

/// Standard normal CDF `Φ(κ)` — A&S 7.1.5 all-positive series (see module
/// docs). Accurate to ≲1e-13 relative for `κ ∈ [0, 8]`; `Φ(x) = 1 − Φ(−x)`
/// for negative arguments.
pub fn phi_cdf(kappa: f64) -> f64 {
    if kappa < 0.0 {
        return 1.0 - phi_cdf(-kappa);
    }
    0.5 * (1.0 + erf_as715(kappa * std::f64::consts::FRAC_1_SQRT_2))
}

/// `erf(x)` via A&S 7.1.5 — the all-positive confluent series. Term
/// recurrence `t_{n+1} = t_n · 2x²/(2n+3)`; stops when the term drops below
/// `1e-17` of the running sum.
fn erf_as715(x: f64) -> f64 {
    let x2 = 2.0 * x * x;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    let mut denom = 3.0f64; // next odd denominator
    loop {
        term *= x2 / denom;
        if term < 1e-17 * sum || term == 0.0 {
            break;
        }
        sum += term;
        denom += 2.0;
    }
    FRAC_2_SQRT_PI * x * (-x * x).exp() * sum
}

/// `2/√π` — the std constant (avoids an approximate-literal constant in
/// source, which `clippy::approx_constant` denies under `-D warnings`).
const FRAC_2_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// Gardner capacity curve `γ_c(κ) = 1 / ((1+κ²)Φ(κ) + κφ(κ))`.
///
/// Domain `κ ≥ 0` (negative inputs are mirrored: `γ_c` is a function of the
/// margin magnitude). `γ_c(0) = 2` exactly.
#[inline]
pub fn gamma_capacity(kappa: f64) -> f64 {
    let kappa = kappa.abs();
    1.0 / ((1.0 + kappa * kappa) * phi_cdf(kappa) + kappa * phi_pdf(kappa))
}

/// Direct numeric inversion of `γ_c(κ) = γ` by bisection on `[0, KAPPA_MAX]`
/// (the golden reference — f64, ~50 iterations, tolerance `1e-13·κ`).
///
/// `γ ≥ 2 ⇒ 0` (no margin needed); `γ ≤ γ_c(KAPPA_MAX) ⇒ KAPPA_MAX` (the
/// grid cap — the bound saturates).
pub fn kappa_max_bisection(gamma: f64) -> f64 {
    if gamma >= 2.0 {
        return 0.0;
    }
    if gamma <= gamma_capacity(KAPPA_MAX) {
        return KAPPA_MAX;
    }
    let mut lo = 0.0f64;
    let mut hi = KAPPA_MAX;
    // γ_c is strictly decreasing: γ_c(lo) ≥ γ > γ_c(hi) at entry.
    while hi - lo > 1e-13 * (1.0 + hi) {
        let mid = 0.5 * (lo + hi);
        if gamma_capacity(mid) > gamma {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The once-computed capacity grid: `kappas[j] = κ_max(exp(u0 + j·du))` on
/// a uniform log-γ axis (see [`KAPPA_GRID_POINTS`]). A fixed-size array
/// behind a `OnceLock` — computed on first use, zero per-call allocation
/// afterwards.
struct GardnerLut {
    u0: f64,
    du: f64,
    kappas: [f64; KAPPA_GRID_POINTS],
}

static LUT: std::sync::OnceLock<GardnerLut> = std::sync::OnceLock::new();

fn lut() -> &'static GardnerLut {
    LUT.get_or_init(|| {
        let u0 = gamma_capacity(KAPPA_MAX).ln();
        let u1 = 2.0f64.ln(); // γ_c(0) = 2 exactly
        let du = (u1 - u0) / (KAPPA_GRID_POINTS - 1) as f64;
        let mut kappas = [0.0f64; KAPPA_GRID_POINTS];
        for (j, k) in kappas.iter_mut().enumerate() {
            *k = kappa_max_bisection((u0 + j as f64 * du).exp());
        }
        GardnerLut { u0, du, kappas }
    })
}

/// Inverted Gardner capacity `κ_max(γ)` via the LUT: `u = ln γ` maps to a
/// direct grid index (uniform axis, O(1) — no binary search) + linear
/// interpolation. Zero allocation.
///
/// Golden-gated against [`kappa_max_bisection`] at relative error < 1e-6
/// (`.benchmarks/702_regime_probe_goat.md`, G1 of M4).
pub fn kappa_max(gamma: f64) -> f64 {
    if gamma.is_nan() {
        return f64::NAN; // propagate honestly — a NaN load is a caller bug
    }
    if gamma <= 0.0 {
        // Non-positive load: infinite tolerance requested (bound saturates).
        return KAPPA_MAX;
    }
    if gamma >= 2.0 {
        return 0.0;
    }
    let table = lut();
    let u = gamma.ln();
    let t = (u - table.u0) / table.du;
    if t <= 1.0 {
        return table.kappas[1];
    }
    // 3-point Lagrange quadratic on the uniform-u grid, centered on the
    // NEAREST node (q ∈ [−0.5, 0.5]). Linear interpolation is not enough
    // here: κ''(u) ≈ π/2 is CONSTANT as κ → 0 (near γ = 2), so the LINEAR
    // error is O(du²) absolute and its RELATIVE error grows like 1/κ
    // (measured 1.5e-6 at κ = 0.0154 with 4096 points). The quadratic kills
    // that term — error O(du³) absolute, flat in relative terms (golden-gated).
    let m = ((t.round() as isize).clamp(1, KAPPA_GRID_POINTS as isize - 2)) as usize;
    let q = (t - m as f64).clamp(-0.5, 0.5);
    let km = table.kappas[m - 1];
    let k0 = table.kappas[m];
    let kp = table.kappas[m + 1];
    k0 + q * 0.5 * (kp - km) + q * q * 0.5 * (kp - 2.0 * k0 + km)
}

/// Basin-radius bound from a margin: `ρ < (κ/2)²`, clamped to `[0, 1]`.
/// (Paper eq 26: a stored pattern with margin `κ` tolerates a corrupted
/// fraction `ρ` when `κ > 2√ρ`.)
#[inline]
pub fn basin_radius_from_kappa(kappa: f64) -> f64 {
    if kappa <= 0.0 {
        return 0.0;
    }
    ((kappa * 0.5) * (kappa * 0.5)).clamp(0.0, 1.0)
}

/// Basin-radius bound from the load: `ρ < (κ_max(γ)/2)²` — the O(1) LUT
/// path. `γ ≤ 0` saturates the bound at 1 (infinite tolerance); `γ ≥ 2`
/// gives 0 (no capacity at any margin).
#[inline]
pub fn basin_radius_bound(gamma: f64) -> f64 {
    basin_radius_from_kappa(kappa_max(gamma))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_c_at_zero_is_two() {
        let g = gamma_capacity(0.0);
        assert!((g - 2.0).abs() < 1e-12, "γ_c(0) = 2: got {g}");
    }

    #[test]
    fn gamma_c_is_strictly_decreasing() {
        let mut prev = gamma_capacity(0.0);
        for i in 1..=160 {
            let k = i as f64 * 0.05;
            let g = gamma_capacity(k);
            assert!(g < prev, "γ_c must decrease at κ={k}");
            prev = g;
        }
    }

    #[test]
    fn phi_known_values() {
        assert!((phi_pdf(0.0) - 0.398_942_280_401_432_7).abs() < 1e-15);
        assert!((phi_cdf(0.0) - 0.5).abs() < 1e-15);
        // Φ(1.959964) = 0.975 (the two-sided 95% quantile).
        assert!((phi_cdf(1.959_964) - 0.975).abs() < 1e-9, "got {}", phi_cdf(1.959_964));
        // Symmetry.
        assert!((phi_cdf(-1.0) - (1.0 - phi_cdf(1.0))).abs() < 1e-15);
    }

    #[test]
    fn inversion_round_trip() {
        for &k in &[0.5, 1.0, 2.0, 3.0, 4.5] {
            let g = gamma_capacity(k);
            let k_back = kappa_max_bisection(g);
            assert!(
                (k_back - k).abs() < 1e-9,
                "round trip κ={k}: γ={g} → κ'={k_back}"
            );
        }
    }

    #[test]
    fn boundaries() {
        assert_eq!(kappa_max(2.0), 0.0);
        assert_eq!(kappa_max(3.0), 0.0);
        assert_eq!(basin_radius_bound(3.0), 0.0);
        assert_eq!(basin_radius_bound(0.0), 1.0);
        assert_eq!(basin_radius_bound(-1.0), 1.0);
        assert_eq!(basin_radius_from_kappa(0.0), 0.0);
        assert_eq!(basin_radius_from_kappa(-2.0), 0.0);
        assert_eq!(basin_radius_from_kappa(10.0), 1.0, "saturates at 1");
    }

    #[test]
    fn bound_is_monotone_in_load() {
        let mut prev = basin_radius_bound(0.05);
        let mut g = 0.15;
        while g < 1.95 {
            let b = basin_radius_bound(g);
            assert!(b <= prev, "bound must not rise with load at γ={g}");
            prev = b;
            g += 0.1;
        }
    }
}
