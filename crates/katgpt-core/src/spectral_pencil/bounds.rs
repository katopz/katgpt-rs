//! `bounds` — global feature-influence bounds + growth envelope +
//! mirror-duality laws (Issue 676 T5).
//!
//! * **Global per-feature bound** (paper Cor. 1): `|f(x+δ)−f(x)| ≤
//!   Σ|δᵢ|·‖Aᵢ‖₂` — the spectral norm of each feature matrix is the
//!   matrix-age analogue of a linear coefficient's magnitude.
//!   [`SpectralNorms`] computes them once at construction (amortized);
//!   exact closed forms for the structured cases (rank-one, diagonal),
//!   power-iteration estimate otherwise (a monotonically-improving LOWER
//!   estimate of `‖A‖₂` — see [`norm_power_iter`]).
//! * **Linear growth envelope**: `|f(x)| ≤ ‖A₀‖₂ + Σ|xᵢ|·‖Aᵢ‖₂`
//!   (`‖A(x)‖₂` triangle + eigenvalue interlacing).
//! * **Mirror duality** `λk(−A) = −λ_{d−k+1}(A)`: k=1 concave ↔ k=d
//!   convex for free — a property-test law in `tests.rs`, the runtime
//!   helper is [`crate::spectral_pencil::sym::SymPacked::negate`].

use crate::spectral_pencil::dense::{DenseScratch, jacobi_eigen};
use crate::spectral_pencil::sym::SymPacked;

/// Per-matrix spectral norms for one pencil, computed once.
///
/// `a0` + `a[i]` norms. Exact for the structured constructors; power
/// estimates otherwise (documented per-entry via [`NormEstimate`]).
#[derive(Clone, Copy, Debug)]
pub struct SpectralNorms<const D: usize, const N: usize> {
    pub a0: f32,
    pub a: [f32; N],
}

/// How each norm in a [`SpectralNorms`] was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormEstimate {
    /// Exact: rank-one `β·ddᵀ` ⇒ `‖A‖₂ = |β|`; diagonal ⇒ `max|dⱼ|`;
    /// α·I + diag ⇒ `max|α + εⱼ|`.
    Exact,
    /// Power-iteration Rayleigh estimate — a LOWER bound converging up
    /// geometrically; the residual gap is bounded by the eigengap.
    PowerIter { iters: u16 },
}

/// Exact spectral norm of a rank-one `β·d·dᵀ` feature matrix.
#[inline]
#[must_use]
pub fn norm_rank_one(beta: f32) -> f32 {
    beta.abs()
}

/// Exact spectral norm of a diagonal matrix.
#[inline]
#[must_use]
pub fn norm_diagonal(diag: &[f32]) -> f32 {
    let mut m = 0.0_f32;
    for &v in diag {
        m = m.max(v.abs());
    }
    m
}

/// Power-iteration estimate of `‖A‖₂ = max(|λmin|, |λmax|)` for a packed
/// symmetric matrix. Rayleigh quotients of unit vectors bracket below,
/// so the result UNDERESTIMATES by at most the eigengap factor — honest
/// direction for an influence bound used as `≤` with a safety factor.
///
/// Deterministic: fixed start vector, fixed iteration count.
#[must_use]
pub fn norm_power_iter<const D: usize>(a: &SymPacked<D>, iters: u16) -> f32 {
    // Deterministic start: 1 + (i/d) — fixed, no RNG draw.
    let mut x = [0.0_f32; D];
    for (i, v) in x.iter_mut().enumerate() {
        *v = 1.0 + (i as f32) / (D as f32);
    }
    let full = a.to_full();
    // Normalize once up front; every subsequent x stays normalized.
    let nx = crate::spectral_pencil::sym::norm2(&x);
    if nx == 0.0 {
        return 0.0;
    }
    for v in x.iter_mut() {
        *v /= nx;
    }
    let mut rayleigh = 0.0_f32;
    for _ in 0..iters {
        // y = A·x (x normalized ⇒ Rayleigh = xᵀy)
        let mut y = [0.0_f32; D];
        for (i, yi) in y.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for (j, &xj) in x.iter().enumerate() {
                acc += f64::from(full[i][j]) * f64::from(xj);
            }
            *yi = acc as f32;
        }
        let mut q = 0.0_f64;
        for i in 0..D {
            q += f64::from(x[i]) * f64::from(y[i]);
        }
        rayleigh = (q as f32).abs();
        let norm_y = crate::spectral_pencil::sym::norm2(&y);
        if norm_y == 0.0 {
            return 0.0; // exact null vector hit
        }
        for (v, &yi) in x.iter_mut().zip(y.iter()) {
            *v = yi / norm_y;
        }
    }
    rayleigh
}

impl<const D: usize, const N: usize> SpectralNorms<D, N> {
    /// Norms via power iteration (default estimate path). Construction
    /// cost: `(N+1)·iters` matvecs — amortize at pencil build time.
    #[must_use]
    pub fn estimate_pencil(a0: &SymPacked<D>, a: &[SymPacked<D>; N], iters: u16) -> Self {
        let mut norms = [0.0_f32; N];
        for (m, out) in a.iter().zip(norms.iter_mut()) {
            *out = norm_power_iter(m, iters);
        }
        Self {
            a0: norm_power_iter(a0, iters),
            a: norms,
        }
    }

    /// Linear growth envelope `‖A₀‖₂ + Σ|xᵢ|·‖Aᵢ‖₂` — an upper bound on
    /// `|f(x)|` for every k.
    #[must_use]
    pub fn growth_envelope(&self, x: &[f32; N]) -> f32 {
        let mut e = self.a0;
        for (xi, &ni) in x.iter().zip(self.a.iter()) {
            e += xi.abs() * ni;
        }
        e
    }

    /// Weyl per-feature influence budget `|δᵢ|·‖Aᵢ‖₂` (paper eq. 3).
    #[must_use]
    pub fn feature_influence_bound(&self, i: usize, delta: f32) -> f32 {
        delta.abs() * self.a[i.min(N - 1)]
    }
}

/// Convenience: full-solve exact norm for small D (the authority the
/// power estimate is checked against in tests; also fine for
/// construction-time use when `(N+1)·Jacobi` fits the budget).
#[must_use]
pub fn norm_jacobi_exact<const D: usize>(a: &SymPacked<D>, scratch: &mut DenseScratch<D>) -> f32 {
    let full = a.to_full();
    jacobi_eigen(&full, false, scratch);
    let lo = scratch.values[0].abs();
    let hi = scratch.values[D - 1].abs();
    lo.max(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_one_and_diagonal_norms_exact() {
        assert_eq!(norm_rank_one(-2.5), 2.5);
        let d = [0.3_f32, -1.2, 0.9];
        assert!((norm_diagonal(&d) - 1.2).abs() < 1e-7);
    }

    #[test]
    fn power_iter_underestimates_jacobi_norm() {
        const D: usize = 5;
        let mut rng = 77_u64;
        let mut full = [[0.0_f32; D]; D];
        for (i, j) in (0..D).flat_map(|i| (i..D).map(move |j| (i, j))) {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((rng >> 33) as f32 / 2.0_f32.powi(31)) * 2.0 - 1.0;
            full[i][j] = v;
            full[j][i] = v;
        }
        let packed = SymPacked::<D>::pack_from_full(&full);
        let mut s = DenseScratch::<D>::new();
        let est = norm_power_iter(&packed, 60);
        let exact = norm_jacobi_exact(&packed, &mut s);
        assert!(est <= exact * (1.0 + 1e-3), "est {est} vs exact {exact}");
        assert!(
            est >= exact * 0.9,
            "est {est} too far below exact {exact} (60 iters)"
        );
    }

    #[test]
    fn growth_envelope_dominates_evaluations() {
        // Envelope uses conservative norms (power estimate + frobenius
        // fallback would be exact-upper); here exact norms via Jacobi.
        const D: usize = 4;
        const N: usize = 3;
        let mut rng = 31_u64;
        let next = |rng: &mut u64| -> f32 {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((*rng >> 33) as f32) / 2.0_f32.powi(31)) * 2.0 - 1.0
        };
        let a0_full = {
            let mut f = [[0.0_f32; D]; D];
            for (i, j) in (0..D).flat_map(|i| (i..D).map(move |j| (i, j))) {
                let v = next(&mut rng);
                f[i][j] = v;
                f[j][i] = v;
            }
            f
        };
        let a: [SymPacked<D>; N] = {
            let mut arr = [SymPacked::zeroed(); N];
            for m in arr.iter_mut() {
                let mut f = [[0.0_f32; D]; D];
                for (i, j) in (0..D).flat_map(|i| (i..D).map(move |j| (i, j))) {
                    let v = next(&mut rng);
                    f[i][j] = v;
                    f[j][i] = v;
                }
                *m = SymPacked::pack_from_full(&f);
            }
            arr
        };
        let a0 = SymPacked::pack_from_full(&a0_full);
        let mut s = DenseScratch::<D>::new();
        // Exact norms.
        let mut norms = [0.0_f32; N];
        for (m, o) in a.iter().zip(norms.iter_mut()) {
            *o = norm_jacobi_exact(m, &mut s);
        }
        let norms = SpectralNorms::<D, N> {
            a0: norm_jacobi_exact(&a0, &mut s),
            a: norms,
        };
        // Sample xs and check |f(x)| ≤ envelope for every k.
        for trial in 0..200 {
            let x = [next(&mut rng), next(&mut rng), next(&mut rng)];
            // build A(x) and full-solve
            let mut ax = a0;
            for (m, &xi) in a.iter().zip(x.iter()) {
                ax.add_scaled_into(m, xi);
            }
            jacobi_eigen(&ax.to_full(), false, &mut s);
            for k in 0..D {
                let f = s.values[k].abs();
                let env = norms.growth_envelope(&x);
                assert!(
                    f <= env * (1.0 + 1e-4) + 1e-5,
                    "trial {trial} k {k}: |f| {f} > envelope {env}"
                );
            }
        }
    }
}
