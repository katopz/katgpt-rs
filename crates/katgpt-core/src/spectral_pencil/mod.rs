//! `spectral_pencil` — the affine matrix pencil scalar gate
//! `f(x) = λk(A₀ + Σ xᵢAᵢ)` (Issue 676, Research 495; arXiv:2608.08003
//! "The Spectral Neuron", Shtoff, TII 2026).
//!
//! A scalar decision function whose input enters **linearly** into a
//! symmetric matrix and whose nonlinearity is reading **one ordered
//! eigenvalue**. Expressivity grows with matrix dimension d while
//! retaining linear-model-style transparency:
//!
//! * **Shape by construction**: k=1 concave, k=d convex
//!   (Rayleigh–Ritz); `Aᵢ ⪰ 0` ⇒ non-decreasing in `xᵢ` (Loewner
//!   monotonicity) — mixable per feature, composable with the k-index.
//! * **Global feature-influence bounds** (Weyl): `|f(x+δ)−f(x)| ≤
//!   Σ|δᵢ|·‖Aᵢ‖₂` — closed-form from coefficients ([`bounds`]).
//! * **Exact local attribution** (Hellmann–Feynman): `∂f/∂xᵢ = vᵀAᵢv`
//!   at simple eigenvalues (the T6 surface, built on the dense
//!   kernel's eigenvector output).
//! * **Seeded eigengap-guaranteed init**: γk ≥ ½ on `‖x‖∞ ≤ 5` proven
//!   via Weyl ([`init`]) — a zero-training nonlinear function
//!   generator with a conditioning certificate.
//!
//! ## Cost (paper §7.3)
//!
//! Dense d=16 ≈ 8K FLOPs/eval (pinned Jacobi); tridiagonal ≈ 800
//! (Sturm bisection, ≈60·d ops). 10,000 NPCs × 20 Hz ⇒ 160 MFLOP/s
//! dense — SIMD-trivial; attribution = n packed dots.
//!
//! ## Determinism policy (load-bearing)
//!
//! Committed readouts use **pinned algorithms only**: the cyclic-Jacobi
//! schedule + tolerance ([`dense`]), the Sturm zero-pivot convention +
//! 60-step bisection ([`tridiag`]), the Householder QR sign-fix
//! ([`init`]). No library eigensolver on any committed path — rotation
//! order variance is not reproducible across versions. Integer Sturm
//! counts are the platform-stable exact class.
//!
//! ## Layout
//!
//! * [`sym`] — isometric `1/√2` packing (T1): Frobenius norm/inner
//!   product become plain SIMD dots on packed vectors.
//! * [`dense`] — pinned cyclic-Jacobi eigen-kernel, `d ≤ 32` (T2).
//! * [`tridiag`] — Sturm count + bisection, the O(d)-param family (T3).
//! * [`init`] — seeded γk≥½ constructors, dense + tridiag + squareplus
//!   PSD parametrization (T4).
//! * [`bounds`] — global influence bounds + growth envelope (T5).
//!
//! T6 (attribution), T7 (shape DSL), T8 (canonical gauge), T9
//! (invertible warp), T10 (bench + GOAT), T11 (module doc), T12 (UQ
//! floor follow-through) land in follow-up cycles per the issue.
//!
//! spectral_pencil — the affine matrix pencil scalar gate f(x) =
//! λk(A₀ + Σ xᵢAᵢ) (Issue 676, Research 495, arXiv:2608.08003 "The
//! Spectral Neuron"). Shape-by-construction + coefficient transparency
//! + seeded γk ≥ ½ init. Opt-in; see the module doc for the determinism
//! policy (pinned Jacobi/Sturm/QR — no library eigensolver on committed
//! paths).

pub mod attribution;
pub mod bounds;
pub mod dense;
#[cfg(feature = "committed_field_blend")]
pub mod field;
pub mod gauge;
pub mod genome;
pub mod init;
pub mod shape;
pub mod sym;
pub mod tridiag;
pub mod warp;

pub use dense::{DenseScratch, JacobiReport, jacobi_eigen, kth_eigenvalue};
pub use init::{
    BOX_R, SeededDenseInit, SeededTriInit, seeded_dense, seeded_tridiag, squareplus,
    squareplus_param_into,
};
pub use sym::{SymPacked, Tridiagonal, packed_len};
pub use tridiag::{TriScratch, count_below, kth_eigenvalue_bisect};

/// A dense affine pencil `A₀ + Σ xᵢAᵢ` with `D×D` symmetric matrices and
/// `N` features. All storage fixed-size (`#[repr(C)]`-friendly members);
/// evaluation is allocation-free through caller-owned scratch.
#[derive(Clone, Debug)]
pub struct DensePencil<const D: usize, const N: usize> {
    pub a0: SymPacked<D>,
    pub a: [SymPacked<D>; N],
}

/// Result of one pencil evaluation with diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PencilEval {
    /// λk(A(x)) — the k-th smallest eigenvalue.
    pub value: f32,
    /// γk at this evaluation: distance from λk to the rest of the
    /// spectrum (paper Def. 3, multiplicity-aware). `None` when `D == 1`.
    /// Small γk ⇒ low-trust attribution (Davis–Kahan ~1/γk).
    pub eigengap: Option<f32>,
    pub jacobi: JacobiReport,
}

impl<const D: usize, const N: usize> DensePencil<D, N> {
    /// Build `A(x)` into `scratch` (matrix semantics; fused packed adds).
    pub fn materialize(&self, x: &[f32; N], scratch: &mut DenseScratch<D>) {
        let mut acc = self.a0;
        for (m, &xi) in self.a.iter().zip(x.iter()) {
            acc.add_scaled_into(m, xi);
        }
        scratch.a = acc.to_full();
    }

    /// Evaluate `λk(A(x))` (k 0-indexed) with the full diagnostic report.
    #[must_use]
    pub fn eval(&self, x: &[f32; N], k: usize, scratch: &mut DenseScratch<D>) -> PencilEval {
        self.materialize(x, scratch);
        let a = scratch.a;
        let jacobi = jacobi_eigen(&a, false, scratch);
        let value = scratch.values[k.min(D - 1)];
        let eigengap = eigengap_at(&scratch.values, k.min(D - 1));
        PencilEval {
            value,
            eigengap,
            jacobi,
        }
    }

    /// Evaluate with eigenvectors (needed for T6 attribution): leaves
    /// `scratch.v` holding the eigenvector of λk in column `k` (sorted).
    #[must_use]
    pub fn eval_with_vectors(
        &self,
        x: &[f32; N],
        k: usize,
        scratch: &mut DenseScratch<D>,
    ) -> PencilEval {
        self.materialize(x, scratch);
        let a = scratch.a;
        let jacobi = jacobi_eigen(&a, true, scratch);
        let value = scratch.values[k.min(D - 1)];
        let eigengap = eigengap_at(&scratch.values, k.min(D - 1));
        PencilEval {
            value,
            eigengap,
            jacobi,
        }
    }
}

/// γk per the paper's Def. 3 (multiplicity-aware): with `[r, s]` the
/// block of eigenvalues equal to `values[k]`,
/// `γk = min(values[k] − values[r−1], values[s+1] − values[k])`, where
/// the out-of-range neighbour is ±∞ (one-sided at the spectrum edge).
#[must_use]
pub fn eigengap_at(values: &[f32], k: usize) -> Option<f32> {
    let d = values.len();
    if d < 2 {
        return None;
    }
    let alpha = values[k];
    let mut r = k;
    while r > 0 && values[r - 1] == alpha {
        r -= 1;
    }
    let mut s = k;
    while s + 1 < d && values[s + 1] == alpha {
        s += 1;
    }
    let below = if r == 0 {
        f32::INFINITY
    } else {
        alpha - values[r - 1]
    };
    let above = if s + 1 == d {
        f32::INFINITY
    } else {
        values[s + 1] - alpha
    };
    Some(below.min(above))
}

/// A tridiagonal affine pencil — the O(d)-parameter family (T3).
#[derive(Clone, Debug)]
pub struct TridiagPencil<const D: usize, const N: usize> {
    pub a0: Tridiagonal<D>,
    pub a: [Tridiagonal<D>; N],
}

impl<const D: usize, const N: usize> TridiagPencil<D, N> {
    /// Evaluate `λk(A(x))` via Sturm bisection (k 0-indexed).
    #[must_use]
    pub fn eval(&self, x: &[f32; N], k: usize, scratch: &mut TriScratch<D>) -> f32 {
        tridiag::fuse_into(
            &self.a0.diag,
            &self.a0.off,
            &self.a.map(|m| m.diag),
            &self.a.map(|m| m.off),
            x,
            scratch,
        );
        // Copy out to avoid the overlapping &/&mut borrow on scratch.
        let diag = scratch.diag;
        let off = scratch.off;
        kth_eigenvalue_bisect(&diag, &off, k)
    }

    /// Exact integer count of eigenvalues strictly below μ (the
    /// platform-stable predicate class).
    #[must_use]
    pub fn count_below(&self, x: &[f32; N], mu: f32, scratch: &mut TriScratch<D>) -> u32 {
        tridiag::fuse_into(
            &self.a0.diag,
            &self.a0.off,
            &self.a.map(|m| m.diag),
            &self.a.map(|m| m.off),
            x,
            scratch,
        );
        let diag = scratch.diag;
        let off = scratch.off;
        let mut work = [0.0_f32; D];
        count_below(&diag, &off, mu, &mut work)
    }
}

#[cfg(test)]
mod tests;
