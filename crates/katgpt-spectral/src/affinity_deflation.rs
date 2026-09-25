//! Rank-1 spectral deflation of a rerank-stage candidate affinity matrix,
//! gated on effective rank (Issue 882 P4 rider (b), Research 586, Bench 897).
//!
//! # The primitive
//!
//! `A′ = A − λ·σ₁·u₁·v₁ᵀ = A − λ·(A v₁) v₁ᵀ`
//!
//! The M×M affinity among rerank candidates (`A_ij = sim(c_i, c_j)`) often
//! carries a **common-mode** component: every candidate is similar to a few
//! generic "hub" candidates, so the leading singular direction is
//! "similar-to-everything" rather than any real cluster. Removing it is the
//! matrix-level twin of Issue 882 P0's query-level differential anchor
//! (`q − λā`): subtract the correlated reference, keep the specific signal.
//!
//! # The gate — the anti-help trap
//!
//! Deflating a HEALTHY matrix removes its strongest real cluster, which is
//! exactly the structure the rerank exists to find. So the deflation is
//! gated: it runs only when the matrix is **collapsed**, i.e. its effective
//! rank is below `θ`. The effective rank used is the **stable rank**
//!
//! `srank(A) = ‖A‖_F² / σ₁² = Σσᵢ² / σ₁²  ∈ [1, rank(A)]`
//!
//! (Rudelson–Vershynin), because it falls out of the σ₁ power iteration the
//! deflation needs anyway at `O(M²)` extra — no eigendecomposition (the
//! Roy–Vetterli entropy erank, `river_valley::effective_rank_into`, is a
//! Jacobi sweep; Bench 897 cross-checks the two agree on which fixtures are
//! collapsed). A dominant common mode puts `srank → 1`; `k` comparable
//! clusters put it near `k`.
//!
//! ⚑ **The gate errs in the SAFE direction by construction.** Power
//! iteration's `‖A v‖` over a unit `v` is a LOWER bound on σ₁ at every
//! iterate, so an unconverged estimate can only OVER-state the stable rank
//! and make the gate refuse. A slow-converging (small spectral gap — i.e.
//! healthy) matrix is never deflated because the iteration ran out of steps.
//!
//! # Contract
//!
//! - `λ = 0` returns immediately: `A` untouched, bit-identical (G3).
//! - Gate refusal (`srank ≥ θ`, or any non-finite quantity — NaN refuses):
//!   `A` untouched, bit-identical; the diagnostics are still reported.
//! - Zero allocation: caller-owned [`DeflationScratch`] (G4).
//! - Deterministic: fixed start vector, fixed iteration order.
//! - The power iteration is the shared [`power_iter_step`] substrate
//!   (`spectral_retract`), not a parallel implementation; `AᵀA` is formed
//!   implicitly (never materialized).
//! - `θ` and `λ` are chosen by direct evaluation on fixtures (Bench 897's
//!   grid), never by gradient descent.

use crate::spectral_retract::power_iter_step;
use katgpt_core::simd::{simd_dot_f32, simd_fused_decay_write};

/// Deflation parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeflationConfig {
    /// Deflation strength `λ` (1.0 removes the full rank-1 component). `0` is
    /// the kill switch — bit-identical no-op.
    pub lambda: f32,
    /// Stable-rank threshold `θ`: deflate only when `srank < θ`.
    /// `f32::INFINITY` disables the gate (the ungated arm — measurement only).
    pub srank_threshold: f32,
    /// Power-iteration step cap.
    pub max_iters: u16,
    /// Early-exit relative tolerance on the `σ₁²` estimate between steps.
    pub tol: f32,
}

impl DeflationConfig {
    /// Default step cap.
    pub const DEFAULT_MAX_ITERS: u16 = 32;
    /// Default relative tolerance.
    pub const DEFAULT_TOL: f32 = 1e-4;

    /// Gated deflation at strength `lambda`, threshold `srank_threshold`.
    pub const fn new(lambda: f32, srank_threshold: f32) -> Self {
        Self {
            lambda,
            srank_threshold,
            max_iters: Self::DEFAULT_MAX_ITERS,
            tol: Self::DEFAULT_TOL,
        }
    }

    /// The kill switch — `λ = 0`.
    pub const fn off() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Ungated deflation — ALWAYS deflates a finite matrix. The anti-help
    /// trap's measurement arm; not a production setting.
    pub const fn ungated(lambda: f32) -> Self {
        Self::new(lambda, f32::INFINITY)
    }
}

/// Caller-owned scratch, reused across calls. Zero allocation after `new`.
#[derive(Debug, Clone)]
pub struct DeflationScratch {
    /// Right singular vector estimate `v₁` (length `cols`).
    pub v: Vec<f32>,
    /// Power-iteration matvec output (length `cols`).
    mv: Vec<f32>,
    /// `A v₁ = σ₁ u₁` (length `rows`) — the deflation's left factor.
    pub av: Vec<f32>,
}

impl DeflationScratch {
    /// Scratch for a `rows × cols` matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            v: vec![0.0; cols],
            mv: vec![0.0; cols],
            av: vec![0.0; rows],
        }
    }

    /// Resize for a new shape (allocates only if it grew).
    pub fn ensure(&mut self, rows: usize, cols: usize) {
        self.v.resize(cols, 0.0);
        self.mv.resize(cols, 0.0);
        self.av.resize(rows, 0.0);
    }
}

/// What the call measured and did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeflationReport {
    /// Leading singular value estimate (a lower bound on the true σ₁).
    pub sigma1: f32,
    /// `‖A‖_F²`.
    pub frob_sq: f32,
    /// `‖A‖_F² / σ₁²` (an upper bound on the true stable rank; `∞`/NaN on a
    /// degenerate matrix).
    pub stable_rank: f32,
    /// Power-iteration steps taken.
    pub iters: u16,
    /// Whether `A` was modified.
    pub deflated: bool,
}

impl DeflationReport {
    const SKIPPED: Self = Self {
        sigma1: 0.0,
        frob_sq: 0.0,
        stable_rank: f32::NAN,
        iters: 0,
        deflated: false,
    };
}

/// Leading right singular vector of row-major `a` (`rows × cols`) into
/// `scratch.v`, `A v₁` into `scratch.av`. Returns `(σ₁, steps)`.
///
/// Power iteration on `AᵀA` formed implicitly (`w = Σᵢ (aᵢ·v) aᵢ`, one pass
/// over `A` per step), through the shared [`power_iter_step`]. Fixed start
/// `v₀ = 1/√cols`. Zero allocation.
pub fn top_singular_into(
    a: &[f32],
    rows: usize,
    cols: usize,
    max_iters: u16,
    tol: f32,
    scratch: &mut DeflationScratch,
) -> (f32, u16) {
    assert_eq!(a.len(), rows * cols, "top_singular_into: matrix size");
    assert!(
        scratch.v.len() >= cols && scratch.mv.len() >= cols && scratch.av.len() >= rows,
        "top_singular_into: scratch too small"
    );
    let v = &mut scratch.v[..cols];
    let mv = &mut scratch.mv[..cols];
    let start = 1.0 / (cols.max(1) as f32).sqrt();
    v.fill(start);

    let matvec = |x: &[f32], out: &mut [f32]| {
        out.fill(0.0);
        for row in a.chunks_exact(cols) {
            let t = simd_dot_f32(row, x, cols);
            simd_fused_decay_write(out, 1.0, row, t);
        }
    };

    let mut prev = 0.0f32;
    let mut steps = 0u16;
    while steps < max_iters {
        let s2 = power_iter_step(v, mv, matvec);
        steps += 1;
        if s2 == 0.0 {
            break; // degenerate — v untouched
        }
        if (s2 - prev).abs() <= tol * s2 {
            break;
        }
        prev = s2;
    }

    let av = &mut scratch.av[..rows];
    for (dst, row) in av.iter_mut().zip(a.chunks_exact(cols)) {
        *dst = simd_dot_f32(row, v, cols);
    }
    let sigma1 = simd_dot_f32(av, av, rows).sqrt();
    (sigma1, steps)
}

/// Stable rank `‖A‖_F² / σ₁²`.
#[inline]
pub fn stable_rank(frob_sq: f32, sigma1: f32) -> f32 {
    frob_sq / (sigma1 * sigma1)
}

/// Gated rank-1 deflation, in place: `A ← A − λ (A v₁) v₁ᵀ` iff
/// `srank(A) < θ`. See the module docs for the contract.
pub fn deflate_rank1_gated_inplace(
    a: &mut [f32],
    rows: usize,
    cols: usize,
    cfg: &DeflationConfig,
    scratch: &mut DeflationScratch,
) -> DeflationReport {
    assert_eq!(
        a.len(),
        rows * cols,
        "deflate_rank1_gated_inplace: matrix size"
    );
    if cfg.lambda == 0.0 || rows == 0 || cols == 0 {
        return DeflationReport::SKIPPED;
    }
    let frob_sq = simd_dot_f32(a, a, a.len());
    let (sigma1, iters) = top_singular_into(a, rows, cols, cfg.max_iters, cfg.tol, scratch);
    let srank = stable_rank(frob_sq, sigma1);
    // NaN in any operand makes every comparison false ⇒ refuse.
    let deflate = cfg.lambda.is_finite() && srank.is_finite() && srank < cfg.srank_threshold;
    if deflate {
        let v = &scratch.v[..cols];
        for (row, &t) in a.chunks_exact_mut(cols).zip(&scratch.av[..rows]) {
            simd_fused_decay_write(row, 1.0, v, -cfg.lambda * t);
        }
    }
    DeflationReport {
        sigma1,
        frob_sq,
        stable_rank: srank,
        iters,
        deflated: deflate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `c·h hᵀ + blocks` — collapsed when `c` dominates.
    fn fixture(m: usize, blocks: usize, c: f32) -> Vec<f32> {
        let mut a = vec![0.0f32; m * m];
        for i in 0..m {
            let hi = 0.3 + 0.7 * ((i * 7919) % m) as f32 / m as f32;
            for j in 0..m {
                let hj = 0.3 + 0.7 * ((j * 7919) % m) as f32 / m as f32;
                let same = (i * blocks / m) == (j * blocks / m);
                a[i * m + j] = c * hi * hj + if same { 1.0 } else { 0.0 };
            }
        }
        a
    }

    #[test]
    fn lambda_zero_is_bit_identical() {
        let a0 = fixture(24, 3, 10.0);
        let mut a = a0.clone();
        let mut s = DeflationScratch::new(24, 24);
        let r = deflate_rank1_gated_inplace(&mut a, 24, 24, &DeflationConfig::off(), &mut s);
        assert!(!r.deflated);
        assert!(a.iter().zip(&a0).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn collapsed_deflates_healthy_refuses() {
        let m = 32;
        let mut s = DeflationScratch::new(m, m);
        let cfg = DeflationConfig::new(1.0, 1.5);

        let mut collapsed = fixture(m, 4, 20.0);
        let r = deflate_rank1_gated_inplace(&mut collapsed, m, m, &cfg, &mut s);
        assert!(r.deflated, "collapsed srank {}", r.stable_rank);
        assert!(r.stable_rank < 1.5);

        let h0 = fixture(m, 4, 0.0);
        let mut healthy = h0.clone();
        let r = deflate_rank1_gated_inplace(&mut healthy, m, m, &cfg, &mut s);
        assert!(!r.deflated, "healthy srank {}", r.stable_rank);
        assert!(
            healthy
                .iter()
                .zip(&h0)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
    }

    #[test]
    fn full_deflation_of_exact_rank1_is_near_zero() {
        let m = 16;
        let mut a = vec![0.0f32; m * m];
        for i in 0..m {
            for j in 0..m {
                a[i * m + j] = (1.0 + i as f32) * (2.0 + j as f32 * 0.5);
            }
        }
        let norm0 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut s = DeflationScratch::new(m, m);
        let r = deflate_rank1_gated_inplace(&mut a, m, m, &DeflationConfig::new(1.0, 1.01), &mut s);
        assert!(r.deflated);
        let norm1 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm1 < 1e-4 * norm0, "residual {norm1} / {norm0}");
    }

    #[test]
    fn nan_refuses_and_is_untouched() {
        let m = 8;
        let mut a = fixture(m, 2, 10.0);
        a[5] = f32::NAN;
        let a0 = a.clone();
        let mut s = DeflationScratch::new(m, m);
        let r = deflate_rank1_gated_inplace(&mut a, m, m, &DeflationConfig::ungated(1.0), &mut s);
        assert!(!r.deflated);
        assert!(a.iter().zip(&a0).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn zero_matrix_refuses() {
        let mut a = vec![0.0f32; 36];
        let mut s = DeflationScratch::new(6, 6);
        let r = deflate_rank1_gated_inplace(&mut a, 6, 6, &DeflationConfig::ungated(1.0), &mut s);
        assert!(!r.deflated);
        assert!(a.iter().all(|&x| x == 0.0));
    }
}
