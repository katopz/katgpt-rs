//! Symmetric eigendecomposition via Householder tridiagonalization + implicit-shift QL.
//!
//! Drop-in alternative to [`crate::karc::jacobi_eigen`] for large symmetric
//! matrices. Same storage convention (`eigvecs[i*n + j] = V[i, j]` row-major,
//! column k of V paired with `eigvals[k]`), but ~5-10× faster at `n ≥ 256`
//! because:
//! - Householder reduction is `O(n³/3)` with cache-friendly blocked access
//!   (one pass over the lower triangle per reflection).
//! - Tridiagonal QL sweeps cost `O(n²)` per sweep, not `O(n³)` like full Jacobi
//!   sweeps. Two-three sweeps typically suffice per eigenvalue.
//!
//! Total cost: `~3·n³` FLOPs vs Jacobi's `~50·n³` (10 sweeps × `~5n³` per sweep).
//!
//! # Algorithm (Golub-van Loan §8.3 + Numerical Recipes §11.3)
//!
//! **Phase 1 — Householder tridiagonalization (Golub-van Loan 8.3.1).**
//! Reduces `A` to symmetric tridiagonal `T = Qᵀ · A · Q` via `n-2` Householder
//! reflections. Q is accumulated into the `eigvecs` output (starting from I).
//!
//! **Phase 2 — Implicit-shift QL with Wilkinson shift (Golub-van Loan 8.3.3,
//! Numerical Recipes `tqli`).** Reduces `T` to diagonal form via Givens
//! rotations, accumulating into Q. The implicit shift avoids the explicit
//! `O(n²)` per-iteration shift computation and the corresponding accuracy loss.
//!
//! # Determinism
//!
//! Pure Rust, deterministic sweep order, no platform-dependent dispatch. Two
//! calls with identical inputs produce bit-identical outputs on the same host.
//! This is what backs the `karc_householder_eig` feature gate's G4 bit-
//! reproducibility contract (Issue 186 T4).
//!
//! # References
//!
//! - Golub-van Loan, *Matrix Computations* 4th ed., §8.3 (Symmetric QR algorithm).
//! - Press-Teukolsky-Vetterling-Flannery, *Numerical Recipes* 3rd ed., §11.3
//!   (`tqli` — Tridiagonal QL Implicit).
//! - Issue 186 — Path B deliberation + acceptance criteria.
//! - Issue 185 — T1+T2 implementation of the consumer
//!   (`karc::large_dh::low_rank_fit_jacobi_bstep`).

#[cfg(test)]
mod tests;

/// Parallel variant (row-parallel rayon). Gated by `karc_householder_eig_par`.
/// See [`par::symmetric_eig_par`] for the entry point.
#[cfg(feature = "karc_householder_eig_par")]
pub mod par;

/// Overflow-guarded Euclidean radius `sqrt(x² + y²)`.
///
/// On the entire range where `x² + y²` cannot overflow (`|x|, |y| < 1e150`,
/// i.e. every input the bounded QL arithmetic can produce — all quantities
/// stay within ~max(|d|, |e|) of the tridiagonal) this is the verbatim
/// inline form, so results are bit-identical to every prior run of this
/// module. Only the overflow-threatened tail routes through the scaled
/// form `hi·sqrt(1 + (lo/hi)²)`, which cannot overflow. The Numerical
/// Recipes reference computes both radii through `pythag` unconditionally;
/// this guarded variant keeps that safety while preserving the bit
/// patterns of the benign range (this module sits under downstream
/// bit-identity pins).
fn hypot_safe(x: f64, y: f64) -> f64 {
    const OVERFLOW_GUARD: f64 = 1e150;
    if x.abs() < OVERFLOW_GUARD && y.abs() < OVERFLOW_GUARD {
        (x * x + y * y).sqrt()
    } else {
        let ax = x.abs();
        let ay = y.abs();
        let (hi, lo) = if ax >= ay { (ax, ay) } else { (ay, ax) };
        if hi == 0.0 {
            0.0
        } else if lo.is_infinite() {
            // hypot(inf, ·) is inf by convention (the inline form agrees).
            f64::INFINITY
        } else {
            let r = lo / hi;
            hi * (1.0 + r * r).sqrt()
        }
    }
}

/// Loud refusal on a non-finite entry: the first (i, j) with a non-finite
/// value panics naming its position. The QL arithmetic cannot recover from
/// a poisoned input — every deflation test reads false against NaN and the
/// iteration would spin to its cap and panic with a confusing NaN dump
/// instead (the Issue 931 audit signature).
fn refuse_non_finite_input(a: &[f64], n: usize) {
    for (idx, &v) in a.iter().enumerate().take(n * n) {
        if !v.is_finite() {
            panic!(
                "symmetric_eig: non-finite input at (i, j) = ({}, {}) = {:e} — \
                 the solver requires a finite symmetric matrix",
                idx / n,
                idx % n,
                v
            );
        }
    }
}

/// Loud refusal on a non-finite intermediate produced mid-solve.
fn refuse_non_finite_intermediate(what: &str, d: &[f64], e: &[f64], n: usize, l: usize, iter: usize) {
    for k in l..n {
        if !d[k].is_finite() {
            panic!(
                "symmetric_eig: non-finite intermediate ({what}) at l={l}, iteration {iter}: \
                 d[{k}] = {:e} — refusing instead of returning garbage",
                d[k]
            );
        }
        if k < n && !e[k].is_finite() {
            panic!(
                "symmetric_eig: non-finite intermediate ({what}) at l={l}, iteration {iter}: \
                 e[{k}] = {:e} — refusing instead of returning garbage",
                e[k]
            );
        }
    }
}

/// Scratch buffers for [`symmetric_eig`]. Owns ~`n²` f64 for the working copy
/// of A plus `O(n)`-sized temps. Lazily sized via [`ensure_capacity`].
///
/// [`ensure_capacity`]: SymmetricEigScratch::ensure_capacity
#[derive(Default)]
pub struct SymmetricEigScratch {
    /// Working copy of `A` (`n×n` f64). Reduced to tridiagonal form during
    /// [`tridiagonalize`]; the diagonal lands in `a_work[i*n+i]` and the
    /// subdiagonal in `a_work[(i+1)*n+i]`.
    a_work: Vec<f64>,
    /// Householder vector `v` (length `n`; only the first `n-k-1` entries are
    /// meaningful at tridiagonalization step `k`).
    v_buf: Vec<f64>,
    /// `p = β · A_sub · v` temp (length `n`).
    p_buf: Vec<f64>,
    /// `w = p - K·v` temp (length `n`).
    w_buf: Vec<f64>,
    /// Tridiagonal subdiagonal `e[i] = T[i, i+1]` (length `n`; `e[n-1]` is a
    /// sentinel set to 0 by the QL phase).
    subdiag: Vec<f64>,
}

impl SymmetricEigScratch {
    /// Allocate empty. Use [`ensure_capacity`] before the first call to
    /// [`symmetric_eig`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Grow buffers to handle an `n×n` eigendecomposition. Idempotent — only
    /// allocates if the current capacity is too small.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.a_work.len() < n * n {
            self.a_work.resize(n * n, 0.0);
        }
        if self.v_buf.len() < n {
            self.v_buf.resize(n, 0.0);
        }
        if self.p_buf.len() < n {
            self.p_buf.resize(n, 0.0);
        }
        if self.w_buf.len() < n {
            self.w_buf.resize(n, 0.0);
        }
        if self.subdiag.len() < n {
            self.subdiag.resize(n, 0.0);
        }
    }
}

/// Symmetric eigendecomposition via Householder tridiagonalization + implicit-shift QL.
///
/// Computes `A = V · diag(λ) · Vᵀ` for a symmetric `n×n` row-major f64 matrix
/// `a_in`, writing eigenvalues into `eigvals` (length `n`, unsorted) and
/// eigenvectors into `eigvecs` (length `n*n`, row-major: `eigvecs[i*n + j] = V[i, j]`,
/// so column `k` of `V` — the eigenvector paired with `eigvals[k]` — is the
/// length-`n` stride-`n` slice starting at `eigvecs[k]`).
///
/// Storage convention matches [`crate::karc::jacobi_eigen`] exactly. This
/// function is a drop-in replacement for the G-path (large `d_h`) call in
/// `karc::large_dh::low_rank_fit_jacobi_bstep`; the A-path (`r×r`) stays on
/// Jacobi where its constant-factor is smaller.
///
/// # When to prefer this over Jacobi
///
/// - `n ≤ 16`: use Jacobi — lower constant, equally accurate.
/// - `n ≥ 256`: use this — 5-10× faster due to the `O(n³)` vs `O(n⁴)`-ish
///   sweep-cost gap (Jacobi does ~10 sweeps × `O(n³)` each; Householder+QL
///   does one `O(n³/3)` reduction + `O(n³)` worth of QL rotations).
/// - `n ≥ 18_720` (Issue 186 target): Jacobi is infeasible (~16 hours
///   projected); Householder+QL is the only modelless path to a feasible
///   one-time G eigendecomp.
///
/// # Panics
///
/// - Non-finite input: any NaN/inf entry panics at entry naming `(i, j)`
///   (Issue 931). QL cannot recover from a poisoned matrix — every deflation
///   test reads false against NaN — so the old behavior (spin to the iteration
///   cap, panic with a NaN dump) is replaced by an immediate, precise refusal.
///   The parallel variant refuses identically.
/// - Non-finite intermediates: if the tridiagonalization or a QL sweep ever
///   produced a non-finite `d`/`e` (not reachable for finite bounded input —
///   all QL quantities stay within ~max(|d|, |e|) — but guarded anyway), the
///   solve refuses immediately naming the phase, position and iteration.
/// - Input slice length mismatches (debug_assert only — release builds skip).
/// - QL fails to converge after `max_iters_per_eigval` iterations per
///   eigenvalue (Numerical Recipes uses 30; Issue 931 callers use 200). This
///   indicates either a pathologically close-to-defective matrix or a bug in
///   the implementation. Caller should regularize (e.g. `A + ε·I`) or
///   investigate.
///
/// The two radii of the QL iteration (the shift radius and the bulge-chase
/// radius) are computed by an overflow-guarded [`hypot_safe`]: bit-identical
/// to the historical inline `sqrt(x²+y²)` on the entire benign range, and
/// scaled-form beyond. The Numerical Recipes reference computes both
/// radii through `pythag` unconditionally; the inline form overflowed at
/// |entries| ≳ 1e154 and would cascade `inf → inf/inf = NaN` — the latent
/// transcription defect class closed here (Issue 931).
///
/// # Determinism
///
/// Pure Rust, fixed sweep order. Bit-identical output for identical input.
pub fn symmetric_eig(
    eigvals: &mut [f64],
    eigvecs: &mut [f64],
    a_in: &[f64],
    scratch: &mut SymmetricEigScratch,
    n: usize,
    max_iters_per_eigval: usize,
) {
    debug_assert_eq!(a_in.len(), n * n, "a_in must be n*n");
    debug_assert_eq!(eigvals.len(), n, "eigvals must be n");
    debug_assert_eq!(eigvecs.len(), n * n, "eigvecs must be n*n");

    // Loud refusal on a poisoned input (Issue 931): NaN/inf entries make every
    // QL deflation test read false, so the solver would spin to its iteration
    // cap and panic with a confusing NaN dump. Refuse here, naming the entry.
    refuse_non_finite_input(a_in, n);

    scratch.ensure_capacity(n);

    // Initialize eigvecs = I (will accumulate Q from Householder + Q' from QL).
    // memset + n diagonal stores instead of n² branchy element writes.
    eigvecs[..n * n].fill(0.0);
    for i in 0..n {
        eigvecs[i * n + i] = 1.0;
    }

    // Copy A into the working buffer.
    scratch.a_work[..n * n].copy_from_slice(&a_in[..n * n]);

    // Phase 1: reduce A to tridiagonal T = Qᵀ A Q, accumulating Q into eigvecs.
    tridiagonalize(
        &mut scratch.a_work[..n * n],
        n,
        eigvecs,
        &mut scratch.v_buf[..n],
        &mut scratch.p_buf[..n],
        &mut scratch.w_buf[..n],
    );

    // Extract diagonal into eigvals and subdiagonal into scratch.subdiag.
    // After tridiagonalize, a_work is tridiagonal:
    //   a_work[i*n + i]     = T[i, i]
    //   a_work[(i+1)*n + i] = T[i+1, i] = T[i, i+1]  (symmetric)
    //   everything else is 0.
    for (i, e) in eigvals.iter_mut().enumerate().take(n) {
        *e = scratch.a_work[i * n + i];
    }
    for i in 0..(n - 1) {
        scratch.subdiag[i] = scratch.a_work[(i + 1) * n + i];
    }
    if n > 0 {
        scratch.subdiag[n - 1] = 0.0; // Sentinel required by tqli.
    }

    // Loud refusal if phase 1 ever produced a non-finite d/e (Issue 931):
    // QL cannot converge from a poisoned tridiagonal — refuse with the phase
    // named instead of spinning.
    if n > 0 {
        refuse_non_finite_intermediate("after tridiagonalize", eigvals, &scratch.subdiag, n, 0, 0);
    }

    // Phase 2: implicit-shift QL on the tridiagonal, accumulating into eigvecs.
    tqli_implicit_shift(
        eigvals,
        &mut scratch.subdiag[..n],
        n,
        eigvecs,
        max_iters_per_eigval,
    );
}

/// Householder tridiagonalization with explicit Q accumulation.
///
/// Reduces `a` (n×n symmetric row-major) to tridiagonal form in place, and
/// accumulates the orthogonal factor into `q` (which must start as `I`).
/// Post-condition: `a` is tridiagonal (the lower triangle below the subdiagonal
/// is zeroed by the reflections; the upper triangle is symmetric).
///
/// Uses scratch buffers `v`, `p`, `w` (each length `n`).
///
/// Algorithm: Golub-van Loan §8.3.1. For each `k = 0..n-2`, compute the
/// Householder vector that zeros `a[k+2..n, k]`, apply `H · A · H` to the
/// bottom-right block, and accumulate `Q · H` into `q`.
fn tridiagonalize(
    a: &mut [f64],
    n: usize,
    q: &mut [f64],
    v: &mut [f64],
    p: &mut [f64],
    w: &mut [f64],
) {
    if n <= 2 {
        // 1x1 and 2x2 symmetric matrices are already tridiagonal.
        return;
    }
    for k in 0..(n - 2) {
        let block_start = k + 1;
        let block_size = n - block_start;

        // ||x||² where x = a[k+1..n, k] is the part of column k below the diagonal.
        let mut sigma = 0.0_f64;
        for i in block_start..n {
            let aik = a[i * n + k];
            sigma += aik * aik;
        }
        if sigma < f64::MIN_POSITIVE {
            // Column already zero below the diagonal — skip (no reflection needed).
            continue;
        }
        let norm_x = sigma.sqrt();

        // alpha = -sign(a[k+1, k]) * ||x||  (sign chosen to avoid cancellation in v[0]).
        let a_k1_k = a[block_start * n + k];
        let alpha = if a_k1_k >= 0.0 { -norm_x } else { norm_x };

        // v[0..block_size] = x - alpha * e_1
        //   v[0] = a[k+1, k] - alpha
        //   v[i>0] = a[k+1+i, k]
        v[0] = a_k1_k - alpha;
        for i in 1..block_size {
            v[i] = a[(block_start + i) * n + k];
        }

        let mut v_norm_sq = 0.0_f64;
        for &vi in v.iter().take(block_size) {
            v_norm_sq += vi * vi;
        }
        if v_norm_sq < f64::MIN_POSITIVE {
            continue;
        }
        let beta = 2.0 / v_norm_sq;

        // Apply A_sub ← H A_sub H where A_sub = a[block_start..n, block_start..n].
        // For symmetric A_sub, this factors as:
        //   p = beta * A_sub * v
        //   K = beta * (v·p) / 2
        //   w = p - K * v
        //   A_sub ← A_sub - v wᵀ - w vᵀ   (symmetric rank-2 update)
        // Slice `v`/`w` down to `block_size` once so the inner loops carry no
        // bounds checks and LLVM can vectorize them (the arithmetic order is
        // unchanged, so the result stays bit-identical).
        let v_blk = &v[..block_size];
        for (i, p_i) in p.iter_mut().enumerate().take(block_size) {
            let row_offset = (block_start + i) * n + block_start;
            let a_row = &a[row_offset..row_offset + block_size];
            let mut s = 0.0_f64;
            for (&a_ij, &vj) in a_row.iter().zip(v_blk.iter()) {
                s += a_ij * vj;
            }
            *p_i = beta * s;
        }
        let p_blk = &p[..block_size];
        let mut vp = 0.0_f64;
        for (&vi, &pi) in v_blk.iter().zip(p_blk.iter()) {
            vp += vi * pi;
        }
        let k_coef = beta * vp * 0.5;
        for ((w_i, &p_i), &v_i) in w[..block_size]
            .iter_mut()
            .zip(p_blk.iter())
            .zip(v_blk.iter())
        {
            *w_i = p_i - k_coef * v_i;
        }
        // Symmetric rank-2 update: A_sub[i,j] -= v[i]*w[j] + w[i]*v[j].
        // Cache-friendly: one pass per row of A_sub; v, w fit in L1/L2.
        let w_blk = &w[..block_size];
        for i in 0..block_size {
            let vi = v_blk[i];
            let wi = w_blk[i];
            let row_offset = (block_start + i) * n + block_start;
            let a_row = &mut a[row_offset..row_offset + block_size];
            for ((a_ij, &wj), &vj) in a_row.iter_mut().zip(w_blk.iter()).zip(v_blk.iter()) {
                *a_ij -= vi * wj + wi * vj;
            }
        }

        // Set the new subdiagonal value at (k+1, k) and zero the rest of column
        // k below the subdiagonal. Symmetric: also update row k.
        a[block_start * n + k] = alpha;
        a[k * n + block_start] = alpha;
        for i in (block_start + 1)..n {
            a[i * n + k] = 0.0;
            a[k * n + i] = 0.0;
        }

        // Accumulate Q ← Q · H where H acts on rows/cols block_start..n.
        // For each row i of Q: s = beta · (Q[i, block_start..n] · v);
        // then Q[i, block_start..n] -= s · v.
        for q_row in q[..n * n].chunks_exact_mut(n) {
            let row_slice = &mut q_row[block_start..block_start + block_size];
            let mut s = 0.0_f64;
            for (&q_ij, &vj) in row_slice.iter().zip(v_blk.iter()) {
                s += q_ij * vj;
            }
            s *= beta;
            for (q_ij, &vj) in row_slice.iter_mut().zip(v_blk.iter()) {
                *q_ij -= s * vj;
            }
        }
    }
}

/// Implicit-shift QL iteration on a symmetric tridiagonal matrix.
///
/// `d[0..n]` is the diagonal; `e[0..n]` is the subdiagonal with `e[i] = T[i, i+1]`
/// for `i = 0..n-1` and `e[n-1]` a sentinel (set to 0). Both are modified in
/// place: on return, `d` holds the eigenvalues (unsorted) and `e` is destroyed.
///
/// `z[n*n]` accumulates the eigenvectors as columns: column `k` of `z` (row-major
/// layout: indices `z[k], z[n+k], ..., z[(n-1)*n + k]`) is the eigenvector for
/// `d[k]` on return. `z` must start as the orthogonal factor `Q` from
/// [`tridiagonalize`] (or identity if the input was already tridiagonal).
///
/// Numerical Recipes `tqli` (3rd ed. §11.3) port. The implicit-shift technique
/// avoids the `O(n)` per-iteration shift application that would otherwise be
/// required for explicit-shift QR/QL; the bulge-chase applies `O(n)` Givens
/// rotations per iteration, each costing `O(n)` for eigenvector accumulation,
/// giving `O(n²)` per deflation and `O(n³)` overall.
fn tqli_implicit_shift(
    d: &mut [f64],
    e: &mut [f64],
    n: usize,
    z: &mut [f64],
    max_iters_per_eigval: usize,
) {
    if n <= 1 {
        return;
    }
    e[n - 1] = 0.0; // Sentinel: the QL bulge-chase never reads e[n-1] legitimately.

    for l in 0..n {
        let mut iter = 0;
        loop {
            // LAPACK `dsteqr`-style global convergence scale: the NR-local
            // check `|e[m]| + dd == dd` (with `dd = |d[m]| + |d[m+1]|`) only
            // fires when `|e[m]|` is negligible relative to the LOCAL
            // eigenvalue magnitudes. For matrices with tiny eigenvalues
            // (near-singular Grams from higher-order R=2 features when
            // `n_samples < d_h`, e.g. KARC large-d_h configs), `dd` is O(1e-10)
            // and the local check requires `|e[m]| < ~1e-26` — below machine
            // precision relative to the OVERALL matrix scale, so QL never
            // deflates and spins forever.
            //
            // Fix: also deflate when `|e[m]| ≤ eps · max(|d|)` over the full
            // diagonal. This is the LAPACK `RELTOL = eps · max(|D|)` criterion.
            // For O(1) eigenvalues this is equivalent to the NR check (the OR
            // is dominated by whichever fires first); for tiny-eigenvalue
            // matrices it provides the missing global-scale fallback.
            let global_max_d: f64 = d.iter().take(n).map(|x| x.abs()).fold(0.0_f64, f64::max);
            let reltol = f64::EPSILON * global_max_d;

            // Find smallest m in [l, n-1) with |e[m]| negligible relative to
            // (|d[m]| + |d[m+1]|). This is the bottom of the active block [l, m]:
            // the matrix splits there, so we can deflate [l, m] independently.
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                // NR's exact check: (|e[m]| + dd) == dd, i.e. |e[m]| is below
                // the rounding threshold of dd.
                //
                // OR the LAPACK global-scale check: |e[m]| ≤ eps · max(|d|).
                if e[m].abs() + dd == dd || e[m].abs() <= reltol {
                    break;
                }
                m += 1;
            }
            if m == l {
                // e[l] is negligible → eigenvalue at position l is deflated.
                break;
            }

            // Loud refusal on non-finite intermediates (Issue 931): a poisoned
            // d/e would never deflate (every negligible test reads false
            // against NaN) — refuse at once instead of spinning to the cap.
            refuse_non_finite_intermediate("ql sweep", d, e, m + 1, l, iter + 1);

            iter += 1;
            if iter > max_iters_per_eigval {
                panic!(
                    "symmetric_eig: QL failed to converge at l={} after {} iterations \
                     (d[l..l+2]={:?}, e[l..l+2]={:?})",
                    l,
                    max_iters_per_eigval,
                    &d[l..(l + 2).min(n)],
                    &e[l..(l + 2).min(n)]
                );
            }

            // Wilkinson shift from the leading 2×2 block at (l, l+1).
            // This is the eigenvalue of [[d[l], e[l]], [e[l], d[l+1]]] closer
            // to d[l+1] (QL pushes mass toward the lower-left).
            let g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = hypot_safe(g, 1.0);
            // The shift: μ = d[m] - (d[l] - μ') where μ' is the closer-of-two
            // eigenvalue of the 2×2 block. Numerical Recipes condenses this
            // to a single expression; see NR §11.3 for the algebra.
            let r_signed = if g >= 0.0 { r } else { -r };
            let mut g = d[m] - d[l] + e[l] / (g + r_signed);

            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;
            let mut broke_with_r_zero = false;

            // Bulge chase: from i = m-1 down to l. Each Givens rotation moves
            // the "bulge" one position toward the top-left; the final rotation
            // at i = l zeros out e[l], deflating d[l].
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r = hypot_safe(f, g);
                e[i + 1] = r;
                if r == 0.0 {
                    // Underflow recovery: if r is exactly zero, f and g are
                    // both zero, meaning e[i] = 0 (so s·e[i] = 0) AND g (the
                    // bulge from the previous step) is 0. Recover by deflating
                    // at i+1 and restarting.
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    broke_with_r_zero = true;
                    break;
                }
                s = f / r;
                c = g / r;
                let g_new = d[i + 1] - p;
                let r2 = (d[i] - g_new) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = g_new + p;
                g = c * r2 - b;

                // Eigenvector accumulation: rotate columns i and i+1 of z by
                // [[c, s], [-s, c]] (the inverse transpose of the active rotation).
                // Applies to ALL rows of z, not just the active block, because the
                // QL step is a similarity transform on the full matrix.
                // Row-wise via chunks_exact_mut: one bounds-checked pair of
                // loads per row instead of two index computations + four
                // bounds checks. `z[..,i]` is loaded once and reused (the
                // original read it twice), so the arithmetic is identical.
                for row in z[..n * n].chunks_exact_mut(n) {
                    let z_i = row[i];
                    let f = row[i + 1];
                    row[i + 1] = s * z_i + c * f;
                    row[i] = c * z_i - s * f;
                }
            }

            if broke_with_r_zero {
                // Restart the outer loop (recompute m).
                continue;
            }

            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}
