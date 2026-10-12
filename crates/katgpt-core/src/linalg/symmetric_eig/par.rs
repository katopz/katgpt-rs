//! Parallel symmetric eigendecomposition — rayon row-parallel variant.
//!
//! Same algorithm + storage convention as [`super::symmetric_eig`] (Householder
//! tridiagonalization + implicit-shift QL), with two row-parallel regions:
//!
//! - **Householder phase**: each of the three O(n²)-per-reflection loops
//!   (matrix-vector product, symmetric rank-2 update, Q accumulation) is
//!   row-parallel. O(n) parallel calls total.
//! - **QL phase**: rotations within one deflation are batched into a single
//!   parallel call per deflation. The (c, s, i) tuples are recorded during
//!   the serial bulge chase, then applied to all rows of `z` in one
//!   `par_chunks_mut(n)` pass per deflation. O(n) parallel calls total.
//!
//! # Why batched QL (the lesson from the first attempt)
//!
//! An earlier draft parallelized each Givens rotation individually. With
//! O(n²) rotations per eigendecomposition and ~10-50 µs rayon overhead per
//! `par_chunks_mut` call, this added O(n² × 50µs) overhead — at n=1024
//! that's ~50 seconds of pure scheduling cost, dwarfing the ~6 seconds of
//! actual work. Measured slowdown was 13-62× (Issue 187 T5 first attempt).
//!
//! Batching drops the parallel-call count from O(n²) to O(n) by recording
//! all (c, s, i) tuples for a deflation's bulge chase, then applying them
//! to each row in a single sequential pass. Per-row work becomes O(n)
//! FLOPs instead of O(1), and per-call work becomes O(n²) FLOPs across
//! `n` row chunks — comfortably above rayon's break-even point.
//!
//! # Determinism
//!
//! Each parallel hot loop processes disjoint rows in fully sequential work.
//! No cross-row reductions, no shared mutable state. The result is
//! **bit-identical** to the serial path on the same input — verified by the
//! `par_vs_serial_*` tests under the `karc_householder_eig_par` feature.
//!
//! # When to use
//!
//! `n ≥ 256` on a multi-core host. Below that, rayon's thread-pool overhead
//! dominates. The serial path remains the default; this variant is gated on
//! the `karc_householder_eig_par` feature.

use rayon::prelude::*;

use super::SymmetricEigScratch;

/// Parallel symmetric eigendecomposition. Drop-in replacement for
/// [`super::symmetric_eig`] when the `karc_householder_eig_par` feature is on.
///
/// See the module-level docs for the determinism contract + the QL batching
/// rationale. Same panics + post-conditions as the serial version.
pub fn symmetric_eig_par(
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

    // Loud refusal on a poisoned input — same seam as the serial path
    // (Issue 931); the parallel path must refuse identically to stay
    // bit-identical on every input the two share.
    super::refuse_non_finite_input(a_in, n);

    scratch.ensure_capacity(n);

    // Initialize eigvecs = I (row-parallel — each row is independent).
    eigvecs.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        // memset the row + one diagonal store, instead of n branchy writes.
        row.fill(0.0);
        if i < row.len() {
            row[i] = 1.0;
        }
    });

    // Copy A into the working buffer (serial — single memcpy).
    scratch.a_work[..n * n].copy_from_slice(&a_in[..n * n]);

    // Phase 1: parallel Householder tridiagonalization.
    tridiagonalize_par(
        &mut scratch.a_work[..n * n],
        n,
        eigvecs,
        &mut scratch.v_buf[..n],
        &mut scratch.p_buf[..n],
        &mut scratch.w_buf[..n],
    );

    // Extract diagonal + subdiagonal (serial — O(n)).
    for (i, e) in eigvals.iter_mut().enumerate().take(n) {
        *e = scratch.a_work[i * n + i];
    }
    for i in 0..(n - 1) {
        scratch.subdiag[i] = scratch.a_work[(i + 1) * n + i];
    }
    if n > 0 {
        scratch.subdiag[n - 1] = 0.0;
    }

    // Loud refusal if phase 1 ever produced a non-finite d/e — same seam as
    // the serial path (Issue 931).
    if n > 0 {
        super::refuse_non_finite_intermediate("after tridiagonalize", eigvals, &scratch.subdiag, n, 0, 0);
    }

    // Phase 2: batched-parallel implicit-shift QL.
    tqli_implicit_shift_par(
        eigvals,
        &mut scratch.subdiag[..n],
        n,
        eigvecs,
        max_iters_per_eigval,
    );
}

/// Parallel Householder tridiagonalization.
///
/// Three row-parallel regions per reflection (matvec, rank-2 update, Q
/// accumulation). Each parallel call processes one reflection's worth of
/// work — O(block_size × block_size) FLOPs across O(block_size) row chunks
/// — comfortably above rayon's break-even.
#[allow(clippy::too_many_arguments)]
fn tridiagonalize_par(
    a: &mut [f64],
    n: usize,
    q: &mut [f64],
    v: &mut [f64],
    p: &mut [f64],
    w: &mut [f64],
) {
    if n <= 2 {
        return;
    }
    for k in 0..(n - 2) {
        let block_start = k + 1;
        let block_size = n - block_start;

        // ||x||² where x = a[k+1..n, k] — serial reduction over a single column.
        let mut sigma = 0.0_f64;
        for i in block_start..n {
            let aik = a[i * n + k];
            sigma += aik * aik;
        }
        if sigma < f64::MIN_POSITIVE {
            continue;
        }
        let norm_x = sigma.sqrt();

        let a_k1_k = a[block_start * n + k];
        let alpha = if a_k1_k >= 0.0 { -norm_x } else { norm_x };

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

        // ── p = β · A_sub · v  ──────────────────────────────────────────
        // Row-parallel: each row i ∈ [0..block_size) reads its own row of
        // A_sub and the full v vector; writes only p[i]. We reborrow `a` as
        // shared for the parallel read (`&mut T` is `!Sync`); NLL releases
        // the borrow before the next mutation of `a`.
        let v_slice = &v[..block_size];
        let a_shared: &[f64] = a;
        let a_block_start = block_start * n + block_start;
        let a_block_stride = n;
        p[..block_size]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, p_i)| {
                let row_offset = a_block_start + i * a_block_stride;
                let mut s = 0.0_f64;
                for (j, &vj) in v_slice.iter().enumerate() {
                    s += a_shared[row_offset + j] * vj;
                }
                *p_i = beta * s;
            });

        let mut vp = 0.0_f64;
        for i in 0..block_size {
            vp += v[i] * p[i];
        }
        let k_coef = beta * vp * 0.5;
        for i in 0..block_size {
            w[i] = p[i] - k_coef * v[i];
        }

        // ── Symmetric rank-2 update: A_sub[i,j] -= v[i]*w[j] + w[i]*v[j] ──
        // Row-parallel across all n rows of a; rows outside the active
        // block [block_start, block_start+block_size) are no-ops (cheap
        // branch in each chunk). The per-row `&mut` slices are disjoint.
        let v_block = &v[..block_size];
        let w_block = &w[..block_size];
        let bs = block_start;
        let bz = block_size;
        a.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            if i < bs || i >= bs + bz {
                return;
            }
            let vi = v_block[i - bs];
            let wi = w_block[i - bs];
            let row_sub = &mut row[bs..bs + bz];
            for j in 0..bz {
                row_sub[j] -= vi * w_block[j] + wi * v_block[j];
            }
        });

        // Set the new subdiagonal + zero the rest of column k (serial — O(n)).
        a[block_start * n + k] = alpha;
        a[k * n + block_start] = alpha;
        for i in (block_start + 1)..n {
            a[i * n + k] = 0.0;
            a[k * n + i] = 0.0;
        }

        // ── Q accumulation: Q[i, block_start..n] -= s · v ──────────────
        // Row-parallel: each row computes its own s = β·(Q[i,:]·v), then
        // updates only its own slice. No cross-row dependency.
        let v_block_q = &v[..block_size];
        q.par_chunks_mut(n).for_each(|q_row| {
            let row_slice = &mut q_row[block_start..block_start + block_size];
            let mut s = 0.0_f64;
            for (j, q_ij) in row_slice.iter().enumerate() {
                s += q_ij * v_block_q[j];
            }
            s *= beta;
            for (j, slot) in row_slice.iter_mut().enumerate() {
                *slot -= s * v_block_q[j];
            }
        });
    }
}

/// Batched-parallel implicit-shift QL iteration.
///
/// **The batching rule (lesson from Issue 187 T5 first attempt):** each
/// Givens rotation's row-update is O(n) FLOPs across n rows — too small to
/// amortize rayon's ~10-50 µs per-call overhead. Naive per-rotation
/// parallelism causes a 13-62× slowdown at n=1024 (1M rayon calls).
///
/// This implementation batches all rotations within one deflation's bulge
/// chase into a single parallel pass: the bulge chase runs serially,
/// recording (c, s, i) into `rot_buf`; then one `par_chunks_mut(n)` call
/// applies all rotations to every row of z. Per-row work becomes O(n)
/// FLOPs (one rotation per deflation step); per-call work becomes O(n²)
/// FLOPs total. The parallel-call count drops from O(n²) to O(n).
///
/// **Bit-identity preserved:** the order of operations within each row
/// matches the serial path exactly — the k-th rotation uses the result of
/// the (k-1)-th rotation on the same row, in the same order. Only the
/// assignment of rows to threads varies.
fn tqli_implicit_shift_par(
    d: &mut [f64],
    e: &mut [f64],
    n: usize,
    z: &mut [f64],
    max_iters_per_eigval: usize,
) {
    if n <= 1 {
        return;
    }
    e[n - 1] = 0.0;

    // Scratch buffer for the recorded rotations of one deflation's bulge
    // chase. Grows on first use; reused across deflations. Worst case is
    // O(n) rotations per deflation.
    let mut rot_buf: Vec<(f64, f64, usize)> = Vec::with_capacity(n);

    for l in 0..n {
        let mut iter = 0;
        loop {
            // LAPACK `dsteqr`-style global convergence scale — see the serial
            // path in `super::tqli_implicit_shift` for the full rationale.
            // The NR-local check `|e[m]| + dd == dd` does not fire for tiny
            // eigenvalues (near-singular Grams from higher-order R=2 features
            // when `n_samples < d_h`); the OR with `|e[m]| ≤ eps · max(|d|)`
            // provides the global-scale fallback. Bit-identical to the serial
            // path (same check, same OR).
            let global_max_d: f64 = d.iter().take(n).map(|x| x.abs()).fold(0.0_f64, f64::max);
            let reltol = f64::EPSILON * global_max_d;

            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() + dd == dd || e[m].abs() <= reltol {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }

            // Loud refusal on non-finite intermediates — same seam as the
            // serial path (Issue 931).
            super::refuse_non_finite_intermediate("ql sweep", d, e, m + 1, l, iter + 1);

            iter += 1;
            if iter > max_iters_per_eigval {
                panic!(
                    "symmetric_eig_par: QL failed to converge at l={} after {} iterations \
                     (d[l..l+2]={:?}, e[l..l+2]={:?})",
                    l,
                    max_iters_per_eigval,
                    &d[l..(l + 2).min(n)],
                    &e[l..(l + 2).min(n)]
                );
            }

            let g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r = super::hypot_safe(g, 1.0);
            let r_signed = if g >= 0.0 { r } else { -r };
            let mut g = d[m] - d[l] + e[l] / (g + r_signed);

            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;
            let mut broke_with_r_zero = false;

            rot_buf.clear();

            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r = super::hypot_safe(f, g);
                e[i + 1] = r;
                if r == 0.0 {
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

                // Record this rotation for the batched row update.
                rot_buf.push((c, s, i));
            }

            // If the bulge chase produced any rotations, apply them to z
            // in a single batched parallel pass. Each row reads z[k, i]
            // and z[k, i+1] for each rotation in order, applying the same
            // sequence of operations as the serial path. Per-row work is
            // O(rot_buf.len()) FLOPs; per-call work is O(n·rot_buf.len()).
            if !rot_buf.is_empty() {
                z.par_chunks_mut(n).for_each(|row| {
                    for &(c_r, s_r, i_r) in &rot_buf {
                        // Load row[i_r] once (the original read it twice) —
                        // identical arithmetic, one fewer bounds-checked load.
                        let z_i = row[i_r];
                        let f_local = row[i_r + 1];
                        row[i_r + 1] = s_r * z_i + c_r * f_local;
                        row[i_r] = c_r * z_i - s_r * f_local;
                    }
                });
            }

            if broke_with_r_zero {
                continue;
            }

            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}
