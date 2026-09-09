//! Quantization-Error Compensating Reader-LoRA (Issue 565 / Research 463).
//!
//! A deterministically-constructed (modelless) low-rank or sparse correction
//! that compensates for the error introduced by quantizing a weight matrix.
//! Given a reference weight `W` (f32) and its quantized form `W_q` (any
//! precision), compute the error `E = W − dequant(W_q)`, then approximate `E`
//! as either:
//!
//! - **Weight-space SVD** (Strategy A): `E ≈ A·B` via truncated SVD of `E`.
//!   Minimizes `‖E − A·B‖_F`. Treats all weight errors equally.
//! - **Output-space SVD** (Strategy B, data-aware): `E ≈ A·B` via reduced-rank
//!   regression on `E·X`, where `X` is a calibration set of input activations.
//!   Minimizes `‖E·X − A·B·X‖_F` — the error on actual outputs. Strictly better
//!   rank-budget choice than weight-space when some weight directions are
//!   never activated by the calibration distribution. (Izenman 1975.)
//! - **Top-K sparse bypass** (Strategy D): store the worst-K elements of `E`
//!   explicitly as a COO sparse matrix. Targets outlier errors at low MAC cost
//!   but with gather overhead on SIMD/WASM hardware.
//!
//! At inference (the reader-LoRA hot-swap of Plan 025):
//!
//! ```text
//! y = W_q · x + α · correction(x)
//! ```
//!
//! where `correction(x) = B · (A · x)` (dense) or `S · x` (sparse).
//!
//! # Modelless compliance
//!
//! All three constructors are closed-form (no gradient descent, no optimizer
//! state). The weight-space + output-space variants consume Plan 301's
//! [`thin_svd_into`] (default-on); the sparse variant is a partial-sort + copy.
//! The corrected weights freeze as a `NeuronShard` via the freeze/thaw
//! ecosystem — this primitive only produces the correction factors.
//!
//! # The Small-Kernel Parameter Paradox (Research 463 §2.4.1)
//!
//! Rank-r LoRA adds `r·(out + in)` parameters to correct a weight matrix of
//! `out × in` parameters. On a 4096×4096 LLM layer, rank-8 is 0.39% overhead;
//! on a 32×288 Moka conv, rank-8 is 27.8% overhead. Small CNNs lack the
//! low-dimensional weight structure that makes LoRA effective for LLMs. This
//! primitive ships regardless — it's reusable substrate for larger models
//! (LLM weights, future game networks) where the error manifold is genuinely
//! low-rank. The PoC ([`Issue 565`](../../../.issues/565_quant_error_lora_poc.md))
//! tests whether it helps on the 105K-param Moka network (predicted: FAIL).

use crate::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};

// ─── Dense low-rank (Strategy A: weight-space SVD) ──────────────────────────

/// A deterministically-constructed low-rank reader-LoRA that compensates for
/// the quantization error of a weight matrix.
///
/// Stores `a` (down-projection, row-major `[rank × in_dim]`) and `b`
/// (up-projection, row-major `[out_dim × rank]`). At inference:
/// `correction(x) = b · (a · x)`, yielding an `[out_dim]` output.
///
/// Construct via [`QuantErrorLora::from_error`] (weight-space SVD) or
/// [`QuantErrorLora::from_error_data_aware`] (output-space SVD).
///
/// Layout note: `a` is `[rank × in_dim]` so `a · x` produces a `[rank]`
/// intermediate; `b` is `[out_dim × rank]` so `b · (intermediate)` produces
/// `[out_dim]`. Both are row-major to match the conv/linear weight convention
/// (`weight[oc * in_dim + ic]`).
pub struct QuantErrorLora {
    /// Down-projection `[rank × in_dim]`, row-major. Row `r` dotted with `x`
    /// yields intermediate `r`.
    pub a: Vec<f32>,
    /// Up-projection `[out_dim × rank]`, row-major. Row `o` dotted with the
    /// `[rank]` intermediate yields output `o`.
    pub b: Vec<f32>,
    /// Scaling factor applied to the correction. Default 1.0.
    pub alpha: f32,
    /// Effective rank stored (≤ the requested rank; trimmed if the error
    /// matrix has fewer significant singular values).
    pub rank: usize,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl QuantErrorLora {
    /// Construct from a reference weight matrix + its quantized form via
    /// **weight-space truncated SVD** (Strategy A).
    ///
    /// Computes `E = W − dequant(W_q)` (the caller passes `w_ref` = `W` and
    /// `w_quant_dequant` = `dequant(W_q)`), then `E ≈ B·A` via rank-r truncated
    /// SVD minimizing `‖E − B·A‖_F`.
    ///
    /// The SVD is computed on the transpose if `out_dim < in_dim` (one-sided
    /// Jacobi requires `m ≥ n`); the factor roles swap transparently.
    ///
    /// `alpha` scales the correction (1.0 = full SVD reconstruction of the
    /// leading-rank error subspace; <1.0 damps it).
    ///
    /// # Panics
    ///
    /// Panics if `w_ref.len() != out_dim * in_dim` or
    /// `w_quant_dequant.len() != out_dim * in_dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_error(
        w_ref: &[f32],
        w_quant_dequant: &[f32],
        out_dim: usize,
        in_dim: usize,
        rank: usize,
        alpha: f32,
        svd_result: &mut SvdResultScratch,
        svd_work: &mut SvdScratch,
    ) -> Self {
        assert_eq!(w_ref.len(), out_dim * in_dim, "w_ref length mismatch");
        assert_eq!(
            w_quant_dequant.len(),
            out_dim * in_dim,
            "w_quant_dequant length mismatch"
        );

        // Error matrix E [out_dim × in_dim], row-major.
        let mut e = vec![0.0f32; out_dim * in_dim];
        for i in 0..e.len() {
            e[i] = w_ref[i] - w_quant_dequant[i];
        }

        let effective_rank = rank.min(out_dim).min(in_dim).max(1);
        let mut a = vec![0.0f32; effective_rank * in_dim];
        let mut b = vec![0.0f32; out_dim * effective_rank];

        // one-sided Jacobi requires m_rows >= n_cols. For E [out × in] with
        // out < in, factor E^T [in × out] = U Σ V^T, so E = V Σ U^T. The rank-r
        // approx of E is then V_r Σ_r U_r^T — A = Σ_r U_r^T (the [r × in] side
        // since V_r is [out × r]... wait, let's be careful with the transpose).
        //
        // E = V Σ U^T where:
        //   - V is [out_dim × out_dim] right singular vectors of E^T (= left
        //     singular vectors of E). Columns are the OUTPUT-space directions.
        //   - U is [in_dim × out_dim] left singular vectors of E^T (= right
        //     singular vectors of E). Columns are the INPUT-space directions.
        //
        // thin_svd_into of E^T [in_dim × out_dim]:
        //   - left_singular_vector(j)  → column j of U_Et, length in_dim  (input direction)
        //   - right_singular_vector(j) → column j of V_Et, length out_dim (output direction)
        //
        // For E = U_E Σ V_E^T (the SVD of E itself):
        //   - U_E = V_Et  (output directions), V_E = U_Et (input directions)
        //
        // We want E_r = B · A where B [out × r], A [r × in]:
        //   B[:, k] = σ_k · (output direction k) = σ_k · V_Et[:, k]
        //   A[k, :] = (input direction k)^T       = U_Et[:, k]^T
        if out_dim >= in_dim {
            // Factor E [out × in] directly. thin_svd_into(E):
            //   left_singular_vector(j)  → U_E[:, j], length out_dim (output dir)
            //   right_singular_vector(j) → V_E[:, j], length in_dim  (input dir)
            thin_svd_into(&e, out_dim, in_dim, svd_result, svd_work);
            let n_singular = svd_result.len();
            let r = effective_rank.min(n_singular);
            for k in 0..r {
                let sigma = svd_result.singular_value(k);
                // A row k = σ_k · V_E[:, k] (input-space direction), length in_dim.
                let v_k = svd_result.right_singular_vector(k);
                for c in 0..in_dim {
                    a[k * in_dim + c] = sigma * v_k[c];
                }
                // B column k = U_E[:, k] (output-space direction), length out_dim.
                // Stored row-major [out_dim × r], so b[o * r + k] = U_E[o, k].
                let u_k = svd_result.left_singular_vector(k);
                for o in 0..out_dim {
                    b[o * r + k] = u_k[o];
                }
            }
            Self {
                a,
                b,
                alpha,
                rank: r,
                in_dim,
                out_dim,
            }
        } else {
            // Factor E^T [in × out]. thin_svd_into(E^T):
            //   left_singular_vector(j)  → U_Et[:, j], length in_dim  (input dir of E)
            //   right_singular_vector(j) → V_Et[:, j], length out_dim (output dir of E)
            let mut e_t = vec![0.0f32; in_dim * out_dim];
            for o in 0..out_dim {
                for i in 0..in_dim {
                    // E^T[i, o] = E[o, i]
                    e_t[i * out_dim + o] = e[o * in_dim + i];
                }
            }
            thin_svd_into(&e_t, in_dim, out_dim, svd_result, svd_work);
            let n_singular = svd_result.len();
            let r = effective_rank.min(n_singular);
            for k in 0..r {
                let sigma = svd_result.singular_value(k);
                // A row k = σ_k · (input direction k) = σ_k · U_Et[:, k], length in_dim.
                let u_k = svd_result.left_singular_vector(k);
                for c in 0..in_dim {
                    a[k * in_dim + c] = sigma * u_k[c];
                }
                // B column k = (output direction k) = V_Et[:, k], length out_dim.
                let v_k = svd_result.right_singular_vector(k);
                for o in 0..out_dim {
                    b[o * r + k] = v_k[o];
                }
            }
            Self {
                a,
                b,
                alpha,
                rank: r,
                in_dim,
                out_dim,
            }
        }
    }

    /// Construct via **output-space (data-aware) reduced-rank regression**
    /// (Strategy B). Minimizes `‖E·X − B·A·X‖_F` — the error on actual outputs
    /// over the calibration set — rather than `‖E − B·A‖_F`.
    ///
    /// Given the calibration set `X` [`in_dim × n_cal`] (column-major: column
    /// `c` is calibration sample `c`), this:
    /// 1. Computes `E_out = E · X` [`out_dim × n_cal`] (the output error on
    ///    each calibration sample).
    /// 2. SVDs `E_out = U Σ V^T`, takes the top-r left singular vectors
    ///    `U_r` [`out_dim × r`] — the principal OUTPUT-error directions under
    ///    the calibration distribution.
    /// 3. Projects the full error onto those directions:
    ///    `A = U_r^T · E` [`r × in_dim`], `B = U_r` [`out_dim × r`].
    ///
    /// This is strictly better rank-budget allocation than weight-space SVD
    /// when some weight directions are never activated by the calibration
    /// distribution (they get zero rank budget here). (Izenman 1975.)
    ///
    /// `x_cal` is column-major `[in_dim × n_cal]`: `x_cal[c * in_dim + i]` is
    /// the `i`-th feature of calibration sample `c`.
    ///
    /// # Panics
    ///
    /// Panics on length mismatches.
    #[allow(clippy::too_many_arguments)]
    pub fn from_error_data_aware(
        w_ref: &[f32],
        w_quant_dequant: &[f32],
        out_dim: usize,
        in_dim: usize,
        x_cal: &[f32], // [in_dim × n_cal], column-major
        n_cal: usize,
        rank: usize,
        alpha: f32,
        svd_result: &mut SvdResultScratch,
        svd_work: &mut SvdScratch,
    ) -> Self {
        assert_eq!(w_ref.len(), out_dim * in_dim);
        assert_eq!(w_quant_dequant.len(), out_dim * in_dim);
        assert_eq!(x_cal.len(), in_dim * n_cal);

        // Error matrix E [out_dim × in_dim], row-major.
        let mut e = vec![0.0f32; out_dim * in_dim];
        for i in 0..e.len() {
            e[i] = w_ref[i] - w_quant_dequant[i];
        }

        // E_out = E · X  [out_dim × n_cal], row-major.
        // E_out[o, c] = Σ_i E[o, i] · X[i, c].
        let mut e_out = vec![0.0f32; out_dim * n_cal];
        for o in 0..out_dim {
            let e_row = &e[o * in_dim..(o + 1) * in_dim];
            let out_row = &mut e_out[o * n_cal..(o + 1) * n_cal];
            for c in 0..n_cal {
                let x_col = &x_cal[c * in_dim..(c + 1) * in_dim];
                let mut acc = 0.0f32;
                let mut i = 0;
                while i + 4 <= in_dim {
                    acc += e_row[i] * x_col[i]
                        + e_row[i + 1] * x_col[i + 1]
                        + e_row[i + 2] * x_col[i + 2]
                        + e_row[i + 3] * x_col[i + 3];
                    i += 4;
                }
                while i < in_dim {
                    acc += e_row[i] * x_col[i];
                    i += 1;
                }
                out_row[c] = acc;
            }
        }

        let effective_rank = rank.min(out_dim).min(in_dim).max(1);

        // SVD E_out [out_dim × n_cal] → U [out_dim], Σ, V [n_cal].
        // U columns are the principal OUTPUT-error directions.
        // one-sided Jacobi needs m >= n; factor E_out directly if out >= n_cal,
        // else factor E_out^T and swap roles.
        let n_singular = if out_dim >= n_cal {
            thin_svd_into(&e_out, out_dim, n_cal, svd_result, svd_work);
            svd_result.len()
        } else {
            // E_out^T [n_cal × out_dim]: left_singular_vector(j) is length out_dim
            // (= output direction of E_out), which is what we want for U_r.
            let mut e_out_t = vec![0.0f32; n_cal * out_dim];
            for o in 0..out_dim {
                for c in 0..n_cal {
                    e_out_t[c * out_dim + o] = e_out[o * n_cal + c];
                }
            }
            thin_svd_into(&e_out_t, n_cal, out_dim, svd_result, svd_work);
            svd_result.len()
        };

        let r = effective_rank.min(n_singular);
        // A = U_r^T · E  [r × in_dim]. A[k, i] = Σ_o U_r[o, k] · E[o, i].
        // B = U_r        [out_dim × r]. B[o, k] = U_r[o, k].
        let mut a = vec![0.0f32; r * in_dim];
        let mut b = vec![0.0f32; out_dim * r];
        for k in 0..r {
            let u_k = svd_result.left_singular_vector(k); // length out_dim (E_out rows)
            // B column k = U_r[:, k].
            for o in 0..out_dim {
                b[o * r + k] = u_k[o];
            }
            // A row k = U_r[:, k]^T · E.
            for i in 0..in_dim {
                let mut acc = 0.0f32;
                let mut o = 0;
                while o + 4 <= out_dim {
                    acc += u_k[o] * e[o * in_dim + i]
                        + u_k[o + 1] * e[(o + 1) * in_dim + i]
                        + u_k[o + 2] * e[(o + 2) * in_dim + i]
                        + u_k[o + 3] * e[(o + 3) * in_dim + i];
                    o += 4;
                }
                while o < out_dim {
                    acc += u_k[o] * e[o * in_dim + i];
                    o += 1;
                }
                a[k * in_dim + i] = acc;
            }
        }

        Self {
            a,
            b,
            alpha,
            rank: r,
            in_dim,
            out_dim,
        }
    }

    /// Apply the dense correction into `y`: `y += alpha * B · (A · x)`.
    ///
    /// This is the reader-LoRA hot path. Uses a caller-supplied `scratch`
    /// `[rank]` buffer for the intermediate `A · x` to avoid per-call
    /// allocation.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != in_dim`, `y.len() != out_dim`, or
    /// `scratch.len() < rank`.
    #[inline]
    pub fn apply_correction_into(&self, x: &[f32], y: &mut [f32], scratch: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim, "x length != in_dim");
        debug_assert_eq!(y.len(), self.out_dim, "y length != out_dim");
        debug_assert!(scratch.len() >= self.rank, "scratch < rank");

        let r = self.rank;
        // Intermediate: A · x → scratch[r].
        #[allow(clippy::needless_range_loop)] // stride math: k indexes scratch[k] AND k*self.in_dim offset into self.a
        for k in 0..r {
            let a_row = &self.a[k * self.in_dim..(k + 1) * self.in_dim];
            let mut acc = 0.0f32;
            let mut i = 0;
            while i + 4 <= self.in_dim {
                acc += a_row[i] * x[i]
                    + a_row[i + 1] * x[i + 1]
                    + a_row[i + 2] * x[i + 2]
                    + a_row[i + 3] * x[i + 3];
                i += 4;
            }
            while i < self.in_dim {
                acc += a_row[i] * x[i];
                i += 1;
            }
            scratch[k] = acc;
        }
        // Accumulate: y += alpha * B · intermediate.
        let scale = self.alpha;
        #[allow(clippy::needless_range_loop)] // stride math: o indexes y[o] AND o*r offset into self.b
        for o in 0..self.out_dim {
            let b_row = &self.b[o * r..(o + 1) * r];
            let mut acc = 0.0f32;
            for k in 0..r {
                acc += b_row[k] * scratch[k];
            }
            y[o] += scale * acc;
        }
    }

    /// Reconstruct the full approximated error matrix `E ≈ B · A` into the
    /// caller-supplied `out` buffer `[out_dim * in_dim]` (row-major). Useful
    /// for measuring reconstruction quality (Frobenius error).
    pub fn reconstruct_error_into(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.out_dim * self.in_dim);
        let r = self.rank;
        for o in 0..self.out_dim {
            let b_row = &self.b[o * r..(o + 1) * r];
            let out_row = &mut out[o * self.in_dim..(o + 1) * self.in_dim];
            #[allow(clippy::needless_range_loop)] // stride math: i indexes out_row[i] AND k*self.in_dim+i offset into self.a
            for i in 0..self.in_dim {
                let mut acc = 0.0f32;
                #[allow(clippy::needless_range_loop)] // stride math: k indexes b_row[k] AND k*self.in_dim+i offset into self.a
                for k in 0..r {
                    acc += b_row[k] * self.a[k * self.in_dim + i];
                }
                out_row[i] = acc;
            }
        }
    }
}

// ─── Sparse bypass (Strategy D: top-K worst errors) ─────────────────────────

/// A sparse (COO) correction storing the top-K worst quantization errors.
///
/// At inference: `y += S · x` where `S` is the sparse residual matrix. This
/// targets outlier weights explicitly at low MAC cost but with gather overhead
/// (random-access reads into `x`). See Research 463 §2.7 for the MAC vs gather
/// trade-off on SIMD/WASM hardware.
pub struct SparseErrorBypass {
    /// Row indices of the selected elements (length = nnz).
    pub rows: Vec<u32>,
    /// Column indices of the selected elements (length = nnz).
    pub cols: Vec<u32>,
    /// Values of the selected elements (length = nnz).
    pub vals: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl SparseErrorBypass {
    /// Construct by selecting the top-`fraction` (by `|E[i,j]|`) elements of
    /// the error matrix `E = W − dequant(W_q)`.
    ///
    /// `fraction` ∈ (0, 1]: 0.05 selects the worst 5% of elements.
    pub fn from_error(
        w_ref: &[f32],
        w_quant_dequant: &[f32],
        out_dim: usize,
        in_dim: usize,
        fraction: f32,
    ) -> Self {
        assert_eq!(w_ref.len(), out_dim * in_dim);
        assert_eq!(w_quant_dequant.len(), out_dim * in_dim);
        let total = out_dim * in_dim;
        let k = ((total as f32) * fraction.clamp(0.0, 1.0)).round() as usize;
        let k = k.max(1).min(total);

        // Build (abs_error, flat_index) pairs, partial-sort top-k by abs_error desc.
        let mut indexed: Vec<(f32, usize)> = (0..total)
            .map(|i| ((w_ref[i] - w_quant_dequant[i]).abs(), i))
            .collect();
        // Partial sort: partition so the top-k are at the front (unordered is fine
        // — we don't need them sorted, just selected).
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            b.0.total_cmp(&a.0)
        });

        let mut rows = Vec::with_capacity(k);
        let mut cols = Vec::with_capacity(k);
        let mut vals = Vec::with_capacity(k);
        for &(abs_err, flat) in &indexed[..k] {
            let _ = abs_err; // unused; we store the signed error.
            let o = flat / in_dim;
            let i = flat % in_dim;
            rows.push(o as u32);
            cols.push(i as u32);
            vals.push(w_ref[flat] - w_quant_dequant[flat]);
        }

        Self {
            rows,
            cols,
            vals,
            out_dim,
            in_dim,
        }
    }

    /// Apply the sparse correction into `y`: `y[row] += val * x[col]` for each
    /// stored element. No scratch needed (sparse scatter).
    #[inline]
    pub fn apply_correction_into(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert_eq!(y.len(), self.out_dim);
        for n in 0..self.vals.len() {
            let o = self.rows[n] as usize;
            let i = self.cols[n] as usize;
            y[o] += self.vals[n] * x[i];
        }
    }

    /// Number of non-zero elements stored.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }
}

// ─── Diagnostics ────────────────────────────────────────────────────────────

/// Compute the cosine similarity between two vectors. Returns 0.0 if either
/// has zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na * nb).sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// Measure how much of the error matrix's energy the rank-r LoRA captures.
///
/// Returns the fraction `‖E_r‖_F² / ‖E‖_F²` — 1.0 means the LoRA captures
/// all the error energy (E is rank ≤ r); lower means the error is spread
/// across more directions.
///
/// Used by the PoC's T12 task (Small-Kernel Paradox confirmation): if this
/// fraction is low at rank-8, the error matrix is near-full-rank and the
/// Small-Kernel Paradox holds.
pub fn captured_energy_fraction(lora: &QuantErrorLora, w_ref: &[f32], w_quant_dequant: &[f32]) -> f32 {
    let total = w_ref.len();
    let mut e_full_sq = 0.0f32;
    for i in 0..total {
        let e = w_ref[i] - w_quant_dequant[i];
        e_full_sq += e * e;
    }
    if e_full_sq <= 0.0 {
        return 1.0;
    }
    let mut e_approx = vec![0.0f32; total];
    lora.reconstruct_error_into(&mut e_approx);
    let mut e_approx_sq = 0.0f32;
    for v in &e_approx {
        e_approx_sq += v * v;
    }
    e_approx_sq / e_full_sq
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn dense_lora_reconstructs_rank_r_matrix() {
        // Build a known rank-3 error matrix: E = B_true · A_true, B [6×3], A [3×8].
        let out_dim = 6;
        let in_dim = 8;
        let rank_true = 3;
        let b_true: Vec<f32> = (0..out_dim * rank_true)
            .map(|i| (i as f32) * 0.1 - 1.0)
            .collect();
        let a_true: Vec<f32> = (0..rank_true * in_dim)
            .map(|i| (i as f32) * 0.05 + 0.5)
            .collect();
        let mut e = vec![0.0f32; out_dim * in_dim];
        for o in 0..out_dim {
            for i in 0..in_dim {
                let mut acc = 0.0;
                for k in 0..rank_true {
                    acc += b_true[o * rank_true + k] * a_true[k * in_dim + i];
                }
                e[o * in_dim + i] = acc;
            }
        }
        // W_ref = E, W_quant = 0 (so the error IS the full matrix).
        let zero = vec![0.0f32; out_dim * in_dim];
        let mut svd_result = SvdResultScratch::with_capacity(out_dim, in_dim);
        let mut svd_work = SvdScratch::with_capacity(in_dim, out_dim);
        let lora = QuantErrorLora::from_error(&e, &zero, out_dim, in_dim, rank_true, 1.0, &mut svd_result, &mut svd_work);
        let frac = captured_energy_fraction(&lora, &e, &zero);
        // A rank-3 matrix should be nearly fully captured at rank 3.
        assert!(frac > 0.999, "captured fraction {frac} should be >0.999 for a rank-3 matrix at rank 3");
    }

    #[test]
    fn dense_lora_apply_matches_reconstruction() {
        // apply_correction_into should match reconstruct + matvec.
        let out_dim = 4;
        let in_dim = 5;
        let rank = 2;
        let w_ref: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.01).collect();
        let w_q: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.005).collect();
        let mut svd_result = SvdResultScratch::with_capacity(out_dim, in_dim);
        let mut svd_work = SvdScratch::with_capacity(in_dim, out_dim);
        let lora = QuantErrorLora::from_error(&w_ref, &w_q, out_dim, in_dim, rank, 1.0, &mut svd_result, &mut svd_work);

        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.1).collect();
        let mut y_apply = vec![0.0f32; out_dim];
        let mut scratch = vec![0.0f32; rank];
        lora.apply_correction_into(&x, &mut y_apply, &mut scratch);

        let mut e_approx = vec![0.0f32; out_dim * in_dim];
        lora.reconstruct_error_into(&mut e_approx);
        let mut y_recon = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let row = &e_approx[o * in_dim..(o + 1) * in_dim];
            for i in 0..in_dim {
                y_recon[o] += row[i] * x[i];
            }
        }
        for o in 0..out_dim {
            assert!(approx_eq(y_apply[o], y_recon[o], 1e-4), "output {o}: apply={:.6} recon={:.6}", y_apply[o], y_recon[o]);
        }
    }

    #[test]
    fn dense_lora_handles_tall_matrix() {
        // out_dim < in_dim (the Moka conv case: 32 × 288). SVD via transpose.
        let out_dim = 4;
        let in_dim = 12;
        let w_ref: Vec<f32> = (0..out_dim * in_dim).map(|i| ((i as f32) * 0.1).sin()).collect();
        let w_q = vec![0.0f32; out_dim * in_dim];
        let mut svd_result = SvdResultScratch::with_capacity(in_dim, out_dim);
        let mut svd_work = SvdScratch::with_capacity(out_dim, in_dim);
        let lora = QuantErrorLora::from_error(&w_ref, &w_q, out_dim, in_dim, 4, 1.0, &mut svd_result, &mut svd_work);
        assert!(lora.rank <= 4);
        assert_eq!(lora.a.len(), lora.rank * in_dim);
        assert_eq!(lora.b.len(), out_dim * lora.rank);
    }

    #[test]
    fn data_aware_lora_matches_weight_space_on_identity_calibration() {
        // If the calibration set spans all input directions (identity matrix),
        // output-space SVD should match weight-space SVD's captured energy
        // fraction (both pick the top-r energy directions).
        let out_dim = 4;
        let in_dim = 4;
        let w_ref: Vec<f32> = (0..out_dim * in_dim).map(|i| ((i as f32) * 0.1).sin()).collect();
        let w_q = vec![0.0f32; out_dim * in_dim];
        let n_cal = in_dim;
        // Identity calibration: X = I.
        let mut x_cal = vec![0.0f32; in_dim * n_cal];
        for i in 0..in_dim {
            x_cal[i * in_dim + i] = 1.0;
        }
        let mut svd_result = SvdResultScratch::with_capacity(out_dim, n_cal);
        let mut svd_work = SvdScratch::with_capacity(n_cal, out_dim);
        let lora_da = QuantErrorLora::from_error_data_aware(
            &w_ref, &w_q, out_dim, in_dim, &x_cal, n_cal, 2, 1.0, &mut svd_result, &mut svd_work,
        );
        let frac_da = captured_energy_fraction(&lora_da, &w_ref, &w_q);

        let mut svd_result2 = SvdResultScratch::with_capacity(out_dim, in_dim);
        let mut svd_work2 = SvdScratch::with_capacity(in_dim, out_dim);
        let lora_ws = QuantErrorLora::from_error(&w_ref, &w_q, out_dim, in_dim, 2, 1.0, &mut svd_result2, &mut svd_work2);
        let frac_ws = captured_energy_fraction(&lora_ws, &w_ref, &w_q);

        // With identity calibration, both should capture the same fraction
        // (the top-2 singular directions' energy). Allow tolerance for SVD
        // numerical differences.
        assert!(
            (frac_da - frac_ws).abs() < 0.05,
            "data-aware frac {frac_da:.4} should ≈ weight-space frac {frac_ws:.4} on identity calibration"
        );
    }

    #[test]
    fn sparse_bypass_selects_worst_errors() {
        let out_dim = 4;
        let in_dim = 4;
        // Error: one large outlier, rest small.
        let mut w_ref = vec![0.0f32; out_dim * in_dim];
        let w_q = vec![0.0f32; out_dim * in_dim];
        w_ref[0] = 10.0; // the big error
        w_ref[1..].fill(0.01);
        let sparse = SparseErrorBypass::from_error(&w_ref, &w_q, out_dim, in_dim, 0.25);
        // 25% of 16 = 4 elements. The worst-4 by |error| includes the 10.0 outlier.
        assert_eq!(sparse.nnz(), 4);
        assert!(sparse.vals.contains(&10.0), "the outlier must be selected");
    }

    #[test]
    fn sparse_bypass_apply_corrects_selected_elements() {
        let out_dim = 3;
        let in_dim = 3;
        let w_ref: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let w_q = vec![0.0f32; 9];
        let sparse = SparseErrorBypass::from_error(&w_ref, &w_q, out_dim, in_dim, 1.0); // all elements
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0f32; out_dim];
        sparse.apply_correction_into(&x, &mut y);
        // y[0] = 1*1, y[1] = 2*2, y[2] = 3*3.
        assert!(approx_eq(y[0], 1.0, 1e-5));
        assert!(approx_eq(y[1], 4.0, 1e-5));
        assert!(approx_eq(y[2], 9.0, 1e-5));
    }

    #[test]
    fn cosine_similarity_basic() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 1.0, 1e-5));
        let c = vec![0.0, 1.0, 0.0];
        assert!(approx_eq(cosine_similarity(&a, &c), 0.0, 1e-5));
    }
}
