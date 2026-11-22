//! Fused multi-branch FUNCATTN — the branch-parallel composition cell
//! (Issue 1238; MBT arXiv:2611.19800 middle-placement shape).
//!
//! `K` matched-width FUNCATTN branches run over one shared token set; their
//! outputs are aggregated by an UNLEARNED fusion policy. This is
//! COMPOSITION-class fusion (every branch always executes, fixed ratio) —
//! distinct from `chiar_blend`, which is ROUTING-class (an input-dependent
//! gate picks which of two operators dominates per token).
//!
//! ```text
//! for b in 0..K (ascending — the fold order is pinned):
//!     xv_b  = x · P_bᵀ                     // in-proj   (n, d_b)
//!     xb_b  = x · P_basisᵀ  (shared, A4)   // or xv_b when no shared basis
//!     h_b   = FUNCATTN(xb_b, xv_b; W_b)    // (n, d_b)
//!     out  += scale · h_b · R_bᵀ           // out-proj  (n, d_model)
//! scale = 1/√K (SumInvSqrtK, jointly trained) | 1/K (Mean, frozen snapshots)
//! ```
//!
//! The scale law is pinned by the Issue-1238 verdict: `1/√K` is legitimate
//! only under JOINT training (later layers adapt to the scale); independently
//! trained snapshots are complete predictors, so they fuse by the plain mean —
//! `1/√K` would inflate the output by ~√K exactly when the predictors agree.
//!
//! Determinism (Issue 940 discipline): branches accumulate into `out` in
//! ascending index order with plain `+=`; there is no tie rule (no pick).
//! Latent-only: nothing here crosses a sync seam.
//!
//! Hot path is allocation-free once [`BranchParallelScratch`] is sized.

use katgpt_core::funcattn::{
    FuncAttnBasis, FuncAttnConfig, FuncAttnError, FuncAttnScratch, funcattn_forward,
};

/// Unlearned fusion policy for the branch sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BranchFusion {
    /// `Σ h_b / √K` — the MBT Table-6 parameter-free normalized sum. Only for
    /// branches trained JOINTLY under this scale.
    SumInvSqrtK,
    /// `Σ h_b / K` — the plain mean. The pinned policy for independently
    /// trained (frozen) snapshots.
    Mean,
}

impl BranchFusion {
    /// Scale applied to every branch's contribution for `k_branches` branches.
    #[inline]
    pub fn scale(self, k_branches: usize) -> f32 {
        let k = k_branches.max(1) as f32;
        match self {
            Self::SumInvSqrtK => 1.0 / k.sqrt(),
            Self::Mean => 1.0 / k,
        }
    }
}

/// Shape + FUNCATTN hyper-parameters shared by every branch (matched width).
#[derive(Clone, Copy, Debug)]
pub struct BranchParallelConfig {
    /// Model (input/output) width.
    pub d_model: usize,
    /// Per-branch FUNCATTN width `d_b`.
    pub d_branch: usize,
    /// Per-branch basis count `k` (the k×k solve runs PER branch).
    pub k: usize,
    pub basis: FuncAttnBasis,
    pub alpha: f32,
    pub temperature: f32,
    pub cholesky_jitter: f32,
    pub fusion: BranchFusion,
}

impl BranchParallelConfig {
    #[inline]
    fn funcattn(&self) -> FuncAttnConfig {
        FuncAttnConfig {
            d: self.d_branch,
            k: self.k,
            basis: self.basis,
            alpha: self.alpha,
            temperature: self.temperature,
            cholesky_jitter: self.cholesky_jitter,
        }
    }

    /// Parameter count of ONE branch (in-proj + FUNCATTN + out-proj), not
    /// counting a shared basis projection.
    pub fn branch_params(&self) -> usize {
        let (m, d, k) = (self.d_model, self.d_branch, self.k);
        2 * m * d + k * d + 3 * d * d
    }
}

/// Borrowed weights of one branch. Row-major throughout.
#[derive(Clone, Copy, Debug)]
pub struct BranchWeights<'a> {
    /// In-proj `P_b`, `(d_branch, d_model)`.
    pub in_proj: &'a [f32],
    /// Optional SHARED basis in-proj `P_basis`, `(d_branch, d_model)` — the
    /// Table-5 shared-slice-basis axis. `None` ⇒ the basis reads `xv_b`.
    pub basis_in_proj: Option<&'a [f32]>,
    /// `W_Φ`, `(k, d_branch)` (pass the same slice to every branch to share Φ).
    pub w_basis: &'a [f32],
    pub w_q: &'a [f32],
    pub w_k: &'a [f32],
    pub w_v: &'a [f32],
    /// Out-proj `R_b`, `(d_model, d_branch)`.
    pub out_proj: &'a [f32],
}

/// Pre-allocated scratch; reuse across calls (zero-alloc hot path).
pub struct BranchParallelScratch {
    xv: Vec<f32>,
    xb: Vec<f32>,
    h: Vec<f32>,
    fa: FuncAttnScratch,
}

impl BranchParallelScratch {
    pub fn new(n: usize, cfg: &BranchParallelConfig) -> Self {
        let nd = n * cfg.d_branch;
        Self {
            xv: vec![0.0; nd],
            xb: vec![0.0; nd],
            h: vec![0.0; nd],
            fa: FuncAttnScratch::new(n, cfg.d_branch, cfg.k),
        }
    }
}

/// `out[i,:] = W · x[i,:]` for `n` rows; `W` is `(rows, cols)`.
#[inline]
fn project_rows(x: &[f32], w: &[f32], n: usize, cols: usize, rows: usize, out: &mut [f32]) {
    for i in 0..n {
        let xr = &x[i * cols..(i + 1) * cols];
        let or = &mut out[i * rows..(i + 1) * rows];
        for (r, o) in or.iter_mut().enumerate() {
            let wr = &w[r * cols..(r + 1) * cols];
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += wr[c] * xr[c];
            }
            *o = acc;
        }
    }
}

/// Branch-parallel forward. `x` is `(n, d_model)`; `out` is `(n, d_model)`
/// and is OVERWRITTEN. Branch order is the slice order (ascending fold).
pub fn branch_parallel_forward(
    x: &[f32],
    branches: &[BranchWeights<'_>],
    cfg: &BranchParallelConfig,
    scratch: &mut BranchParallelScratch,
    out: &mut [f32],
) -> Result<(), FuncAttnError> {
    let (m, d) = (cfg.d_model, cfg.d_branch);
    let n = x.len().checked_div(m).unwrap_or(0);
    debug_assert_eq!(out.len(), n * m, "out must be (n, d_model)");
    debug_assert!(scratch.xv.len() >= n * d, "scratch sized for a smaller n");
    out.fill(0.0);
    if n == 0 || branches.is_empty() {
        return Ok(());
    }
    let nd = n * d;
    let scale = cfg.fusion.scale(branches.len());
    let fa_cfg = cfg.funcattn();
    for br in branches {
        debug_assert_eq!(br.in_proj.len(), d * m);
        debug_assert_eq!(br.out_proj.len(), m * d);
        project_rows(x, br.in_proj, n, m, d, &mut scratch.xv[..nd]);
        let xb: &[f32] = match br.basis_in_proj {
            Some(pb) => {
                project_rows(x, pb, n, m, d, &mut scratch.xb[..nd]);
                &scratch.xb[..nd]
            }
            None => &scratch.xv[..nd],
        };
        funcattn_forward(
            xb,
            &scratch.xv[..nd],
            br.w_basis,
            br.w_q,
            br.w_k,
            br.w_v,
            &fa_cfg,
            &mut scratch.fa,
            &mut scratch.h[..nd],
        )?;
        // out[i, r] += scale · Σ_c R[r, c] · h[i, c]
        for i in 0..n {
            let hr = &scratch.h[i * d..(i + 1) * d];
            let or = &mut out[i * m..(i + 1) * m];
            for (r, o) in or.iter_mut().enumerate() {
                let rr = &br.out_proj[r * d..(r + 1) * d];
                let mut acc = 0.0f32;
                for c in 0..d {
                    acc += rr[c] * hr[c];
                }
                *o += scale * acc;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(d_branch: usize, fusion: BranchFusion) -> BranchParallelConfig {
        BranchParallelConfig {
            d_model: 4,
            d_branch,
            k: 3,
            basis: FuncAttnBasis::Sigmoid,
            alpha: 0.01,
            temperature: 0.1,
            cholesky_jitter: 1e-6,
            fusion,
        }
    }

    fn eye(d: usize) -> Vec<f32> {
        let mut w = vec![0.0; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        w
    }

    fn ramp(len: usize, salt: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 37 + salt * 11) % 29) as f32 / 29.0) - 0.5)
            .collect()
    }

    #[test]
    fn scale_law() {
        assert_eq!(BranchFusion::Mean.scale(2), 0.5);
        assert!((BranchFusion::SumInvSqrtK.scale(2) - 0.707_106_77).abs() < 1e-6);
        assert_eq!(BranchFusion::Mean.scale(1), 1.0);
        assert_eq!(BranchFusion::SumInvSqrtK.scale(1), 1.0);
    }

    /// One identity-projected branch equals bare FUNCATTN at d_branch = d_model.
    #[test]
    fn single_branch_identity_matches_bare_funcattn() {
        let c = cfg(4, BranchFusion::Mean);
        let n = 6;
        let x = ramp(n * 4, 1);
        let (id, wb) = (eye(4), ramp(3 * 4, 2));
        let br = BranchWeights {
            in_proj: &id,
            basis_in_proj: None,
            w_basis: &wb,
            w_q: &id,
            w_k: &id,
            w_v: &id,
            out_proj: &id,
        };
        let mut s = BranchParallelScratch::new(n, &c);
        let mut out = vec![0.0; n * 4];
        branch_parallel_forward(&x, &[br], &c, &mut s, &mut out).unwrap();
        let mut fs = FuncAttnScratch::new(n, 4, 3);
        let mut want = vec![0.0; n * 4];
        funcattn_forward(
            &x,
            &x,
            &wb,
            &id,
            &id,
            &id,
            &c.funcattn(),
            &mut fs,
            &mut want,
        )
        .unwrap();
        assert_eq!(out, want);
    }

    /// Two identical branches under Mean reproduce one branch exactly
    /// (the ensemble's agreement case); under 1/√K they inflate by √2.
    #[test]
    fn mean_of_identical_branches_is_identity() {
        let n = 5;
        let x = ramp(n * 4, 3);
        let (p, wb, q, r) = (ramp(2 * 4, 4), ramp(3 * 2, 5), eye(2), ramp(4 * 2, 6));
        let br = BranchWeights {
            in_proj: &p,
            basis_in_proj: None,
            w_basis: &wb,
            w_q: &q,
            w_k: &q,
            w_v: &q,
            out_proj: &r,
        };
        let c1 = cfg(2, BranchFusion::Mean);
        let mut s = BranchParallelScratch::new(n, &c1);
        let mut one = vec![0.0; n * 4];
        let mut two = vec![0.0; n * 4];
        branch_parallel_forward(&x, &[br], &c1, &mut s, &mut one).unwrap();
        branch_parallel_forward(&x, &[br, br], &c1, &mut s, &mut two).unwrap();
        for (a, b) in one.iter().zip(&two) {
            assert!((a - b).abs() <= 1e-6 * a.abs().max(1.0));
        }
        let cs = cfg(2, BranchFusion::SumInvSqrtK);
        branch_parallel_forward(&x, &[br, br], &cs, &mut s, &mut two).unwrap();
        for (a, b) in one.iter().zip(&two) {
            assert!((a * std::f32::consts::SQRT_2 - b).abs() <= 1e-5 * a.abs().max(1.0));
        }
    }

    #[test]
    fn deterministic_rerun_bit_identical() {
        let n = 7;
        let x = ramp(n * 4, 7);
        let (p0, p1, pb) = (ramp(8, 8), ramp(8, 9), ramp(8, 10));
        let (wb, q, r0, r1) = (ramp(6, 11), eye(2), ramp(8, 12), ramp(8, 13));
        let b0 = BranchWeights {
            in_proj: &p0,
            basis_in_proj: Some(&pb),
            w_basis: &wb,
            w_q: &q,
            w_k: &q,
            w_v: &q,
            out_proj: &r0,
        };
        let b1 = BranchWeights {
            in_proj: &p1,
            out_proj: &r1,
            ..b0
        };
        let c = cfg(2, BranchFusion::SumInvSqrtK);
        let mut s = BranchParallelScratch::new(n, &c);
        let mut a = vec![0.0; n * 4];
        let mut b = vec![0.0; n * 4];
        branch_parallel_forward(&x, &[b0, b1], &c, &mut s, &mut a).unwrap();
        branch_parallel_forward(&x, &[b0, b1], &c, &mut s, &mut b).unwrap();
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a), bits(&b));
    }
}
