//! Joint demixer + canonicalization (T1.4 / T1.5): SVD-init by EVR-greedy
//! class order, deterministic loading-axis ALS, and between-class rank-1
//! reallocation.
//!
//! # The block update (derivation)
//!
//! The model is `X ≈ Σ_r a_r ⊗ M_r` with each component born into a fixed
//! class σ(r) — `a_r` lives on axis σ(r), `M_r` spans the other two axes and
//! is FROZEN at its SVD birth value. Updating `a_r` alone with every other
//! component fixed minimizes `‖E_r − a ⊗ M_r‖²` over `a` where
//! `E_r = X − Σ_{s≠r} a_s ⊗ M_s`: since `‖M_r‖_F = 1`, the minimizer is the
//! closed-form projection
//!
//! ```text
//! a_r[i] = ⟨E_r restricted to σ=i, M_r⟩          (one tensor contraction)
//! ```
//!
//! — no normal-equation solve, no step size, no schedule: one contraction,
//! with the ε-floored scalar divide (`1/σ`) absorbed at construction (slices
//! are unit-norm, so the update itself is divide-free). Each update is the
//! exact joint-loss minimizer over its block, so the sweep loss is monotone
//! non-increasing (the ALS monotonicity of Carroll & Chang 1970 / Harshman
//! 1970, restricted to the loading blocks). Because the slice matrices never
//! move, a component's between-class structure cannot drift under ALS — the
//! class assignment is invariant by construction, exactly as the plan
//! requires.

use crate::linalg::{TuckerConfig, TuckerResultScratch, TuckerScratch, tucker_decompose_into};
use crate::subspace_phase_gate::thin_svd_into;

use super::svd::{complementary_dims, push_class_comps_into, unfold_into};
use super::types::{
    InitMode, NORM_EPS, SliceClass, SliceDecomposition, SliceTcaConfig, SliceTcaError,
    SliceTcaScratch, Tensor3, add_rank1_slice, contract_axis, frob_sq, norm_2,
};

// ─── Initialization ─────────────────────────────────────────────────────────

/// Whether `tucker_decompose_into` accepts this shape + ranks (its per-mode
/// `SVD_MAX_RANK` bound — see `TuckerConfig::new`). This is the documented
/// reason the joint initializer falls back to the Gram-SVD path for large
/// shapes: `[64,128,32]`'s mode-1 unfolding has min-dim 128 > 16.
fn hosvd_supported(shape: [usize; 3], ranks: [usize; 3]) -> bool {
    let tucker_ranks = [ranks[0].max(1), ranks[1].max(1), ranks[2].max(1)];
    TuckerConfig::new(&shape, &tucker_ranks).is_ok()
}

/// Initialize every class with `ranks[σ] > 0`, appending components to `out`
/// in EVR-greedy class order (largest covariability share first; axis index
/// breaks share ties, so the order is deterministic).
pub(crate) fn init_components_into(
    x: &[f32],
    shape: [usize; 3],
    ranks: [usize; 3],
    cfg: &SliceTcaConfig,
    scratch: &mut SliceTcaScratch,
    out: &mut SliceDecomposition,
) -> Result<(), SliceTcaError> {
    if ranks == [0, 0, 0] {
        return Ok(());
    }
    let shares = {
        let mut s = [0.0f32; 3];
        for (axis, spec) in scratch.spec.iter().enumerate() {
            let take = cfg.share_rank.min(spec.len());
            let mut acc = 0.0f32;
            for &lambda in spec.iter().take(take) {
                acc += lambda;
            }
            s[axis] = acc;
        }
        s
    };
    let mut order: [usize; 3] = [0, 1, 2];
    order.sort_by(|&a, &b| shares[b].total_cmp(&shares[a]).then(a.cmp(&b)));

    let use_hosvd = match cfg.init {
        InitMode::GramSvd => false,
        InitMode::Hosvd => {
            if !hosvd_supported(shape, ranks) {
                return Err(SliceTcaError::HosvdShapeUnsupported);
            }
            true
        }
        InitMode::Auto => hosvd_supported(shape, ranks),
    };

    if use_hosvd {
        init_hosvd_into(x, shape, ranks, cfg, scratch, out, &order)
    } else {
        for &axis in &order {
            if ranks[axis] == 0 {
                continue;
            }
            super::svd::fit_single_class_cached_into(
                x,
                shape,
                axis,
                ranks[axis],
                cfg.energy_floor_tau,
                scratch,
                out,
            )?;
        }
        Ok(())
    }
}

/// HOSVD joint initialization (T1.4 — gives `linalg::tucker` its first
/// in-tree consumer): one `tucker_decompose_into` call factors ALL THREE
/// unfoldings; each active class reads its mode's factor columns as the `u_r`
/// basis, then the shared contraction stage builds the components.
fn init_hosvd_into(
    x: &[f32],
    shape: [usize; 3],
    ranks: [usize; 3],
    cfg: &SliceTcaConfig,
    scratch: &mut SliceTcaScratch,
    out: &mut SliceDecomposition,
    order: &[usize; 3],
) -> Result<(), SliceTcaError> {
    let tucker_ranks = [ranks[0].max(1), ranks[1].max(1), ranks[2].max(1)];
    let tucker_cfg = TuckerConfig::new(&shape, &tucker_ranks)
        .map_err(|_| SliceTcaError::HosvdShapeUnsupported)?;
    let mut tucker_scratch = TuckerScratch::with_capacity(&tucker_cfg);
    let mut tucker_result = TuckerResultScratch::with_capacity(&tucker_cfg);
    tucker_decompose_into(x, &tucker_cfg, &mut tucker_scratch, &mut tucker_result)
        .map_err(|_| SliceTcaError::HosvdShapeUnsupported)?;

    let floor_sq = cfg.energy_floor_tau * scratch.norm_sq();
    for &axis in order {
        if ranks[axis] == 0 {
            continue;
        }
        let (d, rest) = unfold_into(x, shape, axis, scratch);
        let (i_n, r_n) = tucker_result.factor_shape(axis);
        debug_assert_eq!(i_n, d);
        let factor = tucker_result.factor(axis);
        push_class_comps_into(
            shape,
            axis,
            ranks[axis].min(r_n),
            floor_sq,
            factor,
            &scratch.unfold[..d * rest],
            &mut scratch.y_vec[..d * rest],
            &mut scratch.slice_buf[..d * rest],
            &mut scratch.loading_buf[..d],
            out,
        )?;
    }
    Ok(())
}

// ─── ALS ────────────────────────────────────────────────────────────────────

/// Run `cfg.als_sweeps` deterministic loading-axis sweeps over the components
/// of `out`, maintaining the residual `E = X − Σ comps` in
/// `scratch.residual`, and record each post-sweep relative loss
/// `‖E‖²/‖X‖²` into `scratch` (monotone by the block-minimizer argument —
/// asserted by tests/benches, never assumed here).
pub(crate) fn als_sweeps_into(
    x: &[f32],
    shape: [usize; 3],
    cfg: &SliceTcaConfig,
    scratch: &mut SliceTcaScratch,
    out: &mut SliceDecomposition,
) {
    let n_comps = out.n_components;
    if n_comps == 0 || cfg.als_sweeps == 0 {
        return;
    }
    let norm_sq = frob_sq(x);
    let total = shape[0] * shape[1] * shape[2];

    // E = X − Σ comps (fixed component order).
    {
        let e = &mut scratch.residual[..total];
        e.copy_from_slice(x);
        for i in 0..n_comps {
            let class = out.classes[i];
            let loading = out.loading(i);
            let (slice, _r, _c) = out.slice_matrix(i);
            add_rank1_slice(e, shape, class, loading, slice, -1.0);
        }
    }

    for _sweep in 0..cfg.als_sweeps {
        if _sweep == 0 {
            // Record the pre-ALS (init) loss first so `last ≤ first` in the
            // G2 gate compares ALS output against the SVD-only start.
            let init = frob_sq(&scratch.residual[..total]);
            scratch.push_loss(init / norm_sq.max(NORM_EPS));
        }
        {
            let e = &mut scratch.residual[..total];
            for i in 0..n_comps {
                let class = out.classes[i];
                let axis = class.axis();
                let d = [shape[0], shape[1], shape[2]][axis];
                // E += a_old ⊗ M (remove component i from the residual).
                {
                    let loading = out.loading(i);
                    let (slice, _r, _c) = out.slice_matrix(i);
                    add_rank1_slice(e, shape, class, loading, slice, 1.0);
                }
                // a_new = ⟨E_σ, M⟩ into the staging buffer (unit-norm slice
                // — no divide), then splice back into the loading slot.
                {
                    let (slice, _r, _c) = out.slice_matrix(i);
                    let buf = &mut scratch.loading_buf[..d];
                    contract_axis(e, shape, class, slice, buf);
                    out.set_loading(i, buf);
                }
                // E −= a_new ⊗ M.
                {
                    let loading = out.loading(i);
                    let (slice, _r, _c) = out.slice_matrix(i);
                    add_rank1_slice(e, shape, class, loading, slice, -1.0);
                }
            }
        }
        let loss = frob_sq(&scratch.residual[..total]);
        scratch.push_loss(loss / norm_sq.max(NORM_EPS));
    }
}

// ─── Public fit entry points ────────────────────────────────────────────────

/// Fit with EXPLICIT per-class ranks — the pipeline every rank selector
/// (knee or blocked-CV) bottoms out in. Stages: shares (EVR-greedy init
/// order + spectra) → init (HOSVD when `TuckerConfig` accepts the shape,
/// Gram-SVD otherwise) → fixed-sweep ALS → canonicalize (sign rule,
/// deterministic trim, variance-desc sort). ALS per-sweep losses are left in
/// `scratch.last_sweep_losses()`.
pub fn fit_with_ranks_into(
    x: &[f32],
    shape: [usize; 3],
    ranks: [usize; 3],
    cfg: &SliceTcaConfig,
    scratch: &mut SliceTcaScratch,
    out: &mut SliceDecomposition,
) -> Result<(), SliceTcaError> {
    validate_fit(x, shape, ranks)?;
    // Shares + spectra (also seeds the EVR-greedy init order).
    let _shares = super::svd::covariability_shares_into(x, shape, cfg.share_rank, scratch);
    run_fit_after_shares(x, shape, ranks, cfg, scratch, out)
}

/// Crate-internal continuation for callers (`fit_slice_into`, the blocked
/// selector) that JUST ran the shares phase on the same tensor — skips the
/// second Gram+SVD pass (the shares phase already filled `spec`, `norm_sq`
/// and `u_cache`).
pub(crate) fn run_fit_after_shares(
    x: &[f32],
    shape: [usize; 3],
    ranks: [usize; 3],
    cfg: &SliceTcaConfig,
    scratch: &mut SliceTcaScratch,
    out: &mut SliceDecomposition,
) -> Result<(), SliceTcaError> {
    validate_fit(x, shape, ranks)?;
    *out = SliceDecomposition::empty(shape);
    scratch.clear_losses();

    init_components_into(x, shape, ranks, cfg, scratch, out)?;
    als_sweeps_into(x, shape, cfg, scratch, out);

    let floor_sq = cfg.energy_floor_tau * scratch.norm_sq();
    out.canonicalize(floor_sq);
    Ok(())
}

fn validate_fit(
    x: &[f32],
    shape: [usize; 3],
    ranks: [usize; 3],
) -> Result<(), SliceTcaError> {
    let total = shape[0] * shape[1] * shape[2];
    if x.len() != total {
        return Err(SliceTcaError::InputSizeMismatch {
            got: x.len(),
            expected: total,
        });
    }
    if shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return Err(SliceTcaError::ZeroDimension);
    }
    for axis in 0..3 {
        let d = [shape[0], shape[1], shape[2]][axis];
        let rest = total / d;
        let bound = d.min(rest).min(super::types::MAX_RANK_PER_CLASS);
        if ranks[axis] > bound {
            return Err(SliceTcaError::RankTooLarge {
                axis,
                rank: ranks[axis],
                bound,
            });
        }
    }
    Ok(())
}

/// Full modelless pipeline (the primary entry point): covariability shares →
/// sigmoid routing → EVR-knee rank selection per active class → joint init →
/// fixed-sweep ALS → canonicalization. `out` is replaced. The shares phase
/// runs ONCE — the fit continuation reuses its spectra/basis caches.
pub fn fit_slice_into(
    x: &[f32],
    shape: [usize; 3],
    cfg: &SliceTcaConfig,
    scratch: &mut SliceTcaScratch,
    out: &mut SliceDecomposition,
) -> Result<(), SliceTcaError> {
    let shares = super::svd::covariability_shares_into(x, shape, cfg.share_rank, scratch);
    let routes = super::svd::route(shares, cfg.route_theta, cfg.route_alpha);
    let ranks = super::rank::select_ranks_knee(scratch.spectra(), scratch.norm_sq(), routes, cfg);
    run_fit_after_shares(x, shape, ranks, cfg, scratch, out)
}

/// Relative reconstruction loss `‖X − X̃‖²_F / ‖X‖²_F` (allocates a
/// reconstruction internally — diagnostics/tests, not a hot path).
pub fn relative_loss(x: &Tensor3, decomp: &SliceDecomposition) -> f32 {
    let mut recon = Tensor3::zeros(x.shape);
    let _ = decomp.reconstruct_into(&mut recon);
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (&a, &b) in x.data.iter().zip(recon.data.iter()) {
        let d = a - b;
        num += d * d;
        den += a * a;
    }
    if den < NORM_EPS {
        return 0.0;
    }
    num / den
}

// ─── T1.5 — between-class rank-1 reallocation ───────────────────────────────

/// Move component `comp` to `new_class` by closed-form projection (T1.5):
/// rank-1-split the slice matrix `M ≈ s·b⊗c` (Eckart–Young, one small SVD),
/// re-express the SAME rank-1 triple with `new_class`'s axis as the loading,
/// renormalize (unit slice, amplitude in the loading), re-canonicalize.
///
/// Returns the introduced residual energy ratio `1 − s²/‖M‖²_F`: `0` for an
/// exactly rank-1 slice (the generic, loss-free case), positive for a
/// generic matrix slice (the documented-degenerate case — the component
/// loses everything beyond its best rank-1 direction; the caller decides
/// whether the ratio is acceptable).
pub fn reallocate_class(
    decomp: &mut SliceDecomposition,
    comp: usize,
    new_class: SliceClass,
    scratch: &mut SliceTcaScratch,
) -> Result<f32, SliceTcaError> {
    if comp >= decomp.n_components {
        return Err(SliceTcaError::ComponentOutOfRange {
            comp,
            n: decomp.n_components,
        });
    }
    let old_class = decomp.classes[comp];
    if old_class == new_class {
        return Ok(0.0);
    }
    let shape = decomp.shape;
    let (slice, rows, cols) = decomp.slice_matrix(comp);
    let m_norm = norm_2(slice);
    if m_norm <= NORM_EPS || slice.is_empty() {
        return Err(SliceTcaError::SliceSizeMismatch {
            got: slice.len(),
            expected: rows * cols,
        });
    }

    // Rank-1 split M ≈ s·b(α)⊗c(β) via the substrate SVD, tall-oriented.
    let (b_vec, c_vec, s) = rank1_split(slice, rows, cols, scratch);

    // Component ≈ s · a(σ) ⊗ b(α) ⊗ c(β); new loading on σ' ∈ {α, β}.
    let a = decomp.loading(comp).to_vec();
    let a_norm = norm_2(&a).max(NORM_EPS);
    let a_unit: Vec<f32> = a.iter().map(|&v| v / a_norm).collect();

    let (new_rows, new_cols) = complementary_dims(shape, new_class);
    let mut new_slice = vec![0.0f32; new_rows * new_cols];
    let new_loading: Vec<f32>;
    if new_class.axis() == old_class.slice_axes().0 {
        // Loading on the first complementary axis (alpha) ⇒ slice = a_unit ⊗ c
        // over (sigma, beta).
        new_loading = b_vec.iter().map(|&v| v * s * a_norm).collect();
        let d_c = c_vec.len();
        for (i, &ai) in a_unit.iter().enumerate() {
            for (j, &cj) in c_vec.iter().enumerate() {
                new_slice[i * d_c + j] = ai * cj;
            }
        }
    } else {
        // Loading on the second complementary axis (beta) ⇒ slice = a_unit ⊗ b
        // over (sigma, alpha).
        new_loading = c_vec.iter().map(|&v| v * s * a_norm).collect();
        let d_b = b_vec.len();
        for (i, &ai) in a_unit.iter().enumerate() {
            for (j, &bj) in b_vec.iter().enumerate() {
                new_slice[i * d_b + j] = ai * bj;
            }
        }
    }

    // Splice: rebuild without `comp`, push the new one, re-canonicalize.
    // Floor 0 — the caller's fit already trimmed; reallocation never trims.
    let mut rebuilt = SliceDecomposition::empty(shape);
    for i in 0..decomp.n_components {
        if i == comp {
            continue;
        }
        let class = decomp.classes[i];
        let weight = decomp.weights[i];
        let (slice, rows, cols) = decomp.slice_matrix(i);
        rebuilt.push_component(class, decomp.loading(i), slice, rows, cols, weight)?;
    }
    let weight = norm_2(&new_loading);
    rebuilt.push_component(new_class, &new_loading, &new_slice, new_rows, new_cols, weight)?;
    rebuilt.canonicalize(0.0);
    *decomp = rebuilt;

    Ok((1.0 - (s * s) / (m_norm * m_norm)).max(0.0))
}

/// Best rank-1 triple `(b, c, s)` of an `rows × cols` matrix via the
/// substrate `thin_svd_into`, transposing wide inputs so the Jacobi SVD
/// always sees a tall matrix.
fn rank1_split(
    m: &[f32],
    rows: usize,
    cols: usize,
    scratch: &mut SliceTcaScratch,
) -> (Vec<f32>, Vec<f32>, f32) {
    if rows >= cols {
        thin_svd_into(m, rows, cols, &mut scratch.svd_result, &mut scratch.svd_work);
        (
            scratch.svd_result.left_singular_vector(0).to_vec(),
            scratch.svd_result.right_singular_vector(0).to_vec(),
            scratch.svd_result.singular_value(0),
        )
    } else {
        // Mᵀ (cols × rows, tall) into scratch.trans, grown if needed
        // (fit-time growth is acceptable; steady-state reuse dominates).
        if scratch.trans.len() < rows * cols {
            scratch.trans.resize(rows * cols, 0.0);
        }
        let trans = &mut scratch.trans[..rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                trans[j * rows + i] = m[i * cols + j];
            }
        }
        thin_svd_into(
            trans,
            cols,
            rows,
            &mut scratch.svd_result,
            &mut scratch.svd_work,
        );
        (
            scratch.svd_result.right_singular_vector(0).to_vec(),
            scratch.svd_result.left_singular_vector(0).to_vec(),
            scratch.svd_result.singular_value(0),
        )
    }
}
