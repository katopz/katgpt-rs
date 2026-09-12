//! Rank selection (T2.1): the EVR-knee deterministic surrogate (default) and
//! an opt-in blocked grid selector.
//!
//! Both selectors are deterministic: fixed evaluation order, sequential
//! blocks, lexicographic `(R_n, R_t, R_k)` ascending tie-breaks. No rayon in
//! any path — the grid is small and the sequential reduction order IS the
//! determinism contract here.
//!
//! # What the blocked selector is (and is not)
//!
//! True held-out prediction CV is STRUCTURALLY IMPOSSIBLE for slice models:
//! every axis indexes free slice parameters (an entity-class slice has one
//! free column per episode; an episode-class loading has one free entry per
//! episode), so a model fit on some episodes predicts ~0 on held episodes at
//! EVERY rank — the CV surface is flat and under-selects by tie-break. Masked
//! fitting would restore a real CV surface but is a hyperparameter curriculum
//! the plan's determinism design deliberately excludes.
//!
//! The opt-in selector is therefore a **blocked structural-fit plateau
//! grid**: split the episodes into contiguous blocks, fit each candidate
//! rank triple on each block independently (the planted structure is present
//! in every block), score by the mean per-block relative residual, and pick
//! the smallest-rank candidate within an ε-plateau of the best score. On
//! clean planted data the plateau starts exactly at the planted ranks —
//! under-ranked fits sit above the plateau, over-ranked fits inside it but
//! larger. It cross-checks the knee surrogate's rank ORDER without claiming
//! held-out predictive validity.

use super::svd::numerical_rank_cap;
use super::types::{
    MAX_RANK_PER_CLASS, NORM_EPS, SliceDecomposition, SliceTcaConfig, SliceTcaError,
    SliceTcaScratch, Tensor3,
};

/// Fraction of an unfolding's spectral energy a class's kept spectrum must
/// cover to be eligible for more than one component (consumed through the
/// substrate's `numerical_rank`).
const NUMERICAL_RANK_ETA: f32 = 0.999;

/// Plateau half-width for the blocked selector: candidates whose mean block
/// residual is within this of the best are considered equivalent, and the
/// smallest total rank wins among them.
const PLATEAU_EPS: f32 = 0.05;

/// EVR-knee surrogate (default selector): for each routed-ON class, count
/// spectrum entries above the absolute energy floor
/// `λ_r ≥ τ · ‖X‖²_F`, capped by `MAX_RANK_PER_CLASS` and the substrate's
/// `numerical_rank` cumulative-energy cap. Routed-OFF classes get 0.
///
/// The knee is a SURROGATE for cross-validated rank — cheap, deterministic,
/// and calibrated on the Plan 596 synthetics (Bench 714 records the
/// surrogate-vs-blocked agreement evidence; it is not a proof for arbitrary
/// tensors).
pub fn select_ranks_knee(
    spectra: [&[f32]; 3],
    norm_sq: f32,
    routes: [f32; 3],
    cfg: &SliceTcaConfig,
) -> [usize; 3] {
    let mut ranks = [0usize; 3];
    let mut sqrt_buf = Vec::new();
    let floor = cfg.energy_floor_tau * norm_sq;
    for axis in 0..3 {
        if routes[axis] < 0.5 {
            continue;
        }
        let spec = spectra[axis];
        let mut r = 0usize;
        while r < spec.len() && r < MAX_RANK_PER_CLASS && spec[r] >= floor {
            r += 1;
        }
        let nr_cap = numerical_rank_cap(spec, NUMERICAL_RANK_ETA, &mut sqrt_buf);
        ranks[axis] = r.min(nr_cap).min(MAX_RANK_PER_CLASS);
    }
    ranks
}

/// Blocked structural-fit plateau grid (opt-in, sequential): `folds`
/// contiguous episode blocks; every candidate rank triple over the routed-ON
/// classes is fit on each block independently; the candidate with the
/// smallest total rank within `PLATEAU_EPS` of the best mean block residual
/// wins (lexicographic ascending tie-break). See the module docs for why
/// this is not — and cannot be — held-out prediction CV.
pub fn select_ranks_blocked_cv(
    x: &[f32],
    shape: [usize; 3],
    cfg: &SliceTcaConfig,
    scratch: &mut SliceTcaScratch,
    folds: usize,
    max_rank: usize,
) -> Result<[usize; 3], SliceTcaError> {
    let [n, t, k] = shape;
    let total = n * t * k;
    if x.len() != total {
        return Err(SliceTcaError::InputSizeMismatch {
            got: x.len(),
            expected: total,
        });
    }
    if folds < 2 || k < folds {
        return Err(SliceTcaError::NotEnoughEpisodes { episodes: k, folds });
    }

    // Route first (shares are global properties of X, unchanged by blocking).
    let shares = super::svd::covariability_shares_into(x, shape, cfg.share_rank, scratch);
    let routes = super::svd::route(shares, cfg.route_theta, cfg.route_alpha);
    let active: Vec<usize> = (0..3).filter(|&a| routes[a] >= 0.5).collect();
    if active.is_empty() {
        return Ok([0, 0, 0]);
    }

    // Fold bounds: contiguous episode blocks, earlier blocks no smaller than
    // later ones (fixed policy, deterministic).
    let mut bounds = vec![0usize; folds + 1];
    for f in 0..folds {
        let width = k / folds + usize::from(f < k % folds);
        bounds[f + 1] = bounds[f] + width;
    }

    // Block tensors, materialized once (fixed scan order).
    let mut blocks = Vec::with_capacity(folds);
    for f in 0..folds {
        let (h0, h1) = (bounds[f], bounds[f + 1]);
        let width = h1 - h0;
        let mut b = Tensor3::zeros([n, t, width]);
        for i in 0..n {
            for p in 0..t {
                let src = (i * t + p) * k + h0;
                let dst = (i * t + p) * width;
                b.data[dst..dst + width].copy_from_slice(&x[src..src + width]);
            }
        }
        blocks.push(([n, t, width], b));
    }

    // Candidate grid: per active class 1..=max_rank, inactive fixed at 0.
    let per_class: [Vec<usize>; 3] = std::array::from_fn(|a| {
        if routes[a] >= 0.5 {
            (1..=max_rank.min(MAX_RANK_PER_CLASS)).collect()
        } else {
            vec![0]
        }
    });

    // CV grid budget: cap the sweeps so the grid stays tractable; the cap is
    // a fixed policy, not a convergence test.
    let mut cv_cfg = *cfg;
    cv_cfg.als_sweeps = cv_cfg.als_sweeps.min(2);

    let mut best_ranks = [0usize; 3];
    let mut best_score = f32::INFINITY;
    let mut best_total = usize::MAX;
    let mut decomp = SliceDecomposition::empty(shape);
    for &r0 in &per_class[0] {
        for &r1 in &per_class[1] {
            for &r2 in &per_class[2] {
                let ranks = [r0, r1, r2];
                let mut score_sum = 0.0f32;
                let mut scored = 0usize;
                for (b_shape, b) in &blocks {
                    // Skip candidates whose ranks exceed the block's bounds.
                    let fits = (0..3).all(|a| {
                        let d = [b_shape[0], b_shape[1], b_shape[2]][a];
                        let rest = b_shape[0] * b_shape[1] * b_shape[2] / d;
                        ranks[a] <= d.min(rest).min(MAX_RANK_PER_CLASS)
                    });
                    if !fits {
                        continue;
                    }
                    super::als::fit_with_ranks_into(&b.data, *b_shape, ranks, &cv_cfg, scratch, &mut decomp)?;
                    score_sum += block_residual(b, *b_shape, &decomp);
                    scored += 1;
                }
                if scored == 0 {
                    continue;
                }
                let score = score_sum / scored as f32;
                let total_rank = ranks[0] + ranks[1] + ranks[2];
                // Plateau rule: strictly better score wins; within the
                // plateau, the smaller total rank wins; ties lexicographic.
                let better = score < best_score - PLATEAU_EPS
                    || (score <= best_score + PLATEAU_EPS && total_rank < best_total)
                    || (score <= best_score + PLATEAU_EPS
                        && total_rank == best_total
                        && ranks < best_ranks);
                if better {
                    best_score = score.min(best_score);
                    best_total = total_rank;
                    best_ranks = ranks;
                }
            }
        }
    }
    Ok(best_ranks)
}

/// Relative residual of a fit on its own block (`‖B − B̂‖²/‖B‖²`), zero-alloc
/// via a caller-provided reconstruction buffer is avoided by computing the
/// reconstruction into the block tensor copy... — instead we rebuild the
/// reconstruction with the block shape through a local buffer (offline path).
fn block_residual(b: &Tensor3, b_shape: [usize; 3], decomp: &SliceDecomposition) -> f32 {
    // Refit the reconstruction at the block shape: the components were FIT
    // at this shape, so `reconstruct_into` is shape-consistent.
    let mut hat = Tensor3::zeros(b_shape);
    if decomp.reconstruct_into(&mut hat).is_err() {
        return f32::INFINITY;
    }
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (&a, &h) in b.data.iter().zip(hat.data.iter()) {
        let d = a - h;
        num += d * d;
        den += a * a;
    }
    if den < NORM_EPS {
        0.0
    } else {
        num / den
    }
}
