//! EM refinement of a diagonal GMM, seeded from the shipped deterministic
//! k-means (Plan 618 T2).
//!
//! The seed is [`fit_codebook_kmeans_into`] (factorized_action —
//! deterministic Lloyd + k-means++, no gradient descent), consumed
//! as-is per the verdict-round-1 law: the EM M-steps are the delta, the
//! constructor is the shipped one. Jeffreys-smoothed weights (`λ = 0.5`)
//! keep every component alive, so the M-step never divides by zero and
//! never has to re-seed mid-refinement.
//!
//! # Determinism
//!
//! - Fixed data order: the input slice order IS the reduction order.
//! - f64 accumulators, f32 artifact bytes (stored parameters).
//! - Iteration-capped; early stop on average-log-likelihood improvement
//!   `< tol`.
//! - Same input bytes + same config → identical artifact → identical
//!   BLAKE3 commitment (the regression tripwire; cross-platform float
//!   summation is pinned by order, the residual risk is the platform
//!   `exp`/`ln` in the E-step logsumexp — the Plan 618 risk row).
//!
//! # Fit-space law
//!
//! Inputs must be PROJECTED points (`&[f32]` of len `E`), never raw
//! D-space vectors — see the mixture module docs.

use super::mixture::{DiagGmm, lse};
use crate::factorized_action::fit_codebook_kmeans_into;
use crate::factorized_action::types::EffectCodebook;

/// `ln(2π)` in f64 for the EM accumulation paths.
const LN_2PI_F64: f64 = 1.837_877_066_409_345_3;

/// Errors returned by [`fit_diag_gmm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EmError {
    /// `samples` is empty.
    EmptySamples,
    /// A sample's length differs from `E`.
    DimMismatch,
}

/// EM fitter configuration. Defaults are POC-scale (the
/// `contrastive_scope` precedent): consumers re-pin for their corpus.
#[derive(Debug, Clone, Copy)]
pub struct EmConfig {
    /// Seed for the k-means++ initialization (SplitMix64 stream).
    pub seed: u64,
    /// Lloyd iterations for the k-means seed.
    pub kmeans_iters: usize,
    /// EM iteration cap.
    pub em_iters: usize,
    /// Early stop when the average log-likelihood improves by less than
    /// this between iterations.
    pub tol: f32,
    /// Variance floor (keeps `σ² > 0` under degenerate collapses).
    pub var_floor: f32,
}

impl Default for EmConfig {
    fn default() -> Self {
        Self {
            seed: 7,
            kmeans_iters: 16,
            em_iters: 50,
            tol: 1e-4,
            var_floor: 1e-6,
        }
    }
}

/// Fit a `K`-component diagonal GMM over `E`-dim samples: k-means seed →
/// hard-assignment M-step init → EM refinement.
///
/// Offline path (runs once per fit) — allocation is acceptable here (the
/// `codebook` module's allocation-discipline precedent); the EVAL path in
/// [`DiagGmm`](super::mixture::DiagGmm) is the zero-alloc half.
///
/// # Panics
///
/// Debug-mode panics if `K == 0` (a zero-component mixture is not a
/// density).
pub fn fit_diag_gmm<const E: usize, const K: usize>(
    samples: &[&[f32]],
    cfg: &EmConfig,
) -> Result<DiagGmm<E, K>, EmError> {
    debug_assert!(K > 0, "K must be > 0");
    if samples.is_empty() {
        return Err(EmError::EmptySamples);
    }
    for s in samples {
        if s.len() != E {
            return Err(EmError::DimMismatch);
        }
    }

    let n = samples.len();

    // ── Seed: the shipped deterministic k-means (consumed, never forked) ──
    let mut codebook: EffectCodebook<K, E> = EffectCodebook::zeroed();
    fit_codebook_kmeans_into::<K, E>(samples, K, cfg.seed, cfg.kmeans_iters, &mut codebook);
    let mut means = codebook.centroids;

    // ── Hard-assignment M-step init ─────────────────────────────────────
    // Assignments by nearest seed centroid (deterministic: first argmin),
    // computed ONCE and reused by counts + per-cluster variance.
    let mut assign: Vec<usize> = vec![0; n];
    for (ni, s) in samples.iter().enumerate() {
        assign[ni] = nearest_centroid::<E, K>(s, &means);
    }
    let mut counts = [0usize; K];
    for &a in assign.iter() {
        counts[a] += 1;
    }

    // Global per-dim variance (the empty-cluster fallback).
    let mut global_var = [0f64; E];
    {
        let mut mean = [0f64; E];
        for s in samples {
            for d in 0..E {
                mean[d] += s[d] as f64;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        for s in samples {
            for d in 0..E {
                let diff = s[d] as f64 - mean[d];
                global_var[d] += diff * diff;
            }
        }
        for g in global_var.iter_mut() {
            *g = (*g / n as f64).max(cfg.var_floor as f64);
        }
    }

    // Per-cluster per-dim variances around the seed means (Lloyd
    // convergence makes the seed means the cluster means).
    let mut variances = [[0f32; E]; K];
    {
        let mut acc = vec![[0f64; E]; K];
        for (ni, s) in samples.iter().enumerate() {
            let k = assign[ni];
            for d in 0..E {
                let diff = s[d] as f64 - means[k][d] as f64;
                acc[k][d] += diff * diff;
            }
        }
        for k in 0..K {
            if counts[k] == 0 {
                for d in 0..E {
                    variances[k][d] = global_var[d] as f32;
                }
            } else {
                let inv = 1.0 / counts[k] as f64;
                for d in 0..E {
                    variances[k][d] = (acc[k][d] * inv).max(cfg.var_floor as f64) as f32;
                }
            }
        }
    }

    // Jeffreys-smoothed weights: `(N_k + 0.5) / (n + K/2)` — every
    // component strictly positive, normalized, deterministic.
    let lambda = 0.5f64;
    let denom = n as f64 + lambda * K as f64;
    let mut weights = [0f64; K];
    for (k, w) in weights.iter_mut().enumerate() {
        *w = (counts[k] as f64 + lambda) / denom;
    }

    // ── EM refinement ───────────────────────────────────────────────────
    // One E-step per iteration; responsibilities cached per sample so the
    // variance M-step (which needs the NEW means) reuses the SAME
    // responsibilities — the textbook M-step, deterministic by loop order.
    let mut terms = [0f32; K];
    let mut resp_cache: Vec<[f64; K]> = vec![[0.0; K]; n];
    let mut mean_acc = vec![[0f64; E]; K];
    let mut var_acc = vec![[0f64; E]; K];
    let mut prev_avg_ll = f64::NEG_INFINITY;

    for _iter in 0..cfg.em_iters {
        // E-step + sufficient statistics for means.
        for acc_row in mean_acc.iter_mut() {
            *acc_row = [0.0; E];
        }
        let mut nk_acc = [0.0; K];
        let mut avg_ll = 0.0f64;

        for (ni, s) in samples.iter().enumerate() {
            let x: &[f32; E] = (*s).try_into().expect("len checked above");
            for (k, t) in terms.iter_mut().enumerate() {
                *t = component_loglik_f64(x, k, &means, &variances, &weights);
            }
            let ll = lse(&terms);
            avg_ll += ll as f64;
            let ll_d = ll as f64;
            for k in 0..K {
                let r = f64::exp(f64::from(terms[k]) - ll_d);
                resp_cache[ni][k] = r;
                nk_acc[k] += r;
                for d in 0..E {
                    mean_acc[k][d] += r * f64::from(x[d]);
                }
            }
        }
        avg_ll /= n as f64;

        if prev_avg_ll.is_finite() && (avg_ll - prev_avg_ll).abs() < f64::from(cfg.tol) {
            break;
        }
        prev_avg_ll = avg_ll;

        // M-step 1: means (empty-ish components freeze their previous mean).
        for k in 0..K {
            if nk_acc[k] > 1e-12 {
                let inv = 1.0 / nk_acc[k];
                for d in 0..E {
                    means[k][d] = (mean_acc[k][d] * inv) as f32;
                }
            }
        }
        // M-step 2: variances around the NEW means, SAME responsibilities.
        for acc_row in var_acc.iter_mut() {
            *acc_row = [0.0; E];
        }
        for (ni, s) in samples.iter().enumerate() {
            let x: &[f32; E] = (*s).try_into().expect("len checked above");
            for k in 0..K {
                let r = resp_cache[ni][k];
                for d in 0..E {
                    let diff = f64::from(x[d]) - f64::from(means[k][d]);
                    var_acc[k][d] += r * diff * diff;
                }
            }
        }
        for k in 0..K {
            if nk_acc[k] > 1e-12 {
                let inv = 1.0 / nk_acc[k];
                for d in 0..E {
                    variances[k][d] = (var_acc[k][d] * inv).max(f64::from(cfg.var_floor)) as f32;
                }
            }
        }
        // M-step 3: weights (normalized in f64, stored f32 below).
        let total: f64 = nk_acc.iter().sum();
        if total > 0.0 {
            let inv = 1.0 / total;
            for (k, w) in weights.iter_mut().enumerate() {
                *w = nk_acc[k] * inv;
            }
        }
    }

    // ── Artifact ────────────────────────────────────────────────────────
    let mut log_weights = [0f32; K];
    for (k, lw) in log_weights.iter_mut().enumerate() {
        *lw = weights[k].ln() as f32;
    }
    Ok(DiagGmm::new_unchecked(means, log_weights, variances))
}

/// Component-`k` log-likelihood in f64 (the EM E-step term; the artifact
/// path stores f32 — this is the accumulation-precision half).
#[inline]
fn component_loglik_f64<const E: usize, const K: usize>(
    x: &[f32; E],
    k: usize,
    means: &[[f32; E]; K],
    variances: &[[f32; E]; K],
    weights: &[f64; K],
) -> f32 {
    let mut mahal = 0.0f32;
    for d in 0..E {
        let diff = x[d] - means[k][d];
        mahal += diff * diff / variances[k][d];
    }
    let mut norm = weights[k].ln();
    for &v in variances[k].iter() {
        norm -= 0.5 * (LN_2PI_F64 + f64::from(v).ln());
    }
    (norm - 0.5 * f64::from(mahal)) as f32
}

/// Nearest centroid by squared Euclidean distance (first argmin wins —
/// the deterministic tie rule, matching the codebook module's).
fn nearest_centroid<const E: usize, const K: usize>(s: &[f32], means: &[[f32; E]; K]) -> usize {
    let mut best = 0usize;
    let mut best_d2 = f32::INFINITY;
    for (i, c) in means.iter().enumerate() {
        let mut d2 = 0.0f32;
        for d in 0..E {
            let diff = s[d] - c[d];
            d2 += diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best = i;
        }
    }
    best
}
