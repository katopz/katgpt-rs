//! Inference-time factorization + aggregation kernels (the hot path).
//!
//! This module implements the zero-allocation hot path of the OTF-LAM
//! primitive (Plan 375):
//!
//! 1. [`EffectCodebook::assign_patch_into`] — Top-1 nearest-neighbor
//!    quantization of one patch.
//! 2. [`finalize_factors`] — normalize weights and count active codes.
//! 3. [`factor_token_into`] — state-aware FiLM-modulated factor token.
//! 4. [`aggregate_action_latent_into`] — sigmoid-gated (or uniform)
//!    normalized weighted average.
//!
//! # Verified mechanism
//!
//! All four steps are distilled from `otf_lam/model.py::OTFLAM.forward()`
//! (Research 374 §10 code-verification addendum):
//!
//! - Aggregation step 6 in the paper's reference impl:
//!   ```python
//!   alpha_sum = alpha.sum(dim=1).clamp_min(self.eps)
//!   z_factor = (alpha * factor_embedding).sum(dim=1) / alpha_sum
//!   ```
//! - Factor embedding uses FiLM `(1+γ)*x+β` pervasively (modelless analog:
//!   single deterministic linear projection per code).
//! - `aggregator_type ∈ {"gate", "mean"}` — `Gate` (sigmoid) is the default
//!   full primitive; `Mean` (uniform α=1) is the G2 ablation proving the
//!   gate adds value.

use crate::sigmoid;

use super::types::{
    AggregatorType, EffectCodebook, FactorizedActionLatent, FilmProjectionBank, TransitionFactors,
};

impl<const K: usize, const D: usize> EffectCodebook<K, D> {
    /// Top-1 nearest-neighbor quantization of one patch.
    ///
    /// Writes the argmin-code index into `out.assignments[patch_idx]` and
    /// increments `out.weights[k*]` by 1. Zero allocation.
    ///
    /// # Panics
    ///
    /// Debug-mode panic if `patch.len() != D` or `patch_idx >= MAX_PATCHES`.
    #[inline]
    pub fn assign_patch_into(&self, patch: &[f32], out: &mut TransitionFactors, patch_idx: usize) {
        debug_assert_eq!(
            patch.len(),
            D,
            "patch dimension {} != codebook D={D}",
            patch.len()
        );
        debug_assert!(
            patch_idx < super::types::MAX_PATCHES,
            "patch_idx {patch_idx} >= MAX_PATCHES"
        );

        let mut best_k = 0usize;
        let mut best_d2 = f32::INFINITY;
        for k in 0..K {
            let c = self.centroid(k);
            let mut d2 = 0.0f32;
            // Inner loop: chunked for SIMD auto-vectorization potential.
            // Unrolled by 4 for the common D=8, D=32 cases.
            let mut d = 0usize;
            while d + 4 <= D {
                let a0 = patch[d] - c[d];
                let a1 = patch[d + 1] - c[d + 1];
                let a2 = patch[d + 2] - c[d + 2];
                let a3 = patch[d + 3] - c[d + 3];
                d2 += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
                d += 4;
            }
            while d < D {
                let a = patch[d] - c[d];
                d2 += a * a;
                d += 1;
            }
            if d2 < best_d2 {
                best_d2 = d2;
                best_k = k;
            }
        }

        out.assignments[patch_idx] = best_k as u16;
        out.weights[best_k] += 1.0;
    }
}

/// Finalize the per-transition factorization output:
///
/// - Normalize `weights[k] /= n_patches` → activation strength `w(k)`.
/// - Set `n_active` = count of codes with non-zero occupancy.
/// - Set `n_patches` field.
///
/// Idempotent w.r.t. repeated calls only if the caller resets `weights`
/// first; this divides by `n_patches` from the raw counts.
///
/// # Panics
///
/// Debug-mode panic if `n_patches == 0` or `n_patches > MAX_PATCHES`.
#[inline]
pub fn finalize_factors(factors: &mut TransitionFactors, n_patches: usize) {
    debug_assert!(n_patches > 0, "n_patches must be > 0");
    debug_assert!(
        n_patches <= super::types::MAX_PATCHES,
        "n_patches {n_patches} > MAX_PATCHES"
    );

    let inv = 1.0f32 / (n_patches as f32);
    let mut n_active = 0usize;
    // Normalize weights: weights is sized to MAX_PATCHES, but only the first
    // K entries (K ≤ MAX_PATCHES in practice; we cap to min) can be non-zero.
    // We iterate over the full bank for safety but only count non-zero.
    for w in factors.weights.iter_mut() {
        if *w > 0.0 {
            *w *= inv;
            n_active += 1;
        }
    }
    factors.n_active = n_active;
    factors.n_patches = n_patches;
}

/// Compute the state-aware factor token `r_k` for code `k`.
///
/// Implements modelless FiLM modulation (Plan 375 T1.4):
/// ```text
/// γ_k = dot(state, g_proj_k)
/// β_k = dot(state, b_proj_k)
/// r_k = (1 + γ_k) * c(k) + β_k
/// ```
///
/// Writes the D-dim factor token into `out[..D]`. Zero allocation.
///
/// If `film` is `None`, falls back to `r_k = c(k)` (identity FiLM: γ=β=0).
///
/// # Panics
///
/// Debug-mode panic if `state.len() != S` (when `film` is `Some`), or
/// `out.len() < D`, or `k >= K`.
#[inline]
pub fn factor_token_into<const K: usize, const D: usize, const S: usize>(
    codebook: &EffectCodebook<K, D>,
    film: Option<&FilmProjectionBank<K, D, S>>,
    k: usize,
    state: &[f32],
    out: &mut [f32],
) {
    debug_assert!(k < K, "code index {k} >= K={K}");
    debug_assert!(out.len() >= D, "out.len() {} < D={D}", out.len());

    let c = codebook.centroid(k);

    let (gamma, beta) = match film {
        Some(b) => {
            debug_assert_eq!(state.len(), S, "state.len() {} != S={S}", state.len());
            let g = b.g_proj_slice(k);
            let bb = b.b_proj_slice(k);
            // dot(state, g_proj_k) and dot(state, b_proj_k)
            let mut dg = 0.0f32;
            let mut db = 0.0f32;
            let mut i = 0usize;
            while i + 4 <= S {
                dg += state[i] * g[i]
                    + state[i + 1] * g[i + 1]
                    + state[i + 2] * g[i + 2]
                    + state[i + 3] * g[i + 3];
                db += state[i] * bb[i]
                    + state[i + 1] * bb[i + 1]
                    + state[i + 2] * bb[i + 2]
                    + state[i + 3] * bb[i + 3];
                i += 4;
            }
            while i < S {
                dg += state[i] * g[i];
                db += state[i] * bb[i];
                i += 1;
            }
            (dg, db)
        }
        None => (0.0, 0.0),
    };

    let scale = 1.0 + gamma;
    for d in 0..D {
        out[d] = scale * c[d] + beta;
    }
}

/// Sigmoid relevance score for a factor token.
///
/// Relevance = L2-norm of `r_k` (a state-aware factor token is "more
/// relevant" when its magnitude is larger — the modelless analog of the
/// paper's learned GateNetwork readout, which is itself just a linear
/// layer ending in `sigmoid(out_linear(r_k))`).
///
/// The relevance score here is `||r_k||` (cheap L2). Combined with the
/// sigmoid gate `sigmoid(β·(relevance − τ))`, this gives a value in
/// `(0, 1)` per code.
#[inline]
pub fn relevance_score(token: &[f32]) -> f32 {
    let mut s = 0.0f32;
    let n = token.len();
    let mut i = 0usize;
    while i + 4 <= n {
        s += token[i] * token[i]
            + token[i + 1] * token[i + 1]
            + token[i + 2] * token[i + 2]
            + token[i + 3] * token[i + 3];
        i += 4;
    }
    while i < n {
        s += token[i] * token[i];
        i += 1;
    }
    s.sqrt()
}

/// Aggregate per-code factor tokens into the action latent via the
/// normalized weighted average.
///
/// For each active code `k`:
/// 1. Compute factor token `r_k` via [`factor_token_into`].
/// 2. Compute weight `α_k`:
///    - `Gate` mode: `α_k = sigmoid(β · (relevance(r_k) − τ))`.
///    - `Mean` mode: `α_k = 1` (uniform).
/// 3. Accumulate `numerator += α_k · r_k`, `denominator += α_k`.
///
/// Final: `z = numerator / (denominator + ε)`, written into `out.0[..D]`.
///
/// Verified against `otf_lam/model.py::OTFLAM.forward()` step 6.
///
/// # Inputs
///
/// - `codebook` — frozen `EffectCodebook<K, D>`.
/// - `film` — optional `FilmProjectionBank<K, D, S>`. Pass `None` to
///   skip state conditioning (r_k = c(k)).
/// - `factors` — finalized `TransitionFactors` (call [`finalize_factors`]
///   first). Only codes with non-zero `weights[k]` are visited.
/// - `state` — state vector for FiLM. Must be `&[]` if `film` is `None`,
///   or length `S` if `film` is `Some`.
/// - `gate_beta` — gate inverse-temperature (paper GateNetwork uses
///   learned β; we use a frozen scalar).
/// - `gate_tau` — gate threshold (same).
/// - `aggregator` — `Gate` (default) or `Mean` (G2 ablation).
/// - `out` — output buffer.
/// - `scratch_token` — D-length scratch buffer for the per-code factor
///   token. Caller pre-allocates once and reuses across calls (zero
///   allocation inside this function).
///
/// # Panics
///
/// Debug-mode panic on dimension mismatch or insufficient scratch size.
#[allow(clippy::too_many_arguments)]
pub fn aggregate_action_latent_into<const K: usize, const D: usize, const S: usize>(
    codebook: &EffectCodebook<K, D>,
    film: Option<&FilmProjectionBank<K, D, S>>,
    factors: &TransitionFactors,
    state: &[f32],
    gate_beta: f32,
    gate_tau: f32,
    aggregator: AggregatorType,
    out: &mut FactorizedActionLatent<D>,
    scratch_token: &mut [f32],
) {
    debug_assert!(
        scratch_token.len() >= D,
        "scratch_token.len() {} < D={D}",
        scratch_token.len()
    );
    debug_assert_eq!(
        out.0.len(),
        D,
        "out latent dimension {} != D={D}",
        out.0.len()
    );

    // Zero the output buffer; we accumulate the numerator in place.
    out.0.fill(0.0);

    let mut denominator = 0.0f32;
    let eps = 1e-8f32;

    // Visit only codes with non-zero weight. Since weights is fixed-size
    // MAX_PATCHES (≥ K in well-formed inputs) we iterate up to min(K, len).
    let k_max = K.min(factors.weights.len());
    for (k, &weight_k) in factors.weights[..k_max].iter().enumerate() {
        if weight_k <= 0.0 {
            continue;
        }

        // 1. Factor token r_k into scratch.
        factor_token_into(codebook, film, k, state, &mut scratch_token[..D]);

        // 2. Weight α_k.
        let alpha = match aggregator {
            AggregatorType::Gate => {
                let rel = relevance_score(&scratch_token[..D]);
                sigmoid(gate_beta * (rel - gate_tau))
            }
            AggregatorType::Mean => 1.0,
        };

        // 3. Accumulate numerator += α_k · r_k into out.0, denominator += α_k.
        let aw = alpha * factors.weights[k];
        for (dst, &src) in out.0.iter_mut().zip(scratch_token.iter()).take(D) {
            *dst += aw * src;
        }
        denominator += aw;
    }

    // Normalize: z = numerator / (denominator + ε).
    let inv = 1.0 / (denominator + eps);
    for x in out.0.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::MAX_PATCHES;
    use super::*;

    /// T1.8 smoke test — assign + aggregate on K=4, D=8, 16 patches.
    #[test]
    fn smoke_assign_and_aggregate() {
        // Hand-crafted codebook: 4 codes, each with a distinct "direction".
        let mut cb: EffectCodebook<4, 8> = EffectCodebook::zeroed();
        // Code 0: all-ones
        cb.centroids[0] = [1.0; 8];
        // Code 1: all-minus-ones
        cb.centroids[1] = [-1.0; 8];
        // Code 2: e_0
        cb.centroids[2] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // Code 3: e_7
        cb.centroids[3] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        // 16 patches: 8 close to code 0, 8 close to code 1.
        let mut factors = TransitionFactors::zeroed();
        let p0 = [0.9f32; 8];
        let p1 = [-0.9f32; 8];
        for i in 0..16 {
            let p = if i < 8 { &p0[..] } else { &p1[..] };
            cb.assign_patch_into(p, &mut factors, i);
        }
        finalize_factors(&mut factors, 16);

        assert_eq!(factors.assignments[0..8], [0u16; 8]);
        assert_eq!(factors.assignments[8..16], [1u16; 8]);
        assert_eq!(factors.weights[0], 0.5);
        assert_eq!(factors.weights[1], 0.5);
        assert_eq!(factors.n_active, 2);
        assert_eq!(factors.n_patches, 16);

        // Aggregate (Gate mode, no FiLM, β=1, τ=0).
        let mut out = FactorizedActionLatent::<8>::zeroed();
        let mut scratch = [0.0f32; 8];
        aggregate_action_latent_into::<4, 8, 0>(
            &cb,
            None::<&FilmProjectionBank<4, 8, 0>>,
            &factors,
            &[],
            1.0,
            0.0,
            AggregatorType::Gate,
            &mut out,
            &mut scratch,
        );

        // Output must be finite.
        for x in out.0.iter() {
            assert!(x.is_finite(), "output not finite: {x}");
        }
        // In range [-2, 2] (centroid magnitude bounded by sqrt(8) ≈ 2.83).
        for x in out.0.iter() {
            assert!(*x >= -3.0 && *x <= 3.0, "output out of range: {x}");
        }
        // Deterministic: same inputs → same outputs.
        let mut out2 = FactorizedActionLatent::<8>::zeroed();
        aggregate_action_latent_into::<4, 8, 0>(
            &cb,
            None::<&FilmProjectionBank<4, 8, 0>>,
            &factors,
            &[],
            1.0,
            0.0,
            AggregatorType::Gate,
            &mut out2,
            &mut scratch,
        );
        assert_eq!(out.0, out2.0);
    }

    /// With uniform `Mean` aggregation, weights are equal to the per-code
    /// occupancy; output should be the centroid-of-centroids weighted by
    /// occupancy.
    #[test]
    fn mean_aggregator_is_uniform_weighted_average() {
        let mut cb: EffectCodebook<2, 4> = EffectCodebook::zeroed();
        cb.centroids[0] = [1.0, 1.0, 1.0, 1.0];
        cb.centroids[1] = [-1.0, -1.0, -1.0, -1.0];

        let mut factors = TransitionFactors::zeroed();
        // 4 patches → code 0, 4 patches → code 1.
        for i in 0..8 {
            let p: &[f32] = if i < 4 {
                &[0.9, 0.9, 0.9, 0.9]
            } else {
                &[-0.9, -0.9, -0.9, -0.9]
            };
            cb.assign_patch_into(p, &mut factors, i);
        }
        finalize_factors(&mut factors, 8);

        let mut out = FactorizedActionLatent::<4>::zeroed();
        let mut scratch = [0.0f32; 4];
        aggregate_action_latent_into::<2, 4, 0>(
            &cb,
            None::<&FilmProjectionBank<2, 4, 0>>,
            &factors,
            &[],
            1.0,
            0.0,
            AggregatorType::Mean,
            &mut out,
            &mut scratch,
        );

        // Symmetric codes → output ≈ 0.
        for x in out.0.iter() {
            assert!(
                x.abs() < 1e-5,
                "uniform mean of ±1 centroids should be 0, got {x}"
            );
        }
    }

    /// `assign_patch_into` correctly picks the nearest centroid.
    #[test]
    fn assign_picks_nearest_centroid() {
        let mut cb: EffectCodebook<3, 2> = EffectCodebook::zeroed();
        cb.centroids[0] = [0.0, 0.0];
        cb.centroids[1] = [10.0, 0.0];
        cb.centroids[2] = [0.0, 10.0];

        let mut f = TransitionFactors::zeroed();
        cb.assign_patch_into(&[9.5, 0.5], &mut f, 0);
        cb.assign_patch_into(&[0.1, 9.9], &mut f, 1);
        cb.assign_patch_into(&[0.0, 0.0], &mut f, 2);

        assert_eq!(f.assignments[0], 1);
        assert_eq!(f.assignments[1], 2);
        assert_eq!(f.assignments[2], 0);
    }

    /// `relevance_score` is the L2 norm.
    #[test]
    fn relevance_score_is_l2_norm() {
        let v = [3.0f32, 4.0];
        assert!((relevance_score(&v) - 5.0).abs() < 1e-5);
    }

    /// `MAX_PATCHES ≥ K` for the typical codebook sizes we use.
    #[test]
    fn max_patches_at_least_typical_k() {
        const _: () = assert!(MAX_PATCHES >= 32, "MAX_PATCHES too small for typical K");
    }
}
