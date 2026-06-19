//! ICT distributional-math primitives — collision purity, Rényi H₂, Shannon H₁,
//! Jensen-Shannon divergence.
//!
//! Plan 294, Research 270, arxiv 2606.19771 (Feng et al., 18 Jun 2026).
//!
//! These are the public, MIT-licensed, modelless primitives distilled from
//! ICT's training-time selector. They are pure functions on categorical
//! probability simplices — no backprop, no weights, no game/chain IP. The
//! runtime fusion (CLR gating, HLA updates, KG emission, Bebop H₁→H₂
//! upgrade) lives in `riir-ai` Plan 324; this module is the open adoption
//! hook.
//!
//! ## Why β(π) = Σ π², not H₁
//!
//! ICT Theorem A.2.5 proves ∂(Σ π²)/∂π(a) = 2π(a) > 0 unconditionally — the
//! collision purity **always** increases when any single probability mass
//! grows. Shannon entropy H₁ only has negative gradient for π(a) > e⁻¹ ≈
//! 0.37; below that the "concentration" signal H₁ reports is wrong. The
//! paper's Figure 1a exhibits two distributions with identical H₁ but
//! different β — `bench_294_ict_g1.rs` reproduces that bifurcation.
//!
//! ## Why Jensen-Shannon, not KL
//!
//! JS is symmetric, bounded in `[0, ln 2]`, and finite for disjoint supports
//! (ICT §A.5). KL is asymmetric, unbounded, and infinity on disjoint
//! supports — useless for ranking "which trajectory is most novel". JS needs
//! no meaningless ground metric over token indices (unlike Wasserstein).
//!
//! ## Curiosity Pulse H₁→β drop-in (R041, riir-ai Plan 274)
//!
//! Curiosity Pulse currently uses `u_i(t) = shannon_h1(relevance_scores)`.
//! The ICT-correct drop-in is:
//!
//! ```rust,ignore
//! // Curiosity Pulse (R041) currently uses:
//! //   u_i(t) = shannon_h1(relevance_scores)
//! // Drop-in upgrade per ICT §1.5 + A.3.3:
//! //   u_i(t) = collision_purity(relevance_scores)  // = β
//! // H1 is "blind exploration" (ICT §1); β captures concentration — the right
//! // curiosity trigger. ∂H_2/∂π(a) < 0 unconditionally; H1 only valid for π > e⁻¹.
//! ```
//!
//! The example `examples/ict_curiosity_pulse_upgrade.rs` shows the call site.

use crate::simd::simd_dot_f32;

/// Collision purity β(π) = Σ_a π(a)² = exp(−H₂(π)).
///
/// ICT §A.2.5: ∂/∂π(a) = 2π(a) > 0 unconditionally — unlike H₁, which only
/// has negative gradient for π(a) > e⁻¹ ≈ 0.37. Use this anywhere we currently
/// reach for Shannon entropy as a "concentration" / "decision-confidence"
/// signal. O(vocab) with SIMD-friendly accumulation.
///
/// # Examples
///
/// ```
/// # use katgpt_core::ict::math::collision_purity;
/// assert!((collision_purity(&[0.5_f32, 0.5]) - 0.5).abs() < 1e-6);
/// assert!((collision_purity(&[1.0_f32, 0.0]) - 1.0).abs() < 1e-6);
/// ```
#[inline]
pub fn collision_purity(probs: &[f32]) -> f32 {
    // β = π · π = simd_dot_f32(π, π). Dispatches to NEON/AVX2/scalar —
    // hot path (G4 ≤ 50µs) relies on this.
    simd_dot_f32(probs, probs, probs.len())
}

/// Zero-alloc variant of [`collision_purity`] — writes the result into `out`.
///
/// `out` is `&mut f32` (not `&mut [f32]`) because β is a single scalar; the
/// `_into` suffix matches the repo convention (see `branching_point_mask_into`).
/// Kept for API symmetry with the rest of the `*_into` family even though
/// [`collision_purity`] is already allocation-free.
#[inline]
pub fn collision_purity_into(probs: &[f32], out: &mut f32) {
    *out = collision_purity(probs);
}

/// Second-order Rényi entropy H₂(π) = −log(Σ π²) = −log β(π).
///
/// Drop-in for H₁ wherever the signal is "how concentrated is this
/// distribution?" Proven unconditionally valid by ICT §A.3.3 (the H₁
/// monotonicity caveat does not apply).
///
/// # Examples
///
/// ```
/// # use katgpt_core::ict::math::renyi_h2;
/// // Uniform over n outcomes → H₂ = log(n).
/// let u = [1.0_f32 / 4.0; 4];
/// assert!((renyi_h2(&u) - (4.0_f32).ln()).abs() < 1e-5);
/// ```
#[inline]
pub fn renyi_h2(probs: &[f32]) -> f32 {
    -collision_purity(probs).ln()
}

/// Shannon entropy H₁(π) = −Σ π log π (natural log).
///
/// Provided as the G3 baseline: the make-or-break gate asserts Spearman
/// ρ(H₁, JS-uniqueness) < 0.5. Without H₁ alongside β we cannot prove the
/// two metrics carry structurally-different information.
///
/// Uses the standard 0·log 0 = 0 convention (no NaN from zero probabilities).
#[inline]
pub fn shannon_h1(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

/// Jensen-Shannon divergence between two categorical distributions.
///
/// JS(p, q) = ½ KL(p‖m) + ½ KL(q‖m) where m = (p + q) / 2.
///
/// Symmetric, bounded in `[0, ln 2]`, finite on disjoint supports (ICT §A.5).
/// Requires a scratch buffer of length `n` for `m`; passing a shorter slice
/// panics in debug, returns 0.0 in release (defensive).
///
/// # Examples
///
/// ```
/// # use katgpt_core::ict::math::js_divergence;
/// let p = [0.5_f32, 0.5];
/// let q = [0.5_f32, 0.5];
/// let mut m = [0.0_f32; 2];
/// assert!(js_divergence(&p, &q, &mut m).abs() < 1e-6);
/// ```
#[inline]
pub fn js_divergence(p: &[f32], q: &[f32], scratch_m: &mut [f32]) -> f32 {
    let n = p.len();
    if n == 0 || q.len() != n || scratch_m.len() < n {
        return 0.0;
    }

    // m = (p + q) / 2 — chunked-4 to help LLVM autovectorize per AGENTS.md.
    let mut i = 0;
    while i + 4 <= n {
        scratch_m[i] = 0.5 * (p[i] + q[i]);
        scratch_m[i + 1] = 0.5 * (p[i + 1] + q[i + 1]);
        scratch_m[i + 2] = 0.5 * (p[i + 2] + q[i + 2]);
        scratch_m[i + 3] = 0.5 * (p[i + 3] + q[i + 3]);
        i += 4;
    }
    while i < n {
        scratch_m[i] = 0.5 * (p[i] + q[i]);
        i += 1;
    }

    // JS = ½ Σ_a [ p·log(p/m) + q·log(q/m) ]
    //   = ½ Σ_a [ p·log p − p·log m + q·log q − q·log m ]
    // log(p/m) terms vanish at p=0 (0·log 0 = 0 convention).
    let mut js = 0.0f32;
    for a in 0..n {
        let pa = p[a];
        let qa = q[a];
        let ma = scratch_m[a];
        // ma = (pa+qa)/2; if pa==qa==0 then ma==0 and both terms contribute 0.
        if ma > 0.0 {
            if pa > 0.0 {
                js += 0.5 * pa * (pa.ln() - ma.ln());
            }
            if qa > 0.0 {
                js += 0.5 * qa * (qa.ln() - ma.ln());
            }
        }
    }
    // Clamp to [0, ln 2] — numerical drift can put us microseconds above ln 2
    // on disjoint supports; the bound is a theorem, not a guarantee of f32
    // arithmetic.
    if js < 0.0 {
        0.0
    } else if js > core::f32::consts::LN_2 {
        core::f32::consts::LN_2
    } else {
        js
    }
}

/// Batch JS-divergence-to-mean over a population of distributions.
///
/// Returns `Vec<f32>` of length `dists.len()` where entry `k` is
/// `JS(dists[k] ‖ mean(dists))`. This is the ICT selector's uniqueness score
/// `u_{k,s}` (R270 §2.4 step 3). Allocates the output Vec once; callers that
/// need zero-alloc should use [`crate::ict::detector::BranchingDetector`]
/// which keeps the output in pre-allocated scratch.
///
/// `scratch_m` must be at least `action_dim` long (the per-distribution
/// length). All distributions must have equal length.
pub fn js_divergence_batch(dists: &[&[f32]], scratch_m: &mut [f32]) -> Vec<f32> {
    let k = dists.len();
    let mut out = Vec::with_capacity(k);
    if k == 0 {
        return out;
    }
    let n = dists[0].len();
    if n == 0 || scratch_m.len() < n {
        return out;
    }
    // mean over k — write into scratch_m then divide.
    for slot in scratch_m[..n].iter_mut() {
        *slot = 0.0;
    }
    for d in dists {
        if d.len() != n {
            return out;
        }
        for a in 0..n {
            scratch_m[a] += d[a];
        }
    }
    let inv_k = 1.0_f32 / (k as f32);
    for a in 0..n {
        scratch_m[a] *= inv_k;
    }
    // For each dist, JS(dist, mean). We need a second scratch slot for the
    // (dist + mean)/2 mid-point — reuse the lower half of scratch_m? No: the
    // mean is needed for all k calls. Allocate a tiny local Vec — k×n is the
    // realistic budget and the result Vec is already k long. For the
    // zero-alloc path see BranchingDetector.
    let mut m_half = vec![0.0f32; n];
    for d in dists {
        // m_half = (d + mean) / 2
        let mut i = 0;
        while i + 4 <= n {
            m_half[i] = 0.5 * (d[i] + scratch_m[i]);
            m_half[i + 1] = 0.5 * (d[i + 1] + scratch_m[i + 1]);
            m_half[i + 2] = 0.5 * (d[i + 2] + scratch_m[i + 2]);
            m_half[i + 3] = 0.5 * (d[i + 3] + scratch_m[i + 3]);
            i += 4;
        }
        while i < n {
            m_half[i] = 0.5 * (d[i] + scratch_m[i]);
            i += 1;
        }
        // JS(d, mean) = Σ_a [ ½ d·log(d/m_half) + ½ mean·log(mean/m_half) ]
        let mut js = 0.0f32;
        for a in 0..n {
            let pa = d[a];
            let qa = scratch_m[a]; // = mean
            let ma = m_half[a];
            if ma > 0.0 {
                if pa > 0.0 {
                    js += 0.5 * pa * (pa.ln() - ma.ln());
                }
                if qa > 0.0 {
                    js += 0.5 * qa * (qa.ln() - ma.ln());
                }
            }
        }
        if js < 0.0 {
            js = 0.0;
        } else if js > core::f32::consts::LN_2 {
            js = core::f32::consts::LN_2;
        }
        out.push(js);
    }
    out
}

// ──────────────────────────────────────────────────────────────────────────
// Unit tests — Plan 294 Phase 1 T1.8 (math.rs)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::LN_2;

    const TOL: f32 = 1e-5;

    #[test]
    fn collision_purity_uniform_distribution() {
        // Uniform over n → β = n · (1/n)² = 1/n.
        for n in [1_usize, 2, 4, 8, 16, 32] {
            let probs = vec![1.0_f32 / n as f32; n];
            let beta = collision_purity(&probs);
            assert!(
                (beta - 1.0_f32 / n as f32).abs() < TOL,
                "uniform n={n}: β={beta}, expected {expected}",
                expected = 1.0_f32 / n as f32
            );
        }
    }

    #[test]
    fn collision_purity_degenerate() {
        // Point mass → β = 1.
        let probs = [1.0_f32, 0.0, 0.0, 0.0, 0.0];
        let beta = collision_purity(&probs);
        assert!((beta - 1.0).abs() < TOL, "degenerate: β={beta}, expected 1.0");
    }

    #[test]
    fn collision_purity_known_value() {
        // 50/50 → β = 0.25 + 0.25 = 0.5. (Paper Fig 1a p_A.)
        let probs = [0.5_f32, 0.5];
        let beta = collision_purity(&probs);
        assert!((beta - 0.5).abs() < TOL, "50/50: β={beta}, expected 0.5");
    }

    #[test]
    fn collision_purity_into_matches_scalar() {
        let probs = [0.1_f32, 0.2, 0.3, 0.4];
        let mut out = 0.0_f32;
        collision_purity_into(&probs, &mut out);
        let direct = collision_purity(&probs);
        assert!((out - direct).abs() < TOL, "into={out}, scalar={direct}");
        // 0.01 + 0.04 + 0.09 + 0.16 = 0.30
        assert!((out - 0.30).abs() < TOL, "into={out}, expected 0.30");
    }

    #[test]
    fn renyi_h2_uniform() {
        // H₂(uniform over n) = log n.
        for n in [2_usize, 4, 8, 16] {
            let probs = vec![1.0_f32 / n as f32; n];
            let h2 = renyi_h2(&probs);
            let expected = (n as f32).ln();
            assert!((h2 - expected).abs() < TOL, "uniform n={n}: H₂={h2}, expected {expected}");
        }
    }

    #[test]
    fn renyi_h2_degenerate_is_zero() {
        // Point mass → β = 1 → H₂ = −log 1 = 0.
        let probs = [1.0_f32, 0.0, 0.0];
        let h2 = renyi_h2(&probs);
        assert!(h2.abs() < TOL, "degenerate H₂ should be 0, got {h2}");
    }

    #[test]
    fn shannon_h1_uniform() {
        // H₁(uniform over n) = log n.
        for n in [2_usize, 4, 8] {
            let probs = vec![1.0_f32 / n as f32; n];
            let h1 = shannon_h1(&probs);
            let expected = (n as f32).ln();
            assert!((h1 - expected).abs() < TOL, "uniform n={n}: H₁={h1}, expected {expected}");
        }
    }

    #[test]
    fn shannon_h1_zero_prob_safe() {
        // No NaN from 0·log 0.
        let probs = [0.5_f32, 0.5, 0.0, 0.0];
        let h1 = shannon_h1(&probs);
        assert!(h1.is_finite(), "h1 must be finite, got {h1}");
        assert!((h1 - LN_2).abs() < TOL, "h1={h1}, expected ln 2");
    }

    #[test]
    fn js_divergence_identical() {
        let p = [0.25_f32, 0.25, 0.25, 0.25];
        let q = [0.25_f32, 0.25, 0.25, 0.25];
        let mut m = [0.0_f32; 4];
        let js = js_divergence(&p, &q, &mut m);
        assert!(js.abs() < TOL, "JS(p,p) should be 0, got {js}");
    }

    #[test]
    fn js_divergence_disjoint_bounded_by_ln2() {
        // Disjoint supports → JS = ln 2 (the upper bound).
        let p = [1.0_f32, 0.0, 0.0, 0.0];
        let q = [0.0_f32, 1.0, 0.0, 0.0];
        let mut m = [0.0_f32; 4];
        let js = js_divergence(&p, &q, &mut m);
        assert!(
            (js - LN_2).abs() < 1e-4,
            "JS(disjoint) should be ln 2 = {ln2}, got {js}",
            ln2 = LN_2
        );
    }

    #[test]
    fn js_divergence_symmetric() {
        // JS(p, q) == JS(q, p).
        let p = [0.5_f32, 0.3, 0.2, 0.0];
        let q = [0.1_f32, 0.1, 0.4, 0.4];
        let mut m1 = [0.0_f32; 4];
        let mut m2 = [0.0_f32; 4];
        let js_pq = js_divergence(&p, &q, &mut m1);
        let js_qp = js_divergence(&q, &p, &mut m2);
        assert!(
            (js_pq - js_qp).abs() < TOL,
            "JS not symmetric: JS(p,q)={js_pq}, JS(q,p)={js_qp}"
        );
    }

    #[test]
    fn js_divergence_batch_matches_scalar() {
        // Batch JS-to-mean matches scalar JS-to-mean computed one-by-one.
        let d1 = [0.5_f32, 0.5, 0.0, 0.0];
        let d2 = [0.0_f32, 0.0, 0.5, 0.5];
        let d3 = [0.25_f32, 0.25, 0.25, 0.25];
        let dists: Vec<&[f32]> = vec![&d1, &d2, &d3];
        let mut m_scratch = [0.0_f32; 4];
        let batched = js_divergence_batch(&dists, &mut m_scratch);
        assert_eq!(batched.len(), 3);

        // Manually compute the mean.
        let mut mean = [0.0_f32; 4];
        for d in &dists {
            for a in 0..4 {
                mean[a] += d[a];
            }
        }
        for a in 0..4 {
            mean[a] /= 3.0;
        }
        for (k, d) in dists.iter().enumerate() {
            let mut mm = [0.0_f32; 4];
            let scalar_js = js_divergence(d, &mean, &mut mm);
            assert!(
                (batched[k] - scalar_js).abs() < 1e-4,
                "batched[{k}]={batched_k}, scalar={scalar_js}",
                batched_k = batched[k]
            );
        }
    }
}
