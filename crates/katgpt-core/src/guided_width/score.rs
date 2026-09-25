//! T3 — decode-free latent scorer for NO-REWARD hosts.
//!
//! Where a rollout reward exists (arenas), `BanditPruner` stands (Research
//! 058 §5.3) — this scorer is for the belief host, where fog-of-war
//! deliberation has no ground truth to reward. The only signals are latent:
//!
//! ```text
//! v_i = w_c · mean_{j≠i} σ(β (cos(h_i − c, h_j − c) − τ))     self-consistency
//!     + w_r · 2σ(−r_i / r_scale)                              convergence residual
//!     + w_d · σ(β_d · cos(h_i − c, d))                         optional frozen direction
//! ```
//!
//! `c` is the decision's start state (hypotheses are compared by where they
//! MOVED, not where they started). Sigmoid only, never softmax. The value is a
//! RANKING signal — it is never published as a calibrated probability (a
//! confidence consumer must pass the Report-the-Floor conformal gate).
//!
//! NaN-safe: a non-finite cosine / residual contributes `0` to its term, and
//! [`select_best`] can never return a NaN-valued branch while a finite one
//! exists (`float_order` discipline; ties keep the LOWEST index, so the
//! deterministic branch 0 wins a tie).

use crate::simd::fast_sigmoid;

use super::types::{LatentValueConfig, MAX_BRANCHES};

#[inline]
fn centered_norm(h: &[f32], c: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in h.iter().zip(c) {
        let t = x - y;
        s += t * t;
    }
    s.sqrt()
}

#[inline]
fn centered_dot(a: &[f32], b: &[f32], c: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..c.len() {
        s += (a[i] - c[i]) * (b[i] - c[i]);
    }
    s
}

#[inline]
fn sig_or_zero(x: f32) -> f32 {
    if x.is_finite() { fast_sigmoid(x) } else { 0.0 }
}

/// Score `n` branch states (`states` is `n × d`, branch-major) into
/// `out[..n]`. `center` (len d) is the decision start state, `residuals[..n]`
/// the final deterministic-update norms, `frozen` an optional direction
/// (len d) for the third term.
///
/// Zero-allocation (`n ≤ MAX_BRANCHES` norms live on the stack; larger `n`
/// is clamped). `O(n² d)`.
pub fn latent_value_into(
    states: &[f32],
    n: usize,
    center: &[f32],
    residuals: &[f32],
    cfg: &LatentValueConfig,
    frozen: Option<&[f32]>,
    out: &mut [f32],
) {
    let d = center.len();
    let n = n.min(MAX_BRANCHES);
    let mut norms = [0.0f32; MAX_BRANCHES];
    for b in 0..n {
        norms[b] = centered_norm(&states[b * d..(b + 1) * d], center);
    }
    let frozen_norm = frozen.map(|f| {
        let mut s = 0.0f32;
        for &x in &f[..d] {
            s += x * x;
        }
        s.sqrt()
    });
    for i in 0..n {
        let hi = &states[i * d..(i + 1) * d];
        // Self-consistency.
        let mut sc = 0.0f32;
        if n > 1 && cfg.w_consistency != 0.0 {
            for j in 0..n {
                if j == i {
                    continue;
                }
                let denom = norms[i] * norms[j];
                let cos = if denom > 1e-24 {
                    centered_dot(hi, &states[j * d..(j + 1) * d], center) / denom
                } else if norms[i] <= 1e-12 && norms[j] <= 1e-12 {
                    // Two hypotheses that both stayed at the start agree.
                    1.0
                } else {
                    0.0
                };
                sc += sig_or_zero(cfg.beta * (cos - cfg.tau));
            }
            sc /= (n - 1) as f32;
        }
        // Convergence residual: 2σ(−r/r_scale) ∈ (0, 1], r = 0 ⇒ 1.
        let r = residuals[i];
        let conv = if cfg.w_residual != 0.0 && cfg.residual_scale > 0.0 {
            2.0 * sig_or_zero(-r / cfg.residual_scale)
        } else {
            0.0
        };
        // Optional frozen direction.
        let dir = match (frozen, frozen_norm) {
            (Some(f), Some(fnorm)) if cfg.w_direction != 0.0 => {
                let denom = norms[i] * fnorm;
                if denom > 1e-24 {
                    let mut dot = 0.0f32;
                    for k in 0..d {
                        dot += (hi[k] - center[k]) * f[k];
                    }
                    sig_or_zero(cfg.beta_direction * dot / denom)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        out[i] = cfg.w_consistency * sc + cfg.w_residual * conv + cfg.w_direction * dir;
    }
}

/// Index of the highest value in `values`; NaN never wins against a finite
/// value, ties keep the LOWEST index. All-NaN / empty ⇒ `0` (the
/// deterministic branch).
pub fn select_best(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut found = false;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if !found || v > best_v {
            best = i;
            best_v = v;
            found = true;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consensus_branch_outscores_the_outlier() {
        // 3 agreeing hypotheses + 1 outlier, equal residuals.
        let c = [0.0f32; 4];
        let states = [
            1.0f32, 0.0, 0.0, 0.0, //
            0.98, 0.05, 0.0, 0.0, //
            0.99, -0.03, 0.01, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ];
        let r = [0.0f32; 4];
        let mut v = [0.0f32; 4];
        latent_value_into(&states, 4, &c, &r, &LatentValueConfig::DEFAULT, None, &mut v);
        assert!(v[0] > v[3] && v[1] > v[3] && v[2] > v[3], "{v:?}");
    }

    #[test]
    fn converged_branch_outscores_a_moving_one() {
        let c = [0.0f32; 2];
        let states = [1.0f32, 0.0, 0.0, 1.0];
        let r = [0.0f32, 0.5];
        let mut v = [0.0f32; 2];
        latent_value_into(&states, 2, &c, &r, &LatentValueConfig::DEFAULT, None, &mut v);
        assert!(v[0] > v[1], "{v:?}");
    }

    #[test]
    fn nan_state_never_wins() {
        let c = [0.0f32; 2];
        let states = [f32::NAN, 0.0, 1.0, 0.0, 0.9, 0.1];
        let r = [0.0f32, 0.0, 0.0];
        let mut v = [0.0f32; 3];
        latent_value_into(&states, 3, &c, &r, &LatentValueConfig::DEFAULT, None, &mut v);
        assert_ne!(select_best(&v), 0, "{v:?}");
        assert_eq!(select_best(&[f32::NAN, 0.1, f32::NAN]), 1);
        assert_eq!(select_best(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(select_best(&[0.5, 0.5, 0.2]), 0, "ties keep the lowest index");
    }

    #[test]
    fn frozen_direction_breaks_a_symmetric_tie() {
        let c = [0.0f32; 2];
        let states = [1.0f32, 0.0, -1.0, 0.0];
        let r = [0.0f32; 2];
        let mut v = [0.0f32; 2];
        let d = [-1.0f32, 0.0];
        latent_value_into(&states, 2, &c, &r, &LatentValueConfig::DEFAULT, Some(&d), &mut v);
        assert_eq!(select_best(&v), 1, "{v:?}");
    }
}
