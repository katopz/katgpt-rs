//! L1-regularized regression (Lasso) via cyclic coordinate descent.
//!
//! Pure inference — no external linear-algebra dependency. This is the only
//! numerical core of the CS-KV probe; keeping it self-contained avoids pulling
//! a full LA crate in for a single small solver.
//!
//! # Objective (Form A — bare-alpha scaling)
//!
//! `minimize_x  (1/2) · ‖Phi · x − y‖² + alpha · ‖x‖₁`
//!
//! where `Phi` is `M × N` (rows are measurement masks cast to `{0.0, 1.0}`)
//! and `y` is length `M`. Note: this is **not** the `(1/2M)` mean-scaled form.
//! With this form the coordinate-descent soft-threshold uses bare `alpha`
//! (no `M` multiplier). The probe default `alpha = 1e-4` is therefore light
//! regularization — appropriate because the probe is overdetermined
//! (`M = 200 ≫ N = 64`) and behaves near-OLS, which is what we want for clean
//! support recovery.
//!
//! # Algorithm
//!
//! Standard cyclic coordinate descent with the residual trick. Let
//! `r_m = y_m − Σ_k Phi[m][k]·x_k` (the *negative* prediction error). For each
//! coordinate `j`:
//!
//! ```text
//! z_j   = Σ_m Phi[m][j]²                           (precomputed once)
//! rho_j = Σ_m Phi[m][j]·r_m + z_j·x_j              (= Σ_m Phi[m][j]·b_m,
//!                                                    b_m = j-excluded target)
//! x_j'  = soft_threshold(rho_j, alpha) / z_j
//! ```
//!
//! Then update the residual incrementally: `r_m −= (x_j' − x_j)·Phi[m][j]`.
//! This avoids recomputing the full `Phi·x` product each sweep (O(M·N) per
//! coordinate → O(M·N²) per sweep would be wasteful; the residual update is
//! O(M) per coordinate).

/// Solve the Lasso in Form A via coordinate descent.
///
/// - `phi`: `M × N` measurement matrix (rows MUST be equal length).
/// - `y`: length `M` centered observations.
/// - `alpha`: L1 penalty (bare, see module docs).
/// - `n_iter`: number of full cyclic sweeps.
///
/// Returns coefficient vector `x` of length `N`. Empty input → empty output.
pub fn lasso(phi: &[Vec<f32>], y: &[f32], alpha: f32, n_iter: usize) -> Vec<f32> {
    let m = phi.len();
    if m == 0 {
        return Vec::new();
    }
    let n = phi[0].len();
    // Degenerate: no features to fit.
    if n == 0 {
        return Vec::new();
    }
    let mut x = vec![0.0_f32; n];

    // Precompute column squared-norms z_j = Σ_m Phi[m][j]². Constant across sweeps.
    let mut z = vec![0.0_f32; n];
    for j in 0..n {
        let mut acc = 0.0_f32;
        for row in phi.iter() {
            // Defensive: rows are expected equal-length; guard ragged input.
            match j < row.len() {
                true => acc += row[j] * row[j],
                false => {}
            }
        }
        z[j] = acc;
    }

    // Residual r_m = y_m − Phi·x. With x = 0 this is just y.
    let mut r = y.to_vec();
    // Trim r to match the row count defensively (ragged y is a caller bug,
    // but we must not index out of bounds).
    if r.len() < m {
        r.resize(m, 0.0);
    }

    for _ in 0..n_iter {
        for j in 0..n {
            // Skip dead coordinates (a column of all zeros can never carry weight).
            if z[j] < 1e-20 {
                continue;
            }
            // rho_j = Σ_m Phi[m][j]·r_m + z_j·x_j
            let mut phi_dot_r = 0.0_f32;
            for (row, &rm) in phi.iter().zip(r.iter()) {
                match j < row.len() {
                    true => phi_dot_r += row[j] * rm,
                    false => {}
                }
            }
            let rho_j = phi_dot_r + z[j] * x[j];
            let x_new = soft_threshold(rho_j, alpha) / z[j];
            let delta = x_new - x[j];
            // Only touch the residual when the update is numerically meaningful.
            if delta.abs() > 1e-12 {
                for (row, rm) in phi.iter().zip(r.iter_mut()) {
                    match j < row.len() {
                        true => *rm -= delta * row[j],
                        false => {}
                    }
                }
                x[j] = x_new;
            }
        }
    }
    x
}

/// Soft-thresholding (proximal) operator: `sign(rho) · max(|rho| − alpha, 0)`.
///
/// Prefer match over chained `if`s per house style. Branchless on sign except
/// for the `|rho| <= alpha` shrink-to-zero band.
#[inline(always)]
fn soft_threshold(rho: f32, alpha: f32) -> f32 {
    match rho.partial_cmp(&alpha) {
        Some(std::cmp::Ordering::Greater) => rho - alpha,
        Some(std::cmp::Ordering::Less) => {
            // rho < alpha. If rho < -alpha → rho + alpha; else (|rho|<=alpha) → 0.
            match rho.partial_cmp(&-alpha) {
                Some(std::cmp::Ordering::Less) => rho + alpha,
                _ => 0.0,
            }
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recovery test for the SOLVER (not the probe). Uses a well-conditioned,
    /// independent-Bernoulli(0.5) Phi so column correlations are low — this
    /// isolates solver correctness from the probe's mask distribution. Only
    /// heads {3, 17, 42} of 64 carry signal; assert the top-3 by |coeff| ⊇ them.
    #[test]
    fn test_lasso_recovers_known_sparse_ground_truth() {
        // Fixed seed for deterministic recovery (Plan 280 Risk #1 is the
        // solver; this test must be reproducible).
        let mut rng = fastrand::Rng::with_seed(0xC507_2806u64);
        let m = 200_usize;
        let n = 64_usize;

        // Independent Bernoulli(0.5) measurement matrix.
        let phi: Vec<Vec<f32>> = (0..m)
            .map(|_| (0..n).map(|_| if rng.bool() { 1.0 } else { 0.0 }).collect())
            .collect();

        // 3-sparse ground truth: heads 3, 17, 42 carry the signal.
        let mut true_x = vec![0.0_f32; n];
        true_x[3] = 2.0;
        true_x[17] = 1.5;
        true_x[42] = 1.0;

        // y = Phi · true_x + small gaussian-ish noise (sum-of-uniforms).
        let y: Vec<f32> = phi
            .iter()
            .map(|row| {
                let pred: f32 = row.iter().zip(true_x.iter()).map(|(a, b)| a * b).sum();
                let noise: f32 = (0..12).map(|_| rng.f32()).sum::<f32>() - 6.0; // ~N(0,1)
                pred + 0.01 * noise
            })
            .collect();

        let alpha = 1e-4_f32;
        let x_hat = lasso(&phi, &y, alpha, 1000);

        // Rank indices by |coefficient|, descending.
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            x_hat[b].abs().partial_cmp(&x_hat[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
        });

        let top3: std::collections::HashSet<usize> = idx.iter().take(3).copied().collect();
        for &signal_head in &[3usize, 17, 42] {
            assert!(
                top3.contains(&signal_head),
                "Lasso failed to surface signal head {signal_head} in top-3; \
                 top-3 = {:?}, coeffs = {:?}",
                top3,
                &x_hat[..8.min(n)],
            );
        }

        // Recovered magnitudes should be close to ground truth (light alpha).
        assert!(
            (x_hat[3] - 2.0).abs() < 0.15,
            "head 3 coeff drift: {}",
            x_hat[3]
        );
    }

    #[test]
    fn test_lasso_empty_inputs() {
        assert!(lasso(&[], &[], 1e-4, 10).is_empty());
        assert!(lasso(&[vec![]], &[0.0], 1e-4, 10).is_empty());
    }

    #[test]
    fn test_soft_threshold_shrinkage() {
        // Tolerance is 1e-6, not 1e-9: soft_threshold returns
        // `rho - alpha` which for 0.7 - 0.5 lands at 0.19999998 in f32 —
        // about 2e-8 of representation error from the ideal 0.2.
        assert!(soft_threshold(0.0, 0.5).abs() < 1e-6);
        assert!((soft_threshold(0.3, 0.5) - 0.0).abs() < 1e-6); // |0.3|<0.5 → 0
        assert!((soft_threshold(0.7, 0.5) - 0.2).abs() < 1e-6);
        assert!((soft_threshold(-0.7, 0.5) + 0.2).abs() < 1e-6);
        assert!((soft_threshold(-0.3, 0.5) - 0.0).abs() < 1e-6);
    }
}
