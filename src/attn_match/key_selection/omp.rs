//! Orthogonal Matching Pursuit (OMP) key selector.
//!
//! Greedily builds the compact key set `Ck` to best match the attention mass:
//! at each step, adds the key whose mass feature column maximally reduces the
//! residual `m − Φw`, then refits `w` via NNLS every τ iterations.
//!
//! Per the paper (Section 3.3, Algorithm 2): OMP outperforms HighestAttnKeys
//! empirically but is slower — selecting multiple keys per step (k > 1) and
//! refitting at intervals (τ > 1) reduces compaction time 4–8× with little
//! degradation.
//!
//! The returned `weights` are `w = exp(β)` directly — no separate β fit needed.

use crate::attn_match::{
    beta_fitter::{fit_beta_nnls, BetaFitConfig},
    key_selection::KeySelection,
    score_matrix::compute_score_matrix,
    STABILITY_EPS,
};

/// Select t keys via Orthogonal Matching Pursuit on the mass feature matrix.
///
/// # Arguments
/// * `keys` - Original `(T, d)` key matrix, flat row-major.
/// * `queries` - Reference queries `(n, d)`, flat row-major.
/// * `t` - Number of keys to select.
/// * `k` - Keys selected per greedy iteration (paper `k`).
/// * `tau` - NNLS refit interval (paper `τ`).
/// * `t_len` - Original sequence length `T`.
/// * `d` - Head dimension.
/// * `n` - Number of reference queries.
/// * `w_lower` / `w_upper` - Box constraints for the NNLS refit.
pub fn select_omp_keys(
    keys: &[f32],
    queries: &[f32],
    t: usize,
    k: usize,
    tau: usize,
    t_len: usize,
    d: usize,
    n: usize,
    w_lower: f32,
    w_upper: f32,
) -> KeySelection {
    assert_eq!(keys.len(), t_len * d);
    assert_eq!(queries.len(), n * d);
    assert!(t <= t_len);
    assert!(k >= 1);
    assert!(tau >= 1);
    assert!(w_lower > 0.0);

    // Step 1: Compute the mass feature matrix Φ ∈ R^{n×T} where Φ_ij = exp(q_i K_j^T / √d).
    let mut phi = vec![0.0f32; n * t_len];
    compute_score_matrix(queries, keys, n, t_len, d, &mut phi);

    // Apply max-shift per row for numerical stability, then exp.
    let mut max_per_row = vec![f32::NEG_INFINITY; n];
    for i in 0..n {
        let row = &phi[i * t_len..(i + 1) * t_len];
        let mut m = row[0];
        for &v in &row[1..] {
            m = m.max(v);
        }
        max_per_row[i] = m;
    }
    for i in 0..n {
        let m = max_per_row[i];
        let row = &mut phi[i * t_len..(i + 1) * t_len];
        for v in row.iter_mut() {
            *v = (*v - m).exp();
        }
    }

    // Step 2: Compute target mass m_i = Σ_j Φ_ij (post-shift).
    let mut m_target = vec![0.0f32; n];
    for i in 0..n {
        let row = &phi[i * t_len..(i + 1) * t_len];
        let mut s = 0.0f32;
        for &v in row {
            s += v;
        }
        m_target[i] = s;
    }

    // Step 3: Greedy selection.
    let mut selected: Vec<usize> = Vec::with_capacity(t);
    let mut residual = m_target.clone();
    let mut in_selected = vec![false; t_len];

    let mut iter_count = 0usize;
    while selected.len() < t {
        // Correlation scores: c_j = residual^T Φ_:,j  (for j not in selected).
        let mut best_j = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        let mut top_k_buf: Vec<(usize, f32)> = Vec::with_capacity(k);
        for j in 0..t_len {
            if in_selected[j] {
                continue;
            }
            let col_j = &phi[j..]; // stride-T access
            let mut corr = 0.0f32;
            for i in 0..n {
                corr += residual[i] * col_j[i * t_len];
            }
            if corr > best_score {
                best_score = corr;
                best_j = j;
            }
            // Track top-k via simple insertion into top_k_buf.
            if top_k_buf.len() < k {
                top_k_buf.push((j, corr));
                if top_k_buf.len() == k {
                    // Heapify or sort to find min for replacement.
                    top_k_buf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }
            } else if corr > top_k_buf[0].1 {
                top_k_buf[0] = (j, corr);
                // Re-sift min to front.
                top_k_buf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        if k == 1 {
            // Single-key path — fastest.
            if !in_selected[best_j] && selected.len() < t {
                selected.push(best_j);
                in_selected[best_j] = true;
            }
        } else {
            // Add top-k new keys (deduplicated).
            for &(j, _) in &top_k_buf {
                if !in_selected[j] && selected.len() < t {
                    selected.push(j);
                    in_selected[j] = true;
                }
                if selected.len() >= t {
                    break;
                }
            }
        }

        iter_count += 1;

        // Periodic NNLS refit.
        if iter_count % tau == 0 || selected.len() >= t {
            // Build A = phi[:, selected] ∈ R^{n × |selected|}.
            let cur_t = selected.len();
            let mut a_sub = vec![0.0f32; n * cur_t];
            for (col, &sel_idx) in selected.iter().enumerate() {
                for i in 0..n {
                    a_sub[i * cur_t + col] = phi[i * t_len + sel_idx];
                }
            }
            // Solve NNLS on the subset.
            let cfg = BetaFitConfig {
                iters: 0, // paper: 0 iters for OMP (clamped LS)
                w_lower,
                w_upper,
                power_iter_steps: 0,
            };
            let beta_result = fit_beta_nnls(&a_sub, &m_target, n, cur_t, &cfg);
            // Update residual: r = m − A w.
            for i in 0..n {
                let row = &a_sub[i * cur_t..(i + 1) * cur_t];
                let mut aw_i = 0.0f32;
                for j in 0..cur_t {
                    aw_i += row[j] * beta_result.weights[j];
                }
                residual[i] = m_target[i] - aw_i;
            }
            // Note: weights are recomputed in the final fit below; we don't keep
            // them here because the subset size will change in subsequent iters.
        }
    }

    // Final NNLS fit on the full selected set.
    let final_t = selected.len();
    let mut a_final = vec![0.0f32; n * final_t];
    for (col, &sel_idx) in selected.iter().enumerate() {
        for i in 0..n {
            a_final[i * final_t + col] = phi[i * t_len + sel_idx];
        }
    }
    let cfg = BetaFitConfig {
        iters: 0,
        w_lower,
        w_upper,
        power_iter_steps: 0,
    };
    let final_result = fit_beta_nnls(&a_final, &m_target, n, final_t, &cfg);

    KeySelection {
        indices: selected,
        weights: final_result.weights,
    }
}

/// Compute the residual mass coverage: `1 - ||residual||_1 / ||m||_1`.
/// Used as the OMP convergence diagnostic (GOAT G3: residual < 5% of initial).
#[inline]
pub fn mass_coverage(residual: &[f32], m: &[f32]) -> f32 {
    let mut res_sum = 0.0f32;
    let mut m_sum = 0.0f32;
    for i in 0..residual.len().min(m.len()) {
        res_sum += residual[i].abs();
        m_sum += m[i].abs();
    }
    if m_sum < STABILITY_EPS {
        return 0.0;
    }
    1.0 - (res_sum / m_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omp_selects_unique_keys() {
        let t_len = 12;
        let d = 4;
        let n = 3;
        let t = 4;
        let keys: Vec<f32> = (0..t_len * d).map(|i| (i as f32) * 0.1).collect();
        let queries: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.2).collect();
        let sel = select_omp_keys(&keys, &queries, t, 1, 1, t_len, d, n, 1e-3, 100.0);
        assert_eq!(sel.indices.len(), t);
        // All indices unique and in range.
        let mut sorted = sel.indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), t);
        for &i in &sel.indices {
            assert!(i < t_len);
        }
    }

    #[test]
    fn test_omp_fast_selects_k_keys_per_iter() {
        let t_len = 16;
        let d = 4;
        let n = 4;
        let t = 8;
        let keys: Vec<f32> = (0..t_len * d).map(|i| (i as f32).sin() * 0.5).collect();
        let queries: Vec<f32> = (0..n * d).map(|i| (i as f32).cos() * 0.3).collect();
        let sel = select_omp_keys(&keys, &queries, t, 4, 2, t_len, d, n, 1e-3, 100.0);
        assert_eq!(sel.indices.len(), t);
        // Indices unique.
        let mut sorted = sel.indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), t);
    }

    #[test]
    fn test_omp_picks_relevant_keys() {
        // Build a scenario where one key has much higher exp score than others.
        let t_len = 5;
        let d = 4;
        let n = 1;
        let t = 1;
        let mut keys = vec![0.0f32; t_len * d];
        // Key 3 strongly aligned with query.
        keys[3 * d] = 5.0;
        keys[3 * d + 1] = 5.0;
        let mut queries = vec![0.0f32; n * d];
        queries[0] = 5.0;
        queries[1] = 5.0;
        let sel = select_omp_keys(&keys, &queries, t, 1, 1, t_len, d, n, 1e-3, 100.0);
        assert_eq!(sel.indices.len(), 1);
        // With one iteration and t=1, OMP should pick the most correlated key.
        assert_eq!(sel.indices[0], 3);
    }

    #[test]
    fn test_mass_coverage_bounds() {
        let m = vec![1.0, 2.0, 3.0];
        let res_zero = vec![0.0, 0.0, 0.0];
        let res_full = vec![1.0, 2.0, 3.0];
        assert!((mass_coverage(&res_zero, &m) - 1.0).abs() < 1e-6);
        assert!((mass_coverage(&res_full, &m) - 0.0).abs() < 1e-6);
    }
}
