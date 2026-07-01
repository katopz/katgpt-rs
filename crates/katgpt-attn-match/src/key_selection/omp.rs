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

// Index-based loops and wide signatures are intentional for the OMP numerical kernel.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::{
    STABILITY_EPS,
    beta_fitter::{BetaFitConfig, fit_beta_nnls},
    key_selection::KeySelection,
    score_matrix::compute_score_matrix,
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

    // Apply max-shift per row for numerical stability, then exp, and compute
    // the per-query target mass `m_i = Σ_j Φ_ij` (post-shift) — all in one
    // fused row-by-row walk. The prior form ran three separate loops over the
    // full `n × t_len` matrix (max scan, exp, sum) and kept a `max_per_row`
    // `Vec<f32>` scratch. Fusing to two passes per row keeps each row hot in L1
    // between the max scan and the exp+sum pass, and eliminates the
    // `max_per_row` allocation entirely.
    let mut m_target = vec![0.0f32; n];
    for i in 0..n {
        // Pass 1: per-row max (immutable borrow of `phi`).
        let m = {
            let row = &phi[i * t_len..(i + 1) * t_len];
            let mut m = row[0];
            for &v in &row[1..] {
                m = m.max(v);
            }
            m
        };
        // Pass 2: shifted exp (in place) + fused mass accumulation.
        let row = &mut phi[i * t_len..(i + 1) * t_len];
        let mut s = 0.0f32;
        for v in row.iter_mut() {
            let e = (*v - m).exp();
            *v = e;
            s += e;
        }
        m_target[i] = s;
    }

    // Step 3: Greedy selection.
    let mut selected: Vec<usize> = Vec::with_capacity(t);
    let mut residual = m_target.clone();
    let mut in_selected = vec![false; t_len];

    // Hoisted scratch buffers — reused across greedy iterations to avoid
    // per-iteration allocation. top_k_buf holds the (idx, corr) candidates
    // for the current iteration; a_sub holds the phi[:, selected] sub-matrix
    // rebuilt on each periodic NNLS refit. corr_all holds the per-key
    // correlation scores computed in a single cache-friendly sweep.
    let mut top_k_buf: Vec<(usize, f32)> = Vec::with_capacity(k);
    let mut a_sub: Vec<f32> = Vec::with_capacity(n * t);
    let mut corr_all: Vec<f32> = vec![0.0f32; t_len];

    let mut iter_count = 0usize;
    while selected.len() < t {
        // Correlation scores: c = Φ^T residual, computed in one cache-friendly
        // pass. The i-outer form reads phi row-by-row (sequential) and writes
        // corr_all sequentially. The iterator form (`zip`) lets LLVM elide
        // bounds checks and auto-vectorize the FMA inner loop with the broadcast
        // scalar `r`.
        corr_all.fill(0.0);
        for (r, phi_row) in residual.iter().zip(phi.chunks_exact(t_len)) {
            for (corr, &pv) in corr_all.iter_mut().zip(phi_row) {
                *corr += pv * r;
            }
        }

        // Pick top-k from corr_all, skipping selected keys.
        top_k_buf.clear();

        if k == 1 {
            // Fast path: argmax only — no top-k buffer maintenance. This is the
            // default selector and the hottest OMP path, so skipping the
            // linear-min-replacement bookkeeping (which is O(k) per candidate
            // even for k=1) saves a branch + a tiny inner scan per candidate.
            let mut best_j = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for j in 0..t_len {
                if in_selected[j] {
                    continue;
                }
                let corr = corr_all[j];
                if corr > best_score {
                    best_score = corr;
                    best_j = j;
                }
            }
            if !in_selected[best_j] && selected.len() < t {
                selected.push(best_j);
                in_selected[best_j] = true;
            }
        } else {
            // Multi-key path: track top-k via unsorted buffer with linear-min
            // replacement. The prior code called `sort_by` on every replacement
            // once the buffer was full — O(k log k) per candidate. This linear
            // scan is O(k) per candidate. Correctness is preserved because
            // `top_k_buf` is consumed without order dependency (dedup via
            // `in_selected`).
            for j in 0..t_len {
                if in_selected[j] {
                    continue;
                }
                let corr = corr_all[j];
                if top_k_buf.len() < k {
                    top_k_buf.push((j, corr));
                } else {
                    // Find the slot with the minimum corr and replace if larger.
                    let mut min_idx = 0;
                    for m in 1..top_k_buf.len() {
                        if top_k_buf[m].1 < top_k_buf[min_idx].1 {
                            min_idx = m;
                        }
                    }
                    if corr > top_k_buf[min_idx].1 {
                        top_k_buf[min_idx] = (j, corr);
                    }
                }
            }
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
        if iter_count.is_multiple_of(tau) || selected.len() >= t {
            // Build A = phi[:, selected] ∈ R^{n × |selected|}.
            // Row-major (i-outer) construction reads each phi row sequentially.
            // The prior col-major form did strided `phi[i*t_len + sel_idx]`
            // reads — one cache miss per i per selected key.
            let cur_t = selected.len();
            a_sub.clear();
            a_sub.resize(n * cur_t, 0.0);
            for (i, phi_row) in phi.chunks_exact(t_len).enumerate() {
                let a_row = &mut a_sub[i * cur_t..(i + 1) * cur_t];
                for (col, &sel_idx) in selected.iter().enumerate() {
                    a_row[col] = phi_row[sel_idx];
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
            for (i, row) in a_sub.chunks_exact(cur_t).enumerate() {
                let aw_i = row
                    .iter()
                    .zip(&beta_result.weights)
                    .map(|(&a, &w)| a * w)
                    .sum::<f32>();
                residual[i] = m_target[i] - aw_i;
            }
            // Note: weights are recomputed in the final fit below; we don't keep
            // them here because the subset size will change in subsequent iters.
        }
    }

    // Final NNLS fit on the full selected set.
    //
    // Reuse `a_sub` (capacity `n * t`) instead of allocating a separate
    // `a_final`. The loop guarantees `selected.len() == t`, so `final_t == t`
    // and `a_sub` has sufficient capacity — `clear() + resize()` won't
    // reallocate. Eliminates one `n * t` allocation per `select_omp_keys`.
    let final_t = selected.len();
    a_sub.clear();
    a_sub.resize(n * final_t, 0.0);
    // Row-major construction — see periodic refit above for rationale.
    for (i, phi_row) in phi.chunks_exact(t_len).enumerate() {
        let a_row = &mut a_sub[i * final_t..(i + 1) * final_t];
        for (col, &sel_idx) in selected.iter().enumerate() {
            a_row[col] = phi_row[sel_idx];
        }
    }
    let cfg = BetaFitConfig {
        iters: 0,
        w_lower,
        w_upper,
        power_iter_steps: 0,
    };
    let final_result = fit_beta_nnls(&a_sub, &m_target, n, final_t, &cfg);

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
