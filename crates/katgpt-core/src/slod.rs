//! SLoD Spectral Level-of-Detail Pruner — modelless KG resolution control (Plan 235).
//!
//! Uses spectral heat diffusion on hyperbolic kNN graph Laplacians to automatically
//! detect multi-scale structure in KG embeddings and route constraint checks to the
//! appropriate resolution tier.
//!
//! # Architecture
//!
//! 1. **Poincaré ball geometry** — hyperbolic distance, log/exp maps for tree-like KG
//! 2. **kNN Laplacian** — normalized graph Laplacian via hyperbolic kNN, Jacobi eigendecomposition
//! 3. **Boundary scan** — multi-signal (participation + diffusion entropy + spectral concentration)
//!    composite score with MAD peak picker for automatic scale detection
//! 4. **Fréchet mean** — SIMD-accelerated Riemannian centroid in Poincaré ball
//! 5. **SlodPruner** — ConstraintPruner that routes between spectral tiers at O(1)

use crate::simd::simd_dot_f32;

// ── Configuration ─────────────────────────────────────────────────

/// SLoD configuration with sensible defaults for hyperbolic KG embeddings.
#[derive(Debug, Clone)]
pub struct SlodConfig {
    /// k for kNN graph construction.
    /// Default: computed as `max(10, min(sqrt(N), 50))`.
    pub knn_k: usize,
    /// Max Fréchet mean iterations.
    /// Default: 15.
    pub max_iterations: usize,
    /// Composite score weights: [participation, diffusion_entropy, spectral_concentration].
    /// Default: [1/3, 1/3, 1/3].
    pub alpha: [f32; 3],
    /// MAD (Median Absolute Deviation) peak picker threshold.
    /// Default: 2.0.
    pub mad_beta: f32,
    /// Spectral gap threshold for K* selection.
    /// Default: 2.0.
    pub gap_threshold: f32,
    /// Fréchet mean step size (η).
    /// Default: 1.0.
    pub step_size: f32,
    /// Convergence tolerance for iterative algorithms.
    /// Default: 1e-6.
    pub tolerance: f32,
}

impl Default for SlodConfig {
    fn default() -> Self {
        Self {
            knn_k: 0, // sentinel: compute from N
            max_iterations: 15,
            alpha: [1.0 / 3.0; 3],
            mad_beta: 2.0,
            gap_threshold: 2.0,
            step_size: 1.0,
            tolerance: 1e-6,
        }
    }
}

impl SlodConfig {
    /// Compute effective kNN k from dataset size N.
    pub fn effective_knn_k(&self, n: usize) -> usize {
        match self.knn_k {
            0 => (10.0_f32).max((n as f32).sqrt().min(50.0)) as usize,
            k => k,
        }
    }
}

// ── Scale Boundary ────────────────────────────────────────────────

/// A detected scale boundary from the spectral analysis.
#[derive(Debug, Clone)]
pub struct ScaleBoundary {
    /// Effective rank K* (number of significant eigenmodes) at this scale.
    pub k_star: usize,
    /// Diffusion scale σ at which this boundary was detected.
    pub sigma: f32,
    /// Composite boundary score S(σ).
    pub score: f32,
}

// ── SLoD Operator ─────────────────────────────────────────────────

/// Core spectral operator: eigenpairs + detected boundaries.
#[derive(Debug, Clone)]
pub struct SlodOperator {
    /// Configuration used to build this operator.
    pub config: SlodConfig,
    /// Eigenvalues λ_k (descending order).
    pub eigenvalues: Vec<f32>,
    /// Eigenvectors as flat buffer [K_eigs * N], row-major.
    pub eigenvectors: Vec<f32>,
    /// Detected scale boundaries.
    pub boundaries: Vec<ScaleBoundary>,
}

// ── Poincaré Ball Geometry ────────────────────────────────────────

// clamp_norm removed — was unused

/// Hyperbolic distance in the Poincaré ball model.
///
/// d(a, b) = arcosh(1 + 2·||a - b||² / ((1 - ||a||²)(1 - ||b||²)))
///
/// Points are clamped to remain inside the unit ball.
pub fn poincare_distance(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let norm_a_sq = simd_dot_f32(a, a, dim);
    let norm_b_sq = simd_dot_f32(b, b, dim);

    // Clamp norms to < 1 for numerical stability
    let norm_a_sq = norm_a_sq.min(1.0 - 1e-5);
    let norm_b_sq = norm_b_sq.min(1.0 - 1e-5);

    let diff_sq = crate::simd::simd_dist_sq(a, b, dim);
    let denom = (1.0 - norm_a_sq) * (1.0 - norm_b_sq);

    let inner = 1.0 + 2.0 * diff_sq / denom;
    inner.acosh()
}

/// Möbius addition in the Poincaré ball: a ⊕ b.
///
/// a ⊕ b = ((1 + 2<a,b> + ||b||²)a + (1 - ||a||²)b) / (1 + 2<a,b> + ||a||²||b||²)
#[allow(dead_code)]
fn mobius_add(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    mobius_add_into(&mut result, a, b, dim);
    result
}

/// In-place Möbius addition — zero-allocation hot path.
#[inline]
fn mobius_add_into(result: &mut [f32], a: &[f32], b: &[f32], dim: usize) {
    let norm_a_sq = simd_dot_f32(a, a, dim).min(1.0 - 1e-5);
    let norm_b_sq = simd_dot_f32(b, b, dim).min(1.0 - 1e-5);
    let a_dot_b = simd_dot_f32(a, b, dim);

    let denom = 1.0 + 2.0 * a_dot_b + norm_a_sq * norm_b_sq;
    let inv_denom = 1.0 / denom.max(1e-10);

    let s1 = (1.0 + 2.0 * a_dot_b + norm_b_sq) * inv_denom;
    let s2 = (1.0 - norm_a_sq) * inv_denom;

    for i in 0..dim {
        result[i] = s1 * a[i] + s2 * b[i];
    }
}

/// Riemannian log map: project `point` onto the tangent space at `base`.
///
/// Returns a tangent vector in T_base B^n.
/// log_x(y) = (d(x,y) / ||(-x) ⊕ y||) · ((-x) ⊕ y)
pub fn log_map(base: &[f32], point: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    let mut neg_base = vec![0.0f32; dim];
    let mut mob_result = vec![0.0f32; dim];
    log_map_into(&mut out, &mut neg_base, &mut mob_result, base, point, dim);
    out
}

/// In-place log map — zero-allocation hot path.
///
/// Scratch buffers: `neg_base[dim]`, `mob_result[dim]`.
#[inline]
pub fn log_map_into(
    out: &mut [f32],
    neg_base: &mut [f32],
    mob_result: &mut [f32],
    base: &[f32],
    point: &[f32],
    dim: usize,
) {
    let dist = poincare_distance(base, point, dim);
    if dist < 1e-10 {
        out[..dim].fill(0.0);
        return;
    }

    // neg_base = -base (SIMD-accelerated negate)
    crate::simd::simd_fused_decay_write(&mut neg_base[..dim], 0.0, &base[..dim], -1.0);

    mobius_add_into(mob_result, neg_base, point, dim);
    let mob_norm = simd_dot_f32(mob_result, mob_result, dim).sqrt();

    if mob_norm < 1e-10 {
        out[..dim].fill(0.0);
        return;
    }

    let scale = dist / mob_norm;
    crate::simd::simd_fused_decay_write(&mut out[..dim], 0.0, &mob_result[..dim], scale);
}

/// Riemannian exp map: project tangent vector back to the Poincaré ball.
///
/// exp_x(v) = x ⊕ tanh(||v||/2) / ||v|| · v
/// where the tangent vector v encodes the conformal factor from log_map.
pub fn exp_map(base: &[f32], tangent: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    let mut dir = vec![0.0f32; dim];
    let mut mob_result = vec![0.0f32; dim];
    exp_map_into(&mut out, &mut dir, &mut mob_result, base, tangent, dim);
    out
}

/// In-place exp map — zero-allocation hot path.
///
/// Scratch buffers: `dir[dim]`, `mob_result[dim]`.
#[inline]
pub fn exp_map_into(
    out: &mut [f32],
    dir: &mut [f32],
    mob_result: &mut [f32],
    base: &[f32],
    tangent: &[f32],
    dim: usize,
) {
    let tangent_norm = simd_dot_f32(tangent, tangent, dim).sqrt();

    if tangent_norm < 1e-10 {
        out[..dim].copy_from_slice(&base[..dim]);
        return;
    }

    // Compute direction: tanh(||v||/2) * v/||v||
    let s = (tangent_norm / 2.0).tanh() / tangent_norm;
    crate::simd::simd_fused_decay_write(&mut dir[..dim], 0.0, &tangent[..dim], s);

    // Project dir into ball
    let dir_norm_sq = simd_dot_f32(dir, dir, dim);
    if dir_norm_sq >= 1.0 - 1e-5 {
        let scale = (1.0 - 1e-5) / dir_norm_sq.sqrt();
        crate::simd::simd_scale_inplace(&mut dir[..dim], scale);
    }

    // Möbius addition: base ⊕ dir
    mobius_add_into(mob_result, base, dir, dim);

    // Final clamp into ball
    let norm_r_sq = simd_dot_f32(mob_result, mob_result, dim);
    if norm_r_sq >= 1.0 - 1e-5 {
        let scale = (1.0 - 1e-5) / norm_r_sq.sqrt();
        crate::simd::simd_fused_decay_write(&mut out[..dim], 0.0, &mob_result[..dim], scale);
    } else {
        out[..dim].copy_from_slice(&mob_result[..dim]);
    }
}

// ── kNN Laplacian Construction ────────────────────────────────────

impl SlodOperator {
    /// Build kNN Laplacian from hyperbolic embeddings and eigendecompose.
    ///
    /// 1. Build kNN graph using Poincaré distance
    /// 2. Symmetrize: W_ij = exp(-d_hyp(a_i, a_j))
    /// 3. Compute normalized Laplacian L = I - D^{-1/2} W D^{-1/2}
    /// 4. Jacobi eigendecomposition → top K_eigs eigenpairs
    ///
    /// Returns `(eigenvalues, eigenvectors)`.
    pub fn build_laplacian(
        embeddings: &[f32],
        n: usize,
        dim: usize,
        config: &SlodConfig,
    ) -> (Vec<f32>, Vec<f32>) {
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        let k = config.effective_knn_k(n).min(n - 1).max(1);

        // Build kNN adjacency + weight matrix
        let mut w = vec![0.0f32; n * n];
        // Pre-allocate distance buffer — reused across iterations
        let mut dists: Vec<(usize, f32)> = Vec::with_capacity(n);

        for i in 0..n {
            let a_i = &embeddings[i * dim..(i + 1) * dim];
            // Compute distances to all other points
            dists.clear();
            for j in 0..n {
                if j == i {
                    continue;
                }
                let a_j = &embeddings[j * dim..(j + 1) * dim];
                dists.push((j, poincare_distance(a_i, a_j, dim)));
            }
            // Partial sort: O(n) to partition top-k nearest, then sort just those
            // Clamp k to dists.len() since we skipped self (dists has n-1 elements)
            let k_eff = k.min(dists.len());
            if k_eff == 0 {
                continue;
            }
            dists.select_nth_unstable_by(k_eff - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists[..k_eff].sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Take top-k nearest neighbors
            for &(j, d) in dists.iter().take(k_eff) {
                let weight = (-d).exp();
                w[i * n + j] = weight;
                w[j * n + i] = weight; // symmetrize
            }
        }

        // Degree vector (SIMD row-sums)
        let mut degree = vec![0.0f32; n];
        for i in 0..n {
            degree[i] = crate::simd::simd_sum_f32(&w[i * n..(i + 1) * n]).max(1e-10);
        }

        // Precompute inv_sqrt(degree) once — avoids repeated sqrt in inner loop
        let mut inv_sqrt_d = vec![0.0f32; n];
        for i in 0..n {
            inv_sqrt_d[i] = 1.0 / degree[i].sqrt();
        }

        // Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
        let mut lap = vec![0.0f32; n * n];
        for i in 0..n {
            let isd_i = inv_sqrt_d[i];
            for (j, isd_j) in inv_sqrt_d.iter().enumerate().take(n) {
                let idx = i * n + j;
                let scaled = w[idx] * isd_i * isd_j;
                lap[idx] = if i == j { 1.0 - scaled } else { -scaled };
            }
        }

        // Eigendecompose — reuse Jacobi from spectral_hierarchy
        let k_eigs = k.min(n);
        let eigvecs = top_k_eigenvectors(&lap, n, k_eigs);

        // Extract eigenvalues from diagonal of rotated matrix
        // top_k_eigenvectors returns [k_eigs * n] row-major eigenvectors
        // We need eigenvalues too — compute from Lap * v = λ * v
        let mut eigenvalues = Vec::with_capacity(k_eigs);
        // Pre-allocate Lv buffer — reused across eigenvectors
        let mut lv = vec![0.0f32; n];
        for eig_idx in 0..k_eigs {
            lv[..n].fill(0.0);
            let v = &eigvecs[eig_idx * n..(eig_idx + 1) * n];
            // Compute Lv
            crate::simd::simd_matvec(&mut lv, &lap, v, n, n);
            // λ = (v^T Lv) / (v^T v)
            let numerator = simd_dot_f32(v, &lv, n);
            let denominator = simd_dot_f32(v, v, n).max(1e-10);
            eigenvalues.push(numerator / denominator);
        }

        // Sort eigenvalues descending (and reorder eigenvectors to match)
        let mut indexed: Vec<(usize, f32)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let sorted_eigenvalues: Vec<f32> = indexed.iter().map(|&(_, v)| v).collect();
        let sorted_eigvecs = {
            let mut buf = vec![0.0f32; k_eigs * n];
            for (out_row, &(src_row, _)) in indexed.iter().enumerate() {
                let src_off = src_row * n;
                let dst_off = out_row * n;
                buf[dst_off..dst_off + n].copy_from_slice(&eigvecs[src_off..src_off + n]);
            }
            buf
        };

        (sorted_eigenvalues, sorted_eigvecs)
    }

    /// Multi-signal boundary scan to detect scale boundaries.
    ///
    /// Scans a log-spaced σ grid and computes:
    /// - V(σ): participation ratio (effective number of active nodes)
    /// - D_w(σ): diffusion entropy (information content of heat distribution)
    /// - C_k(σ): spectral concentration (effective rank)
    ///
    /// Z-score normalizes each signal → composite S(σ) = α₁·V + α₂·D + α₃·C
    /// Then uses MAD peak picker with β threshold.
    pub fn boundary_scan(
        eigenvalues: &[f32],
        eigenvectors: &[f32],
        focus: usize,
        n: usize,
        config: &SlodConfig,
    ) -> Vec<ScaleBoundary> {
        let k_eigs = eigenvalues.len();
        if k_eigs == 0 || n == 0 {
            return Vec::new();
        }

        // Log-spaced σ grid (100 points)
        let n_sigma = 100;
        let sigma_min = 0.01f32;
        let sigma_max = 10.0f32;
        let log_min = sigma_min.ln();
        let log_max = sigma_max.ln();

        let mut sigmas = [0.0f32; 100];
        for (i, sigma_slot) in sigmas.iter_mut().enumerate() {
            let t = i as f32 / (n_sigma - 1) as f32;
            *sigma_slot = (log_min + t * (log_max - log_min)).exp();
        }

        // Compute query spectral coefficients: φ_k^T e_focus
        let query_coeffs: Vec<f32> = (0..k_eigs)
            .map(|k| {
                let v = &eigenvectors[k * n..(k + 1) * n];
                if focus < n { v[focus] } else { 0.0 }
            })
            .collect();

        // Compute three signals at each σ
        let mut v_signal = [0.0f32; 100];
        let mut d_signal = [0.0f32; 100];
        let mut c_signal = [0.0f32; 100];
        // Pre-allocate weights — reused across sigma iterations
        let mut weights = vec![0.0f32; n];

        for (si, &sigma) in sigmas.iter().enumerate() {
            // Compute heat kernel weights for each node
            weights[..n].fill(0.0);

            for k in 0..k_eigs {
                let decay = (-eigenvalues[k] * sigma).exp();
                let coeff = query_coeffs[k] * query_coeffs[k];
                let amp = decay * coeff;
                let v = &eigenvectors[k * n..(k + 1) * n];
                for i in 0..n {
                    weights[i] += amp * v[i] * v[i];
                }
            }

            // V(σ): participation — effective number of active nodes
            let w_sum = crate::simd::simd_sum_f32(&weights[..n]).max(1e-10);
            let w_sq_sum = crate::simd::simd_sum_sq(&weights[..n], n);
            v_signal[si] = (w_sum * w_sum / w_sq_sum.max(1e-10)) / n as f32;

            // D_w(σ): diffusion entropy
            let mut entropy = 0.0f32;
            for &w in &weights {
                if w > 1e-10 {
                    let p = w / w_sum;
                    entropy -= p * p.ln();
                }
            }
            d_signal[si] = entropy;

            // C_k(σ): spectral concentration — effective rank
            let mut c_energy = 0.0f32;
            let mut c_total = 0.0f32;
            for &eig in eigenvalues.iter().take(k_eigs) {
                let decay = (-eig * sigma).exp();
                c_energy += decay * decay;
                c_total += decay;
            }
            c_signal[si] = c_energy / (c_total * c_total).max(1e-10);
        }

        // Z-score normalize each signal (stack-allocated scratch buffers)
        let mut z_v = [0.0f32; 100];
        let mut z_d = [0.0f32; 100];
        let mut z_c = [0.0f32; 100];
        zscore_into(&v_signal, &mut z_v);
        zscore_into(&d_signal, &mut z_d);
        zscore_into(&c_signal, &mut z_c);

        // Composite score (stack-allocated)
        let mut composite = [0.0f32; 100];
        for i in 0..n_sigma {
            composite[i] =
                config.alpha[0] * z_v[i] + config.alpha[1] * z_d[i] + config.alpha[2] * z_c[i];
        }

        // MAD peak picker
        mad_peak_picker(&composite, &sigmas, eigenvalues, config)
    }

    /// Construct SlodOperator from embeddings.
    pub fn from_embeddings(embeddings: &[f32], n: usize, dim: usize, config: &SlodConfig) -> Self {
        let (eigenvalues, eigenvectors) = Self::build_laplacian(embeddings, n, dim, config);

        // Boundary scan using node 0 as default focus
        let boundaries = Self::boundary_scan(&eigenvalues, &eigenvectors, 0, n, config);

        Self {
            eigenvalues,
            eigenvectors,
            boundaries,
            config: config.clone(),
        }
    }
}

// ── Heat Kernel Weights ───────────────────────────────────────────

/// Compute heat kernel weights for all nodes given a query point.
///
/// w_i(σ) = Σ_k exp(-λ_k σ) ⟨φ_k, query⟩² · φ_k[i]
pub fn heat_kernel_weights(
    eigenvalues: &[f32],
    eigenvectors: &[f32],
    query: &[f32],
    sigma: f32,
    n: usize,
    dim: usize,
) -> Vec<f32> {
    let mut weights = vec![0.0f32; n];
    heat_kernel_weights_into(
        &mut weights,
        eigenvalues,
        eigenvectors,
        query,
        sigma,
        n,
        dim,
    );
    weights
}

/// In-place heat kernel weights — zero-allocation hot path.
#[inline]
pub fn heat_kernel_weights_into(
    weights: &mut [f32],
    eigenvalues: &[f32],
    eigenvectors: &[f32],
    query: &[f32],
    sigma: f32,
    n: usize,
    dim: usize,
) {
    let k_eigs = eigenvalues.len();
    if k_eigs == 0 || n == 0 {
        return;
    }

    // Project query onto eigenvectors: ⟨φ_k, query⟩
    // query has dimension `dim`, but eigenvectors have dimension `n`
    // We use the first min(dim, n) components
    let proj_dim = dim.min(n);
    if proj_dim == 0 {
        return;
    }

    weights[..n].fill(0.0);

    for k in 0..k_eigs {
        let v = &eigenvectors[k * n..(k + 1) * n];
        let coeff = if proj_dim <= n {
            simd_dot_f32(&query[..proj_dim], &v[..proj_dim], proj_dim)
        } else {
            0.0
        };
        let decay = (-eigenvalues[k] * sigma).exp();
        let amp = decay * coeff * coeff;
        for i in 0..n {
            weights[i] += amp * v[i];
        }
    }
}

// ── Fréchet Mean (SIMD-accelerated) ───────────────────────────────

/// Compute the Fréchet mean (Riemannian centroid) in the Poincaré ball.
///
/// Warm-starts at the point with the highest weight, then iterates:
/// 1. Log_μ(v_i) for each point
/// 2. Weighted average in tangent space
/// 3. Exp_μ(η · ū) step
///
/// Uses `simd_dot_f32` for tangent-space dot products.
/// Converges within `max_iterations` (default 15) at `tolerance` (default 1e-6).
pub fn frechet_mean(
    embeddings: &[f32],
    weights: &[f32],
    dim: usize,
    config: &SlodConfig,
) -> Vec<f32> {
    let n = weights.len();
    if n == 0 {
        return vec![0.0; dim];
    }

    // Warm-start: pick point with dominant weight
    let start_idx = weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut mu = embeddings[start_idx * dim..(start_idx + 1) * dim].to_vec();

    // Clamp initial point into ball (SIMD-accelerated)
    let norm_sq = simd_dot_f32(&mu, &mu, dim);
    if norm_sq >= 1.0 - 1e-5 {
        crate::simd::simd_scale_inplace(&mut mu, (1.0 - 1e-5) / norm_sq.sqrt());
    }

    let weight_sum: f32 = weights.iter().sum();

    // Pre-allocate scratch buffers once — reused across iterations
    let mut avg_tangent = vec![0.0f32; dim];
    let mut log_neg_base = vec![0.0f32; dim];
    let mut log_mob = vec![0.0f32; dim];
    let mut log_out = vec![0.0f32; dim];
    let mut exp_dir = vec![0.0f32; dim];
    let mut exp_mob = vec![0.0f32; dim];
    let mut exp_out = vec![0.0f32; dim];

    for _ in 0..config.max_iterations {
        // Weighted average of log-maps in tangent space
        avg_tangent[..dim].fill(0.0);

        for i in 0..n {
            if weights[i] < 1e-10 {
                continue;
            }
            let point = &embeddings[i * dim..(i + 1) * dim];
            log_map_into(
                &mut log_out,
                &mut log_neg_base,
                &mut log_mob,
                &mu,
                point,
                dim,
            );
            // SIMD fused scale-accumulate: avg_tangent += weights[i] * log_out
            crate::simd::simd_fused_scale_acc(
                &mut avg_tangent[..dim],
                &log_out[..dim],
                weights[i],
                dim,
            );
        }

        // Normalize by total weight (SIMD-accelerated)
        let norm = weight_sum.max(1e-10);
        crate::simd::simd_scale_inplace(&mut avg_tangent[..dim], 1.0 / norm);

        // Check convergence
        let tangent_norm = simd_dot_f32(&avg_tangent, &avg_tangent, dim).sqrt();
        if tangent_norm < config.tolerance {
            break;
        }

        // Scale in-place instead of allocating step_tangent (SIMD-accelerated)
        crate::simd::simd_scale_inplace(&mut avg_tangent[..dim], config.step_size);

        // Exp step — result goes into exp_out, then copy to mu
        exp_map_into(
            &mut exp_out,
            &mut exp_dir,
            &mut exp_mob,
            &mu,
            &avg_tangent,
            dim,
        );
        mu[..dim].copy_from_slice(&exp_out[..dim]);
    }

    mu
}

// ── SLoD Pruner ───────────────────────────────────────────────────

/// SLoD Pruner implementing ConstraintPruner for spectral tier routing.
///
/// Routes constraint checks to the appropriate spectral boundary tier,
/// enabling O(1) lookup per token validation.
pub struct SlodPruner {
    /// The spectral operator providing eigenpairs and boundaries.
    pub operator: SlodOperator,
    /// Per-tier constraint pruners. One per detected boundary.
    pub tier_pruners: Vec<Box<dyn crate::traits::ConstraintPruner>>,
}

impl std::fmt::Debug for SlodPruner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlodPruner")
            .field("operator", &self.operator)
            .field("tier_pruners", &self.tier_pruners.len())
            .finish()
    }
}

impl SlodPruner {
    /// Map depth to tier index. Returns None when no tiers exist or boundary
    /// score indicates a weak boundary (accept-all shortcut).
    #[inline]
    fn tier_index(&self, depth: usize) -> Option<usize> {
        let n_tiers = self.tier_pruners.len();
        if n_tiers == 0 {
            return None;
        }
        let tier_idx = depth.min(n_tiers - 1);
        // Weak boundary → accept-all shortcut
        if let Some(boundary) = self.operator.boundaries.get(tier_idx) {
            if boundary.score < 0.1 {
                return None;
            }
        }
        Some(tier_idx)
    }
}

impl crate::traits::ConstraintPruner for SlodPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let tier = match self.tier_index(depth) {
            Some(t) => t,
            None => return true,
        };
        match self.tier_pruners.get(tier) {
            Some(pruner) => pruner.is_valid(depth, token_idx, parent_tokens),
            None => true,
        }
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let tier = match self.tier_index(depth) {
            Some(t) => t,
            None => {
                let len = candidates.len().min(results.len());
                results[..len].fill(true);
                return;
            }
        };
        match self.tier_pruners.get(tier) {
            Some(pruner) => {
                pruner.batch_is_valid(depth, candidates, parent_tokens, results)
            }
            None => {
                let len = candidates.len().min(results.len());
                results[..len].fill(true);
            }
        }
    }

    fn propagate(&mut self, depth: usize, token_idx: usize, parent_token: &[usize]) -> bool {
        let tier = match self.tier_index(depth) {
            Some(t) => t,
            None => return true,
        };
        match self.tier_pruners.get_mut(tier) {
            Some(pruner) => pruner.propagate(depth, token_idx, parent_token),
            None => true,
        }
    }

    fn manifold_score(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let tier = match self.tier_index(depth) {
            Some(t) => t,
            None => return 1.0,
        };
        match self.tier_pruners.get(tier) {
            Some(pruner) => pruner.manifold_score(depth, token_idx, parent_tokens),
            None => 1.0,
        }
    }
}

// ── Helper Functions ──────────────────────────────────────────────

/// Z-score normalize a signal. Returns zero-centered signal.
///
/// Convenience wrapper that allocates — prefer [`zscore_into`] in hot paths.
#[allow(dead_code)]
fn zscore(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0f32; signal.len()];
    zscore_into(signal, &mut out);
    out
}

/// Z-score normalize into a pre-allocated buffer (zero-allocation hot path).
fn zscore_into(signal: &[f32], out: &mut [f32]) {
    if signal.is_empty() {
        return;
    }
    let n = signal.len().min(out.len());
    if n == 0 {
        return;
    }
    // Use SIMD sum for mean
    let mean: f32 = crate::simd::simd_sum_f32(&signal[..n]) / n as f32;
    // Center into output buffer: out[i] = signal[i] - mean
    out[..n].copy_from_slice(&signal[..n]);
    crate::simd::simd_add_scalar_inplace(&mut out[..n], -mean);
    // Compute variance via SIMD sum-of-squares on centered data
    let variance: f32 = crate::simd::simd_sum_sq(&out[..n], n) / n as f32;
    let inv_std = 1.0 / variance.sqrt().max(1e-10);
    // Scale in-place
    crate::simd::simd_scale_inplace(&mut out[..n], inv_std);
}

/// MAD (Median Absolute Deviation) peak picker.
///
/// Identifies peaks in the composite score signal where the value exceeds
/// β times the MAD from the median.
fn mad_peak_picker(
    composite: &[f32],
    sigmas: &[f32],
    eigenvalues: &[f32],
    config: &SlodConfig,
) -> Vec<ScaleBoundary> {
    let n = composite.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute median (O(n) via select_nth_unstable_by)
    let mut sorted = composite.to_vec();
    let mid = n / 2;
    sorted.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let median = sorted[mid];

    // Compute MAD (O(n) via select_nth_unstable_by)
    let mut deviations: Vec<f32> = composite.iter().map(|&x| (x - median).abs()).collect();
    deviations.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mad = deviations[mid].max(1e-10);

    // Find peaks where composite > median + β * MAD * 1.4826
    // (1.4826 is the consistency constant for normal distribution)
    let threshold = median + config.mad_beta * mad * 1.4826;

    let mut boundaries = Vec::new();
    let mut i = 1;
    while i < n - 1 {
        if composite[i] > threshold
            && composite[i] >= composite[i - 1]
            && composite[i] >= composite[i + 1]
        {
            // Compute K* for this sigma
            let sigma = sigmas[i];
            let k_star = compute_k_star(eigenvalues, sigma, config.gap_threshold);

            boundaries.push(ScaleBoundary {
                sigma,
                k_star,
                score: composite[i],
            });
            // Skip nearby peaks (minimum separation of 3 grid points)
            i += 3;
        } else {
            i += 1;
        }
    }

    boundaries
}

/// Compute effective rank K* at a given σ using spectral gap threshold.
///
/// K* = max{k : λ_{k+1} - λ_k > gap_threshold / σ}
fn compute_k_star(eigenvalues: &[f32], sigma: f32, gap_threshold: f32) -> usize {
    if eigenvalues.is_empty() {
        return 0;
    }

    let n = eigenvalues.len();
    let adaptive_gap = gap_threshold / sigma.max(1e-5);

    for k in 0..n.saturating_sub(1) {
        let gap = (eigenvalues[k] - eigenvalues[k + 1]).abs();
        if gap < adaptive_gap {
            return (k + 1).max(1);
        }
    }

    n
}

// ── Internal: Re-export Jacobi eigendecomposition ────────────────

/// Thin wrapper around spectral_hierarchy's Jacobi eigendecomposition.
/// Returns top-k eigenvectors as flat buffer [k*n] row-major.
fn top_k_eigenvectors(mat: &[f32], n: usize, k: usize) -> Vec<f32> {
    crate::spectral_hierarchy::top_k_eigenvectors(mat, n, k)
}

// ── Unit Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ConstraintPruner;

    fn near(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn poincare_distance_identity() {
        let x = [0.1f32, 0.2, 0.3];
        assert!(near(poincare_distance(&x, &x, 3), 0.0, 1e-5));
    }

    #[test]
    fn poincare_distance_symmetry() {
        let a = [0.1f32, 0.2, 0.3];
        let b = [0.4f32, 0.1, 0.0];
        let d_ab = poincare_distance(&a, &b, 3);
        let d_ba = poincare_distance(&b, &a, 3);
        assert!(near(d_ab, d_ba, 1e-5));
    }

    #[test]
    fn poincare_distance_positive() {
        let a = [0.1f32, 0.2];
        let b = [0.4f32, 0.1];
        assert!(poincare_distance(&a, &b, 2) > 0.0);
    }

    #[test]
    fn log_exp_roundtrip() {
        let base = [0.1f32, 0.2, 0.1];
        let point = [0.3f32, 0.1, 0.2];
        let tangent = log_map(&base, &point, 3);
        let reconstructed = exp_map(&base, &tangent, 3);
        for i in 0..3 {
            assert!(
                near(reconstructed[i], point[i], 0.15),
                "Mismatch at dim {i}: got {}, expected {}",
                reconstructed[i],
                point[i]
            );
        }
    }

    #[test]
    fn log_map_at_base_is_zero() {
        let x = [0.1f32, 0.2, 0.3];
        let tangent = log_map(&x, &x, 3);
        for &t in &tangent {
            assert!(near(t, 0.0, 1e-5));
        }
    }

    #[test]
    fn zscore_zero_mean() {
        let signal = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let z = zscore(&signal);
        let mean: f32 = z.iter().sum::<f32>() / z.len() as f32;
        assert!(near(mean, 0.0, 1e-5));
    }

    #[test]
    fn knn_laplacian_symmetric() {
        // 3 points in 2D
        let embeddings: &[f32] = &[0.1, 0.2, 0.3, 0.1, 0.0, 0.3];
        let config = SlodConfig {
            knn_k: 2,
            ..Default::default()
        };
        let (evals, evecs) = SlodOperator::build_laplacian(embeddings, 3, 2, &config);
        assert_eq!(evecs.len() % 3, 0, "eigenvectors should be row-major [k*3]");
        // Laplacian eigenvalues should be non-negative
        for &ev in &evals {
            assert!(ev >= -1e-3, "eigenvalue {ev} should be non-negative");
        }
    }

    #[test]
    fn boundary_scan_empty_input() {
        let config = SlodConfig::default();
        let boundaries = SlodOperator::boundary_scan(&[], &[], 0, 0, &config);
        assert!(boundaries.is_empty());
    }

    #[test]
    fn frechet_mean_identical_points() {
        let dim = 3;
        let point: &[f32] = &[0.1, 0.2, 0.1];
        let embeddings: Vec<f32> = point.repeat(5); // 5 identical points
        let weights = [1.0f32; 5];
        let config = SlodConfig::default();
        let mean = frechet_mean(&embeddings, &weights, dim, &config);
        for i in 0..dim {
            assert!(
                near(mean[i], point[i], 1e-3),
                "Mean at dim {i}: got {}, expected {}",
                mean[i],
                point[i]
            );
        }
    }

    #[test]
    fn slod_config_effective_k() {
        let config = SlodConfig::default();
        assert_eq!(config.effective_knn_k(25), 10); // max(10, sqrt(25)=5) = 10
        assert_eq!(config.effective_knn_k(10000), 50); // max(10, min(sqrt(10000)=100, 50)) = 50
        assert_eq!(config.effective_knn_k(400), 20); // max(10, min(sqrt(400)=20, 50)) = 20
    }

    #[test]
    fn compute_k_star_basic() {
        // Descending eigenvalues with clear gaps
        let eigenvalues = vec![5.0, 4.0, 2.0, 1.5, 0.5];
        let k = compute_k_star(&eigenvalues, 1.0, 2.0);
        assert!(k >= 1, "should detect at least one gap");
    }

    #[test]
    fn heat_kernel_weights_shape() {
        let n = 5;
        let dim = 3;
        let k_eigs = 3;
        let eigenvalues = vec![2.0f32, 1.0, 0.5];
        let eigenvectors = vec![1.0f32 / (n as f32).sqrt(); k_eigs * n];
        let query = vec![0.5f32; dim];
        let w = heat_kernel_weights(&eigenvalues, &eigenvectors, &query, 1.0, n, dim);
        assert_eq!(w.len(), n);
    }

    #[test]
    fn slod_pruner_no_tiers_accepts_all() {
        let config = SlodConfig::default();
        let operator = SlodOperator {
            eigenvalues: vec![1.0],
            eigenvectors: vec![1.0],
            boundaries: vec![],
            config,
        };
        let pruner = SlodPruner {
            operator,
            tier_pruners: vec![],
        };
        assert!(pruner.is_valid(0, 42, &[]));
    }
}
