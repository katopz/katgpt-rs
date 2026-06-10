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
    /// Composite score weights: [participation, diffusion_entropy, spectral_concentration].
    /// Default: [1/3, 1/3, 1/3].
    pub alpha: [f32; 3],
    /// MAD (Median Absolute Deviation) peak picker threshold.
    /// Default: 2.0.
    pub mad_beta: f32,
    /// Spectral gap threshold for K* selection.
    /// Default: 2.0.
    pub gap_threshold: f32,
    /// Max Fréchet mean iterations.
    /// Default: 15.
    pub max_iterations: usize,
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
            alpha: [1.0 / 3.0; 3],
            mad_beta: 2.0,
            gap_threshold: 2.0,
            max_iterations: 15,
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
    /// Diffusion scale σ at which this boundary was detected.
    pub sigma: f32,
    /// Effective rank K* (number of significant eigenmodes) at this scale.
    pub k_star: usize,
    /// Composite boundary score S(σ).
    pub score: f32,
}

// ── SLoD Operator ─────────────────────────────────────────────────

/// Core spectral operator: eigenpairs + detected boundaries.
#[derive(Debug, Clone)]
pub struct SlodOperator {
    /// Eigenvalues λ_k (descending order).
    pub eigenvalues: Vec<f32>,
    /// Eigenvectors as flat buffer [K_eigs * N], row-major.
    pub eigenvectors: Vec<f32>,
    /// Detected scale boundaries.
    pub boundaries: Vec<ScaleBoundary>,
    /// Configuration used to build this operator.
    pub config: SlodConfig,
}

// ── Poincaré Ball Geometry ────────────────────────────────────────

// clamp_norm removed — was unused

/// Compute squared Euclidean distance.
#[inline]
fn sq_dist(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let mut s = 0.0f32;
    for i in 0..dim {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

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

    let diff_sq = sq_dist(a, b, dim);
    let denom = (1.0 - norm_a_sq) * (1.0 - norm_b_sq);

    let inner = 1.0 + 2.0 * diff_sq / denom;
    inner.acosh()
}

/// Möbius addition in the Poincaré ball: a ⊕ b.
///
/// a ⊕ b = ((1 + 2<a,b> + ||b||²)a + (1 - ||a||²)b) / (1 + 2<a,b> + ||a||²||b||²)
fn mobius_add(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let norm_a_sq = simd_dot_f32(a, a, dim).min(1.0 - 1e-5);
    let norm_b_sq = simd_dot_f32(b, b, dim).min(1.0 - 1e-5);
    let a_dot_b = simd_dot_f32(a, b, dim);

    let denom = 1.0 + 2.0 * a_dot_b + norm_a_sq * norm_b_sq;
    let inv_denom = 1.0 / denom.max(1e-10);

    let s1 = (1.0 + 2.0 * a_dot_b + norm_b_sq) * inv_denom;
    let s2 = (1.0 - norm_a_sq) * inv_denom;

    let mut result = vec![0.0f32; dim];
    for i in 0..dim {
        result[i] = s1 * a[i] + s2 * b[i];
    }
    result
}

/// Riemannian log map: project `point` onto the tangent space at `base`.
///
/// Returns a tangent vector in T_base B^n.
/// log_x(y) = (d(x,y) / ||(-x) ⊕ y||) · ((-x) ⊕ y)
pub fn log_map(base: &[f32], point: &[f32], dim: usize) -> Vec<f32> {
    let dist = poincare_distance(base, point, dim);
    if dist < 1e-10 {
        return vec![0.0; dim];
    }

    // Möbius addition: (-base) ⊕ point
    let neg_base: Vec<f32> = base.iter().map(|&x| -x).collect();
    let mob = mobius_add(&neg_base, point, dim);
    let mob_norm = simd_dot_f32(&mob, &mob, dim).sqrt();

    if mob_norm < 1e-10 {
        return vec![0.0; dim];
    }

    let scale = dist / mob_norm;
    mob.iter().map(|&v| v * scale).collect()
}

/// Riemannian exp map: project tangent vector back to the Poincaré ball.
///
/// exp_x(v) = x ⊕ tanh(||v||/2) / ||v|| · v
/// where the tangent vector v encodes the conformal factor from log_map.
pub fn exp_map(base: &[f32], tangent: &[f32], dim: usize) -> Vec<f32> {
    let _norm_base_sq = simd_dot_f32(base, base, dim).min(1.0 - 1e-5);
    let tangent_norm = simd_dot_f32(tangent, tangent, dim).sqrt();

    if tangent_norm < 1e-10 {
        return base.to_vec();
    }

    // Compute direction: tanh(||v||/2) * v/||v||
    let s = (tangent_norm / 2.0).tanh() / tangent_norm;
    let mut dir = vec![0.0f32; dim];
    for i in 0..dim {
        dir[i] = s * tangent[i];
    }

    // Project dir into ball
    let dir_norm_sq = simd_dot_f32(&dir, &dir, dim);
    if dir_norm_sq >= 1.0 - 1e-5 {
        let scale = (1.0 - 1e-5) / dir_norm_sq.sqrt();
        dir.iter_mut().for_each(|v| *v *= scale);
    }

    // Möbius addition: base ⊕ dir
    let result = mobius_add(base, &dir, dim);

    // Final clamp into ball
    let norm_r_sq = simd_dot_f32(&result, &result, dim);
    if norm_r_sq >= 1.0 - 1e-5 {
        let scale = (1.0 - 1e-5) / norm_r_sq.sqrt();
        result.iter().map(|v| v * scale).collect()
    } else {
        result
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

        for i in 0..n {
            let a_i = &embeddings[i * dim..(i + 1) * dim];
            // Compute distances to all other points
            let mut dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let a_j = &embeddings[j * dim..(j + 1) * dim];
                    (j, poincare_distance(a_i, a_j, dim))
                })
                .collect();
            dists.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Take top-k nearest neighbors
            for &(j, d) in dists.iter().take(k) {
                let weight = (-d).exp();
                w[i * n + j] = weight;
                w[j * n + i] = weight; // symmetrize
            }
        }

        // Degree vector
        let mut degree = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                degree[i] += w[i * n + j];
            }
            degree[i] = degree[i].max(1e-10);
        }

        // Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
        let mut lap = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                if i == j {
                    lap[idx] = 1.0 - w[idx] / (degree[i] * degree[j]).sqrt();
                } else {
                    lap[idx] = -w[idx] / (degree[i] * degree[j]).sqrt();
                }
            }
        }

        // Eigendecompose — reuse Jacobi from spectral_hierarchy
        let k_eigs = k.min(n);
        let eigvecs = top_k_eigenvectors(&lap, n, k_eigs);

        // Extract eigenvalues from diagonal of rotated matrix
        // top_k_eigenvectors returns [k_eigs * n] row-major eigenvectors
        // We need eigenvalues too — compute from Lap * v = λ * v
        let mut eigenvalues = Vec::with_capacity(k_eigs);
        for eig_idx in 0..k_eigs {
            let v = &eigvecs[eig_idx * n..(eig_idx + 1) * n];
            // Compute Lv
            let mut lv = vec![0.0f32; n];
            for i in 0..n {
                for j in 0..n {
                    lv[i] += lap[i * n + j] * v[j];
                }
            }
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

        let sigmas: Vec<f32> = (0..n_sigma)
            .map(|i| {
                let t = i as f32 / (n_sigma - 1) as f32;
                (log_min + t * (log_max - log_min)).exp()
            })
            .collect();

        // Compute query spectral coefficients: φ_k^T e_focus
        let query_coeffs: Vec<f32> = (0..k_eigs)
            .map(|k| {
                let v = &eigenvectors[k * n..(k + 1) * n];
                if focus < n { v[focus] } else { 0.0 }
            })
            .collect();

        // Compute three signals at each σ
        let mut v_signal = vec![0.0f32; n_sigma];
        let mut d_signal = vec![0.0f32; n_sigma];
        let mut c_signal = vec![0.0f32; n_sigma];

        for (si, &sigma) in sigmas.iter().enumerate() {
            // Compute heat kernel weights for each node
            let mut weights = vec![0.0f32; n];
            let mut _total_energy = 0.0f32;

            for k in 0..k_eigs {
                let decay = (-eigenvalues[k] * sigma).exp();
                let coeff = query_coeffs[k] * query_coeffs[k];
                let amp = decay * coeff;
                let v = &eigenvectors[k * n..(k + 1) * n];
                for i in 0..n {
                    weights[i] += amp * v[i] * v[i];
                }
                _total_energy += amp;
            }

            // V(σ): participation — effective number of active nodes
            let w_sum: f32 = weights.iter().copied().sum::<f32>().max(1e-10);
            let w_sq_sum: f32 = weights.iter().map(|w| w * w).sum();
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
            for &eig in eigenvalues.iter().take(k_eigs) {
                let decay = (-eig * sigma).exp();
                c_energy += decay * decay;
            }
            let c_total: f32 = eigenvalues
                .iter()
                .take(k_eigs)
                .map(|&eig| (-eig * sigma).exp())
                .sum::<f32>()
                .max(1e-10);
            c_signal[si] = c_energy / (c_total * c_total);
        }

        // Z-score normalize each signal
        let z_v = zscore(&v_signal);
        let z_d = zscore(&d_signal);
        let z_c = zscore(&c_signal);

        // Composite score
        let composite: Vec<f32> = (0..n_sigma)
            .map(|i| config.alpha[0] * z_v[i] + config.alpha[1] * z_d[i] + config.alpha[2] * z_c[i])
            .collect();

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
    let k_eigs = eigenvalues.len();
    if k_eigs == 0 || n == 0 {
        return vec![0.0; n];
    }

    // Project query onto eigenvectors: ⟨φ_k, query⟩
    // query has dimension `dim`, but eigenvectors have dimension `n`
    // We use the first min(dim, n) components
    let proj_dim = dim.min(n);
    let mut query_coeffs = vec![0.0f32; k_eigs];
    for k in 0..k_eigs {
        let v = &eigenvectors[k * n..(k + 1) * n];
        if proj_dim > 0 && proj_dim <= n {
            query_coeffs[k] = simd_dot_f32(&query[..proj_dim], &v[..proj_dim], proj_dim);
        }
    }

    // Accumulate weights
    let mut weights = vec![0.0f32; n];
    for k in 0..k_eigs {
        let decay = (-eigenvalues[k] * sigma).exp();
        let coeff_sq = query_coeffs[k] * query_coeffs[k];
        let amp = decay * coeff_sq;
        let v = &eigenvectors[k * n..(k + 1) * n];
        for i in 0..n {
            weights[i] += amp * v[i];
        }
    }

    weights
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

    // Clamp initial point into ball
    let norm_sq = simd_dot_f32(&mu, &mu, dim);
    if norm_sq >= 1.0 - 1e-5 {
        let scale = (1.0 - 1e-5) / norm_sq.sqrt();
        mu.iter_mut().for_each(|v| *v *= scale);
    }

    let weight_sum: f32 = weights.iter().sum();

    for _ in 0..config.max_iterations {
        // Weighted average of log-maps in tangent space
        let mut avg_tangent = vec![0.0f32; dim];

        for i in 0..n {
            if weights[i] < 1e-10 {
                continue;
            }
            let point = &embeddings[i * dim..(i + 1) * dim];
            let tangent = log_map(&mu, point, dim);
            for d in 0..dim {
                avg_tangent[d] += weights[i] * tangent[d];
            }
        }

        // Normalize by total weight
        let norm = weight_sum.max(1e-10);
        for v in avg_tangent.iter_mut() {
            *v /= norm;
        }

        // Check convergence
        let tangent_norm = simd_dot_f32(&avg_tangent, &avg_tangent, dim).sqrt();
        if tangent_norm < config.tolerance {
            break;
        }

        // Exp step
        let step_tangent: Vec<f32> = avg_tangent.iter().map(|&v| v * config.step_size).collect();
        mu = exp_map(&mu, &step_tangent, dim);
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

impl crate::traits::ConstraintPruner for SlodPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Route to appropriate tier based on depth
        let tier = match self.tier_pruners.len() {
            0 => return true, // no tiers → accept all
            n_tiers => {
                // Map depth to tier via spectral boundaries
                let tier_idx = depth.min(n_tiers - 1);
                // Use boundary K* to modulate: deeper → more permissive
                match self.operator.boundaries.get(tier_idx) {
                    Some(boundary) if boundary.score < 0.1 => return true, // weak boundary → accept
                    _ => tier_idx,
                }
            }
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
        match self.tier_pruners.len() {
            0 => {
                let len = candidates.len().min(results.len());
                results[..len].fill(true);
            }
            _ => {
                // Delegate to the routed tier pruner's batch method
                let tier = depth.min(self.tier_pruners.len() - 1);
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
        }
    }

    fn propagate(&mut self, depth: usize, token_idx: usize, parent_token: &[usize]) -> bool {
        match self.tier_pruners.len() {
            0 => true,
            _ => {
                let tier = depth.min(self.tier_pruners.len() - 1);
                match self.tier_pruners.get_mut(tier) {
                    Some(pruner) => pruner.propagate(depth, token_idx, parent_token),
                    None => true,
                }
            }
        }
    }

    fn manifold_score(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        match self.tier_pruners.len() {
            0 => 1.0,
            _ => {
                let tier = depth.min(self.tier_pruners.len() - 1);
                match self.tier_pruners.get(tier) {
                    Some(pruner) => pruner.manifold_score(depth, token_idx, parent_tokens),
                    None => 1.0,
                }
            }
        }
    }
}

// ── Helper Functions ──────────────────────────────────────────────

/// Z-score normalize a signal. Returns zero-centered signal.
fn zscore(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }
    let mean: f32 = signal.iter().sum::<f32>() / signal.len() as f32;
    let variance: f32 =
        signal.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / signal.len() as f32;
    let std = variance.sqrt().max(1e-10);
    signal.iter().map(|&x| (x - mean) / std).collect()
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

    // Compute median
    let mut sorted = composite.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[n / 2];

    // Compute MAD
    let deviations: Vec<f32> = composite.iter().map(|&x| (x - median).abs()).collect();
    let mut sorted_dev = deviations;
    sorted_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = sorted_dev[n / 2].max(1e-10);

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
