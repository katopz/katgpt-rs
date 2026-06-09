//! Kernel scoring functions for ScreeningPruner relevance lifting (Plan 234).

/// Kernel function kind for relevance computation.
#[derive(Debug, Clone, Copy)]
pub enum KernelKind {
    /// Linear: dot(query, candidate)
    Linear,
    /// Gaussian RBF: exp(-||q-c||^2 / sigma^2)
    Gaussian { sigma: f32 },
    /// Polynomial: (dot(q,c) + c)^degree
    Polynomial { degree: f32, c: f32 },
}

/// Compute kernel similarity between two vectors.
pub fn kernel_score(query: &[f32], candidate: &[f32], kind: KernelKind) -> f32 {
    match kind {
        KernelKind::Linear => {
            let mut sum = 0.0f32;
            let len = query.len().min(candidate.len());
            for i in 0..len {
                sum += query[i] * candidate[i];
            }
            sum
        }
        KernelKind::Gaussian { sigma } => {
            let sigma_sq = sigma * sigma;
            let mut dist_sq = 0.0f32;
            let len = query.len().min(candidate.len());
            for i in 0..len {
                let d = query[i] - candidate[i];
                dist_sq += d * d;
            }
            (-dist_sq / sigma_sq).exp()
        }
        KernelKind::Polynomial { degree, c } => {
            let mut dot = 0.0f32;
            let len = query.len().min(candidate.len());
            for i in 0..len {
                dot += query[i] * candidate[i];
            }
            (dot + c).powf(degree)
        }
    }
}

/// SIMD-accelerated Gaussian kernel (chunked f32, 4 elements per iteration).
pub fn kernel_score_simd_gaussian(query: &[f32], candidate: &[f32], sigma: f32) -> f32 {
    let sigma_sq = sigma * sigma;
    let mut dist_sq = 0.0f32;
    let len = query.len().min(candidate.len());
    let chunks = len / 4;
    let remainder = len % 4;

    // Process 4 elements at a time (helps LLVM auto-vectorize)
    for i in 0..chunks {
        let base = i * 4;
        for j in 0..4 {
            let d = query[base + j] - candidate[base + j];
            dist_sq += d * d;
        }
    }

    // Handle remainder
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        let d = query[i] - candidate[i];
        dist_sq += d * d;
    }

    (-dist_sq / sigma_sq).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_kernel_dot_product() {
        let q = [1.0, 2.0, 3.0];
        let c = [4.0, 5.0, 6.0];
        let result = kernel_score(&q, &c, KernelKind::Linear);
        assert!((result - 32.0).abs() < 1e-5); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn gaussian_kernel_identical_vectors() {
        let v = [1.0, 2.0, 3.0];
        let result = kernel_score(&v, &v, KernelKind::Gaussian { sigma: 1.0 });
        assert!((result - 1.0).abs() < 1e-5); // exp(0) = 1.0
    }

    #[test]
    fn gaussian_kernel_distant_vectors() {
        let q = [0.0, 0.0, 0.0];
        let c = [10.0, 10.0, 10.0];
        let result = kernel_score(&q, &c, KernelKind::Gaussian { sigma: 1.0 });
        assert!(
            result < 0.01,
            "distant vectors should have low similarity, got {}",
            result
        );
    }

    #[test]
    fn polynomial_kernel_basic() {
        let q = [1.0, 0.0];
        let c = [1.0, 0.0];
        let result = kernel_score(
            &q,
            &c,
            KernelKind::Polynomial {
                degree: 2.0,
                c: 1.0,
            },
        );
        assert!((result - 4.0).abs() < 1e-5); // (1 + 1)^2 = 4
    }

    #[test]
    fn simd_gaussian_matches_scalar() {
        let q: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let c: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let scalar = kernel_score(&q, &c, KernelKind::Gaussian { sigma: 1.0 });
        let simd = kernel_score_simd_gaussian(&q, &c, 1.0);
        assert!(
            (scalar - simd).abs() < 1e-5,
            "scalar {} != simd {}",
            scalar,
            simd
        );
    }
}
