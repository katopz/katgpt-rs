//! CountSketch dimensionality reduction — O(d) → O(m) projection.
//!
//! Implements the CountSketch data structure from Cormode & Muthukrishnan (2005).
//! Provides an unbiased estimator for inner products in reduced dimensionality.
//!
//! # Properties
//!
//! - **Unbiased**: E[⟨sketch(a), sketch(b)⟩] = ⟨a, b⟩
//! - **Low variance**: Var ≈ O(2/m) · ‖a‖² · ‖b‖²
//! - **O(d)** projection time per vector

use crate::types::Rng;

// ── CountSketch ─────────────────────────────────────────────────

/// CountSketch projection with pre-computed hash/sign pairs.
///
/// Each input dimension `i` is mapped to bucket `h(i)` with sign `s(i) ∈ {-1, +1}`.
/// The sketch of vector `v` is:
/// ```text
/// sketch(v)[j] = Σ_{i : h(i) = j} s(i) · v[i]
/// ```
///
/// Inner product estimation: `⟨a, b⟩ ≈ ⟨sketch(a), sketch(b)⟩` (unbiased).
#[derive(Clone, Debug)]
pub struct CountSketch {
    /// Number of sketch dimensions (buckets).
    sketch_dim: usize,
    /// Hash function: maps input dim → bucket index [0, sketch_dim).
    hash_indices: Vec<usize>,
    /// Sign function: maps input dim → ±1.
    signs: Vec<f32>,
}

impl CountSketch {
    /// Create a new CountSketch with `sketch_dim` buckets for `input_dim`-dimensional vectors.
    ///
    /// Hash/sign pairs are deterministic given the same seed.
    pub fn new(input_dim: usize, sketch_dim: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let hash_indices: Vec<usize> = (0..input_dim)
            .map(|_| (rng.uniform() * sketch_dim as f32) as usize)
            .collect();
        let signs: Vec<f32> = (0..input_dim)
            .map(|_| if rng.uniform() < 0.5 { -1.0f32 } else { 1.0f32 })
            .collect();
        Self {
            sketch_dim,
            hash_indices,
            signs,
        }
    }

    /// Project a vector from O(d) to O(m).
    ///
    /// Time: O(input_dim). Each element is added (with random sign) to one bucket.
    pub fn sketch(&self, vec: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0f32; self.sketch_dim];
        for (i, &val) in vec.iter().enumerate() {
            if i >= self.hash_indices.len() {
                break;
            }
            let bucket = self.hash_indices[i];
            result[bucket] += self.signs[i] * val;
        }
        result
    }

    /// Unbiased inner product estimate via sketch dot product.
    ///
    /// E[estimate] = ⟨a, b⟩. Variance ≈ O(2/m) · ‖a‖² · ‖b‖².
    pub fn inner_product_estimate(&self, a: &[f32], b: &[f32]) -> f32 {
        // Sketch both vectors into a single allocation to avoid two heap allocs
        let mut buf = vec![0.0f32; self.sketch_dim * 2];
        let (sa, sb) = buf.split_at_mut(self.sketch_dim);
        self.sketch_into(a, sa);
        self.sketch_into(b, sb);
        dot(sa, sb)
    }

    /// Project a vector from O(d) to O(m), writing into a caller-provided buffer.
    ///
    /// Time: O(input_dim). Each element is added (with random sign) to one bucket.
    fn sketch_into(&self, vec: &[f32], out: &mut [f32]) {
        out.fill(0.0f32);
        for (i, &val) in vec.iter().enumerate() {
            if i >= self.hash_indices.len() {
                break;
            }
            let bucket = self.hash_indices[i];
            out[bucket] += self.signs[i] * val;
        }
    }

    /// Number of sketch dimensions.
    pub fn sketch_dim(&self) -> usize {
        self.sketch_dim
    }

    /// Input dimension this sketch was built for.
    pub fn input_dim(&self) -> usize {
        self.hash_indices.len()
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Dot product of two f32 slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Compute exact inner product ⟨a, b⟩.
pub fn exact_inner_product(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

/// Compute ‖v‖² (squared L2 norm).
pub fn squared_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const INPUT_DIM: usize = 64;
    const SKETCH_DIM: usize = 256;
    const SEED: u64 = 42;

    fn random_vec(dim: usize, rng: &mut Rng) -> Vec<f32> {
        (0..dim).map(|_| rng.uniform() * 2.0 - 1.0).collect()
    }

    #[test]
    fn test_sketch_output_dimension() {
        let cs = CountSketch::new(INPUT_DIM, SKETCH_DIM, SEED);
        let v = vec![1.0f32; INPUT_DIM];
        let s = cs.sketch(&v);
        assert_eq!(s.len(), SKETCH_DIM, "sketch must have sketch_dim elements");
    }

    #[test]
    fn test_sketch_deterministic() {
        let cs = CountSketch::new(INPUT_DIM, SKETCH_DIM, SEED);
        let mut rng = Rng::new(99);
        let v = random_vec(INPUT_DIM, &mut rng);
        let s1 = cs.sketch(&v);
        let s2 = cs.sketch(&v);
        assert_eq!(s1, s2, "same sketcher + same input = same output");
    }

    #[test]
    fn test_inner_product_unbiased() {
        // Average of many estimates should converge to true inner product.
        let mut rng = Rng::new(123);
        let a = random_vec(INPUT_DIM, &mut rng);
        let b = random_vec(INPUT_DIM, &mut rng);
        let true_ip = exact_inner_product(&a, &b);

        let n_trials = 10_000u64;
        let mut sum_estimates = 0.0f32;
        for seed in 0..n_trials {
            let cs = CountSketch::new(INPUT_DIM, SKETCH_DIM, seed);
            let est = cs.inner_product_estimate(&a, &b);
            sum_estimates += est;
        }
        let avg_estimate = sum_estimates / n_trials as f32;

        // Unbiased: |E[estimate] - true| should be small
        let bias = (avg_estimate - true_ip).abs();
        assert!(
            bias < 0.05,
            "unbiased estimate: avg={avg_estimate:.4}, true={true_ip:.4}, bias={bias:.4}"
        );
    }

    #[test]
    fn test_inner_product_variance_bounded() {
        // Variance should be ≈ O(2/m) · ‖a‖² · ‖b‖².
        let mut rng = Rng::new(456);
        let a = random_vec(INPUT_DIM, &mut rng);
        let b = random_vec(INPUT_DIM, &mut rng);
        let norm_a_sq = squared_norm(&a);
        let norm_b_sq = squared_norm(&b);

        let n_trials = 5_000u64;
        let mut estimates = Vec::with_capacity(n_trials as usize);
        for seed in 0..n_trials {
            let cs = CountSketch::new(INPUT_DIM, SKETCH_DIM, seed);
            estimates.push(cs.inner_product_estimate(&a, &b));
        }

        // Compute empirical variance
        let mean = estimates.iter().sum::<f32>() / n_trials as f32;
        let variance =
            estimates.iter().map(|&e| (e - mean).powi(2)).sum::<f32>() / (n_trials - 1) as f32;

        // Theoretical bound: Var ≤ (2/m) * ‖a‖² * ‖b‖² (with some slack)
        let theoretical_bound = (2.0 / SKETCH_DIM as f32) * norm_a_sq * norm_b_sq;
        let slack = 3.0; // Allow 3× theoretical due to finite samples
        assert!(
            variance < theoretical_bound * slack,
            "variance={variance:.4} should be < {theoretical_bound:.4} × {slack} (slack), \
             ‖a‖²={norm_a_sq:.2}, ‖b‖²={norm_b_sq:.2}"
        );
    }

    #[test]
    fn test_zero_vector_sketch_is_zero() {
        let cs = CountSketch::new(INPUT_DIM, SKETCH_DIM, SEED);
        let v = vec![0.0f32; INPUT_DIM];
        let s = cs.sketch(&v);
        assert!(
            s.iter().all(|&x| x == 0.0),
            "zero vector must produce zero sketch"
        );
    }

    #[test]
    fn test_unit_vector_preserves_norm_in_expectation() {
        // Sketch of a single unit vector e_i should preserve ‖e_i‖² = 1 in expectation.
        let cs = CountSketch::new(INPUT_DIM, SKETCH_DIM, SEED);
        let mut v = vec![0.0f32; INPUT_DIM];
        v[7] = 1.0; // unit vector at index 7
        let s = cs.sketch(&v);
        // Exactly one bucket has ±1, rest are 0
        let nonzero_count = s.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(nonzero_count, 1, "unit vector hits exactly one bucket");
        let norm_sq = squared_norm(&s);
        assert!(
            (norm_sq - 1.0).abs() < 1e-6,
            "unit vector sketch norm²={norm_sq:.4}, expected 1.0"
        );
    }

    #[test]
    fn test_different_seeds_produce_different_sketches() {
        let cs1 = CountSketch::new(INPUT_DIM, SKETCH_DIM, 1);
        let cs2 = CountSketch::new(INPUT_DIM, SKETCH_DIM, 2);
        let mut rng = Rng::new(99);
        let v = random_vec(INPUT_DIM, &mut rng);
        let s1 = cs1.sketch(&v);
        let s2 = cs2.sketch(&v);
        assert_ne!(s1, s2, "different seeds should produce different sketches");
    }

    #[test]
    fn test_linearity() {
        // sketch(a + b) = sketch(a) + sketch(b) (linearity of CountSketch)
        let cs = CountSketch::new(INPUT_DIM, SKETCH_DIM, SEED);
        let mut rng = Rng::new(789);
        let a = random_vec(INPUT_DIM, &mut rng);
        let b = random_vec(INPUT_DIM, &mut rng);
        let mut sum = vec![0.0f32; INPUT_DIM];
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            sum[i] = va + vb;
        }
        let sa = cs.sketch(&a);
        let sb = cs.sketch(&b);
        let ssum = cs.sketch(&sum);

        for j in 0..SKETCH_DIM {
            let expected = sa[j] + sb[j];
            let actual = ssum[j];
            assert!(
                (actual - expected).abs() < 1e-5,
                "linearity violated at bucket {j}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn test_accessors() {
        let cs = CountSketch::new(INPUT_DIM, SKETCH_DIM, SEED);
        assert_eq!(cs.sketch_dim(), SKETCH_DIM);
        assert_eq!(cs.input_dim(), INPUT_DIM);
    }
}
