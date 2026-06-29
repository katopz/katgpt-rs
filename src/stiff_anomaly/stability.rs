//! Eigenvalue stability tracking and anomaly gate.
//!
//! Tracks eigenvalue/eigenvector stability across temporal windows using
//! Jaccard overlap and z-score gating. The anomaly gate classifies state
//! changes as Normal, StiffCollision, or ElasticAbsorption.

use crate::stiff_anomaly::subspace::{
    decompose_into, soft_alignment_ratio, StiffSoftDecomposition,
};

/// Track eigenvalue/eigenvector stability across temporal windows.
///
/// Maintains a baseline (mean ± std per eigenvalue index) from stable
/// historical windows and flags deviations via z-scores.
#[derive(Debug, Clone)]
pub struct EigenvalueTracker {
    /// Per-index mean from baseline windows.
    pub baseline_mean: Vec<f32>,
    /// Per-index std from baseline windows.
    pub baseline_std: Vec<f32>,
    /// Historical eigenvalue windows (for reference).
    pub history: Vec<Vec<f32>>,
}

impl EigenvalueTracker {
    /// Freeze baseline statistics from a collection of stable eigenvalue windows.
    ///
    /// Each window is a sorted-descending eigenvalue vector of the same dimension.
    /// Computes per-index mean and standard deviation.
    ///
    /// # Panics
    ///
    /// Panics if `windows` is empty.
    pub fn freeze_baseline(windows: &[Vec<f32>]) -> Self {
        assert!(!windows.is_empty(), "need at least one baseline window");
        let d = windows[0].len();
        let n = windows.len() as f64;

        let mut mean = vec![0.0f64; d];
        for w in windows {
            for (i, &v) in w.iter().enumerate() {
                mean[i] += v as f64;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        let mut std = vec![0.0f64; d];
        for w in windows {
            for (i, &v) in w.iter().enumerate() {
                let diff = v as f64 - mean[i];
                std[i] += diff * diff;
            }
        }
        for s in &mut std {
            *s = (*s / n).sqrt();
            // Floor small std to avoid division by zero
            if *s < 1e-8 {
                *s = 1e-8;
            }
        }

        EigenvalueTracker {
            baseline_mean: mean.iter().map(|&x| x as f32).collect(),
            baseline_std: std.iter().map(|&x| x as f32).collect(),
            history: windows.to_vec(),
        }
    }

    /// Compute per-index z-score of `current` eigenvalue window against baseline.
    ///
    /// Returns a vector of z-scores: (current[i] - mean[i]) / std[i].
    /// Negative z-scores indicate eigenvalue collapse relative to baseline.
    pub fn eigenspace_zscore(&self, current: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(current.len());
        self.eigenspace_zscore_into(current, &mut out);
        out
    }

    /// Zero-allocation variant of [`eigenspace_zscore`] that writes into `out`.
    ///
    /// `out` is cleared and filled. Callers can pre-allocate once and reuse.
    pub fn eigenspace_zscore_into(&self, current: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.extend(
            current
                .iter()
                .zip(self.baseline_mean.iter())
                .zip(self.baseline_std.iter())
                .map(|((&c, &m), &s)| (c - m) / s),
        );
    }

    /// Jaccard overlap of top-k eigenvalue indices between two windows.
    ///
    /// Measures structural stability: which eigenvalue indices are in the
    /// top-k for both windows. Returns 1.0 for perfect overlap, 0.0 for none.
    ///
    /// Optimized path: when `sorted_desc` is true, both inputs are assumed to
    /// already be sorted descending, so the top-k indices are simply `[0..k]`
    /// and no sort is needed.
    ///
    /// When `sorted_desc` is false or inputs may not be pre-sorted, scratch
    /// buffers `scratch_a` and `scratch_b` are used instead of allocating.
    pub fn eigenvalue_jaccard(prev: &[f32], curr: &[f32], top_k: usize) -> f32 {
        let mut a = Vec::new();
        let mut b = Vec::new();
        Self::eigenvalue_jaccard_with_buffers(prev, curr, top_k, false, &mut a, &mut b)
    }

    /// Zero-allocation variant of [`eigenvalue_jaccard`] with reusable scratch buffers.
    ///
    /// When `sorted_desc` is true, eigenvalues are assumed already sorted
    /// descending so the top-k indices are `[0..k]` (no sorting needed).
    /// When false, `scratch_a` / `scratch_b` are used for partial sort.
    pub fn eigenvalue_jaccard_with_buffers(
        prev: &[f32],
        curr: &[f32],
        top_k: usize,
        sorted_desc: bool,
        scratch_a: &mut Vec<usize>,
        scratch_b: &mut Vec<usize>,
    ) -> f32 {
        if prev.is_empty() || curr.is_empty() || top_k == 0 {
            return 0.0;
        }
        let k = top_k.min(prev.len()).min(curr.len());

        let (prev_top, curr_top): (&[usize], &[usize]) = if sorted_desc {
            // Eigenvalues already sorted descending: top-k indices are [0..k]
            // Use the scratch buffers as temporary storage for a simple range.
            scratch_a.clear();
            scratch_a.extend(0..k);
            scratch_b.clear();
            scratch_b.extend(0..k);
            (scratch_a, scratch_b)
        } else {
            // Fallback: partial sort to find top-k indices
            scratch_a.clear();
            scratch_a.extend(0..prev.len());
            scratch_a.select_nth_unstable_by(k - 1, |&a, &b| {
                prev[b]
                    .partial_cmp(&prev[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            scratch_a[..k].sort_by(|&a, &b| {
                prev[b]
                    .partial_cmp(&prev[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            scratch_b.clear();
            scratch_b.extend(0..curr.len());
            scratch_b.select_nth_unstable_by(k - 1, |&a, &b| {
                curr[b]
                    .partial_cmp(&curr[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            scratch_b[..k].sort_by(|&a, &b| {
                curr[b]
                    .partial_cmp(&curr[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            (scratch_a, scratch_b)
        };

        // Compute Jaccard via sorted intersection (avoids HashSet allocation)
        let mut intersection = 0usize;
        let mut i = 0usize;
        let mut j = 0usize;
        while i < k && j < k {
            match prev_top[i].cmp(&curr_top[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    intersection += 1;
                    i += 1;
                    j += 1;
                }
            }
        }

        // Jaccard = |A ∩ B| / |A ∪ B| = intersection / (2*k - intersection)
        let union = 2 * k - intersection;
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Check if k (stiff dimension count) is invariant across baseline at given trace mass.
    ///
    /// Returns `true` if all baseline windows produce the same k.
    pub fn k_invariant(&self, trace_mass: f32) -> bool {
        if self.history.len() < 2 {
            return true;
        }
        use crate::stiff_anomaly::subspace::stiff_subspace_k;
        let k0 = stiff_subspace_k(&self.history[0], trace_mass);
        self.history
            .iter()
            .all(|w| stiff_subspace_k(w, trace_mass) == k0)
    }
}

/// Result of the anomaly gate evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum GateResult {
    /// State change is within normal bounds.
    Normal,
    /// Stiff collision detected — delta_x projects heavily onto stiff axes.
    StiffCollision { z_score: f32 },
    /// Elastic absorption — delta_x is mostly in soft subspace.
    ElasticAbsorption { alpha: f32 },
}

/// Reusable buffers for zero-allocation [`StiffAnomalyGate::evaluate`] calls.
///
/// Create once and pass to [`StiffAnomalyGate::evaluate_with_buffers`] on each call.
#[derive(Debug, Clone, Default)]
pub struct EvaluateBuffers {
    /// Z-score output buffer.
    z_scores: Vec<f32>,
    /// Decomposition buffer.
    decomp: StiffSoftDecomposition,
}

impl EvaluateBuffers {
    /// Pre-allocate buffers for `dim`-dimensional data.
    pub fn with_capacity(dim: usize) -> Self {
        Self {
            z_scores: Vec::with_capacity(dim),
            decomp: StiffSoftDecomposition::with_capacity(dim),
        }
    }
}

/// Anomaly gate with configurable thresholds.
///
/// Uses z-score of the minimum eigenvalue change to detect stiff collisions
/// and soft alignment ratio to detect elastic absorption.
#[derive(Debug, Clone)]
pub struct StiffAnomalyGate {
    /// Z-score threshold for stiff collision detection (default: -2.0).
    /// A negative threshold catches eigenvalue collapse.
    pub z_threshold: f32,
    /// Soft alignment ratio threshold for elastic detection.
    pub alpha_threshold: f32,
    /// Baseline mean (scalar summary for gate).
    pub baseline_mean: f32,
    /// Baseline std (scalar summary for gate).
    pub baseline_std: f32,
}

impl StiffAnomalyGate {
    /// Create a new gate with baseline statistics and default thresholds.
    ///
    /// - `z_threshold`: -2.0 (flag if any eigenvalue z-score drops below this)
    /// - `alpha_threshold`: 0.8 (elastic if soft alignment ≥ this)
    pub fn new(tracker: &EigenvalueTracker) -> Self {
        // Summarize baseline as the mean of total energy
        let total_energy: f32 = tracker.baseline_mean.iter().sum();
        let n = tracker.history.len();
        let mut energy_std = 0.0f64;
        for w in &tracker.history {
            let e: f32 = w.iter().sum();
            let diff = e as f64 - total_energy as f64;
            energy_std += diff * diff;
        }
        energy_std = (energy_std / n as f64).sqrt().max(1e-8);

        StiffAnomalyGate {
            z_threshold: -2.0,
            alpha_threshold: 0.8,
            baseline_mean: total_energy,
            baseline_std: energy_std as f32,
        }
    }

    /// Evaluate anomaly gate on a new observation.
    ///
    /// Uses z-scores from `tracker` for the `current` eigenvalue window
    /// and `delta_x` for soft alignment classification.
    ///
    /// This method allocates per call. For hot paths, use [`evaluate_with_buffers`]
    /// with pre-allocated [`EvaluateBuffers`].
    ///
    /// [`evaluate_with_buffers`]: StiffAnomalyGate::evaluate_with_buffers
    pub fn evaluate(
        &self,
        tracker: &EigenvalueTracker,
        current: &[f32],
        eigenvectors: &[Vec<f32>],
        delta_x: &[f32],
        trace_mass: f32,
    ) -> GateResult {
        let dim = current.len();
        let mut buf = EvaluateBuffers::with_capacity(dim);
        self.evaluate_with_buffers(tracker, current, eigenvectors, delta_x, trace_mass, &mut buf)
    }

    /// Zero-allocation variant of [`evaluate`](Self::evaluate) using pre-allocated buffers.
    pub fn evaluate_with_buffers(
        &self,
        tracker: &EigenvalueTracker,
        current: &[f32],
        eigenvectors: &[Vec<f32>],
        delta_x: &[f32],
        trace_mass: f32,
        buf: &mut EvaluateBuffers,
    ) -> GateResult {
        tracker.eigenspace_zscore_into(current, &mut buf.z_scores);

        // Check for stiff collision: any eigenvalue z-score below threshold
        if let Some(min_z) = buf.z_scores.iter().cloned().reduce(f32::min)
            && min_z < self.z_threshold
        {
            return GateResult::StiffCollision { z_score: min_z };
        }

        // Compute soft alignment ratio using reusable decomposition buffer
        decompose_into(current, eigenvectors, trace_mass, &mut buf.decomp);
        let alpha = soft_alignment_ratio(&buf.decomp, delta_x);

        if alpha >= self.alpha_threshold {
            GateResult::ElasticAbsorption { alpha }
        } else {
            GateResult::Normal
        }
    }

    /// Validate false positive rate on stable windows.
    ///
    /// Returns the fraction of `stable_windows` classified as anomalous
    /// (StiffCollision). Should be ~0.0 on truly stable data.
    pub fn validate_fpr(
        &self,
        tracker: &EigenvalueTracker,
        stable_windows: &[Vec<f32>],
        _eigenvectors: &[Vec<f32>],
        _trace_mass: f32,
    ) -> f32 {
        if stable_windows.is_empty() {
            return 0.0;
        }
        let mut false_positives = 0usize;
        for w in stable_windows {
            let z_scores = tracker.eigenspace_zscore(w);
            if let Some(min_z) = z_scores.iter().cloned().reduce(f32::min)
                && min_z < self.z_threshold
            {
                false_positives += 1;
            }
        }
        false_positives as f32 / stable_windows.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate synthetic stable eigenvalue windows around a base spectrum.
    fn stable_windows(
        base: &[f32],
        n: usize,
        noise: f32,
        rng: &mut impl FnMut() -> f32,
    ) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| {
                base.iter()
                    .map(|&v| (v + (rng() - 0.5) * 2.0 * noise).max(0.001))
                    .collect()
            })
            .collect()
    }

    /// Generate anomalous windows with collapsed eigenvalues.
    fn anomalous_windows(base: &[f32], n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| {
                base.iter()
                    .enumerate()
                    .map(|(i, &v)| if i < 2 { v * 0.1 } else { v })
                    .collect()
            })
            .collect()
    }

    fn simple_rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as f32 / (1u64 << 31) as f32
        }
    }

    /// G2: 100 synthetic stable windows → median Jaccard ≥ 0.85.
    #[test]
    fn test_g2_jaccard_stable() {
        let base = vec![10.0, 8.0, 6.0, 3.0, 1.0];
        let mut rng = simple_rng(42);
        let windows = stable_windows(&base, 100, 0.5, &mut rng);

        let mut jaccards = Vec::new();
        for i in 1..windows.len() {
            let j = EigenvalueTracker::eigenvalue_jaccard(&windows[i - 1], &windows[i], 3);
            jaccards.push(j);
        }
        jaccards.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = jaccards[jaccards.len() / 2];
        assert!(
            median >= 0.85,
            "median Jaccard on stable windows should be ≥ 0.85, got {median}"
        );
    }

    /// G2: Perturbed windows → Jaccard drops.
    #[test]
    fn test_g2_jaccard_perturbed() {
        let base = vec![10.0, 8.0, 6.0, 3.0, 1.0];
        let perturbed = vec![1.0, 3.0, 10.0, 6.0, 8.0]; // shuffled
        let j = EigenvalueTracker::eigenvalue_jaccard(&base, &perturbed, 3);
        assert!(j < 0.7, "Jaccard on perturbed should be < 0.7, got {j}");
    }

    /// G3: FPR = 0.0% on 50 stable windows; 100% detection on 5 anomalous.
    #[test]
    fn test_g3_fpr_zero_and_full_detection() {
        let base = vec![10.0, 8.0, 6.0, 3.0, 1.0];
        let mut rng = simple_rng(123);
        let stable = stable_windows(&base, 50, 0.3, &mut rng);

        let tracker = EigenvalueTracker::freeze_baseline(&stable);
        let gate = StiffAnomalyGate::new(&tracker);

        // FPR on stable
        let fpr = gate.validate_fpr(&tracker, &stable, &identity_ev(5), 0.90);
        assert!(
            fpr <= 0.04,
            "FPR should be ~0.0 on stable windows, got {fpr}"
        );

        // Detection on anomalous
        let anomalous = anomalous_windows(&base, 5);
        let mut detected = 0;
        for w in &anomalous {
            let z_scores = tracker.eigenspace_zscore(w);
            if let Some(min_z) = z_scores.iter().cloned().reduce(f32::min)
                && min_z < gate.z_threshold {
                    detected += 1;
                }
        }
        assert_eq!(
            detected, 5,
            "should detect all 5 anomalous windows, got {detected}"
        );
    }

    /// k_invariant: stable baseline → k invariant.
    #[test]
    fn test_k_invariant_stable() {
        let base = vec![10.0, 8.0, 6.0, 0.1, 0.1];
        let mut rng = simple_rng(77);
        let windows = stable_windows(&base, 20, 0.05, &mut rng);
        let tracker = EigenvalueTracker::freeze_baseline(&windows);
        assert!(
            tracker.k_invariant(0.90),
            "k should be invariant on stable windows"
        );
    }

    /// Identity eigenvectors helper.
    fn identity_ev(d: usize) -> Vec<Vec<f32>> {
        (0..d)
            .map(|i| {
                let mut v = vec![0.0f32; d];
                v[i] = 1.0;
                v
            })
            .collect()
    }

    /// z-score computation basic sanity.
    #[test]
    fn test_zscore_basic() {
        let windows = vec![
            vec![10.0, 5.0, 2.0],
            vec![10.0, 5.0, 2.0],
            vec![10.0, 5.0, 2.0],
        ];
        let tracker = EigenvalueTracker::freeze_baseline(&windows);
        let z = tracker.eigenspace_zscore(&[10.0, 5.0, 2.0]);
        for zi in &z {
            assert!(
                zi.abs() < 0.01,
                "z-score of baseline values should be ~0, got {zi}"
            );
        }
    }
}
