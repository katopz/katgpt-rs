//! Spectral Irrep Pruner — spectral flatness-based speculative decoding pruning (Plan 246).
//!
//! Detects whether logit distributions exhibit "converged" structure (high spectral flatness
//! in the FFT domain = sharp peaks in logit space) vs "still competing" (low spectral flatness
//! = flat/uniform logit distribution). Inspired by arXiv:2606.02993: converged neurons encode
//! single irreducible representations.
//!
//! # How it works
//!
//! The 1D FFT of the logit vector reveals frequency structure:
//! - **Peaked logits** (one dominant token) → FFT magnitude is spread across frequencies → high spectral flatness → converged
//! - **Uniform logits** (many competing tokens) → FFT magnitude is concentrated in DC → low spectral flatness → uncertain
//!
//! ```text
//! logits → 1D FFT → |spectrum|² → spectral_flatness()
//!     ≥ threshold → converged (allow all tokens)
//!      < threshold → uncertain  (only top-k by logit value)
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! let mut pruner = IrrepPruner::new(0.7);
//! pruner.set_logits(&logits);
//! // Now use as ConstraintPruner
//! if pruner.is_valid(depth, token_idx, parent_tokens) {
//!     // accept token
//! }
//! ```

use crate::traits::ConstraintPruner;
use rustfft::{FftPlanner, num_complex::Complex};

// ── Spectral Flatness Utility ────────────────────────────────────

/// Compute spectral flatness of a logit vector via 1D FFT.
///
/// Spectral flatness = geometric_mean(spectrum) / arithmetic_mean(spectrum)
/// computed on the magnitude spectrum |X[k]|² for k=1..N/2 (DC excluded).
///
/// - Range: [0, 1]
/// - **High (~1.0)**: FFT magnitude is spread uniformly → logit vector has sharp features → **converged**
/// - **Low (~0.0)**: FFT magnitude is concentrated in DC → logit vector is smooth/flat → **uncertain**
///
/// Uses pre-allocated scratch buffer for zero-alloc hot path.
///
/// # Panics
///
/// Panics if `logits` is empty.
pub fn spectral_flatness(
    logits: &[f32],
    scratch: &mut Vec<Complex<f64>>,
    planner: &mut FftPlanner<f64>,
) -> f32 {
    let n = logits.len();
    assert!(n > 0, "spectral_flatness requires non-empty input");

    // Prepare scratch buffer: resize if needed, then fill
    if scratch.len() < n {
        scratch.resize(n, Complex::new(0.0, 0.0));
    }
    for (i, &v) in logits.iter().enumerate() {
        scratch[i] = Complex::new(v as f64, 0.0);
    }

    // 1D FFT (reuses cached planner)
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut scratch[..n]);

    // Magnitude spectrum: |X[k]|² for k=1..N/2 (skip DC)
    let spectrum_end = n / 2;
    if spectrum_end <= 1 {
        return 0.0; // single-bin spectrum is trivially peaked
    }

    let mut log_sum: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut count: usize = 0;

    for bin in scratch.iter().take(spectrum_end + 1).skip(1) {
        let mag_sq = bin.re * bin.re + bin.im * bin.im;
        if mag_sq <= 0.0 {
            continue; // skip zero bins (they'd kill geometric mean)
        }
        log_sum += mag_sq.ln();
        sum += mag_sq;
        count += 1;
    }

    if count == 0 || sum <= 0.0 {
        return 0.0; // all-zero spectrum beyond DC = smooth input = uncertain
    }

    // flatness = geo_mean / arith_mean = exp(log_geo_mean - ln(arith_mean))
    // Single exp avoids intermediate overflow from geo_mean when magnitudes are large.
    let log_geo_mean = log_sum / count as f64;
    let arith_mean = sum / count as f64;
    let flatness = (log_geo_mean - arith_mean.ln()).exp();
    // Clamp to [0, 1] — numerical noise can push slightly above 1.0
    flatness.clamp(0.0, 1.0) as f32
}

// ── IrrepPruner ──────────────────────────────────────────────────

/// Spectral irrep-based pruner for speculative decoding.
///
/// Prunes tokens when the logit distribution is uncertain (low spectral flatness
/// = flat distribution = many competing modes). When the distribution has converged
/// (high spectral flatness = peaked distribution = single dominant mode), all tokens
/// are allowed.
///
/// Inspired by arXiv:2606.02993: converged neurons encode single irreps.
pub struct IrrepPruner {
    /// Cached FFT planner (avoids per-call allocation).
    planner: FftPlanner<f64>,
    /// Pre-allocated scratch buffer for FFT.
    scratch: Vec<Complex<f64>>,
    /// Cached logits for rank-based gating.
    logits: Vec<f32>,
    /// Cached sorted indices (descending by logit value), updated by set_logits.
    /// Used for top-k gating when uncertain.
    sorted_indices: Vec<usize>,
    /// O(1) bitmap for top-k membership checks.
    top_k_bitmap: Vec<bool>,
    /// How many tokens from sorted_indices are "valid" given current flatness.
    valid_count: usize,
    /// Number of top-k tokens to keep when distribution is uncertain.
    /// When flatness < threshold, only tokens ranked in top-k by logit value are valid.
    pub top_k_when_uncertain: usize,
    /// Threshold below which a step is considered "uncertain" (prune aggressively).
    /// Default: 0.3. When spectral flatness < threshold → uncertain → top-k only.
    /// Higher = more aggressive pruning.
    pub convergence_threshold: f32,
    /// Current spectral flatness (updated by set_logits).
    current_flatness: f32,
}

/// Configuration for [`IrrepPruner`].
///
/// Default: threshold=0.7, top_k=10, pre-alloc=0 (lazy).
#[derive(Debug, Clone)]
pub struct IrrepPrunerConfig {
    pub convergence_threshold: f32,
    pub top_k_when_uncertain: usize,
    pub max_vocab: usize,
}

impl Default for IrrepPrunerConfig {
    fn default() -> Self {
        Self {
            convergence_threshold: 0.7,
            top_k_when_uncertain: 10,
            max_vocab: 0,
        }
    }
}

impl IrrepPrunerConfig {
    /// Create config with a specific threshold, keeping other defaults.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            convergence_threshold: threshold,
            ..Default::default()
        }
    }
}

/// Top-level factory: create an [`IrrepPruner`] from config.
pub fn irrep_pruner_from_config(config: &IrrepPrunerConfig) -> IrrepPruner {
    IrrepPruner::from_config(config)
}

impl IrrepPruner {
    /// Create a new IrrepPruner with the given convergence threshold.
    ///
    /// Uses default `top_k_when_uncertain` = 10.
    pub fn new(convergence_threshold: f32) -> Self {
        Self::with_capacity(convergence_threshold, 10, 0)
    }

    /// Create a new IrrepPruner with pre-allocated capacity.
    ///
    /// `max_vocab` pre-allocates buffers for that many logits. Pass 0 for lazy allocation.
    pub fn with_capacity(
        convergence_threshold: f32,
        top_k_when_uncertain: usize,
        max_vocab: usize,
    ) -> Self {
        Self {
            planner: FftPlanner::new(),
            scratch: Vec::with_capacity(max_vocab),
            logits: Vec::with_capacity(max_vocab),
            sorted_indices: Vec::with_capacity(max_vocab),
            top_k_bitmap: Vec::with_capacity(max_vocab),
            valid_count: 0,
            top_k_when_uncertain,
            convergence_threshold,
            current_flatness: 0.0,
        }
    }

    /// Create from config.
    pub fn from_config(config: &IrrepPrunerConfig) -> Self {
        Self::with_capacity(
            config.convergence_threshold,
            config.top_k_when_uncertain,
            config.max_vocab,
        )
    }

    /// Update the pruner with current logits. Must be called before `is_valid`.
    ///
    /// Computes spectral flatness and caches ranking for top-k gating.
    pub fn set_logits(&mut self, logits: &[f32]) {
        if logits.is_empty() {
            self.current_flatness = 0.0;
            self.logits.clear();
            self.sorted_indices.clear();
            self.valid_count = 0;
            return;
        }

        // Compute spectral flatness
        self.current_flatness = spectral_flatness(logits, &mut self.scratch, &mut self.planner);

        // Cache logits — reuse allocation when size matches, resize+copy otherwise
        if self.logits.len() == logits.len() {
            self.logits.copy_from_slice(logits);
        } else {
            self.logits.clear();
            self.logits.extend_from_slice(logits);
        }

        // Determine valid count based on flatness
        // High flatness = converged = allow all tokens
        // Low flatness = uncertain = only top-k
        match self.current_flatness >= self.convergence_threshold {
            true => {
                // Converged: all tokens valid
                self.valid_count = logits.len();
            }
            false => {
                // Uncertain: only top-k tokens valid
                let k = self.top_k_when_uncertain.min(logits.len());
                self.valid_count = k;
                self.rebuild_sorted_indices();
            }
        }
    }

    /// Get current spectral flatness (computed by last `set_logits` call).
    pub fn current_flatness(&self) -> f32 {
        self.current_flatness
    }

    /// Get current convergence score (= spectral flatness).
    ///
    /// High (~1.0) = converged (peaked logits).
    /// Low (~0.0) = uncertain (flat logits).
    pub fn convergence_score(&self) -> f32 {
        self.current_flatness
    }

    /// Check if a token_idx is in the top-k set (only meaningful when uncertain).
    /// O(1) bitmap lookup.
    ///
    /// # Safety
    /// Caller must ensure `token_idx < top_k_bitmap.len()` (guaranteed by is_valid's
    /// out-of-range check and the fact that top_k_bitmap is always >= logits.len()).
    #[inline]
    fn is_in_top_k(&self, token_idx: usize) -> bool {
        debug_assert!(
            token_idx < self.top_k_bitmap.len(),
            "is_in_top_k: token_idx {token_idx} >= bitmap len {}",
            self.top_k_bitmap.len()
        );
        // SAFETY: is_valid already checks token_idx < logits.len(), and
        // rebuild_sorted_indices ensures top_k_bitmap.len() == logits.len().
        unsafe { *self.top_k_bitmap.get_unchecked(token_idx) }
    }

    /// Rebuild sorted indices by descending logit value using partial sort for top-k.
    /// Builds O(1) bitmap for membership checks.
    fn rebuild_sorted_indices(&mut self) {
        let n = self.logits.len();
        self.sorted_indices.clear();
        // SAFETY: indices are in-bounds by construction
        self.sorted_indices.extend(0..n);

        let k = self.valid_count.min(n);
        if k < n {
            // Partial sort: partition into top-k and rest
            self.sorted_indices.select_nth_unstable_by(k, |&a, &b| {
                self.logits[b]
                    .partial_cmp(&self.logits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // Sort just the top-k portion for determinism
            self.sorted_indices[..k].sort_unstable_by(|&a, &b| {
                self.logits[b]
                    .partial_cmp(&self.logits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Build bitmap from sorted top-k.
        // Ensure bitmap covers full vocabulary
        if self.top_k_bitmap.len() < n {
            self.top_k_bitmap.resize(n, false);
        }
        // Bulk clear — single memset, faster than scattered indirect writes
        self.top_k_bitmap[..n].fill(false);
        for &idx in &self.sorted_indices[..k] {
            // SAFETY: idx < n by construction (sorted_indices contains 0..n)
            unsafe {
                *self.top_k_bitmap.get_unchecked_mut(idx) = true;
            }
        }
    }
}

impl ConstraintPruner for IrrepPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        // If no logits set, allow everything
        if self.logits.is_empty() {
            return true;
        }

        // Out-of-range token
        if token_idx >= self.logits.len() {
            return false;
        }

        match self.current_flatness >= self.convergence_threshold {
            // Converged (high flatness): all tokens valid
            true => true,
            // Uncertain (low flatness): only top-k tokens valid
            false => self.is_in_top_k(token_idx),
        }
    }

    fn batch_is_valid(
        &self,
        _depth: usize,
        candidates: &[usize],
        _parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        // If no logits set, allow everything
        if self.logits.is_empty() {
            let len = candidates.len();
            let cap = results.len();
            results[..len.min(cap)].fill(true);
            return;
        }

        match self.current_flatness >= self.convergence_threshold {
            // Converged: all in-range valid
            true => {
                let len = candidates.len().min(results.len());
                for (i, r) in results.iter_mut().enumerate().take(len) {
                    *r = candidates[i] < self.logits.len();
                }
            }
            // Uncertain: check top-k membership via O(1) bitmap
            false => {
                let len = candidates.len().min(results.len());
                let bitmap_len = self.top_k_bitmap.len();
                for (i, r) in results.iter_mut().enumerate().take(len) {
                    let idx = candidates[i];
                    *r = idx < bitmap_len && unsafe { *self.top_k_bitmap.get_unchecked(idx) };
                }
            }
        }
    }

    fn manifold_score(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.convergence_score()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Peaked logit vector (impulse) → FFT magnitude is flat → high spectral flatness.
    #[test]
    fn flatness_peaked_spectrum() {
        // Impulse-like: one large value, rest near zero
        let mut logits = vec![0.01f32; 64];
        logits[0] = 10.0;
        let mut scratch = Vec::new();
        let mut planner = FftPlanner::new();
        let flatness = spectral_flatness(&logits, &mut scratch, &mut planner);
        assert!(
            flatness > 0.7,
            "peaked logit vector should have high flatness (spread FFT), got {flatness}"
        );
    }

    /// Uniform logit vector → FFT magnitude is peaked at DC → low spectral flatness.
    #[test]
    fn flatness_flat_spectrum() {
        let logits = vec![1.0f32; 64];
        let mut scratch = Vec::new();
        let mut planner = FftPlanner::new();
        let flatness = spectral_flatness(&logits, &mut scratch, &mut planner);
        // Uniform input: all energy in DC (which we skip), remaining bins ≈ 0
        assert!(
            flatness < 0.3,
            "uniform logit vector should have low flatness (peaked FFT at DC), got {flatness}"
        );
    }

    /// Typical logit distribution → flatness in (0, 1)
    #[test]
    fn flatness_realistic_logits() {
        // Simulate a realistic logit distribution: one peak, a few medium values, rest near zero
        let mut logits = vec![0.1f32; 128];
        logits[42] = 8.0;
        logits[17] = 3.0;
        logits[99] = 2.5;
        logits[7] = 1.5;
        let mut scratch = Vec::new();
        let mut planner = FftPlanner::new();
        let flatness = spectral_flatness(&logits, &mut scratch, &mut planner);
        assert!(
            (0.0..1.0).contains(&flatness),
            "realistic logits should have flatness in (0, 1), got {flatness}"
        );
    }

    /// High flatness (converged) → all tokens valid
    #[test]
    fn pruner_converged_allows_tokens() {
        let mut pruner = IrrepPruner::new(0.3);
        // Peaked distribution: flatness should be high
        let mut logits = vec![0.01f32; 64];
        logits[0] = 10.0;
        logits[1] = 5.0;
        pruner.set_logits(&logits);

        assert!(
            pruner.current_flatness() >= 0.3,
            "flatness should be above threshold, got {}",
            pruner.current_flatness()
        );

        // All tokens should be valid when converged
        for i in 0..10 {
            assert!(
                pruner.is_valid(0, i, &[]),
                "token {i} should be valid when converged"
            );
        }
    }

    /// Low flatness (uncertain) → only top-k tokens valid
    #[test]
    fn pruner_uncertain_prunes_aggressively() {
        let mut pruner = IrrepPruner::new(0.5);
        pruner.top_k_when_uncertain = 3;

        // Uniform distribution: flatness should be low
        let logits = vec![1.0f32; 64];
        pruner.set_logits(&logits);

        assert!(
            pruner.current_flatness() < 0.5,
            "flatness should be below threshold, got {}",
            pruner.current_flatness()
        );

        // With uniform logits, top-k is arbitrary — but only k tokens should be valid
        let mut valid_count = 0;
        for i in 0..64 {
            if pruner.is_valid(0, i, &[]) {
                valid_count += 1;
            }
        }
        assert_eq!(
            valid_count, 3,
            "should have exactly 3 valid tokens when uncertain, got {valid_count}"
        );
    }

    /// convergence_score = flatness
    #[test]
    fn convergence_score_equals_flatness() {
        let mut pruner = IrrepPruner::new(0.5);
        let logits = vec![1.0f32; 32];
        pruner.set_logits(&logits);

        let flatness = pruner.current_flatness();
        let score = pruner.convergence_score();
        assert!(
            (score - flatness).abs() < 1e-6,
            "convergence_score should equal flatness: score={score}, flatness={flatness}"
        );
    }

    /// No logits set → everything valid
    #[test]
    fn pruner_no_logits_allows_all() {
        let pruner = IrrepPruner::new(0.3);
        assert!(pruner.is_valid(0, 0, &[]), "should be valid with no logits");
        assert!(
            pruner.is_valid(0, 999, &[]),
            "should be valid with no logits"
        );
    }

    /// Empty logits → graceful
    #[test]
    fn pruner_empty_logits() {
        let mut pruner = IrrepPruner::new(0.3);
        pruner.set_logits(&[]);
        assert_eq!(pruner.current_flatness(), 0.0);
        assert!(pruner.is_valid(0, 0, &[]));
    }

    /// Scratch buffer reuse — calling spectral_flatness twice doesn't panic
    #[test]
    fn scratch_buffer_reuse() {
        let mut scratch = Vec::new();
        let mut planner = FftPlanner::new();
        let logits1 = vec![1.0f32; 32];
        let logits2 = vec![0.5f32; 64]; // larger

        let f1 = spectral_flatness(&logits1, &mut scratch, &mut planner);
        let f2 = spectral_flatness(&logits2, &mut scratch, &mut planner);

        assert!(f1.is_finite());
        assert!(f2.is_finite());
    }

    /// batch_is_valid matches is_valid for each candidate
    #[test]
    fn batch_matches_individual() {
        let mut pruner = IrrepPruner::new(0.5);
        pruner.top_k_when_uncertain = 5;

        // Mixed distribution: enough structure that some tokens have higher logits
        let mut logits = vec![0.1f32; 32];
        logits[0] = 10.0;
        logits[1] = 8.0;
        logits[2] = 6.0;
        logits[3] = 4.0;
        logits[4] = 2.0;
        pruner.set_logits(&logits);

        let candidates: Vec<usize> = (0..32).collect();
        let mut batch_results = vec![false; 32];
        pruner.batch_is_valid(0, &candidates, &[], &mut batch_results);

        for (i, &candidate) in candidates.iter().enumerate() {
            let individual = pruner.is_valid(0, candidate, &[]);
            assert_eq!(
                batch_results[i], individual,
                "batch result for token {candidate} doesn't match individual"
            );
        }
    }

    /// manifold_score returns convergence score regardless of token
    #[test]
    fn manifold_score_returns_convergence() {
        let mut pruner = IrrepPruner::new(0.5);
        let logits = vec![1.0f32; 32];
        pruner.set_logits(&logits);

        let score = pruner.manifold_score(0, 0, &[]);
        let expected = pruner.convergence_score();
        assert!(
            (score - expected).abs() < 1e-6,
            "manifold_score should equal convergence_score: got {score}, expected {expected}"
        );
    }
}

// TL;DR: IrrepPruner computes spectral flatness of logit distributions via 1D FFT.
// High flatness (converged) → all tokens valid. Low flatness (uncertain) → only top-k valid.
// Zero-alloc hot path via pre-allocated scratch buffer. Feature-gated behind `spectral_pruner`.
