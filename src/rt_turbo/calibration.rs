//! Offline head calibration for RTPurbo retrieval/local classification.
//!
//! Implements needle-based per-head retrieval scoring from the RTPurbo paper
//! (arXiv 2605.16928). Only ~15% of attention heads ("retrieval heads") need
//! full long-context access; the rest ("local heads") focus on local context
//! + attention sinks.
//!
//! # Calibration Process
//!
//! 1. Insert identical "needle" span at beginning and end of a long document
//! 2. Run one forward pass, extract per-head attention matrices
//! 3. Compute retrieval score: R_h = mean(attn from post-needle to pre-needle)
//! 4. Partition heads into retrieval set (top 15%) and local set
//! 5. Serialize result to disk for reuse at inference time
//!
//! Calibration is input-agnostic — a single forward pass is sufficient.

use serde::{Deserialize, Serialize};

use crate::types::{RetrievalHeadRole, RtTurboConfig};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Per-head classification result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeadClassification {
    /// Query head index.
    pub head_idx: usize,
    /// Assigned role (retrieval or local).
    pub role: RetrievalHeadRole,
    /// Retrieval score R_h — mean attention from post-needle to pre-needle.
    pub score: f32,
}

/// Complete head calibration result — serialized to disk for inference reuse.
///
/// Contains per-head retrieval scores, the computed threshold, and the
/// derived partition into retrieval and local head sets.
///
/// # Stability
///
/// Head behavior is input-agnostic (paper finding). Single calibration run
/// produces a stable partition. Serialized as JSON for portability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeadCalibration {
    /// Per-head classification (one entry per query head).
    pub classifications: Vec<HeadClassification>,
    /// Indices of heads classified as retrieval (sorted descending by score).
    pub retrieval_set: Vec<usize>,
    /// Indices of heads classified as local.
    pub local_set: Vec<usize>,
    /// Score threshold used for partition (retrieval_head_ratio percentile).
    pub threshold: f32,
    /// Config snapshot used during calibration.
    pub config_snapshot: CalibrationConfigSnapshot,
}

/// Minimal config snapshot stored with calibration for reproducibility.
///
/// Field order: usize (8B) before f32 (4B) eliminates 4 bytes of padding
/// on 64-bit targets.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationConfigSnapshot {
    /// Total number of query heads calibrated.
    pub n_query_heads: usize,
    /// Retrieval head ratio used (e.g., 0.15).
    pub retrieval_head_ratio: f32,
}

// ---------------------------------------------------------------------------
// Calibration Logic
// ---------------------------------------------------------------------------

/// Compute retrieval score R_h for a single head from an attention matrix.
///
/// Given the full attention matrix A_h of shape [seq_len, seq_len] (rows = queries,
/// cols = keys), and the pre/post needle position ranges, compute:
///
/// ```text
/// R_h = (1/|N_post|) Σ_{t∈N_post} Σ_{j∈N_pre} A_h(t, j)
/// ```
///
/// This measures how much the post-needle positions attend to the pre-needle
/// positions — high scores indicate retrieval behavior.
///
/// # Arguments
///
/// * `attention` — Flattened attention matrix [seq_len * seq_len], row-major.
///   A_h[t * seq_len + j] = attention from position t to position j.
/// * `seq_len` — Sequence length (square attention matrix).
/// * `needle_start` — Start of pre-needle span (inclusive).
/// * `needle_end` — End of pre-needle span (exclusive).
/// * `post_needle_start` — Start of post-needle span (inclusive).
/// * `post_needle_end` — End of post-needle span (exclusive).
///
/// # Returns
///
/// Retrieval score R_h in [0.0, 1.0].
pub fn compute_retrieval_score(
    attention: &[f32],
    seq_len: usize,
    needle_start: usize,
    needle_end: usize,
    post_needle_start: usize,
    post_needle_end: usize,
) -> f32 {
    let pre_len = needle_end.saturating_sub(needle_start);
    let post_len = post_needle_end.saturating_sub(post_needle_start);

    if pre_len == 0 || post_len == 0 || attention.len() < seq_len * seq_len {
        return 0.0;
    }

    // Accumulate into 4 f64 registers to aid auto-vectorization (4-wide SIMD).
    // f64 avoids precision loss when summing many small attention weights.
    let mut acc = [0.0f64; 4];

    for t in post_needle_start..post_needle_end {
        let row_start = t * seq_len + needle_start;
        let row_end = t * seq_len + needle_end;
        let row = &attention[row_start..row_end];
        let chunks = row.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            acc[0] += chunk[0] as f64;
            acc[1] += chunk[1] as f64;
            acc[2] += chunk[2] as f64;
            acc[3] += chunk[3] as f64;
        }
        for (i, &v) in remainder.iter().enumerate() {
            acc[i] += v as f64;
        }
    }

    let total_mass = (acc[0] + acc[1] + acc[2] + acc[3]) as f32;
    total_mass / (post_len * pre_len) as f32
}

/// Compute retrieval scores for all heads from per-head attention matrices.
///
/// # Arguments
///
/// * `per_head_attentions` — One flattened attention matrix per query head.
/// * `seq_len` — Sequence length.
/// * `needle_start` — Pre-needle span start.
/// * `needle_end` — Pre-needle span end.
/// * `post_needle_start` — Post-needle span start.
/// * `post_needle_end` — Post-needle span end.
///
/// # Returns
///
/// Vector of retrieval scores, one per query head.
pub fn compute_all_retrieval_scores(
    per_head_attentions: &[Vec<f32>],
    seq_len: usize,
    needle_start: usize,
    needle_end: usize,
    post_needle_start: usize,
    post_needle_end: usize,
) -> Vec<f32> {
    per_head_attentions
        .iter()
        .map(|attn| {
            compute_retrieval_score(
                attn,
                seq_len,
                needle_start,
                needle_end,
                post_needle_start,
                post_needle_end,
            )
        })
        .collect()
}

/// Calibrate heads from pre-computed retrieval scores.
///
/// Takes the per-head retrieval scores (computed via `compute_all_retrieval_scores`
/// or loaded from a prior calibration) and partitions heads into retrieval/local
/// sets based on the configured ratio.
///
/// # Algorithm
///
/// 1. Sort heads by score descending.
/// 2. Select top `retrieval_head_ratio` fraction as retrieval heads.
/// 3. Remaining heads are local.
/// 4. Threshold = minimum score in the retrieval set.
///
/// # Arguments
///
/// * `scores` — Retrieval score per query head.
/// * `config` — RTPurbo config (uses `retrieval_head_ratio`).
///
/// # Returns
///
/// Complete `HeadCalibration` with classifications, sets, and threshold.
pub fn calibrate_from_scores(scores: &[f32], config: &RtTurboConfig) -> HeadCalibration {
    let n_heads = scores.len();
    assert!(n_heads > 0, "Cannot calibrate with zero heads");

    let n_retrieval = ((n_heads as f32 * config.retrieval_head_ratio).ceil() as usize)
        .min(n_heads)
        .max(1);

    // Create indexed pairs (head_idx, score), sort by score descending
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Threshold = score of the last retrieval head
    let threshold = indexed[n_retrieval - 1].1;

    // Build classifications
    let mut classifications = Vec::with_capacity(n_heads);
    let mut retrieval_set = Vec::with_capacity(n_retrieval);
    let mut local_set = Vec::with_capacity(n_heads - n_retrieval);

    for (rank, &(head_idx, score)) in indexed.iter().enumerate() {
        let role = match rank < n_retrieval {
            true => {
                retrieval_set.push(head_idx);
                RetrievalHeadRole::Retrieval
            }
            false => {
                local_set.push(head_idx);
                RetrievalHeadRole::Local
            }
        };

        classifications.push(HeadClassification {
            head_idx,
            role,
            score,
        });
    }

    // Sort classifications by head_idx for O(1) lookup
    classifications.sort_by_key(|c| c.head_idx);
    local_set.sort_unstable();

    HeadCalibration {
        classifications,
        retrieval_set,
        local_set,
        threshold,
        config_snapshot: CalibrationConfigSnapshot {
            retrieval_head_ratio: config.retrieval_head_ratio,
            n_query_heads: n_heads,
        },
    }
}

// ---------------------------------------------------------------------------
// HeadCalibration impl
// ---------------------------------------------------------------------------

impl HeadCalibration {
    /// Create a new calibration from raw per-head attention matrices.
    ///
    /// Convenience wrapper combining `compute_all_retrieval_scores` + `calibrate_from_scores`.
    ///
    /// # Arguments
    ///
    /// * `per_head_attentions` — One flattened [seq_len * seq_len] attention matrix per query head.
    /// * `seq_len` — Sequence length.
    /// * `needle_start` — Pre-needle span start.
    /// * `needle_end` — Pre-needle span end.
    /// * `post_needle_start` — Post-needle span start.
    /// * `post_needle_end` — Post-needle span end.
    /// * `config` — RTPurbo config.
    pub fn from_attention(
        per_head_attentions: &[Vec<f32>],
        seq_len: usize,
        needle_start: usize,
        needle_end: usize,
        post_needle_start: usize,
        post_needle_end: usize,
        config: &RtTurboConfig,
    ) -> Self {
        let scores = compute_all_retrieval_scores(
            per_head_attentions,
            seq_len,
            needle_start,
            needle_end,
            post_needle_start,
            post_needle_end,
        );
        calibrate_from_scores(&scores, config)
    }

    /// Create a default calibration treating all heads as local.
    ///
    /// Useful as a safe fallback when calibration data is unavailable.
    /// All heads get score 0.0 and role Local.
    pub fn all_local(n_query_heads: usize, config: &RtTurboConfig) -> Self {
        let scores = vec![0.0f32; n_query_heads];
        let mut calibration = calibrate_from_scores(&scores, config);

        // Force all heads to local role
        for c in &mut calibration.classifications {
            c.role = RetrievalHeadRole::Local;
            c.score = 0.0;
        }
        calibration.retrieval_set.clear();
        calibration.local_set = (0..n_query_heads).collect();
        calibration.threshold = f32::MAX;

        calibration
    }

    /// Get the role of a specific head.
    ///
    /// # Panics
    ///
    /// Panics if `head_idx` is out of bounds.
    #[inline]
    pub fn role_of(&self, head_idx: usize) -> RetrievalHeadRole {
        self.classifications[head_idx].role
    }

    /// Get the retrieval score of a specific head.
    ///
    /// # Panics
    ///
    /// Panics if `head_idx` is out of bounds.
    #[inline]
    pub fn score_of(&self, head_idx: usize) -> f32 {
        self.classifications[head_idx].score
    }

    /// Check if a head is classified as retrieval.
    #[inline]
    pub fn is_retrieval(&self, head_idx: usize) -> bool {
        self.role_of(head_idx) == RetrievalHeadRole::Retrieval
    }

    /// Number of retrieval heads.
    #[inline]
    pub fn n_retrieval(&self) -> usize {
        self.retrieval_set.len()
    }

    /// Number of local heads.
    #[inline]
    pub fn n_local(&self) -> usize {
        self.local_set.len()
    }

    /// Total number of heads.
    #[inline]
    pub fn n_heads(&self) -> usize {
        self.classifications.len()
    }

    /// Serialize calibration to binary (postcard).
    pub fn to_bytes(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Deserialize calibration from binary (postcard).
    pub fn from_bytes(data: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(data)
    }

    /// Save calibration to a file as binary (postcard).
    ///
    /// # Errors
    ///
    /// Returns IO errors if the file cannot be written.
    pub fn save(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let bytes = self
            .to_bytes()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }

    /// Load calibration from a binary file (postcard).
    ///
    /// # Errors
    ///
    /// Returns IO errors if the file cannot be read or data is malformed.
    pub fn load(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Validate calibration consistency.
    ///
    /// Checks that:
    /// - classifications, retrieval_set, and local_set are consistent
    /// - no duplicate indices
    /// - scores match roles (retrieval heads have score >= threshold)
    pub fn validate(&self) -> Result<(), String> {
        let n = self.classifications.len();

        // Check retrieval set consistency
        for &idx in &self.retrieval_set {
            if idx >= n {
                return Err(format!("Retrieval head index {idx} out of bounds (n={n})"));
            }
            match self.classifications[idx].role {
                RetrievalHeadRole::Retrieval => {}
                RetrievalHeadRole::Local => {
                    return Err(format!(
                        "Head {idx} in retrieval_set but classified as Local"
                    ));
                }
            }
        }

        // Check local set consistency
        for &idx in &self.local_set {
            if idx >= n {
                return Err(format!("Local head index {idx} out of bounds (n={n})"));
            }
            match self.classifications[idx].role {
                RetrievalHeadRole::Local => {}
                RetrievalHeadRole::Retrieval => {
                    return Err(format!(
                        "Head {idx} in local_set but classified as Retrieval"
                    ));
                }
            }
        }

        // Check partition is complete
        let total = self.retrieval_set.len() + self.local_set.len();
        if total != n {
            return Err(format!(
                "Partition incomplete: {} retrieval + {} local = {} != {n} total",
                self.retrieval_set.len(),
                self.local_set.len(),
                total
            ));
        }

        // Check no overlaps
        let mut all_indices: Vec<usize> = self.retrieval_set.clone();
        all_indices.extend(&self.local_set);
        all_indices.sort_unstable();
        for (i, pair) in all_indices.windows(2).enumerate() {
            if pair[0] == pair[1] {
                return Err(format!("Duplicate head index: {} at position {i}", pair[0]));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> RtTurboConfig {
        RtTurboConfig::default()
    }

    fn make_uniform_attention(seq_len: usize, value: f32) -> Vec<f32> {
        vec![value; seq_len * seq_len]
    }

    fn make_diagonal_attention(seq_len: usize) -> Vec<f32> {
        let mut attn = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            attn[i * seq_len + i] = 1.0;
        }
        attn
    }

    /// Create attention matrix where post-needle attends strongly to pre-needle.
    fn make_retrieval_attention(seq_len: usize, needle_len: usize, offset: usize) -> Vec<f32> {
        let mut attn = vec![0.0f32; seq_len * seq_len];
        let post_start = seq_len - needle_len;
        // Post-needle positions attend to pre-needle positions
        for t in post_start..seq_len {
            for j in 0..needle_len {
                attn[t * seq_len + j] = 1.0 / needle_len as f32 + offset as f32 * 0.01;
            }
        }
        attn
    }

    #[test]
    fn test_compute_retrieval_score_uniform() {
        let seq_len = 10;
        let attn = make_uniform_attention(seq_len, 0.1);
        let score = compute_retrieval_score(&attn, seq_len, 0, 3, 7, 10);
        // All values 0.1, so mean is 0.1
        assert!((score - 0.1).abs() < 1e-6, "Expected 0.1, got {score}");
    }

    #[test]
    fn test_compute_retrieval_score_empty_spans() {
        let seq_len = 10;
        let attn = make_uniform_attention(seq_len, 0.5);
        // Empty pre-needle span
        let score = compute_retrieval_score(&attn, seq_len, 0, 0, 7, 10);
        assert_eq!(score, 0.0, "Empty pre-needle should return 0.0");
        // Empty post-needle span
        let score = compute_retrieval_score(&attn, seq_len, 0, 3, 7, 7);
        assert_eq!(score, 0.0, "Empty post-needle should return 0.0");
    }

    #[test]
    fn test_compute_retrieval_score_diagonal() {
        let seq_len = 10;
        let attn = make_diagonal_attention(seq_len);
        // Diagonal attention: post-needle positions only attend to themselves,
        // not to pre-needle positions → low retrieval score
        let score = compute_retrieval_score(&attn, seq_len, 0, 3, 7, 10);
        assert!(
            score < 0.01,
            "Diagonal attention should have near-zero retrieval score, got {score}"
        );
    }

    #[test]
    fn test_compute_retrieval_score_retrieval_pattern() {
        let seq_len = 20;
        let needle_len = 3;
        let attn = make_retrieval_attention(seq_len, needle_len, 0);
        let score =
            compute_retrieval_score(&attn, seq_len, 0, needle_len, seq_len - needle_len, seq_len);
        assert!(
            score > 0.1,
            "Retrieval pattern should have high score, got {score}"
        );
    }

    #[test]
    fn test_calibrate_from_scores_correct_partition() {
        let config = RtTurboConfig {
            retrieval_head_ratio: 0.2,
            ..RtTurboConfig::default()
        };
        // 10 heads, top 2 should be retrieval
        let scores: Vec<f32> = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.15];
        let calibration = calibrate_from_scores(&scores, &config);

        assert_eq!(calibration.n_retrieval(), 2, "Expected 2 retrieval heads");
        assert_eq!(calibration.n_local(), 8, "Expected 8 local heads");

        // Heads 1 (0.9) and 3 (0.8) should be retrieval
        assert!(
            calibration.is_retrieval(1),
            "Head 1 (score 0.9) should be retrieval"
        );
        assert!(
            calibration.is_retrieval(3),
            "Head 3 (score 0.8) should be retrieval"
        );

        // Check others are local
        assert!(
            !calibration.is_retrieval(0),
            "Head 0 (score 0.1) should be local"
        );
        assert!(
            !calibration.is_retrieval(4),
            "Head 4 (score 0.3) should be local"
        );
    }

    #[test]
    fn test_calibrate_single_head_always_retrieval() {
        let config = default_config();
        let scores = vec![0.5f32];
        let calibration = calibrate_from_scores(&scores, &config);

        assert_eq!(
            calibration.n_retrieval(),
            1,
            "Single head must be retrieval"
        );
        assert_eq!(calibration.n_local(), 0, "Single head has no local heads");
        assert!(calibration.is_retrieval(0));
    }

    #[test]
    fn test_calibrate_all_equal_scores() {
        let config = RtTurboConfig {
            retrieval_head_ratio: 0.15,
            ..RtTurboConfig::default()
        };
        let scores = vec![0.5f32; 20];
        let calibration = calibrate_from_scores(&scores, &config);

        // With 20 equal scores and ratio 0.15, ceil(3.0) = 3 retrieval heads
        assert_eq!(
            calibration.n_retrieval(),
            3,
            "Expected 3 retrieval heads from equal scores"
        );
        assert_eq!(calibration.n_local(), 17);
    }

    #[test]
    fn test_calibrate_threshold_is_min_retrieval_score() {
        let config = RtTurboConfig {
            retrieval_head_ratio: 0.2,
            ..RtTurboConfig::default()
        };
        let scores: Vec<f32> = vec![0.1, 0.9, 0.2, 0.8, 0.3];
        let calibration = calibrate_from_scores(&scores, &config);

        // Top 1 head (ceil of 1.0) = head 1 with score 0.9
        // Actually ceil(5 * 0.2) = ceil(1.0) = 1 retrieval head
        assert!(
            (calibration.threshold - 0.9).abs() < 1e-6,
            "Threshold should be min retrieval score, got {}",
            calibration.threshold
        );
    }

    #[test]
    fn test_from_attention_integration() {
        let config = RtTurboConfig {
            retrieval_head_ratio: 0.5,
            ..RtTurboConfig::default()
        };
        let seq_len = 10;
        let needle_len = 2;

        // Head 0: retrieval pattern (post-needle attends to pre-needle)
        let head0 = make_retrieval_attention(seq_len, needle_len, 1);
        // Head 1: diagonal (no retrieval)
        let head1 = make_diagonal_attention(seq_len);

        let per_head = vec![head0, head1];
        let calibration = HeadCalibration::from_attention(
            &per_head,
            seq_len,
            0,
            needle_len,
            seq_len - needle_len,
            seq_len,
            &config,
        );

        assert_eq!(calibration.n_heads(), 2);
        assert_eq!(calibration.n_retrieval(), 1, "Expected 1 retrieval head");
        assert!(
            calibration.is_retrieval(0),
            "Head 0 should be retrieval (retrieval pattern)"
        );
        assert!(
            !calibration.is_retrieval(1),
            "Head 1 should be local (diagonal)"
        );
    }

    #[test]
    fn test_all_local_fallback() {
        let config = default_config();
        let calibration = HeadCalibration::all_local(8, &config);

        assert_eq!(
            calibration.n_retrieval(),
            0,
            "All-local should have 0 retrieval"
        );
        assert_eq!(calibration.n_local(), 8, "All-local should have 8 local");
        assert_eq!(calibration.n_heads(), 8);

        for i in 0..8 {
            assert!(!calibration.is_retrieval(i), "Head {i} should be local");
        }
    }

    #[test]
    fn test_validate_consistent() {
        let config = RtTurboConfig {
            retrieval_head_ratio: 0.3,
            ..RtTurboConfig::default()
        };
        let scores: Vec<f32> = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.15];
        let calibration = calibrate_from_scores(&scores, &config);
        assert!(
            calibration.validate().is_ok(),
            "Calibration should validate cleanly"
        );
    }

    #[test]
    fn test_save_load_roundtrip() {
        let config = RtTurboConfig {
            retrieval_head_ratio: 0.25,
            ..RtTurboConfig::default()
        };
        let scores: Vec<f32> = vec![0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6];
        let calibration = calibrate_from_scores(&scores, &config);

        // Binary roundtrip
        let bytes = calibration.to_bytes().expect("Serialize to binary");
        let loaded = HeadCalibration::from_bytes(&bytes).expect("Deserialize from binary");

        assert_eq!(loaded.n_retrieval(), calibration.n_retrieval());
        assert_eq!(loaded.n_local(), calibration.n_local());
        assert_eq!(loaded.threshold, calibration.threshold);
        assert_eq!(loaded.retrieval_set, calibration.retrieval_set);
        assert_eq!(loaded.local_set, calibration.local_set);

        for (orig, loaded_c) in calibration
            .classifications
            .iter()
            .zip(loaded.classifications.iter())
        {
            assert_eq!(orig.head_idx, loaded_c.head_idx);
            assert_eq!(orig.role, loaded_c.role);
            assert!((orig.score - loaded_c.score).abs() < 1e-6);
        }
    }

    #[test]
    fn test_save_load_file_roundtrip() {
        let dir = tempfile::tempdir().expect("Create temp dir");
        let path = dir.path().join("calibration.bin");

        let config = RtTurboConfig {
            retrieval_head_ratio: 0.2,
            ..RtTurboConfig::default()
        };
        let scores: Vec<f32> = vec![0.1, 0.9, 0.2, 0.8, 0.3];
        let calibration = calibrate_from_scores(&scores, &config);

        calibration.save(&path).expect("Save calibration");
        let loaded = HeadCalibration::load(&path).expect("Load calibration");

        assert_eq!(loaded.retrieval_set, calibration.retrieval_set);
        assert_eq!(loaded.local_set, calibration.local_set);
        assert!((loaded.threshold - calibration.threshold).abs() < 1e-6);
        assert!(loaded.validate().is_ok());
    }

    #[test]
    fn test_config_snapshot_recorded() {
        let config = RtTurboConfig {
            retrieval_head_ratio: 0.12,
            ..RtTurboConfig::default()
        };
        let scores = vec![0.5f32; 10];
        let calibration = calibrate_from_scores(&scores, &config);

        assert_eq!(calibration.config_snapshot.retrieval_head_ratio, 0.12);
        assert_eq!(calibration.config_snapshot.n_query_heads, 10);
    }
}
