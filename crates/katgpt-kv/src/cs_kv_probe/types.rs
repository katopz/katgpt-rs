//! Decoupled types for the CS-KV probe module.
//!
//! Plan 280 — distilled from arxiv 2606.13594 (Research 247,
//! "See What I See, Know What I Think"). No game semantics: these structs are
//! generic inference primitives. `Episode` is an opaque labelled inference
//! record; the probe treats the eval function as a black box.

/// One inference episode: a flattened KV cache slice + task outcome label.
///
/// `kv_cache` is the flattened `[D]` slice for one inference pass. The probe
/// never interprets its contents — it is only forwarded to the caller-supplied
/// eval function. `label_success` is the binary task outcome.
#[derive(Debug, Clone)]
pub struct Episode {
    /// Flattened KV cache, length `D`. Passed verbatim to the eval function.
    pub kv_cache: Vec<f32>,
    /// Task outcome. Used by the caller's eval function, not by the probe math.
    pub label_success: bool,
}

impl Episode {
    /// Construct an episode from a flattened KV cache and a success label.
    #[inline]
    pub fn new(kv_cache: Vec<f32>, label_success: bool) -> Self {
        Self { kv_cache, label_success }
    }
}

/// Binary retention mask over `n_heads` heads.
///
/// `bits[h] == true` means head `h` is **retained** (NOT ablated). The probe
/// builds a measurement row by casting `bits` to `{0.0, 1.0}`.
#[derive(Debug, Clone)]
pub struct AblationMask {
    /// Retention flags. Length MUST equal `n_heads`.
    pub bits: Vec<bool>,
    /// Number of heads the mask spans.
    pub n_heads: usize,
}

impl AblationMask {
    /// Construct an all-retain mask (no head ablated) over `n_heads` heads.
    pub fn all_ones(n_heads: usize) -> Self {
        Self { bits: vec![true; n_heads], n_heads }
    }

    /// Fraction of heads retained: `count(bits==true) / n_heads`.
    #[inline]
    pub fn retention_fraction(&self) -> f32 {
        match self.n_heads {
            0 => 0.0,
            n => {
                let retained = self.bits.iter().filter(|&&b| b).count() as f32;
                retained / n as f32
            }
        }
    }

    /// Number of heads ablated (zeroed) by this mask.
    #[inline]
    pub fn n_ablated(&self) -> usize {
        self.bits.iter().filter(|&&b| !b).count()
    }
}

/// Lasso coefficients aggregated per KV group. Higher score = more important.
///
/// Scores are non-negative (the probe aggregates `|coefficient|`). They are NOT
/// normalized to a probability simplex — downstream gating normalizes via a
/// max-divisor and applies a sigmoid-compatible log-bias (see `gate.rs`).
#[derive(Debug, Clone)]
pub struct KvGroupRanking {
    /// Per-group importance score. Length = `n_groups`.
    pub scores: Vec<f32>,
    /// Number of KV groups (== `n_kv_heads` under the standard GQA mapping).
    pub n_groups: usize,
}

impl KvGroupRanking {
    /// Construct a ranking from a score vector. `n_groups` is inferred.
    pub fn from_scores(scores: Vec<f32>) -> Self {
        let n_groups = scores.len();
        Self { scores, n_groups }
    }

    /// Index of the highest-scoring group, or `None` if empty.
    pub fn argmax(&self) -> Option<usize> {
        let (i, _) = self
            .scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        Some(i)
    }

    /// BLAKE3 content hash of the ranking.
    ///
    /// Hashes `n_groups` (as little-endian u64) followed by the raw little-endian
    /// `f32` bytes of `scores`. This gives a stable, byte-identical digest for a
    /// logically-equal ranking across processes — usable as a content-addressed
    /// cache key (per-task-family CS probe results are BLAKE3-committed in the
    /// zone bundle, per Research 247 §2.4).
    ///
    /// NaN payloads are *not* canonicalized — rankings containing NaN are not
    /// well-defined as cache keys; the probe never emits them (the Lasso path
    /// is finite for finite inputs).
    pub fn blake3_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(self.n_groups as u64).to_le_bytes());
        for &s in &self.scores {
            hasher.update(&s.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

/// Interpolator config for `K(ca)`.
///
/// Paper defaults (Research 247): sparse floor `k_sparse = round(0.035 * D)`,
/// dense ceiling `k_dense = round(0.87 * D)`. The interpolator returns an
/// integer top-K budget in `[k_sparse, k_dense]` for a context-awareness scalar
/// `ca ∈ [0, 1]`. Construct via [`DensityBudget::for_dim`].
#[derive(Debug, Clone, Copy)]
pub struct DensityBudget {
    /// Sparse floor. At `ca = 0` the budget collapses to this.
    pub k_sparse: usize,
    /// Dense ceiling. At `ca = 1` the budget expands to this.
    pub k_dense: usize,
    /// Total dimension `D` the budget is anchored to.
    pub d_total: usize,
}

impl DensityBudget {
    /// Construct from the paper defaults for a given total dimension `D`.
    ///
    /// Guarantees: `k_sparse >= 1`, `k_dense <= d_total`, `k_dense >= k_sparse`.
    pub fn for_dim(d_total: usize) -> Self {
        // Guard: a zero/negative dimension is degenerate; floor to 1 so the
        // interpolator always has a valid [1, d_total] range to clamp into.
        let d_total = d_total.max(1);
        let k_sparse = (0.035_f32 * d_total as f32).round() as usize;
        let k_dense = (0.87_f32 * d_total as f32).round() as usize;
        let k_sparse = k_sparse.max(1);
        let k_dense = k_dense.clamp(k_sparse, d_total);
        Self { k_sparse, k_dense, d_total }
    }
}

impl Default for DensityBudget {
    /// Default anchored at `D = 64`. Callers should construct explicitly via
    /// [`DensityBudget::for_dim`] for production use.
    fn default() -> Self {
        Self::for_dim(64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_fraction_and_ablation_count() {
        let mut m = AblationMask::all_ones(8);
        assert_eq!(m.retention_fraction(), 1.0);
        assert_eq!(m.n_ablated(), 0);
        m.bits[0] = false;
        m.bits[3] = false;
        assert!((m.retention_fraction() - 0.75).abs() < 1e-6);
        assert_eq!(m.n_ablated(), 2);
    }

    #[test]
    fn all_ones_preserves_length() {
        let m = AblationMask::all_ones(16);
        assert_eq!(m.bits.len(), 16);
        assert!(m.bits.iter().all(|&b| b));
    }

    #[test]
    fn density_budget_for_dim_satisfies_invariants() {
        for &d in &[1usize, 2, 8, 16, 32, 64, 128] {
            let b = DensityBudget::for_dim(d);
            assert!(b.k_sparse >= 1, "k_sparse>=1 for d={d}");
            assert!(b.k_dense <= d, "k_dense<=d for d={d}");
            assert!(b.k_dense >= b.k_sparse, "k_dense>=k_sparse for d={d}");
        }
    }

    #[test]
    fn density_budget_paper_anchors_at_64() {
        let b = DensityBudget::for_dim(64);
        // 0.035 * 64 = 2.24 -> round = 2; 0.87 * 64 = 55.68 -> round = 56.
        assert_eq!(b.k_sparse, 2);
        assert_eq!(b.k_dense, 56);
    }

    #[test]
    fn empty_mask_retention_is_zero() {
        let m = AblationMask { bits: vec![], n_heads: 0 };
        assert_eq!(m.retention_fraction(), 0.0);
    }

    #[test]
    fn argmax_returns_top_group() {
        let r = KvGroupRanking::from_scores(vec![0.1, 0.9, 0.2, 0.8]);
        assert_eq!(r.argmax(), Some(1));
    }

    #[test]
    fn blake3_hash_is_deterministic_and_content_addressed() {
        // Same scores → same digest.
        let r1 = KvGroupRanking::from_scores(vec![0.1, 0.9, 0.2, 0.8]);
        let r2 = KvGroupRanking::from_scores(vec![0.1, 0.9, 0.2, 0.8]);
        assert_eq!(r1.blake3_hash(), r2.blake3_hash(), "equal rankings must hash equal");

        // Different scores → different digest.
        let r3 = KvGroupRanking::from_scores(vec![0.1, 0.8, 0.2, 0.8]);
        assert_ne!(r1.blake3_hash(), r3.blake3_hash(), "distinct rankings must hash distinct");

        // Different n_groups with same score bytes → different digest (the u64
        // length prefix participates in the hash).
        let r_short = KvGroupRanking::from_scores(vec![0.1, 0.9]);
        let r_long_prefix = KvGroupRanking::from_scores(vec![0.1, 0.9, 0.0, 0.0]);
        assert_ne!(r_short.blake3_hash(), r_long_prefix.blake3_hash());

        // Fixed expected digest length.
        assert_eq!(r1.blake3_hash().len(), 32);
    }
}
