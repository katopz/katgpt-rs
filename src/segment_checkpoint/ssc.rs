//! SSC — Sparse Selective Caching variant (Plan 226 Phase 2).
//!
//! Top-k segment selection for reduced retrieval overhead.
//! Uses `select_nth_unstable` for O(N) partial partition instead of O(N log N) full sort.
//! Feature-gated behind `ssc_spec_draft`.

// ---------------------------------------------------------------------------
// Task 5: Top-k Segment Selection (pure gate-based)
// ---------------------------------------------------------------------------

/// Select top-k segments by gate relevance score.
///
/// Uses `select_nth_unstable` for O(N) partition instead of O(N log N) full sort.
/// Returns results sorted by gate descending.
pub fn top_k_segments(gates: &[(u32, f32)], k: usize) -> Vec<(u32, f32)> {
    if gates.is_empty() || k == 0 {
        return Vec::new();
    }
    if gates.len() <= k {
        let mut out = gates.to_vec();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        return out;
    }

    // Index into original gates array to recover segment_id
    let mut indexed: Vec<(usize, f32)> = gates
        .iter()
        .enumerate()
        .map(|(i, &(_, g))| (i, g))
        .collect();

    // O(N) partial partition — everything at indices < k is >= pivot
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top-k, then sort descending for deterministic ordering
    let mut result: Vec<(u32, f32)> = indexed[..k]
        .iter()
        .map(|&(idx, score)| (gates[idx].0, score))
        .collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Compute gates (sigmoid dot-product) and select top-k in one pass.
///
/// More efficient than computing all gates then selecting separately
/// when you only need the top-k results.
pub fn compute_and_select_top_k(
    query: &[f32],
    summaries: &[(u32, &[f32])],
    k: usize,
) -> Vec<(u32, f32)> {
    if summaries.is_empty() || k == 0 {
        return Vec::new();
    }

    let gates: Vec<(u32, f32)> = summaries
        .iter()
        .map(|&(id, summary)| {
            let dot: f32 = query.iter().zip(summary.iter()).map(|(q, s)| q * s).sum();
            let gate = 1.0 / (1.0 + (-dot).exp()); // sigmoid, NOT softmax
            (id, gate)
        })
        .collect();

    top_k_segments(&gates, k)
}

// ---------------------------------------------------------------------------
// Task 6: SSC-Enhanced Speculative Drafter
// ---------------------------------------------------------------------------

/// SSC-enhanced speculative drafter.
///
/// Feeds top-k cached segment summaries as additional context to the drafter,
/// producing a small sigmoid bias on draft logits informed by long-range context.
pub struct SscDrafter {
    /// Number of top segments to use (capped at 8, paper shows diminishing returns).
    pub k: usize,
    /// Cached segment summaries from last `update_context` call.
    context_summaries: Vec<Vec<f32>>,
}

impl SscDrafter {
    /// Create a new drafter. `k` is capped at 8 per paper findings.
    pub fn new(k: usize) -> Self {
        Self {
            k: k.min(8).max(1),
            context_summaries: Vec::new(),
        }
    }

    /// Update internal context from top-k segments matching the query.
    pub fn update_context(&mut self, query: &[f32], summaries: &[(u32, &[f32])]) {
        if summaries.is_empty() {
            self.context_summaries.clear();
            return;
        }

        let top_k = compute_and_select_top_k(query, summaries, self.k);
        self.context_summaries = top_k
            .iter()
            .filter_map(|&(id, _)| {
                summaries
                    .iter()
                    .find(|&&(sid, _)| sid == id)
                    .map(|&(_, s)| s.to_vec())
            })
            .collect();
    }

    /// Enhance draft logits with a small sigmoid bias from segment context.
    ///
    /// Adds a lightweight context signal (weight 0.1) to avoid overwhelming
    /// the base draft distribution while improving long-range coherence.
    pub fn enhance_draft(&self, draft_logits: &mut [f32]) {
        if self.context_summaries.is_empty() {
            return;
        }

        let dim = draft_logits.len().min(self.context_summaries[0].len());
        if dim == 0 {
            return;
        }

        // Average context vector across top-k summaries
        let n = self.context_summaries.len() as f32;
        let avg_context: Vec<f32> = (0..dim)
            .map(|i| {
                self.context_summaries
                    .iter()
                    .map(|s| s.get(i).copied().unwrap_or(0.0))
                    .sum::<f32>()
                    / n
            })
            .collect();

        // Small sigmoid bias (0.1 weight) per logit
        for (i, logit) in draft_logits.iter_mut().enumerate() {
            if i < dim {
                let dot = *logit * avg_context[i];
                let bias = 1.0 / (1.0 + (-dot).exp()); // sigmoid
                *logit += 0.1 * bias;
            }
        }
    }

    /// Number of currently cached context summaries.
    pub fn context_len(&self) -> usize {
        self.context_summaries.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Task 5 tests --

    #[test]
    fn test_top_k_selects_highest_gates() {
        let gates = vec![(0u32, 0.1), (1u32, 0.9), (2u32, 0.5), (3u32, 0.8)];
        let top = top_k_segments(&gates, 2);
        assert_eq!(top.len(), 2);
        // Sorted descending by gate
        assert!((top[0].1 - 0.9).abs() < 1e-6);
        assert!((top[1].1 - 0.8).abs() < 1e-6);
        assert_eq!(top[0].0, 1); // highest gate
        assert_eq!(top[1].0, 3); // second highest
    }

    #[test]
    fn test_top_k_all_when_k_exceeds_len() {
        let gates = vec![(0u32, 0.1), (1u32, 0.9)];
        let top = top_k_segments(&gates, 5);
        assert_eq!(top.len(), 2);
        // Still sorted descending
        assert!(top[0].1 >= top[1].1);
    }

    #[test]
    fn test_top_k_empty() {
        let gates: Vec<(u32, f32)> = vec![];
        let top = top_k_segments(&gates, 3);
        assert!(top.is_empty());
    }

    #[test]
    fn test_top_k_zero_k() {
        let gates = vec![(0u32, 0.5)];
        let top = top_k_segments(&gates, 0);
        assert!(top.is_empty());
    }

    #[test]
    fn test_compute_and_select() {
        let query = vec![0.5f32; 8];
        let s1 = vec![0.8f32; 8]; // high dot → high gate
        let s2 = vec![0.1f32; 8]; // low dot → low gate
        let s3 = vec![0.6f32; 8];
        let summaries: Vec<(u32, &[f32])> = vec![(0, &s1), (1, &s2), (2, &s3)];

        let top = compute_and_select_top_k(&query, &summaries, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 0); // highest dot(0.5, 0.8) = 3.2
    }

    // -- Task 6 tests --

    #[test]
    fn test_ssc_drafter_enhances_logits() {
        let mut drafter = SscDrafter::new(2);
        let query = vec![0.5f32; 4];
        let s1 = vec![0.8f32; 4];
        let summaries: Vec<(u32, &[f32])> = vec![(0, &s1)];
        drafter.update_context(&query, &summaries);

        let mut logits = vec![0.3f32; 4];
        let original = logits.clone();
        drafter.enhance_draft(&mut logits);
        assert_ne!(logits, original, "logits should be modified");
        // All should be slightly higher (positive sigmoid bias added)
        for (l, o) in logits.iter().zip(original.iter()) {
            assert!(*l > *o, "logit should increase with positive context bias");
        }
    }

    #[test]
    fn test_ssc_drafter_k_capped_at_8() {
        let drafter = SscDrafter::new(100);
        assert_eq!(drafter.k, 8);
    }

    #[test]
    fn test_ssc_drafter_k_minimum_1() {
        let drafter = SscDrafter::new(0);
        assert_eq!(drafter.k, 1);
    }

    #[test]
    fn test_ssc_drafter_empty_summaries() {
        let mut drafter = SscDrafter::new(2);
        let query = vec![0.5f32; 4];
        let summaries: Vec<(u32, &[f32])> = vec![];
        drafter.update_context(&query, &summaries);

        let mut logits = vec![0.3f32; 4];
        let original = logits.clone();
        drafter.enhance_draft(&mut logits);
        assert_eq!(logits, original, "should not modify logits with no context");
    }

    #[test]
    fn test_ssc_drafter_context_len() {
        let mut drafter = SscDrafter::new(2);
        assert_eq!(drafter.context_len(), 0);

        let query = vec![0.5f32; 4];
        let s1 = vec![0.8f32; 4];
        let s2 = vec![0.3f32; 4];
        let summaries: Vec<(u32, &[f32])> = vec![(0, &s1), (1, &s2)];
        drafter.update_context(&query, &summaries);
        assert_eq!(drafter.context_len(), 2);
    }
}
