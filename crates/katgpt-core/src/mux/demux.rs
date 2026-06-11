//! `mux_demux` — deterministic superposition recovery.
//!
//! Given a set of weighted token hypotheses, verifies that they can be
//! uniquely recovered (demultiplexed) from their index.

/// Maximum supported superposition width for stack-allocated demux.
const MAX_DEMUX_K: usize = 32;

/// Result of demultiplexing a superposition back to concrete token IDs.
#[derive(Debug, Clone, PartialEq)]
pub struct DemuxResult {
    /// Ordered token IDs recovered from the superposition.
    pub tokens: Vec<u32>,
    /// Whether the recovery was unique (no duplicate tokens).
    pub is_unique: bool,
}

/// Verifies that a superposition span can be uniquely demultiplexed.
#[derive(Debug, Clone)]
pub struct MuxDemuxVerifier {
    /// Expected superposition width.
    pub k: usize,
}

impl MuxDemuxVerifier {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Demultiplex a superposition: given token IDs and weights,
    /// return them sorted by weight (descending) and verify uniqueness.
    ///
    /// Performs a single allocation (no intermediate buffer), unlike `demux_into`
    /// which clones into the result for caller buffer reuse.
    pub fn demux(&self, tokens: &[u32], weights: &[f32]) -> DemuxResult {
        assert_eq!(tokens.len(), weights.len());
        let n = tokens.len();
        if n == 0 {
            return DemuxResult {
                tokens: Vec::new(),
                is_unique: true,
            };
        }

        // Stack-allocated sort: copy to stack, sort descending by weight.
        let mut pairs: [(u32, f32); MAX_DEMUX_K] = [(0, 0.0); MAX_DEMUX_K];
        let len = n.min(MAX_DEMUX_K);
        for i in 0..len {
            pairs[i] = (tokens[i], weights[i]);
        }
        pairs[..len].sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Single allocation: collect directly into result Vec.
        // O(k²) duplicate scan is fine for k ≤ 32 — trivially fits in L1 cache.
        let mut out_tokens = Vec::with_capacity(len);
        let mut is_unique = true;
        for i in 0..len {
            let token = pairs[i].0;
            if is_unique {
                for j in 0..i {
                    if pairs[j].0 == token {
                        is_unique = false;
                        break;
                    }
                }
            }
            out_tokens.push(token);
        }

        DemuxResult {
            tokens: out_tokens,
            is_unique,
        }
    }

    /// Zero-alloc demultiplexing into a caller-provided buffer.
    ///
    /// `out_tokens` must have capacity >= `tokens.len()`.
    /// Returns a `DemuxResult` whose `tokens` field is a clone of the written
    /// slice (so the caller retains ownership of the buffer).
    pub fn demux_into(
        &self,
        tokens: &[u32],
        weights: &[f32],
        out_tokens: &mut Vec<u32>,
    ) -> DemuxResult {
        assert_eq!(tokens.len(), weights.len());
        let n = tokens.len();
        if n == 0 {
            return DemuxResult {
                tokens: Vec::new(),
                is_unique: true,
            };
        }

        // Stack-allocated sort: copy to stack, sort descending by weight.
        let mut pairs: [(u32, f32); MAX_DEMUX_K] = [(0, 0.0); MAX_DEMUX_K];
        let len = n.min(MAX_DEMUX_K);
        for i in 0..len {
            pairs[i] = (tokens[i], weights[i]);
        }
        pairs[..len].sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Extract sorted tokens into caller buffer + inline duplicate check.
        // O(k²) scan is fine for k ≤ 32 — trivially fits in L1 cache.
        out_tokens.clear();
        let mut is_unique = true;
        for i in 0..len {
            let token = pairs[i].0;
            if is_unique {
                for j in 0..i {
                    if pairs[j].0 == token {
                        is_unique = false;
                        break;
                    }
                }
            }
            out_tokens.push(token);
        }

        DemuxResult {
            tokens: out_tokens.clone(),
            is_unique,
        }
    }
}

/// Convenience function: demultiplex a superposition span.
pub fn mux_demux(tokens: &[u32], weights: &[f32]) -> DemuxResult {
    let verifier = MuxDemuxVerifier::new(tokens.len());
    verifier.demux(tokens, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demux_sorts_by_weight() {
        let tokens = vec![3, 1, 2, 0];
        let weights = vec![0.1, 0.9, 0.5, 0.3];
        let result = mux_demux(&tokens, &weights);
        assert_eq!(result.tokens, vec![1, 2, 0, 3]);
        assert!(result.is_unique);
    }

    #[test]
    fn demux_detects_duplicates() {
        let tokens = vec![1, 1, 2];
        let weights = vec![0.5, 0.3, 0.2];
        let result = mux_demux(&tokens, &weights);
        assert!(!result.is_unique);
    }

    #[test]
    fn demux_empty() {
        let result = mux_demux(&[], &[]);
        assert!(result.tokens.is_empty());
        assert!(result.is_unique);
    }
}
