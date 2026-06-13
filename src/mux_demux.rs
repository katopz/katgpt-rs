//! MuxDemux Verifier — deterministic demultiplexer that recovers token spans from superposition.
//!
//! Given a logit vector representing a multiplexed superposition of K tokens, `mux_demux`
//! deterministically recovers the original token IDs by:
//! 1. Extracting the top-k peaks from the logit distribution
//! 2. Verifying that they follow geometric decay ordering
//! 3. Returning the token IDs if valid, or `None` if not a valid superposition
//!
//! # Design
//!
//! This is a standalone module gated by `mux_demux` only — no `validator` dependency needed.
//! Pure math, no allocations beyond the output vec, WASM-compatible.
//!
//! (Research 158, MUX)

// ── Public API ─────────────────────────────────────────────────────

/// Recover token span from superposition via deterministic demultiplexing.
///
/// Extracts top-k peaks from `logits`, verifies geometric decay ordering with
/// the given `decay` ratio. Returns `Some(token_ids)` if the distribution is a
/// valid multiplexed superposition, `None` otherwise.
///
/// # Arguments
///
/// * `logits` — raw logit vector over the vocabulary
/// * `k` — number of superposed tokens to recover
/// * `decay` — expected geometric decay ratio between consecutive peaks
///
/// # Returns
///
/// `Some(Vec<usize>)` with token IDs in descending logit order, or `None`
/// if the distribution does not exhibit valid superposition.
pub fn mux_demux(logits: &[f32], k: usize, decay: f32) -> Option<Vec<usize>> {
    if logits.is_empty() || k == 0 {
        return None;
    }

    let k = k.min(logits.len());
    let peaks = extract_top_k(logits, k);

    // Need at least 2 peaks for meaningful superposition
    if peaks.len() < 2 {
        return None;
    }

    let top_val = peaks[0].1;

    // Verify geometric decay ordering with 50% tolerance
    let mut decay_acc = decay;
    for i in 1..peaks.len() {
        let expected = top_val * decay_acc;
        let actual = peaks[i].1;
        let tolerance = expected.abs() * 0.5;
        if (actual - expected).abs() > tolerance {
            return None;
        }
        decay_acc *= decay;
    }

    // Reject collapse: top peak should not be >20x the second
    if peaks.len() >= 2 && peaks[1].1.abs() > 1e-8 {
        let ratio = peaks[0].1.abs() / peaks[1].1.abs();
        if ratio > 20.0 {
            return None;
        }
    }

    Some(peaks.into_iter().map(|(idx, _)| idx).collect())
}

/// Simulate multiplexed logits for testing.
///
/// Given a set of token IDs, generates a logit vector where those tokens have
/// geometrically decaying values and all other positions are near zero.
///
/// # Arguments
///
/// * `tokens` — token IDs to place in superposition (first = strongest)
/// * `vocab_size` — total vocabulary size
/// * `decay` — geometric decay ratio between consecutive tokens
///
/// # Returns
///
/// A logit vector of length `vocab_size`.
pub fn simulate_mux_logits(tokens: &[usize], vocab_size: usize, decay: f32) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab_size];
    let base = 10.0f32;
    for (i, &tok) in tokens.iter().enumerate() {
        if tok < vocab_size {
            logits[tok] = base * decay.powi(i as i32);
        }
    }
    logits
}

/// Verifier struct for stateful superposition recovery.
///
/// Wraps `mux_demux` with configurable parameters for reuse across decode steps.
pub struct MuxDemuxVerifier {
    /// Number of superposed tokens to recover.
    pub k: usize,
    /// Expected geometric decay ratio.
    pub decay: f32,
}

impl MuxDemuxVerifier {
    /// Create a new verifier with the given superposition width and decay.
    pub fn new(k: usize, decay: f32) -> Self {
        Self { k, decay }
    }

    /// Recover token span from the given logit vector.
    pub fn verify(&self, logits: &[f32]) -> Option<Vec<usize>> {
        mux_demux(logits, self.k, self.decay)
    }
}

impl Default for MuxDemuxVerifier {
    fn default() -> Self {
        Self { k: 5, decay: 0.9 }
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Extract top-k (index, value) pairs sorted by descending value.
#[inline]
fn extract_top_k(data: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.min(data.len());
    if k == 0 {
        return Vec::new();
    }

    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k);

    for (idx, &val) in data.iter().enumerate() {
        if top.len() < k {
            insert_sorted(&mut top, idx, val);
        } else if val > top[k - 1].1 {
            let last = top.len() - 1;
            top[last] = (idx, val);
            bubble_up(&mut top, last);
        }
    }

    top
}

#[inline]
fn insert_sorted(buf: &mut Vec<(usize, f32)>, idx: usize, val: f32) {
    let pos = buf.partition_point(|&(_, v)| v >= val);
    buf.insert(pos, (idx, val));
}

#[inline]
fn bubble_up(buf: &mut [(usize, f32)], mut pos: usize) {
    while pos > 0 && buf[pos].1 > buf[pos - 1].1 {
        buf.swap(pos, pos - 1);
        pos -= 1;
    }
}

// ---------------------------------------------------------------------------
// MUX-RCD Fusion: Superposition-weighted residual (Plan 258 Phase 4)
// ---------------------------------------------------------------------------

/// Compute MUX-RCD residual from DDTree superposition paths.
///
/// Instead of single-step marginal, weights residuals across DDTree branches:
/// `Δ_i = Σ_path score_path * Δ_path_i`
///
/// This requires DDTree node scores to weight the residual contributions
/// from multiple hypothesis paths.
///
/// # Arguments
/// * `path_scores` — quality score for each path (will be normalized)
/// * `path_marginals` — marginal distribution per path per position: `[path][pos * vocab..]`
/// * `wte` — flat embedding matrix `[vocab_size * n_embd]`
/// * `n_embd` — embedding dimension
/// * `position` — position to compute residual for
/// * `vocab_size` — vocabulary size
/// * `out` — output buffer `[n_embd]`, caller-allocated
#[cfg(all(feature = "mux_demux", feature = "rcd_residual"))]
pub fn compute_mux_residual(
    path_scores: &[f32],
    path_marginals: &[&[f32]],
    wte: &[f32],
    n_embd: usize,
    position: usize,
    vocab_size: usize,
    out: &mut [f32],
) {
    debug_assert!(out.len() >= n_embd);
    out[..n_embd].fill(0.0f32);

    if path_scores.is_empty() || path_marginals.is_empty() {
        return;
    }

    // Normalize path scores to probabilities
    let total_score: f32 = path_scores.iter().copied().filter(|&s| s > 0.0).sum();
    if total_score <= 0.0 {
        return;
    }

    let n_paths = path_scores.len().min(path_marginals.len());

    for path_idx in 0..n_paths {
        let score = path_scores[path_idx];
        if score <= 0.0 {
            continue;
        }
        let weight = score / total_score;
        let marginals = path_marginals[path_idx];
        let offset = position * vocab_size;

        if offset + vocab_size > marginals.len() {
            continue;
        }

        let marginals_p = &marginals[offset..offset + vocab_size];

        // Accumulate weighted codebook sum: weight * Σ_j p_ij * E_j
        for j in 0..vocab_size {
            let p_j = marginals_p[j];
            if p_j < 1e-10 {
                continue;
            }
            let emb_start = j * n_embd;
            let emb_end = emb_start + n_embd;
            if emb_end > wte.len() {
                break;
            }
            let contrib = weight * p_j;
            for (k, &e) in wte[emb_start..emb_end].iter().enumerate() {
                out[k] += contrib * e;
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mux_demux_roundtrip() {
        let tokens = vec![10, 17, 24, 31, 38];
        let logits = simulate_mux_logits(&tokens, 100, 0.9);
        let recovered = mux_demux(&logits, 5, 0.9);
        assert_eq!(recovered, Some(tokens));
    }

    #[test]
    fn test_mux_demux_roundtrip_small_k() {
        let tokens = vec![5, 12, 20];
        let logits = simulate_mux_logits(&tokens, 50, 0.8);
        let recovered = mux_demux(&logits, 3, 0.8);
        assert_eq!(recovered, Some(tokens));
    }

    #[test]
    fn test_mux_demux_rejects_noise() {
        let logits = vec![0.5f32; 100]; // uniform
        let result = mux_demux(&logits, 5, 0.9);
        assert!(result.is_none(), "uniform noise should be rejected");
    }

    #[test]
    fn test_mux_demux_rejects_collapse() {
        let mut logits = vec![0.0f32; 100];
        logits[0] = 100.0;
        logits[1] = 0.01;
        let result = mux_demux(&logits, 5, 0.9);
        assert!(
            result.is_none(),
            "collapsed distribution should be rejected"
        );
    }

    #[test]
    fn test_mux_demux_empty_logits() {
        assert_eq!(mux_demux(&[], 5, 0.9), None);
    }

    #[test]
    fn test_mux_demux_zero_k() {
        assert_eq!(mux_demux(&[1.0, 2.0], 0, 0.9), None);
    }

    #[test]
    fn test_simulate_mux_logits() {
        let tokens = vec![0, 1, 2];
        let logits = simulate_mux_logits(&tokens, 10, 0.5);
        assert!(logits[0] > logits[1]);
        assert!(logits[1] > logits[2]);
        // Non-token positions should be 0
        assert_eq!(logits[3], 0.0);
    }

    #[test]
    fn test_mux_demux_verifier() {
        let verifier = MuxDemuxVerifier::new(3, 0.8);
        let tokens = vec![5, 12, 20];
        let logits = simulate_mux_logits(&tokens, 50, 0.8);
        assert_eq!(verifier.verify(&logits), Some(tokens));
    }

    #[test]
    fn test_mux_demux_verifier_default() {
        let verifier = MuxDemuxVerifier::default();
        assert_eq!(verifier.k, 5);
        assert!((verifier.decay - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_mux_demux_k_exceeds_vocab() {
        let logits = vec![1.0, 2.0];
        // k=10 but only 2 logits — should still work with k clamped to 2
        let result = mux_demux(&logits, 10, 0.9);
        // With only 2 elements, if the second is within tolerance of decay,
        // it should return Some. But [1.0, 2.0] has index 1 as dominant,
        // so peaks = [(1, 2.0), (0, 1.0)], expected[1] = 2.0 * 0.9 = 1.8,
        // actual[1] = 1.0, tolerance = 0.9. |1.0 - 1.8| = 0.8 < 0.9 ✓
        assert!(result.is_some());
        let tokens = result.unwrap();
        assert_eq!(tokens[0], 1); // highest logit
        assert_eq!(tokens[1], 0);
    }
}
