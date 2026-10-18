//! Offline anchor scoring from decode logs (Plan 602 T2.1, distilled from
//! arXiv:2609.20751 §5 — the sparse anchor set analog, Research 575).
//!
//! # The anchor claim
//!
//! The paper factorizes the joint as
//! `q(x) = q(x_A) · Π_{i∉A} q(x_i | x_<i, x_A)` — a sparse set A of anchor
//! positions carrying future context, with the remainder decoded as a
//! conditional chain off the anchors. If |A| ≪ L, a hybrid architecture can
//! keep only the anchors in the expensive bidirectional state. Which
//! positions ARE anchors is an empirical property of the model+task — these
//! scorers rank it **from existing decode logs, weights untouched, any
//! model**:
//!
//! - **first-unmask-in-block frequency** (over N logged trajectories, the
//!   π emitted by Plan 602 T1.3): positions that consistently reveal first
//!   lead their block — the behavioral signature of carrying future context;
//! - **masked-position entropy** (the Issue-587 proposal-law rows `q`, one
//!   row-major `[positions × vocab]` block): positions the model is
//!   CONFIDENT about early (low entropy at commit time) are the ones whose
//!   values constrain the rest — the informational signature.
//!
//! The combined [`anchor_scores`] maps both to [0,1] per position
//! (min-max across the block, degenerate range → 0.5 "no discrimination")
//! and averages: high frequency + low entropy = high anchor score.
//!
//! Offline analysis lane — allocating (returns `Vec<f32>`), pure, no deps.
//! The GOAT arm for the T2.2/T3 lane is the planted-anchor fixture:
//! a uniquely-determining token (point-mass q, always-first π) must rank #1.
//!
//! Plan 602 T2.1: the offline anchor scorer (decode-log ranking toward the
//! paper's sparse anchor set A — same feature, root-level per the plan target).

use crate::dllm::UNMASKED_NEVER;

/// Per-position entropy (nats) of the proposal-law rows.
///
/// `q_rows` is the Issue-587 capture shape: row-major `[positions × vocab]`
/// (partial rows are scored up to `q_rows.len() / vocab` positions).
/// Convention `0·ln 0 = 0`; non-finite or non-positive masses contribute
/// nothing (a malformed row reads as entropy 0 — confident — so callers
/// feeding unnormalized garbage get visible nonsense, not a panic).
pub fn masked_entropies(q_rows: &[f32], vocab: usize) -> Vec<f32> {
    assert!(vocab > 0, "vocab must be positive");
    let n = q_rows.len() / vocab;
    (0..n)
        .map(|row| {
            let q = &q_rows[row * vocab..(row + 1) * vocab];
            let mut h = 0.0f32;
            for &p in q {
                if p > 0.0 && p.is_finite() {
                    h -= p * p.ln();
                }
            }
            h
        })
        .collect()
}

/// Per-position frequency of being the **first unmask in its block**, over
/// N logged trajectories (the Plan 602 T1.3 π buffers, equal length).
///
/// A position counts as first when its step equals the earliest non-sentinel
/// step in its `block_size` window — parallel reveals at the same earliest
/// step ALL count (they were jointly first; joint anchors are the
/// block-parallel regime's normal shape). Blocks whose every entry is the
/// never-committed sentinel contribute nothing. Empty `trajs` returns all
/// zeros (no evidence, not a lie).
pub fn first_unmask_frequencies(trajs: &[&[u32]], block_size: usize) -> Vec<f32> {
    assert!(block_size > 0, "block_size must be positive");
    let n = trajs.first().map(|t| t.len()).unwrap_or(0);
    let mut counts = vec![0u32; n];
    for traj in trajs {
        debug_assert_eq!(
            traj.len(),
            n,
            "all trajectories must share one length (one decode shape)"
        );
        for (b, block) in traj.chunks(block_size).enumerate() {
            let base = b * block_size;
            let earliest = block.iter().copied().filter(|&s| s != UNMASKED_NEVER).min();
            if let Some(m) = earliest {
                for (j, &s) in block.iter().enumerate() {
                    if s == m {
                        counts[base + j] += 1;
                    }
                }
            }
        }
    }
    let denom = trajs.len() as f32;
    counts.into_iter().map(|c| c as f32 / denom).collect()
}

/// Min-max normalize to [0,1]; a degenerate range (no spread) maps every
/// entry to 0.5 — "this component discriminates nothing", never a fake 0/1.
fn minmax_normalized(v: &[f32]) -> Vec<f32> {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &x in v {
        if x.is_finite() {
            lo = lo.min(x);
            hi = hi.max(x);
        }
    }
    if !lo.is_finite() || hi <= lo {
        return vec![0.5; v.len()];
    }
    v.iter().map(|&x| (x - lo) / (hi - lo)).collect()
}

/// Combined per-position anchor score in [0,1]:
/// `0.5 · first_unmask_frequency_normalized + 0.5 · (1 − entropy_normalized)`.
///
/// High score = revealed first consistently AND committed under a confident
/// (low-entropy) proposal law — the behavioral + informational anchor
/// signatures composed. See the module doc for the factorization this
/// ranks toward; the planted-anchor GOAT arm lives in the tests.
pub fn anchor_scores(
    trajs: &[&[u32]],
    q_rows: &[f32],
    vocab: usize,
    block_size: usize,
) -> Vec<f32> {
    let freq = first_unmask_frequencies(trajs, block_size);
    let ent = masked_entropies(q_rows, vocab);
    let freq_n = minmax_normalized(&freq);
    let ent_n = minmax_normalized(&ent);
    freq_n
        .iter()
        .zip(ent_n.iter())
        .map(|(&f, &e)| 0.5 * f + 0.5 * (1.0 - e))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: usize = 8;

    fn uniform_q(n: usize) -> Vec<f32> {
        vec![1.0 / VOCAB as f32; n * VOCAB]
    }

    /// Point-mass row for `pos` (greedy draft: the ExactQ policy's law).
    fn point_mass_q(n: usize, pos: usize) -> Vec<f32> {
        let mut q = uniform_q(n);
        let start = pos * VOCAB;
        for (v, slot) in q[start..start + VOCAB].iter_mut().enumerate() {
            *slot = if v == 1 { 1.0 } else { 0.0 };
        }
        q
    }

    #[test]
    fn uniform_rows_have_log_vocab_entropy() {
        let h = masked_entropies(&uniform_q(4), VOCAB);
        assert_eq!(h.len(), 4);
        let expected = (VOCAB as f32).ln();
        for x in h {
            assert!((x - expected).abs() < 1e-5, "got {x}");
        }
    }

    #[test]
    fn point_mass_has_zero_entropy() {
        let h = masked_entropies(&point_mass_q(3, 1), VOCAB);
        assert_eq!(h.len(), 3);
        assert!(h[1].abs() < 1e-6);
        let uniform = (VOCAB as f32).ln();
        assert!((h[0] - uniform).abs() < 1e-5);
    }

    #[test]
    fn empty_and_partial_rows() {
        assert!(masked_entropies(&[], VOCAB).is_empty());
        // Partial trailing row (len 3·vocab/2) scores floor(1.5)=1 row.
        assert_eq!(masked_entropies(&[0.5f32; VOCAB * 3 / 2], VOCAB).len(), 1);
    }

    #[test]
    fn first_unmask_counts_ties_and_skips_sentinel_blocks() {
        // Block of 4: positions 1 and 2 tie at step 0 (parallel reveal),
        // position 0 at step 2, position 3 never committed.
        let t1 = vec![2u32, 0, 0, UNMASKED_NEVER];
        // Block of 4 all-sentinel: contributes nothing.
        let t2 = vec![UNMASKED_NEVER; 4];
        let trajs: Vec<&[u32]> = vec![&t1, &t2];
        let f = first_unmask_frequencies(&trajs, 4);
        assert_eq!(f.len(), 4);
        // t1: firsts are positions 1,2 → +0.5 each. t2: nothing.
        assert_eq!(f[0], 0.0);
        assert_eq!(f[1], 0.5);
        assert_eq!(f[2], 0.5);
        assert_eq!(f[3], 0.0);
    }

    #[test]
    fn first_unmask_empty_trajs_is_zero() {
        let f = first_unmask_frequencies(&[], 4);
        assert!(f.is_empty());
    }

    #[test]
    fn planted_anchor_ranks_first() {
        // GOAT arm (the T2.2/T3 gate): position 3 is uniquely determining —
        // point-mass law + strictly-earliest reveal in every trajectory.
        let n = 8;
        let mk_traj = |k: u32| -> Vec<u32> {
            let mut pi: Vec<u32> = (0..n as u32).map(|i| 1 + ((i + k) % 3)).collect();
            pi[3] = 0; // strictly earliest, every run.
            pi
        };
        let owned: Vec<Vec<u32>> = (0..10u32).map(mk_traj).collect();
        let trajs: Vec<&[u32]> = owned.iter().map(|v| v.as_slice()).collect();
        let q = point_mass_q(n, 3);

        let scores = anchor_scores(&trajs, &q, VOCAB, n);
        assert_eq!(scores.len(), n);

        // Position 3: freq 1.0 (normalized 1.0) + entropy 0 (normalized 0,
        // 1−0 = 1.0) → 1.0. Every other position: freq 0, entropy max → 0.0.
        assert!((scores[3] - 1.0).abs() < 1e-5, "anchor score {}", scores[3]);
        for (i, &s) in scores.iter().enumerate() {
            if i != 3 {
                assert!(s < 1e-5, "position {i} scored {s} — anchor must be unique");
            }
        }
        // And it IS rank #1.
        let top = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(top, 3);
    }

    #[test]
    fn degenerate_inputs_score_half() {
        // One trajectory, all-sentinel; no q rows: both components
        // discriminate nothing → every score 0.5 (honest no-evidence).
        let pi = vec![UNMASKED_NEVER; 4];
        let trajs: Vec<&[u32]> = vec![&pi];
        let scores = anchor_scores(&trajs, &[], VOCAB, 4);
        assert!(scores.iter().all(|&s| (s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn uniform_anchor_scores_are_flat() {
        // Identical π + identical uniform q: no spread → all 0.5.
        let owned: Vec<Vec<u32>> = vec![vec![0u32; 4]; 5];
        let trajs: Vec<&[u32]> = owned.iter().map(|v| v.as_slice()).collect();
        let scores = anchor_scores(&trajs, &uniform_q(4), VOCAB, 4);
        assert!(scores.iter().all(|&s| (s - 0.5).abs() < 1e-6));
    }
}
