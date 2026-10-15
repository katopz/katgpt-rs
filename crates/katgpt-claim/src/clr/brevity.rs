//! Long2Short brevity tiebreak — pure algorithm, no quality change (Plan 284 T1.6).
//!
//! Given a set of clusters tied (within `eps`) on `total_reliability`, pick the
//! one whose representative trajectory used the fewest tokens/steps. This is
//! the Long2Short zero-sum tiebreak from Research 255.
//!
//! **Issue 940 (TOL) — the `eps` band is BY-DESIGN LENIENCY, documented.** A
//! cluster up to `tiebreak_eps = 1e-3` below the max reliability CAN win when
//! its representative is shorter — the earlier "never changes which answer
//! wins" claim was true only of the strict ordering, not of the band. The
//! band is well-defined per call (candidates are `>= max - eps` against THIS
//! call's max — a threshold band, not a pairwise band, so the classic
//! non-transitivity does not arise within a call), and the in-band rule is
//! deterministic (min tokens, first-encountered). Pins:
//! `band_leniency_is_by_design` (0.9e-3 below max + shorter wins) and
//! `outside_the_band_the_max_wins` (1.1e-3 below loses).
//!
//! The function is allocation-free beyond the linear scan: it returns an index
//! into the `candidates` slice and does not construct any intermediate `Vec`.

use crate::clr::types::{Cluster, Trajectory};

/// Long2Short tiebreak among reliability-tied clusters.
///
/// Algorithm:
///   1. Find `max` `total_reliability` across `candidates`.
///   2. Consider only candidates with `total_reliability >= max - eps`.
///   3. Among those, return the index of the candidate whose representative
///      trajectory (in `trajectories[candidate.representative_idx]`) has the
///      smallest `tokens_or_steps`. Ties broken by first-encountered order.
///
/// # Panics
///
/// Panics if `candidates` is empty — this is a programmer error (the caller
/// should not invoke a tiebreak with no candidates).
///
/// In debug builds, also panics if any `candidate.representative_idx` is out of
/// bounds for `trajectories`.
pub fn brevity_tiebreak<T>(
    candidates: &[&Cluster<T>],
    trajectories: &[Trajectory<T>],
    eps: f32,
) -> usize {
    assert!(
        !candidates.is_empty(),
        "brevity_tiebreak: empty candidates slice (programmer error)"
    );

    // Step 1: scan for max reliability. Single pass, no allocation.
    let mut max_reliability = f32::NEG_INFINITY;
    for c in candidates {
        if c.total_reliability > max_reliability {
            max_reliability = c.total_reliability;
        }
    }
    let threshold = max_reliability - eps;

    // Step 2 + 3: among candidates within `eps` of max, find the one with the
    // fewest tokens/steps on its representative trajectory. First-encountered
    // wins ties (strict `<` comparison).
    let mut best_idx = 0usize;
    let mut best_tokens = usize::MAX;
    for (i, c) in candidates.iter().enumerate() {
        if c.total_reliability < threshold {
            continue;
        }
        debug_assert!(
            c.representative_idx < trajectories.len(),
            "brevity_tiebreak: representative_idx {} out of bounds for {} trajectories",
            c.representative_idx,
            trajectories.len()
        );
        // Defensive guard: in release builds without debug_asserts, a
        // programmer error here would otherwise index out of bounds. Clamp
        // instead of panicking to keep the hot path total.
        if c.representative_idx >= trajectories.len() {
            continue;
        }
        let tokens = trajectories[c.representative_idx].tokens_or_steps;
        if tokens < best_tokens {
            best_tokens = tokens;
            best_idx = i;
        }
    }

    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clr::types::Cluster;

    fn cluster<T: Clone>(
        outcome: T,
        reliability: f32,
        rep_idx: usize,
        members: &[usize],
    ) -> Cluster<T> {
        Cluster {
            outcome,
            total_reliability: reliability,
            representative_idx: rep_idx,
            member_indices: members.to_vec(),
        }
    }

    #[test]
    fn picks_shorter_representative_on_tie() {
        // Two clusters tied on reliability within eps.
        // Cluster A rep (idx 0) has 100 tokens; cluster B rep (idx 1) has 50.
        let trajectories: Vec<Trajectory<()>> = vec![
            Trajectory {
                outcome: (),
                tokens_or_steps: 100,
                claims: vec![],
                log_probs: None,
            },
            Trajectory {
                outcome: (),
                tokens_or_steps: 50,
                claims: vec![],
                log_probs: None,
            },
        ];
        let c0 = cluster((), 0.8, 0, &[0]);
        let c1 = cluster((), 0.8, 1, &[1]);
        let candidates: Vec<&Cluster<()>> = vec![&c0, &c1];
        // eps is large enough that the tie stands.
        let idx = brevity_tiebreak(&candidates, &trajectories, 0.01);
        // Should pick index 1 (cluster B, 50 tokens).
        assert_eq!(idx, 1, "should pick shorter representative");
    }

    #[test]
    fn respects_reliability_when_not_tied() {
        // Cluster A has higher reliability → wins outright, even though longer.
        let trajectories: Vec<Trajectory<()>> = vec![
            Trajectory {
                outcome: (),
                tokens_or_steps: 200,
                claims: vec![],
                log_probs: None,
            },
            Trajectory {
                outcome: (),
                tokens_or_steps: 50,
                claims: vec![],
                log_probs: None,
            },
        ];
        let c0 = cluster((), 0.9, 0, &[0]);
        let c1 = cluster((), 0.5, 1, &[1]);
        let candidates: Vec<&Cluster<()>> = vec![&c0, &c1];
        let idx = brevity_tiebreak(&candidates, &trajectories, 0.01);
        // c0 (0.9) is above c1 (0.5) by more than eps → c0 wins despite length.
        assert_eq!(idx, 0);
    }

    #[test]
    fn first_encountered_wins_token_tie() {
        // Both reps have the same token count → first-encountered (index 0) wins.
        let trajectories: Vec<Trajectory<()>> = vec![
            Trajectory {
                outcome: (),
                tokens_or_steps: 50,
                claims: vec![],
                log_probs: None,
            },
            Trajectory {
                outcome: (),
                tokens_or_steps: 50,
                claims: vec![],
                log_probs: None,
            },
        ];
        let c0 = cluster((), 0.7, 0, &[0]);
        let c1 = cluster((), 0.7, 1, &[1]);
        let candidates: Vec<&Cluster<()>> = vec![&c0, &c1];
        let idx = brevity_tiebreak(&candidates, &trajectories, 0.01);
        assert_eq!(idx, 0);
    }

    #[test]
    #[should_panic(expected = "empty candidates slice")]
    fn panics_on_empty_candidates() {
        let trajectories: Vec<Trajectory<()>> = vec![];
        let candidates: Vec<&Cluster<()>> = vec![];
        let _ = brevity_tiebreak(&candidates, &trajectories, 0.01);
    }

    /// Issue 940 (TOL) pin, band side: a cluster 0.9e-3 BELOW the max, with a
    /// shorter representative, WINS — the leniency is by design, not a bug.
    #[test]
    fn band_leniency_is_by_design() {
        let trajectories = vec![
            Trajectory {
                outcome: (),
                tokens_or_steps: 100,
                claims: vec![],
                log_probs: None,
            },
            Trajectory {
                outcome: (),
                tokens_or_steps: 10,
                claims: vec![],
                log_probs: None,
            },
        ];
        let c_max = cluster((), 1.0000, 0, &[0]);
        let c_below = cluster((), 1.0000 - 0.9e-3, 1, &[1]);
        let candidates: Vec<&Cluster<()>> = vec![&c_max, &c_below];
        // 0.9e-3 < tiebreak_eps 1e-3 → inside the band → the SHORTER
        // representative (cluster 1) wins despite lower reliability.
        let picked = brevity_tiebreak(&candidates, &trajectories, 1e-3);
        assert_eq!(picked, 1, "in-band shorter cluster must win (leniency is by design)");
    }

    /// Issue 940 (TOL) pin, outside side: 1.1e-3 below the max is OUT of the
    /// band — the max-reliability cluster wins regardless of tokens.
    #[test]
    fn outside_the_band_the_max_wins() {
        let trajectories = vec![
            Trajectory {
                outcome: (),
                tokens_or_steps: 100,
                claims: vec![],
                log_probs: None,
            },
            Trajectory {
                outcome: (),
                tokens_or_steps: 10,
                claims: vec![],
                log_probs: None,
            },
        ];
        let c_max = cluster((), 1.0000, 0, &[0]);
        let c_below = cluster((), 1.0000 - 1.1e-3, 1, &[1]);
        let candidates: Vec<&Cluster<()>> = vec![&c_max, &c_below];
        let picked = brevity_tiebreak(&candidates, &trajectories, 1e-3);
        assert_eq!(picked, 0, "out-of-band cluster must not win");
    }
}
