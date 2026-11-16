//! Matched-pair direction construction arm — Issue 1224 (Research 676,
//! the AHA distill: "Who Needs an LLM to Program an LLM?"). Third
//! authoring arm of the [`crate::weight_edit_overlay`] lane, sibling of
//! [`crate::weight_edit_overlay::lp_authoring`] (993, solves v for a
//! GIVEN u) and [`crate::weight_edit_overlay::discrete_authoring`]
//! (1209). This arm supplies the **u-construction recipe**: where the
//! correction direction comes from when no agent writes it.
//!
//! # The mechanism (Research 676 §1, the matched-pair law)
//!
//! Both sides of a behavioral contrast must come from the SAME prefix.
//! The caller manufactures the contrast — from a looping trace, cut a
//! prefix before the detected onset, roll several SEEDED branches from
//! that identical prefix state under ONE sampler, label every branch
//! with the loop detector, keep prefixes yielding ≥ 1 looping AND
//! ≥ 1 productive branch, and difference the two branches' states over
//! an ALIGNED window (same onset-relative offsets applied to both).
//! This module consumes the resulting per-pair window means and does the
//! closed-form half:
//!
//! ```text
//! contrast_p = looping_mean_p − productive_mean_p     (per pair)
//! u          = normalize( mean over TRAIN pairs of contrast_p )
//! ```
//!
//! with the paper's 80/20 discipline pinned as an index-modulus split
//! (`holdout_every`, default 5): every Nth pair (1-based) is held out,
//! the rest train. The held-out pairs are never averaged into `u` —
//! they are scored against it ([`HeldoutSeparation`]), the honest
//! construction check.
//!
//! # What this module does NOT do
//!
//! No branch generation (that is the caller's rollout protocol — the
//! fixture half lives in the gate battery), no labeling (the caller's
//! detector), no LP solve (`lp_authoring` consumes `u` unchanged via
//! `solve_min_l1_authoring_lp` + `compose_direction_overlay`), no
//! serving path (the Bench-976 veto stands), no real-model claim (the
//! 990 posture: fixture-scale only).
//!
//! # Sampler purity (the F2 doctrine, carried at construction)
//!
//! The direction is only as clean as the branch manufacture: if looping
//! branches come from one sampler family and productive branches from
//! another, `u` encodes the sampler, not the behavior (the paper's
//! presence-penalty trap: the mixed-penalty build scored worse AND
//! carried the confound). The construction here is sampler-agnostic —
//! it cannot detect a split — so the gate battery carries the
//! falsifier arm: a deliberately sampler-split build must degrade or
//! the honest negative is recorded. Same-prefix + same-sampler is a
//! CALLER obligation this module's docs state and its battery tests.
//!
//! # Separability ≠ edit quality (the whitened negative)
//!
//! The paper's best-separating direction (whitened, unmatched) made the
//! worst edit. [`HeldoutSeparation`] is reported as a construction
//! diagnostic, never promoted to a quality verdict — the battery's
//! behavioral arms (dose clearance, isolation null, random control)
//! are the parity bar. A direction that separates but fails behavior
//! is the negative result, recorded, not tuned away.
//!
//! # The fixture verdict (bench 985, 2026-11-16 — the recorded scope)
//!
//! The wsp980 fixture battery MEASURED the full parity bar: the mined
//! direction recovers the injected loop axis (cos ≈ 0.97) and passes
//! the complete 993 battery (dose grid clears at the authored dose,
//! isolation null, random control, twin refusal, retention) — the
//! pre-registered PARITY bar is MET, and the construction correctness +
//! determinism it evidences is real. The confound falsifiers (F1
//! mismatched-prefix, F2 sampler-split) CANNOT fire on this fixture:
//! the behavioral stuck-vs-finished axis dominates every injected
//! confound (both confounded builds land within cos 0.95 of the clean
//! direction). Per the issue's own T2b law, the parity claim therefore
//! stays **construction-only**: this module makes no confound-
//! robustness claim and no recipe-value claim beyond the fixture — the
//! recipe's real-model premise (matched pairs mattering over heuristics
//! under confounds) remains UNTESTED here, veto-gated exactly like the
//! rest of the lane (Bench 976). The falsifier non-firing pins are the
//! tripwire that re-opens the record if the fixture or sampler changes.
//!
//! # Determinism (Issue 940 law)
//!
//! Every fold is a sequential `acc +=` in a fixed index order — the
//! per-channel mean folds, the contrast mean, the norm, the held-out
//! projections. The split is index-modulus (no shuffle, no RNG). Same
//! pinned contrasts → bit-identical [`MatchedPairDirection`].

// ─────────────────────────────────────────────────────────────────────
// Public surface
// ─────────────────────────────────────────────────────────────────────

/// One pair's aligned-window state means. Both windows were extracted by
/// the caller at the SAME onset-relative indices from the looping and
/// productive branches of ONE prefix (the alignment law — misaligned or
/// cross-prefix windows are the F1 falsifier's territory, never valid
/// input here).
#[derive(Clone, Debug, PartialEq)]
pub struct PairContrast {
    /// Mean over the looping branch's window states (dim-wide).
    pub looping_mean: Vec<f32>,
    /// Mean over the productive branch's window states (dim-wide).
    pub productive_mean: Vec<f32>,
}

/// The construction's pinned knobs. `holdout_every = 5` is the paper's
/// 80/20 (every 5th pair held out).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchedPairConfig {
    /// Every Nth pair (1-based index divisible by N) is HELD OUT; the
    /// remaining pairs train the direction. Must be ≥ 2 (1 would hold
    /// out everything — [`MatchedPairError::InsufficientPairs`] is the
    /// loud result, but the config itself is malformed).
    pub holdout_every: usize,
}

impl Default for MatchedPairConfig {
    fn default() -> Self {
        Self { holdout_every: 5 }
    }
}

/// Why the construction refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchedPairError {
    /// Malformed input (empty pairs, dim mismatch, non-finite entry,
    /// invalid config).
    Malformed(&'static str),
    /// The split left zero train pairs — nothing to average.
    InsufficientPairs,
    /// The mean train contrast is exactly zero (or non-finite) — no
    /// direction exists. A LOUD refusal, never a silent zero vector.
    DegenerateZeroDirection,
}

/// The held-out construction check. Reported, never promoted to a
/// quality verdict (separability ≠ edit quality — the whitened negative).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldoutSeparation {
    /// Held-out pairs scored.
    pub pairs: usize,
    /// Mean projection of the held-out LOOPING window means onto `u`.
    pub mean_looping_projection: f32,
    /// Mean projection of the held-out PRODUCTIVE window means onto `u`.
    pub mean_productive_projection: f32,
    /// `looping − productive` projection gap (positive = `u` points
    /// loop-ward on the pairs it never saw).
    pub gap: f32,
    /// Fraction of held-out contrasts whose projection sign matches the
    /// gap's sign. `0.0` recorded when the gap is exactly zero (no
    /// held-out signal — the caller judges, this module never folds a
    /// zero gap into a pass).
    pub sign_consistency: f32,
}

/// The constructed direction + its diagnostics. `direction` is unit-L2.
#[derive(Clone, Debug, PartialEq)]
pub struct MatchedPairDirection {
    /// The unit-norm direction `u` (the LP's `compose_direction_overlay`
    /// consumes it verbatim).
    pub direction: Vec<f32>,
    /// Train pairs averaged into `u`.
    pub train_pairs: usize,
    /// Held-out pairs scored (never averaged in).
    pub holdout_pairs: usize,
    /// The held-out check (`None` when no holdout pair exists).
    pub holdout_separation: Option<HeldoutSeparation>,
    /// The train mean contrast's L2 norm BEFORE normalization — the
    /// raw signal-strength diagnostic (a near-zero norm means the
    /// branches barely differed; the direction is real but weak).
    pub mean_contrast_norm: f32,
}

/// Build the matched-pair direction from per-pair aligned window means.
/// Deterministic: same pinned contrasts → bit-identical result.
pub fn build_matched_pair_direction(
    contrasts: &[PairContrast],
    cfg: &MatchedPairConfig,
) -> Result<MatchedPairDirection, MatchedPairError> {
    if contrasts.is_empty() {
        return Err(MatchedPairError::Malformed("no pairs supplied"));
    }
    if cfg.holdout_every < 2 {
        return Err(MatchedPairError::Malformed(
            "holdout_every must be >= 2 (every Nth pair held out)",
        ));
    }
    let dim = contrasts[0].looping_mean.len();
    if dim == 0 {
        return Err(MatchedPairError::Malformed("zero contrast dim"));
    }
    for c in contrasts.iter() {
        if c.looping_mean.len() != dim || c.productive_mean.len() != dim {
            return Err(MatchedPairError::Malformed("pair window-mean dim mismatch"));
        }
        if c.looping_mean.iter().any(|x| !x.is_finite())
            || c.productive_mean.iter().any(|x| !x.is_finite())
        {
            return Err(MatchedPairError::Malformed("non-finite window-mean entry"));
        }
    }

    // Index-modulus split (pinned, no RNG): 1-based index divisible by
    // holdout_every → held out. Pair order is the caller's pinned order.
    let mut train_idx = Vec::new();
    let mut holdout_idx = Vec::new();
    for (i, _) in contrasts.iter().enumerate() {
        if (i + 1) % cfg.holdout_every == 0 {
            holdout_idx.push(i);
        } else {
            train_idx.push(i);
        }
    }
    if train_idx.is_empty() {
        return Err(MatchedPairError::InsufficientPairs);
    }

    // Mean train contrast — per-channel sequential folds over the pinned
    // pair order (Issue-940: the channel loop is OUTER so each channel's
    // accumulator folds pair-by-pair in pinned order).
    let mut mean = vec![0.0f32; dim];
    let inv = 1.0 / train_idx.len() as f32;
    for &p in &train_idx {
        let c = &contrasts[p];
        for (m, (l, pr)) in mean
            .iter_mut()
            .zip(c.looping_mean.iter().zip(c.productive_mean.iter()))
        {
            *m += (l - pr) * inv;
        }
    }

    // Norm + normalize. An exactly-zero (or non-finite) mean is the loud
    // refusal — no direction exists.
    let mut norm_sq = 0.0f32;
    for &m in mean.iter() {
        norm_sq += m * m;
    }
    let norm = norm_sq.sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(MatchedPairError::DegenerateZeroDirection);
    }
    let mut direction = mean;
    for slot in direction.iter_mut() {
        *slot /= norm;
    }

    // Held-out check: project each held-out pair's window means onto u.
    let holdout_separation = if holdout_idx.is_empty() {
        None
    } else {
        let mut loop_sum = 0.0f32;
        let mut prod_sum = 0.0f32;
        let mut signs = Vec::with_capacity(holdout_idx.len());
        for &p in &holdout_idx {
            let c = &contrasts[p];
            let mut lp = 0.0f32;
            for (u, x) in direction.iter().zip(c.looping_mean.iter()) {
                lp += u * x;
            }
            let mut pp = 0.0f32;
            for (u, x) in direction.iter().zip(c.productive_mean.iter()) {
                pp += u * x;
            }
            loop_sum += lp;
            prod_sum += pp;
            let gap_i = lp - pp;
            signs.push(if gap_i > 0.0 {
                1.0
            } else if gap_i < 0.0 {
                -1.0
            } else {
                0.0
            });
        }
        let mean_loop = loop_sum / holdout_idx.len() as f32;
        let mean_prod = prod_sum / holdout_idx.len() as f32;
        let gap = mean_loop - mean_prod;
        let anchor = if gap > 0.0 {
            1.0
        } else if gap < 0.0 {
            -1.0
        } else {
            0.0
        };
        let consistency = if anchor == 0.0 {
            // No held-out signal at all — recorded as zero consistency;
            // the caller judges (this module never folds a zero gap into
            // a pass).
            0.0
        } else {
            let m = signs.iter().filter(|&&s| s == anchor).count() as f32;
            m / holdout_idx.len() as f32
        };
        Some(HeldoutSeparation {
            pairs: holdout_idx.len(),
            mean_looping_projection: mean_loop,
            mean_productive_projection: mean_prod,
            gap,
            sign_consistency: consistency,
        })
    };

    Ok(MatchedPairDirection {
        direction,
        train_pairs: train_idx.len(),
        holdout_pairs: holdout_idx.len(),
        holdout_separation,
        mean_contrast_norm: norm,
    })
}

/// Mean over an aligned window of recorded states (`states[r]` = one
/// state vector). The window is `start ..< start + len`, clamped to the
/// available states; fewer than one usable state returns `None`. The
/// caller enforces its own minimum window length (the fixture gate pins
/// it). Per-channel sequential fold in pinned index order.
pub fn window_mean(states: &[Vec<f32>], start: usize, len: usize) -> Option<Vec<f32>> {
    if states.is_empty() || start >= states.len() {
        return None;
    }
    let end = (start + len).min(states.len());
    let n = end - start;
    if n == 0 {
        return None;
    }
    let dim = states[start].len();
    let mut acc = vec![0.0f32; dim];
    for s in &states[start..end] {
        if s.len() != dim {
            return None;
        }
        for (a, x) in acc.iter_mut().zip(s.iter()) {
            *a += x;
        }
    }
    let inv = 1.0 / n as f32;
    for a in acc.iter_mut() {
        *a *= inv;
    }
    Some(acc)
}

// ─────────────────────────────────────────────────────────────────────
// Unit tests — the synthetic planted fixture
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 32;
    const SPOTS: [usize; 4] = [2, 9, 17, 31];

    fn planted() -> Vec<f32> {
        let mut u = vec![0.0f32; DIM];
        let amp = 1.0 / (SPOTS.len() as f32).sqrt();
        for (k, &j) in SPOTS.iter().enumerate() {
            u[j] = if k % 2 == 0 { amp } else { -amp };
        }
        u
    }

    /// A pair whose contrast is `a · planted` (the productive side shared,
    /// the looping side displaced along the planted direction).
    fn pair(a: f32, shared: f32) -> PairContrast {
        let u = planted();
        let mut looping = vec![shared; DIM];
        for (l, &u_j) in looping.iter_mut().zip(u.iter()) {
            *l += a * u_j;
        }
        PairContrast {
            looping_mean: looping,
            productive_mean: vec![shared; DIM],
        }
    }

    #[test]
    fn planted_contrasts_recover_the_direction() {
        let pairs: Vec<PairContrast> = (0..10).map(|i| pair(0.5 + i as f32 * 0.1, 0.3)).collect();
        let built =
            build_matched_pair_direction(&pairs, &MatchedPairConfig::default()).expect("build");
        let u = planted();
        let mut dot = 0.0f32;
        for (d, x) in built.direction.iter().zip(u.iter()) {
            dot += d * x;
        }
        assert!(
            (dot - 1.0).abs() < 1e-4,
            "direction must recover the planted contrast axis (cos {dot})"
        );
        assert_eq!(built.train_pairs, 8);
        assert_eq!(built.holdout_pairs, 2);
        let sep = built.holdout_separation.expect("holdout pairs exist");
        assert!(sep.gap > 0.0, "held-out gap must point loop-ward");
        assert_eq!(sep.sign_consistency, 1.0);
        assert!(built.mean_contrast_norm > 0.0);
    }

    #[test]
    fn holdout_pairs_never_leak_into_the_direction() {
        // Train pairs along e₀, holdout pairs along e₁: any e₁ mass in the
        // direction would prove leakage.
        let mut pairs = Vec::with_capacity(10);
        for i in 0..10 {
            let mut looping = vec![0.0f32; 2];
            let productive = vec![0.0f32; 2];
            if (i + 1) % 5 == 0 {
                looping[1] = 1.0; // held out: e₁ contrast
            } else {
                looping[0] = 1.0 + i as f32 * 0.1; // train: e₀ contrast
            }
            pairs.push(PairContrast {
                looping_mean: looping,
                productive_mean: productive,
            });
        }
        let built =
            build_matched_pair_direction(&pairs, &MatchedPairConfig::default()).expect("build");
        assert_eq!(built.direction[1], 0.0, "holdout contrast must not leak");
        assert!(built.direction[0] > 0.99);
        assert_eq!(built.train_pairs, 8);
        assert_eq!(built.holdout_pairs, 2);
    }

    #[test]
    fn build_is_bit_identical_across_runs() {
        let pairs: Vec<PairContrast> = (0..7).map(|i| pair(0.4 + i as f32 * 0.2, -0.2)).collect();
        let run = || build_matched_pair_direction(&pairs, &MatchedPairConfig::default()).unwrap();
        let a = run();
        let b = run();
        assert!(
            a.direction
                .iter()
                .zip(b.direction.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
        assert_eq!(a, b);
    }

    #[test]
    fn refusals_are_loud() {
        // Empty input.
        assert_eq!(
            build_matched_pair_direction(&[], &MatchedPairConfig::default()),
            Err(MatchedPairError::Malformed("no pairs supplied"))
        );
        // holdout_every 0 and 1 are malformed.
        for n in [0usize, 1] {
            let cfg = MatchedPairConfig { holdout_every: n };
            let pairs = vec![pair(1.0, 0.0)];
            assert!(matches!(
                build_matched_pair_direction(&pairs, &cfg),
                Err(MatchedPairError::Malformed(_))
            ));
        }
        // Dim mismatch.
        let bad = vec![
            pair(1.0, 0.0),
            PairContrast {
                looping_mean: vec![1.0],
                productive_mean: vec![0.0],
            },
        ];
        assert!(matches!(
            build_matched_pair_direction(&bad, &MatchedPairConfig::default()),
            Err(MatchedPairError::Malformed(_))
        ));
        // Non-finite entry.
        let mut nan = pair(1.0, 0.0);
        nan.looping_mean[0] = f32::NAN;
        assert!(matches!(
            build_matched_pair_direction(&[nan], &MatchedPairConfig::default()),
            Err(MatchedPairError::Malformed(_))
        ));
        // Zero mean contrast (looping == productive everywhere).
        let zero: Vec<PairContrast> = (0..6)
            .map(|_| PairContrast {
                looping_mean: vec![0.5; DIM],
                productive_mean: vec![0.5; DIM],
            })
            .collect();
        assert_eq!(
            build_matched_pair_direction(&zero, &MatchedPairConfig::default()),
            Err(MatchedPairError::DegenerateZeroDirection)
        );
        // With holdout_every ≥ 2 the split can NEVER exhaust the train
        // set (floor(n/N) ≤ n/2 < n), so InsufficientPairs is the
        // defensive unreachable — the config-level refusal is Malformed.
        // Verify the tightest legal split: 2 pairs at every=2 → 1 train,
        // 1 holdout.
        let cfg = MatchedPairConfig { holdout_every: 2 };
        let two = vec![pair(1.0, 0.0), pair(2.0, 0.0)];
        let tight = build_matched_pair_direction(&two, &cfg).expect("tightest legal split");
        assert_eq!(tight.train_pairs, 1);
        assert_eq!(tight.holdout_pairs, 1);
    }

    #[test]
    fn single_pair_trains_without_holdout() {
        let pairs = vec![pair(1.5, 0.1)];
        let built =
            build_matched_pair_direction(&pairs, &MatchedPairConfig::default()).expect("build");
        assert_eq!(built.train_pairs, 1);
        assert_eq!(built.holdout_pairs, 0);
        assert!(built.holdout_separation.is_none());
    }

    #[test]
    fn window_mean_clamps_and_refuses_empty() {
        let states = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        assert_eq!(window_mean(&states, 0, 2), Some(vec![2.0, 3.0]));
        // Length clamps to available states.
        assert_eq!(window_mean(&states, 1, 9), Some(vec![4.0, 5.0]));
        // Start past the end → None.
        assert_eq!(window_mean(&states, 3, 2), None);
        // Empty states → None.
        assert_eq!(window_mean(&[], 0, 2), None);
        // Dim mismatch inside the window → None.
        let ragged = vec![vec![1.0, 2.0], vec![1.0]];
        assert_eq!(window_mean(&ragged, 0, 2), None);
    }
}
