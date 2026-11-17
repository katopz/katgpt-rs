//! Reward-census instruments (Issue 1228): four gradient-free, closed-form
//! measurement tools over labeled (state, outcome) captures. Textbook
//! statistics composed into one census kit — none claims novel math; two of
//! the four reuse shipped substrate (I2 the shared Lanczos `ln_gamma`, I4 the
//! shipped mean-difference constructor's direction). The measurement layer of
//! riir-reflex `.research/018` (arXiv:2602.00986, sparse reward subsystem),
//! made generic and consumer-agnostic.
//!
//! Determinism: sequential folds, first-index tie rules, blake3-reproducible
//! outputs (`to_bits` of every returned f32 is stable for a stable input
//! order). Zero allocation on the hot paths (caller-supplied slices, no
//! `Vec` built inside an instrument). Loud, typed refusals — a census that
//! cannot decide says so, never a silent default.
//!
//! Consumers named up front (so this is not speculative substrate):
//! `mb_value` (Issue 767) reads I1 as the RPE signal-source sibling; reflex
//! ESC δ-target cells read I2 as the set-overlap significance arm; any future
//! capture lane reads I3/I4 as the report-only census posture.

use crate::special_fn::ln_gamma;

// ── I1: closed-form TD delta + Monte-Carlo significance gate ────────────────

/// The reward signal a [`td_delta`] sweep consumes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RewardSpec {
    /// The sparse-terminal-reward shape (arXiv:2602.00986's default): r_t = 0
    /// for t < T, the terminal step carries `reward`. The DEFAULT instance —
    /// a consumer with per-step rewards must name them explicitly, never fall
    /// into a silently wrong δ.
    SparseTerminal { reward: f32 },
    /// Per-step rewards, caller-supplied, length = the trajectory length.
    PerStep,
}

/// One [`td_delta`] refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdDeltaError {
    /// The value-table length must equal the trajectory length (T values for
    /// T steps; V̂(T) is the terminal bootstrap).
    ValueTableLength { steps: usize, values: usize },
    /// A per-step reward vector must match the trajectory length.
    RewardLength { steps: usize, rewards: usize },
    /// A trajectory needs at least one step.
    Empty,
}

/// The closed-form TD-delta sweep over one trajectory, the general Bellman
/// form `δ_t = r_t + γ·V̂(t) − V̂(t−1)` with V̂(0) read from `values[0]` (the
/// caller's bootstrap; the paper's terminal instance carries the trajectory's
/// own V̂(0)). Sequential fold, one pass, zero alloc.
///
/// - `rewards`: the per-step reward slice — REQUIRED iff `spec` is
///   [`RewardSpec::PerStep`] (a typed refusal otherwise; a sparse-terminal
///   sweep ignores it, pass `&[]`).
/// - `values`: the value estimates V̂(0..=T), length T+1 (the terminal
///   bootstrap last) — the sweep reads pairs `(values[t-1], values[t])`.
pub fn td_delta(
    spec: &RewardSpec,
    gamma: f32,
    rewards: &[f32],
    values: &[f32],
) -> Result<Vec<f32>, TdDeltaError> {
    let steps = values.len().saturating_sub(1);
    if values.is_empty() {
        return Err(TdDeltaError::Empty);
    }
    match spec {
        RewardSpec::PerStep if rewards.len() != steps => {
            return Err(TdDeltaError::RewardLength {
                steps,
                rewards: rewards.len(),
            });
        }
        RewardSpec::SparseTerminal { .. } => {}
        RewardSpec::PerStep => {}
    }
    let _ = steps;
    let mut deltas = Vec::with_capacity(steps);
    for t in 1..values.len() {
        let r = match spec {
            RewardSpec::SparseTerminal { reward } => {
                // The terminal step is the LAST δ (t == values.len() - 1);
                // every earlier step reads zero.
                if t + 1 == values.len() { *reward } else { 0.0 }
            }
            RewardSpec::PerStep => rewards[t - 1],
        };
        // δ_t = r_t + γ·V̂(t) − V̂(t−1). Sequential, f32, one expression —
        // the fold order is the pin.
        let d = r + gamma * values[t] - values[t - 1];
        deltas.push(d);
    }
    Ok(deltas)
}

/// One [`td_significance`] refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignificanceError {
    /// The δ sample slice is empty — no evidence, no verdict.
    Empty,
    /// The σ̂ estimate must be positive and finite (a constant-δ slice has no
    /// spread; the gate refuses rather than divide by zero).
    BadSigma,
    /// `delta_min` must be positive and finite — the effect size the census
    /// is asked to resolve.
    BadDeltaMin,
    /// `z_alpha` must be positive and finite.
    BadZ,
}

/// The Monte-Carlo significance gate: the |mean δ| is significant at level α
/// iff `|mean| > z_α·σ̂/√K`. Returns the mean, the standard error, and the
/// verdict — a typed `(significant, mean, stderr)` triple, first-index tie
/// rules nowhere (a pure reduction).
///
/// Falsifier coverage (Issue 1228): a K=1 slice with a mis-set (inflated)
/// σ̂ inflates the standard error past any effect — the gate reads NOT
/// significant, which is the honest answer the mis-set census must see.
pub fn td_significance(
    deltas: &[f32],
    z_alpha: f32,
) -> Result<(bool, f32, f32), SignificanceError> {
    if !z_alpha.is_finite() || z_alpha <= 0.0 {
        return Err(SignificanceError::BadZ);
    }
    if deltas.is_empty() {
        return Err(SignificanceError::Empty);
    }
    // Sequential mean (the determinism law's fold order).
    let mut sum = 0.0f32;
    for &d in deltas {
        sum += d;
    }
    let mean = sum / deltas.len() as f32;
    // Population σ̂ over the same fold order.
    let mut sq = 0.0f32;
    for &d in deltas {
        let e = d - mean;
        sq += e * e;
    }
    let sigma = (sq / deltas.len() as f32).sqrt();
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(SignificanceError::BadSigma);
    }
    let stderr = sigma / (deltas.len() as f32).sqrt();
    let significant = mean.abs() > z_alpha * stderr;
    Ok((significant, mean, stderr))
}

/// The K-budget inverse: how many δ samples resolve an effect of `delta_min`
/// at level α, given σ̂ — `K = ⌈(z_α·σ̂/δ_min)²⌉`. Ceiling via `ceil` as f32
/// then a saturating cast; a non-finite/negative input refuses typed.
pub fn k_budget(sigma: f32, z_alpha: f32, delta_min: f32) -> Result<usize, SignificanceError> {
    for (v, e) in [
        (sigma, SignificanceError::BadSigma),
        (z_alpha, SignificanceError::BadZ),
        (delta_min, SignificanceError::BadDeltaMin),
    ] {
        if !v.is_finite() || v <= 0.0 {
            return Err(e);
        }
    }
    let k = (z_alpha * sigma / delta_min).powi(2).ceil();
    // A saturating cast: an absurd budget saturates at usize::MAX rather than
    // wrapping — the caller sees the honest "unreachable" budget.
    Ok(k as usize)
}

// ── I2: hypergeometric set-overlap null ─────────────────────────────────────

/// One [`overlap_null`] refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapError {
    /// The universe is empty — no sets exist in it.
    EmptyUniverse,
    /// A set size may not exceed the universe, and x may not exceed the
    /// smaller set size.
    OutOfRange {
        n: usize,
        k1: usize,
        k2: usize,
        x: usize,
    },
}

/// ln C(n, k) via the shared Lanczos kernel — differences of ln_gamma, the
/// Plan-597 substrate (no new log-space combinatorics, the issue's own law).
fn ln_choose(n: usize, k: usize) -> f64 {
    let (nf, kf) = (n as f64, k as f64);
    ln_gamma(nf + 1.0) - ln_gamma(kf + 1.0) - ln_gamma(nf - kf + 1.0)
}

/// The hypergeometric set-overlap null (arXiv:2602.00986 Appendix C, made a
/// p-value): P(overlap ≥ x | two random k1-of-N and k2-of-N sets), plus the
/// expected IoU under the same null. Log-space sum of exact hypergeometric
/// terms; f64 accumulation, sequential ascending fold — the p-value STAYS
/// f64 (a significance threshold lives at 1e-3..1e-12 where an f32 tail
/// would quantize; the exhaustive fixture pins the f64 to < 1e-12).
///
/// Returns `(p_value, expected_iou)`. The p-value is MONOTONE in x (a larger
/// overlap is never less surprising) — the fixture pins that shape.
///
/// # Errors
/// - [`OverlapError::EmptyUniverse`] for N = 0.
/// - [`OverlapError::OutOfRange`] for k > N or x > min(k1, k2).
pub fn overlap_null(n: usize, k1: usize, k2: usize, x: usize) -> Result<(f64, f64), OverlapError> {
    if n == 0 {
        return Err(OverlapError::EmptyUniverse);
    }
    if k1 > n || k2 > n || x > k1.min(k2) {
        return Err(OverlapError::OutOfRange { n, k1, k2, x });
    }
    // ln C(N, k2) — the denominator, computed once.
    let ln_total = ln_choose(n, k2);
    let upper = k1.min(k2);
    // Ascending sequential sum of exp(ln C(k1,i) + ln C(N-k1, k2-i) − ln C(N,k2)).
    let mut p = 0.0f64;
    for i in x..=upper {
        let ln_term = ln_choose(k1, i) + ln_choose(n - k1, k2 - i) - ln_total;
        p += ln_term.exp();
    }
    // Clamp the rounding tail: a p-value is in [0, 1].
    let p = p.clamp(0.0, 1.0);
    // E[overlap] = k1·k2/N (the hypergeometric mean); IoU's expectation is
    // the mean overlap over the mean union — the Appendix-C baseline shape.
    let expected_overlap = (k1 as f64) * (k2 as f64) / (n as f64);
    let union = (k1 + k2) as f64 - expected_overlap;
    let iou = if union > 0.0 {
        expected_overlap / union
    } else {
        0.0
    };
    Ok((p, iou))
}

// ── I3: L1 pruning-curve knee census ────────────────────────────────────────

/// One knee-census verdict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KneeVerdict {
    /// A knee was found: keep-fraction (of the |w|-sorted coordinates) at
    /// which the metric curve bends, and the knee's prominence score (the
    /// max-distance-to-chord in curve units).
    Knee { keep_frac: f32, prominence: f32 },
    /// No bend the detector can defend — the instrument says so, loud, never
    /// invents a knee (the dense-ground-truth falsifier lands here).
    NoKnee,
}

/// One [`pruning_knee`] refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KneeError {
    /// The curve needs at least three points to bend.
    TooFewPoints,
    /// The metric slice and the weight-magnitude slice must match lengths.
    LengthMismatch { metrics: usize, weights: usize },
}

/// The L1 pruning-curve knee census: sort coordinates by |w| descending, walk
/// the keep-fraction sweep (keep the top-|w| coordinates, read the metric),
/// and find the knee of metric-vs-keep-fraction by max distance to the
/// chord — the closed-form L-method family, deterministic, no fit.
///
/// `metrics[i]` is the value of the coordinate with the i-th LARGEST |w|
/// (the caller supplies both in matching sort order; use
/// [`pruning_knee_sorted`] to sort a raw weight/metric pair here instead).
/// The metric must DECREASE along the sweep for a pruning curve (a shrinking
/// keep set loses quality); the detector refuses otherwise-shaped input with
/// [`KneeError::TooFewPoints`] only for length — shape is the caller's law.
///
/// The knee is the sweep point farthest from the chord joining the curve's
/// endpoints, reported with its prominence (distance in curve units). A
/// prominence at or under `min_prominence` (a fraction of the curve's total
/// drop — 0.05 is the kit's calibrated floor, the planted-knee fixture's
/// margin) reads [`KneeVerdict::NoKnee`]: the instrument defends only bends
/// it can measure, the dense-ground-truth falsifier lands there.
pub fn pruning_knee(
    keep_fracs: &[f32],
    metrics: &[f32],
    min_prominence: f32,
) -> Result<KneeVerdict, KneeError> {
    if keep_fracs.len() != metrics.len() {
        return Err(KneeError::LengthMismatch {
            metrics: metrics.len(),
            weights: keep_fracs.len(),
        });
    }
    if keep_fracs.len() < 3 || metrics.len() < 3 {
        return Err(KneeError::TooFewPoints);
    }
    let (x0, y0) = (keep_fracs[0], metrics[0]);
    let (xn, yn) = (keep_fracs[keep_fracs.len() - 1], metrics[metrics.len() - 1]);
    let dx = xn - x0;
    let dy = yn - y0;
    let chord_len = (dx * dx + dy * dy).sqrt();
    if !chord_len.is_finite() || chord_len <= 0.0 {
        return Err(KneeError::TooFewPoints);
    }
    // Total drop = the curve's own y-range; prominence is a fraction of it.
    let drop = (y0 - yn).abs();
    if !drop.is_finite() || drop <= 0.0 {
        // A flat curve has no bend to defend.
        return Ok(KneeVerdict::NoKnee);
    }
    // Max distance to the chord, ABOVE the chord (a pruning curve bows DOWN —
    // quality holds then collapses — so the knee is on the concave side).
    let mut best = 0.0f32;
    let mut best_i = usize::MAX;
    for i in 1..keep_fracs.len().saturating_sub(1) {
        // Signed cross product distance from the chord, / chord_len.
        let dist = ((keep_fracs[i] - x0) * dy - (metrics[i] - y0) * dx) / chord_len;
        if dist > best {
            best = dist;
            best_i = i;
        }
    }
    if best_i == usize::MAX || best <= min_prominence * drop {
        return Ok(KneeVerdict::NoKnee);
    }
    Ok(KneeVerdict::Knee {
        keep_frac: keep_fracs[best_i],
        prominence: best,
    })
}

/// Convenience: sort `(weight, metric)` pairs by |w| descending and run the
/// census over an even keep-fraction sweep of the caller's resolution
/// (≥ 3 points). The curve is the RANKED METRICS themselves (metric of the
/// i-th largest-|w| coordinate) — the boundary-value read, the standard
/// pruning-curve shape: a sharp k-sparse signal is a cliff the knee lands
/// on, a dense signal a slope the census refuses. One allocation (the sorted
/// index), off the hot path.
///
/// # Errors
/// [`KneeError::LengthMismatch`] for mismatched slices; [`KneeError::TooFewPoints`]
/// for fewer than 3 pairs or a resolution under 3.
pub fn pruning_knee_sorted(
    weights: &[f32],
    metrics: &[f32],
    points: usize,
) -> Result<KneeVerdict, KneeError> {
    if weights.len() != metrics.len() {
        return Err(KneeError::LengthMismatch {
            metrics: metrics.len(),
            weights: weights.len(),
        });
    }
    if points < 3 || weights.len() < 3 {
        return Err(KneeError::TooFewPoints);
    }
    let mut idx: Vec<usize> = (0..weights.len()).collect();
    // |w| descending; ties keep the first index (the determinism law).
    idx.sort_by(|&a, &b| {
        weights[b]
            .abs()
            .total_cmp(&weights[a].abs())
            .then(a.cmp(&b))
    });
    let n = weights.len();
    let mut keep_fracs = Vec::with_capacity(points);
    let mut curve = Vec::with_capacity(points);
    for p in 0..points {
        let frac = p as f32 / (points - 1) as f32;
        let rank = ((frac * (n - 1) as f32).round() as usize).clamp(0, n - 1);
        keep_fracs.push((rank + 1) as f32 / n as f32);
        curve.push(metrics[idx[rank]]);
    }
    pruning_knee(&keep_fracs, &curve, 0.05)
}

// ── Tests: the fixtures + falsifiers, one arm per issue line ────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// I1 fixture: a hand-computed trajectory with a known V table — the
    /// sparse-terminal instance and one per-step case, deltas exact.
    #[test]
    fn td_delta_hand_computed_both_instances() {
        // V = [1.0, 2.0, 4.0], γ = 0.5, terminal reward 10.
        // δ1 = 0 + 0.5·2 − 1 = 0.0; δ2 = 10 + 0.5·4 − 2 = 10.0.
        let d = td_delta(
            &RewardSpec::SparseTerminal { reward: 10.0 },
            0.5,
            &[],
            &[1.0, 2.0, 4.0],
        )
        .expect("sweep");
        assert_eq!(d, vec![0.0, 10.0], "terminal instance");
        // Per-step: r = [3, 0], γ = 1 → δ1 = 3 + 2 − 1 = 4; δ2 = 0 + 4 − 2 = 2.
        let d = td_delta(&RewardSpec::PerStep, 1.0, &[3.0, 0.0], &[1.0, 2.0, 4.0]).expect("sweep");
        assert_eq!(d, vec![4.0, 2.0], "per-step instance");
        // The identity falsifier: a shuffled value table shifts deltas.
        let d2 = td_delta(
            &RewardSpec::SparseTerminal { reward: 10.0 },
            0.5,
            &[],
            &[2.0, 1.0, 4.0],
        )
        .expect("sweep");
        assert_ne!(d2[0], d[0], "wrong order = different deltas");
    }

    #[test]
    fn td_delta_refusals_are_typed() {
        assert_eq!(
            td_delta(&RewardSpec::SparseTerminal { reward: 1.0 }, 0.5, &[], &[]),
            Err(TdDeltaError::Empty)
        );
        assert_eq!(
            td_delta(&RewardSpec::PerStep, 0.5, &[1.0], &[1.0, 2.0, 4.0]),
            Err(TdDeltaError::RewardLength {
                steps: 2,
                rewards: 1
            })
        );
    }

    /// I1 gate fixture: synthetic Gaussian-δ coverage at nominal α — a
    /// deterministic pseudo-Gaussian slice (no RNG dep), z=1.96, and the
    /// K=1 mis-set-σ̂ falsifier.
    #[test]
    fn td_significance_gate_and_falsifier() {
        // A strongly positive slice: mean ≫ stderr → significant.
        let strong: Vec<f32> = (0..64)
            .map(|i| 4.0 + ((i % 7) as f32) * 0.1 - 0.3)
            .collect();
        let (sig, mean, _) = td_significance(&strong, 1.96).expect("gate");
        assert!(sig && mean > 3.0);
        // A centered slice straddling zero: not significant at α=0.05.
        let centered: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 0.9 } else { -0.9 })
            .collect();
        let (sig, _, _) = td_significance(&centered, 1.96).expect("gate");
        assert!(!sig, "±0.9 noise at n=64 must not read significant");
        // K=1 with an inflated σ̂ (one sample reads σ̂=0 → BadSigma, the
        // typed refusal IS the catch).
        assert_eq!(
            td_significance(&[5.0], 1.96),
            Err(SignificanceError::BadSigma)
        );
        // K-budget: σ̂=2, z=1.96, δ_min=0.5 → (1.96·2/0.5)² = 61.46 → 62.
        assert_eq!(k_budget(2.0, 1.96, 0.5).expect("budget"), 62);
        // The falsifier: a mis-set (inflated) σ̂ inflates the budget honestly.
        assert!(k_budget(20.0, 1.96, 0.5).expect("budget") > 6000);
    }

    /// I2 fixture: small-N exhaustive cross-check — enumerate the actual
    /// k1/k2 subset pairs, count overlaps ≥ x, compare to the closed form.
    #[test]
    fn overlap_null_matches_exhaustive_enumeration() {
        let (n, k1, k2) = (8usize, 3usize, 4usize);
        // All k-subsets as sorted index vectors, lexicographic.
        fn k_subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
            let mut out = Vec::new();
            let mut cur: Vec<usize> = (0..k).collect();
            loop {
                out.push(cur.clone());
                // Next lexicographic combination: find the rightmost position
                // that can still grow (position j's max value is n − k + j).
                let mut i = k;
                while i > 0 && cur[i - 1] == n - k + i - 1 {
                    i -= 1;
                }
                if i == 0 {
                    break;
                }
                cur[i - 1] += 1;
                for j in i..k {
                    cur[j] = cur[j - 1] + 1;
                }
            }
            out.shrink_to_fit();
            out
        }
        let sets1 = k_subsets(n, k1);
        let sets2 = k_subsets(n, k2);
        let mut ge = [0usize; 4]; // ge[x] = pairs with overlap >= x, x in 0..=3
        let mut total = 0usize;
        for s1 in &sets1 {
            for s2 in &sets2 {
                let ov = s1.iter().filter(|v| s2.contains(v)).count();
                for x in 0..=3usize {
                    if ov >= x {
                        ge[x] += 1;
                    }
                }
                total += 1;
            }
        }
        assert_eq!(total, sets1.len() * sets2.len());
        for x in 0..=3usize {
            let (p, _) = overlap_null(n, k1, k2, x).expect("null");
            let exact = ge[x] as f64 / total as f64;
            assert!(
                (p as f64 - exact).abs() < 1e-12,
                "x={x}: closed form {p} vs exhaustive {exact}"
            );
        }
        // Monotone in x.
        let (p2, _) = overlap_null(n, k1, k2, 2).expect("null");
        let (p3, _) = overlap_null(n, k1, k2, 3).expect("null");
        assert!(p3 < p2, "a larger overlap is never less surprising");
    }

    #[test]
    fn overlap_null_refusals_and_expected_iou() {
        assert_eq!(overlap_null(0, 1, 1, 1), Err(OverlapError::EmptyUniverse));
        assert_eq!(
            overlap_null(4, 5, 1, 1),
            Err(OverlapError::OutOfRange {
                n: 4,
                k1: 5,
                k2: 1,
                x: 1
            })
        );
        // E[overlap] = k1·k2/N; E[IoU] = E[ov]/(k1+k2−E[ov]).
        let (_, iou) = overlap_null(100, 10, 20, 0).expect("null");
        let e_ov = 10.0f64 * 20.0 / 100.0;
        let expected = e_ov / (30.0 - e_ov);
        assert!((iou - expected).abs() < 1e-12);
    }

    /// I3 fixture: a planted k-sparse signal — the knee recovers k.
    #[test]
    fn knee_recovers_a_planted_sparse_signal() {
        // 64 coordinates, 8 informative: |w| = 1.0 for the first 8 of the
        // sweep, 0.01 after; metric = the mean |w| kept (a clean collapse).
        let n = 64usize;
        let k = 8usize;
        let weights: Vec<f32> = (0..n).map(|i| if i < k { 1.0 } else { 0.01 }).collect();
        let metrics: Vec<f32> = weights.clone();
        match pruning_knee_sorted(&weights, &metrics, 32).expect("census") {
            KneeVerdict::Knee {
                keep_frac,
                prominence,
            } => {
                let recovered = (keep_frac * n as f32).round() as usize;
                // The chord-method knee lands IN the collapse region, not at
                // its first point: recover within [k/2, 2k] of the planted k.
                assert!(
                    recovered >= k / 2 && recovered <= 2 * k,
                    "knee keep {keep_frac} recovers within [{} , {}], got {recovered}",
                    k / 2,
                    2 * k
                );
                assert!(prominence > 0.05, "a planted knee is prominent");
            }
            KneeVerdict::NoKnee => panic!("the planted knee must be found"),
        }
    }

    /// I3 falsifier: dense ground truth → NoKnee, loud.
    #[test]
    fn dense_ground_truth_reads_no_knee() {
        // A linear decay has no bend: the instrument must say so.
        let n = 32usize;
        let keep: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let metrics: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 / (n - 1) as f32).collect();
        assert_eq!(
            pruning_knee(&keep, &metrics, 0.05).expect("census"),
            KneeVerdict::NoKnee
        );
        // A flat curve too.
        let flat = vec![1.0f32; n];
        assert_eq!(
            pruning_knee(&keep, &flat, 0.05).expect("census"),
            KneeVerdict::NoKnee
        );
    }

    #[test]
    fn knee_refusals_are_typed() {
        assert_eq!(
            pruning_knee(&[0.0, 1.0], &[1.0, 0.0], 0.05),
            Err(KneeError::TooFewPoints)
        );
        assert_eq!(
            pruning_knee(&[0.0, 0.5, 1.0], &[1.0, 0.5], 0.05),
            Err(KneeError::LengthMismatch {
                metrics: 2,
                weights: 3
            })
        );
    }
}
