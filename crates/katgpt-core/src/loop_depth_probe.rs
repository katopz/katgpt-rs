//! Issue 898 — KL effective depth: a measured exit calibration for looped
//! inference (arXiv:2609.19107, distillation `.research/592`).
//!
//! The looped runtime's depth knobs (`Config::loop_min` / `loop_max`, the
//! `gain_cost_halt` ε, the ELT 2× over-iteration cap) are hand-tuned. This
//! module is the offline instrument that measures what those knobs should
//! be, from per-loop readouts the stack already produces (Issue 717's
//! `LoopDeepRun` snapshots with `capture_logits`):
//!
//! 1. [`kl_profile`] — logit-lens KL of every loop's readout against the
//!    final readout, and [`effective_depth`] — the first loop at or after the
//!    KL peak whose readout is within a threshold of the final one.
//! 2. [`spread`] — loop-flatness of a loss-vs-loop-count curve (checkpoint
//!    selection for loop-count-elastic serving).
//! 3. [`write_fractions`] — ‖Δh_τ‖/‖h_τ‖ across iterations, and
//!    [`stall_onset`] — the first loop after which every step writes less
//!    than ε (the principled ε for `gain_cost_halt`).
//! 4. [`DepthHistogram`] — per-token executed depth under any-time exit.
//!
//! Calibration: [`agreement_exit`] is the modelless oracle (the smallest
//! loop count whose readout argmax already equals the final one and never
//! changes again), [`fit_threshold`] picks the KL threshold that best
//! predicts it on one fixture half, and [`holdout_hit_rate`] scores that
//! threshold on the other half — the paper's extrapolation protocol adopted
//! as the gate arm (a threshold without holdout validation is unvalidated
//! extrapolation).
//!
//! ## Why a softmax appears here
//!
//! The repo rule is sigmoid, not softmax, for any gate or router this stack
//! designs. The KL here adds no decision softmax: it MEASURES the model's own
//! readout distribution, which is whatever normalisation the model decodes
//! with — a logit lens must match the readout or it measures something else.
//! The normalisation is computed as a log-sum-exp in f64 and never leaves
//! this module.
//!
//! ## Contract
//!
//! Pure functions over caller-owned slices. Nothing here allocates except
//! where an output `Vec` is passed in, and those are `clear()`ed and refilled
//! (capacity reused across calls — the G4 zero-alloc contract once warm).
//! Non-finite inputs propagate as NaN rather than being silently clamped: a
//! diverged loop must read as diverged.

/// `ln Σ exp(x_i)` in f64 with max-shift. NaN if any input is non-finite.
#[inline]
fn log_sum_exp(x: &[f32]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &v in x {
        let v = v as f64;
        if !v.is_finite() {
            return f64::NAN;
        }
        if v > max {
            max = v;
        }
    }
    if x.is_empty() {
        return f64::NAN;
    }
    let mut z = 0.0f64;
    for &v in x {
        z += (v as f64 - max).exp();
    }
    max + z.ln()
}

/// `KL(P ‖ Q)` in nats between the readout distributions of two logit
/// vectors (`P = decode(p_logits)`, `Q = decode(q_logits)`).
///
/// Zero-allocation, two passes per vector. Returns NaN for mismatched or
/// empty lengths or any non-finite logit. The result is clamped at 0 — the
/// true value is non-negative and a tiny negative is f64 cancellation.
pub fn kl_from_logits(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    if p_logits.len() != q_logits.len() || p_logits.is_empty() {
        return f32::NAN;
    }
    let lse_p = log_sum_exp(p_logits);
    let lse_q = log_sum_exp(q_logits);
    if !lse_p.is_finite() || !lse_q.is_finite() {
        return f32::NAN;
    }
    // KL = Σ p_i (log p_i − log q_i) = Σ p_i (a_i − b_i) − lse_p + lse_q
    let mut acc = 0.0f64;
    for (&a, &b) in p_logits.iter().zip(q_logits) {
        let a = a as f64;
        acc += (a - lse_p).exp() * (a - b as f64);
    }
    (acc - lse_p + lse_q).max(0.0) as f32
}

/// Per-loop KL of each snapshot's readout against `reference` (normally the
/// final readout): `out[τ] = KL(reference ‖ snapshot_τ)`.
///
/// `out` is cleared and refilled (capacity reused).
pub fn kl_profile<'a, I>(reference: &[f32], snapshots: I, out: &mut Vec<f32>)
where
    I: IntoIterator<Item = &'a [f32]>,
{
    out.clear();
    for snap in snapshots {
        out.push(kl_from_logits(reference, snap));
    }
}

/// Index of the first maximum of `xs`, ignoring NaN. `None` when empty or
/// every value is NaN.
#[inline]
fn first_argmax(xs: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in xs.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if v <= b => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i)
}

/// KL effective depth, as a **1-based loop count**: the first loop at or
/// after the KL peak whose readout is within `threshold` nats of the final
/// readout.
///
/// `None` when the profile is empty or no post-peak loop reaches the
/// threshold (possible only when the reference is not the last snapshot —
/// against the final readout the last entry is 0). A NaN anywhere in the
/// profile returns `None`: a diverged loop has no effective depth.
pub fn effective_depth(kl: &[f32], threshold: f32) -> Option<usize> {
    if kl.iter().any(|v| v.is_nan()) {
        return None;
    }
    let peak = first_argmax(kl)?;
    kl[peak..]
        .iter()
        .position(|&v| v <= threshold)
        .map(|off| peak + off + 1)
}

/// Write-fraction spectrum `out[τ−1] = ‖h_τ − h_{τ−1}‖ / ‖h_τ‖` for
/// τ = 1..n over consecutive loop states (one entry fewer than states).
///
/// `‖h_τ‖ == 0` with a non-zero step reads `+∞`; both zero reads `0`. Any
/// non-finite state component reads NaN. Mismatched state lengths read NaN.
/// `out` is cleared and refilled.
pub fn write_fractions<'a, I>(states: I, out: &mut Vec<f32>)
where
    I: IntoIterator<Item = &'a [f32]>,
{
    out.clear();
    let mut prev: Option<&[f32]> = None;
    for cur in states {
        if let Some(p) = prev {
            out.push(step_fraction(p, cur));
        }
        prev = Some(cur);
    }
}

#[inline]
fn step_fraction(prev: &[f32], cur: &[f32]) -> f32 {
    if prev.len() != cur.len() {
        return f32::NAN;
    }
    let (mut d2, mut c2) = (0.0f64, 0.0f64);
    for (&p, &c) in prev.iter().zip(cur) {
        let (p, c) = (p as f64, c as f64);
        let d = c - p;
        d2 += d * d;
        c2 += c * c;
    }
    if !d2.is_finite() || !c2.is_finite() {
        return f32::NAN;
    }
    if d2 == 0.0 {
        return 0.0;
    }
    if c2 == 0.0 {
        return f32::INFINITY;
    }
    (d2 / c2).sqrt() as f32
}

/// First step index `τ` (0-based into a write-fraction spectrum) from which
/// every remaining step writes at most `epsilon` — the loop has stalled.
/// Returned as the **1-based loop count** that suffices (`τ + 1`: the state
/// after that many loops is already the fixed point to within ε).
///
/// `None` when the final step still exceeds ε, or the spectrum is empty or
/// carries a NaN.
pub fn stall_onset(write_fractions: &[f32], epsilon: f32) -> Option<usize> {
    if write_fractions.is_empty() || write_fractions.iter().any(|v| v.is_nan()) {
        return None;
    }
    // Walk back from the end while steps stay under ε.
    let tail = write_fractions
        .iter()
        .rev()
        .take_while(|&&w| w <= epsilon)
        .count();
    match tail {
        0 => None,
        t => Some(write_fractions.len() - t + 1),
    }
}

/// `true` when `xs` never rises by more than `rel_tol × |previous|` — the
/// monotone-decay signature (tested per checkpoint, never assumed). Empty
/// or single-element sequences are trivially monotone; any NaN is not.
pub fn is_monotone_nonincreasing(xs: &[f32], rel_tol: f32) -> bool {
    if xs.iter().any(|v| v.is_nan()) {
        return false;
    }
    xs.windows(2).all(|w| w[1] <= w[0] + rel_tol * w[0].abs())
}

/// `max − min` over the finite values of `xs` — the loop-flatness score of a
/// loss-vs-loop-count curve. `None` when no value is finite.
pub fn spread(xs: &[f32]) -> Option<f32> {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in xs.iter().filter(|v| v.is_finite()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo <= hi).then_some(hi - lo)
}

/// The modelless exit oracle, as a **1-based loop count**: the smallest `k`
/// such that the readout argmax at every loop `≥ k` equals the final one.
///
/// `argmaxes[τ]` is the argmax token of loop τ's readout. The final loop
/// always agrees with itself, so the result is defined for any non-empty
/// input (`None` only when empty).
pub fn agreement_exit(argmaxes: &[usize]) -> Option<usize> {
    let last = *argmaxes.last()?;
    let agreeing_tail = argmaxes.iter().rev().take_while(|&&a| a == last).count();
    Some(argmaxes.len() - agreeing_tail + 1)
}

/// One calibration sample: a KL profile and its oracle exit (1-based).
#[derive(Debug, Clone, Copy)]
pub struct DepthSample<'a> {
    /// `KL(final ‖ loop_τ)` per loop.
    pub kl: &'a [f32],
    /// Oracle exit for this sample (e.g. [`agreement_exit`]).
    pub oracle_exit: usize,
}

/// Result of [`fit_threshold`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdFit {
    /// The chosen KL threshold (nats).
    pub threshold: f32,
    /// Mean |predicted − oracle| exit error on the fitting half, in loops.
    pub mean_abs_error: f32,
    /// Samples the threshold could not predict (NaN profile / unreached).
    pub unpredicted: usize,
}

/// Fit the KL threshold on one fixture half: pick the candidate minimising
/// the mean absolute exit error against each sample's oracle, over the
/// samples it can predict. Ties break toward the **larger** threshold (the
/// earlier exit — cheaper at equal error). An unpredictable sample costs the
/// full profile length, so a threshold cannot win by predicting nothing.
///
/// `None` when either input is empty.
pub fn fit_threshold(samples: &[DepthSample<'_>], candidates: &[f32]) -> Option<ThresholdFit> {
    if samples.is_empty() || candidates.is_empty() {
        return None;
    }
    let mut best: Option<ThresholdFit> = None;
    for &thr in candidates {
        let mut err = 0.0f64;
        let mut unpredicted = 0usize;
        for s in samples {
            match effective_depth(s.kl, thr) {
                Some(k) => err += k.abs_diff(s.oracle_exit) as f64,
                None => {
                    unpredicted += 1;
                    err += s.kl.len().max(1) as f64;
                }
            }
        }
        let fit = ThresholdFit {
            threshold: thr,
            mean_abs_error: (err / samples.len() as f64) as f32,
            unpredicted,
        };
        best = match best {
            None => Some(fit),
            Some(b) if fit.mean_abs_error < b.mean_abs_error => Some(fit),
            Some(b) if fit.mean_abs_error == b.mean_abs_error && fit.threshold > b.threshold => {
                Some(fit)
            }
            keep => keep,
        };
    }
    best
}

/// Fraction of `samples` whose predicted exit at `threshold` lands within
/// `tolerance` loops of the oracle — the holdout arm. An unpredicted sample
/// is a miss. `None` when `samples` is empty.
pub fn holdout_hit_rate(
    samples: &[DepthSample<'_>],
    threshold: f32,
    tolerance: usize,
) -> Option<f32> {
    if samples.is_empty() {
        return None;
    }
    let hits = samples
        .iter()
        .filter(|s| {
            effective_depth(s.kl, threshold).is_some_and(|k| k.abs_diff(s.oracle_exit) <= tolerance)
        })
        .count();
    Some(hits as f32 / samples.len() as f32)
}

/// Fixed-bucket histogram of executed loop depth (1-based), `N` buckets.
/// Depths `≥ N` land in `overflow`; depth 0 is rejected (a loop always
/// executes at least once). Allocation-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthHistogram<const N: usize> {
    /// `counts[k − 1]` = number of tokens that executed exactly `k` loops.
    pub counts: [u32; N],
    /// Tokens whose depth was `> N`.
    pub overflow: u32,
}

impl<const N: usize> Default for DepthHistogram<N> {
    fn default() -> Self {
        Self {
            counts: [0; N],
            overflow: 0,
        }
    }
}

impl<const N: usize> DepthHistogram<N> {
    /// Record one token's executed depth. Returns `false` (and records
    /// nothing) for depth 0.
    pub fn record(&mut self, depth: usize) -> bool {
        match depth {
            0 => false,
            d if d <= N => {
                self.counts[d - 1] = self.counts[d - 1].saturating_add(1);
                true
            }
            _ => {
                self.overflow = self.overflow.saturating_add(1);
                true
            }
        }
    }

    /// Total recorded tokens.
    pub fn total(&self) -> u64 {
        self.counts.iter().map(|&c| c as u64).sum::<u64>() + self.overflow as u64
    }

    /// Mean executed depth over in-range buckets; `None` when nothing in
    /// range was recorded. Overflow is excluded — its depth is unknown, and
    /// the count is reported separately rather than guessed.
    pub fn mean_in_range(&self) -> Option<f64> {
        let n: u64 = self.counts.iter().map(|&c| c as u64).sum();
        if n == 0 {
            return None;
        }
        let s: u64 = self
            .counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (i as u64 + 1) * c as u64)
            .sum();
        Some(s as f64 / n as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_identical_is_zero_and_shift_invariant() {
        let a = [1.0f32, 2.0, -0.5, 0.25];
        assert_eq!(kl_from_logits(&a, &a), 0.0);
        // A constant logit shift is the same distribution.
        let b: Vec<f32> = a.iter().map(|v| v + 7.0).collect();
        assert!(kl_from_logits(&a, &b) < 1e-9);
    }

    #[test]
    fn kl_matches_closed_form_two_point() {
        // P = σ-split of logits (0, ln 3) → (1/4, 3/4); Q uniform (1/2, 1/2).
        let p = [0.0f32, 3f32.ln()];
        let q = [0.0f32, 0.0];
        let want = 0.25 * (0.25f64 / 0.5).ln() + 0.75 * (0.75f64 / 0.5).ln();
        let got = kl_from_logits(&p, &q) as f64;
        assert!((got - want).abs() < 1e-6, "got {got} want {want}");
        // Asymmetric: KL(Q‖P) differs.
        let rev = kl_from_logits(&q, &p) as f64;
        assert!((rev - want).abs() > 1e-3);
    }

    #[test]
    fn kl_rejects_bad_input() {
        assert!(kl_from_logits(&[1.0], &[1.0, 2.0]).is_nan());
        assert!(kl_from_logits(&[], &[]).is_nan());
        assert!(kl_from_logits(&[f32::NAN, 1.0], &[0.0, 1.0]).is_nan());
        assert!(kl_from_logits(&[f32::INFINITY, 1.0], &[0.0, 1.0]).is_nan());
    }

    #[test]
    fn kl_profile_reuses_capacity() {
        let r = [0.0f32, 1.0];
        let s1 = [1.0f32, 0.0];
        let mut out = Vec::with_capacity(4);
        let cap = out.capacity();
        kl_profile(&r, [&s1[..], &r[..]], &mut out);
        assert_eq!(out.len(), 2);
        assert!(out[0] > 0.0 && out[1] == 0.0);
        kl_profile(&r, [&r[..]], &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out.capacity(), cap);
    }

    #[test]
    fn effective_depth_first_post_peak_under_threshold() {
        // Rises then decays: peak at τ=1.
        let kl = [0.2f32, 0.9, 0.4, 0.05, 0.0];
        assert_eq!(effective_depth(&kl, 0.1), Some(4));
        // τ=0 is under threshold but BEFORE the peak — must not count.
        assert_eq!(effective_depth(&kl, 0.3), Some(4));
        assert_eq!(effective_depth(&kl, 0.5), Some(3));
        // Loose threshold: the peak loop itself qualifies.
        assert_eq!(effective_depth(&kl, 1.0), Some(2));
        assert_eq!(effective_depth(&[], 0.1), None);
        assert_eq!(effective_depth(&[0.5, f32::NAN, 0.0], 0.1), None);
        // Reference not the last snapshot: unreachable threshold.
        assert_eq!(effective_depth(&[0.5, 0.4], 0.1), None);
    }

    #[test]
    fn write_fractions_and_identity_canary() {
        let h0 = [1.0f32, 0.0];
        let h1 = [1.0f32, 1.0];
        let h2 = [1.0f32, 1.0];
        let mut wf = Vec::new();
        write_fractions([&h0[..], &h1[..], &h2[..]], &mut wf);
        assert_eq!(wf.len(), 2);
        assert!((wf[0] - (1.0f32 / 2f32.sqrt())).abs() < 1e-6);
        // Identity step reads EXACTLY 0 — the planted canary.
        assert_eq!(wf[1], 0.0);
        assert_eq!(stall_onset(&wf, 0.0), Some(2));
        // Degenerate norms.
        write_fractions([&[1.0f32][..], &[0.0f32][..]], &mut wf);
        assert_eq!(wf[0], f32::INFINITY);
        write_fractions([&[0.0f32][..], &[0.0f32][..]], &mut wf);
        assert_eq!(wf[0], 0.0);
        write_fractions([&[0.0f32][..], &[f32::NAN][..]], &mut wf);
        assert!(wf[0].is_nan());
        write_fractions([&[0.0f32][..], &[0.0f32, 1.0][..]], &mut wf);
        assert!(wf[0].is_nan());
    }

    #[test]
    fn stall_onset_needs_a_quiet_tail() {
        assert_eq!(stall_onset(&[0.5, 0.01, 0.2], 0.05), None);
        assert_eq!(stall_onset(&[0.5, 0.2, 0.01, 0.001], 0.05), Some(3));
        assert_eq!(stall_onset(&[0.01, 0.001], 0.05), Some(1));
        assert_eq!(stall_onset(&[], 0.05), None);
        assert_eq!(stall_onset(&[0.01, f32::NAN], 0.05), None);
    }

    #[test]
    fn monotone_and_spread() {
        assert!(is_monotone_nonincreasing(&[3.0, 2.0, 2.0, 1.0], 0.0));
        assert!(!is_monotone_nonincreasing(&[3.0, 2.0, 2.5], 0.0));
        assert!(is_monotone_nonincreasing(&[3.0, 2.0, 2.1], 0.1));
        assert!(!is_monotone_nonincreasing(&[1.0, f32::NAN], 1.0));
        assert!(is_monotone_nonincreasing(&[], 0.0));
        assert_eq!(spread(&[1.0, 3.0, f32::NAN, 2.0]), Some(2.0));
        assert_eq!(spread(&[f32::NAN]), None);
    }

    #[test]
    fn agreement_exit_oracle() {
        assert_eq!(agreement_exit(&[4, 4, 4]), Some(1));
        assert_eq!(agreement_exit(&[1, 2, 4, 4]), Some(3));
        // Agrees early, flips, returns: the stable tail decides.
        assert_eq!(agreement_exit(&[4, 2, 4]), Some(3));
        assert_eq!(agreement_exit(&[]), None);
    }

    #[test]
    fn fit_and_holdout() {
        let a = [0.9f32, 0.3, 0.02, 0.0];
        let b = [0.8f32, 0.05, 0.01, 0.0];
        let train = [
            DepthSample {
                kl: &a,
                oracle_exit: 3,
            },
            DepthSample {
                kl: &b,
                oracle_exit: 2,
            },
        ];
        let fit = fit_threshold(&train, &[0.001, 0.1, 0.5]).unwrap();
        assert_eq!(fit.threshold, 0.1);
        assert_eq!(fit.mean_abs_error, 0.0);
        assert_eq!(fit.unpredicted, 0);
        assert_eq!(holdout_hit_rate(&train, 0.1, 0), Some(1.0));
        // 0.5 predicts a→2, b→2: one miss at tolerance 0, both hit at 1.
        assert_eq!(holdout_hit_rate(&train, 0.5, 0), Some(0.5));
        assert_eq!(holdout_hit_rate(&train, 0.5, 1), Some(1.0));
        assert!(fit_threshold(&[], &[0.1]).is_none());
        assert!(fit_threshold(&train, &[]).is_none());
        assert!(holdout_hit_rate(&[], 0.1, 1).is_none());
    }

    #[test]
    fn fit_tie_breaks_to_larger_threshold() {
        let a = [0.9f32, 0.0];
        let s = [DepthSample {
            kl: &a,
            oracle_exit: 2,
        }];
        // Every candidate below 0.9 predicts exit 2 exactly.
        let fit = fit_threshold(&s, &[0.01, 0.5, 0.1]).unwrap();
        assert_eq!(fit.threshold, 0.5);
    }

    #[test]
    fn fit_cannot_win_by_predicting_nothing() {
        // Reference not last: threshold 0.0 is unreachable (costs len=2),
        // threshold 0.45 predicts exit 2 with error 0.
        let a = [0.5f32, 0.4];
        let s = [DepthSample {
            kl: &a,
            oracle_exit: 2,
        }];
        let fit = fit_threshold(&s, &[0.0, 0.45]).unwrap();
        assert_eq!(fit.threshold, 0.45);
        let bad = fit_threshold(&s, &[0.0]).unwrap();
        assert_eq!(bad.unpredicted, 1);
        assert_eq!(bad.mean_abs_error, 2.0);
    }

    #[test]
    fn histogram() {
        let mut h = DepthHistogram::<4>::default();
        assert!(!h.record(0));
        assert!(h.record(1) && h.record(3) && h.record(3) && h.record(9));
        assert_eq!(h.counts, [1, 0, 2, 0]);
        assert_eq!(h.overflow, 1);
        assert_eq!(h.total(), 4);
        assert!((h.mean_in_range().unwrap() - 7.0 / 3.0).abs() < 1e-12);
        assert_eq!(DepthHistogram::<2>::default().mean_in_range(), None);
    }
}
