//! Selection hysteresis latch — issue 987 Temporal-MoE follow-up lead
//! (Research 657 §A1; **ours**, not a paper claim).
//!
//! A scored candidate table (lane/domain/LRU row) is re-scored every probe;
//! an argmax re-decides every call and a near-tie band flips the pick back
//! and forth on pure scoring noise. The latch keeps the previous pick
//! unless the challenger's score beats the held pick's score by a sigmoid
//! margin, and an abstain probe REUSES the held pick instead of
//! re-deciding on noise.
//!
//! # The decision rule
//!
//! Per probe with scores `s[0..N]` and held pick `h`:
//!
//! 1. challenger = argmax over `s` (`crate::float_order::cmp_for_max` +
//!    first-index tie — the `pick_domain` comparator, so a latch seeded
//!    cold selects exactly what `pick_domain` would);
//! 2. switch iff `challenger != h` AND
//!    `exact_sigmoid((s[challenger] − s[h]) / temperature) >= threshold`.
//!
//! `exact_sigmoid` (not the fast approximation) because a pick is a
//! committed-value decision — the `crate::sigmoid` doc's own reward-gate
//! law — and because it makes the gate exact: the switch condition is the
//! additive margin `s[challenger] − s[h] >= temperature · lnit(threshold)`
//! (at the defaults, `ln(3) ≈ 1.0986` score units). Sigmoid, never
//! softmax (house law): the gate is a one-vs-held comparison, so there is
//! nothing to normalize over.
//!
//! # Abstain ⇒ reuse
//!
//! `decide(None)` (and `decide(Some(&[]))`) never re-decides: it returns
//! the held pick unchanged, or [`LatchDecision::NoPick`] on a cold abstain.
//! The abstention signal arrives upstream of the latch (the caller's
//! admission/distance gate refused the frame) — the latch is not in the
//! business of second-guessing it.
//!
//! # State surface — the stationary-surface argument
//!
//! The ONLY mutated field is `held: Option<usize>` (the lead's
//! "per-session last-pick + margin": the margin pair is configuration,
//! fixed at construction). Finite state + deterministic transitions means
//! every input stream has a fixed point: once `held == challenger`, the
//! gate is never consulted again for the same frame, and a challenger
//! inside the margin band cannot displace the incumbent. The
//! stationary-surface fixtures below pin the fixed point, the noise band,
//! and the byte-identical replay of a mixed stream.
//!
//! # Consumers (named in issue 987, wired T1 — not here)
//!
//! - reflex `pick_domain` (the corpus router): latch over the per-domain
//!   affinities so routing stops flip-flopping inside the tie band.
//! - instinct LRU eviction path: latch the eviction pick so a lane whose
//!   eviction score hovers around another's does not ping-pong in and out
//!   of residency.
//!
//! # Contract edges, pinned
//!
//! - **Positional stability**: scores are indexed positions of a stable
//!   candidate table. A table that SHRANK past `held` re-seats fresh (the
//!   incumbent no longer exists to defend); a table that only reordered
//!   silently changes what `held` names — callers with mutable tables
//!   should latch over ids, not positions (map ids→positions per probe;
//!   the primitive stays table-agnostic).
//! - **NaN holds**: `cmp_for_max` can never elect a NaN challenger, and a
//!   NaN held score makes the gate NaN → every comparison false → hold.
//!   Hysteresis resists the noisiest input; only a shape change re-seats.
//! - **No panic path**: `with_gate` clamps (`threshold` into the open
//!   `(0.5, 1)` band — ≤ 0.5 would switch on a non-positive advantage,
//!   which is not hysteresis, and ≥ 1 never switches, a dead latch;
//!   `temperature` into `[1e-3, 1e3]` — scale calibration is the
//!   caller's job, the clamps only refuse degenerate configs).
//! - **Zero allocation**, `#[inline]` hot path, deterministic per binary.
//!
//! Feature-gated (`pick_latch`), byte-identical off: nothing outside this
//! module references it. The `#[cfg(test)]` fixtures are SYNTHETIC
//! gate-mechanics fixtures — they prove the latch's decision arithmetic,
//! never serving quality (the 976-corpus lesson).

/// Default gate temperature (score units). The switch margin in score
/// units is `temperature · lnit(threshold)` — at the defaults,
/// `1.0 · ln 3 ≈ 1.0986`. Callers must calibrate this to their score
/// scale (dot-product affinities are unbounded; there is no universal
/// default — this one only fixes the shape).
pub const DEFAULT_LATCH_TEMPERATURE: f32 = 1.0;

/// Default switch threshold on the sigmoid gate — in the open `(0.5, 1)`
/// band. `0.75` ⇒ the challenger must clear `ln(3)` score units at the
/// default temperature.
pub const DEFAULT_LATCH_THRESHOLD: f32 = 0.75;

/// Threshold clamp floor — strictly above ½ so a non-positive advantage
/// can never switch.
pub const MIN_THRESHOLD: f32 = 0.501;

/// Threshold clamp ceiling — strictly below 1 so the latch can never
/// become a silent dead config.
pub const MAX_THRESHOLD: f32 = 0.999;

/// Temperature clamp floor.
pub const MIN_TEMPERATURE: f32 = 1e-3;

/// Temperature clamp ceiling.
pub const MAX_TEMPERATURE: f32 = 1e3;

/// What one [`PickLatch::decide`] probe decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatchDecision {
    /// The held pick stayed (abstain-reuse, or the challenger did not
    /// clear the sigmoid margin, or the challenger IS the held pick).
    Held(usize),
    /// The pick moved: cold start seating, a shape-change re-seat, or the
    /// challenger cleared the margin. The index is installed as held.
    Switched(usize),
    /// Abstain with nothing held — nothing to reuse.
    NoPick,
}

impl LatchDecision {
    /// The effective pick of this decision, if any.
    #[inline]
    pub const fn pick(&self) -> Option<usize> {
        match *self {
            LatchDecision::Held(i) | LatchDecision::Switched(i) => Some(i),
            LatchDecision::NoPick => None,
        }
    }

    /// `true` when this decision moved the pick (or seated the first one).
    #[inline]
    pub const fn is_switch(&self) -> bool {
        matches!(self, LatchDecision::Switched(_))
    }
}

/// Selection hysteresis latch over a scored candidate table.
///
/// Construction is clamped (no panic path); the only mutated field is
/// [`Self::held`]'s backing index — see the module doc's
/// stationary-surface argument.
#[derive(Clone, Copy, Debug)]
pub struct PickLatch {
    held: Option<usize>,
    temperature: f32,
    threshold: f32,
}

impl PickLatch {
    /// Defaults: temperature [`DEFAULT_LATCH_TEMPERATURE`], threshold
    /// [`DEFAULT_LATCH_THRESHOLD`], nothing held.
    pub const fn new() -> Self {
        Self {
            held: None,
            temperature: DEFAULT_LATCH_TEMPERATURE,
            threshold: DEFAULT_LATCH_THRESHOLD,
        }
    }

    /// Explicit gate calibration; both inputs clamp (no panic path).
    pub fn with_gate(temperature: f32, threshold: f32) -> Self {
        Self {
            held: None,
            temperature: clamp_finite(temperature, MIN_TEMPERATURE, MAX_TEMPERATURE),
            threshold: clamp_finite(threshold, MIN_THRESHOLD, MAX_THRESHOLD),
        }
    }

    /// The currently held pick.
    #[inline]
    pub const fn held(&self) -> Option<usize> {
        self.held
    }

    /// Clear the held pick (next probe seats fresh).
    #[inline]
    pub fn reset(&mut self) {
        self.held = None;
    }

    /// Decide one probe. `None` (or an empty slice) is an abstain: reuse
    /// the held pick, never re-decide on noise. See the module doc for
    /// the full rule and the pinned contract edges.
    #[inline]
    pub fn decide(&mut self, probe: Option<&[f32]>) -> LatchDecision {
        let scores = match probe {
            Some(s) if !s.is_empty() => s,
            _ => {
                return match self.held {
                    Some(i) => LatchDecision::Held(i),
                    None => LatchDecision::NoPick,
                };
            }
        };
        let held = match self.held {
            Some(i) if i < scores.len() => i,
            // Cold start, or the table shrank past the held pick — seat
            // the argmax fresh (there is no incumbent to defend).
            _ => {
                let (best, _) = argmax_first_index(scores);
                self.held = Some(best);
                return LatchDecision::Switched(best);
            }
        };
        let (challenger, challenger_score) = argmax_first_index(scores);
        if challenger == held {
            return LatchDecision::Held(held);
        }
        let held_score = scores[held];
        // A NaN held score cannot be compared — hold. (The challenger can
        // never be NaN by the comparator; the gate would still hold via
        // NaN comparisons being false, but the early return keeps that
        // accident out of the hot path.)
        if held_score.is_nan() {
            return LatchDecision::Held(held);
        }
        let gate = crate::exact_sigmoid((challenger_score - held_score) / self.temperature);
        if gate >= self.threshold {
            self.held = Some(challenger);
            LatchDecision::Switched(challenger)
        } else {
            LatchDecision::Held(held)
        }
    }
}

impl Default for PickLatch {
    fn default() -> Self {
        Self::new()
    }
}

/// `clamp` that also maps NaN to the floor (a NaN config must not silently
/// pass through `f32::clamp` as NaN — `f32::clamp` panics on NaN min/max
/// only, but a NaN `value` stays NaN and would poison every comparison).
#[inline]
fn clamp_finite(value: f32, lo: f32, hi: f32) -> f32 {
    if value.is_nan() {
        return lo;
    }
    value.clamp(lo, hi)
}

/// Argmax under the house comparator: `cmp_for_max` (NaN below every
/// real) + first-index tie — the `pick_domain` selection shape. Returns
/// `(index, score)`; never selects a NaN score unless every score is NaN
/// (then index 0).
#[inline]
fn argmax_first_index(scores: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_score = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if crate::float_order::cmp_for_max(s, best_score).is_gt() {
            best = i;
            best_score = s;
        }
    }
    (best, best_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent reference for the cold-start argmax: first index whose
    /// score is not beaten by any LATER score (first-index tie rule),
    /// NaN-never-elected.
    fn reference_argmax(scores: &[f32]) -> usize {
        let mut best = 0usize;
        for i in 1..scores.len() {
            if scores[i] > scores[best] || (scores[best].is_nan() && !scores[i].is_nan()) {
                best = i;
            }
        }
        best
    }

    #[test]
    fn cold_start_seats_argmax_first_index_tie() {
        let mut latch = PickLatch::new();
        assert_eq!(
            latch.decide(Some(&[0.3, 1.2, 1.2, 0.1])),
            LatchDecision::Switched(1),
            "exact tie seats the LOWER index"
        );
        assert_eq!(latch.held(), Some(1));
    }

    #[test]
    fn cold_start_matches_reference_across_random_tables() {
        // Deterministic LCG — same-binary determinism, no unseeded global
        // RNG (the Issue-809 law).
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 2000) as i32 as f32 / 100.0 - 10.0
        };
        for len in 1..=24usize {
            let scores: Vec<f32> = (0..len).map(|_| next()).collect();
            let mut latch = PickLatch::new();
            let d = latch.decide(Some(&scores));
            assert_eq!(
                d.pick(),
                Some(reference_argmax(&scores)),
                "scores={scores:?}"
            );
        }
    }

    #[test]
    fn identical_probe_reaches_a_fixed_point() {
        // The stationary-surface law: repeated identical probes hold after
        // at most the initial seat, forever.
        let scores = [0.4, 0.9, 0.2];
        let mut latch = PickLatch::new();
        assert!(latch.decide(Some(&scores)).is_switch());
        for _ in 0..100 {
            assert_eq!(latch.decide(Some(&scores)), LatchDecision::Held(1));
        }
    }

    #[test]
    fn noise_band_never_switches() {
        // Challenger hovering inside the margin band never displaces the
        // incumbent (defaults: margin = ln 3 ≈ 1.0986 score units).
        let mut latch = PickLatch::new();
        assert!(latch.decide(Some(&[2.0, 1.0])).is_switch());
        for jitter in [1.5, 2.9, 3.0, 0.2, 1.9, 2.99] {
            // challenger idx1 at 1.0 + jitter stays < 2.0 + ln 3 = 3.0986.
            assert_eq!(
                latch.decide(Some(&[2.0, jitter])),
                LatchDecision::Held(0),
                "jitter={jitter}"
            );
        }
    }

    #[test]
    fn clear_margin_switches_then_stationary() {
        let mut latch = PickLatch::new();
        assert_eq!(latch.decide(Some(&[1.0, 0.0])), LatchDecision::Switched(0));
        // Challenger clears the incumbent by > ln 3.
        assert_eq!(latch.decide(Some(&[1.0, 2.2])), LatchDecision::Switched(1));
        for _ in 0..10 {
            assert_eq!(latch.decide(Some(&[1.0, 2.2])), LatchDecision::Held(1));
        }
    }

    #[test]
    fn switch_side_follows_the_gate_not_the_sign() {
        // The decision must agree with `gate >= threshold` computed by the
        // same function the impl uses — pinning >= (boundary switches),
        // not >.
        let mut latch = PickLatch::new();
        latch.decide(Some(&[0.0, 0.0])); // hold index 0
        for adv in [-2.0, 0.0, 0.5, 1.0, 1.0986, 1.0987, 3.0, 40.0, 1e9] {
            let scores = [0.0, adv];
            let expected_gate = crate::exact_sigmoid(adv / DEFAULT_LATCH_TEMPERATURE);
            let expected = if expected_gate >= DEFAULT_LATCH_THRESHOLD {
                LatchDecision::Switched(1)
            } else {
                LatchDecision::Held(0)
            };
            assert_eq!(latch.decide(Some(&scores)), expected, "adv={adv}");
            if expected.is_switch() {
                // Re-seat index 0 for the next iteration.
                latch.reset();
                latch.decide(Some(&[0.0, 0.0]));
            }
        }
    }

    #[test]
    fn abstain_reuses_without_mutation() {
        let mut latch = PickLatch::new();
        assert_eq!(latch.decide(Some(&[0.1, 5.0])), LatchDecision::Switched(1));
        assert_eq!(latch.decide(None), LatchDecision::Held(1));
        assert_eq!(latch.decide(Some(&[])), LatchDecision::Held(1));
        assert_eq!(latch.held(), Some(1), "abstain never mutates");
        // A cold abstain has nothing to reuse.
        let mut cold = PickLatch::new();
        assert_eq!(cold.decide(None), LatchDecision::NoPick);
        assert_eq!(cold.decide(Some(&[])), LatchDecision::NoPick);
    }

    #[test]
    fn shrunken_table_reseats_fresh() {
        let mut latch = PickLatch::new();
        assert!(latch.decide(Some(&[0.0, 0.1, 0.2, 0.3, 4.0])).is_switch());
        assert_eq!(latch.held(), Some(4));
        // Table shrank past the held pick — no incumbent to defend.
        assert_eq!(latch.decide(Some(&[0.5, 0.1])), LatchDecision::Switched(0));
        assert_eq!(latch.held(), Some(0));
    }

    #[test]
    fn nan_held_score_holds() {
        let mut latch = PickLatch::new();
        assert!(latch.decide(Some(&[1.0, 2.0])).is_switch());
        assert_eq!(latch.decide(Some(&[1.0, f32::NAN])), LatchDecision::Held(1));
        // And a NaN challenger cannot win the argmax at all.
        let mut cold = PickLatch::new();
        assert_eq!(
            cold.decide(Some(&[f32::NAN, 0.5])).pick(),
            Some(1),
            "NaN below every real value"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut latch = PickLatch::new();
        latch.decide(Some(&[1.0, 0.0]));
        assert_eq!(latch.held(), Some(0));
        latch.reset();
        assert_eq!(latch.held(), None);
        assert_eq!(latch.decide(None), LatchDecision::NoPick);
        assert!(latch.decide(Some(&[0.0, 3.0])).is_switch());
    }

    #[test]
    fn decision_trace_is_byte_identical_across_replays() {
        let stream: [Option<&[f32]>; 8] = [
            Some(&[1.0, 0.5]),
            Some(&[1.0, 1.4]),      // inside the band — hold
            Some(&[1.0, 2.3]),      // clears — switch
            None,                   // abstain — reuse
            Some(&[2.3, 2.2]),      // held challenger wins — hold
            Some(&[f32::NAN, 2.2]), // NaN challenger — hold
            Some(&[9.9]),           // shape change — reseat
            Some(&[]),
        ];
        let run = || {
            let mut latch = PickLatch::new();
            let trace: Vec<LatchDecision> = stream.iter().map(|p| latch.decide(*p)).collect();
            (trace, latch.held())
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn with_gate_clamps_degenerate_configs_without_panicking() {
        let l = PickLatch::with_gate(f32::NAN, f32::NAN);
        assert_eq!(l.temperature, MIN_TEMPERATURE);
        assert_eq!(l.threshold, MIN_THRESHOLD);
        let l = PickLatch::with_gate(0.0, 0.2); // sub-½ threshold is not hysteresis
        assert_eq!(l.threshold, MIN_THRESHOLD);
        let l = PickLatch::with_gate(1e9, 1.0); // threshold 1 = dead latch
        assert_eq!(l.threshold, MAX_THRESHOLD);
        assert_eq!(l.temperature, MAX_TEMPERATURE);
        // The clamped latch still decides.
        let mut l = PickLatch::with_gate(-5.0, 42.0);
        assert!(l.decide(Some(&[1.0, 2.0])).is_switch());
    }

    #[test]
    fn derived_margin_matches_the_documented_analytic_form() {
        // exact_sigmoid makes the gate exact: switch iff
        // advantage >= temperature · lnit(threshold). Sample the boundary
        // from both sides at a non-default gate.
        let (t, thr) = (0.5_f32, 0.9_f32);
        let margin = t * (thr / (1.0 - thr)).ln(); // lnit(0.9) = ln 9
        let mut latch = PickLatch::with_gate(t, thr);
        latch.decide(Some(&[1.0, 1.0])); // hold 0
        // Just below the margin — hold. Just above — switch.
        assert_eq!(
            latch.decide(Some(&[1.0, 1.0 + margin - 1e-4])),
            LatchDecision::Held(0)
        );
        latch.reset();
        latch.decide(Some(&[1.0, 1.0]));
        assert_eq!(
            latch.decide(Some(&[1.0, 1.0 + margin + 1e-4])),
            LatchDecision::Switched(1)
        );
    }

    #[test]
    fn held_pick_beating_itself_is_not_a_switch() {
        // The held pick IS the argmax — the gate is never consulted.
        let mut latch = PickLatch::new();
        latch.decide(Some(&[0.5, 0.4]));
        assert_eq!(latch.decide(Some(&[9.9, 0.4])), LatchDecision::Held(0));
    }

    #[test]
    fn defaults_are_in_the_hysteresis_band() {
        assert!(DEFAULT_LATCH_TEMPERATURE > 0.0);
        assert!(DEFAULT_LATCH_THRESHOLD > 0.5 && DEFAULT_LATCH_THRESHOLD < 1.0);
        assert!(MIN_THRESHOLD > 0.5 && MAX_THRESHOLD < 1.0 && MIN_THRESHOLD < MAX_THRESHOLD);
        assert!(MIN_TEMPERATURE > 0.0 && MIN_TEMPERATURE < MAX_TEMPERATURE);
    }
}
