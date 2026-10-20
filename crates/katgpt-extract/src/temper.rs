//! Temper — the calibration/abstention gate (Plan 628 T0.5, gate half).
//!
//! The gate half of Proposal 016 §3.3: per-field-kind residual pools re-fit
//! from realized outcomes, the sigmoid confidence readout, the fused
//! abstain, and the wire-shaped answer assembly. The engine half (which
//! upstream rung to escalate, the Elo/Beta strategy routing) is Phase 2 —
//! never here.
//!
//! # Confidence readout — the Bench-817 dispatch, INHERITED
//!
//! The narrow/wide dispatch precedent (katgpt-rs Bench 817's confidence
//! readout verdict) is inherited, never re-derived: a NARROW answer scores
//! by its own margin scaled by the calibration spread —
//! `confidence = sigmoid(raw_margin / spread)` — and a WIDE answer (many
//! candidates, no meaningful margin) scores by corpus distance —
//! [`TemperGate::confidence_distance`] `= sigmoid((best_score − δ)·k)`.
//! Sigmoid everywhere, never softmax (the house law).
//!
//! # Fused abstain law
//!
//! [`TemperGate::gate_located`] answers iff the calibrated confidence ≥
//! `conf_tau` AND the best candidate score ≥ `dist_delta` (confidence ∧
//! distance). Every other outcome abstains; the abstain is GUESSABLE when a
//! candidate exists at all (`best_score > 0`) — the null-with-guess path.
//! The gate emits the VERDICT; the engine assembles the wire answer through
//! [`TemperGate::assemble_answer`], which enforces the extraction wire's
//! answer laws BY CONSTRUCTION (a guess never coexists with a value, never
//! carries a span; a null never carries a span; computed answers never
//! carry a span).
//!
//! # Zero-alloc, deterministic
//!
//! The gate path allocates nothing: pools are fixed `[f32; 256]` rings, the
//! spread sort runs in one fixed scratch. No clock, no rng; the spread is a
//! pure function of the pushed residuals.

use katgpt_core::extraction_wire::{FieldAnswer, FieldKind, FieldSpec, FieldValue, Span};

/// Compile-time ring capacity per field kind. [`TemperConfig::pool_cap`]
/// clamps the EFFECTIVE window (the newest `min(len, pool_cap)` residuals)
/// on read; caps above 256 clamp to the ring size. Runtime-sized pools are
/// deliberately NOT shipped — the gate's memory is fixed and bounded.
const RING_CAPACITY: usize = 256;

/// Floor on the normalized spread (an all-identical residual pool has MAD 0
/// and would put the confidence on a knife edge at `sigmoid(±∞)`).
const CONF_EPS: f32 = 1e-6;

/// The neutral spread for an empty/too-fresh pool: `confidence =
/// sigmoid(margin / 1)` — the raw margin readout until calibration evidence
/// exists.
const NEUTRAL_SPREAD: f32 = 1.0;

/// MAD → σ consistency constant for a normal distribution
/// (`σ ≈ MAD / 0.6745`): the "normalized MAD" of the spread readout.
const MAD_TO_SIGMA: f32 = 0.674_49;

/// The distance-dispatch slope (the Bench-817 inherited readout's k):
/// `sigmoid((best_score − dist_delta)·k)` sweeps ≈ 0.18–0.83 over
/// `dist_delta ± 0.25`. A documented constant, retuned by the T0.6 GOAT —
/// not a tunable knob.
pub const DISTANCE_SLOPE_K: f32 = 8.0;

/// Number of [`FieldKind`] variants — one residual pool per kind.
const N_KINDS: usize = 9;

/// The gate's configuration. Documented defaults; the T0.6 GOAT re-tunes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemperConfig {
    /// The confidence bar for answering. Default `0.6`.
    pub conf_tau: f32,
    /// The best-candidate score bar for answering (the corpus-distance half
    /// of the fused law). Default `0.35`.
    pub dist_delta: f32,
    /// The null prior floored into `p_null` on every abstain. Default `0.5`.
    pub null_tau: f32,
    /// The effective residual window per kind (newest-first; clamped to the
    /// [`RING_CAPACITY`] ring). Default `256`.
    pub pool_cap: usize,
}

impl Default for TemperConfig {
    fn default() -> Self {
        Self {
            conf_tau: 0.6,
            dist_delta: 0.35,
            null_tau: 0.5,
            pool_cap: 256,
        }
    }
}

/// The gate's verdict for one located-field candidate set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateVerdict {
    /// Confident enough to answer: confidence ≥ `conf_tau` AND the best
    /// score ≥ `dist_delta`.
    Answer,
    /// Abstain (null is the answer). `guessable` = a candidate exists at all
    /// (`best_score > 0`) — the null-with-guess path.
    Abstain {
        /// Whether the abstaining answer may carry the engine's low-
        /// confidence candidate as its disclosed guess.
        guessable: bool,
    },
}

/// The last confidence readout, carried from [`TemperGate::gate_located`]
/// into [`TemperGate::assemble_answer`].
#[derive(Clone, Copy, Debug)]
struct GateReadout {
    confidence: f32,
    p_null: f32,
}

/// A fixed-capacity residual ring: `RING_CAPACITY` slots, write head, live
/// length. No deps, no allocation; the oldest entry is overwritten first.
struct ResidualRing {
    buf: [f32; RING_CAPACITY],
    head: usize,
    len: usize,
}

impl ResidualRing {
    const fn new() -> Self {
        Self {
            buf: [0.0; RING_CAPACITY],
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, x: f32) {
        self.buf[self.head] = x;
        self.head = (self.head + 1) % RING_CAPACITY;
        if self.len < RING_CAPACITY {
            self.len += 1;
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Copy the newest `want` entries (oldest-first) into `out`; returns how
    /// many were copied.
    fn copy_window(&self, out: &mut [f32], want: usize) -> usize {
        let take = self.len.min(want).min(out.len());
        let start = self.len - take;
        for (i, slot) in out.iter_mut().take(take).enumerate() {
            let ring_idx = (self.head + RING_CAPACITY - self.len + start + i) % RING_CAPACITY;
            *slot = self.buf[ring_idx];
        }
        take
    }
}

/// Field-kind → pool index (a stable bijection over the nine wire kinds).
fn kind_index(kind: FieldKind) -> usize {
    match kind {
        FieldKind::Text => 0,
        FieldKind::Number => 1,
        FieldKind::Date => 2,
        FieldKind::Time => 3,
        FieldKind::DateTime => 4,
        FieldKind::Duration => 5,
        FieldKind::List => 6,
        FieldKind::Estimate => 7,
        FieldKind::Forecast => 8,
    }
}

/// The plain logistic function `1/(1 + e^(−x))`, implemented locally (the
/// gate never reaches for the core's fast-exp variants — this readout is
/// µs-path, not ns-path, and plain `exp` is platform-portable). Bounds: 0.5
/// at 0; strictly inside (0, 1) for moderate finite `x`; SATURATING to
/// exactly 0/1 for large |x| (f32 `1 + ε` rounds); non-finite input
/// propagates NaN — [`TemperGate::confidence`] sanitizes the readout.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The calibration/abstention gate.
pub struct TemperGate {
    config: TemperConfig,
    pools: [ResidualRing; N_KINDS],
    scratch: [f32; RING_CAPACITY],
    last: Option<GateReadout>,
}

impl TemperGate {
    /// A gate with the given configuration.
    pub fn new(config: TemperConfig) -> Self {
        Self {
            config,
            pools: core::array::from_fn(|_| ResidualRing::new()),
            scratch: [0.0; RING_CAPACITY],
            last: None,
        }
    }

    /// The active configuration.
    pub fn config(&self) -> &TemperConfig {
        &self.config
    }

    /// Feed one realized outcome: the residual `realized − predicted`
    /// enters `kind`'s pool. Non-finite residuals are ignored (a NaN never
    /// poisons the spread).
    pub fn update(&mut self, kind: FieldKind, realized: f32, predicted: f32) {
        let residual = realized - predicted;
        if residual.is_finite() {
            self.pools[kind_index(kind)].push(residual);
        }
    }

    /// Residuals currently held for `kind`.
    pub fn pool_len(&self, kind: FieldKind) -> usize {
        self.pools[kind_index(kind)].len()
    }

    /// The normalized calibration spread for `kind`: the median absolute
    /// deviation of the effective residual window (newest
    /// `min(len, pool_cap)` entries), scaled to a σ estimate (`MAD /
    /// 0.6745`), floored at [`CONF_EPS`] when the pool is degenerate
    /// (identical residuals) and neutral `1.0` with fewer than two
    /// residuals. Zero-alloc (one fixed scratch, exact insertion sort —
    /// bounded O(cap²), priced by the T0.6 G2 bench).
    pub fn calibration_spread(&mut self, kind: FieldKind) -> f32 {
        let cap = self.config.pool_cap.clamp(1, RING_CAPACITY);
        let ring = &self.pools[kind_index(kind)];
        let n = ring.copy_window(&mut self.scratch, cap);
        if n < 2 {
            return NEUTRAL_SPREAD;
        }
        insertion_sort(&mut self.scratch[..n]);
        let median = median_of_sorted(&self.scratch[..n]);
        for v in &mut self.scratch[..n] {
            *v = (*v - median).abs();
        }
        insertion_sort(&mut self.scratch[..n]);
        let mad = median_of_sorted(&self.scratch[..n]);
        let spread = mad / MAD_TO_SIGMA;
        if spread.is_finite() && spread > 0.0 {
            spread
        } else {
            CONF_EPS
        }
    }

    /// The narrow-dispatch confidence: the raw margin scaled by the
    /// calibration spread, through the sigmoid. A tight pool (small spread)
    /// sharpens the readout; an uncalibrated pool reads the raw margin
    /// (neutral spread 1.0). Wire-safe by construction: a non-finite margin
    /// (sigmoid propagates NaN) reads as ZERO confidence — the wire's
    /// probability laws require finite, and the fused law then abstains.
    pub fn confidence(&mut self, kind: FieldKind, raw_margin: f32) -> f32 {
        let spread = self.calibration_spread(kind).max(CONF_EPS);
        let raw = sigmoid(raw_margin / spread);
        if raw.is_finite() {
            raw.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// The wide-dispatch confidence: corpus distance through the sigmoid at
    /// the inherited slope ([`DISTANCE_SLOPE_K`]). For candidate sets with
    /// no meaningful margin, the best score against the distance delta δ is
    /// the evidence.
    pub fn confidence_distance(&self, best_score: f32) -> f32 {
        sigmoid((best_score - self.config.dist_delta) * DISTANCE_SLOPE_K)
    }

    /// The fused law over a located field's best candidate:
    /// ANSWER iff `confidence(kind, raw_margin) ≥ conf_tau` AND
    /// `best_score ≥ dist_delta`; otherwise ABSTAIN, guessable iff a
    /// candidate exists (`best_score > 0`).
    ///
    /// NaN-safe by construction: a NaN margin reads as zero confidence
    /// ([`TemperGate::confidence`]) → abstain; a NaN score fails `> 0` →
    /// not guessable. The readout is stored for
    /// [`TemperGate::assemble_answer`] (one gate call → one assembly; a
    /// second assembly without a new gate call reads the defaults).
    pub fn gate_located(
        &mut self,
        kind: FieldKind,
        best_score: f32,
        raw_margin: f32,
    ) -> GateVerdict {
        let confidence = self.confidence(kind, raw_margin);
        let answer = confidence >= self.config.conf_tau && best_score >= self.config.dist_delta;
        let p_null = if answer {
            // Answering: the null mass is the calibrated complement.
            (1.0 - confidence).clamp(0.0, 1.0)
        } else {
            // Abstaining: the calibrated complement, floored at the declared
            // null prior.
            self.config.null_tau.max(1.0 - confidence).clamp(0.0, 1.0)
        };
        self.last = Some(GateReadout { confidence, p_null });
        if answer {
            GateVerdict::Answer
        } else {
            GateVerdict::Abstain {
                guessable: best_score > 0.0,
            }
        }
    }

    /// Assemble the wire answer under the wire's answer laws, BY
    /// CONSTRUCTION:
    ///
    /// - An [`GateVerdict::Answer`] carries the value; the guess slot is
    ///   cleared (a guess never coexists with an answer); the span is
    ///   passed through for located kinds and cleared for computed kinds
    ///   (estimate/forecast are never located) and for a missing value (a
    ///   null never carries a span — an Answer verdict without a value
    ///   degrades to a bare null).
    /// - An [`GateVerdict::Abstain`] nulls the value and the span; a
    ///   guessable abstain carries the engine's explicit guess, demoting a
    ///   stray `value` into the guess slot when no explicit guess was
    ///   passed (guesses never carry spans — the span is dropped either
    ///   way).
    /// - `p_null` and `confidence` ride the last [`TemperGate::gate_located`]
    ///   readout (answer: `1 − confidence`; abstain: the complement floored
    ///   at `null_tau`). Without a prior gate call the defaults are
    ///   `confidence 0.0`, `p_null = null_tau`.
    ///
    /// The one allocation on this path is the wire's own `id: String` —
    /// assembly is the wire boundary, not the zero-alloc gate path.
    pub fn assemble_answer(
        &mut self,
        field: &FieldSpec,
        verdict: GateVerdict,
        value: Option<FieldValue>,
        span: Option<Span>,
        guess: Option<FieldValue>,
    ) -> FieldAnswer {
        let readout = self.last.take().unwrap_or(GateReadout {
            confidence: 0.0,
            p_null: self.config.null_tau,
        });
        let mut value = value;
        let mut span = span;
        let mut guess = guess;
        match verdict {
            GateVerdict::Answer => {
                guess = None;
                if value.is_none() || field.kind.is_computed() {
                    span = None;
                }
            }
            GateVerdict::Abstain { guessable } => {
                span = None;
                if guessable {
                    guess = guess.or_else(|| value.take());
                } else {
                    guess = None;
                }
                value = None;
            }
        }
        FieldAnswer {
            id: field.id.clone(),
            kind: field.kind,
            value,
            span,
            p_null: readout.p_null,
            confidence: readout.confidence,
            guess,
        }
    }
}

/// Exact, deterministic, allocation-free insertion sort (bounded by the
/// fixed ring capacity).
fn insertion_sort(buf: &mut [f32]) {
    for i in 1..buf.len() {
        let key = buf[i];
        let mut j = i;
        while j > 0 && buf[j - 1] > key {
            buf[j] = buf[j - 1];
            j -= 1;
        }
        buf[j] = key;
    }
}

/// Median of a SORTED slice (average of the two central entries for even
/// lengths — deterministic on any input).
fn median_of_sorted(buf: &[f32]) -> f32 {
    let n = buf.len();
    debug_assert!(n > 0, "median of empty slice");
    if n % 2 == 1 {
        buf[n / 2]
    } else {
        0.5 * (buf[n / 2 - 1] + buf[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::extraction_wire::{EstimateDistribution, FieldConstraints, WireError};

    const EPS: f32 = 1e-6;

    fn spec(id: &str, kind: FieldKind) -> FieldSpec {
        FieldSpec {
            id: id.to_string(),
            kind,
            description: None,
            constraints: FieldConstraints::default(),
            instructions: None,
        }
    }

    fn estimate_value() -> FieldValue {
        FieldValue::Estimate(EstimateDistribution {
            point: 100.0,
            p10: 90.0,
            median: 100.0,
            p90: 110.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 0.8,
        })
    }

    #[test]
    fn sigmoid_bounds_and_center() {
        assert!((sigmoid(0.0) - 0.5).abs() < EPS);
        assert!((sigmoid(100.0) - 1.0).abs() < EPS);
        assert!(sigmoid(-100.0).abs() < EPS);
        for x in [-5.0_f32, -1.0, -0.5, 0.5, 1.0, 5.0] {
            let s = sigmoid(x);
            assert!(s > 0.0 && s < 1.0, "sigmoid({x}) = {s} out of (0,1)");
        }
        // Saturating at f32 precision for large |x| (documented).
        assert_eq!(sigmoid(100.0), 1.0);
        assert_eq!(sigmoid(-100.0), 0.0);
        assert!((0.0_f32..=1.0).contains(&sigmoid(20.0)));
        // Monotone.
        assert!(sigmoid(-1.0) < sigmoid(1.0));
    }

    #[test]
    fn spread_is_deterministic_and_scales_with_pool() {
        let mut a = TemperGate::new(TemperConfig::default());
        let mut b = TemperGate::new(TemperConfig::default());
        for _ in 0..8 {
            a.update(FieldKind::Number, 0.01, 0.0);
            b.update(FieldKind::Number, 0.01, 0.0);
        }
        let sa = a.calibration_spread(FieldKind::Number);
        let sb = b.calibration_spread(FieldKind::Number);
        assert_eq!(sa, sb, "same updates → same spread");
        // Identical residuals → degenerate MAD → the CONF_EPS floor.
        assert!(
            sa <= CONF_EPS,
            "degenerate pool floors at CONF_EPS, got {sa}"
        );
        // A wide pool spreads wider than a tight one.
        let mut wide = TemperGate::new(TemperConfig::default());
        for r in [-1.0_f32, 1.0, -1.0, 1.0] {
            wide.update(FieldKind::Number, r, 0.0);
        }
        let s_wide = wide.calibration_spread(FieldKind::Number);
        // MAD of {−1,−1,1,1} = 1 → normalized ≈ 1.4826.
        assert!((s_wide - 1.0 / MAD_TO_SIGMA).abs() < 1e-3);
        // Fresh pool (< 2 residuals) reads neutral.
        let mut fresh = TemperGate::new(TemperConfig::default());
        fresh.update(FieldKind::Text, 1.0, 0.5);
        assert_eq!(fresh.calibration_spread(FieldKind::Text), NEUTRAL_SPREAD);
    }

    #[test]
    fn pool_cap_clamps_the_window() {
        let mut capped = TemperGate::new(TemperConfig {
            pool_cap: 2,
            ..TemperConfig::default()
        });
        for r in [0.1_f32, 0.1, 0.1, 10.0] {
            capped.update(FieldKind::Number, r, 0.0);
        }
        // Window = newest 2 = {0.1, 10.0}: median 5.05, MAD 4.95.
        let capped_spread = capped.calibration_spread(FieldKind::Number);
        assert!((capped_spread - 4.95 / MAD_TO_SIGMA).abs() < 1e-3);
        // Full window: median 0.1, deviations {0,0,0,9.9}, MAD 0 → floor.
        let mut full = TemperGate::new(TemperConfig::default());
        for r in [0.1_f32, 0.1, 0.1, 10.0] {
            full.update(FieldKind::Number, r, 0.0);
        }
        assert!(full.calibration_spread(FieldKind::Number) <= CONF_EPS);
        // Caps above the ring size clamp to it (no panic, deterministic).
        let mut over = TemperGate::new(TemperConfig {
            pool_cap: 10_000,
            ..TemperConfig::default()
        });
        over.update(FieldKind::Number, 1.0, 0.0);
        assert_eq!(over.calibration_spread(FieldKind::Number), NEUTRAL_SPREAD);
    }

    #[test]
    fn answer_path_when_confident_and_close() {
        // Fresh gate → neutral spread 1.0 → confidence = sigmoid(margin).
        let mut gate = TemperGate::new(TemperConfig::default());
        let verdict = gate.gate_located(FieldKind::Text, 0.5, 1.0);
        assert_eq!(verdict, GateVerdict::Answer);
        // sigmoid(1.0) ≈ 0.7311 ≥ conf_tau 0.6; p_null = 1 − confidence.
        let answer =
            gate.assemble_answer(&spec("vendor", FieldKind::Text), verdict, None, None, None);
        assert!((answer.confidence - sigmoid(1.0)).abs() < EPS);
        assert!((answer.p_null - (1.0 - sigmoid(1.0))).abs() < EPS);
    }

    #[test]
    fn abstain_paths_and_guessable_flag() {
        let mut gate = TemperGate::new(TemperConfig::default());
        // Low confidence, no candidate → plain abstain.
        let v1 = gate.gate_located(FieldKind::Text, 0.0, -1.0);
        assert_eq!(v1, GateVerdict::Abstain { guessable: false });
        // Candidate exists but below the distance bar AND low confidence →
        // abstain, guessable.
        let v2 = gate.gate_located(FieldKind::Text, 0.1, -1.0);
        assert_eq!(v2, GateVerdict::Abstain { guessable: true });
        // Confident but below the distance bar → still abstain (the fused
        // law is a conjunction).
        let v3 = gate.gate_located(FieldKind::Text, 0.1, 10.0);
        assert_eq!(v3, GateVerdict::Abstain { guessable: true });
        // Abstain p_null floors at null_tau (confidence near 1 ⇒ 1−conf ≈ 0,
        // the prior holds the floor).
        let answer = gate.assemble_answer(&spec("vendor", FieldKind::Text), v3, None, None, None);
        assert!((answer.p_null - 0.5).abs() < EPS);
        assert!((answer.confidence - sigmoid(10.0)).abs() < EPS);
    }

    #[test]
    fn nan_margin_degrades_to_abstain() {
        let mut gate = TemperGate::new(TemperConfig::default());
        let verdict = gate.gate_located(FieldKind::Number, 0.0, f32::NAN);
        assert_eq!(verdict, GateVerdict::Abstain { guessable: false });
        let answer = gate.assemble_answer(&spec("n", FieldKind::Number), verdict, None, None, None);
        assert!(
            answer.p_null.is_finite() && (0.0..=1.0).contains(&answer.p_null),
            "NaN never reaches p_null"
        );
        assert!(
            answer.confidence.is_finite() && (0.0..=1.0).contains(&answer.confidence),
            "NaN never reaches confidence"
        );
    }

    #[test]
    fn assemble_moves_value_to_guess_and_drops_span() {
        let mut gate = TemperGate::new(TemperConfig::default());
        let verdict = gate.gate_located(FieldKind::Text, 0.1, -5.0);
        assert_eq!(verdict, GateVerdict::Abstain { guessable: true });
        let answer = gate.assemble_answer(
            &spec("vendor", FieldKind::Text),
            verdict,
            Some(FieldValue::Text("acme".to_string())),
            Some(Span { start: 0, end: 4 }),
            None,
        );
        assert!(answer.value.is_none(), "abstain nulls the value");
        assert!(answer.span.is_none(), "guesses and nulls never carry spans");
        assert_eq!(
            answer.guess,
            Some(FieldValue::Text("acme".to_string())),
            "the stray value is demoted to the disclosed guess"
        );
        // Wire-valid by construction.
        assert!(
            answer
                .validate_against(&spec("vendor", FieldKind::Text), 100, 0)
                .is_ok(),
            "abstain+guess answer must pass the wire's own validation"
        );
    }

    #[test]
    fn assemble_never_carries_both_value_and_guess() {
        let mut gate = TemperGate::new(TemperConfig::default());
        let verdict = gate.gate_located(FieldKind::Text, 0.5, 1.0);
        assert_eq!(verdict, GateVerdict::Answer);
        let answer = gate.assemble_answer(
            &spec("vendor", FieldKind::Text),
            verdict,
            Some(FieldValue::Text("acme".to_string())),
            Some(Span { start: 0, end: 4 }),
            Some(FieldValue::Text("other".to_string())),
        );
        assert!(answer.value.is_some());
        assert!(answer.guess.is_none(), "an answer never carries a guess");
        assert!(
            answer
                .validate_against(&spec("vendor", FieldKind::Text), 100, 0)
                .is_ok(),
            "plain answer must pass the wire's own validation"
        );
    }

    #[test]
    fn assemble_computed_answer_drops_span() {
        let mut gate = TemperGate::new(TemperConfig::default());
        let verdict = gate.gate_located(FieldKind::Estimate, 0.5, 1.0);
        assert_eq!(verdict, GateVerdict::Answer);
        let answer = gate.assemble_answer(
            &spec("price", FieldKind::Estimate),
            verdict,
            Some(estimate_value()),
            Some(Span { start: 0, end: 4 }),
            None,
        );
        assert!(answer.span.is_none(), "computed answers are never located");
        assert!(answer.guess.is_none());
        assert!(
            answer
                .validate_against(&spec("price", FieldKind::Estimate), 100, 0)
                .is_ok(),
            "computed answer must pass the wire's own validation"
        );
    }

    #[test]
    fn assemble_located_answer_passes_span_through() {
        let mut gate = TemperGate::new(TemperConfig::default());
        let verdict = gate.gate_located(FieldKind::Text, 0.5, 1.0);
        let answer = gate.assemble_answer(
            &spec("vendor", FieldKind::Text),
            verdict,
            Some(FieldValue::Text("acme".to_string())),
            Some(Span { start: 3, end: 7 }),
            None,
        );
        assert_eq!(answer.span, Some(Span { start: 3, end: 7 }));
        assert!(
            answer
                .validate_against(&spec("vendor", FieldKind::Text), 100, 0)
                .is_ok(),
            "located answer must pass the wire's own validation"
        );
        // And the wire REFUSES the same answer without the span — the gate
        // passes through; the located law stays the engine's to satisfy.
        let spanless = gate.assemble_answer(
            &spec("vendor", FieldKind::Text),
            GateVerdict::Answer,
            Some(FieldValue::Text("acme".to_string())),
            None,
            None,
        );
        assert!(matches!(
            spanless.validate_against(&spec("vendor", FieldKind::Text), 100, 0),
            Err(WireError::ValueWithoutSpan { .. })
        ));
    }

    #[test]
    fn assemble_without_gate_readout_uses_defaults() {
        let mut gate = TemperGate::new(TemperConfig::default());
        let answer = gate.assemble_answer(
            &spec("vendor", FieldKind::Text),
            GateVerdict::Abstain { guessable: false },
            None,
            None,
            None,
        );
        assert_eq!(answer.confidence, 0.0);
        assert!((answer.p_null - 0.5).abs() < EPS, "null_tau default");
        assert_eq!(answer.id, "vendor");
        assert_eq!(answer.kind, FieldKind::Text);
        assert!(answer.value.is_none() && answer.span.is_none() && answer.guess.is_none());
        assert!(
            answer
                .validate_against(&spec("vendor", FieldKind::Text), 100, 0)
                .is_ok(),
            "bare null must pass the wire's own validation"
        );
    }

    #[test]
    fn non_finite_residuals_never_enter_the_pool() {
        let mut gate = TemperGate::new(TemperConfig::default());
        gate.update(FieldKind::Number, f32::NAN, 0.0);
        gate.update(FieldKind::Number, 1.0, f32::INFINITY);
        assert_eq!(gate.pool_len(FieldKind::Number), 0);
        assert_eq!(gate.calibration_spread(FieldKind::Number), NEUTRAL_SPREAD);
    }

    // ---- G4: allocation gate (crate::alloc_gate; see its module doc) -------

    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn gate_hot_paths_are_zero_alloc() {
        use crate::alloc_gate::{ALLOC_COUNT, MeasuringWindow};
        use std::sync::atomic::Ordering;

        // Pre-fill every read pool past the ring wrap so the measured reads
        // run at steady state (the sorted window is the full pool_cap).
        let mut gate = TemperGate::new(TemperConfig::default());
        for i in 0..(RING_CAPACITY * 2) {
            let residual = (i % 16) as f32 * 0.05;
            gate.update(FieldKind::Number, residual + 0.1, residual);
        }
        assert_eq!(gate.pool_len(FieldKind::Number), RING_CAPACITY);

        // `assemble_answer` is EXCLUDED by design: its `id: String` is the
        // documented wire-boundary allocation (module doc), not gate work.
        let _window = MeasuringWindow::enter();
        ALLOC_COUNT.store(0, Ordering::Relaxed);
        let spread = gate.calibration_spread(FieldKind::Number);
        let confidence = gate.confidence(FieldKind::Number, 0.4);
        let verdict = gate.gate_located(FieldKind::Number, 0.7, 0.4);
        assert_eq!(
            ALLOC_COUNT.load(Ordering::Relaxed),
            0,
            "the temper gate path allocated (spread/confidence/gate_located)"
        );
        drop(_window);
        assert!(spread.is_finite() && spread > 0.0);
        assert!(confidence.is_finite());
        assert!(matches!(
            verdict,
            GateVerdict::Answer | GateVerdict::Abstain { guessable: _ }
        ));
    }
}
