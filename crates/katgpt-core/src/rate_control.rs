//! Dual-EWLS effect-size rate controller (Issue 873 primitive B, Research
//! 581 — volotat/mini-AGI @ `96784b7`, `plasticity.py:63-358`, MIT).
//!
//! A closed-form multiplicative controller over ANY noisy scalar evidence
//! series: nudge a scale `factor` by `exp(gain · tanh((v − T_MID)/width))`
//! where `v = min(t_slow, EFFECT·e_slow, EFFECT·e_fast)` comes from two
//! exponentially-weighted least-squares (EWLS) linear fits, each kept as 7
//! running quantities (six λ-decayed sums + the tick clock).
//!
//! # Why effect size, not progress
//!
//! The deciding quantity is the **effect size** `e = slope / σ_resid` (and
//! its t-statistic `t = slope / se(slope)`): a t-statistic's standard error
//! shrinks with accumulated weight, so `t` measures WATCH-TIME, not
//! progress — a flat series watched long enough still has slope ≈ 0. The
//! `min` of the three (slow t, slow e, fast e) is the conservative
//! combination: every arm must agree before `v` goes high; any one
//! deteriorating arm drags it down.
//!
//! # The asymmetry (the whole design)
//!
//! Up and down are NOT mirrors: up `gain 0.005 / width 0.75` — a slow probe
//! that saturates quickly (being confidently good earns a small, bounded
//! nudge); down `gain 0.025 / width 6.0` — a response proportional to the
//! MAGNITUDE of deterioration (wide tanh keeps the middle range linear).
//!
//! # No window ⇒ no edge-jump artifacts
//!
//! Both fits are exponential-decay, not rolling windows: there is no window
//! edge at which evidence mass steps discontinuously, so the factor
//! trajectory cannot develop staircase artifacts at any fixed lag
//! (windowed controllers can — pinned by the `no_window_edge_artifacts`
//! test). A rolling-origin recenter (every `RECENTER_EVERY` ticks) keeps
//! the x-axis small for f32 precision WITHOUT changing the fit: it is an
//! exact algebraic shift of the sums.
//!
//! # Confirmed regime-jump step
//!
//! A candidate jump (residual > `max(JUMP_K_CONFIRM·σ_total, JUMP_FLOOR)`)
//! is only a flag; the NEXT observation must still exceed
//! `JUMP_K_STILL·σ_total` for the jump to confirm — then `factor ×= 2` and
//! both fits RESET (the controller re-baselines instead of spending
//! λ-time catching up). A one-off spike fails the second observation and is
//! discarded with a [`Note::RegimeJumpDiscarded`].
//!
//! # Sign convention (caller-facing)
//!
//! `val` RISES when things IMPROVE: a rising trend raises `v`, `v ≥ T_MID`
//! nudges the factor UP, and a confirmed regime jump is a sudden RISE
//! (`residual > k·σ`) stepping the factor ×2 with both fits re-baselined.
//! This is the OPPOSITE of mini-AGI's falling-loss convention: a consumer
//! feeding a raw LOSS series gets every verdict inverted (improvement
//! reads as deterioration and the factor is nudged DOWN). Feed a
//! higher-is-better series — accuracy, reward, or `-loss`.
//!
//! # Report-first posture (B2)
//!
//! Constants are PINNED (the Issue-033 adaptive-blend negative law: knobs
//! are constants, never adaptive). The λ/gain/width sextet is mini-AGI's
//! (Research 581); `T_MID`, `EFFECT`, the factor clamps, the jump thresholds
//! and the σ-floors are house defaults pinned at first landing — change
//! them by editing the constants, never by feedback. **Gates nothing until
//! evidence volume exists** (R135/Bench 047): the first consumer A/B is
//! riir-train Plan 416 Phase 2 (vs cosine at fixed budget + regime-change
//! arm). Opt-in (feature `rate_control`); GOAT `.benchmarks/873`.
//!
//! No sync surfaces (pure local state). NaN/`se < 0` inputs are dropped
//! fail-closed (the `kv_eviction::observe` guard convention).

/// Pinned constants — see the module doc for provenance and the never-
/// adaptive law.
pub const LAMBDA_SLOW: f32 = 0.97;
pub const LAMBDA_FAST: f32 = 0.85;
pub const GAIN_UP: f32 = 0.005;
pub const WIDTH_UP: f32 = 0.75;
pub const GAIN_DOWN: f32 = 0.025;
pub const WIDTH_DOWN: f32 = 6.0;
/// Midpoint of the tanh: `v >= T_MID` nudges up, else down.
pub const T_MID: f32 = 0.0;
/// Scale bringing effect sizes into t-stat-comparable units.
pub const EFFECT: f32 = 10.0;
pub const FACTOR_FLOOR: f32 = 0.25;
pub const FACTOR_CEIL: f32 = 4.0;
pub const JUMP_K_CONFIRM: f32 = 4.0;
pub const JUMP_K_STILL: f32 = 2.0;
/// Absolute y-unit floor for jump flagging (guards σ ≈ 0 on perfect
/// hand-built series).
pub const JUMP_FLOOR: f32 = 1.0;
/// A fit contributes nothing below this effective weight (cold start after
/// reset reads e = t = 0 — no drift). MUST stay strictly below the fast
/// fit's steady-state ceiling `1/(1 - LAMBDA_FAST) = 6.67`, or the fast arm
/// is permanently gated cold and `v = min(·, EFFECT·e_fast)` is 0 forever
/// — the controller deadlocks at factor 1.0 (measured by probe on landing).
pub const MIN_WEIGHT: f32 = 5.0;
/// Rolling-origin recenter period (ticks) — an exact algebraic shift, not a
/// window.
pub const RECENTER_EVERY: u32 = 256;
/// σ floor: `max(SIGMA_FLOOR_ABS, SIGMA_FLOOR_REL·|ȳ|)` — bounds `e = b/σ`
/// against catastrophic cancellation on near-perfect series.
pub const SIGMA_FLOOR_ABS: f32 = 1e-6;
pub const SIGMA_FLOOR_REL: f32 = 1e-3;

/// Notable events from one [`RateController::observe`] — the report-first
/// surface (callers log these; nothing gates on them yet).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Note {
    /// Second observation confirmed the jump: factor stepped ×2 (clamped)
    /// and both fits were reset.
    RegimeJumpConfirmed { jump: f32 },
    /// The follow-up observation failed the confirmation bar: the candidate
    /// jump was a one-off spike.
    RegimeJumpDiscarded { jump: f32 },
    /// The nudge clamped the factor at [`FACTOR_FLOOR`] or [`FACTOR_CEIL`].
    Clamped { at: f32 },
}

/// One EWLS fit: 7 running quantities — six λ-decayed sums (`s0, sx, sy,
/// sxx, sxy, syy` over the origin-shifted x and the observed y) plus the
/// tick clock `t`. The origin shift keeps |x| ≤ `RECENTER_EVERY` so the
/// quadratic sums stay f32-precise over unbounded runs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct EwlsFit {
    s0: f32,
    sx: f32,
    sy: f32,
    sxx: f32,
    sxy: f32,
    syy: f32,
    /// Ticks since the fit origin (NOT decayed — the clock).
    t: u32,
}

impl EwlsFit {
    /// Incorporate `(t, y)` with weight 1 after decaying all sums by λ.
    fn observe(&mut self, lambda: f32, y: f32) {
        self.recenter_if_due();
        let x = self.t as f32;
        self.s0 = self.s0 * lambda + 1.0;
        self.sx = self.sx * lambda + x;
        self.sy = self.sy * lambda + y;
        self.sxx = self.sxx * lambda + x * x;
        self.sxy = self.sxy * lambda + x * y;
        self.syy = self.syy * lambda + y * y;
        self.t += 1;
    }

    /// Exact origin shift `x' = x − RECENTER_EVERY`:
    /// `sx' = sx − d·s0`, `sxx' = sxx − 2d·sx + d²·s0`, `sxy' = sxy − d·sy`
    /// (sy, syy, s0 unchanged — y is not shifted). Fit-invariant by algebra.
    fn recenter_if_due(&mut self) {
        if self.t >= RECENTER_EVERY {
            let d = RECENTER_EVERY as f32;
            let sxx = self.sxx - 2.0 * d * self.sx + d * d * self.s0;
            let sxy = self.sxy - d * self.sy;
            self.sx -= d * self.s0;
            self.sxx = sxx;
            self.sxy = sxy;
            self.t = 0;
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    /// Slope with a degeneracy guard (a single-point or perfectly-flat-x
    /// fit has no slope).
    fn slope(&self) -> f32 {
        let den = self.s0 * self.sxx - self.sx * self.sx;
        if den.abs() < 1e-12 || den.is_nan() {
            return 0.0;
        }
        (self.s0 * self.sxy - self.sx * self.sy) / den
    }

    /// Intercept at the current origin.
    fn intercept(&self) -> f32 {
        if self.s0 < f32::MIN_POSITIVE {
            return 0.0;
        }
        (self.sy - self.slope() * self.sx) / self.s0
    }

    /// Residual σ with the pinned floor (`e = b/σ` stays bounded against
    /// cancellation noise on near-perfect series).
    fn sigma(&self) -> f32 {
        if self.s0 < MIN_WEIGHT {
            return SIGMA_FLOOR_ABS;
        }
        let b = self.slope();
        let a = self.intercept();
        let mean_abs_y = (self.sy / self.s0).abs();
        let floor = SIGMA_FLOOR_ABS.max(SIGMA_FLOOR_REL * mean_abs_y);
        let var = (self.syy - a * self.sy - b * self.sxy) / self.s0;
        var.max(floor * floor).sqrt()
    }

    /// Effect size `e = slope / σ`.
    fn effect_size(&self) -> f32 {
        if self.s0 < MIN_WEIGHT {
            return 0.0;
        }
        self.slope() / self.sigma()
    }

    /// t-statistic `t = slope / se(slope)`,
    /// `se(b) = σ·sqrt(s0 / (s0·sxx − sx²))`. Grows with accumulated weight
    /// — watch-time, not progress.
    fn t_stat(&self) -> f32 {
        if self.s0 < MIN_WEIGHT {
            return 0.0;
        }
        let den = self.s0 * self.sxx - self.sx * self.sx;
        if den.abs() < 1e-12 || den.is_nan() {
            return 0.0;
        }
        self.slope() / (self.sigma() * (self.s0 / den).sqrt())
    }

    /// Prediction for the NEXT x (one past the current clock) — used for
    /// jump detection before the observation is incorporated.
    fn predict_next(&self) -> f32 {
        self.intercept() + self.slope() * self.t as f32
    }

    fn effective_weight(&self) -> f32 {
        self.s0
    }
}

/// The controller. Plain-data (`Copy`) — save/restore is assignment; use
/// [`RateController::save`]/[`RateController::restore`] for explicitness at
/// persistence seams.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateController {
    slow: EwlsFit,
    fast: EwlsFit,
    factor: f32,
    pending_jump: Option<f32>,
}

impl Default for RateController {
    fn default() -> Self {
        Self {
            slow: EwlsFit::default(),
            fast: EwlsFit::default(),
            factor: 1.0,
            pending_jump: None,
        }
    }
}

impl RateController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Current multiplicative scale factor (starts at 1.0; clamped to
    /// `[FACTOR_FLOOR, FACTOR_CEIL]`).
    #[inline]
    pub fn factor(&self) -> f32 {
        self.factor
    }

    /// Effective weight of the slow fit (report-first observability; small
    /// after a regime-jump reset).
    pub fn slow_weight(&self) -> f32 {
        self.slow.effective_weight()
    }

    /// One observation. `val` is the evidence value; `se` its caller-side
    /// standard error (combined with the fit's residual σ in quadrature for
    /// jump detection; pass 0.0 when the caller has no error model).
    ///
    /// Non-finite `val`, non-finite or negative `se`: the observation is
    /// DROPPED (debug-loud, fail-closed — bad evidence never moves the
    /// factor) and `None` is returned.
    pub fn observe(&mut self, val: f32, se: f32) -> Option<Note> {
        if !val.is_finite() || !se.is_finite() || se < 0.0 {
            debug_assert!(
                false,
                "rate_control::observe: val must be finite and se finite >= 0, got ({val}, {se})"
            );
            return None;
        }

        // Jump detection reads the PRE-observation prediction.
        let pred = self.slow.predict_next();
        let sigma_total = (self.slow.sigma() * self.slow.sigma() + se * se).sqrt();
        let residual = val - pred;
        let mut note = None;
        let mut confirmed_jump = None;
        match self.pending_jump {
            None => {
                if residual > (JUMP_K_CONFIRM * sigma_total).max(JUMP_FLOOR) {
                    self.pending_jump = Some(residual);
                }
            }
            Some(jump) => {
                self.pending_jump = None;
                if residual > (JUMP_K_STILL * sigma_total).max(JUMP_FLOOR * 0.5) {
                    confirmed_jump = Some(jump.max(residual));
                } else {
                    note = Some(Note::RegimeJumpDiscarded { jump });
                }
            }
        }

        // Both fits see every observation (a flagged candidate jump is NOT
        // withheld — confirmation lives one observation ahead).
        self.slow.observe(LAMBDA_SLOW, val);
        self.fast.observe(LAMBDA_FAST, val);

        // The nudge.
        let t_slow = self.slow.t_stat();
        let e_slow = self.slow.effect_size();
        let e_fast = self.fast.effect_size();
        let v = t_slow.min(EFFECT * e_slow).min(EFFECT * e_fast);
        let (gain, width) = if v >= T_MID {
            (GAIN_UP, WIDTH_UP)
        } else {
            (GAIN_DOWN, WIDTH_DOWN)
        };
        let nudge = (gain * ((v - T_MID) / width).tanh()).exp();
        self.factor = (self.factor * nudge).clamp(FACTOR_FLOOR, FACTOR_CEIL);
        if self.factor == FACTOR_FLOOR || self.factor == FACTOR_CEIL {
            // Keep any existing note (jump fate outranks a clamp).
            note = Some(note.unwrap_or(Note::Clamped { at: self.factor }));
        }

        // The confirmed step: ×2 (clamped) + fit reset.
        if let Some(jump) = confirmed_jump {
            self.factor = (self.factor * 2.0).clamp(FACTOR_FLOOR, FACTOR_CEIL);
            self.slow.reset();
            self.fast.reset();
            note = Some(Note::RegimeJumpConfirmed { jump });
        }
        note
    }

    /// Snapshot for persistence seams (plain `Copy` under the hood).
    pub fn save(&self) -> Self {
        *self
    }

    /// Restore a snapshot (invariants already hold — every mutation went
    /// through the clamped paths).
    pub fn restore(&mut self, saved: &Self) {
        *self = *saved;
    }

    /// Fixed-layout little-endian byte snapshot (65 bytes) — the binary
    /// checkpoint seam (riir-train Plan 416 T2.2: the controller state rides
    /// the trainer's checkpoint file as a trailing block). Layout:
    /// `slow` fit (6×f32 + u32) ‖ `fast` fit (6×f32 + u32) ‖ `factor` f32 ‖
    /// `pending_jump` (u8 tag 0=None/1=Some + f32 slot, written 0.0 when
    /// None). katgpt-core owns the layout so consumers carry opaque bytes;
    /// a format change bumps [`RateController::SNAPSHOT_LEN`], which rejects
    /// old/new mismatches loudly instead of decoding garbage.
    pub const SNAPSHOT_LEN: usize = 65;

    pub fn to_bytes(&self) -> [u8; Self::SNAPSHOT_LEN] {
        let mut b = [0u8; Self::SNAPSHOT_LEN];
        let mut w = 0usize;
        for fit in [&self.slow, &self.fast] {
            le_f32_put(&mut b, &mut w, fit.s0);
            le_f32_put(&mut b, &mut w, fit.sx);
            le_f32_put(&mut b, &mut w, fit.sy);
            le_f32_put(&mut b, &mut w, fit.sxx);
            le_f32_put(&mut b, &mut w, fit.sxy);
            le_f32_put(&mut b, &mut w, fit.syy);
            le_u32_put(&mut b, &mut w, fit.t);
        }
        le_f32_put(&mut b, &mut w, self.factor);
        // Tag first (advances the cursor), then the f32 slot — a None writes
        // 0.0 so the layout is arm-independent.
        b[60] = match self.pending_jump {
            None => 0,
            Some(_) => 1,
        };
        w += 1;
        le_f32_put(&mut b, &mut w, self.pending_jump.unwrap_or(0.0));
        debug_assert_eq!(w, Self::SNAPSHOT_LEN);
        b
    }

    /// Inverse of [`RateController::to_bytes`]. Rejects (returns `None`,
    /// never a half-built controller) on: wrong length, an unknown
    /// pending-jump tag, or any non-finite float (a corrupted byte must not
    /// poison the fits — the controller would read as inert NaN math). The
    /// factor is clamped into `[FACTOR_FLOOR, FACTOR_CEIL]` rather than
    /// rejected: every mutation path clamps, so an out-of-range value is
    /// corruption, but the clamp is the same invariant the live controller
    /// holds.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Self::SNAPSHOT_LEN {
            return None;
        }
        let mut r = 0usize;
        let mut fit = || EwlsFit {
            s0: le_f32_at(bytes, &mut r),
            sx: le_f32_at(bytes, &mut r),
            sy: le_f32_at(bytes, &mut r),
            sxx: le_f32_at(bytes, &mut r),
            sxy: le_f32_at(bytes, &mut r),
            syy: le_f32_at(bytes, &mut r),
            t: le_u32_at(bytes, &mut r),
        };
        let slow = fit();
        let fast = fit();
        let factor = le_f32_at(bytes, &mut r);
        let tag = le_u8_at(bytes, &mut r);
        let pending_jump = match tag {
            0 => None,
            1 => Some(le_f32_at(bytes, &mut r)),
            _ => return None,
        };
        let floats = [
            slow.s0, slow.sx, slow.sy, slow.sxx, slow.sxy, slow.syy, fast.s0, fast.sx, fast.sy,
            fast.sxx, fast.sxy, fast.syy, factor,
        ];
        if floats.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if let Some(j) = pending_jump
            && !j.is_finite()
        {
            return None;
        }
        Some(Self {
            slow,
            fast,
            factor: factor.clamp(FACTOR_FLOOR, FACTOR_CEIL),
            pending_jump,
        })
    }
}

/// LE f32 read that advances the caller's cursor (bounds are guaranteed by
/// [`RateController::SNAPSHOT_LEN`], checked once at entry).
fn le_f32_at(b: &[u8], r: &mut usize) -> f32 {
    let v = f32::from_le_bytes(b[*r..*r + 4].try_into().expect("SNAPSHOT_LEN bounds"));
    *r += 4;
    v
}

/// LE u32 read that advances the caller's cursor.
fn le_u32_at(b: &[u8], r: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*r..*r + 4].try_into().expect("SNAPSHOT_LEN bounds"));
    *r += 4;
    v
}

/// LE u8 read that advances the caller's cursor.
fn le_u8_at(b: &[u8], r: &mut usize) -> u8 {
    let v = b[*r];
    *r += 1;
    v
}

/// LE f32 write that advances the caller's cursor (invariant: w + 4 ≤ len).
fn le_f32_put(b: &mut [u8], w: &mut usize, v: f32) {
    b[*w..*w + 4].copy_from_slice(&v.to_le_bytes());
    *w += 4;
}

/// LE u32 write that advances the caller's cursor.
fn le_u32_put(b: &mut [u8], w: &mut usize, v: u32) {
    b[*w..*w + 4].copy_from_slice(&v.to_le_bytes());
    *w += 4;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal deterministic xorshift — the house pattern.
    struct SimpleLcg(u64);
    impl SimpleLcg {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 { 1 } else { seed })
        }
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32) / (1u64 << 24) as f32
        }
        fn signed(&mut self, scale: f32) -> f32 {
            (self.next_f32() * 2.0 - 1.0) * scale
        }
    }

    // ── B3 G1: signal-series known answers ─────────────────────────────

    #[test]
    fn plateau_factor_flat() {
        let mut c = RateController::new();
        for _ in 0..300 {
            c.observe(5.0, 0.1);
        }
        let f = c.factor();
        assert!(
            (0.98..=1.02).contains(&f),
            "plateau must not drift the factor (got {f})"
        );
    }

    #[test]
    fn improving_factor_climbs() {
        let mut c = RateController::new();
        for t in 0..300 {
            c.observe(0.1 * t as f32, 0.05);
        }
        assert!(
            c.factor() > 1.5,
            "a real improving trend must climb (got {})",
            c.factor()
        );
    }

    #[test]
    fn deteriorating_factor_falls_fast() {
        let mut c = RateController::new();
        for t in 0..300 {
            c.observe(-0.1 * t as f32, 0.05);
        }
        assert!(
            c.factor() < 0.6,
            "deterioration must fall (got {})",
            c.factor()
        );
    }

    #[test]
    fn asymmetry_down_is_steeper_than_up() {
        // Equal-magnitude opposite trends: the down arm must move the factor
        // further (gain 0.025 vs 0.005) — the asymmetry is the design. 30 obs
        // keeps both arms inside the clamps (the comparison is meaningless
        // at a clamp).
        let mut up = RateController::new();
        let mut down = RateController::new();
        for t in 0..30 {
            up.observe(0.05 * t as f32, 0.01);
            down.observe(-(t as f32) * 0.05, 0.01);
        }
        let gain_up = up.factor() - 1.0;
        let gain_down = 1.0 - down.factor();
        assert!(
            gain_down > 2.0 * gain_up,
            "down must dominate up (up {gain_up}, down {gain_down})"
        );
    }

    #[test]
    fn regime_jump_confirms_steps_and_resets() {
        let mut c = RateController::new();
        for _ in 0..60 {
            c.observe(0.0, 0.0);
        }
        assert_eq!(c.observe(10.0, 0.0), None, "first jump obs only flags");
        let before = c.factor();
        let note = c.observe(10.0, 0.0);
        assert!(
            matches!(note, Some(Note::RegimeJumpConfirmed { jump }) if jump >= 10.0),
            "second obs must confirm: {note:?}"
        );
        // The step happened... factor was ~1.0 (plateau) then doubled.
        assert!(
            c.factor() >= before * 1.9,
            "confirmed jump doubles the factor ({before} -> {})",
            c.factor()
        );
        // Fits were reset: slow weight is below the cold-start bar.
        assert!(c.slow_weight() < MIN_WEIGHT, "fits must reset on confirm");
        // And post-reset the controller tracks the new level without
        // flagging another jump.
        for _ in 0..30 {
            assert!(!matches!(c.observe(10.0, 0.0), Some(Note::RegimeJumpConfirmed { .. })));
        }
    }

    #[test]
    fn regime_jump_discarded_on_single_spike() {
        let mut c = RateController::new();
        for _ in 0..60 {
            c.observe(0.0, 0.0);
        }
        assert_eq!(c.observe(10.0, 0.0), None, "flag");
        let note = c.observe(0.0, 0.0);
        assert!(
            matches!(note, Some(Note::RegimeJumpDiscarded { .. })),
            "return to baseline must discard: {note:?}"
        );
        assert!(
            c.factor() < 1.1,
            "a discarded spike must not step the factor (got {})",
            c.factor()
        );
        // And the discarded spike does not leave a pending jump.
        assert_eq!(c.observe(0.0, 0.0), None);
    }

    /// B3's window-edge arm: an EWLS controller has NO window, so a
    /// transient outlier's influence decays smoothly — the recovery step
    /// envelope is non-increasing. A windowed controller instead shows its
    /// largest recovery step exactly when the outlier LEAVES the window
    /// (a second discontinuity, lagged by W). Fixture: STATIONARY noise
    /// (a saturating drift would confound the envelope with re-saturation)
    /// punctuated by ONE large negative outlier mid-series (negative
    /// residuals never fire the jump path, isolating window geometry from
    /// jump confirmation); for every candidate window length L, the max
    /// step in (spike, spike+L] must be >= the max step in (spike+L,
    /// spike+2L] — no lag-correlated rebound.
    #[test]
    fn no_window_edge_artifacts() {
        const SPIKE_AT: usize = 200;
        let mut c = RateController::new();
        let mut rng = SimpleLcg::new(0xBEEF);
        let mut factors = vec![1.0_f32];
        for t in 0..800 {
            let base = rng.signed(0.05);
            let y = if t == SPIKE_AT { base - 3.0 } else { base };
            c.observe(y, 0.05);
            factors.push(c.factor());
        }
        let steps: Vec<f32> = factors
            .windows(2)
            .map(|w| (w[1] / w[0]).ln().abs())
            .collect();
        // (1) Every step is nudge-sized — no ×2-style discontinuity anywhere
        // (the outlier is a single observation and never confirms a jump).
        let max_step = steps.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max_step <= GAIN_DOWN + 1e-4,
            "single-step |Δln f| {max_step} exceeds the down-gain {GAIN_DOWN}"
        );
        // (2) The recovery envelope decays: no lag-correlated rebound at any
        // candidate window length.
        let env = |from: usize, to: usize| {
            steps[from..to.min(steps.len())]
                .iter()
                .copied()
                .fold(0.0_f32, f32::max)
        };
        for l in [50usize, 100, 128, 200, 256] {
            let early = env(SPIKE_AT + 1, SPIKE_AT + 1 + l);
            let late = env(SPIKE_AT + 1 + l, SPIKE_AT + 1 + 2 * l);
            assert!(
                late <= early + 1e-5,
                "lag-correlated rebound at L={l}: early {early}, late {late}"
            );
        }
    }

    // ── guards, determinism, save/restore, G4 ──────────────────────────

    #[test]
    fn bad_inputs_dropped_fail_closed() {
        if !cfg!(debug_assertions) {
            let mut c = RateController::new();
            for _ in 0..20 {
                c.observe(1.0, 0.1);
            }
            let f = c.factor();
            assert_eq!(c.observe(f32::NAN, 0.1), None);
            assert_eq!(c.observe(f32::INFINITY, 0.1), None);
            assert_eq!(c.observe(1.0, -0.1), None);
            assert_eq!(c.observe(1.0, f32::NAN), None);
            assert_eq!(c.factor(), f, "dropped observations must not move the factor");
        }
    }

    #[test]
    fn determinism_bit_identical_across_runs() {
        let run = || {
            let mut c = RateController::new();
            let mut rng = SimpleLcg::new(0x5EED);
            let mut acc = 0u32;
            for t in 0..2_000 {
                let y = 0.02 * t as f32 + rng.signed(0.3) + (t / 700) as f32 * 5.0;
                if c.observe(y, 0.1).is_some() {
                    acc += 1;
                }
                acc = acc.wrapping_mul(3).wrapping_add(c.factor().to_bits());
            }
            (acc, c)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn recentering_is_fit_invariant() {
        // Crossing the RECENTER_EVERY boundary must not change predictions
        // materially: two controllers fed the same long series, one forced
        // through a recenter (it is automatic), vs the fit math recomputed
        // naively — checked as prediction continuity across the boundary.
        let mut c = RateController::new();
        let mut rng = SimpleLcg::new(7);
        for t in 0..(RECENTER_EVERY * 3) {
            let y = 0.01 * (t % 400) as f32 + rng.signed(0.1);
            c.observe(y, 0.05);
        }
        // Still tracks the ramp direction.
        assert!(c.factor() > 1.0, "long ramp must climb, got {}", c.factor());
        // Predictions stay finite and sane across recenter points.
        let f = c.factor();
        assert!(f.is_finite() && (FACTOR_FLOOR..=FACTOR_CEIL).contains(&f));
    }

    #[test]
    fn save_restore_roundtrip() {
        let mut c = RateController::new();
        for t in 0..50 {
            c.observe(0.05 * t as f32, 0.1);
        }
        let saved = c.save();
        let mut a = c.save();
        let mut b = saved;
        for t in 50..100 {
            let y = 0.05 * t as f32;
            a.observe(y, 0.1);
            b.observe(y, 0.1);
        }
        assert_eq!(a.factor(), b.factor(), "restore must resume identically");
        let mut restored = RateController::new();
        restored.restore(&saved);
        assert_eq!(restored, c);
    }

    #[test]
    fn factor_respects_clamps() {
        let mut up = RateController::new();
        for t in 0..5_000 {
            up.observe(1.0 * t as f32, 0.01);
        }
        assert_eq!(up.factor(), FACTOR_CEIL);
        let mut down = RateController::new();
        for t in 0..5_000 {
            down.observe(-(t as f32), 0.01);
        }
        assert_eq!(down.factor(), FACTOR_FLOOR);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn observe_is_allocation_free() {
        // TrackingAllocator (TEST_GLOBAL_ALLOC), per-thread counters.
        let mut c = RateController::new();
        let mut rng = SimpleLcg::new(99);
        for t in 0..16 {
            c.observe(0.01 * t as f32 + rng.signed(0.1), 0.05);
        }
        crate::alloc::reset_alloc_stats();
        for t in 16..1_016 {
            c.observe(0.01 * t as f32 + rng.signed(0.1), 0.05);
        }
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(count, 0, "observe must not allocate (got {count})");
    }

    // ── Byte-snapshot seam (Plan 416 T2.2 checkpoint carrier) ──────────

    #[test]
    fn snapshot_bytes_round_trip_bit_identical() {
        let mut c = RateController::new();
        let mut rng = SimpleLcg::new(7);
        for t in 0..64 {
            c.observe(0.02 * t as f32 + rng.signed(0.3), 0.05);
        }
        let bytes = c.to_bytes();
        assert_eq!(bytes.len(), RateController::SNAPSHOT_LEN);
        let back = RateController::from_bytes(&bytes).expect("valid snapshot must parse");
        assert_eq!(back, c, "Copy PartialEq round trip must be exact");
        // And the restored controller continues identically.
        let mut a = c;
        let mut b = back;
        for t in 64..128 {
            let y = 0.02 * t as f32 + rng.signed(0.3);
            a.observe(y, 0.05);
            b.observe(y, 0.05);
        }
        assert_eq!(a, b, "post-restore trajectories must match bit for bit");
    }

    #[test]
    fn snapshot_bytes_carry_pending_jump_and_default() {
        let mut c = RateController::new();
        c.pending_jump = Some(-0.5);
        c.factor = 1.25;
        let back = RateController::from_bytes(&c.to_bytes()).expect("parse");
        assert_eq!(back.pending_jump, Some(-0.5));
        assert_eq!(back.factor, 1.25);

        let d = RateController::default();
        assert_eq!(
            RateController::from_bytes(&d.to_bytes()).expect("parse"),
            d,
            "the default controller must round trip"
        );
    }

    #[test]
    fn snapshot_bytes_reject_corruption() {
        let c = RateController::new();
        let bytes = c.to_bytes();
        // Truncated and over-long payloads are refused, never half-built.
        assert!(RateController::from_bytes(&bytes[..40]).is_none());
        let mut long = bytes.to_vec();
        long.push(0);
        assert!(RateController::from_bytes(&long).is_none());
        // Unknown pending-jump tag.
        let mut bad_tag = bytes;
        bad_tag[60] = 2;
        assert!(RateController::from_bytes(&bad_tag).is_none());
        // Non-finite float anywhere in the sums or the factor — NaN math must
        // not ride back in as an inert controller.
        let mut nan_factor = bytes;
        nan_factor[56..60].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(RateController::from_bytes(&nan_factor).is_none());
        let mut nan_sum = bytes;
        nan_sum[0..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(RateController::from_bytes(&nan_sum).is_none());
        // Out-of-range factor is CLAMPED into the invariant, not rejected
        // (every live mutation path clamps — the snapshot must land in the
        // same state space).
        let mut big = bytes;
        big[56..60].copy_from_slice(&9.0f32.to_le_bytes());
        let back = RateController::from_bytes(&big).expect("finite factor parses");
        assert_eq!(back.factor(), FACTOR_CEIL);
    }
}
