//! SwiRController — entropy-trend-driven Explicit↔Latent mode switch (paper §3).
//!
//! The controller is a 2-mode state machine:
//!
//! - **Latent**: emit a soft embedding (continuous reasoning). Stay here while
//!   entropy is falling (the model is converging on a region of belief).
//! - **Explicit**: emit a concrete token (discrete reasoning). Stay here while
//!   entropy is rising (the model is exploring) — but only after a minimum dwell
//!   window `W_E→L` has accumulated, to avoid mode chatter.
//!
//! Switches are driven by the *sign* of `entropy − reference_entropy`, not the
//! absolute entropy level. `reference_entropy` is reset on every switch, so the
//! decision is "is this step more or less confident than the *last switch
//! point*?" — a relative, drift-robust signal. This is the key insight from
//! paper §3.3.
//!
//! Two extra guards keep the run bounded:
//!
//! - **Convergence** (paper §3.4): once `switch_count` reaches `½·c_max`,
//!   enqueue `</think>` on the next Explicit step — the model is "wrapping up".
//! - **Termination** (paper §3.4): once `switch_count` exceeds `c_max`, enqueue
//!   `ForceAnswerPrefix` and start a token budget. When the budget hits zero,
//!   the controller emits `Terminate`.
//!
//! The controller is allocation-free after `new()`: the injection queue is a
//! fixed-size ring buffer (paper never injects more than 2 tokens in a row), and
//! `step()` is a pure state-machine update.

use crate::swir::signal_mix::SignalMixKind;
use crate::swir::types::{ControlToken, StepAction, SwiRConfig, SwiRStats, ThinkMode};

/// Maximum tokens the controller can enqueue before the host drains them.
/// Paper injects at most one `</think>` or one `ForceAnswerPrefix` at a time,
/// so 4 is generous headroom — kept as a fixed-size ring for zero-alloc.
const INJECT_QUEUE_CAP: usize = 4;

/// The 2-mode controller. Owns its config and stats; the host drives it via
/// [`step`](Self::step) and reads directives off the returned [`StepAction`].
#[derive(Debug)]
pub struct SwiRController {
    config: SwiRConfig,

    mode: ThinkMode,
    /// Entropy recorded at the last switch instant (or first step). NaN until
    /// the first call to `step`.
    reference_entropy: f32,
    /// Number of consecutive steps spent in the *current* mode. Reset to 0 on
    /// every switch. Used by the asymmetric dwell windows.
    dwell_steps: u32,
    /// Total Latent→Explicit switches observed. Drives convergence / termination.
    /// (Explicit→Latent switches are *not* counted — paper §3.4 counts only
    /// "completed latent exploration rounds".)
    switch_count: u32,

    /// Token budget remaining after `ForceAnswerPrefix` was injected. `None`
    /// means we have not yet hit the termination guard.
    answer_budget_remaining: Option<u32>,

    /// Bounded ring of pending control-token injections.
    inject_queue: [Option<ControlToken>; INJECT_QUEUE_CAP],
    inject_head: usize,
    inject_tail: usize,

    /// Step at which the most recent switch occurred. Used by
    /// [`should_mix_signal`](Self::should_mix_signal) to compute the α_t / β_t
    /// blend only on the step *immediately after* a switch.
    last_switch_step: u32,
    /// Set to `true` for exactly one step after a switch — drives
    /// `should_mix_signal`.
    mix_pending: Option<SignalMixKind>,

    /// Aggregate stats for benchmarks / debug.
    latent_steps: u32,
    explicit_steps: u32,

    /// Most recent per-step kurtosis scalar supplied by the host via
    /// [`observe_kurtosis`](Self::observe_kurtosis). NaN until the first
    /// observation. Drives the G6 auto-fallback (Plan 275 T3.8): when this
    /// exceeds `config.kurtosis_escape_threshold`, the controller refuses to
    /// enter Latent mode for that step.
    last_kurtosis: f32,
}

impl SwiRController {
    /// Create a new controller. Initial mode is [`ThinkMode::Latent`] (paper
    /// starts in latent exploration), `reference_entropy` is NaN until the
    /// first step.
    #[inline]
    pub fn new(config: SwiRConfig) -> Self {
        Self {
            config,
            mode: ThinkMode::Latent,
            reference_entropy: f32::NAN,
            dwell_steps: 0,
            switch_count: 0,
            answer_budget_remaining: None,
            inject_queue: [None; INJECT_QUEUE_CAP],
            inject_head: 0,
            inject_tail: 0,
            last_switch_step: 0,
            mix_pending: None,
            latent_steps: 0,
            explicit_steps: 0,
            last_kurtosis: f32::NAN,
        }
    }

    /// Current mode (for host inspection / dashboards). Not used by `step`.
    #[inline]
    pub fn mode(&self) -> ThinkMode {
        self.mode
    }

    /// Current switch count — drives the convergence / termination guards.
    #[inline]
    pub fn switch_count(&self) -> u32 {
        self.switch_count
    }

    /// Current reference entropy (locked at the first step, updated on each
    /// mode switch). Read-only diagnostic getter for hosts that want to plot
    /// the entropy trend vs. the switching threshold.
    #[inline]
    pub fn reference_entropy(&self) -> f32 {
        self.reference_entropy
    }

    /// Current dwell counter (steps since the last mode switch). Read-only
    /// diagnostic getter — the host needs this to understand whether a
    /// non-switch is due to the dwell window not being satisfied.
    #[inline]
    pub fn dwell_steps(&self) -> u32 {
        self.dwell_steps
    }

    /// Supply a per-step kurtosis scalar for the G6 auto-fallback (Plan 275
    /// T3.8). The host calls this **before** [`step`](Self::step) each decode
    /// position; `step()` consults the stored value when deciding whether to
    /// allow a Latent→Explicit→Latent re-entry. If the host never calls this,
    /// `last_kurtosis` stays NaN and the escape hatch is inert (NaN compares
    /// false to any threshold).
    ///
    /// Rigid-constraint tasks (paper's 3D-surface-shortest-path, formal math
    /// with exact answers) produce heavy-tailed logit distributions → high
    /// excess kurtosis. When `kurtosis > config.kurtosis_escape_threshold`,
    /// the controller refuses to enter Latent mode for that step, bypassing
    /// soft-embedding exploration entirely.
    #[inline]
    pub fn observe_kurtosis(&mut self, kurtosis: f32) {
        self.last_kurtosis = kurtosis;
    }

    /// True iff the G6 escape hatch should fire this step (kurtosis above
    /// threshold). Kept as a helper so the escape logic appears once in `step`.
    #[inline]
    fn kurtosis_escape_fires(&self) -> bool {
        // NaN-safe: NaN > threshold is false, so an un-observed kurtosis never
        // fires the escape hatch.
        self.last_kurtosis > self.config.kurtosis_escape_threshold
    }

    /// Snapshot of aggregate stats.
    pub fn stats(&self) -> SwiRStats {
        SwiRStats {
            switches_total: self.switch_count,
            latent_steps: self.latent_steps,
            explicit_steps: self.explicit_steps,
            mode_at_termination: if self.answer_budget_remaining == Some(0) {
                Some(self.mode)
            } else {
                None
            },
        }
    }

    /// Advance the controller by one step.
    ///
    /// `entropy`: Shannon entropy of the current next-token distribution (the
    /// host computes this; Phase 2's strategy adapter uses
    /// `attn_match_adaptive_cot::entropy_from_logits`).
    /// `step_index`: 0-based decode step (used by the α_t / β_t schedule).
    ///
    /// Returns the [`StepAction`] the host should take this step. The host MUST
    /// drain [`StepAction::InjectControlToken`] results in order — the queue
    /// holds at most `INJECT_QUEUE_CAP` pending injections.
    #[inline]
    pub fn step(&mut self, entropy: f32, step_index: u32) -> StepAction {
        // (1) Drain pending injections first — paper's control-token inserts
        // preempt any emit. If the answer budget is exhausted mid-drain, emit
        // Terminate instead.
        if let Some(token) = self.pop_inject() {
            if let Some(remaining) = self.answer_budget_remaining.as_mut() {
                if *remaining == 0 {
                    return StepAction::Terminate;
                }
                *remaining = remaining.saturating_sub(1);
            }
            return StepAction::InjectControlToken(token);
        }
        // (2) Answer-budget countdown (post-ForceAnswerPrefix). The paper's
        // ForceAnswerPrefix means "stop reasoning, start answering" — we emit
        // tokens without running the mode-switch logic below. Allowing further
        // mode switches during the budget window would inflate switch_count
        // past c_max on alternating entropy schedules, defeating the
        // overthinking guard (paper §3.4). Stats still count as Explicit
        // (we're emitting concrete answer tokens).
        if let Some(remaining) = self.answer_budget_remaining.as_mut() {
            if *remaining == 0 {
                return StepAction::Terminate;
            }
            *remaining = remaining.saturating_sub(1);
            self.explicit_steps = self.explicit_steps.saturating_add(1);
            return StepAction::EmitToken(0);
        }

        // (3) First-step init: lock reference_entropy to the first observation.
        // f32::NAN compares false to everything, so this is the natural sentinel.
        if self.reference_entropy.is_nan() {
            self.reference_entropy = entropy;
            self.dwell_steps = 0;
        }

        // (4) Mode-switch logic (paper §3.3). Decision is based on the sign of
        // (entropy − reference_entropy), i.e. "is this step more or less
        // confident than the last switch point?". Asymmetric dwell windows
        // prevent chatter.
        //
        // G6 auto-fallback (Plan 275 T3.8): if the host-supplied kurtosis is
        // above the escape threshold, refuse to enter or stay in Latent mode.
        // This bypasses soft-embedding exploration on rigid-constraint tasks.
        let kurtosis_escape = self.kurtosis_escape_fires();
        let mut switched_to = None;
        let entropy_below_ref = entropy < self.reference_entropy;
        let entropy_above_ref = entropy > self.reference_entropy;
        match self.mode {
            ThinkMode::Latent if entropy_below_ref => {
                // Latent → Explicit: entropy dropped, model converged → commit
                // a concrete token. Paper default W_L→E = 0 → switch immediately.
                if self.dwell_steps + 1 > self.config.w_l_to_e {
                    self.switch_to(ThinkMode::Explicit, entropy, step_index);
                    switched_to = Some(ThinkMode::Explicit);
                }
            }
            ThinkMode::Latent if kurtosis_escape => {
                // G6 escape: we're in Latent but the host signalled a rigid-
                // constraint task. Force Explicit regardless of entropy trend.
                self.switch_to(ThinkMode::Explicit, entropy, step_index);
                switched_to = Some(ThinkMode::Explicit);
            }
            ThinkMode::Explicit if entropy_above_ref && !kurtosis_escape => {
                // Explicit → Latent: entropy rose, model wants to explore → only
                // allow after W_E→L dwell window to prevent chatter. Suppressed
                // when the G6 escape hatch fires (rigid-constraint task).
                if self.dwell_steps + 1 >= self.config.w_e_to_l {
                    self.switch_to(ThinkMode::Latent, entropy, step_index);
                    switched_to = Some(ThinkMode::Latent);
                }
            }
            _ => {}
        }

        if switched_to.is_none() {
            // Stay in current mode, advance dwell.
            self.dwell_steps = self.dwell_steps.saturating_add(1);
        }

        // (5) Switch-count guards (paper §3.4). Only count Latent→Explicit.
        // Convergence fires at ½·c_max; termination fires above c_max.
        //
        // CRITICAL: only fire these guards on the step where the Latent→Explicit
        // switch JUST happened (switched_to == Some(Explicit)). Firing every step
        // while in Explicit mode causes a livelock: the enqueued CloseThink is
        // drained at the start of the next step (step (1) above), which skips
        // the mode-switch logic (step (3)), which means switch_count never
        // advances, which re-enqueues CloseThink forever. The one-shot trigger
        // on the switch event matches the paper's intent (§3.4 describes these
        // as switch-count thresholds, not continuous conditions).
        if switched_to == Some(ThinkMode::Explicit) {
            let conv_at = self.config.convergence_switch_count();
            if self.switch_count >= conv_at && self.switch_count <= self.config.c_max {
                // Convergence window — enqueue CloseThink on the next step.
                self.try_enqueue(ControlToken::CloseThink);
            } else if self.switch_count > self.config.c_max {
                // Overthinking guard — force answer and start budget countdown.
                self.try_enqueue(ControlToken::ForceAnswerPrefix);
                if self.answer_budget_remaining.is_none() {
                    self.answer_budget_remaining = Some(self.config.answer_budget_b);
                }
            }
        }

        // Bookkeeping for stats.
        match self.mode {
            ThinkMode::Latent => self.latent_steps = self.latent_steps.saturating_add(1),
            ThinkMode::Explicit => self.explicit_steps = self.explicit_steps.saturating_add(1),
        }

        // (6) Emit directive based on current mode. (Any injection enqueued
        // above will be drained on the *next* call to step(), per step (1).)
        match self.mode {
            ThinkMode::Explicit => StepAction::EmitToken(0),
            ThinkMode::Latent => StepAction::EmitSoftEmbedding,
        }
    }

    /// If this is the step immediately after a switch, return the mix kind and
    /// ratio (paper Eq. 4). Returns `None` on non-switch steps.
    ///
    /// Schedule: α_t (Latent entry) = α_0 + (1 − α_0) · step_index / max_steps.
    /// Same shape for β_t (Explicit exit). The ratio *increases* over the run,
    /// so early switches favour the anchor token, late switches favour the soft
    /// embedding.
    #[inline]
    pub fn should_mix_signal(&mut self) -> Option<(SignalMixKind, f32)> {
        let kind = self.mix_pending.take()?;
        let step_at_switch = self.last_switch_step;
        let max_steps = self.config.max_steps.max(1) as f32;
        let t = (step_at_switch as f32 / max_steps).clamp(0.0, 1.0);
        let (base, kind) = match kind {
            SignalMixKind::LatentEntry => (self.config.alpha_0, SignalMixKind::LatentEntry),
            SignalMixKind::ExplicitExit => (self.config.beta_0, SignalMixKind::ExplicitExit),
        };
        let ratio = base + (1.0 - base) * t;
        Some((kind, ratio.clamp(0.0, 1.0)))
    }

    /// Internal: perform a mode switch. Resets reference_entropy + dwell, bumps
    /// switch_count on Latent→Explicit, arms `should_mix_signal` for the next
    /// call.
    fn switch_to(&mut self, new_mode: ThinkMode, entropy: f32, step_index: u32) {
        let prev = self.mode;
        self.mode = new_mode;
        self.reference_entropy = entropy;
        self.dwell_steps = 0;
        self.last_switch_step = step_index;
        // Only Latent→Explicit counts toward convergence / termination guards.
        if prev == ThinkMode::Latent && new_mode == ThinkMode::Explicit {
            self.switch_count = self.switch_count.saturating_add(1);
        }
        // Arm mix_pending so the *next* should_mix_signal() call sees it.
        self.mix_pending = Some(match new_mode {
            ThinkMode::Latent => SignalMixKind::LatentEntry,
            ThinkMode::Explicit => SignalMixKind::ExplicitExit,
        });
    }

    /// Push a control token onto the inject ring. Silently drops if full — the
    /// paper never injects more than one at a time, so a full queue indicates a
    /// host bug (not draining).
    #[inline]
    fn try_enqueue(&mut self, token: ControlToken) {
        let next_tail = (self.inject_tail + 1) % INJECT_QUEUE_CAP;
        if next_tail == self.inject_head {
            // Queue full — host didn't drain. Drop silently rather than panic;
            // the controller's correctness doesn't depend on every injection
            // landing (worst case the run continues in current mode).
            return;
        }
        self.inject_queue[self.inject_tail] = Some(token);
        self.inject_tail = next_tail;
    }

    /// Pop the next pending injection, if any.
    #[inline]
    fn pop_inject(&mut self) -> Option<ControlToken> {
        if self.inject_head == self.inject_tail {
            return None;
        }
        let token = self.inject_queue[self.inject_head].take();
        self.inject_head = (self.inject_head + 1) % INJECT_QUEUE_CAP;
        token
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swir::types::SwiRConfig;

    fn cfg_small() -> SwiRConfig {
        // Tight windows so tests don't have to feed hundreds of steps.
        SwiRConfig {
            w_e_to_l: 2,
            w_l_to_e: 0,
            c_max: 4,
            c_convergence_fraction: 0.5,
            answer_budget_b: 3,
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps: 100,
            kurtosis_escape_threshold: f32::INFINITY,
        }
    }

    #[test]
    fn test_first_step_initializes_reference_entropy() {
        let mut c = SwiRController::new(cfg_small());
        assert!(c.reference_entropy.is_nan(), "must start as NaN");
        let a = c.step(1.5, 0);
        // First step: no switch (NaN init), Latent mode → soft embed.
        assert_eq!(a, StepAction::EmitSoftEmbedding);
        assert!((c.reference_entropy - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_latent_to_explicit_on_confidence_rise() {
        // reference=5.0, next step entropy=2.0 (lower) → Latent→Explicit.
        let mut c = SwiRController::new(cfg_small());
        c.step(5.0, 0); // lock reference at 5.0, Latent
        let a = c.step(2.0, 1); // entropy below ref → switch to Explicit
        assert_eq!(c.mode(), ThinkMode::Explicit);
        assert_eq!(c.switch_count(), 1);
        assert_eq!(a, StepAction::EmitToken(0));
    }

    #[test]
    fn test_explicit_to_latent_requires_dwell_window() {
        // w_e_to_l = 2. Switch to Explicit, then immediately raise entropy —
        // should NOT switch back yet (dwell < window).
        let mut c = SwiRController::new(cfg_small());
        c.step(5.0, 0); // Latent, ref=5
        c.step(2.0, 1); // Latent→Explicit (switch_count=1, dwell=0)
        assert_eq!(c.mode(), ThinkMode::Explicit);
        // Try to switch back: entropy above ref=2.0 → but dwell=1 (<2).
        let _ = c.step(3.0, 2);
        assert_eq!(c.mode(), ThinkMode::Explicit, "dwell too short, must stay");
    }

    #[test]
    fn test_explicit_to_latent_fires_after_dwell() {
        // After w_e_to_l dwell window satisfied, rising entropy triggers switch.
        let mut c = SwiRController::new(cfg_small());
        c.step(5.0, 0); // Latent, ref=5
        c.step(2.0, 1); // → Explicit (switch_count=1, dwell=0, ref=2)
        c.step(2.0, 2); // stay Explicit (entropy not above ref), dwell=1
        c.step(2.0, 3); // stay Explicit, dwell=2
        // Now dwell ≥ w_e_to_l=2. Rising entropy → switch to Latent.
        let _ = c.step(3.0, 4);
        assert_eq!(c.mode(), ThinkMode::Latent, "should switch back to Latent");
        // Explicit→Latent does NOT bump switch_count.
        assert_eq!(c.switch_count(), 1);
    }

    #[test]
    fn test_switch_count_incremented_only_on_latent_to_explicit() {
        let mut c = SwiRController::new(cfg_small());
        c.step(5.0, 0); // Latent
        c.step(2.0, 1); // Latent→Explicit, switch_count=1
        c.step(2.0, 2); // Explicit, dwell 1
        c.step(2.0, 3); // Explicit, dwell 2
        c.step(3.0, 4); // Explicit→Latent, switch_count STAYS 1
        assert_eq!(c.switch_count(), 1, "Explicit→Latent must not bump");
        c.step(1.0, 5); // Latent→Explicit again, switch_count=2
        assert_eq!(c.switch_count(), 2);
    }

    #[test]
    fn test_convergence_trigger_at_half_cmax() {
        // c_max=4, convergence_fraction=0.5 → fires at switch_count=2.
        let mut c = SwiRController::new(cfg_small());
        // Drive 2 Latent→Explicit switches.
        // Switch 1:
        c.step(5.0, 0);
        c.step(2.0, 1); // switch_count=1
        // dwell then switch back to Latent:
        c.step(2.0, 2);
        c.step(2.0, 3);
        c.step(3.0, 4); // back to Latent
        // Now Latent→Explicit for switch 2:
        c.step(1.0, 5); // switch_count=2 → should enqueue CloseThink
        // The injection fires on the NEXT step (queue drained in step (1)).
        let drained = c.step(1.0, 6);
        assert_eq!(
            drained,
            StepAction::InjectControlToken(ControlToken::CloseThink),
            "convergence must enqueue CloseThink at ½c_max"
        );
    }

    #[test]
    fn test_termination_trigger_above_cmax() {
        // c_max=4. Once switch_count exceeds c_max, the controller must enqueue
        // ForceAnswerPrefix and start the answer-budget countdown, eventually
        // emitting Terminate. We drive this by forcing repeated Latent→Explicit
        // switches via the public step() interface.
        let mut c = SwiRController::new(cfg_small());
        // Drive 5 Latent→Explicit switches (> c_max=4). Each iteration: enter
        // Latent with high reference, drop entropy to switch to Explicit, dwell
        // long enough + rising entropy to switch back to Latent for next round.
        for i in 0..5u32 {
            let base = 10.0 * (i as f32 + 1.0);
            c.step(base, 10 * i); // Latent, ref=base
            c.step(base * 0.1, 10 * i + 1); // Latent→Explicit (sc = i+1)
            if c.switch_count() > c.config.c_max {
                break;
            }
            // Dwell in Explicit then rise to switch back to Latent.
            c.step(base * 0.1, 10 * i + 2);
            c.step(base * 0.1, 10 * i + 3);
            c.step(base * 0.5, 10 * i + 4); // Explicit→Latent
        }
        assert!(
            c.switch_count() > c.config.c_max,
            "test precond: switch_count must exceed c_max"
        );
        // Now drain pending injections until we see ForceAnswerPrefix or the
        // budget exhausts and we hit Terminate.
        let mut saw_force_answer = false;
        let mut saw_terminate = false;
        for s in 100..200u32 {
            match c.step(1.0, s) {
                StepAction::InjectControlToken(ControlToken::ForceAnswerPrefix) => {
                    saw_force_answer = true;
                }
                StepAction::Terminate => {
                    saw_terminate = true;
                    break;
                }
                _ => {}
            }
        }
        assert!(
            saw_terminate,
            "must Terminate after budget exhausts (saw_force_answer={saw_force_answer})"
        );
    }

    #[test]
    fn test_terminate_after_answer_budget_exhausted() {
        // answer_budget_b=3. After ForceAnswerPrefix, exactly 3 more steps then
        // Terminate.
        let mut c = SwiRController::new(cfg_small());
        // Force the termination state directly.
        c.answer_budget_remaining = Some(3);
        // Each step should drain the budget by 1; after 3, Terminate.
        let a1 = c.step(1.0, 0);
        let a2 = c.step(1.0, 1);
        let a3 = c.step(1.0, 2);
        let a4 = c.step(1.0, 3);
        // a1..a3 are emits (budget 2, 1, 0 remaining), a4 is Terminate.
        assert_ne!(a1, StepAction::Terminate);
        assert_ne!(a2, StepAction::Terminate);
        assert_ne!(a3, StepAction::Terminate);
        assert_eq!(a4, StepAction::Terminate);
    }

    #[test]
    fn test_signal_mix_schedule_at_switch_instants() {
        // α_t / β_t increase with step_index.
        let mut c = SwiRController::new(SwiRConfig {
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps: 100,
            ..cfg_small()
        });
        // Force a Latent→Explicit switch at step 0.
        c.step(5.0, 0); // Latent, ref=5
        c.step(2.0, 1); // Latent→Explicit at step 1
                        // should_mix_signal now armed with ExplicitExit.
        let mix = c.should_mix_signal();
        assert!(mix.is_some(), "mix must fire on step after switch");
        let (kind, ratio) = mix.unwrap();
        assert_eq!(kind, SignalMixKind::ExplicitExit);
        // β_t at step 1 / max 100 = 0.7 + 0.3 * 0.01 = 0.703.
        assert!((ratio - 0.703).abs() < 1e-4, "β_t wrong: {ratio}");
        // Next call should return None (mix consumed).
        assert!(c.should_mix_signal().is_none());
    }

    #[test]
    fn test_no_signal_mix_on_non_switch_steps() {
        let mut c = SwiRController::new(cfg_small());
        c.step(5.0, 0); // Latent, ref=5 (no switch, first step)
        assert!(
            c.should_mix_signal().is_none(),
            "first step is not a switch — no mix"
        );
        c.step(5.0, 1); // stay Latent (entropy equal, no switch)
        assert!(c.should_mix_signal().is_none());
    }

    // ─── G6 auto-fallback (Plan 275 T3.8) ───────────────────────────────────

    #[test]
    fn test_g6_kurtosis_escape_forces_explicit_from_latent() {
        // Default cfg_small has kurtosis_escape_threshold = ∞, so we override.
        let mut c = SwiRController::new(SwiRConfig {
            kurtosis_escape_threshold: 3.0,
            ..cfg_small()
        });
        // Step 0: initial Latent, ref entropy = 5.
        let a0 = c.step(5.0, 0);
        assert_eq!(c.mode(), ThinkMode::Latent);
        assert!(matches!(a0, StepAction::EmitSoftEmbedding));
        // Observe a high kurtosis → escape hatch fires on the next step.
        c.observe_kurtosis(5.0); // > 3.0 threshold
        // Step 1: even though entropy is equal (no entropy-drop signal), the
        // escape hatch forces Latent→Explicit.
        let a1 = c.step(5.0, 1);
        assert_eq!(c.mode(), ThinkMode::Explicit, "G6 escape must force Explicit");
        assert!(matches!(a1, StepAction::EmitToken(_)));
    }

    #[test]
    fn test_g6_kurtosis_escape_blocks_explicit_to_latent_reentry() {
        // After the escape hatch forces Explicit, subsequent rising-entropy
        // steps must NOT re-enter Latent while kurtosis stays high.
        let mut c = SwiRController::new(SwiRConfig {
            w_e_to_l: 1, // Easy to satisfy dwell so the only blocker is kurtosis.
            kurtosis_escape_threshold: 2.0,
            ..cfg_small()
        });
        c.step(5.0, 0); // Latent, ref=5
        c.observe_kurtosis(10.0); // > 2.0 → escape
        c.step(5.0, 1); // Latent→Explicit via escape
        assert_eq!(c.mode(), ThinkMode::Explicit);
        // Now entropy rises; without the escape hatch this would trigger
        // Explicit→Latent after w_e_to_l=1 dwell. Keep kurtosis high.
        c.observe_kurtosis(10.0);
        let a = c.step(10.0, 2); // entropy > ref (5)
        assert_eq!(c.mode(), ThinkMode::Explicit, "high kurtosis must block Latent re-entry");
        assert!(matches!(a, StepAction::EmitToken(_)));
    }

    #[test]
    fn test_g6_kurtosis_below_threshold_does_not_escape() {
        // kurtosis below threshold → normal behaviour, escape hatch inert.
        let mut c = SwiRController::new(SwiRConfig {
            kurtosis_escape_threshold: 3.0,
            ..cfg_small()
        });
        c.step(5.0, 0); // Latent, ref=5
        c.observe_kurtosis(1.0); // < 3.0, no escape
        c.step(5.0, 1); // No entropy drop, no escape → stay Latent
        assert_eq!(c.mode(), ThinkMode::Latent);
    }

    #[test]
    fn test_g6_nan_kurtosis_is_inert() {
        // Host never calls observe_kurtosis → last_kurtosis stays NaN →
        // escape hatch never fires (NaN > threshold is false). This preserves
        // backward compatibility for hosts that don't wire a kurtosis signal.
        let mut c = SwiRController::new(SwiRConfig {
            kurtosis_escape_threshold: 3.0,
            ..cfg_small()
        });
        c.step(5.0, 0); // Latent, ref=5
        // No observe_kurtosis call — last_kurtosis is NaN.
        c.step(5.0, 1); // Stay Latent (no entropy drop, NaN escape is inert).
        assert_eq!(c.mode(), ThinkMode::Latent);
    }

    #[test]
    fn test_g6_escape_releases_when_kurtosis_drops() {
        // Once kurtosis drops below threshold, normal switching resumes.
        let mut c = SwiRController::new(SwiRConfig {
            w_e_to_l: 1,
            kurtosis_escape_threshold: 2.0,
            ..cfg_small()
        });
        c.step(5.0, 0); // Latent, ref=5
        c.observe_kurtosis(10.0);
        c.step(5.0, 1); // → Explicit (escape)
        // Kurtosis drops below threshold.
        c.observe_kurtosis(1.0);
        // Now rising entropy should allow Explicit→Latent.
        let a = c.step(10.0, 2); // entropy > ref (5), dwell >= w_e_to_l (1)
        assert_eq!(c.mode(), ThinkMode::Latent, "escape released → Latent re-entry allowed");
        assert!(matches!(a, StepAction::EmitSoftEmbedding));
    }
}
