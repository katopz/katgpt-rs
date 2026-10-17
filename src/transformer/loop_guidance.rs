//! LoopCD recurrent-depth contrast guidance (Plan 617, arXiv:2610.02185).
//!
//! The probe_guidance combine family (Issue 865) with its weak side
//! re-sourced to a **completed loop iteration `k`** of the weight-shared
//! looped forward — modelless, zero training, aligned by construction (same
//! block, same head, same prefix; only compute differs):
//!
//! ```text
//! logit mode:   z_k = head(h_k)             # one extra scratch head pass
//!               z′  = z_R + ω·(z_R − z_k)   # = affine_combine(z_R, z_k, 1+ω)
//! hidden mode:  h′  = h_R + ω·(h_R − h_k)   # blended BEFORE head; ONE head pass
//! adaptive:     ω   = ω_max·[1 − (p₁ − p₂)] # top-two softmax margin of z_R
//! ```
//!
//! # Deviations pinned here (verdict-review honesty)
//!
//! - **Hidden-mode adaptive ω consumes the PREVIOUS decode step's margin.**
//!   The Eq-4 gate is computed in-step from `softmax(z_R)` in logit mode
//!   (z_R exists pre-guidance). Hidden mode's one-head-pass law means h_R is
//!   never read out, so the margin comes from the decoder's last actual
//!   output distribution (carried in [`LoopGuidance::last_margin`]; the
//!   decoder's current belief — settled tokens stay settled, guidance
//!   sharpens, margin grows, ω → 0). First call falls back to the fixed
//!   `config.omega`. A pinned unit test keeps this honest.
//! - **ω = 0 posture is a structural no-op** (no capture pass, no combine,
//!   no margin carry) — cheaper than "armed but identity", and
//!   bit-identical by construction rather than by f32 identity argument.
//!
//! # Precedence vs the exit family (Plan 617 T1.3)
//!
//! `AdvantageMarginGate` / `GainCostLoopHalter` / `LoopResidualExit` operate
//! on UNGUIDED per-iteration readouts and decide whether to keep looping.
//! Guidance transforms whichever prediction is FINALLY decoded (exit-
//! iteration or loop-R): the exits never see guided logits, the final decode
//! always does. The two mechanisms compose; neither subsumes the other.
//!
//! # Zero-alloc (T1.5)
//!
//! Scratch is pre-sized at [`LoopGuidance::new`] and only `resize`d (a
//! no-op at matching capacity) inside the forward — steady-state decode
//! allocates zero bytes with guidance armed (pinned by the counting-
//! allocator gate test).
//!
//! # Reference-capture contract
//!
//! `h_k` is captured at the END of loop iteration `ref_loop` (1-based) —
//! the completed-iteration state, post residual-gate. If the loop exits
//! (any halter/gate fires) BEFORE `ref_loop` completes, no reference exists
//! and guidance is a no-op for that step (documented; `ref_loop = 1` with
//! the default exits cannot happen — every exit family runs at `tau ≥ 1`
//! i.e. after iteration 1 completes, which is exactly the default capture
//! point). `ref_loop ≥ loop_count + 1` never captures → permanent no-op;
//! `validate()` warns via refusal only for the structurally-invalid 0.

#![cfg(feature = "loop_guidance")]

use crate::transformer::standard_lm_head;

/// Which space the contrast re-ranks in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum GuidanceMode {
    /// `z′ = z_R + ω·(z_R − z_k)` on the final logits. One extra scratch
    /// head pass at the capture iteration (two head passes per guided step).
    Logits,
    /// `h′ = h_R + ω·(h_R − h_k)` on the hidden state before the head.
    /// Single head pass per guided step (the coda/head cost is unchanged).
    Hidden,
}

/// LoopCD guidance configuration (Plan 617 T0.1).
///
/// Defaults are the structural no-op: `omega = 0.0` (bit-identical off),
/// `ref_loop = 1` (the first completed iteration — Huginn's hidden lane
/// needs a later burn-in reference; hyperparameter transfer is out of
/// scope, the fixture pins the mechanism).
#[derive(Clone, Copy, Debug)]
pub struct LoopGuidanceConfig {
    /// Logit-space or hidden-space contrast.
    pub mode: GuidanceMode,
    /// Fixed strength ω. Ignored when `adaptive` (except as the first-step
    /// fallback in hidden mode). Must be ≥ 0 — negative ω universally
    /// degrades (paper §G.1), refused at validation.
    pub omega: f32,
    /// Adaptive-gate cap: `ω_t = ω_max·[1 − (p₁ − p₂)] ∈ [0, ω_max]`.
    /// Must be ≥ `omega`.
    pub omega_max: f32,
    /// Margin-gate the strength per step (`ω = ω_max·[1 − (p₁ − p₂)]`)
    /// instead of applying the fixed `omega`.
    pub adaptive: bool,
    /// 1-based loop index whose completed hidden state becomes the weak
    /// reference `h_k`. Default 1. 0 is refused.
    pub ref_loop: usize,
}

impl Default for LoopGuidanceConfig {
    fn default() -> Self {
        Self {
            mode: GuidanceMode::Logits,
            omega: 0.0,
            omega_max: 0.0,
            adaptive: false,
            ref_loop: 1,
        }
    }
}

impl LoopGuidanceConfig {
    /// Structural validation (Plan 617 T0.1): `omega ≥ 0` (negative ω
    /// universally degrades per the paper's §G.1 — refuse, never clamp),
    /// `omega_max ≥ omega`, `ref_loop ≥ 1`.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.omega.is_finite() || self.omega < 0.0 {
            return Err(
                "loop_guidance: omega must be finite and >= 0 (negative omega universally degrades, paper §G.1)",
            );
        }
        if !self.omega_max.is_finite() || self.omega_max < self.omega {
            return Err("loop_guidance: omega_max must be finite and >= omega");
        }
        if self.ref_loop == 0 {
            return Err("loop_guidance: ref_loop is 1-based and must be >= 1");
        }
        Ok(())
    }

    /// True when this config can change any output. The ω=0 posture
    /// (fixed-ω with `omega == 0`, or adaptive with `omega_max == 0`) is a
    /// structural no-op: the forward skips capture AND combine entirely.
    #[inline]
    pub fn is_armed(&self) -> bool {
        if self.adaptive {
            self.omega_max > 0.0
        } else {
            self.omega > 0.0
        }
    }
}

/// Caller-owned guidance state: config + pre-sized scratch + the hidden
/// mode's carried margin. Construct once per decode session (the
/// `LoopDeepRun` precedent), pass `Some(&mut g)` per `forward_looped` call.
///
/// The struct is self-resetting per call: `forward_looped` clears
/// `captured` on entry, so a `LoopGuidance` reused across steps is correct
/// without caller bookkeeping.
pub struct LoopGuidance {
    /// Validated at construction (`expect` — a bad config is a programmer
    /// error; use [`LoopGuidanceConfig::validate`] for fallible checks).
    pub config: LoopGuidanceConfig,
    /// `h_k` scratch (hidden mode), n_embd.
    scratch_hidden: Vec<f32>,
    /// `z_k` scratch (logit mode), vocab_size.
    scratch_logits: Vec<f32>,
    /// Whether the reference was captured this call.
    captured: bool,
    /// Hidden mode: the softmax top-two margin of the PREVIOUS step's final
    /// logits (`None` before the first step → fixed-`omega` fallback).
    last_margin: Option<f32>,
}

impl LoopGuidance {
    /// Pre-sizes scratch (the ONE allocation of the object's lifetime) and
    /// validates the config.
    pub fn new(config: LoopGuidanceConfig, vocab_size: usize, n_embd: usize) -> Self {
        config
            .validate()
            .expect("loop_guidance: invalid config passed to LoopGuidance::new");
        Self {
            config,
            scratch_hidden: Vec::with_capacity(n_embd),
            scratch_logits: Vec::with_capacity(vocab_size),
            captured: false,
            last_margin: None,
        }
    }

    /// The object's own view of whether it can change output this call
    /// (config armed AND a reference captured). Exposed for tests.
    #[inline]
    pub fn guidance_active(&self) -> bool {
        self.config.is_armed() && self.captured
    }

    /// Whether the reference was captured THIS call (pub(crate): the tf-loop
    /// macro guards against double capture — its `Layer` mode revisits the
    /// capture point once per layer).
    #[inline]
    pub(crate) fn guidance_captured(&self) -> bool {
        self.captured
    }

    /// Hidden mode: the margin the NEXT step's adaptive gate will consume.
    #[inline]
    pub fn last_margin(&self) -> Option<f32> {
        self.last_margin
    }

    // ── hooks called from forward_looped / forward_training_free_loop ──

    /// Per-call reset: clears the capture flag (scratch contents are dead
    /// once `captured == false`, no zeroing needed).
    #[inline]
    pub(crate) fn begin_step(&mut self) {
        self.captured = false;
    }

    /// Capture `h_k` at the completed `ref_loop` iteration (the caller has
    /// already checked `tau + 1 == ref_loop`). Hidden mode stashes the
    /// state; logit mode runs the ONE extra scratch head pass.
    #[inline]
    pub(crate) fn capture_reference(
        &mut self,
        hidden: &[f32],
        weights_lm_head: &[f32],
        vocab_size: usize,
        n_embd: usize,
    ) {
        debug_assert!(!self.captured, "capture_reference called twice per step");
        match self.config.mode {
            GuidanceMode::Hidden => {
                self.scratch_hidden.resize(hidden.len(), 0.0);
                self.scratch_hidden.copy_from_slice(hidden);
            }
            GuidanceMode::Logits => {
                self.scratch_logits.resize(vocab_size, 0.0);
                crate::transformer::standard_lm_head(
                    &mut self.scratch_logits,
                    hidden,
                    weights_lm_head,
                    vocab_size,
                    n_embd,
                );
            }
        }
        self.captured = true;
    }

    /// Apply the contrast at the readout site. Returns the ω that was used
    /// (0.0 = skipped — not armed, no reference, or ω computed to exactly 0).
    ///
    /// - Hidden mode: `hidden ← affine_combine(hidden, h_k, 1+ω)` IN PLACE,
    ///   called BEFORE the head (the caller then reads out h′ — one head
    ///   pass total). `hidden` must be `Some` in this mode.
    /// - Logit mode: `logits ← affine_combine(logits, z_k, 1+ω)`, called
    ///   AFTER the head on h_R (the in-step Eq-4 margin reads these
    ///   pre-guidance logits — correct at the call site by construction).
    pub(crate) fn apply_at_readout(
        &mut self,
        hidden: Option<&mut [f32]>,
        logits: &mut [f32],
    ) -> f32 {
        if !self.config.is_armed() || !self.captured {
            return 0.0;
        }
        let omega = self.current_omega(logits);
        if omega == 0.0 {
            return 0.0;
        }
        let lam = 1.0 + omega;
        match self.config.mode {
            GuidanceMode::Hidden => {
                let h = hidden.expect("hidden mode apply_at_readout requires the hidden slice");
                katgpt_core::contrast_combine::affine_combine(
                    h,
                    &self.scratch_hidden[..h.len()],
                    lam,
                );
            }
            GuidanceMode::Logits => {
                katgpt_core::contrast_combine::affine_combine(
                    logits,
                    &self.scratch_logits[..logits.len()],
                    lam,
                );
            }
        }
        omega
    }

    /// The ω this step uses: fixed, or margin-gated (Eq 4). Logit mode
    /// computes the margin IN-STEP from the pre-guidance logits; hidden
    /// mode consumes the carried previous-step margin (fallback: fixed).
    #[inline]
    fn current_omega(&self, pre_guidance_logits: &[f32]) -> f32 {
        if !self.config.adaptive {
            return self.config.omega;
        }
        let margin = match self.config.mode {
            GuidanceMode::Logits => top_two_margin(pre_guidance_logits),
            GuidanceMode::Hidden => match self.last_margin {
                Some(m) => m,
                None => return self.config.omega,
            },
        };
        self.config.omega_max * (1.0 - margin)
    }

    /// Hidden mode: record the top-two margin of THIS step's final (post-
    /// guidance) logits — the decoder's actual output distribution — as
    /// the next step's adaptive-ω input. Logit mode must NOT call this
    /// (its gate is in-step, pre-guidance; a carried margin would be a
    /// second, different semantic). No-op in logit mode by design.
    #[inline]
    pub(crate) fn record_output_margin(&mut self, final_logits: &[f32]) {
        if self.config.is_armed() && self.config.mode == GuidanceMode::Hidden {
            self.last_margin = Some(top_two_margin(final_logits));
        }
    }
}

/// Top-two softmax probability gap `p₁ − p₂` of `logits`, zero-allocation.
///
/// Two passes over the slice (no full-softmax buffer): pass 1 finds the top
/// two values; pass 2 accumulates `exp(v − z₁)` for the shared denominator —
/// `p₁ = 1/denom` and `p₂ = exp(z₂−z₁)/denom`, so the margin is exact
/// without materializing either probability:
/// `p₁ − p₂ = (1 − exp(z₂ − z₁)) / Σexp(v − z₁)`.
///
/// Short-circuits: `< 2` elements → 0.0 (no margin); any non-finite value →
/// 1.0 (fail CLOSED — a broken logit row reads settled so ω is withheld;
/// guidance must never maximize contrast strength on a broken step).
pub fn top_two_margin(logits: &[f32]) -> f32 {
    if logits.len() < 2 {
        return 0.0;
    }
    // Pass 1: top two, branch-light.
    let mut z1 = f32::NEG_INFINITY;
    let mut z2 = f32::NEG_INFINITY;
    for &v in logits {
        if !v.is_finite() {
            // Fail closed: a broken logit row reads SETTLED (margin 1 → ω 0,
            // guidance withheld). The step is already broken; maximizing
            // contrast strength on it would only make it worse.
            return 1.0;
        }
        if v > z1 {
            z2 = z1;
            z1 = v;
        } else if v > z2 {
            z2 = v;
        }
    }
    if !z2.is_finite() {
        // Single-finite row: degenerate. Fail CLOSED — report settled (ω
        // withheld); a broken row must not maximize contrast strength.
        return 1.0;
    }
    // Pass 2: shared-shift softmax pieces. p1 = 1/denom and p2 =
    // exp(z2−z1)/denom share the denominator, so the margin is exact without
    // materializing p1: (1 − exp(z2−z1)) / Σexp(v − z1).
    let mut denom = 0.0f32;
    for &v in logits {
        denom += (v - z1).exp();
    }
    (1.0 - (z2 - z1).exp()) / denom
}

/// Eq 4: `ω = ω_max·[1 − (p₁ − p₂)]` — full strength on contested tokens,
/// withheld on settled ones. Exposed for the concentration pin test.
#[inline]
pub fn adaptive_omega(omega_max: f32, margin: f32) -> f32 {
    omega_max * (1.0 - margin)
}

/// Default config for tests/examples: adaptive logit-mode guidance at the
/// paper's generation-strength window midpoint (the window is
/// checkpoint-specific; this is a starting posture, not a transfer claim).
pub fn default_adaptive_config() -> LoopGuidanceConfig {
    LoopGuidanceConfig {
        mode: GuidanceMode::Logits,
        omega: 0.25,
        omega_max: 0.5,
        adaptive: true,
        ref_loop: 1,
    }
}

// ── Plan 617 T2.1/T2.2 — the decision-level settle exit + loop budget ────
//
// Scope note (T2.1): this exit wires into `forward_looped` (WeightShared)
// ONLY. The tf-loop's sub-step integration takes its step size from the
// TOTAL iteration count (`x += (1/k)·(y − x)`), so breaking a tf-loop early
// leaves a half-integrated state — a per-iteration exit is semantically
// incoherent there, unlike the weight-shared loop whose iterations are
// self-contained block passes.

/// Per-iteration readout policy for the settle exit (T2.1's cost-honesty
/// leg: a full `lm_head` readout every loop iteration rivals the loop block
/// itself at real vocabularies — the mitigation must be chosen, not
/// improvised).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettleReadout {
    /// Full `lm_head` readout at every evaluated iteration. Exact; the
    /// vocab×n_embd matvec per loop is the cost.
    Full,
    /// Bounded candidate-set readout: a full readout every `refresh_every`
    /// evaluated iterations (and the first), then only the `top_m` candidate
    /// ROWS are scored in between (m dot products instead of vocab).
    ///
    /// Documented approximation: between refreshes the tracked argmax is the
    /// argmax over the candidate set — a token outside the set that has
    /// become the true argmax is detected at the next refresh (and the
    /// stability streak resets on any change). The tracked decision may lag
    /// reality by at most `refresh_every − 1` iterations.
    CandidateSet { top_m: usize, refresh_every: usize },
}

/// Settle-exit configuration (Plan 617 T2.1). Runtime form of the offline
/// `agreement_exit` oracle (katgpt-core `loop_depth_probe`): exit when
/// `argmax(z_τ)` has been unchanged for `settle_patience` consecutive
/// EVALUATED loop iterations, never before `d_min` completed iterations.
#[derive(Clone, Copy, Debug)]
pub struct SettleExitConfig {
    /// `false` = accounting-only posture (T2.2): loops are counted and the
    /// budget reported, no readout runs, no exit fires — logits untouched.
    pub enabled: bool,
    /// Minimum completed iterations (1-based) before evaluation begins.
    /// Must be ≥ 1.
    pub d_min: usize,
    /// Consecutive equal argmaxes (in the evaluated suffix) required to exit.
    /// Must be ≥ 1.
    pub settle_patience: usize,
    /// Readout policy (see [`SettleReadout`]).
    pub readout: SettleReadout,
}

impl SettleExitConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.d_min == 0 {
            return Err("settle_exit: d_min must be >= 1 (completed iterations)");
        }
        if self.settle_patience == 0 {
            return Err("settle_exit: settle_patience must be >= 1");
        }
        if let SettleReadout::CandidateSet {
            top_m,
            refresh_every,
        } = self.readout
        {
            if top_m == 0 {
                return Err("settle_exit: CandidateSet top_m must be >= 1");
            }
            if refresh_every == 0 {
                return Err("settle_exit: CandidateSet refresh_every must be >= 1");
            }
        }
        Ok(())
    }
}

impl Default for SettleExitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            d_min: 1,
            settle_patience: 2,
            readout: SettleReadout::Full,
        }
    }
}

/// Why the loop ended (T2.2 budget accounting).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitKind {
    /// This settle exit fired (argmax stability reached).
    Settled {
        /// 1-based iteration at which the exit fired.
        at_loop: usize,
    },
    /// All `loop_count` iterations ran; stability was never reached.
    Exhausted,
    /// Another exit family (advantage gate / gain-cost halter / cadence
    /// residual / elastic owner) ended the loop first. The settle exit
    /// cannot name WHICH — the caller knows what it passed.
    Superseded,
}

/// Loop-budget accounting for one `forward_looped` call (T2.2). Consumable
/// by `InferenceOverrides`-style budget knobs later; no router wiring this
/// plan. Guided-vs-unguided comparisons read `loops_used` across two calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopBudget {
    /// Completed loop iterations (1-based count; a loop that exhausted its
    /// budget reports `loop_count`).
    pub loops_used: usize,
    /// The call's total loop budget (`effective_loop_count` result).
    pub loop_count: usize,
    /// Why the loop ended.
    pub exit: ExitKind,
}

/// Caller-owned settle-exit state: config + pre-sized readout scratch + the
/// stability tracker + the budget record. Construct once per decode session
/// (the [`LoopGuidance`] precedent), pass `Some(&mut s)` per `forward_looped`
/// call. Self-resetting per call (`forward_looped` calls `begin_step`).
///
/// # Precedence (T2.1)
///
/// The settle exit evaluates LAST among the per-iteration exits (advantage
/// gate → gain/cost halter → cadence residual → settle): the cheap
/// representation-space exits win same-iteration ties, and the decision-level
/// readout is paid only on iterations they declined. Whatever fires first
/// ends the loop; `LoopBudget::exit` records which class won from this
/// object's point of view.
///
/// # Unguided readouts (T1.3 precedence law)
///
/// The stability readouts are UNGUIDED by design — the exit family operates
/// on per-iteration states, guidance transforms whichever prediction is
/// FINALLY decoded. The two compose; neither subsumes the other.
pub struct SettleExit {
    /// Validated at construction.
    pub config: SettleExitConfig,
    full_scratch: Vec<f32>,
    cand_scratch: Vec<f32>,
    candidates: Vec<usize>,
    prev_argmax: Option<usize>,
    streak: usize,
    loops_run: usize,
    settled_at: Option<usize>,
    since_refresh: usize,
    last_budget: Option<LoopBudget>,
    vocab_size: usize,
    n_embd: usize,
}

impl SettleExit {
    /// Pre-sizes every scratch (the object's ONE allocation moment) and
    /// validates the config. `CandidateSet { top_m }` above `vocab_size`
    /// is clamped to it (the policy degenerates to Full at `==`).
    pub fn new(config: SettleExitConfig, vocab_size: usize, n_embd: usize) -> Self {
        config
            .validate()
            .expect("settle_exit: invalid config passed to SettleExit::new");
        let (top_m, full_cap) = match config.readout {
            SettleReadout::Full => (0usize, vocab_size),
            SettleReadout::CandidateSet { top_m, .. } => (top_m.min(vocab_size), vocab_size),
        };
        Self {
            config,
            full_scratch: Vec::with_capacity(full_cap),
            cand_scratch: Vec::with_capacity(top_m),
            candidates: Vec::with_capacity(top_m),
            prev_argmax: None,
            streak: 0,
            loops_run: 0,
            settled_at: None,
            since_refresh: 0,
            last_budget: None,
            vocab_size,
            n_embd,
        }
    }

    /// The budget of the LAST completed forward call (`None` before the
    /// first `finish`).
    #[inline]
    pub fn last_budget(&self) -> Option<LoopBudget> {
        self.last_budget
    }

    /// 1-based iteration this exit fired at, if it did (this call).
    #[inline]
    pub fn settled_at(&self) -> Option<usize> {
        self.settled_at
    }

    // ── hooks called from forward_looped ──

    /// Per-call reset (the [`LoopGuidance::begin_step`] precedent).
    #[inline]
    pub(crate) fn begin_step(&mut self) {
        self.prev_argmax = None;
        self.streak = 0;
        self.loops_run = 0;
        self.settled_at = None;
        self.since_refresh = 0;
    }

    /// Count one completed loop iteration (called EVERY iteration, enabled
    /// or not — the T2.2 accounting covers the pre-`d_min` window too).
    #[inline]
    pub(crate) fn record_iteration(&mut self) {
        self.loops_run += 1;
    }

    /// One evaluated iteration: readout → argmax → stability update.
    /// Returns true when the exit fires (the caller breaks the loop).
    /// Zero-alloc: every buffer is pre-sized; `resize` at capacity is a no-op.
    pub(crate) fn observe(&mut self, hidden: &[f32], lm_head: &[f32]) -> bool {
        let n = self.n_embd;
        let argmax_now = match self.config.readout {
            SettleReadout::Full => {
                self.full_scratch.resize(self.vocab_size, 0.0);
                standard_lm_head(&mut self.full_scratch, hidden, lm_head, self.vocab_size, n);
                argmax_full(&self.full_scratch)
            }
            SettleReadout::CandidateSet {
                top_m,
                refresh_every,
            } => {
                let m = top_m.min(self.vocab_size);
                let due = self.since_refresh == 0 || self.since_refresh >= refresh_every;
                if due {
                    self.full_scratch.resize(self.vocab_size, 0.0);
                    standard_lm_head(&mut self.full_scratch, hidden, lm_head, self.vocab_size, n);
                    select_top_m(&self.full_scratch, m, &mut self.candidates);
                    self.since_refresh = 1;
                } else {
                    self.since_refresh += 1;
                }
                self.cand_scratch.resize(m, 0.0);
                for (k, &c) in self.candidates.iter().take(m).enumerate() {
                    let row = &lm_head[c * n..(c + 1) * n];
                    let mut acc = 0.0f32;
                    for (w, &h) in row.iter().zip(hidden.iter()) {
                        acc += w * h;
                    }
                    self.cand_scratch[k] = acc;
                }
                self.candidates[argmax_full(&self.cand_scratch)]
            }
        };

        self.streak = if self.prev_argmax == Some(argmax_now) {
            self.streak + 1
        } else {
            1
        };
        self.prev_argmax = Some(argmax_now);

        if self.streak >= self.config.settle_patience {
            self.settled_at = Some(self.loops_run);
            return true;
        }
        false
    }

    /// After the loop: derive the budget (T2.2). `loop_count` is the call's
    /// effective budget; `Superseded` = this exit did not fire but the loop
    /// ended early (another exit family won the race).
    pub(crate) fn finish(&mut self, loop_count: usize) {
        let exit = if let Some(at_loop) = self.settled_at {
            ExitKind::Settled { at_loop }
        } else if self.loops_run < loop_count {
            ExitKind::Superseded
        } else {
            ExitKind::Exhausted
        };
        self.last_budget = Some(LoopBudget {
            loops_used: self.loops_run,
            loop_count,
            exit,
        });
    }
}

/// Argmax over `v` with the house total order (the advantage-gate call-site
/// comparator; NaN-safe, deterministic on ties by first index).
#[inline]
fn argmax_full(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
        .map_or(0, |(i, _)| i)
}

/// Top-`m` indices of `v`, largest first, written into `out` (pre-sized —
/// `clear` + pushes against reserved capacity; the zero-alloc law). m
/// selection passes over the slice, skipping already-chosen indices.
/// O(m·|v|) — at a real vocabulary the m·vocab scans are noise beside the
/// vocab·n_embd readout that produced `v`.
fn select_top_m(v: &[f32], m: usize, out: &mut Vec<usize>) {
    out.clear();
    for _ in 0..m.min(v.len()) {
        let mut best: Option<usize> = None;
        for (i, &val) in v.iter().enumerate() {
            if out.contains(&i) {
                continue;
            }
            best = match best {
                None => Some(i),
                Some(b)
                    if katgpt_core::float_order::cmp_for_max(val, v[b])
                        == std::cmp::Ordering::Greater =>
                {
                    Some(i)
                }
                Some(b) => Some(b),
            };
        }
        match best {
            Some(i) => out.push(i),
            None => break,
        }
    }
}
