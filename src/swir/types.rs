//! Configuration and shared types for SwiR Switch-Thinking.
//!
//! Distilled from SwiReasoning (ICLR 2026, arXiv:2510.05069). Paper defaults
//! below are the Qwen3-8B best-practices (paper Tab. 6); they will likely need
//! per-model tuning. See [`SwiRConfig::default`] for the paper's choices.

/// Current reasoning mode.
///
/// - [`Explicit`](Self::Explicit): emit a concrete token (decode as normal).
/// - [`Latent`](Self::Latent): emit a probability-weighted mixture of the
///   vocabulary embeddings (a point inside the vocab convex hull). This is the
///   "soft embedding" mode the paper uses for continuous exploration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ThinkMode {
    /// Decode one discrete token (paper's Explicit mode).
    Explicit = 0,
    /// Emit a probability-weighted vocab mixture (paper's Latent mode).
    Latent = 1,
}

/// Control tokens injected by the controller at switch / convergence / termination
/// instants. The caller resolves these to concrete token ids via
/// [`ControlToken::resolve_via`] using a
/// [`ControlTokenIds`](crate::thinking_cot::ControlTokenIds) populated from
/// the host tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ControlToken {
    /// `</think>` — close the reasoning block. Used when the controller detects
    /// convergence (switch_count in [½c_max, c_max]).
    CloseThink = 0,
    /// `</think>\n\nThe final answer is` — force the answer prefix. Used when
    /// the controller hits the overthinking guard (switch_count > c_max).
    ForceAnswerPrefix = 1,
}

impl ControlToken {
    /// Resolve this control token to a concrete vocabulary id using the
    /// host-supplied [`ControlTokenIds`](crate::thinking_cot::ControlTokenIds).
    ///
    /// Kept on `ControlToken` (not on `ControlTokenIds`) so the
    /// `thinking_cot` crate-level wiring type can stay free of swir
    /// dependencies — the dependency arrow is `swir → thinking_cot`, not the
    /// reverse.
    #[inline]
    pub fn resolve_via(
        self,
        ids: &crate::thinking_cot::ControlTokenIds,
    ) -> u32 {
        match self {
            ControlToken::CloseThink => ids.think_close,
            ControlToken::ForceAnswerPrefix => ids.force_answer_prefix,
        }
    }
}

/// Result of one [`SwiRController::step`](crate::swir::SwiRController::step)
/// call. This is the public surface the host decode loop branches on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepAction {
    /// Emit a concrete token. The controller fills the slot with `0` — the host
    /// is expected to overwrite it with the actual sampled token id. (Phase 2's
    /// `strategy_adapter` wires the real id.)
    EmitToken(u32),
    /// Emit a soft embedding (Latent mode). The host must call
    /// [`soft_embedding`](crate::swir::soft_embedding) into its own scratch
    /// buffer; optionally apply
    /// [`should_mix_signal`](crate::swir::SwiRController::should_mix_signal).
    EmitSoftEmbedding,
    /// Inject a control token at a switch / convergence / termination instant.
    /// The host translates [`ControlToken`] to a concrete id and feeds it.
    InjectControlToken(ControlToken),
    /// Stop generating. Fired when the answer budget is exhausted after a
    /// `ForceAnswerPrefix`.
    Terminate,
}

/// Configuration for a [`SwiRController`](crate::swir::SwiRController).
///
/// Field names mirror the paper's notation:
///
/// - `w_e_to_l` — Explicit→Latent dwell window (paper `W_E→L`). Number of
///   Explicit steps that must accumulate before a switch to Latent is allowed.
///   Paper best-practice is 512.
/// - `w_l_to_e` — Latent→Explicit dwell window (paper `W_L→E`). 0 means
///   "switch on the first entropy-rising step" (paper default).
/// - `c_max` — maximum number of Latent→Explicit switches before overthinking
///   suppression kicks in. Paper default 20.
/// - `c_convergence_fraction` — fraction of `c_max` at which convergence
///   (close-think) triggers. Paper default 0.5 → 10.
/// - `answer_budget_b` — number of tokens allowed after `ForceAnswerPrefix`
///   before terminating. Paper default 256.
/// - `alpha_0` / `beta_0` — initial signal-mix ratios for Latent entry / Explicit
///   exit respectively (paper Eq. 4). Paper defaults 0.6 / 0.7.
/// - `max_steps` — host-provided horizon for the α_t / β_t linear schedule.
/// - `kurtosis_escape_threshold` — excess kurtosis above which Latent mode is
///   bypassed (Plan 275 T3.8 G6 auto-fallback). The host supplies a per-step
///   kurtosis scalar via [`SwiRController::observe_kurtosis`]; if it exceeds
///   this threshold, the controller refuses to enter Latent mode (rigid-
///   constraint tasks where soft embeddings would hallucinate). `f32::INFINITY`
///   disables the escape hatch. Paper doesn't specify a value; 3.0 mirrors
///   the `kurtosis_gate` Plan 203 default (excess kurtosis of a Gaussian is 0).
#[derive(Debug, Clone, Copy)]
pub struct SwiRConfig {
    /// Explicit→Latent dwell window (paper `W_E→L`). Default 512.
    pub w_e_to_l: u32,
    /// Latent→Explicit dwell window (paper `W_L→E`). 0 = switch immediately on
    /// entropy-rising step. Default 0.
    pub w_l_to_e: u32,
    /// Max Latent→Explicit switches before overthinking suppression. Default 20.
    pub c_max: u32,
    /// Fraction of `c_max` at which convergence (close-think) fires. Default 0.5.
    pub c_convergence_fraction: f32,
    /// Tokens allowed after `ForceAnswerPrefix` before terminating. Default 256.
    pub answer_budget_b: u32,
    /// Initial Latent-entry signal-mix ratio (paper α_0). Default 0.6.
    pub alpha_0: f32,
    /// Initial Explicit-exit signal-mix ratio (paper β_0). Default 0.7.
    pub beta_0: f32,
    /// Horizon used by the α_t / β_t linear schedule. Must be set by the host
    /// (defaults to a conservative 1024 if the host doesn't override).
    pub max_steps: u32,
    /// Excess kurtosis threshold for the G6 auto-fallback (Plan 275 T3.8).
    /// `f32::INFINITY` disables the escape hatch.
    pub kurtosis_escape_threshold: f32,
}

impl Default for SwiRConfig {
    /// Paper best-practices (Qwen3-8B Tab. 6).
    fn default() -> Self {
        Self {
            w_e_to_l: 512,
            w_l_to_e: 0,
            c_max: 20,
            c_convergence_fraction: 0.5,
            answer_budget_b: 256,
            alpha_0: 0.6,
            beta_0: 0.7,
            max_steps: 1024,
            // Disabled by default — opt-in escape hatch per Plan 275 T3.8.
            // Hosts that wire a kurtosis signal (Plan 203/204) lower this to ~3.0.
            kurtosis_escape_threshold: f32::INFINITY,
        }
    }
}

impl SwiRConfig {
    /// Switch count at which convergence (close-think) fires. Equals
    /// `ceil(c_convergence_fraction * c_max)`.
    #[inline]
    pub fn convergence_switch_count(self) -> u32 {
        // ceil to avoid firing at 0 when fraction is tiny.
        let raw = (self.c_convergence_fraction * self.c_max as f32).ceil() as u32;
        raw.max(1)
    }

    /// Build a config with the paper's defaults for a specific embedding dim.
    /// `embedding_dim` is currently informational (used by the host to size the
    /// soft-embedding scratch buffer); the schedule defaults are model-agnostic.
    pub fn default_for_model(_embedding_dim: usize) -> Self {
        Self::default()
    }
}

/// Aggregate statistics from a controller run — useful for benchmarks / debug.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SwiRStats {
    /// Total Latent→Explicit + Explicit→Latent switches observed.
    pub switches_total: u32,
    /// Number of steps spent in Latent mode.
    pub latent_steps: u32,
    /// Number of steps spent in Explicit mode.
    pub explicit_steps: u32,
    /// Mode at termination (None if the run was abandoned without a Terminate).
    pub mode_at_termination: Option<ThinkMode>,
}
