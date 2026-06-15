//! Thinking-strategy trait — the integration point for adaptive decode-loop
//! controllers (Plan 194, extended by Plan 275 / Plan 212 / Plan 195).
//!
//! `thinking_cot` is a *meta-feature* that pulls in the bandit, prune, and
//! probe machinery required by [`ThinkingController`](crate::speculative) plus
//! any number of [`ThinkingStrategy`] implementations. A strategy is a small
//! state machine that sits inside the decode loop and, given the current
//! step's logits, decides what the loop should emit next.
//!
//! Three kinds of strategy are anticipated:
//!
//! - **Token-space** strategies (`EmitToken`) — the standard argmax / sample
//!   path. `ThinkingController`'s bandit-driven early-exit is the canonical
//!   example.
//! - **Latent-space** strategies (`EmitSoftEmbedding`) — emit a continuous
//!   mixture of vocabulary embeddings in place of a discrete token. SwiR
//!   (Plan 275) is the first katgpt-rs primitive to use this.
//! - **Forcing** strategies (`InjectTokens`, `Terminate`) — overwrite the
//!   next-token output to inject control tokens (`</think>`, forced-answer
//!   prefix) or stop the run.
//!
//! The trait is intentionally minimal: `on_step` is the only required method.
//! Strategies that need richer signals (KV-cache state, attention stats, …)
//! should extend [`StepContext`] rather than the trait.
//!
//! # Why this lives in `thinking_cot` (not `swir`)
//!
//! The trait is host-side: any future strategy (CollapseAware, ChainFold,
//! future MUX-arm-bandit strategies) implements it. The dependency arrow is
//! `swir → thinking_cot` (per Plan 275 T2.6), so the trait must be reachable
//! without enabling swir. SwiR-specific wiring types that need to be visible
//! to a strategy-agnostic host (like [`ControlTokenIds`]) therefore live
//! here, not under `swir/`.

/// Concrete vocabulary ids for the control tokens a thinking strategy may
/// inject.
///
/// The host populates this from the tokenizer once at startup and feeds it
/// to each strategy via [`StepContext::control_token_ids`]. Strategies that
/// don't inject control tokens can leave the fields at 0.
///
/// Field names mirror the canonical Qwen3 / DeepSeek `<think>` block syntax;
/// the host is free to populate them with whatever ids its tokenizer uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ControlTokenIds {
    /// `<think>` — reasoning-block open. Currently informational; no strategy
    /// in the tree injects this (the model produces it via the prompt).
    pub think_open: u32,
    /// `</think>` — reasoning-block close. SwiR maps its `CloseThink` control
    /// token here.
    pub think_close: u32,
    /// `</think>\n\nThe final answer is` — forced-answer prefix. SwiR maps
    /// `ForceAnswerPrefix` here. For tokenizers that don't encode this as a
    /// single token, the host should use the `</think>` id and emit the
    /// remaining text via post-processing.
    pub force_answer_prefix: u32,
}

/// A single step's input to a [`ThinkingStrategy`].
///
/// `logits` is the model's next-token distribution **before** any strategy
/// modification; `embedding_matrix` is the flattened
/// `[vocab, embedding_dim]` token-embedding table (read-only — strategies
/// must not mutate model weights); `control_token_ids` carries the host's
/// tokenizer-resolved ids for the control tokens the strategy may inject.
///
/// Lifetime is tied to the host's borrow of the model state for the duration
/// of the `on_step` call. Strategies MUST NOT retain references past the
/// call.
pub struct StepContext<'a> {
    /// Raw logits from the LM head (length = vocab_size).
    pub logits: &'a [f32],
    /// 0-based decode step index within the current generation.
    pub step_index: u32,
    /// Host-configured horizon — strategies use this to schedule time-varying
    /// parameters (e.g. SwiR's α_t / β_t blend schedule).
    pub max_steps: u32,
    /// Flattened `[vocab, embedding_dim]` token-embedding matrix. Required
    /// for latent-mode strategies that compute a soft embedding. Token-space
    /// strategies may ignore this.
    pub embedding_matrix: &'a [f32],
    /// Width of each row in `embedding_matrix`.
    pub embedding_dim: usize,
    /// Host-resolved control-token ids (`</think>`, forced-answer prefix, …).
    pub control_token_ids: ControlTokenIds,
}

/// What a [`ThinkingStrategy`] wants the decode loop to do this step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepDirective {
    /// Emit a discrete token — the decode loop should feed `token_id` to the
    /// model as the input for the next step. Strategies that want the host's
    /// normal sampling path can use this with the host-sampled id.
    EmitToken(u32),
    /// Emit a soft (continuous) embedding in place of a discrete token. The
    /// host uses this slice verbatim as the next-step input embedding
    /// (skipping the token-id lookup entirely).
    ///
    /// The `Vec<f32>` payload is owned by the directive (cloned from the
    /// strategy's scratch) so the borrow checker stays happy across the
    /// strategy / host boundary without lifetime gymnastics. Hot-path
    /// allocations are amortised by the strategy's reusable scratch buffer —
    /// the clone is `embedding_dim * 4` bytes, well under one cache line for
    /// typical `embedding_dim ≤ 4096`.
    EmitSoftEmbedding(Vec<f32>),
    /// Inject one or more concrete control-token ids (e.g. `</think>`,
    /// forced-answer prefix) ahead of any naturally-emitted token. The host
    /// processes the entire `Vec` before resuming the normal decode loop.
    InjectTokens(Vec<u32>),
    /// Stop generating. The host should finalise the run (truncate the
    /// response, write the trace, etc.).
    Terminate,
}

/// A strategy plugged into the decode loop. See the module docs for the three
/// intended flavours (token-space, latent-space, forcing).
///
/// # Lifecycle
///
/// The host constructs the strategy once at the start of generation, calls
/// `on_step` for each decode position, and discards it when
/// [`StepDirective::Terminate`] is returned (or the host's own max-step
/// budget is exhausted).
///
/// # Allocation contract
///
/// Hot-path allocation is discouraged but not forbidden —
/// `EmitSoftEmbedding` requires returning owned data because Rust's borrow
/// checker can't express "this borrow outlives the function call by exactly
/// one iteration". A well-behaved strategy reuses an internal scratch `Vec`
/// across calls and only clones it into the directive when the latent path
/// is actually taken.
pub trait ThinkingStrategy {
    /// Inspect `ctx` and decide what to do this step. May mutate internal
    /// state (mode, switch counters, scratch buffers).
    fn on_step(&mut self, ctx: &mut StepContext<'_>) -> StepDirective;
}
