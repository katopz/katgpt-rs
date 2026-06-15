//! `SwiRStrategyAdapter` — bridges the modelless [`SwiRController`] to the
//! [`ThinkingStrategy`] integration point.
//!
//! The adapter owns:
//!
//! - The [`SwiRController`] itself (mode state, switch counter, inject queue).
//! - A reusable `Vec<f32>` scratch buffer for the soft-embedding output
//!   (`embedding_dim` wide, allocated once on construction).
//! - A reusable `Vec<f32>` probs scratch for the softmax pass (same length as
//!   `logits`, allocated once).
//!
//! Per-step work (`on_step`):
//!
//! 1. Compute `H = entropy_from_logits(ctx.logits)` via the existing
//!    max-shift-stable kernel in `attn_match::adaptive_cot` — avoids a second
//!    entropy implementation and respects the project's "DRY" rule.
//! 2. Advance the controller: `self.ctrl.step(H, ctx.step_index)`.
//! 3. Translate the resulting [`StepAction`] into a [`StepDirective`]:
//!    - `EmitToken(_)` → host samples / argmaxes; we return `EmitToken(0)` and
//!      the host overwrites. (A real model host wires the sampled id here.)
//!    - `EmitSoftEmbedding` → call [`soft_embedding`] into our scratch, apply
//!      [`mix_thinking_signal`](crate::swir::mix_thinking_signal) if
//!      [`SwiRController::should_mix_signal`] fires, and return the buffer
//!      cloned into the directive.
//!    - `InjectControlToken(token)` → resolve via
//!      [`ControlToken::resolve_via`] and wrap in `InjectTokens(vec![id])`.
//!    - `Terminate` → return `Terminate`.
//!
//! # Allocation profile
//!
//! Hot-path allocations are limited to:
//!
//! - The clone into `EmitSoftEmbedding` (unavoidable: the directive owns its
//!   payload because the borrow checker can't tie the strategy's scratch
//!   lifetime to the call).
//! - The `vec![]` for `InjectTokens` (one small `Vec<u32>` per inject step —
//!   the controller injects at most one token per step, and only on a small
//!   fraction of steps).
//!
//! Both are amortised by the scratch reuse pattern documented on
//! [`ThinkingStrategy`].

use crate::swir::{
    entropy_from_logits, mix_thinking_signal, soft_embedding, ControlToken, StepAction,
    SwiRConfig, SwiRController,
};
use crate::thinking_cot::{ControlTokenIds, StepContext, StepDirective, ThinkingStrategy};

/// `ThinkingStrategy` adapter wrapping a [`SwiRController`].
///
/// Construct with [`SwiRStrategyAdapter::new`] (paper-default config) or
/// [`SwiRStrategyAdapter::with_config`] (host-supplied config). The
/// `embedding_dim` argument sizes the internal scratch buffers — it must
/// match the `embedding_dim` field on every [`StepContext`] passed to
/// [`ThinkingStrategy::on_step`].
#[derive(Debug)]
pub struct SwiRStrategyAdapter {
    /// The wrapped controller. Public-by-getter so the host can inspect
    /// mode / stats for dashboards.
    ctrl: SwiRController,
    /// Reusable softmax-probabilities scratch (length = vocab).
    probs_scratch: Vec<f32>,
    /// Reusable soft-embedding scratch (length = embedding_dim).
    soft_scratch: Vec<f32>,
}

impl SwiRStrategyAdapter {
    /// Construct with paper-default config (Qwen3-8B Tab. 6 best-practices).
    ///
    /// `vocab_size` and `embedding_dim` must match the host's model; they
    /// only size the internal scratch buffers.
    #[inline]
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self::with_config(vocab_size, embedding_dim, SwiRConfig::default())
    }

    /// Construct with a host-supplied config.
    #[inline]
    pub fn with_config(vocab_size: usize, embedding_dim: usize, config: SwiRConfig) -> Self {
        Self {
            ctrl: SwiRController::new(config),
            probs_scratch: vec![0.0; vocab_size],
            soft_scratch: vec![0.0; embedding_dim],
        }
    }

    /// Borrow the underlying controller (for `mode()` / `stats()` / tests).
    #[inline]
    pub fn controller(&self) -> &SwiRController {
        &self.ctrl
    }

    /// Mutably borrow the underlying controller.
    #[inline]
    pub fn controller_mut(&mut self) -> &mut SwiRController {
        &mut self.ctrl
    }

    /// Softmax `logits` into `probs_scratch` (max-shift stable) and return a
    /// borrow of the populated prefix.
    ///
    /// Kept as a helper rather than inlined so the kernel shape is visible to
    /// the auto-vectoriser and so a future SIMD-specialised softmax can drop
    /// in without touching `on_step`.
    #[inline]
    fn softmax_into_scratch<'b>(probs_scratch: &'b mut Vec<f32>, logits: &[f32]) -> &'b mut [f32] {
        if logits.is_empty() {
            probs_scratch.clear();
            return probs_scratch;
        }
        // Grow on demand if the host's vocab size grew between calls (rare;
        // mostly a defensive measure for tests that swap embedding matrices).
        if probs_scratch.len() < logits.len() {
            probs_scratch.resize(logits.len(), 0.0);
        }
        let n = logits.len();
        let view = &mut probs_scratch[..n];

        // Max-shift for numerical stability.
        let mut max_logit = f32::NEG_INFINITY;
        for &l in logits {
            if l > max_logit {
                max_logit = l;
            }
        }
        if !max_logit.is_finite() {
            // All -inf or NaN — emit uniform to keep the downstream soft
            // embedding finite. Degenerate input; the entropy kernel returns 0
            // in this case anyway.
            let inv = 1.0 / (n as f32);
            for p in view.iter_mut() {
                *p = inv;
            }
            return view;
        }
        let mut sum_exp = 0.0f32;
        for (i, &l) in logits.iter().enumerate() {
            let e = (l - max_logit).exp();
            view[i] = e;
            sum_exp += e;
        }
        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            let inv = 1.0 / (n as f32);
            for p in view.iter_mut() {
                *p = inv;
            }
            return view;
        }
        let inv_sum = 1.0 / sum_exp;
        for p in view.iter_mut() {
            *p *= inv_sum;
        }
        view
    }
}

impl ThinkingStrategy for SwiRStrategyAdapter {
    fn on_step(&mut self, ctx: &mut StepContext<'_>) -> StepDirective {
        // (1) Entropy via the existing max-shift-stable kernel.
        let entropy = entropy_from_logits(ctx.logits);

        // (2) Advance the controller.
        let action = self.ctrl.step(entropy, ctx.step_index);

        // (3) Translate the StepAction into a StepDirective.
        match action {
            StepAction::EmitToken(_placeholder) => {
                // The controller doesn't know the vocab — it signals "explicit
                // mode, host samples". Host-sampled id is wired by the host;
                // we emit 0 as a placeholder (matching the controller's
                // convention) so the host always overrides.
                StepDirective::EmitToken(0)
            }
            StepAction::EmitSoftEmbedding => {
                // Softmax logits → probs, then accumulate into scratch.
                let embedding_dim = ctx.embedding_dim;
                debug_assert_eq!(
                    self.soft_scratch.len(),
                    embedding_dim,
                    "embedding_dim drift: adapter built for {}, ctx says {}",
                    self.soft_scratch.len(),
                    embedding_dim
                );
                // Zero the scratch (soft_embedding accumulates).
                for x in self.soft_scratch.iter_mut() {
                    *x = 0.0;
                }
                {
                    // Borrow split: pass probs_scratch directly so the borrow
                    // checker sees disjoint fields, then re-borrow soft_scratch
                    // for the accumulate call.
                    let probs = Self::softmax_into_scratch(&mut self.probs_scratch, ctx.logits);
                    soft_embedding(probs, ctx.embedding_matrix, embedding_dim, &mut self.soft_scratch);
                }
                // Apply signal mixing if the controller signalled a switch
                // instant this step. The anchor embedding is the row of the
                // embedding matrix corresponding to the would-be control
                // token (`</think>` for ExplicitExit, forced-answer-prefix
                // for LatentEntry).
                if let Some((kind, ratio)) = self.ctrl.should_mix_signal() {
                    let ids: &ControlTokenIds = &ctx.control_token_ids;
                    let (anchor_id, _label) = match kind {
                        crate::swir::SignalMixKind::LatentEntry => {
                            // Entering Latent — anchor is the leading
                            // reasoning token (`</think>` if present, else
                            // the think_open sentinel).
                            (ids.think_close.max(ids.think_open), "latent-entry")
                        }
                        crate::swir::SignalMixKind::ExplicitExit => {
                            // Exiting Latent — anchor is the forced-answer
                            // prefix (or `</think>` if no dedicated id).
                            (ids.force_answer_prefix.max(ids.think_close), "explicit-exit")
                        }
                    };
                    let _ = _label; // for debug logs / future telemetry
                    if anchor_id != 0 {
                        let row_off = anchor_id as usize * embedding_dim;
                        if row_off + embedding_dim <= ctx.embedding_matrix.len() {
                            let anchor_row =
                                &ctx.embedding_matrix[row_off..row_off + embedding_dim];
                            mix_thinking_signal(&mut self.soft_scratch, anchor_row, ratio);
                        }
                    }
                }
                StepDirective::EmitSoftEmbedding(self.soft_scratch.clone())
            }
            StepAction::InjectControlToken(token) => {
                let id = resolve_control_token(token, &ctx.control_token_ids);
                StepDirective::InjectTokens(vec![id])
            }
            StepAction::Terminate => StepDirective::Terminate,
        }
    }
}

/// Resolve a controller-emitted [`ControlToken`] to a concrete vocab id.
///
/// Free-function twin of
/// [`ControlToken::resolve_via`](crate::swir::ControlToken::resolve_via),
/// kept here so the adapter reads linearly without flipping files.
#[inline]
fn resolve_control_token(token: ControlToken, ids: &ControlTokenIds) -> u32 {
    token.resolve_via(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swir::SwiRConfig;

    /// Build a logits vector whose softmax approximates `target_probs`.
    fn probs_to_logits(probs: &[f32], temperature: f32) -> Vec<f32> {
        // Inverse of softmax with temperature: logit = ln(p) * temperature.
        // The absolute scale doesn't matter — softmax is shift-invariant.
        probs
            .iter()
            .map(|&p| {
                let p = p.max(1e-12);
                (p.ln()) * temperature
            })
            .collect()
    }

    /// Build a flat `[vocab, dim]` embedding matrix from row vectors.
    fn mat(rows: &[Vec<f32>]) -> Vec<f32> {
        let mut out = Vec::new();
        for r in rows {
            out.extend_from_slice(r);
        }
        out
    }

    #[test]
    fn entropy_drives_initial_latent_step() {
        // First step: controller is in Latent mode (default), emits soft.
        let vocab = 4;
        let dim = 3;
        let emb = mat(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ]);
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let logits = probs_to_logits(&probs, 1.0);

        let mut adapter = SwiRStrategyAdapter::new(vocab, dim);
        let ids = ControlTokenIds::default();
        let mut ctx = StepContext {
            logits: &logits,
            step_index: 0,
            max_steps: 16,
            embedding_matrix: &emb,
            embedding_dim: dim,
            control_token_ids: ids,
        };
        let directive = adapter.on_step(&mut ctx);
        match directive {
            StepDirective::EmitSoftEmbedding(v) => {
                assert_eq!(v.len(), dim, "soft embedding width must equal dim");
                // Uniform probs → centroid of the 4 rows = (1.5/4, 1.5/4, 1.5/4).
                let expected = 1.5 / 4.0;
                for d in 0..dim {
                    assert!(
                        (v[d] - expected).abs() < 1e-5,
                        "dim {d}: got {}, expected {}",
                        v[d],
                        expected
                    );
                }
            }
            other => panic!("first step should be EmitSoftEmbedding, got {other:?}"),
        }
        assert_eq!(adapter.controller().mode(), crate::swir::ThinkMode::Latent);
    }

    #[test]
    fn inject_close_think_resolves_to_id() {
        // Force convergence by setting c_max = 2 → convergence at ceil(0.5*2) = 1.
        // Drive one Latent→Explicit switch and verify the next step injects.
        let vocab = 3;
        let dim = 2;
        let emb = mat(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]]);
        let mut adapter = SwiRStrategyAdapter::with_config(
            vocab,
            dim,
            SwiRConfig {
                w_e_to_l: 1, // Switch out of Explicit on first rising-entropy step.
                w_l_to_e: 0,
                c_max: 2,
                c_convergence_fraction: 0.5,
                answer_budget_b: 4,
                alpha_0: 0.6,
                beta_0: 0.7,
                max_steps: 16,
                kurtosis_escape_threshold: f32::INFINITY,
            },
        );
        let ids = ControlTokenIds {
            think_open: 10,
            think_close: 42, // sentinel so we can detect resolution
            force_answer_prefix: 99,
        };

        // Step 0: Latent, high entropy (uniform).
        let high = probs_to_logits(&[0.34, 0.33, 0.33], 1.0);
        let low = probs_to_logits(&[0.98, 0.01, 0.01], 1.0);

        // Step 0 — initial reference, latent.
        let mut ctx = StepContext {
            logits: &high,
            step_index: 0,
            max_steps: 16,
            embedding_matrix: &emb,
            embedding_dim: dim,
            control_token_ids: ids,
        };
        let _ = adapter.on_step(&mut ctx);

        // Step 1 — Latent with entropy *below* reference → switch to Explicit,
        // bumping switch_count to 1 (≥ convergence threshold ½c_max=1) →
        // enqueue CloseThink. The step returns EmitToken(0) and the queue
        // drains on step 2.
        let mut ctx = StepContext {
            logits: &low,
            step_index: 1,
            max_steps: 16,
            embedding_matrix: &emb,
            embedding_dim: dim,
            control_token_ids: ids,
        };
        let d1 = adapter.on_step(&mut ctx);
        assert!(
            matches!(d1, StepDirective::EmitToken(0)),
            "step 1 should emit a token (latent→explicit), got {d1:?}"
        );

        // Step 2 — drain CloseThink.
        let mut ctx = StepContext {
            logits: &low,
            step_index: 2,
            max_steps: 16,
            embedding_matrix: &emb,
            embedding_dim: dim,
            control_token_ids: ids,
        };
        let d2 = adapter.on_step(&mut ctx);
        match d2 {
            StepDirective::InjectTokens(tokens) => {
                assert_eq!(tokens, vec![42], "CloseThink should resolve to ids.think_close");
            }
            other => panic!("step 2 should inject CloseThink, got {other:?}"),
        }
    }

    #[test]
    fn terminate_after_answer_budget_exhausted() {
        // Use a config where the convergence window never fires so we go
        // straight to the overthinking guard after c_max+1 switches.
        // Set c_convergence_fraction huge so conv threshold > c_max: the
        // convergence branch is skipped, overthinking guard kicks in at
        // switch_count > c_max.
        let vocab = 2;
        let dim = 2;
        let emb = mat(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut adapter = SwiRStrategyAdapter::with_config(
            vocab,
            dim,
            SwiRConfig {
                w_e_to_l: 1,
                w_l_to_e: 0,
                c_max: 1,
                c_convergence_fraction: 10.0, // conv threshold = ceil(10*1) = 10 > c_max=1
                answer_budget_b: 1,
                alpha_0: 0.6,
                beta_0: 0.7,
                max_steps: 32,
                kurtosis_escape_threshold: f32::INFINITY,
            },
        );
        let ids = ControlTokenIds {
            think_open: 0,
            think_close: 1,
            force_answer_prefix: 2,
        };

        // Entropy schedule (probs, not logits):
        //   high (uniform) → low (peaky) : Latent → Explicit, switch_count = 1 > c_max
        //                               → overthinking guard, enqueue ForceAnswerPrefix, budget = 1.
        //   Next step drains ForceAnswerPrefix (budget 1→0).
        //   Next step sees budget=0 → Terminate.
        //
        // But we also need 2 switches because switch_count starts at 0 and we
        // need it > c_max=1, so switch_count must reach 2. Easiest: alternate
        // entropy twice.
        let high = probs_to_logits(&[0.5, 0.5], 1.0);
        let low = probs_to_logits(&[0.99, 0.01], 1.0);

        let run = |adapter: &mut SwiRStrategyAdapter, logits: &[f32], step: u32| -> StepDirective {
            let mut ctx = StepContext {
                logits,
                step_index: step,
                max_steps: 32,
                embedding_matrix: &emb,
                embedding_dim: dim,
                control_token_ids: ids,
            };
            adapter.on_step(&mut ctx)
        };

        // Step 0: Latent init (reference entropy = high).
        let _ = run(&mut adapter, &high, 0);
        // Step 1: Latent → Explicit (entropy drops), switch_count = 1.
        //   With conv threshold=10 and c_max=1: 1 >= 10 false, 1 > 1 false → no inject.
        let d1 = run(&mut adapter, &low, 1);
        assert!(
            matches!(d1, StepDirective::EmitToken(0)),
            "step 1 should emit token (no inject yet), got {d1:?}"
        );
        // Step 2: Explicit; entropy still low → no switch; dwell++.
        let _ = run(&mut adapter, &low, 2);
        // Step 3: Explicit → Latent (entropy rises after dwell=2 >= w_e_to_l=1).
        let _ = run(&mut adapter, &high, 3);
        // Step 4: Latent → Explicit (entropy drops), switch_count = 2 > c_max=1
        //   → overthinking guard, enqueue ForceAnswerPrefix, budget = 1.
        //   Step returns EmitToken(0) (inject drains on step 5).
        let _ = run(&mut adapter, &low, 4);
        // Step 5: drain ForceAnswerPrefix. budget was 1, decrement to 0.
        let d5 = run(&mut adapter, &low, 5);
        match d5 {
            StepDirective::InjectTokens(t) => assert_eq!(t, vec![2]),
            other => panic!("step 5 should drain ForceAnswerPrefix, got {other:?}"),
        }
        // Step 6: budget_remaining = 0 → Terminate.
        let d6 = run(&mut adapter, &low, 6);
        assert!(
            matches!(d6, StepDirective::Terminate),
            "step 6 should Terminate, got {d6:?}"
        );
    }

    #[test]
    fn softmax_into_scratch_handles_empty_logits() {
        let mut adapter = SwiRStrategyAdapter::new(0, 4);
        let view = SwiRStrategyAdapter::softmax_into_scratch(&mut adapter.probs_scratch, &[]);
        assert!(view.is_empty());
    }

    #[test]
    fn softmax_into_scratch_handles_uniform() {
        let mut adapter = SwiRStrategyAdapter::new(4, 2);
        let logits = vec![1.0; 4]; // Uniform → softmax uniform = 0.25 each.
        let view = SwiRStrategyAdapter::softmax_into_scratch(&mut adapter.probs_scratch, &logits);
        for &p in view.iter() {
            assert!((p - 0.25).abs() < 1e-5, "got {p}");
        }
    }

    #[test]
    fn softmax_into_scratch_handles_degenerate_neg_inf() {
        let mut adapter = SwiRStrategyAdapter::new(3, 2);
        let logits = vec![f32::NEG_INFINITY; 3];
        let view = SwiRStrategyAdapter::softmax_into_scratch(&mut adapter.probs_scratch, &logits);
        // Uniform fallback.
        let expected = 1.0 / 3.0;
        for &p in view.iter() {
            assert!((p - expected).abs() < 1e-5, "got {p}");
        }
    }

    #[test]
    fn resolve_control_token_helper_matches_method() {
        let ids = ControlTokenIds {
            think_open: 10,
            think_close: 20,
            force_answer_prefix: 30,
        };
        assert_eq!(resolve_control_token(ControlToken::CloseThink, &ids), 20);
        assert_eq!(
            resolve_control_token(ControlToken::ForceAnswerPrefix, &ids),
            30
        );
    }
}
