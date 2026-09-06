use super::*;

/// Batched forward pass — process N tokens at consecutive positions in one call (Issue 020, Path B).
///
/// This is the DenseMesh vertex-parameter-sharing batched entry point. The 4
/// hidden nodes in a `[1, 4, 1]` mesh share one set of `TransformerWeights`
/// (paper §3.3). When their forwards are batched into this single call, we
/// amortise:
///   - function-call overhead (1 call vs N),
///   - `batch_logits` buffer growth (resized once, not allocated per token),
///   - config-derived constants (hoisted outside the token loop).
///
/// Each `tokens[i]` is forwarded at position `pos_start + i` and writes K/V
/// into `cache` at that position. The returned slice for token `i` spans
/// `[i * vocab_size .. (i+1) * vocab_size]` of [`ForwardContext::batch_logits`].
///
/// # Safety of returned slices
///
/// The returned `Vec<&mut [f32]>` contains N disjoint mutable slices into a
/// single `Vec<f32>`. This is sound because the slices are non-overlapping
/// (each spans `vocab_size` consecutive elements), but the borrow checker
/// cannot prove disjointness through raw pointers, so we use
/// `slice::from_raw_parts_mut` inside a small `unsafe` block. The slices are
/// valid for the lifetime `'a` of the `ctx` borrow. Callers must not outlive
/// `ctx`.
///
/// # When to use
///
/// Prefer `forward_batched` when forwarding ≥ 2 tokens of the same model
/// back-to-back (e.g. DenseMesh hidden-layer vertex batch, prefill). For a
/// single token, use [`forward()`] — the batched path has no advantage at N=1.
#[allow(clippy::too_many_arguments)]
pub fn forward_batched<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    pos_start: usize,
    config: &Config,
) -> Vec<&'a mut [f32]> {
    let n_tokens = tokens.len();
    let vocab = config.vocab_size;
    debug_assert!(n_tokens > 0, "forward_batched requires at least one token");

    // Grow the flat batch buffer once (no per-token alloc). `resize` is a no-op
    // when capacity already suffices — callers that repeatedly batch the same
    // width pay zero allocation after the first call (plasma tier).
    ctx.batch_logits.resize(n_tokens * vocab, 0.0);

    // Hoist the config-derived `vocab` stride outside the per-token loop. The
    // per-token forward already hoists its own layer-loop invariants; we only
    // add the batch-stride and output-index arithmetic here.
    for (i, &token) in tokens.iter().enumerate() {
        let pos = pos_start + i;
        // `forward` returns `&mut ctx.logits` (single-token buffer) but also
        // mutably borrows `ctx`. To then write into `ctx.batch_logits` we'd
        // need a second mutable borrow of `ctx`. The borrow checker can't see
        // that `logits` and `batch_logits` are disjoint fields, so we copy
        // through raw pointers. SAFETY: `ctx.logits.len() == vocab` (invariant
        // from ForwardContext::new) and `batch_logits.len() == n_tokens *
        // vocab` (from the resize above). `out_start + vocab <= len`. The two
        // regions never overlap because `logits` is the single-token buffer
        // and `batch_logits` is the flat batch buffer.
        let _logits = forward(ctx, weights, cache, token, pos, config);
        let out_start = i * vocab;
        // SAFETY: see comment above the loop.
        let src = ctx.logits.as_ptr();
        let dst = unsafe { ctx.batch_logits.as_mut_ptr().add(out_start) };
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, vocab);
        }
    }

    // Return disjoint per-token mutable slices into batch_logits. The lifetime
    // `'a` ties the slices to the `ctx` borrow. Each slice is `vocab` long.
    // SAFETY: batch_logits has length `n_tokens * vocab`; each slice covers a
    // disjoint `[i*vocab .. (i+1)*vocab]` range. The raw-pointer reborrow does
    // not violate aliasing because no two returned slices overlap.
    let base = ctx.batch_logits.as_mut_ptr();
    let mut out: Vec<&'a mut [f32]> = Vec::with_capacity(n_tokens);
    for i in 0..n_tokens {
        // SAFETY: offset `i * vocab` is in-bounds (total len = n_tokens * vocab).
        // Slice length `vocab` stays in-bounds. Slices are disjoint across `i`.
        let ptr = unsafe { base.add(i * vocab) };
        let slice: &'a mut [f32] = unsafe { std::slice::from_raw_parts_mut(ptr, vocab) };
        out.push(slice);
    }
    out
}

/// Forward with optional LoRA and domain latent (Plan 038).
/// Convenience wrapper for callers that need both conditioning signals.
#[cfg(feature = "domain_latent")]
#[allow(clippy::too_many_arguments)]
pub fn forward_with_domain_latent<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    cache.advance_pos(pos);
    #[cfg(feature = "coda_fusion")]
    {
        forward_coda(ctx, weights, cache, token, pos, config, lora, domain_latent)
    }
    #[cfg(not(feature = "coda_fusion"))]
    {
        forward_base(ctx, weights, cache, token, pos, config, lora, domain_latent)
    }
}

// ---------------------------------------------------------------------------
// LT2 Looped Inference (Plan 108, Research 73)
// ---------------------------------------------------------------------------

/// Looped transformer forward pass — weight-shared T-pass loop.
///
/// Applies the same layer weights T times in succession, yielding effective
/// depth T×n_layer with no extra parameters. Key insight from LT2: looping
/// uniquely synergizes with subquadratic attention — T loops turn rank-1
/// DPLR state updates into rank-T updates.
///
/// Per-loop residual gate: h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ-1)
/// Zero-init ρ_τ means first iteration is h̃^(1) (no residual from "previous").
///
/// Feature gate: `lt2_looped` (requires `hla_attention`).
///
/// # Plan 283 T2.2 — AdvantageMarginGate integration
///
/// When `weight_shared_advantage_gate` is enabled AND `recursion_gate` is
/// `Some(gate)`, after each `tau` iteration the loop computes logits via the
/// readout `lm_head` matmul and asks the gate whether the step improved the
/// candidate's prediction. If the gate signals dead compute (`should_recurse`
/// returns `false`), the outer loop breaks early, skipping the remaining
/// `loop_count - tau - 1` iterations.
///
/// When `recursion_gate` is `None` (or the feature is off), behavior is
/// byte-identical to the ungated baseline: the full `loop_count` iterations
/// run and no extra work is performed.
///
/// # Overhead estimate (gated path only)
///
/// Per iteration the gate adds one `lm_head` matmul (`vocab_size × n_embd`
/// FLOPs) plus one `should_recurse` check (`O(vocab)`, <1µs for vocab ≤ 128
/// per Bench 056 G3). For a typical micro config (`vocab=27, n_embd=16,
/// n_layer=1`) this is ~432 FLOPs versus ~512 FLOPs per layer pass — about
/// 0.8× one layer's compute. At larger configs the ratio improves further
/// (one `lm_head` matmul vs `n_layer` layer passes). The gate pays for itself
/// if it saves ≥2 iterations (Bench 056 shows 2.68×–6.76× reduction at
/// vocab ≤ 128). Allocations happen once on the first gated iteration, then
/// are reused via `resize`/`clear` (no per-iteration heap traffic).
#[cfg(feature = "lt2_looped")]
#[allow(dead_code, clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn forward_looped<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    ahla_cache: &mut crate::hla::MultiLayerAhlaCache,
    token: usize,
    pos: usize,
    config: &Config,
    residual_gate: &crate::types::ResidualGate,
    sdpa_gate: &crate::types::SdpaOutputGate,
    #[cfg(feature = "sleep_consolidation")] gdn2_cache: Option<
        &'a mut crate::gdn2::MultiLayerGdn2Cache,
    >,
    #[cfg(feature = "sleep_consolidation")] sleep_config: Option<&'a crate::sleep::SleepConfig>,
    // Plan 283 T2.2: optional recursion gate. `None` = byte-identical to
    // baseline (no gate, all `loop_count` iterations run). `Some(gate)` =
    // after each `tau > 0` iteration, compute logits and ask the gate whether
    // the step improved the candidate; break early on dead compute.
    #[cfg(feature = "weight_shared_advantage_gate")] recursion_gate: Option<
        &mut crate::pruners::self_advantage::AdvantageMarginGate,
    >,
    // Issue 035 (Research 273 — ELT Any-Time inference): per-call elastic
    // loop override. `None` = use `config.loop_mode`'s natural loop count
    // (byte-identical to pre-Issue-035 behavior). `Some(L)` runs L loops
    // clamped to `[loop_min, 2×loop_max]` per `Config::effective_loop_count`.
    // No feature gate required (it's a parameter); zero cost when `None`.
    elastic_loop_override: Option<usize>,
    // Plan 304 T2.1: optional gain/cost halter. `None` = byte-identical to
    // pre-Plan-304 behavior (all `loop_count` iterations run). `Some(halter)`
    // = after each iteration, evaluate gain/cost scissors and break early on
    // `HaltDecision::Halt`. Composes with `elastic_loop_override` (Issue 035):
    // if the caller passes `Some(L)` for the override, the halter is IGNORED
    // (static override wins — see T2.2). Feature-gated to keep the no-halter
    // build zero-cost: when `gain_cost_halt` is off, this parameter slot does
    // not exist in the signature, so callers don't pass it either.
    #[cfg(feature = "gain_cost_halt")] halter: Option<
        &mut katgpt_core::gain_cost_halt::GainCostLoopHalter,
    >,
    // Issue 717 T1/T3/T4: opt-in deep-run control (per-K state-norm snapshots
    // + logit-finite tripwire + the `lt2_deep_stability` damping /
    // direction-scale knobs). `None` = bit-identical to pre-Issue-717
    // behavior. The parameter itself is ungated (the Issue 035
    // elastic-override precedent: "no feature gate required, zero cost when
    // None"); only the knob FIELDS are feature-gated, so the stabilization
    // path compiles to nothing under `lt2_looped` alone.
    deep_run: Option<&mut super::loop_deep::LoopDeepRun>,
    // Issue 731 T1: residual-gated early exit (EqR action item 7.2,
    // arXiv:2605.21488). `None` = bit-identical to pre-Issue-731 behavior
    // (all `loop_count` iterations run). `Some(probe)` = after each iteration
    // (τ ≥ 1), feed the probe ‖h_τ − h_{τ−1}‖ and break early when it reports
    // the loop settled: the magnitude window mean < `tau`, OR the cadence
    // verdict is `Settled` — never before the probe's `d_min` floor. The
    // probe is caller-owned (τ / d_min config — the Issue-035 param precedent
    // instead of a Config field: no constructor ripple). Feature-gated so the
    // no-probe build carries no parameter at all (the `gain_cost_halt`
    // precedent).
    #[cfg(feature = "cadence_gate")]
    residual_exit: Option<
        &mut katgpt_core::convergence_cadence::LoopResidualExit,
    >,
) -> &'a mut [f32] {
    use crate::types::HybridPattern;

cache.advance_pos(pos);

    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = crate::types::kv_dim(config);

    // Loop-invariant values hoisted outside all loops
    let scale = ctx.attn_scale;
    let t_n = pos + 1;

    // Issue 035: derive effective loop count, applying elastic override if
    // present. `None` is byte-identical to the prior `match config.loop_mode`
    // block (verified by `Config::effective_loop_count` returning `base`).
    let loop_count = config.effective_loop_count(elastic_loop_override);

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    katgpt_core::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // Plan 283 T2.2 — recursion-gate scratch buffers.
    // Declared at zero capacity so the no-gate path (`recursion_gate == None`)
    // performs no allocation. The gated path resizes them exactly once (first
    // gated iteration) and reuses them thereafter via `resize`/`clear`, which
    // are no-ops once the capacity matches `vocab_size`. This honors the
    // hot-loop rule (no allocation inside the outer loop body).
    #[cfg(feature = "weight_shared_advantage_gate")]
    let mut recursion_gate = recursion_gate;
    #[cfg(feature = "weight_shared_advantage_gate")]
    let mut _gate_scratch_logits: Vec<f32> = Vec::new();
    #[cfg(feature = "weight_shared_advantage_gate")]
    let mut _gate_prev_logits: Vec<f32> = Vec::new();

    // Plan 304 T2.2 + T2.3 — gain/cost halter plumbing.
    //
    // `halter_active` is computed once outside the loop: the halter is ONLY
    // consulted when the caller passed `Some(halter)` AND did NOT pass a static
    // `elastic_loop_override` (T2.2 — static override wins). This bool is
    // `false` in both feature-off builds (cfg-stripped) and feature-on builds
    // where the caller asked for a fixed loop count — so the per-iteration
    // halter branch is statically or branch-predicted-not-taken in all
    // no-op paths. Zero cost when the halter is inactive.
    //
    // `prev_step_buf` holds the previous loop's update direction
    // `h^(tau-1) - h^(tau-2)` so the next iteration can compute cos θ against
    // it via `angular_change`. Allocated ONCE per `forward_looped` call (not
    // per iteration) — honors the hot-loop rule. Matches the existing
    // `_gate_scratch_logits` pattern: declared even in the no-halter path but
    // never grown unless the halter fires.
    #[cfg(feature = "gain_cost_halt")]
    let mut halter = halter;
    #[cfg(feature = "gain_cost_halt")]
    let halter_active = elastic_loop_override.is_none();
    #[cfg(feature = "gain_cost_halt")]
    let mut prev_step_buf: Vec<f32> = Vec::with_capacity(n);
    #[cfg(feature = "gain_cost_halt")]
    let mut curr_step_buf: Vec<f32> = Vec::with_capacity(n);
    // `cost_floor` is cached on the first halter evaluation (tau == 1) as
    // `0.01 × first_step_size`, mirroring LoopCoder-v2's flat Ω(r) tax. See
    // the plan's Open Question 1 resolution (Phase 2 ships the fixed-tax
    // default; riir-ai can override with coherence-decay/staleness).
    #[cfg(feature = "gain_cost_halt")]
    let mut cost_floor: f32 = 0.0;

    // Issue 717 — hoist the deep-run actives ONCE (hot-loop rule: the
    // per-iteration work at the bottom of this loop is data-driven off these
    // bools; `None` / knob-absent ⇒ branch-not-taken ⇒ bit-identical, the
    // G1 contract). `deep_run` is reborrown per use; `mut` only enables
    // `as_deref_mut`.
    let mut deep_run = deep_run;
    #[cfg(feature = "lt2_deep_stability")]
    let scales_active = deep_run
        .as_ref()
        .and_then(|r| r.direction_scales.as_ref())
        .is_some_and(|s| s.radial != 1.0 || s.tangential != 1.0);
    #[cfg(feature = "lt2_deep_stability")]
    let damping_active = deep_run
        .as_ref()
        .and_then(|r| r.damping.as_ref())
        .is_some_and(|d| d.alpha > 0.0);

    // Issue 731 T1 — the residual-exit probe is reborrowed per use.
    #[cfg(feature = "cadence_gate")]
    let mut residual_exit = residual_exit;

    // 2. Outer loop: T passes over all layers
    for tau in 0..loop_count {
        // Plan 428 — Inter-loop RMSNorm: normalize the carry-over hidden state
        // before it enters this loop iteration's layer pass. Applied for tau > 0
        // (the first iteration uses the embedding directly). This controls
        // residual norm growth in weight-shared looped inference — the PoC
        // benchmark (examples/loop_stability_poc.rs) validated 3.34× norm ratio
        // vs 11.19× baseline at T=12. Zero cost when LoopStabilityMode::None.
        // Issue 698 T2 — FixedAnchor composes the norm (GRT's gate consumes LN
        // inputs: normalization is a prerequisite, not a competitor).
        #[cfg(feature = "loop_stability_fix")]
        if tau > 0
            && matches!(
                config.loop_stability_mode,
                crate::types::LoopStabilityMode::InterLoopNorm
                    | crate::types::LoopStabilityMode::FixedAnchor
                    | crate::types::LoopStabilityMode::StateNoise { .. }
            )
        {
            crate::types::rmsnorm(&mut ctx.x[..n]);
        }

        // Issue 698 T6 — per-step state noise (GRT arXiv:2608.15062, the
        // paper's smallest ablation, modelless corollary): BLAKE3-seeded
        // Gaussian per (pos, tau) added to the normed loop input. `scale`
        // is RELATIVE to the state's own RMS. Applied BEFORE the `prev_h`
        // save, so the perturbed state is what the layer pass consumes AND
        // what the drifting gate injects ("state" noise, not input-only).
        // `scale == 0.0` skips the branch entirely → bit-identical to
        // InterLoopNorm (the flag-off pin). Zero cost when the mode is not
        // StateNoise (data-driven branch, never taken in production configs).
        #[cfg(feature = "loop_stability_fix")]
        if tau > 0
            && let crate::types::LoopStabilityMode::StateNoise { scale } =
                config.loop_stability_mode
            && scale != 0.0
        {
            add_blake3_state_noise(&mut ctx.x[..n], pos, tau, scale);
        }

        // Issue 698 T8 — roll h^(τ-2) into prev_prev_h BEFORE prev_h is
        // overwritten, so the conditional gate can compute
        // cos(S(τ−1), S(τ−2)). At gate time (post-pass): prev_h = h^(τ−1),
        // prev_prev_h = h^(τ−2). One extra n_embd copy, paid only when the
        // conditional gate is installed (zero cost otherwise).
        #[cfg(feature = "loop_stability_fix")]
        if residual_gate.conditional.is_some() {
            ctx.prev_prev_h[..n].copy_from_slice(&ctx.prev_h[..n]);
        }

        // Save h^(τ-1) for residual gate
        ctx.prev_h[..n].copy_from_slice(&ctx.x[..n]);

        // Adaptive Depth Tier: cap layer count at inference time (Plan 284 T10).
        // Composes with Hydra: tier sets upper bound, Hydra skips within that bound.
        let max_layer = ctx
            .depth_tier
            .map_or(config.n_layer, |t| t.max_layers(config.n_layer));

        // 3. Inner loop: weight-shared layer pass
        for (layer_idx, layer_weights) in weights.layers.iter().enumerate().take(max_layer) {
            let layer_cache = &mut cache.layers[layer_idx];

            // Determine if this layer uses full SDPA or linear attention
            let is_full = match config.hybrid_pattern {
                HybridPattern::Uniform => true,
                HybridPattern::Interleave { full_ratio } => {
                    (layer_idx % full_ratio) == full_ratio - 1
                }
                HybridPattern::Bookend => layer_idx == 0 || layer_idx == weights.layers.len() - 1,
            };

            // Pre-attention: RMSNorm → save residual
            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

            // QKV projections
            crate::types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            crate::types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            crate::types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

            if is_full {
                // Full SDPA: store K,V in cache and compute standard attention
                let pos_off = pos * kvd;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ctx.k.as_ptr(),
                        layer_cache.key.as_mut_ptr().add(pos_off),
                        kvd,
                    );
                    std::ptr::copy_nonoverlapping(
                        ctx.v.as_ptr(),
                        layer_cache.value.as_mut_ptr().add(pos_off),
                        kvd,
                    );
                }

                // Multi-head attention with GQA
                for h in 0..config.n_head {
                    let kv_group = ctx.kv_group_lut[h] as usize;
                    unsafe {
                        attention_head(
                            &ctx.q,
                            &layer_cache.key,
                            &layer_cache.value,
                            &mut ctx.attn_out,
                            &mut ctx.scores,
                            h * hd,
                            kv_group * hd,
                            kvd,
                            hd,
                            t_n,
                            scale,
                        );
                    }
                }
            } else {
                // Linear attention via AHLA recurrent step
                let ahla_layer = &mut ahla_cache.layers[layer_idx];
                ctx.attn_out[..n].fill(0.0);

                for h in 0..config.n_head {
                    let kv_group = ctx.kv_group_lut[h] as usize;
                    let head_state = &mut ahla_layer.heads[h];

                    crate::hla::ahla_step(
                        &mut ahla_layer.pkv[kv_group],
                        &mut ahla_layer.mk[kv_group],
                        head_state,
                        &ctx.q[h * hd..(h + 1) * hd],
                        &ctx.k[kv_group * hd..(kv_group + 1) * hd],
                        &ctx.v[kv_group * hd..(kv_group + 1) * hd],
                        hd,
                        ahla_cache.gamma,
                        &mut ctx.attn_out[h * hd..(h + 1) * hd],
                        &mut ctx.scores[..hd],
                    );
                }
            }

            // SDPA output gate (if configured): sigmoid(W_gate @ attn_out) ⊙ attn_out
            // Zero-init weights → sigmoid(0) = 0.5 (neutral half-pass).
            // Paper: +0.3–0.5 avg points on zero-shot benchmarks.
            if config.gated_attn && is_full {
                sdpa_gate.forward(&mut ctx.attn_out[..n], n, &mut ctx.scores[..n]);
            }

            // Output projection + residual
            crate::types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
            katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

            // MLP: save residual → RMSNorm → MLP → residual
            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);
            #[cfg(feature = "gated_mlp")]
            {
                // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
                crate::types::matmul(
                    &mut ctx.hidden,
                    &layer_weights.mlp_w1,
                    &ctx.x,
                    config.mlp_hidden,
                    n,
                );
                crate::types::matmul(
                    &mut ctx.hidden2,
                    &layer_weights.mlp_w_up,
                    &ctx.x,
                    config.mlp_hidden,
                    n,
                );
                crate::types::swiglu_inplace(&mut ctx.hidden, &ctx.hidden2);
            }
            #[cfg(not(feature = "gated_mlp"))]
            crate::types::matmul_relu(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            crate::types::matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
            );
            katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);
        }

        // Per-loop residual gate: h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ-1)
        // ρ_τ is zero-init → first iteration: h^(0) = h̃^(0) (no residual)
        // Issue 698 T2 — under FixedAnchor the injected state is the FROZEN
        // anchor (h^(0), hoisted once below) instead of the drifting h^(τ-1):
        // GRT Table 11 (frozen prelude output 2.68 vs drifting h(r−1) 3.38).
        if tau > 0 {
            #[cfg(feature = "loop_stability_fix")]
            let injected: &[f32] =
                if config.loop_stability_mode == crate::types::LoopStabilityMode::FixedAnchor {
                    &ctx.loop_anchor[..n]
                } else {
                    &ctx.prev_h[..n]
                };
            #[cfg(not(feature = "loop_stability_fix"))]
            let injected: &[f32] = &ctx.prev_h[..n];

            // Issue 698 T3 + T8 — the convex blend path. The copy weight g
            // comes EITHER from the pre-built schedule (T3:
            // `convex_schedule`) OR, when that is absent and the conditional
            // gate is installed, from the trajectory itself (T8:
            // `σ(β·(cos(S(τ−1), S(τ−2)) − θ) + b)`, open on divergence).
            // Both feed the same blend — the free bound and the contraction
            // argument apply identically to any g ∈ [0, 1]. Schedule absent
            // AND conditional absent → the additive path below, byte-
            // identical to pre-T3 behavior (data-driven branches, no cfg).
            // T8 gate source, cfg-split the same way as `injected` above —
            // a cfg-dependent match arm reads as `None => None` in the OFF
            // build, where clippy's manual_map/needless_match auto-fix would
            // silently delete the ON-build conditional-gate fallback
            // (Issue 701 R3b slice: the auto-applied rewrite was reverted).
            #[cfg(feature = "loop_stability_fix")]
            let adaptive_g = residual_gate.convex_gate_at(tau).or_else(|| {
                residual_gate.conditional_gate_at(tau, &ctx.prev_h[..n], &ctx.prev_prev_h[..n])
            });
            #[cfg(not(feature = "loop_stability_fix"))]
            let adaptive_g = residual_gate.convex_gate_at(tau);
            if let Some(g) = adaptive_g {
                // Two-pass scalar blend: h ← (1 − g)·h̃, hidden ← g·src,
                // h += hidden. Op order pinned by the T3 spec test
                // (g = 1 → h = src exactly; g = 0 → h unchanged exactly).
                katgpt_core::simd::simd_scale_inplace(&mut ctx.x[..n], 1.0 - g);
                ctx.hidden[..n].copy_from_slice(injected);
                katgpt_core::simd::simd_scale_inplace(&mut ctx.hidden[..n], g);
                katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.hidden[..n]);
            } else {
                let gate_offset = tau * n;
                if gate_offset + n <= residual_gate.gates.len() {
                    // ctx.x += gates ⊙ injected  (element-wise fused multiply-accumulate)
                    ctx.hidden[..n].copy_from_slice(injected);
                    katgpt_core::simd::simd_scale_mul_inplace(
                        &mut ctx.hidden[..n],
                        &residual_gate.gates[gate_offset..gate_offset + n],
                        1.0,
                    );
                    katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.hidden[..n]);
                }
            }
        }

        // Issue 698 T2 — hoist the fixed anchor once the first loop iteration
        // completes: anchor = h^(0), the loop core's own first-pass output.
        // Interpretant note: on our prelude-less arch the tau==0 PRE-pass state
        // is the raw embedding — the paper's DISTINCT, worse anchor arm (Table
        // 11: 3.73 vs 2.68) — so the frozen anchor is the first-pass OUTPUT
        // (the prelude-output analog). One copy per forward, tau==0 only; the
        // loop always runs ≥ 1 iteration, so the anchor is always valid before
        // the first gated iteration (tau == 1) can read it.
        #[cfg(feature = "loop_stability_fix")]
        if tau == 0 && config.loop_stability_mode == crate::types::LoopStabilityMode::FixedAnchor {
            ctx.loop_anchor[..n].copy_from_slice(&ctx.x[..n]);
        }

        // Plan 283 T2.2 — AdvantageMarginGate dead-compute check.
        // Only active when `weight_shared_advantage_gate` is enabled AND the
        // caller passed `Some(gate)`. When `None`, this block is compiled out
        // of the feature-off build and is a runtime no-op in the feature-on
        // build, so the no-gate path stays byte-identical to baseline.
        //
        // The check runs only for `tau > 0` (the first iteration has no
        // pre-recursion logits to compare against). It computes the current
        // iteration's logits via the same `lm_head` matmul used for the final
        // readout, then asks the gate whether the candidate's prediction
        // improved. If not, the remaining iterations are dead compute and we
        // break early.
        #[cfg(feature = "weight_shared_advantage_gate")]
        {
            if let Some(gate) = recursion_gate.as_deref_mut() {
                // Compute this iteration's logits into a local scratch buffer
                // (NOT ctx.logits — that must remain untouched so the final
                // readout at the end of the function is byte-identical to the
                // no-gate path). `resize` is a no-op after the first call.
                _gate_scratch_logits.resize(config.vocab_size, 0.0);
                standard_lm_head(
                    &mut _gate_scratch_logits,
                    &ctx.x,
                    &weights.lm_head,
                    config.vocab_size,
                    n,
                );
                if tau > 0 && !_gate_prev_logits.is_empty() {
                    // Candidate = argmax of the current (post-recursion)
                    // logits — the model's current best prediction.
                    let candidate = _gate_scratch_logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            katgpt_core::float_order::cmp_for_max(**a, **b)
                        }).map_or(0, |(i, _)| i);
                    if !gate.should_recurse(&_gate_prev_logits, &_gate_scratch_logits, candidate) {
                        // Dead compute detected: this iteration did not
                        // improve the candidate's prediction, so further
                        // iterations are unlikely to either. Break the outer
                        // loop and use the current hidden state.
                        break;
                    }
                }
                // Stash this iteration's logits as the next iteration's
                // "pre" distribution. `clear` + `extend_from_slice` reuses
                // the existing allocation (no per-iteration heap traffic
                // after the first call).
                _gate_prev_logits.clear();
                _gate_prev_logits.extend_from_slice(&_gate_scratch_logits);
            }
        }

        // Plan 304 T2.3 — gain/cost halt evaluation.
        //
        // Only active when ALL of: (a) `gain_cost_halt` feature is on,
        // (b) the caller passed `Some(halter)`, (c) no static
        // `elastic_loop_override` was set (`halter_active`, T2.2), and
        // (d) `tau > 0` (the first iteration has no previous hidden state
        // to compute a step against — and `prev_step_buf` is empty). When any
        // condition fails this block is either cfg-stripped or a runtime
        // no-op, so the no-halter path stays byte-identical to pre-Plan-304.
        //
        // **DEVIATION from Plan T2.3 (documented):** the plan called for
        // effective-rank delta as the gain signal. But the per-loop hidden
        // state in `forward_looped` is a SINGLE vector `ctx.x[..n]` (one row,
        // S=1), for which `hidden_erank` returns 0.0 (degenerate — the kernel
        // short-circuits on `s == 1`). We therefore use `step_size` as the
        // gain signal: `||h^(tau) - h^(tau-1)||₂`. This is monotone in
        // refinement, cheaper than erank, and the kernel ships `step_size`
        // exactly for this use (see plan Open Question 2 resolution).
        #[cfg(feature = "gain_cost_halt")]
        if halter_active
            && tau > 0
            && let Some(h) = halter.as_deref_mut()
        {
            // gain = ||h^(tau) - h^(tau-1)||₂. `ctx.prev_h` was saved at
            // the top of this iteration (before the layer pass), so it
            // holds h^(tau-1); `ctx.x` now holds h^(tau) post-pass.
            let gain = katgpt_core::gain_cost_halt::step_size(&ctx.x[..n], &ctx.prev_h[..n]);

            // cost = fixed tax (flat Ω(r), LoopCoder-v2 default).
            // Cached on the first evaluation (tau == 1) as 0.01 × the
            // first step size. Open Question 1 resolution: Phase 2 ships
            // the flat-tax default; riir-ai can override with
            // coherence-decay/staleness by not using this code path.
            if tau == 1 {
                cost_floor = 0.01 * gain;
            }
            let cost = cost_floor;

            // cos θ between the current and previous update directions.
            // curr_step = h^(tau) - h^(tau-1); prev_step_buf holds
            // h^(tau-1) - h^(tau-2) from the prior iteration. On tau == 1
            // there is no tau-2 state, so cos θ is 0.0 (neutral,
            // non-oscillatory — does not trip the detector).
            curr_step_buf.clear();
            for (cur, prev) in ctx.x[..n].iter().zip(ctx.prev_h[..n].iter()) {
                curr_step_buf.push(cur - prev);
            }
            let cos_theta = if prev_step_buf.is_empty() {
                0.0
            } else {
                katgpt_core::gain_cost_halt::angular_change(&curr_step_buf, &prev_step_buf)
            };

            // The halter expects a 1-based loop index (`tau` is 0-based).
            let decision = h.halt_decision(tau + 1, gain, cost, cos_theta);
            if let katgpt_core::gain_cost_halt::HaltDecision::Halt { .. } = decision {
                break;
            }

            // Roll the current step into the previous-step slot for the
            // next iteration's cos θ. `std::mem::swap` avoids a copy;
            // the now-swapped-in `curr_step_buf` will be `clear()`'d
            // at the top of the next evaluation.
            std::mem::swap(&mut curr_step_buf, &mut prev_step_buf);
            h.update_prev_step(gain);
        }

        // Issue 731 T1 — residual-gated early exit evaluation.
        //
        // Only active when `cadence_gate` is on AND the caller passed
        // `Some(probe)`. Runs for τ > 0 (the first iteration has no previous
        // state to step against — same precondition as the halter above), and
        // the probe enforces its own `d_min` completed-iteration floor
        // internally. Break BEFORE the per-iteration deep-run knobs below:
        // once the loop is settled, rescaling an update it will never use is
        // dead work (the same ordering the gain/cost halter uses above).
        #[cfg(feature = "cadence_gate")]
        if tau > 0
            && let Some(exit) = residual_exit.as_deref_mut()
        {
            // Step norm = ‖h^(τ) − h^(τ−1)‖₂ — the same gain signal the
            // Plan-304 halter consumes, computed inline (no gain_cost_halt
            // feature dependency): sum of squared differences + sqrt, no
            // allocation. `ctx.prev_h` was saved at the top of this
            // iteration, so it holds h^(τ−1).
            let mut acc = 0.0_f32;
            for (cur, prev) in ctx.x[..n].iter().zip(ctx.prev_h[..n].iter()) {
                let d = cur - prev;
                acc += d * d;
            }
            if exit.observe(acc.sqrt()) {
                break;
            }
        }

        // ── Issue 717 T4 — tangential/radial update rescale ─────────────
        // Decomposes THIS iteration's full update (post residual-gate
        // injection) against the pre-pass state `prev_h` and rescales the
        // two components. Skipped entirely when the knob is absent/neutral
        // ({1,1}): the decompose-recombine round-trip is NOT bit-identical
        // (reassociation), so neutrality must skip, not recompute.
        //
        // T5 (f32-state contract, carried at the site it protects): the
        // state crossing every loop-iteration boundary is `ctx.x: Vec<f32>`
        // — full f32 in, full f32 out, no sub-f32 storage anywhere in the
        // carry path of ANY arm of this loop. This is the OPPOSITE numerics
        // regime of riir-ai Bench 802, where f16-KV deviation DILUTES with
        // attention context: attention rounding averages out across
        // positions, while weight-tied recurrence AMPLIFIES rounding with
        // depth (upstream sotaku: BF16 @4096 = 43.7% vs FP32 98.6%; compiled
        // chunks retaining f32 intermediates recovered to 92.5%). Deep-loop
        // serving must never route this state through f16/bf16 — pinned
        // behaviorally by `issue_717_t1_t2_deep_baseline::f32_state_contract`.
        #[cfg(feature = "lt2_deep_stability")]
        if tau > 0
            && scales_active
            && let Some(run) = deep_run.as_deref_mut()
            && let Some(s) = run.direction_scales
        {
            super::loop_deep::apply_direction_scales(
                &mut ctx.x[..n],
                &ctx.prev_h[..n],
                s.radial,
                s.tangential,
            );
        }

        // ── Issue 717 T3 — delayed damping ──────────────────────────────
        // `h ← (1−α)·h + α·h_prev` once the burn-in has elapsed (sotaku's
        // runtime, checkpoint-agnostic rescue; closed form `project_lambda`:
        // a locally-linear mode λ → 1−α+αλ). Applied LAST in the iteration
        // body — after every injection/gate/halter path — so halter gain
        // measurements stay undamped while the state that carries forward,
        // and the final readout, see the damped value. `alpha == 0.0` never
        // reaches here (`damping_active` is false ⇒ bit-identical, G1).
        #[cfg(feature = "lt2_deep_stability")]
        if tau > 0
            && damping_active
            && let Some(run) = deep_run.as_deref_mut()
            && let Some(d) = run.damping
            && tau >= d.burn_in
        {
            super::loop_deep::apply_damping(&mut ctx.x[..n], &ctx.prev_h[..n], d.alpha);
        }

        // ── Issue 717 T1 — per-K deep-run stats (zero cost when None) ───
        // Snapshot AFTER the knobs above, so the recorded state is exactly
        // what carries into the next iteration (and, at the last snapshot,
        // into the readout). `robust_norm` is max-abs-scaled — deep-loop
        // states overflow the naive Σv² at ‖x‖ ≳ 1e19 while still far
        // inside the f32 value range — and returns NaN iff the state
        // contains a non-finite value (the tripwire marker).
        if let Some(run) = deep_run.as_deref_mut()
            && run.snapshot_every > 0
            && (tau + 1) % run.snapshot_every == 0
        {
            let x = &ctx.x[..n];
            let norm = super::loop_deep::robust_norm(x);
            run.stats.snapshots_taken += 1;
            run.stats.state_norms.push(norm);
            if norm.is_nan() && run.stats.state_non_finite_at.is_none() {
                run.stats.state_non_finite_at = Some(run.stats.snapshots_taken - 1);
            }
            if run.capture_states {
                run.stats.state_snapshots.push(x.to_vec());
            }
            if run.check_logits {
                // Logit-finite tripwire: one extra `lm_head` matmul per
                // snapshot, opt-in (`check_logits`). Scratch grown once,
                // reused thereafter.
                run.logit_scratch.resize(config.vocab_size, 0.0);
                standard_lm_head(
                    &mut run.logit_scratch,
                    &ctx.x,
                    &weights.lm_head,
                    config.vocab_size,
                    n,
                );
                if run.stats.logits_non_finite_at.is_none()
                    && run.logit_scratch.iter().any(|l| !l.is_finite())
                {
                    run.stats.logits_non_finite_at = Some(run.stats.snapshots_taken - 1);
                }
            }
        }
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    standard_lm_head(
        &mut ctx.logits,
        &ctx.x,
        &weights.lm_head,
        config.vocab_size,
        n,
    );

    // ── Sleep consolidation hook (Plan 154: eviction boundary) ─────
    // After the forward pass, if the KV cache is full, consolidate
    // cached K/V into GDN2 fast-weight state and evict. This frees
    // the cache for the next token while preserving context in S.
    #[cfg(feature = "sleep_consolidation")]
    if let (Some(gdn2), Some(sconf)) = (gdn2_cache, sleep_config)
        && sconf.should_sleep(pos)
    {
        crate::sleep::sleep(ctx, weights, cache, gdn2, sconf, config);
    }

    &mut ctx.logits
}

/// Issue 698 T6 — BLAKE3-seeded Gaussian state noise (zero-allocation).
///
/// Deterministic per `(pos, tau, len)`: the seed hashes ONLY `(pos, tau)`,
/// so the noise FIELD is identical across scales and calls; the amplitude
/// is `scale × rms(x)` computed on entry (relative noise — config- and
/// norm-independent). Box–Muller over BLAKE3-XOF uniform words, applied
/// in ascending index order (sequential f32 — same-platform bit-exact;
/// libm ulp drift off-platform, the T7 cross-platform caveat class).
#[cfg(feature = "loop_stability_fix")]
fn add_blake3_state_noise(x: &mut [f32], pos: usize, tau: usize, scale: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    // RMS before noise (post-norm RMS ≈ 1; computed for exactness).
    let mut ssq = 0.0f32;
    for &v in x.iter() {
        ssq += v * v;
    }
    let amp = scale * (ssq / n as f32).sqrt();

    let mut hasher = blake3::Hasher::new();
    hasher.update(&pos.to_le_bytes());
    hasher.update(&tau.to_le_bytes());
    let mut xof = hasher.finalize_xof();
    let mut bytes = [0u8; 4];
    let mut next_u = || {
        xof.fill(&mut bytes);
        // 24-bit uniform in [0, 1): low 24 bits scaled by 2^-24 (exact in
        // f32 — every value < 2^24 is exactly representable).
        let bits = u32::from_le_bytes(bytes) & 0x00FF_FFFF;
        let u = (bits as f32) * (1.0 / 16777216.0);
        // Box–Muller takes ln(u1): clamp the measure-zero 0.0.
        if u == 0.0 {
            f32::EPSILON
        } else {
            u
        }
    };

    let n2 = n & !1; // largest even ≤ n — full Box–Muller pairs
    let mut i = 0;
    while i < n2 {
        let u1 = next_u();
        let u2 = next_u();
        let r = (-2.0 * u1.ln()).sqrt();
        let ang = std::f32::consts::TAU * u2;
        x[i] += amp * (r * ang.cos());
        x[i + 1] += amp * (r * ang.sin());
        i += 2;
    }
    if n2 < n {
        // Odd tail: one extra Gaussian draw, cos branch only.
        let u1 = next_u();
        let u2 = next_u();
        let r = (-2.0 * u1.ln()).sqrt();
        x[n2] += amp * (r * (std::f32::consts::TAU * u2).cos());
    }
}

#[cfg(all(test, feature = "loop_stability_fix"))]
mod issue698_state_noise_tests {
    use super::add_blake3_state_noise;

    #[test]
    fn deterministic_same_seed_same_field() {
        let mut a = vec![0.5f32; 16];
        let mut b = vec![0.5f32; 16];
        add_blake3_state_noise(&mut a, 3, 7, 0.1);
        add_blake3_state_noise(&mut b, 3, 7, 0.1);
        assert_eq!(a, b);
        // Same (pos, tau) at a DIFFERENT scale: same direction field, larger
        // amplitude (the seed does not depend on scale — the relative-noise
        // contract).
        let mut c = vec![0.5f32; 16];
        add_blake3_state_noise(&mut c, 3, 7, 0.2);
        for i in 0..16 {
            let da = a[i] - 0.5;
            let dc = c[i] - 0.5;
            assert!((dc - 2.0 * da).abs() < 1e-5 * dc.abs().max(1e-6));
        }
    }

    #[test]
    fn field_varies_with_pos_and_tau() {
        // Nonzero base state (relative noise scales by rms(x) — an all-zero
        // state legitimately receives zero noise).
        let base = |v: f32| (0..16).map(|i| v + i as f32 * 0.125).collect::<Vec<_>>();
        let mut a = base(1.0);
        add_blake3_state_noise(&mut a, 0, 1, 0.5);
        let mut b = base(1.0);
        add_blake3_state_noise(&mut b, 1, 1, 0.5);
        let mut c = base(1.0);
        add_blake3_state_noise(&mut c, 0, 2, 0.5);
        assert_ne!(a, b);
        assert_ne!(a, c);
        // The all-zero state stays exactly zero (amp = scale · rms = 0).
        let mut z = vec![0.0f32; 16];
        add_blake3_state_noise(&mut z, 5, 5, 1.0);
        assert!(z.iter().all(|&v| v.to_bits() == 0u32));
    }

    #[test]
    fn amplitude_tracks_rms_and_scale() {
        // rms(x) = 2.0 for all-2 input; scale 0.1 → per-element noise is
        // bounded by ~6σ = 0.1·2·6 (never asserted near the bound; here we
        // assert the empirical rms of the ADDED field ≈ scale·rms within a
        // wide Gaussian band at n=4096).
        let n = 4096;
        let mut x = vec![2.0f32; n];
        add_blake3_state_noise(&mut x, 9, 4, 0.1);
        let mut ssq = 0.0f64;
        for &xi in x.iter() {
            let d = (xi - 2.0) as f64;
            ssq += d * d;
        }
        let noise_rms = (ssq / n as f64).sqrt();
        // Gaussian sample rms concentrates near σ = 0.2 with n = 4096
        // (relative sd of the rms estimate ≈ 1/√(2n) ≈ 1.1%).
        assert!(
            (noise_rms - 0.2).abs() < 0.2 * 0.1,
            "noise rms {noise_rms} not near expected 0.2"
        );
    }
}
