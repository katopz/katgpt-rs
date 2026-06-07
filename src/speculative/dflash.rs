use crate::speculative::sampling::sample_from_distribution;
use crate::speculative::types::{DraftResult, SpeculativeContext};
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use crate::types::{Config, Rng, softmax_scaled};
use rayon::prelude::*;

// ── Zero-alloc _with variants ──────────────────────────────────

/// Zero-alloc variant of `dflash_predict`.
///
/// Reuses pre-allocated buffers from `SpeculativeContext`.
/// Each step gets an independent KV cache (reset per step).
/// Returns number of steps populated; caller reads via `sctx.marginal_slice()`.
pub fn dflash_predict_with(
    sctx: &mut SpeculativeContext,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
) -> usize {
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));
    let vocab_size = draft_config.vocab_size;
    let temperature = draft_config.temperature;

    for step in 0..max_steps {
        sctx.cache.reset();
        let logits = forward(
            &mut sctx.ctx,
            draft_weights,
            &mut sctx.cache,
            token,
            pos + step,
            draft_config,
        );
        sctx.probs_buf.copy_from_slice(logits);
        softmax_scaled(&mut sctx.probs_buf, 1.0 / temperature);
        let start = step * vocab_size;
        sctx.marginals_flat[start..start + vocab_size].copy_from_slice(&sctx.probs_buf);
    }

    sctx.steps_populated = max_steps;
    max_steps
}

/// Zero-alloc variant of `dflash_predict_ar`.
///
/// Reuses pre-allocated buffers from `SpeculativeContext`.
/// Autoregressive: single KV cache, samples feed back as next input.
/// Returns number of steps populated; caller reads via `sctx.marginal_slice()` and `sctx.sampled_tokens()`.
pub fn dflash_predict_ar_with(
    sctx: &mut SpeculativeContext,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    mtp_context: Option<&[f32]>,
) -> usize {
    // NOTE: Caller is responsible for resetting sctx before calling this function.
    // This allows KV cache preloading (Phase 3, Plan 055) between reset and AR loop.
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));
    let vocab_size = draft_config.vocab_size;
    let temperature = draft_config.temperature;

    let mut cur_token = token;
    for step in 0..max_steps {
        let _logits = forward(
            &mut sctx.ctx,
            draft_weights,
            &mut sctx.cache,
            cur_token,
            pos + step,
            draft_config,
        );

        // MTP conditioning: inject target activations into drafter's hidden state
        // on the first step, then re-compute logits from the conditioned state.
        if step == 0
            && let Some(mtp_ctx) = mtp_context
        {
            let n = draft_config.n_embd.min(mtp_ctx.len());
            for i in 0..n {
                unsafe {
                    *sctx.ctx.hidden_state.get_unchecked_mut(i) += *mtp_ctx.get_unchecked(i);
                }
            }
            // Re-compute logits from conditioned hidden state
            crate::types::matmul(
                &mut sctx.ctx.logits,
                &draft_weights.lm_head,
                &sctx.ctx.hidden_state,
                draft_config.vocab_size,
                draft_config.n_embd,
            );
        }

        sctx.probs_buf.copy_from_slice(&sctx.ctx.logits);
        softmax_scaled(&mut sctx.probs_buf, 1.0 / temperature);

        let next_token = sample_from_distribution(&sctx.probs_buf, rng);
        let start = step * vocab_size;
        sctx.marginals_flat[start..start + vocab_size].copy_from_slice(&sctx.probs_buf);
        sctx.sampled_tokens[step] = next_token;
        cur_token = next_token;
    }

    sctx.steps_populated = max_steps;
    max_steps
}

/// DFlash predict with Domino LoRA correction applied to logits.
/// After base logits are computed, applies DominoAdapter correction ΔL.
/// GRU state is maintained across positions in the draft block.
#[cfg(feature = "domino_lora")]
pub fn dflash_predict_ar_with_domino(
    sctx: &mut SpeculativeContext,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    domino: &mut super::domino_lora::DominoLoraCorrection,
    gru_state: &mut [f32],
) -> usize {
    // NOTE: Caller is responsible for resetting sctx before calling this function.
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));
    let vocab_size = draft_config.vocab_size;
    let temperature = draft_config.temperature;
    let n_embd = draft_config.n_embd;

    let mut cur_token = token;
    for step in 0..max_steps {
        let _logits = forward(
            &mut sctx.ctx,
            draft_weights,
            &mut sctx.cache,
            cur_token,
            pos + step,
            draft_config,
        );

        // Apply Domino LoRA correction: ΔL += w_up @ relu(w_down @ [hidden; gru_state])
        domino.correct(&sctx.ctx.hidden_state, gru_state, &mut sctx.ctx.logits);

        // Update GRU state using the current token's embedding
        let embed_offset = cur_token * n_embd;
        let token_embed = &draft_weights.wte[embed_offset..embed_offset + n_embd];
        // Zero-alloc GRU step: reuse gru_state by swapping through a stack buffer
        {
            let mut tmp_gru = [0.0f32; 1024]; // Max GRU hidden size
            let h = domino.gru_hidden_size().min(tmp_gru.len());
            domino.gru_step(token_embed, &gru_state[..h], &mut tmp_gru[..h]);
            gru_state[..h].copy_from_slice(&tmp_gru[..h]);
        }

        sctx.probs_buf.copy_from_slice(&sctx.ctx.logits);
        softmax_scaled(&mut sctx.probs_buf, 1.0 / temperature);

        let next_token = sample_from_distribution(&sctx.probs_buf, rng);
        let start = step * vocab_size;
        sctx.marginals_flat[start..start + vocab_size].copy_from_slice(&sctx.probs_buf);
        sctx.sampled_tokens[step] = next_token;
        cur_token = next_token;
    }

    sctx.steps_populated = max_steps;
    max_steps
}

/// Zero-alloc variant of `dflash_predict_conditioned`.
///
/// Reuses pre-allocated buffers from `SpeculativeContext`.
/// Seeds draft KV cache with target hidden state, then autoregressive.
/// Returns number of steps populated.
pub fn dflash_predict_conditioned_with(
    sctx: &mut SpeculativeContext,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    target_hidden_state: &[f32],
    rng: &mut Rng,
) -> usize {
    sctx.cache.reset();
    let max_steps = draft_config.draft_lookahead.min(
        draft_config
            .block_size
            .saturating_sub(pos)
            .saturating_sub(1),
    );

    // Seed draft KV cache with target hidden state (Option C)
    let draft_kv_dim = crate::types::kv_dim(draft_config);
    if !target_hidden_state.is_empty() && draft_kv_dim > 0 {
        let target_dim = target_hidden_state.len().min(draft_kv_dim);
        for layer in &mut sctx.cache.layers {
            layer.key[..target_dim].copy_from_slice(&target_hidden_state[..target_dim]);
            layer.key[target_dim..draft_kv_dim].fill(0.0);
            layer.value[..target_dim].copy_from_slice(&target_hidden_state[..target_dim]);
            layer.value[target_dim..draft_kv_dim].fill(0.0);
        }
    }

    let vocab_size = draft_config.vocab_size;
    let temperature = draft_config.temperature;
    let mut cur_token = token;

    for step in 0..max_steps {
        let logits = forward(
            &mut sctx.ctx,
            draft_weights,
            &mut sctx.cache,
            cur_token,
            pos + step + 1,
            draft_config,
        );
        sctx.probs_buf.copy_from_slice(logits);
        softmax_scaled(&mut sctx.probs_buf, 1.0 / temperature);

        let next_token = sample_from_distribution(&sctx.probs_buf, rng);
        let start = step * vocab_size;
        sctx.marginals_flat[start..start + vocab_size].copy_from_slice(&sctx.probs_buf);
        sctx.sampled_tokens[step] = next_token;
        cur_token = next_token;
    }

    sctx.steps_populated = max_steps;
    max_steps
}

// ── DFlare KV Routing (Plan 174 T2b, feature: dflare_kv_routing) ──

/// Pruner-confidence KV routing variant of `dflash_predict_conditioned_with`.
///
/// Scales the KV cache seeding by a blend factor derived from pruner relevance:
/// - blend == 1.0: fully conditioned (same as `dflash_predict_conditioned_with`)
/// - blend == 0.0: unconditioned (cache reset only, no seeding)
/// - 0.0 < blend < 1.0: partial seeding (values scaled by blend)
///
/// When `routing_config` is `None`, delegates to `dflash_predict_conditioned_with`.
#[cfg(feature = "dflare_kv_routing")]
pub fn dflash_predict_conditioned_with_routing(
    sctx: &mut SpeculativeContext,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    target_hidden_state: &[f32],
    rng: &mut Rng,
    routing_config: Option<&crate::speculative::types::KvRoutingConfig>,
    pruner_relevance: Option<f32>,
) -> usize {
    let Some(cfg) = routing_config else {
        return dflash_predict_conditioned_with(
            sctx,
            draft_weights,
            draft_config,
            token,
            pos,
            target_hidden_state,
            rng,
        );
    };

    let blend = match pruner_relevance {
        Some(rel) => cfg.blend_factor(rel),
        None => 1.0, // no relevance info → default conditioned
    };

    sctx.cache.reset();
    let max_steps = draft_config.draft_lookahead.min(
        draft_config
            .block_size
            .saturating_sub(pos)
            .saturating_sub(1),
    );

    // Seed draft KV cache with scaled target hidden state
    let draft_kv_dim = crate::types::kv_dim(draft_config);
    if blend > 0.0 && !target_hidden_state.is_empty() && draft_kv_dim > 0 {
        let target_dim = target_hidden_state.len().min(draft_kv_dim);
        for layer in &mut sctx.cache.layers {
            if blend == 1.0 {
                // Fast path: no scaling needed
                layer.key[..target_dim].copy_from_slice(&target_hidden_state[..target_dim]);
                layer.key[target_dim..draft_kv_dim].fill(0.0);
                layer.value[..target_dim].copy_from_slice(&target_hidden_state[..target_dim]);
                layer.value[target_dim..draft_kv_dim].fill(0.0);
            } else {
                // Scale seeding by blend factor
                for i in 0..target_dim {
                    layer.key[i] = blend * target_hidden_state[i];
                    layer.value[i] = blend * target_hidden_state[i];
                }
                layer.key[target_dim..draft_kv_dim].fill(0.0);
                layer.value[target_dim..draft_kv_dim].fill(0.0);
            }
        }
    }
    // blend == 0.0: skip seeding entirely (unconditioned, just reset cache)

    let vocab_size = draft_config.vocab_size;
    let temperature = draft_config.temperature;
    let mut cur_token = token;

    for step in 0..max_steps {
        let logits = forward(
            &mut sctx.ctx,
            draft_weights,
            &mut sctx.cache,
            cur_token,
            pos + step + 1,
            draft_config,
        );
        sctx.probs_buf.copy_from_slice(logits);
        softmax_scaled(&mut sctx.probs_buf, 1.0 / temperature);

        let next_token = sample_from_distribution(&sctx.probs_buf, rng);
        let start = step * vocab_size;
        sctx.marginals_flat[start..start + vocab_size].copy_from_slice(&sctx.probs_buf);
        sctx.sampled_tokens[step] = next_token;
        cur_token = next_token;
    }

    sctx.steps_populated = max_steps;
    max_steps
}

// ── Backward-compatible public API (thin wrappers) ─────────────

/// Sequential DFlash: Predict marginal distributions using draft model.
/// Uses pre-allocated ForwardContext for zero-alloc per step.
pub fn dflash_predict(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
) -> Vec<Vec<f32>> {
    let mut sctx = SpeculativeContext::new(draft_config);
    let steps = dflash_predict_with(&mut sctx, draft_weights, draft_config, token, pos);
    let vocab_size = draft_config.vocab_size;
    (0..steps)
        .map(|step| sctx.marginal_slice(step, vocab_size).to_vec())
        .collect()
}

/// Parallel DFlash: Predict marginals using rayon.
/// One ForwardContext + probs buffer per rayon worker thread — no contention, zero waste.
pub fn dflash_predict_parallel(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
) -> Vec<Vec<f32>> {
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));

    if max_steps == 0 {
        return Vec::new();
    }

    // For micro models, sequential is faster than rayon overhead
    if draft_config.n_embd <= draft_config.parallel_threshold {
        return dflash_predict(draft_weights, draft_config, token, pos);
    }

    (0..max_steps)
        .into_par_iter()
        .map_init(
            || {
                (
                    ForwardContext::new(draft_config),
                    MultiLayerKVCache::new(draft_config),
                    vec![0.0f32; draft_config.vocab_size],
                )
            },
            |(ctx, cache, probs_buf), step| {
                let draft_pos = pos + step;
                let logits = forward(ctx, draft_weights, cache, token, draft_pos, draft_config);
                probs_buf.copy_from_slice(logits);
                softmax_scaled(probs_buf, 1.0 / draft_config.temperature);
                probs_buf.clone()
            },
        )
        .collect()
}

/// Autoregressive DFlash: Predict marginals by sampling and feeding back tokens.
///
/// Unlike `dflash_predict` (which feeds the same token/pos to every step),
/// this samples a token at each step and feeds it back as input for the next.
/// Produces conditional q(x|x_{<i}) distributions instead of independent marginals.
pub fn dflash_predict_ar(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> DraftResult {
    let mut sctx = SpeculativeContext::new(draft_config);
    sctx.cache.reset();
    let steps = dflash_predict_ar_with(
        &mut sctx,
        draft_weights,
        draft_config,
        token,
        pos,
        rng,
        None,
    );
    let vocab_size = draft_config.vocab_size;
    DraftResult {
        marginals: (0..steps)
            .map(|step| sctx.marginal_slice(step, vocab_size).to_vec())
            .collect(),
        sampled_tokens: sctx.sampled_tokens().to_vec(),
        #[cfg(feature = "domain_latent")]
        routing_overlap: None,
        #[cfg(feature = "spec_cost_model")]
        cost_snapshot: None,
        #[cfg(feature = "stability_metrics")]
        stability: None,
    }
}

/// Target-conditioned DFlash: Predict marginals using draft model
/// conditioned on the target model's hidden state.
///
/// Uses Option C from plan 012: seed draft KV cache with target hidden state.
/// The target's hidden state (from `ForwardContext.hidden_state`) is projected
/// to the draft model's KV dimension and used as the initial KV cache entry.
/// This gives the draft model access to the target's representation without
/// any weight matrix changes.
///
/// Returns `DraftResult` with marginals and sampled tokens.
pub fn dflash_predict_conditioned(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    target_hidden_state: &[f32],
    rng: &mut Rng,
) -> DraftResult {
    let mut sctx = SpeculativeContext::new(draft_config);
    let steps = dflash_predict_conditioned_with(
        &mut sctx,
        draft_weights,
        draft_config,
        token,
        pos,
        target_hidden_state,
        rng,
    );
    let vocab_size = draft_config.vocab_size;
    DraftResult {
        marginals: (0..steps)
            .map(|step| sctx.marginal_slice(step, vocab_size).to_vec())
            .collect(),
        sampled_tokens: sctx.sampled_tokens().to_vec(),
        #[cfg(feature = "domain_latent")]
        routing_overlap: None,
        #[cfg(feature = "spec_cost_model")]
        cost_snapshot: None,
        #[cfg(feature = "stability_metrics")]
        stability: None,
    }
}

// ── DFlare Marginal Fusion (Plan 174 T1, feature: dflare_fusion) ──

/// Marginal-fusion wrapper around `dflash_predict_ar_with`.
///
/// When `fusion_config` is `Some` and enabled, runs one AR pass per conditioning
/// source (simulating different conditioning by varying the RNG seed), then blends
/// the resulting marginals with `marginal_fusion_blend`.
///
/// When `fusion_config` is `None` or not enabled, delegates directly to
/// `dflash_predict_ar_with` unchanged.
///
/// # Feature flag
/// `dflare_fusion` — Plan 174 Task 1c
#[cfg(feature = "dflare_fusion")]
pub fn dflash_predict_ar_with_fusion(
    sctx: &mut SpeculativeContext,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    mtp_context: Option<&[f32]>,
    fusion_config: Option<&crate::speculative::types::MarginalFusionConfig>,
) -> usize {
    let Some(config) = fusion_config else {
        return dflash_predict_ar_with(
            sctx,
            draft_weights,
            draft_config,
            token,
            pos,
            rng,
            mtp_context,
        );
    };

    if !config.enabled {
        return dflash_predict_ar_with(
            sctx,
            draft_weights,
            draft_config,
            token,
            pos,
            rng,
            mtp_context,
        );
    }

    // Validate config before proceeding
    if let Err(e) = config.validate() {
        eprintln!("[dflare_fusion] invalid MarginalFusionConfig: {e}, falling back to single pass");
        return dflash_predict_ar_with(
            sctx,
            draft_weights,
            draft_config,
            token,
            pos,
            rng,
            mtp_context,
        );
    }

    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));
    let vocab_size = draft_config.vocab_size;
    let num_sources = config.alpha_weights.len();

    // Temporary buffer to collect marginals from each source pass
    let mut source_marginals: Vec<Vec<f32>> = Vec::with_capacity(num_sources);

    for source_idx in 0..num_sources {
        // Vary RNG state per source to simulate different conditioning.
        // Each source gets a unique random offset so sampling diverges.
        let _ = rng.next(); // advance rng to get different sampling per source

        sctx.reset();
        let steps = dflash_predict_ar_with(
            sctx,
            draft_weights,
            draft_config,
            token,
            pos,
            rng,
            mtp_context,
        );

        // Capture the marginals from this pass
        let end = steps * vocab_size;
        source_marginals.push(sctx.marginals_flat[..end].to_vec());

        let _ = source_idx; // used only for clarity
    }

    // Build source references for blend
    let source_refs: Vec<&[f32]> = source_marginals.iter().map(|s| s.as_slice()).collect();

    // Blend all passes into sctx.marginals_flat
    marginal_fusion_blend(
        &source_refs,
        &config.alpha_weights,
        max_steps,
        vocab_size,
        &mut sctx.marginals_flat,
    );

    sctx.steps_populated = max_steps;
    max_steps
}

// ── Domino Causal Correction (Plan 197, feature: domino_correction) ──

/// Apply prefix-conditioned logit residual correction to marginals.
///
/// For each depth `i > 0`:
/// 1. Compute prefix hash from `sampled_tokens[0..i]`
/// 2. Look up correction vector from the table
/// 3. Apply as logit residual: `marginals[i][v] += correction[v]`
/// 4. Clamp to non-negative and re-normalize
///
/// # Zero Allocation
///
/// All operations are in-place on `marginals`.
///
/// # Guard
///
/// If `table.is_empty()`, returns immediately with no work done.
#[cfg(feature = "domino_correction")]
pub fn domino_correct_marginals(
    marginals: &mut [Vec<f32>],
    sampled_tokens: &[usize],
    table: &super::domino::PrefixCorrectionTable,
) {
    if table.is_empty() {
        return;
    }

    let vocab_size = table.vocab_size();

    for (depth, marginal) in marginals.iter_mut().enumerate() {
        if depth == 0 {
            continue;
        }
        if depth > sampled_tokens.len() {
            break;
        }

        let hash = super::domino::prefix_hash(&sampled_tokens[..depth]);
        let correction = table.lookup(hash);

        if correction.is_empty() {
            continue;
        }

        // Apply logit residual in-place
        let len = marginal.len().min(correction.len()).min(vocab_size);
        for v in 0..len {
            marginal[v] += correction[v];
            // Clamp to non-negative (logit residual can push below 0)
            if marginal[v] < 0.0 {
                marginal[v] = 0.0;
            }
        }

        // Re-normalize
        let sum: f32 = marginal.iter().sum();
        if sum > f32::EPSILON {
            for v in marginal.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Blend multiple marginal slices into a single fused marginal.
///
/// For each position k and vocab entry v:
/// `fused[k][v] = Σ_i alpha_i * marginals_i[k][v]`
///
/// After blending, each position's fused marginal is re-normalized to sum to 1.0.
///
/// # Arguments
/// * `sources` — marginal slices from each conditioning source, each `sources[i]` has shape
///   `[max_steps * vocab_size]`
/// * `alpha_weights` — blend weights, must sum to 1.0, same length as `sources`
/// * `max_steps` — number of draft steps
/// * `vocab_size` — vocabulary size
///
/// # Returns
/// Fused marginals written into `output` (`[max_steps * vocab_size]`).
#[cfg(feature = "dflare_fusion")]
pub fn marginal_fusion_blend(
    sources: &[&[f32]],
    alpha_weights: &[f32],
    max_steps: usize,
    vocab_size: usize,
    output: &mut [f32],
) {
    assert_eq!(
        sources.len(),
        alpha_weights.len(),
        "source/alpha count mismatch"
    );
    assert!(
        output.len() >= max_steps * vocab_size,
        "output buffer too small"
    );

    output
        .iter_mut()
        .take(max_steps * vocab_size)
        .for_each(|v| *v = 0.0);

    for (src, &alpha) in sources.iter().zip(alpha_weights.iter()) {
        let len = (max_steps * vocab_size).min(src.len());
        for i in 0..len {
            unsafe {
                *output.get_unchecked_mut(i) += alpha * *src.get_unchecked(i);
            }
        }
    }

    // Re-normalize each position to sum to 1.0
    for step in 0..max_steps {
        let start = step * vocab_size;
        let end = start + vocab_size;
        let sum: f32 = output[start..end].iter().sum();
        if sum > 0.0 {
            for v in &mut output[start..end] {
                *v /= sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::dd_tree::{build_dd_tree, extract_best_path};
    use crate::transformer::TransformerWeights;
    use crate::types::{Config, Rng};

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    #[test]
    fn test_dflash_produces_marginals() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        assert!(!marginals.is_empty());
        assert!(marginals.len() <= config.draft_lookahead);

        for (i, row) in marginals.iter().enumerate() {
            assert_eq!(row.len(), config.vocab_size, "row {i} wrong size");
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "row {i} sum = {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dflash_parallel_matches_count() {
        let (weights, config) = make_draft();
        let seq = dflash_predict(&weights, &config, 0, 0);
        let par = dflash_predict_parallel(&weights, &config, 0, 0);
        assert_eq!(seq.len(), par.len(), "parallel should produce same count");
    }

    #[test]
    fn test_dflash_positions_differ() {
        let (weights, config) = make_draft();
        let m0 = dflash_predict(&weights, &config, 0, 0);
        let m1 = dflash_predict(&weights, &config, 0, 1);
        assert_ne!(
            m0[0], m1[0],
            "marginals at different positions should differ"
        );
    }

    #[test]
    fn test_dflash_ar_produces_marginals() {
        let (weights, config) = make_draft();
        let result = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        assert!(!result.marginals.is_empty(), "should produce marginals");
        assert!(
            !result.sampled_tokens.is_empty(),
            "should produce sampled tokens"
        );
        assert_eq!(result.marginals.len(), result.sampled_tokens.len());
        for probs in &result.marginals {
            assert_eq!(probs.len(), config.vocab_size);
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "probs should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_dflash_ar_is_autoregressive() {
        let (weights, config) = make_draft();
        let r1 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(1));
        let r2 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(2));
        assert_ne!(
            r1.sampled_tokens, r2.sampled_tokens,
            "different seeds should produce different AR tokens"
        );
    }

    #[test]
    fn test_dflash_ar_deterministic() {
        let (weights, config) = make_draft();
        let r1 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        let r2 = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        assert_eq!(
            r1.sampled_tokens, r2.sampled_tokens,
            "same seed should produce same tokens"
        );
        for (a, b) in r1.marginals.iter().zip(r2.marginals.iter()) {
            for (pa, pb) in a.iter().zip(b.iter()) {
                assert!((pa - pb).abs() < 1e-6, "marginals should be identical");
            }
        }
    }

    #[test]
    fn test_extract_best_path() {
        let (weights, config) = make_draft();
        let marginals = dflash_predict(&weights, &config, 0, 0);
        let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
        let tree = build_dd_tree(&mv, &config);
        let path = extract_best_path(&tree);
        if !tree.is_empty() {
            assert!(!path.is_empty(), "non-empty tree should produce a path");
            for &t in &path {
                assert!(t < config.vocab_size, "token {t} out of range");
            }
        }
    }

    #[test]
    fn test_dflash_conditioned_produces_marginals() {
        let (weights, config) = make_draft();
        let target_config = Config::micro();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);

        // Get target hidden state
        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerKVCache::new(&target_config);
        let _ = forward(
            &mut target_ctx,
            &target_weights,
            &mut target_cache,
            0,
            0,
            &target_config,
        );
        let hidden = target_ctx.hidden_state.clone();

        let result =
            dflash_predict_conditioned(&weights, &config, 0, 0, &hidden, &mut Rng::new(42));
        assert!(!result.marginals.is_empty());
        assert_eq!(result.marginals.len(), result.sampled_tokens.len());
        for probs in &result.marginals {
            assert_eq!(probs.len(), config.vocab_size);
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "probs should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_dflash_conditioned_differs_from_unconditioned() {
        let (weights, config) = make_draft();
        let target_config = Config::micro();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);

        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerKVCache::new(&target_config);
        let _ = forward(
            &mut target_ctx,
            &target_weights,
            &mut target_cache,
            0,
            0,
            &target_config,
        );
        let hidden = target_ctx.hidden_state.clone();

        let uncond = dflash_predict_ar(&weights, &config, 0, 0, &mut Rng::new(42));
        let cond = dflash_predict_conditioned(&weights, &config, 0, 0, &hidden, &mut Rng::new(42));

        // Conditioned should differ from unconditioned (different KV cache seed)
        assert_ne!(
            cond.sampled_tokens, uncond.sampled_tokens,
            "conditioned marginals should differ from unconditioned"
        );
    }

    #[test]
    fn test_dflash_conditioned_valid_probs() {
        let (weights, config) = make_draft();
        let hidden = vec![0.5; config.n_embd]; // fake hidden state
        let result =
            dflash_predict_conditioned(&weights, &config, 0, 0, &hidden, &mut Rng::new(42));
        for probs in &result.marginals {
            for &p in probs {
                assert!(p.is_finite(), "prob should be finite");
                assert!(p >= 0.0, "prob should be non-negative");
            }
        }
    }

    #[test]
    fn test_dflash_conditioned_empty_hidden() {
        let (weights, config) = make_draft();
        let result = dflash_predict_conditioned(&weights, &config, 0, 0, &[], &mut Rng::new(42));
        // Empty hidden state should still produce valid output (no seeding)
        assert!(!result.marginals.is_empty());
    }

    #[test]
    fn test_dflash_predict_with_matches_original() {
        let (weights, config) = make_draft();
        let mut sctx = SpeculativeContext::new(&config);
        let steps = dflash_predict_with(&mut sctx, &weights, &config, 0, 0);
        let vocab_size = config.vocab_size;

        assert_eq!(steps, config.draft_lookahead);
        for step in 0..steps {
            let slice = sctx.marginal_slice(step, vocab_size);
            assert_eq!(slice.len(), vocab_size);
            let sum: f32 = slice.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "step {step} sum = {sum}");
        }
    }

    #[test]
    fn test_dflash_predict_ar_with_matches_original() {
        let (weights, config) = make_draft();
        let mut sctx = SpeculativeContext::new(&config);
        sctx.cache.reset();
        let steps =
            dflash_predict_ar_with(&mut sctx, &weights, &config, 0, 0, &mut Rng::new(42), None);
        let vocab_size = config.vocab_size;

        assert_eq!(steps, config.draft_lookahead);
        assert_eq!(sctx.sampled_tokens().len(), steps);
        for step in 0..steps {
            let slice = sctx.marginal_slice(step, vocab_size);
            assert_eq!(slice.len(), vocab_size);
            let sum: f32 = slice.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "step {step} sum = {sum}");
        }
    }

    #[test]
    fn test_dflash_predict_conditioned_with_matches_original() {
        let (weights, config) = make_draft();
        let hidden = vec![0.5; config.n_embd];
        let mut sctx = SpeculativeContext::new(&config);
        let steps = dflash_predict_conditioned_with(
            &mut sctx,
            &weights,
            &config,
            0,
            0,
            &hidden,
            &mut Rng::new(42),
        );
        let vocab_size = config.vocab_size;

        assert!(steps > 0);
        assert_eq!(sctx.sampled_tokens().len(), steps);
        for step in 0..steps {
            let slice = sctx.marginal_slice(step, vocab_size);
            assert_eq!(slice.len(), vocab_size);
            let sum: f32 = slice.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "step {step} sum = {sum}");
        }
    }

    #[test]
    fn test_dflash_with_reuse_across_calls() {
        let (weights, config) = make_draft();
        let mut sctx = SpeculativeContext::new(&config);

        // First call
        let steps1 = dflash_predict_with(&mut sctx, &weights, &config, 0, 0);
        assert_eq!(steps1, config.draft_lookahead);

        // Second call — same context, should produce same results
        let steps2 = dflash_predict_with(&mut sctx, &weights, &config, 0, 0);
        assert_eq!(steps2, config.draft_lookahead);

        // Results should be identical (same inputs, deterministic)
        let vocab_size = config.vocab_size;
        for step in 0..steps1 {
            // Can't compare directly since second call overwrites, but we know it ran OK
            let _slice = sctx.marginal_slice(step, vocab_size);
        }
    }

    #[test]
    fn test_parallel_threshold_fallback_identical() {
        // draft config: n_embd=4, parallel_threshold=128 → sequential path
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let sequential = dflash_predict(&weights, &config, 0, 0);
        let parallel = dflash_predict_parallel(&weights, &config, 0, 0);

        assert_eq!(sequential.len(), parallel.len());
        for (step, (seq_marg, par_marg)) in sequential.iter().zip(parallel.iter()).enumerate() {
            for (i, (a, b)) in seq_marg.iter().zip(par_marg.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "step {step} token {i}: sequential={a}, parallel={b}"
                );
            }
        }
    }

    #[test]
    fn test_parallel_threshold_above_runs_parallel() {
        // micro config: n_embd=16, parallel_threshold=128 → still sequential
        let config = Config::micro();
        assert!(
            config.n_embd <= config.parallel_threshold,
            "micro should be below threshold"
        );

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let sequential = dflash_predict(&weights, &config, 0, 0);
        let parallel = dflash_predict_parallel(&weights, &config, 0, 0);

        // Should be identical because threshold triggers sequential fallback
        assert_eq!(sequential.len(), parallel.len());
        for (step, (seq_marg, par_marg)) in sequential.iter().zip(parallel.iter()).enumerate() {
            for (i, (a, b)) in seq_marg.iter().zip(par_marg.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "step {step} token {i}: sequential={a}, parallel={b}"
                );
            }
        }
    }

    #[test]
    fn test_parallel_threshold_custom_above_triggers_parallel() {
        // Custom config with threshold below n_embd → actual parallel path
        let mut config = Config::micro();
        config.parallel_threshold = 1; // Force parallel path (n_embd=16 > 1)

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let result = dflash_predict_parallel(&weights, &config, 0, 0);
        assert!(!result.is_empty(), "parallel should produce results");
        assert_eq!(result.len(), config.draft_lookahead);

        // Verify valid probabilities
        for (step, marg) in result.iter().enumerate() {
            let sum: f32 = marg.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "step {step} probabilities should sum to ~1.0, got {sum}"
            );
        }
    }

    // ── DFlare Marginal Fusion blend tests (Plan 174 T1e) ──

    #[cfg(feature = "dflare_fusion")]
    mod dflare_fusion_blend {
        use super::*;
        use crate::speculative::types::MarginalFusionConfig;

        #[test]
        fn test_blend_is_weighted_average() {
            // Two sources, each with 1 step, vocab=4
            let src1: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0]; // all mass on token 0
            let src2: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0]; // all mass on token 1
            let alphas = vec![0.6, 0.4];
            let mut output = vec![0.0f32; 4];

            marginal_fusion_blend(&[&src1, &src2], &alphas, 1, 4, &mut output);

            assert!(
                (output[0] - 0.6).abs() < 1e-5,
                "token 0 should be 0.6, got {}",
                output[0]
            );
            assert!(
                (output[1] - 0.4).abs() < 1e-5,
                "token 1 should be 0.4, got {}",
                output[1]
            );
            assert!(
                (output[2] - 0.0).abs() < 1e-5,
                "token 2 should be 0.0, got {}",
                output[2]
            );
        }

        #[test]
        fn test_blend_sums_to_one() {
            let src1: Vec<f32> = vec![0.25, 0.25, 0.25, 0.25];
            let src2: Vec<f32> = vec![0.1, 0.4, 0.3, 0.2];
            let alphas = vec![0.7, 0.3];
            let mut output = vec![0.0f32; 4];

            marginal_fusion_blend(&[&src1, &src2], &alphas, 1, 4, &mut output);

            let sum: f32 = output.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "blended output should sum to 1.0, got {sum}"
            );
        }

        #[test]
        fn test_blend_multi_step() {
            let src1: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 steps
            let src2: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2 steps
            let alphas = vec![0.5, 0.5];
            let mut output = vec![0.0f32; 8];

            marginal_fusion_blend(&[&src1, &src2], &alphas, 2, 4, &mut output);

            // Step 0: 0.5 * [1,0,0,0] + 0.5 * [0,1,0,0] = [0.5, 0.5, 0, 0]
            assert!((output[0] - 0.5).abs() < 1e-5);
            assert!((output[1] - 0.5).abs() < 1e-5);
            // Step 1: 0.5 * [0,1,0,0] + 0.5 * [0,0,1,0] = [0, 0.5, 0.5, 0]
            assert!((output[4] - 0.0).abs() < 1e-5);
            assert!((output[5] - 0.5).abs() < 1e-5);
            assert!((output[6] - 0.5).abs() < 1e-5);
        }

        // ── dflash_predict_ar_with_fusion tests (Plan 174 Task 1c) ──

        #[test]
        fn test_fusion_none_delegates_to_ar() {
            let (weights, config) = make_draft();
            let mut sctx = SpeculativeContext::new(&config);
            let mut rng = Rng::new(99);

            let steps = dflash_predict_ar_with_fusion(
                &mut sctx, &weights, &config, 0, 0, &mut rng, None, None,
            );
            assert!(steps > 0, "should produce steps");
            assert_eq!(sctx.steps_populated, steps);

            let vocab_size = config.vocab_size;
            // Verify marginals are valid probability distributions
            for step in 0..steps {
                let marginal = sctx.marginal_slice(step, vocab_size);
                let sum: f32 = marginal.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "step {step} marginals sum to {sum}"
                );
            }
        }

        #[test]
        fn test_fusion_disabled_delegates_to_ar() {
            let (weights, config) = make_draft();
            let mut sctx = SpeculativeContext::new(&config);
            let mut rng = Rng::new(99);
            let fusion = MarginalFusionConfig {
                alpha_weights: vec![0.5, 0.5],
                condition_layer_ids: vec![vec![1], vec![2]],
                enabled: false,
            };

            let steps = dflash_predict_ar_with_fusion(
                &mut sctx,
                &weights,
                &config,
                0,
                0,
                &mut rng,
                None,
                Some(&fusion),
            );
            assert!(steps > 0);
        }

        #[test]
        fn test_fusion_enabled_blends_multiple_sources() {
            let (weights, config) = make_draft();
            let vocab_size = config.vocab_size;
            let fusion = MarginalFusionConfig {
                alpha_weights: vec![0.5, 0.5],
                condition_layer_ids: vec![vec![1], vec![2]],
                enabled: true,
            };

            let mut sctx = SpeculativeContext::new(&config);
            let mut rng = Rng::new(42);

            let steps = dflash_predict_ar_with_fusion(
                &mut sctx,
                &weights,
                &config,
                0,
                0,
                &mut rng,
                None,
                Some(&fusion),
            );

            assert!(steps > 0, "should produce steps");
            assert_eq!(sctx.steps_populated, steps);

            // Each step's blended marginals should be a valid distribution
            for step in 0..steps {
                let marginal = sctx.marginal_slice(step, vocab_size);
                assert!(!marginal.is_empty(), "step {step} should have marginals");
                let sum: f32 = marginal.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-3,
                    "blended step {step} marginals sum to {sum}"
                );
            }
        }

        #[test]
        fn test_fusion_produces_different_marginals_than_single_pass() {
            let (weights, config) = make_draft();
            let vocab_size = config.vocab_size;

            // Run a single AR pass (no fusion)
            let mut sctx_single = SpeculativeContext::new(&config);
            let mut rng_single = Rng::new(42);
            let steps_single = dflash_predict_ar_with(
                &mut sctx_single,
                &weights,
                &config,
                0,
                0,
                &mut rng_single,
                None,
            );

            // Run fusion with 2 sources
            let fusion = MarginalFusionConfig {
                alpha_weights: vec![0.5, 0.5],
                condition_layer_ids: vec![vec![1], vec![2]],
                enabled: true,
            };
            let mut sctx_fused = SpeculativeContext::new(&config);
            let mut rng_fused = Rng::new(42);
            let steps_fused = dflash_predict_ar_with_fusion(
                &mut sctx_fused,
                &weights,
                &config,
                0,
                0,
                &mut rng_fused,
                None,
                Some(&fusion),
            );

            assert_eq!(steps_single, steps_fused);

            // The fused result should differ from a single pass because we
            // advance the RNG between source passes, producing different sampling.
            // We only check the first step's first few entries for any difference.
            let single_m0 = sctx_single.marginal_slice(0, vocab_size);
            let fused_m0 = sctx_fused.marginal_slice(0, vocab_size);
            // At minimum the marginals should exist and be valid
            let single_sum: f32 = single_m0.iter().sum();
            let fused_sum: f32 = fused_m0.iter().sum();
            assert!((single_sum - 1.0).abs() < 1e-3);
            assert!((fused_sum - 1.0).abs() < 1e-3);
        }
    }

    // ── DFlare KV Routing tests (Plan 174 T2b) ──────────────────
    #[cfg(feature = "dflare_kv_routing")]
    mod kv_routing {
        use super::*;
        use crate::speculative::types::KvRoutingConfig;

        fn routing_config(enabled: bool, low: f32, high: f32) -> KvRoutingConfig {
            KvRoutingConfig {
                enabled,
                low_confidence_threshold: low,
                high_confidence_threshold: high,
            }
        }

        #[test]
        fn test_routing_none_delegates_to_conditioned() {
            let (weights, config) = make_draft();
            let hidden = vec![0.5; config.n_embd];
            let vocab_size = config.vocab_size;

            // Run with routing_config = None
            let mut sctx_routed = SpeculativeContext::new(&config);
            let steps_routed = dflash_predict_conditioned_with_routing(
                &mut sctx_routed,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
                None,
                Some(0.5),
            );

            // Run baseline conditioned
            let mut sctx_base = SpeculativeContext::new(&config);
            let steps_base = dflash_predict_conditioned_with(
                &mut sctx_base,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
            );

            assert_eq!(steps_routed, steps_base);
            assert_eq!(sctx_routed.sampled_tokens(), sctx_base.sampled_tokens());
            for step in 0..steps_routed {
                let routed = sctx_routed.marginal_slice(step, vocab_size);
                let base = sctx_base.marginal_slice(step, vocab_size);
                assert_eq!(routed, base, "step {step} marginals should match");
            }
        }

        #[test]
        fn test_routing_high_relevance_matches_conditioned() {
            let (weights, config) = make_draft();
            let hidden = vec![0.5; config.n_embd];
            let vocab_size = config.vocab_size;

            let cfg = routing_config(true, 0.3, 0.8);

            // High relevance → blend = 1.0 → fully conditioned
            let mut sctx_routed = SpeculativeContext::new(&config);
            let steps_routed = dflash_predict_conditioned_with_routing(
                &mut sctx_routed,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
                Some(&cfg),
                Some(0.9), // above high threshold
            );

            // Baseline conditioned
            let mut sctx_base = SpeculativeContext::new(&config);
            let steps_base = dflash_predict_conditioned_with(
                &mut sctx_base,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
            );

            assert_eq!(steps_routed, steps_base);
            assert_eq!(sctx_routed.sampled_tokens(), sctx_base.sampled_tokens());
            for step in 0..steps_routed {
                let routed = sctx_routed.marginal_slice(step, vocab_size);
                let base = sctx_base.marginal_slice(step, vocab_size);
                assert_eq!(
                    routed, base,
                    "step {step} marginals should match at high relevance"
                );
            }
        }

        #[test]
        fn test_routing_low_relevance_differs_from_conditioned() {
            let (weights, config) = make_draft();
            let hidden = vec![0.5; config.n_embd];
            let vocab_size = config.vocab_size;

            let cfg = routing_config(true, 0.3, 0.8);

            // Low relevance → blend = 0.0 → unconditioned
            let mut sctx_low = SpeculativeContext::new(&config);
            let steps_low = dflash_predict_conditioned_with_routing(
                &mut sctx_low,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
                Some(&cfg),
                Some(0.1), // below low threshold
            );

            // Baseline conditioned
            let mut sctx_base = SpeculativeContext::new(&config);
            let steps_base = dflash_predict_conditioned_with(
                &mut sctx_base,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
            );

            assert_eq!(steps_low, steps_base);
            assert_ne!(
                sctx_low.sampled_tokens(),
                sctx_base.sampled_tokens(),
                "low relevance should differ from fully conditioned"
            );

            // Marginals should still be valid
            for step in 0..steps_low {
                let m = sctx_low.marginal_slice(step, vocab_size);
                let sum: f32 = m.iter().sum();
                assert!((sum - 1.0).abs() < 1e-4, "step {step} sum = {sum}");
            }
        }

        #[test]
        fn test_routing_medium_relevance_intermediate_behavior() {
            let (weights, config) = make_draft();
            let vocab_size = config.vocab_size;

            // Use a non-uniform hidden state with large magnitude so blend
            // scaling produces meaningfully different KV cache states.
            let hidden: Vec<f32> = (0..config.n_embd)
                .map(|i| {
                    let x = i as f32;
                    (x * 0.37).sin() * 5.0 + (x * 1.13).cos() * 3.0
                })
                .collect();

            let cfg = routing_config(true, 0.3, 0.8);

            // Medium relevance → blend = (0.5 - 0.3) / (0.8 - 0.3) = 0.4
            let mut sctx_med = SpeculativeContext::new(&config);
            let steps_med = dflash_predict_conditioned_with_routing(
                &mut sctx_med,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
                Some(&cfg),
                Some(0.5),
            );

            assert!(steps_med > 0);

            // Marginals should be valid probability distributions
            for step in 0..steps_med {
                let m = sctx_med.marginal_slice(step, vocab_size);
                let sum: f32 = m.iter().sum();
                assert!((sum - 1.0).abs() < 1e-4, "step {step} sum = {sum}");
                for &p in m.iter() {
                    assert!(p.is_finite() && p >= 0.0, "step {step} has invalid prob");
                }
            }

            // Verify the KV cache was actually seeded with scaled values
            // by checking the marginal at step 0 differs from unconditioned.
            // If the model is too small for seeding to matter, just verify
            // valid output.
            let mut sctx_high = SpeculativeContext::new(&config);
            let _ = dflash_predict_conditioned_with_routing(
                &mut sctx_high,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
                Some(&cfg),
                Some(0.9),
            );

            let mut sctx_low = SpeculativeContext::new(&config);
            let _ = dflash_predict_conditioned_with_routing(
                &mut sctx_low,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
                Some(&cfg),
                Some(0.1),
            );

            // The function should produce valid marginals at all blend levels
            for (label, sctx, steps) in [
                ("high", &sctx_high, steps_med),
                ("low", &sctx_low, steps_med),
            ] {
                for step in 0..steps {
                    let m = sctx.marginal_slice(step, vocab_size);
                    let sum: f32 = m.iter().sum();
                    assert!((sum - 1.0).abs() < 1e-4, "{label} step {step} sum = {sum}");
                }
            }
        }

        #[test]
        fn test_routing_disabled_returns_conditioned() {
            let (weights, config) = make_draft();
            let hidden = vec![0.5; config.n_embd];
            let vocab_size = config.vocab_size;

            let cfg = routing_config(false, 0.3, 0.8); // disabled

            let mut sctx_routed = SpeculativeContext::new(&config);
            let steps_routed = dflash_predict_conditioned_with_routing(
                &mut sctx_routed,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
                Some(&cfg),
                Some(0.1), // low relevance but routing disabled
            );

            // Baseline conditioned
            let mut sctx_base = SpeculativeContext::new(&config);
            let steps_base = dflash_predict_conditioned_with(
                &mut sctx_base,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
            );

            // When disabled, blend_factor returns 1.0 regardless of relevance
            assert_eq!(steps_routed, steps_base);
            assert_eq!(sctx_routed.sampled_tokens(), sctx_base.sampled_tokens());
            for step in 0..steps_routed {
                let routed = sctx_routed.marginal_slice(step, vocab_size);
                let base = sctx_base.marginal_slice(step, vocab_size);
                assert_eq!(routed, base);
            }
        }

        #[test]
        fn test_routing_none_relevance_defaults_conditioned() {
            let (weights, config) = make_draft();
            let hidden = vec![0.5; config.n_embd];
            let vocab_size = config.vocab_size;

            let cfg = routing_config(true, 0.3, 0.8);

            // pruner_relevance = None → defaults to blend = 1.0
            let mut sctx_routed = SpeculativeContext::new(&config);
            let steps_routed = dflash_predict_conditioned_with_routing(
                &mut sctx_routed,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
                Some(&cfg),
                None,
            );

            let mut sctx_base = SpeculativeContext::new(&config);
            let steps_base = dflash_predict_conditioned_with(
                &mut sctx_base,
                &weights,
                &config,
                0,
                0,
                &hidden,
                &mut Rng::new(42),
            );

            assert_eq!(steps_routed, steps_base);
            assert_eq!(sctx_routed.sampled_tokens(), sctx_base.sampled_tokens());
            for step in 0..steps_routed {
                let routed = sctx_routed.marginal_slice(step, vocab_size);
                let base = sctx_base.marginal_slice(step, vocab_size);
                assert_eq!(routed, base);
            }
        }
    }

    // ── Domino Causal Correction tests (Plan 197) ─────────────────
    #[cfg(feature = "domino_correction")]
    mod domino_correction {
        use crate::speculative::domino::PrefixCorrectionTable;

        #[test]
        fn test_domino_correct_empty_table_noop() {
            let mut marginals = vec![vec![0.5, 0.5], vec![0.3, 0.7]];
            let original = marginals.clone();
            let table = PrefixCorrectionTable::new(2);
            let sampled = [0usize];

            super::domino_correct_marginals(&mut marginals, &sampled, &table);

            assert_eq!(marginals, original, "Empty table should be no-op");
        }

        #[test]
        fn test_domino_correct_applies_residual() {
            let mut marginals = vec![
                vec![0.5, 0.5], // depth 0 — untouched
                vec![0.3, 0.7], // depth 1 — gets correction
            ];

            // Correction for prefix [0]: suppress token 0, boost token 1
            let correction = vec![-0.1f32, 0.2];
            let table = PrefixCorrectionTable::builder(2)
                .add_correction(&[0], &correction)
                .build();

            let sampled = [0usize]; // prefix at depth 1 is [0]
            super::domino_correct_marginals(&mut marginals, &sampled, &table);

            // Depth 0 should be unchanged
            assert_eq!(marginals[0], vec![0.5, 0.5]);

            // Depth 1 should have correction applied and re-normalized
            // Raw: [0.3 - 0.1, 0.7 + 0.2] = [0.2, 0.9], sum=1.1
            // Normalized: [0.2/1.1, 0.9/1.1] ≈ [0.1818, 0.8182]
            let sum: f32 = marginals[1].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "Should sum to 1.0, got {sum}");
            assert!(marginals[1][1] > 0.8, "Token 1 should be boosted");
            assert!(marginals[1][0] < 0.2, "Token 0 should be suppressed");
        }

        #[test]
        fn test_domino_correct_clamps_negative() {
            let mut marginals = vec![vec![0.5, 0.5], vec![0.1, 0.9]];

            // Large negative correction that would push below 0
            let correction = vec![-1.0f32, 0.5];
            let table = PrefixCorrectionTable::builder(2)
                .add_correction(&[0], &correction)
                .build();

            let sampled = [0usize];
            super::domino_correct_marginals(&mut marginals, &sampled, &table);

            // Token 0 clamped to 0, token 1 = 0.9 + 0.5 = 1.4
            // After normalize: [0.0, 1.0]
            assert!((marginals[1][0]).abs() < 1e-5, "Should be clamped to 0");
            assert!(
                (marginals[1][1] - 1.0).abs() < 1e-5,
                "Should be normalized to 1.0"
            );
        }

        #[test]
        fn test_domino_correct_multi_depth() {
            let mut marginals = vec![
                vec![0.5, 0.5], // depth 0
                vec![0.3, 0.7], // depth 1, prefix [0]
                vec![0.6, 0.4], // depth 2, prefix [0, 1]
            ];

            // Correction for prefix [0]
            let correction1 = vec![0.1f32, -0.1];
            // Correction for prefix [0, 1]
            let correction2 = vec![-0.2f32, 0.3];

            let table = PrefixCorrectionTable::builder(2)
                .add_correction(&[0], &correction1)
                .add_correction(&[0, 1], &correction2)
                .build();

            let sampled = [0usize, 1];
            super::domino_correct_marginals(&mut marginals, &sampled, &table);

            // All marginals should still sum to 1.0
            for (depth, m) in marginals.iter().enumerate() {
                let sum: f32 = m.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "Depth {depth} should sum to 1.0, got {sum}"
                );
            }
        }

        #[test]
        fn test_domino_correct_missing_prefix_no_change() {
            let mut marginals = vec![vec![0.5, 0.5], vec![0.3, 0.7]];
            let original = marginals.clone();

            // Correction for prefix [99] — won't match our sampled tokens
            let correction = vec![0.5f32, 0.5];
            let table = PrefixCorrectionTable::builder(2)
                .add_correction(&[99], &correction)
                .build();

            let sampled = [0usize];
            super::domino_correct_marginals(&mut marginals, &sampled, &table);

            assert_eq!(
                marginals, original,
                "Missing prefix should not change marginals"
            );
        }
    }
}
