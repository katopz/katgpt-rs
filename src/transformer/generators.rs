use super::*;

/// Switches from reader LoRA to writer LoRA at the prefill→decode boundary.
/// Zero-copy: all buffers pre-allocated, no allocations in request path.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_prefill(
    ctx: &mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    config: &Config,
    rng: &mut crate::types::Rng,
    prompt_tokens: &[usize],
    max_gen_tokens: usize,
    lora_pair: &crate::types::LoraPair,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> Vec<usize> {
    // 1. Bidirectional prefill with reader LoRA
    let _ = {
        #[cfg(not(feature = "domain_latent"))]
        {
            forward_prefill(
                ctx,
                prefill,
                weights,
                cache,
                prompt_tokens,
                config,
                lora_pair.reader.as_ref(),
            )
        }
        #[cfg(feature = "domain_latent")]
        {
            forward_prefill(
                ctx,
                prefill,
                weights,
                cache,
                prompt_tokens,
                config,
                lora_pair.reader.as_ref(),
                domain_latent,
            )
        }
    };

    // 2. Sample first generation token from prefill output (fused softmax + sample).
    let mut token = ctx.sample_next_token(config.temperature, rng);

    let mut generated = Vec::with_capacity(max_gen_tokens);
    generated.push(token);

    // 3. Causal decode with writer LoRA
    for (pos, _) in (prompt_tokens.len()..).zip(1..max_gen_tokens) {
        if pos >= config.block_size {
            break;
        }

        let _ = {
            #[cfg(not(feature = "domain_latent"))]
            {
                forward_base(
                    ctx,
                    weights,
                    cache,
                    token,
                    pos,
                    config,
                    lora_pair.writer.as_ref(),
                )
            }
            #[cfg(feature = "domain_latent")]
            {
                forward_base(
                    ctx,
                    weights,
                    cache,
                    token,
                    pos,
                    config,
                    lora_pair.writer.as_ref(),
                    domain_latent,
                )
            }
        };
        token = ctx.sample_next_token(config.temperature, rng);
        generated.push(token);

        if token == config.bos_token {
            break;
        }
    }

    generated
}

/// Generate with prefill and optional domain latent (Plan 038).
/// Convenience wrapper for callers that need domain conditioning during generation.
#[cfg(feature = "domain_latent")]
#[allow(clippy::too_many_arguments)]
pub fn generate_with_prefill_and_domain_latent(
    ctx: &mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    config: &Config,
    rng: &mut crate::types::Rng,
    prompt_tokens: &[usize],
    max_gen_tokens: usize,
    lora_pair: &crate::types::LoraPair,
    domain_latent: Option<&crate::types::DomainLatent>,
) -> Vec<usize> {
    generate_with_prefill(
        ctx,
        prefill,
        weights,
        cache,
        config,
        rng,
        prompt_tokens,
        max_gen_tokens,
        lora_pair,
        domain_latent,
    )
}

/// Generate tokens with collapse-aware adaptive thinking (Plan 212 T4).
///
/// Extends [`generate_with_prefill`] with mid-reasoning collapse detection.
/// When the `CollapseDetector` detects degenerate reasoning (hesitation loops,
/// repetitive tokens), it forces an early exit from thinking mode and switches
/// to answer generation.
///
/// The `thinking_end_token` is the token ID that marks the boundary between
/// thinking and answering (e.g., the `</think|>` token). When collapse is
/// detected, this token is emitted to signal the model to switch modes.
///
/// When `detector` is `None`, behaves identically to [`generate_with_prefill`].
#[cfg(feature = "collapse_aware_thinking")]
#[allow(clippy::too_many_arguments)]
pub fn generate_with_collapse_detection(
    ctx: &mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    config: &Config,
    rng: &mut crate::types::Rng,
    prompt_tokens: &[usize],
    max_gen_tokens: usize,
    lora_pair: &crate::types::LoraPair,
    thinking_end_token: usize,
    detector: Option<&mut dyn katgpt_core::traits::CollapseDetector>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> Vec<usize> {
    use crate::pruners::collapse_detector::{CollapseAction, check_collapse_action};

    // No detector → fall back to standard generation.
    let Some(detector) = detector else {
        #[cfg(not(feature = "domain_latent"))]
        {
            return generate_with_prefill(
                ctx,
                prefill,
                weights,
                cache,
                config,
                rng,
                prompt_tokens,
                max_gen_tokens,
                lora_pair,
            );
        }
        #[cfg(feature = "domain_latent")]
        {
            return generate_with_prefill(
                ctx,
                prefill,
                weights,
                cache,
                config,
                rng,
                prompt_tokens,
                max_gen_tokens,
                lora_pair,
                domain_latent,
            );
        }
    };

    // 1. Prefill phase — same as generate_with_prefill
    let _ = {
        #[cfg(not(feature = "domain_latent"))]
        {
            forward_prefill(
                ctx,
                prefill,
                weights,
                cache,
                prompt_tokens,
                config,
                lora_pair.reader.as_ref(),
            )
        }
        #[cfg(feature = "domain_latent")]
        {
            forward_prefill(
                ctx,
                prefill,
                weights,
                cache,
                prompt_tokens,
                config,
                lora_pair.reader.as_ref(),
                domain_latent,
            )
        }
    };
    let mut token = ctx.sample_next_token(config.temperature, rng);

    let mut generated = Vec::with_capacity(max_gen_tokens);
    generated.push(token);
    let mut pos = prompt_tokens.len();
    let mut in_thinking = true;
    detector.reset();

    // 2. Decode loop with collapse detection
    for _ in 1..max_gen_tokens {
        if pos >= config.block_size {
            break;
        }

        // Check for collapse (only during thinking mode).
        if in_thinking {
            let action =
                check_collapse_action(detector, token as u32, pos - prompt_tokens.len(), true);
            if action == CollapseAction::ForceExit {
                token = thinking_end_token;
                generated.push(token);
                pos += 1;
                in_thinking = false;
                detector.reset();
                continue;
            }
        }

        // Check if we naturally exited thinking mode.
        if in_thinking && token == thinking_end_token {
            in_thinking = false;
            detector.reset();
        }

        // Forward pass.
        let _ = {
            #[cfg(not(feature = "domain_latent"))]
            {
                forward_base(
                    ctx,
                    weights,
                    cache,
                    token,
                    pos,
                    config,
                    lora_pair.writer.as_ref(),
                )
            }
            #[cfg(feature = "domain_latent")]
            {
                forward_base(
                    ctx,
                    weights,
                    cache,
                    token,
                    pos,
                    config,
                    lora_pair.writer.as_ref(),
                    domain_latent,
                )
            }
        };
        token = ctx.sample_next_token(config.temperature, rng);
        generated.push(token);
        pos += 1;

        if token == config.bos_token {
            break;
        }
    }

    generated
}

/// Zero-alloc generation: `ctx`, `cache`, `tokens` all provided by caller.
///
/// `tokens` is cleared and filled with generated token ids.
/// `ctx` and `cache` are reused across calls.
pub fn generate_into(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerKVCache,
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut Rng,
    n_tokens: usize,
    tokens: &mut Vec<usize>,
) {
    tokens.clear();
    let mut token = config.bos_token;
    let mut pos = 0;

    for _ in 0..n_tokens {
        if pos >= config.block_size {
            cache.reset();
            pos = 0;
            token = config.bos_token;
        }

        {
            let _ = forward(ctx, weights, cache, token, pos, config);
        }

        let next_token = ctx.sample_next_token(config.temperature, rng);
        tokens.push(next_token);

        if next_token == config.bos_token {
            cache.reset();
            pos = 0;
            token = config.bos_token;
        } else {
            token = next_token;
            pos += 1;
        }
    }
}

/// Generate tokens autoregressively. Returns generated token ids.
pub fn generate(
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut Rng,
    n_tokens: usize,
) -> Vec<usize> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut tokens = Vec::with_capacity(n_tokens);
    generate_into(
        &mut ctx,
        &mut cache,
        weights,
        config,
        rng,
        n_tokens,
        &mut tokens,
    );
    tokens
}

/// Generate multiple samples in parallel using rayon.
///
/// Each sample gets its own `ForwardContext` + `MultiLayerKVCache` via `map_init`,
/// so there's no contention. The `seeds` slice provides one seed per sample.
/// Returns `Vec<Vec<usize>>` with one token sequence per sample.
pub fn generate_batch(
    weights: &TransformerWeights,
    config: &Config,
    seeds: &[u64],
    n_tokens: usize,
) -> Vec<Vec<usize>> {
    seeds
        .par_iter()
        .map_init(
            || (ForwardContext::new(config), MultiLayerKVCache::new(config)),
            |(ctx, cache), &seed| {
                let mut rng = Rng::new(seed);
                let mut tokens = Vec::with_capacity(n_tokens);
                generate_into(ctx, cache, weights, config, &mut rng, n_tokens, &mut tokens);
                tokens
            },
        )
        .collect()
}
