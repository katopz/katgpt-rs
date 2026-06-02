//! Sleep-time consolidation — N-pass recurrent consolidation over KV cache.
//!
//! When the KV cache fills, this module performs N offline recurrent passes
//! to consolidate the full cached context into GDN2 fast-weight state S.
//! After consolidation, the KV cache can be safely evicted.
//!
//! # Architecture
//!
//! ```text
//! Wake time:  input → single-pass → fill KV cache → output
//! Sleep time: KV cache full → N× consolidation_pass() → evict KV cache → continue
//! ```
//!
//! Each consolidation pass replays all cached K/V pairs through the GDN2 recurrent
//! step, strengthening the fast-weight state. This is the model-based analog of
//! AutoDreamer (Plan 107), applied to GDN2 fast weights instead of modelless logits.
//!
//! Plan 154: Sleep Consolidation — Offline Recursive Memory Consolidation at Eviction.

use super::eviction;
use super::types::SleepConfig;
use crate::gdn2::kernel::{gdn2_state_update, l2_normalize};
use crate::gdn2::types::MultiLayerGdn2Cache;
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::{self, Config};

/// Perform a single recurrent consolidation pass through all cached K/V pairs.
///
/// For each layer, replays all cached K/V pairs through the GDN2 recurrent step,
/// updating the fast-weight state S. This strengthens the consolidated memory
/// without modifying the KV cache itself.
///
/// # Arguments
/// * `kv_cache` — The filled KV cache to consolidate from
/// * `gdn2_cache` — GDN2 fast-weight state that will be updated
/// * `fill_pos` — Number of positions in the KV cache (0..fill_pos)
/// * `config` — Model configuration
/// * `k_normalized` — Pre-allocated scratch buffer (`head_dim` elements), reused across calls
pub fn consolidation_pass(
    kv_cache: &MultiLayerKVCache,
    gdn2_cache: &mut MultiLayerGdn2Cache,
    fill_pos: usize,
    config: &Config,
    k_normalized: &mut [f32],
) {
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);

    for (layer_idx, layer_cache) in kv_cache.layers.iter().enumerate() {
        let gdn2_layer = &mut gdn2_cache.layers[layer_idx];
        let gate_config = gdn2_layer.gate_config;

        // Replay all cached K/V pairs through the recurrent step
        for pos in 0..fill_pos {
            let pos_off = pos * kvd;

            for h in 0..config.n_kv_head {
                let kv_group = h; // In GQA, kv_group = h for kv_head indices
                let head_off = kv_group * hd;

                // Extract K, V from the KV cache
                let k_h = &layer_cache.key[pos_off + head_off..pos_off + head_off + hd];
                let v_h = &layer_cache.value[pos_off + head_off..pos_off + head_off + hd];

                // L2 normalize k into scratch buffer (stability requirement)
                k_normalized.copy_from_slice(k_h);
                l2_normalize(k_normalized);

                // Self-consolidation uses q = k, but consolidation only needs the
                // updated state S — the step-4 readout output is discarded. So we
                // call the state-update half (steps 1–3) and skip the readout matvec
                // entirely. This is bit-identical to the full recurrent step's effect
                // on S (verified by gdn2 split_functions_match_combined tests).
                let s = &mut gdn2_layer.heads[kv_group].s;

                gdn2_state_update(
                    s,
                    k_normalized,
                    v_h,
                    &gdn2_layer.decay_alpha,
                    &gdn2_layer.erase_b,
                    1.0, // scalar write weight
                    &gdn2_layer.write_w_channel,
                    &mut gdn2_layer.temp_buf,
                    &mut gdn2_layer.delta,
                    hd,
                    hd,
                    gate_config,
                );
            }
        }
    }
}

/// Perform sleep consolidation: N recurrent passes followed by eviction.
///
/// This is the main entry point for sleep-time consolidation. When the KV cache
/// is full, call this function to:
/// 1. Run `sleep_passes` consolidation passes over the cached content
/// 2. Evict the KV cache according to the configured strategy
///
/// # Arguments
/// * `ctx` — Forward context (unused in current impl, reserved for future layer-norm passes)
/// * `weights` — Transformer weights (unused in current impl, reserved for future)
/// * `kv_cache` — The KV cache to consolidate and evict
/// * `gdn2_cache` — GDN2 fast-weight state to consolidate into
/// * `sleep_config` — Sleep configuration (passes, eviction strategy)
/// * `config` — Model configuration
///
/// # Returns
/// The number of consolidation passes performed.
pub fn sleep(
    _ctx: &mut ForwardContext,
    _weights: &TransformerWeights,
    kv_cache: &mut MultiLayerKVCache,
    gdn2_cache: &mut MultiLayerGdn2Cache,
    sleep_config: &SleepConfig,
    config: &Config,
) -> usize {
    let fill_pos = kv_cache.fill_pos();

    if fill_pos == 0 {
        return 0; // Nothing to consolidate
    }

    // Pre-allocate scratch buffer once, reuse across all consolidation passes.
    // Avoids per-call heap allocation in the hot path.
    let mut k_normalized = vec![0.0f32; config.head_dim];

    // Run N consolidation passes
    for _pass in 0..sleep_config.sleep_passes {
        consolidation_pass(kv_cache, gdn2_cache, fill_pos, config, &mut k_normalized);
    }

    // Evict KV cache according to strategy
    eviction::evict(kv_cache, sleep_config.eviction, config);

    sleep_config.sleep_passes
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config::micro() // n_layer=1, n_head=4, n_kv_head=4, head_dim=4, block_size=16
    }

    fn random_weights(config: &Config) -> TransformerWeights {
        let mut rng = crate::types::Rng::new(42);
        TransformerWeights::new(config, &mut rng)
    }

    /// Fill KV cache with a simple pattern.
    fn fill_kv(cache: &mut MultiLayerKVCache, config: &Config, n_tokens: usize) {
        let kvd = types::kv_dim(config);
        for pos in 0..n_tokens {
            for layer in &mut cache.layers {
                let off = pos * kvd;
                // Write recognizable values
                for i in 0..kvd {
                    layer.key[off + i] = (pos as f32 * 0.1 + i as f32 * 0.01).max(0.01);
                    layer.value[off + i] = (pos as f32 * 0.2 + i as f32 * 0.02).max(0.01);
                }
            }
            cache.advance_pos(pos);
        }
    }

    #[test]
    fn consolidation_pass_updates_gdn2_state() {
        let config = test_config();
        let mut kv_cache = MultiLayerKVCache::new(&config);
        let mut gdn2_cache = MultiLayerGdn2Cache::new(&config);

        // Fill KV with 8 tokens
        fill_kv(&mut kv_cache, &config, 8);
        let fill_pos = kv_cache.fill_pos();

        // GDN2 state should start at zero
        let initial_sum: f32 = gdn2_cache.layers[0].heads[0]
            .s
            .iter()
            .map(|&v| v.abs())
            .sum();

        let mut k_normalized = vec![0.0f32; config.head_dim];
        consolidation_pass(
            &kv_cache,
            &mut gdn2_cache,
            fill_pos,
            &config,
            &mut k_normalized,
        );

        // GDN2 state should now be non-zero (consolidated from KV)
        let after_sum: f32 = gdn2_cache.layers[0].heads[0]
            .s
            .iter()
            .map(|&v| v.abs())
            .sum();

        assert!(
            after_sum > initial_sum,
            "GDN2 state should be updated after consolidation: before={initial_sum}, after={after_sum}"
        );
    }

    #[test]
    fn consolidation_pass_produces_finite_state() {
        let config = test_config();
        let mut kv_cache = MultiLayerKVCache::new(&config);
        let mut gdn2_cache = MultiLayerGdn2Cache::new(&config);

        fill_kv(&mut kv_cache, &config, 12);

        let mut k_normalized = vec![0.0f32; config.head_dim];
        consolidation_pass(&kv_cache, &mut gdn2_cache, 12, &config, &mut k_normalized);

        // All GDN2 state values should be finite
        for layer in &gdn2_cache.layers {
            for head in &layer.heads {
                for (i, &v) in head.s.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "GDN2 state should be finite: layer/head[{i}]={v}"
                    );
                }
            }
        }
    }

    #[test]
    fn sleep_with_hard_evict_clears_kv() {
        let config = test_config();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut kv_cache = MultiLayerKVCache::new(&config);
        let mut gdn2_cache = MultiLayerGdn2Cache::new(&config);

        fill_kv(&mut kv_cache, &config, 10);
        assert!(kv_cache.fill_pos() > 0);

        let sleep_config = SleepConfig {
            sleep_passes: 2,
            eviction: super::super::types::EvictionStrategy::HardEvict,
            window_size: config.block_size,
        };

        let passes = sleep(
            &mut ctx,
            &weights,
            &mut kv_cache,
            &mut gdn2_cache,
            &sleep_config,
            &config,
        );

        assert_eq!(passes, 2);
        assert_eq!(
            kv_cache.fill_pos(),
            0,
            "KV cache should be empty after hard eviction"
        );
    }

    #[test]
    fn sleep_with_empty_cache_is_noop() {
        let config = test_config();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut kv_cache = MultiLayerKVCache::new(&config);
        let mut gdn2_cache = MultiLayerGdn2Cache::new(&config);

        let sleep_config = SleepConfig::default();
        let passes = sleep(
            &mut ctx,
            &weights,
            &mut kv_cache,
            &mut gdn2_cache,
            &sleep_config,
            &config,
        );

        assert_eq!(passes, 0, "Sleep on empty cache should return 0 passes");
    }

    #[test]
    fn multiple_passes_strengthen_state() {
        let config = test_config();
        let mut kv_cache = MultiLayerKVCache::new(&config);

        fill_kv(&mut kv_cache, &config, 8);
        let fill_pos = kv_cache.fill_pos();

        // Run 1 pass
        let mut gdn2_cache_1 = MultiLayerGdn2Cache::new(&config);
        let mut k_normalized = vec![0.0f32; config.head_dim];
        consolidation_pass(
            &kv_cache,
            &mut gdn2_cache_1,
            fill_pos,
            &config,
            &mut k_normalized,
        );
        let sum_1: f32 = gdn2_cache_1.layers[0].heads[0]
            .s
            .iter()
            .map(|&v| v * v)
            .sum();

        // Run 4 passes
        let mut gdn2_cache_4 = MultiLayerGdn2Cache::new(&config);
        let mut k_normalized = vec![0.0f32; config.head_dim];
        for _ in 0..4 {
            consolidation_pass(
                &kv_cache,
                &mut gdn2_cache_4,
                fill_pos,
                &config,
                &mut k_normalized,
            );
        }
        let sum_4: f32 = gdn2_cache_4.layers[0].heads[0]
            .s
            .iter()
            .map(|&v| v * v)
            .sum();

        // Multiple passes should produce different (generally stronger) state
        // The exact relationship depends on decay/gate values, but they shouldn't be identical
        assert!(
            (sum_1 - sum_4).abs() > 1e-10,
            "Multiple passes should change the state: 1-pass sum={sum_1}, 4-pass sum={sum_4}"
        );

        // Both should be finite
        assert!(sum_1.is_finite());
        assert!(sum_4.is_finite());
    }
}
