//! KV cache eviction strategies after sleep consolidation.
//!
//! After N recurrent consolidation passes bake the KV cache contents into GDN2
//! fast-weight state, the eviction module clears (or slides) the KV cache.
//!
//! Plan 154: Sleep Consolidation.

use super::types::EvictionStrategy;
use crate::transformer::MultiLayerKVCache;
use crate::types;
use crate::types::Config;

/// Apply eviction to the KV cache after sleep consolidation.
///
/// The GDN2 fast-weight state S already holds the consolidated context,
/// so the KV cache can be safely cleared or trimmed.
pub fn evict(cache: &mut MultiLayerKVCache, strategy: EvictionStrategy, config: &Config) {
    match strategy {
        EvictionStrategy::HardEvict => hard_evict(cache, config),
        EvictionStrategy::SlidingWindow { retain } => sliding_window_evict(cache, config, retain),
    }
}

/// Hard eviction: clear the entire KV cache.
///
/// After consolidation, all context lives in GDN2 fast weights.
/// Zero out all KV entries and reset fill_pos.
fn hard_evict(cache: &mut MultiLayerKVCache, _config: &Config) {
    for layer in &mut cache.layers {
        layer.key.fill(0.0);
        layer.value.fill(0.0);
    }
    cache.reset();
}

/// Sliding window eviction: retain the last `retain` tokens, shift to front.
///
/// Copies the last `retain` K/V entries to positions [0..retain) and zeros the rest.
/// This preserves local context for immediate next tokens while older context
/// is captured in GDN2 fast weights.
fn sliding_window_evict(cache: &mut MultiLayerKVCache, config: &Config, retain: usize) {
    let kvd = types::kv_dim(config);
    let fill_pos = cache.fill_pos();

    // Nothing to slide if cache isn't filled past retain
    if fill_pos <= retain {
        return;
    }

    let evict_count = fill_pos - retain;
    let src_start = evict_count * kvd;
    let copy_len = retain * kvd;
    let total_len = fill_pos * kvd;

    for layer in &mut cache.layers {
        // Use copy_within for efficient overlapping shift (no temp allocation).
        layer.key.copy_within(src_start..total_len, 0);
        layer.key[copy_len..].fill(0.0);

        layer.value.copy_within(src_start..total_len, 0);
        layer.value[copy_len..].fill(0.0);
    }

    // Update fill position: reset clears to 0, then set to `retain`.
    // advance_pos(pos) sets fill_pos = max(fill_pos, pos + 1), so with
    // retain > 0 this correctly sets fill_pos = retain.
    cache.reset();
    if retain > 0 {
        cache.advance_pos(retain - 1);
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config::micro() // block_size=16, n_embd=48, head_dim=4, n_kv_head=4
    }

    fn fill_cache(cache: &mut MultiLayerKVCache, config: &Config, n_tokens: usize) {
        let kvd = types::kv_dim(config);
        for pos in 0..n_tokens {
            for layer in &mut cache.layers {
                let off = pos * kvd;
                // Write a recognizable pattern: key = pos as f32 repeated, value = (pos+1) as f32 repeated
                layer.key[off..off + kvd].fill(pos as f32);
                layer.value[off..off + kvd].fill((pos + 1) as f32);
            }
            cache.advance_pos(pos);
        }
    }

    #[test]
    fn hard_evict_clears_everything() {
        let config = test_config();
        let mut cache = MultiLayerKVCache::new(&config);
        fill_cache(&mut cache, &config, 8);

        // Verify cache is non-empty
        assert!(cache.fill_pos() > 0);

        evict(&mut cache, EvictionStrategy::HardEvict, &config);

        // After hard eviction, all entries should be zeroed
        assert_eq!(cache.fill_pos(), 0);
        for layer in &cache.layers {
            assert!(layer.key.iter().all(|&v| v == 0.0));
            assert!(layer.value.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn sliding_window_retains_recent() {
        let config = test_config();
        let mut cache = MultiLayerKVCache::new(&config);
        fill_cache(&mut cache, &config, 10);

        let retain = 3;
        evict(
            &mut cache,
            EvictionStrategy::SlidingWindow { retain },
            &config,
        );

        let kvd = types::kv_dim(&config);
        // After sliding, the first `retain` positions should contain
        // what was previously at positions 7, 8, 9
        for layer in &cache.layers {
            // Position 0 should have old position 7's data (key=7.0, value=8.0)
            assert!(
                (layer.key[0] - 7.0).abs() < 1e-6,
                "Expected key[0]=7.0 after slide"
            );
            assert!(
                (layer.value[0] - 8.0).abs() < 1e-6,
                "Expected value[0]=8.0 after slide"
            );
            // Position 1 should have old position 8's data
            assert!(
                (layer.key[kvd] - 8.0).abs() < 1e-6,
                "Expected key[1]=8.0 after slide"
            );
            // Position 2 should have old position 9's data
            assert!(
                (layer.key[2 * kvd] - 9.0).abs() < 1e-6,
                "Expected key[2]=9.0 after slide"
            );
        }
    }

    #[test]
    fn sliding_window_noop_when_under_retain() {
        let config = test_config();
        let mut cache = MultiLayerKVCache::new(&config);
        fill_cache(&mut cache, &config, 2);

        let retain = 5; // more than fill
        let fill_before = cache.fill_pos();
        evict(
            &mut cache,
            EvictionStrategy::SlidingWindow { retain },
            &config,
        );

        // Should be no-op when fill <= retain
        assert_eq!(cache.fill_pos(), fill_before);
    }
}
