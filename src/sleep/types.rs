//! Sleep consolidation types — SleepConfig, EvictionStrategy.
//!
//! Plan 154: Sleep Consolidation — Offline Recursive Memory Consolidation at Eviction.
//!
//! When the KV cache fills, perform N offline recurrent passes to consolidate context
//! into GDN2 fast weights, then evict. This preserves single-pass wake-time latency
//! for real-time game constraints (20Hz frame sampling).

// ── Eviction Strategy ─────────────────────────────────────────

/// Strategy for evicting KV cache entries after sleep consolidation.
///
/// After N recurrent consolidation passes bake the KV cache contents into GDN2
/// fast-weight state, the eviction strategy determines what happens to the cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EvictionStrategy {
    /// Clear the entire KV cache after sleep consolidation.
    ///
    /// All context is assumed to be captured in the GDN2 fast-weight state S.
    /// Best for: long-context streaming where only the recurrent state matters.
    #[default]
    HardEvict,
    /// Retain the last `window_size - 1` tokens, evict older entries.
    ///
    /// A sliding window that keeps recent tokens for local coherence while
    /// older context lives in GDN2 fast weights. The retained tokens are
    /// shifted to positions [0..window_size-1) after eviction.
    SlidingWindow {
        /// Number of recent tokens to retain after eviction.
        /// Default: 1 (keep only the most recent token for positional continuity).
        retain: usize,
    },
}

// ── Sleep Configuration ───────────────────────────────────────

/// Configuration for sleep-time memory consolidation.
///
/// Controls how many recurrent consolidation passes to run when the KV cache
/// reaches capacity, and what eviction strategy to use afterward.
///
/// # Usage
///
/// ```ignore
/// use katgpt_rs::sleep::types::SleepConfig;
/// let config = SleepConfig::default();
/// assert_eq!(config.sleep_passes, 2);
/// assert_eq!(config.window_size, 512);
/// ```
#[derive(Clone, Debug)]
pub struct SleepConfig {
    /// Number of recurrent consolidation passes at eviction boundary.
    ///
    /// Each pass feeds all cached K/V pairs through the GDN2 recurrent step,
    /// consolidating the full context into the fast-weight state S.
    /// Paper shows N=2-4 provides most of the benefit.
    pub sleep_passes: usize,
    /// KV cache capacity threshold that triggers sleep.
    ///
    /// When the cache reaches this many tokens, sleep consolidation fires.
    /// Should be ≤ `Config::block_size`.
    pub window_size: usize,
    /// Eviction strategy after consolidation.
    ///
    /// Determines what happens to the KV cache after sleep finishes.
    pub eviction: EvictionStrategy,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            sleep_passes: 2,
            window_size: 512,
            eviction: EvictionStrategy::default(),
        }
    }
}

impl SleepConfig {
    /// Create a new sleep config with the given number of passes.
    pub fn new(sleep_passes: usize) -> Self {
        Self {
            sleep_passes,
            ..Default::default()
        }
    }

    /// Create config with sliding window eviction.
    pub fn with_sliding_window(sleep_passes: usize, retain: usize) -> Self {
        Self {
            sleep_passes,
            eviction: EvictionStrategy::SlidingWindow { retain },
            ..Default::default()
        }
    }

    /// Whether sleep should trigger at the given cache position.
    ///
    /// Sleep triggers when `pos + 1 >= window_size` (cache is full).
    pub fn should_sleep(&self, pos: usize) -> bool {
        pos + 1 >= self.window_size
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eviction_strategy_default_is_hard_evict() {
        assert_eq!(EvictionStrategy::default(), EvictionStrategy::HardEvict);
    }

    #[test]
    fn sleep_config_default_passes() {
        let config = SleepConfig::default();
        assert_eq!(config.sleep_passes, 2);
        assert_eq!(config.window_size, 512);
        assert_eq!(config.eviction, EvictionStrategy::HardEvict);
    }

    #[test]
    fn sleep_config_should_sleep_at_boundary() {
        let config = SleepConfig {
            window_size: 16,
            ..Default::default()
        };
        // pos=15 → pos+1=16 >= 16 → should sleep
        assert!(config.should_sleep(15));
        // pos=14 → pos+1=15 < 16 → should not sleep yet
        assert!(!config.should_sleep(14));
    }

    #[test]
    fn sleep_config_with_sliding_window() {
        let config = SleepConfig::with_sliding_window(4, 8);
        assert_eq!(config.sleep_passes, 4);
        assert_eq!(
            config.eviction,
            EvictionStrategy::SlidingWindow { retain: 8 }
        );
    }
}
