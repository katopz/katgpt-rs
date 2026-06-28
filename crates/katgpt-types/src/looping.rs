//! Training-Free Loop types.

use super::*;

// ---------------------------------------------------------------------------
// Training-Free Loop Types (Plan 136)
// ---------------------------------------------------------------------------

/// Sub-step integration strategy for the training-free loop.
///
/// Controls how intermediate loop outputs are combined with the running state.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SubStepStrategy {
    /// Damped Euler: x ← x + (1/K)·(y − x)
    #[default]
    DampedEuler,
    /// K-stage Runge-Kutta: x ← β·y + (1−β)·x
    KStageRK {
        /// Blend factor β ∈ [0, 1]. 0.5 is neutral (equal weight).
        beta: f32,
    },
}

/// Iteration mode for the training-free loop window.
///
/// Controls whether the window is applied as a single block or iterated
/// layer-by-layer within each sub-step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum IterationMode {
    /// Apply the full window [a, b] as one block per sub-step.
    #[default]
    Block,
    /// Apply each layer in the window individually per sub-step.
    Layer,
}

/// KV cache write strategy for the training-free loop.
///
/// Controls which loop iteration writes the canonical KV entries.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Use the final loop iteration's hidden state for KV cache.
    #[default]
    Last,
    /// Use the pre-loop hidden state for KV cache (first iteration).
    First,
}

/// Configuration for training-free loop wrapper (Plan 136).
///
/// Pure inference-time retrofit: re-applies a contiguous mid-stack block of
/// layers with ODE-motivated damped sub-stepping. No training needed.
#[derive(Clone, Debug)]
pub struct TrainingFreeLoopConfig {
    /// Start of the loop window (inclusive layer index).
    pub window_start: usize,
    /// End of the loop window (inclusive layer index).
    pub window_end: usize,
    /// Number of loop iterations (K in the paper).
    pub loop_count: usize,
    /// Sub-step integration strategy.
    pub strategy: SubStepStrategy,
    /// Iteration mode (block vs layer-wise).
    pub iteration_mode: IterationMode,
    /// KV cache write strategy.
    pub cache_strategy: CacheStrategy,
}

impl Default for TrainingFreeLoopConfig {
    fn default() -> Self {
        Self {
            window_start: 0,
            window_end: 0,
            loop_count: 2,
            strategy: SubStepStrategy::KStageRK { beta: 0.5 },
            iteration_mode: IterationMode::Block,
            cache_strategy: CacheStrategy::First,
        }
    }
}

impl TrainingFreeLoopConfig {
    /// Creates a config with sensible defaults for a given `Config`.
    ///
    /// Window heuristic: center at 48% depth, ±1 layer.
    /// For small models (≤4 layers), defaults to (0, n_layer−1).
    /// Uses the paper-recommended K-stage RK strategy with β=0.5.
    pub fn from_config(config: &Config) -> Self {
        let n = config.n_layer;
        let (window_start, window_end) = if n <= 4 {
            (0, n.saturating_sub(1))
        } else {
            let center = (n as f32 * 0.48) as usize;
            (center.saturating_sub(1), (center + 2).min(n - 1))
        };
        Self {
            window_start,
            window_end,
            loop_count: 2,
            strategy: SubStepStrategy::KStageRK { beta: 0.5 },
            iteration_mode: IterationMode::Block,
            cache_strategy: CacheStrategy::First,
        }
    }
}
