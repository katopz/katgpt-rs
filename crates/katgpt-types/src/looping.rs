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
    /// Running mean of K/V across the loop iterations (Issue 698 T5).
    ///
    /// The window iterations already produce k K/V rows at `pos`; this
    /// strategy folds their incremental streaming mean (fixed loop order ⇒
    /// deterministic f32 sum) and writes it back — the dedicated stash
    /// window-forward is deleted (a whole window pass per token saved).
    ///
    /// LOSSY by design: gate callers on per-position argmax stability + a
    /// pinned max_abs band (the Bench-773 rule — max_rel with a floor
    /// denominator cannot certify lossy numerics), never bit-identity.
    /// The k=1 degenerate is exact (one iteration's mean IS that
    /// iteration's K/V, which equals the `First` stash), so plumbing
    /// correctness is bit-testable even though the k>1 path is not.
    Mean,
}

/// Why a loop's cache read is refused (Issue 1241 T3).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheRefusal {
    /// No sustained contraction was certified — the last round is not known
    /// to sit at the fixed point (arXiv:2611.13378 App. D: a fixed-depth
    /// non-contractive core loses +0.0317 BPB under last-round KV).
    NoContractionCertificate,
    /// A limit cycle with no integer period in the scanned range — no
    /// round, and no period average, is a cache-safe representative.
    AperiodicCycle,
}

/// What a looped core may cache — the certificate ORTHOGONAL to the halt
/// reason (Issue 1241 T3; when to stop and what to cache are independent,
/// arXiv:2611.13378 App. D). Produced by katgpt-core's
/// `OscillationVerdict::cache_verdict`; REPORT-ONLY until Issue 1241 T4's
/// head-to-head PoC arms a consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheVerdict {
    /// Contraction certified: the last round is cache-safe
    /// ([`CacheStrategy::Last`]).
    Last,
    /// The cache read is refused.
    Refused(CacheRefusal),
    /// A limit cycle of period `P`: average the K/V over exactly the last
    /// `P` rounds — a mean over one full period cancels the periodic
    /// first-order residual (aliasing), where `Last` keeps one phase of it.
    MeanAtPeriod(u8),
}

impl CacheVerdict {
    /// The shipped strategy this verdict selects (`Mean` for a period —
    /// the caller must window it to exactly `P` rounds); `None` = refused.
    #[inline]
    pub fn strategy(self) -> Option<CacheStrategy> {
        match self {
            Self::Last => Some(CacheStrategy::Last),
            Self::MeanAtPeriod(_) => Some(CacheStrategy::Mean),
            Self::Refused(_) => None,
        }
    }
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
