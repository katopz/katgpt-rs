//! Training-Free Loop Wrapper — ODE-Refined Sub-Stepping (Plan 136).
//!
//! Pure inference-time retrofit: re-applies a contiguous mid-stack block of
//! layers with ODE-motivated damped sub-stepping. No training needed.
//!
//! # Architecture
//!
//! ```text
//! Pre-loop: x ← L₀ ∘ ... ∘ L_{a-1}(x)     [standard, write KV]
//! Anchor:   x̃ ← (L_b ∘ ... ∘ L_a)(x)       [one-shot for β blend]
//! Loop:
//!   for k = 1..K:
//!     y ← (L_b ∘ ... ∘ L_a)(x)             [forward window]
//!     x ← x + (1/K)·(y - x)                [damped Euler sub-step]
//!   x ← β·x̃ + (1-β)·x                      [anchor blend]
//! Stash:    write canonical KV from x (cache=last) or x_pre (cache=first)
//! Post-loop: x ← L_{b+1} ∘ ... ∘ L_{N-1}(x) [standard, write KV]
//! ```
//!
//! Run tests: `cargo test --features tf_loop`

use crate::transformer::{KVCache, MultiLayerKVCache};
use katgpt_core::types::Config;
use katgpt_core::types::kv_dim;

/// Returns a sensible default loop window for a transformer with `n_layers` layers.
///
/// Uses a depth-fraction heuristic: center at 48% depth, ±1 layer.
/// For small models (≤4 layers), defaults to (0, n_layers-1).
///
/// # Examples
/// - 12 layers → (4, 7)  (center ≈ 5.76)
/// - 24 layers → (10, 13) (center ≈ 11.52)
/// - 6 layers → (1, 4)   (center ≈ 2.88)
pub fn default_loop_window(n_layers: usize) -> (usize, usize) {
    if n_layers <= 4 {
        return (0, n_layers.saturating_sub(1));
    }
    let center = (n_layers as f32 * 0.48) as usize;
    let start = center.saturating_sub(1);
    let end = (center + 2).min(n_layers - 1);
    (start, end)
}

/// Applies one damped Euler sub-step in-place.
///
/// Computes: `x[i] ← x[i] + (1/k)·(y[i] − x[i])` for all i.
///
/// This is equivalent to `x[i] ← ((k-1)/k)·x[i] + (1/k)·y[i]`,
/// a convex combination biased toward the current state.
///
/// # Panics
/// Debug-asserts that `x` and `y` have the same length.
/// When `k == 0`, this is a no-op (identity).
pub fn sub_step_damped_euler(x: &mut [f32], y: &[f32], k: usize) {
    debug_assert_eq!(x.len(), y.len(), "x and y must have the same length");
    if k == 0 {
        return;
    }
    let inv_k = 1.0f32 / k as f32;
    for (xi, yi) in x.iter_mut().zip(y.iter()) {
        *xi += inv_k * (*yi - *xi);
    }
}

/// Blends `x` with an anchor state in-place.
///
/// Computes: `x[i] ← beta·anchor[i] + (1−beta)·x[i]` for all i.
///
/// - `beta = 0.0` → pure x (anchor ignored)
/// - `beta = 1.0` → pure anchor (x replaced)
/// - `beta = 0.5` → equal blend
///
/// # Panics
/// Debug-asserts that `x` and `anchor` have the same length.
pub fn anchor_blend(x: &mut [f32], anchor: &[f32], beta: f32) {
    debug_assert_eq!(
        x.len(),
        anchor.len(),
        "x and anchor must have the same length"
    );
    let one_minus_beta = 1.0 - beta;
    for (xi, ai) in x.iter_mut().zip(anchor.iter()) {
        *xi = beta * ai + one_minus_beta * *xi;
    }
}

/// Records per-layer KV cache lengths for later restore.
///
/// For each layer in `layers`, records `key.len()` (which equals `block_size × kv_dim`).
/// The snapshot can be restored via `restore_cache_lengths`.
pub fn snapshot_cache_lengths(
    cache: &MultiLayerKVCache,
    layers: std::ops::Range<usize>,
) -> Vec<usize> {
    layers
        .map(|i| {
            if i < cache.layers.len() {
                cache.layers[i].key.len()
            } else {
                0
            }
        })
        .collect()
}

/// Crops KV cache back to snapshot lengths.
///
/// Zeros out entries beyond the snapshot length for each layer in `layers`.
/// This effectively restores the cache to the state it was in when the snapshot was taken
/// (assuming no entries were written beyond the snapshot boundary).
pub fn restore_cache_lengths(
    cache: &mut MultiLayerKVCache,
    layers: std::ops::Range<usize>,
    snapshot: &[usize],
) {
    for (i, &len) in layers.zip(snapshot.iter()) {
        if i < cache.layers.len() {
            let layer = &mut cache.layers[i];
            // Zero everything beyond the snapshot point
            if len < layer.key.len() {
                layer.key[len..].fill(0.0);
            }
            if len < layer.value.len() {
                layer.value[len..].fill(0.0);
            }
        }
    }
}

/// Records per-layer KV cache fill position (in positions, not elements).
///
/// Unlike `snapshot_cache_lengths` which records total buffer sizes,
/// this records how many positions are actually filled, which is more
/// useful for tracking cache state during loop iterations.
pub fn snapshot_cache_positions(
    cache: &MultiLayerKVCache,
    layers: std::ops::Range<usize>,
    config: &Config,
) -> Vec<usize> {
    let kd = kv_dim(config);
    layers
        .map(|i| {
            if i < cache.layers.len() {
                // Find the last non-zero position
                let key = &cache.layers[i].key;
                let mut max_pos = 0;
                for p in 0..config.block_size {
                    let off = p * kd;
                    if key[off..off + kd].iter().any(|&v| v != 0.0) {
                        max_pos = p + 1;
                    }
                }
                max_pos
            } else {
                0
            }
        })
        .collect()
}

/// Restores KV cache to a snapshot of positions, zeroing beyond.
pub fn restore_cache_positions(
    cache: &mut MultiLayerKVCache,
    layers: std::ops::Range<usize>,
    positions: &[usize],
    config: &Config,
) {
    let kd = kv_dim(config);
    for (i, &pos) in layers.zip(positions.iter()) {
        if i < cache.layers.len() {
            let layer = &mut cache.layers[i];
            let start = pos * kd;
            if start < layer.key.len() {
                layer.key[start..].fill(0.0);
            }
            if start < layer.value.len() {
                layer.value[start..].fill(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── default_loop_window ──────────────────────────────────────

    #[test]
    fn test_window_12_layers() {
        let (s, e) = default_loop_window(12);
        // center = 5.76 → 5, start = 4, end = 7
        assert_eq!(s, 4);
        assert_eq!(e, 7);
    }

    #[test]
    fn test_window_24_layers() {
        let (s, e) = default_loop_window(24);
        // center = 11.52 → 11, start = 10, end = 13
        assert_eq!(s, 10);
        assert_eq!(e, 13);
    }

    #[test]
    fn test_window_6_layers() {
        let (s, e) = default_loop_window(6);
        // center = 2.88 → 2, start = 1, end = 4
        assert_eq!(s, 1);
        assert_eq!(e, 4);
    }

    #[test]
    fn test_window_4_layers() {
        // Small model: entire stack
        let (s, e) = default_loop_window(4);
        assert_eq!(s, 0);
        assert_eq!(e, 3);
    }

    #[test]
    fn test_window_2_layers() {
        let (s, e) = default_loop_window(2);
        assert_eq!(s, 0);
        assert_eq!(e, 1);
    }

    #[test]
    fn test_window_1_layer() {
        let (s, e) = default_loop_window(1);
        assert_eq!(s, 0);
        assert_eq!(e, 0);
    }

    #[test]
    fn test_window_32_layers() {
        let (s, e) = default_loop_window(32);
        // center = 15.36 → 15, start = 14, end = 17
        assert_eq!(s, 14);
        assert_eq!(e, 17);
    }

    // ── sub_step_damped_euler ────────────────────────────────────

    #[test]
    fn test_euler_basic() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        let y = vec![4.0f32, 5.0, 6.0];
        sub_step_damped_euler(&mut x, &y, 2);
        // x[i] = x[i] + 0.5*(y[i] - x[i]) = 0.5*x[i] + 0.5*y[i]
        assert!((x[0] - 2.5).abs() < 1e-6);
        assert!((x[1] - 3.5).abs() < 1e-6);
        assert!((x[2] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_euler_k1_full_step() {
        // K=1: x ← x + 1.0*(y - x) = y (full replacement)
        let mut x = vec![1.0f32, 2.0];
        let y = vec![10.0f32, 20.0];
        sub_step_damped_euler(&mut x, &y, 1);
        assert!((x[0] - 10.0).abs() < 1e-6);
        assert!((x[1] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_euler_k0_identity() {
        // K=0: no-op
        let mut x = vec![1.0f32, 2.0];
        let y = vec![10.0f32, 20.0];
        sub_step_damped_euler(&mut x, &y, 0);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
    }

    #[test]
    fn test_euler_k4_small_step() {
        let mut x = vec![0.0f32];
        let y = vec![4.0f32];
        sub_step_damped_euler(&mut x, &y, 4);
        // x = 0 + 0.25*(4 - 0) = 1.0
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euler_convergence_after_k_steps() {
        // After K sub-steps with same y, x converges toward y
        let mut x = vec![0.0f32];
        let y = vec![10.0f32];
        let k = 4;
        for _ in 0..k {
            sub_step_damped_euler(&mut x, &y, k);
        }
        // After 4 steps of 1/4 each: x = y*(1 - (3/4)^4) ≈ y*0.6836
        // Not exactly y, but closer than after 1 step
        assert!(x[0] > 0.0 && x[0] < 10.0);
        assert!(x[0] > 2.0, "should be closer to y after K steps");
    }

    // ── anchor_blend ─────────────────────────────────────────────

    #[test]
    fn test_blend_beta_zero() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        let anchor = vec![10.0f32, 20.0, 30.0];
        anchor_blend(&mut x, &anchor, 0.0);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
        assert_eq!(x[2], 3.0);
    }

    #[test]
    fn test_blend_beta_one() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        let anchor = vec![10.0f32, 20.0, 30.0];
        anchor_blend(&mut x, &anchor, 1.0);
        assert!((x[0] - 10.0).abs() < 1e-6);
        assert!((x[1] - 20.0).abs() < 1e-6);
        assert!((x[2] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_beta_half() {
        let mut x = vec![0.0f32, 0.0];
        let anchor = vec![10.0f32, 20.0];
        anchor_blend(&mut x, &anchor, 0.5);
        assert!((x[0] - 5.0).abs() < 1e-6);
        assert!((x[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_blend_linearity() {
        // Blend should be a convex combination: β·a + (1-β)·x
        let mut x = vec![2.0f32];
        let anchor = vec![8.0f32];
        let beta = 0.3;
        anchor_blend(&mut x, &anchor, beta);
        let expected = beta * 8.0 + (1.0 - beta) * 2.0;
        assert!((x[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_blend_preserves_finiteness() {
        let mut x = vec![1e10f32, -1e10];
        let anchor = vec![-1e10f32, 1e10];
        anchor_blend(&mut x, &anchor, 0.5);
        assert!(x[0].is_finite());
        assert!(x[1].is_finite());
    }
}
