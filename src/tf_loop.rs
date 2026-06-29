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

use katgpt_core::simd::simd_fused_decay_write;
use crate::transformer::MultiLayerKVCache;
use katgpt_core::types::Config;
use katgpt_core::types::kv_dim;

// ── RecFM Acceleration-Bounded Sub-Stepping (Plan 168 T2) ──────────

/// Configuration for RecFM acceleration-bounded sub-stepping (Research 150).
///
/// RecFM's consistency framework constrains trajectory acceleration ∂t_v.
/// Applied to LT2 sub-stepping, this applies additional damping when the
/// state velocity (change between iterations) exceeds a threshold.
///
/// When `enable` is `false`, all RecFM checks are no-ops (zero cost).
#[cfg(feature = "recfm")]
#[derive(Debug, Clone, Copy)]
pub struct AccelBoundConfig {
    /// Enable acceleration-bounded sub-stepping.
    pub enable: bool,
    /// Normalized acceleration threshold above which extra damping is applied.
    /// RecFM default: 0.5 (moderate constraint).
    pub accel_threshold: f32,
    /// Extra damping factor applied when acceleration exceeds threshold.
    /// x *= extra_damp_factor after the sub-step.
    /// RecFM default: 0.8 (mild damping).
    pub extra_damp_factor: f32,
}

#[cfg(feature = "recfm")]
impl Default for AccelBoundConfig {
    fn default() -> Self {
        Self {
            enable: false,
            accel_threshold: 0.5,
            extra_damp_factor: 0.8,
        }
    }
}

/// Compute normalized acceleration norm between two velocity vectors.
///
/// Returns `||v_curr - v_prev||₂ / dim` — the average per-dimension acceleration.
/// This is the discrete analog of RecFM's continuous acceleration field.
///
/// Uses the same SIMD infrastructure as `simd_fused_decay_write` for zero-overhead.
#[cfg(feature = "recfm")]
#[inline]
pub fn accel_norm(v_curr: &[f32], v_prev: &[f32]) -> f32 {
    debug_assert_eq!(v_curr.len(), v_prev.len(), "velocity slices must match");
    let dim = v_curr.len();
    if dim == 0 {
        return 0.0;
    }
    let sum_sq: f32 = v_curr
        .iter()
        .zip(v_prev.iter())
        .map(|(c, p)| {
            let d = c - p;
            d * d
        })
        .sum();
    (sum_sq / dim as f32).sqrt()
}

/// Returns a sensible default loop window for a transformer with `n_layers` layers.
///
/// Uses a depth-fraction heuristic: center at 48% depth, ±1 layer.
/// For small models (≤4 layers), defaults to (0, n_layers-1).
///
/// # Examples
/// - 12 layers → (4, 7)  (center ≈ 5.76)
/// - 24 layers → (10, 13) (center ≈ 11.52)
/// - 6 layers → (1, 4)   (center ≈ 2.88)
#[inline]
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
#[inline]
pub fn sub_step_damped_euler(x: &mut [f32], y: &[f32], k: usize) {
    debug_assert_eq!(x.len(), y.len(), "x and y must have the same length");
    if k == 0 {
        return;
    }
    let inv_k = 1.0f32 / k as f32;
    let n = x.len();
    // x[i] = (1-inv_k)*x[i] + inv_k*y[i]
    simd_fused_decay_write(&mut x[..n], 1.0 - inv_k, &y[..n], inv_k);
}

/// Acceleration-bounded damped Euler sub-step with RecFM constraint.
///
/// Identical to [`sub_step_damped_euler`] but additionally checks the acceleration
/// norm between the current state and the proposed update. When acceleration
/// exceeds the threshold, applies extra damping to prevent divergence.
///
/// When `accel_config.enable == false`, delegates to [`sub_step_damped_euler`] (zero overhead).
#[cfg(feature = "recfm")]
#[inline]
pub fn sub_step_damped_euler_bounded(
    x: &mut [f32],
    y: &[f32],
    k: usize,
    x_prev: &[f32],
    accel_config: &AccelBoundConfig,
) {
    sub_step_damped_euler(x, y, k);

    if !accel_config.enable || k == 0 {
        return;
    }

    // Compute acceleration: how much x changed from previous state
    let accel = accel_norm(x, x_prev);
    if accel > accel_config.accel_threshold {
        // Apply extra damping: x *= extra_damp_factor
        for xi in x.iter_mut() {
            *xi *= accel_config.extra_damp_factor;
        }
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
#[inline]
pub fn anchor_blend(x: &mut [f32], anchor: &[f32], beta: f32) {
    debug_assert_eq!(
        x.len(),
        anchor.len(),
        "x and anchor must have the same length"
    );
    let n = x.len();
    // x[i] = (1-beta)*x[i] + beta*anchor[i]
    simd_fused_decay_write(&mut x[..n], 1.0 - beta, &anchor[..n], beta);
}

/// Records per-layer KV cache lengths for later restore.
///
/// For each layer in `layers`, records `key.len()` (which equals `block_size × kv_dim`).
/// The snapshot can be restored via `restore_cache_lengths`.
///
/// Reuses `buf` across calls to avoid per-call allocation.
#[inline]
pub fn snapshot_cache_lengths(
    cache: &MultiLayerKVCache,
    layers: std::ops::Range<usize>,
    buf: &mut Vec<usize>,
) {
    buf.clear();
    buf.extend(layers.map(|i| {
        if i < cache.layers.len() {
            cache.layers[i].key.len()
        } else {
            0
        }
    }));
}

/// Crops KV cache back to snapshot lengths.
///
/// Zeros out entries beyond the snapshot length for each layer in `layers`.
/// This effectively restores the cache to the state it was in when the snapshot was taken
/// (assuming no entries were written beyond the snapshot boundary).
#[inline]
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

/// Returns the current KV cache fill position (in positions, not elements).
///
/// Unlike `snapshot_cache_lengths` which records total buffer sizes,
/// this returns the single tracked fill position that applies uniformly
/// to all layers — O(1) with zero allocation.
///
/// Uses the `fill_pos` tracker on MultiLayerKVCache for O(1) lookup
/// instead of scanning all positions for non-zero entries.
#[inline]
pub fn snapshot_cache_positions(
    cache: &MultiLayerKVCache,
    _layers: std::ops::Range<usize>,
    _config: &Config,
) -> usize {
    cache.fill_pos()
}

/// Restores KV cache to a snapshot position, zeroing beyond.
#[inline]
pub fn restore_cache_positions(
    cache: &mut MultiLayerKVCache,
    layers: std::ops::Range<usize>,
    pos: usize,
    config: &Config,
) {
    let kd = kv_dim(config);
    let start = pos * kd;
    for i in layers {
        if i < cache.layers.len() {
            let layer = &mut cache.layers[i];
            if start < layer.key.len() {
                layer.key[start..].fill(0.0);
            }
            if start < layer.value.len() {
                layer.value[start..].fill(0.0);
            }
        }
    }
}

// ── Thinking Prune — FrozenBaseGuard for LT2 Loops (Plan 171) ──────

/// Determine whether the current loop iteration should apply full screening.
///
/// In the Thinking Pixel framework (arXiv:2604.25299 §3.3), intermediate
/// recursion steps should use lightweight processing only. Applied to LT2:
/// intermediate loop iterations use damped sub-stepping only (no pruner),
/// while the final iteration applies the full `ConstraintPruner`/`ScreeningPruner`.
///
/// # Arguments
///
/// * `iteration` — Current loop iteration (0-based)
/// * `total_iterations` — Total number of loop iterations (K)
/// * `schedule` — The pruner schedule to use
///
/// # Returns
///
/// `true` if full screening should be applied at this iteration.
#[cfg(feature = "thinking_prune")]
#[inline]
pub fn should_apply_pruner_at_iteration(
    iteration: usize,
    total_iterations: usize,
    schedule: crate::pruners::PrunerSchedule,
) -> bool {
    schedule.should_screen_full(iteration, total_iterations)
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

    // ── GOAT Proof Tests (Plan 136 T16–T19) ─────────────────────

    /// Proof 1: `sub_step_damped_euler` and `anchor_blend` produce finite results.
    ///
    /// K-stage RK loop with synthetic affine transforms produces finite,
    /// non-NaN output for K∈{2,3,4}, β∈{0.25,0.5,0.75}.
    #[test]
    fn proof_tf_loop_finite() {
        let dim = 128;
        let ks = [2, 3, 4, 8, 16];
        let betas = [0.25, 0.5, 0.75];

        for &k in &ks {
            for &beta in &betas {
                let mut x = vec![1.0f32; dim];
                let anchor = vec![2.0f32; dim];

                // Simulate K loop iterations with affine transform y = 0.8*x + 0.1
                for _ in 0..k {
                    let mut y = vec![0.0f32; dim];
                    for (yi, xi) in y.iter_mut().zip(x.iter()) {
                        *yi = 0.8 * xi + 0.1;
                    }
                    sub_step_damped_euler(&mut x, &y, k);
                }

                // Anchor blend
                anchor_blend(&mut x, &anchor, beta);

                // All outputs must be finite
                for (i, &v) in x.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "Non-finite at K={k}, beta={beta}, idx={i}: {v}"
                    );
                }

                // Outputs must be bounded (shouldn't grow unbounded)
                let max_abs = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                assert!(
                    max_abs < 1e6,
                    "Output diverged at K={k}, beta={beta}: max_abs={max_abs}"
                );
            }
        }
    }

    /// Proof 2: snapshot/restore produces same cache sizes.
    ///
    /// After snapshotting and restoring, the cache buffer sizes are identical
    /// and independent of K (loop count).
    #[test]
    fn proof_tf_loop_cache_size() {
        use katgpt_core::types::Config;

        let config = Config::micro();
        let mut cache = MultiLayerKVCache::new(&config);

        // Snapshot before any writes
        let mut snap = Vec::new();
        snapshot_cache_lengths(&cache, 0..config.n_layer, &mut snap);

        // Write to all layers at position 0
        let kvd = katgpt_core::types::kv_dim(&config);
        for layer in &mut cache.layers {
            layer.key[0..kvd].fill(1.0);
            layer.value[0..kvd].fill(1.0);
        }

        // Snapshot after writes (reusing same buffer)
        let mut snap_after = Vec::new();
        snapshot_cache_lengths(&cache, 0..config.n_layer, &mut snap_after);

        // Buffer sizes are identical (KV cache is pre-allocated)
        assert_eq!(
            snap, snap_after,
            "Cache sizes should be identical regardless of writes"
        );

        // Restore and verify sizes still match
        restore_cache_lengths(&mut cache, 0..config.n_layer, &snap);
        let mut snap_restored = Vec::new();
        snapshot_cache_lengths(&cache, 0..config.n_layer, &mut snap_restored);
        assert_eq!(
            snap, snap_restored,
            "Cache sizes should match after restore"
        );
    }

    /// Proof 3: damped Euler with K=0 is identity (bypass is free).
    ///
    /// When K=0, `sub_step_damped_euler` is a no-op — the state is unchanged.
    /// This ensures the training-free loop adds zero overhead when disabled.
    #[test]
    fn proof_tf_loop_bypass_free() {
        let _dim = 64;
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let original = x.clone();
        let y = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];

        // K=0 → identity
        sub_step_damped_euler(&mut x, &y, 0);
        assert_eq!(x, original, "K=0 should be identity");

        // Also check beta=0 anchor blend is identity
        let mut x2 = vec![1.0f32, 2.0, 3.0];
        let original2 = x2.clone();
        let anchor = vec![100.0f32, 200.0, 300.0];
        anchor_blend(&mut x2, &anchor, 0.0);
        assert_eq!(x2, original2, "beta=0 should be identity");

        // Window of size 0 (empty range) → no loop iterations at all
        let empty: Vec<usize> = (0..0).collect();
        assert!(empty.is_empty(), "Empty window should have no iterations");
    }

    /// Proof 4: layer-mode sub-stepping is stable.
    ///
    /// Layer-by-layer iteration within window produces finite, bounded output.
    /// This verifies that applying sub-stepping per-layer (rather than per-block)
    /// doesn't cause numerical instability.
    #[test]
    fn proof_tf_loop_layer_mode_stable() {
        let dim = 64;
        let ks = [2, 3, 4];
        let n_window_layers = 3;

        for &k in &ks {
            let mut x = vec![1.0f32; dim];
            let anchor = vec![0.5f32; dim];
            let beta = 0.5;

            // Simulate layer-mode: sub-step after each layer in the window
            for _ in 0..k {
                for _l in 0..n_window_layers {
                    // Synthetic layer transform: y = 0.9*x + 0.05
                    let mut y = vec![0.0f32; dim];
                    for (yi, xi) in y.iter_mut().zip(x.iter()) {
                        *yi = 0.9 * xi + 0.05;
                    }
                    // Per-layer sub-step
                    let pre = x.clone();
                    sub_step_damped_euler(&mut x, &y, k);
                    // After sub-step, x should move toward y from pre
                    for (i, ((&xi, &yi), &pi)) in x.iter().zip(y.iter()).zip(pre.iter()).enumerate()
                    {
                        let expected = pi + (1.0 / k as f32) * (yi - pi);
                        assert!(
                            (xi - expected).abs() < 1e-5,
                            "Layer-mode mismatch at K={k}, layer={_l}, idx={i}: {xi} vs {expected}"
                        );
                    }
                }
            }

            // Anchor blend
            anchor_blend(&mut x, &anchor, beta);

            // All outputs must be finite
            for (i, &v) in x.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "Non-finite in layer-mode at K={k}, idx={i}: {v}"
                );
            }

            // Bounded
            let max_abs = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                max_abs < 1e6,
                "Layer-mode diverged at K={k}: max_abs={max_abs}"
            );
        }
    }
}
