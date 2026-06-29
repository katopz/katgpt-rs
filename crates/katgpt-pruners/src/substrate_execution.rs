//! SubstrateGate dual sparsity execution — ReLU ∩ substrate mask intersection (Plan 216 T4-T5).
//!
//! After ReLU activation and before the w2 down-projection, intersects the sparse
//! active set (from ReLU) with the substrate mask (from capability routing).
//! This produces dual sparsity: only channels that are BOTH ReLU-active AND
//! substrate-relevant participate in the matmul.

use super::substrate_types::{SubstrateMask, SubstrateRouter};

// ── sigmoid helper (used by other substrate modules) ──────────

#[inline]
#[allow(dead_code)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── apply_substrate_mask ───────────────────────────────────────

/// Apply substrate mask intersection to ReLU-active channels.
///
/// Given the post-ReLU active indices and values for a single layer,
/// filters out channels that are NOT in the substrate mask.
///
/// Returns new `(active_indices, active_values)` with only the intersection.
///
/// # Arguments
/// * `active_indices` — indices of ReLU-active channels (from sparse_matmul)
/// * `active_values` — corresponding activation values
/// * `mask` — the substrate mask to intersect with
/// * `layer_idx` — which transformer layer this is
///
/// # Performance
/// O(active_count) — scans only ReLU-active channels, not all channels.
pub fn apply_substrate_mask(
    active_indices: &[usize],
    active_values: &[f32],
    mask: &SubstrateMask,
    layer_idx: usize,
) -> (Vec<usize>, Vec<f32>) {
    let count = active_indices.len().min(active_values.len());
    let mut out_indices = Vec::with_capacity(count);
    let mut out_values = Vec::with_capacity(count);

    for i in 0..count {
        let idx = active_indices[i];
        if mask.get(layer_idx, idx) {
            out_indices.push(idx);
            out_values.push(active_values[i]);
        }
    }

    (out_indices, out_values)
}

/// Apply substrate mask intersection in-place to pre-allocated buffers.
///
/// Writes the intersection back into the provided buffers, returning the
/// new alive count. Avoids allocation — writes in-place.
///
/// # Returns
/// New alive count after intersection.
pub fn apply_substrate_mask_inplace(
    active_indices: &mut [usize],
    active_values: &mut [f32],
    mask: &SubstrateMask,
    layer_idx: usize,
) -> usize {
    let count = active_indices.len().min(active_values.len());
    let mut write = 0usize;

    for read in 0..count {
        let idx = active_indices[read];
        if mask.get(layer_idx, idx) {
            active_indices[write] = idx;
            active_values[write] = active_values[read];
            write += 1;
        }
    }

    write
}

// ── should_use_substrate ────────────────────────────────────────

/// Heuristic: should we apply the substrate mask for this layer?
///
/// If the mask's active ratio is too high (> 0.4), the mask overhead
/// exceeds the savings from reduced FLOPs. Better to use dense path.
pub fn should_use_substrate(mask: &SubstrateMask) -> bool {
    mask.recovery_score() > 0.0 && mask.active_ratio() < 0.4
}

/// Compute the FLOPs reduction ratio from applying the substrate mask.
///
/// Returns the fraction of FLOPs saved: 1.0 = all saved, 0.0 = no savings.
pub fn flops_reduction_ratio(mask: &SubstrateMask, active_ratio: f32) -> f32 {
    let substrate_ratio = mask.active_ratio();
    if substrate_ratio >= 1.0 {
        return 0.0;
    }
    // FLOPs ∝ active_channels / total_channels
    // With mask: FLOPs ∝ (active ∩ substrate) / total ≈ active_ratio * substrate_ratio
    let with_mask = active_ratio * substrate_ratio;
    let without_mask = active_ratio;
    if without_mask == 0.0 {
        return 0.0;
    }
    1.0 - (with_mask / without_mask)
}

// ── Dual-sparsity sparse matmul: ReLU ∩ substrate mask → w2 down-projection ──

/// Dual-sparsity sparse matmul for the w2 down-projection.
///
/// Packs ReLU-active neurons, intersects with the substrate mask in-place,
/// then performs sparse matmul with the reduced active set.
/// Same weight layout as `sparse_matmul`: weight[row * cols + col].
///
/// Returns the post-intersection alive count.
#[allow(clippy::too_many_arguments)]
pub fn sparse_matmul_substrate(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
    active_indices: &mut [usize],
    active_values: &mut [f32],
    mask: &SubstrateMask,
    layer_idx: usize,
) -> usize {
    // Phase 1: Pack ReLU-active neuron indices and values
    let mut relu_alive = 0usize;
    for (c, &val) in input.iter().enumerate().take(cols) {
        if val > 0.0 {
            active_indices[relu_alive] = c;
            active_values[relu_alive] = val;
            relu_alive += 1;
        }
    }

    // Phase 1.5: Substrate mask intersection — in-place compaction
    let alive = apply_substrate_mask_inplace(
        &mut active_indices[..relu_alive],
        &mut active_values[..relu_alive],
        mask,
        layer_idx,
    );

    // Skip matmul entirely if intersection killed all neurons
    if alive == 0 {
        return 0;
    }

    // Phase 2: Sparse multiply with reduced active set
    katgpt_core::simd::simd_sparse_matmul_rows(
        output,
        weight,
        &active_indices[..alive],
        &active_values[..alive],
        rows,
        cols,
        alive,
    );

    alive
}

// ── SubstrateExecutionContext ──────────────────────────────────

/// Execution context for substrate-aware forward pass.
///
/// Holds the router and selected mask for the current sequence.
/// Mask selection happens once per sequence (not per token) for efficiency.
pub struct SubstrateExecutionContext<R: SubstrateRouter> {
    router: R,
    /// Currently selected mask for this sequence (cached).
    /// TODO: wire up mask index caching once router exposes index selection.
    #[allow(dead_code)]
    selected_mask_idx: Option<usize>,
}

impl<R: SubstrateRouter> SubstrateExecutionContext<R> {
    pub fn new(router: R) -> Self {
        Self {
            router,
            selected_mask_idx: None,
        }
    }

    /// Select mask for the current sequence based on token context.
    ///
    /// Caches the result — call once at sequence start, then use `current_mask()`.
    pub fn select_for_sequence(
        &mut self,
        tokens: &[usize],
        config: &katgpt_types::Config,
    ) -> Option<&SubstrateMask> {
        self.router.select_mask(tokens, config)
    }

    /// Apply the substrate mask to ReLU-active channels for a layer.
    ///
    /// Returns `(indices, values, alive_count)`. Uses the selected mask.
    /// If no mask is selected or the mask shouldn't be applied, returns None.
    pub fn apply_to_layer(
        &self,
        active_indices: &[usize],
        active_values: &[f32],
        mask: &SubstrateMask,
        layer_idx: usize,
    ) -> Option<(Vec<usize>, Vec<f32>)> {
        if !should_use_substrate(mask) {
            return None;
        }

        let (indices, values) =
            apply_substrate_mask(active_indices, active_values, mask, layer_idx);

        if indices.is_empty() {
            return None;
        }

        Some((indices, values))
    }
}

// ── T11: CPU sparse index-packed matmul with dual mask ──────────

/// CPU-only sparse matmul: intersects ReLU-active channels with the substrate mask,
/// then accumulates `value * weight_row` for each surviving channel.
///
/// The weight matrix is row-major `[d_model][mlp_hidden]` — for channel `c`,
/// the weight row spans `weights[c * d_model .. (c + 1) * d_model]`.
///
/// Returns the number of active channels used after mask intersection (for FLOPs accounting).
///
/// # Arguments
/// * `active_indices` — indices of ReLU-active channels
/// * `active_values` — corresponding activation values (same length as `active_indices`)
/// * `weights` — weight matrix `[d_model * mlp_hidden]`, row-major
/// * `d_model` — model dimension (row stride)
/// * `mask` — substrate mask for intersection
/// * `layer_idx` — which transformer layer
/// * `output` — pre-allocated output buffer of length `d_model`
///
/// # Safety
/// Caller must ensure `weights.len() >= d_model * mlp_hidden` and `output.len() >= d_model`.
pub fn cpu_sparse_substrate_matmul(
    active_indices: &[usize],
    active_values: &[f32],
    weights: &[f32],
    d_model: usize,
    mask: &SubstrateMask,
    layer_idx: usize,
    output: &mut [f32],
) -> usize {
    let count = active_indices.len().min(active_values.len());
    let mut active_used = 0usize;

    for i in 0..count {
        let ch = active_indices[i];

        // Dual sparsity: ReLU-active ∩ substrate-active
        if !mask.get(layer_idx, ch) {
            continue;
        }

        let val = active_values[i];
        let row_start = ch * d_model;
        let row_end = row_start + d_model;

        // Bounds check — skip if weight row is out of range
        if row_end > weights.len() {
            continue;
        }

        // Accumulate: output += val * weights[ch, :]
        let row = &weights[row_start..row_end];
        for (j, &w) in row.iter().enumerate() {
            output[j] += val * w;
        }

        active_used += 1;
    }

    active_used
}

// ── T12: GPU path placeholder ────────────────────────────────────

/// Placeholder for batched GPU substrate matmul.
///
/// When multiple branches share the same substrate mask, a GPU batched
/// matmul would be more efficient. This stub returns an error until
/// an actual GPU backend is integrated.
///
/// # Arguments
/// * `n_branches` — number of branches to batch
/// * `_masks` — substrate masks per branch (unused until GPU impl)
/// * `_layer_idx` — transformer layer index (unused until GPU impl)
///
/// # Errors
/// Always returns an error — GPU substrate path is not yet implemented.
#[allow(unused_variables)]
pub fn gpu_batch_substrate_matmul(
    n_branches: usize,
    masks: &[&SubstrateMask],
    layer_idx: usize,
) -> Result<usize, &'static str> {
    Err("GPU substrate path not yet implemented")
}

// ── T13: Auto-route heuristic ────────────────────────────────────

/// Auto-route: decide whether to use dense path, GPU batch, or CPU sparse matmul.
///
/// Decision logic:
/// 1. If `mask.active_ratio() > 0.4` → dense path (mask overhead exceeds savings) → return `None`
/// 2. If `n_branches > 4 && gpu_available` → would use GPU batch, but GPU not yet
///    implemented, so fall through to CPU sparse
/// 3. Otherwise → CPU sparse path via `cpu_sparse_substrate_matmul`
///
/// Returns `Some(active_channels_used)` when the sparse path was taken, or `None`
/// when the dense path is preferred (mask not applied).
///
/// # Arguments
/// * `active_indices` — ReLU-active channel indices
/// * `active_values` — corresponding activation values
/// * `mask` — substrate mask
/// * `weights` — weight matrix `[d_model * mlp_hidden]`, row-major
/// * `d_model` — model dimension
/// * `output` — pre-allocated output buffer (will be accumulated into)
/// * `layer_idx` — transformer layer index
/// * `gpu_available` — whether GPU is available
/// * `n_branches` — number of branches for batch consideration
#[allow(clippy::too_many_arguments)]
pub fn auto_route_substrate(
    active_indices: &[usize],
    active_values: &[f32],
    mask: &SubstrateMask,
    weights: &[f32],
    d_model: usize,
    output: &mut [f32],
    layer_idx: usize,
    gpu_available: bool,
    n_branches: usize,
) -> Option<usize> {
    // Gate 1: mask too dense → skip, use dense path
    if mask.active_ratio() > 0.4 {
        return None;
    }

    // Gate 2: large batch + GPU available → would batch on GPU
    //         (GPU path not yet implemented, so fall through to CPU)
    if n_branches > 4 && gpu_available {
        // Future: gpu_batch_substrate_matmul(...)
        // For now, fall through to CPU sparse path
    }

    // Gate 3: CPU sparse path
    let active_used = cpu_sparse_substrate_matmul(
        active_indices,
        active_values,
        weights,
        d_model,
        mask,
        layer_idx,
        output,
    );

    if active_used == 0 {
        return None;
    }

    Some(active_used)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mask_with_channels(
        n_layers: usize,
        mlp_hidden: usize,
        channels: &[(usize, usize)],
    ) -> SubstrateMask {
        let mut mask = SubstrateMask::new(
            n_layers,
            mlp_hidden,
            "test".to_string(),
            "model".to_string(),
        );
        for &(layer, ch) in channels {
            mask.set(layer, ch);
        }
        mask.set_recovery_score(0.8);
        mask
    }

    #[test]
    fn test_apply_mask_filters_correctly() {
        let mask = make_mask_with_channels(1, 100, &[(0, 0), (0, 2), (0, 4)]);

        let indices = vec![0, 1, 2, 3, 4, 5];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (out_idx, out_val) = apply_substrate_mask(&indices, &values, &mask, 0);

        assert_eq!(out_idx, vec![0, 2, 4]);
        assert_eq!(out_val, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_apply_mask_empty_mask() {
        let mask = SubstrateMask::new(1, 100, "test".to_string(), "model".to_string());

        let indices = vec![0, 1, 2];
        let values = vec![1.0, 2.0, 3.0];

        let (out_idx, _) = apply_substrate_mask(&indices, &values, &mask, 0);
        assert!(out_idx.is_empty());
    }

    #[test]
    fn test_apply_mask_all_active() {
        let mut mask = SubstrateMask::new(1, 64, "test".to_string(), "model".to_string());
        for ch in 0..64 {
            mask.set(0, ch);
        }
        mask.set_recovery_score(0.9);

        let indices = vec![0, 10, 20, 30];
        let values = vec![1.0, 2.0, 3.0, 4.0];

        let (out_idx, out_val) = apply_substrate_mask(&indices, &values, &mask, 0);

        assert_eq!(out_idx, indices);
        assert_eq!(out_val, values);
    }

    #[test]
    fn test_apply_mask_inplace() {
        let mask = make_mask_with_channels(1, 100, &[(0, 0), (0, 2)]);

        let mut indices = vec![0, 1, 2, 3];
        let mut values = vec![1.0, 2.0, 3.0, 4.0];

        let alive = apply_substrate_mask_inplace(&mut indices, &mut values, &mask, 0);

        assert_eq!(alive, 2);
        assert_eq!(indices[..alive], [0, 2]);
        assert_eq!(values[..alive], [1.0, 3.0]);
    }

    #[test]
    fn test_should_use_substrate() {
        let mut mask = SubstrateMask::new(1, 64, "test".to_string(), "model".to_string());

        // Zero recovery → should not use
        assert!(!should_use_substrate(&mask));

        // Good recovery, low active ratio → should use
        mask.set_recovery_score(0.8);
        mask.set(0, 10); // 1/64 ≈ 1.5% active
        assert!(should_use_substrate(&mask));

        // Good recovery, high active ratio (> 0.4) → should not use
        let mut dense_mask = SubstrateMask::new(1, 64, "dense".to_string(), "model".to_string());
        dense_mask.set_recovery_score(0.8);
        for ch in 0..64 {
            dense_mask.set(0, ch);
        }
        assert!(!should_use_substrate(&dense_mask));
    }

    #[test]
    fn test_flops_reduction_ratio() {
        // 50% ReLU active, 20% substrate active
        let mut mask = SubstrateMask::new(1, 100, "test".to_string(), "model".to_string());
        for ch in 0..20 {
            mask.set(0, ch);
        }
        mask.set_recovery_score(0.8);

        let reduction = flops_reduction_ratio(&mask, 0.5);
        // Expected: 1.0 - (0.5 * 0.2) / 0.5 = 1.0 - 0.2 = 0.8
        assert!(
            (reduction - 0.8).abs() < 0.01,
            "expected 0.8, got {}",
            reduction
        );
    }

    #[test]
    fn test_flops_reduction_full_relu() {
        let mask = SubstrateMask::new(1, 100, "test".to_string(), "model".to_string());
        // No active channels in substrate → 100% reduction (everything blocked)
        let reduction = flops_reduction_ratio(&mask, 0.5);
        assert!(
            (reduction - 1.0).abs() < 0.001,
            "expected 1.0, got {}",
            reduction
        );
    }

    #[test]
    fn test_apply_mask_multilayer() {
        let mask = make_mask_with_channels(2, 100, &[(0, 5), (1, 10)]);

        // Layer 0: channel 5 is active
        let (idx0, _) = apply_substrate_mask(&[5, 15], &[1.0, 2.0], &mask, 0);
        assert_eq!(idx0, vec![5]);

        // Layer 1: channel 10 is active
        let (idx1, _) = apply_substrate_mask(&[5, 10, 15], &[1.0, 2.0, 3.0], &mask, 1);
        assert_eq!(idx1, vec![10]);
    }

    // ── T11: cpu_sparse_substrate_matmul tests ─────────────────

    #[test]
    fn test_cpu_sparse_matmul_basic() {
        // d_model=4, mlp_hidden=8 → weights is [8][4] row-major
        // Channel 0 weights: [1.0, 0.0, 0.0, 0.0]
        // Channel 3 weights: [0.0, 0.0, 0.0, 1.0]
        let weights: Vec<f32> = (0..32)
            .map(|i| {
                let ch = i / 4;
                let col = i % 4;
                if (ch == 0 && col == 0) || (ch == 3 && col == 3) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        // Mask allows channels 0 and 3
        let mask = make_mask_with_channels(1, 8, &[(0, 0), (0, 3)]);

        // All channels active (0..7), values are all 2.0
        let indices: Vec<usize> = (0..8).collect();
        let values: Vec<f32> = vec![2.0; 8];
        let mut output = vec![0.0f32; 4];

        let used =
            cpu_sparse_substrate_matmul(&indices, &values, &weights, 4, &mask, 0, &mut output);

        // Only channels 0 and 3 survive the mask
        assert_eq!(used, 2);
        // Channel 0: val=2.0 * [1,0,0,0] → output[0] += 2.0
        // Channel 3: val=2.0 * [0,0,0,1] → output[3] += 2.0
        assert_eq!(output, vec![2.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_cpu_sparse_matmul_no_intersection() {
        let weights = vec![0.0f32; 16]; // d_model=2, mlp_hidden=8
        let mask = make_mask_with_channels(1, 8, &[(0, 5)]); // only ch 5

        let indices = vec![0usize, 1, 2]; // ch 0,1,2 active via ReLU
        let values = vec![1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 2];

        let used =
            cpu_sparse_substrate_matmul(&indices, &values, &weights, 2, &mask, 0, &mut output);

        assert_eq!(used, 0); // No intersection
        assert_eq!(output, vec![0.0, 0.0]); // Output untouched
    }

    #[test]
    fn test_cpu_sparse_matmul_accumulates() {
        // d_model=2, mlp_hidden=4
        // Channel 1 weights: [2.0, 3.0]
        let weights = vec![0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0];
        let mask = make_mask_with_channels(1, 4, &[(0, 1)]);

        let indices = vec![1usize];
        let values = vec![5.0f32];
        let mut output = vec![1.0f32, 2.0]; // pre-existing output

        let used =
            cpu_sparse_substrate_matmul(&indices, &values, &weights, 2, &mask, 0, &mut output);

        assert_eq!(used, 1);
        // 1.0 + 5.0*2.0 = 11.0, 2.0 + 5.0*3.0 = 17.0
        assert_eq!(output, vec![11.0, 17.0]);
    }

    // ── T12: gpu_batch_substrate_matmul test ────────────────────

    #[test]
    fn test_gpu_batch_stub_returns_error() {
        let mask = make_mask_with_channels(1, 64, &[(0, 0)]);
        let result = gpu_batch_substrate_matmul(8, &[&mask], 0);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "GPU substrate path not yet implemented"
        );
    }

    // ── T13: auto_route_substrate tests ─────────────────────────

    #[test]
    fn test_auto_route_dense_mask_returns_none() {
        // Create a mask with >40% active → dense path, returns None
        let mut mask = SubstrateMask::new(1, 10, "dense".into(), "model".into());
        for ch in 0..10 {
            mask.set(0, ch); // 100% active
        }
        mask.set_recovery_score(0.8);

        let weights = vec![0.0f32; 20]; // d_model=2, mlp_hidden=10
        let mut output = vec![0.0f32; 2];

        let result = auto_route_substrate(
            &[0, 1],
            &[1.0, 2.0],
            &mask,
            &weights,
            2,
            &mut output,
            0,
            false,
            1,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_auto_route_sparse_mask_takes_cpu_path() {
        // Sparse mask: only 1/10 channels active
        let mask = make_mask_with_channels(1, 10, &[(0, 0)]);

        // d_model=2, mlp_hidden=10
        // Channel 0 weights: [3.0, 4.0]
        let mut weights = vec![0.0f32; 20];
        weights[0] = 3.0;
        weights[1] = 4.0;

        let mut output = vec![0.0f32; 2];

        let result =
            auto_route_substrate(&[0], &[2.0], &mask, &weights, 2, &mut output, 0, false, 1);

        assert_eq!(result, Some(1));
        // 2.0 * [3.0, 4.0] = [6.0, 8.0]
        assert_eq!(output, vec![6.0, 8.0]);
    }

    #[test]
    fn test_auto_route_no_intersection_returns_none() {
        // Sparse mask but no overlap with active indices
        let mask = make_mask_with_channels(1, 10, &[(0, 9)]);
        let weights = vec![0.0f32; 20];
        let mut output = vec![0.0f32; 2];

        let result = auto_route_substrate(
            &[0, 1],
            &[1.0, 2.0],
            &mask,
            &weights,
            2,
            &mut output,
            0,
            false,
            1,
        );

        assert!(result.is_none());
        assert_eq!(output, vec![0.0, 0.0]);
    }

    #[test]
    fn test_auto_route_gpu_fallthrough_to_cpu() {
        // n_branches > 4 && gpu_available → would try GPU, falls through to CPU
        let mask = make_mask_with_channels(1, 10, &[(0, 2)]);
        let mut weights = vec![0.0f32; 20];
        weights[4] = 1.0; // Channel 2, col 0
        weights[5] = 1.0; // Channel 2, col 1
        let mut output = vec![0.0f32; 2];

        let result =
            auto_route_substrate(&[2], &[3.0], &mask, &weights, 2, &mut output, 0, true, 8);

        assert_eq!(result, Some(1));
        assert_eq!(output, vec![3.0, 3.0]);
    }
}
