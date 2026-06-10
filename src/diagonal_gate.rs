//! Shared DiagonalGate abstraction for GDN2 and Wall attention.
//!
//! Both GDN2's per-channel decay (Diag(α)) and Wall's per-dimension
//! prefix-sum gate (Diag(g_t)) are instances of the same primitive:
//! a diagonal [d]-dimensional vector applied to attention state.
//!
//! This trait provides a unified interface for:
//! - Computing gate values from projections
//! - Applying diagonal scaling to vectors
//! - Accumulating gate state (prefix sum or decay)

use crate::simd::{
    simd_add_scalar_inplace, simd_exp_inplace, simd_scale_inplace, simd_scale_mul_inplace,
};

// ── Trait ─────────────────────────────────────────────────────────

/// A per-dimension diagonal gate applied to attention state.
///
/// Implemented by GDN2 (channel-wise decay) and Wall (prefix-sum rescale).
pub trait DiagonalGate {
    /// Dimension of the gate vector.
    fn dim(&self) -> usize;

    /// Compute gate logits from a projection: input ⊙ weights + bias.
    /// Writes gate values (after activation) to `out`.
    fn compute_gate(&self, input: &[f32], weights: &[f32], bias: f32, out: &mut [f32]);

    /// Apply diagonal scaling: target *= gate_values.
    fn apply(&self, gate_values: &[f32], target: &mut [f32]);

    /// Apply inverse diagonal scaling: target *= exp(-gate_values).
    fn apply_inverse(&self, gate_values: &[f32], target: &mut [f32]);

    /// Reset gate state to initial values.
    fn reset(&mut self);
}

// ── GDN2 ──────────────────────────────────────────────────────────

/// GDN2 diagonal decay gate: Diag(α) applied to recurrent state.
///
/// GDN2 applies per-channel multiplicative decay to the recurrent state
/// matrix S at each timestep: `S *= Diag(α)`. The decay values are
/// fixed (not projected from input) and default to 0.99.
pub struct Gdn2DiagonalGate {
    /// Per-channel decay values [dim].
    pub alpha: Vec<f32>,
}

/// Default decay value for GDN2 gates.
const GDN2_DEFAULT_DECAY: f32 = 0.99;

impl Gdn2DiagonalGate {
    /// Create a new GDN2 diagonal gate with default decay (0.99).
    pub fn new(dim: usize) -> Self {
        Self {
            alpha: vec![GDN2_DEFAULT_DECAY; dim],
        }
    }

    /// Create from existing decay values.
    pub fn from_alpha(alpha: Vec<f32>) -> Self {
        Self { alpha }
    }
}

impl DiagonalGate for Gdn2DiagonalGate {
    #[inline]
    fn dim(&self) -> usize {
        self.alpha.len()
    }

    /// GDN2 uses fixed decay_alpha (not projected from input).
    /// Fills `out` with the current alpha values.
    #[inline]
    fn compute_gate(&self, _input: &[f32], _weights: &[f32], _bias: f32, out: &mut [f32]) {
        let d = self.dim();
        out[..d].copy_from_slice(&self.alpha);
    }

    /// Apply decay: target[i] *= alpha[i].
    #[inline]
    fn apply(&self, gate_values: &[f32], target: &mut [f32]) {
        let d = self.dim();
        debug_assert_eq!(gate_values.len(), d);
        debug_assert!(target.len() >= d);
        simd_scale_mul_inplace(&mut target[..d], gate_values, 1.0);
    }

    /// Apply inverse: target[i] /= alpha[i] (for backward pass).
    /// Uses multiply by 1/alpha to avoid repeated division.
    #[inline]
    fn apply_inverse(&self, gate_values: &[f32], target: &mut [f32]) {
        let d = self.dim();
        debug_assert_eq!(gate_values.len(), d);
        debug_assert!(target.len() >= d);
        for i in 0..d {
            target[i] /= gate_values[i];
        }
    }

    /// Reset to default decay (0.99).
    #[inline]
    fn reset(&mut self) {
        self.alpha.fill(GDN2_DEFAULT_DECAY);
    }
}

// ── Wall ───────────────────────────────────────────────────────────

/// Wall diagonal gate: prefix-sum accumulated gate for Q/K rescale.
///
/// Wall Attention replaces RoPE with learned diagonal forget gates.
/// Gate values are computed via key projection → log-sigmoid → clamp,
/// then accumulated via prefix sum. The prefix sums are applied as:
/// - Query rescale: `q̃ = exp(P) ⊙ q`
/// - Key rescale: `k̃ = exp(-P) ⊙ k`
pub struct WallDiagonalGate {
    /// Per-dimension prefix sums [dim].
    pub prefix: Vec<f32>,
    /// Gate max clamp value (default 0.87).
    pub gate_max: f32,
}

/// Default gate max clamp value.
const WALL_DEFAULT_GATE_MAX: f32 = 0.87;

impl WallDiagonalGate {
    /// Create a new Wall diagonal gate with zeroed prefix sums.
    pub fn new(dim: usize) -> Self {
        Self {
            prefix: vec![0.0; dim],
            gate_max: WALL_DEFAULT_GATE_MAX,
        }
    }

    /// Create with custom gate_max.
    pub fn with_gate_max(dim: usize, gate_max: f32) -> Self {
        Self {
            prefix: vec![0.0; dim],
            gate_max,
        }
    }

    /// Compute gate from key projection: log-sigmoid clamped to (-gate_max, 0].
    ///
    /// Matches `WallPrefixState::compute_gate_from_key` logic:
    /// 1. gate = key ⊙ weights + bias
    /// 2. gate = -gate (negate logits)
    /// 3. gate = exp(gate)
    /// 4. gate = ln(1 + gate) → softplus
    /// 5. gate = -gate → log_sigmoid
    /// 6. clamp(-gate_max, 0.0)
    #[inline]
    pub fn compute_gate_from_projection(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: f32,
        out: &mut [f32],
    ) {
        let hd = input.len();
        debug_assert_eq!(weights.len(), hd);
        debug_assert!(out.len() >= hd);

        // Step 1: out = input ⊙ weights
        out[..hd].copy_from_slice(&input[..hd]);
        simd_scale_mul_inplace(&mut out[..hd], weights, 1.0);

        // Step 2: out += bias
        simd_add_scalar_inplace(&mut out[..hd], bias);

        // Step 3: out = -out (negate logits)
        simd_scale_inplace(&mut out[..hd], -1.0);

        // Step 4: out = exp(-logits)
        simd_exp_inplace(&mut out[..hd]);

        // Step 5: out = ln(1 + exp(-logit)) → softplus(-logit), then negate
        // log_sigmoid(logit) = -ln(1 + exp(-logit)) = -softplus(-logit)
        simd_add_scalar_inplace(&mut out[..hd], 1.0);
        for v in out[..hd].iter_mut() {
            let log_sig = -(*v).ln();
            *v = log_sig.clamp(-self.gate_max, 0.0);
        }
    }
}

impl DiagonalGate for WallDiagonalGate {
    #[inline]
    fn dim(&self) -> usize {
        self.prefix.len()
    }

    /// Compute gate from projection and write to `out`.
    #[inline]
    fn compute_gate(&self, input: &[f32], weights: &[f32], bias: f32, out: &mut [f32]) {
        self.compute_gate_from_projection(input, weights, bias, out);
    }

    /// Apply query rescale: target[i] *= exp(gate_values[i]).
    #[inline]
    fn apply(&self, gate_values: &[f32], target: &mut [f32]) {
        let d = gate_values.len().min(target.len());
        // target *= exp(gate_values)
        // Copy to temp, exp, multiply — avoids mutating gate_values in place.
        let mut exp_buf = vec![0.0f32; d];
        exp_buf[..d].copy_from_slice(&gate_values[..d]);
        simd_exp_inplace(&mut exp_buf[..d]);
        simd_scale_mul_inplace(&mut target[..d], &exp_buf, 1.0);
    }

    /// Apply key rescale: target[i] *= exp(-gate_values[i]).
    #[inline]
    fn apply_inverse(&self, gate_values: &[f32], target: &mut [f32]) {
        let d = gate_values.len().min(target.len());
        // target *= exp(-gate_values)
        let mut neg_buf = vec![0.0f32; d];
        neg_buf[..d].copy_from_slice(&gate_values[..d]);
        simd_scale_inplace(&mut neg_buf[..d], -1.0);
        simd_exp_inplace(&mut neg_buf[..d]);
        simd_scale_mul_inplace(&mut target[..d], &neg_buf, 1.0);
    }

    /// Reset prefix sums to zero.
    #[inline]
    fn reset(&mut self) {
        self.prefix.fill(0.0);
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const HD: usize = 4;

    #[test]
    fn gdn2_diagonal_gate_apply() {
        let gate = Gdn2DiagonalGate::from_alpha(vec![0.9, 0.8, 0.7, 0.6]);
        let mut target = vec![1.0, 1.0, 1.0, 1.0];
        gate.apply(&gate.alpha, &mut target);
        assert!((target[0] - 0.9).abs() < 1e-6);
        assert!((target[1] - 0.8).abs() < 1e-6);
        assert!((target[2] - 0.7).abs() < 1e-6);
        assert!((target[3] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn gdn2_diagonal_gate_inverse() {
        let gate = Gdn2DiagonalGate::from_alpha(vec![0.9, 0.8, 0.7, 0.6]);
        let mut target = vec![0.9, 0.8, 0.7, 0.6];
        gate.apply_inverse(&gate.alpha, &mut target);
        for v in &target {
            assert!((v - 1.0).abs() < 1e-5, "Expected ~1.0, got {v}");
        }
    }

    #[test]
    fn wall_diagonal_gate_apply() {
        let gate = WallDiagonalGate::new(HD);
        // apply: target *= exp(gate_values)
        let gate_values = vec![0.5, 1.0, -0.5, 0.0];
        let mut target = vec![1.0, 1.0, 1.0, 1.0];
        gate.apply(&gate_values, &mut target);
        // SIMD Cephes exp has ~1e-3 relative accuracy; use 0.05 tolerance.
        let tol = 0.05f32;
        assert!(
            (target[0] - 0.5_f32.exp()).abs() < tol,
            "d0: {} vs {}",
            target[0],
            0.5_f32.exp()
        );
        assert!(
            (target[1] - 1.0_f32.exp()).abs() < tol,
            "d1: {} vs {}",
            target[1],
            1.0_f32.exp()
        );
        assert!((target[2] - (-0.5_f32).exp()).abs() < tol);
        assert!((target[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn wall_diagonal_gate_inverse() {
        let gate = WallDiagonalGate::new(HD);
        // apply_inverse: target *= exp(-gate_values)
        let gate_values = vec![0.5, 1.0, -0.5, 0.0];
        let mut target = vec![1.0, 1.0, 1.0, 1.0];
        gate.apply_inverse(&gate_values, &mut target);
        // SIMD Cephes exp has ~1e-3 relative accuracy; use 0.05 tolerance.
        let tol = 0.05f32;
        assert!((target[0] - (-0.5_f32).exp()).abs() < tol);
        assert!((target[1] - (-1.0_f32).exp()).abs() < tol);
        assert!((target[2] - 0.5_f32.exp()).abs() < tol);
        assert!((target[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn both_reset_to_initial() {
        // GDN2 reset → 0.99
        let mut gdn2 = Gdn2DiagonalGate::from_alpha(vec![0.5, 0.5, 0.5, 0.5]);
        gdn2.reset();
        for v in &gdn2.alpha {
            assert!((v - 0.99).abs() < 1e-6);
        }

        // Wall reset → 0.0
        let mut wall = WallDiagonalGate::new(HD);
        wall.prefix[0] = 5.0;
        wall.prefix[3] = -2.0;
        wall.reset();
        for v in &wall.prefix {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn dim_matches_input() {
        let gdn2 = Gdn2DiagonalGate::new(8);
        assert_eq!(gdn2.dim(), 8);

        let wall = WallDiagonalGate::new(16);
        assert_eq!(wall.dim(), 16);
    }
}
