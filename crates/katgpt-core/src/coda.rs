//! CODA-inspired fused SIMD kernels (Research 67).
//!
//! Algebraic reparameterization: fuse matmul+residual+rmsnorm+activation
//! into single-pass SIMD loops, eliminating intermediate buffer writes.
//!
//! Key identity (CODA §3.2.1):
//!   RMSNorm(x@W + z) * gamma @ W' = r * ((x@W + z) * gamma) @ W'
//!
//! This lets us delay the row-wise RMSNorm scale past the next GEMM.
//!
//! # Buffer Write Savings (per layer)
//!
//! | Operation | Baseline | CODA Fused |
//! |-----------|----------|------------|
//! | out_proj → ctx.x | 1 write | 0 (fused) |
//! | residual add | 1 rmw | 0 (fused) |
//! | rmsnorm (pre-MLP) | 2 passes | 0 (delayed) |
//! | gate_up → hidden | 1 write | 0 (fused) |
//! | activation | 1 pass | 0 (fused) |
//! | down → ctx.x | 1 write | 0 (fused) |
//! | residual add | 1 rmw | 0 (fused) |
//! | **Total** | ~8 passes | ~0 passes |

use crate::simd::{simd_dot_f32, simd_sum_f32};

// ---------------------------------------------------------------------------
// Activation Enum (Plan 103 T9: generic activation dispatch)
// ---------------------------------------------------------------------------

/// Gate activation function for fused MLP kernels.
///
/// Different model architectures use different activations:
/// - Generic: ReLU (standard 2-layer MLP)
/// - LLaMA-family: SiLU (SwiGLU variant)
/// - Gemma 2: tanh-approximated GELU (GeGLU variant)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum GateActivation {
    /// ReLU: max(0, x). Used in standard 2-layer MLP.
    #[default]
    Relu,
    /// SiLU/Swish: x * sigmoid(x). Used in LLaMA, Mistral SwiGLU.
    Silu,
    /// Tanh-approximated GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))).
    /// Used in Gemma 2 GeGLU.
    GegeluTanh,
    /// Approximated GELU using sigmoid: x * sigmoid(1.702 * x).
    /// Used in standard GeGLU.
    Gegelu,
}

impl GateActivation {
    /// Apply the activation function to a single value.
    #[inline(always)]
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            Self::Relu => x.max(0.0),
            Self::Silu => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                x * sigmoid
            }
            Self::GegeluTanh => {
                // Precomputed: sqrt(2/π) ≈ 0.7978845608
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                let inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            }
            Self::Gegelu => {
                let sigmoid = 1.0 / (1.0 + (-1.702 * x).exp());
                x * sigmoid
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T1–T4: MoA Inference — Token-Adaptive Activation Mixing (Plan 158)
// Gated behind `moa_inference` feature. When OFF, all MoA code eliminated.
// ---------------------------------------------------------------------------

#[cfg(feature = "moa_inference")]
/// Number of activations in the MoA dictionary (fixed at 7).
const MOA_DICT_SIZE: usize = 7;

/// MoA (Mixture of Activations) dictionary entries.
///
/// Each activation is used in the token-adaptive mixing context where
/// gating weights are computed per-token from input features.
/// The mixing produces: Σ_k ρ_k σ_k(y) ⊙ Σ_ℓ π_ℓ σ_ℓ(z)
///
/// Separate from [`GateActivation`] since MoA activations are only used
/// in the mixing context (Plan 158, Research 126).
#[cfg(feature = "moa_inference")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum MoaActivation {
    /// Identity: σ(x) = x
    #[default]
    Id,
    /// ReLU: σ(x) = max(0, x)
    Relu,
    /// ReLU²: σ(x) = max(0, x)²
    Relu2,
    /// Leaky ReLU: σ(x) = max(x, ηx), η = 0.01
    LeakyRelu,
    /// GELU: σ(x) = xΦ(x) (sigmoid approximation)
    Gelu,
    /// SiLU/Swish: σ(x) = x · sigmoid(x)
    Silu,
    /// Tanh: σ(x) = tanh(x)
    Tanh,
}

#[cfg(feature = "moa_inference")]
impl MoaActivation {
    /// Apply the MoA activation function to a single value.
    #[inline(always)]
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            Self::Id => x,
            Self::Relu => x.max(0.0),
            Self::Relu2 => {
                let r = x.max(0.0);
                r * r
            }
            Self::LeakyRelu => {
                const ETA: f32 = 0.01;
                if x >= 0.0 { x } else { ETA * x }
            }
            Self::Gelu => {
                // Sigmoid approximation: x * sigmoid(1.702 * x)
                let sigmoid = 1.0 / (1.0 + (-1.702 * x).exp());
                x * sigmoid
            }
            Self::Silu => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                x * sigmoid
            }
            Self::Tanh => x.tanh(),
        }
    }

    /// All MoA activations in canonical order (matches weight layout).
    #[inline(always)]
    pub const fn all() -> [MoaActivation; 7] {
        [
            MoaActivation::Id,
            MoaActivation::Relu,
            MoaActivation::Relu2,
            MoaActivation::LeakyRelu,
            MoaActivation::Gelu,
            MoaActivation::Silu,
            MoaActivation::Tanh,
        ]
    }
}

// ---------------------------------------------------------------------------
// T2: MoA Config Weight Struct (Plan 158)
// ---------------------------------------------------------------------------

/// MoA gating parameters for a single FFN layer.
/// Present only when model was trained with MoA.
///
/// If `None` in weight struct → fall back to standard SwiGLU. Zero regression.
///
/// Weight layout:
/// - `gate_gating`: `[MOA_DICT_SIZE × d_model]` — u_k for each σ_k (gate branch)
/// - `up_gating`: `[MOA_DICT_SIZE × d_model]` — v_ℓ for each σ_ℓ (up branch)
#[cfg(feature = "moa_inference")]
pub struct MoaConfig {
    /// Input dimension d_model
    pub d_model: usize,
    /// Gate branch mixing params: [MOA_DICT_SIZE × d_model]
    pub gate_gating: Vec<f32>,
    /// Up branch mixing params: [MOA_DICT_SIZE × d_model]
    pub up_gating: Vec<f32>,
}

#[cfg(feature = "moa_inference")]
impl MoaConfig {
    /// Create a new MoaConfig with the given dimensions.
    ///
    /// Gate and up gating vectors are initialized to zero (uniform mixing
    /// after sigmoid → each activation gets equal weight).
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            gate_gating: vec![0.0; MOA_DICT_SIZE * d_model],
            up_gating: vec![0.0; MOA_DICT_SIZE * d_model],
        }
    }
}

/// Zero-sized MoA sentinel for when feature is OFF.
/// Eliminated entirely by the compiler.
#[cfg(not(feature = "moa_inference"))]
pub type MoaConfig = ();

// ---------------------------------------------------------------------------
// T3: MoA SwiGLU Forward Pass (Plan 158)
// ---------------------------------------------------------------------------

/// Compute MoA gating weights: π_k = sigmoid(u_k^T x) for k in [0..MOA_DICT_SIZE).
///
/// Returns array of MOA_DICT_SIZE gating weights.
///
/// NOTE: Uses sigmoid, NOT softmax. Paper (arXiv 2605.26647) Table 2 shows
/// sigmoid gating > softmax > tanh. Do not change to softmax.
#[cfg(feature = "moa_inference")]
#[inline(always)]
fn compute_moa_gates(input: &[f32], gating: &[f32], d_model: usize) -> [f32; MOA_DICT_SIZE] {
    let mut weights = [0.0f32; MOA_DICT_SIZE];
    for (k, w) in weights.iter_mut().enumerate() {
        let offset = k * d_model;
        let dot = simd_dot_f32(&gating[offset..offset + d_model], input, d_model);
        *w = 1.0 / (1.0 + (-dot).exp()); // sigmoid
    }
    weights
}

/// Apply k-th MoA activation from the dictionary.
///
/// # Performance
/// This is a convenience function. For tight inner loops, hoist
/// `let activations = MoaActivation::all();` outside the loop and call
/// `activations[k].activate(x)` directly to avoid repeated array construction.
#[cfg(feature = "moa_inference")]
#[inline(always)]
#[allow(dead_code)]
fn moa_activate(k: usize, x: f32, activations: &[MoaActivation; 7]) -> f32 {
    activations[k].activate(x)
}

/// Token-adaptive bi-MoA SwiGLU forward pass.
///
/// Computes: hidden[i] = (Σ_k ρ_k σ_k(gate_proj[i])) * (Σ_ℓ π_ℓ σ_ℓ(up_proj[i]))
///
/// where ρ_k = sigmoid(u_k^T input), π_ℓ = sigmoid(v_ℓ^T input) are
/// token-adaptive gating weights computed from the input.
///
/// NOTE: Uses sigmoid gating, NOT softmax. Paper (arXiv 2605.26647) Table 2:
/// sigmoid gating > softmax > tanh. Do not change to softmax.
///
/// # Arguments
///
/// * `hidden` - Output buffer `[d_ffn]`
/// * `gate_proj` - Gate projection W₁x `[d_ffn]`
/// * `up_proj` - Up projection W₂x `[d_ffn]`
/// * `input` - Original input x `[d_model]` for gating computation
/// * `moa` - MoA gating configuration
#[cfg(feature = "moa_inference")]
#[inline(always)]
pub fn moa_swiglu(
    hidden: &mut [f32],
    gate_proj: &[f32],
    up_proj: &[f32],
    input: &[f32],
    moa: &MoaConfig,
) {
    let d = moa.d_model;
    let n = gate_proj.len(); // d_ffn

    debug_assert!(up_proj.len() >= n, "up_proj too short");
    debug_assert!(hidden.len() >= n, "hidden too short");
    debug_assert!(input.len() >= d, "input too short");

    // Compute token-adaptive gating weights: π_k = sigmoid(u_k^T x)
    let gate_weights = compute_moa_gates(input, &moa.gate_gating, d);
    let up_weights = compute_moa_gates(input, &moa.up_gating, d);

    // Mixed activations: Σ_k ρ_k σ_k(y) ⊙ Σ_ℓ π_ℓ σ_ℓ(z)
    // Per-element to avoid extra allocation
    let activations = MoaActivation::all();
    for i in 0..n {
        let y = gate_proj[i];
        let z = up_proj[i];
        let mut mixed_gate = 0.0f32;
        let mut mixed_up = 0.0f32;
        for j in 0..MOA_DICT_SIZE {
            mixed_gate += gate_weights[j] * activations[j].activate(y);
            mixed_up += up_weights[j] * activations[j].activate(z);
        }
        hidden[i] = mixed_gate * mixed_up;
    }
}

// ---------------------------------------------------------------------------
// T4: SIMD Fused Kernel — matmul + MoA mixing + RMS scale (Plan 158)
// ---------------------------------------------------------------------------

/// Fused kernel: matmul + delayed RMSNorm scale + MoA token-adaptive activation.
///
/// For paired rows in the weight matrix:
/// ```text
/// gate[i] = dot(W_gate[i], input) * rstd
/// up[i]   = dot(W_up[i], input)   * rstd
/// output[i] = (Σ_k ρ_k σ_k(gate[i])) * (Σ_ℓ π_ℓ σ_ℓ(up[i]))
/// ```
///
/// where ρ_k = sigmoid(u_k^T input), π_ℓ = sigmoid(v_ℓ^T input) are computed
/// from the MoA gating vectors and the input.
///
/// NOTE: Uses sigmoid gating, NOT softmax. Paper (arXiv 2605.26647) Table 2:
/// sigmoid gating > softmax > tanh. Do not change to softmax.
///
/// This fuses 4 operations:
/// 1. **Matmul**: gate and up projections from combined weight
/// 2. **Delayed RMS scale**: multiply by `rstd`
/// 3. **MoA gating**: sigmoid(u_k^T input) for each dictionary entry
/// 4. **MoA mixing**: weighted sum over activation dictionary
///
/// The MoA mixing is O(d_ffn × |K|) = O(4d × 7) = O(28d) — negligible vs O(d²) matmul.
///
/// # Weight Layout
///
/// `weight` has shape `[2 * output_dim, input_dim]` in row-major order:
/// - Rows `[0..output_dim]` = gate projection
/// - Rows `[output_dim..2*output_dim]` = up projection
///
/// # Arguments
///
/// * `output` - Output buffer `[output_dim]`
/// * `weight` - Combined gate+up weight `[2 * output_dim * input_dim]`
/// * `input` - Input vector (gamma-scaled O from previous kernel) `[input_dim]`
/// * `rstd` - Inverse RMS scale from [`compute_rstd`]
/// * `moa` - MoA gating configuration (gating vectors + d_model)
/// * `output_dim` - Output dimension (weight has `2 * output_dim` rows)
/// * `input_dim` - Input dimension (weight cols)
#[cfg(feature = "moa_inference")]
#[inline(always)]
pub fn simd_matmul_rmsnorm_moa_swiglu(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rstd: f32,
    moa: &MoaConfig,
    output_dim: usize,
    input_dim: usize,
) {
    debug_assert!(output.len() >= output_dim, "output too short");
    debug_assert!(
        weight.len() >= 2 * output_dim * input_dim,
        "weight too short"
    );
    debug_assert!(input.len() >= input_dim, "input too short");

    let d = moa.d_model;
    debug_assert!(d <= input_dim, "d_model must be <= input_dim");

    // Compute token-adaptive gating weights from input (using first d_model elements)
    let gate_weights = compute_moa_gates(&input[..d], &moa.gate_gating, d);
    let up_weights = compute_moa_gates(&input[..d], &moa.up_gating, d);

    let activations = MoaActivation::all();
    for i in 0..output_dim {
        // Gate projection: rows [0..output_dim]
        let gate_off = i * input_dim;
        let gate_val = simd_dot_f32(
            unsafe { weight.get_unchecked(gate_off..gate_off + input_dim) },
            input,
            input_dim,
        ) * rstd;

        // Up projection: rows [output_dim..2*output_dim]
        let up_off = (output_dim + i) * input_dim;
        let up_val = simd_dot_f32(
            unsafe { weight.get_unchecked(up_off..up_off + input_dim) },
            input,
            input_dim,
        ) * rstd;

        // MoA mixed activation: (Σ_k ρ_k σ_k(gate)) * (Σ_ℓ π_ℓ σ_ℓ(up))
        let mut mixed_gate = 0.0f32;
        let mut mixed_up = 0.0f32;
        for j in 0..MOA_DICT_SIZE {
            mixed_gate += gate_weights[j] * activations[j].activate(gate_val);
            mixed_up += up_weights[j] * activations[j].activate(up_val);
        }

        unsafe {
            *output.get_unchecked_mut(i) = mixed_gate * mixed_up;
        }
    }
}

// ---------------------------------------------------------------------------
// T3: Fused matmul + residual + partial RMS + gamma scaling
// ---------------------------------------------------------------------------

/// Fused kernel: matmul + residual + partial RMS accumulation + gamma scaling.
///
/// For each output element `i` in `[0..rows)`:
/// ```text
/// d[i] = dot(W_row[i], input) + bias[i] + residual[i]
/// partial_sums[i / block_size] += d[i]²
/// o[i] = d[i] * gamma[i]   (or d[i] if gamma is None)
/// ```
///
/// This fuses 4 operations into one SIMD loop:
/// 1. **Matmul**: dot product per output element
/// 2. **Residual add**: fused into the accumulation
/// 3. **RMS accumulation**: partial mean-square for later [`compute_rstd`]
/// 4. **Gamma scaling**: element-wise norm weight multiplication
///
/// # Buffer Layout
///
/// - `output_d`: receives D = matmul + residual (unscaled), typically stored as `xr2`
/// - `output_o`: receives O = D * gamma, typically stored as `x` (input to next matmul)
/// - `partial_sums`: accumulated sum of squares, length >= `ceil(rows / block_size)`
///
/// # Arguments
///
/// * `output_d` - Output buffer for unscaled D `[rows]`
/// * `output_o` - Output buffer for gamma-scaled O `[rows]`
/// * `partial_sums` - RMS accumulation buffer, zeroed internally `[n_blocks]`
/// * `weight` - Weight matrix `[rows * cols]` row-major
/// * `input` - Input vector `[cols]`
/// * `residual` - Residual to add `[rows]`
/// * `gamma` - Optional norm weight `[rows]`, None = identity (all 1s)
/// * `bias` - Optional bias `[rows]`, None = zero (for LoRA integration, T10)
/// * `rows` - Number of output elements (weight rows)
/// * `cols` - Input dimension (weight cols)
/// * `block_size` - Elements per partial_sum block (use `rows` for single block)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn simd_matmul_residual_partial_rms(
    output_d: &mut [f32],
    output_o: &mut [f32],
    partial_sums: &mut [f32],
    weight: &[f32],
    input: &[f32],
    residual: &[f32],
    gamma: Option<&[f32]>,
    bias: Option<&[f32]>,
    rows: usize,
    cols: usize,
    block_size: usize,
) {
    debug_assert!(output_d.len() >= rows, "output_d too short");
    debug_assert!(output_o.len() >= rows, "output_o too short");
    let bs = block_size.max(1);
    let n_blocks = rows.div_ceil(bs);
    debug_assert!(partial_sums.len() >= n_blocks, "partial_sums too short");
    if let Some(g) = gamma {
        debug_assert!(g.len() >= rows, "gamma too short");
    }
    if let Some(b) = bias {
        debug_assert!(b.len() >= rows, "bias too short");
    }

    // Zero partial sums for fresh accumulation
    partial_sums[..n_blocks].fill(0.0);

    let mut block_counter = 0usize;
    let mut block_idx = 0usize;
    for i in 0..rows {
        let row_off = i * cols;
        let acc = simd_dot_f32(
            unsafe { weight.get_unchecked(row_off..row_off + cols) },
            input,
            cols,
        );

        let b = bias.map_or(0.0, |b| unsafe { *b.get_unchecked(i) });
        let r = unsafe { *residual.get_unchecked(i) };
        let d = acc + b + r;

        // Accumulate partial RMS (sum of squares, divided by n later in compute_rstd)
        unsafe {
            *partial_sums.get_unchecked_mut(block_idx) += d * d;
        }
        block_counter += 1;
        if block_counter == bs {
            block_counter = 0;
            block_idx += 1;
        }

        // Gamma scaling (identity if gamma is None)
        let g = gamma.map_or(1.0, |g| unsafe { *g.get_unchecked(i) });
        unsafe {
            *output_d.get_unchecked_mut(i) = d;
            *output_o.get_unchecked_mut(i) = d * g;
        }
    }
}

// ---------------------------------------------------------------------------
// T4: Compute rstd from partial sums
// ---------------------------------------------------------------------------

/// Compute inverse RMS (rstd) from partial sums.
///
/// `rstd = 1 / sqrt(sum(partial_sums) / n_elements + eps)`
///
/// This is the "auxiliary reduction" from CODA §3.2.1 — tiny compared to
/// a full RMSNorm kernel. For BS=1 decode with `n_blocks` blocks, this is
/// O(n_blocks), typically O(1) to O(n_embd / 4).
///
/// # Arguments
///
/// * `partial_sums` - Accumulated sum of squares from [`simd_matmul_residual_partial_rms`]
/// * `n_elements` - Total number of elements (D vector length) for mean computation
/// * `eps` - Epsilon for numerical stability (typically 1e-5)
///
/// # Returns
///
/// The scalar `rstd` value: `1 / sqrt(mean_sq + eps)`
#[inline(always)]
pub fn compute_rstd(partial_sums: &[f32], n_elements: usize, eps: f32) -> f32 {
    if partial_sums.is_empty() || n_elements == 0 {
        return 1.0;
    }
    let sum_sq: f32 = simd_sum_f32(partial_sums);
    let mean_sq = sum_sq / n_elements as f32;
    1.0 / (mean_sq + eps).sqrt()
}

// ---------------------------------------------------------------------------
// T5: Fused matmul + delayed RMS scale + SwiGLU/GeGLU activation
// ---------------------------------------------------------------------------

/// Fused kernel: matmul + delayed RMSNorm scale + gated activation (SwiGLU/GeGLU).
///
/// For paired rows in the weight matrix:
/// ```text
/// gate[i] = dot(W_gate[i], input) * rstd    // gate projection + delayed RMS
/// up[i]   = dot(W_up[i], input)   * rstd    // up projection + delayed RMS
/// output[i] = activation(gate[i]) * up[i]   // gated activation
/// ```
///
/// This fuses 3 operations:
/// 1. **Matmul**: gate and up projections from combined weight
/// 2. **Delayed RMS scale**: multiply by `rstd` (from [`compute_rstd`])
/// 3. **Gated activation**: SiLU, GeGLU, or ReLU depending on model architecture
///
/// # Weight Layout
///
/// `weight` has shape `[2 * output_dim, input_dim]` in row-major order:
/// - Rows `[0..output_dim]` = gate projection
/// - Rows `[output_dim..2*output_dim]` = up projection
///
/// # Arguments
///
/// * `output` - Output buffer `[output_dim]` (half the weight rows)
/// * `weight` - Combined gate+up weight `[2 * output_dim * input_dim]`
/// * `input` - Input vector (gamma-scaled O from previous kernel) `[input_dim]`
/// * `rstd` - Inverse RMS scale from [`compute_rstd`]
/// * `activation` - Gate activation function
/// * `output_dim` - Output dimension (weight has `2 * output_dim` rows)
/// * `input_dim` - Input dimension (weight cols)
#[inline(always)]
pub fn simd_matmul_rmsnorm_swiglu(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rstd: f32,
    activation: GateActivation,
    output_dim: usize,
    input_dim: usize,
) {
    debug_assert!(output.len() >= output_dim, "output too short");
    debug_assert!(
        weight.len() >= 2 * output_dim * input_dim,
        "weight too short"
    );
    debug_assert!(input.len() >= input_dim, "input too short");

    for i in 0..output_dim {
        // Gate projection: rows [0..output_dim]
        let gate_off = i * input_dim;
        let gate_val = simd_dot_f32(
            unsafe { weight.get_unchecked(gate_off..gate_off + input_dim) },
            input,
            input_dim,
        ) * rstd;

        // Up projection: rows [output_dim..2*output_dim]
        let up_off = (output_dim + i) * input_dim;
        let up_val = simd_dot_f32(
            unsafe { weight.get_unchecked(up_off..up_off + input_dim) },
            input,
            input_dim,
        ) * rstd;

        // Gated activation: output = activation(gate) * up
        unsafe {
            *output.get_unchecked_mut(i) = activation.activate(gate_val) * up_val;
        }
    }
}

// ---------------------------------------------------------------------------
// T5b: Fused matmul + delayed RMS scale + activation (non-gated MLP)
// ---------------------------------------------------------------------------

/// Fused kernel: matmul + delayed RMSNorm scale + activation (standard MLP).
///
/// For each output element `i`:
/// ```text
/// output[i] = activation(dot(W_row[i], input) * rstd)
/// ```
///
/// Use this for standard 2-layer MLPs (no gate/up split) with delayed RMS.
/// For gated MLPs (SwiGLU/GeGLU), use [`simd_matmul_rmsnorm_swiglu`] instead.
///
/// # Arguments
///
/// * `output` - Output buffer `[rows]`
/// * `weight` - Weight matrix `[rows * cols]` row-major
/// * `input` - Input vector (gamma-scaled O from previous kernel) `[cols]`
/// * `rstd` - Inverse RMS scale from [`compute_rstd`]
/// * `activation` - Activation function (ReLU, SiLU, etc.)
/// * `rows` - Number of output elements (weight rows)
/// * `cols` - Input dimension (weight cols)
#[inline(always)]
pub fn simd_matmul_rmsnorm_activation(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rstd: f32,
    activation: GateActivation,
    rows: usize,
    cols: usize,
) {
    debug_assert!(output.len() >= rows, "output too short");
    debug_assert!(weight.len() >= rows * cols, "weight too short");
    debug_assert!(input.len() >= cols, "input too short");

    for i in 0..rows {
        let row_off = i * cols;
        let acc = simd_dot_f32(
            unsafe { weight.get_unchecked(row_off..row_off + cols) },
            input,
            cols,
        );
        unsafe {
            *output.get_unchecked_mut(i) = activation.activate(acc * rstd);
        }
    }
}

// ---------------------------------------------------------------------------
// T6: Fused matmul + residual
// ---------------------------------------------------------------------------

/// Fused kernel: matmul + residual add.
///
/// For each output element `i`:
/// ```text
/// output[i] = dot(W_row[i], input) + residual[i]
/// ```
///
/// This fuses the down-projection and residual add-back into one pass,
/// eliminating the intermediate buffer write for the matmul output.
///
/// # Arguments
///
/// * `output` - Output buffer `[rows]`
/// * `weight` - Weight matrix `[rows * cols]` row-major
/// * `input` - Input vector `[cols]`
/// * `residual` - Residual to add `[rows]`
/// * `rows` - Number of output elements (weight rows)
/// * `cols` - Input dimension (weight cols)
#[inline(always)]
pub fn simd_matmul_residual(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    residual: &[f32],
    rows: usize,
    cols: usize,
) {
    debug_assert!(output.len() >= rows, "output too short");
    debug_assert!(weight.len() >= rows * cols, "weight too short");
    debug_assert!(input.len() >= cols, "input too short");
    debug_assert!(residual.len() >= rows, "residual too short");

    for i in 0..rows {
        let row_off = i * cols;
        let acc = simd_dot_f32(
            unsafe { weight.get_unchecked(row_off..row_off + cols) },
            input,
            cols,
        );
        unsafe {
            *output.get_unchecked_mut(i) = acc + *residual.get_unchecked(i);
        }
    }
}

// ---------------------------------------------------------------------------
// T7: Fused matmul + delayed RMS scale + RoPE rotation (stretch)
// ---------------------------------------------------------------------------

/// Fused kernel: matmul + delayed RMSNorm scale + RoPE rotation.
///
/// For paired rows `(2i, 2i+1)` representing adjacent feature dimensions:
/// ```text
/// q_even = dot(W[2i], input) * rstd
/// q_odd  = dot(W[2i+1], input) * rstd
/// cos_val = cos_table[pos * head_dim + i % head_dim]
/// sin_val = sin_table[pos * head_dim + i % head_dim]
/// output[2i]   = q_even * cos_val - q_odd * sin_val
/// output[2i+1] = q_even * sin_val + q_odd * cos_val
/// ```
///
/// This fuses QKV projection + delayed RMS + RoPE into one pass per head.
///
/// # Arguments
///
/// * `output` - Output buffer `[rows]`
/// * `weight` - Weight matrix `[rows * cols]` row-major
/// * `input` - Input vector `[cols]`
/// * `rstd` - Inverse RMS scale from [`compute_rstd`]
/// * `cos_table` - Precomputed cosine values `[max_seq_len * head_dim]`
/// * `sin_table` - Precomputed sine values `[max_seq_len * head_dim]`
/// * `rows` - Total output elements (must be even)
/// * `cols` - Input dimension
/// * `pos` - Current position in sequence
/// * `head_dim` - Dimension per attention head
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn simd_matmul_rmsnorm_rope(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rstd: f32,
    cos_table: &[f32],
    sin_table: &[f32],
    rows: usize,
    cols: usize,
    pos: usize,
    head_dim: usize,
) {
    debug_assert!(output.len() >= rows, "output too short");
    debug_assert!(weight.len() >= rows * cols, "weight too short");
    debug_assert!(input.len() >= cols, "input too short");
    debug_assert!(
        rows.is_multiple_of(2),
        "rows must be even for paired RoPE features"
    );

    let half_rows = rows / 2;
    let pos_base = pos * head_dim;
    let mut dim_idx = 0usize;
    for i in 0..half_rows {
        let even_row = 2 * i;
        let odd_row = 2 * i + 1;

        // Matmul for paired features
        let even_off = even_row * cols;
        let q_even = simd_dot_f32(
            unsafe { weight.get_unchecked(even_off..even_off + cols) },
            input,
            cols,
        ) * rstd;

        let odd_off = odd_row * cols;
        let q_odd = simd_dot_f32(
            unsafe { weight.get_unchecked(odd_off..odd_off + cols) },
            input,
            cols,
        ) * rstd;

        // RoPE rotation: index into precomputed table
        let rope_idx = pos_base + dim_idx;
        dim_idx += 1;
        if dim_idx == head_dim {
            dim_idx = 0;
        }
        debug_assert!(rope_idx < cos_table.len(), "RoPE cos index out of bounds");
        debug_assert!(rope_idx < sin_table.len(), "RoPE sin index out of bounds");
        // Safety: tables are pre-sized to max_seq_len × head_dim; index verified above
        let (cos_val, sin_val) = unsafe {
            (
                *cos_table.get_unchecked(rope_idx),
                *sin_table.get_unchecked(rope_idx),
            )
        };

        unsafe {
            *output.get_unchecked_mut(even_row) = q_even * cos_val - q_odd * sin_val;
            *output.get_unchecked_mut(odd_row) = q_even * sin_val + q_odd * cos_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference matmul: output[i] = dot(W_row[i], input)
    fn ref_matmul(weight: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut output = vec![0.0; rows];
        for (i, row) in output.iter_mut().enumerate() {
            let row_off = i * cols;
            for j in 0..cols {
                *row += weight[row_off + j] * input[j];
            }
        }
        output
    }

    /// Reference RMSNorm: x[i] *= rstd. Returns rstd.
    fn ref_rmsnorm(x: &mut [f32], eps: f32) -> f32 {
        let n = x.len() as f32;
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let mean_sq = sum_sq / n;
        let rstd = 1.0 / (mean_sq + eps).sqrt();
        for v in x.iter_mut() {
            *v *= rstd;
        }
        rstd
    }

    #[test]
    fn test_matmul_residual_correctness() {
        let rows = 8;
        let cols = 4;
        let weight: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
        let input: Vec<f32> = (0..cols).map(|i| (i + 1) as f32).collect();
        let residual: Vec<f32> = (0..rows).map(|i| (i + 1) as f32 * 0.5).collect();

        let mut output = vec![0.0; rows];
        simd_matmul_residual(&mut output, &weight, &input, &residual, rows, cols);

        let matmul_out = ref_matmul(&weight, &input, rows, cols);
        for i in 0..rows {
            let expected = matmul_out[i] + residual[i];
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "Mismatch at {i}: got {}, expected {expected}",
                output[i]
            );
        }
    }

    #[test]
    fn test_compute_rstd_correctness() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let eps = 1e-5;
        let n = values.len();

        let sum_sq: f32 = values.iter().map(|v| v * v).sum();
        let mean_sq = sum_sq / n as f32;
        let expected_rstd = 1.0 / (mean_sq + eps).sqrt();

        let partial_sums = [sum_sq];
        let computed_rstd = compute_rstd(&partial_sums, n, eps);

        assert!(
            (computed_rstd - expected_rstd).abs() < 1e-7,
            "rstd mismatch: got {computed_rstd}, expected {expected_rstd}"
        );
    }

    #[test]
    fn test_matmul_residual_partial_rms_correctness() {
        let rows = 8;
        let cols = 4;
        let eps = 1e-5;
        let weight: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
        let input: Vec<f32> = (0..cols).map(|i| (i + 1) as f32).collect();
        let residual: Vec<f32> = (0..rows).map(|i| (i + 1) as f32 * 0.5).collect();
        let gamma: Vec<f32> = (0..rows).map(|i| 1.0 + i as f32 * 0.1).collect();

        let mut output_d = vec![0.0; rows];
        let mut output_o = vec![0.0; rows];
        let mut partial_sums = vec![0.0; 1];

        simd_matmul_residual_partial_rms(
            &mut output_d,
            &mut output_o,
            &mut partial_sums,
            &weight,
            &input,
            &residual,
            Some(&gamma),
            None,
            rows,
            cols,
            rows,
        );

        // Verify D = matmul + residual
        let matmul_out = ref_matmul(&weight, &input, rows, cols);
        for i in 0..rows {
            let expected_d = matmul_out[i] + residual[i];
            assert!(
                (output_d[i] - expected_d).abs() < 1e-5,
                "D mismatch at {i}: got {}, expected {expected_d}",
                output_d[i]
            );
        }

        // Verify O = D * gamma
        for i in 0..rows {
            let expected_o = output_d[i] * gamma[i];
            assert!(
                (output_o[i] - expected_o).abs() < 1e-5,
                "O mismatch at {i}: got {}, expected {expected_o}",
                output_o[i]
            );
        }

        // Verify rstd matches reference rmsnorm
        let rstd = compute_rstd(&partial_sums, rows, eps);
        let mut d_copy = output_d.clone();
        let ref_rstd = ref_rmsnorm(&mut d_copy, eps);
        assert!(
            (rstd - ref_rstd).abs() < 1e-5,
            "rstd mismatch: got {rstd}, expected {ref_rstd}"
        );
    }

    #[test]
    fn test_matmul_rmsnorm_swiglu_correctness() {
        let output_dim = 4;
        let input_dim = 4;
        let weight: Vec<f32> = (0..2 * output_dim * input_dim)
            .map(|i| i as f32 * 0.1)
            .collect();
        let input: Vec<f32> = (0..input_dim).map(|i| (i + 1) as f32).collect();
        let rstd = 0.5;

        let mut output = vec![0.0; output_dim];
        simd_matmul_rmsnorm_swiglu(
            &mut output,
            &weight,
            &input,
            rstd,
            GateActivation::Silu,
            output_dim,
            input_dim,
        );

        // Verify first element manually
        let gate_off = 0;
        let gate_val: f32 = weight[gate_off..gate_off + input_dim]
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>()
            * rstd;
        let up_off = output_dim * input_dim;
        let up_val: f32 = weight[up_off..up_off + input_dim]
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>()
            * rstd;
        let silu = gate_val / (1.0 + (-gate_val).exp());
        let expected = silu * up_val;

        assert!(
            (output[0] - expected).abs() < 1e-4,
            "SwiGLU mismatch at 0: got {}, expected {expected}",
            output[0]
        );
    }

    #[test]
    fn test_matmul_rmsnorm_activation_relu() {
        let rows = 8;
        let cols = 4;
        let weight: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1 - 0.2).collect();
        let input: Vec<f32> = (0..cols).map(|i| (i + 1) as f32).collect();
        let rstd = 1.0;

        let mut output = vec![0.0; rows];
        simd_matmul_rmsnorm_activation(
            &mut output,
            &weight,
            &input,
            rstd,
            GateActivation::Relu,
            rows,
            cols,
        );

        // Verify: output[i] = max(0, dot(W_row[i], input) * rstd)
        let matmul_out = ref_matmul(&weight, &input, rows, cols);
        for i in 0..rows {
            let expected = (matmul_out[i] * rstd).max(0.0);
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "Relu mismatch at {i}: got {}, expected {expected}",
                output[i]
            );
        }
    }

    #[test]
    fn test_matmul_rmsnorm_rope_correctness() {
        let rows = 4;
        let cols = 4;
        let head_dim = 2;
        let pos = 0;

        let weight: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
        let input: Vec<f32> = (0..cols).map(|i| (i + 1) as f32).collect();
        let cos_table = vec![1.0, 0.5, 1.0, 0.5];
        let sin_table = vec![0.0, 0.866, 0.0, 0.866];
        let rstd = 1.0;

        let mut output = vec![0.0; rows];
        simd_matmul_rmsnorm_rope(
            &mut output,
            &weight,
            &input,
            rstd,
            &cos_table,
            &sin_table,
            rows,
            cols,
            pos,
            head_dim,
        );

        // Verify first pair: rows 0 and 1
        let q_even = simd_dot_f32(&weight[0..cols], &input, cols) * rstd;
        let q_odd = simd_dot_f32(&weight[cols..2 * cols], &input, cols) * rstd;
        let cos_val = cos_table[0];
        let sin_val = sin_table[0];
        let expected_0 = q_even * cos_val - q_odd * sin_val;
        let expected_1 = q_even * sin_val + q_odd * cos_val;

        assert!(
            (output[0] - expected_0).abs() < 1e-5,
            "RoPE[0] mismatch: got {}, expected {expected_0}",
            output[0]
        );
        assert!(
            (output[1] - expected_1).abs() < 1e-5,
            "RoPE[1] mismatch: got {}, expected {expected_1}",
            output[1]
        );
    }

    #[test]
    fn test_no_gamma_equals_plain() {
        let rows = 8;
        let cols = 4;
        let weight: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
        let input: Vec<f32> = (0..cols).map(|i| (i + 1) as f32).collect();
        let residual: Vec<f32> = (0..rows).map(|i| (i + 1) as f32 * 0.5).collect();

        let mut output_d = vec![0.0; rows];
        let mut output_o = vec![0.0; rows];
        let mut partial_sums = vec![0.0; 1];

        simd_matmul_residual_partial_rms(
            &mut output_d,
            &mut output_o,
            &mut partial_sums,
            &weight,
            &input,
            &residual,
            None,
            None,
            rows,
            cols,
            rows,
        );

        // Without gamma, O should equal D
        for i in 0..rows {
            assert!(
                (output_d[i] - output_o[i]).abs() < 1e-7,
                "No-gamma mismatch at {i}: D={}, O={}",
                output_d[i],
                output_o[i]
            );
        }
    }

    #[test]
    fn test_bias_applied() {
        let rows = 4;
        let cols = 2;
        let weight: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
        let input: Vec<f32> = (0..cols).map(|i| (i + 1) as f32).collect();
        let residual: Vec<f32> = vec![0.0; rows];
        let bias: Vec<f32> = (0..rows).map(|i| (i + 1) as f32 * 0.3).collect();

        let mut with_bias = vec![0.0; rows];
        let mut without_bias = vec![0.0; rows];
        let mut partial_sums = vec![0.0; 1];

        simd_matmul_residual_partial_rms(
            &mut with_bias,
            &mut vec![0.0; rows],
            &mut partial_sums,
            &weight,
            &input,
            &residual,
            None,
            Some(&bias),
            rows,
            cols,
            rows,
        );
        partial_sums[0] = 0.0;
        simd_matmul_residual_partial_rms(
            &mut without_bias,
            &mut vec![0.0; rows],
            &mut partial_sums,
            &weight,
            &input,
            &residual,
            None,
            None,
            rows,
            cols,
            rows,
        );

        for i in 0..rows {
            let diff = with_bias[i] - without_bias[i];
            assert!(
                (diff - bias[i]).abs() < 1e-5,
                "Bias mismatch at {i}: diff={diff}, expected={}",
                bias[i]
            );
        }
    }

    #[test]
    fn test_gate_activation_values() {
        // SiLU(0) = 0
        assert!((GateActivation::Silu.activate(0.0)).abs() < 1e-7);
        // SiLU(1) ≈ 0.7311
        let silu_1 = GateActivation::Silu.activate(1.0);
        assert!((silu_1 - 0.7311).abs() < 0.01);

        // ReLU(-1) = 0
        assert!((GateActivation::Relu.activate(-1.0)).abs() < 1e-7);
        // ReLU(2) = 2
        assert!((GateActivation::Relu.activate(2.0) - 2.0).abs() < 1e-7);

        // GegeluTanh(0) ≈ 0
        assert!((GateActivation::GegeluTanh.activate(0.0)).abs() < 1e-7);
        // Gegelu(0) ≈ 0
        assert!((GateActivation::Gegelu.activate(0.0)).abs() < 1e-7);
    }

    #[test]
    fn test_coda_identity_end_to_end() {
        // Verify: RMSNorm(x@W + z) * gamma @ W' = r * ((x@W + z) * gamma) @ W'
        let n = 8;
        let eps = 1e-5;

        let x: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.3).collect();
        let w: Vec<f32> = (0..n * n).map(|i| i as f32 * 0.1 - 0.4).collect();
        let z: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.02).collect();
        let w_prime: Vec<f32> = (0..n * n).map(|i| i as f32 * 0.07 - 0.3).collect();

        // Baseline: RMSNorm(x@W + z) * gamma, then @ W'
        let mut d_baseline = ref_matmul(&w, &x, n, n);
        for i in 0..n {
            d_baseline[i] += z[i];
        }
        let mut d_normed = d_baseline.clone();
        ref_rmsnorm(&mut d_normed, eps);
        for i in 0..n {
            d_normed[i] *= gamma[i];
        }
        let baseline_out = ref_matmul(&w_prime, &d_normed, n, n);

        // CODA: (x@W + z) * gamma, then @ W', then * rstd
        let mut output_d = vec![0.0; n];
        let mut output_o = vec![0.0; n];
        let mut partial_sums = vec![0.0; 1];

        simd_matmul_residual_partial_rms(
            &mut output_d,
            &mut output_o,
            &mut partial_sums,
            &w,
            &x,
            &z,
            Some(&gamma),
            None,
            n,
            n,
            n,
        );

        let rstd = compute_rstd(&partial_sums, n, eps);
        let mut coda_out = ref_matmul(&w_prime, &output_o, n, n);
        for v in coda_out.iter_mut() {
            *v *= rstd;
        }

        // Verify: baseline ≈ coda (within floating-point tolerance)
        for i in 0..n {
            let diff = (baseline_out[i] - coda_out[i]).abs();
            let scale = baseline_out[i].abs().max(coda_out[i].abs()).max(1e-6);
            assert!(
                diff / scale < 1e-4,
                "CODA identity violated at {i}: baseline={}, coda={}, diff={diff}",
                baseline_out[i],
                coda_out[i]
            );
        }
    }

    #[test]
    fn test_partial_sums_multi_block() {
        let rows = 8;
        let cols = 4;
        let block_size = 2;
        let weight: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
        let input: Vec<f32> = (0..cols).map(|i| (i + 1) as f32).collect();
        let residual: Vec<f32> = vec![0.0; rows];

        let mut output_d = vec![0.0; rows];
        let mut output_o = vec![0.0; rows];
        let n_blocks = rows.div_ceil(block_size);
        let mut partial_sums = vec![0.0; n_blocks];

        simd_matmul_residual_partial_rms(
            &mut output_d,
            &mut output_o,
            &mut partial_sums,
            &weight,
            &input,
            &residual,
            None,
            None,
            rows,
            cols,
            block_size,
        );

        let sum_partial: f32 = partial_sums.iter().sum();
        let sum_d_sq: f32 = output_d.iter().map(|v| v * v).sum();
        assert!(
            (sum_partial - sum_d_sq).abs() < 1e-5,
            "Partial sums mismatch: {sum_partial} vs {sum_d_sq}"
        );
    }

    #[test]
    fn test_compute_rstd_empty() {
        assert!((compute_rstd(&[], 0, 1e-5) - 1.0).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // T5: MoA tests (Plan 158) — gated behind moa_inference
    // -----------------------------------------------------------------------

    #[cfg(feature = "moa_inference")]
    #[test]
    fn test_moa_activation_values() {
        // Id
        assert!((MoaActivation::Id.activate(2.0) - 2.0).abs() < 1e-7);
        assert!((MoaActivation::Id.activate(-3.0) - (-3.0)).abs() < 1e-7);

        // Relu
        assert!((MoaActivation::Relu.activate(2.0) - 2.0).abs() < 1e-7);
        assert!((MoaActivation::Relu.activate(-1.0)).abs() < 1e-7);

        // Relu2
        assert!((MoaActivation::Relu2.activate(3.0) - 9.0).abs() < 1e-7);
        assert!((MoaActivation::Relu2.activate(-1.0)).abs() < 1e-7);

        // LeakyRelu
        assert!((MoaActivation::LeakyRelu.activate(2.0) - 2.0).abs() < 1e-7);
        assert!((MoaActivation::LeakyRelu.activate(-2.0) - (-0.02)).abs() < 1e-7);

        // Gelu(0) ≈ 0
        assert!((MoaActivation::Gelu.activate(0.0)).abs() < 1e-7);

        // Silu(0) = 0, Silu(1) ≈ 0.7311
        assert!((MoaActivation::Silu.activate(0.0)).abs() < 1e-7);
        assert!((MoaActivation::Silu.activate(1.0) - 0.7311).abs() < 0.01);

        // Tanh(0) = 0, Tanh(1) ≈ 0.7616
        assert!((MoaActivation::Tanh.activate(0.0)).abs() < 1e-7);
        assert!((MoaActivation::Tanh.activate(1.0) - 1.0_f32.tanh()).abs() < 1e-7);
    }

    #[cfg(feature = "moa_inference")]
    #[test]
    fn test_moa_activation_dict_size() {
        assert_eq!(MoaActivation::all().len(), 7);
    }

    #[cfg(feature = "moa_inference")]
    #[test]
    fn test_moa_config_new() {
        let d = 4;
        let cfg = MoaConfig::new(d);
        assert_eq!(cfg.d_model, d);
        assert_eq!(cfg.gate_gating.len(), MOA_DICT_SIZE * d);
        assert_eq!(cfg.up_gating.len(), MOA_DICT_SIZE * d);
        // All zeros initially
        assert!(cfg.gate_gating.iter().all(|&v| v == 0.0));
        assert!(cfg.up_gating.iter().all(|&v| v == 0.0));
    }

    #[cfg(feature = "moa_inference")]
    #[test]
    fn test_compute_moa_gates_uniform() {
        // With zero gating vectors, all sigmoid(0) = 0.5
        let d = 4;
        let gating = vec![0.0f32; MOA_DICT_SIZE * d];
        let input = vec![1.0f32; d];
        let weights = compute_moa_gates(&input, &gating, d);
        for (k, &w) in weights.iter().enumerate() {
            assert!(
                (w - 0.5).abs() < 1e-6,
                "weight[{}] = {}, expected 0.5",
                k,
                w
            );
        }
    }

    #[cfg(feature = "moa_inference")]
    #[test]
    fn test_compute_moa_gates_biased() {
        // With large positive dot product → sigmoid ≈ 1
        // With large negative dot product → sigmoid ≈ 0
        let d = 2;
        let mut gating = vec![0.0f32; MOA_DICT_SIZE * d];
        // k=0: positive dot → ~1
        gating[0] = 10.0;
        gating[1] = 10.0;
        // k=1: negative dot → ~0
        gating[2] = -10.0;
        gating[3] = -10.0;
        let input = vec![1.0f32; d];
        let weights = compute_moa_gates(&input, &gating, d);
        assert!(weights[0] > 0.99, "k=0 should be ~1, got {}", weights[0]);
        assert!(weights[1] < 0.01, "k=1 should be ~0, got {}", weights[1]);
    }

    #[cfg(feature = "moa_inference")]
    #[test]
    fn test_moa_swiglu_uniform_gates() {
        // With all-zero gating → uniform 0.5 weights → equal mixing
        let d_model = 4;
        let d_ffn = 8;
        let moa = MoaConfig::new(d_model);
        let input = vec![1.0f32; d_model];
        let gate_proj = vec![1.0f32; d_ffn];
        let up_proj = vec![1.0f32; d_ffn];
        let mut hidden = vec![0.0f32; d_ffn];

        moa_swiglu(&mut hidden, &gate_proj, &up_proj, &input, &moa);

        // All outputs should be identical (same gate/up values, same weights)
        // With σ_k(1.0) for each activation, all weights = 0.5:
        // mixed_gate = 0.5 * (Id(1) + Relu(1) + Relu2(1) + LeakyRelu(1) + Gelu(1) + Silu(1) + Tanh(1))
        let activations_on_one: f32 = [
            MoaActivation::Id.activate(1.0),
            MoaActivation::Relu.activate(1.0),
            MoaActivation::Relu2.activate(1.0),
            MoaActivation::LeakyRelu.activate(1.0),
            MoaActivation::Gelu.activate(1.0),
            MoaActivation::Silu.activate(1.0),
            MoaActivation::Tanh.activate(1.0),
        ]
        .iter()
        .copied()
        .sum();
        let expected = (0.5 * activations_on_one) * (0.5 * activations_on_one);

        for (i, &h) in hidden.iter().enumerate() {
            assert!(
                (h - expected).abs() < 1e-4,
                "hidden[{}] = {}, expected {}",
                i,
                h,
                expected
            );
        }
    }

    #[cfg(feature = "moa_inference")]
    #[test]
    fn test_moa_swiglu_fallback_to_silu() {
        // When only Silu activation has weight ~1 and all others ~0,
        // MoA should behave like standard SwiGLU with SiLU.
        let d_model = 4;
        let d_ffn = 4;
        let mut moa = MoaConfig::new(d_model);

        // Set Silu (k=5) gating to large positive → sigmoid → ~1
        // Set all others to large negative → sigmoid → ~0
        for k in 0..MOA_DICT_SIZE {
            let val = if k == 5 { 10.0f32 } else { -10.0f32 };
            for j in 0..d_model {
                moa.gate_gating[k * d_model + j] = val;
                moa.up_gating[k * d_model + j] = val;
            }
        }

        let input = vec![1.0f32; d_model];
        let gate_proj = vec![2.0f32; d_ffn];
        let up_proj = vec![3.0f32; d_ffn];
        let mut hidden = vec![0.0f32; d_ffn];

        moa_swiglu(&mut hidden, &gate_proj, &up_proj, &input, &moa);

        // Should approximate: silu(2.0) * 3.0 * silu(3.0) * 3.0... no wait
        // Actually: (Σ_k ρ_k σ_k(2.0)) * (Σ_ℓ π_ℓ σ_ℓ(3.0))
        // With only Silu active: silu(2.0) * silu(3.0) -- no, up is not activated with silu
        // Hmm, let me reconsider. The MoA mixing is:
        // hidden[i] = (Σ_k ρ_k σ_k(gate[i])) * (Σ_ℓ π_ℓ σ_ℓ(up[i]))
        // With only Silu active for both gate and up:
        // hidden[i] = silu(2.0) * silu(3.0)
        let silu_2 = 2.0f32 / (1.0 + (-2.0f32).exp());
        let silu_3 = 3.0f32 / (1.0 + (-3.0f32).exp());
        let expected = silu_2 * silu_3;

        for (i, &h) in hidden.iter().enumerate() {
            assert!(
                (h - expected).abs() < 1e-3,
                "hidden[{}] = {}, expected {}",
                i,
                h,
                expected
            );
        }
    }

    #[cfg(feature = "moa_inference")]
    #[test]
    fn test_moa_swiglu_correctness_elementwise() {
        // Small test with known values, compute reference by hand
        let d_model = 2;
        let d_ffn = 3;
        let mut moa = MoaConfig::new(d_model);

        // Set specific gating values
        // k=0 (Id): positive weight
        moa.gate_gating[0] = 1.0;
        moa.gate_gating[1] = 0.0;
        moa.up_gating[0] = 0.5;
        moa.up_gating[1] = 0.5;
        // k=1 (Relu): negative weight (off)
        moa.gate_gating[2] = -10.0;
        moa.gate_gating[3] = 0.0;
        moa.up_gating[2] = -10.0;
        moa.up_gating[3] = 0.0;
        // All other k: off
        for k in 2..MOA_DICT_SIZE {
            moa.gate_gating[k * d_model] = -10.0;
            moa.gate_gating[k * d_model + 1] = -10.0;
            moa.up_gating[k * d_model] = -10.0;
            moa.up_gating[k * d_model + 1] = -10.0;
        }

        let input = vec![1.0f32, 0.0f32];
        let gate_proj = vec![2.0f32, -1.0f32, 0.5f32];
        let up_proj = vec![1.0f32, 1.0f32, 1.0f32];
        let mut hidden = vec![0.0f32; d_ffn];

        moa_swiglu(&mut hidden, &gate_proj, &up_proj, &input, &moa);

        // gate_weights: k=0: sigmoid(1.0*1 + 0*0) = sigmoid(1.0) ≈ 0.7311
        //               k=1..6: sigmoid(-10..) ≈ 0
        let w0 = 1.0f32 / (1.0 + (-1.0f32).exp()); // ≈ 0.7311
        // up_weights: k=0: sigmoid(0.5*1 + 0.5*0) = sigmoid(0.5) ≈ 0.6225
        let u0 = 1.0f32 / (1.0 + (-0.5f32).exp()); // ≈ 0.6225

        // For element 0: y=2.0, z=1.0
        // mixed_gate ≈ w0 * Id(2.0) = 0.7311 * 2.0 = 1.4622
        // mixed_up ≈ u0 * Id(1.0) = 0.6225 * 1.0 = 0.6225
        let expected_0 = (w0 * 2.0f32) * (u0 * 1.0f32);
        assert!(
            (hidden[0] - expected_0).abs() < 1e-3,
            "hidden[0] = {}, expected {}",
            hidden[0],
            expected_0
        );

        // For element 1: y=-1.0, z=1.0
        // mixed_gate ≈ w0 * Id(-1.0) = -w0, mixed_up ≈ u0 * Id(1.0) = u0
        let expected_1 = (-w0) * u0;
        assert!(
            (hidden[1] - expected_1).abs() < 1e-3,
            "hidden[1] = {}, expected {}",
            hidden[1],
            expected_1
        );
    }

    #[cfg(feature = "moa_inference")]
    #[test]
    fn test_simd_matmul_rmsnorm_moa_swiglu_correctness() {
        let output_dim = 4;
        let input_dim = 4;
        let weight: Vec<f32> = (0..2 * output_dim * input_dim)
            .map(|i| i as f32 * 0.1)
            .collect();
        let input: Vec<f32> = (0..input_dim).map(|i| (i + 1) as f32).collect();
        let rstd = 0.5;
        let moa = MoaConfig::new(input_dim);

        let mut output = vec![0.0f32; output_dim];
        simd_matmul_rmsnorm_moa_swiglu(
            &mut output,
            &weight,
            &input,
            rstd,
            &moa,
            output_dim,
            input_dim,
        );

        // With zero MoA gating → uniform 0.5 weights for all 7 activations
        // Verify first element manually
        let gate_val: f32 = weight[0..input_dim]
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>()
            * rstd;
        let up_val: f32 = weight[output_dim * input_dim..output_dim * input_dim + input_dim]
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f32>()
            * rstd;

        // All MoA weights = 0.5 (uniform)
        let mixed_gate: f32 = MoaActivation::all()
            .iter()
            .map(|a| 0.5 * a.activate(gate_val))
            .sum();
        let mixed_up: f32 = MoaActivation::all()
            .iter()
            .map(|a| 0.5 * a.activate(up_val))
            .sum();
        let expected = mixed_gate * mixed_up;

        assert!(
            (output[0] - expected).abs() < 1e-3,
            "MoA SwiGLU mismatch at 0: got {}, expected {expected}",
            output[0]
        );
    }

    #[cfg(feature = "moa_inference")]
    #[test]
    fn test_moa_vs_standard_silu_equivalence() {
        // GOAT T5: When MoA config absent → same as standard SwiGLU
        // This test verifies: if MoA only has Silu active, output matches standard SwiGLU
        let output_dim = 4;
        let input_dim = 4;
        let weight: Vec<f32> = (0..2 * output_dim * input_dim)
            .map(|i| i as f32 * 0.1)
            .collect();
        let input: Vec<f32> = (0..input_dim).map(|i| (i + 1) as f32).collect();
        let rstd = 0.5;

        // Standard SwiGLU
        let mut output_standard = vec![0.0f32; output_dim];
        simd_matmul_rmsnorm_swiglu(
            &mut output_standard,
            &weight,
            &input,
            rstd,
            GateActivation::Silu,
            output_dim,
            input_dim,
        );

        // MoA with only Silu active
        let mut moa = MoaConfig::new(input_dim);
        for k in 0..MOA_DICT_SIZE {
            let val = if k == 5 { 10.0f32 } else { -10.0f32 }; // k=5 = Silu
            for j in 0..input_dim {
                moa.gate_gating[k * input_dim + j] = val;
                moa.up_gating[k * input_dim + j] = val;
            }
        }
        let mut output_moa = vec![0.0f32; output_dim];
        simd_matmul_rmsnorm_moa_swiglu(
            &mut output_moa,
            &weight,
            &input,
            rstd,
            &moa,
            output_dim,
            input_dim,
        );

        // Should be close to standard SwiGLU (not exact because other activations
        // have small but nonzero sigmoid(-10) ≈ 4.5e-5 contribution)
        for i in 0..output_dim {
            let diff = (output_moa[i] - output_standard[i]).abs();
            let scale = output_standard[i].abs().max(1e-6);
            assert!(
                diff / scale < 0.01,
                "MoA vs SwiGLU mismatch at {}: moa={}, standard={}, rel_diff={}",
                i,
                output_moa[i],
                output_standard[i],
                diff / scale
            );
        }
    }
}
