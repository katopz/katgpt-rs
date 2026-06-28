//! SIMD-accelerated math primitives.

use super::*;

// ---------------------------------------------------------------------------
// Math utilities — SIMD-accelerated
// ---------------------------------------------------------------------------

/// In-place softmax. Handles empty slices gracefully.
/// Three-pass: find max → shift+exp+sum (fused) → normalize.
#[inline(always)]
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Pass 1: find max for numerical stability (SIMD-accelerated)
    let max_val = crate::simd::simd_max_f32(x);

    // Pass 2: subtract max (SIMD-accelerated)
    crate::simd::simd_add_scalar_inplace(x, -max_val);

    // Pass 3: SIMD exp + sum (fused — saves one full buffer traversal vs separate exp+sum)
    let sum: f32 = crate::simd::simd_exp_sum_inplace(x);
    let inv_sum = 1.0 / sum;
    crate::simd::simd_scale_inplace(x, inv_sum);
}

/// In-place softmax with temperature scaling: `softmax(x / temperature)`.
///
/// Fuses the temperature division into the exp computation, saving one full pass
/// vs separate `for p /= temp; softmax(x)`.
///
/// `inv_temp` should be `1.0 / temperature` — compute once, pass to every call.
#[inline(always)]
pub fn softmax_scaled(x: &mut [f32], inv_temp: f32) {
    if x.is_empty() {
        return;
    }

    // Pass 1: find max for numerical stability (SIMD-accelerated)
    let max_val = crate::simd::simd_max_f32(x);

    // Pass 2: shift and apply temperature in one fused SIMD pass
    crate::simd::simd_fused_sub_scale_inplace(x, max_val, inv_temp);

    // Pass 3: SIMD exp + sum (fused — saves one full buffer traversal vs separate exp+sum)
    let sum: f32 = crate::simd::simd_exp_sum_inplace(x);
    let inv_sum = 1.0 / sum;
    crate::simd::simd_scale_inplace(x, inv_sum);
}

/// In-place RMSNorm (no learnable gain).
/// Two-pass: compute sum-of-squares, then scale.
#[inline(always)]
pub fn rmsnorm(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Pass 1: sum of squares (SIMD-accelerated)
    let sum_sq = crate::simd::simd_sum_sq(x, x.len());

    // Pass 2: scale — stay f32 throughout to avoid f64 round-trip
    let inv_rms = 1.0 / (sum_sq / x.len() as f32 + 1e-5f32).sqrt();
    crate::simd::simd_scale_inplace(x, inv_rms);
}

/// GeGLU activation: hidden = gelu(gate) * up (elementwise).
/// Uses approximate GELU: gelu(x) ≈ x * sigmoid(1.702 * x).
/// `gate` and `up` are [mlp_hidden], output goes to `hidden`.
///
/// SIMD-accelerated: exp() computed via `simd_exp_inplace` on stack buffers.
#[inline(always)]
pub fn gegelu(hidden: &mut [f32], gate: &[f32], up: &[f32]) {
    const CHUNK: usize = 64;
    let mut buf = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= hidden.len() {
        // buf[j] = -1.702 * gate[j] via fused SIMD copy+scale (single pass)
        crate::simd::simd_fused_decay_write(&mut buf, 0.0, &gate[i..i + CHUNK], -1.702);
        // buf[j] = exp(-1.702 * gate[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // hidden[j] = gate[j] * sigmoid(1.702 * gate[j]) * up[j]
        // SIMD: buf = 1 + buf, then buf = 1/buf (sigmoid), then fused gate*sigmoid*up
        crate::simd::simd_add_scalar_inplace(&mut buf, 1.0);
        // Vectorized reciprocal: buf = sigmoid = 1/(1+exp(-1.702*gate))
        crate::simd::simd_reciprocal_inplace(&mut buf);
        // Fused: hidden = gate * up, then scale-multiply by sigmoid
        for j in 0..CHUNK {
            hidden[i + j] = gate[i + j] * up[i + j];
        }
        crate::simd::simd_scale_mul_inplace(&mut hidden[i..i + CHUNK], &buf, 1.0);
        i += CHUNK;
    }
    // Scalar remainder
    for i in i..hidden.len() {
        let g = gate[i];
        let sigmoid = 1.0 / (1.0 + (-1.702 * g).exp());
        hidden[i] = g * sigmoid * up[i];
    }
}

/// GeGLU with tanh GELU approximation (Gemma 2 activation).
/// tanh GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// hidden[i] = gelu_tanh(gate[i]) * up[i]
///
/// SIMD-accelerated: exp() for tanh approximation computed via `simd_exp_inplace`.
#[inline(always)]
pub fn gegelu_tanh(hidden: &mut [f32], gate: &[f32], up: &[f32]) {
    const CHUNK: usize = 64;
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // (2.0f32 / π).sqrt()
    const SCALE_2: f32 = 1.595_769_2; // 2.0 * SQRT_2_OVER_PI
    let mut buf = [0.0f32; CHUNK];
    let mut buf2 = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= hidden.len() {
        // buf[j] = 0.044715 * g via fused SIMD copy+scale (single pass)
        crate::simd::simd_fused_decay_write(&mut buf, 0.0, &gate[i..i + CHUNK], 0.044715);
        crate::simd::simd_scale_mul_inplace(&mut buf, &gate[i..i + CHUNK], 1.0); // buf = 0.044715 * g²
        // Finish cubic via SIMD: buf = 1 + 0.044715*g², then buf = scale_2 * g * (1 + 0.044715*g²)
        crate::simd::simd_add_scalar_inplace(&mut buf, 1.0);
        crate::simd::simd_scale_mul_inplace(&mut buf, &gate[i..i + CHUNK], SCALE_2);
        // buf[j] = exp(2*inner[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // hidden[j] = g * exp(2x) / (exp(2x) + 1) * up[j]
        // Compute denominator (exp + 1) via SIMD, then SIMD tanh + fused mul
        buf2[..CHUNK].copy_from_slice(&buf);
        crate::simd::simd_add_scalar_inplace(&mut buf2, 1.0); // buf2 = exp + 1
        // hidden = gate * up, then hidden *= tanh(inner)
        for j in 0..CHUNK {
            // Branch-free tanh: exp(2x) / (exp(2x) + 1) via division
            buf[j] /= buf2[j];
            hidden[i + j] = gate[i + j] * up[i + j];
        }
        crate::simd::simd_scale_mul_inplace(&mut hidden[i..i + CHUNK], &buf, 1.0);
        i += CHUNK;
    }
    // Scalar remainder
    for i in i..hidden.len() {
        let g = gate[i];
        let inner = SQRT_2_OVER_PI * (g + 0.044715 * g * g * g);
        let gelu_val = 0.5 * g * (1.0 + inner.tanh());
        hidden[i] = gelu_val * up[i];
    }
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
/// Used in LLaMA, Mistral, and other LLaMA-family models for SwiGLU MLP.
///
/// SIMD-accelerated: exp() computed via `simd_exp_inplace` on stack buffers.
#[inline(always)]
pub fn silu(x: &mut [f32]) {
    const CHUNK: usize = 64;
    let mut buf = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= x.len() {
        // buf[j] = -x[j] via fused SIMD copy+scale (single pass)
        crate::simd::simd_fused_decay_write(&mut buf, 0.0, &x[i..i + CHUNK], -1.0);
        // buf[j] = exp(-x[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // x[j] = x[j] / (1 + exp(-x[j]))
        // SIMD: buf = 1 + exp(-x), then buf = 1/buf, then x *= buf elementwise
        crate::simd::simd_add_scalar_inplace(&mut buf, 1.0);
        crate::simd::simd_reciprocal_inplace(&mut buf);
        crate::simd::simd_scale_mul_inplace(&mut x[i..i + CHUNK], &buf, 1.0);
        i += CHUNK;
    }
    // Scalar remainder
    for v in x[i..].iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// SwiGLU activation: SiLU(gate) * up.
/// Used in LLaMA-family models (gate_proj and up_proj are separate weights).
/// Result stored in `hidden`: hidden[i] = silu(gate[i]) * up[i]
///
/// SIMD-accelerated: exp() computed via `simd_exp_inplace` on stack buffers.
#[inline(always)]
pub fn swiglu(hidden: &mut [f32], gate: &[f32], up: &[f32]) {
    const CHUNK: usize = 64;
    let mut buf = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= hidden.len() {
        // buf[j] = -gate[j] via fused SIMD copy+scale (single pass)
        crate::simd::simd_fused_decay_write(&mut buf, 0.0, &gate[i..i + CHUNK], -1.0);
        // buf[j] = exp(-gate[j]) via SIMD
        crate::simd::simd_exp_inplace(&mut buf);
        // hidden[j] = gate[j] / (1 + exp(-gate[j])) * up[j]
        // SIMD: buf = 1 + exp(-gate), then vectorized reciprocal + gate*up
        crate::simd::simd_add_scalar_inplace(&mut buf, 1.0);
        // Vectorized reciprocal: buf = sigmoid = 1/(1+exp(-gate))
        crate::simd::simd_reciprocal_inplace(&mut buf);
        // Fused: hidden = gate * up, then scale-multiply by sigmoid
        for j in 0..CHUNK {
            hidden[i + j] = gate[i + j] * up[i + j];
        }
        crate::simd::simd_scale_mul_inplace(&mut hidden[i..i + CHUNK], &buf, 1.0);
        i += CHUNK;
    }
    // Scalar remainder
    for i in i..hidden.len() {
        let g = gate[i];
        hidden[i] = g / (1.0 + (-g).exp()) * up[i];
    }
}

/// RMSNorm with learnable gamma (gain) vector.
/// Gemma 2 stores gamma as (gamma-1), so +1 is added during load.
/// `x` is normalized in-place then scaled by `gamma[i]`:
///   x[i] = gamma[i] * x[i] / sqrt(mean_sq + eps)
#[inline(always)]
pub fn rmsnorm_with_gamma(x: &mut [f32], gamma: &[f32]) {
    rmsnorm_with_gamma_eps(x, gamma, 1e-5)
}

/// RMSNorm with learnable gamma and configurable epsilon.
#[inline(always)]
pub fn rmsnorm_with_gamma_eps(x: &mut [f32], gamma: &[f32], eps: f64) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let sum_sq = crate::simd::simd_sum_sq(x, n);
    // Cast eps to f32 once — the f64 param is kept for API compat
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps as f32).sqrt();
    crate::simd::simd_scale_mul_inplace(x, gamma, inv_rms);
}

/// Matrix-vector multiply: output = weight @ input.
/// Weight layout: [rows, cols] row-major.
#[inline(always)]
pub fn matmul(output: &mut [f32], weight: &[f32], input: &[f32], rows: usize, cols: usize) {
    crate::simd::simd_matmul_rows(output, weight, input, rows, cols);
}

/// Row-parallel matrix-vector multiply for large weight matrices (Plan 096).
///
/// Splits output rows across rayon threads. Use for large matmuls where
/// row count >> core count (e.g., `down_proj` 2304×9216, `lm_head` 256K×2304).
/// Falls back to sequential [`matmul`] for small matrices (rows < 512).
#[inline(always)]
pub fn matmul_parallel(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    crate::simd::simd_matmul_rows_parallel(output, weight, input, rows, cols);
}

/// Fused matrix-vector multiply + ReLU: output = max(0, weight @ input).
/// Saves one full buffer scan vs separate matmul + ReLU.
/// Used for MLP hidden layer where activation immediately follows projection.
#[inline(always)]
pub fn matmul_relu(output: &mut [f32], weight: &[f32], input: &[f32], rows: usize, cols: usize) {
    crate::simd::simd_matmul_relu_rows(output, weight, input, rows, cols);
}

/// Matrix-vector multiply with f16 weights: output = f16_weight @ f32_input.
/// Weight layout: [rows, cols] row-major, stored as `half::f16`.
///
/// Converts f16 weights to f32 on-the-fly during dot product.
/// Halves memory bandwidth for weight reads vs f32 storage.
#[inline(always)]
pub fn matmul_f16(
    output: &mut [f32],
    weight: &[half::f16],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    crate::simd::simd_matmul_f16_f32_rows(output, weight, input, rows, cols);
}

/// Row-parallel f16×f32 matrix-vector multiply for large weight matrices (Plan 096).
///
/// Splits output rows across rayon threads. Use for large f16 matmuls where
/// row count >> core count (e.g., `down_proj` 2304×9216, `lm_head` 256K×2304).
/// Falls back to sequential [`matmul_f16`] for small matrices (rows < 512).
#[inline(always)]
pub fn matmul_f16_parallel(
    output: &mut [f32],
    weight: &[half::f16],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    crate::simd::simd_matmul_f16_f32_rows_parallel(output, weight, input, rows, cols);
}

/// Sparse matrix-vector multiply for ReLU-activated inputs (TwELL-inspired).
///
/// Only processes columns where `input[c] > 0.0`, skipping dead neurons entirely.
/// Exploits the natural sparsity of ReLU activations in MLP layers where 95-99%
/// of hidden neurons are exactly zero after training with L1 regularization.
///
/// Distilled from "Sparser, Faster, Lighter Transformer Language Models"
/// (arXiv:2603.23198) by Sakana AI & NVIDIA.
///
/// Two-phase execution:
/// 1. Dynamic Packing: scan input, store non-zero indices & values into pre-allocated buffers
/// 2. Sparse Multiply: only iterate weights at alive column indices
///
/// Returns the number of alive (non-zero) neurons for diagnostics/threshold checks.
/// Buffers `active_indices` and `active_values` must be pre-allocated to at least `cols` capacity.
#[cfg(feature = "sparse_mlp")]
#[inline(always)]
pub fn sparse_matmul(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
    active_indices: &mut [usize],
    active_values: &mut [f32],
) -> usize {
    // Phase 1: Pack alive neurons (software TwELL formulation)
    // Branch-predicted: with 95-99% sparsity, the branch is predicted correctly
    // most of the time (~1 cycle), and we avoid the wasted store to active_values
    // for dead neurons that the branch-free version always performed.
    let mut alive = 0;
    for c in 0..cols {
        let val = unsafe { *input.get_unchecked(c) };
        if val > 0.0 {
            unsafe {
                *active_indices.get_unchecked_mut(alive) = c;
                *active_values.get_unchecked_mut(alive) = val;
            }
            alive += 1;
        }
    }

    // Phase 2: Sparse multiply — SIMD-accelerated (Plan 060 T5)
    // NEON gathers 4 elements/iter, AVX2 gathers 8 elements/iter via hardware gather.
    // Scalar fallback for alive ≤ 4 (gather overhead exceeds benefit).
    crate::simd::simd_sparse_matmul_rows(
        output,
        weight,
        active_indices,
        active_values,
        rows,
        cols,
        alive,
    );

    alive
}

/// Sample a token index from a probability distribution.
///
/// Builds a prefix-sum (CDF) then uses binary search for O(log V) lookup
/// instead of the O(V/2) average of a linear scan.
///
/// **Allocates a CDF buffer on every call.** For the hot decode loop, prefer
/// [`sample_token_into`] which reuses a pre-allocated buffer.
pub fn sample_token(probs: &[f32], rng: &mut Rng) -> usize {
    // Redraw on exactly 0.0: `rng.uniform()` can return 0.0 (notably the first draw
    // for low-entropy seeds), which deterministically maps to the first nonzero-mass
    // token via the left boundary of the inverse-CDF map. `partition_point(c <= r)`
    // below already implements the strict comparison, so this guard only fixes the
    // degenerate-zero draw without changing the comparison semantics.
    let mut r = rng.uniform();
    while r == 0.0 {
        r = rng.uniform();
    }
    let n = probs.len();
    if n == 0 {
        return 0;
    }

    // Build cumulative sum array — pre-allocated, direct write avoids per-push bounds check
    let mut cdf = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        sum += p;
        // SAFETY: cdf has length n, i < n by enumeration
        unsafe {
            *cdf.get_unchecked_mut(i) = sum;
        }
    }

    // partition_point: first index where cdf[i] > r — monotonically increasing
    let idx = cdf[..n].partition_point(|&c| c <= r);
    idx.min(n - 1)
}

/// Zero-alloc variant of [`sample_token`] that reuses a pre-allocated CDF buffer.
///
/// Pass a `cdf` buffer (e.g. `ForwardContext::cdf`) to avoid a ~vocab_size allocation
/// on every token decode. The buffer is cleared and refilled each call.
pub fn sample_token_into(probs: &[f32], rng: &mut Rng, cdf: &mut Vec<f32>) -> usize {
    // See `sample_token`: redraw on exactly 0.0 to avoid the degenerate left-boundary draw.
    let mut r = rng.uniform();
    while r == 0.0 {
        r = rng.uniform();
    }
    let n = probs.len();
    if n == 0 {
        return 0;
    }
    cdf.resize(n, 0.0);
    let mut sum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        sum += p;
        unsafe {
            *cdf.get_unchecked_mut(i) = sum;
        }
    }
    // partition_point: first index where cdf[i] > r — monotonically increasing
    // so this is equivalent to binary_search_by with Less/Greater but avoids
    // closure overhead and is branch-predictor friendly.
    let idx = cdf[..n].partition_point(|&c| c <= r);
    idx.min(n - 1)
}

