//! Moka v1 int8 forward pass (Issue 206 T5).
//!
//! Keeps weights as int8 (the on-disk format) instead of dequantizing to f32
//! at load time. Uses platform-native int8 dot kernels:
//! - aarch64: SDOT instruction (inline asm, 16 mul/inst) when dotprod is
//!   detected; vmull+vpaddl fallback on NEON-only targets.
//! - wasm32: extmul workaround (stable Rust — `i32x4_dot_i8x16_s` is NOT
//!   exposed in stdarch, even nightly; Issue 206 action item to file upstream).
//! - other: scalar fallback (development only).
//!
//! ## Quantization scheme
//!
//! - **Weights**: per-output-channel symmetric int8 (the on-disk format —
//!   each output channel has its own f32 scale, applied after the dot).
//! - **Activations**: per-tensor symmetric int8 (one scale per activation
//!   tensor, computed from max-abs).
//!
//! The activation path alternates: int8 (for the dot) → f32 (for bias+ReLU+
//! residual) → quantize back to int8 (for the next layer's dot). The per-
//! tensor activation scale is recomputed at each layer boundary.
//!
//! The math: `result[oc] = bias[oc] + scale_a * scale_w[oc] * dot_i8(act, w[oc])`
//!
//! See [Bench 565](../../.benchmarks/565_int8_int8_sdot_positive.md) for the
//! microbenchmark that proved this path: 2.5–6.3× per-dot on native aarch64
//! SDOT, 1.8–4.4× amortized including quantization overhead.

use std::collections::HashMap;

use crate::board::{AREA as BOARD_AREA, SIZE as BOARD_SIZE};
use crate::moka::{
    BOTTLENECK_CHANNELS, GLOBAL_BLOCK_INTERVAL, INPUT_PLANES, MANIFEST_JSON, NUM_BLOCKS,
    POLICY_CHANNELS, POLICY_MOVES, SCORE_HIDDEN_CHANNELS, TRUNK_CHANNELS, VALUE_CHANNELS,
    Manifest, TensorMeta, WEIGHTS_BIN, global_mean_max_into, load_bias, read_f32, relu_inplace,
};

// ── Weight structs ──────────────────────────────────────────────────

/// Weight+bias pair in int8 format (no dequantization at load time).
struct WbInt8 {
    /// Raw int8 weights, layout `[out_channels, per_channel]` where
    /// `per_channel = k * k * in_ch` for conv weights, or `in_dim` for linear.
    w_i8: Vec<i8>,
    /// Per-output-channel f32 scales. `w_scales[oc]` is the scale for all
    /// weights in output channel `oc`.
    w_scales: Vec<f32>,
    /// Float biases (unchanged from the f32 path — biases stay f32).
    b: Vec<f32>,
}

impl WbInt8 {
    /// Number of output channels (equals `w_scales.len()` and `b.len()`).
    fn out_channels(&self) -> usize {
        self.w_scales.len()
    }
}

struct GlobalBranchInt8 {
    hidden: WbInt8,
    output: WbInt8,
}

struct ResidualBlockInt8 {
    reduce: WbInt8,
    first: WbInt8,
    global: Option<GlobalBranchInt8>,
    second: WbInt8,
    expand: WbInt8,
}

/// Moka weights kept in int8 format. The same vendored `go-model.bin` is
/// loaded, but the int8 bytes are kept as-is (not dequantized to f32).
pub struct MokaWeightsInt8 {
    stem: WbInt8,
    blocks: Vec<ResidualBlockInt8>,
    policy_conv: WbInt8,
    policy_linear: WbInt8,
    value_conv: WbInt8,
    value_hidden: WbInt8,
    value_output: WbInt8,
}

impl MokaWeightsInt8 {
    /// Load weights from the vendored `go-model.bin`, keeping them as int8.
    pub fn load() -> Self {
        let manifest: Manifest =
            serde_json::from_str(MANIFEST_JSON).expect("vendored moka manifest is valid JSON");
        let tensors = &manifest.tensors;
        let get = |prefix: &str| -> WbInt8 { load_int8(tensors, WEIGHTS_BIN, prefix) };

        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            let prefix = format!("residual.{i}");
            let has_global = (i + 1) % GLOBAL_BLOCK_INTERVAL == 0;
            blocks.push(ResidualBlockInt8 {
                reduce: get(&format!("{prefix}.reduce")),
                first: get(&format!("{prefix}.first")),
                global: has_global.then(|| GlobalBranchInt8 {
                    hidden: get(&format!("{prefix}.global.hidden")),
                    output: get(&format!("{prefix}.global.output")),
                }),
                second: get(&format!("{prefix}.second")),
                expand: get(&format!("{prefix}.expand")),
            });
        }

        Self {
            stem: get("stem"),
            blocks,
            policy_conv: get("policy.convolution"),
            policy_linear: get("policy.linear"),
            value_conv: get("value.convolution"),
            value_hidden: get("value.hidden"),
            value_output: get("value.output"),
        }
    }
}

/// Load a weight tensor as int8 (no dequantization) + its per-channel scales.
fn load_int8(tensors: &HashMap<String, TensorMeta>, bytes: &[u8], prefix: &str) -> WbInt8 {
    let w_name = format!("{prefix}.weight");
    let b_name = format!("{prefix}.bias");
    let meta = tensors
        .get(&w_name)
        .unwrap_or_else(|| panic!("moka manifest missing tensor {w_name}"));
    assert_eq!(meta.dtype, "int8", "expected int8 weight tensor {w_name}");

    let out_channels = meta.shape[0];
    let count: usize = meta.shape.iter().product();
    let scale_offset = meta
        .scale_offset
        .unwrap_or_else(|| panic!("{w_name} missing scaleOffset"));
    let w_scales = read_f32(bytes, scale_offset, out_channels);

    // Copy raw int8 bytes — no dequantization. The on-disk layout is already
    // `[out_channels, per_channel]` in int8, matching our `w_i8` layout.
    let data_end = meta.data_offset + count;
    let w_i8: Vec<i8> = bytes[meta.data_offset..data_end].iter().map(|&b| b as i8).collect();

    let b = load_bias(tensors, bytes, &b_name);

    WbInt8 { w_i8, w_scales, b }
}

// ── Activation quantization ─────────────────────────────────────────

/// Quantize a f32 tensor to int8 using per-tensor symmetric scaling.
///
/// Returns the scale (`max_abs / 127.0`). To reconstruct: `original ≈ q * scale`.
/// The quantized values are clamped to `[-128, 127]`.
///
/// For zero input (all zeros or very small), returns scale 1.0 and fills
/// output with zeros — the dot product will be zero regardless.
#[inline]
fn quantize_tensor(input: &[f32], output: &mut [i8]) -> f32 {
    debug_assert_eq!(input.len(), output.len());

    // WASM SIMD128 fast path — the scalar path was the bottleneck on V8 JIT
    // (Issue 206 T6: int8 was 0.88× — quantization overhead dominated).
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { quantize_tensor_wasm_simd(input, output) };
    }

    #[allow(unreachable_code)]
    {
        quantize_tensor_scalar(input, output)
    }
}

/// Scalar quantization (portable fallback + native aarch64 path).
#[inline]
fn quantize_tensor_scalar(input: &[f32], output: &mut [i8]) -> f32 {
    let max_abs = input.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if max_abs < 1e-30 {
        output.fill(0);
        return 1.0;
    }
    let inv_scale = 127.0 / max_abs;
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let q = (*inp * inv_scale).round();
        *out = q.clamp(-128.0, 127.0) as i8;
    }
    max_abs / 127.0
}

/// WASM SIMD128 quantization — vectorizes the max-abs reduction + scale loop.
///
/// Uses `f32x4_abs` + `f32x4_pmax` for the reduction, then `f32x4_mul` +
/// `f32x4_nearest` (round-to-nearest) + manual clamp for the scale step.
/// The final f32x4 → i8 conversion uses extract_lane + cast (the narrow
/// intrinsics need 8-element batches which complicate the tail handling).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn quantize_tensor_wasm_simd(input: &[f32], output: &mut [i8]) -> f32 {
    use core::arch::wasm32::{
        f32x4_abs, f32x4_extract_lane, f32x4_max, f32x4_min, f32x4_mul, f32x4_nearest,
        f32x4_splat, v128_load,
    };
    let len = input.len();

    // Phase 1: SIMD max-abs reduction (4 elements at a time).
    let zero = f32x4_splat(0.0);
    let mut max_vec = zero;
    let mut i = 0;
    let chunks4 = len / 4;
    for _ in 0..chunks4 {
        let v = v128_load(input.as_ptr().add(i).cast());
        let abs_v = f32x4_abs(v);
        max_vec = f32x4_max(max_vec, abs_v);
        i += 4;
    }
    // Horizontal reduce max_vec → scalar.
    let mut max_abs = f32x4_extract_lane::<0>(max_vec)
        .max(f32x4_extract_lane::<1>(max_vec))
        .max(f32x4_extract_lane::<2>(max_vec))
        .max(f32x4_extract_lane::<3>(max_vec));
    // Tail (scalar).
    while i < len {
        max_abs = max_abs.max(input.get_unchecked(i).abs());
        i += 1;
    }

    if max_abs < 1e-30 {
        output.fill(0);
        return 1.0;
    }

    // Phase 2: SIMD scale + round + clamp (f32 still, conversion is scalar).
    let inv_scale = f32x4_splat(127.0 / max_abs);
    let lo_f = f32x4_splat(-128.0);
    let hi_f = f32x4_splat(127.0);

    let mut i = 0;
    let chunks4 = len / 4;
    for _ in 0..chunks4 {
        let v = v128_load(input.as_ptr().add(i).cast());
        let scaled = f32x4_mul(v, inv_scale);
        let rounded = f32x4_nearest(scaled);
        // Clamp to [-128, 127].
        let clamped = f32x4_max(f32x4_min(rounded, hi_f), lo_f);
        // Extract lanes and cast to i8 (the conversion is cheap; the SIMD
        // win was in the max-abs reduction + scale + round above).
        *output.get_unchecked_mut(i) = f32x4_extract_lane::<0>(clamped) as i8;
        *output.get_unchecked_mut(i + 1) = f32x4_extract_lane::<1>(clamped) as i8;
        *output.get_unchecked_mut(i + 2) = f32x4_extract_lane::<2>(clamped) as i8;
        *output.get_unchecked_mut(i + 3) = f32x4_extract_lane::<3>(clamped) as i8;
        i += 4;
    }
    // Tail (scalar).
    let inv_scale_scalar = 127.0 / max_abs;
    while i < len {
        let q = (*input.get_unchecked(i) * inv_scale_scalar).round();
        *output.get_unchecked_mut(i) = q.clamp(-128.0, 127.0) as i8;
        i += 1;
    }

    max_abs / 127.0
}

// ── int8 dot kernels ────────────────────────────────────────────────

/// Scalar int8 dot product (portable fallback — development only).
#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut sum: i32 = 0;
    for i in 0..a.len() {
        sum += (a[i] as i32) * (b[i] as i32);
    }
    sum
}

/// aarch64 SDOT kernel via inline assembly (ARMv8.2-A dotprod).
///
/// Does 16 i8 multiplies + 4 i32 accumulates in ONE instruction. Uses a
/// different execution unit than f32 FMA, so the "FPU saturated" finding
/// (Bench 205) doesn't apply.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_sdot(a: &[i8], b: &[i8], len: usize) -> i32 {
    use core::arch::aarch64::{int32x4_t, vaddvq_s32, vdupq_n_s32, vld1q_s8};
    unsafe {
        let mut acc: int32x4_t = vdupq_n_s32(0);
        let mut i = 0;
        let chunks16 = len / 16;
        for _ in 0..chunks16 {
            let va = vld1q_s8(a.as_ptr().add(i));
            let vb = vld1q_s8(b.as_ptr().add(i));
            core::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc,
                a = in(vreg) va,
                b = in(vreg) vb,
                options(pure, nomem, nostack, preserves_flags),
            );
            i += 16;
        }
        let mut sum = vaddvq_s32(acc);
        while i < len {
            sum += (*a.get_unchecked(i) as i32) * (*b.get_unchecked(i) as i32);
            i += 1;
        }
        sum
    }
}

/// aarch64 vmull+vpaddl kernel (all ARMv8 NEON targets, no dotprod needed).
///
/// Fallback used when dotprod is not available. May appear as dead code when
/// compiled with `target_feature = "dotprod"` — kept for portability.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_vmull(a: &[i8], b: &[i8], len: usize) -> i32 {
    use core::arch::aarch64::{
        int16x8_t, int32x4_t, vaddq_s32, vaddvq_s32, vdupq_n_s32, vget_high_s8, vget_low_s8,
        vld1q_s8, vmull_s8, vpaddlq_s16,
    };
    unsafe {
        let mut acc: int32x4_t = vdupq_n_s32(0);
        let mut i = 0;
        let chunks16 = len / 16;
        for _ in 0..chunks16 {
            let va = vld1q_s8(a.as_ptr().add(i));
            let vb = vld1q_s8(b.as_ptr().add(i));
            let lo: int16x8_t = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
            let hi: int16x8_t = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
            acc = vaddq_s32(acc, vaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi)));
            i += 16;
        }
        let mut sum = vaddvq_s32(acc);
        while i < len {
            sum += (*a.get_unchecked(i) as i32) * (*b.get_unchecked(i) as i32);
            i += 1;
        }
        sum
    }
}

/// WASM SIMD128 int8 dot via extmul workaround.
///
/// Rust's `core::arch::wasm32` does NOT expose `i32x4_dot_i8x16_s` (the
/// WASM `i8x16.dot_s` instruction), even on nightly. This kernel uses
/// extmul + extadd_pairwise (7 instrs per 16 elements) as a workaround.
/// Still ~2× faster than f32 (which needs separate mul+add — see Bench 565).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot_i8_wasm(a: &[i8], b: &[i8], len: usize) -> i32 {
    use core::arch::wasm32::{
        i16x8_extmul_high_i8x16, i16x8_extmul_low_i8x16, i32x4_add, i32x4_extadd_pairwise_i16x8,
        i32x4_extract_lane, i32x4_splat, v128_load,
    };
    unsafe {
        let mut acc0 = i32x4_splat(0);
        let mut acc1 = i32x4_splat(0);
        let mut i = 0;
        let chunks32 = len / 32;
        for _ in 0..chunks32 {
            let va = v128_load(a.as_ptr().add(i).cast());
            let vb = v128_load(b.as_ptr().add(i).cast());
            let lo = i16x8_extmul_low_i8x16(va, vb);
            let hi = i16x8_extmul_high_i8x16(va, vb);
            acc0 = i32x4_add(acc0, i32x4_extadd_pairwise_i16x8(lo));
            acc1 = i32x4_add(acc1, i32x4_extadd_pairwise_i16x8(hi));

            let va2 = v128_load(a.as_ptr().add(i + 16).cast());
            let vb2 = v128_load(b.as_ptr().add(i + 16).cast());
            let lo2 = i16x8_extmul_low_i8x16(va2, vb2);
            let hi2 = i16x8_extmul_high_i8x16(va2, vb2);
            acc0 = i32x4_add(acc0, i32x4_extadd_pairwise_i16x8(lo2));
            acc1 = i32x4_add(acc1, i32x4_extadd_pairwise_i16x8(hi2));
            i += 32;
        }
        let chunks16 = (len - i) / 16;
        for _ in 0..chunks16 {
            let va = v128_load(a.as_ptr().add(i).cast());
            let vb = v128_load(b.as_ptr().add(i).cast());
            let lo = i16x8_extmul_low_i8x16(va, vb);
            let hi = i16x8_extmul_high_i8x16(va, vb);
            acc0 = i32x4_add(acc0, i32x4_extadd_pairwise_i16x8(lo));
            acc1 = i32x4_add(acc1, i32x4_extadd_pairwise_i16x8(hi));
            i += 16;
        }
        let s = i32x4_add(acc0, acc1);
        let mut sum = i32x4_extract_lane::<0>(s)
            + i32x4_extract_lane::<1>(s)
            + i32x4_extract_lane::<2>(s)
            + i32x4_extract_lane::<3>(s);
        while i < len {
            sum += (*a.get_unchecked(i) as i32) * (*b.get_unchecked(i) as i32);
            i += 1;
        }
        sum
    }
}

/// Dispatch to the best available int8 dot kernel for the current platform.
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len();
    debug_assert_eq!(len, b.len());

    #[cfg(target_arch = "aarch64")]
    {
        // Fast path: compile-time dotprod (when -C target-feature=+dotprod or
        // -C target-cpu=native on Apple Silicon).
        #[cfg(target_feature = "dotprod")]
        {
            return unsafe { dot_i8_sdot(a, b, len) };
        }
        // Runtime detection path
        #[cfg(not(target_feature = "dotprod"))]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                return unsafe { dot_i8_sdot(a, b, len) };
            }
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { dot_i8_vmull(a, b, len) };
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return unsafe { dot_i8_wasm(a, b, len) };
    }

    #[allow(unreachable_code)]
    dot_i8_scalar(a, b)
}

// ── Conv2d / Linear (int8) ──────────────────────────────────────────

/// Scaled int8 dot: `scale_a * scale_w * dot_i8(a, w) + bias`.
#[inline(always)]
fn dot_i8_scaled(a: &[i8], scale_a: f32, w: &[i8], scale_w: f32, bias: f32) -> f32 {
    let int_dot = dot_i8(a, w);
    scale_a * scale_w * (int_dot as f32) + bias
}

/// 2D convolution with int8 weights.
///
/// Quantizes the input tensor once (per-tensor scale), then for each spatial
/// position gathers an int8 patch and dots it against each output channel's
/// int8 weight slice. The result is dequantized per-channel and biased.
///
/// `input_i8` is scratch of length `h * w * in_ch`; `patch_i8` is scratch of
/// length `k * k * in_ch`. Both are overwritten.
#[allow(clippy::too_many_arguments)]
fn conv2d_int8_into(
    input: &[f32],
    h: usize,
    w: usize,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    weight: &WbInt8,
    input_i8: &mut [i8],
    patch_i8: &mut [i8],
    out: &mut [f32],
) {
    let patch_len = k * k * in_ch;
    let input_len = h * w * in_ch;

    // Quantize the full input tensor once — scale_a is reused for all patches.
    // Slice input_i8 to match input length (input_i8 is sized for the largest layer).
    let scale_a = quantize_tensor(input, &mut input_i8[..input_len]);

    if k == 1 {
        for pos in 0..h * w {
            let pslice = &input_i8[pos * in_ch..pos * in_ch + in_ch];
            let obase = pos * out_ch;
            for oc in 0..out_ch {
                let wbase = oc * in_ch;
                out[obase + oc] = dot_i8_scaled(pslice, scale_a, &weight.w_i8[wbase..wbase + in_ch], weight.w_scales[oc], weight.b[oc]);
            }
        }
        return;
    }

    let pad = k / 2;
    for y in 0..h {
        for x in 0..w {
            patch_i8[..patch_len].fill(0);
            for ky in 0..k {
                let iy = y + ky;
                if iy < pad || iy >= h + pad {
                    continue;
                }
                let iy = iy - pad;
                for kx in 0..k {
                    let ix = x + kx;
                    if ix < pad || ix >= w + pad {
                        continue;
                    }
                    let ix = ix - pad;
                    let src = (iy * w + ix) * in_ch;
                    let dst = (ky * k + kx) * in_ch;
                    patch_i8[dst..dst + in_ch].copy_from_slice(&input_i8[src..src + in_ch]);
                }
            }

            let obase = (y * w + x) * out_ch;
            let pslice = &patch_i8[..patch_len];
            for oc in 0..out_ch {
                let wbase = oc * patch_len;
                out[obase + oc] = dot_i8_scaled(pslice, scale_a, &weight.w_i8[wbase..wbase + patch_len], weight.w_scales[oc], weight.b[oc]);
            }
        }
    }
}

/// Fully-connected layer with int8 weights.
///
/// Quantizes the input once, then dots it against each output's int8 weights.
fn linear_int8_into(
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    weight: &WbInt8,
    input_i8: &mut [i8],
    out: &mut [f32],
) {
    let scale_a = quantize_tensor(&input[..in_dim], &mut input_i8[..in_dim]);
    for (o, out_slot) in out.iter_mut().enumerate().take(out_dim) {
        let base = o * in_dim;
        *out_slot = dot_i8_scaled(&input_i8[..in_dim], scale_a, &weight.w_i8[base..base + in_dim], weight.w_scales[o], weight.b[o]);
    }
}

// ── Scratch ─────────────────────────────────────────────────────────

/// Scratch buffers for the int8 forward pass. Contains the same f32 activation
/// buffers as `MokaScratch` plus int8 quantization scratch.
pub struct MokaScratchInt8 {
    // f32 activation buffers (mirror MokaScratch)
    trunk: Vec<f32>,
    expand: Vec<f32>,
    hidden_a: Vec<f32>,
    hidden_b: Vec<f32>,
    head4: Vec<f32>,
    head2: Vec<f32>,
    pooled: Vec<f32>,
    gh: Vec<f32>,
    gbias: Vec<f32>,
    value_h: Vec<f32>,
    policy: Vec<f32>,

    // int8 quantization scratch (reused per layer)
    /// Quantized input tensor. Max size = BOARD_AREA * TRUNK_CHANNELS (the
    /// widest activation: the trunk). Reused across layers.
    input_i8: Vec<i8>,
    /// Gathered int8 patch for 3×3 convs. Size = 3*3*TRUNK_CHANNELS.
    patch_i8: Vec<i8>,
}

impl MokaScratchInt8 {
    pub fn new() -> Self {
        Self {
            trunk: vec![0.0; BOARD_AREA * TRUNK_CHANNELS],
            expand: vec![0.0; BOARD_AREA * TRUNK_CHANNELS],
            hidden_a: vec![0.0; BOARD_AREA * BOTTLENECK_CHANNELS],
            hidden_b: vec![0.0; BOARD_AREA * BOTTLENECK_CHANNELS],
            head4: vec![0.0; BOARD_AREA * POLICY_CHANNELS],
            head2: vec![0.0; BOARD_AREA * VALUE_CHANNELS],
            pooled: vec![0.0; BOTTLENECK_CHANNELS * 2],
            gh: vec![0.0; 8],
            gbias: vec![0.0; BOTTLENECK_CHANNELS],
            value_h: vec![0.0; SCORE_HIDDEN_CHANNELS],
            policy: vec![0.0; POLICY_MOVES],
            // input_i8 must hold the largest quantized tensor = trunk (81*32=2592).
            // All other layers' inputs are smaller.
            input_i8: vec![0i8; BOARD_AREA * TRUNK_CHANNELS],
            // patch_i8 must hold the largest 3×3 patch = 3*3*32=288 (trunk channels).
            // But conv on the trunk uses BOTTLENECK_CHANNELS (16) for 3×3, so 3*3*16=144.
            // The stem conv is 3×3 with INPUT_PLANES=12 in_ch → 3*3*12=108.
            // The max is actually 3*3*TRUNK_CHANNELS=288 for the stem output IF we
            // did 3×3 conv on it, but we don't — trunk only sees 1×1 convs after the stem.
            // Safely sized to TRUNK_CHANNELS to cover all cases.
            patch_i8: vec![0i8; 3 * 3 * TRUNK_CHANNELS],
        }
    }
}

impl Default for MokaScratchInt8 {
    fn default() -> Self {
        Self::new()
    }
}

// ── Forward pass ────────────────────────────────────────────────────

/// Moka forward pass using int8 weights.
///
/// Mirrors `moka::forward_with_scratch` exactly, but uses int8 conv/linear
/// layers. The activation path alternates f32 (between layers) and int8
/// (inside each layer's dot product).
pub fn forward_int8_with_scratch(
    weights: &MokaWeightsInt8,
    features: &[f32],
    scratch: &mut MokaScratchInt8,
) -> ([f32; POLICY_MOVES], f32) {
    let MokaScratchInt8 {
        trunk,
        expand,
        hidden_a,
        hidden_b,
        head4,
        head2,
        pooled,
        gh,
        gbias,
        value_h,
        policy,
        input_i8,
        patch_i8,
    } = scratch;

    // Stem: INPUT_PLANES → TRUNK_CHANNELS, 3×3
    conv2d_int8_into(
        features, BOARD_SIZE, BOARD_SIZE, INPUT_PLANES, TRUNK_CHANNELS, 3,
        &weights.stem, input_i8, patch_i8, trunk,
    );
    relu_inplace(&mut trunk[..BOARD_AREA * TRUNK_CHANNELS]);

    for block in &weights.blocks {
        // reduce: TRUNK_CHANNELS → BOTTLENECK_CHANNELS, 1×1
        conv2d_int8_into(
            trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, BOTTLENECK_CHANNELS, 1,
            &block.reduce, input_i8, patch_i8, hidden_a,
        );
        relu_inplace(hidden_a);
        // first: BOTTLENECK_CHANNELS → BOTTLENECK_CHANNELS, 3×3
        conv2d_int8_into(
            hidden_a, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS, 3,
            &block.first, input_i8, patch_i8, hidden_b,
        );
        relu_inplace(hidden_b);

        if let Some(g) = &block.global {
            global_mean_max_into(hidden_b, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, pooled);
            linear_int8_into(pooled, BOTTLENECK_CHANNELS * 2, g.hidden.out_channels(), &g.hidden, input_i8, gh);
            relu_inplace(&mut gh[..g.hidden.out_channels()]);
            linear_int8_into(gh, g.hidden.out_channels(), BOTTLENECK_CHANNELS, &g.output, input_i8, gbias);
            for pos in 0..BOARD_AREA {
                let row = &mut hidden_b[pos * BOTTLENECK_CHANNELS..(pos + 1) * BOTTLENECK_CHANNELS];
                for c in 0..BOTTLENECK_CHANNELS {
                    row[c] += gbias[c];
                }
            }
        }

        // second: BOTTLENECK_CHANNELS → BOTTLENECK_CHANNELS, 3×3
        conv2d_int8_into(
            hidden_b, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, BOTTLENECK_CHANNELS, 3,
            &block.second, input_i8, patch_i8, hidden_a,
        );
        relu_inplace(hidden_a);
        // expand: BOTTLENECK_CHANNELS → TRUNK_CHANNELS, 1×1
        conv2d_int8_into(
            hidden_a, BOARD_SIZE, BOARD_SIZE, BOTTLENECK_CHANNELS, TRUNK_CHANNELS, 1,
            &block.expand, input_i8, patch_i8, expand,
        );

        // residual add + relu
        for i in 0..BOARD_AREA * TRUNK_CHANNELS {
            let v = trunk[i] + expand[i];
            trunk[i] = if v < 0.0 { 0.0 } else { v };
        }
    }

    // Policy head
    conv2d_int8_into(
        trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, POLICY_CHANNELS, 1,
        &weights.policy_conv, input_i8, patch_i8, head4,
    );
    relu_inplace(head4);
    linear_int8_into(head4, POLICY_CHANNELS * BOARD_AREA, POLICY_MOVES, &weights.policy_linear, input_i8, policy);

    // Value head
    conv2d_int8_into(
        trunk, BOARD_SIZE, BOARD_SIZE, TRUNK_CHANNELS, VALUE_CHANNELS, 1,
        &weights.value_conv, input_i8, patch_i8, head2,
    );
    relu_inplace(head2);
    let value_hidden_dim = weights.value_hidden.out_channels();
    linear_int8_into(head2, VALUE_CHANNELS * BOARD_AREA, value_hidden_dim, &weights.value_hidden, input_i8, value_h);
    relu_inplace(&mut value_h[..value_hidden_dim]);
    let mut value_out = [0f32; 1];
    linear_int8_into(value_h, value_hidden_dim, 1, &weights.value_output, input_i8, &mut value_out);

    let mut logits = [0f32; POLICY_MOVES];
    logits.copy_from_slice(&policy[..POLICY_MOVES]);
    (logits, value_out[0].tanh())
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::moka::{MokaScratch, MokaWeights, encode_features_into, forward_with_scratch};

    /// G1 GOAT gate: int8 forward pass must produce game-play-equivalent
    /// output to the f32 baseline. This is the load-bearing correctness gate.
    ///
    /// Two criteria:
    /// 1. **Argmax agreement**: the top-scoring move must match f32 on every
    ///    test board. This is the greedy move-selection criterion.
    /// 2. **Value accuracy**: absolute value diff must stay under 0.1 (the
    ///    value goes through tanh, so this is the post-squash diff).
    ///
    /// We do NOT assert on per-logit error because the raw logits have a wide
    /// dynamic range (some are ~0.1, others ~88) and the int8 quantization
    /// noise is uniform-ish in absolute terms. What matters for PUCT is the
    /// *distribution* (softmax) + the *move selection* (argmax), both of which
    /// are dominated by the high-magnitude logits where int8 is most accurate
    /// relatively.
    #[test]
    fn g1_int8_matches_f32_baseline() {
        let weights_f32 = MokaWeights::load();
        let weights_i8 = MokaWeightsInt8::load();
        let mut scratch_f32 = MokaScratch::new();
        let mut scratch_i8 = MokaScratchInt8::new();

        // Build several distinct mid-game boards.
        let opening_seqs: [[usize; 6]; 4] = [
            [40, 41, 31, 50, 32, 49],
            [0, 1, 9, 10, 18, 19],
            [80, 79, 71, 70, 62, 61],
            [40, 50, 30, 60, 31, 51],
        ];

        let mut max_value_diff = 0f32;
        let mut argmax_mismatches = 0usize;

        for &seq in &opening_seqs {
            let mut board = Board::new();
            let mut hist: Vec<Option<(usize, usize)>> = Vec::new();
            for &mv in &seq {
                if board.is_legal(mv) {
                    board.play(mv);
                    hist.push(Some((mv / BOARD_SIZE, mv % BOARD_SIZE)));
                }
            }
            let last2: Vec<Option<(usize, usize)>> =
                hist.iter().rev().take(2).copied().collect::<Vec<_>>().into_iter().rev().collect();
            let mut features = vec![0.0; crate::moka::INPUT_ELEMENT_COUNT];
            encode_features_into(&board, &last2, &mut features);

            let (p_f32, v_f32) = forward_with_scratch(&weights_f32, &features, &mut scratch_f32);
            let (p_i8, v_i8) = forward_int8_with_scratch(&weights_i8, &features, &mut scratch_i8);

            // Criterion 1: argmax agreement (the move-selection gate)
            // total_cmp: this crate ships with NO katgpt-core dep by design
            // (wasm32-minimal), so float_order is unavailable here; fixtures are
            // probabilities and NaN-free by construction.
            let f32_argmax = p_f32.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i);
            let i8_argmax = p_i8.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i);
            if f32_argmax != i8_argmax {
                argmax_mismatches += 1;
            }

            // Criterion 2: value accuracy
            let vdiff = (v_f32 - v_i8).abs();
            if vdiff > max_value_diff {
                max_value_diff = vdiff;
            }
        }

        eprintln!(
            "g1_int8: max_value_diff={max_value_diff:.4}, argmax_mismatches={argmax_mismatches}/4"
        );

        assert_eq!(argmax_mismatches, 0, "int8 argmax disagrees with f32 on {argmax_mismatches}/4 boards — move selection is wrong");
        assert!(
            max_value_diff < 0.10,
            "int8 vs f32 value diff {max_value_diff:.4} exceeds 0.10"
        );
    }

    /// G2 GOAT gate: int8 forward pass latency. Release-only because debug
    /// builds aren't optimized enough for meaningful perf numbers.
    ///
    /// Measures ns/forward for both f32 and int8 paths and reports the speedup.
    /// Gate: int8 must be ≥1.3× faster than f32.
    ///
    /// The microbenchmark (Bench 565) projected 2.5–3× per-dot, but the
    /// end-to-end forward pass shows a lower speedup because:
    /// 1. Non-dot overhead (patch gather, ReLU, residual add, global pool)
    ///    is unchanged between f32 and int8 paths.
    /// 2. Many layers have small dots (patch_len=16 for expand convs) where
    ///    per-call dispatch overhead dominates.
    /// 3. The f32 scale multiplication (`scale_a * scale_w * int_dot + bias`)
    ///    adds ~80K f32 ops that the f32 path doesn't have.
    ///
    /// Even at 1.4× on native, the WASM speedup is expected to be higher
    /// (WASM f32 has no FMA, so the int8 advantage is larger relatively).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn g2_int8_forward_speedup() {
        use std::hint::black_box;
        use std::time::Instant;

        const ITERS: usize = 2000;

let weights_f32 = MokaWeights::load();
        let weights_i8 = MokaWeightsInt8::load();
        let mut scratch_f32 = MokaScratch::new();
        let mut scratch_i8 = MokaScratchInt8::new();
        let features = vec![0.5f32; crate::moka::INPUT_ELEMENT_COUNT];

        // Warmup
        for _ in 0..50 {
            let _ = black_box(forward_with_scratch(&weights_f32, &features, &mut scratch_f32));
            let _ = black_box(forward_int8_with_scratch(&weights_i8, &features, &mut scratch_i8));
        }

        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _ = black_box(forward_with_scratch(&weights_f32, &features, &mut scratch_f32));
        }
        let f32_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _ = black_box(forward_int8_with_scratch(&weights_i8, &features, &mut scratch_i8));
        }
        let i8_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

        let speedup = f32_ns / i8_ns;
        eprintln!(
            "g2_int8: f32={f32_ns:.0}ns/fwd, int8={i8_ns:.0}ns/fwd, speedup={speedup:.2}x"
        );

        assert!(
            speedup >= 1.3,
            "int8 forward speedup {speedup:.2}× below 1.3× gate (f32={f32_ns:.0}ns, i8={i8_ns:.0}ns)"
        );
    }

    /// G4 GOAT gate: forward pass must be allocation-free in steady state.
    ///
    /// Uses capacity-stability proxy (same pattern as bench_413): after
    /// warmup, the scratch buffer capacities must not grow across N steady-
    /// state calls. This proves the forward path doesn't allocate, because the
    /// only potentially-growing structures are the scratch Vecs — if they
    /// don't grow, nothing else can allocate (the forward path uses only
    /// slices into pre-allocated buffers).
    #[test]
    fn g4_int8_forward_alloc_free() {
        let weights = MokaWeightsInt8::load();
        let mut scratch = MokaScratchInt8::new();
        let features = vec![0.5f32; crate::moka::INPUT_ELEMENT_COUNT];

        // Warmup — let any one-time allocations settle.
        for _ in 0..10 {
            let _ = forward_int8_with_scratch(&weights, &features, &mut scratch);
        }

        // Record total capacity after warmup.
        let cap_before: usize = scratch.trunk.capacity()
            + scratch.expand.capacity()
            + scratch.hidden_a.capacity()
            + scratch.hidden_b.capacity()
            + scratch.head4.capacity()
            + scratch.head2.capacity()
            + scratch.pooled.capacity()
            + scratch.gh.capacity()
            + scratch.gbias.capacity()
            + scratch.value_h.capacity()
            + scratch.policy.capacity()
            + scratch.input_i8.capacity()
            + scratch.patch_i8.capacity();

        // Run steady-state calls.
        for _ in 0..100 {
            let _ = forward_int8_with_scratch(&weights, &features, &mut scratch);
        }

        let cap_after: usize = scratch.trunk.capacity()
            + scratch.expand.capacity()
            + scratch.hidden_a.capacity()
            + scratch.hidden_b.capacity()
            + scratch.head4.capacity()
            + scratch.head2.capacity()
            + scratch.pooled.capacity()
            + scratch.gh.capacity()
            + scratch.gbias.capacity()
            + scratch.value_h.capacity()
            + scratch.policy.capacity()
            + scratch.input_i8.capacity()
            + scratch.patch_i8.capacity();

        assert_eq!(
            cap_before, cap_after,
            "scratch capacities grew during steady-state ({cap_before} → {cap_after}) — forward path is allocating"
        );
        eprintln!("g4_int8: PASS — all scratch capacities stable across 100 steady-state calls");
    }
}
