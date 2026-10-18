//! bf16_convert — SIMD bf16⇄f32 batch conversion kernels (Issue 800 Arm A).
//!
//! SOURCES: Issue 800 Arm A (pufferlib distill, riir-train Research 454 @
//! `6ffa5b10`, MIT — kernel SHAPE only); rounding semantics follow the `half`
//! crate (RNE) because pufferlib's scalar `bits >> 16` truncation is biased.
//!
//! Two directions:
//! - **Widening** (`u16` bf16 bits → `f32`): lossless by construction — a
//!   bf16 IS the top half of an f32, so `f32 = f32::from_bits(bits << 16)`
//!   preserves value, NaN payload (shifted), Inf, and both zero signs.
//! - **Narrowing** (`f32` → `u16` bf16 bits): two explicit arms —
//!   [`f32_to_bf16_rne_into`] (round-to-nearest-even, the default, bit-exact
//!   vs `half::bf16::from_f32` INCLUDING the NaN class) and
//!   [`f32_to_bf16_trunc_into`] (truncation, the pufferlib kernel shape,
//!   biased toward zero — explicit fast opt-in only).
//!
//! SIMD coverage follows the house `simd_lut_dequant` convention: NEON on
//! `aarch64`, AVX2 on `x86_64` **when compiled with
//! `target-feature=+avx2`** (compile-time gating, no runtime detection), and
//! a scalar fallback everywhere else (including wasm32). All kernels are
//! `into_buf`-shaped: they write into caller-owned buffers, zero alloc.
//!
//! OPT-IN behind the `bf16_simd` feature pending the Bench 800 GOAT gate
//! (G1 known-answer vs `half`, G2 ≥4× scalar at 8 lanes, G4 zero-alloc).
//! Consumer after the gate: riir-engine `weight_tensor::dequantize_row`
//! BF16 arm (A4).
//!
//! # API
//!
//! | Function | Direction | Semantics |
//! |---|---|---|
//! | [`bf16_bits_to_f32_into`] | widen `u16` → `f32` | lossless identity shift |
//! | [`f32_to_bf16_rne_into`] | narrow `f32` → `u16` | round-to-nearest-even, bit-exact vs `half` |
//! | [`f32_to_bf16_trunc_into`] | narrow `f32` → `u16` | raw `>> 16`, biased fast arm |
//!
//! Each ships a public `*_scalar_into` twin — the portable reference and the
//! correctness oracle for the SIMD arms (the `simd_lut_dequant` convention);
//! consumers on fallback platforms can call it directly.
//!
//! # Length semantics
//!
//! Every public kernel `debug_assert_eq!(src.len(), dst.len())` and processes
//! `min(src.len(), dst.len())` elements. A length mismatch is a programming
//! error: loud in debug/test builds, a bounded partial conversion in release,
//! and NEVER UB — the SIMD bodies bound every raw-pointer access by
//! `n = min(src.len(), dst.len())`.
//!
//! # The RNE algorithm (bit-exact vs `half`)
//!
//! half 2.3.1–2.7.1 `bfloat/convert.rs::f32_to_bf16` (verified identical
//! across both versions; this workspace locks 2.7.1):
//!
//! ```text
//! NaN (x & 0x7FFF_FFFF > 0x7F80_0000): ((x >> 16) | 0x0040) as u16
//! else round: +1 iff (x & 0x8000) != 0 && (x & 0x17FFF) != 0, then x >> 16
//! ```
//!
//! The NaN special case is load-bearing: the branch-free classic
//! `((x + 0x7FFF + ((x >> 16) & 1)) >> 16)` maps sNaN payload 1
//! (`0x7F80_0001`) to `0x7F80` (+Inf) because its mantissa rounds UP — half
//! instead forces the qNaN bit and stays NaN. The scalar arm ships half's
//! literal form; the SIMD arms ship the add form + NaN select (per-lane
//! blend on `abs(x) > 0x7F80_0000`), equivalent for every non-NaN input
//! including the overflow-to-Inf carry. The G1 oracle (≥2²⁰ raw u32 patterns
//! + all-65536 exhaustive widening) proves scalar and SIMD bit-equal.
//!
//! # Widening vs `half`'s decoder
//!
//! The kernel is the pure identity shift `(bits as u32) << 16` (lossless —
//! value, ±0, Inf, NaN payload-shifted all preserved). `half`'s DECODER
//! (`bf16_to_f32`) additionally ORs `0x0040` into NaN payloads (its
//! qNaN-forcing convention). The G1 widen test therefore asserts
//! value/bit-equality against `half` for every non-NaN input, and NaN-class +
//! sign equality for NaN inputs — the divergence is a decode-layer
//! convention, not a conversion loss.
//!
//! # Truncation is biased by design (pufferlib's shape, not its rounding)
//!
//! [`f32_to_bf16_trunc_into`] is the raw `bits >> 16`. Worked examples:
//! `0x3F81_8000` (≈1.00787) truncates DOWN to `0x3F81` where RNE rounds the
//! tie up to `0x3F82` — truncation rounds toward zero, so over a uniform
//! mantissa distribution its mean absolute error is ½ ulp against RNE's ¼
//! ulp (twice the bias, and sign-correlated instead of symmetric). Worse,
//! truncation can COLLAPSE NaN→Inf: sNaN `0x7F80_0001` truncates to `0x7F80`
//! (+Inf), where RNE preserves the NaN (`0x7FC0`). Fast opt-in arm only.
//!
//! # Example
//!
//! ```rust
//! use katgpt_core::bf16_convert::{bf16_bits_to_f32_into, f32_to_bf16_rne_into};
//!
//! let bf16_bits: [u16; 4] = [0x3F80, 0xBF80, 0x7F80, 0x0000]; // 1.0, -1.0, Inf, +0.0
//! let mut f32s = [0f32; 4];
//! bf16_bits_to_f32_into(&bf16_bits, &mut f32s);
//! assert_eq!(f32s[0], 1.0);
//! assert_eq!(f32s[1], -1.0);
//! assert!(f32s[2].is_infinite());
//! assert_eq!(f32s[3], 0.0);
//!
//! let mut back = [0u16; 4];
//! f32_to_bf16_rne_into(&f32s, &mut back);
//! assert_eq!(back, bf16_bits);
//! ```
//!
//! bf16_convert — SIMD bf16⇄f32 batch conversion (Issue 800 Arm A). Widening
//! u16→f32 is lossless (bits<<16); narrowing ships RNE (bit-exact vs `half`)
//! with truncation as the explicit fast opt-in arm (pufferlib's scalar >>16
//! shape, biased — we take the kernel shape, not the rounding). NEON + AVX2
//! (target_feature-gated) + scalar fallback; into_buf APIs write into
//! caller-owned buffers, zero alloc. OPT-IN pending the Bench 800 GOAT gate
//! (≥4× the scalar loop at 8 lanes); consumer = riir-engine weight_tensor
//! dequantize_row BF16 arm.

// ──────────────────────────────────────────────────────────────────────────
// Single-element semantics — the spec every SIMD lane must reproduce.
// ──────────────────────────────────────────────────────────────────────────

/// Widening: a bf16 IS the top half of an f32 — lossless identity shift
/// (value, ±0, Inf, NaN payload-shifted all preserved).
#[inline(always)]
const fn widen_one(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// RNE narrowing, bit-exact vs `half::bf16::from_f32` — half 2.3.1–2.7.1
/// `bfloat/convert.rs::f32_to_bf16`, verified identical across both versions
/// (this workspace locks 2.7.1). This scalar form IS the reference; the SIMD
/// arms reproduce it with the add form + NaN select (module doc §RNE).
#[inline(always)]
const fn rne_one(bits: u32) -> u16 {
    // NaN: preserve sign, keep the high mantissa bits, force the qNaN bit.
    // Load-bearing: the branch-free add form alone maps sNaN payload 1
    // (0x7F80_0001) to 0x7F80 (+Inf) — half keeps it NaN at 0x7FC0.
    if bits & 0x7FFF_FFFF > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    const ROUND_BIT: u32 = 0x0000_8000;
    if (bits & ROUND_BIT) != 0 && (bits & (3 * ROUND_BIT - 1)) != 0 {
        ((bits >> 16) as u16) + 1
    } else {
        (bits >> 16) as u16
    }
}

/// Truncation narrowing (`bits >> 16`) — pufferlib's kernel shape, the fast
/// opt-in arm. Biased: rounds toward zero, and collapses sNaN `0x7F80_0001`
/// to `0x7F80` (+Inf) — see module doc §Truncation.
#[inline(always)]
const fn trunc_one(bits: u32) -> u16 {
    (bits >> 16) as u16
}

const LEN_MISMATCH: &str = "bf16_convert: src/dst length mismatch (kernels process min(len); debug/test builds flag the mismatch)";

// ──────────────────────────────────────────────────────────────────────────
// Public API — thin SIMD-dispatching kernels (Issue 800 A1 into_buf shape).
// ──────────────────────────────────────────────────────────────────────────

/// Widen bf16 bit patterns to f32, SIMD-dispatched (NEON on `aarch64`, AVX2
/// on `x86_64`+avx2, scalar elsewhere). Writes into caller-owned `dst`
/// (G4: zero alloc). Lossless — see [`widen_one`].
#[inline]
pub fn bf16_bits_to_f32_into(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len(), "{}", LEN_MISMATCH);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { bf16_bits_to_f32_neon(src, dst) }
    }
    // Issue 847 T6: a RUNTIME probe, for the same reason T2 gave the RNE
    // dispatcher one — `target_feature = "avx2"` is OFF by default on
    // x86_64, so this arm compiled to nothing on every ordinary build.
    //
    // ⛔ T2 left this kernel UNDECIDED and the reason turned out to be the
    // INSTRUMENT, not the kernel. The apparent "crossover" (a cell swinging
    // 97 ↔ 173 ns between two runs of one binary) is BUFFER ALIGNMENT: every
    // AVX2-bearing arm here is bimodal on `src.as_ptr() % 32` with the two
    // clusters 1.6-2.2x apart, `vec![]` picks the residue by chance, and the
    // scalar arm beside them is stable to under 1%. Comparing two arms across
    // uncontrolled allocations was comparing two coin flips.
    //
    // Measured with the residue CONTROLLED on a quiet box
    // (`t6_widen_bimodality_vs_buffer_alignment`, ns/call, release):
    //
    //   n      scalar (best..worst)   intrinsics (best..worst)
    //   256    13..14                 6..16
    //   4096   170..234               108..167
    //   32768  1699..2337             1833..2252
    //
    // So the intrinsics win 2.2x at 256 and 1.57x at 4096 on the favourable
    // residue and tie on the others, while n=32768 is memory-bound and a
    // wash (they lose ~1.1x at two residues of four and win at the other
    // two). ⚠ The deployed consumer is riir-engine's
    // `weight_tensor::dequantize_row` — a ROW, i.e. the 4096 regime — which
    // is what decides it; the large-n wash is STATED rather than hidden,
    // because a caller widening a whole tensor in one call is the case this
    // choice does not help.
    #[cfg(target_arch = "x86_64")]
    {
        // `if/else` and NO `return`: the cfg blocks are mutually exclusive,
        // so exactly one compiles and it is already the last statement
        // (`clippy::needless_return`, carried at zero residual here).
        if crate::simd::simd_level() == crate::simd::SimdLevel::Avx2 {
            unsafe { bf16_bits_to_f32_avx2(src, dst) }
        } else {
            bf16_bits_to_f32_scalar_into(src, dst);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        bf16_bits_to_f32_scalar_into(src, dst);
    }
}

/// Narrow f32 to bf16 bit patterns with round-to-nearest-EVEN, SIMD-
/// dispatched. Bit-exact vs `half::bf16::from_f32` for EVERY input class
/// including NaN (G1). Writes into caller-owned `dst` (G4: zero alloc).
#[inline]
pub fn f32_to_bf16_rne_into(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len(), "{}", LEN_MISMATCH);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { f32_to_bf16_rne_neon(src, dst) }
    }
    // Issue 847 T2: a RUNTIME probe, not a compile-time cfg. `target_feature
    // = "avx2"` is OFF by default on x86_64, so the arm below used to compile
    // to nothing on every ordinary build. Measured on this box (release,
    // ns/call): the intrinsics run 29 / 430 / 3495 at n = 256 / 4096 / 32768
    // against the scalar path's 66 / 1037 / 8582 — **2.3-2.5x**, stable
    // across two runs of both builds. `simd_level()` returns `Avx2` only for
    // AVX2+FMA, which is exactly what the kernel's `#[target_feature]`
    // requires.
    #[cfg(target_arch = "x86_64")]
    {
        // `if/else` and NO `return` — the three cfg blocks are mutually
        // exclusive, so exactly one compiles and it is already the last
        // statement. Both `return <call>;` and a trailing bare `return;` are
        // `clippy::needless_return` here, and this crate's gate carries that
        // lint at zero residual.
        if crate::simd::simd_level() == crate::simd::SimdLevel::Avx2 {
            unsafe { f32_to_bf16_rne_avx2(src, dst) }
        } else {
            f32_to_bf16_rne_scalar_into(src, dst);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        f32_to_bf16_rne_scalar_into(src, dst);
    }
}

/// Narrow f32 to bf16 bit patterns by TRUNCATION (pufferlib's kernel shape),
/// SIMD-dispatched. Biased toward zero and NaN-collapsing — see module doc
/// §Truncation; fast opt-in arm only, the default narrow is
/// [`f32_to_bf16_rne_into`]. Writes into caller-owned `dst` (G4: zero alloc).
#[inline]
pub fn f32_to_bf16_trunc_into(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len(), "{}", LEN_MISMATCH);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { f32_to_bf16_trunc_neon(src, dst) }
    }
    // No x86_64 arm, and that is a MEASUREMENT (Issue 847 T5) rather than an
    // omission: an AVX2 transcription of this kernel shipped here and was
    // deleted for losing to the scalar body it replaced — 1578 vs 1138 ns at
    // n=32768, 1.39x, stable over three runs and two build configurations,
    // with the `_autovec` arm (the SAME scalar body under `+avx2` codegen)
    // landing on the intrinsics at 1556-1565 rather than on the scalar. So it
    // is AVX2 CODEGEN for a bare `>> 16`, not a transcription defect: at the
    // default target LLVM already vectorises this loop into something the
    // 256-bit shift + cross-lane `packus` sequence cannot beat.
    // `t2_bf16_autovec_vs_intrinsics` keeps measuring it every run.
    #[cfg(not(target_arch = "aarch64"))]
    {
        f32_to_bf16_trunc_scalar_into(src, dst);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Scalar reference arms — portable fallback + the SIMD correctness oracle
// (the `simd_lut_dequant` convention: public so fallback-platform consumers
// and the G1 parity tests can call them directly).
// ──────────────────────────────────────────────────────────────────────────

/// Scalar widening — portable reference and the SIMD arms' oracle.
#[inline]
pub fn bf16_bits_to_f32_scalar_into(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len(), "{}", LEN_MISMATCH);
    for (bits, out) in src.iter().zip(dst.iter_mut()) {
        *out = widen_one(*bits);
    }
}

/// Scalar RNE narrowing — half's literal algorithm, portable reference and
/// the SIMD arms' oracle.
#[inline]
pub fn f32_to_bf16_rne_scalar_into(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len(), "{}", LEN_MISMATCH);
    for (x, out) in src.iter().zip(dst.iter_mut()) {
        *out = rne_one(x.to_bits());
    }
}

/// Scalar truncation narrowing — portable reference and the SIMD arms'
/// oracle.
#[inline]
pub fn f32_to_bf16_trunc_scalar_into(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len(), "{}", LEN_MISMATCH);
    for (x, out) in src.iter().zip(dst.iter_mut()) {
        *out = trunc_one(x.to_bits());
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Issue 847 T2 — the AUTOVECTORISED arm: the scalar body, compiled with AVX2
// available. Measured under `RUSTFLAGS=-C target-feature=+avx2`, the plain
// scalar loop beat the hand-written intrinsic arm at widening (97 ns vs 169
// at n=4096), which is a claim about the COMPILER rather than about either
// arm — and it is only reachable on a default build through
// `#[target_feature(enable = ..)]`, whose whole purpose is to compile a body
// for an ISA the build does not target.
//
// Gated on the ARCH alone, exactly as the repaired `simd_lut_dequant`
// kernels are: the attribute, not the cfg, is what makes the body compile.
// Callers must have checked `simd::simd_level()`.
// ──────────────────────────────────────────────────────────────────────────

/// SAFETY: requires AVX2. Behaviour is bit-identical to
/// [`bf16_bits_to_f32_scalar_into`] — same body, different codegen — so this
/// arm has NO correctness axis of its own. Widening is `u16 -> u32 << 16 ->
/// f32`, lossless by construction.
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bf16_bits_to_f32_autovec(src: &[u16], dst: &mut [f32]) {
    bf16_bits_to_f32_scalar_into(src, dst);
}

/// SAFETY: requires AVX2. Bit-identical to [`f32_to_bf16_rne_scalar_into`].
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn f32_to_bf16_rne_autovec(src: &[f32], dst: &mut [u16]) {
    f32_to_bf16_rne_scalar_into(src, dst);
}

/// SAFETY: requires AVX2. Bit-identical to [`f32_to_bf16_trunc_scalar_into`].
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn f32_to_bf16_trunc_autovec(src: &[f32], dst: &mut [u16]) {
    f32_to_bf16_trunc_scalar_into(src, dst);
}

// ──────────────────────────────────────────────────────────────────────────
// NEON backend (aarch64) — stable std::arch intrinsics, 8 elements/iter.
// ──────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::uint32x4_t;

/// RNE lane math (4 u32 lanes): half's NaN predicate + the add-form round +
/// the NaN select. Returns UNshifted results — the caller folds the `>> 16`
/// into the narrowing store (`vshrn_n_u32` = shift+narrow in one op).
///
/// Constants (pre-duplicated by the caller, loop-invariant):
/// `round_add = 0x7FFF`, `one = 1`, `abs_mask = 0x7FFF_FFFF`,
/// `exp_all = 0x7F80_0000`, `qnan_shifted = 0x0040_0000` — so
/// `(x | qnan_shifted) >> 16 == (x >> 16) | 0x0040`, half's NaN rule.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn rne_u32x4(
    x: uint32x4_t,
    round_add: uint32x4_t,
    one: uint32x4_t,
    abs_mask: uint32x4_t,
    exp_all: uint32x4_t,
    qnan_shifted: uint32x4_t,
) -> uint32x4_t {
    use core::arch::aarch64::{vaddq_u32, vandq_u32, vbslq_u32, vcgtq_u32, vorrq_u32, vshrq_n_u32};
    // SAFETY: plain NEON arithmetic; the intrinsics demand an unsafe context
    // because this fn does not itself declare #[target_feature] (the aarch64
    // build target guarantees the feature is present at runtime).
    unsafe {
        // NaN predicate: abs(x) > 0x7F80_0000 (half's exact rule; Inf is NOT
        // NaN).
        let is_nan = vcgtq_u32(vandq_u32(x, abs_mask), exp_all);
        // Add-form RNE: t = x + 0x7FFF + lsb_of_kept ⇒ t >> 16 == half's
        // round (tie rounds to even; mantissa/exponent carries propagate —
        // overflow to Inf included).
        let lsb = vandq_u32(vshrq_n_u32(x, 16), one);
        let t = vaddq_u32(x, vaddq_u32(round_add, lsb));
        // NaN result folded pre-shift: (x | 0x0040_0000) >> 16 =
        // (x >> 16) | 0x40.
        let nan_u = vorrq_u32(x, qnan_shifted);
        vbslq_u32(is_nan, nan_u, t)
    }
}

/// SAFETY: caller guarantees `src`/`dst` are valid for reads/writes of
/// `n = min(src.len(), dst.len())` elements (dispatch passes whole slices;
/// every access below is bounded by `n`).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn bf16_bits_to_f32_neon(src: &[u16], dst: &mut [f32]) {
    use core::arch::aarch64::{
        vget_high_u16, vget_low_u16, vld1q_u16, vmovl_u16, vreinterpretq_f32_u32, vshlq_n_u32,
        vst1q_f32,
    };

    unsafe {
        let n = src.len().min(dst.len());
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i + 8 <= n {
            let bits = vld1q_u16(sp.add(i));
            let lo = vmovl_u16(vget_low_u16(bits));
            let hi = vmovl_u16(vget_high_u16(bits));
            vst1q_f32(dp.add(i), vreinterpretq_f32_u32(vshlq_n_u32(lo, 16)));
            vst1q_f32(dp.add(i + 4), vreinterpretq_f32_u32(vshlq_n_u32(hi, 16)));
            i += 8;
        }
        while i < n {
            *dp.add(i) = widen_one(*sp.add(i));
            i += 1;
        }
    }
}

/// SAFETY: as [`bf16_bits_to_f32_neon`].
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn f32_to_bf16_rne_neon(src: &[f32], dst: &mut [u16]) {
    use core::arch::aarch64::{
        vcombine_u16, vdupq_n_u32, vld1q_f32, vreinterpretq_u32_f32, vshrn_n_u32, vst1q_u16,
    };

    unsafe {
        let n = src.len().min(dst.len());
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let round_add = vdupq_n_u32(0x7FFF);
        let one = vdupq_n_u32(1);
        let abs_mask = vdupq_n_u32(0x7FFF_FFFF);
        let exp_all = vdupq_n_u32(0x7F80_0000);
        let qnan_shifted = vdupq_n_u32(0x0040_0000);
        let mut i = 0;
        while i + 8 <= n {
            let x0 = vreinterpretq_u32_f32(vld1q_f32(sp.add(i)));
            let x1 = vreinterpretq_u32_f32(vld1q_f32(sp.add(i + 4)));
            let r0 = rne_u32x4(x0, round_add, one, abs_mask, exp_all, qnan_shifted);
            let r1 = rne_u32x4(x1, round_add, one, abs_mask, exp_all, qnan_shifted);
            let packed = vcombine_u16(vshrn_n_u32(r0, 16), vshrn_n_u32(r1, 16));
            vst1q_u16(dp.add(i), packed);
            i += 8;
        }
        while i < n {
            *dp.add(i) = rne_one((*sp.add(i)).to_bits());
            i += 1;
        }
    }
}

/// SAFETY: as [`bf16_bits_to_f32_neon`].
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn f32_to_bf16_trunc_neon(src: &[f32], dst: &mut [u16]) {
    use core::arch::aarch64::{
        vcombine_u16, vld1q_f32, vreinterpretq_u32_f32, vshrn_n_u32, vst1q_u16,
    };

    unsafe {
        let n = src.len().min(dst.len());
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i + 8 <= n {
            let x0 = vreinterpretq_u32_f32(vld1q_f32(sp.add(i)));
            let x1 = vreinterpretq_u32_f32(vld1q_f32(sp.add(i + 4)));
            // vshrn = shift right 16 + narrow, one op per 4 lanes.
            let packed = vcombine_u16(vshrn_n_u32(x0, 16), vshrn_n_u32(x1, 16));
            vst1q_u16(dp.add(i), packed);
            i += 8;
        }
        while i < n {
            *dp.add(i) = trunc_one((*sp.add(i)).to_bits());
            i += 1;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// AVX2 backend (x86_64 + target_feature = "avx2") — 8 elements/iter.
//
// AVX2 pack note (the documented lane-hazard fix): `_mm256_packus_epi32`
// packs WITHIN each 128-bit lane — with two full-width operands the u16
// order comes out scrambled ([a0..a3, b0..b3, a4..a7, b4..b7] for operands
// a/b). We use the two-__m128i route instead: split the 8 result lanes with
// cast/extracti128 and pack with SSE `_mm_packus_epi32` (packusdw), whose
// result is LINEAR ([a0..a3, b0..b3]) inside a single 128-bit register —
// no 256-bit lane crossing, no permute4x64 fixup. packusdw saturation is a
// non-issue: post-shift lanes are ≤ 0xFFFF by construction, exactly
// packusdw's unsigned clamp bound.
// ──────────────────────────────────────────────────────────────────────────

/// SAFETY: caller guarantees `src`/`dst` are valid for reads/writes of
/// `n = min(src.len(), dst.len())` elements (dispatch passes whole slices;
/// every access below is bounded by `n`).
// Gated on the ARCH alone (Issue 847 T6): `#[target_feature(enable = "avx2")]`
// is what makes the intrinsic body compile, and the cfg is what used to make
// it compile to NOTHING on a default build. The runtime probe above is what
// makes entering it safe.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn bf16_bits_to_f32_avx2(src: &[u16], dst: &mut [f32]) {
    use core::arch::x86_64::{
        _mm_loadu_si128, _mm256_castsi256_ps, _mm256_cvtepu16_epi32, _mm256_slli_epi32,
        _mm256_storeu_ps,
    };

    unsafe {
        let n = src.len().min(dst.len());
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0;
        while i + 8 <= n {
            let bits = _mm_loadu_si128(sp.add(i) as *const core::arch::x86_64::__m128i);
            let wide = _mm256_cvtepu16_epi32(bits);
            let shifted = _mm256_slli_epi32(wide, 16);
            _mm256_storeu_ps(dp.add(i), _mm256_castsi256_ps(shifted));
            i += 8;
        }
        while i < n {
            *dp.add(i) = widen_one(*sp.add(i));
            i += 1;
        }
    }
}

/// SAFETY: as [`bf16_bits_to_f32_avx2`].
// Issue 847 T2: ARCH-only cfg. The `#[target_feature]` attribute, not the
// cfg, is what makes the intrinsic body compile; gating on the compile-time
// feature is what made this kernel unreachable on every ordinary build.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn f32_to_bf16_rne_avx2(src: &[f32], dst: &mut [u16]) {
    use core::arch::x86_64::{
        __m128i, _mm_packus_epi32, _mm_storeu_si128, _mm256_add_epi32, _mm256_and_si256,
        _mm256_blendv_epi8, _mm256_castps_si256, _mm256_castsi256_si128, _mm256_cmpgt_epi32,
        _mm256_extracti128_si256, _mm256_loadu_ps, _mm256_or_si256, _mm256_set1_epi32,
        _mm256_srli_epi32,
    };

    unsafe {
        let n = src.len().min(dst.len());
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let round_add = _mm256_set1_epi32(0x7FFF);
        let one = _mm256_set1_epi32(1);
        let abs_mask = _mm256_set1_epi32(0x7FFF_FFFF_i32);
        let exp_all = _mm256_set1_epi32(0x7F80_0000_i32);
        let qnan_shifted = _mm256_set1_epi32(0x0040_0000_i32);
        let mut i = 0;
        while i + 8 <= n {
            let x = _mm256_castps_si256(_mm256_loadu_ps(sp.add(i)));
            // NaN predicate in the int domain: abs(x) > 0x7F80_0000 (half's
            // exact rule; operands are ≤ 0x7FFFFFFF so signed cmp is sound).
            let is_nan = _mm256_cmpgt_epi32(_mm256_and_si256(x, abs_mask), exp_all);
            // Add-form RNE (same identity as the NEON arm).
            let lsb = _mm256_and_si256(_mm256_srli_epi32(x, 16), one);
            let t = _mm256_add_epi32(x, _mm256_add_epi32(round_add, lsb));
            // NaN result folded pre-shift (same identity as the NEON arm).
            let nan_u = _mm256_or_si256(x, qnan_shifted);
            let sel = _mm256_blendv_epi8(t, nan_u, is_nan);
            // The NEON identity ends at `t >> 16` — shift BEFORE the 32→16
            // pack: `_mm_packus_epi32` SATURATES i32→u16, it does not take the
            // high half, so unshifted t packs as 0x0000/0x7FFF/0xFFFF mask
            // garbage (caught by the first x86_64 execution of this arm,
            // 2026-09-16 — NEON was never wrong; the shift was dropped in the
            // AVX2 transcription).
            let sel16 = _mm256_srli_epi32(sel, 16);
            // Two-__m128i pack: linear order, no 256-bit lane hazard (module
            // doc §AVX2 pack note).
            let lo = _mm256_castsi256_si128(sel16);
            let hi = _mm256_extracti128_si256::<1>(sel16);
            let packed = _mm_packus_epi32(lo, hi);
            _mm_storeu_si128(dp.add(i) as *mut __m128i, packed);
            i += 8;
        }
        while i < n {
            *dp.add(i) = rne_one((*sp.add(i)).to_bits());
            i += 1;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// G1/G4 self-tests (Issue 800 A3). The oracle is the REAL `half` crate —
// raw-bits comparison including the NaN class — never a re-implementation.
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-picked vectors spanning every input class (G1): zeros, ones,
    /// round-up cases, f32 denormals, Inf, qNaN/sNaN ±, exact ties
    /// (even/odd kept lsb), sticky-above-tie, overflow-to-Inf.
    /// NOTE: 65504.0/65520.0 (0x477F_E000/0x477F_F000) exercise round-UP but
    /// land on 65536.0 — they do NOT reach Inf (bf16 spans f32's exponent);
    /// the real Inf-overflow case is 0x7F7F_FFFF.
    #[test]
    fn g1_rne_hand_vectors_bit_exact_vs_half() {
        const HAND: &[u32] = &[
            0x0000_0000, // +0.0
            0x8000_0000, // -0.0
            0x3F80_0000, // 1.0
            0xBF80_0000, // -1.0
            0x477F_E000, // 65504.0 — round-up case
            0x477F_F000, // 65520.0 — round-up case
            0x0080_0000, // f32::MIN_POSITIVE
            0x0000_0001, // smallest f32 denormal
            0x007F_FFFF, // largest f32 denormal
            0x7F80_0000, // +Inf (must stay 0x7F80)
            0xFF80_0000, // -Inf
            0x7FC0_0000, // +qNaN
            0xFFC0_0000, // -qNaN
            0x7F80_0001, // sNaN payload 1 — canonical naive-algo-wrong case
            0xFF80_0001, // -sNaN payload 1
            0x7F7F_8000, // max bf16-representable finite (exact)
            0x7F7F_FFFF, // overflows to Inf on RNE
            0x4000_8000, // exact tie, kept lsb 0 → round DOWN (even)
            0x4001_8000, // exact tie, kept lsb 1 → round UP (even)
            0x4000_8001, // above tie (sticky) → round UP
            0x3F7F_8000,
            0x3F80_7FFF, // below half → down
            0x3F80_8000, // exact tie, kept lsb 0 → down
            0x3F81_8000, // exact tie, kept lsb 1 → up
        ];
        for &bits in HAND {
            let x = f32::from_bits(bits);
            let oracle = half::bf16::from_f32(x).to_bits();
            assert_eq!(rne_one(bits), oracle, "scalar rne(0x{bits:08x})");
        }
        // dispatch parity over the same hand set (exercises the SIMD arm here)
        let src: Vec<f32> = HAND.iter().map(|&b| f32::from_bits(b)).collect();
        let mut dst = vec![0u16; src.len()];
        f32_to_bf16_rne_into(&src, &mut dst);
        for (i, &bits) in HAND.iter().enumerate() {
            assert_eq!(dst[i], rne_one(bits), "dispatch rne(0x{bits:08x})");
        }
        // the load-bearing cases, asserted directly (independent of half)
        assert_eq!(rne_one(0x7F80_0001), 0x7FC0, "sNaN must stay NaN, NOT Inf");
        assert_eq!(rne_one(0x7F7F_FFFF), 0x7F80, "finite overflow → Inf");
        assert_eq!(rne_one(0x4000_8000), 0x4000, "tie rounds DOWN to even");
        assert_eq!(rne_one(0x4001_8000), 0x4002, "tie rounds UP to even");
        assert_eq!(rne_one(0x4000_8001), 0x4001, "above tie rounds UP");
        assert_eq!(rne_one(0x7F80_0000), 0x7F80, "Inf stays Inf");
        assert_eq!(rne_one(0xFF80_0000), 0xFF80, "-Inf stays -Inf");
    }

    /// G1 oracle sweep: 2²⁰ pseudo-random raw u32 bit patterns (fastrand,
    /// fixed seed) — every class hit including NaN (~0.4% of patterns) —
    /// compared RAW-BITS vs `half` for BOTH the scalar arm and the dispatch
    /// (on aarch64 the dispatch IS the NEON arm, so lane-order bugs surface
    /// here for free).
    #[test]
    fn g1_rne_pseudorandom_bit_exact_vs_half() {
        let mut rng = fastrand::Rng::with_seed(0x800_C0FFEE);
        let mut src = [0f32; 4096];
        let mut dst_scalar = [0u16; 4096];
        let mut dst_simd = [0u16; 4096];
        for _ in 0..256 {
            for v in src.iter_mut() {
                *v = f32::from_bits(rng.u32(..));
            }
            f32_to_bf16_rne_scalar_into(&src, &mut dst_scalar);
            f32_to_bf16_rne_into(&src, &mut dst_simd);
            for (i, &x) in src.iter().enumerate() {
                let oracle = half::bf16::from_f32(x).to_bits();
                assert_eq!(dst_scalar[i], oracle, "scalar rne(0x{:08x})", x.to_bits());
                assert_eq!(dst_simd[i], oracle, "dispatch rne(0x{:08x})", x.to_bits());
            }
        }
    }

    /// Every NaN sign/payload variant maps to half's qNaN-forcing rule:
    /// `((bits >> 16) | 0x0040) as u16` — sign preserved, high payload kept,
    /// quiet bit set, result stays NaN-class.
    #[test]
    fn g1_rne_nan_class_semantics() {
        const NANS: &[u32] = &[
            0x7F80_0001,
            0x7F80_0002,
            0x7F80_4000,
            0x7F80_8000,
            0x7FBF_FFFF,
            0x7FC0_0000,
            0x7FFF_FFFF,
            0xFF80_0001,
            0xFFC0_0000,
            0xFFFF_FFFF,
            0x7F81_2345,
            0xFF81_2345,
        ];
        for &bits in NANS {
            assert!(
                bits & 0x7FFF_FFFF > 0x7F80_0000,
                "fixture 0x{bits:08x} must be NaN-class"
            );
            let x = f32::from_bits(bits);
            let ours = rne_one(bits);
            let oracle = half::bf16::from_f32(x).to_bits();
            assert_eq!(ours, oracle, "rne NaN(0x{bits:08x})");
            assert_eq!(ours & 0x7F80, 0x7F80, "result stays NaN-class");
            assert_eq!(ours & 0x0040, 0x0040, "qNaN bit forced");
            assert_eq!(ours & 0x8000, ((bits >> 31) as u16) << 15, "sign preserved");
        }
    }

    /// Widening is EXHAUSTIVELY verified: all 65536 bf16 bit patterns,
    /// scalar + dispatch. Non-NaN inputs must be bit-equal to `half`'s
    /// decode; NaN inputs compared NaN-class + sign (half's decoder ORs
    /// 0x0040 into NaN payloads — a decode-layer convention, not a
    /// conversion loss; our kernel is the pure identity shift, module doc
    /// §Widening).
    #[test]
    fn g1_widen_exhaustive_matches_half_decode() {
        let mut src = [0u16; 65536];
        for (i, v) in src.iter_mut().enumerate() {
            *v = i as u16;
        }
        let mut dst_scalar = [0f32; 65536];
        let mut dst_simd = [0f32; 65536];
        bf16_bits_to_f32_scalar_into(&src, &mut dst_scalar);
        bf16_bits_to_f32_into(&src, &mut dst_simd);
        for (i, (&s, &d)) in dst_scalar.iter().zip(dst_simd.iter()).enumerate() {
            let i = i as u16;
            let oracle = half::bf16::from_bits(i).to_f32();
            assert_eq!(s.to_bits(), d.to_bits(), "widen parity at 0x{i:04x}");
            if oracle.is_nan() {
                assert!(s.is_nan(), "widen(0x{i:04x}) should be NaN");
                assert_eq!(s.is_sign_negative(), oracle.is_sign_negative());
            } else {
                assert_eq!(s.to_bits(), oracle.to_bits(), "widen(0x{i:04x})");
            }
        }
    }

    /// Truncation IS the raw shift: out bits == (bits >> 16) as u16 — scalar
    /// and dispatch. Worked examples document the bias (rounds toward zero;
    /// mean |error| ½ ulp vs RNE's ¼ ulp — the "doubles positive bias"
    /// ledger line) and the NaN collapse hazard (sNaN → ±Inf under trunc).
    #[test]
    fn g1_trunc_is_bits_shift_with_bias_worked_examples() {
        let mut rng = fastrand::Rng::with_seed(0x800_7A0);
        let mut src = [0f32; 1024];
        let mut dst_scalar = [0u16; 1024];
        let mut dst_simd = [0u16; 1024];
        for _ in 0..8 {
            for v in src.iter_mut() {
                *v = f32::from_bits(rng.u32(..));
            }
            f32_to_bf16_trunc_scalar_into(&src, &mut dst_scalar);
            f32_to_bf16_trunc_into(&src, &mut dst_simd);
            for (i, &x) in src.iter().enumerate() {
                let expect = (x.to_bits() >> 16) as u16;
                assert_eq!(dst_scalar[i], expect, "scalar trunc");
                assert_eq!(dst_simd[i], expect, "dispatch trunc");
            }
        }
        // worked bias example: exact tie — RNE rounds up, trunc drops it
        assert_eq!(trunc_one(0x3F81_8000), 0x3F81, "trunc rounds toward zero");
        assert_eq!(rne_one(0x3F81_8000), 0x3F82, "RNE rounds the tie up");
        assert_eq!(trunc_one(0xBF81_8000), 0xBF81, "negative: magnitude down");
        assert_eq!(rne_one(0xBF81_8000), 0xBF82, "negative: RNE magnitude up");
        // NaN collapse hazard: trunc can turn sNaN into Inf; RNE cannot
        assert_eq!(trunc_one(0x7F80_0001), 0x7F80, "sNaN COLLAPSES to +Inf");
        assert_eq!(rne_one(0x7F80_0001), 0x7FC0, "RNE preserves NaN");
    }

    /// Round-trip sanity: widen(rne(x)) must decode to exactly
    /// `half::f32::from_bf16(half::bf16::from_f32(x))` — bit-exact for every
    /// class (rne's NaN outputs already carry the qNaN bit, so half's
    /// qNaN-forcing decode is idempotent on them).
    #[test]
    fn g1_round_trip_widen_rne_bit_exact_vs_half() {
        let mut rng = fastrand::Rng::with_seed(0x800_BABE);
        for _ in 0..(1 << 17) {
            let x = f32::from_bits(rng.u32(..));
            let widened = widen_one(rne_one(x.to_bits()));
            let oracle = half::bf16::from_f32(x).to_f32();
            assert_eq!(widened.to_bits(), oracle.to_bits(), "round trip");
        }
    }

    /// G4 witness: every kernel runs on stack-allocated fixed-size buffers —
    /// the APIs are slice→slice by construction (no Vec/Box/format! in the
    /// hot path).
    #[test]
    fn g4_stack_buffers_zero_alloc_witness() {
        let mut src16: [u16; 64] = [0; 64];
        for (i, v) in src16.iter_mut().enumerate() {
            *v = (i as u16).wrapping_mul(379);
        }
        let mut dst32: [f32; 64] = [0.0; 64];
        bf16_bits_to_f32_into(&src16, &mut dst32);
        assert_eq!(dst32[0], f32::from_bits((src16[0] as u32) << 16));
        assert_eq!(dst32[63], f32::from_bits((src16[63] as u32) << 16));

        let mut src32: [f32; 64] = [0.0; 64];
        for (i, v) in src32.iter_mut().enumerate() {
            *v = f32::from_bits((i as u32).wrapping_mul(0x0100_0001));
        }
        let mut dst16: [u16; 64] = [0; 64];
        f32_to_bf16_rne_into(&src32, &mut dst16);
        f32_to_bf16_trunc_into(&src32, &mut dst16);
        for i in 0..64 {
            assert_eq!(dst16[i], rne_one(src32[i].to_bits()));
        }
    }

    /// SIMD dispatch vs scalar reference must be bit-equal on every platform
    /// (on aarch64 this drives the NEON arms; on fallback platforms dispatch
    /// delegates to scalar, so this is vacuous-but-harmless there). Odd
    /// length 257 forces the scalar-tail path every iteration.
    #[test]
    fn g1_dispatch_parity_scalar() {
        let mut rng = fastrand::Rng::with_seed(0x800_FA11);
        let mut src32 = [0f32; 257];
        let mut src16 = [0u16; 257];
        let mut a16 = [0u16; 257];
        let mut b16 = [0u16; 257];
        let mut a32 = [0f32; 257];
        let mut b32 = [0f32; 257];
        for _ in 0..16 {
            for v in src32.iter_mut() {
                *v = f32::from_bits(rng.u32(..));
            }
            for v in src16.iter_mut() {
                *v = rng.u16(..);
            }
            f32_to_bf16_rne_scalar_into(&src32, &mut a16);
            f32_to_bf16_rne_into(&src32, &mut b16);
            assert_eq!(a16, b16, "rne narrow parity");
            f32_to_bf16_trunc_scalar_into(&src32, &mut a16);
            f32_to_bf16_trunc_into(&src32, &mut b16);
            assert_eq!(a16, b16, "trunc narrow parity");
            bf16_bits_to_f32_scalar_into(&src16, &mut a32);
            bf16_bits_to_f32_into(&src16, &mut b32);
            assert!(
                a32.iter()
                    .zip(b32.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "widen parity"
            );
        }
    }

    /// Direct NEON-vs-scalar parity on a mixed-class buffer — calls the
    /// internal NEON arms directly (not through dispatch). Runs on THIS
    /// machine (M3) — the test that actually executes the NEON lanes against
    /// the scalar reference.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn g1_neon_direct_parity_scalar() {
        const N: usize = 512;
        let specials: [u32; 15] = [
            0x0000_0000,
            0x8000_0000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0000,
            0x7F80_0001,
            0xFF80_0001,
            0x0000_0001,
            0x007F_FFFF,
            0x7F7F_FFFF,
            0x4000_8000,
            0x4001_8000,
            0x3F80_8000,
            0x3F81_8000,
            0x477F_E000,
        ];
        let mut src32 = [0f32; N];
        for (i, v) in src32.iter_mut().enumerate() {
            *v = if i % 7 == 0 {
                f32::from_bits(specials[(i / 7) % specials.len()])
            } else {
                f32::from_bits(0x3E80_0000 | ((i as u32).wrapping_mul(0x0001_009D)))
            };
        }
        let mut src16 = [0u16; N];
        for (i, v) in src16.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(0x9E37_79B1) >> 13) as u16;
        }
        let mut ref16 = [0u16; N];
        let mut neon16 = [0u16; N];
        let mut ref32 = [0f32; N];
        let mut neon32 = [0f32; N];

        f32_to_bf16_rne_scalar_into(&src32, &mut ref16);
        unsafe { f32_to_bf16_rne_neon(&src32, &mut neon16) };
        assert_eq!(ref16, neon16, "NEON rne parity");

        f32_to_bf16_trunc_scalar_into(&src32, &mut ref16);
        unsafe { f32_to_bf16_trunc_neon(&src32, &mut neon16) };
        assert_eq!(ref16, neon16, "NEON trunc parity");

        bf16_bits_to_f32_scalar_into(&src16, &mut ref32);
        unsafe { bf16_bits_to_f32_neon(&src16, &mut neon32) };
        assert!(
            ref32
                .iter()
                .zip(neon32.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "NEON widen parity"
        );
    }

    #[test]
    fn empty_slices_are_noops() {
        let mut a: [f32; 0] = [];
        let mut b: [u16; 0] = [];
        bf16_bits_to_f32_into(&b, &mut a);
        bf16_bits_to_f32_scalar_into(&b, &mut a);
        f32_to_bf16_rne_into(&a, &mut b);
        f32_to_bf16_rne_scalar_into(&a, &mut b);
        f32_to_bf16_trunc_into(&a, &mut b);
        f32_to_bf16_trunc_scalar_into(&a, &mut b);
    }
}
