//! Recurrent Residual Quantization (RRQ) — single-checkpoint multi-precision
//! weight representation via iterated 2-bit residual corrections.
//!
//! > **Source:** Luo, Dong, Cheng, Shen (Intel),
//! > *Recurrent Residual Quantization: A Progressive Multi-Precision
//! > Representation for LLMs*,
//! > [arXiv:2608.04048](https://arxiv.org/abs/2608.04048), Aug 2026.
//!
//! # The representation
//!
//! An RRQ weight matrix `W̃` is stored as `S+1` quantized tensors: one base
//! stage `Ŵ0` + `S` residual stages `R̂1..R̂S`. Each stage is a 2-bit
//! round-to-nearest (RTN) quantized tensor with per-group scale + zero-point.
//! The k-th residual is the quantization error of the (k−1)-th reconstructed
//! approximation:
//!
//! ```text
//! Q_j^0   = Q_{b0}(x, z_0)              # base: 2-bit RTN
//! r_j^0   = x_j − x̂_j^0                 # residual = full − base dequant
//! Q_j^k   = Q_{bk}(r_j^{k-1}, z_k)      # stage k: 2-bit RTN on previous residual
//! r_j^k   = r_j^{k-1} − r̂_j^k           # next residual = prev residual − stage dequant
//! W̃(t)   = Ŵ0 + Σ_{k=1..t} R̂k           # prefix-t reconstruction (additive)
//! ```
//!
//! Default config: `b0 = b1 = b2 = b3 = 2` → 2/4/6/8-bit prefixes. All stages
//! use the same group size (128) and the same per-stage quantizer (RTN). The
//! paper also reports a stronger `SignRoundV2-base` variant — the only place
//! a learned rounding operator enters; the all-RTN variant here tracks it
//! within 0.1 Task Avg at 6/8 bits.
//!
//! # Why modelless
//!
//! RRQ is explicitly **post-training quantization** (PTQ). The construction
//! is pure RTN — no Hessian, no calibration data, no joint multi-bit
//! optimization, no gradient descent. Each stage is independent: adding a
//! precision prefix requires only "configure stage format", not "re-run joint
//! optimization". The §3.5 modelless-unblock check is moot — there is nothing
//! to redirect to riir-train.
//!
//! # The matmul linearity (prefix-t GEMV)
//!
//! Because `W̃(t) = Σ_{k=0..t} stage_k` and matrix multiplication is linear:
//!
//! ```text
//! x · W̃(t) = x · Ŵ0 + Σ_{k=1..t} x · R̂k
//! ```
//!
//! So the prefix-t dot product decomposes into a sum of per-stage GEMVs.
//! Each stage GEMV can use the same SIMD LUT dequant+dot kernel (the
//! natural fusion target for Phase 3). The plan ships the additive primitive
//! here; Phase 3 shipped the fused kernel but its G2 gate FAILED honestly
//! (the 4-stage LUT path is ~6× slower than single-8-bit at parity — see
//! [`RrqWeights::prefix_dot_lut_into`] doc + Plan 568 Phase 3). The arithmetic
//! [`RrqWeights::prefix_dot_into`] remains the recommended hot path.
//!
//! # What this is NOT
//!
//! - **NOT a UQ primitive.** RRQ is a deterministic weight representation,
//!   not a probability distribution. No conformal-naive floor comparison.
//! - **NOT SignRoundV2.** The learned-base variant is a training-method
//!   artifact; this module ships only the all-RTN modelless path.
//! - **NOT Matryoshka / MatGPTQ.** RRQ explicitly replaces nested MSB
//!   bit-slicing with additive residuals. Different mechanism, same problem
//!   (multi-precision from one checkpoint).
//! - **NOT a consumer-ready tier dispatch.** Phase 4 (prefix-t as Plasma →
//!   Hot → Warm tier) is stretch, deferred until a concrete consumer lands.
//!
//! # The Small-Kernel Parameter Paradox (Research 463 §2.4.1)
//!
//! Same caveat as `quant_error_lora.rs`: on small CNNs (Moka-scale), each
//! 2-bit residual stage adds 0.5 bits/weight of codes + per-group scale
//! metadata; for a 32×288 conv that's substantial overhead relative to the
//! 9.2K parameters. RRQ is substrate for larger models (LLM weights, future
//! game networks) where the per-group scale metadata is amortized across
//! many weights.
//!
//! # Cross-references
//!
//! - [Research 467](../.research/467_Recurrent_Residual_Quantization.md) —
//!   the parent research note (verdict: Gain, not Super-GOAT — no consumer).
//! - [Plan 568](../.plans/568_recurrent_residual_quantization.md) — execution
//!   plan. Phase 1 ships here; Phases 2–4 are P1–P3 (deferred).
//! - [`quant_error_lora`](../src/quant_error_lora.rs) — closest cousin
//!   (single SVD correction vs N iterated RTN corrections).
//! - [`simd_lut_dequant`](../src/simd_lut_dequant.rs) — Phase 3 fusion
//!   target (fused multi-stage LUT dequant+dot kernel).
//!
//! Plan 568: Recurrent Residual Quantization (RRQ) — single-checkpoint
//! multi-precision weight representation via iterated 2-bit RTN residual
//! corrections (Luo et al. Intel, arXiv:2608.04048 Aug 2026; Research 467).
//! W̃(t) = Ŵ0 + Σ residuals — base + N stages, each 2-bit RTN with per-group
//! f16 scale + zero-point. Default 1+3 → 2/4/6/8-bit prefixes. Pure modelless
//! PTQ (no Hessian, no calibration). prefix_dot_into exploits matmul linearity.
//! Verdict: Gain (not Super-GOAT — no concrete consumer today). Opt-in until a
//! multi-precision LLM / per-NPC expert base / incremental-upgrade consumer lands.

use half::f16;

// ──────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────

/// Bits per stage code. The paper's default is 2-bit (4 levels) for every
/// stage; this gives 2/4/6/8-bit prefixes for the 1-base + 3-residual config.
pub const BITS_PER_STAGE: u8 = 2;

/// Number of quantization levels per stage = `2^BITS_PER_STAGE` = 4.
pub const LEVELS_PER_STAGE: usize = 1 << BITS_PER_STAGE;

/// Number of 2-bit codes packed into one byte = `8 / BITS_PER_STAGE` = 4.
pub const CODES_PER_BYTE: usize = 8 / BITS_PER_STAGE as usize;

/// Default group size for per-group scale + zero-point (paper default + our
/// existing codec convention).
pub const DEFAULT_GROUP_SIZE: usize = 128;

/// Default number of residual stages (1 base + 3 residuals → 2/4/6/8-bit
/// prefixes).
pub const DEFAULT_N_STAGES: usize = 3;

// ──────────────────────────────────────────────────────────────────────────
// Storage types
// ──────────────────────────────────────────────────────────────────────────

/// One RRQ stage: a 2-bit RTN quantized tensor with per-group scale +
/// zero-point.
///
/// The codes are packed 4-per-byte (2 bits each). Group size defaults to 128
/// (so 32 bytes of codes per group). Scales + zero-points are stored as `f16`
/// to halve metadata overhead (the paper's intent; Appendix G shows ~4–5%
/// metadata overhead vs codes).
#[derive(Clone)]
pub struct RrqStage {
    /// 2-bit packed codes, 4 per byte. `codes[i / 4]` holds code `i` in bits
    /// `(i % 4) * 2 .. (i % 4) * 2 + 2`. Code values are 0..=3.
    pub codes: Vec<u8>,
    /// Per-group scale, `f16`. `scales[g]` is the scale for group `g`.
    pub scales: Vec<f16>,
    /// Per-group zero-point, `f16`. The dequant formula is
    /// `x_hat = zero_point + code * scale`.
    pub zero_points: Vec<f16>,
    /// Number of weights represented (= `rows * cols` for a full stage).
    pub n_elements: usize,
    /// Group size (default 128). The last group may be partial.
    pub group_size: usize,
}

impl RrqStage {
    /// Number of complete + partial groups.
    #[allow(dead_code)]
    fn n_groups(&self) -> usize {
        self.n_elements.div_ceil(self.group_size)
    }

    /// Quantize a slice of f32 values into a 2-bit RTN stage.
    ///
    /// Per group: `scale = (max - min) / (LEVELS - 1)`,
    /// `zero_point = min`, `code = round((x - zp) / scale)` clamped to
    /// `[0, LEVELS-1]`. Dequant: `x_hat = zp + code * scale`.
    ///
    /// **Allocation:** this constructor allocates (it builds the owned
    /// `codes` + `scales` + `zero_points` Vecs). It runs once at model load,
    /// NOT on the inference hot path. The hot-path methods
    /// ([`dequant_into`](Self::dequant_into),
    /// [`dot_acc_into`](Self::dot_acc_into)) are zero-allocation.
    fn quantize_rtn(values: &[f32], group_size: usize) -> Self {
        let n = values.len();
        let n_groups = n.div_ceil(group_size);
        let n_code_bytes = n.div_ceil(CODES_PER_BYTE);
        let mut codes = vec![0u8; n_code_bytes];
        let mut scales = Vec::with_capacity(n_groups);
        let mut zero_points = Vec::with_capacity(n_groups);

        for g in 0..n_groups {
            let start = g * group_size;
            let end = (start + group_size).min(n);
            // Find min/max of this group.
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &v in &values[start..end] {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            // scale = (max - min) / (LEVELS - 1). Guard against degenerate
            // (all-equal) groups: scale = 0 → every code = 0, dequant = min.
            let range = max - min;
            let scale = if range > 0.0 {
                range / (LEVELS_PER_STAGE as f32 - 1.0)
            } else {
                0.0
            };
            let zp = min;
            scales.push(f16::from_f32(scale));
            zero_points.push(f16::from_f32(zp));

            // Quantize each value in the group.
            let scale_recip = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for (i, &v) in values[start..end].iter().enumerate() {
                let code_f = (v - zp) * scale_recip;
                // Round + clamp to [0, LEVELS-1].
                let code = if code_f <= 0.0 {
                    0u8
                } else if code_f >= (LEVELS_PER_STAGE - 1) as f32 {
                    (LEVELS_PER_STAGE - 1) as u8
                } else {
                    (code_f + 0.5) as u8
                };
                // Pack into the codes byte array.
                let idx = start + i;
                let byte_idx = idx / CODES_PER_BYTE;
                let shift = (idx % CODES_PER_BYTE) * 2; // 2 bits per code
                codes[byte_idx] |= code << shift;
            }
        }

        Self {
            codes,
            scales,
            zero_points,
            n_elements: n,
            group_size,
        }
    }

    /// Unpack the 2-bit code at element index `i`.
    #[inline]
    pub fn code_at(&self, i: usize) -> u8 {
        let byte_idx = i / CODES_PER_BYTE;
        let shift = (i % CODES_PER_BYTE) * 2;
        (self.codes[byte_idx] >> shift) & 0x03
    }

    /// Dequantize the code at element index `i` to f32.
    #[inline]
    pub fn dequant_at(&self, i: usize) -> f32 {
        let code = self.code_at(i) as f32;
        let g = i / self.group_size;
        let scale = self.scales[g].to_f32();
        let zp = self.zero_points[g].to_f32();
        zp + code * scale
    }

    /// Dequantize all codes into `out`. `out.len()` must be `>= n_elements`.
    ///
    /// **Zero allocation.** Writes into caller-provided `out`.
    pub fn dequant_into(&self, out: &mut [f32]) {
        assert!(
            out.len() >= self.n_elements,
            "dequant_into: out.len() ({}) < n_elements ({})",
            out.len(),
            self.n_elements
        );
        for (out_i, i) in out.iter_mut().take(self.n_elements).zip(0..self.n_elements) {
            *out_i = self.dequant_at(i);
        }
    }

    /// Accumulate `x · dequant(codes)` into `out` for one stage.
    ///
    /// Treats this stage as a `[rows × cols]` row-major weight matrix (the
    /// layout used by `from_weights_rtn`). Computes
    /// `out[o] += Σ_i x[i] * stage_dequant[o * cols + i]` for each output
    /// row `o`.
    ///
    /// **Zero allocation.** Reads `x` and `out`, dequants codes on the fly.
    ///
    /// **Panics** if `x.len() != cols` or `out.len() != rows` (where
    /// `rows = n_elements / cols`).
    pub fn dot_acc_into(&self, cols: usize, x: &[f32], out: &mut [f32]) {
        let n_elements = self.n_elements;
        assert!(
            n_elements.is_multiple_of(cols),
            "dot_acc_into: n_elements ({n_elements}) not divisible by cols ({cols})"
        );
        let rows = n_elements / cols;
        assert_eq!(
            x.len(),
            cols,
            "dot_acc_into: x.len() ({}) != cols ({})",
            x.len(),
            cols
        );
        assert_eq!(
            out.len(),
            rows,
            "dot_acc_into: out.len() ({}) != rows ({})",
            out.len(),
            rows
        );

        // Row-major: stage_dequant[o * cols + i]. For each output row o,
        // accumulate Σ_i x[i] * dequant(o*cols + i).
        for (o, out_o) in out.iter_mut().enumerate().take(rows) {
            let row_offset = o * cols;
            let mut acc = 0.0_f32;
            for (i, &xi) in x.iter().enumerate().take(cols) {
                let idx = row_offset + i;
                let code = self.code_at(idx) as f32;
                let g = idx / self.group_size;
                let scale = self.scales[g].to_f32();
                let zp = self.zero_points[g].to_f32();
                acc += xi * (zp + code * scale);
            }
            *out_o += acc;
        }
    }

    /// Unpack the 2-bit packed codes into 1-per-byte raw indices.
    ///
    /// `out.len()` must be `>= n_elements`. Each output byte is a code in
    /// `0..LEVELS_PER_STAGE`. Used by [`RrqWeights::prefix_dot_lut_into`] to
    /// feed the multi-stage LUT kernel (Plan 568 Phase 3).
    ///
    /// **Zero allocation.** Writes into caller-provided `out`.
    pub fn codes_unpacked_into(&self, out: &mut [u8]) {
        assert!(
            out.len() >= self.n_elements,
            "codes_unpacked_into: out.len() ({}) < n_elements ({})",
            out.len(),
            self.n_elements
        );
        for (out_i, i) in out.iter_mut().take(self.n_elements).zip(0..self.n_elements) {
            *out_i = self.code_at(i);
        }
    }

    /// Build the 4-entry dequant LUT for group `g`.
    ///
    /// Entry `code` is `zero_point + code * scale` — the RRQ affine baked into
    /// LUT form. This differs from `QuantLut::build`'s `(code - zero) * scale`
    /// convention because RRQ stores `zp` as an additive offset, not a
    /// code-space zero. The two are related by `zero_lut = -zp / scale` but
    /// that form is undefined when `scale == 0`; this direct builder handles
    /// the degenerate case naturally (all entries = zp).
    #[inline]
    pub fn group_lut_at(&self, g: usize) -> [f32; LEVELS_PER_STAGE] {
        let scale = self.scales[g].to_f32();
        let zp = self.zero_points[g].to_f32();
        let mut lut = [0.0_f32; LEVELS_PER_STAGE];
        let mut code = 0;
        while code < LEVELS_PER_STAGE {
            lut[code] = zp + (code as f32) * scale;
            code += 1;
        }
        lut
    }
}

/// A complete RRQ weight matrix: base stage + N residual stages.
///
/// Default config: 1 base + 3 residuals → 2/4/6/8-bit prefixes
/// (`prefix_reconstruct_into(0)` = 2-bit, `(3)` = 8-bit).
pub struct RrqWeights {
    /// Base stage (2-bit RTN of the original weights).
    pub base: RrqStage,
    /// Residual stages (each 2-bit RTN of the previous residual).
    /// `residuals.len()` is typically 3 → 4 total prefixes (t=0..=3).
    pub residuals: Vec<RrqStage>,
    /// Matrix row count.
    pub rows: usize,
    /// Matrix column count.
    pub cols: usize,
}

impl RrqWeights {
    /// Construct an all-RTN RRQ package from f32 weights.
    ///
    /// `n_stages` = number of residual stages (default 3 → 2/4/6/8-bit
    /// prefixes). `group_size` default 128.
    ///
    /// **Algorithm** (paper Algorithm 1):
    /// 1. Quantize base: `codes_0 = rtn_quant(x, scales_0, zps_0)`;
    ///    dequant to `x̂_0`.
    /// 2. For k=1..=n_stages: `r^{k-1} = x − x̂^{k-1}`;
    ///    quantize `rtn_quant(r^{k-1}, scales_k, zps_k)`;
    ///    dequant to `r̂_k`; accumulate `x̂^k = x̂^{k-1} + r̂_k`.
    ///
    /// **Allocation:** runs once at model load (builds the owned stage
    /// Vecs). The hot-path methods are zero-allocation.
    pub fn from_weights_rtn(
        weights: &[f32],
        rows: usize,
        cols: usize,
        n_stages: usize,
        group_size: usize,
    ) -> Self {
        let n = rows * cols;
        assert_eq!(
            weights.len(),
            n,
            "from_weights_rtn: weights.len() ({}) != rows * cols ({})",
            weights.len(),
            n
        );
        assert!(group_size > 0, "from_weights_rtn: group_size must be > 0");

        // Step 1: base stage = RTN of the original weights.
        let base = RrqStage::quantize_rtn(weights, group_size);

        // Reusable buffers for the residual + reconstruction.
        let mut residual = vec![0.0_f32; n];
        let mut recon = vec![0.0_f32; n];
        base.dequant_into(&mut recon);
        for i in 0..n {
            residual[i] = weights[i] - recon[i];
        }

        // Step 2: iteratively quantize the residual.
        let mut residuals = Vec::with_capacity(n_stages);
        for _k in 0..n_stages {
            let stage = RrqStage::quantize_rtn(&residual, group_size);
            // Update reconstruction: x̂^k = x̂^{k-1} + r̂_k.
            let mut stage_dequant = vec![0.0_f32; n];
            stage.dequant_into(&mut stage_dequant);
            for i in 0..n {
                recon[i] += stage_dequant[i];
                residual[i] = weights[i] - recon[i];
            }
            residuals.push(stage);
        }

        Self {
            base,
            residuals,
            rows,
            cols,
        }
    }

    /// Reconstruct weights at prefix-t precision into `out`.
    ///
    /// `t=0` → base only (2-bit); `t=1` → base + 1 residual (4-bit);
    /// `t=2` → 6-bit; `t=3` → 8-bit (for the default 3-residual config).
    ///
    /// `t` is clamped to `residuals.len()` (so `t=usize::MAX` = all stages).
    ///
    /// `out.len()` must be `>= rows * cols`.
    ///
    /// **Allocation:** the reconstruction itself is zero-allocation (reads
    /// codes, writes `out`), but this method allocates a temporary buffer for
    /// per-stage dequant. For the zero-allocation hot path, use
    /// [`prefix_dot_into`](Self::prefix_dot_into) which never materializes
    /// the full reconstruction.
    pub fn prefix_reconstruct_into(&self, t: usize, out: &mut [f32]) {
        let n = self.rows * self.cols;
        assert!(
            out.len() >= n,
            "prefix_reconstruct_into: out.len() ({}) < n ({})",
            out.len(),
            n
        );
        // Base.
        self.base.dequant_into(out);
        // Residuals up to min(t, residuals.len()).
        let t_eff = t.min(self.residuals.len());
        if t_eff == 0 {
            return;
        }
        let mut stage_dequant = vec![0.0_f32; n];
        for k in 0..t_eff {
            self.residuals[k].dequant_into(&mut stage_dequant);
            for i in 0..n {
                out[i] += stage_dequant[i];
            }
        }
    }

    /// Compute `out = x · W̃(t)` at prefix-t precision, exploiting linearity.
    ///
    /// `out = x · Ŵ0 + Σ_{k=1..t} x · R̂k` — sum of per-stage GEMVs. Each
    /// stage GEMV dequants codes on the fly and accumulates into `out`.
    ///
    /// `x.len()` must be `cols`; `out.len()` must be `rows`. `scratch` is a
    /// reusable buffer of length `rows` (zeroed at the start of each call).
    ///
    /// **Allocation:** zero. `out` is zeroed by this call, accumulated into
    /// per-stage via [`RrqStage::dot_acc_into`], then read back. `scratch`
    /// is used as the per-stage accumulator (zeroed per stage).
    pub fn prefix_dot_into(&self, t: usize, x: &[f32], out: &mut [f32], scratch: &mut [f32]) {
        assert_eq!(
            x.len(),
            self.cols,
            "prefix_dot_into: x.len() ({}) != cols ({})",
            x.len(),
            self.cols
        );
        assert_eq!(
            out.len(),
            self.rows,
            "prefix_dot_into: out.len() ({}) != rows ({})",
            out.len(),
            self.rows
        );
        assert_eq!(
            scratch.len(),
            self.rows,
            "prefix_dot_into: scratch.len() ({}) != rows ({})",
            scratch.len(),
            self.rows
        );

        // Zero out.
        for (o, s) in out.iter_mut().zip(scratch.iter_mut()).take(self.rows) {
            *o = 0.0;
            *s = 0.0;
        }

        // Base stage.
        self.base.dot_acc_into(self.cols, x, out);

        // Residual stages up to min(t, residuals.len()).
        let t_eff = t.min(self.residuals.len());
        for k in 0..t_eff {
            // Zero scratch, accumulate this stage into scratch, add to out.
            for s in scratch.iter_mut().take(self.rows) {
                *s = 0.0;
            }
            self.residuals[k].dot_acc_into(self.cols, x, scratch);
            for (o, s) in scratch.iter().enumerate().take(self.rows) {
                out[o] += s;
            }
        }
    }

    /// LUT-accelerated prefix-t dot: `out = x · W̃(t)` via fused multi-stage
    /// SIMD gather, replacing the per-element arithmetic cast with LUT lookups
    /// (Plan 568 Phase 3 — the fusion target).
    ///
    /// For each element, the N stage LUT values are gathered and summed in
    /// registers, then a single FMA accumulates `stage_sum · x[i]` into the
    /// output. This avoids spilling dequantized values to memory AND avoids
    /// reloading `x` once per stage.
    ///
    /// # Requirements
    ///
    /// - `self.cols` must be divisible by `group_size`. (This is the common LLM
    ///   case. For other shapes, use [`prefix_dot_into`] — the arithmetic path.)
    /// - `codes_unpacked_per_stage` must contain `t_eff + 1` slices (base + up to
    ///   `t` residuals), each `>= rows * cols` bytes. The caller unpacks each
    ///   stage's 2-bit packed codes to 1-per-byte once at load via
    ///   [`RrqStage::codes_unpacked_into`], then reuses the buffers across all
    ///   calls. Max 8 stages (4 default + headroom).
    ///
    /// # Allocation
    ///
    /// Zero. Per-group LUTs are built on the stack as `[f32; LEVELS_PER_STAGE]`
    /// (4 entries, 16 bytes). The multi-stage kernel accumulator is register- or
    /// stack-resident. The caller-owned `out` + `codes_unpacked_per_stage` are
    /// reused across calls.
    ///
    /// # Feature gates
    ///
    /// Requires both `rrq_quant` AND `simd_lut_dequant` features.
    #[cfg(feature = "simd_lut_dequant")]
    pub fn prefix_dot_lut_into(
        &self,
        t: usize,
        x: &[f32],
        out: &mut [f32],
        codes_unpacked_per_stage: &[&[u8]],
    ) {
        use crate::simd_lut_dequant::dequant_dot_via_lut_multi_stage_slice;

        let gs = self.base.group_size;
        assert!(
            self.cols > 0 && self.cols.is_multiple_of(gs),
            "prefix_dot_lut_into: cols ({}) must be divisible by group_size ({}) \
             — use prefix_dot_into for the general case",
            self.cols,
            gs
        );
        assert_eq!(
            x.len(),
            self.cols,
            "prefix_dot_lut_into: x.len() ({}) != cols ({})",
            x.len(),
            self.cols
        );
        assert_eq!(
            out.len(),
            self.rows,
            "prefix_dot_lut_into: out.len() ({}) != rows ({})",
            out.len(),
            self.rows
        );

        let t_eff = t.min(self.residuals.len());
        let n_stages = t_eff + 1; // base + t_eff residuals
        assert_eq!(
            codes_unpacked_per_stage.len(),
            n_stages,
            "prefix_dot_lut_into: codes_unpacked_per_stage.len() ({}) != n_stages ({})",
            codes_unpacked_per_stage.len(),
            n_stages
        );
        assert!(
            n_stages <= 8,
            "prefix_dot_lut_into: max 8 stages, got {n_stages}"
        );

        // Stage references (base + active residuals).
        let mut stage_refs: [Option<&RrqStage>; 8] = [None; 8];
        stage_refs[0] = Some(&self.base);
        for k in 0..t_eff {
            stage_refs[k + 1] = Some(&self.residuals[k]);
        }

        // Zero out.
        for o in out.iter_mut().take(self.rows) {
            *o = 0.0;
        }

        let groups_per_row = self.cols / gs;

        for (o, out_o) in out.iter_mut().enumerate().take(self.rows) {
            for g_local in 0..groups_per_row {
                let g_global = o * groups_per_row + g_local;
                let col_start = g_local * gs;

                // Build per-stage LUTs for this group (stack [f32; LEVELS_PER_STAGE]).
                // Each LUT is the RRQ affine baked: lut[code] = zp + code * scale.
                // Build all LUTs first, then take slices (avoids borrow conflict).
                let mut luts_buf: [[f32; LEVELS_PER_STAGE]; 8] = [[0.0; LEVELS_PER_STAGE]; 8];
                for k in 0..n_stages {
                    luts_buf[k] = stage_refs[k].unwrap().group_lut_at(g_global);
                }
                let lut_slices: [&[f32]; 8] = [
                    &luts_buf[0][..],
                    &luts_buf[1][..],
                    &luts_buf[2][..],
                    &luts_buf[3][..],
                    &luts_buf[4][..],
                    &luts_buf[5][..],
                    &luts_buf[6][..],
                    &luts_buf[7][..],
                ];

                // Per-stage code slices for this group's segment.
                let flat_start = o * self.cols + col_start;
                let mut codes_slices: [&[u8]; 8] = [&[]; 8];
                for k in 0..n_stages {
                    codes_slices[k] = &codes_unpacked_per_stage[k][flat_start..flat_start + gs];
                }

                *out_o += dequant_dot_via_lut_multi_stage_slice(
                    &codes_slices[..n_stages],
                    &lut_slices[..n_stages],
                    &x[col_start..col_start + gs],
                );
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Load-time quant-strategy router (Phase 2 — paper §3 PMR + Research 200 KS)
// ──────────────────────────────────────────────────────────────────────────

/// PMR threshold for the default 2-bit base + 2-bit residual split (paper §3
/// Table 3, B=4, r=0.5: `K > 9r`). A group whose peak-to-mean ratio exceeds
/// this benefits from RRQ (the residual stage's finer step size wins once an
/// outlier forces the base step size wide). Below this, direct fixed-bit RTN
/// is at least as accurate at the same total bits and cheaper.
///
/// This is the value for the **default `BITS_PER_STAGE = 2`** config. Other
/// splits have different thresholds (1+3 → 3.29r, 3+1 → 29r); callers pass the
/// matching threshold explicitly to [`select_quant_strategy`].
pub const PMR_THRESHOLD_2_2: f32 = 9.0;

/// KS D-statistic above which a layer is flagged for security review
/// regardless of its PMR (Research 200: attacked layers show D > 0.25 vs
/// < 0.1 for clean layers). Computed by `katgpt_spectral::ks_d_statistic`
/// (Plan 224 OAQG substrate) — NOT recomputed here; the caller passes the
/// scalar. `katgpt-core` is the leaf and must not depend on `katgpt-spectral`,
/// so the router consumes the value via parameter (dependency inversion).
pub const KS_FLAG_THRESHOLD: f32 = 0.25;

/// Default target bit-width when the router recommends direct fixed-bit RTN
/// (the standard LLM serving point).
pub const DEFAULT_DIRECT_RTN_BITS: u8 = 4;

/// Peak-to-Mean Ratio — paper §3 outlier-severity metric.
///
/// For each group of `group_size` consecutive weights, computes
/// `max|x| / mean|x|` (the inlier radius `r` ≈ mean|x|, the outlier magnitude
/// `K` ≈ max|x|). Returns the **max across groups** — the worst-case group.
///
/// The paper's outlier threshold analysis (Research 467 §1.2, Table 3) shows
/// RRQ (2-bit base + 2-bit residual) beats direct 4-bit RTN once `K > 9r`,
/// i.e. once PMR exceeds [`PMR_THRESHOLD_2_2`]. Qwen3 profiles (PMR ≈ 28)
/// cross this; Llama profiles (mean PMR ≈ 5–7) do not.
///
/// **Degenerate groups** (all-zero): return `1.0` (no outlier structure; the
/// minimum since `max|x| ≥ mean|x|` always). A group with one spike among
/// zeros has PMR = `group_size` (finite).
///
/// **Allocation:** zero — single pass over `weights`, no scratch.
///
/// **Aggregation note:** the plan returns max-across-groups (the conservative
/// worst case — flag a layer if ANY group has severe outliers). The paper also
/// reports *mean*-across-groups PMR (more forgiving: a few outlier groups get
/// averaged out). Callers wanting the mean can compute it themselves; this
/// function deliberately exposes the conservative scalar for the security-adjacent
/// decision in [`select_quant_strategy`].
pub fn peak_to_mean_ratio(weights: &[f32], group_size: usize) -> f32 {
    assert!(group_size > 0, "peak_to_mean_ratio: group_size must be > 0");
    let n = weights.len();
    if n == 0 {
        return 1.0;
    }
    let n_groups = n.div_ceil(group_size);
    let mut worst = 1.0_f32;
    for g in 0..n_groups {
        let start = g * group_size;
        let end = (start + group_size).min(n);
        let mut max_abs = 0.0_f32;
        let mut sum_abs = 0.0_f32;
        for &v in &weights[start..end] {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
            sum_abs += a;
        }
        let count = (end - start) as f32;
        let mean_abs = sum_abs / count;
        // PMR = max|x| / mean|x|. Guard 0/0 (all-zero group): treat as flat.
        let pmr = if mean_abs > 0.0 {
            max_abs / mean_abs
        } else {
            1.0
        };
        if pmr > worst {
            worst = pmr;
        }
    }
    worst
}

/// Per-layer load-time quantization strategy recommendation.
///
/// Combines the paper's PMR outlier metric (§3 — does RRQ help at this layer?)
/// with the Research 200 KS D-statistic (is this layer's weight distribution
/// anomalous enough to suggest tampering?). The KS check is a security
/// override: a tampered layer goes to review regardless of how RRQ-friendly
/// its outlier profile looks.
///
/// See [`select_quant_strategy`] for the decision table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantStrategy {
    /// Outlier-heavy (PMR > threshold) AND not flagged (KS ≤ flag threshold).
    /// RRQ's residual stage pays off: its finer step size recovers the
    /// outlier-forced base error. `n_stages` defaults to [`DEFAULT_N_STAGES`]
    /// (3 → 2/4/6/8-bit prefixes); callers with a specific target width can
    /// adjust.
    Rrq {
        /// Number of residual stages recommended (default
        /// [`DEFAULT_N_STAGES`]).
        n_stages: usize,
    },
    /// Flat distribution (PMR ≤ threshold) AND not flagged. Direct fixed-bit
    /// RTN is at least as accurate as RRQ at the same total bits and cheaper
    /// (no per-stage metadata overhead). `bits` defaults to
    /// [`DEFAULT_DIRECT_RTN_BITS`] (4).
    DirectRtn {
        /// Target bit-width recommended (default [`DEFAULT_DIRECT_RTN_BITS`]).
        bits: u8,
    },
    /// KS D-statistic exceeds [`KS_FLAG_THRESHOLD`] — the weight distribution
    /// is anomalous enough to suggest outlier injection (Research 200).
    /// **Do not quantize until reviewed.** The PMR reading is still available
    /// via [`peak_to_mean_ratio`] for the reviewer; the router deliberately
    /// refuses to recommend a quantization path for a flagged layer.
    FlagForReview,
}

/// Load-time quant-strategy router. Decision table (Research 467 §3.2):
///
/// | KS D-stat | PMR | Strategy |
/// |---|---|---|
/// | > [`KS_FLAG_THRESHOLD`] (0.25) | any | [`QuantStrategy::FlagForReview`] (security override) |
/// | ≤ flag threshold | > `pmr_threshold` | [`QuantStrategy::Rrq`] (outlier-heavy → RRQ) |
/// | ≤ flag threshold | ≤ `pmr_threshold` | [`QuantStrategy::DirectRtn`] (flat → direct fixed-bit) |
///
/// `ks_d_stat` is the Kolmogorov-Smirnov D-statistic of this layer's weight
/// distribution against a Gaussian reference, computed by
/// `katgpt_spectral::ks_d_statistic` (Plan 224 OAQG substrate). It is passed
/// in as a scalar because `katgpt-core` (leaf) must not depend on
/// `katgpt-spectral` — the caller bridges the value.
///
/// `pmr_threshold` is the RRQ-beneficial threshold for the caller's chosen
/// base/residual bit split. Use [`PMR_THRESHOLD_2_2`] for the default
/// 2-bit base + 2-bit residual config (paper §3.4, `K > 9r`).
///
/// **Allocation:** zero.
pub fn select_quant_strategy(
    weights: &[f32],
    group_size: usize,
    ks_d_stat: f32,
    pmr_threshold: f32,
) -> QuantStrategy {
    // Security override first — a flagged layer never gets a quantization
    // recommendation regardless of how benign its outlier profile looks.
    if ks_d_stat > KS_FLAG_THRESHOLD {
        return QuantStrategy::FlagForReview;
    }
    let pmr = peak_to_mean_ratio(weights, group_size);
    if pmr > pmr_threshold {
        QuantStrategy::Rrq {
            n_stages: DEFAULT_N_STAGES,
        }
    } else {
        QuantStrategy::DirectRtn {
            bits: DEFAULT_DIRECT_RTN_BITS,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests (G1 correctness + G4 alloc-free regression)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Hand-rolled naive reference: quantize a group of f32 to 2-bit RTN,
    /// return (codes, scale, zero_point). Mirrors `RrqStage::quantize_rtn`
    /// exactly so the G1 test can compare bit-pattern-by-bit-pattern.
    fn ref_quantize_group(values: &[f32]) -> (Vec<u8>, f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in values {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let range = max - min;
        let scale = if range > 0.0 {
            range / (LEVELS_PER_STAGE as f32 - 1.0)
        } else {
            0.0
        };
        let zp = min;
        let scale_recip = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let codes: Vec<u8> = values
            .iter()
            .map(|&v| {
                let code_f = (v - zp) * scale_recip;
                if code_f <= 0.0 {
                    0
                } else if code_f >= (LEVELS_PER_STAGE - 1) as f32 {
                    (LEVELS_PER_STAGE - 1) as u8
                } else {
                    (code_f + 0.5) as u8
                }
            })
            .collect();
        (codes, scale, zp)
    }

    // ─── G1a: stage quantize/dequant round-trip matches reference ──────────

    #[test]
    fn g1_stage_quantize_matches_reference() {
        // 8 values, group_size=8 (one group).
        let values = [-1.0_f32, -0.5, -0.2, 0.0, 0.1, 0.3, 0.7, 1.0];
        let stage = RrqStage::quantize_rtn(&values, 8);

        let (ref_codes, ref_scale, ref_zp) = ref_quantize_group(&values);
        assert_eq!(stage.scales.len(), 1);
        assert_eq!(stage.zero_points.len(), 1);
        // f16 has ~3 decimal digits of precision; use 1e-2 relative tolerance
        // for the scale/zp comparison (the f32 → f16 → f32 round-trip introduces
        // ~1e-3 relative error).
        let scale_tol = (ref_scale.abs() + 1e-30) * 1e-2;
        let zp_tol = (ref_zp.abs() + 1e-30) * 1e-2;
        assert!(
            approx_eq(stage.scales[0].to_f32(), ref_scale, scale_tol),
            "scale mismatch: got {} expected {} (tol {})",
            stage.scales[0].to_f32(),
            ref_scale,
            scale_tol
        );
        assert!(
            approx_eq(stage.zero_points[0].to_f32(), ref_zp, zp_tol),
            "zp mismatch: got {} expected {} (tol {})",
            stage.zero_points[0].to_f32(),
            ref_zp,
            zp_tol
        );

        // Check each code matches.
        for (i, &ref_code) in ref_codes.iter().enumerate() {
            let got_code = stage.code_at(i);
            assert_eq!(
                got_code, ref_code,
                "code[{i}]: got {got_code} expected {ref_code}"
            );
        }
    }

    // ─── G1b: 2-bit packing round-trip ─────────────────────────────────────

    #[test]
    fn g1_code_packing_roundtrip() {
        // 16 values → 4 bytes of packed codes.
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.8).collect();
        let stage = RrqStage::quantize_rtn(&values, 16);

        // Every code should be in [0, 3].
        for i in 0..16 {
            let c = stage.code_at(i);
            assert!(c <= 3, "code[{i}] = {c} out of [0,3]");
        }

        // Dequant + check monotonic: values are increasing, so dequantized
        // values should be non-decreasing (RTN preserves order within a
        // group because the mapping code = round((x-zp)/scale) is monotone).
        let mut deq = vec![0.0_f32; 16];
        stage.dequant_into(&mut deq);
        for i in 1..16 {
            assert!(
                deq[i] >= deq[i - 1] || approx_eq(deq[i], deq[i - 1], 1e-7),
                "monotonicity broken at i={i}: deq[{i}]={} < deq[{}]={}",
                deq[i],
                i - 1,
                deq[i - 1]
            );
        }
    }

    // ─── G1c: prefix reconstruction matches reference sum ──────────────────

    #[test]
    fn g1_prefix_reconstruct_matches_reference() {
        // 4x8 matrix = 32 elements. group_size=8 → 4 groups.
        let rows = 4;
        let cols = 8;
        let n = rows * cols;
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as f32) * 0.07 - 1.1;
                (x * x * x - 0.5 * x).abs() * (if i % 3 == 0 { -1.0 } else { 1.0 })
            })
            .collect();

        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, DEFAULT_N_STAGES, 8);

        // Reference: for each prefix t, compute the sum of dequanted stages
        // directly (independent code path).
        let n_stages = DEFAULT_N_STAGES;
        let mut recon_ref = vec![0.0_f32; n];
        let mut stage_buf = vec![0.0_f32; n];

        // t=0: base only.
        rrq.base.dequant_into(&mut recon_ref);
        let mut recon_got = vec![0.0_f32; n];
        rrq.prefix_reconstruct_into(0, &mut recon_got);
        for i in 0..n {
            assert!(
                recon_got[i].to_bits() == recon_ref[i].to_bits(),
                "t=0 mismatch at i={i}: got {} expected {}",
                recon_got[i],
                recon_ref[i]
            );
        }

        // t=1..=n_stages: add each residual's dequant.
        for t in 1..=n_stages {
            rrq.residuals[t - 1].dequant_into(&mut stage_buf);
            for i in 0..n {
                recon_ref[i] += stage_buf[i];
            }
            rrq.prefix_reconstruct_into(t, &mut recon_got);
            for i in 0..n {
                assert!(
                    recon_got[i].to_bits() == recon_ref[i].to_bits(),
                    "t={t} mismatch at i={i}: got {} expected {}",
                    recon_got[i],
                    recon_ref[i]
                );
            }
        }
    }

    // ─── G1d: prefix dot matches reconstruct-then-dot ──────────────────────

    #[test]
    fn g1_dot_matches_reconstruct_then_dot() {
        let rows = 8;
        let cols = 16;
        let n = rows * cols;
        // Random-ish weights.
        let mut seed: u64 = 0xABCD_1234_5678_EF01;
        let weights: Vec<f32> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 40) as f32) * 2.0 - 1.0
            })
            .collect();

        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, DEFAULT_N_STAGES, 8);

        // Input vector.
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.13 - 1.0).collect();

        // For each prefix, compare prefix_dot_into vs reconstruct-then-dot.
        let mut recon = vec![0.0_f32; n];
        let mut out_dot = vec![0.0_f32; rows];
        let mut out_recon = vec![0.0_f32; rows];
        let mut scratch = vec![0.0_f32; rows];

        for t in 0..=DEFAULT_N_STAGES {
            // Path A: prefix_dot_into.
            rrq.prefix_dot_into(t, &x, &mut out_dot, &mut scratch);

            // Path B: reconstruct then matmul.
            rrq.prefix_reconstruct_into(t, &mut recon);
            for o in 0..rows {
                let mut acc = 0.0_f32;
                for i in 0..cols {
                    acc += x[i] * recon[o * cols + i];
                }
                out_recon[o] = acc;
            }

            // Compare (bit-identical — same arithmetic, different order; allow
            // 1 ULP slack for FMA vs separate mul+add).
            for o in 0..rows {
                let diff = (out_dot[o] - out_recon[o]).abs();
                let ulp = (out_recon[o].abs() + 1e-30) * f32::EPSILON;
                assert!(
                    diff <= ulp.max(1e-6),
                    "t={t} o={o}: dot={} recon={} diff={}",
                    out_dot[o],
                    out_recon[o],
                    diff
                );
            }
        }
    }

    // ─── G1e: more stages → lower reconstruction error ─────────────────────

    #[test]
    fn g1_more_stages_lower_error() {
        // With more residual stages, the prefix-t reconstruction should be
        // closer to the original (the residual shrinks each iteration).
        let rows = 4;
        let cols = 32;
        let n = rows * cols;
        let weights: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).sin() * 1.5).collect();

        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, DEFAULT_N_STAGES, 16);

        let mut recon = vec![0.0_f32; n];
        let mut prev_err = f32::INFINITY;
        for t in 0..=DEFAULT_N_STAGES {
            rrq.prefix_reconstruct_into(t, &mut recon);
            let mut err = 0.0_f32;
            for i in 0..n {
                err += (recon[i] - weights[i]).abs();
            }
            assert!(
                err <= prev_err + 1e-5,
                "t={t}: error {err} should be <= previous {prev_err} (monotone improvement)"
            );
            prev_err = err;
        }

        // Sanity: t=3 (8-bit) should be meaningfully better than t=0 (2-bit).
        let mut err0 = 0.0_f32;
        let mut err3 = 0.0_f32;
        rrq.prefix_reconstruct_into(0, &mut recon);
        for i in 0..n {
            err0 += (recon[i] - weights[i]).abs();
        }
        rrq.prefix_reconstruct_into(DEFAULT_N_STAGES, &mut recon);
        for i in 0..n {
            err3 += (recon[i] - weights[i]).abs();
        }
        assert!(
            err3 < err0 * 0.5,
            "8-bit error {err3} should be < 50% of 2-bit error {err0}"
        );
    }

    // ─── G1f: constant weights → zero residual → exact reconstruction ─────

    #[test]
    fn g1_constant_weights_exact() {
        // If every weight is the same constant, the base stage captures it
        // exactly (one group, range=0, scale=0, every code=0, dequant=min=
        // the constant). Residuals should all be zero.
        let rows = 2;
        let cols = 4;
        let weights = vec![0.42_f32; rows * cols];

        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, 2, 4);

        let mut recon = vec![0.0_f32; rows * cols];
        rrq.prefix_reconstruct_into(2, &mut recon);
        for (i, &r) in recon.iter().enumerate() {
            assert!(
                approx_eq(r, 0.42, 1e-7),
                "constant reconstruction at i={i}: got {r} expected 0.42"
            );
        }
    }

    // ─── G4: alloc-free hot path (prefix_dot_into) ──────────────────────────

    /// The hot-path methods (`dequant_into`, `dot_acc_into`,
    /// `prefix_dot_into`) must be zero-allocation after construction.
    /// We verify by checking that `prefix_dot_into` does not grow any Vec
    /// (no `push`, no `resize`, no `Vec::new`). This is a static check via
    /// code review + a smoke test that runs many calls.
    #[test]
    fn g4_prefix_dot_smoke_zero_vec_growth() {
        let rows = 8;
        let cols = 16;
        let n = rows * cols;
        let weights: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 1.2).collect();
        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, 2, 8);

        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1 - 0.8).collect();
        let mut out = vec![0.0_f32; rows];
        let mut scratch = vec![0.0_f32; rows];

        // Run 100 calls — if any allocated, the test would still pass (we
        // can't easily assert zero allocs in a lib test without the
        // CountingAllocator global), but this catches panics + verifies the
        // API is callable in a tight loop without surprise state.
        for _ in 0..100 {
            rrq.prefix_dot_into(2, &x, &mut out, &mut scratch);
        }
        // Sanity: out is not all zeros (the weights + x are nontrivial).
        let sum: f32 = out.iter().copied().sum();
        assert!(sum.abs() > 0.0, "out sum should be nonzero, got {sum}");
    }

    // ─── G1: peak_to_mean_ratio on known distributions ─────────────────────

    #[test]
    fn g1_pmr_uniform_distribution_is_one() {
        // All |x| equal → max|x| == mean|x| → PMR == 1.
        let weights = [0.5_f32; 128];
        let pmr = peak_to_mean_ratio(&weights, 128);
        assert!(
            approx_eq(pmr, 1.0, 1e-5),
            "uniform PMR should be 1.0, got {pmr}"
        );
    }

    #[test]
    fn g1_pmr_single_spike_equals_group_size() {
        // One spike of magnitude `spike` among `n-1` zeros:
        //   mean|x| = spike/n, max|x| = spike → PMR = n.
        let n = 64_usize;
        let mut weights = vec![0.0_f32; n];
        weights[0] = 10.0;
        let pmr = peak_to_mean_ratio(&weights, n);
        assert!(
            approx_eq(pmr, n as f32, 1e-3),
            "single-spike PMR should be n={n}, got {pmr}"
        );
    }

    #[test]
    fn g1_pmr_all_zero_group_is_flat() {
        // All-zero group → degenerate (0/0) → treat as flat (PMR = 1).
        let weights = [0.0_f32; 32];
        let pmr = peak_to_mean_ratio(&weights, 32);
        assert_eq!(pmr, 1.0, "all-zero PMR should be 1.0, got {pmr}");
    }

    #[test]
    fn g1_pmr_takes_max_across_groups() {
        // Two groups of 4: group 0 flat (PMR 1), group 1 has a spike (PMR 4).
        // Worst-case (max across groups) should report group 1's PMR.
        let weights = [
            // group 0: uniform
            1.0_f32, 1.0, 1.0, 1.0, // group 1: one spike among 3 zeros → PMR = 4
            5.0, 0.0, 0.0, 0.0,
        ];
        let pmr = peak_to_mean_ratio(&weights, 4);
        assert!(
            approx_eq(pmr, 4.0, 1e-5),
            "max-across-groups PMR should be 4.0 (the spike group), got {pmr}"
        );
    }

    // ─── G1: select_quant_strategy classifies Llama vs Qwen profiles ────────

    /// Build a synthetic weight block of `n_groups` groups × `group_size`
    /// weights. If `outlier_factor` > 0, inject one outlier of magnitude
    /// `outlier_factor * inlier` into each group; otherwise draw flat inliers.
    fn synthetic_weights(
        n_groups: usize,
        group_size: usize,
        inlier: f32,
        outlier_factor: f32,
    ) -> Vec<f32> {
        let n = n_groups * group_size;
        let mut w = vec![0.0_f32; n];
        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
        for g in 0..n_groups {
            for j in 0..group_size {
                // Small deterministic jitter around `inlier` (pseudo-noise so
                // mean|x| ≈ inlier, not exactly — realistic).
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let jitter = ((seed >> 40) as f32 / (1u64 << 40) as f32) * 0.2 - 0.1;
                let sign = if (seed >> 20) & 1 == 0 { -1.0 } else { 1.0 };
                w[g * group_size + j] = sign * (inlier + jitter);
            }
            if outlier_factor > 0.0 {
                // Inject one outlier at the group's first slot.
                w[g * group_size] = inlier * outlier_factor;
            }
        }
        w
    }

    #[test]
    fn g1_pmr_classifies_llama_vs_qwen() {
        // Llama-like: mild outliers (PMR ~5, below the 2+2 threshold of 9)
        // → DirectRtn. Qwen-like: severe outliers (PMR ~30, well above 9)
        // → Rrq. The synthetic distributions are tuned to our PMR scale; the
        // paper's Table 9 Llama/Qwen K/MAE numbers (26.5 / 116) are a
        // different normalization and serve as context, not exact targets.
        let group_size = 128;

        // Llama-like: flat inliers, tiny outlier factor 5.
        let llama = synthetic_weights(8, group_size, 0.5, 5.0);
        let llama_pmr = peak_to_mean_ratio(&llama, group_size);
        let llama_strat = select_quant_strategy(&llama, group_size, 0.05, PMR_THRESHOLD_2_2);
        // Llama PMR should be ~5 (one 2.5 outlier among ~0.5 inliers).
        assert!(
            llama_pmr < PMR_THRESHOLD_2_2,
            "Llama-like PMR {llama_pmr:.2} should be < threshold {PMR_THRESHOLD_2_2}, got DirectRtn expected"
        );
        assert_eq!(
            llama_strat,
            QuantStrategy::DirectRtn {
                bits: DEFAULT_DIRECT_RTN_BITS,
            },
            "Llama-like (PMR {llama_pmr:.2}) should select DirectRtn"
        );

        // Qwen-like: severe outliers (factor 30 → PMR ~30).
        let qwen = synthetic_weights(8, group_size, 0.5, 30.0);
        let qwen_pmr = peak_to_mean_ratio(&qwen, group_size);
        let qwen_strat = select_quant_strategy(&qwen, group_size, 0.05, PMR_THRESHOLD_2_2);
        assert!(
            qwen_pmr > PMR_THRESHOLD_2_2,
            "Qwen-like PMR {qwen_pmr:.2} should be > threshold {PMR_THRESHOLD_2_2}, got Rrq expected"
        );
        assert_eq!(
            qwen_strat,
            QuantStrategy::Rrq {
                n_stages: DEFAULT_N_STAGES,
            },
            "Qwen-like (PMR {qwen_pmr:.2}) should select Rrq"
        );
    }

    #[test]
    fn g1_ks_overrides_pmr() {
        // KS D-statistic > flag threshold → FlagForReview regardless of PMR.
        // Use a Qwen-like profile (would otherwise select Rrq) but a high KS.
        let group_size = 128;
        let qwen = synthetic_weights(8, group_size, 0.5, 30.0);
        let pmr = peak_to_mean_ratio(&qwen, group_size);
        assert!(
            pmr > PMR_THRESHOLD_2_2,
            "precondition: Qwen-like PMR {pmr:.2}"
        );

        // High KS (tampered) → FlagForReview, even though PMR says Rrq.
        let strat = select_quant_strategy(&qwen, group_size, 0.40, PMR_THRESHOLD_2_2);
        assert_eq!(strat, QuantStrategy::FlagForReview);

        // Sanity: same weights, low KS → Rrq (the override is KS-gated, not
        // PMR-gated).
        let strat_clean = select_quant_strategy(&qwen, group_size, 0.05, PMR_THRESHOLD_2_2);
        assert_eq!(
            strat_clean,
            QuantStrategy::Rrq {
                n_stages: DEFAULT_N_STAGES,
            }
        );
    }

    #[test]
    fn g1_router_boundary_ks_exactly_at_threshold_is_not_flagged() {
        // KS == KS_FLAG_THRESHOLD exactly → NOT flagged (strict > ). The
        // boundary goes to the PMR decision.
        let flat = [0.5_f32; 128];
        let strat = select_quant_strategy(&flat, 128, KS_FLAG_THRESHOLD, PMR_THRESHOLD_2_2);
        assert_eq!(
            strat,
            QuantStrategy::DirectRtn {
                bits: DEFAULT_DIRECT_RTN_BITS,
            }
        );
    }

    // ─── Phase 3: fused multi-stage LUT dot (Plan 568 Phase 3) ──────────────

    /// G1: `prefix_dot_lut_into` matches `prefix_dot_into` (arithmetic path)
    /// within FP tolerance. The two paths use different reduction orders
    /// (LUT path sums stages per-element before the FMA; arithmetic path
    /// sums per-stage GEMV results), so they won't be bit-identical — but both
    /// approximate the same true mathematical result.
    #[cfg(feature = "simd_lut_dequant")]
    #[test]
    fn g1_prefix_dot_lut_matches_arithmetic() {
        // 4 groups per row (cols=128, gs=32 → 4 groups/row).
        let rows = 16;
        let cols = 128;
        let gs = 32;
        let n = rows * cols;
        let weights: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 2.0 - 1.0).collect();
        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, DEFAULT_N_STAGES, gs);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.05).cos()).collect();

        // Arithmetic path.
        let mut out_arith = vec![0.0_f32; rows];
        let mut scratch = vec![0.0_f32; rows];
        rrq.prefix_dot_into(rrq.residuals.len(), &x, &mut out_arith, &mut scratch);

        // LUT path: unpack codes for each stage.
        let t = rrq.residuals.len();
        let n_stages = t + 1;
        let mut codes_unpacked: Vec<Vec<u8>> = (0..n_stages).map(|_| vec![0u8; n]).collect();
        // Base.
        rrq.base.codes_unpacked_into(&mut codes_unpacked[0]);
        for k in 0..t {
            rrq.residuals[k].codes_unpacked_into(&mut codes_unpacked[k + 1]);
        }
        // Collect refs AFTER all mutation is done (avoids borrow conflict).
        let codes_refs: Vec<&[u8]> = codes_unpacked.iter().map(|v| v.as_slice()).collect();
        let mut out_lut = vec![0.0_f32; rows];
        rrq.prefix_dot_lut_into(t, &x, &mut out_lut, &codes_refs);

        // Compare: relative tolerance 1e-3 (different reduction orders).
        for o in 0..rows {
            let rel = (out_lut[o] - out_arith[o]).abs() / out_arith[o].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "row {}: lut={} arith={} rel={}",
                o,
                out_lut[o],
                out_arith[o],
                rel
            );
        }
    }

    /// G1: `group_lut_at` produces the correct RRQ affine.
    #[test]
    fn g1_group_lut_matches_rrq_affine() {
        let values = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let stage = RrqStage::quantize_rtn(&values, 8);
        let lut = stage.group_lut_at(0);
        // Each code's dequant via group_lut_at must match dequant_at.
        for i in 0..8 {
            let direct = stage.dequant_at(i);
            let via_lut = lut[stage.code_at(i) as usize];
            assert!(
                (direct - via_lut).abs() < 1e-6,
                "i={i}: direct={direct} lut={via_lut}"
            );
        }
    }

    /// G1: `codes_unpacked_into` round-trips with `code_at`.
    #[test]
    fn g1_codes_unpacked_roundtrip() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let stage = RrqStage::quantize_rtn(&values, 16);
        let mut unpacked = vec![0u8; 64];
        stage.codes_unpacked_into(&mut unpacked);
        for (i, &code) in unpacked.iter().enumerate().take(64) {
            assert_eq!(code, stage.code_at(i), "i={i}");
        }
    }

    /// G2 (release-only): 4-stage 2-bit LUT path vs single 8-bit LUT path at
    /// the 8-bit prefix. **Plan 568 Phase 3 G2 — HONEST NEGATIVE RESULT.**
    ///
    /// The hypothesis (amortized gather from tiny 4-entry LUTs staying hot in
    /// L1 while the 1KB 8-bit LUT spills) FAILS empirically: the 4-stage LUT
    /// path runs ~4–6× SLOWER than single-8-bit at the 256×256/gs=128 LLM-tile
    /// scale. Root cause: the 4× code-read + 4× gather overhead dominates the
    /// L1-residency benefit. At gs=128, a 256×256 matrix needs only 2 LUTs per
    /// row (2KB total) — well within L1, so there is no spilling to amortize.
    ///
    /// The negative is structural, not parametric: the only regime where the
    /// hypothesis could hold is when the 8-bit LUTs spill L1 (tens of thousands
    /// of groups → >32KB of LUT data), which requires matrices far larger than
    /// any realistic single-layer tile. At that scale the per-element gather
    /// latency from cold LUTs would hurt BOTH paths equally.
    ///
    /// **Consequence:** the fused multi-stage kernel is correct substrate (G1
    /// PASS) but not a perf win. [`RrqWeights::prefix_dot_into`] (the arithmetic
    /// path from Phase 1) remains the recommended hot path. The LUT path
    /// ([`RrqWeights::prefix_dot_lut_into`]) stays available as opt-in substrate
    /// for consumers whose LUT construction is already amortized (e.g. a future
    /// hardware StreamDQ analog where the gather is near-memory).
    ///
    /// This test RUNS and reports the ratio via `eprintln!` but does NOT panic
    /// on failure — the negative is the documented result, not a regression.
    #[cfg(feature = "simd_lut_dequant")]
    #[cfg_attr(debug_assertions, ignore)]
    #[test]
    fn g2_4stage_lut_vs_single_8bit_documented_negative() {
        use crate::simd_lut_dequant::{Int8Lut, QuantLut, dequant_dot_via_lut};

        // Realistic LLM-tile shape: 256 rows × 256 cols, group_size 128.
        let rows = 256;
        let cols = 256;
        let gs = 128;
        let n = rows * cols;
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                // Outlier-heavy distribution (RRQ's target regime).
                let base = (i as f32 * 0.01).sin();
                if i % 50 == 0 { base * 8.0 } else { base }
            })
            .collect();

        // RRQ path: 4 stages × 2-bit (= 8-bit effective prefix).
        let rrq = RrqWeights::from_weights_rtn(&weights, rows, cols, DEFAULT_N_STAGES, gs);
        let t = rrq.residuals.len(); // 3 → 4 stages total
        let n_stages = t + 1;
        let mut codes_unpacked: Vec<Vec<u8>> = (0..n_stages).map(|_| vec![0u8; n]).collect();
        rrq.base.codes_unpacked_into(&mut codes_unpacked[0]);
        for k in 0..t {
            rrq.residuals[k].codes_unpacked_into(&mut codes_unpacked[k + 1]);
        }
        let codes_refs: Vec<&[u8]> = codes_unpacked.iter().map(|v| v.as_slice()).collect();

        // Single 8-bit path: quantize to 8-bit, one stage, one 256-entry LUT.
        // We simulate this by treating the RRQ 8-bit prefix reconstruction as
        // the 8-bit reference, then quantizing THAT to 8-bit RTN per group.
        let mut recon_8bit = vec![0.0_f32; n];
        rrq.prefix_reconstruct_into(t, &mut recon_8bit);
        // 8-bit RTN: per-group min/max → 256 levels.
        let n_groups_8 = n.div_ceil(gs);
        let mut codes_8bit = vec![0u8; n];
        let mut luts_8bit: Vec<Int8Lut> = Vec::with_capacity(n_groups_8);
        for g in 0..n_groups_8 {
            let start = g * gs;
            let end = (start + gs).min(n);
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &slot in &recon_8bit[start..end] {
                min = min.min(slot);
                max = max.max(slot);
            }
            let range = max - min;
            let scale = if range > 0.0 { range / 255.0 } else { 0.0 };
            let zero_lut = if scale > 0.0 { -min / scale } else { 0.0 };
            luts_8bit.push(Int8Lut::build(scale, zero_lut));
            let recip = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for j in start..end {
                let code_f = (recon_8bit[j] - min) * recip;
                let code = if code_f <= 0.0 {
                    0u8
                } else if code_f >= 255.0 {
                    255u8
                } else {
                    (code_f + 0.5) as u8
                };
                codes_8bit[j] = code;
            }
        }

        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.03).cos()).collect();
        let iterations = 200;

        // ── Benchmark RRQ 4-stage LUT path ──
        let mut out_rrq = vec![0.0_f32; rows];
        // Warmup.
        for _ in 0..10 {
            rrq.prefix_dot_lut_into(t, &x, &mut out_rrq, &codes_refs);
        }
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            rrq.prefix_dot_lut_into(t, &x, &mut out_rrq, &codes_refs);
        }
        let rrq_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

        // ── Benchmark single 8-bit LUT path ──
        // We can't use RrqWeights for this — it's a different quantization.
        // Instead, benchmark a comparable per-group LUT dot manually.
        let mut out_8bit = vec![0.0_f32; rows];
        let bench_8bit = |out: &mut [f32]| {
            for (o, out_o) in out.iter_mut().enumerate().take(rows) {
                let mut acc = 0.0_f32;
                for g in 0..(cols / gs) {
                    let g_global = o * (cols / gs) + g;
                    let col_start = g * gs;
                    let flat_start = o * cols + col_start;
                    acc += dequant_dot_via_lut(
                        &codes_8bit[flat_start..flat_start + gs],
                        &luts_8bit[g_global],
                        &x[col_start..col_start + gs],
                        0,
                        0xFF,
                    );
                }
                *out_o = acc;
            }
        };
        // Warmup.
        for _ in 0..10 {
            bench_8bit(&mut out_8bit);
        }
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            bench_8bit(&mut out_8bit);
        }
        let single_8bit_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

        let ratio = rrq_ns / single_8bit_ns;
        eprintln!(
            "g2_4stage_lut_vs_single_8bit: rrq_4stage={rrq_ns:.1}ns single_8bit={single_8bit_ns:.1}ns ratio={ratio:.3}x \
             (HONEST NEGATIVE — documented, gate was ≤ 1.05x)"
        );
        // Document the negative: the 4-stage LUT path is slower (ratio > 1.05).
        // This is the expected possible outcome per Plan 568 risk table.
        // The test passes (does not panic) so the ratio is visible in CI logs;
        // the arithmetic `prefix_dot_into` remains the recommended path.
        assert!(
            ratio > 1.0,
            "expected the 4-stage LUT path to be slower (documented negative); \
             ratio={ratio:.3} is unexpectedly faster — re-evaluate the G2 verdict"
        );
    }
}
