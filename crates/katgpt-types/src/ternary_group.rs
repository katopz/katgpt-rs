//! Bit-plane packed ternary weights with group-wise f16 scale (Issue 578).
//!
//! The `Q2_0_g128` container. Ternary `{-1, 0, +1}` like [`crate::TernaryWeights`],
//! but the scale is per **128-weight group** (as in [`crate::BinaryWeights`])
//! instead of per row.
//!
//! ## Why a third tier
//!
//! | Tier | Alphabet | Scale | Holds `Q2_0_g128`? |
//! |---|---|---|---|
//! | `TernaryWeights` | `{-1, 0, +1}` | per row, f32 | no — wrong scale granularity |
//! | `BinaryWeights` | `{-1, +1}` | per 128, f16 | no — no zero state |
//! | **`TernaryGroupWeights`** | `{-1, 0, +1}` | per 128, f16 | **yes** |
//!
//! `BinaryWeights::from_ternary_no_zeros` cannot bridge the gap: it returns
//! `None` the moment any weight is zero, and ternary sparsity is the whole
//! point of the format.
//!
//! ## Footprint
//!
//! Two bit-planes = 2 bits/weight, plus 16 bits per 128 weights = **2.125
//! bits/weight**. At 27B params that is 7.17 GB — matching Ternary-Bonsai-27B's
//! ~7.2 GB deployed size, so loading a `Q2_0_g128` tensor into this container is
//! a *repack*, not an expansion.
//!
//! ## Group/block alignment
//!
//! [`GROUP_SIZE`] is 128 and blocks are 64 bits, so **every group spans exactly
//! two whole `u64` blocks**. Kernels rely on this: a group boundary is never in
//! the middle of a block, so the scale can be switched between blocks without
//! splitting a word.

use crate::GROUP_SIZE;
use half::f16;

/// u64 blocks per group — `GROUP_SIZE / 64` = 2 at the shipped group size.
const BLOCKS_PER_GROUP: usize = GROUP_SIZE / 64;

/// Ternary `{-1, 0, +1}` bit-plane weights with a per-128-weight f16 scale.
///
/// 64 weights per block stored as two `u64` bitmasks:
/// - `pos_bits[block]` bit k set → `weight[row][k] = +1`
/// - `neg_bits[block]` bit k set → `weight[row][k] = -1`
/// - both clear → weight = 0 (implicit, no storage)
///
/// **Representation invariant:** `pos_bits[i] & neg_bits[i] == 0` for every `i`.
/// A weight cannot be both `+1` and `-1`. See [`Self::invariant_holds`].
///
/// `group_scale[r * groups_per_row + g]` rescales group `g` of row `r`.
#[cfg(feature = "ternary_group_scale")]
#[derive(Clone, Debug)]
pub struct TernaryGroupWeights {
    pub rows: usize,
    pub cols: usize,
    pub blocks64: usize,       // cols.div_ceil(64)
    pub groups_per_row: usize, // cols.div_ceil(GROUP_SIZE)
    pub pos_bits: Vec<u64>,    // [rows * blocks64]
    pub neg_bits: Vec<u64>,    // [rows * blocks64]
    pub group_scale: Vec<f16>, // [rows * groups_per_row]
}

/// Hook for GPU-accelerated ternary matvec (Issue 599 unblock path).
///
/// When provided to `forward_qwen_deltanet_ternary_with_hook`, replaces
/// `simd_ternary_group_matvec_parallel` for each projection. The GPU
/// implementation (riir-gpu `GemvTernaryCubeCL`) uploads weights once at
/// construction and dispatches CubeCL kernels per matvec call.
///
/// The contract is identical to `simd_ternary_group_matvec_parallel`:
/// `y = w @ x` where `x.len() == w.cols` and `y.len() == w.rows`.
///
/// Thread-safety: implementations must be `Send + Sync` because the forward
/// may be called from any thread. The GPU implementation holds pre-uploaded
/// handles keyed by the `TernaryGroupWeights` pointer identity (the weights
/// themselves are immutable after load).
#[cfg(feature = "ternary_group_scale")]
pub trait TernaryMatvecHook: Send + Sync {
    /// `y = w @ x` — same contract as `simd_ternary_group_matvec_parallel`.
    ///
    /// Implementations SHOULD pre-upload weights at construction time and
    /// cache the GPU handles by `(pos_bits.as_ptr(), neg_bits.as_ptr())`.
    /// The first call for a new weight matrix uploads; subsequent calls hit
    /// the cache.
    fn matvec(&self, w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]);
}

/// Hook for fused FFN dispatch (Issue 601).
///
/// When present, the forward path calls `ffn()` for the SwiGLU MLP portion
/// instead of 3 separate `matvec` calls. This enables the GPU implementation
/// to chain gate+up GEMVs + SwiGLU + down GEMV in a single command buffer,
/// eliminating per-matvec sync readback (2.574× speedup, Issue 600).
///
/// `out` MAY alias `x` (the forward path saves the residual before calling).
#[cfg(feature = "ternary_group_scale")]
pub trait TernaryFfnHook: Send + Sync {
    /// Fused FFN: `out = down @ swiglu(gate @ x, up @ x)`.
    ///
    /// - `gate_w`: [mlp_dim × n_embd]
    /// - `up_w`: [mlp_dim × n_embd]
    /// - `down_w`: [n_embd × mlp_dim]
    /// - `x`: [n_embd] — RMSNorm'd input
    /// - `out`: [n_embd] — FFN output (pre-residual-add)
    fn ffn(
        &self,
        gate_w: &TernaryGroupWeights,
        up_w: &TernaryGroupWeights,
        down_w: &TernaryGroupWeights,
        x: &[f32],
        out: &mut [f32],
    );
}

/// Hook for fused DeltaNet input projections (Issue 602).
///
/// The DeltaNet layer's first step is 4 ternary matvecs that ALL share the
/// same input `x`:
///
/// ```text
/// qkv = in_proj_qkv @ x   (output: q_dim + k_dim + v_dim)
/// z   = in_proj_z   @ x   (output: z_dim = v_dim)
/// a   = in_proj_a   @ x   (output: n_v_heads)
/// b   = in_proj_b   @ x   (output: n_v_heads)
/// ```
///
/// When this hook is present, the forward path calls `input_projections()`
/// once instead of 4 separate `matvec` calls. The GPU implementation chains
/// all 4 GEMVs in a single command buffer (all reading the same uploaded
/// `x`, writing to 4 separate output buffers) and does ONE `read_one` at
/// the end — eliminating 3 of 4 GPU sync points per layer.
///
/// At ~2.5 ms/sync × 3 × 64 layers, this saves ~480 ms/token on M3 Max Metal.
///
/// Output slices MUST NOT alias each other or `x`. The forward path ensures
/// this by writing into pre-allocated scratch buffers.
#[cfg(feature = "ternary_group_scale")]
#[allow(clippy::too_many_arguments, reason = "Fused GPU dispatch interface: 4 weight matrices + 1 input + 4 outputs is inherent to the DeltaNet input projection block")]
pub trait TernaryInputProjHook: Send + Sync {
    /// Fused DeltaNet input projections — all 4 sharing input `x`.
    ///
    /// - `qkv_w`: [(q_dim + k_dim + v_dim) × n_embd]
    /// - `z_w`: [z_dim × n_embd]
    /// - `a_w`: [n_v_heads × n_embd]
    /// - `b_w`: [n_v_heads × n_embd]
    /// - `x`: [n_embd] — RMSNorm'd layer input
    /// - `qkv_out`: [(q_dim + k_dim + v_dim)] — QKV projection output
    /// - `z_out`: [z_dim] — output gate projection
    /// - `a_out`: [n_v_heads] — decay gate projection (raw, pre-softplus)
    /// - `b_out`: [n_v_heads] — update rate projection (raw, pre-sigmoid)
    fn input_projections(
        &self,
        qkv_w: &TernaryGroupWeights,
        z_w: &TernaryGroupWeights,
        a_w: &TernaryGroupWeights,
        b_w: &TernaryGroupWeights,
        x: &[f32],
        qkv_out: &mut [f32],
        z_out: &mut [f32],
        a_out: &mut [f32],
        b_out: &mut [f32],
    );
}



#[cfg(feature = "ternary_group_scale")]
impl TernaryGroupWeights {
    /// Create all-zero weights with unit group scale.
    pub fn new(rows: usize, cols: usize) -> Self {
        let blocks64 = cols.div_ceil(64);
        let groups_per_row = cols.div_ceil(GROUP_SIZE);
        Self {
            rows,
            cols,
            blocks64,
            groups_per_row,
            pos_bits: vec![0u64; rows * blocks64],
            neg_bits: vec![0u64; rows * blocks64],
            group_scale: vec![f16::ONE; rows * groups_per_row],
        }
    }

    /// Set the ternary value at `(row, col)`. Panics if out of bounds or if
    /// `value` is not in `{-1, 0, +1}`.
    pub fn set(&mut self, row: usize, col: usize, value: i8) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        assert!(
            (-1..=1).contains(&value),
            "ternary value must be -1, 0, or +1"
        );
        let idx = row * self.blocks64 + (col >> 6);
        let mask = 1u64 << (col & 63);
        // Clear both planes first, then set at most one — keeps the
        // pos & neg == 0 invariant by construction.
        self.pos_bits[idx] &= !mask;
        self.neg_bits[idx] &= !mask;
        match value {
            1 => self.pos_bits[idx] |= mask,
            -1 => self.neg_bits[idx] |= mask,
            _ => {}
        }
    }

    /// Get the ternary value at `(row, col)`.
    pub fn get(&self, row: usize, col: usize) -> i8 {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let idx = row * self.blocks64 + (col >> 6);
        let mask = 1u64 << (col & 63);
        let pos = (self.pos_bits[idx] & mask) != 0;
        let neg = (self.neg_bits[idx] & mask) != 0;
        pos as i8 - neg as i8
    }

    /// Scale applied to group `g` of row `r`.
    #[inline]
    pub fn scale_at(&self, row: usize, group: usize) -> f32 {
        self.group_scale[row * self.groups_per_row + group].to_f32()
    }

    /// Set the scale for group `g` of row `r`.
    #[inline]
    pub fn set_scale(&mut self, row: usize, group: usize, scale: f32) {
        self.group_scale[row * self.groups_per_row + group] = f16::from_f32(scale);
    }

    /// Check the representation invariant: no weight is both `+1` and `-1`.
    ///
    /// G1 gate helper — a loader that mis-parses `Q2_0_g128` will typically
    /// violate this, so it is a cheap corruption check after loading.
    pub fn invariant_holds(&self) -> bool {
        self.pos_bits
            .iter()
            .zip(&self.neg_bits)
            .all(|(p, n)| p & n == 0)
    }

    /// Widen row-scale ternary weights into the group-scale tier.
    ///
    /// Lossless in the weights (bit-planes are copied verbatim) and lossless in
    /// the scale up to f32→f16 rounding: each row's single `row_scale` is
    /// broadcast across that row's groups.
    ///
    /// This is the direction [`crate::BinaryWeights::from_ternary_no_zeros`]
    /// cannot go — it drops the zero state and returns `None`. Widening always
    /// succeeds, so it returns `Self`, not `Option<Self>`.
    #[cfg(feature = "plasma_path")]
    pub fn from_ternary(tw: &crate::TernaryWeights) -> Self {
        let mut out = Self::new(tw.rows, tw.cols);
        out.pos_bits.copy_from_slice(&tw.pos_bits);
        out.neg_bits.copy_from_slice(&tw.neg_bits);
        for r in 0..tw.rows {
            let scale = f16::from_f32(tw.row_scale[r]);
            let base = r * out.groups_per_row;
            out.group_scale[base..base + out.groups_per_row].fill(scale);
        }
        out
    }

    /// Quantize f32 weights to ternary with **group-wise** error compensation.
    ///
    /// Mirrors [`crate::TernaryWeights::quantize_from_f32`], but the scale and
    /// the error carry reset per 128-weight group rather than per row — finer
    /// error compensation, which is the reason `Q2_0_g128` exists.
    ///
    /// For each group:
    /// ```text
    /// scale     = mean(|w|) over the group
    /// threshold = 0.5 * scale
    /// adjusted  = value + carry
    ///   adjusted >  threshold -> +1
    ///   adjusted < -threshold -> -1
    ///   otherwise             ->  0
    /// carry     = adjusted - q * scale
    /// ```
    pub fn quantize_from_f32(weights: &[f32], rows: usize, cols: usize) -> Self {
        Self::quantize_with_scale_rule(weights, rows, cols, false)
    }

    /// Quantize with **power-of-two** group scales (Issue 845; arXiv:2609.00363).
    ///
    /// The identical carry-compensation pipeline to [`Self::quantize_from_f32`]
    /// with ONE insertion: the group's mean-abs scale is snapped to the nearest
    /// power of two BEFORE the f16 store and the threshold/carry loop. This is
    /// requantize-from-parents in the strongest sense — the payload is
    /// re-derived from the f32 parents under the snapped scale; stored payloads
    /// are never reinterpreted (the paper's +157% artifact was exactly that
    /// reinterpretation).
    ///
    /// Why: multiplication or division by a power of two is an exact exponent
    /// shift, so every scale-application site (CPU reference and every GPU
    /// kernel) contributes ZERO rounding under this arm — cross-backend
    /// bit-identity at the scale sites becomes REQUIRED rather than measured.
    /// PoT values are additionally exact in f16 (the snap clamps the exponent
    /// into [-14, 15]), so the native arm's scale-storage rounding disappears.
    ///
    /// Cost class: the snapped scale sits within ×√2 of the ideal mean-abs and
    /// the carry loop absorbs most of the difference — measured in
    /// `pot_tests::snr_native_vs_pot`, GPU-side budget tracked in riir-ai
    /// `.issues/845`.
    #[cfg(feature = "pot_scales")]
    pub fn quantize_from_f32_pot(weights: &[f32], rows: usize, cols: usize) -> Self {
        Self::quantize_with_scale_rule(weights, rows, cols, true)
    }

    /// Shared quantization core. `snap_scales_to_pow2 = false` reproduces the
    /// legacy pipeline bit-for-bit: the snap insertion sits between the mean-abs
    /// computation and the f16 store and is skipped entirely in the native arm.
    fn quantize_with_scale_rule(
        weights: &[f32],
        rows: usize,
        cols: usize,
        snap_scales_to_pow2: bool,
    ) -> Self {
        Self::quantize_with_group_scale(weights, rows, cols, |_g_start, group| {
            let mut scale = mean_abs_scale(group);
            if snap_scales_to_pow2 {
                scale = snap_f32_to_pow2(scale);
            }
            scale
        })
    }

    /// The one quantization pipeline every scale rule shares (Issue 886 P1
    /// refactor — the baseline, `pot_scales` and `act_aware_fit` arms differ
    /// ONLY in the closure). `group_scale(g_start, group)` returns the
    /// pre-f16 scale for the group starting at column `g_start`; the f16
    /// store, the threshold `0.5·scale` and the carry loop are identical for
    /// every rule, so a rule that returns the mean-abs value produces the
    /// legacy payload bit-for-bit (the G3 anchor).
    pub(crate) fn quantize_with_group_scale<F>(
        weights: &[f32],
        rows: usize,
        cols: usize,
        mut group_scale: F,
    ) -> Self
    where
        F: FnMut(usize, &[f32]) -> f32,
    {
        assert_eq!(weights.len(), rows * cols, "weights slice must be rows*cols");
        let mut out = Self::new(rows, cols);

        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let row_base = r * out.blocks64;
            let group_base = r * out.groups_per_row;

            for g in 0..out.groups_per_row {
                let g_start = g * GROUP_SIZE;
                let g_end = (g_start + GROUP_SIZE).min(cols);
                let group = &row[g_start..g_end];

                let scale = group_scale(g_start, group);
                out.group_scale[group_base + g] = f16::from_f32(scale);
                // Quantize against the f16-rounded scale the kernel will
                // actually apply, not the f32 ideal — otherwise the carry
                // compensates for an error the forward pass never makes.
                let scale = out.group_scale[group_base + g].to_f32();

                let threshold = 0.5 * scale;
                let mut carry = 0.0f32;
                for (i, &val) in group.iter().enumerate() {
                    let adjusted = val + carry;
                    let q = match adjusted {
                        a if a > threshold => 1i8,
                        a if a < -threshold => -1i8,
                        _ => 0i8,
                    };
                    let col = g_start + i;
                    let idx = row_base + (col >> 6);
                    let mask = 1u64 << (col & 63);
                    // Branch-free plane writes; at most one bit is set, so the
                    // pos & neg == 0 invariant holds by construction.
                    out.pos_bits[idx] |= ((q == 1) as u64) * mask;
                    out.neg_bits[idx] |= ((q == -1) as u64) * mask;
                    carry = adjusted - (q as f32 * scale);
                }
            }
        }

        out
    }

    /// Checksum over all values: `Σ_r Σ_g scale[r,g] * (popcount(pos) - popcount(neg))`
    /// within that group. Used for cross-implementation verification.
    pub fn checksum(&self) -> f32 {
        let mut total = 0.0f32;
        for r in 0..self.rows {
            let row_base = r * self.blocks64;
            let group_base = r * self.groups_per_row;
            for g in 0..self.groups_per_row {
                // GROUP_SIZE == 2 * 64, so a group is exactly blocks [2g, 2g+2)
                // — clamped for a trailing partial group.
                let b_start = g * (GROUP_SIZE / 64);
                let b_end = (b_start + GROUP_SIZE / 64).min(self.blocks64);
                let mut sum: i32 = 0;
                for b in b_start..b_end {
                    let idx = row_base + b;
                    sum += self.pos_bits[idx].count_ones() as i32;
                    sum -= self.neg_bits[idx].count_ones() as i32;
                }
                total += self.group_scale[group_base + g].to_f32() * sum as f32;
            }
        }
        total
    }

    /// True iff every stored group scale is exactly a positive normal power of
    /// two (the `pot_scales` construction invariant; 1.0 — the all-zero-group
    /// convention — counts). Assert this before claiming required-equality at
    /// the scale-application sites (Issue 845 G1).
    #[cfg(feature = "pot_scales")]
    pub fn all_scales_are_pow2(&self) -> bool {
        self.group_scale.iter().all(|&s| {
            let b = s.to_f32().to_bits();
            // Positive sign, zero mantissa, non-zero exponent → exactly 2^k.
            // The snap's clamp confines k to [-14, 15] (f16-normal range).
            b & 0x807F_FFFF == 0 && ((b >> 23) & 0xFF) != 0
        })
    }

    /// Bytes of weight payload (bit-planes + scales), excluding `Vec` overhead.
    ///
    /// `2 bits/weight + 16 bits per GROUP_SIZE weights` = 2.125 bits/weight at
    /// `GROUP_SIZE = 128`.
    pub fn encoded_bytes(&self) -> usize {
        self.pos_bits.len() * 8 + self.neg_bits.len() * 8 + self.group_scale.len() * 2
    }

    /// Convert to block-contiguous (AoS) layout (Issue 650).
    ///
    /// Produces a `Vec<TernaryBlockAoS>` where each 128-weight group stores
    /// its scale + pos_bits + neg_bits contiguously. This eliminates the 3×
    /// global-memory-access overhead of the SoA layout in GPU kernels.
    ///
    /// The conversion is a one-time cost at model load (GPU upload). The
    /// per-inference benefit is the co-located memory access pattern.
    ///
    /// Footprint is identical: 34 bytes per 128 weights in both layouts.
    pub fn to_block_contiguous(&self) -> Vec<TernaryBlockAoS> {
        let total_groups = self.rows * self.groups_per_row;
        let mut blocks = Vec::with_capacity(total_groups);
        for r in 0..self.rows {
            let row_block_base = r * self.blocks64;
            let group_base = r * self.groups_per_row;
            for g in 0..self.groups_per_row {
                let b_start = g * BLOCKS_PER_GROUP;
                let mut blk = TernaryBlockAoS {
                    scale: self.group_scale[group_base + g].to_bits(),
                    pos_bits: [0u8; BLOCKS_PER_GROUP * 8],
                    neg_bits: [0u8; BLOCKS_PER_GROUP * 8],
                };
                for i in 0..BLOCKS_PER_GROUP {
                    let b = b_start + i;
                    if b < self.blocks64 {
                        blk.set_pos_word(i, self.pos_bits[row_block_base + b]);
                        blk.set_neg_word(i, self.neg_bits[row_block_base + b]);
                    }
                }
                blocks.push(blk);
            }
        }
        blocks.shrink_to_fit();
        blocks
    }
}

/// The activation-blind baseline group scale: `mean(|group|)`, with the
/// all-zero group guarded to 1.0 so the threshold stays finite and
/// quantization yields all zeros. The exact expression the legacy pipeline
/// used (same iterator sum, same divisor) — the Issue 886 G3 anchor.
#[cfg(feature = "ternary_group_scale")]
#[inline]
pub(crate) fn mean_abs_scale(group: &[f32]) -> f32 {
    let abs_sum: f32 = group.iter().map(|v| v.abs()).sum();
    if abs_sum > 0.0 { abs_sum / group.len() as f32 } else { 1.0 }
}

/// Snap a positive f32 to the nearest power of two, clamped into the
/// f16-normal PoT range `[2^-14, 2^15]` (the quantized scale is stored f16 —
/// Issue 845).
///
/// Exact by construction: `floor(log2 x)` is read from the exponent field (no
/// libm call), `x / 2^e` is an exact PoT division landing in `[1, 2)` for
/// in-range inputs, and the nearest-in-log-space choice is the sign of
/// `ratio² - 2` (the √2 boundary itself sits within 1 ULP of that comparison —
/// either neighbor is a legal "nearest"). Inputs outside the clamp range snap
/// to the range edge, so the result is always exactly representable in f16.
///
/// Non-finite / non-positive input returns 1.0: the quantizer only ever passes
/// a positive mean-abs (all-zero groups are guarded to 1.0 upstream), and
/// 1.0 = 2^0 is the PoT identity.
#[cfg(feature = "ternary_group_scale")]
fn snap_f32_to_pow2(x: f32) -> f32 {
    if !x.is_finite() || x <= 0.0 {
        return 1.0;
    }
    let e = (((x.to_bits() >> 23) & 0xFF) as i32 - 127).clamp(-14, 15);
    let unit = f32::from_bits(((e + 127) as u32) << 23); // 2^e, exact
    let ratio = x / unit; // exact PoT ÷ PoT
    if e < 15 && ratio * ratio >= 2.0 {
        unit + unit // 2^(e+1), exact
    } else {
        unit
    }
}

/// Block-contiguous (AoS) ternary weight block (Issue 650).
///
/// 128 ternary weights packed with their group scale into a single
/// `#[repr(C)]` struct. This layout co-locates the scale + pos plane + neg
/// plane for one group, so GPU kernels can load all three in a single
/// global-memory access instead of three separate fetches.
///
/// ## Footprint
///
/// 34 bytes per 128 weights — identical to [`crate::TernaryGroupWeights`]
/// (SoA) and to `BlockQ2_0` in riir-engine's quant layer. Uses `u16` scale +
/// `[u8; 16]` bit-arrays (not `u64`) to keep alignment at 2 and avoid padding.
///
/// ## Why block-contiguous
///
/// The SoA layout ([`crate::TernaryGroupWeights`]) stores pos_bits, neg_bits,
/// and group_scale in separate arrays. In a GPU kernel, each K-tile iteration
/// loads from 3 different global-memory addresses → 3 cache-line fetches.
/// This layout co-locates them → 1 fetch. For large weight matrices
/// (e.g. ffn_down at N=17408), this is a 3× reduction in global-memory
/// accesses, which is the structural explanation for the 1.89× vs llama.cpp's
/// 9.4× speedup gap (Bench 645, riir-ai).
#[cfg(feature = "ternary_group_scale")]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TernaryBlockAoS {
    /// Per-128-weight group scale (f16 bits).
    pub scale: u16,
    /// 128 pos bits = 16 bytes (2 × u64, stored as raw bytes).
    pub pos_bits: [u8; BLOCKS_PER_GROUP * 8],
    /// 128 neg bits = 16 bytes (2 × u64, stored as raw bytes).
    pub neg_bits: [u8; BLOCKS_PER_GROUP * 8],
}

impl TernaryBlockAoS {
    /// Get the group scale as f32.
    #[inline]
    pub fn scale_f32(&self) -> f32 {
        f16::from_bits(self.scale).to_f32()
    }

    /// Read u64 word `word` (0 or 1) from the pos plane.
    #[inline]
    pub fn pos_word(&self, word: usize) -> u64 {
        let start = word * 8;
        u64::from_le_bytes(
            self.pos_bits[start..start + 8]
                .try_into()
                .expect("word index in range"),
        )
    }

    /// Read u64 word `word` (0 or 1) from the neg plane.
    #[inline]
    pub fn neg_word(&self, word: usize) -> u64 {
        let start = word * 8;
        u64::from_le_bytes(
            self.neg_bits[start..start + 8]
                .try_into()
                .expect("word index in range"),
        )
    }

    /// Write u64 word `word` (0 or 1) to the pos plane.
    #[inline]
    pub fn set_pos_word(&mut self, word: usize, val: u64) {
        let start = word * 8;
        self.pos_bits[start..start + 8].copy_from_slice(&val.to_le_bytes());
    }

    /// Write u64 word `word` (0 or 1) to the neg plane.
    #[inline]
    pub fn set_neg_word(&mut self, word: usize, val: u64) {
        let start = word * 8;
        self.neg_bits[start..start + 8].copy_from_slice(&val.to_le_bytes());
    }
}

/// Block-contiguous ternary weight matrix (Issue 650).
///
/// The AoS counterpart to [`crate::TernaryGroupWeights`]. Each group's
/// scale + pos + neg are contiguous in memory. Construct via
/// [`crate::TernaryGroupWeights::to_block_contiguous`].
#[cfg(feature = "ternary_group_scale")]
#[derive(Clone, Debug)]
pub struct TernaryBlockContiguousWeights {
    pub rows: usize,
    pub cols: usize,
    pub groups_per_row: usize,
    pub blocks: Vec<TernaryBlockAoS>, // [rows * groups_per_row]
}

#[cfg(feature = "ternary_group_scale")]
impl TernaryBlockContiguousWeights {
    /// Construct from a `Vec<TernaryBlockAoS>` + shape metadata.
    pub fn from_blocks(
        blocks: Vec<TernaryBlockAoS>,
        rows: usize,
        cols: usize,
    ) -> Self {
        let groups_per_row = cols.div_ceil(GROUP_SIZE);
        assert_eq!(
            blocks.len(),
            rows * groups_per_row,
            "block count must match rows × groups_per_row"
        );
        Self {
            rows,
            cols,
            groups_per_row,
            blocks,
        }
    }

    /// `y = w @ x` — block-contiguous matvec (Issue 650).
    ///
    /// Computes the same result as `simd_ternary_group_matvec` but reads
    /// from the AoS layout. This is the CPU reference implementation; the
    /// GPU kernel will mirror this access pattern.
    ///
    /// `x.len() == cols`, `y.len() == rows`.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols, "x length must match cols");
        assert_eq!(y.len(), self.rows, "y length must match rows");
        for (r, y_slot) in y.iter_mut().enumerate() {
            let mut row_sum = 0.0f32;
            for g in 0..self.groups_per_row {
                let blk = &self.blocks[r * self.groups_per_row + g];
                let g_start = g * GROUP_SIZE;
                let g_end = (g_start + GROUP_SIZE).min(self.cols);
                let mut group_acc = 0.0f32;
                for (local, &x_val) in x[g_start..g_end].iter().enumerate() {
                    let word = local >> 6;
                    let mask = 1u64 << (local & 63);
                    let pos = (blk.pos_word(word) & mask) != 0;
                    let neg = (blk.neg_word(word) & mask) != 0;
                    let sign = pos as i32 - neg as i32;
                    group_acc += sign as f32 * x_val;
                }
                row_sum += blk.scale_f32() * group_acc;
            }
            *y_slot = row_sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random f32 in [-1, 1). No rand dep, reproducible.
    fn pseudo(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn filled(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
        let mut s = seed;
        let mut w = TernaryGroupWeights::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let v = pseudo(&mut s);
                let q = match v {
                    v if v > 0.33 => 1i8,
                    v if v < -0.33 => -1i8,
                    _ => 0i8,
                };
                w.set(r, c, q);
            }
            for g in 0..w.groups_per_row {
                w.set_scale(r, g, 0.5 + 0.25 * (g % 4) as f32);
            }
        }
        w
    }

    /// G1: `to_block_contiguous` + `matvec` must produce bit-identical results
    /// to the SoA reference matvec. This is the correctness gate for Issue 650
    /// Phase 1 — if this fails, the block-contiguous layout has a repack or
    /// indexing bug.
    #[test]
    fn block_contiguous_matvec_matches_soa() {
        // Test multiple shapes including Bonsai-relevant dimensions.
        let shapes: &[(usize, usize)] = &[
            (1, 128),      // single group
            (3, 256),      // multi-group
            (48, 512),     // ssm_alpha/beta shape
            (1024, 512),   // attn_k/v shape
            (128, 4096),   // small Bonsai projection
            (512, 17408),  // ffn_gate shape (large, multi-group)
        ];

        for &(rows, cols) in shapes {
            let w = filled(rows, cols, 42 + rows as u64);
            let w_bc = TernaryBlockContiguousWeights::from_blocks(
                w.to_block_contiguous(),
                rows,
                cols,
            );

            // Random input vector.
            let mut seed = 100 + rows as u64;
            let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut seed)).collect();

            let mut y_soa = vec![0.0f32; rows];
            let mut y_aos = vec![0.0f32; rows];

            // SoA reference: the scalar matvec from the simd module.
            crate::simd::ternary_group::ternary_group_matvec_scalar(&w, &x, &mut y_soa);
            // AoS: the new block-contiguous matvec.
            w_bc.matvec(&x, &mut y_aos);

            // Bit-identical (not just approximate) — same weights, same math,
            // just different memory layout.
            assert_eq!(
                y_soa, y_aos,
                "SoA vs AoS matvec mismatch at shape ({rows}×{cols})"
            );
        }
    }

    /// Block size sanity: `TernaryBlockAoS` must be 34 bytes.
    #[test]
    fn block_size_is_34_bytes() {
        assert_eq!(
            std::mem::size_of::<TernaryBlockAoS>(),
            34,
            "TernaryBlockAoS must be 34 bytes (2 scale + 16 pos + 16 neg)"
        );
    }

    /// Footprint parity: block-contiguous total bytes == SoA total bytes.
    #[test]
    fn footprint_matches_soa() {
        let w = filled(64, 512, 7);
        let w_bc = w.to_block_contiguous();
        let soa_bytes = w.encoded_bytes();
        let aos_bytes = w_bc.len() * std::mem::size_of::<TernaryBlockAoS>();
        assert_eq!(soa_bytes, aos_bytes, "SoA and AoS footprints must match");
    }

    /// Invariant: pos & neg == 0 preserved through the conversion.
    #[test]
    fn invariant_holds_after_conversion() {
        let w = filled(32, 256, 99);
        assert!(w.invariant_holds(), "source must hold invariant");
        let bc = w.to_block_contiguous();
        for blk in &bc {
            for i in 0..BLOCKS_PER_GROUP {
                assert_eq!(
                    blk.pos_bits[i] & blk.neg_bits[i],
                    0,
                    "pos & neg != 0 after conversion"
                );
            }
        }
    }
}

#[cfg(all(test, feature = "pot_scales"))]
mod pot_tests {
    use super::*;

    /// Deterministic pseudo-random f32 in [-1, 1) (same LCG as `mod tests`).
    fn pseudo(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn pow2i(k: i32) -> f32 {
        f32::from_bits(((k + 127) as u32) << 23)
    }

    /// `snap` keeps exact PoT inputs and picks the log-nearest neighbor.
    #[test]
    fn snap_nearest_pow2() {
        for k in -14..=15 {
            assert_eq!(super::snap_f32_to_pow2(pow2i(k)), pow2i(k), "2^{k}");
        }
        // 3.0: log2 = 1.585 — log-nearer to 4 than to 2.
        assert_eq!(super::snap_f32_to_pow2(3.0), 4.0);
        // 2.8: log2 = 1.485 — log-nearer to 2.
        assert_eq!(super::snap_f32_to_pow2(2.8), 2.0);
        // 0.6: log2 ≈ -0.737 — log-nearer to 0.5.
        assert_eq!(super::snap_f32_to_pow2(0.6), 0.5);
        // 0.55: log2 ≈ -0.862 — log-nearer to 0.5? |−0.862−(−1)| = 0.138 vs
        // |0 − (−0.862)| = 0.862 → 0.5. And 0.75: log2 ≈ −0.415 → 1.0.
        assert_eq!(super::snap_f32_to_pow2(0.75), 1.0);
        // The √2 boundary is a legal-either-way ULP zone — both neighbors are
        // "nearest"; assert membership, not direction.
        let sqrt2 = 2.0f32.sqrt();
        let s = super::snap_f32_to_pow2(sqrt2);
        assert!(s == 1.0 || s == 2.0, "√2 boundary snapped to {s}");
    }

    /// Out-of-range inputs clamp to the f16-normal PoT range; degenerate
    /// inputs fall back to the PoT identity 1.0.
    #[test]
    fn snap_clamps_and_degenerate() {
        assert_eq!(super::snap_f32_to_pow2(1e-40), pow2i(-14));
        assert_eq!(super::snap_f32_to_pow2(1e-20), pow2i(-14));
        assert_eq!(super::snap_f32_to_pow2(1e30), pow2i(15));
        // Non-finite is the defensive fallback (1.0 = PoT identity), matching
        // the doc contract — only finite positive mean-abs values reach snap.
        assert_eq!(super::snap_f32_to_pow2(f32::INFINITY), 1.0);
        assert_eq!(super::snap_f32_to_pow2(0.0), 1.0);
        assert_eq!(super::snap_f32_to_pow2(-2.0), 1.0);
        assert_eq!(super::snap_f32_to_pow2(f32::NAN), 1.0);
    }

    /// PoT is exact in f16 across the entire clamp range — the construction's
    /// "scale storage adds no rounding" claim.
    #[test]
    fn f16_pot_roundtrip_exact() {
        for k in -14..=15 {
            let v = pow2i(k);
            assert_eq!(f16::from_f32(v).to_f32(), v, "2^{k} must be f16-exact");
        }
    }

    /// The PoT arm's output satisfies the construction invariant on every
    /// scale regime.
    #[test]
    fn pot_quantize_all_scales_pow2() {
        for (scale_up, seed) in [(pow2i(-8), 0xA1), (1.0, 0xB2), (pow2i(10), 0xC3)] {
            let mut s = seed;
            let src: Vec<f32> = (0..2 * 512).map(|_| pseudo(&mut s) * scale_up).collect();
            let w = TernaryGroupWeights::quantize_from_f32_pot(&src, 2, 512);
            assert!(w.invariant_holds());
            assert!(
                w.all_scales_are_pow2(),
                "scale_up={scale_up}: all group scales must be PoT"
            );
        }
    }

    /// Scale covariance: quantizing `w·2^k` yields BIT-IDENTICAL payloads and
    /// exactly-×2^k scales. Every intermediate scales by exact PoT factors
    /// (summands, the ÷len mean, the snap, threshold, carry), and IEEE correct
    /// rounding commutes with exact PoT scaling — so the native arm's f16
    /// scale rounding is the ONLY thing that breaks this property, and the PoT
    /// arm restores it. Practical consequence: pre-scaling a weight tensor at
    /// export no longer perturbs the quantized payload.
    #[test]
    fn pot_is_scale_covariant_for_pow2_rescaling() {
        // cols=300 exercises a partial trailing group (44 weights, len not a
        // PoT — the ÷len mean still scales exactly).
        let (rows, cols) = (3usize, 300usize);
        let mut s = 0x845_u64;
        let base: Vec<f32> = (0..rows * cols).map(|_| pseudo(&mut s)).collect();
        let q1 = TernaryGroupWeights::quantize_from_f32_pot(&base, rows, cols);

        for k in -6i32..=6 {
            let two_k = pow2i(k);
            let scaled: Vec<f32> = base.iter().map(|&v| v * two_k).collect();
            let qk = TernaryGroupWeights::quantize_from_f32_pot(&scaled, rows, cols);
            assert_eq!(qk.pos_bits, q1.pos_bits, "k={k}: payload must not move");
            assert_eq!(qk.neg_bits, q1.neg_bits, "k={k}: payload must not move");
            for r in 0..rows {
                for g in 0..q1.groups_per_row {
                    let expected = q1.scale_at(r, g) * two_k;
                    assert_eq!(
                        qk.scale_at(r, g),
                        expected,
                        "k={k} r={r} g={g}: scale must shift by exactly 2^{k}"
                    );
                }
            }
        }
    }

    /// All-zero group keeps the 1.0 convention, which is itself PoT.
    #[test]
    fn zero_group_stays_pot() {
        let mut src = vec![0.0f32; 128];
        let mut s = 0x77_u64;
        src.extend((0..128).map(|_| pseudo(&mut s)));
        let w = TernaryGroupWeights::quantize_from_f32_pot(&src, 1, 256);
        assert_eq!(w.scale_at(0, 0), 1.0, "all-zero group keeps the 1.0 guard");
        assert!(w.all_scales_are_pow2());
        for c in 0..128 {
            assert_eq!(w.get(0, c), 0, "all-zero group quantizes to all zeros");
        }
    }

    /// Sanity: the native arm's scales are genuinely NOT all PoT — the two
    /// arms are different constructions, not a rename.
    #[test]
    fn native_arm_scales_are_not_all_pot() {
        let mut s = 0x999_u64;
        let src: Vec<f32> = (0..4 * 512).map(|_| pseudo(&mut s)).collect();
        let native = TernaryGroupWeights::quantize_from_f32(&src, 4, 512);
        assert!(
            !native.all_scales_are_pow2(),
            "random mean-abs scales are never all PoT — arms must differ"
        );
    }

    /// Relative Frobenius error of the dequantized weights vs the f32 parents.
    /// f64 accumulation keeps the measurement stable.
    fn rel_frobenius_err(src: &[f32], q: &TernaryGroupWeights, rows: usize, cols: usize) -> f32 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for r in 0..rows {
            for c in 0..cols {
                let w = src[r * cols + c] as f64;
                let d = (q.get(r, c) as f32 * q.scale_at(r, c / GROUP_SIZE)) as f64;
                let e = w - d;
                num += e * e;
                den += w * w;
            }
        }
        ((num.sqrt()) / (den.sqrt())) as f32
    }

    /// Weight-space SNR cost of the PoT arm vs the native arm across
    /// deterministic distributions — the CPU half of Issue 845's G2 (the real
    /// budget is ppl on the GPU surfaces, measured in the GPU window).
    ///
    /// Naive worst case: the snap moves the scale by ≤ √2, so a compensation-
    /// free error model bounds the ratio by √2 ≈ 1.414. The carry-compensation
    /// loop absorbs most of it; the pinned bound below (1.35) sits between the
    /// naive worst case and the measured envelope.
    #[test]
    fn snr_native_vs_pot() {
        let (rows, cols) = (4usize, 512usize);

        let mut cases: Vec<(&str, Vec<f32>)> = Vec::new();

        // Uniform [-1, 1).
        let mut s = 0x1111_u64;
        cases.push(("uniform", (0..rows * cols).map(|_| pseudo(&mut s)).collect()));

        // Gaussian-ish (sum of three uniforms → bell-shaped).
        let mut s = 0x2222_u64;
        cases.push((
            "gaussish",
            (0..rows * cols)
                .map(|_| (pseudo(&mut s) + pseudo(&mut s) + pseudo(&mut s)) / 3.0)
                .collect(),
        ));

        // Sparse: ~70% exact zeros, exercising magnitude-varied groups.
        // (pseudo is negative-only: [-1, 0).)
        let mut s = 0x3333_u64;
        cases.push((
            "sparse70",
            (0..rows * cols)
                .map(|_| if pseudo(&mut s) > -0.7 { 0.0 } else { pseudo(&mut s) })
                .collect(),
        ));

        // Row-scaled: each row carries a different PoT magnitude (mixed
        // per-row regimes in one tensor).
        let mut s = 0x4444_u64;
        cases.push((
            "row-scaled",
            (0..rows * cols)
                .map(|i| {
                    let r = i / cols;
                    pseudo(&mut s) * pow2i(r as i32 - 2)
                })
                .collect(),
        ));

        println!("| distribution | native rel-err | PoT rel-err | ratio |");
        println!("|---|---|---|---|");
        for (name, src) in &cases {
            let native = TernaryGroupWeights::quantize_from_f32(src, rows, cols);
            let pot = TernaryGroupWeights::quantize_from_f32_pot(src, rows, cols);
            let e_n = rel_frobenius_err(src, &native, rows, cols);
            let e_p = rel_frobenius_err(src, &pot, rows, cols);
            println!("| {name} | {e_n:.6} | {e_p:.6} | {:.4} |", e_p / e_n);
            assert!(
                e_p <= e_n * 1.35 + 1e-7,
                "{name}: PoT error {e_p} exceeds 1.35x native {e_n} — the carry \
                 loop should absorb the ≤√2 snap factor"
            );
        }
    }
}
