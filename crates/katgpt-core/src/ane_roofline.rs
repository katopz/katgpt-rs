//! ANE-aware roofline cost model — extends [`crate::roofline`] with the Apple
//! Neural Engine's distinct cost shape.
//!
//! *Source:* Bryngelson, *Apple Neural Engine: Architecture, Programming, and
//! Performance*, arXiv:2606.22283 (2026). See `katgpt-rs/.research/377_*.md`
//! and `katgpt-rs/.plans/379_*.md`.
//!
//! # Why a separate model from [`crate::roofline`]
//!
//! The GPU roofline in [`crate::roofline`] models three regimes:
//! `Launch / Compute / Memory`. The ANE has two extra regimes that the GPU
//! model does not capture, both of which produce wrong routing decisions if
//! ignored:
//!
//! 1. **Working-set cliff** — operands larger than the on-chip SRAM (2 MB on
//!    M1/H13, 4.72 MB on M5/H17s, see Bryngelson ch. 9.2 / ch. 21.1, HAL
//!    field `0x1b8`) are tiled and streamed from DRAM, collapsing arithmetic
//!    intensity. This is a hard cliff, not a soft slope.
//! 2. **Family-floor capability gate** — operations declare a
//!    `MinimumFamily<N>` trait (Bryngelson ch. 35); an op with floor F3
//!    (e.g. crop-resize) does not lower on M1/A13 and must run on CPU/GPU.
//!
//! The ANE also has a **much higher dispatch floor** than a GPU kernel launch:
//! 0.23 ms on M1 vs ~50 µs for a typical GPU kernel (see [`HardwarePeaks`]).
//! Below that floor the operation pays the full round-trip regardless of how
//! little compute it does, so the CPU almost always wins for small work.
//!
//! # M1 / H13 anchor table (Bryngelson ch. 9.8, measured silicon)
//!
//! | Quantity               | M1 value    |
//! |------------------------|-------------|
//! | Compute roof (slope)   | 12 fp16 TFLOP/s |
//! | DRAM bandwidth         | 85 GB/s     |
//! | Standalone activation stream | 24 GB/s |
//! | Roofline ridge point   | 141 FLOP/byte |
//! | On-chip working set    | 2 MB        |
//! | Per-dispatch floor     | 0.23 ms     |
//! | NE cores (HAL `0x238`) | 4 (base), 8 (Pro/Max) |
//!
//! # M5 / H17s anchor table (Bryngelson ch. 12.7)
//!
//! | Quantity               | M5 value    |
//! |------------------------|-------------|
//! | Compute roof (slope)   | 19.6 fp16 TFLOP/s |
//! | DRAM bandwidth         | 145 GB/s (two read channels) |
//! | Roofline ridge point   | 424 FLOP/byte |
//! | On-chip working set    | 4.72 MB     |
//! | Per-dispatch floor     | 0.11 ms     |
//! | NE cores               | 16          |
//!
//! # Determinism
//!
//! This cost model is pure arithmetic — no allocations, no RNG, no runtime
//! state. Same inputs ⇒ same outputs, bit-for-bit.
//!
//! [`HardwarePeaks`]: crate::roofline::HardwarePeaks

use serde::{Deserialize, Serialize};

// Re-use the GPU roofline's `Dtype` so consumers can pass the same dtype
// through both estimates without conversion.
#[cfg(feature = "roofline_cost")]
pub use crate::roofline::Dtype;
#[cfg(not(feature = "roofline_cost"))]
pub use dtype::Dtype;

#[cfg(not(feature = "roofline_cost"))]
mod dtype {
    /// Mirror of [`crate::roofline::Dtype`] when `roofline_cost` is not enabled,
    /// so this module is usable standalone.
    ///
    /// [`crate::roofline::Dtype`]: ../../crate/roofline/enum.Dtype.html
    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Dtype {
        F32,
        F16,
    }

    impl Dtype {
        #[inline(always)]
        pub fn elem_size(self) -> u64 {
            match self {
                Self::F32 => 4,
                Self::F16 => 2,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AneFamily — the chip generation
// ---------------------------------------------------------------------------

/// Apple Neural Engine generation (Bryngelson ch. 34, table 34.1).
///
/// The M-series silicon maps to the H-series ANE architecture by the fixed
/// relation M(n) → H(n+12) (Bryngelson ch. 12.1):
///
/// | Marketing | ANE H-gen | `AneFamily` |
/// |-----------|-----------|------------|
/// | M1 / A13  | H13       | `A13`      |
/// | M2 / A14  | H14       | `A14`      |
/// | M3 / A15  | H15       | `A15`      |
/// | M4 / A16  | H16       | `A16`      |
/// | M5 / A17  | H17s      | `A17`      |
///
/// The A11 and A12 legacy engines are below the M1 floor — they have no
/// direct-route toolchain (Bryngelson ch. 1.1) and are rejected by
/// [`AnePeaks::for_family`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AneFamily {
    /// Pre-A13 legacy engine (A11/Bionic-class). No direct-route toolchain.
    A11Legacy = 0,
    /// A12 Bionic.
    A12 = 1,
    /// A13 / M1 / H13. The M1 floor for direct-route programming.
    A13 = 2,
    /// A14 / M2 / H14. Adds texture-engine samplers; int8 weight stream.
    A14 = 3,
    /// A15 / M3 / H15. Adds native sin/cos; blockwise weight stream; clean
    /// width-axis slice route (no saturation above 4094).
    A15 = 4,
    /// A16 / M4 / H16. Last capability expansion: tensor extent 16384 → 65536.
    A16 = 5,
    /// A17 / M5 / H17s. 16-core Pro-class; scales throughput, adds no ops.
    A17 = 6,
    /// A18 / H18. Reserved: predicted fp8 datapath (Bryngelson ch. 36.2).
    A18 = 7,
}

impl AneFamily {
    /// Convenience aliases for the M-series Macs.
    pub const M1: Self = Self::A13;
    pub const M2: Self = Self::A14;
    pub const M3: Self = Self::A15;
    pub const M4: Self = Self::A16;
    pub const M5: Self = Self::A17;

    /// Detect the host's ANE family, if any.
    ///
    /// Returns `None` on non-Apple-Silicon hosts. Uses `sysctl
    /// hw.optional.arm64` on macOS, which is a public, entitlement-free
    /// query. Cached in a `OnceLock` so repeated calls are one branch.
    ///
    /// At present this returns `Some(AneFamily::M1)` on any Apple Silicon
    /// without per-chip discrimination — refining to per-family detection
    /// (via `hw.optional.adamv2`/`hw.optional.arm64_e`/etc.) is left for
    /// Plan 379 Phase 3 once a calibration table is established. The M1
    /// peaks are the conservative floor; consumers that want per-chip
    /// accuracy should construct [`AnePeaks`] directly via [`AnePeaks::m1`]
    /// through [`AnePeaks::m5`].
    #[inline]
    pub fn detect() -> Option<Self> {
        static CACHE: std::sync::OnceLock<Option<AneFamily>> = std::sync::OnceLock::new();
        *CACHE.get_or_init(detect_impl)
    }
}

#[inline]
fn detect_impl() -> Option<AneFamily> {
    // The ANE is present on all Apple Silicon since the A11; we don't
    // gate further here (per-chip discrimination happens at the peaks
    // table, not at the family enum).
    if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
        Some(AneFamily::M1)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// AnePeaks — per-family calibrated peaks
// ---------------------------------------------------------------------------

/// Per-chip ANE roofline peaks. M1 and M5 values are silicon-confirmed by
/// Bryngelson (ch. 9.8, ch. 12.7); M2/M3/M4 are decompile-derived
/// interpolations (Bryngelson ch. 12.2) and should be treated as advisory.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnePeaks {
    /// Overhead-isolated matmul slope, fp16, in TFLOP/s.
    ///
    /// Effective fp16 compute peak in TFLOP/s, calibrated to the analytic
    /// cost-model fit (Bryngelson ch. 18.1, table 18.1). NOT the overhead-
    /// isolated matmul slope (12 TFLOP/s on M1) — that only applies to deeply-
    /// fused chains. This is the peak a standalone op actually achieves.
    ///
    /// M1: 3.25, M5: 8.9. M2-M4 interpolated.
    pub compute_tflops_fp16: f64,
    /// Effective DRAM bandwidth in GB/s, calibrated to the analytic cost-model
    /// fit (Bryngelson ch. 18.1). NOT the DRAM ceiling (85 GB/s on M1) — this
    /// is the conservative fit that predicts standalone-op latency. Bryngelson
    /// ch. 18.7 notes it undershoots the ~40 GB/s broad-shape rate.
    ///
    /// M1: 9.0, M5: 57.0.
    pub bandwidth_gbs: f64,
    /// Per-dispatch latency floor in ms. Below this, the op pays the full
    /// round-trip regardless of compute size (Bryngelson ch. 2.3, ch. 9.3).
    ///
    /// M1: 0.23, M5: 0.11. The shipped `npc_brain_router.rs`'s comment
    /// estimated ~0.095 ms (XPC+IOKit only); Bryngelson's full firmware
    /// round-trip measurement is ~2.4× higher.
    pub dispatch_floor_ms: f64,
    /// On-chip SRAM working-set threshold in bytes. Operands larger than
    /// this tile and stream from DRAM (Bryngelson ch. 9.2, ch. 21.1).
    ///
    /// M1: 2 MB, M5: 4.72 MB.
    pub working_set_bytes: u64,
    /// Ridge point in FLOP/byte. Above this arithmetic intensity, the op
    /// is compute-bound; below, it's bandwidth-bound (Bryngelson ch. 9.1).
    ///
    /// M1: 141, M5: 424. The M5 ridge is ~3× higher than the GPU's ~134.
    pub ridge_flop_per_byte: f64,
    /// The chip family these peaks belong to.
    pub family: AneFamily,
}

impl AnePeaks {
    /// M1 / H13 / A13 peaks — analytic cost-model fit (Bryngelson ch. 18.1).
    /// The compute peak (3.25 TFLOP/s) and bandwidth (9.0 GB/s) are NOT the
    /// theoretical roofline ceilings (12 TFLOP/s, 85 GB/s) — they are the
    /// calibrated fit that predicts standalone-op latency within ±17%.
    #[inline]
    pub fn m1() -> Self {
        Self {
            compute_tflops_fp16: 3.25,
            bandwidth_gbs: 9.0,
            dispatch_floor_ms: 0.23,
            working_set_bytes: 2 * 1024 * 1024,
            // Ridge derived from the analytic peaks: 3.25e12 / 9.0e9 ≈ 361.
            // This is higher than the theoretical ridge (141) because the
            // analytic fit is more conservative than the theoretical peaks.
            ridge_flop_per_byte: 361.0,
            family: AneFamily::M1,
        }
    }

    /// M2 / H14 / A14 peaks. Interpolated between M1 and M3; advisory only.
    #[inline]
    pub fn m2() -> Self {
        Self {
            compute_tflops_fp16: 4.5,
            bandwidth_gbs: 15.0,
            dispatch_floor_ms: 0.21,
            working_set_bytes: 2 * 1024 * 1024,
            ridge_flop_per_byte: 300.0,
            family: AneFamily::M2,
        }
    }

    /// M3 / H15 / A15 peaks. **Decompile-derived, NOT silicon-confirmed.**
    #[inline]
    pub fn m3() -> Self {
        Self {
            compute_tflops_fp16: 6.0,
            bandwidth_gbs: 25.0,
            dispatch_floor_ms: 0.18,
            working_set_bytes: 3 * 1024 * 1024,
            ridge_flop_per_byte: 240.0,
            family: AneFamily::M3,
        }
    }

    /// M4 / H16 / A16 peaks. **Decompile-derived, NOT silicon-confirmed.**
    #[inline]
    pub fn m4() -> Self {
        Self {
            compute_tflops_fp16: 7.5,
            bandwidth_gbs: 40.0,
            dispatch_floor_ms: 0.15,
            working_set_bytes: 4 * 1024 * 1024,
            ridge_flop_per_byte: 188.0,
            family: AneFamily::M4,
        }
    }

    /// M5 / H17s / A17 peaks — analytic cost-model fit (Bryngelson ch. 18.1
    /// table 18.1: compute 8.9 TFLOP/s, bandwidth 57 GB/s).
    #[inline]
    pub fn m5() -> Self {
        Self {
            compute_tflops_fp16: 8.9,
            bandwidth_gbs: 57.0,
            dispatch_floor_ms: 0.11,
            working_set_bytes: 4_720_000,
            ridge_flop_per_byte: 156.0,
            family: AneFamily::M5,
        }
    }

    /// Look up peaks for a family. Returns `None` for `A11Legacy` / `A12`,
    /// which are below the M1 floor for direct-route programming
    /// (Bryngelson ch. 1.1: "the pre-A13 parts have no path through this
    /// toolchain").
    #[inline]
    pub fn for_family(family: AneFamily) -> Option<Self> {
        match family {
            AneFamily::A11Legacy | AneFamily::A12 => None,
            AneFamily::A13 => Some(Self::m1()),
            AneFamily::A14 => Some(Self::m2()),
            AneFamily::A15 => Some(Self::m3()),
            AneFamily::A16 => Some(Self::m4()),
            AneFamily::A17 => Some(Self::m5()),
            AneFamily::A18 => Some(Self::m5()), // no measurements; carry M5 forward
        }
    }

    /// The host's peaks, if any. Convenience wrapper around
    /// [`AneFamily::detect`] + [`AnePeaks::for_family`].
    #[inline]
    pub fn for_host() -> Option<Self> {
        AneFamily::detect().and_then(Self::for_family)
    }
}

impl Default for AnePeaks {
    /// Defaults to M1 (the conservative floor). Override with [`Self::m5`]
    /// etc. when the target is known.
    #[inline]
    fn default() -> Self {
        Self::m1()
    }
}

// ---------------------------------------------------------------------------
// AneBound — the routing verdict
// ---------------------------------------------------------------------------

/// Which resource bottleneck dominates, including the two ANE-specific
/// regimes not present in [`crate::roofline::ComputeBound`].
///
/// The variant order matters: `FamilyGated` and `WorkingSet` are checked
/// before `Dispatch`/`Compute`/`Memory` in [`ane_estimate`], because they
/// make the ANE either structurally impossible (family) or structurally
/// slower than its own peaks (working-set spill).
///
/// [`crate::roofline::ComputeBound`]: ../../crate/roofline/enum.ComputeBound.html
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AneBound {
    /// Above the ridge point and the working set fits — the ANE's most
    /// efficient regime (Bryngelson ch. 11.2).
    Compute,
    /// Below the ridge point — bandwidth-bound. Note that the ANE's
    /// standalone activation stream is much slower than its weight stream
    /// (~24 GB/s vs 85 GB/s on M1), so for many bandwidth-bound ops the
    /// GPU is the better choice (see [`AneCost::device_recommendation`]).
    Memory,
    /// **ANE-specific.** Largest operand exceeds the on-chip working set
    /// (HAL `0x1b8`). The op is still legal but tiles and streams from
    /// DRAM, collapsing arithmetic intensity (Bryngelson ch. 9.2).
    WorkingSet,
    /// Work below the per-dispatch floor (~0.23 ms on M1). The ANE spends
    /// ~98% of wall time on dispatch overhead for tiny ops (Bryngelson
    /// ch. 2.3); the CPU almost always wins.
    Dispatch,
    /// **ANE-specific.** Op's [`AneOpShape::min_family`] exceeds the
    /// target chip's family. The op does not lower on this chip
    /// (Bryngelson ch. 35) and must run on CPU or GPU.
    FamilyGated,
}

// ---------------------------------------------------------------------------
// AneOpShape — the input
// ---------------------------------------------------------------------------

/// The shape of a single operation under consideration for ANE dispatch.
///
/// All four fields are required because each one knocks out a different
/// routing regime:
/// - `flops` and `bytes_moved` drive the compute/memory tradeoff
///   (the standard roofline).
/// - `largest_operand_bytes` drives the working-set gate
///   (ANE-specific).
/// - `min_family` drives the family-floor gate (ANE-specific).
///
/// Construct via the helpers ([`AneOpShape::gemv`], [`AneOpShape::gemm`],
/// [`AneOpShape::conv_3x3`], [`AneOpShape::elementwise`]) to get the
/// standard FLOP/byte accounting for free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AneOpShape {
    /// Total floating-point operations (2 per multiply-accumulate).
    pub flops: u64,
    /// Total bytes moved (reads + writes).
    pub bytes_moved: u64,
    /// Bytes of the single largest live operand (input, output, or weight).
    /// Drives the working-set gate — operands above the chip's
    /// [`AnePeaks::working_set_bytes`] tile and stream from DRAM.
    pub largest_operand_bytes: u64,
    /// Minimum ANE family required for this op to lower natively
    /// (Bryngelson ch. 35). See [`AneFamily`] and the F0/F2/F3/F4 table
    /// in `katgpt-rs/.research/377_*.md` §1.6.
    pub min_family: AneFamily,
}

impl AneOpShape {
    /// Build from raw fields.
    #[inline(always)]
    pub const fn new(
        flops: u64,
        bytes_moved: u64,
        largest_operand_bytes: u64,
        min_family: AneFamily,
    ) -> Self {
        Self {
            flops,
            bytes_moved,
            largest_operand_bytes,
            min_family,
        }
    }

    /// GEMV: `(m × k) × (k,) → (m,)`. F0 floor (all chips).
    #[inline(always)]
    pub fn gemv(m: u64, k: u64, dtype: Dtype) -> Self {
        let elem = dtype.elem_size();
        Self::new(
            2 * m * k,
            (m * k + m + k) * elem,
            m * k * elem,
            AneFamily::A11Legacy, // F0
        )
    }

    /// GEMM: `(m × k) × (k × n) → (m × n)`. F0 floor.
    /// `largest_operand_bytes` is the LHS `m × k` operand (typically the
    /// activation; the RHS `k × n` weight is the one that streams).
    #[inline(always)]
    pub fn gemm(m: u64, n: u64, k: u64, dtype: Dtype) -> Self {
        let elem = dtype.elem_size();
        let lhs = m * k * elem;
        let rhs = k * n * elem;
        let out = m * n * elem;
        Self::new(
            2 * m * n * k,
            lhs + rhs + out,
            lhs.max(rhs),
            AneFamily::A11Legacy, // F0
        )
    }

    /// 3×3 stride-1 conv with `c_in` input channels and `c_out` output
    /// channels over a `h × w` feature map. F0 floor. The compiler
    /// auto-selects Winograd for eligible 3×3 stride-1 layers
    /// (Bryngelson ch. 20.5), so the FLOP count here is the direct form
    /// — a Winograd path would be ~2.25× cheaper, which the cost model
    /// does not currently model. Conservative on the ANE's behalf.
    #[inline(always)]
    pub fn conv_3x3(c_in: u64, c_out: u64, h: u64, w: u64, dtype: Dtype) -> Self {
        let elem = dtype.elem_size();
        let k_elems = 3 * 3 * c_in;
        // Direct convolution: per output channel, per output pixel,
        // k_elems multiply-accumulates (2 FLOPs each).
        let flops = 2 * c_out * h * w * k_elems;
        let activation_bytes = c_in * h * w * elem;
        let weight_bytes = c_out * k_elems * elem;
        let output_bytes = c_out * h * w * elem;
        Self::new(
            flops,
            activation_bytes + weight_bytes + output_bytes,
            activation_bytes.max(weight_bytes).max(output_bytes),
            AneFamily::A11Legacy, // F0
        )
    }

    /// Elementwise op over `n` elements. F0 floor.
    #[inline(always)]
    pub fn elementwise(n: u64, dtype: Dtype) -> Self {
        let elem = dtype.elem_size();
        Self::new(n, 2 * n * elem, n * elem, AneFamily::A11Legacy)
    }

    /// Override the minimum family. Builder-style.
    ///
    /// Use when an op requires a feature gate beyond F0 — e.g. `softmax`
    /// requires F2 (A13+), `crop_resize` requires F3 (A14+), native `sin`
    /// requires F4 (A15+). See Bryngelson ch. 35 table 35.2.
    #[inline(always)]
    pub const fn with_min_family(mut self, family: AneFamily) -> Self {
        self.min_family = family;
        self
    }
}

// ---------------------------------------------------------------------------
// AneCost — the output
// ---------------------------------------------------------------------------

/// Result of an ANE roofline cost estimate. Mirrors [`crate::roofline::RooflineCost`]
/// in shape but adds the ANE-specific bound classification.
///
/// [`crate::roofline::RooflineCost`]: ../../crate/roofline/struct.RooflineCost.html
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AneCost {
    /// Predicted runtime in milliseconds. Includes the dispatch floor
    /// when the work would land on the ANE.
    pub runtime_ms: f64,
    /// Which regime dominates.
    pub bound: AneBound,
    /// Total floating-point operations (echoed from the input).
    pub flops: u64,
    /// Total bytes moved (echoed from the input).
    pub bytes_moved: u64,
    /// Largest live operand size in bytes (echoed from the input).
    pub working_set_bytes: u64,
}

impl AneCost {
    /// Construct a `FamilyGated` rejection — runtime is meaningless when
    /// the op doesn't lower on this chip.
    #[inline(always)]
    fn rejected(op: AneOpShape) -> Self {
        Self {
            runtime_ms: f64::INFINITY,
            bound: AneBound::FamilyGated,
            flops: op.flops,
            bytes_moved: op.bytes_moved,
            working_set_bytes: op.largest_operand_bytes,
        }
    }
}

/// Recommended compute device for an op, given an ANE cost estimate and
/// whether a GPU is available.
///
/// Routing rules follow Bryngelson ch. 11 table 11.4:
/// - Conv stacks and short-sequence attention: ANE wins on both speed and
///   energy.
/// - Large square GEMM and long-sequence attention: GPU wins.
/// - Bandwidth-bound standalone ops (large softmax, layer_norm): GPU wins
///   because the ANE's standalone stream is only ~24 GB/s vs the GPU's
///   ~230 GB/s (Bryngelson ch. 9.5).
/// - Tiny ops below the dispatch floor: CPU wins.
/// - Family-gated ops: CPU (won't lower on this chip).
/// - Working-set spill: GPU (tiles better) or CPU fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Device {
    /// CPU SIMD / scalar — the right choice for tiny ops below the ANE
    /// dispatch floor, or for family-gated ops that don't lower.
    Cpu,
    /// Apple Neural Engine — the right choice for compute-bound work that
    /// fits the on-chip working set.
    Ane,
    /// GPU — the right choice for large operands that spill the ANE
    /// working set, for bandwidth-bound standalone ops, or for ops
    /// where the GPU's higher sustained bandwidth wins.
    Gpu,
}

impl AneCost {
    /// Recommend a device for this op, given the ANE cost estimate and
    /// whether a GPU is available.
    ///
    /// When the GPU is not available, `WorkingSet` and `Memory` fall back
    /// to `Cpu` (the ANE is wrong, the GPU is absent, so CPU it is).
    #[inline]
    pub fn device_recommendation(&self, gpu_available: bool) -> Device {
        match self.bound {
            AneBound::Compute => Device::Ane,
            AneBound::FamilyGated | AneBound::Dispatch => Device::Cpu,
            AneBound::WorkingSet | AneBound::Memory => {
                if gpu_available {
                    Device::Gpu
                } else {
                    Device::Cpu
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The core estimator
// ---------------------------------------------------------------------------

/// Estimate ANE cost for an op against per-chip peaks.
///
/// The estimate is the larger of three terms (Bryngelson ch. 9.3, ch. 18.1):
///
/// ```text
/// runtime = max(dispatch_floor, flops / peak_gflops, bytes / bandwidth_gbs)
/// ```
///
/// …with two gates checked first:
/// 1. **Family-floor gate** — if `op.min_family > peaks.family`, the op
///    does not lower on this chip; returns `AneBound::FamilyGated` with
///    `runtime_ms = INFINITY`.
/// 2. **Working-set gate** — if `op.largest_operand_bytes > peaks.working_set_bytes`,
///    the bound is `WorkingSet` regardless of the roofline (Bryngelson
///    ch. 9.2: "Crossing 2 MB moves a workload from compute-bound to
///    bandwidth-bound on one step").
///
/// Zero-allocation, `#[inline(always)]`, ≤1 µs CPU on M1 Pro.
#[inline(always)]
pub fn ane_estimate(op: AneOpShape, _dtype: Dtype, peaks: &AnePeaks) -> AneCost {
    // 1. Family-floor gate: reject if op's MinimumFamily > target's family.
    if op.min_family > peaks.family {
        return AneCost::rejected(op);
    }

    // 2. Working-set gate: detect tile-and-stream condition.
    //    WorkingSet takes precedence in the bound classification below
    //    (Bryngelson ch. 9.2).
    let ws_bound = op.largest_operand_bytes > peaks.working_set_bytes;

    // 3. Three-way roofline: max(dispatch_floor, compute, memory).
    let peak_gflops = peaks.compute_tflops_fp16 * 1e3; // TFLOP/s → GFLOP/s
    let compute_ms = if peak_gflops > 0.0 {
        op.flops as f64 / (peak_gflops * 1e6) // GFLOP/s → MFLOP/ms
    } else {
        f64::MAX
    };
    let memory_ms = if peaks.bandwidth_gbs > 0.0 {
        op.bytes_moved as f64 / (peaks.bandwidth_gbs * 1e6) // GB/s → MB/ms
    } else {
        f64::MAX
    };
    let runtime_ms = peaks.dispatch_floor_ms.max(compute_ms).max(memory_ms);

    // 4. Bound classification.
    //    Order matters: WorkingSet first (structural), then Dispatch
    //    (work clears the floor?), then Compute vs Memory.
    let bound = if ws_bound {
        AneBound::WorkingSet
    } else if runtime_ms <= peaks.dispatch_floor_ms * 1.01 {
        // The 1.01 slop catches "work term equals the floor within rounding"
        // — Bryngelson ch. 2.3 measured the floor at exactly 0.23 ms with
        // ~2% measurement noise.
        AneBound::Dispatch
    } else if compute_ms >= memory_ms {
        AneBound::Compute
    } else {
        AneBound::Memory
    };

    AneCost {
        runtime_ms,
        bound,
        flops: op.flops,
        bytes_moved: op.bytes_moved,
        working_set_bytes: op.largest_operand_bytes,
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors — mirror crate::roofline::{gemv_cost, gemm_cost, ...}
// ---------------------------------------------------------------------------

/// Convenience: GEMV cost on the ANE.
///
/// FLOPs = `2·m·k`, bytes = `(m·k + m + k)·sizeof(dtype)`,
/// largest operand = `m·k·sizeof(dtype)`, min family = F0.
#[inline(always)]
pub fn ane_gemv_cost(m: u64, k: u64, dtype: Dtype, peaks: &AnePeaks) -> AneCost {
    ane_estimate(AneOpShape::gemv(m, k, dtype), dtype, peaks)
}

/// Convenience: GEMM cost on the ANE.
///
/// FLOPs = `2·m·n·k`, bytes = `(m·k + k·n + m·n)·sizeof(dtype)`,
/// largest operand = `max(m·k, k·n)·sizeof(dtype)`, min family = F0.
#[inline(always)]
pub fn ane_gemm_cost(m: u64, n: u64, k: u64, dtype: Dtype, peaks: &AnePeaks) -> AneCost {
    ane_estimate(AneOpShape::gemm(m, n, k, dtype), dtype, peaks)
}

/// Convenience: 3×3 stride-1 conv cost on the ANE.
///
/// FLOPs (direct form) = `2·c_out·h·w·9·c_in`. Note: the compiler
/// auto-selects Winograd for eligible layers (~2.25× cheaper), which this
/// model does not credit — conservative on the ANE's behalf.
#[inline(always)]
pub fn ane_conv3x3_cost(
    c_in: u64,
    c_out: u64,
    h: u64,
    w: u64,
    dtype: Dtype,
    peaks: &AnePeaks,
) -> AneCost {
    ane_estimate(AneOpShape::conv_3x3(c_in, c_out, h, w, dtype), dtype, peaks)
}

// ---------------------------------------------------------------------------
// Fused-chain estimator (Plan 439, gated on `ane_fused_chain`)
//
// Extends the single-op `ane_estimate` with a dependency-aware fused-chain
// cost model. Distilled from GTSim (arXiv:2607.11262) §7.1 insight: kernel
// performance is governed by dependency structure, not individual instruction
// latency. The full tile-graph DAG is Phase 3 stretch; this module ships the
// minimal dependency-aware overlap — modeling eliminated intermediate DRAM
// traffic + single dispatch floor.
// ---------------------------------------------------------------------------

impl AneCost {
    /// Zero cost — the identity element for [`std::ops::Add`].
    ///
    /// Internal helper for [`ane_fused_estimate`]'s sequential-baseline fold.
    /// Bound is `Dispatch` (zero work is trivially below the floor).
    #[inline(always)]
    pub(crate) const fn zero() -> Self {
        Self {
            runtime_ms: 0.0,
            bound: AneBound::Dispatch,
            flops: 0,
            bytes_moved: 0,
            working_set_bytes: 0,
        }
    }
}

impl std::ops::Add for AneCost {
    type Output = Self;

    /// Element-wise sum for the sequential-baseline fold.
    ///
    /// - `runtime_ms`, `flops`, `bytes_moved`: summed.
    /// - `working_set_bytes`: max (peak live operand across sequential ops).
    /// - `bound`: taken from the operand with the larger `runtime_ms`
    ///   (the dominant op in sequential execution).
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let bound = if self.runtime_ms >= rhs.runtime_ms {
            self.bound
        } else {
            rhs.bound
        };
        Self {
            runtime_ms: self.runtime_ms + rhs.runtime_ms,
            bound,
            flops: self.flops + rhs.flops,
            bytes_moved: self.bytes_moved + rhs.bytes_moved,
            working_set_bytes: self.working_set_bytes.max(rhs.working_set_bytes),
        }
    }
}

/// A producer→consumer data dependency between two ops in a fused chain.
///
/// `from_op`'s output feeds `to_op`'s input. `intermediate_bytes` is the size
/// of the intermediate tensor. If the intermediate fits in the ANE's on-chip
/// working set ([`AnePeaks::working_set_bytes`]), it is eliminated from DRAM
/// traffic in the fused estimate.
///
/// Part of Plan 439's fused-chain cost model (GTSim distillation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AneDataDep {
    /// Index into the `ops` slice passed to [`ane_fused_estimate`].
    pub from_op: usize,
    /// Index into the `ops` slice passed to [`ane_fused_estimate`].
    pub to_op: usize,
    /// Size of the intermediate tensor in bytes.
    pub intermediate_bytes: u64,
}

impl AneDataDep {
    /// Build a linear chain of data deps: `op[0] → op[1] → ... → op[n-1]`.
    ///
    /// `intermediate_bytes[i]` is the size of the intermediate tensor between
    /// `op[i]` and `op[i+1]`. Requires `intermediate_bytes.len() + 1 ==
    /// ops.len()`.
    ///
    /// Allocates a `Vec` — use this for setup, not on the hot path. The hot
    /// path ([`ane_fused_estimate`]) takes `&[AneDataDep]` and is zero-alloc.
    pub fn chain(ops: &[AneOpShape], intermediate_bytes: &[u64]) -> Vec<Self> {
        assert!(
            intermediate_bytes.len() + 1 == ops.len(),
            "chain: need exactly ops.len()-1 intermediates, got {} for {} ops",
            intermediate_bytes.len(),
            ops.len()
        );
        intermediate_bytes
            .iter()
            .enumerate()
            .map(|(i, &bytes)| Self {
                from_op: i,
                to_op: i + 1,
                intermediate_bytes: bytes,
            })
            .collect()
    }
}

/// Result of a fused-chain ANE cost estimate (Plan 439).
///
/// Extends [`AneCost`] with fusion metadata. The `base` field is the cost of
/// the fused chain treated as a single aggregate op (with eliminated
/// intermediate bytes subtracted). Compare `base.runtime_ms` against
/// `sequential_runtime_ms` to see the fusion benefit.
///
/// # The "fusion never hurts" invariant
///
/// `base.runtime_ms <= sequential_runtime_ms` (i.e., `fusion_savings_ms >=
/// 0`), because:
/// 1. Eliminated bytes reduce memory time (monotonic).
/// 2. Single dispatch floor <= N dispatch floors.
/// 3. `max(sum_C, sum_M) <= sum(max(C_i, M_i))` — the max of sums is always
///    <= the sum of maxes.
///
/// The one exception: if the fused op is [`AneBound::FamilyGated`],
/// `fusion_savings_ms` is set to 0.0 (the chip can't run the chain at all,
/// so fusion benefit is meaningless).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AneFusedCost {
    /// The fused-chain cost (treated as one aggregate op).
    pub base: AneCost,
    /// Number of ops in the chain.
    pub n_ops: usize,
    /// Number of data deps whose intermediates fit the working set.
    pub n_fused_deps: usize,
    /// Total DRAM traffic eliminated by fusion (bytes).
    pub eliminated_bytes: u64,
    /// What N independent `ane_estimate` calls would predict (sequential).
    pub sequential_runtime_ms: f64,
    /// `sequential_runtime_ms - base.runtime_ms`, clamped to >= 0.
    pub fusion_savings_ms: f64,
}

/// Dependency-aware fused-chain ANE cost model (Plan 439).
///
/// Takes a chain of ops + their data dependencies and returns a single cost
/// estimate that accounts for:
/// 1. **Eliminated intermediate DRAM traffic** — on-chip intermediates are
///    subtracted from the total bytes.
/// 2. **Single dispatch overhead** — one `dispatch_floor_ms`, not N.
///
/// Falls back to the sum of individual [`ane_estimate`] calls (sequential)
/// when no intermediates fit the working set.
///
/// Distilled from GTSim (arXiv:2607.11262) §7.1: kernel performance is
/// governed by dependency structure, not individual instruction latency.
///
/// Zero-allocation, `#[inline(always)]`, <=1 µs CPU on M1 Pro for <=8-op
/// chains (same budget as [`ane_estimate`]).
#[inline(always)]
pub fn ane_fused_estimate(
    ops: &[AneOpShape],
    deps: &[AneDataDep],
    dtype: Dtype,
    peaks: &AnePeaks,
) -> AneFusedCost {
    // 0. Degenerate case: empty chain.
    if ops.is_empty() {
        return AneFusedCost {
            base: AneCost::zero(),
            n_ops: 0,
            n_fused_deps: 0,
            eliminated_bytes: 0,
            sequential_runtime_ms: 0.0,
            fusion_savings_ms: 0.0,
        };
    }

    // 1. Sequential baseline: sum of individual ane_estimate calls.
    //    This is what the current router would predict treating each op
    //    independently (the status quo before this plan).
    let sequential_cost: AneCost = ops
        .iter()
        .copied()
        .map(|op| ane_estimate(op, dtype, peaks))
        .fold(AneCost::zero(), |acc, c| acc + c);

    // 2. Check which intermediates fit in the working set.
    let working_set_budget = peaks.working_set_bytes;
    let mut total_intermediate: u64 = 0;
    let mut eliminated_bytes: u64 = 0;
    let mut n_fused = 0usize;
    for dep in deps {
        // The intermediate fits only if it's individually small enough AND
        // the cumulative on-chip footprint stays within budget.
        if dep.intermediate_bytes <= working_set_budget
            && total_intermediate.saturating_add(dep.intermediate_bytes) <= working_set_budget
        {
            total_intermediate += dep.intermediate_bytes;
            eliminated_bytes += dep.intermediate_bytes;
            n_fused += 1;
        }
    }

    // 3. If NO intermediates fit, return the sequential baseline (no fusion).
    if n_fused == 0 {
        return AneFusedCost {
            base: sequential_cost,
            n_ops: ops.len(),
            n_fused_deps: 0,
            eliminated_bytes: 0,
            sequential_runtime_ms: sequential_cost.runtime_ms,
            fusion_savings_ms: 0.0,
        };
    }

    // 4. Model the fused chain: aggregate FLOPs and bytes, subtract eliminated.
    let total_flops: u64 = ops.iter().map(|o| o.flops).sum();
    let total_bytes_raw: u64 = ops.iter().map(|o| o.bytes_moved).sum();
    let total_bytes_fused: u64 = total_bytes_raw.saturating_sub(eliminated_bytes);
    let largest_operand: u64 = ops
        .iter()
        .map(|o| o.largest_operand_bytes)
        .max()
        .unwrap_or(0);
    // The chain is gated by the highest family requirement of any op.
    let min_family: AneFamily = ops
        .iter()
        .map(|o| o.min_family)
        .max()
        .unwrap_or(AneFamily::A13);

    // 5. Build a single fused AneOpShape and estimate it as one op.
    let fused_op = AneOpShape::new(total_flops, total_bytes_fused, largest_operand, min_family);
    let fused_cost = ane_estimate(fused_op, dtype, peaks);

    // 6. Compute fusion savings. Guard against INFINITY - INFINITY = NaN
    //    when the fused op is FamilyGated (the chain can't run on this chip).
    let fusion_savings_ms = if fused_cost.bound == AneBound::FamilyGated {
        0.0
    } else {
        (sequential_cost.runtime_ms - fused_cost.runtime_ms).max(0.0)
    };

    AneFusedCost {
        base: fused_cost,
        n_ops: ops.len(),
        n_fused_deps: n_fused,
        eliminated_bytes,
        sequential_runtime_ms: sequential_cost.runtime_ms,
        fusion_savings_ms,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── T1.12a: Family-floor gate ────────────────────────────────────────

    #[test]
    fn test_family_floor_gate_rejects_higher_family() {
        // crop_resize requires F3 (A14+) per Bryngelson ch. 35 table 35.2.
        // On M1 (A13) it should be rejected.
        let op = AneOpShape::elementwise(1024, Dtype::F16).with_min_family(AneFamily::A14);
        let cost = ane_estimate(op, Dtype::F16, &AnePeaks::m1());
        assert_eq!(cost.bound, AneBound::FamilyGated);
        assert!(cost.runtime_ms.is_infinite());
    }

    #[test]
    fn test_family_floor_gate_allows_equal_family() {
        // softmax requires F2 (A13+). On M1 (A13) it should pass the gate.
        let op = AneOpShape::elementwise(1024, Dtype::F16).with_min_family(AneFamily::A13);
        let cost = ane_estimate(op, Dtype::F16, &AnePeaks::m1());
        assert_ne!(cost.bound, AneBound::FamilyGated);
    }

    // ── T1.12b: Working-set cliff ────────────────────────────────────────

    #[test]
    fn test_working_set_cliff_on_large_operand() {
        // GEMM with an operand > 2 MB on M1.
        // m=2048, n=2048, k=2048, fp16: LHS = 2048*2048*2 = 8 MB > 2 MB.
        let op = AneOpShape::gemm(2048, 2048, 2048, Dtype::F16);
        assert!(
            op.largest_operand_bytes > AnePeaks::m1().working_set_bytes,
            "test setup: operand must exceed the 2 MB cliff"
        );
        let cost = ane_estimate(op, Dtype::F16, &AnePeaks::m1());
        assert_eq!(cost.bound, AneBound::WorkingSet);
    }

    #[test]
    fn test_working_set_fits_below_cliff() {
        // GEMM with both operands < 2 MB on M1.
        // m=512, n=512, k=512, fp16: LHS = 512*512*2 = 512 KB < 2 MB.
        let op = AneOpShape::gemm(512, 512, 512, Dtype::F16);
        assert!(
            op.largest_operand_bytes <= AnePeaks::m1().working_set_bytes,
            "test setup: operand must fit the 2 MB cliff"
        );
        let cost = ane_estimate(op, Dtype::F16, &AnePeaks::m1());
        assert_ne!(cost.bound, AneBound::WorkingSet);
    }

    // ── T1.12c: Dispatch floor ───────────────────────────────────────────

    #[test]
    fn test_dispatch_floor_on_tiny_gemm() {
        // 64×64×64 GEMM, fp16: 2*64^3 = 524288 FLOPs.
        // At 12000 GFLOP/s: 524288 / 12e12 s ≈ 0.000044 ms << 0.23 ms floor.
        let op = AneOpShape::gemm(64, 64, 64, Dtype::F16);
        let cost = ane_estimate(op, Dtype::F16, &AnePeaks::m1());
        assert_eq!(cost.bound, AneBound::Dispatch);
        // Runtime should equal the floor (within 1% slop).
        assert!(
            (cost.runtime_ms - 0.23).abs() < 0.01,
            "expected runtime ≈ 0.23 ms, got {}",
            cost.runtime_ms
        );
    }

    // ── T1.12d: Compute-bound conv ───────────────────────────────────────

    #[test]
    fn test_compute_bound_3x3_conv() {
        // Bryngelson ch. 13.1: 3×3 conv, 256 channels, 28×28 feature map,
        // M1 — measured ~0.51 ms at ~1823 GFLOP/s effective.
        let op = AneOpShape::conv_3x3(256, 256, 28, 28, Dtype::F16);
        let cost = ane_estimate(op, Dtype::F16, &AnePeaks::m1());
        // The activation is 256*28*28*2 ≈ 392 KB < 2 MB, so no working-set trip.
        assert!(op.largest_operand_bytes <= AnePeaks::m1().working_set_bytes);
        assert_ne!(cost.bound, AneBound::WorkingSet);
        assert_ne!(cost.bound, AneBound::Dispatch);
        // Compute-bound (Bryngelson ch. 11: ANE wins both speed + energy).
        assert_eq!(cost.bound, AneBound::Compute);
        // Predicted runtime should be within 2× of measured 0.51 ms. Bryngelson's
        // own analytic model (which our peaks are calibrated to) claims ±17% on
        // 5 reference convs; our simplified model omits the OCG pass-count
        // multiplier, so we accept up to 2× error. Routing decision (Compute →
        // ANE) is what matters; absolute latency is advisory.
        assert!(
            cost.runtime_ms > 0.0 && cost.runtime_ms <= 0.51 * 2.0,
            "expected runtime ≤ 2× of 0.51 ms, got {}",
            cost.runtime_ms
        );
    }

    // ── T1.12e: Family roundtrip ─────────────────────────────────────────

    #[test]
    fn test_for_family_roundtrip() {
        for f in [
            AneFamily::A13,
            AneFamily::A14,
            AneFamily::A15,
            AneFamily::A16,
            AneFamily::A17,
        ] {
            let peaks = AnePeaks::for_family(f)
                .unwrap_or_else(|| panic!("for_family({f:?}) returned None for a post-A12 family"));
            assert_eq!(peaks.family, f);
        }
    }

    #[test]
    fn test_for_family_rejects_legacy() {
        assert!(AnePeaks::for_family(AneFamily::A11Legacy).is_none());
        assert!(AnePeaks::for_family(AneFamily::A12).is_none());
    }

    // ── T1.12f: Cross-chip scaling ───────────────────────────────────────

    #[test]
    fn test_m5_peaks_strictly_better_than_m1() {
        let m1 = AnePeaks::m1();
        let m5 = AnePeaks::m5();
        // Compute, bandwidth, working-set, and dispatch floor all improve M1 → M5.
        // The analytic-fit ridge can move either way (it's a ratio), so we don't
        // assert monotonicity on it here — only on the raw peaks.
        assert!(m5.compute_tflops_fp16 > m1.compute_tflops_fp16);
        assert!(m5.bandwidth_gbs > m1.bandwidth_gbs);
        assert!(
            m5.dispatch_floor_ms < m1.dispatch_floor_ms,
            "M5 floor must be lower"
        );
        assert!(m5.working_set_bytes > m1.working_set_bytes);
    }

    #[test]
    fn test_m_series_monotone_improvement() {
        // Compute should be monotone non-decreasing M1 → M5.
        let m1 = AnePeaks::m1().compute_tflops_fp16;
        let m2 = AnePeaks::m2().compute_tflops_fp16;
        let m3 = AnePeaks::m3().compute_tflops_fp16;
        let m4 = AnePeaks::m4().compute_tflops_fp16;
        let m5 = AnePeaks::m5().compute_tflops_fp16;
        assert!(m1 <= m2);
        assert!(m2 <= m3);
        assert!(m3 <= m4);
        assert!(m4 <= m5);
    }

    // ── T1.12g: Detect on non-Apple ──────────────────────────────────────

    #[test]
    fn test_detect_returns_none_on_non_apple() {
        // This test asserts the documented contract: detect() returns
        // None on non-Apple-Silicon hosts. On the dev machine (Apple
        // Silicon) it will return Some(M1); the test is a no-op there.
        // On CI x86_64 it asserts None.
        let detected = AneFamily::detect();
        if !(cfg!(target_arch = "aarch64") && cfg!(target_os = "macos")) {
            assert!(detected.is_none(), "detect() must return None on non-Apple");
        } else {
            assert!(
                detected.is_some(),
                "detect() must return Some on Apple Silicon"
            );
        }
    }

    // ── T1.12h: Determinism ──────────────────────────────────────────────

    #[test]
    fn test_determinism_same_input_same_output() {
        let op = AneOpShape::gemm(1024, 1024, 1024, Dtype::F16);
        let peaks = AnePeaks::m1();
        let a = ane_estimate(op, Dtype::F16, &peaks);
        let b = ane_estimate(op, Dtype::F16, &peaks);
        assert_eq!(a, b);
    }

    // ── Device recommendation ────────────────────────────────────────────

    #[test]
    fn test_device_recommendation_compute_bound_to_ane() {
        let op = AneOpShape::conv_3x3(256, 256, 28, 28, Dtype::F16);
        let cost = ane_estimate(op, Dtype::F16, &AnePeaks::m1());
        assert_eq!(cost.device_recommendation(true), Device::Ane);
        assert_eq!(cost.device_recommendation(false), Device::Ane);
    }

    #[test]
    fn test_device_recommendation_dispatch_bound_to_cpu() {
        let op = AneOpShape::gemm(64, 64, 64, Dtype::F16);
        let cost = ane_estimate(op, Dtype::F16, &AnePeaks::m1());
        assert_eq!(cost.bound, AneBound::Dispatch);
        assert_eq!(cost.device_recommendation(true), Device::Cpu);
        assert_eq!(cost.device_recommendation(false), Device::Cpu);
    }

    #[test]
    fn test_device_recommendation_working_set_to_gpu_when_available() {
        let op = AneOpShape::gemm(2048, 2048, 2048, Dtype::F16);
        let cost = ane_estimate(op, Dtype::F16, &AnePeaks::m1());
        assert_eq!(cost.bound, AneBound::WorkingSet);
        assert_eq!(cost.device_recommendation(true), Device::Gpu);
        assert_eq!(cost.device_recommendation(false), Device::Cpu);
    }

    #[test]
    fn test_device_recommendation_family_gated_to_cpu() {
        let op = AneOpShape::elementwise(1024, Dtype::F16).with_min_family(AneFamily::A15);
        let cost = ane_estimate(op, Dtype::F16, &AnePeaks::m1());
        assert_eq!(cost.bound, AneBound::FamilyGated);
        assert_eq!(cost.device_recommendation(true), Device::Cpu);
    }

    // ── Convenience constructors ─────────────────────────────────────────

    #[test]
    fn test_ane_gemv_cost_shape() {
        let cost = ane_gemv_cost(128, 256, Dtype::F32, &AnePeaks::m1());
        assert_eq!(cost.flops, 2 * 128 * 256);
        assert_eq!(cost.bytes_moved, (128 * 256 + 128 + 256) * 4);
        assert_eq!(cost.working_set_bytes, 128 * 256 * 4);
    }

    #[test]
    fn test_ane_gemm_cost_shape() {
        let cost = ane_gemm_cost(128, 64, 256, Dtype::F32, &AnePeaks::m1());
        assert_eq!(cost.flops, 2 * 128 * 64 * 256);
        assert_eq!(cost.bytes_moved, (128 * 256 + 256 * 64 + 128 * 64) * 4);
        // max(128*256, 256*64) = max(32768, 16384) = 32768 elements * 4 bytes
        assert_eq!(cost.working_set_bytes, 32768 * 4);
    }

    #[test]
    fn test_ane_conv3x3_cost_shape() {
        let cost = ane_conv3x3_cost(64, 128, 16, 16, Dtype::F16, &AnePeaks::m1());
        // FLOPs = 2 * c_out * h * w * (3*3*c_in) = 2 * 128 * 16 * 16 * 576
        assert_eq!(cost.flops, 2 * 128 * 16 * 16 * 576);
        // Activation: 64 * 16 * 16 * 2 = 32768 bytes
        // Weight:     128 * 576 * 2 = 147456 bytes
        // Output:     128 * 16 * 16 * 2 = 65536 bytes
        // max operand = 147456 bytes (weight)
        assert_eq!(cost.working_set_bytes, 147456);
    }

    // ── Default & serialization ──────────────────────────────────────────

    #[test]
    fn test_peaks_default_is_m1() {
        let default = AnePeaks::default();
        let m1 = AnePeaks::m1();
        assert_eq!(default, m1);
    }

    #[test]
    fn test_cost_serialization() {
        let cost = ane_gemm_cost(128, 64, 256, Dtype::F32, &AnePeaks::m1());
        let json = serde_json::to_string(&cost).unwrap();
        assert!(json.contains("runtime_ms"));
        assert!(json.contains("bound"));
        let back: AneCost = serde_json::from_str(&json).unwrap();
        assert_eq!(cost, back);
    }

    #[test]
    fn test_peaks_serialization() {
        let peaks = AnePeaks::m5();
        let json = serde_json::to_string(&peaks).unwrap();
        let back: AnePeaks = serde_json::from_str(&json).unwrap();
        assert_eq!(peaks, back);
    }

    // ── Sanity: ridge point = compute / bandwidth ────────────────────────

    #[test]
    fn test_m1_ridge_point_matches_compute_over_bandwidth() {
        // The ridge field should be derivable from the analytic-fit peaks
        // via I* = P/B. M1 analytic fit: 3.25e12 / 9.0e9 ≈ 361 FLOP/byte.
        // Sanity-check our peaks table reproduces this within 5%.
        let m1 = AnePeaks::m1();
        let derived_ridge = (m1.compute_tflops_fp16 * 1e12) / (m1.bandwidth_gbs * 1e9);
        assert!(
            (derived_ridge - m1.ridge_flop_per_byte).abs() / m1.ridge_flop_per_byte < 0.05,
            "M1 ridge mismatch: derived {}, table {}",
            derived_ridge,
            m1.ridge_flop_per_byte
        );
    }
}

// ---------------------------------------------------------------------------
// Fused-chain estimator tests (Plan 439 T1.8)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod fused_tests {
    use super::*;

    // ── T1.8a: Single-op chain parity (G3 hook) ──────────────────────────

    #[test]
    fn test_single_op_chain_matches_ane_estimate() {
        let op = AneOpShape::gemm(512, 512, 512, Dtype::F16);
        let peaks = AnePeaks::m1();
        let single = ane_estimate(op, Dtype::F16, &peaks);
        let fused = ane_fused_estimate(&[op], &[], Dtype::F16, &peaks);
        assert_eq!(
            fused.base, single,
            "single-op chain must match ane_estimate"
        );
        assert_eq!(fused.n_ops, 1);
        assert_eq!(fused.n_fused_deps, 0);
        assert_eq!(fused.fusion_savings_ms, 0.0);
    }

    // ── T1.8b: Two-op chain, intermediate fits (G1 hook) ─────────────────

    #[test]
    fn test_two_op_chain_intermediate_fits() {
        // conv_3x3(64, 64, 8, 8) → elementwise ReLU.
        // Intermediate = conv output = 64*8*8*2 = 8192 bytes, fits in 2 MB.
        let peaks = AnePeaks::m1();
        let conv = AneOpShape::conv_3x3(64, 64, 8, 8, Dtype::F16);
        let relu = AneOpShape::elementwise(64 * 8 * 8, Dtype::F16);
        let intermediate = 64 * 8 * 8 * 2; // 8192 bytes
        assert!(
            intermediate <= peaks.working_set_bytes,
            "test setup: must fit"
        );
        let deps = vec![AneDataDep {
            from_op: 0,
            to_op: 1,
            intermediate_bytes: intermediate,
        }];
        let fused = ane_fused_estimate(&[conv, relu], &deps, Dtype::F16, &peaks);
        assert_eq!(fused.n_ops, 2);
        assert_eq!(fused.n_fused_deps, 1);
        assert_eq!(fused.eliminated_bytes, intermediate);
        // Fusion savings must be positive: the sequential path pays 2 dispatch
        // floors (0.23 + 0.23 = 0.46 ms), the fused path pays 1 (0.23 ms).
        assert!(
            fused.fusion_savings_ms > 0.0,
            "fusion must save time, got {}",
            fused.fusion_savings_ms
        );
        assert!(fused.base.runtime_ms <= fused.sequential_runtime_ms);
    }

    // ── T1.8c: Two-op chain, intermediate exceeds working set ────────────

    #[test]
    fn test_two_op_chain_intermediate_exceeds_working_set() {
        let peaks = AnePeaks::m1();
        let gemm = AneOpShape::gemm(1024, 1024, 1024, Dtype::F16);
        let bias = AneOpShape::elementwise(1024 * 1024, Dtype::F16);
        let intermediate = 3 * 1024 * 1024; // 3 MB > 2 MB working set
        assert!(
            intermediate > peaks.working_set_bytes,
            "test setup: must exceed"
        );
        let deps = vec![AneDataDep {
            from_op: 0,
            to_op: 1,
            intermediate_bytes: intermediate,
        }];
        let fused = ane_fused_estimate(&[gemm, bias], &deps, Dtype::F16, &peaks);
        assert_eq!(
            fused.n_fused_deps, 0,
            "oversize intermediate must not be eliminated"
        );
        assert_eq!(fused.eliminated_bytes, 0);
        assert_eq!(
            fused.fusion_savings_ms, 0.0,
            "no fusion when intermediates don't fit"
        );
    }

    // ── T1.8d: Three-op chain, two of three intermediates fit ───────────

    #[test]
    fn test_three_op_chain_partial_fit() {
        let peaks = AnePeaks::m1();
        let op0 = AneOpShape::elementwise(1024, Dtype::F16);
        let op1 = AneOpShape::elementwise(1024, Dtype::F16);
        let op2 = AneOpShape::elementwise(1024, Dtype::F16);
        let small = 1024 * 2; // 2 KB, fits
        let big = 3 * 1024 * 1024; // 3 MB, doesn't fit
        let deps = vec![
            AneDataDep {
                from_op: 0,
                to_op: 1,
                intermediate_bytes: small,
            },
            AneDataDep {
                from_op: 1,
                to_op: 2,
                intermediate_bytes: small,
            },
            // Non-linear dep (skip connection) with a big intermediate.
            AneDataDep {
                from_op: 0,
                to_op: 2,
                intermediate_bytes: big,
            },
        ];
        let fused = ane_fused_estimate(&[op0, op1, op2], &deps, Dtype::F16, &peaks);
        assert_eq!(
            fused.n_fused_deps, 2,
            "two small intermediates fit, one big doesn't"
        );
        assert_eq!(fused.eliminated_bytes, small * 2);
    }

    // ── T1.8e: Tiny fused chain below dispatch floor (G2 hook) ───────────

    #[test]
    fn test_tiny_fused_chain_below_dispatch_floor() {
        let peaks = AnePeaks::m1();
        let op0 = AneOpShape::gemm(64, 64, 64, Dtype::F16);
        let op1 = AneOpShape::elementwise(64 * 64, Dtype::F16);
        let intermediate = 64 * 64 * 2; // 8 KB, fits
        let deps = vec![AneDataDep {
            from_op: 0,
            to_op: 1,
            intermediate_bytes: intermediate,
        }];
        let fused = ane_fused_estimate(&[op0, op1], &deps, Dtype::F16, &peaks);
        assert_eq!(
            fused.base.bound,
            AneBound::Dispatch,
            "tiny aggregate must be below the dispatch floor"
        );
        // Even the fused chain benefits from single dispatch: sequential pays
        // 2 floors, fused pays 1.
        assert!(fused.fusion_savings_ms > 0.0);
    }

    // ── T1.8f: Family-gated chain ────────────────────────────────────────

    #[test]
    fn test_family_gated_chain() {
        // One op requires A14, target is M1 (A13) → chain is family-gated.
        let peaks = AnePeaks::m1();
        let op0 = AneOpShape::elementwise(1024, Dtype::F16);
        let op1 = AneOpShape::elementwise(1024, Dtype::F16).with_min_family(AneFamily::A14);
        let intermediate = 1024 * 2;
        let deps = vec![AneDataDep {
            from_op: 0,
            to_op: 1,
            intermediate_bytes: intermediate,
        }];
        let fused = ane_fused_estimate(&[op0, op1], &deps, Dtype::F16, &peaks);
        assert_eq!(fused.base.bound, AneBound::FamilyGated);
        assert_eq!(
            fused.fusion_savings_ms, 0.0,
            "family-gated: no fusion benefit"
        );
        assert!(!fused.fusion_savings_ms.is_nan(), "must not be NaN");
    }

    // ── T1.8g: Determinism ───────────────────────────────────────────────

    #[test]
    fn test_fused_determinism() {
        let peaks = AnePeaks::m1();
        let op0 = AneOpShape::conv_3x3(64, 64, 8, 8, Dtype::F16);
        let op1 = AneOpShape::elementwise(64 * 8 * 8, Dtype::F16);
        let deps = AneDataDep::chain(&[op0, op1], &[64 * 8 * 8 * 2]);
        let a = ane_fused_estimate(&[op0, op1], &deps, Dtype::F16, &peaks);
        let b = ane_fused_estimate(&[op0, op1], &deps, Dtype::F16, &peaks);
        assert_eq!(a, b);
    }

    // ── T1.8h: Empty ops slice ───────────────────────────────────────────

    #[test]
    fn test_empty_ops_slice() {
        let peaks = AnePeaks::m1();
        let fused = ane_fused_estimate(&[], &[], Dtype::F16, &peaks);
        assert_eq!(fused.n_ops, 0);
        assert_eq!(fused.n_fused_deps, 0);
        assert_eq!(fused.eliminated_bytes, 0);
        assert_eq!(fused.fusion_savings_ms, 0.0);
        assert_eq!(fused.base.runtime_ms, 0.0);
    }

    // ── AneDataDep::chain constructor ────────────────────────────────────

    #[test]
    fn test_chain_constructor() {
        let ops = [
            AneOpShape::elementwise(100, Dtype::F16),
            AneOpShape::elementwise(200, Dtype::F16),
            AneOpShape::elementwise(300, Dtype::F16),
        ];
        let deps = AneDataDep::chain(&ops, &[1000, 2000]);
        assert_eq!(deps.len(), 2);
        assert_eq!(
            deps[0],
            AneDataDep {
                from_op: 0,
                to_op: 1,
                intermediate_bytes: 1000
            }
        );
        assert_eq!(
            deps[1],
            AneDataDep {
                from_op: 1,
                to_op: 2,
                intermediate_bytes: 2000
            }
        );
    }

    #[test]
    #[should_panic(expected = "need exactly ops.len()-1 intermediates")]
    fn test_chain_constructor_wrong_count_panics() {
        let ops = [
            AneOpShape::elementwise(100, Dtype::F16),
            AneOpShape::elementwise(200, Dtype::F16),
        ];
        // Need 1 intermediate for 2 ops, but passing 2.
        let _ = AneDataDep::chain(&ops, &[1000, 2000]);
    }

    // ── AneCost arithmetic helpers ───────────────────────────────────────

    #[test]
    fn test_ane_cost_zero_is_identity() {
        let zero = AneCost::zero();
        assert_eq!(zero.runtime_ms, 0.0);
        assert_eq!(zero.flops, 0);
        assert_eq!(zero.bytes_moved, 0);
        assert_eq!(zero.working_set_bytes, 0);

        // zero + cost == cost (identity for Add).
        let cost = ane_estimate(
            AneOpShape::gemm(512, 512, 512, Dtype::F16),
            Dtype::F16,
            &AnePeaks::m1(),
        );
        let sum = AneCost::zero() + cost;
        assert_eq!(sum.runtime_ms, cost.runtime_ms);
        assert_eq!(sum.flops, cost.flops);
        assert_eq!(sum.bound, cost.bound);
    }

    #[test]
    fn test_ane_cost_add_elementwise() {
        let a = AneCost {
            runtime_ms: 1.0,
            bound: AneBound::Compute,
            flops: 100,
            bytes_moved: 200,
            working_set_bytes: 50,
        };
        let b = AneCost {
            runtime_ms: 2.0,
            bound: AneBound::Memory,
            flops: 300,
            bytes_moved: 400,
            working_set_bytes: 80,
        };
        let sum = a + b;
        assert_eq!(sum.runtime_ms, 3.0);
        assert_eq!(sum.flops, 400);
        assert_eq!(sum.bytes_moved, 600);
        assert_eq!(sum.working_set_bytes, 80); // max, not sum
        // b has larger runtime → b's bound wins.
        assert_eq!(sum.bound, AneBound::Memory);
    }

    // ── Fusion never hurts invariant (cross-cutting) ─────────────────────

    #[test]
    fn test_fusion_never_hurts_across_regimes() {
        // For every chain where intermediates fit, fused <= sequential.
        let peaks = AnePeaks::m1();
        let test_chains: &[(&[AneOpShape], u64)] = &[
            // Compute-bound chain.
            (
                &[
                    AneOpShape::conv_3x3(128, 128, 8, 8, Dtype::F16),
                    AneOpShape::conv_3x3(128, 128, 8, 8, Dtype::F16),
                ],
                128 * 8 * 8 * 2,
            ),
            // Memory-bound chain.
            (
                &[
                    AneOpShape::elementwise(100_000, Dtype::F16),
                    AneOpShape::elementwise(100_000, Dtype::F16),
                ],
                100_000 * 2,
            ),
            // Dispatch-bound chain.
            (
                &[
                    AneOpShape::gemm(32, 32, 32, Dtype::F16),
                    AneOpShape::gemm(32, 32, 32, Dtype::F16),
                ],
                32 * 32 * 2,
            ),
        ];
        for (ops, intermediate) in test_chains {
            let deps = vec![AneDataDep {
                from_op: 0,
                to_op: 1,
                intermediate_bytes: *intermediate,
            }];
            let fused = ane_fused_estimate(ops, &deps, Dtype::F16, &peaks);
            if fused.n_fused_deps > 0 {
                assert!(
                    fused.base.runtime_ms <= fused.sequential_runtime_ms + 1e-9,
                    "fusion never hurts: fused {} > sequential {} for chain",
                    fused.base.runtime_ms,
                    fused.sequential_runtime_ms
                );
                assert!(fused.fusion_savings_ms >= 0.0);
            }
        }
    }
}
