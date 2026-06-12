//! Roofline cost model for GPU operator runtime prediction (Plan 159, Research R130).
//!
//! Ports FlashLib's `info/roofline.py` to Rust. Predicts operator runtime in ~5µs
//! CPU-only, replacing ~100ms GemvAutotune benchmarking.
//!
//! The model uses calibrated hardware peak throughput to predict whether an operator
//! is compute-bound, memory-bound, or launch-overhead-bound, and estimates runtime.

use serde::{Deserialize, Serialize};

/// Which resource bottleneck dominates the operation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeBound {
    /// Operation limited by FLOP throughput.
    Compute,
    /// Operation limited by memory bandwidth.
    Memory,
    /// Operation too small to saturate hardware; launch overhead dominates.
    Launch,
}

/// Result of a roofline cost estimate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RooflineCost {
    /// Total floating-point operations.
    pub flops: u64,
    /// Total bytes moved (reads + writes).
    pub bytes_moved: u64,
    /// Predicted runtime in milliseconds.
    pub runtime_ms: f64,
    /// Which resource bottleneck dominates.
    pub bound: ComputeBound,
}

/// Data type for roofline estimation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
}

impl Dtype {
    #[inline(always)]
    fn elem_size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Operator type for roofline estimation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    /// General matrix-vector multiply: (m × k) × (k,) → (m,)
    Gemv,
    /// General matrix-matrix multiply: (m × k) × (k × n) → (m × n)
    Gemm,
    /// Elementwise operation (activation, normalization).
    Elementwise,
    /// Reduction (sum, max over axis).
    Reduction,
}

/// Hardware peak throughput calibration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HardwarePeaks {
    /// Peak compute throughput in GFLOP/s (f32).
    pub peak_gflops_f32: f64,
    /// Peak compute throughput in GFLOP/s (f16).
    pub peak_gflops_f16: f64,
    /// Peak memory bandwidth in GB/s.
    pub peak_bandwidth_gbs: f64,
    /// GPU kernel launch overhead in microseconds.
    pub launch_overhead_us: f64,
}

impl Default for HardwarePeaks {
    fn default() -> Self {
        // Apple M1 Pro calibrated peaks (from benchmarks)
        Self {
            peak_gflops_f32: 550.0,
            peak_gflops_f16: 1100.0,
            peak_bandwidth_gbs: 200.0,
            launch_overhead_us: 50.0,
        }
    }
}

impl HardwarePeaks {
    /// Apple M1 baseline.
    pub fn apple_m1() -> Self {
        Self {
            peak_gflops_f32: 440.0,
            peak_gflops_f16: 880.0,
            peak_bandwidth_gbs: 68.0,
            launch_overhead_us: 50.0,
        }
    }

    /// Apple M2 Pro baseline.
    pub fn apple_m2_pro() -> Self {
        Self {
            peak_gflops_f32: 550.0,
            peak_gflops_f16: 1100.0,
            peak_bandwidth_gbs: 200.0,
            launch_overhead_us: 50.0,
        }
    }

    /// Apple M3 Pro baseline.
    pub fn apple_m3_pro() -> Self {
        Self {
            peak_gflops_f32: 600.0,
            peak_gflops_f16: 1200.0,
            peak_bandwidth_gbs: 150.0,
            launch_overhead_us: 45.0,
        }
    }

    /// Apple M4 Pro baseline.
    pub fn apple_m4_pro() -> Self {
        Self {
            peak_gflops_f32: 700.0,
            peak_gflops_f16: 1400.0,
            peak_bandwidth_gbs: 120.0,
            launch_overhead_us: 40.0,
        }
    }
}

/// Estimate operator cost using the roofline model.
///
/// Given FLOPs and bytes moved, predicts runtime as:
/// ```text
/// runtime = max(launch_overhead, flops / peak_gflops, bytes / peak_bandwidth)
/// ```
///
/// The bottleneck determines the `bound` classification.
#[inline(always)]
pub fn roofline_estimate(
    _op: OpType,
    dtype: Dtype,
    flops: u64,
    bytes: u64,
    hw: &HardwarePeaks,
) -> RooflineCost {
    let peak_gflops = match dtype {
        Dtype::F32 => hw.peak_gflops_f32,
        Dtype::F16 => hw.peak_gflops_f16,
    };

    // Compute time in ms
    let compute_ms = if peak_gflops > 0.0 {
        flops as f64 / (peak_gflops * 1e6) // GFLOP/s → MFLOP/ms
    } else {
        f64::MAX
    };

    // Memory time in ms
    let memory_ms = if hw.peak_bandwidth_gbs > 0.0 {
        bytes as f64 / (hw.peak_bandwidth_gbs * 1e6) // GB/s → MB/ms
    } else {
        f64::MAX
    };

    // Launch overhead in ms
    let launch_ms = hw.launch_overhead_us / 1000.0;

    let runtime_ms = launch_ms.max(compute_ms).max(memory_ms);

    let bound = if runtime_ms <= launch_ms * 1.01 {
        ComputeBound::Launch
    } else if compute_ms >= memory_ms {
        ComputeBound::Compute
    } else {
        ComputeBound::Memory
    };

    RooflineCost {
        runtime_ms,
        flops,
        bytes_moved: bytes,
        bound,
    }
}

/// Convenience: estimate cost for a GEMV operation (m × k) vector.
///
/// FLOPs = 2 * m * k, bytes = (m * k + m + k) * sizeof(dtype).
#[inline(always)]
pub fn gemv_cost(m: u64, k: u64, dtype: Dtype, hw: &HardwarePeaks) -> RooflineCost {
    let elem_size = dtype.elem_size();
    let flops = 2 * m * k;
    let bytes = (m * k + m + k) * elem_size;
    roofline_estimate(OpType::Gemv, dtype, flops, bytes, hw)
}

/// Convenience: estimate cost for a GEMM operation (m × k) × (k × n).
///
/// FLOPs = 2 * m * n * k, bytes = (m*k + k*n + m*n) * sizeof(dtype).
#[inline(always)]
pub fn gemm_cost(m: u64, n: u64, k: u64, dtype: Dtype, hw: &HardwarePeaks) -> RooflineCost {
    let elem_size = dtype.elem_size();
    let flops = 2 * m * n * k;
    let bytes = (m * k + k * n + m * n) * elem_size;
    roofline_estimate(OpType::Gemm, dtype, flops, bytes, hw)
}

/// Convenience: estimate cost for Gram matrix computation G = X·Xᵀ.
///
/// X is (seq_len × d_h), G is (seq_len × seq_len).
/// FLOPs = seq_len² * d_h (upper triangle + mirror ≈ seq_len² * d_h).
/// Bytes = (seq_len * d_h + seq_len²) * sizeof(dtype).
#[inline(always)]
pub fn gram_cost(seq_len: u64, d_h: u64, dtype: Dtype, hw: &HardwarePeaks) -> RooflineCost {
    let elem_size = dtype.elem_size();
    // Upper triangle: seq_len*(seq_len+1)/2 dot products, each of length d_h
    // But we count full for safety: seq_len * seq_len * d_h * 2 FLOPs
    let flops = seq_len * (seq_len + 1) / 2 * d_h * 2;
    let bytes = (seq_len * d_h + seq_len * seq_len) * elem_size;
    roofline_estimate(OpType::Gemm, dtype, flops, bytes, hw)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hw() -> HardwarePeaks {
        HardwarePeaks {
            peak_gflops_f32: 550.0,
            peak_gflops_f16: 1100.0,
            peak_bandwidth_gbs: 200.0,
            launch_overhead_us: 50.0,
        }
    }

    #[test]
    fn test_gemv_small_is_launch_bound() {
        let cost = gemv_cost(1, 1, Dtype::F32, &test_hw());
        assert_eq!(cost.bound, ComputeBound::Launch);
        assert!(cost.runtime_ms >= 0.049); // ~50µs launch overhead
    }

    #[test]
    fn test_gemv_large_is_compute_or_memory_bound() {
        let cost = gemv_cost(4096, 4096, Dtype::F32, &test_hw());
        assert_ne!(cost.bound, ComputeBound::Launch);
    }

    #[test]
    fn test_roofline_f16_faster_than_f32() {
        let f32_cost = gemv_cost(1024, 1024, Dtype::F32, &test_hw());
        let f16_cost = gemv_cost(1024, 1024, Dtype::F16, &test_hw());
        // F16 compute should be faster (2x peak FLOPS), but bytes may dominate
        // At minimum, the compute time should be <= f32 compute time
        assert!(f16_cost.runtime_ms <= f32_cost.runtime_ms * 1.01 + 0.001);
    }

    #[test]
    fn test_gemm_cost_shape() {
        let cost = gemm_cost(128, 64, 256, Dtype::F32, &test_hw());
        assert_eq!(cost.flops, 2 * 128 * 64 * 256);
        assert_eq!(cost.bytes_moved, (128 * 256 + 256 * 64 + 128 * 64) * 4);
    }

    #[test]
    fn test_gram_cost_small() {
        let cost = gram_cost(16, 128, Dtype::F32, &test_hw());
        // Should be launch-bound for small sizes
        assert_eq!(cost.bound, ComputeBound::Launch);
    }

    #[test]
    fn test_gram_cost_values() {
        let cost = gram_cost(64, 128, Dtype::F32, &test_hw());
        // flops = 64*(64+1)/2 * 128 * 2 = 533,120
        assert_eq!(cost.flops, 64 * 65 / 2 * 128 * 2);
    }

    #[test]
    fn test_hardware_peaks_default() {
        let hw = HardwarePeaks::default();
        assert!(hw.peak_gflops_f32 > 0.0);
        assert!(hw.peak_gflops_f16 > 0.0);
        assert!(hw.peak_bandwidth_gbs > 0.0);
        assert!(hw.launch_overhead_us > 0.0);
    }

    #[test]
    fn test_hardware_peaks_family() {
        let m1 = HardwarePeaks::apple_m1();
        let m2 = HardwarePeaks::apple_m2_pro();
        let m3 = HardwarePeaks::apple_m3_pro();
        let m4 = HardwarePeaks::apple_m4_pro();
        // Each generation should have reasonable peaks
        assert!(m1.peak_gflops_f32 > 0.0);
        assert!(m2.peak_gflops_f32 > m1.peak_gflops_f32);
        assert!(m3.peak_gflops_f32 > m2.peak_gflops_f32);
        assert!(m4.peak_gflops_f32 > m3.peak_gflops_f32);
    }

    #[test]
    fn test_compute_bound_classification() {
        // Very small op → launch bound
        let cost = roofline_estimate(OpType::Elementwise, Dtype::F32, 1, 1, &test_hw());
        assert_eq!(cost.bound, ComputeBound::Launch);

        // Heavy compute, little memory → compute bound
        let cost = roofline_estimate(OpType::Gemm, Dtype::F32, 1_000_000_000, 100, &test_hw());
        assert_eq!(cost.bound, ComputeBound::Compute);

        // Little compute, heavy memory → memory bound
        let cost = roofline_estimate(
            OpType::Reduction,
            Dtype::F32,
            100,
            1_000_000_000,
            &test_hw(),
        );
        assert_eq!(cost.bound, ComputeBound::Memory);
    }

    #[test]
    fn test_roofline_serialization() {
        let cost = gemv_cost(128, 256, Dtype::F32, &test_hw());
        let json = serde_json::to_string(&cost).unwrap();
        assert!(json.contains("runtime_ms"));
        assert!(json.contains("Compute") || json.contains("Memory") || json.contains("Launch"));
    }
}
