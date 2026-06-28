//! DEC computation backend selection (T31–T33).
//!
//! Adaptive routing selects the best backend (CPU, SIMD, GPU) based on
//! cochain size. Thresholds are tuned for typical game-workload sizes:
//!
//! | Size range | Backend    | Rationale                              |
//! |------------|------------|----------------------------------------|
//! | n < 1K     | CPU        | Overhead not worth SIMD/GPU dispatch   |
//! | 1K–10K     | SIMD       | Auto-vectorized sparse matvec          |
//! | > 10K      | GPU        | Massively parallel (when available)    |

// ---------------------------------------------------------------------------
// Backend Enum (T31)
// ---------------------------------------------------------------------------

/// DEC computation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DecBackend {
    /// Scalar CPU — no SIMD, fallback for very small problems.
    Cpu,
    /// SIMD-accelerated — auto-vectorized sparse matvec.
    Simd,
    /// GPU compute — wgpu/cubecl sparse kernels (future).
    Gpu,
}

// ---------------------------------------------------------------------------
// Backend Selection (T32)
// ---------------------------------------------------------------------------

/// Threshold: below this cell count, use plain CPU.
const CPU_THRESHOLD: usize = 1_000;
/// Threshold: above this cell count, prefer GPU (if available).
const GPU_THRESHOLD: usize = 10_000;

/// Select backend based on cochain size.
///
/// Thresholds:
/// - `n < 1K` cells → CPU (overhead not worth it)
/// - `1K ≤ n ≤ 10K` → SIMD
/// - `n > 10K` → GPU (when available, else SIMD)
#[inline]
pub fn select_backend(n_cells: usize, gpu_available: bool) -> DecBackend {
    match (n_cells, gpu_available) {
        (n, _) if n < CPU_THRESHOLD => DecBackend::Cpu,
        (n, true) if n > GPU_THRESHOLD => DecBackend::Gpu,
        _ => DecBackend::Simd,
    }
}

// ---------------------------------------------------------------------------
// Tests (T33)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_threshold_small() {
        assert_eq!(select_backend(0, false), DecBackend::Cpu);
        assert_eq!(select_backend(1, false), DecBackend::Cpu);
        assert_eq!(select_backend(999, false), DecBackend::Cpu);
    }

    #[test]
    fn simd_range() {
        assert_eq!(select_backend(1_000, false), DecBackend::Simd);
        assert_eq!(select_backend(5_000, false), DecBackend::Simd);
        assert_eq!(select_backend(10_000, false), DecBackend::Simd);
    }

    #[test]
    fn gpu_threshold() {
        // GPU only when available AND above threshold
        assert_eq!(select_backend(10_001, true), DecBackend::Gpu);
        assert_eq!(select_backend(100_000, true), DecBackend::Gpu);
        // Falls back to SIMD when GPU not available
        assert_eq!(select_backend(100_000, false), DecBackend::Simd);
    }

    #[test]
    fn gpu_not_available_falls_to_simd() {
        assert_eq!(select_backend(50_000, false), DecBackend::Simd);
    }

    #[test]
    fn bench_backend_selection_overhead() {
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = select_backend(5_000, false);
        }
        let overhead = start.elapsed() / 1000;
        println!("Backend selection overhead: {:?}", overhead);
        assert!(
            overhead.as_micros() < 100,
            "Selection overhead too high: {:?}",
            overhead
        );
    }
}
