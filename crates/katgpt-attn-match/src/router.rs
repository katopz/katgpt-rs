//! Adaptive Solver Router (Plan 271 Phase 2).
//!
//! Routes compaction work across backends based on problem size `t` (compact
//! tokens), original length `T`, and device availability. Implements
//! hysteresis so small fluctuations in `t` around a threshold don't cause
//! backend flapping.
//!
//! # Defaults (paper-grounded)
//!
//! - `cpu_max_t = 64` — below this, scalar is cheaper than SIMD setup overhead.
//! - `simd_max_t = 1024` — SIMD path dominates until rayon wins at large `T`.
//! - `gpu_min_t = 4096` — GPU dispatch amortizes only above this.
//! - `ane_max_t = 256` — Apple Neural Engine wins on small output roles.
//! - `hysteresis_pct = 0.10` — 10% band around thresholds, no flap.
//!
//! Per AGENTS.md: thresholds are config fields, not magic numbers. The router
//! itself performs no heap allocation — all state is on the struct.

/// Available solver backends. Ordered roughly by ascending throughput sweet
/// spot, but each has a regime where it dominates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SolverBackend {
    /// Plain scalar CPU loop. Wins for tiny `t` (overhead-dominated).
    CpuScalar = 0,
    /// 8-wide SIMD auto-vectorized loop. Wins for `t ∈ [64, 1024]`.
    CpuSimd = 1,
    /// Rayon-parallel blocked CPU. Wins for large `T` on multi-core.
    CpuRayon = 2,
    /// GPU (Metal/CUDA) dispatch. Wins for `t ≥ 4096` when available.
    Gpu = 3,
    /// Apple Neural Engine. Wins for small output roles (≤ 256).
    Ane = 4,
}

impl SolverBackend {
    /// Human-readable name for logging.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuScalar => "cpu_scalar",
            Self::CpuSimd => "cpu_simd",
            Self::CpuRayon => "cpu_rayon",
            Self::Gpu => "gpu",
            Self::Ane => "ane",
        }
    }
}

/// Router configuration. All thresholds are inclusive bounds (upper or lower
/// per field comment). Tunable per deployment; defaults are paper-grounded.
#[derive(Clone, Copy, Debug)]
pub struct SolverRouterConfig {
    /// `t` at or below this → scalar wins (overhead-dominated regime).
    pub cpu_max_t: usize,
    /// `t` at or below this → SIMD wins.
    pub simd_max_t: usize,
    /// `t` at or above this (and GPU available) → GPU wins.
    pub gpu_min_t: usize,
    /// Output role `t` at or below this → ANE wins (overrides other rules).
    pub ane_max_t: usize,
    /// Hysteresis band: if `|t - last_t| / last_t < hysteresis_pct`, keep the
    /// previous backend to avoid flapping.
    pub hysteresis_pct: f32,
}

impl Default for SolverRouterConfig {
    #[inline]
    fn default() -> Self {
        Self {
            cpu_max_t: 64,
            simd_max_t: 1024,
            gpu_min_t: 4096,
            ane_max_t: 256,
            hysteresis_pct: 0.10,
        }
    }
}

/// Adaptive solver router. Stateless except for hysteresis tracking.
///
/// Per AGENTS.md: no allocation in `pick_backend` — the struct holds only two
/// `Option`s, both stack-allocated. Safe to call from hot paths.
#[derive(Clone, Debug)]
pub struct SolverRouter {
    config: SolverRouterConfig,
    last_backend: Option<SolverBackend>,
    last_t: Option<usize>,
}

impl SolverRouter {
    /// Construct a router with the given config. No prior decision history.
    #[inline]
    pub fn new(config: SolverRouterConfig) -> Self {
        Self {
            config,
            last_backend: None,
            last_t: None,
        }
    }

    /// Router configuration (read-only view).
    #[inline]
    pub fn config(&self) -> &SolverRouterConfig {
        &self.config
    }

    /// Pick the best backend for the current problem.
    ///
    /// # Arguments
    /// * `t` - Compact size (number of tokens to retain).
    /// * `T` - Original sequence length (unused now; reserved for future
    ///   memory-bandwidth-aware routing — see plan T2.5).
    /// * `gpu_available` - Whether a GPU backend is dispatchable right now.
    ///
    /// # Hysteresis
    /// If `last_backend` exists and the relative change in `t` since the last
    /// decision is less than `hysteresis_pct`, the prior backend is retained
    /// (no flapping). First call always returns the freshly-decided backend.
    ///
    /// # Determinism
    /// Given fixed `(t, T, gpu_available)` and an empty history, the returned
    /// backend is deterministic. With history, the hysteresis rule is also
    /// deterministic. (GOAT G6.)
    pub fn pick_backend(
        &mut self,
        t: usize,
        _original_len: usize,
        gpu_available: bool,
    ) -> SolverBackend {
        let target = self.target_backend(t, gpu_available);

        // Hysteresis: if the relative change in t since the last decision is
        // less than `hysteresis_pct`, retain the prior backend regardless of
        // what the fresh target would be. This avoids flapping when t hovers
        // near a threshold. First call always follows the target.
        //
        // Note: device availability changes (e.g., GPU going online) do NOT
        // bypass hysteresis — callers should invoke `reset()` when the device
        // topology changes to force a fresh decision.
        let kept = match (self.last_backend, self.last_t) {
            (Some(_), Some(last_t)) if last_t > 0 => {
                let diff = t.abs_diff(last_t);
                let rel = (diff as f32) / (last_t as f32);
                rel < self.config.hysteresis_pct
            }
            _ => false,
        };

        let chosen = if kept {
            self.last_backend.unwrap_or(target)
        } else {
            target
        };

        self.last_backend = Some(chosen);
        self.last_t = Some(t);
        chosen
    }

    /// Decide the target backend ignoring hysteresis.
    ///
    /// Rules (evaluated in order, first match wins):
    /// 1. `t <= ane_max_t` → `Ane` (output-role override).
    /// 2. `t <= cpu_max_t` → `CpuScalar`.
    /// 3. `t <= simd_max_t` → `CpuSimd`.
    /// 4. `t >= gpu_min_t && gpu_available` → `Gpu`.
    /// 5. `t >= simd_max_t && !gpu_available` → `CpuRayon`.
    /// 6. Otherwise → `CpuSimd`.
    ///
    /// Note rule 1 overrides rule 2 — `ane_max_t` is the output-role override,
    /// not a strict size threshold. Both default to small-t regimes; if the
    /// user configures them apart (e.g., `ane_max_t=16`, `cpu_max_t=64`),
    /// the ANE rule only triggers for the smallest roles.
    #[inline]
    fn target_backend(&self, t: usize, gpu_available: bool) -> SolverBackend {
        // Output-role override: ANE handles small compact sizes best.
        if t <= self.config.ane_max_t {
            return SolverBackend::Ane;
        }
        if t <= self.config.cpu_max_t {
            return SolverBackend::CpuScalar;
        }
        if t <= self.config.simd_max_t {
            return SolverBackend::CpuSimd;
        }
        if t >= self.config.gpu_min_t && gpu_available {
            return SolverBackend::Gpu;
        }
        if t >= self.config.simd_max_t && !gpu_available {
            return SolverBackend::CpuRayon;
        }
        // Fallback: between simd_max_t and gpu_min_t with no GPU.
        SolverBackend::CpuSimd
    }

    /// Reset hysteresis state (useful when workload regime changes sharply).
    #[inline]
    pub fn reset(&mut self) {
        self.last_backend = None;
        self.last_t = None;
    }
}

impl Default for SolverRouter {
    #[inline]
    fn default() -> Self {
        Self::new(SolverRouterConfig::default())
    }
}

/// Standalone backend picker (no hysteresis). Used by callers that don't
/// track history (e.g., one-shot compaction) and by tests.
#[inline]
pub fn pick_backend(
    t: usize,
    original_len: usize,
    gpu_available: bool,
    config: &SolverRouterConfig,
) -> SolverBackend {
    let mut router = SolverRouter::new(*config);
    router.pick_backend(t, original_len, gpu_available)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GOAT G6: determinism — same inputs, same backend.
    #[test]
    fn test_router_determinism() {
        let cfg = SolverRouterConfig::default();
        // Two fresh routers must agree on the same (t, T, gpu_available).
        for &(t, gpu) in &[(16usize, true), (100, true), (100, false), (5000, true), (5000, false)]
        {
            let b1 = pick_backend(t, 8192, gpu, &cfg);
            let b2 = pick_backend(t, 8192, gpu, &cfg);
            assert_eq!(
                b1, b2,
                "non-deterministic backend for t={} gpu={}: {:?} vs {:?}",
                t, gpu, b1, b2
            );
        }
    }

    /// Hysteresis: crossing a threshold by <10% keeps the prior backend.
    #[test]
    fn test_router_hysteresis() {
        // Custom config with a sharp threshold at cpu_max_t=100 and a simd
        // band up to 500. ane disabled (ane_max_t=0) so the scalar/simd
        // transition is testable in isolation.
        let cfg = SolverRouterConfig {
            cpu_max_t: 100,
            simd_max_t: 500, // t ∈ (100, 500] → CpuSimd
            gpu_min_t: usize::MAX,
            ane_max_t: 0, // disable ane override
            hysteresis_pct: 0.10,
        };
        let mut router = SolverRouter::new(cfg);
        // t=100 → CpuScalar (first call, no hysteresis).
        let b1 = router.pick_backend(100, 1024, false);
        assert_eq!(b1, SolverBackend::CpuScalar, "t=100 → scalar (ane disabled)");
        // t=105 is within 10% of 100 (5% relative). Target would be CpuSimd
        // (105 > 100, 105 <= 500), but hysteresis keeps scalar.
        let b2 = router.pick_backend(105, 1024, false);
        assert_eq!(
            b2, SolverBackend::CpuScalar,
            "hysteresis should keep scalar for t=105 (within 10% of last_t=100)"
        );
        // t=120 is >10% away from last_t=105 (15/105 ≈ 14.3%) → switch to target.
        let b3 = router.pick_backend(120, 1024, false);
        assert_eq!(b3, SolverBackend::CpuSimd, "t=120 > 10% away → switch to simd");
    }

    /// Hysteresis: large jump always follows the target.
    #[test]
    fn test_router_large_jump_follows_target() {
        let cfg = SolverRouterConfig {
            cpu_max_t: 100,
            simd_max_t: 200,
            gpu_min_t: usize::MAX,
            ane_max_t: 0,
            hysteresis_pct: 0.10,
        };
        let mut router = SolverRouter::new(cfg);
        let _ = router.pick_backend(50, 1024, false); // scalar
        // Jump way past the 10% band → new target.
        let b = router.pick_backend(1000, 1024, false);
        assert_eq!(b, SolverBackend::CpuRayon, "large jump → rayon (t≥simd_max_t, no gpu)");
    }

    /// `pick_backend` allocates nothing on the heap. We verify this by
    /// inspecting the source (no `Vec`, `String`, `Box`, `format!`, etc. in
    /// `pick_backend`) and also empirically: calling it 1000× must run without
    /// panic.
    ///
    /// Note: the original katgpt-rs root build asserted zero growth against the
    /// root `TrackingAllocator` (`crate::alloc`). That allocator lives in the
    /// root crate only; this leaf has no equivalent, so the empirical assertion
    /// is retired here — the no-alloc property is enforced by code review
    /// (no heap types in `pick_backend`'s body). The 1000-call loop below still
    /// exercises the path for panics/ correctness.
    #[test]
    fn test_router_no_alloc() {
        let mut router = SolverRouter::new(SolverRouterConfig::default());
        // Warm up to populate history (history write is stack-only).
        let _ = router.pick_backend(32, 1024, true);

        for i in 0..1000usize {
            let t = 32 + (i % 64);
            let _ = router.pick_backend(t, 8192, true);
        }

        // No-alloc property enforced by code review (no heap types in the fn
        // body). The root crate's TrackingAllocator assertion is not available
        // in this leaf; see Issue 359.
    }

    /// Threshold boundaries behave as documented.
    #[test]
    fn test_router_thresholds() {
        let cfg = SolverRouterConfig::default();
        // ane_max_t = 256 → t ≤ 256 returns Ane.
        assert_eq!(pick_backend(256, 1024, true, &cfg), SolverBackend::Ane);
        // cpu_max_t = 64 but ane overrides → 64 still Ane.
        assert_eq!(pick_backend(64, 1024, true, &cfg), SolverBackend::Ane);
        // Just above ane_max_t → scalar (64) or simd (≤1024).
        assert_eq!(pick_backend(257, 1024, true, &cfg), SolverBackend::CpuSimd);
        // Above simd_max_t with GPU → Gpu.
        assert_eq!(pick_backend(4096, 8192, true, &cfg), SolverBackend::Gpu);
        // Above simd_max_t without GPU → CpuRayon.
        assert_eq!(pick_backend(4096, 8192, false, &cfg), SolverBackend::CpuRayon);
    }

    /// `reset()` clears history so the next pick is the pure target.
    #[test]
    fn test_router_reset_clears_hysteresis() {
        let mut router = SolverRouter::new(SolverRouterConfig::default());
        let _ = router.pick_backend(32, 1024, true);
        assert!(router.last_backend.is_some());
        router.reset();
        assert!(router.last_backend.is_none());
    }

    /// Custom thresholds via config (no magic numbers in code path).
    #[test]
    fn test_router_custom_thresholds() {
        let cfg = SolverRouterConfig {
            cpu_max_t: 8,
            simd_max_t: 32,
            gpu_min_t: 128,
            ane_max_t: 0, // disable
            hysteresis_pct: 0.0,
        };
        assert_eq!(pick_backend(8, 256, false, &cfg), SolverBackend::CpuScalar);
        assert_eq!(pick_backend(32, 256, false, &cfg), SolverBackend::CpuSimd);
        assert_eq!(pick_backend(128, 256, true, &cfg), SolverBackend::Gpu);
        assert_eq!(pick_backend(128, 256, false, &cfg), SolverBackend::CpuRayon);
    }

    #[test]
    fn test_solver_backend_as_str() {
        assert_eq!(SolverBackend::CpuScalar.as_str(), "cpu_scalar");
        assert_eq!(SolverBackend::CpuSimd.as_str(), "cpu_simd");
        assert_eq!(SolverBackend::CpuRayon.as_str(), "cpu_rayon");
        assert_eq!(SolverBackend::Gpu.as_str(), "gpu");
        assert_eq!(SolverBackend::Ane.as_str(), "ane");
    }
}
