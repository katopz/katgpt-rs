//! NPC Brain Auto-Route — selects optimal backend for NPC sense projection (Plan 255 Part 4).
//!
//! Routing logic:
//! - <100 NPCs → `CpuTernaryBackend` (SIMD, zero overhead)
//! - ≥100 NPCs with ANE resident → `AneNpcBrainBackend` (batch dispatch)
//! - Fallback → `CpuTernaryBackend` (always available)
//!
//! This is separate from `TriggerGate` (which routes general inference by QPS).
//! NPC brain routing is driven by NPC count, not query throughput.

use std::path::Path;

use katgpt_core::sense::backend::{
    CpuTernaryBackend, NpcBrainBackend, NpcBrainInput, NpcBrainOutput,
};

/// Minimum NPC count to consider ANE batching.
/// Below this threshold, the fixed dispatch overhead (~95µs) exceeds
/// the SIMD cost (75ns × npc_count).
const ANE_BATCH_THRESHOLD: usize = 100;

// ---------------------------------------------------------------------------
// BackendChoice — pure routing decision
// ---------------------------------------------------------------------------

/// Which backend the router would select for a given NPC count.
///
/// Pure function — no backend construction needed. Useful for testing
/// routing logic and logging decisions without loading CoreML models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    /// CPU SIMD baseline (always available).
    CpuSimd,
    /// ANE batch dispatch (macOS, feature `ane_npc`).
    AneBatch,
}

impl BackendChoice {
    /// Route for a given NPC count and ANE availability.
    ///
    /// ```
    /// ANE_BATCH_THRESHOLD=100:
    /// - 0 NPCs  → CpuSimd
    /// - 50 NPCs → CpuSimd
    /// - 99 NPCs → CpuSimd
    /// - 100 NPCs → AneBatch (if ANE available)
    /// - 1000 NPCs → AneBatch (if ANE available)
    /// - 1000 NPCs → CpuSimd (if ANE not available)
    /// ```
    pub fn route_for_count(npc_count: usize, ane_available: bool) -> Self {
        if npc_count >= ANE_BATCH_THRESHOLD && ane_available {
            BackendChoice::AneBatch
        } else {
            BackendChoice::CpuSimd
        }
    }
}

impl std::fmt::Display for BackendChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CpuSimd => write!(f, "cpu_simd"),
            Self::AneBatch => write!(f, "ane_batch"),
        }
    }
}

// ---------------------------------------------------------------------------
// NpcBrainRouter — enum-dispatched backend wrapper
// ---------------------------------------------------------------------------

/// Routed NPC brain backend — auto-selects between CPU and ANE.
///
/// Uses enum dispatch (not trait objects) for zero-overhead delegation.
/// The ANE variant is only present when compiled with `ane_npc` on macOS.
pub enum NpcBrainRouter {
    /// CPU SIMD baseline (always available).
    Cpu(CpuTernaryBackend),
    /// ANE batch dispatch (macOS, feature `ane_npc`).
    #[cfg(all(feature = "ane_npc", target_os = "macos"))]
    Ane(crate::npc_ane_backend::AneNpcBrainBackend),
}

impl NpcBrainRouter {
    /// Create a new router, attempting ANE if a model path is provided.
    ///
    /// - If `ane_model_path` is `None`, always uses CPU.
    /// - If `ane_model_path` is `Some(path)`:
    ///   - Tries to load the ANE model and validate residency.
    ///   - On any failure (file not found, ANE not resident, etc.), falls back to CPU.
    ///   - Logs which backend was selected and why.
    pub fn new(ane_model_path: Option<&Path>) -> Self {
        match ane_model_path {
            None => {
                log::info!("NpcBrainRouter: CPU backend (no ANE model path provided)");
                Self::cpu()
            }
            Some(path) => Self::try_ane(path),
        }
    }

    /// Create a CPU-only router.
    pub fn cpu() -> Self {
        NpcBrainRouter::Cpu(CpuTernaryBackend::new())
    }

    /// Try to create an ANE-backed router, falling back to CPU on failure.
    #[cfg(all(feature = "ane_npc", target_os = "macos"))]
    fn try_ane(path: &Path) -> Self {
        use crate::npc_ane_backend::AneNpcBrainBackend;

        match AneNpcBrainBackend::new(path, 1024) {
            Ok(backend) => {
                log::info!("NpcBrainRouter: ANE backend selected (model loaded, residency OK)");
                NpcBrainRouter::Ane(backend)
            }
            Err(e) => {
                log::info!("NpcBrainRouter: CPU fallback (ANE failed: {e})");
                Self::cpu()
            }
        }
    }

    /// Try to create an ANE-backed router — CPU-only stub when `ane_npc` is disabled.
    #[cfg(not(all(feature = "ane_npc", target_os = "macos")))]
    fn try_ane(path: &Path) -> Self {
        log::info!(
            "NpcBrainRouter: CPU backend (ANE not available, path={})",
            path.display()
        );
        Self::cpu()
    }

    /// Which backend choice would be selected for the given NPC count.
    ///
    /// This is a static routing decision — it doesn't consider whether the ANE
    /// model is actually loaded (use `is_ane()` for runtime check).
    pub fn choice_for_count(&self, npc_count: usize) -> BackendChoice {
        BackendChoice::route_for_count(npc_count, self.is_ane())
    }

    /// Whether this router is using ANE.
    pub fn is_ane(&self) -> bool {
        match self {
            NpcBrainRouter::Cpu(_) => false,
            #[cfg(all(feature = "ane_npc", target_os = "macos"))]
            NpcBrainRouter::Ane(_) => true,
        }
    }
}

impl NpcBrainBackend for NpcBrainRouter {
    fn batch_evaluate(
        &mut self,
        inputs: &[NpcBrainInput],
        outputs: &mut [NpcBrainOutput],
    ) -> Result<(), String> {
        match self {
            NpcBrainRouter::Cpu(cpu) => cpu.batch_evaluate(inputs, outputs),
            #[cfg(all(feature = "ane_npc", target_os = "macos"))]
            NpcBrainRouter::Ane(ane) => ane.batch_evaluate(inputs, outputs),
        }
    }

    fn backend_name(&self) -> &'static str {
        match self {
            NpcBrainRouter::Cpu(cpu) => cpu.backend_name(),
            #[cfg(all(feature = "ane_npc", target_os = "macos"))]
            NpcBrainRouter::Ane(ane) => ane.backend_name(),
        }
    }

    fn optimal_batch_size(&self) -> usize {
        match self {
            NpcBrainRouter::Cpu(cpu) => cpu.optimal_batch_size(),
            #[cfg(all(feature = "ane_npc", target_os = "macos"))]
            NpcBrainRouter::Ane(ane) => ane.optimal_batch_size(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── BackendChoice routing tests ──────────────────────────────────────

    #[test]
    fn test_route_for_count_zero_npcs() {
        assert_eq!(
            BackendChoice::route_for_count(0, true),
            BackendChoice::CpuSimd
        );
    }

    #[test]
    fn test_route_for_count_below_threshold() {
        assert_eq!(
            BackendChoice::route_for_count(50, true),
            BackendChoice::CpuSimd
        );
        assert_eq!(
            BackendChoice::route_for_count(99, true),
            BackendChoice::CpuSimd
        );
    }

    #[test]
    fn test_route_for_count_at_threshold_ane_available() {
        assert_eq!(
            BackendChoice::route_for_count(100, true),
            BackendChoice::AneBatch
        );
    }

    #[test]
    fn test_route_for_count_above_threshold_ane_available() {
        assert_eq!(
            BackendChoice::route_for_count(1000, true),
            BackendChoice::AneBatch
        );
    }

    #[test]
    fn test_route_for_count_above_threshold_ane_not_available() {
        assert_eq!(
            BackendChoice::route_for_count(100, false),
            BackendChoice::CpuSimd
        );
        assert_eq!(
            BackendChoice::route_for_count(1000, false),
            BackendChoice::CpuSimd
        );
    }

    #[test]
    fn test_route_for_count_exact_threshold_boundary() {
        // 99 → CPU (just below)
        assert_eq!(
            BackendChoice::route_for_count(99, true),
            BackendChoice::CpuSimd
        );
        // 100 → ANE (exactly at threshold)
        assert_eq!(
            BackendChoice::route_for_count(100, true),
            BackendChoice::AneBatch
        );
    }

    // ── BackendChoice Display ────────────────────────────────────────────

    #[test]
    fn test_backend_choice_display() {
        assert_eq!(BackendChoice::CpuSimd.to_string(), "cpu_simd");
        assert_eq!(BackendChoice::AneBatch.to_string(), "ane_batch");
    }

    // ── NpcBrainRouter construction tests ────────────────────────────────

    #[test]
    fn test_router_new_none_creates_cpu() {
        let router = NpcBrainRouter::new(None);
        assert!(!router.is_ane());
        assert_eq!(router.backend_name(), "cpu_ternary");
    }

    #[test]
    fn test_router_cpu_factory() {
        let router = NpcBrainRouter::cpu();
        assert!(!router.is_ane());
        assert_eq!(router.backend_name(), "cpu_ternary");
    }

    #[test]
    fn test_router_new_invalid_path_falls_back_to_cpu() {
        let router = NpcBrainRouter::new(Some(Path::new("/nonexistent/mlmodelc")));
        assert!(!router.is_ane());
        assert_eq!(router.backend_name(), "cpu_ternary");
    }

    // ── NpcBrainRouter delegation tests ──────────────────────────────────

    #[test]
    fn test_router_batch_evaluate_cpu_path() {
        let mut router = NpcBrainRouter::cpu();
        let inputs = vec![NpcBrainInput::default()];
        let mut outputs = vec![NpcBrainOutput::default()];

        let result = router.batch_evaluate(&inputs, &mut outputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_router_batch_evaluate_length_mismatch() {
        let mut router = NpcBrainRouter::cpu();
        let inputs = vec![NpcBrainInput::default(); 2];
        let mut outputs = vec![NpcBrainOutput::default(); 3];

        let result = router.batch_evaluate(&inputs, &mut outputs);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("mismatch"));
    }

    #[test]
    fn test_router_optimal_batch_size_cpu() {
        let router = NpcBrainRouter::cpu();
        assert_eq!(router.optimal_batch_size(), 1);
    }

    #[test]
    fn test_router_is_ane_cpu() {
        let router = NpcBrainRouter::cpu();
        assert!(!router.is_ane());
    }

    #[test]
    fn test_router_choice_for_count_cpu_router() {
        let router = NpcBrainRouter::cpu();
        // CPU router always routes to CPU regardless of count
        assert_eq!(router.choice_for_count(1000), BackendChoice::CpuSimd);
    }

    // ── ANE batch threshold constant ────────────────────────────────────

    #[test]
    fn test_ane_batch_threshold_value() {
        assert_eq!(ANE_BATCH_THRESHOLD, 100);
    }
}
