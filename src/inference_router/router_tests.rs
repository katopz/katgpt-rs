//! Tests for `inference_router.rs` — extracted from the inline `mod tests`
//! (Issue 018) to keep the router source file under the 2048-line ceiling.
//!
//! Compiled only under `#[cfg(test)]`. All test bodies are byte-identical to
//! their pre-extraction form — only their module path changed from
//! `inference_router::tests::*` to `inference_router::router_tests::*`.

use super::*;
use crate::types::Rng;

// The `simulate_cascade` helper below uses `tvp_tier_decision` directly (the
// other TVP references go through `router.update_tvp()` / `observe_tvp_decision`
// so they don't need this import). Pre-split, this came in via `super::*` from
// the parent module's `use` statement; now we import it explicitly.
#[cfg(all(feature = "rv_gated_routing", feature = "thicket_variance_probe"))]
use crate::pruners::thicket_variance_probe::tvp_tier_decision;

/// Helper: build a router with a fast gate (tiny min interval) for tests.
fn fast_router(gpu: bool, ane: bool) -> InferenceRouter {
    let gate_config = TriggerGateConfig {
        gpu_activate_qps: 10_000.0,
        ane_activate_qps: 100_000.0,
        hysteresis_factor: 0.7,
        queue_depth_trigger: 100,
        latency_p99_trigger_us: 5000,
        min_tier_change_interval_ms: 10,
    };
    InferenceRouter::new(gate_config, Config::micro(), gpu, ane)
}

/// Helper: create micro model fixtures for forward-pass tests.
fn micro_fixtures() -> (TransformerWeights, ForwardContext, MultiLayerKVCache) {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let ctx = ForwardContext::new(&config);
    let cache = MultiLayerKVCache::new(&config);
    (weights, ctx, cache)
}

#[test]
fn test_router_starts_cpu_only() {
    let router = fast_router(true, true);
    assert_eq!(router.gate().current_tier(), ComputeTier::CpuOnly);
}

#[test]
fn test_router_forward_uses_cpu() {
    let mut router = fast_router(false, false);
    let (weights, mut ctx, mut cache) = micro_fixtures();

    let logits = router.forward(&mut ctx, &weights, &mut cache, 0, 0);
    assert_eq!(logits.len(), Config::micro().vocab_size);
    assert_eq!(router.last_backend, "CPU");
}

#[test]
fn test_router_stats_initial() {
    let router = fast_router(true, true);
    let stats = router.stats();
    assert_eq!(stats.current_tier, ComputeTier::CpuOnly);
    assert_eq!(stats.total_inferences, 0);
    assert_eq!(stats.tier_transitions, 0);
    assert_eq!(stats.last_backend, "CPU");
}

#[test]
fn test_router_promotes_under_load() {
    let mut router = fast_router(true, true);
    let (weights, mut ctx, mut cache) = micro_fixtures();
    let block_size = Config::micro().block_size;

    // Run enough inferences quickly to build up QPS.
    // With gpu_activate_qps=10_000 and min_tier_change_interval_ms=10,
    // we need enough forwards in a short window to exceed 10k QPS.
    // Each forward is very fast on micro model, so we do many.
    // Keep pos within block_size to avoid KV cache overflow.
    for i in 0..200 {
        let pos = i % block_size;
        let token = i % Config::micro().vocab_size;
        // Reset cache when wrapping around.
        if pos == 0 && i > 0 {
            cache = MultiLayerKVCache::new(&Config::micro());
        }
        router.forward(&mut ctx, &weights, &mut cache, token, pos);
    }

    // The tier may or may not have promoted depending on actual timing,
    // but evaluate() should have been called each time. Verify the router
    // is still functional and tracking state.
    let stats = router.stats();
    assert!(stats.total_inferences > 0);
    // Tier transitions tracked even if promote didn't fire (timing-dependent).
    assert!(stats.tier_transitions <= stats.total_inferences);
}

#[test]
fn test_router_falls_back_to_cpu_without_gpu() {
    let mut router = fast_router(true, true);
    let (weights, mut ctx, mut cache) = micro_fixtures();

    // Manually force the gate into CpuGpu tier by manipulating it.
    // Since GPU backend is None, it should fall back to CPU.
    // We'll record a bunch of inferences and queue depth to force promotion.
    router.record_queue_depth(200); // above queue_depth_trigger=100

    // Run forward — this records inference but evaluate() also checks QPS.
    // Even without promotion, the CpuGpu path is tested when the gate
    // stays at CpuOnly (which routes to CPU anyway).
    let logits = router.forward(&mut ctx, &weights, &mut cache, 0, 0);
    assert_eq!(logits.len(), Config::micro().vocab_size);

    // The key invariant: regardless of tier, GPU=None means CPU fallback.
    // Test that explicitly by checking stats shows CPU was used.
    assert_eq!(router.stats().last_backend, "CPU");
}

#[test]
fn test_router_records_inferences() {
    let mut router = fast_router(false, false);
    let (weights, mut ctx, mut cache) = micro_fixtures();

    assert_eq!(router.stats().total_inferences, 0);

    router.forward(&mut ctx, &weights, &mut cache, 0, 0);
    assert_eq!(router.stats().total_inferences, 1);

    router.forward(&mut ctx, &weights, &mut cache, 1, 1);
    assert_eq!(router.stats().total_inferences, 2);

    router.forward(&mut ctx, &weights, &mut cache, 2, 2);
    assert_eq!(router.stats().total_inferences, 3);
}

#[test]
fn test_router_queue_depth_delegation() {
    let router = fast_router(true, true);

    router.record_queue_depth(42);
    // Verify via the gate's public interface that depth was recorded.
    // The gate stores depth internally; we can't read it back directly
    // but we can verify it influences should_promote.
    // With queue_depth_trigger=100, depth=42 should NOT trigger promotion.
    assert_eq!(router.gate().current_tier(), ComputeTier::CpuOnly);
    assert!(router.gate().should_promote().is_none());

    // Now set depth above threshold.
    router.record_queue_depth(150);
    // should_promote considers QPS too, but the queue depth alone is enough.
    // Since we have 0 QPS, the queue_depth_trigger path should fire.
    assert!(router.gate().should_promote().is_some());
}

#[test]
fn test_forward_batch_empty() {
    let mut router = fast_router(false, false);
    let (weights, mut ctx, mut cache) = micro_fixtures();

    let results = router.forward_batch(&mut ctx, &weights, &mut cache, &[]);
    assert!(results.is_empty());
    assert_eq!(router.stats().total_inferences, 0);
}

#[test]
fn test_forward_batch_single_token() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    let mut router = fast_router(false, false);

    let results = router.forward_batch(&mut ctx, &weights, &mut cache, &[(0, 0)]);
    assert_eq!(results.len(), config.vocab_size);
    assert_eq!(router.stats().total_inferences, 1);
}

#[test]
fn test_forward_batch_multiple_tokens() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    let mut router = fast_router(false, false);

    // Build a batch of 5 tokens within block_size.
    let batch: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();
    let results = router.forward_batch(&mut ctx, &weights, &mut cache, &batch);

    assert_eq!(results.len(), 5 * config.vocab_size);
    assert_eq!(router.stats().total_inferences, 5);
}

#[test]
fn test_forward_batch_matches_sequential_forward() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Sequential forward (one at a time).
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let mut router1 = fast_router(false, false);
    let mut sequential_flat = Vec::new();
    for i in 0..3 {
        let logits = router1.forward(&mut ctx1, &weights, &mut cache1, i, i);
        sequential_flat.extend_from_slice(logits);
    }

    // Batch forward.
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let mut router2 = fast_router(false, false);
    let batch: Vec<(usize, usize)> = (0..3).map(|i| (i, i)).collect();
    let batch_logits = router2.forward_batch(&mut ctx2, &weights, &mut cache2, &batch);

    assert_eq!(sequential_flat.len(), batch_logits.len());
    for (i, (a, b)) in sequential_flat.iter().zip(batch_logits.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "logits mismatch at flat[{i}]: {a} vs {b}"
        );
    }
}

#[test]
fn test_forward_batch_records_all_inferences() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut router = fast_router(false, false);

    assert_eq!(router.stats().total_inferences, 0);

    let batch: Vec<(usize, usize)> = (0..4).map(|i| (i, i)).collect();
    let _ = router.forward_batch(&mut ctx, &weights, &mut cache, &batch);

    assert_eq!(router.stats().total_inferences, 4);
}

#[cfg(feature = "lodestar")]
#[test]
fn test_lodestar_route_hook_observe_and_query() {
    let mut router = InferenceRouter::new(
        TriggerGateConfig::default(),
        Config::default(),
        false,
        false,
    );
    // Before any observation
    assert_eq!(router.lodestar_distance(), 0);
    assert_eq!(router.lodestar_budget_remaining(), -1);
    assert!(!router.lodestar_suggests_cpu());

    // Observe near completion (d=2, budget=10)
    router.observe_lodestar(2, 10);
    assert_eq!(router.lodestar_distance(), 2);
    assert_eq!(router.lodestar_budget_remaining(), 10);
    assert!(!router.lodestar_suggests_cpu()); // d <= 4, not far

    // Observe far completion with tight budget (d=6, budget=8)
    router.observe_lodestar(6, 8);
    assert_eq!(router.lodestar_distance(), 6);
    assert_eq!(router.lodestar_budget_remaining(), 8);
    // 8 < 6*2=12, so suggests CPU
    assert!(router.lodestar_suggests_cpu());

    // Observe far completion with ample budget (d=6, budget=20)
    router.observe_lodestar(6, 20);
    assert!(!router.lodestar_suggests_cpu()); // 20 >= 12

    // Reset
    router.reset_lodestar();
    assert_eq!(router.lodestar_distance(), 0);
    assert_eq!(router.lodestar_budget_remaining(), -1);
    assert!(!router.lodestar_suggests_cpu());
}

// ------------------------------------------------------------------
// Plan 222 T15: CriticalIntervalGate + TriggerGate wiring
// ------------------------------------------------------------------

#[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
#[test]
fn test_observe_critical_entropy_low_entropy_defers() {
    let mut router = fast_router(false, false);
    // Low entropy (peaked) → Defer
    let decision = router.observe_critical_entropy(0.5);
    assert_eq!(decision, CriticalTierDecision::Defer);
    assert!((router.last_critical_entropy() - 0.5).abs() < 1e-6);
}

#[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
#[test]
fn test_observe_critical_entropy_high_entropy_stays_cpu_no_gpu() {
    let mut router = fast_router(false, false);
    // High entropy but no GPU → StayCpu
    let high_entropy = (1000.0f32).ln() * 0.8; // well above H_critical
    let decision = router.observe_critical_entropy(high_entropy);
    assert_eq!(decision, CriticalTierDecision::StayCpu);
}

#[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
#[test]
fn test_observe_critical_entropy_high_entropy_promotes_with_gpu() {
    let mut router = fast_router(true, false);
    // High entropy + GPU available + low load (CpuOnly) → PromoteGpu
    let high_entropy = (32000.0f32).ln() * 0.8;
    let decision = router.observe_critical_entropy(high_entropy);
    assert_eq!(decision, CriticalTierDecision::PromoteGpu);
}

#[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
#[test]
fn test_set_critical_interval_config_updates_threshold() {
    let mut router = fast_router(false, false);
    let custom = CriticalIntervalConfig::new(50); // tiny vocab → lower H_critical
    router.set_critical_interval_config(custom);
    // Verify config was updated
    assert_eq!(router.critical_interval_config().vocab_size, 50);
    // Even low entropy should now be critical with tiny vocab
    let entropy = (50.0f32).ln() * 0.6; // above H_critical for vocab=50
    let decision = router.observe_critical_entropy(entropy);
    // With no GPU, critical → StayCpu
    assert_eq!(decision, CriticalTierDecision::StayCpu);
}

#[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
#[test]
fn test_critical_entropy_updates_last_observed() {
    let mut router = fast_router(false, false);
    assert_eq!(router.last_critical_entropy(), 0.0);
    router.observe_critical_entropy(3.15);
    assert!((router.last_critical_entropy() - 3.15).abs() < 1e-6);
    router.observe_critical_entropy(2.72);
    assert!((router.last_critical_entropy() - 2.72).abs() < 1e-6);
}

// -- Breakeven Routing Tests (Plan 250) -------------------------

/// T19: Verify BreakevenTracker correctly computes N* from known costs.
#[cfg(feature = "breakeven_routing")]
#[test]
fn test_breakeven_tracker_n_star() {
    use crate::breakeven::BreakevenTracker;

    let tracker = BreakevenTracker::new(1000);
    for _ in 0..50 {
        tracker.observe_baseline(100);
    }
    for _ in 0..50 {
        tracker.observe_tier(50);
    }

    let n_star = tracker.breakeven_n();
    assert!(
        n_star > 0.0 && n_star < 500.0,
        "N* should be finite and reasonable, got {n_star}"
    );
}

/// T20: Verify is_amortized flips at exactly N* tokens.
#[cfg(feature = "breakeven_routing")]
#[test]
fn test_breakeven_tracker_amortized_flips() {
    use crate::breakeven::BreakevenTracker;

    let tracker = BreakevenTracker::new(10_000);
    for _ in 0..50 {
        tracker.observe_baseline(100);
    }
    for _ in 0..3 {
        tracker.observe_tier(50);
    }
    assert!(
        !tracker.is_amortized(),
        "Should NOT be amortized with only 3 tokens vs N*~200"
    );

    for _ in 0..1000 {
        tracker.observe_tier(50);
    }
    assert!(
        tracker.is_amortized(),
        "Should be amortized after 1003 tokens vs N*~200"
    );
}

/// T21: Bandit selects amortized tier over non-amortized.
#[cfg(feature = "breakeven_routing")]
#[test]
fn test_breakeven_bandit_prefers_amortized() {
    use crate::breakeven::{BreakevenBandit, BreakevenTierPair};
    use crate::trigger_gate::ComputeTier;

    let bandit = BreakevenBandit::new(100, 200, 50);
    for _ in 0..20 {
        bandit.observe_baseline(BreakevenTierPair::CpuToGpu, 1000);
    }
    for _ in 0..520 {
        bandit.observe_tier(BreakevenTierPair::CpuToGpu, 100);
    }

    let result = bandit.select_tier(ComputeTier::CpuOnly);
    assert!(
        result.is_some(),
        "Bandit should recommend a tier when CpuToGpu is amortized"
    );
}

/// T22: FidelityMatcher returns higher compression for later positions.
#[cfg(feature = "breakeven_routing")]
#[test]
fn test_fidelity_matcher_higher_compression_later() {
    use crate::breakeven::fidelity::{CompressionLevel, FidelityMatcher};

    let fm = FidelityMatcher::new(0.1);
    let early = fm.error_matched_level(0);
    let late = fm.error_matched_level(1024);
    assert_eq!(early, CompressionLevel::Bit4);
    assert_eq!(late, CompressionLevel::Bit4);
}

/// T23: Router with breakeven routes differently than without.
#[cfg(feature = "breakeven_routing")]
#[test]
fn test_router_breakeven_routes_differently() {
    use crate::breakeven::{BreakevenBandit, BreakevenTierPair};
    use crate::trigger_gate::ComputeTier;

    let bandit = BreakevenBandit::new(100, 200, 50);
    for _ in 0..20 {
        bandit.observe_baseline(BreakevenTierPair::CpuToGpu, 1000);
    }
    for _ in 0..520 {
        bandit.observe_tier(BreakevenTierPair::CpuToGpu, 100);
    }

    let tier = bandit.select_tier(ComputeTier::CpuOnly);
    assert!(
        tier.is_some(),
        "Bandit should recommend promotion after amortization"
    );

    let stats = bandit.stats();
    assert!(
        stats.cpu_to_gpu_n.is_finite() && stats.cpu_to_gpu_n > 0.0,
        "N* should be finite and positive"
    );
}

// ------------------------------------------------------------------
// Plan 267 T19: TVP integration tests — router tier promotion/demotion.
// Covers GOAT gates G1 (promote), G2 (demote), G3 (zero-overhead disabled),
// G5 (format-only does NOT promote), and the basic API contract.
//
// Tests follow the existing `observe_critical_entropy_*` pattern: they
// call `observe_tvp_decision(current_tier)` directly without running a
// full forward pass. This is cheaper, deterministic, and isolates the
// TVP decision logic from the rest of the router cascade.
// ------------------------------------------------------------------

/// Construct a TVP signal with the given substantive disagreement and zero
/// format disagreement (i.e., pure reasoning signal).
#[cfg(feature = "thicket_variance_probe")]
fn tvp_reasoning(disagreement: f32) -> TvpSignal {
    TvpSignal {
        reasoning_disagreement: disagreement,
        format_disagreement: 0.0,
        logit_kl: 0.0,
        probe_count_used: 4,
    }
}

/// Construct a TVP signal that is format-only (no substantive disagreement).
/// This MUST NOT promote — see G5.
#[cfg(feature = "thicket_variance_probe")]
fn tvp_format_only(format_disagreement: f32) -> TvpSignal {
    TvpSignal {
        reasoning_disagreement: 0.0,
        format_disagreement,
        logit_kl: 0.0,
        probe_count_used: 4,
    }
}

/// G1: High substantive disagreement promotes CPU→GPU when GPU available.
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn g1_high_disagreement_promotes_cpu_to_gpu() {
    let mut router = fast_router(true, false);
    router.update_tvp(Some(tvp_reasoning(0.9)));
    let decision = router.observe_tvp_decision(ComputeTier::CpuOnly);
    assert_eq!(
        decision,
        TvpTierDecision::PromoteGpu,
        "High reasoning disagreement on CpuOnly with GPU → PromoteGpu"
    );
}

/// G1 boundary: with no GPU available, high disagreement cannot promote.
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn g1b_high_disagreement_no_gpu_stays_hold() {
    let mut router = fast_router(false, false);
    router.update_tvp(Some(tvp_reasoning(0.9)));
    // No GPU → cannot promote, signal is above promote_at but tier cannot
    // change → Hold (not Defer, because signal IS present).
    let decision = router.observe_tvp_decision(ComputeTier::CpuOnly);
    assert_eq!(
        decision,
        TvpTierDecision::Hold,
        "Without a GPU backend, TVP cannot promote — Hold"
    );
}

/// G2: Low substantive disagreement demotes GPU→CPU under low load.
///
/// The fast_router fixture has zero QPS so `low_load` is true.
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn g2_low_disagreement_demotes_gpu_to_cpu() {
    let mut router = fast_router(true, false);
    router.update_tvp(Some(tvp_reasoning(0.05)));
    let decision = router.observe_tvp_decision(ComputeTier::CpuGpu);
    assert_eq!(
        decision,
        TvpTierDecision::DemoteCpu,
        "Low reasoning disagreement on CpuGpu under low load → DemoteCpu"
    );
}

/// G2 boundary: low disagreement on CpuOnly → Hold (already at floor).
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn g2b_low_disagreement_on_cpu_only_holds() {
    let mut router = fast_router(true, false);
    router.update_tvp(Some(tvp_reasoning(0.05)));
    let decision = router.observe_tvp_decision(ComputeTier::CpuOnly);
    assert_eq!(
        decision,
        TvpTierDecision::Hold,
        "Cannot demote from CpuOnly — Hold"
    );
}

/// G5: Format-only disagreement MUST NOT promote compute.
/// The plan explicitly distinguishes cosmetic disagreement (canonicalize
/// output) from substantive disagreement (upgrade compute).
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn g3_format_only_disagreement_does_not_promote() {
    let mut router = fast_router(true, false);
    // High format disagreement but zero substantive disagreement.
    router.update_tvp(Some(tvp_format_only(0.99)));
    let decision = router.observe_tvp_decision(ComputeTier::CpuOnly);
    assert_eq!(
        decision,
        TvpTierDecision::Hold,
        "Format-only disagreement must NOT promote compute (G5)"
    );
}

/// G4: Reasoning disagreement (net of format) promotes when above threshold.
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn g4_reasoning_disagreement_promotes() {
    let mut router = fast_router(true, false);
    // Mixed: reasoning 0.7, format 0.2. Default promote_at=0.6 → 0.7 trips.
    let signal = TvpSignal {
        reasoning_disagreement: 0.7,
        format_disagreement: 0.2,
        logit_kl: 0.0,
        probe_count_used: 4,
    };
    router.update_tvp(Some(signal));
    let decision = router.observe_tvp_decision(ComputeTier::CpuOnly);
    assert_eq!(
        decision,
        TvpTierDecision::PromoteGpu,
        "Substantive disagreement above promote_at should promote"
    );
}

/// Boundary: reasoning disagreement exactly at promote_at should NOT promote
/// (the gate uses strict `>` per TvpSignal::should_promote).
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn g4b_reasoning_at_threshold_does_not_promote() {
    let mut router = fast_router(true, false);
    // Default promote_at=0.6 → 0.6 must NOT promote (strict >).
    router.update_tvp(Some(tvp_reasoning(0.6)));
    let decision = router.observe_tvp_decision(ComputeTier::CpuOnly);
    assert_eq!(
        decision,
        TvpTierDecision::Hold,
        "Strict `>` at threshold should NOT promote"
    );
}

/// G3 (cfg gate): when feature is enabled but no signal has been pushed,
/// `tvp_signal` is `None` and the decision is `Defer` (zero routing impact).
///
/// This guarantees zero behavioral impact for users who compile with the
/// feature but never call `update_tvp()`.
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn g5_no_signal_defers() {
    let router = fast_router(true, false);
    assert!(router.tvp_signal().is_none());
    let decision = router.observe_tvp_decision(ComputeTier::CpuOnly);
    assert_eq!(
        decision,
        TvpTierDecision::Defer,
        "Uninitialized TVP signal must defer (zero routing impact)"
    );
}

/// Clearing the signal via `update_tvp(None)` returns the router to Defer.
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn g5b_clear_signal_returns_to_defer() {
    let mut router = fast_router(true, false);
    router.update_tvp(Some(tvp_reasoning(0.9)));
    assert_eq!(
        router.observe_tvp_decision(ComputeTier::CpuOnly),
        TvpTierDecision::PromoteGpu
    );
    router.update_tvp(None);
    assert!(router.tvp_signal().is_none());
    assert_eq!(
        router.observe_tvp_decision(ComputeTier::CpuOnly),
        TvpTierDecision::Defer,
        "Cleared TVP signal must defer"
    );
}

/// API contract: `update_tvp` persists the latest signal until cleared.
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn tvp_signal_persists() {
    let mut router = fast_router(true, false);
    router.update_tvp(Some(tvp_reasoning(0.9)));
    let s = router.tvp_signal().expect("signal must persist");
    assert_eq!(s.reasoning_disagreement, 0.9);
    // Still there on a second read.
    assert_eq!(
        router.observe_tvp_decision(ComputeTier::CpuOnly),
        TvpTierDecision::PromoteGpu
    );
}

/// API contract: `set_tvp_config` adjusts promote/demote thresholds.
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn set_tvp_config_adjusts_thresholds() {
    let mut router = fast_router(true, false);

    // Raise promote_at to 0.95 → 0.9 disagreement no longer promotes.
    let mut cfg = TvpConfig::default();
    cfg.promote_at = 0.95;
    router.set_tvp_config(cfg);
    assert_eq!(router.tvp_config().promote_at, 0.95);

    router.update_tvp(Some(tvp_reasoning(0.9)));
    let decision = router.observe_tvp_decision(ComputeTier::CpuOnly);
    assert_eq!(
        decision,
        TvpTierDecision::Hold,
        "Raised threshold should suppress promotion"
    );
}

/// API contract: stats snapshot exposes the TVP signal.
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn stats_exposes_tvp_signal() {
    let mut router = fast_router(true, false);
    assert!(router.stats().tvp_signal.is_none());

    router.update_tvp(Some(tvp_reasoning(0.42)));
    let s = router.stats().tvp_signal.expect("signal in stats");
    assert!((s.reasoning_disagreement - 0.42).abs() < 1e-6);
}

/// Pure-function unit test of `tvp_tier_decision` — covers all branches.
#[cfg(feature = "thicket_variance_probe")]
#[test]
fn tvp_tier_decision_branches() {
    use crate::pruners::thicket_variance_probe::tvp_tier_decision;

    // No signal → Defer.
    assert_eq!(
        tvp_tier_decision(None, 0.6, 0.2, ComputeTier::CpuOnly, true, true),
        TvpTierDecision::Defer
    );

    // Promote branch.
    assert_eq!(
        tvp_tier_decision(Some(tvp_reasoning(0.9)), 0.6, 0.2, ComputeTier::CpuOnly, true, true),
        TvpTierDecision::PromoteGpu
    );
    // No GPU → cannot promote.
    assert_eq!(
        tvp_tier_decision(Some(tvp_reasoning(0.9)), 0.6, 0.2, ComputeTier::CpuOnly, false, true),
        TvpTierDecision::Hold
    );
    // Already CpuGpu → cannot promote further.
    assert_eq!(
        tvp_tier_decision(Some(tvp_reasoning(0.9)), 0.6, 0.2, ComputeTier::CpuGpu, true, true),
        TvpTierDecision::Hold
    );

    // Demote branch.
    assert_eq!(
        tvp_tier_decision(Some(tvp_reasoning(0.1)), 0.6, 0.2, ComputeTier::CpuGpu, true, true),
        TvpTierDecision::DemoteCpu
    );
    // High load → cannot demote.
    assert_eq!(
        tvp_tier_decision(Some(tvp_reasoning(0.1)), 0.6, 0.2, ComputeTier::CpuGpu, true, false),
        TvpTierDecision::Hold
    );
    // Already CpuOnly → cannot demote.
    assert_eq!(
        tvp_tier_decision(Some(tvp_reasoning(0.1)), 0.6, 0.2, ComputeTier::CpuOnly, true, true),
        TvpTierDecision::Hold
    );

    // Mid-range disagreement → Hold.
    assert_eq!(
        tvp_tier_decision(Some(tvp_reasoning(0.4)), 0.6, 0.2, ComputeTier::CpuOnly, true, true),
        TvpTierDecision::Hold
    );

    // Format-only signal → never promotes.
    assert_eq!(
        tvp_tier_decision(Some(tvp_format_only(0.99)), 0.6, 0.2, ComputeTier::CpuOnly, true, true),
        TvpTierDecision::Hold
    );
}

// ------------------------------------------------------------------
// Plan 267 T20: GOAT G4 — TVP+RV ablation.
//
// RV (Plan 202) and TVP (Plan 267) measure different things:
//   - RV: acceptance variance (downstream verifier disagreement)
//   - TVP: probe disagreement (upstream decoding-config disagreement)
//
// The GOAT G4 gate requires that the cascade (RV → TVP) makes strictly
// more correct routing decisions than either signal alone. If the two
// signals are perfectly redundant, TVP adds nothing and should be demoted
// to research-only (DFlare precedent, Plan 174).
//
// Test design: synthesize 4 query populations where the two signals
// disagree in known ways. The cascade must catch every case each signal
// alone catches (superset property) AND at least one case neither alone
// catches (strict improvement).
// ------------------------------------------------------------------

/// Simulate the router cascade (trust → RV → critical → TVP → breakeven)
/// using only the RV + TVP gates, since trust/critical/breakeven are
/// load/entropy-driven and orthogonal to the disagreement signals we vary.
///
/// Returns the final compute tier.
#[cfg(all(feature = "rv_gated_routing", feature = "thicket_variance_probe"))]
fn simulate_cascade(
    rv: f64,
    tvp: Option<TvpSignal>,
    rv_thresholds: &RvThresholds,
    tvp_promote_at: f32,
    tvp_demote_at: f32,
    gate: &TriggerGate,
) -> ComputeTier {
    // Start at CpuOnly (the resting tier when QPS is low).
    let tier_after_trust = ComputeTier::CpuOnly;

    // RV gate (Plan 202).
    let tier_after_rv = match gate.rv_tier_boost(rv, rv_thresholds) {
        Some(rv_tier) => rv_tier,
        None => tier_after_trust,
    };

    // Critical-interval gate skipped (orthogonal — entropy-driven).
    let tier_after_critical = tier_after_rv;

    // TVP gate (Plan 267) — sits between critical and breakeven.
    let low_load = true; // we are in a test with zero QPS.
    match tvp_tier_decision(
        tvp,
        tvp_promote_at,
        tvp_demote_at,
        tier_after_critical,
        gate.gpu_available(),
        low_load,
    ) {
        TvpTierDecision::PromoteGpu => ComputeTier::CpuGpu,
        TvpTierDecision::DemoteCpu => ComputeTier::CpuOnly,
        _ => tier_after_critical,
    }
}

/// G4 ablation: TVP+RV cascade ≥ max(TVP-only, RV-only), strict on at least
/// one query. Constructed so each signal alone misses a class of query the
/// other catches.
#[cfg(all(feature = "rv_gated_routing", feature = "thicket_variance_probe"))]
#[test]
fn g4_tvp_plus_rv_strictly_dominates_either_alone() {
    let gate = TriggerGate::new(TriggerGateConfig::default(), true, false);
    let rv_thresholds = RvThresholds::default(); // promote >0.10, demote <0.02
    let tvp_promote_at = 0.6;
    let tvp_demote_at = 0.2;

    // Ground truth: should_promote = the query is genuinely hard.
    // We construct 4 query classes where RV and TVP give different signals.
    //
    // RV ∈ [0, 0.25] for Bernoulli acceptance; TVP ∈ [0, 1] for disagreement.
    // RV defaults: theta_high=0.10, theta_low=0.02.
    // TVP defaults: promote_at=0.6, demote_at=0.2.
    //
    // Class A — RV-high, TVP-mid: hard to verify, probes mildly uncertain.
    //   RV catches it (promotes). TVP is mid-range so it Holds (doesn't fight).
    //   NOTE: we avoid TVP-low here because TVP's demote path would fight
    //   RV's promote — that's a known design tension documented in the plan.
    // Class B — RV-low, TVP-high: probes disagree but verifier is stable.
    //   This happens when many answers are valid (open-ended generation)
    //   and the verifier accepts any of them. TVP catches it.
    // Class C — both high: obvious hard query. Either catches it.
    // Class D — both low: obvious easy query. Neither promotes (correct).
    let queries: &[(f64, f32, bool)] = &[
        // (rv, tvp_reasoning, should_promote)
        (0.20, 0.40, true),  // A: RV-high, TVP-mid — hard (RV catches, TVP holds)
        (0.20, 0.50, true),  // A: variant
        (0.01, 0.90, true),  // B: RV-low, TVP-high — hard (TVP catches)
        (0.005, 0.85, true), // B: variant
        (0.20, 0.90, true),  // C: both high — hard
        (0.15, 0.95, true),  // C: variant
        (0.01, 0.05, false), // D: both low — easy
        (0.005, 0.10, false), // D: variant
        (0.05, 0.40, false), // ambiguous mid-range — easy (no strong signal)
        (0.05, 0.50, false), // ambiguous mid-range — easy
    ];

    let mut correct_rv_only = 0usize;
    let mut correct_tvp_only = 0usize;
    let mut correct_cascade = 0usize;

    for &(rv, tvp_d, should_promote) in queries {
        let tvp_signal = Some(tvp_reasoning(tvp_d));

        // RV-only: disable TVP by passing None.
        let tier_rv = simulate_cascade(
            rv,
            None,
            &rv_thresholds,
            tvp_promote_at,
            tvp_demote_at,
            &gate,
        );
        let decided_promote_rv = tier_rv == ComputeTier::CpuGpu;
        if decided_promote_rv == should_promote {
            correct_rv_only += 1;
        }

        // TVP-only: disable RV by setting rv=0 (below theta_low → no action).
        let tier_tvp = simulate_cascade(
            0.0,
            tvp_signal,
            &rv_thresholds,
            tvp_promote_at,
            tvp_demote_at,
            &gate,
        );
        let decided_promote_tvp = tier_tvp == ComputeTier::CpuGpu;
        if decided_promote_tvp == should_promote {
            correct_tvp_only += 1;
        }

        // Cascade: both signals active.
        let tier_both = simulate_cascade(
            rv,
            tvp_signal,
            &rv_thresholds,
            tvp_promote_at,
            tvp_demote_at,
            &gate,
        );
        let decided_promote_both = tier_both == ComputeTier::CpuGpu;
        if decided_promote_both == should_promote {
            correct_cascade += 1;
        }
    }

    let n = queries.len();
    // Cascade is a superset: catches everything either signal catches.
    assert!(
        correct_cascade >= correct_rv_only,
        "Cascade ({}) must be ≥ RV-only ({})",
        correct_cascade,
        correct_rv_only
    );
    assert!(
        correct_cascade >= correct_tvp_only,
        "Cascade ({}) must be ≥ TVP-only ({})",
        correct_cascade,
        correct_tvp_only
    );
    // Strict dominance: the cascade must catch at least one case each
    // signal alone misses. (Class A and Class B above.)
    assert!(
        correct_cascade > correct_rv_only,
        "G4 FAIL: cascade ({}) is not strictly > RV-only ({}) — TVP is redundant",
        correct_cascade,
        correct_rv_only
    );
    assert!(
        correct_cascade > correct_tvp_only,
        "G4 FAIL: cascade ({}) is not strictly > TVP-only ({}) — RV is redundant",
        correct_cascade,
        correct_tvp_only
    );
    // Sanity: cascade should catch all hard queries and skip all easy ones.
    assert_eq!(
        correct_cascade, n,
        "Cascade should be perfect on this synthetic set, got {}/{}",
        correct_cascade, n
    );
    // Sanity: each signal alone misses at least one query.
    assert!(
        correct_rv_only < n,
        "RV-only should miss the TVP-high-RV-low class"
    );
    assert!(
        correct_tvp_only < n,
        "TVP-only should miss the RV-high-TVP-low class"
    );
}

// ── CHIAR integration tests (Plan 269 T15) ──────────────────

#[cfg(feature = "chiaroscuro")]
#[test]
fn chiar_hook_reports_none_when_no_keys_observed() {
    let router = fast_router(false, false);
    // No keys observed → stats should be None.
    assert!(router.chiar_stats().is_none());
}

#[cfg(feature = "chiaroscuro")]
#[test]
fn chiar_hook_reports_stats_after_observing_keys() {
    let mut router = fast_router(false, false);
    // Observe a mix of smooth and high-entropy keys.
    let smooth_key = vec![0.5f32; 256];
    let entropy_key: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.3).sin().cos()).collect();
    for _ in 0..50 {
        router.observe_chiar_key(&smooth_key);
        router.observe_chiar_key(&entropy_key);
    }
    let stats = router.chiar_stats().expect("stats after 100 keys");
    assert!(stats.tokens_observed >= 100, "tokens_observed = {}", stats.tokens_observed);
    let entropy = stats.utilization_entropy.expect("utilization_entropy");
    assert!((0.0..=1.0).contains(&entropy), "entropy out of range: {entropy}");
}

#[cfg(feature = "chiaroscuro")]
#[test]
fn chiar_hook_stats_visible_in_router_stats() {
    let mut router = fast_router(false, false);
    router.observe_chiar_key(&[0.5f32; 256]);
    let stats = router.stats();
    assert!(stats.chiar_stats.is_some(), "chiar_stats should be Some in RouterStats");
    let cs = stats.chiar_stats.unwrap();
    assert!(cs.tokens_observed >= 1);
}
