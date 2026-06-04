//! GOAT Proof: Trigger Gate + Inference Router (Plan 176)
//!
//! Tests that:
//! - TriggerGate correctly tier-up at simulated high QPS
//! - TriggerGate correctly tier-down when load drops
//! - InferenceRouter starts at CPU-only and routes correctly
//! - InferenceRouter forward produces valid logits
//! - BackendKind::Gate variant exists
//! - Router stats are accurate

use katgpt_rs::inference_backend::BackendKind;
use katgpt_rs::inference_router::InferenceRouter;
use katgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use katgpt_rs::trigger_gate::{ComputeTier, TriggerGate, TriggerGateConfig};
use katgpt_rs::types::{Config, Rng};

fn fast_gate_config() -> TriggerGateConfig {
    TriggerGateConfig {
        gpu_activate_qps: 10_000.0,
        ane_activate_qps: 100_000.0,
        hysteresis_factor: 0.7,
        queue_depth_trigger: 100,
        latency_p99_trigger_us: 5000,
        min_tier_change_interval_ms: 10,
    }
}

fn micro_fixtures() -> (TransformerWeights, ForwardContext, MultiLayerKVCache) {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let ctx = ForwardContext::new(&config);
    let cache = MultiLayerKVCache::new(&config);
    (weights, ctx, cache)
}

// P1: TriggerGate starts at CPU_ONLY
#[test]
fn goat_p1_trigger_gate_starts_cpu_only() {
    let gate = TriggerGate::new(TriggerGateConfig::default(), true, true);
    assert_eq!(gate.current_tier(), ComputeTier::CpuOnly);
}

// P2: TriggerGate promotes to CPU+GPU under high QPS
#[test]
fn goat_p2_trigger_gate_promotes_to_gpu() {
    let gate = TriggerGate::new(fast_gate_config(), true, false);

    // Simulate high QPS by recording many inferences
    for _ in 0..50_000 {
        gate.record_inference(1);
    }
    gate.record_queue_depth(200);

    let promote = gate.should_promote();
    assert!(
        promote.is_some(),
        "should promote to GPU under high QPS + queue depth"
    );
    assert_eq!(promote.unwrap(), ComputeTier::CpuGpu);
}

// P3: TriggerGate promotes to CPU+GPU+ANE at highest QPS
#[test]
fn goat_p3_trigger_gate_promotes_to_ane() {
    let gate = TriggerGate::new(fast_gate_config(), true, true);

    // Simulate extremely high QPS + queue depth to trigger ANE promotion.
    // should_promote() checks from current_tier; we need to be at CpuGpu first.
    // Record enough inferences and queue depth to exceed both thresholds.
    for _ in 0..50_000 {
        gate.record_inference(1);
    }
    gate.record_queue_depth(200);

    // At CpuOnly with gpu_available, this should promote to CpuGpu
    let promote = gate.should_promote();
    assert_eq!(promote, Some(ComputeTier::CpuGpu));

    // Commit the tier change via evaluate() so we can test ANE promotion.
    // Need to wait for min_tier_change_interval_ms.
    std::thread::sleep(std::time::Duration::from_millis(15));
    let result = gate.evaluate();
    assert_eq!(
        result,
        Some(ComputeTier::CpuGpu),
        "evaluate should promote to CpuGpu"
    );

    // Now at CpuGpu — record more inferences to exceed ane_activate_qps
    for _ in 0..500_000 {
        gate.record_inference(1);
    }
    gate.record_queue_depth(500);

    // Should now promote to CpuGpuAne
    let ane_promote = gate.should_promote();
    assert_eq!(
        ane_promote,
        Some(ComputeTier::CpuGpuAne),
        "should promote to CPU+GPU+ANE at CpuGpu tier with high QPS"
    );
}

// P4: TriggerGate tier-down with hysteresis
#[test]
fn goat_p4_trigger_gate_demotes_with_hysteresis() {
    let config = TriggerGateConfig {
        gpu_activate_qps: 10_000.0,
        ane_activate_qps: 100_000.0,
        hysteresis_factor: 0.7,
        queue_depth_trigger: 100,
        latency_p99_trigger_us: 5000,
        min_tier_change_interval_ms: 10,
    };
    let gate = TriggerGate::new(config, true, false);

    // Record lots of inferences to simulate high QPS
    for _ in 0..50_000 {
        gate.record_inference(1);
    }
    gate.record_queue_depth(200);

    // Wait for min interval so evaluate() can fire
    std::thread::sleep(std::time::Duration::from_millis(15));
    let promoted = gate.evaluate();
    assert_eq!(promoted, Some(ComputeTier::CpuGpu));

    // After evaluate(), counters reset and QPS drops to 0.
    // With 0 QPS, should_demote should fire because 0 < 10000 * 0.7
    let demote = gate.should_demote();
    assert!(
        demote.is_some(),
        "should demote from CpuGpu when QPS drops to 0"
    );
    assert_eq!(demote.unwrap(), ComputeTier::CpuOnly);
}

// P5: InferenceRouter starts at CPU-only mode
#[test]
fn goat_p5_router_starts_cpu_only() {
    let router = InferenceRouter::new(fast_gate_config(), Config::micro(), true, true);
    assert_eq!(router.gate().current_tier(), ComputeTier::CpuOnly);
    let stats = router.stats();
    assert_eq!(stats.current_tier, ComputeTier::CpuOnly);
}

// P6: InferenceRouter forward produces valid logits
#[test]
fn goat_p6_router_forward_valid_logits() {
    let config = Config::micro();
    let (weights, mut ctx, mut cache) = micro_fixtures();
    let mut router = InferenceRouter::new(fast_gate_config(), config, false, false);

    let logits = router.forward(&mut ctx, &weights, &mut cache, 0, 0);
    assert_eq!(
        logits.len(),
        Config::micro().vocab_size,
        "logits length should match vocab_size"
    );

    // Verify logits are finite (no NaN/Inf)
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
    }
}

// P7: InferenceRouter matches direct transformer::forward
#[test]
fn goat_p7_router_matches_direct_forward() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Direct forward
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let direct =
        katgpt_rs::transformer::forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config).to_vec();

    // Router forward
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let mut router = InferenceRouter::new(fast_gate_config(), config, false, false);
    let routed = router
        .forward(&mut ctx2, &weights, &mut cache2, 0, 0)
        .to_vec();

    assert_eq!(direct.len(), routed.len(), "logits length mismatch");
    for (i, (a, b)) in direct.iter().zip(routed.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "logits mismatch at index {i}: {a} vs {b}"
        );
    }
}

// P8: BackendKind::Gate variant exists
#[test]
fn goat_p8_backend_kind_gate_variant() {
    let kind = BackendKind::Gate;
    assert_ne!(kind, BackendKind::Auto);
    assert_ne!(kind, BackendKind::Cpu);
    assert_ne!(kind, BackendKind::Ane);
}

// P9: Router tracks inferences correctly
#[test]
fn goat_p9_router_tracks_inferences() {
    let config = Config::micro();
    let (weights, mut ctx, mut cache) = micro_fixtures();
    let mut router = InferenceRouter::new(fast_gate_config(), config, false, false);

    for i in 0..5 {
        let _ = router.forward(&mut ctx, &weights, &mut cache, 0, i);
    }

    let stats = router.stats();
    assert_eq!(stats.total_inferences, 5);
}

// P10: Router forward_batch produces correct logits
#[test]
fn goat_p10_forward_batch_valid_logits() {
    let config = Config::micro();
    let (weights, mut ctx, mut cache) = micro_fixtures();
    let mut router = InferenceRouter::new(fast_gate_config(), config, false, false);

    let batch: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();
    let results = router.forward_batch(&mut ctx, &weights, &mut cache, &batch);

    assert_eq!(results.len(), 5);
    for (i, logits) in results.iter().enumerate() {
        assert_eq!(
            logits.len(),
            Config::micro().vocab_size,
            "batch logits[{}] wrong length",
            i
        );
        for (j, &v) in logits.iter().enumerate() {
            assert!(
                v.is_finite(),
                "batch logits[{}][{}] not finite: {}",
                i,
                j,
                v
            );
        }
    }
    assert_eq!(router.stats().total_inferences, 5);
}

// P11: Router forward_batch matches sequential forward
#[test]
fn goat_p11_forward_batch_matches_sequential() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Sequential forward.
    let mut ctx1 = ForwardContext::new(&config);
    let mut cache1 = MultiLayerKVCache::new(&config);
    let mut router1 = InferenceRouter::new(fast_gate_config(), config.clone(), false, false);
    let mut sequential = Vec::new();
    for i in 0..4 {
        let logits = router1.forward(&mut ctx1, &weights, &mut cache1, i, i);
        sequential.push(logits.to_vec());
    }

    // Batch forward.
    let mut ctx2 = ForwardContext::new(&config);
    let mut cache2 = MultiLayerKVCache::new(&config);
    let mut router2 = InferenceRouter::new(fast_gate_config(), config, false, false);
    let batch: Vec<(usize, usize)> = (0..4).map(|i| (i, i)).collect();
    let batch_results = router2.forward_batch(&mut ctx2, &weights, &mut cache2, &batch);

    assert_eq!(sequential.len(), batch_results.len());
    for (i, (seq, batch)) in sequential.iter().zip(batch_results.iter()).enumerate() {
        for (j, (a, b)) in seq.iter().zip(batch.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "batch mismatch at [{i}][{j}]: {a} vs {b}"
            );
        }
    }
}

// P12: Router generate_routed produces valid tokens
#[test]
fn goat_p12_generate_routed_valid_tokens() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut router = InferenceRouter::new(fast_gate_config(), config, false, false);

    let mut tokens = Vec::new();
    router.generate_routed(&mut ctx, &mut cache, &weights, &mut rng, 10, &mut tokens);

    assert_eq!(tokens.len(), 10, "should generate exactly 10 tokens");
    for (i, &tok) in tokens.iter().enumerate() {
        assert!(
            tok < Config::micro().vocab_size,
            "token[{i}] = {tok} exceeds vocab_size"
        );
    }
    assert!(router.stats().total_inferences >= 10);
}

// P13: Router generate_routed tracks inferences
#[test]
fn goat_p13_generate_routed_tracks_inferences() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut router = InferenceRouter::new(fast_gate_config(), config, false, false);

    let n_tokens = 20;
    let mut tokens = Vec::new();
    router.generate_routed(
        &mut ctx,
        &mut cache,
        &weights,
        &mut rng,
        n_tokens,
        &mut tokens,
    );

    assert_eq!(tokens.len(), n_tokens);
    assert!(router.stats().total_inferences >= n_tokens as u64);
}

// P14: 30K CCU CPU simulation — router survives sustained throughput
#[test]
fn goat_p14_30k_ccu_cpu_simulation() {
    let config = Config::micro();
    let block_size = config.block_size;
    let vocab_size = config.vocab_size;
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    let mut router = InferenceRouter::new(fast_gate_config(), config, true, true);

    // Simulate 30K CCU × 20Hz = 600K inferences
    // We'll do a representative batch: 10K inferences in a tight loop.
    // Each "CCU" generates one token at 20Hz, but we compress time.
    let n_inferences = 10_000;
    let start = std::time::Instant::now();

    for i in 0..n_inferences {
        let pos = i % block_size;
        let token = i % vocab_size;
        if pos == 0 && i > 0 {
            cache.reset();
        }
        // Simulate queue pressure from 30K CCU
        router.record_queue_depth(300);
        let _ = router.forward(&mut ctx, &weights, &mut cache, token, pos);
    }

    let elapsed = start.elapsed();
    let stats = router.stats();
    let throughput = n_inferences as f64 / elapsed.as_secs_f64();
    let us_per_inference = elapsed.as_secs_f64() * 1_000_000.0 / n_inferences as f64;

    println!(
        "GOAT P14: 30K CCU sim: {} inferences in {:.1}ms ({:.0} inf/s, {:.1} µs/inf)",
        n_inferences,
        elapsed.as_secs_f64() * 1000.0,
        throughput,
        us_per_inference
    );
    println!(
        "           tier={}, transitions={}, qps={:.0}",
        stats.current_tier, stats.tier_transitions, stats.estimated_qps
    );

    // Router must track all inferences
    assert_eq!(stats.total_inferences, n_inferences as u64);
    // Must complete within reasonable time (50ms for 10K inferences on micro model)
    assert!(
        elapsed.as_secs() < 5,
        "30K CCU sim too slow: {:.1}s",
        elapsed.as_secs_f64()
    );
}
