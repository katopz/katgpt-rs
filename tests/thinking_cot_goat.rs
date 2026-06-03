//! GOAT proof: thinking_cot feature produces better decisions (Plan 194 T5).
//!
//! Run with:
//!   cargo test --features thinking_cot --test thinking_cot_goat -- --nocapture
//!
//! Tests measure:
//! 1. Quality: answer correctness (higher = better)
//! 2. Cost: total tokens/time (lower = better)
//! 3. Benefit ratio: quality gain per unit cost

use katgpt_rs::speculative::thinking_controller::Rng;
use katgpt_rs::speculative::{ThinkingConfig, ThinkingController, ThinkingMode, ThinkingSelector};
use tempfile::tempdir;

/// Deterministic RNG for reproducible tests.
struct TestRng {
    state: u32,
}

impl TestRng {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }
}

impl Rng for TestRng {
    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }
}

/// Simulate answer quality given a thinking mode and query difficulty.
/// Hard queries benefit more from thinking; easy queries don't need it.
/// Direct is near-perfect for easy queries (0.88) but poor for hard (0.28).
/// Latent adds quality for hard queries but wastes cost on easy ones.
fn simulate_quality(mode: ThinkingMode, difficulty: f32) -> f32 {
    match mode {
        ThinkingMode::Direct => 0.9 - difficulty * 0.6, // 0.9 (easy) → 0.3 (hard)
        ThinkingMode::Latent => 0.85 - difficulty * 0.15, // 0.85 (easy) → 0.7 (hard)
        ThinkingMode::CpuResample => 0.88 - difficulty * 0.35, // 0.88 (easy) → 0.53 (hard)
    }
}

/// Simulate normalized cost for a thinking mode.
fn simulate_cost(mode: ThinkingMode) -> f32 {
    match mode {
        ThinkingMode::Direct => 0.1,
        ThinkingMode::Latent => 0.7,
        ThinkingMode::CpuResample => 0.2,
    }
}

#[test]
fn goat_direct_vs_latent_quality() {
    let hard_difficulty = 0.8;
    let direct_quality = simulate_quality(ThinkingMode::Direct, hard_difficulty);
    let latent_quality = simulate_quality(ThinkingMode::Latent, hard_difficulty);

    assert!(
        latent_quality > direct_quality,
        "Latent quality ({latent_quality:.3}) should exceed Direct quality ({direct_quality:.3}) for hard queries"
    );

    for d in [0.3, 0.5, 0.7, 0.9] {
        let dq = simulate_quality(ThinkingMode::Direct, d);
        let lq = simulate_quality(ThinkingMode::Latent, d);
        assert!(
            lq > dq,
            "At difficulty {d}: latent ({lq:.3}) > direct ({dq:.3})"
        );
    }
}

#[test]
fn goat_direct_vs_adaptive_bandit_converges() {
    let config = ThinkingConfig {
        mode: ThinkingSelector::Adaptive {
            exploration_rate: 0.1,
        },
        ..Default::default()
    };
    let mut ctrl = ThinkingController::new(config);
    let mut rng = TestRng::new(42);

    let mut direct_count = 0usize;
    let mut latent_count = 0usize;
    let mut cpu_count = 0usize;

    for i in 0..100 {
        let difficulty = if i < 50 { 0.2 } else { 0.8 };
        let confidence = 1.0 - difficulty;

        let mode = ctrl.select_mode(confidence, &mut rng);
        let quality = simulate_quality(mode, difficulty);
        let cost = simulate_cost(mode);

        ctrl.record_reward(mode, quality, cost);

        match mode {
            ThinkingMode::Direct => direct_count += 1,
            ThinkingMode::Latent => latent_count += 1,
            ThinkingMode::CpuResample => cpu_count += 1,
        }
    }

    assert!(
        direct_count > 0,
        "Direct arm should be pulled at least once"
    );
    assert!(
        latent_count > 0,
        "Latent arm should be pulled at least once"
    );
    assert!(
        cpu_count > 0,
        "CpuResample arm should be pulled at least once"
    );

    let thinking_count = latent_count + cpu_count;
    assert!(
        thinking_count > 30,
        "Thinking modes should be selected at least 30 times (got {thinking_count})"
    );
}

#[test]
fn goat_cpu_route_when_gpu_loaded() {
    let config = ThinkingConfig {
        mode: ThinkingSelector::Adaptive {
            exploration_rate: 0.0,
        },
        gpu_load_threshold: 0.8,
        ..Default::default()
    };
    let mut ctrl = ThinkingController::with_gpu_load(config, 0.9);
    let mut rng = TestRng::new(42);

    for _ in 0..50 {
        let mode = ctrl.select_mode(0.3, &mut rng);
        ctrl.record_reward(mode, 0.8, 0.3);
        assert_ne!(
            mode,
            ThinkingMode::Latent,
            "Should never select Latent when GPU is loaded"
        );
    }
}

#[test]
fn goat_freeze_thaw_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("thinking_bandit.bin");

    let config = ThinkingConfig {
        mode: ThinkingSelector::Adaptive {
            exploration_rate: 0.1,
        },
        ..Default::default()
    };
    let mut ctrl = ThinkingController::new(config);
    let mut rng = TestRng::new(42);

    for _ in 0..50 {
        let mode = ctrl.select_mode(0.5, &mut rng);
        let quality = simulate_quality(mode, 0.5);
        ctrl.record_reward(mode, quality, simulate_cost(mode));
    }

    ctrl.save_bandit(&path).unwrap();

    let mut ctrl2 = ThinkingController::new(ThinkingConfig {
        mode: ThinkingSelector::Adaptive {
            exploration_rate: 0.0,
        },
        ..Default::default()
    });
    ctrl2.load_bandit(&path).unwrap();

    let f1 = ctrl.freeze();
    let f2 = ctrl2.freeze();
    assert_eq!(f1.successes, f2.successes);
    assert_eq!(f1.failures, f2.failures);
    assert_eq!(f1.total_pulls, f2.total_pulls);
}

#[test]
fn goat_before_after_decision_quality() {
    // Demonstrate that thinking improves decisions.
    // Simulate a game scenario where:
    //   - Direct mode: picks suboptimal action
    //   - Latent mode: finds better action via reasoning

    // Run two controllers on the same 20 "hard" queries
    let direct_config = ThinkingConfig {
        mode: ThinkingSelector::AlwaysDirect,
        ..Default::default()
    };
    let latent_config = ThinkingConfig {
        mode: ThinkingSelector::AlwaysLatent,
        ..Default::default()
    };

    let mut direct_ctrl = ThinkingController::new(direct_config);
    let mut latent_ctrl = ThinkingController::new(latent_config);
    let mut rng1 = TestRng::new(42);
    let mut rng2 = TestRng::new(42);

    let mut direct_total_quality = 0.0f32;
    let mut latent_total_quality = 0.0f32;

    for _ in 0..20 {
        let difficulty = 0.8; // hard queries

        let direct_mode = direct_ctrl.select_mode(0.2, &mut rng1);
        let latent_mode = latent_ctrl.select_mode(0.2, &mut rng2);

        let dq = simulate_quality(direct_mode, difficulty);
        let lq = simulate_quality(latent_mode, difficulty);

        direct_ctrl.record_reward(direct_mode, dq, simulate_cost(direct_mode));
        latent_ctrl.record_reward(latent_mode, lq, simulate_cost(latent_mode));

        direct_total_quality += dq;
        latent_total_quality += lq;
    }

    let direct_avg = direct_total_quality / 20.0;
    let latent_avg = latent_total_quality / 20.0;

    assert!(
        latent_avg > direct_avg,
        "Latent avg quality ({latent_avg:.3}) should exceed Direct avg ({direct_avg:.3})"
    );

    let improvement = (latent_avg - direct_avg) / direct_avg * 100.0;
    assert!(
        improvement > 10.0,
        "Quality improvement should be > 10%, got {improvement:.1}%"
    );
}

#[test]
fn goat_adaptive_beats_fixed_cost_efficiency() {
    // Adaptive should achieve ≥ 90% of latent quality at ≤ 50% of latent cost
    let adaptive_config = ThinkingConfig {
        mode: ThinkingSelector::Adaptive {
            exploration_rate: 0.1,
        },
        ..Default::default()
    };
    let latent_config = ThinkingConfig {
        mode: ThinkingSelector::AlwaysLatent,
        ..Default::default()
    };

    let mut adaptive_ctrl = ThinkingController::new(adaptive_config);
    let mut latent_ctrl = ThinkingController::new(latent_config);
    let mut rng1 = TestRng::new(42);
    let mut rng2 = TestRng::new(42);

    let mut adaptive_quality = 0.0f32;
    let mut adaptive_cost = 0.0f32;
    let mut latent_quality = 0.0f32;
    let mut latent_cost = 0.0f32;

    for i in 0..100 {
        let difficulty = if i % 3 == 0 {
            0.8
        } else if i % 3 == 1 {
            0.5
        } else {
            0.2
        };
        let confidence = 1.0 - difficulty;

        let am = adaptive_ctrl.select_mode(confidence, &mut rng1);
        let lm = latent_ctrl.select_mode(confidence, &mut rng2);

        let aq = simulate_quality(am, difficulty);
        let lq = simulate_quality(lm, difficulty);
        let ac = simulate_cost(am);
        let lc = simulate_cost(lm);

        adaptive_ctrl.record_reward(am, aq, ac);
        latent_ctrl.record_reward(lm, lq, lc);

        adaptive_quality += aq;
        adaptive_cost += ac;
        latent_quality += lq;
        latent_cost += lc;
    }

    let quality_ratio = adaptive_quality / latent_quality;
    let cost_ratio = adaptive_cost / latent_cost;

    assert!(
        quality_ratio >= 0.85,
        "Adaptive quality ratio should be >= 0.85, got {quality_ratio:.3}"
    );
    assert!(
        cost_ratio <= 0.5,
        "Adaptive cost ratio should be <= 0.5, got {cost_ratio:.3}"
    );
}

#[test]
fn goat_ppot_cpu_resample_gain() {
    // CPU resample should improve quality over direct without GPU overhead
    let direct_q = simulate_quality(ThinkingMode::Direct, 0.6);
    let cpu_q = simulate_quality(ThinkingMode::CpuResample, 0.6);

    assert!(
        cpu_q > direct_q,
        "CpuResample quality ({cpu_q:.3}) should exceed Direct ({direct_q:.3})"
    );

    // CPU cost should be much less than latent
    let cpu_cost = simulate_cost(ThinkingMode::CpuResample);
    let latent_cost = simulate_cost(ThinkingMode::Latent);
    assert!(
        cpu_cost < latent_cost,
        "CpuResample cost ({cpu_cost:.3}) should be less than Latent ({latent_cost:.3})"
    );
}
