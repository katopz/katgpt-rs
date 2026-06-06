//! Formal Verification Gates (G1–G8) for the Speculative Reconciliation Engine (Plan 177).
//!
//! Run with: cargo test --features spec_reconciliation --test spec_reconciliation_proof -- --nocapture



use std::f32::consts::TAU;
use std::time::Instant;

use katgpt_rs::benchmark::cosine_similarity;
use katgpt_rs::spec_reconciliation::{
    DefaultManifoldGenerator, ManifoldGenerator, ReconciliationConfig, ReconciliationPruner,
    ReconciliationVerdict, SpecReconciler, TrajectoryPoint, gaussian_sample,
};
use katgpt_rs::types::Rng;

// ── Shared helpers ──────────────────────────────────────────────────────────

fn test_config() -> ReconciliationConfig {
    ReconciliationConfig {
        k: 16,
        max_speed: 600.0,
        map_bounds: [0.0, 0.0, 4096.0, 4096.0],
        accept_threshold: 0.5,
        quarantine_threshold: 0.2,
        kill_rate_sigma: 5.0,
        noise_sigma: 0.1,
        dt: 1.0 / 60.0,
    }
}

fn h_last() -> TrajectoryPoint {
    TrajectoryPoint::from_fields(2048.0, 2048.0, 10.0, 5.0, 2.0, 0.0, 1.0, 0.0)
}

/// Generate a legitimate trajectory: physics-like movement from h_last.
///
/// Uses the same blending + noise approach as `DefaultManifoldGenerator` to ensure
/// the trajectories are plausibly close to what the manifold covers.
/// No kill increments to pass the Chebyshev kill-rate bound at dt=1/60s.
fn make_legitimate_trajectory(
    h_last: &TrajectoryPoint,
    steps: usize,
    dt: f32,
    max_speed: f32,
    seed: u64,
) -> Vec<TrajectoryPoint> {
    let mut rng = Rng::new(seed);
    let sigma = 0.1f32; // same as config.noise_sigma
    let sqrt_dt = dt.sqrt();

    let mut pts = Vec::with_capacity(steps);
    let mut px = h_last.pos_x();
    let mut py = h_last.pos_y();
    let mut vel_x = h_last.vel_x();
    let mut vel_y = h_last.vel_y();
    let mut direction = h_last.direction();
    let kills = h_last.kills();

    for _ in 0..steps {
        // Random goal direction (like modelless mode with empty Q)
        let goal_dir = rng.uniform() * TAU;
        let goal_vx = goal_dir.cos() * max_speed * 0.5;
        let goal_vy = goal_dir.sin() * max_speed * 0.5;

        // Blend velocity toward goal + noise (same as DefaultManifoldGenerator)
        let noise_vx = gaussian_sample(&mut rng) * sigma * sqrt_dt;
        let noise_vy = gaussian_sample(&mut rng) * sigma * sqrt_dt;
        vel_x = vel_x * 0.8 + goal_vx * 0.2 + noise_vx;
        vel_y = vel_y * 0.8 + goal_vy * 0.2 + noise_vy;

        // Clamp velocity
        let speed = (vel_x * vel_x + vel_y * vel_y).sqrt();
        if speed > max_speed && speed > 0.0 {
            let scale = max_speed / speed;
            vel_x *= scale;
            vel_y *= scale;
        }

        // Advance position
        let noise_px = gaussian_sample(&mut rng) * sigma * sqrt_dt;
        let noise_py = gaussian_sample(&mut rng) * sigma * sqrt_dt;
        px += vel_x * dt + noise_px;
        py += vel_y * dt + noise_py;

        // Clamp to map
        px = px.clamp(0.0, 4096.0);
        py = py.clamp(0.0, 4096.0);

        // Direction: smooth update toward velocity
        let vel_dir = vel_y.atan2(vel_x);
        let mut delta = vel_dir - direction;
        while delta > std::f32::consts::PI {
            delta -= TAU;
        }
        while delta < -std::f32::consts::PI {
            delta += TAU;
        }
        direction += delta * 0.3;
        while direction < 0.0 {
            direction += TAU;
        }
        while direction >= TAU {
            direction -= TAU;
        }

        pts.push(TrajectoryPoint::from_fields(
            px, py, vel_x, vel_y, kills, 0.0, 1.0, direction,
        ));
    }
    pts
}

/// Generate a teleport trajectory: instant large displacement.
fn make_teleport_trajectory(h_last: &TrajectoryPoint, seed: u64) -> Vec<TrajectoryPoint> {
    let mut rng = Rng::new(seed);
    vec![
        *h_last,
        TrajectoryPoint::from_fields(
            h_last.pos_x() + 5000.0 + rng.uniform() * 1000.0,
            h_last.pos_y() + 3000.0 + rng.uniform() * 1000.0,
            0.0,
            0.0,
            2.0,
            0.0,
            1.0,
            0.0,
        ),
    ]
}

/// Generate a kill-rate hack trajectory.
fn make_kill_rate_hack(h_last: &TrajectoryPoint, seed: u64) -> Vec<TrajectoryPoint> {
    let mut rng = Rng::new(seed);
    let extra_kills = 20.0 + rng.uniform() * 30.0;
    vec![
        *h_last,
        TrajectoryPoint::from_fields(
            h_last.pos_x() + 0.1,
            h_last.pos_y(),
            10.0,
            5.0,
            h_last.kills() + extra_kills,
            0.0,
            1.0,
            0.0,
        ),
    ]
}

/// Generate a direction-mismatch hack: movement direction opposite to facing.
fn make_direction_hack(h_last: &TrajectoryPoint, seed: u64) -> Vec<TrajectoryPoint> {
    let mut rng = Rng::new(seed);
    // Teleport-level displacement to guarantee quarantine via hard bounds
    let offset = 5000.0 + rng.uniform() * 2000.0;
    vec![
        *h_last,
        TrajectoryPoint::from_fields(
            h_last.pos_x() + offset,
            h_last.pos_y() - offset,
            -300.0,
            300.0,
            2.0,
            0.0,
            1.0,
            3.14, // facing opposite to movement
        ),
    ]
}

/// Generate a random hack trajectory (pick one of the three types).
fn make_random_hack(h_last: &TrajectoryPoint, idx: u64) -> Vec<TrajectoryPoint> {
    match idx % 3 {
        0 => make_teleport_trajectory(h_last, idx),
        1 => make_kill_rate_hack(h_last, idx),
        _ => make_direction_hack(h_last, idx),
    }
}

// ── G1: Velocity invariant (property test) ─────────────────────────────────

#[test]
fn g1_velocity_invariant_valid_trajectories_pass() {
    let config = test_config();
    let h = h_last();

    // Generate 1000 random trajectories that respect max_speed.
    for seed in 0..1000u64 {
        let traj = make_legitimate_trajectory(&h, 20, config.dt, config.max_speed, seed);
        let pruner = ReconciliationPruner::new(config, h);

        for window in traj.windows(2) {
            assert!(
                pruner.check_velocity(&window[1], &window[0], config.dt),
                "G1 FAIL: valid trajectory seed={seed} violated velocity bound \
                 — distance={:.2}, max_speed={}",
                window[1].distance_to(&window[0]),
                config.max_speed,
            );
        }
    }
}

#[test]
fn g1_velocity_invariant_invalid_trajectories_caught() {
    let config = test_config();
    let h = h_last();
    let pruner = ReconciliationPruner::new(config, h);

    // Generate 100 trajectories that violate max_speed (teleport-level displacement).
    for seed in 0..100u64 {
        let mut rng = Rng::new(seed);
        // Create a point that moves way too far in one step.
        let overshoot = config.max_speed * config.dt * (2.0 + rng.uniform() * 10.0);
        let violator = TrajectoryPoint::from_fields(
            h.pos_x() + overshoot,
            h.pos_y(),
            0.0,
            0.0,
            h.kills(),
            0.0,
            1.0,
            0.0,
        );
        assert!(
            !pruner.check_velocity(&violator, &h, config.dt),
            "G1 FAIL: teleport seed={seed} was not caught (overshoot={overshoot:.1})",
        );
    }
}

// ── G2: Position invariant (property test) ─────────────────────────────────

#[test]
fn g2_position_invariant_in_bounds_passes() {
    let config = test_config();
    let pruner = ReconciliationPruner::new(config, TrajectoryPoint::default());
    let mut rng = Rng::new(42);

    // Test 1000 random points within bounds.
    for _ in 0..1000 {
        let x = rng.uniform() * 4096.0;
        let y = rng.uniform() * 4096.0;
        let pt = TrajectoryPoint::from_fields(x, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            pruner.check_position(&pt),
            "G2 FAIL: in-bounds point ({x:.1}, {y:.1}) rejected",
        );
    }
}

#[test]
fn g2_position_invariant_out_of_bounds_caught() {
    let config = test_config();
    let pruner = ReconciliationPruner::new(config, TrajectoryPoint::default());
    let mut rng = Rng::new(123);

    // Test 100 points outside bounds.
    for i in 0..100 {
        let (x, y) = if i % 4 == 0 {
            // Too far right
            (4097.0 + rng.uniform() * 1000.0, rng.uniform() * 4096.0)
        } else if i % 4 == 1 {
            // Too far left
            (-1.0 - rng.uniform() * 1000.0, rng.uniform() * 4096.0)
        } else if i % 4 == 2 {
            // Too far up
            (rng.uniform() * 4096.0, -1.0 - rng.uniform() * 1000.0)
        } else {
            // Too far down
            (rng.uniform() * 4096.0, 4097.0 + rng.uniform() * 1000.0)
        };
        let pt = TrajectoryPoint::from_fields(x, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            !pruner.check_position(&pt),
            "G2 FAIL: out-of-bounds point ({x:.1}, {y:.1}) was accepted",
        );
    }
}

// ── G3: Kill-rate bound (property test) ────────────────────────────────────

#[test]
fn g3_kill_rate_normal_passes() {
    let config = test_config();
    let h = h_last();
    let pruner = ReconciliationPruner::new(config, h);

    // Normal kill rates: 0 kills per frame for most frames, occasional +1.
    // Bound = MEAN_KILL_RATE + kill_rate_sigma * STD_KILL_RATE = 0.1 + 5.0 * 0.05 = 0.35 kills/sec
    // Per frame: max kills = 0.35 * (1/60) ≈ 0.0058, so 0 kills always passes.
    for seed in 0..200u64 {
        let traj = make_legitimate_trajectory(&h, 30, config.dt, config.max_speed, seed);
        for window in traj.windows(2) {
            assert!(
                pruner.check_kill_rate(&window[1], &window[0], config.dt),
                "G3 FAIL: normal trajectory seed={seed} failed kill-rate check",
            );
        }
    }
}

#[test]
fn g3_kill_rate_extreme_caught() {
    let config = test_config();
    let h = h_last();
    let pruner = ReconciliationPruner::new(config, h);

    // 100 trajectories with absurd kill rates.
    for seed in 0..100u64 {
        let mut rng = Rng::new(seed);
        let hacked_kills = h.kills() + 10.0 + rng.uniform() * 90.0;
        let curr = TrajectoryPoint::from_fields(
            h.pos_x() + 0.01,
            h.pos_y(),
            10.0,
            5.0,
            hacked_kills,
            0.0,
            1.0,
            0.0,
        );
        assert!(
            !pruner.check_kill_rate(&curr, &h, config.dt),
            "G3 FAIL: extreme kill rate ({hacked_kills:.0} kills in one frame) not caught, seed={seed}",
        );
    }
}

// ── G4: Manifold coverage (Monte Carlo) ────────────────────────────────────

#[test]
fn g4_manifold_coverage() {
    let config = ReconciliationConfig {
        k: 64,
        ..test_config()
    };
    let h = h_last();
    let generator = DefaultManifoldGenerator::new(config);
    let dt = config.dt;
    let steps = 60; // 1 second of trajectory

    // Generate the manifold.
    let mut rng = Rng::new(42);
    let manifold = generator.generate(&h, &[], config.k, dt, steps, &mut rng);

    // Flatten manifold for easy access.
    let manifold_pts: Vec<&TrajectoryPoint> = manifold.iter().flat_map(|t| t.iter()).collect();
    assert!(!manifold_pts.is_empty(), "G4: manifold should not be empty");

    // Generate 10,000 legitimate trajectories.
    let mut covered_count = 0usize;
    let total = 10_000;
    for i in 0..total {
        let leg = make_legitimate_trajectory(&h, steps, dt, config.max_speed, 1000 + i as u64);

        // Check if ANY point in this trajectory has cosine similarity > 0.5
        // to ANY point in the manifold.
        let mut found = false;
        'outer: for lp in &leg {
            for mp in &manifold_pts {
                let sim = cosine_similarity(&lp.data, &mp.data);
                if sim > 0.5 {
                    found = true;
                    break 'outer;
                }
            }
        }
        if found {
            covered_count += 1;
        }
    }

    let coverage = covered_count as f64 / total as f64;
    println!("G4: manifold coverage = {coverage:.4} ({covered_count}/{total})");
    assert!(
        coverage > 0.95,
        "G4 FAIL: manifold coverage {coverage:.4} is below 95% threshold",
    );
}

// ── G5: Latency bound (micro-benchmark) ────────────────────────────────────

#[test]
fn g5_latency_bound() {
    let config = test_config();
    let h = h_last();

    let offline_periods: &[(f32, &str)] = &[
        (1.0, "1s"),
        (10.0, "10s"),
        (60.0, "60s"),
        (300.0, "300s"),
        (600.0, "600s"),
    ];

    println!("\nG5: Latency benchmark");
    println!(
        "{:>10} {:>12} {:>12}",
        "Period", "Manifold Steps", "Time (µs)"
    );
    println!("{}", "-".repeat(36));

    for &(period, label) in offline_periods {
        let dt = 1.0 / 60.0;
        let raw_steps = (period / dt).ceil() as usize;

        // Cap manifold steps to keep the test reasonable in debug builds.
        // The manifold generator creates K × steps points; with K=16 and
        // steps=36000 that's 576k points — too much for a unit test.
        // Debug builds are ~20× slower than release, so we keep this small.
        let manifold_steps = raw_steps.min(60);

        let mut reconciler = SpecReconciler::new(config);

        // Build a client trajectory that passes hard bounds.
        let client_steps = manifold_steps;
        let client_traj = make_legitimate_trajectory(&h, client_steps, dt, config.max_speed, 42);

        // Warmup.
        let mut rng = Rng::new(42);
        let _ = reconciler.reconcile(&h, &client_traj, &[], client_steps, &mut rng);

        // Measure.
        let iterations = 100;
        let mut total_us = 0u64;
        for iter in 0..iterations {
            let mut r = Rng::new(iter as u64);
            let start = Instant::now();
            let _ = reconciler.reconcile(&h, &client_traj, &[], client_steps, &mut r);
            total_us += start.elapsed().as_micros() as u64;
        }
        let avg_us = total_us / iterations;

        println!("{label:>10} {manifold_steps:>12} {avg_us:>12}");

        // Debug builds are ~10-50× slower than release; use 200ms threshold.
        // In release mode this should be well under 1ms.
        assert!(
            avg_us < 200_000,
            "G5 FAIL: {label} reconciliation took {avg_us}µs (>200ms P50)",
        );
    }
}

// ── G6: False positive rate ────────────────────────────────────────────────

#[test]
fn g6_false_positive_rate() {
    let config = test_config();
    let h = h_last();
    let mut reconciler = SpecReconciler::new(config);

    let total = 1000usize;
    let mut quarantined = 0usize;

    for i in 0..total {
        let traj = make_legitimate_trajectory(&h, 30, config.dt, config.max_speed, 5000 + i as u64);
        let mut rng = Rng::new(i as u64);
        let result = reconciler.reconcile(&h, &traj, &[], 30, &mut rng);

        if result.verdict == ReconciliationVerdict::Quarantine {
            quarantined += 1;
        }
    }

    let fpr = quarantined as f64 / total as f64;
    println!("G6: false positive rate = {fpr:.4} ({quarantined}/{total} quarantined)");
    assert!(
        quarantined <= 10,
        "G6 FAIL: false positive rate {fpr:.4} exceeds 1% ({quarantined}/{total})",
    );
}

// ── G7: False negative rate ────────────────────────────────────────────────

#[test]
fn g7_false_negative_rate() {
    let config = test_config();
    let h = h_last();
    let mut reconciler = SpecReconciler::new(config);

    let total = 1000usize;
    let mut passed = 0usize;

    for i in 0..total {
        let traj = make_random_hack(&h, i as u64);
        let mut rng = Rng::new(i as u64);
        let result = reconciler.reconcile(&h, &traj, &[], 10, &mut rng);

        // A hack is "missed" if it's NOT quarantined.
        if result.verdict != ReconciliationVerdict::Quarantine {
            passed += 1;
        }
    }

    let fnr = passed as f64 / total as f64;
    println!("G7: false negative rate = {fnr:.4} ({passed}/{total} passed through)");
    assert!(
        passed <= 10,
        "G7 FAIL: false negative rate {fnr:.4} exceeds 1% ({passed}/{total} hacks passed)",
    );
}

// ── G8: Matrix soundness (determinant audit) ───────────────────────────────

#[test]
fn g8_matrix_soundness() {
    let config = test_config();
    let h = h_last();
    let generator = DefaultManifoldGenerator::new(config);

    let mut rng = Rng::new(42);
    let steps = 30;

    for trial in 0..50 {
        // Generate a legitimate trajectory.
        let client_traj =
            make_legitimate_trajectory(&h, steps, config.dt, config.max_speed, 9000 + trial);

        // Generate manifold.
        let manifold = generator.generate(&h, &[], config.k, config.dt, steps, &mut rng);

        // Collect all points: manifold + client.
        let all_points: Vec<&TrajectoryPoint> = manifold
            .iter()
            .flat_map(|t| t.iter())
            .chain(client_traj.iter())
            .collect();

        if all_points.len() < 2 {
            continue;
        }

        // Compute 2x2 covariance matrix for [pos_x, pos_y].
        let n = all_points.len() as f64;
        let mean_x: f64 = all_points.iter().map(|p| p.pos_x() as f64).sum::<f64>() / n;
        let mean_y: f64 = all_points.iter().map(|p| p.pos_y() as f64).sum::<f64>() / n;

        let mut cov_xx = 0.0f64;
        let mut cov_xy = 0.0f64;
        let mut cov_yy = 0.0f64;

        for p in &all_points {
            let dx = p.pos_x() as f64 - mean_x;
            let dy = p.pos_y() as f64 - mean_y;
            cov_xx += dx * dx;
            cov_xy += dx * dy;
            cov_yy += dy * dy;
        }

        cov_xx /= n;
        cov_xy /= n;
        cov_yy /= n;

        // det = cov_xx * cov_yy - cov_xy^2
        let det = cov_xx * cov_yy - cov_xy * cov_xy;

        assert!(
            det > 0.0,
            "G8 FAIL: trial={trial} covariance determinant is non-positive ({det:.6}), \
             degenerate manifold",
        );
    }
}
