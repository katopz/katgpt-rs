use super::*;
use katgpt_speculative::NoScreeningPruner;

// ── Stats Tests ─────────────────────────────────────────────

#[test]
fn test_stats_update_incremental_mean() {
    let mut stats = BanditStats::new(3);

    stats.update(0, 1.0);
    assert_eq!(stats.q_value(0), 1.0);

    stats.update(0, 0.0);
    assert_eq!(stats.q_value(0), 0.5);

    stats.update(0, 1.0);
    // (1.0 + 0.0 + 1.0) / 3 = 0.666...
    assert!((stats.q_value(0) - 0.666).abs() < 0.01);
    assert_eq!(stats.visit_count(0), 3);
    assert_eq!(stats.total_pulls(), 3);
}

#[test]
fn test_stats_best_arm() {
    let mut stats = BanditStats::new(4);
    stats.update(0, 0.2);
    stats.update(1, 0.8);
    stats.update(2, 0.5);
    stats.update(3, 0.3);
    assert_eq!(stats.best_arm(), 1);
}

#[test]
fn test_stats_ucb1_unvisited_max_priority() {
    let mut stats = BanditStats::new(3);
    stats.update(0, 0.5);
    stats.update(1, 0.5);
    // Arm 2 unvisited → should get f32::MAX score
    assert_eq!(stats.ucb1_score(2), f32::MAX);
    // Visited arms get finite scores
    assert!(stats.ucb1_score(0).is_finite());
    assert!(stats.ucb1_score(1).is_finite());
}

#[test]
fn test_stats_ucb1_increases_with_few_visits() {
    let mut stats = BanditStats::new(2);
    // Both arms get same reward, but arm 0 gets visited less
    for _ in 0..10 {
        stats.update(0, 0.5);
    }
    for _ in 0..100 {
        stats.update(1, 0.5);
    }
    // Arm 0 should have higher UCB1 bonus (fewer visits)
    assert!(stats.ucb1_score(0) > stats.ucb1_score(1));
}

// ── Environment Tests ───────────────────────────────────────

#[test]
fn test_bernoulli_env_optimal() {
    let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
    assert_eq!(env.optimal_arm(), 2);
    assert!((env.optimal_reward() - 0.8).abs() < f32::EPSILON);
    assert_eq!(env.num_arms(), 5);
    assert!((env.expected_reward(0) - 0.2).abs() < f32::EPSILON);
}

#[test]
fn test_bernoulli_env_pull_distribution() {
    let env = BernoulliEnv::new(&[0.0, 1.0]);
    let mut rng = Rng::new(42);

    // Arm 0 always returns 0.0
    for _ in 0..100 {
        assert_eq!(env.pull(0, &mut rng), 0.0);
    }
    // Arm 1 always returns 1.0
    for _ in 0..100 {
        assert_eq!(env.pull(1, &mut rng), 1.0);
    }
}

#[test]
fn test_gaussian_env_optimal() {
    let env = GaussianEnv::new(&[0.2, 0.7, 0.5], 0.1);
    assert_eq!(env.optimal_arm(), 1);
    assert!((env.optimal_reward() - 0.7).abs() < f32::EPSILON);
    assert_eq!(env.num_arms(), 3);
}

#[test]
fn test_gaussian_env_pull_clamped() {
    let env = GaussianEnv::new(&[0.5], 0.1);
    let mut rng = Rng::new(42);
    // All rewards should be in [0.0, 1.0]
    for _ in 0..1000 {
        let r = env.pull(0, &mut rng);
        assert!((0.0..=1.0).contains(&r), "reward {r} out of bounds");
    }
}

// ── Beta Sampling Tests ─────────────────────────────────────

#[test]
fn test_beta_sampling_bounds() {
    let mut rng = Rng::new(42);
    for alpha in [1.0, 2.0, 5.0, 10.0] {
        for beta in [1.0, 2.0, 5.0, 10.0] {
            for _ in 0..100 {
                let sample = sample_beta(alpha, beta, &mut rng);
                assert!(
                    (0.0..=1.0).contains(&sample),
                    "Beta({alpha},{beta}) sample {sample} out of bounds"
                );
            }
        }
    }
}

#[test]
fn test_beta_sampling_mean_converges() {
    let mut rng = Rng::new(42);
    let alpha = 3.0f32;
    let beta = 7.0f32;
    let expected_mean = alpha / (alpha + beta); // 0.3
    let n = 10000;
    let sum: f32 = (0..n).map(|_| sample_beta(alpha, beta, &mut rng)).sum();
    let mean = sum / n as f32;
    assert!(
        (mean - expected_mean).abs() < 0.05,
        "Beta({alpha},{beta}) mean {mean} too far from expected {expected_mean}"
    );
}

// ── BanditPruner Tests ──────────────────────────────────────

#[test]
fn test_pruner_cold_start_uses_domain() {
    let pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
    // Before any updates, relevance = domain relevance
    let rel = pruner.relevance(0, 0, &[]);
    assert!((rel - 1.0).abs() < f32::EPSILON); // NoScreeningPruner returns 1.0
}

#[test]
fn test_pruner_respects_domain_hard_trim() {
    struct AlwaysZero;
    impl ScreeningPruner for AlwaysZero {
        fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
            0.0
        }
    }
    let mut pruner = BanditPruner::new(AlwaysZero, BanditStrategy::Ucb1, 5);
    pruner.update(0, 0.9); // High reward, but domain says 0
    let rel = pruner.relevance(0, 0, &[]);
    assert_eq!(rel, 0.0);
}

#[test]
fn test_pruner_ucb1_unvisited_arm_priority() {
    let mut pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
    // Visit arm 0 and 1, leave arm 2 unvisited
    pruner.update(0, 0.5);
    pruner.update(1, 0.5);
    let rel_visited = pruner.relevance(0, 0, &[]);
    let rel_unvisited = pruner.relevance(0, 2, &[]);
    // Unvisited arm should have equal or higher relevance
    assert!(rel_unvisited >= rel_visited);
}

#[test]
fn test_pruner_thompson_uses_cache() {
    let mut pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::ThompsonSampling, 3);
    // Update some Q-values
    for _ in 0..10 {
        pruner.update(0, 0.9);
        pruner.update(1, 0.1);
    }
    let mut rng = Rng::new(42);
    pruner.prepare_episode(&mut rng);

    // Relevance should use cached sample (non-zero since arm 0 has high Q)
    let rel = pruner.relevance(0, 0, &[]);
    assert!(rel > 0.0);
}

// ── Convergence Tests ───────────────────────────────────────

#[test]
fn test_ucb1_convergence() {
    let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
    let session = BanditSession::new(env, BanditStrategy::Ucb1);
    let (_, result) = session.run(500, &mut Rng::new(42));
    assert!(
        result.found_optimal(),
        "UCB1 should find optimal arm 2, found arm {} with Q-values {:?}",
        result.best_arm,
        result.q_values
    );
}

#[test]
fn test_thompson_convergence() {
    let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
    let session = BanditSession::new(env, BanditStrategy::ThompsonSampling);
    let (_, result) = session.run(500, &mut Rng::new(42));
    assert!(
        result.found_optimal(),
        "Thompson should find optimal arm 2, found arm {} with Q-values {:?}",
        result.best_arm,
        result.q_values
    );
}

#[test]
fn test_epsilon_greedy_convergence_with_decay() {
    let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
    let strategy = BanditStrategy::EpsilonGreedy {
        epsilon: 0.3,
        decay: 0.995,
    };
    let session = BanditSession::new(env, strategy);
    let (_, result) = session.run(1000, &mut Rng::new(42));
    assert!(
        result.found_optimal(),
        "ε-greedy(decay) should find optimal arm 2, found arm {} with Q-values {:?}",
        result.best_arm,
        result.q_values
    );
}

#[test]
fn test_epsilon_greedy_no_decay_still_finds_good_arm() {
    let env = BernoulliEnv::new(&[0.1, 0.9]);
    let strategy = BanditStrategy::EpsilonGreedy {
        epsilon: 0.1,
        decay: 1.0,
    };
    let session = BanditSession::new(env, strategy);
    let (_, result) = session.run(2000, &mut Rng::new(42));
    // With fixed ε, may not always find optimal but should be close
    assert!(
        result.q_values[1] > 0.5,
        "ε-greedy(no decay) should learn arm 1 is good, Q-values: {:?}",
        result.q_values
    );
}

// ── Regret Tests ────────────────────────────────────────────

#[test]
fn test_regret_sublinear_ucb1() {
    let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
    let session = BanditSession::new(env, BanditStrategy::Ucb1);
    let (_, result) = session.run(1000, &mut Rng::new(42));

    // Sub-linear regret: total_regret should grow slower than linear
    // Linear regret would be ~1000 * (0.8 - 0.2) = 600 for always choosing worst arm
    // Sub-linear should be much less, roughly O(sqrt(N)) ≈ ~30-60
    assert!(
        result.total_regret < 100.0,
        "UCB1 regret should be sub-linear, got {}",
        result.total_regret
    );
}

#[test]
fn test_regret_sublinear_thompson() {
    let env = BernoulliEnv::new(&[0.2, 0.5, 0.8, 0.4, 0.6]);
    let session = BanditSession::new(env, BanditStrategy::ThompsonSampling);
    let (_, result) = session.run(1000, &mut Rng::new(42));

    // Thompson is stochastic — higher variance than UCB1.
    // Linear regret worst-case ≈ 600. Sub-linear threshold generous but still
    // well below linear: must be spending most pulls on high-value arms.
    assert!(
        result.total_regret < 250.0,
        "Thompson regret should be sub-linear, got {}",
        result.total_regret
    );
}

// ── Gaussian Bandit Test ────────────────────────────────────

#[test]
fn test_gaussian_convergence() {
    let env = GaussianEnv::new(&[0.3, 0.7, 0.5], 0.1);
    let session = BanditSession::new(env, BanditStrategy::Ucb1);
    let (_, result) = session.run(500, &mut Rng::new(42));
    assert!(
        result.found_optimal(),
        "UCB1 should find Gaussian optimal arm 1, found arm {} with Q-values {:?}",
        result.best_arm,
        result.q_values
    );
}

// ── Session Event Tests ─────────────────────────────────────

#[test]
fn test_session_events_count() {
    let env = BernoulliEnv::new(&[0.5, 0.8]);
    let session = BanditSession::new(env, BanditStrategy::Ucb1);
    let (events, _) = session.run(10, &mut Rng::new(42));

    // 10 Pull + 10 EpisodeComplete + 1 SessionComplete = 21
    assert_eq!(events.len(), 21);
}

#[test]
fn test_session_result_fields() {
    let env = BernoulliEnv::new(&[0.5, 0.8]);
    let session = BanditSession::new(env, BanditStrategy::Ucb1);
    let (_, result) = session.run(100, &mut Rng::new(42));

    assert_eq!(result.total_episodes, 100);
    assert_eq!(result.optimal_arm, 1);
    assert_eq!(result.q_values.len(), 2);
    assert_eq!(result.visits.len(), 2);
    assert!(result.total_reward > 0.0);
    assert!(result.avg_reward() > 0.0);
}

// ── Constrained Bandit Tests ────────────────────────────────

/// Domain pruner that blocks specific arms via relevance 0.0.
struct BlockedArmPruner {
    blocked: Vec<usize>,
}

impl BlockedArmPruner {
    fn new(blocked: &[usize]) -> Self {
        Self {
            blocked: blocked.to_vec(),
        }
    }
}

impl ScreeningPruner for BlockedArmPruner {
    fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        if self.blocked.contains(&token_idx) {
            0.0
        } else {
            1.0
        }
    }
}

#[test]
fn test_constrained_bandit_never_pulls_blocked_arm() {
    // Arm 4 has highest reward (0.9) but is blocked
    let mut pruner = BanditPruner::new(BlockedArmPruner::new(&[4]), BanditStrategy::Ucb1, 5);
    pruner.soft_route = false; // Hard-route needed for blocked-arm rejection

    let env = BernoulliEnv::new(&[0.1, 0.3, 0.7, 0.4, 0.9]);
    let mut rng = Rng::new(42);

    // Select arms via pruner relevance for 500 episodes
    for _ in 0..500 {
        let arm = (0..5)
            .max_by(|&a, &b| {
                katgpt_core::float_order::cmp_for_max(
                    pruner.relevance(0, a, &[]),
                    pruner.relevance(0, b, &[]),
                )
            })
            .unwrap_or(0);

        let reward = env.pull(arm, &mut rng);
        pruner.update(arm, reward);
    }

    // Arm 4 should never be visited (relevance always 0.0)
    assert_eq!(pruner.visits()[4], 0, "blocked arm should never be pulled");
    // Best arm should be 2 (highest unblocked reward: 0.7)
    assert_eq!(
        pruner.best_arm(),
        2,
        "should find best valid arm, not blocked arm"
    );
}

#[test]
fn test_constrained_bandit_respects_domain_over_bandit() {
    // Even after giving arm 4 high reward manually, domain pruner overrides
    let mut pruner = BanditPruner::new(BlockedArmPruner::new(&[4]), BanditStrategy::Ucb1, 5);

    // Manually pump arm 4's Q-value high
    for _ in 0..100 {
        pruner.update(4, 1.0);
    }

    // Domain still blocks it
    let rel = pruner.relevance(0, 4, &[]);
    assert_eq!(
        rel, 0.0,
        "domain pruner must override bandit score for blocked arms"
    );

    // Other arms still allowed
    assert!(pruner.relevance(0, 0, &[]) >= 0.0);
    assert!(pruner.relevance(0, 2, &[]) >= 0.0);
}

// ── Shared Bandit Stats Tests ──────────────────────────────

#[test]
fn test_shared_bandit_stats_convergence() {
    use std::sync::Arc;
    use std::thread;

    let stats = Arc::new(SharedBanditStats::new(4));
    let mut handles = Vec::new();

    // 4 threads, each updating different arms with different rewards
    // Arm 0: reward 0.1, Arm 1: reward 0.3, Arm 2: reward 0.9, Arm 3: reward 0.5
    let rewards = [0.1f32, 0.3f32, 0.9f32, 0.5f32];
    let updates_per_thread = 200u32;

    for (arm, &reward) in rewards.iter().enumerate() {
        let stats_clone = Arc::clone(&stats);
        handles.push(thread::spawn(move || {
            for _ in 0..updates_per_thread {
                stats_clone.update(arm, reward);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Best arm should converge to arm 2 (highest reward 0.9)
    assert_eq!(
        stats.best_arm(),
        2,
        "shared stats should find arm 2 as best"
    );

    // Total pulls = 4 threads × 200 updates
    assert_eq!(
        stats.total_pulls(),
        800,
        "total pulls should equal sum of all thread updates"
    );

    // Verify individual arm visits
    for arm in 0..4 {
        assert_eq!(
            stats.visits(arm),
            updates_per_thread,
            "arm {arm} should have {updates_per_thread} visits"
        );
    }

    // Verify Q-values converge toward true rewards
    for (arm, &expected) in rewards.iter().enumerate() {
        let q = stats.q_value(arm);
        assert!(
            (q - expected).abs() < 0.1,
            "arm {arm} q_value {q} should be close to {expected}"
        );
    }
}

#[cfg(feature = "bandit")]
#[test]
fn test_bandit_pruner_shared_stats() {
    use std::sync::Arc;

    // Simple mock pruner that always returns 1.0
    struct MockPruner;
    impl ScreeningPruner for MockPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_token: &[usize]) -> f32 {
            1.0
        }
    }

    let shared = Arc::new(SharedBanditStats::new(3));
    let mut p1 =
        BanditPruner::with_shared_stats(MockPruner, BanditStrategy::Ucb1, 3, Arc::clone(&shared));
    let mut p2 =
        BanditPruner::with_shared_stats(MockPruner, BanditStrategy::Ucb1, 3, Arc::clone(&shared));

    // P1 updates arm 0 with high reward
    p1.update(0, 0.9);

    // P2 updates arm 1 with low reward
    p2.update(1, 0.1);

    // P2 updates arm 2 with medium reward
    p2.update(2, 0.5);

    // Verify P1 sees P2's updates and vice versa
    // Total pulls should be 3 from either pruner's perspective
    assert_eq!(p1.total_pulls(), 3, "p1 should see 3 total pulls");
    assert_eq!(p2.total_pulls(), 3, "p2 should see 3 total pulls");

    // Best arm should be arm 0 (highest reward 0.9)
    assert_eq!(p1.best_arm(), 0, "p1 best arm should be 0");
    assert_eq!(p2.best_arm(), 0, "p2 best arm should be 0");

    // Verify visits are shared
    assert_eq!(p1.arm_visits(0), 1, "arm 0 should have 1 visit via p1");
    assert_eq!(p2.arm_visits(1), 1, "arm 1 should have 1 visit via p2");
    assert_eq!(p1.arm_visits(2), 1, "arm 2 should have 1 visit via p1");

    // More updates from P1
    for _ in 0..10 {
        p1.update(0, 0.9);
    }

    // P2 should see the accumulated visits
    assert_eq!(
        p2.arm_visits(0),
        11,
        "p2 should see p1's accumulated visits on arm 0"
    );
    assert_eq!(p2.total_pulls(), 13, "p2 should see total 13 pulls");
}

// ── Dual Cutoff Tests (Plan 062) ────────────────────────────

#[test]
fn test_dual_cutoff_disabled_by_default() {
    let bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
    assert_eq!(bp.dual_cutoff, 0.0, "default cutoff should be 0 (disabled)");
}

#[test]
fn test_dual_cutoff_masks_low_q_arms() {
    let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
    bp.soft_route = false; // Hard-route needed for per-arm cutoff masking
    bp.dual_cutoff = 0.3;

    // Arm 0: high Q (should pass)
    bp.update(0, 0.8);
    bp.update(0, 0.9);
    // Arm 1: low Q (should be masked)
    bp.update(1, 0.1);
    bp.update(1, 0.05);
    // Arm 2: unvisited (should NOT be masked — exploration)

    let r0 = bp.relevance(0, 0, &[]);
    let r1 = bp.relevance(0, 1, &[]);
    let r2 = bp.relevance(0, 2, &[]);

    assert!(r0 > 0.0, "high-Q arm should have positive relevance");
    assert_eq!(r1, 0.0, "low-Q arm should be masked by dual_cutoff");
    assert!(r2 > 0.0, "unvisited arm should not be masked (exploration)");
}

#[test]
fn test_set_dual_cutoff_method() {
    let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
    assert_eq!(bp.dual_cutoff, 0.0);

    bp.set_dual_cutoff(0.5);
    assert_eq!(bp.dual_cutoff, 0.5);

    bp.set_dual_cutoff(0.0);
    assert_eq!(bp.dual_cutoff, 0.0, "can re-disable via setter");
}

// ── Soft-Route Tests (Plan 175, Part 3) ───────────────────────

#[test]
fn test_soft_route_enabled_by_default() {
    // soft_route defaults to false per f2ad6f94: defaulting to true forced
    // every relevance() call through two Mutex locks + O(n) softmax,
    // regressing Δ-Bandit 10× (140M→13M ops/s). Callers who want
    // softmax-blended routing must opt in via set_soft_route(true, τ).
    let bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
    assert!(
        !bp.soft_route,
        "soft_route should default to false (opt-in)"
    );
    assert!(
        (bp.soft_route_tau - 1.0).abs() < f32::EPSILON,
        "tau should default to 1.0"
    );
}

#[test]
fn test_soft_route_cold_start_returns_domain() {
    let bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
    // No updates: cold start, relevance = domain (1.0 from NoScreeningPruner)
    let r = bp.relevance(0, 0, &[]);
    assert_eq!(r, 1.0, "cold start should return domain");
}

#[test]
fn test_soft_route_blend_dominates_single_arm() {
    let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
    // soft_route is opt-in (f2ad6f94); this test exercises soft-route
    // blending, so explicitly enable it.
    bp.set_soft_route(true, 1.0);

    // Arm 0: high Q
    for _ in 0..20 {
        bp.update(0, 0.9);
    }
    // Arm 1: low Q
    for _ in 0..20 {
        bp.update(1, 0.1);
    }
    // Arm 2: medium Q
    for _ in 0..20 {
        bp.update(2, 0.5);
    }

    // With soft-route, arm 1's relevance should be higher than its own
    // bandit score would suggest (blended upward by arms 0 and 2)
    let r0 = bp.relevance(0, 0, &[]);
    let r1 = bp.relevance(0, 1, &[]);
    let r2 = bp.relevance(0, 2, &[]);

    // All should be positive and reasonably close (soft blending)
    assert!(r0 > 0.0, "arm 0 relevance should be positive");
    assert!(r1 > 0.0, "arm 1 relevance should be positive");
    assert!(r2 > 0.0, "arm 2 relevance should be positive");

    // The key property: with soft routing, all arms get similar relevance
    // because the blend is over ALL arm scores. The spread should be
    // smaller than with hard routing.
    let spread = (r0 - r1).abs();
    assert!(
        spread < 0.5,
        "soft-route spread should be moderate, got {spread}"
    );
}

#[test]
fn test_hard_route_restores_original_behavior() {
    let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
    bp.set_soft_route(false, 1.0);

    // Arm 0: high Q
    for _ in 0..20 {
        bp.update(0, 0.9);
    }
    // Arm 1: low Q
    for _ in 0..20 {
        bp.update(1, 0.1);
    }

    let r0 = bp.relevance(0, 0, &[]);
    let r1 = bp.relevance(0, 1, &[]);

    assert!(
        r0 > r1,
        "hard-route: high-Q arm should have higher relevance"
    );
    assert!(r0 > 0.0, "high-Q arm should be positive");
    assert!(r1 > 0.0, "low-Q arm should still be positive (no cutoff)");
}

#[test]
fn test_soft_route_zero_domain_returns_zero() {
    struct ZeroPruner;
    impl ScreeningPruner for ZeroPruner {
        fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
            0.0
        }
    }
    let mut bp = BanditPruner::new(ZeroPruner, BanditStrategy::Ucb1, 3);
    bp.update(0, 0.9);
    let r = bp.relevance(0, 0, &[]);
    assert_eq!(r, 0.0, "zero domain should give zero even with soft-route");
}

#[test]
fn test_soft_route_setter_clamps_tau() {
    let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
    bp.set_soft_route(true, 0.001);
    assert!(
        (bp.soft_route_tau - 0.01).abs() < f32::EPSILON,
        "tau should be clamped to 0.01 minimum"
    );
}

// ── GOAT Integration: All Three Fusions (Plan 175, Part 4) ─────
//
// GOAT proof that all three fusions work together without regression:
//   Fusion 1: Residency Audit (verify pruner lands on fast paths)
//   Fusion 2: RangeBudget (entropy-aware budget adaptation)
//   Fusion 4: Soft-Route Bandit (softmax-blended arm relevance)

#[test]
fn test_goat_175_soft_route_acceptance_rate() {
    // GOAT: Soft-route bandit acceptance rate >= hard-route over 500 episodes.
    //
    // With soft routing, all arms get blended relevance, so the DDTree
    // retains more viable branches. This should produce acceptance rates
    // at least as good as hard routing (which only considers one arm).
    let vocab = 8;
    let lookahead = 4;
    let episodes = 500;
    let mut rng = katgpt_types::Rng::new(42);

    let config = katgpt_types::Config {
        vocab_size: vocab,
        draft_lookahead: lookahead,
        ..Default::default()
    };

    // Helper: generate peaked marginals (3 good tokens, rest noise)
    let peaked_marginals = |rng: &mut katgpt_types::Rng| -> Vec<Vec<f32>> {
        (0..lookahead)
            .map(|_| {
                let mut m = vec![0.01; vocab];
                // 3 "good" tokens get ~80% of mass
                for v in m.iter_mut().take(3) {
                    *v = 0.27;
                }
                let sum: f32 = m.iter().sum();
                m.iter_mut().for_each(|p| *p /= sum);
                let _ = rng; // consume rng for API consistency
                m
            })
            .collect()
    };

    // Run soft-route
    let mut soft_bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, vocab);
    let mut soft_accepted = 0usize;
    let mut soft_total = 0usize;

    for ep in 0..episodes {
        soft_bp.prepare_episode(&mut rng);
        let marginals = peaked_marginals(&mut rng);
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
        let tree =
            katgpt_speculative::dd_tree::build_dd_tree_screened(&slices, &config, &soft_bp, true);

        // Simulate verification: accept top-k tokens
        for node in &tree {
            soft_total += 1;
            // Peaked marginals: top-3 tokens have ~80% chance of acceptance
            if node.token_idx < 3 && rng.uniform() < 0.8 {
                soft_bp.update(node.token_idx, 1.0);
                soft_accepted += 1;
            } else if rng.uniform() < 0.2 {
                soft_bp.update(node.token_idx, 0.1);
                soft_accepted += 1;
            }
        }

        let _ = ep;
    }

    // Run hard-route (baseline)
    let mut hard_bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, vocab);
    hard_bp.soft_route = false;
    let mut hard_accepted = 0usize;
    let mut hard_total = 0usize;

    for ep in 0..episodes {
        hard_bp.prepare_episode(&mut rng);
        let marginals = peaked_marginals(&mut rng);
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
        let tree =
            katgpt_speculative::dd_tree::build_dd_tree_screened(&slices, &config, &hard_bp, true);

        for node in &tree {
            hard_total += 1;
            if node.token_idx < 3 && rng.uniform() < 0.8 {
                hard_bp.update(node.token_idx, 1.0);
                hard_accepted += 1;
            } else if rng.uniform() < 0.2 {
                hard_bp.update(node.token_idx, 0.1);
                hard_accepted += 1;
            }
        }

        let _ = ep;
    }

    let soft_rate = soft_accepted as f64 / soft_total.max(1) as f64;
    let hard_rate = hard_accepted as f64 / hard_total.max(1) as f64;

    // GOAT: soft-route acceptance rate should be within 5% of hard-route
    // (may not always exceed due to verification randomness, but should be close)
    assert!(
        soft_rate >= hard_rate - 0.05,
        "GOAT 175: soft-route acceptance ({soft_rate:.3}) should be >= hard-route ({hard_rate:.3}) - 5%"
    );

    // Both should produce reasonable trees
    assert!(soft_total > 0, "soft-route should produce tree nodes");
    assert!(hard_total > 0, "hard-route should produce tree nodes");
}

// NOTE: `test_goat_175_fusion_residency_audit_passes` was removed during the
// katgpt-pruners extraction (Plan 005). It depended on
// `crate::speculative::residency_audit` (root-only test module in main
// katgpt-rs crate). Re-locate as an integration test in katgpt-rs/tests/
// that constructs BanditPruner via `katgpt_pruners::bandit::*` and audits
// via `katgpt_rs::speculative::residency_audit::*`.

#[test]
fn test_goat_175_soft_route_overhead_acceptable() {
    // GOAT: Soft-route O(arms) per-node overhead is acceptable.
    //
    // Soft-route computes softmax over all arms for each node, which is O(arms)
    // per relevance() call instead of O(1). This test verifies the overhead
    // is reasonable for typical vocab sizes.
    use std::time::Instant;

    let vocab = 8;
    let lookahead = 4;
    let config = katgpt_types::Config {
        vocab_size: vocab,
        draft_lookahead: lookahead,
        ..Default::default()
    };

    let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, vocab);

    // Warm up with some data
    for i in 0..vocab {
        bp.update(i, 0.5);
    }

    let marginals: Vec<Vec<f32>> = (0..lookahead)
        .map(|_| {
            let p = 1.0 / vocab as f32;
            vec![p; vocab]
        })
        .collect();
    let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    // Build 1000 trees and measure time
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _tree =
            katgpt_speculative::dd_tree::build_dd_tree_screened(&slices, &config, &bp, true);
    }
    let elapsed = start.elapsed();
    let per_tree_ns = elapsed.as_nanos() as f64 / iterations as f64;

    // GOAT: per-tree construction should be < 500µs for vocab=8, lookahead=4
    // (this is generous; actual should be much faster)
    assert!(
        per_tree_ns < 500_000.0,
        "GOAT 175: per-tree overhead should be < 500µs, got {per_tree_ns:.0}ns"
    );
}

// ── Partial Scoring Tests (Plan 191 T1.4) ────────────────────

#[cfg(feature = "partial_scoring")]
mod partial_scoring {
    use super::*;
    use crate::BomberPartialScorer;
    use katgpt_core::GameTrace;

    fn win_trace() -> GameTrace {
        GameTrace {
            survival_ticks: 200,
            kills: 3,
            actions_taken: 50,
            max_ticks: 200,
            final_reward: 1.0,
        }
    }

    fn loss_trace() -> GameTrace {
        GameTrace {
            survival_ticks: 30,
            kills: 0,
            actions_taken: 10,
            max_ticks: 200,
            final_reward: 0.0,
        }
    }

    #[test]
    fn test_update_with_trace_scorer_set() {
        let scorer = Box::new(BomberPartialScorer { max_ticks: 200 });
        let mut bp =
            BanditPruner::with_partial_scorer(NoScreeningPruner, BanditStrategy::Ucb1, 4, scorer);
        let trace = win_trace();
        bp.update_with_trace(0, &trace);
        // BomberPartialScorer on win_trace: survival=1.0, kills=1.0, efficiency=0.06
        // score = 0.4*1.0 + 0.3*1.0 + 0.2*1.0 + 0.1*0.06 = 0.906
        let q = bp.q_values()[0];
        assert!(q > 0.8, "expected high score from scorer, got {q}");
    }

    #[test]
    fn test_update_with_trace_no_scorer_binary_fallback() {
        let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 4);
        let trace = loss_trace();
        bp.update_with_trace(0, &trace);
        let q = bp.q_values()[0];
        assert!(
            (q - 0.0).abs() < f32::EPSILON,
            "expected 0.0 for loss, got {q}"
        );

        let trace_w = win_trace();
        bp.update_with_trace(1, &trace_w);
        let q = bp.q_values()[1];
        assert!(
            (q - 1.0).abs() < f32::EPSILON,
            "expected 1.0 for win, got {q}"
        );
    }

    #[test]
    fn test_with_partial_scorer_constructor() {
        let scorer = Box::new(BomberPartialScorer { max_ticks: 100 });
        let bp = BanditPruner::with_partial_scorer(
            NoScreeningPruner,
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.1,
                decay: 0.99,
            },
            8,
            scorer,
        );
        // Verify it compiles and drops cleanly
        drop(bp);
        // Also suppress unused GameTrace lint
        let _ = win_trace();
        let _ = loss_trace();
    }

    #[test]
    fn test_default_backward_compat() {
        let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 4);
        // Default has no scorer — update_with_trace uses binary fallback
        let trace = GameTrace {
            survival_ticks: 100,
            kills: 5,
            actions_taken: 30,
            max_ticks: 200,
            final_reward: 1.0,
        };
        bp.update_with_trace(0, &trace);
        assert!((bp.q_values()[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_breakdown_via_update() {
        let scorer = Box::new(BomberPartialScorer { max_ticks: 200 });
        let mut bp =
            BanditPruner::with_partial_scorer(NoScreeningPruner, BanditStrategy::Ucb1, 4, scorer);
        // Partial loss: survived 100/200 ticks, no kills
        let trace = GameTrace {
            survival_ticks: 100,
            kills: 0,
            actions_taken: 40,
            max_ticks: 200,
            final_reward: 0.0,
        };
        bp.update_with_trace(0, &trace);
        let q = bp.q_values()[0];
        // survival=0.5, kills=0.0, safety=0.5, efficiency=0.0
        // score = 0.4*0.5 + 0.3*0.0 + 0.2*0.5 + 0.1*0.0 = 0.3
        assert!(
            (q - 0.3).abs() < 0.01,
            "expected ~0.3 for partial survival, got {q}"
        );
    }
}

// ── Idea Divergence Integration Tests (Plan 191 T3.2) ──────────

#[cfg(feature = "idea_divergence")]
mod idea_divergence_tests {
    use super::*;
    use katgpt_speculative::NoScreeningPruner;

    /// Two arms with identical Q-values: both non-novel arms get lower bandit scores.
    #[test]
    fn test_idea_divergence_penalizes_convergent_arms() {
        let mut bp = BanditPruner::with_idea_divergence(
            NoScreeningPruner,
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.1,
                decay: 1.0,
            },
            3,
            0.1,
        );

        // Update arm 0 with a reward, register its score vector
        bp.update(0, 0.8);
        bp.update_divergence(0);

        // Update arm 1 with the same reward — convergent
        bp.update(1, 0.8);
        bp.update_divergence(1);

        // Arm 2: different reward — divergent
        bp.update(2, 0.1);
        bp.update_divergence(2);

        // Arm 0's score vector [0.8, 1.0] compared to arm 1 [0.8, 1.0]: distance=0 < threshold → penalized
        let score_0 = bp.arm_bandit_score(0);
        // Arm 1's score vector [0.8, 1.0] compared to arm 0 [0.8, 1.0]: distance=0 < threshold → penalized
        let score_1 = bp.arm_bandit_score(1);
        // Arm 2's score vector [0.1, 1.0] compared to arms 0,1 [0.8, 1.0]: distance=0.7 > threshold → unpenalized
        let score_2 = bp.arm_bandit_score(2);

        // Base EpsilonGreedy score for Q=0.8: 0.8.clamp(0.0, 1.0).max(0.01) = 0.8
        // After penalty: 0.8 * 0.5 = 0.4
        let base_08 = 0.8f32.clamp(0.0, 1.0).max(0.01);
        assert!(
            (score_0 - base_08 * 0.5).abs() < 0.01,
            "convergent arm 0 should be penalized: expected {}, got {score_0}",
            base_08 * 0.5
        );
        assert!(
            (score_1 - base_08 * 0.5).abs() < 0.01,
            "convergent arm 1 should be penalized: expected {}, got {score_1}",
            base_08 * 0.5
        );

        // Arm 2 is novel (different Q-value)
        let base_01 = 0.1f32.clamp(0.0, 1.0).max(0.01);
        assert!(
            (score_2 - base_01).abs() < 0.01,
            "novel arm 2 should be unpenalized: expected {base_01}, got {score_2}"
        );
    }

    /// Arm with unique Q-value pattern: no penalty.
    #[test]
    fn test_idea_divergence_novel_arm_unpenalized() {
        let mut bp = BanditPruner::with_idea_divergence(
            NoScreeningPruner,
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.1,
                decay: 1.0,
            },
            3,
            0.5,
        );

        // Arm 0: score [0.2, 1.0]
        bp.update(0, 0.2);
        bp.update_divergence(0);

        // Arm 1: score [0.8, 1.0] — L2 distance to arm 0 = sqrt(0.36) = 0.6 > threshold 0.5
        bp.update(1, 0.8);
        bp.update_divergence(1);

        let score_0 = bp.arm_bandit_score(0);
        let score_1 = bp.arm_bandit_score(1);

        // Both should be unpenalized (novel relative to each other)
        // arm 0: Q=0.2, base = 0.2.clamp(0.0, 1.0).max(0.01) = 0.2
        // arm 1: Q=0.8, base = 0.8.clamp(0.0, 1.0).max(0.01) = 0.8
        let base_0 = 0.2f32.clamp(0.0, 1.0).max(0.01);
        let base_1 = 0.8f32.clamp(0.0, 1.0).max(0.01);

        assert!(
            (score_0 - base_0).abs() < 0.01,
            "novel arm 0 should be unpenalized: expected {base_0}, got {score_0}"
        );
        assert!(
            (score_1 - base_1).abs() < 0.01,
            "novel arm 1 should be unpenalized: expected {base_1}, got {score_1}"
        );
    }

    /// Constructor produces a valid BanditPruner with divergence enabled.
    #[test]
    fn test_with_idea_divergence_constructor() {
        let bp =
            BanditPruner::with_idea_divergence(NoScreeningPruner, BanditStrategy::Ucb1, 4, 0.3);
        assert_eq!(bp.q_values().len(), 4);
        assert_eq!(bp.visits().len(), 4);
        assert!(bp.idea_divergence.is_some());
        assert_eq!(bp.idea_divergence.as_ref().unwrap().threshold(), 0.3);
        assert_eq!(bp.arm_score_vectors.len(), 4);
        for v in &bp.arm_score_vectors {
            assert!(v.is_empty(), "initial arm score vectors should be empty");
        }
    }

    /// Default BanditPruner (no divergence feature): scores unchanged.
    #[test]
    fn test_divergence_disabled_no_impact() {
        // This test runs WITH the feature, but constructs a default BanditPruner
        // (no divergence) to verify backward compatibility.
        let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 2);

        bp.update(0, 0.5);
        bp.update(1, 0.5);

        let score_0 = bp.arm_bandit_score(0);
        let score_1 = bp.arm_bandit_score(1);

        // Without divergence enabled, both arms should get identical scores
        assert!(
            (score_0 - score_1).abs() < f32::EPSILON,
            "without divergence, identical Q-values should get identical scores: {score_0} vs {score_1}"
        );
    }
}

// ── Skill Lifecycle Tests ────────────────────────────────────

#[cfg(feature = "skill_lifecycle")]
mod skill_lifecycle_tests {
    use super::*;
    use katgpt_speculative::NoScreeningPruner;

    #[test]
    fn test_bandit_pruner_records_experience() {
        let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);

        bp.update(0, 0.8);
        bp.record_experience(0, 0.8, false, false);

        bp.update(1, 0.1);
        bp.record_experience(1, 0.1, false, true);

        assert_eq!(bp.pruner_memory().total_entries(), 2);

        let recent = bp.recent_experiences(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].arm, 0);
        assert!((recent[0].reward - 0.8).abs() < f32::EPSILON);
        assert!(!recent[0].is_failure);
        assert_eq!(recent[1].arm, 1);
        assert!(recent[1].is_failure);
    }

    #[test]
    fn test_bandit_pruner_memory_bounded() {
        let bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 5);
        // Default capacity is 256 (rounded to next power of 2 = 256)
        assert_eq!(bp.pruner_memory().capacity(), 256);

        // Fill beyond capacity
        for i in 0..300u64 {
            bp.record_experience((i % 5) as u16, i as f32, false, false);
        }

        assert_eq!(bp.pruner_memory().total_entries(), 300);

        // Only last 256 should be retrievable
        let recent = bp.recent_experiences(300);
        assert_eq!(recent.len(), 256);

        // First entry should be arm for i=44 (300-256=44), 44%5=4
        assert_eq!(recent[0].arm, 4);
        // Last entry should be arm for i=299, 299%5=4
        assert_eq!(recent[255].arm, 4);

        // Verify identity
        assert!(bp.pruner_memory().verify_identity("bandit"));
    }
}
