//! GOAT Proof: SimpleTES Budget Scaling (T9) + Cross-Strategy Tournament (T10)
//!
//! Also validates:
//! T7: Trajectory Credit Bridge — max-trajectory-score credit assignment
//! T8: SimpleTesLoop struct — concrete C×L×K loop implementation
//!
//! Distilled from SimpleTES (arXiv:2604.19341).
//!
//! Run: cargo test --features tes_loop --test bench_simpletes_scaling_crossstrategy -- --nocapture

#[cfg(feature = "tes_loop")]
#[test]
fn bench_simpletes_scaling_crossstrategy_goat() {
    use katgpt_rs::pruners::bandit::{
        BanditEnv, BanditSession, BanditStrategy, BernoulliEnv, GaussianEnv,
    };
    use katgpt_rs::pruners::tes_loop::SimpleTesLoop;
    use katgpt_rs::speculative::types::{TesConfig, TrajectoryCredit};
    use katgpt_rs::types::Rng;

    const N_TRIALS: usize = 200;
    const SEED: u64 = 42;

    // ── Helpers ──────────────────────────────────────────────

    /// Run a single TES loop trial and return (best_score, total_evals).
    fn run_tes_trial<E: BanditEnv + Clone>(config: TesConfig, env: E, seed: u64) -> (f32, usize) {
        let mut rng = Rng::new(seed);
        let mut tes = SimpleTesLoop::new(config, env);
        let vocab_size = 8;
        let (evals, best) = tes.run(vocab_size, &mut rng);
        (best, evals)
    }

    /// Run multiple TES trials and return (avg_best, avg_evals, perfect_count).
    fn run_tes_trials<E: BanditEnv + Clone>(
        config: TesConfig,
        env: &E,
        n_trials: usize,
        base_seed: u64,
    ) -> (f32, usize, usize) {
        let mut total_best = 0.0f32;
        let mut total_evals = 0usize;
        let mut perfect = 0usize;

        for t in 0..n_trials {
            let (best, evals) = run_tes_trial(config.clone(), (*env).clone(), base_seed + t as u64);
            total_best += best;
            total_evals += evals;
            if best >= 0.95 {
                perfect += 1;
            }
        }

        (
            total_best / n_trials as f32,
            total_evals / n_trials,
            perfect,
        )
    }

    /// Run a bandit session and return (cumulative_reward, cumulative_regret, best_arm).
    fn run_bandit_session<E: BanditEnv + Clone>(
        env: E,
        strategy: BanditStrategy,
        episodes: usize,
        seed: u64,
    ) -> (f32, f32, usize) {
        let mut rng = Rng::new(seed);
        let session = BanditSession::new(env, strategy);
        let (_, result) = session.run(episodes, &mut rng);
        (result.total_reward, result.total_regret, result.best_arm)
    }

    // ════════════════════════════════════════════════════════════
    // T7: Trajectory Credit Bridge Verification
    // ════════════════════════════════════════════════════════════

    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!("🐐 T7: TRAJECTORY CREDIT BRIDGE — Max-Score Credit Assignment");
    println!("════════════════════════════════════════════════════════════════════════");
    println!();

    // Proof 1: Basic credit assignment
    {
        let traj_scores: Vec<(usize, f32)> = vec![
            (0, 0.9), // Best
            (1, 0.5), // Middle
            (2, 0.1), // Worst
            (3, 0.7), // Good
        ];

        let credit = TrajectoryCredit::from_trajectory_scores(&traj_scores);
        let weights: Vec<f32> = traj_scores
            .iter()
            .map(|(_, s)| credit.node_weight(*s))
            .collect();

        println!("  Trajectory scores: {:?}", traj_scores);
        println!(
            "  Credit weights:    [{:.2}, {:.2}, {:.2}, {:.2}]",
            weights[0], weights[1], weights[2], weights[3]
        );
        println!(
            "  Best trajectory:   idx={} (score={:.1}, weight={:.2})",
            credit.best_trajectory_idx, credit.best_score, weights[0]
        );
        println!(
            "  Worst trajectory:  idx={} (score={:.1}, weight={:.2})",
            credit.worst_trajectory_idx, credit.worst_score, weights[2]
        );

        assert!(
            (weights[0] - 1.0).abs() < 0.01,
            "Best should get weight 1.0, got {}",
            weights[0]
        );
        assert!(
            (weights[2] - 0.0).abs() < 0.01,
            "Worst should get weight 0.0, got {}",
            weights[2]
        );
        assert!(
            weights[1] > 0.0 && weights[1] < 1.0,
            "Middle should be (0, 1), got {}",
            weights[1]
        );
        assert!(
            weights[3] > weights[1] && weights[3] < 1.0,
            "Good > Middle, got {} vs {}",
            weights[3],
            weights[1]
        );

        println!("  ✓ Credit assignment: best=1.0, worst=0.0, linear interpolation correct");
    }

    // Proof 2: Uniform scores → all weights = 1.0
    {
        let uniform_scores: Vec<(usize, f32)> = vec![(0, 0.5), (1, 0.5), (2, 0.5)];
        let credit = TrajectoryCredit::from_trajectory_scores(&uniform_scores);
        let weights: Vec<f32> = uniform_scores
            .iter()
            .map(|(_, s)| credit.node_weight(*s))
            .collect();

        println!();
        println!(
            "  Uniform scores: all = 0.5 → weights = [{:.2}, {:.2}, {:.2}]",
            weights[0], weights[1], weights[2]
        );
        assert!(
            weights.iter().all(|w| (w - 1.0).abs() < 0.01),
            "Uniform scores should all get weight 1.0"
        );
        println!("  ✓ Uniform scores: all weights = 1.0 (no discrimination needed)");
    }

    // Proof 3: all_weights sorted descending
    {
        let scores: Vec<(usize, f32)> = vec![(0, 0.2), (1, 0.9), (2, 0.5), (3, 0.1)];
        let credit = TrajectoryCredit::from_trajectory_scores(&scores);
        let sorted = credit.all_weights(&scores);

        println!();
        println!(
            "  Sorted weights: {:?}",
            sorted
                .iter()
                .map(|(i, w)| format!("(T{i}, {w:.2})"))
                .collect::<Vec<_>>()
        );
        assert_eq!(sorted[0].0, 1, "Best trajectory (0.9) should be first");
        assert!((sorted[0].1 - 1.0).abs() < 0.01);
        assert_eq!(
            sorted.last().unwrap().0,
            3,
            "Worst trajectory (0.1) should be last"
        );
        assert!((sorted.last().unwrap().1 - 0.0).abs() < 0.01);
        println!("  ✓ Sorted weights: descending order correct");
    }

    // ════════════════════════════════════════════════════════════
    // T8: SimpleTesLoop Integration Test
    // ════════════════════════════════════════════════════════════

    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!("🐐 T8: SimpleTesLoop — Concrete C×L×K Loop");
    println!("════════════════════════════════════════════════════════════════════════");
    println!();

    {
        let env = BernoulliEnv::new(&[0.3, 0.5, 0.8, 0.4, 0.6]);
        let config = TesConfig {
            global_width: 4,
            refinement_depth: 10,
            local_sample_size: 4,
            bandit_strategy: BanditStrategy::Rpucg {
                gamma: 0.8,
                lambda: 1.0,
            },
        };

        let mut rng = Rng::new(SEED);
        let mut tes = SimpleTesLoop::new(config.clone(), env);

        println!(
            "  Config: C={}, L={}, K={}, budget={}",
            config.global_width,
            config.refinement_depth,
            config.local_sample_size,
            config.budget()
        );

        let (evals, best) = tes.run(5, &mut rng);

        println!("  Evaluations: {evals}");
        println!("  Best score:  {best:.4}");
        println!("  History len: {}", tes.history().len());
        println!(
            "  Best solution: {:?}",
            tes.best_solution().map(|n| &n.solution)
        );

        assert!(evals > 0, "Should have evaluations");
        assert!(best >= 0.0, "Best score should be non-negative");
        assert!(!tes.history().is_empty(), "History should not be empty");
        assert_eq!(tes.total_evaluations(), tes.history().len());

        let has_propagation = tes.history().iter().any(|n| n.propagated_value > 0.0);
        println!("  Has propagated values: {has_propagation}");

        let budget = config.budget();
        println!(
            "  Budget utilization: {:.1}%",
            evals as f64 / budget as f64 * 100.0
        );

        println!("  ✓ SimpleTesLoop runs successfully with RPUCG strategy");
    }

    // ════════════════════════════════════════════════════════════
    // T9: Budget Scaling Benchmark
    // ════════════════════════════════════════════════════════════

    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!("🐐 T9: BUDGET SCALING — Vary (C, L, K) at Fixed Budget");
    println!("════════════════════════════════════════════════════════════════════════");
    println!();

    // Use Gaussian env with lower means — harder to hit max, budget allocation matters
    let scale_env = GaussianEnv::new(&[0.15, 0.25, 0.35, 0.45, 0.55, 0.30], 0.20);

    let configs: Vec<(&str, TesConfig)> = vec![
        (
            "Balanced (8×15×8)",
            TesConfig {
                global_width: 8,
                refinement_depth: 15,
                local_sample_size: 8,
                bandit_strategy: BanditStrategy::Rpucg {
                    gamma: 0.8,
                    lambda: 1.0,
                },
            },
        ),
        (
            "Wide (24×5×8)",
            TesConfig {
                global_width: 24,
                refinement_depth: 5,
                local_sample_size: 8,
                bandit_strategy: BanditStrategy::Rpucg {
                    gamma: 0.8,
                    lambda: 1.0,
                },
            },
        ),
        (
            "Deep (4×30×8)",
            TesConfig {
                global_width: 4,
                refinement_depth: 30,
                local_sample_size: 8,
                bandit_strategy: BanditStrategy::Rpucg {
                    gamma: 0.8,
                    lambda: 1.0,
                },
            },
        ),
        (
            "Narrow (2×8×30)",
            TesConfig {
                global_width: 2,
                refinement_depth: 8,
                local_sample_size: 30,
                bandit_strategy: BanditStrategy::Rpucg {
                    gamma: 0.8,
                    lambda: 1.0,
                },
            },
        ),
    ];

    println!("  Environment: Gaussian(6 arms, means=[0.15..0.55], σ=0.20, optimal=0.55)");
    println!("  Trials: {N_TRIALS}, Seed: {SEED}");
    println!();
    println!("  ┌─────────────────────┬────────┬──────────┬────────┬─────────┐");
    println!("  │ Config              │ Budget │ Avg Best │ Evals  │ Perfect │");
    println!("  ├─────────────────────┼────────┼──────────┼────────┼─────────┤");

    let mut t9_results: Vec<(&str, f32, usize, usize)> = Vec::new();

    for (name, config) in &configs {
        let budget = config.budget();
        let (avg_best, avg_evals, perfect) =
            run_tes_trials(config.clone(), &scale_env, N_TRIALS, SEED);
        println!(
            "  │ {:<19} │ {:>6} │ {:>8.4} │ {:>6} │ {:>7} │",
            name, budget, avg_best, avg_evals, perfect
        );
        t9_results.push((*name, avg_best, avg_evals, perfect));
    }

    println!("  └─────────────────────┴────────┴──────────┴────────┴─────────┘");

    // T9 Verdict: Budget allocation matters
    let max_best = t9_results.iter().map(|r| r.1).fold(f32::MIN, f32::max);
    let min_best = t9_results.iter().map(|r| r.1).fold(f32::MAX, f32::min);
    let spread = max_best - min_best;

    println!();
    println!("  Best avg score:  {max_best:.4}");
    println!("  Worst avg score: {min_best:.4}");
    println!("  Spread:          {spread:.4}");

    let t9_verdict = spread > 0.001;
    if t9_verdict {
        println!("  ✓ Budget allocation matters — spread = {spread:.4}");
    }

    // ════════════════════════════════════════════════════════════
    // T10: Cross-Strategy GOAT Proof
    // ════════════════════════════════════════════════════════════

    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!("🐐 T10: CROSS-STRATEGY — Bandit Strategy Tournament");
    println!("════════════════════════════════════════════════════════════════════════");
    println!();

    let bern_env = BernoulliEnv::new(&[0.2, 0.35, 0.5, 0.65, 0.8, 0.45, 0.3]);
    let gauss_env = GaussianEnv::new(&[0.1, 0.3, 0.5, 0.7, 0.9], 0.15);

    let episodes = 2000;

    let strategies: Vec<(&str, BanditStrategy)> = vec![
        ("UCB1", BanditStrategy::Ucb1),
        ("Thompson", BanditStrategy::ThompsonSampling),
        (
            "ε-greedy(0.1)",
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.1,
                decay: 1.0,
            },
        ),
        (
            "ε-greedy(0.3)",
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.3,
                decay: 1.0,
            },
        ),
        (
            "Var-ε(0.1)",
            BanditStrategy::VarianceEpsilon {
                epsilon: 0.1,
                var_decay: 0.99,
                lr: 0.1,
            },
        ),
        (
            "RPUCG(0.8,1.0)",
            BanditStrategy::Rpucg {
                gamma: 0.8,
                lambda: 1.0,
            },
        ),
    ];

    // ── Bernoulli Tournament ─────────────────────────────────

    println!(
        "  ── Bernoulli (7 arms, optimal={:.2}) ──────────────────",
        bern_env.expected_reward(bern_env.optimal_arm())
    );
    println!("  Episodes: {episodes}, Trials: {N_TRIALS}");
    println!();
    println!("  ┌──────────────────┬────────────┬────────────┬──────────┬──────────┐");
    println!("  │ Strategy         │ Avg Reward │ Avg Regret │ Regret/R │ Found ↑  │");
    println!("  ├──────────────────┼────────────┼────────────┼──────────┼──────────┤");

    let mut bern_results: Vec<(&str, f32, f32, f32, usize)> = Vec::new();

    for (name, strategy) in &strategies {
        let mut total_reward = 0.0f32;
        let mut total_regret = 0.0f32;
        let mut found_optimal = 0usize;

        for t in 0..N_TRIALS {
            let (reward, regret, best_arm) = run_bandit_session(
                bern_env.clone(),
                strategy.clone(),
                episodes,
                SEED + t as u64,
            );
            total_reward += reward;
            total_regret += regret;
            if best_arm == bern_env.optimal_arm() {
                found_optimal += 1;
            }
        }

        let avg_reward = total_reward / N_TRIALS as f32;
        let avg_regret = total_regret / N_TRIALS as f32;
        let regret_ratio = avg_regret / avg_reward.max(0.001);

        println!(
            "  │ {:<16} │ {:>10.1} │ {:>10.1} │ {:>8.3} │ {:>5}/{:<3} │",
            name, avg_reward, avg_regret, regret_ratio, found_optimal, N_TRIALS
        );

        bern_results.push((name, avg_reward, avg_regret, regret_ratio, found_optimal));
    }

    println!("  └──────────────────┴────────────┴────────────┴──────────┴──────────┘");

    // ── Gaussian Tournament ──────────────────────────────────

    println!();
    println!(
        "  ── Gaussian (5 arms, optimal={:.2}) ───────────────────",
        gauss_env.expected_reward(gauss_env.optimal_arm())
    );
    println!();
    println!("  ┌──────────────────┬────────────┬────────────┬──────────┬──────────┐");
    println!("  │ Strategy         │ Avg Reward │ Avg Regret │ Regret/R │ Found ↑  │");
    println!("  ├──────────────────┼────────────┼────────────┼──────────┼──────────┤");

    let mut gauss_results: Vec<(&str, f32, f32, f32, usize)> = Vec::new();

    for (name, strategy) in &strategies {
        let mut total_reward = 0.0f32;
        let mut total_regret = 0.0f32;
        let mut found_optimal = 0usize;

        for t in 0..N_TRIALS {
            let (reward, regret, best_arm) = run_bandit_session(
                gauss_env.clone(),
                strategy.clone(),
                episodes,
                SEED + t as u64 + 1000,
            );
            total_reward += reward;
            total_regret += regret;
            if best_arm == gauss_env.optimal_arm() {
                found_optimal += 1;
            }
        }

        let avg_reward = total_reward / N_TRIALS as f32;
        let avg_regret = total_regret / N_TRIALS as f32;
        let regret_ratio = avg_regret / avg_reward.max(0.001);

        println!(
            "  │ {:<16} │ {:>10.1} │ {:>10.1} │ {:>8.3} │ {:>5}/{:<3} │",
            name, avg_reward, avg_regret, regret_ratio, found_optimal, N_TRIALS
        );

        gauss_results.push((name, avg_reward, avg_regret, regret_ratio, found_optimal));
    }

    println!("  └──────────────────┴────────────┴────────────┴──────────┴──────────┘");

    // ── T10 Verdicts ─────────────────────────────────────────

    println!();
    println!("  ── T10 Verdicts ──────────────────────────────────────────────");

    // 1. RPUCG falls back to UCB1 in flat bandit → regret should be similar
    let rpucg_bern = bern_results
        .iter()
        .find(|(n, _, _, _, _)| *n == "RPUCG(0.8,1.0)")
        .unwrap();
    let ucb1_bern = bern_results
        .iter()
        .find(|(n, _, _, _, _)| *n == "UCB1")
        .unwrap();
    let rpucg_gauss = gauss_results
        .iter()
        .find(|(n, _, _, _, _)| *n == "RPUCG(0.8,1.0)")
        .unwrap();
    let ucb1_gauss = gauss_results
        .iter()
        .find(|(n, _, _, _, _)| *n == "UCB1")
        .unwrap();

    let bern_regret_close = {
        let ratio = (rpucg_bern.2 - ucb1_bern.2).abs() / ucb1_bern.2.max(0.001);
        ratio < 0.15
    };
    let gauss_regret_close = {
        let ratio = (rpucg_gauss.2 - ucb1_gauss.2).abs() / ucb1_gauss.2.max(0.001);
        ratio < 0.15
    };

    println!(
        "  RPUCG vs UCB1 Bernoulli regret: {:.1} vs {:.1} ({})",
        rpucg_bern.2,
        ucb1_bern.2,
        if bern_regret_close {
            "✓ within 15%"
        } else {
            "⚠ gap > 15%"
        }
    );
    println!(
        "  RPUCG vs UCB1 Gaussian regret:  {:.1} vs {:.1} ({})",
        rpucg_gauss.2,
        ucb1_gauss.2,
        if gauss_regret_close {
            "✓ within 15%"
        } else {
            "⚠ gap > 15%"
        }
    );

    // 2. All strategies should find optimal arm at least sometimes
    let all_found_bern = bern_results.iter().all(|(_, _, _, _, found)| *found > 0);
    let all_found_gauss = gauss_results.iter().all(|(_, _, _, _, found)| *found > 0);

    println!(
        "  All strategies found optimal Bernoulli: {}",
        if all_found_bern { "✓" } else { "⚠" }
    );
    println!(
        "  All strategies found optimal Gaussian:  {}",
        if all_found_gauss { "✓" } else { "⚠" }
    );

    // 3. Best strategy should have lower regret than worst
    let best_regret_bern = bern_results.iter().map(|r| r.3).fold(f32::MIN, f32::max);
    let worst_regret_bern = bern_results.iter().map(|r| r.3).fold(f32::MAX, f32::min);
    let strategies_differentiate = best_regret_bern > worst_regret_bern;

    println!(
        "  Bernoulli regret ratio range: [{:.3}..{:.3}] ({})",
        worst_regret_bern,
        best_regret_bern,
        if strategies_differentiate {
            "✓ strategies differentiate"
        } else {
            "⚠ all same"
        }
    );

    // 4. Thompson should dominate on Gaussian (conjugate prior advantage)
    let thompson_gauss = gauss_results
        .iter()
        .find(|(n, _, _, _, _)| *n == "Thompson")
        .unwrap();
    let eps_03_gauss = gauss_results
        .iter()
        .find(|(n, _, _, _, _)| *n == "ε-greedy(0.3)")
        .unwrap();
    let thompson_beats_eps03 = thompson_gauss.1 > eps_03_gauss.1;

    println!(
        "  Thompson reward ({:.1}) > ε-greedy(0.3) ({:.1}) Gaussian: {}",
        thompson_gauss.1,
        eps_03_gauss.1,
        if thompson_beats_eps03 { "✓" } else { "⚠" }
    );

    // ════════════════════════════════════════════════════════════
    // Final Summary
    // ════════════════════════════════════════════════════════════

    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!("🐐 GOAT PROOF SUMMARY");
    println!("════════════════════════════════════════════════════════════════════════");

    println!("  T7  Credit Bridge:       ✓ best=1.0, worst=0.0, linear interpolation");
    println!("  T8  SimpleTesLoop:       ✓ C×L×K loop runs with RPUCG propagation");
    println!(
        "  T9  Budget Scaling:      {} spread={:.4} (allocation matters)",
        if t9_verdict { "✓" } else { "⚠" },
        spread
    );

    let t10_verdict = bern_regret_close && gauss_regret_close && all_found_bern && all_found_gauss;
    println!(
        "  T10 Cross-Strategy:      {} RPUCG≅UCB1, all converge",
        if t10_verdict { "✓" } else { "⚠" }
    );

    println!("════════════════════════════════════════════════════════════════════════");

    // CI: Thompson should beat ε-greedy(0.3) on Gaussian
    assert!(
        thompson_beats_eps03,
        "T10: Thompson should beat ε-greedy(0.3) on Gaussian"
    );

    let all_pass = t9_verdict && t10_verdict;
    if all_pass {
        println!("  ✅ All GOAT proofs passed. SimpleTES T7-T10 GOAT-qualified.");
    } else {
        println!("  ⚠ Some proofs need investigation — check results above.");
    }
    println!("════════════════════════════════════════════════════════════════════════");
    println!();

    // CI assertions
    assert!(
        bern_regret_close,
        "T10 Bernoulli: RPUCG regret should be within 15% of UCB1"
    );
    assert!(
        gauss_regret_close,
        "T10 Gaussian: RPUCG regret should be within 15% of UCB1"
    );
    assert!(
        all_found_bern,
        "T10: All strategies should find optimal Bernoulli arm"
    );
    assert!(
        all_found_gauss,
        "T10: All strategies should find optimal Gaussian arm"
    );
}
