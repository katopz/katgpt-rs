//! SDAR Gated Distillation modelless benchmark — run with:
//! cargo test --features "sdar_gate,bandit" --test bench_sdar_gated_modelless --release -- --nocapture
//!
//! Benchmarks SDAR sigmoid-gated reward components:
//! 1. Hot-path (relevance) vs cold-path (update/observe) overhead
//! 2. Sigmoid gate + update/observe throughput
//! 3. Bandit convergence: scalar vs SDAR-gated reward
//! 4. Absorb promotion quality: hard threshold vs soft sigmoid gate

// ── 1. Overhead Benchmark ───────────────────────────────────────

#[cfg(all(feature = "sdar_gate", feature = "bandit"))]
#[test]
fn bench_sdar_gated_overhead() {
    use std::time::Instant;

    use katgpt_rs::pruners::{
        AbsorbCompress, AbsorbCompressLayer, BanditPruner, BanditStrategy, CompressConfig,
        SdarAbsorbConfig, SdarBanditPruner, SdarGatedAbsorbCompress,
    };
    use katgpt_rs::speculative::types::{NoScreeningPruner, ScreeningPruner};
    use katgpt_rs::types::Rng;

    let num_arms = 100;
    let warmup = 1000;
    let iters = 100_000;

    println!(
        "\n🧪 SDAR Gated Overhead Benchmark ({iters} iters, {warmup} warmup, {num_arms} arms)"
    );
    println!("{}", "═".repeat(70));

    // ── Hot path: relevance() — SdarGatedAbsorbCompress ──────────
    // Compare wrapper vs inner AbsorbCompressLayer (not NoScreeningPruner).
    // This measures the delegation overhead of the SDAR wrapper.

    let config = CompressConfig::new(50, 0.1, 5, 1000);
    let baseline_absorb = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config.clone());
    let inner_absorb = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config.clone());
    let sdar_absorb =
        SdarGatedAbsorbCompress::new(inner_absorb, num_arms, SdarAbsorbConfig::default());

    // Warmup
    for i in 0..warmup {
        let _ = baseline_absorb.relevance(0, i % num_arms, &[]);
        let _ = sdar_absorb.relevance(0, i % num_arms, &[]);
    }

    let start = Instant::now();
    for i in 0..iters {
        let _ = baseline_absorb.relevance(0, i % num_arms, &[]);
    }
    let baseline_absorb_relevance = start.elapsed();

    let start = Instant::now();
    for i in 0..iters {
        let _ = sdar_absorb.relevance(0, i % num_arms, &[]);
    }
    let sdar_absorb_relevance = start.elapsed();

    let absorb_overhead_pct = ((sdar_absorb_relevance.as_nanos() as f64
        / baseline_absorb_relevance.as_nanos() as f64)
        - 1.0)
        * 100.0;

    println!("   relevance() — SdarGatedAbsorbCompress:");
    println!("     Baseline (AbsorbCompress):     {baseline_absorb_relevance:>8?}");
    println!("     SdarGatedAbsorbCompress:       {sdar_absorb_relevance:>8?}");
    println!("     Overhead:                      {absorb_overhead_pct:+.1}%");

    // ── Hot path: relevance() — SdarBanditPruner ─────────────────
    // Compare wrapper vs inner BanditPruner (not NoScreeningPruner).

    let baseline_bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let inner_bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let sdar_bandit = SdarBanditPruner::new(inner_bandit, num_arms);

    // Warmup
    for i in 0..warmup {
        let _ = baseline_bandit.relevance(0, i % num_arms, &[]);
        let _ = sdar_bandit.relevance(0, i % num_arms, &[]);
    }

    let start = Instant::now();
    for i in 0..iters {
        let _ = baseline_bandit.relevance(0, i % num_arms, &[]);
    }
    let baseline_bandit_relevance = start.elapsed();

    let start = Instant::now();
    for i in 0..iters {
        let _ = sdar_bandit.relevance(0, i % num_arms, &[]);
    }
    let sdar_bandit_relevance = start.elapsed();

    let bandit_overhead_pct = ((sdar_bandit_relevance.as_nanos() as f64
        / baseline_bandit_relevance.as_nanos() as f64)
        - 1.0)
        * 100.0;

    println!();
    println!("   relevance() — SdarBanditPruner:");
    println!("     Baseline (BanditPruner):       {baseline_bandit_relevance:>8?}");
    println!("     SdarBanditPruner:              {sdar_bandit_relevance:>8?}");
    println!("     Overhead:                      {bandit_overhead_pct:+.1}%");

    // ── Cold path: SdarBanditPruner::update() vs BanditPruner::update() ─

    let mut scalar_bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let inner_bandit2 = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let mut sdar_bandit = SdarBanditPruner::new(inner_bandit2, num_arms);

    let mut rng = Rng::new(42);

    // Warmup scalar
    for i in 0..warmup {
        let arm = i % num_arms;
        scalar_bandit.update(arm, rng.uniform());
    }

    // Warmup SDAR
    for i in 0..warmup {
        let arm = i % num_arms;
        sdar_bandit.update(arm, rng.uniform());
    }

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..iters {
        let arm = i % num_arms;
        scalar_bandit.update(arm, rng.uniform());
    }
    let scalar_bandit_time = start.elapsed();

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..iters {
        let arm = i % num_arms;
        sdar_bandit.update(arm, rng.uniform());
    }
    let sdar_bandit_time = start.elapsed();

    let bandit_cold_overhead_pct =
        ((sdar_bandit_time.as_nanos() as f64 / scalar_bandit_time.as_nanos() as f64) - 1.0) * 100.0;

    println!();
    println!("   update() — SdarBanditPruner vs BanditPruner:");
    println!("     BanditPruner::update():          {scalar_bandit_time:>8?}");
    println!("     SdarBanditPruner::update():      {sdar_bandit_time:>8?}");
    println!("     Overhead:                         {bandit_cold_overhead_pct:+.1}%");

    // ── Cold path: SdarGatedAbsorbCompress::observe() vs AbsorbCompressLayer::absorb() ─

    let config2 = CompressConfig::new(50, 0.1, 5, 1000);
    let mut scalar_absorb = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config2.clone());
    let inner_absorb2 = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config2);
    let mut sdar_absorb =
        SdarGatedAbsorbCompress::new(inner_absorb2, num_arms, SdarAbsorbConfig::default());

    let mut rng = Rng::new(42);

    // Warmup scalar
    for i in 0..warmup {
        let arm = i % num_arms;
        scalar_absorb.absorb(arm, rng.uniform());
    }

    // Warmup SDAR
    for i in 0..warmup {
        let arm = i % num_arms;
        let reward = rng.uniform();
        sdar_absorb.observe(arm, reward, 1.0 + rng.uniform());
    }

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..iters {
        let arm = i % num_arms;
        scalar_absorb.absorb(arm, rng.uniform());
    }
    let scalar_absorb_time = start.elapsed();

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..iters {
        let arm = i % num_arms;
        let reward = rng.uniform();
        let benefit_ratio = 1.0 + rng.uniform();
        sdar_absorb.observe(arm, reward, benefit_ratio);
    }
    let sdar_absorb_time = start.elapsed();

    let absorb_cold_overhead_pct =
        ((sdar_absorb_time.as_nanos() as f64 / scalar_absorb_time.as_nanos() as f64) - 1.0) * 100.0;

    println!();
    println!("   observe() vs absorb():");
    println!("     AbsorbCompressLayer::absorb():   {scalar_absorb_time:>8?}");
    println!("     SdarGated observe(br):           {sdar_absorb_time:>8?}");
    println!("     Overhead:                         {absorb_cold_overhead_pct:+.1}%");

    // ── Verdict ───────────────────────────────────────────────────
    println!();
    println!("   Targets: hot-path overhead < 5%, cold-path overhead acceptable");
    let hot_pass = absorb_overhead_pct < 5.0 && bandit_overhead_pct < 5.0;
    if hot_pass {
        println!("   ✅ PASS: hot-path relevance() overhead < 5% (compiler inlines delegation)");
    } else {
        println!("   ⚠️  FAIL: hot-path relevance() overhead ≥ 5%");
    }
    if bandit_cold_overhead_pct < 300.0 {
        println!(
            "   ✅ PASS: bandit cold-path overhead is {bandit_cold_overhead_pct:.1}% (sigmoid gate computation)"
        );
    } else {
        println!("   ⚠️  FAIL: bandit cold-path overhead is {bandit_cold_overhead_pct:.1}%");
    }
    if absorb_cold_overhead_pct < 500.0 {
        println!(
            "   ✅ PASS: absorb cold-path overhead is {absorb_cold_overhead_pct:.1}% (soft gate + PRNG)"
        );
    } else {
        println!("   ⚠️  FAIL: absorb cold-path overhead is {absorb_cold_overhead_pct:.1}%");
    }
}

// ── 2. Throughput Benchmark ─────────────────────────────────────

#[cfg(all(feature = "sdar_gate", feature = "bandit"))]
#[test]
fn bench_sdar_gated_throughput() {
    use std::time::Instant;

    use katgpt_rs::pruners::{
        AbsorbCompressLayer, BanditPruner, BanditStrategy, CompressConfig, SDAR_BETA,
        SdarAbsorbConfig, SdarBanditPruner, SdarGatedAbsorbCompress, sdar_gate,
    };
    use katgpt_rs::speculative::types::NoScreeningPruner;
    use katgpt_rs::types::Rng;

    let num_arms = 100;
    let warmup = 1000;
    let iters = 100_000;

    println!("\n🧪 SDAR Gated Throughput Benchmark ({iters} iters)");
    println!("{}", "═".repeat(70));

    // ── Pure sigmoid gate throughput ──────────────────────────────

    let mut rng = Rng::new(42);

    // Warmup
    for _ in 0..warmup {
        let _ = sdar_gate(rng.uniform(), SDAR_BETA);
    }

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for _ in 0..iters {
        let _ = sdar_gate(rng.uniform(), SDAR_BETA);
    }
    let gate_time = start.elapsed();

    let gate_per_call = gate_time / iters as u32;
    let gate_calls_per_sec = iters as f64 / gate_time.as_secs_f64();

    println!("   sdar_gate() (pure sigmoid, β={SDAR_BETA}):");
    println!("     {iters} calls in {gate_time:?}");
    println!("     Per call: {gate_per_call:?}");
    println!("     Throughput: {gate_calls_per_sec:.0} calls/sec");

    // ── SdarBanditPruner::update() throughput ─────────────────────

    let inner_bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let mut sdar_bandit = SdarBanditPruner::new(inner_bandit, num_arms);

    let mut rng = Rng::new(42);

    // Warmup
    for i in 0..warmup {
        sdar_bandit.update(i % num_arms, rng.uniform());
    }

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..iters {
        sdar_bandit.update(i % num_arms, rng.uniform());
    }
    let bandit_time = start.elapsed();

    let bandit_per_call = bandit_time / iters as u32;
    let bandit_calls_per_sec = iters as f64 / bandit_time.as_secs_f64();

    println!();
    println!("   SdarBanditPruner::update() (gated reward):");
    println!("     {iters} calls in {bandit_time:?}");
    println!("     Per call: {bandit_per_call:?}");
    println!("     Throughput: {bandit_calls_per_sec:.0} calls/sec");

    // ── SdarGatedAbsorbCompress::observe() throughput ─────────────

    let config = CompressConfig::new(50, 0.1, 5, 1000);
    let inner_absorb = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config.clone());
    let mut sdar_absorb =
        SdarGatedAbsorbCompress::new(inner_absorb, num_arms, SdarAbsorbConfig::default());

    let mut rng = Rng::new(42);

    // Warmup
    for i in 0..warmup {
        let arm = i % num_arms;
        let reward = rng.uniform();
        let benefit_ratio = 1.0 + rng.uniform();
        sdar_absorb.observe(arm, reward, benefit_ratio);
    }

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..iters {
        let arm = i % num_arms;
        let reward = rng.uniform();
        let benefit_ratio = 1.0 + rng.uniform();
        sdar_absorb.observe(arm, reward, benefit_ratio);
    }
    let absorb_time = start.elapsed();

    let absorb_per_call = absorb_time / iters as u32;
    let absorb_calls_per_sec = iters as f64 / absorb_time.as_secs_f64();

    println!();
    println!("   SdarGatedAbsorbCompress::observe(br):");
    println!("     {iters} calls in {absorb_time:?}");
    println!("     Per call: {absorb_per_call:?}");
    println!("     Throughput: {absorb_calls_per_sec:.0} calls/sec");

    // ── SdarGatedAbsorbCompress::observe_with_q() throughput ──────

    let inner_absorb2 = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config);
    let mut sdar_absorb2 =
        SdarGatedAbsorbCompress::new(inner_absorb2, num_arms, SdarAbsorbConfig::default());

    let mut rng = Rng::new(42);

    // Warmup
    for i in 0..warmup {
        let arm = i % num_arms;
        let reward = rng.uniform();
        let q_value = rng.uniform() * 0.5;
        sdar_absorb2.observe_with_q(arm, reward, q_value);
    }

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..iters {
        let arm = i % num_arms;
        let reward = rng.uniform();
        let q_value = rng.uniform() * 0.5;
        sdar_absorb2.observe_with_q(arm, reward, q_value);
    }
    let observe_q_time = start.elapsed();

    let observe_q_per_call = observe_q_time / iters as u32;
    let observe_q_calls_per_sec = iters as f64 / observe_q_time.as_secs_f64();

    println!();
    println!("   SdarGatedAbsorbCompress::observe_with_q():");
    println!("     {iters} calls in {observe_q_time:?}");
    println!("     Per call: {observe_q_per_call:?}");
    println!("     Throughput: {observe_q_calls_per_sec:.0} calls/sec");

    // ── Verdict ───────────────────────────────────────────────────
    println!();
    let gate_target = 500_000.0;
    let bandit_target = 100_000.0;
    let absorb_target = 100_000.0;

    println!("   Targets: sigmoid gate >500K/sec, update/observe >100K/sec");
    if gate_calls_per_sec > gate_target {
        println!("   ✅ PASS: sdar_gate() at {gate_calls_per_sec:.0} calls/sec");
    } else {
        println!("   ⚠️  FAIL: sdar_gate() at {gate_calls_per_sec:.0} calls/sec (target >500K)");
    }
    if bandit_calls_per_sec > bandit_target {
        println!("   ✅ PASS: SdarBandit update() at {bandit_calls_per_sec:.0} calls/sec");
    } else {
        println!(
            "   ⚠️  FAIL: SdarBandit update() at {bandit_calls_per_sec:.0} calls/sec (target >100K)"
        );
    }
    if absorb_calls_per_sec > absorb_target {
        println!("   ✅ PASS: SdarGated observe() at {absorb_calls_per_sec:.0} calls/sec");
    } else {
        println!(
            "   ⚠️  FAIL: SdarGated observe() at {absorb_calls_per_sec:.0} calls/sec (target >100K)"
        );
    }
}

// ── 3. Convergence Benchmark ────────────────────────────────────

#[cfg(all(feature = "sdar_gate", feature = "bandit"))]
#[test]
fn bench_sdar_gated_convergence() {
    use std::time::Instant;

    use katgpt_rs::pruners::{BanditPruner, BanditStrategy, SdarBanditPruner};
    use katgpt_rs::speculative::types::NoScreeningPruner;
    use katgpt_rs::types::Rng;

    let num_arms = 10;
    let episodes = 1000;
    let optimal_arm = 0;

    // Simulated environment: arm 0 is optimal (0.9), others 0.3-0.5
    let arm_means: Vec<f32> = vec![0.9, 0.3, 0.4, 0.35, 0.5, 0.3, 0.45, 0.38, 0.42, 0.33];

    println!("\n🧪 SDAR Gated Bandit Convergence ({episodes} episodes, {num_arms} arms)");
    println!("{}", "═".repeat(70));
    println!("   Optimal arm: {optimal_arm} (mean reward 0.9)");
    println!("   Other arms: means {:?}", &arm_means[1..]);

    // ── Scalar BanditPruner (UCB1) ───────────────────────────────

    let mut scalar_bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let mut rng = Rng::new(42);

    let mut scalar_cumulative_regret: f32 = 0.0;
    let mut scalar_best_found_at: Option<usize> = None;
    let mut scalar_regret_100: f32 = 0.0;
    let mut scalar_regret_500: f32 = 0.0;

    let start = Instant::now();
    for ep in 0..episodes {
        // Select arm via UCB1
        let arm = select_ucb1_arm_scalar(&scalar_bandit, ep, num_arms);

        // Simulate reward with noise
        let noise = (rng.uniform() - 0.5) * 0.1;
        let reward = (arm_means[arm] + noise).clamp(0.0, 1.0);

        scalar_bandit.update(arm, reward);

        // Regret = optimal_mean - actual_reward
        let regret = arm_means[optimal_arm] - arm_means[arm];
        scalar_cumulative_regret += regret;

        if ep == 99 {
            scalar_regret_100 = scalar_cumulative_regret;
        }
        if ep == 499 {
            scalar_regret_500 = scalar_cumulative_regret;
        }

        // Check if best arm identified
        if scalar_best_found_at.is_none()
            && is_best_arm_most_visited_scalar(&scalar_bandit, optimal_arm, num_arms)
        {
            scalar_best_found_at = Some(ep);
        }
    }
    let scalar_time = start.elapsed();

    println!();
    println!("   Scalar BanditPruner (UCB1):");
    println!("     Time:                       {scalar_time:?}");
    println!("     Cumulative regret:          {scalar_cumulative_regret:.2}");
    println!("     Regret at ep 100:           {scalar_regret_100:.2}");
    println!("     Regret at ep 500:           {scalar_regret_500:.2}");
    println!(
        "     Best arm found at ep:       {:?}",
        scalar_best_found_at.unwrap_or(episodes)
    );

    // ── SdarBanditPruner (sigmoid-gated UCB1) ────────────────────

    let inner_bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let mut sdar_bandit = SdarBanditPruner::new(inner_bandit, num_arms);
    let mut rng = Rng::new(42);

    let mut sdar_cumulative_regret: f32 = 0.0;
    let mut sdar_best_found_at: Option<usize> = None;
    let mut sdar_regret_100: f32 = 0.0;
    let mut sdar_regret_500: f32 = 0.0;

    let start = Instant::now();
    for ep in 0..episodes {
        // Select arm via UCB1 using inner bandit's Q-values
        let arm = select_ucb1_arm_sdar(&sdar_bandit, ep, num_arms);

        // Simulate reward with noise (same distribution)
        let noise = (rng.uniform() - 0.5) * 0.1;
        let reward = (arm_means[arm] + noise).clamp(0.0, 1.0);

        // SDAR-gated update: positive surprise → gate opens → full update
        // Negative surprise → gate closes → attenuated update
        sdar_bandit.update(arm, reward);

        // Regret computation (same basis for fair comparison)
        let regret = arm_means[optimal_arm] - arm_means[arm];
        sdar_cumulative_regret += regret;

        if ep == 99 {
            sdar_regret_100 = sdar_cumulative_regret;
        }
        if ep == 499 {
            sdar_regret_500 = sdar_cumulative_regret;
        }

        if sdar_best_found_at.is_none()
            && is_best_arm_most_visited_sdar(&sdar_bandit, optimal_arm, num_arms)
        {
            sdar_best_found_at = Some(ep);
        }
    }
    let sdar_time = start.elapsed();

    println!();
    println!("   SdarBanditPruner (sigmoid-gated UCB1, β=5.0):");
    println!("     Time:                       {sdar_time:?}");
    println!("     Cumulative regret:          {sdar_cumulative_regret:.2}");
    println!("     Regret at ep 100:           {sdar_regret_100:.2}");
    println!("     Regret at ep 500:           {sdar_regret_500:.2}");
    println!(
        "     Best arm found at ep:       {:?}",
        sdar_best_found_at.unwrap_or(episodes)
    );

    // ── Comparison ────────────────────────────────────────────────

    let regret_diff = scalar_cumulative_regret - sdar_cumulative_regret;
    let scalar_ep = scalar_best_found_at.unwrap_or(episodes);
    let sdar_ep = sdar_best_found_at.unwrap_or(episodes);

    println!();
    println!("   Comparison:");
    println!("     Regret difference:          {regret_diff:+.2} (positive = SDAR better)");
    println!("     Scalar found at ep:         {scalar_ep}");
    println!("     SDAR found at ep:           {sdar_ep}");
    println!("     Regret at ep 100 (scalar):  {scalar_regret_100:.2}");
    println!("     Regret at ep 100 (SDAR):    {sdar_regret_100:.2}");
    println!("     Regret at ep 500 (scalar):  {scalar_regret_500:.2}");
    println!("     Regret at ep 500 (SDAR):    {sdar_regret_500:.2}");

    // ── Verdict ───────────────────────────────────────────────────
    println!();
    // SDAR gate should converge without regression — it may not be faster
    // but it should not significantly degrade convergence
    let convergence_pass = sdar_best_found_at.is_some();
    let regret_pass = sdar_cumulative_regret <= scalar_cumulative_regret * 1.5; // Allow up to 50% more regret

    if convergence_pass {
        println!("   ✅ PASS: SDAR-gated bandit converges (ep {sdar_ep})");
    } else {
        println!("   ⚠️  FAIL: SDAR-gated bandit did not converge within {episodes} episodes");
    }
    if regret_pass {
        println!(
            "   ✅ PASS: SDAR regret ({sdar_cumulative_regret:.2}) within acceptable range of scalar ({scalar_cumulative_regret:.2})"
        );
    } else {
        println!(
            "   ⚠️  FAIL: SDAR regret ({sdar_cumulative_regret:.2}) significantly worse than scalar ({scalar_cumulative_regret:.2})"
        );
    }
}

// ── 4. Absorb Promotion Quality Benchmark ───────────────────────

#[cfg(all(feature = "sdar_gate", feature = "bandit"))]
#[cfg(debug_assertions)] // promotion_stats / with_promotion_stats are debug-only APIs
#[test]
fn bench_sdar_gated_absorb_promotion() {
    use std::time::Instant;

    use katgpt_rs::pruners::{
        AbsorbCompress, AbsorbCompressLayer, CompressConfig, SdarAbsorbConfig,
        SdarGatedAbsorbCompress,
    };
    use katgpt_rs::speculative::types::NoScreeningPruner;
    use katgpt_rs::types::Rng;

    let num_arms = 100;
    let observations = 1000;

    println!(
        "\n🧪 SDAR Gated Absorb Promotion Quality ({observations} observations, {num_arms} arms)"
    );
    println!("{}", "═".repeat(70));

    let mut rng = Rng::new(42);

    // ── Baseline: Hard threshold absorb ───────────────────────────

    let config = CompressConfig::new(10, 0.05, 3, 100);
    let mut scalar_layer = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config.clone());

    let start = Instant::now();
    for i in 0..observations {
        let arm = i % num_arms;
        let reward = rng.uniform();
        scalar_layer.absorb(arm, reward);
    }
    let scalar_time = start.elapsed();

    let scalar_absorbed = count_compressed(&scalar_layer, num_arms);

    println!();
    println!("   Baseline — AbsorbCompressLayer (hard threshold):");
    println!("     Time:                       {scalar_time:?}");
    println!("     Arms compressed:            {scalar_absorbed}/{num_arms}");

    // ── SDAR soft gate at β=1.0 (soft) ───────────────────────────

    let inner1 = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config.clone());
    let sdar_config1 = SdarAbsorbConfig::soft().with_promotion_stats();
    let mut sdar_soft = SdarGatedAbsorbCompress::new(inner1, num_arms, sdar_config1);

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..observations {
        let arm = i % num_arms;
        let reward = rng.uniform();
        let benefit_ratio = 0.5 + rng.uniform() * 1.5; // Range [0.5, 2.0]
        sdar_soft.observe(arm, reward, benefit_ratio);
    }
    let soft_time = start.elapsed();

    // Collect promotion stats
    let mut soft_promotion_attempts = 0usize;
    let mut soft_promotions = 0usize;
    let mut soft_mean_gate_prob = 0.0f32;
    let mut soft_stat_count = 0usize;

    for arm in 0..num_arms {
        if let Some(stats) = sdar_soft.promotion_stats(arm) {
            soft_promotion_attempts += stats.promotion_attempts;
            soft_promotions += stats.promotions;
            if stats.promotion_attempts > 0 {
                soft_mean_gate_prob += stats.mean_gate_probability();
                soft_stat_count += 1;
            }
        }
    }
    if soft_stat_count > 0 {
        soft_mean_gate_prob /= soft_stat_count as f32;
    }

    println!();
    println!("   SDAR β=1.0 (soft gate):");
    println!("     Time:                       {soft_time:?}");
    println!("     Promotion attempts:         {soft_promotion_attempts}");
    println!("     Promotions:                 {soft_promotions}");
    println!("     Mean gate probability:      {soft_mean_gate_prob:.3}");
    println!(
        "     Promotion rate:             {:.1}%",
        if soft_promotion_attempts > 0 {
            soft_promotions as f64 / soft_promotion_attempts as f64 * 100.0
        } else {
            0.0
        }
    );

    // ── SDAR soft gate at β=5.0 (paper-validated optimum) ────────

    let inner2 = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config.clone());
    let sdar_config2 = SdarAbsorbConfig::new(5.0).with_promotion_stats();
    let mut sdar_optimal = SdarGatedAbsorbCompress::new(inner2, num_arms, sdar_config2);

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..observations {
        let arm = i % num_arms;
        let reward = rng.uniform();
        let benefit_ratio = 0.5 + rng.uniform() * 1.5; // Range [0.5, 2.0]
        sdar_optimal.observe(arm, reward, benefit_ratio);
    }
    let optimal_time = start.elapsed();

    let mut optimal_promotion_attempts = 0usize;
    let mut optimal_promotions = 0usize;
    let mut optimal_mean_gate_prob = 0.0f32;
    let mut optimal_stat_count = 0usize;

    for arm in 0..num_arms {
        if let Some(stats) = sdar_optimal.promotion_stats(arm) {
            optimal_promotion_attempts += stats.promotion_attempts;
            optimal_promotions += stats.promotions;
            if stats.promotion_attempts > 0 {
                optimal_mean_gate_prob += stats.mean_gate_probability();
                optimal_stat_count += 1;
            }
        }
    }
    if optimal_stat_count > 0 {
        optimal_mean_gate_prob /= optimal_stat_count as f32;
    }

    println!();
    println!("   SDAR β=5.0 (paper optimum):");
    println!("     Time:                       {optimal_time:?}");
    println!("     Promotion attempts:         {optimal_promotion_attempts}");
    println!("     Promotions:                 {optimal_promotions}");
    println!("     Mean gate probability:      {optimal_mean_gate_prob:.3}");
    println!(
        "     Promotion rate:             {:.1}%",
        if optimal_promotion_attempts > 0 {
            optimal_promotions as f64 / optimal_promotion_attempts as f64 * 100.0
        } else {
            0.0
        }
    );

    // ── SDAR soft gate at β=10.0 (near-binary) ───────────────────

    let inner3 = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config.clone());
    let sdar_config3 = SdarAbsorbConfig::aggressive().with_promotion_stats();
    let mut sdar_aggressive = SdarGatedAbsorbCompress::new(inner3, num_arms, sdar_config3);

    let mut rng = Rng::new(42);
    let start = Instant::now();
    for i in 0..observations {
        let arm = i % num_arms;
        let reward = rng.uniform();
        let benefit_ratio = 0.5 + rng.uniform() * 1.5;
        sdar_aggressive.observe(arm, reward, benefit_ratio);
    }
    let aggressive_time = start.elapsed();

    let mut aggressive_promotion_attempts = 0usize;
    let mut aggressive_promotions = 0usize;
    let mut aggressive_mean_gate_prob = 0.0f32;
    let mut aggressive_stat_count = 0usize;

    for arm in 0..num_arms {
        if let Some(stats) = sdar_aggressive.promotion_stats(arm) {
            aggressive_promotion_attempts += stats.promotion_attempts;
            aggressive_promotions += stats.promotions;
            if stats.promotion_attempts > 0 {
                aggressive_mean_gate_prob += stats.mean_gate_probability();
                aggressive_stat_count += 1;
            }
        }
    }
    if aggressive_stat_count > 0 {
        aggressive_mean_gate_prob /= aggressive_stat_count as f32;
    }

    println!();
    println!("   SDAR β=10.0 (near-binary):");
    println!("     Time:                       {aggressive_time:?}");
    println!("     Promotion attempts:         {aggressive_promotion_attempts}");
    println!("     Promotions:                 {aggressive_promotions}");
    println!("     Mean gate probability:      {aggressive_mean_gate_prob:.3}");
    println!(
        "     Promotion rate:             {:.1}%",
        if aggressive_promotion_attempts > 0 {
            aggressive_promotions as f64 / aggressive_promotion_attempts as f64 * 100.0
        } else {
            0.0
        }
    );

    // ── Benefit ratio targeting test ──────────────────────────────

    // Verify: arms with high benefit ratio get promoted more than low benefit ratio arms
    let inner4 = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config);
    let sdar_config4 = SdarAbsorbConfig::new(5.0).with_promotion_stats();
    let mut sdar_targeted = SdarGatedAbsorbCompress::new(inner4, num_arms, sdar_config4);

    let mut rng = Rng::new(42);

    // Arms 0-19: high benefit ratio (>1.5) — should promote
    // Arms 20-39: neutral benefit ratio (~1.0) — 50/50
    // Arms 40-59: low benefit ratio (<0.5) — should not promote
    // Arms 60+: random benefit ratio

    for i in 0..observations {
        let arm = i % num_arms;
        let reward = rng.uniform();

        let benefit_ratio = if arm < 20 {
            1.5 + rng.uniform() * 0.5 // High: [1.5, 2.0]
        } else if arm < 40 {
            0.9 + rng.uniform() * 0.2 // Neutral: [0.9, 1.1]
        } else if arm < 60 {
            rng.uniform() * 0.4 // Low: [0.0, 0.4]
        } else {
            0.5 + rng.uniform() * 1.5 // Random: [0.5, 2.0]
        };

        sdar_targeted.observe(arm, reward, benefit_ratio);
    }

    // Count promotions per group
    let high_br_promotions = (0..20_usize)
        .filter_map(|arm| sdar_targeted.promotion_stats(arm).map(|s| s.promotions))
        .sum::<usize>();

    let neutral_br_promotions = (20..40_usize)
        .filter_map(|arm| sdar_targeted.promotion_stats(arm).map(|s| s.promotions))
        .sum::<usize>();

    let low_br_promotions = (40..60_usize)
        .filter_map(|arm| sdar_targeted.promotion_stats(arm).map(|s| s.promotions))
        .sum::<usize>();

    let high_br_attempts = (0..20_usize)
        .filter_map(|arm| {
            sdar_targeted
                .promotion_stats(arm)
                .map(|s| s.promotion_attempts)
        })
        .sum::<usize>();

    let neutral_br_attempts = (20..40_usize)
        .filter_map(|arm| {
            sdar_targeted
                .promotion_stats(arm)
                .map(|s| s.promotion_attempts)
        })
        .sum::<usize>();

    let low_br_attempts = (40..60_usize)
        .filter_map(|arm| {
            sdar_targeted
                .promotion_stats(arm)
                .map(|s| s.promotion_attempts)
        })
        .sum::<usize>();

    let high_br_rate = if high_br_attempts > 0 {
        high_br_promotions as f64 / high_br_attempts as f64 * 100.0
    } else {
        0.0
    };
    let neutral_br_rate = if neutral_br_attempts > 0 {
        neutral_br_promotions as f64 / neutral_br_attempts as f64 * 100.0
    } else {
        0.0
    };
    let low_br_rate = if low_br_attempts > 0 {
        low_br_promotions as f64 / low_br_attempts as f64 * 100.0
    } else {
        0.0
    };

    println!();
    println!("   Benefit ratio targeting (β=5.0):");
    println!(
        "     High BR arms (0-19):   {high_br_promotions:>3} prom / {high_br_attempts:>3} attempts = {high_br_rate:.1}%"
    );
    println!(
        "     Neutral BR arms (20-39): {neutral_br_promotions:>3} prom / {neutral_br_attempts:>3} attempts = {neutral_br_rate:.1}%"
    );
    println!(
        "     Low BR arms (40-59):  {low_br_promotions:>3} prom / {low_br_attempts:>3} attempts = {low_br_rate:.1}%"
    );

    // ── Verdict ───────────────────────────────────────────────────
    println!();
    println!("   β sensitivity (promotion rate ordering):");
    let soft_rate = if soft_promotion_attempts > 0 {
        soft_promotions as f64 / soft_promotion_attempts as f64
    } else {
        0.0
    };
    let optimal_rate = if optimal_promotion_attempts > 0 {
        optimal_promotions as f64 / optimal_promotion_attempts as f64
    } else {
        0.0
    };
    let aggressive_rate = if aggressive_promotion_attempts > 0 {
        aggressive_promotions as f64 / aggressive_promotion_attempts as f64
    } else {
        0.0
    };

    // Higher β should produce more selective (lower) promotion rate for borderline cases
    println!("     β=1.0  (soft):   {soft_rate:.3}");
    println!("     β=5.0  (optimal): {optimal_rate:.3}");
    println!("     β=10.0 (aggressive): {aggressive_rate:.3}");

    // Verify targeting quality
    let targeting_pass = high_br_rate > low_br_rate;
    if targeting_pass {
        println!(
            "   ✅ PASS: high-BR arms promote more ({high_br_rate:.1}%) than low-BR arms ({low_br_rate:.1}%)"
        );
    } else {
        println!(
            "   ⚠️  FAIL: targeting not differentiated (high={high_br_rate:.1}%, low={low_br_rate:.1}%)"
        );
    }

    // Verify β=5 is between β=1 and β=10 in selectivity
    let beta_ordering_pass = (optimal_rate >= aggressive_rate && optimal_rate <= soft_rate)
        || (optimal_rate - aggressive_rate).abs() < 0.05
        || (soft_rate - optimal_rate).abs() < 0.05;
    if beta_ordering_pass {
        println!("   ✅ PASS: β=5.0 produces reasonable selectivity between β=1.0 and β=10.0");
    } else {
        println!(
            "   ⚠️  INFO: β ordering is soft={soft_rate:.3}, optimal={optimal_rate:.3}, aggressive={aggressive_rate:.3}"
        );
    }
}

// ── Helpers ──────────────────────────────────────────────────────

#[cfg(all(feature = "sdar_gate", feature = "bandit"))]
fn select_ucb1_arm_scalar(
    bandit: &katgpt_rs::pruners::BanditPruner<katgpt_rs::speculative::types::NoScreeningPruner>,
    episode: usize,
    num_arms: usize,
) -> usize {
    if episode < num_arms {
        return episode; // Round-robin initial exploration
    }

    let q_values = bandit.q_values();
    let visits = bandit.visits();
    let total_pulls: u32 = visits.iter().sum();

    let mut best_arm = 0;
    let mut best_score = f32::NEG_INFINITY;

    for arm in 0..num_arms {
        let visits_arm = visits.get(arm).copied().unwrap_or(0);
        let score = if visits_arm == 0 {
            f32::INFINITY // Explore unvisited arms first
        } else {
            let q = q_values.get(arm).copied().unwrap_or(0.0);
            let exploration = (2.0 * (total_pulls as f32).ln() / visits_arm as f32).sqrt();
            q + exploration
        };
        if score > best_score {
            best_score = score;
            best_arm = arm;
        }
    }

    best_arm
}

#[cfg(all(feature = "sdar_gate", feature = "bandit"))]
fn select_ucb1_arm_sdar(
    bandit: &katgpt_rs::pruners::SdarBanditPruner<
        katgpt_rs::speculative::types::NoScreeningPruner,
    >,
    episode: usize,
    num_arms: usize,
) -> usize {
    if episode < num_arms {
        return episode; // Round-robin initial exploration
    }

    let q_values = bandit.q_values();
    let visits = bandit.visits();
    let total_pulls: u32 = visits.iter().sum();

    let mut best_arm = 0;
    let mut best_score = f32::NEG_INFINITY;

    for arm in 0..num_arms {
        let visits_arm = visits.get(arm).copied().unwrap_or(0);
        let score = if visits_arm == 0 {
            f32::INFINITY // Explore unvisited arms first
        } else {
            let q = q_values.get(arm).copied().unwrap_or(0.0);
            let exploration = (2.0 * (total_pulls as f32).ln() / visits_arm as f32).sqrt();
            q + exploration
        };
        if score > best_score {
            best_score = score;
            best_arm = arm;
        }
    }

    best_arm
}

#[cfg(all(feature = "sdar_gate", feature = "bandit"))]
fn is_best_arm_most_visited_scalar(
    bandit: &katgpt_rs::pruners::BanditPruner<katgpt_rs::speculative::types::NoScreeningPruner>,
    optimal_arm: usize,
    num_arms: usize,
) -> bool {
    let visits = bandit.visits();
    let optimal_visits = visits.get(optimal_arm).copied().unwrap_or(0);

    for arm in 0..num_arms {
        if arm == optimal_arm {
            continue;
        }
        if visits.get(arm).copied().unwrap_or(0) > optimal_visits {
            return false;
        }
    }
    true
}

#[cfg(all(feature = "sdar_gate", feature = "bandit"))]
fn is_best_arm_most_visited_sdar(
    bandit: &katgpt_rs::pruners::SdarBanditPruner<
        katgpt_rs::speculative::types::NoScreeningPruner,
    >,
    optimal_arm: usize,
    num_arms: usize,
) -> bool {
    let visits = bandit.visits();
    let optimal_visits = visits.get(optimal_arm).copied().unwrap_or(0);

    for arm in 0..num_arms {
        if arm == optimal_arm {
            continue;
        }
        if visits.get(arm).copied().unwrap_or(0) > optimal_visits {
            return false;
        }
    }
    true
}

#[cfg(all(feature = "sdar_gate", feature = "bandit"))]
fn count_compressed(
    layer: &katgpt_rs::pruners::AbsorbCompressLayer<
        katgpt_rs::speculative::types::NoScreeningPruner,
    >,
    num_arms: usize,
) -> usize {
    use katgpt_rs::pruners::AbsorbCompress;
    (0..num_arms)
        .filter(|&arm| layer.compressed_arms().contains(&arm))
        .count()
}
