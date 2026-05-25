//! SpecHop Cost Model Demo — α/β/p → k* and RelLat Prediction (Plan 131, T41)
//!
//! Demonstrates the theoretical cost model from the SpecHop paper (arXiv:2605.21965):
//! - Theorem 2: Optimal thread count k* = ⌈(1+β)/(α+β)⌉
//! - Theorem 3: Oracle relative latency RelLat* = 1 − p(1−α)/(1+β)
//! - Theorem 4: Bounded-window RelLat_k and starvation probability P_starve
//!
//! Also shows the configurator reward function that decides whether to activate
//! SpecHop based on measured inference statistics.
//!
//! Run: `cargo run --example spechop_02_cost_model --features spechop`

use katgpt_rs::spechop::{
    InferenceStats, SpecHopConfig, bounded_rel_lat, compute_optimal_k, oracle_rel_lat,
    should_activate_spechop, spechop_configurator_reward, starvation_prob,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SpecHop Cost Model — Paper Examples (arXiv:2605.21965)    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Theorem 2: Optimal Thread Count ──────────────────────────
    println!("═══ Theorem 2: Optimal Thread Count k* ═══");
    println!("  k* = ⌈(1 + β) / (α + β)⌉");
    println!();

    let paper_examples = [
        // (α, β, expected k*)
        (0.2, 0.15, 4),
        (0.3, 0.75, 2),
        (0.1, 0.5, 3),
        (0.5, 0.5, 2),
        (0.05, 0.1, 7),
    ];

    for (alpha, beta, expected) in paper_examples {
        let k = compute_optimal_k(alpha, beta);
        let status = if k == expected { "✓" } else { "✗" };
        println!("  α={alpha:.2}, β={beta:.2} → k*={k} (paper: {expected}) {status}");
    }
    println!();

    // ── Theorem 3: Oracle Relative Latency ───────────────────────
    println!("═══ Theorem 3: Oracle Relative Latency RelLat* ═══");
    println!("  RelLat* = 1 − p(1−α)/(1+β)");
    println!("  (Best-case with unbounded speculative window)");
    println!();

    let alpha = 0.2;
    let beta = 0.15;

    for p in [0.3, 0.5, 0.7, 0.9, 1.0] {
        let rel = oracle_rel_lat(alpha, beta, p);
        let speedup = 1.0 / rel;
        println!(
            "  p={p:.1} → RelLat*={rel:.4} ({speedup:.2}× speedup, {}% latency reduction)",
            ((1.0 - rel) * 100.0) as usize
        );
    }
    println!();

    // ── Theorem 4: Bounded-Window RelLat ─────────────────────────
    println!("═══ Theorem 4: Bounded-Window RelLat_k ═══");
    println!("  RelLat_k approaches RelLat* as k → ∞");
    println!();

    let alpha = 0.2;
    let beta = 0.15;
    let p = 0.7;
    let oracle = oracle_rel_lat(alpha, beta, p);

    println!("  Oracle RelLat* = {oracle:.4}");
    println!();

    println!("  k │ RelLat_k │ Gap from oracle");
    println!("  ──┼──────────┼─────────────────");

    for k in [1, 2, 3, 4, 8, 16, 32, 64] {
        let bounded = bounded_rel_lat(alpha, beta, p, k);
        let gap = bounded - oracle;
        let bar = "█".repeat((gap * 200.0) as usize);
        println!("  {k:>2} │ {bounded:.4}  │ +{gap:.4} {bar}");
    }
    println!();

    // ── Starvation Probability ───────────────────────────────────
    println!("═══ Starvation Probability (Theorem 4, CLT) ═══");
    println!("  P_starve ≈ Φ((1+β − k(α+β)) / (ν√(kα² + (k−1)β² + 1)))");
    println!();

    let k_optimal = compute_optimal_k(alpha, beta);
    let volatility = 0.4;

    println!("  k*={k_optimal}, ν={volatility}");
    println!();

    println!("  k │ P_starve │ Status");
    println!("  ──┼──────────┼──────────");

    for k in [2, 3, k_optimal, 6, 8, 16] {
        let p_starve = starvation_prob(k, alpha, beta, volatility);
        let status = if p_starve < 0.05 {
            "✓ < 5%"
        } else if p_starve < 0.10 {
            "~ < 10%"
        } else {
            "✗ > 10%"
        };
        println!("  {k:>2} │ {p_starve:.4}   │ {status}");
    }
    println!();

    // ── Configurator Reward ──────────────────────────────────────
    println!("═══ Configurator Reward ═══");
    println!("  reward = latency_reduction / α");
    println!("  Activate SpecHop when reward > 1.0");
    println!();

    let scenarios = [
        ("Tool-bound (α=0.2, β=0.15)", 0.2, 0.15, 0.7, k_optimal),
        ("Decode-bound (α=0.2, β=2.0)", 0.2, 2.0, 0.7, 4),
        ("Slow speculator (α=0.5, β=0.15)", 0.5, 0.15, 0.7, 2),
        ("Perfect spec (α=0.1, β=0.15, p=1.0)", 0.1, 0.15, 1.0, 4),
        ("Poor spec (α=0.2, β=0.15, p=0.1)", 0.2, 0.15, 0.1, 4),
    ];

    for (label, a, b, p, k) in scenarios {
        let reward = spechop_configurator_reward(a, b, p, k);
        let rel = oracle_rel_lat(a, b, p);
        let decision = if reward > 1.0 { "ACTIVATE" } else { "SKIP" };
        println!("  {label}");
        println!("    k={k}, p={p}, RelLat={rel:.3}, reward={reward:.2} → {decision}");
    }
    println!();

    // ── Auto-k from Measured Stats ───────────────────────────────
    println!("═══ Auto-k from Measured Stats ═══");
    println!();

    // Scenario 1: Tool-bound web search agent
    let mut stats = InferenceStats::new();
    // Simulate 50 observations: spec=20ns, target=100ns, decode=15ns, 70% hit rate
    for i in 0..50 {
        let hit = (i % 10) < 7; // 70% hit rate
        stats.observe(20.0, 100.0, 15.0, hit);
    }

    println!("  Web Search Agent (50 observations):");
    println!(
        "    Measured α = {:.3} (spec/target latency ratio)",
        stats.alpha()
    );
    println!("    Measured β = {:.3} (decode/target ratio)", stats.beta());
    println!("    Measured p = {:.3} (speculator hit rate)", stats.p());

    match stats.auto_k() {
        Some(k) => println!("    Auto-k* = {k}"),
        None => println!("    Auto-k* = insufficient data"),
    }

    match should_activate_spechop(&stats) {
        Some(k) => println!("    Decision: ACTIVATE with k={k} ✓"),
        None => println!("    Decision: SKIP (conditions not met)"),
    }
    println!();

    // Scenario 2: Code generation (decode-bound)
    let mut stats_decode = InferenceStats::new();
    // Simulate: spec=20ns, target=100ns, decode=200ns (heavy generation)
    for i in 0..50 {
        let hit = (i % 10) < 7;
        stats_decode.observe(20.0, 100.0, 200.0, hit);
    }

    println!("  Code Generation Agent (50 observations):");
    println!("    Measured α = {:.3}", stats_decode.alpha());
    println!("    Measured β = {:.3}", stats_decode.beta());
    println!("    Measured p = {:.3}", stats_decode.p());

    match should_activate_spechop(&stats_decode) {
        Some(k) => println!("    Decision: ACTIVATE with k={k} ✓"),
        None => println!("    Decision: SKIP (decode-bound, β > 0.8)"),
    }
    println!();

    // ── Config Quick Reference ───────────────────────────────────
    println!("═══ SpecHopConfig Quick Reference ═══");
    println!();

    let configs = [
        (
            "Conservative",
            SpecHopConfig {
                alpha: 0.2,
                beta: 0.15,
                p: 0.7,
                k: Some(4),
                volatility: 0.4,
            },
        ),
        (
            "Aggressive",
            SpecHopConfig {
                alpha: 0.1,
                beta: 0.1,
                p: 0.9,
                k: None, // auto-compute
                volatility: 0.3,
            },
        ),
        (
            "Cheap speculator",
            SpecHopConfig {
                alpha: 0.05,
                beta: 0.5,
                p: 0.5,
                k: None,
                volatility: 0.5,
            },
        ),
    ];

    for (name, config) in configs {
        let k = config.effective_k();
        let oracle = oracle_rel_lat(config.alpha, config.beta, config.p);
        let bounded = bounded_rel_lat(config.alpha, config.beta, config.p, k);
        let p_starve = starvation_prob(k, config.alpha, config.beta, config.volatility);
        let speedup = 1.0 / bounded;

        println!("  {name}:");
        println!(
            "    α={:.2}, β={:.2}, p={:.1}, k={k}",
            config.alpha, config.beta, config.p
        );
        println!("    RelLat*={oracle:.3}, RelLat_{k}={bounded:.3} ({speedup:.2}× speedup)");
        println!("    P_starve={p_starve:.4}");
        println!();
    }

    println!("══════════════════════════════════════════════════════════════");
    println!("  Summary: SpecHop is beneficial when α is low (fast speculator)");
    println!("  and β is moderate (tool-bound workloads with non-trivial tool calls).");
    println!("  The cost model provides principled thread sizing via k* and predicts");
    println!("  achievable latency reduction before deploying speculation.");
    println!("══════════════════════════════════════════════════════════════");
}
