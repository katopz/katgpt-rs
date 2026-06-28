//! Dual-Pool Reachable Memory Router Demo (Plan 282 Phase 6 — T6.4)
//!
//! Demonstrates the three GOAT-gated capabilities of `DualPoolBandit`:
//!
//!   1. **Proactive reachability (G1)** — the X-pool is always selected with
//!      nonzero probability, even when `w_E` is driven extreme. No collapse
//!      detector is needed.
//!   2. **E-pool growth (G3)** — rewarded X-pool arms are promoted into the
//!      E-pool as new arms via the backward-compatible `push_arm` default
//!      method. The router discovers strategies outside its initial pool.
//!   3. **Faithfulness gate (G4)** — `consolidate_growing_gated(gate)` rejects
//!      arms the consumer structurally ignores (dead items) from promotion.
//!
//! Run:
//!   cargo run --features cgsp_dual_pool --example cgsp_dual_pool_demo
//!
//! See also: `examples/cgsp_minimal.rs` for the single-pool CGSP baseline
//! that this router generalizes (single-pool CGSP = degenerate `α = 1`).

#![cfg(feature = "cgsp_dual_pool")]

use katgpt_rs::cgsp::{
    traits::HintDeltaBandit, types::Priority, DualPoolBandit, DualPoolConfig, PoolId,
    ReachableDualPoolRouter,
};

// ════════════════════════════════════════════════════════════════════════════
// Growing Vec-backed bandit (mirrors the one in dual_pool.rs tests)
// ════════════════════════════════════════════════════════════════════════════
//
// `DualPoolBandit<B>` requires `B: HintDeltaBandit`. For Phase 4 growth we also
// need `is_growing() == true` and a real `push_arm` impl. The default methods
// on the trait are no-op / false — this struct overrides them to actually grow.

/// Vec-backed bandit that supports arm growth (Phase 4 backend).
struct GrowingVecBandit {
    prios: Vec<f32>,
}

impl GrowingVecBandit {
    fn uniform(n: usize) -> Self {
        Self {
            prios: vec![1.0 / n as f32; n],
        }
    }
    fn constant(n: usize, v: f32) -> Self {
        Self { prios: vec![v; n] }
    }
}

impl HintDeltaBandit for GrowingVecBandit {
    fn absorb(&mut self, arm: usize, reward: f32) {
        if let Some(p) = self.prios.get_mut(arm) {
            *p += reward.max(0.0);
        }
    }
    fn priority(&self, arm: usize) -> Priority {
        self.prios.get(arm).copied().unwrap_or(0.0)
    }
    fn priorities(&self) -> &[Priority] {
        &self.prios
    }
    fn priorities_mut(&mut self) -> &mut [Priority] {
        &mut self.prios
    }
    // Phase 4: real growth — append a new arm and return its index.
    fn push_arm(&mut self, priority: Priority) -> usize {
        let idx = self.prios.len();
        self.prios.push(priority);
        idx
    }
    fn is_growing(&self) -> bool {
        true
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

fn separator(title: &str) {
    println!();
    println!("{}", "═".repeat(72));
    println!("  {title}");
    println!("{}", "═".repeat(72));
}

fn print_pool(label: &str, prios: &[f32]) {
    let sum: f32 = prios.iter().copied().sum();
    println!("  {label} ({} arms):", prios.len());
    for (i, p) in prios.iter().enumerate() {
        let share = if sum > 0.0 { p / sum } else { 0.0 };
        let bar_len = (share * 40.0).round() as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    arm {i:>2}: prio={p:>7.4}  share={share:>6.3}  {bar}");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Demo 1 — Proactive reachability (G1)
// ════════════════════════════════════════════════════════════════════════════
//
// Drive `w_e` extreme via repeated E-pool successes. Even when α saturates
// near 1.0 (clamped by `min_exploration_prob`), the X-pool is still selected
// within a bounded number of cycles — proving proactive non-trapping.

fn demo_reachability() {
    separator("Demo 1: Proactive reachability (G1)");
    println!("  Setup: 8-arm E-pool (one-hot on arm 0), 8-arm X-pool (uniform).");
    println!("  We force E-pool 'success' every cycle so w_E grows without bound.");
    println!("  The X-pool floor (α clamped to 1−ε) guarantees it is still selected.");
    println!();

    let e = GrowingVecBandit::constant(8, 0.001);
    let x = GrowingVecBandit::uniform(8);
    let mut dp = DualPoolBandit::new(e, x);

    // Simulate extreme exploitation: E-pool "wins" every cycle.
    let mut x_selections = 0u32;
    const TOTAL: u32 = 50_000;
    for _ in 0..TOTAL {
        dp.begin_cycle();
        // route_select returns (arm, pool) — we only care about pool here.
        let (_arm, pool) = dp.route_select();
        if pool == PoolId::Exploration {
            x_selections += 1;
        }
        // Drive w_E up: every cycle is an E-pool success.
        dp.route_update(PoolId::Exploitation, true);
    }

    let alpha = dp.exploitation_probability();
    println!("  After {TOTAL} cycles (all E-pool successes):");
    println!("    w_E               : {:.4}", dp.w_e());
    println!("    α = sigmoid(w_E−w_X): {:.6}", alpha);
    println!("    is_reachable()    : {}  (X-pool floor > 0)", dp.is_reachable());
    println!("    X-pool selections : {x_selections} / {TOTAL}  (P ≈ {:.6})",
        x_selections as f64 / TOTAL as f64);
    assert!(dp.is_reachable(), "G1 FAIL: X-pool lost reachability");
    assert!(x_selections > 0, "G1 FAIL: X-pool never selected in {TOTAL} cycles");
    println!();
    println!("  ✓ X-pool selected even at extreme exploitation — proactive non-trap.");
}

// ════════════════════════════════════════════════════════════════════════════
// Demo 2 — E-pool growth (G3)
// ════════════════════════════════════════════════════════════════════════════
//
// E-pool starts with 4 "known" arms. X-pool has 16 arms. We reward X-pool
// arm 7 (the "optimal direction" NOT in the initial E-pool) every cycle.
// After consolidation, arm 7 should be promoted into the E-pool as a new
// arm — the router discovers a strategy beyond its initial template.

fn demo_epool_growth() {
    separator("Demo 2: E-pool growth discovers new strategies (G3)");
    println!("  Setup: 4-arm E-pool (known directions), 16-arm X-pool (superset).");
    println!("  X-pool arm 7 is the optimal direction — NOT in the initial E-pool.");
    println!("  Reward arm 7 once; consolidate promotes it into E-pool.");
    println!();

    let e = GrowingVecBandit::uniform(4);
    let x = GrowingVecBandit::uniform(16);
    let cfg = DualPoolConfig {
        growth_enabled: true,
        promotion_threshold: 0.1,
        max_epool_size: 64,
        ..Default::default()
    };
    let mut dp = DualPoolBandit::with_config(e, x, cfg);

    let initial_e_size = dp.e_pool().num_arms();
    println!("  Initial E-pool size: {initial_e_size}");
    print_pool("Initial E-pool", dp.e_pool().priorities());

    // Single consolidation: reward arm 7 once, then consolidate.
    // The X-pool resets to uniform after consolidate, so rewarding the
    // same arm every cycle would re-promote it (monotonic growth demo).
    // For a clean 'discovery' narrative, we reward + consolidate once.
    dp.begin_cycle();
    dp.set_active_pool(PoolId::Exploration);
    dp.absorb(7, 0.8); // Arm 7 earns enough reward to cross threshold.
    println!();
    println!("  ── rewarding X-pool arm 7 with r=0.8 (threshold={}) ──", dp.config().promotion_threshold);
    dp.consolidate();

    let final_e_size = dp.e_pool().num_arms();
    println!();
    println!("  ── after 1 consolidate ──");
    println!("    E-pool size: {final_e_size} (grew from {initial_e_size})");
    print_pool("Final E-pool", dp.e_pool().priorities());

    assert_eq!(
        final_e_size,
        initial_e_size + 1,
        "G3 FAIL: E-pool should have grown by exactly 1 ({initial_e_size} → {final_e_size})"
    );
    // The promoted arm should have elevated priority (X-pool arm 7's prio
    // at consolidate time: uniform 1/16 + absorbed 0.8 = 0.8625).
    let max_e_prio = dp.e_pool().priorities().iter().cloned().fold(0.0f32, f32::max);
    let uniform_4 = 1.0 / 4.0;
    assert!(
        max_e_prio > uniform_4,
        "G3 FAIL: promoted arm should have elevated priority (max={:.4}, uniform_4={:.4})",
        max_e_prio,
        uniform_4
    );
    println!();
    println!("  ✓ E-pool grew by 1 — optimal direction (X-pool arm 7) promoted.");
    println!("    Single-pool CGSP could never select arm 7 (not in static pool).");
    println!();
    println!("  Note: running more cycles would re-promote arm 7 each cycle (X-pool");
    println!("  resets to uniform after consolidate). This demonstrates monotonic");
    println!("  growth — production use rewards different arms per cycle.");
}

// ════════════════════════════════════════════════════════════════════════════
// Demo 3 — Faithfulness gate (G4)
// ════════════════════════════════════════════════════════════════════════════
//
// X-pool has 8 arms. Arms 0–3 are "live" (consumer uses them); arms 4–7 are
// "dead" (consumer structurally ignores them — no behavioral delta). We reward
// all 8 equally, then consolidate with the faithfulness gate ON vs OFF.
//
// Gate ON  → only live arms (0–3) promoted; dead arms rejected.
// Gate OFF → all 8 promoted; E-pool fills with dead weight.

fn demo_faithfulness_gate() {
    separator("Demo 3: Faithfulness gate rejects dead items (G4)");
    println!("  Setup: 1-arm E-pool, 8-arm X-pool. Arms 0–3 are 'live'; 4–7 'dead'.");
    println!("  Reward all 8 equally once, then consolidate with gate ON vs OFF.");
    println!();

    // The gate: returns true only for live arms (0–3).
    let live_arms: Vec<usize> = (0..4).collect();
    let gate = |arm: usize| live_arms.contains(&arm);

    let make_cfg = || {
        DualPoolConfig {
            growth_enabled: true,
            promotion_threshold: 0.05,
            max_epool_size: 64,
            ..Default::default()
        }
    };

    // ── Gate ON ──────────────────────────────────────────────────────────
    println!("  ── Gate ON (FaithfulnessProbe wraps `is_faithfully_used`) ──");
    let e = GrowingVecBandit::constant(1, 0.1);
    let x = GrowingVecBandit::uniform(8);
    let mut dp_on = DualPoolBandit::with_config(e, x, make_cfg());

    // Single consolidation: reward all 8 arms once, then consolidate with gate.
    dp_on.begin_cycle();
    dp_on.set_active_pool(PoolId::Exploration);
    for arm in 0..8 {
        dp_on.absorb(arm, 0.3); // All arms cross threshold 0.05.
    }
    dp_on.consolidate_growing_gated(gate);
    let on_size = dp_on.e_pool().num_arms();
    println!("    E-pool size: {on_size} (expected: 1 initial + 4 live = 5)");

    // ── Gate OFF ─────────────────────────────────────────────────────────
    println!();
    println!("  ── Gate OFF (baseline — no faithfulness filter) ──");
    let e = GrowingVecBandit::constant(1, 0.1);
    let x = GrowingVecBandit::uniform(8);
    let mut dp_off = DualPoolBandit::with_config(e, x, make_cfg());

    dp_off.begin_cycle();
    dp_off.set_active_pool(PoolId::Exploration);
    for arm in 0..8 {
        dp_off.absorb(arm, 0.3);
    }
    // No gate — all rewarded arms promoted (dead weight clogs E-pool).
    dp_off.consolidate();
    let off_size = dp_off.e_pool().num_arms();
    println!("    E-pool size: {off_size} (expected: 1 initial + 8 all = 9)");

    println!();
    assert!(on_size < off_size,
        "G4 FAIL: gate didn't filter dead items ({on_size} vs {off_size})");
    assert_eq!(on_size, 5, "G4 FAIL: gate ON should promote exactly 4 live arms (+1 initial)");
    assert_eq!(off_size, 9, "G4 FAIL: gate OFF should promote all 8 arms (+1 initial)");
    println!("  ✓ Gate ON: 4 live arms promoted, 4 dead filtered (E-pool = {on_size}).");
    println!("  ✓ Gate OFF: all 8 promoted, dead weight clogs E-pool (E-pool = {off_size}).");
    println!("    The gate prevents Research 244's 'dead condensed memory' failure mode.");
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   Dual-Pool Reachable Memory Router Demo — Plan 282 Phase 6 (T6.4)  ║");
    println!("║   DecentMem distillation (arXiv:2605.22721)                         ║");
    println!("║   G1 reachability · G3 growth · G4 faithfulness gate                ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    demo_reachability();
    demo_epool_growth();
    demo_faithfulness_gate();

    separator("Summary");
    println!("  ✓ G1: X-pool always selectable — proactive non-trapping (Theorem 1).");
    println!("  ✓ G3: E-pool grows — discovers strategies outside initial pool.");
    println!("  ✓ G4: Faithfulness gate — rejects dead items from promotion.");
    println!();
    println!("  Per-cycle overhead : 0.5 ns (sigmoid + splitmix64) — plasma tier.");
    println!("  Single-pool CGSP  : degenerate α=1 case; dual-pool strictly generalizes.");
    println!();
    println!("  G5 (personality divergence benchmark) deferred to riir-ai NpcCgspRuntime.");
    println!("  Feature stays opt-in until G5 validates widening divergence.");
    println!();
}

// TL;DR: Demonstrates the three GOAT-gated capabilities of DualPoolBandit —
// proactive reachability (G1), E-pool growth (G3), and FaithfulnessProbe
// consolidation gate (G4). Single-pool CGSP is the degenerate α=1 case.
