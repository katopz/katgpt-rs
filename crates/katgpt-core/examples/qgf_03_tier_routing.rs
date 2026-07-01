//! Plan 268 Phase 6 T13 — QGF tier routing demo.
//!
//! Shows the two-orthogonal-axis routing model that QGF exposes:
//!
//! 1. **Backend route** (`QgfComputeRoute`): which silicon services the
//!    gradient query — `CpuSimd`, `GpuBatch`, or `AneCritic`. Decided by
//!    `route_for(action_space_size, batch_size)`. O(1), no allocation.
//! 2. **Oracle tier** (Plasma/Hot/Warm/Cold/Freeze): which value function
//!    provides the gradient. Decided by the caller's choice of oracle struct.
//!    Independent of the backend route — a Plasma-tier oracle can run on
//!    CPU SIMD or ANE depending on action-space size.
//!
//! The two are orthogonal. This demo prints both axes for representative
//! game-NPC configurations so the engineer can see at a glance which
//! (tier, route) combination applies to their workload.
//!
//! # What's NOT in this demo
//!
//! Real backend dispatch (actual GPU / ANE forward passes) lives in
//! `riir-gpu` / `npc_ane_backend` — private runtimes outside katgpt-core.
//! This demo only shows the routing *decision* (the O(1) policy), not the
//! dispatch. The decision is the modelless primitive; the dispatch is the
//! integration layer.
//!
//! Run with: `cargo run --example qgf_03_tier_routing --features qgf_drafter --release`

#![cfg(feature = "qgf_drafter")]

use katgpt_core::qgf::{route_for, QgfComputeRoute};
use katgpt_core::traits::{NoGuidanceOracle, QGradientOracle};

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 268 T8/T9 — QGF tier + backend routing                      ║");
    println!("║  Paper: Zhou et al. 2026, arXiv:2606.11087                        ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Two orthogonal axes decide how a QGF gradient query is served:");
    println!();
    println!("  Axis 1 — Backend route (silicon): CpuSimd / GpuBatch / AneCritic");
    println!("           Decided by `route_for(action_space_size, batch_size)`.");
    println!("  Axis 2 — Oracle tier (value fn):  Plasma / Hot / Warm / Cold / Freeze");
    println!("           Decided by the caller's choice of oracle struct.");
    println!();

    // ── Axis 1: backend routing table ──────────────────────────────────────
    println!("── Axis 1: backend route (O(1) decision policy) ──");
    println!();
    println!("  Rules:");
    println!("    action_space < 1024            → CpuSimd  (small enough for SIMD dot)");
    println!("    batch ≥ 8 && action_space ≥ 1024 → GpuBatch  (amortize kernel launch)");
    println!("    otherwise                      → CpuSimd  (default safe path)");
    println!();
    println!("  {:<32} {:>14} {:>12} {:>14}",
        "workload", "action_space", "batch", "route");
    let workloads = [
        ("bomber npc (ternary)",        16,    1),
        ("dungeon npc (HLA D=8)",       64,    1),
        ("leo_all_goals (mid-game)",    256,   4),
        ("leo_all_goals (batch)",       256,   16),
        ("large action head",           1024,  8),
        ("huge latent (D=4096)",        4096,  1),
        ("huge latent batch",           4096,  16),
    ];
    for (name, a, b) in workloads {
        let r = route_for(a, b);
        println!("  {:<32} {:>14} {:>12} {:>14}",
            name, a, b, route_str(r));
    }
    println!();
    println!("  Reading: most game-NPC workloads land on CpuSimd (small action space).");
    println!("  GpuBatch only wins when BOTH action_space ≥ 1024 AND batch ≥ 8.");
    println!();

    // ── Axis 2: oracle tiers ───────────────────────────────────────────────
    println!("── Axis 2: oracle tier (value-function source) ──");
    println!();
    println!("  {:<10} {:<32} {:<14} {:<14}",
        "tier", "oracle struct", "feature", "confidence");
    println!("  {}", "─".repeat(72));
    let tiers = [
        ("Plasma",  "ActionBridgeOracle",  "action_bridge",   "1.0"),
        ("Plasma",  "FlowFieldOracle",     "flow_field_nav",  "1.0"),
        ("Hot",     "LeoHeadOracle",       "leo_all_goals",   "1.0"),
        ("Freeze",  "BfnProxyOracle",      "(always)",        "0.3"),
        ("Freeze",  "NoGuidanceOracle",    "qgf_oracle",      "0.0"),
    ];
    for (tier, oracle, feat, conf) in tiers {
        println!("  {:<10} {:<32} {:<14} {:<14}",
            tier, oracle, feat, conf);
    }
    println!();
    println!("  Latency targets per tier (paper §4, our T9 mapping):");
    println!("    Plasma  < 100ns    ActionBridge ternary i8 + f32 Q dot");
    println!("    Hot     < 1µs      LeoHead cached f32 Q-values");
    println!("    Warm    ~ 1ms      GPU batched Q-critic forward");
    println!("    Cold    ~ 10ms     Turso Q-table snapshots");
    println!("    Freeze  = 0ns      NoGuidanceOracle (pure BC reference)");
    println!();

    // ── Freeze-tier live demo: NoGuidanceOracle ────────────────────────────
    //
    // The freeze tier is always-available: NoGuidanceOracle is in the
    // `qgf_oracle` feature, no external runtime needed. It's the graceful-
    // degradation path when no trained critic is loaded.
    println!("── Freeze-tier demo: NoGuidanceOracle ──");
    let oracle = NoGuidanceOracle;
    let mut buf = [0.0f32; 4];
    oracle.q_gradient_into(&(), &(), &mut buf);
    let conf = oracle.confidence(&());
    println!("  q_gradient_into(&mut [0.0; 4]) → {:?}", buf);
    println!("  confidence() = {conf}");
    println!();
    println!("  Reading: zero gradient + zero confidence → adaptive guidance weight");
    println!("  collapses to 0 → output is byte-identical to pure BC reference.");
    println!("  This is the freeze-tier equivalence (G2 of GOAT gate).");
    println!();

    // ── Combinations matrix ────────────────────────────────────────────────
    println!("── (tier × route) combinations for representative workloads ──");
    println!();
    println!("  {:<28} {:<10} {:<14} {:<32}",
        "workload", "tier", "route", "expected cost");
    println!("  {}", "─".repeat(86));
    let combos = [
        ("bomber npc 1v1",          "Plasma",  16,    1,  "< 100ns (CpuSimd)"),
        ("dungeon npc small batch", "Plasma",  64,    4,  "< 100ns (CpuSimd)"),
        ("leo_all_goals (single)",  "Hot",     256,   1,  "< 1µs   (CpuSimd)"),
        ("leo_all_goals (batch=8)", "Hot",     256,   8,  "< 1µs   (CpuSimd)"),
        ("large latent (D=4096)",   "Warm",    4096,  1,  "~ 1ms   (CpuSimd fallback)"),
        ("large latent (batch=16)", "Warm",    4096,  16, "~ 1ms   (GpuBatch)"),
        ("episode-end snapshot",    "Cold",    4096,  64, "~ 10ms  (GpuBatch)"),
        ("engine boot, no critic",  "Freeze",  16,    1,  "0ns     (no-op)"),
    ];
    for (name, tier, a, b, cost) in combos {
        let r = route_for(a, b);
        println!("  {:<28} {:<10} {:<14} {:<32}",
            name, tier, route_str(r), cost);
    }
    println!();
    println!("  Reading: Plasma/Hot are hot-path tiers (every step), Cold/Freeze");
    println!("  are amortization/safety tiers. Warm is the batch-training path.");
    println!();

    println!("── Summary ──");
    println!("  • Backend route is O(1): two comparisons, no allocation, < 100ns/call.");
    println!("  • Oracle tier is caller-chosen: struct choice encodes the value fn.");
    println!("  • Freeze tier (NoGuidanceOracle) is always-available: graceful degradation.");
    println!("  • Real backend dispatch lives in riir-gpu / npc_ane_backend (private).");
    println!();
    println!("See .plans/268_qgf_test_time_q_guided_flow.md Phase 4 T8/T9.");
    println!("See qgf/route.rs source for the routing policy (33 lines).");
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn route_str(r: QgfComputeRoute) -> &'static str {
    match r {
        QgfComputeRoute::CpuSimd => "CpuSimd",
        QgfComputeRoute::GpuBatch => "GpuBatch",
        QgfComputeRoute::AneCritic => "AneCritic",
    }
}
