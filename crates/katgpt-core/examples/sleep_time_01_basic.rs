//! Plan 334 Phase 3 T3.1 — minimal Sleep-Time Query Anticipator walkthrough.
//!
//! End-to-end runnable demo of the open `sleep_time` primitive
//! (arXiv:2504.13171 — Lin et al. Letta/Berkeley). Shows the full pipeline
//! in four clearly labelled sections:
//!
//! 1. **Construct K=4 anticipated-query directions** (hardcoded catalog). The
//!    direction catalog is the per-domain IP that lives in the consumer — here
//!    we hardcode four toy directions for the walkthrough.
//! 2. **Run `SleepTimeAnticipator::anticipate`** on a context `c`. Emits the
//!    `AnticipatedQuerySet` artifact (the paper's "c'"): one slot per direction
//!    carrying the precomputed latent answer `z_i` and its predictability score
//!    `p_i`. The artifact is BLAKE3-committed (so a tamper is detectable).
//! 3. **Run wake-time `consume()`** on two queries — one predictable (high
//!    gate → mostly the precomputed slot), one unpredictable (low gate →
//!    mostly fresh compute). Print the sigmoid-gated blend explicitly so the
//!    smooth blend is visible (no hard argmax switch — per AGENTS.md).
//! 4. **Print the `AmortizationCostModel`** — the paper's §5.3 amortization
//!    factor at N=1 player vs N=10 players. This is the headline result: one
//!    sleep-time compute serves many wake-time consumers.
//!
//! # Why this is modelless (katgpt-rs mandate)
//!
//! Every step is closed-form algebra:
//! - `IdentityFunctorOp`: `z_i = c + dir_i` (no training, no backprop).
//! - `DotPredictabilityScorer`: `p_i = sigmoid(α · dot(c, dir_i) + β)`.
//! - `consume()`: dot-product + sigmoid gate + linear blend.
//!
//! Real consumers (riir-ai Plan 341) swap `IdentityFunctorOp` for
//! `latent_functor` extraction or `karc_forecast`, and swap the scorer for
//! a curiosity-inversion variant — see `sleep_time_02_curiosity_inversion.rs`.
//!
//! Run with: `cargo run --example sleep_time_01_basic --features sleep_time_anticipation --release`

use katgpt_core::sleep_time::{
    consume, consume_gate, AmortizationCostModel, AnticipatedQueryDir, DotPredictabilityScorer,
    IdentityFunctorOp, SleepTimeAnticipator, SleepTimeScratch,
};

// ── Constants ───────────────────────────────────────────────────────────────

/// Latent dim. Small (D=4) for didactic clarity. Real HLA-scale consumers
/// use D=8; style-weight-scale consumers use D=64.
const D: usize = 4;

/// Catalog size. Paper uses K≤10; we expect K≤8 per NPC. K=4 here for clarity.
const K: usize = 4;

/// Gate threshold τ. Higher = require higher predictability to use the cache.
/// τ=0.5 means "use the cache iff dot(c, dir) > 0" under the default scorer.
const TAU: f32 = 0.5;

/// Gate sharpness β. Higher = sharper transition around τ.
const BETA: f32 = 4.0;

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 334 Phase 3 T3.1 — Sleep-Time Query Anticipator (basic)   ║");
    println!("║  Paper: Lin et al. 2025, arXiv:2504.13171 (Letta/Berkeley)      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Setup: D={D} latent dims, K={K} catalog directions, τ={TAU}, β={BETA}.");
    println!("Sleep-time op: IdentityFunctorOp (z_i = c + dir_i — synthetic default).");
    println!("Scorer: DotPredictabilityScorer (p_i = sigmoid(α·dot(c, dir_i) + β)).");
    println!();

    // ── Step 1: hardcoded anticipated-query direction catalog ──────────────
    //
    // In a real consumer (riir-ai Plan 341), these would be the per-NPC-type
    // direction vectors — shopkeeper-zone queries, quest-giver-zone queries,
    // lore-NPC queries, etc. Here we hardcode four toy directions to show the
    // shape of the catalog without pulling in game IP.
    let dirs: [AnticipatedQueryDir<D>; K] = [
        AnticipatedQueryDir::new([1.0, 0.0, 0.0, 0.0]), // "what's for sale?"
        AnticipatedQueryDir::new([0.0, 1.0, 0.0, 0.0]), // "where is the dungeon?"
        AnticipatedQueryDir::new([0.0, 0.0, 1.0, 0.0]), // "tell me about the king"
        AnticipatedQueryDir::new([0.0, 0.0, 0.0, 1.0]), // "do you have a quest?"
    ];

    println!("── Step 1: anticipated-query catalog (hardcoded) ──");
    for (i, d) in dirs.iter().enumerate() {
        println!("  dir[{i}] = {:?}  blake3={}…", d.direction, hex_short(&d.blake3));
    }
    println!();

    // ── Step 2: anticipate() — emit the c' artifact ────────────────────────
    //
    // The sleep-time operator S(c) → c'. For each direction i, the anticipator
    // runs the op (z_i = c + dir_i here) and scores predictability. The output
    // is the AnticipatedQuerySet — BLAKE3-committed, reusable across consumers.
    let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
        op: IdentityFunctorOp,
        scorer: DotPredictabilityScorer::default(), // α=1.0, β=0.0
        budgets: [100, 100, 100, 100],              // sleep-time token budgets
        tau: TAU,
        beta: BETA,
    };

    // Context: aligned with dir[0] ("what's for sale?") and *anti-aligned*
    // with dir[2] / dir[3]. So slot 0 will have high predictability; slots 2,3
    // will have low predictability — this is what makes Step 3's high/low-gate
    // contrast visible (the gate reads the *best-matching* slot's p_i).
    let c: [f32; D] = [2.0, 0.1, -2.0, -2.0];

    let mut scratch = SleepTimeScratch::new();
    let c_prime = anticipator.anticipate(&c, &dirs, &mut scratch);

    println!("── Step 2: anticipate() → c' artifact (sleep-time compute) ──");
    println!("  context c = {:?}", c);
    println!("  c'.blake3 = {}…  (commits all slot bytes)", hex_short(&c_prime.blake3));
    println!("  c'.version = {}", c_prime.version);
    println!("  commitment verifies: {}", c_prime.verify_commitment());
    println!();
    println!("  {:<6} {:<24} {:<24} {:>14}", "slot", "direction", "precomputed z_i", "predictability");
    for (i, slot) in c_prime.slots.iter().enumerate() {
        println!(
            "  [{i:<4}] {:<24} {:<24} {:>14.6}",
            fmt_array(&slot.dir.direction),
            fmt_array(&slot.precomputed),
            slot.predictability
        );
    }
    println!();
    println!("  Reading: slot 0 (aligned with c) has the highest p_i. Slots 2,3");
    println!("  (anti-aligned with c) have low p_i. This is what Step 3's high/low-gate");
    println!("  contrast rides on: the gate reads the best-matching slot's p_i.");
    println!();

    // ── Step 3: consume() — wake-time gated blend ──────────────────────────
    //
    // Two queries: one predictable (close to dir[0]), one unpredictable
    // (orthogonal to the whole catalog). The gate = sigmoid(β · (p − τ)).
    // Predictable query → gate ≈ 1 → output is the precomputed slot.
    // Unpredictable query → gate ≈ 0 → output is fresh_think (fallback).
    // In between, a smooth blend — never a hard argmax switch (AGENTS.md).

    // fresh_think fallback: closed-form, no real compute (this is a demo).
    // In a real consumer this would be the LLM-in-the-loop or KARC forecast.
    let fresh_think = |q: &[f32; D]| {
        let mut out = [0.0f32; D];
        for j in 0..D {
            out[j] = -q[j]; // toy "fresh answer" so the blend is visible
        }
        out
    };

    let q_predictable: [f32; D] = [3.0, 0.0, 0.0, 0.0]; // matches slot 0 (high p)
    let q_unpredictable: [f32; D] = [0.0, 0.0, 3.0, 0.0]; // matches slot 2 (low p)

    println!("── Step 3: consume() — wake-time sigmoid-gated blend ──");
    println!("  τ={TAU}, β={BETA}. gate = sigmoid(β·(p_{{i*}} − τ)).");
    println!("  fresh_think(q) = −q (toy fallback so the blend is visible).");
    println!();

    let (best_p, gate_p) = consume_gate(&q_predictable, &c_prime, TAU, BETA);
    let out_p = consume(&q_predictable, &c_prime, TAU, BETA, fresh_think);
    println!("  predictable query q = {:?}", q_predictable);
    println!("    best slot i* = {best_p}  predictability p_{{i*}} = {:.6}", c_prime.slots[best_p].predictability);
    println!("    gate = {:.6}  (high → use precomputed)", gate_p);
    println!("    out  = {}  ≈ precomputed slot (gate·z + (1−gate)·fresh)", fmt_array(&out_p));
    println!();

    let (best_u, gate_u) = consume_gate(&q_unpredictable, &c_prime, TAU, BETA);
    let out_u = consume(&q_unpredictable, &c_prime, TAU, BETA, fresh_think);
    println!("  unpredictable query q = {:?}", q_unpredictable);
    println!("    best slot i* = {best_u}  predictability p_{{i*}} = {:.6}", c_prime.slots[best_u].predictability);
    println!("    gate = {:.6}  (low → fall through to fresh)", gate_u);
    println!("    out  = {}  ≈ fresh_think output", fmt_array(&out_u));
    println!();
    println!("  Note the SMOOTH blend — never a hard argmax switch (AGENTS.md).");
    println!();

    // ── Step 4: AmortizationCostModel — the headline economic result ───────
    //
    // Paper §5.3: cost_total = sleep_cost + N · t · b_max · (1 − E[gate]).
    // Amortization factor < 1.0 means pre-computing wins. Paper reports ~2.5×
    // gain at N=10 (amortization_factor ≈ 0.4 with typical sleep budget).
    let model = AmortizationCostModel {
        t: 10.0,    // latency premium (paper default: wake-time compute is 10× pricier)
        b_max: 100, // wake-time compute budget per consumer (tokens)
        tau: TAU,
        beta: BETA,
    };

    // sleep_cost = sum of per-direction budgets. IdentityFunctorOp ignores
    // budget, but the cost model treats it as the price we paid at sleep-time.
    let sleep_cost: f32 = anticipator.budgets.iter().map(|&b| b as f32).sum();
    let e_gate = 0.5; // expected gate hit rate (would be measured in production)

    println!("── Step 4: AmortizationCostModel (paper §5.3) ──");
    println!("  t = {} (latency premium), b_max = {} (wake-time budget/consumer)", model.t, model.b_max);
    println!("  sleep_cost = Σ budgets[i] = {sleep_cost:.0}");
    println!("  E[gate] = {e_gate} (expected fraction of queries that hit the cache)");
    println!();
    println!("  {:<24} {:>14} {:>14} {:>14}", "scenario", "total_cost", "amort. factor", "pre-compute?");
    for n in [1u32, 5, 10, 50] {
        let total = model.total_cost(sleep_cost, n, e_gate);
        let factor = model.amortization_factor(sleep_cost, n, e_gate);
        let should = model.should_pre_compute(sleep_cost, n, e_gate);
        let should_str = if should { "YES" } else { "no" };
        println!(
            "  N={n:<20} {:>14.1} {:>14.4} {:>14}",
            total, factor, should_str
        );
    }
    println!();
    println!("  Reading: at N=1, pre-computing barely pays off (or doesn't). At N=10+");
    println!("  consumers amortizing the same c', the per-consumer wake cost drops");
    println!("  sharply. Paper's headline ~2.5× gain is the N=10 regime.");
    println!();

    println!("── Summary ──");
    println!("  • Sleep-time: anticipate(c) → c' artifact (one BLAKE3-committed output).");
    println!("  • Wake-time: consume(q, c') → smooth sigmoid-gated blend. Zero-alloc hot path.");
    println!("  • Amortization: one c' serves many consumers; factor < 1.0 = pre-compute wins.");
    println!();
    println!("See .plans/334_sleep_time_query_anticipator_primitive.md Phase 3 T3.1.");
    println!("See sleep_time_02_curiosity_inversion.rs for the curiosity↔predictability inversion.");
}

// ── Display helpers ────────────────────────────────────────────────────────

fn hex_short(b: &[u8; 32]) -> String {
    format!("{:02x}{:02x}{:02x}{:02x}", b[0], b[1], b[2], b[3])
}

fn fmt_array<const D: usize>(a: &[f32; D]) -> String {
    let parts: Vec<String> = a.iter().map(|x| format!("{:.3}", x)).collect();
    format!("[{}]", parts.join(", "))
}
