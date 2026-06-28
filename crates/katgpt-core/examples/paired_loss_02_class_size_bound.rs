//! Plan 335 Phase 3 T3.3 — Proposition 1 class-size bound demonstration.
//!
//! Standalone demonstration of Proposition 1
//! (`DKL(p⋆_τ ‖ p_ϕ,τ) ≤ log|V_τ|`) — the information-theoretic justification
//! of our raw-vs-latent sync boundary (Research 319 §2.2). The bound answers a
//! single, sharp question:
//!
//! > For a token class with vocabulary size `|V_τ|`, how much loss can a
//! > richer feature map (latent encoding, recurrence, attention) *possibly*
//! > recover over the class-only predictor?
//!
//! The answer is `log|V_τ|` nats — the worst-case ceiling. This has two
//! regimes that map directly onto our domain classification (AGENTS.md):
//!
//! - **Small `|V_τ|` (physical domain):** the ceiling is near-zero. Raw
//!   commitment is information-theoretically sufficient — there's no room
//!   for latent encoding to help, so encoding position as an embedding then
//!   decoding back for sync would be pure lossy round-trip overhead.
//! - **Large `|V_τ|` (semantic domain):** the ceiling is loose. Latent
//!   encoding earns its keep — there's genuine room for a richer feature
//!   to capture structure the raw scalar can't express.
//!
//! # What this example shows
//!
//! 1. The bound for a spectrum of class sizes: boolean (2) → u8 (256) →
//!    grid coord (65_536) → open-class noun (50_000) → full BPE vocab
//!    (50_257) → Unicode code point (1_112_064).
//! 2. The raw-vs-latent decision for each: which classes should stay raw
//!    (synced, committed, replayed) vs which should operate in latent space.
//! 3. A worked annotation: build a tiny [`PairedLossGap`] over a few classes,
//!    annotate with bounds, and show how `gap_to_bound_ratio` reads.
//!
//! # Why this matters (the cross-repo contract)
//!
//! Per AGENTS.md "Latent vs Raw Space Rules":
//! - `MapPos { x, y }` → raw, synced. Physical domain; small `V_τ`; raw is
//!   sufficient by Proposition 1.
//! - `HlaCacheProxy` → latent, local. Semantic domain; large `V_τ`; latent
//!   earns its keep. The scalar outputs (5 affect values) cross the sync
//!   boundary via a bridge function.
//! - `NeuronShard { style_weights, hla_moments }` → latent, committed to
//!   Cold tier as-is. Already a fixed-size Pod; BLAKE3-committed.
//!
//! This example is the theoretical-validation artifact for that contract.
//!
//! Run with: `cargo run --example paired_loss_02_class_size_bound --features paired_loss_diagnostic`

use std::collections::HashMap;

use katgpt_core::{
    ClassSizeBound, PairedLossGap, TokenClass,
};

// ── The bound table ─────────────────────────────────────────────────────────

/// One row of the Proposition 1 bound table.
struct BoundRow {
    name: &'static str,
    v_tau: usize,
    /// Which AGENTS.md domain this class lives in.
    domain: &'static str,
    /// Whether raw commitment is sufficient, or latent earns its keep.
    recommendation: &'static str,
}

/// The illustrative class spectrum. Ordered from smallest to largest `V_τ`.
const BOUND_TABLE: &[BoundRow] = &[
    BoundRow {
        name: "boolean",
        v_tau: 2,
        domain: "physical",
        recommendation: "raw sufficient — bit-exact sync, no latent benefit",
    },
    BoundRow {
        name: "u4 / small enum",
        v_tau: 16,
        domain: "physical",
        recommendation: "raw sufficient — small state space",
    },
    BoundRow {
        name: "u8",
        v_tau: 256,
        domain: "physical",
        recommendation: "raw sufficient — 5.5 nats ceiling, still small",
    },
    BoundRow {
        name: "u16 grid coord",
        v_tau: 65_536,
        domain: "physical (large)",
        recommendation: "raw still wins — deterministic replay needs exact bits",
    },
    BoundRow {
        name: "open-class noun",
        v_tau: 50_000,
        domain: "semantic",
        recommendation: "latent earns its keep — 10.8 nats of room",
    },
    BoundRow {
        name: "full BPE vocab",
        v_tau: 50_257,
        domain: "semantic",
        recommendation: "latent earns its keep — token-embedding territory",
    },
    BoundRow {
        name: "Unicode code point",
        v_tau: 1_112_064,
        domain: "semantic (huge)",
        recommendation: "latent essential — 13.9 nats, embedding mandatory",
    },
];

fn print_bound_table() {
    println!("── Proposition 1 bound across the class-size spectrum ──");
    println!(
        "  DKL(p⋆_τ ‖ p_ϕ,τ) ≤ log|V_τ|  — the worst-case reducible loss (nats)"
    );
    println!();
    println!(
        "  {:<22} {:>12} {:>12}  {:<18} {}",
        "class", "|V_τ|", "log|V_τ|", "domain", "recommendation"
    );
    for row in BOUND_TABLE {
        let bound = ClassSizeBound::for_vocab_size(row.v_tau);
        let ceiling = bound.reducible_loss_ceiling();
        println!(
            "  {:<22} {:>12} {:>12.4}  {:<18} {}",
            row.name, row.v_tau, ceiling, row.domain, row.recommendation
        );
    }
    println!();
}

// ── The raw-vs-latent decision table ────────────────────────────────────────

/// Map the bound onto the AGENTS.md sync-boundary rule. This is the
/// operational consequence of Proposition 1: which classes cross the
/// `SyncBlock → ChainConsensus` quorum boundary as raw values, and which
/// stay latent-local with only scalar projections crossing.
fn print_raw_vs_latent_decision() {
    println!("── Raw-vs-latent sync-boundary decision (AGENTS.md) ──");
    println!("  Rule: small V_τ → raw + synced; large V_τ → latent + local;");
    println!("        bridge functions cross the boundary (scalar projections).");
    println!();

    // Physical domain examples (raw, synced).
    let physical: &[(&str, usize, &str)] = &[
        ("MapPos { x, y }", 65_536, "raw, synced via SyncBlock, replayed bit-exact"),
        ("HP / wallet balance", 1_000_000, "raw, synced — quorum needs exact scalar"),
        ("ForceVector { fx, fy }", 65_536, "raw physics — feeds latent via bridge"),
        ("boolean gate", 2, "raw, 1 bit — latent is pure overhead"),
    ];
    println!("  Physical domain (raw + synced):");
    for (name, v, note) in physical {
        let bound = ClassSizeBound::for_vocab_size(*v);
        println!(
            "    {:<24} V_τ={:<10} ceiling={:.4} nats  → {}",
            name,
            v,
            bound.reducible_loss_ceiling(),
            note
        );
    }
    println!();

    // Semantic domain examples (latent, local).
    let semantic: &[(&str, usize, &str)] = &[
        ("HlaCacheProxy", 50_000, "latent, local — 5 affect scalars bridge to sync"),
        ("NeuronShard weights", 50_257, "latent, Cold-tier committed as Pod (BLAKE3)"),
        ("emotion projection", 50_000, "latent — dot-product + sigmoid, not synced"),
        ("zone embedding", 50_000, "latent — drives zone attention, not synced"),
    ];
    println!("  Semantic domain (latent + local):");
    for (name, v, note) in semantic {
        let bound = ClassSizeBound::for_vocab_size(*v);
        println!(
            "    {:<24} V_τ={:<10} ceiling={:.4} nats  → {}",
            name,
            v,
            bound.reducible_loss_ceiling(),
            note
        );
    }
    println!();

    println!("  Bridge pattern (raw → latent, latent → raw):");
    println!("    raw → latent : dot-product projection onto direction vectors,");
    println!("                   bounded by sigmoid (never softmax, per AGENTS.md).");
    println!("    latent → raw : clamp to valid range. Never reconstruct position");
    println!("                   from embedding — emit raw alongside latent.");
    println!("    Bridges are zero-allocation, gateable, sync-independent.");
    println!();
}

// ── Worked annotation example ───────────────────────────────────────────────

/// A tiny worked example: two log-prob traces over 6 tokens, three classes,
/// annotated with Proposition 1 bounds. Shows how `gap_to_bound_ratio` reads
/// on a controlled fixture.
fn print_worked_annotation() {
    println!("── Worked annotation: tiny A/B with Proposition 1 bounds ──");
    println!("  6 tokens: 2 Content (large V_τ), 2 Function (small V_τ),");
    println!("  2 Other (no bound — demonstrates NaN handling).");
    println!();

    // Δ = [+0.8, +0.9, +0.05, +0.04, 0.0, 0.0]
    // Content (large V_τ=50000): mean Δ = 0.85 → ratio = 0.85/10.82 ≈ 0.079
    // Function (small V_τ=8): mean Δ = 0.045 → ratio = 0.045/2.08 ≈ 0.022
    // Other (no bound): mean Δ = 0.0 → ratio = NaN
    let a = [-1.0f32, -1.1, -2.0, -2.05, -3.0, -3.0];
    let b = [-1.8f32, -2.0, -2.05, -2.09, -3.0, -3.0]; // B better on Content/Function
    let gap = PairedLossGap::from_log_probs(&a, &b);
    let classes = [
        TokenClass::Content,
        TokenClass::Content,
        TokenClass::Function,
        TokenClass::Function,
        TokenClass::Other,
        TokenClass::Other,
    ];

    let mut bounds = HashMap::new();
    bounds.insert(TokenClass::Content, ClassSizeBound::for_vocab_size(50_000));
    bounds.insert(TokenClass::Function, ClassSizeBound::for_vocab_size(8));
    // Other: deliberately omitted.

    let report = gap.annotate_with_class_bounds(&classes, &bounds);

    println!(
        "  {:<12} {:>8} {:>12} {:>10} {:>14}  {}",
        "class", "count", "mean Δ", "log|V_τ|", "ratio", "interpretation"
    );
    for row in &report.rows {
        let lv_str = if row.log_v_tau.is_nan() {
            "NaN".to_string()
        } else {
            format!("{:.4}", row.log_v_tau)
        };
        let ratio_str = if row.gap_to_bound_ratio.is_nan() {
            "NaN".to_string()
        } else {
            format!("{:+.6}", row.gap_to_bound_ratio)
        };
        let interp = interpret_ratio(row.gap_to_bound_ratio);
        println!(
            "  {:<12} {:>8} {:>+12.6} {:>10} {:>14}  {}",
            row.class.label(),
            row.count,
            row.mean_gap,
            lv_str,
            ratio_str,
            interp
        );
    }
    println!();
    println!("  Reading: Content captures ~8% of its (huge) ceiling — lots of");
    println!("  room remains. Function captures ~2% of its (small) ceiling — but");
    println!("  the absolute gap is also tiny, so the bound is nearly saturated");
    println!("  in practical terms. Other: NaN — consumer must supply V_τ.");
    println!();
}

/// Human-readable interpretation of a `gap_to_bound_ratio`.
fn interpret_ratio(r: f32) -> &'static str {
    if r.is_nan() {
        "no bound supplied"
    } else if r < 0.0 {
        "A/B backwards (B worse than A)"
    } else if r > 1.0 {
        "exceeds bound — V_τ too small?"
    } else if r > 0.75 {
        "near ceiling — little room left"
    } else if r > 0.25 {
        "mid-range — partial capture"
    } else {
        "far from ceiling — room to grow"
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 335 Phase 3 T3.3 — Proposition 1 Class-Size Bound          ║");
    println!("║  Paper §5: DKL(p⋆_τ ‖ p_ϕ,τ) ≤ log|V_τ|                          ║");
    println!("║  Research 319 §2.2: raw-vs-latent theoretical validation         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Proposition 1 bounds the reducible loss from ANY richer feature map");
    println!("ϕ (latent encoding, recurrence, attention) by log|V_τ| — the natural");
    println!("log of the class vocabulary size. Small V_τ → raw is sufficient;");
    println!("large V_τ → latent earns its keep.");
    println!();
    println!("IMPORTANT: this is a *bound*, not an equality (Research 319 §5 R4).");
    println!("The actual reducible loss can be much smaller. Don't overclaim raw");
    println!("is *optimal* — only that the *room for latent to help* is bounded.");
    println!();

    print_bound_table();
    print_raw_vs_latent_decision();
    print_worked_annotation();

    println!("── Summary ──");
    println!("  • Physical domain (small V_τ): raw commitment is information-");
    println!("    theoretically sufficient. Encoding as embedding then decoding");
    println!("    for sync is pure lossy overhead + anti-cheat breakage.");
    println!("  • Semantic domain (large V_τ): latent encoding earns its keep.");
    println!("    Sync the scalar projections (bridge outputs), not the embedding.");
    println!("  • The boundary is the sync layer: SyncBlock → ChainConsensus quorum");
    println!("    commit → Cold tier = raw, deterministic, bit-identical replay.");
    println!("    Everything else = latent, local, bridge-projected.");
    println!();
    println!("See .research/319_Paired_Token_Loss_Gap_Discourse_State_Diagnostic.md");
    println!("§2.2 for the full raw-vs-latent justification mapping.");
    println!("See AGENTS.md 'Latent vs Raw Space Rules' for the operational contract.");
}
