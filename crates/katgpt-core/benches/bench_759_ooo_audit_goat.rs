//! Issue 759 T4 — Direction-Bank Audit GOAT Gate (`direction_bank_audit`).
//!
//! Exercises the GOAT gate for the ooo_audit primitive (Research 552,
//! Issa/Liu/Ballé/Klindt bioRxiv 2026.09.05.748439): asymmetric
//! Odd-One-Out interpretability + Cross-OOO diversity + greedy curation.
//!
//! # Gates
//!
//! - **G1** — Correctness: the planted-bank fixture (4 true clusters + 2
//!   duplicates + 2 noise units, shared with the unit tests) curates to
//!   exactly the 4 planted features; duplicates pruned, noise dropped,
//!   true-unit OOO >= the paper's 0.8 bar.
//! - **G2** — Latency: `audit_bank_into` + `greedy_curate` at the reference
//!   bank scale U=64 units × N=1024 exemplars, K=8 (median over batched
//!   runs). Budget: < 10 ms per full audit (offline freeze-time gate —
//!   orders of magnitude under any consolidation window).
//! - **G3** — Feature isolation: verified outside this binary via
//!   `cargo check` at default / `--features direction_bank_audit`.
//! - **G4** — Zero alloc: 0 allocations across 100 steady-state
//!   audit+curate cycles (CountingAllocator, per-thread, with the
//!   Issue 714 live-counter canary).

use katgpt_core::ooo_audit::{
    BankAudit, CurationResult, OooAuditConfig, audit_bank_into, exemplar_rbf_sim_into,
    fixtures::planted_cluster_bank, greedy_curate, AuditScratch,
};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

fn gate_g1_correctness() -> GateResult {
    println!("\n--- G1: planted-bank curation recovery ---");

    let (x, a) = planted_cluster_bank(4, 20, 8, 2, 2, 7);
    let n = 4 * 20;
    let u = 4 + 2 + 2;
    let mut sim = vec![0.0f32; n * n];
    exemplar_rbf_sim_into(&x, 8, 1.0, &mut sim);

    let cfg = OooAuditConfig::default();
    let mut scratch = AuditScratch::new();
    let mut audit = BankAudit::new(u, n, cfg.top_k);
    audit_bank_into(&a, &sim, &cfg, &mut scratch, &mut audit);
    let mut result = CurationResult::new(u);
    greedy_curate(&audit, &cfg, &mut scratch, &mut result);

    let count = result.unique_feature_count();
    let true_units_ok = {
        let mut kept = result.kept.clone();
        kept.sort_unstable();
        kept == vec![0, 1, 2, 3]
    };
    let min_true_ooo = (0..4).map(|i| audit.ooo[i]).fold(f32::INFINITY, f32::min);

    println!("  unique features:  {count} (planted: 4)");
    println!("  true units kept:  {true_units_ok}");
    println!("  min true-unit OOO: {min_true_ooo:.4} (bar 0.8)");
    println!("  pruned redundant:  {:?} (planted dups: [4, 5])", result.pruned_redundant);
    println!("  dropped noise:     {:?} (planted noise: [6, 7])", result.dropped_uninterpretable);

    let mut pruned_sorted = result.pruned_redundant.clone();
    pruned_sorted.sort_unstable();
    let mut dropped_sorted = result.dropped_uninterpretable.clone();
    dropped_sorted.sort_unstable();

    let passed = count == 4
        && true_units_ok
        && min_true_ooo >= 0.8
        && pruned_sorted == vec![4, 5]
        && dropped_sorted == vec![6, 7];
    GateResult {
        name: "G1 correctness",
        passed,
        detail: format!(
            "count={count} min_true_ooo={min_true_ooo:.3} pruned={} dropped={}",
            pruned_sorted.len(),
            dropped_sorted.len()
        ),
    }
}

fn gate_g2_latency() -> GateResult {
    println!("\n--- G2: audit latency (U=64, N=1024, K=8) ---");

    // Reference-scale bank: 16 clusters × 64 exemplars × 32 dims; 24
    // duplicates + 24 noise units fill the bank to U=64.
    let (x, a) = planted_cluster_bank(16, 64, 32, 24, 24, 11);
    let n = 16 * 64;
    let u = 16 + 24 + 24;
    let mut sim = vec![0.0f32; n * n];
    exemplar_rbf_sim_into(&x, 32, 1.0, &mut sim);

    let cfg = OooAuditConfig::default();
    let mut scratch = AuditScratch::new();
    let mut audit = BankAudit::new(u, n, cfg.top_k);
    let mut result = CurationResult::new(u);

    // Warmup (also grows the MEI inner vectors — excluded from timing).
    audit_bank_into(&a, &sim, &cfg, &mut scratch, &mut audit);
    greedy_curate(&audit, &cfg, &mut scratch, &mut result);

    const ITERS: u32 = 50;
    let mut audit_us = Vec::with_capacity(ITERS as usize);
    let mut curate_us = Vec::with_capacity(ITERS as usize);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        audit_bank_into(black_box(&a), black_box(&sim), black_box(&cfg), &mut scratch, &mut audit);
        audit_us.push(t0.elapsed().as_micros() as u64);
        let t1 = Instant::now();
        greedy_curate(black_box(&audit), black_box(&cfg), &mut scratch, &mut result);
        curate_us.push(t1.elapsed().as_micros() as u64);
    }
    audit_us.sort_unstable();
    curate_us.sort_unstable();
    let audit_med = audit_us[audit_us.len() / 2];
    let curate_med = curate_us[curate_us.len() / 2];
    let total = audit_med + curate_med;

    println!("  audit_bank_into median:  {audit_med} µs");
    println!("  greedy_curate median:    {curate_med} µs");
    println!("  full audit cycle:        {total} µs (budget 10,000 µs)");

    GateResult {
        name: "G2 latency",
        passed: total < 10_000,
        detail: format!("audit={audit_med}µs curate={curate_med}µs total={total}µs < 10ms"),
    }
}

fn gate_g4_zero_alloc() -> GateResult {
    println!("\n--- G4: zero allocations, steady-state (100 audit+curate cycles) ---");
    assert_counter_is_live();

    let (x, a) = planted_cluster_bank(4, 20, 8, 2, 2, 7);
    let n = 4 * 20;
    let u = 4 + 2 + 2;
    let mut sim = vec![0.0f32; n * n];
    exemplar_rbf_sim_into(&x, 8, 1.0, &mut sim);

    let cfg = OooAuditConfig::default();
    let mut scratch = AuditScratch::new();
    let mut audit = BankAudit::new(u, n, cfg.top_k);
    let mut result = CurationResult::new(u);

    // Warm once: first call grows inner MEI vectors (documented allocation).
    audit_bank_into(&a, &sim, &cfg, &mut scratch, &mut audit);
    greedy_curate(&audit, &cfg, &mut scratch, &mut result);

    let ((), allocs) = alloc_delta(|| {
        for _ in 0..100 {
            audit_bank_into(&a, &sim, &cfg, &mut scratch, &mut audit);
            greedy_curate(&audit, &cfg, &mut scratch, &mut result);
        }
    });

    println!("  allocations in 100 steady-state cycles: {allocs}");
    GateResult {
        name: "G4 zero-alloc",
        passed: allocs == 0,
        detail: format!("{allocs} allocs / 100 cycles"),
    }
}

fn main() {
    println!("Issue 759 T4 — direction_bank_audit GOAT gate");
    println!("primitive: asymmetric OOO + Cross-OOO + greedy curation");

    let gates = vec![
        gate_g1_correctness(),
        gate_g2_latency(),
        gate_g4_zero_alloc(),
    ];

    println!("\n================ GOAT VERDICT ================");
    let mut all = true;
    for g in &gates {
        println!("  [{}] {} — {}", if g.passed { "PASS" } else { "FAIL" }, g.name, g.detail);
        all &= g.passed;
    }
    if all {
        println!("  → G1 + G2 + G4 ALL PASS (G3 feature isolation: cargo check).");
        println!("  → PRIMITIVE GOAT holds; stays OPT-IN (no-default-consumer rule).");
        println!("  → Promotion needs a production consumer win (riir-ai freeze gate /");
        println!("    riir-clippy corpus axis, Research 552 F1/F2).");
    } else {
        println!("  → GOAT FAIL — keep opt-in, record the failing gate in Bench 758.");
    }
    std::process::exit(if all { 0 } else { 1 });
}
