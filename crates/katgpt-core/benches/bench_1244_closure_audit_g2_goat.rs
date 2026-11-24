//! Issue 1244 — the G2 budget bench for `closure_audit` (harness = false;
//! `required-features = ["closure_audit"]`).
//!
//! **The G2 floor bar:** the ridge sweep (BOTH sides, full λ grid) at
//! T=1e4, D=64 must stay ≤ 50 ms in a RELEASE build (median of 3 runs).
//! This is a FLOOR bar on an offline instrument — never a perf claim;
//! the primitive claims NO GOAT (ungoat discipline).
//!
//! **The kNN disclosed budget:** the full distance pass is priced
//! explicitly (the issue's own text expects ≈0.3–2 s single-threaded at
//! T=1e4); the capped posture (ref cap 2048) is quoted beside it. No bar
//! — a disclosed line.
//!
//! Runbook: a QUIET rig only (never a loaded box); quote
//! `../riir-reflex/scripts/bench_preflight.sh`'s `PROVENANCE:` line
//! beside every number quoted from this bench.

// The counting_allocator!() macro defines ALLOC_COUNT / alloc_delta /
// assert_counter_is_live at THIS crate root on invocation (textual scope
// after the `mod common;` declaration — the house convention).
#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::closure_audit::{
    ClosureAuditConfig, FixtureKind, FixtureSpec, build_fixture, run,
};
use std::time::Instant;

/// The T=1e4 / D=64 budget corpus — the identity world with a pure-noise
/// microstate (deterministic per seed; timing-representative: the ridge
/// sweep and kNN passes are content-blind at these dims).
fn budget_corpus() -> (Vec<f32>, Vec<f32>) {
    let (traj, micro) = build_fixture(&FixtureSpec {
        kind: FixtureKind::Identity,
        steps: 10_000,
        dim: 64,
        micro_dim: 64,
        seed: 0x1244_0001,
    });
    (
        traj.iter().flatten().copied().collect(),
        micro.iter().flatten().copied().collect(),
    )
}

fn timed_run(config: &ClosureAuditConfig, traj: &[f32], micro: &[f32]) -> (f64, f64) {
    let t = Instant::now();
    let report = run(&[traj], 64, Some((&[micro], 64)), config).expect("budget run");
    let wall = t.elapsed().as_secs_f64() * 1e3;
    assert!(
        wall >= report.budget.total_ms * 0.5,
        "run budget self-report sanity"
    );
    (
        wall,
        report.budget.ridge_summary_ms + report.budget.ridge_augmented_ms,
    )
}

fn main() {
    println!("== Issue 1244 closure_audit G2 budget bench ==");
    println!("   posture: release required; quiet rig required; quote bench_preflight PROVENANCE");
    let (flat, mflat) = budget_corpus();
    println!(
        "   corpus: identity world T={} D=64 micro=64 (seed 0x1244_0001)",
        flat.len() / 64
    );

    // ---- G2: the ridge-sweep floor bar (median of 3) ----
    // The ridge-only posture: an empty kNN grid runs the ridge arms alone.
    let config = ClosureAuditConfig {
        knn_ks: Vec::new(),
        ..ClosureAuditConfig::default()
    };
    let mut ridge_walls = Vec::with_capacity(3);
    for i in 0..3 {
        let (wall, sweep) = timed_run(&config, &flat, &mflat);
        ridge_walls.push(sweep);
        println!("   ridge run {i}: sweep={sweep:.3}ms total-run={wall:.1}ms");
    }
    ridge_walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ridge_median = ridge_walls[1];
    println!("   G2 ridge sweep (both sides, median of 3): {ridge_median:.3} ms  [bar ≤ 50.0 ms]");
    let g2_pass = ridge_median <= 50.0;

    // ---- the kNN disclosed budget: full pass vs capped ----
    let full = ClosureAuditConfig::default();
    let (full_wall, _) = timed_run(&full, &flat, &mflat);
    println!("   kNN FULL pass (cap none): total-run={full_wall:.1} ms (disclosed, no bar)");
    let capped = ClosureAuditConfig {
        knn_ref_cap: Some(2048),
        ..ClosureAuditConfig::default()
    };
    let (capped_wall, _) = timed_run(&capped, &flat, &mflat);
    println!(
        "   kNN CAPPED pass (ref cap 2048): total-run={capped_wall:.1} ms (disclosed, no bar)"
    );

    // ---- the alloc census (the offline instrument's disclosed budget) ----
    let (_, allocs) =
        alloc_delta(|| run(&[&flat], 64, Some((&[&mflat], 64)), &full).expect("alloc census run"));
    println!("   alloc census (full run): {allocs} allocations (disclosed offline budget)");

    // Counter liveness (Issue 714): prove the census above measured.
    assert_counter_is_live();

    if g2_pass {
        println!("   G2 VERDICT: PASS (ridge sweep {ridge_median:.3} ms ≤ 50.0 ms)");
    } else {
        println!(
            "   G2 VERDICT: FAIL (ridge sweep {ridge_median:.3} ms > 50.0 ms) — investigate before landing"
        );
        std::process::exit(1);
    }
}
