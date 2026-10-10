//! Issue 930 T6 (G2b) — SyncBank observe latency gate (informational
//! absolutes, the belief_dual_bench harness shape).
//!
//! Targets (Issue 930): D=8 sub-µs/observe, D=64 µs-class/observe at
//! `--release`. The observe path is `K·D(D+1)/2` packed FMAs + same-order ops
//! (D=8 → 108; D=64 → 6,240) — 100×+ headroom under the targets, so the
//! generous bar catches only gross regressions (an accidental alloc or an
//! O(D²)→O(D³) slip), not noise. Bar context recorded beside the number:
//! box state matters (AGENTS.md §Feature Flag Discipline G2) — this gate
//! asserts the ABSOLUTE ceiling only; quote measured numbers from a
//! preflighted quiet run, never from a loaded CI box.
//!
//! Plain `cargo test` (debug) cannot meet release-class bars — run with
//! `--release` (the T6 line in Issue 930). Debug builds IGNORE the tests
//! loudly (`#[cfg_attr(debug_assertions, ignore)]`, the riir-train parity-gate
//! posture) so the target never reports a silent green zero: in debug you see
//! `1 ignored`, never `ok. 0 passed`.

use katgpt_types::sync_bank::{SyncBank, SYNC_BANK_LADDER};
use std::time::Instant;

/// Box-state guard (AGENTS.md §Feature Flag Discipline G2 — a latency number
/// without its box state is not a measurement). Reads the 1-minute load
/// average and skips LOUD when the box is oversubscribed: a contaminated
/// number is worse than a deferral, and a green under load proves nothing.
/// The sibling-build rule of thumb: cores × 2.
fn skip_if_box_loaded(test_name: &str) -> bool {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let load = match std::fs::read_to_string("/proc/loadavg")
        .or_else(|_| {
            // macOS: no /proc — shell out to uptime once.
            let out = std::process::Command::new("sh")
                .arg("-c")
                .arg("uptime")
                .output();
            out.map(|o| {
                let text = String::from_utf8_lossy(&o.stdout).into_owned();
                // "load averages: 60.06 60.86 52.93" — first number
                text
                    .split("load averages:")
                    .nth(1)
                    .and_then(|rest| {
                        rest.split_whitespace().next().and_then(|f| f.parse::<f64>().ok())
                    })
                    .unwrap_or(0.0)
                    .to_string()
            })
            .map_err(|_| ())
        }) {
        Ok(text) => text
            .split_whitespace()
            .next()
            .and_then(|f| f.parse::<f64>().ok())
            .unwrap_or(0.0),
        Err(_) => 0.0,
    };
    if load > cores as f64 * 2.0 {
        println!(
            "SKIP {test_name}: box load {load:.2} > {} (cores×2) — a latency number under this load is a measurement of the scheduler, not the kernel. Re-run on a quiet box.",
            cores as f64 * 2.0
        );
        return true;
    }
    println!("box state: load {load:.2} / {cores} cores (quiet-enough window)");
    false
}

fn observe_ns_per<const D: usize, const K: usize>(iters: u32) -> f64 {
    // the canonical ladder is the K=3 bank set; derive it per-size so the
    // generic helper compiles for any K (all call sites use K=3)
    let ladder = [0.25f32, 0.0625, 0.015625];
    let ladder: [f32; K] = std::array::from_fn(|i| ladder[i % 3]);
    let mut bank = SyncBank::<D, K>::new(ladder);
    let mut z = [0.5f32; D];
    for (i, v) in z.iter_mut().enumerate() {
        *v = 0.3 + (i % 9) as f32 * 0.05;
    }
    // warmup
    for tick in 0..1_000u32 {
        z[0] = 0.3 + (tick % 13) as f32 * 0.01;
        bank.observe(&z);
    }
    let mut sink = 0.0f32;
    let start = Instant::now();
    for tick in 0..iters {
        z[0] = 0.3 + (tick % 13) as f32 * 0.01;
        bank.observe(&z);
        sink += bank.surprise_max();
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    elapsed / iters as f64 * 1e9
}

#[test]
#[cfg_attr(debug_assertions, ignore = "latency gate — run with --release (debug numbers are meaningless)")]
fn g2b_observe_latency_d8_sub_microsecond() {
    if skip_if_box_loaded("g2b_observe_latency_d8") {
        return;
    }
    let ns = observe_ns_per::<8, 3>(200_000);
    assert!(
        ns < 1_000.0,
        "D=8 observe {ns:.1} ns exceeds the sub-µs target"
    );
    println!("sync_bank observe D=8 K=3: {ns:.1} ns/observe (target < 1000 ns)");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "latency gate — run with --release (debug numbers are meaningless)")]
fn g2b_observe_latency_d64_microsecond_class() {
    if skip_if_box_loaded("g2b_observe_latency_d64") {
        return;
    }
    let ns = observe_ns_per::<64, 3>(50_000);
    assert!(
        ns < 5_000.0,
        "D=64 observe {ns:.1} ns exceeds the µs-class target (5 µs)"
    );
    println!("sync_bank observe D=64 K=3: {ns:.1} ns/observe (target < 5000 ns)");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "latency gate — run with --release (debug numbers are meaningless)")]
fn g2b_readout_latency_d8_cheap() {
    if skip_if_box_loaded("g2b_readout_latency_d8") {
        return;
    }
    let mut bank = SyncBank::<8, 3>::new(SYNC_BANK_LADDER);
    let z = [0.5f32; 8];
    for _ in 0..64 {
        bank.observe(&z);
    }
    let mut level = [0.0f32; 3];
    let mut argmax = [0.0f32; 3];
    let mut surprise = [0.0f32; 3];
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..200_000u32 {
        bank.relation_level_into(&mut level, &mut argmax);
        bank.relation_surprise_into(&mut surprise);
        sink += level[0] + surprise[2];
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    let ns = elapsed / 200_000.0 * 1e9;
    assert!(ns < 2_000.0, "D=8 readouts {ns:.1} ns exceed the 2 µs ceiling");
    println!("sync_bank readouts D=8 K=3: {ns:.1} ns/pair (ceiling 2000 ns)");
}
