//! Issue 887 — ladder_gate GOAT gate (G2 latency + G4 alloc).
//!
//! G2  latency: `on_eval` is O(1) — the streak arithmetic only — and
//!     `on_eval_with_retention` is O(k) over the probe slice: the retained
//!     path at k=64 must stay µs-scale against the probe-free path (the
//!     paper's re-probing is the CALLER's forward cost; the gate adds only
//!     the scan). Ratios use `ab_median_ratio` (the paired interleave
//!     protocol — never two sequential arms; ±21.7% box drift measured on
//!     this workspace); absolutes use `best_of_us` with `black_box` on
//!     result AND arguments (LTO deletes a dead-result timing loop).
//! G4  alloc-free: 0 allocations across the driven-ladder steady state
//!     (both paths, both actions) — the FSM is arithmetic over
//!     caller-owned data.
//!
//! Harness = false, std::time::Instant (the bench_411/813 pattern), exit 1
//! on any gate failure. Run:
//!   cargo test -p katgpt-core --features ladder_gate \
//!     --test bench_887_ladder_gate_goat -- --nocapture --release
//!
//! Box state (the G2 law — recorded beside any published figure): M3 Max,
//! macOS 26.6.2, AC power, 2026-09-25 — see .benchmarks/891_ladder_gate_goat.md.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::{ab_median_ratio, best_of_us};
use katgpt_core::ladder_gate::{LadderAction, LadderGate, LadderGateConfig};

// ── G4: counting allocator ────────────────────────────────────────────────

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

// ── G2: latency ───────────────────────────────────────────────────────────

fn g2_probe_free_path_is_ns_scale_o1() {
    let a_hi = 0.95_f32;
    let a_lo = 0.5_f32;
    // best_of_us yields µs for the WHOLE 1000-eval driven ladder (fresh
    // gate per rep: the FSM's per-eval cost is what must be µs-irrelevant,
    // and a dead-result loop would be folded — hence black_box everywhere).
    let us = best_of_us(20, 100, || {
        let t = std::time::Instant::now();
        let mut gate = LadderGate::new(LadderGateConfig::default());
        for i in 0..1_000_u32 {
            let a = if i % 2 == 0 { a_hi } else { a_lo };
            black_box(gate.on_eval(black_box(a)));
        }
        black_box(gate.stage());
        t.elapsed()
    });
    let per_eval_ns = us * 1000.0 / 1000.0;
    println!("on_eval: {per_eval_ns:.1} ns/eval (1000-eval driven ladder)");
    assert!(
        us < 1_000.0,
        "G2: 1000 on_eval calls took {us} µs (bar 1 ms per ladder)"
    );
}

fn g2_retention_path_at_k8_stays_us_scale() {
    let probes8 = [0.95_f32; 8];
    let us = best_of_us(20, 100, || {
        let t = std::time::Instant::now();
        let mut gate = LadderGate::new(LadderGateConfig::default());
        let mut advances = 0_u32;
        for i in 0..1_000_u32 {
            // alternate probe shapes so neither arm folds (the harness's
            // own law: constant input is what lets the optimiser hoist).
            let probes: &[f32] = if i % 2 == 0 { &probes8 } else { &[] };
            if matches!(
                black_box(gate.on_eval_with_retention(black_box(0.95), black_box(probes))),
                LadderAction::Advance
            ) {
                advances += 1;
            }
        }
        black_box(advances);
        black_box(gate.stage());
        t.elapsed()
    });
    let per_eval_ns = us * 1000.0 / 1000.0;
    println!("on_eval_with_retention k≤8: {per_eval_ns:.1} ns/eval");
    assert!(
        us < 1_000.0,
        "G2: 1000 retention-path calls took {us} µs (bar 1 ms per ladder)"
    );
}

fn g2_retention_scan_grows_linearly_not_worse() {
    // O(k) sanity: the k=64 retained scan must not blow up against the
    // k=8 scan — a super-linear ratio would mean accidental quadratic
    // behavior. Paired interleave (the box-drift law), identical ladder
    // shape on both arms so the ratio isolates the scan.
    let probes8 = [0.95_f32; 8];
    let probes64 = [0.95_f32; 64];
    let ratio = ab_median_ratio(
        30,
        3,
        10,
        |it| {
            let mut gate = LadderGate::new(LadderGateConfig::default());
            for i in 0..1_000_u32 {
                let p: &[f32] = if (i as usize + it).is_multiple_of(2) {
                    &probes8
                } else {
                    &[]
                };
                black_box(gate.on_eval_with_retention(black_box(0.95), black_box(p)));
            }
            black_box(gate.stage());
        },
        |it| {
            let mut gate = LadderGate::new(LadderGateConfig::default());
            for i in 0..1_000_u32 {
                let p: &[f32] = if (i as usize + it).is_multiple_of(2) {
                    &probes64
                } else {
                    &[]
                };
                black_box(gate.on_eval_with_retention(black_box(0.95), black_box(p)));
            }
            black_box(gate.stage());
        },
    );
    ratio.report("retention k64 vs k8");
    println!(
        "   k64/k8 median ratio {:.3} (linear scan ≈ 8× of a tiny base; blowup bar 40×)",
        ratio.median
    );
    assert!(
        ratio.median < 40.0,
        "G2: k64 scan is {:.3}× k8 — super-linear blowup",
        ratio.median
    );
}

// ── G4: alloc-free ────────────────────────────────────────────────────────

fn g4_steady_state_is_alloc_free() {
    let mut g = LadderGate::new(LadderGateConfig::default());
    let probes8 = [0.95_f32; 8];
    // warm-up (any one-time laziness lands here)
    for i in 0..1_000_u32 {
        let probes: &[f32] = if i % 2 == 0 { &probes8 } else { &[] };
        black_box(g.on_eval_with_retention(0.95, probes));
    }
    let before = allocs();
    let mut actions = 0_u32;
    for i in 0..10_000_u32 {
        let probes: &[f32] = if i % 3 == 0 { &probes8 } else { &[] };
        if !matches!(
            black_box(g.on_eval_with_retention(black_box(0.95), black_box(probes))),
            LadderAction::Hold
        ) {
            actions += 1;
        }
    }
    let delta = allocs() - before;
    assert_eq!(
        delta, 0,
        "G4: {actions} advances/retreats allocated {delta}"
    );
    println!("G4: 0 allocations over 10k steady-state evals ({actions} advances/retreats)");
    black_box(g.stage());
}

fn main() {
    println!("bench_887: ladder_gate GOAT gate — G2 latency + G4 alloc-free");
    g2_probe_free_path_is_ns_scale_o1();
    g2_retention_path_at_k8_stays_us_scale();
    g2_retention_scan_grows_linearly_not_worse();
    g4_steady_state_is_alloc_free();
    println!("bench_887: ALL GATES PASSED (G2 O(1) + O(k) µs-scale, G4 0 allocs)");
}
