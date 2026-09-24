//! Issue 882 P0 — differential anchor GOAT gate (G2 latency + G4 alloc).
//!
//! G2  latency: the one-axpy correction is O(d) against the O(C·d)
//!     scoring pass — corrected-vs-plain full pass ratio ≤ 1.02 at
//!     d=512/C=256 (`ab_median_ratio`, the paired interleave protocol —
//!     never two sequential arms; ±21.7% box drift measured on this
//!     workspace); the standalone axpy ≤ 1 µs at d=512 (`best_of_us`,
//!     black_box on result AND arguments — LTO deletes a dead-result
//!     timing loop).
//! G4  alloc-free: 0 steady-state allocations in the corrected scoring
//!     pass (caller scratch) and in the substrate's observe loop (the
//!     883 P0 calibration path — rows pre-allocated at construction).
//!
//! Harness = false, std::time::Instant (the bench_411/813 pattern), exit 1
//! on any gate failure. Run:
//!   cargo test -p katgpt-core --features differential_anchor \
//!     --test bench_886_differential_anchor_goat -- --nocapture --release
//!
//! Box state (the G2 law — recorded beside any published figure):
//! measured on the 4090 workstation (i7-13700K, Windows 11, AC power,
//! idle desktop) — see the bench record in .benchmarks/886.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::{ab_median_ratio, best_of_us};
use katgpt_core::differential_anchor::{correct_query_into, AnchorBuilder, AnchorSource};
use katgpt_core::fitted_anchor_table::StreamingMeanTable;

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

// ── fixtures ──────────────────────────────────────────────────────────────

const D: usize = 512;
const C: usize = 256;
const Q: usize = 64;

/// Deterministic pseudo-embedding (xorshift over the index — no RNG dep,
/// fixed across runs/platforms by construction).
fn row(seed: u32, d: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..d)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) & 0xffff) as f32 / 65535.0 - 0.5
        })
        .collect()
}

fn main() {
    let mut queries: Vec<Vec<f32>> = (0..Q).map(|i| row((i * 31 + 1) as u32, D)).collect();
    let queries_snapshot = queries.clone();
    let cands: Vec<Vec<f32>> = (0..C).map(|i| row((i * 7 + 1000) as u32, D)).collect();
    let qrefs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
    let crefs: Vec<&[f32]> = cands.iter().map(|c| c.as_slice()).collect();
    let anchor = AnchorBuilder::fit(AnchorSource::MeanQuery, &qrefs, &crefs);
    let lam = 0.55_f32;

    // ── G2a: the standalone axpy ≤ 1 µs at d=512 ──────────────────────
    // (`best_of_us` returns µs over a BATCH of 64 corrections — a single
    // 512-wide axpy is below timer resolution in release; result AND
    // arguments black_box'd so LTO cannot delete the work.)
    let mut scratch = vec![0.0f32; D];
    const BATCH: usize = 64;
    let axpy_us = best_of_us(20, 100, || {
        let t = std::time::Instant::now();
        for b in 0..BATCH {
            let q = &queries[b % Q];
            correct_query_into(q, &anchor, lam, &mut scratch);
            black_box(&mut scratch);
        }
        black_box(&queries[0]);
        black_box(&anchor);
        t.elapsed()
    }) / BATCH as f64;
    println!("G2a axpy @ d={D}: {axpy_us:.4} µs/call (bar 1.0 µs)");
    assert!(axpy_us <= 1.0, "G2a FAILED: axpy {axpy_us:.4} µs > 1.0 µs");

    // ── G2b: corrected vs plain full scoring pass ≤ 1.02× ─────────────
    // Arm A: plain pass (query · every candidate). Arm B: one correction
    // per query + every candidate. Both black_box'd; the ratio's arms are
    // the SAME work plus the O(d) axpy — the overhead being gated.
    let mut scores_a = vec![0.0f32; C];
    let mut scores_b = vec![0.0f32; C];
    let mut qh = vec![0.0f32; D];
    let ratio = ab_median_ratio(
        30,
        3,
        10,
        |_it| {
            for q in &queries_snapshot {
                for (c, d) in cands.iter().enumerate() {
                    scores_a[c] = black_box(q).iter().zip(black_box(d)).map(|(x, y)| x * y).sum();
                }
                black_box(&mut scores_a);
            }
        },
        |_it| {
            for q in &queries_snapshot {
                correct_query_into(q, &anchor, lam, &mut qh);
                for (c, d) in cands.iter().enumerate() {
                    scores_b[c] = black_box(&qh).iter().zip(black_box(d)).map(|(x, y)| x * y).sum();
                }
                black_box(&mut scores_b);
            }
        },
    );
    let r = ratio.median;
    println!(
        "G2b corrected/plain full pass: median {r:.4} (bar 1.02; min {:.4} max {:.4})",
        ratio.min(),
        ratio.max()
    );
    assert!(r <= 1.02, "G2b FAILED: corrected/plain {r:.4} > 1.02");

    // ── G4: 0 steady-state allocs ─────────────────────────────────────
    // (a) the corrected scoring pass with caller scratch;
    let before = allocs();
    for q in &queries {
        correct_query_into(q, &anchor, lam, &mut qh);
        for (c, d) in cands.iter().enumerate() {
            scores_b[c] = qh.iter().zip(d.iter()).map(|(x, y)| x * y).sum();
        }
        black_box(&scores_b);
    }
    let pass_allocs = allocs() - before;
    println!("G4a corrected scoring pass allocs: {pass_allocs}");
    assert_eq!(pass_allocs, 0, "G4a FAILED: steady-state allocs {pass_allocs}");

    // (b) the substrate's observe loop (the 883 calibration path) over a
    // pre-allocated table + a grand-only tail observation.
    let mut table = StreamingMeanTable::new(64, D);
    let obs = row(7, D);
    let before = allocs();
    for i in 0..64 {
        table.observe(i % 64, black_box(&obs));
    }
    table.observe_tail(black_box(&obs));
    let obs_allocs = allocs() - before;
    println!("G4b substrate observe loop allocs: {obs_allocs}");
    assert_eq!(obs_allocs, 0, "G4b FAILED: observe allocs {obs_allocs}");
    // The report path allocates by design (offline) — just exercise it.
    let rep = table.r_squared();
    assert!(!rep.empty);
    println!(
        "substrate r_squared: n={} aggregate={:.4} tracked_mass={:.3}",
        rep.n, rep.aggregate, rep.tracked_mass
    );

    // (c) in-place correction is alloc-free too.
    let before = allocs();
    for q in queries.iter_mut() {
        katgpt_core::differential_anchor::correct_query(q, &anchor, lam);
    }
    let inplace_allocs = allocs() - before;
    assert_eq!(inplace_allocs, 0, "G4c FAILED: in-place allocs {inplace_allocs}");

    println!("bench_886: ALL GATES PASSED (G2a axpy {axpy_us:.3}µs, G2b {r:.4}, G4 0 allocs)");
}
