//! Plan 631 Phase 2 T2.1 — the `harness_evolution` GOAT gate
//! (harness = false, the bench_039 pattern: asserted gates, exit-code
//! verdict, `cargo bench --bench harness_evolution_goat --features
//! harness_evolution` at the RELEASE posture).
//!
//! # What this bench IS
//!
//! An EXTRACTION gate (the primitive is the scenelab arena's shipped rule
//! verbatim): the load-bearing axes are
//!
//! - **G4 alloc** — the comparator + champion-scan path allocates ZERO
//!   across the measured loop (a counting `GlobalAlloc`, the bench_707
//!   pattern). These are pure f64/u8 functions; a single alloc would be a
//!   regression, not a tuning knob.
//! - **G1 stability** — total-order stability across 10⁶ shuffles of tied
//!   inputs: a full tie (same class, same fired fraction) scores EXACTLY
//!   `0.5` in BOTH argument orders, and an all-equal rating table yields
//!   the FIRST index under every shuffle (the determinism law's
//!   first-index rule — no processing-order or backing-map dependence).
//! - **G2 readout (informational)** — ns/call for the comparator, the
//!   third-tier form, and the scan. NO latency bar: this is an enabling
//!   primitive; the gain axis lives in the consumers' benches (sdk Plan
//!   009 T4 / riir-ai Plan 642 G2) and is referenced, never restated
//!   here (T2.2).
//!
//! Shuffles come from an owned LCG (no external RNG, identical stream on
//! every platform); results fold into a `black_box`-style sink so the
//! optimiser cannot delete the measured work (the loud-zero lesson).

use katgpt_core::harness_evolution::{
    VerdictRank, champion_index, two_tier_score, two_tier_score_with_ticks,
};
use std::time::Instant;

const SHUFFLES: u64 = 1_000_000;
const G4_CALLS: u64 = 200_000;

/// Tiny deterministic LCG (PCG-style output shift) — no external RNG dep,
/// identical stream for identical seeds on every platform.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn verdict(pass: bool) -> &'static str {
    if pass {
        "PASS"
    } else {
        "FAIL"
    }
}

fn main() {
    println!("── harness_evolution GOAT gate (Plan 631 T2.1) ──");
    let g4 = run_g4();
    let g1 = run_g1_stability();
    let g2 = run_g2_readout();
    let pass = g4 && g1 && g2;
    println!("════════════════════════════════════════");
    println!(
        "  G4 alloc {} · G1 stability {} · G2 readout {}",
        verdict(g4),
        verdict(g1),
        verdict(g2)
    );
    println!("  overall: {}", verdict(pass));
    if !pass {
        std::process::exit(1);
    }
}

// ─── G4 — zero alloc ───────────────────────────────────────────────────

fn run_g4() -> bool {
    use std::alloc::{GlobalAlloc, Layout};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingAllocator;
    static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            unsafe { std::alloc::System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
    }
    #[global_allocator]
    static A: CountingAllocator = CountingAllocator;

    println!("\n── G4 alloc ──");
    // Warm the path once (any lazily-initialized machinery pays here, not
    // in the measured window).
    let warm = two_tier_score(VerdictRank::PASS, VerdictRank::FAIL, 0.5, 0.25)
        + champion_index(&[1.0]).unwrap_or(0) as f64;

    let ratings = [1000.0_f64; 8];
    let mut lcg = Lcg::new(0x6316_3100);
    let mut sink = warm;
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..G4_CALLS {
        let a_rank = VerdictRank((lcg.next_u64() % 3) as u8);
        let b_rank = VerdictRank((lcg.next_u64() % 3) as u8);
        let fa = (lcg.next_u64() % 1000) as f64 / 1000.0;
        let fb = (lcg.next_u64() % 1000) as f64 / 1000.0;
        sink += two_tier_score(a_rank, b_rank, fa, fb);
        sink += two_tier_score_with_ticks(a_rank, b_rank, fa, fb, fa, fb);
        sink += champion_index(&ratings).unwrap_or(0) as f64;
    }
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    println!("  {allocs} allocations across {G4_CALLS} comparator+scan calls (sink {sink:e})");
    let pass = allocs == 0;
    println!("  G4 {}", verdict(pass));
    pass
}

// ─── G1 — total-order stability over 10⁶ shuffles ──────────────────────

fn run_g1_stability() -> bool {
    println!("\n── G1 stability (10⁶ shuffles of tied inputs) ──");
    let mut lcg = Lcg::new(0x6316_3101);

    // A fully-tied comparator pair must score exactly 0.5 in BOTH
    // argument orders — every draw, no state, no order dependence.
    let ranks = [
        (VerdictRank::PASS, VerdictRank::PASS),
        (VerdictRank::PARTIAL, VerdictRank::PARTIAL),
        (VerdictRank::FAIL, VerdictRank::FAIL),
    ];
    let fracs = [0.0_f64, 0.25, 0.5, 0.75, 1.0];
    let mut violations: u64 = 0;
    let mut sink = 0.0_f64;
    for s in 0..SHUFFLES {
        let (ra, rb) = ranks[(lcg.next_u64() % 3) as usize];
        let f = fracs[(lcg.next_u64() % 5) as usize];
        let ab = two_tier_score(ra, rb, f, f);
        let ba = two_tier_score(rb, ra, f, f);
        if ab != 0.5 || ba != 0.5 {
            violations += 1;
        }
        sink += ab + ba;
        // Every 8th draw doubles as a shuffle-stability probe on the
        // scan: an all-equal table yields the FIRST index no matter how
        // the (identical) values are permuted into the slots.
        if s & 7 == 0 {
            let mut table = [1000.0_f64; 12];
            for i in (1..table.len()).rev() {
                let j = (lcg.next_u64() % (i as u64 + 1)) as usize;
                table.swap(i, j);
            }
            if champion_index(&table) != Some(0) {
                violations += 1;
            }
        }
    }
    println!("  {SHUFFLES} shuffles, {violations} stability violations (sink {sink:e})");
    let pass = violations == 0;
    println!("  G1 {}", verdict(pass));
    pass
}

// ─── G2 — informational ns/call readout (no bar) ────────────────────────

fn run_g2_readout() -> bool {
    println!("\n── G2 readout (informational — no bar; consumer benches own the gain axis) ──");
    let a = VerdictRank::PASS;
    let b = VerdictRank::PARTIAL;
    let n = 1_000_000_u64;
    let mut sink = 0.0_f64;

    let t = Instant::now();
    for i in 0..n {
        let f = (i % 1000) as f64 / 1000.0;
        sink += two_tier_score(a, b, f, f * 0.5);
    }
    let cmp_ns = t.elapsed().as_nanos() as f64 / n as f64;

    let t = Instant::now();
    for i in 0..n {
        let f = (i % 1000) as f64 / 1000.0;
        sink += two_tier_score_with_ticks(a, b, f, f * 0.5, f, f * 0.5);
    }
    let ticks_ns = t.elapsed().as_nanos() as f64 / n as f64;

    let ratings = [1000.0_f64; 16];
    let t = Instant::now();
    for _ in 0..n {
        sink += champion_index(&ratings).unwrap_or(0) as f64;
    }
    let scan_ns = t.elapsed().as_nanos() as f64 / n as f64;

    println!(
        "  two_tier_score {cmp_ns:.2} ns/call · _with_ticks {ticks_ns:.2} ns/call · champion_index(16) {scan_ns:.2} ns/call"
    );
    let pass = sink.is_finite() && cmp_ns.is_finite() && scan_ns.is_finite();
    println!("  G2 {}", verdict(pass));
    pass
}
