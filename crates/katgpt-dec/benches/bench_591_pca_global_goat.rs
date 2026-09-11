//! Plan 591 Phase 3 — PCA global-function layer GOAT gate (G1–G4), Bench 707.
//!
//! Arena: 64×64 `grid_2d` tile map, four seed cells in a diamond (Manhattan
//! spacing 8) around the map center — the content-generation shape (a starting
//! structure to unify). Competitors, all on the SAME untouched Plan 454 kernel
//! and parameters:
//!
//! - **(a) pure-local** [`stochastic_birth_death_step`] — no decision layer.
//!   A pure-local rule has no global read, so it can neither verify nor
//!   terminate at earlier connectivity; its only connectivity-guaranteed
//!   stopping point is its fixpoint (every vertex alive). Halts at fill.
//! - **(b) sync** [`step_pca_sync`] + [`GlobalTargetGate`] (Betti0 ≤ 1).
//! - **(c) async** [`step_pca_async`] + same gate (Betti0 is non-incremental,
//!   so async degenerates to sync semantics — pinned by a unit test; both arms
//!   run anyway so the bench cannot silently diverge from that pin).
//! - **(d) static-generator incumbent** — ONE fixed connected pattern placed
//!   once, zero iterations: the "do nothing dynamic" baseline the paper's
//!   98→19 Sokoban number competes with. Connectivity by construction
//!   (BFS-verified anyway); identical artifact for every seed.
//!
//! # Gates
//!
//! - **G1 correctness** — every arm's final state verified by the INDEPENDENT
//!   BFS reference (Plan 591 Phase 0 finding: `betti_numbers(cx)` is
//!   state-blind and cannot serve; the pinned BFS harness is the reference).
//!   Determinism: same seed → bit-identical final field, ≥100 seeds per arm.
//! - **G2 perf** — iteration collapse. PRIMARY: pure-local fixpoint ticks vs
//!   pca halt ticks (gate ≥ 3×; paper suggests 3–8×, Sokoban 98→19).
//!   ALSO RECORDED RAW: first-tick-to-b0==1 per arm — identical across arms BY
//!   CONSTRUCTION (the kernel dynamics are shared until the global gate trips;
//!   the trip costs one extra tick). The collapse value of the global function
//!   IS the early stop, which pure-local cannot express.
//! - **G3 no-regression** — in-bench: 10 ticks through `step_pca_sync` with a
//!   never-tripping gate are bit-identical to 10 stock kernel ticks (same
//!   seed) — the decision wrapper perturbs nothing when untripped. External:
//!   the flag-off test count and untouched kernel sources (bench doc).
//! - **G4 alloc** — zero allocations across 100 ticks of each step path
//!   (scratch-only), via the shared CountingAllocator.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/plan591 \
//! cargo run -p katgpt-dec --release --features pca_global \
//!   --bench bench_591_pca_global_goat -- --nocapture
//! ```

#![cfg(feature = "pca_global")]

// Shared CountingAllocator macro (mirrors katgpt-core Issue 044 T3).
#[path = "../tests/common/counting_allocator.rs"]
mod counting_allocator;

use katgpt_dec::{
    BirthDeathParams, CellComplex, CochainField, GlobalTargetGate, PcaGlobalFn, PcaScratch,
    SplitMix64, StopWhen, step_pca_async, step_pca_sync, stochastic_birth_death_step,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

counting_allocator!();

// ===========================================================================
// Arena
// ===========================================================================

const W: usize = 64;
const H: usize = 64;
const DIM: usize = 2; // alive + morphogen (Plan 454 layout)
const N_CELLS: usize = W * H;
const MAX_TICKS: usize = 400; // hard cap; the arena converges far earlier
const G_SEEDS: u64 = 10; // G1/G2 seed sweep
const DET_SEEDS: u64 = 100; // G1 determinism sweep
const ALLOC_TICKS: usize = 100; // G4 measured window

/// grid_2d row-major vertex index (z-slowest convention collapses in 2D).
#[inline]
fn vidx(x: usize, y: usize) -> usize {
    y * W + x
}

fn arena() -> CellComplex {
    CellComplex::grid_2d(W, H)
}

/// Seed diamond: four cells at Manhattan spacing 8 around the center. The
/// blobs merge at tick 4 (T_connect), the gate trips at tick 5; the farthest
/// corner sits 60 Manhattan ticks from any seed (T_fill). Both endpoints are
/// MEASURED every run, never assumed.
fn seed_vertices() -> [usize; 4] {
    [vidx(28, 32), vidx(36, 32), vidx(32, 28), vidx(32, 36)]
}

fn seed_field(field: &mut CochainField) {
    for v in field.data.iter_mut() {
        *v = 0.0;
    }
    for s in seed_vertices() {
        field.data[s * DIM] = 1.0;
        field.data[s * DIM + 1] = 1.0;
    }
}

#[inline]
fn alive_at(field: &CochainField, v: usize) -> bool {
    field.data[v * DIM] > 0.5
}

fn alive_count(field: &CochainField) -> usize {
    (0..N_CELLS).filter(|&v| alive_at(field, v)).count()
}

/// Independent BFS connectivity reference — 4-neighborhood, mirrors the Phase
/// 0 pin's harness (`betti_numbers(cx)` is state-blind; Plan 591 Phase 0
/// finding). Returns (component count, largest component size).
fn bfs_components(field: &CochainField) -> (usize, usize) {
    let mut seen = vec![false; N_CELLS];
    let mut count = 0;
    let mut max = 0;
    for start in 0..N_CELLS {
        if !alive_at(field, start) || seen[start] {
            continue;
        }
        count += 1;
        let mut size = 0;
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(v) = stack.pop() {
            size += 1;
            let x = v % W;
            let y = v / W;
            let visit = |w: usize, seen: &mut Vec<bool>, stack: &mut Vec<usize>| {
                if alive_at(field, w) && !seen[w] {
                    seen[w] = true;
                    stack.push(w);
                }
            };
            if x > 0 {
                visit(v - 1, &mut seen, &mut stack);
            }
            if x + 1 < W {
                visit(v + 1, &mut seen, &mut stack);
            }
            if y > 0 {
                visit(v - W, &mut seen, &mut stack);
            }
            if y + 1 < H {
                visit(v + W, &mut seen, &mut stack);
            }
        }
        max = max.max(size);
    }
    (count, max)
}

fn bit_identical(a: &CochainField, b: &CochainField) -> bool {
    a.data.len() == b.data.len()
        && a.data.iter().zip(b.data.iter()).all(|(&x, &y)| x.to_bits() == y.to_bits())
}

// ===========================================================================
// Competitors
// ===========================================================================

/// (a) Pure-local: the stock kernel, no decision layer. Returns
/// (ticks_to_fixpoint, ticks_to_first_b0_1, field). A pure-local rule cannot
/// verify connectivity, so its halt is the fixpoint: every vertex alive.
fn run_pure_local(params: &BirthDeathParams, seed: u64) -> (usize, usize, CochainField) {
    let cx = arena();
    let mut field = CochainField::zeros(0, N_CELLS, DIM);
    seed_field(&mut field);
    let mut lap = CochainField::zeros(0, N_CELLS, DIM);
    let mut dropout = vec![0u8; N_CELLS];
    let mut rng = SplitMix64::new(seed);
    let mut t_connect = 0;
    for t in 1..=MAX_TICKS {
        stochastic_birth_death_step(&cx, &mut field, params, &mut rng, &mut lap, &mut dropout);
        if t_connect == 0 && bfs_components(&field).0 == 1 {
            t_connect = t;
        }
        if alive_count(&field) == N_CELLS {
            return (t, t_connect, field);
        }
    }
    (MAX_TICKS, t_connect, field)
}

/// (b)/(c) PCA arms: same kernel + the global gate (Betti0 ≤ 1). Halts on the
/// tick the gate trips — the first PRE-tick evaluation reporting one
/// component, i.e. one tick AFTER the BFS-verified connection. Returns
/// (ticks_to_halt, ticks_to_first_b0_1, field).
fn run_pca(params: &BirthDeathParams, seed: u64, use_async: bool) -> (usize, usize, CochainField) {
    let cx = arena();
    let mut field = CochainField::zeros(0, N_CELLS, DIM);
    seed_field(&mut field);
    let mut scratch = PcaScratch::for_complex(&cx, DIM);
    let gate = GlobalTargetGate { target: 1.0, stop_when: StopWhen::Below };
    let mut rng = SplitMix64::new(seed);
    let mut t_connect = 0;
    for t in 1..=MAX_TICKS {
        let global = if use_async {
            step_pca_async(
                black_box(&cx),
                black_box(&mut field),
                black_box(params),
                black_box(&mut rng),
                black_box(&PcaGlobalFn::Betti0),
                black_box(&gate),
                black_box(&mut scratch),
            )
        } else {
            step_pca_sync(
                black_box(&cx),
                black_box(&mut field),
                black_box(params),
                black_box(&mut rng),
                black_box(&PcaGlobalFn::Betti0),
                black_box(&gate),
                black_box(&mut scratch),
            )
        };
        if t_connect == 0 && bfs_components(&field).0 == 1 {
            t_connect = t;
        }
        if global <= 1.0 {
            return (t, t_connect, field);
        }
    }
    (MAX_TICKS, t_connect, field)
}

/// (d) Static-generator incumbent: one fixed connected pattern, zero
/// iterations. The row 32 line from x=16 to x=48 (33 cells).
fn static_incumbent() -> CochainField {
    let mut field = CochainField::zeros(0, N_CELLS, DIM);
    for x in 16..=48 {
        let v = vidx(x, 32);
        field.data[v * DIM] = 1.0;
        field.data[v * DIM + 1] = 1.0;
    }
    field
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

// ===========================================================================
// G1 — correctness (BFS-verified finals) + determinism (≥100 seeds)
// ===========================================================================

fn g1_correctness(params: &BirthDeathParams) -> bool {
    for seed in 0..G_SEEDS {
        let (_, _, pure) = run_pure_local(params, seed);
        if bfs_components(&pure).0 != 1 {
            println!("  seed {seed}: pure-local final state NOT connected");
            return false;
        }
        for (name, use_async) in [("sync", false), ("async", true)] {
            let (_, _, field) = run_pca(params, seed, use_async);
            if bfs_components(&field).0 != 1 {
                println!("  seed {seed}: pca-{name} final state NOT connected");
                return false;
            }
        }
    }
    true
}

fn g1_determinism(params: &BirthDeathParams) -> bool {
    for seed in 0..DET_SEEDS {
        let (t1, _, f1) = run_pure_local(params, seed);
        let (t2, _, f2) = run_pure_local(params, seed);
        if t1 != t2 || !bit_identical(&f1, &f2) {
            println!("  seed {seed}: pure-local not bit-identical");
            return false;
        }
        for (name, use_async) in [("sync", false), ("async", true)] {
            let (t1, _, f1) = run_pca(params, seed, use_async);
            let (t2, _, f2) = run_pca(params, seed, use_async);
            if t1 != t2 || !bit_identical(&f1, &f2) {
                println!("  seed {seed}: pca-{name} not bit-identical");
                return false;
            }
        }
    }
    true
}

// ===========================================================================
// G2 — iteration collapse (median over G_SEEDS)
// ===========================================================================

struct G2Row {
    pure_fill: f64,
    pure_connect: f64,
    sync_halt: f64,
    sync_connect: f64,
    async_halt: f64,
    async_connect: f64,
}

fn g2_iteration_collapse(params: &BirthDeathParams) -> (G2Row, f64, f64, bool) {
    let mut pure_fill = Vec::with_capacity(G_SEEDS as usize);
    let mut pure_connect = Vec::with_capacity(G_SEEDS as usize);
    let mut sync_halt = Vec::with_capacity(G_SEEDS as usize);
    let mut sync_connect = Vec::with_capacity(G_SEEDS as usize);
    let mut async_halt = Vec::with_capacity(G_SEEDS as usize);
    let mut async_connect = Vec::with_capacity(G_SEEDS as usize);

    for seed in 0..G_SEEDS {
        let (fill, connect, _) = run_pure_local(params, seed);
        pure_fill.push(fill as f64);
        pure_connect.push(connect as f64);
        let (halt, connect, _) = run_pca(params, seed, false);
        sync_halt.push(halt as f64);
        sync_connect.push(connect as f64);
        let (halt, connect, _) = run_pca(params, seed, true);
        async_halt.push(halt as f64);
        async_connect.push(connect as f64);
    }

    let row = G2Row {
        pure_fill: median(&mut pure_fill),
        pure_connect: median(&mut pure_connect),
        sync_halt: median(&mut sync_halt),
        sync_connect: median(&mut sync_connect),
        async_halt: median(&mut async_halt),
        async_connect: median(&mut async_connect),
    };

    // PRIMARY gate: pure-local fixpoint ticks vs pca halt ticks.
    let ratio_sync = row.pure_fill / row.sync_halt;
    let ratio_async = row.pure_fill / row.async_halt;
    // ALSO RECORDED RAW (identical dynamics until the gate trips — expected 1):
    let ratio_literal = row.pure_connect / row.sync_connect;
    let pass = ratio_sync >= 3.0 && ratio_async >= 3.0;
    (row, (ratio_sync + ratio_async) / 2.0, ratio_literal, pass)
}

// ===========================================================================
// G3 — no-regression: the wrapper perturbs nothing when the gate never trips
// ===========================================================================

fn g3_kernel_unchanged(params: &BirthDeathParams) -> bool {
    let cx = arena();

    let mut pure = CochainField::zeros(0, N_CELLS, DIM);
    seed_field(&mut pure);
    let mut lap = CochainField::zeros(0, N_CELLS, DIM);
    let mut dropout = vec![0u8; N_CELLS];
    let mut rng = SplitMix64::new(7);
    for _ in 0..10 {
        stochastic_birth_death_step(&cx, &mut pure, params, &mut rng, &mut lap, &mut dropout);
    }

    // Same kernel through step_pca_sync with a NEVER-tripping gate
    // (Betti0 ≤ 0 is unreachable while the seeds live).
    let mut wrapped = CochainField::zeros(0, N_CELLS, DIM);
    seed_field(&mut wrapped);
    let mut scratch = PcaScratch::for_complex(&cx, DIM);
    let never = GlobalTargetGate { target: 0.0, stop_when: StopWhen::Below };
    let mut rng = SplitMix64::new(7);
    for _ in 0..10 {
        step_pca_sync(
            &cx,
            &mut wrapped,
            params,
            &mut rng,
            &PcaGlobalFn::Betti0,
            &never,
            &mut scratch,
        );
    }
    bit_identical(&pure, &wrapped)
}

// ===========================================================================
// G4 — zero alloc per tick (scratch-only), sync + async
// ===========================================================================

fn g4_alloc(params: &BirthDeathParams) -> (usize, usize) {
    let cx = arena();

    // Warmup (setup allocations allowed), then a measured 100-tick window.
    let mut field = CochainField::zeros(0, N_CELLS, DIM);
    seed_field(&mut field);
    let mut scratch = PcaScratch::for_complex(&cx, DIM);
    let gate = GlobalTargetGate { target: 1.0, stop_when: StopWhen::Below };
    let mut rng = SplitMix64::new(3);
    step_pca_sync(&cx, &mut field, params, &mut rng, &PcaGlobalFn::Betti0, &gate, &mut scratch);
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..ALLOC_TICKS {
        step_pca_sync(
            black_box(&cx),
            black_box(&mut field),
            black_box(params),
            black_box(&mut rng),
            black_box(&PcaGlobalFn::Betti0),
            black_box(&gate),
            black_box(&mut scratch),
        );
    }
    let sync_allocs = ALLOC_COUNT.load(Ordering::Relaxed) - before;

    seed_field(&mut field);
    let mut rng = SplitMix64::new(3);
    step_pca_async(&cx, &mut field, params, &mut rng, &PcaGlobalFn::Betti0, &gate, &mut scratch);
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..ALLOC_TICKS {
        step_pca_async(
            black_box(&cx),
            black_box(&mut field),
            black_box(params),
            black_box(&mut rng),
            black_box(&PcaGlobalFn::Betti0),
            black_box(&gate),
            black_box(&mut scratch),
        );
    }
    let async_allocs = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    (sync_allocs, async_allocs)
}

// ===========================================================================
// Driver
// ===========================================================================

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS ✅" } else { "FAIL ❌" }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 591 Phase 3 — PCA global-function GOAT gate (G1–G4), Bench 707 ║");
    println!("║  Arena: {W}×{H} grid_2d, 4-seed diamond (spacing 8), paper_defaults   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let params = BirthDeathParams::paper_defaults();
    let mut gain_pass = true; // G1 + G2
    let mut eng_pass = true; // G3 + G4

    // --- G1: correctness + determinism ---
    print!("[G1]  BFS-verified finals ({G_SEEDS} seeds/arm) + determinism ({DET_SEEDS} seeds/arm)... ");
    let g1c = g1_correctness(&params);
    let g1d = g1_determinism(&params);
    println!("done");
    println!("  G1 correctness (final b0==1 by independent BFS)  → {}", verdict(g1c));
    println!("  G1 determinism (bit-identical, {DET_SEEDS} seeds × 3 arms)  → {}", verdict(g1d));
    gain_pass &= g1c && g1d;

    // --- G2: iteration collapse ---
    print!("[G2]  Iteration collapse (median of {G_SEEDS} seeds)... ");
    let (row, ratio, ratio_literal, g2) = g2_iteration_collapse(&params);
    println!("done");
    println!(
        "  G2 primary: pure-local fixpoint {:.0} ticks vs pca halt sync {:.0} / async {:.0} → collapse {:.2}× (gate ≥ 3×)  → {}",
        row.pure_fill, row.sync_halt, row.async_halt, ratio, verdict(g2)
    );
    println!(
        "  G2 raw diagnostic: first-tick-to-b0==1 pure {:.0} vs pca sync {:.0} / async {:.0} → {:.2}× (identical pre-trip dynamics, BY CONSTRUCTION)",
        row.pure_connect, row.sync_connect, row.async_connect, ratio_literal
    );
    gain_pass &= g2;

    // --- G3: no-regression ---
    print!("[G3]  Wrapper-noop (10 ticks, never-tripping gate vs stock kernel)... ");
    let g3 = g3_kernel_unchanged(&params);
    println!("done");
    println!(
        "  G3 in-bench: untripped step_pca_sync bit-identical to stock kernel  → {}",
        verdict(g3)
    );
    println!("  G3 external: kernel sources untouched by Plan 591; flag-off test count asserted separately (bench doc)");
    eng_pass &= g3;

    // --- G4: zero alloc ---
    print!("[G4]  Zero-alloc ({ALLOC_TICKS} ticks per step path)... ");
    let (sync_allocs, async_allocs) = g4_alloc(&params);
    println!("done");
    println!(
        "  G4 allocs in {ALLOC_TICKS} ticks: sync {sync_allocs}, async {async_allocs} (gate = 0/0)  → {}",
        verdict(sync_allocs == 0 && async_allocs == 0)
    );
    eng_pass &= sync_allocs == 0 && async_allocs == 0;

    // --- Static-generator incumbent (informational, never gated) ---
    let incumbent = static_incumbent();
    let (b0, largest) = bfs_components(&incumbent);
    println!();
    println!(
        "  Static incumbent: 0 iterations, BFS b0={b0}, largest component {largest} — fixed artifact, zero seed variation (context row, not a gate)"
    );

    // --- Wall-clock context (one timed sweep, informational) ---
    let t0 = Instant::now();
    let _ = run_pure_local(&params, 0);
    let pure_wall = t0.elapsed();
    let t0 = Instant::now();
    let _ = run_pca(&params, 0, true);
    let pca_wall = t0.elapsed();
    println!(
        "  Wall context (1 seed): pure-local run {:?}, pca-async run {:?} (pca arm pays the per-tick global evaluation AND finishes earlier)",
        pure_wall, pca_wall
    );

    println!();
    println!("GOAT verdict: gain (G1+G2) {} · engineering (G3+G4) {}", verdict(gain_pass), verdict(eng_pass));
    println!("Ruling: {}", if gain_pass && eng_pass {
        "PROMOTE pca_global to default (all gates pass)"
    } else {
        "STAY OPT-IN (record raw numbers in Bench 707)"
    });
}
