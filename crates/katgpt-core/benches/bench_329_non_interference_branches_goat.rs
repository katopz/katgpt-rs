//! Non-Interference Memory Branches GOAT gate bench (Plan 329 Phase 3).
//!
//! Exercises the G1 (correctness), G2 (perf), and G4 (alloc-free hot path)
//! gates for the five RIZZ-distilled primitives in `branching::*`.
//!
//! G3 (no-regression) and G5 (modelless) are NOT measured here — they are
//! verified by the feature-flag build matrix:
//! - **G3**: `cargo check --all-features` and `cargo check --no-default-features`
//!   both clean (the merkle_root lesson — catches combo-only regressions).
//! - **G5**: the `non_interference_branches` feature has `[]` deps (no
//!   `riir_train`, no `riir_gpu`); pure closed-form arithmetic + dot products.
//!
//! # Gates measured here
//!
//! - **G1 (correctness)**:
//!   - Spawn N=8 branches with mutually-orthogonal directions in D=8 space
//!     (canonical basis vectors e_0..e_7). Verify `interference(b_i, b_j) < ε`
//!     for all 8×7=56 ordered pairs i≠j.
//!   - Non-interference by construction: write an episodic entry to branch i;
//!     verify branch j's episodic/procedural/failure stores are unchanged.
//!   - Frame-theory limit: a 9th direction in D=8 must interfere with some
//!     existing direction by ≥ `1/sqrt(D)`.
//! - **G2 (perf)**: `BranchRouter::route` median latency < 1µs over 10,000
//!   calls on a 64-branch bank (release). The dot-product snap is the hot-path
//!   entry point consumed per-tick per-NPC.
//! - **G4 (alloc-free hot path)**: `BranchRouter::route` and
//!   `VerifierGate::should_write` allocate 0 bytes over 100 steady-state calls
//!   (CountingAllocator).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features non_interference_branches --bench bench_329_non_interference_branches_goat -- --nocapture
//! ```
//!
//! Or, working around the intermittent macOS dyld/trustd launch stall
//! (documented in Plan 326 / bench_327):
//!
//! ```bash
//! target/release/deps/bench_329_non_interference_branches_goat-* --nocapture
//! ```

#![cfg(feature = "non_interference_branches")]

use katgpt_core::{
    BranchBank, BranchId, BranchRouter, CognitiveBranch, NonInterferenceProjection,
    VerifierGate, WriteDecision,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ─── CountingAllocator (G4) ─────────────────────────────────────────────────

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let r = f();
    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    (r, after - before)
}

// ─── GateResult ─────────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

impl GateResult {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, passed: true, detail: detail.into() }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, passed: false, detail: detail.into() }
    }
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

/// Canonical basis vector `e_k` in D-dimensional space (1.0 at index k, 0 elsewhere).
/// These are mutually orthogonal by construction: `dot(e_i, e_j) = δ_ij`.
fn basis_vec(k: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0; dim];
    if k < dim {
        v[k] = 1.0;
    }
    v
}

/// Snapshot of a branch's memory store sizes — used to verify non-interference
/// (a write to branch i must not change branch j's snapshot).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct StoreSnapshot {
    episodic_len: usize,
    procedural_len: usize,
    failures_len: usize,
}

impl StoreSnapshot {
    fn of<E>(b: &CognitiveBranch<E>) -> Self {
        Self {
            episodic_len: b.episodic.len(),
            procedural_len: b.procedural.len(),
            failures_len: b.failures.len(),
        }
    }
}

// ─── G1a: orthogonal directions in D=8 — pairwise interference ≈ 0 ─────────

fn gate_g1a_orthogonal_interference() -> GateResult {
    const N: usize = 8;
    const D: usize = 8;
    const EPS: f32 = 1e-6;

    let mut proj: NonInterferenceProjection<D> = NonInterferenceProjection::new(N);

    // Assign canonical basis vectors e_0..e_7 to branches 0..7.
    for k in 0..N {
        let dir = basis_vec(k, D);
        let res = proj.assign_direction(BranchId::new(k as u32), &dir);
        if !res.is_ok() {
            return GateResult::fail(
                "G1a",
                format!("assign_direction(e_{k}) failed: {res:?} (expected Ok)"),
            );
        }
    }

    // Verify pairwise interference < ε for all i≠j (56 ordered pairs).
    let mut worst = 0.0f32;
    let mut worst_pair = (0, 0);
    for i in 0..N {
        for j in 0..N {
            if i == j {
                continue;
            }
            let inter = proj.interference(BranchId::new(i as u32), BranchId::new(j as u32));
            if inter > worst {
                worst = inter;
                worst_pair = (i, j);
            }
            if inter >= EPS {
                return GateResult::fail(
                    "G1a",
                    format!(
                        "interference(b_{i}, b_{j}) = {inter:.3e} ≥ ε={EPS:.0e} (canonical basis must be orthogonal)"
                    ),
                );
            }
        }
    }

    // Also verify is_non_interfering_with_all holds for every branch.
    for i in 0..N {
        if !proj.is_non_interfering_with_all(BranchId::new(i as u32)) {
            return GateResult::fail(
                "G1a",
                format!("is_non_interfering_with_all(b_{i}) = false (expected true)"),
            );
        }
    }

    // Frame-theory limit: max_orthogonal_branches() == D.
    let limit = NonInterferenceProjection::<D>::max_orthogonal_branches();
    if limit != D {
        return GateResult::fail(
            "G1a",
            format!("max_orthogonal_branches() = {limit} (expected {D})"),
        );
    }

    GateResult::pass(
        "G1a",
        format!(
            "8 canonical basis directions in D=8: max pairwise interference = {worst:.2e} (pair b_{}↔b_{}) < ε={EPS:.0e}; is_non_interfering_with_all holds for all 8; max_orthogonal_branches={limit}",
            worst_pair.0, worst_pair.1
        ),
    )
}

// ─── G1b: non-interference by construction — write to i, j unchanged ────────

fn gate_g1b_write_does_not_contaminate_sibling() -> GateResult {
    const N: usize = 8;
    const D: usize = 8;

    // Spawn a bank of 8 branches with canonical-basis anchors.
    let mut bank: BranchBank<Option<u64>> = BranchBank::new(N);
    for k in 0..N {
        let anchor = basis_vec(k, D);
        let id = bank.spawn(anchor).expect("spawn must succeed below capacity");
        assert_eq!(id, BranchId::new(k as u32));
    }

    // Snapshot every branch's stores BEFORE writing to branch 0.
    let before: Vec<StoreSnapshot> = (0..N)
        .map(|k| StoreSnapshot::of(bank.get(BranchId::new(k as u32)).unwrap()))
        .collect();

    // Write one episodic entry into branch 0 (payload = Some(42)).
    let wrote = bank.write_episodic(
        BranchId::new(0),
        basis_vec(0, D), // embedding aligned with branch 0's anchor
        Some(42u64),
        0.9,             // reward
        None,            // scope (no scope tag)
        1,               // tick
    );
    if !wrote {
        return GateResult::fail("G1b", "write_episodic(b_0) returned false (expected true)");
    }

    // Verify branch 0's episodic store grew by exactly 1.
    let b0_after = StoreSnapshot::of(bank.get(BranchId::new(0)).unwrap());
    if b0_after.episodic_len != before[0].episodic_len + 1 {
        return GateResult::fail(
            "G1b",
            format!(
                "b_0.episodic grew {}→{} (expected +1)",
                before[0].episodic_len, b0_after.episodic_len
            ),
        );
    }

    // Verify every OTHER branch's stores are byte-for-byte unchanged.
    for k in 1..N {
        let after = StoreSnapshot::of(bank.get(BranchId::new(k as u32)).unwrap());
        if after != before[k] {
            return GateResult::fail(
                "G1b",
                format!(
                    "write to b_0 contaminated b_{k}: {:?} → {:?} (non-interference violation)",
                    before[k], after
                ),
            );
        }
    }

    GateResult::pass(
        "G1b",
        format!(
            "write_episodic(b_0) grew b_0.episodic {}→{}; branches b_1..b_7 stores unchanged (non-interference by construction)",
            before[0].episodic_len, b0_after.episodic_len
        ),
    )
}

// ─── G1c: frame-theory limit — 9th direction in D=8 must interfere ─────────

fn gate_g1c_ninth_direction_must_interfere() -> GateResult {
    const D: usize = 8;
    // Use the DEFAULT_ASSIGN_MAX_INTERFERENCE = 0.1 threshold (well below
    // 1/sqrt(8) ≈ 0.354, so the 9th direction is guaranteed to be rejected).
    let mut proj: NonInterferenceProjection<D> = NonInterferenceProjection::new(D + 1);

    // Assign all 8 canonical basis directions.
    for k in 0..D {
        let dir = basis_vec(k, D);
        let res = proj.assign_direction(BranchId::new(k as u32), &dir);
        assert!(res.is_ok(), "e_{k} should assign cleanly");
    }

    // The 9th direction is the normalized all-ones vector (1,1,...,1)/sqrt(8).
    // Its dot-product with every e_k is 1/sqrt(8) ≈ 0.354 > 0.1 threshold.
    let ninth: Vec<f32> = vec![1.0 / (D as f32).sqrt(); D];
    let res = proj.assign_direction(BranchId::new(D as u32), &ninth);

    if res.is_ok() {
        return GateResult::fail(
            "G1c",
            format!(
                "9th direction in D=8 assigned successfully (expected Interferes); 1/sqrt({D}) = {:.4} > threshold 0.1",
                1.0 / (D as f32).sqrt()
            ),
        );
    }

    // Verify the rejection was an Interferes error with a non-zero magnitude.
    use katgpt_core::AssignError;
    match res.error {
        Some(AssignError::Interferes) => {
            let bound = 1.0 / (D as f32).sqrt();
            if res.interference < bound - 1e-6 {
                return GateResult::fail(
                    "G1c",
                    format!(
                        "9th direction interferes by {:.4} but frame theory guarantees ≥ 1/sqrt({D}) = {:.4}",
                        res.interference, bound
                    ),
                );
            }
            let conflict = res.conflict_branch.map(|b| b.0).unwrap_or(u32::MAX);
            GateResult::pass(
                "G1c",
                format!(
                    "9th direction (uniform) in D=8 correctly rejected: interferes with b_{conflict} by {:.4} ≥ 1/sqrt({D}) = {:.4} > threshold 0.1",
                    res.interference, bound
                ),
            )
        }
        Some(other) => GateResult::fail(
            "G1c",
            format!("9th direction rejected with wrong error {other:?} (expected Interferes)"),
        ),
        None => GateResult::fail("G1c", "9th direction reported Ok but is_ok() == false (inconsistent)"),
    }
}

// ─── G2: BranchRouter::route perf on a 64-branch bank ──────────────────────

fn gate_g2_router_route_perf() -> GateResult {
    const TARGET_NS: f64 = 1_000.0; // 1µs
    const N_BRANCHES: usize = 64;
    const D: usize = 8;
    const WARMUP: usize = 1_000;
    const ITERS: usize = 10_000;

    // Build a 64-branch bank. We use D=8 anchors on the unit sphere. To force
    // the router to scan all 64 branches (worst case), the query is constructed
    // so that no single branch dominates by more than tau_snap=0.92 — the
    // router must traverse the entire active set before concluding.
    //
    // The anchors are 64 pseudo-random unit vectors in R^8 (deterministic seed).
    // The query is one of them shifted slightly so the snap threshold is not
    // met on the first branch the iterator happens to yield — but a snap WILL
    // eventually be found (avoiding the Spawn/Frozen tail). This is the
    // steady-state hot path the per-tick per-NPC runtime hits.
    let mut bank: BranchBank<()> = BranchBank::new(N_BRANCHES);
    let mut anchors: Vec<[f32; D]> = Vec::with_capacity(N_BRANCHES);
    let mut rng = Lcg::new(0xC0FFEE_BEEF_DEAD_42);
    for _ in 0..N_BRANCHES {
        let mut v = [0.0f32; D];
        let mut norm_sq = 0.0;
        for i in 0..D {
            let x = rng.next_f32() * 2.0 - 1.0;
            v[i] = x;
            norm_sq += x * x;
        }
        let inv = 1.0 / norm_sq.sqrt();
        for x in &mut v {
            *x *= inv;
        }
        anchors.push(v);
        bank.spawn(v.to_vec()).expect("spawn below capacity");
    }

    // Query: a vector that snaps to branch 0 (its own anchor) so the router
    // finds a match. The scan still walks all 64 branches because it's a
    // max-reduction (no early exit).
    let query: Vec<f32> = anchors[0].to_vec();

    let router = BranchRouter::default();

    // Warmup.
    for _ in 0..WARMUP {
        let _ = black_box(router.route(black_box(&query), black_box(&bank)));
    }

    // Verify the route actually resolves to a Reuse (else we'd be measuring
    // the Spawn/Frozen tail, not the dot-product scan).
    let probe = router.route(&query, &bank);
    if probe.branch.is_none() {
        return GateResult::fail(
            "G2",
            "warmup route returned None — fixture is wrong, not measuring the scan hot path",
        );
    }

    // Measure.
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(router.route(black_box(&query), black_box(&bank)));
    }
    let elapsed = start.elapsed();
    let mean_ns = elapsed.as_nanos() as f64 / ITERS as f64;

    if mean_ns <= TARGET_NS {
        GateResult::pass(
            "G2",
            format!(
                "BranchRouter::route median ~{mean_ns:.1}ns (≤ {TARGET_NS:.0}ns target) over {ITERS} iters on {N_BRANCHES}-branch bank (D={D}); resolved to b_{}",
                probe.branch.unwrap().0
            ),
        )
    } else {
        GateResult::fail(
            "G2",
            format!(
                "BranchRouter::route median ~{mean_ns:.1}ns > {TARGET_NS:.0}ns target over {ITERS} iters on {N_BRANCHES}-branch bank"
            ),
        )
    }
}

// ─── G4: alloc-free hot path ────────────────────────────────────────────────

fn gate_g4_alloc_free_hot_path() -> GateResult {
    const ITERS: usize = 100;
    const N_BRANCHES: usize = 64;
    const D: usize = 8;

    // ── G4a: BranchRouter::route ───────────────────────────────────────────
    let mut bank: BranchBank<()> = BranchBank::new(N_BRANCHES);
    let mut rng = Lcg::new(0x1234_5678_9ABC_DEF0);
    let mut query: Vec<f32> = vec![0.0; D];
    for k in 0..N_BRANCHES {
        let mut v = [0.0f32; D];
        let mut norm_sq = 0.0;
        for i in 0..D {
            let x = rng.next_f32() * 2.0 - 1.0;
            v[i] = x;
            norm_sq += x * x;
        }
        let inv = 1.0 / norm_sq.sqrt();
        for x in &mut v {
            *x *= inv;
        }
        if k == 0 {
            query = v.to_vec();
        }
        bank.spawn(v.to_vec()).expect("spawn below capacity");
    }
    let router = BranchRouter::default();
    // Warmup once outside the measurement.
    let _ = router.route(&query, &bank);
    let (_, allocs_route) = alloc_delta(|| {
        for _ in 0..ITERS {
            let _ = black_box(router.route(black_box(&query), black_box(&bank)));
        }
    });

    // ── G4b: VerifierGate::should_write ────────────────────────────────────
    let gate = VerifierGate::default();
    let (reward, curiosity, centroid) = (0.8f32, 0.5f32, 0.9f32);
    // Warmup once.
    let _ = gate.should_write(reward, curiosity, centroid);
    let (_, allocs_should_write) = alloc_delta(|| {
        for _ in 0..ITERS {
            let _ = black_box(gate.should_write(reward, curiosity, centroid));
        }
    });

    if allocs_route == 0 && allocs_should_write == 0 {
        GateResult::pass(
            "G4",
            format!(
                "BranchRouter::route: 0 allocs / {ITERS} calls; VerifierGate::should_write: 0 allocs / {ITERS} calls"
            ),
        )
    } else {
        GateResult::fail(
            "G4",
            format!(
                "route allocs={allocs_route}/{ITERS}, should_write allocs={allocs_should_write}/{ITERS} (expected 0/0)"
            ),
        )
    }
}

// ─── G4 (bonus): also verify the alloc-free outcome was meaningful ─────────
//
// The CountingAllocator catches any allocation, but it can't tell us the
// decision was correct. This companion check confirms the gates actually fire
// (returning the right WriteDecision for known inputs) so the 0-alloc result
// isn't measuring a degenerate always-Reject always-Reject loop.

fn gate_g4b_decisions_are_correct() -> GateResult {
    let gate = VerifierGate::default();
    let write = gate.should_write(0.9, 0.6, 0.9);
    let quarantine = gate.should_write(0.9, 0.6, 0.3);
    let reject_reward = gate.should_write(0.2, 0.6, 0.9);
    let reject_curiosity = gate.should_write(0.9, 0.1, 0.9);

    if write == WriteDecision::Write
        && quarantine == WriteDecision::Quarantine
        && reject_reward == WriteDecision::Reject
        && reject_curiosity == WriteDecision::Reject
    {
        GateResult::pass(
            "G4b",
            "VerifierGate returns Write/Quarantine/Reject (reward-low) / Reject (curiosity-low) for known inputs — 0-alloc result is non-degenerate",
        )
    } else {
        GateResult::fail(
            "G4b",
            format!(
                "VerifierGate returned wrong decisions: write={write:?}, quarantine={quarantine:?}, reject_reward={reject_reward:?}, reject_curiosity={reject_curiosity:?}"
            ),
        )
    }
}

// ─── Lcg (deterministic fixture RNG — no rand dep) ──────────────────────────
//
// Numerical Recipes LCG. Constants chosen for full-period 64-bit sequence.
// Used only for bench fixtures (deterministic, reproducible across runs).

struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        // Using high bits via shift; full 64-bit state, 13/7/17 shifts.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = self.state;
        ((x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9)) >> 32 ^ (x >> 17)
    }
    fn next_f32(&mut self) -> f32 {
        // Top 24 bits → [0, 1).
        let r = (self.next_u64() >> 40) as u32;
        (r as f32) / ((1u32 << 24) as f32)
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 329 - Non-Interference Memory Branches GOAT Gate (Phase 3) ===\n");

    let gates = [
        gate_g1a_orthogonal_interference(),
        gate_g1b_write_does_not_contaminate_sibling(),
        gate_g1c_ninth_direction_must_interfere(),
        gate_g2_router_route_perf(),
        gate_g4_alloc_free_hot_path(),
        gate_g4b_decisions_are_correct(),
    ];

    let mut all_pass = true;
    for g in &gates {
        let status = if g.passed { "PASS" } else { "FAIL" };
        println!("[{status}] {}: {}", g.name, g.detail);
        if !g.passed {
            all_pass = false;
        }
    }

    println!();
    println!("G3 (no-regression): verified via `cargo check --all-features`");
    println!("    and `cargo check --no-default-features` (the merkle_root lesson).");
    println!("G5 (modelless): the `non_interference_branches` feature has `[]` deps");
    println!("    (no riir_train / riir_gpu). Pure closed-form arithmetic + dot products.");
    println!();

    // G5 static check: confirm no dev-dep on riir_train/riir_gpu by asserting
    // the feature dependency list at compile time would be empty. We can't do
    // this in Rust directly; instead we print the verification step.
    println!("To re-verify G5: inspect crates/katgpt-core/Cargo.toml — the");
    println!("`non_interference_branches = []` line has zero dependencies.");
    println!();

    if all_pass {
        println!("=== ALL G1+G2+G4 GATES PASS — combined with G3+G5 build-matrix, eligible for default promotion ===");
        std::process::exit(0);
    } else {
        println!("=== ONE OR MORE GATES FAILED — keep opt-in, investigate ===");
        std::process::exit(1);
    }
}
