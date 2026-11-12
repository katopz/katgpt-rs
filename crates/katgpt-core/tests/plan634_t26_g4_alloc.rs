#![cfg(feature = "diffusion_diversity")]
//! Plan 634 T2.6 (G4) — the zero-alloc census over the diffusion-diversity
//! hot paths, the house counting-allocator shape (plan 625 T2.4 precedent).
//!
//! Separate test binary (the `#[global_allocator]` convention — crate-binary-
//! unique). ONE counted test walks every hot-path segment, so the counter is
//! serial by construction (no cross-test contamination window — the
//! basis_split gate's `SERIALIZE` mutex exists only because IT ships two
//! counted tests).
//!
//! Hot paths (the plan's bar: alloc census ZERO):
//! - `div_tau` — τ matvecs + the ℓ2(1/π) norm fold
//! - `diffusion_similarity_local` — the exact local posture (2τ matvecs)
//! - `SimGraph::diffuse_transpose` — the exported matvec primitive itself
//! - `neighbor_diversity_pick` — the walk-time selector (Alg. 4). Its QP
//!   solver originally `collect()`ed a free-list Vec PER ACTIVE-SET PASS —
//!   this census's first light caught it by inspection and the solver moved
//!   to a stack-resident list (k ≤ 64 asserted); the census now asserts the
//!   zero its doc comment has always claimed.
//! - `DiffusionEmbedding` query paths — `similarity_spectral_exact` +
//!   `spectral_coordinates` (the build is build-time and stays OUTSIDE the
//!   counted windows).
//!
//! Reported (NOT asserted — the documented build-time class, plan T1.6):
//! `batch_select_diffusion` allocates per call by design (embedding build +
//! candidate staging); the census prints its count so a future zero-alloc
//! rewrite is visible in the same run.

use katgpt_core::diversity::diffusion::{
    DiffusionEmbedding, LocalScratch, SPECTRAL_N_CAP, batch_select_diffusion,
    diffusion_similarity_local, neighbor_diversity_pick,
};

#[path = "plan634/mod.rs"]
mod plan634_common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACKING: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) == 1 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Runs `body` once uncounted (warm-up: scratch capacities, allocator
/// arenas), then counts `iters` steady-state passes. Returns the alloc count.
fn counted(mut body: impl FnMut(), iters: usize) -> usize {
    body(); // warm-up — uncounted
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    TRACKING.store(1, Ordering::Relaxed);
    for _ in 0..iters {
        body();
    }
    TRACKING.store(0, Ordering::Relaxed);
    ALLOC_COUNT.load(Ordering::Relaxed)
}

#[test]
fn g4_zero_alloc_hot_paths() {
    // The battery fixture (plan634_common's pinned planted clique graph —
    // r = 2, c = 1 asserted at build).
    let (g, _cluster_of, _r, _d, _meta) = plan634_common::planted_clique_graph(8, 8);
    let n = g.n;
    let tau = 4;
    let mut scratch = LocalScratch::new(n);
    let mu = vec![1.0 / n as f32; n];
    let mut q = vec![0.0_f32; n];
    let mut r = vec![0.0_f32; n];
    // `out` pairs with |N(x)| = the UNWEIGHTED neighbor count (cols), not
    // the weighted `degree()` — planted(8,8) node 0 reads 23 columns at
    // weighted degree 9.0 (7 intra × 1.0 + 16 meta-join × 1/M).
    let (cols0, _) = g.row(0);
    let mut pick_out = vec![0.0_f32; cols0.len()];

    // ── div_tau (method on the graph) ──
    let count = counted(
        || {
            let d = g.div_tau(&mu, tau, &mut q, &mut r);
            black_box(d);
        },
        200,
    );
    assert_eq!(
        count, 0,
        "G4 FAIL: {count} allocations in div_tau (τ={tau}, n={n})"
    );

    // ── diffusion_similarity_local (cycling pairs) ──
    let mut pair = 0usize;
    let count = counted(
        || {
            pair = (pair + 7) % n;
            let s = diffusion_similarity_local(&g, 0, pair, tau, &mut scratch);
            black_box(s);
        },
        200,
    );
    assert_eq!(
        count, 0,
        "G4 FAIL: {count} allocations in diffusion_similarity_local (τ={tau}, n={n})"
    );

    // ── diffuse_transpose (the exported matvec primitive) ──
    let input = vec![0.5_f32; n];
    let mut out = vec![0.0_f32; n];
    let mut tr_scratch = vec![0.0_f32; n];
    let count = counted(
        || {
            g.diffuse_transpose(&input, &mut out, &mut tr_scratch);
            black_box(&out);
        },
        200,
    );
    assert_eq!(
        count, 0,
        "G4 FAIL: {count} allocations in diffuse_transpose (n={n})"
    );

    // ── neighbor_diversity_pick (the walk-time selector; node 0, degree 9) ──
    let count = counted(
        || {
            neighbor_diversity_pick(&g, 0, tau, 0.0, &mut scratch, &mut pick_out);
            black_box(&pick_out);
        },
        50,
    );
    assert_eq!(
        count,
        0,
        "G4 FAIL: {count} allocations in neighbor_diversity_pick (k={}, τ={tau}) — \
         the QP solver must stay stack-resident",
        pick_out.len()
    );

    // ── spectral query paths (build OUTSIDE the counted window) ──
    assert!(n <= SPECTRAL_N_CAP);
    let emb = DiffusionEmbedding::build(&g);
    let count = counted(
        || {
            let s = emb.similarity_spectral_exact(3, 11, tau);
            black_box(s);
        },
        200,
    );
    assert_eq!(
        count, 0,
        "G4 FAIL: {count} allocations in similarity_spectral_exact (k=n, n={n})"
    );
    let count = counted(
        || {
            let c = emb.spectral_coordinates::<16>(5, tau);
            black_box(c);
        },
        200,
    );
    assert_eq!(
        count, 0,
        "G4 FAIL: {count} allocations in spectral_coordinates::<16> (n={n})"
    );

    // ── batch_select_diffusion — the documented BUILD-TIME class ──
    let pool: Vec<usize> = (0..16).map(|i| i * 4).collect();
    let mut out = vec![u16::MAX; 8];
    let count = counted(
        || {
            let admitted = batch_select_diffusion::<16>(&g, &pool, tau, 0.0, &mut out);
            black_box(admitted);
        },
        1,
    );
    println!(
        "batch_select_diffusion census: {count} allocations per call \
         (n={n}, pool={} — documented build-time class, T1.6; reported, not gated)",
        pool.len()
    );
    black_box(&out);
}
