//! walk_rank — GOAT gate benchmark (riir-neuron-db Issue 974 T3; the ndb 640
//! lift).
//!
//! Groups:
//! - **`g2_latency`** — stage-1 `walk_rank` at the DEFAULT posture (W=32,
//!   K=8) on a synthetic 100k-node view (the ndb bench 517 surface point,
//!   re-run katgpt-core-side on the plain trait). Bar: ≤ 10 ms/call (the
//!   issue budget; ndb measured 0.13 ms worst cell on the real graph —
//!   75× headroom). A synthetic `StdHashMap` view is the worst case for the
//!   adapter (no papaya read locks, no guard), so this isolates the MATH
//!   cost.
//! - **`g4_alloc`** — allocations per stage-1 call at defaults (criterion's
//!   allocator is not instrumented here; the O(config)-not-O(steps) bar is
//!   asserted by the in-module `alloc_per_call_is_bounded_not_per_step`
//!   test — this group measures the latency of the sensitivity ladder
//!   instead).
//! - **`g1_determinism`** — byte-identical re-runs asserted on the bench
//!   fixture itself (a bench that can't verify its own input is a bar
//!   satisfied by absent work).
//!
//! Loud-zero defense: every group asserts a non-empty ranking before any
//! timing.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features walk_rank --bench walk_rank_goat
//! ```

#![cfg(feature = "walk_rank")]

use std::collections::HashMap as StdHashMap;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use katgpt_core::walk_rank::{
    ContextBudget, WalkGraphView, WalkRankConfig, budgeted_context, walk_rank,
};

/// Deterministic LCG (the unseeded-global-RNG gate): fixture construction
/// only — the ranker's own rng is seeded per config.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

/// Test hash with an embedded index in `hash[..8]`.
fn h(index: u64) -> [u8; 32] {
    let mut z = [0u8; 32];
    z[..8].copy_from_slice(&index.to_le_bytes());
    z
}

/// Synthetic view: a 100k-node graph with a ring backbone + deterministic
/// shortcut chords (LCG) — every node has ≥ 3 edges, so non-backtracking
/// walks always have a choice (the real-work shape, unlike a bare chain).
struct SyntheticView {
    adj: StdHashMap<[u8; 32], Vec<[u8; 32]>>,
    ticks: StdHashMap<[u8; 32], u64>,
}

impl SyntheticView {
    fn build(n: usize) -> Self {
        let mut rng = Lcg(0x6170_706C_65_6D6Fu64);
        let mut v = Self {
            adj: StdHashMap::with_capacity(n),
            ticks: StdHashMap::with_capacity(n),
        };
        for i in 0..n as u64 {
            let node = h(i);
            v.ticks.insert(node, i); // correlated ticks (ascending, like production)
            // Ring backbone (bidirectional).
            let next = h((i + 1) % n as u64);
            let prev = h((i + n as u64 - 1) % n as u64);
            v.adj.entry(node).or_default().push(next);
            v.adj.entry(node).or_default().push(prev);
            // Two deterministic shortcut chords.
            for _ in 0..2 {
                let j = rng.next_u64() % n as u64;
                if j != i {
                    v.adj.entry(node).or_default().push(h(j));
                }
            }
        }
        v
    }
}

impl WalkGraphView for SyntheticView {
    fn neighbors_of(&self, hash: &[u8; 32], out: &mut Vec<[u8; 32]>) {
        if let Some(nbs) = self.adj.get(hash) {
            out.extend_from_slice(nbs);
        }
    }

    fn tick_of(&self, hash: &[u8; 32]) -> Option<u64> {
        self.ticks.get(hash).copied()
    }
}

const N_NODES: usize = 100_000;
const N_QUERIES: usize = 8;

/// 8 spread query nodes.
fn queries(n: usize) -> Vec<[u8; 32]> {
    let stride = (n / N_QUERIES).max(1);
    (0..N_QUERIES).map(|k| h((k * stride) as u64)).collect()
}

fn g2_latency(c: &mut Criterion) {
    let view = SyntheticView::build(N_NODES);
    let qs = queries(N_NODES);
    let defaults = WalkRankConfig::default();

    // Warm + loud zero: the bar is meaningless if the walk reaches nothing.
    for q in &qs {
        let warm = walk_rank(&view, q, &defaults);
        assert!(
            !warm.is_empty(),
            "LOUD ZERO: walk_rank returned nothing on the synthetic fixture — the bar would be vacuous"
        );
        black_box(&warm);
    }

    let mut group = c.benchmark_group("g2_latency");
    group.throughput(criterion::Throughput::Elements(N_NODES as u64));
    group.bench_function("defaults_w32_k8_n100k", |b| {
        b.iter(|| {
            for q in &qs {
                let r = walk_rank(&view, black_box(q), &defaults);
                black_box(&r);
            }
        })
    });
    // The sensitivity ladder (the paper's cost shape; ndb bench 517 measured
    // 3.4×/6.2× — re-run upstream-side at the two smaller postures).
    for (w, k) in [(16u32, 8u32), (8, 4)] {
        let cfg = WalkRankConfig {
            walks: w,
            walk_len: k,
            ..WalkRankConfig::default()
        };
        group.bench_function(format!("sensitivity_w{w}_k{k}_n100k"), |b| {
            b.iter(|| {
                for q in &qs {
                    let r = walk_rank(&view, black_box(q), &cfg);
                    black_box(&r);
                }
            })
        });
    }
    group.finish();
}

fn g1_determinism(c: &mut Criterion) {
    let view = SyntheticView::build(10_000);
    let qs = queries(10_000);
    let defaults = WalkRankConfig::default();

    // The fixture's own identity check: run once outside timing, assert the
    // re-run is byte-identical, THEN time the (now proven stable) call.
    let first: Vec<_> = qs.iter().map(|q| walk_rank(&view, q, &defaults)).collect();
    for (q, r1) in qs.iter().zip(&first) {
        let r2 = walk_rank(&view, q, &defaults);
        assert_eq!(
            r1, &r2,
            "byte-identical re-runs (G1) must hold on the bench fixture"
        );
        assert!(!r1.is_empty(), "LOUD ZERO on the determinism fixture");
    }

    let mut group = c.benchmark_group("g1_determinism");
    group.bench_function("rerun_identity_w32_k8_n10k", |b| {
        b.iter(|| {
            for (q, r1) in qs.iter().zip(&first) {
                let r2 = walk_rank(&view, black_box(q), &defaults);
                assert_eq!(r1, &r2, "G1 must hold on EVERY timed iteration");
                black_box(&r2);
            }
        })
    });
    // Stage-2 on the same fixture (the context builder's cost is the ranked
    // list × budget; defaults 64 nodes / depth 2).
    let budget = ContextBudget::default();
    group.bench_function("budgeted_context_defaults_n10k", |b| {
        b.iter(|| {
            for (q, ranked) in qs.iter().zip(&first) {
                let ctx = budgeted_context(&view, ranked, black_box(q), &budget);
                black_box(&ctx);
            }
        })
    });
    group.finish();
}

criterion_group!(benches, g2_latency, g1_determinism);
criterion_main!(benches);
