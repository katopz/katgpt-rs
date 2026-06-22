//! Benchmark: RTDC Phase 3 Candidate C — `subtree_inclusion` proof (Plan 302, Issue 002).
//!
//! Measures latency for `prove_subtree_inclusion` + `verify_subtree_inclusion`
//! against the depth-2 `verify_at_depth` baseline. The CG6 cost gate is:
//!
//!   verify_subtree_inclusion cost ≤ 5.0× depth-2 verify cost
//!
//! At K = RTDC_SUBTREE_DEFAULT_K = 8 the theoretical ratio is exactly 5.0×
//! (10 BLAKE3 finalize calls vs 2 for depth-2 verify). The bench also runs
//! K = 23 (95% catch probability at f = 1/8) as the upper-bound reference.
//!
//! All benchmarks behind the `rtdc_subtree_inclusion` feature gate.

#![cfg(feature = "rtdc_subtree_inclusion")]

use criterion::{Criterion, criterion_group, criterion_main};
use katgpt_core::merkle::{HASH_SIZE, MERKLE_OCTREE_LEAVES, MerkleOctree};
use katgpt_core::rtdc::{DepthTieredMerkleOctree, DepthTieredRoots, RTDC_SUBTREE_DEFAULT_K};
use katgpt_core::slod::ScaleBoundary;

// ── Setup helpers (mirrors rtdc.rs::tests::populated_octree / dummy_boundaries) ──

fn generate_leaf_hashes() -> [[u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES] {
    let mut leaves = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
    for i in 0..MERKLE_OCTREE_LEAVES {
        let mut buf = [0u8; 32];
        buf[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        leaves[i] = *blake3::hash(&buf).as_bytes();
    }
    leaves
}

fn dummy_boundaries() -> Vec<ScaleBoundary> {
    vec![
        ScaleBoundary {
            sigma: 0.5,
            score: 1.0,
            k_star: 8,
        },
        ScaleBoundary {
            sigma: 2.0,
            score: 0.5,
            k_star: 2,
        },
    ]
}

fn build_tree() -> DepthTieredMerkleOctree {
    let leaves = generate_leaf_hashes();
    let octree = MerkleOctree::build_from_leaves(&leaves);
    DepthTieredMerkleOctree::build(octree, &dummy_boundaries())
        .expect("dummy boundaries are well-formed")
}

// K used for the 95%-at-f=1/8 reference point (Issue 002 estimate).
const K_95PCT_F_1_8: usize = 23;

// ── Benches ──────────────────────────────────────────────────────────

fn bench_build_depth_tiered(c: &mut Criterion) {
    let leaves = generate_leaf_hashes();
    c.bench_function("rtdc_build_depth_tiered_from_leaves", |b| {
        b.iter(|| {
            let octree = MerkleOctree::build_from_leaves(&leaves);
            let tree =
                DepthTieredMerkleOctree::build(octree, &dummy_boundaries()).expect("build ok");
            std::hint::black_box(&tree);
        });
    });
}

fn bench_prove_subtree(c: &mut Criterion) {
    let tree = build_tree();

    c.bench_function("rtdc_prove_subtree_0_2_k8", |b| {
        b.iter(|| {
            let p = tree
                .prove_subtree_inclusion(0, 2, 0xC0FFEE, RTDC_SUBTREE_DEFAULT_K)
                .expect("prove ok");
            std::hint::black_box(&p);
        });
    });

    c.bench_function("rtdc_prove_subtree_1_2_k8", |b| {
        b.iter(|| {
            let p = tree
                .prove_subtree_inclusion(1, 2, 0xDEADBEEF, RTDC_SUBTREE_DEFAULT_K)
                .expect("prove ok");
            std::hint::black_box(&p);
        });
    });
}

fn bench_verify_subtree(c: &mut Criterion) {
    let tree = build_tree();
    let roots: DepthTieredRoots = *tree.roots();

    let proof_0_2_k8 = tree
        .prove_subtree_inclusion(0, 2, 0xC0FFEE, RTDC_SUBTREE_DEFAULT_K)
        .expect("prove ok");
    let proof_1_2_k8 = tree
        .prove_subtree_inclusion(1, 2, 0xDEADBEEF, RTDC_SUBTREE_DEFAULT_K)
        .expect("prove ok");
    let proof_0_2_k23 = tree
        .prove_subtree_inclusion(0, 2, 0x95CA_5432, K_95PCT_F_1_8)
        .expect("prove ok");

    // CG6 cost gate — main number reported in .benchmarks/303_*.
    c.bench_function("rtdc_verify_subtree_0_2_k8", |b| {
        b.iter(|| {
            let ok = DepthTieredMerkleOctree::verify_subtree_inclusion(&proof_0_2_k8, &roots);
            std::hint::black_box(ok);
        });
    });

    c.bench_function("rtdc_verify_subtree_1_2_k8", |b| {
        b.iter(|| {
            let ok = DepthTieredMerkleOctree::verify_subtree_inclusion(&proof_1_2_k8, &roots);
            std::hint::black_box(ok);
        });
    });

    // Upper-bound reference: 95% catch probability at f=1/8. Exceeds the CG6
    // cost gate (12.5× theoretical) but documents the cost/confidence tradeoff.
    c.bench_function("rtdc_verify_subtree_0_2_k23_95pct_f1_8", |b| {
        b.iter(|| {
            let ok = DepthTieredMerkleOctree::verify_subtree_inclusion(&proof_0_2_k23, &roots);
            std::hint::black_box(ok);
        });
    });
}

fn bench_verify_depth2_baseline(c: &mut Criterion) {
    // The denominator of the CG6 cost ratio. Uses the existing Phase 1
    // verify_at_depth(d=2) path — same code path a depth-2 light client
    // would invoke.
    let tree = build_tree();
    let roots: DepthTieredRoots = *tree.roots();
    let proof_depth2 = tree.prove_at_depth(0, 2).expect("prove_at_depth ok");

    c.bench_function("rtdc_verify_depth2_baseline", |b| {
        b.iter(|| {
            let ok = DepthTieredMerkleOctree::verify_at_depth(&proof_depth2, &roots);
            std::hint::black_box(ok);
        });
    });
}

criterion_group!(
    benches,
    bench_build_depth_tiered,
    bench_prove_subtree,
    bench_verify_subtree,
    bench_verify_depth2_baseline,
);

criterion_main!(benches);
