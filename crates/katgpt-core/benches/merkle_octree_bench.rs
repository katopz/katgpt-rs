//! Benchmark: Merkle Octree Curator Consensus (Plan 253).
//!
//! Measures latency for Merkle build, proof gen/verify, curator verification, and bandit.
//! All benchmarks behind `merkle_octree` feature gate.
//!
//! GOAT targets:
//!   - Merkle build (64 leaves):     < 5µs
//!   - Proof generate:               < 1µs
//!   - Proof verify:                 < 1µs
//!   - Curator verify module:        < 2µs
//!   - Bandit sample + update:       < 100ns
//!   - Merkle build from embeddings: < 5µs

#![cfg(feature = "merkle_octree")]

use criterion::{Criterion, criterion_group, criterion_main};
use katgpt_core::curator::{CuratorBandit, CuratorVerifier};
use katgpt_core::merkle::{HASH_SIZE, MERKLE_OCTREE_LEAVES, MerkleOctree, MerkleProof};
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::types::{SenseKind, TernaryDir};

// ── Setup helpers ────────────────────────────────────────────────

/// Generate 64 distinct pre-hashed leaves.
fn generate_leaf_hashes() -> [[u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES] {
    let mut leaves = [[0u8; HASH_SIZE]; MERKLE_OCTREE_LEAVES];
    for i in 0..MERKLE_OCTREE_LEAVES {
        let mut buf = [0u8; 32];
        buf[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        leaves[i] = *blake3::hash(&buf).as_bytes();
    }
    leaves
}

/// Generate 64 KG embeddings for benchmarking.
fn generate_embeddings() -> Vec<KgEmbedding> {
    (0..MERKLE_OCTREE_LEAVES)
        .map(|i| KgEmbedding {
            entity_hash: i as u64,
            relation_hash: (i as u64).wrapping_add(1000),
            embedding: [
                i as f32 * 0.01,
                (i as f32 + 1.0) * 0.02,
                (i as f32 + 2.0) * 0.03,
                (i as f32 + 3.0) * 0.04,
                (i as f32 + 4.0) * 0.05,
                (i as f32 + 5.0) * 0.06,
                (i as f32 + 6.0) * 0.07,
                (i as f32 + 7.0) * 0.08,
            ],
            confidence: 0.5 + (i as f32 / MERKLE_OCTREE_LEAVES as f32) * 0.5,
            sign: i % 2 == 0,
        })
        .collect()
}

/// Build a SenseModule suitable for curator verification.
fn make_test_module() -> katgpt_core::types::SenseModule {
    let mut dirs = [TernaryDir::zero(); 8];
    dirs[0] = TernaryDir {
        pos_bits: 0b011,
        neg_bits: 0b100,
        row_scale: 1.0,
    };
    katgpt_core::types::SenseModule {
        octree_bits: [0; 4],
        directions: dirs,
        confidence: 0.9,
        kind: SenseKind::SpatialSense,
        version: 1,
        octree_depth: 3,
        n_directions: 1,
        _reserved: 0,
        commitment: [0u8; 32],
    }
}

// ── Benchmarks ───────────────────────────────────────────────────

fn bench_merkle_build(c: &mut Criterion) {
    let leaf_hashes = generate_leaf_hashes();

    c.bench_function("merkle_build_from_leaves", |b| {
        b.iter(|| {
            let tree = MerkleOctree::build_from_leaves(&leaf_hashes);
            std::hint::black_box(&tree);
        });
    });
}

fn bench_merkle_proof_generate(c: &mut Criterion) {
    let leaf_hashes = generate_leaf_hashes();
    let tree = MerkleOctree::build_from_leaves(&leaf_hashes);

    c.bench_function("merkle_proof_generate_leaf0", |b| {
        b.iter(|| {
            let proof = MerkleProof::generate(&tree, 0);
            std::hint::black_box(&proof);
        });
    });
}

fn bench_merkle_proof_verify(c: &mut Criterion) {
    let leaf_hashes = generate_leaf_hashes();
    let tree = MerkleOctree::build_from_leaves(&leaf_hashes);
    let proof = MerkleProof::generate(&tree, 0).expect("proof for leaf 0");
    let root = *tree.root();

    c.bench_function("merkle_proof_verify_leaf0", |b| {
        b.iter(|| {
            let valid = proof.verify(&root);
            std::hint::black_box(valid);
        });
    });
}

fn bench_curator_verify_module(c: &mut Criterion) {
    let leaf_hashes = generate_leaf_hashes();
    let tree = MerkleOctree::build_from_leaves(&leaf_hashes);
    let verifier = CuratorVerifier::new();
    let module = make_test_module();

    c.bench_function("curator_verify_module", |b| {
        b.iter(|| {
            let verdict = verifier.verify_module(&module, &tree);
            std::hint::black_box(&verdict);
        });
    });
}

fn bench_curator_bandit_sample_update(c: &mut Criterion) {
    let mut bandit = CuratorBandit::new(8);

    c.bench_function("curator_bandit_sample_update", |b| {
        b.iter(|| {
            let weight = bandit.sample(0);
            bandit.update(0, weight > 0.5);
            std::hint::black_box(weight);
        });
    });
}

fn bench_merkle_build_from_embeddings(c: &mut Criterion) {
    let embeddings = generate_embeddings();
    let builder = SenseOctreeBuilder::new(3);

    c.bench_function("merkle_build_from_64_embeddings", |b| {
        b.iter(|| {
            let (_module, tree) = builder.build_with_merkle(SenseKind::SpatialSense, &embeddings);
            std::hint::black_box(&tree);
        });
    });
}

criterion_group!(
    benches,
    bench_merkle_build,
    bench_merkle_proof_generate,
    bench_merkle_proof_verify,
    bench_curator_verify_module,
    bench_curator_bandit_sample_update,
    bench_merkle_build_from_embeddings,
);

criterion_main!(benches);
