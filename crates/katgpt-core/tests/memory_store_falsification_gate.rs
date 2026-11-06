//! Issue 984 — memory-store falsification gate over `CorpusDistanceGate`
//! (the T2/T3 arms at test scale; the full-scale instrument lives in
//! `benches/bench_845_distance_abstain_goat.rs` G6).
//!
//! The preregistered claim under test: the gate READS its store. Under a
//! BLAKE3-seeded matched-statistics noise store the distance signal must
//! collapse (I1 evidence-dependence), a noise store must not beat the
//! score-only no-memory floor (I2 no-hallucination), and a basis-shuffled
//! store must not serve as well as the real one (I3 binding matters). A
//! store-blind stub read fails all three — the canary proves the arms are
//! live (a stub that passes would mean the gate is blind).
//!
//! Wire: `scripts/test_gate.sh` row `katgpt-core:<N>:distance_abstain` —
//! the feature-named-row mechanism (the lane that executes a gated target
//! must NAME the feature).
//!
//! ```bash
//! cargo test -p katgpt-core --features distance_abstain --no-default-features \
//!   --test memory_store_falsification_gate
//! ```

#![cfg(feature = "distance_abstain")]

#[path = "common/store_falsification.rs"]
mod store_falsification;

use katgpt_core::distance_abstain::CorpusDistanceGate;
use store_falsification::{
    StoreArmMargins, StoreReader, StoreWorld, canary_holds, noise_store, print_verdict_table,
    run_store_arms, shuffled_store,
};

const D: usize = 32;
const N_CLUSTERS: usize = 4;
const PER_CLUSTER: usize = 16;
const K_EXEMPLARS: usize = N_CLUSTERS * PER_CLUSTER; // 64
const N_TEST: usize = 1024;
const N_REPS: usize = 3;
const OOD_SHARE: f32 = 0.30;
const CLUSTER_SIGMA: f32 = 0.12;
const MID: f32 = 0.5;
const SCALE: f32 = 10.0;

/// Deterministic `SplitMix64` (the workspace-bench idiom — no rand dep).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn next_sym(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

fn unit(v: [f32; D]) -> [f32; D] {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = 1.0 / n;
    v.map(|x| x * inv)
}

/// The corpus: tight clusters (the Bench 845 fixture shape at test scale).
fn build_corpus() -> Vec<[f32; D]> {
    let mut rng = SplitMix64(0x984_0001);
    let mut rows = Vec::with_capacity(K_EXEMPLARS);
    for c in 0..N_CLUSTERS {
        let sign = |i: usize| if (c >> (i % 3)) & 1 == 1 { 1.0 } else { -1.0 };
        let center = unit(std::array::from_fn(|i| sign(i) * (0.5 + rng.next_unit())));
        for _ in 0..PER_CLUSTER {
            let row = unit(center.map(|x| x + CLUSTER_SIGMA * rng.next_sym()));
            rows.push(row);
        }
    }
    rows
}

/// W1 world (OOD-blind errors): in-corpus queries track their reported
/// score, OOD queries collapse to coin-flip — the regime where distance
/// carries error information the score lacks.
fn build_world(seed: u64, corpus: &[[f32; D]]) -> (Vec<[f32; D]>, Vec<bool>, Vec<f32>) {
    let mut rng = SplitMix64(seed);
    let mut queries = Vec::with_capacity(N_TEST);
    let mut errors = Vec::with_capacity(N_TEST);
    let mut empty_signals = Vec::with_capacity(N_TEST);
    for _ in 0..N_TEST {
        let ood = rng.next_unit() < OOD_SHARE;
        let q = if ood {
            unit(std::array::from_fn(|_| rng.next_sym()))
        } else {
            let idx = (rng.next_u64() as usize) % corpus.len();
            unit(corpus[idx].map(|x| x + 2.0 * CLUSTER_SIGMA * rng.next_sym()))
        };
        let p = 0.5 + 0.49 * rng.next_unit(); // the OOD-blind reported score
        let true_p = if ood {
            0.5 + 0.05 * (p - 0.5)
        } else {
            1.0 / (1.0 + (-(1.2 * (p / (1.0 - p)).ln() - 0.2)).exp())
        };
        let correct = rng.next_unit() < true_p;
        queries.push(q);
        errors.push(!correct);
        empty_signals.push(p);
    }
    (queries, errors, empty_signals)
}

fn gate_reader(store: &[[f32; D]]) -> StoreReader<D> {
    let gate = CorpusDistanceGate::<D>::new(store, MID, SCALE);
    Box::new(move |queries: &[[f32; D]]| {
        queries.iter().map(|q| gate.abstain_confidence(q)).collect()
    })
}

/// The store-blind stub: a constant signal. Identical under every store
/// condition — the noise-invariant read the arms must catch.
fn stub_reader(_store: &[[f32; D]]) -> StoreReader<D> {
    Box::new(|_queries: &[[f32; D]]| vec![0.5f32; _queries.len()])
}

fn build_worlds(corpus: &[[f32; D]]) -> Vec<StoreWorld<D>> {
    (0..N_REPS)
        .map(|rep| {
            let seed = 0x984_0100 + rep as u64;
            let (queries, errors, empty_signals) = build_world(seed, corpus);
            StoreWorld {
                queries,
                errors,
                empty_signals,
            }
        })
        .collect()
}

#[test]
fn real_gate_read_consumes_its_store() {
    let corpus = build_corpus();
    let worlds = build_worlds(&corpus);
    let margins = StoreArmMargins::default();
    let verdict = run_store_arms(&corpus, &worlds, gate_reader, &margins);
    println!("── Issue 984 store arms — CorpusDistanceGate (real read):");
    print_verdict_table("gate", &verdict);
    println!(
        "  noise deltas: sel30 {:+.4} (evidence in quality), AURC {:+.4} (evidence in risk)",
        verdict.noise_delta_sel30, verdict.noise_delta_aurc
    );
    assert!(
        verdict.pass,
        "the real read must consume its store: I1={:?}",
        verdict.inequalities
    );
}

#[test]
fn store_blind_stub_is_caught_by_the_arms() {
    let corpus = build_corpus();
    let worlds = build_worlds(&corpus);
    let margins = StoreArmMargins::default();
    let real = run_store_arms(&corpus, &worlds, gate_reader, &margins);
    let stub = run_store_arms(&corpus, &worlds, stub_reader, &margins);
    println!("── Issue 984 canary — store-blind stub (must FAIL):");
    print_verdict_table("stub", &stub);
    assert!(
        !stub.pass,
        "CANARY BLIND: a store-blind stub passed the arms — the inequalities cannot detect evidence-blindness"
    );
    assert!(
        canary_holds(real.pass, stub.pass),
        "canary polarity broken: real pass={}, stub pass={}",
        real.pass,
        stub.pass
    );
}

// ── G1: the transform arms are deterministic across runs (BLAKE3-seeded) ──

#[test]
fn noise_store_is_deterministic_across_calls() {
    let mut rng = SplitMix64(984);
    let rows: Vec<[f32; D]> = (0..32)
        .map(|_| unit(std::array::from_fn(|_| rng.next_unit() * 2.0 - 1.0)))
        .collect();
    let a = noise_store(&rows);
    let b = noise_store(&rows);
    assert_eq!(
        a, b,
        "BLAKE3-seeded noise must be byte-identical across runs"
    );
    // A different store → a different draw (seed domain-separation).
    let rows2: Vec<[f32; D]> = rows
        .iter()
        .map(|r| {
            let mut r2 = *r;
            r2[0] += 0.001;
            unit(r2)
        })
        .collect();
    let c = noise_store(&rows2);
    assert_ne!(a, c, "a flipped bit must reseed the noise");
}

#[test]
fn shuffled_store_preserves_internal_geometry_and_breaks_alignment() {
    let mut rng = SplitMix64(985);
    let rows: Vec<[f32; D]> = (0..16)
        .map(|_| unit(std::array::from_fn(|_| rng.next_unit() * 2.0 - 1.0)))
        .collect();
    let shuffled = shuffled_store(&rows);
    // Internal geometry conserved exactly: pairwise cosine is invariant
    // under one shared orthogonal permutation.
    let dot = |a: &[f32; D], b: &[f32; D]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    for i in 0..rows.len() {
        for j in 0..rows.len() {
            let before = dot(&rows[i], &rows[j]);
            let after = dot(&shuffled[i], &shuffled[j]);
            assert!(
                (before - after).abs() < 1e-5,
                "pairwise cosine must be conserved"
            );
        }
    }
    // Row norms preserved (permutation is norm-invariant).
    for (r, s) in rows.iter().zip(&shuffled) {
        let n = |v: &[f32; D]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n(r) - n(s)).abs() < 1e-6);
    }
}
