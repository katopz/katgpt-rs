//! Issue 985 — soft-read gate tests (T3/T5): determinism replay, zero-alloc
//! read path, zero-gate ε semantics, boundedness at N ∈ {1, 16, 512}.
//! The QUALITY bar (≥ hard top-16 AND > empty) lives in
//! `benches/bench_985_soft_read_goat.rs` — this file pins the mechanics.
//!
//! ```bash
//! cargo test -p katgpt-core --features soft_read --no-default-features
//! --test soft_read_gates
//! ```

#![cfg(feature = "soft_read")]

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::soft_read::{SlotMode, SoftRead, SoftReadScratch};
use std::hint::black_box;
use std::sync::atomic::Ordering;

const D: usize = 32;
const K: usize = 16;

/// Deterministic `SplitMix64` (the workspace idiom — no rand dep).
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
    if n > 0.0 {
        let inv = 1.0 / n;
        v.map(|x| x * inv)
    } else {
        v
    }
}

/// Deterministic store: a few tight clusters + filler (the composition
/// fixture's geometry at test scale).
fn seeded_store(seed: u64, n: usize) -> Vec<[f32; D]> {
    let mut rng = SplitMix64(seed);
    let mut rows: Vec<[f32; D]> = Vec::with_capacity(n);
    for c in 0..4 {
        let center: [f32; D] =
            std::array::from_fn(|i| if ((i + c) / 2) % 2 == 0 { 1.0 } else { -1.0 });
        let cluster_n = (n / 3).clamp(1, 8);
        for _ in 0..cluster_n {
            rows.push(unit(center.map(|x| x + 0.1 * rng.next_sym())));
        }
    }
    while rows.len() < n {
        rows.push(unit(std::array::from_fn(|_| rng.next_sym())));
    }
    rows
}

#[test]
fn read_path_is_zero_alloc_in_steady_state() {
    let rows = seeded_store(985, 64);
    let soft = SoftRead::<D, K>::new(&rows);
    let mut scratch = SoftReadScratch::<D, K>::new();
    let q = rows[0];
    // One warm call (fills the scratch; steady state starts after it).
    let _ = soft.read_into(&q, &rows, &mut scratch);
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let mut sink = 0.0f32;
    for r in rows.iter() {
        let reads = soft.read_into(r, &rows, &mut scratch);
        sink += SoftRead::<D, K>::best_answer_cosine(&reads, &q);
    }
    black_box(sink);
    let delta = ALLOC_COUNT.load(Ordering::Relaxed).saturating_sub(before);
    assert_eq!(delta, 0, "steady-state read_into allocated {delta} times");
}

#[test]
fn store_size_arm_bounded_and_bar_direction_at_n_1_16_512() {
    for n in [1usize, 16, 512] {
        let rows = seeded_store(985 + n as u64, n);
        let soft = SoftRead::<D, K>::new(&rows);
        let mut scratch = SoftReadScratch::new();
        let q = unit(rows[0]);
        let reads = soft.read_into(&q, &rows, &mut scratch);
        for (k, m) in reads.iter().enumerate() {
            let mag: f32 = m.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                mag.is_finite() && mag <= 1.0 + 1e-3,
                "N={n} slot {k}: read magnitude {mag} unbounded"
            );
        }
        // At N=1 the single row is trivially readable by some slot view.
        if n == 1 {
            let best = SoftRead::<D, K>::best_answer_cosine(&reads, &rows[0]);
            assert!(
                best > 0.9,
                "N=1: the only row must be readable (best cos {best:.3})"
            );
        }
    }
}

#[test]
fn zero_gate_biases_collapse_the_read() {
    let rows = seeded_store(986, 64);
    let soft = SoftRead::<D, K>::new(&rows).with_biases([-50.0; K]);
    let mut scratch = SoftReadScratch::new();
    let reads = soft.read_into(&rows[0], &rows, &mut scratch);
    for (k, m) in reads.iter().enumerate() {
        let mag: f32 = m.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(mag < 1e-4, "slot {k}: zero-gate read |M|={mag} not ≈ 0");
    }
}

#[test]
fn replay_bit_identical_and_append_rekeys() {
    let rows = seeded_store(987, 64);
    let a = SoftRead::<D, K>::new(&rows);
    let b = SoftRead::<D, K>::new(&rows);
    let mut s1 = SoftReadScratch::new();
    let mut s2 = SoftReadScratch::new();
    let q = rows[3];
    let r1 = a.read_into(&q, &rows, &mut s1);
    let r2 = b.read_into(&q, &rows, &mut s2);
    assert_eq!(r1, r2, "(store, query) replay must be bit-identical");
    // Slot bank keyed to the store: an append rekeys (fresh bank differs).
    let mut rows2 = rows.clone();
    rows2.push(unit(rows[0].map(|x| x + 0.01)));
    let c = SoftRead::<D, K>::new(&rows2);
    let r3 = c.read_into(&q, &rows2, &mut s1);
    assert_ne!(r1[0], r3[0], "append must rekey the bank");
    // Orthonormal mode is a distinct, valid posture.
    let o = SoftRead::<D, K>::with_mode(&rows, SlotMode::Orthonormal);
    let r4 = o.read_into(&q, &rows, &mut s2);
    assert!(r4.iter().all(|m| m.iter().all(|x| x.is_finite())));
}
