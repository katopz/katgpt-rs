//! Issue 1229 G2 — latency of the Lepski k arm vs the fixed-K_max incumbent.
//!
//! Pre-registered (2026-11-20, katgpt-rs-f3, BEFORE the first run):
//!
//! - **Posture:** `--release`, `BlockTopKRouter` inner, head_dim 64, block 16
//!   keys, k band [4, 32] (the Plan-256 serving band), n_blocks ∈ {64, 256,
//!   1024}, 64 queries per n — half FOCUSED (a 4-block cliff), half SCATTERED
//!   (no structure).
//! - **Arms:** A = fixed-K_max (inner at k = 32), B = `LepskiKRouter`.
//!   Each call = route + sparse attention over the selected blocks (q·k for
//!   every key of every selected block, sigmoid-weighted value accumulation)
//!   — the serving posture the issue's "≤ fixed-K_max" expectation names.
//! - **Instrument:** interleaved median-of-ratios (`ab_timing`), B/A.
//! - **G2 PASS** iff the end-to-end median ratio B/A ≤ 1.00 at EVERY n.
//! - **Disclosed, NOT verdict inputs:** routing-only B/A (the arm's overhead —
//!   the bulk-MAD σ̂ is two O(n) selections, not "+log-n folds" as the issue
//!   text guessed), B vs the variance-sigmoid arm, mean k̂ per query class.
//!
//! Run: `cargo test --release -p katgpt-attn --features msa_adaptive_k_lepski
//!       --test issue_1229_lepski_g2_latency -- --nocapture`

#![cfg(feature = "msa_adaptive_k_lepski")]

use katgpt_attn::dash_attn::adaptive_k::{AdaptiveKConfig, AdaptiveKRouter};
use katgpt_attn::dash_attn::adaptive_k_lepski::{LepskiKConfig, LepskiKRouter};
use katgpt_attn::dash_attn::block_topk::{BlockTopKCache, BlockTopKRouter};
use katgpt_attn::dash_attn::vortex_flow::{VortexFlow, VortexScratch};
use std::hint::black_box;

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;
use ab_timing::ab_median_ratio;

const HD: usize = 64;
const BS: usize = 16;
const K_MIN: usize = 4;
const K_MAX: usize = 32;
const N_QUERIES: usize = 64;
const NS: [usize; 3] = [64, 256, 1024];

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

struct Fixture {
    cache: BlockTopKCache,
    keys: Vec<f32>,
    values: Vec<f32>,
    queries: Vec<Vec<f32>>,
}

fn fixture(n: usize) -> Fixture {
    let mut rng = Rng(0x1229_0002 ^ n as u64);
    let keys: Vec<f32> = (0..n * BS * HD).map(|_| rng.next_f32()).collect();
    let values: Vec<f32> = (0..n * BS * HD).map(|_| rng.next_f32()).collect();
    let router = BlockTopKRouter::new(true);
    let mut cache = router.cache_new(n, HD);
    for b in 0..n {
        let r = b * BS * HD..(b + 1) * BS * HD;
        router.forward_cache(&mut cache, &keys[r.clone()], &values[r], b, HD);
    }
    let mut queries = Vec::with_capacity(N_QUERIES);
    for q in 0..N_QUERIES {
        if q % 2 == 0 {
            // FOCUSED: a 4-block cliff — the sum of 4 centroids, amplified.
            let mut v = vec![0.0f32; HD];
            for t in 0..4 {
                let c = cache.centroid((q * 7 + t * 13) % n);
                for (o, &x) in v.iter_mut().zip(c) {
                    *o += 8.0 * x;
                }
            }
            queries.push(v);
        } else {
            // SCATTERED: unstructured direction.
            queries.push((0..HD).map(|_| rng.next_f32()).collect());
        }
    }
    Fixture {
        cache,
        keys,
        values,
        queries,
    }
}

/// Sparse attention over the selected blocks (sigmoid weights — never softmax).
fn attend(fx: &Fixture, q: &[f32], blocks: &[usize], out: &mut [f32; HD]) {
    out.fill(0.0);
    let scale = 1.0 / (HD as f32).sqrt();
    for &b in blocks {
        for t in 0..BS {
            let off = (b * BS + t) * HD;
            let k = &fx.keys[off..off + HD];
            let mut s = 0.0f32;
            for (a, c) in q.iter().zip(k) {
                s += a * c;
            }
            let w = 1.0 / (1.0 + (-s * scale).exp());
            for (o, &v) in out.iter_mut().zip(&fx.values[off..off + HD]) {
                *o += w * v;
            }
        }
    }
}

fn run<R: VortexFlow<Cache = BlockTopKCache>>(
    r: &R,
    fx: &Fixture,
    n: usize,
    i: usize,
    scratch: &mut VortexScratch,
    out: &mut [f32; HD],
    with_attention: bool,
) -> usize {
    let q = &fx.queries[i % N_QUERIES];
    let d = r.forward_indexer(q, &fx.cache, n, K_MAX, scratch);
    if with_attention {
        attend(fx, q, &d.blocks, out);
    }
    black_box(&out);
    d.blocks.len()
}

#[test]
fn g2_lepski_vs_fixed_k_max() {
    let lepski = LepskiKRouter::new(BlockTopKRouter::new(true), LepskiKConfig::new(K_MIN, K_MAX));
    let variance = AdaptiveKRouter::new(
        BlockTopKRouter::new(true),
        AdaptiveKConfig::new(K_MIN, K_MAX),
    );
    let fixed = BlockTopKRouter::new(true);
    let mut verdict = true;

    for n in NS {
        let fx = fixture(n);
        let (mut sa, mut sb) = (VortexScratch::new(n), VortexScratch::new(n));
        let (mut oa, mut ob) = ([0.0f32; HD], [0.0f32; HD]);

        // Mean k̂ per class (disclosed).
        let (mut kf, mut ks) = (0usize, 0usize);
        for i in 0..N_QUERIES {
            let k = run(&lepski, &fx, n, i, &mut sb, &mut ob, false);
            if i % 2 == 0 { kf += k } else { ks += k }
        }
        let half = (N_QUERIES / 2) as f64;

        let iters = (200_000 / n).max(64);
        let e2e = ab_median_ratio(
            31,
            iters,
            iters,
            |i| {
                black_box(run(&fixed, &fx, n, i, &mut sa, &mut oa, true));
            },
            |i| {
                black_box(run(&lepski, &fx, n, i, &mut sb, &mut ob, true));
            },
        );
        let route = ab_median_ratio(
            31,
            iters,
            iters,
            |i| {
                black_box(run(&fixed, &fx, n, i, &mut sa, &mut oa, false));
            },
            |i| {
                black_box(run(&lepski, &fx, n, i, &mut sb, &mut ob, false));
            },
        );
        let vs_var = ab_median_ratio(
            31,
            iters,
            iters,
            |i| {
                black_box(run(&variance, &fx, n, i, &mut sa, &mut oa, false));
            },
            |i| {
                black_box(run(&lepski, &fx, n, i, &mut sb, &mut ob, false));
            },
        );
        let pass = e2e.median <= 1.00;
        verdict &= pass;
        println!(
            "n={n:5}  e2e B/A {:.3} [{:.3}..{:.3}] {}  | route-only B/A {:.3}  B/variance {:.3}  | k̂ focused {:.1} scattered {:.1}",
            e2e.median,
            e2e.min(),
            e2e.max(),
            if pass { "PASS" } else { "FAIL" },
            route.median,
            vs_var.median,
            kf as f64 / half,
            ks as f64 / half,
        );
    }
    assert!(
        verdict,
        "G2 FAIL: end-to-end Lepski/fixed-K_max median ratio > 1.00 at some n"
    );
}
