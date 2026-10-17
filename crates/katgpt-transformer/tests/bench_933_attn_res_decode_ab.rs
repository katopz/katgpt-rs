//! Issue 933 — `apply_attn_res` decode: push-time `Σ v²` cache + zipped
//! accumulate + prepared score weight, against the pre-933 kernel.
//!
//! Arms (all three compute the same function):
//! - **ref** — the pre-933 `apply_attn_res`, frozen verbatim below: re-folds
//!   `score_weight` and re-reduces every summary's `Σ v²` on every call.
//! - **live** — today's `apply_attn_res` (re-folds `score_weight`, reads the
//!   cached `Σ v²`).
//! - **prepared** — `apply_attn_res_prepared` (folded once, cached `Σ v²`).
//!
//! The parity tests run by default and pin **bit-identity** (same reductions,
//! same summation order), which is stronger than the issue's 1e-6 bar. The
//! timing A/B is `#[ignore]` (wall clock; run alone, `--release`):
//!
//! ```bash
//! cargo test -p katgpt-transformer --features transformer_attn_res \
//!   --test bench_933_attn_res_decode_ab --release -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Instrument: the shared Issue-723 interleaved median-of-ratios with
//! per-iteration input churn and the output consumed on every arm (fat-LTO
//! anti-hoist; a vanished arm is a named instrument failure).
#![cfg(feature = "transformer_attn_res")]

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use std::hint::black_box;

use katgpt_core::simd::{simd_dot_f32, simd_sum_sq};
use katgpt_core::softmax;
use katgpt_transformer::attn_res::{
    AttnResBlockState, AttnResConfig, AttnResScratch, AttnResWeights, apply_attn_res,
    apply_attn_res_prepared,
};

/// The pre-933 kernel, verbatim (score_weight re-fold + per-call `Σ v²`).
fn apply_attn_res_ref<'a>(
    config: &AttnResConfig,
    weights: &AttnResWeights,
    residuals: &[Vec<f32>],
    scratch: &'a mut AttnResScratch,
    prefix_sum: &[f32],
) -> &'a mut [f32] {
    let d = config.d();
    scratch.recompute_score_weight(weights);
    let num_entries = residuals.len() + 1;
    let scores = &mut scratch.scores[..num_entries];
    let eps = config.rms_eps;
    for (i, residual) in residuals.iter().enumerate() {
        let sum_sq = simd_sum_sq(residual, d);
        let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
        let raw_dot = simd_dot_f32(residual, &scratch.score_weight, d);
        scores[i] = inv_rms * raw_dot;
    }
    {
        let sum_sq = simd_sum_sq(prefix_sum, d);
        let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
        let raw_dot = simd_dot_f32(prefix_sum, &scratch.score_weight, d);
        scores[num_entries - 1] = inv_rms * raw_dot;
    }
    softmax(scores);
    let out = &mut scratch.out[..d];
    for x in out.iter_mut() {
        *x = 0.0;
    }
    for (i, residual) in residuals.iter().enumerate() {
        let prob = scores[i];
        for (j, &rv) in residual.iter().enumerate() {
            out[j] += prob * rv;
        }
    }
    let prob = scores[num_entries - 1];
    for (j, &pv) in prefix_sum.iter().enumerate() {
        out[j] += prob * pv;
    }
    out
}

/// Deterministic splitmix64-derived values in [-1, 1).
fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

struct Fixture {
    config: AttnResConfig,
    weights: AttnResWeights,
    state: AttnResBlockState,
    prefixes: Vec<Vec<f32>>,
}

fn fixture(entries: usize) -> Fixture {
    let config = AttnResConfig::kimi_k3_0_40b();
    let d = config.d();
    // Learned-scale weights so the softmax is not near-uniform.
    let weights = AttnResWeights {
        norm_weight: fill(1, d),
        proj_weight: fill(2, d).into_iter().map(|w| w * 0.2).collect(),
    };
    let mut state = AttnResBlockState::new_with_capacity(d, entries);
    for e in 0..entries {
        // Distinct scales per summary so the RMS term matters.
        let scale = 0.5 + e as f32 * 0.37;
        let v: Vec<f32> = fill(100 + e as u64, d)
            .into_iter()
            .map(|x| x * scale)
            .collect();
        state.push(&v);
    }
    let prefixes = (0..16).map(|k| fill(1000 + k, d)).collect();
    Fixture {
        config,
        weights,
        state,
        prefixes,
    }
}

fn assert_bit_identical(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: out[{i}] {x} vs {y}");
    }
}

#[test]
fn live_and_prepared_are_bit_identical_to_the_pre_933_kernel() {
    for entries in [0usize, 1, 2, 8, 32] {
        let f = fixture(entries);
        let mut s_ref = AttnResScratch::new(&f.config, entries);
        let mut s_live = AttnResScratch::new(&f.config, entries);
        let mut s_prep = AttnResScratch::new(&f.config, entries);
        s_prep.recompute_score_weight(&f.weights);
        for p in &f.prefixes {
            let r = apply_attn_res_ref(&f.config, &f.weights, f.state.residuals(), &mut s_ref, p)
                .to_vec();
            let l = apply_attn_res(&f.config, &f.weights, &f.state, &mut s_live, p).to_vec();
            let q = apply_attn_res_prepared(&f.config, &f.state, &mut s_prep, p).to_vec();
            assert_bit_identical(&r, &l, &format!("live N={entries}"));
            assert_bit_identical(&r, &q, &format!("prepared N={entries}"));
        }
    }
}

/// Stale-view canary: the cache must follow every way a summary can change
/// (clear + re-push of different content, and pool slot reuse). A cache keyed
/// on anything but the pushed content would read the OLD `Σ v²` here.
#[test]
fn the_sum_sq_cache_follows_clear_and_repush() {
    let f = fixture(4);
    let d = f.config.d();
    let mut state = f.state.clone();
    state.clear();
    // Re-push different content into the recycled slots, at very different scales.
    for e in 0..4u64 {
        let v: Vec<f32> = fill(500 + e, d)
            .into_iter()
            .map(|x| x * (3.0 + e as f32))
            .collect();
        state.push(&v);
    }
    let mut s_ref = AttnResScratch::new(&f.config, 4);
    let mut s_live = AttnResScratch::new(&f.config, 4);
    let p = &f.prefixes[0];
    let r = apply_attn_res_ref(&f.config, &f.weights, state.residuals(), &mut s_ref, p).to_vec();
    let l = apply_attn_res(&f.config, &f.weights, &state, &mut s_live, p).to_vec();
    assert_bit_identical(&r, &l, "after clear+repush");
    // And the re-pushed content really moved the answer (the canary is live).
    let mut s_old = AttnResScratch::new(&f.config, 4);
    let old = apply_attn_res(&f.config, &f.weights, &f.state, &mut s_old, p).to_vec();
    assert!(
        old.iter().zip(&l).any(|(a, b)| a.to_bits() != b.to_bits()),
        "re-push changed nothing"
    );
}

#[test]
#[ignore = "wall-clock A/B; run alone with --release"]
fn bench_933_decode_ab() {
    for entries in [2usize, 8, 32] {
        let f = fixture(entries);
        let mut s_a = AttnResScratch::new(&f.config, entries);
        let mut s_b = AttnResScratch::new(&f.config, entries);
        let mut s_c = AttnResScratch::new(&f.config, entries);
        s_c.recompute_score_weight(&f.weights);
        let n = f.prefixes.len();
        let sink = std::cell::Cell::new(0.0f32);

        let live = ab_timing::ab_median_ratio(
            31,
            2000,
            500,
            |i| {
                let out = apply_attn_res_ref(
                    black_box(&f.config),
                    black_box(&f.weights),
                    black_box(f.state.residuals()),
                    &mut s_a,
                    black_box(&f.prefixes[i % n]),
                );
                sink.set(sink.get() + out[i % out.len()]);
            },
            |i| {
                let out = apply_attn_res(
                    black_box(&f.config),
                    black_box(&f.weights),
                    black_box(&f.state),
                    &mut s_b,
                    black_box(&f.prefixes[i % n]),
                );
                sink.set(sink.get() + out[i % out.len()]);
            },
        );
        live.report(&format!("N={entries} live/ref"));

        let prep = ab_timing::ab_median_ratio(
            31,
            2000,
            500,
            |i| {
                let out = apply_attn_res_ref(
                    black_box(&f.config),
                    black_box(&f.weights),
                    black_box(f.state.residuals()),
                    &mut s_a,
                    black_box(&f.prefixes[i % n]),
                );
                sink.set(sink.get() + out[i % out.len()]);
            },
            |i| {
                let out = apply_attn_res_prepared(
                    black_box(&f.config),
                    black_box(&f.state),
                    &mut s_c,
                    black_box(&f.prefixes[i % n]),
                );
                sink.set(sink.get() + out[i % out.len()]);
            },
        );
        prep.report(&format!("N={entries} prepared/ref"));
        black_box(sink.get());
    }
}
