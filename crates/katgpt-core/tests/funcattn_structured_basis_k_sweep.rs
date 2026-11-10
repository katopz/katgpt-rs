//! Plan 332 — Phase 3 k-sweep for the principled structured basis constructors.
//!
//! The Phase 0 probe used k=8; Research 257 §5 item 5 explicitly flags
//! k=4..16 for the NPC regime as an open sweep that has never been run. Now
//! that we have principled bases (Plan 332 Phase 1), we can fill this gap.
//!
//! Sweeps k ∈ {4, 8, 16, 32} for each basis variant (random, DCT-log,
//! Haar-packet, hand-crafted) at τ=0.5. Reports cos(out, target) for each
//! and identifies the elbow where random-orthogonal catches up to principled.
//!
//! Hypothesis (Plan 332 T3.3): principled bases help most at small k where
//! random is rank-starved.
//!
//! Output is informational (`println!`); this test only asserts the sanity
//! floor (all variants produced finite output). The actual finding is written
//! to `.benchmarks/332_structured_basis_k_sweep.md` (T3.4) by the human /
//! agent running the test, NOT auto-generated.

use katgpt_core::funcattn::{
    FuncAttnBasis, FuncAttnConfig, FuncAttnScratch, funcattn_forward, make_dct_log_basis,
    make_haar_packet_basis,
};

const D: usize = 64;
const N: usize = 20;

// Shared Plan 332 basis-harness helpers — see `common/basis_harness.rs`.
// All 5 helpers (`lcg_next`, `l2_normalize`, `gram_schmidt_rows`, `cosine`,
// `random_orthonormal_w`) are bit-identical to `funcattn_structured_basis_g1.rs`.
#[path = "common/basis_harness.rs"]
mod basis_harness;
use basis_harness::{cosine, gram_schmidt_rows, l2_normalize, lcg_next, random_orthonormal_w};

/// Build the multi-scale input X (matches the Phase 0 probe exactly).
fn make_multiscale_x(seed: u64, n_scales: usize) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let freqs: Vec<f32> = (0..n_scales).map(|i| 0.3 + 0.7 * (i as f32)).collect();
    let mut x = vec![0.0f32; N * D];
    for i in 0..N {
        let t = i as f32;
        for j in 0..D {
            let mut v = 0.0f32;
            for (si, &f) in freqs.iter().enumerate() {
                let amp = 1.0 / (si + 1) as f32;
                v += amp * (f * (t + 0.1 * j as f32) * (si as f32 + 1.0)).sin();
            }
            x[i * D + j] = v + 0.05 * lcg_next(&mut s);
        }
    }
    let mut dirs = vec![0.0f32; n_scales * D];
    for si in 0..n_scales {
        let f = 0.3 + 0.7 * si as f32;
        for j in 0..D {
            dirs[si * D + j] = (f * 0.1 * j as f32 * (si as f32 + 1.0)).sin();
        }
        l2_normalize(&mut dirs[si * D..(si + 1) * D]);
    }
    (x, dirs)
}

// `random_orthonormal_w` lives in the `basis_harness` shared module.

/// Hand-crafted structured basis aligned to the known signal directions.
fn hand_crafted_w(signal_dirs: &[f32], n_scales: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut w = vec![0.0f32; k * D];
    for si in 0..n_scales.min(k) {
        w[si * D..(si + 1) * D].copy_from_slice(&signal_dirs[si * D..(si + 1) * D]);
    }
    let mut s = seed;
    for v in w[(n_scales.min(k)) * D..k * D].iter_mut() {
        *v = lcg_next(&mut s);
    }
    gram_schmidt_rows(&mut w, k, D);
    w
}

/// Identity `(d, d)` matrix.
fn identity_mat(d: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

#[test]
fn funcattn_structured_basis_k_sweep() {
    let (x, signal_dirs) = make_multiscale_x(42, 4);

    // Target: linear smoothing operator (same as Phase 2 GOAT gate).
    let mut y_target = vec![0.0f32; N * D];
    for i in 0..N {
        let prev = if i > 0 { i - 1 } else { 0 };
        let next = if i + 1 < N { i + 1 } else { N - 1 };
        for j in 0..D {
            y_target[i * D + j] =
                0.25 * x[prev * D + j] + 0.5 * x[i * D + j] + 0.25 * x[next * D + j];
        }
    }

    let w_q = random_orthonormal_w(999, D, D);
    let w_k = random_orthonormal_w(888, D, D);
    let w_v = identity_mat(D);
    let tau = 0.5f32;

    let run = |w_basis: &[f32], k: usize| -> f32 {
        let cfg = FuncAttnConfig {
            d: D,
            k,
            basis: FuncAttnBasis::Sigmoid,
            temperature: tau,
            alpha: 0.5,
            cholesky_jitter: 1e-6,
        };
        let mut scratch = FuncAttnScratch::new(N, D, k);
        let mut out = vec![0.0f32; N * D];
        funcattn_forward(
            &x,
            &x,
            w_basis,
            &w_q,
            &w_k,
            &w_v,
            &cfg,
            &mut scratch,
            &mut out,
        )
        .expect("forward");
        for v in &out {
            assert!(v.is_finite(), "non-finite forward output");
        }
        cosine(&out, &y_target)
    };

    println!("\n=== Plan 332 Phase 3 — k-sweep (d={D}, n={N}, τ={tau}) ===");
    println!(
        "{:>4}  {:>10}  {:>10}  {:>12}  {:>10}  {:>10}  {:>10}",
        "k", "random", "DCT-log", "Haar-packet", "hand-craft", "Δ(DCT-rand)", "Δ(Haar-r)"
    );

    let mut random_curve = Vec::new();
    let mut dct_curve = Vec::new();
    let mut haar_curve = Vec::new();
    let mut hand_curve = Vec::new();

    for &k in &[4usize, 8, 16, 32] {
        let w_rand = random_orthonormal_w(100, k, D);
        let w_hand = hand_crafted_w(&signal_dirs, 4, k, 200);
        let w_dct = make_dct_log_basis(k, D);
        let w_haar = make_haar_packet_basis(k, D);

        let cos_rand = run(&w_rand, k);
        let cos_hand = run(&w_hand, k);
        let cos_dct = run(&w_dct, k);
        let cos_haar = run(&w_haar, k);

        random_curve.push(cos_rand);
        hand_curve.push(cos_hand);
        dct_curve.push(cos_dct);
        haar_curve.push(cos_haar);

        println!(
            "{:>4}  {:>+10.4}  {:>+10.4}  {:>+12.4}  {:>+10.4}  {:>+10.4}  {:>+10.4}",
            k,
            cos_rand,
            cos_dct,
            cos_haar,
            cos_hand,
            cos_dct - cos_rand,
            cos_haar - cos_rand
        );
    }

    // Identify the elbow: smallest k at which random catches up to within
    // 0.02 of the better principled basis. Below that k, principled wins.
    println!("\n--- Elbow analysis (Plan 332 T3.3) ---");
    let ks = [4usize, 8, 16, 32];
    let mut found_elbow = false;
    for (idx, &k) in ks.iter().enumerate() {
        let best_principled = dct_curve[idx].max(haar_curve[idx]);
        let gap = best_principled - random_curve[idx];
        println!(
            "  k={k:>2}: best principled = {best_principled:+.4}, random = {:+.4}, gap = {gap:+.4}",
            random_curve[idx]
        );
        if !found_elbow && gap < 0.02 {
            println!("  ↑ elbow: at k={k}, random catches up to within 0.02 of principled.",);
            found_elbow = true;
        }
    }
    if !found_elbow {
        println!(
            "  no elbow found in k ∈ {{4, 8, 16, 32}} — principled beats random at all sizes."
        );
    }

    // Hypothesis check: principled should help MORE at small k.
    let small_k_gap = (dct_curve[0] - random_curve[0]).max(haar_curve[0] - random_curve[0]);
    let large_k_gap = (dct_curve[3] - random_curve[3]).max(haar_curve[3] - random_curve[3]);
    println!(
        "\nHypothesis (T3.3): principled helps more at small k. k=4 gap = {small_k_gap:+.4}, k=32 gap = {large_k_gap:+.4}. {}",
        if small_k_gap > large_k_gap {
            "CONFIRMED"
        } else {
            "REJECTED"
        }
    );

    println!(
        "\nNext step (T3.4): transcribe this output into .benchmarks/332_structured_basis_k_sweep.md"
    );
}
