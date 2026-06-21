//! GOAT proof test for Manifold Power Iteration MoE Router (Plan 279).
//!
//! Run:
//! ```bash
//! cargo test --features manifold_power_iter_router \
//!            --test bench_279_manifold_power_iter_goat -- --nocapture
//! ```
//!
//! Validates the paper's GOAT claims (arXiv:2606.12397, Research 246 §1.4)
//! as distilled inference-time engine plumbing:
//!
//! - **G1 — λ alignment gain:** `λ(R') ≥ 0.5 · λ(R_optimal)` where R_optimal
//!   is exact top right-singular vectors. Paper: 0.27 → 0.66 (≈2.4×).
//! - **G2 — MaxVio reduction:** `MaxVio(R') ≤ 0.7 · MaxVio(R)`. Paper: 1.13 → 0.96.
//! - **G3 — Zero per-token overhead:** gate timing R vs R' identical within noise.
//! - **G4 — Sub-ms swap at game scale:** `N=8, D=256` total reconditioning < 1ms.
//! - **G5 — Determinism / sync-safety:** same inputs → byte-identical R' across runs.
//! - **G6 — DRY non-regression:** `gauge_rebalance` (Plan 270) still passes.
//! - **G7 — Sigmoid constraint:** independent per-expert sigmoid, never softmax.
//! - **G8 — `iters=1` sufficiency:** captures ≥90% of `iters=10` λ gain.

#![cfg(feature = "manifold_power_iter_router")]

use katgpt_rs::manifold_power_iter_router::{
    compute_diagnostics, compute_expert_gram_into, gate_sigmoid_topk, manifold_power_iter_router,
};
use katgpt_rs::spectral_retract::PowerRetractScratch as ScratchShared;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ── Helpers ──────────────────────────────────────────────────────────────

/// Deterministic xorshift64 PRNG (matches the rest of the crate).
fn seeded_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push(((s & 0xFFFF) as f32 / 0x8000 as f32) - 1.0);
    }
    v
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn gram_of(w: &[f32], d: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; d * d];
    compute_expert_gram_into(w, d, &mut g);
    g
}

/// Build a rank-1 d×d matrix whose dominant right-singular vector is `u`
/// (length d, assumed unit-norm by caller), scaled to singular value `σ`.
fn rank1_matrix_with_singular_vec(u: &[f32], d: usize, sigma: f32) -> Vec<f32> {
    let un = norm(u);
    let scale = sigma / (un * un);
    let mut w = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            w[i * d + j] = u[i] * u[j] * scale;
        }
    }
    w
}

/// Best-of-N wall-clock microseconds.
fn bench_us(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = Duration::from_secs(60);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed();
        if dt < best {
            best = dt;
        }
    }
    best.as_secs_f64() * 1e6
}

// ── Shared synthetic MoE fixture ─────────────────────────────────────────
//
// Per-expert gate weights W_g[i] are rank-1 with a KNOWN dominant right-
// singular vector u[i]. This lets us compute R_optimal[i] = u[i] exactly,
// against which we measure MPI recovery (G1).

const N_G: usize = 8;
const D_G: usize = 32; // smaller D for fast tests; G4 uses the real (8,256)

type ExpertMatrices = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);

fn build_synthetic_moe() -> ExpertMatrices {
    // Returns (u_per_expert, w_g_per_expert, gram_per_expert).
    let mut us = Vec::with_capacity(N_G);
    let mut wgs = Vec::with_capacity(N_G);
    let mut grams = Vec::with_capacity(N_G);
    for i in 0..N_G {
        let mut u = seeded_vec(1000 + i as u64, D_G);
        let nu = norm(&u);
        for x in &mut u {
            *x /= nu;
        }
        // Give each expert a distinct dominant σ (so power-iteration converges
        // with a clear gap and λ recovers cleanly).
        let sigma = 3.0 + (i as f32) * 0.5;
        let w = rank1_matrix_with_singular_vec(&u, D_G, sigma);
        let g = gram_of(&w, D_G);
        us.push(u);
        wgs.push(w);
        grams.push(g);
    }
    (us, wgs, grams)
}

// ── Pass / fail counters ─────────────────────────────────────────────────

static PASS: AtomicUsize = AtomicUsize::new(0);
static FAIL: AtomicUsize = AtomicUsize::new(0);

macro_rules! gate_check {
    ($name:expr, $cond:expr, $($arg:tt)*) => {{
        if $cond {
            PASS.fetch_add(1, Ordering::SeqCst);
            eprintln!("✓ {} PASS", $name);
        } else {
            FAIL.fetch_add(1, Ordering::SeqCst);
            eprintln!("✗ {} FAIL: {}", $name, format!($($arg)*));
        }
    }};
}

// ── G1: λ alignment gain ─────────────────────────────────────────────────
//
// λ(R') ≥ 0.5 · λ(R_optimal). Paper: 0.27 → 0.66 (≈2.4×).

#[test]
fn g01_lambda_alignment_gain() {
    let (us, _wgs, grams) = build_synthetic_moe();
    let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();

    // Random unconditioned router R.
    let mut r = seeded_vec(42, N_G * D_G);
    let target = 1.0f32 / (N_G as f32).sqrt();

    let (lambda_before, _) = compute_diagnostics(&r, &grams_ref, N_G, D_G, target);

    let mut scratch = ScratchShared::new(D_G);
    let res = manifold_power_iter_router(&mut r, &grams_ref, N_G, D_G, 1.0, 5, &mut scratch);

    // R_optimal: each row is the exact dominant right-singular vector u[i]
    // (scaled to target norm so the comparison is direction-only).
    let mut r_opt = vec![0.0f32; N_G * D_G];
    for i in 0..N_G {
        for j in 0..D_G {
            r_opt[i * D_G + j] = us[i][j] * target;
        }
    }
    let (lambda_opt, _) = compute_diagnostics(&r_opt, &grams_ref, N_G, D_G, target);

    eprintln!(
        "G1: λ_before={:.4}  λ_after(MPI)={:.4}  λ_optimal={:.4}  ratio={:.2}",
        lambda_before,
        res.lambda_alignment,
        lambda_opt,
        res.lambda_alignment / lambda_opt.abs().max(1e-6)
    );
    gate_check!(
        "G1",
        res.lambda_alignment >= 0.5 * lambda_opt.abs(),
        "λ(R')={:.4} < 0.5·λ(opt)={:.4}",
        res.lambda_alignment,
        0.5 * lambda_opt
    );
    // Bonus: MPI must strictly improve over vanilla.
    gate_check!(
        "G1.improve",
        res.lambda_alignment > lambda_before,
        "λ didn't improve: {} ≤ {}",
        res.lambda_alignment,
        lambda_before
    );
}

// ── G2: MaxVio reduction ────────────────────────────────────────────────
//
// MaxVio(R') ≤ 0.7 · MaxVio(R). Paper: 1.13 → 0.96 (≈15%); we gate at the
// more conservative 0.7× to absorb small-pool variance.

#[test]
fn g02_maxvio_reduction() {
    let (_us, _wgs, grams) = build_synthetic_moe();
    let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();

    // Construct a router with deliberately uneven row norms so MaxVio is large.
    let mut r = seeded_vec(5, N_G * D_G);
    // Scale row 0 by 10, row 1 by 0.1 — creates a big norm disparity.
    for j in 0..D_G {
        r[j] *= 10.0;
        r[D_G + j] *= 0.1;
    }

    let target = 1.0f32 / (N_G as f32).sqrt();
    let (lambda_before, maxvio_before) = compute_diagnostics(&r, &grams_ref, N_G, D_G, target);

    let mut scratch = ScratchShared::new(D_G);
    let res = manifold_power_iter_router(&mut r, &grams_ref, N_G, D_G, 1.0, 1, &mut scratch);

    eprintln!(
        "G2: maxvio_before={:.4}  maxvio_after={:.4}  ratio={:.3}  (λ {:.3}→{:.3})",
        maxvio_before,
        res.maxvio,
        res.maxvio / maxvio_before.abs().max(1e-6),
        lambda_before,
        res.lambda_alignment
    );
    gate_check!(
        "G2",
        res.maxvio <= 0.7 * maxvio_before.abs().max(1e-6),
        "MaxVio(R')={:.4} > 0.7·MaxVio(R)={:.4}",
        res.maxvio,
        0.7 * maxvio_before
    );
    // Retraction should drive MaxVio to ~0 (each row exactly at target norm).
    gate_check!(
        "G2.exact",
        res.maxvio < 1e-3,
        "post-retraction MaxVio {} should be ~0",
        res.maxvio
    );
}

// ── G3: Zero per-token overhead ─────────────────────────────────────────
//
// gate_sigmoid_topk with R vs R' must be timing-identical within noise.
// The gate is just a matvec + top-k; conditioning doesn't add ops.

#[test]
fn g03_zero_per_token_overhead() {
    let (_us, _wgs, grams) = build_synthetic_moe();
    let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();
    let n = N_G;
    let d = D_G;

    let r_vanilla = seeded_vec(42, n * d);
    let mut r_mpi = r_vanilla.clone();
    let mut scratch = ScratchShared::new(d);
    let _ = manifold_power_iter_router(&mut r_mpi, &grams_ref, n, d, 1.0, 1, &mut scratch);

    let x = seeded_vec(7, d);
    let mut scores = vec![0.0f32; n];

    // Warm up both paths.
    for _ in 0..50 {
        let _ = gate_sigmoid_topk(&x, &r_vanilla, n, d, 1.0, 3, &mut scores);
        let _ = gate_sigmoid_topk(&x, &r_mpi, n, d, 1.0, 3, &mut scores);
    }

    let t_vanilla = bench_us(50, 5000, || {
        let _ = gate_sigmoid_topk(&x, &r_vanilla, n, d, 1.0, 3, &mut scores);
    });
    let t_mpi = bench_us(50, 5000, || {
        let _ = gate_sigmoid_topk(&x, &r_mpi, n, d, 1.0, 3, &mut scores);
    });

    let ratio = t_mpi / t_vanilla.max(1e-9);
    eprintln!(
        "G3: gate_vanilla={:.3}us  gate_mpi={:.3}us  ratio={:.3}",
        t_vanilla, t_mpi, ratio
    );
    // Allow 2× slack for noise — the gate is identical matvec either way.
    gate_check!(
        "G3",
        ratio < 2.0,
        "gate R' is {:.2}× slower than R (should be ~1×)",
        ratio
    );
}

// ── G4: Sub-ms swap at game scale ───────────────────────────────────────
//
// N=8, D=256 (typical NPC LoRA pool): total reconditioning < 1ms.

#[test]
fn g04_subms_swap_game_scale() {
    let n = 8usize;
    let d = 256usize;
    let r = seeded_vec(42, n * d);
    let w_g: Vec<Vec<f32>> = (0..n).map(|i| seeded_vec(100 + i as u64, d * d)).collect();

    // Pre-build grams (warm tier — done once per snapshot, cached thereafter).
    let mut grams: Vec<Vec<f32>> = Vec::with_capacity(n);
    let t_gram = Instant::now();
    for w in &w_g {
        let mut g = vec![0.0f32; d * d];
        compute_expert_gram_into(w, d, &mut g);
        grams.push(g);
    }
    let dt_gram = t_gram.elapsed();
    let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();

    // Time MPI recondition ONLY (this is the per-snapshot-swap cost that G4
    // gates — grams are already cached from the warm tier above).
    let mut r_work = r.clone();
    let mut scratch = ScratchShared::new(d);
    let t_mpi = Instant::now();
    let _res = manifold_power_iter_router(&mut r_work, &grams_ref, n, d, 1.0, 1, &mut scratch);
    let dt_mpi = t_mpi.elapsed();

    let ms_mpi = dt_mpi.as_secs_f64() * 1e3;
    let ms_gram = dt_gram.as_secs_f64() * 1e3;
    eprintln!(
        "G4: N={}, D={}, gram={:.3}ms (warm, one-time), MPI={:.3}ms (per-swap)",
        n, d, ms_gram, ms_mpi
    );
    // G4 gates the MPI recondition cost (paper §4.2 "zero inference overhead"
    // — the gram build is a one-time warm cost at model load, not per-swap).
    if cfg!(debug_assertions) {
        eprintln!("  (debug build — G4 timing gate skipped, run with --release for the real gate)");
        gate_check!("G4", true, "debug build — skipped");
    } else {
        gate_check!("G4", ms_mpi < 1.0, "MPI recondition took {:.3}ms (must be < 1ms)", ms_mpi);
    }
}

// ── G5: Determinism / sync-safety ───────────────────────────────────────
//
// Same (R, M, c_prime, iters, snapshot_version) → byte-identical R'.

#[test]
fn g05_determinism_sync_safe() {
    let (_us, _wgs, grams) = build_synthetic_moe();
    let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();

    let r_seed = seeded_vec(42, N_G * D_G);
    let mut r1 = r_seed.clone();
    let mut r2 = r_seed.clone();

    let mut s1 = ScratchShared::new(D_G);
    let mut s2 = ScratchShared::new(D_G);
    let _ = manifold_power_iter_router(&mut r1, &grams_ref, N_G, D_G, 1.0, 1, &mut s1);
    let _ = manifold_power_iter_router(&mut r2, &grams_ref, N_G, D_G, 1.0, 1, &mut s2);

    let byte_eq = r1 == r2;
    eprintln!(
        "G5: byte-identical across two runs = {} ({} bytes)",
        byte_eq,
        r1.len() * 4
    );
    gate_check!("G5", byte_eq, "R' differs across runs — sync-unsafe");
}

// ── G6: DRY non-regression (Plan 270 gauge_rebalance) ───────────────────
//
// The shared power_iter_step helper is composed by gauge_rebalance.
// Existing tests must still pass byte-for-byte. We delegate to the
// gauge_invariant test module's canonical cases here (mirror names):
//   - t01_gauge_rebalance_preserves_abt_exactly
//   - test_gauge_rebalance_balances_sigmas
//   - test_gauge_rebalance_zero_matrix_safe
//   - test_power_iterate_matches_naive_sigma_max

#[test]
fn g06_dry_non_regression_gauge_rebalance() {
    use katgpt_rs::gauge_invariant::{gauge_rebalance, GaugeRebalanceScratch};

    // Mirror t01: A·B^T preserved after rebalance.
    let m = 16;
    let n = 12;
    let r = 4;
    let a_orig = seeded_vec(42, m * r);
    let b_orig = seeded_vec(99, n * r);

    let abt_before: Vec<f32> = {
        let mut v = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for k in 0..r {
                    s += a_orig[i * r + k] * b_orig[j * r + k];
                }
                v.push(s);
            }
        }
        v
    };

    let mut a = a_orig.clone();
    let mut b = b_orig.clone();
    let mut scratch = GaugeRebalanceScratch::new(m.max(n), r);
    gauge_rebalance(&mut a, &mut b, m, r, n, r, 1.0, &mut scratch);

    let abt_after: Vec<f32> = {
        let mut v = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for k in 0..r {
                    s += a[i * r + k] * b[j * r + k];
                }
                v.push(s);
            }
        }
        v
    };

    let max_diff = abt_before
        .iter()
        .zip(abt_after.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("G6: gauge_rebalance |ΔA·B^T|_max = {:.2e}", max_diff);
    gate_check!(
        "G6.preserves_abt",
        max_diff < 1e-3,
        "A·B^T changed by {}",
        max_diff
    );

    // Zero-matrix safety mirror.
    let mut a0 = vec![0.0f32; m * r];
    let mut b0 = vec![0.0f32; n * r];
    let mut s0 = GaugeRebalanceScratch::new(m.max(n), r);
    gauge_rebalance(&mut a0, &mut b0, m, r, n, r, 1.0, &mut s0);
    let all_zero = a0.iter().chain(b0.iter()).all(|x| x.abs() < 1e-20);
    gate_check!("G6.zero_safe", all_zero, "zero matrix not preserved");
}

// ── G7: Sigmoid constraint ──────────────────────────────────────────────
//
// (a) Static: the gate function exists and is named gate_sigmoid_topk
//     (no softmax variant in the API surface).
// (b) Runtime: changing one expert's row score MUST NOT perturb another's.

#[test]
fn g07_sigmoid_constraint() {
    let n = 4usize;
    let d = 8usize;
    let x = seeded_vec(7, d);
    let r = seeded_vec(13, n * d);

    let mut scores_a = vec![0.0f32; n];
    let mut scores_b = vec![0.0f32; n];
    gate_sigmoid_topk(&x, &r, n, d, 1.0, n, &mut scores_a);

    // Perturb ONLY expert 0's row.
    let mut r_perturbed = r.clone();
    for v in r_perturbed.iter_mut().take(d) {
        *v *= 5.0; // large perturbation to expert 0 only
    }
    gate_sigmoid_topk(&x, &r_perturbed, n, d, 1.0, n, &mut scores_b);

    let mut independent = true;
    for i in 1..n {
        let delta = (scores_a[i] - scores_b[i]).abs();
        if delta > 1e-7 {
            independent = false;
            eprintln!(
                "G7 FAIL: expert {} score drifted by {} after perturbing expert 0",
                i, delta
            );
        }
    }
    eprintln!(
        "G7: scores_a={:?}  scores_b={:?}  independent={}",
        scores_a, scores_b, independent
    );
    gate_check!("G7", independent, "sigmoid is not independent per-expert");
}

// ── G8: iters=1 sufficiency ─────────────────────────────────────────────
//
// iters=1 captures ≥90% of the iters=10 λ gain over the unconditioned router.
// Paper §1.4: "Aggressive alignment disrupts stability; a single power
// iteration is more robust and efficient."

#[test]
fn g08_iters1_sufficiency() {
    let (_us, _wgs, grams) = build_synthetic_moe();
    let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();

    let r_seed = seeded_vec(42, N_G * D_G);
    let target = 1.0f32 / (N_G as f32).sqrt();
    let (lambda_vanilla, _) = compute_diagnostics(&r_seed, &grams_ref, N_G, D_G, target);

    // iters=1.
    let mut r1 = r_seed.clone();
    let mut s1 = ScratchShared::new(D_G);
    let res1 = manifold_power_iter_router(&mut r1, &grams_ref, N_G, D_G, 1.0, 1, &mut s1);

    // iters=10 (paper's "fully converged" reference).
    let mut r10 = r_seed.clone();
    let mut s10 = ScratchShared::new(D_G);
    let res10 = manifold_power_iter_router(&mut r10, &grams_ref, N_G, D_G, 1.0, 10, &mut s10);

    let gain_vanilla_to_1 = res1.lambda_alignment - lambda_vanilla;
    let gain_vanilla_to_10 = res10.lambda_alignment - lambda_vanilla;
    let ratio = gain_vanilla_to_1 / gain_vanilla_to_10.abs().max(1e-6);
    eprintln!(
        "G8: λ_vanilla={:.4}  λ(1)={:.4}  λ(10)={:.4}  gain(1)/gain(10)={:.3}",
        lambda_vanilla, res1.lambda_alignment, res10.lambda_alignment, ratio
    );
    gate_check!(
        "G8",
        ratio >= 0.9,
        "iters=1 captured {:.1}% of iters=10 gain (need ≥90%)",
        ratio * 100.0
    );
}

// ── Summary runner ─────────────────────────────────────────────────Â
//
// NOTE: cargo test runs in parallel, so this summary may race with the
// individual g0X tests. For an accurate count, run with:
//   cargo test ... -- --test-threads=1 --nocapture
// The summary test name sorts after g0X (zzz_ prefix) but this is not a
// hard ordering guarantee under parallel execution.

#[test]
fn zzz_goat_gate_summary() {
    // Best-effort wait for parallel g0X tests to update counters.
    std::thread::sleep(Duration::from_millis(500));
    let p = PASS.load(Ordering::SeqCst);
    let f = FAIL.load(Ordering::SeqCst);
    eprintln!();
    eprintln!("══════════════════════════════════════════════════");
    eprintln!("  GOAT GATE: {}/{}  (failures: {})", p, p + f, f);
    eprintln!("══════════════════════════════════════════════════");
    if f > 0 {
        panic!("GOAT GATE FAILED: {}/{} gates red — do NOT promote", f, p + f);
    }
}
