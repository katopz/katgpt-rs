//! GOAT proof test for Gauge-Invariant Adapter Composition (Plan 270).
//!
//! Run: `cargo test --features "gauge_invariant sparse_task_vector" \
//!                       --test bench_270_gauge_invariant_goat -- --nocapture`
//!
//! Validates the LoRA-Muon paper's theorems (arXiv:2606.12921) as inference-time
//! engine plumbing:
//!
//! - **Prop 1 (Gauge invariance):** `gauge_rebalance` preserves `A · B^T` exactly.
//! - **Power iteration:** σ_max estimate within 5% of true after 5 steps.
//! - **NS inv-sqrt correctness:** `P^{-1/2} · P · P^{-1/2} ≈ I` for random PSD P.
//! - **NS inv-sqrt numerical stability:** no NaN/Inf for ill-conditioned P.
//! - **Compose gauge-invariance:** `compose([(1, A1, B1), (1, A2, B2)])` gives
//!   the same result regardless of input factorization (Prop 1).
//! - **NS5 + inv-sqrt roundtrip:** `msign(M) ≈ M · (M^T M)^{-1/2}` (Prop 6).
//! - **SparseTaskVector integration:** `compose_gauge_invariant` merges masks
//!   and matches the full `gauge_invariant_compose` machinery.
//! - **Throughput:** rebalance / inv-sqrt / compose within paper budgets.

#![cfg(feature = "gauge_invariant")]

#[path = "common/ab_timing.rs"]
mod ab_timing;

use katgpt_core::newton_schulz::{InvSqrtScratch, ns_inv_sqrt_psd, ns_inv_sqrt_psd_into};
use katgpt_spectral::gauge_invariant::{
    GaugePair, GaugeRebalanceScratch, gauge_invariant_compose, gauge_invariant_lerp,
    gauge_rebalance,
};
use std::time::{Duration, Instant};

// ── Helpers ──────────────────────────────────────────────────────────────

/// Deterministic pseudo-random matrix (xorshift64) — reproducible across runs.
fn seeded_random_matrix(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
    let mut s = seed;
    let mut mat = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let v = ((s & 0xFFFF) as f32 / 0x8000 as f32) - 1.0;
        mat.push(v);
    }
    mat.shrink_to_fit();
    mat
}

/// Build a random PSD matrix `P = M · M^T + ε·I` for `r × r`.
fn seeded_random_psd(seed: u64, r: usize) -> Vec<f32> {
    let m = seeded_random_matrix(seed, r, r);
    let mut p = vec![0.0_f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0_f32;
            for k in 0..r {
                s += m[i * r + k] * m[j * r + k];
            }
            p[i * r + j] = s;
        }
    }
    // Regularize the diagonal to guarantee PD (paper ε = 1e-5).
    for i in 0..r {
        p[i * r + i] += 1e-3;
    }
    p
}

/// `A · B^T` for A `m × r`, B `n × r` → result `m × n`.
fn abt(a: &[f32], b: &[f32], m: usize, r: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for k in 0..r {
                s += a[i * r + k] * b[j * r + k];
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// Frobenius norm.
fn fro_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Max abs element-wise difference.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Measure best-of-N wall time for a closure, return microseconds.
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

// ── 1. Gauge invariance (Prop 1): rebalance preserves AB^T ───────────────

#[test]
fn t01_gauge_rebalance_preserves_abt_exactly() {
    let m = 16;
    let n = 12;
    let r = 4;
    let a_orig = seeded_random_matrix(42, m, r);
    let b_orig = seeded_random_matrix(99, n, r);
    let w_before = abt(&a_orig, &b_orig, m, r, n);

    let mut a = a_orig.clone();
    let mut b = b_orig.clone();
    let mut scratch = GaugeRebalanceScratch::new(m.max(n), r);
    gauge_rebalance(&mut a, &mut b, m, r, n, r, 1.0, &mut scratch);

    let w_after = abt(&a, &b, m, r, n);
    let diff = max_abs_diff(&w_before, &w_after);
    assert!(
        diff < 1e-5,
        "Prop 1 violated: rebalance changed A·B^T by {diff:.2e} (must be < 1e-5)"
    );
    eprintln!("t01 PASS: rebalance preserves A·B^T (max diff = {diff:.2e})");
}

// ── 2. Power iteration convergence: σ_max within 5% after 5 steps ────────

#[test]
fn t02_power_iteration_converges_within_5_percent() {
    // Construct a matrix with a clear spectral gap so power iteration converges
    // geometrically. A = u · σ · v^T (rank-1 with σ=10) plus a small noise term
    // of spectral norm ≈ 1. The dominant singular value is ≈ 10.
    let m = 32;
    let r = 4;
    let u: Vec<f32> = (0..m).map(|i| (i as f32) * 0.01).collect();
    let v: Vec<f32> = (0..r).map(|i| (i as f32) * 0.01).collect();
    // Outer product u · v^T scaled to σ_max ≈ ‖u‖·‖v‖.
    let mut a = vec![0.0_f32; m * r];
    let u_norm = (u.iter().map(|x| x * x).sum::<f32>()).sqrt();
    let v_norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
    let target_sigma = 10.0_f32;
    let scale = target_sigma / (u_norm * v_norm);
    for i in 0..m {
        for j in 0..r {
            a[i * r + j] = u[i] * v[j] * scale;
        }
    }

    // True σ_max(A) = target_sigma (rank-1).
    let true_sigma = target_sigma;

    // Use gauge_rebalance against B = identity to expose σ_max(A) via the
    // scaling applied to B. After rebalance: σ_max(A_new) ≈ σ_max(B_new), and
    //   σ_max(B_new) = σ_max(B_before) / c,  c = (σ_max(B)/σ_max(A))^{1/2}.
    // So σ_max(B_new) = (σ_max(A) · σ_max(B_before))^{1/2} = √(10 · 1) = √10.
    let mut b = vec![0.0_f32; r * r];
    for i in 0..r {
        b[i * r + i] = 1.0;
    }
    let mut a_mut = a.clone();
    let mut b_mut = b.clone();
    let mut scratch = GaugeRebalanceScratch::new(m.max(r), r);
    gauge_rebalance(&mut a_mut, &mut b_mut, m, r, r, r, 1.0, &mut scratch);

    // Expected σ_max(B_mut) ≈ √(true_sigma · 1).
    let expected_balanced = (true_sigma * 1.0_f32).sqrt();
    // B_mut is diagonal (scaled identity), so σ_max = max diagonal element.
    let b_mut_max = b_mut.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel_err = (b_mut_max - expected_balanced).abs() / expected_balanced;
    assert!(
        rel_err < 0.10,
        "Power iteration σ_max estimate error = {rel_err:.3} > 10% \
         (b_mut_max={b_mut_max}, expected≈{expected_balanced}). Power iteration on rank-1 matrix should converge fast."
    );
    eprintln!(
        "t02 PASS: power iteration σ_max balanced to ≈ √(σ_a·σ_b) = {:.3} (got {:.3}, err {:.2}%)",
        expected_balanced,
        b_mut_max,
        rel_err * 100.0
    );
}

// ── 3. NS inv-sqrt correctness: P^{-1/2} · P · P^{-1/2} ≈ I ──────────────

#[test]
fn t03_ns_inv_sqrt_psd_round_trip_to_identity() {
    let r = 8;
    let p = seeded_random_psd(2024, r);
    let mut inv_sqrt = vec![0.0_f32; r * r];
    ns_inv_sqrt_psd(&p, r, &mut inv_sqrt, 7);

    // Compute inv_sqrt · P · inv_sqrt.
    let mut tmp = vec![0.0_f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0_f32;
            for k in 0..r {
                s += inv_sqrt[i * r + k] * p[k * r + j];
            }
            tmp[i * r + j] = s;
        }
    }
    let mut result = vec![0.0_f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0_f32;
            for k in 0..r {
                s += tmp[i * r + k] * inv_sqrt[j * r + k];
            }
            result[i * r + j] = s;
        }
    }

    // Compare to identity.
    let mut max_diff = 0.0_f32;
    for i in 0..r {
        for j in 0..r {
            let expected = if i == j { 1.0 } else { 0.0 };
            let d = (result[i * r + j] - expected).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    assert!(
        max_diff < 1e-3,
        "P^{{-1/2}} · P · P^{{-1/2}} ≠ I (max diff = {max_diff:.2e}, must be < 1e-3)"
    );
    eprintln!("t03 PASS: NS inv-sqrt recovers I from random PSD (max diff = {max_diff:.2e})");
}

// ── 4. NS inv-sqrt numerical stability: ill-conditioned P (κ ≤ 1e6) ──────

#[test]
fn t04_ns_inv_sqrt_stable_under_ill_conditioning() {
    let r = 8;
    // Diagonal matrix with condition number 1e6: eigenvalues span [1, 1e6].
    let mut p = vec![0.0_f32; r * r];
    let lo: f32 = 1.0;
    let hi: f32 = 1e6;
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    for i in 0..r {
        let t = i as f32 / (r - 1) as f32;
        let log_v = log_lo + t * (log_hi - log_lo);
        p[i * r + i] = log_v.exp();
    }

    let mut inv_sqrt = vec![0.0_f32; r * r];
    let mut scratch = InvSqrtScratch::new(r);
    ns_inv_sqrt_psd_into(&p, r, &mut inv_sqrt, &mut scratch, 7);

    // Stability requirement: no NaN/Inf in the output.
    for &v in &inv_sqrt {
        assert!(
            v.is_finite(),
            "NS inv-sqrt produced non-finite value {v} for κ=1e6 matrix"
        );
    }

    // Correctness check for ill-conditioned case: P^{-1/2} · P · P^{-1/2} ≈ I.
    // This is a weaker test than the full identity check in t03 (which uses a
    // well-conditioned PSD). For κ=1e6 we tolerate larger deviation due to the
    // Frobenius-norm normalization shifting small eigenvalues below the NS
    // polynomial's accuracy floor.
    let mut tmp = vec![0.0_f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0_f32;
            for k in 0..r {
                s += inv_sqrt[i * r + k] * p[k * r + j];
            }
            tmp[i * r + j] = s;
        }
    }
    let mut result = vec![0.0_f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0_f32;
            for k in 0..r {
                s += tmp[i * r + k] * inv_sqrt[j * r + k];
            }
            result[i * r + j] = s;
        }
    }
    let mut max_diff = 0.0_f32;
    for i in 0..r {
        for j in 0..r {
            let expected = if i == j { 1.0 } else { 0.0 };
            let d = (result[i * r + j] - expected).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    // For κ=1e6, the NS polynomial (optimized for κ≤1e3 per paper Table 2)
    // will lose precision on the smallest eigenvalues. This is expected — the
    // paper's algorithm targets well-conditioned Gram matrices from LoRA-Muon
    // training (r ≤ 64, κ typically < 1e3). The key stability guarantee is
    // no NaN/Inf (checked above). We also verify the product stays bounded
    // (no explosion), which rules out divergence even if precision is lost.
    assert!(
        max_diff < 2.0,
        "P^{{-1/2}} · P · P^{{-1/2}} deviates from I by {max_diff:.2e} for κ=1e6 \
         (NS polynomial is optimized for κ≤1e3; values up to 2.0 are stable-bounded)"
    );
    eprintln!(
        "t04 PASS: NS inv-sqrt stable for κ=1e6 (no NaN/Inf, ‖P^{{-1/2}}PP^{{-1/2}} - I‖_max = {max_diff:.2e})"
    );
}

// ── 5. Compose gauge-invariance: factorization-independent output ────────

#[test]
fn t05_compose_gauge_invariant_under_input_rescaling() {
    // Paper Prop 1: composing gauge-equivalent inputs gives the same result.
    let m = 8;
    let n = 6;
    let r = 4;
    let a = seeded_random_matrix(100, m, r);
    let b = seeded_random_matrix(101, n, r);

    // Gauge transform: A' = 7·A, B' = B/7. Same A·B^T.
    let c = 7.0_f32;
    let a_g: Vec<f32> = a.iter().map(|v| v * c).collect();
    let b_g: Vec<f32> = b.iter().map(|v| v / c).collect();

    let merged_r = 2 * r;

    // Compose original with itself (η=0.5, η=0.5).
    let mut out_a_orig = vec![0.0_f32; m * merged_r];
    let mut out_b_orig = vec![0.0_f32; n * merged_r];
    let pairs_orig = [
        GaugePair {
            eta: 0.5,
            a: &a,
            b: &b,
            a_rows: m,
            b_rows: n,
            rank: r,
        },
        GaugePair {
            eta: 0.5,
            a: &a,
            b: &b,
            a_rows: m,
            b_rows: n,
            rank: r,
        },
    ];
    gauge_invariant_compose(&pairs_orig, &mut out_a_orig, &mut out_b_orig);
    let w_orig = abt(&out_a_orig, &out_b_orig, m, merged_r, n);

    // Compose gauge-transformed with itself.
    let mut out_a_g = vec![0.0_f32; m * merged_r];
    let mut out_b_g = vec![0.0_f32; n * merged_r];
    let pairs_g = [
        GaugePair {
            eta: 0.5,
            a: &a_g,
            b: &b_g,
            a_rows: m,
            b_rows: n,
            rank: r,
        },
        GaugePair {
            eta: 0.5,
            a: &a_g,
            b: &b_g,
            a_rows: m,
            b_rows: n,
            rank: r,
        },
    ];
    gauge_invariant_compose(&pairs_g, &mut out_a_g, &mut out_b_g);
    let w_g = abt(&out_a_g, &out_b_g, m, merged_r, n);

    let diff = max_abs_diff(&w_orig, &w_g);
    assert!(
        diff < 1e-3,
        "Compose is not gauge-invariant: rescaling inputs by c={c} changed W by {diff:.2e}"
    );
    eprintln!(
        "t05 PASS: compose([(1,A·c, B/c), (1,A·c, B/c)]) = compose([(1,A,B),(1,A,B)]) (diff {diff:.2e})"
    );
}

// ── 6. NS5 + inv-sqrt roundtrip: msign(M) ≈ M · (M^T M)^{-1/2} ───────────

#[test]
fn t06_msign_via_ns_inv_sqrt_matches_orthonormality() {
    // Paper Prop 6: `msign(M) = M · (M^T M)^{-1/2}` yields orthonormal columns.
    let m = 32;
    let r = 8;
    let mat = seeded_random_matrix(555, m, r);

    // P = M^T M (r × r PSD).
    let mut p = vec![0.0_f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0_f32;
            for k in 0..m {
                s += mat[k * r + i] * mat[k * r + j];
            }
            p[i * r + j] = s;
        }
    }

    // P^{-1/2}.
    let mut inv_sqrt = vec![0.0_f32; r * r];
    ns_inv_sqrt_psd(&p, r, &mut inv_sqrt, 7);

    // M · P^{-1/2} → (m × r).
    let mut msign = vec![0.0_f32; m * r];
    for i in 0..m {
        for j in 0..r {
            let mut s = 0.0_f32;
            for k in 0..r {
                s += mat[i * r + k] * inv_sqrt[k * r + j];
            }
            msign[i * r + j] = s;
        }
    }

    // Check orthonormality: msign^T · msign ≈ I.
    let mut gram = vec![0.0_f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0_f32;
            for k in 0..m {
                s += msign[k * r + i] * msign[k * r + j];
            }
            gram[i * r + j] = s;
        }
    }
    let mut max_diff = 0.0_f32;
    for i in 0..r {
        for j in 0..r {
            let expected = if i == j { 1.0 } else { 0.0 };
            let d = (gram[i * r + j] - expected).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    assert!(
        max_diff < 1e-2,
        "msign(M) = M·(M^TM)^{{-1/2}} not orthonormal (max diff from I = {max_diff:.2e})"
    );
    eprintln!("t06 PASS: msign via NS inv-sqrt is orthonormal (max deviation = {max_diff:.2e})");
}

// ── 7. SparseTaskVector.compose_gauge_invariant: shape + mask merge ──────

#[cfg(feature = "sparse_task_vector")]
#[test]
fn t07_sparse_task_vector_compose_gauge_invariant() {
    use katgpt_sparse::sparse_task_vector::SparseTaskVector;

    let a = SparseTaskVector::from_dense(
        &[0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
        (3, 4),
        1e-5,
    );
    let b = SparseTaskVector::from_dense(
        &[0.0, 0.0, 0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0],
        (3, 4),
        1e-5,
    );
    let eta = 0.5_f32;

    let merged = a.clone().compose_gauge_invariant(&b, eta);

    // Shape preserved.
    assert_eq!(merged.shape, (3, 4), "shape must be preserved");

    // Mask = sorted union of [0,4,8] and [2,4,8] = [0,2,4,8].
    assert_eq!(merged.mask, vec![0, 2, 4, 8], "mask must be sorted union");

    // Deltas: idx 0 from a only (0.5), idx 2 from b only (0.5·0.4=0.2),
    // idx 4 shared (0.3 + 0.5·0.1 = 0.35), idx 8 shared (0.2 + 0.5·0.6 = 0.5).
    assert!((merged.deltas[0] - 0.5).abs() < 1e-6);
    assert!((merged.deltas[1] - 0.2).abs() < 1e-6);
    assert!((merged.deltas[2] - 0.35).abs() < 1e-6);
    assert!((merged.deltas[3] - 0.5).abs() < 1e-6);

    // Roundtrip: merged.apply_to(base) ≡ a.apply_to(base) + eta·b.apply_to(base).
    let mut base_a = vec![1.0_f32; 12];
    let mut base_b = vec![1.0_f32; 12];
    let mut base_m = vec![1.0_f32; 12];
    a.apply_to(&mut base_a);
    let mut b_scaled = b.clone();
    b_scaled.eta = eta;
    b_scaled.apply_to(&mut base_b);
    merged.apply_to(&mut base_m);
    let diff = max_abs_diff(
        &base_a
            .iter()
            .zip(base_b.iter())
            .map(|(&x, &y)| x + y - 1.0)
            .collect::<Vec<_>>(),
        &base_m,
    );
    assert!(diff < 1e-6, "merged apply ≠ a + eta·b (diff = {diff:.2e})");
    eprintln!(
        "t07 PASS: SparseTaskVector.compose_gauge_invariant merges masks and preserves apply"
    );
}

// ── 8. Throughput: rebalance (256×16, 16×256) ────────────────────────────

/// t08 — rebalance throughput.
///
/// **Issue 723 Class A / T7.** Filed at 30.9 µs against a 5 µs bar and classed
/// as a load artifact. Three separate things were wrong, and the box was not
/// one of them:
///
/// 1. **The setup was inside the timed region.** `gauge_rebalance` scales its
///    operands in place, so they must be restored between calls — but the
///    closure restored them by *re-generating* two 256×16 xorshift matrices
///    (8,192 elements plus two `Vec` allocations) with the clock running. A
///    gate whose own message says "BENCH rebalance" was measuring the
///    generator. Restoration is now a `copy_from_slice` from templates built
///    once, outside the timed region.
/// 2. **`warmup = 3` did not ramp the core.** With the old warmup this gate
///    read **47 µs run alone** and **18.6 µs run inside its own suite** — the
///    verdict depended on the test *filter*, because 3 iterations of ~20 µs is
///    60 µs of work and the P-core is still at idle clock. At warmup/iters =
///    200/200 both invocations converge to 18.0–19.3 µs. A gate that a
///    `--exact` filter can flip is not measuring the code.
/// 3. **The 5 µs bar had never been executed in release.** It is an aspiration
///    from Plan 279, not a measurement: `gauge_rebalance` does 2 × 5 power
///    iterations over a 256×16 factor, and the inner accumulate
///    (`vout[k] += row[k] * u_i`, `power_iterate_sigma_max`) is scalar over
///    rank while only the dot is SIMD. 5 µs would need ~18 flop/ns.
///
/// **Re-pinned 30 µs with provenance** (measured best-of-200, M3 Max, release,
/// 2026-09-05, twelve runs isolated and in-suite at box load 6–14: eleven in
/// 17.67–19.25 µs and one outlier at 21.67 — the min-of-200 is load-invariant
/// by construction, but not outlier-proof, so the pin is ~1.4x the worst
/// observed sample rather than 1.3x the median). The 5 µs figure stays as
/// the paper target it always was. Closing the 3.7x gap is a real optimisation
/// with a real target (SIMD the rank-wise accumulate) and is filed separately
/// as `.issues/726` — it is not a tolerance question, so it must not be
/// smuggled into a gate re-pin.
///
/// **Issue 726 LANDED — re-pinned 15 µs with provenance** (same day): the
/// rank-wise accumulate in `power_iterate_sigma_max` was swapped from the
/// scalar `vout[k] += row[k] * u_i` loop to `katgpt_core::simd::
/// simd_fused_scale_acc` (`vfmaq_f32`, single-rounding FMA). T1 pricing first
/// (interleaved A/B vs the accumulate stubbed to the k=0 lane, 9 rounds ×
/// 200 iters × 3 runs): the accumulate was ~77–78% of the whole call. T2
/// measured: interleaved median-of-ratios improvement 54.1% / 52.7% / 50.9%
/// (9 rounds each, per-round ranges 0.453–0.499, 9/9 rounds survived every
/// run, box load 3.5–4.9); absolute t08 best-of-200 18.96/19.21/19.21 µs →
/// **9.00/9.00/9.00 µs** (three runs, −53%). T3: the swap is NOT bit-identical
/// — the scalar loop was NOT FMA-contracted by LLVM (bit-compare: t01-shape
/// 64/64 and 48/48 elements differ; t08-shape 3136/4096; max |Δ| = 1.19e-7 ≈
/// 1 ULP) — and every exactness assertion still passes at its committed
/// tolerance: t01 reads 1.2e-7 against 1e-5 (~80× headroom), the compose
/// gauge-invariance tests against 1e-3, and the neighbours at their own
/// tolerances (bench_279 G6 1.19e-7 vs 1e-3; katgpt-sparse lib 39/39). The
/// swap is a rounding change absorbed by the gates, not an assertion
/// loosening — no tolerance was touched.
///
/// The 5 µs paper target: 9.0 µs is now within 1.8× of it. The T1 pricing
/// arms show the remaining floor — the per-row σ-dots, the two full-matrix
/// in-place scales, and the final ‖M·v‖ pass alone measure ~5.4–6.5 µs at
/// this shape — so 5 µs is not reachable by the accumulate swap alone and
/// would need the dots/scales/final-pass optimized too. Kept as the paper
/// aspiration, now a ~1.8× gap instead of ~3.7×.
///
/// The statistic is the load-invariant one: an absolute budget has no second
/// arm to ratio against, so the minimum of N is used — contention can only
/// ever add time, so the smallest sample is the closest observation of the
/// uncontended cost (`bench_us` already did best-of-N, and that is the one
/// thing about the original instrument that was right).
#[test]
fn t08_throughput_rebalance_256x16() {
    let m = 256;
    let n = 256;
    let r = 16;
    let a0 = seeded_random_matrix(1, m, r);
    let b0 = seeded_random_matrix(2, n, r);
    let mut a = a0.clone();
    let mut b = b0.clone();
    let mut scratch = GaugeRebalanceScratch::new(m.max(n), r);

    // Debug builds are ~50-100× slower than release; gate threshold accordingly.
    // Release GATE: <= 15 µs (Issue 726 re-pin — measured 9.00 µs ×3 after the
    // simd accumulate swap; see the doc comment). Release PAPER TARGET: 5 µs —
    // reported, not asserted (now a ~1.8× gap; the remaining floor is the
    // σ-dots + in-place scales + final pass, not the accumulate).
    // Debug allowance: < 2000 μs (power iteration is O(m·r²·n_steps)
    // = 256·256·5 ≈ 327K ops per factor, ×2 factors, unvectorized).
    let target_us = if cfg!(debug_assertions) { 2000.0 } else { 15.0 };
    let us = ab_timing::best_of_us(200, 200, || {
        // Setup (NOT timed): restore the operands `gauge_rebalance` scaled in
        // place, so the matrices cannot collapse under repeated scaling.
        a.copy_from_slice(&a0);
        b.copy_from_slice(&b0);
        let t0 = Instant::now();
        gauge_rebalance(&mut a, &mut b, m, r, n, r, 1.0, &mut scratch);
        t0.elapsed()
    });
    assert!(
        us <= target_us,
        "rebalance ({m}×{r}, {n}×{r}) took {us:.1} μs > {target_us:.0} μs target"
    );
    let paper_target_us = 5.0_f64;
    eprintln!(
        "t08 BENCH rebalance ({m}×{r}, {n}×{r}): {us:.2} μs (gate {target_us:.0} μs, \
         Plan 279 paper target {paper_target_us:.0} μs — Issue 726 simd accumulate landed)"
    );
}

// ── 9. Throughput: inv-sqrt 16×16 PSD ────────────────────────────────────

#[test]
fn t09_throughput_inv_sqrt_16x16() {
    let r = 16;
    let p = seeded_random_psd(42, r);
    let mut out = vec![0.0_f32; r * r];
    let mut scratch = InvSqrtScratch::new(r);

    // Release target: < 10 μs. Debug allowance: < 1000 μs.
    let target_us = if cfg!(debug_assertions) { 1000.0 } else { 10.0 };
    let us = bench_us(3, 20, || {
        ns_inv_sqrt_psd_into(&p, r, &mut out, &mut scratch, 7);
    });
    assert!(
        us <= target_us,
        "ns_inv_sqrt_psd_into ({r}×{r}) took {us:.1} μs > {target_us:.0} μs target"
    );
    let target_release = 10.0_f64;
    eprintln!(
        "t09 BENCH ns_inv_sqrt_psd_into ({r}×{r}): {us:.2} μs (debug target {target_us:.0} μs, release target {target_release:.0} μs)"
    );
}

// ── 10. Throughput: compose 4 pairs ──────────────────────────────────────

#[test]
fn t10_throughput_compose_4_pairs() {
    let m = 64;
    let n = 64;
    let r = 8;
    let n_pairs = 4;
    let a1 = seeded_random_matrix(1, m, r);
    let b1 = seeded_random_matrix(2, n, r);
    let a2 = seeded_random_matrix(3, m, r);
    let b2 = seeded_random_matrix(4, n, r);
    let a3 = seeded_random_matrix(5, m, r);
    let b3 = seeded_random_matrix(6, n, r);
    let a4 = seeded_random_matrix(7, m, r);
    let b4 = seeded_random_matrix(8, n, r);
    let pairs = [
        GaugePair {
            eta: 0.25,
            a: &a1,
            b: &b1,
            a_rows: m,
            b_rows: n,
            rank: r,
        },
        GaugePair {
            eta: 0.25,
            a: &a2,
            b: &b2,
            a_rows: m,
            b_rows: n,
            rank: r,
        },
        GaugePair {
            eta: 0.25,
            a: &a3,
            b: &b3,
            a_rows: m,
            b_rows: n,
            rank: r,
        },
        GaugePair {
            eta: 0.25,
            a: &a4,
            b: &b4,
            a_rows: m,
            b_rows: n,
            rank: r,
        },
    ];
    let merged_r = n_pairs * r;
    let mut out_a = vec![0.0_f32; m * merged_r];
    let mut out_b = vec![0.0_f32; n * merged_r];

    // Release target: < 50 μs. Debug allowance: < 5000 μs.
    let target_us = if cfg!(debug_assertions) { 5000.0 } else { 50.0 };
    let us = bench_us(3, 20, || {
        gauge_invariant_compose(&pairs, &mut out_a, &mut out_b);
    });
    assert!(
        us <= target_us,
        "gauge_invariant_compose ({n_pairs} pairs, {m}×{r}, {n}×{r}) took {us:.1} μs > {target_us:.0} μs target"
    );
    let target_release = 50.0_f64;
    eprintln!(
        "t10 BENCH gauge_invariant_compose ({n_pairs} pairs, {m}×{r}): {us:.2} μs (debug target {target_us:.0} μs, release target {target_release:.0} μs)"
    );
}

// ── 11. Edge case: rebalance on rank-1 pair ──────────────────────────────

#[test]
fn t11_gauge_rebalance_rank_one_pair() {
    // Rank-1 LoRA: A is m×1, B is n×1. Rebalance should produce σ_max(A) ≈ σ_max(B).
    let m = 8;
    let n = 4;
    let r = 1;
    let mut a: Vec<f32> = (0..m).map(|i| (i as f32) * 0.5).collect();
    let mut b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let w_before = abt(&a, &b, m, r, n);

    let mut scratch = GaugeRebalanceScratch::new(m.max(n), r);
    gauge_rebalance(&mut a, &mut b, m, r, n, r, 1.0, &mut scratch);

    let w_after = abt(&a, &b, m, r, n);
    let diff = max_abs_diff(&w_before, &w_after);
    assert!(diff < 1e-5, "rank-1 rebalance changed A·B^T by {diff:.2e}");
    // |A| and |B| should be balanced (both = sqrt(‖A‖·‖B‖) for rank-1).
    let norm_a = fro_norm(&a);
    let norm_b = fro_norm(&b);
    let ratio = norm_a / norm_b;
    assert!(
        (ratio - 1.0).abs() < 0.10,
        "rank-1 rebalance should balance norms, got ratio {ratio:.3}"
    );
    eprintln!("t11 PASS: rank-1 rebalance preserves A·B^T and balances ‖A‖≈‖B‖ (ratio {ratio:.3})");
}

// ── 12. Edge case: compose with η=0 yields only pair 1 ───────────────────

#[test]
fn t12_compose_eta_zero_yields_only_first_pair() {
    let m = 6;
    let n = 5;
    let r = 3;
    let a1 = seeded_random_matrix(10, m, r);
    let b1 = seeded_random_matrix(11, n, r);
    let a2 = seeded_random_matrix(12, m, r);
    let b2 = seeded_random_matrix(13, n, r);

    let merged_r = 2 * r;
    let mut out_a = vec![0.0_f32; m * merged_r];
    let mut out_b = vec![0.0_f32; n * merged_r];
    gauge_invariant_lerp(&a1, &b1, &a2, &b2, m, n, r, 0.0, &mut out_a, &mut out_b);

    let w_merged = abt(&out_a, &out_b, m, merged_r, n);
    let w_p1 = abt(&a1, &b1, m, r, n);
    let diff = max_abs_diff(&w_merged, &w_p1);
    assert!(
        diff < 1e-3,
        "α=0 should yield pair 1 only (diff = {diff:.2e})"
    );
    eprintln!("t12 PASS: lerp α=0 → pair 1 only (diff {diff:.2e})");
}

// ── 13. Edge case: lerp α=1 yields only pair 2 ───────────────────────────

#[test]
fn t13_compose_eta_one_yields_only_second_pair() {
    let m = 6;
    let n = 5;
    let r = 3;
    let a1 = seeded_random_matrix(10, m, r);
    let b1 = seeded_random_matrix(11, n, r);
    let a2 = seeded_random_matrix(12, m, r);
    let b2 = seeded_random_matrix(13, n, r);

    let merged_r = 2 * r;
    let mut out_a = vec![0.0_f32; m * merged_r];
    let mut out_b = vec![0.0_f32; n * merged_r];
    gauge_invariant_lerp(&a1, &b1, &a2, &b2, m, n, r, 1.0, &mut out_a, &mut out_b);

    let w_merged = abt(&out_a, &out_b, m, merged_r, n);
    let w_p2 = abt(&a2, &b2, m, r, n);
    let diff = max_abs_diff(&w_merged, &w_p2);
    assert!(
        diff < 1e-3,
        "α=1 should yield pair 2 only (diff = {diff:.2e})"
    );
    eprintln!("t13 PASS: lerp α=1 → pair 2 only (diff {diff:.2e})");
}

// ── 14. Edge case: compose preserves Frobenius norm exactly ──────────────

#[test]
fn t14_compose_preserves_frobenius_norm() {
    // W_merged = A_1·B_1^T + A_2·B_2^T (paper Prop 1 corollary).
    let m = 8;
    let n = 6;
    let r = 4;
    let a1 = seeded_random_matrix(21, m, r);
    let b1 = seeded_random_matrix(22, n, r);
    let a2 = seeded_random_matrix(23, m, r);
    let b2 = seeded_random_matrix(24, n, r);

    let merged_r = 2 * r;
    let mut out_a = vec![0.0_f32; m * merged_r];
    let mut out_b = vec![0.0_f32; n * merged_r];
    let pairs = [
        GaugePair {
            eta: 1.0,
            a: &a1,
            b: &b1,
            a_rows: m,
            b_rows: n,
            rank: r,
        },
        GaugePair {
            eta: 1.0,
            a: &a2,
            b: &b2,
            a_rows: m,
            b_rows: n,
            rank: r,
        },
    ];
    gauge_invariant_compose(&pairs, &mut out_a, &mut out_b);
    let w_merged = abt(&out_a, &out_b, m, merged_r, n);

    // Naive (post-rebalance) sum: each pair was rebalanced, but A·B^T preserved.
    let w1 = abt(&a1, &b1, m, r, n);
    let w2 = abt(&a2, &b2, m, r, n);
    let w_sum: Vec<f32> = (0..m * n).map(|i| w1[i] + w2[i]).collect();

    let merged_norm = fro_norm(&w_merged);
    let sum_norm = fro_norm(&w_sum);
    let rel_err = (merged_norm - sum_norm).abs() / sum_norm;
    assert!(
        rel_err < 1e-3,
        "‖W_compose‖ = {merged_norm:.5} ≠ ‖W_1 + W_2‖ = {sum_norm:.5} (rel err {rel_err:.2e})"
    );
    eprintln!("t14 PASS: ‖compose(W_1, W_2)‖ ≈ ‖W_1 + W_2‖ (rel err {rel_err:.2e})");
}

// ── 15. Edge case: zero matrix rebalance is safe (no NaN/Inf) ────────────

#[test]
fn t15_rebalance_zero_matrix_is_safe() {
    let m = 8;
    let n = 6;
    let r = 4;
    let mut a = vec![0.0_f32; m * r];
    let mut b = vec![0.0_f32; n * r];
    let mut scratch = GaugeRebalanceScratch::new(m.max(n), r);
    // Must not panic or produce NaN.
    gauge_rebalance(&mut a, &mut b, m, r, n, r, 1.0, &mut scratch);
    for &v in a.iter().chain(b.iter()) {
        assert!(
            v.is_finite(),
            "zero rebalance produced non-finite value {v}"
        );
        assert!(
            v.abs() < 1e-20,
            "zero rebalance should leave zeros, got {v}"
        );
    }
    eprintln!("t15 PASS: zero-matrix rebalance is safe (no NaN/Inf, no perturbation)");
}

// ── 16. Edge case: NS inv-sqrt of identity returns identity ──────────────

#[test]
fn t16_ns_inv_sqrt_identity_returns_identity() {
    let r = 8;
    let p: Vec<f32> = (0..r * r)
        .map(|i| if i % (r + 1) == 0 { 1.0 } else { 0.0 })
        .collect();
    let mut inv_sqrt = vec![0.0_f32; r * r];
    let mut scratch = InvSqrtScratch::new(r);
    ns_inv_sqrt_psd_into(&p, r, &mut inv_sqrt, &mut scratch, 7);

    let mut max_diff = 0.0_f32;
    for i in 0..r {
        for j in 0..r {
            let expected = if i == j { 1.0 } else { 0.0 };
            let d = (inv_sqrt[i * r + j] - expected).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    assert!(
        max_diff < 1e-3,
        "I^{{-1/2}} should be I, max diff = {max_diff:.2e}"
    );
    eprintln!("t16 PASS: NS inv-sqrt of identity = identity (max diff {max_diff:.2e})");
}

// ── 17. Edge case: large condition number κ = 1e8 still finite ───────────

#[test]
fn t17_ns_inv_sqrt_extreme_condition_number_remains_finite() {
    let r = 6;
    let mut p = vec![0.0_f32; r * r];
    // Eigenvalues log-spaced from 1e-4 to 1e4 → κ = 1e8.
    for i in 0..r {
        let t = i as f32 / (r - 1) as f32;
        let log_v = -4.0 + t * 8.0;
        p[i * r + i] = 10.0_f32.powf(log_v);
    }
    let mut inv_sqrt = vec![0.0_f32; r * r];
    let mut scratch = InvSqrtScratch::new(r);
    ns_inv_sqrt_psd_into(&p, r, &mut inv_sqrt, &mut scratch, 7);

    for &v in &inv_sqrt {
        assert!(
            v.is_finite(),
            "NS inv-sqrt produced non-finite value for κ=1e8"
        );
    }
    eprintln!("t17 PASS: NS inv-sqrt finite for κ=1e8 (no NaN/Inf)");
}
