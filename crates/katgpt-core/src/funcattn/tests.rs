//! Tests for funcattn (extracted from mod.rs by Issue 176).

use super::*;

/// Deterministic PRNG (xorshift64*), reproducible across runs.
fn make_rng(seed: u64) -> impl Iterator<Item = f32> {
    let mut state = seed.max(1);
    std::iter::from_fn(move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bits = (state >> 11) as u32;
        let u01 = bits as f32 / (u32::MAX as f32);
        Some(u01 * 2.0 - 1.0)
    })
}

fn fill_rand(buf: &mut [f32], seed: u64) {
    let mut rng = make_rng(seed);
    for x in buf.iter_mut() {
        *x = rng.next().unwrap();
    }
}

/// Reference (allocating, scalar) dual-form FUNCATTN, matching
/// `Functional_attention.py::FunctionalMap_Attention_Structured_Mesh_2D.forward`
/// for a single (B=1, H=1) head.
#[allow(clippy::too_many_arguments)]
fn funcattn_reference(
    x_basis: &[f32],
    x_value: &[f32],
    w_basis: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    n: usize,
    d: usize,
    k: usize,
    basis: FuncAttnBasis,
    alpha: f32,
    temperature: f32,
) -> Vec<f32> {
    let inv_temp = 1.0 / temperature;
    // (1) Φ = row_norm(act((x_basis · w_basis) / τ))
    let mut phi = vec![0.0f32; n * k];
    for i in 0..n {
        for j in 0..k {
            let mut s = 0.0;
            for dd in 0..d {
                s += x_basis[i * d + dd] * w_basis[j * d + dd];
            }
            phi[i * k + j] = s * inv_temp;
        }
        let row = &mut phi[i * k..(i + 1) * k];
        match basis {
            FuncAttnBasis::Softmax => {
                let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for x in row.iter_mut() {
                    *x = (*x - mx).exp();
                }
                let s: f32 = row.iter().sum();
                if s > 0.0 {
                    for x in row.iter_mut() {
                        *x /= s;
                    }
                }
            }
            FuncAttnBasis::Sigmoid => {
                for x in row.iter_mut() {
                    *x = 1.0 / (1.0 + (-*x).exp());
                }
                let s: f32 = row.iter().sum();
                if s > 0.0 {
                    for x in row.iter_mut() {
                        *x /= s;
                    }
                }
            }
        }
    }

    // (2) col_sum and slice_token = (Φᵀ · x_value) / (col_sum + ε)
    let mut col_sum = vec![0.0f32; k];
    for g in 0..k {
        for i in 0..n {
            col_sum[g] += phi[i * k + g];
        }
    }
    let mut slice_token = vec![0.0f32; k * d];
    for g in 0..k {
        for dd in 0..d {
            let mut s = 0.0;
            for i in 0..n {
                s += phi[i * k + g] * x_value[i * d + dd];
            }
            slice_token[g * d + dd] = s / (col_sum[g] + 1e-5);
        }
    }

    // (3) to_q, to_k, to_v
    let apply_linear = |w: &[f32], out: &mut Vec<f32>| {
        for g in 0..k {
            for j in 0..d {
                let mut s = 0.0;
                for i in 0..d {
                    s += w[j * d + i] * slice_token[g * d + i];
                }
                out[g * d + j] = s;
            }
        }
    };
    let mut q_slice = vec![0.0f32; k * d];
    let mut k_slice = vec![0.0f32; k * d];
    let mut v_slice = vec![0.0f32; k * d];
    apply_linear(w_q, &mut q_slice);
    apply_linear(w_k, &mut k_slice);
    apply_linear(w_v, &mut v_slice);

    // (4) Dual-form solve: Z = reg⁻¹ · K̃ᵀ, reg = (1-α)·K̃ᵀ·K̃ + α·I
    // Compute reg (d, d) via Gauss-Jordan inversion.
    let one_minus_alpha = 1.0 - alpha;
    let mut reg = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0;
            for l in 0..k {
                s += k_slice[l * d + i] * k_slice[l * d + j];
            }
            reg[i * d + j] = one_minus_alpha * s;
        }
    }
    for i in 0..d {
        reg[i * d + i] += alpha;
    }
    // Invert reg via Gauss-Jordan with partial pivoting.
    let mut aug = vec![0.0f32; d * 2 * d];
    for i in 0..d {
        for j in 0..d {
            aug[i * 2 * d + j] = reg[i * d + j];
        }
        aug[i * 2 * d + d + i] = 1.0;
    }
    for col in 0..d {
        let mut piv = col;
        for r in (col + 1)..d {
            if aug[r * 2 * d + col].abs() > aug[piv * 2 * d + col].abs() {
                piv = r;
            }
        }
        if piv != col {
            for j in 0..2 * d {
                aug.swap(col * 2 * d + j, piv * 2 * d + j);
            }
        }
        let diag = aug[col * 2 * d + col];
        assert!(diag.abs() > 1e-20, "singular reg in reference");
        let inv_diag = 1.0 / diag;
        for j in 0..2 * d {
            aug[col * 2 * d + j] *= inv_diag;
        }
        for r in 0..d {
            if r != col {
                let factor = aug[r * 2 * d + col];
                if factor != 0.0 {
                    for j in 0..2 * d {
                        aug[r * 2 * d + j] -= factor * aug[col * 2 * d + j];
                    }
                }
            }
        }
    }
    // reg⁻¹ is now in aug[:, d:2d].
    // Z = reg⁻¹ · K̃ᵀ. Z[l, j] = Σ_i reg⁻¹[l, i] · K̃ᵀ[i, j] = Σ_i reg⁻¹[l, i] · K̃[j, i].
    // We need Zᵀ (k, d): Zᵀ[j, l] = Z[l, j] = Σ_i reg⁻¹[l, i] · K̃[j, i].
    let mut z_op_t = vec![0.0f32; k * d];
    for j in 0..k {
        for l in 0..d {
            let mut s = 0.0;
            for i in 0..d {
                s += aug[l * 2 * d + d + i] * k_slice[j * d + i];
            }
            z_op_t[j * d + l] = s;
        }
    }

    // (5) C = Q̃ · Z. C[i, j] = Σ_l Q̃[i, l] · Z[l, j] = dot(Q̃ row i, Zᵀ row j).
    let mut c_op = vec![0.0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for l in 0..d {
                s += q_slice[i * d + l] * z_op_t[j * d + l];
            }
            c_op[i * k + j] = s;
        }
    }

    // (6) out_slice = C · Ṽ.
    let mut out_slice = vec![0.0f32; k * d];
    for i in 0..k {
        for dd in 0..d {
            let mut s = 0.0;
            for l in 0..k {
                s += c_op[i * k + l] * v_slice[l * d + dd];
            }
            out_slice[i * d + dd] = s;
        }
    }

    // (7) out = Φ · out_slice.
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        for dd in 0..d {
            let mut s = 0.0;
            for g in 0..k {
                s += phi[i * k + g] * out_slice[g * d + dd];
            }
            out[i * d + dd] = s;
        }
    }
    out
}

fn frobenius(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn run_forward(
    n: usize,
    d: usize,
    k: usize,
    alpha: f32,
    temperature: f32,
    basis: FuncAttnBasis,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut x_basis = vec![0.0f32; n * d];
    let mut x_value = vec![0.0f32; n * d];
    let mut w_basis = vec![0.0f32; k * d];
    let mut w_q = vec![0.0f32; d * d];
    let mut w_k = vec![0.0f32; d * d];
    let mut w_v = vec![0.0f32; d * d];
    fill_rand(&mut x_basis, seed);
    fill_rand(&mut x_value, seed + 1);
    fill_rand(&mut w_basis, seed + 2);
    fill_rand(&mut w_q, seed + 3);
    fill_rand(&mut w_k, seed + 4);
    fill_rand(&mut w_v, seed + 5);

    let cfg = FuncAttnConfig {
        d,
        k,
        basis,
        alpha,
        temperature,
        cholesky_jitter: 1e-6,
    };
    let mut scratch = FuncAttnScratch::new(n, d, k);
    let mut out = vec![0.0f32; n * d];
    funcattn_forward(
        &x_basis,
        &x_value,
        &w_basis,
        &w_q,
        &w_k,
        &w_v,
        &cfg,
        &mut scratch,
        &mut out,
    )
    .expect("forward should succeed");

    let ref_out = funcattn_reference(
        &x_basis,
        &x_value,
        &w_basis,
        &w_q,
        &w_k,
        &w_v,
        n,
        d,
        k,
        basis,
        alpha,
        temperature,
    );
    (out, ref_out)
}

// ── Cross-check against reference (the most important correctness gate) ─

#[test]
fn matches_reference_sigmoid() {
    let (out, ref_out) = run_forward(16, 8, 4, 0.5, 0.5, FuncAttnBasis::Sigmoid, 42);
    let err = frobenius(&out, &ref_out) / frobenius(&ref_out, &vec![0.0; ref_out.len()]).max(1e-30);
    assert!(
        err < 1e-3,
        "sigmoid forward disagrees with reference: relative error = {err}"
    );
}

#[test]
fn matches_reference_softmax() {
    let (out, ref_out) = run_forward(16, 8, 4, 0.5, 0.5, FuncAttnBasis::Softmax, 142);
    let err = frobenius(&out, &ref_out) / frobenius(&ref_out, &vec![0.0; ref_out.len()]).max(1e-30);
    assert!(
        err < 1e-3,
        "softmax forward disagrees with reference: relative error = {err}"
    );
}

#[test]
fn matches_reference_extreme_alpha() {
    // α near 0 (almost pure K̃ᵀK̃) and α near 1 (almost pure I) — both must work.
    for &alpha in &[0.01f32, 0.99] {
        let (out, ref_out) = run_forward(
            12,
            6,
            4,
            alpha,
            0.5,
            FuncAttnBasis::Sigmoid,
            999 + alpha.to_bits() as u64,
        );
        let err =
            frobenius(&out, &ref_out) / frobenius(&ref_out, &vec![0.0; ref_out.len()]).max(1e-30);
        assert!(
            err < 1e-3,
            "α={alpha}: forward disagrees with reference: relative error = {err}"
        );
    }
}

#[test]
fn matches_reference_temperature_sweep() {
    for &temp in &[0.1f32, 0.5, 1.0, 5.0] {
        let (out, ref_out) = run_forward(
            12,
            6,
            4,
            0.5,
            temp,
            FuncAttnBasis::Sigmoid,
            7000 + (temp * 100.0) as u64,
        );
        let err =
            frobenius(&out, &ref_out) / frobenius(&ref_out, &vec![0.0; ref_out.len()]).max(1e-30);
        assert!(
            err < 1e-3,
            "τ={temp}: forward disagrees with reference: relative error = {err}"
        );
    }
}

// ── G1: Mechanics (finite output, no NaN/Inf) ──────────────────

#[test]
fn g1_finite_output_random_inputs() {
    let n = 64;
    let d = 32;
    let k = 8;
    let mut x_basis = vec![0.0f32; n * d];
    let mut x_value = vec![0.0f32; n * d];
    let mut w_basis = vec![0.0f32; k * d];
    let mut w_q = vec![0.0f32; d * d];
    let mut w_k = vec![0.0f32; d * d];
    let mut w_v = vec![0.0f32; d * d];
    fill_rand(&mut x_basis, 12345);
    fill_rand(&mut x_value, 12346);
    fill_rand(&mut w_basis, 12347);
    fill_rand(&mut w_q, 12348);
    fill_rand(&mut w_k, 12349);
    fill_rand(&mut w_v, 12350);

    let cfg = FuncAttnConfig {
        d,
        k,
        basis: FuncAttnBasis::Sigmoid,
        alpha: 0.5,
        temperature: 0.5,
        cholesky_jitter: 1e-6,
    };
    let mut scratch = FuncAttnScratch::new(n, d, k);
    let mut out = vec![0.0f32; n * d];
    funcattn_forward(
        &x_basis,
        &x_value,
        &w_basis,
        &w_q,
        &w_k,
        &w_v,
        &cfg,
        &mut scratch,
        &mut out,
    )
    .expect("forward");
    for x in &out {
        assert!(x.is_finite(), "non-finite output: {x}");
    }
}

#[test]
fn g1_sweep_input_norm_and_alpha() {
    // Sweep B ∈ {1, 10, 100} and α ∈ {0.01, 0.5, 0.99}; assert finite output.
    // Unlike the additive-λ primal form, the convex combo α∈(0,1) guarantees
    // PD for any input scale — no NotPositiveDefinite expected.
    let n = 32;
    let d = 16;
    let k = 4;
    for (b_idx, &b_scale) in [1.0f32, 10.0, 100.0].iter().enumerate() {
        for (a_idx, &alpha) in [0.01f32, 0.5, 0.99].iter().enumerate() {
            let seed = 1000 + b_idx as u64 * 10 + a_idx as u64;
            let mut x_basis = vec![0.0f32; n * d];
            let mut x_value = vec![0.0f32; n * d];
            let mut w_basis = vec![0.0f32; k * d];
            let mut w_q = vec![0.0f32; d * d];
            let mut w_k = vec![0.0f32; d * d];
            let mut w_v = vec![0.0f32; d * d];
            fill_rand(&mut x_basis, seed);
            fill_rand(&mut x_value, seed + 1);
            fill_rand(&mut w_basis, seed + 2);
            fill_rand(&mut w_q, seed + 3);
            fill_rand(&mut w_k, seed + 4);
            fill_rand(&mut w_v, seed + 5);
            for x in x_basis.iter_mut() {
                *x *= b_scale;
            }
            for x in x_value.iter_mut() {
                *x *= b_scale;
            }

            let cfg = FuncAttnConfig {
                d,
                k,
                basis: FuncAttnBasis::Sigmoid,
                alpha,
                temperature: 0.5,
                cholesky_jitter: 1e-6,
            };
            let mut scratch = FuncAttnScratch::new(n, d, k);
            let mut out = vec![0.0f32; n * d];
            funcattn_forward(
                &x_basis,
                &x_value,
                &w_basis,
                &w_q,
                &w_k,
                &w_v,
                &cfg,
                &mut scratch,
                &mut out,
            )
            .expect("convex combo should be PD for any α ∈ (0, 1)");
            for x in &out {
                assert!(x.is_finite(), "non-finite output at B={b_scale}, α={alpha}");
            }
        }
    }
}

#[test]
fn g1_lipschitz_bounded() {
    // Verify empirical Lipschitz constant is finite and reasonable.
    // Prop 4.5 is stated for the additive-λ form; for the convex-combo form
    // the bound becomes a function of α/(1-α) instead. We just check finiteness.
    let n = 64;
    let d = 32;
    let k = 8;
    let mut x_basis = vec![0.0f32; n * d];
    let mut x_value = vec![0.0f32; n * d];
    let mut w_basis = vec![0.0f32; k * d];
    let mut w_q = vec![0.0f32; d * d];
    let mut w_k = vec![0.0f32; d * d];
    let mut w_v = vec![0.0f32; d * d];
    fill_rand(&mut x_basis, 12345);
    fill_rand(&mut x_value, 12346);
    fill_rand(&mut w_basis, 12347);
    fill_rand(&mut w_q, 12348);
    fill_rand(&mut w_k, 12349);
    fill_rand(&mut w_v, 12350);

    let cfg = FuncAttnConfig {
        d,
        k,
        basis: FuncAttnBasis::Sigmoid,
        alpha: 0.5,
        temperature: 0.5,
        cholesky_jitter: 1e-6,
    };
    let mut scratch = FuncAttnScratch::new(n, d, k);
    let mut out = vec![0.0f32; n * d];
    funcattn_forward(
        &x_basis,
        &x_value,
        &w_basis,
        &w_q,
        &w_k,
        &w_v,
        &cfg,
        &mut scratch,
        &mut out,
    )
    .expect("forward");

    // Perturb x_basis by ‖Δ‖ = 1 and check ‖A(X+Δ) − A(X)‖ is finite.
    let mut delta = vec![0.0f32; n * d];
    fill_rand(&mut delta, 12345 + 100);
    let d_norm = frobenius(&delta, &vec![0.0; n * d]);
    for x in delta.iter_mut() {
        *x /= d_norm.max(1e-30);
    }
    let mut x_pert = x_basis.clone();
    for i in 0..n * d {
        x_pert[i] += delta[i];
    }
    let mut out_pert = vec![0.0f32; n * d];
    funcattn_forward(
        &x_pert,
        &x_value,
        &w_basis,
        &w_q,
        &w_k,
        &w_v,
        &cfg,
        &mut scratch,
        &mut out_pert,
    )
    .expect("perturbed forward");

    let lip = frobenius(&out, &out_pert);
    assert!(lip.is_finite(), "Lipschitz ratio not finite");
    // Empirically this is ~1-50 for random normalized inputs at α=0.5.
    assert!(lip < 1.0e6, "empirical Lipschitz too large: {lip}");
}

// ── Partition-of-unity check ──────────────────────────────────

#[test]
fn basis_rows_partition_of_unity() {
    let n = 8;
    let d = 16;
    let k = 6;
    let mut x = vec![0.0f32; n * d];
    let mut w = vec![0.0f32; k * d];
    fill_rand(&mut x, 7);
    fill_rand(&mut w, 8);
    let mut out = vec![0.0f32; n * k];

    for &kind in &[FuncAttnBasis::Softmax, FuncAttnBasis::Sigmoid] {
        for &temp in &[0.1f32, 0.5, 1.0, 5.0] {
            compute_basis_into(&x, &w, &[], n, d, k, kind, temp, &mut out);
            for i in 0..n {
                let row = &out[i * k..(i + 1) * k];
                let sum: f32 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "row {i} doesn't sum to 1 for {kind:?} τ={temp}: sum = {sum}"
                );
                for &v in row {
                    assert!(v >= 0.0, "negative basis entry for {kind:?} τ={temp}");
                }
            }
        }
    }
}

// ── Cholesky unit tests ────────────────────────────────────────

#[test]
fn cholesky_inplace_basic_spd() {
    // A = [[4, 2], [2, 3]] is SPD with L = [[2, 0], [1, √2]] (lower triangular).
    // Stored row-major: [L[0,0], L[0,1], L[1,0], L[1,1]] = [2, 0, 1, √2].
    let mut a = vec![4.0f32, 2.0, 2.0, 3.0];
    assert!(cholesky_inplace(&mut a, 2));
    assert!((a[0] - 2.0).abs() < 1e-5, "L[0,0] = {}", a[0]);
    assert!(
        a[1].abs() < 1e-20,
        "L[0,1] upper tri must be zero, got {}",
        a[1]
    );
    assert!((a[2] - 1.0).abs() < 1e-5, "L[1,0] = {}", a[2]);
    assert!((a[3] - 2.0f32.sqrt()).abs() < 1e-5, "L[1,1] = {}", a[3]);
}

#[test]
fn cholesky_inplace_indefinite_fails() {
    let mut a = vec![1.0f32, 2.0, 2.0, 1.0]; // indefinite
    assert!(!cholesky_inplace(&mut a, 2));
}

#[test]
fn cholesky_solve_known_system() {
    // A = [[4, 2], [2, 3]], b = [1, 1]; solution x = [1/8, 1/4]
    let mut a = vec![4.0f32, 2.0, 2.0, 3.0];
    assert!(cholesky_inplace(&mut a, 2));
    let b = vec![1.0f32, 1.0];
    let mut y = vec![0.0f32; 2];
    let mut x = vec![0.0f32; 2];
    cholesky_solve_into(&a, &b, 2, &mut y, &mut x);
    assert!((x[0] - 0.125).abs() < 1e-5, "x[0] = {}", x[0]);
    assert!((x[1] - 0.25).abs() < 1e-5, "x[1] = {}", x[1]);
}

// ── Larger-size sanity (catches indexing bugs, partial G4 smoke) ─

#[test]
fn forward_large_n_smoke() {
    // n=2048, k=16, d=64 — forward should complete without index errors or NaN.
    // Full G4 timing is in the bench file.
    let n = 2048;
    let d = 64;
    let k = 16;
    let mut x_basis = vec![0.0f32; n * d];
    let mut x_value = vec![0.0f32; n * d];
    let mut w_basis = vec![0.0f32; k * d];
    let mut w_q = vec![0.0f32; d * d];
    let mut w_k = vec![0.0f32; d * d];
    let mut w_v = vec![0.0f32; d * d];
    fill_rand(&mut x_basis, 9001);
    fill_rand(&mut x_value, 9002);
    fill_rand(&mut w_basis, 9003);
    fill_rand(&mut w_q, 9004);
    fill_rand(&mut w_k, 9005);
    fill_rand(&mut w_v, 9006);

    let cfg = FuncAttnConfig {
        d,
        k,
        basis: FuncAttnBasis::Sigmoid,
        alpha: 0.5,
        temperature: 0.5,
        cholesky_jitter: 1e-6,
    };
    let mut scratch = FuncAttnScratch::new(n, d, k);
    let mut out = vec![0.0f32; n * d];
    funcattn_forward(
        &x_basis,
        &x_value,
        &w_basis,
        &w_q,
        &w_k,
        &w_v,
        &cfg,
        &mut scratch,
        &mut out,
    )
    .expect("forward at n=2048");
    for x in &out {
        assert!(x.is_finite(), "non-finite at large n");
    }
}

// ── Degenerate-input guard ─────────────────────────────────────

#[test]
fn forward_zero_weights_alpha_positive_succeeds() {
    // All-zero w_k → K̃ all zero → reg = α·I (well-conditioned for α > 0).
    // Convex combo guarantees PD; output should be finite (possibly zero).
    let n = 4;
    let d = 8;
    let k = 4;
    let x_basis = vec![0.5f32; n * d];
    let x_value = vec![0.5f32; n * d];
    let w_basis = vec![0.1f32; k * d]; // non-zero so Φ isn't 0/0
    let w_q = vec![0.0f32; d * d];
    let w_k = vec![0.0f32; d * d];
    let w_v = vec![0.0f32; d * d];

    let cfg = FuncAttnConfig {
        d,
        k,
        basis: FuncAttnBasis::Sigmoid,
        alpha: 0.5,
        temperature: 0.5,
        cholesky_jitter: 1e-6,
    };
    let mut scratch = FuncAttnScratch::new(n, d, k);
    let mut out = vec![0.0f32; n * d];
    let res = funcattn_forward(
        &x_basis,
        &x_value,
        &w_basis,
        &w_q,
        &w_k,
        &w_v,
        &cfg,
        &mut scratch,
        &mut out,
    );
    assert!(
        res.is_ok(),
        "convex combo should succeed with α > 0 even for zero K̃"
    );
    for x in &out {
        assert!(x.is_finite(), "non-finite output for zero w_k");
    }
}

// ── pre_rotate_basis_weights_into (Plan 286 T5.1) ─────────────

/// `d × d` identity matrix, row-major.
fn identity_matrix(d: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

/// `d × d` row-orthonormal matrix via Gram-Schmidt on random rows.
fn random_orthonormal_rows(d: usize, seed: u64) -> Vec<f32> {
    let mut m = vec![0.0f32; d * d];
    fill_rand(&mut m, seed);
    for i in 0..d {
        for j in 0..i {
            let dot = (0..d).map(|c| m[i * d + c] * m[j * d + c]).sum::<f32>();
            for c in 0..d {
                m[i * d + c] -= dot * m[j * d + c];
            }
        }
        let nrm = (0..d)
            .map(|c| m[i * d + c] * m[i * d + c])
            .sum::<f32>()
            .sqrt();
        let nrm = if nrm < 1e-12 { 1.0 } else { nrm };
        for c in 0..d {
            m[i * d + c] /= nrm;
        }
    }
    m
}

#[test]
fn pre_rotate_identity_eigenvectors_is_noop() {
    // V = I → W · Iᵀ = W (unchanged).
    let k = 4;
    let d = 8;
    let mut w_basis = vec![0.0f32; k * d];
    fill_rand(&mut w_basis, 4242);
    let original = w_basis.clone();
    let identity = identity_matrix(d);
    pre_rotate_basis_weights_into(&mut w_basis, &identity, k, d);
    let diff = frobenius(&w_basis, &original);
    assert!(
        diff < 1e-5,
        "identity rotation should be no-op: diff = {diff}"
    );
}

#[test]
fn pre_rotate_preserves_row_norms() {
    // V orthogonal → ‖V · w_row‖ = ‖w_row‖ for every row.
    let k = 4;
    let d = 8;
    let mut w_basis = vec![0.0f32; k * d];
    fill_rand(&mut w_basis, 4243);
    let original_norms: Vec<f32> = (0..k)
        .map(|r| {
            (0..d)
                .map(|c| w_basis[r * d + c] * w_basis[r * d + c])
                .sum::<f32>()
                .sqrt()
        })
        .collect();
    let v = random_orthonormal_rows(d, 17);
    pre_rotate_basis_weights_into(&mut w_basis, &v, k, d);
    for kk in 0..k {
        let new_norm = (0..d)
            .map(|c| w_basis[kk * d + c] * w_basis[kk * d + c])
            .sum::<f32>()
            .sqrt();
        assert!(
            (new_norm - original_norms[kk]).abs() < 1e-4,
            "row {} norm changed: {} → {}",
            kk,
            original_norms[kk],
            new_norm
        );
    }
}

#[test]
fn pre_rotate_preserves_orthogonality_of_w_basis() {
    // If w_basis rows are orthonormal and V is orthogonal, then W · Vᵀ is
    // still row-orthonormal (W · Wᵀ unchanged).
    let k = 4;
    let d = 8;
    let w_basis = random_orthonormal_rows(d, 31); // d×d, take first k rows
    let mut w_basis = w_basis.into_iter().take(k * d).collect::<Vec<_>>();
    let v = random_orthonormal_rows(d, 32);
    pre_rotate_basis_weights_into(&mut w_basis, &v, k, d);

    // Check the k×k Gram matrix is still identity.
    for a in 0..k {
        for b in 0..k {
            let dot = (0..d)
                .map(|c| w_basis[a * d + c] * w_basis[b * d + c])
                .sum::<f32>();
            let expected = if a == b { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-3,
                "Gram[{a},{b}] = {dot} (want {expected})"
            );
        }
    }
}

#[test]
fn pre_rotate_forward_output_still_finite_and_partition_of_unity() {
    // Sanity: after rotation, the forward pass still produces finite output
    // and the basis rows still partition to 1 (verify the rotation doesn't
    // break the partition-of-unity invariant that Prop 4.3 relies on).
    let n = 32;
    let d = 8;
    let k = 4;
    let mut w_basis = random_orthonormal_rows(d, 41)
        .into_iter()
        .take(k * d)
        .collect::<Vec<_>>();
    let w_q = random_orthonormal_rows(d, 42);
    let w_k = random_orthonormal_rows(d, 43);
    let w_v = random_orthonormal_rows(d, 44);
    let v = random_orthonormal_rows(d, 45);
    pre_rotate_basis_weights_into(&mut w_basis, &v, k, d);

    // Verify partition-of-unity directly via compute_basis_into.
    let mut x = vec![0.0f32; n * d];
    fill_rand(&mut x, 99);
    let mut phi = vec![0.0f32; n * k];
    compute_basis_into(
        &x,
        &w_basis,
        &[],
        n,
        d,
        k,
        FuncAttnBasis::Sigmoid,
        0.1,
        &mut phi,
    );
    for i in 0..n {
        let row_sum: f32 = phi[i * k..(i + 1) * k].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-4,
            "row {i} sum = {row_sum} (want 1.0)"
        );
    }

    // Forward still finite.
    let cfg = FuncAttnConfig {
        d,
        k,
        basis: FuncAttnBasis::Sigmoid,
        alpha: 0.5,
        temperature: 0.1,
        cholesky_jitter: 1e-6,
    };
    let mut scratch = FuncAttnScratch::new(n, d, k);
    let mut out = vec![0.0f32; n * d];
    funcattn_forward(
        &x,
        &x,
        &w_basis,
        &w_q,
        &w_k,
        &w_v,
        &cfg,
        &mut scratch,
        &mut out,
    )
    .expect("forward after rotation");
    for v in &out {
        assert!(v.is_finite(), "non-finite output after eigen-rotation");
    }
}

// ── Plan 332 — Principled structured basis constructors ─────────

/// Verify `W·W^T ≈ I_k` (rows are orthonormal) for a constructed basis.
fn check_row_orthonormal(w: &[f32], k: usize, d: usize, tol: f32, label: &str) {
    assert_eq!(w.len(), k * d, "{label}: wrong length");
    for i in 0..k {
        // Diagonal: row norm should be 1.
        let mut norm_sq = 0.0f32;
        for j in 0..d {
            norm_sq += w[i * d + j] * w[i * d + j];
        }
        assert!(
            (norm_sq - 1.0).abs() < tol,
            "{label}: row {i} norm^2 = {norm_sq}, expected 1.0 (tol {tol})"
        );
        // Off-diagonal: orthogonal to all earlier rows.
        for j in 0..i {
            let mut dot = 0.0f32;
            for l in 0..d {
                dot += w[i * d + l] * w[j * d + l];
            }
            assert!(
                dot.abs() < tol,
                "{label}: rows ({i},{j}) dot = {dot}, expected 0 (tol {tol})"
            );
        }
    }
}

/// T1.1 unit test: DCT-log basis is row-orthonormal.
#[test]
fn dct_log_basis_is_row_orthonormal() {
    for &(k, d) in &[(1usize, 8usize), (4, 16), (8, 64), (16, 64), (8, 128)] {
        let w = make_dct_log_basis(k, d);
        check_row_orthonormal(&w, k, d, 1e-5, &format!("DCT-log k={k} d={d}"));
    }
}

/// T1.1 unit test: DCT-log basis covers log-spaced frequencies.
///
/// We verify by reconstructing the per-row dominant frequency from the
/// zero-crossing count of the post-Gram-Schmidt rows. The i-th row's
/// dominant frequency should be monotonically non-decreasing in i.
#[test]
fn dct_log_basis_covers_log_spaced_frequencies() {
    let (k, d) = (8, 64);
    let w = make_dct_log_basis(k, d);
    // Count sign changes in each row interior as a proxy for frequency.
    let mut sign_changes = Vec::with_capacity(k);
    for i in 0..k {
        let row = &w[i * d..(i + 1) * d];
        let mut count = 0usize;
        for j in 1..d {
            if row[j - 1].signum() != row[j].signum() && row[j] != 0.0 {
                count += 1;
            }
        }
        sign_changes.push(count);
    }
    // Coarsest row (i=0, f=1) should have ~2 sign changes (one full cycle);
    // finest row (i=k-1, f=d/2) should have many. Monotone non-decreasing.
    println!("DCT-log sign-change profile (k={k}, d={d}): {sign_changes:?}");
    assert!(
        sign_changes[0] <= sign_changes[k - 1],
        "DCT-log should span coarse→fine: first={}, last={}",
        sign_changes[0],
        sign_changes[k - 1]
    );
    // The coarsest row must have meaningfully fewer sign changes than the
    // finest (otherwise we didn't actually span log-spaced frequencies).
    assert!(
        sign_changes[k - 1] >= 2 * sign_changes[0],
        "DCT-log frequency spread too narrow: first={}, last={}",
        sign_changes[0],
        sign_changes[k - 1]
    );
}

/// T1.2 unit test: Haar-packet basis is row-orthonormal.
#[test]
fn haar_packet_basis_is_row_orthonormal() {
    for &(k, d) in &[(1usize, 8usize), (4, 16), (8, 64), (16, 64), (7, 128)] {
        let w = make_haar_packet_basis(k, d);
        check_row_orthonormal(&w, k, d, 1e-5, &format!("Haar-packet k={k} d={d}"));
    }
}

/// T1.2 unit test: Haar-packet basis spans multiple scales.
///
/// Row 0 must be the DC component (constant sign — zero sign changes).
/// Later rows must have progressively more localized support: sign changes
/// increase as we move to finer-scale wavelets.
#[test]
fn haar_packet_basis_spans_multiple_scales() {
    let (k, d) = (8, 64);
    let w = make_haar_packet_basis(k, d);

    // Row 0: DC = constant sign (no interior sign changes).
    let dc_row = &w[0..d];
    let dc_sign = dc_row[0].signum();
    for &v in dc_row {
        assert_eq!(v.signum(), dc_sign, "Haar DC row should be constant-sign");
    }

    // Each subsequent row should have at least one sign change (it's a
    // wavelet, not a scaling function). Count sign changes per row.
    let mut sign_counts = Vec::with_capacity(k - 1);
    for i in 1..k {
        let row = &w[i * d..(i + 1) * d];
        let mut count = 0usize;
        for j in 1..d {
            if row[j - 1].signum() != row[j].signum() && row[j] != 0.0 {
                count += 1;
            }
        }
        assert!(
            count >= 1,
            "Haar row {i} should have ≥1 sign change (got {count})"
        );
        sign_counts.push(count);
    }
    println!("Haar-packet sign-change profile (k={k}, d={d}): {sign_counts:?}");
    // The coarsest wavelet (row 1, support=d) has exactly 1 sign change.
    assert_eq!(
        sign_counts[0], 1,
        "Haar row 1 (coarsest wavelet) should have exactly 1 sign change"
    );
}

/// T1.1+T1.2 cross-check: both bases plug into `funcattn_forward` cleanly
/// (G3 — drop-in replacement sanity). Output must be finite + partition of
/// unity on Φ (the existing forward-pass invariant).
#[test]
fn structured_bases_forward_pass_clean() {
    let (n, d, k) = (12, 64, 8);
    let cfg = FuncAttnConfig {
        d,
        k,
        basis: FuncAttnBasis::Sigmoid,
        alpha: 0.5,
        temperature: 0.5,
        cholesky_jitter: 1e-6,
    };
    // Random input (deterministic).
    let mut x = vec![0.0f32; n * d];
    let mut rng = make_rng(12345);
    for v in x.iter_mut() {
        *v = rng.next().unwrap_or(0.0);
    }
    let w_q = identity_matrix(d);
    let w_k = identity_matrix(d);
    let w_v = identity_matrix(d);

    for (label, w_basis) in [
        ("DCT-log", make_dct_log_basis(k, d)),
        ("Haar-packet", make_haar_packet_basis(k, d)),
    ] {
        let mut scratch = FuncAttnScratch::new(n, d, k);
        let mut out = vec![0.0f32; n * d];
        funcattn_forward(
            &x,
            &x,
            &w_basis,
            &w_q,
            &w_k,
            &w_v,
            &cfg,
            &mut scratch,
            &mut out,
        )
        .unwrap_or_else(|e| panic!("{label}: forward failed: {e:?}"));
        for v in &out {
            assert!(v.is_finite(), "{label}: non-finite forward output");
        }
        // Φ partition-of-unity (compute_basis_into separately for clarity).
        let mut phi = vec![0.0f32; n * k];
        compute_basis_into(
            &x,
            &w_basis,
            &[],
            n,
            d,
            k,
            FuncAttnBasis::Sigmoid,
            0.5,
            &mut phi,
        );
        for i in 0..n {
            let row_sum: f32 = phi[i * k..(i + 1) * k].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "{label}: Φ row {i} sum = {row_sum}, expected 1.0"
            );
        }
    }
}
