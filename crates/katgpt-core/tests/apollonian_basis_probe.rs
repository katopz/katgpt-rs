//! Probe: does a STRUCTURED basis produce a materially different Φ than
//! random-orthogonal? This tests the T5.1 "rotation invariance" claim.
//!
//! T5.1 found that PCA-eigenbasis pre-rotation was 17-25% worse. The plan's
//! explanation was "row-normalization is invariant to basis direction". But if
//! w_basis is random-orthogonal and V (eigenvectors) is orthogonal, then
//! W·V^T is ALSO random-orthogonal — so PCA pre-rotation just gives another
//! random orthogonal basis with no expected change. The T5.1 result would then
//! be statistical noise, not invariance.
//!
//! This probe distinguishes the two hypotheses:
//!   H_invariance: Φ is ~the same regardless of which orthogonal W you pick
//!   H_structure:  a structured W (aligned to input signal) produces a
//!                 materially different (and possibly better) Φ
//!
//! If H_invariance holds, Apollonian (or any structured basis) is doomed.
//! If H_structure holds, the T5.1 conclusion was overstated and structured
//! bases (incl. Apollonian) deserve a real test.

#![cfg(feature = "funcattn")]

use katgpt_core::funcattn::{
    compute_basis_into, funcattn_forward, FuncAttnBasis, FuncAttnConfig, FuncAttnScratch,
};

/// L2 normalize a vector in place.
fn l2_normalize(v: &mut [f32]) {
    let mut s = 0.0f32;
    for &x in v.iter() {
        s += x * x;
    }
    let norm = s.sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// Gram-Schmidt orthogonalize the rows of `w` (k rows, d cols, row-major).
/// Produces a row-orthogonal matrix (rows are orthonormal).
fn gram_schmidt_rows(w: &mut [f32], k: usize, d: usize) {
    for i in 0..k {
        // Subtract projections onto previous rows.
        for j in 0..i {
            let mut dot = 0.0f32;
            for l in 0..d {
                dot += w[i * d + l] * w[j * d + l];
            }
            for l in 0..d {
                w[i * d + l] -= dot * w[j * d + l];
            }
        }
        l2_normalize(&mut w[i * d..(i + 1) * d]);
    }
}

/// Cosine similarity between two flattened vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

/// Effective rank of Φ: exp(entropy of the average row distribution.
/// High effective rank = Φ rows are spread across basis dims (good, expressive).
/// Low effective rank = Φ rows are concentrated on few dims (saturated).
fn effective_rank(phi: &[f32], n: usize, k: usize) -> f32 {
    // Average column distribution: p[g] = (1/n) Σ_i Φ[i,g]. Σ_g p[g] = 1.
    let mut p = vec![0.0f32; k];
    for i in 0..n {
        for g in 0..k {
            p[g] += phi[i * k + g];
        }
    }
    let mut entropy = 0.0f32;
    let nf = n as f32;
    for &pg in p.iter() {
        if pg > 1e-12 {
            let q = pg / nf;
            entropy -= q * q.ln();
        }
    }
    entropy.exp()
}

/// Mean per-row "sharpness": how far is each Φ row from uniform (1/k)?
/// 0.0 = perfectly uniform (no information), higher = sharper/more discriminative.
fn mean_sharpness(phi: &[f32], n: usize, k: usize) -> f32 {
    let uniform = 1.0 / k as f32;
    let mut sum = 0.0f32;
    for i in 0..n {
        let mut dev = 0.0f32;
        for g in 0..k {
            let d = phi[i * k + g] - uniform;
            dev += d * d;
        }
        sum += dev.sqrt();
    }
    sum / n as f32
}

const D: usize = 64;
const N: usize = 20;
const K: usize = 8;

/// Build input X with known multi-scale structure: each token is a random
/// combination of `n_scales` sinusoids at frequencies freq[0..n_scales].
/// The "signal subspace" is the span of the top n_scales PCA directions.
fn make_multiscale_x(seed: u64, n_scales: usize) -> (Vec<f32>, Vec<f32>) {
    // Deterministic LCG for reproducibility.
    let mut s = seed;
    let mut rng = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (1u64 << 31) as f32 - 0.5
    };
    let freqs: Vec<f32> = (0..n_scales).map(|i| 0.3 + 0.7 * (i as f32)).collect();
    let mut x = vec![0.0f32; N * D];
    for i in 0..N {
        let t = i as f32;
        for j in 0..D {
            let mut v = 0.0f32;
            for (si, &f) in freqs.iter().enumerate() {
                let amp = 1.0 / (si + 1) as f32; // higher scales have less energy
                v += amp * (f * (t + 0.1 * j as f32) * (si as f32 + 1.0)).sin();
            }
            x[i * D + j] = v + 0.05 * rng(); // small noise
        }
    }
    // Return (x, signal_directions) — directions are the frequency phase vectors.
    // For a proper structured basis we'd want the PCA eigenvectors, but for the
    // probe we use the known generative structure: each "scale" corresponds to
    // a phase ramp across the d=64 dims.
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

/// Build a random row-orthonormal w_basis (k, d).
fn random_orthonormal_w(seed: u64) -> Vec<f32> {
    random_orthonormal_w_rect(seed, K, D)
}

/// Build a random row-orthonormal matrix (k rows, d cols, row-major).
/// Requires k <= d for row-orthonormality to be achievable.
fn random_orthonormal_w_rect(seed: u64, k: usize, d: usize) -> Vec<f32> {
    let mut s = seed;
    let mut rng = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (1u64 << 31) as f32 - 0.5
    };
    let mut w = vec![0.0f32; k * d];
    for w_i in w.iter_mut() {
        *w_i = rng();
    }
    gram_schmidt_rows(&mut w, k, d);
    w
}

/// Build a STRUCTURED w_basis: first `n_scales` rows align with the signal
/// directions, remaining rows are random orthogonal complement.
fn structured_w(signal_dirs: &[f32], n_scales: usize, seed: u64) -> Vec<f32> {
    let mut w = vec![0.0f32; K * D];
    // First n_scales rows = signal directions (already L2-normalized).
    for si in 0..n_scales.min(K) {
        w[si * D..(si + 1) * D].copy_from_slice(&signal_dirs[si * D..(si + 1) * D]);
    }
    // Remaining rows = random.
    let mut s = seed;
    let mut rng = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (1u64 << 31) as f32 - 0.5
    };
    for w_i in w[(n_scales.min(K)) * D..].iter_mut() {
        *w_i = rng();
    }
    // Re-orthogonalize all rows (signal dirs stay ~intact since they're orthogonal
    // by construction of distinct frequencies, random rows get orthogonalized
    // against them and each other).
    gram_schmidt_rows(&mut w, K, D);
    w
}

#[test]
fn probe_phi_sensitivity_to_basis_choice() {
    let (x, signal_dirs) = make_multiscale_x(42, 4);
    let _cfg = FuncAttnConfig {
        d: D,
        k: K,
        basis: FuncAttnBasis::Sigmoid,
        temperature: 0.5, // default
        alpha: 0.5,
        cholesky_jitter: 1e-6,
    };

    // --- Random orthogonal basis (the T5.1 baseline) ---
    let w_rand = random_orthonormal_w(100);
    let mut phi_rand = vec![0.0f32; N * K];
    compute_basis_into(
        &x, &w_rand, &[], N, D, K, FuncAttnBasis::Sigmoid, 0.5, &mut phi_rand,
    );

    // --- Structured basis (aligned to the 4 signal scales) ---
    let w_struct = structured_w(&signal_dirs, 4, 200);
    let mut phi_struct = vec![0.0f32; N * K];
    compute_basis_into(
        &x, &w_struct, &[], N, D, K, FuncAttnBasis::Sigmoid, 0.5, &mut phi_struct,
    );

    // --- A SECOND random orthogonal basis (to measure the noise floor) ---
    let w_rand2 = random_orthonormal_w(300);
    let mut phi_rand2 = vec![0.0f32; N * K];
    compute_basis_into(
        &x, &w_rand2, &[], N, D, K, FuncAttnBasis::Sigmoid, 0.5, &mut phi_rand2,
    );

    let cos_rand_rand = cosine(&phi_rand, &phi_rand2);
    let cos_rand_struct = cosine(&phi_rand, &phi_struct);
    let er_rand = effective_rank(&phi_rand, N, K);
    let er_struct = effective_rank(&phi_struct, N, K);
    let sh_rand = mean_sharpness(&phi_rand, N, K);
    let sh_struct = mean_sharpness(&phi_struct, N, K);

    println!("\n=== PROBE: Φ sensitivity to basis choice (τ=0.5) ===");
    println!("cos(Φ_rand1, Φ_rand2)   = {cos_rand_rand:.4}  <- noise floor (two random bases)");
    println!("cos(Φ_rand1, Φ_struct)  = {cos_rand_struct:.4}  <- structured vs random");
    println!("effective_rank(rand)     = {er_rand:.3}");
    println!("effective_rank(struct)   = {er_struct:.3}");
    println!("mean_sharpness(rand)     = {sh_rand:.4}");
    println!("mean_sharpness(struct)   = {sh_struct:.4}");

    // KEY ASSERTION: if H_invariance were true, cos(rand,struct) ≈ cos(rand,rand).
    // If H_structure holds, cos(rand,struct) should be MATERIALLY lower than
    // the noise floor — the structured basis produces a different Φ.
    println!(
        "\nΔ (noise_floor - struct_diff) = {:.4}",
        cos_rand_rand - cos_rand_struct
    );
    println!("If Δ > 0.05, structured basis materially changes Φ → H_structure.");
    println!("If Δ ≈ 0,     basis choice doesn't matter → H_invariance (T5.1 claim).");
}

#[test]
fn probe_transport_quality_structured_vs_random() {
    // The real test: does a structured basis improve the transport OUTPUT?
    //
    // IMPORTANT: FUNCATTN's output is `Φ · C · Ṽ` where Ṽ = w_v · (diag(col_sum))⁻¹ · Φᵀ · X.
    // This is LINEAR in X (Φ is a fixed nonlinear function of X, but the whole
    // operator is linear in the values once Φ is computed). So the target MUST
    // be a linear/smooth function of X for FUNCATTN to have any chance. A
    // nonlinear target (cos of sinusoid mix) is unreachable by construction and
    // would make both bases fail equally — which is what the first version of
    // this test showed and was a FLAW in the test, not evidence against bases.
    //
    // Corrected target: y[i,:] = smooth linear blend of X rows (a low-pass
    // filter across the sequence). This is exactly the kind of operator
    // FUNCATTN is designed for (functional correspondence between sequences).
    let (x, signal_dirs) = make_multiscale_x(42, 4);

    // Target: y[i] = weighted average of X[i-1], X[i], X[i+1] (a smoothing op).
    // This is a linear operator on X — representable by FUNCATTN. A good basis
    // that captures the multi-scale structure should reconstruct it better at
    // small k than a random basis.
    let mut y_target = vec![0.0f32; N * D];
    for i in 0..N {
        let prev = if i > 0 { i - 1 } else { 0 };
        let next = if i + 1 < N { i + 1 } else { N - 1 };
        for j in 0..D {
            y_target[i * D + j] = 0.25 * x[prev * D + j]
                + 0.5 * x[i * D + j]
                + 0.25 * x[next * D + j];
        }
    }

    // Random-orthonormal Q/K projections; identity w_v so the transport must
    // discover the smoothing purely from the basis routing.
    let w_q = random_orthonormal_w_rect(999, D, D);
    let w_k = random_orthonormal_w_rect(888, D, D);
    let mut w_v = vec![0.0f32; D * D];
    for i in 0..D {
        w_v[i * D + i] = 1.0; // identity: pass values through
    }

    let run = |w_basis: &[f32], tau: f32, label: &str| -> f32 {
        let cfg = FuncAttnConfig {
            d: D,
            k: K,
            basis: FuncAttnBasis::Sigmoid,
            temperature: tau,
            alpha: 0.5,
            cholesky_jitter: 1e-6,
        };
        let mut scratch = FuncAttnScratch::new(N, D, K);
        let mut out = vec![0.0f32; N * D];
        funcattn_forward(&x, &x, w_basis, &w_q, &w_k, &w_v, &cfg, &mut scratch, &mut out)
            .expect("forward");
        let cos = cosine(&out, &y_target);
        let mse: f32 = (0..N * D)
            .map(|i| {
                let d = out[i] - y_target[i];
                d * d
            })
            .sum::<f32>()
            / (N * D) as f32;
        println!("{label} (τ={tau}): cos(out, y) = {cos:+.4}, MSE = {mse:.4}");
        cos
    };

    println!("\n=== PROBE: transport quality (smoothing target, representable) ===");
    for tau in [0.5f32, 0.1] {
        let cos_rand = run(&random_orthonormal_w(100), tau, "random-orth   ");
        let cos_struct = run(&structured_w(&signal_dirs, 4, 200), tau, "structured    ");
        println!("  Δcos (struct - rand) = {:+.4}\n", cos_struct - cos_rand);
    }
}

#[test]
fn probe_sharp_temperature_makes_basis_choice_matter() {
    // T5.1 / Plan 286 noted sigmoid needs τ=0.1 (sharp) to work. At τ=0.5 the
    // sigmoid is in its flat regime and Φ ≈ uniform regardless of basis. This
    // probe re-runs the sensitivity test at τ=0.1 to see if basis choice matters
    // more when the sigmoid is actually discriminative.
    let (x, signal_dirs) = make_multiscale_x(42, 4);

    let w_rand = random_orthonormal_w(100);
    let w_struct = structured_w(&signal_dirs, 4, 200);
    let w_rand2 = random_orthonormal_w(300);

    for tau in [0.5f32, 0.1] {
        let mut phi_rand = vec![0.0f32; N * K];
        let mut phi_struct = vec![0.0f32; N * K];
        let mut phi_rand2 = vec![0.0f32; N * K];
        compute_basis_into(&x, &w_rand, &[], N, D, K, FuncAttnBasis::Sigmoid, tau, &mut phi_rand);
        compute_basis_into(&x, &w_struct, &[], N, D, K, FuncAttnBasis::Sigmoid, tau, &mut phi_struct);
        compute_basis_into(&x, &w_rand2, &[], N, D, K, FuncAttnBasis::Sigmoid, tau, &mut phi_rand2);

        let cr = cosine(&phi_rand, &phi_rand2);
        let cs = cosine(&phi_rand, &phi_struct);
        let sh_r = mean_sharpness(&phi_rand, N, K);
        let sh_s = mean_sharpness(&phi_struct, N, K);
        println!(
            "τ={tau}: cos(rand,rand2)={cr:.4}  cos(rand,struct)={cs:.4}  sharp rand={sh_r:.4} struct={sh_s:.4}"
        );
    }
}
