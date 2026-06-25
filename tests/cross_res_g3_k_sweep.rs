//! Cross-Resolution Spectral Transport — G3 k-sweep characterization
//! (Plan 310 T2.3).
//!
//! ## Purpose
//!
//! Sweep k ∈ {4, 8, 16, 32, 64} for the 64 → 256 transport and characterize
//! how reconstruction cos varies with rank. **No hard pass/fail** — this is a
//! characterization gate, not a kill switch. The output is recorded in Research
//! 291 §5.3 to document the recommended k per tier pair.
//!
//! ## Setup
//!
//! - Band-limited src_state at three band-fractions (0.7, 0.85, 0.95) — models
//!   personalities of varying low-rank-ness.
//! - For each (k, band_frac) combo, transport 64 → 256 → 64, measure round-trip
//!   cosine similarity over 50 samples.
//!
//! ## Output
//!
//! A 2D table (k × band_frac) of mean cosines. The "elbow" — the smallest k
//! that achieves cos ≥ 0.90 for a target band_frac — is the recommended k for
//! that personality class. Documented in Research 291 §5.3 after running.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features cross_resolution_transport --release \
//!   --test cross_res_g3_k_sweep -- --nocapture
//! ```

#![cfg(feature = "cross_resolution_transport")]

use katgpt_core::cross_resolution::{
    CrossResolutionBases, CrossResScratch, transport_cross_resolution_into,
};
use katgpt_core::simd;

const D_SRC: usize = 64;
const D_DST: usize = 256;
const N_SAMPLES: usize = 50;
const K_SWEEP: &[usize] = &[4, 8, 16, 32, 64];
const BAND_FRACS: &[f32] = &[0.70, 0.85, 0.95];
/// Intrinsic rank of the personality subspace (FIXED across the k-sweep).
/// Realistic deployment value — personality vectors are low-rank per Research
/// 257 §5.5. We construct band-limited samples in a fixed rank-8 subspace
/// (the "true" personality basis) and then sweep the transport rank k. When
/// k < INTRINSIC_K, transport cannot fully capture the personality → cos < 1.
/// When k ≥ INTRINSIC_K, transport is lossless on the band-limited part →
/// cos = sqrt(band_frac). The elbow is at k = INTRINSIC_K.
const INTRINSIC_K: usize = 8;

struct Rng {
    s: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { s: seed.max(1) }
    }
    fn next_f32(&mut self) -> f32 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        let bits = (self.s >> 11) as u32;
        let u01 = bits as f32 / u32::MAX as f32;
        u01 * 2.0 - 1.0
    }
}

#[allow(clippy::needless_range_loop)] // orthonormalization math, explicit indexing clearer
fn random_orthonormal(dim: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    assert!(k <= dim);
    let mut cols: Vec<Vec<f32>> = (0..k).map(|_| (0..dim).map(|_| rng.next_f32()).collect()).collect();
    for i in 0..k {
        for j in 0..i {
            let dot: f32 = cols[i].iter().zip(cols[j].iter()).map(|(a, b)| a * b).sum();
            for r in 0..dim {
                cols[i][r] -= dot * cols[j][r];
            }
        }
        let norm: f32 = cols[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
        for r in 0..dim {
            cols[i][r] *= inv;
        }
    }
    let mut m = vec![0.0f32; dim * k];
    for r in 0..dim {
        for c in 0..k {
            m[r * k + c] = cols[c][r];
        }
    }
    m
}

#[allow(clippy::needless_range_loop)] // orthonormalization math, explicit indexing clearer
fn bandlimited_sample(
    dim: usize,
    k: usize,
    basis: &[f32],
    band_frac: f32,
    rng: &mut Rng,
) -> Vec<f32> {
    let a: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
    let mut band_part = vec![0.0f32; dim];
    for r in 0..dim {
        let row = &basis[r * k..(r + 1) * k];
        band_part[r] = simd::simd_dot_f32(row, &a, k);
    }
    let band_norm = simd::simd_dot_f32(&band_part, &band_part, dim).sqrt();
    if band_norm > 1e-12 {
        for x in &mut band_part {
            *x /= band_norm;
        }
    }
    let mut noise = vec![0.0f32; dim];
    for x in &mut noise {
        *x = rng.next_f32();
    }
    let mut spectral = vec![0.0f32; k];
    for j in 0..k {
        let mut acc = 0.0f32;
        for r in 0..dim {
            acc += basis[r * k + j] * noise[r];
        }
        spectral[j] = acc;
    }
    for r in 0..dim {
        let row = &basis[r * k..(r + 1) * k];
        let proj = simd::simd_dot_f32(row, &spectral, k);
        noise[r] -= proj;
    }
    let noise_norm = simd::simd_dot_f32(&noise, &noise, dim).sqrt();
    if noise_norm > 1e-12 {
        for x in &mut noise {
            *x /= noise_norm;
        }
    }
    let sb = band_frac.sqrt();
    let sn = (1.0 - band_frac).max(0.0).sqrt();
    let mut v = vec![0.0f32; dim];
    for r in 0..dim {
        v[r] = sb * band_part[r] + sn * noise[r];
    }
    v
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot = simd::simd_dot_f32(a, b, a.len());
    let na = simd::simd_dot_f32(a, a, a.len()).sqrt();
    let nb = simd::simd_dot_f32(b, b, b.len()).sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

#[test]
#[allow(clippy::needless_range_loop)] // orthonormalization math, explicit indexing clearer
fn g3_k_sweep_characterization() {
    // Fixed personality basis (rank INTRINSIC_K) — the "true" subspace the
    // personalities live in. Independent of the transport rank k below.
    let mut basis_rng = Rng::new(0xC300_F1EDu64);
    let personality_basis = random_orthonormal(D_SRC, INTRINSIC_K, &mut basis_rng);

    println!("\nG3 K-SWEEP (d_src={}, d_dst={}, intrinsic_k={}, n_samples={} per cell)",
        D_SRC, D_DST, INTRINSIC_K, N_SAMPLES);
    println!("======================================================================");
    print!("{:<8}", "k\\bf");
    for &bf in BAND_FRACS {
        print!("{:>12}", format!("bf={bf:.2}"));
    }
    println!();
    println!("----------------------------------------------------------------------");

    // For each transport rank k, build transport bases (rank k) but generate
    // band-limited samples in the FIXED personality_basis (rank INTRINSIC_K).
    // When k < INTRINSIC_K, the transport basis cannot align with the full
    // personality subspace → cos drops. When k ≥ INTRINSIC_K, transport is
    // lossless on the personality subspace → cos = sqrt(bf).
    //
    // To make the alignment realistic (not adversarial), we build the transport
    // phi_src as a random rotation whose first INTRINSIC_K columns span a
    // random subspace of the personality subspace + its orthogonal complement.
    // For k < INTRINSIC_K, only k of the INTRINSIC_K personality directions
    // survive; for k ≥ INTRINSIC_K, all of them survive.
    for &k in K_SWEEP {
        let mut rng = Rng::new(0xC300_0000u64 + k as u64);
        // Build transport phi_src: first min(k, INTRINSIC_K) columns are random
        // linear combinations of personality_basis columns; remaining columns
        // (if k > INTRINSIC_K) are random orthonormal vectors in the complement.
        let n_from_personality = k.min(INTRINSIC_K);
        let n_from_complement = k - n_from_personality;
        // Random coefficients for the personality-subspace part.
        let mut phi_cols: Vec<Vec<f32>> = Vec::with_capacity(k);
        for _ in 0..n_from_personality {
            // Random linear combination of personality_basis columns.
            let mut coeffs = vec![0.0f32; INTRINSIC_K];
            for c in &mut coeffs {
                *c = rng.next_f32();
            }
            let mut col = vec![0.0f32; D_SRC];
            for r in 0..D_SRC {
                let row = &personality_basis[r * INTRINSIC_K..(r + 1) * INTRINSIC_K];
                col[r] = simd::simd_dot_f32(row, &coeffs, INTRINSIC_K);
            }
            phi_cols.push(col);
        }
        // Complement part: random vectors, projected out of personality subspace.
        for _ in 0..n_from_complement {
            let mut col = vec![0.0f32; D_SRC];
            for r in 0..D_SRC {
                col[r] = rng.next_f32();
            }
            // Project out personality subspace.
            let mut spectral = vec![0.0f32; INTRINSIC_K];
            for j in 0..INTRINSIC_K {
                let mut acc = 0.0f32;
                for r in 0..D_SRC {
                    acc += personality_basis[r * INTRINSIC_K + j] * col[r];
                }
                spectral[j] = acc;
            }
            for r in 0..D_SRC {
                let row = &personality_basis[r * INTRINSIC_K..(r + 1) * INTRINSIC_K];
                let proj = simd::simd_dot_f32(row, &spectral, INTRINSIC_K);
                col[r] -= proj;
            }
            phi_cols.push(col);
        }
        // Gram-Schmidt the phi_cols to make them orthonormal.
        for i in 0..k {
            for j in 0..i {
                let dot: f32 = phi_cols[i].iter().zip(phi_cols[j].iter()).map(|(a, b)| a * b).sum();
                for r in 0..D_SRC {
                    phi_cols[i][r] -= dot * phi_cols[j][r];
                }
            }
            let norm: f32 = phi_cols[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
            for r in 0..D_SRC {
                phi_cols[i][r] *= inv;
            }
        }
        // Pack to row-major (D_SRC × k).
        let mut phi_src = vec![0.0f32; D_SRC * k];
        for r in 0..D_SRC {
            for c in 0..k {
                phi_src[r * k + c] = phi_cols[c][r];
            }
        }
        let psi_dst = random_orthonormal(D_DST, k, &mut rng);
        let forward = CrossResolutionBases::new(phi_src, psi_dst, D_SRC, D_DST, k)
            .expect("forward bases should construct");
        let reverse = CrossResolutionBases::new(
            forward.psi_dst.clone(),
            forward.phi_src.clone(),
            D_DST,
            D_SRC,
            k,
        )
        .expect("reverse bases should construct");
        let mut scratch = CrossResScratch::new(k);

        print!("{:<8}", k);
        for &bf in BAND_FRACS {
            let mut sum_cos = 0.0f32;
            for _ in 0..N_SAMPLES {
                // Band-limited sample in the FIXED personality_basis, NOT the
                // transport phi_src — this is the key change from v1. The
                // transport rank k determines how much of the personality
                // subspace the transport can capture.
                let src = bandlimited_sample(D_SRC, INTRINSIC_K, &personality_basis, bf, &mut rng);
                let mut dst = vec![0.0f32; D_DST];
                transport_cross_resolution_into(&src, &forward, &mut scratch, &mut dst);
                let mut recon = vec![0.0f32; D_SRC];
                transport_cross_resolution_into(&dst, &reverse, &mut scratch, &mut recon);
                sum_cos += cosine(&src, &recon);
            }
            let mean = sum_cos / N_SAMPLES as f32;
            print!("{:>12.4}", mean);
        }
        println!();
    }
    println!("======================================================================");
    println!("\nInterpretation: when k < intrinsic_k={}, the transport basis cannot fully", INTRINSIC_K);
    println!("capture the personality subspace → cos drops. When k ≥ intrinsic_k, the");
    println!("transport is lossless on the personality subspace → cos ≈ sqrt(bf).");
    println!("For bf=0.85, sqrt(0.85) ≈ 0.92. The elbow at k=intrinsic_k is the", );
    println!("recommended minimum transport rank for this personality class.");
    println!("\nG3 PASS (characterization only — no hard gate). Update Research 291");
    println!("§5.3 'Recommended k per tier pair' with the elbow values from this table.");
    println!("\nNote: an unaligned random transport basis at k=intrinsic_k captures only");
    println!("~k/d_src of the personality subspace in expectation. The main loop above");
    println!("uses an ALIGNED basis (first min(k, INTRINSIC_K) columns span a random");
    println!("subspace of the personality subspace) — this is the realistic deployment");
    println!("scenario where bases are trained offline to align with personalities.");
}
