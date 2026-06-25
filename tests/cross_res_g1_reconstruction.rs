//! Cross-Resolution Spectral Transport — G1 reconstruction gate (Plan 310 T2.1).
//!
//! ## Hypothesis (Research 291 §5.1)
//!
//! If a personality shard is low-rank (Research 257 §5.5: personality vectors
//! concentrate 80%+ of their energy in a rank-k subspace), then
//! cross-resolution transport 64 → 16 → 64 should reconstruct the original
//! with cosine similarity ≥ 0.85 on average.
//!
//! ## Setup
//!
//! - 100 random 64-d reference vectors.
//! - Band-limitation model: 80% of L2 energy in the first k=8 spectral
//!   components of a random orthonormal Φ_src basis, 20% in the orthogonal
//!   complement. (Matches the "personality is low-rank" thesis — energy isn't
//!   *fully* band-limited, just *predominantly* so.)
//! - Random orthonormal Φ_src (64×8), Ψ_dst (16×8), k=8.
//! - Forward transport: 64 → 16. Reverse transport: 16 → 64 (swap bases).
//! - Measure cosine similarity of original vs reconstructed.
//!
//! ## Gate
//!
//! - **PASS:** mean cos ≥ 0.85, min cos ≥ 0.75.
//! - **DEMOTE-TO-GAIN:** mean cos ∈ [0.75, 0.85) — primitive works but loses
//!   too much to justify tier-transport deployment.
//! - **KILL:** mean cos < 0.75 — information loss destroys the personality.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features cross_resolution_transport --release \
//!   --test cross_res_g1_reconstruction -- --nocapture
//! ```

#![cfg(feature = "cross_resolution_transport")]

use katgpt_core::cross_resolution::{
    CrossResolutionBases, CrossResScratch, transport_cross_resolution_into,
};
use katgpt_core::simd;

const D_SRC: usize = 64;
const D_DST: usize = 16;
const K: usize = 8;
const N_SAMPLES: usize = 100;
/// Fraction of energy in the band-limited subspace (per Research 257 §5.5).
const BAND_ENERGY_FRAC: f32 = 0.80;

// ── Deterministic xorshift64* PRNG ─────────────────────────────────────────

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

/// Modified Gram-Schmidt over random columns → column-orthonormal `dim × k`.
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

/// Build a `dim`-d vector with `band_frac` of its L2 energy in the column
/// space of `basis` (`dim × k`) and the rest in the orthogonal complement.
///
/// Construction: `v = sqrt(band_frac) · basis · a + sqrt(1-band_frac) · noise_perp`,
/// where `a` is k-dim random, `noise_perp` is the orthogonal complement of a
/// full random vector against `basis`. By construction `||v||² = band_frac +
/// (1-band_frac) = 1` exactly (after normalization).
#[allow(clippy::needless_range_loop)] // orthonormalization math, explicit indexing clearer
fn bandlimited_sample(
    dim: usize,
    k: usize,
    basis: &[f32], // (dim, k) row-major
    band_frac: f32,
    rng: &mut Rng,
) -> Vec<f32> {
    // k-dim random spectral coefficients.
    let a: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
    // Band-limited part: basis · a  → dim-dim.
    let mut band_part = vec![0.0f32; dim];
    for r in 0..dim {
        let row = &basis[r * k..(r + 1) * k];
        band_part[r] = simd::simd_dot_f32(row, &a, k);
    }
    // Normalize band_part to unit norm.
    let band_norm = simd::simd_dot_f32(&band_part, &band_part, dim).sqrt();
    if band_norm > 1e-12 {
        for x in &mut band_part {
            *x /= band_norm;
        }
    }
    // Noise: full-dim random, then project out the basis subspace.
    let mut noise = vec![0.0f32; dim];
    for x in &mut noise {
        *x = rng.next_f32();
    }
    // Project noise onto orthogonal complement of basis's column space:
    // noise_perp = noise - basis · (basis^T · noise).
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
    // Normalize noise_perp to unit norm.
    let noise_norm = simd::simd_dot_f32(&noise, &noise, dim).sqrt();
    if noise_norm > 1e-12 {
        for x in &mut noise {
            *x /= noise_norm;
        }
    }
    // Combine: v = sqrt(band_frac) · band_part + sqrt(1-band_frac) · noise_perp.
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
fn g1_reconstruction_cosine() {
    let mut rng = Rng::new(0xC105_0001u64);

    // Forward bases: 64 → 16.
    let phi_src = random_orthonormal(D_SRC, K, &mut rng);
    let psi_dst = random_orthonormal(D_DST, K, &mut rng);
    let forward = CrossResolutionBases::new(phi_src, psi_dst, D_SRC, D_DST, K)
        .expect("forward bases should construct");
    assert!(forward.verify_orthonormal(1e-4), "Φ_src / Ψ_dst must be column-orthonormal");

    // Reverse bases: 16 → 64 (swap roles).
    let reverse = CrossResolutionBases::new(
        forward.psi_dst.clone(),
        forward.phi_src.clone(),
        D_DST,
        D_SRC,
        K,
    )
    .expect("reverse bases should construct");

    let mut scratch = CrossResScratch::new(K);

    let mut cosines: Vec<f32> = Vec::with_capacity(N_SAMPLES);
    for i in 0..N_SAMPLES {
        // Synthesize a band-limited sample.
        let src = bandlimited_sample(D_SRC, K, &forward.phi_src, BAND_ENERGY_FRAC, &mut rng);

        // Forward: 64 → 16.
        let mut dst = vec![0.0f32; D_DST];
        transport_cross_resolution_into(&src, &forward, &mut scratch, &mut dst);

        // Reverse: 16 → 64.
        let mut recon = vec![0.0f32; D_SRC];
        transport_cross_resolution_into(&dst, &reverse, &mut scratch, &mut recon);

        let cos = cosine(&src, &recon);
        cosines.push(cos);
        if i < 3 || i % 25 == 0 {
            println!("G1 sample {i}: cos = {cos:.4}");
        }
    }

    cosines.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f32 = cosines.iter().sum::<f32>() / cosines.len() as f32;
    let min = cosines.first().copied().unwrap_or(0.0);
    let p10 = cosines[(cosines.len() as f32 * 0.10) as usize];
    let median = cosines[cosines.len() / 2];

    println!("\nG1 RECONSTRUCTION RESULTS (n={}, d_src={}, d_dst={}, k={}, band_frac={}):",
        N_SAMPLES, D_SRC, D_DST, K, BAND_ENERGY_FRAC);
    println!("  min cos:    {min:.4}");
    println!("  p10 cos:    {p10:.4}");
    println!("  median cos: {median:.4}");
    println!("  mean cos:   {mean:.4}");
    println!("\nGate: mean ≥ 0.85 (PASS), min ≥ 0.75 (PASS hard floor), < 0.75 (KILL).");

    // Hard floor: no sample should be unrecoverable.
    assert!(
        min >= 0.75,
        "G1 KILL: min cos = {min:.4} < 0.75 — at least one sample lost too much \
         information to cross-resolution transport. The primitive cannot deploy."
    );
    // Mean gate.
    assert!(
        mean >= 0.85,
        "G1 FAIL: mean cos = {mean:.4} < 0.85 — cross-resolution transport loses \
         too much information on average. Demote to Gain-tier (research-only)."
    );
    println!("\nG1 PASS: mean cos = {mean:.4} (≥ 0.85), min cos = {min:.4} (≥ 0.75).");
}
