//! Off-Principal Retrieval Demo — Plan 264 Phase 2 (Research 231).
//!
//! Demonstrates the retrieval-accuracy gain from projecting task vectors off
//! the principal subspace of the source weight matrix before scoring. Matches
//! the GOAT G4 gate: off-principal top-1 accuracy must beat raw cosine top-1
//! accuracy by ≥5 percentage points on synthetic OPD-shaped adapters.
//!
//! # Before / After
//!
//! - **Before (raw cosine):** query · adapter — dominated by the shared
//!   principal component, ties near-randomly across adapters with similar
//!   principal magnitudes.
//! - **After (off-principal):** project both query and adapter off the
//!   principal subspace, then dot. Isolates the per-adapter task signal.
//!
//! Run: `cargo run --features off_principal_retrieval --example off_principal_retrieval`

#![cfg(feature = "off_principal_retrieval")]

use katgpt_rs::off_principal::OffPrincipalIndex;
use katgpt_rs::simd::simd_dot_f32;

const D: usize = 64;
const N_ADAPTERS: usize = 8;

fn main() {
    println!("=== Plan 264 Phase 2 — Off-Principal Retrieval Demo ===\n");
    println!("Synthetic setup: d={}, {} adapters", D, N_ADAPTERS);
    println!("Each adapter: principal on axis 0 (varying magnitude) +");
    println!("             unique off-principal signal on axis 1+i\n");

    // Source weight: dominant principal direction along axis 0.
    let mut w_src = vec![0.0_f32; D];
    w_src[0] = 10.0;
    for (i, w) in w_src.iter_mut().enumerate().take(D).skip(1) {
        *w = 0.01 * (i as f32);
    }
    let idx = OffPrincipalIndex::new(&w_src, (D, 1), 0.10);
    println!(
        "OffPrincipalIndex: d={}, k={}, k_frac={:.2}",
        idx.d, idx.k, idx.k_frac
    );
    println!(
        "BLAKE3(src) = {}",
        hex_hash(&idx.src_hash)
    );
    println!();

    // Build adapters: principal varies (breaks cosine), off-principal is unique.
    let mut adapters: Vec<Vec<f32>> = Vec::with_capacity(N_ADAPTERS);
    for i in 0..N_ADAPTERS {
        let mut a = vec![0.0_f32; D];
        a[0] = 8.0 + 0.5 * (i as f32);
        a[1 + i] = 1.0;
        adapters.push(a);
    }

    let mut rng = make_rng(0x5eed_5eed);
    let n_trials = 200;
    let mut cosine_correct = 0usize;
    let mut off_principal_correct = 0usize;

    let mut q_scratch = vec![0.0_f32; idx.k + D];
    let mut a_scratch = vec![0.0_f32; idx.k + D];

    for trial in 0..n_trials {
        let gt = trial % N_ADAPTERS;
        let mut query = vec![0.0_f32; D];
        query[0] = 10.0;
        query[1 + gt] = 0.5;
        for v in query.iter_mut().skip(2) {
            *v += 0.02 * rng();
        }

        // Raw cosine.
        let mut cosine_best = f32::NEG_INFINITY;
        let mut cosine_argmax = 0;
        for (j, a) in adapters.iter().enumerate() {
            let dot = simd_dot_f32(&query, a, D);
            if dot > cosine_best {
                cosine_best = dot;
                cosine_argmax = j;
            }
        }
        if cosine_argmax == gt {
            cosine_correct += 1;
        }

        // Off-principal.
        let q_off = idx.project(&query, &mut q_scratch);
        let mut op_best = f32::NEG_INFINITY;
        let mut op_argmax = 0;
        for (j, a) in adapters.iter().enumerate() {
            let a_off = idx.project(a, &mut a_scratch);
            let dot = simd_dot_f32(q_off, a_off, D);
            if dot > op_best {
                op_best = dot;
                op_argmax = j;
            }
        }
        if op_argmax == gt {
            off_principal_correct += 1;
        }
    }

    let cosine_acc = cosine_correct as f32 / n_trials as f32;
    let op_acc = off_principal_correct as f32 / n_trials as f32;
    let gain = (op_acc - cosine_acc) * 100.0;

    println!("Results over {} trials:", n_trials);
    println!("  Raw cosine top-1 accuracy:     {:.1}%", cosine_acc * 100.0);
    println!("  Off-principal top-1 accuracy: {:.1}%", op_acc * 100.0);
    println!("  Gain:                         {:+.1} pp", gain);
    println!();

    if gain >= 5.0 {
        println!("✅ GOAT G4 PASS: off-principal beats cosine by ≥5pp");
    } else {
        println!("❌ GOAT G4 FAIL: gain {:.1}pp < 5pp", gain);
        std::process::exit(1);
    }
}

/// Simple deterministic PRNG (xorshift64) for reproducible results.
fn make_rng(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f32 / (1u64 << 52) as f32) * 2.0 - 1.0
    }
}

/// Format a BLAKE3 hash as a truncated hex string for display.
fn hex_hash(hash: &[u8; 32]) -> String {
    let mut s = String::with_capacity(16);
    for &b in &hash[..8] {
        s.push_str(&format!("{:02x}", b));
    }
    s.push('…');
    s
}
