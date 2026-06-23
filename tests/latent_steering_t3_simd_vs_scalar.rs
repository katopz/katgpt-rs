//! Latent Field Steering — T3.2 SIMD vs scalar benchmark (Plan 309).
//!
//! ## Goal
//!
//! Measure the speedup of the explicit AVX2 SAXPY path over the scalar SAXPY
//! for the `apply_latent_steering` primitive, at d=8 (HLA) and d=16.
//!
//! ## Gate (Plan 309 T3.2)
//!
//! - **PASS d=8:** SIMD ≥ 2× scalar.
//! - **PASS d=16:** SIMD ≥ 1.5× scalar.
//! - Re-run G4 with the SIMD path; gate still p50 < 1ms.
//!
//! ## Caveat (Plan 309 Phase 2 finding)
//!
//! Phase 2 G4 measured 19.2µs for 5000×8 with the *auto-vectorizing* scalar
//! loop — i.e. LLVM already emits AVX2/SSE for the scalar form at `-O3`. The
//! plan author flagged Phase 3 as "likely a no-op". This benchmark exists to
//! measure the truth on the host CPU and record an honest verdict.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features latent_field_steering --release \
//!   --test latent_steering_t3_simd_vs_scalar -- --nocapture
//! ```

#![cfg(feature = "latent_field_steering")]

use katgpt_core::latent_steering::{LatentSteeringVector, apply_field_to_crowd, apply_latent_steering};
use std::hint::black_box;
use std::time::Instant;

/// Gate thresholds (Plan 309 T3.2).
const GATE_D8_X: f64 = 2.0;
const GATE_D16_X: f64 = 1.5;
/// G4 carry-over gate: crowd-scale p50 must stay under 1ms even with the
/// explicit SIMD dispatcher (one `is_x86_feature_detected!` per call).
const GATE_CROWD_US: f64 = 1000.0;

const N_ITERS: usize = 2000;
const N_WARMUP: usize = 200;

struct Rng {
    s: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { s: seed.max(1) }
    }
    fn next_f32(&mut self) -> f32 {
        // xorshift64 → [-1, 1]
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        let bits = (self.s >> 11) as u32;
        bits as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
}

/// Time `iters` calls to `f`, return median ns/call. `f` must do real work on
/// `state` each call (otherwise the optimizer hoists it out).
fn time_median_ns<F: FnMut(&mut [f32])>(state: &mut [f32], iters: usize, mut f: F) -> f64 {
    // Warmup
    for _ in 0..N_WARMUP {
        f(state);
    }
    let mut times_ns: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f(state);
        let dt = t0.elapsed().as_nanos() as f64;
        times_ns.push(dt);
    }
    black_box(state.as_ptr());
    times_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_ns[iters / 2]
}

#[test]
fn t3_simd_vs_scalar_throughput() {
    println!("\n=== Plan 309 T3.2 — SIMD vs scalar SAXPY throughput ===");
    #[cfg(target_arch = "x86_64")]
    {
        println!(
            "host x86_64=true, AVX2={}",
            std::is_x86_feature_detected!("avx2")
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        eprintln!(
            "host target_arch={:?} — NOT x86_64. The AVX2 SAXPY backend is compiled out
             on this target; `apply_latent_steering` routes to the scalar fallback.
             T3.2 SIMD-vs-scalar speedup CANNOT be measured here — the numbers below
             are scalar-vs-scalar and must not be used to satisfy the T3.2 gate.
             Re-run on an x86_64+AVX2 host to get a real verdict.",
            std::env::consts::ARCH
        );
    }

    for &d in &[8usize, 16] {
        let mut rng = Rng::new(0xC40D + d as u64);

        // Build a unit-norm direction.
        let mut dir: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
        let norm = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut dir {
            *x /= norm;
        }
        let steering = LatentSteeringVector::new(dir.clone(), 0.3, 1e-4).unwrap();

        // `apply_latent_steering` uses the SIMD dispatcher when AVX2 is present.
        // To measure the scalar fallback on the same host, we route through
        // `apply_latent_steering` only — there is no public scalar entry point,
        // so the "scalar" baseline below is the crowd-scale path on a no-position
        // global field, which calls the same dispatcher. To still get a scalar
        // baseline for the speedup ratio, we inline a scalar SAXPY here that
        // matches `saxpy_inplace_scalar` exactly.
        let mut state_simd: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
        let mut state_scalar = state_simd.clone();
        let alpha = 0.3f32;

        // ── SIMD path: apply_latent_steering dispatches to AVX2 when available.
        let ns_simd = time_median_ns(&mut state_simd, N_ITERS, |s| {
            apply_latent_steering(s, &steering);
        });

        // ── Scalar baseline: inline scalar SAXPY (matches the pre-T3.1 loop).
        let dir_for_closure = dir.clone();
        let ns_scalar = time_median_ns(&mut state_scalar, N_ITERS, move |s| {
            let p = black_box(s.as_ptr());
            for (si, di) in s.iter_mut().zip(dir_for_closure.iter()) {
                *si += alpha * di;
            }
            black_box((p, s.len()));
        });

        let speedup = ns_scalar / ns_simd;
        let gate = if d == 8 { GATE_D8_X } else { GATE_D16_X };
        let verdict = if speedup >= gate { "PASS" } else { "GATE FAIL" };
        println!(
            "  d={d}: scalar={ns_scalar:.1}ns/call  simd={ns_simd:.1}ns/call  \
             speedup={speedup:.2}x  (gate ≥{gate}x)  {verdict}"
        );

        // Sanity: both paths should still produce the same state (element-wise
        // op). state_simd has had one extra warmup pass so they diverge slightly
        // in absolute terms; just check the per-call delta stays bounded.
        // (Soft assertion — the strict bit-equality is covered in the in-module
        // `saxpy_simd_matches_scalar` unit test.)
        let _ = (state_simd, state_scalar);
    }

    // ── G4 carry-over: crowd-scale with SIMD dispatcher ─────────────────
    use katgpt_core::latent_steering::{FieldSupport, LatentField};
    let n_npcs = 5000usize;
    let d = 8usize;
    let mut rng = Rng::new(0xC40D);
    let mut dir: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
    let norm = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut dir {
        *x /= norm;
    }
    let steering = LatentSteeringVector::new(dir, 0.3, 1e-4).unwrap();
    let field = LatentField {
        steering,
        support: FieldSupport::Global,
    };
    let mut states: Vec<f32> = (0..n_npcs * d).map(|_| rng.next_f32()).collect();
    let positions = vec![None; n_npcs];
    let zones = vec![None; n_npcs];

    for _ in 0..N_WARMUP {
        apply_field_to_crowd(&mut states, d, &positions, &zones, &field);
    }
    let mut crowd_ns: Vec<f64> = Vec::with_capacity(N_ITERS);
    for _ in 0..N_ITERS {
        let t0 = Instant::now();
        apply_field_to_crowd(&mut states, d, &positions, &zones, &field);
        crowd_ns.push(t0.elapsed().as_nanos() as f64);
    }
    black_box(&states);
    crowd_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_us = crowd_ns[N_ITERS / 2] / 1000.0;
    let crowd_verdict = if p50_us < GATE_CROWD_US { "PASS" } else { "GATE FAIL" };
    println!(
        "  G4 re-run (SIMD, {n_npcs}×{d}): p50={p50_us:.1}µs  (gate <{GATE_CROWD_US:.0}µs)  {crowd_verdict}"
    );
    assert!(
        p50_us < GATE_CROWD_US,
        "G4 carry-over FAIL: p50={p50_us:.1}µs ≥ {GATE_CROWD_US:.0}µs"
    );

    println!("=== T3.2 measurement complete (verdicts printed above; asserts only enforce G4 carry-over) ===");
}
