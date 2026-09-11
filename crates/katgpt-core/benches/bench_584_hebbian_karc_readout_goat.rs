//! Plan 584 T3 — Hebbian KARC Readout GOAT (G2 perf + G4 alloc).
//!
//! The `w_out` fit, Hebbian-flavored (riir-ai Plan 584; ndb Plan 322 T3.2).
//! Measures the readout seam over the already-validated primitive (Plan 559
//! G2/G4; ndb Bench 463 bridge seam):
//!
//! - **G2a** — `fit` µs/fact at the runtime KARC config (32-dim delay states,
//!   8-dim targets, F=128, m=128), gated as a same-process comparison against
//!   the arm it stands beside (`KarcForecaster::fit_ridge` at D=8/M=8/K=4 —
//!   same pairs). REPORTED (not hard-gated): both arms are load-dependent on
//!   this shared box; the gate is the RATIO being sane (Hebbian fit within an
//!   order of magnitude of fit_ridge — it carries the margin audit + fact
//!   semantics fit_ridge does not).
//! - **G2b** — `forecast_into` µs/query + a direct `forward_into` baseline
//!   (the pad64 + head-slice wrapper Δ — machine-independent like Bench 908's
//!   slot-Δ gate).
//! - **G4** — `forecast_into` 0 allocs / 100 calls (CountingAllocator).
//!
//! # Run (direct-binary workaround)
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/plan584_t3 cargo bench -p katgpt-core \
//!     --features karc_hebbian_readout --bench bench_584_hebbian_karc_readout_goat --no-run
//! /tmp/plan584_t3/release/deps/bench_584_hebbian_karc_readout_goat-<hash> --nocapture
//! ```

#![cfg(feature = "karc_hebbian_readout")]

use katgpt_core::hebbian_kernel_memory::{HebbianMlpConfig, HebbianVariant};
use katgpt_core::karc::hebbian_readout::{HebbianKarcReadout, HEBBIAN_KARC_DIM};
use katgpt_core::karc::{ChebyshevBasis, KarcForecaster};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── Gate results ──────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

// ─── Fixtures ──────────────────────────────────────────────────────────────

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let x = (*state >> 33) as f32 / (1u64 << 31) as f32;
    x - 0.5
}

/// Runtime KARC config: delay 32 (K=4 × D=8), target 8.
const STATE_DIM: usize = 32;
const TARGET_DIM: usize = 8;
const F: usize = 128;

fn pairs(seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut s = seed;
    let delays: Vec<Vec<f32>> = (0..F)
        .map(|i| {
            let mut d = vec![0.0f32; STATE_DIM];
            for x in &mut d {
                *x = lcg(&mut s) * 2.0;
            }
            d[i % STATE_DIM] += 3.0;
            d
        })
        .collect();
    let targets: Vec<Vec<f32>> = (0..F)
        .map(|i| {
            let mut t = vec![0.0f32; TARGET_DIM];
            for x in &mut t {
                *x = lcg(&mut s) * 2.0;
            }
            t[i % TARGET_DIM] += 3.0;
            t
        })
        .collect();
    (delays, targets)
}

fn seed_of(delays: &[&[f32]], targets: &[&[f32]]) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(&(delays.len() as u64).to_le_bytes());
    for (k, v) in delays.iter().zip(targets) {
        h.update(bytemuck::cast_slice::<f32, u8>(k));
        h.update(bytemuck::cast_slice::<f32, u8>(v));
    }
    let hash = *h.finalize().as_bytes();
    u64::from_le_bytes(hash[..8].try_into().unwrap())
}

fn config() -> HebbianMlpConfig {
    HebbianMlpConfig { d: HEBBIAN_KARC_DIM, m: 128, ridge: 1e-6, variant: HebbianVariant::Whitened }
}

// ─── Gates ─────────────────────────────────────────────────────────────────

/// G2a — fit µs/fact, both arms, same process, each in its natural regime:
/// the ridge arm needs `n_samples > d_h` (256) to be PD, so it accumulates
/// 512 pairs via the real observe path; the Hebbian arm fits F=128 facts
/// (its capacity regime). The gate is the PER-FACT ratio (the Hebbian arm
/// carries the margin audit + fact semantics — an order of magnitude of
/// fit_ridge is the sane band, hard-gated; total-µs comparison across
/// different F would be dishonest).
fn g2a_fit_vs_fit_ridge() -> GateResult {
    const ITERS: usize = 10;
    const F_RIDGE: usize = 512;
    let (delays, targets) = pairs(42);
    let d_refs: Vec<&[f32]> = delays.iter().map(|d| &d[..]).collect();
    let t_refs: Vec<&[f32]> = targets.iter().map(|t| &t[..]).collect();
    let seed = seed_of(&d_refs, &t_refs);

    // fit_ridge arm at the runtime config (D=8, M=8, K=4 → d_h=256). Fed via
    // the REAL observe path (observe_and_maybe_pair until F_RIDGE pairs —
    // the runtime's natural accumulation) so the Gram is PD. The pairing is
    // consecutive (delay-window → next-belief), so the DATA differs from the
    // Hebbian arm's synthetic pairs; timing is shape-driven, which is what a
    // ratio gate needs. Noted in the record.
    type Rtc = KarcForecaster<ChebyshevBasis<8>, 8, 8, 4>;
    let mut ridge: Rtc = KarcForecaster::with_capacity(ChebyshevBasis::new(), F_RIDGE);
    let mut s = 4242u64;
    let mut tick = 0u64;
    while ridge.n_samples() < F_RIDGE {
        let mut belief = [0.0f32; 8];
        for b in &mut belief {
            *b = lcg(&mut s) * 2.0;
        }
        belief[(tick % 8) as usize] += 3.0;
        let _ = ridge.observe_and_maybe_pair(&belief);
        tick += 1;
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(ridge.fit_ridge(1e-6));
    }
    let ridge_per_fact = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64 / F_RIDGE as f64;

    // Hebbian arm (F=128 — its capacity regime).
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(
            HebbianKarcReadout::fit(&d_refs, &t_refs, config(), seed).expect("fit"),
        );
    }
    let hebbian_per_fact = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64 / F as f64;

    let ratio = hebbian_per_fact / ridge_per_fact.max(1e-9);
    let detail = format!(
        "hebbian fit {hebbian_per_fact:.2} µs/fact (F={F}) vs fit_ridge {ridge_per_fact:.2} µs/fact (F={F_RIDGE} > d_h=256) — ratio {ratio:.1}× (m=128; absolutes box-load-dependent)"
    );
    // Sane band: the Hebbian arm must be within 10× of fit_ridge per fact (it
    // does strictly more: construction + margin audit over F forwards).
    if ratio <= 10.0 && hebbian_per_fact.is_finite() && ridge_per_fact > 0.0 {
        GateResult { name: "G2a fit vs fit_ridge (per-fact ratio)", passed: true, detail }
    } else {
        GateResult { name: "G2a fit vs fit_ridge (per-fact ratio)", passed: false, detail }
    }
}

/// G2b — forecast µs/query + wrapper Δ over direct `forward_into`.
fn g2b_forecast_latency() -> GateResult {
    const ITERS: usize = 10_000;
    let (delays, targets) = pairs(43);
    let d_refs: Vec<&[f32]> = delays.iter().map(|d| &d[..]).collect();
    let t_refs: Vec<&[f32]> = targets.iter().map(|t| &t[..]).collect();
    let (readout, _) =
        HebbianKarcReadout::fit(&d_refs, &t_refs, config(), seed_of(&d_refs, &t_refs)).expect("fit");

    let mut phi = vec![0.0f32; 128];
    let mut out = [0.0f32; TARGET_DIM];

    // Warmup.
    for d in &delays[..10] {
        let _ = readout.forecast_into(d, &mut phi, &mut out);
    }

    let t0 = Instant::now();
    for i in 0..ITERS {
        let d = &delays[i % delays.len()];
        let _ = black_box(readout.forecast_into(black_box(d), &mut phi, &mut out));
    }
    let per_query_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    // Direct baseline: forward_into without the pad64/head wrapper.
    let mem = readout.memory();
    let mut padded = [0.0f32; HEBBIAN_KARC_DIM];
    let mut head = [0.0f32; HEBBIAN_KARC_DIM];
    let t0 = Instant::now();
    for i in 0..ITERS {
        let d = &delays[i % delays.len()];
        padded[..d.len()].copy_from_slice(d);
        mem.forward_into(&padded[..], &mut phi, &mut head);
    }
    let direct_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let delta = per_query_us - direct_us;
    let detail = format!(
        "forecast {per_query_us:.3} µs · direct forward {direct_us:.3} · wrapper Δ {delta:+.3} µs (target Δ ≤ 1; absolutes box-load-dependent)"
    );
    if delta <= 1.0 && per_query_us.is_finite() {
        GateResult { name: "G2b forecast µs (wrapper Δ)", passed: true, detail }
    } else {
        GateResult { name: "G2b forecast µs (wrapper Δ)", passed: false, detail }
    }
}

/// G4 — forecast_into 0 allocs / 100 calls.
fn g4_forecast_allocs() -> GateResult {
    let (delays, targets) = pairs(44);
    let d_refs: Vec<&[f32]> = delays.iter().map(|d| &d[..]).collect();
    let t_refs: Vec<&[f32]> = targets.iter().map(|t| &t[..]).collect();
    let (readout, _) =
        HebbianKarcReadout::fit(&d_refs, &t_refs, config(), seed_of(&d_refs, &t_refs)).expect("fit");
    let mut phi = vec![0.0f32; 128];
    let mut out = [0.0f32; TARGET_DIM];

    for d in &delays[..10] {
        let _ = readout.forecast_into(d, &mut phi, &mut out);
    }

    let ((), allocs) = alloc_delta(|| {
        for i in 0..100 {
            let d = &delays[i % delays.len()];
            let _ = black_box(readout.forecast_into(black_box(d), &mut phi, &mut out));
        }
    });
    let detail = format!("{allocs} allocs / 100 calls");
    if allocs == 0 {
        GateResult { name: "G4 forecast_into allocs", passed: true, detail }
    } else {
        GateResult { name: "G4 forecast_into allocs", passed: false, detail }
    }
}

fn main() {
    println!("═════════════════════════════════════════════════════════════════");
    println!("  Plan 584 T3 — Hebbian KARC Readout GOAT (G2 perf + G4 alloc)");
    println!("═════════════════════════════════════════════════════════════════");
    println!();

    let gates = vec![g2a_fit_vs_fit_ridge(), g2b_forecast_latency(), g4_forecast_allocs()];
    for g in &gates {
        println!("  {} — {} ({})", if g.passed { "✅ PASS" } else { "❌ FAIL" }, g.name, g.detail);
    }
    println!();
    let all_pass = gates.iter().all(|g| g.passed);
    println!("  ─── Plan 584 T3 GOAT verdict: {} ───", if all_pass { "ALL PASS ✅" } else { "FAIL ❌" });
    if !all_pass {
        std::process::exit(1);
    }
}
