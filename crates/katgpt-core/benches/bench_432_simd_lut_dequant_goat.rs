//! SIMD LUT DeQuant GOAT bench (Plan 431 Phase 4).
//!
//! Exercises G1–G6 for the `simd_lut_dequant` primitive — the software analog
//! of StreamDQ near-memory weight dequantization (arXiv:2607.08993).
//!
//! # Gates
//!
//! - **G1 (bit-exact correctness)**: The LUT path must produce **bit-identical**
//!   outputs to the arithmetic cast path `(code - zero) * scale` across all 256
//!   possible code bytes × 3 formats (UInt4, Int4, Int8). Max abs diff must be
//!   exactly `0.0`. This is the load-bearing correctness contract.
//!
//! - **G2 (latency — the make-or-break gate)**: Median timing on three workloads:
//!     - Single-block dequant (256 elements): LUT vs arithmetic. Target: ≥ 1.0×
//!       (no regression — LUT build cost may dominate at small sizes).
//!     - Full-row dequant (4096 elements): LUT vs arithmetic. Target: ≥ 1.2×.
//!     - Fused dequant+dot (4096 elements): fused vs two-step. Target: ≥ 1.3×.
//!
//!   If G2 FAILS on all three: document negative result, keep opt-in, do NOT
//!   promote to default. The plan's realistic target is 1.0–1.5× (the paper's
//!   7× is hardware-only).
//!
//! - **G3 (feature isolation)**: The feature is purely additive — default
//!   features build clean, and `--features simd_lut_dequant` builds clean.
//!
//! - **G4 (alloc-free hot path)**: After warmup, 100 steady-state calls through
//!   `dequant_via_lut` and `dequant_dot_via_lut` allocate 0 times (counted via
//!   a global `CountingAllocator`). LUT is stack `[f32; N]`, output is
//!   caller-owned `&mut [f32]`.
//!
//! - **G5 (SIMD-level)**: Report which SIMD path is active on the current arch.
//!
//! - **G6 (determinism)**: Same inputs produce same outputs across 100 calls.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --bench bench_432_simd_lut_dequant_goat -- --nocapture
//! ```
//!
//! Or (working around the dyld/trustD stall on macOS):
//!
//! ```bash
//! cargo bench -p katgpt-core --bench bench_432_simd_lut_dequant_goat --no-run
//! target/release/deps/bench_432_simd_lut_dequant_goat-<hash>
//! ```

use katgpt_core::simd_lut_dequant::{
    Int4Lut, Int8Lut, QuantLut, UInt4Lut, dequant_arithmetic_ref, dequant_dot_via_lut,
    dequant_via_lut, dequant_via_lut_scalar,
};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Deterministic xorshift PRNG (no `rand` dep).
struct Rng {
    state: u32,
}
impl Rng {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF } else { seed },
        }
    }
    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }
}

fn timed_median_ns(iters: usize, batches: usize, mut body: impl FnMut()) -> f64 {
    for _ in 0..(batches.min(20)) {
        body();
    }
    let mut batch_times_ns: Vec<u64> = Vec::with_capacity(batches);
    for _ in 0..batches {
        let start = Instant::now();
        for _ in 0..iters {
            body();
        }
        batch_times_ns.push(start.elapsed().as_nanos() as u64);
    }
    batch_times_ns.sort_unstable();
    let mid = batch_times_ns.len() / 2;
    let median_batch_ns = batch_times_ns[mid] as f64;
    median_batch_ns / iters as f64
}

#[inline(never)]
fn bb<T>(x: T) -> T {
    black_box(x)
}

// ─── Gate runner ────────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

impl GateResult {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: true,
            detail: detail.into(),
        }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            detail: detail.into(),
        }
    }
}

// ─── G1: Bit-exact correctness ──────────────────────────────────────────────

fn gate_g1_bit_exact() -> GateResult {
    let mut max_diff_overall: f32 = 0.0;

    // UInt4: all 256 code bytes, low nibble.
    {
        let scale = 0.3_f32;
        let zero = 5.5_f32;
        let lut = UInt4Lut::build(scale, zero);
        let codes: Vec<u8> = (0..=255u32).map(|i| i as u8).collect();
        let mut out_lut = vec![0.0_f32; 256];
        let mut out_ref = vec![0.0_f32; 256];
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out_lut);
        dequant_arithmetic_ref(&codes, scale, zero, 0, 0x0F, false, &mut out_ref);
        for i in 0..256 {
            max_diff_overall = max_diff_overall.max((out_lut[i] - out_ref[i]).abs());
        }
    }

    // Int4: signed nibble.
    {
        let scale = 0.7_f32;
        let zero = -2.0_f32;
        let lut = Int4Lut::build(scale, zero);
        let codes: Vec<u8> = (0..=255u32).map(|i| i as u8).collect();
        let mut out_lut = vec![0.0_f32; 256];
        let mut out_ref = vec![0.0_f32; 256];
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out_lut);
        dequant_arithmetic_ref(&codes, scale, zero, 0, 0x0F, true, &mut out_ref);
        for i in 0..256 {
            max_diff_overall = max_diff_overall.max((out_lut[i] - out_ref[i]).abs());
        }
    }

    // Int8: byte-aligned.
    {
        let scale = 0.1_f32;
        let zero = 50.0_f32;
        let lut = Int8Lut::build(scale, zero);
        let codes: Vec<u8> = (0..=255u32).map(|i| i as u8).collect();
        let mut out_lut = vec![0.0_f32; 256];
        let mut out_ref = vec![0.0_f32; 256];
        dequant_via_lut(&codes, &lut, 0, 0xFF, &mut out_lut);
        dequant_arithmetic_ref(&codes, scale, zero, 0, 0xFF, false, &mut out_ref);
        for i in 0..256 {
            max_diff_overall = max_diff_overall.max((out_lut[i] - out_ref[i]).abs());
        }
    }

    let detail = format!(
        "max abs diff across UInt4+Int4+Int8 × 256 codes: {max_diff_overall:.1e} (must be exactly 0.0)"
    );

    if max_diff_overall == 0.0 {
        GateResult::pass("G1", detail)
    } else {
        GateResult::fail("G1", detail)
    }
}

// ─── G2: Latency ────────────────────────────────────────────────────────────

fn gate_g2_latency() -> GateResult {
    let mut rng = Rng::new(0x431_431);
    let scale = 0.15_f32;
    let zero = 4.0_f32;

    // ── Workload 1: single-block dequant (256 elements) ─────────────────
    let n_block = 256;
    let codes_block: Vec<u8> = (0..n_block).map(|_| rng.next_u32() as u8).collect();
    let lut = UInt4Lut::build(scale, zero);
    let mut out = vec![0.0_f32; n_block];

    // Arithmetic path baseline.
    let ns_arith_block = timed_median_ns(10_000, 100, || {
        dequant_arithmetic_ref(&codes_block, scale, zero, 0, 0x0F, false, &mut out);
        bb(&out);
    });
    // LUT path.
    let ns_lut_block = timed_median_ns(10_000, 100, || {
        dequant_via_lut(&codes_block, &lut, 0, 0x0F, &mut out);
        bb(&out);
    });
    let speedup_block = ns_arith_block / ns_lut_block;

    // ── Workload 2: full-row dequant (4096 elements) ────────────────────
    let n_row = 4096;
    let codes_row: Vec<u8> = (0..n_row).map(|_| rng.next_u32() as u8).collect();
    let mut out_row = vec![0.0_f32; n_row];

    let ns_arith_row = timed_median_ns(2_000, 100, || {
        dequant_arithmetic_ref(&codes_row, scale, zero, 0, 0x0F, false, &mut out_row);
        bb(&out_row);
    });
    let ns_lut_row = timed_median_ns(2_000, 100, || {
        dequant_via_lut(&codes_row, &lut, 0, 0x0F, &mut out_row);
        bb(&out_row);
    });
    let speedup_row = ns_arith_row / ns_lut_row;

    // ── Workload 3: fused dequant+dot (4096 elements) ───────────────────
    let x_row: Vec<f32> = (0..n_row).map(|_| rng.next_f32()).collect();

    // Two-step: dequant to buffer, then scalar dot (the "current path").
    let ns_twostep = timed_median_ns(2_000, 100, || {
        dequant_via_lut_scalar(&codes_row, lut.as_f32_slice(), 0, 0x0F, &mut out_row);
        let dot: f32 = out_row.iter().zip(&x_row).map(|(a, b)| a * b).sum();
        bb(dot);
    });
    // Fused: no intermediate buffer.
    let ns_fused = timed_median_ns(2_000, 100, || {
        let dot = dequant_dot_via_lut(&codes_row, &lut, &x_row, 0, 0x0F);
        bb(dot);
    });
    let speedup_fused = ns_twostep / ns_fused;

    let detail = format!(
        "single-block(256): LUT {ns_lut_block:.1}ns vs arith {ns_arith_block:.1}ns = {speedup_block:.3}x (target >=1.0x) | \
         full-row(4096): LUT {ns_lut_row:.1}ns vs arith {ns_arith_row:.1}ns = {speedup_row:.3}x (target >=1.2x) | \
         fused-dot(4096): {ns_fused:.1}ns vs two-step {ns_twostep:.1}ns = {speedup_fused:.3}x (target >=1.3x)"
    );

    // G2 PASS if EITHER full-row >=1.2x OR fused >=1.3x. Single-block is
    // informational (LUT overhead may dominate at small sizes).
    let pass_row = speedup_row >= 1.2;
    let pass_fused = speedup_fused >= 1.3;
    if pass_row || pass_fused {
        GateResult::pass("G2", detail)
    } else {
        GateResult::fail("G2", detail)
    }
}

// ─── G3: Feature isolation ──────────────────────────────────────────────────

fn gate_g3_feature_isolation() -> GateResult {
    // G3 is verified at the clippy level (default features + --features both
    // build clean). Here we just verify the feature compiles and runs.
    let lut = UInt4Lut::build(1.0, 0.0);
    let codes = [0x01_u8, 0x02];
    let mut out = [0.0_f32; 2];
    dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out);
    let detail = format!(
        "simd_lut_dequant feature compiles and runs: dequant([0x01,0x02], UInt4, shift=0) = {out:?}"
    );
    GateResult::pass("G3", detail)
}

// ─── G4: Alloc-free hot path ────────────────────────────────────────────────

fn gate_g4_alloc_free() -> GateResult {
    use std::sync::atomic::Ordering;

    let lut = UInt4Lut::build(0.3, 5.0);
    let codes: Vec<u8> = (0..4096).map(|i| (i * 7) as u8).collect();
    let mut out = vec![0.0_f32; 4096];
    let x: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();

    // Warmup (Vec allocations, etc.)
    for _ in 0..10 {
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out);
        let _dot = dequant_dot_via_lut(&codes, &lut, &x, 0, 0x0F);
    }

    let before = ALLOC_COUNT.load(Ordering::SeqCst);
    for _ in 0..100 {
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out);
    }
    let after_dequant = ALLOC_COUNT.load(Ordering::SeqCst);
    for _ in 0..100 {
        let _dot = dequant_dot_via_lut(&codes, &lut, &x, 0, 0x0F);
    }
    let after_dot = ALLOC_COUNT.load(Ordering::SeqCst);

    let dequant_allocs = after_dequant - before;
    let dot_allocs = after_dot - after_dequant;

    let detail = format!(
        "dequant_via_lut: {dequant_allocs} allocs / 100 calls | \
         dequant_dot_via_lut: {dot_allocs} allocs / 100 calls (both must be 0)"
    );

    if dequant_allocs == 0 && dot_allocs == 0 {
        GateResult::pass("G4", detail)
    } else {
        GateResult::fail("G4", detail)
    }
}

// ─── G5: SIMD-level report ──────────────────────────────────────────────────

fn gate_g5_simd_report() -> GateResult {
    let arch = if cfg!(target_arch = "aarch64") {
        "aarch64 NEON (scalar gather — NEON has no native gather instruction)"
    } else if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        "x86_64 AVX2 (hardware gather via _mm256_i32gather_ps)"
    } else if cfg!(target_arch = "wasm32") {
        "wasm32 (scalar fallback — WASM SIMD128 has no gather)"
    } else {
        "scalar fallback (no SIMD path for this arch)"
    };

    let detail = format!("Active backend: {arch}");
    GateResult::pass("G5", detail)
}

// ─── G6: Determinism ────────────────────────────────────────────────────────

fn gate_g6_determinism() -> GateResult {
    let lut = UInt4Lut::build(0.5, 3.0);
    let codes: Vec<u8> = (0..256).map(|i| (i * 17) as u8).collect();
    let mut out_first = vec![0.0_f32; 256];
    dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out_first);

    let mut mismatches = 0;
    for _ in 0..99 {
        let mut out = vec![0.0_f32; 256];
        dequant_via_lut(&codes, &lut, 0, 0x0F, &mut out);
        for i in 0..256 {
            if out[i].to_bits() != out_first[i].to_bits() {
                mismatches += 1;
            }
        }
    }

    let detail = format!("{mismatches} mismatches across 100 identical calls (must be 0)");
    if mismatches == 0 {
        GateResult::pass("G6", detail)
    } else {
        GateResult::fail("G6", detail)
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  Plan 431 — SIMD LUT DeQuant GOAT Gate");
    println!("  Research 418 — StreamDQ software analog (arXiv:2607.08993)");
    println!("══════════════════════════════════════════════════════════════════════");
    println!();

    let gates = vec![
        gate_g1_bit_exact(),
        gate_g2_latency(),
        gate_g3_feature_isolation(),
        gate_g4_alloc_free(),
        gate_g5_simd_report(),
        gate_g6_determinism(),
    ];

    let mut all_pass = true;
    for g in &gates {
        let status = if g.passed { "✅ PASS" } else { "❌ FAIL" };
        println!("{status}  {}: {}", g.name, g.detail);
        if !g.passed {
            all_pass = false;
        }
        println!();
    }

    println!("══════════════════════════════════════════════════════════════════════");
    let g2 = &gates[1];
    let g2_detail = &g2.detail;
    if all_pass {
        println!("  VERDICT: ALL GATES PASS — promote `simd_lut_dequant` to default-on");
    } else if g2.passed {
        println!("  VERDICT: G2 PASS (latency win confirmed) — promote to default-on");
        println!("  (non-G2 failures are informational; review before promoting)");
    } else {
        println!("  VERDICT: G2 FAIL — keep `simd_lut_dequant` opt-in");
        println!("  The LUT path is slower than the arithmetic path on this platform.");
        println!("  G2 detail: {g2_detail}");
        println!();
        println!("  This is an honest negative result. The infrastructure stays for");
        println!("  future FP8/INT8 consumers where the LUT might win bigger.");
    }
    println!("══════════════════════════════════════════════════════════════════════");

    if !all_pass {
        std::process::exit(1);
    }
}
