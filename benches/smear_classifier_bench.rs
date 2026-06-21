// @katgpt-rs/benches/smear_classifier_bench.rs
//
//! Plan 298 Phase 3 — GOAT Gate G3: latency.
//!
//! Measures `CosineSmearClassifier::classify` at the k×d sweep called out in
//! Plan 298 §G3: `k ∈ {2, 4, 8}`, `d ∈ {8, 16, 32}`. Target: **≤ 200 ns** for
//! `k=8, d=32` on Apple Silicon arm64 (the plasma-tier budget — what makes
//! the classifier viable at audit cadence on 20 Hz × thousands-of-NPCs
//! crowd-scale cognitive allocation).
//!
//! ## Why not criterion
//!
//! `criterion` is a `katgpt-core` dev-dep but NOT a `katgpt-rs` dev-dep. This
//! file follows the repo bench convention set by
//! `benches/faithfulness_probe_bench.rs` and `benches/bench_294_ict_perf.rs`:
//! `std::time::Instant` + `std::hint::black_box` + `harness = false` +
//! `fn main()`.
//!
//! ## Run
//!
//! ```text
//! cargo bench --features smear_classifier --bench smear_classifier_bench
//! # or:
//! cargo run --release --features smear_classifier --bench smear_classifier_bench
//! ```
//!
//! Release build is required — debug builds do not engage SIMD autovectorization
//! and the G3 target is unreachable without it.

#![cfg(feature = "smear_classifier")]

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::faithfulness::smear::{CosineSmearClassifier, SmearClassifier};

const WARMUP_ITERS: usize = 1_000;
const BENCH_ITERS: usize = 100_000;
/// Plan 298 §G3 latency target for k=8, d=32.
const TARGET_NS_K8_D32: f64 = 200.0;

/// LCG for deterministic weight generation (no fastrand dep at the bench level
/// — keeps the bench self-contained; matches `tests/bench_294_ict_g2.rs`).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_f32(&mut self) -> f32 {
        // Numerical Recipes LCG.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Top 24 bits → [0, 1).
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Build a `[k*d]` weight slice with full-rank structure — the worst-case
/// runtime (all rows significant, all k*(k-1)/2 pairs evaluated).
fn build_full_rank_weights(k: usize, d: usize) -> Vec<f32> {
    let mut rng = Lcg::new(0xBEEF_298u64);
    let mut w = vec![0.0_f32; k * d];
    for row in 0..k {
        let mut nrm = 0.0_f32;
        for j in 0..d {
            let v = rng.next_f32() * 2.0 - 1.0;
            w[row * d + j] = v;
            nrm += v * v;
        }
        // Normalize each row to unit norm so all survive the epsilon filter.
        let nrm = nrm.sqrt();
        if nrm > 0.0 {
            for j in 0..d {
                w[row * d + j] /= nrm;
            }
        }
    }
    w
}

fn bench_combo(k: usize, d: usize) -> f64 {
    let clf = CosineSmearClassifier::default();
    let weights = build_full_rank_weights(k, d);
    let pairs = k * (k - 1) / 2;
    let mut scratch = vec![0.0_f32; k + pairs];

    // Warmup.
    let mut sink = 0.0_f32;
    for _ in 0..WARMUP_ITERS {
        let r = clf.classify(
            black_box(&weights),
            black_box(k),
            black_box(d),
            black_box(&mut scratch),
        );
        sink += r.semantic_distance;
    }
    if black_box(sink.is_nan()) {
        eprintln!("warmup sink nan (impossible for finite inputs)");
    }

    // Timed.
    let mut sink2 = 0.0_f32;
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let r = clf.classify(
            black_box(&weights),
            black_box(k),
            black_box(d),
            black_box(&mut scratch),
        );
        sink2 += r.semantic_distance;
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;
    if black_box(sink2.is_nan()) {
        eprintln!("timed sink nan (impossible for finite inputs)");
    }
    elapsed_ns / BENCH_ITERS as f64
}

fn main() {
    // Debug builds don't engage SIMD autovectorization and the k*(k-1)/2
    // pairwise dot loop runs ~5× slower; the 200 ns plasma-tier target is
    // unreachable in debug. We scale the threshold 5× in debug mode and print
    // a clear banner so the bench gives an honest verdict in both modes
    // without silently lying. Authoritative measurement requires
    // `cargo run --release --features smear_classifier --bench smear_classifier_bench`.
    let is_debug = cfg!(debug_assertions);
    let effective_target = if is_debug {
        TARGET_NS_K8_D32 * 5.0
    } else {
        TARGET_NS_K8_D32
    };

    println!("=== Plan 298 G3 — CosineSmearClassifier::classify latency ===");
    if is_debug {
        println!(
            "⚠️  DEBUG build — target scaled 5× ({:.0}→{:.0} ns).",
            TARGET_NS_K8_D32, effective_target
        );
        println!("    Rerun with --release for the authoritative plasma-tier target.");
    }
    println!(
        "warmup={}, timed={}, target k=8 d=32 ≤ {:.0} ns{}",
        WARMUP_ITERS,
        BENCH_ITERS,
        effective_target,
        if is_debug { " (debug-scaled)" } else { "" }
    );
    println!();

    let mut results: Vec<(usize, usize, f64)> = Vec::new();

    println!("{:>4} {:>4} {:>14} {:>10}", "k", "d", "ns/op", "verdict");
    for &k in &[2_usize, 4, 8] {
        for &d in &[8_usize, 16, 32] {
            let ns = bench_combo(k, d);
            let verdict = if k == 8 && d == 32 {
                if ns <= effective_target {
                    "PASS ✅"
                } else {
                    "FAIL ❌"
                }
            } else {
                "—"
            };
            println!("{:>4} {:>4} {:>14.1} {:>10}", k, d, ns, verdict);
            results.push((k, d, ns));
        }
    }

    println!();

    // ── G3 verdict. ──
    let k8_d32 = results
        .iter()
        .find(|(k, d, _)| *k == 8 && *d == 32)
        .map(|(_, _, ns)| *ns)
        .expect("k=8, d=32 must be in the sweep");
    let pass = k8_d32 <= effective_target;
    if pass {
        println!(
            "G3 PASS: k=8, d=32 at {:.1} ns/op ≤ {:.0} ns target{}.",
            k8_d32,
            effective_target,
            if is_debug { " (debug-scaled)" } else { "" }
        );
        if is_debug {
            println!("Note: debug-mode PASS is necessary-but-not-sufficient;");
            println!("      rerun in release for the authoritative plasma-tier gate.");
        } else {
            println!("Classifier is viable for plasma-tier audit cadence.");
        }
    } else {
        println!(
            "G3 FAIL: k=8, d=32 at {:.1} ns/op > {:.0} ns target{}.",
            k8_d32,
            effective_target,
            if is_debug { " (debug-scaled)" } else { "" }
        );
        println!("Per Plan 298 T3.4: keep opt-in. The classifier is correct");
        println!("(G1 passes) and useful (G2 passes) but exceeds the plasma");
        println!("latency budget. Document the failure mode in");
        println!(".benchmarks/298_smear_classifier_goat.md.");
    }

    // Exit code: 0 on PASS, non-zero on FAIL (so CI can pick it up).
    std::process::exit(if pass { 0 } else { 1 });
}
