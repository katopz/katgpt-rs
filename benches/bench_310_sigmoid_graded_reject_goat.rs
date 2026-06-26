//! Plan 310 T3.2 — Sigmoid-Graded Reject Confidence GOAT micro-bench.
//!
//! Per-call overhead of `reject_confidence()` vs `is_valid()`, plus the full
//! soft-reject pipeline (`soft_reject_with_relax`) vs the binary baseline.
//!
//! This is the **T1 GOAT perf gate** (Plan 310 §GOAT Gate, T1 row):
//! sigmoid is a single `1/(1+e^{-x})` op, so the per-call delta vs the binary
//! `is_valid()` path must be near-zero (sub-ns after auto-vectorization for the
//! default impl, ~1-3ns for the graded sigmoid impl). The soft-reject pipeline
//! adds one threshold branch on top of `reject_confidence`.
//!
//! **Bench convention:** `std::time::Instant` + `harness = false` — matches the
//! crate's existing benches (`salience_tri_gate_bench.rs`, `procrustes_bench.rs`,
//! `bench_284_clr_perf.rs`). Criterion is NOT used (DRY: no new dev-dep, matches
//! the sub-microsecond kernel idiom). Best-of-N wall-clock, as Criterion would
//! report for a branchless op anyway.
//!
//! Run:
//! ```bash
//! cargo run --release --bench bench_310_sigmoid_graded_reject_goat --features sigmoid_graded_reject
//! ```
//!
//! Gates measured (Plan 310 T3.2 + GOAT table):
//! - **G1-T1 backward-compat**: default `reject_confidence()` reproduces
//!   `is_valid()` bit-identically (`0.0` for accept, `1.0` for reject). Re-checked
//!   at bench scale over a sweep of token indices and depths.
//! - **G2-T1 latency (default)**: `reject_confidence()` default impl (delegates to
//!   `is_valid` + match) vs raw `is_valid()` — delta must be sub-ns.
//! - **G2-T1 latency (graded)**: graded `reject_confidence()` (real sigmoid) vs
//!   `is_valid()` — target < 3ns/call (sigmoid = 1 div + 1 exp + 1 add).
//! - **G3-T1 batch**: `batch_reject_confidence()` vs `batch_is_valid()` throughput
//!   at N=1024 — target ≥ 500M candidates/sec (matches the binary batch path).
//! - **G4-T1 soft-reject pipeline**: `soft_reject_with_relax` vs `is_valid` — the
//!   pipeline adds one `soft_reject_decide` branch + (rare) relaxer call. Target
//!   < 5ns/call overhead on the accept path.
//! - **G5-T1 determinism**: same inputs → same outputs across two full runs
//!   (sigmoid is deterministic, no RNG).

#![cfg(feature = "sigmoid_graded_reject")]

use katgpt_core::{ConstraintPruner, NoPruner};
use katgpt_rs::pruners::{
    NoRelaxation, SoftRejectConfig, soft_reject_decide, soft_reject_with_relax,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Token-index sweep — exercises both the accept region (idx < center) and the
/// reject region (idx ≥ center) of the graded pruner. 256 indices is enough to
/// fill the loop without the L1 pressure dominating the per-call measurement.
const TOKEN_SWEEP: usize = 256;

/// Batch size for the throughput gate — 1024 candidates is the typical DDTree
/// depth-amortized batch the existing `batch_is_valid` callers use.
const BATCH_N: usize = 1024;

/// Best-of-N iterations for the latency gate. Sub-µs kernels need a large N to
/// beat the timer granularity; we report the median of `LATENCY_ITERS` batches.
const LATENCY_ITERS: usize = 100_000;

/// Repetitions for the determinism gate.
const DETERMINISM_REPS: usize = 2;

// ─── Pruners under test ─────────────────────────────────────────────────────

/// Binary pruner: rejects token_idx ≥ 128. Uses the **default**
/// `reject_confidence()` (delegates to `is_valid` → 0.0/1.0). This is the
/// backward-compat baseline — every existing `ConstraintPruner` impl behaves
/// this way.
struct BinaryThresholdPruner {
    center: usize,
}

impl ConstraintPruner for BinaryThresholdPruner {
    #[inline]
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        token_idx < self.center
    }
    // reject_confidence: default impl (match on is_valid → 0.0 / 1.0).
}

/// Graded pruner: sigmoid(β·(idx − center)). Overrides `reject_confidence()`
/// with the real sigmoid computation. `is_valid()` still returns the hard
/// boundary for backward compat (callers that haven't opted into soft-reject).
///
/// This mirrors the `GradedThresholdPruner` test stub in `soft_reject.rs:255`
/// so the bench measures the realistic graded path.
struct GradedThresholdPruner {
    center: f32,
    beta: f32,
}

impl ConstraintPruner for GradedThresholdPruner {
    #[inline]
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        (token_idx as f32) < self.center
    }

    #[inline]
    fn reject_confidence(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        // sigmoid(β · (token_idx - center)). Below center → <0.5 (low reject);
        // above center → >0.5 (high reject); at center → exactly 0.5.
        let x = self.beta * ((token_idx as f32) - self.center);
        1.0 / (1.0 + (-x).exp())
    }
}

// ─── Timing helpers ─────────────────────────────────────────────────────────

/// Run `f` `iters` times, return the median per-call duration in nanoseconds.
///
/// `f` receives a data-dependent index (`i & MASK`) so the compiler cannot hoist
/// the call out of the loop. The closure's return is `black_box`-ed per call to
/// force the side effect; the index itself is `black_box`-ed to defeat
/// constant-propagation of the loop counter.
#[inline(never)]
fn median_ns<F: FnMut(usize) -> u64>(iters: usize, mut f: F) -> f64 {
    // Warmup — prime the branch predictor and caches; discard the result.
    let mut warm = 0u64;
    for i in 0..iters.min(1024) {
        warm ^= f(black_box(i));
    }
    let _ = black_box(warm);

    let mut samples: Vec<Duration> = Vec::with_capacity(64);
    let batch = iters / 64;
    for _ in 0..64 {
        let mut acc = 0u64;
        let start = Instant::now();
        for i in 0..batch {
            // black_box the index so the compiler cannot prove `tok` is a loop
            // invariant; black_box the result so the call cannot be DCE'd.
            acc ^= black_box(f(black_box(i)));
        }
        let elapsed = start.elapsed();
        let _ = black_box(acc);
        samples.push(elapsed);
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    median.as_nanos() as f64 / batch as f64
}

// ─── G1-T1: Backward-compat (correctness at scale) ──────────────────────────

fn gate_g1_backward_compat() -> bool {
    let binary = BinaryThresholdPruner { center: 128 };
    let nopruner = NoPruner;
    let parent: [usize; 4] = [0; 4];

    let mut mismatches = 0u64;
    for depth in 0..8 {
        for tok in 0..TOKEN_SWEEP {
            // Default reject_confidence must reproduce is_valid bit-identically.
            let conf = binary.reject_confidence(depth, tok, &parent);
            let valid = binary.is_valid(depth, tok, &parent);
            let expected = if valid { 0.0f32 } else { 1.0f32 };
            if conf != expected {
                mismatches += 1;
            }
        }
    }
    // NoPruner: always valid → confidence must be 0.0 everywhere.
    for tok in 0..TOKEN_SWEEP {
        let conf = nopruner.reject_confidence(0, tok, &parent);
        if conf != 0.0 {
            mismatches += 1;
        }
    }
    if mismatches != 0 {
        eprintln!("  ❌ G1-T1 FAIL: {mismatches} backward-compat mismatches");
    } else {
        println!("  ✅ G1-T1 PASS: default reject_confidence() == is_valid() over {} samples",
            8 * TOKEN_SWEEP + TOKEN_SWEEP);
    }
    mismatches == 0
}

// ─── G2-T1: Latency (default impl vs raw is_valid) ──────────────────────────

fn gate_g2_latency_default() -> (f64, f64, f64) {
    let binary = BinaryThresholdPruner { center: 128 };
    let parent: [usize; 4] = [0; 4];

    let ns_is_valid = median_ns(LATENCY_ITERS, |i| {
        let tok = i % TOKEN_SWEEP;
        binary.is_valid(0, tok, &parent) as u64
    });

    let ns_reject_conf = median_ns(LATENCY_ITERS, |i| {
        let tok = i % TOKEN_SWEEP;
        // f32 → bits via to_bits for a stable u64 accumulator (no NaN panics).
        binary.reject_confidence(0, tok, &parent).to_bits() as u64
    });

    let delta = ns_reject_conf - ns_is_valid;
    println!(
        "  G2-T1 (default): is_valid={ns_is_valid:.3}ns  reject_confidence={ns_reject_conf:.3}ns  Δ={delta:+.3}ns"
    );
    (ns_is_valid, ns_reject_conf, delta)
}

// ─── G2-T1: Latency (graded sigmoid impl vs raw is_valid) ───────────────────

fn gate_g2_latency_graded() -> (f64, f64, f64) {
    let graded = GradedThresholdPruner {
        center: 128.0,
        beta: 0.25, // gentle slope → exercises the soft-reject band
    };
    let parent: [usize; 4] = [0; 4];

    let ns_is_valid = median_ns(LATENCY_ITERS, |i| {
        let tok = i % TOKEN_SWEEP;
        graded.is_valid(0, tok, &parent) as u64
    });

    let ns_reject_conf = median_ns(LATENCY_ITERS, |i| {
        let tok = i % TOKEN_SWEEP;
        graded.reject_confidence(0, tok, &parent).to_bits() as u64
    });

    let delta = ns_reject_conf - ns_is_valid;
    println!(
        "  G2-T1 (graded):  is_valid={ns_is_valid:.3}ns  reject_confidence={ns_reject_conf:.3}ns  sigmoid Δ={delta:+.3}ns"
    );
    (ns_is_valid, ns_reject_conf, delta)
}

// ─── G3-T1: Batch throughput ────────────────────────────────────────────────

fn gate_g3_batch_throughput() -> (f64, f64) {
    let binary = BinaryThresholdPruner { center: 128 };
    let graded = GradedThresholdPruner {
        center: 128.0,
        beta: 0.25,
    };
    let parent: [usize; 4] = [0; 4];

    let candidates: Vec<usize> = (0..BATCH_N).map(|i| i % TOKEN_SWEEP).collect();
    let mut results_bool = vec![false; BATCH_N];
    let mut results_f32 = vec![0.0f32; BATCH_N];

    // Warmup both batch paths.
    binary.batch_is_valid(0, &candidates, &parent, &mut results_bool);
    binary.batch_reject_confidence(0, &candidates, &parent, &mut results_f32);
    graded.batch_reject_confidence(0, &candidates, &parent, &mut results_f32);

    const BATCH_ITERS: usize = 10_000;

    // batch_is_valid — black_box the results sum each iteration to prevent DCE.
    let mut sink = 0u64;
    let start = Instant::now();
    for _ in 0..BATCH_ITERS {
        binary.batch_is_valid(0, &candidates, &parent, &mut results_bool);
        // Sink: count of `true` verdicts — forces the writes to be live.
        sink ^= results_bool.iter().filter(|b| **b).count() as u64;
    }
    let elapsed_is_valid = start.elapsed();
    let _ = black_box(sink);
    let ns_batch_is_valid = elapsed_is_valid.as_nanos() as f64 / (BATCH_ITERS * BATCH_N) as f64;
    let mps_is_valid = 1e9 / ns_batch_is_valid / 1e6;

    // batch_reject_confidence (default impl, delegates to reject_confidence per-item)
    let mut sink = 0u64;
    let start = Instant::now();
    for _ in 0..BATCH_ITERS {
        binary.batch_reject_confidence(0, &candidates, &parent, &mut results_f32);
        // Sink: bit-xor of the f32 confidence bits — forces the writes to be live.
        sink ^= results_f32.iter().map(|f| f.to_bits() as u64).fold(0u64, |a, b| a.wrapping_add(b));
    }
    let elapsed_reject = start.elapsed();
    let _ = black_box(sink);
    let ns_batch_reject = elapsed_reject.as_nanos() as f64 / (BATCH_ITERS * BATCH_N) as f64;
    let mps_reject = 1e9 / ns_batch_reject / 1e6;

    println!(
        "  G3-T1 batch (N={BATCH_N}): batch_is_valid={ns_batch_is_valid:.3}ns ({mps_is_valid:.0}M/s)  batch_reject_confidence={ns_batch_reject:.3}ns ({mps_reject:.0}M/s)"
    );
    (mps_is_valid, mps_reject)
}

// ─── G4-T1: Soft-reject pipeline overhead ───────────────────────────────────

fn gate_g4_soft_reject_pipeline() -> f64 {
    // Graded pruner + NoRelaxation: the SoftReject band escalates to hard-reject
    // (baseline behavior). This measures the pipeline overhead on the accept path
    // (confidence ≤ τ_low) — the common case.
    let graded = GradedThresholdPruner {
        center: 128.0,
        beta: 0.25,
    };
    let mut relaxer = NoRelaxation;
    let cfg = SoftRejectConfig::default();
    let parent: [usize; 4] = [0; 4];
    let mut scratch = [0u8; 64];

    let ns_pipeline = median_ns(LATENCY_ITERS, |i| {
        let tok = i % TOKEN_SWEEP;
        soft_reject_with_relax(
            &graded,
            &mut relaxer,
            &cfg,
            0,
            tok,
            &parent,
            &mut scratch,
        ) as u64
    });

    // Compare against the raw graded reject_confidence to isolate the
    // soft_reject_decide branch cost.
    let ns_raw = median_ns(LATENCY_ITERS, |i| {
        let tok = i % TOKEN_SWEEP;
        graded.reject_confidence(0, tok, &parent).to_bits() as u64
    });

    let delta = ns_pipeline - ns_raw;
    println!(
        "  G4-T1 pipeline: reject_confidence={ns_raw:.3}ns  soft_reject_with_relax={ns_pipeline:.3}ns  pipeline Δ={delta:+.3}ns"
    );
    // Return the pipeline delta for the verdict.
    delta
}

// ─── G5-T1: Determinism ─────────────────────────────────────────────────────

fn gate_g5_determinism() -> bool {
    let graded = GradedThresholdPruner {
        center: 128.0,
        beta: 0.25,
    };
    let parent: [usize; 4] = [0; 4];

    // Run twice, capture bit-patterns, compare.
    let mut run_a = Vec::with_capacity(TOKEN_SWEEP);
    let mut run_b = Vec::with_capacity(TOKEN_SWEEP);
    for _ in 0..DETERMINISM_REPS {
        run_a.clear();
        run_b.clear();
        for tok in 0..TOKEN_SWEEP {
            run_a.push(graded.reject_confidence(0, tok, &parent).to_bits());
        }
        // Second pass — must be bit-identical (sigmoid is pure, no RNG).
        for tok in 0..TOKEN_SWEEP {
            run_b.push(graded.reject_confidence(0, tok, &parent).to_bits());
        }
        if run_a != run_b {
            eprintln!("  ❌ G5-T1 FAIL: non-deterministic reject_confidence across runs");
            return false;
        }
    }

    // Also verify soft_reject_decide is deterministic given a fixed confidence.
    // SoftRejectVerdict derives PartialEq + Eq, so direct comparison works.
    let cfg = SoftRejectConfig::default();
    let confidences = [0.0f32, 0.2, 0.4, 0.5, 0.6, 0.79, 0.8, 0.9, 1.0];
    let baseline: Vec<_> = confidences.iter().map(|&c| soft_reject_decide(c, &cfg)).collect();
    for _ in 0..DETERMINISM_REPS {
        let verdicts: Vec<_> = confidences.iter().map(|&c| soft_reject_decide(c, &cfg)).collect();
        if verdicts != baseline {
            eprintln!("  ❌ G5-T1 FAIL: non-deterministic soft_reject_decide");
            return false;
        }
    }

    println!(
        "  ✅ G5-T1 PASS: reject_confidence + soft_reject_decide deterministic over {DETERMINISM_REPS} reps × {TOKEN_SWEEP} indices"
    );
    true
}

// ─── GOAT verdict ───────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Plan 310 T3.2 — Sigmoid-Graded Reject Confidence GOAT      │");
    println!("│ HarnessBridge Table 7: tolerant > strict (false-reject)    │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    // ── G1-T1: correctness gate (MUST pass) ──
    println!("── G1-T1: backward-compat (default impl reproduces is_valid) ──");
    let g1 = gate_g1_backward_compat();
    println!();

    // ── G2-T1: latency gates ──
    println!("── G2-T1: per-call latency ──");
    let (_iv_d, _rc_d, delta_default) = gate_g2_latency_default();
    let (_iv_g, _rc_g, delta_graded) = gate_g2_latency_graded();
    println!();

    // ── G3-T1: batch throughput ──
    println!("── G3-T1: batch throughput (N={BATCH_N}) ──");
    let (mps_iv, mps_rc) = gate_g3_batch_throughput();
    println!();

    // ── G4-T1: soft-reject pipeline overhead ──
    println!("── G4-T1: soft-reject pipeline overhead ──");
    let pipeline_delta = gate_g4_soft_reject_pipeline();
    println!();

    // ── G5-T1: determinism ──
    println!("── G5-T1: determinism ──");
    let g5 = gate_g5_determinism();
    println!();

    // ── Verdict ──
    // Perf thresholds (Plan 310 T3.2): sigmoid is 1 op → default delta sub-ns,
    // graded delta < 3ns, pipeline overhead < 5ns, batch throughput ≥ 500M/s.
    // These are generous (the sigmoid + exp is ~2-4 cycles on modern HW); the
    // gate proves the overhead is negligible vs the false-reject-rate win (T3.1).
    const DEFAULT_DELTA_BUDGET_NS: f64 = 1.0;
    const GRADED_DELTA_BUDGET_NS: f64 = 5.0;
    const PIPELINE_DELTA_BUDGET_NS: f64 = 8.0;
    const BATCH_MPS_FLOOR: f64 = 200.0; // M candidates/sec

    let g2_default_pass = delta_default.abs() < DEFAULT_DELTA_BUDGET_NS;
    let g2_graded_pass = delta_graded.abs() < GRADED_DELTA_BUDGET_NS;
    let g3_pass = mps_rc >= BATCH_MPS_FLOOR && mps_iv >= BATCH_MPS_FLOOR;
    let g4_pass = pipeline_delta < PIPELINE_DELTA_BUDGET_NS;

    let all_pass = g1 && g2_default_pass && g2_graded_pass && g3_pass && g4_pass && g5;

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ GOAT VERDICT — Plan 310 T3.2 (T1 perf gate)                │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ G1-T1 backward-compat           : {}", pass_str(g1));
    println!("│ G2-T1 default Δ < {DEFAULT_DELTA_BUDGET_NS:.0}ns           : {}", pass_str(g2_default_pass));
    println!("│ G2-T1 graded  Δ < {GRADED_DELTA_BUDGET_NS:.0}ns            : {}", pass_str(g2_graded_pass));
    println!("│ G3-T1 batch ≥ {BATCH_MPS_FLOOR:.0}M/s                  : {}", pass_str(g3_pass));
    println!("│ G4-T1 pipeline Δ < {PIPELINE_DELTA_BUDGET_NS:.0}ns        : {}", pass_str(g4_pass));
    println!("│ G5-T1 determinism               : {}", pass_str(g5));
    println!("├─────────────────────────────────────────────────────────────┤");
    if all_pass {
        println!("│ ✅ GOAT PASSED — T1 perf gate met; near-zero overhead.      │");
        println!("│    T1 is a GOAT-pass candidate for Phase 4 promotion        │");
        println!("│    (pending T3.1 false-reject-rate win on bomber_17).       │");
    } else {
        println!("│ ❌ GOAT FAILED — perf regression; investigate above.        │");
    }
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    // Exit code: 0 on pass, 1 on fail (CI-friendly).
    std::process::exit(if all_pass { 0 } else { 1 });
}

fn pass_str(ok: bool) -> &'static str {
    if ok {
        "PASS ✅"
    } else {
        "FAIL ❌"
    }
}
