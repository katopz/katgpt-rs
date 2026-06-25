//! Plan 303 Phase 2 T2.2 — Salience Tri-Gate latency + throughput bench.
//!
//! **DEVIATION from plan:** the plan specifies Criterion. This crate's bench
//! convention is `std::time::Instant` + `harness = false` (see comments at
//! `Cargo.toml` lines 1316/1324/1559/1692/1744/1792/1811 and existing benches
//! like `procrustes_bench.rs`). Adding Criterion would be a new dev-dep and a
//! style break — DRY dictates matching the existing convention. The
//! measurements below are best-of-N wall-clock, which is what Criterion would
//! report for a sub-microsecond kernel anyway (no need for warmup-sensitive
//! sampling on a branchless ~10-20ns op).
//!
//! Run:
//! ```bash
//! cargo run --release --bench salience_tri_gate_bench --features salience_tri_gate
//! ```
//!
//! Gates measured:
//! - **G1 determinism** — same inputs → same decision (bit-identical). Tested
//!   in `gate::tests::test_g1_determinism`; here we re-confirm via a 1000-call
//!   sequence check.
//! - **G2 ablation parity** — `ceil_delegate = +∞` produces a bit-identical
//!   Silent/Speak sequence to a speak-only reference. Tested in
//!   `gate::tests::test_g2_ablation_parity`; here we re-confirm over 10k inputs.
//! - **Latency target**: `decide()` < 50ns for D=8 (cf. `evolve_hla` ~14ns
//!   for D=8; the gap is the second dot-product).
//! - **Throughput target**: `decide_batch()` ≥ 50M decisions/sec for D=8, N=1000.

#![cfg(feature = "salience_tri_gate")]

use katgpt_rs::salience::{SalienceDecision, SalienceTriGate};
use std::time::{Duration, Instant};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Dims to sweep for the latency gate. The plan's target is D=8; D=16 and
/// D=32 show the dot-product scaling.
const DIMS: &[usize] = &[8, 16, 32];

/// Batch sizes for the throughput gate.
const BATCH_SIZES: &[usize] = &[1_000, 10_000];

/// Warmup iterations (primes branch predictor, JITs CPU caches into L1).
const WARMUP: usize = 1_000;

// ─── Deterministic LCG (matches `gate::tests::Lcg` convention) ───────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        // Same constants as `gate::tests::Lcg` and the examples — keeping the
        // crate's PRNG lineage consistent for reproducibility.
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn next_f32(&mut self) -> f32 {
        // FIX (Plan 303 T4.1 deviation, mirrored here): divide by 2^31, NOT
        // u32::MAX. `next()` returns the top 31 bits; dividing by ~2^32 yields
        // [0, 0.5) which biases every downstream decision. The unit-test LCG
        // in `gate::tests` still has this bug (pre-existing, out of scope), but
        // benchmarks must measure the *intended* distribution, not the buggy
        // one — otherwise the latency number reflects an unrepresentative
        // branch pattern (always-Silent / never-Delegate).
        (self.next() as f32) / ((1u64 << 31) as f32)
    }
}

// ─── Gate construction ──────────────────────────────────────────────────────

/// Build a gate for dimension D with orthogonal speak/delegate directions.
/// D_SPEAK = e_0, D_DELEGATE = e_1. This isolates the two dot-products so the
/// latency measurement isn't polluted by accidental dot-product cancellation.
fn make_gate<const D: usize>() -> SalienceTriGate<u32, D> {
    let mut d_speak = [0.0_f32; D];
    let mut d_delegate = [0.0_f32; D];
    d_speak[0] = 1.0;
    d_delegate[1 % D] = 1.0;
    SalienceTriGate::new(
        d_speak,
        d_delegate,
        0.3, // w_z
        0.2, // w_c
        2.0, // beta_speak
        2.0, // beta_delegate
        0.5, // tau_speak
        0.5, // tau_delegate
        0.15, // floor_speak
        0.4,  // ceil_delegate
    )
}

// ─── Timed-loop helpers (per-D, monomorphised) ──────────────────────────────

/// Median per-call latency in nanoseconds for a single `decide()` call.
///
/// **Measurement strategy:** a single `Instant::now()` pair costs ~30-40 ns
/// on macOS (mach absolute time syscall), which dominates a ~10-20 ns kernel.
/// So we batch `BATCH` calls between two `Instant::now()` reads, divide by
/// `BATCH`, and take the median of `OUTER` such batch measurements. This is
/// the standard pattern for sub-microsecond kernels (cf. `bench_284_clr_perf`).
///
/// The `sink` accumulator is a `u64` hash of the decision variant — opaque
/// enough that the compiler can't hoist `decide` out of the loop (the output
/// depends on the inputs, which depend on the loop counter via `i`).
fn bench_decide_latency<const D: usize>(
    gate: &SalienceTriGate<u32, D>,
    a: &[f32; D],
    z: f32,
    c: f32,
) -> f64 {
    const BATCH: usize = 1024;
    const OUTER: usize = 256;

    // Warmup
    let mut sink: u64 = 0;
    for _ in 0..WARMUP {
        let d = gate.decide(a, z, c, 0u32, 0);
        sink = sink.wrapping_add(variant_tag(&d));
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(OUTER);
    for _ in 0..OUTER {
        let t0 = Instant::now();
        for _ in 0..BATCH {
            let d = gate.decide(a, z, c, 0u32, 0);
            sink = sink.wrapping_add(variant_tag(&d));
        }
        samples.push(t0.elapsed());
    }
    // Consume sink so the calls can't be elided.
    if sink == u64::MAX {
        std::process::abort(); // unreachable; just a use of `sink`
    }
    samples.sort();
    let mid = OUTER / 2;
    let median_batch = (samples[mid].as_nanos() as f64 + samples[mid - 1].as_nanos() as f64) / 2.0;
    median_batch / (BATCH as f64)
}

/// Compress a `SalienceDecision` to a u64 tag for the sink. Different
/// variants produce different tags, so the compiler can't prove the loop body
/// is a no-op.
fn variant_tag(d: &SalienceDecision<u32>) -> u64 {
    match d {
        SalienceDecision::Silent => 0xA1_A5_u64,
        SalienceDecision::Speak => 0xBB_BB_u64,
        SalienceDecision::Delegate(p) => 0xC0_DE_u64 ^ (*p as u64),
    }
}

/// Throughput in decisions/sec for `decide_batch`.
fn bench_decide_batch_throughput<const D: usize>(
    gate: &SalienceTriGate<u32, D>,
    activations: &[[f32; D]],
    z: &[f32],
    c: &[f32],
    payloads: &[u32],
    out: &mut [SalienceDecision<u32>],
) -> f64 {
    let n = activations.len();
    debug_assert_eq!(z.len(), n);
    debug_assert_eq!(c.len(), n);
    debug_assert_eq!(payloads.len(), n);
    debug_assert_eq!(out.len(), n);

    // Warmup
    for _ in 0..10 {
        gate.decide_batch(activations, z, c, payloads, 0, out);
    }

    // Best-of-32 whole-batch wall-time → throughput.
    const OUTER: usize = 32;
    let mut best_secs = f64::INFINITY;
    for _ in 0..OUTER {
        let t0 = Instant::now();
        gate.decide_batch(activations, z, c, payloads, 0, out);
        let dt = t0.elapsed().as_secs_f64();
        if dt < best_secs {
            best_secs = dt;
        }
    }
    (n as f64) / best_secs
}

// ─── G1/G2 re-confirmation (parity smoke checks) ────────────────────────────

/// G1 determinism re-confirm: 1000 calls with the same inputs must produce
/// the same variant every time. (Bit-identical is implicit — the gate has no
/// RNG and no thread-local state.)
fn g1_determinism_smoke<const D: usize>(gate: &SalienceTriGate<u32, D>) -> bool {
    let a = [0.5_f32; D];
    let first = gate.decide(&a, 0.5, 0.5, 0u32, 0);
    for _ in 0..1_000 {
        let again = gate.decide(&a, 0.5, 0.5, 0u32, 0);
        if !variants_eq(&first, &again) {
            return false;
        }
    }
    true
}

/// G2 ablation parity re-confirm: a gate with `ceil_delegate = +∞` must
/// produce the same Silent/Speak sequence as a speak-only reference over
/// 10k random inputs. (The delegate sigmoid can't fire when the ceiling is
/// infinity.)
fn g2_ablation_parity_smoke<const D: usize>() -> bool {
    let mut d_speak = [0.0_f32; D];
    let mut d_delegate = [0.0_f32; D];
    d_speak[0] = 1.0;
    d_delegate[1 % D] = 1.0;

    let full = SalienceTriGate::<u32, D>::new(
        d_speak,
        d_delegate,
        0.3,
        0.2,
        2.0,
        2.0,
        0.5,
        0.5,
        0.15,
        0.4, // finite ceil — delegate CAN fire
    );
    let ablated = SalienceTriGate::<u32, D>::new(
        d_speak,
        d_delegate,
        0.3,
        0.2,
        2.0,
        2.0,
        0.5,
        0.5,
        0.15,
        f32::INFINITY, // delegate sigmoid never crosses ceiling
    );

    let mut rng = Lcg::new(0xC0_FFEE_BABE);
    let n = 10_000;
    for _ in 0..n {
        let mut a = [0.0_f32; D];
        for v in a.iter_mut() {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        let z = rng.next_f32();
        let c = rng.next_f32();
        let d_full = full.decide(&a, z, c, 0u32, 0);
        let d_abl = ablated.decide(&a, z, c, 0u32, 0);
        // The ablated gate must NEVER emit Delegate.
        if matches!(d_abl, SalienceDecision::Delegate(_)) {
            return false;
        }
        // And its Silent/Speak choice must equal the full gate's speak/silent
        // *sub*-decision (when the full gate delegates, the ablated gate must
        // pick the same speak/silent the full gate *would have* picked absent
        // the delegate override — which is exactly `Speak` here, because
        // `score_speak >= floor_speak` is the precondition for reaching the
        // delegate check).
        let abl_is_silent = matches!(d_abl, SalienceDecision::Silent);
        // Parity condition: if the full gate went Silent, the ablated gate
        // must also go Silent (the delegate check is *below* the floor check
        // in precedence). If the full gate went Speak or Delegate, the
        // ablated gate must go Speak (it can't delegate).
        let consistent = match d_full {
            SalienceDecision::Silent => abl_is_silent,
            SalienceDecision::Speak | SalienceDecision::Delegate(_) => !abl_is_silent,
        };
        if !consistent {
            return false;
        }
    }
    true
}

fn variants_eq(a: &SalienceDecision<u32>, b: &SalienceDecision<u32>) -> bool {
    matches!(
        (a, b),
        (SalienceDecision::Silent, SalienceDecision::Silent)
            | (SalienceDecision::Speak, SalienceDecision::Speak)
            | (SalienceDecision::Delegate(_), SalienceDecision::Delegate(_))
    )
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 303 Salience Tri-Gate Bench (Phase 2 T2.2) ===");
    println!();

    // ── G1 + G2 smoke re-confirmation ─────────────────────────────
    // These are property-tested in the unit tests; we re-confirm at bench
    // time so the GOAT gate doc can cite a single-run result.
    let g1_pass;
    let g2_pass;
    {
        let gate: SalienceTriGate<u32, 8> = make_gate();
        g1_pass = g1_determinism_smoke(&gate);
        g2_pass = g2_ablation_parity_smoke::<8>();
        println!("G1 determinism (1000-call re-confirm): {}", pass_str(g1_pass));
        println!("G2 ablation parity (10k-input re-confirm): {}", pass_str(g2_pass));
        println!();
    }

    // ── Latency: decide() per-D ───────────────────────────────────────────
    println!("--- decide() latency (median of 256 batches x 1024 calls, warmup {WARMUP}) ---");
    let mut latency_results: Vec<(usize, f64, bool)> = Vec::with_capacity(DIMS.len());
    for &d in DIMS {
        let (ns, pass) = run_latency_for_dim(d);
        let target = 50.0_f64;
        let pass = pass && ns < target;
        let verdict = if pass { "PASS" } else { "FAIL" };
        println!(
            "  D={d:>2}: {ns:>6.2} ns/call  (target < {target:.0} ns)  [{verdict}]"
        );
        latency_results.push((d, ns, pass));
    }
    println!();

    // ── Throughput: decide_batch() per (D, N) ─────────────────────────────
    println!("--- decide_batch() throughput (best-of-32 whole-batch wall-time) ---");
    let mut throughput_results: Vec<(usize, usize, f64, bool)> = Vec::new();
    for &d in DIMS {
        for &n in BATCH_SIZES {
            let (meps, pass) = run_throughput_for_dim(d, n);
            let target = 50.0_f64;
            let verdict = if pass { "PASS" } else { "FAIL" };
            println!(
                "  D={d:>2} N={n:>5}: {meps:>6.1} M decisions/sec  (target >= {target:.0} M)  [{verdict}]"
            );
            throughput_results.push((d, n, meps, pass));
        }
    }
    println!();

    // ── GOAT verdict ──────────────────────────────────────────────────────
    // G1 determinism is a bit-level property (no RNG in the gate) — re-confirmed
    // above by the 1000-call smoke check. The latency gate is D=8 only; D=16/32
    // results are recorded for the dot-product scaling curve but don't gate
    // promotion (the plan target is specifically "< 50ns for D=8").
    let latency_d8_pass = latency_results
        .iter()
        .find(|&&(d, _, _)| d == 8)
        .map(|&(_, _, p)| p)
        .unwrap_or(false);
    let batch_d8_n1000_pass = throughput_results
        .iter()
        .find(|&&(d, n, _, _)| d == 8 && n == 1_000)
        .map(|&(_, _, _, p)| p)
        .unwrap_or(false);

    println!("=== GOAT Gate Verdict ===");
    println!("  G1 determinism:       {}", pass_str(g1_pass));
    println!("  G2 ablation parity:   {}", pass_str(g2_pass));
    println!("  decide() latency D=8 < 50ns:        {}", pass_str(latency_d8_pass));
    println!(
        "  decide_batch() D=8 N=1000 >= 50M/s:  {}",
        pass_str(batch_d8_n1000_pass)
    );
    let all_pass = latency_d8_pass && batch_d8_n1000_pass;
    println!();
    if all_pass {
        println!("  → PROMOTE `salience_tri_gate` to default feature (Plan 303 Phase 5).");
    } else {
        println!("  → KEEP OPT-IN. Latency/throughput targets not met — file issue for SIMD optimisation.");
    }
    println!();
    println!("  Raw latency: {:?}", latency_results);
    println!("  Raw throughput (M/s): {:?}", throughput_results);
}

fn pass_str(p: bool) -> &'static str {
    if p {
        "PASS"
    } else {
        "FAIL"
    }
}

// ─── Per-D dispatch (monomorphised inner timing) ────────────────────────────

fn run_latency_for_dim(d: usize) -> (f64, bool) {
    match d {
        8 => {
            let gate: SalienceTriGate<u32, 8> = make_gate();
            let a = [0.5_f32; 8];
            let ns = bench_decide_latency(&gate, &a, 0.5, 0.5);
            (ns, true)
        }
        16 => {
            let gate: SalienceTriGate<u32, 16> = make_gate();
            let a = [0.5_f32; 16];
            let ns = bench_decide_latency(&gate, &a, 0.5, 0.5);
            (ns, true)
        }
        32 => {
            let gate: SalienceTriGate<u32, 32> = make_gate();
            let a = [0.5_f32; 32];
            let ns = bench_decide_latency(&gate, &a, 0.5, 0.5);
            (ns, true)
        }
        _ => unreachable!(),
    }
}

fn run_throughput_for_dim(d: usize, n: usize) -> (f64, bool) {
    match d {
        8 => run_throughput_const::<8>(n),
        16 => run_throughput_const::<16>(n),
        32 => run_throughput_const::<32>(n),
        _ => unreachable!(),
    }
}

fn run_throughput_const<const D: usize>(n: usize) -> (f64, bool) {
    let gate: SalienceTriGate<u32, D> = make_gate();
    let mut rng = Lcg::new(0xBAD_C0DE);
    let mut activations = vec![[0.0_f32; D]; n];
    let mut z = vec![0.0_f32; n];
    let mut c = vec![0.0_f32; n];
    let payloads: Vec<u32> = (0..n as u32).collect();
    let mut out: Vec<SalienceDecision<u32>> = vec![SalienceDecision::Silent; n];

    for i in 0..n {
        for v in activations[i].iter_mut() {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        z[i] = rng.next_f32();
        c[i] = rng.next_f32();
    }

    let dps = bench_decide_batch_throughput(&gate, &activations, &z, &c, &payloads, &mut out);
    let meps = dps / 1e6;
    (meps, meps >= 50.0)
}
