//! Linking-Fold GOAT gate bench (Plan 410 Phase 4 T4.1).
//!
//! Exercises G1 (correctness smoke), G2 (perf), and G5 (determinism) for the
//! `linking_fold` primitives distilled from Ren & Lim, "Low-dimensional
//! topology of deep neural networks" (arXiv:2606.31856, ICML 2026).
//!
//! G3 (no-regression) is the feature-flag build matrix, verified externally
//! (`cargo check --features linking_fold` clean). G4 (alloc-free hot path)
//! lives in a separate CountingAllocator test binary
//! (`tests/linking_fold_alloc_check.rs`) so the allocator doesn't pick up
//! allocations from this bench's detector fixture construction.
//!
//! # Gates measured here
//!
//! - **G1 (correctness smoke)**: synthetic thickened Hopf link (paper §G.1)
//!   → `link = ±1`; two unlinked circles → `link = 0`; one coordinate-fold
//!   pass per axis on the Hopf link → `link = 0`. Mirrors the three headline
//!   unit tests but at the bench's larger n=1000 scale (vs n=80 in the lib
//!   tests), confirming the detector scales.
//! - **G2 detector cold-path**: `detect_linking` on n = 2×200 point clouds
//!   at d = 8 (HLA scale), median of 11 runs. **Target: ≤ 500 ms**
//!   (audit-cadence budget, Issue 050 Option A, resolved 2026-07-07).
//!   The original plan budgeted 50 ms @ n = 2×1000; the brute-force
//!   implementation is O(β²) and that target is unreachable without
//!   algorithmic work (tracked as the Option B follow-up). The detector
//!   is explicitly audit-cadence (run once per session / sleep-cycle),
//!   not per-tick — see the `linking_detector.rs` module doc.
//! - **G2 fold hot-path**: `fold_projection_into` and `fold_gelu_into` median
//!   latency at d = 8 (HLA tick budget) and d = 64 (shard scale). Targets:
//!   ≤ 50 ns/call at d = 8, ≤ 500 ns/call at d = 64.
//! - **G5 (determinism)**: `detect_linking` returns the same integer `link`
//!   across 3 runs on the same input; `fold_projection_into` is bit-identical
//!   across 100 runs.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/lf_goat cargo bench -p katgpt-core \
//!   --features linking_fold --bench bench_410_linking_fold_goat -- --nocapture
//! ```
//!
//! Or, working around the intermittent macOS dyld/trustd launch stall
//! (documented in Plan 326 / bench_327):
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/lf_goat target/release/deps/bench_410_linking_fold_goat-* --nocapture
//! ```

#![cfg(feature = "linking_fold")]

use katgpt_core::linking_fold::{
    LinkingDetectorConfig, LinkingVerdict, detect_linking, detect_linking_into,
    fold_gelu_into, fold_projection_into,
};
use std::hint::black_box;
use std::time::Instant;

// ── Constants ──────────────────────────────────────────────────────────────

/// HLA-scale ambient dimension (the 5 synced affect scalars + 3 spares).
const D_HLA: usize = 8;
/// Shard-scale ambient dimension (NeuronShard style_weights).
const D_SHARD: usize = 64;
/// Per-cloud point count for the G2 detector cold-path gate.
///
/// The original plan budgeted n=2×1000 at ≤50ms; the brute-force Issue-050
/// path was O(β_X·β_Y·L²·n_sub²) — measured 407 ms at n=2×200 and minutes
/// extrapolated at n=2×1000 — so Issue 050 Option A (2026-07-07) accepted a
/// 500 ms audit budget at this n. The Issue 757 Option B work (2026-09-12:
/// single-pass k-NN, longest-first witness ordering, chunk-level certified
/// Gauss pruning, Y-cycle grid) landed 3.67 ms here — 31.6× under the audit
/// budget — and RESTORED the original 50 ms @ n=2×1000 target (27.9 ms
/// linked; enforced as G2b). The detector stays audit-cadence; see the
/// `linking_detector.rs` module doc for the n-scaling guidance.
const N_DETECTOR: usize = 200;
/// Fold latency iterations (matches bench_377's ITERS).
const FOLD_ITERS: usize = 10_000;
/// Detector latency iterations — cold path, so fewer runs (each is expensive).
const DETECTOR_ITERS: usize = 11;
/// **Audit-cadence budget** (Issue 050 Option A, resolved 2026-07-07). The
/// detector is explicitly audit-cadence (run once per session / sleep-cycle,
/// not per-tick) — see `linking_detector.rs` module doc. A 500ms budget at
/// n=2×200 is the fit-for-purpose target for that cadence. Measured cost:
///   n=2×80  (lib tests):   ~25 ms
///   n=2×200 (this bench):  ~407 ms ✅ (this budget)
///   n=2×1000 (plan target): minutes (extrapolated — do not call without subsampling)
///
/// The `DETECTOR_ORIG_BUDGET_MS` below is reported alongside as historical
/// context only — the detector does NOT meet it, and reaching it requires
/// the Option B optimization (batch early-exit + cycle pruning).
const DETECTOR_AUDIT_BUDGET_MS: f64 = 500.0;
/// Historical original-plan budget (50 ms @ n=2×1000) — RESTORED as an
/// enforced gate (G2b) by the Issue 757 Option B work: measured 27.9 ms
/// linked / 59.9 ms unlinked-separated at n=2×1000 (thickness-scaled
/// fixture; see `scaling_sweep`'s doc for why a fixed thickness breaks
/// detection at n ≥ ~300).
const DETECTOR_ORIG_BUDGET_MS: f64 = 50.0;
/// Per-cloud size for the G2b original-budget gate (Issue 757).
const N_DETECTOR_ORIG: usize = 1000;
/// Runs for the G2b gate (each ~30-60 ms — 5 runs keeps the bench bounded).
const DETECTOR_ORIG_RUNS: usize = 5;
const FOLD_HLA_BUDGET_NS: f64 = 50.0;
const FOLD_SHARD_BUDGET_NS: f64 = 500.0;

// ── Fixtures (mirror the lib tests, but at bench scale) ────────────────────

/// Thickened Hopf link embedded in R^d. The first 3 coords are the paper §G.1
/// parametrization; the remaining d−3 coords are zero (the link lives in a
/// 3D subspace of the ambient space, which is the typical HLA/shard case).
fn thickened_hopf_link_d(n_per_circle: usize, thickness: f32, d: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(d >= 3, "Hopf link needs d >= 3");
    let mut x = vec![0.0_f32; n_per_circle * d];
    let mut y = vec![0.0_f32; n_per_circle * d];
    for i in 0..n_per_circle {
        let t = (i as f32 / n_per_circle as f32) * 2.0 * std::f32::consts::PI;
        // Small normal perturbation (preserves topology).
        let nx = (i as f32 * 7.13).sin() * thickness;
        let ny = (i as f32 * 5.31).cos() * thickness;
        x[i * d] = t.cos() + nx;
        x[i * d + 1] = t.sin() + ny;
        x[i * d + 2] = 0.0 + (i as f32 * 3.7).sin() * thickness * 0.5;

        let s = (i as f32 / n_per_circle as f32) * 2.0 * std::f32::consts::PI;
        y[i * d] = 1.0 + s.cos() + (i as f32 * 4.1).sin() * thickness;
        y[i * d + 1] = 0.0 + (i as f32 * 6.7).cos() * thickness;
        y[i * d + 2] = s.sin() + (i as f32 * 2.9).sin() * thickness * 0.5;
    }
    (x, y)
}

/// Two unlinked circles in R^d (separated along axis 2 by 10 units).
fn unlinked_circles_d(n_per_circle: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(d >= 3, "needs d >= 3");
    let mut x = vec![0.0_f32; n_per_circle * d];
    let mut y = vec![0.0_f32; n_per_circle * d];
    for i in 0..n_per_circle {
        let t = (i as f32 / n_per_circle as f32) * 2.0 * std::f32::consts::PI;
        x[i * d] = t.cos();
        x[i * d + 1] = t.sin();
        x[i * d + 2] = 0.0;
        y[i * d] = t.cos();
        y[i * d + 1] = t.sin();
        y[i * d + 2] = 10.0;
    }
    (x, y)
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Plan 410 — Linking-Fold GOAT gate (G1 smoke / G2 perf / G5 det)");
    println!(
        "  D_HLA={D_HLA}, D_SHARD={D_SHARD}, N_DETECTOR={N_DETECTOR}, FOLD_ITERS={FOLD_ITERS}"
    );
    println!("══════════════════════════════════════════════════════════════════\n");

    let g1 = gate_g1_correctness_smoke();
    let g2_detector = gate_g2_detector_cold_path();
    let g2b_detector_orig = gate_g2b_detector_orig_budget();
    let (g2_fold_hla_abs, g2_fold_hla_gelu, g2_fold_shard_abs, g2_fold_shard_gelu) =
        gate_g2_fold_hot_path();
    let g5 = gate_g5_determinism();

    // Issue 757 T0.1: opt-in scaling sweep — keeps the default gate run
    // bounded while giving the bench a repeatable n-scaling instrument.
    if std::env::var("LINKING_SCALING").ok().as_deref() == Some("1") {
        scaling_sweep();
    }

    println!();
    println!("──────────────────────────────────────────────────────────────────");
    println!("  VERDICT");
    println!("──────────────────────────────────────────────────────────────────");
    println!(
        "  G1 correctness smoke:         {}  (Hopf=±1, unlinked=0, fold unlinks)",
        verdict(g1)
    );
    println!(
        "  G2 detector cold-path:        {}  ({:.2} ms; audit budget {:.0} ms @ n=2×{}, d={})",
        verdict(g2_detector.pass_audit),
        g2_detector.ms,
        DETECTOR_AUDIT_BUDGET_MS,
        N_DETECTOR,
        D_HLA
    );
    println!(
        "  G2b detector orig budget:     {}  ({:.2} ms ≤ {:.0} ms @ n=2×{}, linked, thickness π/n; the restored Plan-410 original target — Issue 757)",
        verdict(g2b_detector_orig.pass_audit),
        g2b_detector_orig.ms,
        DETECTOR_ORIG_BUDGET_MS,
        N_DETECTOR_ORIG
    );
    println!(
        "  G2 fold hot-path (Abs, D={}):  {}  ({:.2} ns ≤ {:.0} ns)",
        D_HLA,
        verdict(g2_fold_hla_abs.pass),
        g2_fold_hla_abs.ns,
        FOLD_HLA_BUDGET_NS
    );
    println!(
        "  G2 fold hot-path (Gelu, D={}): {}  ({:.2} ns ≤ {:.0} ns)",
        D_HLA,
        verdict(g2_fold_hla_gelu.pass),
        g2_fold_hla_gelu.ns,
        FOLD_HLA_BUDGET_NS
    );
    println!(
        "  G2 fold hot-path (Abs, D={}):  {}  ({:.2} ns ≤ {:.0} ns)",
        D_SHARD,
        verdict(g2_fold_shard_abs.pass),
        g2_fold_shard_abs.ns,
        FOLD_SHARD_BUDGET_NS
    );
    println!(
        "  G2 fold hot-path (Gelu, D={}): {}  ({:.2} ns ≤ {:.0} ns)",
        D_SHARD,
        verdict(g2_fold_shard_gelu.pass),
        g2_fold_shard_gelu.ns,
        FOLD_SHARD_BUDGET_NS
    );
    println!(
        "  G5 determinism:               {}  (detector + fold bit-identical)",
        verdict(g5)
    );
    println!();
    println!("  NOTE: G4 (alloc-free hot path) lives in tests/linking_fold_alloc_check.rs.");
    println!("  NOTE: G3 (no-regression) is the feature-flag build matrix.");
    println!();
}

fn verdict(pass: bool) -> &'static str {
    if pass { "✅ PASS" } else { "❌ FAIL" }
}

/// Issue 757 T0.1: real-path scaling curve for `detect_linking` — linked
/// (early-exit, time-to-witness) and unlinked (full sweep) fixtures at
/// n ∈ {100, 200, 500, 1000} per cloud, d = D_HLA, median of 3 after a
/// warm-up run. Opt-in via `LINKING_SCALING=1` so the default GOAT-gate run
/// stays bounded; the curve is recorded in the Issue 757 bench doc.
///
/// Fixture note (measured, Issue 757): the linked fixture's thickness SCALES
/// with n — `min(0.05, π/n)`. A fixed 0.05 under-samples the tube once the
/// ring spacing `2π/n` drops below it (n ≥ ~300): the kNN cycles go jagged
/// and the Gauss quadrature error swamps the ±1 integer, so BOTH the pre-
/// 757 and the 757 detector read link=0 (verified on the pre-757 code in a
/// clean worktree: n=100 ✓, n=200 ✓, n=300 ✗). `π/n` keeps the tube exactly
/// resolvable at every n (thickness = half the spacing).
fn scaling_sweep() {
    println!("\n── Issue 757 T0.1: scaling sweep (median of 3, d={D_HLA}) ──");
    println!("  n       linked ms   link   unlinked ms   link");
    let cfg = LinkingDetectorConfig::default();
    for n in [100usize, 200, 500, 1000] {
        let thickness = (std::f32::consts::PI / n as f32).min(0.05);
        let (xl, yl) = thickened_hopf_link_d(n, thickness, D_HLA);
        let (xu, yu) = unlinked_circles_d(n, D_HLA);
        let mut xp = vec![[0.0_f32; 3]; n];
        let mut yp = vec![[0.0_f32; 3]; n];
        let mut med3 = |x: &[f32], y: &[f32]| -> (f64, LinkingVerdict) {
            let mut vs = Vec::with_capacity(3);
            let mut verdict = LinkingVerdict::not_linked();
            for run in 0..=3 {
                let t0 = Instant::now();
                verdict = detect_linking_into(x, y, D_HLA, &mut xp, &mut yp, &cfg);
                if run > 0 {
                    vs.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
            }
            vs.sort_by(|a, b| a.total_cmp(b));
            (vs[1], verdict)
        };
        let (ms_l, v_l) = med3(&xl, &yl);
        let (ms_u, v_u) = med3(&xu, &yu);
        println!(
            "  {n:<6}  {ms_l:>9.2}   {:>4}   {ms_u:>9.2}     {:>4}",
            v_l.link, v_u.link
        );
    }
}

// ── G1: correctness smoke at bench scale ───────────────────────────────────

fn gate_g1_correctness_smoke() -> bool {
    println!("── G1: correctness smoke (n={N_DETECTOR}) ──");
    let cfg = LinkingDetectorConfig::default();

    // Hopf link → link = ±1.
    let (x, y) = thickened_hopf_link_d(N_DETECTOR, 0.05, D_HLA);
    let v_hopf = detect_linking(&x, &y, D_HLA, &cfg);
    let hopf_ok = v_hopf.linked && v_hopf.link.abs() == 1;
    println!(
        "   Hopf link:    linked={}, link={}  (expect linked=true, |link|=1)",
        v_hopf.linked, v_hopf.link
    );

    // Unlinked circles → link = 0.
    let (xu, yu) = unlinked_circles_d(N_DETECTOR, D_HLA);
    let v_unlinked = detect_linking(&xu, &yu, D_HLA, &cfg);
    let unlinked_ok = !v_unlinked.linked && v_unlinked.link == 0;
    println!(
        "   Unlinked:     linked={}, link={}  (expect linked=false, link=0)",
        v_unlinked.linked, v_unlinked.link
    );

    // Fold unlinks the Hopf link: apply one coordinate-fold pass per axis
    // reflecting onto the positive orthant (paper Fig. 9).
    let mut xf = x.clone();
    let mut yf = y.clone();
    let center_hla = vec![0.0_f32; D_HLA];
    for row in xf.chunks_mut(D_HLA) {
        fold_projection_into(row, &center_hla);
    }
    for row in yf.chunks_mut(D_HLA) {
        fold_projection_into(row, &center_hla);
    }
    let v_after = detect_linking(&xf, &yf, D_HLA, &cfg);
    let fold_ok = !v_after.linked && v_after.link == 0;
    println!(
        "   After fold:   linked={}, link={}  (expect linked=false, link=0)",
        v_after.linked, v_after.link
    );

    hopf_ok && unlinked_ok && fold_ok
}

// ── G2: detector cold-path latency ─────────────────────────────────────────

struct DetectorResult {
    /// Passes the recalibrated audit-cadence budget (500ms @ n=2×200).
    pass_audit: bool,
    ms: f64,
}

fn gate_g2_detector_cold_path() -> DetectorResult {
    println!(
        "\n── G2: detector cold-path (n=2×{N_DETECTOR}, d={D_HLA}, {DETECTOR_ITERS} runs, median) ──"
    );
    println!(
        "   audit-cadence budget: {DETECTOR_AUDIT_BUDGET_MS:.0}ms @ n=2×{N_DETECTOR}, d={D_HLA}"
    );
    println!(
        "   (orig plan budget {DETECTOR_ORIG_BUDGET_MS:.0}ms @ n=2×{N_DETECTOR_ORIG} enforced below as G2b — Issue 757 Option B landed.)"
    );
    let (x, y) = thickened_hopf_link_d(N_DETECTOR, 0.05, D_HLA);
    let cfg = LinkingDetectorConfig::default();

    // Warm-up (one run, not timed — fills caches, JIT-style first-call effects).
    let _ = black_box(detect_linking(
        black_box(&x),
        black_box(&y),
        black_box(D_HLA),
        &cfg,
    ));

    let mut samples_ms: Vec<f64> = Vec::with_capacity(DETECTOR_ITERS);
    for _ in 0..DETECTOR_ITERS {
        let t0 = Instant::now();
        let v = detect_linking(black_box(&x), black_box(&y), black_box(D_HLA), &cfg);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        // Keep the result live so the optimizer can't elide the call.
        let _ = black_box(v);
        samples_ms.push(elapsed);
    }
    samples_ms.sort_by(|a, b| a.total_cmp(b));
    let median_ms = samples_ms[samples_ms.len() / 2];
    let pass_audit = median_ms <= DETECTOR_AUDIT_BUDGET_MS;
    println!(
        "   median = {:.3} ms  (min {:.3}, max {:.3})",
        median_ms,
        samples_ms[0],
        *samples_ms.last().unwrap()
    );
    DetectorResult {
        pass_audit,
        ms: median_ms,
    }
}

// ── G2b: the restored original budget (Issue 757) ─────────────────────────

/// The Plan-410 original target, enforced again: `detect_linking` on the
/// linked thickness-scaled fixture at n=2×1000 must (a) still detect
/// (|link| = 1) and (b) fit ≤ 50 ms — the budget the brute-force path
/// missed by ~3 orders of magnitude and the Issue 757 work restored.
fn gate_g2b_detector_orig_budget() -> DetectorResult {
    println!(
        "\n── G2b: detector original budget (n=2×{N_DETECTOR_ORIG}, d={D_HLA}, {DETECTOR_ORIG_RUNS} runs, median) ──"
    );
    let thickness = (std::f32::consts::PI / N_DETECTOR_ORIG as f32).min(0.05);
    let (x, y) = thickened_hopf_link_d(N_DETECTOR_ORIG, thickness, D_HLA);
    let cfg = LinkingDetectorConfig::default();

    // Warm-up (not timed).
    let mut xp = vec![[0.0_f32; 3]; N_DETECTOR_ORIG];
    let mut yp = vec![[0.0_f32; 3]; N_DETECTOR_ORIG];
    let warm = detect_linking_into(&x, &y, D_HLA, &mut xp, &mut yp, &cfg);
    assert!(
        warm.linked && warm.link.abs() == 1,
        "G2b correctness: must detect |link|=1 at n=2×{N_DETECTOR_ORIG}, got {warm:?}"
    );
    println!(
        "   correctness: linked=true, |link|={} (thickness={thickness:.4})",
        warm.link.abs()
    );

    let mut samples_ms: Vec<f64> = Vec::with_capacity(DETECTOR_ORIG_RUNS);
    for _ in 0..DETECTOR_ORIG_RUNS {
        let t0 = Instant::now();
        let v = detect_linking_into(
            black_box(&x),
            black_box(&y),
            black_box(D_HLA),
            &mut xp,
            &mut yp,
            &cfg,
        );
        samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let _ = black_box(v);
    }
    samples_ms.sort_by(|a, b| a.total_cmp(b));
    let median_ms = samples_ms[samples_ms.len() / 2];
    let pass_audit = median_ms <= DETECTOR_ORIG_BUDGET_MS;
    println!(
        "   median = {:.3} ms  (min {:.3}, max {:.3})  [budget {:.0} ms]",
        median_ms,
        samples_ms[0],
        *samples_ms.last().unwrap(),
        DETECTOR_ORIG_BUDGET_MS
    );
    DetectorResult { pass_audit, ms: median_ms }
}

// ── G2: fold hot-path latency ──────────────────────────────────────────────

struct FoldResult {
    pass: bool,
    ns: f64,
}

#[allow(clippy::type_complexity)]
fn gate_g2_fold_hot_path() -> (FoldResult, FoldResult, FoldResult, FoldResult) {
    println!("\n── G2: fold hot-path ({FOLD_ITERS} iters/case, median of last 80%) ──");

    let r_hla_abs = bench_fold(D_HLA, FoldVariant::Abs);
    let r_hla_gelu = bench_fold(D_HLA, FoldVariant::Gelu);
    let r_shard_abs = bench_fold(D_SHARD, FoldVariant::Abs);
    let r_shard_gelu = bench_fold(D_SHARD, FoldVariant::Gelu);

    println!(
        "   Abs  D={:<3}: {:>8.2} ns  (budget {:.0} ns)",
        D_HLA, r_hla_abs.ns, FOLD_HLA_BUDGET_NS
    );
    println!(
        "   Gelu D={:<3}: {:>8.2} ns  (budget {:.0} ns)",
        D_HLA, r_hla_gelu.ns, FOLD_HLA_BUDGET_NS
    );
    println!(
        "   Abs  D={:<3}: {:>8.2} ns  (budget {:.0} ns)",
        D_SHARD, r_shard_abs.ns, FOLD_SHARD_BUDGET_NS
    );
    println!(
        "   Gelu D={:<3}: {:>8.2} ns  (budget {:.0} ns)",
        D_SHARD, r_shard_gelu.ns, FOLD_SHARD_BUDGET_NS
    );

    (r_hla_abs, r_hla_gelu, r_shard_abs, r_shard_gelu)
}

#[derive(Clone, Copy)]
enum FoldVariant {
    Abs,
    Gelu,
}

fn bench_fold(d: usize, variant: FoldVariant) -> FoldResult {
    // Deterministic input — vary slightly each iter to defeat const-folding,
    // but keep it cheap (no alloc in the timing loop).
    let mut state = vec![0.0_f32; d];
    for (i, s) in state.iter_mut().enumerate() {
        *s = (i as f32) * 0.01 - 0.3;
    }
    let center = vec![0.0_f32; d];
    let alpha = 10.0_f32;

    // Warm-up.
    for _ in 0..1_000 {
        match variant {
            FoldVariant::Abs => fold_projection_into(black_box(&mut state), black_box(&center)),
            FoldVariant::Gelu => {
                fold_gelu_into(black_box(&mut state), black_box(&center), black_box(alpha));
            }
        }
    }

    // Reset state to a known non-trivial value after warm-up (warm-up folds
    // it onto the positive half-line, where subsequent folds are no-ops —
    // still exercises the loop, but we want the negative-half-line path too).
    let base: Vec<f32> = (0..d).map(|i| (i as f32) * 0.013 - 0.4).collect();

    let mut samples_ns: Vec<f64> = Vec::with_capacity(FOLD_ITERS);
    for k in 0..FOLD_ITERS {
        // Re-seed state each iter so the fold always does real work (the
        // negative-half-line reflection). Vary by a tiny deterministic amount.
        let tweak = (k as f32) * 1e-6;
        for i in 0..d {
            state[i] = base[i] + tweak;
        }
        let t0 = Instant::now();
        match variant {
            FoldVariant::Abs => fold_projection_into(black_box(&mut state), black_box(&center)),
            FoldVariant::Gelu => {
                fold_gelu_into(black_box(&mut state), black_box(&center), black_box(alpha));
            }
        }
        samples_ns.push(t0.elapsed().as_nanos() as f64);
    }
    samples_ns.sort_by(|a, b| a.total_cmp(b));
    // Trim 10% off each end (outliers from scheduling jitter), take the mean
    // of the middle 80%. Median alone is fine too, but trimmed-mean is more
    // stable on macOS where dyld/trustd can spike single samples.
    let trim = FOLD_ITERS / 10;
    let trimmed = &samples_ns[trim..FOLD_ITERS - trim];
    let ns = trimmed.iter().sum::<f64>() / trimmed.len() as f64;

    let budget = if d <= D_HLA {
        FOLD_HLA_BUDGET_NS
    } else {
        FOLD_SHARD_BUDGET_NS
    };
    FoldResult {
        pass: ns <= budget,
        ns,
    }
}

// ── G5: determinism ────────────────────────────────────────────────────────

fn gate_g5_determinism() -> bool {
    println!("\n── G5: determinism ──");
    let (x, y) = thickened_hopf_link_d(N_DETECTOR, 0.05, D_HLA);
    let cfg = LinkingDetectorConfig::default();

    // Detector: same integer link across 3 runs.
    let v1 = detect_linking(&x, &y, D_HLA, &cfg);
    let v2 = detect_linking(&x, &y, D_HLA, &cfg);
    let v3 = detect_linking(&x, &y, D_HLA, &cfg);
    let det_ok = v1 == v2 && v2 == v3;
    println!(
        "   detector: link={} ×3, verdict equal = {}",
        v1.link, det_ok
    );

    // Fold: bit-identical across 100 runs (closed-form, no state).
    let base: Vec<f32> = (0..D_HLA).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let center = vec![0.0_f32; D_HLA];
    let mut reference = base.clone();
    fold_projection_into(&mut reference, &center);
    let mut fold_ok = true;
    for _ in 0..100 {
        let mut s = base.clone();
        fold_projection_into(&mut s, &center);
        if s != reference {
            fold_ok = false;
            break;
        }
    }
    println!("   fold_projection_into: bit-identical across 100 runs = {fold_ok}");

    det_ok && fold_ok
}

// (G4 alloc-free hot path lives in tests/linking_fold_alloc_check.rs —
// installing a CountingAllocator in this binary would skew the Instant::now
// timing loops above, so the two gates are deliberately separated.)
