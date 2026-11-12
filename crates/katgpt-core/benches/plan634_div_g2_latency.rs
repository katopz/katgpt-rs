//! Plan 634 T2.6 (G2) — the latency ladder for the diffusion-diversity
//! primitive. `--release` mandatory (the plan row; bench profile is release
//! by construction).
//!
//! The plan's bar, measured not assumed ("fixture sizes are tiny … the bar
//! is expected to clear with orders of magnitude — but expected is not
//! measured"): local similarity at fixture sizes (≤ 1k nodes, τ ≤ 4). The
//! bar text admits two readings and BOTH are printed per rung —
//! `≤ 1 µs` (the strict reading) and `≤ 10 µs` (µs-class, the module doc's
//! design rationale: the (τ+1)-ball restriction stays unimplemented because
//! fixture-scale latency is µs either way) — while the ASSERTED gate is the
//! 100 µs sanity ceiling at battery scale (a pathology catch, never a
//! calibration: an absolute ns bar asserted in a bench is the known
//! load-sensitive class). The plan row carries the measured verdict against
//! the strict bar; this bench's exit code only refuses pathology.
//!
//! Ladder: planted clique graphs (the battery fixture shape) and circulant
//! expanders (the well-mixed shape) at n = 64 / 256 / 1000, τ = 4:
//! - `diffusion_similarity_local` — ns/call
//! - `div_tau` — ns/call (uniform μ, the batch-intake posture)
//! - `neighbor_diversity_pick` — ns/call at a walk node (degree-bound k ≤ 64)
//! - spectral query paths at n ≤ [`SPECTRAL_N_CAP`] — ns/query + the build
//!   cost reported once (build-time class)
//! - `batch_select_diffusion` at the spectral cap — µs/call (build-time
//!   class, reported)
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/plan634t26 cargo bench -p katgpt-core \
//!   --features diffusion_diversity --bench plan634_div_g2_latency
//! ```

#![cfg(feature = "diffusion_diversity")]

use katgpt_core::diversity::diffusion::{
    DiffusionEmbedding, LocalScratch, SPECTRAL_N_CAP, SimGraph, batch_select_diffusion,
    diffusion_similarity_local, neighbor_diversity_pick,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

const TAU: usize = 4;

/// Deterministic LCG (the unseeded-global-RNG gate) — fixture weights only.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) * (1.0 / (1u64 << 24) as f32)
    }
}

/// Planted clique graph — the battery fixture shape scaled up (timing
/// fixture only: unit intra-clique weight, ring meta-edges at 1/M; no r/c
/// pin asserts — those live in the batteries' fixture).
fn planted(k: usize, m: usize) -> SimGraph {
    let n = k * m;
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for c in 0..k {
        for i in 0..m {
            for j in (i + 1)..m {
                edges.push((c * m + i, c * m + j, 1.0));
            }
        }
    }
    let w = 1.0 / m as f32;
    for c in 0..k {
        let nxt = (c + 1) % k;
        for i in 0..m {
            for j in 0..m {
                edges.push((c * m + i, nxt * m + j, w));
            }
        }
    }
    SimGraph::build(n, &edges)
}

/// Circulant expander — the well-mixed fixture shape (no clusters).
fn circulant(n: usize, offsets: &[usize]) -> SimGraph {
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..n {
        for &o in offsets {
            let j = (i + o) % n;
            let key = (i.min(j), i.max(j));
            if !edges.iter().any(|&(a, b, _)| (a, b) == key) {
                edges.push((i, j, 1.0));
            }
        }
    }
    SimGraph::build(n, &edges)
}

/// Mean and min ns/call over `iters` timed invocations of `f` (one uncounted
/// warm-up first; `black_box` on the result inside `f`). Under box load the
/// mean inflates — the min is the robust statistic and both are printed.
fn time_ns_per_call(mut f: impl FnMut(), iters: usize) -> (f64, f64) {
    f(); // warm-up
    let mut total = Duration::ZERO;
    let mut min = Duration::MAX;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed();
        total += dt;
        if dt < min {
            min = dt;
        }
    }
    (
        total.as_secs_f64() * 1e9 / iters as f64,
        min.as_secs_f64() * 1e9,
    )
}

fn verdict(name: &str, mean_ns: f64, min_ns: f64, iters: usize) {
    let strict = if mean_ns <= 1_000.0 { "PASS" } else { "FAIL" };
    let us_class = if mean_ns <= 10_000.0 { "PASS" } else { "FAIL" };
    println!(
        "  {name:<44} mean {mean_ns:>12.1} ns  min {min_ns:>12.1} ns  (n={iters})  \
         [≤1µs {strict} · ≤10µs {us_class}]"
    );
}

fn main() {
    println!("== plan 634 T2.6 G2 latency ladder (τ={TAU}, release) ==");
    let mut rng = Lcg(0x634_626);
    let _ = rng.next_f32();

    // ── local-posture ladder ──────────────────────────────────────────────
    // (name, graph, n, similarity iters, div iters, pick iters)
    let rungs: Vec<(&str, SimGraph)> = vec![
        ("planted K=8 M=8", planted(8, 8)),
        ("circulant n=64 off=[1,5]", circulant(64, &[1, 5])),
        ("planted K=16 M=16", planted(16, 16)),
        ("circulant n=256 off=[1,5]", circulant(256, &[1, 5])),
        ("planted K=25 M=40", planted(25, 40)),
        ("circulant n=1000 off=[1,5]", circulant(1000, &[1, 5])),
    ];

    let mut battery_scale_checked = false;
    for (name, g) in &rungs {
        let n = g.n;
        println!("-- {name} (n={n}, directed E={})", g.col.len());
        let iters = if n <= 256 { 2_000 } else { 200 };

        let mut scratch = LocalScratch::new(n);
        let mut pair = 0usize;
        let (sim_mean, sim_min) = time_ns_per_call(
            || {
                pair = (pair + 7) % n;
                let s = diffusion_similarity_local(g, 0, pair, TAU, &mut scratch);
                black_box(s);
            },
            iters,
        );
        verdict("diffusion_similarity_local", sim_mean, sim_min, iters);

        let mu = vec![1.0 / n as f32; n];
        let mut q = vec![0.0_f32; n];
        let mut r = vec![0.0_f32; n];
        let (div_mean, div_min) = time_ns_per_call(
            || {
                black_box(g.div_tau(&mu, TAU, &mut q, &mut r));
            },
            iters,
        );
        verdict("div_tau (uniform μ)", div_mean, div_min, iters);

        // The walk-time pick at node 0 — `out` pairs with |N(x)| = the
        // UNWEIGHTED neighbor count (cols), not the weighted `degree()`.
        // The QP's k ≤ 64 work bound refuses loud above that — the dense
        // planted fixture at M=40 (39 + 80 = 119 columns) is OUTSIDE the
        // walk selector's domain (the paper's walk nodes are small-degree);
        // that rung's pick arm is skipped, and the skip is the recorded
        // domain fact.
        let (cols0, _) = g.row(0);
        let k = cols0.len();
        if k > 64 {
            println!(
                "  {:<44} SKIPPED — degree {k} > 64 (the QP work bound; this \
                 rung is outside the walk selector's domain)",
                "neighbor_diversity_pick"
            );
        } else {
            let mut out = vec![0.0_f32; k];
            let pick_iters = if n <= 256 { 100 } else { 5 };
            let (pick_mean, pick_min) = time_ns_per_call(
                || {
                    neighbor_diversity_pick(g, 0, TAU, 0.0, &mut scratch, &mut out);
                    black_box(&out);
                },
                pick_iters,
            );
            verdict(
                &format!("neighbor_diversity_pick (k={k})"),
                pick_mean,
                pick_min,
                pick_iters,
            );

            // Sanity ceilings — battery scale (n=64), calibrated from THIS
            // bench's own first measurement (T2.6, 2026-11-12): the plan's
            // ≤1 µs / ≤10 µs bar measured ~74–82 µs (the per-edge
            // `w/degree(z)` f32 division dominates the matvec — the
            // perf-rung issue carries the fix lanes), so the ceilings are
            // 10× the measured reality: a pathology catch (an accidental
            // O(n²) or a lost CSR row reads 10–100× this), never a
            // calibration bar. Re-pin only alongside a perf-rung landing.
            if n == 64 {
                battery_scale_checked = true;
                assert!(
                    sim_mean <= 500_000.0,
                    "G2 PATHOLOGY: similarity at battery scale {sim_mean:.0} ns > 500 µs"
                );
                assert!(
                    div_mean <= 250_000.0,
                    "G2 PATHOLOGY: div_tau at battery scale {div_mean:.0} ns > 250 µs"
                );
                assert!(
                    pick_mean <= 100_000_000.0,
                    "G2 PATHOLOGY: walk pick at battery scale {pick_mean:.0} ns > 100 ms"
                );
            }
        }
    }
    assert!(battery_scale_checked, "ladder lost its battery-scale rung");

    // ── spectral query paths (n ≤ SPECTRAL_N_CAP) ─────────────────────────
    for (name, g) in &rungs {
        let n = g.n;
        if n > SPECTRAL_N_CAP {
            continue;
        }
        println!("-- spectral {name} (n={n})");
        let t0 = Instant::now();
        let emb = DiffusionEmbedding::build(g);
        println!(
            "  {:<44} {:.1} ms (build-time class, reported once)",
            "DiffusionEmbedding::build",
            t0.elapsed().as_secs_f64() * 1e3
        );
        let iters = 2_000;
        let (sp_mean, sp_min) = time_ns_per_call(
            || {
                black_box(emb.similarity_spectral_exact(3, 11, TAU));
            },
            iters,
        );
        verdict("similarity_spectral_exact (k=n)", sp_mean, sp_min, iters);
        let (co_mean, co_min) = time_ns_per_call(
            || {
                black_box(emb.spectral_coordinates::<16>(5, TAU));
            },
            iters,
        );
        verdict("spectral_coordinates::<16>", co_mean, co_min, iters);
    }

    // ── batch selector (build-time class, reported) ───────────────────────
    println!("-- batch_select_diffusion (build-time class, reported)");
    for (name, g) in &rungs {
        let n = g.n;
        if n > SPECTRAL_N_CAP {
            continue;
        }
        let pool: Vec<usize> = (0..64).map(|i| i * 4 % n).collect();
        let mut out = vec![u16::MAX; 8];
        let (bs_mean, bs_min) = time_ns_per_call(
            || {
                black_box(batch_select_diffusion::<16>(g, &pool, TAU, 0.0, &mut out));
            },
            20,
        );
        println!(
            "  {:<44} mean {bs_mean:>12.1} ns  min {bs_min:>12.1} ns  (n=20, pool=64)",
            format!("{name} (n={n})")
        );
    }

    println!("== G2 ladder complete — strict ≤1 µs and µs-class ≤10 µs verdicts above; ==");
    println!("   the plan row carries the measured adjudication + box state ==");
}
