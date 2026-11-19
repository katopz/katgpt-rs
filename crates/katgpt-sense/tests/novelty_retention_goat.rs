//! Plan 639 Phase 2 — the GOAT bench (CPU fixture lane, pre-registered).
//!
//! Run (the verdict is PRINTED; `--nocapture` is part of the command):
//!
//! ```text
//! cargo test --release -p katgpt-sense --features novelty_retention \
//!     --test novelty_retention_goat -- --nocapture --test-threads=1
//! ```
//!
//! ## Metric definitions (pinned BEFORE the first run — commit order is the proof)
//!
//! Both metrics are over the 8 belief slots against the perfect-tracker target
//! `target_drive(h, η)` of each slot's kind (`KIND_MAP`). Every run starts
//! WARM: belief = target at tick 0, so no warm-up transient enters a metric.
//!
//! - **adaptation** (lower = better) — case (b) `Steps`. For each step tick
//!   `s ∈ STEP_TICKS` and slot `i`: the first-passage tick count `k ≥ 0` at
//!   which the belief has covered ≥ 90% of the way from the pre-step target to
//!   the post-step target (sign-aware). Never reached inside the 100-tick
//!   inter-step window → `k = 100` (the cap is reported as `capped`). Mean over
//!   3 steps × 8 slots.
//! - **noise_rmse** (lower = better) — RMSE of belief vs target over EVERY tick
//!   of case (a) `Steady` ∪ case (c) `Spikes` (the hidden state never moves in
//!   either, so all error is noise / transient leakage).
//! - Reported, NOT judged: `mixed_rmse` (case (d), all ticks).
//!
//! ## Verdict arithmetic (Plan 639 T2.2, verbatim conditions)
//!
//! - **(i)** ∃ α on the PINNED 9-point curve with, for EVERY seed,
//!   `cell.adaptation ≤ 0.7·p.adaptation ∧ cell.noise_rmse ≤ 1.1·p.noise_rmse`
//!   (the SAME α for both margins and all 3 seeds).
//! - **(ii)** no fixed-α point dominates the seed-mean cell (`≤` on both axes)
//!   — judged against the DENSE curve (α = 0.50..0.99 step 0.01), a strict
//!   superset of the pinned curve, so the sparse-grid escape the plan warns
//!   about is closed up front rather than on demand.
//! - **(iii)** distance below the piecewise-linear dense curve (in noise_rmse
//!   at the cell's adaptation) `> 0`. A cell whose adaptation lies outside the
//!   dense curve's range reports `OUT-OF-RANGE` and does NOT pass (iii).
//!
//! Plus G4 (zero-alloc armed forward, counted by this binary's allocator) and
//! T2.4 (interleaved median-of-ratios kernel wall, gated vs incumbent; bar
//! ≤ 1.15×, release only — the debug ratio is printed but NOT judged).
//!
//! α_base stays the shipped default `[0.9; 6]` for every gated cell — the grid
//! varies only the four pre-registered axes (κ, μ, β, g_max).

mod novelty_fixture;

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use katgpt_sense::novelty_retention::NoveltyRetentionConfig;
use katgpt_sense::reconstruction::{ReconstructionConfig, ReconstructionState, TripleEvidence};
use novelty_fixture::{Case, STEP_TICKS, Trace, target_drive, trace};

// ── G4 instrument: a counting allocator local to this binary ────────────────

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

// ── pinned constants ────────────────────────────────────────────────────────

const SEEDS: [u64; 3] = [1, 2, 3];
const PINNED_ALPHAS: [f32; 9] = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97];
const KAPPAS: [f32; 3] = [0.2, 0.4, 0.8];
const MUS: [f32; 3] = [0.0, 0.2, 0.4];
const BETAS: [f32; 2] = [4.0, 8.0];
const GMAXES: [f32; 2] = [0.99, 0.995];
const ADAPT_CAP: usize = 100;
const ADAPT_FRAC: f32 = 0.9;

fn dense_alphas() -> Vec<f32> {
    (50..=99).map(|k| k as f32 / 100.0).collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct Point {
    adaptation: f64,
    noise_rmse: f64,
    mixed_rmse: f64,
    capped: usize,
}

#[derive(Clone, Copy)]
enum Arm {
    Fixed(f32),
    Gated(NoveltyRetentionConfig),
}

fn state_for(arm: Arm, warm: [f32; 8]) -> ReconstructionState {
    let mut cfg = ReconstructionConfig::default();
    match arm {
        Arm::Fixed(a) => {
            cfg.belief_retention = [a; 6];
            ReconstructionState::with_config(warm, cfg)
        }
        Arm::Gated(n) => {
            let mut st = ReconstructionState::with_config(warm, cfg);
            assert!(
                st.arm_novelty_retention(n),
                "arm refused (NOVELTY_RETENTION=0 set, or {n:?} breaks a guard) — the bench is meaningless demoted"
            );
            st
        }
    }
}

fn targets(tr: &Trace, eta: f32) -> Vec<[f32; 8]> {
    tr.hidden
        .iter()
        .map(|h| {
            let mut t = [0.0f32; 8];
            for (i, &m) in TripleEvidence::KIND_MAP.iter().enumerate() {
                t[i] = target_drive(h[m], eta);
            }
            t
        })
        .collect()
}

/// Run one arm over one trace; returns the per-tick belief.
fn run(arm: Arm, tr: &Trace, tg: &[[f32; 8]]) -> Vec<[f32; 8]> {
    let mut st = state_for(arm, tg[0]);
    tr.observed
        .iter()
        .map(|x| {
            st.evolve_belief_additive(x);
            *st.belief()
        })
        .collect()
}

fn rmse(b: &[[f32; 8]], t: &[[f32; 8]]) -> (f64, usize) {
    let mut se = 0.0f64;
    for (bb, tt) in b.iter().zip(t) {
        for i in 0..8 {
            let d = (bb[i] - tt[i]) as f64;
            se += d * d;
        }
    }
    (se, b.len() * 8)
}

fn measure(arm: Arm, seed: u64) -> Point {
    let eta = ReconstructionConfig::default().additive_drive_scale;
    // adaptation — case (b)
    let tr = trace(Case::Steps, seed);
    let tg = targets(&tr, eta);
    let b = run(arm, &tr, &tg);
    let (mut sum, mut n, mut capped) = (0usize, 0usize, 0usize);
    for &s in &STEP_TICKS {
        for i in 0..8 {
            let from = tg[s - 1][i];
            let to = tg[s][i];
            let need = ADAPT_FRAC * (to - from);
            let mut k = ADAPT_CAP;
            for j in 0..ADAPT_CAP {
                let covered = b[s + j][i] - from;
                if (need > 0.0 && covered >= need) || (need < 0.0 && covered <= need) {
                    k = j;
                    break;
                }
            }
            capped += usize::from(k == ADAPT_CAP);
            sum += k;
            n += 1;
        }
    }
    // noise_rmse — cases (a) ∪ (c)
    let (mut se, mut cnt) = (0.0f64, 0usize);
    for case in [Case::Steady, Case::Spikes] {
        let tr = trace(case, seed);
        let tg = targets(&tr, eta);
        let (s, c) = rmse(&run(arm, &tr, &tg), &tg);
        se += s;
        cnt += c;
    }
    let tr = trace(Case::Mixed, seed);
    let tg = targets(&tr, eta);
    let (ms, mc) = rmse(&run(arm, &tr, &tg), &tg);
    Point {
        adaptation: sum as f64 / n as f64,
        noise_rmse: (se / cnt as f64).sqrt(),
        mixed_rmse: (ms / mc as f64).sqrt(),
        capped,
    }
}

fn mean_point(ps: &[Point]) -> Point {
    let k = ps.len() as f64;
    Point {
        adaptation: ps.iter().map(|p| p.adaptation).sum::<f64>() / k,
        noise_rmse: ps.iter().map(|p| p.noise_rmse).sum::<f64>() / k,
        mixed_rmse: ps.iter().map(|p| p.mixed_rmse).sum::<f64>() / k,
        capped: ps.iter().map(|p| p.capped).sum(),
    }
}

/// Noise RMSE of the piecewise-linear curve at `adapt`; `None` outside range.
fn curve_at(curve: &[(f32, Point)], adapt: f64) -> Option<f64> {
    let mut pts: Vec<(f64, f64)> = curve
        .iter()
        .map(|(_, p)| (p.adaptation, p.noise_rmse))
        .collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    for w in pts.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if adapt >= x0 && adapt <= x1 {
            if x1 == x0 {
                return Some(y0.min(y1));
            }
            return Some(y0 + (y1 - y0) * (adapt - x0) / (x1 - x0));
        }
    }
    None
}

struct Verdict {
    i_alpha: Option<f32>,
    ii: bool,
    dominator: Option<f32>,
    iii: Option<f64>,
}

fn judge(
    cell_seeds: &[Point],
    pinned_seeds: &[(f32, Vec<Point>)],
    dense_mean: &[(f32, Point)],
) -> Verdict {
    let i_alpha = pinned_seeds.iter().find_map(|(a, ps)| {
        let ok = cell_seeds
            .iter()
            .zip(ps)
            .all(|(c, p)| c.adaptation <= 0.7 * p.adaptation && c.noise_rmse <= 1.1 * p.noise_rmse);
        ok.then_some(*a)
    });
    let cm = mean_point(cell_seeds);
    let dominator = dense_mean
        .iter()
        .find(|(_, p)| p.adaptation <= cm.adaptation && p.noise_rmse <= cm.noise_rmse)
        .map(|(a, _)| *a);
    let iii = curve_at(dense_mean, cm.adaptation).map(|y| y - cm.noise_rmse);
    Verdict {
        i_alpha,
        ii: dominator.is_none(),
        dominator,
        iii,
    }
}

#[test]
fn t2_goat_sweep() {
    // T2.1 — the incumbent curve (pinned 9 points, per seed) + the dense curve.
    let pinned: Vec<(f32, Vec<Point>)> = PINNED_ALPHAS
        .iter()
        .map(|&a| {
            (
                a,
                SEEDS.iter().map(|&s| measure(Arm::Fixed(a), s)).collect(),
            )
        })
        .collect();
    let dense: Vec<(f32, Point)> = dense_alphas()
        .into_iter()
        .map(|a| {
            let ps: Vec<Point> = SEEDS.iter().map(|&s| measure(Arm::Fixed(a), s)).collect();
            (a, mean_point(&ps))
        })
        .collect();

    println!("\n## T2.1 incumbent curve (pinned α, per seed: adaptation / noise_rmse)");
    println!(
        "| α | adapt s1 | adapt s2 | adapt s3 | noise s1 | noise s2 | noise s3 | mixed (mean) | capped |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for (a, ps) in &pinned {
        let m = mean_point(ps);
        println!(
            "| {a:.2} | {:.2} | {:.2} | {:.2} | {:.4} | {:.4} | {:.4} | {:.4} | {} |",
            ps[0].adaptation,
            ps[1].adaptation,
            ps[2].adaptation,
            ps[0].noise_rmse,
            ps[1].noise_rmse,
            ps[2].noise_rmse,
            m.mixed_rmse,
            m.capped
        );
    }
    println!("\n## T2.1b dense curve (seed mean)");
    println!("| α | adaptation | noise_rmse | mixed_rmse | capped |");
    println!("|---|---|---|---|---|");
    for (a, p) in &dense {
        println!(
            "| {a:.2} | {:.2} | {:.4} | {:.4} | {} |",
            p.adaptation, p.noise_rmse, p.mixed_rmse, p.capped
        );
    }

    // T2.2 — the gated grid.
    println!("\n## T2.2 gated grid (α_base 0.9; seed mean; verdict per T2.2)");
    println!(
        "| κ | μ | β | g_max | adaptation | noise_rmse | mixed_rmse | capped | (i) α | (ii) | dominator | (iii) dist | PASS |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut passing: Vec<(NoveltyRetentionConfig, Point, f64)> = Vec::new();
    let mut cells = 0usize;
    for &kappa in &KAPPAS {
        for &mu in &MUS {
            for &beta in &BETAS {
                for &g_max in &GMAXES {
                    let ncfg = NoveltyRetentionConfig {
                        kappa,
                        mu,
                        beta,
                        g_max,
                        ..NoveltyRetentionConfig::default()
                    };
                    let ps: Vec<Point> = SEEDS
                        .iter()
                        .map(|&s| measure(Arm::Gated(ncfg), s))
                        .collect();
                    let m = mean_point(&ps);
                    let v = judge(&ps, &pinned, &dense);
                    let pass = v.i_alpha.is_some() && v.ii && v.iii.is_some_and(|d| d > 0.0);
                    cells += 1;
                    println!(
                        "| {kappa} | {mu} | {beta} | {g_max} | {:.2} | {:.4} | {:.4} | {} | {} | {} | {} | {} | {} |",
                        m.adaptation,
                        m.noise_rmse,
                        m.mixed_rmse,
                        m.capped,
                        v.i_alpha.map_or("—".into(), |a| format!("{a:.2}")),
                        if v.ii { "✓" } else { "✗" },
                        v.dominator.map_or("—".into(), |a| format!("{a:.2}")),
                        v.iii.map_or("OUT-OF-RANGE".into(), |d| format!("{d:+.4}")),
                        if pass { "**PASS**" } else { "fail" }
                    );
                    if pass {
                        passing.push((ncfg, m, v.iii.unwrap()));
                    }
                }
            }
        }
    }
    assert_eq!(cells, KAPPAS.len() * MUS.len() * BETAS.len() * GMAXES.len());
    passing.sort_by(|a, b| b.2.total_cmp(&a.2));
    println!(
        "\nVERDICT: {}/{} gated cells pass (i)+(ii)+(iii)",
        passing.len(),
        cells
    );
    match passing.first() {
        Some((c, p, d)) => println!(
            "BEST: κ={} μ={} β={} g_max={} adaptation={:.2} noise_rmse={:.4} dist_below_curve={:+.4}",
            c.kappa, c.mu, c.beta, c.g_max, p.adaptation, p.noise_rmse, d
        ),
        None => println!("BEST: none — T2.3 falsifier: DONE-NEGATIVE"),
    }
}

#[test]
fn g4_armed_forward_is_alloc_free() {
    let tr = trace(Case::Mixed, 1);
    let mut st = state_for(Arm::Gated(NoveltyRetentionConfig::default()), [0.0; 8]);
    st.evolve_belief_additive(&tr.observed[0]);
    let before = ALLOCS.load(Ordering::Relaxed);
    for x in &tr.observed {
        st.evolve_belief_additive(black_box(x));
    }
    let after = ALLOCS.load(Ordering::Relaxed);
    black_box(st.belief());
    // The counter is process-wide; this binary runs with --test-threads=1 for
    // the verdict, and any parallel test allocation can only INFLATE it (a
    // false red, never a false green).
    assert_eq!(
        after - before,
        0,
        "armed forward allocated {} times",
        after - before
    );
    println!(
        "G4: armed forward over {} ticks: 0 allocations",
        tr.observed.len()
    );
}

#[test]
fn t24_kernel_cost_ratio() {
    let tr = trace(Case::Mixed, 1);
    let obs = &tr.observed;
    let mut inc = state_for(Arm::Fixed(0.9), [0.0; 8]);
    let mut gat = state_for(Arm::Gated(NoveltyRetentionConfig::default()), [0.0; 8]);
    let (mut sink_a, mut sink_b) = (0.0f32, 0.0f32);
    let r = ab_timing::ab_median_ratio(
        51,
        20_000,
        4_000,
        |i| {
            inc.evolve_belief_additive(black_box(&obs[i % obs.len()]));
            sink_a += black_box(inc.belief()[i & 7]);
        },
        |i| {
            gat.evolve_belief_additive(black_box(&obs[i % obs.len()]));
            sink_b += black_box(gat.belief()[i & 7]);
        },
    );
    black_box((sink_a, sink_b));
    r.report("T2.4 gated/incumbent kernel wall");
    println!(
        "T2.4: incumbent {:.2} ns/tick, gated {:.2} ns/tick, median ratio {:.3}× (bar ≤ 1.15×, release only; debug={})",
        r.a_ns_per_iter(),
        r.b_ns_per_iter(),
        r.median,
        cfg!(debug_assertions)
    );
}
