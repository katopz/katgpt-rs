//! Plan 277 Issue 026 — Unified Surprise Bus Super-GOAT Sweep.
//!
//! Validates whether the single paper-default α-pair (0.3, 0.03) that drives
//! all four TemporalDerivativeKernel consumers is a **real universal property**
//! (Super-GOAT) or merely coincidental (GOAT only).
//!
//! For each consumer, we sweep α_fast × α_slow across the grid:
//!   α_fast ∈ {0.1, 0.2, 0.3, 0.5, 0.8}
//!   α_slow ∈ {0.01, 0.03, 0.05, 0.1}
//! skipping invalid combos (α_fast ≤ α_slow). We then check whether the
//! paper-default (0.3, 0.03) cell is within 10% of the best metric for that
//! consumer.
//!
//! ## The four consumers
//!
//! | Fusion | Consumer | N | α-setter |
//! |--------|----------|---|----------|
//! | F1 | HLA companion (sense) | 8 | ReconstructionConfig |
//! | F2 | δ-Mem write gate | 8 | enable_surprise_gate_with_alphas |
//! | F3 | Collapse detector | 1 | with_temporal_deriv_alphas |
//! | F4 | Derivative curiosity | 64 | DerivativeCuriosity::with_alphas |
//!
//! Run with:
//! ```bash
//! cargo test --features 'temporal_deriv sense_composition delta_mem \
//!   collapse_aware_thinking cgsp' --test bench_277_unified_surprise_bus \
//!   -- --nocapture --test-threads=1
//! ```

#![cfg(all(
    feature = "temporal_deriv",
    feature = "sense_composition",
    feature = "delta_mem",
    feature = "collapse_aware_thinking",
    feature = "cgsp"
))]

use katgpt_core::sense::{ReconstructionConfig, ReconstructionState};
use katgpt_core::traits::CollapseDetector;
use katgpt_core::ThinkingBudget;
use katgpt_core::cgsp::{
    CgspConfig, DerivativeCuriosity, Direction, EntropyCollapse, HintDeltaBandit,
    Priority, ScratchBuffers, Target, entropy_nats,
};
use katgpt_rs::pruners::{DeltaMemoryConfig, DeltaMemoryState, S2FCollapseDetector};

// ── α-grid ────────────────────────────────────────────────────────────────
const ALPHA_FAST: [f32; 5] = [0.1, 0.2, 0.3, 0.5, 0.8];
const ALPHA_SLOW: [f32; 4] = [0.01, 0.03, 0.05, 0.1];

/// Paper-default α-pair (the "unified surprise bus" candidate).
const PAPER_AF: f32 = 0.3;
const PAPER_AS: f32 = 0.03;

/// Pareto tolerance: the paper-default must be within this fraction of the
/// best observed metric to count as "in the Pareto region".
const WITHIN_FRAC: f32 = 0.10;

/// Sentinel for an invalid (skipped) α-combo.
const INVALID: f32 = f32::NAN;

#[inline]
fn valid_combo(af: f32, as_: f32) -> bool {
    af > as_ && as_ > 0.0 && af <= 1.0
}

// ───────────────────────────────────────────────────────────────────────────
// F1 — HLA Surprise Companion (N=8, katgpt-core/sense)
// ───────────────────────────────────────────────────────────────────────────
//
// 1000-tick emotional-event trace. HLA starts at [0;8]. Events at tick 200
// (dim 0, +0.6), 500 (dim 1, +0.4), 800 (dim 2, +0.5). Between events HLA is
// constant. Metric: recall (events detected) and FPR (non-event peak ticks).
// We report a single score = recall · (1 − FPR) for the Pareto comparison.

const F1_TRACE_LEN: usize = 1000;
const F1_EVENTS: [(usize, [f32; 8]); 3] = [
    (200, [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (500, [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (800, [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
];
const F1_WINDOW: usize = 20;

fn f1_metric(af: f32, as_: f32) -> f32 {
    let mut config = ReconstructionConfig::default();
    config.temporal_deriv_alpha_fast = af;
    config.temporal_deriv_alpha_slow = as_;
    let mut state = ReconstructionState::with_config([0.0; 8], config);

    let mut surprise = [0.0f32; F1_TRACE_LEN];
    for t in 0..F1_TRACE_LEN {
        // Inject scripted event delta at the event tick.
        for &(tick, delta) in &F1_EVENTS {
            if t == tick {
                state.inject_hla_delta(delta);
            }
        }
        // No-op evolve (zero evidence) — kernel still observes the HLA.
        state.evolve_hla();
        surprise[t] = state.surprise_norm();
    }

    // Peak threshold = 0.5 × max surprise (per the sweep spec).
    let max_s = surprise.iter().cloned().fold(0.0f32, f32::max);
    let threshold = 0.5 * max_s;

    // Local-maximum peaks above threshold.
    let mut peaks: Vec<usize> = Vec::new();
    for t in 0..F1_TRACE_LEN {
        if surprise[t] <= threshold {
            continue;
        }
        let prev = if t == 0 { f32::MIN } else { surprise[t - 1] };
        let next = if t + 1 == F1_TRACE_LEN {
            f32::MIN
        } else {
            surprise[t + 1]
        };
        if surprise[t] >= prev && surprise[t] >= next {
            peaks.push(t);
        }
    }

    // Recall: fraction of events whose window contains ≥1 peak.
    let mut events_hit = 0usize;
    for &(tick, _) in &F1_EVENTS {
        let lo = tick.saturating_sub(F1_WINDOW);
        let hi = (tick + F1_WINDOW).min(F1_TRACE_LEN - 1);
        if (lo..=hi).any(|t| peaks.contains(&t)) {
            events_hit += 1;
        }
    }
    let recall = events_hit as f32 / F1_EVENTS.len() as f32;

    // FPR: out-of-window peak ticks / trace length.
    let fp = peaks
        .iter()
        .filter(|&&t| !F1_EVENTS.iter().any(|&(e, _)| t.abs_diff(e) <= F1_WINDOW))
        .count();
    let fpr = fp as f32 / F1_TRACE_LEN as f32;

    // Combined score: recall penalized by false-positive rate.
    recall * (1.0 - fpr)
}

// ───────────────────────────────────────────────────────────────────────────
// F2 — δ-Mem Temporal Write Gate (N=8, root crate)
// ───────────────────────────────────────────────────────────────────────────
//
// 1000-write synthetic stream (block-structured): 5 boring blocks (identical
// centroid_bg key/value, 200 each) interleaved with 5 novel blocks (distinct
// one-hot centroid, 0 each... actually 1000 total). The surprise gate
// suppresses repetitive background writes. Metric: suppression %.

const F2_RANK: usize = 8;

#[inline]
fn l2_normalize(v: &mut [f32; F2_RANK]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

fn f2_build_stream() -> (Vec<([f32; F2_RANK], [f32; F2_RANK])>, usize) {
    // Boring centroid: first 4 dims active.
    let mut centroid_bg = [0.0f32; F2_RANK];
    for i in 0..4 {
        centroid_bg[i] = 1.0;
    }
    l2_normalize(&mut centroid_bg);

    // Novel centroids: one-hot in dims 4..8 (near-orthogonal to bg and each other).
    let mut novel_centroids = [[0.0f32; F2_RANK]; 4];
    for (idx, c) in novel_centroids.iter_mut().enumerate() {
        c[4 + idx] = 1.0;
    }

    let mut bg_val = [0.0f32; F2_RANK];
    bg_val[0] = 1.0;
    l2_normalize(&mut bg_val);

    // Block structure: alternating boring (200) and novel (50) blocks.
    const BORING_BLOCK: usize = 200;
    const NOVEL_BLOCK: usize = 50;

    let mut stream: Vec<([f32; F2_RANK], [f32; F2_RANK])> = Vec::new();
    for pair in 0..5 {
        for _ in 0..BORING_BLOCK {
            stream.push((centroid_bg, bg_val));
        }
        let nc = novel_centroids[pair % 4];
        let mut nv_val = [0.0f32; F2_RANK];
        nv_val[pair % F2_RANK] = 1.0;
        l2_normalize(&mut nv_val);
        for _ in 0..NOVEL_BLOCK {
            stream.push((nc, nv_val));
        }
    }
    let total = stream.len();
    (stream, total)
}

fn f2_metric(af: f32, as_: f32, stream: &[([f32; F2_RANK], [f32; F2_RANK])]) -> f32 {
    let mut gated = DeltaMemoryState::new(DeltaMemoryConfig::default());
    gated.enable_surprise_gate_with_alphas(af, as_);
    gated.set_theta_surprise(0.10);
    for (k, v) in stream {
        gated.write(k, v);
    }
    gated.write_suppression_rate()
}

// ───────────────────────────────────────────────────────────────────────────
// F3 — Collapse Detector Fusion (N=1, root crate)
// ───────────────────────────────────────────────────────────────────────────
//
// 24 gradual-convergence entropy traces: e(t) = e_star + (e0 - e_star)·exp(-t/τ).
// The derivative signal flags "coasting" when |d(entropy)/dt| < τ_deriv for a
// sustained period. Metric: FN reduction % = traces_caught / 24.

const F3_N_TRACES: usize = 24;
const F3_TRACE_LEN: usize = 200;
const F3_THRESHOLD: u32 = 8;
const F3_TAU_DERIV: f32 = 0.01;

fn f3_gradual_trace(e0: f32, e_star: f32, tau: f32, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for t in 0..len {
        let decay = (-(t as f32) / tau).exp();
        out.push(e_star + (e0 - e_star) * decay);
    }
    out
}

fn f3_build_traces() -> Vec<Vec<f32>> {
    let mut traces = Vec::with_capacity(F3_N_TRACES);
    for i in 0..F3_N_TRACES {
        let e_star = 0.30 + 0.40 * (i as f32) / (F3_N_TRACES as f32 - 1.0);
        let tau = 4.0 * (1.0 + ((i % 3) as f32));
        let e0 = 1.2 + 0.3 * ((i % 4) as f32);
        traces.push(f3_gradual_trace(e0, e_star, tau, F3_TRACE_LEN));
    }
    traces
}

fn f3_metric(af: f32, as_: f32, traces: &[Vec<f32>]) -> f32 {
    // Hesitation tokens = {1,2,3}; we emit tokens ≥ 100 so the ring never matches.
    let budget = ThinkingBudget {
        max_tokens: 4096,
        collapse_threshold: F3_THRESHOLD,
        efficiency_gamma: 0.5,
    };
    let mut caught = 0usize;
    for trace in traces {
        let mut det = S2FCollapseDetector::new(vec![1, 2, 3], &budget)
            .with_temporal_deriv_alphas(af, as_)
            .with_tau_deriv(F3_TAU_DERIV);
        let mut fired = false;
        for (t, &entropy) in trace.iter().enumerate() {
            det.observe_entropy(entropy);
            let hard = det.check_collapse(100 + t as u32, t);
            let soft = det.derivative_collapse_detected();
            if hard || soft {
                fired = true;
                break;
            }
        }
        if fired {
            caught += 1;
        }
    }
    // Hesitation-only catches 0/24 (no hesitation tokens emitted).
    // FN reduction = caught / total.
    caught as f32 / F3_N_TRACES as f32
}

// ───────────────────────────────────────────────────────────────────────────
// F4 — Derivative Curiosity (N=64, katgpt-core/cgsp)
// ───────────────────────────────────────────────────────────────────────────
//
// Force one-hot collapse onto arm 3, count cycles to recover (entropy ≥ τ_low).
// Metric: recovery cycles (lower = better). The CGSP baseline is 1 cycle, so
// the gate target is recovery ≤ 2×.

const F4_TAU_LOW: f32 = 0.30;
const F4_MAX_CYCLES: usize = 50;

/// Minimal bandit backed by a Vec<f32>, matching the cgsp test helper contract.
struct VecBandit {
    prios: Vec<f32>,
}
impl VecBandit {
    fn uniform(n: usize) -> Self {
        Self {
            prios: vec![1.0 / n as f32; n],
        }
    }
}
impl HintDeltaBandit for VecBandit {
    fn absorb(&mut self, arm: usize, reward: f32) {
        if let Some(p) = self.prios.get_mut(arm) {
            *p += reward.max(0.0);
        }
    }
    fn priority(&self, arm: usize) -> Priority {
        self.prios.get(arm).copied().unwrap_or(0.0)
    }
    fn priorities(&self) -> &[Priority] {
        &self.prios
    }
    fn priorities_mut(&mut self) -> &mut [Priority] {
        &mut self.prios
    }
}

fn f4_unit_direction(dim: usize, axis: usize) -> Direction {
    let mut coords = vec![0.0f32; dim];
    coords[axis.min(dim.saturating_sub(1))] = 1.0;
    Direction { coords }
}

fn f4_metric(af: f32, as_: f32) -> usize {
    let pool: Vec<Direction> = (0..8).map(|i| f4_unit_direction(8, i % 8)).collect();
    let mut dc: DerivativeCuriosity<64> =
        DerivativeCuriosity::new(pool.clone(), 5).with_alphas(af, as_);
    let mut bandit = VecBandit::uniform(8);
    let mut collapse = EntropyCollapse::default();
    let config = CgspConfig::default();
    let target = Target::new(pool[0].clone());
    let mut scratch = ScratchBuffers::new(8, 8);

    // Force one-hot collapse onto arm 3.
    for (i, p) in bandit.priorities_mut().iter_mut().enumerate() {
        *p = if i == 3 { 1.0 } else { 0.0 };
    }

    for cycle in 1..=F4_MAX_CYCLES {
        let _ = dc.cycle_curiosity(&target, &mut bandit, &mut scratch, &mut collapse, &config);
        let h = entropy_nats(bandit.priorities());
        if h >= F4_TAU_LOW {
            return cycle;
        }
    }
    F4_MAX_CYCLES
}

// ───────────────────────────────────────────────────────────────────────────
// Grid printing + Pareto analysis
// ───────────────────────────────────────────────────────────────────────────

/// Print a 5×4 metric grid (α_fast rows × α_slow cols), marking the
/// paper-default (0.3, 0.03) cell with a trailing ` ◄ paper`.
fn print_grid_f32(title: &str, unit: &str, metric_fn: impl Fn(f32, f32) -> f32) {
    println!("\n┌─ {title} ─┐  (higher = better)  unit: {unit}");
    print!("│ α_fast＼α_slow │");
    for &as_ in &ALPHA_SLOW {
        print!("  {:>7.4} ", as_);
    }
    println!("│");
    for &af in &ALPHA_FAST {
        print!("│ {:>11.3}   │", af);
        for &as_ in &ALPHA_SLOW {
            if !valid_combo(af, as_) {
                print!("    ---   ");
            } else {
                let m = metric_fn(af, as_);
                let marker = if (af - PAPER_AF).abs() < 1e-6 && (as_ - PAPER_AS).abs() < 1e-6 {
                    " ◄"
                } else {
                    "  "
                };
                print!(" {:>7.4}{}", m, marker);
            }
        }
        println!("│");
    }
    println!("└────────────────┘");
}

fn print_grid_usize(title: &str, unit: &str, metric_fn: impl Fn(f32, f32) -> usize) {
    println!("\n┌─ {title} ─┐  (lower = better)  unit: {unit}");
    print!("│ α_fast＼α_slow │");
    for &as_ in &ALPHA_SLOW {
        print!("  {:>7.4} ", as_);
    }
    println!("│");
    for &af in &ALPHA_FAST {
        print!("│ {:>11.3}   │", af);
        for &as_ in &ALPHA_SLOW {
            if !valid_combo(af, as_) {
                print!("    ---   ");
            } else {
                let m = metric_fn(af, as_);
                let marker = if (af - PAPER_AF).abs() < 1e-6 && (as_ - PAPER_AS).abs() < 1e-6 {
                    " ◄"
                } else {
                    "  "
                };
                print!(" {:>7}{}", m, marker);
            }
        }
        println!("│");
    }
    println!("└────────────────┘");
}

/// For "higher is better" metrics: is the paper-default within `frac` of the
/// best valid metric? Returns (paper_metric, best_metric, within).
fn within_higher(paper: f32, values: &[f32]) -> (f32, f32, bool) {
    let best = values.iter().cloned().fold(0.0f32, f32::max);
    let within = paper >= best * (1.0 - WITHIN_FRAC);
    (paper, best, within)
}

/// For "lower is better" metrics: is the paper-default within `frac` of the
/// best valid metric? Returns (paper_metric, best_metric, within).
fn within_lower(paper: f32, values: &[f32]) -> (f32, f32, bool) {
    let best = values
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let within = paper <= best * (1.0 + WITHIN_FRAC);
    (paper, best, within)
}

// ───────────────────────────────────────────────────────────────────────────
// The sweep test
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn unified_surprise_bus_sweep() {
    println!("\n══════════════════════════════════════════════════════════════════");
    println!("  Plan 277 Issue 026 — Unified Surprise Bus Super-GOAT Sweep");
    println!("  Paper-default α-pair: ({}, {})  |  Pareto tolerance: ±{:.0}%",
             PAPER_AF, PAPER_AS, WITHIN_FRAC * 100.0);
    println!("══════════════════════════════════════════════════════════════════");

    // ── F1: HLA companion ──────────────────────────────────────────────────
    let mut f1_grid = [[INVALID; 4]; 5];
    for (i, &af) in ALPHA_FAST.iter().enumerate() {
        for (j, &as_) in ALPHA_SLOW.iter().enumerate() {
            if valid_combo(af, as_) {
                f1_grid[i][j] = f1_metric(af, as_);
            }
        }
    }
    let f1_flat: Vec<f32> = f1_grid.iter().flatten().filter(|v| !v.is_nan()).copied().collect();
    let f1_paper = f1_metric(PAPER_AF, PAPER_AS);
    let (f1_p, f1_best, f1_within) = within_higher(f1_paper, &f1_flat);

    println!("\n── F1: HLA Surprise Companion (N=8) ──");
    print_grid_f32("recall·(1−FPR)", "score", |af, as_| f1_metric(af, as_));
    println!("  paper ({},{}) = {:.4}  |  best = {:.4}  |  within ±{:.0}%: {}",
             PAPER_AF, PAPER_AS, f1_p, f1_best, WITHIN_FRAC * 100.0,
             if f1_within { "YES ✓" } else { "NO ✗" });

    // ── F2: δ-Mem gate ─────────────────────────────────────────────────────
    let (stream, n_writes) = f2_build_stream();
    println!("\n── F2: δ-Mem Write Gate (N=8) ──  stream: {n_writes} writes");
    let mut f2_grid = [[INVALID; 4]; 5];
    for (i, &af) in ALPHA_FAST.iter().enumerate() {
        for (j, &as_) in ALPHA_SLOW.iter().enumerate() {
            if valid_combo(af, as_) {
                f2_grid[i][j] = f2_metric(af, as_, &stream);
            }
        }
    }
    let f2_flat: Vec<f32> = f2_grid.iter().flatten().filter(|v| !v.is_nan()).copied().collect();
    let f2_paper = f2_metric(PAPER_AF, PAPER_AS, &stream);
    let (f2_p, f2_best, f2_within) = within_higher(f2_paper, &f2_flat);

    print_grid_f32("suppression %", "frac", |af, as_| f2_metric(af, as_, &stream));
    println!("  paper ({},{}) = {:.4}  |  best = {:.4}  |  within ±{:.0}%: {}",
             PAPER_AF, PAPER_AS, f2_p, f2_best, WITHIN_FRAC * 100.0,
             if f2_within { "YES ✓" } else { "NO ✗" });

    // ── F3: Collapse detector ──────────────────────────────────────────────
    let traces = f3_build_traces();
    println!("\n── F3: Collapse Detector (N=1) ──  {F3_N_TRACES} gradual-convergence traces");
    let mut f3_grid = [[INVALID; 4]; 5];
    for (i, &af) in ALPHA_FAST.iter().enumerate() {
        for (j, &as_) in ALPHA_SLOW.iter().enumerate() {
            if valid_combo(af, as_) {
                f3_grid[i][j] = f3_metric(af, as_, &traces);
            }
        }
    }
    let f3_flat: Vec<f32> = f3_grid.iter().flatten().filter(|v| !v.is_nan()).copied().collect();
    let f3_paper = f3_metric(PAPER_AF, PAPER_AS, &traces);
    let (f3_p, f3_best, f3_within) = within_higher(f3_paper, &f3_flat);

    print_grid_f32("FN reduction", "frac", |af, as_| f3_metric(af, as_, &traces));
    println!("  paper ({},{}) = {:.4}  |  best = {:.4}  |  within ±{:.0}%: {}",
             PAPER_AF, PAPER_AS, f3_p, f3_best, WITHIN_FRAC * 100.0,
             if f3_within { "YES ✓" } else { "NO ✗" });

    // ── F4: Derivative curiosity ───────────────────────────────────────────
    println!("\n── F4: Derivative Curiosity (N=64) ──  recovery from one-hot collapse");
    let mut f4_grid = [[0usize; 4]; 5];
    let mut f4_valid: Vec<f32> = Vec::new();
    for (i, &af) in ALPHA_FAST.iter().enumerate() {
        for (j, &as_) in ALPHA_SLOW.iter().enumerate() {
            if valid_combo(af, as_) {
                let c = f4_metric(af, as_);
                f4_grid[i][j] = c;
                f4_valid.push(c as f32);
            }
        }
    }
    let f4_paper = f4_metric(PAPER_AF, PAPER_AS);
    let (_f4_p, f4_best, f4_within) = within_lower(f4_paper as f32, &f4_valid);

    print_grid_usize("recovery cycles", "cycles", |af, as_| f4_metric(af, as_));
    println!("  paper ({},{}) = {}  |  best = {}  |  within ±{:.0}%: {}",
             PAPER_AF, PAPER_AS, f4_paper, f4_best as usize, WITHIN_FRAC * 100.0,
             if f4_within { "YES ✓" } else { "NO ✗"});

    // ── Pareto verdict ─────────────────────────────────────────────────────
    let n_pass = [f1_within, f2_within, f3_within, f4_within]
        .iter()
        .filter(|&&b| b)
        .count();
    let super_goat = n_pass == 4;

    println!("\n══════════════════════════════════════════════════════════════════");
    println!("  PARETO ANALYSIS — is (0.3, 0.03) within ±{:.0}% of best for ALL 4?",
             WITHIN_FRAC * 100.0);
    println!("  F1 HLA companion   : {}", if f1_within { "YES ✓" } else { "NO ✗" });
    println!("  F2 δ-Mem gate      : {}", if f2_within { "YES ✓" } else { "NO ✗" });
    println!("  F3 Collapse detect : {}", if f3_within { "YES ✓" } else { "NO ✗" });
    println!("  F4 Deriv curiosity : {}", if f4_within { "YES ✓" } else { "NO ✗"});
    println!("  ─────────────────────────────────────────────────────────");
    println!("  {}/4 consumers have the paper-default in their Pareto region.", n_pass);
    if super_goat {
        println!("  VERDICT: SUPER-GOAT ✓ — unified α-pair (0.3, 0.03) is universal.");
    } else {
        println!("  VERDICT: GOAT only — unified α-pair is NOT universal for all 4.");
        println!("           Per-consumer tuning recommended for the {} outlier(s).", 4 - n_pass);
    }
    println!("══════════════════════════════════════════════════════════════════\n");
}
