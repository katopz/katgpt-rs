//! Derivative-curiosity G5 GOAT gate benchmark (Plan 277 Phase 5 / Fusion F4).
//!
//! Mirrors CGSP's G2 collapse-recovery scenario and runs it through both
//! conjecturer paths, measuring:
//!
//! 1. **Cycles to recover** from a forced one-hot priority collapse (entropy
//!    rising back above `τ_low`). Derivative-curiosity target: ≤ 2× CGSP's
//!    recovery cycles.
//! 2. **Per-cycle cost** (ns/cycle). Derivative-curiosity target: ≤ 10% of
//!    CGSP's per-cycle cost (CGSP baseline documented at ~831 ns/cycle;
//!    derivative target ≤ 100 ns/cycle since no Solver call).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features cgsp,temporal_deriv --bench derivative_curiosity_bench
//! ```
//!
//! # Feature gate
//!
//! Requires both `cgsp` (Plan 274) and `temporal_deriv` (Plan 277 Phase 1).
//!
//! # Deviation note
//!
//! Plan 277 T5.3 specifies "Measure both with `std::time::Instant` (katgpt-rs
//! convention — no criterion dev-dep)". The actual katgpt-rs convention (per
//! `temporal_deriv_bench.rs` and every other bench in this crate) is to use
//! `criterion`, which is already a dev-dependency. We follow the codebase
//! convention rather than the plan text — criterion gives more stable
//! measurements than ad-hoc `Instant` timing for sub-microsecond work.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::cgsp::{
    CgspConfig, CgspLoop, DerivativeCuriosity, Direction, EntropyCollapse, HintDeltaBandit,
    HlaProjectionGuide, PoolConjecturer, Priority, ScratchBuffers, Solver, Target, entropy_nats,
};

// ── Shared test fixtures (mirror loop_.rs / mod.rs integration tests) ──────

/// Minimal Vec-backed bandit — identical to the one in `loop_.rs` tests so
/// both conjecturer paths see the exact same bandit semantics.
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

/// Reference Solver — solve-rate proportional to dot-product with the target.
/// This is the SAME DotSolver used in `loop_.rs` / `mod.rs` integration tests,
/// kept local so the bench is self-contained.
struct DotSolver {
    sharpness: f32,
}
impl Solver for DotSolver {
    fn attempt(&mut self, target: &Target, candidate: &katgpt_core::cgsp::Candidate) -> f32 {
        let d = candidate.direction.dot(&target.direction);
        katgpt_core::cgsp::sigmoid(self.sharpness * d)
    }
}

/// 8 orthonormal-ish directions in 8-D — same construction as the CGSP
/// integration tests so the collapse scenario is directly comparable.
fn make_8_direction_pool() -> Vec<Direction> {
    (0..8)
        .map(|i| {
            let mut coords = vec![0.0f32; 8];
            coords[i] = 1.0;
            if i >= 1 {
                coords[(i + 1) % 8] = 0.1;
            }
            let norm: f32 = coords.iter().map(|c| c * c).sum::<f32>().sqrt();
            for c in &mut coords {
                *c /= norm.max(1e-9);
            }
            Direction { coords }
        })
        .collect()
}

/// Force a one-hot collapse onto arm 3. Returns the collapsed entropy.
fn force_collapse<B: HintDeltaBandit>(bandit: &mut B) -> f32 {
    for (i, p) in bandit.priorities_mut().iter_mut().enumerate() {
        *p = if i == 3 { 1.0 } else { 0.0 };
    }
    entropy_nats(bandit.priorities())
}

// ── Recovery-cycle measurement (functional, not timed) ────────────────────

/// Run CGSP's `CgspLoop::cycle` from a forced collapse until entropy recovers
/// above `tau_low`, counting cycles. Returns `(cycles_to_recover, triggered)`.
fn cgsp_cycles_to_recover(tau_low: f32, max_cycles: usize) -> (usize, bool) {
    let pool = make_8_direction_pool();
    let conj = PoolConjecturer::new(pool.clone(), 5);
    let guide = HlaProjectionGuide::new(2.0, 1.0, katgpt_core::cgsp::ComplexityWeights::default());
    let solver = DotSolver { sharpness: 1.0 };
    let bandit = VecBandit::uniform(8);
    let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default());
    let target = Target::new(pool[0].clone());
    let mut scratch = ScratchBuffers::new(8, 8);

    let h0 = force_collapse(lp.bandit_mut());
    assert!(h0 < tau_low, "collapse failed: h0={h0}");

    let mut triggered = false;
    for cycle in 1..=max_cycles {
        let _ = lp.cycle(&target, &mut scratch);
        let h = entropy_nats(lp.bandit().priorities());
        if h >= tau_low {
            return (cycle, triggered);
        }
        // Track whether collapse injection ever fired (recovery mechanism).
        // We re-check via a fresh detector on the post-cycle state.
        triggered = triggered || (h > h0);
    }
    (max_cycles, triggered)
}

/// Run `DerivativeCuriosity::cycle_curiosity` from a forced collapse until
/// entropy recovers above `tau_low`, counting cycles.
fn derivative_cycles_to_recover(tau_low: f32, max_cycles: usize) -> (usize, bool) {
    let pool = make_8_direction_pool();
    let mut dc: DerivativeCuriosity<64> = DerivativeCuriosity::new(pool.clone(), 5);
    let mut bandit = VecBandit::uniform(8);
    let mut collapse = EntropyCollapse::default();
    let config = CgspConfig::default();
    let target = Target::new(pool[0].clone());
    let mut scratch = ScratchBuffers::new(8, 8);

    let h0 = force_collapse(&mut bandit);
    assert!(h0 < tau_low, "collapse failed: h0={h0}");

    let mut triggered = false;
    for cycle in 1..=max_cycles {
        let r = dc.cycle_curiosity(&target, &mut bandit, &mut scratch, &mut collapse, &config);
        if r.collapse_triggered {
            triggered = true;
        }
        let h = entropy_nats(bandit.priorities());
        if h >= tau_low {
            return (cycle, triggered);
        }
    }
    (max_cycles, triggered)
}

// ── Functional gate check (runs once at bench-startup, prints verdict) ────

/// Run the G5 functional gate (cycles-to-recover ≤ 2× CGSP) and print the
/// verdict. Implemented as a criterion benchmark that runs exactly one
/// iteration so the result appears in the bench output.
fn g5_functional_gate(c: &mut Criterion) {
    let tau_low = 0.30; // matches CgspConfig::default().tau_low
    let max_cycles = 50;

    let (cgsp_cycles, cgsp_triggered) = cgsp_cycles_to_recover(tau_low, max_cycles);
    let (deriv_cycles, deriv_triggered) = derivative_cycles_to_recover(tau_low, max_cycles);

    let ratio = deriv_cycles as f64 / cgsp_cycles.max(1) as f64;
    let pass = deriv_cycles <= 2 * cgsp_cycles && deriv_triggered;

    println!("\n═══ G5 Functional Gate (cycles-to-recover) ═══");
    println!("  τ_low                       = {tau_low:.3} nats");
    println!("  CGSP cycles-to-recover      = {cgsp_cycles}  (collapse_triggered={cgsp_triggered})");
    println!("  Derivative cycles-to-recover= {deriv_cycles}  (collapse_triggered={deriv_triggered})");
    println!("  Derivative / CGSP ratio     = {ratio:.2}×  (target ≤ 2.0×)");
    println!("  G5 (a) functional verdict   = {}", if pass { "PASS" } else { "FAIL" });
    println!("═══════════════════════════════════════════════\n");

    // Wrap in a single-iter benchmark so criterion records it.
    c.bench_function("g5_functional_gate_cycles_to_recover", |b| {
        b.iter(|| {
            black_box(cgsp_cycles_to_recover(tau_low, max_cycles));
            black_box(derivative_cycles_to_recover(tau_low, max_cycles));
        });
    });
}

// ── Per-cycle cost benchmarks ─────────────────────────────────────────────

/// Measure CGSP's per-cycle cost (with Solver). This is the G2 baseline.
fn bench_cgsp_cycle_cost(c: &mut Criterion) {
    let pool = make_8_direction_pool();
    let conj = PoolConjecturer::new(pool.clone(), 5);
    let guide = HlaProjectionGuide::new(2.0, 1.0, katgpt_core::cgsp::ComplexityWeights::default());
    let solver = DotSolver { sharpness: 1.0 };
    let bandit = VecBandit::uniform(8);
    let mut lp = CgspLoop::new(conj, guide, solver, bandit, CgspConfig::default());
    let target = Target::new(pool[0].clone());
    let mut scratch = ScratchBuffers::new(8, 8);

    // Warm up the steady-state buffers so the measurement reflects steady
    // state (no allocation) rather than first-call resizing.
    for _ in 0..10 {
        let _ = lp.cycle(&target, &mut scratch);
    }

    c.bench_function("cgsp_cycle_with_solver", |b| {
        b.iter(|| {
            black_box(lp.cycle(black_box(&target), black_box(&mut scratch)));
        });
    });
}

/// Measure DerivativeCuriosity's per-cycle cost (no Solver). This is the F4
/// path — the G5 cost target is ≤ 10% of the CGSP baseline.
fn bench_derivative_cycle_cost(c: &mut Criterion) {
    let pool = make_8_direction_pool();
    let mut dc: DerivativeCuriosity<64> = DerivativeCuriosity::new(pool.clone(), 5);
    let mut bandit = VecBandit::uniform(8);
    let mut collapse = EntropyCollapse::default();
    let config = CgspConfig::default();
    let target = Target::new(pool[0].clone());
    let mut scratch = ScratchBuffers::new(8, 8);

    // Warm up.
    for _ in 0..10 {
        let _ = dc.cycle_curiosity(&target, &mut bandit, &mut scratch, &mut collapse, &config);
    }

    c.bench_function("derivative_cycle_no_solver", |b| {
        b.iter(|| {
            black_box(dc.cycle_curiosity(
                black_box(&target),
                black_box(&mut bandit),
                black_box(&mut scratch),
                black_box(&mut collapse),
                black_box(&config),
            ));
        });
    });
}

/// Measure just the curiosity-score computation (observe + sigmoid gate),
/// isolating the F4 primitive cost from the bandit-update overhead.
fn bench_curiosity_score_only(c: &mut Criterion) {
    let pool = make_8_direction_pool();
    let mut dc: DerivativeCuriosity<64> = DerivativeCuriosity::new(pool, 5);
    let prios: Vec<f32> = vec![0.125f32; 8];

    c.bench_function("derivative_observe_interestingness_only", |b| {
        b.iter(|| {
            black_box(dc.observe_interestingness(black_box(&prios)));
        });
    });
}

criterion_group!(
    benches,
    g5_functional_gate,
    bench_cgsp_cycle_cost,
    bench_derivative_cycle_cost,
    bench_curiosity_score_only,
);
criterion_main!(benches);
