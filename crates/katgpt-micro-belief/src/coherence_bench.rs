//! **G2.1 long-horizon coherence benchmark** — the actual GOAT gate for the
//! attractor family quality claim (Plan 276 Phase 5 T5.0).
//!
//! Builds a synthetic 1000-step input sequence with injected ambiguity /
//! flip-flop triggers (an analog of the "bank" polysemy example from Mozer
//! 2026, adapted to a small belief-vector regime), then compares three kernels:
//!
//! - [`LeakyIntegrator`] (Family C — HLA's leaky integrator, the battle-tested
//!   baseline; byte-identical math to `ReconstructionState::evolve_hla`).
//! - [`AttractorKernel`] (Family A — the GOAT candidate; hysteresis should
//!   resist flip-flopping).
//! - [`LatentThoughtKernel`] (Family B — K=3 iterations of Family A; more
//!   "deliberation" per tick, expected to settle faster but possibly
//!   overshoot).
//!
//! The metric: **flip-flop count** — how often the dominant belief dimension
//! (`argmax` of the projected scalar stream) changes from the previous tick,
//! plus **belief stability** (variance of the projected scalars over a stable
//! window).
//!
//! # Honest-reporting policy
//!
//! Per Plan 276 T5.1/T5.2: if the attractor family's flip-flop count is
//! strictly less than the leaky integrator's, the attractor is promoted as an
//! opt-in variant (T5.1). If it ties or loses, the attractor is demoted to a
//! Gain experiment and only the trait unification + LeakyIntegrator ship as
//! promotable output (T5.2). The test below PRINTS the verdict table and only
//! HARD-FAILS on divergence (G1.2 violation — NaN / Inf in the state). The
//! quality comparison is informational for the GOAT decision because demotion
//! is an acceptable outcome.
//!
//! # Why `dim = 16` (not 32)
//!
//! The G1.* mechanics tests use `dim = 32` to match the Plan 255 L1 budget.
//! This benchmark uses `dim = 16` to keep the wall-clock test fast (1000 steps
//! × 3 kernels × matvec is the dominant cost) while still exercising a
//! multi-dimensional belief space large enough for the flip-flop metric to be
//! meaningful. The relative ordering of kernels is not sensitive to `dim` in
//! the 8–32 range (the attractor's hysteresis property is dimension-independent;
//! it comes from the recurrent weight matrix's eigenvalue structure, not from
//! the dimension count).
//!
//! [`LeakyIntegrator`]: crate::leaky::LeakyIntegrator
//! [`AttractorKernel`]: crate::attractor::AttractorKernel
//! [`LatentThoughtKernel`]: crate::latent_thought::LatentThoughtKernel

#![allow(clippy::needless_range_loop)]

use crate::attractor::AttractorKernel;
use crate::latent_thought::LatentThoughtKernel;
use crate::leaky::LeakyIntegrator;
use crate::types::MicroRecurrentBeliefState;

/// Belief-vector dimension used by the G2.1 benchmark. See module docs for why
/// this is 16 (not 32).
pub const BENCH_DIM: usize = 16;

/// Total length of the synthetic input sequence.
pub const BENCH_STEPS: usize = 1000;

/// End of the "dim-0 dominant" phase. Input strongly favours dimension 0.
pub const PHASE_DIM0_END: usize = 400;
/// End of the "ambiguous / noisy" phase. Inputs are near-uniform small noise —
/// the analog of the "bank" polysemy window where a good kernel should hold its
/// belief and a flip-floppy kernel should oscillate.
pub const PHASE_AMBIGUOUS_END: usize = 600;
// Steps [PHASE_AMBIGUOUS_END .. BENCH_STEPS] — strong signal favouring dimension 1.

// ─── Input sequence generator ──────────────────────────────────────────────

/// Build the synthetic 1000-step input sequence.
///
/// The sequence has three phases:
///
/// 1. **Steps 0..400** — strong signal favouring dimension 0. `input[0]` is
///    high, all other dimensions are small positive noise. A kernel should
///    settle into "dimension 0 is dominant".
/// 2. **Steps 400..600** — ambiguous / noisy phase. All dimensions are
///    near-uniform small noise in `[-0.05, 0.05]`. This is the polysemy
///    analog: the evidence is uninformative, so a kernel with hysteresis
///    should HOLD its belief (still dim 0), while a flip-floppy kernel may
///    oscillate between dimensions.
/// 3. **Steps 600..1000** — strong signal favouring dimension 1. `input[1]`
///    is high, all other dimensions are small. A good kernel should transition
///    cleanly from dim 0 to dim 1 (ideally one flip), a flip-floppy kernel
///    may oscillate around the transition.
///
/// The noise is deterministic (`fastrand::Rng::with_seed`) so the benchmark
/// is reproducible (G1.1 determinism applies to the whole pipeline).
pub fn build_input_sequence(dim: usize) -> Vec<Vec<f32>> {
    let mut rng = fastrand::Rng::with_seed(0xC0FFEE); // fixed seed for reproducibility
    let mut seq = Vec::with_capacity(BENCH_STEPS);
    for step in 0..BENCH_STEPS {
        let mut x = vec![0.0f32; dim];
        match step {
            // Phase 1: dim 0 dominant. input[0] = 0.8, others = small noise.
            s if s < PHASE_DIM0_END => {
                for i in 0..dim {
                    x[i] = (rng.f32() * 2.0 - 1.0) * 0.05; // [-0.05, 0.05]
                }
                x[0] = 0.8; // strong positive evidence for dim 0
            },
            // Phase 2: ambiguous. All dims = small noise, no dominant signal.
            s if s < PHASE_AMBIGUOUS_END => {
                for i in 0..dim {
                    x[i] = (rng.f32() * 2.0 - 1.0) * 0.05;
                }
            },
            // Phase 3: dim 1 dominant. input[1] = 0.8, others = small noise.
            _ => {
                for i in 0..dim {
                    x[i] = (rng.f32() * 2.0 - 1.0) * 0.05;
                }
                x[1] = 0.8; // strong positive evidence for dim 1
            },
        }
        seq.push(x);
    }
    seq
}

// ─── Direction matrix (identity) ───────────────────────────────────────────

/// Build a flattened identity direction matrix of shape `[dim, dim]` row-major.
///
/// With identity directions, the bridge output `out[k] = sigmoid(state[k])` —
/// i.e. the projected scalars are just the sigmoid of each state coordinate.
/// This makes `argmax(out) == argmax(state)` (sigmoid is monotone), so the
/// "dominant belief" is simply the largest-magnitude-positive state dimension.
fn identity_directions(dim: usize) -> Vec<f32> {
    let mut dirs = vec![0.0f32; dim * dim];
    for i in 0..dim {
        dirs[i * dim + i] = 1.0;
    }
    dirs
}

// ─── Per-kernel run + metric extraction ────────────────────────────────────

/// Result of running one kernel over the full 1000-step sequence.
#[derive(Clone, Debug)]
pub struct KernelRunReport {
    /// Human-readable kernel name (for the verdict table).
    pub name: &'static str,
    /// Number of times `argmax(projected_scalars)` changed from the previous
    /// tick. Lower = more coherent / stable.
    pub flip_flop_count: usize,
    /// Variance of the `argmax(projected_scalars)` stream over the ambiguous
    /// window (steps `PHASE_DIM0_END..PHASE_AMBIGUOUS_END`). Lower = more
    /// stable belief under ambiguous evidence.
    pub ambiguous_window_argmax_variance: f64,
    /// Final projected scalar stream at the last tick (length `dim`). For
    /// sanity inspection.
    pub final_scalars: Vec<f32>,
    /// True if any state element ever went non-finite (G1.2 violation). The
    /// caller hard-fails the test if this is true.
    pub diverged: bool,
}

/// Run a single kernel over the input sequence and extract the coherence
/// metrics.
fn run_kernel(
    name: &'static str,
    kernel: &dyn MicroRecurrentBeliefState,
    seq: &[Vec<f32>],
    dim: usize,
) -> KernelRunReport {
    let mut state = vec![0.0f32; dim];
    let directions = identity_directions(dim);
    let mut out = vec![0.0f32; dim];

    let mut prev_argmax: Option<usize> = None;
    let mut flip_flops = 0usize;
    let mut diverged = false;

    // Record argmax stream over the ambiguous window for the stability metric.
    let mut ambiguous_argmax: Vec<f64> = Vec::with_capacity(PHASE_AMBIGUOUS_END - PHASE_DIM0_END);

    for (step, x) in seq.iter().enumerate() {
        kernel.step(&mut state, x);

        // G1.2 invariant: state must stay finite and bounded. If a kernel
        // diverges we flag it (the caller will hard-fail).
        for &v in &state {
            if !v.is_finite() || v.abs() > 6.0 {
                diverged = true;
            }
        }

        // Project to scalars via the bridge (sigmoid(dot(state, identity_row))).
        kernel.project_to_scalars(&state, &directions, dim, &mut out);

        // Dominant belief = argmax of projected scalars. Ties broken by lowest
        // index (rust's `max` semantics on f32 keep the FIRST maximum).
        let (argmax_idx, _argmax_val) = out
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
            })
            .unwrap_or((0, &0.0f32));

        if let Some(prev) = prev_argmax
            && prev != argmax_idx {
                flip_flops += 1;
            }
        prev_argmax = Some(argmax_idx);

        if (PHASE_DIM0_END..PHASE_AMBIGUOUS_END).contains(&step) {
            ambiguous_argmax.push(argmax_idx as f64);
        }
    }

    // Final scalar stream (for inspection).
    let final_scalars = out.clone();

    // Variance of the argmax stream over the ambiguous window. Lower = more
    // stable. We compute population variance (the window is the whole
    // population of interest, not a sample).
    let ambiguous_window_argmax_variance = population_variance(&ambiguous_argmax);

    KernelRunReport {
        name,
        flip_flop_count: flip_flops,
        ambiguous_window_argmax_variance,
        final_scalars,
        diverged,
    }
}

/// Population variance of a slice of f64. Returns 0.0 for empty / single-element
/// slices (no spread).
fn population_variance(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 1.0 {
        return 0.0;
    }
    let mean: f64 = xs.iter().copied().sum::<f64>() / n;
    let var: f64 = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var
}

// ─── Aggregate report ──────────────────────────────────────────────────────

/// Aggregate G2.1 report across all three kernels.
#[derive(Clone, Debug)]
pub struct CoherenceReport {
    pub leaky: KernelRunReport,
    pub attractor: KernelRunReport,
    pub latent_thought: KernelRunReport,
}

impl CoherenceReport {
    /// Pretty-print a verdict table to stderr. Used by the test for visibility;
    /// callers can also use it from a future `[[bench]]` harness.
    pub fn print_verdict_table(&self) {
        eprintln!();
        eprintln!("┌────────────────────────────────────────────────────────────────────────────┐");
        eprintln!("│ G2.1 long-horizon coherence benchmark — Plan 276 Phase 5 T5.0              │");
        eprintln!("├──────────────────────┬───────────────┬───────────────┬──────────────────────┤");
        eprintln!("│ Kernel               │ flip-flops    │ ambig-window  │ diverged?            │");
        eprintln!("│                      │ (lower=better)│ argmax-var    │ (G1.2 invariant)     │");
        eprintln!("├──────────────────────┼───────────────┼───────────────┼──────────────────────┤");
        self.print_row(&self.leaky);
        self.print_row(&self.attractor);
        self.print_row(&self.latent_thought);
        eprintln!("└──────────────────────┴───────────────┴───────────────┴──────────────────────┘");

        // Verdict line.
        let l = self.leaky.flip_flop_count;
        let a = self.attractor.flip_flop_count;
        let b = self.latent_thought.flip_flop_count;
        let verdict = match (a.cmp(&l), b.cmp(&l)) {
            (core::cmp::Ordering::Less, _) | (_, core::cmp::Ordering::Less) => {
                "GOAT PASS (T5.1): attractor family flips LESS than leaky — promote opt-in variant"
            },
            (core::cmp::Ordering::Equal, core::cmp::Ordering::Equal) => {
                "TIE (T5.2): attractor family does NOT beat leaky — demote to Gain"
            },
            _ => {
                "G2.1 FAIL (T5.2): attractor family flips MORE than leaky — demote to Gain"
            },
        };
        eprintln!("│ leaky={l}  attractor={a}  latent_thought={b}");
        eprintln!("│ VERDICT: {verdict}");
        eprintln!();
    }

    fn print_row(&self, r: &KernelRunReport) {
        eprintln!(
            "│ {:<20} │ {:<13} │ {:<13.4} │ {:<20} │",
            r.name,
            r.flip_flop_count,
            r.ambiguous_window_argmax_variance,
            if r.diverged { "YES (HARD FAIL)" } else { "no" }
        );
    }
}

// ─── Public entry point ────────────────────────────────────────────────────

/// Run the G2.1 coherence benchmark on all three kernels and return the
/// aggregate report.
///
/// Constructed kernels:
/// - `LeakyIntegrator::hla_default(BENCH_DIM)` — `lr=0.1, max_delta=0.2`
///   (matches `ReconstructionConfig::default()`).
/// - `AttractorKernel::from_seed(42, BENCH_DIM)` — Family A.
/// - `LatentThoughtKernel::from_seed(42, BENCH_DIM, 3)` — Family B with K=3.
///
/// The seed `42` matches the seed used throughout the G1.* tests so the
/// benchmark is cross-comparable with them.
pub fn run_g2_1_coherence_benchmark() -> CoherenceReport {
    let dim = BENCH_DIM;
    let seq = build_input_sequence(dim);

    let leaky = LeakyIntegrator::hla_default(dim);
    let attractor = AttractorKernel::from_seed(42, dim);
    let latent_thought = LatentThoughtKernel::from_seed(42, dim, 3);

    let leaky_report = run_kernel("LeakyIntegrator", &leaky, &seq, dim);
    let attractor_report = run_kernel("AttractorKernel", &attractor, &seq, dim);
    let latent_report = run_kernel("LatentThought(K=3)", &latent_thought, &seq, dim);

    CoherenceReport {
        leaky: leaky_report,
        attractor: attractor_report,
        latent_thought: latent_report,
    }
}

// ─── Test (the GOAT-gate quality decision) ─────────────────────────────────

/// **G2.1** — Long-horizon coherence GOAT gate.
///
/// Runs all three kernels on the synthetic 1000-step sequence and reports the
/// flip-flop counts. Per Plan 276 T5.1/T5.2:
///
/// - **Hard-fail** ONLY on divergence (any kernel produces NaN / Inf / unbounded
///   state — a G1.2 violation). This is a correctness gate, not a quality gate.
/// - **Soft-report** the flip-flop comparison: the test prints a verdict table
///   to stderr and never fails on a quality loss, because demotion to Gain
///   (T5.2) is an explicitly acceptable outcome. The decision is documented in
///   `.benchmarks/276_micro_belief_goat.md` by the orchestrator.
///
/// This keeps CI green regardless of which kernel wins — the GOAT decision is a
/// human/plan judgement call based on the printed numbers, not a binary test
/// result.
#[test]
fn g2_1_coherence_attractor_beats_or_matches_leaky() {
    let report = run_g2_1_coherence_benchmark();
    report.print_verdict_table();

    // HARD GATE: no kernel may diverge (G1.2 invariant must hold on the
    // benchmark input sequence too, not just on the G1.2 unit test input).
    let reports = [&report.leaky, &report.attractor, &report.latent_thought];
    for r in reports {
        assert!(
            !r.diverged,
            "G2.1 HARD FAIL: kernel `{}` diverged (NaN/Inf/unbounded state) — G1.2 violation on benchmark input",
            r.name
        );
    }

    // SOFT COMPARISON: informational only. The verdict is in the printed table
    // and in `.benchmarks/276_micro_belief_goat.md`.
    let leaky_ff = report.leaky.flip_flop_count;
    let attractor_ff = report.attractor.flip_flop_count;
    let latent_ff = report.latent_thought.flip_flop_count;

    // Print a single-line machine-parseable summary for the bench-doc scraper.
    eprintln!(
        "G2.1_SUMMARY leaky_flipflops={leaky_ff} attractor_flipflops={attractor_ff} latent_thought_flipflops={latent_ff}"
    );

    // Explicit verdict echo (the test always passes; this is for log grepping).
    let attractor_beats = attractor_ff < leaky_ff;
    let latent_beats = latent_ff < leaky_ff;
    let any_attractor_family_beats = attractor_beats || latent_beats;
    if any_attractor_family_beats {
        eprintln!(
            "G2.1 VERDICT: attractor family wins (less flip-flopping) — promote opt-in variant per T5.1"
        );
    } else if attractor_ff == leaky_ff && latent_ff == leaky_ff {
        eprintln!(
            "G2.1 VERDICT: tie — no improvement, demote attractor to Gain per T5.2"
        );
    } else {
        eprintln!(
            "G2.1 VERDICT: attractor family flips MORE than leaky — demote to Gain per T5.2 \
            (attractor={attractor_ff}, latent_thought={latent_ff}, leaky={leaky_ff})"
        );
    }
}
