//! Benchmark: OctreeCTC Reconstructive Navigation (Plan 248).
//!
//! Measures per-cycle latency for 3-step reconstruction (scalar vs SIMD vs matvec vs batch).
//! GOAT Gate: <200ns per 3-step reconstruction cycle.
//!
//! Measures:
//!   - Scalar `reconstruct()` — baseline
//!   - SIMD `reconstruct_simd()` — SIMD evolve (historically slower at 6×8)
//!   - Matvec `reconstruct_matvec()` — pre-computed weight matrix, single matvec expand
//!   - Multi-entity batch — N NPCs × same brain config, amortized SIMD
//!   - Per-step breakdown: expand → route → accumulate → evolve_hla
//!
//! Key finding: At 6 modules × 8-dim HLA, scalar wins for single entity.
//! SIMD only wins when batched across N ≥ 4 entities (48N f32 ops amortize NEON setup).

use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::sense::reconstruction::{
    BatchProjectionWeights, ProjectionWeights, ReconstructionConfig, ReconstructionState,
};
use katgpt_core::types::SenseKind;

const ITERS: usize = 10_000;

// ── Plan 277 G2: Synthetic Emotional-Event Trace ──────────────────
//
// Mirrors the in-crate unit test `surprise_detects_emotional_events_g2_gate`
// but lives in the bench suite for visibility + diagnostic output. Same
// proven design: zero-init HLA (no startup transient), additive-only events
// (raw norm is monotone non-decreasing → peaks at the LAST tick, never at an
// event), absolute detection threshold.
#[cfg(feature = "temporal_deriv")]
const G2_TRACE_LEN: usize = 1000;

#[cfg(feature = "temporal_deriv")]
const G2_EVENT_TICKS: &[usize] = &[200, 500, 800];

/// Half-window around an event tick where a surprise peak counts as a true
/// positive. Plan 277 T2.6 specifies ±10 ticks.
#[cfg(feature = "temporal_deriv")]
const G2_EVENT_WINDOW: usize = 10;

/// Absolute surprise_norm threshold for peak detection. Calibrated to sit
/// above the converged-stationary floor (~0) and below the event spike
/// magnitude given α_fast=0.3, α_slow=0.03.
#[cfg(feature = "temporal_deriv")]
const G2_DETECT_THRESHOLD: f32 = 0.05;

/// Returns the additive event delta to inject at the given tick, or zero.
///
/// Each event adds magnitude to a DIFFERENT dimension so the raw HLA norm is
/// strictly monotone non-decreasing (it only ever goes up). This means the
/// raw norm's global peak is at the LAST tick — never at an event — which is
/// the baseline contrast the G2 gate demonstrates.
///
/// - t=200: combat onset → dim 0 (arousal) +0.6
/// - t=500: loot drop → dim 1 (valence) +0.4
/// - t=800: encounter → dim 2 (social) +0.5
#[cfg(feature = "temporal_deriv")]
fn g2_event_delta(tick: usize) -> [f32; 8] {
    match tick {
        200 => [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        500 => [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        800 => [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        _ => [0.0; 8],
    }
}

fn make_brain_with_6_modules() -> NpcBrain {
    let builder = SenseOctreeBuilder::new(3);
    let kinds = [
        SenseKind::CommonSense,
        SenseKind::FighterSense,
        SenseKind::GameTheorySense,
        SenseKind::SpatialSense,
        SenseKind::SocialSense,
        SenseKind::SkillSense,
    ];
    let modules: Vec<_> = kinds
        .iter()
        .enumerate()
        .map(|(i, &kind)| {
            let emb = KgEmbedding {
                entity_hash: kind as u64,
                relation_hash: kind as u64,
                embedding: [0.5; 8],
                sign: true,
                confidence: 1.0,
            };
            let m = builder.build(kind, &[emb]);
            // Vary confidence per module
            let mut m = m;
            m.confidence = 0.3 + 0.1 * i as f32;
            m.commit();
            m
        })
        .collect();

    let mut brain = NpcBrain::compose(modules);
    brain.hla_state = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8];
    brain
}

// ── Full Cycle Benchmarks ────────────────────────────────────────

fn bench_reconstruct_scalar(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct(brain);
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct(brain);
        std::hint::black_box(&state);
    }
    start.elapsed().as_nanos() as f64 / ITERS as f64
}

fn bench_reconstruct_simd(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct_simd(brain);
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct_simd(brain);
        std::hint::black_box(&state);
    }
    start.elapsed().as_nanos() as f64 / ITERS as f64
}

fn bench_reconstruct_matvec(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    // Pre-compute weights ONCE (production path — weights survive across ticks)
    let weights = ProjectionWeights::from_brain(brain);

    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct_with_weights(&weights);
    }

    // Benchmark: state creation is cheap, weights are pre-computed
    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct_with_weights(&weights);
        std::hint::black_box(&state);
    }
    start.elapsed().as_nanos() as f64 / ITERS as f64
}

// ── Per-Step Breakdown ───────────────────────────────────────────

fn bench_step_scalar(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand(brain);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla();
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand(brain);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla();
        std::hint::black_box(&state);
    }
    start.elapsed().as_nanos() as f64 / ITERS as f64
}

fn bench_step_matvec(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    let weights = ProjectionWeights::from_brain(brain);

    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand_with_weights(&weights);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla();
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand_with_weights(&weights);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla();
        std::hint::black_box(&state);
    }
    start.elapsed().as_nanos() as f64 / ITERS as f64
}

// ── Multi-Entity Batch ───────────────────────────────────────────

/// Benchmark batch expand across N entities.
/// Measures throughput (ns per entity) for different batch sizes.
fn bench_batch_expand(brain: &NpcBrain, n_entities: usize) -> f64 {
    let batch_weights = BatchProjectionWeights::new(brain, n_entities);

    // Prepare N varied HLA states
    let mut hla_batch = vec![0.0f32; n_entities * 8];
    for e in 0..n_entities {
        let off = e * 8;
        let base = 0.1 * (e + 1) as f32;
        hla_batch[off..off + 8].copy_from_slice(&[
            base,
            base + 0.2,
            base + 0.1,
            base + 0.3,
            base + 0.15,
            base + 0.05,
            base + 0.25,
            base + 0.35,
        ]);
    }
    let mut activations_out = vec![0.0f32; n_entities * 6];

    // Warmup
    for _ in 0..100 {
        batch_weights.expand_batch(&hla_batch, &mut activations_out);
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        batch_weights.expand_batch(&hla_batch, &mut activations_out);
        std::hint::black_box(&activations_out);
    }
    let total_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;
    total_ns / n_entities as f64 // per-entity cost
}

/// Benchmark: scalar expand per entity (baseline for batch comparison).
fn bench_scalar_expand_per_entity(brain: &NpcBrain, n_entities: usize) -> f64 {
    let mut hla_batch = vec![0.0f32; n_entities * 8];
    for e in 0..n_entities {
        let off = e * 8;
        let base = 0.1 * (e + 1) as f32;
        hla_batch[off..off + 8].copy_from_slice(&[
            base,
            base + 0.2,
            base + 0.1,
            base + 0.3,
            base + 0.15,
            base + 0.05,
            base + 0.25,
            base + 0.35,
        ]);
    }

    // Warmup
    for _ in 0..100 {
        for e in 0..n_entities {
            let off = e * 8;
            let hla: &[f32; 8] = unsafe { &*(&hla_batch[off] as *const f32 as *const [f32; 8]) };
            for module in &brain.modules {
                let _ = std::hint::black_box(module.project(hla));
            }
        }
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        for e in 0..n_entities {
            let off = e * 8;
            let hla: &[f32; 8] = unsafe { &*(&hla_batch[off] as *const f32 as *const [f32; 8]) };
            for module in &brain.modules {
                let _ = std::hint::black_box(module.project(hla));
            }
        }
    }
    let total_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;
    total_ns / n_entities as f64
}

/// Benchmark: matvec expand per entity (single-entity pre-computed matrix).
fn bench_matvec_expand_per_entity(brain: &NpcBrain, n_entities: usize) -> f64 {
    let weights = ProjectionWeights::from_brain(brain);

    let mut hla_batch = vec![0.0f32; n_entities * 8];
    for e in 0..n_entities {
        let off = e * 8;
        let base = 0.1 * (e + 1) as f32;
        hla_batch[off..off + 8].copy_from_slice(&[
            base,
            base + 0.2,
            base + 0.1,
            base + 0.3,
            base + 0.15,
            base + 0.05,
            base + 0.25,
            base + 0.35,
        ]);
    }

    // Warmup
    for _ in 0..100 {
        for e in 0..n_entities {
            let off = e * 8;
            let mut dots = [0.0f32; 6];
            katgpt_core::simd::simd_matmul_rows(
                &mut dots,
                &weights.matrix,
                &hla_batch[off..off + 8],
                6,
                8,
            );
            std::hint::black_box(dots);
        }
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        for e in 0..n_entities {
            let off = e * 8;
            let mut dots = [0.0f32; 6];
            katgpt_core::simd::simd_matmul_rows(
                &mut dots,
                &weights.matrix,
                &hla_batch[off..off + 8],
                6,
                8,
            );
            std::hint::black_box(dots);
        }
    }
    let total_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;
    total_ns / n_entities as f64
}

// ── Plan 277 G2: Trace Runner & Peak Detector ────────────────────

/// Run the 1000-tick synthetic emotional-event trace and return two signal
/// vectors: `(surprise_norm_trace, hla_norm_trace)`, each of length
/// `G2_TRACE_LEN`.
///
/// Flow per tick:
///   1. If an event fires this tick, `inject_hla_delta` applies the step
///      change to the HLA (does NOT touch the surprise kernel).
///   2. `evolve_hla` runs the leaky integrator step (no-op when total=0 since
///      no evidence is accumulated) AND feeds the post-update HLA into the
///      surprise kernel via `observe_surprise_inner`.
///   3. Record `surprise_norm()` and `‖hla‖₂`.
///
/// HLA starts at `[0.0; 8]` — this matches the kernel's zero-init EMAs, so
/// there is no startup transient (the first observe sees a zero signal → zero
/// derivative). Between events the HLA is constant (leaky_step no-ops on
/// zero evidence), so the only signal the kernel sees is the event-driven
/// step change.
#[cfg(feature = "temporal_deriv")]
fn g2_run_trace() -> (Vec<f32>, Vec<f32>) {
    let config = ReconstructionConfig::default();
    let mut state = ReconstructionState::with_config([0.0; 8], config);

    let mut surprise_trace = Vec::with_capacity(G2_TRACE_LEN);
    let mut hla_norm_trace = Vec::with_capacity(G2_TRACE_LEN);

    for tick in 0..G2_TRACE_LEN {
        let delta = g2_event_delta(tick);
        if delta != [0.0; 8] {
            state.inject_hla_delta(delta);
        }
        state.evolve_hla();
        surprise_trace.push(state.surprise_norm());
        let mut sq = 0.0f32;
        for &x in state.hla() {
            sq += x * x;
        }
        hla_norm_trace.push(sq.max(0.0).sqrt());
    }
    (surprise_trace, hla_norm_trace)
}

/// Find local maxima in `trace` that exceed `G2_DETECT_THRESHOLD`. Returns
/// peak tick indices.
///
/// A tick `t` is a peak if:
///   - `trace[t] > threshold`
///   - `trace[t] >= trace[t-1]` (non-strictly rising — handles plateaus)
///   - `trace[t] >= trace[t+1]` (non-strictly falling — handles plateaus)
///
/// Boundary ticks use `f32::MIN` for the missing neighbor, so a tick 0 value
/// above threshold that is ≥ tick 1 counts as a peak (defensive — should not
/// happen with the zero-init design, but keeps the detector total).
#[cfg(feature = "temporal_deriv")]
fn g2_find_peaks(trace: &[f32]) -> Vec<usize> {
    let mut peaks = Vec::new();
    for t in 0..trace.len() {
        if trace[t] <= G2_DETECT_THRESHOLD {
            continue;
        }
        let prev = if t == 0 { f32::MIN } else { trace[t - 1] };
        let next = if t + 1 == trace.len() {
            f32::MIN
        } else {
            trace[t + 1]
        };
        if trace[t] >= prev && trace[t] >= next {
            peaks.push(t);
        }
    }
    peaks
}

/// Evaluate the G2 gate on a peak set. Returns `(recall, false_positive_rate)`.
///
/// - `recall` = fraction of events that have ≥1 peak within ±`G2_EVENT_WINDOW`
///   ticks.
/// - `false_positive_rate` = fraction of peaks that fall OUTSIDE all event
///   windows.
#[cfg(feature = "temporal_deriv")]
fn g2_eval_gate(peaks: &[usize]) -> (f32, f32) {
    let n_events = G2_EVENT_TICKS.len();
    let mut events_detected = 0usize;
    for &evt in G2_EVENT_TICKS {
        let lo = evt.saturating_sub(G2_EVENT_WINDOW);
        let hi = (evt + G2_EVENT_WINDOW).min(G2_TRACE_LEN);
        if peaks.iter().any(|&p| p >= lo && p <= hi) {
            events_detected += 1;
        }
    }
    let recall = events_detected as f32 / n_events as f32;

    let n_peaks = peaks.len();
    let false_positives = if n_peaks == 0 {
        0
    } else {
        peaks
            .iter()
            .filter(|&&p| {
                !G2_EVENT_TICKS
                    .iter()
                    .any(|&evt| p >= evt.saturating_sub(G2_EVENT_WINDOW) && p <= evt + G2_EVENT_WINDOW)
            })
            .count()
    };
    let fpr = false_positives as f32 / n_peaks as f32;
    (recall, fpr)
}

/// Run the full Plan 277 G2 gate and print the verdict.
///
/// Gate: recall ≥ 0.80 AND false_positive_rate ≤ 0.10.
///
/// Also demonstrates the baseline contrast: because events are additive-only
/// on distinct dimensions, the raw `‖hla‖₂` is monotone non-decreasing. Its
/// global peak is therefore at the LAST tick — far from any event. The
/// surprise kernel's global peak, by contrast, is near an event. This is the
/// orthogonality proof: the derivative and the raw norm peak at different
/// places, so they carry complementary information.
#[cfg(feature = "temporal_deriv")]
fn run_g2_gate() {
    println!("\n=== Plan 277 G2: HLA Surprise — Synthetic Emotional-Event Trace ===");
    println!("Trace length: {G2_TRACE_LEN} ticks");
    println!("Events: {G2_EVENT_TICKS:?} (combat@200 dim0, loot@500 dim1, encounter@800 dim2)");
    println!("Event window: ±{G2_EVENT_WINDOW} ticks");
    println!("Detection threshold: {G2_DETECT_THRESHOLD}\n");

    let (surprise_trace, hla_norm_trace) = g2_run_trace();

    // ── Surprise signal ──
    let s_max = surprise_trace.iter().cloned().fold(0.0f32, f32::max);
    let s_peaks = g2_find_peaks(&surprise_trace);
    let (s_recall, s_fpr) = g2_eval_gate(&s_peaks);

    println!("Surprise signal max: {s_max:.4}");
    println!("Peaks found: {} at ticks {:?}", s_peaks.len(), s_peaks);
    println!("Recall:  {s_recall:.2}  (target ≥ 0.80)");
    println!("FPR:     {s_fpr:.2}  (target ≤ 0.10)");
    let s_pass = s_recall >= 0.80 && s_fpr <= 0.10;
    println!(
        "G2 gate (surprise): {}",
        if s_pass { "PASS ✅" } else { "FAIL ❌" }
    );

    // ── Baseline: raw HLA norm ──
    //
    // Events are additive-only on distinct dims → raw norm is monotone
    // non-decreasing → global peak is at the LAST tick, not at any event.
    // The surprise kernel peaks AT events; the raw norm peaks at the end.
    // This is the orthogonality the G2 gate proves.
    //
    // Note: raw norm is flat from the last event (t=800) to t=999 (no decay —
    // leaky_step no-ops on zero evidence). `max_by` returns the LAST element
    // among ties, so raw_argmax = 999, which is far from any event. This
    // matches the semantics of the in-crate unit test.
    let raw_argmax = (0..G2_TRACE_LEN)
        .max_by(|&a, &b| {
            hla_norm_trace[a]
                .partial_cmp(&hla_norm_trace[b])
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .expect("non-empty trace");
    let h_max = hla_norm_trace[raw_argmax];
    let h_near_event = G2_EVENT_TICKS
        .iter()
        .any(|&e| raw_argmax.abs_diff(e) <= G2_EVENT_WINDOW);

    // Surprise global argmax (last occurrence among ties, same semantics).
    let surprise_argmax = (0..G2_TRACE_LEN)
        .max_by(|&a, &b| {
            surprise_trace[a]
                .partial_cmp(&surprise_trace[b])
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .expect("non-empty trace");
    let s_near_event = G2_EVENT_TICKS
        .iter()
        .any(|&e| surprise_argmax.abs_diff(e) <= G2_EVENT_WINDOW);
    let argmax_gap = raw_argmax.abs_diff(surprise_argmax);

    println!("\nBaseline: raw ‖hla‖₂");
    println!("Max: {h_max:.4} at tick {raw_argmax} (near event: {h_near_event})");
    println!(
        "Surprise max: {s_max:.4} at tick {surprise_argmax} (near event: {s_near_event})"
    );
    println!("Argmax gap: {argmax_gap} ticks (target > {G2_EVENT_WINDOW} for orthogonality)");

    // ── Verdict ──
    let n_events = G2_EVENT_TICKS.len();
    println!("\n--- G2 Verdict ---");
    let orthogonality_pass = !h_near_event && s_near_event && argmax_gap > G2_EVENT_WINDOW;
    if s_pass && orthogonality_pass {
        println!(
            "PASS: surprise_norm() detects {}/{} events with {:.0}% precision;",
            (s_recall * n_events as f32) as usize,
            n_events,
            (1.0 - s_fpr) * 100.0
        );
        println!("      raw norm peaks at tick {raw_argmax} (away from events) — orthogonality proven.");
    } else {
        if !s_pass {
            println!("FAIL: recall={s_recall:.2} (need ≥0.80), fpr={s_fpr:.2} (need ≤0.10)");
        }
        if !orthogonality_pass {
            println!("FAIL: orthogonality (raw_argmax={raw_argmax}, surprise_argmax={surprise_argmax}, gap={argmax_gap})");
        }
    }

    // Hard assertion so `cargo bench` fails the gate (non-zero exit) on regression.
    assert!(
        s_pass,
        "G2 gate FAILED: surprise recall={s_recall:.2} (need ≥0.80), fpr={s_fpr:.2} (need ≤0.10). \
         The derivative kernel must detect emotional events; if this fails, check alpha_fast/alpha_slow tuning."
    );
    assert!(
        orthogonality_pass,
        "G2 orthogonality FAILED: raw norm argmax={raw_argmax} should be far from events, \
         surprise argmax={surprise_argmax} should be near an event."
    );
}

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 248: OctreeCTC Reconstruction Benchmark ===\n");

    let brain = make_brain_with_6_modules();
    let config = ReconstructionConfig::default(); // 3 steps

    // Report SIMD level
    let level = katgpt_core::simd::simd_level();
    println!("SIMD level: {level:?}");

    println!(
        "Config: max_steps={}, lr={}",
        config.max_steps, config.hla_learning_rate
    );
    println!("Modules: {}", brain.modules.len());
    println!("Iterations: {ITERS}\n");

    // ── Full 3-Step Cycle ──
    println!("=== Full 3-Step Cycle ===");
    let scalar_ns = bench_reconstruct_scalar(&brain, config);
    let simd_ns = bench_reconstruct_simd(&brain, config);
    let matvec_ns = bench_reconstruct_matvec(&brain, config);

    println!("Scalar:            {scalar_ns:>8.1} ns/cycle");
    println!(
        "SIMD (evolve):     {simd_ns:>8.1} ns/cycle  ({:.2}×)",
        scalar_ns / simd_ns
    );
    println!(
        "Matvec (batched):  {matvec_ns:>8.1} ns/cycle  ({:.2}×)",
        scalar_ns / matvec_ns
    );

    // Find GOAT
    let best_ns = scalar_ns.min(simd_ns).min(matvec_ns);
    let goat_pass = best_ns < 200.0;
    println!(
        "\nGOAT (<200ns): {} — best = {best_ns:.1} ns",
        if goat_pass { "PASS ✅" } else { "FAIL ❌" }
    );

    // ── Per-Step Breakdown ──
    println!("\n=== Per-Step Breakdown (expand+route+accumulate+evolve) ===");
    let step_scalar_ns = bench_step_scalar(&brain, config);
    let step_matvec_ns = bench_step_matvec(&brain, config);

    println!("Scalar step:       {step_scalar_ns:>8.1} ns");
    println!(
        "Matvec step:       {step_matvec_ns:>8.1} ns  ({:.2}×)",
        step_scalar_ns / step_matvec_ns
    );

    // ── Multi-Entity Batch Expand ──
    println!("\n=== Multi-Entity Batch Expand (per-entity ns) ===");
    println!(
        "{:>4} {:>12} {:>12} {:>12} {:>8}",
        "N", "scalar", "matvec", "batch", "best"
    );
    println!("{}", "-".repeat(52));

    for &n in &[1, 2, 4, 8, 16, 32] {
        let s_ns = bench_scalar_expand_per_entity(&brain, n);
        let m_ns = bench_matvec_expand_per_entity(&brain, n);
        let b_ns = bench_batch_expand(&brain, n);
        let best = s_ns.min(m_ns).min(b_ns);
        let best_label = if best == s_ns {
            "scalar"
        } else if best == m_ns {
            "matvec"
        } else {
            "batch"
        };
        println!("{n:>4} {s_ns:>10.1} ns {m_ns:>10.1} ns {b_ns:>10.1} ns {best_label:>8}");
    }

    // ── Correctness ──
    println!("\n=== Correctness ===");

    // Matvec matches scalar
    let weights = ProjectionWeights::from_brain(&brain);
    let mut state_scalar = ReconstructionState::with_config(brain.hla_state, config);
    let _ = state_scalar.reconstruct(&brain);

    let mut state_matvec = ReconstructionState::with_config(brain.hla_state, config);
    let _ = state_matvec.reconstruct_with_weights(&weights);

    let mut max_diff = 0.0f32;
    for i in 0..8 {
        let diff = (state_scalar.hla()[i] - state_matvec.hla()[i]).abs();
        max_diff = max_diff.max(diff);
    }
    println!("Max HLA diff (scalar vs matvec): {max_diff:.6e}");
    assert!(
        max_diff < 1e-4,
        "Matvec should match scalar, diff={max_diff}"
    );
    println!("Matvec equivalence: PASS ✅");

    // Batch expand matches scalar
    let batch_weights = BatchProjectionWeights::new(&brain, 4);
    let hla_batch = [
        0.3f32, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8, 0.2f32, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7,
        0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8f32, 0.6, 0.4, 0.2, 0.7, 0.5, 0.3, 0.1,
    ];
    let mut activations_out = [0.0f32; 24];
    batch_weights.expand_batch(&hla_batch, &mut activations_out);

    // Verify entity 0 matches scalar expand
    let mut state_0 =
        ReconstructionState::with_config([0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8], config);
    let scalar_acts = state_0.expand(&brain);
    let mut max_batch_diff = 0.0f32;
    for i in 0..6 {
        let diff = (scalar_acts[i] - activations_out[i]).abs();
        max_batch_diff = max_batch_diff.max(diff);
    }
    println!("Max batch diff (entity 0): {max_batch_diff:.6e}");
    assert!(
        max_batch_diff < 1e-4,
        "Batch expand should match scalar, diff={max_batch_diff}"
    );
    println!("Batch equivalence: PASS ✅");

    // ── Plan 277 G2: HLA Surprise Gate ──
    // Only runs when the `temporal_deriv` feature is enabled (alongside the
    // bench's required `sense_composition`).
    #[cfg(feature = "temporal_deriv")]
    {
        run_g2_gate();
    }
}
