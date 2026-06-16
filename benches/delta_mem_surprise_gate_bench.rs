//! δ-Mem Temporal Surprise-Gate Benchmark (Plan 277 Phase 3, G3).
//!
//! # G3 gate criterion
//!
//! **PASS** requires ALL of:
//! - Write suppression ≥ 30% (`write_suppression_rate ≥ 0.30`).
//! - Recall loss ≤ 5% (mean cosine recall of gated vs baseline within 5%).
//!
//! # Synthetic query stream
//!
//! Rank-8 keys are L2-normalized (matching `FeatureHasher::hash_key`). The
//! stream interleaves:
//!
//! - **Background writes** (80%): keys sampled from a tight cluster around a
//!   slowly-drifting centroid. These are "boring" — the surprise gate should
//!   suppress most of them after the kernel's slow EMA locks onto the
//!   centroid direction.
//! - **Event writes** (20%): keys drawn from well-separated directions
//!   (rotated by ~90° from the running centroid). These are sharp directional
//!   changes → high surprise → written.
//!
//! # Recall metric
//!
//! After the full stream, we probe the memory with each **event key** and
//! measure the cosine similarity between `read(k_event)` and the written
//! `v_event`. Higher = better recall of the surprising events.
//!
//! The hypothesis: the gated variant writes fewer background samples, so
//! background writes overwrite the event associations less. Recall of events
//! should be **equal or better**, not worse.
//!
//! # θ_surprise
//!
//! Default 0.05 (from `DEFAULT_THETA_SURPRISE`). The bench also sweeps a few
//! values to document sensitivity.
//!
//! Run with:
//! ```bash
//! cargo run --release --bench delta_mem_surprise_gate_bench --features delta_mem,temporal_deriv
//! ```

use std::time::Instant;

use katgpt_rs::pruners::delta_mem::{DeltaMemoryConfig, DeltaMemoryState};

// Import the default θ so the bench always tracks the production default.
// This bench requires both delta_mem + temporal_deriv (see Cargo.toml
// required-features), so the const is always in scope.
use katgpt_rs::pruners::delta_mem::state::DEFAULT_THETA_SURPRISE;

// ── Stream construction ─────────────────────────────────────────────────

const RANK: usize = 8;
const N_TOTAL: usize = 1_000;
const FRAC_EVENTS: f32 = 0.20; // 20% events, 80% background
const SEED: u64 = 0xC0FFEE_BABE_1234;

/// Deterministic LCG (no extra crate dependency; fastrand is available but
/// this keeps the bench self-contained and reproducible).
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes LCG.
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
    fn next_gauss(&mut self) -> f32 {
        // Box-Muller.
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

/// L2-normalize an 8-vector in place, returning the normalized vec.
fn l2_normalize(v: &mut [f32; RANK]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// A single write in the synthetic stream.
struct StreamSample {
    key: [f32; RANK],
    value: [f32; RANK],
    is_event: bool,
}

/// Build the synthetic stream. Deterministic given `SEED`.
///
/// Background keys: centroid (fixed) + small gaussian noise → cluster.
/// Event keys: a rotated centroid (≈90°) — a sharp directional jump.
/// Background and event indices are interleaved by a deterministic hash of
/// the sample index so the pattern is not periodic.
fn build_stream() -> Vec<StreamSample> {
    let mut rng = Rng(SEED);
    let n_events = (N_TOTAL as f32 * FRAC_EVENTS) as usize;

    // Decide which indices are events (deterministic, ~20%).
    let mut is_event = vec![false; N_TOTAL];
    let mut remaining = n_events;
    // Place events at roughly evenly-spaced positions with jitter.
    let mut pos = 0usize;
    while remaining > 0 && pos < N_TOTAL {
        is_event[pos] = true;
        remaining -= 1;
        let stride = (N_TOTAL / n_events.max(1)).max(1);
        let jitter = (rng.next_u32() as usize) % 3;
        pos += stride + jitter;
    }

    // Two well-separated centroids on the 8-sphere.
    let mut centroid_bg = [0.0f32; RANK];
    let mut centroid_evt = [0.0f32; RANK];
    for i in 0..RANK {
        centroid_bg[i] = if i < 4 { 1.0 } else { 0.0 };
        centroid_evt[i] = if i >= 4 { 1.0 } else { 0.0 }; // ~orthogonal
    }
    l2_normalize(&mut centroid_bg);
    l2_normalize(&mut centroid_evt);

    let mut out = Vec::with_capacity(N_TOTAL);
    for i in 0..N_TOTAL {
        let (mut key, is_evt) = if is_event[i] {
            // Event: centroid_evt + mild noise → sharp jump from bg centroid.
            let mut k = centroid_evt;
            for x in k.iter_mut() {
                *x += 0.05 * rng.next_gauss();
            }
            (k, true)
        } else {
            // Background: centroid_bg + mild noise → cluster.
            let mut k = centroid_bg;
            for x in k.iter_mut() {
                *x += 0.03 * rng.next_gauss();
            }
            (k, false)
        };
        l2_normalize(&mut key);

        // Value: distinct per-event so recall is meaningful. For events we use
        // a fixed signature so probe → written-value comparison is clean.
        // For background we use a random-ish value (irrelevant to recall).
        let mut value = [0.0f32; RANK];
        if is_evt {
            // Event values: encode the event index modulo a small set so the
            // probe can find the written value.
            value[i % RANK] = 1.0;
        } else {
            for x in value.iter_mut() {
                *x = 0.5 * rng.next_f32() - 0.25;
            }
        }
        l2_normalize(&mut value);

        out.push(StreamSample {
            key,
            value,
            is_event: is_evt,
        });
    }
    out
}

// ── Recall metric ───────────────────────────────────────────────────────

/// Compute mean cosine similarity between `read(k)` and the stored `v` over
/// all EVENT samples. This is the recall metric: how well does the memory
/// retrieve the value associated with a surprising key?
fn event_recall_cosine(state: &DeltaMemoryState, stream: &[StreamSample]) -> f32 {
    let mut sum: f32 = 0.0;
    let mut count: usize = 0;
    for s in stream.iter().filter(|s| s.is_event) {
        let readout = state.read(&s.key);
        let mut dot: f32 = 0.0;
        let mut na: f32 = 0.0;
        let mut nb: f32 = 0.0;
        for (a, b) in readout.iter().zip(s.value.iter()) {
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let denom = na.sqrt().max(1e-8) * nb.sqrt().max(1e-8);
        sum += dot / denom;
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    sum / count as f32
}

// ── Bench scenarios ─────────────────────────────────────────────────────

struct ScenarioResult {
    name: String,
    writes_total: u64,
    writes_gated: u64,
    suppression_rate: f32,
    recall_cosine: f32,
    elapsed_us: u128,
}

/// Run δ-Mem over the stream in the given mode, returning metrics.
fn run_scenario(name: &str, stream: &[StreamSample], gated: bool, theta: f32) -> ScenarioResult {
    let config = DeltaMemoryConfig::default(); // rank 8
    let mut state = DeltaMemoryState::new(config);

    #[cfg(feature = "temporal_deriv")]
    if gated {
        let installed = state.enable_surprise_gate();
        debug_assert!(installed, "rank-8 must install the gate");
        state.set_theta_surprise(theta);
    }
    #[cfg(not(feature = "temporal_deriv"))]
    let _ = (gated, theta);

    let t0 = Instant::now();
    for s in stream.iter() {
        state.write(&s.key, &s.value);
    }
    let elapsed = t0.elapsed();

    #[cfg(feature = "temporal_deriv")]
    let (wt, wg, supp) = (
        state.writes_total(),
        state.writes_gated(),
        state.write_suppression_rate(),
    );
    #[cfg(not(feature = "temporal_deriv"))]
    let (wt, wg, supp) = (0, 0, 0.0);

    let recall = event_recall_cosine(&state, stream);

    ScenarioResult {
        name: name.to_string(),
        writes_total: wt,
        writes_gated: wg,
        suppression_rate: supp,
        recall_cosine: recall,
        elapsed_us: elapsed.as_micros(),
    }
}

fn print_row(r: &ScenarioResult) {
    println!(
        "  {:<34} | writes={:>5} gated={:>5} | supp={:>6.2}% | recall_cos={:.4} | {:>7}μs",
        r.name, r.writes_total, r.writes_gated, r.suppression_rate * 100.0, r.recall_cosine,
        r.elapsed_us,
    );
}

fn main() {
    println!("═══ Plan 277 Phase 3 — G3 Gate: δ-Mem Temporal Surprise Gate ═══");
    println!();
    println!(
        "Stream: {} writes, {}% events, rank={}, L2-normalized keys",
        N_TOTAL,
        (FRAC_EVENTS * 100.0) as u32,
        RANK
    );
    println!();

    let stream = build_stream();
    let n_events = stream.iter().filter(|s| s.is_event).count();
    let n_bg = stream.len() - n_events;
    println!("  events: {}, background: {}", n_events, n_bg);
    println!();

    // ── Baseline (no gate) ──────────────────────────────────────────────
    let baseline = run_scenario("baseline (always write)", &stream, false, 0.0);

    // ── Gated, default θ ───────────────────────────────────────────────
    // This bench requires both delta_mem + temporal_deriv (see Cargo.toml
    // required-features), so DEFAULT_THETA_SURPRISE is always in scope.
    let gated_default = run_scenario(
        "gated θ=default (from const)",
        &stream,
        true,
        DEFAULT_THETA_SURPRISE,
    );

    // ── Sensitivity sweep ───────────────────────────────────────────────
    let gated_003 = run_scenario("gated θ=0.03 (lenient)", &stream, true, 0.03);
    let gated_005 = run_scenario("gated θ=0.05 (prev default)", &stream, true, 0.05);
    let gated_015 = run_scenario("gated θ=0.15 (strict)", &stream, true, 0.15);
    let gated_020 = run_scenario("gated θ=0.20 (very strict)", &stream, true, 0.20);

    println!("── Results ──────────────────────────────────────────────────────");
    print_row(&baseline);
    print_row(&gated_default);
    println!("── θ sensitivity ────────────────────────────────────────────────");
    print_row(&gated_003);
    print_row(&gated_005);
    print_row(&gated_015);
    print_row(&gated_020);
    println!();

    // ── G3 verdict ──────────────────────────────────────────────────────
    let supp_pct = gated_default.suppression_rate * 100.0;
    let recall_loss_pct =
        (baseline.recall_cosine - gated_default.recall_cosine).max(0.0) / baseline.recall_cosine
            * 100.0;

    let supp_pass = supp_pct >= 30.0;
    let recall_pass = recall_loss_pct <= 5.0;

    println!("── G3 Verdict (gated θ={:.2} = default vs baseline) ────────────", DEFAULT_THETA_SURPRISE);
    println!(
        "  write suppression: {:.2}%  (target ≥ 30%)  → {}",
        supp_pct,
        if supp_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  recall loss:       {:.2}%  (target ≤ 5%)   → {}",
        recall_loss_pct,
        if recall_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  baseline recall_cos={:.4}  gated recall_cos={:.4}",
        baseline.recall_cosine, gated_default.recall_cosine
    );
    println!();
    println!(
        "  G3 OVERALL: {}",
        if supp_pass && recall_pass {
            "PASS"
        } else {
            "FAIL"
        }
    );

    // Exit code: 0 on PASS, 2 on FAIL (so CI can gate on it).
    std::process::exit(if supp_pass && recall_pass { 0 } else { 2 });
}
