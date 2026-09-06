//! Proposal 011 Phase 5 — SWE Trajectory Geometry Synthetic PoC (T5.1–T5.3b).
//!
//! Defend-or-refute PoC for Layer 4 of Proposal 011 (the modelless reframe):
//! **even when the underlying model proposes zero valid patches, the inference
//! loop's trajectory through patch-space has measurable geometry that differs
//! across failure modes — and that geometry is freezable via shipped substrate.**
//!
//! This PoC composes three already-shipped DEFAULT-ON / opt-in katgpt-core
//! primitives against *synthetic* trajectories that mimic distinct SWE-attempt
//! failure modes. No Rust-SWE-bench, no rubrc, no Kimi-K3 needed — it tests
//! whether the substrate itself has discriminative power, before any real-model
//! integration.
//!
//! # The sub-PoCs
//!
//! - **T5.1** — Does `latent_trajectory_geometry::from_states` produce
//!   measurably different geometry across distinct synthetic failure modes
//!   (committed-wrong / oscillation / drift / stuck / converged-correct)?
//! - **T5.2** — Does CUCG `evaluate` fire `Compress` on a synthetic test-pass
//!   event (high coherence, low rank, positive divergence, low novelty) and
//!   NOT fire on surrounding churn?
//! - **T5.3** — Does `CommittedFieldBlend::commit` produce a stable,
//!   BLAKE3-committable, non-degenerate blend from an all-fail trajectory
//!   summary? (Random-direction baseline — documents the concentration-of-
//!   measure failure that T5.3b fixes.)
//! - **T5.3b** — Do data-derived directions from cluster centroids fix the
//!   T5.3 failure? Dual-strategy test: geometry-encoded summary (should PASS)
//!   vs endpoint-position summary (should FAIL). The contrast documents the
//!   design constraint for T5.5: summary encoder MUST capture geometry.
//!
//! # Run
//!
//! ```bash
//! cargo bench --manifest-path Cargo.toml \
//!     --features "latent_trajectory_geometry closed_unit_compaction committed_field_blend" \
//!     --bench bench_011_swe_trajectory_geometry_poc -- --nocapture
//! ```
//!
//! See Issue 569 (removed per noise-reduction rule; resolution in `katgpt-rs/.docs/09_feature_catalog/opt_in_features.md` §29) (T5.1–T5.3)
//! and Issue 570 (T5.3b, same resolution location) for the
//! full defend-or-refute protocols + outcome-action tables.

#![cfg(feature = "latent_trajectory_geometry")]
#![allow(clippy::needless_range_loop)] // index loops used intentionally for multi-array indexing

use katgpt_core::compaction::rubrics::search::SearchRubric;
use katgpt_core::compaction::{
    Backstop, ClosedUnitCompactionGate, CompactionDecision, FireRule, RubricScratch,
};
use katgpt_core::latent_trajectory_geometry::{from_states, LatentTrajectoryGeometry};
use katgpt_core::personality_composition::sigmoid::sigmoid;
use katgpt_core::{ArchetypeFieldSource, TriArchetypeBlend};

// ─── Constants ─────────────────────────────────────────────────────────────

/// Latent state dimension for the synthetic trajectories. Matches the HLA
/// scale (Plan 342's G2 gate workload) so the perf numbers transfer.
const DIM: usize = 8;

/// Trajectory length (number of latent states). 100 steps = a moderate SWE
/// attempt length (~100 forward passes worth of patch-proposal evolution).
const N_STEPS: usize = 100;

// ─── Deterministic LCG (matches bench_342 convention) ──────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    }
}

// ─── Synthetic failure-mode trajectory builders ────────────────────────────
//
// Each builder constructs a `Vec<Vec<f32>>` of length `N_STEPS + 1`, each inner
// Vec of length `DIM`. The seeds are fixed so the PoC is reproducible.

/// **Committed-wrong**: monotone drift toward a wrong attractor.
/// Each step adds a fixed-direction displacement, so the trajectory is a
/// near-straight line in latent space. Predicted: low curvature, high length.
fn build_committed_wrong(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let mut state: Vec<f32> = (0..DIM).map(|_| rng.next_f32() * 0.1).collect();
    let direction: Vec<f32> = {
        let mut d: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        let norm = d.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in d.iter_mut() {
            *x /= norm;
        }
        d
    };
    let step_size = 0.15; // large enough that length accumulates fast
    let mut traj = Vec::with_capacity(N_STEPS + 1);
    traj.push(state.clone());
    for _ in 0..N_STEPS {
        for j in 0..DIM {
            state[j] += step_size * direction[j];
        }
        traj.push(state.clone());
    }
    traj.shrink_to_fit();
    traj
}

/// **Oscillation**: ping-pong between two wrong patches (attractors A and B).
/// Predicted: very high curvature (~π), high length. This is the canonical
/// "model can't commit" signature from Plan 342's Phase 3 gate.
fn build_oscillation(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let attractor_a: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    let attractor_b: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    let mut traj = Vec::with_capacity(N_STEPS + 1);
    for i in 0..=N_STEPS {
        // Alternate between A and B every step (period 2).
        let target = if i % 2 == 0 { &attractor_a } else { &attractor_b };
        traj.push(target.clone());
    }
    traj.shrink_to_fit();
    traj
}

/// **Drift**: rotating through wrong answers. Each step rotates the displacement
/// vector by a fixed angle, producing a circular trajectory. Predicted: mid
/// curvature (~π/2), high length.
fn build_drift(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let mut state: Vec<f32> = (0..DIM).map(|_| rng.next_f32() * 0.1).collect();
    // Two orthonormal directions spanning the rotation plane.
    let mut u: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    let norm_u = u.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in u.iter_mut() {
        *x /= norm_u;
    }
    let mut v: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    // Gram-Schmidt: v = v - (v·u)u, then normalize.
    let dot = v.iter().zip(u.iter()).map(|(a, b)| a * b).sum::<f32>();
    for j in 0..DIM {
        v[j] -= dot * u[j];
    }
    let norm_v = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in v.iter_mut() {
        *x /= norm_v;
    }
    let radius = 0.3;
    let omega = 0.4; // radians per step
    let mut traj = Vec::with_capacity(N_STEPS + 1);
    traj.push(state.clone());
    let mut theta = 0.0_f32;
    for _ in 0..N_STEPS {
        theta += omega;
        for j in 0..DIM {
            // Rotate the projection onto (u, v) — keeps other components fixed.
            let proj_u = state[j] + radius * (omega * u[j]); // baseline drift
            let _ = proj_u;
            state[j] += radius * (theta.cos() * u[j] + theta.sin() * v[j]) * omega;
        }
        traj.push(state.clone());
    }
    traj.shrink_to_fit();
    traj
}

/// **Stuck**: frozen at one point (degenerate trajectory). Predicted: ~0 length.
fn build_stuck(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let fixed: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    let mut traj = Vec::with_capacity(N_STEPS + 1);
    for _ in 0..=N_STEPS {
        traj.push(fixed.clone());
    }
    traj.shrink_to_fit();
    traj
}

/// **Converged-correct**: monotone drift toward a "correct" attractor, but with
/// smaller step size + deceleration (the model is converging, not just running).
/// Predicted: low curvature, moderate length (less than committed-wrong).
fn build_converged_correct(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let target: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    let mut state: Vec<f32> = (0..DIM).map(|_| rng.next_f32() * 0.1).collect();
    let mut traj = Vec::with_capacity(N_STEPS + 1);
    traj.push(state.clone());
    for _ in 0..N_STEPS {
        // Move 5% of the way to target each step (exponential convergence).
        for j in 0..DIM {
            state[j] += 0.05 * (target[j] - state[j]);
        }
        traj.push(state.clone());
    }
    traj.shrink_to_fit();
    traj
}

fn build_refs(traj: &[Vec<f32>]) -> Vec<&[f32]> {
    traj.iter().map(|v| v.as_slice()).collect()
}

// ─── T5.1: trajectory geometry discriminates failure modes ─────────────────

struct FailureModeResult {
    name: &'static str,
    geom: LatentTrajectoryGeometry,
}

fn run_t51() -> (Vec<FailureModeResult>, bool) {
    let modes: Vec<(&'static str, Vec<Vec<f32>>)> = vec![
        ("committed_wrong", build_committed_wrong(42)),
        ("oscillation", build_oscillation(137)),
        ("drift", build_drift(256)),
        ("stuck", build_stuck(314)),
        ("converged_correct", build_converged_correct(577)),
    ];

    let results: Vec<FailureModeResult> = modes
        .into_iter()
        .map(|(name, traj)| {
            let refs = build_refs(&traj);
            let geom = from_states(&refs);
            FailureModeResult { name, geom }
        })
        .collect();

    // Verdict: at least 3 distinct failure modes produce (curvature, length)
    // pairs that differ by > 0.5 rad on curvature OR > 20% on length.
    let mut distinct_pairs = 0;
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let d_curv = (results[i].geom.mean_curvature - results[j].geom.mean_curvature).abs();
            let max_len = results[i].geom.length.max(results[j].geom.length).max(1e-6);
            let rel_len_diff = (results[i].geom.length - results[j].geom.length).abs() / max_len;
            if d_curv > 0.5 || rel_len_diff > 0.20 {
                distinct_pairs += 1;
            }
        }
    }
    // 3 distinct modes = at least 3 distinguishable pairs among the 5.
    // The 10 pairwise comparisons among 5 modes; we require ≥ 3 to be distinct.
    let pass = distinct_pairs >= 3;
    (results, pass)
}

// ─── T5.2: CUCG fires on synthetic test-pass events ────────────────────────

/// A synthetic feature stream simulating an SWE attempt:
///   - steps 0..20: warmup churn (low coherence, high novelty — agent exploring)
///   - step 20: test-pass event (high coherence, low rank, positive divergence,
///     low novelty — closed unit, summarizable, progress, not stuck)
///   - steps 21..40: post-pass churn again
///   - step 40: second test-pass event
///   - steps 41..60: churn
///
/// The CUCG should fire `Compress` AT steps 20 and 40, and NOT fire elsewhere.
fn run_t52() -> (Vec<(usize, CompactionDecision<4>)>, bool) {
    let rubric = SearchRubric::default();
    let gate = ClosedUnitCompactionGate::builder(rubric)
        .fire_rule(FireRule::search_rule_4())
        .backstop(Backstop::None)
        .build();

    let mut scratch = RubricScratch::with_capacity(8, 2);
    let mut decisions: Vec<(usize, CompactionDecision<4>)> = Vec::with_capacity(61);

    for step in 0..=60usize {
        // Build the (coherence, intrinsic_rank, divergence_since_last, novelty_rate)
        // feature tuple for this step.
        let (coherence, rank, div, novelty) = match step {
            20 | 40 => (0.85, 4.0, 1.5, 0.2), // test-pass event
            0..=19 => (0.35, 16.0, 0.1, 4.0), // warmup churn
            _ => (0.40, 14.0, 0.2, 3.5),      // post-pass churn
        };

        scratch.clear();
        scratch.f32_buf.extend_from_slice(&[coherence, rank, div, novelty]);
        scratch.usize_buf.push(step); // span_end

        // trajectory_prefix is just the step counter encoded as bytes — the
        // rubric reads features from scratch, not from the bytes.
        let prefix = (step as u64).to_le_bytes();
        let decision = gate.evaluate(&prefix, step, 10_000, None, &mut scratch);
        decisions.push((step, decision));
    }

    // Verdict: Compress fires at steps 20 AND 40, and does NOT fire at any
    // other step.
    let fired_at = |step: usize| {
        decisions
            .iter()
            .find(|(s, _)| *s == step).is_some_and(|(_, d)| matches!(d, CompactionDecision::Compress { .. }))
    };
    let pass = fired_at(20)
        && fired_at(40)
        && decisions.iter().all(|(s, d)| {
            // Every non-event step must NOT be Compress.
            if *s == 20 || *s == 40 {
                true
            } else {
                !matches!(d, CompactionDecision::Compress { .. })
            }
        });
    (decisions, pass)
}

// ─── T5.3: committed_field_blend from all-fail trajectory summary ──────────

/// A frozen-test archetype field used for T5.3. Implements the trait with a
/// simple linear dynamics + a stable BLAKE3 commitment derived from its scale.
struct LinearField {
    scale: f32,
    commitment: [u8; 32],
}

impl LinearField {
    fn new(scale: f32, id: u8) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"LinearField_p011");
        hasher.update(&[id]);
        hasher.update(&scale.to_le_bytes());
        Self {
            scale,
            commitment: *hasher.finalize().as_bytes(),
        }
    }
}

impl ArchetypeFieldSource<32> for LinearField {
    fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        for j in 0..32 {
            dz_scratch[j] = self.scale * z[j];
        }
        &mut dz_scratch[..32]
    }
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

/// Run T5.3: commit a blend from an all-fail trajectory summary and verify
/// determinism + non-degeneracy.
///
/// We test TWO summary strategies because the choice of summary is the
/// load-bearing design decision for Layer 4:
///
/// - **Mean summary** (naive): average of all latent states. For an
///   oscillation trajectory that ping-pongs between two attractors A and B,
///   the mean lands near `(A+B)/2` — which may be near zero if A ≈ −B,
///   producing near-zero dot products → sigmoid ≈ 0.5 → no dominant archetype.
///   This is the honest failure mode: averaging washes out failure signal.
///
/// - **Endpoint summary** (canonical FAME input): the trajectory's final
///   latent state. This is what `committed_blend_01_three_archetypes` uses —
///   the personality snapshot is taken at a specific moment, not averaged.
///   For an oscillation trajectory, the endpoint is one of the two attractors,
///   which has a definite direction → non-degenerate blend.
///
/// The verdict table reports both. The PoC passes if the endpoint strategy
/// produces a deterministic + non-degenerate blend. The mean-strategy failure
/// is documented as a design constraint, not a primitive failure.
fn run_t53() -> (T53Result, T53Result, bool) {
    let osc_traj = build_oscillation(137);

    // Mean summary.
    let mut mean_summary = vec![0.0_f32; 32];
    for state in &osc_traj {
        for (j, &x) in state.iter().take(32).enumerate() {
            mean_summary[j] += x;
        }
    }
    let n = osc_traj.len() as f32;
    for x in mean_summary.iter_mut() {
        *x /= n;
    }
    while mean_summary.len() < 32 {
        mean_summary.push(0.0);
    }

    // Endpoint summary (the last state of the trajectory).
    let last_state = osc_traj.last().expect("oscillation traj non-empty");
    let mut endpoint_summary = vec![0.0_f32; 32];
    for (j, &x) in last_state.iter().take(32).enumerate() {
        endpoint_summary[j] = x;
    }

    let mean_result = commit_and_probe(&mean_summary, "mean");
    let endpoint_result = commit_and_probe(&endpoint_summary, "endpoint");

    // Verdict: the endpoint strategy must be deterministic + non-degenerate.
    // The mean strategy's failure is informational (design constraint).
    let pass = endpoint_result.deterministic && endpoint_result.non_degenerate;
    (mean_result, endpoint_result, pass)
}

#[derive(Clone)]
struct T53Result {
    strategy: &'static str,
    hash1: [u8; 32],
    hash2: [u8; 32],
    gates: [f32; 3],
    deterministic: bool,
    non_degenerate: bool,
}

fn commit_and_probe(summary: &[f32], strategy: &'static str) -> T53Result {
    // Three archetype fields + three direction vectors (deterministic).
    let f0 = LinearField::new(0.7, 0);
    let f1 = LinearField::new(-0.3, 1);
    let f2 = LinearField::new(0.5, 2);
    let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

    let mut dirs_arr: [[f32; 32]; 3] = [[0.0; 32]; 3];
    {
        let mut rng = Lcg::new(0x0113);
        for v in dirs_arr.iter_mut() {
            let mut row = [0.0_f32; 32];
            for x in row.iter_mut() {
                *x = rng.next_f32();
            }
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in row.iter_mut() {
                *x /= norm;
            }
            *v = row;
        }
    }

    let mut blend = TriArchetypeBlend::uncommitted();
    let hash1 = blend.commit(summary, &dirs_arr, &fields, 1);

    // Re-commit on the same summary — must produce bit-identical hash.
    let mut blend2 = TriArchetypeBlend::uncommitted();
    let hash2 = blend2.commit(summary, &dirs_arr, &fields, 1);

    let pi = blend.pi;
    let tau = TriArchetypeBlend::DEFAULT_TAU;
    let gates = [sigmoid(pi[0] / tau), sigmoid(pi[1] / tau), sigmoid(pi[2] / tau)];

    let deterministic = hash1 == hash2;
    let non_degenerate = gates.iter().any(|&g| g > 0.6);

    T53Result {
        strategy,
        hash1,
        hash2,
        gates,
        deterministic,
        non_degenerate,
    }
}

// ─── T5.3b: data-derived directions fix the concentration-of-measure failure ─
//
// Issue 569's T5.3 found that RANDOM direction vectors produce degenerate gates
// (concentration of measure). The documented design constraint was: "real freezer
// needs data-derived directions." This sub-PoC tests that constraint.
//
// Pre-implementation analysis revealed a SECOND compounding cause: the synthetic
// trajectories use random targets per seed, so endpoint POSITIONS do not cluster
// by failure mode — the discriminative signal is in the GEOMETRY, not position.
//
// Dual-strategy test isolates the two causes:
// - Strategy A (geometry summary): encode geometry features → clusters by mode.
// - Strategy B (endpoint summary): raw endpoint position → does NOT cluster.
// The contrast documents the design constraint for T5.5.

/// The three failure modes used as archetypes for T5.3b.
const T53B_MODES: [&str; 3] = ["oscillation", "committed-wrong", "converged-correct"];

/// Number of trajectories per mode (train + test split).
const T53B_TRAJ_PER_MODE: usize = 5;

/// Train split: first N seeds used to derive directions.
const T53B_TRAIN_SEEDS: usize = 3;

/// Encode trajectory geometry into a 32-dim summary.
///
/// Places 4 normalized geometry features in dims [0..4], replicated into
/// dims [4..8] for dot-product stability (avoids sparse-vector edge effects).
/// The remaining 24 dims are zero — the effective dimensionality is 8, which
/// is low enough that concentration-of-measure does not apply to data-derived
/// directions.
fn geometry_summary(traj: &[Vec<f32>]) -> [f32; 32] {
    let refs = build_refs(traj);
    let geom = from_states(&refs);
    // Normalize features to roughly [-1, 1] for stable dot products.
    let length_norm = (geom.length / 20.0_f32).min(1.0);
    let curvature_norm = (geom.mean_curvature / core::f32::consts::PI).min(1.0);
    let cosine_norm = geom.min_adjacent_cosine; // already in [-1, 1]
    let n_steps_norm = (geom.n_steps as f32) / (N_STEPS as f32);

    let mut summary = [0.0_f32; 32];
    // Replicate into two 4-dim blocks for dot-product dynamics.
    for block in summary.chunks_mut(4).take(2) {
        block[0] = length_norm;
        block[1] = curvature_norm;
        block[2] = cosine_norm;
        block[3] = n_steps_norm;
    }
    summary
}

/// Encode trajectory endpoint position into a 32-dim summary (padded with zeros).
/// Strategy B: uses raw latent position, NOT geometry.
fn endpoint_summary(traj: &[Vec<f32>]) -> [f32; 32] {
    let last = traj.last().expect("trajectory non-empty");
    let mut summary = [0.0_f32; 32];
    for (j, &x) in last.iter().take(DIM.min(32)).enumerate() {
        summary[j] = x;
    }
    summary
}

/// Derive archetype direction vectors from cluster centroids of train summaries.
///
/// direction_k = normalize(centroid_k - global_centroid)
///
/// This is the nearest-centroid classifier direction: the vector from the global
/// mean toward cluster k's center. When probed with a summary from cluster k,
/// dot(summary_k, direction_k) is large and positive; for other clusters, small.
fn derive_directions(train_summaries: &[[[f32; 32]; 3]]) -> [[f32; 32]; 3] {
    let n_modes = train_summaries.len();
    let n_per_mode = train_summaries[0].len();

    // Compute per-mode centroids.
    let mut centroids = [[0.0_f32; 32]; 3];
    for (mode_idx, mode_sums) in train_summaries.iter().enumerate() {
        for s in mode_sums.iter() {
            for j in 0..32 {
                centroids[mode_idx][j] += s[j];
            }
        }
        let inv = 1.0 / (n_per_mode as f32);
        for j in 0..32 {
            centroids[mode_idx][j] *= inv;
        }
    }

    // Compute global centroid.
    let mut global = [0.0_f32; 32];
    for c in &centroids {
        for j in 0..32 {
            global[j] += c[j];
        }
    }
    let inv_modes = 1.0 / (n_modes as f32);
    for j in 0..32 {
        global[j] *= inv_modes;
    }

    // Direction_k = normalize(centroid_k - global).
    let mut dirs = [[0.0_f32; 32]; 3];
    for k in 0..n_modes {
        let mut norm_sq = 0.0_f32;
        for j in 0..32 {
            dirs[k][j] = centroids[k][j] - global[j];
            norm_sq += dirs[k][j] * dirs[k][j];
        }
        let norm = norm_sq.sqrt().max(1e-9);
        for j in 0..32 {
            dirs[k][j] /= norm;
        }
    }
    dirs
}

/// Build a trajectory for a given failure mode + seed.
fn build_trajectory_for_mode(mode_idx: usize, seed: u64) -> Vec<Vec<f32>> {
    match mode_idx {
        0 => build_oscillation(seed),
        1 => build_committed_wrong(seed),
        2 => build_converged_correct(seed),
        _ => unreachable!(),
    }
}

#[derive(Clone)]
#[allow(dead_code)] // diagnostic fields read during debugging
struct T53bProbeResult {
    mode_name: &'static str,
    strategy: &'static str,
    gates: [f32; 3],
    matching_gate: f32,
    argmax_k: usize,
    correct: bool,
}

struct T53bStrategyResult {
    strategy: &'static str,
    probes: Vec<T53bProbeResult>,
    accuracy: f32,
    pass: bool,
}

/// Run T5.3b: data-derived direction PoC with dual strategies.
fn run_t53b() -> (T53bStrategyResult, T53bStrategyResult) {
    // Build all trajectories: [mode][seed].
    let mut all_trajs: Vec<Vec<Vec<Vec<f32>>>> = Vec::with_capacity(3);
    for mode_idx in 0..3 {
        let mut mode_trajs = Vec::with_capacity(T53B_TRAJ_PER_MODE);
        for seed in 0..T53B_TRAJ_PER_MODE {
            mode_trajs.push(build_trajectory_for_mode(mode_idx, seed as u64 * 100 + 7));
        }
        all_trajs.push(mode_trajs);
    }

    // Compute summaries for both strategies, split into train/test.
    let run_strategy = |summary_fn: fn(&[Vec<f32>]) -> [f32; 32],
                        strategy_name: &'static str|
     -> T53bStrategyResult {
        // Train: [mode][train_seed] summaries.
        let mut train: Vec<Vec<[f32; 32]>> = Vec::with_capacity(3);
        for mode_idx in 0..3 {
            let mut mode_sums: Vec<[f32; 32]> = Vec::with_capacity(T53B_TRAIN_SEEDS);
            for seed in 0..T53B_TRAIN_SEEDS {
                mode_sums.push(summary_fn(&all_trajs[mode_idx][seed]));
            }
            train.push(mode_sums);
        }

        // Derive directions from centroids.
        let train_arr: [[[f32; 32]; 3]; 3] = [
            [train[0][0], train[0][1], train[0][2]],
            [train[1][0], train[1][1], train[1][2]],
            [train[2][0], train[2][1], train[2][2]],
        ];
        let dirs = derive_directions(&train_arr);

        // Archetype fields (same as T5.3 — scale-only differences).
        let f0 = LinearField::new(0.7, 0);
        let f1 = LinearField::new(-0.3, 1);
        let f2 = LinearField::new(0.5, 2);
        let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

        // Probe with test seeds.
        let n_test = T53B_TRAJ_PER_MODE - T53B_TRAIN_SEEDS;
        let mut probes: Vec<T53bProbeResult> = Vec::with_capacity(3 * n_test);
        let mut n_correct = 0usize;
        let total = 3 * n_test;

        for mode_idx in 0..3 {
            for seed in T53B_TRAIN_SEEDS..T53B_TRAJ_PER_MODE {
                let summary = summary_fn(&all_trajs[mode_idx][seed]);
                let mut blend = TriArchetypeBlend::uncommitted();
                blend.commit(&summary, &dirs, &fields, 1);
                let tau = TriArchetypeBlend::DEFAULT_TAU;
                let gates = [
                    sigmoid(blend.pi[0] / tau),
                    sigmoid(blend.pi[1] / tau),
                    sigmoid(blend.pi[2] / tau),
                ];
                let matching_gate = gates[mode_idx];
                let argmax_k = gates
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        katgpt_core::float_order::cmp_for_max(**a, **b)
                    }).map_or(0, |(k, _)| k);
                let correct = matching_gate > 0.6 && argmax_k == mode_idx;
                if correct {
                    n_correct += 1;
                }
                probes.push(T53bProbeResult {
                    mode_name: T53B_MODES[mode_idx],
                    strategy: strategy_name,
                    gates,
                    matching_gate,
                    argmax_k,
                    correct,
                });
            }
        }

        let accuracy = n_correct as f32 / total as f32;
        let pass = accuracy >= 0.8;
        T53bStrategyResult {
            strategy: strategy_name,
            probes,
            accuracy,
            pass,
        }
    };

    let geom_result = run_strategy(geometry_summary, "geometry");
    let endpoint_result = run_strategy(endpoint_summary, "endpoint");
    (geom_result, endpoint_result)
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Proposal 011 Phase 5 — SWE Trajectory Geometry Synthetic PoC      ║");
    println!("║  Issue 569 + 570 — defend-or-refute Layer 4 (modelless reframe)   ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Config: DIM={DIM}, N_STEPS={N_STEPS}, seeds fixed (reproducible).");
    println!();

    // ── T5.1 ────────────────────────────────────────────────────────────
    println!("── T5.1: trajectory geometry discriminates failure modes ──────");
    let (results, t51_pass) = run_t51();
    println!(
        "  {:>20}  {:>10}  {:>14}  {:>14}  {:>10}",
        "failure_mode", "n_steps", "length", "mean_curv", "min_cos"
    );
    println!("  {}", "-".repeat(76));
    for r in &results {
        println!(
            "  {:>20}  {:>10}  {:>14.4}  {:>14.4}  {:>10.4}",
            r.name, r.geom.n_steps, r.geom.length, r.geom.mean_curvature, r.geom.min_adjacent_cosine
        );
    }
    println!();
    println!(
        "  T5.1 verdict: {} — ≥3 distinct (curvature, length) pairs.",
        if t51_pass { "PASS" } else { "FAIL" }
    );
    println!();

    // ── T5.2 ────────────────────────────────────────────────────────────
    println!("── T5.2: CUCG fires on synthetic test-pass events ──────────────");
    let (decisions, t52_pass) = run_t52();
    let mut compress_steps: Vec<usize> = Vec::new();
    for (step, d) in &decisions {
        if matches!(d, CompactionDecision::Compress { .. }) {
            compress_steps.push(*step);
        }
    }
    println!("  Compress fired at steps: {compress_steps:?}");
    println!("  (expected: [20, 40] — synthetic test-pass events)");
    println!();
    println!(
        "  T5.2 verdict: {} — fires exactly at the test-pass events.",
        if t52_pass { "PASS" } else { "FAIL" }
    );
    println!();

    // ── T5.3 ────────────────────────────────────────────────────────
    println!("── T5.3: committed_field_blend from all-fail summary ──────────");
    let (mean_r, endpoint_r, t53_pass) = run_t53();
    for r in [&mean_r, &endpoint_r] {
        println!("  strategy: {}", r.strategy);
        println!("    hash1 (first 8 bytes): {:02x?}", &r.hash1[..8]);
        println!("    hash2 (first 8 bytes): {:02x?}", &r.hash2[..8]);
        println!("    deterministic: {}", r.deterministic);
        println!("    pi-derived sigmoid gates (per archetype):");
        for (k, g) in r.gates.iter().enumerate() {
            println!("      archetype {k}: sigmoid(pi_{k}/tau) = {g:.4}");
        }
        let max_gate = r.gates.iter().cloned().fold(0.0_f32, f32::max);
        println!("    max gate = {max_gate:.4} (non-degenerate threshold: > 0.6)");
        println!("    non_degenerate: {}", r.non_degenerate);
        println!();
    }
    println!(
        "  T5.3 verdict: {} — endpoint summary produces deterministic + non-degenerate",
        if t53_pass { "PASS" } else { "FAIL" }
    );
    println!("  blend from failure signal. (mean summary may fail — design constraint.)");
    println!();

    // ── T5.3b ───────────────────────────────────────────────────────────
    println!("── T5.3b: data-derived directions fix concentration-of-measure ──");
    println!("  (Issue 570: tests the design constraint from T5.3's failure)");
    let (t53b_geom, t53b_endpoint) = run_t53b();
    for sr in [&t53b_geom, &t53b_endpoint] {
        println!("  strategy: {}", sr.strategy);
        println!(
            "  {:>22}  {:>10}  {:>10}  {:>10}  {:>8}",
            "mode", "gate_0", "gate_1", "gate_2", "correct"
        );
        println!("  {}", "-".repeat(68));
        for p in &sr.probes {
            let gate_str = format!(
                "{:>10.4}  {:>10.4}  {:>10.4}",
                p.gates[0], p.gates[1], p.gates[2]
            );
            let match_info = format!("  match={:.4} argmax={}", p.matching_gate, p.argmax_k);
            println!(
                "  {:>22}  {}{}  {:>8}",
                p.mode_name, gate_str, match_info,
                if p.correct { "YES" } else { "no" }
            );
        }
        println!("  accuracy: {:.0}% (threshold ≥80%)", sr.accuracy * 100.0);
        println!("  verdict: {}", if sr.pass { "PASS" } else { "FAIL" });
        println!();
    }
    let t53b_pass = t53b_geom.pass;
    if t53b_geom.pass && !t53b_endpoint.pass {
        println!("  Contrast confirmed: geometry summary PASS + endpoint summary FAIL.");
        println!("  Design constraint: SweTrajectoryFreezer summary encoder MUST capture");
        println!("  failure-mode-discriminative geometry, not just raw latent position.");
    } else if t53b_geom.pass && t53b_endpoint.pass {
        println!("  Unexpected: both strategies pass — positional summaries DO cluster.");
    } else {
        println!("  Both strategies fail — data-derived directions insufficient.");
    }
    println!();

    // ── Overall ─────────────────────────────────────────────────────────
    let gates_pass = [
        ("T5.1", "geometry discriminates failure modes", t51_pass),
        ("T5.2", "CUCG fires on test-pass events", t52_pass),
        ("T5.3", "random-direction blend (baseline)", t53_pass),
        ("T5.3b", "data-derived directions (geometry)", t53b_pass),
    ];
    println!("──────────────────────────────────────────────────────────────────");
    println!("┌───────┬──────────────────────────────────────────────────┬────────┐");
    println!("│ Sub   │ Claim                                             │ Verdict│");
    println!("├───────┼──────────────────────────────────────────────────┼────────┤");
    for (gate, claim, pass) in &gates_pass {
        let verdict = if *pass { "✅ PASS" } else { "❌ FAIL" };
        println!("│ {gate:5} │ {claim:<50} │ {verdict} │");
    }
    println!("└───────┴──────────────────────────────────────────────────┴────────┘");
    println!();

    // T5.3b supersedes T5.3 on the non-degeneracy axis: if data-derived
    // geometry directions produce non-degenerate gates, the modelless FAME
    // path is validated (T5.3's random-direction failure was a configuration
    // issue, not a primitive insufficiency).
    let layer4_validated = t51_pass && t52_pass && t53b_pass;
    if layer4_validated {
        println!("═ LAYER 4 VALIDATED — modelless reframe holds on synthetic data ═");
        println!();
        println!("T5.1 (geometry discriminates) + T5.2 (CUCG fires on test-pass) +");
        println!("T5.3b (data-derived directions produce non-degenerate FAME blend)");
        println!("all PASS. T5.3's random-direction failure was a parameterization issue,");
        println!("not a primitive insufficiency — fixed by data-derived directions.");
        println!();
        println!("Design constraint for T5.5 (SweTrajectoryFreezer): summary encoder");
        println!("MUST capture failure-mode-discriminative geometry (not just position).");
        println!();
        println!("Next step: T5.4 (real Kimi-K3 trajectories, gated on P032 Phase 5)");
        println!("or T5.5 (SweTrajectoryFreezer impl with geometry-summary encoder).");
    } else {
        let failed: Vec<&str> = gates_pass
            .iter()
            .filter(|(g, _, p)| !p && *g != "T5.3")
            .map(|(g, _, _)| *g)
            .collect();
        if failed.is_empty() {
            println!("═ PARTIAL — only T5.3 (random-direction baseline) failed ═");
            println!("This is expected: T5.3 was the control. T5.3b supersedes it.");
        } else {
            println!("═ PARTIAL — {} failed: {} ═", failed.len(), failed.join(", "));
            println!();
            for (g, claim, _) in &gates_pass {
                if failed.contains(g) {
                    println!("  {g}: {claim}");
                }
            }
            println!();
            println!("Document the failures + decide whether to refine or defer.");
        }
    }
}
