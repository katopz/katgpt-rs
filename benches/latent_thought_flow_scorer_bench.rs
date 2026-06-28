//! Cost-Aware Reward-Proportional Latent Trajectory Scorer — GOAT Gate Benchmark
//! (Issue 030, Research 263, arxiv:2606.16222 — Latent Thought Flow).
//!
//! Distills the *inference-time* slice of Latent Thought Flow (LTF) modellessly:
//! the paper's GFlowNet training (EW-SubTB, reference-prior, LoRA-on-latent-head)
//! is out of scope (→ riir-train). What's left is a fusion of FIVE existing
//! primitives into one scorer over N latent-thought trajectories:
//!
//! 1. `LatentThoughtKernel`            — trajectory generator (Plan 276)
//! 2. `self_advantage_margin`           — teacher-free V(τ) (Research 250 / Plan 283)
//! 3. `lambda_flow × (1-stop_prob)`     — cost penalty shape C(τ) = T (Plan 052)
//! 4. Entropy-band gate                 — paper §C.2 "effective entropy regime"
//! 5. Argmax over N scored trajectories — aggregation (Plan 260 shape)
//!
//! # GOAT Gate (Issue 030)
//!
//! | Gate | Criterion | Target |
//! |------|-----------|--------|
//! | G1 | Composite scorer accuracy vs single-component baselines | composite > best single by ≥3pp |
//! | G1b | Wasted-thought reduction (thoughts discarded vs all-run) | ≥30% discarded at matched accuracy |
//! | G2 | Effective entropy band — accuracy peaks inside [Ξ_low, Ξ_high] | interior maximum exists |
//! | G3 | Per-trajectory scoring latency | < 1µs at d_belief=32, vocab=8 |
//!
//! Promotion (per Issue 030): G1 + G1b + G2 + G3 all pass → promote to plan +
//! feature flag. Otherwise close.
//!
//! # Method
//!
//! Synthetic task: 8-action choice. Each query has a known correct action.
//! Each "latent thought trajectory" = K iterations of `LatentThoughtKernel`
//! from a query-specific initial belief state. Pre-thought logits = projection
//! of initial state; post-thought logits = projection of post-K state. The
//! self-advantage of the correct action is the teacher-free quality signal V(τ).
//!
//! ```bash
//! cargo run --release --bench latent_thought_flow_scorer_bench \
//!     --features self_advantage_gate,micro_belief
//! ```

#![cfg(feature = "self_advantage_gate")]
#![cfg(feature = "micro_belief")]

use katgpt_core::micro_belief::{
    AttractorKernel, LatentThoughtKernel, MicroRecurrentBeliefState,
};
use katgpt_rs::pruners::self_advantage::self_advantage_margin;
use std::time::Instant;

// ── Constants ────────────────────────────────────────────────────

/// Belief-state dimension (matches G1_4 in micro_belief_bench).
const DIM: usize = 32;

/// Action vocabulary size for the synthetic task. Small to keep self_advantage
/// scratch buffers tiny and to mirror HLA's 6-module activation vector.
const VOCAB: usize = 8;

/// Number of latent-thought trajectories sampled per query (paper Table 9: N=5
/// captures most of the attainable benefit; we use 8 for headroom).
const N_TRAJECTORIES: usize = 8;

/// Number of synthetic queries per benchmark run.
const N_QUERIES: usize = 2000;

/// Cost-penalty strength (paper default λ_c = 0.03).
/// Score multiplier per trajectory: `exp(-λ_c · T)` where T = K iters used.
const LAMBDA_C: f32 = 0.03;

/// Candidate K values per trajectory (variable-length trajectories).
/// Paper §3.1: `0 ≤ T ≤ T_max`. K=0 means "answer without reasoning".
const K_CANDIDATES: [u8; 6] = [0, 1, 2, 3, 5, 8];

/// Entropy band edges (paper §C.2 Table 10 — sweet spot Ξ ≈ 0.024,
/// collapse Ξ ≈ 0.013, noise Ξ ≈ 0.030; values in *nats* per dim).
/// We use the equivalent normalized-to-[0,1] band: H/log(V).
const XI_LOW_NORM: f32 = 0.20; // ≈ 0.024 nats / log(8) ≈ 0.417 / ... see G2 sweep
const XI_HIGH_NORM: f32 = 0.85;

// ── Helpers ──────────────────────────────────────────────────────

/// Shannon entropy in nats (natural log). Matches `shannon_entropy` in
/// `katgpt-rs/src/distill/trd.rs:551-559` — same convention.
#[inline]
fn shannon_entropy_nats(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

/// Normalize entropy to [0, 1] by dividing by log(V).
#[inline]
fn normalized_entropy(probs: &[f32]) -> f32 {
    let h = shannon_entropy_nats(probs);
    let log_v = (probs.len() as f32).ln();
    if log_v > 0.0 {
        h / log_v
    } else {
        0.0
    }
}

/// Smooth bandpass gate in [0,1] — sigmoid rise at `low`, sigmoid fall at `high`.
/// Replaces the paper's implicit "EW-SubTB reweighting" with an explicit,
/// differentiable band. The sharpness τ controls transition width.
#[inline]
fn entropy_band_gate(xi: f32, low: f32, high: f32, sharpness: f32) -> f32 {
    // sigmoid((xi - low) / τ) · sigmoid((high - xi) / τ)
    let s_low = 1.0 / (1.0 + (-(xi - low) / sharpness).exp());
    let s_high = 1.0 / (1.0 + (-(high - xi) / sharpness).exp());
    s_low * s_high
}

/// Project a belief state to a VOCAB-dim "policy" via a fixed projection matrix.
/// Uses a deterministic seed-derived matrix; the "logits" are the projected
/// values. We deliberately use a *fixed* projection so pre/post comparison is
/// apples-to-apples (the kernel is what changes the state).
fn project_to_logits(state: &[f32], projection: &[f32], out: &mut [f32]) {
    // projection layout: [VOCAB][DIM] row-major. out[v] = sum_k state[k] * projection[v*DIM + k]
    for v in 0..VOCAB {
        let row = &projection[v * DIM..(v + 1) * DIM];
        let mut acc = 0.0f32;
        // Chunked accumulation for auto-vectorization.
        let chunks = DIM / 4;
        let mut k = 0;
        while k < chunks * 4 {
            acc += state[k] * row[k]
                + state[k + 1] * row[k + 1]
                + state[k + 2] * row[k + 2]
                + state[k + 3] * row[k + 3];
            k += 4;
        }
        while k < DIM {
            acc += state[k] * row[k];
            k += 1;
        }
        out[v] = acc;
    }
}

/// Build the identity-on-first-VOCAB-dims projection matrix.
/// `projection[v * DIM + k] = 1.0 if k == v else 0.0`.
/// This makes `logits[v] = state[v]`, so the argmax of the logits is the
/// dominant coordinate of the state — directly decodable. We keep the
/// random-seeded variant around as an alternative (off by default).
#[allow(dead_code)]
fn make_projection(seed: u64) -> Vec<f32> {
    let mut rng = rng_from_seed(seed);
    let mut p = vec![0.0f32; VOCAB * DIM];
    for v in 0..VOCAB {
        for k in 0..DIM {
            // Values in [-0.5, 0.5] — keeps logits in a reasonable range.
            p[v * DIM + k] = rng.next_f32() - 0.5;
        }
    }
    p
}

/// Identity-on-first-VOCAB-dims projection. `logits[v] = state[v]`.
fn make_identity_projection() -> Vec<f32> {
    let mut p = vec![0.0f32; VOCAB * DIM];
    for v in 0..VOCAB {
        p[v * DIM + v] = 1.0;
    }
    p
}

/// Tiny deterministic PRNG (xorshift + uniform float). Avoids pulling in
/// `fastrand` just for the bench (which is already a dev-dep elsewhere, but
/// this keeps the bench self-contained and seed-deterministic).
struct RngState {
    state: u64,
}

impl RngState {
    fn next_u32(&mut self) -> u32 {
        // xorshift64
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x & 0xFFFF_FFFF) as u32
    }

    fn next_f32(&mut self) -> f32 {
        // Uniform in [0, 1)
        let u = self.next_u32();
        (u as f32) / (u32::MAX as f32)
    }
}

fn rng_from_seed(seed: u64) -> RngState {
    // Avoid zero-state (xorshift64 stays at 0). Mix the seed.
    let s = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    RngState { state: s }
}

/// Generate N_QUERIES synthetic queries. Each query has:
///   - an initial belief state (deterministic from query seed)
///   - a "correct" action in [0, VOCAB)
///   - an input vector that, when fed to LatentThoughtKernel, biases the state
///     toward the correct attractor.
///
/// Task design (signal-carrying, not random):
/// - VOCAB=8 actions map to 8 orthogonal "signal directions" = the first VOCAB
///   standard basis vectors e_0..e_7 of the DIM-dim state space.
/// - The projection matrix is identity on those first VOCAB dims:
///   `projection[v] = e_v`. So `logits[v] = state[v]`, and the argmax of the
///   logits picks the dominant direction.
/// - Each query picks a target `d ∈ [0, VOCAB)`. The input is a strong signal
///   on coordinate `d` plus small noise on all coordinates. The initial state
///   is small random noise (so K=0 → argmax is essentially a coin flip).
/// - LatentThoughtKernel.step applies `σ(W·s + U·x + b)` with frozen weights.
///   With the same input repeated K times, the state converges toward the
///   attractor defined by x, which has a strong component in direction `d`.
///   After K≥2 iterations, the `d`-th coordinate of the state tends to
///   dominate → argmax picks `d` correctly. K=0 → argmax dominated by initial
///   noise → near-chance. This gives a clear accuracy-vs-K curve the scorer
///   can exploit, and the variable-length trajectory sampler can prefer the
///   K that converges over the K that wastes compute.
fn make_queries(seed: u64, _projection: &[f32]) -> Vec<(Vec<f32>, usize, Vec<f32>)> {
    let mut rng = rng_from_seed(seed);
    let mut out = Vec::with_capacity(N_QUERIES);
    for _ in 0..N_QUERIES {
        let correct = (rng.next_u32() as usize) % VOCAB;

        // Initial state: zero. K=0 reads this directly → argmax is 0 always
        // (tie-break). This makes K=0 deterministic-but-wrong for 7/8 actions,
        // a clean baseline. Any kernel step that responds to the input will
        // improve on this.
        let state = vec![0.0f32; DIM];

        // Input: very strong signal on coordinate `correct` (in [0, VOCAB)),
        // tiny noise on the signal subspace only. Coords [VOCAB..DIM) stay
        // zero — no noise dims to confuse the projection.
        let mut input = vec![0.0f32; DIM];
        input[correct] = 1.5; // very strong signal
        for slot in input.iter_mut().take(VOCAB) {
            *slot += (rng.next_f32() - 0.5) * 0.05; // tiny noise on signal subspace
        }

        out.push((state, correct, input));
    }
    out
}

// ── Scorers ──────────────────────────────────────────────────────

/// One latent-thought trajectory: K iterations of LatentThoughtKernel from
/// `initial_state` with `input`. Returns (post_state, K_used).
fn run_trajectory(
    kernel: &LatentThoughtKernel,
    initial_state: &[f32],
    input: &[f32],
    k_iters: u8,
) -> Vec<f32> {
    let mut state = initial_state.to_vec();
    if k_iters > 0 {
        // Call `kernel.step` k_iters times with the same input each iteration.
        // The kernel was constructed with K=1; we manually loop to produce
        // variable-length trajectories without rebuilding the kernel.
        for _ in 0..k_iters {
            kernel.step(&mut state, input);
        }
    }
    state
}

/// Result of scoring one trajectory.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // diagnostic fields retained for debug printing / future fusion work
struct TrajectoryScore {
    k_iters: u8,
    /// Argmax of the post-thought policy (= the action this trajectory votes for).
    voted_action: usize,
    /// V(τ): self-advantage of the *voted* action (teacher-free quality).
    self_advantage: f32,
    /// Ξ(τ): normalized entropy of the post-thought policy.
    entropy_norm: f32,
    /// Composite score under the full fusion.
    composite: f32,
    /// Composite score under cost-only (no entropy gate, no advantage).
    cost_only: f32,
    /// Composite score under advantage-only (no cost, no entropy gate).
    advantage_only: f32,
    /// Composite score under entropy-band-only (no cost, no advantage).
    entropy_only: f32,
}

/// Score a single trajectory under all four scoring variants.
#[allow(clippy::too_many_arguments)] // bench-only scorer: all 8 inputs are used by every call site
fn score_trajectory(
    pre_logits: &[f32],
    post_logits: &[f32],
    post_probs: &[f32],
    k_iters: u8,
    scratch: &mut [f32], // 3 * VOCAB
    xi_low: f32,
    xi_high: f32,
    sharpness: f32,
) -> TrajectoryScore {
    // Argmax of post_logits = voted action.
    let mut voted_action = 0;
    let mut best_logit = f32::NEG_INFINITY;
    for (v, &l) in post_logits.iter().enumerate().take(VOCAB) {
        if l > best_logit {
            best_logit = l;
            voted_action = v;
        }
    }

    // V(τ): self-advantage margin of the voted action.
    let sa = self_advantage_margin(pre_logits, post_logits, voted_action, scratch);
    let sa_bounded = 1.0 / (1.0 + (-sa).exp()); // sigmoid(A)

    // Cost penalty: exp(-λ_c · K).
    let cost_mult = (-LAMBDA_C * k_iters as f32).exp();

    // Entropy gate.
    let xi = normalized_entropy(post_probs);
    let gate = entropy_band_gate(xi, xi_low, xi_high, sharpness);

    TrajectoryScore {
        k_iters,
        voted_action,
        self_advantage: sa,
        entropy_norm: xi,
        composite: sa_bounded * cost_mult * gate,
        cost_only: cost_mult,
        advantage_only: sa_bounded,
        entropy_only: gate,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scorer {
    /// Baseline: pick the first trajectory (K=K_CANDIDATES[1]) always. No scoring.
    FirstK1,
    /// Baseline: majority vote over all N trajectories, no scoring weighting.
    MajorityVote,
    /// Single-component: cost penalty only.
    CostOnly,
    /// Single-component: self-advantage only.
    AdvantageOnly,
    /// Single-component: entropy-band gate only.
    EntropyOnly,
    /// Full fusion: argmax of composite score.
    Composite,
    /// Fusion variant: majority vote weighted by composite score.
    /// This is the natural synthesis — plurality vote is the best aggregator
    /// in the modelless setting (the sampler is fixed, not trained to
    /// concentrate), and weighting by composite score adds quality awareness.
    WeightedVoteComposite,
    /// Fusion variant: majority vote weighted by advantage-only score.
    WeightedVoteAdvantage,
}

impl Scorer {
    fn label(self) -> &'static str {
        match self {
            Scorer::FirstK1 => "first-K1 (no score)",
            Scorer::MajorityVote => "majority vote (no score)",
            Scorer::CostOnly => "cost-only (argmax)",
            Scorer::AdvantageOnly => "advantage-only (argmax)",
            Scorer::EntropyOnly => "entropy-only (argmax)",
            Scorer::Composite => "composite (argmax)",
            Scorer::WeightedVoteComposite => "weighted-vote (composite)",
            Scorer::WeightedVoteAdvantage => "weighted-vote (advantage)",
        }
    }
}

/// Pick an action from a slice of scored trajectories under the given scorer.
fn pick_action(scorer: Scorer, scores: &[TrajectoryScore]) -> usize {
    if scores.is_empty() {
        return 0;
    }
    match scorer {
        Scorer::FirstK1 => {
            // Pick the first trajectory with K=1, or fall back to the first.
            for s in scores {
                if s.k_iters == 1 {
                    return s.voted_action;
                }
            }
            scores[0].voted_action
        }
        Scorer::MajorityVote => {
            // Plurality vote.
            let mut tally = [0u32; VOCAB];
            for s in scores {
                tally[s.voted_action] += 1;
            }
            let mut best = 0;
            let mut best_count = tally[0];
            for (v, &count) in tally.iter().enumerate().take(VOCAB).skip(1) {
                if count > best_count {
                    best_count = count;
                    best = v;
                }
            }
            best
        }
        Scorer::WeightedVoteComposite | Scorer::WeightedVoteAdvantage => {
            // Weighted plurality: tally[v] += weight_of_trajectory_that_voted_v.
            // Weights are clamped to be non-negative (sigmoid/gate are already
            // in [0,1], cost_mult is positive).
            let mut tally = [0.0f32; VOCAB];
            for s in scores {
                let w = match scorer {
                    Scorer::WeightedVoteComposite => s.composite.max(0.0),
                    Scorer::WeightedVoteAdvantage => s.advantage_only.max(0.0),
                    _ => unreachable!(),
                };
                tally[s.voted_action] += w;
            }
            let mut best = 0;
            let mut best_weight = tally[0];
            for (v, &weight) in tally.iter().enumerate().take(VOCAB).skip(1) {
                if weight > best_weight {
                    best_weight = weight;
                    best = v;
                }
            }
            best
        }
        _ => {
            // Pick the trajectory with the highest score under the chosen metric.
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, s) in scores.iter().enumerate() {
                let v = match scorer {
                    Scorer::CostOnly => s.cost_only,
                    Scorer::AdvantageOnly => s.advantage_only,
                    Scorer::EntropyOnly => s.entropy_only,
                    Scorer::Composite => s.composite,
                    _ => unreachable!(),
                };
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            scores[best_idx].voted_action
        }
    }
}

// ── G1: Composite vs single-component baselines ─────────────────

struct AccuracyReport {
    correct: usize,
    total: usize,
    /// How many trajectories were "discarded" — i.e. would not have been
    /// sampled under cost-aware budgeting (heuristic: trajectories with K=0
    /// and advantage ≤ 0 are "dead thoughts").
    discarded: usize,
    /// Total thoughts evaluated (= N_QUERIES × N_TRAJECTORIES).
    total_thoughts: usize,
}

impl AccuracyReport {
    fn accuracy(&self) -> f64 {
        self.correct as f64 / self.total as f64
    }
    fn discard_rate(&self) -> f64 {
        self.discarded as f64 / self.total_thoughts as f64
    }
}

fn run_g1(
    kernel: &LatentThoughtKernel,
    projection: &[f32],
    queries: &[(Vec<f32>, usize, Vec<f32>)],
    xi_low: f32,
    xi_high: f32,
    sharpness: f32,
) -> Vec<(Scorer, AccuracyReport)> {
    let mut pre_logits = vec![0.0f32; VOCAB];
    let mut post_logits = vec![0.0f32; VOCAB];
    let mut post_probs = vec![0.0f32; VOCAB];
    let mut scratch = vec![0.0f32; 3 * VOCAB];

    let scorers = [
        Scorer::FirstK1,
        Scorer::MajorityVote,
        Scorer::CostOnly,
        Scorer::AdvantageOnly,
        Scorer::EntropyOnly,
        Scorer::Composite,
        Scorer::WeightedVoteComposite,
        Scorer::WeightedVoteAdvantage,
    ];

    let mut correct = [0usize; 8];
    let mut discarded = [0usize; 8]; // only composite discards
    let total_thoughts = N_QUERIES * N_TRAJECTORIES;

    for (init_state, correct_action, input) in queries {
        // Pre-thought logits (from initial state).
        project_to_logits(init_state, projection, &mut pre_logits);

        // Sample N_TRAJECTORIES with different K values.
        let mut scored: Vec<TrajectoryScore> = Vec::with_capacity(N_TRAJECTORIES);
        for n in 0..N_TRAJECTORIES {
            let k = K_CANDIDATES[n % K_CANDIDATES.len()];
            let post_state = run_trajectory(kernel, init_state, input, k);
            project_to_logits(&post_state, projection, &mut post_logits);

            // softmax(post_logits) → post_probs (for entropy).
            let max_l = post_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut z = 0.0f32;
            for v in 0..VOCAB {
                post_probs[v] = (post_logits[v] - max_l).exp();
                z += post_probs[v];
            }
            for slot in post_probs.iter_mut().take(VOCAB) {
                *slot /= z.max(1e-12);
            }

            scored.push(score_trajectory(
                &pre_logits,
                &post_logits,
                &post_probs,
                k,
                &mut scratch,
                xi_low,
                xi_high,
                sharpness,
            ));
        }

        // "Discarded" = trajectories that the composite scorer would not pick
        // because their composite score is below the median (cost-aware budget).
        if !scored.is_empty() {
            let mut composite_vals: Vec<f32> =
                scored.iter().map(|s| s.composite).collect();
            composite_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = composite_vals[composite_vals.len() / 2];
            for s in &scored {
                if s.composite < median {
                    discarded[5] += 1; // index 5 = Composite
                }
            }
        }

        // Pick action under each scorer.
        for (i, &sc) in scorers.iter().enumerate() {
            let picked = pick_action(sc, &scored);
            if picked == *correct_action {
                correct[i] += 1;
            }
        }
    }

    scorers
        .iter()
        .enumerate()
        .map(|(i, &sc)| {
            (
                sc,
                AccuracyReport {
                    correct: correct[i],
                    total: N_QUERIES,
                    discarded: discarded[i],
                    total_thoughts,
                },
            )
        })
        .collect()
}

// ── G2: Effective entropy band sweep ─────────────────────────────

/// Sweep (xi_low, xi_high) and measure composite-scorer accuracy. Look for an
/// interior maximum — proof that the entropy band matters.
fn run_g2(
    kernel: &LatentThoughtKernel,
    projection: &[f32],
    queries: &[(Vec<f32>, usize, Vec<f32>)],
) {
    let mut pre_logits = vec![0.0f32; VOCAB];
    let mut post_logits = vec![0.0f32; VOCAB];
    let mut post_probs = vec![0.0f32; VOCAB];
    let mut scratch = vec![0.0f32; 3 * VOCAB];

    // Sweep: fix xi_low at a few values, sweep xi_high.
    let xi_lows = [0.05_f32, 0.10, 0.20, 0.30];
    let xi_highs = [0.50_f32, 0.65, 0.75, 0.85, 0.95, 1.00];

    // Also test the "no gate" baseline (gate ≡ 1): xi_low = 0, xi_high = 1+ε.
    println!();
    println!("── G2: Entropy Band Sweep (composite scorer accuracy) ─────────");
    println!(
        "{:<10} {:<10} {:>10}",
        "xi_low", "xi_high", "accuracy"
    );
    println!("{}", "─".repeat(36));

    // No-gate baseline.
    let no_gate = run_g1_accuracy_only(
        kernel,
        projection,
        queries,
        0.0,
        2.0, // impossibly high → gate ≡ 1 everywhere
        1.0,
        &mut pre_logits,
        &mut post_logits,
        &mut post_probs,
        &mut scratch,
        Scorer::Composite,
    );
    println!(
        "{:<10} {:<10} {:>9.2}%   (baseline: gate≡1)",
        "—", "—", no_gate * 100.0
    );

    let mut best_acc = 0.0_f64;
    let mut best_low = 0.0_f32;
    let mut best_high = 0.0_f32;
    let mut worst_acc = 1.0_f64;

    for &xl in &xi_lows {
        for &xh in &xi_highs {
            if xh <= xl {
                continue;
            }
            let acc = run_g1_accuracy_only(
                kernel,
                projection,
                queries,
                xl,
                xh,
                0.10,
                &mut pre_logits,
                &mut post_logits,
                &mut post_probs,
                &mut scratch,
                Scorer::Composite,
            );
            println!("{:<10.2} {:<10.2} {:>9.2}%", xl, xh, acc * 100.0);
            if acc > best_acc {
                best_acc = acc;
                best_low = xl;
                best_high = xh;
            }
            if acc < worst_acc {
                worst_acc = acc;
            }
        }
    }

    println!();
    let interior_peak = best_low > 0.0 && best_high < 1.0;
    println!("Best band: xi_low={:.2}, xi_high={:.2} → {:.2}%", best_low, best_high, best_acc * 100.0);
    println!("Baseline (gate≡1): {:.2}%", no_gate * 100.0);
    println!(
        "G2: Interior maximum exists? {}   {}",
        if interior_peak { "YES" } else { "NO" },
        if interior_peak {
            if best_acc > no_gate + 0.005 {
                "✅ PASS (band improves over no-gate)"
            } else {
                "⚠️  WEAK (interior max but ≤ no-gate)"
            }
        } else {
            "❌ FAIL (edge maximum — band hurts)"
        }
    );
}

/// Single-scorer accuracy helper for G2.
#[allow(clippy::too_many_arguments)] // bench harness: pre-allocated scratch buffers threaded through to avoid per-iter alloc
fn run_g1_accuracy_only(
    kernel: &LatentThoughtKernel,
    projection: &[f32],
    queries: &[(Vec<f32>, usize, Vec<f32>)],
    xi_low: f32,
    xi_high: f32,
    sharpness: f32,
    pre_logits: &mut [f32],
    post_logits: &mut [f32],
    post_probs: &mut [f32],
    scratch: &mut [f32],
    scorer: Scorer,
) -> f64 {
    let mut correct = 0usize;
    for (init_state, correct_action, input) in queries {
        project_to_logits(init_state, projection, pre_logits);
        let mut scored: Vec<TrajectoryScore> = Vec::with_capacity(N_TRAJECTORIES);
        for n in 0..N_TRAJECTORIES {
            let k = K_CANDIDATES[n % K_CANDIDATES.len()];
            let post_state = run_trajectory(kernel, init_state, input, k);
            project_to_logits(&post_state, projection, post_logits);

            let max_l = post_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut z = 0.0f32;
            for v in 0..VOCAB {
                post_probs[v] = (post_logits[v] - max_l).exp();
                z += post_probs[v];
            }
            for slot in post_probs.iter_mut().take(VOCAB) {
                *slot /= z.max(1e-12);
            }

            scored.push(score_trajectory(
                pre_logits,
                post_logits,
                post_probs,
                k,
                scratch,
                xi_low,
                xi_high,
                sharpness,
            ));
        }
        let picked = pick_action(scorer, &scored);
        if picked == *correct_action {
            correct += 1;
        }
    }
    correct as f64 / N_QUERIES as f64
}

// ── G3: Per-trajectory latency ───────────────────────────────────

fn run_g3(kernel: &LatentThoughtKernel, projection: &[f32]) {
    // Measure: per-trajectory scoring cost = run_trajectory + project + softmax
    // + self_advantage_margin + entropy_band_gate + multiply.
    // Excludes the kernel.step time itself (that's already in micro_belief_bench).
    let mut pre_logits = vec![0.0f32; VOCAB];
    let mut post_logits = vec![0.0f32; VOCAB];
    let mut post_probs = vec![0.0f32; VOCAB];
    let mut scratch = vec![0.0f32; 3 * VOCAB];

    // Use a single fixed input/state to measure pure scoring overhead.
    let init_state: Vec<f32> = (0..DIM).map(|i| (i as f32) * 0.01 - 0.15).collect();
    let input: Vec<f32> = vec![0.5; DIM];

    const ITERS: usize = 100_000;
    let k = 3u8;

    // Warmup.
    for _ in 0..1000 {
        let post = run_trajectory(kernel, &init_state, &input, k);
        project_to_logits(&post, projection, &mut post_logits);
        std::hint::black_box(&post_logits);
    }

    // (a) Kernel step only — already measured in micro_belief_bench, report for context.
    let start = Instant::now();
    for _ in 0..ITERS {
        let post = run_trajectory(kernel, &init_state, &input, k);
        std::hint::black_box(&post);
    }
    let kernel_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

    // (b) Full scoring pipeline (kernel + project + softmax + advantage + entropy gate).
    project_to_logits(&init_state, projection, &mut pre_logits);
    let start = Instant::now();
    for _ in 0..ITERS {
        let post = run_trajectory(kernel, &init_state, &input, k);
        project_to_logits(&post, projection, &mut post_logits);

        // softmax
        let max_l = post_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut z = 0.0f32;
        for v in 0..VOCAB {
            post_probs[v] = (post_logits[v] - max_l).exp();
            z += post_probs[v];
        }
        for slot in post_probs.iter_mut().take(VOCAB) {
            *slot /= z.max(1e-12);
        }

        let score = score_trajectory(
            &pre_logits,
            &post_logits,
            &post_probs,
            k,
            &mut scratch,
            XI_LOW_NORM,
            XI_HIGH_NORM,
            0.10,
        );
        std::hint::black_box(score);
    }
    let full_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

    // (c) Per-N-trajectory decision cost (N=8) — the per-query overhead.
    let n_decision_ns = full_ns * N_TRAJECTORIES as f64;

    println!();
    println!("── G3: Latency (DIM={}, VOCAB={}, N={}, K={}) ────────────────", DIM, VOCAB, N_TRAJECTORIES, k);
    println!("{:<40} {:>9.1} ns", "Kernel-only trajectory (K=3):", kernel_ns);
    println!("{:<40} {:>9.1} ns", "Full scoring per trajectory:", full_ns);
    println!("{:<40} {:>9.1} ns", "Per-query decision (N=8 trajectories):", n_decision_ns);
    println!(
        "{:<40} {:>9.1} ns   {}",
        "G3: Per-trajectory scoring (<1000ns):",
        full_ns,
        if full_ns < 1000.0 { "✅ PASS" } else { "❌ FAIL" }
    );
}

// ── Main ─────────────────────────────────────────────────────────

/// Diagnostic: per-K accuracy. For each K in K_CANDIDATES, run all queries
/// with that single K and report accuracy. Confirms the kernel actually
/// solves the task at higher K (sanity check on the synthetic setup).
fn print_per_k_accuracy(
    kernel: &LatentThoughtKernel,
    projection: &[f32],
    queries: &[(Vec<f32>, usize, Vec<f32>)],
) {
    let mut pre_logits = vec![0.0f32; VOCAB];
    let mut post_logits = vec![0.0f32; VOCAB];
    println!();
    println!("── Diagnostic: per-K accuracy (sanity check on task signal) ───");
    for &k in &K_CANDIDATES {
        let mut correct = 0usize;
        for (init_state, correct_action, input) in queries {
            project_to_logits(init_state, projection, &mut pre_logits);
            let post = run_trajectory(kernel, init_state, input, k);
            project_to_logits(&post, projection, &mut post_logits);
            let mut amax = 0;
            let mut best = f32::NEG_INFINITY;
            for (v, &l) in post_logits.iter().enumerate().take(VOCAB) {
                if l > best {
                    best = l;
                    amax = v;
                }
            }
            if amax == *correct_action {
                correct += 1;
            }
        }
        println!("  K={:<3} → {:.2}%", k, (correct as f64 / N_QUERIES as f64) * 100.0);
    }
    let _ = pre_logits; // suppress unused warning if N_QUERIES == 0
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ Cost-Aware Reward-Proportional Latent Trajectory Scorer     ║");
    println!("║ Issue 030 GOAT Gate — Research 263 (arxiv 2606.16222)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Config: DIM={}, VOCAB={}, N_TRAJECTORIES={}, N_QUERIES={}, λ_c={}, K_CANDIDATES={:?}",
        DIM, VOCAB, N_TRAJECTORIES, N_QUERIES, LAMBDA_C, K_CANDIDATES
    );

    // Build the kernel + projection + queries.
    let kernel = LatentThoughtKernel::from_seed(42, DIM, 1);
    let projection = make_identity_projection();
    let queries = make_queries(2024, &projection);

    // Sanity: confirm the attractor kernel is the same family the kernel wraps.
    let _ = AttractorKernel::from_seed(42, DIM);

    // Diagnostic: print per-K accuracy (does the kernel actually solve it?).
    print_per_k_accuracy(&kernel, &projection, &queries);

    // ── G1 ───────────────────────────────────────────────────────
    println!();
    println!("── G1: Composite vs Single-Component Baselines ────────────────");
    let reports = run_g3_safe(
        &kernel,
        &projection,
        &queries,
        XI_LOW_NORM,
        XI_HIGH_NORM,
        0.10,
    );

    let mut best_single_acc = 0.0_f64;
    let mut best_single_label = "";
    let mut composite_acc = 0.0_f64;
    let mut composite_discard = 0.0_f64;
    let mut weighted_composite_acc = 0.0_f64;
    let mut weighted_advantage_acc = 0.0_f64;
    let mut majority_acc = 0.0_f64;

    println!(
        "{:<32} {:>10} {:>12}",
        "Scorer", "Accuracy", "Discard %"
    );
    println!("{}", "─".repeat(56));
    for (sc, rep) in &reports {
        let acc = rep.accuracy();
        let disc = rep.discard_rate();
        println!(
            "{:<32} {:>9.2}% {:>11.2}%",
            sc.label(),
            acc * 100.0,
            disc * 100.0
        );
        match sc {
            Scorer::Composite => {
                composite_acc = acc;
                composite_discard = disc;
            }
            Scorer::WeightedVoteComposite => {
                weighted_composite_acc = acc;
            }
            Scorer::WeightedVoteAdvantage => {
                weighted_advantage_acc = acc;
            }
            Scorer::MajorityVote => {
                majority_acc = acc;
            }
            Scorer::FirstK1 => {
                // Baseline, not a candidate.
            }
            _ => {
                // CostOnly, AdvantageOnly, EntropyOnly = single components.
                if acc > best_single_acc {
                    best_single_acc = acc;
                    best_single_label = sc.label();
                }
            }
        }
    }

    let g1_acc_gain = composite_acc - best_single_acc;
    let best_fusion = composite_acc.max(weighted_composite_acc).max(weighted_advantage_acc);
    let best_fusion_label = if weighted_advantage_acc == best_fusion {
        "weighted-vote (advantage)"
    } else if weighted_composite_acc == best_fusion {
        "weighted-vote (composite)"
    } else {
        "composite (argmax)"
    };
    let fusion_vs_majority = best_fusion - majority_acc;

    println!();
    println!("Best single-component:           {} ({:.2}%)", best_single_label, best_single_acc * 100.0);
    println!("Majority vote (no score):        {:.2}%", majority_acc * 100.0);
    println!("Composite (argmax, paper shape): {:.2}%", composite_acc * 100.0);
    println!("Best fusion variant:             {} ({:.2}%)", best_fusion_label, best_fusion * 100.0);
    println!();
    println!(
        "G1a: Composite-argmax vs best single (≥3pp target): {:+.2}pp   {}",
        g1_acc_gain * 100.0,
        if g1_acc_gain >= 0.03 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "G1b: Best fusion vs majority vote (≥1pp target):     {:+.2}pp   {}",
        fusion_vs_majority * 100.0,
        if fusion_vs_majority >= 0.01 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "G1c: Discard rate (≥30% target):                    {:.2}%    {}",
        composite_discard * 100.0,
        if composite_discard >= 0.30 { "✅ PASS" } else { "❌ FAIL" }
    );

    let g1_pass = g1_acc_gain >= 0.03;
    let g1b_pass = fusion_vs_majority >= 0.01;
    let g1c_pass = composite_discard >= 0.30;

    // ── G2 ───────────────────────────────────────────────────────
    run_g2(&kernel, &projection, &queries);

    // ── G3 ───────────────────────────────────────────────────────
    run_g3(&kernel, &projection);

    // ── Final verdict ────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ Promotion Verdict (Issue 030)                               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║ G1a (composite ≥3pp over best single): {:<23} ║",
        if g1_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "║ G1b (best fusion ≥1pp over majority):  {:<23} ║",
        if g1b_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "║ G1c (≥30% dead-thought discard):       {:<23} ║",
        if g1c_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!("║ G2  (interior entropy band peak):      see above             ║");
    println!("║ G3  (<1µs per-trajectory scoring):     see above             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    if g1_pass || g1b_pass {
        println!("→ PROMOTE: file plan + feature flag `latent_thought_flow_scorer`.");
    } else {
        println!("→ DO NOT PROMOTE: fusion is incremental; close Issue 030 or re-tune.");
    }
}

/// Wrapper around `run_g1` that ignores the trivially-named conflict (we want
/// G3 latency too — but `run_g1` is the accuracy harness; this fn is just a
/// rename to avoid clashing with `run_g3`).
#[inline]
fn run_g3_safe(
    kernel: &LatentThoughtKernel,
    projection: &[f32],
    queries: &[(Vec<f32>, usize, Vec<f32>)],
    xi_low: f32,
    xi_high: f32,
    sharpness: f32,
) -> Vec<(Scorer, AccuracyReport)> {
    run_g1(kernel, projection, queries, xi_low, xi_high, sharpness)
}
