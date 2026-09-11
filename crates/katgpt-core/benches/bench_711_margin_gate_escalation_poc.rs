//! Margin-gated verification escalation PoC — defend-wrong bench (Issue 745 /
//! Plan 595, Research 548: TriSpec distill, arXiv:2601.23180).
//!
//! Three controlled toy worlds × three competitors, measuring whether a
//! zero-training `top1 − top2` margin gate on already-materialized scores can
//! cut expensive-verifier invocations within the stated loss budget ε.
//!
//! # The three worlds (the defend-wrong matrix)
//!
//! - **World A — cluster geometry** (the Bench 320 / Zhou et al. premise):
//!   a genuine event activates a CORRELATED PAIR of indicators; lone decisive
//!   spikes are spurious. Correct deployment: `CoherentTrusted`. This is the
//!   CASCADE mapping (gate over 8 sigmoid probe scores).
//! - **World B — decisive geometry** (the cascade-side TriSpec analog): a
//!   genuine event is a single decisive spike; ambiguous co-fires are the
//!   wrong-answer shape. Correct deployment: `DecisiveTrusted`. Same cascade
//!   mapping.
//! - **World C — accept mapping** (the TriSpec-faithful surface): a V=32
//!   token distribution at the draft-accept point. Correct drafts are peaked
//!   (gap ≥ ~0.52); wrong drafts are ambiguous (gap ∈ [0.05, 0.45]) EXCEPT a
//!   `tail_rate` fraction that are confidently wrong with the peaked shape —
//!   the lossy tail, made explicit. The gate is `margin_split` + `MarginGate`
//!   directly (no probe bank); escalation = the target re-decide (Leviathan
//!   residual substitution in the shipped d2f mapping; oracle here).
//!
//! # Measured verdicts the gates assert (a flip must be conscious)
//!
//! 1. **G1a — the cascade mapping is REFUTED at ε** (worlds A + B): sigmoid
//!    score-domain saturation + the noise floor make the trusted/escalated
//!    gap distributions overlap; NO (polarity, λ) cell achieves the
//!    invocation cut at ≤ ε flagged-case regression. This is the recorded
//!    negative result for the issue's stage-1-augment claim; if a future
//!    cell QUALIFIES, this gate reds and forces a verdict revisIT.
//! 2. **G1b/G1c/G3 — the accept mapping is VIABLE, ε-sensitive** (world C):
//!    at tail 0.5% a qualifying λ exists (regression ≤ ε at a real cut); at
//!    tail 2% (double TriSpec's ≤1% ceiling) the control MUST fire — no λ
//!    qualifies. The gate is a *calibrated* win: it inherits the tail rate.
//! 3. **G2/G4 — mechanism properties**: `margin_split` is effectively free
//!    per candidate and the gated `run()` is ≤1.5× the baseline cascade;
//!    zero allocations.
//! 4. **T4 — bandit-λ**: (world B) paired UCB1 vs fixed λ; (world C) UCB1
//!    with an ε-feasibility mask — and the FINDING that the mapped
//!    `meta_router::compute_reward` is BLIND to the lossy tail (a trusted
//!    wrong and an escalated wrong both score 0, so the ε-violating λ ties
//!    the best arm on raw reward): the mask is load-bearing, not decoration.
//!
//! # Competitors (per world)
//!
//! Baseline cascade (always-escalate gate = `IndicatorCascade` semantics),
//! margin-gated cascade (both polarities where applicable, λ sweep),
//! always-verify upper bound (oracle on every candidate).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features margin_gate \
//!   --bench bench_711_margin_gate_escalation_poc -- --nocapture
//! ```

#![cfg(feature = "margin_gate")]

use katgpt_core::pruners::indicator_cascade::IndicatorVerifier;
use katgpt_core::pruners::indicator_probe_bank::{IndicatorLabel, IndicatorProbeBank};
use katgpt_core::pruners::margin_gate::{
    MarginDecision, MarginGate, MarginGatedCascade, MarginPolarity, margin_split,
};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── GateResult ─────────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

impl GateResult {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: true,
            detail: detail.into(),
        }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            detail: detail.into(),
        }
    }
}

// ─── Deterministic RNG (xorshift32 + Box–Muller normals) ────────────────────

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn uniform01(state: &mut u32) -> f32 {
    (xorshift32(state) as f64 / u32::MAX as f64) as f32
}

fn normal(state: &mut u32) -> f32 {
    // Box–Muller; the discarded second variate keeps the stream simple.
    let u1 = uniform01(state).max(1e-9);
    let u2 = uniform01(state);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn uniform_range(state: &mut u32, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * uniform01(state)
}

// ─── Synthetic bank (the Bench 320 shape, smaller D) ────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
/// Generic discriminants only — the planted GEOMETRY (which shape is the
/// genuine event) carries the meaning, not the names (Bench 320 convention).
enum SyntheticIndicator {
    Ind0 = 0,
    Ind1 = 1,
    Ind2 = 2,
    Ind3 = 3,
    Ind4 = 4,
    Ind5 = 5,
    Ind6 = 6,
    Ind7 = 7,
}

impl IndicatorLabel for SyntheticIndicator {
    fn as_u8(&self) -> u8 {
        *self as u8
    }
    fn from_u8(d: u8) -> Option<Self> {
        match d {
            0 => Some(Self::Ind0),
            1 => Some(Self::Ind1),
            2 => Some(Self::Ind2),
            3 => Some(Self::Ind3),
            4 => Some(Self::Ind4),
            5 => Some(Self::Ind5),
            6 => Some(Self::Ind6),
            7 => Some(Self::Ind7),
            _ => None,
        }
    }
    const COUNT: usize = 8;
}

const N_IND: usize = 8;
const D: usize = 32;
const AXIS_STRIDE: usize = 4; // indicator i lives on coord i*4 (disjoint axes)
const DIR_SCALE: f32 = 4.0;
const THRESHOLD: f32 = 2.0;
const NOISE_STD: f32 = 0.5; // per-coord state noise → raw noise σ = 2.0
const TAU_FIRE: f32 = 0.96;
const PI_POS: f32 = 0.10; // prior probability of a genuine event
const N_TRIALS: usize = 40_000;

fn demo_bank() -> Arc<IndicatorProbeBank<SyntheticIndicator, D>> {
    let mut directions = vec![0.0f32; N_IND * D];
    for i in 0..N_IND {
        directions[i * D + i * AXIS_STRIDE] = DIR_SCALE;
    }
    Arc::new(IndicatorProbeBank::new(directions, vec![THRESHOLD; N_IND]).unwrap())
}

// ─── Cascade worlds (A: cluster / B: decisive) ──────────────────────────────

/// Which cascade toy world a trial set comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum World {
    /// Genuine event = correlated PAIR; lone decisive spike = spurious.
    Cluster,
    /// Genuine event = single DECISIVE spike; ambiguous co-fire = wrong answer.
    Decisive,
}

impl World {
    fn name(self) -> &'static str {
        match self {
            Self::Cluster => "A:cluster(coherent-trusted)",
            Self::Decisive => "B:decisive(decisive-trusted)",
        }
    }
    /// The polarity a CORRECT deployment uses on this world.
    fn correct_polarity(self) -> MarginPolarity {
        match self {
            Self::Cluster => MarginPolarity::CoherentTrusted,
            Self::Decisive => MarginPolarity::DecisiveTrusted,
        }
    }
}

struct Trials {
    states: Vec<[f32; D]>,
    planted: Vec<bool>,
}

/// Sample one candidate state for `world`. `planted` candidates carry the
/// world's genuine-event shape; negatives carry the world's wrong-answer
/// shape (or pure noise).
fn sample_trial(world: World, rng: &mut u32) -> ([f32; D], bool) {
    let mut state = [0.0f32; D];
    for c in state.iter_mut() {
        *c = normal(rng) * NOISE_STD;
    }
    let planted = uniform01(rng) < PI_POS;
    let i = (xorshift32(rng) % N_IND as u32) as usize;
    match (world, planted) {
        // Genuine cluster event: correlated pair (i, i^1), both strong.
        (World::Cluster, true) => {
            let a = uniform_range(rng, 2.0, 3.0);
            let j = i ^ 1;
            state[i * AXIS_STRIDE] += a;
            state[j * AXIS_STRIDE] += a;
        }
        // Negative in the cluster world: pure noise (lone spikes are spurious).
        (World::Cluster, false) => {}
        // Genuine decisive event: ONE strong spike.
        (World::Decisive, true) => {
            let a = uniform_range(rng, 2.0, 3.0);
            state[i * AXIS_STRIDE] += a;
        }
        // Negative in the decisive world: 30% ambiguous co-fires (the
        // wrong-answer shape: no decisive verdict, the "confused draft"),
        // 70% pure noise.
        (World::Decisive, false) => {
            if uniform01(rng) < 0.30 {
                let j = (i + 1 + (xorshift32(rng) % 7) as usize) % N_IND;
                let a1 = uniform_range(rng, 2.0, 3.0);
                let a2 = uniform_range(rng, 0.6, 3.0);
                state[i * AXIS_STRIDE] += a1;
                state[j * AXIS_STRIDE] += a2;
            }
        }
    }
    (state, planted)
}

fn sample_world(world: World, seed: u32) -> Trials {
    let mut rng = seed;
    let mut states = Vec::with_capacity(N_TRIALS);
    let mut planted = Vec::with_capacity(N_TRIALS);
    for _ in 0..N_TRIALS {
        let (s, p) = sample_trial(world, &mut rng);
        states.push(s);
        planted.push(p);
    }
    Trials { states, planted }
}

// ─── Oracle stage-2 verifier (perfect judge, stateful per-candidate truth) ──

struct OracleVerifier {
    /// Whether the CURRENT candidate under adjudication is a genuine event.
    current_truth: AtomicUsize, // 0 = clean, 1 = planted
}

impl OracleVerifier {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            current_truth: AtomicUsize::new(0),
        })
    }
    fn set_truth(&self, planted: bool) {
        self.current_truth
            .store(planted as usize, Ordering::Relaxed);
    }
}

impl IndicatorVerifier<SyntheticIndicator> for OracleVerifier {
    fn verify(&self, _label: SyntheticIndicator, _scores: &[f32]) -> bool {
        self.current_truth.load(Ordering::Relaxed) == 1
    }
}

// ─── Metrics ────────────────────────────────────────────────────────────────

#[derive(Default, Clone, Copy)]
struct Metrics {
    candidates: usize,
    positives: usize,
    /// Adjudicated candidates (fired in the cascade mapping; EVERY candidate
    /// in the accept mapping). The flagged-case denominator.
    fired: usize,
    verifier_calls: usize,
    confirmed: usize,
    /// Confirmed on a genuine/correct candidate (true positive, end-to-end).
    tp: usize,
    /// Confirmed on a wrong candidate (false positive, end-to-end). In the
    /// accept mapping this IS the lossy tail (trusted ∧ wrong).
    fp: usize,
    /// Adjudicated wrong candidates the expensive path correctly REJECTED.
    neg_rejected: usize,
    /// Trusted by the margin gate but NOT correct (the lossy tail).
    trusted_fp: usize,
}

impl Metrics {
    fn tpr(&self) -> f64 {
        if self.positives == 0 {
            0.0
        } else {
            self.tp as f64 / self.positives as f64
        }
    }
    fn fpr(&self) -> f64 {
        let negs = self.candidates - self.positives;
        if negs == 0 {
            0.0
        } else {
            self.fp as f64 / negs as f64
        }
    }
    /// Flagged-case accuracy: of the adjudicated candidates, the fraction
    /// whose final verdict matches the oracle. With a perfect oracle the
    /// baseline sits at 1.0; the gated regression is `1 − flag_accuracy`.
    fn flag_accuracy(&self) -> f64 {
        if self.fired == 0 {
            1.0
        } else {
            (self.tp + self.neg_rejected) as f64 / self.fired as f64
        }
    }
    fn call_rate(&self) -> f64 {
        if self.candidates == 0 {
            0.0
        } else {
            self.verifier_calls as f64 / self.candidates as f64
        }
    }
}

fn fmt_row(name: &str, m: &Metrics) -> String {
    format!(
        "{name:<40} calls={:>5.3}  FPR={:.5}  TPR={:.4}  flagAcc={:.5}  tailFP={}  adjudicated={}",
        m.call_rate(),
        m.fpr(),
        m.tpr(),
        m.flag_accuracy(),
        m.trusted_fp,
        m.fired
    )
}

/// Run the cascade (gated). The BASELINE (Bench 320 semantics: verifier on
/// EVERY flag) is the same runner with a gate that can never trust
/// (`gap ≥ ∞` is always false → every fire escalates) — identical decision
/// path to `IndicatorCascade::run`, with consistent fired/confirmed
/// bookkeeping across all three competitors.
fn run_margin_gated(
    cascade: &MarginGatedCascade<SyntheticIndicator, D>,
    oracle: &OracleVerifier,
    trials: &Trials,
) -> Metrics {
    let mut m = Metrics::default();
    let mut scratch = [0.0f32; N_IND];
    for (state, &planted) in trials.states.iter().zip(trials.planted.iter()) {
        m.candidates += 1;
        m.positives += planted as usize;
        oracle.set_truth(planted);
        let decision = cascade.run(state, &mut scratch);
        match decision {
            MarginDecision::Clean => {}
            MarginDecision::Trusted(_) => {
                m.fired += 1;
                m.confirmed += 1;
                m.tp += planted as usize;
                m.fp += (!planted) as usize;
                m.trusted_fp += (!planted) as usize;
            }
            MarginDecision::Escalated(label) => {
                m.fired += 1;
                m.verifier_calls += 1;
                let verdict = oracle.verify(label, &scratch);
                m.confirmed += verdict as usize;
                m.tp += (verdict && planted) as usize;
                m.fp += (verdict && !planted) as usize;
                m.neg_rejected += (!verdict && !planted) as usize;
            }
        }
    }
    m
}

/// ALWAYS-VERIFY upper bound: the expensive judge adjudicates EVERY candidate
/// (call_rate = 1.0 by construction; with a perfect oracle its accuracy is
/// 1.0 — the accuracy ceiling the gated arms are measured against).
fn run_always_verify(
    oracle: &OracleVerifier,
    bank: &IndicatorProbeBank<SyntheticIndicator, D>,
    trials: &Trials,
) -> Metrics {
    let mut m = Metrics::default();
    let mut scratch = [0.0f32; N_IND];
    for (state, &planted) in trials.states.iter().zip(trials.planted.iter()) {
        m.candidates += 1;
        m.positives += planted as usize;
        oracle.set_truth(planted);
        bank.project_all_into(state, &mut scratch);
        m.verifier_calls += 1;
        let verdict = match bank.or_fused_fire(&scratch, TAU_FIRE) {
            Some(label) => oracle.verify(label, &scratch),
            None => oracle.verify(SyntheticIndicator::Ind0, &scratch),
        };
        // The always-verify arm adjudicates EVERY candidate — its flagged
        // case set is the whole stream.
        m.fired += 1;
        m.confirmed += verdict as usize;
        m.tp += (verdict && planted) as usize;
        m.fp += (verdict && !planted) as usize;
        m.neg_rejected += (!verdict && !planted) as usize;
    }
    m
}

// ─── G1a: the cascade-mapping refutation (worlds A + B) ────────────────────

const EPS_FLAG_ACC: f64 = 0.005; // the stated ε (the lossy axis, made explicit)
const MIN_CUT: f64 = 0.20; // a qualifying cell must cut invocations ≥ 20%

fn gate_cascade_refutation() -> Vec<GateResult> {
    const LAMBDAS: [f32; 4] = [0.1, 0.3, 0.5, 0.7];
    let bank = demo_bank();
    let oracle = OracleVerifier::new();
    let mut results = Vec::new();

    // The baseline cascade: same runner, a gate that can never trust.
    let baseline_gate = MarginGate {
        lambda: f32::INFINITY,
        polarity: MarginPolarity::DecisiveTrusted,
    };

    for &world in &[World::Cluster, World::Decisive] {
        let trials = sample_world(world, 0x7450 + world as u32);
        let baseline = run_margin_gated(
            &MarginGatedCascade::new(bank.clone(), oracle.clone(), TAU_FIRE, baseline_gate),
            &oracle,
            &trials,
        );
        let upper = run_always_verify(&oracle, &bank, &trials);

        println!("\n── World {} ──", world.name());
        println!("  {}", fmt_row("baseline cascade (Bench 320)", &baseline));
        println!("  {}", fmt_row("always-verify upper bound", &upper));
        println!("  λ trade curve (both polarities):");
        println!(
            "  {:<40} {:>7} {:>9} {:>8} {:>9} {:>8}",
            "arm", "calls", "FPR", "TPR", "flagAcc", "cut"
        );

        let mut any_qualifies = false;
        // Best ATTEMPT for the detail line: max cut subject to reg ≤ ε; if
        // none, min regression subject to cut ≥ MIN_CUT; else max cut.
        let mut best: Option<(f64, f64, f32, MarginPolarity)> = None; // (cut, reg, λ, pol)
        for &lam in LAMBDAS.iter() {
            for &polarity in &[
                MarginPolarity::CoherentTrusted,
                MarginPolarity::DecisiveTrusted,
            ] {
                let cascade = MarginGatedCascade::new(
                    bank.clone(),
                    oracle.clone(),
                    TAU_FIRE,
                    MarginGate {
                        lambda: lam,
                        polarity,
                    },
                );
                let m = run_margin_gated(&cascade, &oracle, &trials);
                let cut = 1.0 - (m.verifier_calls as f64 / baseline.verifier_calls.max(1) as f64);
                let reg = baseline.flag_accuracy() - m.flag_accuracy();
                let pol = match polarity {
                    MarginPolarity::CoherentTrusted => "coherent",
                    MarginPolarity::DecisiveTrusted => "decisive",
                };
                println!(
                    "  {:<40} {:>7.3} {:>9.5} {:>8.4} {:>9.5} {:>+7.1}%",
                    format!("gated λ={lam:.1} {pol}"),
                    m.call_rate(),
                    m.fpr(),
                    m.tpr(),
                    m.flag_accuracy(),
                    cut * 100.0
                );
                if cut >= MIN_CUT && reg <= EPS_FLAG_ACC {
                    any_qualifies = true;
                }
                let better = match best {
                    None => true,
                    Some((b_cut, b_reg, _, _)) => {
                        if reg <= EPS_FLAG_ACC {
                            b_reg > EPS_FLAG_ACC || cut > b_cut
                        } else if b_reg <= EPS_FLAG_ACC {
                            false
                        } else if cut >= MIN_CUT && b_cut >= MIN_CUT {
                            reg < b_reg
                        } else {
                            cut > b_cut
                        }
                    }
                };
                if better {
                    best = Some((cut, reg, lam, polarity));
                }
            }
        }

        let (cut, reg, lam, polarity) = best.expect("sweep produced no row");
        let pol = match polarity {
            MarginPolarity::CoherentTrusted => "coherent",
            MarginPolarity::DecisiveTrusted => "decisive",
        };
        if !any_qualifies {
            results.push(GateResult::pass(
                "G1a",
                format!(
                    "world {}: refutation HOLDS — best attempt λ={lam:.1} {pol}: cut {:+.1}%, regression {:.5}; no cell reaches cut ≥ {}% at regression ≤ ε({EPS_FLAG_ACC})",
                    world.name(), cut * 100.0, reg, MIN_CUT * 100.0
                ),
            ));
        } else {
            results.push(GateResult::fail(
                "G1a",
                format!(
                    "world {}: a cell now QUALIFIES (cut ≥ {}% at regression ≤ ε) — the recorded refutation no longer holds; revisit the verdict",
                    world.name(), MIN_CUT * 100.0
                ),
            ));
        }
    }
    results.shrink_to_fit();
    results
}

// ─── World C: the TriSpec-faithful accept mapping ───────────────────────────

const V: usize = 32; // toy vocab

struct TokenTrials {
    probs: Vec<[f32; V]>,
    correct: Vec<bool>,
}

/// Sample one draft distribution at the accept point.
///
/// - **Correct** (50%): peaked on the winner token — top1 ∈ [0.55, 0.95],
///   the (1−top1) residual spread over the other 31 tokens (top2 ≤ ~0.03),
///   so `gap ∈ [~0.52, ~0.95]`: decisive.
/// - **Wrong** (50%):
///   - with prob `1 − tail_rate`: ambiguous — two near-rival tops with
///     `gap ∈ [0.05, 0.45]` (the confused-draft shape the target rejects);
///   - with prob `tail_rate`: CONFIDENTLY wrong — the peaked shape on a
///     wrong token. This is the lossy tail, a world property the gate
///     inherits: at any λ above the ambiguous band these are trusted.
fn sample_token_trial(tail_rate: f32, rng: &mut u32) -> ([f32; V], bool) {
    let correct = uniform01(rng) < 0.5;
    let winner = (xorshift32(rng) % V as u32) as usize;
    let mut p = [0.0f32; V];
    if correct || uniform01(rng) < tail_rate {
        // Peaked (decisive) shape — correct signature, or lossy tail.
        let top1 = uniform_range(rng, 0.55, 0.95);
        p[winner] = top1;
        let residual = 1.0 - top1;
        let mut acc = 0.0f32;
        let mut weights = [0.0f32; V];
        for (i, w) in weights.iter_mut().enumerate() {
            if i != winner {
                let x = uniform01(rng) + 1e-3;
                *w = x;
                acc += x;
            }
        }
        for (i, pi) in p.iter_mut().enumerate() {
            if i != winner {
                *pi = residual * weights[i] / acc;
            }
        }
    } else {
        // Ambiguous shape: two near-rival tops, gap ∈ [0.05, 0.45].
        let gap = uniform_range(rng, 0.05, 0.45);
        let top1 = uniform_range(rng, 0.40, 0.55);
        let top2 = (top1 - gap).max(0.01);
        let loser = (winner + 1 + (xorshift32(rng) % (V as u32 - 1)) as usize) % V;
        p[winner] = top1;
        p[loser] = top2;
        let residual = (1.0 - top1 - top2).max(0.0);
        let mut acc = 0.0f32;
        let mut weights = [0.0f32; V];
        for (i, w) in weights.iter_mut().enumerate() {
            if i != winner && i != loser {
                let x = uniform01(rng) + 1e-3;
                *w = x;
                acc += x;
            }
        }
        for (i, pi) in p.iter_mut().enumerate() {
            if i != winner && i != loser {
                *pi = residual * weights[i] / acc;
            }
        }
    }
    (p, correct)
}

fn sample_token_world(seed: u32, tail_rate: f32) -> TokenTrials {
    let mut rng = seed;
    let mut probs = Vec::with_capacity(N_TRIALS);
    let mut correct = Vec::with_capacity(N_TRIALS);
    for _ in 0..N_TRIALS {
        let (p, c) = sample_token_trial(tail_rate, &mut rng);
        probs.push(p);
        correct.push(c);
    }
    TokenTrials { probs, correct }
}

/// Run the accept mapping: every candidate is a draft proposal (fired = N);
/// trusted ⟹ accept the draft token WITHOUT the target; distrusted ⟹ the
/// target re-decides (Leviathan residual substitution in the shipped d2f
/// mapping; here the oracle accepts iff the draft was correct).
fn run_accept_gate(trials: &TokenTrials, gate: MarginGate) -> Metrics {
    let mut m = Metrics::default();
    for (p, &correct) in trials.probs.iter().zip(trials.correct.iter()) {
        m.candidates += 1;
        m.positives += correct as usize;
        m.fired += 1;
        let split = margin_split(p);
        if gate.trusted(split.gap) {
            m.confirmed += 1;
            m.tp += correct as usize;
            m.fp += (!correct) as usize;
            m.trusted_fp += (!correct) as usize;
        } else {
            // The target's re-decide is correct by premise (oracle).
            m.verifier_calls += 1;
            m.confirmed += correct as usize;
            m.tp += correct as usize;
            m.neg_rejected += (!correct) as usize;
        }
    }
    m
}

/// G1b (viable at tail 0.5%) + G3 (the qualifying cut) + G1c (the tail
/// sensitivity control MUST fire at tail 2%).
fn gate_accept_mapping() -> Vec<GateResult> {
    const LAMBDAS: [f32; 3] = [0.3, 0.5, 0.7];
    let viable_tail = 0.005; // half of TriSpec's ≤1% lossy ceiling
    let control_tail = 0.02; // 2× TriSpec's ceiling — the control MUST fail
    let mut results = Vec::new();

    for &tail in &[viable_tail, control_tail] {
        let is_control = tail > viable_tail;
        let trials = sample_token_world(0xC000 + (tail * 1e6) as u32, tail);
        // Upper bound: never trust → the target re-decides everything.
        let upper = run_accept_gate(
            &trials,
            MarginGate {
                lambda: f32::INFINITY,
                polarity: MarginPolarity::DecisiveTrusted,
            },
        );
        // The never-escalate extreme: accept everything (accuracy = 1−tail).
        let always_accept = run_accept_gate(
            &trials,
            MarginGate {
                lambda: f32::NEG_INFINITY,
                polarity: MarginPolarity::DecisiveTrusted,
            },
        );

        println!("\n── World C:accept(tail={:.1}%) ──", tail * 100.0);
        println!(
            "  {}",
            fmt_row("always-verify oracle (never trust)", &upper)
        );
        println!(
            "  {}",
            fmt_row("never-escalate (trust all)", &always_accept)
        );
        println!("  λ trade curve (DecisiveTrusted):");
        println!(
            "  {:<40} {:>7} {:>9} {:>8} {:>9} {:>8}",
            "arm", "calls", "FPR", "TPR", "flagAcc", "cut"
        );

        let mut qualifying: Vec<(f64, f64, f32)> = Vec::new(); // (cut, reg, λ)
        for &lam in LAMBDAS.iter() {
            let gate = MarginGate {
                lambda: lam,
                polarity: MarginPolarity::DecisiveTrusted,
            };
            let m = run_accept_gate(&trials, gate);
            let cut = 1.0 - m.call_rate();
            let reg = 1.0 - m.flag_accuracy(); // oracle baseline accuracy = 1.0
            println!(
                "  {:<40} {:>7.3} {:>9.5} {:>8.4} {:>9.5} {:>+7.1}%",
                format!("gated λ={lam:.1}"),
                m.call_rate(),
                m.fpr(),
                m.tpr(),
                m.flag_accuracy(),
                cut * 100.0
            );
            if reg <= EPS_FLAG_ACC && cut >= MIN_CUT {
                qualifying.push((cut, reg, lam));
            }
        }

        if !is_control {
            // G1b + G3: a qualifying λ exists, and its cut is real.
            if let Some((cut, reg, lam)) = qualifying
                .iter()
                .copied()
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            {
                results.push(GateResult::pass(
                        "G1b",
                        format!(
                            "world C accept(tail={:.1}%): λ={lam:.1} qualifies — regression {reg:.5} ≤ ε({EPS_FLAG_ACC}) at cut {:+.1}%",
                            tail * 100.0, cut * 100.0
                        ),
                    ));
                results.push(GateResult::pass(
                        "G3",
                        format!(
                            "world C accept(tail={:.1}%): target re-decide invocations cut {:+.1}% ≥ {}% at λ={lam:.1}",
                            tail * 100.0, cut * 100.0, MIN_CUT * 100.0
                        ),
                    ));
            } else {
                results.push(GateResult::fail(
                        "G1b",
                        format!(
                            "world C accept(tail={:.1}%): no λ qualifies (cut ≥ {}% at regression ≤ ε) — the accept mapping does NOT meet the loss budget here",
                            tail * 100.0, MIN_CUT * 100.0
                        ),
                    ));
                results.push(GateResult::fail(
                    "G3",
                    "no qualifying λ at the viable tail — see G1b".to_string(),
                ));
            }
        } else {
            // G1c: the sensitivity control must fire — doubling the tail
            // (2× TriSpec's ceiling) must break EVERY λ.
            if qualifying.is_empty() {
                results.push(GateResult::pass(
                    "G1c",
                    format!(
                        "world C accept(tail={:.1}%): control FIRES — no λ qualifies at double TriSpec's ceiling; the gate is genuinely ε-sensitive",
                        tail * 100.0
                    ),
                ));
            } else {
                results.push(GateResult::fail(
                    "G1c",
                    format!(
                        "world C accept(tail={:.1}%): control BLIND — {} λ still qualify at a tail 2× over TriSpec's ceiling; ε is not actually enforced",
                        tail * 100.0, qualifying.len()
                    ),
                ));
            }
        }
    }
    results.shrink_to_fit();
    results
}

// ─── G2: margin operator + cascade hot-path latency ─────────────────────────

fn median_ns<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos() as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[iters / 2]
}

fn gate_g2() -> GateResult {
    let bank = demo_bank();
    let oracle = OracleVerifier::new();
    let mut rng = 0x595u32;
    let mut scores = [0.0f32; N_IND];

    bank.project_all_into(
        &{
            let mut s = [0.0f32; D];
            for c in s.iter_mut() {
                *c = normal(&mut rng) * NOISE_STD;
            }
            s
        },
        &mut scores,
    );

    // margin_split latency at the cascade's N=8 score width. BATCHED: a
    // single sub-5ns call is below the Instant clock's useful resolution —
    // 64 calls per timed sample amortize the now() overhead.
    const BATCH: usize = 64;
    let split_ns = median_ns(2_000, || {
        for _ in 0..BATCH {
            black_box(margin_split(black_box(&scores)));
        }
    }) / BATCH as f64;
    // Absolute bar: the operator must stay two orders of magnitude under any
    // real verifier invocation (µs–ms), i.e. effectively free per candidate.

    // Gated run vs baseline run (always-escalate gate = IndicatorCascade
    // semantics), same state stream.
    let trials = sample_world(World::Decisive, 0x5951);
    let base_cascade = MarginGatedCascade::new(
        bank.clone(),
        oracle.clone(),
        TAU_FIRE,
        MarginGate {
            lambda: f32::INFINITY,
            polarity: MarginPolarity::DecisiveTrusted,
        },
    );
    let gated_cascade = MarginGatedCascade::new(
        bank,
        oracle,
        TAU_FIRE,
        MarginGate {
            lambda: 0.3,
            polarity: MarginPolarity::DecisiveTrusted,
        },
    );
    let mut idx = 0usize;
    let mut scratch = [0.0f32; N_IND];
    let base_ns = median_ns(5_000, || {
        let state = &trials.states[idx % trials.states.len()];
        black_box(base_cascade.run(black_box(state), &mut scratch));
        idx += 1;
    });
    let mut idx = 0usize;
    let gated_ns = median_ns(5_000, || {
        let state = &trials.states[idx % trials.states.len()];
        black_box(gated_cascade.run(black_box(state), &mut scratch));
        idx += 1;
    });

    let ratio = gated_ns / base_ns;
    if split_ns < 50.0 && ratio <= 1.5 {
        GateResult::pass(
            "G2",
            format!(
                "margin_split {split_ns:.1}ns (<50ns @ N={N_IND}); gated run {gated_ns:.0}ns vs baseline {base_ns:.0}ns (ratio {ratio:.2}× ≤ 1.5×)"
            ),
        )
    } else {
        GateResult::fail(
            "G2",
            format!(
                "margin_split {split_ns:.1}ns (target <50ns @ N={N_IND}); gated run {gated_ns:.0}ns vs baseline {base_ns:.0}ns (ratio {ratio:.2}×, target ≤1.5×)"
            ),
        )
    }
}

// ─── G4: alloc-free gated run ───────────────────────────────────────────────

fn gate_g4() -> GateResult {
    assert_counter_is_live();
    let bank = demo_bank();
    let oracle = OracleVerifier::new();
    let trials = sample_world(World::Cluster, 0x5952);
    let cascade = MarginGatedCascade::new(
        bank,
        oracle,
        TAU_FIRE,
        MarginGate {
            lambda: 0.2,
            polarity: MarginPolarity::CoherentTrusted,
        },
    );
    let mut scratch = [0.0f32; N_IND];
    // Warmup (first touch of any lazy state), then the measured window.
    for state in trials.states.iter().take(64) {
        let _ = cascade.run(state, &mut scratch);
    }
    let (_, allocs) = alloc_delta(|| {
        for i in 0..100 {
            let state = &trials.states[i % trials.states.len()];
            black_box(cascade.run(black_box(state), &mut scratch));
        }
    });
    if allocs == 0 {
        GateResult::pass(
            "G4",
            "0 allocs / 100 mixed gated runs (trusted + escalated)",
        )
    } else {
        GateResult::fail("G4", format!("{allocs} allocs / 100 gated runs (target 0)"))
    }
}

// ─── T4: bandit-λ under the mapped meta-router reward ──────────────────────

/// `meta_router::compute_reward` mapped onto the verification outcome:
/// `acceptance` ↔ the final verdict matches the oracle;
/// `latency_improvement` ↔ the expensive verifier was NOT invoked (1.0) vs
/// invoked (0.0). Reward = correct × (1 + saved) ∈ [0, 2], exactly the
/// router's shape.
///
/// FINDING (world C): this reward is BLIND to the lossy tail — a trusted
/// wrong and an escalated wrong both score 0, so the ε-violating λ ties the
/// best arm on raw reward. The mask in [`MaskedUcb1`] is load-bearing.
fn cascade_reward(correct: bool, verifier_invoked: bool) -> f32 {
    let acceptance = if correct { 1.0f32 } else { 0.0 };
    let saved = if verifier_invoked { 0.0f32 } else { 1.0 };
    acceptance * (1.0 + saved)
}

/// UCB1 (the `bench_374` ReMax pattern; katgpt-attn's MetaRouter is
/// upstream of katgpt-core, so the prod-router wiring is consumer-side).
struct Ucb1 {
    counts: Vec<u64>,
    sums: Vec<f64>,
    t: u64,
}

impl Ucb1 {
    fn new(k: usize) -> Self {
        Self {
            counts: vec![0; k],
            sums: vec![0.0; k],
            t: 0,
        }
    }
    fn select(&mut self) -> usize {
        let k = self.counts.len();
        for i in 0..k {
            if self.counts[i] == 0 {
                return i;
            }
        }
        let ln_t = (self.t as f64).ln();
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..k {
            let exploit = self.sums[i] / self.counts[i] as f64;
            let explore = (2.0f64 * ln_t / self.counts[i] as f64).sqrt();
            let score = exploit + explore;
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }
    fn update(&mut self, arm: usize, reward: f32) {
        self.counts[arm] += 1;
        self.sums[arm] += reward as f64;
        self.t += 1;
    }
}

/// UCB1 with an ε-feasibility mask: arms whose wrong rate is CONFIDENTLY
/// above ε are excluded from selection (the reward shape cannot see the
/// lossy tail, so the constraint must enter as a mask, not the reward).
///
/// The mask must be a one-sided CONFIDENCE bound, not a point estimate —
/// measured failure (Bench 711 T4c): a good arm with a true 0.25% tail drew
/// 2 wrongs in its first 196 pulls (1.02% > ε) and was permanently masked by
/// a point-estimate test; it never got another pull to rehabilitate the
/// estimate, and the bandit stuck 42% of its rounds on a suboptimal arm.
/// The z=3 test below condemns only when the wrong count exceeds the ε
/// budget by 3√(nε) — a genuinely violating arm (λ=0.3, 19% tail) crosses
/// that within the grace window, while a 0.25%-tail arm stays feasible with
/// margin (P(false condemnation at n=400) < 0.01%).
struct MaskedUcb1 {
    inner: Ucb1,
    trusted_wrong: Vec<u64>,
    /// Arms below this many pulls are unjudged (optimistically feasible).
    min_samples: u64,
}

impl MaskedUcb1 {
    fn new(k: usize) -> Self {
        Self {
            inner: Ucb1::new(k),
            trusted_wrong: vec![0; k],
            min_samples: 400,
        }
    }
    fn select(&mut self, eps: f64) -> usize {
        let k = self.inner.counts.len();
        for i in 0..k {
            if self.inner.counts[i] == 0 {
                return i;
            }
        }
        let feasible = |i: usize| {
            let n = self.inner.counts[i] as f64;
            if n < self.min_samples as f64 {
                return true;
            }
            let w = self.trusted_wrong[i] as f64;
            w <= eps * n + 3.0 * (n * eps * (1.0 - eps)).sqrt()
        };
        let ln_t = (self.inner.t as f64).ln();
        let mut best: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..k {
            if !feasible(i) {
                continue;
            }
            let exploit = self.inner.sums[i] / self.inner.counts[i] as f64;
            let explore = (2.0f64 * ln_t / self.inner.counts[i] as f64).sqrt();
            let score = exploit + explore;
            if score > best_score {
                best_score = score;
                best = Some(i);
            }
        }
        match best {
            Some(i) => i,
            // Repair path: every judged arm violated ε — pull the least
            // infeasible arm so its estimate can recover (or stay condemned).
            None => (0..k)
                .min_by(|&a, &b| {
                    let ra = self.trusted_wrong[a] as f64 / self.inner.counts[a] as f64;
                    let rb = self.trusted_wrong[b] as f64 / self.inner.counts[b] as f64;
                    ra.partial_cmp(&rb).unwrap()
                })
                .unwrap(),
        }
    }
    fn update(&mut self, arm: usize, reward: f32, trusted_wrong: bool) {
        self.inner.update(arm, reward);
        if trusted_wrong {
            self.trusted_wrong[arm] += 1;
        }
    }
    fn total_trusted_wrong(&self) -> u64 {
        self.trusted_wrong.iter().sum()
    }
    fn total_rounds(&self) -> u64 {
        self.inner.t
    }
    fn pulls(&self) -> Vec<u64> {
        self.inner.counts.clone()
    }
}

/// T4 on the cascade mapping (world B, correct polarity): paired UCB1 vs the
/// fixed λ set.
fn gate_t4_cascade() -> GateResult {
    const LAMBDA_ARMS: [f32; 3] = [0.3, 0.5, 0.7];
    const SEEDS: u32 = 60;
    const TOLERANCE: f64 = 0.01;
    let bank = demo_bank();
    let oracle = OracleVerifier::new();
    let world = World::Decisive;

    let mut fixed_sums = [0.0f64; 3];
    let mut bandit_sum = 0.0f64;
    let mut paired_diff_sum = 0.0f64;
    for seed in 0..SEEDS {
        let trials = sample_world(world, 0xA000 + seed * 16 + world as u32);
        let mut scratch = [0.0f32; N_IND];

        let mut arm_totals = [0.0f64; 3];
        for (arm, &lam) in LAMBDA_ARMS.iter().enumerate() {
            let cascade = MarginGatedCascade::new(
                bank.clone(),
                oracle.clone(),
                TAU_FIRE,
                MarginGate {
                    lambda: lam,
                    polarity: world.correct_polarity(),
                },
            );
            let mut total = 0.0f64;
            for (state, &planted) in trials.states.iter().zip(trials.planted.iter()) {
                oracle.set_truth(planted);
                let (invoked, correct) = match cascade.run(state, &mut scratch) {
                    MarginDecision::Clean => (false, !planted),
                    MarginDecision::Trusted(_) => (false, planted),
                    MarginDecision::Escalated(_) => (true, planted),
                };
                total += cascade_reward(correct, invoked) as f64;
            }
            arm_totals[arm] = total;
            fixed_sums[arm] += total;
        }

        let cascades: Vec<_> = LAMBDA_ARMS
            .iter()
            .map(|&lam| {
                MarginGatedCascade::new(
                    bank.clone(),
                    oracle.clone(),
                    TAU_FIRE,
                    MarginGate {
                        lambda: lam,
                        polarity: world.correct_polarity(),
                    },
                )
            })
            .collect();
        let mut bandit = Ucb1::new(LAMBDA_ARMS.len());
        let mut total = 0.0f64;
        for (state, &planted) in trials.states.iter().zip(trials.planted.iter()) {
            let arm = bandit.select();
            oracle.set_truth(planted);
            let (invoked, correct) = match cascades[arm].run(state, &mut scratch) {
                MarginDecision::Clean => (false, !planted),
                MarginDecision::Trusted(_) => (false, planted),
                MarginDecision::Escalated(_) => (true, planted),
            };
            let r = cascade_reward(correct, invoked);
            bandit.update(arm, r);
            total += r as f64;
        }
        bandit_sum += total;
        paired_diff_sum += total - arm_totals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }

    let n = SEEDS as f64;
    let fixed_means: Vec<f64> = fixed_sums.iter().map(|s| s / n).collect();
    let bandit_mean = bandit_sum / n;
    let best_fixed = fixed_means
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let paired_mean_diff = paired_diff_sum / n;
    let detail = format!(
        "world B cascade: bandit mean {bandit_mean:.1} vs fixed λ means {:?} (best {best_fixed:.1}); paired per-seed diff {paired_mean_diff:+.1} (tolerance −{:.1}% of best)",
        fixed_means
            .iter()
            .map(|m| format!("{m:.1}"))
            .collect::<Vec<_>>(),
        TOLERANCE * 100.0,
    );
    if paired_mean_diff >= -TOLERANCE * best_fixed {
        GateResult::pass("T4b", detail)
    } else {
        GateResult::fail("T4b", detail)
    }
}

/// T4 on the accept mapping (world C): ε-feasibility-MASKED UCB1. The fixed
/// arms' raw rewards are also reported to document the reward-blindness
/// finding (the ε-violating λ ties the best arm unmasked).
fn gate_t4_accept() -> GateResult {
    const LAMBDA_ARMS: [f32; 3] = [0.3, 0.5, 0.7];
    const SEEDS: u32 = 60;
    const TOLERANCE: f64 = 0.01;
    const TAIL: f32 = 0.005;
    let mut fixed_sums = [0.0f64; 3];
    let mut fixed_wrongs = [0.0f64; 3];
    let mut bandit_sum = 0.0f64;
    let mut bandit_wrongs = 0.0f64;
    let mut bandit_rounds = 0.0f64;
    let mut bandit_pulls_sum = [0.0f64; 3];
    let mut paired_diff_sum = 0.0f64;

    for seed in 0..SEEDS {
        let trials = sample_token_world(0xD000 + seed * 16, TAIL);

        let run_arm = |gate: MarginGate| -> (f64, u64) {
            let mut total = 0.0f64;
            let mut wrongs = 0u64;
            for (p, &correct) in trials.probs.iter().zip(trials.correct.iter()) {
                let split = margin_split(p);
                if gate.trusted(split.gap) {
                    total += cascade_reward(correct, false) as f64;
                    wrongs += (!correct) as u64;
                } else {
                    total += cascade_reward(correct, true) as f64;
                }
            }
            (total, wrongs)
        };

        let mut arm_totals = [0.0f64; 3];
        let mut arm_wrongs = [0.0f64; 3];
        for (arm, &lam) in LAMBDA_ARMS.iter().enumerate() {
            let gate = MarginGate {
                lambda: lam,
                polarity: MarginPolarity::DecisiveTrusted,
            };
            let (total, wrongs) = run_arm(gate);
            arm_totals[arm] = total;
            arm_wrongs[arm] = wrongs as f64;
            fixed_sums[arm] += total;
            fixed_wrongs[arm] += wrongs as f64;
        }
        // Per-seed feasibility: a fixed arm is feasible iff ITS regression on
        // this seed's stream is within ε (post hoc, since the fixed arms are
        // the reference policy set, not adaptive agents).
        let best_feasible = arm_totals
            .iter()
            .zip(arm_wrongs.iter())
            .filter(|(_, w)| **w / N_TRIALS as f64 <= EPS_FLAG_ACC)
            .map(|(t, _)| *t)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut bandit = MaskedUcb1::new(LAMBDA_ARMS.len());
        let mut bandit_total = 0.0f64;
        for (p, &correct) in trials.probs.iter().zip(trials.correct.iter()) {
            let arm = bandit.select(EPS_FLAG_ACC);
            let gate = MarginGate {
                lambda: LAMBDA_ARMS[arm],
                polarity: MarginPolarity::DecisiveTrusted,
            };
            let split = margin_split(p);
            let (reward, trusted_wrong) = if gate.trusted(split.gap) {
                (cascade_reward(correct, false), !correct)
            } else {
                (cascade_reward(correct, true), false)
            };
            bandit.update(arm, reward, trusted_wrong);
            bandit_total += reward as f64;
        }
        bandit_sum += bandit_total;
        bandit_wrongs += bandit.total_trusted_wrong() as f64;
        bandit_rounds += bandit.total_rounds() as f64;
        for (acc, c) in bandit_pulls_sum.iter_mut().zip(bandit.pulls()) {
            *acc += c as f64;
        }
        paired_diff_sum += bandit_total - best_feasible;
    }

    let n = SEEDS as f64;
    let fixed_means: Vec<f64> = fixed_sums.iter().map(|s| s / n).collect();
    let fixed_regs: Vec<f64> = fixed_wrongs
        .iter()
        .map(|w| w / (n * N_TRIALS as f64))
        .collect();
    let bandit_mean = bandit_sum / n;
    let bandit_rate = bandit_wrongs / bandit_rounds;
    let bandit_pulls: Vec<u64> = bandit_pulls_sum.iter().map(|c| (c / n) as u64).collect();
    let feasible: Vec<bool> = fixed_regs.iter().map(|r| *r <= EPS_FLAG_ACC).collect();
    let best_feasible_mean = fixed_means
        .iter()
        .zip(feasible.iter())
        .filter(|(_, f)| **f)
        .map(|(m, _)| *m)
        .fold(f64::NEG_INFINITY, f64::max);
    let paired_mean_diff = paired_diff_sum / n;
    let detail = format!(
        "world C accept(tail={TAIL}): fixed λ means {:?} regressions {:?} (feasible: {:?}); \
         masked bandit mean {bandit_mean:.1}, trusted-wrong rate {bandit_rate:.5} (ε={EPS_FLAG_ACC}), mean pulls {bandit_pulls:?}; \
         paired per-seed diff vs best feasible {paired_mean_diff:+.1} (tolerance −{:.1}%)",
        fixed_means
            .iter()
            .map(|m| format!("{m:.1}"))
            .collect::<Vec<_>>(),
        fixed_regs
            .iter()
            .map(|r| format!("{r:.4}"))
            .collect::<Vec<_>>(),
        feasible,
        TOLERANCE * 100.0,
    );
    if bandit_rate <= EPS_FLAG_ACC && paired_mean_diff >= -TOLERANCE * best_feasible_mean {
        GateResult::pass("T4c", detail)
    } else {
        GateResult::fail("T4c", detail)
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Issue 745 / Plan 595 — Margin-Gated Verification Escalation PoC (Bench 711) ===");
    println!(
        "competitors: baseline cascade | margin-gated (both polarities, λ sweep) | always-verify"
    );
    println!(
        "ε (flagged-case accuracy regression) = {EPS_FLAG_ACC}; min qualifying cut = {}%",
        MIN_CUT * 100.0
    );

    let mut gates: Vec<GateResult> = Vec::new();
    gates.extend(gate_cascade_refutation());
    gates.extend(gate_accept_mapping());
    gates.push(gate_g2());
    gates.push(gate_g4());
    gates.push(gate_t4_cascade());
    gates.push(gate_t4_accept());

    println!("\n=== Gates ===");
    let mut all_pass = true;
    for g in &gates {
        let status = if g.passed { "PASS" } else { "FAIL" };
        println!("[{status}] {}: {}", g.name, g.detail);
        if !g.passed {
            all_pass = false;
        }
    }

    println!();
    if all_pass {
        println!(
            "=== ALL GATES PASS — verdicts recorded: cascade mapping REFUTED at ε (worlds A/B), accept mapping VIABLE at tail ≤ 0.5% and ε-sensitive (world C); margin_gate stays OPT-IN pending a live consumer ==="
        );
        std::process::exit(0);
    } else {
        println!("=== ONE OR MORE GATES FAILED ===");
        std::process::exit(1);
    }
}
