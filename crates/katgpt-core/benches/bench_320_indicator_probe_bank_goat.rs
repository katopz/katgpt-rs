//! Indicator Probe Bank GOAT gate bench (Plan 320 Phase 4).
//!
//! Exercises the full GOAT gate for the indicator-probe-bank family on a
//! synthetic planted-structure bank (no game semantics, no real
//! direction-vector training). G6 (feature-off zero-overhead) is verified
//! separately via `cargo check --no-default-features` (no bench code compiled
//! in when the feature is off).
//!
//! # Gates measured here
//!
//! - **G1 (indicator-level AU-ROC)**: per-indicator AU-ROC on held-out states.
//!   Pass: all 8 indicators ≥ 0.85.
//! - **G2 (OR-fusion transcript-TPR / turn-FPR)**: sweep `tau_fire`; pass:
//!   transcript-TPR ≥ 0.85 at turn-FPR ≤ 0.05.
//! - **G3 (cascade FPR reduction)**: stub verifier (label-coherence check);
//!   pass: stage-2 reduces turn-FPR by ≥5× at transcript-TPR cost ≤10pp.
//! - **G4 (hot-path latency + alloc-free)**: `project_all_into` + `or_fused_fire`
//!   median <200ns per call; 0 allocs / 100 calls (CountingAllocator).
//! - **G5 (similarity block recovery)**: `cluster(0.6, 0.6)` ARI ≥ 0.9.
//! - **G7 (wire-format integrity)**: flip one byte → `BankLoadError::HashMismatch`.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --bench bench_320_indicator_probe_bank_goat -- --nocapture
//! ```
//!
//! If the dyld/trustd stall hits, run the compiled binary directly:
//!
//! ```bash
//! DYLD_PRINT_STATISTICS=1 target/release/bench_320_indicator_probe_bank_goat-* --nocapture
//! ```

use katgpt_core::pruners::indicator_cascade::{IndicatorCascade, IndicatorVerifier};
use katgpt_core::pruners::indicator_probe_bank::{
    BankLoadError, IndicatorLabel, IndicatorProbeBank,
};
use katgpt_core::pruners::indicator_similarity::IndicatorSimilarityMatrix;
use std::hint::black_box;
use std::sync::Arc;
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

// ─── Synthetic bank: 8 indicators, D=72, planted block structure ────────────

/// The 8 strongest indicators from Zhou et al. 2026 (>0.92 AU-ROC). These are
/// the GENERIC label discriminants only — no game semantics ship here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
enum SyntheticIndicator {
    ActionConcealment = 0,
    MaliciousActionPlanning = 1,
    ObserverSuspicionModeling = 2,
    MisalignmentCoverStory = 3,
    StrategicOmission = 4,
    RecognizedProblemSuppression = 5,
    ErrorCalibration = 6,
    StrategicUnderperformance = 7,
}

impl IndicatorLabel for SyntheticIndicator {
    fn as_u8(&self) -> u8 {
        *self as u8
    }
    fn from_u8(d: u8) -> Option<Self> {
        match d {
            0 => Some(Self::ActionConcealment),
            1 => Some(Self::MaliciousActionPlanning),
            2 => Some(Self::ObserverSuspicionModeling),
            3 => Some(Self::MisalignmentCoverStory),
            4 => Some(Self::StrategicOmission),
            5 => Some(Self::RecognizedProblemSuppression),
            6 => Some(Self::ErrorCalibration),
            7 => Some(Self::StrategicUnderperformance),
            _ => None,
        }
    }
    const COUNT: usize = 8;
}

const D: usize = 72;

/// Deterministic PRNG (xorshift32) so the synthetic bank is reproducible.
fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn rand_f32(state: &mut u32) -> f32 {
    // Uniform-ish in [0, 1).
    (xorshift32(state) as f32) / (u32::MAX as f32)
}

fn rand_normal(state: &mut u32) -> f32 {
    // Box-Muller (two uniforms → one normal). Good enough for synthetic data.
    let u1 = rand_f32(state).max(1e-10);
    let u2 = rand_f32(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

/// Build the synthetic bank with 8 planted directions on D=72. Each direction
/// is a sparse unit vector on a disjoint coordinate axis (every 9th coord)
/// scaled by `DIR_SCALE` so the dot-product signal clearly separates
/// positives (direction + noise) from negatives (noise-only). The threshold
/// is set so a clean-direction positive fires (raw ≈ DIR_SCALE) and a noise-only
/// negative does not (raw ≈ N(0, DIR_SCALE·NOISE_STD)).
const DIR_SCALE: f32 = 4.0;
const NOISE_STD: f32 = 0.5;
fn build_synthetic_bank() -> IndicatorProbeBank<SyntheticIndicator, D> {
    let mut dirs = vec![0.0f32; SyntheticIndicator::COUNT * D];
    // Place each direction on its own disjoint axis (every 9th coordinate),
    // scaled so the per-direction dot is strongly positive on a planted state.
    for i in 0..SyntheticIndicator::COUNT {
        dirs[i * D + i * 9] = DIR_SCALE;
    }
    // Threshold at half the signal: a clean positive raw ≈ DIR_SCALE (sigmoid→~1);
    // a noise-only negative raw ≈ N(0, DIR_SCALE·NOISE_STD) = N(0, 2.0), so
    // sigmoid(N(0,2) - DIR_SCALE/2) is almost always < 0.5.
    let thresholds = vec![DIR_SCALE * 0.5; SyntheticIndicator::COUNT];
    IndicatorProbeBank::new(dirs, thresholds).expect("synthetic bank shape")
}

/// Generate `n_per` positive + negative states for a single indicator.
/// Positives: `direction_i + noise(NOISE_STD)`. Negatives: pure noise(NOISE_STD)
/// (no direction component, so the dot on indicator i's axis is just noise).
/// Returns (states, labels) where labels[k] = true if state k is a positive.
fn generate_states_for_indicator(
    bank: &IndicatorProbeBank<SyntheticIndicator, D>,
    indicator_idx: usize,
    n_per: usize,
    seed: u32,
) -> (Vec<[f32; D]>, Vec<bool>) {
    let mut state = seed.wrapping_add(indicator_idx as u32 * 7919);
    let dir = bank.direction(indicator_idx);
    let mut states = Vec::with_capacity(2 * n_per);
    let mut labels = Vec::with_capacity(2 * n_per);
    for _ in 0..n_per {
        // Positive: direction + noise.
        let mut s = [0.0f32; D];
        for d in 0..D {
            s[d] = dir[d] + NOISE_STD * rand_normal(&mut state);
        }
        states.push(s);
        labels.push(true);
        // Negative: pure noise (same std; no direction component).
        let mut n = [0.0f32; D];
        for n_d in n.iter_mut().take(D) {
            *n_d = NOISE_STD * rand_normal(&mut state);
        }
        states.push(n);
        labels.push(false);
    }
    (states, labels)
}

// ─── AU-ROC (trapezoidal sweep) ────────────────────────────────────────────

/// Compute AU-ROC from per-example (score, label) pairs.
/// `labels[k] = true` means positive. Uses the rank-counting identity:
///   AUROC = (Σ_{i∈pos, j∈neg} [score_i > score_j] + 0.5·[score_i == score_j])
///           / (n_pos · n_neg).
fn au_roc(scores: &[f32], labels: &[bool]) -> f64 {
    let mut pos: Vec<f32> = scores
        .iter()
        .zip(labels)
        .filter(|&(_, &l)| l)
        .map(|(s, _)| *s)
        .collect();
    let mut neg: Vec<f32> = scores
        .iter()
        .zip(labels)
        .filter(|&(_, &l)| !l)
        .map(|(s, _)| *s)
        .collect();
    pos.sort_by(|a, b| a.total_cmp(b));
    neg.sort_by(|a, b| a.total_cmp(b));
    // Count pairs with pos > neg via two-pointer walk over the sorted neg array.
    // For each pos score (ascending), find how many neg scores are strictly
    // smaller, plus 0.5 for ties.
    let mut total = 0.0f64;
    let mut neg_idx = 0usize;
    let n_neg = neg.len();
    for &p in &pos {
        // Advance neg_idx while neg[neg_idx] < p.
        while neg_idx < n_neg && neg[neg_idx] < p {
            neg_idx += 1;
        }
        // neg[0..neg_idx] are strictly smaller than p.
        let strictly_smaller = neg_idx as f64;
        // Count ties: neg[neg_idx..] equal to p.
        let mut tie_idx = neg_idx;
        while tie_idx < n_neg && neg[tie_idx] == p {
            tie_idx += 1;
        }
        let ties = (tie_idx - neg_idx) as f64;
        total += strictly_smaller + 0.5 * ties;
    }
    let n_pos = pos.len() as f64;
    let n_neg_f = n_neg as f64;
    if n_pos == 0.0 || n_neg_f == 0.0 {
        return 0.5;
    }
    total / (n_pos * n_neg_f)
}

// ─── G1: indicator-level AU-ROC ─────────────────────────────────────────────

fn gate_g1_indicator_au_roc() -> GateResult {
    let bank = build_synthetic_bank();
    let n_per = 500; // 500 pos + 500 neg per indicator = 1000 each
    let mut worst = 1.0f64;
    let mut per_indicator = Vec::with_capacity(SyntheticIndicator::COUNT);
    for i in 0..SyntheticIndicator::COUNT {
        let (states, labels) = generate_states_for_indicator(&bank, i, n_per, 0xBADC0FFE);
        let scores: Vec<f32> = states
            .iter()
            .map(|s| {
                let label = SyntheticIndicator::from_u8(i as u8).unwrap();
                bank.project(s, label)
            })
            .collect();
        let auc = au_roc(&scores, &labels);
        per_indicator.push((i, auc));
        if auc < worst {
            worst = auc;
        }
    }
    let threshold = 0.85;
    if worst >= threshold {
        let detail = per_indicator
            .iter()
            .map(|(i, a)| format!("ind{i}={a:.3}"))
            .collect::<Vec<_>>()
            .join(", ");
        GateResult::pass(
            "G1",
            format!("all 8 indicators AU-ROC ≥ {threshold:.2}; worst = {worst:.3}; {detail}"),
        )
    } else {
        let detail = per_indicator
            .iter()
            .map(|(i, a)| format!("ind{i}={a:.3}"))
            .collect::<Vec<_>>()
            .join(", ");
        GateResult::fail(
            "G1",
            format!("worst AU-ROC {worst:.3} < {threshold:.2}; {detail}"),
        )
    }
}

// ─── G2: OR-fusion transcript-TPR / turn-FPR ───────────────────────────────

/// Generate a transcript-like eval set. Positives are **misalignment clusters**:
/// each fires TWO indicators (a planted pair) so the stage-2 label-coherence
/// verifier has a corroborating second signal to confirm. Negatives are pure
/// noise (no planted direction), so any firing is a single spurious spike that
/// the verifier can reject.
fn generate_transcript(
    bank: &IndicatorProbeBank<SyntheticIndicator, D>,
    n_total: usize,
    n_positive: usize,
    seed: u32,
) -> (Vec<[f32; D]>, Vec<bool>) {
    let mut state = seed;
    let mut states = Vec::with_capacity(n_total);
    let mut labels = Vec::with_capacity(n_total);
    // Positives: a planted pair of indicators (i, i^1) both fire — sum their
    // directions + noise. This mirrors the paper's premise that a genuine
    // misalignment event shows CORRELATED indicator activation (a cluster),
    // not a single isolated spike.
    for _ in 0..n_positive {
        let i = (xorshift32(&mut state) as usize) % SyntheticIndicator::COUNT;
        let j = i ^ 1; // adjacent pair (block partner)
        let dir_i = bank.direction(i);
        let dir_j = bank.direction(j);
        let mut s = [0.0f32; D];
        for d in 0..D {
            s[d] = dir_i[d] + dir_j[d] + NOISE_STD * rand_normal(&mut state);
        }
        states.push(s);
        labels.push(true);
    }
    // Negatives: pure noise (no direction component → any firing is spurious).
    for _ in 0..(n_total - n_positive) {
        let mut n = [0.0f32; D];
        for n_d in n.iter_mut().take(D) {
            *n_d = NOISE_STD * rand_normal(&mut state);
        }
        states.push(n);
        labels.push(false);
    }
    (states, labels)
}

/// Sweep tau_fire; find the smallest tau achieving turn-FPR ≤ 0.05 and report
/// transcript-TPR there. transcript-TPR = fraction of positive transcripts
/// where the bank fires (any label). turn-FPR = fraction of negative
/// transcripts where the bank fires (false alarm).
fn gate_g2_or_fusion_tpr_fpr() -> GateResult {
    let bank = build_synthetic_bank();
    let (states, labels) = generate_transcript(&bank, 1000, 50, 0xC0FFEE);
    let mut scores = [0.0f32; SyntheticIndicator::COUNT];

    // For each state, compute the max sigmoid score across indicators.
    let max_scores: Vec<f32> = states
        .iter()
        .map(|s| {
            bank.project_all_into(s, &mut scores);
            scores.iter().copied().fold(0.0f32, f32::max)
        })
        .collect();

    // Sweep tau from 0.5 to 0.99; find the largest transcript-TPR achievable
    // at turn-FPR ≤ 0.05.
    let mut best_tpr = 0.0f64;
    let mut best_tau = 0.0f32;
    let mut best_fpr = 1.0f64;
    let fpr_budget = 0.05;
    let mut steps = 0u32;
    let mut tau = 0.5f32;
    while tau < 1.0 {
        let mut tp = 0u64;
        let mut fp = 0u64;
        let mut n_pos = 0u64;
        let mut n_neg = 0u64;
        for (i, &mx) in max_scores.iter().enumerate() {
            if labels[i] {
                n_pos += 1;
                if mx > tau {
                    tp += 1;
                }
            } else {
                n_neg += 1;
                if mx > tau {
                    fp += 1;
                }
            }
        }
        let tpr = tp as f64 / n_pos as f64;
        let fpr = fp as f64 / n_neg as f64;
        if fpr <= fpr_budget && tpr > best_tpr {
            best_tpr = tpr;
            best_tau = tau;
            best_fpr = fpr;
        }
        tau += 0.01;
        steps += 1;
    }
    let _ = steps;

    if best_tpr >= 0.85 {
        GateResult::pass(
            "G2",
            format!(
                "transcript-TPR = {best_tpr:.3} at turn-FPR = {best_fpr:.3} (≤ 0.05), tau = {best_tau:.2}"
            ),
        )
    } else {
        GateResult::fail(
            "G2",
            format!(
                "best transcript-TPR = {best_tpr:.3} < 0.85 at turn-FPR ≤ 0.05 (best tau = {best_tau:.2}, FPR = {best_fpr:.3})"
            ),
        )
    }
}

// ─── G3: cascade FPR reduction ─────────────────────────────────────────────

/// Label-coherence stub verifier: confirm only if a SECOND indicator also
/// fires above `tau_fire` (a corroborating spike). True positives are
/// 2-indicator clusters (both spike to ≈1.0); false positives are single
/// spurious spikes (the 2nd-highest sits at the noise floor ≈0.5). This is
/// intentionally simple (not a real LLM judge) — per the plan's risk note,
/// real-world FPR reduction is measured in riir-ai Phase 2.
struct LabelCoherenceVerifier {
    tau_fire: f32,
}

impl IndicatorVerifier<SyntheticIndicator> for LabelCoherenceVerifier {
    fn verify(&self, _label: SyntheticIndicator, scores: &[f32]) -> bool {
        // Second-highest score (the corroborating indicator).
        let mut top2 = [0.0f32; 2];
        for &s in scores {
            if s > top2[0] {
                top2[1] = top2[0];
                top2[0] = s;
            } else if s > top2[1] {
                top2[1] = s;
            }
        }
        top2[1] > self.tau_fire
    }
}

fn gate_g3_cascade_fpr_reduction() -> GateResult {
    let bank = build_synthetic_bank();
    // Use a denser transcript so stage-1 has meaningful FPR to reduce.
    let (states, labels) = generate_transcript(&bank, 1000, 200, 0xFACE);
    let n_pos = labels.iter().filter(|&&l| l).count() as u64;
    let n_neg = labels.iter().filter(|&&l| !l).count() as u64;

    // Sweep tau_fire to find an operating point where stage-1 FPR is moderate
    // (room for stage-2 to reduce) and stage-2 (label-coherence verifier
    // requiring a 2nd indicator above the SAME tau_fire) cuts FPR by ≥5× at
    // ≤10pp TPR cost. True positives have 2 indicators spiking to ≈1.0, so any
    // tau_fire < 1.0 confirms them; false positives are single spikes whose
    // 2nd-highest sits near the noise floor.
    let mut best_ratio = 0.0f64;
    let mut best_tau = 0.0f32;
    let mut best_fpr_s1 = 1.0f64;
    let mut best_fpr_s2 = 1.0f64;
    let mut best_tpr_s1 = 0.0f64;
    let mut best_tpr_s2 = 0.0f64;

    let bank_arc: Arc<IndicatorProbeBank<SyntheticIndicator, D>> = Arc::new(bank);

    let mut tau = 0.5f32;
    while tau < 0.99 {
        let verifier: Arc<dyn IndicatorVerifier<SyntheticIndicator>> =
            Arc::new(LabelCoherenceVerifier { tau_fire: tau });
        let cascade = IndicatorCascade::new(bank_arc.clone(), verifier, tau);
        let mut fpr_s1 = 0u64;
        let mut tpr_s1 = 0u64;
        let mut fpr_s2 = 0u64;
        let mut tpr_s2 = 0u64;
        let mut scores = [0.0f32; SyntheticIndicator::COUNT];
        let mut cascade_scratch = [0.0f32; SyntheticIndicator::COUNT];
        for (i, s) in states.iter().enumerate() {
            // Stage-1 alone.
            bank_arc.project_all_into(s, &mut scores);
            let fires_s1 = bank_arc.or_fused_fire(&scores, tau).is_some();
            if fires_s1 {
                if labels[i] {
                    tpr_s1 += 1;
                } else {
                    fpr_s1 += 1;
                }
            }
            // Stage-2 (cascade) — exercises the shipped IndicatorCascade::run API.
            if cascade.run(s, &mut cascade_scratch).is_some() {
                if labels[i] {
                    tpr_s2 += 1;
                } else {
                    fpr_s2 += 1;
                }
            }
        }
        let fpr_s1_r = fpr_s1 as f64 / n_neg as f64;
        let tpr_s1_r = tpr_s1 as f64 / n_pos as f64;
        let fpr_s2_r = fpr_s2 as f64 / n_neg as f64;
        let tpr_s2_r = tpr_s2 as f64 / n_pos as f64;
        let ratio = if fpr_s2_r > 0.0 {
            fpr_s1_r / fpr_s2_r
        } else if fpr_s1_r > 0.0 {
            // Stage-2 drove FPR to exactly 0 — treat as a strong win (use a
            // large sentinel so the ≥5× check passes, but cap so the detail
            // line stays readable).
            100.0
        } else {
            0.0
        };
        let tpr_cost = tpr_s1_r - tpr_s2_r;
        if ratio >= 5.0 && tpr_cost <= 0.10 && ratio > best_ratio {
            best_ratio = ratio;
            best_tau = tau;
            best_fpr_s1 = fpr_s1_r;
            best_fpr_s2 = fpr_s2_r;
            best_tpr_s1 = tpr_s1_r;
            best_tpr_s2 = tpr_s2_r;
        }
        tau += 0.02;
    }

    if best_ratio >= 5.0 {
        GateResult::pass(
            "G3",
            format!(
                "cascade reduces turn-FPR {:.3}→{:.3} ({:.1}× reduction) at tau={:.2}; transcript-TPR {:.3}→{:.3} (cost {:.1}pp ≤ 10pp)",
                best_fpr_s1,
                best_fpr_s2,
                best_ratio,
                best_tau,
                best_tpr_s1,
                best_tpr_s2,
                (best_tpr_s1 - best_tpr_s2) * 100.0
            ),
        )
    } else {
        GateResult::fail(
            "G3",
            format!(
                "best cascade FPR reduction {best_ratio:.1}× (need ≥5×); at best tau={best_tau:.2}: FPR {best_fpr_s1:.3}→{best_fpr_s2:.3}, TPR {best_tpr_s1:.3}→{best_tpr_s2:.3}"
            ),
        )
    }
}

// ─── G4: hot-path latency + alloc-free ─────────────────────────────────────

const PERF_ITERS: usize = 1_000_000;
const ALLOC_ITERS: usize = 100;

fn gate_g4_hot_path_latency_and_alloc() -> GateResult {
    let bank = build_synthetic_bank();
    let state = {
        let mut s = [0.0f32; D];
        let dir = bank.direction(0);
        s[..D].copy_from_slice(&dir[..D]);
        s
    };
    let mut scores = [0.0f32; SyntheticIndicator::COUNT];

    // Warmup.
    for _ in 0..1000 {
        bank.project_all_into(black_box(&state), black_box(&mut scores));
        let _ = black_box(bank.or_fused_fire(black_box(&scores), black_box(0.5)));
    }

    // Latency: project_all_into + or_fused_fire.
    let start = Instant::now();
    for _ in 0..PERF_ITERS {
        bank.project_all_into(black_box(&state), black_box(&mut scores));
        let _ = black_box(bank.or_fused_fire(black_box(&scores), black_box(0.5)));
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / PERF_ITERS as f64;

    // Alloc-free.
    let (_, allocs) = alloc_delta(|| {
        for _ in 0..ALLOC_ITERS {
            bank.project_all_into(black_box(&state), black_box(&mut scores));
            let _ = black_box(bank.or_fused_fire(black_box(&scores), black_box(0.5)));
        }
    });

    let latency_pass = ns_per_call < 200.0;
    let alloc_pass = allocs == 0;
    if latency_pass && alloc_pass {
        GateResult::pass(
            "G4",
            format!(
                "project_all_into + or_fused_fire: {ns_per_call:.1} ns/call (target < 200ns); {allocs} allocs / {ALLOC_ITERS} calls"
            ),
        )
    } else {
        let mut reasons = Vec::new();
        if !latency_pass {
            reasons.push(format!("{ns_per_call:.1} ns/call ≥ 200ns"));
        }
        if !alloc_pass {
            reasons.push(format!("{allocs} allocs / {ALLOC_ITERS} calls (need 0)"));
        }
        GateResult::fail("G4", reasons.join("; "))
    }
}

// ─── G5: similarity block recovery (ARI) ───────────────────────────────────

/// Build a bank with planted block structure: 4 blocks of 2 indicators, each
/// block on disjoint 2-dim subspaces with within-block cosine ≈ 0.69.
fn build_block_bank() -> IndicatorProbeBank<SyntheticIndicator, D> {
    let mut dirs = vec![0.0f32; SyntheticIndicator::COUNT * D];
    // 4 blocks, each using 2 coordinates (8 indicators → 8 coords, all disjoint).
    for block in 0..4 {
        let i0 = 2 * block;
        let i1 = 2 * block + 1;
        let c0 = i0; // first coord for this block
        let c1 = i1; // second coord (disjoint)
        dirs[i0 * D + c0] = 1.0;
        dirs[i0 * D + c1] = 0.4;
        dirs[i1 * D + c0] = 0.4;
        dirs[i1 * D + c1] = 1.0;
    }
    let thresholds = vec![0.3f32; SyntheticIndicator::COUNT];
    IndicatorProbeBank::new(dirs, thresholds).expect("block bank shape")
}

/// ARI (same algorithm as the indicator_similarity test module).
fn adjusted_rand_index(a: &[Vec<usize>], b: &[Vec<usize>]) -> f64 {
    let mut all: Vec<usize> = Vec::new();
    for g in a {
        all.extend_from_slice(g);
    }
    for g in b {
        all.extend_from_slice(g);
    }
    all.sort_unstable();
    all.dedup();

    let to_a = |l: usize| -> usize {
        for (ci, g) in a.iter().enumerate() {
            if g.contains(&l) {
                return ci;
            }
        }
        usize::MAX
    };
    let to_b = |l: usize| -> usize {
        for (ci, g) in b.iter().enumerate() {
            if g.contains(&l) {
                return ci;
            }
        }
        usize::MAX
    };
    let na = a.len();
    let nb = b.len();
    let mut nij = vec![0u64; na * nb];
    for &l in &all {
        let ia = to_a(l);
        let ib = to_b(l);
        if ia != usize::MAX && ib != usize::MAX {
            nij[ia * nb + ib] += 1;
        }
    }
    let comb2 = |n: u64| -> f64 {
        if n < 2 {
            0.0
        } else {
            let n = n as f64;
            n * (n - 1.0) / 2.0
        }
    };
    let sum_comb_nij: f64 = nij.iter().map(|&n| comb2(n)).sum();
    let ai_sums: Vec<u64> = (0..na)
        .map(|i| (0..nb).map(|j| nij[i * nb + j]).sum())
        .collect();
    let bj_sums: Vec<u64> = (0..nb)
        .map(|j| (0..na).map(|i| nij[i * nb + j]).sum())
        .collect();
    let sum_comb_a: f64 = ai_sums.iter().map(|&n| comb2(n)).sum();
    let sum_comb_b: f64 = bj_sums.iter().map(|&n| comb2(n)).sum();
    let total: u64 = all.len() as u64;
    let expected = sum_comb_a * sum_comb_b / comb2(total);
    let max_index = 0.5 * (sum_comb_a + sum_comb_b);
    if (max_index - expected).abs() < 1e-12 {
        return 1.0;
    }
    (sum_comb_nij - expected) / (max_index - expected)
}

fn gate_g5_similarity_block_recovery() -> GateResult {
    let bank = build_block_bank();
    let m = IndicatorSimilarityMatrix::<SyntheticIndicator>::from_bank(&bank);
    // cluster returns Vec<Vec<L>>; convert to Vec<Vec<usize>> for ARI.
    let clusters_labels = m.cluster(0.6, 0.6);
    let clusters: Vec<Vec<usize>> = clusters_labels
        .iter()
        .map(|g| g.iter().map(|l| l.as_u8() as usize).collect())
        .collect();
    // Planted: 4 blocks of {0,1},{2,3},{4,5},{6,7}.
    let planted: Vec<Vec<usize>> = (0..4).map(|b| vec![2 * b, 2 * b + 1]).collect();
    let ari = adjusted_rand_index(&clusters, &planted);
    if ari >= 0.9 {
        GateResult::pass(
            "G5",
            format!("cluster(0.6, 0.6) ARI = {ari:.3} ≥ 0.9 vs planted 4×2 blocks"),
        )
    } else {
        GateResult::fail(
            "G5",
            format!("cluster(0.6, 0.6) ARI = {ari:.3} < 0.9; clusters = {clusters:?}"),
        )
    }
}

// ─── G7: wire-format integrity (tamper-evident) ────────────────────────────

fn gate_g7_wire_integrity() -> GateResult {
    let bank = build_synthetic_bank();
    let bytes = bank.to_frozen_bytes();
    // Round-trip OK.
    let reloaded = IndicatorProbeBank::<SyntheticIndicator, D>::from_frozen_bytes(&bytes);
    if reloaded.is_err() {
        return GateResult::fail(
            "G7",
            format!("clean round-trip failed: {:?}", reloaded.err()),
        );
    }
    // Tamper: flip one byte in the directions body.
    let mut tampered = bytes.clone();
    let header_len = 4 + 8 + 2 + 2 + 32; // 48
    tampered[header_len] ^= 0xFF;
    let result = IndicatorProbeBank::<SyntheticIndicator, D>::from_frozen_bytes(&tampered);
    match result {
        Err(BankLoadError::HashMismatch) => GateResult::pass(
            "G7",
            "tampered direction byte correctly rejected with HashMismatch",
        ),
        Err(other) => GateResult::fail("G7", format!("expected HashMismatch, got {other:?}")),
        Ok(_) => GateResult::fail(
            "G7",
            "tampered bytes loaded without error (tamper NOT evident)",
        ),
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 320 — Indicator Probe Bank GOAT Gate (Phase 4) ===\n");

    let gates = [
        gate_g1_indicator_au_roc(),
        gate_g2_or_fusion_tpr_fpr(),
        gate_g3_cascade_fpr_reduction(),
        gate_g4_hot_path_latency_and_alloc(),
        gate_g5_similarity_block_recovery(),
        gate_g7_wire_integrity(),
    ];

    let mut all_pass = true;
    for g in &gates {
        let status = if g.passed { "PASS" } else { "FAIL" };
        println!("[{status}] {}: {}", g.name, g.detail);
        if !g.passed {
            all_pass = false;
        }
    }

    println!();
    println!("G6 (feature-off zero-overhead) is verified via `cargo check --no-default-features`");
    println!("(no indicator_probe_bank code compiled in when the feature is off).");
    println!();
    if all_pass {
        println!("=== ALL GATES PASS — eligible for Phase 5 promotion decision ===");
        std::process::exit(0);
    } else {
        println!("=== ONE OR MORE GATES FAILED — keep opt-in, investigate ===");
        std::process::exit(1);
    }
}
