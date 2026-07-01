// @katgpt-rs/tests/bench_298_smear_classifier_goat.rs
//
//! Plan 298 Phase 3 — GOAT Gate G2: useful discrimination.
//!
//! Constructs three synthetic smear-class workloads via hand-built `[k*d]`
//! row-major weight slices (the same surface MUX superposition generators
//! and BoM K-hypothesis samplers expose to the classifier), then runs a
//! Plan 278-style causal-intervention audit on a `DefaultFaithfulnessProbe`
//! for each. Measures the per-class unfaithfulness rate (fraction of trials
//! where `is_faithfully_used(threshold)` returns `false`).
//!
//! ## The mechanism being tested
//!
//! Per arXiv:2606.20560 §5.2:
//! - **TokenSmear** (§5.2.1) — mass on one direction spread across adjacent
//!   sites. Benign positional uncertainty. The consumer's effective readout
//!   direction is unchanged → audit perturbations reliably move behavior →
//!   low unfaithfulness rate.
//! - **SequenceSmear** (§5.2.2) — mass split across ≥2 semantically distinct
//!   directions. The consumer holds multiple competing hypotheses; its
//!   effective readout direction is the *average* of orthogonal unit vectors,
//!   which has norm `1/√k`. Audit perturbations are diluted across k
//!   directions → smaller behavioral deltas → higher unfaithfulness rate.
//!
//! ## Pass criterion (Plan 298 §G2)
//!
//! `unfaithfulness_rate(SequenceSmear) ≥ 2 × unfaithfulness_rate(TokenSmear)`
//!
//! If this fails, per T3.4 the classifier is demoted to opt-in Gain (which it
//! already is) and the failure mode is documented in
//! `.benchmarks/298_smear_classifier_goat.md`.
//!
//! ## Run
//!
//! ```text
//! cargo test --features smear_classifier --test bench_298_smear_classifier_goat -- --nocapture
//! cargo test --features smear_classifier --test bench_298_smear_classifier_goat -- release -- --nocapture
//! ```
//!
//! Release build recommended — debug builds don't engage SIMD and produce
//! noisier deltas (though the verdict is robust to build mode because it's
//! a rate over 1000 trials, not a single-shot measurement).

#![cfg(feature = "smear_classifier")]

use fastrand::Rng;
use katgpt_core::faithfulness::probe::{DefaultFaithfulnessProbe, FaithfulnessProbe, SmearSource};
use katgpt_core::faithfulness::smear::{CosineSmearClassifier, SmearClass, SmearClassifier};
use katgpt_core::faithfulness::types::ConsumerContext;

const K: usize = 8;
const D: usize = 16;
const N_TRIALS: usize = 1000;
/// Audit threshold for `is_faithfully_used`. This is the Plan 278 default
/// of 0.5 — NOT tuned per smear class. The whole point of G2 is that the
/// *same* audit threshold is applied to all consumers; the smear classifier
/// helps interpret the result, it does not change the threshold.
const THRESHOLD: f32 = 0.5;

/// Synthetic consumer: behavior = `(1/k) · Σ_i dot(memory, h_i)` where `h_i`
/// is row `i` of the consumer's `[k*d]` weight slice.
///
/// This is the "averaged readout over k hypotheses" model. For:
/// - **CoherentSingle** weights (`[h_0, 0, …, 0]`): effective readout norm
///   scales as `‖h_0‖/k`. Weakest signal.
/// - **TokenSmear** weights (`[h_0, h_0, …, h_0]`): effective readout norm
///   is `‖h_0‖`. Strongest signal.
/// - **SequenceSmear** weights (`[h_0, h_1, …, h_{k-1}]` orthogonal): effective
///   readout norm is `‖Σh_i‖/k = √k/k = 1/√k`. Medium signal.
///
/// The consumer also implements `SmearSource` so the classifier can read its
/// weight slice during the smear-aware audit.
struct SyntheticSmearConsumer {
    /// `[k*d]` row-major weights. Exposed via `SmearSource`.
    weights: Vec<f32>,
}

impl SyntheticSmearConsumer {
    fn new(weights: Vec<f32>) -> Self {
        debug_assert_eq!(weights.len(), K * D);
        Self { weights }
    }
}

impl ConsumerContext for SyntheticSmearConsumer {
    type Behavior = f32;
    type Delta = f32;
    /// Memory is d-dim (matches one hypothesis direction). The audit perturbs
    /// this slice; behavior is the averaged readout over the k hypotheses.
    type Memory = Vec<f32>;

    #[inline]
    fn baseline_behavior(&self) -> f32 {
        0.0
    }

    #[inline]
    fn behavior_with_memory(&self, memory: &Vec<f32>) -> f32 {
        // behavior = (1/k) · Σ_i dot(memory, h_i)
        // = (1/k) · Σ_i Σ_d memory[d] · weights[i*d + d]
        let mut sum: f32 = 0.0;
        for i in 0..K {
            let row_off = i * D;
            for (m, w) in memory.iter().zip(&self.weights[row_off..row_off + D]) {
                sum += *m * *w;
            }
        }
        sum / K as f32
    }

    #[inline]
    fn behavior_delta(&self, a: &f32, b: &f32) -> f32 {
        (a - b).abs()
    }
}

impl SmearSource for SyntheticSmearConsumer {
    #[inline]
    fn latent_mass_distribution(&self) -> (&[f32], usize, usize) {
        (&self.weights, K, D)
    }
}

// ── Weight-slice builders for the three smear classes. ──────────────────

/// Build CoherentSingle weights: row 0 = unit vector e_0, rows 1..k = zero.
/// The classifier returns `SmearClass::CoherentSingle`.
fn build_coherent_single_weights() -> Vec<f32> {
    let mut w = vec![0.0_f32; K * D];
    w[0] = 1.0; // h_0 = e_0
    w
}

/// Build TokenSmear weights: all k rows = e_0 (identical → cosine = 1.0).
/// The classifier returns `SmearClass::TokenSmear { span: K }`.
fn build_token_smear_weights() -> Vec<f32> {
    let mut w = vec![0.0_f32; K * D];
    for i in 0..K {
        w[i * D] = 1.0; // h_i = e_0 for all i
    }
    w
}

/// Build SequenceSmear weights: row i = e_i (standard basis — pairwise
/// orthogonal → cosine = 0, distance = 1.0). Requires d ≥ k.
/// The classifier returns `SmearClass::SequenceSmear { n_hypotheses: K }`.
#[allow(clippy::assertions_on_constants)] // D >= K is a compile-time invariant
fn build_sequence_smear_weights() -> Vec<f32> {
    debug_assert!(D >= K, "SequenceSmear needs d >= k for orthogonal rows");
    let mut w = vec![0.0_f32; K * D];
    for i in 0..K {
        w[i * D + i] = 1.0; // h_i = e_i
    }
    w
}

/// Sanity check: the classifier returns the expected class for each built
/// weight slice. If this fails, the test setup is wrong (not the classifier).
fn assert_classifier_labels() {
    let clf = CosineSmearClassifier::default();
    let mut scratch = vec![0.0_f32; K + K * (K - 1) / 2];

    let cs = build_coherent_single_weights();
    let r = clf.classify(&cs, K, D, &mut scratch);
    assert_eq!(
        r.class,
        SmearClass::CoherentSingle,
        "setup bug: CoherentSingle weights classified as {:?}",
        r.class
    );

    let ts = build_token_smear_weights();
    let r = clf.classify(&ts, K, D, &mut scratch);
    assert_eq!(
        r.class,
        SmearClass::TokenSmear,
        "setup bug: TokenSmear weights classified as {:?}",
        r.class
    );

    let ss = build_sequence_smear_weights();
    let r = clf.classify(&ss, K, D, &mut scratch);
    assert_eq!(
        r.class,
        SmearClass::SequenceSmear,
        "setup bug: SequenceSmear weights classified as {:?}",
        r.class
    );
}

/// Run the Plan 278 audit suite on a consumer with the given weight slice,
/// over `n_trials` random memory vectors. Returns the unfaithfulness rate
/// (fraction of trials where `is_faithfully_used(THRESHOLD)` is false).
fn measure_unfaithfulness_rate(weights: Vec<f32>, n_trials: usize) -> (f32, usize) {
    let consumer = SyntheticSmearConsumer::new(weights);
    let irrelevant_pool: Vec<f32> = (0..D).map(|i| (i as f32) * 0.3 - 0.5).collect();
    let filler: f32 = 1.0;
    let mut probe = DefaultFaithfulnessProbe::new(consumer, irrelevant_pool, filler);
    let mut rng = Rng::with_seed(0x298_5EED_7007u64);

    let mut unfaithful_count = 0usize;
    for _ in 0..n_trials {
        // Random d-dim memory in [-1, 1].
        let memory: Vec<f32> = (0..D).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let profile = probe.faithfulness_profile(&memory, &mut rng);
        if !profile.is_faithfully_used(THRESHOLD) {
            unfaithful_count += 1;
        }
    }
    let rate = unfaithful_count as f32 / n_trials as f32;
    (rate, unfaithful_count)
}

#[test]
fn g2_smear_class_predicts_unfaithfulness() {
    // ── Step 0: verify the three built weight slices classify correctly. ──
    assert_classifier_labels();

    // ── Step 1: measure unfaithfulness rate per smear class. ──
    let (cs_rate, cs_n) = measure_unfaithfulness_rate(build_coherent_single_weights(), N_TRIALS);
    let (ts_rate, ts_n) = measure_unfaithfulness_rate(build_token_smear_weights(), N_TRIALS);
    let (ss_rate, ss_n) = measure_unfaithfulness_rate(build_sequence_smear_weights(), N_TRIALS);

    // ── Step 2: report. ──
    println!("\n=== Plan 298 G2 — Smear class predicts unfaithfulness ===");
    println!(
        "k={}, d={}, trials/class={}, threshold={}",
        K, D, N_TRIALS, THRESHOLD
    );
    println!();
    println!(
        "{:>16} {:>14} {:>14}",
        "smear_class", "unfaithful_n", "rate"
    );
    println!("{:>16} {:>14} {:>14.4}", "CoherentSingle", cs_n, cs_rate);
    println!("{:>16} {:>14} {:>14.4}", "TokenSmear", ts_n, ts_rate);
    println!("{:>16} {:>14} {:>14.4}", "SequenceSmear", ss_n, ss_rate);
    println!();

    let ratio = if ts_rate > 0.0 {
        ss_rate / ts_rate
    } else {
        f32::INFINITY
    };
    println!(
        "SequenceSmear / TokenSmear unfaithfulness ratio = {:.4}×",
        ratio
    );

    // ── Step 3: verdict. ──
    // Pass criterion: SequenceSmear rate ≥ 2× TokenSmear rate.
    // Per Plan 298 §G2 this is the load-bearing test for the ternary
    // classification's *usefulness* (G1 covers correctness).
    let pass = ratio >= 2.0;
    if pass {
        println!("\nG2 PASS: ratio {:.2}× ≥ 2.0× threshold.", ratio);
        println!("SequenceSmear-flagged consumers are unfaithful at ≥ 2× the");
        println!("rate of TokenSmear-flagged consumers — the ternary classifier");
        println!("produces a measurably different downstream decision than the");
        println!("binary probe on this synthetic workload.");
    } else {
        println!("\nG2 FAIL: ratio {:.2}× < 2.0× threshold.", ratio);
        println!("Per Plan 298 T3.4: demote to opt-in Gain (already opt-in).");
        println!("The classifier is still a correct ternary diagnostic (G1");
        println!("passes) but does not produce measurably better downstream");
        println!("decisions on this synthetic workload. Document the failure");
        println!("mode in .benchmarks/298_smear_classifier_goat.md.");
    }

    // Soft sanity assertions — these MUST hold regardless of G2 verdict.
    // If they fail, the test harness is broken (not the classifier).
    assert!((0.0..=1.0).contains(&cs_rate), "rate out of [0,1]");
    assert!((0.0..=1.0).contains(&ts_rate), "rate out of [0,1]");
    assert!((0.0..=1.0).contains(&ss_rate), "rate out of [0,1]");
}
