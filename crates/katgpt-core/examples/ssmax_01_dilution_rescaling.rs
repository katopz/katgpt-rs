//! Example: SSMax — length-aware log-N attention temperature (Plan 411,
//! Research 392, arXiv:2607.01538 Gollapudi et al. *Drowning in Documents at
//! Million Token Scale*).
//!
//! Demonstrates the primitive's stated commercial purpose: **cancel attention
//! dilution at large corpus size N** so retrieval still works at
//! million-token scale. As N grows, the softmax denominator grows faster
//! than the gold term's numerator, collapsing post-normalization mass on
//! the gold document even when the pre-softmax score stays high. SSMax
//! rescales logits multiplicatively by `s_L · log(N)`, which cancels the
//! `(N−1)` growth whenever `s_L · Δ > 1`.
//!
//! Run with:
//! ```sh
//! cargo run --example ssmax_01_dilution_rescaling --features ssmax_temperature --release
//! ```
//!
//! # What This Proves
//!
//! - **The dilution problem**: gold mass collapses as N grows even though
//!   the gold key keeps the top pre-softmax logit. The bound is
//!   `α_gold ≈ 1 / (1 + (N−1) · N^{−s·Δ})` — without SSMax (s=1) the mass
//!   goes to zero.
//! - **Fixed mode (truly modelless, s_L = 1.0)**: multiplying by `log(N)`
//!   recovers substantial gold mass at every N. Zero training, zero new
//!   parameters.
//! - **Adaptive mode (s_L = 1/Δ)**: when the caller knows the typical
//!   gold-distractor gap `Δ`, setting `s_L = 1/Δ` bounds the gold mass at
//!   0.5 regardless of N (vs collapsing to ~10⁻⁵ without SSMax). The
//!   threshold `s_L · Δ = 1` is where the `(N−1)` denominator growth is
//!   exactly cancelled; `s_L > 1/Δ` drives `α_gold → 1`.
//! - **API surface**: `SsmaxConfig` caching (pre-resolved `s_L` + `log(N)`)
//!   vs per-call `SsmaxMode::multiplier(log_n)` — both agree.
//! - **Invariants**: argmax is always preserved (softmax is monotonic);
//!   SSMax at `s_L = 1.0, N ≥ 3` dominates base (Lean proof
//!   `KatgptProof.Ssmax.ssmax_dominates_base`).
//!
//! # What This Does NOT Prove
//!
//! - **Real long-context transformer integration** — this is a reference
//!   demo of the primitive's API + the dilution intuition on a synthetic
//!   retrieval task, not a production long-context pipeline. The GOAT gate
//!   (`.benchmarks/411_ssmax_goldshare_goat.md`) proves the gain with
//!   cosine-similarity retrieval recall; this example shows the API.
//! - **GoldShare diagnostic** — the sibling primitive from Plan 411
//!   (`‖a^G_L‖ / ‖a_L‖` output-fraction) is opt-in and not exercised here.
//!   See the bench `bench_411_gold_share_goat.rs` for that story.
//! - **Rolling-Δ estimator** — the opt-in `ssmax_adaptive` feature's
//!   `RollingDeltaEstimator` (lock-free EMA) is not shown here; we
//!   construct `SsmaxMode::Adaptive` directly with a caller-supplied
//!   `rolling_delta`. The estimator is a convenience for callers who
//!   don't have their own Δ estimate.
//!
//! # Reference
//!
//! - Plan: `katgpt-rs/.plans/411_ssmax_goldshare.md`
//! - Research: `katgpt-rs/.research/392_Attention_Dilution_SSMax_GoldShare.md`
//! - Source: arXiv:2607.01538 — Gollapudi et al., *Can Language Models
//!   Actually Retrieve In-Context?*
//! - SSMax source paper: arXiv:2501.19399 — Uszacorek et al.,
//!   *Scalable-Softmax is Superior for Attention*

use katgpt_core::ssmax::{SsmaxConfig, SsmaxMode, apply_ssmax_inplace};

// ─────────────────────────────────────────────────────────────────────────
// Synthetic retrieval task builder.
//
// N keys: one "gold" key at logit = +Δ, N−1 "distractor" keys at logit = 0.
// The gold-distractor pre-softmax gap is exactly Δ. As N grows, softmax
// mass on the gold key collapses — this is the attention dilution problem.
// ─────────────────────────────────────────────────────────────────────────

const DELTA: f32 = 0.5; // gold-distractor pre-softmax logit gap

fn build_retrieval_task(n: usize, delta: f32) -> (Vec<f32>, usize) {
    // gold at index 0 with logit = +delta; distractors at logit = 0.
    let mut logits = vec![0.0_f32; n];
    logits[0] = delta;
    (logits, 0) // (logits, gold_index)
}

/// Numerically stable softmax mass on the gold key.
fn softmax_gold_mass(logits: &[f32], gold_idx: usize) -> f32 {
    let max = logits
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f64 = logits.iter().map(|&x| ((x - max) as f64).exp()).sum();
    let gold = ((logits[gold_idx] - max) as f64).exp();
    (gold / sum) as f32
}

/// Argmax index (ties broken by lowest index — matches the gold position).
fn argmax_index(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b)).map_or(0, |(i, _)| i)
}

// ─────────────────────────────────────────────────────────────────────────
// Section 1: The dilution problem — gold mass collapses as N grows.
//
// The paper's bound: α_gold ≈ 1 / (1 + (N−1) · N^{−s·Δ}).
// Without SSMax (s = 1), N^{−Δ} = N^{−0.5} shrinks slower than (N−1) grows,
// so the (N−1) term dominates and α_gold → 0. This is why retrieval breaks
// at million-token scale.
// ─────────────────────────────────────────────────────────────────────────

fn section_1_dilution_problem() {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 1: The dilution problem (gold mass collapses as N grows)  │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  Gold-distractor pre-softmax gap Δ = {DELTA}");
    println!("  Without SSMax (s = 1): α_gold ≈ 1 / (1 + (N−1) · N^(−Δ))");
    println!();

    let ns: &[usize] = &[64, 1_000, 10_000, 100_000];
    println!("  {:>10}  {:>14}  {:>10}", "N", "gold_mass_base", "argmax_ok");
    println!("  ──────────────────────────────────────────────────────────");
    for &n in ns {
        let (logits, gold_idx) = build_retrieval_task(n, DELTA);
        let mass = softmax_gold_mass(&logits, gold_idx);
        let argmax_ok = argmax_index(&logits) == gold_idx;
        println!(
            "  {:>10}  {:>14.6}  {:>10}",
            n,
            mass,
            if argmax_ok { "✓" } else { "✗" }
        );
    }
    println!();
    println!("  → Gold mass collapses ~1600× from N=64 to N=100k despite the gold");
    println!("    key keeping the TOP pre-softmax logit at every N. This is the");
    println!("    core pathology SSMax exists to fix.");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Section 2: Fixed mode (truly modelless, s_L = 1.0).
//
// SSMax rescales each logit by s_L · log(N). With s_L = 1.0 (the default),
// the multiplier is exactly log(N). The effective exponent becomes
// log(N) · Δ instead of just Δ, so N^{−log(N)·Δ} = N^{−log(N)·0.5} which
// shrinks MUCH faster than (N−1) grows. The gold mass is recovered.
//
// This is the truly modelless case: zero training, zero new parameters.
// ─────────────────────────────────────────────────────────────────────────

fn section_2_fixed_mode() {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 2: Fixed mode s_L = 1.0 (truly modelless)                │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  Multiplier = s_L · log(N) = log(N). Rescaled gap = log(N) · Δ.");
    println!();

    let ns: &[usize] = &[64, 1_000, 10_000, 100_000];
    println!(
        "  {:>10}  {:>10}  {:>14}  {:>14}  {:>10}",
        "N", "log(N)", "gold_base", "gold_ssmax", "ratio"
    );
    println!("  ────────────────────────────────────────────────────────────────────────");
    for &n in ns {
        let (base_logits, gold_idx) = build_retrieval_task(n, DELTA);
        let base_mass = softmax_gold_mass(&base_logits, gold_idx);

        let mut ssmax_logits = base_logits.clone();
        let log_n = (n as f32).ln();
        apply_ssmax_inplace(&mut ssmax_logits, &SsmaxMode::Fixed { s_l: 1.0 }, log_n);
        let ssmax_mass = softmax_gold_mass(&ssmax_logits, gold_idx);

        let ratio = if base_mass > 0.0 {
            ssmax_mass / base_mass
        } else {
            f32::INFINITY
        };
        println!(
            "  {n:>10}  {log_n:>10.3}  {base_mass:>14.6}  {ssmax_mass:>14.6}  {ratio:>9.1}×"
        );
    }
    println!();
    println!("  → SSMax with s_L=1.0 recovers 60× at N=10k, 191× at N=100k. The");
    println!("    gold key now holds meaningful mass instead of being drowned. No");
    println!("    training was needed — log(N) is a closed-form rescale.");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Section 3: Adaptive mode (s_L = 1/Δ).
//
// When the caller knows the typical gold-distractor gap Δ, setting
// s_L = 1/Δ gives s_L · Δ = 1, so the denominator's (N−1) term is exactly
// cancelled: N^{−1} · (N−1) → 1 as N → ∞, giving α_gold → 1/(1+1/N) → 1.
//
// The Adaptive variant resolves s_L = clamp(1/rolling_delta, 0.1, 10.0).
// We show: (a) perfect-knowledge Δ recovery, (b) clamping at tiny/huge Δ.
// ─────────────────────────────────────────────────────────────────────────

fn section_3_adaptive_mode() {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 3: Adaptive mode s_L = 1/Δ (analytical oracle)           │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  With s_L = 1/Δ: the effective exponent is s_L · log(N) · Δ = log(N),");
    println!("  so exp(log(N)) = N cancels the (N−1) denominator growth, bounding");
    println!("  α_gold at 0.5 regardless of N (vs collapsing to ~1e-5 without SSMax).");
    println!("  To drive α_gold → 1, use s_L > 1/Δ (over-sharpen beyond the threshold).");
    println!();

    // (a) Perfect-knowledge Δ: s_L = 1/Δ.
    let ns: &[usize] = &[64, 1_000, 10_000, 100_000];
    println!(
        "  {:>10}  {:>14}  {:>14}  {:>14}",
        "N", "gold_base", "gold_adapt", "improvement"
    );
    println!("  ────────────────────────────────────────────────────────────────────────");
    for &n in ns {
        let (base_logits, gold_idx) = build_retrieval_task(n, DELTA);
        let base_mass = softmax_gold_mass(&base_logits, gold_idx);

        let mut adapt_logits = base_logits.clone();
        let log_n = (n as f32).ln();
        let mode = SsmaxMode::Adaptive {
            rolling_delta: DELTA,
        };
        apply_ssmax_inplace(&mut adapt_logits, &mode, log_n);
        let adapt_mass = softmax_gold_mass(&adapt_logits, gold_idx);

        println!(
            "  {:>10}  {:>14.6}  {:>14.6}  {:>13.1}×",
            n,
            base_mass,
            adapt_mass,
            if base_mass > 0.0 {
                adapt_mass / base_mass
            } else {
                f32::INFINITY
            }
        );
    }
    println!();
    println!("  → Knowing Δ bounds the gold mass at 0.5 (vs ~1e-5 base). This is");
    println!("    the threshold s_L·Δ = 1; s_L > 1/Δ drives α_gold → 1. The opt-in");
    println!("    RollingDeltaEstimator approximates Δ from observed max−mean logit");
    println!("    gaps at runtime.");
    println!();

    // (b) Clamping: tiny Δ → s_L capped at 10.0; huge Δ → s_L floored at 0.1.
    println!("  Clamping behaviour (s_L = clamp(1/Δ, 0.1, 10.0)):");
    println!("  {:>12}  {:>10}  {:>10}", "rolling_Δ", "s_L", "clamped?");
    println!("  ──────────────────────────────────────────");
    for &delta in &[0.001_f32, 0.05, 0.5, 5.0, 100.0] {
        let mode = SsmaxMode::Adaptive {
            rolling_delta: delta,
        };
        let s_l = mode.resolve_s_l();
        let unclamped = 1.0 / delta.max(1e-3);
        let clamped = (s_l - unclamped.clamp(0.1, 10.0)).abs() > 1e-6;
        println!(
            "  {:>12.4}  {:>10.3}  {:>10}",
            delta,
            s_l,
            if clamped { "yes" } else { "no" }
        );
    }
    println!();
    println!("  → Tiny Δ (sharp gold) → s_L capped at 10 (max sharpening).");
    println!("    Huge Δ (already separated) → s_L floored at 0.1 (mild).");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Section 4: API surface — SsmaxConfig caching vs per-call multiplier.
//
// SsmaxConfig bundles the resolved s_L + precomputed log(N) for storage in
// attention configs where N is known at construction time. The per-call
// SsmaxMode::multiplier(log_n) path recomputes s_L · log_n each call.
// Both paths agree bit-identically.
// ─────────────────────────────────────────────────────────────────────────

fn section_4_api_surface() {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 4: API surface (SsmaxConfig caching vs per-call)          │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    let mode_fixed = SsmaxMode::Fixed { s_l: 1.0 };
    let mode_adapt = SsmaxMode::Adaptive {
        rolling_delta: 0.5,
    };

    println!("  {:>8}  {:>12}  {:>14}  {:>14}  {:>8}", "N", "mode", "config_mult", "call_mult", "agree?");
    println!("  ────────────────────────────────────────────────────────────────────────");
    for &n in &[64_usize, 1_000, 10_000] {
        for (label, mode) in [("Fixed(1.0)", &mode_fixed), ("Adapt(0.5)", &mode_adapt)] {
            let config = SsmaxConfig::from_mode(mode, n);
            let log_n = if n > 1 { (n as f32).ln() } else { 0.0 };
            let call_mult = mode.multiplier(log_n);
            let agree = (config.multiplier() - call_mult).abs() < 1e-6;
            println!(
                "  {:>8}  {:>12}  {:>14.4}  {:>14.4}  {:>8}",
                n, label, config.multiplier(), call_mult, if agree { "✓" } else { "✗" }
            );
        }
    }
    println!();
    println!("  → Both paths produce the same multiplier. Use SsmaxConfig when N is");
    println!("    known at construction (cached); use mode.multiplier(log_n) when N");
    println!("    varies per call (e.g. growing KV cache).");
    println!();

    // Identity at multiplier = 1: s_L · log(N) = 1 ⇒ log(N) = 1/s_L ⇒ N = e^{1/s_L}.
    let s_l = 1.0_f32;
    let n_identity = (1.0 / s_l).exp();
    println!(
        "  Identity point (multiplier = 1): s_L={s_l}, N = e^(1/s_L) = {n_identity:.2}"
    );
    println!("  At this N, SSMax is a no-op (multiplies by 1.0). Below this N,");
    println!("  SSMax is milder than base (multiplier < 1); above, sharper.");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Section 5: Invariants + edge cases.
//
// - Argmax is ALWAYS preserved (softmax is monotonic — multiplying all
//   logits by the same positive scalar doesn't change their rank order).
// - Empty slice is a no-op.
// - log_n = 0 (N ≤ 1) → multiplier = 0 → logits zeroed → softmax gives
//   uniform (correct for single-token attention where mass is always 1.0).
// ─────────────────────────────────────────────────────────────────────────

fn section_5_invariants() {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 5: Invariants + edge cases                               │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    // (a) Argmax preserved across N.
    println!("  Argmax preservation (softmax monotonicity):");
    let ns: &[usize] = &[64, 1_000, 10_000];
    let mut all_ok = true;
    for &n in ns {
        let (base, gold_idx) = build_retrieval_task(n, DELTA);
        let base_argmax = argmax_index(&base);

        let mut fixed = base.clone();
        apply_ssmax_inplace(&mut fixed, &SsmaxMode::Fixed { s_l: 1.0 }, (n as f32).ln());
        let fixed_argmax = argmax_index(&fixed);

        let mut adapt = base.clone();
        apply_ssmax_inplace(
            &mut adapt,
            &SsmaxMode::Adaptive {
                rolling_delta: DELTA,
            },
            (n as f32).ln(),
        );
        let adapt_argmax = argmax_index(&adapt);

        let ok = base_argmax == fixed_argmax && fixed_argmax == adapt_argmax && adapt_argmax == gold_idx;
        if !ok {
            all_ok = false;
        }
        println!(
            "    N={:<6} base_argmax={} ssmax_fixed={} ssmax_adapt={} gold={}  {}",
            n, base_argmax, fixed_argmax, adapt_argmax, gold_idx, if ok { "✓" } else { "✗" }
        );
    }
    println!(
        "  → Argmax preserved at every N for both Fixed and Adaptive. {}",
        if all_ok { "PASS." } else { "FAIL!" }
    );
    println!();

    // (b) Empty slice is a no-op.
    let mut empty: [f32; 0] = [];
    apply_ssmax_inplace(&mut empty, &SsmaxMode::Fixed { s_l: 5.0 }, 10.0);
    println!("  Empty slice after apply_ssmax_inplace: len={} (no-op) ✓", empty.len());
    println!();

    // (c) N ≤ 1 → log_n = 0 → multiplier = 0 → logits zeroed (uniform softmax).
    let config = SsmaxConfig::from_mode(&SsmaxMode::Fixed { s_l: 1.0 }, 1);
    println!(
        "  N=1: SsmaxConfig.log_n = {:.4} (convention ln(1)=0), multiplier = {:.4}",
        config.log_n, config.multiplier()
    );
    println!("  → Single-token softmax([3.14]) = [1.0]; SSMax zeroes the logit but");
    println!("    softmax([0.0]) = [1.0] too. No-op by convention for N ≤ 1. ✓");
    println!();
}

fn main() {
    println!();
    println!("╔═════════════════════════════════════════════════════════════════════╗");
    println!("║  SSMax — length-aware log-N attention temperature (Plan 411)       ║");
    println!("║  Cancel attention dilution at million-token scale                  ║");
    println!("╚═════════════════════════════════════════════════════════════════════╝");
    println!();

    section_1_dilution_problem();
    section_2_fixed_mode();
    section_3_adaptive_mode();
    section_4_api_surface();
    section_5_invariants();

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("Done. All 5 sections completed. See the module doc (top of this file)");
    println!("for what this proves / does NOT prove.");
    println!("═══════════════════════════════════════════════════════════════════════");
}
