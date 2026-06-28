//! Plan 335 Phase 3 T3.2 — runnable micro-GPT A/B diagnostic.
//!
//! Reproduces the Phase 2 G4 characterized-bias fixture as a standalone
//! example. Shows the full diagnostic workflow end-to-end:
//!
//! 1. Build two log-probability traces over the same prefixes (a baseline A
//!    and a mechanism-ON B that differs on state-conditioned tokens).
//! 2. Compute [`PairedLossGap`] (per-token `Δ_i = ℓ_A − ℓ_B`).
//! 3. Print the **tag-stratified means table** — raw `Δ̄` per token class.
//! 4. Print the **filtered aggregates table** — `ALL_TOKENS` vs
//!    `TOP-K∩NO-COPY` vs `COPY-N-ONLY`, showing the amplification (paper §6
//!    Figure 7: filtered loss doubles the Transformer–Hybrid separation).
//! 5. Print the **Proposition 1 annotation table** — per-class
//!    `gap_to_bound_ratio = mean_gap / log|V_τ|`, showing which classes are
//!    near their theoretical ceiling vs which have room for a richer feature.
//!
//! # The characterized-bias fixture
//!
//! Random-init micro-GPTs don't exhibit the paper's pattern (it's a trained-
//! model property — see Plan 313 / Issue 003 and the Phase 2 G4 rationale in
//! `.benchmarks/335_paired_loss_goat.md`). This fixture models the
//! characterized bias directly:
//!
//! - **Content / Function** (state-conditioned readout): B is better.
//!   `Δ ~ Normal(+shift, σ)` with `shift ≫ 0`.
//! - **CopyN** (visible-prefix retrieval): both models retrieve equally well.
//!   `Δ ~ Normal(+ε, σ)` with `ε ≈ 0`.
//! - **Other / brackets**: pure noise. `Δ ~ Normal(0, σ)`.
//!
//! This mirrors the Plan 313 / Issue 003 differential signature (the
//! `ac_prefix` doubled-signal bias on copy/position tokens). Real trained-
//! model A/B is a non-blocking riir-train follow-up (same deferral pattern
//! as Plan 313's multi-layer equivalence).
//!
//! Run with: `cargo run --example paired_loss_01_micro_gpt_ab --features paired_loss_diagnostic`

use std::collections::HashMap;

use katgpt_core::{
    ClassGapReport, ClassSizeBound, CopyNGramTagger, FilterKind, FilterScratch, PairedLossGap,
    TokenClass, TokenTagger,
};

// ── Deterministic xorshift RNG (reproducible fixture; no external dep) ──────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f32 {
        let bits = ((self.next() >> 40) as u32 & 0x007f_ffff) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * core::f32::consts::PI * u2;
        r * theta.cos()
    }
}

// ── The characterized-bias fixture (mirrors the Phase 2 G4 bench) ───────────

/// Build two log-prob traces + a class array modeling the Plan 313 / Issue
/// 003 differential signature.
///
/// See module docs for the bias model. Returns `(log_probs_a, log_probs_b,
/// classes)`. The base log-prob for both models is drawn from a realistic
/// NLL range `[-10, 0]`; the gap is the differential.
fn build_characterized_bias_fixture(
    l: usize,
    content_shift: f32,
    function_shift: f32,
    copy_shift: f32,
    noise_sigma: f32,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<TokenClass>) {
    let mut rng = Rng::new(seed);
    let mut a = Vec::with_capacity(l);
    let mut b = Vec::with_capacity(l);
    let mut classes = Vec::with_capacity(l);

    // Token-id sequence with repeated 2-grams so CopyNGramTagger fires.
    // Small vocab so repeats occur naturally.
    let vocab = 32u32;
    let tokens: Vec<u32> = (0..l).map(|i| (i as u32 * 7 + 3) % vocab).collect();
    let copy_tagger = CopyNGramTagger::new(2);

    for i in 0..l {
        // Classify via CopyNGramTagger first; fall through to round-robin
        // open-class assignment for non-copy positions.
        let cls = {
            let raw = copy_tagger.classify(tokens[i], i, &tokens);
            match raw {
                TokenClass::CopyN(_) => raw,
                _ => match i % 5 {
                    0 => TokenClass::Content,
                    1 => TokenClass::Function,
                    2 => TokenClass::Other,
                    3 => TokenClass::BracketOpen,
                    _ => TokenClass::BracketClose,
                },
            }
        };
        classes.push(cls);

        let base = -10.0 + 10.0 * rng.uniform();
        let delta = match cls {
            TokenClass::Content => content_shift + noise_sigma * rng.normal(),
            TokenClass::Function => function_shift + noise_sigma * rng.normal(),
            TokenClass::CopyN(_) => copy_shift + noise_sigma * rng.normal(),
            _ => noise_sigma * rng.normal(),
        };
        // ℓ_A = base + Δ/2, ℓ_B = base − Δ/2 → Δ = ℓ_A − ℓ_B.
        a.push(base + 0.5 * delta);
        b.push(base - 0.5 * delta);
    }
    (a, b, classes)
}

// ── Display helpers ─────────────────────────────────────────────────────────

/// Render a `TokenClass` with its payload for CopyN (e.g. `CopyN(2)`). The
/// `label()` helper omits the n; here we want it visible in the report.
fn class_name(cls: TokenClass) -> String {
    match cls {
        TokenClass::CopyN(n) => format!("CopyN({n})"),
        other => other.label().to_string(),
    }
}

fn print_tag_stratified_table(gap: &PairedLossGap, classes: &[TokenClass]) {
    println!("── Tag-stratified raw means (paper §3 Analysis I) ──");
    println!(
        "  {:<14} {:>8} {:>14}",
        "class", "count", "mean Δ"
    );
    // Distinct classes present, in a stable display order.
    let display_order = [
        TokenClass::Content,
        TokenClass::Function,
        TokenClass::Other,
        TokenClass::BracketOpen,
        TokenClass::BracketClose,
    ];
    for cls in display_order {
        let mean = gap.mean_gap_for_class(classes, cls);
        let count = classes.iter().filter(|c| **c == cls).count();
        if count > 0 {
            println!(
                "  {:<14} {:>8} {:>+14.6}",
                class_name(cls),
                count,
                mean
            );
        }
    }
    // CopyN(n) rows: collect distinct n values in ascending order.
    let mut copy_ns: Vec<u8> = classes
        .iter()
        .filter_map(|c| match c {
            TokenClass::CopyN(n) => Some(*n),
            _ => None,
        })
        .collect();
    copy_ns.sort_unstable();
    copy_ns.dedup();
    for n in copy_ns {
        let target = TokenClass::CopyN(n);
        let mean = gap.mean_gap_for_class(classes, target);
        let count = classes.iter().filter(|c| **c == target).count();
        println!(
            "  {:<14} {:>8} {:>+14.6}",
            class_name(target),
            count,
            mean
        );
    }
    println!();
}

fn print_filtered_aggregates_table(
    gap: &PairedLossGap,
    classes: &[TokenClass],
    scratch: &mut FilterScratch,
) {
    println!("── Filtered aggregates (paper §6) ──");
    let all_tokens = gap.filtered_mean_with_scratch(classes, FilterKind::AllTokens, scratch);
    let topk_nocopy = gap.filtered_mean_with_scratch(
        classes,
        FilterKind::TopKNoCopy {
            k: 10,
            max_ngram: 4,
        },
        scratch,
    );
    let copy_only =
        gap.filtered_mean_with_scratch(classes, FilterKind::CopyNOnly { n: 2 }, scratch);

    let amplification = if all_tokens.abs() < 1e-9 {
        f32::INFINITY
    } else {
        topk_nocopy.abs() / all_tokens.abs()
    };

    println!("  {:<22} {:>+14.6}", "ALL_TOKENS", all_tokens);
    println!("  {:<22} {:>+14.6}", "TOP-K∩NO-COPY (k=10)", topk_nocopy);
    println!("  {:<22} {:>+14.6}", "COPY-N-ONLY (n=2)", copy_only);
    println!();
    println!(
        "  Amplification |TOP-K∩NO-COPY| / |ALL_TOKENS| = {:.3}×",
        amplification
    );
    println!(
        "  Paper §6 Fig 7: filtered loss ~doubles the separation. Target ≥ 1.5×."
    );
    let verdict = if amplification >= 1.5 { "PASS ✓" } else { "FAIL ✗" };
    println!("  Verdict: {verdict}");
    println!();
}

fn print_proposition_1_table(report: &ClassGapReport) {
    println!("── Proposition 1 annotation (paper §5) ──");
    println!(
        "  ratio = mean_gap / log|V_τ|. ratio→1: near ceiling (little room).",
    );
    println!(
        "  ratio→0: room for richer feature. NaN: no bound supplied."
    );
    println!(
        "  {:<14} {:>8} {:>12} {:>10} {:>14}",
        "class", "count", "mean Δ", "log|V_τ|", "ratio"
    );
    for row in &report.rows {
        let ratio_str = if row.gap_to_bound_ratio.is_nan() {
            "NaN".to_string()
        } else {
            format!("{:+.6}", row.gap_to_bound_ratio)
        };
        let lv_str = if row.log_v_tau.is_nan() {
            "NaN".to_string()
        } else {
            format!("{:.6}", row.log_v_tau)
        };
        println!(
            "  {:<14} {:>8} {:>+12.6} {:>10} {:>14}",
            class_name(row.class),
            row.count,
            row.mean_gap,
            lv_str,
            ratio_str
        );
    }
    println!();
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 335 Phase 3 T3.2 — micro-GPT A/B Paired Loss Diagnostic    ║");
    println!("║  Paper: Li & Merrill 2026, arXiv:2606.20936 (AI2)                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let l = 8192;
    println!("Setup: L = {l} tokens, characterized-bias fixture (Plan 313 sig).");
    println!("  Content / Function: Δ ~ Normal(+0.080 / +0.060, 0.020)  (B-favored)");
    println!("  CopyN(2):           Δ ~ Normal(+0.005,         0.020)  (near-zero)");
    println!("  Other / brackets:   Δ ~ Normal(+0.000,         0.020)  (pure noise)");
    println!();

    let (a, b, classes) = build_characterized_bias_fixture(
        l,
        /*content_shift*/ 0.080,
        /*function_shift*/ 0.060,
        /*copy_shift*/ 0.005,
        /*noise_sigma*/ 0.020,
        0x1234_5678,
    );

    let gap = PairedLossGap::from_log_probs(&a, &b);

    // Table 1: tag-stratified raw means.
    print_tag_stratified_table(&gap, &classes);

    // Table 2: filtered aggregates (the headline amplification result).
    let mut scratch = FilterScratch::with_capacity(l);
    print_filtered_aggregates_table(&gap, &classes, &mut scratch);

    // Table 3: Proposition 1 annotation. Supply bounds for the classes we
    // expect to see; classes without a bound report NaN ratio (still shown).
    let mut bounds = HashMap::new();
    // Content: open-class noun vocab (large V_τ → loose bound → latent earns).
    bounds.insert(TokenClass::Content, ClassSizeBound::for_vocab_size(50_000));
    // Function: closed-class vocab (small V_τ → tight bound).
    bounds.insert(TokenClass::Function, ClassSizeBound::for_vocab_size(500));
    // CopyN(2): tiny visible-prefix retrieval class (smallest V_τ).
    bounds.insert(TokenClass::CopyN(2), ClassSizeBound::for_vocab_size(32));
    // Brackets: closed delimiter set (small V_τ).
    bounds.insert(TokenClass::BracketOpen, ClassSizeBound::for_vocab_size(8));
    bounds.insert(TokenClass::BracketClose, ClassSizeBound::for_vocab_size(8));
    // Other: deliberately omitted → demonstrates NaN-ratio handling.

    let report = gap.annotate_with_class_bounds(&classes, &bounds);
    print_proposition_1_table(&report);

    // Closing interpretation.
    println!("── Interpretation ──");
    println!("  • Content / Function (state-conditioned): the richer feature (B)");
    println!("    captures a measurable fraction of the Proposition 1 ceiling.");
    println!("  • CopyN(2) (visible-prefix retrieval): tiny gap, but the bound is");
    println!("    also tiny → the ratio can still be large. The diagnostic does");
    println!("    NOT say \"the feature helps here\" — only \"the gap, normalized");
    println!("    by the worst-case room.\" Use CopyNOnly filter to isolate.");
    println!("  • Other (no bound): NaN ratio — the consumer must supply a V_τ");
    println!("    for the annotation to be meaningful. mean_gap is still valid.");
    println!();
    println!("See .plans/335_paired_loss_gap_diagnostic_primitive.md Phase 3.");
    println!("See .benchmarks/335_paired_loss_goat.md for the GOAT gate record.");
}
