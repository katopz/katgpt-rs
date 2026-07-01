//! Plan 268 Phase 6 T13 — QGF variance-adaptive guidance weight (F4).
//!
//! Distills the paper's `1/β` guidance-strength knob into a per-query
//! sigmoid gate: `weight = sigmoid(k · (confidence − threshold))`. This is
//! a novel extension the paper does not explore — the paper uses a *fixed*
//! `1/β` tuned per-domain; we make it adaptive using the critic's own
//! variance as the confidence signal.
//!
//! Four labelled steps:
//!
//! 1. **`adaptive_guidance_weight` curve** — print the sigmoid response to
//!    confidence ∈ [0, 1] for two steepness values (gentle 4.0, sharp 12.0).
//!    Demonstrates: low confidence → ~0% (safe fallback), high confidence →
//!    ~100% (aggressive), threshold = 0.5 always = 50%.
//! 2. **Tilt with adaptive weight — high-confidence oracle** — a stable
//!    critic (LeoHead-style cached Q) gets strong guidance. The induced
//!    categorical concentrates sharply toward the Q optimum.
//! 3. **Tilt with adaptive weight — low-confidence oracle** — a noisy critic
//!    (BFN rejection proxy, freeze-tier fallback) collapses the weight
//!    toward 0. The output is ~byte-identical to the BC reference — QGF
//!    refuses to guide when the critic is unreliable.
//! 4. **Stability** — verify the adaptive weight is always in `[0, 1]`,
//!    finite, and monotone in confidence. This is G5 of the GOAT gate.
//!
//! # Why adaptive and not fixed
//!
//! The paper's Fig 20 shows `1/β` has a sweet spot: too low → no improvement,
//! too high → off-manifold exploitation. A *fixed* `1/β` requires per-domain
//! tuning. The adaptive variant discovers the right `1/β` per-query from
//! the critic's own confidence signal — no tuning, no off-manifold risk on
//! unfamiliar states.
//!
//! # Why sigmoid, not softmax
//!
//! Per AGENTS.md: sigmoid is per-query (independent of other queries), SIMD-
//! friendly, and bounded. Softmax would couple queries and require a
//! normalization pass.
//!
//! Run with: `cargo run --example qgf_02_adaptive_weight --features qgf_adaptive --release`

#![cfg(feature = "qgf_adaptive")]

use katgpt_core::qgf::adaptive_guidance_weight;
use katgpt_core::qgf::QGuidedDrafter;
use katgpt_core::traits::{QGradientOracle, SpeculativeGenerator};

// ── Constants ───────────────────────────────────────────────────────────────

const N: usize = 8;

/// Threshold: confidence level at which guidance is half-maximal (50%).
/// Paper default — guidance activates when critic is above average confidence.
const THRESHOLD: f32 = 0.5;

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 268 F4 — QGF variance-adaptive guidance weight              ║");
    println!("║  Paper: Zhou et al. 2026, arXiv:2606.11087 (our extension)        ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Adaptive weight formula: sigmoid(k · (confidence − threshold))");
    println!("  threshold = {THRESHOLD} (guidance half-maximal here)");
    println!("  k         = steepness (gentle 4.0, sharp 12.0)");
    println!();

    // ── Step 1: sigmoid response curve ─────────────────────────────────────
    println!("── Step 1: sigmoid response curve ──");
    println!("  {:<12} {:>14} {:>14}", "confidence", "weight (k=4)", "weight (k=12)");
    for i in 0..=10 {
        let c = i as f32 / 10.0;
        let w_gentle = adaptive_guidance_weight(c, THRESHOLD, 4.0);
        let w_sharp = adaptive_guidance_weight(c, THRESHOLD, 12.0);
        println!("  {:<12.2} {:>14.6} {:>14.6}", c, w_gentle, w_sharp);
    }
    println!();
    println!("  Reading: low confidence → ~0 (pure BC), high confidence → ~1");
    println!("  (aggressive). At threshold=0.5, both curves are exactly 0.5.");
    println!("  Steeper k = sharper on/off (paper warns: too-steer pushes off-manifold).");
    println!();

    // ── Step 2: tilt with high-confidence oracle ───────────────────────────
    //
    // A deterministic-tier oracle (LeoHead / cached-Q): confidence = 1.0.
    // Adaptive weight → sigmoid(12·(1−0.5)) = sigmoid(6) ≈ 0.998 → near-full
    // guidance. Strong tilt, sharp concentration.
    let mut ref_logits = [-2.0f32; N];
    ref_logits[2] = 4.0; // BC mode
    let q_landscape = [0.05, 0.05, 0.20, 0.05, 0.05, 0.10, 1.00, 0.05];

    let high_conf_oracle = KnownLandscapeOracle {
        q_values: q_landscape.to_vec(),
        confidence: 1.0,
    };
    let drafter_high = QGuidedDrafter::new(UnitGen, high_conf_oracle);

    let mut logits_high = ref_logits;
    let mut grad_high = [0.0f32; N];
    let weight_high = adaptive_guidance_weight(1.0, THRESHOLD, 12.0);
    // Use tilt_logits_adaptive to demo the F4 path end-to-end.
    let applied_high = drafter_high.tilt_logits_adaptive(
        &(),
        &(),
        &mut logits_high,
        &mut grad_high,
        0, // step
        THRESHOLD,
        12.0, // steepness
    );

    println!("── Step 2: high-confidence oracle (LeoHead-tier) ──");
    println!("  confidence = 1.0  →  adaptive weight = {:.6}", weight_high);
    println!("  tilt applied? {applied_high}");
    println!("  tilted logits: {:?}", logits_high);
    let e_ref = expected_q(&ref_logits, &q_landscape);
    let e_high = expected_q(&logits_high, &q_landscape);
    println!("  E[Q]: ref={:.4} → guided={:.4}  (relative gain {:.1}%)",
        e_ref,
        e_high,
        (e_high - e_ref) / e_ref.abs().max(1e-9) * 100.0
    );
    println!();

    // ── Step 3: tilt with low-confidence oracle ────────────────────────────
    //
    // A noisy-tier oracle (BFN rejection proxy, freeze-tier fallback):
    // confidence = 0.05. Adaptive weight → sigmoid(12·(0.05−0.5)) = sigmoid(−5.4)
    // ≈ 0.0045 → guidance collapses to ~0.5% — QGF refuses to guide.
    let low_conf_oracle = KnownLandscapeOracle {
        q_values: q_landscape.to_vec(),
        confidence: 0.05,
    };
    let drafter_low = QGuidedDrafter::new(UnitGen, low_conf_oracle);

    let mut logits_low = ref_logits;
    let mut grad_low = [0.0f32; N];
    let weight_low = adaptive_guidance_weight(0.05, THRESHOLD, 12.0);
    let applied_low = drafter_low.tilt_logits_adaptive(
        &(),
        &(),
        &mut logits_low,
        &mut grad_low,
        0,
        THRESHOLD,
        12.0,
    );

    println!("── Step 3: low-confidence oracle (BFN/freeze-tier) ──");
    println!("  confidence = 0.05  →  adaptive weight = {:.6}", weight_low);
    println!("  tilt applied? {applied_low}  (weight = 0.0045 → near-zero tilt)");
    println!("  tilted logits: {:?}", logits_low);
    let e_low = expected_q(&logits_low, &q_landscape);
    println!("  E[Q]: ref={:.4} → 'guided'={:.4}  (relative gain {:.2}%)",
        e_ref,
        e_low,
        (e_low - e_ref) / e_ref.abs().max(1e-9) * 100.0
    );
    println!();
    println!("  Reading: low confidence → ~0% guidance → output ≈ pure BC reference.");
    println!("  This is the freeze-tier equivalence (G2): QGF is a no-op when the");
    println!("  critic is unreliable. The default path is never corrupted.");
    println!();

    // ── Step 4: stability (G5 GOAT gate) ───────────────────────────────────
    println!("── Step 4: stability check (G5 of GOAT gate) ──");
    let mut bounded = true;
    let mut finite = true;
    let mut monotone = true;
    let mut prev = 0.0f32;
    for i in -100..=100 {
        let c = i as f32 / 50.0; // range [-2.0, 2.0] — includes extreme inputs
        let w = adaptive_guidance_weight(c, THRESHOLD, 12.0);
        if !(0.0..=1.0).contains(&w) {
            bounded = false;
        }
        if !w.is_finite() {
            finite = false;
        }
        if i > 0 && w < prev - 1e-6 {
            monotone = false;
        }
        prev = w;
    }
    println!("  bounded ∈ [0, 1] for confidence ∈ [-2, 2]: {bounded}");
    println!("  finite for all tested inputs:              {finite}");
    println!("  monotonically increasing in confidence:    {monotone}");
    assert!(bounded && finite && monotone, "G5 stability check failed");
    println!();
    println!("  All three properties hold → adaptive guidance is numerically safe.");
    println!();

    println!("── Summary ──");
    println!("  • Adaptive `1/β` = sigmoid(k·(confidence − threshold)) — per-query.");
    println!("  • High confidence (cached Q, LeoHead) → strong guidance (paper's `1/β` regime).");
    println!("  • Low confidence (BFN, freeze-tier)   → ~0 guidance → pure BC reference.");
    println!("  • Bounded, finite, monotone — G5 PASS.");
    println!();
    println!("See .plans/268_qgf_test_time_q_guided_flow.md Phase 3 T7 (F4).");
    println!("See .benchmarks/268_qgf_goat.md for G5 stability gate details.");
}

// ── Test fixtures ───────────────────────────────────────────────────────────

struct UnitGen;

impl SpeculativeGenerator for UnitGen {
    type Condition = ();
    type Output = ();
    type Error = ();

    fn generate(
        &mut self,
        _condition: &Self::Condition,
        _rng: &mut fastrand::Rng,
    ) -> Result<Vec<Self::Output>, Self::Error> {
        Ok(vec![()])
    }
}

/// Oracle with a *caller-set* confidence. Demonstrates that the adaptive
/// weight responds to the critic's own confidence signal — not to the Q
/// values themselves.
struct KnownLandscapeOracle {
    q_values: Vec<f32>,
    confidence: f32,
}

impl QGradientOracle for KnownLandscapeOracle {
    type State = ();
    type Action = ();

    fn q_gradient_at(&self, _state: &Self::State, _action: &Self::Action) -> Vec<f32> {
        self.q_values.clone()
    }

    fn q_gradient_into(
        &self,
        _state: &Self::State,
        _action: &Self::Action,
        out: &mut [f32],
    ) {
        let n = out.len().min(self.q_values.len());
        out[..n].copy_from_slice(&self.q_values[..n]);
        for slot in &mut out[n..] {
            *slot = 0.0;
        }
    }

    fn confidence(&self, _state: &Self::State) -> f32 {
        self.confidence
    }
}

// ── Visualization helpers ───────────────────────────────────────────────────

fn expected_q(logits: &[f32], q: &[f32]) -> f32 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    let mut eq = 0.0f32;
    let n = logits.len().min(q.len());
    for i in 0..n {
        let p = (logits[i] - max_logit).exp();
        sum_exp += p;
        eq += p * q[i];
    }
    eq / sum_exp
}
