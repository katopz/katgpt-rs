//! Plan 334 Phase 3 T3.2 — the curiosity↔predictability inversion.
//!
//! This example demonstrates the **load-bearing theoretical contribution** of
//! sleep-time compute: that curiosity and predictability are inverses.
//! Predictability scores whether a query is anticipated; curiosity scores
//! whether the context itself is novel (off the forecaster's manifold). When
//! a context is novel (high curiosity), *no* direction is predictable —
//! pre-computing is a waste, the wake-time fresh-think path should run. When
//! a context is on-manifold (low curiosity), the catalog anticipates it well
//! and pre-computing pays off.
//!
//! # The synthetic KARC-like ridge forecaster
//!
//! We model the forecaster as a low-dim linear ridge over a delay basis: the
//! context is assumed to evolve on the manifold spanned by the first
//! `manifold_dim` axes, with `forecast(x) = decay · x` on that subspace.
//! The **curiosity residual** is the off-manifold component's magnitude:
//!
//! ```text
//! curiosity(x) = ‖x − forecast(x)‖ = ‖x_off_manifold‖ + (1−decay)·‖x_on_manifold‖
//! ```
//!
//! - **Low-curiosity context** (mostly on-manifold): small residual → high
//!   predictability → `should_pre_compute` returns **true**.
//! - **High-curiosity context** (mostly off-manifold): large residual → low
//!   predictability → `should_pre_compute` returns **false**.
//!
//! This is the same math as KARC (Plan 308) — a closed-form ridge fit over a
//! delay basis — distilled to its smallest form for the demo. No training, no
//! backprop (modelless, katgpt-rs mandate).
//!
//! # The PredictabilityScorer trait-swap mechanism
//!
//! The primitive ships only `DotPredictabilityScorer` (`p = sigmoid(α·dot+β)`).
//! This example defines its own `CuriosityInversionScorer` that implements the
//! public `PredictabilityScorer` trait:
//!
//! ```text
//! p = sigmoid(α · (curiosity_ref − curiosity(c, dir)))
//! ```
//!
//! This is the generalized curiosity-inversion form: `curiosity_ref` is the
//! "typical" curiosity level. Contexts with curiosity below the reference are
//! predictable (p > 0.5); above it, unpredictable (p < 0.5). The special case
//! `curiosity_ref = 0` reduces to the `p = 1 − sigmoid(α·curiosity)` form
//! referenced in `sleep_time/predictability.rs` (which bounds p ≤ 0.5 — fine
//! for relative ranking, but the generalized form gives a cleaner high/low
//! contrast for this demo).
//!
//! and feeds it to `SleepTimeAnticipator`. The anticipator is generic over the
//! scorer — the consumer swaps scorers without touching the orchestration.
//!
//! # How curiosity becomes `should_pre_compute`
//!
//! `AmortizationCostModel::should_pre_compute(sleep_cost, N, e_gate)` reads the
//! **expected gate hit rate** `E[gate]`, not the per-direction `p_i` directly.
//! The chain is:
//!
//! ```text
//!   curiosity(c, dir_i) ──► p_i = sigmoid(α·(curiosity_ref − curiosity))
//!                         ──► gate_i = sigmoid(β·(p_i − τ))
//!                         ──► E[gate] = mean_i(gate_i)
//!                         ──► should_pre_compute(sleep_cost, N, E[gate])
//! ```
//!
//! This example walks that chain end-to-end and shows the verdict flip:
//! low-curiosity context → pre-compute; high-curiosity context → don't.
//!
//! Run with: `cargo run --example sleep_time_02_curiosity_inversion --features sleep_time_anticipation --release`

use katgpt_core::sleep_time::{
    consume, consume_gate, AmortizationCostModel, AnticipatedQueryDir, IdentityFunctorOp,
    PredictabilityScorer, SleepTimeAnticipator, SleepTimeScratch,
};

// ── Constants ───────────────────────────────────────────────────────────────

const D: usize = 4;
const K: usize = 4;
const TAU: f32 = 0.5;
const BETA: f32 = 4.0;

// ── A synthetic KARC-like ridge forecaster ──────────────────────────────────

/// A minimal closed-form ridge forecaster over a delay basis.
///
/// Models the context as evolving on a low-dim linear manifold: the first
/// `manifold_dim` axes are "on-manifold" (forecast = `decay · x`), the rest
/// are "off-manifold" (forecast = 0, surprise). The curiosity residual is:
///
/// ```text
/// curiosity(x) = ‖x_off_manifold‖² + (1 − decay)² · ‖x_on_manifold‖²
/// ```
///
/// (Squared — we'll sqrt outside to get an L2 residual. Or just use the
/// squared form directly since sigmoid is monotone and the demo only needs
/// the *direction* of the inversion, not calibrated magnitudes.)
///
/// This is the smallest possible distillation of the KARC ridge fit
/// (Plan 308): closed-form, deterministic, modelless.
#[derive(Clone, Copy, Debug)]
struct SyntheticKarcRidge {
    /// Manifold dimension. Axes `[0, manifold_dim)` are on-manifold.
    manifold_dim: usize,
    /// On-manifold decay (forecast = decay · x on the manifold subspace).
    /// decay=1.0 → perfect forecast on-manifold (no curiosity there).
    decay: f32,
}

impl SyntheticKarcRidge {
    /// Curiosity residual of `c` under this forecaster. Higher = more novel.
    ///
    /// We return the squared L2 residual (no sqrt — sigmoid is monotone and
    /// we don't need calibrated magnitudes for the demo).
    fn curiosity_sq<const D: usize>(&self, c: &[f32; D]) -> f32 {
        let mut off_manifold_sq = 0.0f32;
        let mut on_manifold_sq = 0.0f32;
        for j in 0..D {
            if j < self.manifold_dim {
                // On-manifold: forecast = decay · x, residual = (1 − decay) · x.
                let r = (1.0 - self.decay) * c[j];
                on_manifold_sq += r * r;
            } else {
                // Off-manifold: forecast = 0, residual = x (the whole thing).
                off_manifold_sq += c[j] * c[j];
            }
        }
        off_manifold_sq + on_manifold_sq
    }
}

// ── The curiosity-inversion predictability scorer ───────────────────────

/// `p = sigmoid(α·(curiosity_ref − curiosity(c)))` — the curiosity↔predictability
/// inversion (generalized form with a reference curiosity).
///
/// The curiosity is direction-independent here (a property of the context
/// alone), so every direction gets the same `p_i` for a given `c`. In a richer
/// scorer the curiosity could be direction-conditional (e.g. curiosity for
/// direction `i` = residual of `forecast(c + dir_i)`); the trait signature
/// supports both — the scorer is free to ignore `dir`.
///
/// `curiosity_ref` is the "typical" curiosity level — the neutral point where
/// p = 0.5. Contexts with curiosity below `curiosity_ref` are predictable
/// (p > 0.5); above it, unpredictable (p < 0.5). The special case
/// `curiosity_ref = 0` reduces to the `p = 1 − sigmoid(α·curiosity)` form
/// referenced in `sleep_time/predictability.rs`.
///
/// This is the curiosity-inversion variant referenced in
/// `sleep_time/predictability.rs` doc comments as "riir-ai Plan 341 territory".
/// We implement it here, in the example, to demonstrate the trait-swap
/// mechanism without expanding the shipped API surface.
#[derive(Clone, Copy, Debug)]
struct CuriosityInversionScorer {
    /// Curiosity → predictability slope. Higher = sharper inversion.
    alpha: f32,
    /// Reference curiosity: the "typical" level. curiosity < ref → p > 0.5.
    curiosity_ref: f32,
    /// The synthetic KARC-like forecaster.
    forecaster: SyntheticKarcRidge,
}

impl<const D: usize> PredictabilityScorer<D> for CuriosityInversionScorer {
    #[inline]
    fn predictability(&self, c: &[f32; D], _dir: &AnticipatedQueryDir<D>) -> f32 {
        let curiosity = self.forecaster.curiosity_sq(c);
        // p = sigmoid(α·(curiosity_ref − curiosity)).
        // curiosity < ref → positive arg → p > 0.5 (predictable).
        // curiosity > ref → negative arg → p < 0.5 (unpredictable).
        sigmoid(self.alpha * (self.curiosity_ref - curiosity))
    }
}

// ── sigmoid (local copy — the simd kernel is crate-private) ─────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 334 Phase 3 T3.2 — Curiosity ↔ Predictability Inversion   ║");
    println!("║  Paper: Lin et al. 2025, arXiv:2504.13171 (Letta/Berkeley)      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("The load-bearing claim: curiosity and predictability are INVERSES.");
    println!("High-curiosity context (novel, off-manifold) → low predictability →");
    println!("pre-computing is a WASTE, the wake-time fresh-think path should run.");
    println!("Low-curiosity context (on-manifold) → high predictability →");
    println!("pre-computing PAYS OFF across N consumers.");
    println!();

    // ── The synthetic forecaster ───────────────────────────────────────────
    //
    // Manifold = first 2 of 4 axes. decay=1.0 means on-manifold forecasts are
    // perfect (zero residual there); any component on axes 2 or 3 is pure
    // off-manifold surprise.
    let forecaster = SyntheticKarcRidge {
        manifold_dim: 2,
        decay: 1.0,
    };
    let scorer = CuriosityInversionScorer {
        alpha: 2.0,
        // Reference curiosity = 0.5. Contexts with curiosity < 0.5 get p > 0.5
        // (predictable); contexts with curiosity > 0.5 get p < 0.5.
        curiosity_ref: 0.5,
        forecaster,
    };

    println!("── Synthetic KARC-like ridge forecaster ──");
    println!(
        "  manifold_dim = {} of {} axes. decay = {}. Off-manifold axes = [{}, {}).",
        forecaster.manifold_dim,
        D,
        forecaster.decay,
        forecaster.manifold_dim,
        D
    );
    println!("  curiosity(x) = ‖x_off_manifold‖²  (on-manifold contributes 0 since decay=1)");
    println!();

    // ── Two contexts: one low-curiosity, one high-curiosity ────────────────
    //
    // Low-curiosity: c_lo is entirely on the manifold (first 2 axes only).
    // High-curiosity: c_hi is entirely off the manifold (last 2 axes only).
    let c_lo: [f32; D] = [1.0, 0.5, 0.0, 0.0]; // on-manifold
    let c_hi: [f32; D] = [0.0, 0.0, 1.0, 0.8]; // off-manifold

    let cur_lo = forecaster.curiosity_sq(&c_lo);
    let cur_hi = forecaster.curiosity_sq(&c_hi);

    println!("── Two contexts ──");
    println!("  c_lo = {:?}  (on-manifold)   curiosity = {:.4}", c_lo, cur_lo);
    println!("  c_hi = {:?}  (off-manifold)  curiosity = {:.4}", c_hi, cur_hi);
    println!();

    // ── Build the catalog (same toy 4-direction catalog as T3.1) ───────────
    let dirs: [AnticipatedQueryDir<D>; K] = [
        AnticipatedQueryDir::new([1.0, 0.0, 0.0, 0.0]),
        AnticipatedQueryDir::new([0.0, 1.0, 0.0, 0.0]),
        AnticipatedQueryDir::new([0.0, 0.0, 1.0, 0.0]),
        AnticipatedQueryDir::new([0.0, 0.0, 0.0, 1.0]),
    ];

    // ── Anticipate with the curiosity-inversion scorer ─────────────────────
    //
    // Same anticipator orchestration as T3.1, but with our custom scorer.
    // The anticipator is generic over Scorer — no orchestration code changed.
    let mk_anticipator = || SleepTimeAnticipator::<D, K, IdentityFunctorOp, CuriosityInversionScorer> {
        op: IdentityFunctorOp,
        scorer,
        budgets: [100, 100, 100, 100],
        tau: TAU,
        beta: BETA,
    };

    let mut scratch = SleepTimeScratch::new();
    let anticipator = mk_anticipator();
    let c_prime_lo = anticipator.anticipate(&c_lo, &dirs, &mut scratch);
    let c_prime_hi = anticipator.anticipate(&c_hi, &dirs, &mut scratch);

    println!("── anticipate() with CuriosityInversionScorer ──");
    println!("  p_i = sigmoid(α·(curiosity_ref − curiosity(c))). α=2, curiosity_ref=0.5.");
    println!("  Curiosity is context-only here, so every slot in c'_lo has the same");
    println!("  p, and every slot in c'_hi too.");
    println!();
    println!("  {:<10} {:>14} {:>14}", "context", "curiosity", "slot p_i");
    let p_lo = c_prime_lo.slots[0].predictability;
    let p_hi = c_prime_hi.slots[0].predictability;
    println!("  c_lo       {:>14.4} {:>14.6}   (HIGH p → predictable)", cur_lo, p_lo);
    println!("  c_hi       {:>14.4} {:>14.6}   (LOW  p → unpredictable)", cur_hi, p_hi);
    println!();
    println!("  Verdict: high-curiosity context c_hi gets LOW predictability across");
    println!("  the whole catalog — no direction anticipates it. c_lo gets HIGH p.");
    println!();

    // ── consume() shows the gate flip ─────────────────────────────────────
    //
    // Same query in both contexts. The gate = sigmoid(β·(p_{i*} − τ)) reads
    // the slot predictability, which came from the curiosity inversion. So:
    //   - q in c_lo context → gate ≈ 1 → use precomputed.
    //   - q in c_hi context → gate ≈ 0 → fall through to fresh.
    let q: [f32; D] = [1.0, 0.0, 0.0, 0.0]; // close to dir[0]
    let fresh_think = |qq: &[f32; D]| {
        let mut out = [0.0f32; D];
        for j in 0..D {
            out[j] = -qq[j];
        }
        out
    };

    println!("── consume() — gate flips with curiosity ──");
    println!("  query q = {:?} in both contexts. fresh_think(q) = −q.", q);
    println!();

    let (best_lo, gate_lo) = consume_gate(&q, &c_prime_lo, TAU, BETA);
    let out_lo = consume(&q, &c_prime_lo, TAU, BETA, fresh_think);
    println!("  in c_lo (low curiosity):");
    println!("    best slot i* = {best_lo}  gate = {:.6}  (HIGH → precomputed)", gate_lo);
    println!("    out = {}  ≈ precomputed slot", fmt_array(&out_lo));

    let (best_hi, gate_hi) = consume_gate(&q, &c_prime_hi, TAU, BETA);
    let out_hi = consume(&q, &c_prime_hi, TAU, BETA, fresh_think);
    println!("  in c_hi (high curiosity):");
    println!("    best slot i* = {best_hi}  gate = {:.6}  (LOW → fresh think)", gate_hi);
    println!("    out = {}  ≈ fresh_think output", fmt_array(&out_hi));
    println!();
    println!("  Same query, same catalog, same τ/β — different verdict. That's the");
    println!("  curiosity inversion doing its work through the public API.");
    println!();

    // ── The economic verdict: should_pre_compute flips ────────────────────
    //
    // E[gate] = mean over the K slots of gate_i. Since curiosity is context-
    // only here, all K slots share the same gate, so E[gate] = gate_i.
    let model = AmortizationCostModel {
        t: 10.0,
        b_max: 100,
        tau: TAU,
        beta: BETA,
    };
    // sleep_cost is tuned so the verdict flips between the two contexts:
    // break-even E[gate] = sleep_cost / (N·t·b_max) = 2000/10000 = 0.2.
    // c_lo (E[gate] ≈ 0.72) is above → pre-compute. c_hi (E[gate] ≈ 0.16)
    // is below → don't pre-compute.
    let sleep_cost: f32 = 2000.0; // Σ budgets (4 dirs × 500 each)
    let n_consumers = 10u32;

    println!("── AmortizationCostModel::should_pre_compute (paper §5.3) ──");
    println!(
        "  t={}, b_max={}, sleep_cost={}, N={}",
        model.t, model.b_max, sleep_cost, n_consumers
    );
    println!();
    println!("  {:<10} {:>12} {:>12} {:>16}", "context", "E[gate]", "amort. factor", "should_pre_compute?");

    for (label, cur, e_gate, c_prime) in [
        ("c_lo", cur_lo, gate_lo, &c_prime_lo),
        ("c_hi", cur_hi, gate_hi, &c_prime_hi),
    ] {
        let _ = c_prime; // artifact already built above; we have e_gate from the gate.
        let factor = model.amortization_factor(sleep_cost, n_consumers, e_gate);
        let should = model.should_pre_compute(sleep_cost, n_consumers, e_gate);
        let should_str = if should { "YES ✓" } else { "no ✗" };
        println!(
            "  {label:<10} {:>12.4} {:>12.4} {:>16}   (curiosity={cur:.4})",
            e_gate, factor, should_str
        );
    }
    println!();
    println!("  Verdict: low-curiosity context (on-manifold) → pre-compute. The");
    println!("  sleep-time compute will be hit by N consumers; the amortization");
    println!("  factor is < 1.0. High-curiosity context (off-manifold) → DON'T");
    println!("  pre-compute. The cache would miss anyway; pay fresh-think at wake.");
    println!();

    println!("── Summary ──");
    println!("  • Curiosity and predictability are inverses (the paper's core claim).");
    println!("  • CuriosityInversionScorer implements the public PredictabilityScorer");
    println!("    trait — no primitive changes needed. The anticipator is generic.");
    println!("  • The inversion propagates: curiosity → p_i → gate_i → E[gate] →");
    println!("    should_pre_compute. Low-curiosity → pre-compute; high → don't.");
    println!();
    println!("See .plans/334_sleep_time_query_anticipator_primitive.md Phase 3 T3.2.");
    println!("See sleep_time/predictability.rs for the trait + shipped DotPredictabilityScorer.");
}

// ── Display helpers ────────────────────────────────────────────────────────

fn fmt_array<const D: usize>(a: &[f32; D]) -> String {
    let parts: Vec<String> = a.iter().map(|x| format!("{:.3}", x)).collect();
    format!("[{}]", parts.join(", "))
}
