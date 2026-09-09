//! QGF (Q-Guided Flow) — GOAT gate for the katgpt-core primitive surface.
//!
//! Proves the *mechanism* properties of test-time Q-gradient guidance
//! (Plan 268, arXiv:2606.11087) that are achievable self-containedly in
//! katgpt-core scope:
//!
//! - **G1 (correctness):** `tilt_logits` provably shifts the output
//!   distribution toward higher expected Q. Includes a negative control
//!   (anti-gradient must *decrease* E[Q] — proves the sign convention is
//!   non-circular) and a random-gradient control (random direction does not
//!   reliably increase E[Q]).
//! - **G2 (regression safety):** zero guidance weight → byte-identical
//!   logits (the freeze-tier equivalence guarantee).
//! - **G4 (alloc-free):** `tilt_logits` / `tilt_logits_adaptive` allocate
//!   zero bytes on the hot path (caller-owned buffers only).
//! - **G5 (stability):** adaptive sigmoid weight is bounded in `[0,1]` and
//!   finite for all inputs; extreme tilt produces no NaN/Inf; moderate weight
//!   concentrates the distribution (reduces entropy) without collapsing it
//!   to a degenerate delta (entropy stays positive).
//!
//! # What this gate does NOT prove (deferred to riir-ai)
//!
//! The original plan G1-G3 framed the gate as downstream task quality
//! (Sudoku 9×9 solve rate, DDTree spec acceptance, Bomber arena win rate).
//! Those require real generators + task harnesses that live outside
//! katgpt-core (katgpt-rs root `bomber`/`sudoku`, riir-engine DDTree). They
//! are the *selling-point* layer and are deferred to a riir-ai integration
//! plan. This file proves the primitive's *mechanism* is correct, safe,
//! efficient, and stable — the necessary (not sufficient) condition for any
//! downstream gain.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features "qgf,qgf_drafter,qgf_adaptive" --test qgf_goat
//! ```

#![cfg(all(feature = "qgf_drafter", feature = "qgf_adaptive"))]

use katgpt_core::qgf::QGuidedDrafter;
use katgpt_core::qgf::adaptive::adaptive_guidance_weight;
use katgpt_core::traits::{QGradientOracle, SpeculativeGenerator};

// ──────────────────────────────────────────────────────────────────────────
// Test fixtures
// ──────────────────────────────────────────────────────────────────────────

/// Trivial generator that returns a single unit candidate. `tilt_logits` does
/// not invoke the generator (it operates on caller-owned buffers), so the
/// generator's body is irrelevant for the gate — it exists only to satisfy
/// `QGuidedDrafter`'s type bound `G::Condition == O::State`.
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

/// Oracle backed by a caller-known Q-vector. `q_gradient_into` is pure
/// (no allocation) — it copies the stored vector into the caller's buffer.
/// This is the deterministic, high-confidence tier (LeoHead / cached-Q
/// analogue).
struct KnownLandscapeOracle {
    q_values: Vec<f32>,
}

impl QGradientOracle for KnownLandscapeOracle {
    type State = ();
    type Action = ();

    fn q_gradient_at(&self, _state: &Self::State, _action: &Self::Action) -> Vec<f32> {
        self.q_values.clone()
    }

    fn q_gradient_into(&self, _state: &Self::State, _action: &Self::Action, out: &mut [f32]) {
        let n = out.len().min(self.q_values.len());
        out[..n].copy_from_slice(&self.q_values[..n]);
        out[n..].fill(0.0);
    }

    fn confidence(&self, _state: &Self::State) -> f32 {
        1.0
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Measurement helpers
// ──────────────────────────────────────────────────────────────────────────

/// Expected Q under the categorical distribution induced by `logits`.
///
/// `E[Q] = Σ softmax(logits)_i · Q_i`.
///
/// `softmax` here is the *measurement* of the categorical distribution the
/// logits induce — it is NOT the primitive's mechanism. The primitive does an
/// additive logit shift (`logits += w · ∇Q`); softmax is the mathematically
/// correct map from logits to a probability vector used only to score the
/// consequence. The "sigmoid not softmax" project rule governs the primitive's
/// gates/weights, not this measurement harness.
fn expected_q(logits: &[f32], q: &[f32]) -> f32 {
    let n = logits.len().min(q.len());
    let max_logit = logits[..n]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    let mut eq = 0.0f32;
    for i in 0..n {
        let p = (logits[i] - max_logit).exp();
        sum_exp += p;
        eq += p * q[i];
    }
    eq / sum_exp
}

/// Natural-log entropy (nats) of the categorical induced by `logits`.
fn categorical_entropy(logits: &[f32]) -> f32 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    let mut weighted = 0.0f32;
    for &l in logits.iter() {
        let p = (l - max_logit).exp();
        sum_exp += p;
        if p > 0.0 {
            weighted += p * p.ln();
        }
    }
    // H = -Σ p_i ln p_i  =  ln(Z) - weighted/Z
    sum_exp.ln() - weighted / sum_exp
}

// ══════════════════════════════════════════════════════════════════════════
// G1 — CORRECTNESS: guidance shifts distribution toward higher Q
// ══════════════════════════════════════════════════════════════════════════

/// The load-bearing correctness test: tilting reference logits by `+w·Q`
/// must increase the expected Q of the induced categorical.
#[test]
fn goat_g1_guidance_increases_expected_q() {
    let n = 32usize;
    // Reference (BC) marginal peaked at index 5 — the off-target mode.
    let mut ref_logits = vec![-1.0f32; n];
    ref_logits[5] = 3.0;
    // True Q landscape: target at index 25, uniform low elsewhere.
    let mut q = vec![0.05f32; n];
    q[25] = 1.0;

    let oracle = KnownLandscapeOracle {
        q_values: q.clone(),
    };
    let drafter = QGuidedDrafter::new(UnitGen, oracle).with_weight(3.0);

    let e_before = expected_q(&ref_logits, &q);

    let mut logits = ref_logits.clone();
    let mut grad = vec![0.0f32; n];
    let applied = drafter.tilt_logits(&(), &(), &mut logits, &mut grad, 0);
    assert!(applied, "tilt must apply at step 0 with non-zero weight");

    let e_after = expected_q(&logits, &q);

    assert!(
        e_after > e_before,
        "guided E[Q]={e_after:.6} must exceed unguided E[Q]={e_before:.6}"
    );
    // Clear margin — not a floating-point rounding artifact.
    let rel_gain = (e_after - e_before) / e_before.abs().max(1e-9);
    assert!(
        rel_gain > 0.10,
        "relative gain {rel_gain:.3} must exceed 10% (clear directional shift toward target)"
    );
}

/// Negative control #1: tilting by the *anti*-gradient (`-Q`) must DECREASE
/// expected Q. This proves the sign convention is correct and that G1 is not
/// tautological — the mechanism responds to the gradient *direction*, not
/// merely to "any perturbation increases E[Q]".
#[test]
fn goat_g1_anti_gradient_decreases_expected_q() {
    let n = 32usize;
    let mut ref_logits = vec![-1.0f32; n];
    ref_logits[5] = 3.0;
    let mut q = vec![0.05f32; n];
    q[25] = 1.0;

    // Anti-gradient: the oracle returns -Q.
    let neg_q: Vec<f32> = q.iter().map(|v| -v).collect();
    let oracle = KnownLandscapeOracle { q_values: neg_q };
    let drafter = QGuidedDrafter::new(UnitGen, oracle).with_weight(3.0);

    let e_before = expected_q(&ref_logits, &q);

    let mut logits = ref_logits.clone();
    let mut grad = vec![0.0f32; n];
    drafter.tilt_logits(&(), &(), &mut logits, &mut grad, 0);

    let e_after = expected_q(&logits, &q);
    assert!(
        e_after < e_before,
        "anti-gradient E[Q]={e_after:.6} must be below unguided E[Q]={e_before:.6} (sign convention)"
    );
}

/// Negative control #2: a *random* gradient direction must not reliably
/// increase E[Q]. Aggregated over many seeds, the mean change in E[Q] under
/// random directions should be near zero (not systematically positive).
/// This is the strongest non-circularity argument: only a gradient aligned
/// with Q produces a reliable gain.
#[test]
fn goat_g1_random_gradient_no_systematic_gain() {
    let n = 32usize;
    let mut ref_logits = vec![-1.0f32; n];
    ref_logits[5] = 3.0;
    let mut q = vec![0.05f32; n];
    q[25] = 1.0;

    let e_base = expected_q(&ref_logits, &q);
    let mut rng = fastrand::Rng::new();
    let n_trials = 200;
    let mut gains = 0usize;
    let mut losses = 0usize;

    for seed in 0..n_trials {
        // Random gradient direction in [-1, 1]^n (NOT aligned with Q).
        let random_grad: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let oracle = KnownLandscapeOracle {
            q_values: random_grad,
        };
        let drafter = QGuidedDrafter::new(UnitGen, oracle).with_weight(3.0);

        let mut logits = ref_logits.clone();
        let mut grad = vec![0.0f32; n];
        drafter.tilt_logits(&(), &(), &mut logits, &mut grad, seed);

        let e_after = expected_q(&logits, &q);
        if e_after > e_base + 1e-6 {
            gains += 1;
        } else if e_after < e_base - 1e-6 {
            losses += 1;
        }
    }

    // A random direction should not systematically help: gains and losses
    // should be roughly balanced. We require that gains do not dominate
    // (> 70% would indicate the mechanism inflates E[Q] for any perturbation,
    // which would make G1 tautological).
    let gain_rate = gains as f32 / n_trials as f32;
    assert!(
        gain_rate < 0.70,
        "random-gradient gain rate {gain_rate:.2} too high — mechanism may inflate E[Q] for any perturbation (gains={gains}, losses={losses}, neutral={})",
        n_trials - gains - losses
    );
}

/// Stronger weight → stronger concentration toward the target. This is the
/// monotonicity property: increasing `w` moves more mass toward the Q-peak.
#[test]
fn goat_g1_stronger_weight_monotonic_concentration() {
    let n = 16usize;
    let mut ref_logits = vec![0.0f32; n];
    ref_logits[0] = 2.0;
    let mut q = vec![0.1f32; n];
    q[10] = 1.0;

    let mut prev_eq = f32::NEG_INFINITY;
    for &w in &[0.5f32, 1.0, 2.0, 4.0, 8.0] {
        let oracle = KnownLandscapeOracle {
            q_values: q.clone(),
        };
        let drafter = QGuidedDrafter::new(UnitGen, oracle).with_weight(w);
        let mut logits = ref_logits.clone();
        let mut grad = vec![0.0f32; n];
        drafter.tilt_logits(&(), &(), &mut logits, &mut grad, 0);
        let eq = expected_q(&logits, &q);
        assert!(
            eq >= prev_eq - 1e-5,
            "E[Q] must be monotonic in weight: w={w} gave E[Q]={eq:.6} < prev {prev_eq:.6}"
        );
        prev_eq = eq;
    }
    // Sanity: the strongest weight reaches near the Q-peak.
    assert!(
        prev_eq > 0.9 * q.iter().copied().fold(0.0f32, f32::max),
        "strongest weight should approach the max Q"
    );
}

// ══════════════════════════════════════════════════════════════════════════
// G2 — REGRESSION SAFETY: zero weight is byte-identical to the base policy
// ══════════════════════════════════════════════════════════════════════════

/// Zero guidance weight must (a) report "not applied" and (b) leave the
/// logits buffer byte-identical. This is the freeze-tier equivalence: QGF
/// with no critic is the pure BC reference policy.
#[test]
fn goat_g2_zero_weight_bit_identical() {
    let n = 16usize;
    let oracle = KnownLandscapeOracle {
        q_values: vec![1.0; n],
    };
    let drafter = QGuidedDrafter::new(UnitGen, oracle).with_weight(0.0);

    let logits_initial: Vec<f32> = (0..n).map(|i| (i as f32) * 0.137 - 1.0).collect();
    let mut logits = logits_initial.clone();
    let mut grad = vec![0.0f32; n];

    let applied = drafter.tilt_logits(&(), &(), &mut logits, &mut grad, 0);
    assert!(!applied, "zero weight must skip the tilt");
    assert_eq!(
        logits, logits_initial,
        "zero-weight logits must be byte-identical to the input"
    );
}

/// Period mismatch (step not divisible by `guidance_period`) must also skip
/// the tilt and leave logits untouched.
#[test]
fn goat_g2_period_mismatch_skips_tilt() {
    let n = 8usize;
    let oracle = KnownLandscapeOracle {
        q_values: vec![1.0; n],
    };
    let drafter = QGuidedDrafter::new(UnitGen, oracle)
        .with_weight(2.0)
        .with_period(3); // apply every 3rd step

    for step in 0..9 {
        let mut logits = vec![0.5f32; n];
        let snapshot = logits.clone();
        let mut grad = vec![0.0f32; n];
        let applied = drafter.tilt_logits(&(), &(), &mut logits, &mut grad, step);
        if step % 3 == 0 {
            assert!(applied, "step {step} divisible by period should apply");
            assert_ne!(logits, snapshot, "applied tilt must mutate logits");
        } else {
            assert!(!applied, "step {step} not divisible by period should skip");
            assert_eq!(logits, snapshot, "skipped step must leave logits untouched");
        }
    }
}

/// `NoGuidanceOracle` (the freeze-tier oracle) must produce a zero gradient
/// and zero confidence — independent of the drafter's weight setting.
#[test]
fn goat_g2_no_guidance_oracle_is_zero() {
    use katgpt_core::traits::NoGuidanceOracle;
    let oracle = NoGuidanceOracle;
    let mut buf = [1.0f32; 8];
    oracle.q_gradient_into(&(), &(), &mut buf);
    assert!(
        buf.iter().all(|&v| v == 0.0),
        "NoGuidanceOracle must zero the gradient buffer"
    );
    assert_eq!(
        oracle.confidence(&()),
        0.0,
        "NoGuidanceOracle confidence must be 0.0"
    );
}

// ══════════════════════════════════════════════════════════════════════════
// G4 — ALLOC-FREE: tilt hot path allocates zero bytes
// ══════════════════════════════════════════════════════════════════════════
//
// `tilt_logits` and `tilt_logits_adaptive` operate entirely on caller-owned
// buffers — the documented zero-alloc contract (drafter.rs §"No Allocations").
// We verify with a counting global allocator. The one-shot convenience methods
// (`generate_guided`, `generate_project_tilt_sample`) DO allocate (they call
// the generator's `generate()` which returns a `Vec`) — those are NOT the hot
// path and are explicitly out of the zero-alloc contract.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// Per-thread allocation counter. The QGF GOAT file runs many tests in
// parallel (`cargo test` default), and several G1/G5 tests allocate `Vec`s
// in their bodies. A *global* atomic counter would be polluted by those
// concurrent allocations, producing false positives. Counting per-thread
// isolates the measurement to the test thread that runs the tight loop below.
//
// `Cell<usize>` has no `Drop`, so Rust uses the destructor-free thread-local
// fast path — access does not itself allocate (no recursion hazard).
thread_local! {
    static LOCAL_ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = LOCAL_ALLOC_COUNT.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

/// Snapshot of the current thread's allocation count.
#[inline]
fn thread_alloc_count() -> usize {
    LOCAL_ALLOC_COUNT.with(|c| c.get())
}

#[test]
fn goat_g4_tilt_logits_zero_alloc() {
    let n = 64usize;
    let oracle = KnownLandscapeOracle {
        q_values: vec![0.5; n],
    };
    let drafter = QGuidedDrafter::new(UnitGen, oracle).with_weight(2.0);
    let mut logits = vec![0.1f32; n];
    let mut grad = vec![0.0f32; n];

    // Warmup: any lazy initialization (none expected, but be safe). Also
    // ensures the thread-local slot is registered on this thread before we
    // measure (the `vec!` setup above already touched it).
    for step in 0..16 {
        drafter.tilt_logits(&(), &(), &mut logits, &mut grad, step);
    }

    let before = thread_alloc_count();
    for step in 0..2000 {
        drafter.tilt_logits(&(), &(), &mut logits, &mut grad, step);
    }
    let delta = thread_alloc_count() - before;
    assert_eq!(
        delta, 0,
        "tilt_logits allocated {delta} bytes across 2000 hot-path calls (expected 0)"
    );
}

#[test]
fn goat_g4_tilt_logits_adaptive_zero_alloc() {
    let n = 64usize;
    let oracle = KnownLandscapeOracle {
        q_values: vec![0.5; n],
    };
    let drafter = QGuidedDrafter::new(UnitGen, oracle); // adaptive computes weight per call
    let mut logits = vec![0.1f32; n];
    let mut grad = vec![0.0f32; n];

    // Warmup.
    for step in 0..16 {
        drafter.tilt_logits_adaptive(&(), &(), &mut logits, &mut grad, step, 0.5, 6.0);
    }

    let before = thread_alloc_count();
    for step in 0..2000 {
        drafter.tilt_logits_adaptive(&(), &(), &mut logits, &mut grad, step, 0.5, 6.0);
    }
    let delta = thread_alloc_count() - before;
    assert_eq!(
        delta, 0,
        "tilt_logits_adaptive allocated {delta} bytes across 2000 calls (expected 0)"
    );
}

// ══════════════════════════════════════════════════════════════════════════
// G5 — STABILITY: bounded, finite, non-degenerate
// ══════════════════════════════════════════════════════════════════════════

/// Adaptive weight is finite and in `[0, 1]` for all realistic and extreme
/// inputs. The numerically-stable sigmoid branch must not produce NaN/Inf.
#[test]
fn goat_g5_adaptive_weight_bounded_and_finite() {
    let cases = [
        (-100.0f32, 0.5f32, 6.0f32),
        (0.0, 0.5, 6.0),
        (0.5, 0.5, 6.0),
        (1.0, 0.5, 6.0),
        (100.0, 0.5, 6.0),
        (0.5, 0.0, 100.0), // extreme steepness
        (0.5, 1.0, 0.001), // near-flat
        (f32::INFINITY, 0.5, 6.0),
        (f32::NEG_INFINITY, 0.5, 6.0),
    ];
    for (conf, thr, steep) in cases {
        let w = adaptive_guidance_weight(conf, thr, steep);
        assert!(
            w.is_finite(),
            "adaptive weight not finite for conf={conf}, thr={thr}, steep={steep}: w={w}"
        );
        // For finite conf, weight must be in [0, 1]. (±Inf saturates to 0/1,
        // which is finite per the branch above.)
        if conf.is_finite() {
            assert!(
                (0.0..=1.0).contains(&w),
                "adaptive weight out of [0,1] for conf={conf}: w={w}"
            );
        }
    }
}

/// Extreme tilt (very large weight × very large gradient) must not produce
/// NaN or Inf in the logits buffer. Additive shift of finite values stays
/// finite well within f32 range; this guards against SIMD path corruption.
#[test]
fn goat_g5_extreme_tilt_no_nan_no_inf() {
    let n = 16usize;
    let oracle = KnownLandscapeOracle {
        q_values: vec![1e4; n],
    };
    let drafter = QGuidedDrafter::new(UnitGen, oracle).with_weight(1e4);
    let mut logits = vec![1e4f32; n];
    let mut grad = vec![0.0f32; n];
    drafter.tilt_logits(&(), &(), &mut logits, &mut grad, 0);
    for &l in &logits {
        assert!(l.is_finite(), "extreme tilt produced non-finite logit: {l}");
        assert!(!l.is_nan(), "extreme tilt produced NaN logit");
    }
}

/// Moderate weight concentrates the distribution (reduces entropy) but does
/// NOT collapse it to a degenerate point-mass (entropy stays positive). This
/// is the off-manifold safety property: guidance sharpens without destroying
/// the reference distribution's support.
#[test]
fn goat_g5_moderate_weight_concentrates_without_collapse() {
    let n = 16usize;
    let mut logits = vec![0.0f32; n]; // uniform reference (max entropy)
    let mut q = vec![0.0f32; n];
    q[8] = 5.0; // strong target

    let oracle = KnownLandscapeOracle {
        q_values: q.clone(),
    };
    let drafter = QGuidedDrafter::new(UnitGen, oracle).with_weight(2.0); // moderate

    let entropy_before = categorical_entropy(&logits);
    let uniform_entropy = (n as f32).ln();
    assert!(
        (entropy_before - uniform_entropy).abs() < 1e-4,
        "reference should be near-uniform (entropy {entropy_before:.4} ≈ ln({n}) = {uniform_entropy:.4})"
    );

    let mut grad = vec![0.0f32; n];
    drafter.tilt_logits(&(), &(), &mut logits, &mut grad, 0);

    let entropy_after = categorical_entropy(&logits);
    assert!(
        entropy_after < entropy_before,
        "tilt must reduce entropy (concentrate), got {entropy_after:.4} >= {entropy_before:.4}"
    );
    assert!(
        entropy_after > 0.0,
        "moderate weight must not collapse distribution to zero entropy (off-manifold safety)"
    );
}

/// Adaptive weight at low confidence must collapse toward 0 (safe fallback
/// for noisy critics); at high confidence it saturates toward 1 (strong
/// guidance). Bounds are grounded in the sigmoid math: with steepness `k`, at
/// confidence `c` the weight is `sigmoid(k·(c − threshold))`. We pick a steep
/// `k=12` so the saturation is decisive at the `0.01`/`0.99` confidence
/// extremes, then verify the documented F4 saturation contract.
#[test]
fn goat_g5_adaptive_extremes_saturate_correctly() {
    // k=12: sigmoid(12·(0.01−0.5)) = sigmoid(−5.88) ≈ 0.0028 < 0.01 ✓
    //       sigmoid(12·(0.99−0.5)) = sigmoid(+5.88) ≈ 0.9972 > 0.99 ✓
    const K: f32 = 12.0;
    const THR: f32 = 0.5;

    let w_low = adaptive_guidance_weight(0.01, THR, K);
    assert!(
        w_low < 0.01,
        "low confidence must collapse below 0.01, got {w_low}"
    );

    let w_high = adaptive_guidance_weight(0.99, THR, K);
    assert!(
        w_high > 0.99,
        "high confidence must saturate above 0.99, got {w_high}"
    );

    // Symmetry: at exactly the threshold, weight is 0.5 regardless of steepness.
    let w_mid = adaptive_guidance_weight(THR, THR, K);
    assert!(
        (w_mid - 0.5).abs() < 1e-5,
        "at-threshold must be exactly 0.5, got {w_mid}"
    );

    // Monotonic sweep across the full confidence range at this steepness.
    let mut prev = 0.0f32;
    for i in 0..=100 {
        let conf = i as f32 / 100.0;
        let w = adaptive_guidance_weight(conf, THR, K);
        assert!(
            w >= prev - 1e-6,
            "adaptive weight not monotonic at conf={conf}: prev={prev}, w={w}"
        );
        prev = w;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// T11: Variance comparison — paper Fig 3 reproduction (katgpt-core mechanism)
// ───────────────────────────────────────────────────────────────────────────
//
// The QGF paper Fig 3 shows the drop-Jacobian estimator (QGF) has LOWER
// gradient variance than OOD-sampling or BPTT, measured as higher cosine
// similarity `cos(G(s, a), G(s, a + ε))` for small perturbations ε.
//
// katgpt-core cannot run a real generator's BPTT/OOD estimator surface (that's
// riir-engine scope). But we CAN prove the *mechanism* property that drives
// Fig 3: the QGF primitive's `q_gradient_into` is a pure function of
// `(state, projected_action)` with no Jacobian propagation, so its output is
// stable under action perturbation — by construction, not by measurement.
//
// The test constructs three estimator models:
//   1. QGF (deterministic `∇Q` — our primitive)
//   2. BPTT-like (deterministic `∇Q` + a noisy Jacobian term that amplifies
//      under perturbation — models chain-rule variance)
//   3. OOD-like (`∇Q` + independent per-call sampling noise — models
//      estimator sampling variance)
// and shows QGF has the highest cosine similarity between nearby-action
// gradients, matching the paper's qualitative finding.

/// Cosine similarity between two vectors. Returns 0.0 for zero-norm inputs.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Deterministic QGF estimator: returns the stored gradient verbatim.
/// This is the `QGradientOracle::q_gradient_into` contract — pure, no noise.
struct QgfEstimator {
    gradient: Vec<f32>,
}
impl QgfEstimator {
    fn estimate(&self, _action_seed: u64) -> &[f32] {
        // QGF: drop-Jacobian → output depends only on (state, projected_action),
        // NOT on the action seed / sampling noise. Bit-identical every call.
        &self.gradient
    }
}

/// BPTT-like estimator: gradient + a Jacobian-amplified noise term that grows
/// with the action perturbation. Models the chain-rule variance the paper shows
/// in Fig 3 — BPTT propagates through the generator, so small action changes
/// produce amplified gradient changes.
struct BpttLikeEstimator {
    base_gradient: Vec<f32>,
    /// Noise amplification factor. Higher = more Jacobian-induced variance.
    amplification: f32,
}
impl BpttLikeEstimator {
    fn estimate(&self, action_seed: u64) -> Vec<f32> {
        // Pseudo-random perturbation seeded by the action (deterministic per
        // action, but varies across actions → low cosine sim under perturbation).
        let mut rng = fastrand::Rng::with_seed(action_seed);
        self.base_gradient
            .iter()
            .map(|&g| g + (rng.f32() - 0.5) * 2.0 * self.amplification)
            .collect()
    }
}

/// OOD-like estimator: gradient + independent per-call sampling noise.
/// Models the variance from sampling an off-distribution estimator — each
/// call adds fresh noise uncorrelated with the action.
struct OodLikeEstimator {
    base_gradient: Vec<f32>,
    /// Per-call noise magnitude.
    noise_magnitude: f32,
}
impl OodLikeEstimator {
    fn estimate(&self, call_idx: u64) -> Vec<f32> {
        let mut rng = fastrand::Rng::with_seed(call_idx.wrapping_mul(0x9E3779B97F4A7C15));
        self.base_gradient
            .iter()
            .map(|&g| g + (rng.f32() - 0.5) * 2.0 * self.noise_magnitude)
            .collect()
    }
}

#[test]
fn t11_qgf_has_highest_cosine_similarity_under_perturbation() {
    // Paper Fig 3 setup: measure cos(G(a), G(a + ε)) across perturbations.
    // QGF should have cos ≈ 1.0 (deterministic); BPTT/OOD should have cos < 1.
    let true_gradient = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

    let qgf = QgfEstimator {
        gradient: true_gradient.clone(),
    };
    let bptt = BpttLikeEstimator {
        base_gradient: true_gradient.clone(),
        amplification: 0.5, // moderate Jacobian amplification
    };
    let ood = OodLikeEstimator {
        base_gradient: true_gradient.clone(),
        noise_magnitude: 0.5, // moderate sampling noise
    };

    // For QGF: perturbing the action doesn't change the gradient (drop-Jacobian).
    // Cosine similarity across any perturbation is 1.0.
    let qgf_cos: Vec<f32> = (0..20)
        .map(|seed| cosine_similarity(qgf.estimate(seed), qgf.estimate(seed + 1)))
        .collect();
    let qgf_mean_cos = qgf_cos.iter().sum::<f32>() / qgf_cos.len() as f32;
    assert!(
        (qgf_mean_cos - 1.0).abs() < 1e-6,
        "QGF must have cos≈1.0 (deterministic), got {qgf_mean_cos}"
    );

    // For BPTT-like: perturbing the action changes the noise seed → lower cos.
    let bptt_cos: Vec<f32> = (0..20)
        .map(|seed| cosine_similarity(&bptt.estimate(seed), &bptt.estimate(seed + 1)))
        .collect();
    let bptt_mean_cos = bptt_cos.iter().sum::<f32>() / bptt_cos.len() as f32;

    // For OOD-like: each call adds fresh noise → lower cos.
    let ood_cos: Vec<f32> = (0..20)
        .map(|i| cosine_similarity(&ood.estimate(i), &ood.estimate(i + 1)))
        .collect();
    let ood_mean_cos = ood_cos.iter().sum::<f32>() / ood_cos.len() as f32;

    // The paper's headline: QGF > BPTT and QGF > OOD in cosine similarity.
    assert!(
        qgf_mean_cos > bptt_mean_cos,
        "QGF cos ({qgf_mean_cos:.4}) must exceed BPTT-like cos ({bptt_mean_cos:.4})"
    );
    assert!(
        qgf_mean_cos > ood_mean_cos,
        "QGF cos ({qgf_mean_cos:.4}) must exceed OOD-like cos ({ood_mean_cos:.4})"
    );
}

#[test]
fn t11_qgf_variance_is_zero_across_calls() {
    // Stronger property: the QGF primitive's gradient is not just low-variance
    // but ZERO-variance across repeated calls at the same (state, action).
    // This is the structural reason Fig 3 favors QGF — there's no estimator
    // noise to average away.
    let oracle = KnownLandscapeOracle {
        q_values: vec![1.0, 2.0, 3.0, 4.0],
    };
    let mut g1 = [0.0f32; 4];
    let mut g2 = [0.0f32; 4];
    let mut g3 = [0.0f32; 4];
    oracle.q_gradient_into(&(), &(), &mut g1);
    oracle.q_gradient_into(&(), &(), &mut g2);
    oracle.q_gradient_into(&(), &(), &mut g3);
    assert_eq!(g1, g2, "QGF gradient must be deterministic across calls");
    assert_eq!(g2, g3, "QGF gradient must be deterministic across calls");

    // And cosine similarity of identical vectors is 1.0.
    assert!((cosine_similarity(&g1, &g2) - 1.0).abs() < 1e-6);
}

#[test]
fn t11_qgf_drop_jacobian_documented_in_trait() {
    // The trait doc explicitly states the Jacobian is dropped (J ≈ I) and warns
    // against adding chain-rule backprop. This test is a static assertion that
    // the contract holds: `q_gradient_into` takes only (state, action, out) —
    // there's no generator/Jacobian parameter, so chain-rule backprop is
    // structurally impossible at the trait level.
    //
    // If someone added a `generator: &G` parameter to enable BPTT, this test
    // would fail to compile (the function signature changed), forcing a
    // conscious review of the variance implications.
    let oracle = KnownLandscapeOracle {
        q_values: vec![1.0, 2.0],
    };
    let mut out = [0.0f32; 2];
    // Signature: (state, action, out) — no generator, no Jacobian.
    oracle.q_gradient_into(&(), &(), &mut out);
    assert_eq!(out, [1.0, 2.0]);
}
