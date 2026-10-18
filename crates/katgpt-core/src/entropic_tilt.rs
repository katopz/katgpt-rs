//! Entropic (max-seeking) advantage tilt — the TTT-Discover within-batch
//! objective as shared, modelless math.
//!
//! Source: arXiv:2601.16175 "Learning to Discover at Test Time"
//! (TTT-Discover), distilled at `.research/019_TTT_Discover_Test_Time_Training.md`.
//! The objective itself is published prior art (RS-GRPO arXiv:2509.24261,
//! RSPO arXiv:2508.01174) — this module transfers it, it does not claim it.
//!
//! ## Why it lives here
//!
//! Two consumers need the identical arithmetic and must not fork it:
//!
//! - **riir-clippy** (`selection_entropic`, Issue 026) — candidate selection
//!   over a heal pool; consumes the batch form and maps through `sigmoid`.
//! - **riir-train** (`loss_grpo`, Plan 341) — the GRPO advantage estimator for
//!   game self-play; consumes the leave-one-out form.
//!
//! riir-clippy shipped first and this module is the hoist mandated by Plan 341
//! ("do NOT duplicate past one implementation + one import").
//!
//! ## Mechanism
//!
//! ```text
//! q_β(i) = exp(β·r_i) / Σ_j exp(β·r_j)      (tilted distribution)
//! w_β(i) = n · q_β(i)                        (weights, mean 1 over the group)
//! A_β(i) = w_β(i) − 1                        (advantages, sum 0)
//! ```
//!
//! with β solved per group by bisection so the tilt stays within a KL budget:
//! `KL(q_β ‖ uniform) = γ` (the paper's default `γ = ln 2`).
//!
//! Auto-adaptation is the point: a group of consistent small improvements
//! permits a large β (aggressive max-seeking — tiny gaps get amplified into
//! decisive credit); an outlier-dominated group forces small β, and the budget
//! bounds the maximum weight regardless of gap magnitude (one lucky rollout
//! cannot own the update). The tilt is shift/scale invariant in the rewards
//! (`r → a·r + b` yields identical advantages — paper Appendix A.1).
//!
//! ## Batch vs leave-one-out
//!
//! [`tilt_advantages_into`] uses the batch denominator `Σ_j` (includes self);
//! [`tilt_advantages_loo_into`] uses `Σ_{j≠i}` (paper A.1).
//!
//! Which one is correct depends on what the consumer does with the output:
//!
//! - **Ranking-only consumers** (riir-clippy's argmax blend) may use either —
//!   both forms are strictly monotone in `r_i` within a fixed group, so the
//!   induced ordering is identical.
//! - **Gradient-scaling consumers** (GRPO) should prefer LOO: the advantage
//!   *magnitude* multiplies the policy gradient, and the batch form leaks the
//!   sample's own reward into its own baseline. That self-contribution biases
//!   the estimator toward zero, and the bias is `O(1/n)` — negligible for large
//!   groups, material at the group sizes self-play actually uses (K = 8–16).
//!
//! **LOO is not zero-sum.** The batch form satisfies `Σ_i A_i = 0` by
//! construction (`Σ_i w_i = n`); removing the self term from each denominator
//! breaks that identity. This is expected — LOO is a debiased estimator, not a
//! centered one — and callers that rely on zero-sum must use the batch form.
//!
//! ## Why a softmax normalization here does not violate the house
//! sigmoid-not-softmax rule
//!
//! The rule targets ROUTING/GATING decisions where softmax forces exclusive
//! competition between additive gates. Here the Boltzmann normalization is the
//! published objective's tilting mechanism, and its output is an ADVANTAGE
//! (signed, zero-sum over the group — not a competing gate).
//!
//! entropic_tilt — KL-budgeted max-seeking advantage tilt (TTT-Discover
//! arXiv:2601.16175; prior art RS-GRPO / RSPO). The max-seeking counterpart to
//! `best_belief` above: that scores a candidate from its OWN history counts,
//! this scores it from the SHAPE of the current group. Shared math with exactly
//! two consumers — riir-clippy `selection_entropic` (Issue 026, ranking) and
//! riir-train `loss_grpo` (Plan 341, gradient scaling) — hoisted here rather
//! than forked. Opt-in pending the Plan 341 Phase 2/3 GOAT.

/// KL budget γ — the paper's default `ln 2`.
pub const KL_BUDGET_LN2: f32 = core::f32::consts::LN_2;

/// Bisection tolerance on the achieved KL.
const KL_TOL: f32 = 1e-4;

/// Bisection iterations (halves the β bracket 60× — far past f32 precision).
const BISECT_ITERS: usize = 60;

/// Upper β bracket. Rewards are expected pre-normalized to a bounded range
/// (belief scores in (0, 1), or z-scored returns); β = 1e3 already saturates
/// any such group to one-hot, so the bracket spans the full uniform→argmax
/// range.
const BETA_MAX: f32 = 1e3;

/// `KL(q_β ‖ uniform)` for the group, computed streaming (zero allocation).
///
/// `KL = Σ q_i · ln(n · q_i)` — with the `r_max`-subtracted parameterization,
/// `ln(n·q_i) = ln n + β(r_i − r_max) − ln Z` where
/// `Z = Σ exp(β(r_j − r_max)) ≥ 1` (the max element contributes `exp(0)`).
#[inline]
#[allow(clippy::cast_precision_loss)] // group sizes ≤ ~10²
fn kl_vs_uniform(rewards: &[f32], beta: f32, r_max: f32) -> f32 {
    let n = rewards.len() as f32;
    let ln_n = n.ln();
    let mut z = 0.0f32;
    for &r in rewards {
        z += (beta * (r - r_max)).exp();
    }
    let ln_z = z.ln();
    let mut kl = 0.0f32;
    for &r in rewards {
        let q = (beta * (r - r_max)).exp() / z;
        // q·ln(n·q): an underflowed q = 0 contributes 0, not NaN.
        if q > 0.0 {
            kl += q * (ln_n + beta * (r - r_max) - ln_z);
        }
    }
    kl
}

/// Solve β ≥ 0 by bisection so `KL(q_β ‖ uniform) ≈ gamma`.
///
/// KL is monotonically non-decreasing in β (β = 0 → uniform → KL = 0;
/// β → ∞ → one-hot on argmax → KL = ln n). Returns the converged β.
/// Degenerate groups (all rewards equal) have KL = 0 for every β, so the budget
/// is unreachable and the upper bracket is returned — harmless, because the
/// weights ignore β entirely in that case.
///
/// Zero-allocation: the KL is recomputed streaming per bisection iteration.
#[must_use]
pub fn solve_beta(rewards: &[f32], gamma: f32) -> f32 {
    debug_assert!(
        rewards.iter().all(|r| r.is_finite()),
        "entropic tilt requires finite rewards"
    );
    if rewards.is_empty() {
        return 0.0;
    }
    let r_max = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut lo = 0.0f32; // KL(0) = 0 ≤ gamma
    let mut hi = BETA_MAX;
    if kl_vs_uniform(rewards, hi, r_max) <= gamma {
        // Budget unreachable (or exactly reachable only at saturation —
        // n = 2 hits KL = ln 2 at one-hot): every β beyond is equivalent.
        return hi;
    }
    for _ in 0..BISECT_ITERS {
        let mid = 0.5 * (lo + hi);
        let kl = kl_vs_uniform(rewards, mid, r_max);
        if (kl - gamma).abs() < KL_TOL {
            return mid;
        }
        if kl < gamma {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Write the tilted weights `w_β(i) = n · q_β(i)` (mean 1 over the group) into
/// `weights_out` (cleared + refilled — caller-reused scratch).
///
/// `beta` typically comes from [`solve_beta`]. Overflow-safe by construction:
/// every exponent is `β(r − r_max) ≤ 0`.
pub fn tilted_weights(rewards: &[f32], beta: f32, weights_out: &mut Vec<f32>) {
    debug_assert!(
        rewards.iter().all(|r| r.is_finite()),
        "entropic tilt requires finite rewards"
    );
    #[allow(clippy::cast_precision_loss)] // group sizes ≤ ~10²
    let n = rewards.len() as f32;
    let r_max = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f32;
    for &r in rewards {
        z += (beta * (r - r_max)).exp();
    }
    // z ≥ 1 always (the max element contributes exp(0) = 1) — no guard needed.
    weights_out.clear();
    weights_out.reserve(rewards.len());
    for &r in rewards {
        weights_out.push(n * (beta * (r - r_max)).exp() / z);
    }
}

/// Solve β for the group + write advantages `A_β(i) = w_β(i) − 1` into
/// `advantages_out` (cleared + refilled — caller-reused scratch).
///
/// The batch (self-inclusive) form: rewards in, zero-sum advantages out
/// (sum ≈ 0 within f32 tolerance). Returns the solved β.
pub fn tilt_advantages_into(rewards: &[f32], gamma: f32, advantages_out: &mut Vec<f32>) -> f32 {
    let beta = solve_beta(rewards, gamma);
    tilted_weights(rewards, beta, advantages_out);
    for a in advantages_out.iter_mut() {
        *a -= 1.0;
    }
    beta
}

/// Leave-one-out variant (paper A.1) — the form gradient-scaling consumers want.
///
/// Each sample's baseline excludes its own reward:
///
/// ```text
/// w_i^LOO = exp(β·r_i) / [ (1/(n−1)) · Σ_{j≠i} exp(β·r_j) ]
/// A_i     = w_i^LOO − 1
/// ```
///
/// Solves β on the FULL group (the KL budget describes the group's shape, and
/// re-solving per held-out subset would make β sample-dependent and the
/// advantages incomparable within the group), then applies the LOO denominator.
///
/// Groups of size ≤ 1 have no baseline to leave out and yield a single `0.0` —
/// matching the "no advantage signal" convention of z-scored GRPO. Degenerate
/// all-equal groups yield exactly zero advantages, so no update is produced.
///
/// **Not zero-sum** — see the module docs. Returns the solved β.
pub fn tilt_advantages_loo_into(rewards: &[f32], gamma: f32, advantages_out: &mut Vec<f32>) -> f32 {
    debug_assert!(
        rewards.iter().all(|r| r.is_finite()),
        "entropic tilt requires finite rewards"
    );
    advantages_out.clear();
    match rewards.len() {
        // No group shape: emit the neutral advantage rather than a NaN from a
        // zero-width baseline.
        0 => return 0.0,
        1 => {
            advantages_out.push(0.0);
            return 0.0;
        }
        _ => {}
    }
    let beta = solve_beta(rewards, gamma);
    let r_max = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    #[allow(clippy::cast_precision_loss)] // group sizes ≤ ~10²
    let n_minus_1 = (rewards.len() - 1) as f32;

    // One pass for Z, then a per-sample subtraction — O(n), not O(n²).
    let mut z = 0.0f32;
    for &r in rewards {
        z += (beta * (r - r_max)).exp();
    }
    advantages_out.reserve(rewards.len());
    for &r in rewards {
        let e_i = (beta * (r - r_max)).exp();
        // Z ≥ e_i by construction; the max element makes Z ≥ 1, and with
        // n ≥ 2 at least one other term is > 0, so z_loo > 0.
        let z_loo = z - e_i;
        let w = if z_loo > 0.0 {
            n_minus_1 * e_i / z_loo
        } else {
            1.0
        };
        advantages_out.push(w - 1.0);
    }
    beta
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// (a) Shift/scale invariance — paper Appendix A.1.
    #[test]
    fn tilt_is_shift_scale_invariant() {
        let base = [0.2f32, 0.5, 0.6, 0.9];
        let shifted: Vec<f32> = base.iter().map(|r| 3.0 * r + 7.0).collect();
        let mut a1 = Vec::new();
        let mut a2 = Vec::new();
        tilt_advantages_into(&base, KL_BUDGET_LN2, &mut a1);
        tilt_advantages_into(&shifted, KL_BUDGET_LN2, &mut a2);
        for (i, (x, y)) in a1.iter().zip(&a2).enumerate() {
            assert!(
                (x - y).abs() < 1e-3,
                "advantage {i} drifted under r → 3r+7: {x} vs {y}"
            );
        }
    }

    /// (a') The LOO form must inherit the invariance — it is the form the
    /// GRPO consumer uses, and reward scale there is arbitrary.
    ///
    /// **Tolerance is RELATIVE here, unlike the batch test above, and that is a
    /// property of the quantity rather than a weaker assertion.** Batch
    /// advantages are bounded (`A ∈ [−1, n−1]`), so an absolute tolerance is
    /// meaningful; LOO advantages are unbounded above (measured 4.7× the batch
    /// magnitude on this very group — 10.12 vs 2.15), so a fixed absolute bound
    /// silently tightens as the winner's weight grows. The residual is
    /// `solve_beta`'s `KL_TOL` and nothing else: the solved β ratio is
    /// 3.00047588 against an exact 3.0 (1.6e-4 relative, i.e. the bisection
    /// tolerance), which propagates to 3.7e-4 relative here.
    /// `loo_invariance_is_exact_at_matched_beta` pins that attribution, so this
    /// bound is justified rather than merely loose.
    #[test]
    fn loo_tilt_is_shift_scale_invariant() {
        let base = [0.2f32, 0.5, 0.6, 0.9];
        let shifted: Vec<f32> = base.iter().map(|r| 3.0 * r + 7.0).collect();
        let mut a1 = Vec::new();
        let mut a2 = Vec::new();
        tilt_advantages_loo_into(&base, KL_BUDGET_LN2, &mut a1);
        tilt_advantages_loo_into(&shifted, KL_BUDGET_LN2, &mut a2);
        for (i, (x, y)) in a1.iter().zip(&a2).enumerate() {
            let rel = (x - y).abs() / x.abs().max(1e-9);
            assert!(
                rel < 1e-3,
                "LOO advantage {i} drifted under r → 3r+7: {x} vs {y} (rel {rel:e})"
            );
        }
    }

    /// Attribution for the looser bound above: with the exponents matched by
    /// construction (β scaled exactly by 1/a instead of re-solved), the tilt is
    /// invariant to f32 precision. Any drift the solve-and-tilt path shows is
    /// therefore `solve_beta`'s bisection tolerance, NOT an asymmetry in the
    /// LOO denominator — which is the failure this test exists to rule out.
    #[test]
    fn loo_invariance_is_exact_at_matched_beta() {
        let base = [0.2f32, 0.5, 0.6, 0.9];
        let scale = 3.0f32;
        let shifted: Vec<f32> = base.iter().map(|r| scale * r + 7.0).collect();
        let beta = solve_beta(&base, KL_BUDGET_LN2);
        let (mut w1, mut w2) = (Vec::new(), Vec::new());
        tilted_weights(&base, beta, &mut w1);
        // β(r − r_max) is invariant under r → a·r + b when β → β/a, so the
        // exponents — and hence the weights — must coincide exactly.
        tilted_weights(&shifted, beta / scale, &mut w2);
        for (i, (x, y)) in w1.iter().zip(&w2).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "weight {i} is not invariant at matched β: {x} vs {y}"
            );
        }
    }

    /// (b) Exact KL budget at convergence.
    #[test]
    fn solved_beta_hits_kl_budget() {
        let rewards = [0.1f32, 0.4, 0.35, 0.8, 0.5];
        let beta = solve_beta(&rewards, KL_BUDGET_LN2);
        let r_max = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let kl = kl_vs_uniform(&rewards, beta, r_max);
        assert!(
            (kl - KL_BUDGET_LN2).abs() < 1e-3,
            "solved β achieves KL {kl}, budget {KL_BUDGET_LN2}"
        );
    }

    /// (c) Degenerate groups: all-equal rewards → uniform weights, zero
    /// advantages, regardless of β.
    #[test]
    fn all_equal_rewards_yield_uniform_weights() {
        let rewards = [0.5f32; 6];
        let mut adv = Vec::new();
        let beta = tilt_advantages_into(&rewards, KL_BUDGET_LN2, &mut adv);
        assert_eq!(adv.len(), 6);
        for (i, &a) in adv.iter().enumerate() {
            assert!(a.abs() < 1e-5, "advantage {i} should be 0, got {a}");
        }
        // Any β yields uniform here — the returned β is the (harmless)
        // bracket value; weights must not depend on it.
        let mut w = Vec::new();
        tilted_weights(&rewards, beta, &mut w);
        for &x in &w {
            assert!((x - 1.0).abs() < 1e-5, "weight should be 1.0, got {x}");
        }
    }

    /// (c') The load-bearing degenerate case for RL: a group where every
    /// rollout scored the same must produce NO update. This is the arithmetic
    /// that Plan 336 Bench 468 found z-scored GRPO getting right by accident
    /// (advantage bit pattern 0x00000000) — the entropic form must match it
    /// exactly, in BOTH the batch and LOO denominators.
    #[test]
    fn degenerate_group_produces_no_update_in_either_form() {
        for rewards in [[0.0f32; 8], [1.0f32; 8], [-3.5f32; 8]] {
            let mut batch = Vec::new();
            let mut loo = Vec::new();
            tilt_advantages_into(&rewards, KL_BUDGET_LN2, &mut batch);
            tilt_advantages_loo_into(&rewards, KL_BUDGET_LN2, &mut loo);
            for (i, (&b, &l)) in batch.iter().zip(&loo).enumerate() {
                assert!(b.abs() < 1e-6, "batch advantage {i} = {b}, want 0");
                assert!(l.abs() < 1e-6, "LOO advantage {i} = {l}, want 0");
            }
        }
    }

    /// (c) Degenerate groups: a single candidate has no shape.
    #[test]
    fn single_element_group_is_neutral() {
        let rewards = [0.7f32];
        let mut adv = Vec::new();
        tilt_advantages_into(&rewards, KL_BUDGET_LN2, &mut adv);
        assert_eq!(adv.len(), 1);
        // w = n·q = 1·1 = 1 → A = 0.
        assert!(adv[0].abs() < 1e-5);

        // LOO has no baseline at all here; it must still emit one neutral
        // entry rather than dividing by an empty denominator.
        let mut loo = Vec::new();
        tilt_advantages_loo_into(&rewards, KL_BUDGET_LN2, &mut loo);
        assert_eq!(loo.len(), 1);
        assert!(loo[0].abs() < 1e-5);
        assert!(loo[0].is_finite());
    }

    /// Empty groups must not panic or emit spurious entries.
    #[test]
    fn empty_group_yields_empty_output() {
        let mut adv = vec![9.0f32; 4]; // pre-dirtied scratch
        tilt_advantages_into(&[], KL_BUDGET_LN2, &mut adv);
        assert!(adv.is_empty());
        let mut loo = vec![9.0f32; 4];
        tilt_advantages_loo_into(&[], KL_BUDGET_LN2, &mut loo);
        assert!(loo.is_empty());
    }

    /// (d) Rare-success binary group upweights the successes (the RS-GRPO
    /// exploration-dilemma shape) — and stays zero-sum in the batch form.
    #[test]
    fn rare_successes_get_upweighted() {
        let rewards = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut adv = Vec::new();
        tilt_advantages_into(&rewards, KL_BUDGET_LN2, &mut adv);
        assert!(adv[7] > 0.0, "the rare success must be upweighted");
        for (i, &a) in adv[..7].iter().enumerate() {
            assert!(a < 0.0, "failure {i} must be downweighted, got {a}");
        }
        let sum: f32 = adv.iter().sum();
        assert!(sum.abs() < 1e-4, "advantages must be zero-sum, sum {sum}");
    }

    /// (d') LOO preserves the sign structure — the property the policy
    /// gradient actually consumes — while NOT being zero-sum.
    #[test]
    fn loo_preserves_sign_structure_but_not_zero_sum() {
        let rewards = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut loo = Vec::new();
        tilt_advantages_loo_into(&rewards, KL_BUDGET_LN2, &mut loo);
        assert!(
            loo[7] > 0.0,
            "the rare success must be upweighted under LOO"
        );
        for (i, &a) in loo[..7].iter().enumerate() {
            assert!(a < 0.0, "failure {i} must be downweighted, got {a}");
        }
        assert!(
            loo.iter().all(|a| a.is_finite()),
            "LOO must not emit inf/NaN on a saturated binary group"
        );
    }

    /// LOO's debiasing must exceed the batch form's self-contribution shrinkage
    /// at the small group sizes self-play uses. The batch denominator includes
    /// the sample's own mass, pulling |A| toward 0; LOO removes it, so the
    /// winner's advantage must be strictly larger in magnitude.
    #[test]
    fn loo_debiases_the_self_contribution() {
        let rewards = [0.1f32, 0.2, 0.9, 0.3];
        let (mut batch, mut loo) = (Vec::new(), Vec::new());
        tilt_advantages_into(&rewards, KL_BUDGET_LN2, &mut batch);
        tilt_advantages_loo_into(&rewards, KL_BUDGET_LN2, &mut loo);
        assert!(
            loo[2] > batch[2],
            "LOO must not shrink the winner: LOO {} vs batch {}",
            loo[2],
            batch[2]
        );
    }

    /// Auto-adaptation: bigger reward gaps need SMALLER β to hit the same
    /// budget (the outlier group is prevented from tilting harder just
    /// because its outlier is extreme).
    #[test]
    fn beta_shrinks_as_gaps_grow() {
        let tight = [0.49f32, 0.50, 0.51]; // consistent small differences
        let wide = [0.0f32, 0.5, 1.0]; // outlier-dominated
        let b_tight = solve_beta(&tight, KL_BUDGET_LN2);
        let b_wide = solve_beta(&wide, KL_BUDGET_LN2);
        assert!(
            b_wide < b_tight,
            "wider gaps should need smaller β: wide {b_wide} vs tight {b_tight}"
        );
    }

    /// The budget caps concentration: the max tilted weight is bounded well
    /// below the one-hot value (n), for both moderate and extreme groups.
    #[test]
    #[allow(clippy::cast_precision_loss)] // group sizes ≤ ~10²
    fn kl_budget_bounds_max_weight() {
        let moderate = [0.4f32, 0.5, 0.6];
        let extreme = [0.0f32, 0.0, 1.0];
        for group in [moderate, extreme] {
            let beta = solve_beta(&group, KL_BUDGET_LN2);
            let mut w = Vec::new();
            tilted_weights(&group, beta, &mut w);
            let w_max = w.iter().copied().fold(0.0f32, f32::max);
            // One-hot at n = 3 would be w = 3.0; the ln-2 budget must keep
            // the max meaningfully below that.
            assert!(
                w_max < 2.9,
                "budget must cap the max weight ({w_max} for {group:?})"
            );
            // And the mean stays 1 (zero-sum advantages).
            let mean: f32 = w.iter().sum::<f32>() / w.len() as f32;
            assert!((mean - 1.0).abs() < 1e-4);
        }
    }

    /// Determinism: identical inputs → bit-identical outputs. Load-bearing for
    /// the Phase 2 A/B — a non-deterministic estimator would make the two arms
    /// incomparable across seeds.
    #[test]
    fn tilt_is_deterministic() {
        let rewards = [0.3f32, 0.31, 0.9, 0.42, 0.15];
        let (mut a1, mut a2) = (Vec::new(), Vec::new());
        let b1 = tilt_advantages_into(&rewards, KL_BUDGET_LN2, &mut a1);
        let b2 = tilt_advantages_into(&rewards, KL_BUDGET_LN2, &mut a2);
        assert_eq!(b1.to_bits(), b2.to_bits());
        assert_eq!(a1.len(), a2.len());
        for (x, y) in a1.iter().zip(&a2) {
            assert_eq!(x.to_bits(), y.to_bits());
        }

        let (mut l1, mut l2) = (Vec::new(), Vec::new());
        tilt_advantages_loo_into(&rewards, KL_BUDGET_LN2, &mut l1);
        tilt_advantages_loo_into(&rewards, KL_BUDGET_LN2, &mut l2);
        for (x, y) in l1.iter().zip(&l2) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    /// G2 alloc-freedom (the honest equivalent of a `CountingAllocator`
    /// spot-check for a pure-f32 module): with pre-allocated scratch, the
    /// solve+weights path never grows capacity across many calls.
    #[test]
    #[allow(clippy::cast_precision_loss)] // loop counters → f32 reward jitter
    fn tilt_scratch_capacity_stabilizes() {
        let rewards = [0.9f32, 0.4, 0.4, 0.55, 0.3, 0.62];
        let mut out = Vec::new();
        let mut loo_out = Vec::new();
        // Warm-up: first call sizes the scratch.
        tilt_advantages_into(&rewards, KL_BUDGET_LN2, &mut out);
        tilt_advantages_loo_into(&rewards, KL_BUDGET_LN2, &mut loo_out);
        let cap_after_warmup = out.capacity();
        let loo_cap_after_warmup = loo_out.capacity();
        for k in 0..1000u32 {
            // Vary rewards slightly — same length, different values.
            let r: Vec<f32> = rewards
                .iter()
                .enumerate()
                .map(|(i, &x)| (x + (k % 7) as f32 * 0.001 * (i as f32 + 1.0)) % 1.0)
                .collect();
            tilt_advantages_into(&r, KL_BUDGET_LN2, &mut out);
            tilt_advantages_loo_into(&r, KL_BUDGET_LN2, &mut loo_out);
            assert_eq!(out.capacity(), cap_after_warmup, "capacity grew mid-loop");
            assert_eq!(
                loo_out.capacity(),
                loo_cap_after_warmup,
                "LOO capacity grew mid-loop"
            );
        }
    }
}
