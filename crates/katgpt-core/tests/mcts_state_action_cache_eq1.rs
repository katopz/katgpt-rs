//! Plan 390 Phase 2 T2.1 — UnMaskFork Eq. 1 property test.
//!
//! Verifies the switching-policy dominance inequality that motivates the
//! state-action pair cache:
//!
//! ```text
//!   Σ_z min_a ε_a(z)  ≤  min_a Σ_z ε_a(z)
//!      (switching)         (single best static)
//! ```
//!
//! The left-hand side is the error of a *state-dependent switching* policy
//! (pick the best action per state); the right-hand side is the error of the
//! best *single static* action applied everywhere. The inequality holds
//! pointwise (the per-state min dominates any fixed choice) and is **strict**
//! whenever no single action dominates in every state.
//!
//! This is a pure-math test — it does NOT use the cache or the search. It
//! proves the theoretical foundation before the Phase 3 benchmark tests the
//! practical cache benefit.

/// Synthetic kernel-error landscape: `ε_a(z)` for `a ∈ {0,1,2}` and `z` over a
/// discrete grid of states. Each action has a different "region of strength":
/// - action 0 wins (lowest error) in states 0..50
/// - action 1 wins in states 50..100
/// - action 2 is never the best (a "decoy" action — present to exercise the
///   min over a 3-action set without affecting the dominance result)
fn kernel_error(action: usize, z: usize) -> f32 {
    match action {
        // Action 0: low error in [0, 50), high error elsewhere.
        // Gradient kept shallow (0.002/z) so action 0 stays below the decoy
        // (action 2 at 0.3+0.001*z, max 0.399) throughout its strength region.
        0 => {
            if z < 50 {
                0.1 + 0.002 * (z as f32) // 0.10..0.20 in its strength region
            } else {
                0.5 + 0.004 * ((z - 50) as f32) // 0.50..0.70 outside
            }
        }
        // Action 1: high error in [0, 50), low error in [50, 100).
        1 => {
            if z < 50 {
                0.5 + 0.002 * (z as f32) // 0.50..0.60 outside its region
            } else {
                0.1 + 0.004 * ((z - 50) as f32) // 0.10..0.30 in its strength region
            }
        }
        // Action 2: uniformly mediocre (never the best — a decoy).
        // Stays between the winner (0.10..0.30) and the loser (0.50..0.70).
        2 => 0.3 + 0.001 * (z as f32), // 0.30..0.40
        _ => f32::INFINITY,
    }
}

/// Number of states in the grid.
const N_STATES: usize = 100;

/// Number of actions.
const N_ACTIONS: usize = 3;

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn eq1_switching_dominates_static() {
    // LHS: Σ_z min_a ε_a(z)  — the switching policy picks the best action per state.
    let mut lhs = 0.0f32;
    for z in 0..N_STATES {
        let min_err = (0..N_ACTIONS)
            .map(|a| kernel_error(a, z))
            .fold(f32::INFINITY, f32::min);
        lhs += min_err;
    }

    // RHS: min_a Σ_z ε_a(z)  — the best single static action applied everywhere.
    let mut rhs = f32::INFINITY;
    for a in 0..N_ACTIONS {
        let total: f32 = (0..N_STATES).map(|z| kernel_error(a, z)).sum();
        rhs = rhs.min(total);
    }

    // The inequality must hold: LHS ≤ RHS (switching never worse than static).
    let eps = 1e-4;
    assert!(
        lhs <= rhs + eps,
        "Eq. 1 violated: LHS (switching) = {lhs} should be <= RHS (static) = {rhs}"
    );
}

#[test]
fn eq1_strict_inequality_when_no_single_action_dominates() {
    // In this fixture, action 0 wins in [0,50) and action 1 wins in [50,100).
    // No single action dominates everywhere → the inequality must be STRICT
    // (switching is strictly better than any static choice).
    let mut lhs = 0.0f32;
    for z in 0..N_STATES {
        let min_err = (0..N_ACTIONS)
            .map(|a| kernel_error(a, z))
            .fold(f32::INFINITY, f32::min);
        lhs += min_err;
    }

    let mut rhs = f32::INFINITY;
    for a in 0..N_ACTIONS {
        let total: f32 = (0..N_STATES).map(|z| kernel_error(a, z)).sum();
        rhs = rhs.min(total);
    }

    // Strict inequality: LHS < RHS (not just ≤).
    let margin = 0.1; // expect a substantial margin, not a razor-thin one
    assert!(
        lhs < rhs - margin,
        "Eq. 1 should be strict (switching strictly better) on this fixture: \
         LHS = {lhs}, RHS = {rhs}, expected LHS < RHS - {margin}"
    );
}

#[test]
fn eq1_tight_when_one_action_dominates_everywhere() {
    // Construct a fixture where one action dominates in EVERY state.
    // Then the switching policy picks that same action everywhere, so
    // LHS == RHS (the inequality is tight — switching provides no benefit).
    fn uniform_kernel(action: usize, z: usize) -> f32 {
        match action {
            0 => 0.1 + 0.01 * (z as f32), // action 0 is best everywhere
            1 => 0.5 + 0.01 * (z as f32),
            2 => 0.3 + 0.01 * (z as f32),
            _ => f32::INFINITY,
        }
    }

    let mut lhs = 0.0f32;
    for z in 0..N_STATES {
        let min_err = (0..N_ACTIONS)
            .map(|a| uniform_kernel(a, z))
            .fold(f32::INFINITY, f32::min);
        lhs += min_err;
    }

    let mut rhs = f32::INFINITY;
    for a in 0..N_ACTIONS {
        let total: f32 = (0..N_STATES).map(|z| uniform_kernel(a, z)).sum();
        rhs = rhs.min(total);
    }

    // Tight: LHS == RHS (action 0 dominates everywhere, so switching = static).
    let eps = 1e-4;
    assert!(
        (lhs - rhs).abs() < eps,
        "Eq. 1 should be tight when one action dominates everywhere: \
         LHS = {lhs}, RHS = {rhs}, expected |LHS - RHS| < {eps}"
    );
}

#[test]
fn eq1_switching_picks_different_actions_in_different_regions() {
    // Verify that the argmin_a ε_a(z) actually switches between regions.
    // In [0,50), action 0 should win; in [50,100), action 1 should win.
    let mut action0_wins = 0usize;
    let mut action1_wins = 0usize;
    let mut action2_wins = 0usize;

    for z in 0..N_STATES {
        let best_action = (0..N_ACTIONS)
            .min_by(|&a, &b| {
                crate::float_order::cmp_for_min(kernel_error(a, z), kernel_error(b, z))
            })
            .unwrap();
        match best_action {
            0 => action0_wins += 1,
            1 => action1_wins += 1,
            2 => action2_wins += 1,
            _ => {}
        }
    }

    // Action 0 wins in [0,50), action 1 wins in [50,100), action 2 never wins.
    assert_eq!(action0_wins, 50, "action 0 should win in states 0..50");
    assert_eq!(action1_wins, 50, "action 1 should win in states 50..100");
    assert_eq!(action2_wins, 0, "action 2 (decoy) should never win");

    // This confirms the switching policy genuinely interleaves actions 0 and 1
    // — the structural condition for the strict inequality above.
}
