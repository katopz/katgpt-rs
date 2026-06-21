//! Plan 300 Phase 4 — G1 homogeneous-equivalence regression gate.
//!
//! Verifies that `PerPlayerGame` with all players sharing the same `(P, D)`
//! produces the same `ρ⋆` as `CceLp::solve(d, p)` on that single `(P, D)`.
//! This is the "wrapper is a strict generalization" check: any Plan 295
//! homogeneous use case must round-trip through the heterogeneous API without
//! behavior change.
//!
//! Covers the 3 canonical Plan 295 examples: RPS, chicken, emission-abatement.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features cce_moderator --test heterogeneous_g1 -- --nocapture
//! ```

#![cfg(feature = "cce_moderator")]

use katgpt_rs::cce::{
    CceLp, Deviation, DeviationClass, OccupationMeasure, PayoffTensor, PerPlayerGame,
};

/// Tolerance for entry-wise comparison. The BFS enumeration may produce
/// degenerate optima (multiple optima with the same objective value), but the
/// objective value must match to high precision.
const OBJ_TOL: f32 = 1e-4;
const ENTRY_TOL: f32 = 1e-3;

/// Compare homogeneous `CceLp::solve(d, p)` against `solve_heterogeneous` on a
/// `PerPlayerGame` with `n_identical` copies of `(p, d)`.
fn assert_homogeneous_equivalence<const N: usize, const A: usize, P, D>(
    p: &P,
    d: &D,
    n_identical: usize,
    label: &str,
) where
    P: PayoffTensor<N, A>,
    D: DeviationClass<N, A>,
{
    let rho_homogeneous = CceLp::new()
        .solve(d, p)
        .unwrap_or_else(|e| panic!("{label}: homogeneous LP failed: {e:?}"));

    let players: Vec<(&P, &D)> = (0..n_identical).map(|_| (p, d)).collect();
    let game = PerPlayerGame::new(players);
    let rho_heterogeneous = CceLp::new()
        .solve_heterogeneous(&game)
        .unwrap_or_else(|e| panic!("{label}: heterogeneous LP failed: {e:?}"));

    // Objective must match (gamma0 with identical players = homogeneous gamma0).
    let obj_h = p.gamma0(&rho_homogeneous);
    let obj_e = p.gamma0(&rho_heterogeneous);
    assert!(
        (obj_h - obj_e).abs() < OBJ_TOL,
        "{label}: objective mismatch homogeneous {obj_h} vs heterogeneous {obj_e}"
    );

    // Entry-by-entry (tolerant — degenerate optima may differ in support).
    for i in 0..N * A {
        let a = rho_homogeneous.entries[i];
        let b = rho_heterogeneous.entries[i];
        assert!(
            (a - b).abs() < ENTRY_TOL,
            "{label}: entry {i} mismatch homogeneous {a} vs heterogeneous {b}"
        );
    }

    eprintln!(
        "{label}: G1 PASS — homogeneous γ₀={obj_h:.6}, heterogeneous γ₀={obj_e:.6}, \
         n_identical={n_identical}"
    );
}

// ---------------------------------------------------------------------------
// Emission-abatement (N=2, A=2)
// ---------------------------------------------------------------------------

#[test]
fn g1_emission_abatement() {
    struct Emission;
    impl PayoffTensor<2, 2> for Emission {
        fn reward_follow(&self, state: usize, action: usize) -> f32 {
            const C: [[f32; 2]; 2] = [[1.0, 3.0], [2.0, 5.0]];
            C[state][action]
        }
        fn gamma0(&self, rho: &OccupationMeasure<2, 2>) -> f32 {
            self.gamma(rho)
        }
    }
    struct EmitDevs {
        v: Vec<Deviation<2, 2>>,
    }
    impl DeviationClass<2, 2> for EmitDevs {
        fn deviations(&self) -> &[Deviation<2, 2>] {
            &self.v
        }
    }
    let d = EmitDevs {
        v: vec![
            Deviation::<2, 2>::constant(0, 0),
            Deviation::<2, 2>::constant(1, 1),
        ],
    };
    let p = Emission;
    // Test with 1, 2, 4, and 8 identical players.
    for n in [1, 2, 4, 8] {
        assert_homogeneous_equivalence(&p, &d, n, "emission-abatement");
    }
}

// ---------------------------------------------------------------------------
// Chicken (N=4, A=2)
// ---------------------------------------------------------------------------

#[test]
fn g1_chicken() {
    const R: [[f32; 2]; 2] = [[3.0, 1.0], [4.0, 0.0]];

    struct Chicken;
    impl PayoffTensor<4, 2> for Chicken {
        fn reward_follow(&self, state: usize, action: usize) -> f32 {
            let s_2 = state % 2;
            -R[action][s_2]
        }
        fn gamma0(&self, rho: &OccupationMeasure<4, 2>) -> f32 {
            self.gamma(rho)
        }
    }
    struct ChickenDevs {
        v: Vec<Deviation<4, 2>>,
    }
    impl DeviationClass<4, 2> for ChickenDevs {
        fn deviations(&self) -> &[Deviation<4, 2>] {
            &self.v
        }
    }
    let d = ChickenDevs {
        v: vec![
            Deviation::<4, 2>::constant(0, 0),
            Deviation::<4, 2>::constant(1, 1),
        ],
    };
    let p = Chicken;
    // Chicken: N=4, A=2 → n_vars = 8 + 2P, n_cons = 1 + 2P.
    //   P=2 → C(12, 5) = 792       (~ms)
    //   P=4 → C(16, 9) = 11440     (~10ms)
    //   P=8 → C(24, 17) = 346104   (~15s in debug — too slow for unit test)
    for n in [1, 2, 4] {
        assert_homogeneous_equivalence(&p, &d, n, "chicken");
    }
}

// ---------------------------------------------------------------------------
// RPS (N=9, A=3)
// ---------------------------------------------------------------------------

#[test]
fn g1_rps() {
    const R: [[f32; 3]; 3] = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]];

    struct Rps;
    impl PayoffTensor<9, 3> for Rps {
        fn reward_follow(&self, state: usize, action: usize) -> f32 {
            let s_2 = state % 3;
            -R[action][s_2]
        }
        fn gamma0(&self, rho: &OccupationMeasure<9, 3>) -> f32 {
            self.gamma(rho)
        }
    }
    struct RpsDevs {
        v: Vec<Deviation<9, 3>>,
    }
    impl DeviationClass<9, 3> for RpsDevs {
        fn deviations(&self) -> &[Deviation<9, 3>] {
            &self.v
        }
    }
    let d = RpsDevs {
        v: vec![
            Deviation::<9, 3>::constant(0, 0),
            Deviation::<9, 3>::constant(1, 1),
            Deviation::<9, 3>::constant(2, 2),
        ],
    };
    let p = Rps;
    // RPS has N=9, A=3, so n_vars = 27 + P*3. C(27+3P, 1+3P) grows fast:
    //   P=1 → C(30, 4) = 27405      (~ms)
    //   P=2 → C(33, 7) = 4.27M     (~seconds)
    //   P=3 → C(36, 10) = 254M     (~minutes — too slow for unit test)
    // Test only P=1 and P=2 for RPS; the smaller-N games cover the higher
    // player counts.
    for n in [1, 2] {
        assert_homogeneous_equivalence(&p, &d, n, "rps");
    }
}
