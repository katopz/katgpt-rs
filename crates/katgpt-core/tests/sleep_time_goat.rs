//! Plan 334 — Sleep-Time Query Anticipator GOAT gate (G1/G2/G7).
//!
//! Synthetic math gates only — proves the mechanics, not the game quality.
//! The quality gates G2/G3/G4 (predictability correlation, cross-player
//! amortization economics, wake-time latency under load) require a real
//! predictability-labeled corpus and live in riir-ai Plan 341.
//!
//! - **G1 mechanics** — anticipate/consume round-trip, blend correctness,
//!   predictability range, determinism.
//! - **G2 cost model correctness** — amortization factor matches paper §5.3,
//!   `should_pre_compute` boundary, monotonicity.
//! - **G7 BLAKE3 commitment** — tamper detection, determinism, slot-level
//!   commitment.
//!
//! The G5 zero-alloc gate lives in a separate binary
//! (`sleep_time_alloc_check.rs`) so the CountingAllocator doesn't pick up
//! allocations from these tests running in parallel. The G6 latency gate
//! lives in `benches/sleep_time_consume_bench.rs`.

#![cfg(feature = "sleep_time_anticipation")]

use katgpt_core::sleep_time::{
    consume, consume_gate, AmortizationCostModel, AnticipatedQueryDir, AnticipatedQuerySet,
    AnticipatedSlot, DotPredictabilityScorer, IdentityFunctorOp, PredictabilityScorer,
    SleepTimeAnticipator, SleepTimeScratch,
};

// ─────────────────────────────────────────────────────────────────────
// G1 — mechanics
// ─────────────────────────────────────────────────────────────────────

/// G1.1: anticipate() produces K slots, each populated.
#[test]
fn g1_anticipate_emits_populated_slots() {
    const D: usize = 4;
    const K: usize = 3;
    let dirs = [
        AnticipatedQueryDir::new([1.0, 0.0, 0.0, 0.0]),
        AnticipatedQueryDir::new([0.0, 1.0, 0.0, 0.0]),
        AnticipatedQueryDir::new([0.0, 0.0, 1.0, 0.0]),
    ];
    let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
        op: IdentityFunctorOp,
        scorer: DotPredictabilityScorer::default(),
        budgets: [128; K],
        tau: 0.5,
        beta: 4.0,
    };
    let c = [0.7, 0.0, 0.0, 0.0];
    let mut scratch = SleepTimeScratch::new();
    let artifact = anticipator.anticipate(&c, &dirs, &mut scratch);

    assert_eq!(artifact.slots.len(), K);
    for (i, slot) in artifact.slots.iter().enumerate() {
        assert_eq!(
            slot.dir.blake3, dirs[i].blake3,
            "slot {} dir commitment mismatch",
            i
        );
        assert!(
            (0.0..=1.0).contains(&slot.predictability),
            "slot {} predictability {} out of [0,1]",
            i,
            slot.predictability
        );
    }
}

/// G1.2: consume() determinism — same (q, c', tau, beta, fresh_think) → same out.
#[test]
fn g1_consume_is_deterministic() {
    const D: usize = 4;
    const K: usize = 2;
    let dirs = [
        AnticipatedQueryDir::new([1.0; D]),
        AnticipatedQueryDir::new([-1.0; D]),
    ];
    let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
        op: IdentityFunctorOp,
        scorer: DotPredictabilityScorer::default(),
        budgets: [100; K],
        tau: 0.5,
        beta: 4.0,
    };
    let c = [0.3; D];
    let mut scratch = SleepTimeScratch::new();
    let artifact = anticipator.anticipate(&c, &dirs, &mut scratch);

    let q = [0.5; D];
    let mk_fresh = |closure_seed: u32| move |fq: &[f32; D]| {
        let mut z = [0.0f32; D];
        for j in 0..D {
            z[j] = fq[j] + closure_seed as f32;
        }
        z
    };
    let out1 = consume(&q, &artifact, 0.5, 4.0, mk_fresh(1));
    let out2 = consume(&q, &artifact, 0.5, 4.0, mk_fresh(1));
    for j in 0..D {
        assert_eq!(out1[j].to_bits(), out2[j].to_bits(), "dim {} not bit-identical", j);
    }
}

/// G1.3: consume() smooth blend — at gate=0.5, out is exactly 50/50 blend.
#[test]
fn g1_consume_blend_is_smooth() {
    const D: usize = 2;
    const K: usize = 1;
    // Hand-built artifact with predictability exactly == tau → gate = sigmoid(0) = 0.5.
    let slots = [AnticipatedSlot {
        dir: AnticipatedQueryDir::new([1.0, 0.0]),
        precomputed: [10.0, 20.0],
        predictability: 0.5,
    }];
    let blake3 = AnticipatedQuerySet::<D, K>::commit_slots(&slots);
    let artifact = AnticipatedQuerySet {
        slots,
        blake3,
        version: 0,
    };
    let q = [1.0, 0.0];
    let out = consume(&q, &artifact, 0.5, 4.0, |_| [30.0, 40.0]);
    // 0.5 * [10, 20] + 0.5 * [30, 40] = [20, 30].
    assert!((out[0] - 20.0).abs() < 1e-6, "blend x: {}", out[0]);
    assert!((out[1] - 30.0).abs() < 1e-6, "blend y: {}", out[1]);
}

/// G1.4: predictability ∈ [0,1] for all (c, dir) inputs.
#[test]
fn g1_predictability_range_in_unit_interval() {
    let scorer = DotPredictabilityScorer::new(3.0, -2.0);
    let dir = AnticipatedQueryDir::new([1.0; 8]);
    for &scale in &[-100.0f32, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
        let c = [scale; 8];
        let p = scorer.predictability(&c, &dir);
        assert!(
            (0.0..=1.0).contains(&p),
            "predictability {} out of [0,1] at scale {}",
            p,
            scale
        );
    }
}

/// G1.5: consume_gate() returns (best_i, gate) consistent with the artifact.
#[test]
fn g1_consume_gate_finds_best_match() {
    const D: usize = 2;
    const K: usize = 2;
    let dirs = [
        AnticipatedQueryDir::new([1.0, 0.0]),
        AnticipatedQueryDir::new([0.0, 1.0]),
    ];
    let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
        op: IdentityFunctorOp,
        scorer: DotPredictabilityScorer::default(),
        budgets: [100; K],
        tau: 0.5,
        beta: 4.0,
    };
    let c = [1.0, 1.0];
    let mut scratch = SleepTimeScratch::new();
    let artifact = anticipator.anticipate(&c, &dirs, &mut scratch);

    // Query aligned with dir[1].
    let q = [0.0, 1.0];
    let (best_i, gate) = consume_gate(&q, &artifact, 0.5, 4.0);
    assert_eq!(best_i, 1, "best match should be slot 1 (aligned with q)");
    assert!((0.0..=1.0).contains(&gate), "gate out of [0,1]");
}

// ─────────────────────────────────────────────────────────────────────
// G2 — cost model correctness
// ─────────────────────────────────────────────────────────────────────

/// G2.1: amortization factor < 1 at the paper's reference point (N=10, e_gate=0.5).
#[test]
fn g2_amortization_factor_wins_at_paper_reference() {
    let m = AmortizationCostModel::default();
    // Paper §5.3: ~2.5× gain at N=10 → factor ≈ 0.4. Sleep budget is the
    // caller's choice; pick a moderate one.
    let sleep_cost = 5_000.0_f32;
    let factor = m.amortization_factor(sleep_cost, 10, 0.5);
    assert!(
        factor < 1.0,
        "G2 FAIL: pre-compute should win at e_gate=0.5, N=10; got factor {}",
        factor
    );
}

/// G2.2: `should_pre_compute` flips at the break-even boundary.
#[test]
fn g2_should_pre_compute_boundary() {
    let m = AmortizationCostModel::default();
    let e_gate = 0.5;
    let n = 10u32;
    let break_even = (n as f32) * m.t * (m.b_max as f32) * e_gate;
    assert!(
        m.should_pre_compute(0.99 * break_even, n, e_gate),
        "below break-even should pre-compute"
    );
    assert!(
        !m.should_pre_compute(1.01 * break_even, n, e_gate),
        "above break-even should NOT pre-compute"
    );
}

/// G2.3: total_cost is monotone decreasing in e_gate (paper §5.3 headline).
#[test]
fn g2_total_cost_monotone_decreasing() {
    let m = AmortizationCostModel::default();
    let sleep_cost = 10_000.0_f32;
    let n = 10u32;
    let mut prev = f32::INFINITY;
    let mut e = 0.0f32;
    while e <= 1.0 {
        let c = m.total_cost(sleep_cost, n, e);
        assert!(
            c <= prev + 1e-6,
            "total_cost not monotone decreasing in e_gate: e={} cost={} prev={}",
            e,
            c,
            prev
        );
        prev = c;
        e += 0.1;
    }
}

/// G2.4: break_even_n solves the should_pre_compute equation.
#[test]
fn g2_break_even_n_consistency() {
    let m = AmortizationCostModel::default();
    let sleep_cost = 10_000.0_f32;
    let e_gate = 0.4;
    let n_be = m.break_even_n(sleep_cost, e_gate);
    let n_above = (n_be.ceil() as u32) + 1;
    let n_below = (n_be.floor() as u32).saturating_sub(1).max(1);
    assert!(
        m.should_pre_compute(sleep_cost, n_above, e_gate),
        "above break-even N should pre-compute"
    );
    assert!(
        !m.should_pre_compute(sleep_cost, n_below, e_gate),
        "below break-even N should NOT pre-compute"
    );
}

// ─────────────────────────────────────────────────────────────────────
// G7 — BLAKE3 commitment
// ─────────────────────────────────────────────────────────────────────

/// G7.1: anticipate() with same inputs → same BLAKE3 (determinism).
#[test]
fn g7_anticipate_commitment_deterministic() {
    const D: usize = 4;
    const K: usize = 2;
    let dirs = [
        AnticipatedQueryDir::new([1.0; D]),
        AnticipatedQueryDir::new([-1.0; D]),
    ];
    let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
        op: IdentityFunctorOp,
        scorer: DotPredictabilityScorer::default(),
        budgets: [100; K],
        tau: 0.5,
        beta: 4.0,
    };
    let c = [0.3; D];
    let mut scratch = SleepTimeScratch::new();
    let a1 = anticipator.anticipate(&c, &dirs, &mut scratch);
    let a2 = anticipator.anticipate(&c, &dirs, &mut scratch);
    assert_eq!(a1.blake3, a2.blake3, "deterministic anticipate");
    assert!(a1.verify_commitment(), "commitment verifies");
}

/// G7.2: changing any slot field (precomputed or predictability) changes the BLAKE3.
#[test]
fn g7_tamper_detection() {
    let mk_slots = || {
        [
            AnticipatedSlot {
                dir: AnticipatedQueryDir::new([1.0, 0.0]),
                precomputed: [2.0, 3.0],
                predictability: 0.8,
            },
            AnticipatedSlot {
                dir: AnticipatedQueryDir::new([0.0, 1.0]),
                precomputed: [4.0, 5.0],
                predictability: 0.6,
            },
        ]
    };
    let h_clean = AnticipatedQuerySet::<2, 2>::commit_slots(&mk_slots());

    // Tamper precomputed[0] by 1 ULP.
    let mut tampered = mk_slots();
    tampered[0].precomputed[0] = f32::from_bits(tampered[0].precomputed[0].to_bits() + 1);
    let h_precomp = AnticipatedQuerySet::<2, 2>::commit_slots(&tampered);
    assert_ne!(h_clean, h_precomp, "tamper precomputed must change BLAKE3");

    // Tamper predictability[1] by 1 ULP.
    let mut tampered = mk_slots();
    tampered[1].predictability = f32::from_bits(tampered[1].predictability.to_bits() + 1);
    let h_pred = AnticipatedQuerySet::<2, 2>::commit_slots(&tampered);
    assert_ne!(h_clean, h_pred, "tamper predictability must change BLAKE3");

    // Tamper dir[0] (which has its own commitment) by 1 ULP on direction.
    let mut tampered = mk_slots();
    tampered[0].dir.direction[0] = f32::from_bits(tampered[0].dir.direction[0].to_bits() + 1);
    // Note: we re-blake3 the dir to keep the slot's dir.blake3 consistent.
    // The slot commitment only reads dir.blake3, not dir.direction — so this
    // tamper is detectable only via dir.verify_commitment(), NOT via the set
    // commitment (which trusts dir.blake3). This is by design: the dir's own
    // blake3 is its commitment; the set commitment hashes the dir commitments,
    // not the raw direction bytes.
    let h_dir_via_trusted_blake3 = AnticipatedQuerySet::<2, 2>::commit_slots(&tampered);
    // h_dir_via_trusted_blake3 should equal h_clean (dir.blake3 unchanged).
    assert_eq!(
        h_clean, h_dir_via_trusted_blake3,
        "set commitment trusts dir.blake3, not raw direction"
    );
    // But dir.verify_commitment() catches the tamper.
    assert!(
        !tampered[0].dir.verify_commitment(),
        "dir tamper must be caught by dir.verify_commitment()"
    );
}

/// G7.3: AnticipatedQuerySet::verify_commitment() is a working audit hook.
#[test]
fn g7_verify_commitment_audit_hook() {
    const D: usize = 2;
    const K: usize = 2;
    let dirs = [
        AnticipatedQueryDir::new([1.0, 0.0]),
        AnticipatedQueryDir::new([0.0, 1.0]),
    ];
    let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
        op: IdentityFunctorOp,
        scorer: DotPredictabilityScorer::default(),
        budgets: [100; K],
        tau: 0.5,
        beta: 4.0,
    };
    let c = [0.5, 0.5];
    let mut scratch = SleepTimeScratch::new();
    let artifact = anticipator.anticipate(&c, &dirs, &mut scratch);
    assert!(artifact.verify_commitment(), "fresh artifact must verify");

    // Tamper in place: corrupt blake3, verify should fail.
    let mut corrupted = artifact.clone();
    corrupted.blake3[0] = corrupted.blake3[0].wrapping_add(1);
    assert!(
        !corrupted.verify_commitment(),
        "corrupted blake3 must fail verify"
    );
}

/// G7.4: AnticipatedQueryDir commitment is bit-distinguishable on 1-ULP changes.
#[test]
fn g7_direction_commitment_ulp_sensitive() {
    let d1 = AnticipatedQueryDir::new([1.0, 2.0, 3.0]);
    let d2 = AnticipatedQueryDir::new([1.0, 2.0, f32::from_bits(3.0f32.to_bits() + 1)]);
    assert_ne!(
        d1.blake3, d2.blake3,
        "1-ULP direction change must produce different BLAKE3"
    );
}
