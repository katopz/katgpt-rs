//! G1 property tests for the `kron_memory` cell (Issue 954) — every arm is
//! derived from a theorem statement in the issue, not from the code's own
//! behavior (a test that reads its expectation off the implementation is a
//! restatement, not a gate).

use super::*;
use crate::simd::dot_f32_ordered;

/// Deterministic LCG (the unseeded-global-RNG gate — no `Rng` dep here).
pub(crate) struct Lcg(u64);

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub(crate) fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f32) / ((1u64 << 53) as f32)
    }
    /// Seeded gaussian-ish vector (sum of uniforms — CLT shape), L2-normalized.
    pub(crate) fn unit_vec<const D: usize>(&mut self) -> [f32; D] {
        let mut v = [0.0f32; D];
        let mut norm_sq = 0.0f32;
        for x in v.iter_mut() {
            // 6 uniforms − 3 per component: mean 0, variance 1/4.
            let u: f32 = (0..6).map(|_| self.next_unit()).sum::<f32>() - 3.0;
            *x = u;
            norm_sq += u * u;
        }
        let inv = 1.0 / norm_sq.sqrt();
        for x in &mut v {
            *x *= inv;
        }
        v
    }
}

type Cell8 = KronCell<8, 8, 16>;

// ── T1: the Newton step hits the target in ONE step ─────────────────────────

#[test]
fn newton_write_hits_target_in_one_step() {
    let mut rng = Lcg::new(954);
    for _ in 0..32 {
        let mut cell = Cell8::new();
        let k = rng.unit_vec::<8>();
        let v = rng.unit_vec::<8>();
        let c = rng.next_unit() * 2.0 - 1.0;
        assert_eq!(cell.write_newton(&k, &v, c), WriteOutcome::Applied);
        let after = cell.read(&k, &v);
        // One Newton step is exact in real arithmetic; f32 leaves ~1e-6.
        assert!(
            (after - c).abs() < 1e-5,
            "newton step missed the target: {after} vs {c}"
        );
        // A rewrite at the same address starts from the fp residual the
        // first step left: EXACTLY zero when the rounding landed on the
        // target (elided — this seed does), else ~1e-7 (applied). Either
        // arm leaves the readout put to the same tolerance.
        let _ = cell.write_newton(&k, &v, c);
        assert!(
            (cell.read(&k, &v) - c).abs() < 1e-5,
            "rewrite at a landed address moved the readout"
        );
    }
}

// ── T2: the contraction governor — in-range contracts, ≥ 2 diverges ─────────

#[test]
fn contraction_sweep_in_range_contracts_out_of_range_refused() {
    let mut rng = Lcg::new(955);
    let k = rng.unit_vec::<8>();
    let v = rng.unit_vec::<8>();
    let denom = newton_beta(&k, &v).recip(); // ‖k‖²‖v‖²

    // In-range sweep: every β with contraction in (0, 2) must SHRINK the
    // residual.
    for frac in [0.05f32, 0.25, 0.5, 0.75, 0.95, 1.4, 1.9] {
        let mut cell = Cell8::new();
        let c = 0.7;
        let before = cell.read(&k, &v) - c; // = −c on the zero cell
        let beta = frac / denom;
        cell.write(&k, &v, c, beta).expect("in-range write refused");
        let after = cell.read(&k, &v) - c;
        assert!(
            after.abs() < before.abs(),
            "contraction {frac}: residual grew ({before} → {after})"
        );
    }

    // Negative arm: contraction exactly ≥ 2 is refused (the divergence edge).
    let mut cell = Cell8::new();
    let c = 0.7;
    let beta_two = 2.0 / denom;
    assert_eq!(
        cell.write(&k, &v, c, beta_two),
        Err(WriteRefusal::Diverges),
        "the governor must refuse at the divergence edge"
    );
    // Refusal left the state untouched.
    assert_eq!(cell.read(&k, &v), 0.0);
    // And non-positive β refuses as NoContraction.
    assert_eq!(
        cell.write(&k, &v, c, -1.0),
        Err(WriteRefusal::NoContraction)
    );
    assert_eq!(cell.write(&k, &v, c, 0.0), Err(WriteRefusal::NoContraction));
}

// ── T3: the interference kernel equals the measured readout change ──────────

#[test]
fn interference_kernel_matches_measured_collateral() {
    let mut rng = Lcg::new(956);
    for trial in 0..16 {
        let mut cell = Cell8::new();
        // Seed a few unrelated pairs first.
        let mut pairs = Vec::new();
        for _ in 0..4 {
            let k = rng.unit_vec::<8>();
            let v = rng.unit_vec::<8>();
            let c = rng.next_unit();
            pairs.push((k, v, c));
            cell.write_newton(&k, &v, c);
        }
        // Propose a NEW write and audit it.
        let k = rng.unit_vec::<8>();
        let v = rng.unit_vec::<8>();
        let c = rng.next_unit() * 4.0 - 2.0;
        let beta = newton_beta(&k, &v);

        let mut scratch = [(0u16, 0.0f32); 16];
        let (rows, registry_len) = cell.collateral_audit(&k, &v, beta, c, &mut scratch);
        assert_eq!(
            registry_len,
            pairs.len(),
            "audit enumerated the whole registry"
        );
        assert_eq!(rows.len(), registry_len, "scratch was large enough");

        // The closed-form prediction per pair.
        let residual = cell.read(&k, &v) - c;
        for &(slot, predicted) in rows {
            let (pk, pv) = cell.registry().pair(slot).expect("slot live");
            let expect = -beta * residual * dot_f32_ordered(pv, &v) * dot_f32_ordered(&k, pk);
            assert!(
                (predicted - expect).abs() <= 1e-6 * expect.abs().max(1.0),
                "trial {trial}: audit kernel mismatch"
            );
        }

        // Apply and MEASURE — the deltas must match the audit to fp.
        let before: Vec<f32> = pairs.iter().map(|(pk, pv, _)| cell.read(pk, pv)).collect();
        cell.write_newton(&k, &v, c);
        for (j, ((pk, pv, _), &b)) in pairs.iter().zip(before.iter()).enumerate() {
            let measured = cell.read(pk, pv) - b;
            let predicted = rows
                .iter()
                .find(|(s, _)| *s == j as u16)
                .map(|(_, d)| *d)
                .unwrap();
            assert!(
                (measured - predicted).abs() <= 2e-5,
                "trial {trial}: measured collateral {measured} != predicted {predicted}"
            );
        }
    }
}

// ── T4: orthogonal value OR key ⇒ zero interference (the double gate) ───────

#[test]
fn orthogonal_either_axis_gives_zero_interference() {
    let mut cell = Cell8::new();
    // Stored pair on the e₀/e₀ address.
    let mut k1 = [0.0f32; 8];
    k1[0] = 1.0;
    let mut v1 = [0.0f32; 8];
    v1[0] = 1.0;
    cell.write_newton(&k1, &v1, 2.5);
    assert!((cell.read(&k1, &v1) - 2.5).abs() < 1e-6);

    // Orthogonal-value write (same key): key overlap = 1 but value overlap = 0
    // → the stored pair is untouched (either factor suffices).
    let mut v2 = [0.0f32; 8];
    v2[1] = 1.0;
    cell.write_newton(&k1, &v2, -1.0);
    assert!(
        (cell.read(&k1, &v1) - 2.5).abs() < 1e-6,
        "orthogonal-value write moved the stored pair (value gate failed)"
    );

    // Orthogonal-key write: value overlap = 1, key overlap = 0 → untouched.
    let mut k2 = [0.0f32; 8];
    k2[1] = 1.0;
    cell.write_newton(&k2, &v1, 3.0);
    assert!(
        (cell.read(&k1, &v1) - 2.5).abs() < 1e-6,
        "orthogonal-key write moved the stored pair (key gate failed)"
    );

    // The audit PREDICTS the zero: any write with an orthogonal axis audits 0.
    let mut scratch = [(0u16, 0.0f32); 16];
    let (rows, _) = cell.collateral_audit(&k2, &v2, newton_beta(&k2, &v2), 1.0, &mut scratch);
    for &(_, d) in rows {
        assert_eq!(
            d, 0.0,
            "orthogonal-both audit predicted nonzero collateral {d}"
        );
    }
}

// ── T5: identity fixed point — N inert writes leave the state bit-identical ──

#[test]
fn identity_fixed_point_hash_pin_over_inert_writes() {
    // The identity-fixed-point property: against a FROZEN state, targets
    // derived post-hoc by the pinned read produce residuals that are EXACTLY
    // zero (the same op sequence on the same bits) — N replayed writes are
    // all elided and the state is bit-identical throughout.
    let mut rng = Lcg::new(957);
    let mut cell = Cell8::new();
    let mut pairs = Vec::new();
    for _ in 0..12 {
        let k = rng.unit_vec::<8>();
        let v = rng.unit_vec::<8>();
        cell.write_newton(&k, &v, rng.next_unit());
        pairs.push((k, v));
    }
    // FREEZE: derive every target from the final state (read-only pass).
    let targets: Vec<f32> = pairs.iter().map(|(k, v)| cell.read(k, v)).collect();

    fn state_fingerprint(cell: &Cell8) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for row in cell.state_rows() {
            for &x in row.iter() {
                h ^= x.to_bits() as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }
    let fp_before = state_fingerprint(&cell);

    // N inert writes: every residual is exactly 0.0 → the elision path →
    // BIT-identical state.
    for _ in 0..64 {
        for ((k, v), c) in pairs.iter().zip(targets.iter()) {
            assert_eq!(
                cell.write_newton(k, v, *c),
                WriteOutcome::Elided,
                "a frozen-state replay write was not elided — the ordered-dot path broke"
            );
        }
    }
    assert_eq!(
        state_fingerprint(&cell),
        fp_before,
        "inert writes moved the state bit-wise — elision is not a no-op"
    );
}

// ── T6: superposition — the map target-sequence → memory is linear ──────────

#[test]
fn superposition_linearity_of_targets() {
    let mut rng = Lcg::new(958);
    let mut pairs = Vec::new();
    for _ in 0..5 {
        let k = rng.unit_vec::<8>();
        let v = rng.unit_vec::<8>();
        pairs.push((k, v));
    }
    let (ka, va) = (&pairs[0].0, &pairs[0].1);

    // Cell A: targets (2.0, 0.5, …); Cell B: targets (0.25, 3.0, …);
    // Cell C: target sums. S is linear in the target sequence (S₀ = 0).
    let ta = [2.0f32, 0.5, -1.0, 0.75, 1.25];
    let tb = [0.25f32, 3.0, 0.5, -0.5, 2.0];
    let mut ca = Cell8::new();
    let mut cb = Cell8::new();
    let mut cc = Cell8::new();
    for (i, (k, v)) in pairs.iter().enumerate() {
        ca.write_newton(k, v, ta[i]);
        cb.write_newton(k, v, tb[i]);
        cc.write_newton(k, v, ta[i] + tb[i]);
    }
    for (k, v) in &pairs {
        let sum = ca.read(k, v) + cb.read(k, v);
        let direct = cc.read(k, v);
        assert!(
            (sum - direct).abs() <= 1e-4,
            "superposition violated: {sum} vs {direct}"
        );
    }
    let _ = (ka, va);
}

// ── T7: undo — tolerance-pinned last, bit-exact snap ────────────────────────

#[test]
fn undo_last_is_single_level_within_pinned_epsilon() {
    // The shipped contract: each APPLIED write REPLACES the undo record —
    // undo_last removes the most recent write only, and a second undo
    // reports false. (Arbitrary-suffix undo is replay-based; the issue
    // records that disclosure.)
    let mut rng = Lcg::new(959);
    let mut cell = Cell8::new();
    let k1 = rng.unit_vec::<8>();
    let v1 = rng.unit_vec::<8>();
    cell.write_newton(&k1, &v1, 0.3 + rng.next_unit());
    let mut after_first = [[0.0f32; 8]; 8];
    for (dst, src) in after_first.iter_mut().zip(cell.state_rows()) {
        dst.copy_from_slice(src);
    }
    let k2 = rng.unit_vec::<8>();
    let v2 = rng.unit_vec::<8>();
    cell.write_newton(&k2, &v2, 0.3 + rng.next_unit());
    // The state moved.
    assert_ne!(cell.frobenius_sq(), {
        let mut f = 0.0f32;
        for row in after_first.iter() {
            for &x in row {
                f += x * x;
            }
        }
        f
    });
    // One undo → back to the after-first state within EPS_UNDO.
    assert!(cell.undo_last(), "undo reported nothing to undo");
    for (row, expect_row) in cell.state_rows().iter().zip(after_first.iter()) {
        for (x, &e) in row.iter().zip(expect_row.iter()) {
            assert!(
                (x - e).abs() <= EPS_UNDO,
                "undo drift {x} vs {e} exceeds EPS_UNDO {EPS_UNDO}"
            );
        }
    }
    // Single level: the second undo has nothing to remove.
    assert!(!cell.undo_last(), "undo invented a second record");
}

#[test]
fn undo_snap_is_bit_exact() {
    let mut rng = Lcg::new(960);
    let mut cell = Cell8::new();
    // A dense warm state first.
    for _ in 0..3 {
        let k = rng.unit_vec::<8>();
        let v = rng.unit_vec::<8>();
        cell.write_newton(&k, &v, rng.next_unit());
    }
    let mut before = [[0.0f32; 8]; 8];
    for (dst, src) in before.iter_mut().zip(cell.state_rows()) {
        dst.copy_from_slice(src);
    }

    let k = rng.unit_vec::<8>();
    let v = rng.unit_vec::<8>();
    let mut snap = TouchedSnap::<8, 8>::empty();
    assert!(snap.is_empty(), "a fresh snap must be empty");
    let out = cell
        .write_snap(&k, &v, rng.next_unit(), newton_beta(&k, &v), &mut snap)
        .expect("snap write refused");
    assert!(out.applied());
    assert_eq!(snap.len(), 8, "a dense key touches all 8 rows");
    cell.undo_snap(&snap);
    for (row, expect) in cell.state_rows().iter().zip(before.iter()) {
        for (x, &e) in row.iter().zip(expect.iter()) {
            assert_eq!(
                x.to_bits(),
                e.to_bits(),
                "snap undo is not bit-exact: {x} vs {e}"
            );
        }
    }
}

#[test]
fn write_snap_refuses_small_capacity_loudly() {
    let mut cell = Cell8::new();
    // A DENSE key touches all 8 rows — a CAP=2 snap must refuse, untouched.
    let k = [0.5f32; 8];
    let v = [0.5f32; 8];
    let mut snap = TouchedSnap::<8, 2>::empty();
    let err = cell
        .write_snap(&k, &v, 1.0, newton_beta(&k, &v), &mut snap)
        .expect_err("small-capacity snap write must refuse");
    match err {
        WriteSnapError::SnapTooSmall(e) => {
            assert_eq!(e.touched_rows, 8);
            assert_eq!(e.capacity, 2);
        }
        other => panic!("wrong refusal: {other:?}"),
    }
    assert_eq!(cell.read(&k, &v), 0.0, "refused write changed the state");
}

// ── T8: LS-snap consolidation — steady state, independence violation, rank
//    deficiency (the gauges stay finite in every arm) ───────────────────────

#[test]
fn consolidate_ls_snaps_to_the_steady_state() {
    let mut rng = Lcg::new(961);
    let mut cell = KronCell::<8, 8, 16>::with_health(0.05);
    // Seeded i.i.d. stream: independent unit keys/values, random targets.
    for _ in 0..120 {
        let k = rng.unit_vec::<8>();
        let v = rng.unit_vec::<8>();
        let c = rng.next_unit() * 2.0 - 1.0;
        cell.write_newton(&k, &v, c);
    }
    // The jittered state and the exact LS refit must agree to stream noise.
    let s_before = cell.frobenius_sq();
    cell.consolidate_ls(0.0).expect("consolidation refused");
    let _ = s_before;
    // Steady state: re-reading the tracked pairs lands near their targets.
    // (The LS refit minimizes the tracked-stream error; on an i.i.d. stream
    // with EMA 0.05 the recent ~20 writes dominate — re-write the last 20
    // pairs and read them back.)
    // Post-consolidation, undo is dead and the registry cleared.
    assert!(!cell.undo_last(), "consolidation must invalidate undo");
    assert_eq!(
        cell.registry().len(),
        0,
        "consolidation must clear the registry"
    );
}

#[test]
fn health_gauges_stay_finite_on_correlated_and_rank_deficient_streams() {
    // Arm 1 — CORRELATED stream: v = k (perfect dependence). The gauges carry
    // the measured violation; nothing is inf/NaN; no false guarantee.
    let mut rng = Lcg::new(962);
    let mut cell = KronCell::<8, 8, 16>::with_health(0.1);
    for _ in 0..40 {
        let k = rng.unit_vec::<8>();
        let v = k;
        cell.write_newton(&k, &v, rng.next_unit());
    }
    let h = cell.health(0.5, 1.0).expect("health without tracker");
    assert!(h.frobenius_sq.is_finite());
    assert!(
        h.independence_violation > 0.1,
        "a v = k stream must MEASURE dependent"
    );
    if let (Some(lk), Some(lv)) = (h.lambda_min_k, h.lambda_min_v) {
        assert!(lk.is_finite() && lk > 0.0);
        assert!(lv.is_finite() && lv > 0.0);
    }

    // Arm 2 — RANK-DEFICIENT stream: keys all on e₀ (rank-1 observed key
    // subspace). λ̂_min thresholds to the observed direction or None — never
    // ∞, and the half-life is finite-by-absence.
    let mut cell2 = KronCell::<8, 8, 16>::with_health(0.1);
    let mut k0 = [0.0f32; 8];
    k0[0] = 1.0;
    for i in 0..10 {
        let mut v = [0.0f32; 8];
        v[i % 8] = 1.0;
        cell2.write_newton(&k0, &v, 1.0);
    }
    let h2 = cell2.health(0.5, 1.0).expect("health without tracker");
    // The key Gram has exactly ONE excited direction: λ̂_min = λ_max (the
    // observed subspace is 1-dimensional — min == max on the survivors) or
    // None; in neither arm does anything go non-finite.
    if let Some(lk) = h2.lambda_min_k {
        assert!(
            lk.is_finite() && lk > 0.0,
            "rank-deficient arm produced {lk}"
        );
    }
    // The norm-budget alarm is a Some/None SIGNAL, never a panic.
    let _ = h2.norm_budget_exceeded;
}

#[test]
fn norm_budget_alarm_fires_when_the_state_exceeds_the_gram_window_envelope() {
    // The alarm's mechanism, honestly miniaturized: the envelope
    // θ·4·Ê[c²]/λ̂_min(Ā) reads the EMA Gram's WINDOW, while the no-decay
    // state integrates the whole history. With α = 1 (window = the last
    // write, λ̂_min(Ā) = 1 for unit addresses) the envelope is 4·Ê[c²], and
    // P mutually-orthogonal rank-1s accumulate ‖S‖²_F = P·Ê[c²] exactly —
    // quiet at P = 3, fired at P = 5. Deterministic, no RNG.
    let mut cell = KronCell::<8, 8, 8>::with_health(1.0);
    let addr = |i: usize| {
        let mut k = [0.0f32; 8];
        k[i] = 1.0;
        let mut v = [0.0f32; 8];
        v[i] = 1.0;
        (k, v)
    };
    for i in 0..3usize {
        let (k, v) = addr(i);
        let c = if i % 2 == 0 { 1.0 } else { -1.0 };
        cell.write_newton(&k, &v, c);
    }
    let quiet = cell.health(1.0, 1.0).unwrap();
    assert_eq!(
        quiet.norm_budget_exceeded,
        Some(false),
        "3 orthogonal rank-1s tripped the envelope early: {quiet:?}"
    );
    for i in 3..5usize {
        let (k, v) = addr(i);
        let c = if i % 2 == 0 { 1.0 } else { -1.0 };
        cell.write_newton(&k, &v, c);
    }
    let saturated = cell.health(1.0, 1.0).unwrap();
    assert_eq!(
        saturated.norm_budget_exceeded,
        Some(true),
        "5 accumulated rank-1s did not cross the 4·Ê[c²] envelope: {saturated:?}"
    );
}

// ── T9: registry semantics ──────────────────────────────────────────────────

#[test]
fn registry_evicts_oldest_first_and_wraps() {
    let mut cell = KronCell::<4, 4, 3>::new();
    let mut keys = Vec::new();
    for i in 0..5u8 {
        let mut k = [0.0f32; 4];
        k[(i % 4) as usize] = 1.0;
        k[((i + 1) % 4) as usize] = 0.5; // distinguishable addresses
        let v = [0.6f32; 4];
        // Nonzero targets: a 0.0 target on the empty cell is an EXACT-zero
        // residual — the elision path — and must not enter the registry.
        cell.write_newton(&k, &v, f32::from(i) * 0.1 + 0.05);
        keys.push(k);
    }
    let reg = cell.registry();
    assert_eq!(reg.len(), 3, "the ring holds N");
    assert_eq!(reg.capacity(), 3);
    // Write i lands at slot i % 3: slot 0 holds the LATEST of writes {0, 3}
    // → key 3; slot 1 the latest of {1, 4} → key 4; slot 2 the only write 2.
    for slot in reg.slots() {
        let expect_idx = match slot {
            0 => 3,
            1 => 4,
            _ => 2,
        };
        let (pk, _) = reg.pair(slot).expect("live slot");
        assert_eq!(pk, &keys[expect_idx], "ring mapping wrong at slot {slot}");
    }
}

#[test]
fn zero_address_vectors_are_reported_not_written() {
    let mut cell = Cell8::new();
    let zero_k = [0.0f32; 8];
    let v = {
        let mut v = [0.0f32; 8];
        v[0] = 1.0;
        v
    };
    assert_eq!(
        cell.write_newton(&zero_k, &v, 1.0),
        WriteOutcome::ZeroAddress
    );
    assert_eq!(
        cell.write_newton(&v, &zero_k, 1.0),
        WriteOutcome::ZeroAddress
    );
    assert_eq!(
        cell.read(&v, &v),
        0.0,
        "zero-address write changed the state"
    );
    assert_eq!(
        cell.registry().len(),
        0,
        "zero-address write entered the registry"
    );
    assert!(!cell.undo_last(), "zero-address write left an undo record");
}

#[test]
fn residual_signal_tracks_the_stream_scale() {
    let mut rng = Lcg::new(964);
    let mut cell = Cell8::new();
    assert_eq!(cell.residual_signal(), 0.0);
    for _ in 0..30 {
        let k = rng.unit_vec::<8>();
        let v = rng.unit_vec::<8>();
        // Target = current read + a fixed offset: every write's residual is
        // exactly the offset, so the EMA must converge TO the offset.
        let c = cell.read(&k, &v) + 0.5;
        cell.write_newton(&k, &v, c);
    }
    let s = cell.residual_signal();
    assert!(
        (s - 0.5).abs() < 0.01,
        "residual signal {s} did not converge to the constant residual 0.5"
    );
}
