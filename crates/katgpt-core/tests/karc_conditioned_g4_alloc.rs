//! Issue 1239 — the G4 alloc canary for `karc_conditioned_readout`
//! (separate binary: `#[global_allocator]` is crate-binary-unique).
//!
//! The G4 law for this primitive: the ONLINE halves —
//! [`KarcConditionedReadout::apply_into`] and
//! [`KarcConditionedReadout::observe`] — allocate 0 bytes once the bank is
//! constructed. The fit helpers ([`fit_correction_into`]) are cold-path
//! consolidation work and are NOT alloc-free by design (the Gram/Cholesky
//! chain reuses [`BankFitScratch`]; only `v_out` grows once per fit).

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::{
    BankFitScratch, GateConfig, KarcConditionedReadout, archetype_direction_into,
    fit_correction_into,
};

/// A deterministic LCG (the same convention the module unit tests use).
struct Lcg(u64);
impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32
    }
}

/// A small bank (D=4, d_h=48, two keyed corrections) with gates ARMED —
/// the observe session pushes the keyed EWMA streams past `min_obs` so the
/// gated apply path is exercised, not the abstain shortcut.
/// The armed-bank fixture bundle: (bank, feature rows, truths, conditions).
type ArmedBank = (
    KarcConditionedReadout,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<u8>,
);

fn armed_bank() -> ArmedBank {
    let d = 4;
    let d_h = 48;
    let mut lcg = Lcg(0x1D39_0000_0000_0001);
    let base: Vec<f32> = (0..d * d_h).map(|_| lcg.next_unit() * 0.1).collect();
    let mut bank = KarcConditionedReadout::new(base, d, d_h, GateConfig::default());
    let u0: Vec<f32> = {
        let mut u = (0..d).map(|_| lcg.next_unit()).collect::<Vec<f32>>();
        let norm = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut u {
            *x /= norm;
        }
        u
    };
    let v0: Vec<f32> = (0..d_h).map(|_| lcg.next_unit() * 0.5).collect();
    let u1: Vec<f32> = {
        let mut u = (0..d).map(|_| lcg.next_unit()).collect::<Vec<f32>>();
        let norm = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut u {
            *x /= norm;
        }
        u
    };
    let v1: Vec<f32> = (0..d_h).map(|_| lcg.next_unit() * 0.5).collect();
    bank.add_correction(0, u0, v0);
    bank.add_correction(1, u1, v1);
    // Feature rows keyed to each condition (48 floats each).
    let feats: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..d_h).map(|_| lcg.next_unit()).collect())
        .collect();
    // Truths sized so the keyed corrections measurably help condition 0's
    // residual stream (truth = base + u0·(v0·ψ) + small noise): the gates
    // ARM and the gated apply path runs.
    let mut truth: Vec<Vec<f32>> = Vec::new();
    for (i, f) in feats.iter().enumerate() {
        let mut pred = [0.0f32; 4];
        bank.base_predict_into(f, &mut pred);
        let mut t = pred.to_vec();
        if i % 2 == 0 {
            let coef: f32 = f
                .iter()
                .zip(bank.correction(0).v.iter())
                .map(|(a, b)| a * b)
                .sum();
            for (j, u) in bank.correction(0).u.iter().enumerate() {
                t[j] += coef * u;
            }
        }
        truth.push(t);
    }
    let conds: Vec<u8> = (0..8).map(|i| (i % 2) as u8).collect();
    (bank, feats, truth, conds)
}

#[test]
fn apply_and_observe_are_zero_alloc_once_armed() {
    assert_counter_is_live();
    let (mut bank, feats, truth, conds) = armed_bank();
    let d = bank.d();
    let mut scratch = vec![0.0f32; d];
    let mut out = vec![0.0f32; d];
    // Warm: drive both keyed streams past min_obs so the gates arm and the
    // correction AXPY runs every subsequent tick.
    for round in 0..48 {
        for (i, f) in feats.iter().enumerate() {
            bank.observe(f, conds[i], &truth[i], &mut scratch);
            bank.apply_into(f, conds[i], &mut out);
            let _ = round;
        }
    }
    assert!(bank.gate_sigma(0) > 0.0, "fixture must arm the gated path");
    // ── The steady-state online session: 2000 apply + observe pairs ──
    let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    for round in 0..250 {
        for (i, f) in feats.iter().enumerate() {
            bank.observe(f, conds[i], &truth[i], &mut scratch);
            bank.apply_into(f, conds[i], &mut out);
            bank.apply_fixed(f, conds[i], 1.0, &mut out);
            let _ = round;
        }
    }
    let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "apply_into/apply_fixed/observe must allocate 0 bytes once armed"
    );
}

#[test]
fn fit_helpers_reuse_scratch_allocating_only_v_growth() {
    // The cold-path contract: the Gram/Cholesky chain stays inside the
    // scratch; across repeated fits the ONLY allocation is `v_out`'s first
    // growth (reserve covers the first, so subsequent fits allocate 0).
    let d_h = 48;
    let d = 4;
    let n = 256;
    let mut lcg = Lcg(0x1D39_0000_0000_0002);
    let feats: Vec<f32> = (0..n * d_h).map(|_| lcg.next_unit()).collect();
    let res: Vec<f32> = (0..n * d).map(|_| lcg.next_unit()).collect();
    let u = [0.25f32, 0.5, -0.5, 0.25];
    let mut scratch = BankFitScratch::with_capacity(d_h);
    let mut v = Vec::new();
    fit_correction_into(&feats, &res, &u, 1e-4, d_h, &mut scratch, &mut v);
    assert_eq!(v.len(), d_h);
    // Second fit: `v` already holds capacity — everything must be reused.
    let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    fit_correction_into(&feats, &res, &u, 1e-4, d_h, &mut scratch, &mut v);
    let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "a repeat fit with warm scratch + grown v must allocate 0"
    );
    // The archetype helper is stack-only.
    let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let mut dir = [0.0f32; 4];
    assert!(archetype_direction_into(&res, d, &mut dir));
    let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "archetype_direction_into must not allocate"
    );
}
