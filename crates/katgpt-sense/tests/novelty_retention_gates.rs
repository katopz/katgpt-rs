//! Plan 639 T1.4 — the novelty-gated retention gate suite.
//!
//! Arms: G1 identity (unarmed ≡ incumbent law, bit-exact), arithmetic
//! (κ = 0 ∧ μ = 0 armed ≡ incumbent, bit-exact), polarity (sustained step
//! LOWERS retention; pure-transient burst RAISES it), NaN, boundary
//! (`g ∈ [g_min, g_max]`), channel-freeze (case (e): `(1 − g)·drive ≠ 0`
//! whenever `drive ≠ 0`, every channel, every tick).
//!
//! The GOAT verdict (adaptation vs noise-RMSE against the fixed-α curve) is
//! Phase 2's bench, NOT this suite — this suite only proves the primitive is
//! the law it claims and cannot lock a channel.

mod novelty_fixture;

use katgpt_sense::novelty_retention::{NoveltyRetention, NoveltyRetentionConfig};
use katgpt_sense::reconstruction::{ReconstructionConfig, ReconstructionState, TripleEvidence};
use novelty_fixture::{Case, SPIKE_TICKS, STEP_TICKS, trace};

const SEEDS: [u64; 3] = [1, 2, 3];

/// Inline reference of the incumbent fixed-α additive law (Issue 777 T2).
fn incumbent_step(belief: &mut [f32; 8], act: &[f32; 6], cfg: &ReconstructionConfig) {
    let drive = drive_of(act, cfg.additive_drive_scale);
    for (b, &m) in belief.iter_mut().zip(TripleEvidence::KIND_MAP.iter()) {
        let a = cfg.belief_retention[m];
        *b = (a * *b + (1.0 - a) * drive[m]).clamp(-1.0, 1.0);
    }
}

fn drive_of(act: &[f32; 6], eta: f32) -> [f32; 6] {
    act.map(|k| 2.0 * katgpt_types::simd::fast_sigmoid(eta * k.max(0.0)) - 1.0)
}

fn bits(b: &[f32; 8]) -> [u32; 8] {
    b.map(f32::to_bits)
}

#[test]
fn g1_unarmed_is_incumbent_bit_exact() {
    let cfg = ReconstructionConfig::default();
    for case in Case::ALL {
        for seed in SEEDS {
            let tr = trace(case, seed);
            let mut st = ReconstructionState::with_config([0.0; 8], cfg);
            let mut reference = [0.0f32; 8];
            for x in &tr.observed {
                st.evolve_belief_additive(x);
                incumbent_step(&mut reference, x, &cfg);
                assert_eq!(bits(st.belief()), bits(&reference), "{case:?} seed {seed}");
            }
        }
    }
}

#[test]
fn g1_disarm_returns_to_incumbent_path() {
    let cfg = ReconstructionConfig::default();
    let mut st = ReconstructionState::with_config([0.0; 8], cfg);
    if !st.arm_novelty_retention(NoveltyRetentionConfig::default()) {
        // NOVELTY_RETENTION=0 in the environment: the DEMOTE literal holds.
        assert!(st.novelty_retention().is_none());
        return;
    }
    assert!(st.novelty_retention().is_some());
    st.disarm_novelty_retention();
    assert!(st.novelty_retention().is_none());
    let mut reference = *st.belief();
    for x in &trace(Case::Mixed, 7).observed {
        st.evolve_belief_additive(x);
        incumbent_step(&mut reference, x, &cfg);
    }
    assert_eq!(bits(st.belief()), bits(&reference));
}

#[test]
fn arithmetic_identity_kappa0_mu0_is_incumbent_bit_exact() {
    let cfg = ReconstructionConfig::default();
    let zero = NoveltyRetentionConfig {
        kappa: 0.0,
        mu: 0.0,
        ..NoveltyRetentionConfig::default()
    };
    for case in Case::ALL {
        for seed in SEEDS {
            let tr = trace(case, seed);
            let mut st = ReconstructionState::with_config([0.0; 8], cfg);
            if !st.arm_novelty_retention(zero) {
                return; // demoted by env — the identity is then trivially the G1 arm.
            }
            let mut reference = [0.0f32; 8];
            for x in &tr.observed {
                st.evolve_belief_additive(x);
                incumbent_step(&mut reference, x, &cfg);
                assert_eq!(bits(st.belief()), bits(&reference), "{case:?} seed {seed}");
            }
        }
    }
}

/// Mean gate over a window of ticks for one trace.
fn gates(case: Case, seed: u64, ncfg: NoveltyRetentionConfig) -> Vec<[f32; 6]> {
    let eta = ReconstructionConfig::default().additive_drive_scale;
    let alpha = ReconstructionConfig::default().belief_retention;
    let mut nr = NoveltyRetention::new(ncfg).unwrap();
    let mut out = Vec::new();
    let mut g = [0.0f32; 6];
    for x in &trace(case, seed).observed {
        nr.gate(&drive_of(x, eta), &alpha, &mut g);
        out.push(g);
    }
    out
}

fn mean(g: &[[f32; 6]], lo: usize, hi: usize, m: usize) -> f32 {
    g[lo..hi].iter().map(|r| r[m]).sum::<f32>() / (hi - lo) as f32
}

#[test]
fn polarity_sustained_step_lowers_retention() {
    let ncfg = NoveltyRetentionConfig::default();
    for seed in SEEDS {
        let g = gates(Case::Steps, seed, ncfg);
        for &s in &STEP_TICKS {
            for m in 0..6 {
                let before = mean(&g, s - 20, s, m);
                let after = mean(&g, s + 2, s + 12, m);
                assert!(
                    after < before - 0.05,
                    "seed {seed} step@{s} kind {m}: post-step g {after} not below pre-step {before}"
                );
            }
        }
    }
}

#[test]
fn polarity_transient_burst_raises_retention() {
    let ncfg = NoveltyRetentionConfig::default();
    for seed in SEEDS {
        let g = gates(Case::Spikes, seed, ncfg);
        for &s in &SPIKE_TICKS {
            for m in 0..6 {
                let before = mean(&g, s - 20, s, m);
                // The burst ONSET tick: the innovation is unexplained by any
                // sustained divergence, so the gate must hold the belief.
                let onset = g[s][m];
                assert!(
                    onset > before + 0.02,
                    "seed {seed} burst@{s} kind {m}: onset g {onset} not above steady {before}"
                );
            }
        }
    }
}

#[test]
fn nan_drive_falls_back_to_alpha_base() {
    let mut nr = NoveltyRetention::new(NoveltyRetentionConfig::default()).unwrap();
    let alpha = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
    let mut g = [0.0f32; 6];
    nr.gate(&[0.2; 6], &alpha, &mut g);
    let drive = [f32::NAN, 0.2, f32::INFINITY, 0.2, f32::NEG_INFINITY, 0.2];
    nr.gate(&drive, &alpha, &mut g);
    for m in [0, 2, 4] {
        assert_eq!(g[m], alpha[m], "non-finite drive kind {m} must fall back");
    }
    // The poisoned tick must not have entered the EMAs: a clean steady
    // stream afterwards still reads zero novelty → exactly α_base.
    for _ in 0..5 {
        nr.gate(&[0.2; 6], &alpha, &mut g);
    }
    for m in 0..6 {
        assert!((g[m] - alpha[m]).abs() < 1e-6, "kind {m}: {}", g[m]);
    }
    // Kernel level: a NaN activation is zeroed by the positivity guard and the
    // armed belief stays finite.
    let mut st = ReconstructionState::new([0.0; 8]);
    if st.arm_novelty_retention(NoveltyRetentionConfig::default()) {
        st.evolve_belief_additive(&[f32::NAN, 0.5, 0.5, 0.5, 0.5, 0.5]);
        assert!(st.belief().iter().all(|v| v.is_finite()));
    }
}

#[test]
fn boundary_gate_stays_in_bounds() {
    let extreme = [
        NoveltyRetentionConfig::default(),
        NoveltyRetentionConfig {
            kappa: 1.0,
            mu: 0.0,
            beta: 64.0,
            ..NoveltyRetentionConfig::default()
        },
        NoveltyRetentionConfig {
            kappa: 0.0,
            mu: 5.0,
            beta: 64.0,
            ..NoveltyRetentionConfig::default()
        },
    ];
    for ncfg in extreme {
        for case in Case::ALL {
            for row in gates(case, 1, ncfg) {
                for v in row {
                    assert!(
                        v >= ncfg.g_min && v <= ncfg.g_max,
                        "{case:?} {ncfg:?}: g {v} out of [{}, {}]",
                        ncfg.g_min,
                        ncfg.g_max
                    );
                }
            }
        }
    }
}

#[test]
fn channel_freeze_never_locks_under_sustained_noise() {
    // The worst case for a lock: a large μ pushes every noisy tick to g_max.
    let ncfg = NoveltyRetentionConfig {
        mu: 5.0,
        beta: 64.0,
        ..NoveltyRetentionConfig::default()
    };
    let eta = ReconstructionConfig::default().additive_drive_scale;
    let alpha = ReconstructionConfig::default().belief_retention;
    for seed in SEEDS {
        let mut nr = NoveltyRetention::new(ncfg).unwrap();
        let mut g = [0.0f32; 6];
        let tr = trace(Case::HighNoise, seed);
        let mut saturated = 0usize;
        for (t, x) in tr.observed.iter().enumerate() {
            let d = drive_of(x, eta);
            nr.gate(&d, &alpha, &mut g);
            for m in 0..6 {
                let write = (1.0 - g[m]) * d[m];
                if d[m] != 0.0 {
                    assert!(
                        write != 0.0,
                        "seed {seed} t {t} kind {m}: channel locked (g {})",
                        g[m]
                    );
                }
                assert!(1.0 - g[m] > 0.0);
                saturated += usize::from(g[m] == ncfg.g_max);
            }
        }
        // The arm is only meaningful if the guard was actually exercised.
        assert!(
            saturated > 0,
            "seed {seed}: g_max never reached — arm is vacuous"
        );
    }
    // And the kernel's belief still tracks under the saturated gate: it moves.
    let mut st = ReconstructionState::new([0.0; 8]);
    if st.arm_novelty_retention(ncfg) {
        let tr = trace(Case::HighNoise, 1);
        let mut prev = *st.belief();
        let mut moved = 0usize;
        for x in &tr.observed {
            st.evolve_belief_additive(x);
            moved += usize::from(bits(st.belief()) != bits(&prev));
            prev = *st.belief();
        }
        assert!(
            moved * 10 > tr.observed.len() * 9,
            "belief froze: moved {moved}"
        );
    }
}

/// Issue 1231: the FUSED armed kernel (drive → dual-EMA → g → belief in one
/// per-kind loop) is bit-identical to the unfused composition it replaced —
/// drive pass, `NoveltyRetention::gate` (padded 8-lane), then the belief pass
/// over KIND_MAP — on every fixture case/seed and across gate configs. Also
/// pins the gate's observable state (`last_gate`/sustained/transient).
#[test]
fn fused_kernel_is_bit_identical_to_gate() {
    let cfg = ReconstructionConfig::default();
    let base = NoveltyRetentionConfig::default();
    let configs = [
        base,
        NoveltyRetentionConfig {
            kappa: 0.8,
            mu: 0.4,
            beta: 16.0,
            ..base
        },
        NoveltyRetentionConfig {
            g_min: 0.5,
            g_max: 0.6,
            ..base
        },
    ];
    for ncfg in configs {
        for case in Case::ALL {
            for seed in SEEDS {
                let tr = trace(case, seed);
                let mut st = ReconstructionState::with_config([0.0; 8], cfg);
                if !st.arm_novelty_retention(ncfg) {
                    return; // demoted by env — nothing armed to compare.
                }
                let mut gate = NoveltyRetention::new(ncfg).unwrap();
                let mut reference = [0.0f32; 8];
                let mut g = [0.0f32; 6];
                for x in &tr.observed {
                    st.evolve_belief_additive(x);
                    let drive = drive_of(x, cfg.additive_drive_scale);
                    gate.gate(&drive, &cfg.belief_retention, &mut g);
                    for (b, &m) in reference.iter_mut().zip(TripleEvidence::KIND_MAP.iter()) {
                        *b = (g[m] * *b + (1.0 - g[m]) * drive[m]).clamp(-1.0, 1.0);
                    }
                    assert_eq!(bits(st.belief()), bits(&reference), "{case:?} seed {seed}");
                    let nr = st.novelty_retention().unwrap();
                    let b6 = |a: &[f32; 6]| a.map(f32::to_bits);
                    assert_eq!(b6(nr.last_gate()), b6(gate.last_gate()));
                    assert_eq!(b6(nr.last_sustained()), b6(gate.last_sustained()));
                    assert_eq!(b6(nr.last_transient()), b6(gate.last_transient()));
                }
            }
        }
    }
}
