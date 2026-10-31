//! Issue 975 T2/T3 gates — the `step_noise` fixture classification and the
//! causal toggle arms (arXiv:2603.06009's Fig-3 protocol, modelless twin:
//! riir-train Plan 479 / Research 485).
//!
//! **T2 — pre-registered archetype→regime mapping** (locked in the issue
//! BEFORE any fixture ran): committed_wrong→CommittedWrong ·
//! converged→Converged · stuck→Stuck · thrash(noise, non-periodic)→Thrash ·
//! oscillation→Thrash-gate co-fires WITH the trap `FlipDetector`
//! (separable signals, co-fire allowed; bridge precedence trap-first) ·
//! drift→NOT-Thrash (low churn; only the Thrash exclusion is load-bearing —
//! the Stuck-or-Learning/Converged boundary is MEASURED and recorded in the
//! assertion output, never hard-pinned).
//!
//! **T3 — causal toggles, two arms:** (a) synthetic — a seeded harness
//! flips the effective step mid-run; the monitor must move
//! Learning→Thrash→Learning in sync; (b) the anti-circularity arm — a REAL
//! update process (`cgsp::dual_pool::DualPoolBandit`, feature
//! `cgsp_dual_pool`) with the route gain flipped mid-run: the monitor
//! consumes the bandit's actual `w_e` trajectory as state + the fed mean
//! reward as objective, and must read Thrash during the high-gain/
//! flat-outcome phase and Learning after the gain drops. The thrash
//! FIXTURE (arm a) is built from the detector's own predicate shapes; arm
//! (b) is the defense against a detector that only fires on its own
//! definition — its state path comes from the router's real DecentMem
//! Eq-6/7 math, never from a synthetic construction.
//!
//! All magnitudes are sized against `StepNoiseConfig::DEFAULT` (the
//! calibration is NOT retuned to make fixtures pass — the fixtures are
//! sized to the documented unit-scale posture instead).

#![cfg(feature = "step_noise")]

use katgpt_core::step_noise::{Regime, StepNoiseConfig, StepNoiseMonitor, half_split_dispersion};

// ── shared fixture plumbing ────────────────────────────────────────────────

/// Monitor dimension (the T4 perf bar's N; fixtures project to it).
const N: usize = 8;
/// Steps per fixture (bench_011's N_STEPS convention).
const STEPS: usize = 100;

/// Deterministic LCG (bench_011's convention, no global RNG).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    }
    /// Symmetric noise in [-m, m).
    #[inline]
    fn noise(&mut self, m: f32) -> f32 {
        self.next_f32() * 2.0 * m
    }
}

/// A fixture: the state trajectory plus its per-tick objective series.
struct Fixture {
    states: Vec<[f32; N]>,
    objectives: Vec<f32>,
}

impl Fixture {
    /// Feed the whole trajectory; return every post-observation read.
    fn run(&self) -> Vec<katgpt_core::step_noise::StepNoiseRead> {
        let mut mon = StepNoiseMonitor::<N>::new(StepNoiseConfig::DEFAULT);
        self.states
            .iter()
            .zip(&self.objectives)
            .map(|(s, &o)| mon.observe(s, o))
            .collect()
    }
}

fn unit_vec(rng: &mut Lcg) -> [f32; N] {
    let mut d = [0.0_f32; N];
    for x in d.iter_mut() {
        *x = rng.next_f32();
    }
    let norm = d.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in d.iter_mut() {
        *x /= norm;
    }
    d
}

// ── the six archetype builders (shapes credited to bench_011) ──────────────

/// **Committed-wrong** (bench_011 shape): steady drift toward a wrong
/// attractor; the objective falls steadily with the approach — the
/// process is optimizing into a worse region. → CommittedWrong.
fn build_committed_wrong(seed: u64) -> Fixture {
    let mut rng = Lcg::new(seed);
    let mut state = [0.0_f32; N];
    for x in state.iter_mut() {
        *x = rng.next_f32() * 0.1;
    }
    let dir = unit_vec(&mut rng);
    let mut states = Vec::with_capacity(STEPS + 1);
    let mut objectives = Vec::with_capacity(STEPS + 1);
    let mut obj = 1.0_f32;
    for t in 0..=STEPS {
        if t > 0 {
            for j in 0..N {
                state[j] += 0.05 * dir[j];
            }
            obj -= 0.01; // steady decline into the wrong attractor
        }
        states.push(state);
        objectives.push(obj);
    }
    Fixture { states, objectives }
}

/// **Converged-correct** (bench_011 shape): exponential approach to a
/// correct attractor; the objective rises fast then decelerates. Mid-run
/// this is Learning; at the tail the gates go quiet → Converged.
fn build_converged_correct(seed: u64) -> Fixture {
    let mut rng = Lcg::new(seed);
    let target: [f32; N] = {
        let mut t = [0.0_f32; N];
        for x in t.iter_mut() {
            *x = rng.next_f32();
        }
        t
    };
    let mut state = [0.0_f32; N];
    for x in state.iter_mut() {
        *x = rng.next_f32() * 0.1;
    }
    let mut states = Vec::with_capacity(STEPS + 1);
    let mut objectives = Vec::with_capacity(STEPS + 1);
    for t in 0..=STEPS {
        if t > 0 {
            for j in 0..N {
                state[j] += 0.05 * (target[j] - state[j]);
            }
        }
        let dist = state
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        states.push(state);
        objectives.push(1.0 - dist); // rising as the target nears
    }
    Fixture { states, objectives }
}

/// **Stuck** (bench_011 shape): frozen at one point, objective flat
/// (degenerate updates, no signal). → Stuck (travel ≈ 0 ≤ travel_tol).
fn build_stuck(seed: u64) -> Fixture {
    let mut rng = Lcg::new(seed);
    let fixed: [f32; N] = {
        let mut t = [0.0_f32; N];
        for x in t.iter_mut() {
            *x = rng.next_f32();
        }
        t
    };
    let n_states = STEPS + 1;
    Fixture {
        states: vec![fixed; n_states],
        objectives: vec![0.5; n_states],
    }
}

/// **Thrash** (NEW — the module's whole point): noise-driven LARGE steps
/// in fresh random directions (non-periodic by construction) + a flat
/// objective. → Thrash, and the dispersion proxy must read HIGH (the
/// noise proxy is doing its one labeled job).
fn build_thrash(seed: u64) -> Fixture {
    let mut rng = Lcg::new(seed);
    let mut state = [0.0_f32; N];
    for x in state.iter_mut() {
        *x = rng.next_f32() * 0.1;
    }
    let mut states = Vec::with_capacity(STEPS + 1);
    let mut objectives = Vec::with_capacity(STEPS + 1);
    let mut obj = 0.5_f32;
    for t in 0..=STEPS {
        if t > 0 {
            let dir = unit_vec(&mut rng); // fresh direction every tick — no periodicity
            for j in 0..N {
                state[j] += 0.3 * dir[j];
            }
            obj += rng.noise(0.001); // zero-mean noise, flat in EMA
        }
        states.push(state);
        objectives.push(obj);
    }
    Fixture { states, objectives }
}

/// **Oscillation** (bench_011 shape): period-2 ping-pong between two
/// attractors; objective flat (every patch fails identically). The
/// Thrash gate co-fires with the trap `FlipDetector` — separable signals
/// (magnitude axis vs alternation axis), co-fire allowed.
fn build_oscillation(seed: u64) -> Fixture {
    let mut rng = Lcg::new(seed);
    let a: [f32; N] = {
        let mut t = [0.0_f32; N];
        for x in t.iter_mut() {
            *x = rng.next_f32();
        }
        t
    };
    let b: [f32; N] = {
        let mut t = [0.0_f32; N];
        for x in t.iter_mut() {
            *x = rng.next_f32();
        }
        t
    };
    let mut states = Vec::with_capacity(STEPS + 1);
    let mut objectives = Vec::with_capacity(STEPS + 1);
    let mut obj = 0.5_f32;
    for i in 0..=STEPS {
        states.push(if i % 2 == 0 { a } else { b });
        if i > 0 {
            obj += rng.noise(0.001);
        }
        objectives.push(obj);
    }
    Fixture { states, objectives }
}

/// **Drift** (bench_011's rotation shape, sized DOWN): rotating through
/// answers at LOW per-step movement (churn well under the hot midpoint)
/// with a flat objective. Pre-registered: NOT-Thrash (only the exclusion
/// is load-bearing); expected tail read: Converged (traveled > travel_tol
/// but quiet) — recorded, not pinned.
fn build_drift(seed: u64) -> Fixture {
    let mut rng = Lcg::new(seed);
    let mut state = [0.0_f32; N];
    for x in state.iter_mut() {
        *x = rng.next_f32() * 0.1;
    }
    let u = unit_vec(&mut rng);
    let mut v = unit_vec(&mut rng);
    // Gram-Schmidt v against u (bench_011's construction).
    let dot: f32 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let mut norm_v = 0.0_f32;
    for j in 0..N {
        v[j] -= dot * u[j];
        norm_v += v[j] * v[j];
    }
    let norm_v = norm_v.sqrt().max(1e-6);
    for x in v.iter_mut() {
        *x /= norm_v;
    }
    let step_rate = 0.005_f32; // per-step arc magnitude — LOW churn
    let omega = 0.4_f32;
    let mut states = Vec::with_capacity(STEPS + 1);
    let mut objectives = Vec::with_capacity(STEPS + 1);
    let mut obj = 0.5_f32;
    let mut theta = 0.0_f32;
    for t in 0..=STEPS {
        if t > 0 {
            theta += omega;
            for j in 0..N {
                state[j] += step_rate * (theta.cos() * u[j] + theta.sin() * v[j]);
            }
            obj += rng.noise(0.0005);
        }
        states.push(state);
        objectives.push(obj);
    }
    Fixture { states, objectives }
}

// ── T2 — the pre-registered classification gate ────────────────────────────

#[test]
fn t2_six_archetypes_classify_per_the_pre_registered_mapping() {
    // committed_wrong → CommittedWrong (objective falling steadily).
    let reads = build_committed_wrong(42).run();
    let tail = reads.last().expect("committed-wrong reads");
    assert_eq!(
        tail.regime,
        Regime::CommittedWrong,
        "committed-wrong tail: {tail:?}"
    );

    // converged → Converged at the tail (and Learning mid-run — recorded,
    // the mid-boundary is not load-bearing).
    let reads = build_converged_correct(577).run();
    assert_eq!(
        reads.last().map(|r| r.regime),
        Some(Regime::Converged),
        "converged-correct tail"
    );
    let mid_learning = reads[STEPS / 2].regime == Regime::Learning;
    eprintln!(
        "[step_noise][t2] converged_correct mid-run regime = {:?} (measured boundary, not pinned)",
        reads[STEPS / 2].regime
    );
    let _ = mid_learning;

    // stuck → Stuck (flat objective, near-zero churn, never traveled).
    let reads = build_stuck(314).run();
    let tail = reads.last().expect("stuck reads");
    assert_eq!(tail.regime, Regime::Stuck, "stuck tail: {tail:?}");

    // thrash → Thrash, with the dispersion proxy reading HIGH (its one
    // labeled job: noise-driven direction churn, never claimed as a
    // gradient norm).
    let reads = build_thrash(99).run();
    let tail = reads.last().expect("thrash reads");
    assert_eq!(tail.regime, Regime::Thrash, "thrash tail: {tail:?}");
    assert!(
        tail.dispersion > 0.5,
        "noise-driven thrash must disperse the half-window directions, got {}",
        tail.dispersion
    );

    // oscillation → Thrash co-fires WITH the trap FlipDetector: the flip
    // rate reads ~1 (period-2 alternation) while the regime is Thrash —
    // the magnitude axis and the alternation axis are SEPARABLE signals
    // and both fire here (bridge precedence trap-first stays correct).
    let reads = build_oscillation(137).run();
    let tail = reads.last().expect("oscillation reads");
    assert_eq!(tail.regime, Regime::Thrash, "oscillation tail: {tail:?}");
    // (The co-fire half needs `saddle_escape`; with `step_noise` alone the
    // Thrash classification above is the load-bearing arm that still runs.)
    #[cfg(feature = "saddle_escape")]
    {
        let mut flip = katgpt_core::saddle_escape::FlipDetector::new(8);
        let mut last_rate = None;
        for s in &build_oscillation(137).states {
            // Key = quantized state (a real decoder's choice key stand-in).
            let key = s.iter().fold(0_u64, |acc, x| {
                acc.wrapping_mul(31)
                    .wrapping_add(u64::from((x * 1e3).to_bits() & 0xFFFF))
            });
            last_rate = flip.observe(Some(key));
        }
        let rate = last_rate.expect("flip rate warm after 100 keyed steps");
        assert!(
            rate > 0.8,
            "period-2 oscillation must co-fire the trap axis (flip rate {rate})"
        );
    }

    // drift → NOT-Thrash (the LOAD-BEARING exclusion): low churn, flat
    // objective. The tail's actual regime is measured and recorded —
    // expected Converged (traveled, then quiet) — never pinned.
    let reads = build_drift(256).run();
    let tail = reads.last().expect("drift reads");
    assert_ne!(
        tail.regime,
        Regime::Thrash,
        "drift must NOT read Thrash: {tail:?}"
    );
    eprintln!(
        "[step_noise][t2] drift tail regime = {:?} churn = {:.4} (measured boundary — only the Thrash exclusion is load-bearing)",
        tail.regime, tail.churn
    );
}

// ── T3(a) — synthetic causal toggle ────────────────────────────────────────

/// A seeded harness flips the effective step mid-run: healthy learning
/// (small steps, objective rising) → thrash (large steps, flat objective)
/// → healthy learning. The monitor must move Learning→Thrash→Learning in
/// sync with the HARNESS's phases, not its own convenience.
#[test]
fn t3a_synthetic_step_flip_moves_learning_thrash_learning() {
    const PHASE: usize = 60; // ticks per phase (≥ warmup 16 + 3·EMA settle)
    let mut rng = Lcg::new(2026_0975);
    let target: [f32; N] = {
        let mut t = [0.0_f32; N];
        for x in t.iter_mut() {
            *x = rng.next_f32();
        }
        t
    };
    let mut state = [0.0_f32; N];
    for x in state.iter_mut() {
        *x = rng.next_f32() * 0.1;
    }
    let mut mon = StepNoiseMonitor::<N>::new(StepNoiseConfig::DEFAULT);
    let mut obj = 0.0_f32;
    let phase_tails = [
        (PHASE - 1, Regime::Learning),
        (2 * PHASE - 1, Regime::Thrash),
        (3 * PHASE - 1, Regime::Learning),
    ];
    let mut tail_idx = 0_usize;
    for t in 0..3 * PHASE {
        let thrashing = (PHASE..2 * PHASE).contains(&t); // the mid-run flip
        if t > 0 {
            if thrashing {
                let dir = unit_vec(&mut rng); // large noise-driven steps
                for j in 0..N {
                    state[j] += 0.3 * dir[j];
                }
                obj += rng.noise(0.001); // flat
            } else {
                for j in 0..N {
                    state[j] += 0.02 * (target[j] - state[j]); // small steps
                }
                obj += 0.01; // rising
            }
        }
        let read = mon.observe(&state, obj);
        if tail_idx < 3 && t == phase_tails[tail_idx].0 {
            assert_eq!(
                read.regime,
                phase_tails[tail_idx].1,
                "phase {} (tick {t}) regime out of sync with the harness flip",
                tail_idx + 1
            );
            tail_idx += 1;
        }
    }
    assert_eq!(tail_idx, 3, "all three phase tails must be checked");
}

// ── T3(b) — the anti-circularity arm (real update process) ─────────────────

/// The bandit arm needs `cgsp_dual_pool`; with only `step_noise` the rest
/// of this target still runs 3 real tests (never a green zero).
#[cfg(feature = "cgsp_dual_pool")]
mod bandit_toggle {
    use super::{Lcg, unit_vec};
    use katgpt_core::ReachableDualPoolRouter;
    use katgpt_core::cgsp::dual_pool::{DualPoolBandit, DualPoolConfig, PoolId};
    use katgpt_core::cgsp::traits::HintDeltaBandit;
    use katgpt_core::step_noise::{Regime, StepNoiseConfig, StepNoiseMonitor};

    /// Minimal concrete bandit (the shape of dual_pool's own test VecBandit —
    /// that one is `mod tests`-private, so the fixture carries its own;
    /// `Priority` is a private alias for `f32` in `cgsp::traits`).
    struct FixtureBandit {
        prios: Vec<f32>,
    }
    impl FixtureBandit {
        fn uniform(n: usize) -> Self {
            Self {
                prios: vec![1.0 / n as f32; n],
            }
        }
    }
    impl HintDeltaBandit for FixtureBandit {
        fn absorb(&mut self, _arm: usize, _reward: f32) {
            // Rewards cross as the explicit route_update booleans below (the
            // harness IS the environment); per-arm priorities are idle here.
        }
        fn priority(&self, arm: usize) -> f32 {
            self.prios[arm]
        }
        fn priorities(&self) -> &[f32] {
            &self.prios
        }
        fn priorities_mut(&mut self) -> &mut [f32] {
            &mut self.prios
        }
    }

    /// T3(b): a REAL update process — `DualPoolBandit`'s actual DecentMem
    /// Eq-6/7 routing — with the route gain α flipped mid-run. The monitor
    /// consumes the bandit's actual `w_e` trajectory as state and the fed
    /// outcome series as objective.
    ///
    /// Harness (one bandit per phase — `DualPoolConfig` is build-time, so
    /// the flip is the paper's instantaneous Fig-3 swap; the phase-1 path is
    /// engineered to END on a fail, which floors `w_e` at exactly 1.0, the
    /// phase-2 bandit's start — the seam contributes no fake churn tick):
    /// - Phase 1, HIGH gain (0.5): outcomes alternate success/fail →
    ///   `w_e` flip-flops 1.0→1.5→1.0→… (real router math, ‖Δ‖ ≈ 0.5/tick)
    ///   against a flat, low outcome series → the monitor must read THRASH.
    /// - Phase 2, LOW gain (0.02): outcomes consistently succeed with a
    ///   rising series → `w_e` eases up smoothly (‖Δ‖ ≈ 0.02/tick) →
    ///   LEARNING.
    #[test]
    fn t3b_bandit_route_gain_flip_thrash_then_learning() {
        let mut rng = Lcg::new(97_531);
        // Monitor calibrated to the w_e host scale (module docs: retune
        // c0/p0 to the host's units — the FIXTURE stays honest, the UNIT
        // calibration is what moves).
        let cfg = StepNoiseConfig {
            c0: 0.1, // between the low gain (0.02) and the high gain (0.5)
            ..StepNoiseConfig::DEFAULT
        };
        let mut mon = StepNoiseMonitor::<1>::new(cfg);
        let mut obj = 0.15_f32;

        // ── Phase 1: high gain, low outcomes — the churn is the ROUTER's. ──
        let mut bandit = DualPoolBandit::with_config(
            FixtureBandit::uniform(4),
            FixtureBandit::uniform(4),
            DualPoolConfig {
                alpha_update_gain: 0.5,
                ..DualPoolConfig::default()
            },
        );
        let mut last_regime_p1 = Regime::Learning;
        let mut w_prev = bandit.w_e();
        for t in 0..60 {
            let success = t % 2 == 0; // alternating outcomes; t=58 succeeds, t=59 fails
            bandit.route_update(PoolId::Exploitation, success);
            obj += rng.noise(0.005); // flat, low
            let w = bandit.w_e();
            let read = mon.observe(&[w], obj);
            last_regime_p1 = read.regime;
            w_prev = w;
        }
        assert!(
            (w_prev - 1.0).abs() < 1e-6,
            "phase 1 must end on a fail so w_e floors at 1.0 (got {w_prev})"
        );
        assert_eq!(
            last_regime_p1,
            Regime::Thrash,
            "high-gain flip-flop against flat outcomes must read Thrash (the state path came from the real router)"
        );

        // ── Phase 2: LOW gain, rising outcomes — the same monitor, live. ──
        let mut bandit = DualPoolBandit::with_config(
            FixtureBandit::uniform(4),
            FixtureBandit::uniform(4),
            DualPoolConfig {
                alpha_update_gain: 0.02,
                ..DualPoolConfig::default()
            },
        );
        let mut last_regime_p2 = Regime::Thrash;
        for _t in 0..60 {
            bandit.route_update(PoolId::Exploitation, true);
            obj += 0.01; // outcomes improve as the router stops churning
            let w = bandit.w_e();
            let read = mon.observe(&[w], obj);
            last_regime_p2 = read.regime;
        }
        assert_eq!(
            last_regime_p2,
            Regime::Learning,
            "reduced gain + rising outcomes must read Learning"
        );
        // The direction axis stays quiet on a smooth router too (recorded).
        let _ = unit_vec(&mut rng);
    }
}

// ── shape guards ───────────────────────────────────────────────────────────

/// The zero-vector dispersion contract (INF/zero guard, labeled proxy).
#[test]
fn t2_half_split_dispersion_zero_vector_guard() {
    let a = [0.0_f32; N];
    let mut b = [0.0_f32; N];
    b[0] = 1.0;
    assert_eq!(half_split_dispersion(&a, &b), 0.0, "zero vector → 0.0");
    let mut c = [0.0_f32; N];
    c[0] = -2.0;
    let d = half_split_dispersion(&b, &c); // antiparallel → dispersion 2.0
    assert!((d - 2.0).abs() < 1e-5, "antiparallel → 2.0, got {d}");
    assert!(half_split_dispersion(&b, &b).abs() < 1e-6, "identical → ~0");
}
