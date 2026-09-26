#![cfg(feature = "arm_drift_alignment")]
//! Plan 610 GOAT gate — `arm_drift_alignment` (per-arm trajectory-aligned
//! curiosity, Research 591). Bench 900.
//!
//! The fixture and every bar below were pre-registered in
//! `.plans/610_arm_drift_alignment.md` (commit `9451fd5ba`) before any of
//! these gates ran.
//!
//! - **G1** planted drift: the aligned score ranks the drifting family F above
//!   the orthogonal family S, including F's four held-out arms, whose OWN
//!   priorities never move except through renormalization. Also a pure-noise
//!   negative control and the preconditioner's scale-invariance contract.
//! - **G2** discrimination: the global-norm incumbent is tied by construction,
//!   and the own-arm drift magnitude baseline must fail the held-out ranking.
//! - **G3** loop A/B over 32 paired seeds: aligned vs global-norm vs a
//!   matched-uniform bonus vs extrinsic-only. Readout only. The promotion
//!   verdict is recorded in Bench 900; the test asserts the pins.
//! - **G4** zero steady-state allocations, plus interleaved median-of-ratios
//!   cost against the incumbent (`--release`).
//!
//! Run: `cargo test --release --features arm_drift_alignment --test
//! plan_610_arm_drift_alignment_goat -- --nocapture --test-threads=1`

#[path = "common/ab_timing.rs"]
mod ab_timing;
#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
#[path = "common/alloc_tracking.rs"]
mod alloc_tracking;

use katgpt_core::cgsp::arm_alignment::DEFAULT_SCALE_ALPHA;
use katgpt_core::cgsp::loop_::renormalize_priorities;
use katgpt_core::cgsp::traits::{CuriosityConjecturer, HintDeltaBandit};
use katgpt_core::cgsp::types::{Candidate, Direction, Priority, Target};
use katgpt_core::cgsp::{
    DerivativeCuriosity, DriftPreconditioner, DriftSummary, FirstMomentDrift, PoolConjecturer,
    SecondMomentDrift, TrajectoryAlignedCuriosity, alignment_score,
};
use katgpt_core::simd::simd_fused_decay_write;
use katgpt_core::temporal_deriv::TemporalDerivativeKernel;

const DIM: usize = 16;
const N_ARMS: usize = 16;
const N_F: usize = 8;
const N_DRIVERS: usize = 4;
const STEPS: usize = 200;
const READ_FROM: usize = 100;
const SEEDS: u64 = 16;
const BETA: f32 = 4.0;

// ── Deterministic RNG (seeded; no global draws) ─────────────────────────

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0xD1B5_4A32_D192_ED03)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Stateless uniform keyed on (seed, cycle, slot): common random numbers.
fn keyed_uniform(seed: u64, cycle: usize, slot: usize) -> f32 {
    let mut r = Rng::new(seed.wrapping_mul(0x100_0000_01B3) ^ ((cycle as u64) << 8) ^ slot as u64);
    r.next_u64();
    r.uniform()
}

// ── Fixture ──────────────────────────────────────────────────────────────

fn unit(mut v: Vec<f32>) -> Direction {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    Direction { coords: v }
}

/// F = 8 arms at `e0 + 0.2·N(0,I)`; S = `±e1..±e4` (centered) or `+e1..+e8`.
fn build_pool(seed: u64, centered_s: bool) -> Vec<Direction> {
    let mut rng = Rng::new(seed.wrapping_add(7));
    let mut pool = Vec::with_capacity(N_ARMS);
    for _ in 0..N_F {
        let mut v: Vec<f32> = (0..DIM).map(|_| 0.2 * rng.gauss()).collect();
        v[0] += 1.0;
        pool.push(unit(v));
    }
    for i in 0..N_ARMS - N_F {
        let mut v = vec![0.0f32; DIM];
        match centered_s {
            true => v[1 + i / 2] = if i % 2 == 0 { 1.0 } else { -1.0 },
            false => v[1 + i] = 1.0,
        }
        pool.push(Direction { coords: v });
    }
    pool
}

fn normalize(w: &[f32; N_ARMS]) -> [f32; N_ARMS] {
    let z: f32 = w.iter().sum();
    w.map(|x| x / z)
}

/// Drivers `F[0..4]` ramp +0.02/step; additive N(0, 0.02) on every weight.
fn planted_trajectory(seed: u64) -> Vec<[f32; N_ARMS]> {
    let mut rng = Rng::new(seed.wrapping_add(1000));
    (0..STEPS)
        .map(|t| {
            let mut w = [1.0f32; N_ARMS];
            for (j, x) in w.iter_mut().enumerate() {
                if j < N_DRIVERS {
                    *x += 0.02 * t as f32;
                }
                *x = (*x + 0.02 * rng.gauss()).max(0.01);
            }
            normalize(&w)
        })
        .collect()
}

/// No ramp; weights `1 + N(0, 0.1)` i.i.d. per step.
fn noise_trajectory(seed: u64) -> Vec<[f32; N_ARMS]> {
    let mut rng = Rng::new(seed.wrapping_add(2000));
    (0..STEPS)
        .map(|_| {
            let mut w = [1.0f32; N_ARMS];
            w.iter_mut()
                .for_each(|x| *x = (*x + 0.1 * rng.gauss()).max(0.01));
            normalize(&w)
        })
        .collect()
}

// ── Scorers ──────────────────────────────────────────────────────────────

/// Mean per-arm aligned score over steps `READ_FROM..` via the shipped type.
fn aligned_scores(pool: &[Direction], traj: &[[f32; N_ARMS]], kappa: f32) -> [f32; N_ARMS] {
    let summary = FirstMomentDrift::<DIM>::new()
        .with_beta(BETA)
        .with_preconditioner(DEFAULT_SCALE_ALPHA, kappa);
    let mut tac = TrajectoryAlignedCuriosity::with_summary(pool.to_vec(), 0, summary);
    let mut acc = [0.0f32; N_ARMS];
    for (t, p) in traj.iter().enumerate() {
        tac.observe_drift(p);
        if t >= READ_FROM {
            for (a, g) in acc.iter_mut().zip(pool) {
                *a += tac.summary().score_direction(g);
            }
        }
    }
    acc.map(|a| a / (STEPS - READ_FROM) as f32)
}

/// Mean per-arm score of the Issue 899 second-moment summary.
fn second_scores(pool: &[Direction], traj: &[[f32; N_ARMS]]) -> [f32; N_ARMS] {
    let mut sm: SecondMomentDrift<N_ARMS> = SecondMomentDrift::for_pool(pool);
    let cands: Vec<Candidate> = pool
        .iter()
        .enumerate()
        .map(|(k, g)| Candidate::new(g.clone(), k))
        .collect();
    let mut acc = [0.0f32; N_ARMS];
    for (t, p) in traj.iter().enumerate() {
        sm.observe(p, pool);
        if t >= READ_FROM {
            for (a, c) in acc.iter_mut().zip(&cands) {
                *a += sm.score(c, pool);
            }
        }
    }
    acc.map(|a| a / (STEPS - READ_FROM) as f32)
}

/// The latent drift sequence `d_t` (mean pull through the 10:1 kernel).
fn drift_sequence(pool: &[Direction], traj: &[[f32; N_ARMS]]) -> Vec<[f32; DIM]> {
    let mut kernel: TemporalDerivativeKernel<DIM> = TemporalDerivativeKernel::default();
    traj.iter()
        .enumerate()
        .map(|(t, p)| {
            // Same SIMD axpy as the shipped summary, so the replay pin is
            // bit-exact rather than tolerance-bound.
            let mut m = [0.0f32; DIM];
            for (&w, g) in p.iter().zip(pool) {
                if w != 0.0 {
                    simd_fused_decay_write(&mut m, 1.0, &g.coords, w);
                }
            }
            // Warm start, as the shipped summary does (Issue 899).
            if t == 0 {
                kernel.fast = m;
                kernel.slow = m;
            }
            kernel.observe(&m)
        })
        .collect()
}

/// Mean per-arm score from an explicit drift sequence and a `û` rule.
fn scores_from_drift(
    pool: &[Direction],
    drift: &[[f32; DIM]],
    mut u_rule: impl FnMut(&[f32; DIM], &mut [f32; DIM]),
) -> [f32; N_ARMS] {
    let mut acc = [0.0f32; N_ARMS];
    let mut u = [0.0f32; DIM];
    for (t, d) in drift.iter().enumerate() {
        u_rule(d, &mut u);
        if t >= READ_FROM {
            for (a, g) in acc.iter_mut().zip(pool) {
                *a += alignment_score(&g.coords, &u, BETA);
            }
        }
    }
    acc.map(|a| a / (STEPS - READ_FROM) as f32)
}

/// Preconditioner off: `û = d / ‖d‖`.
fn unconditioned(d: &[f32; DIM], u: &mut [f32; DIM]) {
    let n = d.iter().map(|x| x * x).sum::<f32>().sqrt();
    for (uj, dj) in u.iter_mut().zip(d) {
        *uj = if n > 1e-12 { dj / n } else { 0.0 };
    }
}

/// Own-arm baseline: `|fast_j − slow_j|` of the priority vector itself.
fn own_arm_scores(traj: &[[f32; N_ARMS]]) -> [f32; N_ARMS] {
    let mut kernel: TemporalDerivativeKernel<N_ARMS> = TemporalDerivativeKernel::default();
    let mut acc = [0.0f32; N_ARMS];
    for (t, p) in traj.iter().enumerate() {
        let d = kernel.observe(p);
        if t >= READ_FROM {
            for (a, x) in acc.iter_mut().zip(d) {
                *a += x.abs();
            }
        }
    }
    acc
}

/// Incumbent: one global `sigmoid(β·‖d‖)` per step, broadcast to every arm.
fn incumbent_scores(pool: &[Direction], traj: &[[f32; N_ARMS]]) -> [f32; N_ARMS] {
    let mut dc: DerivativeCuriosity<N_ARMS> = DerivativeCuriosity::new(pool.to_vec(), 0);
    let mut acc = 0.0f32;
    for (t, p) in traj.iter().enumerate() {
        let s = dc.observe_interestingness(p);
        if t >= READ_FROM {
            acc += s;
        }
    }
    [acc; N_ARMS]
}

// ── Statistics ───────────────────────────────────────────────────────────

/// Rank-AUC of `pos` over `neg`; ties count 0.5.
fn auc(pos: &[f32], neg: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for &p in pos {
        for &n in neg {
            s += match p.partial_cmp(&n) {
                Some(std::cmp::Ordering::Greater) => 1.0,
                Some(std::cmp::Ordering::Equal) => 0.5,
                _ => 0.0,
            };
        }
    }
    s / (pos.len() * neg.len()) as f64
}

fn auc_f_vs_s(s: &[f32; N_ARMS]) -> f64 {
    auc(&s[..N_F], &s[N_F..])
}

fn auc_heldout(s: &[f32; N_ARMS]) -> f64 {
    auc(&s[N_DRIVERS..N_F], &s[N_F..])
}

/// Kendall τ over pairs untied in `a` (a pair tied in `b` counts discordant).
fn kendall_tau_untied(a: &[f32], b: &[f32]) -> f64 {
    let (mut conc, mut disc) = (0i64, 0i64);
    for i in 0..a.len() {
        for j in i + 1..a.len() {
            let da = a[i] - a[j];
            if da.abs() <= 1e-6 {
                continue;
            }
            match da * (b[i] - b[j]) > 0.0 {
                true => conc += 1,
                false => disc += 1,
            }
        }
    }
    (conc - disc) as f64 / (conc + disc).max(1) as f64
}

fn mean_sd(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let v = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, v.sqrt())
}

fn fmin(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::INFINITY, f64::min)
}

// ── G1 + G2 ──────────────────────────────────────────────────────────────

#[test]
fn g1_planted_drift_ranks_family_and_heldout_arms() {
    let (mut all, mut held, mut own_held, mut inc, mut k0_held) =
        (vec![], vec![], vec![], vec![], vec![]);
    let (mut off_all, mut off_held) = (vec![], vec![]);
    for seed in 0..SEEDS {
        let pool = build_pool(seed, true);
        let traj = planted_trajectory(seed);
        let s = aligned_scores(&pool, &traj, 0.1);
        all.push(auc_f_vs_s(&s));
        held.push(auc_heldout(&s));
        own_held.push(auc_heldout(&own_arm_scores(&traj)));
        inc.push(auc_f_vs_s(&incumbent_scores(&pool, &traj)));
        k0_held.push(auc_heldout(&aligned_scores(&pool, &traj, 0.0)));
        let off = scores_from_drift(&pool, &drift_sequence(&pool, &traj), unconditioned);
        off_all.push(auc_f_vs_s(&off));
        off_held.push(auc_heldout(&off));
    }
    let (m_all, m_held) = (mean_sd(&all).0, mean_sd(&held).0);
    let (m_own, m_inc, m_k0) = (mean_sd(&own_held).0, mean_sd(&inc).0, mean_sd(&k0_held).0);
    println!("\n═══ Plan 610 G1/G2 — planted drift ({SEEDS} seeds) ═══");
    println!(
        "  aligned κ=0.1   AUC(F vs S)      mean {m_all:.3}  min {:.3}  (bar ≥ 0.8)",
        fmin(&all)
    );
    println!(
        "  aligned κ=0.1   AUC(held-out F)  mean {m_held:.3}  min {:.3}  (bar ≥ 0.8)",
        fmin(&held)
    );
    println!(
        "  own-arm |d_j|   AUC(held-out F)  mean {m_own:.3}  min {:.3}  (G2: must be < 0.8)",
        fmin(&own_held)
    );
    println!("  incumbent ‖d‖   AUC(F vs S)      mean {m_inc:.3}  (tied by construction)");
    println!("  aligned κ=0     AUC(held-out F)  mean {m_k0:.3}  (characterization)");
    println!(
        "  precond OFF     AUC(F vs S) {:.3}  held-out {:.3}  (post-hoc characterization)",
        mean_sd(&off_all).0,
        mean_sd(&off_held).0
    );
    // Measured-verdict pin (Bench 900): G1 FAILED its pre-registered bars.
    // The per-coordinate preconditioner inflates the drivers' within-decade
    // noise coordinates to parity with the drift axis. This reds when the
    // verdict flips, so the record cannot go stale silently.
    let g1_pass = m_all >= 0.8 && m_held >= 0.8;
    println!(
        "  G1 pre-registered verdict: {}",
        if g1_pass { "PASS" } else { "FAIL" }
    );
    assert!(
        !g1_pass,
        "G1 verdict changed to PASS — update Bench 900 + Plan 610"
    );
    // Preconditioner off wins G1 only because it also scores 1.0 under pure
    // noise: a cluster-density prior, not a drift detector.
    assert!(mean_sd(&off_held).0 > 0.95, "P-off characterization moved");
    assert!(
        m_own < 0.8,
        "G2 FAIL: own-arm baseline passes held-out ({m_own:.3})"
    );
    assert!(
        (m_inc - 0.5).abs() < 1e-9,
        "incumbent must be tied: {m_inc}"
    );
}

#[test]
fn g1_negative_control_pure_noise_is_flat() {
    let (mut on, mut off) = (vec![], vec![]);
    for seed in 0..SEEDS {
        let pool = build_pool(seed, true);
        let traj = noise_trajectory(seed);
        on.push(auc_f_vs_s(&aligned_scores(&pool, &traj, 0.1)));
        let drift = drift_sequence(&pool, &traj);
        off.push(auc_f_vs_s(&scores_from_drift(&pool, &drift, unconditioned)));
    }
    let (m_on, sd_on) = mean_sd(&on);
    let (m_off, sd_off) = mean_sd(&off);
    println!("\n═══ Plan 610 G1 negative control — pure noise ({SEEDS} seeds) ═══");
    println!(
        "  preconditioned κ=0.1  AUC(F vs S) mean {m_on:.3} sd {sd_on:.3}  (bar |m−0.5| ≤ 0.15)"
    );
    println!(
        "  preconditioner OFF    AUC(F vs S) mean {m_off:.3} sd {sd_off:.3}  (characterization)"
    );
    // Measured-verdict pins (Bench 900 addendum). With the zero-init
    // transient removed (Issue 899 warm start), the preconditioned form FAILS
    // the negative control: axis-aligned scaling inflates the coordinates only
    // F's directions touch. Preconditioner off is flat. The first Bench 900
    // run read these the other way round (0.395 / 1.000), transient-driven.
    let on_pass = (m_on - 0.5).abs() <= 0.15;
    println!(
        "  preconditioned verdict: {}",
        if on_pass { "PASS" } else { "FAIL" }
    );
    assert!(
        !on_pass,
        "negative-control verdict changed — update Bench 900"
    );
    assert!(
        (m_off - 0.5).abs() <= 0.15,
        "P-off noise AUC moved: {m_off:.3}"
    );
}

#[test]
fn g1_scale_invariance_contract() {
    let (mut worst_k0, mut worst_dauc, mut worst_tau) = (0.0f32, 0.0f64, 1.0f64);
    for seed in 0..SEEDS {
        let pool = build_pool(seed, true);
        let drift = drift_sequence(&pool, &planted_trajectory(seed));
        let mut rng = Rng::new(seed.wrapping_add(3000));
        let s: [f32; DIM] = std::array::from_fn(|_| 0.5 + 1.5 * rng.uniform());
        let scaled: Vec<[f32; DIM]> = drift
            .iter()
            .map(|d| std::array::from_fn(|j| d[j] * s[j]))
            .collect();

        // κ = 0: û must be identical at every step.
        let mut a: DriftPreconditioner<DIM> = DriftPreconditioner::new(DEFAULT_SCALE_ALPHA, 0.0);
        let mut b: DriftPreconditioner<DIM> = DriftPreconditioner::new(DEFAULT_SCALE_ALPHA, 0.0);
        let (mut ua, mut ub) = ([0.0f32; DIM], [0.0f32; DIM]);
        for (d, d2) in drift.iter().zip(&scaled) {
            a.precondition(d, &mut ua);
            b.precondition(d2, &mut ub);
            for j in 0..DIM {
                worst_k0 = worst_k0.max((ua[j] - ub[j]).abs());
            }
        }

        // κ = 0.1: rank order and AUC preserved.
        let mut pa: DriftPreconditioner<DIM> = DriftPreconditioner::new(DEFAULT_SCALE_ALPHA, 0.1);
        let mut pb: DriftPreconditioner<DIM> = DriftPreconditioner::new(DEFAULT_SCALE_ALPHA, 0.1);
        let sa = scores_from_drift(&pool, &drift, |d, u| {
            pa.precondition(d, u);
        });
        let sb = scores_from_drift(&pool, &scaled, |d, u| {
            pb.precondition(d, u);
        });
        worst_dauc = worst_dauc.max((auc_f_vs_s(&sa) - auc_f_vs_s(&sb)).abs());
        worst_tau = worst_tau.min(kendall_tau_untied(&sa, &sb));
    }
    println!("\n═══ Plan 610 G1 scale invariance ({SEEDS} rescalings, s_j ~ U[0.5,2]) ═══");
    println!("  κ=0    max|û − û'|        {worst_k0:.2e}  (bar ≤ 1e-4)");
    println!("  κ=0.1  max |ΔAUC|         {worst_dauc:.4}  (bar ≤ 0.02)");
    println!("  κ=0.1  min Kendall τ      {worst_tau:.3}  (bar ≥ 0.9)");
    // Pin (Bench 900 addendum): exact at κ = 0 only while every coordinate's
    // drift is far above the absolute epsilon. With the warm start, untouched
    // coordinates sit at float noise, where `SCALE_EPS` breaks homogeneity.
    let k0_pass = worst_k0 <= 1e-4;
    println!(
        "  κ=0 pre-registered verdict: {}",
        if k0_pass { "PASS" } else { "FAIL" }
    );
    assert!(
        !k0_pass,
        "κ=0 invariance verdict changed — update Bench 900"
    );
    // Measured-verdict pin (Bench 900): the relative floor breaks the
    // per-coordinate contract at κ = 0.1 (a rescaling moves coordinates
    // across the floor). The module docs scope the contract to κ = 0.
    let k01_pass = worst_dauc <= 0.02 && worst_tau >= 0.9;
    println!(
        "  κ=0.1 pre-registered verdict: {}",
        if k01_pass { "PASS" } else { "FAIL" }
    );
    assert!(
        !k01_pass,
        "κ=0.1 invariance verdict changed — update Bench 900"
    );
}

#[test]
fn g2_characterization_uncentered_stationary_family() {
    // Not a gate: S = +e1..+e8 has a non-zero centroid, so a drift toward F
    // is genuinely a drift AWAY from S, and `|·|` credits S by design.
    let mut v = vec![];
    for seed in 0..SEEDS {
        let pool = build_pool(seed, false);
        v.push(auc_f_vs_s(&aligned_scores(
            &pool,
            &planted_trajectory(seed),
            0.1,
        )));
    }
    let (m, sd) = mean_sd(&v);
    println!("\n═══ Plan 610 characterization — S with non-zero centroid ═══");
    println!("  aligned κ=0.1  AUC(F vs S) mean {m:.3} sd {sd:.3}  (recorded, no bar)");
    assert!(m.is_finite());
}

#[test]
fn replay_matches_shipped_type() {
    // The replay helpers (drift_sequence + DriftPreconditioner) must be the
    // shipped computation, not a second implementation drifting from it.
    let pool = build_pool(3, true);
    let traj = planted_trajectory(3);
    let drift = drift_sequence(&pool, &traj);
    let mut tac: TrajectoryAlignedCuriosity<FirstMomentDrift<DIM>> =
        TrajectoryAlignedCuriosity::new(pool.clone(), 0);
    let mut p: DriftPreconditioner<DIM> = DriftPreconditioner::default();
    let mut u = [0.0f32; DIM];
    for (q, d) in traj.iter().zip(&drift) {
        tac.observe_drift(q);
        p.precondition(d, &mut u);
        for (j, (a, b)) in tac.summary().drift_direction().iter().zip(&u).enumerate() {
            assert!((a - b).abs() < 1e-5, "replay diverged at coord {j}");
        }
    }
}

// ── G3 — loop A/B ────────────────────────────────────────────────────────

const G3_SEEDS: u64 = 32;
const G3_CAP: usize = 600;
const G3_EXT_WINDOW: usize = 300;
const ETA: f32 = 0.05;
const LAMBDA: f32 = 0.5;
const ACQUIRE_MASS: f32 = 0.75;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Arm {
    Aligned,
    GlobalNorm,
    MatchedUniform,
    ExtrinsicOnly,
    /// Issue 899: second-moment per-arm bonus.
    Second,
    /// Issue 899: second-moment conjecturer, every sample gets its mean score.
    SecondUniform,
    /// Post-hoc: first moment with the preconditioner off (κ → ∞).
    FirstOff,
}

struct VecBandit {
    prios: Vec<f32>,
}
impl HintDeltaBandit for VecBandit {
    fn absorb(&mut self, arm: usize, reward: f32) {
        if let Some(p) = self.prios.get_mut(arm) {
            *p += reward.max(0.0);
        }
    }
    fn priority(&self, arm: usize) -> Priority {
        self.prios.get(arm).copied().unwrap_or(0.0)
    }
    fn priorities(&self) -> &[Priority] {
        &self.prios
    }
    fn priorities_mut(&mut self) -> &mut [Priority] {
        &mut self.prios
    }
}

enum Conj {
    First(TrajectoryAlignedCuriosity<FirstMomentDrift<DIM>>),
    Second(TrajectoryAlignedCuriosity<SecondMomentDrift<N_ARMS>>),
    Global(DerivativeCuriosity<N_ARMS>),
}

/// Returns `(cycles_to_acquire, extrinsic_reward_over_window)`.
fn run_loop(arm: Arm, seed: u64) -> (usize, f32) {
    run_loop_with(arm, seed, false)
}

/// `reversed`: S is the extrinsically better family and the readout is S's
/// mass share (post-hoc characterization — does aligned only amplify a
/// COHERENT family's drift?).
fn run_loop_with(arm: Arm, seed: u64, reversed: bool) -> (usize, f32) {
    let pool = build_pool(seed, true);
    let mut conj = match arm {
        Arm::GlobalNorm => Conj::Global(DerivativeCuriosity::new(pool.clone(), seed)),
        Arm::Second | Arm::SecondUniform => Conj::Second(TrajectoryAlignedCuriosity::with_summary(
            pool.clone(),
            seed,
            SecondMomentDrift::for_pool(&pool),
        )),
        Arm::FirstOff => Conj::First(TrajectoryAlignedCuriosity::with_summary(
            pool.clone(),
            seed,
            FirstMomentDrift::new()
                .with_beta(BETA)
                .with_preconditioner(DEFAULT_SCALE_ALPHA, 1e6),
        )),
        _ => Conj::First(TrajectoryAlignedCuriosity::with_summary(
            pool.clone(),
            seed,
            FirstMomentDrift::new().with_beta(BETA),
        )),
    };
    let target = Target::new(pool[0].clone());
    let mut bandit = VecBandit {
        prios: vec![1.0 / N_ARMS as f32; N_ARMS],
    };
    let mut cands = vec![Candidate::new(Direction::zeros(DIM), usize::MAX); 4];
    let mut cdf = Vec::with_capacity(N_ARMS);
    let (mut acquired, mut ext_total) = (G3_CAP, 0.0f32);
    for c in 0..G3_CAP {
        match &mut conj {
            Conj::First(t) => t.sample_candidates(&target, &bandit.prios, &mut cands, &mut cdf),
            Conj::Second(t) => t.sample_candidates(&target, &bandit.prios, &mut cands, &mut cdf),
            Conj::Global(g) => g.sample_candidates(&target, &bandit.prios, &mut cands, &mut cdf),
        }
        for (slot, cand) in cands.iter().enumerate() {
            let a = cand.pool_index;
            if a == usize::MAX {
                continue;
            }
            let p_ext = if (a < N_F) != reversed { 0.20 } else { 0.10 };
            let ext = if keyed_uniform(seed, c, slot) < p_ext {
                1.0
            } else {
                0.0
            };
            let bonus = match (&conj, arm) {
                (Conj::First(t), Arm::Aligned | Arm::FirstOff) => t.last_alignment_scores()[slot],
                (Conj::First(t), Arm::MatchedUniform) => t.last_interestingness(),
                (Conj::Second(t), Arm::Second) => t.last_alignment_scores()[slot],
                (Conj::Second(t), Arm::SecondUniform) => t.last_interestingness(),
                (Conj::Global(g), _) => g.last_interestingness(),
                _ => 0.5,
            };
            bandit.absorb(a, ETA * (ext + LAMBDA * 2.0 * (bonus - 0.5)));
            if c < G3_EXT_WINDOW {
                ext_total += ext;
            }
        }
        renormalize_priorities(&mut bandit.prios);
        // `renormalize_priorities` rescales to max = 1, NOT sum = 1, so the
        // pre-registered readout (F's share of the priority mass) must divide.
        let total: f32 = bandit.prios.iter().sum();
        let mass_f = match reversed {
            false => bandit.prios[..N_F].iter().sum::<f32>(),
            true => bandit.prios[N_F..].iter().sum::<f32>(),
        } / total;
        if acquired == G3_CAP && mass_f >= ACQUIRE_MASS {
            acquired = c + 1;
        }
        if acquired < G3_CAP && c + 1 >= G3_EXT_WINDOW {
            break;
        }
    }
    (acquired, ext_total)
}

fn paired_ci(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let (m, sd) = mean_sd(&d);
    let h = 1.96 * sd / (d.len() as f64).sqrt();
    (m, m - h, m + h)
}

#[test]
fn g3_loop_ab_paired() {
    let arms = [
        Arm::Aligned,
        Arm::GlobalNorm,
        Arm::MatchedUniform,
        Arm::ExtrinsicOnly,
    ];
    let mut cyc: Vec<Vec<f64>> = vec![vec![]; 4];
    let mut ext: Vec<Vec<f64>> = vec![vec![]; 4];
    for seed in 0..G3_SEEDS {
        for (i, &arm) in arms.iter().enumerate() {
            let (c, e) = run_loop(arm, seed);
            cyc[i].push(c as f64);
            ext[i].push(e as f64);
        }
    }
    // Determinism pin: a rerun is bit-identical.
    assert_eq!(run_loop(Arm::Aligned, 5), run_loop(Arm::Aligned, 5));

    println!("\n═══ Plan 610 G3 — loop A/B ({G3_SEEDS} paired seeds, cap {G3_CAP}) ═══");
    for (i, arm) in arms.iter().enumerate() {
        let censored = cyc[i].iter().filter(|&&c| c >= G3_CAP as f64).count();
        println!(
            "  {:<15} cycles-to-acquire mean {:>6.1} sd {:>6.1} (censored {censored})  ext@{G3_EXT_WINDOW} mean {:>6.1}",
            format!("{arm:?}"),
            mean_sd(&cyc[i]).0,
            mean_sd(&cyc[i]).1,
            mean_sd(&ext[i]).0,
        );
    }
    let mut pass = true;
    for (j, name) in [
        (1usize, "GlobalNorm"),
        (2, "MatchedUniform"),
        (3, "ExtrinsicOnly"),
    ] {
        let (m, lo, hi) = paired_ci(&cyc[0], &cyc[j]);
        let (em, elo, ehi) = paired_ci(&ext[0], &ext[j]);
        println!(
            "  Aligned − {name:<14} Δcycles {m:+7.1} [{lo:+7.1}, {hi:+7.1}]   Δext {em:+6.1} [{elo:+6.1}, {ehi:+6.1}]"
        );
        if j <= 2 {
            pass &= hi < 0.0;
        }
    }
    println!(
        "  G3 pre-registered verdict (A beats B and C, CI excludes 0): {}",
        if pass {
            "PASS"
        } else {
            "NOT PASSED — demotion clause"
        }
    );
    for v in cyc.iter().chain(ext.iter()) {
        assert!(v.iter().all(|x| x.is_finite()));
    }
    assert!(
        pass,
        "G3 verdict changed from the recorded PASS — update Bench 900"
    );
}

#[test]
fn g3_characterization_reversed_reward() {
    // Post-hoc, no bar: S (spread, zero-centroid) is the better family.
    let arms = [
        Arm::Aligned,
        Arm::GlobalNorm,
        Arm::MatchedUniform,
        Arm::ExtrinsicOnly,
    ];
    let mut cyc: Vec<Vec<f64>> = vec![vec![]; 4];
    for seed in 0..G3_SEEDS {
        for (i, &arm) in arms.iter().enumerate() {
            cyc[i].push(run_loop_with(arm, seed, true).0 as f64);
        }
    }
    println!("\n═══ Plan 610 characterization — reversed reward (S better) ═══");
    for (i, arm) in arms.iter().enumerate() {
        let censored = cyc[i].iter().filter(|&&c| c >= G3_CAP as f64).count();
        println!(
            "  {:<15} cycles-to-acquire-S mean {:>6.1} (censored {censored})",
            format!("{arm:?}"),
            mean_sd(&cyc[i]).0
        );
    }
    for (j, name) in [
        (1usize, "GlobalNorm"),
        (2, "MatchedUniform"),
        (3, "ExtrinsicOnly"),
    ] {
        let (m, lo, hi) = paired_ci(&cyc[0], &cyc[j]);
        println!("  Aligned − {name:<14} Δcycles {m:+7.1} [{lo:+7.1}, {hi:+7.1}]");
        // Pin (Bench 900): aligned is SLOWER when the better family has a
        // zero pull centroid. The G3 PASS is coherent-cluster momentum.
        assert!(
            lo > 0.0,
            "reversed-reward sign changed vs {name} — update Bench 900"
        );
    }
}

#[test]
fn g3_pin_sampling_bit_identical_to_pool_conjecturer() {
    let pool = build_pool(11, true);
    let target = Target::new(pool[0].clone());
    let mut tac: TrajectoryAlignedCuriosity<FirstMomentDrift<DIM>> =
        TrajectoryAlignedCuriosity::new(pool.clone(), 99);
    let mut bare = PoolConjecturer::new(pool, 99);
    let (mut ca, mut cb) = (
        vec![Candidate::new(Direction::zeros(DIM), usize::MAX); 4],
        vec![Candidate::new(Direction::zeros(DIM), usize::MAX); 4],
    );
    let (mut fa, mut fb) = (Vec::new(), Vec::new());
    for p in planted_trajectory(11).iter() {
        tac.sample_candidates(&target, p, &mut ca, &mut fa);
        bare.sample_candidates(&target, p, &mut cb, &mut fb);
        for (x, y) in ca.iter().zip(&cb) {
            assert_eq!(x.pool_index, y.pool_index);
            let bits = |d: &Direction| d.coords.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&x.direction), bits(&y.direction));
        }
    }
}

// ── G4 — allocation + cost ───────────────────────────────────────────────

#[test]
#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
fn g4_cycle_aligned_is_alloc_free_when_warm() {
    use katgpt_core::alloc::{get_alloc_stats, reset_alloc_stats};
    use katgpt_core::cgsp::loop_::{CgspConfig, EntropyCollapse};
    use katgpt_core::cgsp::types::ScratchBuffers;
    reset_alloc_stats();
    let _v: Vec<u8> = Vec::with_capacity(8);
    let (sentinel, _) = get_alloc_stats();
    assert!(
        sentinel > 0,
        "TrackingAllocator not installed — alloc gate vacuous"
    );

    let pool = build_pool(1, true);
    let mut tac: TrajectoryAlignedCuriosity<FirstMomentDrift<DIM>> =
        TrajectoryAlignedCuriosity::new(pool.clone(), 1);
    let mut bandit = VecBandit {
        prios: vec![1.0 / N_ARMS as f32; N_ARMS],
    };
    let mut collapse = EntropyCollapse::default();
    let config = CgspConfig::default();
    let target = Target::new(pool[0].clone());
    let mut scratch = ScratchBuffers::new(config.k, N_ARMS);
    for _ in 0..100 {
        let _ = tac.cycle_aligned(&target, &mut bandit, &mut scratch, &mut collapse, &config);
    }
    reset_alloc_stats();
    for _ in 0..1000 {
        let r = tac.cycle_aligned(&target, &mut bandit, &mut scratch, &mut collapse, &config);
        std::hint::black_box(r.stats.mean_r_synth);
    }
    let (count, bytes) = get_alloc_stats();
    assert_eq!(
        count, 0,
        "cycle_aligned allocated {count} times ({bytes} B) over 1000 warm cycles"
    );
    println!("[G4] 1000 warm cycle_aligned calls: 0 allocations");

    // The incumbent's Solver-free cycle carried a per-cycle resize-default
    // allocation until Plan 610 G4 found it (same shape, fixed alongside).
    let mut dc: DerivativeCuriosity<N_ARMS> = DerivativeCuriosity::new(pool.clone(), 1);
    for _ in 0..100 {
        let _ = dc.cycle_curiosity(&target, &mut bandit, &mut scratch, &mut collapse, &config);
    }
    reset_alloc_stats();
    for _ in 0..1000 {
        let r = dc.cycle_curiosity(&target, &mut bandit, &mut scratch, &mut collapse, &config);
        std::hint::black_box(r.stats.mean_r_synth);
    }
    let (count, bytes) = get_alloc_stats();
    assert_eq!(
        count, 0,
        "cycle_curiosity allocated {count} times ({bytes} B) over 1000 warm cycles"
    );
    println!("[G4] 1000 warm DerivativeCuriosity::cycle_curiosity calls: 0 allocations");

    // Issue 899: the second-moment summary (its n×n kernels are built before
    // the counter is reset — construction is the only allocation it makes).
    let mut tac2 = TrajectoryAlignedCuriosity::with_summary(
        pool.clone(),
        1,
        SecondMomentDrift::<N_ARMS>::for_pool(&pool),
    );
    for _ in 0..100 {
        let _ = tac2.cycle_aligned(&target, &mut bandit, &mut scratch, &mut collapse, &config);
    }
    reset_alloc_stats();
    for _ in 0..1000 {
        let r = tac2.cycle_aligned(&target, &mut bandit, &mut scratch, &mut collapse, &config);
        std::hint::black_box(r.stats.mean_r_synth);
    }
    let (count, bytes) = get_alloc_stats();
    assert_eq!(
        count, 0,
        "second-moment cycle_aligned allocated {count} times ({bytes} B)"
    );
    println!("[G4] 1000 warm second-moment cycle_aligned calls: 0 allocations");
}

#[test]
fn g4_cost_vs_incumbent() {
    let pool = build_pool(2, true);
    let target = Target::new(pool[0].clone());
    let traj = planted_trajectory(2);
    let mut dc: DerivativeCuriosity<N_ARMS> = DerivativeCuriosity::new(pool.clone(), 2);
    let mut tac: TrajectoryAlignedCuriosity<FirstMomentDrift<DIM>> =
        TrajectoryAlignedCuriosity::new(pool.clone(), 2);
    let mut tac2 = TrajectoryAlignedCuriosity::with_summary(
        pool.clone(),
        2,
        SecondMomentDrift::<N_ARMS>::for_pool(&pool),
    );
    let mut cc = vec![Candidate::new(Direction::zeros(DIM), usize::MAX); 4];
    let mut fc = Vec::with_capacity(N_ARMS);
    let mut ca = vec![Candidate::new(Direction::zeros(DIM), usize::MAX); 4];
    let mut cb = ca.clone();
    let (mut fa, mut fb) = (Vec::with_capacity(N_ARMS), Vec::with_capacity(N_ARMS));
    let (mut sink_a, mut sink_b) = (0.0f32, 0.0f32);

    let sample = ab_timing::ab_median_ratio(
        31,
        2000,
        500,
        |i| {
            dc.sample_candidates(
                &target,
                std::hint::black_box(&traj[i % STEPS]),
                &mut ca,
                &mut fa,
            );
            sink_a += dc.last_interestingness() + ca[0].pool_index as f32;
        },
        |i| {
            tac.sample_candidates(
                &target,
                std::hint::black_box(&traj[i % STEPS]),
                &mut cb,
                &mut fb,
            );
            sink_b += tac.last_interestingness() + cb[0].pool_index as f32;
        },
    );
    let observe = ab_timing::ab_median_ratio(
        31,
        2000,
        500,
        |i| sink_a += dc.observe_interestingness(std::hint::black_box(&traj[i % STEPS])),
        |i| sink_b += tac.observe_drift(std::hint::black_box(&traj[i % STEPS])),
    );
    let (mut sink_c, mut sink_d) = (0.0f32, 0.0f32);
    let (mut cd, mut fd) = (ca.clone(), Vec::with_capacity(N_ARMS));
    let sample2 = ab_timing::ab_median_ratio(
        31,
        2000,
        500,
        |i| {
            dc.sample_candidates(
                &target,
                std::hint::black_box(&traj[i % STEPS]),
                &mut cd,
                &mut fd,
            );
            sink_c += dc.last_interestingness() + cd[0].pool_index as f32;
        },
        |i| {
            tac2.sample_candidates(
                &target,
                std::hint::black_box(&traj[i % STEPS]),
                &mut cc,
                &mut fc,
            );
            sink_d += tac2.last_interestingness() + cc[0].pool_index as f32;
        },
    );
    std::hint::black_box((sink_a, sink_b, sink_c, sink_d));
    println!(
        "\n═══ Plan 610 G4 — cost (a = incumbent, b = aligned; {N_ARMS} arms × dim {DIM}) ═══"
    );
    sample.report("sample_candidates");
    observe.report("observe only     ");
    sample2.report("second-moment sample_candidates (Issue 899)");
    #[cfg(not(debug_assertions))]
    assert!(
        sample.median <= 2.0,
        "G4 FAIL: aligned sample_candidates {:.3}× incumbent (bar ≤ 2.0×)",
        sample.median
    );
    // Measured-verdict pin (Bench 901): the simplex-centered null costs four
    // SIMD row dots per scored arm and lands at ~2.07×, just over the bar.
    #[cfg(not(debug_assertions))]
    assert!(
        sample2.median > 2.0 && sample2.median < 3.0,
        "Issue 899 G4 verdict moved ({:.3}×) — update Bench 901",
        sample2.median
    );
}

// ── Issue 899 — the second-moment, null-normalized redesign ─────────────

#[test]
fn issue_899_second_moment_gates() {
    // G1 + negative control, same fixture and seeds as Bench 900.
    let (mut all, mut held, mut noise) = (vec![], vec![], vec![]);
    for seed in 0..SEEDS {
        let pool = build_pool(seed, true);
        let s = second_scores(&pool, &planted_trajectory(seed));
        all.push(auc_f_vs_s(&s));
        held.push(auc_heldout(&s));
        noise.push(auc_f_vs_s(&second_scores(&pool, &noise_trajectory(seed))));
    }
    let (m_all, m_held, m_noise) = (mean_sd(&all).0, mean_sd(&held).0, mean_sd(&noise).0);

    // G3 forward (F better) and reversed (S better), 32 paired seeds.
    let arms = [
        Arm::Second,
        Arm::GlobalNorm,
        Arm::SecondUniform,
        Arm::ExtrinsicOnly,
    ];
    let mut fwd: Vec<Vec<f64>> = vec![vec![]; 4];
    let mut rev: Vec<Vec<f64>> = vec![vec![]; 4];
    for seed in 0..G3_SEEDS {
        for (i, &arm) in arms.iter().enumerate() {
            fwd[i].push(run_loop_with(arm, seed, false).0 as f64);
            rev[i].push(run_loop_with(arm, seed, true).0 as f64);
        }
    }
    assert_eq!(run_loop(Arm::Second, 5), run_loop(Arm::Second, 5));

    println!("\n═══ Issue 899 — second-moment, null-normalized ({SEEDS}/{G3_SEEDS} seeds) ═══");
    println!(
        "  G1 AUC(F vs S)      mean {m_all:.3}  min {:.3}  (bar ≥ 0.8)",
        fmin(&all)
    );
    println!(
        "  G1 AUC(held-out F)  mean {m_held:.3}  min {:.3}  (bar ≥ 0.8)",
        fmin(&held)
    );
    println!(
        "  noise AUC(F vs S)   mean {m_noise:.3}  sd {:.3}  (bar |m−0.5| ≤ 0.15)",
        mean_sd(&noise).1
    );
    for (label, cyc) in [("forward (F better)", &fwd), ("reversed (S better)", &rev)] {
        println!("  G3 {label}:");
        for (i, arm) in arms.iter().enumerate() {
            let censored = cyc[i].iter().filter(|&&c| c >= G3_CAP as f64).count();
            println!(
                "    {:<14} cycles-to-acquire mean {:>6.1} (censored {censored})",
                format!("{arm:?}"),
                mean_sd(&cyc[i]).0
            );
        }
        for (j, name) in [
            (1usize, "GlobalNorm"),
            (2, "SecondUniform"),
            (3, "ExtrinsicOnly"),
        ] {
            let (m, lo, hi) = paired_ci(&cyc[0], &cyc[j]);
            println!("    Second − {name:<14} Δcycles {m:+7.1} [{lo:+7.1}, {hi:+7.1}]");
        }
    }
    let g1 = m_all >= 0.8 && m_held >= 0.8;
    let neg = (m_noise - 0.5).abs() <= 0.15;
    let (_, _, hi_gn) = paired_ci(&fwd[0], &fwd[1]);
    let (_, _, hi_mu) = paired_ci(&fwd[0], &fwd[2]);
    let g3_fwd = hi_gn < 0.0 && hi_mu < 0.0;
    let (_, lo_rev, _) = paired_ci(&rev[0], &rev[2]);
    let g3_rev = lo_rev <= 0.0;
    let v = |b: bool| if b { "PASS" } else { "FAIL" };
    println!(
        "  verdicts: G1 {}  negative {}  G3-forward {}  G3-reversed {}",
        v(g1),
        v(neg),
        v(g3_fwd),
        v(g3_rev)
    );
    for c in fwd.iter().chain(rev.iter()) {
        assert!(c.iter().all(|x| x.is_finite()));
    }
    // Measured-verdict pins (Bench 901, v2 simplex-centered null + warm start).
    assert!(
        !g1,
        "Issue 899 G1 verdict changed (held-out was 0.766) — update Bench 901"
    );
    assert!(
        neg,
        "Issue 899 negative control regressed — update Bench 901"
    );
    assert!(g3_fwd, "Issue 899 G3-forward regressed — update Bench 901");
    assert!(g3_rev, "Issue 899 G3-reversed regressed — update Bench 901");
}

#[test]
fn characterization_first_moment_preconditioner_off_loop() {
    // Post-hoc (Bench 901): κ → ∞ makes the floor dominate every coordinate,
    // so `û → d/‖d‖` exactly — the preconditioner-off form, which passes G1
    // and the negative control once the transient is gone. Does it survive
    // the loop in both directions?
    let arms = [Arm::FirstOff, Arm::GlobalNorm, Arm::MatchedUniform];
    let mut fwd: Vec<Vec<f64>> = vec![vec![]; 3];
    let mut rev: Vec<Vec<f64>> = vec![vec![]; 3];
    for seed in 0..G3_SEEDS {
        for (i, &arm) in arms.iter().enumerate() {
            fwd[i].push(run_loop_with(arm, seed, false).0 as f64);
            rev[i].push(run_loop_with(arm, seed, true).0 as f64);
        }
    }
    println!("\n═══ Characterization — first moment, preconditioner OFF (κ→∞) ═══");
    for (label, cyc) in [("forward", &fwd), ("reversed", &rev)] {
        let (m, lo, hi) = paired_ci(&cyc[0], &cyc[2]);
        let (g, glo, ghi) = paired_ci(&cyc[0], &cyc[1]);
        println!(
            "  {label:<8} FirstOff {:>6.1}  − MatchedUniform {m:+7.1} [{lo:+7.1}, {hi:+7.1}]  − GlobalNorm {g:+7.1} [{glo:+7.1}, {ghi:+7.1}]",
            mean_sd(&cyc[0]).0
        );
    }
    let (_, lo_rev, _) = paired_ci(&rev[0], &rev[2]);
    // Pin: the first moment's spread-family blindness is structural, not a
    // preconditioner artifact — it still loses when S is the better family.
    assert!(
        lo_rev > 0.0,
        "P-off reversed-reward sign changed — update Bench 901"
    );
}

#[test]
fn issue_899_pin_sampling_bit_identical() {
    let pool = build_pool(11, true);
    let target = Target::new(pool[0].clone());
    let mut tac = TrajectoryAlignedCuriosity::with_summary(
        pool.clone(),
        99,
        SecondMomentDrift::<N_ARMS>::for_pool(&pool),
    );
    let mut bare = PoolConjecturer::new(pool, 99);
    let mut ca = vec![Candidate::new(Direction::zeros(DIM), usize::MAX); 4];
    let mut cb = ca.clone();
    let (mut fa, mut fb) = (Vec::new(), Vec::new());
    for p in planted_trajectory(11).iter() {
        tac.sample_candidates(&target, p, &mut ca, &mut fa);
        bare.sample_candidates(&target, p, &mut cb, &mut fb);
        for (x, y) in ca.iter().zip(&cb) {
            assert_eq!(x.pool_index, y.pool_index);
        }
    }
}
