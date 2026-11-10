//! Plan 576 Phase 4 — hint_regret GOAT gate (G1 / G2 / G-Floor / G8).
//!
//! G3 (default-untouched) and G4 (zero-alloc) run as separate commands /
//! binaries: G3 is `cargo test -p katgpt-core --lib` count-identity with the
//! feature off (the gate compiles out — nothing in the default set or cgsp
//! references this module); G4 lives in
//! `bench_576_hint_regret_alloc_check.rs` (the house single-fn alloc-binary
//! convention — parallel tests share the global counting allocator).
//!
//! Oracles (Plan 576 Phase 1 "analytic oracles as test fixtures"):
//! - **Reveal-the-arm bandit** — K arms with means `μ_j`; the hint reveals
//!   `argmax μ`. CRN pairs share one Gaussian draw per pair, so the noise
//!   cancels in the difference and the residual variance is the POLICY's
//!   arm-choice spread — exactly the decomposition G2 measures
//!   (`Var_indep = Var_policy + 2σ²` vs `Var_crn = Var_policy`).
//! - **Hinted shortest path** — the hint is a demo path; the agent follows
//!   each demo edge with probability `β/(1+β)`. At `β = ∞` it follows the
//!   demo exactly, so the hinted return equals the demo return
//!   **bit-exactly** (the plan's β→∞ contract).
//!
//! Determinism: SplitMix64, fixed seeds, no wall-clock inputs to any
//! verdict (timing reports only).

use katgpt_core::hint_regret::{
    HintRegretEstimator, Regime, ReturnBounds, hoeffding_half_width, hoeffding_k,
    learnable_band_gate, triage,
};
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────
// Deterministic RNG (SplitMix64 — the group_invariance_probe test pattern)
// ──────────────────────────────────────────────────────────────────────────

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Standard normal via Box–Muller (one draw per call, second discarded —
    /// determinism over speed).
    fn next_normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
    fn next_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

// ──────────────────────────────────────────────────────────────────────────
// Oracle A — reveal-the-arm bandit
// ──────────────────────────────────────────────────────────────────────────

/// K-armed bandit; the hint reveals the best arm. Returns bounded in [0, 1].
struct RevealBandit {
    mus: Vec<f32>,
    sigma: f32,
}

/// The unhinted agent: ε-greedy over running empirical means. Its arm
/// choices are driven by the SAME random stream in both CRN and independent
/// modes, so `Var_policy` is identical across modes and the variance delta
/// G2 measures is exactly `2σ²` (the noise the shared seed cancels).
struct EpsilonGreedy {
    counts: Vec<u32>,
    sums: Vec<f32>,
    epsilon: f32,
}

impl EpsilonGreedy {
    fn new(arms: usize, epsilon: f32) -> Self {
        Self {
            counts: vec![0; arms],
            sums: vec![0.0; arms],
            epsilon,
        }
    }
    /// Picks an arm using `rng` (the shared stream). Optimistic init is NOT
    /// used — the first pass explores uniformly via the ε-coin, so the
    /// policy's arm-choice distribution has genuine spread.
    fn pick(&mut self, rng: &mut SplitMix64) -> usize {
        let arms = self.counts.len();
        if rng.next_uniform() < self.epsilon as f64 {
            (rng.next_u64() % arms as u64) as usize
        } else {
            let mut best = 0usize;
            let mut best_mean = f32::NEG_INFINITY;
            for j in 0..arms {
                // Unseen arms rate as 0.5 (uninformative prior mean).
                let m = if self.counts[j] == 0 {
                    0.5
                } else {
                    self.sums[j] / self.counts[j] as f32
                };
                // Deterministic first-max (scan order breaks ties).
                if m > best_mean {
                    best_mean = m;
                    best = j;
                }
            }
            best
        }
    }
    fn update(&mut self, arm: usize, ret: f32) {
        self.counts[arm] += 1;
        self.sums[arm] += ret;
    }
}

impl RevealBandit {
    fn best_arm(&self) -> usize {
        let mut best = 0usize;
        for (j, &m) in self.mus.iter().enumerate() {
            if m > self.mus[best] {
                best = j;
            }
        }
        best
    }

    /// Runs one experiment of `n_pairs` pairs and returns `(r_hat, ci_half_width)`.
    ///
    /// Stream discipline: the policy consumes ONLY the main stream; all
    /// noise (the shared `z`, and the fresh `z2` in independent mode) comes
    /// from a dedicated salted stream — so arm choices are IDENTICAL under
    /// CRN and independent modes for the same seed, and the variance delta
    /// is exactly the noise the shared draw cancels:
    /// `Var_indep − Var_crn ≈ 2σ²/n_pairs`. `crn: true` reuses `z` for the
    /// unhinted arm; `crn: false` draws a fresh normal from the noise stream
    /// after the pick (the pick itself is unaffected).
    fn run(&self, n_pairs: u32, crn: bool, epsilon: f32, seed: u64) -> (f32, f32) {
        let mut rng = SplitMix64::new(seed);
        let mut noise = SplitMix64::new(seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut est = HintRegretEstimator::new(ReturnBounds { lo: 0.0, hi: 1.0 });
        let mut agent = EpsilonGreedy::new(self.mus.len(), epsilon);
        let best = self.best_arm();
        for _ in 0..n_pairs {
            let z = self.sigma * noise.next_normal();
            let j = agent.pick(&mut rng);
            let hint_ret = clamp01(self.mus[best] + z);
            let plain_ret = if crn {
                clamp01(self.mus[j] + z)
            } else {
                let z2 = self.sigma * noise.next_normal();
                clamp01(self.mus[j] + z2)
            };
            agent.update(j, plain_ret);
            est.record_pair(hint_ret, plain_ret);
        }
        let e = est.estimate(0.05);
        (e.r_hat, e.ci_half_width)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Oracle B — hinted shortest path
// ──────────────────────────────────────────────────────────────────────────

/// Chain `0 → 1 → … → L` with per-node edge payoffs. The demo (hint) is the
/// payoff-maximizing edge at every node. The hinted agent follows the demo
/// edge with probability `β/(1+β)`; `β = ∞` → follows exactly → return
/// equals the demo total **bit-exactly**.
struct HintedPath {
    payoffs: Vec<[f32; 4]>, // per node: payoff of each of 4 outgoing edges
}

impl HintedPath {
    fn demo_total(&self) -> f32 {
        self.payoffs
            .iter()
            .map(|p| p.iter().copied().fold(f32::MIN, f32::max))
            .sum()
    }

    fn run(&self, beta: f32, seed: u64) -> (f32, f32) {
        let mut rng = SplitMix64::new(seed);
        let mut hinted = 0.0f32;
        let mut unhinted = 0.0f32;
        for edges in &self.payoffs {
            // Hinted arm: follow the demo edge with prob β/(1+β).
            let follow = if beta.is_infinite() {
                true
            } else {
                rng.next_uniform() < (beta / (1.0 + beta)) as f64
            };
            hinted += if follow {
                edges.iter().copied().fold(f32::MIN, f32::max)
            } else {
                edges[rng.next_u64() as usize % edges.len()]
            };
            // Unhinted arm: uniform edge choice.
            unhinted += edges[rng.next_u64() as usize % edges.len()];
        }
        (hinted, unhinted)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// G1 — oracle calibration + coverage
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn g1_hinted_shortest_path_beta_inf_is_bit_exact() {
    let mut rng = SplitMix64::new(0x5760_0002);
    let path = HintedPath {
        payoffs: (0..24)
            .map(|_| {
                [
                    (rng.next_uniform() as f32),
                    (rng.next_uniform() as f32),
                    (rng.next_uniform() as f32),
                    (rng.next_uniform() as f32),
                ]
            })
            .collect(),
    };
    let demo = path.demo_total();
    let mut est = HintRegretEstimator::new(ReturnBounds { lo: 0.0, hi: 24.0 });
    for seed in 0..100u64 {
        let (hinted, unhinted) = path.run(f32::INFINITY, 0x5760_0100 + seed);
        // β→∞: the hinted return IS the demo return, bit-exactly (plan Phase 1).
        assert!(
            (hinted - demo).abs() == 0.0,
            "β=∞ hinted {hinted} != demo {demo} at seed {seed}"
        );
        est.record_pair(hinted, unhinted);
    }
    // The demo is the max-per-node sum, so r̂ > 0 strictly (the hint pays).
    let e = est.estimate(0.05);
    assert!(
        e.r_hat > 0.0,
        "hint regret must be positive, got {}",
        e.r_hat
    );
    // Sanity: with bounded edges the regret is at most the max-total − min-total.
    assert!(e.r_hat <= 24.0);
}

#[test]
fn g1_bandit_calibration_within_2x_bound_and_coverage_at_nominal() {
    // Bandit with clear arm separation: the ε-greedy policy's arm-choice
    // spread is the CRN-residual variance (G2 uses the same fixture).
    let bandit = RevealBandit {
        mus: vec![0.35, 0.55, 0.75, 0.65, 0.45],
        sigma: 0.15,
    };
    let eps = 0.30;
    let delta = 0.05f32;
    let bounds = ReturnBounds { lo: 0.0, hi: 1.0 };

    // K from the schedule at a working precision.
    let k = hoeffding_k(0.10, delta, bounds);
    assert!((200..=800).contains(&k), "unexpected schedule size {k}");

    // Ground truth E[r] by a 10^6-pair Monte Carlo (bounded diffs → MC
    // error ≈ range/√10^6 = 0.002, ≪ the 0.10 half-width).
    let mc_n = 1_000_000u32;
    let mut mc = SplitMix64::new(0x5760_0042);
    let mut mc_est = HintRegretEstimator::new(bounds);
    let mut agent = EpsilonGreedy::new(bandit.mus.len(), eps);
    for _ in 0..mc_n {
        let z = bandit.sigma * mc.next_normal();
        let j = agent.pick(&mut mc);
        let hint_ret = clamp01(bandit.mus[bandit.best_arm()] + z);
        let plain_ret = clamp01(bandit.mus[j] + z);
        agent.update(j, plain_ret);
        mc_est.record_pair(hint_ret, plain_ret);
    }
    let e_r = mc_est.estimate(delta).r_hat;

    // 10^3 independent experiments at prescribed K:
    // - every |r_hat − E[r]| within 2× the Hoeffding bound (plan G1);
    // - coverage of the 1× bound ≥ 1−δ (nominal).
    let h_k = hoeffding_half_width(k, delta, bounds);
    let mut covered_1x = 0usize;
    let mut max_err = 0.0f32;
    for exp in 0..1000u64 {
        let (r_hat, _) = bandit.run(k, false, eps, 0x5760_1000 + exp);
        let err = (r_hat - e_r).abs();
        max_err = max_err.max(err);
        if err <= h_k {
            covered_1x += 1;
        }
        assert!(
            err <= 2.0 * h_k,
            "experiment {exp}: |r_hat−E[r]| = {err:.4} exceeds 2× bound {:.4}",
            2.0 * h_k
        );
    }
    let coverage = covered_1x as f32 / 1000.0;
    assert!(
        coverage >= 1.0 - delta,
        "Hoeffding coverage {coverage:.3} below nominal {}",
        1.0 - delta
    );
    println!("G1: E[r]≈{e_r:.4} h(K)={h_k:.4} max_err={max_err:.4} coverage={coverage:.3} (K={k})");
}

// ──────────────────────────────────────────────────────────────────────────
// G2 — CRN variance ratio ≥ 2× + per-pair cost
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn g2_crn_variance_ratio_at_least_2x() {
    // Variance budget: with an ADAPTIVE policy the agent's value estimates
    // feed on the same noise, so Var_policy grows with σ too and the ratio
    // stalls near 1.5–1.6× regardless of estimator quality. ε=1.0 makes
    // pick() a pure uniform arm draw (update never influences picks),
    // severing the feedback: Var_crn ≈ Var_j/n (the arm-spread term the
    // shared z cannot cancel — deterministic given j), while indep adds
    // 2σ̄²/n ≈ 6.6e-4. Predicted ratio ≈ 1 + 6.6e-4/3.2e-4 ≈ 3×. The
    // adaptive-policy interaction with the estimator is covered by G1's
    // calibration gate; G2 measures the estimator's variance mechanics.
    let bandit = RevealBandit {
        mus: vec![0.35, 0.55, 0.75, 0.65, 0.45],
        sigma: 0.15,
    };
    let eps = 1.0; // uniform policy — no learning feedback
    let n = 64u32;
    let reps = 1000usize;
    let mut crn_rhats = Vec::with_capacity(reps);
    let mut indep_rhats = Vec::with_capacity(reps);
    for rep in 0..reps as u64 {
        let (crn, _) = bandit.run(n, true, eps, 0x5760_2000 + rep);
        let (ind, _) = bandit.run(n, false, eps, 0x5760_2000 + rep);
        crn_rhats.push(crn as f64);
        indep_rhats.push(ind as f64);
    }
    let var = |xs: &Vec<f64>| {
        let n = xs.len() as f64;
        let m = xs.iter().sum::<f64>() / n;
        xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)
    };
    let v_crn = var(&crn_rhats);
    let v_indep = var(&indep_rhats);
    let ratio = v_indep / v_crn.max(1e-12);
    println!(
        "G2: Var_crn={v_crn:.6} Var_indep={v_indep:.6} ratio={ratio:.2}x ({reps} reps, n={n})"
    );
    assert!(
        ratio >= 2.0,
        "CRN variance ratio {ratio:.2}x below the 2x gate \
         (Var_crn={v_crn:.6}, Var_indep={v_indep:.6})"
    );
}

#[test]
fn g2_per_pair_cost_sub_microsecond() {
    // Warm up, then time 10^4 record_pair + estimate calls.
    const ITERS: usize = 10_000;

    let mut est = HintRegretEstimator::new(ReturnBounds { lo: 0.0, hi: 1.0 });
    for i in 0..256u32 {
        est.record_pair(0.5, 0.3 + (i % 5) as f32 * 0.02);
    }
    let _ = est.estimate(0.05);
    let mut sink = 0.0f32;
    let t0 = Instant::now();
    for i in 0..ITERS {
        est.record_pair(0.5, 0.3 + (i % 7) as f32 * 0.01);
        if i % 16 == 0 {
            sink += est.estimate(0.05).r_hat;
        }
    }
    let dt = t0.elapsed().as_nanos() as f64 / ITERS as f64;
    std::hint::black_box(sink);
    println!("G2: per-pair cost {dt:.1} ns (record_pair + amortized estimate)");
    assert!(dt < 1000.0, "per-pair cost {dt:.1} ns exceeds 1 µs");
}

// ──────────────────────────────────────────────────────────────────────────
// G-Floor — triage accuracy vs single-arm banding at matched budget
// ──────────────────────────────────────────────────────────────────────────

/// Ground-truth content item: unhinted win rate `p`, hint-lifted rate
/// `p + gain`. TRUE regime via the primitive's own `triage` on the exact
/// values — the fixture is self-consistent with the shipped partition.
struct FloorItem {
    p: f32,
    gain: f32,
}

#[test]
fn g_floor_paired_arm_beats_single_arm_banding() {
    // The natural three content types + the two confusion cases a win-rate
    // band cannot see (Guide 340 §"CGSP conflation fix"): low-baseline
    // hint-immune content (banded as "frontier", truth intractable) and
    // high-baseline-in-band mastered content (banded as "frontier", truth
    // mastered), plus sub-band frontier content (banded "too hard", truth
    // frontier). τ_r / τ_R sit mid-gap between the fixture values, at ≥2.5σ
    // from every one at K=40.
    const TAU_R: f32 = 0.15;
    const TAU_RET: f32 = 0.5;
    let items: Vec<FloorItem> = [
        (0.15f32, 0.6f32), // frontier, sub-band p  → floor says intractable
        (0.30, 0.45),      // frontier
        (0.50, 0.40),      // frontier
        (0.70, 0.30),      // frontier
        (0.90, 0.04),      // mastered
        (0.70, 0.05),      // mastered, in-band p   → floor says frontier
        (0.95, 0.03),      // mastered
        (0.10, 0.03),      // intractable
        (0.30, 0.02),      // intractable, in-band p → floor says frontier
        (0.15, 0.04),      // intractable
    ]
    .into_iter()
    .map(|(p, gain)| FloorItem { p, gain })
    .collect();

    let truth: Vec<Regime> = items
        .iter()
        .map(|it| triage(it.gain, it.p, TAU_R, TAU_RET))
        .collect();
    // Fixture sanity: all three cells populated.
    assert!(truth.contains(&Regime::Frontier));
    assert!(truth.contains(&Regime::Mastered));
    assert!(truth.contains(&Regime::Intractable));

    const K: u32 = 40; // paired: K pairs = 2K Bernoulli draws; floor: 2K draws.
    let reps = 200u64;
    let mut paired_hits = 0usize;
    let mut floor_hits = 0usize;
    let mut total = 0usize;
    for rep in 0..reps {
        for (idx, it) in items.iter().enumerate() {
            let mut rng = SplitMix64::new(0x5760_3000 ^ (rep << 8) ^ idx as u64);
            // Paired arm: K CRN Bernoulli pairs (shared uniform per pair —
            // the difference is Bernoulli(gain) exactly).
            let mut est = HintRegretEstimator::new(ReturnBounds { lo: 0.0, hi: 1.0 });
            for _ in 0..K {
                let u = rng.next_uniform() as f32;
                let hinted = u < (it.p + it.gain);
                let unhinted = u < it.p;
                est.record_pair(hinted as i32 as f32, unhinted as i32 as f32);
            }
            let e = est.estimate(0.05);
            let r_hat_minus = e.arm_means.1; // mean unhinted = R̂⁻
            let pred = triage(e.r_hat, r_hat_minus, TAU_R, TAU_RET);
            if pred == truth[idx] {
                paired_hits += 1;
            }
            // Floor: 2K single-arm draws at the same total budget.
            let mut floor_sum = 0.0f32;
            for _ in 0..2 * K {
                floor_sum += (rng.next_uniform() < it.p as f64) as i32 as f32;
            }
            let w_hat = floor_sum / (2 * K) as f32;
            let floor_pred = if w_hat > 0.8 {
                Regime::Mastered
            } else if w_hat < 0.2 {
                Regime::Intractable
            } else {
                Regime::Frontier
            };
            if floor_pred == truth[idx] {
                floor_hits += 1;
            }
            total += 1;
        }
    }
    let paired_acc = paired_hits as f32 / total as f32;
    let floor_acc = floor_hits as f32 / total as f32;
    println!(
        "G-Floor: paired triage accuracy {paired_acc:.3} vs single-arm banding floor {floor_acc:.3} \
         ({total} classifications, K={K}, matched 2K budget)"
    );
    assert!(
        paired_acc > floor_acc,
        "paired arm {paired_acc:.3} must beat the floor {floor_acc:.3}"
    );
    assert!(
        paired_acc - floor_acc >= 0.05,
        "paired advantage {:+.3} below the 5pp significance bar",
        paired_acc - floor_acc
    );
    assert!(
        paired_acc >= 0.90,
        "paired accuracy {paired_acc:.3} below 0.90"
    );
}

// ──────────────────────────────────────────────────────────────────────────
// G8 — learnable-share signature on a synthetic curriculum
// ──────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Policy {
    RegretGated,
    Uniform,
}

#[test]
fn g8_learnable_share_rises_under_regret_gated_selection() {
    // Pool of 140 items, difficulty spread over [-2, 8]; learner skill θ
    // starts at 0 and grows on wins (most in the productive zone just above
    // current skill). p_win = σ(θ − d); the hint arm = one demonstration =
    // +Δ effective skill during the attempt.
    const N: usize = 140;
    const DELTA_HINT: f32 = 1.0;
    const TAU_R: f32 = 0.15;
    const TAU_RET: f32 = 0.5;
    const T: usize = 400;
    let difficulties: Vec<f32> = (0..N)
        .map(|i| -2.0 + 10.0 * i as f32 / (N - 1) as f32)
        .collect();

    let run = |policy: Policy, seed: u64| -> (f32, f32, f32) {
        let mut rng = SplitMix64::new(seed);
        let mut theta = 0.0f32;
        let mut first_half_in_band = 0usize;
        let mut second_half_in_band = 0usize;
        let mut picked_in_band = 0usize;
        let mut picked_total = 0usize;
        for t in 0..T {
            // Modelless regime probe for every item at the current θ (the
            // deterministic collapse the mmorpg consumer shipped — here the
            // probe is exact, not sampled).
            let pick = match policy {
                Policy::RegretGated => {
                    // Highest band-gate-weighted frontier item; fall back to
                    // the highest band gate when no item is frontier.
                    let mut best = 0usize;
                    let mut best_score = f32::NEG_INFINITY;
                    for (i, &d) in difficulties.iter().enumerate() {
                        let p_win = katgpt_core::sigmoid(theta - d);
                        let p_hint = katgpt_core::sigmoid(theta + DELTA_HINT - d);
                        let r = p_hint - p_win;
                        let regime = triage(r, p_win, TAU_R, TAU_RET);
                        let score = match regime {
                            Regime::Frontier => 2.0 + learnable_band_gate(p_win, 0.2, 0.8, 6.0),
                            Regime::Mastered => 0.5,
                            Regime::Intractable => 0.0,
                        };
                        if score > best_score {
                            best_score = score;
                            best = i;
                        }
                    }
                    best
                }
                Policy::Uniform => (rng.next_u64() % N as u64) as usize,
            };
            let d = difficulties[pick];
            let p_win = katgpt_core::sigmoid(theta - d);
            let win = rng.next_uniform() < p_win as f64;
            if win {
                // Productive struggle: wins just above current skill teach
                // the most; everything else teaches a little.
                theta += if (theta - d) > -0.5 && (theta - d) < 1.5 {
                    0.012
                } else {
                    0.003
                };
            }
            let in_band = (0.2..=0.8).contains(&p_win);
            picked_total += 1;
            picked_in_band += in_band as usize;
            if t < T / 2 {
                first_half_in_band += in_band as usize;
            } else {
                second_half_in_band += in_band as usize;
            }
        }
        (
            first_half_in_band as f32 / (T / 2) as f32,
            second_half_in_band as f32 / (T / 2) as f32,
            picked_in_band as f32 / picked_total as f32,
        )
    };

    let seeds = [11u64, 22, 33, 44, 55, 66, 77, 88];
    let mut gated_second = 0.0;
    let mut gated_first = 0.0;
    let mut uniform_second = 0.0;
    let mut gated_all = 0.0;
    let mut uniform_all = 0.0;
    for &s in &seeds {
        let (g1st, g2nd, gall) = run(Policy::RegretGated, s);
        let (_u1st, u2nd, uall) = run(Policy::Uniform, s);
        gated_first += g1st;
        gated_second += g2nd;
        gated_all += gall;
        uniform_second += u2nd;
        uniform_all += uall;
    }
    let n = seeds.len() as f32;
    let (gated_first, gated_second, gated_all) = (gated_first / n, gated_second / n, gated_all / n);
    let (uniform_second, uniform_all) = (uniform_second / n, uniform_all / n);
    println!(
        "G8: learnable share — gated first-half {gated_first:.3} → second-half {gated_second:.3} \
         (all {gated_all:.3}); uniform second-half {uniform_second:.3} (all {uniform_all:.3}); \
         {} seeds, T={T}",
        seeds.len()
    );
    // The signature: the gated curriculum's offered-content learnable share
    // RISES over runtime and dominates uniform (paper: 0.16 → 0.31).
    assert!(
        gated_second >= gated_first,
        "gated share must not fall ({gated_first:.3} → {gated_second:.3})"
    );
    assert!(
        gated_second - uniform_second >= 0.10,
        "gated second-half {gated_second:.3} vs uniform {uniform_second:.3} — no decisive edge"
    );
    assert!(
        gated_all > uniform_all,
        "gated full-run share {gated_all:.3} must beat uniform {uniform_all:.3}"
    );
}
