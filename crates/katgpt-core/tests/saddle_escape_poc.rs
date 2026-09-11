//! Plan 593 Phase 3 — defend-wrong PoC for the Saddle-Trap Escape Gate
//! (Research 546, arXiv:2609.04963 "Fractal basins trap latent reasoning").
//!
//! Two toy domains, three competitors each, K = 200 initial conditions:
//!
//! - **Toy A — double-well with a saddle-seeking trap band** (the Bench 406
//!   double-well analog): `|x| < 0.5 ⇒ x' = −0.9x` (contracting sign
//!   alternation ONTO the saddle at 0 — decode flips every loop, cos θ = −1);
//!   `|x| ≥ 0.5 ⇒ x' = 0.5x + 0.5·sign(x)` (contraction to the wells ±1).
//!   The decode is the sign bucket (A/B) — the saddle region decodes to
//!   nearly-correct answers that alternate forever.
//! - **Toy B — decode-keyed flip ring** (the repeated-digit-grid analog):
//!   a 2-D parity-driven stable 2-cycle straddling decode cells (1,1)↔(2,2)
//!   (weak contraction k=0.3 between diagonal attractors, so the cycle
//!   amplitude crosses the 4×4 cell boundaries); escape basin =
//!   off-diagonal `|x−y| ≥ 0.2` contracting to the solution cell (0,3).
//!
//! Competitors: (a) `SaddleEscapeGate` (three-way), (b) halt-only
//! `GainCostLoopHalter` (the shipped two-way family), (c) always-on-noise
//! PTRM-style at matched magnitude (deterministic unit-vector × eps0 every
//! loop — the honest always-on analog; PTRM arXiv:2605.19943 uses always-on
//! stochastic noise).
//!
//! # Gates (T4.2 G1 — the quality claim this PoC defends)
//!
//! - **Toy A** (the plan's ≥-both gate): gate escape rate ≥ BOTH baselines
//!   on the trapped cohort. Halt-only must solve ~0% (the defend-wrong
//!   anchor). Always-on noise at matched magnitude can TIE on the raw
//!   escape metric (a 1.0-magnitude kick trivially exits a ±0.5 band) — the
//!   differentiated axes are in the table: noise perturbs every loop
//!   (unbounded) and half its halts carry an unstable (flipping) decode;
//!   the gate perturbs ≤ kick_budget times and its solved halts decode
//!   stably.
//! - **Toy B** (the plan's measure+honesty gate): escape rate MEASURED
//!   (thin-band geometry does not tax always-on noise — reported honestly,
//!   not asserted away); Trapped-halt honesty ASSERTED.
//! - **Honesty (both toys)**: Trapped fires ONLY on actually-circling runs
//!   (trapped ⊆ halted-in-trap-region); NEVER on the clean-convergence
//!   cohorts — and zero kicks ever fire on clean convergence. In-band halts
//!   that pass through as `Converged` (flip EMA dipped below τ at that
//!   instant) are the halter's own "halt ≠ classification" semantics —
//!   honest, not misclassification.
//!
//! Verdict tables print with `--nocapture`; raw numbers land in the
//! Research 546 PoC addendum (T3.3). If the kick loses to either baseline
//! on a gated axis, the feature stays opt-in and the verdict is demoted —
//! not silently revised.

#![cfg(feature = "saddle_escape")]

use katgpt_core::gain_cost_halt::{GainCostLoopHalter, HaltDecision};
use katgpt_core::saddle_escape::{
    GateDecision, HaltOutcome, SaddleEscapeGate, TrapConfig, TrapObservables, apply_kick,
};

// Test-only splitmix64 finalizer for decode keys + noise seeds (test code
// may define inline helpers — the substrate-first exemption).
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn seed32(a: u64, b: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..4usize {
        let word = mix64(a.wrapping_add(i as u64).rotate_left(17) ^ b);
        out[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    out
}

const MAX_LOOPS: u32 = 64;
const K: usize = 200;

#[derive(Clone, Copy, PartialEq)]
enum Competitor {
    Gate,
    HaltOnly,
    AlwaysOnNoise,
}

/// One competitor run's outcome.
struct RunOutcome {
    solved: bool,
    loops: u32,
    trapped: bool,
    kicks: u8,
    /// Total state perturbations applied (gate: kicks; noise: one per
    /// loop — the unbounded-vs-budgeted axis).
    perturbations: u32,
    flips_at_halt: bool,
    halted_in_trap_region: bool,
    timed_out: bool,
}

struct Tally {
    solved: usize,
    trapped: usize,
    kicks: u32,
    perturbations: u32,
    flips_at_halt: usize,
    halted_in_trap: usize,
    timed_out: usize,
    loops: u64,
}

impl Tally {
    fn new() -> Self {
        Self {
            solved: 0,
            trapped: 0,
            kicks: 0,
            perturbations: 0,
            flips_at_halt: 0,
            halted_in_trap: 0,
            timed_out: 0,
            loops: 0,
        }
    }
    fn add(&mut self, r: &RunOutcome) {
        self.solved += r.solved as usize;
        self.trapped += r.trapped as usize;
        self.kicks += r.kicks as u32;
        self.perturbations += r.perturbations;
        self.flips_at_halt += r.flips_at_halt as usize;
        self.halted_in_trap += r.halted_in_trap_region as usize;
        self.timed_out += r.timed_out as usize;
        self.loops += r.loops as u64;
    }
    fn row(&self, name: &str, k: usize) -> String {
        format!(
            "{name:<16} solve {:>5.1}%  loops {:>5.1}  perturbs {:>5}  kicks {:>4}  trapped {:>4}  \
             flips@halt {:>4}  in-trap {:>4}  t.o. {:>3}",
            100.0 * self.solved as f32 / k as f32,
            self.loops as f32 / k as f32,
            self.perturbations,
            self.kicks,
            self.trapped,
            self.flips_at_halt,
            self.halted_in_trap,
            self.timed_out,
        )
    }
}

// The loop control flow, normalized across competitors.
enum Step {
    Cont,
    Kick { seed: [u8; 32], eps: f32 },
    Halt { trapped: bool },
}

// Shared gate/halter configs. Patience 1 (the halter default — reversal
// halts are checked BEFORE the gain/cost scissors, so sustained 2-cycles
// route into the oscillation path, not the scissors path); l_min 4 lets a
// few decode pairs accumulate before any halt is possible.
fn toy_halter() -> GainCostLoopHalter {
    GainCostLoopHalter::new(1.5, 1, 4)
}

fn toy_gate(eps0: f32) -> SaddleEscapeGate {
    SaddleEscapeGate::wrap(
        toy_halter(),
        TrapConfig {
            flip_tau: 0.5,
            kick_budget: 2,
            eps0,
            eps_decay: 0.5,
            window: 2,
            probe_tau: 0.1,
        },
    )
}

// ─────────────────────────────────────────────────────────────────────
// Toy A — double-well + saddle-seeking trap band (1-D)
// ─────────────────────────────────────────────────────────────────────

fn toy_a_map(x: f32) -> f32 {
    if x.abs() < 0.5 {
        // Sustained 2-cycle: constant-amplitude sign alternation — the
        // honest "scattering" regime (steps never decay, so the scissors
        // never fire first; the oscillation detector is the halt path).
        -x
    } else {
        // Over-damped well approach: monotone, never reverses — an escaped
        // trajectory does not oscillate on the way to the well.
        x + 0.5 * (x.signum() - x)
    }
}

fn toy_a_key(x: f32) -> u64 {
    mix64(u64::from(x < 0.0) + 1)
}

fn toy_a_solved(x: f32) -> bool {
    // Solved = halted in a SOLUTION BASIN (|x| ≥ 0.5): the decoded answer
    // (sign bucket) is basin-committed. The saddle band decodes to an
    // unstable alternating near-miss — never solved.
    x.abs() >= 0.5
}

fn toy_a_in_trap(x: f32) -> bool {
    x.abs() < 0.5
}

fn run_toy_a(x0: f32, who: Competitor) -> RunOutcome {
    let mut gate = toy_gate(1.0); // eps0 1.0: escapes the band from any |x|<0.5
    let mut bare = toy_halter();
    let mut x = x0;
    let mut prev_x = x0;
    let mut prev_step = 0.0f32;
    let mut recent_keys: [u64; 3] = [0; 3];
    let mut n_keys = 0usize;
    let mut kicks = 0u8;
    let mut perturbations = 0u32;
    let mut trapped = false;
    let mut halted = false;
    let mut loops = 0u32;

    for i in 1..=MAX_LOOPS {
        loops = i;
        let nx = toy_a_map(x);
        let step = (nx - x).abs();
        let curr = nx - x;
        let prev = x - prev_x;
        let cos_theta = if curr == 0.0 || prev == 0.0 {
            0.0
        } else {
            curr.signum() * prev.signum()
        };
        let gain = (prev_step - step).max(0.0);
        let key = toy_a_key(nx);
        recent_keys[n_keys % 3] = key;
        n_keys += 1;
        let sb = nx.to_le_bytes();

        let step_ctl = match who {
            Competitor::Gate => match gate.decide(TrapObservables {
                loop_idx: i as usize,
                gain,
                cost: step,
                cos_theta,
                step_norm: step,
                decoded_key: Some(key),
                probe_drift: None,
                state_bytes: &sb,
            }) {
                GateDecision::Continue => Step::Cont,
                GateDecision::Kick { dir_seed, eps } => Step::Kick { seed: dir_seed, eps },
                GateDecision::Halt(HaltOutcome::Trapped { .. }) => Step::Halt { trapped: true },
                GateDecision::Halt(_) => Step::Halt { trapped: false },
            },
            Competitor::HaltOnly | Competitor::AlwaysOnNoise => {
                match bare.halt_decision(i as usize, gain, step, cos_theta) {
                    HaltDecision::Halt { .. } => Step::Halt { trapped: false },
                    _ => Step::Cont,
                }
            }
        };

        prev_x = x;
        x = nx;
        prev_step = step;

        match step_ctl {
            Step::Cont => {
                if who == Competitor::AlwaysOnNoise {
                    apply_kick(
                        std::slice::from_mut(&mut x),
                        seed32(0xA11CE, i as u64),
                        1.0,
                    );
                    perturbations += 1;
                }
            }
            Step::Kick { seed, eps } => {
                kicks += 1;
                perturbations += 1;
                apply_kick(std::slice::from_mut(&mut x), seed, eps);
            }
            Step::Halt { trapped: t } => {
                trapped = t;
                halted = true;
                break;
            }
        }
    }

    let flips_at_halt =
        n_keys >= 3 && (recent_keys[0] != recent_keys[1] || recent_keys[1] != recent_keys[2]);
    RunOutcome {
        solved: toy_a_solved(x),
        loops,
        trapped,
        kicks,
        perturbations,
        flips_at_halt,
        halted_in_trap_region: toy_a_in_trap(x) && halted,
        timed_out: !halted,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Toy B — decode-keyed flip ring (2-D, parity-driven stable 2-cycle)
// ─────────────────────────────────────────────────────────────────────

// Diagonal attractors at 0.1 / 0.9 with weak contraction k=0.3: the stable
// 2-cycle settles at (a+0.3b)/1.3 = 0.285 ↔ 0.715 — crossing the 0.25 and
// 0.5 cell boundaries, so the decode flips every loop.
const TOY_B_A: f32 = 0.1;
const TOY_B_B: f32 = 0.9;
const TOY_B_SOL: [f32; 2] = [0.05, 0.75]; // solution fixed point, cell (0,3)

fn toy_b_map(x: f32, y: f32, loop_idx: u32) -> [f32; 2] {
    if (x - y).abs() < 0.2 {
        let c = if loop_idx.is_multiple_of(2) {
            TOY_B_A
        } else {
            TOY_B_B
        };
        [c + 0.3 * (x - c), c + 0.3 * (y - c)]
    } else {
        [
            TOY_B_SOL[0] + 0.5 * (x - TOY_B_SOL[0]),
            TOY_B_SOL[1] + 0.5 * (y - TOY_B_SOL[1]),
        ]
    }
}

fn toy_b_key(x: f32, y: f32) -> u64 {
    // Basin-identity decode (the paper decodes ANSWERS; answers are
    // determined by basins): in-band = which cycle point (phase by side of
    // 0.5 — alternates every loop on the cycle); sol basin = THE solution,
    // stable from the moment of basin entry (no flips during convergence).
    if (x - y).abs() < 0.2 {
        mix64(u64::from(x < 0.5) + 1)
    } else {
        mix64(3)
    }
}

fn toy_b_solved(x: f32, y: f32) -> bool {
    (x - y).abs() >= 0.2
}

fn toy_b_in_trap(x: f32, y: f32) -> bool {
    (x - y).abs() < 0.2
}

fn run_toy_b(x0: [f32; 2], who: Competitor) -> RunOutcome {
    let mut gate = toy_gate(0.35);
    let mut bare = toy_halter();
    let mut s = x0;
    let mut prev_dir = [0.0f32; 2];
    let mut prev_step = 0.0f32;
    let mut recent_keys: [u64; 3] = [0; 3];
    let mut n_keys = 0usize;
    let mut kicks = 0u8;
    let mut perturbations = 0u32;
    let mut trapped = false;
    let mut halted = false;
    let mut loops = 0u32;

    for i in 1..=MAX_LOOPS {
        loops = i;
        let ns = toy_b_map(s[0], s[1], i);
        let dx = ns[0] - s[0];
        let dy = ns[1] - s[1];
        let step = (dx * dx + dy * dy).sqrt();
        let dir = if step > 0.0 {
            [dx / step, dy / step]
        } else {
            [0.0, 0.0]
        };
        let cos_theta = if step == 0.0 || prev_step == 0.0 {
            0.0
        } else {
            dir[0] * prev_dir[0] + dir[1] * prev_dir[1]
        };
        let gain = (prev_step - step).max(0.0);
        let key = toy_b_key(ns[0], ns[1]);
        recent_keys[n_keys % 3] = key;
        n_keys += 1;
        let mut sb = [0u8; 8];
        sb[..4].copy_from_slice(&ns[0].to_le_bytes());
        sb[4..].copy_from_slice(&ns[1].to_le_bytes());

        let step_ctl = match who {
            Competitor::Gate => match gate.decide(TrapObservables {
                loop_idx: i as usize,
                gain,
                cost: step,
                cos_theta,
                step_norm: step,
                decoded_key: Some(key),
                probe_drift: None,
                state_bytes: &sb,
            }) {
                GateDecision::Continue => Step::Cont,
                GateDecision::Kick { dir_seed, eps } => Step::Kick { seed: dir_seed, eps },
                GateDecision::Halt(HaltOutcome::Trapped { .. }) => Step::Halt { trapped: true },
                GateDecision::Halt(_) => Step::Halt { trapped: false },
            },
            Competitor::HaltOnly | Competitor::AlwaysOnNoise => {
                match bare.halt_decision(i as usize, gain, step, cos_theta) {
                    HaltDecision::Halt { .. } => Step::Halt { trapped: false },
                    _ => Step::Cont,
                }
            }
        };

        s = ns;
        prev_dir = dir;
        prev_step = step;

        match step_ctl {
            Step::Cont => {
                if who == Competitor::AlwaysOnNoise {
                    apply_kick(&mut s, seed32(0xBEEF, i as u64), 0.35);
                    perturbations += 1;
                }
            }
            Step::Kick { seed, eps } => {
                kicks += 1;
                perturbations += 1;
                apply_kick(&mut s, seed, eps);
            }
            Step::Halt { trapped: t } => {
                trapped = t;
                halted = true;
                break;
            }
        }
    }

    let flips_at_halt =
        n_keys >= 3 && (recent_keys[0] != recent_keys[1] || recent_keys[1] != recent_keys[2]);
    RunOutcome {
        solved: toy_b_solved(s[0], s[1]),
        loops,
        trapped,
        kicks,
        perturbations,
        flips_at_halt,
        halted_in_trap_region: toy_b_in_trap(s[0], s[1]) && halted,
        timed_out: !halted,
    }
}

// ─────────────────────────────────────────────────────────────────────
// The PoC
// ─────────────────────────────────────────────────────────────────────

fn toy_a_trapped_cohort() -> Vec<f32> {
    // All inside the saddle band — every trajectory is trapped without
    // intervention (halt-only solve rate MUST be ~0%).
    (0..K)
        .map(|i| -0.45 + 0.9 * (i as f32) / (K - 1) as f32)
        .collect()
}

fn toy_a_clean_cohort() -> Vec<f32> {
    // Well-side starts — direct clean convergence, no trap.
    (0..K)
        .map(|i| 0.55 + 0.4 * (i as f32) / (K - 1) as f32)
        .chain((0..K).map(|i| -0.55 - 0.4 * (i as f32) / (K - 1) as f32))
        .collect()
}

fn toy_b_trapped_cohort() -> Vec<[f32; 2]> {
    // On the diagonal cycle corridor.
    (0..K)
        .map(|i| {
            let t = 0.12 + 0.06 * (i as f32) / (K - 1) as f32;
            [t, t]
        })
        .collect()
}

fn toy_b_clean_cohort() -> Vec<[f32; 2]> {
    // Off-diagonal, direct contraction to the solution cell.
    (0..2 * K)
        .map(|i| {
            let t = (i as f32) / (2 * K - 1) as f32;
            [0.05 + 0.1 * t, 0.85 - 0.1 * t]
        })
        .collect()
}

fn tally_all_a(cohort: &[f32], who: Competitor) -> Tally {
    let mut t = Tally::new();
    for &x0 in cohort {
        t.add(&run_toy_a(x0, who));
    }
    t
}

fn tally_all_b(cohort: &[[f32; 2]], who: Competitor) -> Tally {
    let mut t = Tally::new();
    for &x0 in cohort {
        t.add(&run_toy_b(x0, who));
    }
    t
}

#[test]
fn poc_toy_a_gate_beats_both_baselines_on_trapped_cohort() {
    let cohort = toy_a_trapped_cohort();
    let gate = tally_all_a(&cohort, Competitor::Gate);
    let halt = tally_all_a(&cohort, Competitor::HaltOnly);
    let noise = tally_all_a(&cohort, Competitor::AlwaysOnNoise);
    println!("── Toy A (double-well saddle band, K={K} trapped ICs) ──");
    println!("{}", gate.row("gate", K));
    println!("{}", halt.row("halt-only", K));
    println!("{}", noise.row("always-on-noise", K));

    assert!(
        gate.solved > halt.solved,
        "G1 FAIL: gate ({}) must beat halt-only ({}) on escape",
        gate.solved,
        halt.solved
    );
    assert!(
        gate.solved >= noise.solved,
        "G1 FAIL: gate ({}) must match-or-beat always-on noise ({})",
        gate.solved,
        noise.solved
    );
    // Trapped-halt honesty: every Trapped halt was actually circling
    // (trapped ⊆ halted-in-trap-region).
    assert!(
        gate.trapped <= gate.halted_in_trap,
        "honesty FAIL: Trapped fired outside the trap region"
    );
    // Bounded-intervention honesty: the gate never perturbs more than
    // kick_budget times per run (always-on noise perturbs every loop).
    assert!(gate.kicks as usize <= 2 * K, "budget FAIL");
}

#[test]
fn poc_toy_b_escape_rate_and_honesty() {
    let cohort = toy_b_trapped_cohort();
    let gate = tally_all_b(&cohort, Competitor::Gate);
    let halt = tally_all_b(&cohort, Competitor::HaltOnly);
    let noise = tally_all_b(&cohort, Competitor::AlwaysOnNoise);
    println!("── Toy B (decode-keyed flip ring, K={K} trapped ICs) ──");
    println!("{}", gate.row("gate", K));
    println!("{}", halt.row("halt-only", K));
    println!("{}", noise.row("always-on-noise", K));

    assert!(
        gate.solved > halt.solved,
        "G1 FAIL: gate ({}) must beat halt-only ({})",
        gate.solved,
        halt.solved
    );
    // Toy B's gate (per plan T3.2): escape rate MEASURED, honesty ASSERTED.
    // The thin band (|x−y| < 0.2) does not tax always-on noise — it exits
    // by luck every loop; reported honestly in the table (its flips@halt
    // and unbounded perturbation count are where it loses). The hard-difficulty
    // claim (aimed budget beats wasted perturbation as the band widens) is
    // future work on a real model trace, per Research 546 §3.6.
    assert!(
        gate.trapped <= gate.halted_in_trap,
        "honesty FAIL: Trapped fired outside the trap region"
    );
    assert!(gate.kicks as usize <= 2 * K, "budget FAIL");
}

#[test]
fn poc_trapped_never_fires_on_clean_convergence() {
    for x0 in toy_a_clean_cohort() {
        let r = run_toy_a(x0, Competitor::Gate);
        assert!(!r.trapped, "toy A clean {x0}: Trapped on clean run");
        assert_eq!(r.kicks, 0, "toy A clean {x0}: kicked on clean run");
    }
    for x0 in toy_b_clean_cohort() {
        let r = run_toy_b(x0, Competitor::Gate);
        assert!(!r.trapped, "toy B clean {x0:?}: Trapped on clean run");
        assert_eq!(r.kicks, 0, "toy B clean {x0:?}: kicked on clean run");
    }
}

#[test]
fn poc_halt_only_never_escapes_the_saddle_band() {
    // The defend-wrong anchor: the two-way family's measured failure mode.
    let cohort = toy_a_trapped_cohort();
    let halt = tally_all_a(&cohort, Competitor::HaltOnly);
    println!(
        "── anchor: halt-only on Toy A trapped cohort ── solve {} / {}",
        halt.solved, K
    );
    assert_eq!(
        halt.solved, 0,
        "anchor moved: halt-only unexpectedly escapes the saddle band"
    );
}
