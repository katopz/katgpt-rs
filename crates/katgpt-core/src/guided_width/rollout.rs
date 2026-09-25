//! The guided width rollout driver: N parallel hypotheses of K steps over a
//! caller-supplied deterministic refinement step, perturbed by a pluggable
//! ε arm, optionally guided by a frozen direction table, trap-handled by the
//! shipped `saddle_escape` detector + kick, and selected decode-free.
//!
//! Per branch `b` and step `t` (branch 0 is the deterministic branch — it is
//! never perturbed, kicked or killed, so the width set always contains the
//! incumbent trajectory):
//!
//! ```text
//! u      = step(h)                       deterministic proposal (caller)
//! Δ      = u − h ;  r_b = ‖Δ‖            convergence residual
//! w_b    = r_b < stuck_tol ? w_b+1 : 0    stagnation count (T2)
//! if b > 0 and t < K−1:                   (the LAST step is noise-free, so the
//!     σ = gate.sigma(w_b)                  returned state is a proposal output)
//!     h = u + P(v; Δ)·σ                   T1 arm
//!         + κ·σ·s_b·d_{j_b}               T5 μ≠0 term (table present)
//!     T6: FlipDetector(belief_key(h)); trapped ⇒ kick (apply_kick) while
//!         budget remains, else KILL and re-spawn a fresh hypothesis in the
//!         slot for the remaining steps (freed budget → width)
//! ```
//!
//! Init (T4): branch 0 = `h0`; branch `b ≥ 1` = `h0 + σ_max·spread·Sobol_b`
//! plus the μ≠0 commitment `κ·σ_max·s_b·d_{j_b}` when a table is present.
//! After K steps every branch is scored by [`super::latent_value_into`] and
//! the best is written to `out`.
//!
//! Compute: exactly `N·K` step calls when nothing is killed without
//! respawn; the equal-compute depth baseline is ONE trajectory of `N·K`.
//! Branches share nothing until selection, so the dependency depth is `K`
//! (the O(K) parallel-latency property; the serial cost is `N·K` steps).

use crate::saddle_escape::{FlipDetector, apply_kick};

use super::init::{mix64, sobol_init_into};
use super::perturb::{Perturbation, add_guidance};
use super::score::{latent_value_into, select_best};
use super::types::{
    GuidedWidthConfig, GuidedWidthScratch, Guidance, Hooks, MAX_BRANCHES, MAX_DIRECTIONS,
    RolloutReport,
};
use crate::diversity::temp::blake3_noise_fill;

const SEED_TAG: &[u8] = b"katgpt.guided_width.rollouts.v1";

/// Pre-decode belief-state key for the trap detector (T6 new claim (i)):
/// the sign pattern of `h − center`, one bit per coordinate, folded into 64
/// bits by rotation for `d > 64`. No decoding, no reward — a latent
/// observable of which orthant the hypothesis sits in.
#[inline]
pub fn belief_key(h: &[f32], center: &[f32]) -> u64 {
    let mut key = 0u64;
    for (i, (x, c)) in h.iter().zip(center).enumerate() {
        if x - c > 0.0 {
            key ^= 1u64.rotate_left((i % 64) as u32 + (i / 64) as u32);
        }
    }
    key
}

#[inline]
fn norm(v: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// Assign the committed direction `(j, sign)` of branch `b ≥ 1` from the
/// posterior ranking: branch `b` takes the `((b−1) mod r)`-th ranked
/// direction; a direction whose bias clears `bias_tau` keeps its learned
/// sign, otherwise successive laps alternate `+, −`.
fn assign_direction(g: &Guidance<'_>, order: &[u16], b: usize) -> (u16, i8) {
    let r = order.len();
    let i = b - 1;
    let j = order[i % r];
    let lap = i / r;
    let sign = if g.table.bias(j as usize) >= g.bias_tau || lap.is_multiple_of(2) {
        1
    } else {
        -1
    };
    (j, sign)
}

fn kick_seed(h: &[f32], branch: usize, step: usize, kick: u8, base: u64) -> [u8; 32] {
    let mut hs = blake3::Hasher::new();
    for x in h {
        hs.update(&x.to_bits().to_le_bytes());
    }
    hs.update(&(branch as u64).to_le_bytes());
    hs.update(&(step as u64).to_le_bytes());
    hs.update(&[kick]);
    hs.update(&base.to_le_bytes());
    *hs.finalize().as_bytes()
}

/// Run one guided-width decision from `h0`, writing the selected hypothesis
/// into `out` (len `h0.len()`).
///
/// - `step` is the host's deterministic refinement (`h ← f(h)` in place);
///   for the belief host it is `evolve_belief` (see `belief_host`).
/// - `perturb` is the ε arm ([`super::Transversal`] for dense latents,
///   `hodge_arm::MassConserving` for cochain fields).
/// - `hooks` carries the optional table, frozen scorer direction and trap
///   probe.
///
/// **Kill switch:** `cfg.is_kill_switch()` (N ≤ 1 or σ_max = 0) runs
/// `out = h0; K × step(out)` verbatim and returns `incumbent = true`.
///
/// Zero-allocation for `h0.len() == scratch.dim()` and
/// `cfg.n_branches ≤ scratch.n_max()` (N is clamped to the capacity).
///
/// # Panics
///
/// If `h0.len() != out.len()` or `h0.len() != scratch.dim()`.
pub fn guided_width_rollouts<F, P>(
    h0: &[f32],
    cfg: &GuidedWidthConfig,
    hooks: Hooks<'_>,
    perturb: &mut P,
    step: &mut F,
    scratch: &mut GuidedWidthScratch,
    out: &mut [f32],
) -> RolloutReport
where
    F: FnMut(&mut [f32]),
    P: Perturbation,
{
    let d = h0.len();
    assert_eq!(out.len(), d, "guided_width_rollouts: out.len() != h0.len()");
    assert_eq!(scratch.d, d, "guided_width_rollouts: scratch dim mismatch");
    let k = cfg.k_steps;

    // ── Kill switch: the incumbent path, verbatim ─────────────────────
    if cfg.is_kill_switch() {
        out.copy_from_slice(h0);
        for _ in 0..k {
            step(out);
        }
        scratch.last_n = 0;
        return RolloutReport {
            best: 0,
            best_value: f32::NAN,
            direction: None,
            step_evals: k as u32,
            kicks: 0,
            respawns: 0,
            killed: 0,
            incumbent: true,
        };
    }

    let n = cfg.n_branches.min(scratch.n_max).min(MAX_BRANCHES);
    let sigma_max = cfg.gate.sigma_max_sanitised();
    let kappa = if cfg.guidance_gain.is_finite() {
        cfg.guidance_gain
    } else {
        0.0
    };
    let spread = sigma_max * cfg.init_spread;
    let Hooks {
        guidance,
        frozen_direction,
        mut probe,
    } = hooks;

    // Deterministic base seed: BLAKE3(tag ‖ cfg.seed).
    let base = {
        let mut hs = blake3::Hasher::new();
        hs.update(SEED_TAG);
        hs.update(&cfg.seed);
        let b = hs.finalize();
        u64::from_le_bytes(b.as_bytes()[..8].try_into().unwrap_or([0; 8]))
    };

    // ── Guidance ranking (table present, usable, admitted by the arm) ──
    let mut order = [0u16; MAX_DIRECTIONS];
    let n_dirs = match &guidance {
        Some(g) if perturb.admits_guidance() && g.table.dim() == d && !g.table.is_empty() => {
            g.posterior
                .rank_into(g.epsilon, &mut order[..g.table.len()])
        }
        _ => 0,
    };
    let guidance = if n_dirs > 0 { guidance } else { None };

    // ── Init (T4 + T5 commitment) ─────────────────────────────────────
    {
        let GuidedWidthScratch { states, sobol, .. } = scratch;
        sobol_init_into(h0, n, spread, mix64(base ^ 0x5B0B), sobol, &mut states[..n * d]);
    }
    for b in 0..n {
        scratch.w_stuck[b] = 0;
        scratch.kicks[b] = 0;
        scratch.generation[b] = 0;
        scratch.alive[b] = true;
        scratch.residual[b] = 0.0;
        scratch.dir[b] = None;
        if let Some(t) = cfg.trap {
            scratch.flips[b] = FlipDetector::new(t.window);
        }
        if b > 0
            && let Some(g) = &guidance
        {
            let (j, s) = assign_direction(g, &order[..n_dirs], b);
            scratch.dir[b] = Some((j, s));
            add_guidance(
                &mut scratch.states[b * d..(b + 1) * d],
                g.table.direction(j as usize),
                s,
                kappa * sigma_max,
            );
        }
    }

    let mut report = RolloutReport {
        best: 0,
        best_value: f32::NAN,
        direction: None,
        step_evals: 0,
        kicks: 0,
        respawns: 0,
        killed: 0,
        incumbent: false,
    };

    // ── K steps × N branches ──────────────────────────────────────────
    for t in 0..k {
        let noisy_step = t + 1 < k;
        for b in 0..n {
            if !scratch.alive[b] {
                continue;
            }
            let GuidedWidthScratch {
                states,
                delta,
                noise,
                residual,
                w_stuck,
                dir,
                flips,
                kicks,
                generation,
                alive,
                ..
            } = scratch;
            let h = &mut states[b * d..(b + 1) * d];
            delta.copy_from_slice(h);
            step(h);
            report.step_evals += 1;
            for i in 0..d {
                delta[i] = h[i] - delta[i];
            }
            let r = norm(delta);
            residual[b] = r;
            w_stuck[b] = if r < cfg.gate.stuck_tol {
                w_stuck[b].saturating_add(1)
            } else {
                0
            };
            if b == 0 || !noisy_step {
                continue;
            }

            let sigma_t = cfg.gate.sigma(w_stuck[b]);
            let seed_bt = mix64(
                base ^ (b as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
                    ^ ((t as u64) << 20)
                    ^ ((generation[b] as u64) << 50),
            );
            perturb.perturb(h, delta, noise, seed_bt, sigma_t);
            if let (Some(g), Some((j, s))) = (&guidance, dir[b]) {
                add_guidance(h, g.table.direction(j as usize), s, kappa * sigma_t);
            }

            // ── T6: trap-kill-reallocate ─────────────────────────────
            let Some(tc) = cfg.trap else { continue };
            let rate = flips[b].observe(Some(belief_key(h, h0)));
            let Some(rate) = rate else { continue };
            if rate < tc.flip_tau {
                continue;
            }
            let probe_ok = match probe.as_mut() {
                None => true,
                Some(p) => p(delta) >= tc.probe_tau, // NaN never confirms
            };
            if !probe_ok {
                continue;
            }
            if kicks[b] < tc.kick_budget {
                let mut eps = tc.eps0;
                for _ in 0..kicks[b] {
                    eps *= tc.eps_decay;
                }
                if eps.is_finite() && eps > 0.0 {
                    kicks[b] += 1;
                    let ks = kick_seed(h, b, t, kicks[b], base);
                    apply_kick(h, ks, eps);
                    report.kicks = report.kicks.saturating_add(1);
                }
                flips[b] = FlipDetector::new(tc.window);
                continue;
            }
            // Kick budget spent: KILL. Re-spend the remaining budget on a
            // fresh hypothesis in this slot (needs ≥ 2 steps left to be
            // anything but a single noise-free step).
            if tc.respawn && t + 2 < k {
                generation[b] = generation[b].saturating_add(1);
                h.copy_from_slice(h0);
                let s = mix64(seed_bt ^ 0xE5A_4E5F);
                for (c, chunk) in h.chunks_mut(8).enumerate() {
                    let mut blk = [0.0f32; 8];
                    let blk = &mut blk[..chunk.len()];
                    blake3_noise_fill(mix64(s ^ c as u64), spread, blk);
                    for (x, &e) in chunk.iter_mut().zip(blk.iter()) {
                        *x += e;
                    }
                }
                if let (Some(g), Some((j, sg))) = (&guidance, dir[b]) {
                    add_guidance(h, g.table.direction(j as usize), sg, kappa * sigma_max);
                }
                w_stuck[b] = 0;
                kicks[b] = 0;
                flips[b] = FlipDetector::new(tc.window);
                report.respawns = report.respawns.saturating_add(1);
            } else {
                alive[b] = false;
                report.killed = report.killed.saturating_add(1);
            }
        }
    }

    // ── Decode-free selection (T3) ────────────────────────────────────
    {
        let GuidedWidthScratch {
            states,
            residual,
            values,
            alive,
            ..
        } = scratch;
        latent_value_into(
            &states[..n * d],
            n,
            h0,
            &residual[..n],
            &cfg.score,
            frozen_direction,
            &mut values[..n],
        );
        // A killed (stopped) branch never wins.
        for b in 0..n {
            if !alive[b] {
                values[b] = f32::NAN;
            }
        }
    }
    let best = select_best(&scratch.values[..n]);
    out.copy_from_slice(&scratch.states[best * d..(best + 1) * d]);
    scratch.last_n = n;
    report.best = best;
    report.best_value = scratch.values[best];
    report.direction = scratch.dir[best];
    report
}

#[cfg(test)]
mod tests {
    use super::super::perturb::Transversal;
    use super::super::table::{DirectionPosterior, DirectionTable};
    use super::super::types::{StagnationGate, TrapReallocConfig};
    use super::*;

    /// Double-well per coordinate: h ← h + η(h − h³) — deterministic, with
    /// a saddle at 0 and minima at ±1.
    fn double_well(h: &mut [f32]) {
        for x in h.iter_mut() {
            *x += 0.2 * (*x - *x * *x * *x);
        }
    }

    fn cfg(n: usize, k: usize, sigma: f32) -> GuidedWidthConfig {
        GuidedWidthConfig {
            n_branches: n,
            k_steps: k,
            gate: StagnationGate {
                sigma_max: sigma,
                ..StagnationGate::DEFAULT
            },
            seed: [3u8; 32],
            ..GuidedWidthConfig::DEFAULT
        }
    }

    fn incumbent(h0: &[f32], k: usize) -> Vec<f32> {
        let mut h = h0.to_vec();
        for _ in 0..k {
            double_well(&mut h);
        }
        h
    }

    #[test]
    fn kill_switch_n1_and_sigma0_are_the_incumbent_bit_identically() {
        let h0 = [0.05f32, -0.3, 0.0, 0.7, -0.01, 0.2, 0.4, -0.9];
        let want = incumbent(&h0, 16);
        for c in [cfg(1, 16, 0.25), cfg(8, 16, 0.0), cfg(8, 16, f32::NAN)] {
            let mut s = GuidedWidthScratch::with_capacity(8, 8);
            let mut out = [0.0f32; 8];
            let rep = guided_width_rollouts(
                &h0,
                &c,
                Hooks::default(),
                &mut Transversal::default(),
                &mut double_well,
                &mut s,
                &mut out,
            );
            assert!(rep.incumbent);
            assert_eq!(rep.step_evals, 16);
            assert_eq!(out.map(f32::to_bits).to_vec(), want.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn width_is_deterministic_and_spends_exactly_nk() {
        let h0 = [0.0f32; 8];
        let c = cfg(8, 16, 0.25);
        let mut s1 = GuidedWidthScratch::with_capacity(8, 8);
        let mut s2 = GuidedWidthScratch::with_capacity(8, 8);
        let (mut o1, mut o2) = ([0.0f32; 8], [0.0f32; 8]);
        let r1 = guided_width_rollouts(&h0, &c, Hooks::default(), &mut Transversal::default(), &mut double_well, &mut s1, &mut o1);
        let r2 = guided_width_rollouts(&h0, &c, Hooks::default(), &mut Transversal::default(), &mut double_well, &mut s2, &mut o2);
        assert_eq!(r1, r2);
        assert_eq!(o1.map(f32::to_bits), o2.map(f32::to_bits));
        assert_eq!(r1.step_evals, 8 * 16);
        assert!(!r1.incumbent);
        // The deterministic branch sits on the saddle; width leaves it.
        assert_eq!(s1.branch(0), &h0);
        assert!((1..8).any(|b| s1.branch(b).iter().any(|x| x.abs() > 0.5)));
    }

    #[test]
    fn table_absent_and_empty_table_are_bit_identical() {
        let h0 = [0.1f32, -0.1, 0.0, 0.0, 0.2, 0.0, -0.3, 0.05];
        let c = cfg(6, 12, 0.3);
        let run = |hooks: Hooks<'_>| {
            let mut s = GuidedWidthScratch::with_capacity(6, 8);
            let mut o = [0.0f32; 8];
            let r = guided_width_rollouts(&h0, &c, hooks, &mut Transversal::default(), &mut double_well, &mut s, &mut o);
            (r, o.map(f32::to_bits), s.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>())
        };
        let none = run(Hooks::default());
        let empty = DirectionTable::from_parts(8, vec![], vec![]).unwrap();
        let post = DirectionPosterior::new(0);
        let with_empty = run(Hooks {
            guidance: Some(Guidance { table: &empty, posterior: &post, epsilon: 0.05, bias_tau: 0.5 }),
            ..Hooks::default()
        });
        assert_eq!(none, with_empty);
        // A dimension-mismatched table is ignored the same way.
        let wrong = DirectionTable::from_parts(4, vec![1.0, 0.0, 0.0, 0.0], vec![1.0]).unwrap();
        let post1 = DirectionPosterior::new(1);
        let with_wrong = run(Hooks {
            guidance: Some(Guidance { table: &wrong, posterior: &post1, epsilon: 0.05, bias_tau: 0.5 }),
            ..Hooks::default()
        });
        assert_eq!(none, with_wrong);
    }

    #[test]
    fn a_table_changes_the_rollout_and_reports_its_direction() {
        let h0 = [0.0f32; 8];
        let c = cfg(4, 10, 0.3);
        let mut dirs = vec![0.0f32; 8];
        dirs[0] = 1.0;
        let tab = DirectionTable::from_parts(8, dirs, vec![1.0]).unwrap();
        let post = DirectionPosterior::new(1);
        let mut s = GuidedWidthScratch::with_capacity(4, 8);
        let mut o = [0.0f32; 8];
        let rep = guided_width_rollouts(
            &h0,
            &c,
            Hooks {
                guidance: Some(Guidance { table: &tab, posterior: &post, epsilon: 0.05, bias_tau: 0.5 }),
                ..Hooks::default()
            },
            &mut Transversal::default(),
            &mut double_well,
            &mut s,
            &mut o,
        );
        // Every guided branch is pushed +e0 (bias 1 ⇒ learned sign) and so
        // falls into the +1 well on coordinate 0.
        for b in 1..4 {
            assert_eq!(s.branch_direction(b), Some((0, 1)));
            assert!(s.branch(b)[0] > 0.5, "branch {b}: {:?}", s.branch(b));
        }
        if rep.best > 0 {
            assert_eq!(rep.direction, Some((0, 1)));
        }
    }

    #[test]
    fn trap_realloc_kicks_then_respawns_within_budget() {
        // A step that oscillates the sign of every coordinate: the belief
        // key flips every step — a trap by construction.
        let mut flip = |h: &mut [f32]| {
            for x in h.iter_mut() {
                *x = -*x;
            }
        };
        let h0 = [0.5f32; 8];
        let mut c = cfg(4, 24, 0.25);
        c.trap = Some(TrapReallocConfig { window: 2, ..TrapReallocConfig::DEFAULT });
        let mut s = GuidedWidthScratch::with_capacity(4, 8);
        let mut o = [0.0f32; 8];
        let rep = guided_width_rollouts(&h0, &c, Hooks::default(), &mut Transversal::default(), &mut flip, &mut s, &mut o);
        assert!(rep.kicks > 0, "{rep:?}");
        assert!(rep.respawns > 0, "{rep:?}");
        assert_eq!(rep.step_evals, 4 * 24, "respawn re-spends, never adds, budget");
        // Without respawn the trapped branches are stopped instead.
        c.trap = Some(TrapReallocConfig { window: 2, respawn: false, ..TrapReallocConfig::DEFAULT });
        let rep2 = guided_width_rollouts(&h0, &c, Hooks::default(), &mut Transversal::default(), &mut flip, &mut s, &mut o);
        assert!(rep2.killed > 0 && rep2.respawns == 0, "{rep2:?}");
        assert!(rep2.step_evals < 4 * 24);
        assert!(s.values()[1..].iter().any(|v| v.is_nan()));
    }

    #[test]
    fn a_nan_probe_never_confirms_a_trap() {
        let mut flip = |h: &mut [f32]| {
            for x in h.iter_mut() {
                *x = -*x;
            }
        };
        let h0 = [0.5f32; 8];
        let mut c = cfg(4, 24, 0.25);
        c.trap = Some(TrapReallocConfig { window: 2, ..TrapReallocConfig::DEFAULT });
        let mut nan_probe = |_: &[f32]| f32::NAN;
        let mut s = GuidedWidthScratch::with_capacity(4, 8);
        let mut o = [0.0f32; 8];
        let rep = guided_width_rollouts(
            &h0,
            &c,
            Hooks { probe: Some(&mut nan_probe), ..Hooks::default() },
            &mut Transversal::default(),
            &mut flip,
            &mut s,
            &mut o,
        );
        assert_eq!((rep.kicks, rep.respawns, rep.killed), (0, 0, 0));
    }

    #[test]
    fn belief_key_is_the_sign_pattern() {
        let c = [0.0f32; 4];
        assert_eq!(belief_key(&[1.0, -1.0, 1.0, -1.0], &c), 0b0101);
        assert_eq!(belief_key(&[f32::NAN; 4], &c), 0);
    }
}
