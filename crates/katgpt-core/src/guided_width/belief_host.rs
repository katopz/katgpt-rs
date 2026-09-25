//! Belief-host adapter: guided width rollouts over the incumbent
//! `katgpt-sense` `ReconstructionState::evolve_belief` (the deterministic
//! single `[f32; 8]` trajectory Research 590 routes GRAM to).
//!
//! The deterministic step IS the incumbent: each branch step loads the
//! branch belief into the state, calls `evolve_belief()` unchanged, and reads
//! it back. The accumulated evidence is not touched by `evolve_belief`, so
//! every branch refines against the same evidence. The selected belief is
//! written back into the state.
//!
//! Kill switch (G3): with `n_branches ≤ 1` or `σ_max = 0` the driver runs
//! `K` `evolve_belief()` calls on the state's own belief — the incumbent
//! loop, bit-identical (pinned by test).
//!
//! ⚠ Consumer note (riir-ai Issue 1008): under `temporal_deriv` the surprise
//! kernel observes every branch step (N·K observations per decision instead
//! of K). A consumer that reads the surprise channel must account for that
//! (or observe only the selected belief afterwards).

use crate::sense::reconstruction::ReconstructionState;

use super::perturb::Perturbation;
use super::rollout::guided_width_rollouts;
use super::types::{GuidedWidthConfig, GuidedWidthScratch, Hooks, RolloutReport};

/// Belief dimension of the host.
pub const BELIEF_DIM: usize = 8;

/// Run one guided-width deliberation over `state`'s belief and write the
/// selected hypothesis back into it. `scratch` must be built with
/// `dim == BELIEF_DIM`. Zero-allocation.
pub fn guided_evolve_belief<P: Perturbation>(
    state: &mut ReconstructionState,
    cfg: &GuidedWidthConfig,
    hooks: Hooks<'_>,
    perturb: &mut P,
    scratch: &mut GuidedWidthScratch,
) -> RolloutReport {
    let h0 = *state.belief();
    let mut out = [0.0f32; BELIEF_DIM];
    let mut step = |h: &mut [f32]| {
        state.belief_mut().copy_from_slice(h);
        state.evolve_belief();
        h.copy_from_slice(state.belief());
    };
    let report = guided_width_rollouts(&h0, cfg, hooks, perturb, &mut step, scratch, &mut out);
    *state.belief_mut() = out;
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guided_width::{StagnationGate, Transversal};
    use crate::sense::reconstruction::ReconstructionState;

    fn evidence_state() -> ReconstructionState {
        let mut s = ReconstructionState::new([0.1, -0.2, 0.05, 0.0, 0.3, -0.1, 0.0, 0.2]);
        let acts = [0.9f32, 0.1, 0.4, 0.0, 0.7, 0.2];
        s.accumulate(&[true; 6], &acts);
        s
    }

    #[test]
    fn kill_switch_is_the_incumbent_evolve_belief_loop() {
        let k = 12;
        let mut reference = evidence_state();
        for _ in 0..k {
            reference.evolve_belief();
        }
        for (n, sigma) in [(1usize, 0.25f32), (8, 0.0)] {
            let mut s = evidence_state();
            let cfg = GuidedWidthConfig {
                n_branches: n,
                k_steps: k,
                gate: StagnationGate {
                    sigma_max: sigma,
                    ..StagnationGate::DEFAULT
                },
                ..GuidedWidthConfig::DEFAULT
            };
            let mut scratch = GuidedWidthScratch::with_capacity(8, BELIEF_DIM);
            let rep = guided_evolve_belief(
                &mut s,
                &cfg,
                Hooks::default(),
                &mut Transversal::default(),
                &mut scratch,
            );
            assert!(rep.incumbent);
            assert_eq!(
                s.belief().map(f32::to_bits),
                reference.belief().map(f32::to_bits),
                "n={n} σ={sigma}"
            );
        }
    }

    #[test]
    fn width_on_the_belief_host_is_deterministic_and_bounded() {
        let cfg = GuidedWidthConfig {
            n_branches: 8,
            k_steps: 12,
            seed: [9u8; 32],
            ..GuidedWidthConfig::DEFAULT
        };
        let mut a = evidence_state();
        let mut b = evidence_state();
        let mut sa = GuidedWidthScratch::with_capacity(8, BELIEF_DIM);
        let mut sb = GuidedWidthScratch::with_capacity(8, BELIEF_DIM);
        let ra = guided_evolve_belief(&mut a, &cfg, Hooks::default(), &mut Transversal::default(), &mut sa);
        let rb = guided_evolve_belief(&mut b, &cfg, Hooks::default(), &mut Transversal::default(), &mut sb);
        assert_eq!(ra, rb);
        assert_eq!(a.belief().map(f32::to_bits), b.belief().map(f32::to_bits));
        assert_eq!(ra.step_evals, 8 * 12);
        assert!(a.belief().iter().all(|x| x.is_finite() && x.abs() <= 1.0));
    }
}
