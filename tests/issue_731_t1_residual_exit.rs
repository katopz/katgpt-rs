#![cfg(all(feature = "lt2_looped", feature = "cadence_gate"))]
//! Issue 731 T1 — the residual-gated early exit, e2e through the production
//! loop path (EqR action item 7.2, arXiv:2605.21488 Equilibrium Reasoners).
//!
//! The probe's kernel content (the d_min floor, the L = 3 magnitude window,
//! the shape arm via `ConvergenceCadence`, the non-finite policy) is pinned
//! by unit tests inside `crates/katgpt-core/src/convergence_cadence.rs`.
//! This bench proves the COMPOSED behavior on the T1/407 fixture (micro +
//! seed-42 + InterLoopNorm):
//!
//! 1. **G1 (bit-identity, probe present but never firing)** — `Some(probe)`
//!    with an unreachable `d_min` is bit-identical to `None` across all 27
//!    micro-vocab tokens: feeding the probe must not perturb the loop.
//! 2. **Fired-exit ≡ elastic equivalence (bit-level)** — a probe that fires
//!    at completed iteration `k` produces logits bit-identical to
//!    `elastic_loop_override = Some(k)` on the same token (the Issue-698-T4
//!    halt ≡ elastic equivalence, re-proven for the residual exit): the exit
//!    breaks the same loop at the same iteration, nothing else differs.
//! 3. **The d_min floor holds through the real path** — a probe with a large
//!    `tau` (magnitude arm always satisfied once the window fills) cannot
//!    fire before its floor; `fired_at_iteration()` must equal the floor.
//!
//! # Run
//!
//! ```bash
//! cargo test --features cadence_gate --test issue_731_t1_residual_exit -- --nocapture
//! ```

use katgpt_core::convergence_cadence::LoopResidualExit;
use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{Config, HlaMode, HybridPattern, LoopMode, ResidualGate, Rng, SdpaOutputGate};

/// Loop count of the baseline reference (large enough that an early exit
/// visibly cuts work).
const R_REF: usize = 32;

/// All micro-vocab tokens as prompts (Config::micro() vocab = 27).
const N_PROMPTS: usize = 27;

/// Fixture seed (matches the 407 / T1 / Issue-698 convention).
const SEED: u64 = 42;

fn make_config() -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: R_REF };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = HlaMode::Ahla;
    config
}

fn make_fixture(config: &Config) -> (TransformerWeights, ResidualGate, SdpaOutputGate) {
    let mut rng = Rng::new(SEED);
    let weights = TransformerWeights::new(config, &mut rng);
    let residual_gate = ResidualGate::new(R_REF, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    (weights, residual_gate, sdpa_gate)
}

#[allow(clippy::too_many_arguments)]
fn run(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    token: usize,
    elastic: Option<usize>,
    probe: Option<&mut LoopResidualExit>,
) -> Vec<f32> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ahla_cache = MultiLayerAhlaCache::new(config);
    let logits = forward_looped(
        &mut ctx,
        weights,
        &mut cache,
        &mut ahla_cache,
        token,
        0,
        config,
        residual_gate,
        sdpa_gate,
        None,
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        elastic,
        #[cfg(feature = "gain_cost_halt")]
        None,
        None, // Issue 717: deep_run — None = bit-identical baseline
        probe, // Issue 731 T1: the residual-exit probe (cadence_gate builds)
    );
    logits.to_vec()
}

/// G1 — a present-but-never-firing probe is bit-identical to `None` on every
/// token: observing the loop must not perturb it.
#[test]
fn g1_probe_present_never_firing_is_bit_identical_to_none() {
    let config = make_config();
    let (weights, residual_gate, sdpa_gate) = make_fixture(&config);

    for token in 0..N_PROMPTS {
        let baseline = run(
            &config,
            &weights,
            &residual_gate,
            &sdpa_gate,
            token,
            None,
            None,
        );
        // Unreachable floor: `seen + 1 < usize::MAX` holds for any real run,
        // so the probe can never fire — but it IS fed every iteration.
        let mut probe = LoopResidualExit::new(1.0, usize::MAX);
        let with_probe = run(
            &config,
            &weights,
            &residual_gate,
            &sdpa_gate,
            token,
            None,
            Some(&mut probe),
        );
        assert_eq!(
            baseline, with_probe,
            "token {token}: a fed-but-never-firing probe changed the logits"
        );
        assert!(probe.fired_at_iteration().is_none());
    }
}

/// Fired-exit ≡ elastic equivalence: a probe armed with `tau = +∞`-class
/// magnitude fires at the first moment its window allows (the L = 3 window
/// fills at the 3rd observation = completed iteration 4), and the resulting
/// logits are bit-identical to `elastic_loop_override = Some(4)`.
#[test]
fn fired_exit_is_bit_identical_to_elastic_override() {
    let config = make_config();
    let (weights, residual_gate, sdpa_gate) = make_fixture(&config);

    for token in 0..N_PROMPTS {
        let mut probe = LoopResidualExit::new(f32::INFINITY, 2);
        let exited = run(
            &config,
            &weights,
            &residual_gate,
            &sdpa_gate,
            token,
            None,
            Some(&mut probe),
        );
        assert_eq!(
            probe.fired_at_iteration(),
            Some(4),
            "token {token}: the magnitude arm must fire exactly when the L = 3 window fills (completed iteration 4)"
        );
        let elastic = run(
            &config,
            &weights,
            &residual_gate,
            &sdpa_gate,
            token,
            Some(4),
            None,
        );
        assert_eq!(
            exited, elastic,
            "token {token}: exit-at-4 must be bit-identical to elastic = 4"
        );
    }
}

/// The d_min floor holds through the real path: with `d_min = 8` the probe
/// cannot fire before completed iteration 8 even with `tau = +∞`, and the
/// exit ≡ elastic equivalence carries the floor through.
#[test]
fn d_min_floor_holds_and_exit_matches_elastic() {
    let config = make_config();
    let (weights, residual_gate, sdpa_gate) = make_fixture(&config);

    for token in 0..N_PROMPTS {
        let mut probe = LoopResidualExit::new(f32::INFINITY, 8);
        let exited = run(
            &config,
            &weights,
            &residual_gate,
            &sdpa_gate,
            token,
            None,
            Some(&mut probe),
        );
        assert_eq!(
            probe.fired_at_iteration(),
            Some(8),
            "token {token}: the floor is 8 — no earlier fire, no later one (magnitude arm is armed from the first full window)"
        );
        let elastic = run(
            &config,
            &weights,
            &residual_gate,
            &sdpa_gate,
            token,
            Some(8),
            None,
        );
        assert_eq!(
            exited, elastic,
            "token {token}: exit-at-8 must be bit-identical to elastic = 8"
        );
    }
}
