#![cfg(all(feature = "lt2_looped", feature = "loop_alignment_probe"))]
//! Issue 929 — loop-alignment probe wiring gates (Research 614, DiscoLoop).
//!
//! Proves the measurement PATH end to end on synthetic weights:
//! `forward_looped` → `LoopDeepRun` per-loop capture (`capture_states` +
//! `capture_logits`, the Issue-717/898 seam) → `katgpt_core::
//! loop_alignment_probe::probe_alignment` read per loop, plus the AUROC
//! pipeline and the fixture generator consumed the way the runner consumes
//! them. Zero forward-path coupling: the `None` deep-run baseline is covered
//! by the existing bit-identity gates (goat_108 family); this target only
//! asserts the instrument.
//!
//! Run: `cargo test --features loop_alignment_probe --test
//! issue_929_loop_alignment_probe -- --nocapture`

use katgpt_core::loop_alignment_probe::{
    auroc, bootstrap_auroc_ci, generate_two_hop_fixture, probe_alignment, FixtureSpec,
};
use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, LoopDeepRun, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{Config, HybridPattern, LoopMode, ResidualGate, Rng, SdpaOutputGate};

const LOOPS: usize = 4;

fn looped_config() -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: LOOPS };
    config.hybrid_pattern = HybridPattern::Uniform;
    config
}

fn probe_all_loops(
    config: &Config,
    weights: &TransformerWeights,
    token: u32,
) -> Vec<katgpt_core::loop_alignment_probe::AlignmentProbe> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ahla_cache = MultiLayerAhlaCache::new(config);
    let mut deep_run = LoopDeepRun::new(1); // every loop
    deep_run.capture_states = true;
    deep_run.capture_logits = true;
    let residual_gate = ResidualGate::new(LOOPS, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);

    let _logits = forward_looped(
        &mut ctx,
        weights,
        &mut cache,
        &mut ahla_cache,
        token as usize,
        0,
        config,
        &residual_gate,
        &sdpa_gate,
        #[cfg(feature = "sleep_consolidation")]
        None,
        #[cfg(feature = "sleep_consolidation")]
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        None, // elastic_loop_override (Issue 035) — natural loop count
        #[cfg(feature = "gain_cost_halt")]
        None,
        Some(&mut deep_run), // Issue 717: per-loop capture
        #[cfg(feature = "cadence_gate")]
        None,
        #[cfg(feature = "loop_guidance")]
        None,
        #[cfg(feature = "loop_guidance")]
        None,
    );

    let stats = &deep_run.stats;
    assert_eq!(
        stats.snapshots_taken, LOOPS,
        "snapshot_every=1 must observe every loop"
    );
    assert_eq!(stats.state_snapshots.len(), LOOPS);
    assert_eq!(stats.logit_snapshot_count(), LOOPS);

    let dim = config.n_embd;
    (0..LOOPS)
        .map(|k| {
            let state = &stats.state_snapshots[k];
            let logits = stats
                .logit_snapshot(k)
                .expect("logit count asserted == LOOPS");
            probe_alignment(state, logits, &weights.lm_head, dim)
        })
        .collect()
}

/// T1 — the wiring: every loop's H^(k) and readout flow through the probe,
/// the probes are finite/total, and the read is deterministic across runs
/// (same weights + seed → byte-identical probe vectors).
#[test]
fn probe_reads_loopdeep_capture_end_to_end() {
    let config = looped_config();
    let mut rng = Rng::new(929);
    let weights = TransformerWeights::new(&config, &mut rng);

    let first = probe_all_loops(&config, &weights, 3);
    let second = probe_all_loops(&config, &weights, 3);
    assert_eq!(first, second, "probe reads must be deterministic");

    for (k, p) in first.iter().enumerate() {
        assert!(p.argmax < config.vocab_size, "loop {k}: argmax in vocab");
        assert!(p.top1.is_finite() && p.top2.is_finite(), "loop {k}: finite top-2");
        assert!(
            p.cos_alignment >= -1.0 && p.cos_alignment <= 1.0,
            "loop {k}: cos in [-1, 1], got {}",
            p.cos_alignment
        );
        assert!(
            (p.margin - (p.top1 - p.top2)).abs() < 1e-5,
            "loop {k}: margin == top1 - top2"
        );
    }

    // Different token → a different read (the instrument is not constant).
    let other = probe_all_loops(&config, &weights, 7);
    assert_ne!(
        first, other,
        "probe must observe the state, not echo a constant"
    );
}

/// T2 — the readout leg: alignment + margin separate constructed correct /
/// incorrect loops with AUROC 1.0, and the bootstrap CI is deterministic
/// with a positive lower bound (the G1 shape the runner reports).
#[test]
fn auroc_pipeline_separates_constructed_alignments() {
    // Aligned+confident = correct; misaligned+weak = wrong.
    let scores: Vec<f32> = (0..20)
        .map(|i| {
            if i % 2 == 0 {
                0.9 // aligned correct
            } else {
                0.1 // misaligned wrong
            }
        })
        .collect();
    let labels: Vec<bool> = (0..20).map(|i| i % 2 == 0).collect();
    let point = auroc(&scores, &labels);
    assert!((point - 1.0).abs() < 1e-12);

    let ci = bootstrap_auroc_ci(&scores, &labels, 400, 929);
    assert_eq!(ci.iters, 400);
    assert!(ci.ci_lo > 0.95, "perfect separation: high LB, got {ci:?}");
    let again = bootstrap_auroc_ci(&scores, &labels, 400, 929);
    assert_eq!(ci, again, "bootstrap must be seed-deterministic");
}

/// T3 — the fixture leg: the generator produces deterministic, unique-answer
/// two-hop items with in-context gold facts, and the rendered prompts carry
/// the bridge (what the probe reads out at loop 1) and the answer cue.
#[test]
fn fixture_feeds_pretrained_probe_shape() {
    let spec = FixtureSpec {
        entities_per_pool: 40,
        relations: 8,
        out_degree: 3,
        facts_per_prompt: 8,
        two_hop_queries: 10,
        one_hop_queries: 5,
        seed: 4242,
    };
    let fixture = generate_two_hop_fixture(spec);
    assert_eq!(fixture.items.len(), 2 * (10 + 5));
    for item in fixture.items.iter().filter(|i| i.is_two_hop) {
        assert!(item.prompt.contains("Answer:"));
        assert!(item.prompt.contains(&item.bridge));
        assert_ne!(item.bridge, item.answer);
        // The composed question names the two roles in order.
        assert!(item.prompt.contains("Question: Who is the"));
    }
    // One-hop controls anchor the correctness classes the AUROC needs.
    assert!(fixture.items.iter().any(|i| !i.is_two_hop));
    assert!(fixture.unique_compositions_per_pool > 0);
}
