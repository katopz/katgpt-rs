//! Plan 617 T1.5 — the LoopCD guidance zero-alloc pin (G4 house pattern).
//!
//! Own test target (its own process) so the process-wide counting allocator
//! observes nothing but this test's work — the
//! `probe_guidance_alloc_gate` precedent (a multi-test target cannot hold
//! this pin: sibling tests run on parallel threads and bump the same
//! process-wide counter).
//!
//! Contract: a GUIDED steady-state decode step allocates exactly what the
//! UNGUIDED step allocates (guidance ADDS zero — the scratch is pre-sized
//! at `LoopGuidance::new`, the object's one allocation, and only
//! `resize`d — a no-op at capacity — inside the forward).
//!
//! Run: `cargo test --release --features loop_guidance,lt2_looped --test
//! loop_guidance_alloc_gate`

#![cfg(all(feature = "loop_guidance", feature = "lt2_looped"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::loop_guidance::{
    GuidanceMode, LoopGuidance, LoopGuidanceConfig, SettleExit, SettleExitConfig,
};
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{Config, HybridPattern, LoopMode, ResidualGate, Rng, SdpaOutputGate};

static ALLOCS: AtomicU64 = AtomicU64::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn make_config(loop_count: usize) -> Config {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count };
    config.hybrid_pattern = HybridPattern::Uniform;
    config
}

fn run_once_with(
    config: &Config,
    weights: &TransformerWeights,
    pos: usize,
    guidance: Option<&mut LoopGuidance>,
) -> Vec<f32> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ahla_cache = MultiLayerAhlaCache::new(config);
    let residual_gate = ResidualGate::new(4, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    let logits = forward_looped(
        &mut ctx,
        weights,
        &mut cache,
        &mut ahla_cache,
        0,
        pos,
        config,
        &residual_gate,
        &sdpa_gate,
        None,
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        None,
        #[cfg(feature = "gain_cost_halt")]
        None,
        None, // deep_run
        #[cfg(feature = "cadence_gate")]
        None,
        #[cfg(feature = "loop_guidance")]
        guidance,
        #[cfg(feature = "loop_guidance")]
        None,
    );
    logits.to_vec()
}

#[test]
fn steady_state_decode_adds_zero_allocations() {
    // ONE test fn, sequential arms: this target's pin holds only in a single
    // thread (the process-wide counter sees sibling tests on parallel test
    // threads — the reason this is its own target).
    let config = make_config(4);
    let mut rng = Rng::new(77);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Warmup: scratch resize-to-capacity no-ops settle here (the first
    // armed step also warms any lazy machinery on the armed paths).
    let _ = run_once_with(&config, &weights, 0, None);

    let before = ALLOCS.load(Ordering::Relaxed);
    let _ = run_once_with(&config, &weights, 1, None);
    let unguided = ALLOCS.load(Ordering::Relaxed) - before;

    // (a) guidance armed (T1.5 contract).
    let gcfg = LoopGuidanceConfig {
        mode: GuidanceMode::Logits,
        omega: 0.3,
        omega_max: 0.5,
        adaptive: true,
        ref_loop: 1,
    };
    let mut g = LoopGuidance::new(gcfg, config.vocab_size, config.n_embd);
    let _ = run_once_with(&config, &weights, 2, Some(&mut g)); // warm
    let before = ALLOCS.load(Ordering::Relaxed);
    let _ = run_once_with(&config, &weights, 2, Some(&mut g));
    let guided = ALLOCS.load(Ordering::Relaxed) - before;
    assert_eq!(
        guided, unguided,
        "guided steady-state decode must add ZERO allocations over unguided \
         (unguided={unguided}, guided={guided})"
    );

    // (b) settle exit armed, Full readout (T2.1 contract).
    let scfg = SettleExitConfig {
        enabled: true,
        d_min: 1,
        settle_patience: 16, // never fires — every iteration evaluates
        readout: katgpt_rs::transformer::loop_guidance::SettleReadout::Full,
    };
    let mut s = SettleExit::new(scfg, config.vocab_size, config.n_embd);
    let _ = run_once_with_settle(&config, &weights, 2, Some(&mut s)); // warm
    let before = ALLOCS.load(Ordering::Relaxed);
    let _ = run_once_with_settle(&config, &weights, 2, Some(&mut s));
    let settled = ALLOCS.load(Ordering::Relaxed) - before;
    assert_eq!(
        settled, unguided,
        "full-readout settle exit must add ZERO allocations over unguided \
         (unguided={unguided}, settled={settled})"
    );

    // (c) settle exit armed, CandidateSet readout (the refresh + bounded
    // scoring path — the top-m selection must reuse the pre-sized buffer).
    let scfg = SettleExitConfig {
        enabled: true,
        d_min: 1,
        settle_patience: 16,
        readout: katgpt_rs::transformer::loop_guidance::SettleReadout::CandidateSet {
            top_m: 4,
            refresh_every: 2,
        },
    };
    let mut s = SettleExit::new(scfg, config.vocab_size, config.n_embd);
    let _ = run_once_with_settle(&config, &weights, 2, Some(&mut s)); // warm
    let before = ALLOCS.load(Ordering::Relaxed);
    let _ = run_once_with_settle(&config, &weights, 2, Some(&mut s));
    let settled = ALLOCS.load(Ordering::Relaxed) - before;
    assert_eq!(
        settled, unguided,
        "candidate-set settle exit must add ZERO allocations over unguided \
         (unguided={unguided}, settled={settled})"
    );
}

fn run_once_with_settle(
    config: &Config,
    weights: &TransformerWeights,
    pos: usize,
    settle: Option<&mut SettleExit>,
) -> Vec<f32> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ahla_cache = MultiLayerAhlaCache::new(config);
    let residual_gate = ResidualGate::new(4, config.n_embd);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    let logits = forward_looped(
        &mut ctx,
        weights,
        &mut cache,
        &mut ahla_cache,
        0,
        pos,
        config,
        &residual_gate,
        &sdpa_gate,
        None,
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        None,
        #[cfg(feature = "gain_cost_halt")]
        None,
        None, // deep_run
        #[cfg(feature = "cadence_gate")]
        None,
        #[cfg(feature = "loop_guidance")]
        None,
        #[cfg(feature = "loop_guidance")]
        settle,
    );
    logits.to_vec()
}
