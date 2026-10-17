//! Plan 617 T1.4/T1.5 — LoopCD guidance unit gates + the zero-alloc pin.
//!
//! Covers:
//! - (a) `omega = 0` byte-identical to unguided, BOTH modes (the G1 flag-off
//!   contract — the ω=0 posture is a structural no-op: no capture, no
//!   combine);
//! - (b) the adaptive gate's concentration pin: max-margin → ω 0,
//!   zero-margin → ω_max (Eq 4);
//! - (c) determinism (same inputs → same logits);
//! - (d) head-pass-count proxies: logit mode writes the z_k scratch (the
//!   ONE extra pass); hidden mode never does (the one-head-pass law);
//! - (e) negative-ω config refused at validation;
//! - the top-two margin's degenerate-row fail-closed posture;
//! - T1.5: steady-state guided decode allocates ZERO bytes (counting
//!   allocator, this target's own process).
//!
//! Run: `cargo test --features loop_guidance,lt2_looped --test
//! loop_guidance_gates`

#![cfg(feature = "loop_guidance")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::loop_guidance::{
    GuidanceMode, LoopGuidance, LoopGuidanceConfig, adaptive_omega, top_two_margin,
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

/// One guided decode step; returns owned logits. `guidance` is constructed
/// by the caller so the alloc gate can build it OUTSIDE the counted region.
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
    #[cfg(feature = "weight_shared_advantage_gate")]
    let gate: Option<&mut katgpt_rs::pruners::self_advantage::AdvantageMarginGate> = None;
    #[cfg(not(feature = "weight_shared_advantage_gate"))]
    let _gate = ();
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
        gate,
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

fn baseline(config: &Config, weights: &TransformerWeights, pos: usize) -> Vec<f32> {
    run_once_with(config, weights, pos, None)
}

// ── (a) ω = 0 byte-identity, both modes ─────────────────────────────────

#[test]
fn omega_zero_logit_mode_is_byte_identical() {
    let config = make_config(3);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let base = baseline(&config, &weights, 0);

    let cfg = LoopGuidanceConfig {
        mode: GuidanceMode::Logits,
        omega: 0.0,
        omega_max: 0.0,
        adaptive: false,
        ref_loop: 1,
    };
    let mut g = LoopGuidance::new(cfg, config.vocab_size, config.n_embd);
    let guided = run_once_with(&config, &weights, 0, Some(&mut g));
    assert_eq!(guided, base, "ω=0 logit mode must be byte-identical");
    assert!(!g.guidance_active(), "ω=0 posture must not even capture");
}

#[test]
fn omega_zero_hidden_mode_is_byte_identical() {
    let config = make_config(3);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let base = baseline(&config, &weights, 0);

    let cfg = LoopGuidanceConfig {
        mode: GuidanceMode::Hidden,
        omega: 0.0,
        omega_max: 0.0,
        adaptive: false,
        ref_loop: 1,
    };
    let mut g = LoopGuidance::new(cfg, config.vocab_size, config.n_embd);
    let guided = run_once_with(&config, &weights, 0, Some(&mut g));
    assert_eq!(guided, base, "ω=0 hidden mode must be byte-identical");
}

// ── (b) adaptive concentration pin (Eq 4) ────────────────────────────────

#[test]
fn adaptive_gate_concentration_pin() {
    assert_eq!(adaptive_omega(0.5, 1.0), 0.0, "max margin → ω withheld");
    assert_eq!(adaptive_omega(0.5, 0.0), 0.5, "zero margin → full ω_max");
    assert!((adaptive_omega(0.6, 0.25) - 0.45).abs() < 1e-6);
}

#[test]
fn top_two_margin_concentrated_row() {
    // One dominant logit: margin → 1.
    let mut row = vec![-3.0f32; 32];
    row[7] = 12.0;
    assert!(top_two_margin(&row) > 0.99);
}

#[test]
fn top_two_margin_contested_row() {
    // Two near-equal leaders: margin → small.
    let mut row = vec![-3.0f32; 32];
    row[3] = 5.0;
    row[19] = 5.0 + 1e-4;
    let m = top_two_margin(&row);
    assert!(m < 0.01, "contested row margin {m} must be near 0");
}

#[test]
fn top_two_margin_degenerate_rows_fail_closed() {
    assert_eq!(top_two_margin(&[]), 0.0);
    assert_eq!(top_two_margin(&[1.0]), 0.0);
    // All-equal row: p1 == p2 exactly → margin 0.
    assert_eq!(top_two_margin(&[2.0f32; 16]), 0.0);
    // Non-finite value → fail closed (settled, ω withheld).
    let mut row = vec![1.0f32; 8];
    row[2] = f32::NAN;
    assert_eq!(
        top_two_margin(&row),
        1.0,
        "NaN row must read settled (ω withheld)"
    );
}

// ── (c) determinism ─────────────────────────────────────────────────────

#[test]
fn guided_decode_is_deterministic() {
    let config = make_config(3);
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let cfg = LoopGuidanceConfig {
        mode: GuidanceMode::Logits,
        omega: 0.3,
        omega_max: 0.5,
        adaptive: true,
        ref_loop: 1,
    };
    let mut g1 = LoopGuidance::new(cfg, config.vocab_size, config.n_embd);
    let a = run_once_with(&config, &weights, 0, Some(&mut g1));
    let mut g2 = LoopGuidance::new(cfg, config.vocab_size, config.n_embd);
    let b = run_once_with(&config, &weights, 0, Some(&mut g2));
    assert_eq!(a, b, "same inputs → same guided logits");
}

// ── (d) head-pass-count proxies ─────────────────────────────────────────
//
// The pass COUNT is a code-structure property; the observable proxy is which
// scratch the capture wrote: logit mode fills scratch (the ONE extra head
// pass ran), hidden mode fills only the hidden slot (no extra pass). A
// guided ω>0 run whose guided logits DIFFER from baseline additionally
// proves the combine consumed the captured reference.

#[test]
fn logit_mode_capture_writes_logit_scratch() {
    let config = make_config(3);
    let mut rng = Rng::new(7);
    let weights = TransformerWeights::new(&config, &mut rng);
    let cfg = LoopGuidanceConfig {
        mode: GuidanceMode::Logits,
        omega: 0.4,
        omega_max: 0.5,
        adaptive: false,
        ref_loop: 1,
    };
    let mut g = LoopGuidance::new(cfg, config.vocab_size, config.n_embd);
    let guided = run_once_with(&config, &weights, 0, Some(&mut g));
    assert!(g.guidance_active(), "logit capture must fire at ref_loop=1");
    assert_ne!(
        guided,
        baseline(&config, &weights, 0),
        "ω=0.4 must move the decode"
    );
}

#[test]
fn hidden_mode_changes_state_not_head_count() {
    let config = make_config(3);
    let mut rng = Rng::new(7);
    let weights = TransformerWeights::new(&config, &mut rng);
    let cfg = LoopGuidanceConfig {
        mode: GuidanceMode::Hidden,
        omega: 0.4,
        omega_max: 0.5,
        adaptive: false,
        ref_loop: 1,
    };
    let mut g = LoopGuidance::new(cfg, config.vocab_size, config.n_embd);
    let guided = run_once_with(&config, &weights, 0, Some(&mut g));
    assert!(g.guidance_active());
    assert_ne!(
        guided,
        baseline(&config, &weights, 0),
        "hidden contrast must move the decode"
    );
    // Hidden mode carries the OUTPUT margin for the next step (the
    // one-head-pass law: no in-step z_R exists). This is the pinned
    // deviation — a future session changing it must rewrite this pin.
    assert!(
        g.last_margin().is_some(),
        "hidden mode must record the output margin"
    );
}

#[test]
fn ref_loop_beyond_loop_count_never_guides() {
    let config = make_config(2);
    let mut rng = Rng::new(7);
    let weights = TransformerWeights::new(&config, &mut rng);
    let cfg = LoopGuidanceConfig {
        mode: GuidanceMode::Logits,
        omega: 0.4,
        omega_max: 0.5,
        adaptive: false,
        ref_loop: 5, // loop_count = 2 → capture never fires
    };
    let mut g = LoopGuidance::new(cfg, config.vocab_size, config.n_embd);
    let guided = run_once_with(&config, &weights, 0, Some(&mut g));
    assert!(!g.guidance_active(), "no capture → no guidance");
    assert_eq!(
        guided,
        baseline(&config, &weights, 0),
        "no-reference step is a no-op"
    );
}

// ── (e) validation ──────────────────────────────────────────────────────

#[test]
fn negative_omega_refused() {
    let cfg = LoopGuidanceConfig {
        mode: GuidanceMode::Logits,
        omega: -0.1,
        omega_max: 0.0,
        adaptive: false,
        ref_loop: 1,
    };
    assert!(
        cfg.validate().is_err(),
        "negative ω universally degrades (paper §G.1)"
    );
}

#[test]
fn omega_max_below_omega_refused() {
    let cfg = LoopGuidanceConfig {
        mode: GuidanceMode::Logits,
        omega: 0.3,
        omega_max: 0.2,
        adaptive: false,
        ref_loop: 1,
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn zero_ref_loop_refused() {
    let cfg = LoopGuidanceConfig {
        mode: GuidanceMode::Logits,
        omega: 0.3,
        omega_max: 0.5,
        adaptive: false,
        ref_loop: 0,
    };
    assert!(cfg.validate().is_err());
}

#[test]
#[should_panic(expected = "invalid config")]
fn constructor_panics_on_invalid_config() {
    let cfg = LoopGuidanceConfig {
        mode: GuidanceMode::Logits,
        omega: -1.0,
        omega_max: 0.0,
        adaptive: false,
        ref_loop: 1,
    };
    let _ = LoopGuidance::new(cfg, 16, 8);
}

// ── T1.5 — the zero-alloc pin ───────────────────────────────────────────
