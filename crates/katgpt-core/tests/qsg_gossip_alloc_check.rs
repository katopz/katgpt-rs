//! Plan 589 — QSG zero-allocation (G4) GOAT gate.
//!
//! Step / run / reducers must allocate zero bytes: population, message
//! scratch, pair source, and uniforms are all caller-owned. Same
//! `counting_allocator` pattern as `group_invariance_probe_g4.rs` /
//! `karc_alloc_check.rs`.
#![cfg(feature = "qsg_gossip")]

use katgpt_core::qsg_gossip::{
    MessageMode, QsgConfig, UniformStream, qsg_disagreement_v, qsg_gossip_run_into,
    qsg_init_uniform_into, qsg_mean_into, qsg_polarization_u, qsg_uncertainty,
    uniform_ordered_pair,
};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

/// Minimal in-house RNG (splitmix64 — the house test pattern).
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 16_777_216.0
    }
}

#[test]
fn g4_qsg_step_run_and_reducers_allocate_zero() {
    // The canary FIRST (Issue 714): a dead counter passes every gate.
    assert_counter_is_live();

    let n = 256;
    let k = 8;
    let cfg = QsgConfig::new(n, k, 0.5, MessageMode::TopM(3)).unwrap();
    let steps = 1000;

    // Warmup + all scratch allocation OUTSIDE the measured window.
    let mut beliefs = vec![0.0f32; n * k];
    qsg_init_uniform_into(&mut beliefs, n, k);
    let mut uniforms = vec![0.0f32; (steps + 1) * (cfg.mode.draws() as usize)];
    let mut urng = SplitMix64::new(0xA110_C8ED);
    for u in uniforms.iter_mut() {
        *u = urng.uniform();
    }
    let mut pairs = Vec::with_capacity(steps + 1);
    for _ in 0..(steps + 1) {
        pairs.push(uniform_ordered_pair(n, urng.uniform(), urng.uniform()));
    }
    let mut stream = UniformStream::new(&uniforms);
    let mut pair_iter = pairs.iter();
    let mut next_pair = move || *pair_iter.next().expect("pair budget exhausted");
    let mut mean = vec![0.0f32; k];

    // Warm tick (lazy internals would otherwise land in the window); the
    // closure's iterator continues into the measured window, so the pair
    // budget carries one extra entry.
    qsg_gossip_run_into(&mut beliefs, &cfg, 1, &mut next_pair, &mut stream);

    ALLOC_COUNT.store(0, Ordering::Relaxed);
    DEALLOC_COUNT.store(0, Ordering::Relaxed);

    // Measured window: the full run + every reducer, all on caller scratch.
    qsg_gossip_run_into(&mut beliefs, &cfg, steps, &mut next_pair, &mut stream);
    qsg_mean_into(&mut mean, &beliefs, n, k);
    let u = qsg_polarization_u(&mean);
    let v = qsg_disagreement_v(&beliefs, &mean, n);
    let fuel = qsg_uncertainty(&beliefs[..k]);
    let _ = (u, v, fuel);

    let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        allocs, 0,
        "qsg hot path allocated {allocs} times over {steps} steps + reducers (expected 0)"
    );
}
