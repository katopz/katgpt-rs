//! Issue 926 — absorbed-decode G4 zero-allocation gate.
//!
//! Steady-state absorbed decode must allocate 0 bytes:
//! - `mla_forward_token_absorbed` (256 steady-state steps at a fixed cache
//!   length — scratch, cache, rope tables and scores are all pre-allocated)
//!
//! Separate test binary (the `asentmax_alloc_check` convention): the global
//! CountingAllocator must not pick up allocations from parallel tests.
//! The allocator macro is included from katgpt-core's shared test commons
//! via a filesystem `#[path]` include — one source of truth; katgpt-core
//! is a guaranteed-present path dep of this crate.

#![cfg(feature = "mla_absorbed")]

#[path = "../../katgpt-core/tests/common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_attn::mla::{
    MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights, mla_forward_token_absorbed,
};
use katgpt_kv::shard_kv::rope::RopeFreqs;

#[test]
fn g4_absorbed_decode_steady_state_allocates_nothing() {
    // Issue 714 — a counter that became a no-op would pass every alloc gate
    // at once; prove it is live BEFORE the measured window.
    assert_counter_is_live();

    let config = MlaConfig::kimi_k3_0_40b();
    let weights = MlaWeights::random(&config, 42);
    let max_seq = 512;
    let mut cache = MlaKVCache::new(&config, max_seq);
    let mut scratch = MlaForwardScratch::new(&config, max_seq);
    let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);

    let d = config.hidden_size;
    let mut h: Vec<f32> = (0..d).map(|i| (i as f32).sin() * 0.1).collect();

    // Warm-up: fill the cache to a steady length + warm the SIMD paths.
    for step in 0..64 {
        for v in h.iter_mut() {
            *v = (*v + step as f32 * 0.01).clamp(-1.0, 1.0);
        }
        let out =
            mla_forward_token_absorbed(&config, &weights, &mut cache, &mut scratch, &mut rope, &h);
        std::hint::black_box(&out[..8.min(d)]);
    }

    // Steady state: rewind keeps seq constant (the Ouro pattern), so every
    // step overwrites the same cache slot — pure decode, no growth.
    let steady = cache.seq_len;
    let mut step_input = h.clone();
    let (_, delta) = alloc_delta(|| {
        for _ in 0..256 {
            for v in step_input.iter_mut() {
                *v = (*v + 0.013).clamp(-1.0, 1.0);
            }
            cache.rewind_to(steady);
            let out = mla_forward_token_absorbed(
                &config, &weights, &mut cache, &mut scratch, &mut rope, &step_input,
            );
            std::hint::black_box(&out[..8.min(d)]);
        }
    });
    assert_eq!(
        delta, 0,
        "steady-state absorbed decode allocated {delta} allocation calls"
    );
    eprintln!("g4_absorbed_decode: 256 steady-state steps, 0 allocations (cache seq={steady})");
}
