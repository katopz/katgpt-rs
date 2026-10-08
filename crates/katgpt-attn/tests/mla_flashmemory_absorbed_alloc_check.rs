//! Issue 926 SECONDARY scope — flashmemory absorbed-decode G4 zero-allocation
//! gate.
//!
//! Steady-state absorbed SPARSE decode must allocate 0 bytes:
//! - `mla_forward_token_flashmemory_absorbed` (256 steady-state steps at a
//!   fixed cache length — scratch, cache, rope tables, block centroids and the
//!   selector's per-head selection capacity are all pre-allocated; the
//!   fallback path is a stack array; the per-block latent accumulator is the
//!   pre-allocated `lat_acc` scratch)
//!
//! Includes periodic selection refreshes inside the measured window (the real
//! decode cadence): `PerHeadSelection::new` pre-reserves `max_blocks` per
//! head, so refresh-path pushes never reallocate after construction — this
//! gate would catch a regression that dropped that reservation.
//!
//! Separate test binary (the `asentmax_alloc_check` convention): the global
//! CountingAllocator must not pick up allocations from parallel tests.
//! The allocator macro is included from katgpt-core's shared test commons
//! via a filesystem `#[path]` include — one source of truth; katgpt-core
//! is a guaranteed-present path dep of this crate.
//!
//! Needs BOTH features (`mla_absorbed` + `flashmemory_sparse`): the whole-file
//! `#![cfg(all(...))]` zeroes the count otherwise, and the `required-features`
//! row in Cargo.toml protects the reader (the Issue-713 green-zero rule).

#![cfg(all(feature = "mla_absorbed", feature = "flashmemory_sparse"))]

#[path = "../../katgpt-core/tests/common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_attn::dash_attn::flashmemory_sparse::{
    FlashMemoryBlockCache, FlashMemoryConfig, FlashMemorySelector,
    mla_forward_token_flashmemory_absorbed,
};
use katgpt_attn::mla::{MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights};
use katgpt_kv::shard_kv::rope::RopeFreqs;

#[test]
fn g4_fm_absorbed_decode_steady_state_allocates_nothing() {
    // Issue 714 — a counter that became a no-op would pass every alloc gate
    // at once; prove it is live BEFORE the measured window.
    assert_counter_is_live();

    let config = MlaConfig::kimi_k3_0_40b();
    let weights = MlaWeights::random(&config, 42);
    let max_seq: usize = 512;
    let fm_config = FlashMemoryConfig {
        block_size: 16,
        refresh_period: 16,
        threshold: 0.5,
    };
    let max_blocks = max_seq.div_ceil(fm_config.block_size).max(1);
    let mut cache = MlaKVCache::new(&config, max_seq);
    let mut scratch = MlaForwardScratch::new(&config, max_seq);
    let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);
    let mut selector = FlashMemorySelector::new(fm_config.clone(), config.n_heads, max_blocks);

    let d = config.hidden_size;
    let mut h: Vec<f32> = (0..d).map(|i| (i as f32).sin() * 0.1).collect();

    // Warm-up: fill the cache to a steady length, warm the SIMD paths, and
    // exercise several selection refreshes (> 2 refresh periods) so the
    // selection Vecs reach their steady capacity BEFORE the measured window.
    for step in 0..64 {
        for v in h.iter_mut() {
            *v = (*v + step as f32 * 0.01).clamp(-1.0, 1.0);
        }
        let out = mla_forward_token_flashmemory_absorbed(
            &config, &weights, &mut cache, &mut scratch, &mut rope, &h,
            &mut block_cache, &mut selector, step,
        );
        std::hint::black_box(&out[..8.min(d)]);
    }

    // Steady state: rewind keeps seq constant (the Ouro pattern), so every
    // step overwrites the same cache slot — pure decode, no growth. Refreshes
    // still fire every 16 steps (period 16 < 256): the selection pushes land
    // in the pre-reserved capacity.
    let steady = cache.seq_len;
    let mut step_input = h.clone();
    let (_, delta) = alloc_delta(|| {
        for i in 0..256 {
            let step = 64 + i;
            for v in step_input.iter_mut() {
                *v = (*v + 0.013).clamp(-1.0, 1.0);
            }
            cache.rewind_to(steady);
            let out = mla_forward_token_flashmemory_absorbed(
                &config, &weights, &mut cache, &mut scratch, &mut rope, &step_input,
                &mut block_cache, &mut selector, step,
            );
            std::hint::black_box(&out[..8.min(d)]);
        }
    });
    assert_eq!(
        delta, 0,
        "steady-state flashmemory absorbed decode allocated {delta} allocation calls"
    );
    eprintln!(
        "g4_fm_absorbed_decode: 256 steady-state steps, 0 allocations (cache seq={steady}, refreshes={})",
        selector.refresh_count()
    );
}
