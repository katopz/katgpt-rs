//! Phase 7 G4 — Kimi-K3-0.40B forward path zero-allocation steady state.
//!
//! Verifies the per-token decode hot path performs **zero heap allocations**
//! after warm-up. This is the alloc gate that justifies the "zero-copy /
//! zero-alloc" claim made in the loader + `AttnResBlockState` refactor
//! (Phase 7, 2026-08-02).
//!
//! # What's tested
//!
//! The hot path is:
//!   `kimi_k3_forward_token` (or `_timed`) → 8 decoder layers → output
//!   attn-res → final norm → lm_head
//!
//! Each layer touches:
//!   - Embedding copy (`runtime.hidden.copy_from_slice`)
//!   - 6× KDA / 2× MLA attention (with pre-allocated KV cache + scratch)
//!   - Dense / MoE FFN (with pre-allocated scratch)
//!   - 2× `apply_attn_res` (self + mlp; scratch pre-allocated)
//!   - `AttnResBlockState::push` (2× per token at block boundaries — the
//!     Phase 7 fix moved this from `to_vec()` to a pre-allocated pool)
//!
//! # Mechanism
//!
//! Uses the `katgpt_core::alloc::TrackingAllocator` installed under
//! `cfg(debug_assertions)` in `katgpt-rs/src/lib.rs`. The test:
//!   1. Builds a runtime sized for the real config
//!   2. Runs a warm-up token (sizes all caches + scratch + pool slots)
//!   3. Resets the alloc counter
//!   4. Runs N decode tokens
//!   5. Asserts the counter delta is zero
//!
//! # Why debug-only
//!
//! The `TrackingAllocator` is a debug-only global allocator. Release builds
//! skip this test (it would compile-fail on the missing symbols, so we
//! `#[cfg(debug_assertions)]` the whole module). The zero-alloc claim in
//! release builds is verified by code review: there are no `to_vec()`,
//! `Vec::new()`, or `Vec::with_capacity()` calls inside the forward hot path
//! (the per-token loop body of `kimi_k3_forward_token`).
//!
//! Run:
//! ```sh
//! cargo test --features kimi_k3_loader --test kimi_k3_g4_alloc_free -- --nocapture --ignored
//! ```

#![cfg(all(feature = "kimi_k3_loader", debug_assertions))]

use std::path::Path;

// Issue 721 T3: install the tracking allocator in THIS test binary (the root
// lib no longer registers a `#[global_allocator]` as a library).
#[path = "common/alloc_tracking.rs"]
mod alloc_tracking;

use katgpt_rs::kimi_k3::loader::load_kimi_k3;
use katgpt_rs::kimi_k3::model::{
    ForwardTiming, KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token,
    kimi_k3_forward_token_timed,
};

fn model_dir() -> String {
    std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/data/kimi-k3-0.40b")
    })
}

fn model_path() -> String {
    format!("{}/model.safetensors", model_dir())
}

fn model_exists() -> bool {
    Path::new(&model_path()).exists()
}

/// Decode N tokens after warm-up + assert zero allocations on the calling
/// thread for the entire decode loop.
///
/// We use the timed forward variant because it exercises the exact same code
/// path as the example (plus a few `Instant::now()` calls which don't alloc).
#[test]
#[ignore = "requires real model.safetensors (~1.5 GB) at data/kimi-k3-0.40b/"]
fn g4_kimi_k3_forward_zero_alloc_steady_state() {
    if !model_exists() {
        eprintln!(
            "⏭️  skipping: {} not found (set KIMI_K3_MODEL_DIR to override)",
            model_path()
        );
        return;
    }

    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let weights = load_kimi_k3(&model_path()).expect("load failed");
    let mut runtime = KimiK3Runtime::new(&config, 64);

    // ── Warm-up: run one forward so every cache/scratch/pool slot is sized ──
    // This is the critical step: the first forward sizes the MLA KV cache,
    // KDA state, MoE expert scratch, and the AttnResBlockState pool (2 slots
    // for block_size=4, 8 layers). Without warm-up, the first decode would
    // allocate these — that's one-time init, not steady-state alloc.
    let _ = kimi_k3_forward_token(&config, &weights, &mut runtime, 1u32);

    // ── Reset alloc counter ─────────────────────────────────────────────────
    katgpt_core::alloc::reset_alloc_stats();
    let (count_before, bytes_before) = katgpt_core::alloc::get_alloc_stats();

    // Sentinel: confirm the allocator is actually counting on this thread.
    // If this stays at zero after an intentional alloc, the global allocator
    // isn't installed (the per-target `alloc_tracking` module is what installs
    // it — see `tests/common/alloc_tracking.rs`, katgpt-rs Issue 721 T3) and
    // the test can't measure anything.
    let _sentinel: Vec<u8> = vec![0u8; 8];
    let (count_after_sentinel, _) = katgpt_core::alloc::get_alloc_stats();
    assert!(
        count_after_sentinel > count_before,
        "TrackingAllocator not counting on this thread — global allocator missing?"
    );

    // Reset again after the sentinel check.
    katgpt_core::alloc::reset_alloc_stats();

    // ── Run N decode tokens (the steady-state hot path) ─────────────────────
    let n_decode = 16usize;
    let mut timing = ForwardTiming::default();
    let mut current_tok: u32 = 1; // BOS

    // First, run ONE more warm-up token AFTER the counter is reset but BEFORE
    // we start measuring. This catches any lazy-init allocations that the
    // first warm-up token didn't trigger (e.g. a cache that grows on the 2nd
    // call but not the 1st).
    let _ = kimi_k3_forward_token(&config, &weights, &mut runtime, current_tok);

    // Reset again — the second warm-up may have allocated.
    katgpt_core::alloc::reset_alloc_stats();

    for _ in 0..n_decode {
        let logits = kimi_k3_forward_token_timed(
            &config,
            &weights,
            &mut runtime,
            current_tok,
            &mut timing,
        );
        // Greedy argmax (no allocation — operates on the existing logits slice)
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        current_tok = best_idx as u32;
    }

    // ── Assert zero allocations ─────────────────────────────────────────────
    let (count_after, bytes_after) = katgpt_core::alloc::get_alloc_stats();
    let _ = count_before; // suppress unused warning (baseline captured pre-sentinel)
    let _ = bytes_before;

    // The cleanest read: count_after is relative to the LAST reset (the one
    // after the sentinel), which is the decode loop's baseline.
    let decode_alloc_count = count_after;
    let decode_alloc_bytes = bytes_after;

    println!(
        "   g4_kimi_k3_forward_zero_alloc: {n_decode} decode tokens → {decode_alloc_count} allocs, {decode_alloc_bytes} bytes"
    );

    assert_eq!(
        decode_alloc_count, 0,
        "G4 FAIL: forward hot path allocated {decode_alloc_count} times ({decode_alloc_bytes} bytes) \
         across {n_decode} decode tokens. Expected zero. Check for `to_vec()`, `Vec::new()`, or \
         `Vec::with_capacity()` in the per-token loop body of `kimi_k3_forward_token`."
    );
    assert_eq!(
        decode_alloc_bytes, 0,
        "G4 FAIL: forward hot path allocated {decode_alloc_bytes} bytes across {n_decode} tokens."
    );
    println!("   ✅ G4 PASS: forward hot path is zero-alloc across {n_decode} decode tokens.");
}

/// Sanity test: verify `AttnResBlockState::push` does NOT allocate when the
/// pool has capacity, and DOES allocate when the pool is exhausted.
///
/// This is a unit-level test of the Phase 7 fix (pool-based push). It doesn't
/// need real model weights — just the block state struct.
#[test]
fn g4_block_state_push_uses_pool_no_alloc() {
    use katgpt_transformer::attn_res::AttnResBlockState;

    let d = 1024;
    let max_entries = 2; // Kimi-K3-0.40B: 8 layers / block_size 4 = 2 boundaries

    let mut state = AttnResBlockState::new_with_capacity(d, max_entries);
    let hidden = vec![0.5f32; d];

    katgpt_core::alloc::reset_alloc_stats();

    // Push within capacity — should NOT allocate.
    state.push(&hidden);
    state.push(&hidden);
    assert_eq!(state.len(), 2);

    let (count, _bytes) = katgpt_core::alloc::get_alloc_stats();
    assert_eq!(
        count, 0,
        "push within pool capacity must not allocate — got {count} allocs"
    );

    // Clear moves slots back to pool — should NOT allocate.
    state.clear();
    let (count_after_clear, _) = katgpt_core::alloc::get_alloc_stats();
    assert_eq!(
        count_after_clear, 0,
        "clear must not allocate — got {count_after_clear} allocs"
    );

    // Push again after clear — should reuse pooled slots, NOT allocate.
    state.push(&hidden);
    state.push(&hidden);
    let (count_after_reuse, _) = katgpt_core::alloc::get_alloc_stats();
    assert_eq!(
        count_after_reuse, 0,
        "push after clear must reuse pooled slots (zero alloc) — got {count_after_reuse} allocs"
    );

    // Push BEYOND capacity — this WILL allocate (contract violation fallback).
    state.push(&hidden);
    let (count_after_overflow, _) = katgpt_core::alloc::get_alloc_stats();
    assert!(
        count_after_overflow > 0,
        "push beyond pool capacity must allocate (fallback path) — got 0 allocs"
    );
}
