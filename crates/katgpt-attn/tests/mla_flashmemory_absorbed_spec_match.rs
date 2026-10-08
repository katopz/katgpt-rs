//! Issue 926 SECONDARY scope — flashmemory absorbed-decode G1 spec match.
//!
//! The absorbed selected-block path
//! ([`katgpt_attn::dash_attn::flashmemory_sparse::mla_forward_token_flashmemory_absorbed`])
//! must agree with the reconstruct reference
//! ([`katgpt_attn::dash_attn::flashmemory_sparse::mla_forward_token_flashmemory_reconstruct`])
//! at the EXISTING spec bar: 1e-4 max-abs on the small config, 1e-3 at the
//! full kimi_k3_0_40b dims (the same two-tier bar the dense lane's
//! `mla_absorbed_spec_match` pins). fp reassociation only — the absorbed
//! algebra is `q_c·(W_UK·c_kv_j) ≡ (W_UKᵀ·q_c)·c_kv_j` and
//! `Σ_j w_j·(W_UV·c_kv_j) ≡ W_UV·(Σ_j w_j·c_kv_j)` over the SELECTED tokens.
//!
//! Block SELECTION must be identical on both sides (the task contract:
//! absorption changes how selected rows are read, never which rows are
//! selected). Both sides run the same `flashmemory_decode_prefix` — the same
//! q_c, the same block centroids, the same selector state (separate instances
//! fed identical inputs) — so any selection drift would surface as a huge
//! diff, far beyond the tolerance.
//!
//! Runs the FULL token sequence per scenario (per-step outputs compared, not
//! just the last), and covers the sparse-specific paths the dense test cannot:
//! multi-block selections, the fallback-to-recent-block safety net, and the
//! periodic-refresh cached-selection window (seq grows while the selection is
//! frozen between refreshes).
//!
//! Needs BOTH features (`mla_absorbed` + `flashmemory_sparse`): the whole-file
//! `#![cfg(all(...))]` zeroes the count otherwise, and the `required-features`
//! row in Cargo.toml protects the reader (the Issue-713 green-zero rule).

#![cfg(all(feature = "mla_absorbed", feature = "flashmemory_sparse"))]

use katgpt_attn::dash_attn::flashmemory_sparse::{
    FlashMemoryBlockCache, FlashMemoryConfig, FlashMemorySelector,
    mla_forward_token_flashmemory_absorbed, mla_forward_token_flashmemory_reconstruct,
};
use katgpt_attn::mla::{MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights};
use katgpt_kv::shard_kv::rope::RopeFreqs;

fn small_config() -> MlaConfig {
    MlaConfig {
        kv_lora_rank: 32,
        q_lora_rank: 64,
        qk_nope_head_dim: 16,
        qk_rope_head_dim: 8,
        v_head_dim: 16,
        n_heads: 4,
        hidden_size: 128,
        use_output_gate: true,
        use_nope: false,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
    }
}

/// Run BOTH paths over the same token sequence, each with its own
/// cache/scratch/rope/block_cache/selector instances fed IDENTICAL inputs
/// (so selections match by construction), and return the max per-step
/// output diff across the whole rollout.
#[allow(clippy::too_many_arguments)]
fn run_pair(
    config: &MlaConfig,
    fm_config: &FlashMemoryConfig,
    weights: &MlaWeights,
    tokens: &[Vec<f32>],
) -> (Vec<f32>, Vec<f32>, f32) {
    let max_seq = tokens.len();
    let mut cache_a = MlaKVCache::new(config, max_seq);
    let mut cache_b = MlaKVCache::new(config, max_seq);
    let mut scratch_a = MlaForwardScratch::new(config, max_seq);
    let mut scratch_b = MlaForwardScratch::new(config, max_seq);
    let mut rope_a = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    let mut rope_b = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    let mut block_cache_a = FlashMemoryBlockCache::new(config, fm_config, max_seq);
    let mut block_cache_b = FlashMemoryBlockCache::new(config, fm_config, max_seq);
    let max_blocks = max_seq.div_ceil(fm_config.block_size).max(1);
    let mut selector_a = FlashMemorySelector::new(fm_config.clone(), config.n_heads, max_blocks);
    let mut selector_b = FlashMemorySelector::new(fm_config.clone(), config.n_heads, max_blocks);

    let mut recon_last = Vec::new();
    let mut abs_last = Vec::new();
    let mut max_pair_diff = 0.0f32;
    for (step, h) in tokens.iter().enumerate() {
        let out_recon = mla_forward_token_flashmemory_reconstruct(
            config, weights, &mut cache_a, &mut scratch_a, &mut rope_a, h,
            &mut block_cache_a, &mut selector_a, step,
        );
        let out_abs = mla_forward_token_flashmemory_absorbed(
            config, weights, &mut cache_b, &mut scratch_b, &mut rope_b, h,
            &mut block_cache_b, &mut selector_b, step,
        );
        let step_diff = out_recon
            .iter()
            .zip(out_abs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        max_pair_diff = max_pair_diff.max(step_diff);
        recon_last = out_recon.to_vec();
        abs_last = out_abs.to_vec();
    }
    (recon_last, abs_last, max_pair_diff)
}

fn assert_pair(
    config: &MlaConfig,
    fm_config: &FlashMemoryConfig,
    weights: &MlaWeights,
    tokens: &[Vec<f32>],
    tol: f32,
    label: &str,
) {
    let (recon, abs, max_diff) = run_pair(config, fm_config, weights, tokens);
    assert!(
        max_diff < tol,
        "{label}: flashmemory absorbed/recon max_diff = {max_diff:.2e} (tol {tol:.0e})"
    );
    // Sanity: the last outputs must have the hidden shape and be finite.
    assert_eq!(recon.len(), config.hidden_size, "{label}: recon shape");
    assert_eq!(abs.len(), config.hidden_size, "{label}: absorbed shape");
    assert!(
        abs.iter().all(|v| v.is_finite()),
        "{label}: non-finite absorbed output"
    );
    eprintln!("{label}: flashmemory absorbed/recon max_diff = {max_diff:.2e} (tol {tol:.0e})");
}

fn token_seq(config: &MlaConfig, n: usize, seed: u64) -> Vec<Vec<f32>> {
    // Deterministic, seed-driven tokens (SimpleRng-style LCG, same spirit as
    // MlaWeights::random) so no fixture dependency.
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).max(1);
    (0..n)
        .map(|_| {
            (0..config.hidden_size)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 33) as f32 / (u64::MAX >> 33) as f32) * 0.6 - 0.3
                })
                .collect()
        })
        .collect()
}

#[test]
fn g1_fm_absorbed_single_token_zero_position() {
    let config = small_config();
    let weights = MlaWeights::random(&config, 42);
    let fm_config = FlashMemoryConfig {
        block_size: 4,
        refresh_period: 100,
        threshold: 0.3,
    };
    let tokens = token_seq(&config, 1, 1);
    assert_pair(&config, &fm_config, &weights, &tokens, 1e-4, "pos=0 single token");
}

#[test]
fn g1_fm_absorbed_multi_block_selection() {
    // Most blocks selected (low threshold): the latent accumulator runs over
    // SEVERAL contiguous blocks — the multi-block simd_transpose_matvec_acc
    // alignment the dense lane never exercises.
    let config = small_config();
    let weights = MlaWeights::random(&config, 7);
    let fm_config = FlashMemoryConfig {
        block_size: 4,
        refresh_period: 100,
        threshold: 0.3,
    };
    let tokens = token_seq(&config, 24, 2);
    assert_pair(&config, &fm_config, &weights, &tokens, 1e-4, "multi-block selection");
}

#[test]
fn g1_fm_absorbed_fallback_recent_block() {
    // Impossibly high threshold → every head falls back to the most recent
    // block (the safety net). The accumulator must handle the single-block
    // fallback path identically on both sides.
    let config = small_config();
    let weights = MlaWeights::random(&config, 99);
    let fm_config = FlashMemoryConfig {
        block_size: 4,
        refresh_period: 100,
        threshold: 0.999,
    };
    let tokens = token_seq(&config, 12, 3);
    assert_pair(&config, &fm_config, &weights, &tokens, 1e-4, "fallback recent block");
}

#[test]
fn g1_fm_absorbed_periodic_refresh_cached_selection() {
    // refresh_period=3 over 12 steps: the selection is FROZEN between
    // refreshes while seq keeps growing — later tokens sit in blocks the
    // cached selection never picked. Both sides must freeze identically.
    let config = small_config();
    let weights = MlaWeights::random(&config, 11);
    let fm_config = FlashMemoryConfig {
        block_size: 4,
        refresh_period: 3,
        threshold: 0.5,
    };
    let tokens = token_seq(&config, 12, 4);
    assert_pair(&config, &fm_config, &weights, &tokens, 1e-4, "periodic refresh cached selection");
}

#[test]
fn g1_fm_absorbed_output_gate_off() {
    let mut config = small_config();
    config.use_output_gate = false;
    let weights = MlaWeights::random(&config, 23);
    let fm_config = FlashMemoryConfig {
        block_size: 4,
        refresh_period: 100,
        threshold: 0.3,
    };
    let tokens = token_seq(&config, 6, 5);
    assert_pair(&config, &fm_config, &weights, &tokens, 1e-4, "gate off");
}

#[test]
fn g1_fm_absorbed_use_nope() {
    // use_nope skips RoPE application; scores are content + unrotated rope
    // dots. The absorbed path must keep the same structure (content through
    // q_abs, rope part unchanged).
    let mut config = small_config();
    config.use_nope = true;
    let weights = MlaWeights::random(&config, 31);
    let fm_config = FlashMemoryConfig {
        block_size: 4,
        refresh_period: 100,
        threshold: 0.3,
    };
    let tokens = token_seq(&config, 6, 6);
    assert_pair(&config, &fm_config, &weights, &tokens, 1e-4, "use_nope");
}

#[test]
fn g1_fm_absorbed_kimi_k3_0_40b_full_dims() {
    // Full 0.40B dims — the geometry the ~60× FLOP claim is stated at. Tiny
    // blocks (2 tokens) so the 4-token rollout has real blocks to select.
    let config = MlaConfig::kimi_k3_0_40b();
    let weights = MlaWeights::random(&config, 777);
    let fm_config = FlashMemoryConfig {
        block_size: 2,
        refresh_period: 100,
        threshold: 0.3,
    };
    let tokens = token_seq(&config, 4, 7);
    assert_pair(&config, &fm_config, &weights, &tokens, 1e-3, "kimi full dims");
}

#[test]
fn g1_fm_absorbed_longer_sequence_small_config() {
    // Longer rollout on the small config: the softmax-weighted latent
    // accumulation runs over more selected tokens (and more refresh
    // boundaries), widening the association gap the bar must absorb.
    let config = small_config();
    let weights = MlaWeights::random(&config, 555);
    let fm_config = FlashMemoryConfig {
        block_size: 8,
        refresh_period: 7,
        threshold: 0.4,
    };
    let tokens = token_seq(&config, 64, 8);
    assert_pair(&config, &fm_config, &weights, &tokens, 1e-4, "64-token rollout");
}
