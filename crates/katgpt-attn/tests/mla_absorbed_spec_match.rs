//! Issue 926 — absorbed-decode G1 spec match.
//!
//! The absorbed path ([`katgpt_attn::mla::mla_forward_token_absorbed`]) must
//! agree with the reconstruct reference ([`katgpt_attn::mla::mla_forward_token`])
//! at the EXISTING spec bar: 1e-4 max-abs on the small config, 1e-3 at the
//! full kimi_k3_0_40b dims (the same two-tier bar `mla_g1_spec_match` pins
//! against the f64 reference). fp reassociation only — the absorbed algebra is
//! `q_c·(W_UK·c_kv) ≡ (W_UKᵀ·q_c)·c_kv` and
//! `Σ_j w_j·(W_UV·c_kv_j) ≡ W_UV·(Σ_j w_j·c_kv_j)`; the reconstruct side is
//! itself f64-verified in `mla_g1_spec_match`, so the pair composes:
//! |absorbed − f64| ≤ |absorbed − recon| + |recon − f64|.
//!
//! Runs the FULL token sequence per scenario (per-token outputs compared, not
//! just the last), because the absorbed value accumulation is where the
//! association differs — short prefixes would under-exercise it.

#![cfg(feature = "mla_absorbed")]

use katgpt_attn::mla::{
    MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights, mla_forward_token,
    mla_forward_token_absorbed,
};
use katgpt_kv::shard_kv::rope::RopeFreqs;

fn small_config() -> MlaConfig {
    MlaConfig {
        kv_lora_rank: 8,
        q_lora_rank: 12,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 4,
        v_head_dim: 4,
        n_heads: 2,
        hidden_size: 16,
        use_output_gate: true,
        use_nope: false,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
    }
}

/// Run BOTH paths over the same token sequence, feeding each step's output
/// back as the next step's input is NOT done (both paths see identical
/// inputs); instead every per-step output pair is compared, and the max diff
/// across the whole rollout is returned.
fn run_pair(
    config: &MlaConfig,
    weights: &MlaWeights,
    tokens: &[Vec<f32>],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let max_seq = tokens.len();
    let mut cache_a = MlaKVCache::new(config, max_seq);
    let mut cache_b = MlaKVCache::new(config, max_seq);
    let mut scratch_a = MlaForwardScratch::new(config, max_seq);
    let mut scratch_b = MlaForwardScratch::new(config, max_seq);
    let mut rope_a = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    let mut rope_b = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);

    let mut recon_last = Vec::new();
    let mut abs_last = Vec::new();
    let mut max_pair_diff = 0.0f32;
    for h in tokens {
        let out_recon = mla_forward_token(config, weights, &mut cache_a, &mut scratch_a, &mut rope_a, h);
        let out_abs = mla_forward_token_absorbed(
            config, weights, &mut cache_b, &mut scratch_b, &mut rope_b, h,
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
    (recon_last, abs_last, vec![max_pair_diff])
}

fn assert_pair(config: &MlaConfig, weights: &MlaWeights, tokens: &[Vec<f32>], tol: f32, label: &str) {
    let (recon, abs, diffs) = run_pair(config, weights, tokens);
    let max_diff = diffs[0];
    assert!(
        max_diff < tol,
        "{label}: absorbed/recon max_diff = {max_diff:.2e} (tol {tol:.0e})"
    );
    // Sanity: the last outputs must have the hidden shape and be finite.
    assert_eq!(recon.len(), config.hidden_size, "{label}: recon shape");
    assert_eq!(abs.len(), config.hidden_size, "{label}: absorbed shape");
    assert!(
        abs.iter().all(|v| v.is_finite()),
        "{label}: non-finite absorbed output"
    );
    eprintln!("{label}: absorbed/recon max_diff = {max_diff:.2e} (tol {tol:.0e})");
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
fn g1_absorbed_single_token_zero_position() {
    let config = small_config();
    let weights = MlaWeights::random(&config, 42);
    let tokens = token_seq(&config, 1, 1);
    assert_pair(&config, &weights, &tokens, 1e-4, "pos=0 single token");
}

#[test]
fn g1_absorbed_single_token_nonzero_position() {
    let config = small_config();
    let weights = MlaWeights::random(&config, 7);
    let tokens = token_seq(&config, 6, 2);
    assert_pair(&config, &weights, &tokens, 1e-4, "pos=5 rope active");
}

#[test]
fn g1_absorbed_multi_token_sequence() {
    let config = small_config();
    let weights = MlaWeights::random(&config, 99);
    let tokens = token_seq(&config, 12, 3);
    assert_pair(&config, &weights, &tokens, 1e-4, "12-token rollout");
}

#[test]
fn g1_absorbed_output_gate_off() {
    let mut config = small_config();
    config.use_output_gate = false;
    let weights = MlaWeights::random(&config, 11);
    let tokens = token_seq(&config, 6, 4);
    assert_pair(&config, &weights, &tokens, 1e-4, "gate off");
}

#[test]
fn g1_absorbed_use_nope() {
    // use_nope skips RoPE application; scores are content + unrotated rope
    // dots. The absorbed path must keep the same structure (content through
    // q_abs, rope part unchanged).
    let mut config = small_config();
    config.use_nope = true;
    let weights = MlaWeights::random(&config, 23);
    let tokens = token_seq(&config, 6, 5);
    assert_pair(&config, &weights, &tokens, 1e-4, "use_nope");
}

#[test]
fn g1_absorbed_kimi_k3_0_40b_full_dims() {
    // Full 0.40B dims — the geometry the ~60× FLOP claim is stated at.
    let config = MlaConfig::kimi_k3_0_40b();
    let weights = MlaWeights::random(&config, 777);
    let tokens = token_seq(&config, 4, 6);
    assert_pair(&config, &weights, &tokens, 1e-3, "kimi full dims");
}

#[test]
fn g1_absorbed_longer_sequence_small_config() {
    // Longer rollout on the small config: the softmax-weighted latent
    // accumulation runs over more tokens, widening the association gap the
    // bar must absorb.
    let config = small_config();
    let weights = MlaWeights::random(&config, 555);
    let tokens = token_seq(&config, 64, 7);
    assert_pair(&config, &weights, &tokens, 1e-4, "64-token rollout");
}
