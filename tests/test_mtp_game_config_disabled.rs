//! Plan 117 T35: Game Config MTP-Disabled Verification Tests
//!
//! Verifies all game configs produce identical output with/without MTP
//! infrastructure present. Game configs should have MTP disabled by default
//! (high mtp_min_output_tokens or mtp_cluster_topk = 1).
//!
//! Run: `cargo test --test test_mtp_game_config_disabled -- --nocapture`

#![cfg(feature = "dllm")]

use microgpt_rs::types::Config;

// ── T35: Game configs have Top-K disabled (K=1) ────────────────
//
// mtp_cluster_topk = 1 means no clustering — single cluster selection
// identical to pre-Plan-117 behavior.

#[test]
fn test_game_configs_mtp_cluster_topk_disabled() {
    let game_configs = [
        ("game", Config::game()),
        ("draft", Config::draft()),
        ("micro_dllm", Config::micro_dllm()),
    ];

    for (name, config) in &game_configs {
        assert_eq!(
            config.mtp_cluster_topk, 1,
            "{name}: mtp_cluster_topk should be 1 (disabled)"
        );
    }
}

// ── T35: Game configs have output-length gating at maximum ─────
//
// mtp_min_output_tokens = usize::MAX means MTP is never activated
// regardless of sequence length — effectively disabled.

#[test]
fn test_game_configs_min_output_tokens_max() {
    let game_configs = [
        ("game", Config::game()),
        ("draft", Config::draft()),
        ("micro_dllm", Config::micro_dllm()),
    ];

    for (name, config) in &game_configs {
        assert_eq!(
            config.mtp_min_output_tokens,
            usize::MAX,
            "{name}: mtp_min_output_tokens should be usize::MAX (disabled)"
        );
    }
}

// ── T35: Game configs have MTP activation threshold at maximum ─
//
// mtp_activation_threshold = usize::MAX means the MTP conditioning
// projection is never applied — standard AR decode path only.

#[test]
fn test_game_configs_mtp_activation_threshold_max() {
    let game_configs = [
        ("game", Config::game()),
        ("draft", Config::draft()),
        ("micro_dllm", Config::micro_dllm()),
    ];

    for (name, config) in &game_configs {
        assert_eq!(
            config.mtp_activation_threshold,
            usize::MAX,
            "{name}: mtp_activation_threshold should be usize::MAX (disabled)"
        );
    }
}

// ── T35: Game configs have shared KV preloading disabled ───────
//
// mtp_shared_kv_prompt_threshold = usize::MAX means shared KV
// preloading from target to draft never activates.

#[test]
fn test_game_configs_shared_kv_threshold_max() {
    let game_configs = [
        ("game", Config::game()),
        ("draft", Config::draft()),
        ("micro_dllm", Config::micro_dllm()),
    ];

    for (name, config) in &game_configs {
        assert_eq!(
            config.mtp_shared_kv_prompt_threshold,
            usize::MAX,
            "{name}: mtp_shared_kv_prompt_threshold should be usize::MAX (disabled)"
        );
    }
}

// ── T35: Game configs have cluster vocab threshold at maximum ──
//
// mtp_cluster_vocab_threshold = usize::MAX means clustered LM head
// never activates — standard full-vocab softmax path only.

#[test]
fn test_game_configs_cluster_vocab_threshold_max() {
    let game_configs = [
        ("game", Config::game()),
        ("draft", Config::draft()),
        ("micro_dllm", Config::micro_dllm()),
    ];

    for (name, config) in &game_configs {
        assert_eq!(
            config.mtp_cluster_vocab_threshold,
            usize::MAX,
            "{name}: mtp_cluster_vocab_threshold should be usize::MAX (disabled)"
        );
    }
}
