//! Tests for Q-K=V projection sharing (Plan 185).
//!
//! G1–G7: weight merge helpers, cache layout, and defaults.

use katgpt_core::types::{AttentionProjection, CacheLayout};
use katgpt_rs::kv_share::{cache_layout, merge_kv_bias, merge_kv_weights};

#[test]
fn g1_merge_kv_weights_produces_correct_average() {
    let w_k = vec![2.0, 4.0, 6.0, 8.0];
    let w_v = vec![10.0, 20.0, 30.0, 40.0];
    let merged = merge_kv_weights(&w_k, &w_v);
    assert_eq!(merged, vec![6.0, 12.0, 18.0, 24.0]);
}

#[test]
fn g2_merge_kv_bias_produces_correct_average() {
    let b_k = vec![1.0, -1.0, 0.5];
    let b_v = vec![3.0, 1.0, -0.5];
    let merged = merge_kv_bias(&b_k, &b_v);
    assert_eq!(merged, vec![2.0, 0.0, 0.0]);
}

#[test]
fn g3_cache_layout_full_is_kv() {
    assert_eq!(cache_layout(AttentionProjection::Full), CacheLayout::KV);
}

#[test]
fn g4_cache_layout_shared_kv_is_k() {
    assert_eq!(cache_layout(AttentionProjection::SharedKV), CacheLayout::K);
}

#[test]
fn g5_attention_projection_default_is_full() {
    assert_eq!(AttentionProjection::default(), AttentionProjection::Full);
}

#[test]
fn g6_merging_identical_weights_returns_same_weights() {
    let w = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let merged = merge_kv_weights(&w, &w);
    assert_eq!(merged, w);
}

#[test]
fn g7_merging_with_zeros_returns_half() {
    let w = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let zeros = vec![0.0; 5];
    let merged = merge_kv_weights(&w, &zeros);
    assert_eq!(merged, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}
