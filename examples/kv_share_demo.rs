//! Q-K=V Projection Sharing Demo — Inference KV Cache Halving
//!
//! Demonstrates how Q-K=V sharing reduces KV cache memory and attention FLOPs.
//!
//! Run: `cargo run --example kv_share_demo --features kv_share`

#[cfg(feature = "kv_share")]
use katgpt_rs::kv_share::{
    merge_kv_weights, cache_layout, cache_slots_per_layer,
    memory_per_token, attention_flops_factor,
};
#[cfg(feature = "kv_share")]
use katgpt_core::types::{AttentionProjection, CacheLayout};

#[cfg(feature = "kv_share")]
fn main() {
    println!("=== Q-K=V Projection Sharing Demo ===\n");

    // 1. Weight merging
    let w_k: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let w_v: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
    let w_kv = merge_kv_weights(&w_k, &w_v);
    println!("Weight merging:");
    println!("  W_k = {w_k:?}");
    println!("  W_v = {w_v:?}");
    println!("  W_kv = {w_kv:?}\n");

    // 2. Cache layout
    let full_layout = cache_layout(AttentionProjection::Full);
    let shared_layout = cache_layout(AttentionProjection::SharedKV);
    println!("Cache layout:");
    println!("  Full:    {full_layout:?}");
    println!("  SharedKV: {shared_layout:?}\n");

    // 3. Memory comparison
    let head_dim = 64;

    println!("Memory per token (head_dim={head_dim}):");
    println!("  Full:    {} bytes", memory_per_token(CacheLayout::KV, head_dim));
    println!("  SharedKV: {} bytes", memory_per_token(CacheLayout::K, head_dim));
    println!("  Savings:  50%\n");

    println!("Cache slots per layer (base=1024):");
    let base_slots = 1024;
    println!("  Full:    {} slots", cache_slots_per_layer(CacheLayout::KV, base_slots));
    println!("  SharedKV: {} slots", cache_slots_per_layer(CacheLayout::K, base_slots));
    println!("  Capacity: 2x (same memory)\n");

    // 4. FLOPs reduction
    println!("Attention FLOPs factor:");
    println!("  Full:    {:.1}x", attention_flops_factor(AttentionProjection::Full));
    println!("  SharedKV: {:.2}x (33% reduction)", attention_flops_factor(AttentionProjection::SharedKV));

    println!("\n=== Demo Complete ===");
    println!("Key insight: K=V sharing halves KV cache memory with ~3% perplexity cost.");
}

#[cfg(not(feature = "kv_share"))]
fn main() {
    eprintln!("This example requires the `kv_share` feature.");
    eprintln!("Run: cargo run --example kv_share_demo --features kv_share");
}
