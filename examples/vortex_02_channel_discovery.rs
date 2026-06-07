//! VortexFlow Channel Discovery calibration example (Plan 196 T14).
//!
//! Demonstrates routing channel discovery: identifies which channel groups are
//! routing-critical, then runs ChannelAwareRouter with discovered channels.
//!
//! Run: `cargo run --example vortex_02_channel_discovery --features vortex_flow`

#![cfg(feature = "vortex_flow")]

use katgpt_rs::dash_attn::{
    ChannelAwareCache, ChannelAwareRouter, RoutingChannelDiscovery, VortexFlow, VortexScratch,
};

const HEAD_DIM: usize = 32;
const N_BLOCKS: usize = 16;
const BLOCK_SIZE: usize = 4;
const TOP_K: usize = 4;

fn main() {
    println!("=== VortexFlow Channel Discovery (Plan 196 T14) ===\n");

    // ── 1. Synthetic KV Cache ────────────────────────────────
    println!("── Step 1: Generate Synthetic KV Cache ──");
    println!("  head_dim={HEAD_DIM}, n_blocks={N_BLOCKS}, block_size={BLOCK_SIZE}");

    // Create centroids where dimensions 0-7 carry most routing signal
    // and dimensions 8-31 are near-uniform noise
    let mut centroids = vec![0.0f32; N_BLOCKS * HEAD_DIM];
    for bi in 0..N_BLOCKS {
        let base = bi * HEAD_DIM;
        // First 8 dims: strong directional signal (one-hot-like)
        let direction = bi % 8;
        centroids[base + direction] = 2.0 + (bi as f32 * 0.5);
        // Remaining dims: low-variance noise
        for d in 8..HEAD_DIM {
            centroids[base + d] = 0.01 * ((bi + d) as f32 * 0.3).sin();
        }
    }
    println!("  Created {N_BLOCKS} centroids (dims 0-7: strong signal, dims 8-31: noise)");

    // Generate calibration queries aligned with the strong-signal dimensions
    let n_queries = 8;
    let mut queries = vec![0.0f32; n_queries * HEAD_DIM];
    for qi in 0..n_queries {
        let base = qi * HEAD_DIM;
        queries[base + (qi % 8)] = 1.5 + (qi as f32 * 0.3);
        // Add some noise in non-signal dimensions
        for d in 8..HEAD_DIM {
            queries[base + d] = 0.01 * (d as f32 * 0.1);
        }
    }
    println!("  Generated {n_queries} calibration queries");

    // ── 2. Run RoutingChannelDiscovery ──────────────────────
    println!("\n── Step 2: Channel Discovery Calibration ──");
    let discovery = RoutingChannelDiscovery::new();
    println!(
        "  Config: {} groups, {:.0}% threshold",
        discovery.n_groups,
        discovery.critical_threshold * 100.0
    );

    let mask = discovery.calibrate(HEAD_DIM, &centroids, &queries, TOP_K);

    println!("\n  Routing-critical channels:");
    let routing_channels = mask.routing_channels();
    let routing_dim = routing_channels.len();
    for group in 0..discovery.n_groups {
        let group_size = HEAD_DIM.div_ceil(discovery.n_groups);
        let g_start = group * group_size;
        let g_end = (g_start + group_size).min(HEAD_DIM);
        let critical_in_group = (g_start..g_end).filter(|&d| mask.channels[d]).count();
        let total_in_group = g_end - g_start;
        let status = match critical_in_group == total_in_group {
            true => "CRITICAL",
            false => match critical_in_group > 0 {
                true => "PARTIAL",
                false => "non-critical",
            },
        };
        println!(
            "    Group {group} (dims {g_start}-{g_end}): {critical_in_group}/{total_in_group} critical [{status}]"
        );
    }

    println!(
        "\n  Total routing channels: {routing_dim}/{HEAD_DIM} ({:.0}% compression)",
        (1.0 - routing_dim as f32 / HEAD_DIM as f32) * 100.0
    );

    // ── 3. Build ChannelAwareCache with discovered channels ──
    println!("\n── Step 3: ChannelAwareRouter with Discovered Channels ──");
    let router = ChannelAwareRouter::new(true);
    let mut cache = ChannelAwareCache::new(N_BLOCKS, HEAD_DIM, routing_channels.clone());
    let mut scratch = VortexScratch::new(N_BLOCKS);

    // Populate cache using centroids as block keys
    for bi in 0..N_BLOCKS {
        let base = bi * HEAD_DIM;
        let keys = centroids[base..base + HEAD_DIM].to_vec();
        let vals = vec![0.0; HEAD_DIM];
        router.forward_cache(&mut cache, &keys, &vals, bi, HEAD_DIM);
    }
    println!("  Cached {N_BLOCKS} blocks with routing_dim={routing_dim}");

    // ── 4. Route and compare ─────────────────────────────────
    println!("\n── Step 4: Routing Comparison ──");

    // Query targeting block 5's direction (5 % 8 = 5)
    let target_block = 5usize;
    let mut query = vec![0.0f32; HEAD_DIM];
    query[target_block % 8] = 1.5;

    // Channel-aware routing
    let decision_channel = router.forward_indexer(&query, &cache, N_BLOCKS, TOP_K, &mut scratch);

    println!("  Channel-aware routing (top-{TOP_K}):");
    for (rank, &block) in decision_channel.blocks.iter().enumerate() {
        let marker = match block == target_block {
            true => " ← target",
            false => "",
        };
        println!(
            "    #{rank}: block {block}, weight={:.4}{marker}",
            decision_channel.weights[rank]
        );
    }

    // Full-dim routing for comparison
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut full_scores: Vec<(usize, f32)> = (0..N_BLOCKS)
        .map(|bi| {
            let centroid = &centroids[bi * HEAD_DIM..(bi + 1) * HEAD_DIM];
            let dot: f32 = query.iter().zip(centroid.iter()).map(|(a, b)| a * b).sum();
            (bi, dot * scale)
        })
        .collect();
    full_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n  Full-dim routing (top-{TOP_K}):");
    for (rank, &(block, score)) in full_scores.iter().take(TOP_K).enumerate() {
        let marker = match block == target_block {
            true => " ← target",
            false => "",
        };
        println!("    #{rank}: block {block}, score={score:.4}{marker}");
    }

    // Compute overlap
    let full_top_k: Vec<usize> = full_scores.iter().take(TOP_K).map(|&(b, _)| b).collect();
    let overlap = decision_channel
        .blocks
        .iter()
        .filter(|b| full_top_k.contains(b))
        .count();
    let overlap_pct = overlap as f32 / TOP_K as f32 * 100.0;

    println!("\n  Block overlap: {overlap}/{TOP_K} = {overlap_pct:.0}%");

    // ── 5. Memory overhead ───────────────────────────────────
    println!("\n── Step 5: Memory Analysis ──");
    let full_bytes = N_BLOCKS * HEAD_DIM * 4; // f32 = 4 bytes
    let routing_bytes = N_BLOCKS * routing_dim * 4;
    let overhead_pct = routing_bytes as f32 / full_bytes as f32 * 100.0;
    println!("  Full-dim storage: {full_bytes} bytes");
    println!("  Routing storage:  {routing_bytes} bytes ({overhead_pct:.0}%)");
    println!(
        "  Memory saving:    {} bytes ({:.0}%)",
        full_bytes - routing_bytes,
        100.0 - overhead_pct
    );

    // ── Summary ──────────────────────────────────────────────
    println!("\n=== Summary ===");
    println!("  Discovered {routing_dim} routing-critical channels out of {HEAD_DIM}");
    println!(
        "  Routing compression: {:.0}%",
        (1.0 - routing_dim as f32 / HEAD_DIM as f32) * 100.0
    );
    println!("  Routing quality (overlap): {overlap_pct:.0}%");
    let found = decision_channel.blocks.contains(&target_block);
    println!("  Target block {target_block} found: {found}");

    // GOAT gate
    if found && overlap_pct >= 75.0 {
        println!("\n  🐐 GOAT gate: PASS (found target, overlap ≥ 75%)");
    } else {
        println!("\n  ⚠ GOAT gate: CHECK (found={found}, overlap={overlap_pct:.0}%)");
    }
}
