#![cfg(feature = "rt_turbo")]

//! RTPurbo decode bench example — routing efficiency at various sequence lengths.
//!
//! Demonstrates:
//! 1. Set up a synthetic RtTurboCache with 8 heads (2 retrieval, 6 local)
//! 2. Run decode on sequence lengths: 1024, 4096, 16384
//! 3. Print selected token counts per retrieval head
//! 4. Compare retrieval vs local head routing patterns
//! 5. Show sparsity ratios and efficiency stats

use katgpt_rs::rt_turbo::*;

// ── Deterministic PRNG (no `rand` dependency) ──────────────────

/// Xorshift64 PRNG for reproducible synthetic data.
struct SeedRng {
    state: u64,
}

impl SeedRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Returns f32 in [0.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        let bits = ((self.next_u64() >> 41) as u32) | 0x3F80_0000;
        f32::from_bits(bits) - 1.0
    }
}

// ── Synthetic Data Generation ──────────────────────────────────

/// Create synthetic KV cache: `[seq_len][2 * n_heads * head_dim]`.
fn make_kv_cache(seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<Vec<f32>> {
    let kv_dim = 2 * n_heads * head_dim;
    (0..seq_len)
        .map(|pos| vec![pos as f32 * 0.01; kv_dim])
        .collect()
}

/// Create synthetic pre-RoPE query vectors with peaked patterns.
///
/// Each head gets a slightly different query so retrieval heads
/// produce varied scores across positions.
fn make_query_peaked(n_heads: usize, head_dim: usize, rng: &mut SeedRng) -> Vec<Vec<f32>> {
    (0..n_heads)
        .map(|h| {
            let bias = h as f32 * 0.1;
            (0..head_dim)
                .map(|d| {
                    let base = ((d + h) as f32 * 0.05).sin() + bias;
                    base + rng.next_f32() * 0.1
                })
                .collect()
        })
        .collect()
}

/// Create synthetic pre-RoPE key vectors: `[seq_len][n_heads * head_dim]`.
///
/// Includes peaked regions (simulating "needle" positions) so
/// retrieval heads can discover high-scoring positions.
fn make_key_pre_rope(
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    rng: &mut SeedRng,
) -> Vec<Vec<f32>> {
    let total_dim = n_heads * head_dim;

    // Create needle positions at ~10% and ~90% of sequence
    let needle_a_start = seq_len / 10;
    let needle_a_end = needle_a_start + (seq_len / 20).max(4);
    let needle_b_start = seq_len * 9 / 10;
    let needle_b_end = (needle_b_start + (seq_len / 20).max(4)).min(seq_len);

    (0..seq_len)
        .map(|pos| {
            let is_needle = (pos >= needle_a_start && pos < needle_a_end)
                || (pos >= needle_b_start && pos < needle_b_end);

            (0..total_dim)
                .map(|d| {
                    let base = (pos as f32 * 0.01 + d as f32 * 0.001).sin();
                    let needle_boost = if is_needle { 2.0 } else { 0.0 };
                    base + needle_boost + rng.next_f32() * 0.05
                })
                .collect()
        })
        .collect()
}

/// Run a single decode bench at the given sequence length.
fn run_decode_bench(seq_len: usize, rng: &mut SeedRng) -> DecodeBenchResult {
    let n_heads = 8;
    let head_dim = 64;
    let low_dim = 16;

    // 2 retrieval heads (top 25% since 2/8 = 0.25 > 0.15 rounds up to ceil(0.15*8)=2)
    let n_retrieval = 2;

    // Config with realistic defaults
    let config = katgpt_rs::types::RtTurboConfig {
        retrieval_head_ratio: 0.15,
        low_dim,
        top_p: 0.9,
        sliding_window: 8192,
        sink_tokens: 4,
        block_size: 64,
    };

    // Build calibration: first 2 heads are retrieval (highest scores)
    let mut scores = vec![0.0f32; n_heads];
    scores[0] = 0.95; // retrieval
    scores[1] = 0.05; // local
    scores[2] = 0.05; // local
    scores[3] = 0.85; // retrieval
    scores[4] = 0.05; // local
    scores[5] = 0.05; // local
    scores[6] = 0.05; // local
    scores[7] = 0.05; // local

    let calibration = calibrate_from_scores(&scores, &config);

    // Build projection with xavier init
    let projection = RetrievalProjection::xavier(n_retrieval, head_dim, low_dim);

    // Build cache
    let mut cache = RtTurboCache::new(calibration.clone(), projection, config, 0);

    // Generate synthetic data
    let kv_cache = make_kv_cache(seq_len, n_heads, head_dim);
    let query = make_query_peaked(n_heads, head_dim, rng);
    let key_pre_rope = make_key_pre_rope(seq_len, n_heads, head_dim, rng);

    // Run decode
    let result = cache.decode(&kv_cache, &query, &key_pre_rope);

    // Compute stats
    let local_window_size = result.local_window.1.saturating_sub(result.local_window.0);
    let local_flops = local_window_size + result.sink_indices.len();
    let retrieval_flops_per_head: Vec<usize> =
        result.selected_indices.iter().map(|s| s.len()).collect();
    let retrieval_flops: usize = retrieval_flops_per_head.iter().sum();

    // Dense baseline: all heads attend to all positions
    let dense_flops = n_heads * seq_len;

    // RtTurbo total: local heads × window + retrieval heads × selected
    let rt_turbo_flops = (calibration.n_local() * local_flops) + retrieval_flops;

    DecodeBenchResult {
        seq_len,
        n_retrieval: calibration.n_retrieval(),
        n_local: calibration.n_local(),
        selected_per_retrieval: retrieval_flops_per_head,
        sink_indices: result.sink_indices,
        local_window_size,
        rt_turbo_flops,
        dense_flops,
        sparsity: 1.0 - (rt_turbo_flops as f64 / dense_flops as f64),
    }
}

struct DecodeBenchResult {
    seq_len: usize,
    n_retrieval: usize,
    n_local: usize,
    selected_per_retrieval: Vec<usize>,
    sink_indices: Vec<usize>,
    local_window_size: usize,
    rt_turbo_flops: usize,
    dense_flops: usize,
    sparsity: f64,
}

fn main() {
    println!("=== RTPurbo Decode Bench (Plan 126) ===\n");

    let mut rng = SeedRng::new(42);
    let seq_lengths = [1024, 4096, 16384];

    // ── Header ──────────────────────────────────────────────────
    println!("Configuration:");
    println!("  Heads:           8 (2 retrieval, 6 local)");
    println!("  Head dim:        64");
    println!("  Low dim:         16");
    println!("  Top-p:           0.9");
    println!("  Sliding window:  8192");
    println!("  Sink tokens:     4");
    println!("  Block size:      64");
    println!();

    println!("┌──────────┬──────────────────────────┬────────────────┬────────────┬──────────┐");
    println!("│ Seq Len  │ Retrieval Heads (tokens) │ Local (window) │ RtTurbo    │ Sparsity │");
    println!("│          │  H0       H3             │ heads×window   │ FLOPs      │          │");
    println!("├──────────┼──────────────────────────┼────────────────┼────────────┼──────────┤");

    let mut results = Vec::new();

    for &seq_len in &seq_lengths {
        let result = run_decode_bench(seq_len, &mut rng);
        results.push(result);
    }

    for r in &results {
        let h0_tokens = r.selected_per_retrieval.first().copied().unwrap_or(0);
        let h3_tokens = r.selected_per_retrieval.get(1).copied().unwrap_or(0);

        let local_desc = format!(
            "{n_local}×{window}",
            n_local = r.n_local,
            window = r.local_window_size
        );

        let sparsity_pct = format!("{:.1}%", r.sparsity * 100.0);

        println!(
            "│ {seq_len:>8} │ {h0_tokens:>8}  {h3_tokens:>8}     │ {local_desc:>14} │ {flops:>10} │ {sparsity:>8} │",
            seq_len = r.seq_len,
            h0_tokens = h0_tokens,
            h3_tokens = h3_tokens,
            flops = r.rt_turbo_flops,
            sparsity = sparsity_pct,
        );
    }

    println!("└──────────┴──────────────────────────┴────────────────┴────────────┴──────────┘");

    // ── Detailed breakdown ──────────────────────────────────────
    println!("\nDetailed Breakdown:");

    for r in &results {
        let dense_flops = r.dense_flops;
        let savings = dense_flops as f64 / r.rt_turbo_flops as f64;
        let in_window = r.seq_len <= 8192;

        println!("\n  Seq Len = {seq_len}", seq_len = r.seq_len);
        println!("    Retrieval heads ({n_ret}):", n_ret = r.n_retrieval);

        for (i, &tokens) in r.selected_per_retrieval.iter().enumerate() {
            let head_idx = match i {
                0 => 0,
                1 => 3,
                _ => i,
            };
            let pct = tokens as f64 / r.seq_len as f64 * 100.0;
            println!("      Head {head_idx}: {tokens} tokens selected ({pct:.1}% of seq_len)");
        }

        println!(
            "    Local heads ({n_loc}): {window} tokens each (sliding window {window_status})",
            n_loc = r.n_local,
            window = r.local_window_size,
            window_status = if in_window {
                "covers full seq"
            } else {
                "partial coverage"
            }
        );
        println!(
            "    Sink tokens: {sinks} (always included)",
            sinks = r.sink_indices.len()
        );
        println!(
            "    Dense FLOPs: {dense}, RtTurbo FLOPs: {rt} ({savings:.1}× reduction)",
            dense = dense_flops,
            rt = r.rt_turbo_flops,
            savings = savings,
        );
    }

    // ── Prefill comparison ──────────────────────────────────────
    println!("\nPrefill Routing:");
    let config = katgpt_rs::types::RtTurboConfig::default();

    let scores = vec![0.95, 0.05, 0.05, 0.85, 0.05, 0.05, 0.05, 0.05];
    let calibration = calibrate_from_scores(&scores, &config);
    let projection = RetrievalProjection::xavier(calibration.n_retrieval(), 64, 16);
    let cache = RtTurboCache::new(calibration, projection, config, 0);

    let prefill_result = cache.prefill();
    println!(
        "  All heads attend densely during prefill: {n_heads} heads",
        n_heads = prefill_result.n_total_heads
    );
    println!(
        "  Dense heads: {count}",
        count = prefill_result.dense_heads.len()
    );
    println!("  (Sparse routing applies only during decode phase)");

    // ── Summary ─────────────────────────────────────────────────
    println!("\n=== Summary ===");

    for r in &results {
        let savings = r.dense_flops as f64 / r.rt_turbo_flops as f64;
        println!(
            "  seq_len={seq_len:>5}: {sparsity:.1}% sparsity, {savings:.1}× fewer FLOPs than dense",
            seq_len = r.seq_len,
            sparsity = r.sparsity * 100.0,
            savings = savings,
        );
    }

    println!(
        "\n✅ Decode bench complete — retrieval heads scan selectively, local heads use window only"
    );
}
