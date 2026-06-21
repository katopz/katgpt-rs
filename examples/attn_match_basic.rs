//! Attention Matching basic example — before/after KV cache compaction.
//!
//! Demonstrates the AM pipeline on synthetic data:
//! 1. Generate a synthetic KV cache with cluster structure
//! 2. Generate synthetic reference queries
//! 3. Compact with all three selectors (HighestAttn, OMP, OMP-fast)
//! 4. Print reconstruction error, β distribution, memory savings
//!
//! Run with:
//! ```bash
//! cargo run --example attn_match_basic --features attn_match --release
//! ```

use katgpt_rs::attn_match::{
    compact::compact,
    types::{AmConfig, KeySelector},
};

fn synth_block_kv(t_len: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let mut keys = vec![0.0f32; t_len * d];
    let mut values = vec![0.0f32; t_len * d];
    let half = t_len / 2;
    for i in 0..t_len {
        let block_id = if i < half { 0 } else { 1 };
        for k in 0..d {
            let sign = if block_id == 0 { 1.0 } else { -1.0 };
            let x = ((i + k * 7) as f32) * 0.05;
            keys[i * d + k] = sign * (0.5 + x.sin() * 0.3);
            values[i * d + k] = sign * (1.0 + x.cos() * 0.4);
        }
    }
    (keys, values)
}

fn synth_queries(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut q = vec![0.0f32; n * d];
    let mut state = seed;
    for v in q.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = ((state >> 33) as f32) / (1u64 << 31) as f32 - 0.5;
        *v = r * 0.5;
    }
    q
}

fn print_separator() {
    println!("{}", "-".repeat(78));
}

fn run_compaction(
    label: &str,
    keys: &[f32],
    values: &[f32],
    queries: &[f32],
    shape: (usize, usize, usize, usize),
    selector: KeySelector,
) {
    let (t_len, d, n, t) = shape;
    let mut cfg = match selector {
        KeySelector::HighestAttnKeys => AmConfig::highest_attn(t),
        KeySelector::Omp => AmConfig::omp(t),
        KeySelector::OmpFast => AmConfig::omp_fast(t),
    };
    cfg.selector = selector;
    cfg.report_reconstruction = true;

    let start = std::time::Instant::now();
    let result = compact(keys, values, queries, t_len, d, n, &cfg).expect("compact failed");
    let elapsed = start.elapsed();

    println!("\n{}: selector = {:?}", label, selector);
    println!(
        "  Original T = {}, compact t = {}, compression = {:.1}×",
        t_len,
        result.compact_len,
        result.compression_ratio()
    );
    println!("  Wall-clock: {:?}", elapsed);

    let report = result.report.as_ref().expect("report should be present");
    println!(
        "  Reconstruction: attn-output rel error = {:.4}, mass rel error = {:.4}",
        report.relative_attn_output_error, report.relative_mass_error
    );
    println!(
        "  Selected mass coverage (RMS): {:.4}",
        report.selected_mass_coverage
    );

    let beta_min = result.beta.iter().copied().fold(f32::INFINITY, f32::min);
    let beta_max = result
        .beta
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let beta_mean = result.beta.iter().copied().sum::<f32>() / result.beta.len() as f32;
    println!(
        "  β stats: min = {:.4}, max = {:.4}, mean = {:.4} (paper: log(T/t) baseline = {:.4})",
        beta_min,
        beta_max,
        beta_mean,
        ((t_len as f32) / (t as f32)).ln()
    );

    let original_bytes = t_len * d * 2 * std::mem::size_of::<f32>();
    let compact_bytes = t * (d * 2 + 1) * std::mem::size_of::<f32>();
    println!(
        "  Memory: original = {} bytes, compact = {} bytes, saved = {} bytes ({:.1}% reduction)",
        original_bytes,
        compact_bytes,
        original_bytes.saturating_sub(compact_bytes),
        100.0 * (1.0 - (compact_bytes as f32) / (original_bytes as f32))
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Attention Matching KV Cache Compaction — modelless demo (Plan 271)    ║");
    println!("║  Paper: arxiv 2602.16284 — Zweiger, Fu, Guo, Kim (MIT, ICML 2026)      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    let t_len: usize = 512;
    let d: usize = 64;
    let n: usize = 128;
    let t: usize = 64; // 8× compression

    println!(
        "\nSynthetic config: T = {} tokens, d = {} head_dim, n = {} reference queries",
        t_len, d, n
    );
    println!("Target compaction: t = {} tokens ({:.1}× compression)", t, (t_len as f32 / t as f32));

    let (keys, values) = synth_block_kv(t_len, d);
    let queries = synth_queries(n, d, 42);

    print_separator();
    run_compaction(
        "Pipeline 1",
        &keys,
        &values,
        &queries,
        (t_len, d, n, t),
        KeySelector::HighestAttnKeys,
    );
    print_separator();
    run_compaction(
        "Pipeline 2",
        &keys,
        &values,
        &queries,
        (t_len, d, n, t),
        KeySelector::Omp,
    );
    print_separator();
    run_compaction(
        "Pipeline 3",
        &keys,
        &values,
        &queries,
        (t_len, d, n, t),
        KeySelector::OmpFast,
    );
    print_separator();

    println!("\nAll three selectors produce compact (Ck, β, Cv) for the same input.");
    println!("OMP variants produce lower reconstruction error at slight compute cost.");
    println!("β mean should approach log(T/t) = log({:.1}) = {:.4} for uniform compaction.",
        (t_len as f32 / t as f32),
        ((t_len as f32) / (t as f32)).ln()
    );

    println!("\n✓ Demo complete. See .research/233_Attention_Matching_KV_Compaction.md");
    println!("  and .plans/271_attention_matching_compaction.md for full plan.");
}
