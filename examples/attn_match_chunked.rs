//! Chunked compaction example — Plan 271 Phase 4.
//!
//! Demonstrates both KV-based (overlap-aware) and text-based (position-aware)
//! chunked compaction on a synthetic long context (T=8192, d=64, 4 chunks of
//! 2048). Prints per-chunk reconstruction errors, total compacted length, and
//! memory savings.
//!
//! Run with:
//! ```bash
//! cargo run --example attn_match_chunked --features attn_match --release
//! ```
//!
//! For full RoPE preservation in text-based mode, also enable `still_kv`:
//! ```bash
//! cargo run --example attn_match_chunked --features attn_match,still_kv --release
//! ```

use katgpt_rs::attn_match::{
    chunked::{ChunkedCompactor, TextChunk},
    types::AmConfig,
};

fn synth_block_kv(t_len: usize, d: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
    // 4 semantic blocks so attention has non-trivial structure.
    let mut keys = vec![0.0f32; t_len * d];
    let mut values = vec![0.0f32; t_len * d];
    let block = t_len / 4;
    for i in 0..t_len {
        let b = i / block;
        for k in 0..d {
            let sign = if b.is_multiple_of(2) { 1.0 } else { -1.0 };
            let x = ((i + b * 13 + k * 7 + seed as usize) as f32) * 0.03;
            keys[i * d + k] = sign * (0.4 + x.sin() * 0.3);
            values[i * d + k] = sign * (0.8 + x.cos() * 0.5);
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
        *v = r * 0.4;
    }
    q
}

fn bytes_for(tokens: usize, d: usize) -> usize {
    tokens * d * 2 * std::mem::size_of::<f32>()
}

fn print_separator() {
    println!("{}", "-".repeat(78));
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Attention Matching — Chunked Compaction Demo (Plan 271 Phase 4)       ║");
    println!("║  Paper: arxiv 2602.16284 — Zweiger, Fu, Guo, Kim (MIT, ICML 2026)      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    let t_len: usize = 8192;
    let d: usize = 64;
    let n: usize = 128;
    let chunk_size: usize = 2048;
    let overlap: usize = 64;
    let compact_per_chunk: usize = 256; // 8× per chunk

    println!(
        "\nConfig: T = {} tokens, d = {}, n = {} reference queries",
        t_len, d, n
    );
    println!(
        "Chunking: chunk_size = {}, overlap = {}, compact_per_chunk = {} ({:.1}× per chunk)",
        chunk_size,
        overlap,
        compact_per_chunk,
        chunk_size as f32 / compact_per_chunk as f32
    );

    let (keys, values) = synth_block_kv(t_len, d, 42);
    let queries = synth_queries(n, d, 42);

    let cfg = AmConfig::highest_attn(compact_per_chunk);
    let original_bytes = bytes_for(t_len, d);

    // ── KV-based ────────────────────────────────────────────────────────────
    print_separator();
    println!("[1] KV-based chunked compaction (overlap = {})", overlap);
    let compactor_kv = ChunkedCompactor::new(chunk_size, overlap);
    let start = std::time::Instant::now();
    let out_kv = compactor_kv
        .compact_kv_based(&keys, &values, &queries, t_len, d, n, &cfg)
        .expect("kv-based compact");
    let elapsed_kv = start.elapsed();

    println!("  Wall-clock: {:?}", elapsed_kv);
    println!("  Chunks processed: {}", out_kv.per_chunk.len());
    println!(
        "  Total compact length: {} (compression {:.1}×)",
        out_kv.total_compact_len,
        t_len as f32 / out_kv.total_compact_len as f32
    );

    println!("\n  Per-chunk reconstruction errors:");
    println!("  {:>8} {:>10} {:>10} {:>14}", "chunk", "src_len", "cmp_len", "rel_error");
    for (i, m) in out_kv.per_chunk.iter().enumerate() {
        println!(
            "  {:>8} {:>10} {:>10} {:>14.6}",
            i, m.chunk_len, m.compact_len, m.reconstruction_error
        );
    }
    println!(
        "\n  Mean reconstruction error: {:.6}",
        out_kv.mean_reconstruction_error()
    );
    println!(
        "  Boundary (first+last) error: {:.6}",
        out_kv.boundary_reconstruction_error()
    );

    let compact_bytes_kv = out_kv.total_compact_len * (2 * d + 1) * std::mem::size_of::<f32>();
    println!(
        "\n  Memory: original = {} bytes, compact = {} bytes, saved = {} bytes ({:.1}% reduction)",
        original_bytes,
        compact_bytes_kv,
        original_bytes.saturating_sub(compact_bytes_kv),
        100.0 * (1.0 - compact_bytes_kv as f32 / original_bytes as f32)
    );

    // ── Text-based ─────────────────────────────────────────────────────────
    print_separator();
    println!("[2] Text-based chunked compaction (per-chunk positions preserved)");
    let chunks: Vec<TextChunk> = (0..4)
        .map(|i| {
            let start_pos = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(t_len);
            let len = end - start_pos;
            TextChunk {
                keys: keys[start_pos * d..end * d].to_vec(),
                values: values[start_pos * d..end * d].to_vec(),
                start_pos,
                chunk_len: len,
            }
        })
        .collect();
    let queries_per_chunk: Vec<Vec<f32>> = (0..4).map(|_| queries.clone()).collect();

    let compactor_txt = ChunkedCompactor::new(chunk_size, 0);
    let start = std::time::Instant::now();
    let out_txt = compactor_txt
        .compact_text_based(&chunks, &queries_per_chunk, &cfg)
        .expect("text-based compact");
    let elapsed_txt = start.elapsed();

    println!("  Wall-clock: {:?}", elapsed_txt);
    println!("  Chunks processed: {}", out_txt.per_chunk.len());
    println!(
        "  Total compact length: {} (compression {:.1}×)",
        out_txt.total_compact_len,
        t_len as f32 / out_txt.total_compact_len as f32
    );

    println!("\n  Per-chunk reconstruction errors:");
    println!(
        "  {:>8} {:>10} {:>10} {:>12} {:>14}",
        "chunk", "src_len", "cmp_len", "start_pos", "rel_error"
    );
    for (i, m) in out_txt.per_chunk.iter().enumerate() {
        println!(
            "  {:>8} {:>10} {:>10} {:>12} {:>14.6}",
            i, m.chunk_len, m.compact_len, m.chunk_start, m.reconstruction_error
        );
    }
    println!(
        "\n  Mean reconstruction error: {:.6}",
        out_txt.mean_reconstruction_error()
    );

    let compact_bytes_txt = out_txt.total_compact_len * (2 * d + 1) * std::mem::size_of::<f32>();
    println!(
        "\n  Memory: original = {} bytes, compact = {} bytes, saved = {} bytes ({:.1}% reduction)",
        original_bytes,
        compact_bytes_txt,
        original_bytes.saturating_sub(compact_bytes_txt),
        100.0 * (1.0 - compact_bytes_txt as f32 / original_bytes as f32)
    );

    // ── Comparison ─────────────────────────────────────────────────────────
    print_separator();
    println!("[3] KV-based vs Text-based comparison");
    println!(
        "  {:>30} {:>15} {:>15}",
        "", "KV-based", "Text-based"
    );
    println!(
        "  {:>30} {:>15} {:>15}",
        "total_compact_len",
        out_kv.total_compact_len,
        out_txt.total_compact_len
    );
    println!(
        "  {:>30} {:>15.6} {:>15.6}",
        "mean recon error",
        out_kv.mean_reconstruction_error(),
        out_txt.mean_reconstruction_error()
    );
    println!(
        "  {:>30} {:>15.6} {:>15.6}",
        "boundary recon error",
        out_kv.boundary_reconstruction_error(),
        out_txt.boundary_reconstruction_error()
    );

    println!();
    println!("KV-based with overlap captures boundary context → lower boundary error.");
    println!("Text-based preserves per-chunk positions → RoPE-consistent for downstream.");
    println!();
    println!("✓ Demo complete. See .plans/271_attention_matching_compaction.md");
}
