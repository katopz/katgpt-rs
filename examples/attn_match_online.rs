//! Online compaction example — Plan 271 Phase 5.
//!
//! Simulates AIME-style long reasoning: generate 4096 synthetic "reasoning"
//! tokens, compacting the prefix once `phys_budget` is reached while keeping
//! the most-recent `recent_window` tokens uncompacted. Shows that the logical
//! sequence length stays bounded while the physical token count grows.
//!
//! Run with:
//! ```bash
//! cargo run --example attn_match_online --features attn_match --release
//! ```

use katgpt_rs::attn_match::{online::OnlineCompactor, types::AmConfig};

fn synth_token(offset: usize, d: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut k = vec![0.0f32; d];
    let mut v = vec![0.0f32; d];
    for j in 0..d {
        let x = ((offset + j * 7 + seed as usize) as f32) * 0.03;
        k[j] = x.sin() * 0.5;
        v[j] = x.cos() * 0.4;
    }
    (k, v)
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
    println!("║  Attention Matching — Online Compaction Demo (Plan 271 Phase 5)        ║");
    println!("║  Paper: arxiv 2602.16284 — Zweiger, Fu, Guo, Kim (MIT, ICML 2026)      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    let d: usize = 64;
    let n: usize = 64;
    let phys_budget: usize = 512;
    let recent_window: usize = 128;
    let compact_size: usize = 256;
    let total_generate: usize = 4096;

    println!(
        "\nConfig: phys_budget = {}, recent_window = {}, compact_size = {}",
        phys_budget, recent_window, compact_size
    );
    println!(
        "Trigger threshold: pos >= {} (phys_budget + recent_window)",
        phys_budget + recent_window
    );
    println!("Will generate {} tokens total.", total_generate);

    let cfg = AmConfig::highest_attn(compact_size);
    let compactor = OnlineCompactor::new(phys_budget, recent_window);
    let queries = synth_queries(n, d, 42);

    // Pre-populate up to phys_budget + recent_window so the first compaction
    // fires immediately on the next generated token.
    let mut keys: Vec<f32> = Vec::with_capacity(total_generate * d);
    let mut values: Vec<f32> = Vec::with_capacity(total_generate * d);
    for i in 0..(phys_budget + recent_window) {
        let (k, v) = synth_token(i, d, i as u32);
        keys.extend_from_slice(&k);
        values.extend_from_slice(&v);
    }
    let mut pos = phys_budget + recent_window;

    println!("\nInitial cache: {} tokens, {} bytes", pos, bytes_for(pos, d));

    let mut compaction_count = 0usize;
    let mut compaction_log: Vec<(usize, usize, usize, usize)> = Vec::new();

    print_separator();
    println!(
        "{:>10} {:>12} {:>12} {:>14} {:>14}",
        "step", "phys_pos", "logical_len", "phys_bytes", "cmp_bytes"
    );

    for step in 0..(total_generate - pos) {
        // Generate one token.
        let (k, v) = synth_token(pos + step, d, (pos + step) as u32);
        keys.extend_from_slice(&k);
        values.extend_from_slice(&v);
        pos += 1;

        // Maybe compact.
        match compactor
            .maybe_compact(&keys, &values, &queries, pos, d, n, &cfg)
            .expect("compact ok")
        {
            Some(r) => {
                compaction_count += 1;
                let phys_bytes_before = bytes_for(pos, d);
                let cmp_bytes = r.total_bytes(d);
                let logical_len = r.total_logical_len;
                compaction_log.push((pos, logical_len, phys_bytes_before, cmp_bytes));

                println!(
                    "{:>10} {:>12} {:>12} {:>14} {:>14}",
                    step + 1, pos, logical_len, phys_bytes_before, cmp_bytes
                );

                // Apply compaction: replace cache with [compact_prefix | recent].
                keys = r.compact_prefix.compact_keys.clone();
                keys.extend_from_slice(&r.recent_keys);
                values = r.compact_prefix.compact_values.clone();
                values.extend_from_slice(&r.recent_values);
                pos = r.total_logical_len;
            }
            None => {
                // Not yet at threshold.
            }
        }
    }

    // ── Summary ────────────────────────────────────────────────────────────
    print_separator();
    println!("Summary");
    println!("  Tokens generated (post initial): {}", total_generate - (phys_budget + recent_window));
    println!("  Compactions triggered: {}", compaction_count);
    println!();

    if !compaction_log.is_empty() {
        println!("  Per-compaction log:");
        println!(
            "  {:>8} {:>14} {:>14} {:>14} {:>14}",
            "cmp#", "phys_before", "logical_after", "phys_bytes", "cmp_bytes"
        );
        for (i, (phys_before, logical_after, pb, cb)) in compaction_log.iter().enumerate() {
            println!(
                "  {:>8} {:>14} {:>14} {:>14} {:>14}",
                i + 1, phys_before, logical_after, pb, cb
            );
        }

        let max_logical = compaction_log
            .iter()
            .map(|(_, l, _, _)| *l)
            .max()
            .unwrap_or(0);
        let max_phys = compaction_log
            .iter()
            .map(|(p, _, _, _)| *p)
            .max()
            .unwrap_or(0);
        println!();
        println!(
            "  Max logical length observed: {} (bound: {})",
            max_logical,
            phys_budget + recent_window
        );
        println!("  Max physical position observed: {}", max_phys);
        println!(
            "  Logical stays bounded at ~phys_budget + recent_window = {} while physical grows.",
            phys_budget + recent_window
        );

        let bytes_first = compaction_log.first().map(|(_, _, pb, _)| *pb).unwrap_or(0);
        let bytes_last = compaction_log.last().map(|(_, _, _, cb)| *cb).unwrap_or(0);
        println!(
            "\n  Memory: peak physical = {} bytes, post-compaction = {} bytes ({:.1}% reduction)",
            bytes_first,
            bytes_last,
            if bytes_first > 0 {
                100.0 * (1.0 - bytes_last as f32 / bytes_first as f32)
            } else {
                0.0
            }
        );
    }

    println!();
    println!("Online compaction keeps KV cache bounded during unbounded generation.");
    println!("Logical sequence length stays at phys_budget + recent_window indefinitely.");
    println!();
    println!("✓ Demo complete. See .plans/271_attention_matching_compaction.md");
}
