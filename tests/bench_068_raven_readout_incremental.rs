//! Plan 068: Raven Readout Zero-Alloc + TurboQuant Incremental Dequant Benchmarks.
//!
//! Benchmarks two optimizations:
//! 1. `raven_readout_into` (zero-alloc, fused 2-pass) vs `raven_readout` (allocating, 3-pass)
//! 2. `forward_turboquant` incremental dequant vs full re-dequant per step
//!
//! Quality gates: outputs must match within 1e-6.
//!
//! Run: cargo test bench_068 -- --nocapture

use std::hint::black_box;
use std::time::Instant;

use microgpt_rs::transformer::{
    ForwardContext, RavenKVCache, TransformerWeights, forward_turboquant, raven_readout,
    raven_readout_into, raven_update,
};
use microgpt_rs::turboquant::TurboQuantKVCache;
use microgpt_rs::types::{Config, Rng, kv_dim};

// ─── Raven Readout Benchmark ─────────────────────────────────────

#[test]
fn bench_068_raven_readout_zero_alloc() {
    let config = Config::micro();
    let num_slots = 32;
    let top_k = 4;
    let kvd = kv_dim(&config);

    let warmup = 1_000;
    let iters = 100_000;

    println!("\n🧪 Plan 068: Raven Readout Zero-Alloc Benchmark");
    println!("   num_slots={num_slots}, kv_dim={kvd}, warmup={warmup}, iters={iters}");
    println!("{}", "═".repeat(70));

    // Setup: populate cache with synthetic data
    let mut cache = RavenKVCache::new(&config, num_slots, top_k);
    for slot in 0..num_slots {
        let k: Vec<f32> = (0..kvd)
            .map(|d| ((d + slot * 7) as f32 * 0.1).sin())
            .collect();
        let v: Vec<f32> = (0..kvd)
            .map(|d| ((d + slot * 3) as f32 * 0.05).cos())
            .collect();
        let mut r_t = vec![0.0f32; num_slots];
        r_t[slot] = 1.0;
        raven_update(
            &mut cache.keys,
            &mut cache.values,
            &k,
            &v,
            &r_t,
            cache.forget_rate,
            num_slots,
            kvd,
        );
    }

    let query: Vec<f32> = (0..kvd).map(|d| (d as f32 * 0.13).sin()).collect();

    // ── Quality gate: outputs must match ──
    let mut scores_buf = vec![0.0f32; num_slots];
    let mut output_buf = vec![0.0f32; kvd];
    let result_alloc = raven_readout(&query, &cache.keys, &cache.values, num_slots, kvd);
    let result_into = raven_readout_into(
        &query,
        &cache.keys,
        &cache.values,
        num_slots,
        kvd,
        &mut scores_buf,
        &mut output_buf,
    );

    let max_diff = result_alloc
        .iter()
        .zip(result_into.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "   Quality gate: max_diff = {max_diff:.2e} ({})",
        if max_diff < 1e-6 {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );
    assert!(
        max_diff < 1e-6,
        "raven_readout_into output diverged: max_diff={max_diff}"
    );

    // ── Benchmark: allocating raven_readout ──
    for _ in 0..warmup {
        black_box(raven_readout(
            &query,
            &cache.keys,
            &cache.values,
            num_slots,
            kvd,
        ));
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(raven_readout(
            &query,
            &cache.keys,
            &cache.values,
            num_slots,
            kvd,
        ));
    }
    let alloc_time = start.elapsed();
    let alloc_us = alloc_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Benchmark: zero-alloc raven_readout_into ──
    for _ in 0..warmup {
        black_box(raven_readout_into(
            &query,
            &cache.keys,
            &cache.values,
            num_slots,
            kvd,
            &mut scores_buf,
            &mut output_buf,
        ));
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(raven_readout_into(
            &query,
            &cache.keys,
            &cache.values,
            num_slots,
            kvd,
            &mut scores_buf,
            &mut output_buf,
        ));
    }
    let into_time = start.elapsed();
    let into_us = into_time.as_nanos() as f64 / iters as f64 / 1000.0;

    let speedup = alloc_us / into_us;
    println!();
    println!("   ┌─────────────────────────┬──────────┬───────────┐");
    println!("   │ Variant                  │ μs/call  │ Speedup   │");
    println!("   ├─────────────────────────┼──────────┼───────────┤");
    println!("   │ raven_readout (alloc)    │ {alloc_us:7.2}  │ baseline  │");
    println!("   │ raven_readout_into       │ {into_us:7.2}  │ {speedup:7.2}×  │");
    println!("   └─────────────────────────┴──────────┴───────────┘");
    println!(
        "   Δ = {:.1}% {}",
        (speedup - 1.0) * 100.0,
        if speedup > 1.0 {
            "faster ✅"
        } else {
            "slower ⚠️"
        }
    );
}

// ─── Incremental Dequant Benchmark ───────────────────────────────

#[test]
fn bench_068_incremental_dequant_full_sequence() {
    let config = Config::micro();
    let kvd = kv_dim(&config);
    let n_positions = config.block_size; // 128

    let warmup = 10;
    let iters = 100;

    println!("\n🧪 Plan 068: Incremental Dequant Benchmark (full sequence decode)");
    println!(
        "   kv_dim={kvd}, n_layer={}, block_size={n_positions}, warmup={warmup}, iters={iters}",
        config.n_layer
    );
    println!("{}", "═".repeat(70));

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // ── Full re-dequant (baseline): reset tq_dequant_pos each step ──
    let mut ctx = ForwardContext::new(&config);
    let mut tq_cache = TurboQuantKVCache::new(&config, 3, 3);

    for _ in 0..warmup {
        ctx.reset_tq_dequant();
        tq_cache.reset();
        let mut local_rng = Rng::new(42);
        for pos in 0..n_positions {
            let token = local_rng.next() as usize % config.vocab_size;
            // Force full re-dequant by resetting tracker each step
            ctx.reset_tq_dequant();
            black_box(forward_turboquant(
                &mut ctx,
                &weights,
                &mut tq_cache,
                token,
                pos,
                &config,
            ));
        }
    }
    let start = Instant::now();
    for _ in 0..iters {
        ctx.reset_tq_dequant();
        tq_cache.reset();
        let mut local_rng = Rng::new(42);
        for pos in 0..n_positions {
            let token = local_rng.next() as usize % config.vocab_size;
            // Force full re-dequant by resetting tracker each step
            ctx.reset_tq_dequant();
            black_box(forward_turboquant(
                &mut ctx,
                &weights,
                &mut tq_cache,
                token,
                pos,
                &config,
            ));
        }
    }
    let full_redequant_time = start.elapsed();
    let full_us = full_redequant_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Incremental dequant: let tq_dequant_pos track naturally ──
    let mut ctx2 = ForwardContext::new(&config);
    let mut tq_cache2 = TurboQuantKVCache::new(&config, 3, 3);

    for _ in 0..warmup {
        ctx2.reset_tq_dequant();
        tq_cache2.reset();
        let mut local_rng = Rng::new(42);
        for pos in 0..n_positions {
            let token = local_rng.next() as usize % config.vocab_size;
            black_box(forward_turboquant(
                &mut ctx2,
                &weights,
                &mut tq_cache2,
                token,
                pos,
                &config,
            ));
        }
    }
    let start = Instant::now();
    for _ in 0..iters {
        ctx2.reset_tq_dequant();
        tq_cache2.reset();
        let mut local_rng = Rng::new(42);
        for pos in 0..n_positions {
            let token = local_rng.next() as usize % config.vocab_size;
            black_box(forward_turboquant(
                &mut ctx2,
                &weights,
                &mut tq_cache2,
                token,
                pos,
                &config,
            ));
        }
    }
    let incremental_time = start.elapsed();
    let incr_us = incremental_time.as_nanos() as f64 / iters as f64 / 1000.0;

    let speedup = full_us / incr_us;
    let per_tok_full = full_us / n_positions as f64;
    let per_tok_incr = incr_us / n_positions as f64;

    println!();
    println!("   ┌──────────────────────────────┬──────────────┬──────────────┐");
    println!("   │ Variant                       │ Total (μs)   │ μs/token     │");
    println!("   ├──────────────────────────────┼──────────────┼──────────────┤");
    println!("   │ Full re-dequant (reset/step)  │ {full_us:10.1}   │ {per_tok_full:10.2}   │");
    println!("   │ Incremental dequant           │ {incr_us:10.1}   │ {per_tok_incr:10.2}   │");
    println!("   └──────────────────────────────┴──────────────┴──────────────┘");
    println!(
        "   Speedup: {speedup:.2}× ({:.1}% {})",
        (speedup - 1.0) * 100.0,
        if speedup > 1.0 {
            "faster ✅"
        } else {
            "slower ⚠️"
        }
    );
}

#[test]
fn bench_068_incremental_dequant_steady_state() {
    let config = Config::micro();
    let kvd = kv_dim(&config);
    let half_pos = config.block_size / 2; // 64

    let warmup = 100;
    let iters = 10_000;

    println!("\n🧪 Plan 068: Incremental Dequant Steady-State Benchmark");
    println!("   Steady pos={half_pos}, kv_dim={kvd}, warmup={warmup}, iters={iters}");
    println!("{}", "═".repeat(70));

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // ── Full re-dequant at pos=64: dequantizes all 65 positions each step ──
    let mut ctx = ForwardContext::new(&config);
    let mut tq_cache = TurboQuantKVCache::new(&config, 3, 3);

    // Prefill up to half_pos
    for pos in 0..half_pos {
        let token = rng.next() as usize % config.vocab_size;
        forward_turboquant(&mut ctx, &weights, &mut tq_cache, token, pos, &config);
    }

    for _ in 0..warmup {
        ctx.reset_tq_dequant(); // Force full rebuild
        let token = rng.next() as usize % config.vocab_size;
        black_box(forward_turboquant(
            &mut ctx,
            &weights,
            &mut tq_cache,
            token,
            half_pos,
            &config,
        ));
    }
    let start = Instant::now();
    for _ in 0..iters {
        ctx.reset_tq_dequant(); // Force full rebuild
        let token = rng.next() as usize % config.vocab_size;
        black_box(forward_turboquant(
            &mut ctx,
            &weights,
            &mut tq_cache,
            token,
            half_pos,
            &config,
        ));
    }
    let full_time = start.elapsed();
    let full_us = full_time.as_nanos() as f64 / iters as f64 / 1000.0;

    // ── Incremental at pos=64: only dequantizes 1 new position ──
    let mut ctx2 = ForwardContext::new(&config);
    let mut tq_cache2 = TurboQuantKVCache::new(&config, 3, 3);
    let mut rng2 = Rng::new(42);

    // Prefill up to half_pos (incremental builds up naturally)
    for pos in 0..half_pos {
        let token = rng2.next() as usize % config.vocab_size;
        forward_turboquant(&mut ctx2, &weights, &mut tq_cache2, token, pos, &config);
    }

    for _ in 0..warmup {
        let token = rng2.next() as usize % config.vocab_size;
        black_box(forward_turboquant(
            &mut ctx2,
            &weights,
            &mut tq_cache2,
            token,
            half_pos,
            &config,
        ));
    }
    let start = Instant::now();
    for _ in 0..iters {
        let token = rng2.next() as usize % config.vocab_size;
        black_box(forward_turboquant(
            &mut ctx2,
            &weights,
            &mut tq_cache2,
            token,
            half_pos,
            &config,
        ));
    }
    let incr_time = start.elapsed();
    let incr_us = incr_time.as_nanos() as f64 / iters as f64 / 1000.0;

    let speedup = full_us / incr_us;

    println!();
    println!("   ┌──────────────────────────────┬──────────────┐");
    println!("   │ Variant (steady pos={half_pos:3})     │ μs/step      │");
    println!("   ├──────────────────────────────┼──────────────┤");
    println!("   │ Full re-dequant ({half_pos:3}+1 pos)    │ {full_us:9.2} μs  │");
    println!("   │ Incremental (1 pos)           │ {incr_us:9.2} μs  │");
    println!("   └──────────────────────────────┴──────────────┘");
    println!(
        "   Speedup: {speedup:.2}× ({:.1}% {})",
        (speedup - 1.0) * 100.0,
        if speedup > 1.0 {
            "faster ✅"
        } else {
            "slower ⚠️"
        }
    );
    println!(
        "   Dequant ops saved: {} → 1 per layer per step",
        half_pos + 1
    );
}

// ─── Quality Gate: Incremental produces identical logits ────────

#[test]
fn bench_068_incremental_dequant_quality() {
    let config = Config::micro();
    let n_positions = 16; // shorter for quality check

    println!("\n🧪 Plan 068: Incremental Dequant Quality Gate");
    println!("   n_positions={n_positions}, checking logit identity");
    println!("{}", "═".repeat(70));

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Run with full re-dequant (reset each step)
    let mut ctx_full = ForwardContext::new(&config);
    let mut tq_full = TurboQuantKVCache::new(&config, 3, 3);
    let mut rng_full = Rng::new(42);

    let mut logits_full: Vec<Vec<f32>> = Vec::new();
    for pos in 0..n_positions {
        let token = rng_full.next() as usize % config.vocab_size;
        ctx_full.reset_tq_dequant(); // Force full rebuild
        let result = forward_turboquant(&mut ctx_full, &weights, &mut tq_full, token, pos, &config);
        logits_full.push(result.to_vec());
    }

    // Run with incremental dequant (natural tracking)
    let mut ctx_incr = ForwardContext::new(&config);
    let mut tq_incr = TurboQuantKVCache::new(&config, 3, 3);
    let mut rng_incr = Rng::new(42);

    let mut logits_incr: Vec<Vec<f32>> = Vec::new();
    for pos in 0..n_positions {
        let token = rng_incr.next() as usize % config.vocab_size;
        let result = forward_turboquant(&mut ctx_incr, &weights, &mut tq_incr, token, pos, &config);
        logits_incr.push(result.to_vec());
    }

    // Compare all logits
    let mut max_diff_overall = 0.0f32;
    let mut worst_pos = 0;
    let mut worst_idx = 0;

    for (pos, (full, incr)) in logits_full.iter().zip(logits_incr.iter()).enumerate() {
        let pos_max_diff = full
            .iter()
            .zip(incr.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if pos_max_diff > max_diff_overall {
            max_diff_overall = pos_max_diff;
            worst_pos = pos;
            // Find worst index
            for (i, (a, b)) in full.iter().zip(incr.iter()).enumerate() {
                if (a - b).abs() >= max_diff_overall * 0.99 {
                    worst_idx = i;
                }
            }
        }
        print!(
            "   pos {:>3}: max_diff = {:.2e} {}\n",
            pos,
            pos_max_diff,
            if pos_max_diff < 1e-5 { "✅" } else { "⚠️" }
        );
    }

    println!();
    println!(
        "   Overall: max_diff = {max_diff_overall:.2e} at pos={worst_pos} logit[{worst_idx}] ({})",
        if max_diff_overall < 1e-5 {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );
    assert!(
        max_diff_overall < 1e-5,
        "Incremental dequant logits diverged: max_diff={max_diff_overall} at pos={worst_pos}"
    );
}
