//! GOAT Benchmark 023: Block-Diagonal Rotation (PlanarQuant & IsoQuant) vs OCTOPUS vs TurboQuant.
//!
//! Plan 100 Tasks T12-T14: synthetic quality sweep + rotation cost comparison.
//!
//! Metrics:
//! 1. Reconstruction MSE (per-coordinate) — ↓ better
//! 2. Cosine similarity (original vs reconstructed) — ↑ better
//! 3. Inner-product absolute error — ↓ better
//! 4. Compression ratio vs f32 baseline
//! 5. Rotation FMAs (theoretical)
//! 6. Parameter count
//!
//! Run with:
//!   cargo test -p katgpt-rs --features "planar_quant,iso_quant,octopus,turboquant" --test bench_block_diagonal_goat -- --nocapture
//!
//! Partial run (only planar_quant + octopus):
//!   cargo test -p katgpt-rs --features "planar_quant,octopus,turboquant" --test bench_block_diagonal_goat -- --nocapture

#![cfg(any(
    feature = "planar_quant",
    feature = "iso_quant",
    feature = "hybrid_oct_pq"
))]

use katgpt_core::Rng;

// ── Per-coord helpers (duplicated from octopus/forward.rs to avoid feature gate coupling) ──

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    dot / (na * nb)
}

fn per_coord_mse(original: &[f32], reconstructed: &[f32]) -> f32 {
    let n = original.len() as f32;
    original
        .iter()
        .zip(reconstructed)
        .map(|(o, r)| (o - r) * (o - r))
        .sum::<f32>()
        / n
}

fn ip_error(a: &[f32], b: &[f32], query: &[f32]) -> f32 {
    let ip_orig: f32 = a.iter().zip(query).map(|(x, q)| x * q).sum();
    let ip_recon: f32 = b.iter().zip(query).map(|(x, q)| x * q).sum();
    (ip_orig - ip_recon).abs()
}

fn gaussian_vec(dim: usize, rng: &mut Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.normal()).collect()
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    if n < 1.0 {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

// ── Backend abstraction ──

/// Quality metrics for one backend at one (d, bits, seed) combo.
#[derive(Default)]
struct BackendResult {
    mse: Vec<f64>,
    cos: Vec<f64>,
    ip_err: Vec<f64>,
}

impl BackendResult {
    fn add(&mut self, mse: f64, cos: f64, ip: f64) {
        self.mse.push(mse);
        self.cos.push(cos);
        self.ip_err.push(ip);
    }

    fn summary(&self) -> (f64, f64, f64, f64) {
        let (mse_m, _) = mean_std(&self.mse);
        let (cos_m, _) = mean_std(&self.cos);
        let (ip_m, _) = mean_std(&self.ip_err);
        (mse_m, 0.0, cos_m, ip_m)
    }
}

// ── PlanarQuant backend ──

#[cfg(feature = "planar_quant")]
fn bench_planar_quant(
    dim: usize,
    bits: u8,
    seed: u64,
    keys: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> BackendResult {
    use katgpt_rs::planar_quant::{PlanarQuantConfig, PlanarQuantKVCache};

    let n_keys = keys.len();
    let config = PlanarQuantConfig {
        key_bits: bits,
        val_bits: bits,
        seed,
        n_layers: 1,
        kv_dim: dim,
        max_seq_len: n_keys + 16,
    };
    let mut cache = PlanarQuantKVCache::with_config(&config);

    for (pos, key) in keys.iter().enumerate() {
        cache.store_key(0, pos, key);
    }

    let mut result = BackendResult::default();
    for (pos, orig) in keys.iter().enumerate() {
        let recon = cache.dequantize_key(0, pos);
        result.add(
            per_coord_mse(orig, &recon) as f64,
            cosine_similarity(orig, &recon) as f64,
            0.0,
        );
    }

    // IP error
    for q in queries {
        for (pos, orig) in keys.iter().enumerate() {
            let recon = cache.dequantize_key(0, pos);
            result.ip_err.push(ip_error(orig, &recon, q) as f64);
        }
    }

    result
}

// ── IsoQuant backend ──

#[cfg(feature = "iso_quant")]
fn bench_iso_quant(
    dim: usize,
    bits: u8,
    seed: u64,
    keys: &[Vec<f32>],
    queries: &[Vec<f32>],
    mode: katgpt_rs::iso_quant::IsoQuantMode,
) -> BackendResult {
    use katgpt_rs::iso_quant::{IsoQuantConfig, IsoQuantKVCache};

    let n_keys = keys.len();
    let config = IsoQuantConfig {
        key_bits: bits,
        val_bits: bits,
        seed,
        n_layers: 1,
        kv_dim: dim,
        max_seq_len: n_keys + 16,
        mode,
    };
    let mut cache = IsoQuantKVCache::new(&config);

    for (pos, key) in keys.iter().enumerate() {
        cache.store_key(0, pos, key);
    }

    let mut result = BackendResult::default();
    for (pos, orig) in keys.iter().enumerate() {
        let recon = cache.dequantize_key(0, pos);
        result.add(
            per_coord_mse(orig, &recon) as f64,
            cosine_similarity(orig, &recon) as f64,
            0.0,
        );
    }

    for q in queries {
        for (pos, orig) in keys.iter().enumerate() {
            let recon = cache.dequantize_key(0, pos);
            result.ip_err.push(ip_error(orig, &recon, q) as f64);
        }
    }

    result
}

// ── OCTOPUS backend ──

#[cfg(feature = "octopus")]
fn bench_octopus(
    dim: usize,
    bits: u8,
    seed: u64,
    keys: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> BackendResult {
    use katgpt_rs::octopus::{OctopusConfig, OctopusKVCache};

    let n_keys = keys.len();
    let config = OctopusConfig {
        key_bits: bits,
        val_bits: bits,
        seed,
        n_layers: 1,
        kv_dim: dim,
        max_seq_len: n_keys + 16,
        use_qjl_residual: false,
        use_joint_rounding: true,
    };
    let mut cache = OctopusKVCache::with_config(&config);

    for (pos, key) in keys.iter().enumerate() {
        cache.store_key(0, pos, key);
    }

    let mut result = BackendResult::default();
    for (pos, orig) in keys.iter().enumerate() {
        let recon = cache.dequantize_key(0, pos);
        result.add(
            per_coord_mse(orig, &recon) as f64,
            cosine_similarity(orig, &recon) as f64,
            0.0,
        );
    }

    for q in queries {
        for (pos, orig) in keys.iter().enumerate() {
            let recon = cache.dequantize_key(0, pos);
            result.ip_err.push(ip_error(orig, &recon, q) as f64);
        }
    }

    result
}

// ── TurboQuant backend ──

#[cfg(feature = "turboquant")]
fn bench_turboquant(
    dim: usize,
    bits: u8,
    seed: u64,
    keys: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> BackendResult {
    use katgpt_rs::turboquant::{TurboQuantKVCache, TurboQuantKVCacheConfig};

    let n_keys = keys.len();
    let config = TurboQuantKVCacheConfig {
        key_bits: bits,
        val_bits: bits,
        seed,
        n_layers: 1,
        kv_dim: dim,
        max_seq_len: n_keys + 16,
    };
    let mut cache = TurboQuantKVCache::with_config(&config);

    for (pos, key) in keys.iter().enumerate() {
        cache.store_key(0, pos, key);
    }

    let mut result = BackendResult::default();
    for (pos, orig) in keys.iter().enumerate() {
        let recon = cache.dequantize_key(0, pos);
        result.add(
            per_coord_mse(orig, &recon) as f64,
            cosine_similarity(orig, &recon) as f64,
            0.0,
        );
    }

    for q in queries {
        for (pos, orig) in keys.iter().enumerate() {
            let recon = cache.dequantize_key(0, pos);
            result.ip_err.push(ip_error(orig, &recon, q) as f64);
        }
    }

    result
}

// ── Hybrid OCT+PQ backend ──

#[cfg(feature = "hybrid_oct_pq")]
fn bench_hybrid_oct_pq(
    dim: usize,
    bits: u8,
    seed: u64,
    keys: &[Vec<f32>],
    queries: &[Vec<f32>],
) -> BackendResult {
    use katgpt_rs::hybrid_oct_pq::{HybridOctPqConfig, HybridOctPqKVCache};

    let n_keys = keys.len();
    let config = HybridOctPqConfig {
        key_bits: bits,
        val_bits: bits,
        seed,
        n_layers: 1,
        kv_dim: dim,
        max_seq_len: n_keys + 16,
        use_joint_rounding: true,
    };
    let mut cache = HybridOctPqKVCache::with_config(&config);

    for (pos, key) in keys.iter().enumerate() {
        cache.store_key(0, pos, key);
    }

    let mut result = BackendResult::default();
    for (pos, orig) in keys.iter().enumerate() {
        let recon = cache.dequantize_key(0, pos);
        result.add(
            per_coord_mse(orig, &recon) as f64,
            cosine_similarity(orig, &recon) as f64,
            0.0,
        );
    }

    for q in queries {
        for (pos, orig) in keys.iter().enumerate() {
            let recon = cache.dequantize_key(0, pos);
            result.ip_err.push(ip_error(orig, &recon, q) as f64);
        }
    }

    result
}

// ══════════════════════════════════════════════════════════════
// GOAT Test 1: Main quality sweep — all backends at d=128
// ══════════════════════════════════════════════════════════════

#[test]
fn goat_block_diagonal_quality_sweep() {
    let dim = 128usize;
    let bits_list = [2u8, 3, 4];
    let n_keys = 512;
    let n_queries = 64;
    let n_seeds = 8;

    println!("\n🧪 GOAT 023: Block-Diagonal Rotation Quality Sweep (d={dim})");
    println!("{}", "═".repeat(110));
    println!("Config: {n_keys} Gaussian keys, {n_queries} Gaussian queries, {n_seeds} seeds");
    println!();

    // Generate shared test data across all seeds
    let mut all_keys: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut all_queries: Vec<Vec<Vec<f32>>> = Vec::new();
    for seed in 0..n_seeds {
        let mut rng = Rng::new(seed as u64 * 7919 + 13);
        let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();
        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|_| gaussian_vec(dim, &mut rng))
            .collect();
        all_keys.push(keys);
        all_queries.push(queries);
    }

    // Header
    println!(
        "{:<5} │ {:<10} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} {:>10}",
        "bits", "Backend", "MSE", "Cos", "MSE", "Cos", "MSE", "Cos", "MSE", "Cos", "MSE", "Cos",
    );
    println!(
        "      │            │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} {:>10}",
        "─── TQ ───",
        "─────────",
        "── OCT ──",
        "─────────",
        "── PQ ──",
        "─────────",
        "── IQ-F ──",
        "─────────",
        "── IQ-R ──",
        "─────────",
    );
    println!("{}", "─".repeat(110));

    for &bits in &bits_list {
        let mut line = format!("{:<5} │ ", bits);

        // TurboQuant
        #[cfg(feature = "turboquant")]
        {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_turboquant(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &all_keys[seed],
                    &all_queries[seed],
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            let (mse, _, cos, _) = agg.summary();
            line.push_str(&format!("{:<10} │ {:>10.6} {:>10.6} │", "TQ", mse, cos));
        }
        #[cfg(not(feature = "turboquant"))]
        {
            line.push_str(&format!("{:<10} │ {:>10} {:>10} │", "TQ", "—", "—"));
        }

        // OCTOPUS
        #[cfg(feature = "octopus")]
        {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_octopus(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &all_keys[seed],
                    &all_queries[seed],
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            let (mse, _, cos, _) = agg.summary();
            line.push_str(&format!(" {:>10.6} {:>10.6} │", mse, cos));
        }
        #[cfg(not(feature = "octopus"))]
        {
            line.push_str(&format!(" {:>10} {:>10} │", "—", "—"));
        }

        // PlanarQuant
        #[cfg(feature = "planar_quant")]
        {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_planar_quant(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &all_keys[seed],
                    &all_queries[seed],
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            let (mse, _, cos, _) = agg.summary();
            line.push_str(&format!(" {:>10.6} {:>10.6} │", mse, cos));
        }
        #[cfg(not(feature = "planar_quant"))]
        {
            line.push_str(&format!(" {:>10} {:>10} │", "—", "—"));
        }

        // IsoQuant Full
        #[cfg(feature = "iso_quant")]
        {
            use katgpt_rs::iso_quant::IsoQuantMode;
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_iso_quant(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &all_keys[seed],
                    &all_queries[seed],
                    IsoQuantMode::Full,
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            let (mse, _, cos, _) = agg.summary();
            line.push_str(&format!(" {:>10.6} {:>10.6} │", mse, cos));
        }
        #[cfg(not(feature = "iso_quant"))]
        {
            line.push_str(&format!(" {:>10} {:>10} │", "—", "—"));
        }

        // IsoQuant Fast
        #[cfg(feature = "iso_quant")]
        {
            use katgpt_rs::iso_quant::IsoQuantMode;
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_iso_quant(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &all_keys[seed],
                    &all_queries[seed],
                    IsoQuantMode::Fast,
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            let (mse, _, cos, _) = agg.summary();
            line.push_str(&format!(" {:>10.6} {:>10.6} │", mse, cos));
        }
        #[cfg(not(feature = "iso_quant"))]
        {
            line.push_str(&format!(" {:>10} {:>10} │", "—", "—"));
        }

        println!("{line}");
    }
    println!();
}

// ══════════════════════════════════════════════════════════════
// GOAT Test 2: Pairwise comparison table (d=128, bits=3)
// ══════════════════════════════════════════════════════════════

#[test]
fn goat_block_diagonal_pairwise_comparison() {
    let dim = 128usize;
    let bits = 3u8;
    let n_keys = 512;
    let n_queries = 64;
    let n_seeds = 8;

    println!("\n📊 GOAT 023: Pairwise Comparison (d={dim}, bits={bits})");
    println!("{}", "═".repeat(90));

    let mut all_keys: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut all_queries: Vec<Vec<Vec<f32>>> = Vec::new();
    for seed in 0..n_seeds {
        let mut rng = Rng::new(seed as u64 * 7919 + 13);
        let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();
        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|_| gaussian_vec(dim, &mut rng))
            .collect();
        all_keys.push(keys);
        all_queries.push(queries);
    }

    // Collect results per backend
    #[allow(dead_code)]
    struct BackendMetrics {
        name: &'static str,
        mse: f64,
        mse_std: f64,
        cos: f64,
        cos_std: f64,
        ip: f64,
    }

    let mut backends: Vec<BackendMetrics> = Vec::new();

    #[cfg(feature = "turboquant")]
    {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_turboquant(
                dim,
                bits,
                seed as u64 * 1000 + 42,
                &all_keys[seed],
                &all_queries[seed],
            );
            let (mse_m, _, cos_m, ip_m) = r.summary();
            agg.add(mse_m, cos_m, ip_m);
        }
        let (mse, mse_std, cos, ip) = agg.summary();
        backends.push(BackendMetrics {
            name: "TurboQuant",
            mse,
            mse_std,
            cos,
            cos_std: 0.0,
            ip,
        });
    }

    #[cfg(feature = "octopus")]
    {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_octopus(
                dim,
                bits,
                seed as u64 * 1000 + 42,
                &all_keys[seed],
                &all_queries[seed],
            );
            let (mse_m, _, cos_m, ip_m) = r.summary();
            agg.add(mse_m, cos_m, ip_m);
        }
        let (mse, mse_std, cos, ip) = agg.summary();
        backends.push(BackendMetrics {
            name: "OCTOPUS",
            mse,
            mse_std,
            cos,
            cos_std: 0.0,
            ip,
        });
    }

    #[cfg(feature = "planar_quant")]
    {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_planar_quant(
                dim,
                bits,
                seed as u64 * 1000 + 42,
                &all_keys[seed],
                &all_queries[seed],
            );
            let (mse_m, _, cos_m, ip_m) = r.summary();
            agg.add(mse_m, cos_m, ip_m);
        }
        let (mse, mse_std, cos, ip) = agg.summary();
        backends.push(BackendMetrics {
            name: "PlanarQuant",
            mse,
            mse_std,
            cos,
            cos_std: 0.0,
            ip,
        });
    }

    #[cfg(feature = "iso_quant")]
    {
        use katgpt_rs::iso_quant::IsoQuantMode;
        for (mode, name) in [
            (IsoQuantMode::Full, "IsoQuant-F"),
            (IsoQuantMode::Fast, "IsoQuant-R"),
        ] {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_iso_quant(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &all_keys[seed],
                    &all_queries[seed],
                    mode,
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            let (mse, mse_std, cos, ip) = agg.summary();
            backends.push(BackendMetrics {
                name,
                mse,
                mse_std,
                cos,
                cos_std: 0.0,
                ip,
            });
        }
    }

    if backends.is_empty() {
        println!(
            "No backends enabled. Enable planar_quant, iso_quant, octopus, or turboquant features."
        );
        return;
    }

    // Find MSE winner for reference
    let min_mse = backends.iter().map(|b| b.mse).fold(f64::INFINITY, f64::min);
    let max_cos = backends
        .iter()
        .map(|b| b.cos)
        .fold(f64::NEG_INFINITY, f64::max);

    println!(
        "{:<14} │ {:>12} │ {:>8} │ {:>10} │ {:>8} │ {:>7}",
        "Backend", "MSE", "MSE Δ%", "Cos", "Cos Δ%", "IP Err",
    );
    println!("{}", "─".repeat(90));

    for b in &backends {
        let mse_pct = if min_mse > 0.0 {
            (b.mse - min_mse) / min_mse * 100.0
        } else {
            0.0
        };
        let cos_pct = if max_cos > 0.0 {
            (b.cos - max_cos) / max_cos * 100.0
        } else {
            0.0
        };
        let mse_marker = if (b.mse - min_mse).abs() < 1e-10 {
            " ★"
        } else {
            ""
        };
        let cos_marker = if (b.cos - max_cos).abs() < 1e-10 {
            " ★"
        } else {
            ""
        };
        println!(
            "{:<14} │ {:>10.6}{mse_marker} │ {:>+7.1}% │ {:>8.6}{cos_marker} │ {:>+7.2}% │ {:>9.4}",
            b.name, b.mse, mse_pct, b.cos, cos_pct, b.ip,
        );
    }
    println!("  ★ = best in column");
    println!();
}

// ══════════════════════════════════════════════════════════════
// GOAT Test 3: Rotation cost comparison (theoretical)
// ══════════════════════════════════════════════════════════════

#[test]
fn goat_rotation_cost_comparison() {
    println!("\n⚡ GOAT 023: Rotation Cost Comparison");
    println!("{}", "═".repeat(80));

    let dims = [64usize, 128, 256, 512];

    println!(
        "{:<6} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} {:>10}",
        "d",
        "TQ FMAs",
        "TQ Params",
        "PQ FMAs",
        "PQ Params",
        "IQ-F FMAs",
        "IQ-F Params",
        "IQ-R FMAs",
        "IQ-R Params",
    );
    println!("{}", "─".repeat(80));

    for &dim in &dims {
        // TurboQuant: d×d rotation matrix
        let tq_fmas = dim * dim;
        let tq_params = dim * dim;

        // PlanarQuant: ceil(d/2) groups, 4 FMAs per group (2 for forward)
        let pq_groups = dim.div_ceil(2);
        let pq_fmas = pq_groups * 4; // 2 FMAs × 2 components per group
        let pq_params = pq_groups * 2; // (cos, sin) per group

        // IsoQuant Full: ceil(d/4) groups, 2 Hamilton products × 16 FMAs = 32 FMAs per group
        let iq_groups = dim.div_ceil(4);
        let iq_full_fmas = iq_groups * 32;
        let iq_full_params = iq_groups * 4 * 2; // q_L + q_R, 4 components each

        // IsoQuant Fast: ceil(d/4) groups, 1 Hamilton product × 16 FMAs per group
        let iq_fast_fmas = iq_groups * 16;
        let iq_fast_params = iq_groups * 4; // q_L only

        let tq_pq_ratio = tq_fmas as f64 / pq_fmas as f64;
        let tq_iq_ratio = tq_fmas as f64 / iq_full_fmas as f64;

        println!(
            "{:<6} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} {:>10}",
            dim,
            tq_fmas,
            tq_params,
            pq_fmas,
            pq_params,
            iq_full_fmas,
            iq_full_params,
            iq_fast_fmas,
            iq_fast_params,
        );
        println!(
            "       │ PQ/TQ={:.0}× faster │ IQ-F/TQ={:.0}× faster │ IQ-R/TQ={:.0}× faster",
            tq_pq_ratio,
            tq_iq_ratio,
            tq_fmas as f64 / iq_fast_fmas as f64,
        );
    }

    println!();
    println!(
        "Key: TQ=TurboQuant (WHT), PQ=PlanarQuant (2D Givens), IQ-F=IsoQuant Full (4D quat), IQ-R=IsoQuant Reduced (4D quat left-only)"
    );
    println!();
}

// ══════════════════════════════════════════════════════════════
// GOAT Test 4: PlanarQuant vs OCTOPUS head-to-head at d=128
// ══════════════════════════════════════════════════════════════

#[test]
fn goat_planar_quant_vs_octopus_head_to_head() {
    let dim = 128usize;
    let bits_list = [2u8, 3, 4];
    let n_keys = 512;
    let n_queries = 64;
    let n_seeds = 8;

    println!("\n⚔️  GOAT 023: PlanarQuant vs OCTOPUS Head-to-Head (d={dim})");
    println!("{}", "═".repeat(80));

    println!(
        "{:<5} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} │ {:>7}",
        "bits", "PQ MSE", "PQ Cos", "OCT MSE", "OCT Cos", "MSE Δ%", "Winner",
    );
    println!("{}", "─".repeat(80));

    for &bits in &bits_list {
        let mut rng = Rng::new(42);

        let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();
        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|_| gaussian_vec(dim, &mut rng))
            .collect();

        // PlanarQuant
        #[cfg(feature = "planar_quant")]
        let (pq_mse, pq_cos, _) = {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_planar_quant(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            let (mse, _, cos, ip) = agg.summary();
            (mse, cos, ip)
        };
        #[cfg(not(feature = "planar_quant"))]
        let (pq_mse, pq_cos) = (0.0, 0.0);

        // OCTOPUS
        #[cfg(feature = "octopus")]
        let (oct_mse, oct_cos) = {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_octopus(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
                let (mse_m, _, cos_m, _) = r.summary();
                agg.add(mse_m, cos_m, 0.0);
            }
            let (mse, _, cos, _) = agg.summary();
            (mse, cos)
        };
        #[cfg(not(feature = "octopus"))]
        let (oct_mse, oct_cos) = (0.0, 0.0);

        #[cfg(all(feature = "planar_quant", feature = "octopus"))]
        {
            let mse_delta = (pq_mse - oct_mse) / oct_mse * 100.0;
            let winner = if pq_mse < oct_mse { "PQ" } else { "OCT" };
            println!(
                "{:<5} │ {:>10.6} {:>10.6} │ {:>10.6} {:>10.6} │ {:>+9.1}% │ {:>7}",
                bits, pq_mse, pq_cos, oct_mse, oct_cos, mse_delta, winner,
            );
        }

        #[cfg(not(all(feature = "planar_quant", feature = "octopus")))]
        {
            println!(
                "{:<5} │ (enable both planar_quant and octopus features)",
                bits
            );
        }
    }
    println!();
}

// ══════════════════════════════════════════════════════════════
// GOAT Test 5: IsoQuant Full vs Fast quality trade-off
// ══════════════════════════════════════════════════════════════

#[test]
#[cfg(feature = "iso_quant")]
fn goat_iso_quant_full_vs_fast() {
    use katgpt_rs::iso_quant::IsoQuantMode;

    let dim = 128usize;
    let bits_list = [2u8, 3, 4];
    let n_keys = 512;
    let n_seeds = 8;

    println!("\n🔬 GOAT 023: IsoQuant Full vs Fast Quality Trade-off (d={dim})");
    println!("{}", "═".repeat(70));

    println!(
        "{:<5} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>8}",
        "bits", "Full MSE", "Full Cos", "Fast MSE", "Fast Cos", "FMAs Δ",
    );
    println!("{}", "─".repeat(70));

    for &bits in &bits_list {
        let mut rng = Rng::new(42);
        let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();
        let queries: Vec<Vec<f32>> = (0..64).map(|_| gaussian_vec(dim, &mut rng)).collect();

        let full = {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_iso_quant(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &keys,
                    &queries,
                    IsoQuantMode::Full,
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            agg.summary()
        };

        let fast = {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_iso_quant(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &keys,
                    &queries,
                    IsoQuantMode::Fast,
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            agg.summary()
        };

        let n_groups = dim.div_ceil(4);
        let full_fmas = n_groups * 32; // 2 Hamilton products × 16
        let fast_fmas = n_groups * 16; // 1 Hamilton product × 16
        let fmas_delta = format!("{} vs {}", full_fmas, fast_fmas);

        println!(
            "{:<5} │ {:>10.6} {:>10.6} │ {:>10.6} {:>10.6} │ {:>8}",
            bits, full.0, full.2, fast.0, fast.2, fmas_delta,
        );
    }
    println!();
}

// ══════════════════════════════════════════════════════════════
// GOAT Test 6: Dimension scaling (bits=3, varying d)
// ══════════════════════════════════════════════════════════════

#[test]
fn goat_dimension_scaling() {
    let dims = [64usize, 128, 256];
    let bits = 3u8;
    let n_keys = 512;
    let n_seeds = 4;

    println!("\n📏 GOAT 023: Dimension Scaling (bits={bits})");
    println!("{}", "═".repeat(90));

    println!(
        "{:<6} │ {:>10} {:>10} │ {:>10} {:>10} │ {:>10} {:>10}",
        "d", "PQ MSE", "PQ Cos", "OCT MSE", "OCT Cos", "IQ-F MSE", "IQ-F Cos",
    );
    println!("{}", "─".repeat(90));

    for &dim in &dims {
        let mut rng = Rng::new(42);
        let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();
        let queries: Vec<Vec<f32>> = (0..64).map(|_| gaussian_vec(dim, &mut rng)).collect();

        #[cfg(feature = "planar_quant")]
        let (pq_mse, pq_cos) = {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_planar_quant(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
                let (mse_m, _, cos_m, _) = r.summary();
                agg.add(mse_m, cos_m, 0.0);
            }
            let (mse, _, cos, _) = agg.summary();
            (mse, cos)
        };
        #[cfg(not(feature = "planar_quant"))]
        let (pq_mse, pq_cos) = (0.0, 0.0);

        #[cfg(feature = "octopus")]
        let (oct_mse, oct_cos) = {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_octopus(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
                let (mse_m, _, cos_m, _) = r.summary();
                agg.add(mse_m, cos_m, 0.0);
            }
            let (mse, _, cos, _) = agg.summary();
            (mse, cos)
        };
        #[cfg(not(feature = "octopus"))]
        let (oct_mse, oct_cos) = (0.0, 0.0);

        #[cfg(feature = "iso_quant")]
        let (iq_mse, iq_cos) = {
            use katgpt_rs::iso_quant::IsoQuantMode;
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_iso_quant(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &keys,
                    &queries,
                    IsoQuantMode::Full,
                );
                let (mse_m, _, cos_m, _) = r.summary();
                agg.add(mse_m, cos_m, 0.0);
            }
            let (mse, _, cos, _) = agg.summary();
            (mse, cos)
        };
        #[cfg(not(feature = "iso_quant"))]
        let (iq_mse, iq_cos) = (0.0, 0.0);

        println!(
            "{:<6} │ {:>10.6} {:>10.6} │ {:>10.6} {:>10.6} │ {:>10.6} {:>10.6}",
            dim, pq_mse, pq_cos, oct_mse, oct_cos, iq_mse, iq_cos,
        );
    }
    println!();
}

// ══════════════════════════════════════════════════════════════
// GOAT Test 7: 3-Way Matrix (d=128, bits=3, all backends)
// ══════════════════════════════════════════════════════════════

#[test]
fn goat_three_way_matrix() {
    let dim = 128usize;
    let bits = 3u8;
    let n_keys = 512;
    let n_seeds = 8;

    println!("\n📋 GOAT 023: 3-Way Comparison Matrix (d={dim}, bits={bits})");
    println!("{}", "═".repeat(100));

    let mut rng = Rng::new(42);
    let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..64).map(|_| gaussian_vec(dim, &mut rng)).collect();

    println!("┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│ Metric           │ TurboQuant   │ OCTOPUS      │ PlanarQuant  │ IsoQuant-F   │");
    println!("├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");

    // MSE row
    #[cfg(feature = "turboquant")]
    let tq_mse = {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_turboquant(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
            let (mse_m, _, _, _) = r.summary();
            agg.add(mse_m, 0.0, 0.0);
        }
        agg.summary().0
    };
    #[cfg(not(feature = "turboquant"))]
    let tq_mse: f64 = 0.0;

    #[cfg(feature = "octopus")]
    let oct_mse = {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_octopus(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
            let (mse_m, _, _, _) = r.summary();
            agg.add(mse_m, 0.0, 0.0);
        }
        agg.summary().0
    };
    #[cfg(not(feature = "octopus"))]
    let oct_mse: f64 = 0.0;

    #[cfg(feature = "planar_quant")]
    let pq_mse = {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_planar_quant(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
            let (mse_m, _, _, _) = r.summary();
            agg.add(mse_m, 0.0, 0.0);
        }
        agg.summary().0
    };
    #[cfg(not(feature = "planar_quant"))]
    let pq_mse: f64 = 0.0;

    #[cfg(feature = "iso_quant")]
    let iq_mse = {
        use katgpt_rs::iso_quant::IsoQuantMode;
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_iso_quant(
                dim,
                bits,
                seed as u64 * 1000 + 42,
                &keys,
                &queries,
                IsoQuantMode::Full,
            );
            let (mse_m, _, _, _) = r.summary();
            agg.add(mse_m, 0.0, 0.0);
        }
        agg.summary().0
    };
    #[cfg(not(feature = "iso_quant"))]
    let iq_mse: f64 = 0.0;

    println!(
        "│ MSE              │ {:>12.6} │ {:>12.6} │ {:>12.6} │ {:>12.6} │",
        tq_mse, oct_mse, pq_mse, iq_mse
    );

    // Cosine row
    #[cfg(feature = "turboquant")]
    let tq_cos = {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_turboquant(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
            let (_, _, cos_m, _) = r.summary();
            agg.add(0.0, cos_m, 0.0);
        }
        agg.summary().2
    };
    #[cfg(not(feature = "turboquant"))]
    let tq_cos: f64 = 0.0;

    #[cfg(feature = "octopus")]
    let oct_cos = {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_octopus(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
            let (_, _, cos_m, _) = r.summary();
            agg.add(0.0, cos_m, 0.0);
        }
        agg.summary().2
    };
    #[cfg(not(feature = "octopus"))]
    let oct_cos: f64 = 0.0;

    #[cfg(feature = "planar_quant")]
    let pq_cos = {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_planar_quant(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
            let (_, _, cos_m, _) = r.summary();
            agg.add(0.0, cos_m, 0.0);
        }
        agg.summary().2
    };
    #[cfg(not(feature = "planar_quant"))]
    let pq_cos: f64 = 0.0;

    #[cfg(feature = "iso_quant")]
    let iq_cos = {
        use katgpt_rs::iso_quant::IsoQuantMode;
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_iso_quant(
                dim,
                bits,
                seed as u64 * 1000 + 42,
                &keys,
                &queries,
                IsoQuantMode::Full,
            );
            let (_, _, cos_m, _) = r.summary();
            agg.add(0.0, cos_m, 0.0);
        }
        agg.summary().2
    };
    #[cfg(not(feature = "iso_quant"))]
    let iq_cos: f64 = 0.0;

    println!(
        "│ Cosine           │ {:>12.6} │ {:>12.6} │ {:>12.6} │ {:>12.6} │",
        tq_cos, oct_cos, pq_cos, iq_cos
    );

    // Rotation cost rows
    let tq_fmas = dim * dim;
    let pq_fmas = dim.div_ceil(2) * 4;
    let iq_fmas = dim.div_ceil(4) * 32;
    let tq_params = dim * dim;
    let pq_params = dim.div_ceil(2) * 2;
    let iq_params = dim.div_ceil(4) * 4 * 2;

    println!(
        "│ Rotation FMAs    │ {:>12} │ {:>12} │ {:>12} │ {:>12} │",
        tq_fmas, tq_fmas, pq_fmas, iq_fmas
    );
    println!(
        "│ Params           │ {:>12} │ {:>12} │ {:>12} │ {:>12} │",
        tq_params, tq_params, pq_params, iq_params
    );
    println!(
        "│ FMAs ratio vs TQ │ {:>12} │ {:>12} │ {:>11.0}× │ {:>11.0}× │",
        "1.0×",
        "1.0×",
        tq_fmas as f64 / pq_fmas as f64,
        tq_fmas as f64 / iq_fmas as f64,
    );

    println!("├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");

    // Winner determination
    #[allow(unused_mut)]
    let mut winner_row = String::from("│ Winner           │");
    #[cfg(all(
        feature = "turboquant",
        feature = "octopus",
        feature = "planar_quant",
        feature = "iso_quant"
    ))]
    {
        let all_mse = [tq_mse, oct_mse, pq_mse, iq_mse];
        let all_cos = [tq_cos, oct_cos, pq_cos, iq_cos];
        let names = ["TQ", "OCT", "PQ", "IQ-F"];
        let mse_winner = names[all_mse
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0];
        let _cos_winner = names[all_cos
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0];
        winner_row.push_str(&format!(
            " {:>12} │ {:>12} │ {:>12} │ {:>12} │",
            if mse_winner == "TQ" { "★" } else { "" },
            if mse_winner == "OCT" { "★" } else { "" },
            if mse_winner == "PQ" {
                "★ MSE"
            } else {
                "★ speed"
            },
            if mse_winner == "IQ-F" { "★" } else { "" },
        ));
    }
    #[cfg(not(all(
        feature = "turboquant",
        feature = "octopus",
        feature = "planar_quant",
        feature = "iso_quant"
    )))]
    {
        winner_row.push_str(
            " (enable all features for winner) │              │              │              │",
        );
    }
    println!("{winner_row}");

    println!("└──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
    println!();
}

// ══════════════════════════════════════════════════════════════
// GOAT Test 8: Production Stack Verdict
// ══════════════════════════════════════════════════════════════

#[test]
fn goat_production_stack_verdict() {
    println!("\n🏆 GOAT 023: Production Stack Verdict");
    println!("{}", "═".repeat(80));

    let dim = 128usize;
    let n_seeds = 8;
    let n_keys = 512;

    let mut rng = Rng::new(42);
    let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..64).map(|_| gaussian_vec(dim, &mut rng)).collect();

    // Collect all results at bits=3
    let bits = 3u8;

    #[derive(Default)]
    struct Summary {
        mse: f64,
        cos: f64,
        fmas: usize,
        params: usize,
        available: bool,
    }

    #[allow(unused_mut)]
    let mut tq = Summary::default();
    #[allow(unused_mut)]
    let mut oct = Summary::default();
    #[allow(unused_mut)]
    let mut pq = Summary::default();
    #[allow(unused_mut)]
    let mut iqf = Summary::default();

    #[cfg(feature = "turboquant")]
    {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_turboquant(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
            let (mse_m, _, cos_m, _) = r.summary();
            agg.add(mse_m, cos_m, 0.0);
        }
        let (mse, _, cos, _) = agg.summary();
        tq.mse = mse;
        tq.cos = cos;
        tq.fmas = dim * dim;
        tq.params = dim * dim;
        tq.available = true;
    }

    #[cfg(feature = "octopus")]
    {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_octopus(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
            let (mse_m, _, cos_m, _) = r.summary();
            agg.add(mse_m, cos_m, 0.0);
        }
        let (mse, _, cos, _) = agg.summary();
        oct.mse = mse;
        oct.cos = cos;
        oct.fmas = dim * dim; // OCTOPUS uses WHT too
        oct.params = dim * dim;
        oct.available = true;
    }

    #[cfg(feature = "planar_quant")]
    {
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_planar_quant(dim, bits, seed as u64 * 1000 + 42, &keys, &queries);
            let (mse_m, _, cos_m, _) = r.summary();
            agg.add(mse_m, cos_m, 0.0);
        }
        let (mse, _, cos, _) = agg.summary();
        pq.mse = mse;
        pq.cos = cos;
        pq.fmas = dim.div_ceil(2) * 4;
        pq.params = dim.div_ceil(2) * 2;
        pq.available = true;
    }

    #[cfg(feature = "iso_quant")]
    {
        use katgpt_rs::iso_quant::IsoQuantMode;
        let mut agg = BackendResult::default();
        for seed in 0..n_seeds {
            let r = bench_iso_quant(
                dim,
                bits,
                seed as u64 * 1000 + 42,
                &keys,
                &queries,
                IsoQuantMode::Full,
            );
            let (mse_m, _, cos_m, _) = r.summary();
            agg.add(mse_m, cos_m, 0.0);
        }
        let (mse, _, cos, _) = agg.summary();
        iqf.mse = mse;
        iqf.cos = cos;
        iqf.fmas = dim.div_ceil(4) * 32;
        iqf.params = dim.div_ceil(4) * 4 * 2;
        iqf.available = true;
    }

    // Print verdict
    println!("Results at d={dim}, bits={bits}, {n_keys} keys, {n_seeds} seeds:");
    println!();

    if tq.available {
        println!(
            "  TurboQuant:  MSE={:.6}  Cos={:.6}  FMAs={}  Params={}",
            tq.mse, tq.cos, tq.fmas, tq.params
        );
    }
    if oct.available {
        let mse_vs_tq = if tq.available && tq.mse > 0.0 {
            format!(" ({:+.0}% vs TQ)", (oct.mse - tq.mse) / tq.mse * 100.0)
        } else {
            String::new()
        };
        println!(
            "  OCTOPUS:     MSE={:.6}{mse_vs_tq}  Cos={:.6}  FMAs={}  Params={}",
            oct.mse, oct.cos, oct.fmas, oct.params
        );
    }
    if pq.available {
        let mse_vs_tq = if tq.available && tq.mse > 0.0 {
            format!(" ({:+.0}% vs TQ)", (pq.mse - tq.mse) / tq.mse * 100.0)
        } else {
            String::new()
        };
        let pq_fmas = pq.fmas;
        let speedup = if tq.available {
            format!(" ({:.0}× fewer FMAs)", tq.fmas as f64 / pq.fmas as f64)
        } else {
            String::new()
        };
        println!(
            "  PlanarQuant: MSE={:.6}{mse_vs_tq}  Cos={:.6}  FMAs={pq_fmas}{speedup}  Params={}",
            pq.mse, pq.cos, pq.params
        );
    }
    if iqf.available {
        let mse_vs_tq = if tq.available && tq.mse > 0.0 {
            format!(" ({:+.0}% vs TQ)", (iqf.mse - tq.mse) / tq.mse * 100.0)
        } else {
            String::new()
        };
        let iqf_fmas = iqf.fmas;
        let speedup = if tq.available {
            format!(" ({:.0}× fewer FMAs)", tq.fmas as f64 / iqf.fmas as f64)
        } else {
            String::new()
        };
        println!(
            "  IsoQuant-F:  MSE={:.6}{mse_vs_tq}  Cos={:.6}  FMAs={iqf_fmas}{speedup}  Params={}",
            iqf.mse, iqf.cos, iqf.params
        );
    }

    println!();

    // Decision
    if pq.available && oct.available {
        if pq.mse < oct.mse {
            println!("🏆 VERDICT: PlanarQuant WINS on both MSE and speed!");
            println!(
                "   PlanarQuant: MSE={:.6} (better), FMAs={} ({:.0}× fewer than OCTOPUS's {})",
                pq.mse,
                pq.fmas,
                oct.fmas as f64 / pq.fmas as f64,
                oct.fmas
            );
            println!("   → Recommend: Promote PlanarQuant to default-on");
        } else {
            let gap_pct = (pq.mse - oct.mse) / oct.mse * 100.0;
            let speedup = oct.fmas as f64 / pq.fmas as f64;
            println!("🏆 VERDICT: Mixed result — each backend has a clear advantage:");
            println!(
                "   MSE quality:   OCTOPUS wins ({:.6} vs {:.6}, {gap_pct:.0}% better)",
                oct.mse, pq.mse
            );
            println!(
                "   Rotation speed: PlanarQuant wins ({} FMAs vs {} FMAs, {speedup:.0}× fewer)",
                pq.fmas, oct.fmas
            );
            println!(
                "   → Recommend: Keep OCTOPUS as MSE-optimal default, PlanarQuant as speed-optimal alternative"
            );
            println!("   → Future: Hybrid (OCTOPUS encoding + PlanarQuant rotation) as Plan 101");
        }
    } else if pq.available {
        println!("🏆 VERDICT: PlanarQuant available — check MSE vs baseline");
    } else if oct.available {
        println!("🏆 VERDICT: OCTOPUS available — enable planar_quant feature for comparison");
    } else {
        println!("🏆 VERDICT: Enable planar_quant and/or octopus features for comparison");
    }

    println!();
    println!("Production Stack (after GOAT 023):");
    println!("  1. OCTOPUS       — default-on, best MSE quality (Bench 022)");
    let pq_fmas_display = if pq.available { pq.fmas } else { 0 };
    let oct_pq_ratio = if oct.available && pq.available {
        oct.fmas as f64 / pq.fmas as f64
    } else if pq.available {
        64.0
    } else {
        0.0
    };
    println!(
        "  2. PlanarQuant   — opt-in, best rotation speed ({:.0}× fewer FMAs, {} FMAs vs {} FMAs)",
        oct_pq_ratio,
        if pq.available {
            pq_fmas_display.to_string()
        } else {
            "—".to_string()
        },
        if oct.available {
            oct.fmas.to_string()
        } else {
            "—".to_string()
        }
    );
    println!("  3. IsoQuant-F    — opt-in, best quality at 4-bit, 4D blocks");
    println!("  4. SpectralQuant — default-on, calibrated water-fill");
    println!("  5. TurboQuant    — legacy baseline");
    println!();
}

// ══════════════════════════════════════════════════════════════
// GOAT Test 9: MaxSim Late-Interaction Scoring (Plan 100 T13)
//
// MaxSim computes Σ_i max_j dot(q_i, k_j) — amplifies quantization
// error 12-14× compared to single-vector dot products. This test
// measures how well each backend preserves MaxSim after quantization.
// ══════════════════════════════════════════════════════════════

#[cfg(feature = "maxsim")]
#[test]
fn goat_maxsim_late_interaction() {
    use katgpt_rs::simd::maxsim_score;

    let dim = 128usize;
    let n_keys = 512usize;
    let n_queries = 4usize;
    let n_seeds = 4usize;
    let bits_list: [u8; 3] = [2, 3, 4];

    println!("\n🔬 GOAT 023-T13: MaxSim Late-Interaction Scoring");
    println!("{}", "═".repeat(90));
    println!("  Config: {n_keys} keys × {n_queries} queries × d={dim}, {n_seeds} rotation seeds");
    println!("  Metric: relative error = |maxsim_dequant - maxsim_truth| / |maxsim_truth|");
    println!("  MaxSim amplifies quantization error ~12-14× vs single-vector dot products");
    println!();

    // ── Generate fixed data ──
    let mut rng = Rng::new(42);
    let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| gaussian_vec(dim, &mut rng))
        .collect();

    // Flatten queries for maxsim_score: [q0, q1, q2, q3] → flat [n_queries * dim]
    let queries_flat: Vec<f32> = queries.iter().flat_map(|q| q.iter().copied()).collect();

    // ── Ground truth MaxSim on uncompressed keys ──
    let keys_flat: Vec<f32> = keys.iter().flat_map(|k| k.iter().copied()).collect();
    let truth = maxsim_score(&queries_flat, &keys_flat, n_queries, n_keys, dim);
    println!("  Ground-truth MaxSim: {truth:.6}");
    println!();

    if truth.abs() < 1e-8 {
        println!("  ⚠ Truth too close to zero, skipping test.");
        return;
    }

    // ── Per-backend results: (bits, seed) → relative errors ──
    #[derive(Default)]
    struct MaxSimResult {
        name: String,
        rel_errors: Vec<f64>,
        available: bool,
    }

    println!(
        "  {:8} │ {:>8} │ {:>8} │ {:>8} │ {:>10}",
        "bits", "seed", "maxsim", "rel_err%", "amplif."
    );
    println!(
        "  {}─┼─{}─┼─{}─┼─{}─┼─{}",
        "─".repeat(8),
        "─".repeat(8),
        "─".repeat(8),
        "─".repeat(8),
        "─".repeat(10)
    );

    for &bits in &bits_list {
        let mut pq_res = MaxSimResult {
            name: "PQ".into(),
            ..Default::default()
        };
        let mut iqf_res = MaxSimResult {
            name: "IQ-Fast".into(),
            ..Default::default()
        };
        let mut iqr_res = MaxSimResult {
            name: "IQ-Full".into(),
            ..Default::default()
        };
        let mut oct_res = MaxSimResult {
            name: "OCT".into(),
            ..Default::default()
        };
        let mut tq_res = MaxSimResult {
            name: "TQ".into(),
            ..Default::default()
        };

        for seed in 0..n_seeds {
            let seed_val = seed as u64 * 1000 + 42;

            // ── PlanarQuant ──
            #[cfg(feature = "planar_quant")]
            {
                use katgpt_rs::planar_quant::{PlanarQuantConfig, PlanarQuantKVCache};
                let config = PlanarQuantConfig {
                    key_bits: bits,
                    val_bits: bits,
                    seed: seed_val,
                    n_layers: 1,
                    kv_dim: dim,
                    max_seq_len: n_keys + 16,
                };
                let mut cache = PlanarQuantKVCache::with_config(&config);
                for (pos, key) in keys.iter().enumerate() {
                    cache.store_key(0, pos, key);
                }
                let dequant_flat: Vec<f32> = (0..n_keys)
                    .flat_map(|pos| cache.dequantize_key(0, pos))
                    .collect();
                let ms = maxsim_score(&queries_flat, &dequant_flat, n_queries, n_keys, dim);
                let rel_err = ((ms - truth) / truth).abs() as f64;
                pq_res.rel_errors.push(rel_err);
                pq_res.available = true;
                let pct = rel_err * 100.0;
                println!(
                    "  {:8} │ {:8} │ PQ       │ {ms:8.4} │ {pct:8.2}% │",
                    bits, seed
                );
            }

            // ── IsoQuant Fast ──
            #[cfg(feature = "iso_quant")]
            {
                use katgpt_rs::iso_quant::{IsoQuantConfig, IsoQuantKVCache, IsoQuantMode};
                let config = IsoQuantConfig {
                    key_bits: bits,
                    val_bits: bits,
                    seed: seed_val,
                    n_layers: 1,
                    kv_dim: dim,
                    max_seq_len: n_keys + 16,
                    mode: IsoQuantMode::Fast,
                };
                let mut cache = IsoQuantKVCache::new(&config);
                for (pos, key) in keys.iter().enumerate() {
                    cache.store_key(0, pos, key);
                }
                let dequant_flat: Vec<f32> = (0..n_keys)
                    .flat_map(|pos| cache.dequantize_key(0, pos))
                    .collect();
                let ms = maxsim_score(&queries_flat, &dequant_flat, n_queries, n_keys, dim);
                let rel_err = ((ms - truth) / truth).abs() as f64;
                iqf_res.rel_errors.push(rel_err);
                iqf_res.available = true;
                let pct = rel_err * 100.0;
                println!(
                    "  {:8} │ {:8} │ IQ-Fast  │ {ms:8.4} │ {pct:8.2}% │",
                    bits, seed
                );
            }

            // ── IsoQuant Full ──
            #[cfg(feature = "iso_quant")]
            {
                use katgpt_rs::iso_quant::{IsoQuantConfig, IsoQuantKVCache, IsoQuantMode};
                let config = IsoQuantConfig {
                    key_bits: bits,
                    val_bits: bits,
                    seed: seed_val,
                    n_layers: 1,
                    kv_dim: dim,
                    max_seq_len: n_keys + 16,
                    mode: IsoQuantMode::Full,
                };
                let mut cache = IsoQuantKVCache::new(&config);
                for (pos, key) in keys.iter().enumerate() {
                    cache.store_key(0, pos, key);
                }
                let dequant_flat: Vec<f32> = (0..n_keys)
                    .flat_map(|pos| cache.dequantize_key(0, pos))
                    .collect();
                let ms = maxsim_score(&queries_flat, &dequant_flat, n_queries, n_keys, dim);
                let rel_err = ((ms - truth) / truth).abs() as f64;
                iqr_res.rel_errors.push(rel_err);
                iqr_res.available = true;
                let pct = rel_err * 100.0;
                println!(
                    "  {:8} │ {:8} │ IQ-Full  │ {ms:8.4} │ {pct:8.2}% │",
                    bits, seed
                );
            }

            // ── OCTOPUS ──
            #[cfg(feature = "octopus")]
            {
                use katgpt_rs::octopus::{OctopusConfig, OctopusKVCache};
                let config = OctopusConfig {
                    key_bits: bits,
                    val_bits: bits,
                    seed: seed_val,
                    n_layers: 1,
                    kv_dim: dim,
                    max_seq_len: n_keys + 16,
                    use_qjl_residual: false,
                    use_joint_rounding: true,
                };
                let mut cache = OctopusKVCache::with_config(&config);
                for (pos, key) in keys.iter().enumerate() {
                    cache.store_key(0, pos, key);
                }
                let dequant_flat: Vec<f32> = (0..n_keys)
                    .flat_map(|pos| cache.dequantize_key(0, pos))
                    .collect();
                let ms = maxsim_score(&queries_flat, &dequant_flat, n_queries, n_keys, dim);
                let rel_err = ((ms - truth) / truth).abs() as f64;
                oct_res.rel_errors.push(rel_err);
                oct_res.available = true;
                let pct = rel_err * 100.0;
                println!(
                    "  {:8} │ {:8} │ OCT      │ {ms:8.4} │ {pct:8.2}% │",
                    bits, seed
                );
            }

            // ── TurboQuant ──
            #[cfg(feature = "turboquant")]
            {
                use katgpt_rs::turboquant::{TurboQuantKVCache, TurboQuantKVCacheConfig};
                let config = TurboQuantKVCacheConfig {
                    key_bits: bits,
                    val_bits: bits,
                    seed: seed_val,
                    n_layers: 1,
                    kv_dim: dim,
                    max_seq_len: n_keys + 16,
                };
                let mut cache = TurboQuantKVCache::with_config(&config);
                for (pos, key) in keys.iter().enumerate() {
                    cache.store_key(0, pos, key);
                }
                let dequant_flat: Vec<f32> = (0..n_keys)
                    .flat_map(|pos| cache.dequantize_key(0, pos))
                    .collect();
                let ms = maxsim_score(&queries_flat, &dequant_flat, n_queries, n_keys, dim);
                let rel_err = ((ms - truth) / truth).abs() as f64;
                tq_res.rel_errors.push(rel_err);
                tq_res.available = true;
                let pct = rel_err * 100.0;
                println!(
                    "  {:8} │ {:8} │ TQ       │ {ms:8.4} │ {pct:8.2}% │",
                    bits, seed
                );
            }
        }

        // ── Summary for this bits level ──
        println!(
            "  {}─┼─{}─┼─{}─┼─{}─┼─{}",
            "─".repeat(8),
            "─".repeat(8),
            "─".repeat(8),
            "─".repeat(8),
            "─".repeat(10)
        );
        println!("  bits={bits} mean relative error:");

        let all_results: Vec<&MaxSimResult> = vec![&pq_res, &iqf_res, &iqr_res, &oct_res, &tq_res];
        let mut best_name = String::new();
        let mut best_err = f64::INFINITY;

        for res in &all_results {
            match res.available {
                true => {
                    let (mean, std) = mean_std(&res.rel_errors);
                    let name = &res.name;
                    println!(
                        "    {name:10} rel_err = {mean:.4} ± {std:.4} ({pct:.2}%)",
                        pct = mean * 100.0
                    );
                    if mean < best_err {
                        best_err = mean;
                        best_name = res.name.clone();
                    }
                }
                false => {
                    let name = &res.name;
                    println!("    {name:10} (not available)");
                }
            }
        }

        match best_name.is_empty() {
            true => println!("    (no backends available)"),
            false => println!("    🏆 bits={bits}: {best_name} wins (rel_err={best_err:.4})"),
        }
        println!();
    }

    // ── Final verdict ──
    println!("{}", "═".repeat(90));
    println!("  MaxSim Late-Interaction Summary:");
    println!("  • MaxSim aggregates max dot products → error amplification 12-14× expected");
    println!("  • Higher bits (3→4) should cut error roughly in half");
    println!("  • Backends with better per-vector cosine will also win here");
    println!(
        "  • If relative error < 10% at bits=4, the backend is viable for late-interaction retrieval"
    );
    println!();
}

// ══════════════════════════════════════════════════════════════
// GOAT Test 10: Hybrid OCT+PQ quality sweep (Plan 101, T11)
// ══════════════════════════════════════════════════════════════

#[test]
fn goat_hybrid_oct_pq_quality_sweep() {
    let dim = 128usize;
    let bits_list = [2u8, 3, 4];
    let n_keys = 512;
    let n_queries = 64;
    let n_seeds = 8;

    println!("\n🧪 GOAT 024: Hybrid OCT+PQ Quality Sweep (d={dim})");
    println!("{}", "═".repeat(110));
    println!("Config: {n_keys} Gaussian keys, {n_queries} Gaussian queries, {n_seeds} seeds");
    println!("Backends: Hybrid OCT+PQ, Pure OCTOPUS, Pure PlanarQuant, TurboQuant");
    println!("Hypothesis: Hybrid MSE within 5% of pure OCTOPUS, with 64× fewer rotation FMAs");
    println!();

    // Generate shared test data
    let mut all_keys: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut all_queries: Vec<Vec<Vec<f32>>> = Vec::new();
    for seed in 0..n_seeds {
        let mut rng = Rng::new(seed as u64 * 7919 + 13);
        let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();
        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|_| gaussian_vec(dim, &mut rng))
            .collect();
        all_keys.push(keys);
        all_queries.push(queries);
    }

    println!(
        "  {:5} │ {:>12} │ {:>12} │ {:>12} │ {:>12} │ {:>8} │ {:>8}",
        "bits", "Hybrid MSE", "OCT MSE", "PQ MSE", "TQ MSE", "H/O ratio", "Winner"
    );
    println!(
        "  {}─┼─{}─┼─{}─┼─{}─┼─{}─┼─{}─┼─{}",
        "─".repeat(5),
        "─".repeat(12),
        "─".repeat(12),
        "─".repeat(12),
        "─".repeat(12),
        "─".repeat(8),
        "─".repeat(8)
    );

    for &bits in &bits_list {
        // ── Hybrid OCT+PQ ──
        #[cfg(feature = "hybrid_oct_pq")]
        let hybrid_agg = {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_hybrid_oct_pq(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &all_keys[seed],
                    &all_queries[seed],
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            agg
        };

        // ── Pure OCTOPUS ──
        #[cfg(feature = "octopus")]
        let oct_agg = {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_octopus(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &all_keys[seed],
                    &all_queries[seed],
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            agg
        };

        // ── Pure PlanarQuant ──
        #[cfg(feature = "planar_quant")]
        let pq_agg = {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_planar_quant(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &all_keys[seed],
                    &all_queries[seed],
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            agg
        };

        // ── TurboQuant ──
        #[cfg(feature = "turboquant")]
        let tq_agg = {
            let mut agg = BackendResult::default();
            for seed in 0..n_seeds {
                let r = bench_turboquant(
                    dim,
                    bits,
                    seed as u64 * 1000 + 42,
                    &all_keys[seed],
                    &all_queries[seed],
                );
                let (mse_m, _, cos_m, ip_m) = r.summary();
                agg.add(mse_m, cos_m, ip_m);
            }
            agg
        };

        // ── Print results ──
        #[allow(unused_variables)]
        let na = "N/A";
        #[allow(unused_mut)]
        let mut winner = String::new();
        #[allow(unused_mut, unused_assignments)]
        let mut best_mse = f64::INFINITY;

        #[cfg(feature = "hybrid_oct_pq")]
        {
            let (h_mse, _, _h_cos, _h_ip) = hybrid_agg.summary();
            let (_h_mse_std, _) = mean_std(&hybrid_agg.mse);
            if h_mse < best_mse {
                best_mse = h_mse;
                winner = "Hybrid".into();
            }
            print!("  {bits:5} │ {h_mse:12.6} │");
        }
        #[cfg(not(feature = "hybrid_oct_pq"))]
        {
            print!("  {bits:5} │ {na:>12} │");
        }

        #[cfg(feature = "octopus")]
        {
            let (o_mse, _, _o_cos, _o_ip) = oct_agg.summary();
            if o_mse < best_mse {
                best_mse = o_mse;
                winner = "OCT".into();
            }
            print!(" {o_mse:12.6} │");
        }
        #[cfg(not(feature = "octopus"))]
        {
            print!(" {na:>12} │");
        }

        #[cfg(feature = "planar_quant")]
        {
            let (p_mse, _, _p_cos, _p_ip) = pq_agg.summary();
            if p_mse < best_mse {
                best_mse = p_mse;
                winner = "PQ".into();
            }
            print!(" {p_mse:12.6} │");
        }
        #[cfg(not(feature = "planar_quant"))]
        {
            print!(" {na:>12} │");
        }

        #[cfg(feature = "turboquant")]
        #[allow(unused_assignments)]
        {
            let (t_mse, _, _t_cos, _t_ip) = tq_agg.summary();
            if t_mse < best_mse {
                best_mse = t_mse;
                winner = "TQ".into();
            }
            print!(" {t_mse:12.6} │");
        }
        #[cfg(not(feature = "turboquant"))]
        {
            print!(" {na:>12} │");
        }

        #[cfg(all(feature = "hybrid_oct_pq", feature = "octopus"))]
        {
            let (h_mse, _, _, _) = hybrid_agg.summary();
            let (o_mse, _, _, _) = oct_agg.summary();
            let ratio = h_mse / o_mse;
            print!(" {ratio:8.3}× │");
        }
        #[cfg(not(all(feature = "hybrid_oct_pq", feature = "octopus")))]
        {
            print!(" {na:>8} │");
        }

        println!(" {winner:8}");
    }

    println!(
        "  {}─┼─{}─┼─{}─┼─{}─┼─{}─┼─{}─┼─{}",
        "─".repeat(5),
        "─".repeat(12),
        "─".repeat(12),
        "─".repeat(12),
        "─".repeat(12),
        "─".repeat(8),
        "─".repeat(8)
    );

    // ── Rotation cost comparison ──
    println!();
    println!("  Rotation Cost (d={dim}):");
    println!("  {:15} │ {:>8} │ {:>8}", "Backend", "FMAs", "Params");
    println!(
        "  {}─┼─{}─┼─{}",
        "─".repeat(15),
        "─".repeat(8),
        "─".repeat(8)
    );
    println!("  {:15} │ {:8} │ {:8}", "TurboQuant", 16_384, 16_384);
    println!("  {:15} │ {:8} │ {:8}", "OCTOPUS", 16_384, 16_384);
    println!("  {:15} │ {:8} │ {:8}", "PlanarQuant", 256, 128);
    println!("  {:15} │ {:8} │ {:8}", "Hybrid OCT+PQ", 256, 128);
    println!("  {:15} │ {:8} │ {:>8}", "", "", "64× faster".to_string());
    println!();

    // ── Verdict ──
    #[cfg(all(feature = "hybrid_oct_pq", feature = "octopus"))]
    {
        // Check hypothesis at 3-bit (the critical regime)
        let mut hybrid_mse_3 = 0.0f64;
        let mut oct_mse_3 = 0.0f64;
        for seed in 0..n_seeds {
            let r = bench_hybrid_oct_pq(
                dim,
                3,
                seed as u64 * 1000 + 42,
                &all_keys[seed],
                &all_queries[seed],
            );
            let (m, _, _, _) = r.summary();
            hybrid_mse_3 += m;
        }
        for seed in 0..n_seeds {
            let r = bench_octopus(
                dim,
                3,
                seed as u64 * 1000 + 42,
                &all_keys[seed],
                &all_queries[seed],
            );
            let (m, _, _, _) = r.summary();
            oct_mse_3 += m;
        }
        hybrid_mse_3 /= n_seeds as f64;
        oct_mse_3 /= n_seeds as f64;
        let ratio = hybrid_mse_3 / oct_mse_3;

        if ratio < 1.05 {
            println!(
                "  ✅ PASS: Hybrid MSE is {ratio:.3}× of OCTOPUS at 3-bit (within 5% threshold)"
            );
        } else if ratio < 1.10 {
            println!("  ⚠ MARGINAL: Hybrid MSE is {ratio:.3}× of OCTOPUS at 3-bit (within 10%)");
        } else {
            println!("  ❌ FAIL: Hybrid MSE is {ratio:.3}× of OCTOPUS at 3-bit (exceeds 10%)");
        }
        println!("  Hybrid: {hybrid_mse_3:.6}, OCTOPUS: {oct_mse_3:.6}");
    }

    println!();
}

// ══════════════════════════════════════════════════════════════
// GOAT Test 11: Hybrid OCT+PQ MaxSim (Plan 101, T12)
// ══════════════════════════════════════════════════════════════

#[cfg(feature = "maxsim")]
#[test]
fn goat_hybrid_maxsim_late_interaction() {
    use katgpt_rs::simd::maxsim_score;

    let dim = 128usize;
    let n_keys = 512usize;
    let n_queries = 4usize;
    let n_seeds = 4usize;
    let bits_list: [u8; 3] = [2, 3, 4];

    println!("\n🔬 GOAT 024-T12: Hybrid OCT+PQ MaxSim Late-Interaction Scoring");
    println!("{}", "═".repeat(90));
    println!("  Config: {n_keys} keys × {n_queries} queries × d={dim}, {n_seeds} rotation seeds");
    println!("  Metric: relative error = |maxsim_dequant - maxsim_truth| / |maxsim_truth|");
    println!();

    // ── Generate fixed data ──
    let mut rng = Rng::new(42);
    let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| gaussian_vec(dim, &mut rng))
        .collect();

    let queries_flat: Vec<f32> = queries.iter().flat_map(|q| q.iter().copied()).collect();
    let keys_flat: Vec<f32> = keys.iter().flat_map(|k| k.iter().copied()).collect();
    let truth = maxsim_score(&queries_flat, &keys_flat, n_queries, n_keys, dim);
    println!("  Ground-truth MaxSim: {truth:.6}");
    println!();

    if truth.abs() < 1e-8 {
        println!("  ⚠ Truth too close to zero, skipping test.");
        return;
    }

    #[derive(Default)]
    struct MaxSimResult {
        name: String,
        rel_errors: Vec<f64>,
        available: bool,
    }

    println!(
        "  {:8} │ {:>8} │ {:>8} │ {:>10} │ {:>8}",
        "bits", "seed", "backend", "rel_err%", "abs_err"
    );
    println!(
        "  {}─┼─{}─┼─{}─┼─{}─┼─{}",
        "─".repeat(8),
        "─".repeat(8),
        "─".repeat(8),
        "─".repeat(10),
        "─".repeat(8)
    );

    for &bits in &bits_list {
        let mut hybrid_res = MaxSimResult {
            name: "Hybrid".into(),
            ..Default::default()
        };
        let mut oct_res = MaxSimResult {
            name: "OCT".into(),
            ..Default::default()
        };
        let mut pq_res = MaxSimResult {
            name: "PQ".into(),
            ..Default::default()
        };

        for seed in 0..n_seeds {
            let seed_val = seed as u64 * 1000 + 42;

            // ── Hybrid OCT+PQ ──
            #[cfg(feature = "hybrid_oct_pq")]
            {
                use katgpt_rs::hybrid_oct_pq::{HybridOctPqConfig, HybridOctPqKVCache};
                let config = HybridOctPqConfig {
                    key_bits: bits,
                    val_bits: bits,
                    seed: seed_val,
                    n_layers: 1,
                    kv_dim: dim,
                    max_seq_len: n_keys + 16,
                    use_joint_rounding: true,
                };
                let mut cache = HybridOctPqKVCache::with_config(&config);
                for (pos, key) in keys.iter().enumerate() {
                    cache.store_key(0, pos, key);
                }
                let dequant_flat: Vec<f32> = (0..n_keys)
                    .flat_map(|pos| cache.dequantize_key(0, pos))
                    .collect();
                let ms = maxsim_score(&queries_flat, &dequant_flat, n_queries, n_keys, dim);
                let rel_err = ((ms - truth) / truth).abs() as f64;
                hybrid_res.rel_errors.push(rel_err);
                hybrid_res.available = true;
                let pct = rel_err * 100.0;
                println!(
                    "  {:8} │ {:8} │ Hybrid   │ {pct:10.4}% │ {ms:8.4}",
                    bits, seed
                );
            }

            // ── OCTOPUS ──
            #[cfg(feature = "octopus")]
            {
                use katgpt_rs::octopus::{OctopusConfig, OctopusKVCache};
                let config = OctopusConfig {
                    key_bits: bits,
                    val_bits: bits,
                    seed: seed_val,
                    n_layers: 1,
                    kv_dim: dim,
                    max_seq_len: n_keys + 16,
                    use_qjl_residual: false,
                    use_joint_rounding: true,
                };
                let mut cache = OctopusKVCache::with_config(&config);
                for (pos, key) in keys.iter().enumerate() {
                    cache.store_key(0, pos, key);
                }
                let dequant_flat: Vec<f32> = (0..n_keys)
                    .flat_map(|pos| cache.dequantize_key(0, pos))
                    .collect();
                let ms = maxsim_score(&queries_flat, &dequant_flat, n_queries, n_keys, dim);
                let rel_err = ((ms - truth) / truth).abs() as f64;
                oct_res.rel_errors.push(rel_err);
                oct_res.available = true;
                let pct = rel_err * 100.0;
                println!(
                    "  {:8} │ {:8} │ OCT      │ {pct:10.4}% │ {ms:8.4}",
                    bits, seed
                );
            }

            // ── PlanarQuant ──
            #[cfg(feature = "planar_quant")]
            {
                use katgpt_rs::planar_quant::{PlanarQuantConfig, PlanarQuantKVCache};
                let config = PlanarQuantConfig {
                    key_bits: bits,
                    val_bits: bits,
                    seed: seed_val,
                    n_layers: 1,
                    kv_dim: dim,
                    max_seq_len: n_keys + 16,
                };
                let mut cache = PlanarQuantKVCache::with_config(&config);
                for (pos, key) in keys.iter().enumerate() {
                    cache.store_key(0, pos, key);
                }
                let dequant_flat: Vec<f32> = (0..n_keys)
                    .flat_map(|pos| cache.dequantize_key(0, pos))
                    .collect();
                let ms = maxsim_score(&queries_flat, &dequant_flat, n_queries, n_keys, dim);
                let rel_err = ((ms - truth) / truth).abs() as f64;
                pq_res.rel_errors.push(rel_err);
                pq_res.available = true;
                let pct = rel_err * 100.0;
                println!(
                    "  {:8} │ {:8} │ PQ       │ {pct:10.4}% │ {ms:8.4}",
                    bits, seed
                );
            }
        }

        // ── Summary for this bits level ──
        println!(
            "  {}─┼─{}─┼─{}─┼─{}─┼─{}",
            "─".repeat(8),
            "─".repeat(8),
            "─".repeat(8),
            "─".repeat(10),
            "─".repeat(8)
        );
        println!("  bits={bits} mean relative error:");

        let all_results: Vec<&MaxSimResult> = vec![&hybrid_res, &oct_res, &pq_res];
        let mut best_name = String::new();
        let mut best_err = f64::INFINITY;

        for res in &all_results {
            match res.available {
                true => {
                    let (mean, std) = mean_std(&res.rel_errors);
                    let name = &res.name;
                    println!(
                        "    {name:10} rel_err = {mean:.4} ± {std:.4} ({pct:.2}%)",
                        pct = mean * 100.0
                    );
                    if mean < best_err {
                        best_err = mean;
                        best_name = res.name.clone();
                    }
                }
                false => {
                    let name = &res.name;
                    println!("    {name:10} (not available)");
                }
            }
        }

        match best_name.is_empty() {
            true => println!("    (no backends available)"),
            false => println!("    🏆 bits={bits}: {best_name} wins (rel_err={best_err:.4})"),
        }
        println!();
    }

    println!("{}", "═".repeat(90));
    println!("  Hybrid OCT+PQ MaxSim Summary:");
    println!("  • Hybrid uses PQ 2D rotation (256 FMAs) + OCT triplet encoding");
    println!(
        "  • If Hybrid MaxSim ≈ PQ MaxSim, PQ rotation preserves angular structure for triplets"
    );
    println!("  • If Hybrid MaxSim ≈ OCT MaxSim, the triplet encoding dominates MaxSim behavior");
    println!();
}
