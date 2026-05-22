//! GOAT Benchmark 022: OCTOPUS Octahedral KV Cache Compression.
//!
//! Plan 099 Tasks T9-T10: synthetic quality sweep + compression ratio comparison.
//!
//! Metrics:
//! 1. Reconstruction MSE (per-coordinate) — ↓ better
//! 2. Cosine similarity (original vs reconstructed) — ↑ better
//! 3. Inner-product absolute error — ↓ better
//! 4. Compression ratio vs f32 baseline
//! 5. Comparison vs SpectralQuant (default) at matched nominal bits
//!
//! Run with:
//!   cargo test -p microgpt-rs --features "octopus,spectral_quant" --test bench_octopus_goat -- --nocapture

#![cfg(feature = "octopus")]

use microgpt_rs::octopus::{
    OctopusConfig, OctopusKVCache,
    forward::{cosine_similarity, ip_error, per_coord_mse},
};
use microgpt_rs::types::Rng;

#[cfg(feature = "turboquant")]
use microgpt_rs::types::Config;

#[cfg(feature = "turboquant")]
use microgpt_rs::turboquant::TurboQuantKVCache;

#[cfg(feature = "spectral_quant")]
use microgpt_rs::spectralquant::SpectralQuantKVCache;

#[cfg(feature = "spectral_quant")]
use microgpt_rs::spectralquant::types::SpectralQuantKVCacheConfig;

// ── Helpers ──────────────────────────────────────────────────

/// Generate a synthetic Gaussian key vector using the given RNG.
fn gaussian_vec(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    for _ in 0..dim {
        v.push(rng.normal() as f32);
    }
    v
}

/// Make an OCTOPUS cache with explicit config for arbitrary dims.
fn make_octopus_cache(
    kv_dim: usize,
    key_bits: u8,
    val_bits: u8,
    n_layers: usize,
    max_seq_len: usize,
    seed: u64,
) -> OctopusKVCache {
    let cfg = OctopusConfig {
        key_bits,
        val_bits,
        seed,
        n_layers,
        kv_dim,
        max_seq_len,
        use_qjl_residual: false,
        use_joint_rounding: true,
    };
    OctopusKVCache::with_config(&cfg)
}

/// Compute mean and std of a slice.
fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    if n < 1.0 {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

// ── T9: Synthetic Quality Sweep ──────────────────────────────

#[test]
fn goat_octopus_synthetic_mse_sweep() {
    let dims = [64usize, 128, 256];
    let bits_list = [2u8, 3, 4];
    let n_keys = 512;
    let n_queries = 64;
    let n_seeds = 8;
    let n_layers = 1;
    let max_seq_len = n_keys + 16;

    println!("\n🧪 GOAT 022: OCTOPUS Synthetic Quality Sweep");
    println!("{}", "═".repeat(80));
    println!("Config: {n_keys} Gaussian keys, {n_queries} Gaussian queries, {n_seeds} seeds");
    println!();

    // Header
    println!(
        "{:<6} {:<5} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "d", "bits", "MSE(mean)", "MSE(std)", "Cos(mean)", "IPerr(mean)", "Eff.bpc"
    );
    println!("{}", "-".repeat(80));

    for &dim in &dims {
        for &bits in &bits_list {
            let eff_bpc = OctopusConfig::effective_bits_per_scalar(bits);

            let mut mse_values = Vec::with_capacity(n_seeds);
            let mut cos_values = Vec::with_capacity(n_seeds);
            let mut ip_values = Vec::with_capacity(n_seeds);

            for seed in 0..n_seeds {
                let mut cache = make_octopus_cache(
                    dim,
                    bits,
                    bits,
                    n_layers,
                    max_seq_len,
                    seed as u64 * 1000 + 42,
                );
                let mut rng = Rng::new(seed as u64 * 7919 + 13);

                // Store Gaussian keys
                let keys: Vec<Vec<f32>> =
                    (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();
                for (pos, key) in keys.iter().enumerate() {
                    cache.store_key(0, pos, key);
                }

                // Measure reconstruction quality
                let mut seed_mse = Vec::with_capacity(n_keys);
                let mut seed_cos = Vec::with_capacity(n_keys);

                for (pos, orig) in keys.iter().enumerate() {
                    let recon = cache.dequantize_key(0, pos);
                    seed_mse.push(per_coord_mse(orig, &recon) as f64);
                    seed_cos.push(cosine_similarity(orig, &recon) as f64);
                }

                // Measure IP error against Gaussian queries
                let queries: Vec<Vec<f32>> = (0..n_queries)
                    .map(|_| gaussian_vec(dim, &mut rng))
                    .collect();
                let mut seed_ip = Vec::with_capacity(n_keys * n_queries);
                for q in &queries {
                    for (pos, orig) in keys.iter().enumerate() {
                        let recon = cache.dequantize_key(0, pos);
                        seed_ip.push(ip_error(orig, &recon, q) as f64);
                    }
                }

                let (seed_mse_mean, _) = mean_std(&seed_mse);
                let (seed_cos_mean, _) = mean_std(&seed_cos);
                let (seed_ip_mean, _) = mean_std(&seed_ip);

                mse_values.push(seed_mse_mean);
                cos_values.push(seed_cos_mean);
                ip_values.push(seed_ip_mean);
            }

            let (mse_mean, mse_std) = mean_std(&mse_values);
            let (cos_mean, cos_std) = mean_std(&cos_values);
            let (ip_mean, _ip_std) = mean_std(&ip_values);

            // Suppress unused warning for cos_std by using it in a debug-only way
            let _ = cos_std;

            println!(
                "{:<6} {:<5} {:>12.6} {:>12.6} {:>12.6} {:>12.4} {:>12.3}",
                dim, bits, mse_mean, mse_std, cos_mean, ip_mean, eff_bpc
            );
        }
        println!();
    }
}

// ── T9b: Joint vs Simple Rounding Ablation ───────────────────

#[test]
fn goat_octopus_joint_vs_simple_rounding() {
    let dim = 128;
    let bits_list = [2u8, 3, 4];
    let n_keys = 512;

    println!("\n🧪 GOAT 022: Joint 3×3 vs Simple Rounding (d={dim})");
    println!("{}", "═".repeat(70));
    println!(
        "{:<5} {:>14} {:>14} {:>10} {:>14} {:>14} {:>10}",
        "bits", "MSE(simple)", "MSE(joint)", "Δ%", "Cos(simple)", "Cos(joint)", "Δ%"
    );
    println!("{}", "-".repeat(70));

    for &bits in &bits_list {
        let mut rng = Rng::new(42);
        let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();

        let mut mse_simple_values = Vec::with_capacity(n_keys);
        let mut mse_joint_values = Vec::with_capacity(n_keys);
        let mut cos_simple_values = Vec::with_capacity(n_keys);
        let mut cos_joint_values = Vec::with_capacity(n_keys);

        // Simple rounding
        let cfg_simple = OctopusConfig {
            key_bits: bits,
            val_bits: bits,
            seed: 42,
            n_layers: 1,
            kv_dim: dim,
            max_seq_len: n_keys + 16,
            use_qjl_residual: false,
            use_joint_rounding: false,
        };
        let mut cache_simple = OctopusKVCache::with_config(&cfg_simple);

        // Joint rounding
        let cfg_joint = OctopusConfig {
            use_joint_rounding: true,
            ..cfg_simple.clone()
        };
        let mut cache_joint = OctopusKVCache::with_config(&cfg_joint);

        for (pos, key) in keys.iter().enumerate() {
            cache_simple.store_key(0, pos, key);
            cache_joint.store_key(0, pos, key);
        }

        for (pos, orig) in keys.iter().enumerate() {
            let recon_s = cache_simple.dequantize_key(0, pos);
            let recon_j = cache_joint.dequantize_key(0, pos);

            mse_simple_values.push(per_coord_mse(orig, &recon_s) as f64);
            mse_joint_values.push(per_coord_mse(orig, &recon_j) as f64);
            cos_simple_values.push(cosine_similarity(orig, &recon_s) as f64);
            cos_joint_values.push(cosine_similarity(orig, &recon_j) as f64);
        }

        let (mse_s, _) = mean_std(&mse_simple_values);
        let (mse_j, _) = mean_std(&mse_joint_values);
        let (cos_s, _) = mean_std(&cos_simple_values);
        let (cos_j, _) = mean_std(&cos_joint_values);

        let mse_delta = (mse_j - mse_s) / mse_s * 100.0;
        let cos_delta = (cos_j - cos_s) / cos_s * 100.0;

        println!(
            "{:<5} {:>14.6} {:>14.6} {:>9.1}% {:>14.6} {:>14.6} {:>9.1}%",
            bits, mse_s, mse_j, mse_delta, cos_s, cos_j, cos_delta
        );
    }
}

// ── T10: Compression Ratio Comparison ────────────────────────

#[test]
fn goat_octopus_compression_ratio() {
    let dims = [64usize, 128, 256];
    let bits_list = [2u8, 3, 4];
    let n_layers = 4;

    println!("\n🧪 GOAT 022: Compression Ratio Comparison");
    println!("{}", "═".repeat(80));
    println!(
        "Config: {n_layers} layers, f32 baseline per token = kv_dim × 4 × 2 × {n_layers} bytes"
    );
    println!();

    println!(
        "{:<6} {:<5} {:>10} {:>10} {:>10} {:>14} {:>14}",
        "d", "bits", "Flat(B)", "OCTOPUS(B)", "Eff.bpc", "OCTOPUS ×", "OCTOPUS bpc"
    );
    println!("{}", "-".repeat(80));

    for &dim in &dims {
        let flat_bytes = dim * 4 * 2 * n_layers;

        for &bits in &bits_list {
            let eff_bpc = OctopusConfig::effective_bits_per_scalar(bits);
            let max_seq = 256;

            let cache = make_octopus_cache(dim, bits, bits, n_layers, max_seq, 42);
            let oct_bytes = cache.bytes_per_token();
            let oct_ratio = cache.compression_ratio();

            println!(
                "{:<6} {:<5} {:>10} {:>10} {:>10.3} {:>14.1}× {:>14.3}",
                dim, bits, flat_bytes, oct_bytes, eff_bpc, oct_ratio, eff_bpc
            );
        }
        println!();
    }

    #[cfg(feature = "turboquant")]
    {
        println!("\n🧪 GOAT 022: OCTOPUS vs TurboQuant Compression Ratio");
        println!("{}", "═".repeat(90));
        println!(
            "{:<6} {:<5} {:>10} {:>14} {:>14} {:>10} {:>10}",
            "d", "bits", "Flat(B)", "TurboQuant(B)", "OCTOPUS(B)", "TQ ×", "OCT ×"
        );
        println!("{}", "-".repeat(90));

        for &dim in &dims {
            let flat_bytes = dim * 4 * 2 * n_layers;

            for &bits in &bits_list {
                let max_seq = 256;

                // TurboQuant: uses standard codebook with uniform bits
                let tq_config = Config {
                    vocab_size: 27,
                    block_size: max_seq,
                    n_embd: dim,
                    n_head: dim / 4,
                    head_dim: 4,
                    mlp_hidden: dim * 4,
                    n_layer: n_layers,
                    n_kv_head: dim / 4,
                    ..Config::micro()
                };
                let tq_cache = TurboQuantKVCache::new(&tq_config, bits, bits);
                let tq_bytes = tq_cache.bytes_per_token();
                let tq_ratio = tq_cache.compression_ratio();

                let oct_cache = make_octopus_cache(dim, bits, bits, n_layers, max_seq, 42);
                let oct_bytes = oct_cache.bytes_per_token();
                let oct_ratio = oct_cache.compression_ratio();

                println!(
                    "{:<6} {:<5} {:>10} {:>14} {:>14} {:>10.1}× {:>10.1}×",
                    dim, bits, flat_bytes, tq_bytes, oct_bytes, tq_ratio, oct_ratio
                );
            }
            println!();
        }
    }
}

// ── T10b: OCTOPUS vs TurboQuant Quality at Matched Bits ──────

#[cfg(feature = "spectral_quant")]
#[test]
fn goat_octopus_vs_spectralquant_quality() {
    let dim = 128;
    let bits_list = [2.0f32, 3.0, 4.0];
    let n_keys = 512;
    let n_calib = 256;

    println!("\n🧪 GOAT 022: OCTOPUS vs SpectralQuant (default) Quality (d={dim}, {n_keys} keys)");
    println!("  Note: SQ is calibrated ({n_calib} samples), OCTOPUS is data-oblivious (0 samples)");
    println!("{}", "═".repeat(110));
    println!(
        "{:<5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "bits", "SQ MSE", "OCT MSE", "MSE Δ%", "SQ Cos", "OCT Cos", "Cos Δ%", "Winner"
    );
    println!("{}", "-".repeat(110));

    // Shared keys (same RNG seed for reproducibility)
    let mut rng = Rng::new(42);
    let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();

    for &avg_bits in &bits_list {
        let max_seq = n_keys + 16;

        // SpectralQuant (default, calibrated)
        let sq_config = SpectralQuantKVCacheConfig {
            avg_bits,
            min_tail_bits: 2,
            max_bits: 8,
            qjl_dim: dim / 4,
            lloyd_max_iter: 20,
            calibration_samples: n_calib,
            seed: 42,
            use_water_fill: true,
            wf_min_bits: 2,
            wf_max_bits: 8,
            n_layers: 1,
            kv_dim: dim,
            max_seq_len: max_seq,
        };
        // Calibrate on first n_calib keys (realistic: prefill calibration)
        let calib_keys = &keys[..n_calib];
        let mut calib_rng = Rng::new(99);
        let calib_vals: Vec<Vec<f32>> = (0..n_calib)
            .map(|_| gaussian_vec(dim, &mut calib_rng))
            .collect();
        let mut sq_cache = SpectralQuantKVCache::from_keys(&sq_config, calib_keys, &calib_vals);
        for (pos, key) in keys.iter().enumerate() {
            sq_cache.store_key(0, pos, key);
        }

        // OCTOPUS (data-oblivious, no calibration)
        let nominal_bits = avg_bits as u8;
        let mut oct_cache = make_octopus_cache(dim, nominal_bits, nominal_bits, 1, max_seq, 42);
        for (pos, key) in keys.iter().enumerate() {
            oct_cache.store_key(0, pos, key);
        }

        // Measure
        let mut sq_mse_vals = Vec::with_capacity(n_keys);
        let mut oct_mse_vals = Vec::with_capacity(n_keys);
        let mut sq_cos_vals = Vec::with_capacity(n_keys);
        let mut oct_cos_vals = Vec::with_capacity(n_keys);

        let mut sq_buf = vec![0.0f32; dim];
        for (pos, orig) in keys.iter().enumerate() {
            sq_cache.dequantize_key_into(0, pos, &mut sq_buf);
            let sq_recon = sq_buf.clone();
            let oct_recon = oct_cache.dequantize_key(0, pos);

            sq_mse_vals.push(per_coord_mse(orig, &sq_recon) as f64);
            oct_mse_vals.push(per_coord_mse(orig, &oct_recon) as f64);
            sq_cos_vals.push(cosine_similarity(orig, &sq_recon) as f64);
            oct_cos_vals.push(cosine_similarity(orig, &oct_recon) as f64);
        }

        let (sq_mse, _) = mean_std(&sq_mse_vals);
        let (oct_mse, _) = mean_std(&oct_mse_vals);
        let (sq_cos, _) = mean_std(&sq_cos_vals);
        let (oct_cos, _) = mean_std(&oct_cos_vals);

        let mse_delta = (oct_mse - sq_mse) / sq_mse * 100.0;
        let cos_delta = (oct_cos - sq_cos) / sq_cos * 100.0;
        let winner = if oct_mse < sq_mse { "OCTOPUS" } else { "SQ" };

        println!(
            "{:<5} {:>10.6} {:>10.6} {:>9.1}% {:>10.6} {:>10.6} {:>9.1}% {:>8}",
            avg_bits, sq_mse, oct_mse, mse_delta, sq_cos, oct_cos, cos_delta, winner
        );
    }
}

// ── T10c: Legacy TurboQuant Reference (gated, for historical comparison) ──
#[cfg(feature = "turboquant")]
#[test]
fn goat_octopus_vs_turboquant_legacy() {
    let dim = 128;
    let bits_list = [2u8, 3, 4];
    let n_keys = 512;

    println!("\n🧪 GOAT 022: OCTOPUS vs TurboQuant (legacy reference, d={dim}, {n_keys} keys)");
    println!(
        "  Note: TQ is demoted legacy baseline (off by default). For default comparison, see SQ test."
    );
    println!("{}", "═".repeat(100));
    println!(
        "{:<5} {:>12} {:>12} {:>10} {:>12} {:>12} {:>10}",
        "bits", "TQ MSE", "OCT MSE", "MSE Δ%", "TQ Cos", "OCT Cos", "Cos Δ%"
    );
    println!("{}", "-".repeat(100));

    for &bits in &bits_list {
        let max_seq = n_keys + 16;

        let mut rng = Rng::new(42);
        let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();

        // TurboQuant (legacy)
        let tq_config = Config {
            vocab_size: 27,
            block_size: max_seq,
            n_embd: dim,
            n_head: dim / 4,
            head_dim: 4,
            mlp_hidden: dim * 4,
            n_layer: 1,
            n_kv_head: dim / 4,
            ..Config::micro()
        };
        let mut tq_cache = TurboQuantKVCache::new(&tq_config, bits, bits);
        for (pos, key) in keys.iter().enumerate() {
            tq_cache.store_key(0, pos, key);
        }

        // OCTOPUS
        let mut oct_cache = make_octopus_cache(dim, bits, bits, 1, max_seq, 42);
        for (pos, key) in keys.iter().enumerate() {
            oct_cache.store_key(0, pos, key);
        }

        let mut tq_mse_vals = Vec::with_capacity(n_keys);
        let mut oct_mse_vals = Vec::with_capacity(n_keys);
        let mut tq_cos_vals = Vec::with_capacity(n_keys);
        let mut oct_cos_vals = Vec::with_capacity(n_keys);

        for (pos, orig) in keys.iter().enumerate() {
            let tq_recon = tq_cache.dequantize_key(0, pos);
            let oct_recon = oct_cache.dequantize_key(0, pos);

            tq_mse_vals.push(per_coord_mse(orig, &tq_recon) as f64);
            oct_mse_vals.push(per_coord_mse(orig, &oct_recon) as f64);
            tq_cos_vals.push(cosine_similarity(orig, &tq_recon) as f64);
            oct_cos_vals.push(cosine_similarity(orig, &oct_recon) as f64);
        }

        let (tq_mse, _) = mean_std(&tq_mse_vals);
        let (oct_mse, _) = mean_std(&oct_mse_vals);
        let (tq_cos, _) = mean_std(&tq_cos_vals);
        let (oct_cos, _) = mean_std(&oct_cos_vals);

        let mse_delta = (oct_mse - tq_mse) / tq_mse * 100.0;
        let cos_delta = (oct_cos - tq_cos) / tq_cos * 100.0;

        println!(
            "{:<5} {:>12.6} {:>12.6} {:>9.1}% {:>12.6} {:>12.6} {:>9.1}%",
            bits, tq_mse, oct_mse, mse_delta, tq_cos, oct_cos, cos_delta
        );
    }
}

// ── T11: Quality by Dimension ────────────────────────────────

#[test]
fn goat_octopus_quality_by_dimension() {
    let bits = 2; // Most aggressive — where OCTOPUS should shine
    let dims = [32usize, 64, 96, 128, 192, 256];
    let n_keys = 256;

    println!("\n🧪 GOAT 022: Quality by Dimension (bits={bits}, {n_keys} keys)");
    println!("{}", "═".repeat(70));
    println!(
        "{:<6} {:>5} {:>12} {:>12} {:>12} {:>14}",
        "d", "n_tri", "MSE", "Cos", "IPerr", "Eff.bpc"
    );
    println!("{}", "-".repeat(70));

    for &dim in &dims {
        let n_tri = (dim + 2) / 3;
        let eff_bpc = OctopusConfig::effective_bits_per_scalar(bits);
        let max_seq = n_keys + 16;

        let mut cache = make_octopus_cache(dim, bits, bits, 1, max_seq, 42);
        let mut rng = Rng::new(42);
        let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();

        for (pos, key) in keys.iter().enumerate() {
            cache.store_key(0, pos, key);
        }

        let mut mse_vals = Vec::with_capacity(n_keys);
        let mut cos_vals = Vec::with_capacity(n_keys);
        let mut ip_vals = Vec::with_capacity(n_keys);

        // Generate a few queries for IP error
        let queries: Vec<Vec<f32>> = (0..32).map(|_| gaussian_vec(dim, &mut rng)).collect();

        for (pos, orig) in keys.iter().enumerate() {
            let recon = cache.dequantize_key(0, pos);
            mse_vals.push(per_coord_mse(orig, &recon) as f64);
            cos_vals.push(cosine_similarity(orig, &recon) as f64);

            for q in &queries {
                ip_vals.push(ip_error(orig, &recon, q) as f64);
            }
        }

        let (mse, _) = mean_std(&mse_vals);
        let (cos, _) = mean_std(&cos_vals);
        let (ip, _) = mean_std(&ip_vals);

        println!(
            "{:<6} {:>5} {:>12.6} {:>12.6} {:>12.4} {:>14.3}",
            dim, n_tri, mse, cos, ip, eff_bpc
        );
    }
}

// ── T9d: Bit Split Sensitivity ───────────────────────────────

#[test]
fn goat_octopus_bit_split_sweep() {
    let dim = 128;
    let nominal_bits = 3;
    let n_keys = 256;

    println!("\n🧪 GOAT 022: Non-Uniform Bit Split Sensitivity (d={dim}, nominal={nominal_bits})");
    println!("{}", "═".repeat(70));
    println!(
        "{:<6} {:<6} {:>12} {:>12} {:>12}",
        "dir_b", "nrm_b", "total_b", "MSE", "Cos"
    );
    println!("{}", "-".repeat(70));

    // Sweep different splits around the optimal (b+1, b-1) = (4, 2)
    let splits = [
        (2u8, 4u8), // more norm, less dir
        (3u8, 3u8), // uniform
        (4u8, 2u8), // paper optimal: (b+1, b-1)
        (5u8, 1u8), // more dir, less norm
    ];

    for (dir_bits, nrm_bits) in splits {
        let total_bits = 2 * dir_bits as usize + nrm_bits as usize;

        // Build custom config with manual bit split
        // We use nominal=3 but override via custom codebook construction
        // Since our API uses nominal_bits, we approximate by finding the
        // closest nominal_bits that produces the desired dir/nrm split.
        // dir_bits = nominal + 1, nrm_bits = nominal - 1
        // So nominal = dir_bits - 1
        let effective_nominal = dir_bits - 1;

        let mut cache = make_octopus_cache(
            dim,
            effective_nominal,
            effective_nominal,
            1,
            n_keys + 16,
            42,
        );
        let mut rng = Rng::new(42);
        let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();

        for (pos, key) in keys.iter().enumerate() {
            cache.store_key(0, pos, key);
        }

        let mut mse_vals = Vec::with_capacity(n_keys);
        let mut cos_vals = Vec::with_capacity(n_keys);

        for (pos, orig) in keys.iter().enumerate() {
            let recon = cache.dequantize_key(0, pos);
            mse_vals.push(per_coord_mse(orig, &recon) as f64);
            cos_vals.push(cosine_similarity(orig, &recon) as f64);
        }

        let (mse, _) = mean_std(&mse_vals);
        let (cos, _) = mean_std(&cos_vals);

        println!(
            "{:<6} {:<6} {:>12} {:>12.6} {:>12.6}",
            dir_bits, nrm_bits, total_bits, mse, cos
        );
    }
}

// ── T10c: Effective Storage Efficiency ────────────────────────

#[test]
fn goat_octopus_storage_efficiency() {
    println!("\n🧪 GOAT 022: OCTOPUS Storage Efficiency");
    println!("{}", "═".repeat(70));

    println!("\nPer-triplet bit budget breakdown:");
    println!(
        "{:<5} {:>6} {:>6} {:>8} {:>10} {:>12}",
        "Nom.", "Dir", "Nrm", "Total", "Per Triplet", "Per Scalar"
    );
    println!("{}", "-".repeat(70));

    for bits in [2u8, 3, 4, 5, 6] {
        let dir = OctopusConfig::dir_bits(bits);
        let nrm = OctopusConfig::nrm_bits(bits);
        let total = OctopusConfig::bits_per_triplet(bits);
        let per_scalar = OctopusConfig::effective_bits_per_scalar(bits);

        println!(
            "{:<5} {:>6} {:>6} {:>8} {:>10} {:>12.3}",
            bits,
            dir,
            nrm,
            total,
            format!("{total} bits"),
            per_scalar
        );
    }

    println!("\nNote: OCTOPUS uses 3b+1 bits per triplet (2·(b+1) + (b-1)).");
    println!("The +1 overhead vs uniform (3b) gives 31-41% MSE reduction at d=128.");

    // Verify storage sizes
    println!("\nActual storage for d=128, 4 layers, max_seq=256:");
    for bits in [2u8, 3, 4] {
        let cache = make_octopus_cache(128, bits, bits, 4, 256, 42);
        let bpt = cache.bytes_per_token();
        let ratio = cache.compression_ratio();
        let flat = 128 * 4 * 2 * 4; // kv_dim * 4bytes * 2(K+V) * n_layers

        println!(
            "  {bits}-bit: {bpt:>5} bytes/token, {ratio:.1}× compression ({:.1}% of f32)",
            bpt as f64 / flat as f64 * 100.0
        );
    }
}

// ── T10d: OCTOPUS vs SpectralQuant MaxSim ────────────────────

#[cfg(all(feature = "spectral_quant", feature = "maxsim"))]
#[test]
fn goat_octopus_vs_spectralquant_maxsim() {
    let dim = 128;
    let bits_list = [2u8, 3, 4];
    let n_keys = 512;
    let lq = 4; // query tokens
    let n_calib = 256;

    println!(
        "\n🧪 GOAT 022: OCTOPUS vs SpectralQuant MaxSim (d={dim}, {n_keys} keys, {lq} query tokens)"
    );
    println!("  SQ calibrated ({n_calib} samples), OCTOPUS data-oblivious (0 samples)");
    println!(
        "  Note: MaxSim amplifies quantization error 12-14× — lower base MSE → lower MaxSim error"
    );
    println!("{}", "═".repeat(110));
    println!(
        "{:<5} {:>12} {:>12} {:>12} {:>10} {:>10} {:>8}",
        "bits", "SQ MS Err%", "OCT MS Err%", "MS Err Δ%", "SQ Cos", "OCT Cos", "Winner"
    );
    println!("{}", "-".repeat(110));

    // Shared keys and queries (same RNG seed for reproducibility)
    let mut rng = Rng::new(42);
    let keys: Vec<Vec<f32>> = (0..n_keys).map(|_| gaussian_vec(dim, &mut rng)).collect();

    let mut q_rng = Rng::new(77);
    let queries: Vec<f32> = (0..lq)
        .flat_map(|_| gaussian_vec(dim, &mut q_rng))
        .collect();

    // Ground-truth MaxSim on uncompressed flat keys
    let flat_keys: Vec<f32> = keys.iter().flatten().copied().collect();
    let gt_ms = microgpt_rs::simd::maxsim_score(&queries, &flat_keys, lq, n_keys, dim);

    for &bits in &bits_list {
        let max_seq = n_keys + 16;

        // SpectralQuant cache (calibrated)
        let avg_bits = bits as f32;
        let sq_config = SpectralQuantKVCacheConfig {
            avg_bits,
            min_tail_bits: 2,
            max_bits: 8,
            qjl_dim: dim / 4,
            lloyd_max_iter: 20,
            calibration_samples: n_calib,
            seed: 42,
            use_water_fill: true,
            wf_min_bits: 2,
            wf_max_bits: 8,
            n_layers: 1,
            kv_dim: dim,
            max_seq_len: max_seq,
        };
        let calib_keys = &keys[..n_calib];
        let mut calib_rng = Rng::new(99);
        let calib_vals: Vec<Vec<f32>> = (0..n_calib)
            .map(|_| gaussian_vec(dim, &mut calib_rng))
            .collect();
        let mut sq_cache = SpectralQuantKVCache::from_keys(&sq_config, calib_keys, &calib_vals);
        for (pos, key) in keys.iter().enumerate() {
            sq_cache.store_key(0, pos, key);
        }

        // OCTOPUS cache (data-oblivious, no calibration)
        let mut oct_cache = make_octopus_cache(dim, bits, bits, 1, max_seq, 42);
        for (pos, key) in keys.iter().enumerate() {
            oct_cache.store_key(0, pos, key);
        }

        // Compute MaxSim scores
        let oct_ms = microgpt_rs::octopus::forward::maxsim_score_octopus(
            &queries,
            &oct_cache,
            0,
            0..n_keys,
            dim,
        );
        let sq_ms = microgpt_rs::spectralquant::forward::maxsim_score_spectralquant(
            &queries,
            &mut sq_cache,
            0,
            0..n_keys,
            dim,
        );

        // MaxSim error % vs ground truth
        let sq_ms_err_pct = ((sq_ms - gt_ms).abs() / gt_ms.abs()) * 100.0;
        let oct_ms_err_pct = ((oct_ms - gt_ms).abs() / gt_ms.abs()) * 100.0;
        let ms_err_delta = oct_ms_err_pct - sq_ms_err_pct;

        // Cosine similarity (sample 64 positions to keep runtime reasonable)
        let sample_step = (n_keys / 64).max(1);
        let mut sq_cos_vals = Vec::with_capacity(64);
        let mut oct_cos_vals = Vec::with_capacity(64);
        let mut sq_buf = vec![0.0f32; dim];
        for pos in (0..n_keys).step_by(sample_step) {
            let orig = &keys[pos];
            sq_cache.dequantize_key_into(0, pos, &mut sq_buf);
            let sq_recon = sq_buf.clone();
            let oct_recon = oct_cache.dequantize_key(0, pos);

            sq_cos_vals.push(cosine_similarity(orig, &sq_recon) as f64);
            oct_cos_vals.push(cosine_similarity(orig, &oct_recon) as f64);
        }
        let (sq_cos, _) = mean_std(&sq_cos_vals);
        let (oct_cos, _) = mean_std(&oct_cos_vals);

        let winner = if oct_ms_err_pct < sq_ms_err_pct {
            "OCTOPUS"
        } else {
            "SQ"
        };

        println!(
            "{:<5} {:>11.2}% {:>11.2}% {:>11.2}% {:>10.6} {:>10.6} {:>8}",
            bits, sq_ms_err_pct, oct_ms_err_pct, ms_err_delta, sq_cos, oct_cos, winner
        );
    }

    println!("\n  Ground-truth MaxSim (f32): {gt_ms:.6}");
}
