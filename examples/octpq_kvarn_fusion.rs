//! OCT+PQ+KVarN Fusion Experiment (KVarN T6).
//!
//! Compares six quantization pipelines on synthetic 128×128 tiles:
//!   A) Hadamard → VarN → RTN            (pure KVarN)
//!   B) RTN only                         (baseline, no rotation, no VarN)
//!   C) VarN → RTN                       (VarN alone, no Hadamard)
//!   D) OCT triplet → VarN → RTN         (OCT encoding + VarN)
//!   E) Givens → OCT triplet → VarN → RTN (full OCT+PQ+VarN stack)
//!   F) OCT triplet → RTN                (OCT alone, no VarN — T6 comparison baseline)
//!
//! Measures MSE, cosine similarity, imbalance before/after VarN, and convergence iterations.
//!
//! Run: `cargo run --example octpq_kvarn_fusion --features "kvarn,hybrid_oct_pq"`

#![cfg(all(feature = "kvarn", feature = "hybrid_oct_pq"))]

use katgpt_rs::kvarn::hadamard::hadamard_rows;
use katgpt_rs::kvarn::var_norm::VarNormConfig;

// ── Constants ─────────────────────────────────────────────────────

const TILE_ROWS: usize = 128;
const TILE_COLS: usize = 128;
const RTN_BITS: i32 = 4; // 4-bit RTN
const NUM_TILES: usize = 5;
const SEED: u64 = 0xDEADBEEFCAFEBABE;

// ── Deterministic RNG (xorshift64) ───────────────────────────────

struct SeedRng {
    state: u64,
}

impl SeedRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f32 in (-1, 1).
    fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64();
        // Map to [0, 1) via upper 24 bits, then scale to (-1, 1)
        let normalized = ((bits >> 40) as f32) / (1u64 << 24) as f32;
        normalized * 2.0 - 1.0
    }
}

// ── Local helpers (replicate pub(crate) functions) ───────────────

fn col_stds(tile: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; cols];
    if rows == 0 {
        return result;
    }
    let mut mean = vec![0.0f32; cols];
    for i in 0..rows {
        for j in 0..cols {
            mean[j] += tile[i * cols + j];
        }
    }
    let inv_rows = 1.0 / rows as f32;
    for m in mean.iter_mut() {
        *m *= inv_rows;
    }
    for i in 0..rows {
        for j in 0..cols {
            let d = tile[i * cols + j] - mean[j];
            result[j] += d * d;
        }
    }
    for r in result.iter_mut() {
        *r = (*r * inv_rows).sqrt();
    }
    result
}

fn row_stds(tile: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; rows];
    if cols == 0 {
        return result;
    }
    let inv_cols = 1.0 / cols as f32;
    for (i, res) in result.iter_mut().enumerate() {
        let mut mean = 0.0f32;
        let off = i * cols;
        for j in 0..cols {
            mean += tile[off + j];
        }
        mean *= inv_cols;
        let mut var = 0.0f32;
        for j in 0..cols {
            let d = tile[off + j] - mean;
            var += d * d;
        }
        *res = (var * inv_cols).sqrt();
    }
    result
}

fn ratio_max_min(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let lo = vals.iter().copied().fold(f32::MAX, f32::min).max(1e-8);
    let hi = vals.iter().copied().fold(f32::MIN, f32::max).max(1e-8);
    hi / lo
}

fn imbalance(col_s: &[f32], row_s: &[f32]) -> f32 {
    ratio_max_min(col_s) + ratio_max_min(row_s)
}

// ── Quantization helpers ─────────────────────────────────────────

/// RTN quantize+dequantize: symmetric uniform with given bits.
fn rtn_roundtrip(tile: &mut [f32], bits: i32) {
    let levels = ((1i32 << bits) - 1) as f32;
    // Per-tile scale: find absmax
    let absmax = tile.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if absmax < 1e-12 {
        return;
    }
    let scale = levels / (2.0 * absmax);
    for v in tile.iter_mut() {
        let q = (*v * scale).round().clamp(-levels / 2.0, levels / 2.0);
        *v = q / scale;
    }
}

/// Compute MSE between two buffers.
fn mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        / n
}

/// Compute cosine similarity between two buffers.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

// ── Pipeline results ─────────────────────────────────────────────

struct PipelineResult {
    #[allow(dead_code)]
    name: &'static str,
    mse: f32,
    cosine: f32,
    imbalance_before: f32,
    imbalance_after: f32,
    /// Number of Sinkhorn iterations to reach 95% of total imbalance reduction.
    /// None if no VarN was applied.
    convergence_iter: Option<usize>,
}

/// Compute imbalance metric from a tile directly.
fn compute_imbalance(tile: &[f32], rows: usize, cols: usize) -> f32 {
    let cs = col_stds(tile, rows, cols);
    let rs = row_stds(tile, rows, cols);
    imbalance(&cs, &rs)
}

/// Run variance normalization with convergence tracking.
/// Returns (scales, convergence_iter) where convergence_iter is the first iteration
/// where imbalance reaches within 5% of the final best.
fn variance_normalize_with_convergence(
    tile: &mut [f32],
    rows: usize,
    cols: usize,
    config: &VarNormConfig,
) -> (katgpt_rs::kvarn::var_norm::VarianceNormScales, usize) {
    let imb_before = compute_imbalance(tile, rows, cols);
    let original = tile.to_vec();

    // Run full VarN
    let scales = katgpt_rs::kvarn::variance_normalize(tile, rows, cols, config);

    let imb_after = compute_imbalance(tile, rows, cols);
    let total_reduction = imb_before - imb_after;

    if total_reduction < 1e-8 {
        return (scales, 0);
    }

    // Find first iteration that achieves 95% of total imbalance reduction
    let threshold = imb_before - total_reduction * 0.95;
    let convergence_iter = find_convergence_iter(&original, rows, cols, config, threshold);

    (scales, convergence_iter)
}

/// Find the first iteration where imbalance drops below threshold.
fn find_convergence_iter(
    original: &[f32],
    rows: usize,
    cols: usize,
    config: &VarNormConfig,
    threshold: f32,
) -> usize {
    for k in 1..=config.iterations {
        let partial_config = VarNormConfig {
            iterations: k,
            ..config.clone()
        };
        let mut partial_tile = original.to_vec();
        katgpt_rs::kvarn::variance_normalize(&mut partial_tile, rows, cols, &partial_config);
        let imb = compute_imbalance(&partial_tile, rows, cols);
        if imb <= threshold {
            return k;
        }
    }
    config.iterations
}

/// RTN quantize+dequantize, then undo VarN scales.
/// Returns the reconstructed tile in the original domain.
fn rtn_roundtrip_with_scales(
    tile: &[f32],
    rows: usize,
    cols: usize,
    bits: i32,
    scales: &katgpt_rs::kvarn::var_norm::VarianceNormScales,
) -> Vec<f32> {
    let mut buf = tile.to_vec();
    rtn_roundtrip(&mut buf, bits);
    // Undo VarN scales: reconstructed = tile * s_row[i] * s_col[j]
    for i in 0..rows {
        for j in 0..cols {
            buf[i * cols + j] *= scales.s_row[i] * scales.s_col[j];
        }
    }
    buf
}

/// Run Pipeline A: Hadamard → VarN → RTN (pure KVarN).
fn pipeline_a(
    original: &[f32],
    rows: usize,
    cols: usize,
    config: &VarNormConfig,
) -> PipelineResult {
    let mut tile = original.to_vec();

    // Measure imbalance on raw data
    let cs = col_stds(&tile, rows, cols);
    let rs = row_stds(&tile, rows, cols);
    let imb_before = imbalance(&cs, &rs);

    // Hadamard rows
    hadamard_rows(&mut tile, cols);

    // Variance normalize
    let (_scales, convergence_iter_a) =
        variance_normalize_with_convergence(&mut tile, rows, cols, config);

    // Measure imbalance after VarN
    let cs2 = col_stds(&tile, rows, cols);
    let rs2 = row_stds(&tile, rows, cols);
    let imb_after = imbalance(&cs2, &rs2);

    // RTN quantize+dequantize
    rtn_roundtrip(&mut tile, RTN_BITS);

    // Reconstruct: undo Hadamard (self-inverse), undo VarN scales
    // Note: for a fair MSE comparison we compare the RTN output directly.
    // The actual reconstruction would undo Hadamard and VarN, but since
    // Hadamard is orthogonal and self-inverse, we apply inverse Hadamard
    // to get back to the original domain for MSE measurement.
    hadamard_rows(&mut tile, cols); // inverse = forward (self-inverse)

    // Undo VarN scales: reconstructed = tile * s_row[i] * s_col[j]
    for i in 0..rows {
        for j in 0..cols {
            tile[i * cols + j] *= _scales.s_row[i] * _scales.s_col[j];
        }
    }

    let mse_val = mse(original, &tile);
    let cos = cosine_sim(original, &tile);

    PipelineResult {
        name: "A: Had→VarN→RTN",
        mse: mse_val,
        cosine: cos,
        imbalance_before: imb_before,
        imbalance_after: imb_after,
        convergence_iter: Some(convergence_iter_a),
    }
}

/// Run Pipeline B: RTN only (baseline — no rotation, no VarN).
fn pipeline_b(original: &[f32], rows: usize, cols: usize) -> PipelineResult {
    let mut tile = original.to_vec();

    let cs = col_stds(&tile, rows, cols);
    let rs = row_stds(&tile, rows, cols);
    let imb_before = imbalance(&cs, &rs);
    let imb_after = imb_before; // no VarN applied

    rtn_roundtrip(&mut tile, RTN_BITS);

    let mse_val = mse(original, &tile);
    let cos = cosine_sim(original, &tile);

    PipelineResult {
        name: "B: RTN only      ",
        mse: mse_val,
        cosine: cos,
        imbalance_before: imb_before,
        imbalance_after: imb_after,
        convergence_iter: None,
    }
}

/// Run Pipeline C: VarN → RTN (VarN alone, no Hadamard).
fn pipeline_c(
    original: &[f32],
    rows: usize,
    cols: usize,
    config: &VarNormConfig,
) -> PipelineResult {
    let mut tile = original.to_vec();

    let cs = col_stds(&tile, rows, cols);
    let rs = row_stds(&tile, rows, cols);
    let imb_before = imbalance(&cs, &rs);

    // Variance normalize (no Hadamard)
    let (_scales, convergence_iter_c) =
        variance_normalize_with_convergence(&mut tile, rows, cols, config);

    let cs2 = col_stds(&tile, rows, cols);
    let rs2 = row_stds(&tile, rows, cols);
    let imb_after = imbalance(&cs2, &rs2);

    // RTN
    rtn_roundtrip(&mut tile, RTN_BITS);

    // Undo VarN scales
    for i in 0..rows {
        for j in 0..cols {
            tile[i * cols + j] *= _scales.s_row[i] * _scales.s_col[j];
        }
    }

    let mse_val = mse(original, &tile);
    let cos = cosine_sim(original, &tile);

    PipelineResult {
        name: "C: VarN→RTN      ",
        mse: mse_val,
        cosine: cos,
        imbalance_before: imb_before,
        imbalance_after: imb_after,
        convergence_iter: Some(convergence_iter_c),
    }
}

/// Run Pipeline D: OCT triplet encode → VarN → RTN → inverse triplet.
///
/// OCT triplet encoding: for each group of 3 consecutive values,
/// encode as (v0+v1)/2, (v1+v2)/2, (v2+v0)/2 — a simplified octahedral mapping.
fn pipeline_d(
    original: &[f32],
    rows: usize,
    cols: usize,
    config: &VarNormConfig,
) -> PipelineResult {
    let mut tile = original.to_vec();

    // OCT triplet encode: groups of 3 → 3 averages
    let mut tripled = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in (0..cols).step_by(3) {
            let v0 = tile[i * cols + j];
            let v1 = if j + 1 < cols {
                tile[i * cols + j + 1]
            } else {
                0.0
            };
            let v2 = if j + 2 < cols {
                tile[i * cols + j + 2]
            } else {
                0.0
            };
            tripled[i * cols + j] = (v0 + v1) / 2.0;
            if j + 1 < cols {
                tripled[i * cols + j + 1] = (v1 + v2) / 2.0;
            }
            if j + 2 < cols {
                tripled[i * cols + j + 2] = (v2 + v0) / 2.0;
            }
        }
    }
    tile = tripled;

    let imb_before = compute_imbalance(&tile, rows, cols);
    let (scales, convergence_iter_d) =
        variance_normalize_with_convergence(&mut tile, rows, cols, config);
    let imb_after = compute_imbalance(&tile, rows, cols);

    let reconstructed = rtn_roundtrip_with_scales(&tile, rows, cols, RTN_BITS, &scales);

    // Inverse OCT triplet
    let mut inversed = reconstructed;
    for i in 0..rows {
        for j in (0..cols).step_by(3) {
            let v0 = inversed[i * cols + j];
            let v1 = if j + 1 < cols {
                inversed[i * cols + j + 1]
            } else {
                0.0
            };
            let v2 = if j + 2 < cols {
                inversed[i * cols + j + 2]
            } else {
                0.0
            };
            // Pseudo-inverse: v0_orig ≈ v0 - v1 + v2
            inversed[i * cols + j] = v0 + v1 - v2;
            if j + 1 < cols {
                inversed[i * cols + j + 1] = -v0 + v1 + v2;
            }
            if j + 2 < cols {
                inversed[i * cols + j + 2] = v0 - v1 + v2;
            }
        }
    }

    PipelineResult {
        name: "D: OCT→VarN→RTN  ",
        mse: mse(original, &inversed),
        cosine: cosine_sim(original, &inversed),
        imbalance_before: imb_before,
        imbalance_after: imb_after,
        convergence_iter: Some(convergence_iter_d),
    }
}

/// Run Pipeline E: Givens → OCT triplet → VarN → RTN → inverse OCT → inverse Givens.
///
/// Full OCT+PQ+VarN stack with simplified 2D Givens rotation.
fn pipeline_e(
    original: &[f32],
    rows: usize,
    cols: usize,
    config: &VarNormConfig,
) -> PipelineResult {
    let mut tile = original.to_vec();

    // Simplified Givens rotation: rotate pairs of columns by π/4
    let angle = std::f32::consts::FRAC_PI_4;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    for i in 0..rows {
        for j in (0..cols).step_by(2) {
            if j + 1 < cols {
                let a = tile[i * cols + j];
                let b = tile[i * cols + j + 1];
                tile[i * cols + j] = a * cos_a + b * sin_a;
                tile[i * cols + j + 1] = -a * sin_a + b * cos_a;
            }
        }
    }

    // OCT triplet encode
    let mut tripled = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in (0..cols).step_by(3) {
            let v0 = tile[i * cols + j];
            let v1 = if j + 1 < cols {
                tile[i * cols + j + 1]
            } else {
                0.0
            };
            let v2 = if j + 2 < cols {
                tile[i * cols + j + 2]
            } else {
                0.0
            };
            tripled[i * cols + j] = (v0 + v1) / 2.0;
            if j + 1 < cols {
                tripled[i * cols + j + 1] = (v1 + v2) / 2.0;
            }
            if j + 2 < cols {
                tripled[i * cols + j + 2] = (v2 + v0) / 2.0;
            }
        }
    }
    tile = tripled;

    let imb_before = compute_imbalance(&tile, rows, cols);
    let (scales, convergence_iter) =
        variance_normalize_with_convergence(&mut tile, rows, cols, config);
    let imb_after = compute_imbalance(&tile, rows, cols);

    let reconstructed = rtn_roundtrip_with_scales(&tile, rows, cols, RTN_BITS, &scales);

    // Inverse OCT triplet
    let mut inversed = reconstructed;
    for i in 0..rows {
        for j in (0..cols).step_by(3) {
            let v0 = inversed[i * cols + j];
            let v1 = if j + 1 < cols {
                inversed[i * cols + j + 1]
            } else {
                0.0
            };
            let v2 = if j + 2 < cols {
                inversed[i * cols + j + 2]
            } else {
                0.0
            };
            inversed[i * cols + j] = v0 + v1 - v2;
            if j + 1 < cols {
                inversed[i * cols + j + 1] = -v0 + v1 + v2;
            }
            if j + 2 < cols {
                inversed[i * cols + j + 2] = v0 - v1 + v2;
            }
        }
    }

    // Inverse Givens rotation (transpose: swap sin signs)
    for i in 0..rows {
        for j in (0..cols).step_by(2) {
            if j + 1 < cols {
                let a = inversed[i * cols + j];
                let b = inversed[i * cols + j + 1];
                inversed[i * cols + j] = a * cos_a - b * sin_a;
                inversed[i * cols + j + 1] = a * sin_a + b * cos_a;
            }
        }
    }

    PipelineResult {
        name: "E: Giv→OCT→VN→RTN",
        mse: mse(original, &inversed),
        cosine: cosine_sim(original, &inversed),
        imbalance_before: imb_before,
        imbalance_after: imb_after,
        convergence_iter: Some(convergence_iter),
    }
}

/// Run Pipeline F: OCT triplet → RTN (no VarN — baseline for T6 comparison).
///
/// This isolates the effect of VarN by comparing D (OCT→VarN→RTN) vs F (OCT→RTN).
fn pipeline_f(original: &[f32], rows: usize, cols: usize) -> PipelineResult {
    let mut tile = original.to_vec();

    // OCT triplet encode: groups of 3 → 3 averages
    let mut tripled = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in (0..cols).step_by(3) {
            let v0 = tile[i * cols + j];
            let v1 = if j + 1 < cols {
                tile[i * cols + j + 1]
            } else {
                0.0
            };
            let v2 = if j + 2 < cols {
                tile[i * cols + j + 2]
            } else {
                0.0
            };
            tripled[i * cols + j] = (v0 + v1) / 2.0;
            if j + 1 < cols {
                tripled[i * cols + j + 1] = (v1 + v2) / 2.0;
            }
            if j + 2 < cols {
                tripled[i * cols + j + 2] = (v2 + v0) / 2.0;
            }
        }
    }
    tile = tripled;

    let imb_before = compute_imbalance(&tile, rows, cols);

    // RTN only (no VarN)
    rtn_roundtrip(&mut tile, RTN_BITS);

    // Inverse OCT triplet
    let mut inversed = tile;
    for i in 0..rows {
        for j in (0..cols).step_by(3) {
            let v0 = inversed[i * cols + j];
            let v1 = if j + 1 < cols {
                inversed[i * cols + j + 1]
            } else {
                0.0
            };
            let v2 = if j + 2 < cols {
                inversed[i * cols + j + 2]
            } else {
                0.0
            };
            inversed[i * cols + j] = v0 + v1 - v2;
            if j + 1 < cols {
                inversed[i * cols + j + 1] = -v0 + v1 + v2;
            }
            if j + 2 < cols {
                inversed[i * cols + j + 2] = v0 - v1 + v2;
            }
        }
    }

    PipelineResult {
        name: "F: OCT→RTN (no VN)",
        mse: mse(original, &inversed),
        cosine: cosine_sim(original, &inversed),
        imbalance_before: imb_before,
        imbalance_after: imb_before, // no VarN applied
        convergence_iter: None,
    }
}

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  KVarN T6 — OCT+PQ+KVarN Fusion Experiment                     ║");
    println!("║  Hadamard + Variance Normalization + RTN                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Tile size:   {TILE_ROWS}×{TILE_COLS}");
    println!("  RTN bits:    {RTN_BITS}");
    println!("  Num tiles:   {NUM_TILES}");
    println!("  Seed:        0x{SEED:016X}");
    println!();

    let config = VarNormConfig::default();
    println!(
        "  VarNorm config: iters={}, clamp=[{}, {}], tile_size={}",
        config.iterations, config.log_clamp_lo, config.log_clamp_hi, config.tile_size
    );
    println!();

    let mut rng = SeedRng::new(SEED);

    // Accumulate results across tiles
    let mut results_a = Vec::with_capacity(NUM_TILES);
    let mut results_b = Vec::with_capacity(NUM_TILES);
    let mut results_c = Vec::with_capacity(NUM_TILES);
    let mut results_d = Vec::with_capacity(NUM_TILES);
    let mut results_e = Vec::with_capacity(NUM_TILES);
    let mut results_f = Vec::with_capacity(NUM_TILES);

    for t in 0..NUM_TILES {
        // Generate random tile
        let mut original = vec![0.0f32; TILE_ROWS * TILE_COLS];
        // Mix of scales for heterogenous magnitudes
        for i in 0..TILE_ROWS {
            let row_scale = 0.1 * (1.0 + (i as f32).ln_1p()); // row-dependent magnitude
            for j in 0..TILE_COLS {
                original[i * TILE_COLS + j] = rng.next_f32() * row_scale * 10.0;
            }
        }

        let ra = pipeline_a(&original, TILE_ROWS, TILE_COLS, &config);
        let rb = pipeline_b(&original, TILE_ROWS, TILE_COLS);
        let rc = pipeline_c(&original, TILE_ROWS, TILE_COLS, &config);
        let rd = pipeline_d(&original, TILE_ROWS, TILE_COLS, &config);
        let re = pipeline_e(&original, TILE_ROWS, TILE_COLS, &config);
        let rf = pipeline_f(&original, TILE_ROWS, TILE_COLS);

        results_a.push(ra);
        results_b.push(rb);
        results_c.push(rc);
        results_d.push(rd);
        results_e.push(re);
        results_f.push(rf);

        println!(
            "  Tile {t}: A={:.6} B={:.6} C={:.6} D={:.6} E={:.6} F={:.6}",
            results_a[t].mse,
            results_b[t].mse,
            results_c[t].mse,
            results_d[t].mse,
            results_e[t].mse,
            results_f[t].mse
        );
    }

    // ── Aggregate results ────────────────────────────────────────
    let avg = |results: &[PipelineResult], f: fn(&PipelineResult) -> f32| -> f32 {
        results.iter().map(f).sum::<f32>() / results.len() as f32
    };

    let avg_mse_a = avg(&results_a, |r| r.mse);
    let avg_mse_b = avg(&results_b, |r| r.mse);
    let avg_mse_c = avg(&results_c, |r| r.mse);
    let avg_mse_d = avg(&results_d, |r| r.mse);
    let avg_mse_e = avg(&results_e, |r| r.mse);
    let avg_mse_f = avg(&results_f, |r| r.mse);

    let avg_cos_a = avg(&results_a, |r| r.cosine);
    let avg_cos_b = avg(&results_b, |r| r.cosine);
    let avg_cos_c = avg(&results_c, |r| r.cosine);
    let avg_cos_d = avg(&results_d, |r| r.cosine);
    let avg_cos_e = avg(&results_e, |r| r.cosine);
    let avg_cos_f = avg(&results_f, |r| r.cosine);

    let avg_imb_before_a = avg(&results_a, |r| r.imbalance_before);
    let avg_imb_after_a = avg(&results_a, |r| r.imbalance_after);
    let avg_imb_before_c = avg(&results_c, |r| r.imbalance_before);
    let avg_imb_after_c = avg(&results_c, |r| r.imbalance_after);
    let avg_imb_before_b = avg(&results_b, |r| r.imbalance_before);
    let avg_imb_before_d = avg(&results_d, |r| r.imbalance_before);
    let avg_imb_after_d = avg(&results_d, |r| r.imbalance_after);
    let avg_imb_before_e = avg(&results_e, |r| r.imbalance_before);
    let avg_imb_after_e = avg(&results_e, |r| r.imbalance_after);

    // ── Comparison table ─────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  Pipeline Comparison (avg over {NUM_TILES} tiles, {TILE_ROWS}×{TILE_COLS}, {RTN_BITS}-bit RTN)"
    );
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  ┌──────────────────┬────────────┬────────────┬────────────┬────────────┐");
    println!("  │ Pipeline         │  Avg MSE   │  Cos Sim   │ Imb Before │ Imb After  │");
    println!("  ├──────────────────┼────────────┼────────────┼────────────┼────────────┤");
    println!(
        "  │ A: Had→VarN→RTN  │ {avg_mse_a:>10.6} │ {avg_cos_a:>10.6} │ {avg_imb_before_a:>10.2} │ {avg_imb_after_a:>10.2} │"
    );
    println!(
        "  │ B: RTN only      │ {avg_mse_b:>10.6} │ {avg_cos_b:>10.6} │ {avg_imb_before_b:>10.2} │         N/A │"
    );
    println!(
        "  │ C: VarN→RTN      │ {avg_mse_c:>10.6} │ {avg_cos_c:>10.6} │ {avg_imb_before_c:>10.2} │ {avg_imb_after_c:>10.2} │"
    );
    println!(
        "  │ D: OCT→VarN→RTN  │ {avg_mse_d:>10.6} │ {avg_cos_d:>10.6} │ {avg_imb_before_d:>10.2} │ {avg_imb_after_d:>10.2} │"
    );
    println!(
        "  │ E: Giv→OCT→VN→RTN│ {avg_mse_e:>10.6} │ {avg_cos_e:>10.6} │ {avg_imb_before_e:>10.2} │ {avg_imb_after_e:>10.2} │"
    );
    let avg_imb_before_f = avg(&results_f, |r| r.imbalance_before);
    println!(
        "  │ F: OCT→RTN       │ {avg_mse_f:>10.6} │ {avg_cos_f:>10.6} │ {avg_imb_before_f:>10.2} │         N/A │"
    );
    println!("  └──────────────────┴────────────┴────────────┴────────────┴────────────┘");
    println!();

    // ── Relative improvements ────────────────────────────────────
    let mse_a_vs_b = (avg_mse_b - avg_mse_a) / avg_mse_b * 100.0;
    let mse_c_vs_b = (avg_mse_b - avg_mse_c) / avg_mse_b * 100.0;
    let mse_d_vs_b = (avg_mse_b - avg_mse_d) / avg_mse_b * 100.0;
    let mse_e_vs_b = (avg_mse_b - avg_mse_e) / avg_mse_b * 100.0;
    let mse_a_vs_c = (avg_mse_c - avg_mse_a) / avg_mse_c * 100.0;
    let mse_d_vs_f = if avg_mse_f > 1e-10 {
        (avg_mse_f - avg_mse_d) / avg_mse_f * 100.0
    } else {
        0.0
    };
    let mse_e_vs_f = if avg_mse_f > 1e-10 {
        (avg_mse_f - avg_mse_e) / avg_mse_f * 100.0
    } else {
        0.0
    };

    println!("  Relative MSE improvement over baseline (B):");
    println!("    A vs B: {mse_a_vs_b:+.2}%  (Had + VarN)");
    println!("    C vs B: {mse_c_vs_b:+.2}%  (VarN only)");
    println!("    D vs B: {mse_d_vs_b:+.2}%  (OCT + VarN)");
    println!("    E vs B: {mse_e_vs_b:+.2}%  (Givens + OCT + VarN)");
    println!("    A vs C: {mse_a_vs_c:+.2}%  (added Hadamard)");
    println!();

    // ── T6: OCT+PQ+VarN vs OCT+PQ alone ────────────────────────
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │ T6: VarN effect on OCT pipelines (D vs F, E vs F)          │");
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();
    println!("    OCT+VarN (D) vs OCT alone (F): {mse_d_vs_f:+.2}% MSE change");
    println!("    Giv+OCT+VarN (E) vs OCT alone (F): {mse_e_vs_f:+.2}% MSE change");
    let t6_d_pass = avg_mse_d <= avg_mse_f;
    let t6_e_pass = avg_mse_e <= avg_mse_f;
    if t6_d_pass || t6_e_pass {
        println!("    ✓ VarN improves OCT pipeline quality");
    } else {
        println!("    ⚠ OCT pseudo-inverse is lossy — VarN may not improve quality");
        println!("      with simplified OCT encoding. Real OCT codec needed.");
    }
    println!();

    // ── Imbalance reduction ──────────────────────────────────────
    let imb_reduction_a = (avg_imb_before_a - avg_imb_after_a) / avg_imb_before_a * 100.0;
    let imb_reduction_c = (avg_imb_before_c - avg_imb_after_c) / avg_imb_before_c * 100.0;
    let imb_reduction_d = (avg_imb_before_d - avg_imb_after_d) / avg_imb_before_d * 100.0;
    let imb_reduction_e = (avg_imb_before_e - avg_imb_after_e) / avg_imb_before_e * 100.0;

    println!("  VarN imbalance reduction:");
    println!(
        "    Pipeline A: {imb_reduction_a:.1}%  ({avg_imb_before_a:.2} → {avg_imb_after_a:.2})"
    );
    println!(
        "    Pipeline C: {imb_reduction_c:.1}%  ({avg_imb_before_c:.2} → {avg_imb_after_c:.2})"
    );
    println!(
        "    Pipeline D: {imb_reduction_d:.1}%  ({avg_imb_before_d:.2} → {avg_imb_after_d:.2})"
    );
    println!(
        "    Pipeline E: {imb_reduction_e:.1}%  ({avg_imb_before_e:.2} → {avg_imb_after_e:.2})"
    );
    println!();

    // ── Convergence analysis ───────────────────────────────────
    let avg_convergence = |results: &[PipelineResult]| -> f32 {
        let iters: Vec<f32> = results
            .iter()
            .filter_map(|r| r.convergence_iter)
            .map(|k| k as f32)
            .collect();
        if iters.is_empty() {
            return f32::NAN;
        }
        iters.iter().sum::<f32>() / iters.len() as f32
    };

    let conv_a = avg_convergence(&results_a);
    let conv_c = avg_convergence(&results_c);
    let conv_d = avg_convergence(&results_d);
    let conv_e = avg_convergence(&results_e);

    println!("  VarN convergence (iters to 95% of total imbalance reduction):");
    if !conv_a.is_nan() {
        println!("    Pipeline A (Had+VarN):   {conv_a:.1} iters (target ≤ 4)");
    }
    if !conv_c.is_nan() {
        println!("    Pipeline C (VarN only):  {conv_c:.1} iters (target ≤ 4)");
    }
    if !conv_d.is_nan() {
        println!("    Pipeline D (OCT+VarN):   {conv_d:.1} iters (target ≤ 4)");
    }
    if !conv_e.is_nan() {
        println!("    Pipeline E (Giv+OCT+VN): {conv_e:.1} iters (target ≤ 4)");
    }
    let conv_target = 4.0;
    let conv_pass = [conv_a, conv_c, conv_d, conv_e]
        .iter()
        .all(|c| c.is_nan() || *c <= conv_target);
    if conv_pass {
        println!("    ✓ T6 convergence target met: all pipelines ≤ 4 iters");
    } else {
        println!("    ⚠ Some pipelines need > 4 iters for 95% convergence");
        println!("      Givens rotation provides partial equalization; VarN adds remaining.");
    }
    println!();

    // ── Hypothesis check ─────────────────────────────────────────
    println!("  ┌─────────────────────────────────────────────────────────────────┐");
    println!("  │ Hypothesis: VarN should improve quality when combined with any  │");
    println!("  │ rotation method.                                                 │");
    println!("  └─────────────────────────────────────────────────────────────────┘");
    println!();

    let all_beat_b = avg_mse_a < avg_mse_b
        && avg_mse_c < avg_mse_b
        && avg_mse_d < avg_mse_b
        && avg_mse_e < avg_mse_b;
    let any_beat_b = avg_mse_a < avg_mse_b
        || avg_mse_c < avg_mse_b
        || avg_mse_d < avg_mse_b
        || avg_mse_e < avg_mse_b;

    if all_beat_b {
        println!("  ✓ CONFIRMED: All VarN pipelines (A–E) outperform baseline (B).");
        println!(
            "    Best: {:.2}% MSE reduction",
            mse_a_vs_b.max(mse_c_vs_b).max(mse_d_vs_b).max(mse_e_vs_b)
        );
    } else if any_beat_b {
        println!("  ⚠ PARTIAL: Some VarN pipelines beat baseline but not all.");
        let winners: Vec<&str> = [
            (avg_mse_a < avg_mse_b, "A (Had+VarN)"),
            (avg_mse_c < avg_mse_b, "C (VarN)"),
            (avg_mse_d < avg_mse_b, "D (OCT+VarN)"),
            (avg_mse_e < avg_mse_b, "E (Givens+OCT+VarN)"),
        ]
        .iter()
        .filter(|(wins, _)| *wins)
        .map(|(_, name)| *name)
        .collect();
        println!("    Winners: {}", winners.join(", "));
    } else {
        println!("  ✗ NOT CONFIRMED: No VarN pipeline beat baseline on this data.");
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Experiment complete — {NUM_TILES} tiles × 6 pipelines");
    println!("═══════════════════════════════════════════════════════════════════");
}
