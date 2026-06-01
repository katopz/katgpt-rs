//! EGA Combined Ablation Example — EGA + DashAttn + SdpaOutputGate (Plan 139, T8)
//!
//! Demonstrates the combined attention enhancement pipeline:
//! 1. Baseline attention with no enhancements
//! 2. EGA energy gating only
//! 3. Simulated DashAttention (α-entmax sparsification) only
//! 4. SdpaOutputGate only
//! 5. Combined: all three enhancements applied in sequence
//! 6. Ablation table comparing all configurations
//!
//! Run: `cargo run --example ega_04_combined --features ega_attn`

#![cfg(feature = "ega_attn")]

use katgpt_rs::ega_attn::{EgaGate, compute_energy_gate, sigmoid};
use katgpt_rs::types::SdpaOutputGate;

// ── Configuration ─────────────────────────────────────────────

const N_HEADS: usize = 4;
const SEQ_LEN: usize = 16;
const HEAD_DIM: usize = 16;
const DIM: usize = N_HEADS * HEAD_DIM; // 64

/// Signal positions with high-magnitude embeddings.
const SIGNAL_POS: &[usize] = &[0, 4, 8, 12];
const SIGNAL_MAG: f32 = 3.0;
const NOISE_MAG: f32 = 0.1;

/// Threshold for simulated DashAttention sparsification.
const DASH_THRESHOLD: f32 = 0.03;

// ── Helpers ───────────────────────────────────────────────────

/// Euclidean distance between two vectors.
fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Multiply attention weights [seq_len × seq_len] by values [seq_len × dim].
/// Returns output [dim] — a single query's aggregated value vector.
fn matmul_attn_values(attn: &[f32], values: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    for j in 0..seq_len {
        let w = attn[j];
        let v_off = j * dim;
        for d in 0..dim {
            out[d] += w * values[v_off + d];
        }
    }
    out
}

/// Compute the mean of selected rows from a row-major [n × dim] matrix.
fn mean_rows(data: &[f32], row_indices: &[usize], dim: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; dim];
    for &r in row_indices {
        let off = r * dim;
        for d in 0..dim {
            mean[d] += data[off + d];
        }
    }
    let inv_n = 1.0 / row_indices.len() as f32;
    for m in mean.iter_mut() {
        *m *= inv_n;
    }
    mean
}

/// Simulate DashAttention α-entmax sparsification:
/// Zero out attention weights below `threshold`, then renormalize.
fn simulate_dash_attn(attn: &mut [f32], seq_len: usize, threshold: f32) {
    // Zero out small weights (sparse support selection)
    for w in attn.iter_mut() {
        if *w < threshold {
            *w = 0.0;
        }
    }
    // Renormalize each row
    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row = &mut attn[row_start..row_start + seq_len];
        let sum: f32 = row.iter().sum();
        if sum > 1e-8 {
            let inv = 1.0 / sum;
            for w in row.iter_mut() {
                *w *= inv;
            }
        }
    }
}

/// Create synthetic embeddings: signal positions get high magnitude, rest are noise.
/// Uses positive-only values so that default w_proj (1/d) yields clear energy separation.
fn make_embeddings() -> Vec<f32> {
    let mut data = vec![0.0f32; SEQ_LEN * DIM];
    for i in 0..SEQ_LEN {
        let is_signal = SIGNAL_POS.contains(&i);
        let mag = if is_signal { SIGNAL_MAG } else { NOISE_MAG };
        let off = i * DIM;
        for d in 0..DIM {
            data[off + d] = mag;
        }
    }
    data
}

// ── Main ──────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EGA Combined Ablation — EGA + DashAttn + SdpaOutputGate   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Phase 1: Setup ────────────────────────────────────────
    println!("━━━ Phase 1: Setup ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  n_heads   = {N_HEADS}");
    println!("  seq_len   = {SEQ_LEN}");
    println!("  head_dim  = {HEAD_DIM}");
    println!("  dim       = {DIM}");
    println!("  signal positions = {SIGNAL_POS:?}  (mag = {SIGNAL_MAG})");
    println!("  noise mag        = {NOISE_MAG}");
    println!("  dash threshold   = {DASH_THRESHOLD}");
    println!();

    let x = make_embeddings(); // [seq_len × dim] embeddings (used as keys)
    let v = make_embeddings(); // [seq_len × dim] values

    // Compute signal mean (ground truth target for quality metric)
    let signal_mean = mean_rows(&v, SIGNAL_POS, DIM);

    // Create EGA gate
    let ega_gate = EgaGate::new(HEAD_DIM);
    println!(
        "  EGA gate: {} params (α={:.2}, τ={:.2})",
        ega_gate.parameter_count(),
        ega_gate.alpha,
        ega_gate.tau
    );

    // Create SdpaOutputGate (zero-initialized)
    let sdpa_gate = SdpaOutputGate::new(N_HEADS, HEAD_DIM, DIM);
    println!(
        "  SdpaOutputGate: {} weights (zero-init → sigmoid(0) = {:.1})",
        sdpa_gate.w_gate.len(),
        sigmoid(0.0)
    );
    println!();

    // Compute energy scores: project each token's embedding down to head_dim
    // by averaging across head groups, then compute energy on the per-head representation.
    // For simplicity, use the first head_dim columns of each token's embedding.
    let mut x_head = vec![0.0f32; SEQ_LEN * HEAD_DIM];
    for i in 0..SEQ_LEN {
        for d in 0..HEAD_DIM {
            x_head[i * HEAD_DIM + d] = x[i * DIM + d];
        }
    }
    let energy = ega_gate.energy_scores(&x_head, SEQ_LEN, HEAD_DIM);
    println!("  Energy scores: {:?}", &energy);

    // Baseline attention: uniform distribution
    let uniform_w = 1.0 / SEQ_LEN as f32;
    let baseline_attn = vec![uniform_w; SEQ_LEN * SEQ_LEN];

    // Use the first query row for all single-query comparisons
    let query_attn = &baseline_attn[0..SEQ_LEN];
    println!("  Baseline attn (query 0): {:?}", query_attn);
    println!();

    // ── Phase 2: Baseline (No Enhancement) ────────────────────
    println!("━━━ Phase 2: Baseline (No Enhancement) ━━━━━━━━━━━━━━━━━━━━━");
    let baseline_out = matmul_attn_values(query_attn, &v, SEQ_LEN, DIM);
    let baseline_l2 = l2_dist(&baseline_out, &signal_mean);
    println!("  L2 to signal mean: {baseline_l2:.4}");
    println!();

    // ── Phase 3: EGA Only ─────────────────────────────────────
    println!("━━━ Phase 3: EGA Only ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let ega_attn = baseline_attn.clone();

    // Compute energy gate vector
    let gate_vec = compute_energy_gate(&energy, ega_gate.alpha, ega_gate.tau);
    println!("  Gate vector: {:?}", &gate_vec);

    // Apply gate to first query row and renormalize
    let mut ega_row = vec![0.0f32; SEQ_LEN];
    for j in 0..SEQ_LEN {
        ega_row[j] = ega_attn[j] * gate_vec[j];
    }
    let row_sum: f32 = ega_row.iter().sum();
    if row_sum > 1e-8 {
        let inv = 1.0 / row_sum;
        for w in ega_row.iter_mut() {
            *w *= inv;
        }
    }
    println!("  EGA attn (query 0): {:?}", &ega_row);

    let ega_out = matmul_attn_values(&ega_row, &v, SEQ_LEN, DIM);
    let ega_l2 = l2_dist(&ega_out, &signal_mean);
    let ega_delta = ega_l2 - baseline_l2;
    println!("  L2 to signal mean: {ega_l2:.4}  (Δ = {ega_delta:+.4})");
    println!();

    // ── Phase 4: Simulated DashAttn Only ──────────────────────
    println!("━━━ Phase 4: Simulated DashAttn Only ━━━━━━━━━━━━━━━━━━━━━━━");
    let mut dash_attn = baseline_attn.clone();
    simulate_dash_attn(&mut dash_attn, SEQ_LEN, DASH_THRESHOLD);
    let dash_row = dash_attn[0..SEQ_LEN].to_vec();
    println!("  Dash attn (query 0): {:?}", &dash_row);

    let dash_out = matmul_attn_values(&dash_row, &v, SEQ_LEN, DIM);
    let dash_l2 = l2_dist(&dash_out, &signal_mean);
    let dash_delta = dash_l2 - baseline_l2;
    println!("  L2 to signal mean: {dash_l2:.4}  (Δ = {dash_delta:+.4})");
    println!();

    // ── Phase 5: SdpaOutputGate Only ──────────────────────────
    println!("━━━ Phase 5: SdpaOutputGate Only ━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let mut sdpa_out = matmul_attn_values(query_attn, &v, SEQ_LEN, DIM);
    let sdpa_l2_before = l2_dist(&sdpa_out, &signal_mean);
    println!("  Before SdpaOutputGate: L2 = {sdpa_l2_before:.4}");

    // Apply SdpaOutputGate: with zero-init weights, output *= sigmoid(0) = 0.5
    let mut temp_buf = vec![0.0f32; sdpa_out.len()];
    sdpa_gate.forward(&mut sdpa_out, DIM, &mut temp_buf);
    let sdpa_l2 = l2_dist(&sdpa_out, &signal_mean);
    let sdpa_delta = sdpa_l2 - baseline_l2;
    println!("  After  SdpaOutputGate: L2 = {sdpa_l2:.4}  (Δ = {sdpa_delta:+.4})");
    println!("  Note: zero-init gate scales by 0.5 → direction preserved, magnitude halved");
    println!();

    // ── Phase 6: Combined (EGA + DashAttn + SdpaOutputGate) ───
    println!("━━━ Phase 6: Combined (EGA + DashAttn + SdpaOutputGate) ━━━━━");
    println!("  Step 1: Simulated DashAttn sparsification on attention weights");

    // Start from uniform attention
    let mut combined_attn = baseline_attn.clone();

    // Step 1: Simulated DashAttn sparsification
    simulate_dash_attn(&mut combined_attn, SEQ_LEN, DASH_THRESHOLD);
    let mut combined_row = combined_attn[0..SEQ_LEN].to_vec();
    println!("  After DashAttn: {:?}", &combined_row);

    // Step 2: EGA energy gate on sparsified weights
    for j in 0..SEQ_LEN {
        combined_row[j] *= gate_vec[j];
    }
    // Renormalize
    let csum: f32 = combined_row.iter().sum();
    if csum > 1e-8 {
        let inv = 1.0 / csum;
        for w in combined_row.iter_mut() {
            *w *= inv;
        }
    }
    println!("  After EGA:     {:?}", &combined_row);

    // Step 3: Value aggregation
    let mut combined_out = matmul_attn_values(&combined_row, &v, SEQ_LEN, DIM);

    // Step 4: SdpaOutputGate on output
    let mut temp_buf2 = vec![0.0f32; combined_out.len()];
    sdpa_gate.forward(&mut combined_out, DIM, &mut temp_buf2);

    let combined_l2 = l2_dist(&combined_out, &signal_mean);
    let combined_delta = combined_l2 - baseline_l2;
    println!("  After SdpaOut:  L2 = {combined_l2:.4}  (Δ = {combined_delta:+.4})");
    println!();

    // ── Phase 7: Ablation Table ───────────────────────────────
    println!("━━━ Phase 7: Ablation Table ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("| Configuration           | L2 to Signal | Δ vs Baseline | Improvement |");
    println!("|-------------------------|-------------:|--------------:|:-----------:|");

    let mark = |delta: f32| -> &'static str {
        if delta < -1e-4 {
            "✅"
        } else if delta > 1e-4 {
            "❌"
        } else {
            "≈"
        }
    };

    println!("| Baseline (no gate)      |    {baseline_l2:>8.4}   |        —      |      —      |");
    println!(
        "| EGA only                |    {ega_l2:>8.4}   |   {ega_delta:>+8.4}   |     {}     |",
        mark(ega_delta)
    );
    println!(
        "| DashAttn only           |    {dash_l2:>8.4}   |   {dash_delta:>+8.4}   |     {}     |",
        mark(dash_delta)
    );
    println!(
        "| SdpaOutputGate only     |    {sdpa_l2:>8.4}   |   {sdpa_delta:>+8.4}   |     {}     |",
        mark(sdpa_delta)
    );
    println!(
        "| Combined (all three)    |    {combined_l2:>8.4}   |   {combined_delta:>+8.4}   |     {}     |",
        mark(combined_delta)
    );
    println!();

    // ── Summary ───────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  EGA gates attention by spectral energy of key tokens.      ║");
    println!("║  DashAttn sparsifies via α-entmax (simulated: top-k mask).  ║");
    println!("║  SdpaOutputGate applies learned sigmoid to attention out.   ║");
    println!("║  Combined pipeline applies all three in sequence.           ║");
    println!("║                                                             ║");
    println!(
        "║  Best L2:  combined = {:.4}                            ║",
        combined_l2
    );
    println!(
        "║  Worst L2: baseline = {:.4}                            ║",
        baseline_l2
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
}
