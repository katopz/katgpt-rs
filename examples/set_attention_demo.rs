//! Plan 354 Phase 4 — Cross-Datapoint Set Attention: consensus-averaging demo.
//!
//! Non-game use case: a sensor network of 16 nodes in 2 environments
//! (8 on a factory floor, 8 outdoors) each emit a noisy 4-dim reading.
//! Set attention refines each reading by sigmoid-weighted averaging over
//! similar sensors — same-environment sensors (anti-correlated centers)
//! reinforce each other while cross-environment sensors suppress each other.
//!
//! This demonstrates the open primitive is useful beyond NPC AI: it is a
//! generic, permutation-equivariant, sigmoid-gated (NEVER softmax) denoiser
//! that needs no training. The identity projections (W_Q = W_K = W_V = I)
//! are the modelless floor — the math reduces to sigmoid-weighted consensus.
//!
//! Run: `cargo run --example set_attention_demo`
//!
//! Source: arXiv:2106.02584 (Kossen et al., NeurIPS 2021) — Non-Parametric
//! Transformers, Attention Between Datapoints (ABD), inference-time half.

use katgpt_core::set_attention::{
    SetAttentionConfig, identity, set_sigmoid_attention_into,
};

const D: usize = 4; // reading dimensionality
const N: usize = 16; // sensor count (8 per cluster)
const K: usize = D; // query/key projection dim (k == d → identity projection)

/// Deterministic LCG noise so the demo is reproducible without an RNG crate.
struct Lcg {
    state: u32,
}
impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }
    fn next_f32(&mut self) -> f32 {
        // Numerical Recipes constants; output in [0, 1).
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to roughly [-0.08, +0.08] noise amplitude.
        ((self.state >> 8) as f32 / 16777216.0 - 0.5) * 0.16
    }
}

fn main() {
    // ── Two cluster centers, chosen anti-correlated so cross-cluster dot
    //    products are negative (sigmoid < 0.5 → suppression) and within-cluster
    //    dot products are positive (sigmoid > 0.5 → reinforcement). This is the
    //    regime where sigmoid attention discriminates; all-positive centers
    //    would make every α > 0.5 and collapse to uniform averaging.
    //
    //    Magnitude 0.5 (not 0.3): the kernel normalises the dot product by √k
    //    (k=4 → ÷2), so the sigmoid argument is (h_i·h_j)/2 · β. With |center|=0.5
    //    the within-cluster dot product is 1.0 → argument 0.5β, giving sharp
    //    discrimination at β=5 (σ(2.5)≈0.92 within, σ(-2.5)≈0.08 cross). At
    //    |center|=0.3 the argument is only 0.18β and β=3 leaves cross-cluster
    //    σ≈0.37 — too much wrong-cluster pull, denoise fails.
    let center_a: [f32; D] = [0.50, -0.50, 0.50, -0.50]; // factory floor
    let center_b: [f32; D] = [-0.50, 0.50, -0.50, 0.50]; // outdoors

    // ── Build noisy sensor readings: 8 per cluster, center + LCG noise.
    let mut lcg = Lcg::new(0x5eed_1234);
    let mut states = vec![0.0f32; N * D];
    let mut true_centers = [[0.0f32; D]; N];
    let mut cluster_labels = [0u8; N];
    let half = N / 2;
    for i in 0..N {
        let (center, label) = if i < half {
            (&center_a, 0u8)
        } else {
            (&center_b, 1u8)
        };
        cluster_labels[i] = label;
        true_centers[i] = *center;
        for d in 0..D {
            states[i * D + d] = center[d] + lcg.next_f32();
        }
    }

    // ── Modelless floor: identity projections. W_Q = W_K = I_d (k == d), W_V = None (identity).
    let w_q = identity(D);
    let w_k = identity(D);

    // ── Config: β=8.0 (very sharp — within-cluster σ≈0.98, cross-cluster σ≈0.02),
    //    γ=0.5 (half-step toward consensus). Single pass: the modelless floor
    //    (identity projections) gives sigmoid-weighted consensus, which is a
    //    genuine but modest denoiser — it is not a trained autoencoder. The
    //    demo honestly reports whatever reduction the single pass achieves.
    let cfg = SetAttentionConfig::new(8.0, 0.5);

    // ── Pre-allocated scratch (zero-alloc steady state, per the primitive's contract).
    let mut output = vec![0.0f32; N * D];
    let mut scratch_q = vec![0.0f32; N * K];
    let mut scratch_k = vec![0.0f32; N * K];
    let mut scratch_alpha = vec![0.0f32; N];

    set_sigmoid_attention_into(
        &states,
        &w_q,
        &w_k,
        None, // W_V = identity
        &mut output,
        &cfg,
        N,
        D,
        K,
        &mut scratch_q,
        &mut scratch_k,
        &mut scratch_alpha,
    )
    .expect("set_attention dims are correct by construction");
    let final_readings: &[f32] = &output;

    // ── Measure noise reduction: mean L2 distance from each reading to its
    //    true cluster center, before vs after.
    let l2 = |reading: &[f32], center: &[f32; D]| -> f32 {
        reading
            .iter()
            .zip(center.iter())
            .map(|(r, c)| (r - c).powi(2))
            .sum::<f32>()
            .sqrt()
    };
    let err_before: f32 =
        (0..N).map(|i| l2(&states[i * D..i * D + D], &true_centers[i])).sum::<f32>() / N as f32;
    let err_after: f32 =
        (0..N).map(|i| l2(&final_readings[i * D..i * D + D], &true_centers[i])).sum::<f32>()
            / N as f32;

    println!("=== Set Attention: Sensor Consensus Averaging (Plan 354 Phase 4) ===");
    println!();
    println!("Setup: {N} sensors (8 per cluster), {D}-dim readings, 2 clusters (factory / outdoor)");
    println!("Modelless floor: W_Q = W_K = W_V = I (no trained projections)");
    println!("Config: beta={} (sharp), gamma={} (single pass)", cfg.beta, cfg.gamma);
    println!();

    let label_str = |l: u8| if l == 0 { "factory" } else { "outdoor" };
    println!("  sensor | cluster  | err_before | err_after  | reduction");
    println!("  -------|----------|------------|------------|----------");
    for i in 0..N {
        let eb = l2(&states[i * D..i * D + D], &true_centers[i]);
        let ea = l2(&final_readings[i * D..i * D + D], &true_centers[i]);
        let pct = if eb > 1e-9 {
            (1.0 - ea / eb) * 100.0
        } else {
            0.0
        };
        println!(
            "  {:>5}  | {:>8} | {:>10.4} | {:>10.4} | {:>5.1}%",
            i + 1,
            label_str(cluster_labels[i]),
            eb,
            ea,
            pct
        );
    }
    println!();
    println!("  mean L2 error before: {:.4}", err_before);
    println!("  mean L2 error after:  {:.4}", err_after);
    let overall = (1.0 - err_after / err_before) * 100.0;
    println!("  overall noise reduction: {:.1}%", overall);

    // ── Verdict: the modelless floor denoises by within-cluster averaging.
    //    A meaningful reduction (> 10%) confirms the primitive adds value as a
    //    generic consensus operator, not just as NPC crowd coherence. Trained
    //    projections (W_Q/W_K from riir-train) would sharpen this further.
    if overall > 10.0 {
        println!();
        println!("  PASS: set attention denoises sensor readings modellessly ({:.1}% reduction).", overall);
        println!("  The open primitive is useful beyond NPC AI — no training required.");
    } else {
        println!();
        println!("  NOTE: reduction {overall:.1}% is below the 10% demo threshold;");
        println!("  the modelless floor (identity W_Q/W_K) is a weak denoiser by design;");
        println!("  trained projections would sharpen discrimination further.");
    }
}
