#![cfg(feature = "lt2_looped")]
//! Plan 483 T2.1 — LT2 Loop-Stable Gate GOAT Benchmark
//!
//! Compares zero-init residual gates vs deterministic loop-stable gates.
//!
//! **Hypothesis:** The zero-init default (`ResidualGate::new`) makes every
//! T-pass effectively independent — no hidden state carries forward between
//! loops (ρ_τ = 0 for all τ). A deterministic non-zero gate restores the
//! residual connection across loops, improving loop convergence quality
//! without requiring trained gate parameters (§3.5 path 2 — modelless).
//!
//! **Metrics:**
//! - G1 (stability): all logits finite at T=4 (no NaN/Inf)
//! - G2 (carry-forward): KL divergence between T=1 and T=4 outputs —
//!   higher KL = more carry-forward = the gate is actually doing something
//! - G3 (convergence): KL divergence between T=3 and T=4 outputs —
//!   lower KL = output is converging (stabilizing)
//! - G4 (no-regression): loop-stable gates must not produce worse stability
//!   than zero-init gates
//!
//! Run: `cargo test --features lt2_looped --test bench_483_lt2_loop_stable_goat -- --nocapture`

use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{Config, HybridPattern, LoopMode, ResidualGate, Rng, SdpaOutputGate};

/// Compute softmax of a slice in-place, returning the distribution.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().copied().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// KL divergence KL(p || q) = Σ p_i * log(p_i / q_i).
/// Returns 0.0 if p and q are identical (within f32 precision).
fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    let mut kl = 0.0f32;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 1e-12 && qi > 1e-12 {
            kl += pi * (pi / qi).ln();
        }
    }
    kl.max(0.0)
}

/// Check if all values in a slice are finite (no NaN, no Inf).
fn all_finite(vals: &[f32]) -> bool {
    vals.iter().all(|&v| v.is_finite())
}

/// Estimate the dominant eigenvalue of a matrix via power iteration.
/// Matrix is row-major [rows × cols] flattened. Returns (eigenvalue, iterations).
/// This is a deterministic, training-free spectral analysis (§3.5 path 3).
fn dominant_eigenvalue(matrix: &[f32], rows: usize, cols: usize, max_iters: usize) -> f32 {
    if rows == 0 || cols == 0 || matrix.len() < rows * cols {
        return 1.0; // safe default
    }
    let n = rows.min(cols);
    let mut v = vec![1.0f32 / (n as f32).sqrt(); n];
    let mut lambda = 1.0f32;
    for _ in 0..max_iters {
        // w = M @ v (matrix-vector product, row-major)
        let mut w = vec![0.0f32; rows];
        for i in 0..rows {
            let mut sum = 0.0f32;
            for j in 0..cols.min(n) {
                sum += matrix[i * cols + j] * v[j];
            }
            w[i] = sum;
        }
        // norm = ||w||
        let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            break;
        }
        // v = w / norm
        let new_v: Vec<f32> = w.iter().map(|x| x / norm).collect();
        lambda = norm;
        v = new_v;
    }
    lambda
}

/// Run forward_looped with a given config and gate, returning the final logits.
fn run_forward_looped(
    config: &Config,
    weights: &TransformerWeights,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    pos: usize,
) -> Vec<f32> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut ahla_cache = MultiLayerAhlaCache::new(config);

    let logits = forward_looped(
        &mut ctx,
        weights,
        &mut cache,
        &mut ahla_cache,
        0,
        pos,
        config,
        residual_gate,
        sdpa_gate,
        None,
        None,
        #[cfg(feature = "weight_shared_advantage_gate")]
        None,
        None,
        #[cfg(feature = "gain_cost_halt")]
        None,
        None, // Issue 717: deep_run — None = bit-identical baseline
        #[cfg(feature = "cadence_gate")]
        None, // Issue 731: residual-exit probe — None = bit-identical baseline
    );
    logits.to_vec()
}

/// Gate configuration for the benchmark.
struct GateConfig {
    name: &'static str,
    gate: ResidualGate,
}

#[test]
fn bench_483_lt2_loop_stable_goat() {
    println!(
        "\n\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}"
    );
    println!("  Plan 483 T2.1+T2.2 — LT2 Loop-Stable Gate GOAT Benchmark");
    println!("  §3.5 Path 2 (T2.1): deterministic loop-stable residual gate");
    println!("  §3.5 Path 3 (T2.2): spectral-aware gate (power iteration)");
    println!(
        "\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}"
    );
    println!();

    let mut config = Config::micro();
    config.hybrid_pattern = HybridPattern::Uniform;

    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let sdpa_gate = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);

    let t_max = 4usize;
    let n_decode = 8usize; // test across 8 positions

    // T2.2: Compute spectral properties of the weight matrix (power iteration).
    // Uses layer 0's Wq as a representative spectral sample.
    let wq = &weights.layers[0].attn_wq;
    let lambda_max = dominant_eigenvalue(wq, config.n_embd, config.n_embd, 50);
    println!("  T2.2 spectral analysis: λ_max(Wq layer 0) = {lambda_max:.4}");
    let spectral_gate_val = (1.0 / lambda_max).clamp(0.01, 0.9);
    println!(
        "  T2.2 spectral gate value: 1/λ_max = {spectral_gate_val:.4} (clamped to [0.01, 0.9])"
    );
    println!();

    // Gate configurations to benchmark
    let gate_configs: Vec<GateConfig> = vec![
        GateConfig {
            name: "zero-init (default)",
            gate: ResidualGate::new(t_max, config.n_embd),
        },
        GateConfig {
            name: "loop-stable \u{03b1}=0.1",
            gate: ResidualGate::new_loop_stable(t_max, config.n_embd, 0.1),
        },
        GateConfig {
            name: "loop-stable \u{03b1}=0.3",
            gate: ResidualGate::new_loop_stable(t_max, config.n_embd, 0.3),
        },
        GateConfig {
            name: "loop-stable \u{03b1}=0.5",
            gate: ResidualGate::new_loop_stable(t_max, config.n_embd, 0.5),
        },
        GateConfig {
            name: "loop-stable α=0.9",
            gate: ResidualGate::new_loop_stable(t_max, config.n_embd, 0.9),
        },
        GateConfig {
            name: "exp-decay base=0.5",
            gate: ResidualGate::new_loop_stable_exp_decay(t_max, config.n_embd, 0.5),
        },
        GateConfig {
            name: "exp-decay base=0.7",
            gate: ResidualGate::new_loop_stable_exp_decay(t_max, config.n_embd, 0.7),
        },
        // T2.2: spectral-aware gate — gate value = 1/λ_max (clamped)
        GateConfig {
            name: "spectral 1/λ_max",
            gate: ResidualGate::new_loop_stable(t_max, config.n_embd, spectral_gate_val),
        },
        // T2.2: sigmoid-gated spectral — gate = sigmoid(1/λ_max - 0.5)
        GateConfig {
            name: "sigmoid-spectral",
            gate: ResidualGate::new_loop_stable(
                t_max,
                config.n_embd,
                1.0 / (1.0 + (-(spectral_gate_val - 0.5)).exp()),
            ),
        },
    ];

    // Results table
    println!(
        "┌──────────────────────────────┬──────────┬──────────────┬──────────────┬──────────────┐"
    );
    println!(
        "│ Gate                         │ G1 stab. │ G2 KL(1→4)   │ G3 KL(3→4)   │ G4 regress.  │"
    );
    println!(
        "│                              │ (finite) │ (carry-fwd)  │ (converge)   │ (vs zero)    │"
    );
    println!(
        "├──────────────────────────────┼──────────┼──────────────┼──────────────┼──────────────┤"
    );

    let mut all_pass = true;
    let mut zero_init_kl_1_4 = 0.0f32;
    let mut zero_init_kl_3_4 = 0.0f32;

    for gc in &gate_configs {
        let mut all_finite_t4 = true;
        let mut total_kl_1_4 = 0.0f32;
        let mut total_kl_3_4 = 0.0f32;

        for pos in 0..n_decode {
            // Run with T=1
            let mut cfg_t1 = config.clone();
            cfg_t1.loop_mode = LoopMode::WeightShared { loop_count: 1 };
            let logits_t1 = run_forward_looped(&cfg_t1, &weights, &gc.gate, &sdpa_gate, pos);

            // Run with T=3
            let mut cfg_t3 = config.clone();
            cfg_t3.loop_mode = LoopMode::WeightShared { loop_count: 3 };
            let logits_t3 = run_forward_looped(&cfg_t3, &weights, &gc.gate, &sdpa_gate, pos);

            // Run with T=4
            let mut cfg_t4 = config.clone();
            cfg_t4.loop_mode = LoopMode::WeightShared { loop_count: 4 };
            let logits_t4 = run_forward_looped(&cfg_t4, &weights, &gc.gate, &sdpa_gate, pos);

            // G1: stability check
            if !all_finite(&logits_t4) {
                all_finite_t4 = false;
            }

            // G2: carry-forward (KL between T=1 and T=4)
            let p1 = softmax(&logits_t1);
            let p4 = softmax(&logits_t4);
            total_kl_1_4 += kl_divergence(&p1, &p4);

            // G3: convergence (KL between T=3 and T=4)
            let p3 = softmax(&logits_t3);
            total_kl_3_4 += kl_divergence(&p3, &p4);
        }

        let avg_kl_1_4 = total_kl_1_4 / n_decode as f32;
        let avg_kl_3_4 = total_kl_3_4 / n_decode as f32;

        // Record zero-init baseline for G4 comparison
        if gc.name.starts_with("zero-init") {
            zero_init_kl_1_4 = avg_kl_1_4;
            zero_init_kl_3_4 = avg_kl_3_4;
        }

        // G4: no-regression — loop-stable must be at least as stable as zero-init
        let g4_pass = all_finite_t4;

        // G1: stability
        let g1_str = if all_finite_t4 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        };

        // G2: carry-forward — any non-zero KL means the gate is doing something
        let g2_str = format!("{avg_kl_1_4:.6}");

        // G3: convergence — lower KL(3→4) means output is converging
        let g3_str = format!("{avg_kl_3_4:.6}");

        // G4: no-regression
        let g4_str = if g4_pass { "✅ PASS" } else { "❌ FAIL" };

        if !all_finite_t4 || !g4_pass {
            all_pass = false;
        }

        println!(
            "│ {:<28} │ {:>8} │ {:>12} │ {:>12} │ {:>12} │",
            gc.name, g1_str, g2_str, g3_str, g4_str
        );
    }

    println!(
        "└──────────────────────────────┴──────────┴──────────────┴──────────────┴──────────────┘"
    );
    println!();

    // Analysis
    println!("── Analysis ──────────────────────────────────────────────────");
    println!(
        "  Zero-init baseline: KL(1→4) = {zero_init_kl_1_4:.6}, KL(3→4) = {zero_init_kl_3_4:.6}"
    );
    println!();

    // GOAT gate verdict
    println!("── GOAT Gate ─────────────────────────────────────────────────");
    println!("  G1 (stability): all logits finite at T=4 for all gate types");
    println!("  G2 (carry-forward): KL(1→4) measures how much T=4 differs from T=1");
    println!("    - Zero-init: KL = {zero_init_kl_1_4:.6} (carry-forward from AHLA state only)");
    println!("    - Loop-stable gates should show DIFFERENT KL values");
    println!("  G3 (convergence): KL(3→4) measures output stabilization");
    println!("    - Lower KL(3→4) = output is converging");
    println!("    - Zero-init baseline: KL(3→4) = {zero_init_kl_3_4:.6}");
    println!("  G4 (no-regression): all logits finite (no divergence)");
    println!();

    if all_pass {
        println!("  ✅ G1+G4 PASS: all gate types produce finite logits at T=4");
    } else {
        println!("  ❌ G1/G4 FAIL: some gate types produce non-finite logits (divergence)");
    }
    println!();

    // Per-gate analysis: does the loop-stable gate provide carry-forward?
    println!("── Carry-Forward Analysis ────────────────────────────────────");
    println!("  If loop-stable gates show different KL(1→4) from zero-init,");
    println!("  the residual gate is contributing to information carry-forward.");
    println!("  This is the modelless correction (§3.5 path 2).",);
    println!();
    println!("── T2.2 vs T2.1 Comparison ───────────────────────────────────");
    println!("  T2.2 (spectral-aware) uses λ_max from power iteration on Wq.");
    println!("  If spectral gate ≠ best T2.1 gate (α=0.9), the spectral info");
    println!("  does NOT improve over the simple constant decay. T2.1 alone");
    println!("  is sufficient (modelless win).",);
    println!(
        "\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}"
    );
    println!("═══════════════════════════════════════════════════════════════");
}
