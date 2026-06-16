//! Product-Policy Sharpening example (Plan 283, Research 250).
//!
//! Demonstrates controllable inference-time policy interpolation via the
//! product-policy family `π_w ∝ π̂^{1−w} · π+^w` (Eq. 16). Sweeps the trust
//! weight `w` from 0.0 (skip reasoning) to 2.0 (extrapolation) and shows
//! how the output distribution sharpens.
//!
//! Run with:
//! ```bash
//! cargo run --example pruner_04_product_policy --features product_policy_sharpen --release
//! ```
//!
//! Output: quality vs compute tradeoff curve, showing entropy reduction
//! as `w` increases.

use katgpt_rs::pruners::self_advantage::ProductPolicySharpen;

/// Vocabulary size for the synthetic model.
const VOCAB: usize = 16;

/// Compute Shannon entropy of a probability distribution (in bits).
fn entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0_f32;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    h
}

/// Find the argmax (most probable token).
fn argmax(probs: &[f32]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Product-Policy Sharpening — Controllable Reasoning Trust   ║");
    println!("║  Plan 283, Research 250, arxiv:2511.16886 (Eq. 16)          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Synthetic pre-recursion logits (π̂): broad, uncertain.
    let pre: Vec<f32> = vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 0.5, 2.0, 1.5, 1.0, 0.5, 2.5, 1.0, 0.5, 1.0, 1.5];

    // Synthetic post-recursion logits (π+): sharpened toward token 3.
    let post: Vec<f32> = vec![-1.0, 0.5, 1.0, 6.0, 2.0, -0.5, -1.0, 1.0, 0.5, -0.5, -1.5, 1.5, -0.5, -1.0, -0.5, 0.5];

    println!("📊 Pre-recursion logits (π̂):  argmax={}, entropy={:.3} bits",
        argmax_from_logits(&pre), entropy_from_logits(&pre));
    println!("📊 Post-recursion logits (π+): argmax={}, entropy={:.3} bits",
        argmax_from_logits(&post), entropy_from_logits(&post));
    println!();

    // ── Sweep trust weight w ─────────────────────────────────────
    println!("┌──────────┬──────────────┬───────────────┬──────────────────┐");
    println!("│ w        │ Argmax Token │ Max Prob      │ Entropy (bits)   │");
    println!("├──────────┼──────────────┼───────────────┼──────────────────┤");

    for &w in &[0.0_f32, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] {
        let sharpener = ProductPolicySharpen::new(w);
        let mut probs = vec![0.0_f32; VOCAB];
        sharpener.sharpen_normalized(&pre, &post, &mut probs);
        let top = argmax(&probs);
        let max_p = probs[top];
        let h = entropy(&probs);
        let label = match w {
            w if w == 0.0 => "  (skip reasoning)",
            w if w == 1.0 => "  (full reasoning)",
            w if w > 1.0 => "  (extrapolation)",
            _ => "",
        };
        println!("│ {:<8.2} │ {:<12} │ {:<13.4} │ {:<16.3} │{}",
            w, top, max_p, h, label);
    }
    println!("└──────────┴──────────────┴───────────────┴──────────────────┘");
    println!();

    // ── Visualize the distribution at key w values ───────────────
    println!("📈 Distribution shape at selected w values (token 3 = target):");
    for &w in &[0.0_f32, 0.5, 1.0, 2.0] {
        let sharpener = ProductPolicySharpen::new(w);
        let mut probs = vec![0.0_f32; VOCAB];
        sharpener.sharpen_normalized(&pre, &post, &mut probs);
        print!("   w={:>4.1}: ", w);
        for (i, &p) in probs.iter().enumerate() {
            let bars = (p * 50.0) as usize;
            if i == 3 {
                print!("\x1b[32m{}\x1b[0m", "█".repeat(bars));
            } else if bars > 0 {
                print!("{}", "▁".repeat(bars));
            }
            print!(" ");
        }
        println!();
    }
    println!();
    println!("💡 Key insight: w>1.0 extrapolates beyond the model's own update,");
    println!("   sharpening the distribution further. This is inference-time");
    println!("   temperature control without modifying the model.");
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    logits.iter().map(|&v| (v - max).exp() / sum).collect()
}

fn argmax_from_logits(logits: &[f32]) -> usize {
    argmax(&softmax(logits))
}

fn entropy_from_logits(logits: &[f32]) -> f32 {
    entropy(&softmax(logits))
}
