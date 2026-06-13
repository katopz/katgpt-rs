#![cfg(feature = "rt_turbo")]

//! RTPurbo calibration example — offline head classification demo.
//!
//! Demonstrates:
//! 1. Create synthetic attention patterns for 8 heads
//! 2. Compute per-head retrieval scores
//! 3. Calibrate heads into retrieval/local
//! 4. Print classification results and stats
//! 5. Serialize calibration to JSON for offline reuse

use katgpt_rs::rt_turbo::*;
use katgpt_rs::types::RetrievalHeadRole;

// ── Deterministic PRNG (no `rand` dependency) ──────────────────

/// Xorshift64 PRNG for reproducible synthetic data.
struct SeedRng {
    state: u64,
}

impl SeedRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Returns f32 in [0.0, 1.0).
    fn next_f32(&mut self) -> f32 {
        let bits = ((self.next_u64() >> 41) as u32) | 0x3F80_0000;
        f32::from_bits(bits) - 1.0
    }
}

// ── Synthetic Attention Generation ─────────────────────────────

/// Generate a seq_len × seq_len attention matrix for a single head.
///
/// For "retrieval" heads: high attention from post-needle to pre-needle.
/// For "local" heads: attention concentrated on diagonal / nearby positions.
fn make_retrieval_attention(
    seq_len: usize,
    needle_start: usize,
    needle_end: usize,
    post_needle_start: usize,
    post_needle_end: usize,
    strength: f32,
    rng: &mut SeedRng,
) -> Vec<f32> {
    let mut attn = vec![0.01f32; seq_len * seq_len];

    // Uniform background noise
    for val in attn.iter_mut() {
        *val += rng.next_f32() * 0.01;
    }

    // Retrieval pattern: post-needle → pre-needle
    for t in post_needle_start..post_needle_end {
        for j in needle_start..needle_end {
            attn[t * seq_len + j] = strength + rng.next_f32() * 0.1;
        }
    }

    // Normalize each row
    for t in 0..seq_len {
        let row_start = t * seq_len;
        let row_end = row_start + seq_len;
        let sum: f32 = attn[row_start..row_end].iter().sum();
        if sum > 0.0 {
            for val in attn[row_start..row_end].iter_mut() {
                *val /= sum;
            }
        }
    }

    attn
}

/// Generate a local attention matrix (diagonal-heavy).
fn make_local_attention(seq_len: usize, rng: &mut SeedRng) -> Vec<f32> {
    let mut attn = vec![0.001f32; seq_len * seq_len];

    // Diagonal + nearby positions
    for t in 0..seq_len {
        let window_start = t.saturating_sub(4);
        let window_end = (t + 5).min(seq_len);
        for j in window_start..window_end {
            let dist = (t as f32 - j as f32).abs();
            attn[t * seq_len + j] = (1.0 / (1.0 + dist)) + rng.next_f32() * 0.05;
        }
    }

    // Normalize each row
    for t in 0..seq_len {
        let row_start = t * seq_len;
        let row_end = row_start + seq_len;
        let sum: f32 = attn[row_start..row_end].iter().sum();
        if sum > 0.0 {
            for val in attn[row_start..row_end].iter_mut() {
                *val /= sum;
            }
        }
    }

    attn
}

fn main() {
    println!("=== RTPurbo Offline Calibration Demo (Plan 126) ===\n");

    let n_heads = 8;
    let seq_len = 64;
    let needle_start = 4;
    let needle_end = 12;
    let post_needle_start = 48;
    let post_needle_end = 56;
    let mut rng = SeedRng::new(42);

    // ── Step 1: Generate synthetic attention patterns ──────────
    println!("Step 1: Generating synthetic attention for {n_heads} heads (seq_len={seq_len})");
    println!("  Needle span: [{needle_start}, {needle_end})");
    println!("  Post-needle span: [{post_needle_start}, {post_needle_end})\n");

    // Designate heads 0, 3, 5 as retrieval, rest as local
    let retrieval_heads = [0, 3, 5];

    let mut all_scores: Vec<f32> = Vec::with_capacity(n_heads);

    for head_idx in 0..n_heads {
        let is_retrieval = retrieval_heads.contains(&head_idx);
        let attention = if is_retrieval {
            make_retrieval_attention(
                seq_len,
                needle_start,
                needle_end,
                post_needle_start,
                post_needle_end,
                0.8,
                &mut rng,
            )
        } else {
            make_local_attention(seq_len, &mut rng)
        };

        let score = compute_retrieval_score(
            &attention,
            seq_len,
            needle_start,
            needle_end,
            post_needle_start,
            post_needle_end,
        );
        let kind = if is_retrieval { "retrieval" } else { "local" };
        println!("  Head {head_idx}: ground-truth={kind:>9}, R_h={score:.4}");
        all_scores.push(score);
    }

    // ── Step 2: Calibrate heads ────────────────────────────────
    println!("\nStep 2: Running calibration (retrieval_head_ratio=0.15)...\n");

    let config = katgpt_rs::types::RtTurboConfig::default();
    let calibration = calibrate_from_scores(&all_scores, &config);

    // ── Step 3: Print classification results ───────────────────
    println!("Classification Results:");
    println!(
        "  Threshold:  {threshold:.4}",
        threshold = calibration.threshold
    );
    println!(
        "  Retrieval:  {n_ret} heads",
        n_ret = calibration.n_retrieval()
    );
    println!("  Local:      {n_loc} heads", n_loc = calibration.n_local());
    println!();

    println!("  ┌─────────┬──────────┬──────────┬────────────────┐");
    println!("  │ Head ID │ Score    │ Role     │ Classification │");
    println!("  ├─────────┼──────────┼──────────┼────────────────┤");

    for head_idx in 0..n_heads {
        let role = calibration.role_of(head_idx);
        let score = calibration.score_of(head_idx);
        let role_str = match role {
            RetrievalHeadRole::Retrieval => "RETRIEVAL",
            RetrievalHeadRole::Local => "LOCAL",
        };
        let is_retrieval = retrieval_heads.contains(&head_idx);
        let ground_truth = if is_retrieval { "retrieval" } else { "local" };
        let match_indicator = if (role == RetrievalHeadRole::Retrieval) == is_retrieval {
            "✅"
        } else {
            "❌"
        };
        println!(
            "  │ {head_idx:>7} │ {score:>8.4} │ {role_str:>8} │ {ground_truth:>12} {match_indicator} │"
        );
    }
    println!("  └─────────┴──────────┴──────────┴────────────────┘");

    // ── Step 4: Calibration stats ──────────────────────────────
    println!("\nCalibration Statistics:");
    let ret_scores: Vec<f32> = calibration
        .classifications
        .iter()
        .filter(|c| c.role == RetrievalHeadRole::Retrieval)
        .map(|c| c.score)
        .collect();
    let loc_scores: Vec<f32> = calibration
        .classifications
        .iter()
        .filter(|c| c.role == RetrievalHeadRole::Local)
        .map(|c| c.score)
        .collect();

    if let Some(&max_ret) = ret_scores.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
        println!("  Max retrieval score:  {max_ret:.4}");
    }
    if let Some(&min_loc) = loc_scores.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
        println!("  Min local score:      {min_loc:.4}");
    }
    let gap = calibration.threshold;
    println!("  Score gap (threshold): {gap:.4}");
    println!(
        "  Retrieval ratio:      {:.1}% (target: {:.0}%)",
        calibration.n_retrieval() as f32 / calibration.n_heads() as f32 * 100.0,
        config.retrieval_head_ratio * 100.0,
    );

    // ── Step 5: Binary serialization ─────────────────────────────
    println!("\nStep 5: Serialization demo");
    let bytes = calibration.to_bytes().expect("binary serialization");
    let n_bytes = bytes.len();
    println!("  Binary size: {n_bytes} bytes");
    println!(
        "  Round-trip: {}",
        if HeadCalibration::from_bytes(&bytes).is_ok() {
            "✅ OK"
        } else {
            "❌ FAIL"
        }
    );

    // Validate
    match calibration.validate() {
        Ok(()) => println!("  Validation: ✅ OK"),
        Err(e) => println!("  Validation: ❌ {e}"),
    }

    println!(
        "\n✅ Calibration complete — {n_ret}/{n_heads} heads classified as retrieval",
        n_ret = calibration.n_retrieval(),
        n_heads = calibration.n_heads(),
    );
}
