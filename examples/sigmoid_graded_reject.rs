//! Plan 310 T1.6 — Sigmoid-Graded Reject Confidence demo.
//!
//! Shows tolerant-vs-strict rejection on a synthetic constraint-violation batch.
//! Demonstrates the false-reject rate reduction from HarnessBridge Table 7:
//! tolerant rejection strictly beats strict rejection because false-reject
//! cost > false-pass cost.
//!
//! Run with: `cargo run --example sigmoid_graded_reject --features sigmoid_graded_reject`

use katgpt_core::traits::ConstraintPruner;
use katgpt_rs::pruners::soft_reject::{
    NoRelaxation, RelaxationStrategy, SoftRejectConfig, SoftRejectVerdict,
    soft_reject_decide, soft_reject_with_relax,
};

/// Mock graded pruner: rejects tokens whose "evidence strength" (token_idx)
/// exceeds a threshold, but emits a *sigmoid* confidence instead of a hard bit.
/// This exercises the SoftReject band.
struct GradedRangePruner {
    /// Tokens below `lo` are definitely valid; above `hi` definitely invalid.
    lo: usize,
    hi: usize,
    beta: f32,
}

impl ConstraintPruner for GradedRangePruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        // Hard validity: strictly below the midpoint.
        token_idx < ((self.lo + self.hi) / 2)
    }

    fn reject_confidence(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        // Sigmoid ramp from lo (≈0 reject) to hi (≈1 reject), centered at midpoint.
        let center = ((self.lo + self.hi) as f32) * 0.5;
        let x = self.beta * ((token_idx as f32) - center);
        1.0 / (1.0 + (-x).exp())
    }
}

/// Relaxation strategy: accept any candidate whose token_idx is within
/// `tolerance` of the valid region (simulates "widen the constraint").
struct WidenToleranceRelax {
    boundary: usize,
    tolerance: usize,
}

impl RelaxationStrategy for WidenToleranceRelax {
    fn retry(
        &mut self,
        _depth: usize,
        token_idx: usize,
        _parent_tokens: &[usize],
        _scratch: &mut [u8],
    ) -> bool {
        // Accept if within tolerance of the boundary (relaxed constraint).
        token_idx <= self.boundary + self.tolerance
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Plan 310 T1.6 — Sigmoid-Graded Reject Confidence Demo");
    println!("  HarnessBridge Table 7: tolerant > strict rejection");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Graded pruner: tokens 0..9 ramp from clearly-valid to clearly-invalid.
    let pruner = GradedRangePruner {
        lo: 0,
        hi: 10,
        beta: 1.5,
    };
    let cfg = SoftRejectConfig::default(); // τ_low=0.4, τ_high=0.8
    let candidates: Vec<usize> = (0..12).collect();

    println!("── Decision table (τ_low={}, τ_high={}) ──", cfg.tau_low, cfg.tau_high);
    println!("{:<8} {:<12} {:<14} {:<10}", "token", "conf", "verdict", "is_valid");
    println!("{}", "-".repeat(44));
    for &tok in &candidates {
        let conf = pruner.reject_confidence(0, tok, &[]);
        let verdict = soft_reject_decide(conf, &cfg);
        let is_valid = pruner.is_valid(0, tok, &[]);
        println!(
            "{:<8} {:<12.4} {:<14} {:<10}",
            tok,
            conf,
            format!("{:?}", verdict),
            is_valid
        );
    }
    println!();

    // False-reject rate comparison: strict (is_valid) vs tolerant (soft_reject + relax).
    println!("── False-reject rate: strict vs tolerant ──");
    let mut strict_rejects = 0usize;
    let mut tolerant_rejects = 0usize;
    let mut soft_reject_band_count = 0usize;

    // Tolerant path: widen tolerance by 1 (accept tokens within 1 of boundary).
    let mut relaxer = WidenToleranceRelax {
        boundary: 5,
        tolerance: 1,
    };
    let mut scratch = [0u8; 32];

    for &tok in &candidates {
        let strict_valid = pruner.is_valid(0, tok, &[]);
        if !strict_valid {
            strict_rejects += 1;
        }

        let tolerant_valid =
            soft_reject_with_relax(&pruner, &mut relaxer, &cfg, 0, tok, &[], &mut scratch);
        if !tolerant_valid {
            tolerant_rejects += 1;
        }

        let conf = pruner.reject_confidence(0, tok, &[]);
        if matches!(soft_reject_decide(conf, &cfg), SoftRejectVerdict::SoftReject) {
            soft_reject_band_count += 1;
        }
    }

    let strict_rate = strict_rejects as f32 / candidates.len() as f32;
    let tolerant_rate = tolerant_rejects as f32 / candidates.len() as f32;
    println!("  Strict (is_valid):           {}/{} rejected ({:.1}%)", strict_rejects, candidates.len(), strict_rate * 100.0);
    println!("  Tolerant (soft_reject+relax): {}/{} rejected ({:.1}%)", tolerant_rejects, candidates.len(), tolerant_rate * 100.0);
    println!("  SoftReject band candidates:  {}", soft_reject_band_count);
    println!(
        "  False-reject reduction:      {:.1}pp (strict − tolerant)",
        (strict_rate - tolerant_rate) * 100.0
    );
    println!();

    // Backward-compat demo: a binary pruner (default reject_confidence) must
    // behave identically under soft_reject_with_relax — the SoftReject band is
    // unreachable because the default only emits 0.0 / 1.0.
    println!("── Backward-compat: binary pruner (default reject_confidence) ──");
    struct BinaryPruner(usize);
    impl ConstraintPruner for BinaryPruner {
        fn is_valid(&self, _d: usize, tok: usize, _p: &[usize]) -> bool {
            tok < self.0
        }
    }
    let bin = BinaryPruner(5);
    let mut no_relax = NoRelaxation;
    let mut bin_scratch = [0u8; 8];
    let mut mismatches = 0usize;
    for tok in 0..12 {
        let via_is_valid = bin.is_valid(0, tok, &[]);
        let via_soft =
            soft_reject_with_relax(&bin, &mut no_relax, &cfg, 0, tok, &[], &mut bin_scratch);
        if via_is_valid != via_soft {
            mismatches += 1;
        }
    }
    println!("  Mismatches between is_valid and soft_reject_with_relax: {}", mismatches);
    assert_eq!(mismatches, 0, "binary pruner must reproduce is_valid exactly");
    println!("  ✓ Default binary impl unchanged (zero-behavior-change guarantee).");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Demo complete. The tolerant path rejects fewer candidates");
    println!("  by routing borderline cases through a relaxed retry instead");
    println!("  of hard-failing them outright.");
    println!("═══════════════════════════════════════════════════════════════");
}
