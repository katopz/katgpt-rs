//! Plan 353 Phase 3 — G2 synthetic IoU→behavioral-delta correlation gate.
//!
//! # What this test reproduces
//!
//! The source paper (arXiv:2606.19317 Hayes/Li/Andreas §3 Fig 5b) reports that
//! IoU between a real attention head and a surrogate is a valid cheap proxy for
//! the expensive causal substitution cost, with Pearson `r > 0.9`. This test
//! reproduces that finding on a **synthetic** harness:
//!
//! 1. Synthesize a "real" attention matrix with paper-Fig-4b-like structure
//!    (first-token + lower-diagonal — the dominant GPT-2 head categories).
//! 2. Generate a family of surrogates at controlled IoU ∈ {0.0, 0.2, 0.4, 0.6,
//!    0.8, 1.0} by blending the real matrix with structured noise.
//! 3. For each surrogate: measure (a) IoU vs real, and (b) behavioral delta —
//!    the KL divergence between softmax(real) and softmax(surrogate) on a
//!    downstream linear-projection "task" (a scalar perplexity proxy).
//! 4. Compute Spearman ρ between IoU and behavioral delta across the surrogate
//!    family. **Target: ρ ≤ −0.9** (negative because high IoU → low delta).
//!
//! # Synthetic-harness limitations (T3.3 — honest disclosure)
//!
//! This is **NOT** a real attention head. The correlation is measured on
//! controlled-noise surrogates, not on real transformer forward passes. The
//! real-head G2 validation (Plan 353 T3.4) is **deferred to riir-ai** — it
//! requires a real transformer forward pass, which lives outside this crate
//! (katgpt-rs ships primitives, not transformer runtimes).
//!
//! Specifically:
//! - The "real" attention matrix is a hand-constructed first-token +
//!   lower-diagonal pattern, not the output of a trained attention head.
//! - The "behavioral delta" is a single-layer linear projection + KL
//!   divergence, not a multi-layer transformer's perplexity shift.
//! - The surrogate family is constructed by noise-blending to hit target IoU
//!   levels, not by running the paper's program-synthesis pipeline.
//!
//! The synthetic harness is a **necessary-but-not-sufficient** check: it
//! verifies that IoU and behavioral delta are anti-correlated on a controlled
//! substrate. The real-head validation is the actual GOAT bar and lives
//! downstream. The synthetic result is reported honestly here; do not claim
//! real-head validation based on this test alone.

#![cfg(feature = "functional_substitution_gate")]
#![allow(clippy::float_cmp)]

use katgpt_core::functional_substitution::iou;

// ──────────────────────────────────────────────────────────────────────────
// Tiny inline numerical helpers — no new heavy deps (per Plan 353 T3.2).
// ──────────────────────────────────────────────────────────────────────────

/// Deterministic splitmix64-based LCG so the harness is reproducible across
/// runs and platforms (no `thread_rng()` nondeterminism).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        // splitmix64.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 32) as f32 / (u32::MAX as f32 + 1.0)
    }
}

/// Row-stochastic softmax over a slice. Used in the downstream "task" model
/// to convert projected scalars into a probability distribution for KL.
///
/// NOTE: softmax is used **here in the G2 synthetic harness only**, to compute
/// a KL divergence — the gate itself and the iou primitive never use softmax
/// (per AGENTS.md sigmoid rule). The harness needs a probability distribution
/// to define KL; softmax is the standard choice for that measurement.
fn softmax_into(logits: &[f32], out: &mut [f32]) {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (o, &l) in out.iter_mut().zip(logits.iter()) {
        let e = (l - m).exp();
        *o = e;
        sum += e;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// Symmetric KL divergence (Jeffrey's): `0.5·(KL(p‖q) + KL(q‖p))`. Used as
/// the behavioral-delta metric because it's symmetric under swap of real and
/// surrogate — the harness doesn't care which is "ground truth".
fn symmetric_kl(p: &[f32], q: &[f32]) -> f32 {
    debug_assert_eq!(p.len(), q.len());
    let n = p.len();
    let mut total = 0.0f32;
    let eps = 1e-12;
    for i in 0..n {
        let pi = p[i].max(eps);
        let qi = q[i].max(eps);
        total += pi * (pi / qi).ln();
        total += qi * (qi / pi).ln();
    }
    0.5 * total
}

/// Rank a slice, returning ranks (1-based, ties resolved by average rank).
/// Inline implementation per Plan 353 T3.2 (no new heavy dep like `ndarray`).
fn rank(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    // Stable sort by value ascending.
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        // Find the run of ties starting at i.
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Average rank for the tied group is (i+1 + j) / 2 (1-based, inclusive).
        let avg = ((i + 1) + j) as f32 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg;
        }
        i = j;
    }
    ranks
}

/// Spearman rank correlation between two slices. Returns ρ ∈ [−1, 1].
///
/// Computed as Pearson correlation of the ranks. Inline per Plan 353 T3.2.
fn spearman(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let rx = rank(x);
    let ry = rank(y);
    let mx = rx.iter().copied().sum::<f32>() / n as f32;
    let my = ry.iter().copied().sum::<f32>() / n as f32;
    let mut cov = 0.0f32;
    let mut vx = 0.0f32;
    let mut vy = 0.0f32;
    for i in 0..n {
        let dx = rx[i] - mx;
        let dy = ry[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 0.0 || vy <= 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

// ──────────────────────────────────────────────────────────────────────────
// Synthetic attention head harness
// ──────────────────────────────────────────────────────────────────────────

/// Synthesize a "real" attention matrix (n × n) with the dominant GPT-2 head
/// structure from paper Fig 4b: a blend of first-token attention and
/// lower-diagonal (induction-head-like) attention. Each row is row-stochastic.
fn synthesize_real_attention(n: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let weight_first_token = if j == 0 { 0.6 } else { 0.0 };
            // Lower-diagonal: attend to the previous token strongly, with
            // decaying weight further back.
            let dist = (i - j) as f32;
            let weight_diag = if j > 0 {
                (-dist).exp() * 0.4
            } else {
                0.0
            };
            a[i * n + j] = weight_first_token + weight_diag;
        }
        // Row-normalize.
        let row_sum: f32 = a[i * n..(i + 1) * n].iter().copied().sum();
        if row_sum > 0.0 {
            let inv = 1.0 / row_sum;
            for j in 0..n {
                a[i * n + j] *= inv;
            }
        }
    }
    a
}

/// Blend the real attention matrix with uniform noise to hit a target IoU.
///
/// `alpha` ∈ [0, 1] controls the blend: `alpha = 1.0` → real; `alpha = 0.0`
/// → pure noise. The resulting matrix is re-row-normalized.
fn blend_with_noise(real: &[f32], n: usize, alpha: f32, rng: &mut Lcg) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        // Generate a random row-stochastic noise distribution.
        let mut noise = vec![0.0f32; n];
        let mut nsum = 0.0f32;
        for nj in noise.iter_mut() {
            *nj = rng.next_f32().max(1e-6);
            nsum += *nj;
        }
        let ninv = if nsum > 0.0 { 1.0 / nsum } else { 0.0 };
        let mut row_sum = 0.0f32;
        for j in 0..n {
            out[i * n + j] = alpha * real[i * n + j] + (1.0 - alpha) * noise[j] * ninv;
            row_sum += out[i * n + j];
        }
        let rinv = if row_sum > 0.0 { 1.0 / row_sum } else { 0.0 };
        for j in 0..n {
            out[i * n + j] *= rinv;
        }
    }
    out
}

/// The downstream "task" model: a fixed linear projection of each row to a
/// scalar "logit", then softmax across rows to produce a probability
/// distribution over query positions. This is the simplest possible
/// downstream consumer — the perplexity proxy.
fn downstream_task_projection(attn: &[f32], n: usize, projection: &[f32]) -> Vec<f32> {
    let mut logits = vec![0.0f32; n];
    for i in 0..n {
        let mut s = 0.0f32;
        for j in 0..n {
            s += attn[i * n + j] * projection[j];
        }
        logits[i] = s;
    }
    let mut probs = vec![0.0f32; n];
    softmax_into(&logits, &mut probs);
    probs
}

// ──────────────────────────────────────────────────────────────────────────
// G2 — Spearman correlation target ρ ≤ −0.9
// ──────────────────────────────────────────────────────────────────────────

/// Measure (IoU, behavioral_delta) pairs across a controlled-IoU surrogate
/// family. Returns (ious, deltas) for Spearman correlation.
fn measure_correlation_pairs(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let real = synthesize_real_attention(n);
    let mut rng = Lcg::new(seed);

    // Fixed downstream projection — deterministic so the only variable is the
    // surrogate. Emphasizes early positions to interact with the first-token
    // structure of the real head.
    let projection: Vec<f32> = (0..n)
        .map(|j| (-((j as f32) / (n as f32))).exp())
        .collect();

    let real_probs = downstream_task_projection(&real, n, &projection);

    // Sweep alpha across a fine grid to get a well-populated scatter.
    // alpha = 1.0 → IoU = 1.0 (identity); alpha = 0.0 → IoU ≈ 0 (pure noise).
    let alphas: Vec<f32> = (0..=20)
        .map(|k| k as f32 / 20.0)
        .collect();

    let mut ious = Vec::with_capacity(alphas.len());
    let mut deltas = Vec::with_capacity(alphas.len());

    for &alpha in &alphas {
        let surrogate = blend_with_noise(&real, n, alpha, &mut rng);
        // IoU averaged across query rows (mean of per-row IoU). This is the
        // paper's per-head IoU definition.
        let mut iou_sum = 0.0f32;
        for i in 0..n {
            let row_real = &real[i * n..(i + 1) * n];
            let row_surr = &surrogate[i * n..(i + 1) * n];
            iou_sum += iou(row_real, row_surr);
        }
        let mean_iou = iou_sum / n as f32;

        let surr_probs = downstream_task_projection(&surrogate, n, &projection);
        let delta = symmetric_kl(&real_probs, &surr_probs);

        ious.push(mean_iou);
        deltas.push(delta);
    }

    (ious, deltas)
}

/// G2 (T3.2): Spearman ρ between IoU and behavioral delta must be ≤ −0.9.
///
/// This reproduces the paper's `r > 0.9` finding on the synthetic harness.
/// Negative because high IoU → low behavioral delta (surrogate is close to
/// real → downstream behavior is close).
#[test]
fn g2_iou_delta_spearman_below_minus_09() {
    let n = 32; // n=32 query positions — enough structure for the head pattern.
    let (ious, deltas) = measure_correlation_pairs(n, 0xB353_A1A9);

    let rho = spearman(&ious, &deltas);

    println!("G2 synthetic harness (n={n}):");
    println!("  IoU range: [{:.4}, {:.4}]", ious_min_max(&ious).0, ious_min_max(&ious).1);
    println!("  Δ range:   [{:.4}, {:.4}]", ious_min_max(&deltas).0, ious_min_max(&deltas).1);
    println!("  Spearman ρ(IoU, Δ) = {rho:.4}");
    println!("  Target: ρ ≤ -0.9");

    // Honest target: ρ must be strongly negative (≤ -0.9). If this fails, we
    // report it honestly — Gain-tier features can ship opt-in with an honest
    // G2 result, per Plan 353's guidance.
    assert!(
        rho <= -0.9,
        "G2 FAILED: Spearman ρ = {rho:.4}, target ≤ -0.9. \
         IoU and behavioral delta are not sufficiently anti-correlated on the \
         synthetic harness."
    );
}

/// G2 robustness: the correlation holds across multiple seeds and sizes
/// (not just the one above). Catches a coincidental-pass on the main seed.
#[test]
fn g2_iou_delta_spearman_robust_across_seeds() {
    let mut worst_rho = 1.0f32; // start at the worst possible (no anti-correlation).
    for n in [16usize, 32, 64] {
        for seed in [0x5EE0_AAAA_u64, 0x5EE0_BBBB, 0x5EE0_CCCC] {
            let (ious, deltas) = measure_correlation_pairs(n, seed);
            let rho = spearman(&ious, &deltas);
            println!("  n={n} seed={seed:#x}: ρ = {rho:.4}");
            if rho < worst_rho {
                worst_rho = rho;
            }
        }
    }
    // Every seed should clear the target — verify by checking the worst.
    assert!(
        worst_rho <= -0.9,
        "G2 robust FAILED: worst Spearman ρ = {worst_rho:.4}, target ≤ -0.9"
    );
}

/// G2 sanity: at alpha=1.0 (identity surrogate), IoU = 1.0 AND delta ≈ 0.
/// This is the anchor point of the correlation — if it fails, the harness
/// itself is broken, not the gate.
#[test]
fn g2_identity_anchor_iou_one_delta_zero() {
    let n = 32;
    let real = synthesize_real_attention(n);
    let projection: Vec<f32> = (0..n)
        .map(|j| (-((j as f32) / (n as f32))).exp())
        .collect();
    let real_probs = downstream_task_projection(&real, n, &projection);

    // alpha = 1.0 → surrogate == real (modulo noise blend of 0).
    let mut rng = Lcg::new(0x1DE7_1717);
    let identity = blend_with_noise(&real, n, 1.0, &mut rng);

    // Per-row IoU averaged.
    let mut iou_sum = 0.0f32;
    for i in 0..n {
        iou_sum += iou(&real[i * n..(i + 1) * n], &identity[i * n..(i + 1) * n]);
    }
    let mean_iou = iou_sum / n as f32;
    assert!((mean_iou - 1.0).abs() < 1e-4, "identity IoU = {mean_iou}, expected 1.0");

    let identity_probs = downstream_task_projection(&identity, n, &projection);
    let delta = symmetric_kl(&real_probs, &identity_probs);
    assert!(delta < 1e-6, "identity delta = {delta}, expected ≈ 0");
}

// ──────────────────────────────────────────────────────────────────────────
// Small helpers
// ──────────────────────────────────────────────────────────────────────────

fn ious_min_max(xs: &[f32]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &x in xs {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
    }
    (lo, hi)
}
