//! MAG G2 — contrast separability gate (Plan 418 Phase 2 T2.2).
//!
//! **THE HEADLINE KILL-IT GATE.** Verifies that contrast directions mined from
//! model-self-labeled classes (here: class labels that the "model" assigns based
//! on its own verdict) produce linearly-separable projections. If this gate
//! fails, the MAG primitive demotes to a research-only Gain — the unsupervised
//! acquisition step is the entire value proposition.
//!
//! Protocol:
//! 1. Generate 200 samples from 2 overlapping Gaussians in ℝ^64
//!    (μ₁ = [+2, 0, ...], μ₂ = [−2, 0, ...], σ configurable).
//! 2. y_M = true Gaussian label (the "model's verdict" — noise-free in this
//!    synthetic gate; the σ-controlled overlap IS the noise).
//! 3. Mine u_Q via mine_contrast_direction(positive=y_M==true, negative=y_M==false).
//! 4. LOO: project all samples onto u_Q, classify each by threshold = midpoint
//!    of the other 199 samples' class-mean projections.
//! 5. Gate: LOO accuracy ≥ 0.75 at σ=1.5; ≥ 0.60 at σ=3.0.

use katgpt_core::mag::mine_contrast_direction;

const D: usize = 64;
const N_TOTAL: usize = 200;
const N_PER_CLASS: usize = 100;

#[inline]
fn gaussian(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(1e-10);
    let u2 = rng.f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

/// Generate samples from 2 Gaussians. Returns (samples, labels) where
/// labels[i] = true for class 1 (μ₁), false for class 2 (μ₂).
fn generate_two_class(rng: &mut fastrand::Rng, sigma: f32) -> (Vec<[f32; D]>, Vec<bool>) {
    let mut mu1 = [0.0_f32; D];
    let mut mu2 = [0.0_f32; D];
    mu1[0] = 2.0;
    mu2[0] = -2.0;

    let mut samples = vec![[0.0_f32; D]; N_TOTAL];
    let mut labels = vec![false; N_TOTAL];

    for i in 0..N_PER_CLASS {
        // Class 1 (label = true)
        for j in 0..D {
            samples[i][j] = mu1[j] + sigma * gaussian(rng);
        }
        labels[i] = true;
    }
    for i in 0..N_PER_CLASS {
        // Class 2 (label = false)
        let idx = N_PER_CLASS + i;
        for j in 0..D {
            samples[idx][j] = mu2[j] + sigma * gaussian(rng);
        }
        labels[idx] = false;
    }

    (samples, labels)
}

/// LOO classification accuracy on a 1D projection.
///
/// For each sample i: compute the class means from the OTHER samples,
/// classify i by which class mean its projection is closest to.
/// This is sign-invariant — works regardless of whether the mined direction
/// points positive→negative or negative→positive.
fn loo_accuracy(projections: &[f32], labels: &[bool]) -> f32 {
    let n = projections.len();
    assert_eq!(n, labels.len());
    let mut correct = 0;

    for i in 0..n {
        // Compute class means excluding sample i.
        let mut sum_true = 0.0_f32;
        let mut cnt_true = 0usize;
        let mut sum_false = 0.0_f32;
        let mut cnt_false = 0usize;
        for j in 0..n {
            if j == i {
                continue;
            }
            if labels[j] {
                sum_true += projections[j];
                cnt_true += 1;
            } else {
                sum_false += projections[j];
                cnt_false += 1;
            }
        }
        let mean_true = if cnt_true > 0 {
            sum_true / cnt_true as f32
        } else {
            0.0
        };
        let mean_false = if cnt_false > 0 {
            sum_false / cnt_false as f32
        } else {
            0.0
        };

        // Classify by nearest class mean (sign-invariant).
        let dist_true = (projections[i] - mean_true).abs();
        let dist_false = (projections[i] - mean_false).abs();
        let predicted = dist_true <= dist_false;
        if predicted == labels[i] {
            correct += 1;
        }
    }

    correct as f32 / n as f32
}

/// Run the G2 gate at a given sigma. Returns (accuracy, cos_to_true_direction).
fn run_g2(sigma: f32, seed: u64) -> (f32, f32) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let (samples, labels) = generate_two_class(&mut rng, sigma);

    // Partition by label into positive (true) and negative (false).
    let positive: Vec<[f32; D]> = samples
        .iter()
        .zip(&labels)
        .filter(|&(_, &l)| l)
        .map(|(s, _)| *s)
        .collect();
    let negative: Vec<[f32; D]> = samples
        .iter()
        .zip(&labels)
        .filter(|&(_, &l)| !l)
        .map(|(s, _)| *s)
        .collect();

    assert!(!positive.is_empty() && !negative.is_empty());

    // Mine contrast direction u_Q.
    let dir = mine_contrast_direction(&positive, &negative).expect("mine_contrast_direction");

    // Cosine with true direction [1, 0, ..., 0] (μ₁ − μ₂ = +4 in dim 0).
    // Absolute value because the contrast direction's sign depends on the
    // positive/negative class assignment — the LINE is what matters, not which
    // way it points.
    let cos_true = dir.as_slice()[0].abs(); // |[1,0,...] · unit_dir| = |dir[0]|

    // Project all samples onto u_Q.
    let projections: Vec<f32> = samples
        .iter()
        .map(|s| {
            let mut z = 0.0;
            for (j, sj) in s.iter().enumerate() {
                z += sj * dir.as_slice()[j];
            }
            z
        })
        .collect();

    let acc = loo_accuracy(&projections, &labels);
    (acc, cos_true)
}

#[test]
fn g2_contrast_separable_sigma_1_5() {
    let (acc, cos) = run_g2(1.5, 0xA200_0001);
    println!("G2 σ=1.5: LOO accuracy = {acc:.4}, cos to true dir = {cos:.4} (gate ≥ 0.75)");
    assert!(
        acc >= 0.75,
        "G2 FAIL at σ=1.5: LOO accuracy = {acc:.4} < 0.75. Primitive must demote to Gain."
    );
}

#[test]
fn g2_contrast_separable_sigma_3_0() {
    let (acc, cos) = run_g2(3.0, 0xA200_0002);
    println!("G2 σ=3.0: LOO accuracy = {acc:.4}, cos to true dir = {cos:.4} (gate ≥ 0.60)");
    assert!(
        acc >= 0.60,
        "G2 FAIL at σ=3.0: LOO accuracy = {acc:.4} < 0.60. Separability ceiling too low."
    );
}
