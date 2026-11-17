//! Modelless probe construction — deterministic direction-vector fit from
//! labeled activations, with **no gradient descent**.
//!
//! Plan 292 Phase 4 T4.2 (modelless path). The paper (Kortukov et al. 2026)
//! trains the probe direction `w_B` via logistic regression on
//! `(mid-layer activation, future-behavior-probability label)` pairs gathered
//! by resampling. Per `AGENTS.md`, logistic regression (iterative optimization)
//! lives in `riir-train`, not in this public modelless engine.
//!
//! This module implements the **modelless fallback**: the **mean-difference**
//! (a.k.a. centroid-difference) direction. Given a set of activations split by
//! a binary label, the direction is:
//!
//! ```text
//! w = mean(act | label = 1) − mean(act | label = 0)
//! ```
//!
//! This is the classic closed-form probe used as a baseline in nearly every
//! mechanistic-interpretability probe paper (it is the LDA / Fisher
//! discriminant direction under a shared-spherical-covariance assumption).
//! It is:
//!
//! - **Deterministic** — same inputs → same direction, bit-for-bit.
//! - **Closed-form** — single pass over the data, no iteration, no learning
//!   rate, no convergence. Not "training" in the gradient-descent sense.
//! - **Freeze/thaw compatible** — the output is a `FutureBehaviorProbe`
//!   artifact with an embedded BLAKE3 hash. Load it via
//!   `FutureBehaviorProbe::load_from_bytes` at runtime; swap via
//!   `FutureBehaviorProbe::swap_direction`.
//!
//! # When to use this vs a riir-train probe
//!
//! - **Use this** when you need a *correct-by-construction* probe for
//!   mechanism-level GOAT gates, integration tests, or cold-start bootstrapping
//!   before a trained probe is available. It captures the dominant signal when
//!   the behavior is approximately linearly separable in the activation.
//! - **Prefer a trained probe** (riir-train logistic regression) when you need
//!   the *tightest* decision boundary, calibrated probabilities, or the
//!   behavior is only weakly linearly separable. Logistic regression optimizes
//!   the direction for probability calibration; mean-difference optimizes for
//!   class separation. The paper shows linear probes capture most of the
//!   signal (Research 267 §1.3), so the gap is usually small.
//!
//! # Latent-to-latent contract
//!
//! Input: `(activation, label)` pairs — the activation is a latent residual
//! snapshot, the label is a scalar behavior indicator. Output: a frozen probe
//! artifact. No raw values cross; the direction is itself a latent construct.

use crate::future_probe::FutureBehaviorProbe;

/// A labeled activation sample for modelless probe construction.
///
/// `activation` is the residual-stream snapshot at the sentence-end token at
/// the target layer. `label` is the binary behavior indicator: `true` =
/// behavior exhibited (e.g. the model refused), `false` = not exhibited.
/// Probabilistic labels (the paper's resampling recipe produces a behavior
/// probability in `[0, 1]`) are thresholded at `0.5` by
/// [`construct_probe_via_mean_difference`] — the mean-difference method is a
/// binary classifier by construction. For soft labels, prefer a trained
/// logistic-regression probe (riir-train).
#[derive(Debug, Clone)]
pub struct LabeledActivation<'a> {
    /// Residual-stream activation at the sentence-end token. Length `d_model`.
    pub activation: &'a [f32],
    /// `true` = behavior exhibited, `false` = not.
    pub label: bool,
}

/// Construct a [`FutureBehaviorProbe`] via the modelless mean-difference method.
///
/// Direction: `w = mean(act | label=true) − mean(act | label=false)` (raw,
/// unnormalized — the magnitude carries the class-separation strength, per
/// the standard mech-interp convention). Bias: `−w · centroid`, where
/// `centroid = mean(all activations)`. This places the σ(·) = 0.5 decision
/// boundary at the global activation centroid — a balanced prior when class
/// proportions are unknown.
///
/// # Arguments
///
/// - `samples`: the labeled activations. Must contain at least one `true` and
///   one `false` sample (otherwise the direction is undefined — there's no
///   contrast). All activations must have the same length.
/// - `layer`: the layer index this probe was constructed against. Stored in the
///   artifact so loaders can verify they're reading the right layer.
/// - `behavior`: free-form behavior label (e.g. `"refusal"`).
///
/// # Returns
///
/// `Ok(probe)` if construction succeeded. `Err` if the samples are degenerate
/// (all-same-label, empty, mismatched lengths, or zero-norm direction — the
/// last case means the class centroids are identical, i.e. the behavior is not
/// linearly separable in this activation space at all).
///
/// # Determinism
///
/// Bit-for-bit reproducible given the same `samples` (iteration order matters —
/// pass samples in a stable order if you need a stable BLAKE3 hash across runs).
/// The BLAKE3 hash is computed by [`FutureBehaviorProbe::new`], so the artifact
/// is immediately freeze/thaw-ready.
pub fn construct_probe_via_mean_difference(
    samples: &[LabeledActivation<'_>],
    layer: usize,
    behavior: impl Into<Box<str>>,
) -> Result<FutureBehaviorProbe, MeanDifferenceError> {
    if samples.is_empty() {
        return Err(MeanDifferenceError::NoSamples);
    }
    let d_model = samples[0].activation.len();
    if d_model == 0 {
        return Err(MeanDifferenceError::ZeroDimension);
    }
    for (i, s) in samples.iter().enumerate() {
        if s.activation.len() != d_model {
            return Err(MeanDifferenceError::MismatchedDimension {
                first: d_model,
                at: i,
                got: s.activation.len(),
            });
        }
    }

    // Accumulate per-class sums and the global sum in a single pass.
    let mut sum_pos = vec![0.0_f32; d_model];
    let mut sum_neg = vec![0.0_f32; d_model];
    let mut n_pos: usize = 0;
    let mut n_neg: usize = 0;
    for s in samples {
        let target = if s.label { &mut sum_pos } else { &mut sum_neg };
        for (acc, &v) in target.iter_mut().zip(s.activation.iter()) {
            *acc += v;
        }
        if s.label {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }
    if n_pos == 0 || n_neg == 0 {
        return Err(MeanDifferenceError::SingleClass { n_pos, n_neg });
    }

    // Centroids.
    let inv_pos = 1.0_f32 / n_pos as f32;
    let inv_neg = 1.0_f32 / n_neg as f32;
    let mut direction = vec![0.0_f32; d_model];
    for i in 0..d_model {
        let mean_pos = sum_pos[i] * inv_pos;
        let mean_neg = sum_neg[i] * inv_neg;
        direction[i] = mean_pos - mean_neg;
    }

    // NOTE: the direction is intentionally NOT L2-normalized. The raw
    // mean-difference magnitude carries the class-separation strength — this
    // matches the standard mech-interp convention ("difference of means"
    // probes are used unnormalized; the magnitude is the signal). Normalizing
    // would discard the separation information and produce under-confident
    // probabilities. The paper's logistic regression doesn't normalize either
    // (it optimizes the full magnitude for calibration).
    let mut norm_sq = 0.0_f32;
    for &v in &direction {
        norm_sq += v * v;
    }
    if norm_sq == 0.0 {
        // Class centroids are bit-identical → no linear signal exists.
        return Err(MeanDifferenceError::ZeroNormDirection);
    }

    // Bias: place the σ=0.5 boundary at the global centroid. This is a balanced
    // prior: P(behavior) ≈ class-base-rate at the centroid by construction.
    let total = (n_pos + n_neg) as f32;
    let bias = {
        let mut dot_centroid = 0.0_f32;
        for i in 0..d_model {
            let centroid_i = (sum_pos[i] + sum_neg[i]) / total;
            dot_centroid += direction[i] * centroid_i;
        }
        -dot_centroid
    };

    Ok(FutureBehaviorProbe::new(direction, bias, layer, behavior))
}

// ── I4 (Issue 1228): the difference-of-means direction READOUT ───────────────
//
// CONSUME, don't re-derive: the direction comes from the shipped constructor
// above; this half scores labeled activations through it (σ((h − b)·d) — the
// probe's own forecast shape) and reads the separation as an AUC, refusing a
// no-signal direction TYPED — a promotion never rides a silent AUC ≈ 0.5
// pass. The no-signal arm is wider than the constructor's ZeroNormDirection
// (identical centroids): a non-zero direction can still rank no better than
// chance on the samples it was built from — that census refuses loud too.

/// One [`behavior_readout`] refusal.
#[derive(Debug, Clone, PartialEq)]
pub enum ReadoutError {
    /// The constructor refused (mirror the typed cause; no re-derivation).
    Construction(MeanDifferenceError),
    /// The scored AUC reads at-or-under the chance band (+`band`) — the
    /// direction separates nothing on these samples; promotion refuses.
    NoSignal { auc: f32, band: f32 },
    /// The samples need at least one of each label for an AUC to exist.
    SingleClass,
}

impl std::fmt::Display for ReadoutError {
    #[cold]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Construction(e) => write!(f, "probe construction refused: {e}"),
            Self::NoSignal { auc, band } => {
                write!(
                    f,
                    "no signal: AUC {auc:.4} at or under the chance band +{band:.2} — this direction separates nothing; promotion refuses"
                )
            }
            Self::SingleClass => write!(f, "the samples need one of each label for an AUC"),
        }
    }
}

impl std::error::Error for ReadoutError {}

/// Mann-Whitney AUC of the positive-class scores over the negative-class
/// scores: P(score(pos) > score(neg)) + ½·P(equal). Rank-sum form, deterministic
/// (ties credit ½, sort by `total_cmp` then FIRST-INDEX within ties — the
/// determinism law). O(n log n), one scratch Vec (off the hot path).
#[must_use]
pub fn auc_mann_whitney(pos: &[f32], neg: &[f32]) -> f32 {
    if pos.is_empty() || neg.is_empty() {
        return 0.5;
    }
    // One merged (score, is_pos, index) sort; ties keep insertion order.
    let mut all: Vec<(f32, bool, usize)> = Vec::with_capacity(pos.len() + neg.len());
    all.extend(pos.iter().enumerate().map(|(i, &s)| (s, true, i)));
    all.extend(neg.iter().enumerate().map(|(i, &s)| (s, false, i)));
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    // Rank-sum over the positives; ties credit half a win per tied pair.
    let mut rank_sum = 0.0f64;
    let mut i = 0usize;
    while i < all.len() {
        let mut j = i;
        while j < all.len() && all[j].0 == all[i].0 {
            j += 1;
        }
        // Ranks are 1-based, averaged within the tie block: r = (i+1 + j) / 2.
        let avg_rank = ((i + 1 + j) as f64) / 2.0;
        for item in &all[i..j] {
            if item.1 {
                rank_sum += avg_rank;
            }
        }
        i = j;
    }
    let n_pos = pos.len() as f64;
    let n_neg = neg.len() as f64;
    // U = R − n_pos(n_pos+1)/2; AUC = U / (n_pos·n_neg).
    let u = rank_sum - n_pos * (n_pos + 1.0) / 2.0;
    (u / (n_pos * n_neg)) as f32
}

/// The census readout over the SHIPPED constructor's probe: build the probe
/// from `samples`, score every sample through it, read the AUC, and refuse a
/// direction whose AUC sits at or under the chance band (`0.5 + band`; the
/// kit's calibrated floor is `band = 0.05`, the fixture's planted margin is
/// far above it). Returns `(auc, probe)` — the caller reuses the probe only
/// after the refusal is cleared (a refused readout never yields a probe worth
/// promoting).
///
/// # Errors
/// [`ReadoutError::Construction`] mirrors the constructor's typed refusals;
/// [`ReadoutError::SingleClass`] for an AUC with no contrast;
/// [`ReadoutError::NoSignal`] for the chance-band AUC — loud, never silent.
pub fn behavior_readout(
    samples: &[LabeledActivation<'_>],
    layer: usize,
    behavior: impl Into<Box<str>>,
    band: f32,
) -> Result<(f32, FutureBehaviorProbe), ReadoutError> {
    let probe = construct_probe_via_mean_difference(samples, layer, behavior)
        .map_err(ReadoutError::Construction)?;
    let mut pos: Vec<f32> = Vec::new();
    let mut neg: Vec<f32> = Vec::new();
    for s in samples {
        let score = probe.forecast(s.activation).probability;
        if s.label {
            pos.push(score);
        } else {
            neg.push(score);
        }
    }
    if pos.is_empty() || neg.is_empty() {
        return Err(ReadoutError::SingleClass);
    }
    let auc = auc_mann_whitney(&pos, &neg);
    if auc <= 0.5 + band {
        return Err(ReadoutError::NoSignal { auc, band });
    }
    Ok((auc, probe))
}

/// Errors from modelless probe construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeanDifferenceError {
    /// No samples provided.
    NoSamples,
    /// All activations have zero dimension.
    ZeroDimension,
    /// An activation's dimension doesn't match the first sample's.
    MismatchedDimension {
        /// Dimension of `samples[0]`.
        first: usize,
        /// Index of the offending sample.
        at: usize,
        /// Dimension of the offending sample.
        got: usize,
    },
    /// All samples share one label — no contrast, direction undefined.
    SingleClass {
        /// Count of positive (label=true) samples.
        n_pos: usize,
        /// Count of negative (label=false) samples.
        n_neg: usize,
    },
    /// Direction norm is zero — the class centroids are identical, so the
    /// behavior is not linearly separable in this activation space.
    ZeroNormDirection,
}

impl std::fmt::Display for MeanDifferenceError {
    #[cold]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSamples => write!(f, "modelless probe: no samples provided"),
            Self::ZeroDimension => write!(f, "modelless probe: activations have zero dimension"),
            Self::MismatchedDimension { first, at, got } => write!(
                f,
                "modelless probe: sample {at} has dimension {got}, expected {first}"
            ),
            Self::SingleClass { n_pos, n_neg } => write!(
                f,
                "modelless probe: single-class data (pos={n_pos}, neg={n_neg}) — no contrast"
            ),
            Self::ZeroNormDirection => write!(
                f,
                "modelless probe: direction norm is zero — class centroids identical"
            ),
        }
    }
}

impl std::error::Error for MeanDifferenceError {}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use crate::future_probe::BehaviorForecast;

    /// Linearly-separable 1D signal: activation[0] = +2 ± noise for label=true,
    /// −2 ± noise for label=false. Mean-difference should recover direction[0] as
    /// the dominant axis and the probe should classify cleanly.
    #[test]
    fn mean_difference_recovers_linearly_separable_direction() {
        // 10 positive, 10 negative, signal only in dim 0.
        let acts_pos: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![2.0 + 0.1 * (i as f32 - 5.0), 0.0, 0.0, 0.0])
            .collect();
        let acts_neg: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![-2.0 + 0.1 * (i as f32 - 5.0), 0.0, 0.0, 0.0])
            .collect();
        let mut samples = Vec::with_capacity(acts_pos.len());
        for a in &acts_pos {
            samples.push(LabeledActivation {
                activation: a,
                label: true,
            });
        }
        for a in &acts_neg {
            samples.push(LabeledActivation {
                activation: a,
                label: false,
            });
        }

        let probe = construct_probe_via_mean_difference(&samples, 7, "test_behavior")
            .expect("linearly-separable data must construct cleanly");

        // Direction should be ~[±1, 0, 0, 0] (sign depends on which class is "pos").
        // We defined pos = label=true = +2, so direction[0] > 0 after normalization.
        assert!(
            probe.forecast(&[2.0, 0.0, 0.0, 0.0]).probability > 0.99,
            "positive-class activation should forecast > 0.99"
        );
        assert!(
            probe.forecast(&[-2.0, 0.0, 0.0, 0.0]).probability < 0.01,
            "negative-class activation should forecast < 0.01"
        );
        // probe.layer() round-trips.
        assert_eq!(probe.layer(), 7);
    }

    #[test]
    fn rejects_empty_samples() {
        let err = construct_probe_via_mean_difference(&[], 0, "x").unwrap_err();
        assert_eq!(err, MeanDifferenceError::NoSamples);
    }

    #[test]
    fn rejects_single_class() {
        let a = vec![1.0, 0.0];
        let samples = [LabeledActivation {
            activation: &a,
            label: true,
        }];
        let err = construct_probe_via_mean_difference(&samples, 0, "x").unwrap_err();
        assert_eq!(err, MeanDifferenceError::SingleClass { n_pos: 1, n_neg: 0 });
    }

    #[test]
    fn rejects_mismatched_dimension() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let samples = [
            LabeledActivation {
                activation: &a,
                label: true,
            },
            LabeledActivation {
                activation: &b,
                label: false,
            },
        ];
        let err = construct_probe_via_mean_difference(&samples, 0, "x").unwrap_err();
        assert_eq!(
            err,
            MeanDifferenceError::MismatchedDimension {
                first: 2,
                at: 1,
                got: 3
            }
        );
    }

    #[test]
    fn rejects_identical_centroids() {
        // Same activation for both classes → centroids identical → zero norm.
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let samples = [
            LabeledActivation {
                activation: &a,
                label: true,
            },
            LabeledActivation {
                activation: &b,
                label: false,
            },
        ];
        let err = construct_probe_via_mean_difference(&samples, 0, "x").unwrap_err();
        assert_eq!(err, MeanDifferenceError::ZeroNormDirection);
    }

    /// Determinism: same samples in same order → bit-identical probe (same hash).
    #[test]
    fn deterministic_given_same_input_order() {
        let acts: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                if i < 4 {
                    vec![2.0, 0.0, 0.0]
                } else {
                    vec![-2.0, 0.0, 0.0]
                }
            })
            .collect();
        let samples: Vec<_> = acts
            .iter()
            .map(|a| LabeledActivation {
                activation: a,
                label: a[0] > 0.0,
            })
            .collect();

        let p1 = construct_probe_via_mean_difference(&samples, 3, "det").unwrap();
        let p2 = construct_probe_via_mean_difference(&samples, 3, "det").unwrap();
        assert_eq!(
            p1.artifact_hash(),
            p2.artifact_hash(),
            "same input order must produce bit-identical probe"
        );
    }

    /// Order-sensitivity: swapping sample order changes the floating-point sum
    /// rounding, so the hash MAY differ. This is documented behavior — pass
    /// samples in a stable order for a stable hash. The test just verifies
    /// both orders produce a valid probe that classifies correctly.
    #[test]
    fn both_orders_classify_correctly() {
        let mut acts: Vec<Vec<f32>> = (0..8)
            .map(|i| {
                if i % 2 == 0 {
                    vec![2.0, 0.0, 0.0]
                } else {
                    vec![-2.0, 0.0, 0.0]
                }
            })
            .collect();
        let mk = |order: &Vec<Vec<f32>>| -> FutureBehaviorProbe {
            let samples: Vec<_> = order
                .iter()
                .map(|a| LabeledActivation {
                    activation: a,
                    label: a[0] > 0.0,
                })
                .collect();
            construct_probe_via_mean_difference(&samples, 1, "order").unwrap()
        };
        let p1 = mk(&acts);
        acts.reverse();
        let p2 = mk(&acts);
        // Both classify correctly regardless of order.
        let f: BehaviorForecast = p1.forecast(&[2.0, 0.0, 0.0]);
        assert!(f.probability > 0.99);
        let f: BehaviorForecast = p2.forecast(&[2.0, 0.0, 0.0]);
        assert!(f.probability > 0.99);
    }

    /// Noisy signal: mean-difference recovers the dominant axis even with noise
    /// in non-signal dimensions. Verifies robustness.
    #[test]
    fn recovers_direction_amid_noise_in_other_dims() {
        let d = 8;
        // 20 samples, signal in dim 0, noise in dims 1..8.
        let acts: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let mut act = vec![0.0_f32; d];
                let label = i % 2 == 0;
                act[0] = if label { 1.5 } else { -1.5 };
                // Noise: bounded, zero-mean across the class.
                for (j, slot) in act.iter_mut().enumerate().skip(1) {
                    *slot = ((i as f32 * 0.3 + j as f32) - 3.0) * 0.05;
                }
                act
            })
            .collect();
        let samples: Vec<_> = acts
            .iter()
            .map(|a| LabeledActivation {
                activation: a,
                label: a[0] > 0.0,
            })
            .collect();
        let probe = construct_probe_via_mean_difference(&samples, 2, "noisy").expect("noise case");
        // Direction[0] should dominate (signal is ~30× the noise magnitude).
        // Positive-class activation → high forecast.
        let mut act_pos = vec![0.0_f32; d];
        act_pos[0] = 1.5;
        assert!(probe.forecast(&act_pos).probability > 0.9);
        let mut act_neg = vec![0.0_f32; d];
        act_neg[0] = -1.5;
        assert!(probe.forecast(&act_neg).probability < 0.1);
    }

    /// I4 fixture (Issue 1228): a planted direction → the readout's AUC meets
    /// the analytic linear-Bayes bound on the fixture. Equal-variance noise
    /// σ around class means ±c·u: the projection along the (recovered)
    /// direction reads d′ = 2c‖u‖/σ and the Bayes AUC = Φ(d′/2) — the
    /// rank-based AUC on a finite sample must land at/above it minus the
    /// sampling band (the bound is the CEILING; the empirical AUC of the
    /// Bayes direction converges down to it, so the assertion pins ≥ bound − ε).
    #[test]
    fn planted_direction_auc_meets_the_bayes_bound() {
        // Deterministic LCG — no RNG dep (the deltanet fixture recipe).
        let mut x = 0x1228_0098_6u64;
        let mut next = move || {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (x >> 40) as f32 / 16_777_216.0
        };
        const D: usize = 8;
        const N_PER: usize = 256;
        let c = 1.0f32;
        let sigma = 0.5f32;
        // Planted direction u (unit-ish, non-axis so the mean-difference
        // genuinely has to find it).
        let u: [f32; D] = [0.3, -0.5, 0.2, 0.6, -0.1, 0.4, -0.3, 0.2];
        let norm_sq: f32 = u.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt();
        let mut pos_acts: Vec<[f32; D]> = Vec::with_capacity(N_PER);
        let mut neg_acts: Vec<[f32; D]> = Vec::with_capacity(N_PER);
        for _ in 0..N_PER {
            let mut pos_act = [0.0f32; D];
            let mut neg_act = [0.0f32; D];
            for dim in 0..D {
                // Equal spherical covariance: per-dim uniform noise scaled to
                // ~unit variance (3.464/2 ≈ the uniform half-width of a
                // var-1 sum-of-uniforms), aligned ±c·u on the planted axis.
                pos_act[dim] = c * u[dim] + sigma * (next() - 0.5) * 3.464;
                neg_act[dim] = -c * u[dim] + sigma * (next() - 0.5) * 3.464;
            }
            pos_acts.push(pos_act);
            neg_acts.push(neg_act);
        }
        let mut samples = Vec::with_capacity(2 * N_PER);
        for i in 0..N_PER {
            samples.push(LabeledActivation {
                activation: &pos_acts[i],
                label: true,
            });
            samples.push(LabeledActivation {
                activation: &neg_acts[i],
                label: false,
            });
        }
        let (auc, _probe) = behavior_readout(&samples, 3, "planted", 0.05).expect("readout");
        // The analytic bound at the POPULATION level: d′ = 2c·‖u‖/σ_effective,
        // where the spherical noise of half-width 3.464·σ/2 ≈ σ_eff·(uniform
        // sum) — the fixture's per-dim noise std is (3.464/2)·σ/√3 ≈ σ. With
        // ‖u‖ over D dims, the projected separation d′ = 2c·‖u‖/σ and the
        // Bayes AUC = Φ(d′/2). Φ via the erf-free logistic pin: the fixture
        // asserts the EMPIRICAL AUC against the bound MINUS the finite-sample
        // band (128-vs-128 ranks, ±0.05 generous).
        let d_prime = 2.0 * c * norm / sigma;
        // Φ(d′/2) via tanh approximation of the error function — closed form,
        // no libm dep: Φ(z) ≈ 0.5·(1 + tanh(0.797885·(z + 0.044715·z³)))
        let z = d_prime * 0.5;
        let phi = 0.5 * (1.0 + (0.797_885 * (z + 0.044_715 * z * z * z)).tanh());
        assert!(
            auc >= phi - 0.05,
            "AUC {auc:.4} must meet the Bayes bound {phi:.4} minus the finite-sample band"
        );
    }

    /// I4 falsifier: identical class centroids (the constructor's own
    /// ZeroNormDirection) and a near-tie direction both refuse TYPED — never
    /// a silent AUC ≈ 0.5 pass.
    #[test]
    fn no_signal_refuses_typed() {
        // Identical centroids → the constructor's typed refusal, mirrored:
        // both classes centered on the same point.
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let samples = vec![
            LabeledActivation {
                activation: &a,
                label: true,
            },
            LabeledActivation {
                activation: &b,
                label: true,
            },
            LabeledActivation {
                activation: &a,
                label: false,
            },
            LabeledActivation {
                activation: &b,
                label: false,
            },
        ];
        match behavior_readout(&samples, 0, "identical", 0.05) {
            Err(ReadoutError::Construction(MeanDifferenceError::ZeroNormDirection)) => {}
            other => panic!("expected the mirrored ZeroNormDirection refusal, got {other:?}"),
        }
        // A direction that separates nothing ON THE SAMPLES (labels
        // scrambled against pure noise) → NoSignal at the chance band. Many
        // samples: the in-sample AUC of a noise direction concentrates at
        // ~0.5 (± the 1/√n rank noise), so the arm lands IN the band
        // reliably — a small sample would flatter the direction (the
        // in-sample overfit the band exists to catch).
        let mut x = 0x516_4A1u64.wrapping_add(1);
        let mut next = move || {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (x >> 40) as f32 / 16_777_216.0
        };
        let raw: Vec<[f32; 2]> = (0..512).map(|_| [next() - 0.5, next() - 0.5]).collect();
        let samples: Vec<LabeledActivation<'_>> = raw
            .iter()
            .enumerate()
            .map(|(i, act)| LabeledActivation {
                activation: act,
                label: i % 2 == 0,
            })
            .collect();
        match behavior_readout(&samples, 0, "scrambled", 0.05) {
            Err(ReadoutError::NoSignal { auc, band }) => {
                assert!(auc <= 0.5 + band, "the scrambled arm must land in the band");
            }
            other => panic!("expected NoSignal, got {other:?}"),
        }
    }
}
