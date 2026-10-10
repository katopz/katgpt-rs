#![cfg(all(feature = "parallax_attn", feature = "prior_logit_lane"))]
//! Issue 819 T2 (Research 566) — per-key additive prior-logit lane gates.
//!
//! G1  lane-off bit-identity: `prior_logits = None` vs an all-zero table are
//!     bit-identical through both activations (the ±0.0 score perturbation
//!     dies at `exp(±0) = 1`), on BOTH score sites (the plain core path and
//!     the parallax-correction inline loop), and `None` matches the
//!     uniform-weight reference for the pre-lane math.
//! G1b known-logit closed-form recovery: orthogonal q/k ⇒ zero content
//!     scores; the lane alone drives the weights — sigmoid realizes
//!     `sigmoid(ℓ_j)/Σsigmoid(ℓ_k)`, softmax `e^{ℓ_j}/Σe^{ℓ_k}` (the KL-prior forms).
//! G1c monotonicity: raising `ℓ_j` never lowers key j's weight.
//! G2  planted distractors: with identical content scores, ANY constant bias
//!     cancels exactly under normalization (w = 1/L), while the per-key lane
//!     breaks the symmetry — the low-signal selectivity a constant prior
//!     provably cannot express.
//! G5  lane + SSMax compose: finite weights summing to 1 under both stages.
//!
//! Fixture: L identical query rows (all e_0) over L keys (all e_1) with
//! V = I — raw dots are exactly 0 in f32 and `out[i][j]` IS the attention
//! weight `p(i, j)`.
//!
//! Reference: katgpt-rs Research 566 §2 (sigmoid KL-prior derivation),
//! arXiv:2601.15380 (GOAT). Feature-gated per the no-default-consumer rule.

use std::sync::Arc;

use katgpt_core::parallax_attn::{
    ParallaxActivation, ParallaxConfig, tiled_attention_parallax_forward,
};

/// Owned lane table (the config carries `Arc<[f32]>` — one cold-path
/// allocation per table build, no lifetime on the config type).
fn lane_from(values: Vec<f32>) -> Option<Arc<[f32]>> {
    Some(Arc::from(values.into_boxed_slice()))
}

/// L identical query rows (e_0) over L keys (e_1) with V = I, dim = L.
fn fixture(l: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = l;
    let mut q = vec![0.0f32; l * d];
    for i in 0..l {
        q[i * d] = 1.0; // every query row = e_0
    }
    let mut k = vec![0.0f32; l * d];
    for j in 0..l {
        k[j * d + 1] = 1.0; // every key = e_1 — orthogonal to e_0 in exact f32
    }
    let mut v = vec![0.0f32; l * d];
    for j in 0..l {
        v[j * d + j] = 1.0; // identity → out[i][j] = p(i, j)
    }
    (q, k, v)
}

fn cfg_with(activation: ParallaxActivation, prior: Option<Arc<[f32]>>) -> ParallaxConfig {
    ParallaxConfig {
        gate_scale: 0.0, // pure attention — isolates the lane
        activation,
        prior_logits: prior,
        ..Default::default()
    }
}

/// Forward through the plain core path (gate_scale = 0 → ρ is zero → the
/// `tiled_attention_core` score site). Returns query row 0 = weight vector.
fn forward_core(l: usize, cfg: &ParallaxConfig) -> Vec<f32> {
    let (q, k, v) = fixture(l);
    let mut out = vec![0.0f32; l * l];
    let r = vec![0.0f32; l * l];
    let x = vec![0.0f32; l];
    tiled_attention_parallax_forward(&q, &k, &v, &mut out, l, l, 1.0, &r, &x, cfg, None);
    out[..l].to_vec()
}

/// Forward through the parallax-correction path (gate_scale ≠ 0 + non-zero
/// W_R → the inline score loop in `tiled_attention_parallax_forward_retaining`).
fn forward_plx(l: usize, cfg: &ParallaxConfig) -> Vec<f32> {
    let (q, k, v) = fixture(l);
    let mut out = vec![0.0f32; l * l];
    let r: Vec<f32> = (0..l * l).map(|i| ((i as f32) * 0.05).cos()).collect();
    let x: Vec<f32> = (0..l).map(|i| (i as f32) * 0.1).collect();
    tiled_attention_parallax_forward(&q, &k, &v, &mut out, l, l, 1.0, &r, &x, cfg, None);
    out[..l].to_vec()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── G1: lane-off / zeros-lane bit-identity, both score sites ──────────────

#[test]
fn zeros_lane_is_bit_identical_to_none_both_activations() {
    let l = 16;
    let zeros = lane_from(vec![0.0; l]);
    for act in [ParallaxActivation::Sigmoid, ParallaxActivation::Softmax] {
        let none = forward_core(l, &cfg_with(act, None));
        let zero = forward_core(l, &cfg_with(act, zeros.clone()));
        assert_eq!(
            none, zero,
            "all-zero lane must be output-identical to None (core path, {act:?})"
        );

        let none_plx = forward_plx(l, &cfg_with(act, None));
        let zero_plx = forward_plx(l, &cfg_with(act, zeros.clone()));
        assert_eq!(
            none_plx, zero_plx,
            "all-zero lane must be output-identical to None (correction path, {act:?})"
        );
        // NOTE: no equality is asserted ACROSS the two paths — with a non-zero
        // W_R they diverge by the parallax correction (different outputs, same
        // lane contract).
    }
}

/// Lane-off sigmoid output must equal the uniform-weight reference: the
/// fixture has exactly-zero content scores, so sigmoid(±0) = 0.5 for every key.
#[test]
fn none_lane_matches_pre_lane_reference_sigmoid() {
    let l = 12;
    let w = forward_core(l, &cfg_with(ParallaxActivation::Sigmoid, None));

    let expect = 1.0f32 / l as f32;
    for &wij in w.iter() {
        assert_eq!(
            wij, expect,
            "lane-off sigmoid weight must equal the uniform reference"
        );
    }
}

// ── G1b: known-logit closed-form recovery ────────────────────────────────

#[test]
fn lane_realizes_kl_prior_closed_forms() {
    let l = 8;
    // Distinct prior logits; content scores exactly 0 → z_j = ℓ_j.
    let lane_table = [-3.0f32, 0.0, 1.5, -0.5, 4.0, -2.0, 0.75, 2.25];
    let lane = lane_from(lane_table.to_vec());

    // Sigmoid: p_j = sigmoid(ℓ_j)/Σsigmoid(ℓ_k) — the Research 566 §2 form.
    let w_sig = forward_core(l, &cfg_with(ParallaxActivation::Sigmoid, lane.clone()));
    let denom: f32 = lane_table.iter().map(|&x| sigmoid(x)).sum();
    for j in 0..l {
        let p = sigmoid(lane_table[j]) / denom;
        assert!(
            (w_sig[j] - p).abs() < 1e-6,
            "sigmoid lane weight {j}: got {}, want {p} (KL-prior form σ(ℓ_j)/Σσ(ℓ_k))",
            w_sig[j]
        );
    }

    // Softmax: p_j = e^{ℓ_j}/Σe^{ℓ_k} — the paper's `softmax(s/τ + log π)`.
    let w_sm = forward_core(l, &cfg_with(ParallaxActivation::Softmax, lane.clone()));
    let max_l = lane_table.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let denom: f32 = lane_table.iter().map(|&x| (x - max_l).exp()).sum();
    for j in 0..l {
        let p = (lane_table[j] - max_l).exp() / denom;
        assert!(
            (w_sm[j] - p).abs() < 1e-6,
            "softmax lane weight {j}: got {}, want {p} (softmax(s + log π))",
            w_sm[j]
        );
    }
}

// ── G1c: monotonicity in ℓ_j ─────────────────────────────────────────────

#[test]
fn lane_is_monotone_per_key() {
    let l = 6;
    let mut prev = 0.0f32;
    for &gold in &[-2.0f32, 0.0, 1.0, 2.0, 4.0] {
        let mut lane = vec![0.0f32; l];
        lane[0] = gold;
        let w = forward_core(l, &cfg_with(ParallaxActivation::Sigmoid, lane_from(lane)));
        assert!(
            w[0] > prev,
            "gold weight must be strictly monotone in ℓ_0: ℓ={gold} w={} prev={prev}",
            w[0]
        );
        prev = w[0];
    }
}

// ── G2: planted distractors — the constant-bias impossibility ────────────

#[test]
fn per_key_lane_beats_constant_bias_at_low_signal_selectivity() {
    let l = 8;

    // ANY constant bias b gives exactly uniform weights when content scores
    // are equal: sigmoid(s+b)/Σsigmoid(s+b) = 1/l. The constant prior cannot select.
    for b in [-4.0f32, -1.0, 0.0, 2.0] {
        let w = forward_core(
            l,
            &cfg_with(ParallaxActivation::Sigmoid, lane_from(vec![b; l])),
        );
        for (j, &wij) in w.iter().enumerate() {
            assert!(
                (wij - 1.0 / l as f32).abs() < 1e-6,
                "constant bias b={b} must cancel to uniform weight: w[{j}]={wij}"
            );
        }
    }

    // The per-key lane breaks the symmetry: gold at +4 over 0-context.
    let mut lane = vec![0.0f32; l];
    lane[0] = 4.0;
    let w = forward_core(l, &cfg_with(ParallaxActivation::Sigmoid, lane_from(lane)));
    let expect = sigmoid(4.0) / (sigmoid(4.0) + (l as f32 - 1.0) * sigmoid(0.0));
    assert!(
        (w[0] - expect).abs() < 1e-6,
        "lane weight must match the KL-prior closed form: {} vs {expect}",
        w[0]
    );
    assert!(
        w[0] > 1.5 / l as f32,
        "lane must break the constant-bias symmetry: w0={} vs uniform {}",
        w[0],
        1.0 / l as f32
    );
    // Conservation: weights sum to 1 (Nadaraya-Watson requirement).
    let total: f32 = w.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "weights must sum to 1: {total}");
}

// ── G5: lane + SSMax composition (all-features build) ────────────────────

#[cfg(feature = "ssmax_temperature")]
#[test]
fn lane_composes_with_ssmax() {
    use katgpt_core::ssmax::SsmaxMode;

    let l = 8;
    let mut lane = vec![0.0f32; l];
    lane[0] = 4.0;

    // Pipeline order: scores → SSMax rescale → lane add → normalize. The
    // content scores are all exactly ±0.0, so the SSMax multiplier rescales
    // zeros and the weights stay the lane-only KL-prior form.
    let cfg = ParallaxConfig {
        gate_scale: 0.0,
        activation: ParallaxActivation::Sigmoid,
        ssmax: Some(SsmaxMode::Fixed { s_l: 1.0 }),
        prior_logits: lane_from(lane),
        #[cfg(feature = "scale_invariant_attn")]
        scale_invariant: None,
    };
    let w = forward_core(l, &cfg);
    let expect = sigmoid(4.0) / (sigmoid(4.0) + (l as f32 - 1.0) * sigmoid(0.0));
    assert!(
        (w[0] - expect).abs() < 1e-6,
        "lane+ssmax: zero-content keys keep the prior-only form: {} vs {expect}",
        w[0]
    );
    let total: f32 = w.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "weights must sum to 1: {total}");
}
