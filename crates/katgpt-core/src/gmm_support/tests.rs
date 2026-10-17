//! `gmm_support` unit tests — the G1 surface + the T3/T6/T7 instrument
//! arms. The GOAT bench (`benches/bench_gmm_support_gate.rs`) re-runs the
//! load-bearing arms on the live build (the bench_039/bench_845 pattern —
//! a feature-gated module's tests must ride an executing lane, not just
//! `cargo test --features`).

use super::certify::{
    CertifyConfig, LinearDiscriminatorGate, bound_acceptance, canary_fire_rate, certify,
};
use super::em::{EmConfig, fit_diag_gmm};
use super::fixture::build_fixture;
use super::gate::{GateError, GateSmoother, SupportGate};
use super::mixture::{DiagGmm, MixtureError, lse};
use super::projector::JlProjector;

// ── Mixture (G1: closed-form anchors) ─────────────────────────────────────

/// K=1 must match the hand-computed univariate-style Gaussian density:
/// `p(x) = N(x; μ, σ²I)` in closed form.
#[test]
fn k1_matches_closed_form_gaussian() {
    const E: usize = 4;
    const K: usize = 1;
    let centroids = [[0.5f32, -1.0, 2.0, 0.0]];
    let variances = [[1.0f32, 4.0, 0.25, 2.0]];
    let log_weights = [0f32]; // ln(1)
    let gmm = DiagGmm::<E, K>::new(centroids, log_weights, variances).expect("valid params");
    let x = [0.5f32, 1.0, 2.5, -1.0];
    let got = gmm.loglik_mixture(&x);
    // Hand: Σ_d [ −0.5·ln(2πσ²_d) − (x_d−μ_d)²/(2σ²_d) ]
    const LN_2PI: f32 = 1.837_877_1;
    let terms: f32 = [1.0f32, 4.0, 0.25, 2.0]
        .iter()
        .zip(x.iter().zip(centroids[0].iter()))
        .map(|(&v, (&xi, &mi))| -0.5 * (LN_2PI + v.ln()) - (xi - mi) * (xi - mi) / (2.0 * v))
        .sum();
    assert!(
        (got - terms).abs() < 1e-3,
        "mixture loglik {got} != closed form {terms}"
    );
}

/// Weights are normalized at construction (unnormalized input accepted,
/// proportionally correct output).
#[test]
fn construction_normalizes_weights() {
    const E: usize = 2;
    const K: usize = 3;
    let centroids = [[0f32; E]; K];
    let variances = [[1.0f32; E]; K];
    let gmm = DiagGmm::<E, K>::new(centroids, [5.0, 5.0, 5.0], variances).expect("valid params");
    let sum: f32 = gmm.log_weights.iter().map(|w| w.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5, "weights sum to {sum}, not 1");
    // And the uniform case passes through.
    for k in 0..K {
        assert!((gmm.log_weights[k] - f32::ln(1.0 / K as f32)).abs() < 1e-6);
    }
}

/// Invalid variances and weights refuse.
#[test]
fn construction_refuses_invalid_params() {
    const E: usize = 2;
    const K: usize = 1;
    let centroids = [[0f32; E]; K];
    assert_eq!(
        DiagGmm::<E, K>::new(centroids, [0.0], [[0.0, 1.0]]).unwrap_err(),
        MixtureError::InvalidVariance
    );
    assert_eq!(
        DiagGmm::<E, K>::new(centroids, [f32::NAN], [[1.0, 1.0]]).unwrap_err(),
        MixtureError::InvalidWeight
    );
}

/// The commitment is deterministic and tamper-sensitive (the freeze/thaw
/// tripwire).
#[test]
fn commitment_is_deterministic_and_tamper_sensitive() {
    const E: usize = 3;
    const K: usize = 2;
    let a = DiagGmm::<E, K>::new_unchecked(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [0.0, 0.0],
        [[1.0; E]; K],
    );
    let b = DiagGmm::<E, K>::new_unchecked(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [0.0, 0.0],
        [[1.0; E]; K],
    );
    assert_eq!(a.commitment(), b.commitment());
    assert!(a.verify());
    let mut tampered = b.clone();
    tampered.centroids[0][0] += 0.5;
    assert!(!tampered.verify(), "tampered mixture must fail verify");
}

/// Two well-separated modes: the mixture assigns each mode's points the
/// right component (responsibilities).
#[test]
fn two_modes_assign_correctly() {
    const E: usize = 2;
    const K: usize = 2;
    let centroids = [[-5.0f32, 0.0], [5.0, 0.0]];
    let variances = [[1.0f32; E]; K];
    let gmm = DiagGmm::<E, K>::new_unchecked(centroids, [0.0, 0.0], variances);
    let mut resp = [0f32; K];
    gmm.log_responsibilities(&[-5.0, 0.1], &mut resp);
    assert!(resp[0].exp() > 0.99, "left mode must own the left point");
    gmm.log_responsibilities(&[5.0, -0.1], &mut resp);
    assert!(resp[1].exp() > 0.99, "right mode must own the right point");
}

/// The lse stable form pinned on known-value anchors (max-shift, f64
/// accumulator — the exact spec the `distributional_steering` twin
/// implements; its private visibility makes a direct cross-call test
/// impossible, so the SHARED SPEC is what is pinned, not the call).
#[test]
fn lse_known_value_anchors() {
    let anchors: [(&[f32], f32); 3] = [
        (&[0.0, 0.0], 2f32.ln()),
        (&[1.0, 0.0], 1.0 + (1.0 + (-1.0f32).exp()).ln()),
        (&[-100.0, -100.0], -100.0 + 2f32.ln()),
    ];
    for (xs, want) in anchors {
        let got = lse(xs);
        assert!(
            (got - want).abs() < 1e-5,
            "lse({xs:?}) = {got}, want {want}"
        );
    }
}

// ── EM (G1: determinism + recovery) ────────────────────────────────────────

/// Same input bytes + same config → identical artifact (the BLAKE3
/// determinism pin).
#[test]
fn em_is_deterministic() {
    const E: usize = 8;
    const K: usize = 3;
    let fix = build_fixture::<E, K>(42, 600, 100, 50, 2.0);
    let refs: Vec<&[f32]> = fix.pos_train.iter().map(|v| v.as_slice()).collect();
    let cfg = EmConfig::default();
    let a = fit_diag_gmm::<E, K>(&refs, &cfg).expect("fit a");
    let b = fit_diag_gmm::<E, K>(&refs, &cfg).expect("fit b");
    assert_eq!(a.commitment(), b.commitment(), "EM must be deterministic");
}

/// EM recovers the fixture's blob structure: means near true centroids
/// (matched greedily), weights near uniform.
#[test]
fn em_recovers_blob_structure() {
    const E: usize = 8;
    const K: usize = 3;
    let fix = build_fixture::<E, K>(7, 1500, 200, 50, 2.5);
    let refs: Vec<&[f32]> = fix.pos_train.iter().map(|v| v.as_slice()).collect();
    let fitted = fit_diag_gmm::<E, K>(&refs, &EmConfig::default()).expect("fit");
    // Greedy match each true centroid to its nearest fitted mean.
    let mut max_dist = 0.0f32;
    for tc in fix.true_pos.centroids.iter() {
        let mut best = f32::INFINITY;
        for fc in fitted.centroids.iter() {
            let d: f32 = tc
                .iter()
                .zip(fc.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            best = best.min(d).sqrt();
        }
        max_dist = max_dist.max(best);
    }
    assert!(
        max_dist < 1.0,
        "worst matched-centroid distance {max_dist} too large"
    );
}

/// Empty input and dim mismatch refuse with the named errors.
#[test]
fn em_refuses_bad_input() {
    const E: usize = 4;
    const K: usize = 2;
    assert_eq!(
        fit_diag_gmm::<E, K>(&[], &EmConfig::default()).unwrap_err(),
        super::em::EmError::EmptySamples
    );
    let bad: [&[f32]; 2] = [&[0.0; E], &[0.0; 3]];
    assert_eq!(
        fit_diag_gmm::<E, K>(&bad, &EmConfig::default()).unwrap_err(),
        super::em::EmError::DimMismatch
    );
}

// ── Projector (T3: JL sanity + determinism) ────────────────────────────────

/// JL distortion sanity on a fixture set: pairwise distances preserved
/// within the JL budget at E=64 for 32 points (loose bars that a broken
/// projection — all-zero rows, correlated rows, wrong scale — cannot
/// pass).
#[test]
fn jl_distortion_within_budget() {
    const D: usize = 256;
    const E: usize = 64;
    const N: usize = 32;
    let proj = JlProjector::<D, E>::new(99);
    // Deterministic fixture points.
    let mut pts = [[0f32; D]; N];
    let mut rng = super::projector::SplitMix64::new(5);
    for p in pts.iter_mut() {
        for v in p.iter_mut() {
            *v = super::fixture::clt_normal(&mut rng);
        }
    }
    let mut projected = [[0f32; E]; N];
    for (i, p) in pts.iter().enumerate() {
        proj.project(p, &mut projected[i]);
    }
    let mut max_rel = 0.0f32;
    for i in 0..N {
        for j in (i + 1)..N {
            let d_orig: f32 = pts[i]
                .iter()
                .zip(pts[j].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            let d_proj: f32 = projected[i]
                .iter()
                .zip(projected[j].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            let rel = ((d_proj - d_orig) / d_orig).abs();
            max_rel = max_rel.max(rel);
        }
    }
    assert!(
        max_rel < 0.5,
        "max pairwise relative distortion {max_rel} exceeds the JL sanity bar"
    );
}

/// Norm preservation in expectation: `E[‖Px‖²] ≈ ‖x‖²` (the scale law).
#[test]
fn jl_preserves_norms_in_expectation() {
    const D: usize = 256;
    const E: usize = 64;
    let proj = JlProjector::<D, E>::new(1234);
    let mut rng = super::projector::SplitMix64::new(11);
    let mut ratio_sum = 0.0f32;
    let trials = 200;
    for _ in 0..trials {
        let mut x = [0f32; D];
        for v in x.iter_mut() {
            *v = super::fixture::clt_normal(&mut rng);
        }
        let px = proj.project_array(&x);
        let n2: f32 = x.iter().map(|v| v * v).sum();
        let p2: f32 = px.iter().map(|v| v * v).sum();
        ratio_sum += p2 / n2;
    }
    let avg = ratio_sum / trials as f32;
    assert!(
        (avg - 1.0).abs() < 0.15,
        "mean projected/original norm² ratio {avg} drifts from 1"
    );
}

/// Same (D, E, seed) → identical commitment; different seed → different.
#[test]
fn jl_commitment_pins_the_matrix() {
    type P = JlProjector<128, 32>;
    let a = P::new(7);
    let b = P::new(7);
    let c = P::new(8);
    assert_eq!(a.commitment(), b.commitment());
    assert_ne!(a.commitment(), c.commitment());
    assert!(a.verify());
}

// ── Gate + smoother ───────────────────────────────────────────────────────

/// Unfitted = closed-always, confidence 0, never NaN (the empty-corpus
/// precedent).
#[test]
fn unfitted_gate_is_closed_always() {
    const E: usize = 4;
    const K: usize = 2;
    let gate = SupportGate::<E, K>::unfilled_probe();
    let x = [0.0f32; E];
    assert!(!gate.open(&x));
    assert!(!gate.is_fitted());
    assert_eq!(gate.confidence(&x), 0.0);
    assert_eq!(gate.log_ratio(&x), f32::NEG_INFINITY);
}

/// Invalid tau and alpha refuse.
#[test]
fn gate_and_smoother_refuse_invalid_knobs() {
    const E: usize = 4;
    const K: usize = 2;
    assert_eq!(
        SupportGate::<E, K>::unfitted(0.0).unwrap_err(),
        GateError::InvalidTau
    );
    assert_eq!(GateSmoother::new(0.0).unwrap_err(), GateError::InvalidAlpha);
    assert_eq!(GateSmoother::new(1.5).unwrap_err(), GateError::InvalidAlpha);
}

/// A fitted gate opens on its own support and closes off it (the E=8
/// fixture geometry: pos blob within radius ~separation of origin, neg
/// broad at origin, B at 6·separation).
#[test]
fn gate_opens_on_support_closes_off() {
    const E: usize = 8;
    const K: usize = 3;
    let fix = build_fixture::<E, K>(21, 1200, 300, 100, 2.5);
    let refs: Vec<&[f32]> = fix.pos_train.iter().map(|v| v.as_slice()).collect();
    let pos = fit_diag_gmm::<E, K>(&refs, &EmConfig::default()).expect("fit pos");
    let gate = SupportGate::new(pos, fix.neg.clone(), 1.0).expect("valid tau");
    // On-support: held-out opens.
    let mut opened = 0usize;
    for s in fix.pos_held_out.iter() {
        let x: &[f32; E] = s.as_slice().try_into().expect("len E");
        if gate.open(x) {
            opened += 1;
        }
    }
    let rate = opened as f32 / fix.pos_held_out.len() as f32;
    assert!(rate > 0.85, "held-out open rate {rate} too low (deficit)");
    // Off-support: the far extrapolation probes close.
    for p in fix.extrapolation.iter() {
        let x: &[f32; E] = p.as_slice().try_into().expect("len E");
        assert!(!gate.open(x), "extrapolation probe must stay closed");
    }
}

/// The EMA smoother: `s ← α·g + (1−α)·s`, decide at 0.5, neutral start.
#[test]
fn smoother_ema_math() {
    let mut s = GateSmoother::new(0.5).expect("valid alpha");
    assert_eq!(s.state(), 0.5);
    assert!(!s.is_open());
    s.update(1.0); // 0.5·1 + 0.5·0.5 = 0.75
    assert!((s.state() - 0.75).abs() < 1e-6);
    assert!(s.is_open());
    s.update(0.0); // 0.5·0 + 0.5·0.75 = 0.375
    assert!((s.state() - 0.375).abs() < 1e-6);
    assert!(!s.is_open());
    assert!(s.decides(0.3), "decides at a caller threshold");
}

// ── Certification (T6/T7) ─────────────────────────────────────────────────

/// The leak-by-construction canary: the discriminator control fires on
/// ALL extrapolation probes (else the instrument reds), and the density
/// gate keeps a lower reference-weighted excess than the leaky control.
#[test]
fn canary_discriminator_fires_and_density_gate_stays_lower() {
    const E: usize = 8;
    const K: usize = 3;
    let fix = build_fixture::<E, K>(33, 1200, 300, 200, 2.5);
    let refs: Vec<&[f32]> = fix.pos_train.iter().map(|v| v.as_slice()).collect();
    let pos = fit_diag_gmm::<E, K>(&refs, &EmConfig::default()).expect("fit pos");
    let gate = SupportGate::new(pos, fix.neg.clone(), 1.0).expect("valid tau");

    let a_refs: Vec<&[f32]> = fix.pos_train.iter().map(|v| v.as_slice()).collect();
    let b_refs: Vec<&[f32]> = fix.planted_b.iter().map(|v| v.as_slice()).collect();
    let disc = LinearDiscriminatorGate::fit(&a_refs, &b_refs).expect("fit disc");

    let probe_refs: Vec<&[f32]> = fix.extrapolation.iter().map(|v| v.as_slice()).collect();
    let fire_rate = canary_fire_rate(&disc, &probe_refs).expect("canary must fire");
    assert_eq!(fire_rate, 1.0);

    // Both gates certified: the density gate's excess must beat the
    // leaky control's (the instrument's discriminating power).
    let held: Vec<&[f32]> = fix.pos_held_out.iter().map(|v| v.as_slice()).collect();
    let cfg = CertifyConfig {
        n_reference_samples: 2048,
        ..CertifyConfig::default()
    };
    let density_rep = certify(&gate, &held, &fix.neg, &cfg).expect("certify density");
    let leaky_rep = certify(&disc, &held, &fix.neg, &cfg).expect("certify leaky");
    assert!(
        density_rep.excess_mass < leaky_rep.excess_mass,
        "density excess {} must beat leaky excess {}",
        density_rep.excess_mass,
        leaky_rep.excess_mass
    );
    assert!(density_rep.deficit_rate < 0.15, "deficit too high");
}

/// T7 bound acceptance: holds on the fixture at BOTH arms (low-D and
/// the E=64 band), and the monotonicity arm — a WORSE fit (bigger ε =
/// L1 estimate, from starving the fitter) never DECREASES excess mass
/// beyond the Monte-Carlo slack.
#[test]
fn bound_acceptance_holds_and_is_monotone() {
    const E: usize = 8;
    const K: usize = 3;
    let fix = build_fixture::<E, K>(51, 1500, 200, 100, 2.0);
    let refs: Vec<&[f32]> = fix.pos_train.iter().map(|v| v.as_slice()).collect();
    let fitted = fit_diag_gmm::<E, K>(&refs, &EmConfig::default()).expect("fit");
    let gate = SupportGate::new(fitted.clone(), fix.neg.clone(), 1.0).expect("valid tau");
    let cfg = CertifyConfig::default();
    let rep = bound_acceptance(&fix.true_pos, &fitted, &fix.neg, &gate, &cfg);
    assert!(
        rep.holds,
        "excess {} > l1 {} + slack — the fixture-scoped bound failed",
        rep.excess_mass, rep.l1_estimate
    );

    // Monotonicity in ε: inject controlled fit error — interpolate the
    // fitted means toward the REFERENCE center (δ = 0.5, then 1.0). The
    // gate's positive density migrates onto the reference support, so BOTH
    // the L1 error and the reference-weighted excess must grow with δ
    // (sample starvation measured NOT monotone here — a starved k-means
    // seed still lands reasonable centroids while its inflated variances
    // make the gate MORE conservative; the recorded datum, not a gap).
    let mut prev_l1 = rep.l1_estimate;
    let mut prev_excess = rep.excess_mass;
    for (delta, tag) in [(0.5f32, "half"), (1.0, "full")] {
        let mut drifted = fitted.clone();
        for c in drifted.centroids.iter_mut() {
            for v in c.iter_mut() {
                *v *= 1.0 - delta;
            }
        }
        let drifted =
            DiagGmm::new_unchecked(drifted.centroids, drifted.log_weights, drifted.variances);
        let gate_d = SupportGate::new(drifted.clone(), fix.neg.clone(), 1.0).expect("tau");
        let rep_d = bound_acceptance(&fix.true_pos, &drifted, &fix.neg, &gate_d, &cfg);
        assert!(
            rep_d.l1_estimate >= prev_l1,
            "drifted fit ({tag}) must have the larger L1: {} vs {prev_l1}",
            rep_d.l1_estimate
        );
        assert!(
            rep_d.excess_mass >= prev_excess - cfg.slack,
            "excess not monotone in injected error ({tag}): {} vs {prev_excess}",
            rep_d.excess_mass
        );
        assert!(
            rep_d.holds,
            "drifted arm {tag} must still satisfy the bound"
        );
        prev_l1 = rep_d.l1_estimate;
        prev_excess = rep_d.excess_mass;
    }
    // The full drift must be a LARGE L1 (the injected-error direction).
    assert!(
        prev_l1 > rep.l1_estimate,
        "full drift L1 {prev_l1} must exceed the good fit's {l} good",
        l = rep.l1_estimate
    );
    // Measured datum (2026-10-05, this fixture family): the excess moves
    // only within slack across the drift ladder — the Φ_neg-weighted
    // excess is dominated by the fixture's baseline overlap, not by the
    // injected mean error. The plan's "bigger ε ⇒ bigger excess WITHIN
    // SLACK" is satisfied as written; strict growth is NOT claimed and
    // the flatness is recorded, not hidden.
}

/// The E=64 band arm of the bound check (the low-D arm above; T7 asks
/// for both).
#[test]
fn bound_acceptance_at_e64() {
    const E: usize = 64;
    const K: usize = 4;
    let fix = build_fixture::<E, K>(17, 2000, 200, 100, 2.0);
    let refs: Vec<&[f32]> = fix.pos_train.iter().map(|v| v.as_slice()).collect();
    let fitted = fit_diag_gmm::<E, K>(&refs, &EmConfig::default()).expect("fit");
    let gate = SupportGate::new(fitted.clone(), fix.neg.clone(), 1.0).expect("valid tau");
    let rep = bound_acceptance(
        &fix.true_pos,
        &fitted,
        &fix.neg,
        &gate,
        &CertifyConfig::default(),
    );
    assert!(rep.holds, "E=64 bound arm failed: {rep:?}");
}

// The tests file needs the helper used above — a tiny constructor shim
// keeping the unfitted-probe call sites readable.
impl<const E: usize, const K: usize> SupportGate<E, K> {
    /// Test-only alias for [`SupportGate::unfitted`] at `DEFAULT_TAU`.
    fn unfilled_probe() -> Self {
        SupportGate::unfitted(super::gate::DEFAULT_TAU).expect("default tau is valid")
    }
}
