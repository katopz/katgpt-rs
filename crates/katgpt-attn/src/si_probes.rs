//! Scale-invariant attention diagnostics (Plan 622 Phase 5).
//!
//! The paper's two desiderata (arXiv:2505.17083; Research 610), made
//! measurable against the shipped schedule LUT
//! ([`katgpt_core::scale_invariant::ScaleInvariantLut`] — consumed through
//! [`ScaleInvariantLut::pair`], never re-derived here):
//!
//! 1. **Per-decade mass Θ(1)** — the expected unnormalized attention mass
//!    over a multiplicative distance window `(t, tΔ]` stays bounded as `t`
//!    grows. Under Gaussian base logits the per-key expectation is *exact*:
//!    `f(s) = E[exp(a_s·L + m_s)] = exp(m_s + a_s²/2)`, so the decade mass is
//!    a deterministic sum, cross-checked against the harmonic integral
//!    `ατ·ln((tΔ+τ)/(t+τ))`.
//!    ⚠ No Monte-Carlo arm for this probe: the per-key mass variance is
//!    `e²(1−e^{−a_s²}) = Θ(1)` while the decade mean is `Θ(1)`, so a single
//!    window sample is heavy-tailed (relative std
//!    `~ e·√(t(Δ−1))/(ατ·ln Δ)` — hundreds at the top decades). The law is
//!    about the expectation, and the expectation is computable exactly.
//! 2. **Per-decade sparsity sub-log** — the boundary pin makes
//!    `a_s² + m_s = β/α` CONSTANT in `s` (from `m = −a² + β/α`), so
//!    `E[Ã ln Ã] = (β/α)·f(s)` and the normalized distance-entropy has the
//!    exact identity `H(T) = ln Z(T) − 1` with `Z(T) = Σ f(s) ≈
//!    ατ·ln(T/τ+1)` — doubly-logarithmic growth against the unscaled
//!    Gaussian control's exact `ln(T)`.
//!
//! The sigmoid arm's per-distance curve (`σ̃ = σ(a_t·L + m_t)`: mass,
//! negentropy, per-mass entropy) is computed by the shared adaptive-Simpson
//! quadrature ([`katgpt_core::scale_invariant::gaussian_expectation`] — the
//! Phase-2 G1 instrument, promoted for exactly this reuse). Monte-Carlo was
//! tried and REFUTED for this curve: at t = 10⁵ the sigmoid mass is ~7e-5,
//! so even 100k draws see ~7 events — no tail support. Quadrature is exact
//! to 1e-12 and deterministic. Controls use the identity schedule
//! `(a, m) = (1, 0)` — the unscaled Gaussian-logit negative control.
//!
//! # Home note (recorded deviation)
//!
//! Plan 622 Phase 5 names `katgpt-attn/src/chiaroscuro/` as the home; that
//! directory is Plan 269's CHIAR module behind the unrelated `chiaroscuro`
//! feature — gating these probes on CHIAR's flag would couple two unrelated
//! primitives. The probes therefore live in this sibling module behind the
//! `scale_invariant_attn` forwarding feature (the `gdn_tree_verify`
//! precedent). The plan's crate choice (katgpt-attn) is honored.
//!
//! τ sweep (App. H protocol — the modelless half)
//!
//! [`assess_tau_sweep`] assembles the in-dist-ppl verdict for a τ sweep; the
//! ppl measurement loop itself needs the real checkpoint and rides Phase 7's
//! box/GPU gates — this is the pure verdict assembly the Phase-7 bench
//! consumes.
//!
//! Every probe is deterministic: exact expectation sums for the mass and
//! softmax-entropy laws, adaptive-Simpson quadrature for the sigmoid arm —
//! no RNG anywhere.

use katgpt_core::scale_invariant::{gaussian_expectation, ScaleInvariantLut};

// ──────────────────────────────────────────────────────────────────────
// Schedule-derived constants (never hard-coded)
// ──────────────────────────────────────────────────────────────────────────

/// The schedule constants derived from the LUT's own boundary entry:
/// `(a₀, m₀) = (1, 0)` ⟹ `β/α = m₀ + a₀²` and `log α = β/α − a₀²/2`.
/// Returns `(beta_over_alpha, alpha)`.
pub fn schedule_constants(lut: &ScaleInvariantLut) -> (f64, f64) {
    let (a0, m0) = lut.pair(0);
    let (a0, m0) = (a0 as f64, m0 as f64);
    let beta_over_alpha = m0 + a0 * a0;
    let alpha = (beta_over_alpha - a0 * a0 * 0.5).exp();
    (beta_over_alpha, alpha)
}

/// Exact per-key expected mass at causal distance `s` under Gaussian base
/// logits: `f(s) = exp(m_s + a_s²/2)`.
#[inline]
pub fn expected_key_mass(lut: &ScaleInvariantLut, s: usize) -> f64 {
    let (a, m) = lut.pair(s);
    let (a, m) = (a as f64, m as f64);
    (m + a * a * 0.5).exp()
}

/// The unscaled negative control's per-key expected mass: the identity
/// schedule `(a, m) = (1, 0)` gives `E[exp(L)] = e^{1/2}` — constant in `s`.
#[inline]
pub fn control_key_mass(_s: usize) -> f64 {
    (0.5f64).exp()
}

/// Half-open decade window `(t, tΔ]` as an inclusive distance range.
fn decade_range(t: usize, delta: f64) -> (usize, usize) {
    (t + 1, (t as f64 * delta).floor() as usize)
}

// ──────────────────────────────────────────────────────────────────────────
// Probe A — per-decade mass (desideratum 1, exact)
// ──────────────────────────────────────────────────────────────────────────

/// One decade-mass row: the exact expected mass over `(t, tΔ]`, the paper's
/// unnormalized log-mass `Σ (a_s² + m_s)·f(s)` (Lemma 2's quantity — equals
/// `β/α · mass` by the boundary-pin constancy), and the harmonic-integral
/// cross-check `ατ·ln((tΔ+τ)/(t+τ))`.
#[derive(Debug, Clone)]
pub struct DecadeMassRow {
    pub t: usize,
    pub delta: f64,
    /// `Σ f(s)` over the window — exact.
    pub mass: f64,
    /// `Σ (a_s² + m_s)·f(s)` — the paper's per-decade sparsity quantity.
    pub log_mass: f64,
    /// `ατ·ln((tΔ+τ)/(t+τ))` — the integral form (incl. the `+τ` ramp
    /// correction) the exact sum must match to ~2% at `t ≥ 100`.
    pub integral: f64,
}

/// Sweep the decade-mass probe over a `t` ladder × Δ set. All exact — no
/// sampling (see the module doc for why MC would be statistically vacuous).
///
/// Panics when a window's far edge exceeds the LUT — build `max_ctx` to
/// cover `t·Δ` for the largest row.
pub fn decade_mass_report(
    lut: &ScaleInvariantLut,
    t_ladder: &[usize],
    deltas: &[f64],
) -> Vec<DecadeMassRow> {
    let (_, alpha) = schedule_constants(lut);
    let tau = lut.tau() as f64;
    let mut rows = Vec::with_capacity(t_ladder.len() * deltas.len());
    for &t in t_ladder {
        for &delta in deltas {
            let (lo, hi) = decade_range(t, delta);
            assert!(
                hi <= lut.max_distance(),
                "decade window exceeds the LUT: t={t} Δ={delta} needs distance {hi}, \
                 LUT covers {}",
                lut.max_distance()
            );
            let mut mass = 0.0f64;
            let mut log_mass = 0.0f64;
            for s in lo..=hi {
                let f = expected_key_mass(lut, s);
                mass += f;
                let (a, m) = lut.pair(s);
                log_mass += ((a as f64) * (a as f64) + m as f64) * f;
            }
            let integral = alpha * tau * (((t as f64 * delta) + tau) / (t as f64 + tau)).ln();
            rows.push(DecadeMassRow {
                t,
                delta,
                mass,
                log_mass,
                integral,
            });
        }
    }
    rows
}

// ──────────────────────────────────────────────────────────────────────────
// Probe B — normalized distance-entropy totals (desideratum 2, exact)
// ──────────────────────────────────────────────────────────────────────────

/// Total normalized entropy of the distance distribution `p(s) ∝ f(s)` over
/// distances `1..=t_max` (distance 0 excluded — the sink carve-out's
/// distance-side), computed exactly. The identity schedule (control) gives
/// the uniform `ln(t_max)`.
pub fn softmax_total_entropy<F>(key_mass: F, t_max: usize) -> f64
where
    F: Fn(usize) -> f64,
{
    let mut z = 0.0f64;
    let mut f_ln_f = 0.0f64;
    for s in 1..=t_max {
        let f = key_mass(s);
        z += f;
        f_ln_f += f * f.ln();
    }
    z.ln() - f_ln_f / z
}

/// One rung of the total-entropy ladder.
#[derive(Debug, Clone)]
pub struct SoftmaxEntropyRow {
    pub t_max: usize,
    /// si: exact `H(T)`; must equal `ln Z − 1` (the boundary-pin identity).
    pub h_si: f64,
    /// Control (identity schedule): exact; must equal `ln(T)`.
    pub h_control: f64,
    /// `Z(T) = Σ f(s)` for the si arm — the closed form's input.
    pub z_si: f64,
}

/// Sweep the total-entropy probe over a `t_max` ladder.
pub fn softmax_entropy_ladder(
    lut: &ScaleInvariantLut,
    t_ladder: &[usize],
) -> Vec<SoftmaxEntropyRow> {
    let mut rows = Vec::with_capacity(t_ladder.len());
    for &t_max in t_ladder {
        assert!(
            t_max <= lut.max_distance(),
            "t_max={t_max} exceeds the LUT (covers {})",
            lut.max_distance()
        );
        let h_si = softmax_total_entropy(|s| expected_key_mass(lut, s), t_max);
        let h_control = softmax_total_entropy(control_key_mass, t_max);
        let z_si: f64 = (1..=t_max).map(|s| expected_key_mass(lut, s)).sum();
        rows.push(SoftmaxEntropyRow {
            t_max,
            h_si,
            h_control,
            z_si,
        });
    }
    rows
}

// ──────────────────────────────────────────────────────────────────────────
// Probe C — sigmoid per-distance curve (seeded MC; the Phase-7 report)
// ──────────────────────────────────────────────────────────────────────────

/// One per-distance sigmoid row, at causal distance `t`.
#[derive(Debug, Clone)]
pub struct SigmoidCurveRow {
    pub t: usize,
    /// `E[σ̃]` — the raw MC mean.
    pub e_sigma: f64,
    /// `E[σ̃]·(t/τ+1)/α` — the mass shape; the G1 tilt identity's curve
    /// (asserted to its band here, exactly in katgpt-core's G1 tests).
    pub mass_scaled: f64,
    /// `−E[σ̃ ln σ̃]` — the per-key negentropy (paper's unnormalized form).
    pub neg_entropy: f64,
    /// `neg_entropy / E[σ̃]` — the per-mass entropy curve.
    pub per_mass: f64,
}

impl SigmoidCurveRow {
    fn exact(t: usize, tau: f64, alpha: f64, a: f64, m: f64) -> Self {
        let e_x = gaussian_expectation(m, a, |x| 1.0 / (1.0 + (-x).exp()));
        let neg = -gaussian_expectation(m, a, |x| {
            let s = 1.0 / (1.0 + (-x).exp());
            if s > 0.0 {
                s * s.ln()
            } else {
                0.0
            }
        });
        SigmoidCurveRow {
            t,
            e_sigma: e_x,
            mass_scaled: e_x * (t as f64 / tau + 1.0) / alpha,
            neg_entropy: neg,
            per_mass: if e_x > 0.0 { neg / e_x } else { 0.0 },
        }
    }
}

/// The si arm's sigmoid curve over a distance ladder (exact quadrature —
/// no MC: the σ tail at the far rungs has no sample support).
pub fn sigmoid_curve_report(lut: &ScaleInvariantLut, t_ladder: &[usize]) -> Vec<SigmoidCurveRow> {
    let (_, alpha) = schedule_constants(lut);
    let tau = lut.tau() as f64;
    let mut rows = Vec::with_capacity(t_ladder.len());
    for &t in t_ladder {
        assert!(t <= lut.max_distance(), "t={t} exceeds the LUT");
        let (a, m) = lut.pair(t);
        rows.push(SigmoidCurveRow::exact(t, tau, alpha, a as f64, m as f64));
    }
    rows
}

/// The unscaled control's sigmoid curve: the identity schedule `(1, 0)` —
/// every per-distance quantity is constant in `t` (flat curve). The
/// `mass_scaled` column divides by the paper-default τ=10 shape and
/// `α = e^{1/2}` purely for row-shape compatibility;
/// [`SigmoidCurveRow::e_sigma`] is the quantity the control asserts on.
pub fn sigmoid_control_curve(t_ladder: &[usize]) -> Vec<SigmoidCurveRow> {
    let alpha = (0.5f64).exp();
    t_ladder
        .iter()
        .map(|&t| SigmoidCurveRow::exact(t, 10.0, alpha, 1.0, 0.0))
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────
// τ sweep verdict (App. H protocol — the modelless half)
// ──────────────────────────────────────────────────────────────────────────

/// One τ-sweep row from the Phase-7 bench: the τ value and the in-dist ppl
/// delta it cost (percent, positive = worse).
#[derive(Debug, Clone)]
pub struct TauSweepRow {
    pub tau: f32,
    pub in_dist_ppl_delta_pct: f64,
}

/// The App. H verdict: every τ's in-dist ppl delta must stay within the
/// pinned tolerance (the length-generalization win may not be bought with
/// in-dist regression at any swept τ).
#[derive(Debug, Clone, PartialEq)]
pub enum TauSweepVerdict {
    Pass,
    /// The offending τ values with their deltas, ascending by τ.
    Fail(Vec<(f32, f64)>),
}

/// Assemble the verdict. Refuses (panics) on an empty sweep — a verdict over
/// zero rows would be a green zero.
pub fn assess_tau_sweep(rows: &[TauSweepRow], tolerance_pct: f64) -> TauSweepVerdict {
    assert!(
        !rows.is_empty(),
        "tau sweep: no rows — run the sweep before assessing it"
    );
    let mut offenders: Vec<(f32, f64)> = rows
        .iter()
        .filter(|r| r.in_dist_ppl_delta_pct > tolerance_pct)
        .map(|r| (r.tau, r.in_dist_ppl_delta_pct))
        .collect();
    offenders.sort_by(|a, b| a.0.total_cmp(&b.0));
    if offenders.is_empty() {
        TauSweepVerdict::Pass
    } else {
        TauSweepVerdict::Fail(offenders)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests — the laws asserted at run time, constants derived, never pinned
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TAU: f32 = 10.0;
    /// The entropy + sigmoid ladder; the LUT covers the largest rung.
    const T_LADDER: [usize; 4] = [100, 1_000, 10_000, 100_000];

    fn lut() -> ScaleInvariantLut {
        ScaleInvariantLut::build(TAU, 100_001)
    }

    #[test]
    fn decade_mass_matches_the_harmonic_integral_and_is_theta_one() {
        // Δ=10 at t=10⁵ needs distance 10⁶ — this probe builds its own LUT.
        let big = ScaleInvariantLut::build(TAU, 1_000_001);
        let ts: [usize; 4] = [100, 1_000, 10_000, 100_000];
        let rows = decade_mass_report(&big, &ts, &[2.0, 10.0]);
        assert_eq!(rows.len(), 8);
        let (beta_over_alpha, _) = schedule_constants(&big);

        for r in &rows {
            // Exact sum vs the integral form (incl. the +τ correction):
            // within 2% at every t ≥ 100.
            let rel = (r.mass / r.integral - 1.0).abs();
            assert!(
                rel <= 0.02,
                "t={} Δ={}: mass {} vs integral {} (rel {rel})",
                r.t,
                r.delta,
                r.mass,
                r.integral
            );
            // Lemma 2's quantity: log_mass == (β/α)·mass (the pin makes
            // a² + m constant); the f32 pair() inputs bound the agreement
            // at ~1e-7 relative — assert at 1e-5.
            let expected = beta_over_alpha * r.mass;
            assert!(
                (r.log_mass - expected).abs() <= 1e-5 * expected.abs(),
                "t={} Δ={}: log_mass {} vs (β/α)·mass {expected}",
                r.t,
                r.delta,
                r.log_mass
            );
        }

        // Θ(1) across the ladder body (t ≥ 10³, the +τ ramp done): the mass
        // moves < 1% from 10³ to 10⁵ at both Δ (the 10² row carries the ramp
        // correction, asserted via the integral match above).
        let at = |t: usize, d: f64| {
            rows.iter()
                .find(|r| r.t == t && r.delta == d)
                .map(|r| r.mass)
                .unwrap()
        };
        for d in [2.0, 10.0] {
            let ratio = at(100_000, d) / at(1_000, d);
            assert!(
                (ratio - 1.0).abs() <= 0.01,
                "decade mass drifted {ratio} across 10³→10⁵ at Δ={d}"
            );
        }
    }

    #[test]
    fn softmax_entropy_is_sublog_and_the_per_decade_logmass_collapses() {
        let lut = lut();
        let rows = softmax_entropy_ladder(&lut, &T_LADDER);

        for r in &rows {
            // Control: uniform over T keys → exactly ln(T).
            let ctrl_closed = (r.t_max as f64).ln();
            assert!(
                (r.h_control - ctrl_closed).abs() <= 1e-9,
                "T={}: H_control {} vs ln T {ctrl_closed}",
                r.t_max,
                r.h_control
            );
            // si entropy strictly under the control at every rung…
            assert!(
                r.h_si < r.h_control,
                "T={}: H_si {} ≥ control {}",
                r.t_max,
                r.h_si,
                r.h_control
            );
            // …with the ratio decreasing (the ~½·ln T asymptote approaching).
            let ratio = r.h_si / r.h_control;
            assert!(
                ratio < 0.96,
                "T={}: H_si/control = {ratio} — not sub-log",
                r.t_max
            );
        }
        let ratios: Vec<f64> = rows.iter().map(|r| r.h_si / r.h_control).collect();
        for w in ratios.windows(2) {
            assert!(w[1] < w[0], "H_si/control not decreasing: {w:?}");
        }

        // The paper's ACTUAL per-decade sparsity quantity is the
        // UNNORMALIZED log-mass (Lemma 2): si per-decade log-mass is Θ(1)
        // while the control's grows ∝ window length — the ratio collapses
        // like τ/t. Assert the collapse shape, not one flat threshold.
        let ts: [usize; 3] = [1_000, 10_000, 100_000];
        let big = ScaleInvariantLut::build(TAU, 1_000_001);
        let si_rows = decade_mass_report(&big, &ts, &[2.0]);
        let mut prev_ratio = f64::INFINITY;
        for r in &si_rows {
            let n_window = r.t as f64 * r.delta - r.t as f64;
            let ctrl_log_mass = control_key_mass(0) * n_window; // (a²+m)=1, e^{1/2} per key
            let ratio = r.log_mass / ctrl_log_mass;
            assert!(
                ratio < 0.01,
                "t={}: si log-mass {} vs control {ctrl_log_mass} (ratio {ratio})",
                r.t,
                r.log_mass
            );
            assert!(
                ratio < prev_ratio,
                "t={}: sparsity ratio {ratio} not collapsing (prev {prev_ratio})",
                r.t
            );
            prev_ratio = ratio;
        }
        // The top rung: collapsed to the 1e-4 class (∝ 1/t verified above).
        assert!(
            prev_ratio < 1e-4,
            "top-rung sparsity ratio {prev_ratio} — not collapsed"
        );
    }

    #[test]
    fn sigmoid_curve_mass_band_holds_and_control_is_flat() {
        let lut = lut();
        let ladder: Vec<usize> = vec![100, 316, 1_000, 3_162, 10_000, 31_622, 100_000];

        let si = sigmoid_curve_report(&lut, &ladder);
        let ctrl = sigmoid_control_curve(&ladder);

        // si mass shape: the G1 tilt curve stays in (0.14, 0.45) for
        // a_t ∈ [1, 5] (exact quadrature).
        for r in &si {
            assert!(
                (0.14..=0.45).contains(&r.mass_scaled),
                "t={}: mass_scaled {} outside the tilt band",
                r.t,
                r.mass_scaled
            );
        }

        // Control flatness: identity schedule → E[σ(L)] = ½ at every
        // distance (exact quadrature — assert at 1e-12) and the per-mass
        // entropy curve is exactly flat.
        for r in &ctrl {
            assert!(
                (r.e_sigma - 0.5).abs() < 1e-12,
                "t={}: control E[σ] = {} ≠ ½",
                r.t,
                r.e_sigma
            );
        }
        let per_mass: Vec<f64> = ctrl.iter().map(|r| r.per_mass).collect();
        let mn = per_mass.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = per_mass.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            mx - mn < 1e-12,
            "control per-mass curve not flat: [{mn}, {mx}]"
        );
    }

    #[test]
    fn sigmoid_si_curve_is_sublog_in_every_quantity() {
        let lut = lut();
        let ladder: Vec<usize> = vec![100, 316, 1_000, 3_162, 10_000, 31_622, 100_000];
        let si = sigmoid_curve_report(&lut, &ladder);

        // Exact quadrature: the per-key negentropy DECAYS with distance
        // (mass decay dominates the link's widening) while the per-mass
        // entropy GROWS — both strictly sub-log against ln t.
        let ln_t: Vec<f64> = ladder.iter().map(|t| (*t as f64).ln()).collect();

        // neg_entropy(t)/neg_entropy(t₀) < ln(t)/ln(t₀) — the curve is
        // sub-log (it decays, so this holds trivially per rung; the law
        // asserted is that it never turns log-shaped).
        for (i, r) in si.iter().enumerate().skip(1) {
            let ratio = r.neg_entropy / si[0].neg_entropy;
            let log_ratio = ln_t[i] / ln_t[0];
            assert!(
                ratio < log_ratio,
                "t={}: negentropy ratio {ratio} ≥ log ratio {log_ratio}",
                r.t
            );
        }

        // per_mass grows, but per_mass/ln t falls — the √log-t shape.
        let per_mass: Vec<f64> = si.iter().map(|r| r.per_mass).collect();
        for w in per_mass.windows(2) {
            assert!(
                w[1] > w[0],
                "per_mass not increasing: {:.6} → {:.6}",
                w[0],
                w[1]
            );
        }
        let scaled: Vec<f64> = per_mass.iter().zip(&ln_t).map(|(p, l)| p / l).collect();
        for w in scaled.windows(2) {
            assert!(w[1] < w[0], "per_mass/ln t not decreasing: {w:?}");
        }
    }

    #[test]
    fn tau_sweep_verdict_passes_within_tolerance_and_names_offenders() {
        let rows = vec![
            TauSweepRow {
                tau: 1.0,
                in_dist_ppl_delta_pct: 0.4,
            },
            TauSweepRow {
                tau: 10.0,
                in_dist_ppl_delta_pct: 1.2,
            },
            TauSweepRow {
                tau: 100.0,
                in_dist_ppl_delta_pct: 2.9,
            },
        ];
        assert_eq!(assess_tau_sweep(&rows, 3.0), TauSweepVerdict::Pass);
        assert_eq!(
            assess_tau_sweep(&rows, 1.0),
            TauSweepVerdict::Fail(vec![(10.0, 1.2), (100.0, 2.9)])
        );
        // Exactly-at-tolerance passes (the tolerance is inclusive).
        assert_eq!(assess_tau_sweep(&rows[..1], 0.4), TauSweepVerdict::Pass);
    }

    #[test]
    #[should_panic(expected = "tau sweep: no rows")]
    fn tau_sweep_refuses_an_empty_sweep() {
        let _ = assess_tau_sweep(&[], 3.0);
    }
}
