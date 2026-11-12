//! Belief-contraction certificates + derived staleness schedules for the
//! modality-additive belief-kernel family (Issue 1000; Research 667,
//! arXiv:2611.09217).
//!
//! # Two spaces — every helper names its space
//!
//! The kernel iterates the **discrete map**
//!
//! ```text
//! h_{t+1,i} = α_{m[i]}·h_{t,i} + (1−α_{m[i]})·drive_i   (clamped to [−1,1])
//! ```
//!
//! whose trajectory-pair distance decays as `ρ^t` with `ρ = α` EXACT and rate
//! `κ = −ln α` ([`contraction_rate`], [`settling_ticks`],
//! [`disturbance_fade_ticks`]). The paper's analysis lives in the
//! **continuous residual field** `ġ = −(1−α)(h − drive)` (Jacobian
//! `−(1−α)·I`, uniformly monotone): a residual cell
//! `ż = κ·(−α_i·z_i + λ_i·tanh(·))` is contractive iff `α_i − |λ_i| ≥ 1`
//! ([`residual_cell_certificate`]). Bridge between spaces (Research 667
//! §4.1): `δκ ↔ 1 − α_m`; exact Euler of the linear dissipative field gives
//! `ρ = 1 − δκ`, and the paper's general nonlinear bound `ρ = 1 − δκ/2`
//! (valid under `δκ·L_z² ≤ 1`, [`step_law`]) SPECIALIZES to it for the
//! additive family. One kernel, one number: `ρ = α_m`.
//!
//! The divisive kernel ([`crate::leaky_core::leaky_step`]) is NOT a
//! contraction — its update is state-independent and fades at the clamp
//! boundary. That is a recorded measured distinction (Research 667 §4.1),
//! never a defect; the additive kernel is the one that carries a certificate.
//!
//! Zero allocation, `f32` only, no new deps, sigmoid never softmax. Pure
//! helpers compiled ungated beside [`crate::leaky_core`] (the same
//! single-source-of-truth rationale): inert until a consumer calls them; no
//! promotion claim (Issue 1000 non-goals — T3's derived-λ consumer seam is
//! gated on a noise-input decision and lives in riir-engine, not here).

/// Structural radius of the clamped belief ball: the clamp domain `[−1,1]⁸`
/// gives `‖z‖ ≤ √8`. Conservative by construction — the additive drive is
/// `2σ(η·k)−1 ∈ (−1,1)`, so the reachable set is tighter still; the slack is
/// fixture tolerance budget (F4), never slack in the law (Research 667 C2).
pub const ADDITIVE_BALL_RADIUS: f32 = 2.0 * core::f32::consts::SQRT_2;

/// Typed refusal for every certificate / schedule input deficit. One variant
/// per named deficit — fail-closed, never a silent fallback.
#[derive(Debug, Clone, PartialEq)]
pub enum BeliefContractError {
    /// Retention `α ∉ (0,1)`: the discrete map has no exact contraction rate
    /// outside the open unit interval (`α = 1` never contracts — κ = 0;
    /// `α ≤ 0` is not a filter). NaN fails both comparisons and refuses here.
    RetentionOutOfRange { alpha: f32 },
    /// A schedule input that must be strictly positive (`ε`, `r_max`, `‖δ‖`,
    /// `L̄`, `κ_fade`, the settle horizon) was not. `name` names the deficit.
    NonPositiveInput { name: &'static str, value: f32 },
    /// Continuous-flow certificate violated: `α_i − |λ_i| < 1` — the residual
    /// cell is not uniformly monotone. `deficit = 1 − (α_i − |λ_i|)` names how
    /// far below the iff the cell sits.
    NotContractive {
        alpha: f32,
        lambda_abs: f32,
        deficit: f32,
    },
    /// S4 step-law outside the paper's validated region: `δκ·L_z² > 1` (the
    /// general nonlinear bound `ρ = 1 − δκ/2` is a guarantee only inside it),
    /// or `δκ ≥ 2` where the bound stops being a decay at all. `value` is the
    /// quantity that exceeded its limit.
    StepOutsideSafeRegion { value: f32 },
}

impl core::fmt::Display for BeliefContractError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RetentionOutOfRange { alpha } => {
                write!(
                    f,
                    "belief retention alpha={alpha} outside (0,1): no exact contraction rate"
                )
            }
            Self::NonPositiveInput { name, value } => {
                write!(f, "belief_contract {name}={value} must be > 0")
            }
            Self::NotContractive {
                alpha,
                lambda_abs,
                deficit,
            } => {
                write!(
                    f,
                    "residual cell not contractive: alpha={alpha} - |lambda|={lambda_abs} is {deficit} below 1"
                )
            }
            Self::StepOutsideSafeRegion { value } => {
                write!(
                    f,
                    "step outside safe region: {value} exceeds its limit (delta_kappa*L_z^2 <= 1 and delta_kappa < 2)"
                )
            }
        }
    }
}

impl std::error::Error for BeliefContractError {}

/// Discrete-space certificate + rate for one retention channel (Research 667
/// C1). `ρ = α` exact, `κ = −ln α`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContractionRate {
    /// Per-tick distance multiplier (`ρ^t` decay law). Equals `alpha`.
    pub rho: f32,
    /// Per-tick contraction rate `−ln α` (nats/tick; ≈ 0.105 at the shipped
    /// `α = 0.9` — half-life ≈ 6.6 ticks at a 20 Hz cognition tick).
    pub kappa: f32,
}

/// Discrete-space certificate for one retention channel: `ρ = α` exact,
/// `κ = −ln α`; refuses `α ∉ (0,1)` by name (including NaN and `α = 1`,
/// which never contracts).
#[inline]
pub fn contraction_rate(alpha: f32) -> Result<ContractionRate, BeliefContractError> {
    // Open-interval check; NaN is refused by the explicit arm.
    if alpha.is_nan() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(BeliefContractError::RetentionOutOfRange { alpha });
    }
    Ok(ContractionRate {
        rho: alpha,
        kappa: -alpha.ln(),
    })
}

/// S1 — discrete-space settling horizon: ticks until an initial separation
/// bounded by `2·r_max` decays below `eps`:
/// `t = ceil( ln(2·r_max / ε) / (−ln α) )`. For the clamped 8-dim belief ball
/// pass `r_max = ADDITIVE_BALL_RADIUS` (√8 — conservative; the reachable set
/// is tighter). Returns 0 when the budget already holds at t = 0. Saturates
/// at `u32::MAX` for near-unity `α` (the honest answer: effectively never).
pub fn settling_ticks(alpha: f32, eps: f32, r_max: f32) -> Result<u32, BeliefContractError> {
    let rate = contraction_rate(alpha)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(BeliefContractError::NonPositiveInput {
            name: "eps",
            value: eps,
        });
    }
    if !r_max.is_finite() || r_max <= 0.0 {
        return Err(BeliefContractError::NonPositiveInput {
            name: "r_max",
            value: r_max,
        });
    }
    let span = 2.0 * r_max;
    if span <= eps {
        return Ok(0);
    }
    // ln(2·r_max/ε) / κ — f32 is integer-exact to 2^24, far past any real
    // horizon this family produces; near-unity α saturates at u32::MAX.
    let t = (span / eps).ln() / rate.kappa;
    Ok(t.ceil().max(0.0).min(u32::MAX as f32) as u32)
}

/// S2 — discrete-space disturbance fade: ticks of clean observations after
/// which a disturbance residual bounded by `noise_bound` has washed out below
/// `eps`: `t = ceil( ln(‖δ‖ / ε) / κ_fade )`. THE CALLER NAMES THE FADE RATE:
/// the full `κ_m` for the exact-linear additive kernel (its disturbance-driven
/// trajectory difference obeys `Δh_{t+1} = α_m·Δh_t` once the disturbance
/// stops), or the generic nonlinear bound `κ_m / 2`. This is a washout law
/// for PAST disturbances after clean observations RESUME — never a statement
/// about what an observation GAP costs: during a gap no new information
/// arrives and error vs a moving truth grows unchecked; that side is governed
/// by the physical flow and the re-observe cadence (Research 667 S2).
pub fn disturbance_fade_ticks(
    kappa_fade: f32,
    noise_bound: f32,
    eps: f32,
) -> Result<u32, BeliefContractError> {
    if !kappa_fade.is_finite() || kappa_fade <= 0.0 {
        return Err(BeliefContractError::NonPositiveInput {
            name: "kappa_fade",
            value: kappa_fade,
        });
    }
    if !noise_bound.is_finite() || noise_bound <= 0.0 {
        return Err(BeliefContractError::NonPositiveInput {
            name: "noise_bound",
            value: noise_bound,
        });
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(BeliefContractError::NonPositiveInput {
            name: "eps",
            value: eps,
        });
    }
    if noise_bound <= eps {
        return Ok(0);
    }
    let t = (noise_bound / eps).ln() / kappa_fade;
    Ok(t.ceil().max(0.0).min(u32::MAX as f32) as u32)
}

/// S3 — useful-gain ceiling: the largest per-tick rate worth paying for a
/// settle horizon of `settle_target` ticks under readout Lipschitz `l_bar`
/// and observation-noise bound `noise_bound`:
/// `κ* = ln(1/(L̄·δ)) / t_o`. `None` is the TYPED SNR REFUSAL — `L̄·δ ≥ 1`
/// means no contraction rate can beat a noise product at or above unity
/// (fail-closed; the deficit is the `L̄·δ` product itself). Non-positive
/// inputs refuse the same way.
///
/// NOTE: the observability order ℓ does NOT enter κ* — it enters the
/// steady-state FLOOR (`∝ κ^ℓ`), which is where a diminishing-returns lint
/// belongs; at ℓ = 0 (the additive kernel's identity readout) the floor is
/// flat in κ and no κ* ceiling is observable (F3's consistency arm,
/// Research 667 §5).
#[inline]
pub fn useful_gain(settle_target: f32, noise_bound: f32, l_bar: f32) -> Option<f32> {
    if !settle_target.is_finite() || settle_target <= 0.0 {
        return None;
    }
    if !noise_bound.is_finite() || noise_bound <= 0.0 {
        return None;
    }
    if !l_bar.is_finite() || l_bar <= 0.0 {
        return None;
    }
    let product = l_bar * noise_bound;
    if product.is_nan() || product >= 1.0 {
        // `L̄·δ ≥ 1` (NaN refuses explicitly): the typed SNR refusal.
        return None;
    }
    Some((1.0 / product).ln() / settle_target)
}

/// Continuous-flow-space certificate for a residual cell
/// `ż = κ·(−α_i·z_i + λ_i·tanh(·))` — the paper's uniform-monotonicity iff:
/// contractive ⟺ `α_i − |λ_i| ≥ 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualCellCertificate {
    /// `α_i − |λ_i| − 1 ≥ 0` — how far above the iff the cell sits.
    pub margin: f32,
    /// Worst-case flow decay rate `α_i − |λ_i|` (the field's measure).
    pub decay_rate: f32,
}

/// Continuous-flow-space certificate: `α_i − |λ_i| ≥ 1` (the paper's iff).
/// Refuses by name with the deficit when violated. Bridge to discrete space:
/// `δκ ↔ 1 − α_m`; exact Euler of the linear part gives `ρ = 1 − δκ`.
#[inline]
pub fn residual_cell_certificate(
    alpha: f32,
    lambda: f32,
) -> Result<ResidualCellCertificate, BeliefContractError> {
    let lambda_abs = lambda.abs();
    let measure = alpha - lambda_abs;
    if measure.is_nan() || measure < 1.0 {
        // NaN measure refuses explicitly: an incomparable cell is not certified.
        return Err(BeliefContractError::NotContractive {
            alpha,
            lambda_abs,
            deficit: 1.0 - measure,
        });
    }
    Ok(ResidualCellCertificate {
        margin: measure - 1.0,
        decay_rate: measure,
    })
}

/// S4 — step-law validation + per-tick confidence multiplier for Euler
/// integration of a residual latent update with field Lipschitz constant
/// `l_z` and step `delta_kappa` (both in the field's own units). Refuses
/// `δκ·L_z² > 1` — outside the paper's validated region the general
/// nonlinear bound is a guarantee no longer — and also `δκ ≥ 2`, where
/// `ρ = 1 − δκ/2` stops being a decay. Annotation-level for current
/// consumers; the additive kernel's exact `ρ = 1 − δκ = α_m` is tighter and
/// lives in [`contraction_rate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepLaw {
    /// Guaranteed per-tick distance multiplier bound `1 − δκ/2` (valid only
    /// inside the safe region this function certifies).
    pub rho_bound: f32,
}

#[inline]
pub fn step_law(delta_kappa: f32, l_z: f32) -> Result<StepLaw, BeliefContractError> {
    if !delta_kappa.is_finite() || delta_kappa <= 0.0 {
        return Err(BeliefContractError::NonPositiveInput {
            name: "delta_kappa",
            value: delta_kappa,
        });
    }
    if !l_z.is_finite() || l_z < 0.0 {
        return Err(BeliefContractError::NonPositiveInput {
            name: "l_z",
            value: l_z,
        });
    }
    let scaled = delta_kappa * l_z * l_z;
    if scaled > 1.0 {
        return Err(BeliefContractError::StepOutsideSafeRegion { value: scaled });
    }
    if delta_kappa >= 2.0 {
        return Err(BeliefContractError::StepOutsideSafeRegion { value: delta_kappa });
    }
    Ok(StepLaw {
        rho_bound: 1.0 - delta_kappa / 2.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const KAPPA_09: f32 = 0.105_360_516; // −ln(0.9)

    #[test]
    fn contraction_rate_matches_closed_form() {
        let c = contraction_rate(0.9).expect("0.9 in (0,1)");
        assert_eq!(c.rho, 0.9);
        assert!(
            (c.kappa - KAPPA_09).abs() < 1e-6,
            "kappa {} vs {KAPPA_09}",
            c.kappa
        );
    }

    #[test]
    fn contraction_rate_refuses_out_of_open_interval() {
        for alpha in [0.0, -0.5, 1.0, 1.5, f32::NAN] {
            assert!(
                matches!(
                    contraction_rate(alpha),
                    Err(BeliefContractError::RetentionOutOfRange { .. })
                ),
                "alpha={alpha} must refuse"
            );
        }
    }

    /// S1 at the shipped α: `ceil(ln(2√8/0.1)/0.10536) = 39` ticks
    /// (Research 667 quotes ≈38 from the same arithmetic).
    #[test]
    fn settling_ticks_shipped_point() {
        let t = settling_ticks(0.9, 0.1, ADDITIVE_BALL_RADIUS).expect("valid");
        assert_eq!(
            t, 39,
            "ln(2*2.8284/0.1)=4.0355 / 0.10536 = 38.30 -> ceil 39"
        );
    }

    #[test]
    fn settling_ticks_edge_cases() {
        // Budget already holds.
        assert_eq!(settling_ticks(0.9, 10.0, ADDITIVE_BALL_RADIUS).unwrap(), 0);
        // Near-unity α: effectively never (saturating, not wrapping).
        let t = settling_ticks(1.0 - 1e-6, 0.1, ADDITIVE_BALL_RADIUS).unwrap();
        assert!(t > 1_000_000, "near-unity horizon {t}");
        // Input deficits refuse by name.
        assert!(matches!(
            settling_ticks(0.9, 0.0, 1.0),
            Err(BeliefContractError::NonPositiveInput { name: "eps", .. })
        ));
        assert!(matches!(
            settling_ticks(0.9, 0.1, 0.0),
            Err(BeliefContractError::NonPositiveInput { name: "r_max", .. })
        ));
    }

    /// S2 at the shipped α: `ceil(ln(1/0.1)/0.10536) = 22` ticks.
    #[test]
    fn disturbance_fade_shipped_point() {
        let t = disturbance_fade_ticks(KAPPA_09, 1.0, 0.1).expect("valid");
        assert_eq!(t, 22, "ln(10)=2.3026 / 0.10536 = 21.85 -> ceil 22");
        // Generic nonlinear bound (κ/2) is the conservative upper schedule.
        let t_half = disturbance_fade_ticks(KAPPA_09 / 2.0, 1.0, 0.1).unwrap();
        assert!(t_half >= t);
        assert_eq!(disturbance_fade_ticks(KAPPA_09, 0.05, 0.1).unwrap(), 0);
        assert!(matches!(
            disturbance_fade_ticks(0.0, 1.0, 0.1),
            Err(BeliefContractError::NonPositiveInput {
                name: "kappa_fade",
                ..
            })
        ));
    }

    /// S3: closed form at the F4 posture; typed SNR refusal at `L̄·δ ≥ 1`.
    #[test]
    fn useful_gain_closed_form_and_refusal() {
        // ln(1/(1.0*0.05))/38 = 2.9957/38 = 0.07883.
        let g = useful_gain(38.0, 0.05, 1.0).expect("feasible");
        assert!((g - 0.078_83).abs() < 1e-4, "gain {g}");
        // `L̄·δ = 1.0` and above: typed refusal (None = the deficit shape).
        assert_eq!(useful_gain(38.0, 1.0, 1.0), None);
        assert_eq!(useful_gain(38.0, 0.05, 25.0), None);
        // Non-positive inputs refuse the same way.
        assert_eq!(useful_gain(0.0, 0.05, 1.0), None);
        assert_eq!(useful_gain(38.0, f32::NAN, 1.0), None);
    }

    /// Continuous-flow certificate: the iff holds with margin above 1,
    /// refuses with the deficit below it.
    #[test]
    fn residual_cell_certificate_iff() {
        let c = residual_cell_certificate(2.0, 0.5).expect("2 - 0.5 >= 1");
        assert!((c.margin - 0.5).abs() < 1e-6);
        assert!((c.decay_rate - 1.5).abs() < 1e-6);
        // Boundary: exactly 1 is contractive (iff is ≥) — 1.5 − 0.5 is exact
        // in f32 (both values binary-representable).
        assert!(residual_cell_certificate(1.5, 0.5).is_ok());
        // Violations name the deficit.
        let e = residual_cell_certificate(1.0, 0.05).expect_err("0.95 < 1");
        assert!(
            matches!(e, BeliefContractError::NotContractive { deficit, .. } if (deficit - 0.05).abs() < 1e-6)
        );
        assert!(matches!(
            residual_cell_certificate(f32::NAN, 0.0),
            Err(BeliefContractError::NotContractive { .. })
        ));
    }

    /// S4: inside the safe region the bound holds against the exact linear
    /// multiplier; outside it the certificate refuses loud.
    #[test]
    fn step_law_safe_region_and_refusals() {
        // δκL_z² = 0.1·4 = 0.4 <= 1: bound 0.95 ≥ exact linear multiplier
        // (1 − δκ·L_z = 0.8) for the field f(z) = −L_z·z.
        let law = step_law(0.1, 2.0).expect("inside safe region");
        assert!((law.rho_bound - 0.95).abs() < 1e-6);
        let exact = 1.0 - 0.1 * 2.0;
        assert!(
            exact <= law.rho_bound + 1e-6,
            "exact {exact} must sit under the bound"
        );
        // δκL_z² = 1.6 > 1: refused (the paper's guarantee region).
        assert!(matches!(
            step_law(0.1, 4.0),
            Err(BeliefContractError::StepOutsideSafeRegion { .. })
        ));
        // δκ ≥ 2: the multiplier stops being a decay — refused even with L_z = 0.
        assert!(matches!(
            step_law(2.0, 0.0),
            Err(BeliefContractError::StepOutsideSafeRegion { .. })
        ));
        assert!(matches!(
            step_law(0.0, 1.0),
            Err(BeliefContractError::NonPositiveInput {
                name: "delta_kappa",
                ..
            })
        ));
        assert!(matches!(
            step_law(0.1, -1.0),
            Err(BeliefContractError::NonPositiveInput { name: "l_z", .. })
        ));
    }

    /// F3 PRIMARY ARM (issue 1000 T2) — the κ^ℓ law made observable on a
    /// synthetic ℓ = 1 observed-integrator cell: latent velocity, observed
    /// position, noise on the readout. The estimator is the κ-filtered
    /// differentiator `v̂_{t+1} = (1−κ)·v̂_t + κ·(y_t − y_{t−1})`; for constant
    /// velocity the signal passes exactly and the noise floor is the leaky
    /// filter's response to the differenced bounded noise, whose steady peak
    /// is `2κδ/(2−κ)` — RISING in κ. Asserts the rise (`floor(2κ*) >
    /// floor(κ*)`) and the closed-form scaling, so "more gain raises your
    /// floor" has its observable exactly where ℓ ≥ 1.
    #[test]
    fn f3_primary_floor_rises_for_kappa_above_kappa_star() {
        let v = 0.5f32; // constant latent velocity
        let delta = 0.05f32; // readout noise bound
        let warm = 2_000usize;
        let meas = 400usize;

        let floor_at = |kappa: f32| -> f32 {
            let mut p = 0.0f32; // position
            let mut y_prev = 0.0f32;
            let mut v_hat = 0.0f32;
            let mut peak = 0.0f32;
            for t in 0..warm + meas {
                p += v;
                // Alternating ±δ bounded readout noise (zero-mean, worst-case
                // shaped; the deterministic stand-in for a bounded noise
                // process — Research 667 F3 primary).
                let noise = if t % 2 == 0 { delta } else { -delta };
                let y = p + noise;
                let diff = y - y_prev;
                y_prev = y;
                v_hat = (1.0 - kappa) * v_hat + kappa * diff;
                if t >= warm {
                    peak = peak.max((v_hat - v).abs());
                }
            }
            peak
        };

        let kappa_star = 0.05f32;
        let f_lo = floor_at(kappa_star);
        let f_hi = floor_at(2.0 * kappa_star);
        assert!(
            f_hi > f_lo,
            "floor must rise for kappa > kappa*: floor(2k*)={f_hi} vs floor(k*)={f_lo}"
        );
        // Closed form `2κδ/(2−κ)` — the κ^1 scaling visible.
        let expect_lo = 2.0 * kappa_star * delta / (2.0 - kappa_star);
        assert!(
            (f_lo - expect_lo).abs() < 0.1 * expect_lo,
            "floor(k*) {f_lo} vs closed form {expect_lo}"
        );
        let expect_hi = 2.0 * (2.0 * kappa_star) * delta / (2.0 - 2.0 * kappa_star);
        assert!(
            (f_hi - expect_hi).abs() < 0.1 * expect_hi,
            "floor(2k*) {f_hi} vs closed form {expect_hi}"
        );
    }

    /// F5 (issue 1000 T2) — S4's condition is live. Inside the safe region a
    /// linear residual cell contracts with its exact multiplier under the
    /// certified bound; a cell outside ANY safe multiplier diverges
    /// geometrically; and the certificate refuses just over the edge. NOTE on
    /// normalization: the LINEAR cell's discrete divergence edge is
    /// `|1 − δκ·L_z| > 1` (`δκ·L_z > 2`); the certificate's safe region
    /// (`δκ·L_z² ≤ 1`) is the paper's guarantee region for the GENERAL
    /// nonlinear bound — stricter by design, so "just over 1" fires at the
    /// certificate (typed refusal) while the simulated divergence arm uses a
    /// multiplier past the linear edge.
    #[test]
    fn f5_stability_edge_is_live() {
        let delta_kappa = 0.1f32;
        let l_z = 2.0f32;

        // Under the edge: certified, and the simulated linear cell contracts
        // with its exact multiplier (1 − δκ·L_z) sitting under the bound.
        let law = step_law(delta_kappa, l_z).expect("inside safe region");
        let exact = 1.0 - delta_kappa * l_z;
        let mut z = 1.0f32;
        for _ in 0..64 {
            z += delta_kappa * (-l_z * z);
        }
        let measured = z.abs(); // |z_64| = exact^64
        assert!(measured < 0.5, "contracted to {measured}");
        assert!(exact <= law.rho_bound + 1e-6);

        // The certificate refuses just over the edge (δκL_z² = 1.1 > 1).
        assert!(matches!(
            step_law(delta_kappa, 10.0 * (1.1f32 / delta_kappa).sqrt()),
            Err(BeliefContractError::StepOutsideSafeRegion { .. })
        ));

        // Past the linear divergence edge (δκ·L_z = 2.2 → multiplier −1.2):
        // geometric growth — the divergence the safe region exists to exclude.
        let mut z = 1.0f32;
        let diverging_step = 2.2f32 / l_z;
        for _ in 0..16 {
            z += diverging_step * (-l_z * z);
        }
        assert!(z.abs() > 10.0, "expected divergence, |z|={}", z.abs());
    }
}
