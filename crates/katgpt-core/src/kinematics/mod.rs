//! Kinematic rollout primitive — the modelless core of Latent Dynamics
//! Reasoning (Plan 578 / Research 506, arXiv:2608.09926 Li et al.).
//!
//! LDR's headline result (ID-OOD extrapolation gap 20×+ smaller than a video
//! diffusion baseline) is — by the paper's **own ablation table** — carried by
//! its *fixed* kinematic integrator, not the learned residual: removing
//! dynamics reasoning widens the gap 13× while keeping the reasoning and
//! swapping only the latent degrades to the 2nd-best result. This module
//! ships that fixed math as generic operators with zero weights:
//!
//! | Operator | What it does |
//! |---|---|
//! | [`KinState::observe_into`] | ring-fed finite-difference state estimation (order ladder 0→3) |
//! | [`kinematic_extrapolate_into`] | O(1) closed-form k-step rollout (Newton backward form) |
//! | [`Sched`] | deterministic stand-ins for the learned ≥3rd-order residual |
//! | [`perception::time_to_contact`] | looming τ = σ/σ̇ (Lee optic-flow TTC) |
//! | [`perception::RegimeClassifier`] | closed-form kinematic regime classification + sigmoid hysteresis |
//! | [`perception::ResidualMonitor`] | prediction-residual surprise (z-score / CUSUM) + impulse-vs-force + restitution |
//! | [`perception::closest_approach`] | two-body t*, miss distance, intercept, elastic resolve |
//! | [`perception::extrapolation_horizon`] | error-propagation bound B(k) + admission horizon k* |
//!
//! # The math (and why it is exact)
//!
//! The paper's semi-implicit Euler chain (orders 2→1→0) is a *difference
//! engine*. With observations `x[m-3..=m]` sampled `Δt` apart, form the
//! **backward differences** at the anchor (latest observation):
//!
//! ```text
//! ∇¹ = x[m] − x[m-1]
//! ∇² = x[m] − 2·x[m-1] + x[m-2]
//! ∇³ = x[m] − 3·x[m-1] + 3·x[m-2] − x[m-3]
//! ```
//!
//! and the **Newton backward (Gregory–Newton) closed form**
//!
//! ```text
//! x̂(k) = x[m] + C(k,1)·∇¹ + C(k+1,2)·∇² + C(k+2,3)·∇³
//! ```
//!
//! is the unique degree-≤3 polynomial through the window evaluated at
//! `m + k` — **exactly**, for any horizon, with zero loops. That is the
//! provable strengthening of the paper's empirical 20× ID-OOD gap: on the
//! analytic family the in-distribution and out-of-distribution error are
//! *identically zero*, hence gap ≡ 0 by construction (the paper's OOD ranges
//! change the *magnitudes*, not the polynomial family).
//!
//! The state stores the differences scaled to physical units
//! (`vel = ∇¹/Δt`, `acc = ∇²/Δt²`, `jerk = ∇³/Δt³`); `acc` and `jerk` are the
//! physical acceleration / jerk **exactly** on quadratic / cubic motion,
//! while `vel` is the mean velocity over the last step (the backward
//! difference lags instantaneous velocity by half a step — exact under the
//! binomial form above; use [`central_velocity`] for an O(Δt²) unbiased
//! instantaneous estimate).
//!
//! # Bit-identity with the step-by-step chain
//!
//! The O(1) closed form and the O(k) difference-engine chain
//! (`d2 += d3; d1 += d2; s += d1`, top-down order — see
//! [`reference_chain_extrapolate_into`]) evaluate the *same polynomial*
//! through different float operation sequences. On the exactness fixtures
//! (dyadic-representable trajectories) every operation in both paths is
//! exact, so the two agree **bit-for-bit**; on arbitrary float data they
//! agree to a few ULP (measured in the GOAT bench — `.benchmarks/680`). No
//! O(1) rearrangement of an O(k) accumulation can be bit-identical in
//! general; the exactness family is where the identity holds exactly, and
//! that is where the plan needs it.
//!
//! # f32 exactness horizon (honest limit)
//!
//! The lattice coefficient `C(k+2,3)` exceeds f32's 24-bit exact-integer
//! range at `k ≥ 288` (C(290,3) = 4,086,980 is the last 24-bit-exact value;
//! C(1002,3) = 167,167,000 needs 25 mantissa bits). Degree ≤ 2 trajectories
//! stay bit-exact through the full lattice (`C(1025,2) = 525,250 ≪ 2²⁴`). The
//! GOAT bench documents the measured boundary: exact-0 at k ∈ {1,10,100} for
//! const-jerk, k ∈ {1,10,100,1000} for uniform/parabola, and a ~2⁻²⁴ relative
//! band beyond the boundary.
//!
//! # Sync-boundary discipline
//!
//! Everything here is **think-brain** state: extrapolated positions are
//! subjective, local, never synced, never used to validate movement claims
//! (prediction ≠ validation — the anti-cheat rule). Physical (raw) state
//! crossing a sync surface must be observed, not extrapolated.
//!
//! # Modelless
//!
//! Pure f32 arithmetic, zero heap, zero deps, `#[repr(C)]` POD state,
//! per-channel independent (SIMD-able). The `Sched` family replaces the
//! paper's learned `tanh`-MLP residual with deterministic closed forms.
//!
//! Kinematic rollout primitive (Plan 578 / Research 506, arXiv:2608.09926 —
//! LDR, Li et al.): the modelless core of latent dynamics reasoning —
//! finite-difference state (order ladder 0→3), O(1) Newton-backward closed-
//! form k-step rollout exact on degree ≤ 3 motion (the provable ID-OOD gap ≡ 0
//! strengthening of the paper's empirical ~20×), deterministic jerk/drag
//! schedules, looming time-to-contact, regime predicates with sigmoid
//! hysteresis, residual-surprise events (z/CUSUM/impulse + restitution),
//! two-body closest approach/intercept, and the UQ-bearing extrapolation-
//! horizon admission bound (RANK-ONLY verdict vs the conformal floor —
//! .benchmarks/677). Pure f32 math, zero deps, zero allocs, #[repr(C)] POD
//! state. DEFAULT-ON since 2026-08-26 (Bench 677 GOAT G1–G4 ALL PASS — the
//! KARC precedent); consumer PoC: riir-ai Issue 757.

pub mod fixture;
pub mod perception;

#[cfg(test)]
mod tests;

/// Maximum extrapolation horizon supported by the coefficient lattice.
///
/// `C(1026,3) ≈ 1.79e8` is beyond f32's exact-integer range (documented
/// above); the lattice stores the correctly-rounded f32 values so consumers
/// get stable bits at every supported k.
pub const K_MAX: usize = 1024;

/// Ladder cap: the observation budget that saturates the order ladder.
pub const MAX_LADDER_OBS: u8 = 4;

// ===== coefficient lattice =====

/// One lattice row: the three Newton-backward binomials at horizon k.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LatticeRow {
    /// C(k,1) = k
    pub(crate) b1: f32,
    /// C(k+1,2) = k(k+1)/2
    pub(crate) b2: f32,
    /// C(k+2,3) = k(k+1)(k+2)/6
    pub(crate) b3: f32,
}

/// The full lattice, computed once per process in f64 then rounded to f32
/// (best available rounding — an incremental f32 recurrence would accumulate
/// rounding). Static storage: no heap allocation per call after init.
static LATTICE: std::sync::OnceLock<Vec<LatticeRow>> = std::sync::OnceLock::new();

pub(crate) fn lattice() -> &'static [LatticeRow] {
    LATTICE.get_or_init(|| {
        let mut rows = Vec::with_capacity(K_MAX + 1);
        for k in 0..=K_MAX {
            let kf = k as f64;
            rows.push(LatticeRow {
                b1: kf as f32,
                b2: (kf * (kf + 1.0) * 0.5) as f32,
                b3: (kf * (kf + 1.0) * (kf + 2.0) / 6.0) as f32,
            });
        }
        rows
    })
}

/// Observation / extrapolation errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KinError {
    /// A position sample (or derived quantity) was NaN / infinite.
    NonFinite,
    /// The sample period was zero, negative, or non-finite.
    BadDt,
    /// Observation ticks must be strictly increasing (uniform-Δt stencil).
    NonMonotonicTick,
    /// Requested horizon exceeds [`K_MAX`].
    HorizonTooFar,
    /// Not enough observations for the requested operation.
    NotEnoughObs,
}

impl core::fmt::Display for KinError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            KinError::NonFinite => write!(f, "non-finite sample"),
            KinError::BadDt => write!(f, "sample period must be finite and > 0"),
            KinError::NonMonotonicTick => write!(f, "ticks must be strictly increasing"),
            KinError::HorizonTooFar => write!(f, "horizon exceeds K_MAX"),
            KinError::NotEnoughObs => write!(f, "not enough observations"),
        }
    }
}

impl std::error::Error for KinError {}

/// Finite-difference kinematic state over `D` independent channels.
///
/// The anchor is the **latest observation**; `vel`/`acc`/`jerk` are the
/// backward differences scaled to physical units (see the module doc for the
/// exactness argument). `#[repr(C)]` POD — blittable, zero heap, per-channel
/// independent.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct KinState<const D: usize> {
    /// Anchor position = the latest observation (order-0 estimate).
    pub pos: [f32; D],
    /// `∇¹/Δt` — order-1 coefficient (mean velocity over the last step).
    pub vel: [f32; D],
    /// `∇²/Δt²` — order-2 coefficient (physical acceleration, exact on
    /// quadratics).
    pub acc: [f32; D],
    /// `∇³/Δt³` — order-3 coefficient (physical jerk, exact on cubics).
    /// Zero until `n_obs ≥ 4`.
    pub jerk: [f32; D],
    /// Tick of the anchor observation.
    pub tick: u32,
    /// Number of observations absorbed (ladder position; saturates at 4).
    pub n_obs: u8,
    /// Uniform sample period (set once at construction).
    pub dt: f32,
}

impl<const D: usize> KinState<D> {
    /// Construct an empty state with sample period `dt`.
    ///
    /// Screens: `dt` must be finite and > 0 ([`KinError::BadDt`]).
    pub fn new(dt: f32) -> Result<Self, KinError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(KinError::BadDt);
        }
        Ok(Self {
            pos: [0.0; D],
            vel: [0.0; D],
            acc: [0.0; D],
            jerk: [0.0; D],
            tick: 0,
            n_obs: 0,
            dt,
        })
    }

    /// Absorb one observation, advancing the finite-difference ladder.
    ///
    /// The recurrence in scaled space (an exact translation of the backward
    /// difference table):
    ///
    /// ```text
    /// vel_new  = (x − pos_old) / Δt
    /// acc_new  = (vel_new − vel_old) / Δt      (once n_obs ≥ 2)
    /// jerk_new = (acc_new − acc_old) / Δt      (once n_obs ≥ 3)
    /// ```
    ///
    /// Ladder: 1 obs → order 0, 2 → order 1, 3 → order 2, 4+ → order 3
    /// (higher orders stay zero until their observation budget arrives).
    ///
    /// Screens: non-finite samples refused; ticks must strictly increase
    /// (the stencil assumes uniform Δt — resample irregular streams first).
    #[allow(clippy::needless_range_loop)] // ch indexes parallel pos/vel/acc/jerk arrays
    pub fn observe_into(&mut self, pos: &[f32; D], tick: u32) -> Result<(), KinError> {
        for &x in pos {
            if !x.is_finite() {
                return Err(KinError::NonFinite);
            }
        }
        if self.n_obs > 0 && tick <= self.tick {
            return Err(KinError::NonMonotonicTick);
        }
        let dt = self.dt;
        for ch in 0..D {
            let x = pos[ch];
            if self.n_obs == 0 {
                self.pos[ch] = x;
                continue;
            }
            let vel_new = (x - self.pos[ch]) / dt;
            if self.n_obs >= 2 {
                let acc_new = (vel_new - self.vel[ch]) / dt;
                if self.n_obs >= 3 {
                    let jerk_new = (acc_new - self.acc[ch]) / dt;
                    self.jerk[ch] = jerk_new;
                }
                self.acc[ch] = acc_new;
            }
            self.vel[ch] = vel_new;
            self.pos[ch] = x;
        }
        self.tick = tick;
        self.n_obs = self.n_obs.saturating_add(1).min(MAX_LADDER_OBS);
        Ok(())
    }

    /// Effective ladder order given an observation-noise scale `eps_obs`:
    /// drops velocity / acceleration / jerk terms that are statistically
    /// insignificant against the noise propagated into their difference
    /// estimators (√2·ε for ∇¹, √6·ε for ∇², √20·ε for ∇³; 2σ screen).
    ///
    /// Order 0/1/2/3 → keeps pos / +vel / +acc / +jerk. With `eps_obs = 0`
    /// this always returns the full ladder order — the exactness fixtures
    /// rely on that.
    #[must_use]
    pub fn significant_order(&self, eps_obs: f32) -> u8 {
        let ladder = match self.n_obs {
            0 | 1 => 0,
            2 => 1,
            3 => 2,
            _ => 3,
        };
        if eps_obs <= 0.0 || !eps_obs.is_finite() {
            return ladder;
        }
        // Per-channel significant order; aggregate to the max so a single
        // genuinely-curving channel keeps its order.
        let mut order = 0u8;
        for ch in 0..D {
            let mut o = 0u8;
            if ladder >= 1 && self.vel[ch].abs() * self.dt > 2.0 * SQRT2 * eps_obs {
                o = 1;
            }
            if ladder >= 2 && self.acc[ch].abs() * self.dt * self.dt > 2.0 * SQRT6 * eps_obs {
                o = 2;
            }
            if ladder >= 3
                && self.jerk[ch].abs() * self.dt * self.dt * self.dt > 2.0 * SQRT20 * eps_obs
            {
                o = 3;
            }
            order = order.max(o);
        }
        order
    }

    /// A copy of this state with ladder orders above `max_order` zeroed (for
    /// noise-aware reduced-order extrapolation). Zero-alloc — `KinState` is
    /// `Copy`.
    #[must_use]
    pub fn capped(&self, max_order: u8) -> Self {
        let mut s = *self;
        if max_order < 1 {
            s.vel = [0.0; D];
        }
        if max_order < 2 {
            s.acc = [0.0; D];
        }
        if max_order < 3 {
            s.jerk = [0.0; D];
        }
        s
    }
}

/// √2 — i.i.d.-noise standard-deviation factor of the first backward
/// difference (variance factor 1+1 = 2).
pub const SQRT2: f32 = core::f32::consts::SQRT_2;

/// √6 — i.i.d.-noise standard-deviation factor of the second backward
/// difference (variance factor 1+4+1 = 6).
pub const SQRT6: f32 = 2.449_489_7;

/// √20 — i.i.d.-noise standard-deviation factor of the third backward
/// difference (variance factor 1+9+9+1 = 20).
pub const SQRT20: f32 = 4.472_136;

// ===== schedules =====

/// Deterministic stand-in for the paper's learned ≥3rd-order residual
/// (`f_θ`, a tanh-MLP). Every variant has a closed-form rollout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sched {
    /// `f ≡ 0` — exact on degree ≤ 2 motion (the paper's frozen-math arm).
    ZeroJerk,
    /// Constant physical jerk `j` — exact on degree ≤ 3 motion when `j` is
    /// the true jerk.
    ConstJerk { j: f32 },
    /// Use the ladder's measured 3rd-order term — exact on degree ≤ 3 with
    /// ≥ 4 clean observations.
    Measured,
    /// `j_max·tanh(λ·|vel|)` clamped correction: saturating jerk estimate
    /// evaluated **once at the anchor** (then a constant-jerk rollout). The
    /// feature is the channel's speed — the velocity input of the paper's
    /// `f_θ(ṡ, ŝ)`.
    ClampedCorrection { j_max: f32, lambda: f32 },
    /// Geometric drag: `a_{n+1} = ρ·a_n`, ρ ∈ (0,1), with the chain order
    /// `v += a·Δt; s += v·Δt; a *= ρ` (the state's `acc` is the `a` used in
    /// the last update). Closed-form rollout (see [`drag_schedule_weight`])
    /// with terminal velocity `vel + Δt·acc·ρ/(1−ρ)`
    /// ([`terminal_velocity`]).
    GeometricDrag { rho: f32 },
}

impl Sched {
    /// The effective third-order coefficient (physical jerk units) this
    /// schedule contributes for one channel, or `None` for the drag family
    /// (not a jerk rollout — it has its own closed form).
    #[inline]
    fn jerk_for(&self, vel_ch: f32, measured_jerk: f32) -> Option<f32> {
        match *self {
            Self::ZeroJerk => Some(0.0),
            Self::ConstJerk { j } => Some(j),
            Self::Measured => Some(measured_jerk),
            Self::ClampedCorrection { j_max, lambda } => {
                Some(j_max * (lambda * vel_ch.abs()).tanh())
            }
            Self::GeometricDrag { .. } => None,
        }
    }
}

/// Terminal velocity under geometric drag (chain order `v += aΔt; s += vΔt;
/// a *= ρ`, state `acc` = the `a` of the last update):
/// `ṡ_∞ = ṡ₀ + Δt·acc·ρ/(1−ρ)`.
///
/// Returns `None` for ρ outside (0,1). (The `Δt·a/(1−ρ)` form quoted in
/// Research 506 assumes the undecayed-first-step convention; this module's
/// chain decays after the update, hence the ρ factor.)
#[must_use]
pub fn terminal_velocity(vel: f32, acc: f32, dt: f32, rho: f32) -> Option<f32> {
    if !(0.0..1.0).contains(&rho) {
        return None;
    }
    Some(vel + dt * acc * rho / (1.0 - rho))
}

// ===== extrapolation =====

/// O(1) closed-form k-step kinematic rollout (Newton backward form).
///
/// ```text
/// out[ch] = pos + C(k,1)·Δt·vel + C(k+1,2)·Δt²·acc + C(k+2,3)·Δt³·j₃
/// ```
///
/// where `j₃` comes from `sched` (drag uses its own closed form).
/// Coefficients are read from the precomputed lattice (stable bits across
/// calls). Requires `k ≤ K_MAX` and `n_obs ≥ 1`.
///
/// Accumulation order is ascending in expected magnitude
/// `((pos + t1) + t2) + t3` — documented for the bit-identity discussion in
/// the module doc.
#[allow(clippy::needless_range_loop)] // ch indexes parallel state/output arrays
pub fn kinematic_extrapolate_into<const D: usize>(
    state: &KinState<D>,
    k: u32,
    sched: &Sched,
    out: &mut [f32; D],
) -> Result<(), KinError> {
    if state.n_obs == 0 {
        return Err(KinError::NotEnoughObs);
    }
    if k as usize > K_MAX {
        return Err(KinError::HorizonTooFar);
    }
    let row = lattice()[k as usize];
    let dt = state.dt;
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    for ch in 0..D {
        out[ch] = if let Sched::GeometricDrag { rho } = *sched {
            let g = drag_schedule_weight(row.b1, rho);
            state.pos[ch] + row.b1 * dt * state.vel[ch] + dt2 * state.acc[ch] * g
        } else {
            let j3 = sched.jerk_for(state.vel[ch], state.jerk[ch]).unwrap_or(0.0);
            let t1 = row.b1 * dt * state.vel[ch];
            let t2 = row.b2 * dt2 * state.acc[ch];
            let t3 = row.b3 * dt3 * j3;
            (state.pos[ch] + t1) + t2 + t3
        };
    }
    Ok(())
}

/// Noise-aware reduced-order variant: masks ladder orders above `max_order`
/// (see [`KinState::significant_order`]) then runs the full closed form.
pub fn kinematic_extrapolate_capped_into<const D: usize>(
    state: &KinState<D>,
    k: u32,
    sched: &Sched,
    max_order: u8,
    out: &mut [f32; D],
) -> Result<(), KinError> {
    kinematic_extrapolate_into(&state.capped(max_order), k, sched, out)
}

/// Geometric-drag position weight for the chain
/// `v += aΔt; s += vΔt; a *= ρ`:
///
/// ```text
/// G(k,ρ) = [kρ − kρ² − ρ² + ρ^(k+2)] / (1−ρ)²
/// ```
///
/// Verified against the chain by hand at k=1 (`G = ρ`) and k=2
/// (`G = 2ρ + ρ²`); `ŝ(k) = pos + kΔt·vel + Δt²·acc·G(k,ρ)`. Evaluated with
/// `powf` (the only non-lattice cost of the drag schedule).
#[inline]
fn drag_schedule_weight(b1: f32, rho: f32) -> f32 {
    let k = b1; // b1 == k exactly (integer-valued float)
    let omr = 1.0 - rho;
    (k * rho - k * rho * rho - rho * rho + rho.powf(k + 2.0)) / (omr * omr)
}

/// O(k) reference rollout — the difference-engine chain the closed form
/// replaces.
///
/// Per step, top-down (so each level uses the pre-update value of the level
/// above it — the Difference Engine convention):
///
/// ```text
/// d2 += d3;  d1 += d2;  s += d1
/// ```
///
/// Unrolling this chain reproduces the Newton backward binomials exactly
/// (C(k,1), C(k+1,2), C(k+2,3)); on dyadic-representable fixtures both paths
/// are exact and therefore **bit-identical** (see the module doc for the
/// honest general-data statement). Public so consumers and tests can
/// cross-check the O(1) form against the definition.
///
/// The drag arm runs the drag recurrence directly
/// (`v += a·Δt; s += v·Δt; a *= ρ` after the initial decay), mirroring the
/// paper's semi-implicit order (velocity first, then position).
#[allow(clippy::needless_range_loop)] // ch indexes parallel state/output arrays
pub fn reference_chain_extrapolate_into<const D: usize>(
    state: &KinState<D>,
    k: u32,
    sched: &Sched,
    out: &mut [f32; D],
) -> Result<(), KinError> {
    if state.n_obs == 0 {
        return Err(KinError::NotEnoughObs);
    }
    let dt = state.dt;
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    for ch in 0..D {
        if let Sched::GeometricDrag { rho } = *sched {
            // Drag chain, matching the closed form's convention: the state's
            // `acc` is the a used in the LAST update, so every FUTURE step
            // first decays then applies (`a_{n+m} = ρ^m·a_n`, m ≥ 1 —
            // see `drag_schedule_weight`).
            let mut s = state.pos[ch];
            let mut v = state.vel[ch];
            let mut a = state.acc[ch];
            for _ in 0..k {
                a *= rho;
                v += a * dt;
                s += v * dt;
            }
            out[ch] = s;
            continue;
        }
        // Differences in raw (unscaled) units.
        let mut s = state.pos[ch];
        let mut d1 = state.vel[ch] * dt;
        let mut d2 = state.acc[ch] * dt2;
        let d3 = sched.jerk_for(state.vel[ch], state.jerk[ch]).unwrap_or(0.0) * dt3;
        for _ in 0..k {
            d2 += d3;
            d1 += d2;
            s += d1;
        }
        out[ch] = s;
    }
    Ok(())
}

/// Instantaneous velocity via the central 3-point stencil:
/// `(x[m] − x[m−2]) / (2Δt)` — O(Δt²) unbiased, unlike the backward
/// difference (which lags half a step). Caller supplies the last three
/// observations (the middle one is the stencil's center — kept in the
/// signature for window clarity); the state itself only retains the
/// differenced coefficients.
#[must_use]
pub fn central_velocity(x_m: f32, _x_m1: f32, x_m2: f32, dt: f32) -> f32 {
    (x_m - x_m2) / (2.0 * dt)
}

/// Sum of squared observation weights of the k-step extrapolator at a given
/// ladder order — the i.i.d.-noise variance amplification factor.
///
/// The extrapolation is a fixed linear combination of the last ≤ 4
/// observations:
///
/// ```text
/// x̂(k) = w_m·x[m] + w_{m-1}·x[m-1] + w_{m-2}·x[m-2] + w_{m-3}·x[m-3]
/// ```
///
/// so `Var(x̂) = σ² · wss(k, order)`. This is the propagation kernel behind
/// [`perception::extrapolation_horizon`]'s bound and the UQ-floor interval.
#[must_use]
pub fn extrapolation_weight_ss(k: u32, order: u8) -> f32 {
    let k = k.min(K_MAX as u32) as usize;
    let row = lattice()[k];
    match order {
        0 => 1.0,
        1 => {
            let w0 = 1.0 + row.b1;
            let w1 = -row.b1;
            w0 * w0 + w1 * w1
        }
        2 => {
            let w0 = 1.0 + row.b1 + row.b2;
            let w1 = -(row.b1 + 2.0 * row.b2);
            let w2 = row.b2;
            w0 * w0 + w1 * w1 + w2 * w2
        }
        _ => {
            let w0 = 1.0 + row.b1 + row.b2 + row.b3;
            let w1 = -(row.b1 + 2.0 * row.b2 + 3.0 * row.b3);
            let w2 = row.b2 + 3.0 * row.b3;
            let w3 = -row.b3;
            w0 * w0 + w1 * w1 + w2 * w2 + w3 * w3
        }
    }
}

#[cfg(test)]
pub(crate) fn lattice_rows_for_test() -> &'static [LatticeRow] {
    lattice()
}
