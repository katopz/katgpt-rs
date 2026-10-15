//! Phase-carrying belief kernel — Issue 939 POC (Research 620, arXiv:2610.13341).
//!
//! The reconstruction-lane belief kernels ([`crate::reconstruction`]'s leaky
//! integrators, `evolve_belief` / `evolve_belief_additive`) carry MAGNITUDE
//! only: a belief knows *how much*, never *when* or *with what period*. This
//! module adds the two POC primitives from Issue 939 (both opt-in, no
//! consumer wiring — promotion is defer-until-pull):
//!
//! 1. **[`ModularPhaseCounter`]** — the exact undamped phase code (Issue 939
//!    P1, the deliverable): a `u32` count mod `m` + a precomputed
//!    `cos(2π·c/m)` lookup. Bit-deterministic by construction, zero drift
//!    over unbounded streams, cheaper than per-read sincos. The paper's E.1
//!    shows gradient descent converges to exactly this shape as a modular
//!    counter; we CONSTRUCT it instead of training it.
//! 2. **[`DampedPhaseBlock`]** — the non-trivial primitive (P1/P4): a 2×2
//!    underdamped κ-block `z ← ρ·R(θ)·z + u` per belief channel. Phase and
//!    magnitude fade together (`ρⁿ·R(nθ)` in closed form), so the channel
//!    carries *when*-memory with a confidence envelope, and the paper's C.3
//!    steady state `‖u‖/|1−ρe^{iθ}|` bounds its own saturation — exposed as
//!    [`DampedPhaseBlock::steady_state_bound`] /
//!    [`DampedPhaseBlock::write_budget_for`] so a consumer sets per-channel
//!    write budgets from the θ-dependence (ρ→1: `2·sin(θ/2)`; small-θ needs
//!    the leaky kernel's own budget).
//!
//! House laws: sigmoid readouts via `fast_sigmoid` (never softmax);
//! zero-allocation hot path (the LUT is built once at construction); fixed
//! `[f32; 2]` per-channel block; single-rounding `mul_add` folds (the Issue
//! 940 kernel discipline). This is LOCAL state — nothing here crosses the
//! sync boundary, so the raw/latent bridge rule is not engaged; if a consumer
//! ever syncs the readout, the SCALAR (already clamped) is the crossing
//! surface, never the `[f32; 2]` block.
//!
//! Substrate note (Issue 939 signal-diff): katgpt-core's
//! `phase_rotation_coupling` (Plan 322) is a phase-modulated MIX between two
//! named vectors (`out = cosα·a + sinα·b`) — a gate, not a state integrator.
//! Research 169 (LinOSS) is free oscillation as architecture; Research 371 is
//! population-level DMFT. Neither owns a per-channel constructed phase code
//! on the belief substrate — that is the only novelty claim this POC makes
//! (an engineering claim, never a Super-GOAT).

/// Errors from the phase-belief constructors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseBeliefError {
    /// `ModularPhaseCounter` needs a modulus in `2..=MAX_MODULUS` (m=1 has no
    /// phase; the LUT would be a single point).
    ModulusOutOfRange,
    /// `DampedPhaseBlock` needs `0 < ρ ≤ 1` (ρ=0 erases phase every step;
    /// ρ>1 diverges).
    RhoOutOfRange,
    /// `DampedPhaseBlock` needs a finite `θ` (NaN/∞ would poison the
    /// precomputed rotation).
    ThetaNotFinite,
}

impl core::fmt::Display for PhaseBeliefError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ModulusOutOfRange => write!(f, "modulus must be in 2..={MAX_MODULUS}"),
            Self::RhoOutOfRange => write!(f, "rho must be in (0, 1]"),
            Self::ThetaNotFinite => write!(f, "theta must be finite"),
        }
    }
}

impl std::error::Error for PhaseBeliefError {}

/// Largest supported modulus: the `cos(2π·c/m)` LUT stays tiny and cache-hot.
pub const MAX_MODULUS: u32 = 256;

/// Exact undamped modular phase code (Issue 939 P1 deliverable).
///
/// Counts events mod `m` in an integer and answers `cos(2π·c/m)` from a
/// precomputed LUT — bit-deterministic across platforms and stream orders by
/// construction (no floating-point accumulation anywhere on the count path).
/// The damped [`DampedPhaseBlock`] is the fading-memory counterpart; when a
/// consumer needs the EXACT count mod m (parity, mod-3 cadence), this is the
/// form to read.
#[derive(Debug, Clone, PartialEq)]
pub struct ModularPhaseCounter {
    m: u32,
    count: u32,
    /// `cos_lut[c] = cos(2π·c/m)` for `c in 0..m` — built once, read-only.
    cos_lut: Box<[f32]>,
    /// `sin_lut[c] = sin(2π·c/m)` — the quadrature partner. BOTH components
    /// are the phase code: `cos` alone is injective on ℤ_m only for m=2
    /// (at m=3, cos(2π/3) = cos(4π/3) = −1/2); the paper's readout is the
    /// full phase direction.
    sin_lut: Box<[f32]>,
}

impl ModularPhaseCounter {
    /// Builds the counter for modulus `m` (2..=[`MAX_MODULUS`]).
    pub fn new(m: u32) -> Result<Self, PhaseBeliefError> {
        if !(2..=MAX_MODULUS).contains(&m) {
            return Err(PhaseBeliefError::ModulusOutOfRange);
        }
        let step = core::f32::consts::TAU / m as f32;
        let mut cos_lut = Vec::with_capacity(m as usize);
        let mut sin_lut = Vec::with_capacity(m as usize);
        for c in 0..m {
            let (s, csn) = (c as f32 * step).sin_cos();
            cos_lut.push(csn);
            sin_lut.push(s);
        }
        Ok(Self {
            m,
            count: 0,
            cos_lut: cos_lut.into_boxed_slice(),
            sin_lut: sin_lut.into_boxed_slice(),
        })
    }

    /// Records one event: `count = (count + 1) mod m`.
    pub fn observe(&mut self) {
        self.count += 1;
        if self.count >= self.m {
            self.count -= self.m;
        }
    }

    /// Records `n` events at once (streaming bulk form; exact for any `n`).
    pub fn observe_n(&mut self, n: u32) {
        // u32 add can overflow only past 2³² events; the mod-m reduce keeps it
        // exact without widening.
        self.count = (self.count + n % self.m) % self.m;
    }

    /// The event count mod `m`.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// The modulus.
    pub fn modulus(&self) -> u32 {
        self.m
    }

    /// `cos(2π·count/m)` — the cosine half of the phase code (LUT, O(1)).
    ///
    /// +1 at count 0, sweeping the unit circle as events accrue. ⚠ Injective
    /// on ℤ_m ONLY for m=2 (parity); m≥3 readers take [`Self::readout_pair`]
    /// — cos(2π/3) = cos(4π/3) is the m=3 degeneracy.
    pub fn readout(&self) -> f32 {
        self.cos_lut[self.count as usize]
    }

    /// `[cos φ, sin φ]` — the FULL phase code (both quadratures), injective
    /// on ℤ_m for every m. This is the paper's actual readout surface.
    pub fn readout_pair(&self) -> [f32; 2] {
        [self.cos_lut[self.count as usize], self.sin_lut[self.count as usize]]
    }

    /// The drive-shaped scalar readout: `fast_sigmoid(gain · cos φ)` —
    /// smooth, house-compliant (sigmoid, never softmax), ready for FSM/drive
    /// consumption. Fully discriminative for m=2 (parity drives); m≥3
    /// consumers that must separate ALL residues read [`Self::readout_pair`]
    /// (or two of these, one per quadrature).
    pub fn drive_readout(&self, gain: f32) -> f32 {
        katgpt_types::simd::fast_sigmoid(gain * self.readout())
    }
}

/// Damped 2×2 phase block (Issue 939 P1 non-trivial primitive + P4 budget).
///
/// Per-channel state `z ∈ [f32; 2]` evolved by the underdamped κ-block
///
/// ```text
/// z_{t+1} = ρ·R(θ)·z_t + u_t          (R = 2D rotation, 0 < ρ ≤ 1)
/// ```
///
/// With no writes the state decays EXACTLY as `ρⁿ·R(nθ)·z_0` (up to f32
/// rounding): the phase `nθ mod 2π` carries *when*, the magnitude `ρⁿ‖z_0‖`
/// carries confidence — phase and magnitude fade together, which is what no
/// leaky scalar kernel can express. With constant writes the paper's C.3
/// steady state bounds the accumulation:
///
/// ```text
/// ‖z_∞‖ ≤ ‖u‖/|1−ρe^{iθ}| = ‖u‖/√(1−2ρcosθ+ρ²)
/// ```
///
/// — the per-channel write budget for a target bound `ε` is
/// `ε·|1−ρe^{iθ}|`. θ-dependence (Issue 939 P4 reviewer note): at ρ→1 the
/// denominator → `2·sin(θ/2)` (large-θ channels get the rotation benefit
/// cheap); SMALL-θ channels need ≈ the leaky kernel's own budget
/// (`√((1−ρ)²+θ²)`).
///
/// Zero-allocation: `step` borrows only; the rotation is two precomputed
/// constants (cos θ, sin θ) folded with single-rounding `mul_add`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DampedPhaseBlock {
    z: [f32; 2],
    rho: f32,
    cos_theta: f32,
    sin_theta: f32,
    /// `|1−ρe^{iθ}|` — the write-budget scale factor (P4), precomputed.
    budget_scale: f32,
}

impl DampedPhaseBlock {
    /// Builds the block for damping radius `ρ` and per-step rotation `θ`.
    pub fn new(rho: f32, theta: f32) -> Result<Self, PhaseBeliefError> {
        // rho ≤ 0, > 1, or NaN all refuse (NaN fails both comparisons
        // explicitly rather than via a negated partial-ord operator).
        let rho_ok = rho > 0.0 && rho <= 1.0;
        if !rho_ok {
            return Err(PhaseBeliefError::RhoOutOfRange);
        }
        if !theta.is_finite() {
            return Err(PhaseBeliefError::ThetaNotFinite);
        }
        let (sin_theta, cos_theta) = theta.sin_cos();
        // |1−ρe^{iθ}|² = (1−ρcosθ)² + (ρsinθ)² — both terms non-negative,
        // single sqrt, no catastrophic cancellation for the ρ→1 small-θ edge
        // (1−ρcosθ stays well-conditioned).
        let dr = 1.0 - rho * cos_theta;
        let budget_scale = (dr * dr + (rho * sin_theta) * (rho * sin_theta)).sqrt();
        Ok(Self {
            z: [0.0; 2],
            rho,
            cos_theta,
            sin_theta,
            budget_scale,
        })
    }

    /// One evolution step: `z ← ρ·R(θ)·z + u`.
    ///
    /// Write `u` may be `[0.0, 0.0]` (pure decay — the free run). Ordered
    /// per-component `mul_add` folds (Issue 940 kernel discipline).
    pub fn step(&mut self, write: [f32; 2]) {
        let [x, y] = self.z;
        let (c, s, r) = (self.cos_theta, self.sin_theta, self.rho);
        // R(θ)·z = (c·x − s·y, s·x + c·y), scaled by ρ, fused with the write.
        self.z[0] = (r * c).mul_add(x, -(r * s) * y + write[0]);
        self.z[1] = (r * s).mul_add(x, (r * c) * y + write[1]);
    }

    /// The raw 2D state (by value — `[f32; 2]` is `Copy`).
    pub fn state(&self) -> [f32; 2] {
        self.z
    }

    /// `‖z‖` — the confidence envelope (decays as `ρⁿ` on a free run).
    pub fn magnitude(&self) -> f32 {
        self.z[0].mul_add(self.z[0], self.z[1] * self.z[1]).sqrt()
    }

    /// `cos φ` where `φ` is the state's phase — O(1), no atan2.
    ///
    /// Defined as 1.0 for the zero state (phase undefined; the neutral
    /// readout matches `count 0`).
    pub fn phase_cos(&self) -> f32 {
        let m = self.magnitude();
        if m <= 0.0 {
            1.0
        } else {
            self.z[0] / m
        }
    }

    /// `sin φ` — the phase's quadrature partner (same zero-state convention).
    pub fn phase_sin(&self) -> f32 {
        let m = self.magnitude();
        if m <= 0.0 {
            0.0
        } else {
            self.z[1] / m
        }
    }

    /// Drive-shaped readout `fast_sigmoid(gain · cos φ)`.
    pub fn drive_readout(&self, gain: f32) -> f32 {
        katgpt_types::simd::fast_sigmoid(gain * self.phase_cos())
    }

    /// Closed-form free-run magnitude after `n` decay steps: `ρⁿ·‖z‖`.
    ///
    /// The exact predictor a consumer uses to schedule re-observation
    /// (read confidence without stepping).
    pub fn magnitude_after(&self, n: u32) -> f32 {
        self.magnitude() * self.rho.powi(n as i32)
    }

    /// Closed-form free-run phase cosine after `n` decay steps: the phase of
    /// `R(nθ)·z` — `cos(φ + nθ)`.
    pub fn phase_cos_after(&self, n: u32) -> f32 {
        // f32::sin_cos returns (sin, cos) — named in that order here.
        let (sin_nt, cos_nt) = (n as f32 * self.rotation_per_step()).sin_cos();
        let (cos_phi, sin_phi) = (self.phase_cos(), self.phase_sin());
        cos_phi.mul_add(cos_nt, -(sin_phi * sin_nt))
    }

    /// Rotation per step in radians (θ as constructed).
    pub fn rotation_per_step(&self) -> f32 {
        self.sin_theta.atan2(self.cos_theta)
    }

    /// The paper's C.3 steady-state bound for a constant write of magnitude
    /// `write_norm`: `‖z_∞‖ ≤ write_norm / |1−ρe^{iθ}|`.
    pub fn steady_state_bound(&self, write_norm: f32) -> f32 {
        write_norm / self.budget_scale
    }

    /// The P4 design constraint: the max write magnitude for a target
    /// steady-state bound `ε` — `write_budget = ε·|1−ρe^{iθ}|`.
    pub fn write_budget_for(&self, eps: f32) -> f32 {
        eps * self.budget_scale
    }
}
