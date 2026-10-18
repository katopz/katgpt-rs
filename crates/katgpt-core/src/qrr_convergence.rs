//! QRR quotient-convergence extraction — Issue 945 P1/P3/P6 (Research 623,
//! arXiv:2610.18174, Quotient Recurrent Reasoner). P2's halter verdict lives
//! in [`crate::gain_cost_halt`] and consumes this module's estimate types;
//! P6 (the report-only offline attribution instrument: Sylvester solve,
//! complement energy split, α-intervention replay) sits at the bottom of
//! this module.
//!
//! The paper's convergence-law family needs NO oracle and NO training: the
//! dominant mode `(r, ω)` of an iterative loop is recoverable from the
//! delta sequence alone by a 2nd-order autoregressive fit —
//! `δ_{t+2} = a·δ_{t+1} + b·δ_t` — whose companion roots of `z² − a·z − b`
//! ARE the linearized Jacobian's eigenvalues (Theorem 3.2 / Lemma 3.1):
//! complex pair `r·e^{±iω}` with `r = √(−b)`, `ω = acos(a/2r)`; real
//! dominant root `λ` (negative ⇒ alternating flip `S = −r`). From `r`:
//! settling budget `τ_δ = log(1/δ)/(−log r)`, invertible (`budget B` buys
//! tolerance `r^B`), and the `r = 1` bounded-non-convergence certificate
//! (two-signal deployable form: unit modulus AND complement stagnation —
//! one alone never acts, the tension law).
//!
//! # Numerics laws pinned here
//!
//! - **Ridge identifiability repair:** an EXACT single-mode delta sequence
//!   (pure geometric / pure flip) makes the AR(2) normal equations
//!   rank-deficient (both regressors are proportional). A tiny relative
//!   Tikhonov ridge selects the minimum-norm solution of that solution
//!   line, whose dominant companion root still recovers the true mode
//!   exactly (pinned by test: λ = −0.7, 0.5, 1.0). Never remove the ridge.
//! - **Blind-channel guard:** the halter feeds the SIGNED one-lag
//!   projection `s_t = gain_t · cosθ_t`. Near a 90° rotation that
//!   projection is rounding noise on a near-zero dot — the channel is
//!   BLIND there, and a verdict fitted from it would be fabrication. The
//!   fitter refuses (returns `None`) when the window's variance sits at
//!   rounding scale of the current step (`‖s‖² ≤ (gain·BLIND_FRAC)²`),
//!   which is exactly the Issue-720 trigger-happy cos≈0 edge — the
//!   incumbent verdict stands there.
//! - **Stillness guard:** an exactly-zero delta window (`None`) — the
//!   aperiodic/stuck arm; a linear ramp honestly reports its unit root
//!   (`r ≈ 1`, which is the CORRECT convergence statement for a ramp).
//!
//! Zero allocation throughout: fixed `[f32; 3·MODE_WINDOW]` ring, O(1)
//! running sums, no `Vec`, no RNG (gates use a deterministic LCG).
//!
//! Issue 945 P1/P3/P6 (Research 623, arXiv:2610.18174 QRR) — the
//! quotient-convergence extraction: P1 per-channel delta-sequence mode fitter
//! (companion roots → (r, ω) with no Jacobian oracle), P3 settling-budget
//! predictor + two-signal r=1 certificate, P6 the report-only offline
//! attribution instrument (Sylvester solve + energy split + α-intervention
//! replay). P2's halter verdict (the modulus
//! channel over the oscillation detector) lives in gain_cost_halt behind the
//! SAME feature and arms via `with_mode_verdict()`. Opt-in POC — no promotion
//! claim; P4 rides `phase_belief` defer-until-pull (Issue 939), P6 is
//! report-only offline (issue file carries the split).

/// Sliding-window triple count for the AR(2) fit. 16 triples ≈ 19 steps of
/// history (2 priming deltas + 16 fitted) — long enough to average noise,
/// short enough to track mode changes within a patience-scale loop budget.
pub const MODE_WINDOW: usize = 16;

/// Minimum fitted triples before [`ModeFitter::estimate`] trusts the sums
/// (2 unknowns, 6 overdetermined equations). Below this the verdict is
/// `None` and the halter falls back to the incumbent behavior — document
/// that arming [`crate::gain_cost_halt::GainCostLoopHalter::with_mode_verdict`]
/// with `oscillation_patience < MIN_FIT_SAMPLES` still false-halts the first
/// reversals (the discriminant is a patience-scale refinement).
pub const MIN_FIT_SAMPLES: usize = 6;

/// Settle band: a fitted modulus below `1 − MODE_SETTLE_EPS` is a SETTLING
/// oscillation (converging THROUGH the flip/spiral — never halt-and-damp);
/// at or above it, a limit cycle (bounded, never converges).
pub const MODE_SETTLE_EPS: f32 = 0.01;

/// Certificate band: the `r = 1` certificate's first signal requires the
/// fitted modulus inside `[1 − NEAR_ONE_EPS, 1 + NEAR_ONE_EPS]`.
pub const NEAR_ONE_EPS: f32 = 0.01;

/// Relative blind-channel floor: when the signed-projection variance drops
/// below `(last_gain × BLIND_FRAC)²` the channel carries no mode information
/// (a near-90° rotation's projection is rounding noise) — refuse the verdict.
const BLIND_FRAC: f32 = 1e-4;

/// Relative Tikhonov ridge on the normal-equation diagonal — the
/// identifiability repair for single-mode (rank-deficient) windows.
const RIDGE_FRAC: f64 = 1e-6;

/// Which companion-root family the fit recovered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ModeKind {
    /// Complex pair `r·e^{±iω}` — a spiral/spin mode (`ω` is carried).
    ComplexPair,
    /// Real negative dominant root `−r` — the paper's FPRM alternating flip.
    RealFlip,
    /// Real positive dominant root — non-oscillatory mode (the signed
    /// one-lag projection channel's natural geometry: `s_t = gain_t·cosθ_t`
    /// is geometric with ratio = the state modulus).
    RealPositive,
}

/// The dominant recovered mode of a delta sequence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModeEstimate {
    /// Which root family the fit recovered.
    pub kind: ModeKind,
    /// Dominant modulus — the paper's rate-limiting `r` (Theorem 3.2).
    pub r: f32,
    /// Angular rate for [`ModeKind::ComplexPair`] (radians/step), else `None`.
    pub omega: Option<f32>,
}

/// The P2 verdict discriminant over the oscillation channel
/// ([`crate::gain_cost_halt`]): the INCUMBENT detector sees only the angle
/// (`cosθ < 0` = reversal); the fitted MODULUS separates a settling
/// oscillation from a limit cycle under the same reversal evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum OscillationVerdict {
    /// `r < 1 − MODE_SETTLE_EPS` — converging THROUGH the oscillation.
    /// NEVER a halt-and-damp (the paper's tension law, App. A.4.4:
    /// suppressing a settling alternating mode DEGRADES the computation).
    OscillatingButSettling,
    /// `r ≥ 1 − MODE_SETTLE_EPS` — a true limit cycle; the incumbent
    /// oscillation halt stands (maintained true-halt latency).
    LimitCycle,
}

impl OscillationVerdict {
    /// Classify a fitted mode against the settle band.
    #[inline]
    pub fn from_mode(mode: &ModeEstimate) -> Self {
        if mode.r < 1.0 - MODE_SETTLE_EPS {
            Self::OscillatingButSettling
        } else {
            Self::LimitCycle
        }
    }
}

/// P1 — per-channel sliding-window AR(2) mode fitter (Issue 945).
///
/// Feed the SIGNED per-loop scalar each step ([`Self::observe_signed`] for
/// the halter's one-lag projection, or a raw delta channel); the fitter
/// maintains O(1) running sums over a fixed [`MODE_WINDOW`] ring of
/// `(δ_{t+1}, δ_t, δ_{t+2})` triples and solves the 3-coefficient normal
/// equations on demand. One instance per channel; the halter wires one
/// (the scalar projection) — multi-channel dominant-mode selection is a
/// consumer-side composition over per-channel instances.
///
/// Kill switch (the P2 demotion is decision-only): env
/// `KATGPT_QRR_OSC_VERDICT=0` — see
/// [`oscillation_verdict_kill_switch`].
#[derive(Clone, Debug)]
pub struct ModeFitter {
    /// Ring of `(x1, x2, y) = (δ_{t+1}, δ_t, δ_{t+2})` triples.
    window: [f32; 3 * MODE_WINDOW],
    /// Next write slot (triple index).
    head: usize,
    /// Triples currently in the window (≤ [`MODE_WINDOW`]).
    filled: usize,
    /// `δ_{t−1}` (the most recent delta before the current one).
    prev1: f32,
    /// `δ_{t−2}`.
    prev2: f32,
    /// Total deltas observed this episode (primes the first two).
    seen: usize,
    /// Running window sums: `Σ x1²`, `Σ x2²`, `Σ x1·x2`, `Σ x1·y`, `Σ x2·y`.
    /// f64 ACCUMULATION over f32 deltas — measured necessity, not style: the
    /// sliding `+=`/`-=` update over a decaying window cancels catastrophically
    /// in f32 (each eviction subtracts a large early term from a sum that has
    /// since shrinks by decades; the residual error measured ~3–14% of the
    /// sums and put the recovered root at ρ² = 0.83 instead of ρ = 0.8),
    /// while f64 sums reproduce the exact-arithmetic fit to 1e-15.
    s11: f64,
    s22: f64,
    s12: f64,
    s1y: f64,
    s2y: f64,
    /// `|gain|` of the most recent fed sample (the blind-channel scale).
    last_gain: f32,
    /// Consecutive [`Self::estimate`] results with `r` inside the
    /// [`NEAR_ONE_EPS`] band (the certificate's mode-half signal).
    near_one_streak: u32,
}

impl ModeFitter {
    /// Cold fitter — no deltas, no sums.
    pub fn new() -> Self {
        Self {
            window: [0.0; 3 * MODE_WINDOW],
            head: 0,
            filled: 0,
            prev1: 0.0,
            prev2: 0.0,
            seen: 0,
            s11: 0.0,
            s22: 0.0,
            s12: 0.0,
            s1y: 0.0,
            s2y: 0.0,
            last_gain: 0.0,
            near_one_streak: 0,
        }
    }

    /// Feed one loop's SIGNED projection: `gain × cosθ` (the halter's
    /// channel — `gain` is the per-loop step magnitude, `cosθ` the angular
    /// alignment of the last two update directions). Non-finite inputs are
    /// DROPPED (a corrupt signal must not poison the sums; the verdict
    /// stays the last good one, or `None` → incumbent).
    #[inline]
    pub fn observe_signed(&mut self, gain: f32, cos_theta: f32) {
        self.observe(gain * cos_theta);
        if gain.is_finite() {
            self.last_gain = gain.abs();
        }
    }

    /// Feed one raw signed delta. Non-finite values are dropped.
    #[inline]
    pub fn observe(&mut self, delta: f32) {
        if !delta.is_finite() {
            return;
        }
        if self.seen >= 2 {
            let (x1, x2, y) = (self.prev1, self.prev2, delta);
            if self.filled == MODE_WINDOW {
                // Evict the oldest triple's contributions, then overwrite.
                let base = self.head * 3;
                let (ox1, ox2, oy) = (
                    self.window[base],
                    self.window[base + 1],
                    self.window[base + 2],
                );
                self.s11 -= (ox1 * ox1) as f64;
                self.s22 -= (ox2 * ox2) as f64;
                self.s12 -= (ox1 * ox2) as f64;
                self.s1y -= (ox1 * oy) as f64;
                self.s2y -= (ox2 * oy) as f64;
            } else {
                self.filled += 1;
            }
            let base = self.head * 3;
            self.window[base] = x1;
            self.window[base + 1] = x2;
            self.window[base + 2] = y;
            self.s11 += (x1 * x1) as f64;
            self.s22 += (x2 * x2) as f64;
            self.s12 += (x1 * x2) as f64;
            self.s1y += (x1 * y) as f64;
            self.s2y += (x2 * y) as f64;
            self.head = (self.head + 1) % MODE_WINDOW;
        }
        self.prev2 = self.prev1;
        self.prev1 = delta;
        self.seen += 1;
    }

    /// Triples currently fitted over (diagnostic; ≥ [`MIN_FIT_SAMPLES`]
    /// before [`Self::estimate`] returns a verdict).
    #[inline]
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// Consecutive near-one estimates (the certificate's mode half).
    #[inline]
    pub fn near_one_streak(&self) -> u32 {
        self.near_one_streak
    }

    /// Fit the window and return the dominant mode, or `None` when the
    /// verdict is not trustworthy: cold window, exact stillness, or a
    /// blind channel (variance at rounding scale of the current step —
    /// the near-90° projection). Advances the near-one streak ONLY on a
    /// returned estimate, so a `None` does not preserve a stale streak.
    pub fn estimate(&mut self) -> Option<ModeEstimate> {
        let Some(estimate) = self.fit_sums() else {
            self.near_one_streak = 0;
            return None;
        };
        if (1.0 - NEAR_ONE_EPS..=1.0 + NEAR_ONE_EPS).contains(&estimate.r) {
            self.near_one_streak = self.near_one_streak.saturating_add(1);
        } else {
            self.near_one_streak = 0;
        }
        Some(estimate)
    }

    /// Pure-sums fit (no streak side effects) with the guard chain:
    /// cold → stillness → blind → rank-deficient → solve → roots.
    fn fit_sums(&self) -> Option<ModeEstimate> {
        if self.filled < MIN_FIT_SAMPLES {
            return None;
        }
        // Stillness: an exactly-zero window has no mode (aperiodic/stuck).
        if self.s11 == 0.0 && self.s22 == 0.0 {
            return None;
        }
        // Blind channel: the projection carries no mode information when its
        // variance is rounding noise on the current step scale (near-90°).
        let blind_floor = (self.last_gain * BLIND_FRAC) as f64;
        if self.s11 + self.s22 <= blind_floor * blind_floor {
            return None;
        }
        // Relative Tikhonov ridge — the single-mode identifiability repair.
        // Cauchy–Schwarz (`s12² ≤ s11·s22`) keeps the ridged determinant
        // strictly positive whenever the window is non-still.
        let ridge = RIDGE_FRAC * (self.s11 + self.s22);
        let det = (self.s11 + ridge) * (self.s22 + ridge) - self.s12 * self.s12;
        if det <= 0.0 {
            return None;
        }
        // Inverse of [[s11+λ, s12], [s12, s22+λ]] — a's cofactor is the
        // s22-side diagonal (swapping them recovers the ρ² shadow, not ρ).
        let a = ((self.s22 + ridge) * self.s1y - self.s12 * self.s2y) / det;
        let b = ((self.s11 + ridge) * self.s2y - self.s12 * self.s1y) / det;
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        // Companion roots of z² − a·z − b = 0.
        let disc = a * a + 4.0 * b;
        if disc < 0.0 {
            // Complex pair: r² = −b (product of roots), 2r·cosω = a.
            let r = ((-b).sqrt() as f32).abs();
            if !(r.is_finite() && r > 0.0) {
                return None;
            }
            let omega = (a / (2.0 * r as f64)).clamp(-1.0, 1.0).acos() as f32;
            return Some(ModeEstimate {
                kind: ModeKind::ComplexPair,
                r,
                omega: Some(omega),
            });
        }
        let sq = disc.sqrt();
        let l1 = ((a + sq) * 0.5) as f32;
        let l2 = ((a - sq) * 0.5) as f32;
        let dominant = if l1.abs() >= l2.abs() { l1 } else { l2 };
        if !dominant.is_finite() {
            return None;
        }
        let kind = if dominant < 0.0 {
            ModeKind::RealFlip
        } else {
            ModeKind::RealPositive
        };
        Some(ModeEstimate {
            kind,
            r: dominant.abs(),
            omega: None,
        })
    }
}

impl Default for ModeFitter {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// P3 — settling-budget predictor: steps for a mode of modulus `r` to decay
/// to relative tolerance `δ` — `τ_δ = log(1/δ)/(−log r)` (arXiv:2610.18174
/// Thm 3.2). `None` unless `0 < r < 1` and `0 < δ < 1` (`r ≥ 1` never
/// converges — that is the certificate's finding, not an error).
#[inline]
pub fn settling_steps(r: f32, delta: f32) -> Option<f32> {
    if !(r.is_finite() && r > 0.0 && r < 1.0 && delta.is_finite() && delta > 0.0 && delta < 1.0) {
        return None;
    }
    // ln(1/δ)/(−ln r) — both logs negative, the ratio positive.
    Some(delta.ln() / r.ln())
}

/// P3 — budget inverse: the tolerance a loop budget `B` buys against mode
/// modulus `r` — `δ_B = r^B` (invertibility of the budget law). `None`
/// unless `0 < r < 1` and `B ≥ 0`.
#[inline]
pub fn tolerance_for_budget(r: f32, budget: f32) -> Option<f32> {
    if !(r.is_finite() && r > 0.0 && r < 1.0 && budget.is_finite() && budget >= 0.0) {
        return None;
    }
    Some(r.powf(budget))
}

/// P3 — the `r = 1` bounded-non-convergence certificate, TWO-SIGNAL
/// deployable form: unit modulus AND complement stagnation. One signal
/// alone NEVER acts (the tension law): a settling spiral has `r < 1`
/// (not near-one even though its complement is "stagnating" in the raw
/// residual), and a rotation with a growing radius has unit-ish modulus
/// but its complement is PROGRESSING.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct R1Certificate {
    /// The mode half: fitted modulus inside `[1 − NEAR_ONE_EPS, 1 + NEAR_ONE_EPS]`
    /// for a persistence streak ([`ModeFitter::near_one_streak`]).
    pub near_one: bool,
    /// The complement half: consumer-owned stagnation evidence — in the
    /// halter's scalar world, the per-loop step magnitude no longer
    /// decaying (the natural source is the existing concavity noise floor,
    /// `CONCAVITY_NOISE_FRAC × first_gain`).
    pub complement_stagnant: bool,
}

impl R1Certificate {
    /// Both signals — never fires on one alone.
    #[inline]
    pub fn fires(&self) -> bool {
        self.near_one && self.complement_stagnant
    }
}

/// The P2 kill switch: env `KATGPT_QRR_OSC_VERDICT=0` (the exact literal)
/// demotes the mode verdict to the incumbent oscillation behavior —
/// byte-identical decisions, while the fitter stays armed for observation
/// ([`OscillationVerdict`] accessors stay live). Read ONCE at arming
/// ([`crate::gain_cost_halt::GainCostLoopHalter::with_mode_verdict`]),
/// never on the hot path.
#[inline]
pub fn oscillation_verdict_kill_switch() -> bool {
    std::env::var("KATGPT_QRR_OSC_VERDICT").is_ok_and(|v| v == "0")
}

// ─── Issue 945 P6 — Sylvester attribution + α-intervention replay ───────────
// Report-only, OFFLINE instrument (Research 623 §1: Theorem 3.2 + the α
// protocol). Consumers are FUTURE work (the bench_011/013 feature
// extractors, the Issue 717 damping decisions) — nothing in this section is
// wired into any hot path. Unlike the fitter above it allocates (`Vec`) and
// computes in f64 — the solve AND every magnitude it feeds, for two measured
// reasons: the α-identity's 1e-9 bar is unreachable through an f32-quantized
// `P` (quantization alone injects Θ(1e-7) into the telescoping identity
// `Σ A^{k−1−i}(PS−AP−B_c)S^i` — three orders over the bar), and an f32 norm
// underflows at long horizons (ρ(A) = 0.5 at t = 200 ≈ 6e-61) and would fake
// an exact zero share. The same measured-necessity class as the f64 window
// sums in [`ModeFitter`]; the captured system itself still enters as f32
// ([`SylvesterBlocks`]).

/// P6 — the constructed/known linear system around a captured stall point.
/// In the ordered coordinates `(u, e)` — `u ∈ R²` the oscillatory block,
/// `e ∈ R^k` the complement (`d = 2 + k`) — the coupled system reads
/// `u_{t+1} = S u_t`, `e_{t+1} = B_c u_t + A e_t`, with `B_c` the complement
/// rows of `b` (rows `2..`; the `U` rows must be exactly zero — coupling
/// into the invariant subspace would break the decomposition). The exact
/// Sylvester decomposition (`P S − A P = B_c`, unique iff
/// `σ(A) ∩ σ(S) = ∅`) splits the complement response into
/// `e_t = A^t(e₀ − P u₀) + P S^t u₀`: intrinsic transient vs mode-coupled
/// stall carrier.
#[derive(Clone, Debug, PartialEq)]
pub struct SylvesterBlocks {
    /// The 2×2 oscillatory block: the complex pair `r·e^{±iω}` as
    /// `[[r·cosω, −r·sinω], [r·sinω, r·cosω]]`, or the real flip
    /// `[[-r, 0], [0, −r]]`.
    pub s: [[f32; 2]; 2],
    /// Complement dynamics, `k×k` — stable and spectrally disjoint from
    /// `s` (disjointness is the uniqueness precondition;
    /// [`solve_sylvester`] refuses an overlapping system).
    pub a: Vec<Vec<f32>>,
    /// Coupling, `d×2` with `d = k + 2`; ONLY the complement rows (`2..`)
    /// may be nonzero (enforced by [`solve_sylvester`]).
    pub b: Vec<Vec<f32>>,
    /// The recorded 2-column oscillatory basis, `[I₂; 0]` embedded in
    /// `R^d` — how fitted oscillatory coordinates map back into full-state
    /// space. Carried for the future extractor consumers; the functions in
    /// this section work in the `(u, e)` coordinates directly and do not
    /// read it.
    pub u: Vec<Vec<f32>>,
}

/// P6 — solve `P S − A P = B_c` for `P` (`k×2`) via the Kronecker form
/// `(Sᵀ ⊗ I_k − I₂ ⊗ A) vec(P) = vec(B_c)` — a `2k×2k` dense system
/// (column-major `vec`), one Gaussian elimination with partial pivoting.
/// The solve is delivered in f64 DELIBERATELY (not this file's usual f32
/// boundary): an f32-quantized `P` carries Θ(1e-7) into the α-identity's
/// telescoping residual, three orders over its 1e-9 bar, and nothing here
/// syncs `P` (no boundary to protect — the P5 fp32 law is about rotation
/// carry, and no rotation is carried by a linear solve).
///
/// `None` when the system is singular — exactly the spectral overlap
/// `σ(A) ∩ σ(S) ≠ ∅`, since the Kronecker spectrum is `{λᵢ(S) − μⱼ(A)}`.
/// That overlap is the paper's uniqueness precondition, so a refusal here
/// is the FINDING, not an error. Also `None` on a shape violation.
pub fn solve_sylvester(blocks: &SylvesterBlocks) -> Option<Vec<Vec<f64>>> {
    let k = blocks.a.len();
    if k == 0 || blocks.a.iter().any(|row| row.len() != k) {
        return None;
    }
    if blocks.b.len() != k + 2 || blocks.b.iter().any(|row| row.len() != 2) {
        return None;
    }
    if blocks.b[0].iter().any(|&x| x != 0.0) || blocks.b[1].iter().any(|&x| x != 0.0) {
        return None; // the U rows must not couple (block-triangular premise)
    }
    let n = 2 * k;
    // M[(jc·k + i)][(lc·k + i')] = Sᵀ[jc][lc]·δ(i,i') − A[i][i']·δ(jc,lc).
    let mut m = vec![vec![0.0_f64; n]; n];
    for jc in 0..2 {
        for lc in 0..2 {
            let st = blocks.s[lc][jc] as f64;
            for i in 0..k {
                m[jc * k + i][lc * k + i] += st;
            }
        }
    }
    for j in 0..2 {
        for i in 0..k {
            for l in 0..k {
                m[j * k + i][j * k + l] -= blocks.a[i][l] as f64;
            }
        }
    }
    let mut rhs = vec![0.0_f64; n];
    for j in 0..2 {
        for i in 0..k {
            rhs[j * k + i] = blocks.b[2 + i][j] as f64;
        }
    }
    // Partial pivoting; a pivot at rounding scale of the matrix magnitude
    // is the singular (spectral-overlap) verdict. The tolerance separates
    // cleanly: the fixtures' smallest genuine |λᵢ(S) − μⱼ(A)| is O(0.5)
    // while an exact overlap converges to ~1e-16.
    let scale = m.iter().flatten().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    let singular_tol = 1e-10 * scale.max(1.0) * n as f64;
    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = m[col][col].abs();
        for (ri, row) in m.iter().enumerate().skip(col + 1) {
            let v = row[col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = ri;
            }
        }
        if pivot_val <= singular_tol {
            return None;
        }
        m.swap(col, pivot_row);
        rhs.swap(col, pivot_row);
        let piv = m[col][col];
        let pivot_tail = m[col][col + 1..].to_vec();
        let rhs_pivot = rhs[col];
        for (ri, row) in m.iter_mut().enumerate().skip(col + 1) {
            let f = row[col] / piv;
            if f == 0.0 {
                continue;
            }
            for (pv, val) in pivot_tail.iter().zip(row.iter_mut().skip(col + 1)) {
                *val -= f * pv;
            }
            rhs[ri] -= f * rhs_pivot;
        }
    }
    let mut x = vec![0.0_f64; n];
    for r in (0..n).rev() {
        let mut acc = rhs[r];
        for c in (r + 1)..n {
            acc -= m[r][c] * x[c];
        }
        x[r] = acc / m[r][r];
    }
    // Unstack vec(P) (column-major) back to the k×2 matrix.
    Some((0..k).map(|i| vec![x[i], x[k + i]]).collect())
}

/// f64 2-norm (offline helper).
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// f64 row `i` of `P·(x, y)` for the k×2 coupling `p`.
fn p_row(p: &[Vec<f64>], i: usize, x: f64, y: f64) -> f64 {
    p[i][0] * x + p[i][1] * y
}

/// f64 `A·v` for the k×k complement dynamics.
fn matvec_a(a: &[Vec<f32>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v).map(|(&x, &y)| x as f64 * y).sum())
        .collect()
}

/// f64 `S·w` for the 2×2 oscillatory block.
fn matvec_s(s: &[[f32; 2]; 2], w: &[f64]) -> Vec<f64> {
    vec![
        s[0][0] as f64 * w[0] + s[0][1] as f64 * w[1],
        s[1][0] as f64 * w[0] + s[1][1] as f64 * w[1],
    ]
}

/// P6 — the Theorem 3.2 complement-energy split at the horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergySplit {
    /// `‖P Sᵗ u₀‖` — the mode-coupled response (inherits the oscillatory
    /// rate `r`; at `r ≈ 1` this is the stall carrier).
    pub coupled: f64,
    /// `‖Aᵗ(e₀ − P u₀)‖` — the complement's own (intrinsic) transient.
    pub intrinsic: f64,
    /// `‖e_t‖` — the actual complement magnitude (NOT the scalar sum of
    /// the two shares; the split terms need not align).
    pub total: f64,
}

/// P6 — simulate `e_t = A^t(e₀ − P u₀) + P S^t u₀` in f64 and split the
/// complement energy at horizon `t`. At `t = 0` the decomposition identity
/// gives `total = ‖e₀‖` (the sanity pin the gates hold).
pub fn energy_split(
    blocks: &SylvesterBlocks,
    p: &[Vec<f64>],
    u0: &[f32],
    e0: &[f32],
    t: usize,
) -> EnergySplit {
    let mut v: Vec<f64> = (0..blocks.a.len())
        .map(|i| e0[i] as f64 - p_row(p, i, u0[0] as f64, u0[1] as f64))
        .collect();
    let mut w = vec![u0[0] as f64, u0[1] as f64];
    for _ in 0..t {
        v = matvec_a(&blocks.a, &v);
        w = matvec_s(&blocks.s, &w);
    }
    let coupled: Vec<f64> = (0..v.len()).map(|i| p_row(p, i, w[0], w[1])).collect();
    let intrinsic = norm2(&v);
    let total = norm2(
        &v.iter()
            .zip(&coupled)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>(),
    );
    EnergySplit {
        coupled: norm2(&coupled),
        intrinsic,
        total,
    }
}

/// P6 — prediction-match tolerance for [`alpha_replay_check`]. The identity
/// arm measures f64 roundoff (~1e-15); a real coupling mismatch surfaces
/// orders of magnitude above this.
pub const ALPHA_REPLAY_TOL: f64 = 1e-5;

/// P6 — the canonical replay probe state: `u₀ = [1, 0]`, `e₀ᵢ = 1/(i + 1)`
/// — the initial condition [`alpha_replay_check`] intervenes on. Exposed so
/// gates and the report instrument reproduce the exact same state (`dim_k`
/// is the complement dimension).
pub fn replay_probe_state(dim_k: usize) -> (Vec<f32>, Vec<f32>) {
    (
        vec![1.0, 0.0],
        (0..dim_k).map(|i| 1.0 / (i as f32 + 1.0)).collect(),
    )
}

/// P6 — one row of the α-replay report table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaReplayVerdict {
    /// The applied amplitude projection.
    pub alpha: f32,
    /// `‖e_k‖` measured by stepping the coupled recurrence from the
    /// intervened state.
    pub measured: f64,
    /// `‖A^k(e₀ − αPu₀) + α·P S^k u₀‖` — the decomposition's prediction.
    pub closed_form: f64,
    /// Max-abs coordinate deviation, measured vs closed form.
    pub deviation: f64,
    /// `deviation < ALPHA_REPLAY_TOL`.
    pub matches: bool,
}

/// P6 — the paper's α-intervention causal check, closed form. Project the
/// oscillatory amplitude ONCE (`u₀ → α·u₀`, the complement untouched),
/// resume the linear recurrence `k` steps, and compare the measured
/// complement response against the decomposition's closed form
/// `A^k(e₀ − αPu₀) + α·P S^k u₀` — the α-scaled coupled contribution. For
/// the linear instrument the identity is EXACT up to f64 roundoff at every
/// α (`α = 1` the identity arm; `α = 0` zeroes the coupled drive), so a
/// deviation beyond [`ALPHA_REPLAY_TOL`] is a first-class recorded
/// NEGATIVE: the attribution's `p` does not describe the measured coupling.
/// Replay-only by contract — never a production write.
///
/// `p` is passed in (not re-solved) precisely so a stale attribution
/// against the measured `blocks.b` is expressible — that mismatch IS the
/// negative path. `k` is the replay horizon in steps; the complement
/// dimension is `blocks.a.len()`.
pub fn alpha_replay_check(
    blocks: &SylvesterBlocks,
    p: &[Vec<f64>],
    alpha: f32,
    k: usize,
) -> AlphaReplayVerdict {
    let (u0, e0) = replay_probe_state(blocks.a.len());
    let a = f64::from(alpha);
    let u0a = [u0[0] as f64 * a, u0[1] as f64 * a];
    // Measured: step the TRUE coupled recurrence (drive = B_c) from the
    // intervened state — the closed form is built from `p`, so a `p` that
    // does not describe the measured coupling is exactly what the deviation
    // exposes.
    let mut u: Vec<f64> = u0a.to_vec();
    let mut e: Vec<f64> = e0.iter().map(|&x| x as f64).collect();
    for _ in 0..k {
        let drive: Vec<f64> = (0..e.len())
            .map(|i| blocks.b[2 + i][0] as f64 * u[0] + blocks.b[2 + i][1] as f64 * u[1])
            .collect();
        e = matvec_a(&blocks.a, &e)
            .iter()
            .zip(drive)
            .map(|(x, d)| x + d)
            .collect();
        u = matvec_s(&blocks.s, &u);
    }
    // Closed form: A^k(e₀ − αPu₀) + α·P S^k u₀ (linearity folds α into u₀).
    let mut v: Vec<f64> = (0..e0.len())
        .map(|i| e0[i] as f64 - p_row(p, i, u0a[0], u0a[1]))
        .collect();
    let mut w = u0a.to_vec();
    for _ in 0..k {
        v = matvec_a(&blocks.a, &v);
        w = matvec_s(&blocks.s, &w);
    }
    let closed: Vec<f64> = v
        .iter()
        .zip((0..v.len()).map(|i| p_row(p, i, w[0], w[1])))
        .map(|(x, c)| x + c)
        .collect();
    let deviation = e
        .iter()
        .zip(&closed)
        .map(|(x, c)| (x - c).abs())
        .fold(0.0_f64, f64::max);
    AlphaReplayVerdict {
        alpha,
        measured: norm2(&e),
        closed_form: norm2(&closed),
        deviation,
        matches: deviation < ALPHA_REPLAY_TOL,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact single-mode recovery through the ridge (the rank-deficiency
    /// repair must not lose the true mode).
    #[test]
    fn ridge_keeps_the_true_mode_dominant_on_single_mode_windows() {
        for lambda in [-0.7_f32, 0.5, 1.0] {
            let mut f = ModeFitter::new();
            let mut d = 1.0_f32;
            for _ in 0..MODE_WINDOW + 4 {
                f.observe(d);
                d *= lambda;
            }
            let m = f.estimate().expect("single-mode window must fit");
            assert!(
                (m.r - lambda.abs()).abs() < 1e-3,
                "lambda {lambda}: recovered r {} vs {}",
                m.r,
                lambda.abs()
            );
        }
    }

    /// Exact stillness is inert (the aperiodic/stuck arm).
    #[test]
    fn zero_deltas_are_inert() {
        let mut f = ModeFitter::new();
        for _ in 0..MODE_WINDOW + 4 {
            f.observe(0.0);
        }
        assert_eq!(
            f.estimate(),
            None,
            "an exactly-still window must report None"
        );
    }

    /// Non-finite feeds are dropped, never poison the sums.
    #[test]
    fn non_finite_feeds_are_dropped() {
        let mut f = ModeFitter::new();
        f.observe_signed(f32::NAN, -1.0);
        f.observe_signed(f32::INFINITY, -1.0);
        let mut d = 1.0_f32;
        for _ in 0..MODE_WINDOW + 4 {
            f.observe_signed(d, -1.0);
            d *= 0.5;
        }
        let m = f
            .estimate()
            .expect("finite window after dropped feeds must fit");
        assert!((m.r - 0.5).abs() < 1e-3);
    }
}
