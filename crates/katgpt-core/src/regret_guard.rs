//! Regret-guard — adversary-aware serving gate (Issue 979 / Research 647;
//! arXiv:2603.28499 "Next-Token Prediction and Regret Minimization",
//! Mohri, Sanford, Schneider, Vodrahalli, Wu; ICML 2026).
//!
//! Two opt-in primitives for a serving lane that consumes a sequential
//! outcome stream (realized outcomes + a known utility table):
//!
//! 1. [`RegretGapGate`] — the paper's Algorithm 1 as a runtime gate: a
//!    frozen (incumbent) predictor leg and a Polya-urn reference leg
//!    (≡ Hedge by Lem 2.3's QBR derivation) are scored on the SAME
//!    realized outcome sequence; the per-round gap is
//!    `avg-regret(frozen) − avg-regret(urn)` and a **sticky OR-latch**
//!    switches the served prediction source to the urn when
//!    `gap ≥ η·ln|A| + √(8(1+α)·ln T / s)` — index-pinned to the paper's
//!    Alg 1: the Azuma root term runs over the switch-prefix length `s`,
//!    the quantal-price term over the horizon `T` (Lem 6.1's per-round
//!    quantal price `η·ln|A|`; Thm 7.1 restates the same band). The urn
//!    leg is exp-family (Hedge) and the frozen leg's action distribution
//!    is a quantal best response — the documented softmax exception class:
//!    softmax appears ONLY on these two action distributions; sigmoid
//!    stays on every gate/threshold/projection/mixing (there are none
//!    here — the band and gap are raw arithmetic).
//!
//! 2. [`VDriftDetector`] — the U-free binary escape (paper §8, Algorithm
//!    3): V-shaped proper scores `ℓ_v(p) = −|p−v|` over an ε-grid, the
//!    per-prefix max V-regret `Δ̂_t`, and the concentration threshold
//!    `√(ln(N_ε·t²/δ)/t)`. **Binary only** — the |Θ| ≥ 4 decomposition
//!    fails (Kleinberg et al.), so this type refuses nothing at run time
//!    and simply has no wider-alphabet surface (the scope pin; a wider
//!    alphabet has no honest detector here).
//!
//! # Loud refusals (fail-closed)
//!
//! The gate REFUSES to construct without a utility table `U: A×Θ → [−1,1]`
//! (the U-free escape is the [`VDriftDetector`] only), on non-finite or
//! out-of-range utilities, on degenerate alphabets, and per-round on a
//! malformed frozen state distribution (non-finite / negative / not
//! summing to 1). A serving lane wired to garbage input must stop, not
//! silently serve.
//!
//! # Temperature parameterization (η)
//!
//! The paper's Lemma 2.3 proof uses `η = T^(−1/2)`, giving the urn leg's
//! effective Hedge learning rate `λ_t = √T/(|Θ|+t−1)` and the per-round
//! quantal price `η·ln|A| = ln|A|/√T` — the form pinned in Issue 979
//! (the abs-page pseudocode renders "1/T"; the proof and the band
//! arithmetic pin `√T`; the issue's two consistent formulas carry it).
//! [`RegretGapConfig::qbr_eta`] exposes the knob; the default is
//! `1/√T`. Deviating from `1/√T` changes the Lem-2.3 constant, never the
//! mechanism.
//!
//! # Determinism (Issue-940 law)
//!
//! All accumulators are f64, folded in fixed index order (actions, then
//! states). No HashMap anywhere; per-round work is O(|A|·|Θ|) — the
//! paper's own cost for the regret condition (§5.2 cost analysis) — with
//! O(|A| + |Θ|) state and **zero allocation in steady state** (fixed
//! buffers reused; [`RegretGapGate::observe`] and [`VDriftDetector::
//! observe`] allocate nothing after construction).
//!
//! # Disclosure (Issue 979 §Disclosures)
//!
//! The gate needs realized outcomes + a known utility table; single-shot
//! decision serving without sequential feedback has no regret signal —
//! wiring onto a non-sequential consumer is a category error. The
//! certificate (de Bruijn, Thm 4.1) is pre-flight; the gate is runtime —
//! never conflate them. Promotion stays opt-in pending a consumer
//! measurement: the gate is real when a lane shows the firing case with
//! an incumbent-silent ground truth.

/// Loud construction / per-round refusals. Fail-closed: every variant is
/// a stop condition, never a degradation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegretGuardError {
    /// Utility table missing or empty — the gate refuses (the U-free
    /// escape is [`VDriftDetector`]).
    UtilityMissing,
    /// `u_flat.len() != n_actions * n_states` (or a dimension is zero /
    /// degenerate).
    UtilityShape { got: usize, want: usize },
    /// A utility entry is non-finite or outside `[-1, 1]` (the paper's
    /// `U: A×Θ → [−1,1]` contract feeds the band constants).
    UtilityRange { action: usize, state: usize },
    /// Config out of range: `alpha <= 0`, `horizon_t < 2`, `qbr_eta <= 0`.
    ConfigInvalid,
    /// The frozen state distribution is malformed (non-finite, negative,
    /// or not summing to 1 within tolerance).
    FrozenDistInvalid,
    /// `theta_t` outside `[0, n_states)`.
    BadOutcome,
}

impl core::fmt::Display for RegretGuardError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UtilityMissing => write!(
                f,
                "regret-guard: utility table missing (the U-free escape is the V-drift detector only)"
            ),
            Self::UtilityShape { got, want } => {
                write!(f, "regret-guard: utility table shape {got} != {want}")
            }
            Self::UtilityRange { action, state } => write!(
                f,
                "regret-guard: U[{action}][{state}] non-finite or outside [-1,1]"
            ),
            Self::ConfigInvalid => write!(
                f,
                "regret-guard: config invalid (need alpha > 0, horizon_t >= 2, qbr_eta > 0)"
            ),
            Self::FrozenDistInvalid => write!(
                f,
                "regret-guard: frozen state distribution malformed (finite, non-negative, sums to 1)"
            ),
            Self::BadOutcome => write!(f, "regret-guard: outcome index outside the state space"),
        }
    }
}

/// Gate configuration.
#[derive(Clone, Debug)]
pub struct RegretGapConfig {
    /// The TV-budget exponent α > 0 (Thm 3.1: `d_TV ≤ |A|·T^(−α)`; the
    /// band's Azuma root carries `(1+α)`).
    pub alpha: f64,
    /// The time horizon `T` (the serving lane's eval window / episode
    /// budget). Enters the band's quantal-price and root terms.
    pub horizon_t: usize,
    /// QBR temperature `η`. Default (issue-pinned): `1/√T` — see the
    /// module doc. `None` = the default.
    pub qbr_eta: Option<f64>,
}

impl RegretGapConfig {
    /// The issue-pinned default posture: α = 0.5, η = 1/√T, horizon T.
    pub fn new(alpha: f64, horizon_t: usize) -> Result<Self, RegretGuardError> {
        if alpha <= 0.0 || alpha.is_nan() || horizon_t < 2 {
            return Err(RegretGuardError::ConfigInvalid);
        }
        Ok(Self {
            alpha,
            horizon_t,
            qbr_eta: None,
        })
    }
}

/// One round's step report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegretStep {
    /// The sticky latch fired THIS round (first crossing). After this the
    /// served prediction source is the urn from the next round on.
    pub switched_now: bool,
    /// Current avg-regret gap `(S_urn − S_frozen)/s`.
    pub gap: f64,
    /// Current switch band `η·ln|A| + √(8(1+α)·ln T / s)`.
    pub band: f64,
    /// Rounds observed (the prefix length `s`).
    pub rounds: usize,
    /// Realized utility of the frozen leg's QBR this round.
    pub frozen_util: f64,
    /// Realized utility of the urn leg's QBR this round.
    pub urn_util: f64,
}

/// The adversary-aware serving gate (paper Alg 1 as a runtime latch).
///
/// Construct once per served stream with the known utility table; feed
/// [`Self::observe`] one round at a time with the frozen model's state
/// distribution and the revealed outcome; serve the urn leg's predictions
/// from the round after [`Self::switched`] first reads true.
///
/// # The two legs (both exp-family — the documented softmax exception)
///
/// - **Frozen leg**: the caller's model. Its action distribution is the
///   quantal best response to its own state distribution:
///   `π_f(a) ∝ exp(U(a, μ_t)/η)`.
/// - **Urn leg**: the Polya-urn state predictor
///   `μ_PU(θ) = (1 + #{s<t: θ_s=θ})/(|Θ|+t−1)`; its QBR factors into
///   exp-weights with the round-t Hedge rate
///   `λ_t = 1/(η(|Θ|+t−1))` over the realized per-action utilities
///   (Lemma 2.3) — i.e. the urn leg IS Hedge up to a fixed prior term
///   `U(a, uniform)`-shaped constant.
///
/// # Regret accounting (shared, O(1) per round beyond the QBRs)
///
/// Both legs are scored on the same outcomes, so their external regrets
/// share the per-action realized-utility vector `realized[a] =
/// Σ_s U(a,θ_s)`: `Regret_leg(s) = (max_a realized[a] − S_leg)/s`, and
/// the **gap** collapses to `(S_urn − S_frozen)/s` — the realized utility
/// the incumbent forfeited to the no-regret reference.
pub struct RegretGapGate {
    n_actions: usize,
    n_states: usize,
    u: Vec<f64>,
    u_rowsum: Vec<f64>,
    realized: Vec<f64>,
    state_counts: Vec<f64>,
    sum_frozen: f64,
    sum_urn: f64,
    rounds: usize,
    switched: bool,
    latched_round: Option<usize>,
    // Fixed scratch (zero-alloc steady state): reused per round.
    ell_frozen: Vec<f64>,
    ell_urn: Vec<f64>,
    pi_frozen: Vec<f64>,
    pi_urn: Vec<f64>,
    // Band constants.
    eta: f64,
    quantal_price: f64,
    band_root: f64,
    // Disclosure surface: the 0.42-same-seed law lives on d_NT, never
    // full-sequence TV; the served/frozen next-token TV is computed by
    // the consumer from these.
    tv_sum: f64,
}

impl RegretGapGate {
    /// Construct with the utility table in row-major `[A][Θ]` order.
    /// Refuses loud (see [`RegretGuardError`]) on any malformed input.
    pub fn new(
        u_flat: &[f32],
        n_actions: usize,
        n_states: usize,
        cfg: &RegretGapConfig,
    ) -> Result<Self, RegretGuardError> {
        if n_actions < 2 || n_states < 2 {
            return Err(RegretGuardError::UtilityShape {
                got: n_actions * n_states,
                want: 0,
            });
        }
        let want = n_actions * n_states;
        if u_flat.is_empty() {
            return Err(RegretGuardError::UtilityMissing);
        }
        if u_flat.len() != want {
            return Err(RegretGuardError::UtilityShape {
                got: u_flat.len(),
                want,
            });
        }
        if cfg.alpha <= 0.0 || cfg.alpha.is_nan() || cfg.horizon_t < 2 {
            return Err(RegretGuardError::ConfigInvalid);
        }
        let eta = match cfg.qbr_eta {
            None => 1.0 / (cfg.horizon_t as f64).sqrt(),
            Some(e) if e > 0.0 => e,
            Some(_) => return Err(RegretGuardError::ConfigInvalid),
        };
        let mut u = vec![0.0f64; want];
        for (i, &v) in u_flat.iter().enumerate() {
            let v = v as f64;
            if !v.is_finite() || v < -1.0 || v > 1.0 {
                return Err(RegretGuardError::UtilityRange {
                    action: i / n_states,
                    state: i % n_states,
                });
            }
            u[i] = v;
        }
        // Row sums (the urn prior factor), sequential fold in state order.
        let mut u_rowsum = vec![0.0f64; n_actions];
        for a in 0..n_actions {
            let mut acc = 0.0f64;
            for th in 0..n_states {
                acc += u[a * n_states + th];
            }
            u_rowsum[a] = acc;
        }
        let ln_a = (n_actions as f64).ln();
        let ln_t = (cfg.horizon_t as f64).ln();
        Ok(Self {
            n_actions,
            n_states,
            u,
            u_rowsum,
            realized: vec![0.0f64; n_actions],
            state_counts: vec![0.0f64; n_states],
            sum_frozen: 0.0,
            sum_urn: 0.0,
            rounds: 0,
            switched: false,
            latched_round: None,
            ell_frozen: vec![0.0; n_actions],
            ell_urn: vec![0.0; n_actions],
            pi_frozen: vec![0.0; n_actions],
            pi_urn: vec![0.0; n_actions],
            eta,
            quantal_price: eta * ln_a,
            band_root: (8.0 * (1.0 + cfg.alpha) * ln_t).sqrt(),
            tv_sum: 0.0,
        })
    }

    /// Feed one round: the frozen model's state distribution `mu` (over
    /// `Θ`) and the revealed outcome `theta_t`. Returns the step report.
    /// Zero allocation; O(|A|·|Θ|).
    ///
    /// Post-latch rounds keep the urn bookkeeping running (it IS the
    /// served leg) and keep the frozen stats updated for observability,
    /// but the latch never re-fires or un-fires (sticky OR).
    pub fn observe(&mut self, mu: &[f32], theta_t: usize) -> Result<RegretStep, RegretGuardError> {
        if theta_t >= self.n_states {
            return Err(RegretGuardError::BadOutcome);
        }
        if mu.len() != self.n_states {
            return Err(RegretGuardError::FrozenDistInvalid);
        }
        let mut mass = 0.0f64;
        for &p in mu {
            let p = p as f64;
            if !p.is_finite() || p < 0.0 {
                return Err(RegretGuardError::FrozenDistInvalid);
            }
            mass += p;
        }
        if (mass - 1.0).abs() > 1e-3 {
            return Err(RegretGuardError::FrozenDistInvalid);
        }

        // ── Frozen leg: action utilities ℓ[a] = Σ_θ μ[θ]·U[a][θ],
        //    sequential in state order. ──
        for a in 0..self.n_actions {
            let row = &self.u[a * self.n_states..(a + 1) * self.n_states];
            let mut acc = 0.0f64;
            for (u_ath, mu_th) in row.iter().zip(mu.iter()) {
                acc += *mu_th as f64 * u_ath;
            }
            self.ell_frozen[a] = acc;
        }
        qbr_into(&self.ell_frozen, self.eta, &mut self.pi_frozen);

        // ── Urn leg: Polya-urn state predictor → action utilities. ──
        let denom = (self.n_states + self.rounds) as f64;
        for a in 0..self.n_actions {
            self.ell_urn[a] = (self.u_rowsum[a] + self.realized[a]) / denom;
        }
        qbr_into(&self.ell_urn, self.eta, &mut self.pi_urn);

        // ── Realized utilities this round: U(π, θ_t) = Σ_a π[a]·U[a][θ_t].
        //    Sequential in action order (Issue-940 pin). ──
        let mut u_f = 0.0f64;
        let mut u_u = 0.0f64;
        for a in 0..self.n_actions {
            let ua = self.u[a * self.n_states + theta_t];
            u_f += self.pi_frozen[a] * ua;
            u_u += self.pi_urn[a] * ua;
        }

        // ── Update accumulators. ──
        for a in 0..self.n_actions {
            self.realized[a] += self.u[a * self.n_states + theta_t];
        }
        self.state_counts[theta_t] += 1.0;
        self.sum_frozen += u_f;
        self.sum_urn += u_u;
        self.rounds += 1;

        // Next-token TV between the two legs this round (disclosure
        // surface — the consumer's d_NT law reads this, never
        // full-sequence TV).
        let mut tv = 0.0f64;
        for a in 0..self.n_actions {
            tv += (self.pi_frozen[a] - self.pi_urn[a]).abs();
        }
        self.tv_sum += tv / 2.0;

        // ── The band + sticky OR-latch (Alg 1 verbatim, indexes pinned:
        //    root over the prefix length s, quantal price over T). ──
        let s = self.rounds as f64;
        let gap = (self.sum_urn - self.sum_frozen) / s;
        let band = self.quantal_price + self.band_root / s.sqrt();
        let mut switched_now = false;
        if !self.switched && gap >= band {
            self.switched = true;
            self.latched_round = Some(self.rounds);
            switched_now = true;
        }

        Ok(RegretStep {
            switched_now,
            gap,
            band,
            rounds: self.rounds,
            frozen_util: u_f,
            urn_util: u_u,
        })
    }

    /// The sticky latch: once true, always true.
    #[inline]
    pub fn switched(&self) -> bool {
        self.switched
    }

    /// The round at which the latch fired (None = never).
    #[inline]
    pub fn latched_round(&self) -> Option<usize> {
        self.latched_round
    }

    /// Current gap `(S_urn − S_frozen)/s`.
    #[inline]
    pub fn gap(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            (self.sum_urn - self.sum_frozen) / self.rounds as f64
        }
    }

    /// Current band `η·ln|A| + √(8(1+α)·ln T / s)`.
    #[inline]
    pub fn band(&self) -> f64 {
        if self.rounds == 0 {
            self.quantal_price + self.band_root
        } else {
            self.quantal_price + self.band_root / (self.rounds as f64).sqrt()
        }
    }

    /// Rounds observed.
    #[inline]
    pub fn rounds(&self) -> usize {
        self.rounds
    }

    /// The two legs' most recent QBR action distributions
    /// (`(frozen, urn)`) — the consumer's d_NT surface.
    #[inline]
    pub fn last_pis(&self) -> (&[f64], &[f64]) {
        (&self.pi_frozen, &self.pi_urn)
    }

    /// Cumulative mean next-token TV between the legs
    /// `(1/s)·Σ_t d_TV(π_frozen,t, π_urn,t)` — the fidelity axis is
    /// `d_NT`, never full-sequence TV (the 0.42 same-seed law, Research
    /// 647 §1).
    #[inline]
    pub fn mean_next_token_tv(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.tv_sum / self.rounds as f64
        }
    }
}

/// Quantal best response over action utilities, max-subtract stabilized:
/// `π[a] ∝ exp((ℓ[a] − max ℓ)/η)`. This is the paper's exp-family — the
/// documented softmax exception class (Lem 2.3 + the QBR definition);
/// `sigmoid` never replaces it here. Overflow-proof for any η > 0.
fn qbr_into(ell: &[f64], eta: f64, out: &mut [f64]) {
    let mut max_ell = ell[0];
    for &v in &ell[1..] {
        if v > max_ell {
            max_ell = v;
        }
    }
    let mut acc = 0.0f64;
    for (o, &v) in out.iter_mut().zip(ell.iter()) {
        let e = ((v - max_ell) / eta).exp();
        *o = e;
        acc += e;
    }
    for o in out.iter_mut() {
        *o /= acc;
    }
}

// ─────────────────────────────────────────────────────────────────────
// T2 — V-drift detector (binary, U-free)
// ─────────────────────────────────────────────────────────────────────

/// V-drift configuration.
#[derive(Clone, Debug)]
pub struct VDriftConfig {
    /// Grid step ε ∈ (0, 1]: tips `v ∈ {0, ε, …, 1}`, `N_ε = ⌊1/ε⌋ + 1`.
    pub eps: f64,
    /// Confidence δ ∈ (0, 1) (the threshold's log term).
    pub delta: f64,
}

/// Per-round report.
#[derive(Clone, Copy, Debug)]
pub struct VDriftStep {
    /// Max V-regret `Δ̂_t` over the grid.
    pub drift: f64,
    /// Concentration threshold `√(ln(N_ε·t²/δ)/t)`.
    pub threshold: f64,
    /// The sticky fire bit flipped THIS round.
    pub fired_now: bool,
}

/// The U-free binary drift detector (paper §8 / Algorithm 3): the
/// empirical max V-regret of the frozen predictor's per-round
/// probability-of-1 forecasts `p_t` against the binary outcomes, with
/// the prefix-uniform concentration threshold.
///
/// # Scope pin (the |Θ| ≥ 4 refusal, enforced by absence)
///
/// This type IS the binary detector — it consumes `p ∈ [0,1]` forecasts
/// of a binary outcome. There is no wider-alphabet surface to call: the
/// V-shaped decomposition fails at |Θ| ≥ 4 (Kleinberg et al.), and a
/// silently-computed cover would be exactly the failure the issue pins.
/// A multi-state lane needs the [`RegretGapGate`] (with a supplied U).
///
/// # The closed-form V-loss (piecewise constant — why this is O(N_ε))
///
/// With `g_v(p) = sign(v−p)`, the bivariate V-score
/// `ℓ_v(p,θ) = ℓ_v(p) ∓ (…)` is CONSTANT on each side of the tip:
/// `p < v: (θ=0 → −v, θ=1 → 1−v)`; `p > v: (θ=0 → v, θ=1 → v−1)`;
/// `p = v: 0`. The best constant-p loss is therefore
/// `min{0, −vZ + (1−v)O, vZ + (v−1)O}` with `Z/O` the global zero/one
/// counts — no search over p (unit-tested against a brute-force
/// constant-p sweep).
pub struct VDriftDetector {
    tips: Vec<f64>,
    loss_sums: Vec<f64>,
    zeros: u64,
    ones: u64,
    rounds: usize,
    fired: bool,
    log_delta: f64,
    n_eps: usize,
}

impl VDriftDetector {
    /// Construct the ε-grid detector. Refuses `eps ∉ (0,1]` or
    /// `delta ∉ (0,1)`.
    pub fn new(cfg: &VDriftConfig) -> Result<Self, RegretGuardError> {
        if cfg.eps <= 0.0
            || cfg.eps.is_nan()
            || cfg.eps > 1.0
            || cfg.delta <= 0.0
            || cfg.delta.is_nan()
            || cfg.delta >= 1.0
        {
            return Err(RegretGuardError::ConfigInvalid);
        }
        let n = ((1.0 / cfg.eps).floor() as usize) + 1;
        let mut tips = Vec::with_capacity(n);
        for i in 0..n {
            tips.push((i as f64) * cfg.eps);
        }
        // Pin the top tip at exactly 1.0 (f32/f64 grid rounding).
        tips[n - 1] = 1.0;
        Ok(Self {
            loss_sums: vec![0.0; n],
            tips,
            zeros: 0,
            ones: 0,
            rounds: 0,
            fired: false,
            log_delta: cfg.delta.ln(),
            n_eps: n,
        })
    }

    /// The V-loss value for tip `v`, forecast `p`, outcome `θ` — the
    /// piecewise-constant form (exposed for the spec-match test).
    pub fn v_loss(v: f64, p: f64, theta: usize) -> f64 {
        if p == v {
            return 0.0;
        }
        if p < v {
            if theta == 0 { -v } else { 1.0 - v }
        } else if theta == 0 {
            v
        } else {
            v - 1.0
        }
    }

    /// The best constant-p loss for tip `v` given `Z` zeros and `O` ones:
    /// `min{0, −vZ + (1−v)O, vZ + (v−1)O}` (p = v achieves 0 exactly).
    pub fn best_constant_loss(v: f64, zeros: u64, ones: u64) -> f64 {
        let z = zeros as f64;
        let o = ones as f64;
        let left = -v * z + (1.0 - v) * o;
        let right = v * z + (v - 1.0) * o;
        let m = left.min(right);
        if m < 0.0 { m } else { 0.0 }
    }

    /// Feed one round: the frozen predictor's probability-of-1 forecast
    /// `p ∈ [0,1]` and the binary outcome `theta ∈ {0, 1}`. Zero
    /// allocation; O(N_ε).
    pub fn observe(&mut self, p: f64, theta: usize) -> Result<VDriftStep, RegretGuardError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(RegretGuardError::FrozenDistInvalid);
        }
        if theta > 1 {
            return Err(RegretGuardError::BadOutcome);
        }
        for (i, &v) in self.tips.iter().enumerate() {
            self.loss_sums[i] += Self::v_loss(v, p, theta);
        }
        if theta == 0 {
            self.zeros += 1;
        } else {
            self.ones += 1;
        }
        self.rounds += 1;
        let t = self.rounds as f64;
        let mut drift = f64::NEG_INFINITY;
        for (i, &v) in self.tips.iter().enumerate() {
            let regret =
                (self.loss_sums[i] - Self::best_constant_loss(v, self.zeros, self.ones)) / t;
            if regret > drift {
                drift = regret;
            }
        }
        let threshold = self.threshold();
        let mut fired_now = false;
        if !self.fired && drift > threshold {
            self.fired = true;
            fired_now = true;
        }
        Ok(VDriftStep {
            drift,
            threshold,
            fired_now,
        })
    }

    /// The concentration threshold `B·√(ln(N_ε·t²/δ)/t)` (the issue's
    /// conservative √-form of Lemma 8.8's `O(B·log(N_ε t²/δ)/t)` — it
    /// fires LESS eagerly, the safe direction for a serving gate; the
    /// paper's tighter form is the reference). `B = 2` is the V-loss
    /// range `|ℓ_v(p,θ) − min_p| ≤ 2`: the regret of a single round
    /// spans `min − max = 2` (measured at t = 1 — the B constant is
    /// load-bearing, not decorative).
    pub fn threshold(&self) -> f64 {
        const LOSS_RANGE_B: f64 = 2.0;
        if self.rounds < 2 {
            // Warmup floor: one observation cannot evidence drift (the
            // L_min-floor law — the same shape GainCostLoopHalter
            // ships; the t=1 threshold is vacuous against a range-2
            // regret).
            return f64::INFINITY;
        }
        let t = self.rounds as f64;
        LOSS_RANGE_B * (((self.n_eps as f64) * t * t / -self.log_delta).ln() / t).sqrt()
    }

    /// The sticky fire bit.
    pub fn fired(&self) -> bool {
        self.fired
    }

    /// Current max V-regret `Δ̂_t`.
    pub fn drift(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        let t = self.rounds as f64;
        let mut drift = f64::NEG_INFINITY;
        for (i, &v) in self.tips.iter().enumerate() {
            let regret =
                (self.loss_sums[i] - Self::best_constant_loss(v, self.zeros, self.ones)) / t;
            if regret > drift {
                drift = regret;
            }
        }
        drift
    }

    /// Rounds observed.
    pub fn rounds(&self) -> usize {
        self.rounds
    }
}

// ─────────────────────────────────────────────────────────────────────
// Alg 2 pullback (the Δ-context arm's helper)
// ─────────────────────────────────────────────────────────────────────

/// Algorithm 2's pullback for the indicator decision problem (A = Θ,
/// `U(a,θ) = 𝕀[a=θ]`, |A| = 2 in the arm): find the state distribution
/// `μ` whose QBR at temperature `η = 1/Δ` equals `pi_bar` — the
/// suffix-average of the sub-predictors' QBRs.
///
/// Derivation (closed form): `QBR(μ,η)[a] = exp(Δμ[a])/Z = π̄[a]` gives
/// `μ[a] = ln π̄[a]/Δ + (ln Z)/Δ`, and `Σ_a μ[a] = 1` pins
/// `ln Z = (Δ − Σ_a ln π̄[a])/|A|`. Returns `None` when any `pi_bar[a]
/// ≤ 0` (QBR outputs are strictly positive — a zero means the caller
/// averaged something that was not a QBR).
pub fn suffix_average_pullback(pi_bar: &[f64], delta: usize) -> Option<Vec<f64>> {
    if pi_bar
        .iter()
        .any(|&p| p <= 0.0 || p.is_nan() || !p.is_finite())
    {
        return None;
    }
    let k = pi_bar.len() as f64;
    let d = delta as f64;
    let mut ln_sum = 0.0f64;
    for &p in pi_bar {
        ln_sum += p.ln();
    }
    let ln_z = (d - ln_sum) / k;
    let mut mu = Vec::with_capacity(pi_bar.len());
    for &p in pi_bar {
        mu.push((p.ln() + ln_z) / d);
    }
    Some(mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ATOL: f64 = 1e-9;

    #[test]
    fn qbr_is_stable_at_extreme_temperature() {
        // η·ℓ = ±√T-scale exponent — the max-subtract must not overflow.
        let ell = [-1.0f64, 1.0];
        let mut out = [0.0f64; 2];
        qbr_into(&ell, 1e-4, &mut out);
        assert!(out[0].is_finite() && out[1].is_finite());
        assert!((out[0] + out[1] - 1.0).abs() < ATOL);
        assert!(out[1] > 0.999999, "larger utility dominates the QBR");
    }

    #[test]
    fn qbr_uniform_at_zero_utilities() {
        let ell = [0.0f64; 3];
        let mut out = [0.0f64; 3];
        qbr_into(&ell, 0.5, &mut out);
        for v in out {
            assert!((v - 1.0 / 3.0).abs() < ATOL);
        }
    }

    fn indicator_gate(cfg: &RegretGapConfig) -> RegretGapGate {
        // Adversarial online prediction: A = Θ = {0,1}, U(a,θ) = 𝕀[a=θ].
        RegretGapGate::new(&[1.0, 0.0, 0.0, 1.0], 2, 2, cfg).expect("indicator gate")
    }

    #[test]
    fn gate_refuses_loud_on_bad_inputs() {
        // Missing U.
        assert!(matches!(
            RegretGapGate::new(&[], 2, 2, &RegretGapConfig::new(0.5, 64).unwrap()),
            Err(RegretGuardError::UtilityMissing)
        ));
        // Wrong shape.
        assert!(matches!(
            RegretGapGate::new(&[1.0, 0.0], 2, 2, &RegretGapConfig::new(0.5, 64).unwrap()),
            Err(RegretGuardError::UtilityShape { .. })
        ));
        // Out-of-range entry.
        assert!(matches!(
            RegretGapGate::new(
                &[1.0, 0.0, 0.0, 1.5],
                2,
                2,
                &RegretGapConfig::new(0.5, 64).unwrap()
            ),
            Err(RegretGuardError::UtilityRange { .. })
        ));
        // Non-finite entry.
        assert!(matches!(
            RegretGapGate::new(
                &[1.0, 0.0, 0.0, f32::NAN],
                2,
                2,
                &RegretGapConfig::new(0.5, 64).unwrap()
            ),
            Err(RegretGuardError::UtilityRange { .. })
        ));
        // Bad config (alpha = 0): the config struct is built directly.
        let bad_cfg = RegretGapConfig {
            alpha: 0.0,
            horizon_t: 64,
            qbr_eta: None,
        };
        assert!(matches!(
            RegretGapGate::new(&[1.0, 0.0, 0.0, 1.0], 2, 2, &bad_cfg),
            Err(RegretGuardError::ConfigInvalid)
        ));
        // Malformed frozen dist + bad outcome.
        let mut g = indicator_gate(&RegretGapConfig::new(0.5, 64).unwrap());
        assert_eq!(
            g.observe(&[0.5, 0.2], 0),
            Err(RegretGuardError::FrozenDistInvalid)
        );
        assert_eq!(g.observe(&[0.5, 0.5], 5), Err(RegretGuardError::BadOutcome));
    }

    #[test]
    fn urn_leg_qbr_matches_hedge_form() {
        // Lemma 2.3: the urn leg's QBR must equal exp-weights over
        // (prior + realized) utilities at rate 1/(η(|Θ|+t−1)). Check the
        // round-2 prediction (the first one with non-empty history): the
        // stored π after k observes was computed with rounds = k−1.
        let cfg = RegretGapConfig::new(0.5, 100).unwrap();
        let mut g = indicator_gate(&cfg);
        let eta = g.eta;
        let _ = g.observe(&[0.7, 0.3], 1).expect("round 1");
        // Round 1's stored π had EMPTY history → uniform (sanity).
        let (_, urn1) = g.last_pis();
        assert!((urn1[0] - 0.5).abs() < 1e-12);
        // Round 2's prediction: counts = (0, 1), realized = (0, 1),
        // denom = |Θ| + rounds = 3 → ℓ_urn = ((1+0)/3, (1+1)/3), i.e.
        // exp-weights at λ = 1/(η·3) over (1, 2).
        let _ = g.observe(&[0.7, 0.3], 1).expect("round 2");
        let lam = 1.0 / (eta * 3.0);
        let e0 = (lam * 1.0f64).exp();
        let e1 = (lam * 2.0f64).exp();
        let pi1 = e1 / (e0 + e1);
        let (_, urn) = g.last_pis();
        assert!(
            (urn[1] - pi1).abs() < 1e-12,
            "urn QBR {} vs Hedge form {}",
            urn[1],
            pi1
        );
    }

    #[test]
    fn latch_is_sticky_and_single() {
        // Constant counter-oracle: frozen always predicts 1, adversary
        // plays 0 — the gate must fire exactly once and stay latched.
        let cfg = RegretGapConfig::new(0.5, 512).unwrap();
        let mut g = indicator_gate(&cfg);
        let mut fires = 0;
        let mut latched_at = None;
        for _ in 0..300 {
            let step = g.observe(&[0.0, 1.0], 0).expect("round");
            if step.switched_now {
                fires += 1;
                latched_at = Some(step.rounds);
            }
        }
        assert_eq!(fires, 1, "the latch fires exactly once");
        assert!(latched_at.is_some(), "counter-oracle must fire the latch");
        assert!(g.switched());
        assert_eq!(g.latched_round(), latched_at);
    }

    #[test]
    fn calibrated_in_distribution_stream_stays_silent() {
        // Frozen = the true Ber(1/3) predictor; the stream sampled from
        // Ber(1/3) is in-distribution → the gate must stay silent
        // (Thm 3.1's switch probability ≤ |A|·T^(−α)).
        let cfg = RegretGapConfig::new(0.5, 512).unwrap();
        let mut g = indicator_gate(&cfg);
        let mut seed: u64 = 0x9799_7979;
        let mut fired = false;
        for _ in 0..512 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u01 = ((seed >> 11) as f64) / ((1u64 << 53) as f64);
            let theta = if u01 < 1.0 / 3.0 { 0 } else { 1 };
            let step = g.observe(&[1.0 / 3.0, 2.0 / 3.0], theta).expect("round");
            fired |= step.switched_now;
        }
        assert!(!fired, "in-distribution calibrated stream must not switch");
        assert!(!g.switched());
        assert_eq!(g.latched_round(), None);
    }

    #[test]
    fn v_loss_matches_brute_force_definition() {
        // ℓ_v(p,θ) via the sign-subgradient definition vs the
        // piecewise-constant form, over a dense p sweep.
        let eps = 1.0 / 7.0;
        let mut v = 0.0f64;
        while v <= 1.0 {
            let mut p = 0.0f64;
            while p <= 1.0 {
                let g = if v - p > 0.0 {
                    1.0
                } else if v - p < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let base = -(p - v).abs();
                let want0 = base - p * g;
                let want1 = base + (1.0 - p) * g;
                assert!(
                    (VDriftDetector::v_loss(v, p, 0) - want0).abs() < ATOL,
                    "v={v} p={p} θ=0"
                );
                assert!(
                    (VDriftDetector::v_loss(v, p, 1) - want1).abs() < ATOL,
                    "v={v} p={p} θ=1"
                );
                p += 0.001;
            }
            v += eps;
        }
    }

    #[test]
    fn best_constant_loss_matches_brute_force_sweep() {
        // min_p L(p) by brute force over a dense grid vs the closed form.
        let mut seed: u64 = 42;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for _ in 0..64 {
            // Random (v, Z, O).
            let v = next();
            let z = (next() * 200.0) as u64;
            let o = (next() * 200.0) as u64;
            let want = VDriftDetector::best_constant_loss(v, z, o);
            // Brute force: L(p) is piecewise constant with the single
            // kink at p = v; sample both sides + the tip.
            let mut got = f64::INFINITY;
            for p in [0.0, v / 2.0, v * 0.999, v, v * 1.001, (v + 1.0) / 2.0, 1.0] {
                let mut l = 0.0f64;
                for _ in 0..z {
                    l += VDriftDetector::v_loss(v, p, 0);
                }
                for _ in 0..o {
                    l += VDriftDetector::v_loss(v, p, 1);
                }
                if l < got {
                    got = l;
                }
            }
            assert!(
                (want - got).abs() < 1e-9,
                "v={v} Z={z} O={o}: closed {want} vs brute {got}"
            );
        }
    }

    #[test]
    fn v_drift_fires_on_counter_oracle_stays_silent_calibrated() {
        let cfg = VDriftConfig {
            eps: 0.1,
            delta: 0.05,
        };
        // Counter-oracle: frozen always forecasts p = 0.9; adversary
        // plays 0 → the V-regret gap explodes.
        let mut d = VDriftDetector::new(&cfg).expect("detector");
        let mut fired_at = None;
        for _ in 0..400 {
            let step = d.observe(0.9, 0).expect("round");
            if step.fired_now {
                fired_at = Some(d.rounds());
            }
        }
        assert!(
            fired_at.is_some(),
            "counter-oracle must fire the V-detector"
        );

        // Calibrated: frozen = the true Ber(1/2) — never fires.
        let mut d2 = VDriftDetector::new(&cfg).expect("detector");
        let mut seed: u64 = 0x1234_5678;
        let mut fired2 = false;
        for _ in 0..1000 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u01 = ((seed >> 11) as f64) / ((1u64 << 53) as f64);
            let theta = if u01 < 0.5 { 0 } else { 1 };
            let step = d2.observe(0.5, theta).expect("round");
            fired2 |= step.fired_now;
        }
        assert!(!fired2, "calibrated 50/50 forecaster must never fire");
    }

    #[test]
    fn pullback_roundtrips_through_qbr() {
        // QBR(pullback(π̄), 1/Δ) == π̄ (the Alg-2 averaging identity).
        let delta = 32usize;
        let eta = 1.0 / delta as f64;
        let mus = [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]];
        let mut pis = Vec::new();
        for m in &mus {
            let mut pi = [0.0f64; 2];
            qbr_into(m, eta, &mut pi);
            pis.push(pi);
        }
        let mut pi_bar = [0.0f64; 2];
        for pi in &pis {
            pi_bar[0] += pi[0] / pis.len() as f64;
            pi_bar[1] += pi[1] / pis.len() as f64;
        }
        let mu = suffix_average_pullback(&pi_bar, delta).expect("pullback");
        let mut got = [0.0f64; 2];
        qbr_into(&mu, eta, &mut got);
        assert!(
            (got[0] - pi_bar[0]).abs() < 1e-9,
            "{} vs {}",
            got[0],
            pi_bar[0]
        );
        assert!((got[1] - pi_bar[1]).abs() < 1e-9);
        // μ is a distribution.
        assert!((mu[0] + mu[1] - 1.0).abs() < 1e-9);
        for m in mu {
            assert!(m > 0.0);
        }
    }
}
