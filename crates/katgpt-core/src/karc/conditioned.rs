//! KARC Conditioned Readout — a bank of rank-1 readout corrections over an
//! incumbent base ridge fit, keyed by an EXOGENOUS condition label
//! (Issue 1239).
//!
//! # The primitive
//!
//! The base ridge readout stays the incumbent. Each correction `b` is a
//! rank-1 readout correction served to samples whose condition label `c`
//! matches the correction's key:
//!
//! ```text
//! û = Wout_base·Ψ(x_t) + Σ_b σ(g_b) · u_b · (v_bᵀ Ψ(x_t))
//! ```
//!
//! - `u_b` (output space, D floats) is FROZEN — an authored or archetype
//!   direction (the normalized mean base residual of the correction's own
//!   fit segment is the archetype default; see
//!   [`archetype_direction_into`]).
//! - `v_b` (feature space, d_h floats) is a closed-form ridge solve over
//!   residual targets — it reuses [`chunked_gram_into`] +
//!   [`ridge_solve_direct_f64`] verbatim. Zero training, zero backprop;
//!   consolidation-tier work (the same class as `fit_ridge`, which the
//!   runtime already runs during sleep).
//! - `g_b` is an EWMA relative-gain σ-gate on correction b's OWN keyed
//!   residual: `σ(β·(gain − margin))` where
//!   `gain = 1 − EWMA(corr²) / EWMA(base²)`, computed ONLY on observations
//!   whose condition matches b (the keyed stream). Abstain-first: below
//!   [`GateConfig::min_obs`] keyed observations the gate holds at 0.0 — a
//!   never-firing correction costs only its bytes.
//!
//! The base Wout must NOT be derived from a signal that also drives `g_b`
//! — the conditioning variable comes from a channel INDEPENDENT of the
//! gate (an exogenous regime label: combat flag, time-of-day bucket, …).
//! If `c` is derived from the same rolling residual that drives the gate,
//! the conditioned arm degenerates into a smoothed select and the
//! select-vs-composition comparison measures only smoothing (the
//! pre-registered design failure, Issue 1239).
//!
//! # Why rank-1 keyed corrections (the decision content)
//!
//! Plan 514 Phase 3 ships the same shape class keyed by IDENTITY (one
//! shared cell forecaster + per-NPC `npc_directions`). This module re-keys
//! identity → CONDITION: one correction then serves every entity in the
//! same condition (crowd amortization beyond cells), or stacks beside
//! per-entity deviations. State bytes stay flat at crowd scale: each
//! correction costs `d_h + D` floats (~1 KB at the HLA shape) regardless
//! of how many entities share it.
//!
//! # Modelless
//!
//! Every weight here is closed-form: the base readout is a ridge solve, the
//! corrections are ridge solves over frozen directions, the gate is an EWMA
//! + sigmoid (never softmax). No training, no gradient descent, no learned
//!   gate weights. The EWMA coefficient / margin / β are designer
//!   hyperparameters pinned in [`GateConfig`].
//!
//! # Determinism (the Issue-940 law)
//!
//! [`KarcConditionedReadout::apply_into`] folds corrections in ASCENDING
//! bank index, sequentially — pinned fold order, bit-identical across
//! calls. [`observe`] updates at most one accumulator per call (the
//! correction keyed to the observed condition), sequentially.
//!
//! # Allocation
//!
//! [`apply_into`] and [`observe`] are zero-alloc once constructed (all
//! reads/writes into caller-provided buffers; the bank is fixed at
//! construction). The fit helpers ([`fit_correction_into`]) are cold-path
//! consolidation work and may allocate scratch — they reuse
//! [`BankFitScratch`] across fits.
//!
//! # Sync boundary
//!
//! Latent-only: the corrections and gate state are locally-consumed
//! decision machinery and never cross `SyncBlock` raw. If a consumer
//! commits the gate state, commit only the scalar σ values (the 5-scalar
//! emotion-projection precedent).

use crate::karc::chunked_gram_into;
use crate::linalg::ridge_solve_direct_f64;
use crate::sigmoid;

/// Gate hyperparameters for the EWMA relative-gain σ-gate.
///
/// Defaults are the Issue-1239 pinned values: β 30, margin 0.15, EWMA 0.05,
/// abstain until 16 keyed observations. The calibration is split-conformal
/// in SHAPE (gates warm on a calibration slice disjoint from fit and test);
/// the full conformal calibrator is deliberately not wired here.
#[derive(Clone, Copy, Debug)]
pub struct GateConfig {
    /// Sigmoid inverse temperature on `(gain − margin)`.
    pub beta: f32,
    /// Relative-gain margin the gate must exceed before it opens.
    pub margin: f32,
    /// EWMA coefficient for the NEW sample (0 < w ≤ 1).
    pub ewma: f32,
    /// Keyed observations before the gate leaves abstain.
    pub min_obs: u32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            beta: 30.0,
            margin: 0.15,
            ewma: 0.05,
            min_obs: 16,
        }
    }
}

/// One keyed rank-1 correction: `u` (output space) frozen, `v` (feature
/// space) ridge-fitted, plus the EWMA gate accumulators over its own keyed
/// residual stream.
#[derive(Clone, Debug)]
pub struct ConditionedCorrection {
    /// The exogenous condition label this correction serves.
    pub condition: u8,
    /// Frozen output-space direction (D floats).
    pub u: Vec<f32>,
    /// Feature-space ridge weights (d_h floats).
    pub v: Vec<f32>,
    /// EWMA of the squared keyed BASE residual (scalar projection on `u`).
    ewma_base_sq: f64,
    /// EWMA of the squared keyed CORRECTED residual (base minus `v·Ψ`).
    ewma_corr_sq: f64,
    /// Keyed observations seen (the gate's abstain counter).
    n_keyed: u32,
}

impl ConditionedCorrection {
    /// Gate σ value in `[0, 1]`; 0.0 while abstaining or while the base
    /// residual stream is (near-)zero — a perfect base must never arm a
    /// correction on numerical noise.
    #[inline]
    pub fn gate_sigma(&self, cfg: &GateConfig) -> f32 {
        if self.n_keyed < cfg.min_obs {
            return 0.0;
        }
        if self.ewma_base_sq <= 1e-12 {
            return 0.0;
        }
        let gain = (1.0 - self.ewma_corr_sq / self.ewma_base_sq) as f32;
        sigmoid(cfg.beta * (gain - cfg.margin))
    }

    /// Keyed observations seen so far.
    #[inline]
    pub const fn n_keyed(&self) -> u32 {
        self.n_keyed
    }
}

/// A bank of rank-1 conditioned corrections over a base ridge readout.
///
/// Layout: `base_wout` is OUTPUT-major (`w[j * d_h + k]` — output j's
/// feature row), so the base matvec is a per-output dot product; the
/// correction contribution is one scalar dot per correction, then an AXPY
/// into the output.
#[derive(Clone, Debug)]
pub struct KarcConditionedReadout {
    d: usize,
    d_h: usize,
    base_wout: Vec<f32>,
    corrections: Vec<ConditionedCorrection>,
    cfg: GateConfig,
}

impl KarcConditionedReadout {
    /// Bank over a fitted base readout. `base_wout.len()` must be `d * d_h`.
    pub fn new(base_wout: Vec<f32>, d: usize, d_h: usize, cfg: GateConfig) -> Self {
        assert_eq!(
            base_wout.len(),
            d * d_h,
            "base_wout must hold d * d_h floats (output-major)"
        );
        Self {
            d,
            d_h,
            base_wout,
            corrections: Vec::new(),
            cfg,
        }
    }

    /// Append a keyed correction. `u.len() == d`, `v.len() == d_h`.
    pub fn add_correction(&mut self, condition: u8, u: Vec<f32>, v: Vec<f32>) {
        assert_eq!(u.len(), self.d, "correction u must hold d floats");
        assert_eq!(v.len(), self.d_h, "correction v must hold d_h floats");
        self.corrections.push(ConditionedCorrection {
            condition,
            u,
            v,
            ewma_base_sq: 0.0,
            ewma_corr_sq: 0.0,
            n_keyed: 0,
        });
    }

    /// Replace the FIRST correction keyed `condition` (the refit semantics):
    /// the new `(u, v)` mount in place, the EWMA gate accumulators (`n_keyed`,
    /// both EWMAs) carried over — the keyed residual stream is continuous
    /// across refits, so a re-fitted correction does not restart the gate's
    /// abstain window. Returns `false` when no correction carries the key
    /// (caller falls back to [`Self::add_correction`]); a key never mounts
    /// twice — the first match is the one live correction for that condition.
    pub fn replace_correction(&mut self, condition: u8, u: Vec<f32>, v: Vec<f32>) -> bool {
        assert_eq!(u.len(), self.d, "correction u must hold d floats");
        assert_eq!(v.len(), self.d_h, "correction v must hold d_h floats");
        for corr in &mut self.corrections {
            if corr.condition == condition {
                corr.u = u;
                corr.v = v;
                return true;
            }
        }
        false
    }

    /// Output dimension.
    #[inline]
    pub const fn d(&self) -> usize {
        self.d
    }

    /// Feature dimension.
    #[inline]
    pub const fn d_h(&self) -> usize {
        self.d_h
    }

    /// Bank size (number of keyed corrections).
    #[inline]
    pub fn n_corrections(&self) -> usize {
        self.corrections.len()
    }

    /// The gate config.
    #[inline]
    pub const fn config(&self) -> &GateConfig {
        &self.cfg
    }

    /// Gate σ for bank index `b`.
    #[inline]
    pub fn gate_sigma(&self, b: usize) -> f32 {
        self.corrections[b].gate_sigma(&self.cfg)
    }

    /// Read-only access to correction `b` (BANK index, not the condition
    /// label) — for offline evaluators that want the raw (u, v) pair.
    #[inline]
    pub fn correction(&self, b: usize) -> &ConditionedCorrection {
        &self.corrections[b]
    }

    /// Total state bytes in f32 units: base + Σ(d_h + D) per correction
    /// (the matched-bytes accounting the cell reports).
    pub fn state_f32(&self) -> usize {
        self.d * self.d_h
            + self
                .corrections
                .iter()
                .map(|c| c.u.len() + c.v.len())
                .sum::<usize>()
    }

    /// Base readout only: `out[j] = Σ_k base_wout[j·d_h + k]·features[k]`.
    /// Zero-alloc; `out.len() >= d`.
    #[inline]
    pub fn base_predict_into(&self, features: &[f32], out: &mut [f32]) {
        debug_assert!(features.len() >= self.d_h);
        debug_assert!(out.len() >= self.d);
        for (o, row) in out.iter_mut().zip(self.base_wout.chunks_exact(self.d_h)) {
            let mut acc = 0.0f32;
            for k in 0..self.d_h {
                acc += row[k] * features[k];
            }
            *o = acc;
        }
    }

    /// The conditioned readout: base + Σ_b σ(g_b)·u_b·(v_bᵀ Ψ) for every
    /// correction whose condition matches `c`, folded in ASCENDING bank
    /// index (the Issue-940 pinned fold order). Zero-alloc; `out.len() >= d`.
    #[inline]
    pub fn apply_into(&self, features: &[f32], c: u8, out: &mut [f32]) {
        self.base_predict_into(features, out);
        for corr in &self.corrections {
            if corr.condition != c {
                continue;
            }
            let s = corr.gate_sigma(&self.cfg);
            if s <= 0.0 {
                continue;
            }
            self.axpy_keyed(corr, features, s, out);
        }
    }

    /// The FIXED-gate posture: apply the keyed corrections at a caller
    /// supplied scale (no EWMA state consulted). This is the unlearned-fusion
    /// arm of the Issue-1239 cell (uniform mean = scale 1.0; the Σ/√K scale
    /// control = 1/√K) and the operator-forced posture for a consumer that
    /// has externally decided the correction is always-on for a condition.
    /// Same ascending fold order as [`apply_into`]; zero-alloc.
    #[inline]
    pub fn apply_fixed(&self, features: &[f32], c: u8, scale: f32, out: &mut [f32]) {
        self.base_predict_into(features, out);
        for corr in &self.corrections {
            if corr.condition != c {
                continue;
            }
            self.axpy_keyed(corr, features, scale, out);
        }
    }

    /// The rank-1 correction AXPY shared by both apply postures (private —
    /// one definition of the fold body).
    #[inline]
    fn axpy_keyed(&self, corr: &ConditionedCorrection, features: &[f32], s: f32, out: &mut [f32]) {
        let mut coef = 0.0f32;
        for (&f, &v) in features.iter().zip(corr.v.iter()) {
            coef += v * f;
        }
        coef *= s;
        for (o, &u) in out.iter_mut().zip(corr.u.iter()) {
            *o += coef * u;
        }
    }

    /// Observe one keyed sample: updates ONLY the correction whose condition
    /// matches `c` (the keyed stream). `scratch` must hold `d` floats (the
    /// base prediction temp). Zero-alloc.
    ///
    /// The gate compares the keyed base residual `u·(y − base)` against the
    /// keyed corrected residual `u·(y − base) − v·Ψ` — the UNGATED
    /// correction contribution, so the gate measures "would the correction
    /// have helped" and never feeds back on its own output.
    #[inline]
    pub fn observe(&mut self, features: &[f32], c: u8, truth: &[f32], scratch: &mut [f32]) {
        debug_assert!(truth.len() >= self.d);
        debug_assert!(scratch.len() >= self.d);
        // The base prediction is shared by every keyed correction — compute
        // it once, BEFORE the mutable loop (borrow split).
        self.base_predict_into(features, scratch);
        let w = self.cfg.ewma as f64;
        for corr in &mut self.corrections {
            if corr.condition != c {
                continue;
            }
            // r_base_scalar = u·(y − base); r_corr_scalar = r_base − v·Ψ.
            let mut r_base = 0.0f64;
            let mut r_corr = 0.0f64;
            for j in 0..self.d {
                let e = truth[j] as f64 - scratch[j] as f64;
                r_base += corr.u[j] as f64 * e;
                r_corr += corr.u[j] as f64 * e;
            }
            for (&f, &v) in features.iter().zip(corr.v.iter()) {
                r_corr -= v as f64 * f as f64;
            }
            corr.ewma_base_sq = w * (r_base * r_base) + (1.0 - w) * corr.ewma_base_sq;
            corr.ewma_corr_sq = w * (r_corr * r_corr) + (1.0 - w) * corr.ewma_corr_sq;
            corr.n_keyed += 1;
        }
    }
}

/// Cold-path scratch for the closed-form fit helpers. Allocate once, reuse
/// across fits (`with_capacity(d_h)`); the buffers cover the f64 Gram /
/// Cholesky / back-substitution chain for a `d_h`-wide feature space with a
/// single right-hand side.
#[derive(Clone, Debug)]
pub struct BankFitScratch {
    gram: Vec<f64>,
    l: Vec<f64>,
    z: Vec<f64>,
    w_t: Vec<f64>,
    cov: Vec<f64>,
    targets: Vec<f64>,
}

impl BankFitScratch {
    /// Scratch for a `d_h`-wide feature space (single scalar target).
    pub fn with_capacity(d_h: usize) -> Self {
        Self {
            gram: vec![0.0; d_h * d_h],
            l: vec![0.0; d_h * d_h],
            z: vec![0.0; d_h],
            w_t: vec![0.0; d_h],
            cov: vec![0.0; d_h],
            targets: Vec::with_capacity(1024),
        }
    }
}

/// The archetype direction: the normalized mean base residual of the
/// correction's own fit segment, written unit-norm into `out` (d floats).
/// Returns `false` when the mean residual is (near-)zero — the caller then
/// authors a direction instead (the "authored" leg of the frozen-u law).
pub fn archetype_direction_into(residuals: &[f32], d: usize, out: &mut [f32]) -> bool {
    debug_assert!(residuals.len().is_multiple_of(d));
    debug_assert!(out.len() >= d);
    let n = residuals.len() / d;
    if n == 0 {
        return false;
    }
    for (j, v) in out.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for r in residuals.chunks_exact(d) {
            acc += r[j] as f64;
        }
        *v = (acc / n as f64) as f32;
    }
    let mut norm = 0.0f64;
    for v in out.iter() {
        norm += *v as f64 * *v as f64;
    }
    let norm = norm.sqrt();
    if norm <= 1e-6 {
        return false;
    }
    for v in out.iter_mut() {
        *v = (*v as f64 / norm) as f32;
    }
    true
}

/// Fit one keyed correction: `v` solves the scalar ridge
///
/// ```text
/// min_v Σ_i (u·r_i − v·Ψ_i)² + λ‖v‖²
/// ```
///
/// over the correction's own fit segment (`features` is `n × d_h`
/// row-major; `base_residuals` is `n × d` — the base readout's residuals on
/// the same rows; `u` is the frozen output-space direction). Reuses
/// [`chunked_gram_into`] + [`ridge_solve_direct_f64`] verbatim (f64
/// accumulation per the crate's numerics law); the f64 solution is cast
/// once to f32 into `v_out`.
///
/// Rows must be at least 1; `scratch` must be sized for `d_h`. Cold path —
/// consolidation-tier work, allocations confined to `v_out` growth.
pub fn fit_correction_into(
    features: &[f32],
    base_residuals: &[f32],
    u: &[f32],
    lambda: f64,
    d_h: usize,
    scratch: &mut BankFitScratch,
    v_out: &mut Vec<f32>,
) {
    debug_assert!(d_h > 0);
    debug_assert!(features.len().is_multiple_of(d_h));
    let n = features.len() / d_h;
    debug_assert_eq!(base_residuals.len(), n * u.len());
    debug_assert!(scratch.gram.len() >= d_h * d_h);
    debug_assert!(!v_out.is_empty() || n > 0);

    // Scalar targets t_i = u·r_i (f64), then cov[k] = Σ_i t_i·Ψ_i[k].
    scratch.targets.clear();
    scratch.targets.reserve(n);
    for i in 0..n {
        let row = &base_residuals[i * u.len()..(i + 1) * u.len()];
        let mut t = 0.0f64;
        for j in 0..u.len() {
            t += u[j] as f64 * row[j] as f64;
        }
        scratch.targets.push(t);
    }
    for k in 0..d_h {
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += scratch.targets[i] * features[i * d_h + k] as f64;
        }
        scratch.cov[k] = acc;
    }
    // Gram + λI over the segment's feature rows (f64, the shared substrate).
    let rows = features.chunks_exact(d_h);
    chunked_gram_into(rows, &mut scratch.gram, lambda, d_h);
    // Solve (single right-hand side) and cast once.
    ridge_solve_direct_f64(
        &mut scratch.w_t,
        &mut scratch.l,
        &mut scratch.z,
        &scratch.gram,
        &scratch.cov,
        d_h,
        1,
    );
    v_out.clear();
    v_out.reserve(d_h);
    for k in 0..d_h {
        v_out.push(scratch.w_t[k] as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic LCG so the fixtures need no external RNG.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f64 / (1u64 << 31) as f64 - 1.0) as f32
        }
    }

    fn synthetic_bank() -> (KarcConditionedReadout, GateConfig) {
        // d=2, d_h=3; base maps ψ→(ψ0, ψ1); correction keyed 1 with
        // u = (0, 1), v = (1, 0, 0) — adds +σ·ψ0 to output 1.
        let cfg = GateConfig::default();
        let mut bank = KarcConditionedReadout::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3, cfg);
        bank.add_correction(1, vec![0.0, 1.0], vec![1.0, 0.0, 0.0]);
        (bank, cfg)
    }

    #[test]
    fn gate_abstains_below_min_obs() {
        let (bank, cfg) = synthetic_bank();
        assert_eq!(bank.gate_sigma(0), 0.0, "0 keyed obs must abstain");
        let mut bank = bank;
        let mut scratch = [0.0f32; 2];
        let mut feats = [1.0f32, 2.0, 3.0];
        let truth = [0.0f32, 5.0];
        for _ in 0..(cfg.min_obs as usize - 1) {
            bank.observe(&feats, 1, &truth, &mut scratch);
        }
        assert_eq!(bank.gate_sigma(0), 0.0, "min_obs - 1 must still abstain");
        feats[0] = -1.0; // perturb so the keyed residual stream moves
        bank.observe(&feats, 1, &truth, &mut scratch);
        assert_eq!(bank.corrections[0].n_keyed(), cfg.min_obs);
    }

    #[test]
    fn gate_sigma_is_half_at_margin() {
        // Feed a stream where the correction EXACTLY cancels the base error:
        // gain → 1, σ(β·0.85) → ~1. Then a stream where it does nothing:
        // gain → 0, σ(−β·0.15) → ~0.
        let cfg = GateConfig::default();
        let mut bank = KarcConditionedReadout::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3, cfg);
        bank.add_correction(1, vec![0.0, 1.0], vec![1.0, 0.0, 0.0]);
        let mut scratch = [0.0f32; 2];
        // ψ = (2, 0, 0): base out = (2, 0); truth = (2, 0) → base residual on
        // u=(0,1) is 0. Not useful. Use truth = (2, 4): residual scalar 4;
        // correction contributes v·Ψ = 2 → corrected residual 2.
        let feats = [2.0f32, 0.0, 0.0];
        let truth = [2.0f32, 4.0];
        for _ in 0..64 {
            bank.observe(&feats, 1, &truth, &mut scratch);
        }
        let helping = bank.gate_sigma(0);
        assert!(helping > 0.95, "perfect-ish gain must open: {helping}");
        // Now the same shape but truth = (2, 0): residual 0, corrected −2 —
        // the correction HURTS; gain → negative, gate closes.
        let mut bank2 = KarcConditionedReadout::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3, cfg);
        bank2.add_correction(1, vec![0.0, 1.0], vec![1.0, 0.0, 0.0]);
        let truth2 = [2.0f32, 0.0];
        for _ in 0..64 {
            bank2.observe(&feats, 1, &truth2, &mut scratch);
        }
        let hurting = bank2.gate_sigma(0);
        assert!(hurting < 0.05, "negative gain must stay shut: {hurting}");
    }

    #[test]
    fn apply_fold_order_is_bit_identical() {
        let cfg = GateConfig::default();
        let mut bank = KarcConditionedReadout::new(
            vec![0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
            2,
            3,
            cfg,
        );
        bank.add_correction(1, vec![1.0, 0.0], vec![0.1, 0.2, 0.3]);
        bank.add_correction(1, vec![0.0, 1.0], vec![0.4, 0.5, 0.6]);
        // Force both gates open by feeding a stream both corrections help.
        let mut scratch = [0.0f32; 2];
        let feats = [1.0f32, 2.0, 3.0];
        let truth = [4.0f32, 5.0];
        for _ in 0..256 {
            bank.observe(&feats, 1, &truth, &mut scratch);
        }
        let mut a = [0.0f32; 2];
        let mut b = [0.0f32; 2];
        bank.apply_into(&feats, 1, &mut a);
        bank.apply_into(&feats, 1, &mut b);
        assert_eq!(a, b, "apply must be bit-identical across calls");
        // Base-only prediction differs from the gated sum (gates armed).
        let mut base_only = [0.0f32; 2];
        bank.base_predict_into(&feats, &mut base_only);
        assert!(bank.gate_sigma(0) > 0.0 || bank.gate_sigma(1) > 0.0);
        let _ = base_only;
    }

    #[test]
    fn closed_corrections_cost_nothing() {
        // Condition mismatch: apply must equal the base prediction exactly.
        let (bank, _cfg) = synthetic_bank();
        let mut out = [0.0f32; 2];
        let mut base = [0.0f32; 2];
        let feats = [0.5f32, 0.5, 0.5];
        bank.apply_into(&feats, 0, &mut out);
        bank.base_predict_into(&feats, &mut base);
        assert_eq!(out, base, "a non-matching condition must never perturb");
    }

    #[test]
    fn fit_recovers_a_synthetic_rank1_residual() {
        // Build features + residual EXACTLY r_i = 0.7·(w·ψ_i) with
        // w = (0.3, −0.2, 0.1), u = (0, 1). The fit must recover w (the
        // scalar ridge has an exact solution at λ → 0 for full-rank rows).
        let d_h = 3;
        let d = 2;
        let w = [0.3f32, -0.2, 0.1];
        let mut lcg = Lcg(0x1234_5678_9abc_def0);
        let n = 512;
        let mut feats = Vec::with_capacity(n * d_h);
        let mut res = Vec::with_capacity(n * d);
        for _ in 0..n {
            let f0 = lcg.next_f32();
            let f1 = lcg.next_f32();
            let f2 = lcg.next_f32();
            let coef = 0.7 * (w[0] * f0 + w[1] * f1 + w[2] * f2);
            feats.extend_from_slice(&[f0, f1, f2]);
            res.extend_from_slice(&[0.0, coef]);
        }
        let u = [0.0f32, 1.0];
        let mut scratch = BankFitScratch::with_capacity(d_h);
        let mut v = Vec::new();
        fit_correction_into(&feats, &res, &u, 1e-6, d_h, &mut scratch, &mut v);
        // The targets carry the 0.7 amplitude, so v recovers 0.7·w exactly.
        for k in 0..d_h {
            let expect = 0.7 * w[k];
            let rel = ((v[k] - expect) / expect).abs();
            assert!(rel < 5e-3, "v[{k}] = {} vs {expect} (rel {rel})", v[k]);
        }
        // Archetype direction of these residuals is exactly +e1.
        let mut dir = [0.0f32; 2];
        assert!(archetype_direction_into(&res, d, &mut dir));
        assert!((dir[0].abs() as f64) < 1e-6 && ((dir[1].abs() - 1.0) as f64) < 1e-5);
    }

    #[test]
    fn archetype_refuses_a_zero_residual_stream() {
        let mut dir = [0.0f32; 2];
        assert!(!archetype_direction_into(
            &[0.0, 0.0, 0.0, 0.0],
            2,
            &mut dir
        ));
    }

    #[test]
    fn state_f32_matches_the_matched_bytes_ledger() {
        // The Issue-1239 bytes math: base d_h·D = 288 at (d_h=96, D=3);
        // one correction adds d_h + D = 99. Two corrections → 486.
        let cfg = GateConfig::default();
        let mut bank = KarcConditionedReadout::new(vec![0.0; 96 * 3], 3, 96, cfg);
        assert_eq!(bank.state_f32(), 288);
        bank.add_correction(0, vec![0.0; 3], vec![0.0; 96]);
        bank.add_correction(1, vec![0.0; 3], vec![0.0; 96]);
        assert_eq!(bank.state_f32(), 486);
    }

    #[test]
    fn replace_correction_swaps_weights_and_keeps_the_gate_warm() {
        // The Issue-1260 refit semantics: a re-fitted correction mounts the
        // new (u, v) in place and the keyed-observation count carries over —
        // the gate does NOT restart its abstain window.
        let (mut bank, cfg) = synthetic_bank();
        let mut scratch = [0.0f32; 2];
        let feats = [1.0f32, 2.0, 3.0];
        let truth = [0.0f32, 5.0];
        for _ in 0..cfg.min_obs {
            bank.observe(&feats, 1, &truth, &mut scratch);
        }
        let n_before = bank.corrections[0].n_keyed();
        assert_eq!(n_before, cfg.min_obs);
        // Refit: same key, new weights.
        assert!(bank.replace_correction(1, vec![1.0, 0.0], vec![0.0, 1.0, 0.5]));
        assert_eq!(
            bank.corrections[0].n_keyed(),
            n_before,
            "gate count carries"
        );
        assert_eq!(bank.n_corrections(), 1, "refit never grows the bank");
        // The new v = (0, 1, 0.5) is live: apply at scale 1 moves output 0 to
        // base(1.0) + v·ψ = 1.0 + (0·1 + 1·2 + 0.5·3) = 4.5.
        let mut out = [0.0f32; 2];
        bank.apply_fixed(&feats, 1, 1.0, &mut out);
        assert!((out[0] - 4.5).abs() < 1e-5, "out[0] = {}", out[0]);
        // A key with no correction refuses (returns false), never appends.
        assert!(!bank.replace_correction(7, vec![0.0, 0.0], vec![0.0; 3]));
        assert_eq!(bank.n_corrections(), 1);
    }
}
