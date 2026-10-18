//! Calibrated sigmoid gate — Platt-style refit for decision/confidence scalars.
//!
//! Every decision/confidence scalar in the stack is a sigmoid output, and its
//! calibration is proven nowhere: Lean proves boundedness (range (0,1));
//! nothing proves the numbers *mean* anything ("fear = 0.8" is not known to
//! fire ~80% of the time). This module adds the missing, published, modelless
//! half: a per-direction two-parameter refit (temperature + bias) over
//! recorded `(sigmoid_output, binary_outcome)` pairs — Platt 1999; Guo et al.
//! 2017 (arXiv:1706.04599); Kadavath 2022 (arXiv:2207.05221).
//!
//! Modelless mandate: the ONLY mutated state is two latent scalars per
//! direction plus the evidence window — mutation class #3 (a direction
//! vector + sigmoid gate updated from labeled outcomes), never base weights.
//! The fit is a deterministic convex 2-parameter Newton solve (fixed
//! iteration count, smoothed targets à la Platt), not gradient descent on a
//! network; replaying the same pairs yields bit-identical parameters.
//!
//! Shape (Issue 810):
//! - `observe(p, outcome)` — record one pair into a fixed-capacity FIFO ring
//!   (zero alloc after construction).
//! - `refit()` — off-hot-path refit of `(w, c)` in `p_cal = sigmoid(w·z + c)`
//!   where `z = logit(p)`; equivalent to the issue's
//!   `sigmoid((logit(p) − b)/T)` with `T = 1/w`, `b = −c/w`.
//! - `apply(p)` — hot path: one logit, one fma, one sigmoid. Zero-alloc;
//!   the same cost class as any bare sigmoid gate.
//! - `commitment()` — BLAKE3 over versioned canonical bytes (params + total
//!   observation count), the small-artifact convention of
//!   `closure::commitment`; freeze/thaw consumers treat it like any
//!   snapshot.
//!
//! Monotonicity guard: `w > 0` is enforced (projection to `W_MIN`) —
//! calibration must never reorder decisions. Over the reals that also
//! preserves ranking and every threshold-optimal decision (G3); in f32 it
//! is carried by TWO further mechanisms (Issues 909–911): the
//! resolution-aware `w` floor (a shipped map must resolve the window's own
//! score gaps — distinct scores stay distinct in f32) and the
//! zero-tolerance AUC guard (`refit` refuses any fit whose map drops the
//! evidence window's own AUC; identity keeps the raw ordering). Out-of-
//! window saturation is the consumer's re-measure (reflex Bench 095: 15/15
//! suites preserved).
//!
//! Solver (Issue 910): the Lin–Lin–Weng 2007 form of Platt's solve —
//! base-rate start `(0, ln((n⁺+1)/(n⁻+1)))` + Armijo-backtracked Newton.
//! Platt's original undamped full step from the identity start is the
//! documented flaw: on a narrow-z window the first Hessian is near-singular
//! and the step lands the iterate in a saturated corner where the Hessian
//! collapses (the Issue-909 stall, measured 10.8×/23× above the achievable
//! loss on the real reflex windows; an f64 mirror stalls identically — not
//! a precision defect).
//!
//! UQ "Report the Floor" rule (Research 322 / Plan 340): a calibrated gate
//! must still beat the dumb baselines — the constant base-rate predictor and
//! the uncalibrated sigmoid itself — on Brier/log-loss (G2). Calibration is
//! not free: if the refit cannot beat those floors on the evidence window,
//! the honest answer is the identity transform, which the occupancy floor
//! makes the default until evidence says otherwise.
//!
//! Calibrated sigmoid gate (Issue 810) — Platt-style temperature/bias refit
//! for decision/confidence scalars. Opt-in (`sigmoid_calibration`).

/// Canonical sigmoid (per the project rule: sigmoid, never softmax).
use crate::sigmoid;

/// Probability clip for logit/log-loss numerics: keeps `logit` finite
/// (|logit| ≤ ~9.2) without measurably moving any decision threshold.
const P_CLIP: f32 = 1.0e-4;
/// Monotonicity floor for the fitted scale `w` (1/T). A window whose MLE
/// scale is anti-correlated (`w ≤ 0`) is projected here: the gate keeps a
/// valid (rank-preserving) shape instead of inverting decisions.
pub const W_MIN: f32 = 1.0e-3;
/// Newton iteration cap for the 2-parameter refit. Convex problem, quadratic
/// convergence — 32 is far past convergence and exists only to bound the
/// off-hot-path refit deterministically.
const NEWTON_MAX_ITERS: usize = 32;
/// Convergence tolerance on the parameter-update norm.
const NEWTON_TOL: f32 = 1.0e-7;
/// Armijo backtracking constants (Lin–Lin–Weng 2007): accept a step of
/// length `t` when `L(θ+tΔ) ≤ L(θ) + ARMIJO_C·t·∇L·Δ`; halve up to
/// [`LS_MAX_HALVINGS`] times. Platt's original undamped full step is the
/// known flaw this fixes (Issue 910): from the identity start on a
/// narrow-z window, the first Hessian is near-singular and the full step
/// lands the iterate in a saturated corner where the Hessian collapses —
/// the loop then breaks at a point up to ~15× the achievable loss.
const ARMIJO_C: f64 = 1e-4;
const LS_MAX_HALVINGS: usize = 32;
/// Ranking-guard slack (Issues 909–911): PURE float-arithmetic headroom for
/// the rank-sum comparison — NOT a statistical tolerance. The
/// resolution-aware `w` floor (see [`resolution_w_floor`]) makes an accepted
/// fit's mapped window keep every distinct raw score distinct in f32, so
/// the cal ordering IS the raw ordering and a real AUC drop cannot pass;
/// the guard is the enforced invariant, not a filter. (The earlier SE
/// tolerance form admitted ~0.08 AUC of tie loss at banking77's minority
/// size — the same order as the harm the module set out to fix — and is
/// recorded in Issue 911.)
const RANKING_AUC_TOLERANCE: f64 = 1e-9;
/// The ulp margin the resolution floor guarantees per minimum distinct-gap
/// (2 ulps: one for the arithmetic, one for the rounding of `w·z+c`).
const RESOLUTION_FLOOR_ULPS: f32 = 2.0;
/// Cap on the resolution floor — a saturated corner the cap cannot resolve
/// falls to the zero-tolerance guard (and from there to identity).
const RESOLUTION_FLOOR_MAX: f32 = 1.0;

/// The resolution-aware `w` floor (Issue 911): the minimum scale at which
/// the map `sigmoid(w·z + c)` still resolves the window's own score gaps in
/// f32 — `k·ulp(q̄) / (q̄(1−q̄)·Δz_min)`, where `q̄` is the mapped level and
/// `Δz_min` the smallest gap between DISTINCT window logits. Below this,
/// adjacent scores tie after mapping and every tie is pure ranking harm (a
/// monotone map gains nothing by merging). Near-constant fits are clamped
/// UP to the floor (the loss is flat there); steep MLEs sit far above it
/// untouched. Returns [`W_MIN`] when the window has no distinct gaps to
/// resolve.
fn resolution_w_floor(z: &[f32], c: f32) -> f32 {
    let mut sorted = z.to_vec();
    // Issue 940 (NaN-hardening): total_cmp, not partial_cmp().unwrap() — a NaN
    // input sorts deterministically (NaN-last) instead of aborting the fit.
    sorted.sort_unstable_by(f32::total_cmp);
    sorted.dedup();
    if sorted.len() <= 1 {
        return W_MIN;
    }
    let dz_min = sorted
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f32::INFINITY, f32::min);
    if !dz_min.is_finite() || dz_min <= 0.0 {
        return W_MIN;
    }
    let q = sigmoid(c).clamp(1e-6, 1.0 - 1e-6);
    let ulp = (f32::from_bits(q.to_bits() + 1) - q).max(f32::MIN_POSITIVE);
    (RESOLUTION_FLOOR_ULPS * ulp / (q * (1.0 - q) * dz_min)).clamp(W_MIN, RESOLUTION_FLOOR_MAX)
}
/// Commitment format version (bump on any change to the canonical bytes or
/// to the fit behavior — v2: the LLW solver fix, Issue 910; the same
/// evidence window now produces different (correct) parameters).
const COMMITMENT_VERSION: u8 = 2;

/// A single-direction calibrated sigmoid gate.
///
/// Single allocation at construction (the evidence window); `observe` and
/// `apply` are zero-alloc. Not `Clone` (evidence windows are
/// identity-carrying; snapshot via params + commitment instead).
pub struct SigmoidGateCalibrator {
    /// Fitted scale `w` in `p_cal = sigmoid(w·z + c)` (w > 0; 1/T).
    w: f32,
    /// Fitted intercept `c`.
    c: f32,
    /// Evidence window: raw sigmoid outputs (clamped on use).
    ps: Vec<f32>,
    /// Evidence window: binary outcomes as 0.0/1.0.
    ys: Vec<f32>,
    /// Next write index (FIFO ring).
    head: usize,
    /// Number of live entries (`≤ capacity`).
    len: usize,
    /// Total observations ever recorded (monotone; part of the commitment
    /// so a stale snapshot cannot pose as fresher evidence).
    n_obs_total: u64,
    /// Minimum window occupancy before `refit()` will move parameters.
    min_obs: usize,
}

impl SigmoidGateCalibrator {
    /// Construct an identity-calibrated gate (`T = 1, b = 0` — `apply` is
    /// the identity until evidence says otherwise).
    ///
    /// `capacity` is the evidence window size (FIFO eviction of the oldest
    /// pair); `min_obs` is the occupancy floor below which [`Self::refit`]
    /// keeps the identity parameters.
    pub fn new(capacity: usize, min_obs: usize) -> Self {
        assert!(capacity >= 1, "capacity must be >= 1");
        Self {
            w: 1.0,
            c: 0.0,
            ps: Vec::with_capacity(capacity),
            ys: Vec::with_capacity(capacity),
            head: 0,
            len: 0,
            n_obs_total: 0,
            min_obs: min_obs.min(capacity),
        }
    }

    /// Record one `(sigmoid_output, binary_outcome)` pair.
    ///
    /// `p` outside `(0, 1)` is invalid and clamps into `(P_CLIP, 1 −
    /// P_CLIP)` on use — the gate treats it as saturation, not error.
    /// Zero-alloc; O(1).
    #[inline]
    pub fn observe(&mut self, p: f32, outcome: bool) {
        let y = f32::from(outcome);
        let cap = self.ps.capacity();
        if self.len < cap {
            self.ps.push(p);
            self.ys.push(y);
            self.len += 1;
            self.head = if cap == 0 { 0 } else { (self.head + 1) % cap };
        } else {
            self.ps[self.head] = p;
            self.ys[self.head] = y;
            self.head = (self.head + 1) % cap;
        }
        self.n_obs_total += 1;
    }

    /// Hot path: calibrated probability for a raw sigmoid output.
    ///
    /// `p_cal = sigmoid(w · logit(p) + c)` — one logit, one fma, one
    /// sigmoid. Monotone in `p` for every reachable parameter state
    /// (`w ≥ W_MIN > 0`), so the ORDER is preserved by construction; the
    /// f32-resolution half of that guarantee (distinct scores staying
    /// distinct) is carried by the resolution floor + the zero-tolerance
    /// AUC guard in [`Self::refit`].
    ///
    /// Identity fast path: at the constructed parameters `(w, c) = (1, 0)`
    /// the input is returned **bit-identically** — a `logit → sigmoid`
    /// roundtrip is not bit-exact in f32, and the documented contract
    /// ("apply is the identity until evidence says otherwise") plus the
    /// cold-start bit-identity requirements of downstream consumers
    /// (riir-ai Issue 964 C1/C3) demand the exact value, not a float twin.
    #[inline]
    pub fn apply(&self, p: f32) -> f32 {
        if self.w == 1.0 && self.c == 0.0 {
            return p;
        }
        sigmoid(self.w.mul_add(logit_clamped(p), self.c))
    }

    /// Off-hot-path refit of `(w, c)` over the evidence window.
    ///
    /// Deterministic solve on the smoothed-target logistic loss (Platt
    /// 1999): labels `t⁺ = (n⁺+1)/(n⁺+2)` / `t⁻ = 1/(n⁻+2)` keep the
    /// problem strictly convex on separable windows. The solver is the
    /// Lin–Lin–Weng 2007 form (base-rate start + Armijo-backtracked
    /// Newton — Platt's undamped identity-init iteration stalls in a
    /// saturated corner on narrow-z windows; Issues 909/910), under a
    /// three-candidate loss-argmin net (see [`fit_window`]).
    /// Identity parameters below `min_obs` occupancy. Returns `true` when
    /// the window was big enough to fit (parameters may still land on
    /// identity — refused by the saturation guard below).
    ///
    /// Monotonicity: every candidate carries `w > 0`, and if the
    /// unconstrained MLE wants `w ≤ 0`, the fit is projected — `w` pinned
    /// to [`W_MIN`] and `c` re-solved on the 1-D problem. The gate never
    /// inverts decisions from evidence.
    ///
    /// Output-side ranking guard (Issues 909–911 / reflex Issue 056): a fit
    /// that DEGRADES the evidence window's own ranking quality — the mapped
    /// confidences' rank-averaged AUC against the outcomes dropping below
    /// the raw confidences' AUC at all — is refused and the parameters stay
    /// identity. The resolution floor makes an accepted fit's mapped window
    /// keep every distinct raw score distinct in f32 (the cal ordering IS
    /// the raw ordering), so any real drop is a defect; identity keeps the
    /// raw ordering (and `apply`'s bit-identity fast path).
    pub fn refit(&mut self) -> bool {
        let n = self.len;
        if n < self.min_obs.max(2) {
            return false;
        }
        let (t, z) = self.smoothed_pairs();
        // `fit_window` resolves each candidate against the resolution floor
        // (Issue 911) before the loss-argmin, so the winner's map keeps the
        // window's distinct scores distinct in f32.
        let (w, c) = fit_window(&t, &z);
        if (w, c) != (1.0, 0.0) && !window_preserves_ranking(&z, &t, w, c) {
            self.w = 1.0;
            self.c = 0.0;
            return true;
        }
        self.w = w;
        self.c = c;
        true
    }

    /// Smoothed Platt targets + logits for the current window, oldest first
    /// (deterministic order; allocation is fine — refit is off-hot-path).
    fn smoothed_pairs(&self) -> (Vec<f32>, Vec<f32>) {
        let n = self.len;
        let cap = self.ps.capacity();
        let n_pos: u32 = self.ys[..n].iter().map(|&y| (y > 0.5) as u32).sum();
        let n_neg = n as u32 - n_pos;
        // Platt 1999 target smoothing — keeps the loss strictly convex on
        // separable windows (all-0 / all-1), where the raw MLE diverges.
        let t_pos = (n_pos as f32 + 1.0) / (n_pos as f32 + 2.0);
        let t_neg = 1.0 / (n_neg as f32 + 2.0);
        let mut ts = Vec::with_capacity(n);
        let mut zs = Vec::with_capacity(n);
        for i in 0..n {
            let idx = (self.head + cap - n + i) % cap;
            ts.push(if self.ys[idx] > 0.5 { t_pos } else { t_neg });
            zs.push(logit_clamped(self.ps[idx]));
        }
        (ts, zs)
    }

    /// Current parameters as `(T, b)` — the issue's parametrization
    /// `p_cal = sigmoid((logit(p) − b)/T)`: `T = 1/w`, `b = −c/w`.
    /// `T > 0` always (the monotonicity guard).
    pub fn params(&self) -> (f32, f32) {
        (1.0 / self.w, -self.c / self.w)
    }

    /// Raw `(w, c)` fitted parameters (the stored form).
    pub fn params_raw(&self) -> (f32, f32) {
        (self.w, self.c)
    }

    /// Live window occupancy.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` iff no recorded pairs.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total observations ever recorded (monotone counter).
    #[inline]
    pub fn n_obs_total(&self) -> u64 {
        self.n_obs_total
    }

    /// BLAKE3 commitment over versioned canonical bytes: version, `w`, `c`,
    /// window capacity, total observation count. Deterministic; changes on
    /// every observation (via `n_obs_total`) and every refit that moves
    /// parameters — a frozen snapshot can be checked against the evidence
    /// it claims to summarize.
    pub fn commitment(&self) -> [u8; 32] {
        let mut bytes = [0u8; 21];
        bytes[0] = COMMITMENT_VERSION;
        bytes[1..5].copy_from_slice(&self.w.to_bits().to_le_bytes());
        bytes[5..9].copy_from_slice(&self.c.to_bits().to_le_bytes());
        let cap = self.ps.capacity() as u32;
        bytes[9..13].copy_from_slice(&cap.to_le_bytes());
        bytes[13..21].copy_from_slice(&self.n_obs_total.to_le_bytes());
        blake3::hash(&bytes).into()
    }
}

/// Solve for the intercept `c` alone at a pinned scale `w` (1-D convex
/// Newton; used by the monotonicity projection).
fn solve_c_at_w(t: &[f32], z: &[f32], w: f32) -> f32 {
    let mut c = 0.0f32;
    for _ in 0..NEWTON_MAX_ITERS {
        let (g, h) = t
            .iter()
            .zip(z)
            .fold((0.0f32, 0.0f32), |(g, h), (&ti, &zi)| {
                let p = sigmoid(w.mul_add(zi, c));
                (g + p - ti, h + p * (1.0 - p))
            });
        if h < f32::EPSILON {
            break;
        }
        let dc = -g / h;
        c += dc;
        if !c.is_finite() {
            return 0.0;
        }
        if dc * dc < NEWTON_TOL * NEWTON_TOL {
            break;
        }
    }
    c
}

/// Deterministic damped 2-parameter Newton solve on the smoothed-target
/// logistic loss, including the monotonicity projection — the solver half
/// of [`SigmoidGateCalibrator::refit`].
///
/// The Lin–Lin–Weng 2007 form of Platt's solver (the fix for Platt's
/// original undamped iteration): start at the base-rate point
/// `(0, ln((n⁺+1)/(n⁻+1)))` and take Newton steps under Armijo
/// backtracking, so every accepted step strictly reduces the loss and a
/// near-singular Hessian can no longer catapult the iterate into a
/// saturated corner (the Issue-909/910 stall, measured 10.8×/23× above
/// the achievable loss on the real reflex windows). Extracted from
/// `refit` so the Issue-909 audit can compare the production solve
/// against an f64 mirror on the same window. Identity-safe: `w > 0` on
/// return (the `W_MIN` projection).
fn solve_smoothed_mle(t: &[f32], z: &[f32]) -> (f32, f32) {
    // LLW init: the base-rate start (never the identity — the identity's
    // first Hessian is the near-singular one on narrow bands).
    let n_pos = t.iter().filter(|&&ti| ti > 0.5).count() as f32;
    let n_neg = t.len() as f32 - n_pos;
    let (mut w, mut c) = (0.0f32, ((n_pos + 1.0) / (n_neg + 1.0)).ln());
    for _ in 0..NEWTON_MAX_ITERS {
        let (g0, g1, h00, h01, h11) = grad_hess(t, z, w, c);
        let det = h00 * h11 - h01 * h01;
        if !det.is_finite() || det.abs() < f32::EPSILON {
            break;
        }
        let inv_det = 1.0 / det;
        let dw = -inv_det * (h11 * g0 - h01 * g1);
        let dc = -inv_det * (h00 * g1 - h01 * g0);
        // Armijo backtracking (loss to MINIMIZE): accept the first step
        // length meeting the sufficient-decrease condition; a step that
        // cannot satisfy it after LS_MAX_HALVINGS ends the solve (the
        // iterate is loss-monotone by construction).
        let l0 = smoothed_loss(t, z, w, c);
        let slope = (g0 * dw + g1 * dc) as f64; // < 0 for a descent direction
        let mut step = 1.0f32;
        let mut accepted = false;
        for _ in 0..LS_MAX_HALVINGS {
            let l_try = smoothed_loss(t, z, w + step * dw, c + step * dc);
            if l_try <= l0 + ARMIJO_C * (step as f64) * slope {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
        w += step * dw;
        c += step * dc;
        if !w.is_finite() || !c.is_finite() {
            w = W_MIN;
            c = 0.0;
            break;
        }
        if (step * dw) * (step * dw) + (step * dc) * (step * dc) < NEWTON_TOL * NEWTON_TOL {
            break;
        }
    }
    // Monotonicity projection: an anti-correlated window degrades to a
    // near-constant monotone gate rather than inverting decisions.
    // (`is_nan` is explicit — a NaN scale must project, not skip.)
    if w.is_nan() || w <= W_MIN {
        w = W_MIN;
        c = solve_c_at_w(t, z, w);
    }
    (w, c)
}

/// The smoothed logistic loss (the solver's own objective) with f64
/// accumulation — the Issue-909 candidate-comparison evaluator. Isolates
/// PARAMETER quality from f32 accumulation noise; deterministic.
fn smoothed_loss(t: &[f32], z: &[f32], w: f32, c: f32) -> f64 {
    t.iter().zip(z).fold(0.0f64, |acc, (&ti, &zi)| {
        let q = (sigmoid(w.mul_add(zi, c)) as f64).clamp(1e-12, 1.0 - 1e-12);
        acc - (ti as f64 * q.ln() + (1.0 - ti as f64) * (1.0 - q).ln())
    })
}

/// Apply the resolution floor to one candidate (Issue 911): below the
/// floor, clamp `w` up and re-solve the intercept at the clamped scale
/// (the loss is flat in `w` near constant maps, so the cost is negligible;
/// steep candidates pass through untouched).
fn apply_resolution_floor(t: &[f32], z: &[f32], w: f32, c: f32) -> (f32, f32) {
    if w >= RESOLUTION_FLOOR_MAX {
        return (w, c);
    }
    let floor = resolution_w_floor(z, c);
    if w >= floor {
        return (w, c);
    }
    (floor, solve_c_at_w(t, z, floor))
}

/// The refit net (Issue 909/910): the loss-argmin over three deterministic
/// candidates, UNDER the Lin–Lin–Weng solver.
///
/// The LLW solve (base-rate start + Armijo backtracking) reaches the
/// smoothed MLE where Platt's undamped iteration stalled in a saturated
/// corner at up to ~15× the achievable loss (measured on the real reflex
/// windows — see the audit test). The argmin net underneath exists because
/// the solver's stop conditions (det-break, halving budget, iteration cap)
/// are all legitimate-but-imperfect: any residual suboptimality falls back
/// to the better of:
///
/// 1. the LLW solve result (optimal wherever the solve converged — the
///    measured case on all three fixture windows, including a sharp
///    T=0.115 fit on the banking77 band the stall used to destroy);
/// 2. the near-constant base-rate candidate `(W_MIN, solve_c_at_w(W_MIN))`
///    — the Report-the-Floor floor, only when the window genuinely carries
///    no z-usable signal (a fit that cannot beat it is not a fit);
/// 3. identity `(1, 0)` — keep the raw readout when even the constant map
///    loses to it.
///
/// All three are monotone (`w > 0`), so G3's ranking guarantee holds for
/// every choice; the saturation guard in [`SigmoidGateCalibrator::refit`]
/// then enforces the f32-resolution half of the promise.
fn fit_window(t: &[f32], z: &[f32]) -> (f32, f32) {
    let (w0, c0) = solve_smoothed_mle(t, z);
    let (mut w, mut c) = apply_resolution_floor(t, z, w0, c0);
    let mut best = smoothed_loss(t, z, w, c);
    let c_const = solve_c_at_w(t, z, W_MIN);
    let (w_const, c_const) = apply_resolution_floor(t, z, W_MIN, c_const);
    let l_const = smoothed_loss(t, z, w_const, c_const);
    if l_const < best {
        w = w_const;
        c = c_const;
        best = l_const;
    }
    let l_identity = smoothed_loss(t, z, 1.0, 0.0);
    if l_identity < best {
        w = 1.0;
        c = 0.0;
    }
    (w, c)
}

/// Rank-averaged AUC (Mann–Whitney) with ties credited ½: `P(score₊ >
/// score₋) + ½·P(=)` over every pos/neg pair. Deterministic; `f64::NAN`
/// for a single-class set (the caller skips the guard there — no ranking
/// claim to protect).
fn auc_rank_average(keys: &[f32], ys: &[bool]) -> f64 {
    let n_pos = ys.iter().filter(|y| **y).count();
    let n_neg = ys.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    // Issue 940 (NaN-hardening): total_cmp, not partial_cmp().unwrap() — a NaN
    // key sorts deterministically instead of aborting the AUC computation.
    idx.sort_unstable_by(|&a, &b| keys[a].total_cmp(&keys[b]));
    let mut ranks = vec![0.0f64; keys.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && keys[idx[j + 1]] == keys[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    let sum_pos: f64 = ranks
        .iter()
        .zip(ys)
        .filter(|(_, y)| **y)
        .map(|(r, _)| r)
        .sum();
    let (n_pos, n_neg) = (n_pos as f64, n_neg as f64);
    (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

/// Ranking-guard predicate (Issues 909/910 / reflex Issue 056): does the
/// fitted map `sigmoid(w·z + c)` preserve the evidence window's own ranking
/// quality — the mapped AUC staying at or above the raw AUC (within
/// [`RANKING_AUC_TOLERANCE`])? A monotone map can only tie-merge (never
/// reorder); a merge is harm exactly when it merges an informative pair,
/// which the AUC drop measures directly. The Platt-stall class (every
/// window value mapped into one f32 bucket) reads as cal-AUC ≈ ½ vs a
/// raw AUC ≫ ½ — refused; a signal-free window's honest constant map ties
/// everything with BOTH AUCs at chance — allowed. Single-class windows
/// (AUC undefined) pass vacuously.
fn window_preserves_ranking(z: &[f32], t: &[f32], w: f32, c: f32) -> bool {
    let ys: Vec<bool> = t.iter().map(|&ti| ti > 0.5).collect();
    let auc_raw = auc_rank_average(z, &ys);
    if auc_raw.is_nan() {
        return true;
    }
    let mapped: Vec<f32> = z.iter().map(|&zi| sigmoid(w.mul_add(zi, c))).collect();
    let auc_cal = auc_rank_average(&mapped, &ys);
    // Zero tolerance (Issue 911): with the resolution floor every accepted
    // fit keeps distinct scores distinct, so the cal ordering IS the raw
    // ordering — any real drop is a defect, refused. The slack is pure
    // float arithmetic on the rank sums, never a statistical allowance.
    auc_cal >= auc_raw - RANKING_AUC_TOLERANCE
}

/// Gradient and Hessian (loss convention: log-loss to MINIMIZE) of the
/// smoothed logistic loss at `(w, c)`:
/// `L = −Σ [t·ln σ(s) + (1−t)·ln(1−σ(s))]`, `s = w·z + c`.
///
/// Returns `(∂L/∂w, ∂L/∂c, ∂²L/∂w², ∂²L/∂w∂c, ∂²L/∂c²)` — the Hessian of
/// the logistic loss is positive semidefinite, so the Newton step
/// `p ← p − H⁻¹g` descends.
fn grad_hess(t: &[f32], z: &[f32], w: f32, c: f32) -> (f32, f32, f32, f32, f32) {
    t.iter().zip(z).fold(
        (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32),
        |(g0, g1, h00, h01, h11), (&ti, &zi)| {
            let p = sigmoid(w.mul_add(zi, c));
            let d = p - ti; // ∂L/∂s
            let r = p * (1.0 - p); // ∂²L/∂s²
            (
                g0 + d * zi,
                g1 + d,
                h00 + r * zi * zi,
                h01 + r * zi,
                h11 + r,
            )
        },
    )
}

/// `ln(p/(1−p))` with `p` clamped into `(P_CLIP, 1−P_CLIP)`.
#[inline]
fn logit_clamped(p: f32) -> f32 {
    let p = p.clamp(P_CLIP, 1.0 - P_CLIP);
    (p / (1.0 - p)).ln()
}

/// Fixed-size bank of independently calibrated gates — the consumer shape
/// for multi-scalar surfaces (the 5 affect scalars, a verifier panel).
pub struct CalibratedGateSet<const N: usize> {
    gates: [SigmoidGateCalibrator; N],
}

impl<const N: usize> CalibratedGateSet<N> {
    /// Construct `N` identity-calibrated gates with one shared config.
    pub fn new(capacity: usize, min_obs: usize) -> Self {
        Self {
            gates: std::array::from_fn(|_| SigmoidGateCalibrator::new(capacity, min_obs)),
        }
    }

    /// Record one pair for direction `dir` (zero-alloc).
    #[inline]
    pub fn observe(&mut self, dir: usize, p: f32, outcome: bool) {
        self.gates[dir].observe(p, outcome);
    }

    /// Calibrated probability for direction `dir` (zero-alloc hot path).
    #[inline]
    pub fn apply(&self, dir: usize, p: f32) -> f32 {
        self.gates[dir].apply(p)
    }

    /// Refit one direction; returns `false` below its `min_obs`.
    pub fn refit(&mut self, dir: usize) -> bool {
        self.gates[dir].refit()
    }

    /// Refit every direction; returns the count that moved.
    pub fn refit_all(&mut self) -> usize {
        self.gates
            .iter_mut()
            .map(|g| g.refit())
            .filter(|moved| *moved)
            .count()
    }

    /// Borrow one direction's calibrator.
    pub fn gate(&self, dir: usize) -> &SigmoidGateCalibrator {
        &self.gates[dir]
    }
}

impl<const N: usize> Default for CalibratedGateSet<N> {
    fn default() -> Self {
        Self::new(256, 32)
    }
}

// ── Metrics (public substrate for gates and consumers) ─────────────────────

/// Brier score: mean `(p − y)²`. Lower is better.
pub fn brier_score(predictions: &[f32], outcomes: &[f32]) -> f32 {
    debug_assert_eq!(predictions.len(), outcomes.len());
    let n = predictions.len().max(1) as f32;
    predictions
        .iter()
        .zip(outcomes)
        .map(|(p, y)| (p - y).powi(2))
        .sum::<f32>()
        / n
}

/// Mean log-loss with probabilities clipped to `[P_CLIP, 1 − P_CLIP]`.
/// Lower is better.
pub fn log_loss(predictions: &[f32], outcomes: &[f32]) -> f32 {
    debug_assert_eq!(predictions.len(), outcomes.len());
    let n = predictions.len().max(1) as f32;
    predictions
        .iter()
        .zip(outcomes)
        .map(|(p, y)| {
            let p = p.clamp(P_CLIP, 1.0 - P_CLIP);
            let y = y.clamp(0.0, 1.0);
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        })
        .sum::<f32>()
        / n
}

/// Binned expected calibration error (equal-width bins over `[0, 1]`):
/// `ECE = Σ_b (n_b/N)·|mean_p_b − mean_y_b|`. Lower is better.
pub fn expected_calibration_error(predictions: &[f32], outcomes: &[f32], bins: usize) -> f32 {
    debug_assert_eq!(predictions.len(), outcomes.len());
    debug_assert!(bins >= 1);
    let n = predictions.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum_p = vec![0.0f32; bins];
    let mut sum_y = vec![0.0f32; bins];
    let mut cnt = vec![0u32; bins];
    for (p, y) in predictions.iter().zip(outcomes) {
        let b = ((p.clamp(0.0, 1.0) * bins as f32) as usize).min(bins - 1);
        sum_p[b] += p;
        sum_y[b] += *y;
        cnt[b] += 1;
    }
    (0..bins)
        .filter(|&b| cnt[b] > 0)
        .map(|b| {
            let c = cnt[b] as f32;
            c / n as f32 * (sum_p[b] / c - sum_y[b] / c).abs()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift* stream — no unseeded global RNG (Issue 809).
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed.max(1))
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn f01(&mut self) -> f32 {
            // Top 53 bits → uniform [0, 1).
            (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
        }
        /// Approx standard-normal via centered Irwin–Hall (4 uniforms).
        fn normal(&mut self) -> f32 {
            (self.f01() + self.f01() + self.f01() + self.f01() - 2.0) * 1.2247
        }
        fn bernoulli(&mut self, p: f32) -> bool {
            self.f01() < p
        }
    }

    /// The G1/G2 fixture: a gate whose raw sigmoid is systematically
    /// overconfident relative to the truth
    /// `p_true = sigmoid(1.6·z − 0.4)` (recoverable `T = 0.625, b = 0.25`).
    fn miscalibrated_corpus(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = Rng::new(0xC0FF_EE01);
        let mut p_raw = Vec::with_capacity(n);
        let mut p_true = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let z = rng.normal();
            p_raw.push(sigmoid(z));
            let pt = sigmoid(1.6_f32.mul_add(z, -0.4));
            p_true.push(pt);
            let yi = rng.bernoulli(pt);
            y.push(f32::from(yi));
        }
        (p_raw, p_true, y)
    }

    /// G1 — decision-level calibration: fit on the train half, halve ECE on
    /// the eval half AND land ≤ 0.05; recover the planted (T, b).
    #[test]
    fn g1_ece_halves_and_lands_under_005() {
        let (p_raw, _p_true, y) = miscalibrated_corpus(4096);
        let split = p_raw.len() / 2;
        let mut cal = SigmoidGateCalibrator::new(4096, 32);
        for i in 0..split {
            cal.observe(p_raw[i], y[i] > 0.5);
        }
        assert!(cal.refit(), "window above min_obs must refit");

        let raw_eval: Vec<f32> = p_raw[split..].to_vec();
        let cal_eval: Vec<f32> = raw_eval.iter().map(|&p| cal.apply(p)).collect();
        let y_eval: Vec<f32> = y[split..].to_vec();

        let ece_raw = expected_calibration_error(&raw_eval, &y_eval, 15);
        let ece_cal = expected_calibration_error(&cal_eval, &y_eval, 15);
        let (t_fit, b_fit) = cal.params();
        eprintln!(
            "  G1  ECE(raw)={ece_raw:.4}  ECE(cal)={ece_cal:.4}  T={t_fit:.4} b={b_fit:.4} (planted T=0.625 b=0.25)"
        );
        assert!(
            ece_cal < ece_raw,
            "ECE must improve: raw {ece_raw:.4} -> cal {ece_cal:.4}"
        );
        assert!(
            ece_cal < ece_raw / 2.0,
            "ECE must at least halve: raw {ece_raw:.4} -> cal {ece_cal:.4}"
        );
        assert!(ece_cal <= 0.05, "ECE_cal {ece_cal:.4} must land <= 0.05");

        // Planted-transform recovery: T ≈ 1/1.6 = 0.625, b ≈ 0.4/1.6 = 0.25.
        let (t, b) = cal.params();
        assert!(
            (0.42..=0.84).contains(&t),
            "temperature {t:.4} should recover ~0.625"
        );
        assert!(
            (0.02..=0.48).contains(&b),
            "bias {b:.4} should recover ~0.25"
        );
    }

    /// G2 — Report the Floor: Brier/log-loss must beat BOTH dumb baselines
    /// (the constant base-rate predictor and the uncalibrated sigmoid).
    #[test]
    fn g2_beats_base_rate_and_uncalibrated_floors() {
        let (p_raw, _p_true, y) = miscalibrated_corpus(4096);
        let split = p_raw.len() / 2;
        let base_rate = y[..split].iter().sum::<f32>() / split as f32;

        let mut cal = SigmoidGateCalibrator::new(4096, 32);
        for i in 0..split {
            cal.observe(p_raw[i], y[i] > 0.5);
        }
        assert!(cal.refit());

        let raw_eval: Vec<f32> = p_raw[split..].to_vec();
        let cal_eval: Vec<f32> = raw_eval.iter().map(|&p| cal.apply(p)).collect();
        let floor_eval = vec![base_rate; raw_eval.len()];
        let y_eval: Vec<f32> = y[split..].to_vec();

        let ll = |p: &[f32]| log_loss(p, &y_eval);
        let br = |p: &[f32]| brier_score(p, &y_eval);
        eprintln!(
            "  G2  logloss cal={:.4} raw={:.4} floor={:.4} | brier cal={:.4} raw={:.4} floor={:.4}",
            ll(&cal_eval),
            ll(&raw_eval),
            ll(&floor_eval),
            br(&cal_eval),
            br(&raw_eval),
            br(&floor_eval)
        );
        assert!(
            ll(&cal_eval) < ll(&raw_eval),
            "log-loss must beat uncalibrated: {} vs {}",
            ll(&cal_eval),
            ll(&raw_eval)
        );
        assert!(
            ll(&cal_eval) < ll(&floor_eval),
            "log-loss must beat base-rate floor: {} vs {}",
            ll(&cal_eval),
            ll(&floor_eval)
        );
        assert!(
            br(&cal_eval) < br(&raw_eval),
            "Brier must beat uncalibrated: {} vs {}",
            br(&cal_eval),
            br(&raw_eval)
        );
        assert!(
            br(&cal_eval) < br(&floor_eval),
            "Brier must beat base-rate floor: {} vs {}",
            br(&cal_eval),
            br(&floor_eval)
        );
    }

    /// G3 — ranking + decision invariance: monotone transform preserves the
    /// argsort, the best-threshold accuracy, and moves the fire rate TOWARD
    /// the truth (not away).
    #[test]
    fn g3_ranking_and_decisions_preserved() {
        let (p_raw, p_true, y) = miscalibrated_corpus(4096);
        let split = p_raw.len() / 2;
        let mut cal = SigmoidGateCalibrator::new(4096, 32);
        for i in 0..split {
            cal.observe(p_raw[i], y[i] > 0.5);
        }
        assert!(cal.refit());

        // (a) Monotone on a spread fixture (no f32 ties at this spacing).
        let mut prev = f32::MIN;
        for i in 0..200 {
            let p = 0.002 + 0.996 * i as f32 / 199.0;
            let q = cal.apply(p);
            assert!(q >= prev, "apply must be non-decreasing (p={p})");
            prev = q;
        }

        // (b) Best-threshold decision accuracy identical (monotone ⇒ the
        // achievable partition set is unchanged).
        let eval: Vec<f32> = p_raw[split..].to_vec();
        let y_eval: Vec<f32> = y[split..].to_vec();
        let best_acc = |probs: &[f32]| {
            let mut pts: Vec<f32> = probs.to_vec();
            pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            pts.dedup();
            let mut best = 0.0f32;
            for &thr in &pts {
                let acc = probs
                    .iter()
                    .zip(&y_eval)
                    .map(|(p, y)| {
                        let fire = *p >= thr;
                        f32::from(fire == (*y > 0.5))
                    })
                    .sum::<f32>()
                    / probs.len() as f32;
                best = best.max(acc);
            }
            best
        };
        let cal_eval: Vec<f32> = eval.iter().map(|&p| cal.apply(p)).collect();
        let a_raw = best_acc(&eval);
        let a_cal = best_acc(&cal_eval);
        assert!(
            (a_raw - a_cal).abs() < 1.0 / eval.len() as f32,
            "optimal-threshold accuracy must be unchanged: {a_raw} vs {a_cal}"
        );

        // (c) Fire rate at the calibrated 0.5 threshold moves toward the
        // truth's own fire rate (ABSTAIN band shrinks, not explodes).
        let rate = |f: &dyn Fn(usize) -> bool| -> f32 {
            (0..eval.len()).map(|i| f32::from(f(i))).sum::<f32>() / eval.len() as f32
        };
        let true_rate = rate(&|i| p_true[split + i] >= 0.5);
        let raw_rate = rate(&|i| eval[i] >= 0.5);
        let cal_rate = rate(&|i| cal_eval[i] >= 0.5);
        assert!(
            (cal_rate - true_rate).abs() <= (raw_rate - true_rate).abs(),
            "fire-rate error must not grow: raw |{:.4}| cal |{:.4}| vs true {true_rate:.4}",
            (raw_rate - true_rate).abs(),
            (cal_rate - true_rate).abs()
        );
    }

    /// G4 — the observe/apply hot loop is allocation-free (Issue-741
    /// predicate: runs in dev AND under `--release --features
    /// alloc_tracking`).
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    #[test]
    fn g4_alloc_free_observe_apply() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};
        let mut cal = SigmoidGateCalibrator::new(64, 8);
        for i in 0..64 {
            let p = 0.05 + 0.9 * (i % 13) as f32 / 13.0;
            cal.observe(p, i % 3 == 0);
        }
        cal.refit();
        reset_alloc_stats();
        for i in 0..1000 {
            let p = 0.05 + 0.9 * (i % 17) as f32 / 17.0;
            cal.observe(p, i % 2 == 0);
            let _ = cal.apply(p);
        }
        let (count, _bytes) = get_alloc_stats();
        assert_eq!(count, 0, "observe+apply must be zero-alloc, saw {count}");
    }

    /// Determinism (G-adjacent): identical streams → bit-identical params
    /// and commitment; refit changes the commitment.
    #[test]
    fn deterministic_fit_and_commitment() {
        let (p_raw, _p_true, y) = miscalibrated_corpus(512);
        let mut a = SigmoidGateCalibrator::new(512, 32);
        let mut b = SigmoidGateCalibrator::new(512, 32);
        for i in 0..512 {
            a.observe(p_raw[i], y[i] > 0.5);
            b.observe(p_raw[i], y[i] > 0.5);
        }
        let pre_refit = a.commitment();
        assert_eq!(a.refit(), b.refit());
        assert_eq!(a.params_raw(), b.params_raw());
        assert_eq!(a.commitment(), b.commitment());
        assert_ne!(
            a.commitment(),
            pre_refit,
            "refit must move the commitment (params changed)"
        );
    }

    /// Monotonicity guard: an anti-correlated window (high p ⇒ y=0, low p ⇒
    /// y=1) must project to `W_MIN` instead of inverting decisions.
    #[test]
    fn anti_correlated_window_projects_to_w_min() {
        let mut cal = SigmoidGateCalibrator::new(64, 8);
        for i in 0..64 {
            let p = 0.02 + 0.96 * i as f32 / 63.0;
            cal.observe(p, p < 0.5); // y=1 exactly when p is LOW
        }
        assert!(cal.refit());
        let (w, c) = cal.params_raw();
        assert!(
            w >= W_MIN && w.is_finite() && c.is_finite(),
            "projection must pin w to W_MIN: got ({w}, {c})"
        );
        if w == W_MIN {
            // Near-constant monotone gate: outputs finite, non-decreasing.
            let mut prev = f32::MIN;
            for i in 0..50 {
                let p = 0.01 + 0.98 * i as f32 / 49.0;
                let q = cal.apply(p);
                assert!(q.is_finite() && q >= prev);
                prev = q;
            }
        }
        let (t, _b) = cal.params();
        assert!(t > 0.0, "T must stay positive under projection");
    }

    /// Separable window (all outcomes true): Platt smoothing keeps the fit
    /// finite instead of diverging.
    #[test]
    fn pure_window_smoothing_stays_finite() {
        let mut cal = SigmoidGateCalibrator::new(64, 8);
        for i in 0..64 {
            cal.observe(0.02 + 0.96 * i as f32 / 63.0, true);
        }
        assert!(cal.refit());
        let (w, c) = cal.params_raw();
        assert!(w.is_finite() && w > 0.0 && c.is_finite());
        assert!(cal.apply(0.7).is_finite());
    }

    /// Occupancy floor: below `min_obs` the gate stays identity.
    #[test]
    fn below_min_obs_stays_identity() {
        let mut cal = SigmoidGateCalibrator::new(256, 32);
        for i in 0..16 {
            cal.observe(0.3 + 0.001 * i as f32, i % 2 == 0);
        }
        assert!(!cal.refit(), "below min_obs refit must refuse");
        let (t, b) = cal.params();
        assert_eq!((t, b), (1.0, 0.0));
        assert!((cal.apply(0.3) - 0.3).abs() < 1e-6, "identity apply");
    }

    /// Cold start is bit-identical, not merely close: `new`'s documented
    /// contract ("apply is the identity until evidence says otherwise") is
    /// an exact-value contract. A `logit → sigmoid` roundtrip is NOT
    /// bit-exact in f32, so the fast path must return the input unchanged —
    /// downstream consumers commit/sync the raw sigmoid output through the
    /// cold-start gate (riir-ai Issue 964 C1/C3 bit-identity requirements).
    #[test]
    fn cold_start_apply_is_bit_identical() {
        let cal = SigmoidGateCalibrator::new(64, 8);
        // Saturated ends included: the clamp only exists on the slow path.
        for i in 0..254 {
            let p = (i + 1) as f32 / 255.0;
            assert_eq!(cal.apply(p).to_bits(), p.to_bits(), "p = {p}");
        }
    }

    /// A below-`min_obs` refit leaves the bit-identity fast path armed —
    /// observations recorded but no evidence acted on yet.
    #[test]
    fn below_min_obs_apply_stays_bit_identical() {
        let mut cal = SigmoidGateCalibrator::new(256, 32);
        for i in 0..16 {
            cal.observe(0.25 + 0.02 * i as f32, i % 3 == 0);
        }
        assert!(!cal.refit());
        for i in 0..254 {
            let p = (i + 1) as f32 / 255.0;
            assert_eq!(cal.apply(p).to_bits(), p.to_bits(), "p = {p}");
        }
    }

    /// FIFO eviction: capacity 4 under 6 observations keeps the last 4 and
    /// the monotone total.
    #[test]
    fn fifo_eviction_keeps_newest() {
        let mut cal = SigmoidGateCalibrator::new(4, 2);
        for i in 0..6 {
            cal.observe(0.1 * i as f32, i % 2 == 0);
        }
        assert_eq!(cal.len(), 4);
        assert_eq!(cal.n_obs_total(), 6);
        // Window holds p ∈ {0.2, 0.3, 0.4, 0.5}; the fit must consume those,
        // not the evicted {0.0, 0.1} — observable via refit success.
        assert!(cal.refit());
    }

    /// The bank delegates per-direction (the 5-affect-scalar shape).
    #[test]
    fn gate_set_delegates_per_direction() {
        let mut set = CalibratedGateSet::<5>::new(128, 16);
        let mut rng = Rng::new(7);
        for _ in 0..64 {
            let p = sigmoid(rng.normal());
            set.observe(0, p, rng.bernoulli(p));
            // Direction 1 deliberately anti-correlated with direction 0's p.
            set.observe(1, p, rng.bernoulli(1.0 - p));
            set.observe(4, 0.9, true);
        }
        // Gates 0/1/4 carry 64 observations each; 2/3 were never observed
        // (identity, refit refuses below min_obs).
        assert_eq!(set.refit_all(), 3);
        let (t0, _b0) = set.gate(0).params();
        let (t1, _b1) = set.gate(1).params();
        assert!(t0 > 0.0 && t1 > 0.0);
        // Direction 4 is separable-positive: smoothing keeps it finite.
        assert!(set.apply(4, 0.9).is_finite());
        // Directions 2/3 never observed: still identity.
        assert!((set.apply(2, 0.25) - 0.25).abs() < 1e-6);
        assert!(!set.refit(3));
    }

    /// Metric helpers on known vectors.
    #[test]
    fn metrics_known_vectors() {
        let p = vec![0.8, 0.2, 0.6];
        let y = vec![1.0, 0.0, 0.0];
        assert!((brier_score(&p, &y) - ((0.04 + 0.04 + 0.36) / 3.0)).abs() < 1e-6);
        // Perfect prediction ⇒ zero log-loss (after clip, ~0), zero ECE.
        let perfect = vec![1.0, 0.0, 0.0];
        assert!(log_loss(&perfect, &y) < 1e-3);
        assert_eq!(expected_calibration_error(&perfect, &y, 10), 0.0);
        // Uniform-vs-0.5 outcomes: ECE = |0.5 − ȳ| style check via bins.
        let uni = vec![0.5, 0.5, 0.5, 0.5];
        let y2 = vec![1.0, 0.0, 1.0, 0.0];
        assert!((expected_calibration_error(&uni, &y2, 10) - 0.0).abs() < 1e-6);
    }

    // ── Issue 909 / reflex Issue 056: the output-side saturation guard ──
    //
    // The fixtures are the EXACT production cal windows the reflex engine
    // fitted on (dumped via RIIR_DEBUG_CAL_WINDOW at the deployed posture,
    // 2026-09-30; same pairs → same FIFO window → same params — the replay
    // is verified against the reflex-measured temperatures below).

    /// Load a dumped cal window (`<suite>.window`: `conf correct` per line).
    fn load_window_fixture(name: &str) -> Vec<(f32, bool)> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/cal_saturation_909/"
        );
        let text = std::fs::read_to_string(format!("{path}{name}.window"))
            .unwrap_or_else(|e| panic!("fixture {name}: {e}"));
        text.lines()
            .map(|l| {
                let mut it = l.split_whitespace();
                let p: f32 = it.next().unwrap().parse().unwrap();
                let y: u8 = it.next().unwrap().parse().unwrap();
                (p, y == 1)
            })
            .collect()
    }

    /// Replay a window through a fresh calibrator at the deployed config
    /// (`cal_capacity 512, cal_min_obs 64` — reflex `EngineConfig::default`).
    fn replay_window(pairs: &[(f32, bool)]) -> SigmoidGateCalibrator {
        let mut cal = SigmoidGateCalibrator::new(512, 64);
        for &(p, y) in pairs {
            cal.observe(p, y);
        }
        cal.refit();
        cal
    }

    /// The reflex-measured collapse class on the REAL windows — at the LLW
    /// solver the fits are the TRUE smoothed MLEs (the Issue-910 review's
    /// reference optima, reproduced by production): banking77 recovers a
    /// SHARP temperature (T≈0.115, w≈8.67) over the narrow band the Platt
    /// stall used to destroy — the band carries real signal (raw window
    /// AUC 0.85); xnli_en gets its flat-but-informative T≈5.2 fit. Both are
    /// monotone AND untied in f32: the calibrated ranking key equals the raw
    /// one (the Issue-056 harm gone), and the fits beat the constant floor
    /// (Report-the-Floor at the window level — see the audit test).
    #[test]
    fn real_windows_get_the_true_mle_at_the_llw_solver() {
        for (suite, w_lo, w_hi) in [("banking77", 6.0f32, 11.0f32), ("xnli_en", 0.05f32, 0.6f32)] {
            let pairs = load_window_fixture(suite);
            assert!(pairs.len() >= 64, "{suite}: window must clear min_obs");
            let cal = replay_window(&pairs);
            let (w, c) = cal.params_raw();
            assert!(
                (w_lo..=w_hi).contains(&w),
                "{suite}: w={w} should be the reference MLE (≈{w_lo}..{w_hi})"
            );
            assert!(c.is_finite(), "{suite}: c finite");
            // Monotone + untied over the window's own distinct confs: the
            // Issue-056 harm (tie-collapse to index order) is gone.
            let mut confs: Vec<f32> = pairs.iter().map(|p| p.0).collect();
            confs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            confs.dedup();
            let mut mapped: Vec<f32> = confs.iter().map(|&p| cal.apply(p)).collect();
            let distinct_before = mapped.len();
            mapped.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            mapped.dedup();
            assert!(
                mapped.len() as f32 >= 0.5 * distinct_before as f32,
                "{suite}: guard must pass the LLW fit ({} distinct of {})",
                mapped.len(),
                distinct_before
            );
            let mut prev = f32::MIN;
            for &p in &confs {
                let q = cal.apply(p);
                assert!(q >= prev, "{suite}: apply must be non-decreasing");
                prev = q;
            }
        }
    }

    /// The sane control on the real data: massive_intent_en's fit (measured
    /// T=0.275 — the LLW solve converges to the same optimum the old
    /// identity-init Newton found on this well-conditioned window). The net
    /// must keep the solver result and the guard must not fire.
    #[test]
    fn llw_keeps_the_sane_massive_fit() {
        let pairs = load_window_fixture("massive_intent_en");
        let cal = replay_window(&pairs);
        let (w, c) = cal.params_raw();
        assert!((w, c) != (1.0, 0.0), "the sane fit must survive");
        assert!(
            (3.5..3.8).contains(&w),
            "the reference MLE w≈3.63 must hold"
        );
        let (t, _b) = cal.params();
        assert!((0.1..=1.0).contains(&t), "measured T≈0.275, got {t}");
    }

    /// Report-the-Floor at the WINDOW level (the Issue-910 review's held-out
    /// demand): fit on the first half of each real window, then evaluate
    /// log-loss + Brier on the held-out half against a constant base-rate
    /// map derived from the same fit half.
    ///
    /// The fitted map must never lose to the floor, and on the signal-rich
    /// banking77 band must strictly win.
    #[test]
    fn held_out_floor_comparison_on_real_windows() {
        for suite in ["banking77", "xnli_en", "massive_intent_en"] {
            let pairs = load_window_fixture(suite);
            let split = pairs.len() / 2;
            let (fit_half, eval_half) = pairs.split_at(split);
            let mut cal = SigmoidGateCalibrator::new(512, 64);
            for &(p, y) in fit_half {
                cal.observe(p, y);
            }
            cal.refit();
            let base = fit_half.iter().filter(|p| p.1).count() as f32 / fit_half.len() as f32;
            let ll = |pred: &dyn Fn(f32) -> f32| -> (f64, f64) {
                let (mut l, mut b) = (0.0f64, 0.0f64);
                for &(p, y) in eval_half {
                    let q = pred(p).clamp(1e-12, 1.0 - 1e-12) as f64;
                    let y = f64::from(y);
                    l -= y * q.ln() + (1.0 - y) * (1.0 - q).ln();
                    b += (q - y) * (q - y);
                }
                (l / eval_half.len() as f64, b / eval_half.len() as f64)
            };
            let (ll_fit, b_fit) = ll(&|p| cal.apply(p));
            let (ll_const, b_const) = ll(&|_| base);
            eprintln!(
                "  floor {suite}: fit ll={ll_fit:.4} brier={b_fit:.4} | constant ll={ll_const:.4} brier={b_const:.4} (base {base:.3})"
            );
            // Never worse than the floor (tie allowed — a genuinely
            // signal-free window falls back to it by construction).
            assert!(
                ll_fit <= ll_const + 1e-9,
                "{suite}: held-out log-loss {ll_fit} must not lose to the floor {ll_const}"
            );
            assert!(
                b_fit <= b_const + 1e-9,
                "{suite}: held-out Brier {b_fit} must not lose to the floor {b_const}"
            );
            if suite == "banking77" {
                assert!(
                    ll_fit < ll_const - 1e-3,
                    "banking77's signal-rich band must strictly beat the floor held-out ({ll_fit} vs {ll_const})"
                );
            }
        }
    }

    /// The synthetic property sweep (the Issue-910 review's demand): across
    /// band widths × base rates × window sizes, with and without planted
    /// z-signal, the refit (i) never loses to the constant base-rate floor
    /// on its own window, (ii) never tie-collapses the mapped window, and
    /// (iii) stays monotone. Deterministic RNG (no global draws).
    #[test]
    fn property_sweep_never_loses_the_floor_and_never_collapses() {
        let mut n_checked = 0;
        for band in [0.02f32, 0.1, 0.5, 2.0] {
            for base in [0.2f32, 0.5, 0.8] {
                for n in [64usize, 200] {
                    for signal in [0.0f32, 2.0] {
                        let mut rng = Rng::new(
                            0x910_0001
                                ^ (band.to_bits() as u64)
                                ^ (base.to_bits() as u64)
                                ^ (n as u64)
                                ^ (signal.to_bits() as u64),
                        );
                        let z0 = -3.0f32;
                        let mut cal = SigmoidGateCalibrator::new(512, 32);
                        for i in 0..n {
                            let z = z0 + band * (i as f32 + rng.f01()) / n as f32;
                            // base rate via an intercept; signal via the slope.
                            let p_true = sigmoid(
                                signal.mul_add(z - z0 - band / 2.0, (base / (1.0 - base)).ln()),
                            );
                            cal.observe(sigmoid(z), rng.bernoulli(p_true));
                        }
                        assert!(cal.refit(), "window must fit");
                        let (t, zvec) = cal.smoothed_pairs();
                        let (wf, cf) = cal.params_raw();
                        let l_fit = smoothed_loss(&t, &zvec, wf, cf);
                        // The floor: constant at the smoothed-target mean.
                        let mean_t = (t.iter().map(|&ti| ti as f64).sum::<f64>()) / t.len() as f64;
                        let l_const =
                            smoothed_loss(&t, &zvec, W_MIN, (mean_t / (1.0 - mean_t)).ln() as f32);
                        // The floor comparison allows the resolution
                        // floor's clamp cost (measured ≤ ~1e-5 relative on
                        // the tightest noise cells — the loss is flat in w
                        // near constant maps, but not infinitely so) — the
                        // stall class this guards against sat 10× above the
                        // floor, five orders past this bar.
                        assert!(
                            l_fit <= l_const * (1.0 + 1e-4),
                            "band={band} base={base} n={n} signal={signal}: fit {l_fit} must not lose to the floor {l_const}"
                        );
                        // Ranking preserved on the window itself (the guard
                        // invariant — distinct-count collapse WITHOUT AUC
                        // harm is the honest constant answer and is fine).
                        assert!(
                            window_preserves_ranking(&zvec, &t, wf, cf),
                            "band={band} base={base} n={n} signal={signal}: AUC degraded"
                        );
                        // Monotone over the sorted distinct z.
                        let mut sorted = zvec.clone();
                        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                        sorted.dedup();
                        let mut prev = f32::MIN;
                        for &zi in &sorted {
                            let q = sigmoid(wf.mul_add(zi, cf));
                            assert!(q >= prev, "monotone");
                            prev = q;
                        }
                        n_checked += 1;
                    }
                }
            }
        }
        assert!(
            n_checked >= 48,
            "the sweep must cover ≥48 cells (got {n_checked})"
        );
    }

    // ── Issue 909 repair 3: the Newton early-break audit ──

    fn sigmoid64(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn logit64(p: f64) -> f64 {
        let p = p.clamp(P_CLIP as f64, 1.0 - P_CLIP as f64);
        (p / (1.0 - p)).ln()
    }

    /// f64-mirrored Platt targets + logits for a window.
    fn smoothed_pairs64(pairs: &[(f32, bool)]) -> (Vec<f64>, Vec<f64>) {
        let n = pairs.len();
        let n_pos = pairs.iter().filter(|p| p.1).count() as f64;
        let n_neg = n as f64 - n_pos;
        let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
        let t_neg = 1.0 / (n_neg + 2.0);
        pairs
            .iter()
            .map(|&(p, y)| {
                let t = if y { t_pos } else { t_neg };
                (t, logit64(p as f64))
            })
            .unzip()
    }

    /// The smoothed logistic loss in f64 at arbitrary parameters — the
    /// audit's objective (isolates parameter quality from f32 accumulation).
    fn smoothed_loss64(pairs: &[(f32, bool)], w: f64, c: f64) -> f64 {
        let (ts, zs) = smoothed_pairs64(pairs);
        ts.iter()
            .zip(&zs)
            .map(|(&t, &z)| {
                let q = sigmoid64(w.mul_add(z, c)).clamp(1e-12, 1.0 - 1e-12);
                -(t * q.ln() + (1.0 - t) * (1.0 - q).ln())
            })
            .sum()
    }

    /// f64 mirror of the production LLW solve (`solve_smoothed_mle`
    /// structure verbatim: base-rate start, Armijo-backtracked Newton, det
    /// early-break, step-norm tolerance, non-finite guard, W_MIN projection
    /// with the 1-D c re-solve) — the Issue-910 audit REFERENCE optimum.
    fn solve_f64_reference(pairs: &[(f32, bool)]) -> (f64, f64) {
        let (t, z) = smoothed_pairs64(pairs);
        let grad_hess64 = |w: f64, c: f64| {
            t.iter().zip(&z).fold(
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64),
                |(g0, g1, h00, h01, h11), (&ti, &zi)| {
                    let p = sigmoid64(w.mul_add(zi, c));
                    let d = p - ti;
                    let r = p * (1.0 - p);
                    (
                        g0 + d * zi,
                        g1 + d,
                        h00 + r * zi * zi,
                        h01 + r * zi,
                        h11 + r,
                    )
                },
            )
        };
        let loss64 = |w: f64, c: f64| {
            t.iter().zip(&z).fold(0.0f64, |acc, (&ti, &zi)| {
                let q = sigmoid64(w.mul_add(zi, c)).clamp(1e-12, 1.0 - 1e-12);
                acc - (ti * q.ln() + (1.0 - ti) * (1.0 - q).ln())
            })
        };
        let solve_c_at_w64 = |w: f64| -> f64 {
            let mut c = 0.0f64;
            for _ in 0..NEWTON_MAX_ITERS {
                let (g, h) = t
                    .iter()
                    .zip(&z)
                    .fold((0.0f64, 0.0f64), |(g, h), (&ti, &zi)| {
                        let p = sigmoid64(w.mul_add(zi, c));
                        (g + p - ti, h + p * (1.0 - p))
                    });
                if h < f64::EPSILON {
                    break;
                }
                let dc = -g / h;
                c += dc;
                if !c.is_finite() {
                    return 0.0;
                }
                if dc * dc < (NEWTON_TOL as f64) * (NEWTON_TOL as f64) {
                    break;
                }
            }
            c
        };
        let n_pos = t.iter().filter(|&&ti| ti > 0.5).count() as f64;
        let n_neg = t.len() as f64 - n_pos;
        let (mut w, mut c) = (0.0f64, ((n_pos + 1.0) / (n_neg + 1.0)).ln());
        for _ in 0..NEWTON_MAX_ITERS {
            let (g0, g1, h00, h01, h11) = grad_hess64(w, c);
            let det = h00 * h11 - h01 * h01;
            if !det.is_finite() || det.abs() < f64::EPSILON {
                break;
            }
            let inv_det = 1.0 / det;
            let dw = -inv_det * (h11 * g0 - h01 * g1);
            let dc = -inv_det * (h00 * g1 - h01 * g0);
            let l0 = loss64(w, c);
            let slope = g0 * dw + g1 * dc;
            let mut step = 1.0f64;
            let mut accepted = false;
            for _ in 0..LS_MAX_HALVINGS {
                if loss64(w + step * dw, c + step * dc) <= l0 + ARMIJO_C * step * slope {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                break;
            }
            w += step * dw;
            c += step * dc;
            if !w.is_finite() || !c.is_finite() {
                w = W_MIN as f64;
                c = 0.0;
                break;
            }
            if (step * dw) * (step * dw) + (step * dc) * (step * dc)
                < (NEWTON_TOL as f64) * (NEWTON_TOL as f64)
            {
                break;
            }
        }
        if w.is_nan() || w <= W_MIN as f64 {
            w = W_MIN as f64;
            c = solve_c_at_w64(w);
        }
        (w, c)
    }

    /// The Issue-910 audit: the production LLW solve must reach the f64
    /// reference optimum on the REAL windows (within f32 noise), and the
    /// reference optimum must BEAT the constant base-rate floor wherever
    /// the window carries z-usable signal — the corrected adjudication from
    /// the Issue-909 review (the round-1 three-candidate fallback shipped
    /// the FLOOR on banking77 while the true MLE — a sharp T=0.115 fit over
    /// [0.235, 0.9995] — sat 26% below it in loss; raw AUC on that window
    /// is 0.85, the band carries real signal).
    #[test]
    fn llw_solve_reaches_the_reference_optimum_on_real_windows() {
        for suite in ["banking77", "xnli_en", "massive_intent_en"] {
            let pairs = load_window_fixture(suite);
            let mut cal = SigmoidGateCalibrator::new(512, 64);
            for &(p, y) in &pairs {
                cal.observe(p, y);
            }
            let (t32, z32) = cal.smoothed_pairs();
            let (w_prod, c_prod) = solve_smoothed_mle(&t32, &z32);
            let (w_ref, c_ref) = solve_f64_reference(&pairs);
            let l_prod = smoothed_loss64(&pairs, w_prod as f64, c_prod as f64);
            let l_ref = smoothed_loss64(&pairs, w_ref, c_ref);
            // The floor: constant at the smoothed-target mean.
            let n_pos = pairs.iter().filter(|p| p.1).count() as f64;
            let rate = n_pos / pairs.len() as f64;
            let l_const = smoothed_loss64(&pairs, W_MIN as f64, (rate / (1.0 - rate)).ln());
            eprintln!(
                "  audit {suite}: prod (w={w_prod:.4}, c={c_prod:.4}) loss={l_prod:.4} | ref (w={w_ref:.4}, c={c_ref:.4}) loss={l_ref:.4} | floor={l_const:.4}"
            );
            // (a) The production solve reaches the reference optimum
            // (both run the same LLW algorithm; only precision differs).
            assert!(
                l_prod <= l_ref + 1e-3 * l_ref.max(1.0),
                "{suite}: production loss {l_prod} must match the reference {l_ref}"
            );
            // (b) The reference optimum BEATS the constant floor wherever
            // the window carries signal — the banking77/xnli bands DO
            // (the round-1 "no z-usable signal" reading was wrong).
            assert!(
                l_ref < l_const,
                "{suite}: reference optimum {l_ref} must beat the constant floor {l_const}"
            );
            // (c) The chosen fit (production net) never loses to the floor
            // AND reaches the reference (the net is a no-op under the LLW
            // solver on these windows).
            let (wf, cf) = fit_window(&t32, &z32);
            let l_chosen = smoothed_loss64(&pairs, wf as f64, cf as f64);
            assert!(
                l_chosen <= l_prod + 1e-6,
                "{suite}: the net must keep the solver result (chosen {l_chosen} vs solve {l_prod})"
            );
        }
    }

    /// The ranking-guard predicate directly (the wiring is exercised via
    /// `refit` above; this pins the classifier on synthetic windows):
    /// (a) an informative window whose map ties everything away — refused
    ///     (cal AUC collapses to chance while raw carries signal);
    /// (b) a signal-free window's constant map — ALLOWED (both AUCs at
    ///     chance; ties destroyed nothing) — the case the distinct-count
    ///     form of this guard wrongly refused (Issue 910);
    /// (c) a monotone untied map over the informative window — allowed.
    #[test]
    fn ranking_guard_classifies_harm_not_ties() {
        // (a) informative narrow band: upper half positive.
        let z: Vec<f32> = (0..64).map(|i| -3.1 + 0.05 * i as f32 / 63.0).collect();
        let y_info: Vec<bool> = (0..64).map(|i| i >= 32).collect();
        // Saturated-at-1.0 map: every output exactly 1.0 in f32 → cal AUC
        // collapses to chance while raw separates perfectly (AUC 1.0).
        assert!(!window_preserves_ranking(
            &z,
            &y_flags_from(&y_info),
            2.0,
            34.5
        ));
        // (c) gentle monotone map: AUC preserved.
        assert!(window_preserves_ranking(
            &z,
            &y_flags_from(&y_info),
            2.0,
            6.5
        ));
        // (b) signal-free: outcomes alternate by sorted position → raw AUC
        // exactly ½; the constant map ties everything → cal AUC ½ → allowed.
        let y_noise: Vec<bool> = (0..64).map(|i| i % 2 == 0).collect();
        assert!(window_preserves_ranking(
            &z,
            &y_flags_from(&y_noise),
            2.0,
            34.5
        ));
        // Single-class window: vacuous pass (AUC undefined).
        let y_all: Vec<bool> = vec![true; 64];
        assert!(window_preserves_ranking(
            &z,
            &y_flags_from(&y_all),
            2.0,
            34.5
        ));
    }

    /// The resolution floor (Issue 911) — ties stopped at their source: on a
    /// pure-noise narrow band (where the honest fit IS near-constant and the
    /// pre-911 distinct-count guard forced identity at 2.3× the floor's
    /// loss), the floored fit keeps EVERY distinct raw score distinct in
    /// f32 — the mapped distinct count equals the raw one exactly.
    #[test]
    fn resolution_floor_keeps_every_distinct_score() {
        let band = 0.02f32;
        let n = 200usize;
        let z0 = -3.0f32;
        let mut rng = Rng::new(0x911_0002);
        let mut cal = SigmoidGateCalibrator::new(512, 32);
        for i in 0..n {
            let z = z0 + band * (i as f32 + rng.f01()) / n as f32;
            cal.observe(sigmoid(z), rng.bernoulli(0.5));
        }
        assert!(cal.refit());
        let (t, zvec) = cal.smoothed_pairs();
        let mut sorted = zvec.clone();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        sorted.dedup();
        let mut mapped: Vec<f32> = sorted.iter().map(|&zi| cal.apply(sigmoid(zi))).collect();
        mapped.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        mapped.dedup();
        assert_eq!(
            mapped.len(),
            sorted.len(),
            "the floored near-constant fit must keep all {} distinct scores (got {})",
            sorted.len(),
            mapped.len()
        );
        // …and the zero-tolerance guard passes it.
        let (w, c) = cal.params_raw();
        assert!(window_preserves_ranking(&zvec, &t, w, c));
        // The loss stays at the floor (the clamp's cost is negligible).
        let l_fit = smoothed_loss(&t, &zvec, w, c);
        let mean_t = (t.iter().map(|&ti| ti as f64).sum::<f64>()) / t.len() as f64;
        let l_const = smoothed_loss(&t, &zvec, W_MIN, (mean_t / (1.0 - mean_t)).ln() as f32);
        assert!(
            l_fit <= l_const + 1.0,
            "the floored fit must stay near the constant floor ({l_fit} vs {l_const})"
        );
    }

    /// `window_preserves_ranking` takes the SMOOTHED targets as its outcome
    /// axis; the tests express windows as bools for clarity — encode them
    /// the way `refit` would (t_pos > 0.5 > t_neg).
    fn y_flags_from(ys: &[bool]) -> Vec<f32> {
        ys.iter().map(|&y| if y { 0.99 } else { 0.01 }).collect()
    }

    // ── Issue 940 (NaN-hardening): the calibration sorts never panic ──────

    #[test]
    fn resolution_w_floor_nan_input_sorts_instead_of_panicking() {
        // A NaN in the evidence window used to abort the fit
        // (partial_cmp().unwrap()); total_cmp sorts it deterministically and
        // `f32::min` ignores the NaN diff, so the floor computes from the
        // finite gaps.
        let w = resolution_w_floor(&[0.1, f32::NAN, 0.5, 0.7], 0.0);
        assert!(w >= W_MIN, "floor must stay at or above W_MIN, got {w}");
        // And an all-NaN window degrades to the floor, not a panic.
        assert_eq!(resolution_w_floor(&[f32::NAN; 4], 0.0), W_MIN);
    }

    #[test]
    fn auc_rank_average_nan_key_sorts_instead_of_panicking() {
        // A NaN key used to abort the AUC; total_cmp ranks it deterministically
        // (last) and the AUC computes.
        let auc = auc_rank_average(&[0.1, f32::NAN, 0.9], &[true, false, true]);
        assert!((0.0..=1.0).contains(&auc), "AUC out of range: {auc}");
    }
}
