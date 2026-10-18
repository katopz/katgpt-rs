//! Plan 581 — `twist_smc`: opaque-reward twisted-SMC steering + modelless
//! twist amortization.
//!
//! **Research 517** (CDM — Contrastive Distribution Matching for Amortized
//! SMC in Discrete Diffusion, [arXiv:2605.23346](https://arxiv.org/abs/2605.23346))
//! · **Plan 361** (riir-train trained-head counterpart — same GOAT gate,
//! different arm) · extends the shipped distributional-steering substrate
//! (Plan 577 / Bench 682) from closed-form measure-rewards to **opaque
//! black-box rewards**.
//!
//! # The mechanism (modelless slice of CDM)
//!
//! Steer a particle population toward `p* ∝ p·ψ` with `ψ ∝ exp(β·V̂)` where
//! `V̂ ≈ E[R(x₀)|x_t]` is estimated WITHOUT gradient descent:
//!
//! 1. **x̂₀ proxy** ([`X0ProxyReward`]) — the discrete-diffusion analog of
//!    Tweedie's shortcut: the denoiser already emits `p(x₀|x_t)`, so `r(x̂₀)`
//!    (1 reward query per particle-step) replaces the `M`-rollout
//!    Monte-Carlo twist estimate (the paper's ~50× cost).
//! 2. **State-keyed value memo** ([`ValueMemo`]) — resampled particles
//!    revisit prefixes; a `BLAKE3(state ‖ t)`-keyed cache means a resampled
//!    particle never re-queries the scorer.
//! 3. **One-shot ridge readout** ([`RidgeTwistTable`]) — a closed-form
//!    `(XᵀX + λI)⁻¹Xᵀy` fit over cached `(features, R)` pairs amortizes the
//!    twist to a dot product (zero reward queries) once fitted.
//! 4. **β / KL-budget selection** ([`select_beta_by_budget`]) — the
//!    anti-mode-collapse knob (R517 §1.5 / the paper's own DNA instability
//!    caveat): per-step β solved so the induced tilt stays inside a KL
//!    budget, reusing `entropic_tilt::solve_beta` (one implementation, two
//!    consumers — the hoist rule).
//!
//! # Consistency footing (No-GD advocate row 1)
//!
//! Self-normalized twisted SMC is consistent for ANY positive ψ — every
//! amortization here is **variance reduction, never correctness**. The β
//! drifts per step (budget-solved), which changes *which* tilted target the
//! population converges to, not whether the weighted measure is consistent.
//!
//! # Contract notes
//!
//! - **Weights-only steering is the default consumer shape** for opaque
//!   rewards: the twist reweights via the incremental ratio
//!   `log ψ_t − log ψ_{t−1}` ([`twist_step_into`]) and never needs `∇Ψ`
//!   (the FD-gradient path in `distributional_steering::ClosureReward`
//!   costs `2d` scorer evals per particle — documented there).
//! - **Determinism** (T3.4): papaya is read-path lock-free and iteration
//!   order never enters results — every result is a keyed lookup + fixed
//!   iteration-order arithmetic; two-run bit-identity is pinned by a test
//!   and the Bench 692 gate.
//! - **Finite discipline**: scorer outputs and V̂ values are `debug_assert`ed
//!   finite at every boundary (house `is_finite` discipline).
//! - **UQ discipline** (Plan 340 rule): the weighted measure is a
//!   ranking/steering signal; any future distribution/coverage claim must
//!   first beat the conformal-naive floor.
//!
//! twist_cache — Plan 581 opaque-reward twisted-SMC steering + modelless
//! twist amortization (Research 517, arXiv:2605.23346 CDM): x̂₀ posterior-mean
//! reward proxy (1 query/particle-step vs M rollouts), BLAKE3 state-keyed
//! value memo, one-shot ridge readout table, β/KL-budget selection
//! (entropic_tilt hoist rule). Consistency for any positive ψ — amortization
//! is variance reduction, never correctness. Opt-in; the trained-head
//! counterpart is riir-train Plan 361 (same GOAT gate, two arms — Bench 692).

use core::sync::atomic::{AtomicU64, Ordering};

use papaya::HashMap;

// ──────────────────────────────────────────────────────────────────────────
// T3.1 — state-keyed value memo
// ──────────────────────────────────────────────────────────────────────────

/// A cached twist value with its insertion tick (TTL bookkeeping).
#[derive(Debug, Clone, Copy)]
struct CachedValue {
    value: f32,
    tick: u32,
}

/// State-keyed value memo for the twist (Plan 581 T3.1 / Research 517
/// amortization row ii).
///
/// Keyed on `BLAKE3(state bytes ‖ t)` — the plan's literal key spec. A hit is
/// therefore an EXACT `(state, step)` replay: resampled duplicates within the
/// step (the dominant source), and persistent agents re-entering the same
/// state at the same step (episode replay). A different `t` is a different
/// entry by construction — freshness-correct values per step, and the
/// per-lookup staleness check is structurally unreachable (an entry's tick
/// always equals its key's `t`).
///
/// The `ttl` is the **eviction window**: at capacity pressure, entries older
/// than `t − ttl` are dropped FIRST (they can only ever be hit by a
/// replayed/older step), then a full clear — the deterministic simple policy.
/// Eviction is by predicate, never by iteration order (house rule).
pub struct ValueMemo {
    map: HashMap<[u8; 32], CachedValue>,
    cap: usize,
    ttl: u32,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ValueMemo {
    /// New memo with `cap` entry capacity and `ttl`-tick staleness window.
    pub fn new(cap: usize, ttl: u32) -> Self {
        assert!(cap > 0, "ValueMemo requires cap > 0");
        Self {
            map: HashMap::new(),
            cap,
            ttl,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Hit ⇒ cached value; miss ⇒ `compute()` + insert.
    ///
    /// The value is treated as a deterministic function of `(state, t)`
    /// (the memo contract); racing inserts converge to the same key.
    pub fn lookup_or_insert(&self, state: &[f32], t: u32, compute: impl FnOnce() -> f32) -> f32 {
        debug_assert!(
            state.iter().all(|v| v.is_finite()),
            "ValueMemo state must be finite"
        );
        let key = memo_key(state, t);
        let pinned = self.map.pin();
        if let Some(&cv) = pinned.get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return cv.value;
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let value = compute();
        debug_assert!(
            value.is_finite(),
            "ValueMemo compute must return a finite value"
        );
        if pinned.len() >= self.cap {
            let ttl = self.ttl;
            pinned.retain(|_, cv| t.saturating_sub(cv.tick) <= ttl);
            if pinned.len() >= self.cap {
                pinned.clear();
            }
        }
        let _ = pinned.insert(key, CachedValue { value, tick: t });
        value
    }

    /// Drop every entry (persistent-agent reset).
    pub fn clear(&self) {
        self.map.pin().clear();
    }

    /// Live entry count (approximate under concurrency).
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Entry count == 0 (approximate under concurrency).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cumulative hits (the Bench 692 memo-utility axis).
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Cumulative misses.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}

/// `BLAKE3(state bytes ‖ t le-bytes)` — bit-exact states at the same step
/// share an entry (the resampled-duplicate dedup axis).
fn memo_key(state: &[f32], t: u32) -> [u8; 32] {
    let bytes: &[u8] = bytemuck::cast_slice(state);
    let mut hasher = blake3::Hasher::new();
    hasher.update(bytes);
    hasher.update(&t.to_le_bytes());
    *hasher.finalize().as_bytes()
}

// ──────────────────────────────────────────────────────────────────────────
// T3.2 — one-shot ridge readout table
// ──────────────────────────────────────────────────────────────────────────

/// One-shot ridge twist table (Plan 581 T3.2): `log(1+R) ≈ w·f` fit by
/// closed-form normal equations over a cached `(features, R)` buffer —
/// deterministic, no iterations, zero reward queries at inference.
///
/// Reuses the crate's f64 Cholesky ridge substrate (`linalg::ridge_solve`
/// — the KARC/PEIRA `(N + λI)⁻¹` house pattern). The caller owns the
/// feature map (cheap deterministic per-state features, e.g. moments +
/// step fraction); the table is the twist readout.
///
/// **Extrapolation guard (Bench 692 finding):** a linear readout fit on
/// the collector's state support predicts monotonically past the observed
/// envelope — steering on the raw extrapolation runs the population off
/// the support (measured: downstream collapses to the reward-0 region).
/// The fit therefore records the observed target range and [`Self::value`]
/// clamps to it: outputs saturate at the collected max instead of
/// diverging. Within the collected range the readout is exact ridge.
#[derive(Debug, Clone)]
pub struct RidgeTwistTable {
    dim: usize,
    w: Vec<f32>,
    y_lo: f32,
    y_hi: f32,
}

impl RidgeTwistTable {
    /// Fit `w` minimizing `‖Xw − y‖² + λ‖w‖²` with `y = ln(1 + max(R, 0))`.
    ///
    /// Rewards are scores in the `R ≥ 0` domain (ln₁p clamp is a documented
    /// guard — shift negative-reward domains before calling). `features` is
    /// flat `N×d` row-major; a constant-1 feature row gives the fit an
    /// intercept. Cold path (one-shot fit).
    pub fn fit(features: &[f32], rewards: &[f32], dim: usize, lambda: f64) -> Self {
        let n = rewards.len();
        assert!(dim > 0, "RidgeTwistTable requires dim > 0");
        assert!(n >= dim, "ridge fit needs n ({n}) >= dim ({dim}) samples");
        assert!(
            features.len() == n * dim,
            "features.len() ({}) must equal n ({n}) * dim ({dim})",
            features.len()
        );
        let mut gram = vec![0.0f64; dim * dim];
        let mut cov = vec![0.0f64; dim];
        let mut y_lo = f32::INFINITY;
        let mut y_hi = f32::NEG_INFINITY;
        for (row, &r) in features.chunks_exact(dim).zip(rewards.iter()) {
            let y = (r.max(0.0) as f64).ln_1p();
            let y32 = y as f32;
            if y32 < y_lo {
                y_lo = y32;
            }
            if y32 > y_hi {
                y_hi = y32;
            }
            for (i, &fi) in row.iter().enumerate().take(dim) {
                let fid = fi as f64;
                cov[i] += fid * y;
                for (j, &fj) in row.iter().enumerate().take(dim).skip(i) {
                    gram[i * dim + j] += fid * fj as f64;
                }
            }
        }
        // Symmetrize the lower triangle + add the λI ridge.
        for i in 0..dim {
            for j in 0..i {
                gram[i * dim + j] = gram[j * dim + i];
            }
            gram[i * dim + i] += lambda;
        }
        let mut w_t = vec![0.0f64; dim];
        let mut l_scratch = vec![0.0f64; dim * dim];
        let mut z_scratch = vec![0.0f64; dim];
        crate::linalg::ridge_solve_direct_f64(
            &mut w_t,
            &mut l_scratch,
            &mut z_scratch,
            &gram,
            &cov,
            dim,
            1,
        );
        Self {
            dim,
            w: w_t.iter().map(|&v| v as f32).collect(),
            y_lo,
            y_hi,
        }
    }

    /// Twist readout `w·f`, clamped to the collected target range (the
    /// extrapolation guard — see the type doc). Zero reward queries.
    /// `features.len() >= dim`.
    pub fn value(&self, features: &[f32]) -> f32 {
        debug_assert!(features.len() >= self.dim, "feature row too short");
        let mut s = 0.0f32;
        for (i, &f) in features.iter().enumerate().take(self.dim) {
            s += self.w[i] * f;
        }
        s.clamp(self.y_lo, self.y_hi)
    }

    /// Feature dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Fitted weights (read-only; diagnostics / handoff).
    pub fn weights(&self) -> &[f32] {
        &self.w
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T3.3 — β / KL-budget selection + the twisted-SMC weight glue
// ──────────────────────────────────────────────────────────────────────────

/// KL-budgeted β for the twist `ψ ∝ exp(β·V̂)` (Plan 581 T3.3).
///
/// Delegates to `entropic_tilt::solve_beta` — one implementation, two
/// consumers (the hoist rule). The budget is the anti-mode-collapse knob
/// (R517 §1.5 / the CDM paper's own DNA-training instability caveat):
/// β = 0 recovers the base proposal, saturation pins the tilt to argmax.
#[must_use]
pub fn select_beta_by_budget(values: &[f32], kl_budget: f32) -> f32 {
    crate::entropic_tilt::solve_beta(values, kl_budget)
}

/// Incremental twisted-SMC weight update, weights-only (Plan 581).
///
/// `A_i += log ψ_t(x_t^i) − log ψ_{t−1}(slot)` with `ψ ∝ exp(β·V̂)` and β
/// solved per step under `kl_budget`. `prev_log_psi` carries the previous
/// step's centered log-ψ per slot (zeros at run start; reset by
/// [`twist_after_resample`]). Log-ψ is centered on the step max (`≤ 0`,
/// overflow-free); per-step centering constants cancel in every downstream
/// consumer (all weight reads are LSE-normalized).
///
/// **Scale invariance + the runaway guard (Bench 692 finding):** values are
/// span-normalized to `[0, 1]` before `solve_beta`, so the KL budget is a
/// statement about the TILT (scale-free), not about the raw value scale —
/// and a degenerate span (all values equal, e.g. every state clamped to the
/// same readout ceiling) produces β = 0 and NO tilt. Without the guard, a
/// saturated value range leaves only f32 round-off differences, β saturates
/// at the solver bracket, and the tilt amplifies round-off into a runaway
/// population collapse (measured in Bench 692: 12 resamples dragging every
/// particle to a reward-0 region).
///
/// **β drift caveat** (honest): β is re-solved each step, so without
/// resampling the effective target drifts mildly across steps. Consistency
/// is unaffected (any positive ψ works — see the module consistency
/// footing); with ESS-guarded resampling the ratio chain restarts each
/// resample anyway.
///
/// Scratch shape: `values.len() == log_weights.len() == prev_log_psi.len()`.
pub fn twist_step_into(
    values: &[f32],
    kl_budget: f32,
    log_weights: &mut [f32],
    prev_log_psi: &mut [f32],
    beta_out: &mut f32,
) {
    debug_assert_eq!(values.len(), log_weights.len());
    debug_assert_eq!(values.len(), prev_log_psi.len());
    debug_assert!(
        values.iter().all(|v| v.is_finite()),
        "twist values must be finite"
    );
    let mut v_min = f32::INFINITY;
    let mut v_max = f32::NEG_INFINITY;
    for &v in values {
        if v < v_min {
            v_min = v;
        }
        if v > v_max {
            v_max = v;
        }
    }
    // Degenerate-span guard: no signal → no tilt (β = 0). The epsilon is
    // scale-aware (relative to the magnitude in play).
    let span = v_max - v_min;
    let eps = 1e-5 * (1.0 + v_max.abs());
    if span <= eps {
        *beta_out = 0.0;
        return;
    }
    let inv_span = 1.0 / span;
    let beta = solve_beta_span_normalized(values, v_min, inv_span, kl_budget);
    *beta_out = beta;
    for i in 0..values.len() {
        // Span-normalized, centered on the step max: log ψ ∈ [−β, 0].
        let norm = (values[i] - v_min) * inv_span;
        let log_psi = beta * (norm - 1.0);
        log_weights[i] += log_psi - prev_log_psi[i];
        prev_log_psi[i] = log_psi;
    }
}

/// `solve_beta` over the span-normalized values (allocated scratch kept
/// local — cold relative to the reward-query cost it amortizes).
fn solve_beta_span_normalized(values: &[f32], v_min: f32, inv_span: f32, kl_budget: f32) -> f32 {
    let norm: Vec<f32> = values.iter().map(|&v| (v - v_min) * inv_span).collect();
    select_beta_by_budget(&norm, kl_budget)
}

/// Restart the twist ratio chain after a resample (the consumer resets
/// log-weights to uniform on the resampled population; the next
/// [`twist_step_into`] then reweights by `ψ_t` alone — a fresh one-shot
/// twist, still consistent for any positive ψ).
pub fn twist_after_resample(prev_log_psi: &mut [f32]) {
    prev_log_psi.fill(0.0);
}

/// Effective sample size of the LSE-normalized weights:
/// `(Σw)² / Σw²` — the resampling-guard diagnostic the Bench 692 gate
/// tracks. f64 accumulation; `log_weights` may be unnormalized.
#[must_use]
pub fn ess_from_log_weights(log_weights: &[f32]) -> f32 {
    let mut m = f64::NEG_INFINITY;
    for &l in log_weights {
        let ld = l as f64;
        if ld > m {
            m = ld;
        }
    }
    if m == f64::NEG_INFINITY || !m.is_finite() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &l in log_weights {
        let w = (l as f64) - m;
        let w = w.exp();
        sum += w;
        sum_sq += w * w;
    }
    if sum <= 0.0 {
        return 0.0;
    }
    ((sum * sum) / sum_sq) as f32
}

// ──────────────────────────────────────────────────────────────────────────
// T2 — x̂₀ posterior-mean reward proxy
// ──────────────────────────────────────────────────────────────────────────

/// x̂₀ readout mode (Plan 581 T2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum X0ProxyMode {
    /// `x̂₀ = candidates[argmax_j p(j|x_t)]` (ties → lowest index, the house
    /// argmax convention); **1 reward query per particle per step** — the
    /// budget-bearing mode (T2.2 contract).
    Argmax = 0,
    /// `V̂ = Σ_j p_j · r(c_j)` over the SHARED candidate set — `K` queries
    /// per step total (scalar/grid domains where candidates are shared
    /// across particles; the plan's "expectation for scalar domains").
    Expectation = 1,
}

/// x̂₀ posterior-mean reward proxy (Plan 581 T2 / Research 517 amortization
/// row i): consumes caller-provided per-particle marginals `p(x₀|x_t)` — the
/// same tensor `dllm_solver` / ppot consumers already produce — and
/// evaluates the opaque scorer on the predicted clean sample.
///
/// The scorer is a plain `Fn(&[f32]) -> f32` (any black-box: classifier,
/// validator composite, external oracle). Reward-call counts are carried on
/// the struct (`reward_queries`) so the T2.2 cost contract is measurable at
/// the type, not by harness bookkeeping.
pub struct X0ProxyReward<F> {
    reward: F,
    mode: X0ProxyMode,
    queries: AtomicU64,
}

impl<F> X0ProxyReward<F>
where
    F: Fn(&[f32]) -> f32,
{
    /// New proxy over a black-box scorer.
    pub fn new(mode: X0ProxyMode, reward: F) -> Self {
        Self {
            reward,
            mode,
            queries: AtomicU64::new(0),
        }
    }

    /// V̂ per particle (T2.1).
    ///
    /// `marginals`: flat `N×K` (row per particle; argmax/expectation are
    /// tolerant of unnormalized rows). `candidates`: flat `K×d` shared
    /// candidate clean states. `out`: len `N`.
    ///
    /// Cost contract (T2.2, pinned by test): Argmax == `N` scorer calls per
    /// step (vs `M·K` for a full MC twist); Expectation == `K` (the shared
    /// grid is scored once per call — allocates one K-sized buffer, the
    /// documented cold path).
    pub fn values_into(
        &self,
        marginals: &[f32],
        k: usize,
        candidates: &[f32],
        dim: usize,
        out: &mut [f32],
    ) {
        assert!(k > 0, "X0ProxyReward requires k > 0");
        assert!(dim > 0, "X0ProxyReward requires dim > 0");
        debug_assert_eq!(marginals.len(), out.len() * k);
        debug_assert_eq!(candidates.len(), k * dim);
        debug_assert!(
            marginals.iter().all(|m| m.is_finite()),
            "marginals must be finite"
        );
        match self.mode {
            X0ProxyMode::Argmax => {
                for (i, o) in out.iter_mut().enumerate() {
                    let row = &marginals[i * k..(i + 1) * k];
                    let mut best = 0usize;
                    let mut best_p = f32::NEG_INFINITY;
                    for (j, &p) in row.iter().enumerate() {
                        if p > best_p {
                            best_p = p;
                            best = j;
                        }
                    }
                    let x0 = &candidates[best * dim..(best + 1) * dim];
                    let v = (self.reward)(x0);
                    debug_assert!(v.is_finite(), "proxy scorer must return finite");
                    self.queries.fetch_add(1, Ordering::Relaxed);
                    *o = v;
                }
            }
            X0ProxyMode::Expectation => {
                // Score each shared candidate ONCE (K calls), then weight.
                let rc: Vec<f32> = candidates
                    .chunks_exact(dim)
                    .map(|c| (self.reward)(c))
                    .collect();
                for v in &rc {
                    debug_assert!(v.is_finite(), "proxy scorer must return finite");
                }
                self.queries.fetch_add(k as u64, Ordering::Relaxed);
                for (i, o) in out.iter_mut().enumerate() {
                    let row = &marginals[i * k..(i + 1) * k];
                    let mut acc = 0.0f64;
                    for (j, &p) in row.iter().enumerate() {
                        acc += p as f64 * rc[j] as f64;
                    }
                    *o = acc as f32;
                }
            }
        }
    }

    /// Cumulative scorer calls (the T2.2 budget axis).
    pub fn reward_queries(&self) -> u64 {
        self.queries.load(Ordering::Relaxed)
    }

    /// Readout mode.
    pub fn mode(&self) -> X0ProxyMode {
        self.mode
    }
}

/// Proxy-quality diagnostic (Plan 581 T2.3): Spearman rank correlation of
/// proxy values vs true terminal rewards on a caller-supplied held-out set
/// (average ranks for ties). Exported as a diagnostic, NOT a gate — the
/// end-to-end gate is Bench 692. Cold path (allocates two f64 rank
/// buffers); delegates to the shipped `numeric_stability::spearman_rho`.
pub fn proxy_spearman(proxy: &[f32], true_rewards: &[f32]) -> f64 {
    assert_eq!(
        proxy.len(),
        true_rewards.len(),
        "proxy_spearman length mismatch"
    );
    let px: Vec<f64> = proxy.iter().map(|&v| v as f64).collect();
    let tx: Vec<f64> = true_rewards.iter().map(|&v| v as f64).collect();
    crate::numeric_stability::spearman_rho(&px, &tx)
}

// ──────────────────────────────────────────────────────────────────────────
// Tests (Plan 581 T1.3/T2.2/T2.3/T3.4)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 — the house deterministic RNG convention.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_normal(&mut self) -> f32 {
            let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
            let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            ((-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()) as f32
        }
        fn next_uniform(&mut self) -> f32 {
            ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) as f32
        }
    }

    #[test]
    fn memo_hit_miss_and_key_distinguishes_state_and_t() {
        let memo = ValueMemo::new(64, u32::MAX);
        let s = [1.0f32, 2.0];
        let mut calls = 0u64;
        let v1 = memo.lookup_or_insert(&s, 3, || {
            calls += 1;
            7.0
        });
        let v2 = memo.lookup_or_insert(&s, 3, || {
            calls += 1;
            99.0 // must not run — same (state, t)
        });
        assert_eq!(v1, 7.0);
        assert_eq!(v2, 7.0);
        assert_eq!(calls, 1);
        assert_eq!(memo.hits(), 1);
        assert_eq!(memo.misses(), 1);
        // Different t ⇒ different key ⇒ miss + fresh compute.
        let v3 = memo.lookup_or_insert(&s, 4, || {
            calls += 1;
            9.0
        });
        assert_eq!(v3, 9.0);
        // Different state ⇒ different key.
        let s2 = [1.0f32, 2.5];
        let v4 = memo.lookup_or_insert(&s2, 3, || {
            calls += 1;
            11.0
        });
        assert_eq!(v4, 11.0);
        assert_eq!(calls, 3);
        assert_eq!(memo.misses(), 3);
        assert_eq!(memo.len(), 3);
    }

    #[test]
    fn memo_ttl_staleness_recomputes() {
        // With t in the key, a different t is a DIFFERENT entry (freshness-
        // correct by construction) — the ttl is the eviction window, not a
        // per-lookup check (see the ValueMemo doc). This test pins the
        // cross-t miss + the eviction-window predicate.
        let memo = ValueMemo::new(64, 5);
        let s = [0.5f32];
        let mut calls = 0u64;
        let mut compute = || {
            calls += 1;
            1.0
        };
        memo.lookup_or_insert(&s, 0, &mut compute);
        memo.lookup_or_insert(&s, 3, &mut compute); // different key (t=3) ⇒ miss
        memo.lookup_or_insert(&s, 3, &mut compute); // exact (state, t) replay ⇒ hit
        assert_eq!(calls, 2); // 2 misses compute; the (state, t) hit does not
        assert_eq!(memo.hits(), 1);
        assert_eq!(memo.misses(), 2);
        assert_eq!(memo.len(), 2);
    }

    #[test]
    fn memo_capacity_eviction_drops_old_ticks_first() {
        // Fill the cap at t=0, then insert at t=100 with ttl=5: the t=0
        // entries are outside the window and must be evicted (not the
        // fresh one).
        let memo = ValueMemo::new(4, 5);
        for i in 0..4u32 {
            memo.lookup_or_insert(&[i as f32], 0, || 0.0);
        }
        assert_eq!(memo.len(), 4);
        memo.lookup_or_insert(&[100.0f32], 100, || 9.0);
        assert_eq!(memo.len(), 1, "stale-tick entries evicted at capacity");
        // The survivor is the fresh entry — it hits without recompute.
        let mut calls = 0u64;
        let v = memo.lookup_or_insert(&[100.0f32], 100, || {
            calls += 1;
            0.0
        });
        assert_eq!(v, 9.0);
        assert_eq!(calls, 0);
    }

    #[test]
    fn memo_capacity_cap_bounds_entries() {
        let memo = ValueMemo::new(4, u32::MAX);
        for i in 0..8u32 {
            let s = [i as f32];
            memo.lookup_or_insert(&s, 0, || 0.0);
            assert!(memo.len() <= 4, "cap exceeded: {}", memo.len());
        }
        assert!(memo.len() <= 4);
        assert_eq!(memo.misses(), 8);
    }

    #[test]
    fn memo_clear_resets() {
        let memo = ValueMemo::new(8, u32::MAX);
        memo.lookup_or_insert(&[1.0], 0, || 1.0);
        assert!(!memo.is_empty());
        memo.clear();
        assert!(memo.is_empty());
        let mut calls = 0u64;
        memo.lookup_or_insert(&[1.0], 0, || {
            calls += 1;
            1.0
        });
        assert_eq!(calls, 1); // cleared ⇒ miss
    }

    #[test]
    fn ridge_fit_recovers_linear_ground_truth() {
        // Ground truth: w* = [0.5, 1.0, 2.0] (all-positive features ⇒ y > 0,
        // keeping rewards in the R ≥ 0 ln₁p domain), y = w*·f exactly,
        // R = expm1(y).
        let dim = 3usize;
        let w_true = [0.5f32, 1.0, 2.0];
        let mut rng = SplitMix64(0x517);
        let n = 200usize;
        let mut features = Vec::with_capacity(n * dim);
        let mut rewards = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let f: [f32; 3] = [
                0.1 + rng.next_uniform(),
                0.1 + rng.next_uniform(),
                0.1 + rng.next_uniform(),
            ];
            let y: f32 = f.iter().zip(w_true.iter()).map(|(&a, &b)| a * b).sum();
            features.extend_from_slice(&f);
            rewards.push(y.exp_m1());
            ys.push(y);
        }
        let table = RidgeTwistTable::fit(&features, &rewards, dim, 1e-8);
        for (i, &wt) in w_true.iter().enumerate() {
            assert!(
                (table.weights()[i] - wt).abs() < 1e-3,
                "w[{i}] = {} vs {wt}",
                table.weights()[i]
            );
        }
        // Readout reproduces y on a fresh row.
        let f = [0.9f32, 0.4, 0.2];
        let y_true: f32 = f.iter().zip(w_true.iter()).map(|(&a, &b)| a * b).sum();
        assert!((table.value(&f) - y_true).abs() < 1e-3);
        let _ = ys;
    }

    #[test]
    fn ridge_readout_clamps_outside_collected_range() {
        // The extrapolation guard (Bench 692 finding): in-range readouts are
        // exact; beyond the collected envelope the value saturates instead
        // of diverging (a runaway twist must not chase linear extrapolation).
        let dim = 1usize;
        let w_true = 2.0f32;
        let n = 50usize;
        let mut features = Vec::with_capacity(n);
        let mut rewards = Vec::with_capacity(n);
        for i in 0..n {
            let f = 0.1 + 0.9 * (i as f32) / (n - 1) as f32; // ∈ [0.1, 1.0]
            let y = w_true * f;
            features.push(f);
            rewards.push(y.exp_m1());
        }
        let table = RidgeTwistTable::fit(&features, &rewards, dim, 1e-8);
        // In-range: exact.
        assert!((table.value(&[0.5]) - 1.0).abs() < 1e-3);
        // Beyond the collected support: clamped to the observed y-range,
        // never larger.
        assert!(table.value(&[50.0]) <= table.value(&[1.0]) + 1e-4);
        assert!(table.value(&[-50.0]) >= table.value(&[0.1]) - 1e-4);
    }

    #[test]
    fn select_beta_delegates_to_entropic_tilt() {
        let values = [0.1f32, 0.4, 0.9, 0.2];
        assert_eq!(
            select_beta_by_budget(&values, 0.5),
            crate::entropic_tilt::solve_beta(&values, 0.5)
        );
    }

    #[test]
    fn twist_step_first_step_weights_proportional_to_psi() {
        // Fresh prev (zeros): A_i = log ψ_i. Saturated budget ⇒ β = BETA_MAX
        // (on the span-normalized values — [1, 0] after normalization).
        let values = [1.0f32, 0.0];
        let mut log_w = [0.0f32; 2];
        let mut prev = [0.0f32; 2];
        let mut beta = 0.0f32;
        twist_step_into(&values, 10.0, &mut log_w, &mut prev, &mut beta);
        assert_eq!(beta, 1e3); // budget ≥ ln 2 ⇒ one-hot saturation bracket
        assert_eq!(log_w[0], 0.0);
        assert_eq!(log_w[1], -1e3);
        assert_eq!(prev, log_w); // prev carries centered log ψ
    }

    #[test]
    fn twist_step_incremental_ratio_hand_check() {
        // Step 2 from the state above: values [0.0, 2.0] span-normalize to
        // [0, 1]. The step-1 deficit of particle 1 is priced in; the RATIO
        // cancels it — the currently good particle dominates.
        let values = [0.0f32, 2.0];
        let mut log_w = [0.0f32, -1e3];
        let mut prev = [0.0f32, -1e3];
        let mut beta = 0.0f32;
        twist_step_into(&values, 10.0, &mut log_w, &mut prev, &mut beta);
        assert_eq!(log_w[0], -1e3);
        assert_eq!(log_w[1], 0.0);
    }

    #[test]
    fn twist_uniform_values_leave_weights_unchanged() {
        let values = [0.5f32, 0.5, 0.5];
        let mut log_w = [0.3f32, -0.7, 2.0];
        let mut prev = [0.0f32; 3];
        let mut beta = 0.0f32;
        twist_step_into(&values, 1.0, &mut log_w, &mut prev, &mut beta);
        assert_eq!(log_w, [0.3, -0.7, 2.0]); // degenerate span ⇒ β=0, no tilt
        assert_eq!(beta, 0.0);
        // The runaway guard: round-off-scale spread must ALSO stay inert
        // (Bench 692 — β saturation amplifying clamp-ceiling round-off).
        let noisy = [1.0f32, 1.0 + 1e-8, 1.0 - 1e-8];
        let mut log_w2 = [0.3f32, -0.7, 2.0];
        let mut prev2 = [0.0f32; 3];
        let mut beta2 = 99.0f32;
        twist_step_into(&noisy, 1.0, &mut log_w2, &mut prev2, &mut beta2);
        assert_eq!(beta2, 0.0, "round-off span must trip the degenerate guard");
        assert_eq!(log_w2, [0.3, -0.7, 2.0]);
    }

    #[test]
    fn twist_step_is_scale_invariant() {
        // The KL budget is a statement about the TILT: scaling the value
        // axis must not change the weight outcome.
        let base = [0.2f32, 0.9, 0.4, 1.1, 0.7];
        let mut w1 = [0.0f32; 5];
        let mut p1 = [0.0f32; 5];
        let mut b1 = 0.0f32;
        twist_step_into(&base, 0.8, &mut w1, &mut p1, &mut b1);
        let scaled: Vec<f32> = base.iter().map(|&v| 37.0 * v + 12.0).collect();
        let mut w2 = [0.0f32; 5];
        let mut p2 = [0.0f32; 5];
        let mut b2 = 0.0f32;
        twist_step_into(&scaled, 0.8, &mut w2, &mut p2, &mut b2);
        for i in 0..5 {
            assert!(
                (w1[i] - w2[i]).abs() < 1e-4,
                "weight {i}: {} vs {}",
                w1[i],
                w2[i]
            );
        }
    }

    #[test]
    fn twist_after_resample_restarts_chain_as_fresh_reweight() {
        let mut prev = [0.0f32, -1e3];
        twist_after_resample(&mut prev);
        // Post-resample the consumer reset log_w to uniform; the next step
        // must act as a FRESH one-shot twist (weights ∝ ψ_t alone).
        let mut log_w = [0.0f32; 2];
        let values = [2.0f32, 1.0];
        let mut beta = 0.0f32;
        twist_step_into(&values, 10.0, &mut log_w, &mut prev, &mut beta);
        assert_eq!(log_w[0], 0.0);
        assert_eq!(log_w[1], -1e3);
    }

    #[test]
    fn ess_uniform_is_n_and_onehot_is_one() {
        assert!((ess_from_log_weights(&[0.0; 8]) - 8.0).abs() < 1e-4);
        assert!((ess_from_log_weights(&[0.0, -30.0]) - 1.0).abs() < 1e-4);
        // Scale invariance (ESS is a ratio — normalization cancels).
        assert!((ess_from_log_weights(&[10.0, 10.0, 10.0]) - 3.0).abs() < 1e-3);
    }

    #[test]
    fn proxy_argmax_cost_contract_is_n_queries() {
        // T2.2: Argmax == 1 query per particle per step (vs M·K for full MC).
        let n = 7usize;
        let k = 5usize;
        let dim = 2usize;
        let mut marginals = vec![0.0f32; n * k];
        for row in marginals.chunks_exact_mut(k) {
            for (j, m) in row.iter_mut().enumerate() {
                *m = (j + 1) as f32; // argmax → last candidate
            }
        }
        let candidates: Vec<f32> = (0..k * dim).map(|i| i as f32).collect();
        let proxy = X0ProxyReward::new(X0ProxyMode::Argmax, |x: &[f32]| x[0] * 10.0);
        let mut out = vec![0.0f32; n];
        proxy.values_into(&marginals, k, &candidates, dim, &mut out);
        assert_eq!(proxy.reward_queries(), n as u64);
        // argmax = candidate 4 = [8.0, 9.0] ⇒ v = 80 for every particle.
        for o in &out {
            assert!((o - 80.0).abs() < 1e-5);
        }
    }

    #[test]
    fn proxy_expectation_cost_contract_is_k_queries() {
        let n = 7usize;
        let k = 4usize;
        let dim = 2usize;
        let mut marginals = vec![0.0f32; n * k];
        for row in marginals.chunks_exact_mut(k) {
            row.copy_from_slice(&[0.1, 0.7, 0.2, 0.0]);
        }
        let candidates: Vec<f32> = (0..k * dim).map(|i| i as f32).collect();
        let proxy = X0ProxyReward::new(X0ProxyMode::Expectation, |x: &[f32]| x[0] * 10.0);
        let mut out = vec![0.0f32; n];
        proxy.values_into(&marginals, k, &candidates, dim, &mut out);
        assert_eq!(proxy.reward_queries(), k as u64); // shared grid scored once
        // Candidate rows are [0,1],[2,3],[4,5],[6,7] ⇒ x[0] ∈ {0,2,4,6};
        // E[v] = 0.1·0 + 0.7·20 + 0.2·40 + 0.0·60 = 22.0
        for o in &out {
            assert!((o - 22.0).abs() < 1e-5);
        }
    }

    #[test]
    fn proxy_spearman_diagnostic_bounds() {
        let proxy = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let truth_ascending = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let truth_descending = [50.0f32, 40.0, 30.0, 20.0, 10.0];
        assert!((proxy_spearman(&proxy, &truth_ascending) - 1.0).abs() < 1e-9);
        assert!((proxy_spearman(&proxy, &truth_descending) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn twist_pipeline_two_run_bit_identity() {
        // T3.4: two fresh runs of a memo + twist + resample pipeline are
        // bit-identical (papaya iteration order never enters results).
        let run = || -> (Vec<f32>, u64, u64) {
            use crate::distributional_steering::{WeightedPopulation, systematic_resample_into};
            let memo = ValueMemo::new(1024, u32::MAX);
            let mut rng = SplitMix64(0x581);
            let n = 16usize;
            let dim = 2usize;
            let mut states: Vec<f32> = (0..n * dim)
                .map(|_| rng.next_uniform() * 2.0 - 1.0)
                .collect();
            let mut log_w = vec![0.0f32; n];
            let mut prev = vec![0.0f32; n];
            let mut beta = 0.0f32;
            let reward = |x: &[f32]| -(x[0] * x[0] + x[1] * x[1]); // at origin
            for t in 0..8u32 {
                let mut vals = vec![0.0f32; n];
                for (i, s) in states.chunks_exact(dim).enumerate() {
                    vals[i] = memo.lookup_or_insert(s, t, || reward(s));
                }
                twist_step_into(&vals, 0.5, &mut log_w, &mut prev, &mut beta);
                if ess_from_log_weights(&log_w) < n as f32 * 0.5 {
                    let mut w = vec![0.0f32; n];
                    {
                        let mut lw = log_w.clone();
                        let pop = WeightedPopulation::new(&states, &mut lw, dim);
                        pop.weights_into(&mut w);
                    }
                    let u = rng.next_uniform() * 0.999 + 0.000_5;
                    let mut ancestors = vec![0u32; n];
                    systematic_resample_into(&w, n, u, &mut ancestors);
                    let mut next = vec![0.0f32; n * dim];
                    for (slot, &a) in ancestors.iter().enumerate() {
                        let a = a as usize;
                        next[slot * dim..(slot + 1) * dim]
                            .copy_from_slice(&states[a * dim..(a + 1) * dim]);
                    }
                    states = next;
                    log_w.iter_mut().for_each(|l| *l = 0.0);
                    twist_after_resample(&mut prev);
                }
                for s in states.iter_mut() {
                    *s += rng.next_normal() * 0.05;
                }
            }
            (log_w, memo.hits(), memo.misses())
        };
        let (a, ha, ma) = run();
        let (b, hb, mb) = run();
        assert_eq!(a, b, "two-run bit-identity (T3.4)");
        assert_eq!((ha, ma), (hb, mb), "memo counters deterministic");
        assert!(ma > 0);
    }
}
