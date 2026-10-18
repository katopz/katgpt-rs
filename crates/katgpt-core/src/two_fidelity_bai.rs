//! Two-Fidelity Certified Best-Action Identification (2FFS) — Plan 615.
//!
//! Certified two-fidelity best-action identification for stochastic minimax
//! trees, distilled from Chen & Chen, *Two-Fidelity Best-Action
//! Identification for Stochastic Minimax Tree* (arXiv:2606.01708, NeurIPS
//! 2026) — Research 601. Every node can be evaluated by a **fast oracle**
//!       (cheap, deterministic, biased within a known nondecreasing envelope
//!       `B(h)`, `B(0) = 0`) or a **slow oracle** (expensive, stochastic,
//!       unbiased, queryable at any exposed node). The algorithm races minimax-style
//!       fast expansion against MCTS-style slow certification until the root action
//!       is certified ε-optimal with confidence 1−δ.
//!
//! This module ships the Phase-1 search (plan T1.1–T1.7): the [`MinimaxSpace`]
//! consumer trait, the per-node interval state with minimax backup (paper
//! Eqs. 4/6, Lemmas 2.3/B.2), the time-uniform sub-Gaussian radius [`beta`],
//! the confidence allocation, the result type, and the root
//! leader/challenger loop with per-(node, side, scale) resolution (plan
//! T1.5/T1.6): [`two_fidelity_search`] races the local slow-sample route
//! against the budgeted recursive route (Eq. 7) under latched `Done_s(v,k)`
//! bitflags, with deterministic tie-breaks, the empty-intersection guard
//! (paper B.3 convention: terminate + default action, never loop), and the
//! ρ₀ = 0 early exit.
//!
//! Pure query-allocation + interval bookkeeping — zero gradient descent, zero
//! weights, zero deps beyond `arrayvec`. Opt-in feature `two_fidelity_bai`;
//! never promoted to default (the default search slot keeps `mcts_search`;
//! this fills the empty *certified-search* slot).
//!
//! two_fidelity_bai — certified two-fidelity BAI for minimax trees (Plan 615,
//! Research 601, arXiv:2606.01708): interval minimax backup + time-uniform
//! slow CIs + ε-stop with first-class certificates. Opt-in
//! (`two_fidelity_bai`); the default search slot keeps mcts_search.

use arrayvec::ArrayVec;
use std::collections::HashMap;

/// Upper bound on tree nodes the search state pre-allocates (the `mcts.rs`
/// `MAX_TREE_SIZE` precedent). The confidence allocation
/// [`SearchConfig::delta_for_node`] divides δ uniformly over this cap.
pub const NODE_CAP: u32 = 65_536;

/// Upper bound on children per node (the `mcts.rs` `MAX_UNEXPANDED = 16`
/// precedent; covers the Plan 615 bench settings b ∈ {3, 6, 8} with headroom).
/// A [`MinimaxSpace`] impl pushing more than this panics — loud, by design.
pub const MAX_CHILDREN: usize = 16;

/// Max-node / min-node role of a tree node (paper Definition 2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// Max node: backup takes the child-wise maximum (Eq. 6).
    Max,
    /// Min node: backup takes the child-wise minimum (Eq. 6).
    Min,
}

/// The consumer-supplied minimax space (plan T1.1; paper Definitions 2.2/2.3).
///
/// The search never owns tree storage — the consumer exposes nodes through
/// this trait and the search keeps only per-node interval bookkeeping keyed
/// by `u32` node handles (slab indices, the `mcts.rs` precedent).
///
/// # Contracts the consumer must honor
///
/// - `bias_envelope`: `B(h) ≥ 0` **nondecreasing** in remaining depth `h`,
///   with `B(0) = 0` (paper Eq. 1). The fast oracle may be arbitrarily wrong
///   inside the envelope, never outside it. A loose-but-honest envelope
///   (paper Thm 3.8) only costs samples; an understated envelope breaks PAC.
/// - `slow_sample`: i.i.d. draws with mean `V*(v)` (the true minimax value at
///   the node) and sub-Gaussian parameter `sigma` — unbiased, unlike
///   BAI-MCTS leaf rollouts. Unlike BAI-MCTS, queryable at *any* exposed
///   node.
/// - `children_into` reveals children (expansion); at most
///   [`MAX_CHILDREN`] pushes or the impl panics (loud).
pub trait MinimaxSpace {
    /// Max or Min role of `node`.
    fn kind(&self, node: u32) -> NodeKind;

    /// Remaining depth `h(node) ≥ 0` — the argument of `bias_envelope`.
    /// Depth 0 = leaf: `bias_envelope(0) = 0` pins the fast oracle exact.
    fn remaining_depth(&self, node: u32) -> u8;

    /// Reveal + push the children of `node` into `out` (expansion).
    fn children_into(&self, node: u32, out: &mut ArrayVec<u32, MAX_CHILDREN>);

    /// Fast-oracle value `V_F(v)` — deterministic, biased within
    /// `bias_envelope(remaining_depth(node))`.
    fn fast_value(&self, node: u32) -> f32;

    /// Fast-oracle bias envelope `B(h)` — nondecreasing in `h`, `B(0) = 0`.
    fn bias_envelope(&self, remaining_depth: u8) -> f32;

    /// One slow-oracle sample — i.i.d., mean `V*(v)`, sub-Gaussian `sigma`.
    fn slow_sample(&self, node: u32, rng: &mut impl RngCore) -> f32;
}

/// Minimal RNG surface the search needs (kept local so the module has zero
/// RNG dependency; consumers pass whatever they own).
pub trait RngCore {
    /// Next uniform `f64` in `[0, 1)`.
    fn f64(&mut self) -> f64;
}

/// Per-node interval state (plan T1.2; paper Eqs. 4/6, Lemmas 2.3/B.2).
///
/// Three interval layers per node:
/// - **fast** `[V_F − B(h), V_F + B(h)]` — fixed at exposure;
/// - **slow** — the *running intersection* of time-uniform CIs over the
///   sample history (Eq. 4): maintained as running min-of-lower /
///   max-of-upper endpoints (monotone shrink ⇒ Lemma B.2 nesting holds by
///   construction), with Welford moments for the mean;
/// - **child backup** (Eq. 6): Max node → `[max L_u, max U_u]`; Min node →
///   `[min L_u, min U_u]`. Lemma 2.3: validity and width preserved upward
///   (a max of intervals of width ≤ w has width ≤ w).
///
/// The **effective** interval is `local ∩ child` (paper Lemma 2.4: all
/// simultaneously valid on the union-bound event of probability ≥ 1−δ).
///
/// All accumulation is `f64`; oracle values arrive `f32`.
#[derive(Clone, Debug)]
pub struct NodeInterval {
    fast_lo: f64,
    fast_hi: f64,

    slow_lo: f64,
    slow_hi: f64,
    slow_n: u32,
    welford_mean: f64,
    welford_m2: f64,

    child_lo: f64,
    child_hi: f64,
    child_set: bool,
}

impl NodeInterval {
    /// Fast interval fixed at exposure: `V_F ± B(h)`.
    pub fn from_fast(fast_value: f32, bias: f32) -> Self {
        let v = f64::from(fast_value);
        let b = f64::from(bias);
        Self {
            fast_lo: v - b,
            fast_hi: v + b,
            slow_lo: f64::NEG_INFINITY,
            slow_hi: f64::INFINITY,
            slow_n: 0,
            welford_mean: 0.0,
            welford_m2: 0.0,
            child_lo: f64::NEG_INFINITY,
            child_hi: f64::INFINITY,
            child_set: false,
        }
    }

    /// Oracle-less interval: the fast component is `(−∞, +∞)` so `local` is
    /// the slow CI alone. The single-fidelity propagation consumer
    /// (LUCB-MCTS-style baselines — Issue 916's corrected BAI arm) and any
    /// oracle-free CI carrier: internals install the Eq. 6 child fold via
    /// [`Self::set_child_backup`], leaves accumulate [`Self::observe_slow`].
    pub fn slow_only() -> Self {
        Self {
            fast_lo: f64::NEG_INFINITY,
            fast_hi: f64::INFINITY,
            slow_lo: f64::NEG_INFINITY,
            slow_hi: f64::INFINITY,
            slow_n: 0,
            welford_mean: 0.0,
            welford_m2: 0.0,
            child_lo: f64::NEG_INFINITY,
            child_hi: f64::INFINITY,
            child_set: false,
        }
    }

    /// Absorb one slow sample (Eq. 4): Welford update + running-intersection
    /// clamp with the time-uniform radius [`beta`]. Monotone shrink — the
    /// Lemma B.2 nesting invariant — is asserted in debug builds.
    pub fn observe_slow(&mut self, sample: f32, sigma: f64, delta_v: f64) {
        let y = f64::from(sample);
        self.slow_n += 1;
        let n = f64::from(self.slow_n);
        // Welford (numerically stable mean / M2; the M2 tail is carried for
        // diagnostics, the radius is the time-uniform bound, not the
        // pointwise CLT — Eq. 3).
        let delta = y - self.welford_mean;
        self.welford_mean += delta / n;
        self.welford_m2 += delta * (y - self.welford_mean);

        let radius = beta(self.slow_n, delta_v, sigma);
        let lo = self.welford_mean - radius;
        let hi = self.welford_mean + radius;
        let prev_lo = self.slow_lo;
        let prev_hi = self.slow_hi;
        // Running intersection: min-of-lower / max-of-upper.
        self.slow_lo = if self.slow_n == 1 {
            lo
        } else {
            prev_lo.max(lo)
        };
        self.slow_hi = if self.slow_n == 1 {
            hi
        } else {
            prev_hi.min(hi)
        };
        debug_assert!(
            self.slow_lo >= prev_lo.max(f64::NEG_INFINITY) - 1e-12
                && self.slow_hi <= prev_hi.min(f64::INFINITY) + 1e-12,
            "Lemma B.2 nesting violated: [{}, {}] -> [{}, {}]",
            prev_lo,
            prev_hi,
            self.slow_lo,
            self.slow_hi
        );
    }

    /// Local interval: `fast ∩ slow` (no samples yet ⇒ the fast interval
    /// alone).
    pub fn local(&self) -> (f64, f64) {
        (
            self.fast_lo.max(self.slow_lo),
            self.fast_hi.min(self.slow_hi),
        )
    }

    /// Install the child-backup interval (Eq. 6) computed by the caller from
    /// the children's *effective* intervals. Lemma 2.3 — backup preserves
    /// validity and never widens — is asserted in debug builds (the max/min
    /// of the children's endpoints is width-dominated by the widest child).
    pub fn set_child_backup(&mut self, kind: NodeKind, children: &[(f64, f64)]) {
        debug_assert!(!children.is_empty(), "backup over zero children");
        // Max: L = max child L, U = max child U (Eq. 6); Min: min — each
        // accumulator starts at its kind's identity.
        let (mut lo, mut hi) = match kind {
            NodeKind::Max => (f64::NEG_INFINITY, f64::NEG_INFINITY),
            NodeKind::Min => (f64::INFINITY, f64::INFINITY),
        };
        for &(cl, ch) in children {
            // Eq. 6: L always reduces the children's LOWER endpoints (cl);
            // U always reduces the children's UPPER endpoints (ch). Only the
            // reduction (max vs min) is kind-dependent.
            lo = match kind {
                NodeKind::Max => cl.max(lo),
                NodeKind::Min => cl.min(lo),
            };
            hi = match kind {
                NodeKind::Max => ch.max(hi),
                NodeKind::Min => ch.min(hi),
            };
        }
        self.child_lo = lo;
        self.child_hi = hi;
        self.child_set = true;
        // Lemma 2.3 (width half): the installed interval is never wider than
        // the widest child interval it was folded from.
        let widest_child = children
            .iter()
            .map(|&(cl, ch)| ch - cl)
            .fold(0.0_f64, f64::max);
        debug_assert!(
            hi - lo <= widest_child + 1e-9,
            "Lemma 2.3 violated: backup width {} > widest child width {}",
            hi - lo,
            widest_child
        );
    }

    /// Effective interval: `local ∩ child` (child absent ⇒ local alone).
    pub fn effective(&self) -> (f64, f64) {
        let (ll, lh) = self.local();
        if self.child_set {
            (ll.max(self.child_lo), lh.min(self.child_hi))
        } else {
            (ll, lh)
        }
    }

    /// The installed child-backup interval alone (Eq. 6), without the local
    /// intersection — how the parent reads a child.
    pub fn child_backup(&self) -> Option<(f64, f64)> {
        self.child_set.then_some((self.child_lo, self.child_hi))
    }

    /// Sample count observed so far.
    pub fn slow_samples(&self) -> u32 {
        self.slow_n
    }

    /// Welford sample mean (diagnostics; the *certified* center is the
    /// interval, never the point estimate).
    pub fn slow_mean(&self) -> Option<f64> {
        (self.slow_n > 0).then_some(self.welford_mean)
    }
}

/// Time-uniform sub-Gaussian radius (plan T1.3; paper Eq. 3).
///
/// Stitched geometric-stage construction (Howard et al. 2021 /
/// Kaufmann–Koolen 2021 class): stages `n_k = ceil(1.5^(k-1))`, per-stage
/// budgets `δ_k = δ·2^(−k)` (Σ δ_k = δ). For `n ∈ [n_k, n_{k+1})` the radius
/// is the stage-constant `σ·sqrt(2·ln(2·n_k/δ_k)/n_k)` — a union bound over
/// the stage's ≤ n_k sample counts gives P(any in-stage violation) ≤ δ_k,
/// so with probability ≥ 1−δ the bound holds **simultaneously for all n**
/// (validity, paper Eq. 3). Monotone non-increasing in n: constant within a
/// stage, and the stage sequence `t_k` is decreasing for every δ ≤ 2/3
/// (ratio test: `(c' + 2L)/(c' + L) ≤ 1.5` ⟺ `L ≤ c'` with
/// `L = ln 2 + ln 1.5`, `c' = ln(2/δ)`). `beta → 0` as `n → ∞`.
#[must_use]
pub fn beta(n: u32, delta_v: f64, sigma: f64) -> f64 {
    assert!(n >= 1, "beta defined for n ≥ 1");
    assert!(
        (0.0 < delta_v && delta_v < 1.0),
        "delta_v must be in (0, 1), got {delta_v}"
    );
    assert!(sigma.is_finite() && sigma > 0.0, "sigma must be > 0");
    let k = stage_index(n);
    let n_k = stage_start(k);
    // δ_k = δ · 2^-k  (weights w_k = (1-q) q^{k-1} with q = 1/2).
    let delta_k = delta_v * (-f64::from(k)).exp2();
    let ln_term = (2.0 * f64::from(n_k) / delta_k).ln();
    sigma * (2.0 * ln_term / f64::from(n_k)).sqrt()
}

/// Stage index `k` with `n_k = ceil(1.5^(k-1)) ≤ n < n_{k+1}` (k ≥ 1).
fn stage_index(n: u32) -> u32 {
    // Closed form + exact correction (1.5^k is exact in f64 for k ≤ 81).
    let mut k = 1 + (f64::from(n).ln() / 1.5f64.ln()).floor() as u32;
    while stage_start(k) > n {
        k -= 1;
    }
    while stage_start(k + 1) <= n {
        k += 1;
    }
    k
}

/// `n_k = ceil(1.5^(k-1))` — exact for k ≤ 81 (1.5 is 3/2, binary-exact).
fn stage_start(k: u32) -> u32 {
    if k <= 1 {
        1
    } else {
        let p = 1.5f64.powi(i32::try_from(k).unwrap_or(81) - 1);
        p.ceil() as u32
    }
}

/// Race-budget scale `α(h) = (h+1)²` (plan T1.4; paper §3.1 — depth-indexed
/// schedule giving the `O(D²)` cost bound of Thm 3.6).
#[must_use]
pub fn race_scale(remaining_depth: u8) -> f64 {
    f64::from(remaining_depth + 1) * f64::from(remaining_depth + 1)
}

/// Search configuration (plan T1.4).
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Certified-optimality slack ε: the stop guarantees the returned action
    /// is ε-optimal with confidence 1−δ.
    pub epsilon: f32,
    /// Failure budget δ (paper PAC parameter).
    pub delta: f64,
    /// Slow-oracle cost `c` in the unified cost model
    /// `cost = n_fast + c·n_slow` (fast = 1, fixed per setting).
    pub slow_cost: f32,
    /// A-priori node cap for the confidence allocation (≤ [`NODE_CAP`]).
    pub node_cap: u32,
    /// Slow-oracle sub-Gaussian parameter (consumer-declared, paper Assump. 2).
    pub sigma: f64,
    /// Per-node sample-accounting trace (Issue 916 cause-(b) measurement).
    /// Pure observation: when set, [`TwoFidelityResult::node_stats`] carries
    /// one row per sampled node at search end. Never affects the search
    /// (no allocation, no stop-rule, no confidence change), so this is not
    /// a confidence-allocation knob and needs no union-proof ceremony.
    pub trace_nodes: bool,
    /// Issue 916 (b-lever): resolve with the reference algorithm's **lazy
    /// discharge + parent-capped scales** instead of the v1 blocking-child
    /// descent. Opt-in (`false` = the v1 search, bit-identical).
    ///
    /// Ported from the authors' implementation
    /// (`github.com/PeterLauLukChen/2FFS`, `method/twoffs/twoffs.py`:
    /// `cert` / `active_children` / `child_obligation` / `capped_scale` /
    /// `root_obligation` / `resolve`), which is how the paper's §1.3 item 2
    /// lazy discharge and the Prop. B.10 effective-gap bound
    /// (`Δ_v^eff ≤ 2ρ_k`: certify only to the precision that can affect the
    /// root) are realised in code:
    /// - a child obligation runs at a scale **no finer than its parent's**
    ///   (`capped_scale`), so a subtree is never driven past the resolution
    ///   the consuming decision can use — the measured ladder overshoot;
    /// - in the comparison cases (Max, L) / (Min, U) a child whose opposite
    ///   endpoint is already within `ρ_k/2` of the parent's bound is
    ///   discharged without being certified, and the live blocker's
    ///   OPPOSITE side is refined first (it is what keeps it live);
    /// - the Eq. 7 race budget is per (node, scale), and an expansion is
    ///   its own work unit.
    ///
    /// Soundness is unchanged by construction: the knob only changes WHICH
    /// node receives the next sample or expansion. Every interval is still
    /// the honest fast envelope ∩ the time-uniform slow CI ∩ the Eq. 6
    /// backup, the δ_v allocation is untouched, and the ε-stop rule is the
    /// same — time-uniform CIs are valid under any adaptive sampling rule,
    /// so PAC (Thm 3.1) rides on interval validity + the stop exactly as in
    /// v1. Divergence from the reference, documented: certificates latch
    /// when EVALUATED (the reference sweeps every explored node per step);
    /// a latch can only be missed, never invented, so this costs work and
    /// never correctness. The reference's "slow-CI width ≤ ρ_k/2" branch is
    /// subsumed here (`local()` ⊇ effective, so the effective-width test
    /// fires first).
    pub lazy_discharge: bool,
}

impl SearchConfig {
    /// Uniform per-node failure budget `δ_v = δ / node_cap` (plan T1.4; the
    /// union bound over nodes then sums to ≤ δ).
    ///
    /// Dynamic trees exceeding the cap would need a dyadic-rank allocation
    /// (documented alternative in the plan); this crate pins the a-priori cap
    /// instead — the `mcts.rs` `MAX_TREE_SIZE` precedent — and refuses to
    /// exceed it.
    #[must_use]
    pub fn delta_for_node(&self) -> f64 {
        self.delta / f64::from(self.node_cap.max(1))
    }
}

/// Cost ledger in the unified model `cost = n_fast + c·n_slow` (plan T2.2's
/// scoring basis; ops are reported separately by the bench, never mixed with
/// `advance()`-budget units).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    pub fast: u64,
    pub slow: u64,
}

impl Cost {
    /// Unified-model cost with the config's slow cost.
    #[must_use]
    pub fn unified(&self, slow_cost: f32) -> f64 {
        f64::from(slow_cost) * self.slow as f64 + self.fast as f64
    }
}

/// Search outcome (plan T1.7) — the certificate is first-class output.
#[derive(Clone, Debug)]
pub struct TwoFidelityResult {
    /// Certified ε-optimal root action (child handle at the root).
    pub best_action: u32,
    /// Query cost in both oracles.
    pub cost: Cost,
    /// True when the ε-stop fired with the certificate; `false` only on the
    /// empty-intersection guard path (plan T1.6, paper B.3 convention —
    /// terminates + defaults, never loops).
    pub certified: bool,
    /// Root children's effective intervals at stop (the certificate body).
    pub root_intervals: Vec<(u32, f64, f64)>,
    /// Per-node sample accounting at stop — empty unless
    /// [`SearchConfig::trace_nodes`] (Issue 916 cause-(b) measurement; the
    /// attribution question is which nodes consumed the slow budget and at
    /// what width targets, vs the paper's Figure/Table decomposition).
    pub node_stats: Vec<NodeStat>,
}

/// One node's slow-sample ledger row at search end (tracing only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeStat {
    pub node: u32,
    pub kind: NodeKind,
    pub depth: u8,
    /// Slow samples this node received (single-sample work units).
    pub slow_n: u64,
    /// Latched scale bitmasks per side (bit `k` of side `s`).
    pub done: [u64; 2],
    /// Effective interval width at stop.
    pub final_width: f64,
    /// The width target the node was last being driven toward
    /// (`ρ_k / 2` at its smallest unresolved scale; 0.0 when fully latched).
    pub target_width: f64,
}

/// The interval side a resolution refines (paper's per-side certificates:
/// `L` = lower endpoint, `U` = upper endpoint).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Lower endpoint `L_v` — the leader's side at the root.
    Lower,
    /// Upper endpoint `U_v` — the challenger's side at the root.
    Upper,
}

impl Side {
    fn index(self) -> usize {
        match self {
            Side::Lower => 0,
            Side::Upper => 1,
        }
    }
}

/// Per-node search bookkeeping (tree storage stays in the consumer's
/// [`MinimaxSpace`] — the search holds only intervals, latches, and the
/// revealed child list).
struct NodeState {
    interval: NodeInterval,
    kind: NodeKind,
    depth: u8,
    children: ArrayVec<u32, MAX_CHILDREN>,
    expanded: bool,
    /// Latched `Done_s(v, k)` bitflags — bit `k` of `done[s]` says side `s`
    /// of this node is certified at scale `ρ_k`. Latching is monotone
    /// (zero rework per node/side/scale, paper §3.2).
    done: [u64; 2],
    /// Cumulative recursive-route spend at this node, unified cost units
    /// (fast = 1, slow = `c`). Compared against the Eq. 7 race budget.
    rec_spend: f64,
    /// Slow samples received (Issue 916 accounting trace; unconditional
    /// one-add per sample — samples dominate the cost, the counter is noise).
    slow_n: u64,
}

/// Eq. 7 race budget for the recursive route at `(v, k)`: zero when the fast
/// envelope is already within a quarter of the scale (recursion cannot buy
/// meaningfully tighter children), else `c · m_v(ρ_k)` — the local route's
/// cost — so the recursive route may spend up to `α_h` times what local
/// certification would have cost.
fn gamma_v(bias: f64, rho: f64, slow_cost: f32, local_samples: u32) -> f64 {
    if bias <= rho / 4.0 {
        0.0
    } else {
        f64::from(slow_cost) * f64::from(local_samples)
    }
}

/// `m_v(ρ)`: the smallest sample count whose slow-interval width `2β(n)`
/// certifies side width `width_target`. β is stage-constant, so only the
/// ~[`M_SAMPLES_STAGE_MAX`] distinct stage radii matter — [`stage_radii`]
/// precomputes them once (the profiled hot path: the free-function form
/// re-evaluated β's ln/sqrt/exp2 per stage per call, ~97% of recording-run
/// wall at D ≥ 7), and [`m_samples_with`] walks the table with plain f64
/// compares. The free [`m_samples`] builds the table per call and stays the
/// spec/reference form (unit-tested); the search consumes the precomputed
/// pair — identical values, identical branches, bit-identical results.
const M_SAMPLES_STAGE_MAX: u32 = 200;

/// β(n_k, δ_v, σ) for every stage start n_k, k = 1.. — the table stops at
/// the last stage whose n_k is representable (`stage_start` saturates `as
/// u32` once 1.5^(k−1) > u32::MAX — k = 57 — and β there would overflow
/// `stage_index`'s correction walk; the pre-table lazy form never evaluated
/// those stages for honest widths). `m_samples_with` returns the
/// `u32::MAX` backstop past the table — the certified-search degradation
/// the callers already document — where the pre-table form would have
/// panicked mid-walk. Built once per search for its fixed (δ_v, σ).
fn stage_radii(delta_v: f64, sigma: f64) -> Vec<f64> {
    let mut radii = Vec::with_capacity(64);
    for k in 1..=M_SAMPLES_STAGE_MAX {
        let n_k = stage_start(k);
        if n_k == u32::MAX {
            break;
        }
        radii.push(beta(n_k, delta_v, sigma));
    }
    radii.shrink_to_fit();
    radii
}

/// The `m_v(ρ)` walk over precomputed stage radii: the first stage whose
/// radius certifies `width_target`, as the sample count n_k; `u32::MAX`
/// past the table (unreachable for honest widths — the caller's iteration
/// cap is the belt-and-braces backstop).
fn m_samples_with(radii: &[f64], width_target: f64) -> u32 {
    debug_assert!(width_target.is_finite() && width_target > 0.0);
    for (i, &r) in radii.iter().enumerate() {
        if 2.0 * r <= width_target {
            return stage_start(i as u32 + 1);
        }
    }
    u32::MAX
}

/// Reference form (test-side): builds the stage table per call and walks it
/// — the spec [`m_samples_with`]'s precomputed table is pinned against. The
/// production search never calls this (its table is built once per search;
/// the profiled pre-table form rebuilt it per call — the hot-path finding).
#[cfg(test)]
fn m_samples(width_target: f64, delta_v: f64, sigma: f64) -> u32 {
    m_samples_with(&stage_radii(delta_v, sigma), width_target)
}

/// The certified two-fidelity search (plan T1.5/T1.6; paper Algorithm 1).
///
/// Expands the root (fast-querying its children — root is a Max node per
/// paper Definition 2.1), then loops: leader = argmax lower endpoint,
/// challenger = argmax upper endpoint among the rest; stop when
/// `L_â ≥ max_{a≠â} U_a − ε` (ε-stop). Each unresolved iteration refines the
/// *coarser* of (leader's L-side, challenger's U-side) by one work unit via
/// the budgeted local-vs-recursive race. PAC soundness rides only on
/// interval validity (honest envelope + time-uniform slow CIs, union δ) and
/// the stop rule — the race budget is a route-selection heuristic and never
/// gates correctness; `rho_0 = 0` (all root intervals exact) exits through
/// the stop rule on the first pass, and the empty-intersection guard
/// (detectable `E_δ` violation) terminates with `certified = false` and the
/// current leader as the default action (paper B.3 convention).
///
/// Config validation is asserted here (`ε ≥ 0` finite, `δ ∈ (0,1)`, positive
/// finite `c`/`σ`, `1 ≤ node_cap ≤ NODE_CAP`).
pub fn two_fidelity_search<S: MinimaxSpace, R: RngCore>(
    space: &S,
    root: u32,
    config: &SearchConfig,
    rng: &mut R,
) -> TwoFidelityResult {
    assert!(
        config.epsilon.is_finite() && config.epsilon >= 0.0,
        "epsilon must be finite and >= 0"
    );
    assert!(config.delta > 0.0 && config.delta < 1.0, "delta in (0, 1)");
    assert!(
        config.slow_cost.is_finite() && config.slow_cost > 0.0,
        "slow_cost must be finite and > 0"
    );
    assert!(config.sigma.is_finite() && config.sigma > 0.0, "sigma > 0");
    assert!(
        config.node_cap >= 1 && config.node_cap <= NODE_CAP,
        "node_cap in [1, NODE_CAP]"
    );
    assert_eq!(space.kind(root), NodeKind::Max, "root is a Max node");

    let mut search = Searcher {
        space,
        config,
        rng,
        states: HashMap::with_capacity(1024),
        rho0: 0.0,
        cost: Cost::default(),
        stage_radii: None,
        m_exp: HashMap::new(),
    };
    // The root's own bookkeeping node (its interval is never read — only
    // kind/depth for expansion bookkeeping).
    let root_depth = space.remaining_depth(root);
    search.states.insert(
        root,
        NodeState {
            interval: NodeInterval::from_fast(0.0, 0.0),
            kind: NodeKind::Max,
            depth: root_depth,
            children: ArrayVec::new(),
            expanded: false,
            done: [0; 2],
            rec_spend: 0.0,
            slow_n: 0,
        },
    );
    let root_children = search.expand(root);
    assert!(!root_children.is_empty(), "root must expose >= 1 child");

    // ρ_0 = widest root-child interval (paper §3.1); the dyadic ladder is
    // ρ_k = ρ_0 · 2^-k. ρ_0 = 0 (all root intervals exact) is the T1.6 early
    // exit — the first stop-rule pass below fires on exact points.
    search.rho0 = root_children
        .iter()
        .map(|c| search.width_of(*c))
        .fold(0.0_f64, f64::max);

    let epsilon = f64::from(config.epsilon);
    // Belt-and-braces only: every work unit shrinks an interval, expands a
    // bounded node, or latches a scale, so the loop is finite on honest
    // inputs (paper Thm 3.6); the cap turns pathological inputs into the
    // B.3 default-action exit instead of a hang.
    let max_iters = 64 * u64::from(config.node_cap);
    let mut iters: u64 = 0;

    loop {
        // Guard first (T1.6): an empty effective interval at the root is a
        // detectable E_δ violation — terminate + default action, certified
        // = false. On honest oracles this never fires (every interval
        // contains V* on E_δ, and max/min backups preserve containment).
        if root_children.iter().any(|c| search.is_empty(*c)) {
            return search.finish(root_children, false);
        }

        // Leader = argmax L (tie → lowest handle); challenger's margin =
        // max U over the rest.
        let leader = search.pick_extreme(root_children.as_slice(), Side::Lower, true);
        let leader_l = search.endpoint(leader, Side::Lower);
        let challenger_u = root_children
            .iter()
            .filter(|c| **c != leader)
            .map(|c| search.endpoint(*c, Side::Upper))
            .fold(f64::NEG_INFINITY, f64::max);

        // ε-stop (paper §3.2): single-child roots stop here trivially.
        if leader_l >= challenger_u - epsilon {
            return search.finish(root_children, true);
        }
        if iters >= max_iters {
            return search.finish(root_children, false);
        }

        // Refine the coarser unresolved side of the leader/challenger pair
        // (tie → the leader's L-side, deterministic).
        let challenger =
            search.pick_extreme_excluding(root_children.as_slice(), Side::Upper, true, leader);
        if config.lazy_discharge {
            // Reference `root_obligation`: the leader's L-side at its active
            // scale vs the challenger's coarser contender side; no finite
            // obligation left ⇒ the reference's uncertified exit.
            match search.ld_root_obligation(leader, challenger) {
                Some((node, side, k)) => {
                    search.ld_resolve(node, side, k, f64::INFINITY);
                }
                None => return search.finish(root_children, false),
            }
            iters += 1;
            continue;
        }
        let target = if search.width_of(leader) >= search.width_of(challenger) {
            (leader, Side::Lower)
        } else {
            (challenger, Side::Upper)
        };
        search.resolve_step(target.0, target.1);
        iters += 1;
    }
}

struct Searcher<'a, S: MinimaxSpace, R: RngCore> {
    space: &'a S,
    config: &'a SearchConfig,
    rng: &'a mut R,
    states: HashMap<u32, NodeState>,
    rho0: f64,
    cost: Cost,
    /// Precomputed stage radii for the search's fixed (δ_v, σ) — the
    /// `m_v(ρ)` walk ([`m_samples_with`]) reads this table instead of
    /// re-evaluating β per stage per call (the profiled hot path). Built
    /// LAZILY on first resolution: a search whose ε-stop fires before any
    /// `resolve_step` never touches β, which keeps σ = 0 exact-oracle
    /// configs legal exactly as the pre-table lazy form did (β asserts
    /// σ > 0; reaching m_v with σ = 0 panics identically either way).
    stage_radii: Option<Vec<f64>>,
    /// Lazy-discharge mode only: recursive-route spend per (node, scale) —
    /// the reference's `m_exp`, compared against the Eq. 7 race budget at
    /// that scale. Empty (never touched) in v1 mode.
    m_exp: HashMap<(u32, u8), f64>,
}

/// Outcome of one lazy-discharge resolution unit (the reference's
/// `Status`): latched/certified with no spend, made progress, or refused
/// because the caller's remaining recursive budget cannot pay for the unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LdStatus {
    Cert,
    Progress,
    Blocked,
}

/// Finest dyadic scale the `done` bitmask can latch (`u64` per side).
const LD_MAX_SCALE: u8 = 63;

impl<'a, S: MinimaxSpace, R: RngCore> Searcher<'a, S, R> {
    fn delta_v(&self) -> f64 {
        self.config.delta_for_node()
    }

    fn state(&mut self, node: u32) -> &mut NodeState {
        self.states.get_mut(&node).expect("node state exists")
    }

    fn width_of(&self, node: u32) -> f64 {
        let (lo, hi) = self.states[&node].interval.effective();
        hi - lo
    }

    fn endpoint(&self, node: u32, side: Side) -> f64 {
        let (lo, hi) = self.states[&node].interval.effective();
        match side {
            Side::Lower => lo,
            Side::Upper => hi,
        }
    }

    fn is_empty(&self, node: u32) -> bool {
        let (lo, hi) = self.states[&node].interval.effective();
        lo > hi
    }

    /// Expand `node`: reveal + fast-query all children (b fast queries),
    /// create their interval states, install the Eq. 6 backup. Returns the
    /// children (root call) and charges `Cost.fast`.
    fn expand(&mut self, node: u32) -> ArrayVec<u32, MAX_CHILDREN> {
        let kind = self.state(node).kind;
        let mut children = ArrayVec::new();
        self.space.children_into(node, &mut children);
        for &child in children.as_slice() {
            let child_depth = self.space.remaining_depth(child);
            let child_bias = self.space.bias_envelope(child_depth);
            let fast = self.space.fast_value(child);
            self.cost.fast += 1;
            self.states.insert(
                child,
                NodeState {
                    interval: NodeInterval::from_fast(fast, child_bias),
                    kind: self.space.kind(child),
                    depth: child_depth,
                    children: ArrayVec::new(),
                    expanded: false,
                    done: [0; 2],
                    rec_spend: 0.0,
                    slow_n: 0,
                },
            );
        }
        let st = self.state(node);
        st.children = children.clone();
        st.expanded = true;
        self.refresh_backup(node, kind);
        children
    }

    fn refresh_backup(&mut self, node: u32, kind: NodeKind) {
        let children = self.state(node).children.clone();
        let mut pairs = ArrayVec::<(f64, f64), MAX_CHILDREN>::new();
        for &child in children.as_slice() {
            let (lo, hi) = self.states[&child].interval.effective();
            let _ = pairs.try_push((lo, hi));
        }
        if !pairs.is_empty() {
            self.state(node)
                .interval
                .set_child_backup(kind, pairs.as_slice());
        }
    }

    /// The live/blocking child for `(node, side)` — selector cases resolve
    /// the endpoint witness; comparison cases the still-live extreme
    /// (children outside the ρ_k/2 margin are discharged lazily by never
    /// being picked). Ties → lowest handle; empty-effective children are
    /// skipped (guard territory). Returns `None` when every child is empty
    /// (the caller falls back to the local route — local reversibility).
    fn blocking_child(&self, node: u32, side: Side) -> Option<u32> {
        let st = self.states.get(&node)?;
        let want_max = st.kind == NodeKind::Max;
        let mut best: Option<u32> = None;
        let mut best_key = 0.0_f64;
        for &child in st.children.as_slice() {
            if self.is_empty(child) {
                continue;
            }
            let key = self.endpoint(child, side);
            let better = match best {
                None => true,
                Some(b) => {
                    let improves = if want_max {
                        key > best_key
                    } else {
                        key < best_key
                    };
                    improves || (key == best_key && child < b)
                }
            };
            if better {
                best = Some(child);
                best_key = key;
            }
        }
        best
    }

    /// Arg-extreme over `children` by effective endpoint (ties → lowest
    /// handle). Root-loop selector for leader/challenger.
    fn pick_extreme(&self, children: &[u32], side: Side, want_max: bool) -> u32 {
        self.pick_extreme_excluding(children, side, want_max, u32::MAX)
    }

    fn pick_extreme_excluding(
        &self,
        children: &[u32],
        side: Side,
        want_max: bool,
        exclude: u32,
    ) -> u32 {
        let mut best: Option<u32> = None;
        let mut best_key = 0.0_f64;
        for &child in children {
            if child == exclude {
                continue;
            }
            let key = self.endpoint(child, side);
            let better = match best {
                None => true,
                Some(b) => {
                    let improves = if want_max {
                        key > best_key
                    } else {
                        key < best_key
                    };
                    improves || (key == best_key && child < b)
                }
            };
            if better {
                best = Some(child);
                best_key = key;
            }
        }
        best.unwrap_or_else(|| children.first().copied().unwrap_or(exclude))
    }

    /// One local slow sample (the local route; always available — paper's
    /// local reversibility). Returns the unified cost spent (`c`).
    fn sample_local(&mut self, node: u32) -> f64 {
        let (sigma, delta_v) = (self.config.sigma, self.delta_v());
        let sample = self.space.slow_sample(node, self.rng);
        self.cost.slow += 1;
        self.state(node).slow_n += 1;
        self.state(node)
            .interval
            .observe_slow(sample, sigma, delta_v);
        f64::from(self.config.slow_cost)
    }

    /// One resolution work unit toward certifying `side` of `node` at the
    /// smallest unresolved scale (T1.5). Returns the unified cost spent.
    fn resolve_step(&mut self, node: u32, side: Side) -> f64 {
        let s = side.index();
        let (done_bits, depth, expanded) = {
            let st = self.state(node);
            (st.done[s], st.depth, st.expanded)
        };
        // Scale ladder: smallest unresolved k. A fully-latched ladder (all
        // 64 scales) cannot certify finer — degrade to the local route; the
        // root loop's cap remains the backstop.
        let k = done_bits.trailing_ones();
        if k >= 64 {
            return self.sample_local(node);
        }
        let rho = self.rho0 * (-(f64::from(k))).exp2();

        // Observable certificate: effective width within ρ_k/2 → latch.
        if self.width_of(node) <= rho / 2.0 {
            self.state(node).done[s] |= 1 << k;
            return 0.0;
        }

        // Leaves have B(0) = 0 (recursion is structurally impossible and
        // Eq. 7 returns Γ = 0): the local route is the only route.
        if depth == 0 {
            return self.sample_local(node);
        }

        // Eq. 7 race budget; recursive spend is tracked per node (v1
        // simplification: cumulative across scales — conservative, and never
        // correctness-bearing: both routes only tighten valid intervals).
        let bias = f64::from(self.space.bias_envelope(depth));
        if self.stage_radii.is_none() {
            self.stage_radii = Some(stage_radii(self.config.delta_for_node(), self.config.sigma));
        }
        let m = m_samples_with(self.stage_radii.as_deref().unwrap_or(&[]), rho / 2.0);
        let gamma = gamma_v(bias, rho, self.config.slow_cost, m);
        let budget = race_scale(depth) * gamma;
        let spend = self.state(node).rec_spend;
        if gamma > 0.0 && spend < budget {
            if !expanded {
                let children = self.expand(node);
                self.state(node).rec_spend += f64::from(children.len() as u32);
            }
            if let Some(child) = self.blocking_child(node, side) {
                let spent = self.resolve_step(child, side);
                let kind = self.state(node).kind;
                self.refresh_backup(node, kind);
                self.state(node).rec_spend += spent;
                return spent;
            }
        }
        self.sample_local(node)
    }

    // ── Lazy-discharge resolver (Issue 916 b-lever; `lazy_discharge`) ──────
    //
    // A transcription of the reference implementation's resolver; method
    // names keep the reference's (`cert`, `comp`, `active_children`,
    // `capped_scale`, `child_obligation`, `resolve`) so the two can be read
    // side by side. Scales are `u8` in `[0, LD_MAX_SCALE]`; `None` stands
    // for the reference's infinite scale.

    fn ld_rho(&self, k: u8) -> f64 {
        self.rho0.max(1e-12) * (-f64::from(k)).exp2()
    }

    /// The dyadic band of the current width: `ρ_k/2 < w ≤ ρ_k` (`None` once
    /// the width is numerically zero — nothing left to certify).
    fn ld_width_scale(&self, node: u32) -> Option<u8> {
        let w = self.width_of(node).max(0.0);
        if w <= 1e-14 {
            return None;
        }
        if w > self.ld_rho(0) {
            return Some(0);
        }
        (0..=LD_MAX_SCALE)
            .find(|&k| {
                let rho = self.ld_rho(k);
                rho / 2.0 < w && w <= rho
            })
            .or(Some(LD_MAX_SCALE))
    }

    fn ld_is_done(&self, node: u32, side: Side, k: u8) -> bool {
        self.states[&node].done[side.index()] >> k & 1 == 1
    }

    fn ld_set_done(&mut self, node: u32, side: Side, k: u8) {
        self.state(node).done[side.index()] |= 1 << k;
    }

    /// `Comp_s(v, k)`: latched, or observably certified now (latched on the
    /// spot — the evaluate-time latch documented on the config knob).
    fn ld_comp(&mut self, node: u32, side: Side, k: u8) -> bool {
        if self.ld_is_done(node, side, k) {
            return true;
        }
        let certified = self.ld_cert(node, side, k);
        if certified {
            self.ld_set_done(node, side, k);
        }
        certified
    }

    fn ld_is_comparison(&self, node: u32, side: Side) -> bool {
        matches!(
            (self.states[&node].kind, side),
            (NodeKind::Max, Side::Lower) | (NodeKind::Min, Side::Upper)
        )
    }

    /// `Cert_s(v, k)` (reference `cert`): width within `ρ_k/2`; or, once
    /// expanded, the comparison-case discharge (every child either outside
    /// the margin of the parent's bound or itself certified) / the
    /// selector-case witness certification.
    fn ld_cert(&mut self, node: u32, side: Side, k: u8) -> bool {
        let tol = self.ld_rho(k) / 2.0;
        if self.width_of(node) <= tol {
            return true;
        }
        let (kind, expanded) = {
            let st = &self.states[&node];
            (st.kind, st.expanded)
        };
        if !expanded {
            return false;
        }
        let children = self.states[&node].children.clone();
        if children.is_empty() {
            return false;
        }
        match (kind, side) {
            (NodeKind::Max, Side::Lower) => {
                let lambda = children
                    .iter()
                    .map(|&c| self.endpoint(c, Side::Lower))
                    .fold(f64::NEG_INFINITY, f64::max);
                children.iter().all(|&c| {
                    self.endpoint(c, Side::Upper) <= lambda + tol || self.ld_comp(c, Side::Lower, k)
                })
            }
            (NodeKind::Min, Side::Upper) => {
                let lambda = children
                    .iter()
                    .map(|&c| self.endpoint(c, Side::Upper))
                    .fold(f64::INFINITY, f64::min);
                children.iter().all(|&c| {
                    self.endpoint(c, Side::Lower) >= lambda - tol || self.ld_comp(c, Side::Upper, k)
                })
            }
            _ => {
                let active = self.ld_active_children(node, side, k);
                !active.is_empty() && active.iter().all(|&c| self.ld_comp(c, side, k))
            }
        }
    }

    /// Reference `active_children`: selector cases → the endpoint witnesses
    /// (exact ties kept); comparison cases → the children still inside the
    /// `ρ_k/2` margin and not yet certified (the lazily-undischarged set).
    fn ld_active_children(&mut self, node: u32, side: Side, k: u8) -> ArrayVec<u32, MAX_CHILDREN> {
        let tol = self.ld_rho(k) / 2.0;
        let kind = self.states[&node].kind;
        let children = self.states[&node].children.clone();
        let mut out = ArrayVec::new();
        if children.is_empty() {
            return out;
        }
        match (kind, side) {
            (NodeKind::Min, Side::Lower) | (NodeKind::Max, Side::Upper) => {
                let want_max = kind == NodeKind::Max;
                let best = children.iter().map(|&c| self.endpoint(c, side)).fold(
                    if want_max {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    },
                    |a, b| {
                        if want_max { a.max(b) } else { a.min(b) }
                    },
                );
                for &c in children.as_slice() {
                    if (self.endpoint(c, side) - best).abs() <= 1e-14 {
                        out.push(c);
                    }
                }
            }
            (NodeKind::Max, Side::Lower) => {
                let lambda = children
                    .iter()
                    .map(|&c| self.endpoint(c, Side::Lower))
                    .fold(f64::NEG_INFINITY, f64::max);
                for &c in children.as_slice() {
                    if self.endpoint(c, Side::Upper) > lambda + tol
                        && !self.ld_comp(c, Side::Lower, k)
                    {
                        out.push(c);
                    }
                }
            }
            (NodeKind::Min, Side::Upper) => {
                let lambda = children
                    .iter()
                    .map(|&c| self.endpoint(c, Side::Upper))
                    .fold(f64::INFINITY, f64::min);
                for &c in children.as_slice() {
                    if self.endpoint(c, Side::Lower) < lambda - tol
                        && !self.ld_comp(c, Side::Upper, k)
                    {
                        out.push(c);
                    }
                }
            }
        }
        out
    }

    /// Reference `active_scale`: the first uncertified scale at or below the
    /// width's own band.
    fn ld_active_scale(&mut self, node: u32, side: Side) -> Option<u8> {
        let start = self.ld_width_scale(node)?;
        (start..=LD_MAX_SCALE).find(|&k| !self.ld_comp(node, side, k))
    }

    /// Reference `capped_scale`: the coarsest uncertified scale **no finer
    /// than the parent's** — the propagated-resolution cap.
    fn ld_capped_scale(&mut self, node: u32, side: Side, parent_k: u8) -> Option<u8> {
        (0..=parent_k).find(|&k| !self.ld_comp(node, side, k))
    }

    /// Reference `contender_scale`: the challenger's coarser side (tie → L).
    fn ld_contender_scale(&mut self, node: u32) -> (Side, Option<u8>) {
        let k_l = self.ld_active_scale(node, Side::Lower);
        let k_u = self.ld_active_scale(node, Side::Upper);
        match (k_l, k_u) {
            (Some(l), Some(u)) if l <= u => (Side::Lower, k_l),
            (Some(_), None) => (Side::Lower, k_l),
            _ => (Side::Upper, k_u),
        }
    }

    /// Reference `root_obligation` (leader/challenger already picked by the
    /// shared root loop — identical selectors and tie-breaks).
    fn ld_root_obligation(&mut self, leader: u32, challenger: u32) -> Option<(u32, Side, u8)> {
        let k_leader = self.ld_active_scale(leader, Side::Lower);
        let (side_c, k_c) = self.ld_contender_scale(challenger);
        match (k_leader, k_c) {
            (Some(kl), Some(kc)) if kl <= kc => Some((leader, Side::Lower, kl)),
            (Some(kl), None) => Some((leader, Side::Lower, kl)),
            (_, Some(kc)) => Some((challenger, side_c, kc)),
            (None, None) => None,
        }
    }

    /// Reference `child_obligation`: comparison cases pick the live blocker
    /// (max U under Max-L, min L under Min-U) and refine its OPPOSITE side
    /// first; selector cases pick the widest uncertified witness. Both at
    /// the parent-capped scale (`None` = nothing left below the cap).
    fn ld_child_obligation(
        &mut self,
        node: u32,
        side: Side,
        k: u8,
    ) -> Option<(u32, Side, Option<u8>)> {
        if self.ld_is_comparison(node, side) {
            let active = self.ld_active_children(node, side, k);
            if active.is_empty() {
                return None;
            }
            let want_max = self.states[&node].kind == NodeKind::Max;
            let opposite = match side {
                Side::Lower => Side::Upper,
                Side::Upper => Side::Lower,
            };
            // Blocker: max U (Max) / min L (Min) — i.e. the opposite
            // endpoint; first occurrence on ties (Python `max`/`min`).
            let mut blocker = active[0];
            for &c in active.iter().skip(1) {
                let (kc, kb) = (self.endpoint(c, opposite), self.endpoint(blocker, opposite));
                if (want_max && kc > kb) || (!want_max && kc < kb) {
                    blocker = c;
                }
            }
            if let Some(k_opp) = self.ld_capped_scale(blocker, opposite, k) {
                return Some((blocker, opposite, Some(k_opp)));
            }
            let k_same = self.ld_capped_scale(blocker, side, k);
            return Some((blocker, side, k_same));
        }
        let active = self.ld_active_children(node, side, k);
        let mut pick: Option<u32> = None;
        for &c in active.as_slice() {
            if self.ld_comp(c, side, k) {
                continue;
            }
            // Widest first; first occurrence on ties (Python `max`).
            if pick.is_none_or(|p| self.width_of(c) > self.width_of(p)) {
                pick = Some(c);
            }
        }
        let child = pick?;
        let ck = self.ld_capped_scale(child, side, k);
        Some((child, side, ck))
    }

    /// Reference `local_step`: one slow sample unless the caller's cap
    /// cannot pay for it.
    fn ld_local_step(&mut self, node: u32, cap: f64) -> (f64, LdStatus) {
        if cap < f64::from(self.config.slow_cost) {
            return (0.0, LdStatus::Blocked);
        }
        (self.sample_local(node), LdStatus::Progress)
    }

    /// Eq. 7 race budget at `(node, k)` — the same rule v1 uses, evaluated
    /// per scale.
    fn ld_race_budget(&mut self, node: u32, k: u8) -> f64 {
        let depth = self.states[&node].depth;
        let rho = self.ld_rho(k);
        let bias = f64::from(self.space.bias_envelope(depth));
        if self.stage_radii.is_none() {
            self.stage_radii = Some(stage_radii(self.config.delta_for_node(), self.config.sigma));
        }
        let m = m_samples_with(self.stage_radii.as_deref().unwrap_or(&[]), rho / 2.0);
        race_scale(depth) * gamma_v(bias, rho, self.config.slow_cost, m)
    }

    /// Reference `resolve(node, side, scale, cap)`: one work unit toward
    /// `Comp_side(node, k)` — latch, local sample, expansion, or one
    /// recursive unit into the obligation child, under the caller's
    /// remaining recursive budget `cap`.
    fn ld_resolve(&mut self, node: u32, side: Side, k: u8, cap: f64) -> (f64, LdStatus) {
        if self.ld_comp(node, side, k) {
            return (0.0, LdStatus::Cert);
        }
        let (depth, expanded) = {
            let st = &self.states[&node];
            (st.depth, st.expanded)
        };
        if depth == 0 {
            return self.ld_local_step(node, cap);
        }
        let spent = self.m_exp.get(&(node, k)).copied().unwrap_or(0.0);
        let remaining = self.ld_race_budget(node, k) - spent;
        if remaining <= 0.0 {
            return self.ld_local_step(node, cap);
        }
        let rec_cap = cap.min(remaining);

        if !expanded {
            let mut probe = ArrayVec::<u32, MAX_CHILDREN>::new();
            self.space.children_into(node, &mut probe);
            let child_cost = probe.len() as f64;
            if child_cost > remaining {
                return self.ld_local_step(node, cap);
            }
            if child_cost > cap {
                return (0.0, LdStatus::Blocked);
            }
            let children = self.expand(node);
            let cost = children.len() as f64;
            *self.m_exp.entry((node, k)).or_insert(0.0) += cost;
            return (cost, LdStatus::Progress);
        }

        let Some((child, child_side, child_k)) = self.ld_child_obligation(node, side, k) else {
            if self.ld_comp(node, side, k) {
                return (0.0, LdStatus::Cert);
            }
            return self.ld_local_step(node, cap);
        };
        let Some(child_k) = child_k else {
            return self.ld_local_step(node, cap);
        };
        let (q, status) = self.ld_resolve(child, child_side, child_k, rec_cap);
        let kind = self.states[&node].kind;
        self.refresh_backup(node, kind);
        if status == LdStatus::Blocked {
            if cap < remaining {
                return (0.0, LdStatus::Blocked);
            }
            return self.ld_local_step(node, cap);
        }
        *self.m_exp.entry((node, k)).or_insert(0.0) += q;
        (q, status)
    }

    /// Freeze the result: leader (argmax L, tie → lowest handle), the cost
    /// ledger, and the root children's effective intervals. When tracing,
    /// emit the per-node sample ledger (Issue 916 cause-(b) measurement).
    fn finish(
        &self,
        root_children: ArrayVec<u32, MAX_CHILDREN>,
        certified: bool,
    ) -> TwoFidelityResult {
        let best_action = self.pick_extreme(root_children.as_slice(), Side::Lower, true);
        let mut root_intervals = Vec::with_capacity(root_children.len());
        for child in root_children.iter() {
            let (lo, hi) = self.states[child].interval.effective();
            root_intervals.push((*child, lo, hi));
        }
        let node_stats = if self.config.trace_nodes {
            let rho0 = self.rho0;
            let mut rows: Vec<NodeStat> = self
                .states
                .iter()
                .filter(|(_, st)| st.slow_n > 0)
                .map(|(&node, st)| {
                    let k_unresolved = st.done[0].trailing_ones().min(st.done[1].trailing_ones());
                    let target = if k_unresolved >= 64 {
                        0.0
                    } else {
                        rho0 * (-(f64::from(k_unresolved))).exp2() / 2.0
                    };
                    NodeStat {
                        node,
                        kind: st.kind,
                        depth: st.depth,
                        slow_n: st.slow_n,
                        done: st.done,
                        final_width: {
                            let (lo, hi) = st.interval.effective();
                            hi - lo
                        },
                        target_width: target,
                    }
                })
                .collect();
            rows.sort_by(|a, b| b.slow_n.cmp(&a.slow_n).then(a.node.cmp(&b.node)));
            rows
        } else {
            Vec::new()
        };
        TwoFidelityResult {
            best_action,
            cost: self.cost,
            certified,
            root_intervals,
            node_stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64* — no global RNG (the global_rng_gate bans
    /// unseeded free-function draws; tests seed their own).
    struct Xs(u64);
    impl RngCore for Xs {
        fn f64(&mut self) -> f64 {
            // xorshift64* (Vigna), top 53 bits to [0,1).
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            let z = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((z >> 11) as f64) / (1u64 << 53) as f64
        }
    }

    // ── T1.3: beta monotone / limit / coverage ───────────────────────────

    #[test]
    fn beta_is_nonincreasing_in_n_across_delta_range() {
        for &delta_v in &[0.1, 1e-3, 1e-7] {
            let sigma = 0.5;
            let mut prev = f64::INFINITY;
            for n in 1..=200_000u32 {
                let b = beta(n, delta_v, sigma);
                assert!(
                    b <= prev + 1e-15,
                    "beta increased at n={n} delta_v={delta_v}: {prev} -> {b}"
                );
                prev = b;
            }
        }
    }

    #[test]
    fn beta_vanishes_as_n_grows() {
        let b_1k = beta(1_000, 1e-6, 0.5);
        let b_1m = beta(1_000_000, 1e-6, 0.5);
        assert!(b_1m < 0.05 * b_1k, "beta must decay: {b_1k} -> {b_1m}");
    }

    #[test]
    fn beta_holds_simultaneously_over_trajectories() {
        // Coverage ≥ 1 − δ_v over many full trajectories (the plan's G1
        // pre-arm). U[0,1] is 0.5-sub-Gaussian; seeded xorshift only.
        const N_TRAJ: u32 = 2_000;
        const N_STEPS: u32 = 2_000;
        const DELTA_V: f64 = 0.01;
        const SIGMA: f64 = 0.5;
        let mu = 0.5;
        let mut violations = 0u32;
        for t in 0..N_TRAJ {
            let mut rng = Xs(0x9E37_79B9_7F4A_7C15 ^ (u64::from(t) << 1 | 1));
            let mut sum = 0.0f64;
            for n in 1..=N_STEPS {
                sum += rng.f64();
                let mean = sum / f64::from(n);
                if (mean - mu).abs() > beta(n, DELTA_V, SIGMA) {
                    violations += 1;
                    break;
                }
            }
        }
        // Expected ≈ δ_v·N_TRAJ = 20; the bound must hold at ≥ 1−δ_v per
        // trajectory, so > 40 flags a broken bound (Poisson tail ~5e-4).
        assert!(
            violations * 2 <= N_TRAJ,
            "coverage violated in {violations}/{N_TRAJ} trajectories (bound δ_v={DELTA_V})"
        );
    }

    // ── T1.2: interval machinery ─────────────────────────────────────────

    #[test]
    fn slow_intersection_is_monotone_nesting() {
        // Lemma B.2: successive slow CIs shrink the interval (debug assert
        // inside observe_slow also fires); endpoints move monotonically.
        let mut iv = NodeInterval::from_fast(0.5, 0.4);
        let mut rng = Xs(0xDEAD_BEEF_CAFE_0001);
        // Hand-rolled: absorb draws from a tightly pinned distribution.
        let (mut lo, mut hi) = iv.local();
        for i in 0..200u32 {
            // Alternating bias so any widening would be caught.
            let y = if i % 2 == 0 { 0.60 } else { 0.40 } + 0.01 * rng.f64();
            iv.observe_slow(y as f32, 0.5, 1e-6);
            let (nlo, nhi) = iv.local();
            assert!(nlo >= lo - 1e-12, "local lo moved left: {lo} -> {nlo}");
            assert!(nhi <= hi + 1e-12, "local hi moved right: {hi} -> {nhi}");
            lo = nlo;
            hi = nhi;
        }
        assert_eq!(iv.slow_samples(), 200);
    }

    #[test]
    fn backup_preserves_width_lemma_2_3() {
        // Max over children of width-w intervals has width ≤ w; same for Min.
        let children = [(0.0, 0.5), (0.1, 0.4), (-0.2, 0.3)];
        let mut max_node = NodeInterval::from_fast(0.0, 0.0);
        max_node.set_child_backup(NodeKind::Max, &children);
        let (l, h) = max_node.effective();
        assert!((h - l) <= 0.5 + 1e-12, "max backup widened: [{l}, {h}]");
        let mut min_node = NodeInterval::from_fast(0.0, 0.0);
        min_node.set_child_backup(NodeKind::Min, &children);
        let (l, h) = min_node.effective();
        assert!((h - l) <= 0.5 + 1e-12, "min backup widened: [{l}, {h}]");
        // Eq. 6 values: Max → [max L, max U] = [0.1, 0.5]; Min → [−0.2, 0.3].
        let mut m = NodeInterval::from_fast(0.0, 0.0);
        m.set_child_backup(NodeKind::Max, &children);
        assert_eq!(m.child_backup(), Some((0.1, 0.5)));
        let mut m = NodeInterval::from_fast(0.0, 0.0);
        m.set_child_backup(NodeKind::Min, &children);
        assert_eq!(m.child_backup(), Some((-0.2, 0.3)));
    }

    #[test]
    fn effective_intersects_local_with_child() {
        let mut iv = NodeInterval::from_fast(0.5, 0.1); // fast [0.4, 0.6]
        iv.set_child_backup(NodeKind::Max, &[(0.45, 0.55)]);
        // local (no slow) = [0.4, 0.6]; effective = [0.45, 0.55].
        assert_eq!(iv.effective(), (0.45, 0.55));
    }

    // ── T1.4 helpers ─────────────────────────────────────────────────────

    #[test]
    fn slow_only_carries_the_slow_ci_and_the_child_fold_alone() {
        // The oracle-less carrier (Issue 916's LUCB-MCTS baseline): local is
        // the running slow CI, never an accidental [0, 0] fast point; an
        // installed child fold passes through `effective` untouched.
        let mut leaf = NodeInterval::slow_only();
        assert_eq!(leaf.local(), (f64::NEG_INFINITY, f64::INFINITY));
        leaf.observe_slow(0.3, 0.05, 1e-3);
        let (lo, hi) = leaf.local();
        assert!(lo.is_finite() && hi.is_finite() && hi > lo);
        let mut node = NodeInterval::slow_only();
        node.set_child_backup(NodeKind::Min, &[(0.2, 0.4), (0.3, 0.9)]);
        assert_eq!(node.effective(), (0.2, 0.4));
    }

    #[test]
    fn delta_allocation_sums_to_delta() {
        let cfg = SearchConfig {
            epsilon: 0.1,
            delta: 0.05,
            slow_cost: 4.0,
            node_cap: NODE_CAP,
            sigma: 0.5,
            trace_nodes: false,
            lazy_discharge: false,
        };
        let per = cfg.delta_for_node();
        assert!((per * f64::from(NODE_CAP) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn race_scale_matches_h_plus_one_squared() {
        assert_eq!(race_scale(0), 1.0);
        assert_eq!(race_scale(3), 16.0);
        assert_eq!(race_scale(9), 100.0);
    }

    #[test]
    fn stage_grid_and_index_are_consistent() {
        for n in 1..=100_000u32 {
            let k = stage_index(n);
            assert!(stage_start(k) <= n, "n={n} k={k}");
            assert!(stage_start(k + 1) > n, "n={n} k={k}");
            assert!(k >= 1);
        }
    }

    #[test]
    fn unified_cost_model_matches_definition() {
        let c = Cost { fast: 100, slow: 7 };
        assert!((c.unified(4.0) - 128.0).abs() < 1e-12);
        assert_eq!(Cost::default().unified(3.0), 0.0);
    }

    // ── T1.5/T1.6: the root leader/challenger search ────────────────────

    /// Hand-built minimax tree: linear node ids, node 0 = root (Max). True
    /// leaf values are the slow means; `exact()` folds true minimax. The
    /// envelope is the CONTRACT shape `bias_scale · (1 − 2^-h)` (Issue 915
    /// en-route note: the pre-fix `bias_scale · 2^-h` was DECREASING in h
    /// with B(0) = bias_scale ≠ 0, contradicting the trait contract it
    /// documents); slow noise is uniform ±`noise_amp` (sub-Gaussian
    /// parameter `noise_amp/√3`, so a declared σ ≥ noise_amp is honest).
    /// B(0) = 0 pins depth-0 nodes' fast oracle EXACT — tests wanting a
    /// loose fast oracle push children at depth ≥ 1.
    struct VecTree {
        kind: Vec<NodeKind>,
        depth: Vec<u8>,
        children: Vec<Vec<u32>>,
        fast: Vec<f32>,
        slow: Vec<f32>,
        bias_scale: f32,
        noise_amp: f32,
    }

    impl VecTree {
        fn new(bias_scale: f32, noise_amp: f32) -> Self {
            Self {
                kind: vec![NodeKind::Max],
                depth: vec![0],
                children: vec![Vec::new()],
                fast: vec![0.0],
                slow: vec![0.0],
                bias_scale,
                noise_amp,
            }
        }

        fn push(&mut self, parent: u32, kind: NodeKind, depth: u8, fast: f32, slow: f32) -> u32 {
            let id = self.kind.len() as u32;
            self.kind.push(kind);
            self.depth.push(depth);
            self.children.push(Vec::new());
            self.fast.push(fast);
            self.slow.push(slow);
            self.children[parent as usize].push(id);
            id
        }

        fn exact(&self, node: u32) -> f32 {
            match self.children[node as usize].as_slice() {
                &[] => self.slow[node as usize],
                kids => {
                    let mut vals = kids.iter().map(|c| self.exact(*c));
                    let first = vals.next().unwrap_or(0.0);
                    match self.kind[node as usize] {
                        NodeKind::Max => vals.fold(first, f32::max),
                        NodeKind::Min => vals.fold(first, f32::min),
                    }
                }
            }
        }
    }

    impl MinimaxSpace for VecTree {
        fn kind(&self, node: u32) -> NodeKind {
            self.kind[node as usize]
        }

        fn remaining_depth(&self, node: u32) -> u8 {
            self.depth[node as usize]
        }

        fn children_into(&self, node: u32, out: &mut ArrayVec<u32, MAX_CHILDREN>) {
            for &c in &self.children[node as usize] {
                out.push(c);
            }
        }

        fn fast_value(&self, node: u32) -> f32 {
            self.fast[node as usize]
        }

        fn bias_envelope(&self, remaining_depth: u8) -> f32 {
            self.bias_scale * (1.0 - 2.0_f32.powi(-i32::from(remaining_depth)))
        }

        fn slow_sample(&self, node: u32, rng: &mut impl RngCore) -> f32 {
            self.slow[node as usize] + ((rng.f64() - 0.5) * 2.0 * f64::from(self.noise_amp)) as f32
        }
    }

    fn search_cfg(epsilon: f32, sigma: f64) -> SearchConfig {
        SearchConfig {
            epsilon,
            delta: 0.05,
            slow_cost: 4.0,
            node_cap: NODE_CAP,
            sigma,
            trace_nodes: false,
            lazy_discharge: false,
        }
    }

    #[test]
    fn exact_fast_oracle_stops_on_rho0_zero_without_slow_samples() {
        let mut t = VecTree::new(0.0, 0.0);
        t.push(0, NodeKind::Max, 0, 0.3, 0.3);
        t.push(0, NodeKind::Max, 0, 0.7, 0.7);
        let r = two_fidelity_search(&t, 0, &search_cfg(0.05, 0.1), &mut Xs(11));
        assert!(r.certified);
        assert_eq!(r.best_action, 2);
        assert_eq!(r.cost.slow, 0, "exact root intervals exit before sampling");
        assert_eq!(r.cost.fast, 2);
    }

    #[test]
    fn single_child_root_certifies_immediately() {
        let mut t = VecTree::new(0.0, 0.0);
        t.push(0, NodeKind::Max, 0, 0.4, 0.4);
        let r = two_fidelity_search(&t, 0, &search_cfg(0.05, 0.1), &mut Xs(12));
        assert!(r.certified);
        assert_eq!(r.best_action, 1);
    }

    #[test]
    fn ties_break_to_the_lowest_handle() {
        let mut t = VecTree::new(0.0, 0.0);
        t.push(0, NodeKind::Max, 0, 0.5, 0.5);
        t.push(0, NodeKind::Max, 0, 0.5, 0.5);
        let r = two_fidelity_search(&t, 0, &search_cfg(0.05, 0.1), &mut Xs(13));
        assert_eq!(r.best_action, 1, "equal L endpoints resolve to handle 1");
    }

    #[test]
    fn empty_intersection_guard_terminates_with_default_action() {
        // Dishonest fast oracle: node 1 declares B(2) = 0.75 (depth-2
        // children; the contract shape gives B(0) = 0, which would make the
        // liar's [10, 10] trivially certifiable before any sample) but
        // reports fast 10.0 for a true value of 0.0 — an E_δ violation the
        // guard must detect (one slow sample collapses node 1's local
        // interval to empty), then terminate with certified = false and the
        // current leader as default action (paper B.3 convention), never
        // looping.
        let mut t = VecTree::new(1.0, 0.1);
        t.push(0, NodeKind::Max, 2, 10.0, 0.0); // the liar
        t.push(0, NodeKind::Max, 2, 9.0, 9.0);
        let r = two_fidelity_search(&t, 0, &search_cfg(0.01, 0.1), &mut Xs(14));
        assert!(!r.certified);
        assert_eq!(r.best_action, 1, "default action = leader at guard time");
        assert!(r.cost.slow >= 1);
    }

    #[test]
    fn slow_only_route_terminates_certified_on_the_true_argmax() {
        // A ~1e6-scale envelope makes the fast oracle useless (depth-1
        // children: the contract shape gives B(1) = 5e5 — B(0) = 0 would
        // make depth-0 fast exact); termination must ride the local slow
        // route alone (local reversibility), interleaving the leader-L /
        // challenger-U targets until the CIs separate beyond the stop rule.
        let mut t = VecTree::new(1.0e6, 0.05);
        t.push(0, NodeKind::Max, 1, 0.0, 0.2);
        t.push(0, NodeKind::Max, 1, 0.0, 0.8);
        let r = two_fidelity_search(&t, 0, &search_cfg(0.05, 0.05), &mut Xs(15));
        assert!(r.certified);
        assert_eq!(r.best_action, 2);
        assert!(
            r.cost.slow >= 2,
            "both root sides must be sampled before certification"
        );
    }

    #[test]
    fn random_small_trees_recommend_epsilon_optimal_roots() {
        random_small_trees_pac(false);
    }

    /// Issue 916 b-lever: the same PAC smoke under the lazy-discharge
    /// resolver — intervals and the stop are shared, so the certificate
    /// guarantees must hold identically; only the routing differs.
    #[test]
    fn lazy_discharge_random_small_trees_recommend_epsilon_optimal_roots() {
        random_small_trees_pac(true);
    }

    fn random_small_trees_pac(lazy_discharge: bool) {
        // PAC smoke (the heavy suite is T2's bench): 32 deterministic
        // D=3/b=3 trees, contract-honest envelopes (leaf fast EXACT — B(0)
        // = 0; bias 0.04 at h=2 against declared 0.12·(1−2^−2) = 0.09; bias
        // 0.02 at h=1 against declared 0.06), a ≥0.3 root gap, ε = 0.1.
        // Every recommendation must be certified and ε-optimal. Seeded
        // everywhere — no global RNG, fully deterministic.
        let mut errors = 0usize;
        for seed in 0..32u64 {
            let mut xs = Xs(0x9E37_79B9_7F4A_7C15 ^ seed);
            let mut t = VecTree::new(0.12, 0.05);
            // Root children 1..=3: child 1's subtree is worth ~0.8, the
            // other two ≤ 0.4 (a gap the stop rule can certify cheaply).
            for which in 0..3u32 {
                let kind = if which % 2 == 0 {
                    NodeKind::Max
                } else {
                    NodeKind::Min
                };
                let d1 = t.push(0, kind, 1, 0.0, 0.0);
                for _ in 0..3 {
                    let d2 = t.push(d1, NodeKind::Max, 2, 0.0, 0.0);
                    for _ in 0..3 {
                        let mean = if which == 0 {
                            0.7 + 0.2 * xs.f64() as f32
                        } else {
                            0.4 * xs.f64() as f32
                        };
                        let leaf = t.push(d2, NodeKind::Max, 0, 0.0, mean);
                        // B(0) = 0 pins leaf fast values EXACT (contract).
                        t.fast[leaf as usize] = mean;
                    }
                    let sign = if d2.is_multiple_of(2) { 1.0_f32 } else { -1.0 };
                    t.fast[d2 as usize] = t.exact(d2) + 0.04 * sign;
                }
                let sign = if d1.is_multiple_of(2) { 1.0_f32 } else { -1.0 };
                t.fast[d1 as usize] = t.exact(d1) + 0.02 * sign;
            }
            let true_best = (1..=3u32).map(|c| t.exact(c)).fold(f32::MIN, f32::max);
            let cfg = SearchConfig {
                lazy_discharge,
                ..search_cfg(0.1, 0.05)
            };
            let r = two_fidelity_search(&t, 0, &cfg, &mut Xs(seed + 1_000));
            assert!(r.certified, "seed {seed}: must terminate certified");
            // End-to-end validity (Thm 3.1's certificate body): on honest
            // oracles every reported root interval contains the true value.
            for &(handle, lo, hi) in &r.root_intervals {
                let v = f64::from(t.exact(handle));
                assert!(
                    v >= lo - 1e-9 && v <= hi + 1e-9,
                    "seed {seed}: true value {v} outside certified interval [{lo}, {hi}] for node {handle}"
                );
            }
            if t.exact(r.best_action) < true_best - 0.1 {
                errors += 1;
            }
        }
        assert!(
            errors <= 2,
            "PAC smoke: {errors} errors over 32 trees (δ = 0.05)"
        );
    }

    /// The v1 scenario set under `lazy_discharge`: exact-oracle early exit,
    /// single child, tie-break, the empty-intersection guard, the slow-only
    /// route, and seed determinism.
    #[test]
    fn lazy_discharge_passes_the_v1_scenarios() {
        let ld = |eps: f32, sigma: f64| SearchConfig {
            lazy_discharge: true,
            ..search_cfg(eps, sigma)
        };
        // Exact fast oracle (B ≡ 0): ρ₀ = 0, stop fires with zero samples.
        let mut t = VecTree::new(0.0, 0.0);
        t.push(0, NodeKind::Max, 2, 0.3, 0.3);
        t.push(0, NodeKind::Max, 2, 0.7, 0.7);
        let r = two_fidelity_search(&t, 0, &ld(0.01, 0.1), &mut Xs(21));
        assert!(r.certified && r.best_action == 2 && r.cost.slow == 0);

        let mut t = VecTree::new(0.0, 0.0);
        t.push(0, NodeKind::Max, 0, 0.4, 0.4);
        let r = two_fidelity_search(&t, 0, &ld(0.05, 0.1), &mut Xs(22));
        assert!(r.certified && r.best_action == 1);

        let mut t = VecTree::new(0.0, 0.0);
        t.push(0, NodeKind::Max, 0, 0.5, 0.5);
        t.push(0, NodeKind::Max, 0, 0.5, 0.5);
        let r = two_fidelity_search(&t, 0, &ld(0.05, 0.1), &mut Xs(23));
        assert_eq!(r.best_action, 1);

        let mut t = VecTree::new(1.0, 0.1);
        t.push(0, NodeKind::Max, 2, 10.0, 0.0);
        t.push(0, NodeKind::Max, 2, 9.0, 9.0);
        let r = two_fidelity_search(&t, 0, &ld(0.01, 0.1), &mut Xs(24));
        assert!(
            !r.certified,
            "guard (or no-obligation exit) must not certify a liar"
        );

        let build = || {
            let mut t = VecTree::new(1.0e6, 0.05);
            t.push(0, NodeKind::Max, 1, 0.0, 0.2);
            t.push(0, NodeKind::Max, 1, 0.0, 0.8);
            t
        };
        let a = two_fidelity_search(&build(), 0, &ld(0.05, 0.05), &mut Xs(25));
        assert!(a.certified && a.best_action == 2 && a.cost.slow >= 2);
        let b = two_fidelity_search(&build(), 0, &ld(0.05, 0.05), &mut Xs(25));
        assert_eq!(
            (a.best_action, a.cost, a.root_intervals),
            (b.best_action, b.cost, b.root_intervals)
        );
    }

    #[test]
    fn search_is_deterministic_for_a_seed() {
        let build = || {
            let mut t = VecTree::new(1.0e6, 0.05);
            t.push(0, NodeKind::Max, 1, 0.0, 0.2);
            t.push(0, NodeKind::Max, 1, 0.0, 0.8);
            t
        };
        let a = two_fidelity_search(&build(), 0, &search_cfg(0.05, 0.05), &mut Xs(91));
        let b = two_fidelity_search(&build(), 0, &search_cfg(0.05, 0.05), &mut Xs(91));
        assert_eq!(a.best_action, b.best_action);
        assert_eq!(a.cost, b.cost);
        assert_eq!(a.certified, b.certified);
        assert_eq!(a.root_intervals, b.root_intervals);
    }

    #[test]
    fn gamma_budget_rule_and_m_samples_are_consistent() {
        // Eq. 7: Γ = 0 inside the quarter-scale cutoff, else c·m.
        assert_eq!(gamma_v(0.1, 0.8, 3.0, 7), 0.0);
        assert!((gamma_v(0.3, 0.8, 3.0, 7) - 21.0).abs() < 1e-12);
        // m: monotone in the width target, and the returned count certifies.
        let loose = m_samples(1.0, 1e-3, 0.5);
        let tight = m_samples(0.1, 1e-3, 0.5);
        assert!(loose < tight);
        assert!(loose >= 1);
        let m = m_samples(0.5, 1e-3, 0.5);
        assert!(2.0 * beta(m, 1e-3, 0.5) <= 0.5, "m must certify its target");
        // The searcher's precomputed walk must agree with the reference form
        // on a width grid — the free fn builds the same table per call, so a
        // divergence would mean the hot path and the spec have parted ways.
        let radii = stage_radii(1e-3, 0.5);
        for width in [0.9, 0.5, 0.2, 0.05, 0.01, 1e-4, 1e-8] {
            assert_eq!(
                m_samples(width, 1e-3, 0.5),
                m_samples_with(&radii, width),
                "reference and precomputed m_v disagree at width {width}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "epsilon")]
    fn rejects_negative_epsilon() {
        let t = VecTree::new(0.0, 0.0);
        let cfg = SearchConfig {
            epsilon: -0.1,
            ..search_cfg(0.05, 0.1)
        };
        let _ = two_fidelity_search(&t, 0, &cfg, &mut Xs(16));
    }

    #[test]
    #[should_panic(expected = "node_cap")]
    fn rejects_zero_node_cap() {
        let t = VecTree::new(0.0, 0.0);
        let cfg = SearchConfig {
            node_cap: 0,
            ..search_cfg(0.05, 0.1)
        };
        let _ = two_fidelity_search(&t, 0, &cfg, &mut Xs(17));
    }
}
