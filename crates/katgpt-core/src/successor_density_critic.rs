//! Successor-density goal critic — the tabular, modelless CRL extraction
//! (Issue 860 / riir-ai Research 386; arXiv:2206.07568, Eysenbach et al.).
//!
//! In countable `(s, a, g)` domains the contrastive-RL critic's Bayes
//! optimum is a **log density ratio** (paper §4.2, Ma & Collins NCE
//! consistency): `f*(s,a,g) = log( p(s_{t+} = g | s,a) / p(g) )`, where
//! `s_{t+}` is the state observed at a geometrically-drawn future step and
//! `p(g)` is the marginal goal probability. This module ships the closed-form
//! count-based estimator of exactly that ratio — no gradient descent
//! anywhere:
//!
//! ```text
//! score(s,a,g) = ln[ (N(s,a,g)+α) · (N + α·G) / ((N(s,a)+α·G) · (N(g)+α)) ]
//! ```
//!
//! with Laplace smoothing `α > 0` and `G` the number of goal cells. The
//! numerator is the smoothed discounted successor measure
//! `p(s_{t+}=g|s,a)`, the denominator the smoothed marginal `p(g)`; the
//! score is the log-ratio. Per Lemma 4.1 of the paper, `argmax_a` and
//! `argmax_g` over this score are ranking-preserved against the true
//! goal-conditioned value: the `p(g)` term is a goal-only constant that
//! cancels in every `argmax_a` (and symmetrically in `argmax_g` the
//! `(s,a)`-only terms cancel). The conditioning is the paper's own: `s_{t+}`
//! is drawn by CONTINUING along the behavior policy after the observed
//! `(s, a)` — the critic evaluates the data-collection policy's discounted
//! future, and Lemma 4.1's identity is w.r.t. that goal-averaged policy
//! (Research 386 honesty note). That cancellation is an *executable
//! property* here — see [`SuccessorDensityBuilder`] tests and the Bench 818
//! G1 gate — not a claim taken from the paper.
//!
//! # Sampling the successor measure (deterministic, zero-variance)
//!
//! The paper's §3 hindsight sampler draws `t ~ Geom(1−γ)` and relabels the
//! trajectory with goal `g = s_t`. Monte-Carlo draws of that sampler are an
//! unbiased estimator of a deterministic expectation, and a tabular count
//! table gains nothing from the variance. [`SamplerKind::Discounted`] (the
//! default) therefore accumulates the *expectation* directly: transition
//! `(s_i, a_i)` adds `(1−γ)·γ^k` to `N(s_i, a_i, s_{i+1+k})` for every
//! observed future state — via an O(L·G) reverse sweep, not the naive
//! O(L²) per-step walk. [`SamplerKind::CLearning`] applies the paper's
//! App. D classifier blend on top: the next state gets weight
//! `(1−γ)/(2−γ)`, each far-future state `j ≥ 1` steps past the next gets
//! `(1−γ)·γ^(j−1)/(2−γ)`. **The blend
//! re-weights the horizon** (relatively more far mass than the pure
//! geometric measure), so it is the paper's function-approximator training
//! correction and is shipped for parity — it is deliberately NOT the
//! G1-gated default, because exact log-ratio consistency is the property
//! G1 certifies and the blend trades that consistency away by design.
//!
//! Horizons truncate at the trajectory end: unobserved tail mass is simply
//! unassigned (the estimator is unbiased over the observed continuation).
//!
//! # What this is not
//!
//! - Not a trained embedder: the (s, a, g) ids are the caller's
//!   discretization. Continuous observations are upstream's job (riir-train
//!   Plan 413's φ-encoder lane).
//! - Not a density model: only the log-ratio is exposed, because only the
//!   ratio is what ranking needs (Lemma 4.1).
//! - Sparse-cell cold start: unseen cells sit at the α floor — the score
//!   there is the smoothed prior ratio, not a measurement. Consistency is
//!   asymptotic in visits (G1 bounds the Laplace error analytically).
//!
//! # Consumers (pull-gated, filed as their own lanes)
//!
//! riir-ai Issue 991 (per-NPC goal salience `argmax_g`, think-brain only,
//! never synced), riir-train Plan 413 Phase 1 tabular arm (`map_walk`
//! gridworld). Opt-in feature `successor_density_critic`; promotion to
//! default requires the GOAT gate (Bench 818) AND a live consumer.
//!
//! Successor-density goal critic (Issue 860 / riir-ai Research 386,
//! arXiv:2206.07568 CRL): the tabular, modelless CRL extraction — a
//! count-based closed-form estimator of the paper's log-density-ratio goal
//! critic f*(s,a,g) = log p(s_t+ = g | s,a)/p(g) over dense [S][A][S]
//! weighted-count tables, with deterministic zero-variance successor
//! samplers, argmax_a / argmax_g (goal salience), a sigmoid link (Bench 048;
//! sigmoid never softmax), and BLAKE3-committed freeze/thaw (the
//! contrastive_scope pattern). Lemma 4.1's ranking preservation is an
//! executable property here (goal-prior perturbation leaves argmax_a
//! bit-identical), not a paper claim. Consumers (riir-ai Issue 991 per-NPC
//! goal salience — think-brain only, never synced; riir-train Plan 413
//! tabular arm) file consumer-side at adoption. Opt-in POC.

use std::sync::Arc;

/// How `observe_trajectory` distributes successor mass over a transition's
/// observed future states.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerKind {
    /// Deterministic expectation of the paper's §3 geometric hindsight
    /// sampler: `(1−γ)·γ^k` mass on the state `k` steps ahead. Exactly
    /// consistent with the discounted successor measure — the default the
    /// G1 gate certifies.
    Discounted = 0,
    /// The paper's App. D C-learning blend: next state `(1−γ)/(2−γ)`, each
    /// far-future state `j ≥ 1` steps past the next gets
    /// `(1−γ)·γ^(j−1)/(2−γ)` (the far draw's own `(1−γ)·γ^(j−1)` geometric
    /// shape times the `1/(2−γ)` blend weight). As the horizon grows the
    /// per-transition mass converges to `w_next + w_far = 1` — the same
    /// normalization as [`SamplerKind::Discounted`], a different horizon
    /// profile (relatively more far mass). Requires `γ > 0`: at `γ = 0` no
    /// far future exists and the blend's far arm is undefined. NOT the
    /// G1-gated default — the blend re-weights the horizon by design, so it
    /// is the paper's function-approximator training correction, shipped
    /// for parity, and trades away the exact log-ratio consistency G1
    /// certifies.
    CLearning = 1,
}

/// Static domain + estimator configuration for the critic.
///
/// Goals share the state id space — the hindsight samplers only ever
/// assign visited states as goals, so the goal axis IS the state axis
/// (`G == S`). Callers that think in zone/item goal keys encode them into
/// the same discretization.
#[derive(Debug, Clone)]
pub struct SdcConfig {
    /// Number of discretized states (== number of goal cells).
    pub n_states: u32,
    /// Number of discrete actions.
    pub n_actions: u32,
    /// Laplace smoothing `α > 0`. Unseen cells sit at this floor.
    pub alpha: f64,
    /// Discount `γ ∈ [0, 1)`. `γ = 0` → only the immediate next state
    /// carries mass. `γ ≥ 1` is refused: the geometric measure does not
    /// normalize there.
    pub gamma: f32,
    /// Successor-mass sampler (see [`SamplerKind`]).
    pub sampler: SamplerKind,
}

impl SdcConfig {
    fn validate(&self) {
        assert!(self.n_states >= 1, "n_states must be >= 1");
        assert!(self.n_actions >= 1, "n_actions must be >= 1");
        assert!(
            self.alpha > 0.0 && self.alpha.is_finite(),
            "alpha must be finite and > 0 (alpha=0 makes unseen cells log(0) NaN)"
        );
        assert!(
            self.gamma >= 0.0 && self.gamma < 1.0,
            "gamma must lie in [0, 1) — the geometric successor measure does not normalize at gamma >= 1"
        );
        if self.sampler == SamplerKind::CLearning {
            assert!(
                self.gamma > 0.0,
                "CLearning sampler requires gamma > 0 — at gamma = 0 no far future exists and the blend's far arm is undefined"
            );
        }
    }
}

/// The core Laplace-smoothed log-count ratio, exposed as a pure function so
/// the Lemma-4.1 ranking-invariance property can be exercised against
/// perturbed marginals directly (and so consumers/benches reuse one
/// arithmetic instead of re-deriving it).
///
/// All inputs must be non-negative and finite; `alpha > 0`; `n_goals >= 1`.
/// Every denominator is `>= alpha` or `>= alpha*n_goals > 0`, so the result
/// is finite by construction.
#[inline]
#[must_use]
pub fn laplace_log_ratio(
    n_sag: f64,
    n_sa: f64,
    n_g: f64,
    n_total: f64,
    alpha: f64,
    n_goals: f64,
) -> f64 {
    // One multiply, one divide, fixed order — bit-identical rebuilds.
    let num = (n_sag + alpha) * (n_total + alpha * n_goals);
    let den = (n_sa + alpha * n_goals) * (n_g + alpha);
    (num / den).ln()
}

/// Streaming count-table builder over a discretized `(s, a, g)` domain.
///
/// Fixed-capacity by construction: the count tables are dense
/// `[S][A][S]` / `[S][A]` / `[S]` arrays sized once at construction (the
/// domain is bounded — the house rule prefers fixed tables over key maps
/// where the caller discretizes). All lookups are direct indexing; no
/// allocation after construction. Weighted increments make the counts
/// fractional (`f64`), which is what the deterministic samplers produce.
#[derive(Debug, Clone)]
pub struct SuccessorDensityBuilder {
    cfg: SdcConfig,
    /// `N(s,a,g)` flattened as `(s * n_actions + a) * n_states + g`.
    n_sag: Vec<f64>,
    /// `N(s,a)` — per-(state, action) successor mass.
    n_sa: Vec<f64>,
    /// `N(g)` — marginal goal mass.
    n_g: Vec<f64>,
    /// `N` — total successor mass (== Σ `n_g` == Σ `n_sa` up to f64 rounding;
    /// the maintained value is committed, never re-derived).
    n_total: f64,
}

impl SuccessorDensityBuilder {
    /// New builder. Panics on an invalid [`SdcConfig`].
    #[must_use]
    pub fn new(cfg: SdcConfig) -> Self {
        cfg.validate();
        let s = cfg.n_states as usize;
        let a = cfg.n_actions as usize;
        Self {
            n_sag: vec![0.0; s * a * s],
            n_sa: vec![0.0; s * a],
            n_g: vec![0.0; s],
            n_total: 0.0,
            cfg,
        }
    }

    /// The builder's configuration.
    #[must_use]
    pub fn config(&self) -> &SdcConfig {
        &self.cfg
    }

    /// Total successor mass accumulated so far.
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        self.n_total
    }

    /// `N(s,a,g)` — the raw successor count of one cell (bench / consumer
    /// readout; the score consumes this through [`Self::score`]).
    #[must_use]
    pub fn sag_mass(&self, s: u32, a: u32, g: u32) -> f64 {
        let (i, _, _) = self.indices(s, a, g);
        self.n_sag[i]
    }

    /// `N(s,a)` — per-(state, action) successor mass.
    #[must_use]
    pub fn row_mass(&self, s: u32, a: u32) -> f64 {
        let (_, i, _) = self.indices(s, a, 0);
        self.n_sa[i]
    }

    /// `N(g)` — marginal goal mass.
    #[must_use]
    pub fn goal_mass(&self, g: u32) -> f64 {
        self.n_g[g as usize]
    }

    /// Observe one transition `(s, a → next)` with no continuation. The
    /// horizon truncates immediately: the discounted sampler puts
    /// `(1−γ)` on `next`, the C-learning sampler `w_next = (1−γ)/(2−γ)`.
    pub fn observe_step(&mut self, s: u32, a: u32, next: u32) {
        let s_us = s as usize;
        let a_us = a as usize;
        let g_us = next as usize;
        debug_assert!(s_us < self.cfg.n_states as usize);
        debug_assert!(a_us < self.cfg.n_actions as usize);
        debug_assert!(g_us < self.cfg.n_states as usize);
        let w = self.next_weight();
        self.add_mass(s_us, a_us, g_us, w);
    }

    /// Observe a trajectory: `states.len() == actions.len() + 1`,
    /// transition `i` being `(states[i], actions[i] → states[i+1])`.
    ///
    /// Successor mass for all `T` transitions is laid down in ONE reverse
    /// sweep — O(L·G) total, not the naive O(L²) — by maintaining the
    /// geometrically-weighted far-future histogram while walking the
    /// trajectory backwards. Deterministic: the accumulation order is
    /// fixed, so re-observation is bit-identical.
    ///
    /// # Panics
    /// On a length mismatch or an out-of-domain id (debug builds assert;
    /// release builds clamp-guard by ignoring the call — callers are
    /// in-process, the ids come from the caller's own discretization).
    pub fn observe_trajectory(&mut self, states: &[u32], actions: &[u32]) {
        assert_eq!(
            states.len(),
            actions.len() + 1,
            "states.len() must be actions.len() + 1"
        );
        let s_count = self.cfg.n_states as usize;
        let a_count = self.cfg.n_actions as usize;
        debug_assert!(
            states
                .iter()
                .chain(actions.iter())
                .all(|&id| (id as usize) < s_count)
        );
        debug_assert!(actions.iter().all(|&id| (id as usize) < a_count));

        let gamma = self.cfg.gamma as f64;
        let (w_next, w_far) = self.sampler_weights();
        // f[g] = Σ_{j>=1} γ^{j-1} · 1[states[i+1+j] == g] — the
        // geometrically-weighted far-future histogram for transition i.
        // Built backwards; folded forward one step per iteration.
        let mut f = vec![0.0_f64; s_count];
        let mut far_scalar_sum = 0.0_f64; // Σ_g f[g], maintained alongside
        for i in (0..actions.len()).rev() {
            let s_us = states[i] as usize;
            let a_us = actions[i] as usize;
            let next_us = states[i + 1] as usize;
            let row = s_us * a_count + a_us;
            // Next-state arm.
            self.add_mass(s_us, a_us, next_us, w_next);
            // Far-future arm (weights over f). Skipped entirely when γ = 0
            // (the histogram is identically zero there).
            if w_far != 0.0 {
                let row_base = row * s_count;
                for (g, &fg) in f.iter().enumerate() {
                    let w = w_far * fg;
                    if w != 0.0 {
                        self.n_sag[row_base + g] += w;
                        self.n_g[g] += w;
                    }
                }
                self.n_sa[row] += w_far * far_scalar_sum;
                self.n_total += w_far * far_scalar_sum;
            }
            // Fold states[i+1] in for transition i-1:
            // f_{i-1}[g] = 1[states[i+1] == g] + γ·f_i[g] — EVERY entry
            // decays by γ, then the next state's indicator lands on top.
            for v in f.iter_mut() {
                *v *= gamma;
            }
            f[next_us] += 1.0;
            far_scalar_sum *= gamma;
            far_scalar_sum += 1.0;
        }
    }

    /// Smoothed log-ratio score, recomputed on demand from the maintained
    /// counts (f64 precision — the frozen table rounds to f32).
    #[must_use]
    pub fn score(&self, s: u32, a: u32, g: u32) -> f64 {
        let (i_sag, i_sa, i_g) = self.indices(s, a, g);
        laplace_log_ratio(
            self.n_sag[i_sag],
            self.n_sa[i_sa],
            self.n_g[i_g],
            self.n_total,
            self.cfg.alpha,
            self.cfg.n_states as f64,
        )
    }

    /// `argmax_a score(s, ·, g)` — ties keep the lowest action. Zero-alloc,
    /// one pass. Finite by construction (`α > 0`), so no NaN ordering
    /// guard is needed; documented rather than defended with a comparator.
    #[must_use]
    pub fn argmax_a(&self, s: u32, g: u32) -> u32 {
        let s_us = s as usize;
        let g_us = g as usize;
        let a_count = self.cfg.n_actions as usize;
        let s_count = self.cfg.n_states as usize;
        let mut best_a = 0_usize;
        let mut best = f64::NEG_INFINITY;
        for a in 0..a_count {
            let v = laplace_log_ratio(
                self.n_sag[(s_us * a_count + a) * s_count + g_us],
                self.n_sa[s_us * a_count + a],
                self.n_g[g_us],
                self.n_total,
                self.cfg.alpha,
                self.cfg.n_states as f64,
            );
            if v > best {
                best = v;
                best_a = a;
            }
        }
        best_a as u32
    }

    /// `argmax_g score(s, a, ·)` — ties keep the lowest goal. This is the
    /// goal-salience readout ("which of my goals is most reachable from
    /// this state-action pair right now").
    #[must_use]
    pub fn argmax_g(&self, s: u32, a: u32) -> u32 {
        let (i_sag, i_sa, _) = self.indices(s, a, 0);
        let s_count = self.cfg.n_states as usize;
        let mut best_g = 0_usize;
        let mut best = f64::NEG_INFINITY;
        for g in 0..s_count {
            let v = laplace_log_ratio(
                self.n_sag[i_sag + g],
                self.n_sa[i_sa],
                self.n_g[g],
                self.n_total,
                self.cfg.alpha,
                self.cfg.n_states as f64,
            );
            if v > best {
                best = v;
                best_g = g;
            }
        }
        best_g as u32
    }

    /// Freeze into the immutable, BLAKE3-committed read table. Computes
    /// every score once (f64 → f32 rounding) — the exact layout the
    /// table's O(1) lookups consume.
    #[must_use]
    pub fn finish(self) -> SuccessorDensityTable {
        let s = self.cfg.n_states as usize;
        let alpha = self.cfg.alpha;
        let n_goals = self.cfg.n_states as f64;
        let scores: Vec<f32> = self
            .n_sag
            .iter()
            .enumerate()
            .map(|(cell, &v)| {
                let row = cell / s;
                let g = cell % s;
                let n_sa_v = self.n_sa[row];
                let n_g_v = self.n_g[g];
                laplace_log_ratio(v, n_sa_v, n_g_v, self.n_total, alpha, n_goals) as f32
            })
            .collect();
        let frozen = self.freeze_bytes();
        let commitment = *blake3::hash(&frozen).as_bytes();
        SuccessorDensityTable {
            inner: Arc::new(TableInner {
                cfg: self.cfg,
                scores,
                n_total: self.n_total,
                commitment,
            }),
        }
    }

    /// Canonical builder-side freeze layout (BLAKE3-committed, all
    /// little-endian):
    /// `[n_states u32][n_actions u32][alpha f64 bits][gamma f32 bits]
    /// [sampler u8][n_total f64 bits][n_sa f64 × S·A][n_g f64 × S]
    /// [n_sag f64 × S·A·S]`.
    ///
    /// Weighted counts are fractional (`f64`), so counts are committed at
    /// full bit precision (`contrastive_scope` casts integer counts to u32;
    /// that shortcut would round the sampler weights away).
    fn freeze_bytes(&self) -> Vec<u8> {
        let s = self.cfg.n_states as usize;
        let a = self.cfg.n_actions as usize;
        // Header: 4 + 4 + 8 + 4 + 1 + 8 = 29 bytes.
        let mut bytes = Vec::with_capacity(29 + (s * a + s + s * a * s) * 8);
        bytes.extend_from_slice(&self.cfg.n_states.to_le_bytes());
        bytes.extend_from_slice(&self.cfg.n_actions.to_le_bytes());
        bytes.extend_from_slice(&self.cfg.alpha.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.cfg.gamma.to_bits().to_le_bytes());
        bytes.push(self.cfg.sampler as u8);
        bytes.extend_from_slice(&self.n_total.to_bits().to_le_bytes());
        for &v in &self.n_sa {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for &v in &self.n_g {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for &v in &self.n_sag {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        bytes
    }

    fn next_weight(&self) -> f64 {
        match self.cfg.sampler {
            SamplerKind::Discounted => 1.0 - self.cfg.gamma as f64,
            SamplerKind::CLearning => (1.0 - self.cfg.gamma as f64) / (2.0 - self.cfg.gamma as f64),
        }
    }

    /// `(w_next, w_far)` where the far arm for a state `j >= 1` steps ahead
    /// carries `w_far · γ^{j-1}` (Discounted) or `w_far · γ^{j-1}`
    /// (C-learning — the histogram `f` already carries the `γ^{j-1}`).
    fn sampler_weights(&self) -> (f64, f64) {
        let gamma = self.cfg.gamma as f64;
        match self.cfg.sampler {
            SamplerKind::Discounted => (1.0 - gamma, (1.0 - gamma) * gamma),
            SamplerKind::CLearning => {
                ((1.0 - gamma) / (2.0 - gamma), (1.0 - gamma) / (2.0 - gamma))
            }
        }
    }

    /// Add `w` mass to one `(s, a, g)` cell and every marginal it feeds.
    /// Fixed accumulation order — this is what makes re-observation
    /// bit-identical.
    fn add_mass(&mut self, s_us: usize, a_us: usize, g_us: usize, w: f64) {
        let s_count = self.cfg.n_states as usize;
        let a_count = self.cfg.n_actions as usize;
        let row = s_us * a_count + a_us;
        self.n_sag[row * s_count + g_us] += w;
        self.n_g[g_us] += w;
        self.n_sa[row] += w;
        self.n_total += w;
    }

    fn indices(&self, s: u32, a: u32, g: u32) -> (usize, usize, usize) {
        let s_us = s as usize;
        let a_us = a as usize;
        let g_us = g as usize;
        let s_count = self.cfg.n_states as usize;
        let a_count = self.cfg.n_actions as usize;
        (
            (s_us * a_count + a_us) * s_count + g_us,
            s_us * a_count + a_us,
            g_us,
        )
    }
}

/// Shared inner state of the frozen table — `Arc` so [`clone`](Clone) is
/// cheap and lock-free reads stay immutable by construction (the
/// `contrastive_scope` pattern; no `papaya` dep until a live-update
/// consumer exists).
struct TableInner {
    cfg: SdcConfig,
    /// Precomputed f32 scores, same flat `(s·A + a)·S + g` layout.
    scores: Vec<f32>,
    n_total: f64,
    commitment: [u8; 32],
}

impl std::fmt::Debug for TableInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableInner")
            .field("cfg", &self.cfg)
            .field("scores_len", &self.scores.len())
            .field("n_total", &self.n_total)
            .field("commitment", &hex_short(&self.commitment))
            .finish()
    }
}

fn hex_short(c: &[u8; 32]) -> String {
    let mut s = String::with_capacity(18);
    for b in &c[..8] {
        s.push_str(&format!("{b:02x}"));
    }
    s.push('…');
    s
}

/// Immutable, BLAKE3-committed successor-density score table.
///
/// Every lookup is a direct index into the precomputed score vector —
/// zero-alloc, branch-free apart from the argmax comparison. The
/// commitment covers the builder's raw weighted counts + parameters
/// (tamper-evidence for freeze/thaw through untrusted storage); the
/// scores themselves are the thaw-side payload, exactly the
/// `contrastive_scope` convention.
#[derive(Debug, Clone)]
pub struct SuccessorDensityTable {
    inner: Arc<TableInner>,
}

impl SuccessorDensityTable {
    /// The smoothed log-ratio score for one `(s, a, g)` cell.
    #[inline]
    #[must_use]
    pub fn score(&self, s: u32, a: u32, g: u32) -> f32 {
        let (i, _, _) = Self::indices(&self.inner.cfg, s, a, g);
        self.inner.scores[i]
    }

    /// `argmax_a score(s, ·, g)` — ties keep the lowest action.
    #[must_use]
    pub fn argmax_a(&self, s: u32, g: u32) -> u32 {
        let cfg = &self.inner.cfg;
        let s_us = s as usize;
        let g_us = g as usize;
        let a_count = cfg.n_actions as usize;
        let s_count = cfg.n_states as usize;
        let mut best_a = 0_usize;
        let mut best = f32::NEG_INFINITY;
        for a in 0..a_count {
            let v = self.inner.scores[(s_us * a_count + a) * s_count + g_us];
            if v > best {
                best = v;
                best_a = a;
            }
        }
        best_a as u32
    }

    /// `argmax_g score(s, a, ·)` — ties keep the lowest goal (goal
    /// salience).
    #[must_use]
    pub fn argmax_g(&self, s: u32, a: u32) -> u32 {
        let cfg = &self.inner.cfg;
        let (base, _, _) = Self::indices(cfg, s, a, 0);
        let s_count = cfg.n_states as usize;
        let mut best_g = 0_usize;
        let mut best = f32::NEG_INFINITY;
        for (g, &v) in self.inner.scores[base..base + s_count].iter().enumerate() {
            if v > best {
                best = v;
                best_g = g;
            }
        }
        best_g as u32
    }

    /// Sigmoid link over the score — a `(0, 1)` probability-shaped readout
    /// for consumers that want "how likely is `g` the (discounted)
    /// successor of `(s, a)` relative to the prior". Sigmoid, never
    /// softmax: the score is a per-pair log-ratio, not a competition over
    /// a shared normalizer (house rule; Bench 048's d ≈ log n margin
    /// argument). Numerically stable two-branch form.
    #[inline]
    #[must_use]
    pub fn p_successor(&self, s: u32, a: u32, g: u32) -> f32 {
        let x = self.score(s, a, g) as f64;
        // Bench-844 substrate delegation (Issue 861) — f64 compute + narrow,
        // the exact shape `exact_sigmoid_f64`'s doc names. Bit-identical to
        // the pre-substrate inline form.
        crate::exact_sigmoid_f64(x) as f32
    }

    /// The configuration the table was built under.
    #[must_use]
    pub fn config(&self) -> &SdcConfig {
        &self.inner.cfg
    }

    /// Total successor mass at build time.
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        self.inner.n_total
    }

    /// BLAKE3 commitment over the canonical builder serialization.
    #[must_use]
    pub fn commitment(&self) -> &[u8; 32] {
        &self.inner.commitment
    }

    /// The full precomputed score slice, flat `(s·A + a)·S + g` layout
    /// (consumers fusing their own scans over the table).
    #[must_use]
    pub fn scores(&self) -> &[f32] {
        &self.inner.scores
    }

    /// Freeze to canonical bytes:
    /// `[n_states u32][n_actions u32][alpha f64][gamma f32][sampler u8]
    /// [n_total f64][commitment 32B][scores f32 × S·A·S]`, little-endian.
    /// The commitment rides INSIDE the payload (carried, not recomputed —
    /// scores are not invertible to weighted counts).
    #[must_use]
    pub fn freeze(&self) -> Vec<u8> {
        let cfg = &self.inner.cfg;
        let n = self.inner.scores.len();
        // Header: 4 + 4 + 8 + 4 + 1 + 8 + 32 = 61 bytes.
        let mut bytes = Vec::with_capacity(61 + n * 4);
        bytes.extend_from_slice(&cfg.n_states.to_le_bytes());
        bytes.extend_from_slice(&cfg.n_actions.to_le_bytes());
        bytes.extend_from_slice(&cfg.alpha.to_bits().to_le_bytes());
        bytes.extend_from_slice(&cfg.gamma.to_bits().to_le_bytes());
        bytes.push(cfg.sampler as u8);
        bytes.extend_from_slice(&self.inner.n_total.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.inner.commitment);
        for &v in &self.inner.scores {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        bytes
    }

    /// Thaw from [`SuccessorDensityTable::freeze`] bytes. Verifies the
    /// length layout and the parameter consistency; the inner commitment
    /// is the builder-side count commitment (carried, not recomputed).
    ///
    /// Returns `None` on truncated / malformed input.
    #[must_use]
    pub fn thaw(bytes: &[u8]) -> Option<Self> {
        // Full header (61 bytes) must be present before any slice.
        if bytes.len() < 61 {
            return None;
        }
        let n_states = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let n_actions = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let alpha = f64::from_bits(u64::from_le_bytes(bytes[8..16].try_into().ok()?));
        let gamma = f32::from_bits(u32::from_le_bytes(bytes[16..20].try_into().ok()?));
        let sampler_byte = bytes[20];
        let sampler = match sampler_byte {
            0 => SamplerKind::Discounted,
            1 => SamplerKind::CLearning,
            _ => return None,
        };
        let n_total = f64::from_bits(u64::from_le_bytes(bytes[21..29].try_into().ok()?));
        let commitment: [u8; 32] = bytes[29..61].try_into().ok()?;
        let n = bytes.len() - 61;
        if !n.is_multiple_of(4) {
            return None;
        }
        let n_scores = n / 4;
        if n_states == 0
            || n_actions == 0
            || n_scores != n_states as usize * n_actions as usize * n_states as usize
        {
            return None;
        }
        let cfg = SdcConfig {
            n_states,
            n_actions,
            alpha,
            gamma,
            sampler,
        };
        let mut scores = Vec::with_capacity(n_scores);
        for i in 0..n_scores {
            let off = 61 + i * 4;
            scores.push(f32::from_bits(u32::from_le_bytes(
                bytes[off..off + 4].try_into().ok()?,
            )));
        }
        Some(Self {
            inner: Arc::new(TableInner {
                cfg,
                scores,
                n_total,
                commitment,
            }),
        })
    }

    fn indices(cfg: &SdcConfig, s: u32, a: u32, g: u32) -> (usize, usize, usize) {
        let s_us = s as usize;
        let a_us = a as usize;
        let g_us = g as usize;
        let s_count = cfg.n_states as usize;
        let a_count = cfg.n_actions as usize;
        (
            (s_us * a_count + a_us) * s_count + g_us,
            s_us * a_count + a_us,
            g_us,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(s: u32, a: u32, alpha: f64, gamma: f32, sampler: SamplerKind) -> SdcConfig {
        SdcConfig {
            n_states: s,
            n_actions: a,
            alpha,
            gamma,
            sampler,
        }
    }

    #[test]
    #[should_panic(expected = "alpha")]
    fn config_rejects_zero_alpha() {
        let _ = SuccessorDensityBuilder::new(cfg(4, 2, 0.0, 0.5, SamplerKind::Discounted));
    }

    #[test]
    #[should_panic(expected = "gamma")]
    fn config_rejects_gamma_at_one() {
        let _ = SuccessorDensityBuilder::new(cfg(4, 2, 1.0, 1.0, SamplerKind::Discounted));
    }

    #[test]
    #[should_panic(expected = "states.len()")]
    fn trajectory_rejects_length_mismatch() {
        let mut b = SuccessorDensityBuilder::new(cfg(4, 2, 1.0, 0.5, SamplerKind::Discounted));
        b.observe_trajectory(&[0, 1, 2], &[0]);
    }

    /// Single transition, discounted: `(1−γ)` mass on the next state only.
    /// Dyadic weights (`γ = 0.5`) → exact f64 asserts.
    #[test]
    fn single_transition_discounted_exact() {
        let mut b = SuccessorDensityBuilder::new(cfg(4, 2, 1.0, 0.5, SamplerKind::Discounted));
        b.observe_step(1, 0, 3);
        let w = 0.5_f64; // (1 − 0.5)
        let cell =
            |s: u32, a: u32, g: u32| b.n_sag[((s as usize) * 2 + a as usize) * 4 + g as usize];
        let sa = |s: usize, a: usize| s * 2 + a;
        assert_eq!(cell(1, 0, 3), w);
        assert_eq!(b.n_sa[sa(1, 0)], w);
        assert_eq!(b.n_g[3], w);
        assert_eq!(b.n_total, w);
        // Unvisited cells stay at the smoothed prior; score() must agree
        // with the pure function applied to the same counts.
        let expect = laplace_log_ratio(cell(1, 0, 3), w, w, w, 1.0, 4.0);
        assert_eq!(b.score(1, 0, 3), expect);
    }

    /// Two-transition trajectory, discounted, hand-computed.
    /// states [0,1,2], actions [0,0], γ = 0.5:
    /// transition 0 puts 0.5 on state 1 (next) and 0.25 on state 2 (far);
    /// transition 1 puts 0.5 on state 2.
    #[test]
    fn trajectory_discounted_hand_computed() {
        let mut b = SuccessorDensityBuilder::new(cfg(4, 2, 1.0, 0.5, SamplerKind::Discounted));
        b.observe_trajectory(&[0, 1, 2], &[0, 0]);
        let cell = |s: usize, a: usize, g: usize| b.n_sag[(s * 2 + a) * 4 + g];
        let sa = |s: usize, a: usize| s * 2 + a;
        assert_eq!(cell(0, 0, 1), 0.5);
        assert_eq!(cell(0, 0, 2), 0.25);
        assert_eq!(cell(1, 0, 2), 0.5);
        assert_eq!(b.n_sa[0], 0.75);
        assert_eq!(b.n_sa[sa(1, 0)], 0.5);
        assert_eq!(b.n_g[1], 0.5);
        assert_eq!(b.n_g[2], 0.75);
        assert_eq!(b.n_total, 1.25);
    }

    /// C-learning blend on the same trajectory: `w_next` = (1−γ)/(2−γ) = 1/3;
    /// the far state (j = 1, γ^0 = 1) carries the same folded weight
    /// (1−γ)/(2−γ)·γ^0 = 1/3. Non-dyadic → tolerance compare.
    #[test]
    fn trajectory_clearning_blend() {
        let mut b = SuccessorDensityBuilder::new(cfg(4, 2, 1.0, 0.5, SamplerKind::CLearning));
        b.observe_trajectory(&[0, 1, 2], &[0, 0]);
        let w_next = 1.0 / 3.0;
        let w_far = 1.0 / 3.0; // (1−γ)/(2−γ)·γ^0 = (0.5/1.5)·1
        let cell = |s: usize, a: usize, g: usize| b.n_sag[(s * 2 + a) * 4 + g];
        assert!((cell(0, 0, 1) - w_next).abs() < 1e-15);
        assert!((cell(0, 0, 2) - w_far).abs() < 1e-15);
        assert!((cell(1, 0, 2) - w_next).abs() < 1e-15);
    }

    /// THE Lemma-4.1 executable property: perturbing the goal marginal by
    /// any positive factor leaves every `argmax_a` bit-identical — the
    /// goal-only constant `(N+αG)/(N(g)+α)` cancels across the argmax.
    #[test]
    fn lemma_ranking_invariance_under_goal_prior_perturbation() {
        // Populated from a seeded deterministic walk over a 4-state ring.
        let mut b = SuccessorDensityBuilder::new(cfg(4, 2, 1.0, 0.5, SamplerKind::Discounted));
        let mut x = 0x9E3779B97F4A7C15_u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut states = Vec::with_capacity(2001);
        let mut actions = Vec::with_capacity(2000);
        states.push(0);
        let mut cur = 0_u32;
        for _ in 0..2000 {
            let act = (next() % 2) as u32;
            cur = match act {
                0 => cur,           // stay
                _ => (cur + 1) % 4, // step forward
            };
            states.push(cur);
            actions.push(act);
        }
        b.observe_trajectory(&states, &actions);

        let s_count = 4_usize;
        let a_count = 2_usize;
        let argmax_for = |goal_bias: f64| -> Vec<u32> {
            let mut out = Vec::new();
            for s in 0..s_count {
                for g in 0..s_count {
                    let mut best_a = 0_usize;
                    let mut best = f64::NEG_INFINITY;
                    for a in 0..a_count {
                        let v = laplace_log_ratio(
                            b.n_sag[(s * a_count + a) * s_count + g],
                            b.n_sa[s * a_count + a],
                            b.n_g[g] * goal_bias,
                            b.n_total,
                            1.0,
                            4.0,
                        );
                        if v > best {
                            best = v;
                            best_a = a;
                        }
                    }
                    out.push(best_a as u32);
                }
            }
            out
        };
        let baseline = argmax_for(1.0);
        for bias in [0.001_f64, 0.5, 3.7, 1.0e6] {
            assert_eq!(argmax_for(bias), baseline, "bias {bias} moved an argmax_a");
        }
        // And the builder's own argmax_a agrees with the pure-function
        // baseline (the implementation is the property, not just the algebra).
        for s in 0..4_u32 {
            for g in 0..4_u32 {
                assert_eq!(b.argmax_a(s, g), baseline[(s as usize) * 4 + g as usize]);
            }
        }
    }

    // Analytic helper REMOVED: the closed-form "stay → point mass"
    // measure was the wrong oracle (greedy-action-repeat, not the paper's
    // behavior-continued conditioning) — superseded by
    // [`ring_measure_exact`].

    /// Exact behavior-continued discounted measure for the 4-state
    /// stay/step ring, by fixed-point iteration on the Bellman recursion
    /// (γ = 0.5 → contraction 0.5; 200 sweeps converge far below f64
    /// resolution). The target is the paper's own conditioning: `s_{t+}`
    /// is drawn by CONTINUING along the behavior policy after the observed
    /// `(s, a)` — not by repeating the action greedily, which is a
    /// different (off-policy) measure and the wrong oracle.
    fn ring_measure_exact(gamma: f64) -> [[[f64; 4]; 2]; 4] {
        let mut p = [[[0.0f64; 4]; 2]; 4];
        for _ in 0..200 {
            let mut nxt = [[[0.0f64; 4]; 2]; 4];
            for (s, nxt_s) in nxt.iter_mut().enumerate() {
                for (a, nxt_sa) in nxt_s.iter_mut().enumerate() {
                    let s2 = match a {
                        0 => s,
                        _ => (s + 1) % 4,
                    };
                    for (g, slot) in nxt_sa.iter_mut().enumerate() {
                        let hit = if s2 == g { 1.0 } else { 0.0 };
                        *slot = (1.0 - gamma) * hit + gamma * 0.5 * (p[s2][0][g] + p[s2][1][g]);
                    }
                }
            }
            p = nxt;
        }
        p
    }

    /// G1 (unit-scale): empirical conditional successor mass converges to
    /// the analytic behavior-continued discounted measure, and every
    /// `argmax_a` with a decisive exact gap (≥ 0.02, ≫ sampling noise)
    /// matches the exact-measure argmax. Near-tie pairs are counted and
    /// excluded, never silently folded into the pass.
    #[test]
    fn g1_exactness_and_ranking_vs_analytic_ring() {
        let gamma = 0.5_f64;
        let mut b = SuccessorDensityBuilder::new(cfg(4, 2, 1.0, 0.5, SamplerKind::Discounted));
        let mut x = 0xDEADBEEFCAFEBABE_u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut states = Vec::with_capacity(100_001);
        let mut actions = Vec::with_capacity(100_000);
        states.push(0);
        let mut cur = 0_u32;
        for _ in 0..100_000 {
            let act = (next() % 2) as u32;
            cur = match act {
                0 => cur,
                _ => (cur + 1) % 4,
            };
            states.push(cur);
            actions.push(act);
        }
        b.observe_trajectory(&states, &actions);

        let exact = ring_measure_exact(gamma);
        let mut max_cond_err = 0.0_f64;
        for (s, exact_s) in exact.iter().enumerate() {
            for (a, exact_sa) in exact_s.iter().enumerate() {
                for (g, &exact_v) in exact_sa.iter().enumerate() {
                    let cond = b.n_sag[(s * 2 + a) * 4 + g] / b.n_sa[s * 2 + a];
                    max_cond_err = max_cond_err.max((cond - exact_v).abs());
                }
            }
        }
        assert!(
            max_cond_err <= 0.01,
            "conditional-measure error {max_cond_err} > 0.01 vs the behavior-continued exact measure"
        );

        // Ranking: argmax_a matches wherever the exact measure is decisive;
        // near-ties are counted, not hidden.
        let table = b.finish();
        let mut included = 0_usize;
        for s in 0..4_u32 {
            for g in 0..4_u32 {
                let gap =
                    (exact[s as usize][0][g as usize] - exact[s as usize][1][g as usize]).abs();
                if gap < 0.02 {
                    continue;
                }
                included += 1;
                let exact_best =
                    if exact[s as usize][0][g as usize] >= exact[s as usize][1][g as usize] {
                        0
                    } else {
                        1
                    };
                assert_eq!(
                    table.argmax_a(s, g),
                    exact_best,
                    "argmax_a({s},·,{g}) disagrees with the exact measure (gap {gap})"
                );
            }
        }
        assert!(
            included >= 8,
            "ranking gate kept only {included} decisive (s,g) pairs of 16"
        );
    }

    /// Laplace error bound: |smoothed − unsmoothed| is analytically bounded
    /// by the log of the smoothing correction factors — asserted exactly,
    /// no sampling tolerance.
    #[test]
    fn laplace_error_is_bounded_by_the_correction_factors() {
        let mut b = SuccessorDensityBuilder::new(cfg(4, 2, 0.5, 0.5, SamplerKind::Discounted));
        b.observe_trajectory(&[0, 1, 2, 0], &[1, 1, 0]);
        let (i_sag, i_sa, i_g) = b.indices(0, 1, 2);
        let smoothed = b.score(0, 1, 2);
        let unsmoothed = (b.n_sag[i_sag] * b.n_total / (b.n_sa[i_sa] * b.n_g[i_g])).ln();
        // Exact per-term correction bound:
        //   |ln((x+α)/x)| ≤ ln((x+α)/x) summed over the four smoothing sites.
        let a = b.config().alpha;
        let bound = ((b.n_sag[i_sag] + a) / b.n_sag[i_sag]).ln().abs()
            + ((b.n_total + a * 4.0) / b.n_total).ln().abs()
            + ((b.n_sa[i_sa] + a * 4.0) / b.n_sa[i_sa]).ln().abs()
            + ((b.n_g[i_g] + a) / b.n_g[i_g]).ln().abs();
        assert!(
            (smoothed - unsmoothed).abs() <= bound,
            "Laplace drift {} exceeded the analytic bound {}",
            (smoothed - unsmoothed).abs(),
            bound
        );
    }

    /// Freeze/thaw round trip: scores bit-identical, commitment equal,
    /// params equal; truncated and malformed inputs refuse.
    #[test]
    fn freeze_thaw_round_trip() {
        let mut b = SuccessorDensityBuilder::new(cfg(4, 2, 1.0, 0.5, SamplerKind::CLearning));
        b.observe_trajectory(&[0, 1, 2, 3], &[0, 1, 0]);
        let table = b.finish();
        let frozen = table.freeze();
        let thawed = SuccessorDensityTable::thaw(&frozen).expect("well-formed");
        assert_eq!(thawed.scores(), table.scores());
        assert_eq!(thawed.commitment(), table.commitment());
        assert_eq!(thawed.config().n_states, 4);
        assert_eq!(thawed.config().sampler, SamplerKind::CLearning);
        assert_eq!(thawed.total_mass(), table.total_mass());
        // argmax agreement through the round trip.
        for s in 0..4_u32 {
            for g in 0..4_u32 {
                assert_eq!(thawed.argmax_a(s, g), table.argmax_a(s, g));
            }
        }
        // Truncated → None.
        assert!(SuccessorDensityTable::thaw(&frozen[..frozen.len() - 1]).is_none());
        // Unknown sampler byte → None.
        let mut bad = frozen.clone();
        bad[20] = 9;
        assert!(SuccessorDensityTable::thaw(&bad).is_none());
    }

    /// Sigmoid link: the readout is the sigmoid OF THE SAME CELL's score
    /// (link identity), always in `(0, 1)`, and monotone in conditional
    /// mass BETWEEN CELLS OF THE SAME GOAL (the prior term cancels within
    /// one goal; across goals the score deliberately mixes in the prior,
    /// so cross-goal monotonicity is NOT a property — that is the point of
    /// the ratio).
    #[test]
    fn sigmoid_link_identity_and_same_goal_monotone() {
        // Fixture with rows that have MULTI-modal futures, so same-goal
        // conditionals actually differ (a single-future trajectory makes
        // every conditional 1.0 — degenerate, asserts nothing).
        let mut b = SuccessorDensityBuilder::new(cfg(4, 2, 1.0, 0.5, SamplerKind::Discounted));
        b.observe_trajectory(&[0, 1, 2, 3, 1], &[1, 1, 1, 0]);
        let t = b.finish();
        let mut lo = 1.0_f32;
        let mut hi = 0.0_f32;
        for s in 0..4_u32 {
            for a in 0..2_u32 {
                for g in 0..4_u32 {
                    let p = t.p_successor(s, a, g);
                    assert!(
                        p > 0.0 && p < 1.0,
                        "p_successor({s},{a},{g}) = {p} out of (0,1)"
                    );
                    // Link identity: p == sigmoid(score) within f32 rounding.
                    // The inline form is the INDEPENDENT ORACLE (Issue 861):
                    // production delegates to `crate::exact_sigmoid_f64`; this
                    // copy must stay inline or the assert becomes circular.
                    let x = t.score(s, a, g) as f64;
                    let sig = if x >= 0.0 {
                        1.0 / (1.0 + (-x).exp())
                    } else {
                        let e = x.exp();
                        e / (1.0 + e)
                    } as f32;
                    assert!(
                        (p - sig).abs() < 1e-6,
                        "link identity broken at ({s},{a},{g})"
                    );
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        assert!(lo < hi, "degenerate table: all p_successor equal");
        // Same-goal monotonicity: cond(1|0,1) = 0.6 > cond(1|1,1) ≈ 0.286
        // — both rows exist, the prior term cancels within the goal.
        assert!(
            t.p_successor(0, 1, 1) > t.p_successor(1, 1, 1),
            "same-goal monotonicity broken"
        );
    }

    /// G4 (unit-scale): the steady-state read path allocates nothing.
    /// Gated `any(debug_assertions, alloc_tracking)` per the Issue-741 rule
    /// (a measurement is a capability, not a profile); full-scale G4 lives
    /// in the Bench 818 GOAT target.
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    #[test]
    fn g4_read_path_alloc_free() {
        let mut b = SuccessorDensityBuilder::new(cfg(16, 4, 1.0, 0.5, SamplerKind::Discounted));
        let mut x = 0x0123456789ABCDEF_u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut states = Vec::with_capacity(2001);
        let mut actions = Vec::with_capacity(2000);
        states.push(0);
        let mut cur = 0_u32;
        for _ in 0..2000 {
            let act = (next() % 4) as u32;
            cur = match act {
                0 => cur,
                _ => (cur + act) % 16,
            };
            states.push(cur);
            actions.push(act);
        }
        b.observe_trajectory(&states, &actions);
        let table = b.finish();
        crate::alloc::reset_alloc_stats();
        let mut acc = 0.0_f32;
        for i in 0..10_000_u32 {
            let s = i % 16;
            let a = i % 4;
            let g = (i + 1) % 16;
            acc += table.score(s, a, g);
            acc += table.p_successor(s, a, g);
            std::hint::black_box(acc);
        }
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(count, 0, "read path must not allocate (got {count} allocs)");
    }
}
