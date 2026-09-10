//! `nonergodic` — two-level nonergodic Bayes filter (Plan 592, Research 545).
//!
//! Tracks K per-generator inner beliefs `η_n` (dim D each) plus an online
//! posterior `w_n` over WHICH generator is producing the token stream. Each
//! tick, every component is updated by its own likelihood (block numerator,
//! independent) and the components are coupled only by one scalar LSE
//! normalizer (the denominator). The telescoping readout writes `w_n·η_n`
//! into a `[K·D]` row-major block — the same slot layout `bom.rs` uses.
//!
//! Source: "The geometry of nonergodic composition" (Simplex blog,
//! simplex.pub/nonergodic-geometry/, 2026-09-09); Research 545 distills the
//! runtime primitive for this crate. Classical ancestry — Multiple
//! Hypothesis Tracking (Reid 1979) and IMM filtering (Blom & Bar-Shalom
//! 1988) — is cited, not claimed: the deltas here are latent inner beliefs,
//! the telescoping readout, the zero-alloc fixed-array runtime, and the
//! commit/revive behavior semantics.
//!
//! # Update rule (per tick, one token)
//!
//! ```text
//! for each n:  ℓ_n = likelihood_n(η_n, token)        // block numerator, independent
//!              η_n ← normalize(update_n(η_n, token)) // inner Bayes step
//! log_w[n] += ln ℓ_n
//! Z = LSE(log_w)                                     // the ONLY coupling: one scalar
//! log_w[n] −= Z ;  w[n] = exp(log_w[n])              // per-tick renormalization
//!                                                     // (numerically stable form;
//!                                                     //  unnormalized accumulation is
//!                                                     //  a documented non-goal —
//!                                                     //  underflow-prone)
//! readout:     out[n·D + j] = w[n] · η_n[j]           // the paper's (w₁η₁,…,w_Kη_K)
//! ```
//!
//! Because each `ℓ_n` is the 1-mass removed by generator n's own transition
//! operator, the product of `ℓ_n` over a sequence telescopes to exactly
//! `P(sequence | generator n)` — so `w_n` is the exact posterior
//! `μ_n·P(seq|n) / Σ_k μ_k·P(seq|k)`, verified against brute-force
//! enumeration in `tests/nonergodic_g1_exactness.rs`.
//!
//! # Zero-allocation + determinism contract
//!
//! [`NonergodicFilter::tick`] and [`NonergodicFilter::telescope_into`] use
//! only fixed-size arrays and caller-provided scratch — no `Vec`/`Box` in the
//! hot path (asserted by `tests/nonergodic_g4_alloc.rs`). The LSE runs in a
//! fixed order (max pass, then sum pass), so tick output is bit-identical
//! across runs for the same input sequence.
//!
//! # Latent vs raw boundary
//!
//! The filter state (`η_n`, `w_n`, `log_w`) is latent and local to the
//! observer — never synced. A caller may project *committed scalars* (e.g.
//! the argmax index and the `1 − max w` uncertainty from
//! [`NonergodicFilter::committed`]) across a sync boundary; the full
//! posterior and the inner beliefs stay local.
//!
//! # References
//!
//! - Plan: `katgpt-rs/.plans/592_nonergodic_belief_kernel.md`
//! - Research: `katgpt-rs/.research/545_Nonergodic_Belief_Decomposition.md`
//!

// Const-generic fixed-array code indexes `[K]`/`[D]` slots directly; iterator
// rewrites obscure the slot layout (types.rs precedent).
#![allow(clippy::needless_range_loop)]

use katgpt_types::simd::{fast_sigmoid, simd_dot_f32};

// ─────────────────────────────────────────────────────────────────────────────
// ComponentModel trait
// ─────────────────────────────────────────────────────────────────────────────

/// One ergodic component (generator) of a nonergodic composition.
///
/// A component is a latent-state token generator whose per-token transition
/// operator `T^(token)` defines: the probability mass it assigns to a token
/// given an inner state distribution, and the conditional posterior over its
/// own hidden states after seeing the token. HMM blocks (as in the source
/// blog), memoryless coins, and any model satisfying the contract below all
/// qualify.
///
/// # Contract
///
/// - `likelihood(eta, token)` returns the probability mass the component
///   assigns to `token` given the (normalized) inner state distribution
///   `eta` — i.e. the 1-norm of the *unnormalized* update `eta·T^(token)`.
///   It must be finite and non-negative; `0.0` for a token outside the
///   component's alphabet.
/// - `update_into(eta, token, out)` writes the *normalized* posterior inner
///   state distribution (the block Bayes step `η' ∝ η·T^(token)`) into
///   `out[0..dim]`. Both methods must agree: `likelihood` must equal the 1-norm
///   of the unnormalized product that `update_into` normalizes. The reference
///   impls below derive both from one set of closed-form entries so they
///   cannot drift; external impls are checked by the G1 identity tests.
/// - `likelihood`/`update_into` must not allocate and must be deterministic
///   (no hidden RNG, no thread-dependent reduction order).
/// - All impls are `Send + Sync` (the filter holds `&dyn` references across
///   caller threads).
///
/// The trait is deliberately abstract — no game/domain vocabulary (this is a
/// public MIT crate). Object-safe so a filter can hold a heterogeneous
/// `[&dyn ComponentModel; K]` (all components must share the inner-state
/// dimension D of the filter they join).
pub trait ComponentModel: Send + Sync {
    /// Probability mass this component assigns to `token` given `eta`.
    fn likelihood(&self, eta: &[f32], token: u8) -> f32;

    /// Write the normalized posterior inner state into `out[0..dim]`.
    fn update_into(&self, eta: &[f32], token: u8, out: &mut [f32]);
}

// ─────────────────────────────────────────────────────────────────────────────
// BernoulliCoin / bernoulli_pair — the blog's "two coins" reference components
// ─────────────────────────────────────────────────────────────────────────────

/// A memoryless Bernoulli coin (inner-state dimension D = 1).
///
/// Token `1` = heads (probability `p`), token `0` = tails (probability
/// `1 − p`), any other token → likelihood `0.0`. The belief over the single
/// hidden state is trivially `[1.0]` forever, which is exactly why the blog's
/// coin example reduces generator identification to counting heads and tails.
#[derive(Clone, Copy, Debug)]
pub struct BernoulliCoin {
    p: f32,
}

impl BernoulliCoin {
    /// Create a coin with `P(heads) = p`. Panics if `p ∉ (0, 1)` — a coin
    /// with `p` at either extreme makes the other token strictly impossible,
    /// which collapses the composition's posterior arithmetic for all
    /// streams containing that token.
    #[inline]
    pub fn new(p: f32) -> Self {
        assert!(p > 0.0 && p < 1.0, "BernoulliCoin: p must be in (0, 1), got {p}");
        Self { p }
    }
}

impl ComponentModel for BernoulliCoin {
    #[inline]
    fn likelihood(&self, eta: &[f32], token: u8) -> f32 {
        debug_assert_eq!(eta.len(), 1, "BernoulliCoin: eta.len() must be 1");
        match token {
            1 => self.p,
            0 => 1.0 - self.p,
            _ => 0.0,
        }
    }

    #[inline]
    fn update_into(&self, eta: &[f32], token: u8, out: &mut [f32]) {
        debug_assert_eq!(eta.len(), 1, "BernoulliCoin: eta.len() must be 1");
        debug_assert!(!out.is_empty(), "BernoulliCoin: out must have len >= 1");
        // The single-state belief is [1.0] regardless of the token; an
        // out-of-alphabet token writes 0 mass (the filter guards on the
        // likelihood, so this value is never consumed as a posterior).
        out[0] = match token {
            0 | 1 => 1.0,
            _ => 0.0,
        };
    }
}

/// Convenience: the blog's two-coin composition pair.
#[inline]
pub fn bernoulli_pair(p0: f32, p1: f32) -> [BernoulliCoin; 2] {
    [BernoulliCoin::new(p0), BernoulliCoin::new(p1)]
}

// ─────────────────────────────────────────────────────────────────────────────
// Mess3Block — the blog's 3-state reference component
// ─────────────────────────────────────────────────────────────────────────────

/// The Mess3 process (blog appendix A.2, "The Mess3 process"): three hidden
/// states, three tokens (`0`/`1`/`2` = a/b/c), parameters `α` and `x` with
/// dependent quantities `β = (1−α)/2` and `y = 1 − 2x`.
///
/// Token-labeled transition entries (`T^(t)[i][j]` = probability of moving
/// from state `i` to `j` AND emitting token `t`):
///
/// ```text
/// T^(t)[i][j] = cw(j, t) · g(i, j)
///   cw(j, t) = α if j == t else β        // token column weight
///   g(i, j)  = y if i == j  else x       // geometry factor
/// ```
///
/// Rows sum to 1 across ALL tokens (Σ_t Σ_j T^(t)[i][j] = 1), so each
/// individual `T^(t)` is sub-stochastic per row and its row mass is exactly
/// the likelihood of that token — the standard HMM belief-update convention
/// (`η' = ηT(x) / ηT(x)𝟙`).
///
/// Closed forms used by the impl (no matrix materialization):
/// - row mass: `r[i] = x + cw(i, t)·(y − x)`
/// - unnormalized update: `out[j] = cw(j, t)·(η[j]·y + (S − η[j])·x)` with
///   `S = Ση`
///
/// The blog's two composed components are `(α=0.6, x=0.15)` and
/// `(α=0.66, x=0.5)`; the boundary case `x = 0.5` (so `y = 0`, a pure-switch
/// process) is valid and accepted.
#[derive(Clone, Copy, Debug)]
pub struct Mess3Block {
    alpha: f32,
    x: f32,
    beta: f32,
    y: f32,
}

impl Mess3Block {
    /// Create a Mess3 component. Panics unless `α ∈ (0, 1)` and
    /// `x ∈ (0, 0.5]` (outside that range some transition entry goes
    /// negative or a row mass vanishes).
    #[inline]
    pub fn new(alpha: f32, x: f32) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "Mess3Block: alpha must be in (0, 1), got {alpha}");
        assert!(x > 0.0 && x <= 0.5, "Mess3Block: x must be in (0, 0.5], got {x}");
        let beta = (1.0 - alpha) / 2.0;
        let y = 1.0 - 2.0 * x;
        Self { alpha, x, beta, y }
    }

    /// One transition entry `T^(t)[i][j]` (unit-test/diagnostic access; not
    /// compiled out of tests so the non-test lib build carries no dead code).
    #[cfg(test)]
    #[inline]
    fn entry(&self, i: usize, j: usize, token: u8) -> f32 {
        let cw = self.col_weight(j, token);
        if i == j { cw * self.y } else { cw * self.x }
    }

    #[inline]
    fn col_weight(&self, j: usize, token: u8) -> f32 {
        if j as u8 == token { self.alpha } else { self.beta }
    }

    /// Closed-form row mass of `T^(token)` for state `i`.
    #[inline]
    fn row_sum(&self, i: usize, token: u8) -> f32 {
        self.x + self.col_weight(i, token) * (self.y - self.x)
    }
}

impl ComponentModel for Mess3Block {
    #[inline]
    fn likelihood(&self, eta: &[f32], token: u8) -> f32 {
        debug_assert_eq!(eta.len(), 3, "Mess3Block: eta.len() must be 3");
        if !(0..=2).contains(&token) {
            return 0.0;
        }
        eta[0] * self.row_sum(0, token)
            + eta[1] * self.row_sum(1, token)
            + eta[2] * self.row_sum(2, token)
    }

    #[inline]
    fn update_into(&self, eta: &[f32], token: u8, out: &mut [f32]) {
        debug_assert_eq!(eta.len(), 3, "Mess3Block: eta.len() must be 3");
        debug_assert!(out.len() >= 3, "Mess3Block: out must have len >= 3");
        if !(0..=2).contains(&token) {
            out[0] = 0.0;
            out[1] = 0.0;
            out[2] = 0.0;
            return;
        }
        let s = eta[0] + eta[1] + eta[2];
        for j in 0..3 {
            out[j] = self.col_weight(j, token) * (eta[j] * self.y + (s - eta[j]) * self.x);
        }
        let total = out[0] + out[1] + out[2];
        if total > 0.0 && total.is_finite() {
            let inv = 1.0 / total;
            for v in out.iter_mut().take(3) {
                *v *= inv;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SyntheticBlock — bench/test-grade D-state component (not from the paper)
// ─────────────────────────────────────────────────────────────────────────────

/// A deterministic synthetic D-state, 2-token HMM component for benchmarks
/// and alloc gates (Plan 592 G2/G4 at D ∈ {8, 32, 64}). NOT a paper model.
///
/// Structure: a fixed row-stochastic transition matrix `P` and a per-state
/// token-1 emission probability vector; token operator is
/// `T^(t)[i][j] = r_i(t)·P[i][j]` with `r_i(1) = emit_one[i]`,
/// `r_i(0) = 1 − emit_one[i]` — so `Σ_t T^(t) = P` is stochastic and each
/// `T^(t)` carries row mass `r_i(t)` (the HMM convention). Contents derive
/// from `(dim, seed)` by integer arithmetic — no RNG, bit-identical across
/// runs.
#[derive(Clone, Copy, Debug)]
pub struct SyntheticBlock {
    dim: usize,
    emit_one: [f32; 64],
    trans: [[f32; 64]; 64],
}

impl SyntheticBlock {
    /// Build a synthetic component over `dim` states (1..=64).
    #[inline]
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!((1..=64).contains(&dim), "SyntheticBlock: dim must be in 1..=64, got {dim}");
        let mut emit_one = [0.0f32; 64];
        let mut trans = [[0.0f32; 64]; 64];
        let s = seed as usize;
        for i in 0..dim {
            emit_one[i] = 0.1 + 0.8 * (((i * 37 + 11 + s * 13) % 100) as f32 / 99.0);
            let mut row_sum = 0.0f32;
            for j in 0..dim {
                let w = 1.0 + ((i + 7 * j + 3 + s * 5) % 5) as f32;
                trans[i][j] = w;
                row_sum += w;
            }
            let inv = 1.0 / row_sum;
            for j in 0..dim {
                trans[i][j] *= inv;
            }
        }
        Self { dim, emit_one, trans }
    }

    #[inline]
    fn row_mass(&self, i: usize, token: u8) -> f32 {
        match token {
            1 => self.emit_one[i],
            0 => 1.0 - self.emit_one[i],
            _ => 0.0,
        }
    }
}

impl ComponentModel for SyntheticBlock {
    /// `ℓ = Σ_i η_i·r_i(token)` — a D-length dot (SIMD via
    /// `katgpt_types::simd::simd_dot_f32`).
    #[inline]
    fn likelihood(&self, eta: &[f32], token: u8) -> f32 {
        debug_assert_eq!(eta.len(), self.dim, "SyntheticBlock: eta.len() must equal dim");
        if !(0..=1).contains(&token) {
            return 0.0;
        }
        let mut scratch = [0.0f32; 64];
        for i in 0..self.dim {
            scratch[i] = self.row_mass(i, token);
        }
        simd_dot_f32(eta, &scratch, self.dim)
    }

    /// `out[j] = Σ_i (η_i·r_i(token))·P[i][j]`, normalized. The product runs
    /// row-by-row over `P` (contiguous in `j`) so the inner loop
    /// auto-vectorizes; zero allocation.
    #[inline]
    fn update_into(&self, eta: &[f32], token: u8, out: &mut [f32]) {
        debug_assert_eq!(eta.len(), self.dim, "SyntheticBlock: eta.len() must equal dim");
        debug_assert!(out.len() >= self.dim, "SyntheticBlock: out too short");
        if !(0..=1).contains(&token) {
            for v in out.iter_mut().take(self.dim) {
                *v = 0.0;
            }
            return;
        }
        let mut t = [0.0f32; 64];
        for i in 0..self.dim {
            t[i] = eta[i] * self.row_mass(i, token);
        }
        for v in out.iter_mut().take(self.dim) {
            *v = 0.0;
        }
        for i in 0..self.dim {
            let ti = t[i];
            let row = &self.trans[i];
            for j in 0..self.dim {
                out[j] += ti * row[j];
            }
        }
        let total: f32 = out[..self.dim].iter().sum();
        if total > 0.0 && total.is_finite() {
            let inv = 1.0 / total;
            for v in out.iter_mut().take(self.dim) {
                *v *= inv;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NonergodicFilter
// ─────────────────────────────────────────────────────────────────────────────

/// Two-level nonergodic Bayes filter over K components of inner dimension D
/// (Plan 592, Research 545 §2).
///
/// State: `eta: [[f32; D]; K]` (per-component inner beliefs), `log_w: [f32;
/// K]` (posterior log-weights, renormalized per tick so `Σ exp = 1`), and the
/// derived linear copy `w` kept in sync for O(1) readout accessors.
///
/// Type parameters: `K` ≤ 16 and `D` ≤ 64 keep the whole filter on the stack
/// (largest footprint `K=16, D=64`: η 4 KiB + weights 128 B + models 256 B of
/// fat pointers). Component models are borrowed (`&'a dyn ComponentModel`) —
/// the filter never owns or allocates them.
///
/// # Construction
///
/// [`NonergodicFilter::new`] takes the model array and the prior `μ` over
/// components (normalized internally). Inner beliefs start at the uniform
/// distribution `1/D` — the blog's `η^(∅) = (⅓,⅓,⅓)` for Mess3 and the
/// trivial `[1.0]` for coins — use [`NonegodicFilter::with_initial_eta`] for
/// components whose start distribution differs.
///
/// # Panics (construction only)
///
/// `new` panics on a prior with a negative, non-finite, or all-zero mass;
/// `with_initial_eta` panics on a row with non-positive/non-finite mass. The
/// hot path never panics.
pub struct NonergodicFilter<'a, const K: usize, const D: usize> {
    models: [&'a dyn ComponentModel; K],
    eta: [[f32; D]; K],
    log_w: [f32; K],
    w: [f32; K],
}

impl<'a, const K: usize, const D: usize> NonergodicFilter<'a, K, D> {
    /// Build a filter over `models` with prior weights `prior` (normalized
    /// internally; entries may be zero — a zero-prior component can never
    /// revive, which is exact Bayes).
    pub fn new(models: [&'a dyn ComponentModel; K], prior: [f32; K]) -> Self {
        let total: f32 = prior.iter().sum();
        assert!(
            total > 0.0 && total.is_finite(),
            "nonergodic: prior must sum to a finite positive mass, got {total}"
        );
        let inv = 1.0 / total;
        let mut w = [0.0f32; K];
        let mut log_w = [0.0f32; K];
        for n in 0..K {
            assert!(
                prior[n] >= 0.0 && prior[n].is_finite(),
                "nonergodic: prior[{n}] must be finite and non-negative, got {}",
                prior[n]
            );
            w[n] = prior[n] * inv;
            log_w[n] = w[n].ln();
        }
        let mut eta = [[0.0f32; D]; K];
        for row in &mut eta {
            for v in row.iter_mut() {
                *v = 1.0 / (D as f32);
            }
        }
        Self { models, eta, log_w, w }
    }

    /// Builder: override the initial inner beliefs per component (rows are
    /// normalized internally).
    pub fn with_initial_eta(mut self, init: [[f32; D]; K]) -> Self {
        for n in 0..K {
            let s: f32 = init[n].iter().sum();
            assert!(
                s > 0.0 && s.is_finite(),
                "nonergodic: initial eta[{n}] must sum to a finite positive mass, got {s}"
            );
            let inv = 1.0 / s;
            for j in 0..D {
                self.eta[n][j] = init[n][j] * inv;
            }
        }
        self
    }

    /// Advance the filter one token.
    ///
    /// Per component: likelihood from the current inner belief, inner Bayes
    /// step (normalized), `log_w[n] += ln ℓ_n`. Then the single coupling: an
    /// LSE over `log_w`, subtracted in place, with the linear `w` refreshed.
    ///
    /// Guards (hot path, no panics):
    /// - a component whose likelihood is `0` or non-finite for the token gets
    ///   `log_w = −∞` (excluded from the posterior — exact Bayes) and its
    ///   inner belief is left untouched (the posterior inner state is
    ///   undefined when the token is impossible);
    /// - if EVERY component assigns zero mass (a token outside the whole
    ///   composition's alphabet) the LSE is `−∞` and the filter state is
    ///   left unchanged — the caller fed a token the composition cannot
    ///   explain;
    /// - a component whose update returns a degenerate (zero/non-finite-mass)
    ///   posterior keeps its previous inner belief (model-contract violation
    ///   defense; contract-abiding models never hit this).
    #[inline]
    pub fn tick(&mut self, token: u8) {
        // [f32; D] zero-init costs ~2.5 ns at D=64 (memset 256 B) — negligible
        // against the K·D update work and avoids the uninit-slice unsafe path.
        let mut next = [0.0f32; D];
        for n in 0..K {
            let l = self.models[n].likelihood(&self.eta[n], token);
            if l > 0.0 && l.is_finite() {
                self.models[n].update_into(&self.eta[n], token, &mut next);
                let total: f32 = next.iter().sum();
                if total > 0.0 && total.is_finite() {
                    // Defensive renormalization — idempotent (÷~1.0) for
                    // contract-abiding models, protects against third-party
                    // ones ("update_into + normalize" per Plan 592 T1.2).
                    let inv = 1.0 / total;
                    for v in next.iter_mut() {
                        *v *= inv;
                    }
                    self.eta[n] = next;
                }
                self.log_w[n] += l.ln();
            } else {
                self.log_w[n] = f32::NEG_INFINITY;
            }
        }
        let z = lse(&self.log_w);
        if z.is_finite() {
            for n in 0..K {
                self.log_w[n] -= z;
                self.w[n] = self.log_w[n].exp();
            }
        }
        // else: every component assigns zero probability — state unchanged.
    }

    /// Advance over a token slice (`tick` in order).
    #[inline]
    pub fn tick_many(&mut self, tokens: &[u8]) {
        for &t in tokens {
            self.tick(t);
        }
    }

    /// Component posterior weights (Σ = 1 up to f32 rounding).
    #[inline]
    pub fn weights(&self) -> &[f32; K] {
        &self.w
    }

    /// Inner belief of component `n` (normalized; sums to 1 up to rounding).
    #[inline]
    pub fn component(&self, n: usize) -> &[f32; D] {
        debug_assert!(n < K, "nonergodic: component index {n} out of range 0..{K}");
        &self.eta[n]
    }

    /// Telescoping readout: writes `w_n·η_n` into `out` as a `[K·D]`
    /// row-major block (BoM slot layout). Block `n`'s entries sum to `w[n]`
    /// (the inner beliefs are normalized). The paper's
    /// `η = (w₁η₁, …, w_Kη_K)` object, formed at readout — zero extra state.
    ///
    /// `out` must have length `K * D` (debug-asserted). A slice, not
    /// `[f32; K * D]`, because stable Rust forbids arithmetic on const
    /// generics in const positions (plan T1.3 micro-deviation).
    #[inline]
    pub fn telescope_into(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), K * D, "nonergodic: telescope_into needs len K*D");
        for n in 0..K {
            let wn = self.w[n];
            for j in 0..D {
                out[n * D + j] = wn * self.eta[n][j];
            }
        }
    }

    /// Commitment summary: `(argmax_n w_n, 1 − max_n w_n)` — the most likely
    /// component and its principled uncertainty (no hand-tuned decay
    /// constant; Research 545 §2). Ties resolve to the lowest index.
    #[inline]
    pub fn committed(&self) -> (usize, f32) {
        let mut best = 0usize;
        let mut bw = f32::NEG_INFINITY;
        for n in 0..K {
            if self.w[n] > bw {
                bw = self.w[n];
                best = n;
            }
        }
        (best, 1.0 - bw)
    }

    /// Revive margin: the gap between the top-2 posterior log-weights
    /// (`≥ 0`; `+∞` when fewer than two components hold finite mass or `K <
    /// 2`). A collapsed hypothesis revives when contradictory evidence drives
    /// this gap toward zero and through it — the filter REPORTS the margin;
    /// the consumer decides the threshold. Sigmoid gate convenience:
    /// [`NonergodicFilter::revive_gate`].
    #[inline]
    pub fn revive_margin(&self) -> f32 {
        if K < 2 {
            return f32::INFINITY;
        }
        let (first, second) = top2(&self.log_w);
        first - second
    }

    /// Sigmoid-gated view of [`NonergodicFilter::revive_margin`]:
    /// `sigmoid(−margin)` ∈ (0, 0.5] — 0 when fully committed, → 0.5 as the
    /// top-2 tie. The filter never gates internally; this is a readout for
    /// consumer-side thresholds.
    #[inline]
    pub fn revive_gate(&self) -> f32 {
        fast_sigmoid(-self.revive_margin())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Log-sum-exp over a fixed-size array (max pass, then sum pass — fixed
/// order, deterministic). Returns the input unchanged if the max is not
/// finite (all `−∞` → `−∞`; any NaN poisons the sum → NaN, which the caller
/// treats as "skip").
#[inline]
fn lse<const N: usize>(vals: &[f32; N]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for v in vals {
        if *v > m {
            m = *v;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0f32;
    for v in vals {
        s += (*v - m).exp();
    }
    m + s.ln()
}

/// Two largest values (strict comparisons — equal values both count, giving
/// a zero gap, which is the correct revive margin for tied weights).
#[inline]
fn top2<const N: usize>(vals: &[f32; N]) -> (f32, f32) {
    let mut first = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for v in vals {
        if *v > first {
            second = first;
            first = *v;
        } else if *v > second {
            second = *v;
        }
    }
    (first, second)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (model math + filter semantics; heavy enumeration lives in
// tests/nonergodic_g1_exactness.rs)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn mess3_entries_match_blog_instance() {
        let m = Mess3Block::new(0.6, 0.15);
        // Blog appendix: T(a) row 1 = [αy, βx, βx] = [0.42, 0.03, 0.03].
        assert!(close(m.entry(0, 0, 0), 0.42, 1e-6));
        assert!(close(m.entry(0, 1, 0), 0.03, 1e-6));
        assert!(close(m.entry(0, 2, 0), 0.03, 1e-6));
        // T(b) row 2 = [βx, αy, βx].
        assert!(close(m.entry(1, 0, 1), 0.03, 1e-6));
        assert!(close(m.entry(1, 1, 1), 0.42, 1e-6));
        assert!(close(m.entry(1, 2, 1), 0.03, 1e-6));
        // T(c) row 3 = [βx, βx, αy].
        assert!(close(m.entry(2, 0, 2), 0.03, 1e-6));
        assert!(close(m.entry(2, 1, 2), 0.03, 1e-6));
        assert!(close(m.entry(2, 2, 2), 0.42, 1e-6));
    }

    #[test]
    fn mess3_rows_sum_to_one_across_tokens() {
        for &(alpha, x) in &[(0.6f32, 0.15f32), (0.66, 0.5), (0.45, 0.25), (0.75, 0.05)] {
            let m = Mess3Block::new(alpha, x);
            for i in 0..3 {
                let mut total = 0.0f32;
                for t in 0u8..3 {
                    for j in 0..3 {
                        total += m.entry(i, j, t);
                    }
                }
                assert!(close(total, 1.0, EPS), "alpha={alpha} x={x} state={i}: {total}");
            }
        }
    }

    #[test]
    fn mess3_closed_forms_match_direct_products() {
        for &(alpha, x) in &[(0.6f32, 0.15f32), (0.66, 0.5), (0.45, 0.25)] {
            let m = Mess3Block::new(alpha, x);
            let eta = [0.2f32, 0.5, 0.3];
            for t in 0u8..3 {
                let mut direct_l = 0.0f32;
                for i in 0..3 {
                    let mut rs = 0.0f32;
                    for j in 0..3 {
                        rs += m.entry(i, j, t);
                    }
                    direct_l += eta[i] * rs;
                }
                assert!(close(m.likelihood(&eta, t), direct_l, 1e-6));
                let l = m.likelihood(&eta, t);
                let mut upd = [0.0f32; 3];
                m.update_into(&eta, t, &mut upd);
                for j in 0..3 {
                    let mut direct = 0.0f32;
                    for i in 0..3 {
                        direct += eta[i] * m.entry(i, j, t);
                    }
                    // update is normalized: recovered product = upd · ℓ
                    assert!(close(upd[j] * l, direct, 1e-6));
                }
            }
        }
    }

    #[test]
    fn coin_likelihoods_and_constant_belief() {
        let coin = BernoulliCoin::new(0.7);
        let eta = [1.0f32];
        assert!(close(coin.likelihood(&eta, 1), 0.7, 1e-7));
        assert!(close(coin.likelihood(&eta, 0), 0.3, 1e-7));
        assert_eq!(coin.likelihood(&eta, 5), 0.0);
        let mut out = [0.0f32; 1];
        coin.update_into(&eta, 1, &mut out);
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn filter_matches_blog_two_coin_formula() {
        let coins = bernoulli_pair(0.5, 0.7);
        let models: [&dyn ComponentModel; 2] = [&coins[0], &coins[1]];
        let mut f = NonergodicFilter::<2, 1>::new(models, [0.5, 0.5]);
        for t in [1u8, 0, 1, 1] {
            f.tick(t);
        }
        // Blog §"Two coins": w_A = μA·pA^nH·(1−pA)^nT / (… + μB·pB^nH·(1−pB)^nT)
        // for H,T,H,H (nH=3, nT=1).
        let a = 0.5 * 0.5f64.powi(3) * 0.5f64;
        let b = 0.5 * 0.7f64.powi(3) * 0.3f64;
        let wa = a / (a + b);
        assert!((f.weights()[0] as f64 - wa).abs() < 1e-5);
    }

    #[test]
    fn filter_sum_w_one_over_random_stream() {
        let mess = [Mess3Block::new(0.6, 0.15), Mess3Block::new(0.66, 0.5)];
        let models: [&dyn ComponentModel; 2] = [&mess[0], &mess[1]];
        let mut f = NonergodicFilter::<2, 3>::new(models, [0.5, 0.5]);
        let mut rng = fastrand::Rng::with_seed(592);
        for i in 0..500 {
            f.tick(rng.u8(0..3));
            if i % 50 == 0 {
                let s: f32 = f.weights().iter().sum();
                assert!((s - 1.0).abs() <= EPS, "tick {i}: Σw={s}");
            }
        }
    }

    #[test]
    fn telescope_block_sums_equal_weights() {
        let mess = [
            Mess3Block::new(0.6, 0.15),
            Mess3Block::new(0.66, 0.5),
            Mess3Block::new(0.45, 0.25),
        ];
        let models: [&dyn ComponentModel; 3] = [&mess[0], &mess[1], &mess[2]];
        let mut f = NonergodicFilter::<3, 3>::new(models, [0.5, 0.3, 0.2]);
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..20 {
            f.tick(rng.u8(0..3));
        }
        let mut tele = [0.0f32; 9];
        f.telescope_into(&mut tele);
        for n in 0..3 {
            let block: f32 = tele[n * 3..(n + 1) * 3].iter().sum();
            assert!((block - f.weights()[n]).abs() <= EPS);
        }
    }

    #[test]
    fn impossible_token_leaves_state_unchanged() {
        let mess = [Mess3Block::new(0.6, 0.15), Mess3Block::new(0.66, 0.5)];
        let models: [&dyn ComponentModel; 2] = [&mess[0], &mess[1]];
        let mut f = NonergodicFilter::<2, 3>::new(models, [0.5, 0.5]);
        f.tick(200); // outside every component's alphabet
        assert!(close(f.weights()[0], 0.5, EPS));
        assert!(close(f.weights()[1], 0.5, EPS));
        for j in 0..3 {
            assert!(close(f.component(0)[j], 1.0 / 3.0, EPS));
        }
    }

    #[test]
    #[should_panic(expected = "prior")]
    fn negative_prior_panics() {
        let coins = bernoulli_pair(0.5, 0.7);
        let models: [&dyn ComponentModel; 2] = [&coins[0], &coins[1]];
        let _ = NonergodicFilter::<2, 1>::new(models, [-0.5, 1.5]);
    }

    #[test]
    #[should_panic(expected = "prior")]
    fn zero_total_prior_panics() {
        let coins = bernoulli_pair(0.5, 0.7);
        let models: [&dyn ComponentModel; 2] = [&coins[0], &coins[1]];
        let _ = NonergodicFilter::<2, 1>::new(models, [0.0, 0.0]);
    }

    #[test]
    fn committed_and_revive_semantics() {
        let coins = bernoulli_pair(0.2, 0.9);
        let models: [&dyn ComponentModel; 2] = [&coins[0], &coins[1]];
        let mut f = NonergodicFilter::<2, 1>::new(models, [0.5, 0.5]);
        for _ in 0..40 {
            f.tick(1); // heads: coin 1 (p=0.9) explains best
        }
        let (idx, uncertainty) = f.committed();
        assert_eq!(idx, 1);
        assert!(uncertainty < 1e-3, "uncertainty={uncertainty}");
        assert!(f.revive_gate() < 1e-2, "gate={}", f.revive_gate());
        // Contradiction stream revives the collapsed coin 0. 40 heads built
        // a +60-nat commitment; each tail shifts ln(0.8/0.1) = 2.08 nats
        // toward coin 0 — 45 tails swing the net to ≈ −33 nats.
        for _ in 0..45 {
            f.tick(0);
        }
        let (idx2, unc2) = f.committed();
        assert_eq!(idx2, 0);
        assert!(unc2 < 1e-2, "unc2={unc2}");
    }

    #[test]
    fn revive_margin_single_component_is_infinite() {
        let coin = BernoulliCoin::new(0.5);
        let models: [&dyn ComponentModel; 1] = [&coin];
        let f = NonergodicFilter::<1, 1>::new(models, [1.0]);
        assert_eq!(f.revive_margin(), f32::INFINITY);
    }

    #[test]
    fn with_initial_eta_normalizes_rows() {
        let coins = bernoulli_pair(0.5, 0.7);
        let models: [&dyn ComponentModel; 2] = [&coins[0], &coins[1]];
        let f = NonergodicFilter::<2, 1>::new(models, [0.5, 0.5]).with_initial_eta([[2.0], [4.0]]);
        assert_eq!(f.component(0)[0], 1.0);
        assert_eq!(f.component(1)[0], 1.0);
    }

    #[test]
    fn synthetic_block_is_consistent() {
        let b = SyntheticBlock::new(16, 3);
        let eta = [1.0f32 / 16.0; 16];
        for t in [0u8, 1] {
            let l = b.likelihood(&eta, t);
            assert!(l > 0.0 && l.is_finite());
            let mut upd = [0.0f32; 16];
            b.update_into(&eta, t, &mut upd);
            for j in 0..16 {
                let mut direct = 0.0f32;
                for i in 0..16 {
                    let r = match t {
                        0 => 1.0 - b.emit_one[i],
                        _ => b.emit_one[i],
                    };
                    direct += eta[i] * r * b.trans[i][j];
                }
                assert!(
                    (upd[j] * l - direct).abs() <= 1e-5,
                    "token={t} j={j}: {} vs {direct}",
                    upd[j] * l
                );
            }
        }
        assert_eq!(b.likelihood(&eta, 9), 0.0);
    }
}
