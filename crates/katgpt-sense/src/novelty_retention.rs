//! Novelty-gated retention for the modality-additive belief kernel
//! (Plan 639 Phase 1, feature `novelty_retention`, DEFAULT-OFF).
//!
//! The incumbent [`evolve_belief_additive`] uses a FIXED per-kind retention
//! `α_base[m]` with an input-dependent drive. This module derives the
//! retention from the input instead (input-gated decay — the Mamba Δ /
//! arXiv:2611.21608 §3.2.2 horizontal-update mechanism; an ADOPTION, not a
//! novelty claim), fused with the shipped dual fast/slow EMA machinery:
//!
//! ```text
//! g_t[m] = clamp(α_base[m]·(1 − κ·ŝ_t[m]) + μ·n̂_t[m], g_min, g_max)
//! ŝ = r(β·s/2),  n̂ = r(β·n/2),  r(u) = u/(1+u)   (both ∈ [0, 1) for s, n ≥ 0)
//! ```
//!
//! **Saturation law (Plan 639 Phase 2 G3 verdict):** Phase 1 shipped the
//! centered logistic `2σ(β·s) − 1`; it passed G2 (36/36 cells) but cost 3.06×
//! the incumbent kernel (12 extra `fast_sigmoid`s per tick, bar ≤ 1.15×). The
//! rational map `r` has the same zero (r(0) = 0 exactly), the same slope at 0
//! (β/2) and the same range, is monotone and transcendental-free, and both
//! signals share one division per channel. Both laws' tables are in the
//! Phase 2 bench record.
//!
//! Per kind `m`, over the per-kind drive `x_t[m]` (the kernel's own sigmoid
//! drive — no new pass over the input):
//!
//! - `fast`, `slow` — per-kind EMAs (6×2 floats of state, O(1) per channel);
//! - `e = x_t − fast_{t−1}` — this tick's innovation;
//! - **sustained novelty** `s = |fast_{t−1} − slow_{t−1}|` — the band-pass
//!   divergence BEFORE this tick, so it only grows once a change has
//!   PERSISTED (a single tick is indistinguishable from a spike — persistence
//!   is the evidence; a one-tick spike leaves `s` at its steady level);
//! - **transient share** `n = max(|e| − s, 0)` — the part of the innovation
//!   the sustained divergence does NOT explain (noise / spike).
//!
//! Both signals are read before the EMAs observe `x_t`, then the EMAs update.
//!
//! **Centering (recorded deviation from the plan's literal law, Claude
//! verdict in Plan 639):** the plan wrote `σ(β·s)`, which is 0.5 at zero
//! novelty and so silently shifts the retention of a steady world to
//! `α(1 − κ/2) + μ/2`. A centered map (0 → 0) means a novelty-free channel
//! keeps EXACTLY `α_base` (clamped) — the gate only acts on evidence. The
//! κ = 0 ∧ μ = 0 arithmetic identity holds in both forms.
//!
//! **Guards:** `g_max < 1` and `g_min > 0` — a saturated gate must never zero
//! `(1 − g)·drive` and freeze a channel (brain/FSM no-stuck law). A
//! non-finite `g` falls back to `α_base[m]` (incumbent behaviour).
//!
//! **Determinism (Issue 940):** the belief crosses the sync boundary, so
//! every op here is per-channel scalar arithmetic in a fixed order — no
//! reductions, no SIMD lane sums (`sigmoid_surprise_gate` is deliberately
//! NOT reused: its body folds with `simd_dot_f32`).
//!
//! [`evolve_belief_additive`]: crate::reconstruction::ReconstructionState::evolve_belief_additive

/// The env-switch DEMOTE literal: `NOVELTY_RETENTION=0` refuses to arm the
/// gate (the kernel stays on the incumbent fixed-α code path).
pub const NOVELTY_RETENTION_ENV: &str = "NOVELTY_RETENTION";

/// `true` when the environment demotes the gate (`NOVELTY_RETENTION=0`).
/// Read once at arm time, never per tick.
#[inline]
pub fn novelty_retention_demoted() -> bool {
    std::env::var(NOVELTY_RETENTION_ENV).is_ok_and(|v| v == "0")
}

/// Fixed configuration of the retention law (no learned parameters).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoveltyRetentionConfig {
    /// Sustained-novelty pull-down κ ∈ [0, 1]: how far a really-changed
    /// world lowers retention (faster adaptation).
    pub kappa: f32,
    /// Transient push-up μ ≥ 0: how far unexplained innovation raises
    /// retention (the belief holds through noise).
    pub mu: f32,
    /// Saturation gain β on both signals (`r(β·signal/2)`).
    pub beta: f32,
    /// Lower retention bound (`> 0`).
    pub g_min: f32,
    /// Upper retention bound (`< 1` — the channel-freeze guard).
    pub g_max: f32,
    /// Fast EMA coefficient on the per-kind drive.
    pub alpha_fast: f32,
    /// Slow EMA coefficient on the per-kind drive.
    pub alpha_slow: f32,
}

impl Default for NoveltyRetentionConfig {
    fn default() -> Self {
        Self {
            kappa: 0.4,
            mu: 0.2,
            beta: 8.0,
            g_min: 0.05,
            g_max: 0.995,
            // The shipped Plan 277 F1 dual-EMA ratio (~10×).
            alpha_fast: 0.3,
            alpha_slow: 0.03,
        }
    }
}

impl NoveltyRetentionConfig {
    /// Validate the guard invariants (`0 < g_min ≤ g_max < 1`, finite
    /// coefficients, EMA rates in (0, 1]). Returns `None` when violated.
    pub fn validated(self) -> Option<Self> {
        let finite = [
            self.kappa,
            self.mu,
            self.beta,
            self.g_min,
            self.g_max,
            self.alpha_fast,
            self.alpha_slow,
        ]
        .iter()
        .all(|v| v.is_finite());
        let ok = finite
            && self.g_min > 0.0
            && self.g_min <= self.g_max
            && self.g_max < 1.0
            && (0.0..=1.0).contains(&self.kappa)
            && self.mu >= 0.0
            && self.beta >= 0.0
            && self.alpha_fast > 0.0
            && self.alpha_fast <= 1.0
            && self.alpha_slow > 0.0
            && self.alpha_slow <= 1.0;
        ok.then_some(self)
    }
}

/// Per-kind dual-EMA novelty state (6×2 floats) + the last computed gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoveltyRetention {
    config: NoveltyRetentionConfig,
    fast: [f32; 6],
    slow: [f32; 6],
    /// `false` until the first observation seeds the EMAs at the drive
    /// (a cold start must not read the first tick as a step from 0).
    seeded: bool,
    last_gate: [f32; 6],
    last_sustained: [f32; 6],
    last_transient: [f32; 6],
}

impl NoveltyRetention {
    /// Build from a config; `None` when the config breaks a guard invariant.
    pub fn new(config: NoveltyRetentionConfig) -> Option<Self> {
        let config = config.validated()?;
        Some(Self {
            config,
            fast: [0.0; 6],
            slow: [0.0; 6],
            seeded: false,
            last_gate: [f32::NAN; 6],
            last_sustained: [0.0; 6],
            last_transient: [0.0; 6],
        })
    }

    /// The configuration in force.
    #[inline]
    pub fn config(&self) -> &NoveltyRetentionConfig {
        &self.config
    }

    /// Retention gate written by the last [`gate`](Self::gate) call
    /// (NaN before the first).
    #[inline]
    pub fn last_gate(&self) -> &[f32; 6] {
        &self.last_gate
    }

    /// Sustained-novelty signal `s` per kind from the last call.
    #[inline]
    pub fn last_sustained(&self) -> &[f32; 6] {
        &self.last_sustained
    }

    /// Transient-share signal `n` per kind from the last call.
    #[inline]
    pub fn last_transient(&self) -> &[f32; 6] {
        &self.last_transient
    }

    /// Observe this tick's per-kind drive and write the per-kind retention
    /// into `out`. O(1) per channel, alloc-free, scalar-ordered.
    #[inline]
    pub fn gate(&mut self, drive: &[f32; 6], alpha_base: &[f32; 6], out: &mut [f32; 6]) {
        let c = self.config;
        if !self.seeded {
            // Seed only from finite drives; a NaN seed would poison the EMAs.
            for (m, &d) in drive.iter().enumerate() {
                let x = if d.is_finite() { d } else { 0.0 };
                self.fast[m] = x;
                self.slow[m] = x;
            }
            self.seeded = true;
        }
        let half_beta = 0.5 * c.beta;
        // Padded to 8 lanes in locals (pad lanes: x = fast = slow = α = 0,
        // finite, inert) so the body lowers to two 4-wide vectors; selects,
        // never branches. Every op is ELEMENTWISE — no reduction, so lane
        // grouping cannot change a bit (Issue 940).
        let mut x8 = [0.0f32; 8];
        let mut f8 = [0.0f32; 8];
        let mut s8 = [0.0f32; 8];
        let mut a8 = [0.0f32; 8];
        x8[..6].copy_from_slice(drive);
        f8[..6].copy_from_slice(&self.fast);
        s8[..6].copy_from_slice(&self.slow);
        a8[..6].copy_from_slice(alpha_base);
        let mut g8 = [0.0f32; 8];
        let mut sus8 = [0.0f32; 8];
        let mut tr8 = [0.0f32; 8];
        for m in 0..8 {
            let x = x8[m];
            let ok = x.is_finite();
            let fast = f8[m];
            let slow = s8[m];
            // NaN/inf input: EMAs untouched, signals 0, incumbent retention.
            let xs = if ok { x } else { fast };
            let e = xs - fast;
            let s = if ok { (fast - slow).abs() } else { 0.0 };
            let n = (e.abs() - s).max(0.0);
            f8[m] = fast + c.alpha_fast * e;
            s8[m] = if ok {
                slow + c.alpha_slow * (xs - slow)
            } else {
                slow
            };
            // Rational saturation r(u) = u/(1+u), u = β·signal/2: r(0) = 0
            // exactly, slope β/2 at 0 (= the centered logistic's), range
            // [0, 1). Both signals share ONE division:
            // ŝ = a(1+b)/((1+a)(1+b)), n̂ = b(1+a)/((1+a)(1+b)).
            let a = half_beta * s;
            let b = half_beta * n;
            let inv = 1.0 / ((1.0 + a) * (1.0 + b));
            let s_hat = a * (1.0 + b) * inv;
            let n_hat = b * (1.0 + a) * inv;
            let g = a8[m] * (1.0 - c.kappa * s_hat) + c.mu * n_hat;
            // max/min ≡ clamp on the finite branch (the only one selected).
            g8[m] = if ok && g.is_finite() {
                g.max(c.g_min).min(c.g_max)
            } else {
                a8[m]
            };
            sus8[m] = s;
            tr8[m] = n;
        }
        self.fast.copy_from_slice(&f8[..6]);
        self.slow.copy_from_slice(&s8[..6]);
        out.copy_from_slice(&g8[..6]);
        self.last_gate.copy_from_slice(&g8[..6]);
        self.last_sustained.copy_from_slice(&sus8[..6]);
        self.last_transient.copy_from_slice(&tr8[..6]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_configs_refuse() {
        let base = NoveltyRetentionConfig::default();
        for bad in [
            NoveltyRetentionConfig { g_max: 1.0, ..base },
            NoveltyRetentionConfig { g_min: 0.0, ..base },
            NoveltyRetentionConfig {
                g_min: 0.9,
                g_max: 0.8,
                ..base
            },
            NoveltyRetentionConfig {
                kappa: f32::NAN,
                ..base
            },
            NoveltyRetentionConfig {
                alpha_slow: 0.0,
                ..base
            },
        ] {
            assert!(NoveltyRetention::new(bad).is_none(), "{bad:?} must refuse");
        }
        assert!(NoveltyRetention::new(base).is_some());
    }

    #[test]
    fn steady_channel_keeps_alpha_base() {
        let mut nr = NoveltyRetention::new(NoveltyRetentionConfig::default()).unwrap();
        let mut g = [0.0; 6];
        for _ in 0..50 {
            nr.gate(&[0.3; 6], &[0.9; 6], &mut g);
        }
        for v in g {
            assert!(
                (v - 0.9).abs() < 1e-6,
                "zero novelty must keep α_base, got {v}"
            );
        }
    }
}
