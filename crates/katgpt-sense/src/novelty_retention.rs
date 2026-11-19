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
//! ŝ = 2σ(β·s) − 1,  n̂ = 2σ(β·n) − 1          (both ∈ [0, 1) for s, n ≥ 0)
//! ```
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
//! `α(1 − κ/2) + μ/2`. `2σ(β·s) − 1` maps 0 → 0, so a novelty-free channel
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

use katgpt_types::simd::fast_sigmoid;

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
    /// Sigmoid gain β on both signals.
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
        for m in 0..6 {
            let x = drive[m];
            if !x.is_finite() {
                // NaN/inf input: leave the EMAs untouched, incumbent retention.
                self.last_sustained[m] = 0.0;
                self.last_transient[m] = 0.0;
                out[m] = alpha_base[m];
                self.last_gate[m] = out[m];
                continue;
            }
            let e = x - self.fast[m];
            let s = (self.fast[m] - self.slow[m]).abs();
            let n = (e.abs() - s).max(0.0);
            self.fast[m] += c.alpha_fast * e;
            self.slow[m] += c.alpha_slow * (x - self.slow[m]);
            let s_hat = 2.0 * fast_sigmoid(c.beta * s) - 1.0;
            let n_hat = 2.0 * fast_sigmoid(c.beta * n) - 1.0;
            let g = alpha_base[m] * (1.0 - c.kappa * s_hat) + c.mu * n_hat;
            out[m] = if g.is_finite() {
                g.clamp(c.g_min, c.g_max)
            } else {
                alpha_base[m]
            };
            self.last_sustained[m] = s;
            self.last_transient[m] = n;
            self.last_gate[m] = out[m];
        }
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
