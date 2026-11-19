//! Novelty-gated cache decay γ (Plan 639 Phase 3 T3.3, Issue 1232; feature
//! `novelty_gamma`, DEFAULT-OFF, per-instance armed).
//!
//! The incumbent HLA cache decays every accumulator by ONE fixed γ per tick
//! (`A_t = γ·A_{t−1} + Δ`). [`HlaNoveltyGamma`] derives that γ from the
//! stream instead, through the shared Plan 639 law
//! ([`katgpt_sense::novelty_retention::NoveltyChannel`] — consumed, never
//! re-implemented):
//!
//! - **observation** (one scalar per cache-tick):
//!   `x_t = (1/hd)·Σ_i (q_t[i] − mQ_{t−1}[i] / Z_{t−1})²` — the query's
//!   surprise against the cache's own first moment, where `Z_t = γ_t·Z_{t−1}
//!   + 1` is the decayed mass (the recursion `mQ` itself follows);
//! - a query-regime shift raises `x` PERSISTENTLY ⇒ κ pulls γ DOWN (forget
//!   the stale regime faster); a one-tick outlier raises it ONCE ⇒ μ pushes γ
//!   UP (hold through it); a steady stream keeps γ ≈ `base_gamma`.
//!
//! The gate is a separate struct the caller owns beside its cache: the cache
//! types, kernels and serde layout are untouched, so the off build — and an
//! unarmed caller passing its fixed γ — is byte-identical to the incumbent.
//!
//! **Local latent state only.** Nothing here crosses the sync boundary on its
//! own; the 5 synced scalars never pass through this code.
//!
//! **Determinism (Issue 940).** γ scales `h`/`mQ`, which feed the COMMITTED
//! `hla_moments`, so the armed path is bit-deterministic across platforms:
//! elementwise f32 only (the channel law is scalar IEEE arithmetic), and the
//! signal fold is a pinned 8-accumulator order with a fixed combine tree —
//! never a platform-lane SIMD reduction. The gate state (EMAs + mass) is NOT
//! in the cache's serde snapshot: a replay resumed from a snapshot must
//! restore the same gate state (or replay from genesis), else it diverges.
//!
//! **Verdict (Issue 1232, CLOSED NEGATIVE):** G1 ✓ (identity arm bit-exact),
//! G3 ✓ (added 7.87 ns per cache-tick at hd = 8, bmo median of 3), G4 ✓
//! (alloc-free) — but **G2 ✗, 0/36 cells**: on the pre-registered regime-
//! shift vs noisy-steady fixture the gated cells ride the fixed-γ trade-off
//! curve (best κ 0.8 μ 0.4 β 8: +0.006 below the curve, but no pinned γ
//! gives the 0.7× adaptation at ≤ 1.1× noise). Kept opt-in as a measured-
//! negative lane (`tests/novelty_gamma_goat.rs` re-runs the verdict).

use katgpt_sense::novelty_retention::{NoveltyChannel, novelty_retention_demoted};

/// The shared Plan 639 law's configuration (re-exported so a consumer needs
/// no direct katgpt-sense edge).
pub use katgpt_sense::novelty_retention::NoveltyRetentionConfig;

/// A per-cache novelty gate on the HLA decay γ. `Copy`, zero-alloc,
/// 16 bytes of state + the config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HlaNoveltyGamma {
    config: NoveltyRetentionConfig,
    channel: NoveltyChannel,
    mass: f32,
}

impl HlaNoveltyGamma {
    /// Arm a gate for a FRESH (or just-reset) cache. `None` when `config`
    /// breaks a guard ([`NoveltyRetentionConfig::validated`]) or the
    /// environment demotes the law (`NOVELTY_RETENTION=0`) — the caller then
    /// keeps its fixed γ (the incumbent path).
    pub fn arm(config: NoveltyRetentionConfig) -> Option<Self> {
        if novelty_retention_demoted() {
            return None;
        }
        Some(Self {
            config: config.validated()?,
            channel: NoveltyChannel::new(),
            mass: 0.0,
        })
    }

    /// The armed configuration.
    #[inline]
    pub fn config(&self) -> &NoveltyRetentionConfig {
        &self.config
    }

    /// The tracked decayed mass `Z` (`mQ / Z` is the cache's mean query).
    #[inline]
    pub fn mass(&self) -> f32 {
        self.mass
    }

    /// Forget the stream — call together with the cache's `reset()`.
    #[inline]
    pub fn reset(&mut self) {
        self.channel = NoveltyChannel::new();
        self.mass = 0.0;
    }

    /// The gated γ for this tick. Call BEFORE the cache update, with this
    /// tick's query `q` and the representative head's query accumulator `mq`
    /// as it stands BEFORE the update (`mQ_{t−1}`); then feed the returned γ
    /// to the kernel (`hla_state_update` / `hla_layer_update`). Equals
    /// `base_gamma` (clamped to `[g_min, g_max]`) on a steady stream; a
    /// non-finite query returns `base_gamma` and leaves the EMAs untouched.
    #[inline]
    pub fn step(&mut self, q: &[f32], mq: &[f32], base_gamma: f32) -> f32 {
        let x = query_surprise(q, mq, self.mass);
        let g = self.channel.step(&self.config, x, base_gamma);
        self.mass = g * self.mass + 1.0;
        g
    }
}

/// `(1/hd)·Σ (q[i] − mq[i]/mass)²` over `hd = q.len()` — the query's mean
/// squared residual against the cache's mean query (`mass == 0` ⇒ mean 0).
///
/// Pinned reduction order (Issue 940): lane `i % 8` accumulates element `i`
/// in ascending order, lanes combine as `((0+1)+(2+3))+((4+5)+(6+7))`. Plain
/// scalar ops, so any vectorization LLVM applies is lane-exact.
#[inline]
pub fn query_surprise(q: &[f32], mq: &[f32], mass: f32) -> f32 {
    debug_assert_eq!(q.len(), mq.len());
    let n = q.len().min(mq.len());
    if n == 0 {
        return 0.0;
    }
    let inv = if mass > 0.0 { 1.0 / mass } else { 0.0 };
    let mut acc = [0.0f32; 8];
    let (qc, qt) = q[..n].as_chunks::<8>();
    let (mc, mt) = mq[..n].as_chunks::<8>();
    for (qq, mm) in qc.iter().zip(mc) {
        for l in 0..8 {
            let d = qq[l] - mm[l] * inv;
            acc[l] += d * d;
        }
    }
    for (l, (&qv, &mv)) in qt.iter().zip(mt).enumerate() {
        let d = qv - mv * inv;
        acc[l] += d * d;
    }
    let s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    s / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surprise_matches_sequential_reference_shape() {
        let q: Vec<f32> = (0..13).map(|i| i as f32 * 0.25 - 1.0).collect();
        let mq: Vec<f32> = (0..13).map(|i| (i as f32 * 0.7).sin()).collect();
        let mass = 3.5f32;
        let seq: f64 = q
            .iter()
            .zip(&mq)
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64 / mass as f64;
                d * d
            })
            .sum::<f64>()
            / 13.0;
        assert!((query_surprise(&q, &mq, mass) as f64 - seq).abs() < 1e-5);
        assert_eq!(query_surprise(&[], &[], 1.0), 0.0);
        // mass 0 ⇒ mean query 0.
        assert_eq!(query_surprise(&[2.0], &[5.0], 0.0), 4.0);
    }

    #[test]
    fn invalid_config_refuses() {
        let bad = NoveltyRetentionConfig {
            g_max: 1.0,
            ..NoveltyRetentionConfig::default()
        };
        assert!(HlaNoveltyGamma::arm(bad).is_none());
    }

    #[test]
    fn identity_arm_returns_clamped_base_and_tracks_mass() {
        let cfg = NoveltyRetentionConfig {
            kappa: 0.0,
            mu: 0.0,
            ..NoveltyRetentionConfig::default()
        };
        let Some(mut g) = HlaNoveltyGamma::arm(cfg) else {
            return; // demoted by the environment
        };
        let mut mq = [0.0f32; 8];
        let mut z = 0.0f32;
        for t in 0..50 {
            let q = [((t * 7) % 5) as f32 - 2.0; 8];
            let gamma = g.step(&q, &mq, 0.9);
            assert_eq!(gamma, 0.9f32.max(cfg.g_min).min(cfg.g_max));
            z = gamma * z + 1.0;
            for (m, &v) in mq.iter_mut().zip(&q) {
                *m = gamma * *m + v;
            }
            assert_eq!(g.mass(), z);
        }
        g.reset();
        assert_eq!(g.mass(), 0.0);
    }
}
