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
///
/// State is stored PADDED to 8 lanes (Issue 1231): lanes 6/7 hold x = fast =
/// slow = 0, which is inert under [`cell`] (finite, zero signals) — so the
/// per-tick body runs on whole 8-lane arrays with no 6↔8 copies and lowers to
/// two 4-wide vectors. The public accessors expose the 6 real kinds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoveltyRetention {
    config: NoveltyRetentionConfig,
    fast: [f32; 8],
    slow: [f32; 8],
    /// `false` until the first observation seeds the EMAs at the drive
    /// (a cold start must not read the first tick as a step from 0).
    seeded: bool,
    last_gate: [f32; 8],
    last_sustained: [f32; 8],
    last_transient: [f32; 8],
}

#[inline(always)]
fn six(a: &[f32; 8]) -> &[f32; 6] {
    a[..6].try_into().expect("8-lane state holds 6 kinds")
}

impl NoveltyRetention {
    /// `None` on an invalid config.
    pub fn new(config: NoveltyRetentionConfig) -> Option<Self> {
        let config = config.validated()?;
        let mut last_gate = [0.0; 8];
        last_gate[..6].copy_from_slice(&[f32::NAN; 6]);
        Some(Self {
            config,
            fast: [0.0; 8],
            slow: [0.0; 8],
            seeded: false,
            last_gate,
            last_sustained: [0.0; 8],
            last_transient: [0.0; 8],
        })
    }

    /// The validated config this gate runs.
    #[inline]
    pub fn config(&self) -> &NoveltyRetentionConfig {
        &self.config
    }

    /// The per-kind retention written by the last tick (NaN before the
    /// first observation).
    #[inline]
    pub fn last_gate(&self) -> &[f32; 6] {
        six(&self.last_gate)
    }

    /// The per-kind sustained signal `|fast − slow|` of the last tick.
    #[inline]
    pub fn last_sustained(&self) -> &[f32; 6] {
        six(&self.last_sustained)
    }

    /// The per-kind transient signal `max(0, |x − fast| − sustained)` of the
    /// last tick.
    #[inline]
    pub fn last_transient(&self) -> &[f32; 6] {
        six(&self.last_transient)
    }

    /// Seed (first tick only) from finite drives; a NaN seed would poison
    /// the EMAs. Pad lanes carry drive 0 → stay 0.
    #[inline(always)]
    fn seed(&mut self, x8: &[f32; 8]) {
        if !self.seeded {
            let seeded = x8.map(|x| if x.is_finite() { x } else { 0.0 });
            self.fast = seeded;
            self.slow = seeded;
            self.seeded = true;
        }
    }

    /// One 8-lane tick over padded drive `x8` and base retention `a8`;
    /// returns the padded gate. Every op is ELEMENTWISE (Issue 940).
    #[inline(always)]
    fn tick8(&mut self, x8: &[f32; 8], a8: &[f32; 8]) -> [f32; 8] {
        self.seed(x8);
        let c = self.config;
        let half_beta = 0.5 * c.beta;
        // Whole-array locals in, whole-array stores out: per-lane writes
        // through `&mut self` defeat vectorization (Issue 1231 measurement).
        let (f0, s0) = (self.fast, self.slow);
        let mut f8 = [0.0f32; 8];
        let mut s8 = [0.0f32; 8];
        let mut g8 = [0.0f32; 8];
        let mut sus8 = [0.0f32; 8];
        let mut tr8 = [0.0f32; 8];
        for m in 0..8 {
            let r = cell(&c, half_beta, x8[m], f0[m], s0[m], a8[m]);
            f8[m] = r.fast;
            s8[m] = r.slow;
            g8[m] = r.g;
            sus8[m] = r.sustained;
            tr8[m] = r.transient;
        }
        self.fast = f8;
        self.slow = s8;
        self.last_gate = g8;
        self.last_sustained = sus8;
        self.last_transient = tr8;
        g8
    }

    /// Observe this tick's per-kind drive and write the per-kind retention
    /// into `out`. O(1) per channel, alloc-free, scalar-ordered.
    #[inline]
    pub fn gate(&mut self, drive: &[f32; 6], alpha_base: &[f32; 6], out: &mut [f32; 6]) {
        let mut x8 = [0.0f32; 8];
        let mut a8 = [0.0f32; 8];
        x8[..6].copy_from_slice(drive);
        a8[..6].copy_from_slice(alpha_base);
        let g8 = self.tick8(&x8, &a8);
        out.copy_from_slice(&g8[..6]);
    }

    /// The FUSED armed belief kernel (Issue 1231): drive (σ over the 6
    /// activations, padded), the dual-EMA update + gate, and the belief
    /// update — all on 8-lane arrays, no drive pass → copies → gate → belief
    /// pass. The belief's 8 dims read kinds through `kind_map`
    /// (`[0..6, 0, 1]`): g and drive are shuffled into dim order once, then
    /// one 8-lane update. Same per-element ops as `gate` + the incumbent
    /// belief loop → bit-identical (pinned by
    /// `fused_kernel_is_bit_identical_to_gate`).
    #[inline]
    pub fn evolve_additive(
        &mut self,
        activations: &[f32; 6],
        eta: f32,
        alpha_base: &[f32; 6],
        kind_map: &[usize; 8],
        belief: &mut [f32; 8],
    ) {
        let mut x8 = [0.0f32; 8];
        let mut a8 = [0.0f32; 8];
        for m in 0..6 {
            x8[m] = 2.0 * katgpt_types::simd::fast_sigmoid(eta * activations[m].max(0.0)) - 1.0;
            a8[m] = alpha_base[m];
        }
        let g8 = self.tick8(&x8, &a8);
        let mut gd = [0.0f32; 8];
        let mut dd = [0.0f32; 8];
        for i in 0..8 {
            gd[i] = g8[kind_map[i]];
            dd[i] = x8[kind_map[i]];
        }
        for i in 0..8 {
            belief[i] = (gd[i] * belief[i] + (1.0 - gd[i]) * dd[i]).clamp(-1.0, 1.0);
        }
    }
}

/// One channel's update, shared by [`NoveltyRetention::gate`] (padded 8-lane
/// pass) and [`NoveltyRetention::step_kind`] (fused per-kind loop) — ONE
/// definition, so the two paths cannot drift. Every op is elementwise
/// (Issue 940); no reduction, so lane grouping or fusion cannot change a bit.
struct Cell {
    fast: f32,
    slow: f32,
    g: f32,
    sustained: f32,
    transient: f32,
}

#[inline(always)]
fn cell(
    c: &NoveltyRetentionConfig,
    half_beta: f32,
    x: f32,
    fast: f32,
    slow: f32,
    alpha_base: f32,
) -> Cell {
    let ok = x.is_finite();
    // NaN/inf input: EMAs untouched, signals 0, incumbent retention.
    let xs = if ok { x } else { fast };
    let e = xs - fast;
    let s = if ok { (fast - slow).abs() } else { 0.0 };
    let n = (e.abs() - s).max(0.0);
    let fast_next = fast + c.alpha_fast * e;
    let slow_next = if ok {
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
    let g = alpha_base * (1.0 - c.kappa * s_hat) + c.mu * n_hat;
    // max/min ≡ clamp on the finite branch (the only one selected).
    let g = if ok && g.is_finite() {
        g.max(c.g_min).min(c.g_max)
    } else {
        alpha_base
    };
    Cell {
        fast: fast_next,
        slow: slow_next,
        g,
        sustained: s,
        transient: n,
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
