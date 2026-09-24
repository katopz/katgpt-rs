//! Streaming attention-SNR accumulators (Issue 882 P1 / Research 586).
//!
//! Exact softmax **entropy** and **participation ratio** maintained inside an
//! online-softmax loop, plus a per-head measured sharpening `τ_h` fitted by
//! 1-D bisection to a target normalized entropy and handed to the shipped
//! SSMax socket ([`crate::ssmax::SsmaxMode::Fixed`]).
//!
//! # The recurrences
//!
//! Online softmax keeps a running max `m` and normalizer `l = Σ e^{x−m}`.
//! Two more registers make both statistics exact:
//!
//! ```text
//! T  = Σ e^{x−m} · (x−m)        (entropy numerator)
//! R₂ = Σ e^{2(x−m)}             (collision mass numerator)
//!
//! H  = ln l − T/l               (exact Shannon entropy, nats)
//! PR = l² / R₂ = 1 / Σ p²       (participation ratio, "effective #keys")
//! ```
//!
//! When the max moves `m → m'` (`Δ = m − m' ≤ 0`, correction `c = e^Δ`) every
//! stored term rescales in closed form — `x − m' = (x − m) + Δ`:
//!
//! ```text
//! l  ← c · l
//! T  ← c · (T + Δ · l_old)
//! R₂ ← c² · R₂
//! ```
//!
//! So the cost is two extra multiply-adds per element on top of the `exp`
//! the loop already pays, and a handful of scalar ops per max change. Nothing
//! is materialized: registers only.
//!
//! # What it measures — and what it does not (trap 2)
//!
//! Entropy is **concentration**, not relevance. A head can be confidently
//! wrong: a strong distractor gives low entropy and zero gold mass. The
//! bench pins that fixture as a measured negative. The falsifier for any
//! policy driven by these statistics is attention-to-answer mass `m_Y`
//! (Issue 882 P4), never entropy alone.
//!
//! # Measured per-head sharpening
//!
//! [`fit_tau`] solves `H(softmax(τ·x)) / ln N = h*` for `τ` by bisection in
//! log-space. The tempered entropy is monotone non-increasing in `τ > 0`, so
//! bisection is exact up to its iteration budget and never diverges. The
//! fitted `τ` plugs into the SSMax multiply socket as
//! `SsmaxMode::Fixed { s_l: τ / ln N }` ([`TauFit::ssmax_mode`]) — the
//! measured-per-head complement to SSMax's global `s_L · log N` schedule. No
//! enum variant is added: downstream exhaustive matches stay intact.
//! [`TauEma`] smooths per-head fits across steps in log-space (latent state,
//! the sanctioned mutation class — no base weights move).
//!
//! # Allocation discipline (G4)
//!
//! Every function here is allocation-free: `Copy` state, slice inputs,
//! caller-owned outputs.

use crate::ssmax::SsmaxMode;

// ──────────────────────────────────────────────────────────────────────────
// Streaming accumulator
// ──────────────────────────────────────────────────────────────────────────

/// Online-softmax state carrying the two extra SNR registers.
///
/// All quantities are in **logit units** — pass `scale · q·k`, not raw dot
/// products (the tiled kernel does this conversion itself).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftmaxSnr {
    /// Running max `m`.
    pub m: f32,
    /// Normalizer `l = Σ e^{x−m}`.
    pub l: f32,
    /// Entropy numerator `T = Σ e^{x−m}(x−m)`.
    pub t: f32,
    /// Collision numerator `R₂ = Σ e^{2(x−m)}`.
    pub r2: f32,
    /// Number of finite logits observed.
    pub n: u32,
}

impl Default for SoftmaxSnr {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftmaxSnr {
    /// Empty state (`m = −∞`, all sums zero).
    #[inline]
    pub const fn new() -> Self {
        Self {
            m: f32::NEG_INFINITY,
            l: 0.0,
            t: 0.0,
            r2: 0.0,
            n: 0,
        }
    }

    /// Rescale the stored sums for a max move `m → m_new` (`m_new ≥ m`).
    ///
    /// No-op on an empty state, where `Δ = −∞` would otherwise produce
    /// `−∞ · 0 = NaN` in the `T` update.
    #[inline]
    pub fn rebase(&mut self, m_new: f32) {
        if m_new <= self.m {
            return;
        }
        if self.l > 0.0 {
            let delta = self.m - m_new;
            let c = delta.exp();
            self.t = c * (self.t + delta * self.l);
            self.r2 *= c * c;
            self.l *= c;
        }
        self.m = m_new;
    }

    /// Fold in one already-shifted term: `y = x − m`, `e = e^y`.
    ///
    /// The kernel-side entry point: the tiled loop has `y` and `e` in hand.
    #[inline(always)]
    pub fn add_shifted_sums(&mut self, l: f32, t: f32, r2: f32, n: u32) {
        self.l += l;
        self.t += t;
        self.r2 += r2;
        self.n += n;
    }

    /// Observe a chunk of logits (masked keys may be `−∞` and are skipped).
    ///
    /// One max pass, then one fused pass (the kernel arm is the SIMD path;
    /// this is the scalar decode-row / reference form).
    pub fn observe(&mut self, logits: &[f32]) {
        let mut cm = f32::NEG_INFINITY;
        for &x in logits {
            cm = cm.max(x);
        }
        if cm == f32::NEG_INFINITY {
            return; // fully masked chunk
        }
        self.rebase(cm);
        let m = self.m;
        let (mut l, mut t, mut r2, mut n) = (0.0f32, 0.0f32, 0.0f32, 0u32);
        for &x in logits {
            let y = x - m;
            let e = y.exp();
            // −∞ key: e = 0, but y·e = NaN — select 0 instead (branch-free).
            let finite = y.is_finite();
            t += if finite { e * y } else { 0.0 };
            l += e;
            r2 += e * e;
            n += finite as u32;
        }
        self.add_shifted_sums(l, t, r2, n);
    }

    /// Merge another partial state (split-K / parallel reduction).
    pub fn merge(&mut self, other: &Self) {
        if other.l <= 0.0 {
            return;
        }
        let mut o = *other;
        let m = self.m.max(o.m);
        self.rebase(m);
        o.rebase(m);
        self.add_shifted_sums(o.l, o.t, o.r2, o.n);
    }

    /// Exact Shannon entropy in nats: `ln l − T/l`. `0` on an empty state.
    #[inline]
    pub fn entropy(&self) -> f32 {
        if self.l <= 0.0 {
            return 0.0;
        }
        // Clamp: rounding can push a near-one-hot row a hair below zero.
        (self.l.ln() - self.t / self.l).max(0.0)
    }

    /// Entropy normalized by `ln N` into `[0, 1]` (`1` = uniform).
    #[inline]
    pub fn normalized_entropy(&self) -> f32 {
        normalize_entropy(self.entropy(), self.n as usize)
    }

    /// Collision mass `Σ p² = R₂ / l²`.
    #[inline]
    pub fn collision_mass(&self) -> f32 {
        if self.l <= 0.0 {
            return 0.0;
        }
        self.r2 / (self.l * self.l)
    }

    /// Participation ratio `1 / Σ p²` — the effective number of attended keys.
    #[inline]
    pub fn participation_ratio(&self) -> f32 {
        if self.r2 <= 0.0 {
            return 0.0;
        }
        (self.l * self.l) / self.r2
    }

    /// Snapshot as a compact per-row record.
    #[inline]
    pub fn row_stats(&self) -> SnrRowStats {
        SnrRowStats {
            entropy: self.entropy(),
            participation_ratio: self.participation_ratio(),
            n: self.n,
        }
    }
}

/// `H / ln N`, with `N ≤ 1` mapping to `0` (a single key carries no choice).
#[inline]
pub fn normalize_entropy(h: f32, n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    (h / (n as f32).ln()).clamp(0.0, 1.0)
}

/// Per-query-row SNR statistics, written by the tiled kernel.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SnrRowStats {
    /// Shannon entropy of the row's attention distribution (nats).
    pub entropy: f32,
    /// Participation ratio `1/Σp²`.
    pub participation_ratio: f32,
    /// Keys attended.
    pub n: u32,
}

impl SnrRowStats {
    /// Entropy normalized by `ln N`.
    #[inline]
    pub fn normalized_entropy(&self) -> f32 {
        normalize_entropy(self.entropy, self.n as usize)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Measured per-head sharpening τ_h
// ──────────────────────────────────────────────────────────────────────────

/// Bisection bounds + budget for [`fit_tau`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TauFitConfig {
    /// Lower bound on `τ` (> 0).
    pub tau_min: f32,
    /// Upper bound on `τ`.
    pub tau_max: f32,
    /// Bisection iterations in log-space.
    pub iters: u32,
}

impl Default for TauFitConfig {
    /// `[1e-2, 1e2]`, 32 iterations — log-space width `ln(1e4) ≈ 9.2` halves
    /// to `~2e-9`, far below f32 entropy resolution.
    fn default() -> Self {
        Self {
            tau_min: 1e-2,
            tau_max: 1e2,
            iters: 32,
        }
    }
}

/// Outcome of a per-head `τ` fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TauFit {
    /// Fitted multiplier `τ` on the row's logits.
    pub tau: f32,
    /// Normalized entropy actually achieved at `tau`.
    pub achieved: f32,
    /// `true` when the target lay outside `[H(τ_max), H(τ_min)]` and `tau`
    /// is the clamped bound — the head cannot reach the target in range.
    pub saturated: bool,
    /// Keys in the fitted row.
    pub n: u32,
}

impl TauFit {
    /// The fitted `τ` expressed on the SSMax multiply socket:
    /// `s_l · ln N = τ`, so `s_l = τ / ln N`. `N ≤ 1` → identity `s_l = 1`.
    #[inline]
    pub fn ssmax_mode(&self) -> SsmaxMode {
        let log_n = if self.n > 1 {
            (self.n as f32).ln()
        } else {
            0.0
        };
        match log_n > 0.0 {
            true => SsmaxMode::Fixed {
                s_l: self.tau / log_n,
            },
            false => SsmaxMode::Fixed { s_l: 1.0 },
        }
    }
}

/// Normalized entropy of `softmax(tau · x)`, computed in one pass given the
/// row max `m` (pre-computed once per fit). Masked `−∞` keys are skipped.
#[inline]
pub fn tempered_normalized_entropy(logits: &[f32], m: f32, tau: f32) -> f32 {
    let (mut l, mut t, mut n) = (0.0f32, 0.0f32, 0u32);
    for &x in logits {
        let y = tau * (x - m);
        let e = y.exp();
        let finite = y.is_finite();
        t += if finite { e * y } else { 0.0 };
        l += e;
        n += finite as u32;
    }
    if l <= 0.0 {
        return 0.0;
    }
    normalize_entropy((l.ln() - t / l).max(0.0), n as usize)
}

/// Fit `τ` so that `H(softmax(τ·x)) / ln N ≈ target` (target in `(0, 1)`).
///
/// Monotone bisection in `ln τ`. Saturates (and says so) when the target is
/// unreachable inside `[tau_min, tau_max]` — e.g. a constant row, whose
/// entropy is `1` at every `τ`.
pub fn fit_tau(logits: &[f32], target: f32, cfg: &TauFitConfig) -> TauFit {
    let mut m = f32::NEG_INFINITY;
    let mut n = 0u32;
    for &x in logits {
        m = m.max(x);
        n += x.is_finite() as u32;
    }
    let target = target.clamp(0.0, 1.0);
    if n <= 1 || m == f32::NEG_INFINITY {
        return TauFit {
            tau: 1.0,
            achieved: 0.0,
            saturated: true,
            n,
        };
    }
    let h_at = |tau: f32| tempered_normalized_entropy(logits, m, tau);

    // Entropy falls as τ grows: H(τ_min) is the ceiling, H(τ_max) the floor.
    let h_lo_tau = h_at(cfg.tau_min);
    if target >= h_lo_tau {
        return TauFit {
            tau: cfg.tau_min,
            achieved: h_lo_tau,
            saturated: target > h_lo_tau,
            n,
        };
    }
    let h_hi_tau = h_at(cfg.tau_max);
    if target <= h_hi_tau {
        return TauFit {
            tau: cfg.tau_max,
            achieved: h_hi_tau,
            saturated: target < h_hi_tau,
            n,
        };
    }

    let (mut lo, mut hi) = (cfg.tau_min.ln(), cfg.tau_max.ln());
    for _ in 0..cfg.iters {
        let mid = 0.5 * (lo + hi);
        match h_at(mid.exp()) > target {
            true => lo = mid,  // still too flat → sharpen
            false => hi = mid, // too sharp → soften
        }
    }
    let tau = (0.5 * (lo + hi)).exp();
    TauFit {
        tau,
        achieved: h_at(tau),
        saturated: false,
        n,
    }
}

/// Log-space EMA of a per-head `τ` across steps (latent state).
///
/// Smoothing in `ln τ` keeps the average multiplicative — a head swinging
/// between `τ = 0.5` and `τ = 2` averages to `1`, not `1.25`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TauEma {
    log_tau: f32,
    alpha: f32,
    primed: bool,
}

impl TauEma {
    /// New EMA with smoothing weight `alpha ∈ (0, 1]` on the newest fit.
    #[inline]
    pub const fn new(alpha: f32) -> Self {
        Self {
            log_tau: 0.0,
            alpha,
            primed: false,
        }
    }

    /// Fold in a fit. Saturated fits are skipped — a clamped bound is not a
    /// measurement of the head.
    #[inline]
    pub fn update(&mut self, fit: &TauFit) {
        if fit.saturated || fit.tau.is_nan() || fit.tau <= 0.0 {
            return;
        }
        let x = fit.tau.ln();
        match self.primed {
            true => self.log_tau += self.alpha * (x - self.log_tau),
            false => {
                self.log_tau = x;
                self.primed = true;
            }
        }
    }

    /// Current smoothed `τ` (`1.0` before the first unsaturated fit).
    #[inline]
    pub fn tau(&self) -> f32 {
        self.log_tau.exp()
    }

    /// Whether any unsaturated fit has been folded in.
    #[inline]
    pub fn is_primed(&self) -> bool {
        self.primed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(x: &[f32]) -> (f64, f64) {
        let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z: f64 = x.iter().map(|&v| ((v as f64) - m).exp()).sum();
        let mut h = 0.0;
        let mut s2 = 0.0;
        for &v in x {
            let p = ((v as f64) - m).exp() / z;
            if p > 0.0 {
                h -= p * p.ln();
            }
            s2 += p * p;
        }
        (h, 1.0 / s2)
    }

    fn logits(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s >> 8) & 0xffff) as f32 / 65535.0 * 8.0 - 4.0
            })
            .collect()
    }

    #[test]
    fn streaming_matches_reference_under_rising_max() {
        let mut x = logits(1000, 7);
        // Ascending order forces a max move on every chunk — the rescale path.
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut s = SoftmaxSnr::new();
        for c in x.chunks(37) {
            s.observe(c);
        }
        let (h, pr) = reference(&x);
        assert!((s.entropy() as f64 - h).abs() < 1e-4 * h.max(1.0));
        assert!((s.participation_ratio() as f64 - pr).abs() < 1e-3 * pr);
        assert_eq!(s.n, 1000);
    }

    #[test]
    fn merge_equals_single_pass() {
        let x = logits(513, 11);
        let mut whole = SoftmaxSnr::new();
        whole.observe(&x);
        let (a, b) = x.split_at(200);
        let (mut sa, mut sb) = (SoftmaxSnr::new(), SoftmaxSnr::new());
        sa.observe(a);
        sb.observe(b);
        sa.merge(&sb);
        assert!((sa.entropy() - whole.entropy()).abs() < 1e-4);
        assert!((sa.participation_ratio() - whole.participation_ratio()).abs() < 1e-2);
        assert_eq!(sa.n, whole.n);
    }

    #[test]
    fn masked_keys_are_skipped_not_nan() {
        let mut s = SoftmaxSnr::new();
        s.observe(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY, 0.0]);
        assert!((s.entropy() - 2f32.ln()).abs() < 1e-6);
        assert_eq!(s.n, 2);
        assert!((s.normalized_entropy() - 1.0).abs() < 1e-6);
        let mut empty = SoftmaxSnr::new();
        empty.observe(&[f32::NEG_INFINITY; 4]);
        assert_eq!(empty.entropy(), 0.0);
        assert_eq!(empty.participation_ratio(), 0.0);
    }

    #[test]
    fn uniform_and_one_hot_limits() {
        let mut u = SoftmaxSnr::new();
        u.observe(&[1.5; 64]);
        assert!((u.normalized_entropy() - 1.0).abs() < 1e-5);
        assert!((u.participation_ratio() - 64.0).abs() < 1e-3);
        let mut h = [-100.0f32; 64];
        h[9] = 0.0;
        let mut o = SoftmaxSnr::new();
        o.observe(&h);
        assert!(o.entropy() < 1e-6);
        assert!((o.participation_ratio() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fit_tau_hits_target_and_is_monotone() {
        let x = logits(256, 3);
        let cfg = TauFitConfig::default();
        let mut prev_tau = 0.0;
        for &target in &[0.95f32, 0.8, 0.6, 0.4, 0.2] {
            let fit = fit_tau(&x, target, &cfg);
            assert!(!fit.saturated, "target {target} saturated");
            assert!((fit.achieved - target).abs() < 1e-4, "{fit:?}");
            assert!(fit.tau > prev_tau, "lower target must need a sharper τ");
            prev_tau = fit.tau;
        }
    }

    #[test]
    fn fit_tau_saturates_on_constant_row() {
        let fit = fit_tau(&[2.0; 32], 0.5, &TauFitConfig::default());
        assert!(fit.saturated);
    }

    #[test]
    fn ssmax_socket_reproduces_tau() {
        let fit = TauFit {
            tau: 3.0,
            achieved: 0.5,
            saturated: false,
            n: 1024,
        };
        let mode = fit.ssmax_mode();
        let got = mode.multiplier((1024f32).ln());
        assert!((got - 3.0).abs() < 1e-5);
    }

    #[test]
    fn tau_ema_is_log_space_and_skips_saturated() {
        let mut e = TauEma::new(0.5);
        let f = |tau, saturated| TauFit {
            tau,
            achieved: 0.0,
            saturated,
            n: 8,
        };
        e.update(&f(9.0, true));
        assert!(!e.is_primed());
        e.update(&f(0.5, false));
        e.update(&f(2.0, false));
        assert!((e.tau() - 1.0).abs() < 1e-6);
    }
}
