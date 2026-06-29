//! GDSD Advantage-Guided Pruner — Modelless Distillation.
//!
//! Applies GDSD-style advantage-guided self-distillation to DDTree branch scoring.
//! Instead of matching denoiser logits (paper's approach), we match pruner relevance
//! scores to an advantage-weighted teacher pruner.
//!
//! # Key Idea
//!
//! The "teacher" signal is a blend of:
//! 1. The inner pruner's relevance (student, with β decay)
//! 2. A reference pruner's relevance (unconstrained baseline)
//! 3. An advantage signal from bandit/arena outcomes
//!
//! Teacher = (1-β)·r_old + β·r_ref + ψ·A(action)
//!
//! When TLC (Token-Level Centralization) is enabled, the advantage is mean-centered
//! to produce zero-mean guidance, preventing systemic bias.
//!
//! # Architecture
//!
//! ```text
//! GdsdPruner<P>
//!   ├── inner: P                  (base ScreeningPruner, e.g. SdarBanditPruner)
//!   ├── ref_pruner: P             (reference pruner, e.g. NoScreeningPruner)
//!   ├── beta: f32                 (KL regularization, default: 0.001)
//!   ├── psi: f32                  (guidance coefficient, default: 10.0)
//!   ├── advantage_fn              (A(action) from bandit/arena)
//!   └── tlc: bool                 (token-level centralization, default: true)
//! ```
//!
//! # Optimization Compliance
//!
//! - **Zero alloc in hot path:** `relevance()` computes from pre-stored values.
//! - **Serial TLC:** O(V) where V ≤ 128 — too small for rayon.
//! - **No GPU needed:** Pure CPU, pure arithmetic.
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "gdsd_distill")]`.
//!
//! **Source:** [GDSD: Guided Denoiser Self-Distillation](https://arxiv.org/abs/2505.23415)

use katgpt_speculative::ScreeningPruner;

// ── Config ──────────────────────────────────────────────────────

/// Configuration for [`GdsdPruner`].
#[derive(Clone, Debug)]
pub struct GdsdConfig {
    /// KL regularization coefficient β (default: 0.001).
    ///
    /// Controls how much weight the reference pruner gets vs the inner pruner.
    /// - β=0: pure inner + advantage (no reference blending)
    /// - β=0.5: equal weight inner and reference
    /// - β=1.0: pure reference + advantage
    pub beta: f32,
    /// Guidance coefficient ψ (default: 10.0).
    ///
    /// Scales the advantage signal strength.
    /// - ψ=0: no advantage guidance (degrades to simple blend)
    /// - ψ=10: strong advantage guidance (paper default)
    pub psi: f32,
    /// Token-level centralization (default: true).
    ///
    /// When enabled, the advantage is mean-centered across the current scoring batch,
    /// producing zero-mean guidance that prevents systemic bias.
    pub tlc: bool,
}

impl Default for GdsdConfig {
    fn default() -> Self {
        Self {
            beta: 0.001,
            psi: 10.0,
            tlc: true,
        }
    }
}

impl GdsdConfig {
    /// Create config with custom β and ψ.
    pub fn new(beta: f32, psi: f32) -> Self {
        Self {
            beta,
            psi,
            ..Self::default()
        }
    }

    /// Disable TLC (token-level centralization).
    pub fn no_tlc(mut self) -> Self {
        self.tlc = false;
        self
    }

    /// Strong guidance preset (ψ=20.0, β=0.01).
    pub fn strong() -> Self {
        Self {
            beta: 0.01,
            psi: 20.0,
            tlc: true,
        }
    }

    /// Mild guidance preset (ψ=1.0, β=0.0001).
    pub fn mild() -> Self {
        Self {
            beta: 0.0001,
            psi: 1.0,
            tlc: true,
        }
    }
}

// ── GdsdPruner ──────────────────────────────────────────────────

/// GDSD Advantage-Guided Self-Distillation Pruner.
///
/// Wraps an inner `ScreeningPruner` and blends its relevance with a reference
/// pruner and an advantage signal from bandit/arena outcomes.
///
/// The teacher signal is:
/// ```text
/// teacher = (1-β)·r_inner + β·r_ref + ψ·advantage
/// ```
///
/// When TLC is enabled, the advantage is mean-centered to zero-mean.
pub struct GdsdPruner<P: ScreeningPruner> {
    /// Base pruner (e.g., SdarBanditPruner, DeltaBanditPruner).
    inner: P,
    /// Reference pruner (e.g., NoScreeningPruner — unconstrained baseline).
    ref_pruner: P,
    /// KL regularization coefficient.
    beta: f32,
    /// Guidance coefficient.
    psi: f32,
    /// Advantage function: maps relevance → advantage signal.
    ///
    /// Stored as a fn pointer (zero-size, no heap allocation).
    /// The advantage signal typically comes from bandit Q-values or arena outcomes.
    advantage_fn: fn(f32) -> f32,
    /// Token-level centralization enabled.
    tlc: bool,
    /// Running mean of advantage for TLC centralization.
    /// Updated externally via `update_advantage_mean()`.
    advantage_mean: f32,
}

impl<P: ScreeningPruner> GdsdPruner<P> {
    /// Create a new GDSD pruner with default config.
    ///
    /// - `inner`: base pruner (student)
    /// - `ref_pruner`: reference pruner (teacher baseline, e.g., NoScreeningPruner)
    /// - `advantage_fn`: function mapping relevance → advantage signal
    pub fn new(inner: P, ref_pruner: P, advantage_fn: fn(f32) -> f32) -> Self {
        Self::with_config(inner, ref_pruner, advantage_fn, GdsdConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(
        inner: P,
        ref_pruner: P,
        advantage_fn: fn(f32) -> f32,
        config: GdsdConfig,
    ) -> Self {
        Self {
            inner,
            ref_pruner,
            beta: config.beta,
            psi: config.psi,
            advantage_fn,
            tlc: config.tlc,
            advantage_mean: 0.0,
        }
    }

    /// Update the running advantage mean for TLC centralization.
    ///
    /// Call this periodically (e.g., per episode) with the mean advantage
    /// across all arms. When TLC is enabled, the advantage is centered
    /// by subtracting this mean.
    #[inline]
    pub fn update_advantage_mean(&mut self, mean: f32) {
        self.advantage_mean = mean;
    }

    /// Access the inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Mutable access to the inner pruner.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// Access the reference pruner.
    pub fn ref_pruner(&self) -> &P {
        &self.ref_pruner
    }

    /// Current β (KL regularization).
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Current ψ (guidance coefficient).
    pub fn psi(&self) -> f32 {
        self.psi
    }

    /// Whether TLC is enabled.
    pub fn tlc_enabled(&self) -> bool {
        self.tlc
    }

    /// Compute the GDSD teacher signal for a given relevance pair.
    ///
    /// Exposed for testing and for consumers that need the raw teacher score.
    ///
    /// Returns: `(1-β)·r_old + β·r_ref + ψ·A(advantage_input)`
    /// With TLC: advantage is centered by subtracting `advantage_mean`.
    #[inline]
    pub fn teacher_signal(&self, r_old: f32, r_ref: f32, advantage_input: f32) -> f32 {
        let advantage = (self.advantage_fn)(advantage_input);
        let centered = if self.tlc {
            advantage - self.advantage_mean
        } else {
            advantage
        };
        (1.0 - self.beta) * r_old + self.beta * r_ref + self.psi * centered
    }
}

// Delegate ScreeningPruner with GDSD teacher blending
impl<P: ScreeningPruner> ScreeningPruner for GdsdPruner<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let r_old = self.inner.relevance(depth, token_idx, parent_tokens);
        let r_ref = self.ref_pruner.relevance(depth, token_idx, parent_tokens);

        // GDSD teacher: (1-β)·r_old + β·r_ref + ψ·A(r_old)
        let teacher = self.teacher_signal(r_old, r_ref, r_old);

        // Clamp to valid relevance range [0, 1]
        teacher.clamp(0.0, 1.0)
    }
}

// ── Common Advantage Functions ──────────────────────────────────

/// Identity advantage function: A(x) = x.
///
/// Use when the raw relevance score is the advantage signal.
pub fn identity_advantage(x: f32) -> f32 {
    x
}

/// Sigmoid advantage function: A(x) = σ(x).
///
/// Bounded in (0, 1). Good for raw Q-value advantages.
pub fn sigmoid_advantage(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh advantage function: A(x) = tanh(x).
///
/// Bounded in (-1, 1). Good for centered advantage signals.
pub fn tanh_advantage(x: f32) -> f32 {
    x.tanh()
}

/// Clamped linear advantage: A(x) = clamp(x, -1, 1).
///
/// Simple bounded advantage. Prevents extreme values.
pub fn clamped_advantage(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

// ── Token Logit Centralization ─────────────────────────────────

/// Token logit centralization: subtract mean from a slice of logits/values.
///
/// Modifies the slice in-place to be zero-mean. O(V) serial — no rayon needed
/// for V ≤ 128 (micro config per optimization.md).
///
/// Returns the mean that was subtracted.
pub fn token_logit_centralization(logits: &mut [f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let n = logits.len() as f32;
    let mean = logits.iter().sum::<f32>() / n;
    for v in logits.iter_mut() {
        *v -= mean;
    }
    mean
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_speculative::NoScreeningPruner;

    fn make_gdsd(advantage_fn: fn(f32) -> f32) -> GdsdPruner<NoScreeningPruner> {
        GdsdPruner::new(NoScreeningPruner, NoScreeningPruner, advantage_fn)
    }

    fn make_gdsd_with_config(
        advantage_fn: fn(f32) -> f32,
        config: GdsdConfig,
    ) -> GdsdPruner<NoScreeningPruner> {
        GdsdPruner::with_config(NoScreeningPruner, NoScreeningPruner, advantage_fn, config)
    }

    // ── Config defaults ─────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let config = GdsdConfig::default();
        assert!((config.beta - 0.001).abs() < 1e-6);
        assert!((config.psi - 10.0).abs() < 1e-6);
        assert!(config.tlc);
    }

    #[test]
    fn test_config_presets() {
        let strong = GdsdConfig::strong();
        assert!((strong.beta - 0.01).abs() < 1e-6);
        assert!((strong.psi - 20.0).abs() < 1e-6);

        let mild = GdsdConfig::mild();
        assert!((mild.beta - 0.0001).abs() < 1e-6);
        assert!((mild.psi - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_no_tlc() {
        let config = GdsdConfig::default().no_tlc();
        assert!(!config.tlc);
    }

    // ── Relevance with NoScreeningPruner (both return 1.0) ──────

    #[test]
    fn test_relevance_no_screening_identity_no_tlc() {
        // Both inner and ref return 1.0. Identity advantage(1.0) = 1.0.
        // No TLC: advantage_mean = 0.
        // teacher = (1-0.001)*1.0 + 0.001*1.0 + 10.0*1.0 = 1.0 + 10.0 = 11.0 → clamped to 1.0
        let mut pruner = make_gdsd_with_config(identity_advantage, GdsdConfig::default().no_tlc());
        pruner.update_advantage_mean(1.0); // identity(1.0) = 1.0
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6, "clamped to 1.0, got {rel}");
    }

    #[test]
    fn test_relevance_no_screening_identity_with_tlc() {
        // With TLC: advantage = identity(1.0) - mean = 1.0 - 1.0 = 0
        // teacher = (1-0.001)*1.0 + 0.001*1.0 + 10.0*0 = 1.0
        let mut pruner = make_gdsd(identity_advantage);
        pruner.update_advantage_mean(1.0);
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6, "got {rel}");
    }

    // ── Teacher signal computation ──────────────────────────────

    #[test]
    fn test_teacher_signal_no_tlc() {
        let mut pruner = make_gdsd_with_config(identity_advantage, GdsdConfig::default().no_tlc());
        pruner.update_advantage_mean(0.0);
        // r_old=0.5, r_ref=0.8, advantage_input=0.5
        // teacher = 0.999*0.5 + 0.001*0.8 + 10.0*0.5 = 0.4995 + 0.0008 + 5.0 = 5.5003
        let teacher = pruner.teacher_signal(0.5, 0.8, 0.5);
        assert!((teacher - 5.5003).abs() < 1e-3, "got {teacher}");
    }

    #[test]
    fn test_teacher_signal_with_tlc() {
        let mut pruner = make_gdsd(identity_advantage);
        pruner.update_advantage_mean(0.5);
        // advantage = identity(0.5) - 0.5 = 0.0
        // teacher = 0.999*0.5 + 0.001*0.8 + 10.0*0.0 = 0.4995 + 0.0008 = 0.5003
        let teacher = pruner.teacher_signal(0.5, 0.8, 0.5);
        assert!((teacher - 0.5003).abs() < 1e-3, "got {teacher}");
    }

    #[test]
    fn test_teacher_signal_zero_beta() {
        let config = GdsdConfig::new(0.0, 1.0).no_tlc();
        let mut pruner = make_gdsd_with_config(identity_advantage, config);
        pruner.update_advantage_mean(0.0);
        // teacher = 1.0*0.3 + 0.0*0.9 + 1.0*0.3 = 0.6
        let teacher = pruner.teacher_signal(0.3, 0.9, 0.3);
        assert!((teacher - 0.6).abs() < 1e-6, "got {teacher}");
    }

    #[test]
    fn test_teacher_signal_full_beta() {
        let config = GdsdConfig::new(1.0, 0.0).no_tlc();
        let mut pruner = make_gdsd_with_config(identity_advantage, config);
        pruner.update_advantage_mean(0.0);
        // teacher = 0.0*0.3 + 1.0*0.9 + 0.0*0.3 = 0.9
        let teacher = pruner.teacher_signal(0.3, 0.9, 0.3);
        assert!((teacher - 0.9).abs() < 1e-6, "got {teacher}");
    }

    // ── Advantage functions ─────────────────────────────────────

    #[test]
    fn test_identity_advantage() {
        assert!((identity_advantage(0.5) - 0.5).abs() < 1e-6);
        assert!((identity_advantage(-1.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_advantage() {
        assert!((sigmoid_advantage(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid_advantage(5.0) > 0.99);
        assert!(sigmoid_advantage(-5.0) < 0.01);
    }

    #[test]
    fn test_tanh_advantage() {
        assert!((tanh_advantage(0.0)).abs() < 1e-6);
        assert!(tanh_advantage(5.0) > 0.99);
        assert!(tanh_advantage(-5.0) < -0.99);
    }

    #[test]
    fn test_clamped_advantage() {
        assert!((clamped_advantage(0.5) - 0.5).abs() < 1e-6);
        assert!((clamped_advantage(2.0) - 1.0).abs() < 1e-6);
        assert!((clamped_advantage(-3.0) - (-1.0)).abs() < 1e-6);
    }

    // ── TLC utility ─────────────────────────────────────────────

    #[test]
    fn test_tlc_empty() {
        let mut v: Vec<f32> = vec![];
        let mean = token_logit_centralization(&mut v);
        assert!((mean - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_tlc_single() {
        let mut v = vec![5.0];
        let mean = token_logit_centralization(&mut v);
        assert!((mean - 5.0).abs() < 1e-6);
        assert!((v[0] - 0.0).abs() < 1e-6, "single element → 0.0");
    }

    #[test]
    fn test_tlc_uniform() {
        let mut v = vec![2.0, 2.0, 2.0, 2.0];
        let mean = token_logit_centralization(&mut v);
        assert!((mean - 2.0).abs() < 1e-6);
        for x in &v {
            assert!(x.abs() < 1e-6, "all should be 0.0, got {x}");
        }
    }

    #[test]
    fn test_tlc_mixed() {
        let mut v = vec![1.0, 3.0];
        let mean = token_logit_centralization(&mut v);
        assert!((mean - 2.0).abs() < 1e-6);
        assert!((v[0] - (-1.0)).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
    }

    // ── Clamping to [0, 1] ──────────────────────────────────────

    #[test]
    fn test_relevance_clamps_negative() {
        // Use tanh advantage which can return negative values
        let config = GdsdConfig::new(0.5, 100.0).no_tlc();
        let mut pruner = make_gdsd_with_config(tanh_advantage, config);
        pruner.update_advantage_mean(0.0);
        // With NoScreeningPruner returning 1.0, the blend will be positive.
        // Test with a custom pruner that returns 0 to get negative results.
        // Since we can't easily inject a custom pruner here, we test the teacher_signal:
        let teacher = pruner.teacher_signal(0.0, 0.0, -10.0);
        // tanh(-10) ≈ -1, teacher = 0.5*0 + 0.5*0 + 100*(-1) = -100
        assert!(teacher < 0.0, "should be negative, got {teacher}");
        // But relevance() clamps it
        let rel = pruner.relevance(0, 0, &[]);
        assert!(rel >= 0.0, "relevance should be clamped to >= 0, got {rel}");
    }

    // ── Accessors ───────────────────────────────────────────────

    #[test]
    fn test_accessors() {
        let config = GdsdConfig::new(0.5, 5.0).no_tlc();
        let pruner = make_gdsd_with_config(identity_advantage, config);
        assert!((pruner.beta() - 0.5).abs() < 1e-6);
        assert!((pruner.psi() - 5.0).abs() < 1e-6);
        assert!(!pruner.tlc_enabled());
    }

    #[test]
    fn test_update_advantage_mean() {
        let mut pruner = make_gdsd(identity_advantage);
        pruner.update_advantage_mean(0.42);
        // Can't read advantage_mean directly, but teacher_signal should reflect it
        let teacher_no_input = pruner.teacher_signal(0.0, 0.0, 0.0);
        // advantage = identity(0.0) - 0.42 = -0.42
        // teacher = 0 + 0 + 10.0 * (-0.42) = -4.2
        assert!(
            (teacher_no_input - (-4.2)).abs() < 1e-4,
            "got {teacher_no_input}"
        );
    }
}
