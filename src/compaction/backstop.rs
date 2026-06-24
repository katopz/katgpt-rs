//! The `Backstop` enum — token-pct force-compaction override.
//!
//! Even when the rubric declines to fire, a token-pct backstop forces
//! compaction to avoid unbounded context growth. This is the paper's safety
//! net (SelfCompact §3): the rubric decides *when it is safe*; the backstop
//! decides *when it is mandatory regardless of safety*.
//!
//! The existing `OnlineCompactor::trigger_threshold()` is the fixed-position
//! baseline that CUCG *replaces as the primary trigger* but *preserves as the
//! backstop arm* — when CUCG is promoted to default-on, the fixed-interval
//! trigger becomes the `Forced` decision's mechanism, not the primary gate.

/// Token-pct backstop. Forces compaction when the prompt length exceeds a
/// fraction of the context window, regardless of the rubric verdict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Backstop {
    /// Never force — the rubric alone decides. Use with caution: an
    /// over-cautious rubric + `None` backstop means unbounded growth.
    None,
    /// Force compaction when `prompt_len >= pct * ctx_window`.
    ///
    /// `pct` is clamped to `[0.0, 1.0]` at construction. The paper's default
    /// is around 0.30–0.50 (force before the context window is exhausted).
    TokenPct(f32),
    /// Synonym for [`Backstop::None`] — never force. Provided for explicit
    /// "the caller asserts the rubric is the sole authority" configuration.
    Never,
}

impl Backstop {
    /// Construct a `TokenPct` backstop, clamping `pct` to `[0.0, 1.0]`.
    ///
    /// `pct <= 0.0` degenerates to "always force" (use `TokenPct(0.0)`
    /// deliberately if that's intended); `pct >= 1.0` to "force only at the
    /// full window". Values outside the range are silently clamped (this is
    /// the documented behavior — callers may pass sloppy bounds).
    #[inline]
    #[must_use]
    pub fn token_pct(pct: f32) -> Self {
        debug_assert!(
            pct.is_finite(),
            "Backstop::token_pct: pct must be finite, got {pct}"
        );
        Self::TokenPct(pct.clamp(0.0, 1.0))
    }

    /// Returns `true` iff the backstop forces compaction at the given
    /// `prompt_len` / `ctx_window`.
    ///
    /// - [`Backstop::None`] / [`Backstop::Never`] → always `false`.
    /// - [`Backstop::TokenPct(p)`] → `prompt_len >= (p * ctx_window) as usize`.
    ///
    /// Zero-allocation: a single comparison on the hot path.
    #[inline]
    #[must_use]
    pub fn should_force(&self, prompt_len: usize, ctx_window: usize) -> bool {
        match self {
            Self::None | Self::Never => false,
            Self::TokenPct(pct) => {
                // Avoid f64 promotion: compute the threshold in usize via
                // `(pct * ctx_window as f32) as usize`. Float rounding at the
                // boundary is acceptable — the backstop is a coarse safety
                // net, not a precise trigger.
                let threshold = (*pct * ctx_window as f32) as usize;
                prompt_len >= threshold
            }
        }
    }

    /// Returns the configured pct, or `None` for [`Backstop::None`] /
    /// [`Backstop::Never`].
    #[inline]
    #[must_use]
    pub const fn pct(&self) -> Option<f32> {
        match self {
            Self::TokenPct(p) => Some(*p),
            Self::None | Self::Never => None,
        }
    }
}

impl Default for Backstop {
    /// Default is `TokenPct(0.30)` — the paper's conservative backstop.
    #[inline]
    fn default() -> Self {
        Self::TokenPct(0.30)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_pct_forces_at_threshold() {
        let b = Backstop::token_pct(0.30);
        // ctx_window = 1000 → threshold = 300.
        assert!(!b.should_force(299, 1000), "below threshold: no force");
        assert!(b.should_force(300, 1000), "at threshold: force");
        assert!(b.should_force(500, 1000), "above threshold: force");
    }

    #[test]
    fn token_pct_just_below_threshold_does_not_force() {
        let b = Backstop::token_pct(0.50);
        assert!(!b.should_force(499, 1000));
        assert!(b.should_force(500, 1000));
    }

    #[test]
    fn none_never_never_force() {
        assert!(!Backstop::None.should_force(10_000, 1000));
        assert!(!Backstop::Never.should_force(10_000, 1000));
    }

    #[test]
    fn token_pct_clamps_out_of_range() {
        // Negative clamps to 0 → always forces (threshold 0).
        let neg = Backstop::token_pct(-0.5);
        assert!(matches!(neg, Backstop::TokenPct(0.0)));
        assert!(
            neg.should_force(0, 1000),
            "pct=0 → threshold 0 → always force"
        );

        // > 1 clamps to 1 → forces only at full window.
        let big = Backstop::token_pct(1.5);
        assert!(matches!(big, Backstop::TokenPct(1.0)));
        assert!(!big.should_force(999, 1000));
        assert!(big.should_force(1000, 1000));
    }

    #[test]
    fn pct_accessor() {
        assert_eq!(Backstop::None.pct(), None);
        assert_eq!(Backstop::Never.pct(), None);
        assert_eq!(Backstop::token_pct(0.42).pct(), Some(0.42));
    }

    #[test]
    fn default_is_30_pct() {
        assert_eq!(Backstop::default(), Backstop::TokenPct(0.30));
    }

    #[test]
    fn zero_ctx_window_edge_case() {
        // ctx_window = 0 → threshold = 0 → any prompt_len >= 0 forces.
        // This is the degenerate "no budget" case; the backstop correctly
        // forces immediately (the rubric should ideally have fired first).
        let b = Backstop::token_pct(0.30);
        assert!(b.should_force(0, 0));
        assert!(b.should_force(1, 0));
    }
}
