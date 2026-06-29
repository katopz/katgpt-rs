//! Unnormalized KL anchoring for bandit Q-values.
//!
//! From SDPG paper: anchors policy to frozen reference via UFKL or URKL.
//! Prevents Q-value collapse (mode collapse analog) in long self-play.

/// KL anchor type for bandit Q-value regularization.
#[derive(Clone, Debug)]
pub enum KlAnchor {
    /// Unnormalized Forward KL — mass-corrected anchoring.
    /// L(a) = β * [Q_ref(a)/Q(a) + log(Q(a)/Q_ref(a))]
    Ufkl { beta: f32 },

    /// Unnormalized Reverse KL — variance-bounded, mode-seeking.
    /// L(a) = β * 0.5 * [log(Q(a)/Q_ref(a))]²
    /// Recommended default (paper: variance-bounded, no division by Q).
    Urkl { beta: f32 },
}

impl KlAnchor {
    /// Default URKL anchor with β=0.01.
    pub fn default_urkl() -> Self {
        KlAnchor::Urkl { beta: 0.01 }
    }

    /// Default UFKL anchor with β=0.01.
    pub fn default_ufkl() -> Self {
        KlAnchor::Ufkl { beta: 0.01 }
    }

    /// Compute per-arm anchoring adjustment to subtract from Q-values.
    ///
    /// Returns Vec<f32> of adjustments. Subtract from Q to apply anchoring.
    pub fn anchor_loss(&self, q: &[f32], q_ref: &[f32]) -> Vec<f32> {
        assert_eq!(q.len(), q_ref.len());
        match self {
            KlAnchor::Ufkl { beta } => q
                .iter()
                .zip(q_ref.iter())
                .map(|(&qi, &ri)| {
                    if qi > 1e-8 && ri > 1e-8 {
                        let ratio = qi / ri;
                        // L(a) = β * (q_ref/q + ln(q/q_ref) - 1)
                        // At q=q_ref: ratio=1, so (1/ratio + ln(ratio) - 1) = (1 + 0 - 1) = 0
                        beta * (1.0 / ratio + ratio.ln() - 1.0)
                    } else if ri > 1e-8 {
                        // Q collapsed to ~0 but ref is positive → strong pull up
                        beta * ri / qi.max(1e-8)
                    } else {
                        0.0
                    }
                })
                .collect(),
            KlAnchor::Urkl { beta } => q
                .iter()
                .zip(q_ref.iter())
                .map(|(&qi, &ri)| {
                    if qi > 1e-8 && ri > 1e-8 {
                        let log_ratio = (qi / ri).ln();
                        beta * 0.5 * log_ratio * log_ratio
                    } else {
                        0.0
                    }
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ufkl_zero_loss_when_equal() {
        let anchor = KlAnchor::Ufkl { beta: 0.01 };
        let q = vec![1.0, 2.0, 3.0];
        let loss = anchor.anchor_loss(&q, &q);
        for l in &loss {
            assert!(
                l.abs() < 1e-5,
                "UFKL loss should be ~0 when Q=Q_ref, got {}",
                l
            );
        }
    }

    #[test]
    fn test_ufkl_positive_loss_when_diverged() {
        let anchor = KlAnchor::Ufkl { beta: 0.01 };
        let q = vec![1.0, 5.0, 1.0];
        let q_ref = vec![1.0, 1.0, 1.0];
        let loss = anchor.anchor_loss(&q, &q_ref);
        assert!(
            loss[1] > 0.0,
            "UFKL loss should be positive when Q diverges, got {}",
            loss[1]
        );
    }

    #[test]
    fn test_urkl_zero_loss_when_equal() {
        let anchor = KlAnchor::Urkl { beta: 0.01 };
        let q = vec![1.0, 2.0, 3.0];
        let loss = anchor.anchor_loss(&q, &q);
        for l in &loss {
            assert!(
                l.abs() < 1e-5,
                "URKL loss should be ~0 when Q=Q_ref, got {}",
                l
            );
        }
    }

    #[test]
    fn test_urkl_numerical_stability_near_zero() {
        let anchor = KlAnchor::Urkl { beta: 0.01 };
        let q = vec![1e-10, 1.0, 1e-10];
        let q_ref = vec![1.0, 1.0, 1.0];
        let loss = anchor.anchor_loss(&q, &q_ref);
        // Should not panic or produce NaN/Inf
        for l in &loss {
            assert!(
                l.is_finite(),
                "URKL loss should be finite near zero, got {}",
                l
            );
        }
    }
}
