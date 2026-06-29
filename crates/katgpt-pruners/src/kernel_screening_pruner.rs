//! KernelScreeningPruner — kernel-tricked relevance for ScreeningPruner (Plan 234).

use katgpt_core::traits::ScreeningPruner;
use super::kernel_scoring::KernelKind;

/// Wraps a ScreeningPruner and applies kernel transformation to relevance scores.
pub struct KernelScreeningPruner<P> {
    pub inner: P,
    pub kind: KernelKind,
}

impl<P: ScreeningPruner> KernelScreeningPruner<P> {
    pub fn new(inner: P, kind: KernelKind) -> Self {
        Self { inner, kind }
    }
}

impl<P: ScreeningPruner> ScreeningPruner for KernelScreeningPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let raw = self.inner.relevance(depth, token_idx, parent_tokens);
        match self.kind {
            KernelKind::Linear => raw, // identity for linear
            KernelKind::Gaussian { sigma } => {
                let diff = raw - 1.0;
                (-diff * diff / (sigma * sigma)).exp()
            }
            KernelKind::Polynomial { degree, c: poly_c } => (raw + poly_c).powf(degree),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstantScreener {
        val: f32,
    }
    impl ScreeningPruner for ConstantScreener {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            self.val
        }
    }

    #[test]
    fn gaussian_kernel_perfect_relevance() {
        let inner = ConstantScreener { val: 1.0 };
        let kernel =
            KernelScreeningPruner::new(inner, KernelKind::Gaussian { sigma: 1.0 });
        let score = kernel.relevance(0, 0, &[]);
        assert!(
            (score - 1.0).abs() < 1e-5,
            "perfect relevance should be 1.0, got {}",
            score
        );
    }

    #[test]
    fn gaussian_kernel_distant_relevance() {
        let inner = ConstantScreener { val: 0.0 };
        let kernel =
            KernelScreeningPruner::new(inner, KernelKind::Gaussian { sigma: 1.0 });
        let score = kernel.relevance(0, 0, &[]);
        assert!(
            score < 0.5,
            "distant relevance should be low, got {}",
            score
        );
    }

    #[test]
    fn polynomial_kernel_amplifies() {
        let inner = ConstantScreener { val: 0.5 };
        let kernel = KernelScreeningPruner::new(
            inner,
            KernelKind::Polynomial {
                degree: 2.0,
                c: 1.0,
            },
        );
        let score = kernel.relevance(0, 0, &[]);
        // (0.5 + 1.0)^2 = 2.25
        assert!(
            (score - 2.25).abs() < 1e-5,
            "polynomial score should be 2.25, got {}",
            score
        );
    }
}
