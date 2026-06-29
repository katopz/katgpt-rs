//! Feature class vocabulary tag (Plan 292 Phase 1, Research 267).
//!
//! Re-export shim for [`FeatureClass`] from `katgpt-core`. The enum lives next
//! to the [`ScreeningPruner`] trait in `katgpt-core/src/traits.rs` because the
//! trait's default method needs it; this module gives the path the plan called
//! out (`katgpt_rs::pruners::feature_class::FeatureClass`) without duplicating
//! the type definition.
//!
//! # Distilled vocabulary (Kortukov et al. 2026, Research 267 §1.1)
//!
//! | Class | What it reads | Safe use | Example primitives |
//! |-------|---------------|----------|--------------------|
//! | [`FeatureClass::Detection`] | Behavior *already realized* in generated text | Monitor, intervene downstream of the read | `EmotionDirections` (Plan 162), CNA (Plan 087), `FaithfulnessProbe` (Plan 278), `RegimeTransition` (Plan 215) |
//! | [`FeatureClass::Prediction`] | Probability of *future* behavior from intermediate state | Non-invasive steering via candidate selection | `FutureBehaviorProbe` (Plan 292 Phase 2) |
//!
//! The distinction matters because detection-side directions are a *different
//! linear subspace* from prediction-side directions. Treating them as the same
//! is the root cause of activation steering catastrophically degrading output
//! quality (paper §4.2: 10–100% format-filtered outputs at multipliers strong
//! enough to move the needle). FPCG's quality-preservation advantage comes
//! precisely from never mutating the residual stream and instead selecting
//! among already-generated candidates by their prediction-side probe score.

pub use katgpt_core::traits::{FeatureClass, ScreeningPruner};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emotion_vector::EmotionDirections;

    /// Plan 292 T1.5: `EmotionDirections::feature_class()` MUST be Detection.
    /// `EmotionDirections` is the canonical detection-side primitive — its
    /// direction vectors are extracted from contrastive *final-answer* pairs
    /// (Research 144), so they describe behavior already in the text, not a
    /// forecast of future behavior.
    #[test]
    fn emotion_directions_is_detection() {
        let dirs = EmotionDirections::zeros(8);
        // `EmotionDirections` doesn't implement ScreeningPruner directly (it's a
        // pure projection primitive), but the enum tag is the contract. We test
        // the explicit annotation here.
        assert_eq!(dirs.feature_class(), FeatureClass::Detection);
    }

    /// Plan 292 T1.5: the default trait impl returns Detection.
    /// Any type implementing `ScreeningPruner` without overriding
    /// `feature_class()` MUST report Detection.
    #[test]
    fn default_feature_class_is_detection() {
        struct AnonymousScreener;
        impl ScreeningPruner for AnonymousScreener {
            fn relevance(
                &self,
                _depth: usize,
                _token_idx: usize,
                _parent_tokens: &[usize],
            ) -> f32 {
                1.0
            }
        }
        let s = AnonymousScreener;
        assert_eq!(s.feature_class(), FeatureClass::Detection);
    }

    /// Plan 292 T1.5: `FutureBehaviorProbe::feature_class()` MUST be Prediction.
    /// Lives behind the `future_probe` feature; this is the canonical
    /// prediction-side primitive.
    #[cfg(feature = "future_probe")]
    #[test]
    fn future_behavior_probe_is_prediction() {
        use crate::future_probe::FutureBehaviorProbe;
        let probe = FutureBehaviorProbe::new(vec![0.0_f32; 4], 0.0, 0, "test");
        assert_eq!(probe.feature_class(), FeatureClass::Prediction);
    }

    /// Sanity: repr(u8) discriminants match the documented contract.
    #[test]
    fn feature_class_discriminants() {
        assert_eq!(FeatureClass::Detection as u8, 0);
        assert_eq!(FeatureClass::Prediction as u8, 1);
    }
}
