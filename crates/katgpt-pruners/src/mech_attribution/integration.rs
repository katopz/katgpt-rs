//! Integration with ROPD rubric — catalyst-weighted rubric scoring.
//!
//! Provides [`score_with_influence`] which combines a rubric score with a
//! catalyst influence score, weighting data selection by structural catalyst
//! detection.
//!
//! Feature-gated behind `#[cfg(all(feature = "mech_attribution", feature = "ropd_rubric"))]`.

use super::types::{InfluenceConfig, MechInfluenceScore};

/// Combine a rubric score with a catalyst influence score.
///
/// The combined score is:
/// ```text
/// combined = rubric_score * (1 - alpha) + catalyst.catalyst_overlap * alpha * weight
/// ```
///
/// where `alpha` is `config.catalyst_threshold` (reused as blending weight) and
/// `weight` is the catalyst influence multiplier: 1.5 for high-influence samples,
/// 1.0 otherwise.
///
/// When `is_high_influence` is true, the catalyst component gets a 50% boost,
/// reflecting that top-K catalyst-scored samples should dominate selection.
pub fn score_with_influence(
    rubric_score: f32,
    catalyst: &MechInfluenceScore,
    config: &InfluenceConfig,
) -> f32 {
    // Reuse catalyst_threshold as the blending factor (0.0 = pure rubric, 1.0 = pure catalyst)
    let alpha = config.catalyst_threshold.clamp(0.0, 1.0);

    // High-influence samples get a 50% boost on the catalyst component
    let influence_multiplier = if catalyst.is_high_influence { 1.5 } else { 1.0 };

    let combined =
        rubric_score * (1.0 - alpha) + catalyst.catalyst_overlap * alpha * influence_multiplier;

    // Clamp to [0, 1] — both inputs are bounded so the output should be too
    combined.clamp(0.0, 1.0)
}

/// Batch version: apply [`score_with_influence`] to a slice of (rubric_score, MechInfluenceScore) pairs.
///
/// Returns a Vec of combined scores in the same order.
pub fn batch_score_with_influence(
    rubric_scores: &[f32],
    catalyst_scores: &[(usize, MechInfluenceScore)],
    config: &InfluenceConfig,
) -> Vec<(usize, f32)> {
    assert_eq!(
        rubric_scores.len(),
        catalyst_scores.len(),
        "rubric_scores and catalyst_scores must have the same length"
    );

    catalyst_scores
        .iter()
        .enumerate()
        .map(|(i, (idx, catalyst))| {
            let combined = score_with_influence(rubric_scores[i], catalyst, config);
            (*idx, combined)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mech_attribution::CatalystPattern;

    #[test]
    fn test_score_with_influence_pure_rubric() {
        // alpha = 0 means pure rubric score
        let config = InfluenceConfig {
            catalyst_threshold: 0.0,
            ..Default::default()
        };
        let catalyst = MechInfluenceScore {
            catalyst_overlap: 0.9,
            pattern: CatalystPattern::XmlRepetition,
            is_high_influence: true,
        };
        let result = score_with_influence(0.8, &catalyst, &config);
        assert!(
            (result - 0.8).abs() < 1e-5,
            "alpha=0 should give pure rubric, got {result}"
        );
    }

    #[test]
    fn test_score_with_influence_pure_catalyst() {
        // alpha = 1 means pure catalyst
        let config = InfluenceConfig {
            catalyst_threshold: 1.0,
            ..Default::default()
        };
        let catalyst = MechInfluenceScore {
            catalyst_overlap: 0.7,
            pattern: CatalystPattern::CodeSignature,
            is_high_influence: false,
        };
        let result = score_with_influence(0.9, &catalyst, &config);
        assert!(
            (result - 0.7).abs() < 1e-5,
            "alpha=1, not high-influence should give catalyst_overlap, got {result}"
        );
    }

    #[test]
    fn test_score_with_influence_high_influence_boost() {
        let config = InfluenceConfig {
            catalyst_threshold: 1.0,
            ..Default::default()
        };
        let catalyst = MechInfluenceScore {
            catalyst_overlap: 0.6,
            pattern: CatalystPattern::XmlRepetition,
            is_high_influence: true,
        };
        let result = score_with_influence(0.9, &catalyst, &config);
        let expected = 0.6 * 1.5; // 0.9
        assert!(
            (result - expected).abs() < 1e-5,
            "high-influence boost should give {expected}, got {result}"
        );
    }

    #[test]
    fn test_score_with_influence_balanced() {
        // alpha = 0.5
        let config = InfluenceConfig {
            catalyst_threshold: 0.5,
            ..Default::default()
        };
        let catalyst = MechInfluenceScore {
            catalyst_overlap: 0.8,
            pattern: CatalystPattern::LatexFormula,
            is_high_influence: false,
        };
        let result = score_with_influence(0.6, &catalyst, &config);
        // 0.6 * 0.5 + 0.8 * 0.5 * 1.0 = 0.3 + 0.4 = 0.7
        let expected = 0.7;
        assert!(
            (result - expected).abs() < 1e-5,
            "balanced blend should give {expected}, got {result}"
        );
    }
}
