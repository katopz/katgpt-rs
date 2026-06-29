//! Concrete PartialScorer implementations for Plan 191.
//!
//! - [`WinLossScorer`]: backward-compatible binary {0.0, 1.0}
//! - [`BomberPartialScorer`]: weighted blend of survival, kills, efficiency

use katgpt_core::{GameTrace, PartialScorer};

// ── WinLossScorer ────────────────────────────────────────────────

/// Binary scorer: adapter for backward compat. Maps win=1.0, loss=0.0.
///
/// Use this as the control group when benchmarking graduated scorers.
pub struct WinLossScorer;

impl PartialScorer for WinLossScorer {
    #[inline]
    fn partial_score(&self, trace: &GameTrace) -> f32 {
        if trace.final_reward > 0.0 { 1.0 } else { 0.0 }
    }

    fn score_breakdown(&self, trace: &GameTrace) -> Vec<(&'static str, f32)> {
        let score = self.partial_score(trace);
        vec![("win_loss", score)]
    }
}

// ── BomberPartialScorer ──────────────────────────────────────────

/// Bomber-specific partial scorer.
///
/// Score = weighted blend:
/// - survival  (0.4): fraction of max ticks survived
/// - kills     (0.3): kill count capped at 1.0
/// - safety    (0.2): inverse of danger exposure (= survival again)
/// - efficiency(0.1): kills per action, capped at 1.0
pub struct BomberPartialScorer {
    /// Maximum ticks for normalization.
    pub max_ticks: u32,
}

impl PartialScorer for BomberPartialScorer {
    #[inline]
    fn partial_score(&self, trace: &GameTrace) -> f32 {
        let mt = self.max_ticks.max(1) as f32;
        let survival = trace.survival_ticks as f32 / mt;
        let kills = (trace.kills as f32).min(1.0);
        let safety = survival; // bombs_avoided ≈ survival fraction
        let efficiency = if trace.actions_taken > 0 {
            (trace.kills as f32 / trace.actions_taken as f32).min(1.0)
        } else {
            0.0
        };
        0.4 * survival + 0.3 * kills + 0.2 * safety + 0.1 * efficiency
    }

    fn score_breakdown(&self, trace: &GameTrace) -> Vec<(&'static str, f32)> {
        let mt = self.max_ticks.max(1) as f32;
        let survival = trace.survival_ticks as f32 / mt;
        let kills = (trace.kills as f32).min(1.0);
        let safety = survival;
        let efficiency = if trace.actions_taken > 0 {
            (trace.kills as f32 / trace.actions_taken as f32).min(1.0)
        } else {
            0.0
        };
        vec![
            ("survival", 0.4 * survival),
            ("kills", 0.3 * kills),
            ("safety", 0.2 * safety),
            ("efficiency", 0.1 * efficiency),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win_trace() -> GameTrace {
        GameTrace {
            survival_ticks: 200,
            kills: 3,
            actions_taken: 50,
            max_ticks: 200,
            final_reward: 1.0,
        }
    }

    fn loss_trace() -> GameTrace {
        GameTrace {
            survival_ticks: 30,
            kills: 0,
            actions_taken: 10,
            max_ticks: 200,
            final_reward: 0.0,
        }
    }

    #[test]
    fn win_loss_scorer_win() {
        let scorer = WinLossScorer;
        assert!((scorer.partial_score(&win_trace()) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn win_loss_scorer_loss() {
        let scorer = WinLossScorer;
        assert!((scorer.partial_score(&loss_trace()) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn win_loss_scorer_zero_reward() {
        let scorer = WinLossScorer;
        let trace = GameTrace {
            final_reward: 0.0,
            ..Default::default()
        };
        assert!((scorer.partial_score(&trace) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn win_loss_breakdown() {
        let scorer = WinLossScorer;
        let bd = scorer.score_breakdown(&win_trace());
        assert_eq!(bd.len(), 1);
        assert_eq!(bd[0].0, "win_loss");
    }

    #[test]
    fn bomber_scorer_win_high() {
        let scorer = BomberPartialScorer { max_ticks: 200 };
        let score = scorer.partial_score(&win_trace());
        assert!(score > 0.7, "win trace should score high, got {score}");
    }

    #[test]
    fn bomber_scorer_loss_low() {
        let scorer = BomberPartialScorer { max_ticks: 200 };
        let score = scorer.partial_score(&loss_trace());
        assert!(
            score < 0.3,
            "early death trace should score low, got {score}"
        );
    }

    #[test]
    fn bomber_scorer_bounded() {
        let scorer = BomberPartialScorer { max_ticks: 200 };
        let trace = GameTrace {
            survival_ticks: 200,
            kills: 100,
            actions_taken: 1,
            max_ticks: 200,
            final_reward: 1.0,
        };
        let score = scorer.partial_score(&trace);
        assert!(
            (0.0..=1.0).contains(&score),
            "score must be in [0,1], got {score}"
        );
    }

    #[test]
    fn bomber_scorer_survival_weight_dominant() {
        let scorer = BomberPartialScorer { max_ticks: 200 };
        // Survive full game but no kills
        let survive_only = GameTrace {
            survival_ticks: 200,
            kills: 0,
            actions_taken: 50,
            max_ticks: 200,
            final_reward: 1.0,
        };
        let score = scorer.partial_score(&survive_only);
        // survival(0.4) + safety(0.2) = 0.6 even with zero kills
        assert!(
            score >= 0.58,
            "survival-only should score ≥0.58, got {score}"
        );
    }

    #[test]
    fn bomber_scorer_breakdown_sum_matches() {
        let scorer = BomberPartialScorer { max_ticks: 200 };
        let trace = win_trace();
        let total = scorer.partial_score(&trace);
        let bd = scorer.score_breakdown(&trace);
        let sum: f32 = bd.iter().map(|(_, v)| v).sum();
        assert!(
            (total - sum).abs() < 1e-5,
            "breakdown sum {sum} != total {total}"
        );
    }

    #[test]
    fn bomber_scorer_max_ticks_zero_safe() {
        let scorer = BomberPartialScorer { max_ticks: 0 };
        let trace = GameTrace::default();
        // Should not panic — max_ticks.max(1) prevents div-by-zero
        let _ = scorer.partial_score(&trace);
    }
}

// TL;DR: WinLossScorer (binary {0,1}) and BomberPartialScorer (weighted blend: survival 0.4 + kills 0.3 + safety 0.2 + efficiency 0.1).
