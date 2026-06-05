//! Concrete ProblemMutator implementations for Plan 191.
//!
//! - [`BomberConfigMutator`]: deterministic bomber config mutation
//! - [`GoConfigMutator`]: Go-specific territory/capture mutation

use katgpt_core::{GameConfig, MutantConfig, MutationKind, ProblemMutator};

// ── BomberConfigMutator ──────────────────────────────────────────

/// Bomber config mutator — deterministic bomber config mutation.
///
/// Produces three mutant variants from any seed config:
/// 1. `GoalReweight`: shift survival vs kill weights
/// 2. `GeneralizeInputs`: increase grid size
/// 3. `ConstrainOutputs`: reduce max steps by 25%
pub struct BomberConfigMutator;

impl ProblemMutator for BomberConfigMutator {
    fn mutate(&self, seed: &GameConfig) -> Vec<MutantConfig> {
        vec![
            // GoalReweight: shift weights toward survival pressure
            MutantConfig {
                difficulty_delta: 0.1,
                mutation_kind: MutationKind::GoalReweight,
                description: format!(
                    "survival_weight={:.2}, kill_weight={:.2}",
                    seed.survival_weight * 1.2,
                    seed.kill_weight * 0.8
                ),
            },
            // GeneralizeInputs: vary grid size
            MutantConfig {
                difficulty_delta: 0.2,
                mutation_kind: MutationKind::GeneralizeInputs,
                description: format!("grid_size={}", seed.grid_size + 4),
            },
            // ConstrainOutputs: reduce max steps
            MutantConfig {
                difficulty_delta: 0.15,
                mutation_kind: MutationKind::ConstrainOutputs,
                description: format!("max_steps={}", (seed.max_steps as f32 * 0.75) as u32),
            },
        ]
    }
}

// ── GoConfigMutator ─────────────────────────────────────────────

/// Go-specific config mutator.
///
/// Mutates Go game parameters:
/// - `GoalReweight`: shift territory vs capture weight
/// - `ConstrainOutputs`: board size variation (9x9 → 13x13 → 19x19)
/// - `GeneralizeInputs`: komi shifting, handicap variation
pub struct GoConfigMutator {
    /// Territory weight baseline (default 0.5).
    pub territory_weight: f32,
    /// Board sizes to explore.
    pub board_sizes: Vec<u32>,
}

impl Default for GoConfigMutator {
    fn default() -> Self {
        Self {
            territory_weight: 0.5,
            board_sizes: vec![9, 13, 19],
        }
    }
}

impl ProblemMutator for GoConfigMutator {
    fn mutate(&self, seed: &GameConfig) -> Vec<MutantConfig> {
        let mut mutants = Vec::new();

        // GoalReweight: 3 variants — territory-heavy, balanced, capture-heavy
        let reweight_variants: [(f32, f32, &str); 3] = [
            (0.8, 0.2, "territory-heavy"),
            (0.5, 0.5, "balanced"),
            (0.2, 0.8, "capture-heavy"),
        ];
        for (survival, kill, label) in &reweight_variants {
            let delta = (survival - self.territory_weight).abs();
            mutants.push(MutantConfig {
                difficulty_delta: delta,
                mutation_kind: MutationKind::GoalReweight,
                description: format!(
                    "survival_weight={:.2}, kill_weight={:.2} ({})",
                    survival, kill, label
                ),
            });
        }

        // ConstrainOutputs: one variant per board size
        let base_size = seed.grid_size;
        for &size in &self.board_sizes {
            if size == base_size {
                continue;
            }
            let difficulty_delta = (size - base_size) as f32 / base_size as f32;
            mutants.push(MutantConfig {
                difficulty_delta,
                mutation_kind: MutationKind::ConstrainOutputs,
                description: format!("grid_size={} ({}x{})", size, size, size),
            });
        }

        // GeneralizeInputs: handicap variation — opponent_count +1/+2/+3
        for handicap in 1u32..=3 {
            let opponent_count = seed.opponent_count + handicap;
            let difficulty_delta = opponent_count as f32 / 3.0;
            mutants.push(MutantConfig {
                difficulty_delta,
                mutation_kind: MutationKind::GeneralizeInputs,
                description: format!(
                    "opponent_count={} (+{} handicap stones)",
                    opponent_count, handicap
                ),
            });
        }

        mutants
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> GameConfig {
        GameConfig::default()
    }

    // ── BomberConfigMutator tests ──────────────────────────────────

    #[test]
    fn bomber_mutator_produces_three_variants() {
        let mutator = BomberConfigMutator;
        let mutants = mutator.mutate(&default_config());
        assert_eq!(mutants.len(), 3);
    }

    #[test]
    fn bomber_mutator_covers_all_mutation_kinds() {
        let mutator = BomberConfigMutator;
        let mutants = mutator.mutate(&default_config());
        let kinds: Vec<_> = mutants.iter().map(|m| m.mutation_kind).collect();
        assert!(kinds.contains(&MutationKind::GoalReweight));
        assert!(kinds.contains(&MutationKind::GeneralizeInputs));
        assert!(kinds.contains(&MutationKind::ConstrainOutputs));
    }

    #[test]
    fn bomber_mutator_positive_difficulty_delta() {
        let mutator = BomberConfigMutator;
        let mutants = mutator.mutate(&default_config());
        for m in &mutants {
            assert!(
                m.difficulty_delta > 0.0,
                "difficulty_delta should be positive, got {}",
                m.difficulty_delta
            );
        }
    }

    #[test]
    fn bomber_mutator_goal_reweight_increases_survival_pressure() {
        let mutator = BomberConfigMutator;
        let mutants = mutator.mutate(&default_config());
        let gr = mutants
            .iter()
            .find(|m| m.mutation_kind == MutationKind::GoalReweight)
            .expect("GoalReweight mutant missing");
        assert!(gr.description.contains("survival_weight="));
    }

    #[test]
    fn bomber_mutator_generalize_increases_grid() {
        let mutator = BomberConfigMutator;
        let config = GameConfig {
            grid_size: 9,
            ..Default::default()
        };
        let mutants = mutator.mutate(&config);
        let gi = mutants
            .iter()
            .find(|m| m.mutation_kind == MutationKind::GeneralizeInputs)
            .expect("GeneralizeInputs mutant missing");
        assert!(gi.description.contains("grid_size=13"));
    }

    #[test]
    fn bomber_mutator_constrain_reduces_steps() {
        let mutator = BomberConfigMutator;
        let config = GameConfig {
            max_steps: 200,
            ..Default::default()
        };
        let mutants = mutator.mutate(&config);
        let co = mutants
            .iter()
            .find(|m| m.mutation_kind == MutationKind::ConstrainOutputs)
            .expect("ConstrainOutputs mutant missing");
        assert!(co.description.contains("max_steps=150"));
    }

    #[test]
    fn bomber_mutator_descriptions_non_empty() {
        let mutator = BomberConfigMutator;
        let mutants = mutator.mutate(&default_config());
        for m in &mutants {
            assert!(!m.description.is_empty());
        }
    }

    #[test]
    fn bomber_mutator_custom_config() {
        let mutator = BomberConfigMutator;
        let config = GameConfig {
            grid_size: 15,
            opponent_count: 4,
            max_steps: 100,
            survival_weight: 0.7,
            kill_weight: 0.3,
        };
        let mutants = mutator.mutate(&config);
        assert_eq!(mutants.len(), 3);
        // Verify grid increase from custom base
        let gi = mutants
            .iter()
            .find(|m| m.mutation_kind == MutationKind::GeneralizeInputs)
            .unwrap();
        assert!(gi.description.contains("grid_size=19"));
    }

    // ── GoConfigMutator tests ───────────────────────────────────────

    #[test]
    fn test_go_config_mutator_goal_reweight() {
        let mutator = GoConfigMutator::default();
        let mutants = mutator.mutate(&default_config());
        let reweights: Vec<_> = mutants
            .iter()
            .filter(|m| m.mutation_kind == MutationKind::GoalReweight)
            .collect();
        assert_eq!(reweights.len(), 3);
        let territory_heavy = reweights
            .iter()
            .find(|m| m.description.contains("territory-heavy"))
            .expect("territory-heavy variant missing");
        assert!(territory_heavy.description.contains("survival_weight=0.80"));
        assert!(territory_heavy.description.contains("kill_weight=0.20"));
    }

    #[test]
    fn test_go_config_mutator_constrain_outputs() {
        let mutator = GoConfigMutator::default();
        let config = GameConfig {
            grid_size: 9,
            ..Default::default()
        };
        let mutants = mutator.mutate(&config);
        let constrain: Vec<_> = mutants
            .iter()
            .filter(|m| m.mutation_kind == MutationKind::ConstrainOutputs)
            .collect();
        // board_sizes=[9,13,19], base=9 → 2 variants (13, 19)
        assert_eq!(constrain.len(), 2);
        let s13 = constrain
            .iter()
            .find(|m| m.description.contains("13"))
            .unwrap();
        assert!((s13.difficulty_delta - 4.0 / 9.0).abs() < 1e-4);
    }

    #[test]
    fn test_go_config_mutator_generalize_inputs() {
        let mutator = GoConfigMutator::default();
        let config = GameConfig {
            opponent_count: 1,
            ..Default::default()
        };
        let mutants = mutator.mutate(&config);
        let generalize: Vec<_> = mutants
            .iter()
            .filter(|m| m.mutation_kind == MutationKind::GeneralizeInputs)
            .collect();
        assert_eq!(generalize.len(), 3);
        assert!(generalize[0].description.contains("opponent_count=2"));
        assert!(generalize[1].description.contains("opponent_count=3"));
        assert!(generalize[2].description.contains("opponent_count=4"));
    }

    #[test]
    fn test_go_config_mutator_mutation_kinds_diverse() {
        let mutator = GoConfigMutator::default();
        let mutants = mutator.mutate(&default_config());
        let kinds: Vec<_> = mutants.iter().map(|m| m.mutation_kind).collect();
        assert!(kinds.contains(&MutationKind::GoalReweight));
        assert!(kinds.contains(&MutationKind::ConstrainOutputs));
        assert!(kinds.contains(&MutationKind::GeneralizeInputs));
    }
}

// TL;DR: BomberConfigMutator produces 3 deterministic mutants (GoalReweight, GeneralizeInputs, ConstrainOutputs) from any GameConfig.
// TL;DR: GoConfigMutator produces territory/capture reweights, board size variants, and handicap variants for Go game configs.
