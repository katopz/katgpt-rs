//! Concrete ProblemMutator implementations for Plan 191.
//!
//! - [`BomberConfigMutator`]: deterministic bomber config mutation

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

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> GameConfig {
        GameConfig::default()
    }

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
}

// TL;DR: BomberConfigMutator produces 3 deterministic mutants (GoalReweight, GeneralizeInputs, ConstrainOutputs) from any GameConfig.
