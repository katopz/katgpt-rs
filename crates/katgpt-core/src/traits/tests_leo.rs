#[allow(unused_imports)]
use super::*;

// -- T5: sigmoid_bounded_q --

#[test]
fn test_sigmoid_bounded_q_bounds() {
    // Raw Q = 0 → sigmoid(0) = 0.5
    assert!((sigmoid_bounded_q(0.0) - 0.5).abs() < 1e-6);
    // Large positive → approaches 1.0
    assert!(sigmoid_bounded_q(10.0) > 0.99);
    // Large negative → approaches 0.0
    assert!(sigmoid_bounded_q(-10.0) < 0.01);
    // Symmetry
    assert!((sigmoid_bounded_q(1.0) + sigmoid_bounded_q(-1.0) - 1.0).abs() < 1e-6);
}

// -- T1: LeoHead default q_for_goal --

/// Minimal LeoHead impl for testing.
#[allow(dead_code)]
struct DummyLeoHead {
    goals: usize,
    actions: usize,
}

impl LeoHead for DummyLeoHead {
    fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
        vec![0.5; self.goals * self.actions]
    }
    #[inline]
    fn goal_count(&self) -> usize {
        self.goals
    }
    #[inline]
    fn action_count(&self) -> usize {
        self.actions
    }
}

#[test]
fn test_leo_head_q_for_goal() {
    let head = DummyLeoHead {
        goals: 3,
        actions: 4,
    };
    let state = vec![0.0; 8];
    let all_q = head.all_goals_q(&state);
    assert_eq!(all_q.len(), 12); // 3 goals × 4 actions

    let q0 = head.q_for_goal(&all_q, 0);
    assert_eq!(q0.len(), 4);
    assert_eq!(q0, &[0.5; 4]);

    let q2 = head.q_for_goal(&all_q, 2);
    assert_eq!(q2.len(), 4);
}

// -- T3: AllGoalsUpdate td_target + loss --

#[allow(dead_code)]
struct Updater;
impl AllGoalsUpdate for Updater {}

#[test]
fn test_all_goals_td_target() {
    let upd = Updater;
    let rewards = vec![1.0, 0.0, 0.5]; // 3 goals
    let next_q = vec![
        vec![0.1, 0.2], // goal 0: max = 0.2
        vec![0.3, 0.5], // goal 1: max = 0.5
        vec![0.0, 0.1], // goal 2: max = 0.1
    ];
    let gamma = 0.99;
    let targets = upd.td_target(&rewards, &next_q, gamma);
    assert_eq!(targets.len(), 3);
    assert!((targets[0] - (1.0 + 0.99 * 0.2)).abs() < 1e-5);
    assert!((targets[1] - (0.0 + 0.99 * 0.5)).abs() < 1e-5);
    assert!((targets[2] - (0.5 + 0.99 * 0.1)).abs() < 1e-5);
}

#[test]
fn test_all_goals_loss() {
    let predicted = vec![vec![0.8], vec![0.2], vec![0.5]];
    let target = vec![1.0, 0.0, 0.5];
    let loss = <Updater as AllGoalsUpdate>::loss(&predicted, &target);
    // (0.8-1.0)² = 0.04, (0.2-0.0)² = 0.04, (0.5-0.5)² = 0.0
    // MSE = (0.04 + 0.04 + 0.0) / 2 / 3 = 0.01333...
    assert!((loss - 0.5 * (0.04 + 0.04 + 0.0) / 3.0).abs() < 1e-6);
}

// -- T2: DualLeoMixer --

#[allow(dead_code)]
struct Mixer;
impl DualLeoMixer for Mixer {}

#[test]
fn test_dual_leo_mix() {
    let mixer = Mixer;
    let q_leo = vec![0.4, 0.6, 0.2];
    let q_uvfa = vec![0.1, 0.9, 0.3];
    let alpha = 0.3;
    let mixed = mixer.mix(&q_leo, &q_uvfa, alpha);
    // 0.3*0.4 + 0.7*0.1 = 0.19
    assert!((mixed[0] - 0.19).abs() < 1e-6);
    // 0.3*0.6 + 0.7*0.9 = 0.81
    assert!((mixed[1] - 0.81).abs() < 1e-6);
    // 0.3*0.2 + 0.7*0.3 = 0.27
    assert!((mixed[2] - 0.27).abs() < 1e-6);
}

#[test]
fn test_dual_leo_default_alpha() {
    let mixer = Mixer;
    assert!((mixer.default_alpha() - 0.3).abs() < 1e-6);
}

#[test]
fn test_acting_mode_default() {
    assert_eq!(Mixer.acting_mode(), ActingMode::Lc);
}

#[test]
fn test_acting_mode_combine_lc() {
    let mixer = Mixer;
    let q_leo = vec![0.4, 0.6];
    let q_uvfa = vec![0.1, 0.9];
    let combined = mixer.combine(&q_leo, &q_uvfa, 0.3);
    // Same as mix: 0.3*0.4 + 0.7*0.1 = 0.19, 0.3*0.6 + 0.7*0.9 = 0.81
    assert!((combined[0] - 0.19).abs() < 1e-6);
    assert!((combined[1] - 0.81).abs() < 1e-6);
}

#[test]
fn test_alpha_schedule_fixed() {
    assert!(matches!(Mixer.alpha_schedule(), AlphaSchedule::Fixed(0.3)));
    assert!((Mixer.alpha_at_progress(0.0) - 0.3).abs() < 1e-6);
    assert!((Mixer.alpha_at_progress(0.5) - 0.3).abs() < 1e-6);
    assert!((Mixer.alpha_at_progress(1.0) - 0.3).abs() < 1e-6);
}

#[test]
fn test_alpha_schedule_linear_anneal() {
    struct AnnealingMixer;
    impl DualLeoMixer for AnnealingMixer {
        fn alpha_schedule(&self) -> AlphaSchedule {
            AlphaSchedule::LinearAnneal {
                start: 1.0,
                end: 0.0,
            }
        }
    }
    let m = AnnealingMixer;
    assert!((m.alpha_at_progress(0.0) - 1.0).abs() < 1e-6);
    assert!((m.alpha_at_progress(0.5) - 0.5).abs() < 1e-6);
    assert!((m.alpha_at_progress(1.0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_bc_config_default() {
    assert!(Mixer.bc_config().is_none());
}

#[test]
fn test_bc_config_values() {
    let bc = BcConfig::default();
    assert!((bc.policy_coef - 0.1).abs() < 1e-6);
    assert!((bc.value_coef - 0.0).abs() < 1e-6);
    assert_eq!(bc.target, BcTarget::Argmax);
}

// -- T4: AutocurriculumSampler --

#[allow(dead_code)]
struct SimpleAutocurriculum {
    observed: Vec<bool>,
}

impl SimpleAutocurriculum {
    #[allow(dead_code)]
    fn new(total: usize) -> Self {
        Self {
            observed: vec![false; total],
        }
    }
}

impl AutocurriculumSampler for SimpleAutocurriculum {
    fn sample_goal(&self, rng: &mut Rng) -> usize {
        let observed: Vec<_> = self
            .observed
            .iter()
            .enumerate()
            .filter(|&(_, &o)| o)
            .map(|(i, _)| i)
            .collect();
        observed[rng.usize(0..observed.len())]
    }

    fn observe_goal(&mut self, goal: usize) {
        if goal < self.observed.len() {
            self.observed[goal] = true;
        }
    }

    fn observed_count(&self) -> usize {
        self.observed.iter().filter(|&&o| o).count()
    }

    fn total_goal_count(&self) -> usize {
        self.observed.len()
    }
}

#[test]
fn test_autocurriculum_observe_and_count() {
    let mut ac = SimpleAutocurriculum::new(5);
    assert_eq!(ac.observed_count(), 0);
    assert_eq!(ac.total_goal_count(), 5);

    ac.observe_goal(2);
    ac.observe_goal(4);
    assert_eq!(ac.observed_count(), 2);

    // Duplicate observe doesn't change count
    ac.observe_goal(2);
    assert_eq!(ac.observed_count(), 2);
}

#[test]
fn test_autocurriculum_sample_from_observed() {
    let mut ac = SimpleAutocurriculum::new(10);
    ac.observe_goal(3);
    ac.observe_goal(7);
    ac.observe_goal(9);

    let mut rng = Rng::new();
    // Sample many times — should only get 3, 7, or 9
    for _ in 0..100 {
        let g = ac.sample_goal(&mut rng);
        assert!(g == 3 || g == 7 || g == 9, "sampled unobserved goal: {g}");
    }
}

// -- T9d: Q(λ) tests --

#[test]
fn test_td_target_lambda_no_done() {
    let upd = Updater;
    let rewards = vec![1.0, 0.0];
    let next_q_max = vec![0.5, 0.3];
    let next_lambda_return = vec![0.0, 0.0]; // last step, no future λ-return
    let done = vec![false, false];
    // lambda=0: standard TD
    let targets =
        upd.td_target_lambda(&rewards, &next_q_max, &next_lambda_return, &done, 0.99, 0.0);
    assert!((targets[0] - (1.0 + 0.99 * 0.5)).abs() < 1e-5);
    assert!((targets[1] - (0.0 + 0.99 * 0.3)).abs() < 1e-5);
}

#[test]
fn test_td_target_lambda_with_done() {
    let upd = Updater;
    let rewards = vec![1.0, 0.5];
    let next_q_max = vec![0.5, 0.3];
    let next_lambda_return = vec![0.0, 0.0];
    let done = vec![true, false];
    let targets =
        upd.td_target_lambda(&rewards, &next_q_max, &next_lambda_return, &done, 0.99, 0.5);
    // done[0] = true → target = reward = 1.0
    assert!((targets[0] - 1.0).abs() < 1e-5);
    // done[1] = false, lambda=0.5 → r + γ*(0.5*0.0 + 0.5*0.3) = 0.5 + 0.99*0.15
    assert!((targets[1] - (0.5 + 0.99 * 0.15)).abs() < 1e-5);
}

#[test]
fn test_td_target_lambda_with_future_return() {
    let upd = Updater;
    let rewards = vec![0.0];
    let next_q_max = vec![0.2];
    let next_lambda_return = vec![1.0]; // future λ-return accumulated
    let done = vec![false];
    // lambda=1.0: pure MC → r + γ * 1.0 * g_next = 0 + 0.99 * 1.0
    let targets_mc =
        upd.td_target_lambda(&rewards, &next_q_max, &next_lambda_return, &done, 0.99, 1.0);
    assert!((targets_mc[0] - 0.99).abs() < 1e-5);
    // lambda=0.0: one-step TD → r + γ * q_max = 0 + 0.99 * 0.2
    let targets_td =
        upd.td_target_lambda(&rewards, &next_q_max, &next_lambda_return, &done, 0.99, 0.0);
    assert!((targets_td[0] - (0.99 * 0.2)).abs() < 1e-5);
}

// -- T9e: AutocurriculumSampler refinements --

#[test]
fn test_update_goals_seen() {
    let ac = SimpleAutocurriculum::new(3);
    let obs_batch = vec![vec![1.0, 0.0, 0.0]];
    let all_goals = vec![
        vec![1.0, 0.0, 0.0], // matches obs
        vec![0.0, 1.0, 0.0], // no match
        vec![0.0, 0.0, 1.0], // no match
    ];
    let current_mask = vec![false; 3];
    let updated = ac.update_goals_seen(&obs_batch, &all_goals, &current_mask);
    assert!(updated[0], "goal 0 should be seen");
    assert!(!updated[1], "goal 1 should not be seen");
    assert!(!updated[2], "goal 2 should not be seen");
}

#[test]
fn test_update_goals_seen_union() {
    let ac = SimpleAutocurriculum::new(3);
    let obs_batch = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let all_goals = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let current_mask = vec![false; 3];
    let updated = ac.update_goals_seen(&obs_batch, &all_goals, &current_mask);
    assert!(updated[0], "goal 0 should be seen");
    assert!(updated[1], "goal 1 should be seen");
    assert!(!updated[2], "goal 2 should not be seen");
}

#[test]
fn test_autocurriculum_default_methods() {
    let ac = SimpleAutocurriculum::new(5);
    assert_eq!(ac.goals_completed_this_episode(), 0);
    assert!(ac.only_sample_from_seen());
}
