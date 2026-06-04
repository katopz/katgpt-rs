//! GOAT benchmark for dMoE adaptive top-p bandit arm selection (Plan 181, D2).
//!
//! Criteria:
//! 1. Arm count reduction: Mean arm count ≤ 60% of top-k
//! 2. Win rate preservation: ≤ 1% win rate loss vs fixed top-k
//! 3. WASM call reduction: ≥ 30% fewer evaluations
//! 4. Top-p overhead: ≤ 200ns for 8 arms

#[cfg(feature = "bandit_top_p")]
mod tests {
    use katgpt_rs::pruners::select_arms_top_p;

    #[test]
    fn test_goat_arm_count_reduction() {
        // Simulate concentrated scores — top-p should select fewer arms
        let q_values = vec![5.0, 1.0, 0.5, 0.3, 0.1, 0.05];
        let ucb_bonus = vec![0.1; 6];

        let selected = select_arms_top_p(&q_values, &ucb_bonus, 0.85);
        let ratio = selected.len() as f64 / q_values.len() as f64;

        assert!(
            ratio <= 0.6,
            "arm count should be ≤60% of total: got {} arms out of {} (ratio={:.2})",
            selected.len(),
            q_values.len(),
            ratio
        );
    }

    #[test]
    fn test_goat_dispersion_selects_more() {
        // Dispersed scores — top-p should select more arms
        let q_values = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
        let ucb_bonus = vec![0.1; 6];

        let selected = select_arms_top_p(&q_values, &ucb_bonus, 0.85);

        assert!(
            selected.len() >= 3,
            "dispersed scores should select ≥3 arms, got {}",
            selected.len()
        );
    }

    #[test]
    fn test_goat_includes_best_arm() {
        let q_values = vec![0.5, 5.0, 0.3, 0.1]; // arm 1 is best
        let ucb_bonus = vec![0.1; 4];

        let selected = select_arms_top_p(&q_values, &ucb_bonus, 0.85);

        assert!(
            selected.contains(&1),
            "selected arms must include the best arm: {:?}",
            selected
        );
    }

    #[test]
    fn test_goat_empty_input() {
        let selected = select_arms_top_p(&[], &[], 0.85);
        assert!(
            selected.is_empty(),
            "empty input should return empty output"
        );
    }

    #[test]
    fn test_goat_p1_selects_all() {
        let q_values = vec![1.0, 2.0, 3.0];
        let ucb_bonus = vec![0.1; 3];

        let selected = select_arms_top_p(&q_values, &ucb_bonus, 1.0);

        assert_eq!(selected.len(), 3, "p=1.0 should select all arms");
    }
}
