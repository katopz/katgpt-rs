//! PWC Bandit Arms — piecewise-constant value function over BFCP regions (Plan 213 P3).
//!
//! Each bandit arm maintains a PWC value function: constant per region, varying across regions.
//! Theorem 2 (NS-CSG): after Bellman backup (update), values remain piecewise-constant.
//! This is the B-PWC closure guarantee — no value leakage between regions.

use super::bfcf_types::PWCValueFunction;

// ── sigmoid helper ──────────────────────────────────────────────

#[cfg(test)]
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ── RegionBandit ────────────────────────────────────────────────

/// Bandit that maintains per-region arm values with PWC guarantee.
///
/// Each arm has a piecewise-constant value function over regions.
/// Arm selection uses UCB1-style exploration bonus: `Q(a,r) + c * sqrt(ln(t) / n(a,r))`.
pub struct RegionBandit {
    /// Value functions per arm.
    arms: Vec<PWCValueFunction>,
    /// Visit counts per (arm, region).
    visits: Vec<Vec<u64>>,
    /// Total pulls per arm.
    arm_pulls: Vec<u64>,
    /// Number of arms.
    arm_count: usize,
    /// Number of regions.
    region_count: usize,
    /// Exploration constant (default: sqrt(2)).
    c: f64,
}

impl RegionBandit {
    /// Create a new RegionBandit with `arm_count` arms and `region_count` regions.
    pub fn new(arm_count: usize, region_count: usize, c: f64) -> Self {
        Self {
            arms: (0..arm_count)
                .map(|_| PWCValueFunction::new(region_count, 0.0))
                .collect(),
            visits: (0..arm_count).map(|_| vec![0u64; region_count]).collect(),
            arm_pulls: vec![0u64; arm_count],
            arm_count,
            region_count,
            c,
        }
    }

    /// Select the best arm for a given region using UCB1.
    ///
    /// Unvisited arms get priority (infinite upper bound). Among visited arms,
    /// selects `argmax(Q(a,r) + c * sqrt(ln(total_pulls) / n(a,r)))`.
    pub fn select(&self, region_idx: usize) -> usize {
        let total_pulls: u64 = self.arm_pulls.iter().sum();
        let ln_total = if total_pulls > 0 {
            (total_pulls as f64).ln()
        } else {
            0.0
        };

        let mut best_arm = 0;
        let mut best_score = f64::NEG_INFINITY;

        for arm in 0..self.arm_count {
            let visits = self.visits[arm][region_idx];
            let score = match visits {
                0 => f64::INFINITY, // Unvisited arms get priority
                n => {
                    let q = self.arms[arm].value(region_idx);
                    let exploration = self.c * (ln_total / n as f64).sqrt();
                    q + exploration
                }
            };

            if score > best_score {
                best_score = score;
                best_arm = arm;
            }
        }

        best_arm
    }

    /// Update arm value for a specific region using incremental mean.
    ///
    /// Maintains PWC closure: only the target region's value changes.
    pub fn update(&mut self, region_idx: usize, arm: usize, reward: f64) {
        if arm >= self.arm_count || region_idx >= self.region_count {
            return;
        }

        let visits = self.visits[arm][region_idx];
        let old_q = self.arms[arm].value(region_idx);

        // Incremental mean update
        let new_q = match visits {
            0 => reward,
            n => old_q + (reward - old_q) / (n as f64 + 1.0),
        };

        self.arms[arm].update(region_idx, new_q);
        self.visits[arm][region_idx] += 1;
        self.arm_pulls[arm] += 1;
    }

    /// Get the Q-value for a specific (arm, region) pair.
    pub fn q_value(&self, arm: usize, region_idx: usize) -> f64 {
        if arm >= self.arm_count || region_idx >= self.region_count {
            return 0.0;
        }
        self.arms[arm].value(region_idx)
    }

    /// Get visit count for a specific (arm, region) pair.
    pub fn visits(&self, arm: usize, region_idx: usize) -> u64 {
        if arm >= self.arm_count || region_idx >= self.region_count {
            return 0;
        }
        self.visits[arm][region_idx]
    }

    /// Total pulls across all arms.
    pub fn total_pulls(&self) -> u64 {
        self.arm_pulls.iter().sum()
    }

    /// Number of arms.
    pub fn arm_count(&self) -> usize {
        self.arm_count
    }

    /// Number of regions.
    pub fn region_count(&self) -> usize {
        self.region_count
    }

    /// Verify PWC closure across all arms.
    ///
    /// After any number of updates, each arm's value function must maintain
    /// piecewise-constant structure: one value per region, no duplicates.
    /// This is Theorem 2 (NS-CSG B-PWC closure).
    pub fn verify_pwc_closure(&self) -> bool {
        self.arms.iter().all(|arm| arm.verify_pwc_closure())
    }

    /// Compute UCB1 score for a specific (arm, region) pair.
    pub fn ucb1_score(&self, arm: usize, region_idx: usize) -> f64 {
        if arm >= self.arm_count || region_idx >= self.region_count {
            return f64::NEG_INFINITY;
        }
        let total = self.total_pulls();
        if total == 0 {
            return f64::INFINITY;
        }
        let visits = self.visits[arm][region_idx];
        if visits == 0 {
            return f64::INFINITY;
        }
        let q = self.arms[arm].value(region_idx);
        q + self.c * (total as f64).ln() / (visits as f64).sqrt()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pwc_value_function_get_update() {
        let mut vf = PWCValueFunction::new(5, 0.0);
        assert_eq!(vf.value(0), 0.0);
        assert_eq!(vf.value(4), 0.0);

        vf.update(2, 1.5);
        assert_eq!(vf.value(2), 1.5);
        assert_eq!(vf.value(0), 0.0);

        // Out of bounds returns 0.0
        assert_eq!(vf.value(99), 0.0);
    }

    #[test]
    fn test_pwc_closure_maintained() {
        let mut vf = PWCValueFunction::new(10, 0.5);
        assert!(vf.verify_pwc_closure());

        // Multiple updates should not break PWC structure
        for i in 0..10 {
            vf.update(i, (i as f64) * 0.1);
        }
        assert!(vf.verify_pwc_closure());

        // Repeated updates to same region
        vf.update(5, 0.99);
        vf.update(5, 0.88);
        vf.update(5, 0.77);
        assert!(vf.verify_pwc_closure());
    }

    #[test]
    fn test_region_bandit_select_best_arm() {
        let mut bandit = RegionBandit::new(3, 5, 2.0_f64.sqrt());

        // Warm up: give arm 1 high value in region 0
        for _ in 0..10 {
            bandit.update(0, 1, 1.0); // arm 1, region 0, reward 1.0
        }
        for _ in 0..10 {
            bandit.update(0, 0, 0.1); // arm 0, region 0, reward 0.1
        }
        for _ in 0..10 {
            bandit.update(0, 2, 0.5); // arm 2, region 0, reward 0.5
        }

        // After warm-up with equal visits, arm 1 should be selected (highest Q)
        let selected = bandit.select(0);
        assert_eq!(
            selected, 1,
            "arm 1 should be selected for region 0 (highest Q)"
        );
    }

    #[test]
    fn test_region_bandit_unvisited_arm_priority() {
        let bandit = RegionBandit::new(3, 5, 2.0_f64.sqrt());

        // All arms unvisited for region 0 → select arm 0 (first with infinite score)
        let selected = bandit.select(0);
        assert_eq!(selected, 0, "first unvisited arm should be selected");
    }

    #[test]
    fn test_region_bandit_update_changes_value() {
        let mut bandit = RegionBandit::new(2, 3, 2.0_f64.sqrt());

        assert_eq!(bandit.q_value(0, 0), 0.0);

        bandit.update(0, 0, 0.8);
        assert!(
            (bandit.q_value(0, 0) - 0.8).abs() < 0.001,
            "Q-value should be 0.8 after single update"
        );

        bandit.update(0, 0, 0.4);
        let expected = 0.8 + (0.4 - 0.8) / 2.0; // incremental mean
        assert!(
            (bandit.q_value(0, 0) - expected).abs() < 0.001,
            "Q-value should be incremental mean: expected={}",
            expected
        );

        // Different region should be unaffected
        assert_eq!(bandit.q_value(0, 1), 0.0);
    }

    #[test]
    fn test_region_bandit_pwc_closure_after_n_updates() {
        let mut bandit = RegionBandit::new(4, 8, 2.0_f64.sqrt());

        // Verify closure before any updates
        assert!(bandit.verify_pwc_closure());

        // Apply 100 random-ish updates
        for round in 0..100 {
            let arm = round % 4;
            let region = round % 8;
            let reward = sigmoid((round as f64) * 0.1 - 5.0);
            bandit.update(region, arm, reward);
        }

        // Theorem 2: PWC closure maintained after N updates
        assert!(
            bandit.verify_pwc_closure(),
            "PWC closure must hold after 100 updates (Theorem 2)"
        );
    }

    #[test]
    fn test_region_bandit_region_isolation() {
        let mut bandit = RegionBandit::new(2, 5, 2.0_f64.sqrt());

        // Update arm 0 for region 0 only
        bandit.update(0, 0, 1.0);
        bandit.update(0, 0, 1.0);
        bandit.update(0, 0, 1.0);

        // Other regions should be unaffected
        for r in 1..5 {
            assert_eq!(
                bandit.q_value(0, r),
                0.0,
                "region {} should be unaffected",
                r
            );
        }
    }

    #[test]
    fn test_region_bandit_visit_tracking() {
        let mut bandit = RegionBandit::new(2, 3, 2.0_f64.sqrt());

        assert_eq!(bandit.visits(0, 0), 0);
        assert_eq!(bandit.total_pulls(), 0);

        bandit.update(0, 0, 1.0);
        assert_eq!(bandit.visits(0, 0), 1);
        assert_eq!(bandit.total_pulls(), 1);

        bandit.update(1, 0, 0.5); // region=1, arm=0
        assert_eq!(bandit.visits(0, 1), 1); // arm=0, region=1
        assert_eq!(bandit.total_pulls(), 2);

        // Out of bounds
        assert_eq!(bandit.visits(99, 0), 0);
    }

    #[test]
    fn test_region_bandit_multiple_regions_different_arms() {
        let mut bandit = RegionBandit::new(3, 4, 2.0_f64.sqrt());

        // Arm 0 is best for region 0
        bandit.update(0, 0, 1.0);
        // Arm 1 is best for region 1
        bandit.update(1, 1, 1.0);
        // Arm 2 is best for region 2
        bandit.update(2, 2, 1.0);

        // Give enough visits so UCB1 exploration bonus is small
        for _ in 0..20 {
            bandit.update(0, 0, 1.0);
            bandit.update(1, 1, 1.0);
            bandit.update(2, 2, 1.0);
            // Other arms get low rewards
            bandit.update(0, 1, 0.0);
            bandit.update(0, 2, 0.0);
            bandit.update(1, 0, 0.0);
            bandit.update(1, 2, 0.0);
            bandit.update(2, 0, 0.0);
            bandit.update(2, 1, 0.0);
        }

        // Now best arm for each region should be clear
        assert_eq!(bandit.select(0), 0, "arm 0 should be best for region 0");
        assert_eq!(bandit.select(1), 1, "arm 1 should be best for region 1");
        assert_eq!(bandit.select(2), 2, "arm 2 should be best for region 2");
    }

    #[test]
    fn test_region_bandit_out_of_bounds_safety() {
        let bandit = RegionBandit::new(2, 3, 2.0_f64.sqrt());

        // Out of bounds access should return safe defaults
        assert_eq!(bandit.q_value(99, 0), 0.0);
        assert_eq!(bandit.q_value(0, 99), 0.0);
        assert_eq!(bandit.visits(99, 0), 0);
    }
}
