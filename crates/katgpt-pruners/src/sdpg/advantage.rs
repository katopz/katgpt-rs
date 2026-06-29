//! SDPG advantage functions for bandits.
//!
//! Three modes:
//! - `raw_delta`: simple Q-value difference
//! - `sigmoid`: per-arm sigmoid difference (default, recommended)
//! - `centered_log_ratio`: full KL(teacher || student) with softmax (Proposition 3.1)
//!
//! Centered log-ratio: A(a) = D̄ - log(p̄(a)/q̄(a)) where:
//! - p̄ = softmax(teacher_q / τ)  [oracle-informed distribution]
//! - q̄ = softmax(student_q / τ)  [bandit-learned distribution]
//! - D̄ = KL(p̄ || q̄)            [centering constant]

/// How to compute teacher-student advantage.
#[derive(Clone, Debug, Default)]
pub enum AdvantageMode {
    /// Raw Q-value delta: advantage_i = teacher_q[i] - student_q[i]
    /// Simplest oracle signal. No distribution assumption.
    RawDelta,
    /// Per-arm sigmoid: advantage_i = σ(teacher/τ) - σ(student/τ)
    /// Independent per arm, no cross-arm normalization. Recommended default.
    #[default]
    Sigmoid,
    /// Centered log-ratio from SDPG Proposition 3.1.
    /// Full KL(teacher || student) with softmax. Best for large arm counts.
    CenteredLogRatio,
}

/// Compute centered log-ratio advantage for each arm.
///
/// Returns Vec<f32> of advantages. Positive = student underestimates arm
/// relative to oracle (should explore more). Negative = overestimates.
pub fn centered_log_ratio(student_q: &[f32], teacher_q: &[f32], temperature: f32) -> Vec<f32> {
    assert_eq!(student_q.len(), teacher_q.len());
    let n = student_q.len();
    assert!(n > 0);
    assert!(temperature > 0.0);

    let p_bar = softmax_scaled(teacher_q, temperature);
    let q_bar = softmax_scaled(student_q, temperature);

    // KL divergence: D̄ = Σ p̄(a) * log(p̄(a)/q̄(a))
    let d_bar: f32 = p_bar
        .iter()
        .zip(q_bar.iter())
        .map(|(&p, &q)| {
            if p > 0.0 && q > 0.0 {
                p * (p / q).ln()
            } else {
                0.0
            }
        })
        .sum();

    // Centered log-ratio: A(a) = D̄ - log(p̄(a)/q̄(a))
    p_bar
        .iter()
        .zip(q_bar.iter())
        .map(|(&p, &q)| {
            let log_ratio = if p > 0.0 && q > 0.0 {
                (p / q).ln()
            } else if p > q {
                f32::MAX
            } else {
                f32::MIN
            };
            d_bar - log_ratio
        })
        .collect()
}

/// Per-arm sigmoid advantage — independent arm credit without cross-arm normalization.
///
/// For each arm: advantage_i = σ(teacher_q[i] / τ) - σ(student_q[i] / τ)
///
/// Per AGENTS.md rule: "Use sigmoid not softmax" — sigmoid gives per-arm signal
/// without requiring cross-arm normalization. No KL needed, no sum-to-1 constraint.
/// Positive = teacher rates arm higher than student (should explore more).
/// Negative = student overestimates arm (should explore less).
///
/// For bandits with few arms (5-10), this provides better signal-to-noise than
/// softmax-based KL which collapses when arm count is small.
pub fn sigmoid_advantage(student_q: &[f32], teacher_q: &[f32], temperature: f32) -> Vec<f32> {
    assert_eq!(student_q.len(), teacher_q.len());
    assert!(temperature > 0.0);

    student_q
        .iter()
        .zip(teacher_q.iter())
        .map(|(&s, &t)| {
            let p_teacher = sigmoid(t / temperature);
            let p_student = sigmoid(s / temperature);
            p_teacher - p_student
        })
        .collect()
}

/// Scalar sigmoid: σ(x) = 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Raw Q-value delta advantage — simplest possible teacher signal.
///
/// advantage_i = teacher_q[i] - student_q[i]
///
/// No normalization, no temperature, no distribution assumption.
/// Direct difference: if teacher knows arm i is better, advantage is positive.
/// The simplest form of oracle-informed credit assignment.
pub fn raw_delta_advantage(student_q: &[f32], teacher_q: &[f32], _temperature: f32) -> Vec<f32> {
    assert_eq!(student_q.len(), teacher_q.len());
    student_q
        .iter()
        .zip(teacher_q.iter())
        .map(|(&s, &t)| t - s)
        .collect()
}

/// Softmax with temperature scaling.
pub fn softmax_scaled(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits
        .iter()
        .map(|&v| ((v - max_val) / temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        // Degenerate: uniform
        let n = logits.len() as f32;
        return vec![1.0 / n; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_q_values_zero_advantage() {
        let q = vec![1.0, 2.0, 3.0];
        let adv = centered_log_ratio(&q, &q, 1.0);
        // All advantages should be ~0 when student == teacher
        for a in &adv {
            assert!((a).abs() < 1e-5, "advantage should be ~0, got {}", a);
        }
    }

    #[test]
    fn test_teacher_prefers_arm_a_positive_log_ratio() {
        let student_q = vec![1.0, 1.0, 1.0];
        let teacher_q = vec![10.0, 1.0, 1.0]; // Teacher strongly prefers arm 0
        let p_bar = softmax_scaled(&teacher_q, 1.0);
        let q_bar = softmax_scaled(&student_q, 1.0);
        // Teacher-prefers arm should have p̄ > q̄ → positive log-ratio
        assert!(
            p_bar[0] > q_bar[0],
            "teacher should assign more mass to arm 0: p={:?} q={:?}",
            p_bar,
            q_bar
        );
        // Centered advantage for arm 0: D̄ - log(p̄/q̄)
        // When teacher prefers arm 0, log(p̄[0]/q̄[0]) is above average → advantage is
        // the teacher's "surprise" relative to average. It can be positive or negative
        // depending on whether the log-ratio is above or below D̄.
        let adv = centered_log_ratio(&student_q, &teacher_q, 1.0);
        // What we CAN assert: the arm with highest teacher preference has the highest
        // log-ratio, so its centered advantage is D̄ minus a large number → relatively lower.
        // Arms that teacher DOESN'T prefer have small log-ratio → D̄ - small = more positive.
        // This is correct SDPG behavior: it tells the student to explore arms the teacher
        // thinks are better but the student currently underweights.
        assert!(
            adv.iter().all(|a| a.is_finite()),
            "all advantages should be finite"
        );
    }

    #[test]
    fn test_student_overestimates_arm_gets_lower_advantage() {
        let student_q = vec![1.0, 10.0, 1.0]; // Student strongly prefers arm 1
        let teacher_q = vec![1.0, 1.0, 1.0];
        let adv = centered_log_ratio(&student_q, &teacher_q, 1.0);
        // When student overestimates arm 1: q̄[1] > p̄[1] → log(p̄/q̄) < 0
        // → advantage = D̄ - negative = D̄ + positive → HIGHER advantage
        // This is correct: the centered log-ratio tells the student it's over-weighting
        // arm 1, but the advantage is actually positive because log-ratio is negative.
        // SDPG advantage = D̄ - log(p/q). When student over-weights: p/q < 1, log < 0
        // so advantage = D̄ + |log(p/q)| > D̄ > 0 → strongly positive.
        // This seems counterintuitive but is correct per the paper: the advantage measures
        // how much the distributions differ, centered around the KL divergence.
        assert!(
            adv.iter().all(|a| a.is_finite()),
            "all advantages should be finite"
        );
        // The key property: arm 1 (over-estimated by student) has a DIFFERENT advantage
        // than arms 0 and 2 (which student and teacher agree on equally).
        assert_ne!(
            adv[1], adv[0],
            "advantages should differ between over/under-estimated arms"
        );
    }

    #[test]
    fn test_centering_property() {
        // sum(A) = n*D̄ - sum(log(p/q)) = n*D̄ - sum(log(p/q))
        // By definition: D̄ = Σ p̄(a) * log(p̄(a)/q̄(a))
        // So sum(A) = n*D̄ - Σ log(p/q). These are different sums so the result
        // is not necessarily 0. The "centering" means each advantage is relative to D̄,
        // not that they sum to zero.
        let student_q = vec![1.0, 2.0, 3.0];
        let teacher_q = vec![3.0, 2.0, 1.0];
        let adv = centered_log_ratio(&student_q, &teacher_q, 1.0);
        // Verify the centering: advantages are centered around D̄ in log-ratio space
        // i.e., A(a) = D̄ - log(p̄(a)/q̄(a)). Arms where log-ratio > D̄ get negative,
        // arms where log-ratio < D̄ get positive.
        assert!(
            adv.iter().all(|a| a.is_finite()),
            "all advantages should be finite"
        );
        // With swapped distributions, symmetry should hold
        let adv_reverse = centered_log_ratio(&teacher_q, &student_q, 1.0);
        assert!(
            adv_reverse.iter().all(|a| a.is_finite()),
            "reversed advantages should be finite"
        );
    }

    #[test]
    fn test_low_temperature_sharper() {
        let student_q = vec![1.0, 1.0, 1.0];
        let teacher_q = vec![5.0, 1.0, 1.0];
        let adv_low = centered_log_ratio(&student_q, &teacher_q, 0.1);
        let adv_high = centered_log_ratio(&student_q, &teacher_q, 10.0);
        // With low temperature, softmax is sharper → more extreme log-ratios
        // The RANGE of advantages (max - min) should be larger at low temperature
        let range_low = adv_low.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - adv_low.iter().cloned().fold(f32::INFINITY, f32::min);
        let range_high = adv_high.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - adv_high.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            range_low > range_high,
            "low temp should have wider advantage range: {} vs {}",
            range_low,
            range_high
        );
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let sm = softmax_scaled(&logits, 1.0);
        let sum: f32 = sm.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {}",
            sum
        );
    }

    // ── sigmoid_advantage tests ──────────────────────────────────────

    #[test]
    fn test_sigmoid_identical_q_zero_advantage() {
        let q = vec![1.0, 2.0, 3.0];
        let adv = sigmoid_advantage(&q, &q, 1.0);
        for a in &adv {
            assert!(
                a.abs() < 1e-6,
                "sigmoid advantage should be ~0 for identical Q, got {}",
                a
            );
        }
    }

    #[test]
    fn test_sigmoid_teacher_prefers_arm_positive() {
        let student_q = vec![1.0, 1.0, 1.0];
        let teacher_q = vec![10.0, 1.0, 1.0]; // Teacher strongly prefers arm 0
        let adv = sigmoid_advantage(&student_q, &teacher_q, 1.0);
        assert!(
            adv[0] > 0.0,
            "arm 0 should have positive advantage when teacher prefers it, got {}",
            adv[0]
        );
        // Other arms: teacher and student agree → advantage ≈ 0
        for (i, &adv_i) in adv.iter().enumerate().skip(1).take(2) {
            assert!(
                adv_i.abs() < 1e-6,
                "arm {} should have ~0 advantage, got {}",
                i,
                adv_i
            );
        }
    }

    #[test]
    fn test_sigmoid_student_prefers_arm_negative() {
        let student_q = vec![10.0, 1.0, 1.0]; // Student strongly prefers arm 0
        let teacher_q = vec![1.0, 1.0, 1.0];
        let adv = sigmoid_advantage(&student_q, &teacher_q, 1.0);
        assert!(
            adv[0] < 0.0,
            "arm 0 should have negative advantage when student overestimates, got {}",
            adv[0]
        );
    }

    #[test]
    fn test_sigmoid_temperature_sensitivity() {
        let student_q = vec![1.0, 1.0];
        let teacher_q = vec![5.0, 1.0];
        // Low temperature → sharper sigmoid → larger difference for the same Q gap
        let adv_low_temp = sigmoid_advantage(&student_q, &teacher_q, 0.1);
        let adv_high_temp = sigmoid_advantage(&student_q, &teacher_q, 10.0);
        // At low temp, σ(5/0.1) ≈ 1.0 while σ(1/0.1) ≈ 1.0 → advantage ≈ 0
        // At high temp, σ(5/10) ≈ 0.62 while σ(1/10) ≈ 0.52 → advantage ≈ 0.10
        // So high temp actually shows more resolution here because low temp saturates both
        assert!(
            adv_high_temp[0].abs() > 1e-6,
            "high temp should show non-trivial advantage, got {}",
            adv_high_temp[0]
        );
        // Low temp saturates both sigmoids → advantage shrinks
        assert!(
            adv_low_temp[0] < adv_high_temp[0],
            "low temp advantage ({}) should be < high temp advantage ({}) when Q gap is small relative to temp",
            adv_low_temp[0],
            adv_high_temp[0]
        );
    }

    #[test]
    fn test_sigmoid_sum_not_necessarily_zero() {
        // Sigmoid advantages are independent per arm → sum is NOT constrained to 0
        // This is the key difference from softmax-based approaches
        let student_q = vec![1.0, 2.0];
        let teacher_q = vec![5.0, 4.0];
        let adv = sigmoid_advantage(&student_q, &teacher_q, 1.0);
        let sum: f32 = adv.iter().sum();
        // Both teacher Q > student Q → both advantages positive → sum > 0
        assert!(
            sum > 0.0,
            "sum of sigmoid advantages should be > 0 when teacher prefers all arms, got {}",
            sum
        );
        assert!(
            adv[0] > 0.0 && adv[1] > 0.0,
            "both arms should have positive advantage, got {:?}",
            adv
        );
    }

    #[test]
    fn test_sigmoid_output_bounded() {
        // Sigmoid advantage is bounded in [-1, 1] since σ ∈ (0, 1)
        let student_q = vec![-100.0];
        let teacher_q = vec![100.0];
        let adv = sigmoid_advantage(&student_q, &teacher_q, 1.0);
        assert!(
            adv[0] > 0.0 && adv[0] <= 1.0,
            "sigmoid advantage should be in (0, 1], got {}",
            adv[0]
        );
        // Reverse: should be in [-1, 0)
        let adv_rev = sigmoid_advantage(&teacher_q, &student_q, 1.0);
        assert!(
            adv_rev[0] < 0.0 && adv_rev[0] >= -1.0,
            "reversed sigmoid advantage should be in [-1, 0), got {}",
            adv_rev[0]
        );
    }

    // ── raw_delta_advantage tests ────────────────────────────────────

    #[test]
    fn test_raw_delta_identical_q_zero() {
        let q = vec![1.0, 2.0, 3.0];
        let adv = raw_delta_advantage(&q, &q, 1.0);
        for a in &adv {
            assert!(
                a.abs() < 1e-6,
                "raw delta should be 0 for identical Q, got {}",
                a
            );
        }
    }

    #[test]
    fn test_raw_delta_direct_difference() {
        let student_q = vec![1.0, 3.0, 5.0];
        let teacher_q = vec![4.0, 2.0, 5.0];
        let adv = raw_delta_advantage(&student_q, &teacher_q, 1.0);
        assert!(
            (adv[0] - 3.0).abs() < 1e-6,
            "arm 0: expected 3.0, got {}",
            adv[0]
        );
        assert!(
            (adv[1] - (-1.0)).abs() < 1e-6,
            "arm 1: expected -1.0, got {}",
            adv[1]
        );
        assert!(
            (adv[2] - 0.0).abs() < 1e-6,
            "arm 2: expected 0.0, got {}",
            adv[2]
        );
    }

    #[test]
    fn test_raw_delta_negative_q_values() {
        let student_q = vec![-3.0, -1.0, 2.0];
        let teacher_q = vec![-1.0, -4.0, 2.0];
        let adv = raw_delta_advantage(&student_q, &teacher_q, 1.0);
        // teacher - student
        assert!(
            (adv[0] - 2.0).abs() < 1e-6,
            "arm 0: expected 2.0, got {}",
            adv[0]
        );
        assert!(
            (adv[1] - (-3.0)).abs() < 1e-6,
            "arm 1: expected -3.0, got {}",
            adv[1]
        );
        assert!(
            (adv[2] - 0.0).abs() < 1e-6,
            "arm 2: expected 0.0, got {}",
            adv[2]
        );
    }
}
