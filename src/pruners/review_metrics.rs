//! Inference-time review metrics for Reinforced Agent Distillation.
//!
//! Tracks helpful/harmful classifications of reviewer intervention,
//! enabling data-driven decisions about when reviewer feedback is net-positive.
//!
//! Based on arXiv:2604.27233 — "Reinforced Agent: Inference-Time Feedback
//! for Tool-Calling Agents". The paper found a 3.1:1 benefit-to-risk ratio
//! for reasoning reviewers, with +5.5% irrelevance detection and +7.1%
//! multi-turn task improvement.
//!
//! # Classification Matrix
//!
//! | Base Correct | Reviewed Correct | Classification    |
//! |:------------:|:----------------:|:------------------|
//! | false        | true             | Helpful           |
//! | true         | false            | Harmful           |
//! | true         | true             | Both Correct      |
//! | false        | false            | Both Wrong        |
//!
//! # Thread Safety
//!
//! All counters use `AtomicU64` with `Ordering::Relaxed` — these are
//! statistics, not synchronization primitives. Zero lock contention.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Review Summary ──────────────────────────────────────────────

/// Computed review metrics snapshot for display/logging.
#[derive(Clone, Debug, Default)]
pub struct ReviewSummary {
    /// % of cases where base was WRONG and reviewer FIXED it.
    pub helpfulness: f64,
    /// % of cases where base was RIGHT and reviewer BROKE it.
    pub harmfulness: f64,
    /// Helpfulness ÷ Harmfulness (benefit-to-risk ratio).
    /// `f64::INFINITY` when harmfulness is zero.
    pub benefit_ratio: f64,
    /// Total observations classified.
    pub total: u64,
    /// Raw counter: base wrong → reviewed correct.
    pub helpful: u64,
    /// Raw counter: base correct → reviewed wrong.
    pub harmful: u64,
    /// Raw counter: both agreed correct.
    pub both_correct: u64,
    /// Raw counter: both agreed wrong.
    pub both_wrong: u64,
}

// ── Review Metrics ──────────────────────────────────────────────

/// Thread-safe review metrics using atomic counters.
///
/// Classifies each (base_correct, reviewed_correct) pair and increments
/// the appropriate counter. All operations are lock-free.
///
/// # Example
///
/// ```rust,ignore
/// let metrics = ReviewMetrics::new();
///
/// // Base was wrong, reviewer fixed it → helpful
/// metrics.record(false, true);
///
/// // Base was right, reviewer broke it → harmful
/// metrics.record(true, false);
///
/// // Both correct
/// metrics.record(true, true);
///
/// let summary = metrics.summary();
/// println!("{metrics}"); // "helpful=33.3% harmful=33.3% ratio=1.0:1 n=3"
/// ```
/// Snapshot of entropy anomaly statistics for a session (Plan 061).
#[derive(Clone, Debug, Default)]
pub struct EntropyAnomalySummary {
    /// Mean entropy across all recorded positions.
    pub mean: f64,
    /// Maximum single-position entropy observed.
    pub max: f64,
    /// Number of entropy observations recorded.
    pub count: u64,
}

pub struct ReviewMetrics {
    helpful: AtomicU64,
    harmful: AtomicU64,
    both_correct: AtomicU64,
    both_wrong: AtomicU64,
    /// Paths where final answer was correct but intermediate steps were shaky (Plan 054).
    path_hacking: AtomicU64,
    /// Paths where both final AND intermediate steps were correct (Plan 054).
    path_faithful: AtomicU64,
    /// Total paths analyzed for consistency (Plan 054).
    path_total: AtomicU64,
    /// Running sum of path consistency values (Plan 054).
    path_consistency_sum: AtomicU64,
    /// Running sum of entropy values × 10000 (Plan 061: OOD drift signal).
    entropy_sum: AtomicU64,
    /// Number of entropy observations (Plan 061).
    entropy_count: AtomicU64,
    /// Maximum entropy spike observed × 10000 (Plan 061).
    entropy_max: AtomicU64,
    // ── Emotion Vector Fields (Plan 162: Emotion Vector Inference Control) ──
    /// Running sum of valence projections × 10000 (Plan 162).
    emotion_valence_sum: AtomicU64,
    /// Running sum of arousal projections × 10000 (Plan 162).
    emotion_arousal_sum: AtomicU64,
    /// Running sum of desperation projections × 10000 (Plan 162).
    desperation_score_sum: AtomicU64,
    /// Running sum of calm projections × 10000 (Plan 162).
    calm_score_sum: AtomicU64,
    /// Number of emotion observations recorded (Plan 162).
    emotion_count: AtomicU64,
}

impl ReviewMetrics {
    /// Create zero-initialized metrics.
    pub fn new() -> Self {
        Self {
            helpful: AtomicU64::new(0),
            harmful: AtomicU64::new(0),
            both_correct: AtomicU64::new(0),
            both_wrong: AtomicU64::new(0),
            path_hacking: AtomicU64::new(0),
            path_faithful: AtomicU64::new(0),
            path_total: AtomicU64::new(0),
            path_consistency_sum: AtomicU64::new(0),
            entropy_sum: AtomicU64::new(0),
            entropy_count: AtomicU64::new(0),
            entropy_max: AtomicU64::new(0),
            emotion_valence_sum: AtomicU64::new(0),
            emotion_arousal_sum: AtomicU64::new(0),
            desperation_score_sum: AtomicU64::new(0),
            calm_score_sum: AtomicU64::new(0),
            emotion_count: AtomicU64::new(0),
        }
    }

    /// Classify and record a (base, reviewed) outcome.
    ///
    /// Thread-safe: single atomic increment, no lock.
    pub fn record(&self, base_correct: bool, reviewed_correct: bool) {
        match (base_correct, reviewed_correct) {
            (false, true) => self.helpful.fetch_add(1, Ordering::Relaxed),
            (true, false) => self.harmful.fetch_add(1, Ordering::Relaxed),
            (true, true) => self.both_correct.fetch_add(1, Ordering::Relaxed),
            (false, false) => self.both_wrong.fetch_add(1, Ordering::Relaxed),
        };
    }

    /// % of cases where base was WRONG and reviewer FIXED it.
    ///
    /// `helpful / (helpful + both_wrong)` as percentage.
    /// Returns 0.0 when denominator is zero.
    pub fn helpfulness(&self) -> f64 {
        let h = self.helpful.load(Ordering::Relaxed) as f64;
        let w = self.both_wrong.load(Ordering::Relaxed) as f64;
        let denom = h + w;
        if denom == 0.0 { 0.0 } else { h / denom * 100.0 }
    }

    /// % of cases where base was RIGHT and reviewer BROKE it.
    ///
    /// `harmful / (harmful + both_correct)` as percentage.
    /// Returns 0.0 when denominator is zero.
    pub fn harmfulness(&self) -> f64 {
        let h = self.harmful.load(Ordering::Relaxed) as f64;
        let c = self.both_correct.load(Ordering::Relaxed) as f64;
        let denom = h + c;
        if denom == 0.0 { 0.0 } else { h / denom * 100.0 }
    }

    /// Benefit-to-risk ratio: `helpfulness / harmfulness`.
    ///
    /// Returns `f64::INFINITY` when harmfulness is zero
    /// (reviewer never broke a correct base).
    pub fn benefit_ratio(&self) -> f64 {
        let helpfulness = self.helpfulness();
        let harmfulness = self.harmfulness();
        if harmfulness == 0.0 {
            if helpfulness > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            helpfulness / harmfulness
        }
    }

    /// Total number of classified observations.
    pub fn total(&self) -> u64 {
        self.helpful.load(Ordering::Relaxed)
            + self.harmful.load(Ordering::Relaxed)
            + self.both_correct.load(Ordering::Relaxed)
            + self.both_wrong.load(Ordering::Relaxed)
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.helpful.store(0, Ordering::Relaxed);
        self.harmful.store(0, Ordering::Relaxed);
        self.both_correct.store(0, Ordering::Relaxed);
        self.both_wrong.store(0, Ordering::Relaxed);
        self.entropy_sum.store(0, Ordering::Relaxed);
        self.entropy_count.store(0, Ordering::Relaxed);
        self.entropy_max.store(0, Ordering::Relaxed);
        self.emotion_valence_sum.store(0, Ordering::Relaxed);
        self.emotion_arousal_sum.store(0, Ordering::Relaxed);
        self.desperation_score_sum.store(0, Ordering::Relaxed);
        self.calm_score_sum.store(0, Ordering::Relaxed);
        self.emotion_count.store(0, Ordering::Relaxed);
    }

    /// Compute a snapshot of all metrics for display/logging.
    ///
    /// Snapshots all 4 counters once, then derives ratios from the snapshot
    /// to avoid cascading atomic loads (4 loads instead of ~16).
    pub fn summary(&self) -> ReviewSummary {
        let helpful = self.helpful.load(Ordering::Relaxed);
        let harmful = self.harmful.load(Ordering::Relaxed);
        let both_correct = self.both_correct.load(Ordering::Relaxed);
        let both_wrong = self.both_wrong.load(Ordering::Relaxed);

        let helpful_f = helpful as f64;
        let harmful_f = harmful as f64;
        let both_correct_f = both_correct as f64;
        let both_wrong_f = both_wrong as f64;

        let denom_help = helpful_f + both_wrong_f;
        let helpfulness = if denom_help == 0.0 {
            0.0
        } else {
            helpful_f / denom_help * 100.0
        };

        let denom_harm = harmful_f + both_correct_f;
        let harmfulness = if denom_harm == 0.0 {
            0.0
        } else {
            harmful_f / denom_harm * 100.0
        };

        let benefit_ratio = if harmfulness == 0.0 {
            if helpfulness > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            helpfulness / harmfulness
        };

        let total = helpful + harmful + both_correct + both_wrong;

        ReviewSummary {
            helpfulness,
            harmfulness,
            benefit_ratio,
            total,
            helpful,
            harmful,
            both_correct,
            both_wrong,
        }
    }

    /// Raw helpful count (base wrong, reviewer fixed).
    pub fn helpful_count(&self) -> u64 {
        self.helpful.load(Ordering::Relaxed)
    }

    /// Raw harmful count (base correct, reviewer broke).
    pub fn harmful_count(&self) -> u64 {
        self.harmful.load(Ordering::Relaxed)
    }

    /// Raw both-correct count.
    pub fn both_correct_count(&self) -> u64 {
        self.both_correct.load(Ordering::Relaxed)
    }

    /// Raw both-wrong count.
    pub fn both_wrong_count(&self) -> u64 {
        self.both_wrong.load(Ordering::Relaxed)
    }

    // ── Path Consistency (StepCodeReasoner Plan 054) ─────────────

    /// Classify a path by its consistency vs final correctness.
    ///
    /// Maps to StepCodeReasoner's "right answer, wrong logic" detection:
    /// - final_correct && consistency >= threshold → fully_faithful
    /// - final_correct && consistency < threshold → reward_hacking
    /// - !final_correct → not counted (no credit assignment issue)
    ///
    /// Returns a snapshot of the cumulative path consistency summary.
    pub fn classify_path(&self, final_correct: bool, consistency: f32, threshold: f32) {
        if final_correct {
            if consistency >= threshold {
                self.path_faithful.fetch_add(1, Ordering::Relaxed);
            } else {
                self.path_hacking.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.path_total.fetch_add(1, Ordering::Relaxed);
        // Store consistency * 10000 as u64 to preserve precision
        self.path_consistency_sum
            .fetch_add((consistency * 10000.0) as u64, Ordering::Relaxed);
    }

    /// Snapshot of cumulative path consistency statistics (Plan 054).
    pub fn path_consistency_summary(&self) -> PathConsistencySummary {
        let total = self.path_total.load(Ordering::Relaxed);
        let avg_consistency = if total > 0 {
            let sum = self.path_consistency_sum.load(Ordering::Relaxed);
            sum as f64 / (total as f64 * 10000.0)
        } else {
            0.0
        };
        PathConsistencySummary {
            reward_hacking: self.path_hacking.load(Ordering::Relaxed),
            fully_faithful: self.path_faithful.load(Ordering::Relaxed),
            total_paths: total,
            avg_consistency,
        }
    }

    /// Number of paths classified as reward hacking.
    pub fn path_hacking_count(&self) -> u64 {
        self.path_hacking.load(Ordering::Relaxed)
    }

    /// Number of paths classified as fully faithful.
    pub fn path_faithful_count(&self) -> u64 {
        self.path_faithful.load(Ordering::Relaxed)
    }

    // ── Entropy Anomaly (Plan 061: OOD Drift Signal) ─────────────

    /// Record a Shannon entropy observation from PPoT token predictions.
    ///
    /// Call this after each decoding step with the entropy of the chosen token's
    /// marginal distribution. Thread-safe: single atomic updates, no lock.
    ///
    /// Precision: entropy is stored as `h * 10000` in a u64 (4 decimal places).
    pub fn record_entropy(&self, entropy: f32) {
        let scaled = (entropy * 10000.0) as u64;
        self.entropy_sum.fetch_add(scaled, Ordering::Relaxed);
        self.entropy_count.fetch_add(1, Ordering::Relaxed);
        // CAS loop to update max
        loop {
            let current = self.entropy_max.load(Ordering::Relaxed);
            if scaled <= current {
                break;
            }
            match self.entropy_max.compare_exchange_weak(
                current,
                scaled,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }

    /// Snapshot of entropy anomaly statistics for this session.
    pub fn entropy_anomaly_summary(&self) -> EntropyAnomalySummary {
        let count = self.entropy_count.load(Ordering::Relaxed);
        let mean = if count > 0 {
            let sum = self.entropy_sum.load(Ordering::Relaxed);
            sum as f64 / (count as f64 * 10000.0)
        } else {
            0.0
        };
        let max = self.entropy_max.load(Ordering::Relaxed) as f64 / 10000.0;
        EntropyAnomalySummary { mean, max, count }
    }

    /// Whether the session's mean entropy exceeds a threshold.
    ///
    /// Use `ln(vocab_size) * 0.7` as a reasonable default:
    /// - micro config (vocab=32): threshold ≈ 2.42
    /// - standard (vocab=32000): threshold ≈ 6.98
    pub fn is_high_entropy_session(&self, threshold: f64) -> bool {
        let summary = self.entropy_anomaly_summary();
        summary.count > 0 && summary.mean > threshold
    }

    // ── Emotion Vector (Plan 162: Emotion Vector Inference Control) ─────

    /// Record an emotion reading from mid-layer activation projection.
    ///
    /// Thread-safe: atomic updates, no lock. Precision: stored as `v * 10000`
    /// in u64 (4 decimal places), matching entropy convention.
    pub fn record_emotion(&self, valence: f32, arousal: f32, desperation: f32, calm: f32) {
        self.emotion_valence_sum
            .fetch_add((valence * 10000.0) as u64, Ordering::Relaxed);
        self.emotion_arousal_sum
            .fetch_add((arousal * 10000.0) as u64, Ordering::Relaxed);
        self.desperation_score_sum
            .fetch_add((desperation * 10000.0) as u64, Ordering::Relaxed);
        self.calm_score_sum
            .fetch_add((calm * 10000.0) as u64, Ordering::Relaxed);
        self.emotion_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot of emotion profile statistics for this session (Plan 162).
    ///
    /// Returns running means of valence, arousal, desperation, and calm scores.
    pub fn emotion_profile_summary(&self) -> EmotionProfileSummary {
        let count = self.emotion_count.load(Ordering::Relaxed);
        if count == 0 {
            return EmotionProfileSummary::default();
        }
        let scale = count as f64 * 10000.0;
        EmotionProfileSummary {
            valence: self.emotion_valence_sum.load(Ordering::Relaxed) as f64 / scale,
            arousal: self.emotion_arousal_sum.load(Ordering::Relaxed) as f64 / scale,
            desperation: self.desperation_score_sum.load(Ordering::Relaxed) as f64 / scale,
            calm: self.calm_score_sum.load(Ordering::Relaxed) as f64 / scale,
            count,
        }
    }

    /// Whether the session's mean desperation score exceeds a threshold (Plan 162).
    ///
    /// High desperation correlates with reward-hacking-prone regimes.
    /// Use domain-specific threshold; a reasonable starting point is 0.5.
    pub fn is_desperate_session(&self, threshold: f32) -> bool {
        let profile = self.emotion_profile_summary();
        profile.count > 0 && profile.desperation > threshold as f64
    }
}

impl Default for ReviewMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ReviewSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ratio_str = if self.benefit_ratio.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.1}", self.benefit_ratio)
        };
        write!(
            f,
            "helpful={:.1}% harmful={:.1}% ratio={}:1 n={}",
            self.helpfulness, self.harmfulness, ratio_str, self.total
        )
    }
}

impl fmt::Display for ReviewMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.summary();
        let ratio_str = if s.benefit_ratio == f64::INFINITY {
            "∞".to_string()
        } else {
            format!("{:.1}:1", s.benefit_ratio)
        };
        let entropy = self.entropy_anomaly_summary();
        let emotion = self.emotion_profile_summary();
        write!(
            f,
            "helpful={:.1}% harmful={:.1}% ratio={} n={} entropy_mean={:.3} entropy_max={:.3} entropy_n={} valence={:.3} arousal={:.3} desperation={:.3} calm={:.3} emotion_n={}",
            s.helpfulness,
            s.harmfulness,
            ratio_str,
            s.total,
            entropy.mean,
            entropy.max,
            entropy.count,
            emotion.valence,
            emotion.arousal,
            emotion.desperation,
            emotion.calm,
            emotion.count,
        )
    }
}

// ── Path Consistency Summary (StepCodeReasoner Plan 054) ────────

/// Classification result including path consistency (StepCodeReasoner Plan 054).
///
/// Detects "right answer, wrong logic" — paths where the final outcome was
/// correct but intermediate steps were shaky, indicating reward hacking.
#[derive(Clone, Debug, Default)]
pub struct PathConsistencySummary {
    /// Number of paths where final answer was correct but intermediate steps were shaky.
    pub reward_hacking: u64,
    /// Number of paths where both final AND intermediate steps were correct.
    pub fully_faithful: u64,
    /// Total paths analyzed.
    pub total_paths: u64,
    /// Average path consistency across all paths (0.0 to 1.0).
    pub avg_consistency: f64,
}

impl fmt::Display for PathConsistencySummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "faithful={} hacking={} total={} avg_consistency={:.2}",
            self.fully_faithful, self.reward_hacking, self.total_paths, self.avg_consistency
        )
    }
}

// ── Emotion Profile Summary (Plan 162: Emotion Vector Inference Control) ──

/// Snapshot of emotion vector statistics for a session (Plan 162).
///
/// Running means of valence, arousal, desperation, and calm projections
/// from mid-layer activations during decode.
#[derive(Clone, Debug, Default)]
pub struct EmotionProfileSummary {
    /// Mean valence across all recorded decode steps.
    pub valence: f64,
    /// Mean arousal across all recorded decode steps.
    pub arousal: f64,
    /// Mean desperation score across all recorded decode steps.
    pub desperation: f64,
    /// Mean calm score across all recorded decode steps.
    pub calm: f64,
    /// Number of emotion observations recorded.
    pub count: u64,
}

impl fmt::Display for EmotionProfileSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "valence={:.3} arousal={:.3} desperation={:.3} calm={:.3} n={}",
            self.valence, self.arousal, self.desperation, self.calm, self.count
        )
    }
}

// ── Review Strategy ─────────────────────────────────────────────

/// Structured review loop strategy mirroring arXiv:2604.27233's mechanisms.
///
/// The paper evaluates three inference-time feedback strategies:
/// - **Progressive Feedback** (rN): iteratively review and inject rejection feedback.
///   Best performer in the paper.
/// - **Best-of-N Selection** (sN): generate N candidates, reviewer picks best.
///   Maps to DDTree with budget N.
/// - **Best-of-N Grading** (gN): score each candidate 0.0–1.0, pick highest.
///   Maps to DDTree with ScreeningPruner relevance scoring.
///
/// Paper notation: `r2` = ProgressiveFeedback with max_loops=2, `s5` = BestOfNSelection
/// with candidates=5, `g5` = BestOfNGrading with candidates=5.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReviewStrategy {
    /// Progressive feedback: iteratively review and inject rejection feedback.
    /// Paper's rN — up to N review loops.
    ProgressiveFeedback { max_loops: usize },
    /// Best-of-N selection: generate N candidates, reviewer picks best.
    /// Maps to DDTree with budget N.
    BestOfNSelection { candidates: usize },
    /// Best-of-N grading: score each candidate 0.0–1.0, pick highest.
    /// Maps to DDTree with ScreeningPruner relevance scoring.
    BestOfNGrading { candidates: usize },
}

impl Default for ReviewStrategy {
    /// Default: `ProgressiveFeedback { max_loops: 2 }` — paper's best performer.
    fn default() -> Self {
        Self::ProgressiveFeedback { max_loops: 2 }
    }
}

impl fmt::Display for ReviewStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProgressiveFeedback { max_loops } => write!(f, "r{max_loops}"),
            Self::BestOfNSelection { candidates } => write!(f, "s{candidates}"),
            Self::BestOfNGrading { candidates } => write!(f, "g{candidates}"),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_classifications() {
        let metrics = ReviewMetrics::new();

        // Helpful: base wrong, reviewer fixed
        metrics.record(false, true);
        assert_eq!(metrics.helpful_count(), 1);

        // Harmful: base correct, reviewer broke
        metrics.record(true, false);
        assert_eq!(metrics.harmful_count(), 1);

        // Both correct
        metrics.record(true, true);
        assert_eq!(metrics.both_correct_count(), 1);

        // Both wrong
        metrics.record(false, false);
        assert_eq!(metrics.both_wrong_count(), 1);

        assert_eq!(metrics.total(), 4);
    }

    #[test]
    fn test_helpfulness_calculation() {
        let metrics = ReviewMetrics::new();

        // 3 helpful out of (3 helpful + 1 both_wrong) = 75%
        metrics.record(false, true); // helpful
        metrics.record(false, true); // helpful
        metrics.record(false, true); // helpful
        metrics.record(false, false); // both_wrong

        assert_eq!(metrics.helpful_count(), 3);
        assert_eq!(metrics.both_wrong_count(), 1);
        let h = metrics.helpfulness();
        assert!((h - 75.0).abs() < 0.01, "expected 75.0%, got {h}%");
    }

    #[test]
    fn test_harmfulness_calculation() {
        let metrics = ReviewMetrics::new();

        // 1 harmful out of (1 harmful + 3 both_correct) = 25%
        metrics.record(true, false); // harmful
        metrics.record(true, true); // both_correct
        metrics.record(true, true); // both_correct
        metrics.record(true, true); // both_correct

        let h = metrics.harmfulness();
        assert!((h - 25.0).abs() < 0.01, "expected 25.0%, got {h}%");
    }

    #[test]
    fn test_benefit_ratio() {
        let metrics = ReviewMetrics::new();

        // helpfulness=75%, harmfulness=25% → ratio=3.0
        metrics.record(false, true); // helpful
        metrics.record(false, true); // helpful
        metrics.record(false, true); // helpful
        metrics.record(false, false); // both_wrong

        metrics.record(true, false); // harmful
        metrics.record(true, true); // both_correct
        metrics.record(true, true); // both_correct
        metrics.record(true, true); // both_correct

        let ratio = metrics.benefit_ratio();
        assert!((ratio - 3.0).abs() < 0.01, "expected 3.0, got {ratio}");
    }

    #[test]
    fn test_zero_harmful_infinity_ratio() {
        let metrics = ReviewMetrics::new();

        // Only helpful records, no harmful → ratio = ∞
        metrics.record(false, true); // helpful
        metrics.record(false, false); // both_wrong

        // No harmful records: base_correct=true never produced reviewed_correct=false
        let ratio = metrics.benefit_ratio();
        assert!(ratio.is_infinite(), "expected infinity, got {ratio}");
    }

    #[test]
    fn test_empty_metrics_zero() {
        let metrics = ReviewMetrics::new();

        assert_eq!(metrics.total(), 0);
        assert_eq!(metrics.helpfulness(), 0.0);
        assert_eq!(metrics.harmfulness(), 0.0);
        assert_eq!(metrics.benefit_ratio(), 0.0);
    }

    #[test]
    fn test_reset() {
        let metrics = ReviewMetrics::new();
        metrics.record(false, true);
        metrics.record(true, false);
        assert_eq!(metrics.total(), 2);

        metrics.reset();
        assert_eq!(metrics.total(), 0);
        assert_eq!(metrics.helpful_count(), 0);
        assert_eq!(metrics.harmful_count(), 0);
    }

    #[test]
    fn test_display_format() {
        let metrics = ReviewMetrics::new();

        // Create 3 helpful, 1 harmful, 5 both_correct, 1 both_wrong → 10 total
        for _ in 0..3 {
            metrics.record(false, true); // helpful
        }
        metrics.record(true, false); // harmful
        for _ in 0..5 {
            metrics.record(true, true); // both_correct
        }
        metrics.record(false, false); // both_wrong

        let display = format!("{metrics}");
        assert!(
            display.contains("helpful="),
            "display should contain 'helpful=': {display}"
        );
        assert!(
            display.contains("harmful="),
            "display should contain 'harmful=': {display}"
        );
        assert!(
            display.contains("ratio="),
            "display should contain 'ratio=': {display}"
        );
        assert!(
            display.contains("n=10"),
            "display should contain 'n=10': {display}"
        );
    }

    #[test]
    fn test_display_infinity_ratio() {
        let metrics = ReviewMetrics::new();
        metrics.record(false, true);
        let display = format!("{metrics}");
        assert!(
            display.contains("ratio=∞"),
            "display should show ∞ for infinite ratio: {display}"
        );
    }

    #[test]
    fn test_summary_matches_metrics() {
        let metrics = ReviewMetrics::new();
        metrics.record(false, true);
        metrics.record(true, false);
        metrics.record(true, true);
        metrics.record(false, false);

        let summary = metrics.summary();
        assert_eq!(summary.helpful, 1);
        assert_eq!(summary.harmful, 1);
        assert_eq!(summary.both_correct, 1);
        assert_eq!(summary.both_wrong, 1);
        assert_eq!(summary.total, 4);
        assert!((summary.helpfulness - metrics.helpfulness()).abs() < 0.01);
        assert!((summary.harmfulness - metrics.harmfulness()).abs() < 0.01);
    }

    // ── ReviewStrategy Tests ────────────────────────────────────

    #[test]
    fn test_strategy_default() {
        match ReviewStrategy::default() {
            ReviewStrategy::ProgressiveFeedback { max_loops } => {
                assert_eq!(max_loops, 2);
            }
            _ => panic!("default should be ProgressiveFeedback"),
        }
    }

    #[test]
    fn test_strategy_display_progressive() {
        let strategy = ReviewStrategy::ProgressiveFeedback { max_loops: 3 };
        assert_eq!(format!("{strategy}"), "r3");
    }

    #[test]
    fn test_strategy_display_selection() {
        let strategy = ReviewStrategy::BestOfNSelection { candidates: 5 };
        assert_eq!(format!("{strategy}"), "s5");
    }

    #[test]
    fn test_strategy_display_grading() {
        let strategy = ReviewStrategy::BestOfNGrading { candidates: 10 };
        assert_eq!(format!("{strategy}"), "g10");
    }

    #[test]
    fn test_strategy_equality() {
        let a = ReviewStrategy::ProgressiveFeedback { max_loops: 2 };
        let b = ReviewStrategy::ProgressiveFeedback { max_loops: 2 };
        let c = ReviewStrategy::ProgressiveFeedback { max_loops: 3 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_summary_display() {
        let summary = ReviewSummary {
            helpfulness: 36.8,
            harmfulness: 11.7,
            benefit_ratio: 3.1,
            total: 1000,
            helpful: 368,
            harmful: 117,
            both_correct: 400,
            both_wrong: 115,
        };
        let display = format!("{summary}");
        assert!(
            display.contains("helpful=36.8%"),
            "expected helpful=36.8% in: {display}"
        );
        assert!(
            display.contains("harmful=11.7%"),
            "expected harmful=11.7% in: {display}"
        );
        assert!(
            display.contains("ratio=3.1:1"),
            "expected ratio=3.1:1 in: {display}"
        );
        assert!(display.contains("n=1000"), "expected n=1000 in: {display}");
    }

    // ── Entropy Anomaly Tests (Plan 061) ───────────────────────

    #[test]
    fn test_entropy_anomaly_empty() {
        let metrics = ReviewMetrics::new();
        let summary = metrics.entropy_anomaly_summary();
        assert_eq!(summary.count, 0);
        assert!((summary.mean - 0.0).abs() < 1e-6);
        assert!((summary.max - 0.0).abs() < 1e-6);
        assert!(!metrics.is_high_entropy_session(1.0));
    }

    #[test]
    fn test_entropy_anomaly_single_observation() {
        let metrics = ReviewMetrics::new();
        metrics.record_entropy(1.5);
        let summary = metrics.entropy_anomaly_summary();
        assert_eq!(summary.count, 1);
        assert!(
            (summary.mean - 1.5).abs() < 0.001,
            "mean should be ~1.5, got {}",
            summary.mean
        );
        assert!(
            (summary.max - 1.5).abs() < 0.001,
            "max should be ~1.5, got {}",
            summary.max
        );
    }

    #[test]
    fn test_entropy_anomaly_multiple_observations() {
        let metrics = ReviewMetrics::new();
        metrics.record_entropy(1.0);
        metrics.record_entropy(2.0);
        metrics.record_entropy(3.0);
        let summary = metrics.entropy_anomaly_summary();
        assert_eq!(summary.count, 3);
        assert!(
            (summary.mean - 2.0).abs() < 0.01,
            "mean should be ~2.0, got {}",
            summary.mean
        );
        assert!(
            (summary.max - 3.0).abs() < 0.01,
            "max should be ~3.0, got {}",
            summary.max
        );
    }

    #[test]
    fn test_entropy_anomaly_max_updates() {
        let metrics = ReviewMetrics::new();
        metrics.record_entropy(1.0);
        metrics.record_entropy(5.0);
        metrics.record_entropy(2.0);
        let summary = metrics.entropy_anomaly_summary();
        assert!(
            (summary.max - 5.0).abs() < 0.01,
            "max should be 5.0, got {}",
            summary.max
        );
    }

    #[test]
    fn test_is_high_entropy_session_below_threshold() {
        let metrics = ReviewMetrics::new();
        for _ in 0..10 {
            metrics.record_entropy(0.5); // Low entropy — model is confident
        }
        assert!(
            !metrics.is_high_entropy_session(2.0),
            "mean=0.5 should be below threshold=2.0"
        );
    }

    #[test]
    fn test_is_high_entropy_session_above_threshold() {
        let metrics = ReviewMetrics::new();
        for _ in 0..10 {
            metrics.record_entropy(3.0); // High entropy — model is confused
        }
        assert!(
            metrics.is_high_entropy_session(2.0),
            "mean=3.0 should be above threshold=2.0"
        );
    }

    #[test]
    fn test_is_high_entropy_session_no_observations() {
        let metrics = ReviewMetrics::new();
        assert!(
            !metrics.is_high_entropy_session(0.0),
            "no observations should return false"
        );
    }

    #[test]
    fn test_entropy_anomaly_display_includes_entropy_fields() {
        let metrics = ReviewMetrics::new();
        metrics.record(false, true);
        metrics.record_entropy(2.5);
        let display = format!("{metrics}");
        assert!(
            display.contains("entropy_mean="),
            "display should contain 'entropy_mean=': {display}"
        );
        assert!(
            display.contains("entropy_max="),
            "display should contain 'entropy_max=': {display}"
        );
        assert!(
            display.contains("entropy_n=1"),
            "display should contain 'entropy_n=1': {display}"
        );
    }

    #[test]
    fn test_entropy_anomaly_reset() {
        let metrics = ReviewMetrics::new();
        metrics.record_entropy(5.0);
        metrics.record_entropy(3.0);
        metrics.reset();
        let summary = metrics.entropy_anomaly_summary();
        assert_eq!(summary.count, 0, "reset should clear entropy tracking");
        assert!(
            (summary.mean).abs() < 1e-6,
            "reset should clear entropy mean"
        );
        assert!((summary.max).abs() < 1e-6, "reset should clear entropy max");
    }

    #[test]
    fn test_entropy_precision_scaled_storage() {
        // Verify that entropy values are stored with sufficient precision
        let metrics = ReviewMetrics::new();
        let entropy = 2.3456;
        metrics.record_entropy(entropy);
        let summary = metrics.entropy_anomaly_summary();
        assert!(
            (summary.mean - entropy as f64).abs() < 0.001,
            "precision should be within 0.001, got diff of {}",
            (summary.mean - entropy as f64).abs()
        );
    }
}
