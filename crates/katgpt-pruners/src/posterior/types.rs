//! Evidence types for posterior-guided pruner evolution.
//!
//! Mirrors the Bayesian-Agent paper's TrajectoryEvidence but adapted
//! for pruner arms: verified outcomes (not self-assessment), with
//! discrete feature buckets for categorical posterior conditioning.

/// Verified outcome from a pruner arm evaluation.
#[derive(Debug, Clone)]
pub struct PosteriorEvidence {
    /// Which task/episode this evidence comes from.
    pub task_id: u64,
    /// Verified binary outcome (external grader, not self-assessment).
    pub outcome: EvidenceOutcome,
    /// Domain context (e.g., "sudoku", "bomber", "go").
    pub context: EvidenceContext,
    /// Classified failure label, if outcome is failure.
    pub failure_mode: Option<FailureMode>,
    /// Binned token/evaluation count.
    pub eval_bucket: EvalBucket,
    /// Binned wall-clock latency of the arm evaluation.
    pub latency_bucket: LatencyBucket,
}

/// Binary verified outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceOutcome {
    Success,
    Failure,
}

impl EvidenceOutcome {
    /// Returns true for success.
    pub fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

/// Domain context for feature conditioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EvidenceContext {
    Generic = 0,
    Sudoku = 1,
    Bomber = 2,
    Go = 3,
    Dungeon = 4,
    Monopoly = 5,
}

impl EvidenceContext {
    /// Number of known contexts.
    pub const COUNT: usize = 6;

    /// Convert to index for array lookup.
    pub fn as_index(self) -> usize {
        self as usize
    }
}

/// Classified failure mode for posterior conditioning.
/// When the same failure mode recurs, it triggers PATCH action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FailureMode {
    /// Pruner accepted an invalid token/state.
    FalseAccept = 0,
    /// Pruner rejected a valid token/state.
    FalseReject = 1,
    /// Pruner timed out or exceeded budget.
    Timeout = 2,
    /// Output format didn't match expected schema.
    FormatMismatch = 3,
    /// Blank/empty output when non-empty was expected.
    BlankOutput = 4,
    /// Computed correct type but wrong value.
    WrongValue = 5,
    /// Context confusion (wrong row/episode/task used).
    ContextConfusion = 6,
}

impl FailureMode {
    /// Number of known failure modes.
    pub const COUNT: usize = 7;

    /// Convert to index for array lookup.
    pub fn as_index(self) -> usize {
        self as usize
    }
}

/// Binned evaluation count (analogous to paper's token_bucket).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EvalBucket {
    /// 0 evaluations
    Zero = 0,
    /// 1-10
    Low = 1,
    /// 11-100
    Medium = 2,
    /// 101-1000
    High = 3,
    /// 1000+
    VeryHigh = 4,
}

impl EvalBucket {
    pub const COUNT: usize = 5;

    /// Classify a raw evaluation count into a bucket.
    pub fn from_count(count: usize) -> Self {
        match count {
            0 => Self::Zero,
            1..=10 => Self::Low,
            11..=100 => Self::Medium,
            101..=1000 => Self::High,
            _ => Self::VeryHigh,
        }
    }

    pub fn as_index(self) -> usize {
        self as usize
    }
}

/// Binned wall-clock latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LatencyBucket {
    /// < 1μs
    Instant = 0,
    /// 1μs - 100μs
    Fast = 1,
    /// 100μs - 10ms
    Medium = 2,
    /// 10ms - 1s
    Slow = 3,
    /// > 1s
    VerySlow = 4,
}

impl LatencyBucket {
    pub const COUNT: usize = 5;

    /// Classify a raw nanosecond duration into a bucket.
    pub fn from_nanos(nanos: u64) -> Self {
        match nanos {
            0..=999 => Self::Instant,
            1_000..=99_999 => Self::Fast,
            100_000..=9_999_999 => Self::Medium,
            10_000_000..=999_999_999 => Self::Slow,
            _ => Self::VerySlow,
        }
    }

    pub fn as_index(self) -> usize {
        self as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_bucket_classification() {
        assert_eq!(EvalBucket::from_count(0), EvalBucket::Zero);
        assert_eq!(EvalBucket::from_count(5), EvalBucket::Low);
        assert_eq!(EvalBucket::from_count(50), EvalBucket::Medium);
        assert_eq!(EvalBucket::from_count(500), EvalBucket::High);
        assert_eq!(EvalBucket::from_count(5000), EvalBucket::VeryHigh);
    }

    #[test]
    fn latency_bucket_classification() {
        assert_eq!(LatencyBucket::from_nanos(100), LatencyBucket::Instant);
        assert_eq!(LatencyBucket::from_nanos(10_000), LatencyBucket::Fast);
        assert_eq!(LatencyBucket::from_nanos(1_000_000), LatencyBucket::Medium);
        assert_eq!(LatencyBucket::from_nanos(100_000_000), LatencyBucket::Slow);
        assert_eq!(LatencyBucket::from_nanos(2_000_000_000), LatencyBucket::VerySlow);
    }

    #[test]
    fn outcome_is_success() {
        assert!(EvidenceOutcome::Success.is_success());
        assert!(!EvidenceOutcome::Failure.is_success());
    }

    #[test]
    fn context_index_range() {
        for ctx in [
            EvidenceContext::Generic,
            EvidenceContext::Sudoku,
            EvidenceContext::Bomber,
            EvidenceContext::Go,
            EvidenceContext::Dungeon,
            EvidenceContext::Monopoly,
        ] {
            assert!(ctx.as_index() < EvidenceContext::COUNT);
        }
    }
}
