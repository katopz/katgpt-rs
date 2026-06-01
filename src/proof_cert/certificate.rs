use serde::{Deserialize, Serialize};

/// A standalone proof certificate for a verified property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCertificate {
    pub id: String,
    pub property: ProofProperty,
    pub result: ProofResult,
    pub prerequisites: Vec<String>,
    pub implies: Vec<String>,
    pub explanation: String,
    pub evidence: ProofEvidence,
    pub timestamp: u64, // Unix epoch seconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofProperty {
    SpatialConsistency { game: String, board_size: usize },
    DeterministicCorrectness { game: String, n_comparisons: usize },
    RealtimeFeasibility { domain: String, target_latency_us: u64 },
    Convergence { algorithm: String, metric: String },
    Custom { name: String, description: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofResult {
    Full { value: f64, threshold: f64 },
    Conditional {
        value: f64,
        threshold: f64,
        conditions: Vec<String>,
    },
    Partial {
        proved: Vec<String>,
        unproved: Vec<String>,
        reason: String,
    },
    Failed { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofEvidence {
    Benchmark {
        n_samples: usize,
        mean: f64,
        std_dev: f64,
        min: f64,
        max: f64,
    },
    Deterministic {
        seed: u64,
        expected_hash: String,
        actual_hash: String,
    },
    Comparison {
        baseline: String,
        challenger: String,
        delta: f64,
    },
    Custom { data: serde_json::Value },
}

impl ProofCertificate {
    /// Create a new certificate with current timestamp.
    pub fn new(
        id: impl Into<String>,
        property: ProofProperty,
        result: ProofResult,
        evidence: ProofEvidence,
    ) -> Self {
        Self::with_timestamp(id, property, result, evidence, now_epoch_secs())
    }

    /// Create a certificate with an explicit timestamp.
    /// Use this for batch creation to avoid repeated syscalls.
    pub fn with_timestamp(
        id: impl Into<String>,
        property: ProofProperty,
        result: ProofResult,
        evidence: ProofEvidence,
        timestamp: u64,
    ) -> Self {
        Self {
            id: id.into(),
            property,
            result,
            prerequisites: Vec::new(),
            implies: Vec::new(),
            explanation: String::new(),
            evidence,
            timestamp,
        }
    }

    /// Is this certificate's result a pass?
    #[inline]
    pub fn passed(&self) -> bool {
        matches!(&self.result, ProofResult::Full { value, threshold } if *value >= *threshold)
            || matches!(
                &self.result,
                ProofResult::Conditional { value, threshold, .. } if *value >= *threshold
            )
    }
}

#[inline]
fn now_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
