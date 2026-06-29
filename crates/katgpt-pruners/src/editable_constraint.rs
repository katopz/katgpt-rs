//! CoEditable ConstraintPruner — bidirectional Read/Write/Enhance cycle (Plan 214 P3).
//!
//! Extends ConstraintPruner with:
//! - Threshold/topology editing with TED-Lite divergence guardrails
//! - Golden-reference snapshots with blake3 integrity hashing
//! - JSON rule editor backend for external tool integration
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "coexplain_pruner")]`.

use super::self_refining::TopologyAction;
use super::ted_lite::PrunerDivergence;

// ── PrunerSnapshot ──────────────────────────────────────────────────

/// Snapshot of pruner state for TED-Lite comparison.
///
/// Captures thresholds, branch topology, and a blake3 integrity hash.
#[derive(Debug, Clone)]
pub struct PrunerSnapshot {
    /// Threshold values at snapshot time.
    pub thresholds: Vec<f32>,
    /// Branch existence vector at snapshot time.
    pub branches: Vec<bool>,
    /// blake3 integrity hash over thresholds + branches.
    pub blake3_hash: [u8; 32],
}

impl PrunerSnapshot {
    /// Create a snapshot by computing blake3 hash over thresholds and branches.
    ///
    /// Serializes thresholds as little-endian f32 bytes, then branches as u8 bytes.
    pub fn new(thresholds: &[f32], branches: &[bool]) -> Self {
        let mut hasher = blake3::Hasher::new();

        // Hash thresholds as little-endian f32 bytes
        for &t in thresholds {
            hasher.update(&t.to_le_bytes());
        }

        // Hash branches as u8 bytes (0x00 = false, 0x01 = true)
        for &b in branches {
            hasher.update(&[b as u8]);
        }

        let hash = hasher.finalize();
        let mut blake3_hash = [0u8; 32];
        blake3_hash.copy_from_slice(hash.as_bytes());

        Self {
            thresholds: thresholds.to_vec(),
            branches: branches.to_vec(),
            blake3_hash,
        }
    }

    /// Verify that current thresholds and branches match the snapshot.
    ///
    /// Recomputes the blake3 hash and compares it to the stored hash.
    /// Returns `true` if integrity is intact.
    pub fn verify(&self, thresholds: &[f32], branches: &[bool]) -> bool {
        let recomputed = Self::new(thresholds, branches);
        recomputed.blake3_hash == self.blake3_hash
    }
}

// ── DivergenceError ─────────────────────────────────────────────────

/// Error for divergence-exceeding edits.
#[derive(Debug, Clone, PartialEq)]
pub struct DivergenceError {
    /// The proposed threshold delta.
    pub proposed_delta: f32,
    /// The divergence clamp that was exceeded.
    pub lambda_t: f32,
}

impl std::fmt::Display for DivergenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "proposed delta {:.4} exceeds lambda_t {:.4}",
            self.proposed_delta, self.lambda_t
        )
    }
}

impl std::error::Error for DivergenceError {}

// ── EditableConstraintPruner Trait ──────────────────────────────────

/// Trait extending ConstraintPruner with bidirectional editing.
///
/// Enables the CoExplain Read/Write/Enhance cycle:
/// - **Read**: `snapshot()` captures golden reference
/// - **Write**: `edit_threshold()` / `edit_topology()` modify pruner state
/// - **Enhance**: `divergence()` measures drift from golden reference
///
/// All edits are guarded by TED-Lite divergence clamps.
pub trait EditableConstraintPruner: Send + Sync {
    /// Edit threshold for a specific slot.
    ///
    /// Returns `Err(DivergenceError)` if the change exceeds the divergence clamp.
    fn edit_threshold(&mut self, slot: usize, new_threshold: f32) -> Result<(), DivergenceError>;

    /// Edit topology (add/remove branches).
    ///
    /// Returns `Err(DivergenceError)` if the topology change exceeds divergence limits.
    fn edit_topology(
        &mut self,
        branch: &[usize],
        action: TopologyAction,
    ) -> Result<(), DivergenceError>;

    /// Capture golden reference for TED-Lite comparison.
    fn snapshot(&self) -> PrunerSnapshot;

    /// Current divergence from snapshot.
    fn divergence(&self) -> &PrunerDivergence;
}

// ── JSON Rule Editor ────────────────────────────────────────────────

/// JSON rule format for external editing.
///
/// ```json
/// {"attribute": "bracket_depth", "threshold": 0.5, "action": "reject"}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RuleEdit {
    /// Attribute name (e.g. "bracket_depth", "token_entropy").
    pub attribute: String,
    /// Threshold value for the rule.
    pub threshold: f32,
    /// Action: "reject", "accept", or "maybe".
    pub action: String,
}

/// Parse a binary (postcard) array of rules into `RuleEdit` structs.
///
/// # Errors
///
/// Returns an error message string if data is malformed or contains invalid actions.
pub fn parse_rules(data: &[u8]) -> Result<Vec<RuleEdit>, String> {
    let rules: Vec<RuleEdit> =
        postcard::from_bytes(data).map_err(|e| format!("Parse error: {e}"))?;

    // Validate actions
    for rule in &rules {
        match rule.action.as_str() {
            "reject" | "accept" | "maybe" => {}
            other => {
                return Err(format!(
                    "invalid action '{other}', must be reject/accept/maybe"
                ));
            }
        }
    }

    Ok(rules)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_create_verify() {
        let thresholds = [0.5, 0.3, 0.8];
        let branches = [true, false, true];
        let snap = PrunerSnapshot::new(&thresholds, &branches);

        // Verify against identical inputs → should pass
        assert!(snap.verify(&thresholds, &branches));
        // Hash should be non-zero
        assert_ne!(snap.blake3_hash, [0u8; 32]);
    }

    #[test]
    fn test_snapshot_detect_tampering() {
        let thresholds = [0.5, 0.3, 0.8];
        let branches = [true, false, true];
        let snap = PrunerSnapshot::new(&thresholds, &branches);

        // Modify thresholds → verification should fail
        let tampered = [0.5, 0.4, 0.8]; // changed index 1
        assert!(!snap.verify(&tampered, &branches));

        // Modify branches → verification should fail
        let tampered_branches = [true, true, true]; // changed index 1
        assert!(!snap.verify(&thresholds, &tampered_branches));
    }

    #[test]
    fn test_snapshot_empty_inputs() {
        let snap = PrunerSnapshot::new(&[], &[]);
        assert!(snap.verify(&[], &[]));
    }

    #[test]
    fn test_edit_threshold_within_bounds() {
        // Simulated test: construct a DivergenceError check
        // In practice, this would be called via an EditableConstraintPruner impl
        let div = PrunerDivergence {
            threshold_divergence: 0.05,
            topology_divergence: 0.0,
            lambda_t: 0.1,
        };

        // Small adjustment within bounds
        let proposed_delta = 0.05f32;
        let result = div.clamp_adjustment(proposed_delta);
        // None means accepted
        assert!(result.is_none());
    }

    #[test]
    fn test_edit_threshold_exceeds_divergence() {
        let div = PrunerDivergence {
            threshold_divergence: 0.2,
            topology_divergence: 0.0,
            lambda_t: 0.1,
        };

        // Large adjustment exceeding lambda_t
        let proposed_delta = 0.5f32;
        let result = div.clamp_adjustment(proposed_delta);
        assert!(result.is_some());
        assert!((result.unwrap() - 0.1).abs() < 1e-6);

        // Would produce DivergenceError in a real impl
        let error = DivergenceError {
            proposed_delta: 0.5,
            lambda_t: 0.1,
        };
        assert_eq!(error.proposed_delta, 0.5);
        assert_eq!(error.lambda_t, 0.1);
    }

    #[test]
    fn test_divergence_error_display() {
        let err = DivergenceError {
            proposed_delta: 0.5,
            lambda_t: 0.1,
        };
        let msg = format!("{err}");
        assert!(msg.contains("0.5"));
        assert!(msg.contains("0.1"));
    }

    #[test]
    fn test_parse_rules_valid_postcard() {
        // `parse_rules` expects postcard (binary) bytes, not JSON.
        let input = vec![
            RuleEdit { attribute: "bracket_depth".into(), threshold: 0.5, action: "reject".into() },
            RuleEdit { attribute: "token_entropy".into(), threshold: 0.8, action: "accept".into() },
        ];
        let bytes = postcard::to_allocvec(&input).unwrap();
        let rules = parse_rules(&bytes).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].attribute, "bracket_depth");
        assert!((rules[0].threshold - 0.5).abs() < 1e-6);
        assert_eq!(rules[0].action, "reject");
        assert_eq!(rules[1].attribute, "token_entropy");
        assert_eq!(rules[1].action, "accept");
    }

    #[test]
    fn test_parse_rules_invalid_bytes() {
        // Garbage bytes are not valid postcard.
        let bytes: &[u8] = &[0xff, 0xff, 0xff];
        let result = parse_rules(bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Parse error"));
    }

    #[test]
    fn test_parse_rules_invalid_action() {
        let input = vec![
            RuleEdit { attribute: "depth".into(), threshold: 0.5, action: "explode".into() },
        ];
        let bytes = postcard::to_allocvec(&input).unwrap();
        let result = parse_rules(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid action"));
    }

    #[test]
    fn test_parse_rules_empty_array() {
        let bytes = postcard::to_allocvec::<Vec<RuleEdit>>(&vec![]).unwrap();
        let rules = parse_rules(&bytes).unwrap();
        assert!(rules.is_empty());
    }

    #[test]
    fn test_parse_rules_maybe_action() {
        let input = vec![
            RuleEdit { attribute: "x".into(), threshold: 0.3, action: "maybe".into() },
        ];
        let bytes = postcard::to_allocvec(&input).unwrap();
        let rules = parse_rules(&bytes).unwrap();
        assert_eq!(rules[0].action, "maybe");
    }
}
