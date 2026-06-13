//! SpecProof — BLAKE3 commitment and verification for compiled spec rules.
//!
//! Provides tamper detection: hash all compiled rule data into a single
//! BLAKE3 commitment. Verification recomputes the hash and checks for
//! divergence — any mutation to rules, bitmaps, or prefix entries is
//! detected.
//!
//! Feature gate: `spec_pruner`

use std::time::{SystemTime, UNIX_EPOCH};

use blake3::Hasher;

use super::types::{CompactBitmap, CompiledSpec, SpecRule};

/// Cryptographic proof that a compiled spec hasn't been tampered with.
///
/// The commitment covers all rule data (depth, is_allowlist, bitmap bytes,
/// prefix entries) plus the original spec hash. Verification recomputes
/// the commitment from the live spec and compares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecProof {
    /// BLAKE3 hash of the compiled spec rules (not just the spec string).
    pub commitment: [u8; 32],
    /// Original spec source string for verification.
    pub spec_source: String,
    /// Number of rules in the compiled spec.
    pub rule_count: usize,
    /// Sum of allowed bitmap cardinalities across all rules.
    pub total_allowed: usize,
    /// Sum of blocked bitmap cardinalities across all rules.
    pub total_blocked: usize,
}

/// Timestamped commitment for audit trails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecCommitment {
    /// The proof itself.
    pub proof: SpecProof,
    /// Unix epoch milliseconds when this commitment was created.
    pub timestamp: u64,
}

impl SpecProof {
    /// Create a proof from a compiled spec and its original source string.
    ///
    /// Hashes all rule data through BLAKE3: depth, is_allowlist, bitmap
    /// bytes, prefix entries, plus the existing spec_hash.
    pub fn from_spec(spec: &CompiledSpec, source: &str) -> Self {
        let commitment = compute_commitment(spec);
        let mut total_allowed = 0usize;
        let mut total_blocked = 0usize;
        for rule in &spec.rules {
            if rule.is_allowlist {
                total_allowed += rule.allowed.len();
            } else {
                total_blocked += rule.allowed.len();
            }
        }
        SpecProof {
            commitment,
            spec_source: source.to_owned(),
            rule_count: spec.rules.len(),
            total_allowed,
            total_blocked,
        }
    }

    /// Verify that the compiled spec matches this proof's commitment.
    ///
    /// Recomputes the BLAKE3 commitment from the spec and compares.
    /// Returns `true` if the spec hasn't been tampered with.
    pub fn verify(&self, spec: &CompiledSpec) -> bool {
        let recomputed = compute_commitment(spec);
        self.commitment == recomputed
    }

    /// Verify that the given source string matches the stored spec source.
    ///
    /// Simple string equality check. For a stricter check, compare the
    /// BLAKE3 hash of the source against the spec's `spec_hash`.
    pub fn verify_source(&self, source: &str) -> bool {
        self.spec_source == source
    }

    /// Verify that the source hashes to the same value as the spec's `spec_hash`.
    pub fn verify_source_hash(&self, spec: &CompiledSpec) -> bool {
        let source_hash = blake3::hash(self.spec_source.as_bytes());
        source_hash == spec.spec_hash
    }
}

impl SpecCommitment {
    /// Create a new timestamped commitment from a proof.
    pub fn new(proof: SpecProof) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        SpecCommitment { proof, timestamp }
    }

    /// Create a commitment from a compiled spec and its source.
    pub fn from_spec(spec: &CompiledSpec, source: &str) -> Self {
        let proof = SpecProof::from_spec(spec, source);
        Self::new(proof)
    }

    /// Verify the enclosed proof against a compiled spec.
    pub fn verify(&self, spec: &CompiledSpec) -> bool {
        self.proof.verify(spec)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the BLAKE3 commitment over all compiled rule data.
fn compute_commitment(spec: &CompiledSpec) -> [u8; 32] {
    let mut hasher = Hasher::new();

    // Include the existing spec_hash as domain separator / pre-image.
    hasher.update(&spec.spec_hash);

    // Hash vocab_size so changes to bitmap range are detected.
    hasher.update(&spec.vocab_size.to_le_bytes());

    // Hash global bitmaps.
    hash_bitmap(&mut hasher, &spec.global_allowed);
    hash_bitmap(&mut hasher, &spec.global_blocked);

    // Hash each rule in order.
    for rule in &spec.rules {
        hash_rule(&mut hasher, rule);
    }

    *hasher.finalize().as_bytes()
}

/// Feed a single rule into the hasher.
fn hash_rule(hasher: &mut Hasher, rule: &SpecRule) {
    // Depth: None → [0u8], Some(n) → [1u8, n.to_le_bytes()]
    match rule.depth {
        None => {
            hasher.update(&[0u8]);
        }
        Some(d) => {
            hasher.update(&[1u8]);
            hasher.update(&(d as u64).to_le_bytes());
        }
    }

    // is_allowlist flag.
    hasher.update(&[rule.is_allowlist as u8]);

    // Allowed bitmap.
    hash_bitmap(hasher, &rule.allowed);

    // Prefix entries (ordered).
    hasher.update(&(rule.prefix.len() as u64).to_le_bytes());
    for entry in &rule.prefix {
        hasher.update(&(entry.depth as u64).to_le_bytes());
        hasher.update(&(entry.token_idx as u64).to_le_bytes());
    }
}

/// Feed a compact bitmap into the hasher in a deterministic encoding.
fn hash_bitmap(hasher: &mut Hasher, bitmap: &CompactBitmap) {
    match bitmap {
        CompactBitmap::Empty => {
            hasher.update(&[0u8]);
        }
        CompactBitmap::Sparse(indices) => {
            hasher.update(&[1u8]);
            hasher.update(&(indices.len() as u64).to_le_bytes());
            // u16 array — each is 2 bytes, little-endian on LE platforms.
            for &idx in indices {
                hasher.update(&idx.to_le_bytes());
            }
        }
        CompactBitmap::Dense(bits) => {
            hasher.update(&[2u8]);
            // 1024 × u64 = 8192 bytes, fully deterministic.
            for &word in bits.iter() {
                hasher.update(&word.to_le_bytes());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::spec_compile::types::{CompactBitmap, CompiledSpec, PrefixEntry, SpecRule};

    /// Build a minimal compiled spec for testing.
    fn test_spec() -> CompiledSpec {
        let spec_str = "Classify sentiment as positive or negative";
        let spec_hash = blake3::hash(spec_str.as_bytes()).into();

        let rules = vec![
            SpecRule {
                depth: Some(0),
                prefix: vec![],
                allowed: CompactBitmap::from_sorted_indices(vec![100, 200, 300]),
                is_allowlist: true,
            },
            SpecRule {
                depth: Some(1),
                prefix: vec![PrefixEntry {
                    depth: 0,
                    token_idx: 100,
                }],
                allowed: CompactBitmap::from_sorted_indices(vec![10, 20]),
                is_allowlist: true,
            },
            SpecRule {
                depth: None,
                prefix: vec![],
                allowed: CompactBitmap::from_sorted_indices(vec![999]),
                is_allowlist: false,
            },
        ];

        CompiledSpec {
            spec_hash,
            rules,
            vocab_size: 32000,
            global_allowed: CompactBitmap::from_sorted_indices(vec![0, 1, 2]),
            global_blocked: CompactBitmap::empty(),
        }
    }

    const SPEC_SOURCE: &str = "Classify sentiment as positive or negative";

    #[test]
    fn proof_creation_from_compiled_spec() {
        let spec = test_spec();
        let proof = SpecProof::from_spec(&spec, SPEC_SOURCE);

        assert_ne!(proof.commitment, [0u8; 32], "commitment should not be zero");
        assert_eq!(proof.rule_count, 3);
        assert_eq!(proof.spec_source, SPEC_SOURCE);
        // Rule 0: allowlist, 3 tokens; Rule 1: allowlist, 2 tokens.
        assert_eq!(proof.total_allowed, 5);
        // Rule 2: blocklist, 1 token.
        assert_eq!(proof.total_blocked, 1);
    }

    #[test]
    fn verification_passes_on_unmodified_spec() {
        let spec = test_spec();
        let proof = SpecProof::from_spec(&spec, SPEC_SOURCE);
        assert!(proof.verify(&spec), "unmodified spec should verify");
    }

    #[test]
    fn verification_fails_on_tampered_rules() {
        let spec = test_spec();
        let proof = SpecProof::from_spec(&spec, SPEC_SOURCE);

        // Tamper: mutate a rule's bitmap.
        let mut tampered = spec.clone();
        tampered.rules[0].allowed = CompactBitmap::from_sorted_indices(vec![100, 200, 999]);

        assert!(
            !proof.verify(&tampered),
            "tampered spec should fail verification"
        );
    }

    #[test]
    fn verification_fails_on_tampered_depth() {
        let spec = test_spec();
        let proof = SpecProof::from_spec(&spec, SPEC_SOURCE);

        let mut tampered = spec.clone();
        tampered.rules[0].depth = Some(42);

        assert!(
            !proof.verify(&tampered),
            "tampered depth should fail verification"
        );
    }

    #[test]
    fn verification_fails_on_tampered_prefix() {
        let spec = test_spec();
        let proof = SpecProof::from_spec(&spec, SPEC_SOURCE);

        let mut tampered = spec.clone();
        tampered.rules[1].prefix.push(PrefixEntry {
            depth: 1,
            token_idx: 200,
        });

        assert!(
            !proof.verify(&tampered),
            "tampered prefix should fail verification"
        );
    }

    #[test]
    fn verification_fails_on_added_rule() {
        let spec = test_spec();
        let proof = SpecProof::from_spec(&spec, SPEC_SOURCE);

        let mut tampered = spec.clone();
        tampered.rules.push(SpecRule {
            depth: Some(2),
            prefix: vec![],
            allowed: CompactBitmap::empty(),
            is_allowlist: true,
        });

        assert!(
            !proof.verify(&tampered),
            "added rule should fail verification"
        );
    }

    #[test]
    fn verification_fails_on_vocab_size_change() {
        let spec = test_spec();
        let proof = SpecProof::from_spec(&spec, SPEC_SOURCE);

        let mut tampered = spec.clone();
        tampered.vocab_size = 50000;

        assert!(
            !proof.verify(&tampered),
            "vocab_size change should fail verification"
        );
    }

    #[test]
    fn verification_fails_on_global_bitmap_change() {
        let spec = test_spec();
        let proof = SpecProof::from_spec(&spec, SPEC_SOURCE);

        let mut tampered = spec.clone();
        tampered.global_blocked = CompactBitmap::from_sorted_indices(vec![999]);

        assert!(
            !proof.verify(&tampered),
            "global bitmap change should fail verification"
        );
    }

    #[test]
    fn source_verification_passes() {
        let spec = test_spec();
        let proof = SpecProof::from_spec(&spec, SPEC_SOURCE);
        assert!(proof.verify_source(SPEC_SOURCE));
    }

    #[test]
    fn source_verification_fails_on_mismatch() {
        let spec = test_spec();
        let proof = SpecProof::from_spec(&spec, SPEC_SOURCE);
        assert!(!proof.verify_source("Different spec string"));
    }

    #[test]
    fn source_hash_verification_passes() {
        let spec = test_spec();
        let proof = SpecProof::from_spec(&spec, SPEC_SOURCE);
        assert!(proof.verify_source_hash(&spec));
    }

    #[test]
    fn deterministic_proof_creation() {
        let spec = test_spec();
        let proof1 = SpecProof::from_spec(&spec, SPEC_SOURCE);
        let proof2 = SpecProof::from_spec(&spec, SPEC_SOURCE);

        assert_eq!(
            proof1.commitment, proof2.commitment,
            "same spec must produce same commitment"
        );
        assert_eq!(proof1, proof2, "proofs should be fully equal");
    }

    #[test]
    fn commitment_changes_with_source() {
        let spec = test_spec();
        let proof_a = SpecProof::from_spec(&spec, "spec A");
        let proof_b = SpecProof::from_spec(&spec, "spec B");

        // Commitment is the same because it's computed from rules, not the
        // source string — but spec_source differs.
        assert_eq!(
            proof_a.commitment, proof_b.commitment,
            "same rules → same commitment"
        );
        assert_ne!(proof_a.spec_source, proof_b.spec_source);
    }

    #[test]
    fn spec_commitment_has_timestamp() {
        let spec = test_spec();
        let commitment = SpecCommitment::from_spec(&spec, SPEC_SOURCE);

        assert!(commitment.timestamp > 0, "timestamp should be non-zero");
        assert!(commitment.verify(&spec));
    }

    #[test]
    fn empty_rules_spec_proof() {
        let spec_hash = blake3::hash(b"empty spec").into();
        let spec = CompiledSpec {
            spec_hash,
            rules: vec![],
            vocab_size: 1000,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        };

        let proof = SpecProof::from_spec(&spec, "empty spec");
        assert_eq!(proof.rule_count, 0);
        assert_eq!(proof.total_allowed, 0);
        assert_eq!(proof.total_blocked, 0);
        assert!(proof.verify(&spec));
    }

    #[test]
    fn dense_bitmap_in_commitment() {
        // Build a spec with a dense bitmap (>4096 entries).
        let spec_hash = blake3::hash(b"dense spec").into();
        let indices: Vec<u16> = (0..5000u16).collect();
        let rules = vec![SpecRule {
            depth: None,
            prefix: vec![],
            allowed: CompactBitmap::from_sorted_indices(indices),
            is_allowlist: true,
        }];
        let spec = CompiledSpec {
            spec_hash,
            rules,
            vocab_size: 65536,
            global_allowed: CompactBitmap::empty(),
            global_blocked: CompactBitmap::empty(),
        };

        let proof = SpecProof::from_spec(&spec, "dense spec");
        assert_eq!(proof.total_allowed, 5000);
        assert!(proof.verify(&spec));

        // Tamper with the dense bitmap.
        let mut tampered = spec.clone();
        if let CompactBitmap::Dense(ref mut bits) = tampered.rules[0].allowed {
            bits[0] ^= 1; // flip one bit
        }
        assert!(!proof.verify(&tampered));
    }
}
