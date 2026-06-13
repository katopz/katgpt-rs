//! Integration tests for the rt_turbo module.
//!
//! Tests the combined workflow: synthetic attention generation → calibration →
//! classification → validation → serialization roundtrip.

use super::calibration::{
    CalibrationConfigSnapshot, HeadCalibration, HeadClassification, calibrate_from_scores,
    compute_all_retrieval_scores, compute_retrieval_score,
};
use crate::types::{RetrievalHeadRole, RtTurboConfig};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_config() -> RtTurboConfig {
    RtTurboConfig::default()
}

/// Create a seq_len × seq_len attention matrix with uniform values.
fn uniform_attention(seq_len: usize, value: f32) -> Vec<f32> {
    vec![value; seq_len * seq_len]
}

/// Create a diagonal attention matrix (each position attends only to itself).
fn diagonal_attention(seq_len: usize) -> Vec<f32> {
    let mut attn = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        attn[i * seq_len + i] = 1.0;
    }
    attn
}

/// Create an attention matrix where post-needle positions attend strongly
/// to pre-needle positions — simulating retrieval behavior.
fn retrieval_attention(seq_len: usize, needle_len: usize, strength: f32) -> Vec<f32> {
    let mut attn = vec![0.0f32; seq_len * seq_len];
    let post_start = seq_len - needle_len;
    for t in post_start..seq_len {
        let row_off = t * seq_len;
        let val = strength / needle_len as f32;
        for j in 0..needle_len {
            attn[row_off + j] = val;
        }
        // Also give some local attention
        let local_val = (1.0 - strength) / (seq_len as f32);
        for j in 0..seq_len {
            attn[row_off + j] += local_val;
        }
    }
    attn
}

/// Create an attention matrix where post-needle positions attend only locally
/// — simulating local behavior.
fn local_attention(seq_len: usize, window: usize) -> Vec<f32> {
    let mut attn = vec![0.0f32; seq_len * seq_len];
    for t in 0..seq_len {
        let row_off = t * seq_len;
        let start = t.saturating_sub(window);
        let count = t - start + 1;
        let val = 1.0 / count as f32;
        for j in start..=t {
            attn[row_off + j] = val;
        }
    }
    attn
}

/// Generate per-head attention matrices with known retrieval/local behavior.
///
/// First `n_retrieval` heads have retrieval patterns, rest have local patterns.
fn generate_mixed_heads(
    seq_len: usize,
    needle_len: usize,
    n_retrieval: usize,
    n_local: usize,
) -> Vec<Vec<f32>> {
    let mut heads = Vec::with_capacity(n_retrieval + n_local);
    for i in 0..n_retrieval {
        let strength = 0.5 + 0.5 * (i as f32 / n_retrieval.max(1) as f32);
        heads.push(retrieval_attention(seq_len, needle_len, strength));
    }
    for _ in 0..n_local {
        heads.push(local_attention(seq_len, 4));
    }
    heads
}

// ---------------------------------------------------------------------------
// Calibration Integration Tests
// ---------------------------------------------------------------------------

#[test]
fn test_mixed_heads_correctly_classified() {
    let config = RtTurboConfig {
        retrieval_head_ratio: 0.3,
        ..RtTurboConfig::default()
    };
    let seq_len = 20;
    let needle_len = 3;

    // 10 heads: first 3 are retrieval pattern, rest are local
    let per_head = generate_mixed_heads(seq_len, needle_len, 3, 7);

    let calibration = HeadCalibration::from_attention(
        &per_head,
        seq_len,
        0,
        needle_len,
        seq_len - needle_len,
        seq_len,
        &config,
    );

    // With ratio 0.3 and 10 heads, ceil(3.0) = 3 retrieval heads
    assert_eq!(calibration.n_heads(), 10);
    assert_eq!(calibration.n_retrieval(), 3, "Expected 3 retrieval heads");

    // Validate consistency
    assert!(calibration.validate().is_ok());

    // Retrieval heads should be the first 3 (higher retrieval scores)
    for i in 0..3 {
        assert!(
            calibration.is_retrieval(i),
            "Head {i} should be retrieval (retrieval pattern)"
        );
    }
}

#[test]
fn test_all_retrieval_heads() {
    let config = RtTurboConfig {
        retrieval_head_ratio: 1.0,
        ..RtTurboConfig::default()
    };
    let seq_len = 10;
    let needle_len = 2;

    let per_head = generate_mixed_heads(seq_len, needle_len, 4, 0);
    let calibration = HeadCalibration::from_attention(
        &per_head,
        seq_len,
        0,
        needle_len,
        seq_len - needle_len,
        seq_len,
        &config,
    );

    assert_eq!(
        calibration.n_retrieval(),
        4,
        "All heads should be retrieval"
    );
    assert_eq!(calibration.n_local(), 0);
    assert!(calibration.validate().is_ok());
}

#[test]
fn test_all_local_heads() {
    let config = RtTurboConfig {
        retrieval_head_ratio: 0.0,
        ..RtTurboConfig::default()
    };
    let seq_len = 10;
    let needle_len = 2;

    let per_head = generate_mixed_heads(seq_len, needle_len, 0, 4);
    let calibration = HeadCalibration::from_attention(
        &per_head,
        seq_len,
        0,
        needle_len,
        seq_len - needle_len,
        seq_len,
        &config,
    );

    // ratio 0.0 → max(1, ceil(0)) = 1 head forced retrieval
    // This is a safety measure: at least 1 head must be retrieval
    assert!(
        calibration.n_retrieval() >= 1,
        "At least 1 head must be retrieval"
    );
    assert!(calibration.validate().is_ok());
}

#[test]
fn test_single_head_edge_case() {
    let config = default_config();
    let seq_len = 8;
    let needle_len = 2;
    let per_head = vec![retrieval_attention(seq_len, needle_len, 0.8)];

    let calibration = HeadCalibration::from_attention(
        &per_head,
        seq_len,
        0,
        needle_len,
        seq_len - needle_len,
        seq_len,
        &config,
    );

    assert_eq!(calibration.n_heads(), 1);
    assert_eq!(
        calibration.n_retrieval(),
        1,
        "Single head must be retrieval"
    );
    assert_eq!(calibration.n_local(), 0);
    assert!(calibration.is_retrieval(0));
    assert!(calibration.validate().is_ok());
}

#[test]
fn test_retrieval_scores_ordering() {
    let config = RtTurboConfig {
        retrieval_head_ratio: 0.5,
        ..RtTurboConfig::default()
    };

    // Manually crafted scores: heads 0,3 have high scores, heads 1,2 have low
    let scores = vec![0.9f32, 0.1, 0.2, 0.8];
    let calibration = calibrate_from_scores(&scores, &config);

    // Top 2 by score: heads 0 (0.9) and 3 (0.8)
    assert!(
        calibration.is_retrieval(0),
        "Head 0 (score 0.9) should be retrieval"
    );
    assert!(
        calibration.is_retrieval(3),
        "Head 3 (score 0.8) should be retrieval"
    );
    assert!(
        !calibration.is_retrieval(1),
        "Head 1 (score 0.1) should be local"
    );
    assert!(
        !calibration.is_retrieval(2),
        "Head 2 (score 0.2) should be local"
    );

    // Retrieval set should be sorted by score descending: [0, 3]
    assert_eq!(calibration.retrieval_set[0], 0);
    assert_eq!(calibration.retrieval_set[1], 3);
}

// ---------------------------------------------------------------------------
// Compute Retrieval Score Tests
// ---------------------------------------------------------------------------

#[test]
fn test_retrieval_score_high_for_retrieval_pattern() {
    let seq_len = 20;
    let needle_len = 3;
    let attn = retrieval_attention(seq_len, needle_len, 0.9);

    let score =
        compute_retrieval_score(&attn, seq_len, 0, needle_len, seq_len - needle_len, seq_len);

    assert!(
        score > 0.3,
        "Retrieval pattern should have high score, got {score}"
    );
}

#[test]
fn test_retrieval_score_low_for_local_pattern() {
    let seq_len = 20;
    let needle_len = 3;
    let attn = local_attention(seq_len, 4);

    let score =
        compute_retrieval_score(&attn, seq_len, 0, needle_len, seq_len - needle_len, seq_len);

    assert!(
        score < 0.1,
        "Local pattern should have low retrieval score, got {score}"
    );
}

#[test]
fn test_retrieval_score_zero_for_diagonal() {
    let seq_len = 10;
    let attn = diagonal_attention(seq_len);

    let score = compute_retrieval_score(&attn, seq_len, 0, 3, 7, 10);

    assert!(
        score < 0.01,
        "Diagonal attention should have near-zero score, got {score}"
    );
}

#[test]
fn test_compute_all_scores_matches_individual() {
    let seq_len = 10;
    let needle_len = 2;
    let heads = vec![
        retrieval_attention(seq_len, needle_len, 0.8),
        diagonal_attention(seq_len),
        uniform_attention(seq_len, 0.1),
    ];

    let all_scores = compute_all_retrieval_scores(
        &heads,
        seq_len,
        0,
        needle_len,
        seq_len - needle_len,
        seq_len,
    );

    assert_eq!(all_scores.len(), 3);

    // Verify each matches individual computation
    for (i, head_attn) in heads.iter().enumerate() {
        let individual = compute_retrieval_score(
            head_attn,
            seq_len,
            0,
            needle_len,
            seq_len - needle_len,
            seq_len,
        );
        assert!(
            (all_scores[i] - individual).abs() < 1e-6,
            "Head {i}: batch score {} != individual {}",
            all_scores[i],
            individual
        );
    }
}

// ---------------------------------------------------------------------------
// Serialization Roundtrip Tests
// ---------------------------------------------------------------------------

#[test]
fn test_binary_roundtrip_preserves_classification() {
    let config = RtTurboConfig {
        retrieval_head_ratio: 0.25,
        ..RtTurboConfig::default()
    };
    let scores: Vec<f32> = vec![0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6];
    let original = calibrate_from_scores(&scores, &config);

    let bytes = original.to_bytes().expect("Serialize");
    let loaded = HeadCalibration::from_bytes(&bytes).expect("Deserialize");

    // Verify structure
    assert_eq!(loaded.n_heads(), original.n_heads());
    assert_eq!(loaded.n_retrieval(), original.n_retrieval());
    assert_eq!(loaded.n_local(), original.n_local());
    assert_eq!(loaded.retrieval_set, original.retrieval_set);
    assert_eq!(loaded.local_set, original.local_set);
    assert!((loaded.threshold - original.threshold).abs() < 1e-6);

    // Verify per-head classifications
    for (orig, loaded_c) in original
        .classifications
        .iter()
        .zip(loaded.classifications.iter())
    {
        assert_eq!(orig.head_idx, loaded_c.head_idx);
        assert_eq!(orig.role, loaded_c.role);
        assert!((orig.score - loaded_c.score).abs() < 1e-6);
    }

    assert!(loaded.validate().is_ok());
}

#[test]
fn test_file_roundtrip_preserves_data() {
    let dir = tempfile::tempdir().expect("Create temp dir");
    let path = dir.path().join("rt_turbo_calibration.bin");

    let config = RtTurboConfig {
        retrieval_head_ratio: 0.2,
        ..RtTurboConfig::default()
    };
    let scores: Vec<f32> = vec![0.1, 0.9, 0.2, 0.8, 0.3];
    let original = calibrate_from_scores(&scores, &config);

    original.save(&path).expect("Save calibration");
    let loaded = HeadCalibration::load(&path).expect("Load calibration");

    assert_eq!(loaded.retrieval_set, original.retrieval_set);
    assert_eq!(loaded.local_set, original.local_set);
    assert!((loaded.threshold - original.threshold).abs() < 1e-6);
    assert_eq!(
        loaded.config_snapshot, original.config_snapshot,
        "Config snapshot should be preserved"
    );
}

// ---------------------------------------------------------------------------
// Config Snapshot Tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_snapshot_captured() {
    let config = RtTurboConfig {
        retrieval_head_ratio: 0.17,
        ..RtTurboConfig::default()
    };
    let scores = vec![0.5f32; 12];
    let calibration = calibrate_from_scores(&scores, &config);

    assert_eq!(calibration.config_snapshot.retrieval_head_ratio, 0.17);
    assert_eq!(calibration.config_snapshot.n_query_heads, 12);
}

#[test]
fn test_config_snapshot_preserved_through_serialization() {
    let config = RtTurboConfig {
        retrieval_head_ratio: 0.22,
        ..RtTurboConfig::default()
    };
    let scores = vec![0.5f32; 6];
    let original = calibrate_from_scores(&scores, &config);

    let bytes = original.to_bytes().expect("Serialize");
    let loaded = HeadCalibration::from_bytes(&bytes).expect("Deserialize");

    assert_eq!(loaded.config_snapshot, original.config_snapshot);
}

// ---------------------------------------------------------------------------
// Role Accessor Tests
// ---------------------------------------------------------------------------

#[test]
fn test_role_of_and_score_of() {
    let config = RtTurboConfig {
        retrieval_head_ratio: 0.4,
        ..RtTurboConfig::default()
    };
    let scores: Vec<f32> = vec![0.1, 0.9, 0.2, 0.8, 0.3];
    let calibration = calibrate_from_scores(&scores, &config);

    // Head 1 (score 0.9) → Retrieval
    assert_eq!(calibration.role_of(1), RetrievalHeadRole::Retrieval);
    assert!((calibration.score_of(1) - 0.9).abs() < 1e-6);

    // Head 3 (score 0.8) → Retrieval
    assert_eq!(calibration.role_of(3), RetrievalHeadRole::Retrieval);
    assert!((calibration.score_of(3) - 0.8).abs() < 1e-6);

    // Head 0 (score 0.1) → Local
    assert_eq!(calibration.role_of(0), RetrievalHeadRole::Local);
    assert!((calibration.score_of(0) - 0.1).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// All-Local Fallback Tests
// ---------------------------------------------------------------------------

#[test]
fn test_all_local_fallback_no_retrieval() {
    let config = default_config();
    let calibration = HeadCalibration::all_local(16, &config);

    assert_eq!(calibration.n_retrieval(), 0);
    assert_eq!(calibration.n_local(), 16);
    assert_eq!(calibration.n_heads(), 16);

    for i in 0..16 {
        assert!(!calibration.is_retrieval(i), "Head {i} should be local");
        assert_eq!(calibration.role_of(i), RetrievalHeadRole::Local);
        assert!((calibration.score_of(i)).abs() < 1e-6);
    }

    assert!(calibration.validate().is_ok());
}

#[test]
fn test_all_local_serialization_roundtrip() {
    let config = default_config();
    let original = HeadCalibration::all_local(8, &config);

    let bytes = original.to_bytes().expect("Serialize");
    let loaded = HeadCalibration::from_bytes(&bytes).expect("Deserialize");

    assert_eq!(loaded.n_retrieval(), 0);
    assert_eq!(loaded.n_local(), 8);
    assert!(loaded.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Validation Tests
// ---------------------------------------------------------------------------

#[test]
fn test_validate_catches_inconsistent_roles() {
    let calibration = HeadCalibration {
        classifications: vec![
            HeadClassification {
                head_idx: 0,
                role: RetrievalHeadRole::Local,
                score: 0.9,
            },
            HeadClassification {
                head_idx: 1,
                role: RetrievalHeadRole::Retrieval,
                score: 0.1,
            },
        ],
        retrieval_set: vec![0], // Wrong! Head 0 is classified as Local
        local_set: vec![1],
        threshold: 0.9,
        config_snapshot: CalibrationConfigSnapshot {
            retrieval_head_ratio: 0.5,
            n_query_heads: 2,
        },
    };

    let result = calibration.validate();
    assert!(result.is_err(), "Should catch inconsistent role assignment");
    assert!(
        result
            .unwrap_err()
            .contains("in retrieval_set but classified as Local"),
        "Error message should describe the inconsistency"
    );
}

#[test]
fn test_validate_catches_incomplete_partition() {
    let calibration = HeadCalibration {
        classifications: vec![
            HeadClassification {
                head_idx: 0,
                role: RetrievalHeadRole::Retrieval,
                score: 0.9,
            },
            HeadClassification {
                head_idx: 1,
                role: RetrievalHeadRole::Local,
                score: 0.1,
            },
        ],
        retrieval_set: vec![0],
        local_set: vec![], // Missing head 1
        threshold: 0.9,
        config_snapshot: CalibrationConfigSnapshot {
            retrieval_head_ratio: 0.5,
            n_query_heads: 2,
        },
    };

    let result = calibration.validate();
    assert!(result.is_err(), "Should catch incomplete partition");
    assert!(
        result.unwrap_err().contains("Partition incomplete"),
        "Error message should describe partition issue"
    );
}

#[test]
fn test_validate_catches_duplicate_indices() {
    // Head 1 appears twice in retrieval_set — passes role checks but fails overlap check.
    // n=3 heads, all Retrieval. retrieval_set=[0, 1, 1] (duplicate), local_set=[].
    // Step 1 (retrieval role): OK. Step 2 (local role): OK (empty). Step 3 (total=3=n): OK.
    // Step 4 (overlaps): all_indices=[0,1,1] → duplicate detected.
    let calibration = HeadCalibration {
        classifications: vec![
            HeadClassification {
                head_idx: 0,
                role: RetrievalHeadRole::Retrieval,
                score: 0.9,
            },
            HeadClassification {
                head_idx: 1,
                role: RetrievalHeadRole::Retrieval,
                score: 0.8,
            },
            HeadClassification {
                head_idx: 2,
                role: RetrievalHeadRole::Retrieval,
                score: 0.7,
            },
        ],
        retrieval_set: vec![0, 1, 1], // Head 1 duplicated within same set
        local_set: vec![],
        threshold: 0.7,
        config_snapshot: CalibrationConfigSnapshot {
            retrieval_head_ratio: 1.0,
            n_query_heads: 3,
        },
    };

    let result = calibration.validate();
    assert!(result.is_err(), "Should catch duplicate index");
    assert!(
        result.unwrap_err().contains("Duplicate"),
        "Error message should describe duplicate"
    );
}

// ---------------------------------------------------------------------------
// Large Head Count Tests
// ---------------------------------------------------------------------------

#[test]
fn test_large_head_count_calibration() {
    let config = RtTurboConfig {
        retrieval_head_ratio: 0.15,
        ..RtTurboConfig::default()
    };

    // Simulate 32 heads with varying retrieval scores
    let scores: Vec<f32> = (0..32)
        .map(|i| {
            // First 5 heads have high retrieval scores, rest have low
            if i < 5 {
                0.5 + i as f32 * 0.1
            } else {
                0.01 + (i - 5) as f32 * 0.005
            }
        })
        .collect();

    let calibration = calibrate_from_scores(&scores, &config);

    // ceil(32 * 0.15) = ceil(4.8) = 5 retrieval heads
    assert_eq!(calibration.n_retrieval(), 5, "Expected 5 retrieval heads");
    assert_eq!(calibration.n_local(), 27);
    assert!(calibration.validate().is_ok());

    // All high-score heads should be retrieval
    for i in 0..5 {
        assert!(calibration.is_retrieval(i), "Head {i} should be retrieval");
    }
}

// ---------------------------------------------------------------------------
// Default Config Tests
// ---------------------------------------------------------------------------

#[test]
fn test_default_config_values() {
    let config = RtTurboConfig::default();

    assert!((config.retrieval_head_ratio - 0.15).abs() < 1e-6);
    assert_eq!(config.low_dim, 16);
    assert!((config.top_p - 0.9).abs() < 1e-6);
    assert_eq!(config.sliding_window, 8192);
    assert_eq!(config.sink_tokens, 4);
    assert_eq!(config.block_size, 64);
}

#[test]
fn test_retrieval_head_role_default_is_local() {
    assert_eq!(RetrievalHeadRole::default(), RetrievalHeadRole::Local);
}
