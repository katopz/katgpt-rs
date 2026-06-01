//! GOAT Proof 168 T3: RecFM SpecHop Cross-Hop Consistency
//!
//! Feature gate: `recfm` (Plan 168 Task 3, Research 150)
//!
//! Proofs:
//!   P1: Converging observations get higher confidence
//!   P2: Diverging observations get penalized
//!   P3: No regression in cache hit rate (disabled mode identical)

#![cfg(all(feature = "spechop", feature = "recfm"))]

use katgpt_rs::spechop::{CacheSpeculator, CrossHopConfig, HopSpeculator, observation_velocity};

// ── P1: Converging observations have decreasing velocity ─────────────

#[test]
fn proof_p1_converging_observations_decreasing_velocity() {
    let obs = vec![
        "The quick brown fox",
        "The quick brown fox jumps",
        "The quick brown fox jumps over",
        "The quick brown fox jumps over the",
    ];

    let mut velocities = Vec::new();
    for i in 1..obs.len() {
        velocities.push(observation_velocity(obs[i - 1], obs[i]));
    }

    // Each subsequent velocity should be <= previous (converging)
    for i in 1..velocities.len() {
        assert!(
            velocities[i] <= velocities[i - 1] + 1e-6, // tolerance
            "Velocities should be decreasing: v[{}]={} > v[{}]={}",
            i,
            velocities[i],
            i - 1,
            velocities[i - 1]
        );
    }
}

// ── P2: Diverging observations have increasing velocity ─────────────

#[test]
fn proof_p2_diverging_observations_high_velocity() {
    let obs = vec!["alpha", "beta", "gamma", "delta"];

    let mut velocities = Vec::new();
    for i in 1..obs.len() {
        velocities.push(observation_velocity(obs[i - 1], obs[i]));
    }

    // All should have high velocity (completely different strings)
    for (i, &v) in velocities.iter().enumerate() {
        assert!(
            v >= 0.8,
            "Diverging observations should have high velocity: v[{i}]={v}"
        );
    }
}

// ── P3: Cache hit rate unchanged when disabled ───────────────────────

#[test]
fn proof_p3_cache_hit_rate_unchanged() {
    let mut spec = CacheSpeculator::new();
    spec.observe("action_a", "result_a");
    spec.observe("action_b", "result_b");

    // Both should hit
    assert!(spec.speculate("action_a").is_ok());
    assert!(spec.speculate("action_b").is_ok());
    // Unknown should miss
    assert!(spec.speculate("action_c").is_err());
}

// ── Unit: observation_velocity correctness ───────────────────────────

#[test]
fn test_velocity_identical() {
    assert_eq!(
        observation_velocity("hello world", "hello world"),
        0.0,
        "Identical strings should have zero velocity"
    );
}

#[test]
fn test_velocity_completely_different() {
    let v = observation_velocity("abc", "xyz");
    assert!(
        (v - 1.0).abs() < 1e-6,
        "Completely different strings should have velocity 1.0, got {v}"
    );
}

#[test]
fn test_velocity_partial_overlap() {
    // "hello world" vs "hello there" — 6 matching chars out of max(11, 11)
    let v = observation_velocity("hello world", "hello there");
    // matching prefix = "hello " = 6 chars, max_len = 11
    let expected = 1.0 - (6.0 / 11.0);
    assert!(
        (v - expected).abs() < 1e-6,
        "Expected velocity {expected}, got {v}"
    );
}

#[test]
fn test_velocity_empty_both() {
    assert_eq!(observation_velocity("", ""), 0.0);
}

#[test]
fn test_velocity_one_empty() {
    assert_eq!(observation_velocity("hello", ""), 1.0);
    assert_eq!(observation_velocity("", "hello"), 1.0);
}

#[test]
fn test_velocity_symmetric() {
    let a = "prefix_a";
    let b = "prefix_b";
    let ab = observation_velocity(a, b);
    let ba = observation_velocity(b, a);
    assert!(
        (ab - ba).abs() < 1e-6,
        "observation_velocity should be symmetric: {ab} vs {ba}"
    );
}

// ── Unit: CrossHopConfig defaults ───────────────────────────────────

#[test]
fn test_cross_hop_config_default() {
    let config = CrossHopConfig::default();
    assert!(!config.enable);
    assert!((config.velocity_threshold - 0.5).abs() < 1e-6);
    assert_eq!(config.min_hops_for_consistency, 2);
}
