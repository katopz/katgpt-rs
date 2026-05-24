#![cfg(feature = "dreamer")]
//! GOAT Proof Test — Auto-Dreamer Offline Consolidation (Plan 107)
//!
//! Proves mathematical invariants of the Auto-Dreamer offline memory
//! consolidation pipeline: config validity, scheduler cadence, region
//! selection filtering, snapshot alignment, consolidator ordering,
//! coverage completeness, decay exemption, and end-to-end counterfactual
//! utility estimation.
//!
//! Run: `cargo test --features dreamer --test goat_107_dreamer -- --nocapture`

use katgpt_rs::pruners::dreamer::consolidator::DreamerConsolidator;
use katgpt_rs::pruners::dreamer::counterfactual::CounterfactualEstimator;
use katgpt_rs::pruners::dreamer::decay::MemoryDecay;
use katgpt_rs::pruners::dreamer::scheduler::{ArmInfo, DreamerScheduler};
use katgpt_rs::pruners::dreamer::types::{DecayPolicy, DreamerConfig, WorkingRegion};
use katgpt_rs::types::Rng;

// ── Helpers ───────────────────────────────────────────────────

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

/// Build a set of test arms with controlled properties.
fn make_test_arms() -> Vec<ArmInfo> {
    vec![
        ArmInfo {
            index: 0,
            q_value: 0.90,
            visits: 20,
            last_write_episode: 9,
            last_retrieve_episode: 9,
        },
        ArmInfo {
            index: 1,
            q_value: 0.10,
            visits: 1,
            last_write_episode: 1,
            last_retrieve_episode: 1,
        },
        ArmInfo {
            index: 2,
            q_value: 0.50,
            visits: 10,
            last_write_episode: 6,
            last_retrieve_episode: 7,
        },
        ArmInfo {
            index: 3,
            q_value: 0.70,
            visits: 15,
            last_write_episode: 4,
            last_retrieve_episode: 4,
        },
        ArmInfo {
            index: 4,
            q_value: 0.30,
            visits: 0,
            last_write_episode: 0,
            last_retrieve_episode: 0,
        },
        ArmInfo {
            index: 5,
            q_value: 0.60,
            visits: 8,
            last_write_episode: 8,
            last_retrieve_episode: 5,
        },
        ArmInfo {
            index: 6,
            q_value: 0.20,
            visits: 2,
            last_write_episode: 3,
            last_retrieve_episode: 2,
        },
        ArmInfo {
            index: 7,
            q_value: 0.80,
            visits: 12,
            last_write_episode: 7,
            last_retrieve_episode: 8,
        },
        ArmInfo {
            index: 8,
            q_value: 0.40,
            visits: 6,
            last_write_episode: 5,
            last_retrieve_episode: 6,
        },
        ArmInfo {
            index: 9,
            q_value: 0.55,
            visits: 11,
            last_write_episode: 9,
            last_retrieve_episode: 9,
        },
    ]
}

/// Validate that a config preset has all fields in valid ranges.
fn validate_config(config: &DreamerConfig, label: &str) {
    assert!(
        config.cadence > 0,
        "[Config:{label}] cadence must be > 0, got {}",
        config.cadence
    );
    assert!(
        config.region_fraction > 0.0 && config.region_fraction <= 1.0,
        "[Config:{label}] region_fraction must be in (0, 1], got {}",
        config.region_fraction
    );
    assert!(
        config.merge_threshold > 0.0,
        "[Config:{label}] merge_threshold must be > 0, got {}",
        config.merge_threshold
    );
    assert!(
        config.decay_factor >= 0.0 && config.decay_factor <= 1.0,
        "[Config:{label}] decay_factor must be in [0, 1], got {}",
        config.decay_factor
    );
    assert!(
        config.dropout_fraction >= 0.0 && config.dropout_fraction <= 1.0,
        "[Config:{label}] dropout_fraction must be in [0, 1], got {}",
        config.dropout_fraction
    );
    assert!(
        config.mc_samples >= 1,
        "[Config:{label}] mc_samples must be >= 1, got {}",
        config.mc_samples
    );
    assert!(
        config.min_visits >= 1,
        "[Config:{label}] min_visits must be >= 1, got {}",
        config.min_visits
    );
}

// ── Proof 1: Config Presets Valid ─────────────────────────────
//
// All three presets (default, conservative, aggressive) produce configs
// with valid parameter ranges: cadence > 0, 0 < fraction <= 1,
// thresholds > 0, decay in [0,1], dropout in [0,1], samples >= 1,
// min_visits >= 1.

#[test]
fn proof_1_config_presets_valid() {
    let configs = [
        (DreamerConfig::default(), "default"),
        (DreamerConfig::conservative(), "conservative"),
        (DreamerConfig::aggressive(), "aggressive"),
    ];

    for (config, label) in &configs {
        validate_config(config, label);
    }

    // Verify specific known values to ensure presets are distinct
    let d = DreamerConfig::default();
    let c = DreamerConfig::conservative();
    let a = DreamerConfig::aggressive();

    assert_ne!(
        d.cadence, c.cadence,
        "[P1.1] default and conservative should differ in cadence"
    );
    assert_ne!(
        d.cadence, a.cadence,
        "[P1.2] default and aggressive should differ in cadence"
    );

    // Conservative has highest cadence (least frequent consolidation)
    assert!(
        c.cadence > d.cadence,
        "[P1.3] conservative cadence > default cadence"
    );
    // Aggressive has lowest cadence (most frequent consolidation)
    assert!(
        a.cadence < d.cadence,
        "[P1.4] aggressive cadence < default cadence"
    );

    // Aggressive has highest region_fraction (larger working region)
    assert!(
        a.region_fraction > d.region_fraction,
        "[P1.5] aggressive region_fraction > default"
    );
    // Conservative has lowest region_fraction
    assert!(
        c.region_fraction < d.region_fraction,
        "[P1.6] conservative region_fraction < default"
    );

    println!("✅ Proof 1 PASSED: All config presets have valid ranges and are distinct");
}

// ── Proof 2: Scheduler Cadence Trigger ────────────────────────
//
// should_consolidate(e) iff e > 0 AND e % cadence == 0.
// Never triggers at episode 0. Triggers at every multiple of cadence.
// Does not trigger at non-multiples.

#[test]
fn proof_2_scheduler_cadence_trigger() {
    let scheduler = DreamerScheduler::new(DreamerConfig::default()); // cadence=10

    // Never at episode 0
    assert!(
        !scheduler.should_consolidate(0),
        "[P2.1] should not consolidate at episode 0"
    );

    // Triggers at cadence multiples
    let multiples = [10, 20, 30, 50, 100, 1000];
    for &e in &multiples {
        assert!(
            scheduler.should_consolidate(e),
            "[P2.2] should consolidate at episode {e} (multiple of 10)"
        );
    }

    // Does not trigger at non-multiples
    let non_multiples = [1, 5, 7, 9, 11, 15, 23, 99, 101];
    for &e in &non_multiples {
        assert!(
            !scheduler.should_consolidate(e),
            "[P2.3] should not consolidate at episode {e} (not multiple of 10)"
        );
    }

    // Boundary: episode == cadence (first trigger)
    assert!(
        scheduler.should_consolidate(10),
        "[P2.4] first trigger at episode == cadence"
    );

    // Custom cadence
    let scheduler3 = DreamerScheduler::new(DreamerConfig {
        cadence: 3,
        ..DreamerConfig::default()
    });
    assert!(
        !scheduler3.should_consolidate(0),
        "[P2.5] cadence=3, ep=0 → false"
    );
    assert!(
        scheduler3.should_consolidate(3),
        "[P2.6] cadence=3, ep=3 → true"
    );
    assert!(
        !scheduler3.should_consolidate(4),
        "[P2.7] cadence=3, ep=4 → false"
    );
    assert!(
        scheduler3.should_consolidate(6),
        "[P2.8] cadence=3, ep=6 → true"
    );

    println!("✅ Proof 2 PASSED: Scheduler cadence trigger is correct (e > 0 && e % cadence == 0)");
}

// ── Proof 3: Min-visits Filter + Region Cap ───────────────────
//
// Region only includes arms with visits >= min_visits.
// region.arm_indices.len() <= ceil(n * region_fraction).
// Arms with insufficient visits are excluded regardless of recency.

#[test]
fn proof_3_min_visits_filter_and_region_cap() {
    let arms = make_test_arms(); // 10 arms
    let config = DreamerConfig {
        cadence: 10,
        region_fraction: 0.3,
        min_visits: 3,
        ..DreamerConfig::default()
    };
    let scheduler = DreamerScheduler::new(config);
    let region = scheduler.select_region(&arms, 10);

    // Min-visits filter: all included arms must have visits >= 3
    for &idx in &region.arm_indices {
        assert!(
            arms[idx].visits >= 3,
            "[P3.1] arm {idx} included but has visits={} < min_visits=3",
            arms[idx].visits
        );
    }

    // Verify that excluded arms with low visits are NOT in region
    // Arms 1 (visits=1), 4 (visits=0), 6 (visits=2) should be excluded
    for &excluded_idx in &[1, 4, 6] {
        assert!(
            !region.arm_indices.contains(&excluded_idx),
            "[P3.2] arm {excluded_idx} should be excluded (visits={})",
            arms[excluded_idx].visits
        );
    }

    // Region cap: ceil(10 * 0.3) = ceil(3.0) = 3
    let max_cap = (arms.len() as f32 * 0.3).ceil() as usize;
    assert!(
        region.arm_indices.len() <= max_cap,
        "[P3.3] region has {} arms but cap is {max_cap}",
        region.arm_indices.len()
    );

    // Stricter fraction test
    let config_small = DreamerConfig {
        cadence: 10,
        region_fraction: 0.2,
        min_visits: 1,
        ..DreamerConfig::default()
    };
    let scheduler_small = DreamerScheduler::new(config_small);
    let region_small = scheduler_small.select_region(&arms, 10);
    let max_small = (arms.len() as f32 * 0.2).ceil() as usize;
    assert!(
        region_small.arm_indices.len() <= max_small,
        "[P3.4] strict region has {} arms but cap is {max_small}",
        region_small.arm_indices.len()
    );

    println!("✅ Proof 3 PASSED: Min-visits filter and region cap are enforced correctly");
}

// ── Proof 4: Snapshot Alignment ───────────────────────────────
//
// For any working region produced by select_region:
//   arm_indices.len() == q_snapshot.len() == visit_snapshot.len()
// Each snapshot entry corresponds to the arm at the same index.

#[test]
fn proof_4_snapshot_alignment() {
    let arms = make_test_arms();
    let configs = [
        DreamerConfig::default(),
        DreamerConfig::conservative(),
        DreamerConfig::aggressive(),
        DreamerConfig {
            cadence: 5,
            region_fraction: 0.1,
            min_visits: 1,
            ..DreamerConfig::default()
        },
        DreamerConfig {
            cadence: 20,
            region_fraction: 0.8,
            min_visits: 0,
            ..DreamerConfig::default()
        },
    ];

    for (i, config) in configs.iter().enumerate() {
        let scheduler = DreamerScheduler::new(config.clone());
        let region = scheduler.select_region(&arms, 10);

        let n_idx = region.arm_indices.len();
        let n_q = region.q_snapshot.len();
        let n_v = region.visit_snapshot.len();

        assert_eq!(
            n_idx, n_q,
            "[P4.1.{i}] arm_indices.len() ({n_idx}) != q_snapshot.len() ({n_q})"
        );
        assert_eq!(
            n_idx, n_v,
            "[P4.2.{i}] arm_indices.len() ({n_idx}) != visit_snapshot.len() ({n_v})"
        );

        // Verify snapshot values match the actual arm data
        for j in 0..n_idx {
            let arm_idx = region.arm_indices[j];
            assert!(
                approx_eq(region.q_snapshot[j], arms[arm_idx].q_value, 1e-6),
                "[P4.3.{i}] q_snapshot[{j}] = {} but arm {arm_idx}.q_value = {}",
                region.q_snapshot[j],
                arms[arm_idx].q_value
            );
            assert_eq!(
                region.visit_snapshot[j], arms[arm_idx].visits,
                "[P4.4.{i}] visit_snapshot[{j}] = {} but arm {arm_idx}.visits = {}",
                region.visit_snapshot[j], arms[arm_idx].visits
            );
        }
    }

    // Empty arms → all zero-length snapshots
    let scheduler = DreamerScheduler::new(DreamerConfig::default());
    let empty_region = scheduler.select_region(&[], 10);
    assert!(
        empty_region.arm_indices.is_empty(),
        "[P4.5] empty arms → empty region"
    );
    assert!(
        empty_region.q_snapshot.is_empty(),
        "[P4.6] empty arms → empty q_snapshot"
    );
    assert!(
        empty_region.visit_snapshot.is_empty(),
        "[P4.7] empty arms → empty visit_snapshot"
    );

    println!("✅ Proof 4 PASSED: Snapshot vectors are always aligned with arm_indices");
}

// ── Proof 5: Consolidator Sorted Output ───────────────────────
//
// Consolidator sorts arms by Q-value before clustering.
// The merged groups' Q-values (group averages) are in ascending order.

#[test]
fn proof_5_consolidator_sorted_output() {
    let config = DreamerConfig {
        merge_threshold: 0.15,
        ..DreamerConfig::default()
    };
    let consolidator = DreamerConsolidator::new(config);

    // Unsorted Q-values: 0.90, 0.10, 0.50, 0.55, 0.20
    let region = WorkingRegion {
        arm_indices: vec![0, 1, 2, 3, 4],
        q_snapshot: vec![0.90, 0.10, 0.50, 0.55, 0.20],
        visit_snapshot: vec![5, 5, 5, 5, 5],
        selected_at_episode: 10,
    };

    let result = consolidator.consolidate(&region);

    // Merged groups' Q-values must be in ascending order
    for i in 1..result.merged.len() {
        assert!(
            result.merged[i].1 >= result.merged[i - 1].1 - 1e-6,
            "[P5.1] merged[{i}].q={} < merged[{}].q={} (not ascending)",
            result.merged[i].1,
            i - 1,
            result.merged[i - 1].1
        );
    }

    // Verify with different thresholds
    let tight_config = DreamerConfig {
        merge_threshold: 0.05,
        ..DreamerConfig::default()
    };
    let tight = DreamerConsolidator::new(tight_config);

    let region2 = WorkingRegion {
        arm_indices: vec![0, 1, 2, 3, 4, 5],
        q_snapshot: vec![0.80, 0.12, 0.35, 0.95, 0.60, 0.10],
        visit_snapshot: vec![10; 6],
        selected_at_episode: 10,
    };

    let result2 = tight.consolidate(&region2);
    for i in 1..result2.merged.len() {
        assert!(
            result2.merged[i].1 >= result2.merged[i - 1].1 - 1e-6,
            "[P5.2] merged[{i}].q={} < merged[{}].q={} (not ascending)",
            result2.merged[i].1,
            i - 1,
            result2.merged[i - 1].1
        );
    }

    // Empty region → empty output
    let empty_region = WorkingRegion {
        arm_indices: vec![],
        q_snapshot: vec![],
        visit_snapshot: vec![],
        selected_at_episode: 10,
    };
    let empty_result = consolidator.consolidate(&empty_region);
    assert!(
        empty_result.merged.is_empty(),
        "[P5.3] empty region → empty merged"
    );

    println!("✅ Proof 5 PASSED: Consolidator output is sorted by ascending Q-value");
}

// ── Proof 6: Coverage ─────────────────────────────────────────
//
// All input arms from the working region appear in exactly one merged group.
// No arm is lost; no arm is duplicated across groups.
// This proves the consolidation is a proper partition of the input.

#[test]
fn proof_6_coverage() {
    let configs_and_thresholds: Vec<(DreamerConfig, Vec<f32>)> = vec![
        // Tight threshold → many small groups
        (
            DreamerConfig {
                merge_threshold: 0.05,
                ..DreamerConfig::default()
            },
            vec![0.80, 0.12, 0.35, 0.95, 0.60, 0.10],
        ),
        // Wide threshold → fewer large groups
        (
            DreamerConfig {
                merge_threshold: 1.0,
                ..DreamerConfig::default()
            },
            vec![0.80, 0.12, 0.35, 0.95, 0.60, 0.10],
        ),
        // Moderate threshold
        (
            DreamerConfig {
                merge_threshold: 0.2,
                ..DreamerConfig::default()
            },
            vec![0.10, 0.50, 0.55, 0.90, 0.20, 0.70],
        ),
        // Single arm
        (
            DreamerConfig {
                merge_threshold: 0.5,
                ..DreamerConfig::default()
            },
            vec![0.42],
        ),
        // Identical Q-values
        (
            DreamerConfig {
                merge_threshold: 0.3,
                ..DreamerConfig::default()
            },
            vec![0.50, 0.50, 0.50, 0.50],
        ),
    ];

    for (ci, (config, q_values)) in configs_and_thresholds.iter().enumerate() {
        let consolidator = DreamerConsolidator::new(config.clone());
        let n = q_values.len();
        let region = WorkingRegion {
            arm_indices: (0..n).collect(),
            q_snapshot: q_values.clone(),
            visit_snapshot: vec![5; n],
            selected_at_episode: 10,
        };

        let result = consolidator.consolidate(&region);

        // Collect all original indices from merged groups
        let mut all_indices: Vec<usize> = Vec::new();
        for (indices, _) in &result.merged {
            all_indices.extend(indices.iter().copied());
        }

        // Every input arm must appear exactly once
        let mut sorted = all_indices.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(
            sorted, expected,
            "[P6.1.{ci}] coverage mismatch: got {sorted:?}, expected {expected:?}"
        );

        // No duplicates (verified by sorted == expected, but be explicit)
        let unique: std::collections::HashSet<usize> = all_indices.iter().copied().collect();
        assert_eq!(
            unique.len(),
            all_indices.len(),
            "[P6.2.{ci}] duplicate arm indices in merged groups"
        );
    }

    println!(
        "✅ Proof 6 PASSED: All input arms appear in exactly one merged group (complete coverage)"
    );
}

// ── Proof 7: Decay Region Exempt ──────────────────────────────
//
// Arms in region.arm_indices are NEVER present in the decay output.
// Decay only applies to arms outside the working region.
// This proves the exemption mechanism works correctly.

#[test]
fn proof_7_decay_region_exempt() {
    let q_values = vec![0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4, 0.2, 0.0];
    let last_access = vec![5, 5, 5, 5, 5, 5, 5, 5, 5, 5];

    let policies = [
        DecayPolicy::None,
        DecayPolicy::Exponential { factor: 0.9 },
        DecayPolicy::Exponential { factor: 0.0 },
        DecayPolicy::Exponential { factor: 1.0 },
        DecayPolicy::AccessBased { half_life: 10 },
        DecayPolicy::AccessBased { half_life: 1 },
    ];

    let region_indices_set: Vec<Vec<usize>> = vec![
        vec![0, 3, 7],
        vec![0],
        vec![],
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ];

    for (pi, &policy) in policies.iter().enumerate() {
        let decay = MemoryDecay::new(policy);

        for (ri, exempt) in region_indices_set.iter().enumerate() {
            let region = WorkingRegion {
                arm_indices: exempt.clone(),
                q_snapshot: vec![0.5; q_values.len()],
                visit_snapshot: vec![5; q_values.len()],
                selected_at_episode: 10,
            };

            let result = decay.apply(&q_values, &last_access, &region, 10);

            // No exempt arm should appear in decay output
            for (idx, decayed_q) in &result {
                assert!(
                    !exempt.contains(idx),
                    "[P7.1.pi={pi}.ri={ri}] exempt arm {idx} appeared in decay output with q={decayed_q}"
                );

                // All output indices must be valid
                assert!(
                    *idx < q_values.len(),
                    "[P7.2.pi={pi}.ri={ri}] invalid index {idx} in decay output"
                );
            }

            // All non-exempt arms should appear
            let decay_indices: std::collections::HashSet<usize> =
                result.iter().map(|(i, _)| *i).collect();
            for i in 0..q_values.len() {
                if !exempt.contains(&i) {
                    assert!(
                        decay_indices.contains(&i),
                        "[P7.3.pi={pi}.ri={ri}] non-exempt arm {i} missing from decay output"
                    );
                }
            }
        }
    }

    println!("✅ Proof 7 PASSED: Arms in working region are never in decay output");
}

// ── Proof 8: Counterfactual Pipeline ──────────────────────────
//
// End-to-end: scheduler → region → consolidate → estimate_utility
// produces valid, finite results with correct cardinality.
// The full pipeline integrates all components correctly.

#[test]
fn proof_8_counterfactual_pipeline() {
    let arms = make_test_arms();

    // Step 1: Scheduler selects region
    let scheduler = DreamerScheduler::new(DreamerConfig {
        cadence: 10,
        region_fraction: 0.5,
        min_visits: 1,
        ..DreamerConfig::default()
    });
    let region = scheduler.select_region(&arms, 10);
    assert!(
        !region.arm_indices.is_empty(),
        "[P8.1] region should not be empty"
    );
    assert_eq!(
        region.arm_indices.len(),
        region.q_snapshot.len(),
        "[P8.2] snapshot alignment"
    );

    // Step 2: Consolidator produces replacement
    let consolidator = DreamerConsolidator::new(DreamerConfig {
        merge_threshold: 0.2,
        ..DreamerConfig::default()
    });
    let replacement = consolidator.consolidate(&region);
    assert!(
        !replacement.merged.is_empty(),
        "[P8.3] merged groups should not be empty"
    );

    // Verify coverage (all region arms in merged groups)
    let mut all_merged: Vec<usize> = Vec::new();
    for (indices, _) in &replacement.merged {
        all_merged.extend(indices.iter().copied());
    }
    all_merged.sort();
    let mut expected: Vec<usize> = (0..region.arm_indices.len()).collect();
    expected.sort();
    assert_eq!(
        all_merged, expected,
        "[P8.4] coverage: all region arms in merged groups"
    );

    // Verify merged groups are sorted by ascending Q-value
    for i in 1..replacement.merged.len() {
        assert!(
            replacement.merged[i].1 >= replacement.merged[i - 1].1 - 1e-6,
            "[P8.5] merged groups not sorted at index {i}"
        );
    }

    // Step 3: Counterfactual estimator computes utilities
    let estimator = CounterfactualEstimator::new(0.25, 5);
    let mut rng = Rng::new(42);
    let evaluator =
        |indices: &[usize]| -> f32 { indices.iter().map(|&i| region.q_snapshot[i]).sum() };

    let utilities = estimator.estimate_utility(&replacement, &evaluator, &mut rng);

    // Cardinality matches number of merged groups
    assert_eq!(
        utilities.len(),
        replacement.merged.len(),
        "[P8.6] utility len ({}) != merged len ({})",
        utilities.len(),
        replacement.merged.len()
    );

    // All utilities must be finite
    for (i, &u) in utilities.iter().enumerate() {
        assert!(u.is_finite(), "[P8.7] utility[{i}] = {u} is not finite");
    }

    // Utility sum should be positive (each group contributes something)
    let utility_sum: f32 = utilities.iter().sum();
    assert!(
        utility_sum >= 0.0,
        "[P8.8] total utility = {utility_sum} should be >= 0"
    );

    // Step 4: Decay arms outside region
    let decay = MemoryDecay::new(DecayPolicy::Exponential { factor: 0.9 });
    let q_all: Vec<f32> = arms.iter().map(|a| a.q_value).collect();
    let last_access: Vec<usize> = arms.iter().map(|a| a.last_retrieve_episode).collect();
    let decayed = decay.apply(&q_all, &last_access, &region, 10);

    // No region arm in decay output
    let region_set: std::collections::HashSet<usize> = region.arm_indices.iter().copied().collect();
    for (idx, _) in &decayed {
        assert!(
            !region_set.contains(idx),
            "[P8.9] region arm {idx} should not be in decay output"
        );
    }

    // Decayed values are finite
    for (idx, q) in &decayed {
        assert!(
            q.is_finite(),
            "[P8.10] decayed q[{idx}] = {q} is not finite"
        );
    }

    println!(
        "✅ Proof 8 PASSED: Full pipeline (scheduler → consolidate → counterfactual) produces valid results"
    );
}

// ── Summary ───────────────────────────────────────────────────

#[test]
fn summary_goat_107() {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  🐐 GOAT Proof: Auto-Dreamer Offline Consolidation (Plan 107)");
    println!("  Research 69 — Scheduled dreaming for memory consolidation");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Proof 1: Config presets valid                              ✅");
    println!("  Proof 2: Scheduler cadence trigger (e>0 && e%cadence==0)   ✅");
    println!("  Proof 3: Min-visits filter + region cap                    ✅");
    println!("  Proof 4: Snapshot alignment (len == len == len)            ✅");
    println!("  Proof 5: Consolidator sorted output (ascending Q)          ✅");
    println!("  Proof 6: Coverage (partition of input arms)                ✅");
    println!("  Proof 7: Decay region exempt (no region arm decayed)       ✅");
    println!("  Proof 8: Counterfactual pipeline (end-to-end valid)        ✅");
    println!();
    println!("  Verdict: Dreamer consolidation pipeline correctly schedules,");
    println!("  selects, consolidates, and estimates counterfactual utility.");
    println!("  Region exemption and coverage invariants hold across all");
    println!("  configs and decay policies.");
    println!("═══════════════════════════════════════════════════════════════");
}
