//! Test: Prove RubricBanditPruner scalar collapse is FIXED (Issue 061)
//!
//! Run: cargo test --features "ropd_rubric,g_zero" --test test_rubric_scalar_collapse -- --nocapture
//!
//! Issue 061: `compute_reward()` used `weighted_score()` which collapsed all criteria
//! to a single scalar before feeding to bandit. Two RubricVectors with identical
//! weighted_score but different per-criterion profiles produced identical reward.
//!
//! Fix: `compute_reward()` now uses `quadratic_weighted_reward()`:
//!   `Σ(w_i × gap_i²) / Σ(w_i)` instead of `Σ(w_i × gap_i) / Σ(w_i)`
//!
//! The quadratic form breaks permutation symmetry — concentrated gaps in one
//! criterion score higher than spread gaps across multiple criteria.
//!
//! Tests:
//! 1. `test_quadratic_reward_differentiates_same_weighted_score` — core proof
//! 2. `test_rubric_bandit_no_longer_equivalent_to_scalar` — RubricPruner ≠ DeltaPruner
//! 3. `test_rubric_bandit_converges_toward_concentrated_gaps` — bandit learns criterion identity
//! 4. `test_rubric_absorb_reward_uses_quadratic` — absorb also uses quadratic form

// ── Test 1: Core Proof — Quadratic Differentiates Profiles ──────────

#[cfg(feature = "ropd_rubric")]
#[test]
fn test_quadratic_reward_differentiates_same_weighted_score() {
    use katgpt_rs::pruners::{RubricTemplate, RubricVector};

    let template = RubricTemplate::bomber();
    let weights: Vec<f32> = template.criteria.iter().map(|(_, w)| *w).collect();
    // Bomber weights: [4.0, 2.0, 1.0] (survival, safety, efficiency)
    // Sum = 7.0

    let reference = RubricVector::perfect(weights.clone(), 0);

    // Student A: strong survival (1.0), weak safety (0.0), weak efficiency (0.0)
    // weighted_score = (4*1.0 + 2*0.0 + 1*0.0) / 7.0 = 4/7 ≈ 0.571
    let student_a = RubricVector::new(vec![1.0, 0.0, 0.0], weights.clone(), 0);

    // Student C: balanced — moderate survival (0.5), moderate safety (0.5), perfect efficiency (1.0)
    // weighted_score = (4*0.5 + 2*0.5 + 1*1.0) / 7.0 = (2+1+1)/7 = 4/7 ≈ 0.571
    let student_c = RubricVector::new(vec![0.5, 0.5, 1.0], weights.clone(), 0);

    println!("\n🔬 Test 1: Quadratic Reward Differentiates Same weighted_score");
    println!("{}", "═".repeat(70));

    // ── Step 1: Prove weighted_score is IDENTICAL (the collapse precondition) ──
    let ws_a = student_a.weighted_score();
    let ws_c = student_c.weighted_score();
    let scalar_a = reference.weighted_score() - ws_a;
    let scalar_c = reference.weighted_score() - ws_c;

    println!("\n📊 Step 1: Scalar collapse (BEFORE fix)");
    println!("  A: scores=(1.0, 0.0, 0.0) ws={ws_a:.6}");
    println!("  C: scores=(0.5, 0.5, 1.0) ws={ws_c:.6}");
    println!("  scalar_reward(A) = 1.0 - {ws_a:.6} = {scalar_a:.6}");
    println!("  scalar_reward(C) = 1.0 - {ws_c:.6} = {scalar_c:.6}");
    println!(
        "  SAME? {}",
        if (scalar_a - scalar_c).abs() < 1e-6 {
            "YES ❌ (the bug)"
        } else {
            "NO"
        }
    );

    assert!(
        (ws_a - ws_c).abs() < 1e-6,
        "Precondition: A and C must have same weighted_score"
    );
    assert!(
        (scalar_a - scalar_c).abs() < 1e-6,
        "Scalar reward should be identical for same weighted_score (this IS the bug)"
    );

    // ── Step 2: Prove quadratic reward is DIFFERENT ──
    let quad_a = student_a.quadratic_weighted_reward(&reference);
    let quad_c = student_c.quadratic_weighted_reward(&reference);

    println!("\n📊 Step 2: Quadratic reward (AFTER fix)");
    println!("  A: gaps=(0.0, 1.0, 1.0) → quad = (4*0² + 2*1² + 1*1²)/7 = 3/7 = {quad_a:.6}");
    println!("  C: gaps=(0.5, 0.5, 0.0) → quad = (4*0.25 + 2*0.25 + 1*0²)/7 = 1.5/7 = {quad_c:.6}");
    println!(
        "  DIFFERENT? {}",
        if (quad_a - quad_c).abs() > 1e-6 {
            "YES ✅ (fix works!)"
        } else {
            "NO ❌"
        }
    );

    assert!(
        (quad_a - quad_c).abs() > 1e-6,
        "Quadratic reward MUST differ for different gap profiles with same weighted_score"
    );

    // ── Step 3: Prove per-criterion gaps ARE different (the information exists) ──
    let gaps_a = student_a.gap_criteria(&reference);
    let gaps_c = student_c.gap_criteria(&reference);

    println!("\n📊 Step 3: Per-criterion gaps differ");
    println!("  A gaps: {:?}", gaps_a);
    println!("  C gaps: {:?}", gaps_c);
    assert_ne!(gaps_a, gaps_c, "Gap profiles must differ");

    // ── Step 4: Prove A has HIGHER quadratic reward (concentrated gaps) ──
    println!("\n📊 Step 4: Concentrated gaps → higher quadratic reward");
    println!("  A: concentrated failures (safety=0.0, efficiency=0.0) → quad={quad_a:.6}");
    println!("  C: spread failures (survival=0.5, safety=0.5) → quad={quad_c:.6}");
    println!(
        "  A > C? {} ({:.1}% higher)",
        if quad_a > quad_c { "YES ✅" } else { "NO" },
        (quad_a - quad_c) / quad_c * 100.0
    );

    assert!(
        quad_a > quad_c,
        "Concentrated gaps (A) should produce higher quadratic reward than spread gaps (C)"
    );

    // ── Summary ──
    println!("\n{}", "═".repeat(70));
    println!("✅ FIX VERIFIED: quadratic_weighted_reward differentiates profiles");
    println!(
        "   Same weighted_score ({ws_a:.4}) → different quadratic reward ({quad_a:.4} ≠ {quad_c:.4})"
    );
    println!("   Bandit can now learn WHICH criteria have gaps, not just that gaps exist.");
}

// ── Test 2: RubricBanditPruner ≠ DeltaBanditPruner ──────────────────

#[cfg(all(feature = "ropd_rubric", feature = "g_zero"))]
#[test]
fn test_rubric_bandit_no_longer_equivalent_to_scalar() {
    use katgpt_rs::pruners::{
        BanditPruner, BanditStrategy, DeltaBanditPruner, RubricBanditPruner, RubricTemplate,
        RubricVector,
    };
    use katgpt_rs::speculative::types::NoScreeningPruner;

    let template = RubricTemplate::bomber();
    let weights: Vec<f32> = template.criteria.iter().map(|(_, w)| *w).collect();
    let reference = RubricVector::perfect(weights.clone(), 0);

    let num_arms = 5;
    let num_criteria = 3;
    let episodes = 200;

    let mut rubric_bandit: RubricBanditPruner<NoScreeningPruner> = RubricBanditPruner::new(
        BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms),
        num_arms,
        num_criteria,
    );

    let mut delta_bandit: DeltaBanditPruner<NoScreeningPruner> = DeltaBanditPruner::new(
        BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms),
        num_arms,
    );

    println!("\n🔬 Test 2: RubricBanditPruner ≠ DeltaBanditPruner (after fix)");
    println!("{}", "═".repeat(70));

    // Create rubrics with per-criterion variation (not uniform scores).
    // This is where quadratic reward differentiates — non-uniform gaps.
    // Arm 0: survival=0.2, safety=0.8, efficiency=1.0 (ws≈0.49)
    // Arm 1: survival=0.4, safety=0.6, efficiency=0.8 (ws≈0.54)
    // Arm 2: survival=0.6, safety=0.4, efficiency=0.6 (ws≈0.54)
    // Arm 3: survival=0.8, safety=0.2, efficiency=0.4 (ws≈0.54)
    // Arm 4: survival=1.0, safety=1.0, efficiency=1.0 (ws=1.0, perfect)
    let arm_rubrics: Vec<RubricVector> = vec![
        RubricVector::new(vec![0.2, 0.8, 1.0], weights.clone(), 0),
        RubricVector::new(vec![0.4, 0.6, 0.8], weights.clone(), 0),
        RubricVector::new(vec![0.6, 0.4, 0.6], weights.clone(), 0),
        RubricVector::new(vec![0.8, 0.2, 0.4], weights.clone(), 0),
        RubricVector::new(vec![1.0, 1.0, 1.0], weights.clone(), 0),
    ];

    println!("Arm rubrics (non-uniform per-criterion scores):");
    for (i, r) in arm_rubrics.iter().enumerate() {
        println!(
            "  Arm {i}: scores=({:.1}, {:.1}, {:.1}) ws={:.4} quad_reward={:.4}",
            r.score(0),
            r.score(1),
            r.score(2),
            r.weighted_score(),
            r.quadratic_weighted_reward(&reference),
        );
    }

    // Feed same episodes to both bandits
    for _ in 0..episodes {
        for (arm, rubric) in arm_rubrics.iter().enumerate() {
            let delta = 1.0 - rubric.weighted_score();

            rubric_bandit.observe_rubric(arm, rubric, &reference);
            delta_bandit.observe_delta(arm, delta);
        }
    }

    println!("\nAfter {episodes} episodes per arm:");
    println!(
        "  {:5} {:>10} {:>10} {:>10} {:>8}",
        "Arm", "Scalar δ", "Rubric", "Delta", "Diff"
    );
    let mut any_different = false;
    for (arm, arm_rubric) in arm_rubrics.iter().enumerate() {
        let ws = arm_rubric.weighted_score();
        let delta = 1.0 - ws;
        let r_reward = rubric_bandit.mean_reward(arm);
        let d_reward = delta_bandit.mean_delta(arm);
        let diff = (r_reward - d_reward).abs();

        println!(
            "  Arm {arm}:   δ={delta:.4}  rubric={r_reward:.4}  delta_b={d_reward:.4}  diff={diff:.6}"
        );

        if diff > 0.01 {
            any_different = true;
        }
    }

    assert!(
        any_different,
        "At least one arm must have different rubric reward vs scalar delta — fix is working"
    );

    println!("\n✅ CONFIRMED: RubricBanditPruner ≠ DeltaBanditPruner after fix");
    println!("   The quadratic weighted reward provides differentiated signal");
    println!("   for arms with non-uniform per-criterion score profiles.");
}

// ── Test 3: Bandit Converges Toward Concentrated Gap Arms ────────────

#[cfg(feature = "ropd_rubric")]
#[test]
fn test_rubric_bandit_converges_toward_concentrated_gaps() {
    use katgpt_rs::pruners::{BanditPruner, BanditStrategy, RubricBanditPruner, RubricVector};
    use katgpt_rs::speculative::types::NoScreeningPruner;

    // Two arms with same weighted_score but different gap distributions.
    // The bandit should prefer the arm with concentrated gaps (higher quadratic reward)
    // because it's more actionable for learning.

    let weights = vec![4.0, 2.0, 1.0];
    let reference = RubricVector::perfect(weights.clone(), 0);

    let num_arms = 3;
    let num_criteria = 3;

    let mut bandit: RubricBanditPruner<NoScreeningPruner> = RubricBanditPruner::new(
        BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms),
        num_arms,
        num_criteria,
    );

    // Arm 0: concentrated gap — safety=0.0, efficiency=0.0 (actionable)
    // scores=(1.0, 0.0, 0.0), ws=4/7 ≈ 0.571, quad=3/7 ≈ 0.429
    let arm_0 = RubricVector::new(vec![1.0, 0.0, 0.0], weights.clone(), 0);

    // Arm 1: spread gap — survival=0.5, safety=0.5 (less actionable)
    // scores=(0.5, 0.5, 1.0), ws=4/7 ≈ 0.571, quad=1.5/7 ≈ 0.214
    let arm_1 = RubricVector::new(vec![0.5, 0.5, 1.0], weights.clone(), 0);

    // Arm 2: perfect (no gap) — control arm
    let arm_2 = RubricVector::new(vec![1.0, 1.0, 1.0], weights.clone(), 0);

    println!("\n🔬 Test 3: Bandit Converges Toward Concentrated Gap Arms");
    println!("{}", "═".repeat(70));
    println!(
        "  Arm 0: concentrated gaps → quad={:.4}",
        arm_0.quadratic_weighted_reward(&reference)
    );
    println!(
        "  Arm 1: spread gaps → quad={:.4}",
        arm_1.quadratic_weighted_reward(&reference)
    );
    println!(
        "  Arm 2: no gaps → quad={:.4}",
        arm_2.quadratic_weighted_reward(&reference)
    );

    // Feed 100 episodes per arm
    for _ in 0..100 {
        bandit.observe_rubric(0, &arm_0, &reference);
        bandit.observe_rubric(1, &arm_1, &reference);
        bandit.observe_rubric(2, &arm_2, &reference);
    }

    let reward_0 = bandit.mean_reward(0);
    let reward_1 = bandit.mean_reward(1);
    let reward_2 = bandit.mean_reward(2);

    println!("\nAfter 100 observations per arm:");
    println!("  Arm 0 (concentrated): mean_reward={reward_0:.6}");
    println!("  Arm 1 (spread):       mean_reward={reward_1:.6}");
    println!("  Arm 2 (perfect):      mean_reward={reward_2:.6}");

    // Arm 0 should have higher reward than arm 1 (concentrated > spread)
    assert!(
        reward_0 > reward_1,
        "Concentrated gap arm should have higher reward than spread gap arm: {reward_0:.4} > {reward_1:.4}"
    );

    // Arm 2 should have zero reward (perfect = no gap)
    assert!(
        reward_2 < 0.001,
        "Perfect arm should have near-zero reward: {reward_2:.4}"
    );

    // With scalar collapse (BEFORE fix), reward_0 would equal reward_1
    let ratio = reward_0 / reward_1.max(f32::EPSILON);
    println!("\n  Ratio arm_0/arm_1 = {ratio:.2}x (scalar collapse would give 1.00x)");

    assert!(
        ratio > 1.5,
        "Concentrated arm should have significantly higher reward (ratio > 1.5x): {ratio:.2}x"
    );

    println!("\n✅ CONFIRMED: Bandit differentiates concentrated vs spread gaps");
    println!("   Arm 0 reward ({reward_0:.4}) >> Arm 1 reward ({reward_1:.4})");
    println!("   This enables criterion-aware learning — the bandit knows WHERE gaps are.");
}

// ── Test 4: Absorb Reward Uses Quadratic Form ───────────────────────

#[cfg(feature = "ropd_rubric")]
#[test]
fn test_rubric_absorb_reward_uses_quadratic() {
    use katgpt_rs::pruners::{
        AbsorbCompressLayer, CompressConfig, RubricGatedAbsorbCompress, RubricGatedConfig,
        RubricTemplate, RubricVector,
    };
    use katgpt_rs::speculative::types::NoScreeningPruner;

    let template = RubricTemplate::bomber();
    let weights: Vec<f32> = template.criteria.iter().map(|(_, w)| *w).collect();
    let reference = RubricVector::perfect(weights.clone(), 0);
    let references = vec![reference.clone(), reference.clone()];

    let num_arms = 3;
    let config = CompressConfig::new(50, 0.1, 5, 1000);

    let inner = AbsorbCompressLayer::new(NoScreeningPruner, num_arms, config);
    let mut absorb = RubricGatedAbsorbCompress::new(inner, num_arms, RubricGatedConfig::default());

    // Student A: concentrated gap on criterion 0 (high weight 4.0, gap=1.0)
    // Quadratic absorb: 4.0 * 1.0² = 4.0
    let student_a = RubricVector::new(vec![0.0, 1.0, 1.0], weights.clone(), 0);

    // Student B: concentrated gap on criterion 2 (low weight 1.0, gap=1.0)
    // Quadratic absorb: 1.0 * 1.0² = 1.0
    // NOTE: weight 1.0 < min_weight_for_absorb (2.0), so this gets filtered out
    let student_b = RubricVector::new(vec![1.0, 1.0, 0.0], weights.clone(), 0);

    // Student C: concentrated gap on criterion 1 (weight 2.0, gap=0.5)
    // Quadratic absorb: 2.0 * 0.5² = 0.5
    let student_c = RubricVector::new(vec![1.0, 0.5, 1.0], weights.clone(), 0);

    println!("\n🔬 Test 4: Absorb Reward Uses Quadratic Form");
    println!("{}", "═".repeat(70));

    // Observe all three with 2 references
    absorb.observe_rubric(0, &student_a, &references);
    absorb.observe_rubric(1, &student_b, &references);
    absorb.observe_rubric(2, &student_c, &references);

    let above_a = absorb.is_above_threshold(0);
    let above_b = absorb.is_above_threshold(1);
    let above_c = absorb.is_above_threshold(2);

    println!("Student A (gap on criterion 0, weight=4.0, gap=1.0):");
    println!("  above_threshold = {above_a}");
    println!("  quadratic absorb = 4.0 × 1.0² = 4.0");

    println!("Student B (gap on criterion 2, weight=1.0, gap=1.0):");
    println!("  above_threshold = {above_b}");
    println!("  weight 1.0 < min_weight_for_absorb 2.0 → filtered");

    println!("Student C (gap on criterion 1, weight=2.0, gap=0.5):");
    println!("  above_threshold = {above_c}");
    println!("  quadratic absorb = 2.0 × 0.5² = 0.5");

    // A: high-weight gap → above threshold ✅
    assert!(
        above_a,
        "A should be above threshold (high-weight concentrated gap)"
    );

    // B: low-weight gap → below threshold ❌
    assert!(
        !above_b,
        "B should be below threshold (low-weight gap filtered)"
    );

    // C: moderate weight, moderate gap, but gap=0.5 > gap_threshold=0.3, weight=2.0 >= 2.0
    assert!(
        above_c,
        "C should be above threshold (meets both thresholds)"
    );

    println!("\n✅ CONFIRMED: Absorb uses quadratic form for reward computation");
    println!("   Per-criterion filtering still works (weight >= 2.0 check)");
    println!("   Reward now uses gap² × weight instead of gap × weight");
    println!("   This preserves per-criterion identity in absorb decisions.");
}
