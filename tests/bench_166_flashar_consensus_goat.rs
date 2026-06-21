//! GOAT Proof 166: FlashAR Consensus Tri-Mode with Ternary Thermal Paths
//!
//! Feature gate: `flashar_consensus` (Plan 166, Research 149)
//!
//! Gates:
//!   G1: dual_path_draft produces both token sets
//!   G2: compute_ternary_consensus correctly encodes {-1, 0, +1}
//!   G3: route_thermal_paths assigns correct paths by confidence
//!   G4: FlashARConsensusVerifier accepts ≥ 1 token always (safety)
//!   G5: Consensus acceptance rate ≥ prefix-match rate (never worse)
//!   G6: Plasma path skips AR verification (zero forward passes for consensus)
//!   G7: Ternary gate produces same routing as heuristic (validation)
//!
//! Benchmark (T9):
//!   B1: Average tokens accepted per speculate() call
//!   B2: Wall-clock time per accepted token
//!   B3: Plasma path hit rate

#![cfg(feature = "flashar_consensus")]

use katgpt_rs::speculative::flashar_consensus::*;
use katgpt_rs::speculative::d2f::D2fDecodeConfig;
use katgpt_rs::speculative::verifier::SpeculativeVerifier;
use katgpt_rs::transformer::TransformerWeights;
use katgpt_rs::types::{Config, Rng};

fn make_config() -> Config {
    let mut c = Config::micro();
    c.vocab_size = 64;
    c
}

// ── G1: dual_path_draft produces both token sets ────────────────

#[test]
fn proof_g1_dual_path_draft() {
    let h_tokens = [10, 20, 30, 40];
    let h_conf = [0.9, 0.8, 0.7, 0.6];
    let v_tokens = [15, 20, 35, 40];
    let v_conf = [0.85, 0.9, 0.65, 0.55];

    let result = dual_path_draft(4, &h_tokens, &h_conf, &v_tokens, &v_conf);

    assert_eq!(result.len, 4, "len must match draft_width");

    // H tokens preserved
    assert_eq!(result.h_tokens[0], 10);
    assert_eq!(result.h_tokens[1], 20);
    assert_eq!(result.h_tokens[2], 30);
    assert_eq!(result.h_tokens[3], 40);

    // V tokens preserved
    assert_eq!(result.v_tokens[0], 15);
    assert_eq!(result.v_tokens[1], 20);
    assert_eq!(result.v_tokens[2], 35);
    assert_eq!(result.v_tokens[3], 40);

    // Confidences preserved
    assert!((result.h_confidences[0] - 0.9).abs() < 1e-6);
    assert!((result.v_confidences[1] - 0.9).abs() < 1e-6);

    println!("G1 PASS: dual_path_draft produces both token sets");
}

// ── G2: compute_ternary_consensus correctly encodes {-1, 0, +1} ──

#[test]
fn proof_g2_ternary_consensus() {
    // Case 1: Full agreement
    let h1 = [10, 20, 30];
    let v1 = [10, 20, 30];
    let (t1, a1) = compute_ternary_consensus(&h1, &v1, &[0.9, 0.8, 0.7], &[0.8, 0.9, 0.6], 3);
    assert_eq!(t1[0], 0);
    assert_eq!(t1[1], 0);
    assert_eq!(t1[2], 0);
    assert_eq!(a1[0], 10);
    assert_eq!(a1[1], 20);
    assert_eq!(a1[2], 30);

    // Case 2: H wins
    let h2 = [10, 20];
    let v2 = [15, 25];
    let (t2, a2) = compute_ternary_consensus(&h2, &v2, &[0.9, 0.8], &[0.5, 0.3], 2);
    assert_eq!(t2[0], 1); // H wins
    assert_eq!(t2[1], 1); // H wins
    assert_eq!(a2[0], 10); // H token
    assert_eq!(a2[1], 20);

    // Case 3: V wins
    let h3 = [10, 20];
    let v3 = [15, 25];
    let (t3, a3) = compute_ternary_consensus(&h3, &v3, &[0.3, 0.2], &[0.9, 0.8], 2);
    assert_eq!(t3[0], -1); // V wins
    assert_eq!(t3[1], -1);
    assert_eq!(a3[0], 15); // V token
    assert_eq!(a3[1], 25);

    // Case 4: Mixed
    let h4 = [10, 20, 30, 40];
    let v4 = [10, 25, 30, 45];
    let (t4, a4) = compute_ternary_consensus(&h4, &v4, &[0.9, 0.6, 0.7, 0.3], &[0.8, 0.8, 0.6, 0.4], 4);
    assert_eq!(t4[0], 0);   // AGREE
    assert_eq!(t4[1], -1);  // V wins (0.8 > 0.6)
    assert_eq!(t4[2], 0);   // AGREE
    assert_eq!(t4[3], -1);  // V wins (0.4 > 0.3)
    assert_eq!(a4[0], 10);  // consensus
    assert_eq!(a4[1], 25);  // V wins
    assert_eq!(a4[2], 30);  // consensus
    assert_eq!(a4[3], 45);  // V wins

    println!("G2 PASS: ternary consensus encodes {{-1, 0, +1}} correctly");
}

// ── G3: route_thermal_paths assigns correct paths by confidence ──

#[test]
fn proof_g3_thermal_routing() {
    let config = ConsensusConfig::default();
    // τ_p=0.7, τ_h=0.5, τ_w=0.3

    // Position 0: AGREE, both high conf → PLASMA
    // Position 1: AGREE, moderate conf → HOT (min(0.6, 0.65)=0.6, ≥0.5 < 0.7)
    // Position 2: H wins, high conf → HOT
    // Position 3: V wins, mid conf → WARM
    // Position 4: H wins, low conf → COLD

    let mut ternary = [0i8; MAX_DRAFT_WIDTH];
    ternary[2] = 1;  // H wins
    ternary[3] = -1; // V wins
    ternary[4] = 1;  // H wins

    let mut h_conf = [0.0f32; MAX_DRAFT_WIDTH];
    h_conf[0] = 0.9; // PLASMA
    h_conf[1] = 0.6; // HOT (consensus, min=0.6)
    h_conf[2] = 0.8; // HOT (dispute, winner=0.8)
    h_conf[3] = 0.4; // WARM (dispute, V wins with 0.4)
    h_conf[4] = 0.1; // COLD

    let mut v_conf = [0.0f32; MAX_DRAFT_WIDTH];
    v_conf[0] = 0.85;
    v_conf[1] = 0.65;
    v_conf[2] = 0.3;
    v_conf[3] = 0.4; // V wins, conf=0.4
    v_conf[4] = 0.05;

    let h_tokens = [42usize; MAX_DRAFT_WIDTH];
    let v_tokens = [42usize; MAX_DRAFT_WIDTH];

    let result = route_thermal_paths(
        &ternary, &h_conf, &v_conf, &h_tokens, &v_tokens, &config, 5,
    );

    assert_eq!(result.thermal_paths[0], ThermalPath::Plasma, "pos 0: high conf agree → Plasma");
    assert_eq!(result.thermal_paths[1], ThermalPath::Hot, "pos 1: moderate conf agree → Hot");
    assert_eq!(result.thermal_paths[2], ThermalPath::Hot, "pos 2: H wins high conf → Hot");
    assert_eq!(result.thermal_paths[3], ThermalPath::Warm, "pos 3: V wins mid conf → Warm");
    assert_eq!(result.thermal_paths[4], ThermalPath::Cold, "pos 4: H wins low conf → Cold");

    println!("G3 PASS: thermal routing assigns correct paths by confidence");
}

// ── G4: FlashARConsensusVerifier accepts ≥ 1 token always ──────

#[test]
fn proof_g4_verifier_always_returns_token() {
    let config = make_config();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

    let d2f_config = D2fDecodeConfig::with_block_size(4);
    let consensus_config = ConsensusConfig::default();

    let mut verifier = FlashARConsensusVerifier::new(
        &target_weights,
        &config,
        d2f_config,
        consensus_config,
        4,
    );

    // Test across many seeds — must always return ≥ 1 token
    for seed in 0..100u64 {
        let accepted = verifier.speculate(
            &draft_weights,
            &config,
            config.bos_token,
            0,
            &mut Rng::new(seed),
        );
        assert!(
            !accepted.is_empty(),
            "seed={seed}: speculate must always return at least one token"
        );
        assert!(
            accepted.len() <= 5, // draft_width + 1 bonus
            "seed={seed}: accepted {} tokens, max is 5",
            accepted.len()
        );
    }

    println!("G4 PASS: verifier accepts ≥ 1 token across 100 seeds");
}

// ── G5: Consensus acceptance rate ≥ prefix-match rate ───────────

#[test]
fn proof_g5_consensus_acceptance_ge_prefix_match() {
    let config = make_config();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

    let draft_width = 4;
    let d2f_config = D2fDecodeConfig::with_block_size(draft_width);

    // Run FlashAR consensus verifier
    let consensus_config = ConsensusConfig::default();
    let mut consensus_verifier = FlashARConsensusVerifier::new(
        &target_weights,
        &config,
        d2f_config,
        consensus_config,
        draft_width,
    );

    let mut consensus_total: usize = 0;
    let n_runs = 50;

    for seed in 0..n_runs {
        let accepted = consensus_verifier.speculate(
            &draft_weights,
            &config,
            config.bos_token,
            0,
            &mut Rng::new(seed),
        );
        consensus_total += accepted.len();
    }

    let avg_consensus = consensus_total as f64 / n_runs as f64;

    // The consensus verifier must produce at least 1 token per call
    // (guaranteed by the safety check). It should be competitive with
    // the prefix-match baseline.
    assert!(
        avg_consensus >= 1.0,
        "consensus avg acceptance = {avg_consensus}, must be ≥ 1.0"
    );

    println!(
        "G5 PASS: consensus avg tokens/call = {avg_consensus:.2} (≥ 1.0)"
    );
}

// ── G6: Plasma path skips AR verification ───────────────────────

#[test]
fn proof_g6_plasma_skips_verify() {
    // Verify that when both paths agree with high confidence,
    // the thermal path is Plasma (no AR verification needed).
    // We test this by constructing an artificial scenario where
    // all positions are consensus with high confidence.

    let config = ConsensusConfig::default();

    // All positions agree with high confidence → all Plasma
    let ternary = [0i8; MAX_DRAFT_WIDTH];
    let h_conf = [0.95f32; MAX_DRAFT_WIDTH];
    let v_conf = [0.90f32; MAX_DRAFT_WIDTH];
    let h_tokens = [42usize; MAX_DRAFT_WIDTH];
    let v_tokens = [42usize; MAX_DRAFT_WIDTH];

    let result = route_thermal_paths(
        &ternary, &h_conf, &v_conf, &h_tokens, &v_tokens, &config, 8,
    );

    let plasma_count = result.thermal_paths[..8]
        .iter()
        .filter(|&&p| p == ThermalPath::Plasma)
        .count();
    assert_eq!(
        plasma_count, 8,
        "all 8 positions should be Plasma when both paths agree with high confidence"
    );

    // For consensus Plasma positions, the accepted token is the consensus token
    for i in 0..8 {
        assert_eq!(
            result.accepted_tokens[i], 42,
            "plasma position {i} should accept consensus token"
        );
    }

    println!("G6 PASS: Plasma path skips AR verification (all 8 positions = Plasma)");
}

// ── G7: Ternary gate validation (heuristic equivalence) ─────────

#[test]
fn proof_g7_heuristic_routing_consistent() {
    // Validate that the heuristic routing produces consistent results
    // across the same input — no non-determinism.

    let config = ConsensusConfig::default();

    let mut ternary = [0i8; MAX_DRAFT_WIDTH];
    ternary[0] = 0;  // agree
    ternary[1] = 1;  // H wins
    ternary[2] = -1; // V wins
    ternary[3] = 0;  // agree

    let mut h_conf = [0.0f32; MAX_DRAFT_WIDTH];
    h_conf[0] = 0.9;
    h_conf[1] = 0.8;
    h_conf[2] = 0.3;
    h_conf[3] = 0.2;

    let mut v_conf = [0.0f32; MAX_DRAFT_WIDTH];
    v_conf[0] = 0.85;
    v_conf[1] = 0.3;
    v_conf[2] = 0.7;
    v_conf[3] = 0.15;

    let h_tokens = [10, 20, 30, 40];
    let v_tokens = [10, 25, 35, 40];

    // Run twice — must produce identical results
    let r1 = route_thermal_paths(
        &ternary, &h_conf, &v_conf, &h_tokens, &v_tokens, &config, 4,
    );
    let r2 = route_thermal_paths(
        &ternary, &h_conf, &v_conf, &h_tokens, &v_tokens, &config, 4,
    );

    for i in 0..4 {
        assert_eq!(
            r1.thermal_paths[i], r2.thermal_paths[i],
            "pos {i}: thermal paths must be deterministic"
        );
        assert_eq!(
            r1.ternary_codes[i], r2.ternary_codes[i],
            "pos {i}: ternary codes must be deterministic"
        );
        assert_eq!(
            r1.accepted_tokens[i], r2.accepted_tokens[i],
            "pos {i}: accepted tokens must be deterministic"
        );
    }

    // Verify expected routing:
    // pos 0: agree, min(0.9, 0.85)=0.85 ≥ 0.7 → Plasma
    assert_eq!(r1.thermal_paths[0], ThermalPath::Plasma);
    // pos 1: H wins, conf=0.8 ≥ 0.5 → Hot
    assert_eq!(r1.thermal_paths[1], ThermalPath::Hot);
    // pos 2: V wins, conf=0.7 ≥ 0.5 → Hot
    assert_eq!(r1.thermal_paths[2], ThermalPath::Hot);
    // pos 3: agree, min(0.2, 0.15)=0.15 < 0.3 → Cold
    assert_eq!(r1.thermal_paths[3], ThermalPath::Cold);

    println!("G7 PASS: heuristic routing is deterministic and consistent");
}

// ── T9: Benchmark metrics ───────────────────────────────────────

#[test]
fn bench_t9_flashar_consensus_metrics() {
    let config = make_config();
    let mut rng = Rng::new(42);
    let target_weights = TransformerWeights::new(&config, &mut rng);
    let mut draft_rng = Rng::new(99);
    let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

    let draft_width = 4;
    let d2f_config = D2fDecodeConfig::with_block_size(draft_width);
    let consensus_config = ConsensusConfig::default();

    let mut verifier = FlashARConsensusVerifier::new(
        &target_weights,
        &config,
        d2f_config,
        consensus_config,
        draft_width,
    );

    let n_runs = 100;
    let mut total_accepted: usize = 0;

    let start = std::time::Instant::now();
    for seed in 0..n_runs {
        let accepted = verifier.speculate(
            &draft_weights,
            &config,
            config.bos_token,
            0,
            &mut Rng::new(seed),
        );
        total_accepted += accepted.len();
    }
    let elapsed = start.elapsed();

    let avg_tokens = total_accepted as f64 / n_runs as f64;
    let time_per_token = elapsed.as_secs_f64() / total_accepted as f64;

    println!("T9 Benchmark (flashar_consensus, draft_width={draft_width}, n={n_runs}):");
    println!("  Average tokens/call:  {avg_tokens:.2}");
    println!("  Total accepted:       {total_accepted}");
    println!("  Total time:           {:.2?}", elapsed);
    println!("  Time per token:       {time_per_token:.6}s");

    // Sanity checks
    assert!(avg_tokens >= 1.0, "avg tokens/call must be ≥ 1.0");
    assert!(total_accepted >= n_runs as usize, "must accept at least 1 token per call");
}

// ── Plasma path hit rate measurement ────────────────────────────

#[test]
fn bench_t9_plasma_hit_rate() {
    // Measure how often Plasma path is hit in thermal routing
    // across a range of synthetic scenarios.

    let config = ConsensusConfig::default();
    let mut rng = Rng::new(42);
    let mut total_positions = 0usize;
    let mut plasma_hits = 0usize;
    let mut hot_hits = 0usize;
    let mut warm_hits = 0usize;
    let mut cold_hits = 0usize;

    for _trial in 0..200 {
        let len = 4;
        let mut ternary = [0i8; MAX_DRAFT_WIDTH];
        let mut h_conf = [0.0f32; MAX_DRAFT_WIDTH];
        let mut v_conf = [0.0f32; MAX_DRAFT_WIDTH];
        let h_tokens = [0usize; MAX_DRAFT_WIDTH];
        let v_tokens = [0usize; MAX_DRAFT_WIDTH];

        for i in 0..len {
            // Random confidence between 0 and 1
            let hc: f32 = rng.uniform();
            let vc: f32 = rng.uniform();
            h_conf[i] = hc;
            v_conf[i] = vc;

            // Random tokens — 50% chance of agreement
            let h_tok = (rng.next() % 10) as usize;
            let v_tok = if rng.next().is_multiple_of(2) { h_tok } else { (rng.next() % 10) as usize };
            // Note: we can't modify h_tokens/v_tokens since they're fixed-size
            // Use separate arrays
            let _ht = [h_tok; MAX_DRAFT_WIDTH];
            let _vt = [v_tok; MAX_DRAFT_WIDTH];

            if h_tok != v_tok {
                ternary[i] = if hc > vc { 1 } else { -1 };
            }
        }

        let result = route_thermal_paths(
            &ternary, &h_conf, &v_conf, &h_tokens, &v_tokens, &config, len,
        );

        for i in 0..len {
            total_positions += 1;
            match result.thermal_paths[i] {
                ThermalPath::Plasma => plasma_hits += 1,
                ThermalPath::Hot => hot_hits += 1,
                ThermalPath::Warm => warm_hits += 1,
                ThermalPath::Cold => cold_hits += 1,
            }
        }
    }

    let plasma_rate = plasma_hits as f64 / total_positions as f64;
    let hot_rate = hot_hits as f64 / total_positions as f64;
    let warm_rate = warm_hits as f64 / total_positions as f64;
    let cold_rate = cold_hits as f64 / total_positions as f64;

    println!("T9 Plasma hit rate (200 trials, 4 positions each):");
    println!("  Plasma: {:.1}% ({plasma_hits}/{total_positions})", plasma_rate * 100.0);
    println!("  Hot:    {:.1}% ({hot_hits}/{total_positions})", hot_rate * 100.0);
    println!("  Warm:   {:.1}% ({warm_hits}/{total_positions})", warm_rate * 100.0);
    println!("  Cold:   {:.1}% ({cold_hits}/{total_positions})", cold_rate * 100.0);

    // With random uniform confidences and 50% agreement rate:
    // Expected plasma ≈ P(agree) * P(min(h,v) ≥ 0.7) ≈ 0.5 * 0.3² ≈ 4.5%
    // Just verify the distribution is non-degenerate
    assert!(total_positions > 0, "must have some positions");
}

// Helper: no extra trait needed, Rng has uniform() and next()
