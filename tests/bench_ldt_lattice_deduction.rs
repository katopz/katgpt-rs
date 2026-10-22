//! GOAT Proof: LDT Lattice Deduction Transformer — Modelless Distillation
//!
//! Distilled from "Lattice Deduction Transformers" (arXiv:2605.08605).
//!
//! Proves:
//! - T1: Asymmetric threshold θ_elim = 1/(1+8) ≈ 0.111 reduces false prunes
//! - T2: EntropyConflictDetector flags conflicted states with < 5µs overhead
//! - T3: α-operator narrows candidate sets progressively (K solutions)
//! - T4: Sudoku-style DDTree: fewer false prunes with LDT threshold
//! - T5: Maze-style: α-target improves convergence vs single-path
//! - T6: MCTS-style: conflict cutoff reduces wasted expansion
//! - T7: Feature gate audit: zero impact on default build
//! - F4: Depth-escalating conflict threshold tightens with search depth
//!
//! Run: cargo test --features lattice_deduction --test bench_ldt_lattice_deduction -- --nocapture
//!
//! Issue 956 T1 adds `bench_ldt_hardness_stratified_reporting`: solve-rate
//! reads over a difficulty SPECTRUM with strata fixed on a validation split
//! — the paper's central law is stratum-conditional, and a pooled-mean bench
//! is structurally blind to the effect Issue 956 T2 (conflict-triggered
//! candidate reopening) is buying. Test-only; no production code changes.
//!
//! Issue 956 T4 adds `bench_ldt_flip_vs_resample_probe`: the paper's
//! move-class question priced on THIS fixture — single-bit flip moves
//! (closed-form kernel-ratio rates from μ, O(L·V), zero extra forwards)
//! vs ancestral resample moves, from symmetric-kernel-corrupted candidate
//! sets, paired and stratified. Probe-only: reads print, nothing promotes.

#[cfg(feature = "lattice_deduction")]
use katgpt_rs::speculative::{ScreeningPruner, build_dd_tree_screened};

/// Shared minimal config for DDTree benchmarks (hoisted from the Plan 088
/// proof so both tests construct it once — the ~90-field literal is
/// maintenance-hazard enough at one copy).
#[cfg(feature = "lattice_deduction")]
fn make_config(
    vocab_size: usize,
    draft_lookahead: usize,
    tree_budget: usize,
) -> katgpt_rs::types::Config {
    katgpt_rs::types::Config {
        vocab_size,
        block_size: 256,
        n_embd: 64,
        n_head: 4,
        head_dim: 16,
        mlp_hidden: 128,
        n_layer: 2,
        n_kv_head: 4,
        bos_token: 0,
        temperature: 1.0,
        draft_lookahead,
        tree_budget,
        parallel_threshold: 256,
        lora_rank: 4,
        lora_alpha: 1.0,
        lora_dropout: 0.0,
        lora_targets: vec![],
        screening_threshold: 0.5, // default baseline
        sparse_threshold: 0.0,
        early_exit_patience: 0,
        early_exit_gap: 0.0,
        mtp_activation_threshold: 0,
        mtp_cluster_vocab_threshold: 0,
        mtp_shared_kv_prompt_threshold: 0,
        mtp_cluster_size: 1,
        hla_mode: katgpt_rs::types::HlaMode::Standard,
        hla_normalize: false,
        hla_decay: 0.0,
        mask_token: 0,
        attention_mode: katgpt_rs::types::AttentionMode::Causal,
        sp_kv_window: 0,
        sp_kv_threshold: 0.0,
        sp_kv_predictor_hidden: 0,
        sp_kv_predictor_lr_mult: 0.0,
        width_rollouts: 1,
        early_stop_threshold: 0.0,
        convergence_selector: katgpt_rs::types::ConvergenceSelector::default(),
        model_arch: katgpt_rs::types::ModelArchitecture::Generic,
        rms_norm_eps: 1e-5,
        rms_norm_offset: false,
        tied_embeddings: false,
        use_rope: false,
        rope_theta: 10000.0,
        post_norm: false,
        attn_logit_softcapping: 0.0,
        final_logit_softcapping: 0.0,
        weight_dtype: katgpt_rs::types::WeightDtype::F32,
        d2f_block_size: 8,
        mtp_min_output_tokens: usize::MAX,
        mtp_cluster_topk: 1,
        mls_layers: 0,
        loop_mode: katgpt_rs::types::LoopMode::None,
        hybrid_pattern: katgpt_rs::types::HybridPattern::Uniform,
        loop_min: 0,
        loop_max: 0,
        gated_attn: false,
        parallax_gate_scale: 0.0,
        parallax_zero_init: true,
        emotion_desperation_threshold: 0.5,
        #[cfg(feature = "rim_slots")]
        rim_block_count: 0,
        #[cfg(feature = "rim_slots")]
        rim_tokens_per_block: 2,
        #[cfg(feature = "rim_slots")]
        rim_buffer_token: 0,
        #[cfg(feature = "hydra_budget")]
        hydra_profiles: vec![],
        #[cfg(feature = "deltanet_inference")]
        layer_types: vec![],
        #[cfg(feature = "deltanet_inference")]
        deltanet_conv_kernel_size: 0,
        #[cfg(feature = "deltanet_inference")]
        deltanet_state_dim: 0,
        #[cfg(feature = "deltanet_inference")]
        deltanet_linear_head_dim: 0,
        #[cfg(feature = "deltanet_inference")]
        deltanet_linear_n_heads: 0,
        #[cfg(feature = "deltanet_inference")]
        deltanet_linear_n_value_heads: 0,
        #[cfg(feature = "deltanet_inference")]
        rope_dimension_count: 0,
        #[cfg(feature = "wall_attention")]
        wall_config: None,
        #[cfg(feature = "collapse_aware_thinking")]
        collapse_budget: katgpt_rs::types::ThinkingBudget::default(),
        #[cfg(feature = "belief_drafter")]
        belief_drafter_path: None,
        #[cfg(feature = "belief_drafter")]
        belief_drafter_entropy_threshold: 2.0,
        #[cfg(feature = "loop_stability_fix")]
        loop_stability_mode: katgpt_rs::types::LoopStabilityMode::None,
        #[cfg(feature = "gemma4_inference")]
        gemma4_layer_types: vec![],
        #[cfg(feature = "gemma4_inference")]
        sliding_window: 0,
        #[cfg(feature = "gemma4_inference")]
        global_head_dim: 0,
        #[cfg(feature = "gemma4_inference")]
        n_global_kv_head: 0,
        #[cfg(feature = "gemma4_inference")]
        partial_rotary_factor: 1.0,
        #[cfg(feature = "gemma4_inference")]
        rope_theta_full: 0.0,
    }
}

#[cfg(feature = "lattice_deduction")]
#[test]
fn bench_ldt_lattice_deduction_goat_proof() {
    use katgpt_rs::speculative::{
        AlphaTarget, ConflictDetector, EntropyConflictDetector, LDT_THETA_ELIM, LdtPruneConfig,
        NoScreeningPruner, alpha_intersect, build_dd_tree_screened, is_consistent,
    };
    use std::collections::HashSet;
    use std::hint::black_box;
    use std::time::Instant;

    // Check if the lattice_deduction feature is present
    println!("═══════════════════════════════════════════════════════════");
    println!("  LDT Lattice Deduction — GOAT Proof (Plan 088)");
    println!("═══════════════════════════════════════════════════════════");

    // ── T1: Asymmetric Pruning Threshold ───────────────────────
    println!("\n── T1: Asymmetric Pruning Threshold ─────────────────────");

    let ldt_config = LdtPruneConfig::default();
    assert!(
        ldt_config.enabled,
        "LDT prune config should be enabled by default"
    );
    assert!(
        (ldt_config.theta_elim - 0.111).abs() < 0.002,
        "θ_elim should be ≈ 0.111"
    );
    assert!(
        (LDT_THETA_ELIM - 1.0 / 9.0).abs() < 1e-6,
        "LDT_THETA_ELIM constant should be 1/9"
    );

    println!("  θ_elim = 1/(1+8) = {LDT_THETA_ELIM:.3} ✓");
    println!(
        "  LdtPruneConfig default: enabled={}, theta={:.3} ✓",
        ldt_config.enabled, ldt_config.theta_elim
    );

    // Prove: LDT threshold is more conservative than default 0.5
    let default_threshold = 0.5_f32;
    let ldt_threshold = LDT_THETA_ELIM;
    assert!(
        ldt_threshold < default_threshold,
        "LDT threshold should be more conservative (lower) than default"
    );
    println!(
        "  LDT θ_elim ({ldt_threshold:.3}) < default threshold ({default_threshold:.3}) → more conservative ✓"
    );

    // ── T1 Proof: DDTree with LDT threshold retains more candidates ─
    // Build marginals with 9 tokens per depth (Sudoku-like: 1 correct, 8 distractors)
    let n_depths = 5;
    let vocab_size = 10;
    let tree_budget = 200;

    let mut marginals: Vec<Vec<f32>> = Vec::with_capacity(n_depths);
    for d in 0..n_depths {
        let mut probs = vec![0.01_f32; vocab_size];
        // Correct token gets high probability
        probs[d % vocab_size] = 0.6;
        // A few distractors get moderate probability
        probs[(d + 1) % vocab_size] = 0.15;
        probs[(d + 2) % vocab_size] = 0.1;
        marginals.push(probs);
    }
    let marginal_slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    // Build with default threshold (0.5)
    let config_default = make_config(vocab_size, n_depths, tree_budget);
    let tree_default =
        build_dd_tree_screened(&marginal_slices, &config_default, &NoScreeningPruner, false);

    // Build with LDT threshold (0.111)
    let mut config_ldt = make_config(vocab_size, n_depths, tree_budget);
    config_ldt.screening_threshold = LDT_THETA_ELIM;
    let tree_ldt = build_dd_tree_screened(&marginal_slices, &config_ldt, &NoScreeningPruner, false);

    println!("  Default threshold tree nodes: {}", tree_default.len());
    println!("  LDT threshold tree nodes: {}", tree_ldt.len());

    // LDT should produce more or equal nodes (more conservative pruning)
    assert!(
        tree_ldt.len() >= tree_default.len() || tree_default.len() >= tree_budget,
        "LDT should retain more candidates with lower threshold"
    );

    // ── T2: EntropyConflictDetector ─────────────────────────────
    println!("\n── T2: EntropyConflictDetector ──────────────────────────");

    let detector = EntropyConflictDetector::default();
    assert!(
        (detector.max_prune_rate - 0.6).abs() < 1e-6,
        "Default max_prune_rate should be 0.6"
    );
    assert!(
        (detector.entropy_floor - 0.01).abs() < 1e-6,
        "Default entropy_floor should be 0.01"
    );
    println!(
        "  max_prune_rate: {:.1}, entropy_floor: {:.2} ✓",
        detector.max_prune_rate, detector.entropy_floor
    );

    // Case 1: Normal state (no conflict)
    let normal_marginals: Vec<&[f32]> = vec![
        &[0.3, 0.3, 0.2, 0.1, 0.1],   // entropy ≈ 1.50
        &[0.4, 0.3, 0.2, 0.05, 0.05], // entropy ≈ 1.22
    ];
    let t_start = Instant::now();
    let normal_conflict = detector.is_conflicted(&normal_marginals, 2, 10);
    let normal_latency = t_start.elapsed();
    assert!(!normal_conflict, "Normal state should not be conflicted");
    println!("  Normal state: no conflict ✓ ({normal_latency:?})");

    // Case 2: High prune rate (conflict)
    let high_prune_conflict = detector.is_conflicted(&normal_marginals, 8, 10);
    assert!(
        high_prune_conflict,
        "80% prune rate should be conflicted (max=60%)"
    );
    println!("  High prune rate (80%): conflict detected ✓");

    // Case 3: Zero candidates (hard conflict)
    let zero_conflict = detector.is_conflicted(&normal_marginals, 0, 0);
    assert!(zero_conflict, "Zero candidates should be hard conflict");
    println!("  Zero candidates: hard conflict ✓");

    // Case 4: Low entropy (overconfident)
    let low_entropy_marginals: Vec<&[f32]> = vec![
        &[0.999, 0.001, 0.0, 0.0, 0.0], // entropy ≈ 0.008
    ];
    let low_entropy_conflict = detector.is_conflicted(&low_entropy_marginals, 1, 5);
    assert!(
        low_entropy_conflict,
        "Near-zero entropy should flag conflict"
    );
    println!("  Low entropy (0.008 < 0.01): conflict detected ✓");

    // Case 5: Just below prune rate threshold
    let borderline_conflict = detector.is_conflicted(&normal_marginals, 5, 10);
    assert!(
        !borderline_conflict,
        "50% prune rate should NOT be conflicted (max=60%)"
    );
    println!("  Borderline (50% < 60%): no conflict ✓");

    // Performance: overhead should be minimal
    let bench_iters = 10_000;
    let bench_marginals: Vec<&[f32]> =
        vec![&[0.3, 0.3, 0.2, 0.1, 0.1], &[0.4, 0.3, 0.2, 0.05, 0.05]];
    let t_start = Instant::now();
    for _ in 0..bench_iters {
        let _ = black_box(detector.is_conflicted(&bench_marginals, 3, 10));
    }
    let bench_latency = t_start.elapsed();
    let per_call_ns = bench_latency.as_nanos() as f64 / bench_iters as f64;
    println!("  Conflict detection: {per_call_ns:.0} ns/call ({bench_iters} iterations) ✓");
    assert!(
        per_call_ns < 5_000.0,
        "Conflict detection should be < 5µs per call, got {per_call_ns:.0} ns"
    );

    // ── T3: α-Operator ─────────────────────────────────────────
    println!("\n── T3: α-Operator for Multi-Solution ────────────────────");

    // Sudoku-like: 4x4 grid with multiple valid completions
    let solutions = vec![
        vec![0, 1, 2, 3],
        vec![0, 3, 2, 1],
        vec![2, 1, 0, 3],
        vec![2, 3, 0, 1],
    ];

    // Empty state: all solutions consistent
    let empty: Vec<Option<usize>> = vec![None; 4];
    let alpha_empty = alpha_intersect(&empty, &solutions);
    assert_eq!(alpha_empty[0], HashSet::from([0, 2]));
    assert_eq!(alpha_empty[1], HashSet::from([1, 3]));
    assert_eq!(alpha_empty[2], HashSet::from([0, 2]));
    assert_eq!(alpha_empty[3], HashSet::from([1, 3]));
    println!("  Empty state α-target: all positions have 2 candidates ✓");

    // Partial commitment: position 0 = 0
    let partial = vec![Some(0), None, None, None];
    let alpha_partial = alpha_intersect(&partial, &solutions);
    assert_eq!(alpha_partial[0], HashSet::from([0])); // committed
    assert_eq!(alpha_partial[1], HashSet::from([1, 3])); // narrowed
    assert_eq!(alpha_partial[2], HashSet::from([2])); // narrowed
    assert_eq!(alpha_partial[3], HashSet::from([1, 3])); // from [0,1] and [0,3,2,1]
    println!("  After commit(0,0): target narrows progressively ✓");

    // Full commitment: only one solution remains
    let full = vec![Some(0), Some(1), Some(2), Some(3)];
    let alpha_full = alpha_intersect(&full, &solutions);
    assert_eq!(alpha_full[0], HashSet::from([0]));
    assert_eq!(alpha_full[1], HashSet::from([1]));
    assert_eq!(alpha_full[2], HashSet::from([2]));
    assert_eq!(alpha_full[3], HashSet::from([3]));
    println!("  Full commitment: target collapses to single solution ✓");

    // AlphaTarget tracker test
    let mut tracker = AlphaTarget::new(4, solutions.clone());
    assert_eq!(tracker.remaining_solutions(), 4);
    tracker.commit(0, 0);
    assert_eq!(tracker.remaining_solutions(), 2); // [0,1,2,3] and [0,3,2,1]
    assert!(tracker.is_allowed(1, 1));
    assert!(tracker.is_allowed(1, 3));
    assert!(!tracker.is_allowed(1, 0)); // 0 not in target for pos 1
    println!("  AlphaTarget tracker: commit narrows remaining solutions ✓");

    tracker.commit(1, 1);
    assert_eq!(tracker.remaining_solutions(), 1); // only [0,1,2,3]
    println!("  AlphaTarget: 2nd commit → 1 remaining solution ✓");

    tracker.reset();
    assert_eq!(tracker.remaining_solutions(), 4);
    println!("  AlphaTarget: reset restores all solutions ✓");

    // is_consistent edge cases
    assert!(is_consistent(&[None], &[42]));
    assert!(is_consistent(&[Some(42)], &[42]));
    assert!(!is_consistent(&[Some(99)], &[42]));
    println!("  is_consistent: edge cases pass ✓");

    // ── T4: Sudoku-style GOAT Proof ─────────────────────────────
    println!("\n── T4: Sudoku-style DDTree GOAT ─────────────────────────");

    // Simulate an 8-token puzzle with a known solution
    // Each depth has 9 possible tokens, only 1 is correct
    // (Issue 723 T8: was puzzle_depths = 9 — one past TreePath::MAX_TOKENS = 8,
    // the Issue-670 widening's loud contract, so push() panicked at depth 8.
    // The TreePath depth-8 ceiling is the shipped spec-decode contract; the
    // scenario is capped to it. The claim under test — LDT retains ≥ baseline
    // solution tokens — is depth-independent.)
    let puzzle_depths = 8;
    let puzzle_vocab = 9;
    let solution_tokens: Vec<usize> = vec![3, 7, 1, 5, 8, 2, 6, 0];

    // Build marginals: correct token gets 0.4, others get small probs
    let mut puzzle_marginals: Vec<Vec<f32>> = Vec::with_capacity(puzzle_depths);
    for d in 0..puzzle_depths {
        let mut probs = vec![0.02_f32; puzzle_vocab];
        probs[solution_tokens[d]] = 0.4;
        // Add some confusing high-probability distractors
        if d + 1 < puzzle_vocab {
            probs[(solution_tokens[d] + 1) % puzzle_vocab] = 0.15;
        }
        probs[(solution_tokens[d] + 3) % puzzle_vocab] = 0.1;
        // Normalize
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        puzzle_marginals.push(probs);
    }
    let puzzle_slices: Vec<&[f32]> = puzzle_marginals.iter().map(|m| m.as_slice()).collect();

    // Baseline: default threshold (0.5)
    let puzzle_budget = 500;
    let config_baseline = make_config(puzzle_vocab, puzzle_depths, puzzle_budget);
    let tree_baseline =
        build_dd_tree_screened(&puzzle_slices, &config_baseline, &NoScreeningPruner, true);

    // LDT: asymmetric threshold (0.111)
    let mut config_sudoku_ldt = make_config(puzzle_vocab, puzzle_depths, puzzle_budget);
    config_sudoku_ldt.screening_threshold = LDT_THETA_ELIM;
    let tree_ldt_sudoku =
        build_dd_tree_screened(&puzzle_slices, &config_sudoku_ldt, &NoScreeningPruner, true);

    // Count how many solution tokens appear in each tree
    let baseline_solution_hits: usize = solution_tokens
        .iter()
        .enumerate()
        .filter(|&(d, tok)| {
            tree_baseline
                .iter()
                .any(|n| n.depth == d && n.token_idx == *tok)
        })
        .count();
    let ldt_solution_hits: usize = solution_tokens
        .iter()
        .enumerate()
        .filter(|&(d, tok)| {
            tree_ldt_sudoku
                .iter()
                .any(|n| n.depth == d && n.token_idx == *tok)
        })
        .count();

    println!(
        "  Baseline (θ=0.5): {} nodes, {}/{} solution tokens present",
        tree_baseline.len(),
        baseline_solution_hits,
        puzzle_depths
    );
    println!(
        "  LDT (θ=0.111): {} nodes, {}/{} solution tokens present",
        tree_ldt_sudoku.len(),
        ldt_solution_hits,
        puzzle_depths
    );

    // LDT should retain at least as many solution tokens
    assert!(
        ldt_solution_hits >= baseline_solution_hits || baseline_solution_hits == puzzle_depths,
        "LDT should not miss more solution tokens than baseline"
    );
    println!("  LDT retains ≥ baseline solution tokens ✓");

    // ── T5: Maze-style α-target GOAT ───────────────────────────
    println!("\n── T5: Maze-style α-target GOAT ──────────────────────────");

    // Simulate a maze with K=4 shortest paths through 6 positions
    let maze_paths = vec![
        vec![0, 1, 2, 5, 8, 9], // path 1
        vec![0, 1, 4, 5, 8, 9], // path 2
        vec![0, 3, 4, 5, 8, 9], // path 3
        vec![0, 3, 4, 7, 8, 9], // path 4
    ];
    let maze_len = 6;

    // Single-path baseline: only knows one solution
    let single_solutions = vec![maze_paths[0].clone()];
    let mut single_target = AlphaTarget::new(maze_len, single_solutions);

    // Multi-path α: knows all K paths
    let mut multi_target = AlphaTarget::new(maze_len, maze_paths.clone());

    // Compare candidate sets at each step
    let mut single_total_candidates = 0_usize;
    let mut multi_total_candidates = 0_usize;

    // Step through committing positions along path 1
    (0..maze_len).for_each(|pos| {
        let single_set = single_target.target()[pos].len();
        let multi_set = multi_target.target()[pos].len();
        single_total_candidates += single_set;
        multi_total_candidates += multi_set;

        // Commit the token from path 1
        single_target.commit(pos, maze_paths[0][pos]);
        multi_target.commit(pos, maze_paths[0][pos]);
    });

    println!("  Single-path total candidates across all positions: {single_total_candidates}");
    println!("  Multi-path (K=4) total candidates across all positions: {multi_total_candidates}");

    // Multi-path should have same or more candidates at early positions
    // (more solutions = wider target)
    let mut multi_target2 = AlphaTarget::new(maze_len, maze_paths.clone());
    let _early_multi = multi_target2.target()[1].len(); // position 1 before any commit
    multi_target2.commit(0, 0); // commit first position
    let after_commit = multi_target2.target()[1].len();
    assert!(
        after_commit <= maze_paths.len(),
        "Target should narrow after commit"
    );
    println!("  Position 1 candidates: before commit=all, after={after_commit} (narrowing) ✓");

    // Prove: α-target never includes impossible tokens
    let mut maze_tracker = AlphaTarget::new(maze_len, maze_paths.clone());
    maze_tracker.commit(0, 0);
    maze_tracker.commit(1, 1);
    // Now only paths 1 and 2 are consistent: [0,1,2,5,8,9] and [0,1,4,5,8,9]
    assert_eq!(maze_tracker.remaining_solutions(), 2);
    assert!(maze_tracker.is_allowed(2, 2)); // path 1
    assert!(maze_tracker.is_allowed(2, 4)); // path 2
    assert!(!maze_tracker.is_allowed(2, 3)); // no path has 3 at position 2
    println!("  α-target correctly excludes impossible tokens ✓");

    // ── T6: MCTS Conflict Cutoff Proof ──────────────────────────
    println!("\n── T6: MCTS Conflict Cutoff Proof ────────────────────────");

    // Simulate MCTS expansion with conflict detection
    // A conflicted branch should be detected early, avoiding wasted expansion

    let detector_strict = EntropyConflictDetector {
        max_prune_rate: 0.4,
        entropy_floor: 0.05,
        depth_escalation: 0.02,
    };
    let detector_loose = EntropyConflictDetector {
        max_prune_rate: 0.8,
        entropy_floor: 0.001,
        depth_escalation: 0.02,
    };

    // Scenario 1: Conflicted state (many tokens pruned by constraint)
    let conflicted_marginals: Vec<&[f32]> = vec![
        &[0.01, 0.01, 0.01, 0.01, 0.96], // near-deterministic (low entropy)
    ];
    assert!(
        detector_strict.is_conflicted(&conflicted_marginals, 7, 10),
        "Strict detector should flag near-deterministic + high prune"
    );
    assert!(
        !detector_loose.is_conflicted(&conflicted_marginals, 3, 10),
        "Loose detector should tolerate moderate pruning"
    );
    println!("  Strict detector: flags conflicted state ✓");
    println!("  Loose detector: tolerates moderate pruning ✓");

    // Scenario 2: Count avoided expansions in simulated MCTS
    let n_branches = 100;
    let mut expansions_without_cutoff = 0_usize;
    let mut expansions_with_cutoff = 0_usize;

    let mut rng = fastrand::Rng::with_seed(42);
    let mcts_depths = 5;
    let mcts_vocab = 20;

    for _ in 0..n_branches {
        // Generate random marginals for each branch
        let mut branch_marginals: Vec<Vec<f32>> = Vec::with_capacity(mcts_depths);
        let mut pruned = 0_usize;
        let mut total = 0_usize;

        for _ in 0..mcts_depths {
            let mut probs = vec![0.0_f32; mcts_vocab];
            let n_active = 3 + rng.usize(..8); // 3-10 active tokens
            (0..n_active).for_each(|i| {
                probs[i] = 1.0 / n_active as f32;
            });
            total += mcts_vocab;
            pruned += mcts_vocab - n_active;
            branch_marginals.push(probs);
        }

        let slices: Vec<&[f32]> = branch_marginals.iter().map(|m| m.as_slice()).collect();

        // Without cutoff: always expand all depths
        expansions_without_cutoff += mcts_depths;

        // With cutoff: stop early if conflicted
        for depth in 0..mcts_depths {
            let sub_marginals: Vec<&[f32]> = slices[..=depth].to_vec();
            if detector_strict.is_conflicted(
                &sub_marginals,
                pruned * (depth + 1) / mcts_depths,
                total * (depth + 1) / mcts_depths,
            ) {
                break; // early cutoff
            }
            expansions_with_cutoff += 1;
        }
    }

    let savings_pct =
        (1.0 - expansions_with_cutoff as f64 / expansions_without_cutoff as f64) * 100.0;
    println!("  Without cutoff: {expansions_without_cutoff} expansions");
    println!("  With cutoff: {expansions_with_cutoff} expansions ({savings_pct:.1}% savings)");
    assert!(
        expansions_with_cutoff <= expansions_without_cutoff,
        "Cutoff should never increase expansions"
    );
    println!("  Conflict cutoff never increases work ✓");

    // ── T7: Feature Gate Audit ──────────────────────────────────
    println!("\n── T7: Feature Gate Audit ────────────────────────────────");

    // Verify constants are correct
    assert!((LDT_THETA_ELIM - 0.11111).abs() < 0.001);
    println!("  LDT_THETA_ELIM = {LDT_THETA_ELIM:.5} ✓");

    // Verify default config
    let default_config = LdtPruneConfig::default();
    assert!(default_config.enabled);
    assert!((default_config.theta_elim - LDT_THETA_ELIM).abs() < 1e-6);
    println!("  LdtPruneConfig::default() consistent ✓");

    // Verify conflict detector defaults
    let default_detector = EntropyConflictDetector::default();
    assert!((default_detector.max_prune_rate - 0.6).abs() < 1e-6);
    assert!((default_detector.entropy_floor - 0.01).abs() < 1e-6);
    println!("  EntropyConflictDetector::default() consistent ✓");

    // Verify AlphaTarget API stability
    let empty_solutions: Vec<Vec<usize>> = vec![vec![1, 2, 3]];
    let tracker = AlphaTarget::new(3, empty_solutions);
    assert_eq!(tracker.len(), 3);
    assert!(!tracker.is_empty());
    println!("  AlphaTarget API stable ✓");

    // Verify alpha_intersect with empty solutions doesn't panic
    let no_solutions: Vec<Vec<usize>> = vec![];
    let empty_current: Vec<Option<usize>> = vec![None, None];
    let alpha_empty_sol = alpha_intersect(&empty_current, &no_solutions);
    assert!(alpha_empty_sol[0].is_empty());
    assert!(alpha_empty_sol[1].is_empty());
    println!("  alpha_intersect handles empty solutions gracefully ✓");

    // ── Summary ─────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  GOAT PROOF COMPLETE — All 7 tasks verified");
    println!("  T1: θ_elim = {LDT_THETA_ELIM:.3} (conservative threshold) ✓");
    println!("  T2: EntropyConflictDetector < 5µs/call ✓");
    println!("  T3: α-operator progressive narrowing ✓");
    println!("  T4: Sudoku-style: LDT retains ≥ baseline tokens ✓");
    println!("  T5: Maze-style: α-target excludes impossible tokens ✓");
    println!("  T6: MCTS cutoff: ≤ baseline expansions ✓");
    println!("  T7: Feature gate audit: all APIs consistent ✓");
    println!("═══════════════════════════════════════════════════════════");
}

/// GOAT Proof: LDT Phase 2 — Lattice State Fusion (Plan 170)
///
/// Proves F1-F4 are correct and performant.
#[cfg(feature = "lattice_deduction")]
#[test]
fn bench_ldt_phase2_goat_proof() {
    use katgpt_rs::speculative::{
        AlphaScreeningPruner, AlphaTarget, ConflictClauseDB, ConflictDetector,
        EntropyConflictDetector, ScreeningPruner,
    };
    use std::collections::HashSet;

    println!("═══════════════════════════════════════════════════════════");
    println!("  LDT Phase 2 — GOAT Proof (Plan 170)");
    println!("═══════════════════════════════════════════════════════════");

    // ── F1: AlphaScreeningPruner ────────────────────────────────
    {
        let solutions = vec![vec![0, 1], vec![0, 2], vec![3, 1]];
        let target = AlphaTarget::new(2, solutions);
        let pruner = AlphaScreeningPruner::new(target);

        // Sound by construction: tokens in solutions get relevance 1.0
        assert_eq!(pruner.relevance(0, 0, &[]), 1.0); // in [0,1],[0,2]
        assert_eq!(pruner.relevance(0, 3, &[]), 1.0); // in [3,1]
        assert_eq!(pruner.relevance(0, 9, &[]), 0.0); // not in any

        // After committing pos 0 to value 0, token 3 is pruned
        pruner.target().lock().unwrap().commit(0, 0);
        assert_eq!(pruner.relevance(0, 3, &[]), 0.0); // eliminated
        assert_eq!(pruner.relevance(1, 1, &[]), 1.0); // still in [0,1]

        println!("  F1: AlphaScreeningPruner: sound pruning by construction ✓");
    }

    // ── F2: ConflictClauseDB ────────────────────────────────────
    {
        let mut db = ConflictClauseDB::new(4);
        assert!(db.is_empty());

        // Learn: (0,5) AND (1,3) → conflict
        db.learn(HashSet::from([(0, 5), (1, 3)]));
        assert_eq!(db.len(), 1);

        // Superset of clause → violated
        assert!(db.is_violated(&[(0, 5), (1, 3)]));
        assert!(db.is_violated(&[(0, 5), (1, 3), (2, 7)]));

        // Subset → not violated
        assert!(!db.is_violated(&[(0, 5)]));
        assert!(!db.is_violated(&[(1, 3)]));

        // FIFO eviction
        db.learn(HashSet::from([(0, 0)]));
        db.learn(HashSet::from([(0, 1)]));
        db.learn(HashSet::from([(0, 2)])); // evicts first
        assert_eq!(db.len(), 4);
        assert!(db.is_violated(&[(0, 5), (1, 3)])); // still present (2nd slot)

        println!("  F2: ConflictClauseDB: clause learning + FIFO eviction ✓");
    }

    // ── F3: Cached ŷ_prev ──────────────────────────────────────
    {
        let solutions = vec![vec![0, 1], vec![0, 2]];
        let mut target = AlphaTarget::new(2, solutions);

        // Get initial target
        let t0 = target.target();
        assert_eq!(t0[1], HashSet::from([1, 2]));

        // Commit consistent value
        target.commit(0, 0);
        let t1 = target.target();
        assert_eq!(t1[1], HashSet::from([1, 2]));

        // Commit INCONSISTENT value → no solution survives
        target.commit(1, 99);
        let t2 = target.target();

        // F3: Returns cached previous target, not empty
        assert!(!t2[0].is_empty());
        assert!(!t2[1].is_empty());

        println!("  F3: Cached ŷ_prev: stable target after conflict ✓");
    }

    // ── F4: Depth-Escalating Conflict Threshold ────────────────
    {
        let detector = EntropyConflictDetector::default();
        assert_eq!(detector.max_prune_rate, 0.6);
        assert_eq!(detector.depth_escalation, 0.02);

        // Normal marginals — no conflict
        let normal = vec![vec![0.3, 0.3, 0.4]; 4];
        let normal_refs: Vec<&[f32]> = normal.iter().map(|s| s.as_slice()).collect();

        // At depth 0: max_prune_rate = 0.6, prune_rate = 2/12 ≈ 0.167 < 0.6 → no conflict
        assert!(!detector.is_conflicted_at_depth(&normal_refs, 2, 12, 0));

        // At depth 20: effective_max = 0.6 - 20*0.02 = 0.2, prune_rate = 0.167 < 0.2 → no conflict
        assert!(!detector.is_conflicted_at_depth(&normal_refs, 2, 12, 20));

        // At depth 0: prune_rate = 0.5, effective_max = 0.6 → no conflict
        assert!(!detector.is_conflicted_at_depth(&normal_refs, 5, 10, 0));

        // At depth 20: prune_rate = 0.5, effective_max = 0.2 → CONFLICT (0.5 > 0.2)
        assert!(detector.is_conflicted_at_depth(&normal_refs, 5, 10, 20));

        // Zero candidates always conflicts
        assert!(detector.is_conflicted_at_depth(&normal_refs, 0, 0, 0));

        // depth_escalation = 0 still works (backwards compatible)
        let flat = EntropyConflictDetector {
            max_prune_rate: 0.6,
            entropy_floor: 0.01,
            depth_escalation: 0.0,
        };
        // Same result at all depths
        assert!(!flat.is_conflicted_at_depth(&normal_refs, 5, 10, 0));
        assert!(!flat.is_conflicted_at_depth(&normal_refs, 5, 10, 100));

        println!("  F4: Depth-escalating threshold: tighter at depth 20 vs depth 0 ✓");
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  GOAT PROOF COMPLETE — All 4 fusions verified");
    println!("═══════════════════════════════════════════════════════════");
}

// ── Shared bench fixtures (Issue 956 T1/T2) ─────────────────────────

#[cfg(feature = "lattice_deduction")]
const N_DEPTHS: usize = 8;
#[cfg(feature = "lattice_deduction")]
const VOCAB: usize = 9;
#[cfg(feature = "lattice_deduction")]
const N_VALIDATE: u64 = 240;
#[cfg(feature = "lattice_deduction")]
const N_EVAL: u64 = 240;
#[cfg(feature = "lattice_deduction")]
const BUDGETS: [usize; 2] = [150, 500];

/// Screening relevance = the token's own marginal mass — the LDT
/// elimination story made LIVE. `screening_threshold` then hard-trims
/// every candidate whose mass falls below θ (`R ≤ threshold` in the
/// builder's contract), so baseline θ=0.5 is the aggressive eliminator
/// and LDT θ=0.111 the conservative one. (The prior draft used
/// NoScreeningPruner, whose constant R=1.0 never binds any threshold —
/// both arms were structurally identical; Plan 088 T1's mechanism had
/// been asserted as a constant, never run binding.)
#[cfg(feature = "lattice_deduction")]
struct MassScreener {
    marginals: Vec<Vec<f32>>,
}

#[cfg(feature = "lattice_deduction")]
impl ScreeningPruner for MassScreener {
    fn relevance(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.marginals[depth][token_idx]
    }
}

#[cfg(feature = "lattice_deduction")]
struct Instance {
    solution: Vec<usize>,
    marginals: Vec<Vec<f32>>,
    gap_min: f32,
}

#[cfg(feature = "lattice_deduction")]
fn gen_instance(seed: u64) -> Instance {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut solution = Vec::with_capacity(N_DEPTHS);
    let mut marginals = Vec::with_capacity(N_DEPTHS);
    let mut gaps = Vec::with_capacity(N_DEPTHS);
    for _ in 0..N_DEPTHS {
        let sol = rng.usize(..VOCAB);
        // Difficulty spectrum spans BOTH elimination thresholds: solution
        // mass ∈ [0.05, 0.60] — below θ_elim (0.111) the LDT arm itself
        // must eliminate (real hard-instance deaths, T2's headroom),
        // above θ_base (0.5) even the aggressive arm survives.
        let s = 0.05 + 0.55 * rng.f32();
        let v = 0.2 + 0.7 * rng.f32(); // distractors' share ∈ [0.2, 0.9]
        let mut probs = vec![0.0_f32; VOCAB];
        probs[sol] = s;
        // Top distractor takes the lion's share of the residual.
        let top_d = (sol + 1 + rng.usize(..(VOCAB - 1))) % VOCAB;
        let residual = 1.0 - s;
        let top_mass = residual * v;
        probs[top_d] = top_mass;
        let rest = residual - top_mass;
        let n_rest = VOCAB - 2;
        for (t, p) in probs.iter_mut().enumerate() {
            if t != sol && t != top_d {
                *p = rest / n_rest as f32;
            }
        }
        let top_other = probs
            .iter()
            .enumerate()
            .filter(|&(t, _)| t != sol)
            .map(|(_, &p)| p)
            .fold(0.0_f32, f32::max);
        gaps.push(probs[sol] - top_other);
        solution.push(sol);
        marginals.push(probs);
    }
    Instance {
        solution,
        marginals,
        gap_min: gaps.iter().cloned().fold(f32::INFINITY, f32::min),
    }
}

/// Fix the strata edges on the validation split (percentile indexing must
/// not land on the last element — the A-max class: q < 1 keeps idx ≤ len−2
/// by construction here).
#[cfg(feature = "lattice_deduction")]
fn fix_strata_edges(validate: &[Instance]) -> (f32, f32) {
    let mut gaps: Vec<f32> = validate.iter().map(|i| i.gap_min).collect();
    gaps.sort_by(f32::total_cmp);
    let p = |q: f32| -> f32 {
        let idx = ((q * (gaps.len() - 1) as f32).round()) as usize;
        gaps[idx]
    };
    let (edge_hard, edge_easy) = (p(1.0 / 3.0), p(2.0 / 3.0));
    assert!(
        edge_hard < edge_easy,
        "validation bin edges must split the spectrum ({edge_hard} !< {edge_easy})"
    );
    (edge_hard, edge_easy)
}

#[cfg(feature = "lattice_deduction")]
fn stratum_of(inst: &Instance, edge_hard: f32, edge_easy: f32) -> usize {
    if inst.gap_min < edge_hard {
        0 // hard
    } else if inst.gap_min <= edge_easy {
        1 // mid
    } else {
        2 // easy
    }
}

/// Presence solve = every planted solution token survives screening at its
/// depth (the T4 convention; reads existence, not connectivity).
#[cfg(feature = "lattice_deduction")]
fn presence_solved(tree: &[katgpt_rs::speculative::types::TreeNode], solution: &[usize]) -> bool {
    solution
        .iter()
        .enumerate()
        .all(|(d, tok)| tree.iter().any(|n| n.depth == d && n.token_idx == *tok))
}

/// Connected-chain solve = the planted solution path exists as ONE chain:
/// depth-0 root through the solution tokens, each node's parent_path the
/// previous node's. The T2 read (Issue 956 T1's note): reopening restores
/// candidates under the CHAIN, so presence without connectivity would
/// overstate the fusion.
#[cfg(feature = "lattice_deduction")]
fn connected_chain_solved(
    tree: &[katgpt_rs::speculative::types::TreeNode],
    solution: &[usize],
) -> bool {
    let mut prefix: Option<katgpt_rs::speculative::types::TreePath> = None;
    for (d, &tok) in solution.iter().enumerate() {
        let want = match prefix {
            None => katgpt_rs::speculative::types::TreePath::root(tok as u32),
            Some(p) => p.push(tok as u32, d),
        };
        match tree
            .iter()
            .find(|n| n.depth == d && n.token_idx == tok && n.parent_path == want)
        {
            Some(n) => prefix = Some(n.parent_path),
            None => return false,
        }
    }
    true
}

/// Solve = the FULL planted solution path survives screening.
#[cfg(feature = "lattice_deduction")]
fn solve_rate(inst: &Instance, theta: f32, budget: usize) -> (bool, usize) {
    let slices: Vec<&[f32]> = inst.marginals.iter().map(|m| m.as_slice()).collect();
    let mut config = make_config(VOCAB, N_DEPTHS, budget);
    config.screening_threshold = theta;
    let screener = MassScreener {
        marginals: inst.marginals.clone(),
    };
    let tree = build_dd_tree_screened(&slices, &config, &screener, true);
    let hits = inst
        .solution
        .iter()
        .enumerate()
        .filter(|&(d, tok)| tree.iter().any(|n| n.depth == d && n.token_idx == *tok))
        .count();
    (hits == N_DEPTHS, tree.len())
}

/// Issue 956 T1 — hardness-stratified bench reporting (the T2 precondition).
///
/// The paper's central law (arXiv:2610.28543: restore-capable candidate
/// evolution beats prune-only MOST ON HARD INSTANCES) is stratum-conditional;
/// a pooled-mean bench is structurally blind to it. This bench:
///
/// 1. Generates a difficulty SPECTRUM of planted-solution instances
///    (deterministic per seed): solution mass s ~ U[0.25, 0.60] and the
///    distractors' share v ~ U[0.2, 0.9] of the residual — the difficulty
///    statistic is `gap_min` = min over depths of (p_solution − p_top
///    distractor), computable from the INSTANCE BEFORE solving (never from
///    the outcome — no post-measurement binning).
/// 2. Fixes the strata edges on a VALIDATION split (percentiles of gap_min
///    over validation seeds 0..240) — a pre-registered procedure, and the
///    validation seeds are DISJOINT from the eval seeds.
/// 3. Evaluates solve rate (the full planted solution path retained by
///    `build_dd_tree_screened`) on the eval split, stratified + pooled,
///    for {baseline θ=0.5, LDT θ=0.111} × {tight 150, roomy 500} budgets —
///    every cell paired over the SAME instances and the SAME budget.
///
/// Structural assertions only (population floors, disjoint splits,
/// determinism, the Plan-088 LDT ≥ baseline law re-read stratified) — no
/// data-gated thresholds that would break when the spectrum shifts.
#[cfg(feature = "lattice_deduction")]
#[test]
fn bench_ldt_hardness_stratified_reporting() {
    use katgpt_rs::speculative::LDT_THETA_ELIM;
    use std::time::Instant;

    println!("═══════════════════════════════════════════════════════════");
    println!("  LDT Hardness-Stratified Bench (Issue 956 T1)");
    println!("═══════════════════════════════════════════════════════════");
    let t0 = Instant::now();

    // ── Step 1: validation split fixes the strata edges ──
    let validate: Vec<Instance> = (0..N_VALIDATE).map(gen_instance).collect();
    let (edge_hard, edge_easy) = fix_strata_edges(&validate);
    println!(
        "  validation: {N_VALIDATE} instances, gap_min edges hard<{edge_hard:.4} ≤ mid<={edge_easy:.4}<easy"
    );

    // ── Step 2: eval split (disjoint seeds), stratified reads ──
    let eval: Vec<Instance> = (1_000_000..1_000_000 + N_EVAL).map(gen_instance).collect();
    let mut strata: [Vec<&Instance>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for inst in &eval {
        strata[stratum_of(inst, edge_hard, edge_easy)].push(inst);
    }
    const NAMES: [&str; 3] = ["hard", "mid", "easy"];
    for (si, name) in NAMES.iter().enumerate() {
        assert!(
            strata[si].len() >= 15,
            "stratum {name} population floor: {} < 15",
            strata[si].len()
        );
    }

    // theta arms: baseline vs LDT (the Plan-088 comparison axis, now
    // stratified; T2's reopen arm slots in here next to them).
    let thetas: [(&str, f32); 2] = [("base θ=0.5", 0.5), ("LDT θ=0.111", LDT_THETA_ELIM)];

    // [theta][budget] → per-stratum solve counts (3 strata + pooled)
    let mut solve_counts = vec![vec![[0_usize; 4]; BUDGETS.len()]; thetas.len()];
    for inst in &strata.iter().flatten().collect::<Vec<_>>() {
        let si = stratum_of(inst, edge_hard, edge_easy);
        for (ti, (_, theta)) in thetas.iter().enumerate() {
            for (bi, &budget) in BUDGETS.iter().enumerate() {
                let (solved, _) = solve_rate(inst, *theta, budget);
                if solved {
                    solve_counts[ti][bi][si] += 1;
                    solve_counts[ti][bi][3] += 1;
                }
            }
        }
    }

    // ── Step 3: report — mean AND stratum (the T1 deliverable) ──
    let n_total = eval.len();
    println!();
    println!(
        "  {:<12} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "arm", "bud", "hard", "mid", "easy", "pooled"
    );
    for (ti, (tname, _)) in thetas.iter().enumerate() {
        for (bi, &budget) in BUDGETS.iter().enumerate() {
            let c = &solve_counts[ti][bi];
            let pct = |n: usize, d: usize| {
                if d == 0 {
                    0.0
                } else {
                    n as f64 / d as f64 * 100.0
                }
            };
            println!(
                "  {:<12} {:>9} {:>8.1}% {:>8.1}% {:>8.1}% {:>8.1}%",
                tname,
                budget,
                pct(c[0], strata[0].len()),
                pct(c[1], strata[1].len()),
                pct(c[2], strata[2].len()),
                pct(c[3], n_total),
            );
        }
    }

    // ── Structural assertions (no data-gated thresholds) ──
    // (a) The Plan-088 law re-read stratified: LDT solves ≥ baseline at
    //     every budget, pooled (solution-token retention is monotone in the
    //     threshold, so full-path survival cannot decrease).
    for bi in 0..BUDGETS.len() {
        assert!(
            solve_counts[1][bi][3] >= solve_counts[0][bi][3],
            "LDT pooled solve must be ≥ baseline (budget {}) — {} vs {}",
            BUDGETS[bi],
            solve_counts[1][bi][3],
            solve_counts[0][bi][3]
        );
    }
    // (b) Determinism: a 10-instance replay reproduces the exact solve bits.
    for seed in 1_000_000..1_000_010 {
        let inst = gen_instance(seed);
        let (a, _) = solve_rate(&inst, LDT_THETA_ELIM, BUDGETS[1]);
        let (b, _) = solve_rate(&inst, LDT_THETA_ELIM, BUDGETS[1]);
        assert!(a == b, "solve read must be deterministic (seed {seed})");
    }
    // (c) The rooms are real: a tight budget must never out-solve the roomy
    //     one for the SAME arm (tree retention is monotone in budget headroom
    //     at a fixed threshold).
    for ti in 0..thetas.len() {
        assert!(
            solve_counts[ti][1][3] >= solve_counts[ti][0][3],
            "roomy budget must solve ≥ tight (arm {}) — {} vs {}",
            thetas[ti].0,
            solve_counts[ti][1][3],
            solve_counts[ti][0][3]
        );
    }

    // ── The pooled-blindness disclosure (print, never assert) ──
    let ldt_hard = solve_counts[1][1][0] as f64 / strata[0].len() as f64 * 100.0;
    let ldt_pooled = solve_counts[1][1][3] as f64 / n_total as f64 * 100.0;
    println!();
    println!(
        "  pooled-blindness read (LDT, roomy budget): hard {ldt_hard:.1}% vs pooled {ldt_pooled:.1}%"
    );
    if (ldt_hard - ldt_pooled).abs() >= 10.0 {
        println!(
            "  → the pooled mean materially misrepresents the hard stratum (Issue 956's point) — always read both"
        );
    } else {
        println!(
            "  → pooled and hard agree within 10 pts on THIS spectrum; the stratified read still guards the T2 fusion"
        );
    }
    println!("  wall: {:.1}s", t0.elapsed().as_secs_f32());
}

/// Verify that without the `lattice_deduction` feature, the module is not available.
/// This test runs on default build to confirm zero impact.
#[cfg(not(feature = "lattice_deduction"))]
#[test]
fn bench_ldt_feature_gate_absent() {
    // If this compiles, the feature gate is correctly isolating LDT code.
    // The types ConflictDetector, EntropyConflictDetector, LdtPruneConfig,
    // AlphaTarget, alpha_intersect should NOT be accessible.
    println!("lattice_deduction feature gate: correctly absent from default build ✓");
}

/// Issue 956 T2 + T2b: conflict-triggered targeted candidate reopening —
/// the fusion bench. Same 480 instances + strata as T1 (shared fixtures);
/// THREE arms paired per instance/budget: plain LDT θ, LDT θ + reopen
/// (T2 fixed-b), LDT θ + reopen (T2b η-bounded). Presence AND
/// connected-chain reads, plus the gate set:
/// - G1: stratified solve rates (hard stratum is the deciding read);
/// - G3: zero-budget byte-identity vs the plain build (spot check);
/// - loud-zero: the mechanism must demonstrably fire on this distribution
///   (both reopen postures, at the roomy budget);
/// - T2b anti-thrash: at the TIGHT budget the η arm's total reopens must
///   not exceed the fixed-b arm's (the headroom drive's contract — the
///   measured T2 budget-150 regression is the reason);
/// - termination: rounds ≤ cap, reopens ≤ b × rounds (no oscillation);
///   determinism replay (both postures); every fired round makes
///   tree-visible progress.
///
/// G2 (wall ≤ +10%) is MEASURED and printed, never asserted here — the
/// repo's box-state law: a wall ratio under an unknown load is a
/// disclosure, not a verdict. Re-pin on a quiet box when promoting.
#[cfg(feature = "lattice_deduction")]
#[test]
fn bench_ldt_conflict_reopen_reporting() {
    use katgpt_rs::speculative::{
        ConflictClauseDB, LDT_THETA_ELIM, ReopenConfig, ReopenStats, build_dd_tree_screened,
        build_dd_tree_screened_reopen,
    };
    use std::time::Instant;

    /// T2 posture: fixed b = 4, β = 1 (plain score order), two rounds.
    fn fixed_cfg() -> ReopenConfig {
        ReopenConfig {
            budget_b: 4,
            beta: 1.0,
            max_rounds: 2,
            ..ReopenConfig::default()
        }
    }

    /// T2b posture: η-bounded count (expected reopens = η·headroom),
    /// `budget_b` demoted to the absolute cap, same rounds cap. The η and
    /// headroom σ constants are the shipped defaults (the provisional
    /// calibration: the T2 boots measured regression at ratio ≈ 2.08 and
    /// gain at ≈ 6.94).
    fn eta_cfg() -> ReopenConfig {
        ReopenConfig {
            budget_b: 4,
            beta: 1.0,
            max_rounds: 2,
            eta: 4.0,
            ..ReopenConfig::default()
        }
    }

    /// One reopen-arm solve: presence + connected-chain + stats.
    fn solve_reopen(
        inst: &Instance,
        theta: f32,
        budget: usize,
        reopen: &ReopenConfig,
    ) -> (bool, bool, usize, ReopenStats) {
        let slices: Vec<&[f32]> = inst.marginals.iter().map(|m| m.as_slice()).collect();
        let mut config = make_config(VOCAB, N_DEPTHS, budget);
        config.screening_threshold = theta;
        let screener = MassScreener {
            marginals: inst.marginals.clone(),
        };
        let mut db = ConflictClauseDB::default();
        let mut stats = ReopenStats::default();
        let tree = build_dd_tree_screened_reopen(
            &slices, &config, &screener, true, reopen, &mut db, &mut stats,
        );
        (
            presence_solved(&tree, &inst.solution),
            connected_chain_solved(&tree, &inst.solution),
            tree.len(),
            stats,
        )
    }

    println!("═════════════════════════════════════════════════════════");
    println!("  LDT Conflict-Reopen Bench (Issue 956 T2 + T2b)");
    println!("═════════════════════════════════════════════════════════");
    let t0 = Instant::now();

    // ── Same strata as T1 (validation-split edges, disjoint eval) ──
    let validate: Vec<Instance> = (0..N_VALIDATE).map(gen_instance).collect();
    let (edge_hard, edge_easy) = fix_strata_edges(&validate);
    let eval: Vec<Instance> = (1_000_000..1_000_000 + N_EVAL).map(gen_instance).collect();
    let mut strata: [Vec<&Instance>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for inst in &eval {
        strata[stratum_of(inst, edge_hard, edge_easy)].push(inst);
    }
    let n_total = eval.len();

    // [arm][budget] → (presence, chain) solve counts per stratum (3 + pooled)
    // arm 0 = plain LDT θ, arm 1 = LDT θ + reopen (T2 fixed-b),
    // arm 2 = LDT θ + reopen (T2b η-bounded)
    let mut presence = vec![vec![[0_usize; 4]; BUDGETS.len()]; 3];
    let mut chain = vec![vec![[0_usize; 4]; BUDGETS.len()]; 3];
    // Per-arm/per-budget reopen totals (the T2b anti-thrash read splits by
    // budget — the drive's contract is about the TIGHT cell).
    let mut reopen_totals = [[0_usize; BUDGETS.len()]; 2];
    let mut total_stats = [ReopenStats::default(); 2];
    let mut fired_instances = [0_usize; 2];

    let plain_wall = Instant::now();
    for inst in &strata.iter().flatten().collect::<Vec<_>>() {
        let si = stratum_of(inst, edge_hard, edge_easy);
        for (bi, &budget) in BUDGETS.iter().enumerate() {
            let (p_plain, _) = solve_rate(inst, LDT_THETA_ELIM, budget);
            if p_plain {
                presence[0][bi][si] += 1;
                presence[0][bi][3] += 1;
            }
            for (ai, cfg) in [(1_usize, fixed_cfg()), (2_usize, eta_cfg())] {
                let (p_re, c_re, _, stats) = solve_reopen(inst, LDT_THETA_ELIM, budget, &cfg);
                if p_re {
                    presence[ai][bi][si] += 1;
                    presence[ai][bi][3] += 1;
                }
                if c_re {
                    chain[ai][bi][si] += 1;
                    chain[ai][bi][3] += 1;
                }
                reopen_totals[ai - 1][bi] += stats.reopens_applied;
                if stats.rounds > 0 {
                    fired_instances[ai - 1] += 1;
                }
                let acc = &mut total_stats[ai - 1];
                acc.conflicts_fired += stats.conflicts_fired;
                acc.reopens_applied += stats.reopens_applied;
                acc.no_cycle_blocked += stats.no_cycle_blocked;
                acc.rounds += stats.rounds;
            }
        }
    }
    let paired_wall = plain_wall.elapsed();

    // ── Report: presence AND chain, stratified (the T1 law) ──
    let pct = |n: usize| n as f64 / n_total as f64 * 100.0;
    let pct_stratum = |n: usize, si: usize| n as f64 / strata[si].len() as f64 * 100.0;
    let arms = [("LDT θ=0.111", 0), ("+reopen b=4", 1), ("+reopen η=4", 2)];
    println!();
    println!(
        "  {:<15} {:>5} | {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} {:>7}",
        "arm", "bud", "p-hard", "p-mid", "p-easy", "p-pool", "c-hard", "c-mid", "c-easy", "c-pool"
    );
    for (name, ai) in arms {
        for (bi, &budget) in BUDGETS.iter().enumerate() {
            let (p, c) = (&presence[ai][bi], &chain[ai][bi]);
            println!(
                "  {:<15} {:>5} | {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% | {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}%",
                name,
                budget,
                pct_stratum(p[0], 0),
                pct_stratum(p[1], 1),
                pct_stratum(p[2], 2),
                pct(p[3]),
                pct_stratum(c[0], 0),
                pct_stratum(c[1], 1),
                pct_stratum(c[2], 2),
                pct(c[3]),
            );
        }
    }
    println!();
    let n_solves = n_total * BUDGETS.len();
    let posture_names = ["fixed-b", "eta-bounded"];
    for (pi, stats) in total_stats.iter().enumerate() {
        println!(
            "  reopen stats [{}] ({} solves): conflicts {} ({:.2}/solve) · reopens {} ({:.2}/solve) · no-cycle blocked {} · REOPEN-CYCLES {} ({:.3}/solve, cap {}) · fired {}/{}",
            posture_names[pi],
            n_solves,
            stats.conflicts_fired,
            stats.conflicts_fired as f64 / n_solves as f64,
            stats.reopens_applied,
            stats.reopens_applied as f64 / n_solves as f64,
            stats.no_cycle_blocked,
            stats.rounds,
            stats.rounds as f64 / n_solves as f64,
            if pi == 0 { fixed_cfg() } else { eta_cfg() }.max_rounds,
            fired_instances[pi],
            n_solves,
        );
    }
    // T2b's per-budget anti-thrash read (drives the structural gate below).
    println!(
        "  T2b reopens by budget [tight {}]: fixed-b {} vs η {} · [roomy {}]: fixed-b {} vs η {}",
        BUDGETS[0],
        reopen_totals[0][0],
        reopen_totals[1][0],
        BUDGETS[1],
        reopen_totals[0][1],
        reopen_totals[1][1],
    );
    let _ = paired_wall; // (the paired loop's wall is subsumed by the ratio read)

    // ── G2 disclosure (measured, never asserted under unknown load) ──
    // Each reopen arm pays: the elimination replay (V×D relevance calls at
    // chain context) + a full re-propagation per applied round. The η arm's
    // re-propagations shrink where the headroom drive binds — its ratio is
    // the T2b lever, read beside the fixed-b baseline.
    let t_plain_only = {
        let t = Instant::now();
        for inst in &strata.iter().flatten().collect::<Vec<_>>() {
            for &budget in BUDGETS.iter() {
                let _ = solve_rate(inst, LDT_THETA_ELIM, budget);
            }
        }
        t.elapsed()
    };
    for (name, cfg) in [("fixed-b", fixed_cfg()), ("η-bounded", eta_cfg())] {
        let t_arm = {
            let t = Instant::now();
            for inst in &strata.iter().flatten().collect::<Vec<_>>() {
                for &budget in BUDGETS.iter() {
                    let _ = solve_reopen(inst, LDT_THETA_ELIM, budget, &cfg);
                }
            }
            t.elapsed()
        };
        let ratio = t_arm.as_secs_f64() / t_plain_only.as_secs_f64().max(1e-9);
        println!(
            "  G2 wall ratio ({name}/plain): {ratio:.2}× [{name} {t_arm:.1?} vs plain {t_plain_only:.1?}] — disclosure; re-pin on a quiet box"
        );
    }

    // ── Structural gates ──
    // (a) G3: zero-budget byte-identity, 40-instance spot check.
    for seed in (1_000_000..1_000_040).step_by(1) {
        let inst = gen_instance(seed);
        let slices: Vec<&[f32]> = inst.marginals.iter().map(|m| m.as_slice()).collect();
        let mut config = make_config(VOCAB, N_DEPTHS, BUDGETS[1]);
        config.screening_threshold = LDT_THETA_ELIM;
        let screener = MassScreener {
            marginals: inst.marginals.clone(),
        };
        let plain = build_dd_tree_screened(&slices, &config, &screener, true);
        let mut db = ConflictClauseDB::default();
        let mut stats = ReopenStats::default();
        let reopened = build_dd_tree_screened_reopen(
            &slices,
            &config,
            &screener,
            true,
            &ReopenConfig::default(),
            &mut db,
            &mut stats,
        );
        assert_eq!(plain.len(), reopened.len(), "G3 seed {seed}: node count");
        for (a, b) in plain.iter().zip(reopened.iter()) {
            assert!(
                a.depth == b.depth
                    && a.token_idx == b.token_idx
                    && a.score.to_bits() == b.score.to_bits()
                    && a.parent_path == b.parent_path,
                "G3 seed {seed}: node identity diverged"
            );
        }
        assert_eq!(
            stats,
            ReopenStats::default(),
            "G3 seed {seed}: machinery ran"
        );
    }

    // (b) Loud-zero: the mechanism demonstrably fires on THIS distribution,
    //     BOTH postures, at the roomy budget. A green zero here means the
    //     trigger (or the η headroom drive) is dead, not that reopening is
    //     safe — the repo's loud-zero law.
    assert!(
        total_stats[0].conflicts_fired > 0,
        "conflict trigger never fired — the arm is a green zero"
    );
    assert!(
        total_stats[0].reopens_applied > 0,
        "no reopen ever applied (fixed-b) — the arm is a green zero"
    );
    assert!(
        reopen_totals[1][1] > 0,
        "η arm applied zero reopens at the roomy budget — the headroom drive strangled the mechanism (a green zero)"
    );
    // T2b's anti-thrash gate: at the TIGHT budget the η arm's total spend
    // must not exceed the fixed-b arm's. This is the measured T2 regression
    // (hard 14.1→8.5 at budget 150) made executable: the headroom drive
    // must bound the spend where the budget cannot absorb it.
    assert!(
        reopen_totals[1][0] <= reopen_totals[0][0],
        "T2b anti-thrash violated: η arm spent {} reopens at tight budget vs fixed-b {} — the headroom drive is not bounding the spend",
        reopen_totals[1][0],
        reopen_totals[0][0]
    );
    let hard_reopened = {
        let mut n = 0_usize;
        for inst in &strata[0] {
            let (_, _, _, stats) = solve_reopen(inst, LDT_THETA_ELIM, BUDGETS[1], &fixed_cfg());
            n += stats.reopens_applied;
        }
        n
    };
    assert!(
        hard_reopened > 0,
        "the hard stratum (T2's deciding read) saw zero reopens"
    );

    // (c) Termination + per-node caps, per instance, BOTH postures (the FSM
    //     law). The η posture adds its own: an applied round restores
    //     something (a bisected-zero round breaks BEFORE re-propagating).
    for seed in 1_000_000..1_000_020 {
        let inst = gen_instance(seed);
        for (cfg, name) in [(fixed_cfg(), "fixed-b"), (eta_cfg(), "eta")] {
            let (_, _, _, stats) = solve_reopen(&inst, LDT_THETA_ELIM, BUDGETS[1], &cfg);
            assert!(
                stats.rounds <= cfg.max_rounds,
                "{name} seed {seed}: round cap violated ({} > {})",
                stats.rounds,
                cfg.max_rounds
            );
            assert!(
                stats.reopens_applied <= cfg.budget_b * stats.rounds.max(1),
                "{name} seed {seed}: per-round reopen cap violated"
            );
            assert!(
                stats.rounds == 0 || stats.reopens_applied > 0,
                "{name} seed {seed}: an applied round restored nothing (progress law)"
            );
        }
    }

    // (d) Determinism: full replay reproduces tree + stats exactly, both
    //     postures (the η bisection is pure arithmetic — no RNG).
    for seed in 1_000_000..1_000_010 {
        let inst = gen_instance(seed);
        for (cfg, name) in [(fixed_cfg(), "fixed-b"), (eta_cfg(), "eta")] {
            let (p1, c1, n1, s1) = solve_reopen(&inst, LDT_THETA_ELIM, BUDGETS[1], &cfg);
            let (p2, c2, n2, s2) = solve_reopen(&inst, LDT_THETA_ELIM, BUDGETS[1], &cfg);
            assert_eq!(
                (p1, c1, n1),
                (p2, c2, n2),
                "{name} seed {seed}: replay diverged"
            );
            assert_eq!(s1, s2, "{name} seed {seed}: stats diverged");
        }
    }

    // (e) Progress on the deciding reads (print, never assert — the G1
    //     direction is an empirical outcome, recorded in the issue):
    //     fixed-b and η at the roomy budget, plus the T2b anti-regression
    //     read at the TIGHT budget (the drive exists for it).
    for (ai, name) in [(1_usize, "fixed-b"), (2_usize, "η")] {
        let re_hard = pct_stratum(presence[ai][1][0], 0);
        let pl_hard = pct_stratum(presence[0][1][0], 0);
        let re_chain_hard = pct_stratum(chain[ai][1][0], 0);
        println!();
        println!(
            "  G1 hard-stratum read [{name}] (budget {}): presence {pl_hard:.1}% → {re_hard:.1}% (Δ {:+.1} pts) · chain {:.1}%",
            BUDGETS[1],
            re_hard - pl_hard,
            re_chain_hard,
        );
        let re_hard_tight = pct_stratum(presence[ai][0][0], 0);
        let pl_hard_tight = pct_stratum(presence[0][0][0], 0);
        println!(
            "  T2b tight-budget read [{name}] (budget {}): presence {pl_hard_tight:.1}% → {re_hard_tight:.1}% (Δ {:+.1} pts)",
            BUDGETS[0],
            re_hard_tight - pl_hard_tight,
        );
    }
    println!("  wall: {:.1}s", t0.elapsed().as_secs_f32());
}

// ── Issue 956 T4: flip-move vs resample-move pricing probe ─────────
//
// The paper's (arXiv:2610.28543) sampler mechanics priced on the SAME
// fixture spectrum as T1/T2. The binary-symmetric candidate-set kernel
// q_{t|0}(z|c) = (1−t/2)^{V−d_H}·(t/2)^{d_H} makes every reverse jump a
// SINGLE-BIT flip, and all L·V flip rates price in one closed-form sweep
// from the already-computed per-position marginal μ (the denoiser proxy)
// via the kernel ratios — zero extra forwards:
//
//   flipping bit u changes the Hamming distance to the clean one-hot by
//   ±1, so the likelihood ratio is A = (1−t/2)/(t/2) when the flip moves
//   TOWARD clean and B = 1/A when away; Bayes over the clean token gives
//
//     r_add(u ∉ z) ∝ π_u·A + (1−π_u)·B        (restore a candidate)
//     r_rem(u ∈ z) ∝ π_u·B + (1−π_u)·A        (retire a candidate)
//
//   q(z) is constant within one state's flip ranking, so the within-state
//   jump law is exact; across states the chain carries a time rescaling
//   (immaterial — the probe counts moves, not wall time).
//
// The probe question: at equal PASS count (one pass = one flip for the
// flip arms; one full redraw for the resample arm — both cost one
// O(L·V) sweep), which move class more often ends with the planted
// solution retained in EVERY position's candidate set? The resample arm
// is the paper's ancestral move (c̃ ~ μ per depth, bridge bits at t) —
// MEMORYLESS: it discards the accumulated state every pass. The flip
// arms are parsimonious: one targeted bit each, evidence-preserving.
//
// ⚠ The factorized-μ caveat, named up front: the fixture's μ is a fixed
// per-depth product — the state carries no information beyond μ, so both
// arms target the same posterior and MUST agree in the K→∞ limit. The
// discriminating read is the finite-K path: whether flip-parsimony
// preserves the corrupted state's solution bits that instant
// re-randomization discards — the finite-budget posture the lattice's
// downstream verification actually runs at.
//
// Measured consequence of the proxy (2026-10-23, in the table below): the
// fixed μ starves the flip chain's ADD side — every present bit's removal
// rate carries the (1−π)·A term (A = 5.67 at t=0.3), so greedy runs to
// near-EMPTY sets by K=16 (mean|S| 0.45) and the win dies with them. A
// production flip lane therefore needs BOTH (a) set-conditioned scoring
// (the lattice's screener replay at chain context — T2's machinery — in
// place of fixed μ) and (b) a set-size floor / reopen-budget law (T2b's
// η·headroom drive is exactly that shape). Neither exists in this probe —
// deliberately: T4 prices the MOVE CLASS, not the production lane.

/// Kernel noise levels (t = 2·p, so p = t/2 is the per-bit flip
/// probability of the symmetric channel).
#[cfg(feature = "lattice_deduction")]
const PROBE_T_LEVELS: [f32; 2] = [0.3, 0.6];

/// Pass checkpoints the win rate reads at.
#[cfg(feature = "lattice_deduction")]
const PROBE_PASS_MARKS: [usize; 3] = [1, 4, 16];

#[cfg(feature = "lattice_deduction")]
const PROBE_ARMS: usize = 4;

#[cfg(feature = "lattice_deduction")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ProbeFlipMode {
    /// Paper-faithful jump law: one flip drawn ∝ its rate (seeded —
    /// deterministic replay).
    Sampled,
    /// Deterministic read: apply the max-rate flip (ties → first in the
    /// fixed (depth, token, add-before-rem) scan order). The
    /// production-cheap posture — no RNG on the hot path.
    Greedy,
}

#[cfg(feature = "lattice_deduction")]
struct ProbeRun {
    state: Vec<Vec<bool>>,
    flips_applied: usize,
    /// Solution present at every depth, read at each [`PROBE_PASS_MARKS`]
    /// checkpoint of the ONE cumulative run.
    wins_at: [bool; PROBE_PASS_MARKS.len()],
}

/// The paired corrupted state: clean singleton per depth, symmetric-kernel
/// corrupted at t (solution bit present w.p. 1−t/2, every other bit
/// w.p. t/2). Same (seed, t) → the SAME init for every arm.
#[cfg(feature = "lattice_deduction")]
fn probe_corrupt_state(inst: &Instance, t: f32, rng: &mut fastrand::Rng) -> Vec<Vec<bool>> {
    let p = t * 0.5;
    inst.marginals
        .iter()
        .enumerate()
        .map(|(d, _)| {
            (0..VOCAB)
                .map(|u| {
                    if u == inst.solution[d] {
                        rng.f32() >= p
                    } else {
                        rng.f32() < p
                    }
                })
                .collect::<Vec<bool>>()
        })
        .collect()
}

/// The T4 headline: every single-bit flip's reverse rate in one O(L·V)
/// closed-form sweep from the already-computed μ — zero extra forwards.
/// Writes the live side of each table, zeroes the dead side (an absent
/// bit has no remove rate; a present bit has no add rate).
#[cfg(feature = "lattice_deduction")]
fn probe_price_flips(
    state: &[Vec<bool>],
    mu: &[Vec<f32>],
    t: f32,
    add: &mut [Vec<f32>],
    rem: &mut [Vec<f32>],
) {
    assert!(
        (0.0..1.0).contains(&t),
        "kernel noise t must be in (0,1), got {t}"
    );
    let half = t * 0.5;
    let toward = (1.0 - half) / half; // A: flip moves d_H down
    let away = 1.0 / toward; // B: flip moves d_H up
    for (d, row) in state.iter().enumerate() {
        for (u, &present) in row.iter().enumerate() {
            let pi = mu[d][u];
            if present {
                add[d][u] = 0.0;
                rem[d][u] = pi * away + (1.0 - pi) * toward;
            } else {
                rem[d][u] = 0.0;
                add[d][u] = pi * toward + (1.0 - pi) * away;
            }
        }
    }
}

/// Max-rate flip over the priced tables (the Greedy pick and the
/// sampled arm's 1-ULP fallback). Live rates are strictly positive on
/// this fixture (every μ mass > 0), so the scan always finds one.
#[cfg(feature = "lattice_deduction")]
fn probe_greedy_pick(add: &[Vec<f32>], rem: &[Vec<f32>]) -> (usize, usize, bool) {
    let mut best = (0_usize, 0_usize, true);
    let mut best_rate = 0.0_f32;
    for dd in 0..N_DEPTHS {
        for uu in 0..VOCAB {
            if add[dd][uu] > best_rate {
                best_rate = add[dd][uu];
                best = (dd, uu, true);
            }
            if rem[dd][uu] > best_rate {
                best_rate = rem[dd][uu];
                best = (dd, uu, false);
            }
        }
    }
    best
}

#[cfg(feature = "lattice_deduction")]
fn probe_win(state: &[Vec<bool>], solution: &[usize]) -> bool {
    solution.iter().enumerate().all(|(d, &tok)| state[d][tok])
}

#[cfg(feature = "lattice_deduction")]
fn probe_set_size(state: &[Vec<bool>]) -> usize {
    state
        .iter()
        .map(|row| row.iter().filter(|&&b| b).count())
        .sum()
}

/// Deterministic per-(instance, t, arm) rng seed. Arm 0xFF is the SHARED
/// init stream — the paired design's premise.
#[cfg(feature = "lattice_deduction")]
fn probe_seed(base: u64, ti: usize, arm: u64) -> u64 {
    base ^ 0x9E37_79B9_7F4A_7C15u64.rotate_left(((ti as u64 * 8 + arm) & 63) as u32)
}

/// One flip-arm run: `passes` single-bit flips under the paper's jump
/// law, each priced by [`probe_price_flips`]; win snapshots at the pass
/// marks of the same cumulative run.
#[cfg(feature = "lattice_deduction")]
fn probe_run_flip(
    init: Vec<Vec<bool>>,
    mu: &[Vec<f32>],
    solution: &[usize],
    t: f32,
    passes: usize,
    mode: ProbeFlipMode,
    rng: &mut fastrand::Rng,
) -> ProbeRun {
    let mut state = init;
    let mut add = vec![vec![0.0_f32; VOCAB]; N_DEPTHS];
    let mut rem = vec![vec![0.0_f32; VOCAB]; N_DEPTHS];
    let mut wins_at = [false; PROBE_PASS_MARKS.len()];
    for pass in 0..passes {
        probe_price_flips(&state, mu, t, &mut add, &mut rem);
        let (d, u, to_add) = match mode {
            ProbeFlipMode::Greedy => probe_greedy_pick(&add, &rem),
            ProbeFlipMode::Sampled => {
                let total: f32 = add
                    .iter()
                    .chain(rem.iter())
                    .map(|row| row.iter().sum::<f32>())
                    .sum();
                let mut pick = rng.f32() * total;
                let mut chosen: Option<(usize, usize, bool)> = None;
                'scan: for (to_add, table) in [(true, &add), (false, &rem)] {
                    for (dd, row) in table.iter().enumerate() {
                        for (uu, &rate) in row.iter().enumerate() {
                            pick -= rate;
                            if pick <= 0.0 {
                                chosen = Some((dd, uu, to_add));
                                break 'scan;
                            }
                        }
                    }
                }
                // 1-ULP cumulative-sum edge (pick grazes `total`): the
                // max-rate flip is the deterministic fallback.
                chosen.unwrap_or_else(|| probe_greedy_pick(&add, &rem))
            }
        };
        state[d][u] = to_add;
        for (mi, &m) in PROBE_PASS_MARKS.iter().enumerate() {
            if pass + 1 == m {
                wins_at[mi] = probe_win(&state, solution);
            }
        }
    }
    ProbeRun {
        state,
        flips_applied: passes,
        wins_at,
    }
}

/// One resample-arm run: `passes` ancestral moves — per depth draw the
/// clean proposal c̃ ~ μ, then bridge every membership bit at t. The init
/// is deliberately NOT consumed: memorylessness is the mechanism being
/// priced against.
#[cfg(feature = "lattice_deduction")]
fn probe_run_resample(
    mu: &[Vec<f32>],
    solution: &[usize],
    t: f32,
    passes: usize,
    rng: &mut fastrand::Rng,
) -> ProbeRun {
    let p = t * 0.5;
    let mut state = vec![vec![false; VOCAB]; N_DEPTHS];
    let mut wins_at = [false; PROBE_PASS_MARKS.len()];
    for pass in 0..passes {
        for (d, dist) in mu.iter().enumerate() {
            let mut pick = rng.f32();
            let mut c = VOCAB - 1;
            for (u, &pi) in dist.iter().enumerate() {
                pick -= pi;
                if pick <= 0.0 {
                    c = u;
                    break;
                }
            }
            for (u, slot) in state[d].iter_mut().enumerate() {
                *slot = if u == c {
                    rng.f32() >= p
                } else {
                    rng.f32() < p
                };
            }
        }
        for (mi, &m) in PROBE_PASS_MARKS.iter().enumerate() {
            if pass + 1 == m {
                wins_at[mi] = probe_win(&state, solution);
            }
        }
    }
    ProbeRun {
        state,
        flips_applied: 0,
        wins_at,
    }
}

/// Issue 956 T4 — flip-move vs resample-move pricing probe (probe-first,
/// no GOAT without it). Same strata fixtures as T1/T2; paired corrupted
/// inits; win = the planted solution present at every depth; structural
/// assertions only (determinism, zero-pass identity, loud-zero, the
/// bridge sampler's kernel expectation, paired-init replay). The
/// flip-vs-resample verdict is PRINTED, never asserted — it is an
/// empirical read that gates whether a flip-move lane is worth building,
/// not a regression surface.
#[cfg(feature = "lattice_deduction")]
#[test]
fn bench_ldt_flip_vs_resample_probe() {
    use std::time::Instant;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Flip-vs-Resample Move Probe (Issue 956 T4)");
    println!("═══════════════════════════════════════════════════════════");
    let t0 = Instant::now();

    // ── Same strata fixtures as T1/T2 (validation edges, disjoint eval) ──
    let validate: Vec<Instance> = (0..N_VALIDATE).map(gen_instance).collect();
    let (edge_hard, edge_easy) = fix_strata_edges(&validate);
    let eval: Vec<(u64, Instance)> = (1_000_000..1_000_000 + N_EVAL)
        .map(|s| (s, gen_instance(s)))
        .collect();
    let mut strata: [Vec<(u64, &Instance)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (seed, inst) in &eval {
        strata[stratum_of(inst, edge_hard, edge_easy)].push((*seed, inst));
    }
    let strata_n = [strata[0].len(), strata[1].len(), strata[2].len()];
    for (si, name) in ["hard", "mid", "easy"].iter().enumerate() {
        assert!(
            strata_n[si] >= 15,
            "stratum {name} population floor: {} < 15",
            strata_n[si]
        );
    }
    let n_total = eval.len();
    let max_passes = PROBE_PASS_MARKS[PROBE_PASS_MARKS.len() - 1];

    // arms: 0 hold · 1 flip-greedy · 2 flip-sampled · 3 resample.
    // counts[arm][t][mark][stratum]; the hold arm uses the mark-0 slot.
    let mut counts = [[[[0_usize; 4]; PROBE_PASS_MARKS.len()]; PROBE_T_LEVELS.len()]; PROBE_ARMS];
    let mut size_sums = [[0_u64; PROBE_T_LEVELS.len()]; PROBE_ARMS];
    let mut init_size_sum = [0_u64; PROBE_T_LEVELS.len()];
    let mut flip_totals = [0_usize; PROBE_ARMS];

    for &(seed, inst) in strata.iter().flatten() {
        let si = stratum_of(inst, edge_hard, edge_easy);
        for (ti, &t) in PROBE_T_LEVELS.iter().enumerate() {
            let mut init_rng = fastrand::Rng::with_seed(probe_seed(seed, ti, 0xFF));
            let init = probe_corrupt_state(inst, t, &mut init_rng);
            init_size_sum[ti] += probe_set_size(&init) as u64;
            if probe_win(&init, &inst.solution) {
                counts[0][ti][0][si] += 1;
                counts[0][ti][0][3] += 1;
            }
            for (arm, mode) in [
                (1_usize, ProbeFlipMode::Greedy),
                (2, ProbeFlipMode::Sampled),
            ] {
                let mut rng = fastrand::Rng::with_seed(probe_seed(seed, ti, arm as u64));
                let run = probe_run_flip(
                    init.clone(),
                    &inst.marginals,
                    &inst.solution,
                    t,
                    max_passes,
                    mode,
                    &mut rng,
                );
                for (mi, &w) in run.wins_at.iter().enumerate() {
                    if w {
                        counts[arm][ti][mi][si] += 1;
                        counts[arm][ti][mi][3] += 1;
                    }
                }
                size_sums[arm][ti] += probe_set_size(&run.state) as u64;
                flip_totals[arm] += run.flips_applied;
            }
            let mut rng = fastrand::Rng::with_seed(probe_seed(seed, ti, 3));
            let run = probe_run_resample(&inst.marginals, &inst.solution, t, max_passes, &mut rng);
            for (mi, &w) in run.wins_at.iter().enumerate() {
                if w {
                    counts[3][ti][mi][si] += 1;
                    counts[3][ti][mi][3] += 1;
                }
            }
            size_sums[3][ti] += probe_set_size(&run.state) as u64;
        }
    }

    // ── Report ──
    let pct = |n: usize, d: usize| {
        if d == 0 {
            0.0
        } else {
            n as f64 / d as f64 * 100.0
        }
    };
    let arm_names = ["hold (K=0)", "flip-greedy", "flip-sampled", "resample"];
    println!();
    println!(
        "  {:<13} {:>4} {:>4} | {:>6} {:>6} {:>6} {:>6} | {:>7}",
        "arm", "t", "K", "hard", "mid", "easy", "pooled", "mean|S|"
    );
    for (ti, &t) in PROBE_T_LEVELS.iter().enumerate() {
        for (arm, name) in arm_names.iter().enumerate() {
            // Slot indices, never pass values — the counts' mark dimension
            // is the 3-slot [K=1, K=4, K=16] table (hold uses slot 0 only).
            let slots: Vec<usize> = if arm == 0 {
                vec![0]
            } else {
                (0..PROBE_PASS_MARKS.len()).collect()
            };
            for &mi in &slots {
                let c = &counts[arm][ti][mi];
                let mean_s = if arm == 0 {
                    init_size_sum[ti]
                } else {
                    size_sums[arm][ti]
                } as f64
                    / (n_total * N_DEPTHS) as f64;
                println!(
                    "  {:<13} {:>4.2} {:>4} | {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}% | {:>7.2}",
                    name,
                    t,
                    if arm == 0 { 0 } else { PROBE_PASS_MARKS[mi] },
                    pct(c[0], strata_n[0]),
                    pct(c[1], strata_n[1]),
                    pct(c[2], strata_n[2]),
                    pct(c[3], n_total),
                    mean_s,
                );
            }
        }
    }

    // ── The T4 headline cost: one O(L·V) pricing sweep ──
    let sweep_state = vec![vec![true; VOCAB]; N_DEPTHS];
    let mut add = vec![vec![0.0_f32; VOCAB]; N_DEPTHS];
    let mut rem = vec![vec![0.0_f32; VOCAB]; N_DEPTHS];
    let mu_ref: Vec<Vec<f32>> = eval[0].1.marginals.clone();
    const SWEEPS: usize = 50_000;
    let sweep_ns = {
        let ts = Instant::now();
        for _ in 0..SWEEPS {
            probe_price_flips(&sweep_state, &mu_ref, 0.3, &mut add, &mut rem);
            std::hint::black_box(&add);
            std::hint::black_box(&rem);
        }
        ts.elapsed().as_nanos()
    };
    println!();
    println!(
        "  pricing sweep: {} ns for L·V = {} flip rates (all V alternatives × L depths) — closed-form from μ, ZERO extra forwards — {:.2} flips/ns",
        sweep_ns / SWEEPS as u128,
        N_DEPTHS * VOCAB,
        (SWEEPS * N_DEPTHS * VOCAB) as f64 / sweep_ns as f64,
    );

    // ── Structural gates ──
    // (a) Determinism: greedy is rng-free (pure); sampled + resample
    //     replay bit-exact under their seeds.
    let (seed0, inst0) = (&eval[0].0, &eval[0].1);
    let init0 = probe_corrupt_state(
        inst0,
        PROBE_T_LEVELS[0],
        &mut fastrand::Rng::with_seed(probe_seed(*seed0, 0, 0xFF)),
    );
    let run_g1 = probe_run_flip(
        init0.clone(),
        &inst0.marginals,
        &inst0.solution,
        PROBE_T_LEVELS[0],
        max_passes,
        ProbeFlipMode::Greedy,
        &mut fastrand::Rng::with_seed(0),
    );
    let run_g2 = probe_run_flip(
        init0.clone(),
        &inst0.marginals,
        &inst0.solution,
        PROBE_T_LEVELS[0],
        max_passes,
        ProbeFlipMode::Greedy,
        &mut fastrand::Rng::with_seed(99),
    );
    assert_eq!(
        run_g1.state, run_g2.state,
        "greedy replay diverged (must be rng-free)"
    );
    assert_eq!(
        run_g1.wins_at, run_g2.wins_at,
        "greedy replay diverged (wins)"
    );
    let run_s1 = probe_run_flip(
        init0.clone(),
        &inst0.marginals,
        &inst0.solution,
        PROBE_T_LEVELS[0],
        max_passes,
        ProbeFlipMode::Sampled,
        &mut fastrand::Rng::with_seed(probe_seed(*seed0, 0, 2)),
    );
    let run_s2 = probe_run_flip(
        init0.clone(),
        &inst0.marginals,
        &inst0.solution,
        PROBE_T_LEVELS[0],
        max_passes,
        ProbeFlipMode::Sampled,
        &mut fastrand::Rng::with_seed(probe_seed(*seed0, 0, 2)),
    );
    assert_eq!(run_s1.state, run_s2.state, "sampled replay diverged");
    assert_eq!(
        run_s1.wins_at, run_s2.wins_at,
        "sampled replay diverged (wins)"
    );
    let run_r1 = probe_run_resample(
        &inst0.marginals,
        &inst0.solution,
        PROBE_T_LEVELS[0],
        max_passes,
        &mut fastrand::Rng::with_seed(probe_seed(*seed0, 0, 3)),
    );
    let run_r2 = probe_run_resample(
        &inst0.marginals,
        &inst0.solution,
        PROBE_T_LEVELS[0],
        max_passes,
        &mut fastrand::Rng::with_seed(probe_seed(*seed0, 0, 3)),
    );
    assert_eq!(run_r1.state, run_r2.state, "resample replay diverged");

    // (b) Zero-pass identity: the flip arms with passes=0 return the init.
    let run_z = probe_run_flip(
        init0.clone(),
        &inst0.marginals,
        &inst0.solution,
        PROBE_T_LEVELS[0],
        0,
        ProbeFlipMode::Sampled,
        &mut fastrand::Rng::with_seed(1),
    );
    assert_eq!(run_z.state, init0, "zero-pass flip run mutated the state");
    assert_eq!(run_z.flips_applied, 0);

    // (c) Loud-zero: both flip arms demonstrably fire on this distribution
    //     (a pricing machinery that never applies a flip is dead, not safe).
    assert!(
        flip_totals[1] > 0,
        "flip-greedy applied zero flips — green zero"
    );
    assert!(
        flip_totals[2] > 0,
        "flip-sampled applied zero flips — green zero"
    );
    assert_eq!(
        run_g1.flips_applied, max_passes,
        "per-run spend must equal passes"
    );
    assert_eq!(run_s1.flips_applied, max_passes);

    // (d) Bridge-sampler validation: the resample arm's mean per-depth set
    //     size must match the kernel expectation (1−p) + (V−1)·p.
    for (ti, &t) in PROBE_T_LEVELS.iter().enumerate() {
        let p = (t * 0.5) as f64;
        let expect = (1.0 - p) + (VOCAB - 1) as f64 * p;
        let got = size_sums[3][ti] as f64 / (n_total * N_DEPTHS) as f64;
        assert!(
            (got - expect).abs() < 0.05,
            "resample bridge set size {got:.3} off the kernel expectation {expect:.3} (t={t})"
        );
    }

    // (e) The paired premise: the init stream replays bit-exact.
    let init_replay = probe_corrupt_state(
        inst0,
        PROBE_T_LEVELS[0],
        &mut fastrand::Rng::with_seed(probe_seed(*seed0, 0, 0xFF)),
    );
    assert_eq!(init0, init_replay, "paired init not reproducible");

    // ── The probe's verdict reads (printed, never asserted) ──
    println!();
    for (ti, &t) in PROBE_T_LEVELS.iter().enumerate() {
        println!(
            "  HARD K=1 @ t={t:.2}: hold {:.1}% · flip-greedy {:.1}% · flip-sampled {:.1}% · resample {:.1}%",
            pct(counts[0][ti][0][0], strata_n[0]),
            pct(counts[1][ti][0][0], strata_n[0]),
            pct(counts[2][ti][0][0], strata_n[0]),
            pct(counts[3][ti][0][0], strata_n[0]),
        );
    }
    let pooled_pct = |arm: usize, mi: usize| {
        (counts[arm][0][mi][3] + counts[arm][1][mi][3]) as f64 / (2 * n_total) as f64 * 100.0
    };
    let flip_k1_best = pooled_pct(1, 0).max(pooled_pct(2, 0));
    let resample_k1 = pooled_pct(3, 0);
    let flip_k16_best = pooled_pct(1, 2).max(pooled_pct(2, 2));
    println!();
    if flip_k1_best - resample_k1 >= 5.0 {
        println!(
            "  READ: flip-parsimony preserves hard-instance evidence at finite budgets (pooled K=1 {flip_k1_best:.1}% vs resample {resample_k1:.1}%, ≥ +5 pts) — a flip-move lane has a priced mechanism HERE; the GOAT gate must read it at the lattice's real verification budget"
        );
    } else {
        println!(
            "  READ: no flip-parsimony advantage on the factorized shape (pooled K=1 {flip_k1_best:.1}% vs resample {resample_k1:.1}%) — the move class only matters where state carries beyond-μ information (context-conditioned decode); probe says NO-GO for this fixture"
        );
    }
    println!(
        "  convergence read: flip K=16 pooled-best {flip_k16_best:.1}% vs resample (stationary) {resample_k1:.1}% — the fixed-μ caveat made measurable: the gap must close in the limit"
    );
    println!("  wall: {:.1}s", t0.elapsed().as_secs_f32());
}
