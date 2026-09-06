#![cfg(feature = "eqr_convergence")]
//! Issue 732 — Fresh-z₀ breadth-restart arm + the D-first law for
//! best_of_k_rollouts (EqR randomized-initialization axis, Research 079 §10).
//!
//! # Pre-registered design (committed BEFORE the run — the Issue-073-T3 order)
//!
//! ## The gap (issue text, verbatim input)
//!
//! Breadth today = per-rollout SDE noise (γ) around the SAME base marginals —
//! low-variance perturbation inside one basin. EqR's RI axis: independent
//! restarts from FRESH random z₀ (Gaussian, large σ) probing DIFFERENT
//! basins. On shaped landscapes `Top1Converged` beats majority vote; on
//! unshaped ones it can lose (convergence certifies basin membership, not
//! correctness).
//!
//! ## Arms (the full restart × selection grid, matched NFE)
//!
//! RestartMode ∈ {Perturb (γ = 1.0), FreshZ0 (σ = 4.0)} × WidthSelectionMode
//! ∈ {MostFrequent, BestQ, Top1Converged} — six arms, K ∈ {1, 4, 8, 16, 32},
//! 20 trials (seeds 42..61, the bench_119 convention). NFE = D·K per call;
//! all arms share the K grid at each D, so comparisons are NFE-matched by
//! construction. Config::draft() fixture (seed-42 weights, dflash marginals,
//! greedy baseline) — the same unshaped synthetic fixture as bench_119.
//!
//! ## Pre-registered invariants (hard asserts)
//!
//! 1. **K = 1 anchor:** every arm takes the single-tree early return — all
//!    six arms must return the SAME path per trial, identical to the greedy
//!    baseline fixture path.
//! 2. **Replay determinism:** same seed twice → identical path (every arm at
//!    K = 16).
//! 3. **Mechanism check:** FreshZ0 must diversify — the cross-trial unique
//!    returned-path count for FreshZ0 ≥ Perturb's at K = 16 (σ = 4.0 distorts
//!    rankings more than γ = 1.0; if FreshZ0 cannot even match Perturb's
//!    diversity, the axis is not exercising).
//!
//! ## Pre-registered expectations (MEASURED, not asserted — the T2/731
//! precedent; the verdicts are printed and recorded in the issue)
//!
//! - **Negative control (T3):** on this UNSHAPED fixture, Perturb+Top1Converged
//!   must NOT beat Perturb+MostFrequent on quality or top-1 agreement. A win
//!   indicts the fixture, not the method — the control guards the main
//!   comparison's interpretation.
//! - **Main comparison (T2):** FreshZ0+Top1Converged vs Perturb+MostFrequent
//!   at matched K — Δ quality and Δ agreement recorded per K. EqR predicts
//!   fresh restarts + convergence selection win ABOVE a depth knee; below it,
//!   breadth is useless.
//! - **D-first law (T4):** D ∈ {2, 4, 8} — draft_lookahead is capped by
//!   block_size = 16 and the greedy baseline is meaningless at D = 1, so the
//!   issue's {1, 2, 4, 8, 16, 64} grid is measured as {2, 4, 8} on this
//!   fixture (DEVIATION recorded: D = 16 collapses to 8 effective depths via
//!   `min(draft_lookahead, block_size − pos)`; D = 64 is unreachable on
//!   Config::draft()). For each (D, arm): the smallest K whose mean agreement
//!   beats K = 1 by > 1e-4 — the breadth-pays threshold; the law sentence is
//!   read off the table after the run.
//!
//! # Run
//!
//! ```bash
//! cargo test --features eqr_convergence \
//!   --test bench_732_fresh_z0_restart -- --nocapture
//! ```

use katgpt_core::{Config, Rng};
use katgpt_rs::speculative::NoScreeningPruner;
use katgpt_rs::speculative::dd_tree::{
    RestartMode, WidthScaleConfig, WidthSelectionMode, best_of_k_rollouts, build_dd_tree_screened,
    extract_best_path,
};
use katgpt_rs::speculative::dflash::dflash_predict;
use katgpt_rs::speculative::types::SdeConfig;
use katgpt_rs::transformer::TransformerWeights;

const N_TRIALS: usize = 20;
const BASE_SEED: u64 = 42;

/// The six arms, in print order.
const ARMS: [(&str, RestartMode, WidthSelectionMode); 6] = [
    ("Perturb+MostFreq", RestartMode::Perturb, WidthSelectionMode::MostFrequent),
    ("Perturb+BestQ", RestartMode::Perturb, WidthSelectionMode::BestQ),
    ("Perturb+Top1Conv", RestartMode::Perturb, WidthSelectionMode::Top1Converged),
    ("FreshZ0+MostFreq", RestartMode::FreshZ0, WidthSelectionMode::MostFrequent),
    ("FreshZ0+BestQ", RestartMode::FreshZ0, WidthSelectionMode::BestQ),
    ("FreshZ0+Top1Conv", RestartMode::FreshZ0, WidthSelectionMode::Top1Converged),
];

/// Quality = mean base (unnoised) top-1 probability along the path
/// (the bench_119 metric).
fn path_quality(marginals: &[Vec<f32>], path: &[usize]) -> f32 {
    if path.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f32;
    for (depth, &token) in path.iter().enumerate() {
        if depth < marginals.len() {
            total += marginals[depth].get(token).copied().unwrap_or(0.0);
        }
    }
    total / path.len() as f32
}

/// Top-1 agreement with the greedy baseline (the bench_119 metric).
fn top1_agreement(greedy: &[usize], path: &[usize]) -> f32 {
    if path.is_empty() || greedy.is_empty() {
        return 0.0;
    }
    let min_len = path.len().min(greedy.len());
    let matches = (0..min_len).filter(|&i| path[i] == greedy[i]).count();
    matches as f32 / min_len as f32
}

fn sde_config() -> SdeConfig {
    SdeConfig {
        gamma: 1.0,
        ..Default::default()
    }
}

#[test]
fn bench_732_fresh_z0_restart() {
    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let marginals = dflash_predict(&weights, &config, 0, 0);
    let marginals_refs: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let greedy = {
        let tree = build_dd_tree_screened(&marginals_refs, &config, &NoScreeningPruner, false);
        extract_best_path(&tree)
    };
    let d_eff = greedy.len();
    println!("\nfixture: Config::draft(), D (greedy path length) = {d_eff}, vocab = {}", config.vocab_size);

    let k_values = [1usize, 4, 8, 16, 32];

    // ── Invariant 1 — the K = 1 anchor: every arm returns the greedy path ──
    for (name, restart, selection) in ARMS {
        for trial in 0..N_TRIALS {
            let path = best_of_k_rollouts(
                &marginals_refs,
                &config,
                &NoScreeningPruner,
                &sde_config(),
                &WidthScaleConfig {
                    k_rollouts: 1,
                    selection,
                    restart_mode: restart,
                },
                BASE_SEED + trial as u64,
            );
            assert_eq!(
                path, greedy,
                "K = 1 anchor violated: arm {name} trial {trial} diverged from the single-tree path"
            );
        }
    }
    println!("[Invariant 1] K = 1 anchor: all six arms ≡ single-tree path on {N_TRIALS} trials ✓");

    // ── Invariant 2 — replay determinism at K = 16 ─────────────────────
    for (name, restart, selection) in ARMS {
        let wc = || WidthScaleConfig {
            k_rollouts: 16,
            selection,
            restart_mode: restart,
        };
        let a = best_of_k_rollouts(
            &marginals_refs, &config, &NoScreeningPruner, &sde_config(), &wc(), 777,
        );
        let b = best_of_k_rollouts(
            &marginals_refs, &config, &NoScreeningPruner, &sde_config(), &wc(), 777,
        );
        assert_eq!(a, b, "replay determinism violated: arm {name}");
    }
    println!("[Invariant 2] replay determinism at K = 16: all arms ✓");

    // ── Main grid (T2): quality / agreement / diversity per (arm, K) ────
    // diversity = unique returned paths across trials / trials (bench_119).
    println!("\n[T2 grid] γ = 1.0 (Perturb) vs σ = 4.0 (FreshZ0); D = {d_eff}; {N_TRIALS} trials");
    println!("| arm | K | quality | top1-agree | diversity |");
    println!("|---|---|---|---|---|");
    let mut grid: Vec<(&str, usize, f32, f32)> = Vec::new();
    for &k in &k_values {
        for (name, restart, selection) in ARMS {
            let mut qualities = Vec::with_capacity(N_TRIALS);
            let mut agreements = Vec::with_capacity(N_TRIALS);
            let mut paths = Vec::with_capacity(N_TRIALS);
            for trial in 0..N_TRIALS {
                let path = best_of_k_rollouts(
                    &marginals_refs,
                    &config,
                    &NoScreeningPruner,
                    &sde_config(),
                    &WidthScaleConfig {
                        k_rollouts: k,
                        selection,
                        restart_mode: restart,
                    },
                    BASE_SEED + trial as u64,
                );
                qualities.push(path_quality(&marginals, &path));
                agreements.push(top1_agreement(&greedy, &path));
                paths.push(path);
            }
            let q = qualities.iter().sum::<f32>() / N_TRIALS as f32;
            let a = agreements.iter().sum::<f32>() / N_TRIALS as f32;
            let diversity = paths.iter().collect::<std::collections::HashSet<_>>().len() as f32
                / N_TRIALS as f32;
            println!("| {name} | {k} | {q:.4} | {a:.4} | {diversity:.2} |");
            grid.push((name, k, q, a));
        }
    }

    // ── Invariant 3 — the FreshZ0 mechanism check (diversity at K = 16) ──
    // Re-run the two restart arms at K = 16 and compare cross-trial uniques.
    let unique_for = |restart: RestartMode| -> usize {
        (0..N_TRIALS)
            .map(|trial| {
                best_of_k_rollouts(
                    &marginals_refs,
                    &config,
                    &NoScreeningPruner,
                    &sde_config(),
                    &WidthScaleConfig {
                        k_rollouts: 16,
                        selection: WidthSelectionMode::BestQ,
                        restart_mode: restart,
                    },
                    BASE_SEED + trial as u64,
                )
            })
            .collect::<std::collections::HashSet<_>>()
            .len()
    };
    let (u_perturb, u_fresh) = (unique_for(RestartMode::Perturb), unique_for(RestartMode::FreshZ0));
    println!("[Invariant 3] cross-trial uniques at K = 16 (BestQ): Perturb {u_perturb}/{N_TRIALS}, FreshZ0 {u_fresh}/{N_TRIALS}");
    assert!(
        u_fresh >= u_perturb,
        "FreshZ0 (σ = 4.0) diversified LESS than Perturb (γ = 1.0): {u_fresh} < {u_perturb} — the axis is not exercising; re-read the draw before trusting any τ-scale comparison"
    );
    println!("[Invariant 3] FreshZ0 diversity ≥ Perturb ✓");

    // ── Negative control (T3): unshaped fixture — Top1Converged must NOT ──
    // beat MostFrequent under Perturb (both axes, matched K).
    println!("\n[T3 negative control] Perturb arms, quality/agreement by K (Top1Conv must NOT lead MostFreq):");
    let cell = |name: &str, k: usize| -> (f32, f32) {
        grid.iter()
            .find(|(n, kk, _, _)| *n == name && *kk == k)
            .map(|(_, _, q, a)| (*q, *a))
            .unwrap()
    };
    let mut control_violated = false;
    for &k in &k_values[1..] {
        let (q_mf, a_mf) = cell("Perturb+MostFreq", k);
        let (q_tc, a_tc) = cell("Perturb+Top1Conv", k);
        let leads = q_tc > q_mf || a_tc > a_mf;
        control_violated |= leads;
        println!("| K={k} | MostFreq {q_mf:.4}/{a_mf:.4} | Top1Conv {q_tc:.4}/{a_tc:.4} | Top1Conv leads: {leads} |");
    }
    println!("[T3 verdict] control {}", if control_violated { "VIOLATED — a Top1Converged win on an unshaped fixture indicts the FIXTURE (residual proxy is not basin-noise here); the main comparison's interpretation is void" } else { "holds — Top1Converged never leads MostFrequent under Perturb ✓ (unshaped-fixture expectation)" });

    // ── Main comparison readout (T2): FreshZ0+Top1Conv vs Perturb+MostFreq ──
    println!("\n[T2 main comparison] FreshZ0+Top1Conv − Perturb+MostFreq (matched NFE = D·K):");
    for &k in &k_values[1..] {
        let (q_mf, a_mf) = cell("Perturb+MostFreq", k);
        let (q_fz, a_fz) = cell("FreshZ0+Top1Conv", k);
        println!("| K={k} | Δquality {:+.4} | Δagreement {:+.4} |", q_fz - q_mf, a_fz - a_mf);
    }

    // ── T4 — the D-first sweep: D ∈ {2, 4, 8} × K, breadth-pays threshold ──
    println!("\n[T4 D-first sweep] mean top-1 agreement (Perturb+MostFreq | FreshZ0+Top1Conv); breadth-pays K = smallest K beating K = 1 by > 1e-4");
    for &d in &[2usize, 4, 8] {
        let mut cfg = config.clone();
        cfg.draft_lookahead = d;
        let marginals_d = dflash_predict(&weights, &cfg, 0, 0);
        let refs_d: Vec<&[f32]> = marginals_d.iter().map(|s| s.as_slice()).collect();
        let greedy_d = {
            let tree = build_dd_tree_screened(&refs_d, &cfg, &NoScreeningPruner, false);
            extract_best_path(&tree)
        };
        let agreement_for = |restart: RestartMode, selection: WidthSelectionMode, k: usize| -> f32 {
            let mut total = 0.0f32;
            for trial in 0..N_TRIALS {
                let path = best_of_k_rollouts(
                    &refs_d,
                    &cfg,
                    &NoScreeningPruner,
                    &sde_config(),
                    &WidthScaleConfig {
                        k_rollouts: k,
                        selection,
                        restart_mode: restart,
                    },
                    BASE_SEED + trial as u64,
                );
                total += top1_agreement(&greedy_d, &path);
            }
            total / N_TRIALS as f32
        };
        print!("| D={d} |");
        let mut pays: [Option<usize>; 2] = [None; 2];
        for (col, (restart, selection)) in [
            (RestartMode::Perturb, WidthSelectionMode::MostFrequent),
            (RestartMode::FreshZ0, WidthSelectionMode::Top1Converged),
        ]
        .into_iter()
        .enumerate()
        {
            let base = agreement_for(restart, selection, 1);
            print!(" K=1 {base:.4} |");
            for &k in &k_values[1..] {
                let a = agreement_for(restart, selection, k);
                print!(" K={k} {a:.4} |");
                if pays[col].is_none() && a > base + 1e-4 {
                    pays[col] = Some(k);
                }
            }
            print!(" breadth-pays K = {:?} |", pays[col]);
        }
        println!();
    }
    println!("\n[Recorded] D-first law sentence is read off the T4 table after the run (pre-registered form: breadth pays only above a depth knee; the measured knee per arm is the breadth-pays K column).");
}
