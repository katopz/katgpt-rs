//! Issue 917 T3 — deterministic counts bench for the EB lane wiring.
//!
//! EB vs fixed-width-k at matched settings over deterministic seeded
//! synthetic marginals. COUNTS are the metric — per-pass commit-count
//! distributions and DDTree expansion counts — so every number below is
//! box-independent (no timing, no latency claims; that is Bench 917's
//! oracle lane and a future real-checkpoint lane's job).
//!
//! Arms (both lanes): the fixed width-k comparator is the SAME EB policy
//! with the budget disabled — `gamma = INFINITY, max_commit = k` — so the
//! candidate order is identical and only the count rule differs (the T1
//! oracle bench established this identity). `ALL` is the uncapped form
//! (count-equivalent to the shipped all-children expansion). EB arms carry
//! the same `max_commit` cap as the largest fixed-k arm.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/eb_t3_target cargo test -p katgpt-speculative \
//!   --features entropy_bounded_commit \
//!   --bench bench_917_eb_lane_wiring --release -- --nocapture
//! ```

use katgpt_core::entropy_bounded_commit::EbCommitConfig;
use katgpt_speculative::NoPruner;
use katgpt_speculative::entropy_bounded::{
    EbBuildScratch, EbCommitScratch, build_dd_tree_eb_into, dflash_block_commit_eb_with,
};
use katgpt_types::Config;

const DEPTHS: usize = 8;
const VOCAB: usize = 64;
const TREE_BUDGET: usize = 64;
const BLOCKS: usize = 2000;
const DFLASH_STEPS: usize = 8;

fn eb(gamma: f32, max_commit: usize) -> EbCommitConfig {
    EbCommitConfig {
        gamma,
        max_commit,
        proxy: katgpt_core::entropy_bounded_commit::ErrorProxy::Entropy,
    }
}

fn config() -> Config {
    let mut c = Config::draft();
    c.vocab_size = VOCAB;
    c.tree_budget = TREE_BUDGET;
    c.early_exit_patience = 0;
    c.early_exit_gap = 0.0;
    c
}

/// Three marginal families, all deterministic:
/// - FLAT: uniform rows (every surprisal ln V — the stall-prone incumbent
///   case; EB commits the singleton).
/// - PEAKED: 0.7 on a seeded token, geometric decay over the rest — narrow
///   rows where the width should collapse toward the argmax.
/// - RANDOM: seeded normalized rows — the mixed-entropy regime.
fn family(name: &str, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..DEPTHS)
        .map(|_d| match name {
            "FLAT" => vec![1.0 / VOCAB as f32; VOCAB],
            "PEAKED" => {
                let hot = rng.usize(..VOCAB);
                let mut r = vec![0.0f32; VOCAB];
                r[hot] = 0.7;
                let mut rest = 0.3;
                let mut i = 1usize;
                while rest > 0.0 && i < VOCAB {
                    let take = rest * 0.5;
                    r[(hot + i) % VOCAB] = take;
                    rest -= take;
                    i += 1;
                }
                r[hot] += rest;
                r
            }
            _ => {
                let raw: Vec<f32> = (0..VOCAB).map(|_| rng.f32()).collect();
                let sum: f32 = raw.iter().sum();
                raw.iter().map(|&p| p / sum).collect()
            }
        })
        .collect()
}

/// `k:count` pairs of the non-zero histogram buckets, ascending.
fn histogram(hist: &[u64]) -> String {
    hist.iter()
        .enumerate()
        .filter(|&(_, &n)| n > 0)
        .map(|(k, n)| format!("{k}:{n}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn ddtree_table(name: &str, seed: u64) {
    let rows = family(name, seed);
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let config = config();

    println!("\n── DDTree {name} (D={DEPTHS}, V={VOCAB}, budget={TREE_BUDGET}) ──");
    println!(
        "{:<26} {:>6} {:>11} {:>9} {:>7}  commit-count histogram",
        "arm", "nodes", "expansions", "children", "mean_k"
    );

    let arms: Vec<(&str, EbCommitConfig)> = vec![
        ("ALL (shipped width)", eb(f32::INFINITY, usize::MAX)),
        ("fixed k=1", eb(f32::INFINITY, 1)),
        ("fixed k=2", eb(f32::INFINITY, 2)),
        ("fixed k=4", eb(f32::INFINITY, 4)),
        ("fixed k=8", eb(f32::INFINITY, 8)),
        ("EB γ=0.1 cap=8", eb(0.1, 8)),
        ("EB γ=0.3 cap=8", eb(0.3, 8)),
        ("EB γ=1.0 cap=8", eb(1.0, 8)),
        ("EB γ=5.0 cap=8", eb(5.0, 8)),
    ];

    for (label, cfg) in &arms {
        let mut scratch = EbBuildScratch::default();
        let nodes = build_dd_tree_eb_into(&mut scratch, &refs, &config, &NoPruner, cfg).len();
        let expansions = scratch.expansions.max(1);
        println!(
            "{:<26} {:>6} {:>11} {:>9} {:>7.2}  {}",
            label,
            nodes,
            scratch.expansions,
            scratch.children_committed,
            scratch.children_committed as f64 / expansions as f64,
            histogram(&scratch.commit_histogram),
        );
        // Count sanity: no EB arm stalls (every family here has candidates
        // at every depth), and fixed-k arms never exceed their cap.
        assert_eq!(
            scratch.commit_histogram[0], 0,
            "{label}: no stalled expansion"
        );
        let cap = cfg.max_commit.min(VOCAB);
        assert!(
            scratch
                .commit_histogram
                .iter()
                .enumerate()
                .all(|(k, &n)| k <= cap || n == 0),
            "{label}: cap respected"
        );
    }
}

fn dflash_table(name: &str, seed: u64) {
    let rows = family(name, seed.wrapping_add(1));
    let mut flat: Vec<f32> = Vec::with_capacity(DFLASH_STEPS * VOCAB);
    for d in 0..DFLASH_STEPS {
        flat.extend(rows[d % DEPTHS].iter());
    }

    println!(
        "\n── DFlash block commit {name} (steps={DFLASH_STEPS}, V={VOCAB}, blocks={BLOCKS}) ──"
    );
    println!(
        "{:<26} {:>8} {:>9}  commit-count histogram",
        "arm", "mean_k", "max_k"
    );

    let arms: Vec<(&str, EbCommitConfig)> = vec![
        ("ALL (commit-all)", eb(f32::INFINITY, usize::MAX)),
        ("fixed k=1", eb(f32::INFINITY, 1)),
        ("fixed k=2", eb(f32::INFINITY, 2)),
        ("fixed k=4", eb(f32::INFINITY, 4)),
        ("EB γ=0.05 cap=8", eb(0.05, 8)),
        ("EB γ=0.1 cap=8", eb(0.1, 8)),
        ("EB γ=0.3 cap=8", eb(0.3, 8)),
        ("EB γ=1.0 cap=8", eb(1.0, 8)),
        ("EB γ=5.0 cap=8", eb(5.0, 8)),
    ];

    // Each block re-rolls one row's sharpness so the distribution over
    // commit counts is non-degenerate, while staying fully seeded AND
    // identical across arms (the Rng is re-seeded per arm).
    let mut scratch = EbCommitScratch::default();
    let mut tokens = vec![0usize; DFLASH_STEPS];
    let mut block = flat.clone();

    for (label, cfg) in &arms {
        let mut rng = fastrand::Rng::with_seed(seed.wrapping_add(2));
        let mut hist: Vec<u64> = Vec::new();
        let mut total = 0usize;
        let mut max_k = 0usize;
        for b in 0..BLOCKS {
            if b > 0 {
                // Perturb one depth's row deterministically per block.
                let d = rng.usize(..DFLASH_STEPS);
                let src = rng.usize(..VOCAB);
                let dst = rng.usize(..VOCAB);
                let (lo, hi) = if src <= dst { (src, dst) } else { (dst, src) };
                let scale = 0.5 + rng.f32();
                for (i, v) in block[d * VOCAB..(d + 1) * VOCAB].iter_mut().enumerate() {
                    *v = if i == lo {
                        flat[d * VOCAB + i] * scale
                    } else if i == hi {
                        flat[d * VOCAB + i] / scale
                    } else {
                        flat[d * VOCAB + i]
                    };
                }
            }
            let k = dflash_block_commit_eb_with(
                &block,
                DFLASH_STEPS,
                VOCAB,
                &mut scratch,
                &mut tokens,
                cfg,
            );
            total += k;
            max_k = max_k.max(k);
            if hist.len() <= k {
                hist.resize(k + 1, 0);
            }
            hist[k] += 1;
        }
        println!(
            "{:<26} {:>8.2} {:>9}  {}",
            label,
            total as f64 / BLOCKS as f64,
            max_k,
            histogram(&hist),
        );
        assert!(total >= BLOCKS, "{label}: no-stall (≥ 1 commit per block)");
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Issue 917 T3 — EB lane wiring: commit-count + expansion counts  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("counts only — no timing; fixed-k = EB with γ=∞ and cap=k (same order)");

    for (name, seed) in [("FLAT", 11u64), ("PEAKED", 22), ("RANDOM", 33)] {
        ddtree_table(name, seed);
    }
    for (name, seed) in [("FLAT", 11u64), ("PEAKED", 22), ("RANDOM", 33)] {
        dflash_table(name, seed);
    }
}
