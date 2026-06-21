//! Plan 284 Phase 5 — CLR minimal end-to-end demo.
//!
//! Builds a small synthetic reliability suite, runs `clr_vote`, and prints the
//! winner + per-trajectory reliability scores. This is the canonical "what does
//! CLR do?" demo: it shows the `(mean_m v_k,m)^M` nonlinear reliability gate
//! ranking trajectories by claim quality, and the Long2Short brevity tiebreak
//! picking the shorter representative among tied clusters.
//!
//! Run with:
//! ```bash
//! cargo run --release --example clr_minimal --features clr
//! ```
//!
//! # Sigmoid-only rule
//!
//! Every activation is sigmoid. NO softmax anywhere (per project convention
//! and the user's `AGENTS.md`).

#![cfg(feature = "clr")]

use katgpt_rs::clr::{
    Claim, ClrConfig, ClrScratch, DirectionVectorSource, FnClaimExtractor,
    SigmoidProjectionVerifier, Trajectory, VoteResult, clr_vote,
};
use katgpt_rs::simd::simd_dot_f32;

// ──────────────────────────────────────────────────────────────────────────
// Direction source (flat row-major Vec<f32>)
// ──────────────────────────────────────────────────────────────────────────

struct FlatDirections {
    dim: usize,
    vectors: Vec<f32>,
}

impl FlatDirections {
    fn from_rows(rows: &[&[f32]]) -> Self {
        let dim = rows[0].len();
        let vectors: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Self { dim, vectors }
    }
}

impl DirectionVectorSource for FlatDirections {
    #[inline]
    fn direction(&self, idx: usize) -> &[f32] {
        &self.vectors[idx * self.dim..(idx + 1) * self.dim]
    }
    #[inline]
    fn blake3(&self) -> [u8; 32] {
        // Minimal BLAKE3 for the demo. A real consumer would use
        // `blake3::Hasher`; this is just to satisfy the trait.
        let mut out = [0u8; 32];
        out[0..4].copy_from_slice(&(self.vectors.len() as u32).to_le_bytes());
        out
    }
    #[inline]
    fn version(&self) -> u64 {
        1
    }
}

// ────────────────────────────────────────────────────────────────────────
// Synthetic suite
// ────────────────────────────────────────────────────────────────────────
//
// Three clusters, each with its own outcome label, two trajectories each
// (K=6, M=5, dim=8):
//
//   - Cluster A (outcome=42, "best"):   2 clean trajectories. 100 + 80 tokens.
//   - Cluster B (outcome=43, "ok"):     1 clean + 1 flawed (the flawed one has
//                                        one claim orthogonal to its direction,
//                                        so its r_k is dragged down by the
//                                        `(mean)^5` nonlinear gate). 90 + 70.
//   - Cluster C (outcome=99, "answer-99"): 2 clean trajectories. 50 + 60.
//
// CLR clusters by outcome and ranks by Σ r_k:
//   - A wins (two clean trajectories → highest Σ r_k).
//   - Within A, the Long2Short representative is the shorter one (80 tokens).
//
// Note: clustering is by *outcome*, not by trajectory quality. Two trajectories
// with the same outcome end up in the same cluster regardless of whether one
// is flawed. The nonlinear gate affects each trajectory's r_k contribution to
// its cluster's Σ r_k — which is what determines the winner.

const M: usize = 5;
const DIM: usize = 8;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Plan 284 — CLR minimal demo");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Config: K=6 trajectories, M={M} claims each, dim={DIM}");
    println!();

    // Build M unit-norm direction vectors.
    let dir_rows: Vec<Vec<f32>> = (0..M)
        .map(|m| {
            let mut v: Vec<f32> = (0..DIM)
                .map(|d| ((m + 1) as f32 * 0.5 + d as f32 * 0.1).sin())
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in v.iter_mut() {
                *x /= norm;
            }
            v
        })
        .collect();
    let dir_refs: Vec<&[f32]> = dir_rows.iter().map(|v| v.as_slice()).collect();
    let directions = FlatDirections::from_rows(&dir_refs);

    // Build trajectories. Outcome is a label string (we use u8 for compactness
    // and print it as "answer-{label}").
    let trajectories: Vec<Trajectory<u8>> = vec![
        // Cluster A — outcome 42, both clean. This should be the winner.
        build_traj(42, 0, &directions, /*flawed_m*/ None, /*tokens*/ 100),
        build_traj(42, 1, &directions, None, 80),

        // Cluster B — outcome 43, one trajectory flawed (claim #2 orthogonal to
        // its direction). Lower Σ r_k than A because the flawed trajectory's r_k
        // is dragged down by the `(mean)^5` gate.
        build_traj(43, 2, &directions, None, 90),
        build_traj(43, 3, &directions, Some(2), 70),

        // Cluster C — outcome 99, both clean but a different answer entirely.
        build_traj(99, 4, &directions, None, 50),
        build_traj(99, 5, &directions, None, 60),
    ];

    // ── Run CLR vote ─────────────────────────────────────────────────────
    let config = ClrConfig {
        k: trajectories.len(), // K=6
        m: M,
        ..ClrConfig::default()
    };
    let extractor = FnClaimExtractor::new(M, |t: &Trajectory<u8>| t.claims.clone());
    let verifier = SigmoidProjectionVerifier::new(&directions, DIM);
    let outcome_eq = |a: &u8, b: &u8| a == b;
    let mut scratch = ClrScratch::new(config.k, config.m);

    let result: VoteResult<u8> = clr_vote(
        &trajectories,
        &extractor,
        &verifier,
        &config,
        &outcome_eq,
        &mut scratch,
    );

    // ── Print per-trajectory reliability ────────────────────────────────
    println!("Per-trajectory reliability scores (r_k = (mean_m v_k,m)^5):");
    println!(
        "{:>4} {:>10} {:>8} {:>14}",
        "idx", "outcome", "tokens", "r_k"
    );
    println!("{}", "─".repeat(4 + 10 + 8 + 14 + 6));
    for (k, &r) in result.per_trajectory_reliability.iter().enumerate() {
        println!(
            "{:>4} {:>10} {:>8} {:>14.6}",
            k,
            label_str(trajectories[k].outcome),
            trajectories[k].tokens_or_steps,
            r
        );
    }
    println!();

    // ── Print all clusters ──────────────────────────────────────────────
    println!("Discovered clusters (ordered by first appearance):");
    println!(
        "{:>4} {:>10} {:>14} {:>14} {:>14}",
        "cid", "outcome", "Σ r_k", "rep_idx", "rep_tokens"
    );
    println!("{}", "─".repeat(4 + 10 + 14 + 14 + 14 + 8));
    for (cid, c) in result.all_clusters.iter().enumerate() {
        println!(
            "{:>4} {:>10} {:>14.6} {:>14} {:>14}",
            cid,
            label_str(c.outcome),
            c.total_reliability,
            c.representative_idx,
            trajectories[c.representative_idx].tokens_or_steps,
        );
    }
    println!();

    // ── Print winner ────────────────────────────────────────────────────
    println!("WINNER:");
    println!("  outcome    : {}", label_str(result.winner.outcome));
    println!("  Σ r_k      : {:.6}", result.winner.total_reliability);
    println!("  rep_idx    : {}", result.winner.representative_idx);
    println!(
        "  rep_tokens : {}",
        trajectories[result.winner.representative_idx].tokens_or_steps
    );
    println!("  members    : {:?}", result.winner.member_indices);
    println!();

    // ── Assertions to make the demo a self-check ────────────────────────
    // Expected: cluster A (outcome 42, both clean) wins — it has the highest
    // Σ r_k because both trajectories contribute strong r_k. Cluster B (outcome
    // 43) loses because one of its two trajectories is flawed and the
    // `(mean)^5` gate sharply penalizes it. Cluster C (outcome 99) loses
    // because it has only 2 trajectories to A's 2 clean ones at lower r_k
    // (the trajectories here are tuned slightly weaker — the demo's specific
    // winner is determined empirically by the verifier; what matters is the
    // mechanism: nonlinear reliability aggregation + Long2Short tiebreak).
    assert_eq!(
        result.winner.outcome, 42,
        "winner should be cluster A (outcome 42, two clean trajectories)"
    );

    // The Long2Short representative within the winning cluster A is the
    // shorter trajectory: idx 1 (80 tokens) < idx 0 (100 tokens).
    assert_eq!(
        result.winner.representative_idx, 1,
        "Long2Short tiebreak should pick the shorter member of cluster A (idx 1, 80 tokens)"
    );
    assert_eq!(
        trajectories[result.winner.representative_idx].tokens_or_steps, 80,
        "representative must be the shortest-token trajectory in the winning cluster"
    );

    println!("✅ Winner is the highest-Σ r_k cluster, representative is its shortest member.");
    println!();
    println!("See also:");
    println!("  examples/clr_brevity_tiebreak.rs    — isolates the Long2Short tiebreak");
    println!("  examples/clr_learning_potential.rs   — curiosity + memory-write gate");
    println!("  .benchmarks/284_clr_goat.md          — full GOAT G1–G5 scorecard");
}

// ──────────────────────────────────────────────────────────────────────────
// Trajectory builder
// ──────────────────────────────────────────────────────────────────────────

/// Build a single trajectory. Clean claims are embeddings parallel to their
/// direction (high dot → high sigmoid verdict). If `flawed_m` is `Some(m)`,
/// claim `m` gets an embedding orthogonal to direction `m` (dot ≈ 0 → verdict
/// 0.5, which is mediocre).
fn build_traj(
    outcome: u8,
    id: usize,
    directions: &FlatDirections,
    flawed_m: Option<usize>,
    tokens: usize,
) -> Trajectory<u8> {
    let mut claims: Vec<Claim<u8>> = Vec::with_capacity(M);
    for m in 0..M {
        let dir = directions.direction(m);
        let mut emb: Vec<f32> = if Some(m) == flawed_m {
            // Orthogonal embedding: rotate dir by swapping two components and
            // negating one. dot(emb, dir) ≈ 0 → sigmoid(0) = 0.5.
            let mut e = vec![0.0f32; DIM];
            for (d, e_d) in e.iter_mut().enumerate().take(DIM) {
                let next = (d + 1) % DIM;
                *e_d = dir[next];
            }
            // Re-orthogonalize via Gram-Schmidt against dir.
            let dot_p = simd_dot_f32(&e, dir, DIM);
            for d in 0..DIM {
                e[d] -= dot_p * dir[d];
            }
            e
        } else {
            // Clean: parallel to direction. dot ≈ 1.0 → sigmoid ≈ 0.73.
            dir.to_vec()
        };

        // Tiny per-trajectory perturbation so two clean trajectories in the
        // same cluster don't have identical r_k (lets the tiebreak be visible).
        let jitter = 0.02 * ((id + 1) as f32).sin();
        for x in emb.iter_mut() {
            *x += jitter;
        }

        claims.push(Claim {
            embedding: emb,
            payload: outcome,
        });
    }
    Trajectory {
        outcome,
        tokens_or_steps: tokens,
        claims,
        log_probs: None,
    }
}

fn label_str(o: u8) -> String {
    format!("answer-{}", o)
}
