//! CLR voter — the `(mean_m v_k,m)^M` nonlinear reliability gate (Plan 284 T2.1/T2.3).
//!
//! This is the headline primitive of Research 255. Given `K` candidate
//! trajectories, `M` claims per trajectory, a [`ClaimExtractor`] and a
//! [`ClaimVerifier`], produce the winning cluster via:
//!
//! 1. Extract `M` claims from each of the `K` trajectories.
//! 2. Verify each `(k, m)` claim: `v_k,m = sigmoid(dot(emb_k,m, dir_m))`.
//! 3. Per trajectory: `r_k = (mean_m v_k,m)^M` — the nonlinear reliability gate.
//! 4. Cluster trajectories by outcome equivalence (caller-supplied `outcome_eq`).
//! 5. Sum reliabilities per cluster.
//! 6. Tiebreak via [`crate::clr::brevity::brevity_tiebreak`] (Long2Short).
//!
//! # Math
//!
//! All activations are **sigmoid**, never softmax (per project convention and
//! the user's `AGENTS.md` rule). The `^M` exponent is what makes this a
//! *nonlinear* reliability gate: a trajectory with one low verdict gets
//! penalized super-linearly, which is the whole point — a single flawed claim
//! drags the trajectory's reliability below the cluster of flawless ones.
//!
//! # Allocation discipline
//!
//! [`clr_vote_minimal`] is the zero-allocation hot path: it writes into the
//! caller-supplied [`ClrScratch`] and returns two scalars (`winner_idx`,
//! `winner_reliability`). [`clr_vote`] returns a full [`VoteResult`] audit
//! trail and DOES allocate — but only for the output, not on the path between
//! input and decision. Hot-path callers (the per-NPC CLR cycle in riir-ai
//! Plan 316) use `clr_vote_minimal`.
//!
//! # M=5 unrolled path (Plan 284 T3.2)
//!
//! The paper default is `M = 5`. General `powf(M)` is ~10× slower than an
//! unrolled integer power. When `config.m == 5` we use the literal
//! `v * v * v * v * v` form, which LLVM turns into 4 multiplies with no
//! libm call. All other `M` fall back to the general `mean.powf(M)` path.

#![allow(clippy::needless_range_loop)]

use crate::clr::brevity::brevity_tiebreak;
use crate::clr::scratch::ClrScratch;
use crate::clr::traits::{ClaimExtractor, ClaimVerifier};
use crate::clr::types::{Cluster, ReliabilityScore, Trajectory, VoteResult};
use crate::clr::ClrConfig;
use katgpt_core::simd::simd_sum_f32;

/// Run the full CLR vote and return a [`VoteResult`] audit trail.
///
/// # Arguments
///
/// * `trajectories` — `K` candidate trajectories. Must be non-empty.
/// * `extractor` — produces exactly `M = config.m` claims per trajectory.
/// * `verifier` — produces `sigmoid(dot(emb, dir))` per `(trajectory, direction)`.
/// * `config` — CLR runtime config (`m`, `tiebreak_eps`, etc.).
/// * `outcome_eq` — caller-supplied outcome-equivalence predicate used for
///   clustering. For LLMs this is answer-equivalence; for game NPCs it's
///   destination-tile + action-type. Naive O(K²) — fine for K ≤ 32.
/// * `scratch` — pre-allocated buffers. [`ClrScratch::reset_for`] is called
///   internally with `(config.k, config.m)`, so the caller should size
///   `ClrScratch::new(config.k, config.m)` once for zero-alloc reuse.
///
/// # Returns
///
/// A [`VoteResult`] containing the winning cluster plus the full per-trajectory
/// audit trail. The winner is chosen by max `Σ r_k` over cluster members,
/// tiebroken by Long2Short brevity (shorter representative wins on ties within
/// `config.tiebreak_eps`).
///
/// # Panics
///
/// Panics if `trajectories` is empty, if any extraction doesn't return exactly
/// `config.m` claims (debug only), or if `scratch` is undersized.
///
/// # Allocation
///
/// Allocates only for the returned `VoteResult`. The verdict/reliability/cluster
/// computation writes into `scratch` in place. Use [`clr_vote_minimal`] for the
/// zero-allocation hot path.
pub fn clr_vote<T, E, V>(
    trajectories: &[Trajectory<T>],
    extractor: &E,
    verifier: &V,
    config: &ClrConfig,
    outcome_eq: &impl Fn(&T, &T) -> bool,
    scratch: &mut ClrScratch,
) -> VoteResult<T>
where
    T: Clone,
    E: ClaimExtractor<T>,
    V: ClaimVerifier<T>,
{
    assert!(
        !trajectories.is_empty(),
        "clr_vote: empty trajectories slice"
    );
    assert!(
        trajectories.len() <= config.k,
        "clr_vote: trajectories.len() {} exceeds config.k {} (max K)",
        trajectories.len(),
        config.k
    );
    let k_count = trajectories.len();
    let m = config.m;

    // Size + zero the scratch buffers for (max-K, M). config.k doubles as the
    // trajectory-count upper bound (paper fixes K ≤ 32 == embedding dim).
    scratch.reset_for(config.k, config.m);

    // Steps 1+2: extract claims + verify each (k, m). We extract into a
    // per-trajectory Vec<Claim> (this is caller-domain work and allocates —
    // the zero-alloc contract is on the *vote* path, not extraction; a future
    // hot-path variant can take pre-extracted claims if a caller needs it).
    for (k, traj) in trajectories.iter().enumerate() {
        let claims = extractor.extract(traj);
        debug_assert_eq!(
            claims.len(),
            m,
            "clr_vote: trajectory {} produced {} claims, expected {}",
            k,
            claims.len(),
            m
        );
        for (m_idx, claim) in claims.iter().enumerate() {
            let v = verifier.verify(claim, m_idx);
            // Direct index write — no push, no realloc. Buffer is pre-sized.
            scratch.verdicts[k * m + m_idx] = v;
        }
    }

    // Step 3: per-trajectory reliability = (mean_m v_k,m)^M.
    let reliability_slice = &mut scratch.reliability[..k_count];
    for k in 0..k_count {
        let row = &scratch.verdicts[k * m..k * m + m];
        let mean_v = simd_sum_f32(row) / m as f32;
        reliability_slice[k] = reliability_gate(mean_v, m);
    }

    // Step 4: cluster by outcome equivalence (naive O(K²)).
    assign_clusters(trajectories, outcome_eq, &mut scratch.cluster_id[..k_count]);

    // Step 5: aggregate cluster totals + pick representative per cluster.
    // We build the cluster list here — this is output allocation, allowed.
    let clusters = build_clusters(
        trajectories,
        &scratch.cluster_id[..k_count],
        &scratch.reliability[..k_count],
    );

    // Step 6: pick winner via brevity tiebreak among clusters within eps of max.
    let cluster_refs: Vec<&Cluster<T>> = clusters.iter().collect();
    let winner_idx = brevity_tiebreak(&cluster_refs, trajectories, config.tiebreak_eps);
    let winner = clusters[winner_idx].clone();

    VoteResult {
        winner,
        all_clusters: clusters,
        per_trajectory_reliability: scratch.reliability[..k_count].to_vec(),
        per_trajectory_verdicts: scratch.verdicts[..k_count * m].to_vec(),
    }
}

/// Run the CLR vote and return only the winning trajectory index + reliability.
///
/// This is the **zero-allocation hot path**. It skips building the
/// `VoteResult` audit trail (`all_clusters`, `per_trajectory_*`). Used by
/// per-NPC CLR cycles (riir-ai Plan 316) where the caller only needs the
/// decision, not the diagnostic trace.
///
/// # Returns
///
/// `(winner_trajectory_idx, winner_cluster_reliability)` where
/// `winner_trajectory_idx` indexes into the input `trajectories` slice and
/// `winner_cluster_reliability` is the `Σ r_k` of the winning cluster.
///
/// # Allocation
///
/// Zero heap allocation after `ClrScratch::new()`. The only allocations are
/// inside `extractor.extract()` (caller-domain; if a caller needs to avoid
/// those, they can supply an extractor that reuses its own scratch). The vote
/// arithmetic + clustering + tiebreak are all in-place on `scratch` + stack.
///
/// # Panics
///
/// Same as [`clr_vote`].
pub fn clr_vote_minimal<T, E, V>(
    trajectories: &[Trajectory<T>],
    extractor: &E,
    verifier: &V,
    config: &ClrConfig,
    outcome_eq: &impl Fn(&T, &T) -> bool,
    scratch: &mut ClrScratch,
) -> (usize, ReliabilityScore)
where
    E: ClaimExtractor<T>,
    V: ClaimVerifier<T>,
{
    assert!(
        !trajectories.is_empty(),
        "clr_vote_minimal: empty trajectories slice"
    );
    assert!(
        trajectories.len() <= config.k,
        "clr_vote_minimal: trajectories.len() {} exceeds config.k {} (max K)",
        trajectories.len(),
        config.k
    );
    let k_count = trajectories.len();
    let m = config.m;

    scratch.reset_for(config.k, config.m);

    // Steps 1+2: extract + verify.
    for (k, traj) in trajectories.iter().enumerate() {
        let claims = extractor.extract(traj);
        debug_assert_eq!(claims.len(), m);
        for (m_idx, claim) in claims.iter().enumerate() {
            scratch.verdicts[k * m + m_idx] = verifier.verify(claim, m_idx);
        }
    }

    // Step 3: reliability gate.
    for k in 0..k_count {
        let row = &scratch.verdicts[k * m..k * m + m];
        let mean_v = simd_sum_f32(row) / m as f32;
        scratch.reliability[k] = reliability_gate(mean_v, m);
    }

    // Step 4: cluster.
    assign_clusters(trajectories, outcome_eq, &mut scratch.cluster_id[..k_count]);

    // Steps 5+6: find the winning cluster without materializing all clusters.
    // We compute cluster totals into a stack-fixed array (max 256 clusters per
    // the u8 cluster_id). For K ≤ 32 this is wildly over-provisioned but free.
    let mut cluster_total: [f32; 256] = [0.0; 256];
    let mut cluster_rep: [usize; 256] = [0usize; 256];
    let mut cluster_rep_tokens: [usize; 256] = [usize::MAX; 256];
    let mut max_cluster_id: u8 = 0;
    for k in 0..k_count {
        let cid = scratch.cluster_id[k];
        cluster_total[cid as usize] += scratch.reliability[k];
        let tokens = trajectories[k].tokens_or_steps;
        // Track the representative as the member with the fewest tokens
        // (Long2Short) — but only among members we've seen so far. This is
        // done in a single pass; later we tiebreak across clusters by Σ r_k.
        if tokens < cluster_rep_tokens[cid as usize] {
            cluster_rep_tokens[cid as usize] = tokens;
            cluster_rep[cid as usize] = k;
        }
        if cid > max_cluster_id {
            max_cluster_id = cid;
        }
    }

    // Pick the winning cluster by max Σ r_k within eps, tiebroken by brevity
    // (fewest representative tokens). Single pass over [0, max_cluster_id].
    let n_clusters = (max_cluster_id as usize) + 1;
    let mut max_total = f32::NEG_INFINITY;
    for cid in 0..n_clusters {
        if cluster_total[cid] > max_total {
            max_total = cluster_total[cid];
        }
    }
    let threshold = max_total - config.tiebreak_eps;

    let mut winner_cluster: usize = 0;
    let mut winner_tokens = usize::MAX;
    for cid in 0..n_clusters {
        if cluster_total[cid] < threshold {
            continue;
        }
        let tokens = cluster_rep_tokens[cid];
        if tokens < winner_tokens {
            winner_tokens = tokens;
            winner_cluster = cid;
        }
    }

    let winner_trajectory_idx = cluster_rep[winner_cluster];
    (winner_trajectory_idx, cluster_total[winner_cluster])
}

/// The nonlinear reliability gate: `(mean_v)^M`.
///
/// For the paper default `M = 5`, we use the unrolled `v * v * v * v * v`
/// form (Plan 284 T3.2) — 4 multiplies, no libm call. All other `M` fall
/// back to `mean_v.powf(M as f32)`.
///
/// `mean_v` is the average sigmoid verdict across `M` directions, so it's
/// already in `(0, 1)`. Raising to the `M`-th power sharpens the gate: a
/// trajectory with mean verdict 0.8 → `0.8^5 = 0.328`, while mean 0.6 →
/// `0.6^5 = 0.078`. The flawed claim (one low verdict) drags the mean down
/// and the exponent amplifies the penalty — that's the whole point.
#[inline(always)]
fn reliability_gate(mean_v: f32, m: usize) -> ReliabilityScore {
    match m {
        // Paper default. Unrolled integer power — no libm, no powf.
        5 => mean_v * mean_v * mean_v * mean_v * mean_v,
        // Small fixed-M unrolls (cheap, common).
        2 => mean_v * mean_v,
        3 => mean_v * mean_v * mean_v,
        4 => mean_v * mean_v * mean_v * mean_v,
        // General path. powf is ~10× slower but correct for arbitrary M.
        _ => mean_v.powf(m as f32),
    }
}

/// Assign each trajectory a cluster id by outcome equivalence (naive O(K²)).
///
/// First-seen outcome wins the cluster id; subsequent equivalent outcomes get
/// the same id. `cluster_id[k]` is the cluster index for trajectory `k`.
/// Output length must equal `trajectories.len()`.
///
/// This is the algorithm called out as Risk #2 in Plan 284: fine for K ≤ 32,
/// doesn't scale. A future `hash_outcome` variant can pre-hash into a
/// `HashMap<u64, Vec<usize>>` if a real caller needs K=128+.
fn assign_clusters<T>(
    trajectories: &[Trajectory<T>],
    outcome_eq: &impl Fn(&T, &T) -> bool,
    cluster_id: &mut [u8],
) {
    let k_count = trajectories.len();
    debug_assert_eq!(cluster_id.len(), k_count);
    // Trajectory 0 is its own cluster (id 0).
    cluster_id[0] = 0;
    let mut next_id: u8 = 1;
    for k in 1..k_count {
        // Find the first earlier trajectory with an equivalent outcome and
        // reuse its cluster id; otherwise mint a new one.
        let mut assigned: Option<u8> = None;
        for prev in 0..k {
            if outcome_eq(&trajectories[k].outcome, &trajectories[prev].outcome) {
                assigned = Some(cluster_id[prev]);
                break;
            }
        }
        cluster_id[k] = match assigned {
            Some(id) => id,
            None => {
                let id = next_id;
                next_id = next_id.wrapping_add(1);
                debug_assert!(
                    next_id != 0 || k_count <= 256,
                    "clr_vote: more than 256 clusters — u8 cluster_id overflowed"
                );
                id
            }
        };
    }
}

/// Build the list of [`Cluster`]s from cluster assignments + reliabilities.
///
/// Each cluster's `total_reliability` is the sum of member `r_k`. The
/// `representative_idx` is the member with the fewest `tokens_or_steps`
/// (Long2Short preference — matches what `brevity_tiebreak` would pick).
fn build_clusters<T: Clone>(
    trajectories: &[Trajectory<T>],
    cluster_id: &[u8],
    reliability: &[ReliabilityScore],
) -> Vec<Cluster<T>> {
    let k_count = trajectories.len();
    let n_clusters = cluster_id.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);
    let mut clusters: Vec<Cluster<T>> = (0..n_clusters)
        .map(|_| Cluster {
            outcome: trajectories[0].outcome.clone(), // overwritten below
            total_reliability: 0.0,
            representative_idx: 0,
            member_indices: Vec::new(),
        })
        .collect();

    for k in 0..k_count {
        let cid = cluster_id[k] as usize;
        clusters[cid].member_indices.push(k);
        clusters[cid].total_reliability += reliability[k];
        // Representative = fewest tokens (Long2Short). First-seen wins ties.
        let tokens = trajectories[k].tokens_or_steps;
        let cur_best = clusters[cid].representative_idx;
        let cur_best_tokens = trajectories[cur_best].tokens_or_steps;
        if clusters[cid].member_indices.len() == 1 || tokens < cur_best_tokens {
            clusters[cid].representative_idx = k;
            clusters[cid].outcome = trajectories[k].outcome.clone();
        }
    }

    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clr::extractor::FnClaimExtractor;
    use crate::clr::traits::DirectionVectorSource;
    use crate::clr::types::Claim;
    use crate::clr::verifier::SigmoidProjectionVerifier;
    use blake3::Hasher;

    /// Toy direction source: M unit vectors, one per axis (so direction `m`
    /// picks out the `m`-th component of the claim embedding).
    struct AxisDirections {
        #[allow(dead_code)]
        m: usize,
        k: usize,
        vectors: Vec<f32>,
    }

    impl AxisDirections {
        fn new(m: usize, k: usize) -> Self {
            let mut vectors = vec![0.0f32; m * k];
            for mi in 0..m {
                vectors[mi * k + mi] = 1.0; // unit vector along axis mi
            }
            Self { m, k, vectors }
        }
    }

    impl DirectionVectorSource for AxisDirections {
        fn direction(&self, idx: usize) -> &[f32] {
            &self.vectors[idx * self.k..(idx + 1) * self.k]
        }
        fn blake3(&self) -> [u8; 32] {
            let mut h = Hasher::new();
            h.update(bytemuck::cast_slice(&self.vectors));
            let mut out = [0u8; 32];
            out.copy_from_slice(h.finalize().as_bytes());
            out
        }
        fn version(&self) -> u64 {
            1
        }
    }

    /// Build a trajectory whose claim embeddings encode verdicts directly:
    /// claim `m` has embedding `[0,...,v_m,...,0]` so that
    /// `sigmoid(dot(emb, dir_m)) = sigmoid(v_m)`. This gives us full control
    /// over the verdicts without depending on RNG.
    fn trajectory_with_verdicts(outcome: usize, tokens: usize, raw_verdicts: &[f32], k: usize) -> Trajectory<usize> {
        let m = raw_verdicts.len();
        let claims = (0..m)
            .map(|mi| {
                let mut emb = vec![0.0; k];
                // We want sigmoid(emb[mi]) == raw_verdicts[mi], i.e.
                // emb[mi] = logit(raw_verdicts[mi]).
                let p = raw_verdicts[mi].clamp(1e-6, 1.0 - 1e-6);
                emb[mi] = (p / (1.0 - p)).ln();
                Claim { embedding: emb, payload: outcome }
            })
            .collect();
        Trajectory {
            outcome,
            tokens_or_steps: tokens,
            claims,
            log_probs: None,
        }
    }

    #[test]
    fn clr_vote_picks_flawless_cluster() {
        // 5 trajectories, 2 outcome clusters, M=5 verdicts each.
        // Cluster A (outcome=0): all verdicts 0.9 → mean 0.9 → 0.9^5 = 0.59049
        // Cluster B (outcome=1): two trajectories, one with a flawed claim
        //   (verdict 0.1 on direction 2, rest 0.9) and one clean.
        //   Clean:        mean 0.9 → 0.59049
        //   Flawed:       mean (0.9*4 + 0.1)/5 = 0.74 → 0.74^5 = 0.222
        //   Cluster B total = 0.59049 + 0.222 = 0.81249
        // Cluster A total = 0.59049 (single member)
        // → winner = Cluster B (outcome=1).
        let k = 8;
        let m = 5;
        let high = 0.9f32;
        let flawed = 0.1f32;
        let trajs = vec![
            trajectory_with_verdicts(0, 100, &[high, high, high, high, high], k),
            trajectory_with_verdicts(1, 50, &[high, high, high, high, high], k),
            trajectory_with_verdicts(1, 80, &[high, high, flawed, high, high], k),
            trajectory_with_verdicts(1, 60, &[high, high, high, high, high], k),
            trajectory_with_verdicts(2, 40, &[flawed, flawed, flawed, flawed, flawed], k),
        ];

        let dirs = AxisDirections::new(m, k);
        let verifier = SigmoidProjectionVerifier::new(&dirs, k);
        // Extractor just returns the trajectory's pre-built claims.
        let extractor = FnClaimExtractor::new(m, |t: &Trajectory<usize>| t.claims.clone());
        let config = ClrConfig { k, m, ..ClrConfig::default() };
        let mut scratch = ClrScratch::new(trajs.len(), m);

        let result = clr_vote(
            &trajs,
            &extractor,
            &verifier,
            &config,
            &|a: &usize, b: &usize| a == b,
            &mut scratch,
        );

        // Winner is cluster 1 (outcome=1).
        assert_eq!(result.winner.outcome, 1, "winner should be outcome 1");
        // Cluster 1 has 3 members.
        assert_eq!(result.winner.member_indices.len(), 3);
        // Per-trajectory reliability: the flawed one (idx 2) should be lowest
        // in its cluster.
        let r_flawed = result.per_trajectory_reliability[2];
        let r_clean = result.per_trajectory_reliability[1];
        assert!(
            r_flawed < r_clean,
            "flawed trajectory reliability {} should be < clean {}",
            r_flawed,
            r_clean
        );
        // Spot-check the math: clean reliability ≈ 0.9^5 = 0.59049.
        assert!((r_clean - 0.59049_f32).abs() < 1e-4);
    }

    #[test]
    fn clr_vote_minimal_matches_vote_winner() {
        let k = 8;
        let m = 5;
        let high = 0.9f32;
        let flawed = 0.1f32;
        let trajs = vec![
            trajectory_with_verdicts(0, 100, &[high, high, high, high, high], k),
            trajectory_with_verdicts(1, 50, &[high, high, high, high, high], k),
            trajectory_with_verdicts(1, 80, &[high, high, flawed, high, high], k),
            trajectory_with_verdicts(1, 60, &[high, high, high, high, high], k),
        ];
        let dirs = AxisDirections::new(m, k);
        let verifier = SigmoidProjectionVerifier::new(&dirs, k);
        let extractor = FnClaimExtractor::new(m, |t: &Trajectory<usize>| t.claims.clone());
        let config = ClrConfig { k, m, ..ClrConfig::default() };

        // Full vote.
        let mut scratch_full = ClrScratch::new(trajs.len(), m);
        let full = clr_vote(
            &trajs,
            &extractor,
            &verifier,
            &config,
            &|a: &usize, b: &usize| a == b,
            &mut scratch_full,
        );
        // Minimal vote.
        let mut scratch_min = ClrScratch::new(trajs.len(), m);
        let (min_idx, min_rel) = clr_vote_minimal(
            &trajs,
            &extractor,
            &verifier,
            &config,
            &|a: &usize, b: &usize| a == b,
            &mut scratch_min,
        );

        // Minimal winner trajectory should be a member of the full winner cluster.
        assert!(
            full.winner.member_indices.contains(&min_idx),
            "minimal winner {} should be in full winner cluster {:?}",
            min_idx,
            full.winner.member_indices
        );
        // Minimal reliability should match the full winner cluster total.
        assert!(
            (min_rel - full.winner.total_reliability).abs() < 1e-5,
            "minimal reliability {} should match full winner total {}",
            min_rel,
            full.winner.total_reliability
        );
    }

    #[test]
    fn reliability_gate_unrolled_m5_matches_powf() {
        // The unrolled M=5 path must match the general powf path.
        let mean = 0.74f32;
        let unrolled = mean * mean * mean * mean * mean;
        let via_fn = reliability_gate(mean, 5);
        assert!((unrolled - via_fn).abs() < 1e-7);
        // And match powf within f32 precision.
        let general = mean.powf(5.0);
        assert!((general - via_fn).abs() < 1e-6);
    }

    #[test]
    fn brevity_tiebreak_picks_shorter_representative() {
        // Two clusters tied on Σ r_k (both total 1.0 within eps).
        // Cluster A rep has 100 tokens; cluster B rep has 50 tokens.
        // Winner should be cluster B (shorter).
        let k = 8;
        let m = 5;
        let high = 0.9f32;
        let trajs = vec![
            trajectory_with_verdicts(0, 100, &[high, high, high, high, high], k),
            trajectory_with_verdicts(1, 50, &[high, high, high, high, high], k),
        ];
        let dirs = AxisDirections::new(m, k);
        let verifier = SigmoidProjectionVerifier::new(&dirs, k);
        let extractor = FnClaimExtractor::new(m, |t: &Trajectory<usize>| t.claims.clone());
        let config = ClrConfig { k, m, ..ClrConfig::default() };
        let mut scratch = ClrScratch::new(trajs.len(), m);

        let result = clr_vote(
            &trajs,
            &extractor,
            &verifier,
            &config,
            &|a: &usize, b: &usize| a == b,
            &mut scratch,
        );

        // Both clusters have the same reliability (0.59049), so the tiebreak
        // picks the shorter representative → trajectory 1 (50 tokens).
        assert_eq!(result.winner.outcome, 1);
        assert_eq!(result.winner.representative_idx, 1);
    }

    #[test]
    #[should_panic(expected = "empty trajectories slice")]
    fn clr_vote_panics_on_empty() {
        let extractor = FnClaimExtractor::new(5, |_: &Trajectory<usize>| vec![]);
        let dirs = AxisDirections::new(5, 8);
        let verifier = SigmoidProjectionVerifier::new(&dirs, 8);
        let config = ClrConfig::default();
        let mut scratch = ClrScratch::new(1, 5);
        let _ = clr_vote(
            &[],
            &extractor,
            &verifier,
            &config,
            &|a: &usize, b: &usize| a == b,
            &mut scratch,
        );
    }

    /// Spec smoke test (Plan 284 Phase 2 validation §3): 5 trajectories, all
    /// same outcome (single cluster of 5), `clr_vote_minimal` must return
    /// winner_idx 0 and positive reliability.
    #[test]
    fn clr_vote_minimal_smoke_single_cluster() {
        let k = 8;
        let m = 5;
        let high = 0.9f32;
        // 5 trajectories, all outcome 42, varying token counts.
        let trajs: Vec<Trajectory<usize>> = (0..5)
            .map(|i| {
                trajectory_with_verdicts(42, 100 + i * 10, &[high, high, high, high, high], k)
            })
            .collect();
        let dirs = AxisDirections::new(m, k);
        let verifier = SigmoidProjectionVerifier::new(&dirs, k);
        let extractor = FnClaimExtractor::new(m, |t: &Trajectory<usize>| t.claims.clone());
        let config = ClrConfig { k, m, ..ClrConfig::default() };
        let mut scratch = ClrScratch::new(config.k, config.m);

        let (winner_idx, reliability) = clr_vote_minimal(
            &trajs,
            &extractor,
            &verifier,
            &config,
            &|a: &usize, b: &usize| a == b,
            &mut scratch,
        );
        // Single cluster → winner is the min-token representative (traj 0).
        assert_eq!(winner_idx, 0, "winner should be trajectory 0 (fewest tokens)");
        assert!(reliability > 0.0, "reliability must be positive, got {}", reliability);
        // 5 trajectories × 0.9^5 ≈ 0.59049 each → total ≈ 2.95.
        assert!(reliability > 1.0, "5-member cluster reliability should be > 1.0, got {}", reliability);
    }
}
