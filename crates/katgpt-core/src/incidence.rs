//! Incidence-mask algebra: the thought × agent agreement substrate.
//!
//! Distilled from ThoughtComm (arXiv:2510.20733, NeurIPS 2025) via riir-ai
//! Research 364 / Issue 874: the transferable artifact is not thought
//! *recovery* but the **incidence mask as a first-class object** — the
//! paper's Thm 3 says the who-shares-what structure is the identifiable
//! artifact. This stack already builds such masks by construction (CLR
//! observer sets, sheaf restriction maps, npc_comms slices, healer fan-out
//! hits) and computes none of the algebra over them. This module ships it:
//!
//! - **agreement count** `α_j = Σ_k 1[j ∈ support(k)]` — how many distinct
//!   agents' states carry thought j ([`agreement_counts_into`]);
//! - **agreement-tier weights** — one monotone σ-saturated curve
//!   ([`agreement_score`]) with two consumer maps: the routing ladder
//!   [`routing_weight`] (α=1 is EXACTLY the unweighted path — the
//!   degeneration contract; a private thought is never penalized, the
//!   Bench-013 soft-bias lesson: never gate attention) and the contagion
//!   ladder [`contagion_strength`] (α=1 is EXACTLY zero crowd strength —
//!   a single witness cannot stampede the crowd; the measured Plan-019 CLR
//!   failure this fixes);
//! - **shared/private split** — shared = support with α ≥ 2, private =
//!   α == 1 (paper Thms 1+2), plus the **private-retention counter**
//!   [`private_fractions_into`]: never report agreement without it —
//!   collapse = conformity, not correctness (the paper's Appx C.2);
//! - **Hall feasibility** — can every agent be matched to a DISTINCT
//!   supported thought (Hopcroft–Karp, [`hall_max_matching_into`],
//!   zero-alloc scratch);
//! - **mask audit** ([`audit_mask`]) — support sizes, α distribution, and
//!   the density alert (a dense mask is the crowd-panic precondition);
//! - **deterministic tier ordering** ([`rank_by_agreement_into`]) —
//!   α desc, index asc (the Issue-849 lesson: a partial order truncated at
//!   a cap must never leave the tie-break to a per-process hasher).
//!
//! # Mask layout
//!
//! Agent-major row-major: `mask[agent * n_thoughts + thought]`. Every
//! function is a pure computation over caller-owned slices; zero heap
//! allocation on the hot path (the `_into` variants take caller scratch;
//! only the `Vec`-returning conveniences allocate, and each says so).
//!
//! # Domain boundary
//!
//! Think-brain local: masks, α counts, and tier weights are never synced.
//! What crosses a sync surface stays raw (contagion intensity, witness
//! counts — the "sync the scalars" doctrine).

use crate::sigmoid;

/// Mask-density fraction at or above which [`MaskAudit::dense`] fires.
///
/// A dense mask (most agents carry most thoughts) is the degenerate
/// "everyone perceives everything" world — the precondition for the
/// crowd-panic failure the α-weighted contagion arm exists to fix. It is a
/// *warning*, never a gate (the Bench-013 lesson).
pub const DENSITY_ALERT: f32 = 0.5;

// ── counts ──────────────────────────────────────────────────────────────

/// Per-thought agreement counts: `out[j] = Σ_k 1[mask[k][j]]`.
///
/// Zero-alloc; writes only the first `n_thoughts` entries of `out`.
/// Deterministic (fixed index order — no hashing, no iteration-order
/// dependence).
///
/// # Panics
///
/// Panics if `mask.len() != n_agents * n_thoughts` or `out.len() <
/// n_thoughts`.
pub fn agreement_counts_into(
    mask: &[bool],
    n_agents: usize,
    n_thoughts: usize,
    out: &mut [u32],
) {
    assert_eq!(
        mask.len(),
        n_agents * n_thoughts,
        "incidence mask layout: mask.len() == n_agents * n_thoughts"
    );
    assert!(
        out.len() >= n_thoughts,
        "agreement_counts_into: out must hold n_thoughts entries"
    );
    out[..n_thoughts].fill(0);
    for agent in 0..n_agents {
        let row = &mask[agent * n_thoughts..(agent + 1) * n_thoughts];
        for (j, &carried) in row.iter().enumerate() {
            if carried {
                out[j] += 1;
            }
        }
    }
}

/// Per-agent support sizes: `out[k] = Σ_j 1[mask[k][j]]`.
///
/// Zero-alloc; writes only the first `n_agents` entries of `out`.
///
/// # Panics
///
/// Same layout contract as [`agreement_counts_into`] (with `out.len() >=
/// n_agents`).
pub fn support_sizes_into(
    mask: &[bool],
    n_agents: usize,
    n_thoughts: usize,
    out: &mut [u32],
) {
    assert_eq!(
        mask.len(),
        n_agents * n_thoughts,
        "incidence mask layout: mask.len() == n_agents * n_thoughts"
    );
    assert!(
        out.len() >= n_agents,
        "support_sizes_into: out must hold n_agents entries"
    );
    for (k, row) in mask.chunks_exact(n_thoughts).enumerate() {
        out[k] = row.iter().filter(|&&carried| carried).count() as u32;
    }
}

// ── tier weights ────────────────────────────────────────────────────────

/// The raw monotone agreement curve: `σ(κ·(α−1)) ∈ (0, 1)`.
///
/// α=0 and α=1 are anchored to EXACTLY `0.5` (the contract anchor: the
/// degeneration maps below are bit-exact regardless of the sigmoid
/// implementation). Monotone nondecreasing in α for `κ ≥ 0`; saturates
/// toward 1. `κ = 0` flattens the curve entirely (the kill-switch: every
/// α scores 0.5, so every consumer map degenerates to its neutral value).
///
/// `κ` is expected `≥ 0`; a negative `κ` inverts the ladder (caller error).
#[must_use]
pub fn agreement_score(alpha: u32, kappa: f32) -> f32 {
    if alpha <= 1 {
        return 0.5;
    }
    sigmoid(kappa * (alpha as f32 - 1.0))
}

/// Routing tier weight: `0.5 + σ(κ·(α−1))` — the α-ladder for *priority*.
///
/// `w(1) = 1.0` EXACTLY: a private thought routes bit-identically to the
/// unweighted path (the α=1 degeneration contract). Monotone nondecreasing
/// in α (carried by more witnesses → routes ahead); saturates to 1.5. This
/// is a soft bias over scores — never a hard gate (the Bench-013 lesson:
/// a hard gate silently drops plans for half the population).
#[must_use]
pub fn routing_weight(alpha: u32, kappa: f32) -> f32 {
    0.5 + agreement_score(alpha, kappa)
}

/// Contagion crowd strength: `2·σ(κ·(α−1)) − 1 ∈ [0, 1)`.
///
/// `c(1) = 0.0` EXACTLY — a single witness broadcasts at zero crowd
/// strength (one misperceiver cannot stampede the town; the Plan-019 CLR
/// demotion fix). `c(2) = 2·σ(κ) − 1` (≈ 0.96 at the house default
/// κ = 4 — a genuinely-shared threat mobilizes at full strength);
/// monotone nondecreasing; saturates to 1. Multiply a broadcast intensity
/// by this to get the α-weighted relay.
#[must_use]
pub fn contagion_strength(alpha: u32, kappa: f32) -> f32 {
    2.0 * agreement_score(alpha, kappa) - 1.0
}

// ── shared / private split ──────────────────────────────────────────────

/// Split one agent's support into shared and private thoughts.
///
/// Shared = supported thoughts with `alpha[j] >= 2` (another agent also
/// carries them — paper Thm 1); private = supported thoughts with
/// `alpha[j] == 1` (this agent alone — paper Thm 2). Returns
/// `(shared, private)`. An empty support returns `(0, 0)`.
///
/// `alpha` is the global per-thought agreement count
/// ([`agreement_counts_into`]).
#[must_use]
pub fn shared_private_counts(support: &[bool], alpha: &[u32]) -> (u32, u32) {
    assert_eq!(
        support.len(),
        alpha.len(),
        "shared_private_counts: support and alpha must align"
    );
    let mut shared = 0u32;
    let mut private = 0u32;
    for (&carried, &a) in support.iter().zip(alpha.iter()) {
        if !carried {
            continue;
        }
        if a >= 2 {
            shared += 1;
        } else {
            private += 1;
        }
    }
    (shared, private)
}

/// Per-agent private-retention fraction:
/// `private_frac(k) = |S_private(k)| / |support(k)|`.
///
/// The consensus≠accuracy discipline (paper Appx C.2): agreement metrics
/// rising while this falls to 0 is CONFORMITY, not correctness. Empty
/// support → `1.0` (vacuously nothing shared). Zero-alloc; `alpha_scratch`
/// must hold `n_thoughts` entries (receives the agreement counts as a
/// side effect); `out` receives the first `n_agents` fractions.
///
/// # Panics
///
/// Same layout contract as [`agreement_counts_into`].
pub fn private_fractions_into(
    mask: &[bool],
    n_agents: usize,
    n_thoughts: usize,
    alpha_scratch: &mut [u32],
    out: &mut [f32],
) {
    assert!(
        out.len() >= n_agents,
        "private_fractions_into: out must hold n_agents entries"
    );
    agreement_counts_into(mask, n_agents, n_thoughts, alpha_scratch);
    for k in 0..n_agents {
        let row = &mask[k * n_thoughts..(k + 1) * n_thoughts];
        let (shared, private) = shared_private_counts(row, alpha_scratch);
        let support = shared + private;
        out[k] = if support == 0 {
            1.0
        } else {
            private as f32 / support as f32
        };
    }
}

// ── Hall feasibility ────────────────────────────────────────────────────

/// Zero-alloc scratch for [`hall_max_matching_into`].
///
/// Construct once (sized for the largest mask), reuse across calls — the
/// `CollectiveThreatScratch` pattern.
#[derive(Debug, Default)]
pub struct HopcroftKarpScratch {
    match_agent: Vec<Option<usize>>,
    match_thought: Vec<Option<usize>>,
    dist: Vec<u32>,
    queue: Vec<usize>,
}

impl HopcroftKarpScratch {
    /// Pre-allocate for `n_agents` agents × `n_thoughts` thoughts.
    #[must_use]
    pub fn with_capacity(n_agents: usize, n_thoughts: usize) -> Self {
        Self {
            match_agent: Vec::with_capacity(n_agents),
            match_thought: Vec::with_capacity(n_thoughts),
            dist: Vec::with_capacity(n_agents),
            queue: Vec::with_capacity(n_agents),
        }
    }
}

/// Maximum bipartite matching agents → thoughts over the incidence mask
/// (Hopcroft–Karp; deterministic — neighbors scanned in ascending thought
/// index).
///
/// **Hall feasibility** (every agent matched to a DISTINCT supported
/// thought) ⟺ the result equals `n_agents`. Zero steady-state alloc via
/// `scratch` (grown once if a larger mask arrives).
pub fn hall_max_matching_into(
    mask: &[bool],
    n_agents: usize,
    n_thoughts: usize,
    scratch: &mut HopcroftKarpScratch,
) -> usize {
    assert_eq!(
        mask.len(),
        n_agents * n_thoughts,
        "incidence mask layout: mask.len() == n_agents * n_thoughts"
    );
    if n_agents == 0 || n_thoughts == 0 {
        return 0;
    }

    let s = scratch;
    s.match_agent.clear();
    s.match_agent.resize(n_agents, None);
    s.match_thought.clear();
    s.match_thought.resize(n_thoughts, None);
    s.dist.clear();
    s.dist.resize(n_agents, 0);

    let neighbors =
        |a: usize| &mask[a * n_thoughts..(a + 1) * n_thoughts];

    // DFS: augment along level graphs. Nested fn (hoisted by the compiler);
    // small depth — bounded by the agent count.
    fn dfs(a: usize, mask: &[bool], n_thoughts: usize, s: &mut HopcroftKarpScratch) -> bool {
        const INF: u32 = u32::MAX;
        let row = &mask[a * n_thoughts..(a + 1) * n_thoughts];
        for (t, &carried) in row.iter().enumerate() {
            if !carried {
                continue;
            }
            let next_free = match s.match_thought[t] {
                None => true,
                Some(a2) => s.dist[a2] == s.dist[a] + 1 && dfs(a2, mask, n_thoughts, s),
            };
            if next_free {
                s.match_thought[t] = Some(a);
                s.match_agent[a] = Some(t);
                return true;
            }
        }
        s.dist[a] = INF;
        false
    }

    let mut matching = 0usize;
    loop {
        // BFS: layer the free agents; label levels for the DFS phase.
        s.queue.clear();
        const INF: u32 = u32::MAX;
        for a in 0..n_agents {
            if s.match_agent[a].is_none() {
                s.dist[a] = 0;
                s.queue.push(a);
            } else {
                s.dist[a] = INF;
            }
        }
        let mut found_free_thought = false;
        let mut head = 0usize;
        while head < s.queue.len() {
            let a = s.queue[head];
            head += 1;
            for (t, &carried) in neighbors(a).iter().enumerate() {
                if !carried {
                    continue;
                }
                match s.match_thought[t] {
                    None => found_free_thought = true,
                    Some(a2) => {
                        if s.dist[a2] == INF {
                            s.dist[a2] = s.dist[a] + 1;
                            s.queue.push(a2);
                        }
                    }
                }
            }
        }
        if !found_free_thought {
            break;
        }

        for a in 0..n_agents {
            if s.match_agent[a].is_none() && dfs(a, mask, n_thoughts, s) {
                matching += 1;
            }
        }
    }
    matching
}

/// Allocating convenience for [`hall_max_matching_into`].
#[must_use]
pub fn hall_max_matching(mask: &[bool], n_agents: usize, n_thoughts: usize) -> usize {
    let mut scratch = HopcroftKarpScratch::with_capacity(n_agents, n_thoughts);
    hall_max_matching_into(mask, n_agents, n_thoughts, &mut scratch)
}

// ── audit ───────────────────────────────────────────────────────────────

/// Static audit of an incidence mask (T3 guard substrate).
///
/// POD summary: support/α distribution + the density alert. `dense` is a
/// WARNING (a dense mask is the crowd-panic precondition), never a gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskAudit {
    /// Agents (mask rows).
    pub n_agents: usize,
    /// Thoughts (mask columns).
    pub n_thoughts: usize,
    /// Total carried (thought, agent) incidences.
    pub ones: usize,
    /// `ones / (n_agents * n_thoughts)`; 0.0 for an empty mask.
    pub density: f32,
    /// Smallest per-thought α (0 for an empty mask).
    pub min_alpha: u32,
    /// Largest per-thought α.
    pub max_alpha: u32,
    /// Mean per-thought α (`ones / n_thoughts`; 0.0 for an empty mask).
    pub mean_alpha: f32,
    /// Thoughts carried by ≥ 2 agents (shared).
    pub shared_count: usize,
    /// Thoughts carried by exactly 1 agent (private).
    pub private_count: usize,
    /// Density alert — `density >= [`DENSITY_ALERT`]`.
    pub dense: bool,
}

/// Audit an incidence mask (zero-alloc; see [`MaskAudit`]).
#[must_use]
pub fn audit_mask(mask: &[bool], n_agents: usize, n_thoughts: usize) -> MaskAudit {
    assert_eq!(
        mask.len(),
        n_agents * n_thoughts,
        "incidence mask layout: mask.len() == n_agents * n_thoughts"
    );
    let ones = mask.iter().filter(|&&carried| carried).count();
    let mut audit = MaskAudit {
        n_agents,
        n_thoughts,
        ones,
        density: 0.0,
        min_alpha: 0,
        max_alpha: 0,
        mean_alpha: 0.0,
        shared_count: 0,
        private_count: 0,
        dense: false,
    };
    if n_agents == 0 || n_thoughts == 0 {
        return audit;
    }
    let mut alpha = vec![0u32; n_thoughts];
    agreement_counts_into(mask, n_agents, n_thoughts, &mut alpha);
    audit.min_alpha = alpha.iter().copied().min().unwrap_or(0);
    audit.max_alpha = alpha.iter().copied().max().unwrap_or(0);
    audit.mean_alpha = ones as f32 / n_thoughts as f32;
    for &a in &alpha {
        if a >= 2 {
            audit.shared_count += 1;
        } else if a == 1 {
            audit.private_count += 1;
        }
    }
    audit.density = ones as f32 / (n_agents * n_thoughts) as f32;
    audit.dense = audit.density >= DENSITY_ALERT;
    audit
}

// ── tier ordering ───────────────────────────────────────────────────────

/// Rank thought indices by agreement: α descending, index ascending on
/// ties (deterministic — the Issue-849 lesson: a total order before any
/// truncation).
///
/// Allocating convenience; see [`rank_by_agreement_into`] for the
/// zero-alloc form.
/// Allocating convenience for [`rank_by_agreement_into`].
#[must_use]
pub fn rank_by_agreement(alpha: &[u32]) -> Vec<usize> {
    let mut order = vec![0usize; alpha.len()];
    let mut scratch = Vec::new();
    rank_by_agreement_into(alpha, &mut scratch, &mut order);
    order
}

/// Zero-alloc variant of [`rank_by_agreement`]: sorts into `sort_scratch`
/// (cleared, reused) and writes the ordered thought indices to `out`.
/// Writes only the first `alpha.len()` entries of `out`.
///
/// # Panics
///
/// Panics if `out.len() < alpha.len()`.
pub fn rank_by_agreement_into(
    alpha: &[u32],
    sort_scratch: &mut Vec<(u32, usize)>,
    out: &mut [usize],
) {
    assert!(
        out.len() >= alpha.len(),
        "rank_by_agreement_into: out must hold alpha.len() entries"
    );
    sort_scratch.clear();
    sort_scratch.extend(alpha.iter().copied().enumerate().map(|(i, a)| (a, i)));
    sort_scratch.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for (dst, (_, i)) in sort_scratch.iter().enumerate() {
        out[dst] = *i;
    }
}

// ── tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planted_mask_counts_exact() {
        // a0: [T T F F]  a1: [T F T F]  a2: [F T T F]
        let mask = [true, true, false, false, true, false, true, false, false, true, true, false];
        let mut alpha = [0u32; 4];
        agreement_counts_into(&mask, 3, 4, &mut alpha);
        assert_eq!(alpha, [2, 2, 2, 0]);
        let mut sizes = [0u32; 3];
        support_sizes_into(&mask, 3, 4, &mut sizes);
        assert_eq!(sizes, [2, 2, 2]);
    }

    #[test]
    fn all_shared_and_all_private_nonvacuity_controls() {
        // All-shared: every agent carries every thought → α = N everywhere.
        let n_agents = 5;
        let n_thoughts = 3;
        let mask = vec![true; n_agents * n_thoughts];
        let mut alpha = [0u32; 3];
        agreement_counts_into(&mask, n_agents, n_thoughts, &mut alpha);
        assert_eq!(alpha, [5, 5, 5]);
        let (shared, private) = shared_private_counts(&mask[..n_thoughts], &alpha);
        assert_eq!((shared, private), (3, 0));

        // All-private: a diagonal mask → α = 1 everywhere.
        let mut diag = vec![false; 4 * 4];
        for k in 0..4 {
            diag[k * 4 + k] = true;
        }
        let mut alpha = [0u32; 4];
        agreement_counts_into(&diag, 4, 4, &mut alpha);
        assert_eq!(alpha, [1, 1, 1, 1]);
        let (shared, private) = shared_private_counts(&diag[..4], &alpha);
        assert_eq!((shared, private), (0, 1));
    }

    #[test]
    fn degeneration_contracts_are_bit_exact() {
        for kappa in [0.0f32, 0.25, 1.0, 4.0, 100.0] {
            assert_eq!(
                routing_weight(1, kappa).to_bits(),
                1.0f32.to_bits(),
                "routing_weight(1) must be exactly 1.0 at κ={kappa}"
            );
            assert_eq!(
                contagion_strength(1, kappa).to_bits(),
                0.0f32.to_bits(),
                "contagion_strength(1) must be exactly 0.0 at κ={kappa}"
            );
        }
    }

    #[test]
    fn tier_ladders_monotone_and_saturating() {
        for kappa in [0.0f32, 0.5, 1.0, 4.0, 100.0] {
            let mut prev_r = routing_weight(0, kappa);
            let mut prev_c = contagion_strength(0, kappa);
            for a in 1..=64u32 {
                let r = routing_weight(a, kappa);
                let c = contagion_strength(a, kappa);
                assert!(r >= prev_r, "routing_weight not monotone at α={a} κ={kappa}");
                assert!(c >= prev_c, "contagion_strength not monotone at α={a} κ={kappa}");
                assert!((0.0..=1.0).contains(&c), "contagion out of [0,1] at α={a}");
                assert!((1.0..=1.5).contains(&r), "routing weight out of [1,1.5] at α={a}");
                prev_r = r;
                prev_c = c;
            }
        }
    }

    #[test]
    fn kappa_zero_is_the_kill_switch() {
        for a in 0..=10u32 {
            assert_eq!(contagion_strength(a, 0.0).to_bits(), 0.0f32.to_bits());
            assert_eq!(routing_weight(a, 0.0).to_bits(), 1.0f32.to_bits());
        }
    }

    #[test]
    fn private_retention_separates_collapse_from_diverse() {
        // Collapse: 3 agents all carry ONLY the same thought → frac 0.
        let collapse = [true, false, true, false, true, false];
        let mut fracs = [0.0f32; 3];
        let mut scratch = [0u32; 2];
        private_fractions_into(&collapse, 3, 2, &mut scratch, &mut fracs);
        assert_eq!(fracs, [0.0, 0.0, 0.0]);

        // Diverse: diagonal → every supported thought is private → frac 1.
        let diverse = [true, false, false, true, false, false];
        private_fractions_into(&diverse, 3, 2, &mut scratch, &mut fracs);
        assert_eq!(fracs, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn hall_matches_brute_force_and_catches_violation() {
        // Planted violation: 3 agents all support ONLY thought 0 → match 1.
        let violation = [true, false, true, false, true, false];
        assert_eq!(hall_max_matching(&violation, 3, 2), 1);

        // Identity: every agent its own thought → feasible.
        let mut identity = vec![false; 4 * 4];
        for k in 0..4 {
            identity[k * 4 + k] = true;
        }
        assert_eq!(hall_max_matching(&identity, 4, 4), 4);

        // Randomized cross-check vs exhaustive backtracking.
        let mut rng = Xorshift(0x9E37_79B9_7F4A_7C15);
        for _ in 0..300 {
            let n_agents = 1 + (rng.next() as usize % 6);
            let n_thoughts = 1 + (rng.next() as usize % 6);
            let mask: Vec<bool> = (0..n_agents * n_thoughts)
                .map(|_| rng.next() % 100 < 35)
                .collect();
            assert_eq!(
                hall_max_matching(&mask, n_agents, n_thoughts),
                brute_force_max_matching(&mask, n_agents, n_thoughts),
                "Hopcroft–Karp disagrees with brute force (n_agents={n_agents}, n_thoughts={n_thoughts})"
            );
        }
    }

    #[test]
    fn audit_flags_dense_and_passes_sparse() {
        let n_agents = 4;
        let n_thoughts = 4;
        let dense = vec![true; n_agents * n_thoughts];
        let a = audit_mask(&dense, n_agents, n_thoughts);
        assert!(a.dense, "all-shared mask must trip the density alert");
        assert_eq!((a.min_alpha, a.max_alpha), (4, 4));
        assert_eq!(a.shared_count, 4);
        assert_eq!(a.private_count, 0);

        let mut sparse = vec![false; n_agents * n_thoughts];
        for k in 0..n_agents {
            sparse[k * n_thoughts + k] = true;
        }
        let a = audit_mask(&sparse, n_agents, n_thoughts);
        assert!(!a.dense);
        assert_eq!((a.min_alpha, a.max_alpha), (1, 1));
        assert_eq!(a.private_count, 4);

        // Empty mask is not dense.
        let empty = vec![false; 6];
        assert!(!audit_mask(&empty, 3, 2).dense);
    }

    #[test]
    fn permutation_invariance_property() {
        let mut rng = Xorshift(0xDEAD_BEEF_CAFE_F00D);
        for case in 0..40 {
            let n_agents = 2 + (rng.next() as usize % 8);
            let n_thoughts = 2 + (rng.next() as usize % 8);
            let mask: Vec<bool> = (0..n_agents * n_thoughts)
                .map(|_| rng.next() % 100 < 40)
                .collect();

            let mut alpha0 = vec![0u32; n_thoughts];
            agreement_counts_into(&mask, n_agents, n_thoughts, &mut alpha0);
            let mut sizes0 = vec![0u32; n_agents];
            support_sizes_into(&mask, n_agents, n_thoughts, &mut sizes0);
            let mut fracs0 = vec![0.0f32; n_agents];
            let mut scratch = vec![0u32; n_thoughts];
            private_fractions_into(&mask, n_agents, n_thoughts, &mut scratch, &mut fracs0);
            let audit0 = audit_mask(&mask, n_agents, n_thoughts);

            // Permute agents (Fisher–Yates with the same seeded rng stream).
            let mut perm: Vec<usize> = (0..n_agents).collect();
            for i in (1..n_agents).rev() {
                let j = (rng.next() as usize) % (i + 1);
                perm.swap(i, j);
            }
            let mut permuted = vec![false; n_agents * n_thoughts];
            for (dst, &src) in perm.iter().enumerate() {
                let (d, s) = (dst * n_thoughts, src * n_thoughts);
                permuted[d..d + n_thoughts].copy_from_slice(&mask[s..s + n_thoughts]);
            }

            let mut alpha1 = vec![0u32; n_thoughts];
            agreement_counts_into(&permuted, n_agents, n_thoughts, &mut alpha1);
            assert_eq!(alpha0, alpha1, "case {case}: α per thought must be agent-permutation invariant");

            let mut sizes1 = vec![0u32; n_agents];
            support_sizes_into(&permuted, n_agents, n_thoughts, &mut sizes1);
            let mut fracs1 = vec![0.0f32; n_agents];
            private_fractions_into(&permuted, n_agents, n_thoughts, &mut scratch, &mut fracs1);
            for dst in 0..n_agents {
                assert_eq!(sizes1[dst], sizes0[perm[dst]], "case {case}: support sizes must be equivariant");
                assert_eq!(
                    fracs1[dst].to_bits(),
                    fracs0[perm[dst]].to_bits(),
                    "case {case}: private fractions must be equivariant"
                );
            }
            assert_eq!(
                audit_mask(&permuted, n_agents, n_thoughts),
                audit0,
                "case {case}: mask audit must be permutation invariant"
            );
        }
    }

    #[test]
    fn tier_ordering_is_deterministic_total_order() {
        let alpha = [2u32, 5, 2, 0, 5];
        let order = rank_by_agreement(&alpha);
        // α desc, index asc on ties.
        assert_eq!(order, vec![1, 4, 0, 2, 3]);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn hot_path_is_alloc_free() {
        crate::alloc::reset_alloc_stats();
        let n_agents = 32;
        let n_thoughts = 16;
        let mut mask = vec![false; n_agents * n_thoughts];
        let mut rng = Xorshift(42);
        for cell in mask.iter_mut() {
            *cell = rng.next() % 100 < 30;
        }
        let mut counts = vec![0u32; n_thoughts];
        let mut sizes = vec![0u32; n_agents];
        let mut fracs = vec![0.0f32; n_agents];
        let mut alpha_scratch = vec![0u32; n_thoughts];
        let mut order = vec![0usize; n_thoughts];
        let mut order_scratch: Vec<(u32, usize)> = Vec::new();
        let mut hk = HopcroftKarpScratch::with_capacity(n_agents, n_thoughts);
        // Warm-up pass: scratch growth is a first-call cost, not hot-path
        // traffic (the steady-state discipline — measure AFTER warm-up).
        agreement_counts_into(&mask, n_agents, n_thoughts, &mut counts);
        rank_by_agreement_into(&counts, &mut order_scratch, &mut order);
        let _ = hall_max_matching_into(&mask, n_agents, n_thoughts, &mut hk);
        crate::alloc::reset_alloc_stats();
        for _ in 0..100 {
            agreement_counts_into(&mask, n_agents, n_thoughts, &mut counts);
            support_sizes_into(&mask, n_agents, n_thoughts, &mut sizes);
            private_fractions_into(&mask, n_agents, n_thoughts, &mut alpha_scratch, &mut fracs);
            rank_by_agreement_into(&counts, &mut order_scratch, &mut order);
            let _ = hall_max_matching_into(&mask, n_agents, n_thoughts, &mut hk);
            let _ = contagion_strength(7, 4.0);
            let _ = routing_weight(7, 4.0);
        }
        let (allocs, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(allocs, 0, "incidence hot path must be alloc-free");
    }

    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    fn brute_force_max_matching(mask: &[bool], n_agents: usize, n_thoughts: usize) -> usize {
        fn rec(mask: &[bool], n_thoughts: usize, agent: usize, n_agents: usize, used: &mut [bool]) -> usize {
            if agent == n_agents {
                return 0;
            }
            let mut best = rec(mask, n_thoughts, agent + 1, n_agents, used);
            let row = &mask[agent * n_thoughts..(agent + 1) * n_thoughts];
            for (t, &carried) in row.iter().enumerate() {
                if carried && !used[t] {
                    used[t] = true;
                    best = best.max(1 + rec(mask, n_thoughts, agent + 1, n_agents, used));
                    used[t] = false;
                }
            }
            best
        }
        if n_agents == 0 || n_thoughts == 0 {
            return 0;
        }
        let mut used = vec![false; n_thoughts];
        rec(mask, n_thoughts, 0, n_agents, &mut used)
    }
}
