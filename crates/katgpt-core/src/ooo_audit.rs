//! Direction-bank audit — asymmetric Odd-One-Out interpretability,
//! Cross-OOO diversity, and greedy curation for representation banks.
//!
//! Issue 759 / Research 552, distilled from Issa, Liu, Ballé, Klindt
//! "High-dimensional population codes reveal interpretable and diverse
//! features underlying visual perception" (bioRxiv 2026.09.05.748439).
//!
//! # The audit stage of the direction ecosystem
//!
//! The stack acquires direction vectors (MAG mines them unsupervised,
//! EmotionDirections extracts them supervised), injects them (Latent Field
//! Steering / PersonalityWeightedComposition / CommittedFieldBlend), and
//! commits them (freeze/thaw, `MerkleFrozenEnvelope`) — but nothing verifies
//! a bank member is *self-consistent* (interpretable) or *non-redundant*
//! against other members. Because blends SUM sigmoid-gated directions,
//! duplicated concepts double-count effective weight — a correctness bug
//! class, not just waste. This module is that missing audit stage:
//!
//! 1. [`ooo_score`] — **asymmetric Odd-One-Out interpretability**: a unit's
//!    K maximally-activating exemplars (MEIs) define a threshold (their mean
//!    pairwise similarity); the score is the fraction of all OTHER exemplars
//!    whose mean similarity to the MEIs falls *below* it (paper Eq. 6–8).
//!    Asymmetric by design — fair to sparse/non-negative activations that a
//!    symmetric 2-AFC metric penalizes.
//! 2. [`cross_ooo`] — **Cross-OOO diversity**: for a pair of units, the
//!    fraction of their 2K MEIs correctly attributed to the unit of origin
//!    (own-MSI > other-MSI, paper Eq. 9–11). 0.5 = chance = redundant
//!    preferences. This is the redundancy detector direction banks never
//!    had.
//! 3. [`greedy_curate`] — paper Fig. 4a: sort by OOO descending, drop
//!    `OOO < ooo_cutoff`, greedily keep a unit only if its Cross-OOO with
//!    EVERY kept member is `>= cross_cutoff`. The kept-list length is the
//!    bank's **unique-feature count** — a measured distinct-content capacity
//!    usable as a codebook-K saturation signal (fit more centroids than
//!    clusters; the curated count stays at the true cluster count).
//!
//! # Modelless contract
//!
//! Pure f32 matrix reductions over an activation matrix `A ∈ R^{U×N}` and a
//! PLUGGABLE exemplar-similarity matrix `S ∈ R^{N×N}` (the paper uses
//! DreamSim; consumers bring their own metric — [`exemplar_rbf_sim_into`]
//! ships one generic kernel). No training, no gradients, no RNG, no
//! HashMap — deterministic by construction.
//!
//! # Allocation discipline
//!
//! Offline gate (freeze-time / bench-time), but the same discipline as the
//! hot paths: every `*_into` entry point writes into caller-owned output
//! and scratch buffers; after the first call (which grows [`BankAudit`]'s
//! per-unit MEI vectors) steady-state auditing is allocation-free. Verify
//! with the G4 counting-allocator gate (bench_759).
//!
//! # Similarity-matrix conventions
//!
//! `sim` is row-major `N × N`, symmetric, with `sim[i*N + i] == 1.0`.
//! Higher = more similar (DreamSim convention). [`cross_ooo`] skips
//! diagonal self-pairs so shared MEIs between two units degrade the score
//! (as they should) instead of inflating it via `sim(i, i) = 1`.

use std::cmp::Ordering as CmpOrdering;

/// Audit parameters. Paper defaults: both cutoffs 0.8 (visual confirmation
/// that greedy-list MEIs are readily interpretable and diverse).
#[derive(Clone, Copy, Debug)]
pub struct OooAuditConfig {
    /// K — number of maximally-activating exemplars per unit. Must be ≥ 2
    /// (K = 1 leaves the OOO threshold and Cross-OOO own-similarity
    /// undefined).
    pub top_k: usize,
    /// Interpretability cutoff: units below are `dropped_uninterpretable`.
    pub ooo_cutoff: f32,
    /// Diversity cutoff: a unit whose Cross-OOO with ANY kept member falls
    /// below this is `pruned_redundant`.
    pub cross_cutoff: f32,
}

impl Default for OooAuditConfig {
    fn default() -> Self {
        Self {
            top_k: 8,
            ooo_cutoff: 0.8,
            cross_cutoff: 0.8,
        }
    }
}

/// Transient buffers shared across the audit pipeline. Allocate once per
/// consumer, reuse across audits (`reserve_for` grows as needed).
#[derive(Debug, Default)]
pub struct AuditScratch {
    mei_buf: Vec<u32>,
    order: Vec<u32>,
}

impl AuditScratch {
    /// Empty scratch; call [`Self::reserve_for`] before the first audit.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure capacity for a `U × N` audit at top-K (grow-only). Compares
    /// against CAPACITY, not `len` — the buffers keep their previous run's
    /// contents between calls, and `Vec::reserve` sizes for `len + additional`,
    /// so a len-blind call would double the capacity exactly once per buffer
    /// (caught by the bench_759 G4 gate as 2 steady-state allocs).
    pub fn reserve_for(&mut self, unit_count: usize, top_k: usize) {
        let k = top_k.max(1);
        if self.mei_buf.capacity() < k {
            self.mei_buf.clear();
            self.mei_buf.reserve(k);
        }
        let u = unit_count.max(1);
        if self.order.capacity() < u {
            self.order.clear();
            self.order.reserve(u);
        }
    }
}

/// Full audit output for a `U × N` bank. Allocate once per consumer; pass
/// the same struct to every [`audit_bank_into`] call (clears + rewrites).
#[derive(Clone, Debug)]
pub struct BankAudit {
    /// Number of audited units.
    pub unit_count: usize,
    /// Number of exemplars.
    pub exemplar_count: usize,
    /// K used for this audit.
    pub top_k: usize,
    /// Per-unit asymmetric Odd-One-Out score `[U]`.
    pub ooo: Vec<f32>,
    /// Per-unit MEI indices `[U][K]`, activation-descending.
    pub meis: Vec<Vec<u32>>,
    /// Symmetric Cross-OOO matrix, row-major `[U*U]`, diagonal = 1.0.
    pub cross: Vec<f32>,
}

impl BankAudit {
    /// Allocate output buffers for a `U × N` audit at top-K.
    pub fn new(unit_count: usize, exemplar_count: usize, top_k: usize) -> Self {
        Self {
            unit_count,
            exemplar_count,
            top_k,
            ooo: vec![0.0; unit_count],
            meis: vec![Vec::with_capacity(top_k); unit_count],
            cross: vec![0.0; unit_count * unit_count],
        }
    }
}

/// Greedy curation verdict (paper Fig. 4a).
#[derive(Clone, Debug)]
pub struct CurationResult {
    /// Kept units, OOO-descending order (ties → lower unit index first).
    pub kept: Vec<usize>,
    /// Interpretable units rejected for redundancy vs a kept member.
    pub pruned_redundant: Vec<usize>,
    /// Units below `ooo_cutoff`.
    pub dropped_uninterpretable: Vec<usize>,
}

impl CurationResult {
    /// Allocate for a `U`-unit bank.
    pub fn new(unit_count: usize) -> Self {
        Self {
            kept: Vec::with_capacity(unit_count),
            pruned_redundant: Vec::with_capacity(unit_count),
            dropped_uninterpretable: Vec::with_capacity(unit_count),
        }
    }

    /// The bank's unique-feature count — the length of the curated list
    /// (paper Fig. 4: this is what scales past the neuron count under
    /// superposition).
    pub fn unique_feature_count(&self) -> usize {
        self.kept.len()
    }
}

/// Deterministic top-K selection: descending activation, ties → lower
/// exemplar index (bit-identical orderings across runs and platforms).
///
/// O(N·K) insertion into `out` (cleared first); no allocation after the
/// first call.
pub fn select_meis_into(activations: &[f32], top_k: usize, out: &mut Vec<u32>) {
    debug_assert!(top_k >= 2, "top_k must be >= 2 (threshold undefined at K=1)");
    debug_assert!(activations.len() >= top_k, "need at least top_k exemplars");
    out.clear();
    for (idx, &act) in activations.iter().enumerate() {
        if out.len() < top_k {
            out.push(idx as u32);
            // Restore descending order after push (single backward pass;
            // out.len() <= K so this is bounded).
            let mut i = out.len() - 1;
            while i > 0 {
                let a = activations[out[i - 1] as usize];
                let b = activations[out[i] as usize];
                // Strictly-greater keeps stable (lower index first on ties).
                if a < b {
                    out.swap(i - 1, i);
                } else {
                    break;
                }
                i -= 1;
            }
        } else {
            let worst = *out.last().unwrap() as usize;
            let act_worst = activations[worst];
            if act > act_worst {
                *out.last_mut().unwrap() = idx as u32;
                let mut i = out.len() - 1;
                while i > 0 {
                    let a = activations[out[i - 1] as usize];
                    let b = activations[out[i] as usize];
                    if a < b {
                        out.swap(i - 1, i);
                    } else {
                        break;
                    }
                    i -= 1;
                }
            }
        }
    }
}

/// Asymmetric Odd-One-Out interpretability score (paper Eq. 6–8).
///
/// `meis` = the unit's K maximally-activating exemplar indices,
/// `sim` = row-major `N × N` exemplar similarity, higher = more similar.
/// Threshold = mean pairwise similarity among the MEIs; score = fraction of
/// the `N − K` remaining exemplars whose mean similarity to the MEIs is
/// strictly below it. Returns 1.0 when every intruder is below threshold,
/// 0.0 when none are.
pub fn ooo_score(meis: &[u32], sim: &[f32], exemplar_count: usize) -> f32 {
    debug_assert!(meis.len() >= 2, "K >= 2 required (threshold undefined at K=1)");
    let k = meis.len();
    let n = exemplar_count;
    debug_assert!(k < n, "intruder set is empty when K == N");

    // Threshold: mean over unordered MEI pairs (Eq. 6).
    let mut pair_sum = 0.0f32;
    for j in 0..k {
        for l in (j + 1)..k {
            pair_sum += sim[meis[j] as usize * n + meis[l] as usize];
        }
    }
    let threshold = pair_sum / (k * (k - 1) / 2) as f32;

    // Intruder fraction (Eq. 7–8): strict < below the unit's own coherence.
    let mut points = 0u32;
    let mut intruders = 0u32;
    for i in 0..n {
        if meis.iter().any(|&m| m as usize == i) {
            continue;
        }
        let mut s = 0.0f32;
        for &m in meis {
            s += sim[i * n + m as usize];
        }
        if s / (k as f32) < threshold {
            points += 1;
        }
        intruders += 1;
    }
    points as f32 / intruders as f32
}

/// Cross-OOO diversity for a unit pair (paper Eq. 9–11).
///
/// For each MEI of either unit: a point when its mean similarity to its own
/// unit's OTHER MEIs is strictly greater than its mean similarity to the
/// other unit's MEIs. Returns the fraction over the `2K` MEIs; 1.0 =
/// fully distinguishable preferences, ~0.5 = redundant, 0.0 = fully
/// entangled (or identical MEI sets — every comparison ties).
///
/// Identical indices appearing in both sets are skipped on the "other" side
/// so a shared exemplar's `sim(i, i) = 1.0` cannot inflate the score; a
/// shared MEI still contributes no point (own == other for it), which is
/// the correct redundancy signal.
pub fn cross_ooo(meis_a: &[u32], meis_b: &[u32], sim: &[f32], exemplar_count: usize) -> f32 {
    debug_assert!(meis_a.len() >= 2, "K >= 2 required (own-similarity undefined at K=1)");
    debug_assert!(meis_b.len() >= 2, "K >= 2 required");
    let n = exemplar_count;
    let ka = meis_a.len();
    let kb = meis_b.len();

    let mut points = 0u32;
    let mut total = 0u32;

    // Unit A's MEIs.
    for (i, &m) in meis_a.iter().enumerate() {
        let mut own = 0.0f32;
        for (j, &o) in meis_a.iter().enumerate() {
            if j != i {
                own += sim[m as usize * n + o as usize];
            }
        }
        own /= (ka - 1) as f32;
        let mut other = 0.0f32;
        let mut cnt = 0u32;
        for &o in meis_b {
            if o != m {
                other += sim[m as usize * n + o as usize];
                cnt += 1;
            }
        }
        // cnt >= 1 guaranteed: K >= 2 and MEI indices are distinct within a
        // set, so at most one of B's MEIs equals m.
        debug_assert!(cnt >= 1);
        other /= cnt as f32;
        if own > other {
            points += 1;
        }
        total += 1;
    }

    // Unit B's MEIs (symmetric arm).
    for (i, &m) in meis_b.iter().enumerate() {
        let mut own = 0.0f32;
        for (j, &o) in meis_b.iter().enumerate() {
            if j != i {
                own += sim[m as usize * n + o as usize];
            }
        }
        own /= (kb - 1) as f32;
        let mut other = 0.0f32;
        let mut cnt = 0u32;
        for &o in meis_a {
            if o != m {
                other += sim[m as usize * n + o as usize];
                cnt += 1;
            }
        }
        debug_assert!(cnt >= 1);
        other /= cnt as f32;
        if own > other {
            points += 1;
        }
        total += 1;
    }

    points as f32 / total as f32
}

/// Full bank audit: per-unit MEIs + OOO scores, then the symmetric Cross-OOO
/// matrix. `a` is row-major `U × N` (unit-major), `sim` row-major `N × N`.
///
/// Zero-allocation steady-state (caller-owned `scratch` + `out`; the first
/// call grows `out.meis` inner vectors, subsequent calls reuse capacity).
pub fn audit_bank_into(
    a: &[f32],
    sim: &[f32],
    cfg: &OooAuditConfig,
    scratch: &mut AuditScratch,
    out: &mut BankAudit,
) {
    let u = out.unit_count;
    let n = out.exemplar_count;
    debug_assert_eq!(a.len(), u * n, "activation matrix must be U×N");
    debug_assert_eq!(sim.len(), n * n, "similarity matrix must be N×N");
    debug_assert!(cfg.top_k >= 2 && cfg.top_k < n, "2 <= K < N required");
    scratch.reserve_for(u, cfg.top_k);
    out.top_k = cfg.top_k;

    for unit in 0..u {
        let row = &a[unit * n..unit * n + n];
        select_meis_into(row, cfg.top_k, &mut scratch.mei_buf);
        out.meis[unit].clear();
        out.meis[unit].extend_from_slice(&scratch.mei_buf);
        out.ooo[unit] = ooo_score(&scratch.mei_buf, sim, n);
    }

    for i in 0..u {
        out.cross[i * u + i] = 1.0;
        for j in (i + 1)..u {
            let c = cross_ooo(&out.meis[i], &out.meis[j], sim, n);
            out.cross[i * u + j] = c;
            out.cross[j * u + i] = c;
        }
    }
}

/// Greedy curation (paper Fig. 4a): OOO-descending order (ties → lower unit
/// index), drop `OOO < ooo_cutoff`, keep only units whose Cross-OOO with
/// every kept member is `>= cross_cutoff`. Zero-allocation steady-state.
pub fn greedy_curate(
    audit: &BankAudit,
    cfg: &OooAuditConfig,
    scratch: &mut AuditScratch,
    out: &mut CurationResult,
) {
    let u = audit.unit_count;
    scratch.order.clear();
    for idx in 0..u {
        scratch.order.push(idx as u32);
    }
    // Total order: OOO descending, then unit index ascending — deterministic.
    scratch.order.sort_by(|&x, &y| {
        let ox = audit.ooo[x as usize];
        let oy = audit.ooo[y as usize];
        match oy.partial_cmp(&ox) {
            Some(CmpOrdering::Less) => CmpOrdering::Less,
            Some(CmpOrdering::Greater) => CmpOrdering::Greater,
            _ => x.cmp(&y),
        }
    });

    out.kept.clear();
    out.pruned_redundant.clear();
    out.dropped_uninterpretable.clear();

    for &unit_idx in &scratch.order {
        let unit = unit_idx as usize;
        if audit.ooo[unit] < cfg.ooo_cutoff {
            out.dropped_uninterpretable.push(unit);
            continue;
        }
        let redundant = out
            .kept
            .iter()
            .any(|&k| matches!(audit.cross[unit * u + k].partial_cmp(&cfg.cross_cutoff), Some(CmpOrdering::Less)));
        if redundant {
            out.pruned_redundant.push(unit);
        } else {
            out.kept.push(unit);
        }
    }
}

/// Generic exemplar similarity kernel: RBF over row-major `N × D` exemplar
/// vectors, `sim(i, j) = exp(-γ·‖x_i − x_j‖²)`, written symmetric into
/// `out` (`N × N`, diagonal set to exactly 1.0). The paper's DreamSim is a
/// consumer-supplied alternative; this kernel is for synthetic fixtures and
/// latent-space exemplars.
pub fn exemplar_rbf_sim_into(x: &[f32], dim: usize, gamma: f32, out: &mut [f32]) {
    let n = x.len() / dim;
    debug_assert_eq!(x.len(), n * dim);
    debug_assert_eq!(out.len(), n * n);
    for i in 0..n {
        out[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let mut acc = 0.0f32;
            let xi = &x[i * dim..i * dim + dim];
            let xj = &x[j * dim..j * dim + dim];
            // Chunked delta-square sum (LLVM auto-vectorizes 4-wide).
            let d = dim;
            let mut c = 0;
            while c + 4 <= d {
                let d0 = xi[c] - xj[c];
                let d1 = xi[c + 1] - xj[c + 1];
                let d2 = xi[c + 2] - xj[c + 2];
                let d3 = xi[c + 3] - xj[c + 3];
                acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                c += 4;
            }
            while c < d {
                let dd = xi[c] - xj[c];
                acc += dd * dd;
                c += 1;
            }
            let s = (-gamma * acc).exp();
            out[i * n + j] = s;
            out[j * n + i] = s;
        }
    }
}

/// Deterministic planted-cluster fixtures for audit gates (tests + benches).
///
/// Layout contract: `G` true units `0..G` (one per cluster, activation =
/// dot(exemplar, cluster center)), `dup` near-duplicate units `G..G+dup`
/// (true-unit activations + ±`perturbation` noise — the redundancy plants),
/// `noise_units` structureless units at the tail (uniform activations — the
/// uninterpretable plants). Returns `(exemplars N×D row-major, activations
/// U×N row-major)`.
pub mod fixtures {
    /// SplitMix64 — same shape as `factorized_action::codebook`'s private
    /// PRNG (kept local: fixtures must not depend on another feature).
    #[derive(Clone, Copy)]
    pub struct FixtureRng {
        state: u64,
    }

    impl FixtureRng {
        pub fn new(seed: u64) -> Self {
            Self {
                state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            }
        }

        pub fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform `[-1, 1)`.
        pub fn next_sym_f32(&mut self) -> f32 {
            ((self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)) * 2.0 - 1.0
        }
    }

    /// Planted bank: `g` clusters of `m` exemplars in `d` dims. Cluster `q`
    /// owns coordinates `q*2..q*2+2` (value `4.0`); requires `d >= 2*g`.
    /// Exemplars = center + uniform noise scaled by `0.25`. Duplicate units
    /// copy unit `(i - g) % g`'s activations + `±perturbation`.
    pub fn planted_cluster_bank(
        g: usize,
        m: usize,
        d: usize,
        dup: usize,
        noise_units: usize,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        assert!(d >= 2 * g, "d >= 2*g required (each cluster owns 2 coords)");
        assert!(m >= 8, "m >= 8 exemplars per cluster required (K=8 MEIs)");
        let n = g * m;
        let u = g + dup + noise_units;
        let mut rng = FixtureRng::new(seed);

        // Centers: coordinate pair per cluster.
        let mut centers = vec![0.0f32; g * d];
        for q in 0..g {
            centers[q * d + q * 2] = 4.0;
            centers[q * d + q * 2 + 1] = 4.0;
        }

        // Exemplars: center + noise.
        let mut x = vec![0.0f32; n * d];
        for i in 0..n {
            let q = i / m;
            for c in 0..d {
                x[i * d + c] = centers[q * d + c] + 0.25 * rng.next_sym_f32();
            }
        }

        // Activations: true units = dot(exemplar, center).
        let mut a = vec![0.0f32; u * n];
        for unit in 0..g {
            for i in 0..n {
                let mut acc = 0.0f32;
                for c in 0..d {
                    acc += x[i * d + c] * centers[unit * d + c];
                }
                a[unit * n + i] = acc;
            }
        }
        // Duplicates: copy a true unit's row + perturbation.
        for rep in 0..dup {
            let src = rep % g;
            for i in 0..n {
                a[(g + rep) * n + i] = a[src * n + i] + 1e-3 * rng.next_sym_f32();
            }
        }
        // Noise units: uniform activations, no cluster structure.
        for unit_idx in g + dup..u {
            for i in 0..n {
                a[unit_idx * n + i] = rng.next_sym_f32();
            }
        }
        (x, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a symmetric 5×5 sim fixture from the upper-triangle dict.
    fn sim5(pairs: &[((usize, usize), f32)]) -> Vec<f32> {
        let mut s = vec![1.0f32; 25];
        for &((i, j), v) in pairs {
            s[i * 5 + j] = v;
            s[j * 5 + i] = v;
        }
        s
    }

    #[test]
    fn t1_ooo_score_hand_fixture() {
        // N=5, K=2, MEIs {0,1} (activation-descending).
        // threshold = sim(0,1) = 0.875.
        // intruder 2: (0.25 + 0.5)/2 = 0.375  < 0.875 → point
        // intruder 3: (0.75 + 0.75)/2 = 0.75  < 0.875 → point
        // intruder 4: (0.875 + 1.0)/2 = 0.9375 ≥ 0.875 → no point
        // score = 2/3.
        let sim = sim5(&[((0, 1), 0.875), ((2, 0), 0.25), ((2, 1), 0.5), ((3, 0), 0.75), ((3, 1), 0.75), ((4, 0), 0.875), ((4, 1), 1.0)]);
        let score = ooo_score(&[0, 1], &sim, 5);
        assert_eq!(score, 2.0 / 3.0, "hand-computed intruder fraction 2/3");
    }

    #[test]
    fn t1_select_meis_tie_breaks_to_lower_index() {
        let acts = [1.0, 3.0, 3.0, 2.0];
        let mut meis = Vec::new();
        select_meis_into(&acts, 2, &mut meis);
        assert_eq!(meis, vec![1, 2], "ties resolve to lower exemplar index");

        let acts2 = [0.1, 0.2, 5.0, 0.3, 9.0];
        select_meis_into(&acts2, 3, &mut meis);
        assert_eq!(meis, vec![4, 2, 3], "activation-descending order");
    }

    #[test]
    fn t1_cross_ooo_distinguishable_pair_scores_one() {
        // A = {0,1}, B = {2,3}; within-pair sim 0.875, all cross sims 0.25.
        let sim = sim5(&[
            ((0, 1), 0.875),
            ((2, 3), 0.875),
            ((0, 2), 0.25),
            ((0, 3), 0.25),
            ((1, 2), 0.25),
            ((1, 3), 0.25),
        ]);
        let c = cross_ooo(&[0, 1], &[2, 3], &sim, 5);
        assert_eq!(c, 1.0, "disjoint clusters attribute every MEI correctly");
    }

    #[test]
    fn t1_cross_ooo_half_entangled_pair_scores_half() {
        // A = {0,1}, B = {2,3}. own = 0.9375 for both members of each pair.
        // MEI0 other = (0.75+1.0)/2 = 0.875 → point. MEI1 other = 1.0 → no.
        // MEI2 other = (0.75+1.0)/2 = 0.875 → point. MEI3 other = 1.0 → no.
        // 2/4 = 0.5 — the paper's chance-level redundancy signal.
        let sim = sim5(&[
            ((0, 1), 0.9375),
            ((2, 3), 0.9375),
            ((0, 2), 0.75),
            ((0, 3), 1.0),
            ((1, 2), 1.0),
            ((1, 3), 1.0),
        ]);
        let c = cross_ooo(&[0, 1], &[2, 3], &sim, 5);
        assert_eq!(c, 0.5, "half-attributable preferences = chance level");
    }

    #[test]
    fn t1_cross_ooo_identical_mei_sets_score_zero() {
        // Identical MEI sets: every own/other comparison ties (strict >
        // fails) → 0.0, the maximal redundancy signal. Shared exemplars are
        // skipped on the other side so sim(i,i)=1 never inflates the score.
        let sim = sim5(&[((0, 1), 0.9), ((2, 3), 0.9), ((0, 2), 0.5), ((1, 3), 0.5)]);
        let c = cross_ooo(&[0, 1], &[0, 1], &sim, 5);
        assert_eq!(c, 0.0, "identical preference sets are maximally redundant");
    }

    #[test]
    fn t1_greedy_curate_prunes_duplicate_keeps_distinct() {
        // Hand-built audit: unit0 OOO 0.9, unit1 (duplicate) OOO 0.85 with
        // cross(0,1)=0.5, unit2 OOO 0.82 cross ≥0.95 vs both, unit3 OOO 0.5.
        let u = 4;
        let mut audit = BankAudit::new(u, 10, 2);
        audit.ooo = vec![0.9, 0.85, 0.82, 0.5];
        audit.meis = vec![vec![0, 1], vec![0, 1], vec![2, 3], vec![4, 5]];
        for i in 0..u {
            audit.cross[i * u + i] = 1.0;
        }
        let set = |cross: &mut Vec<f32>, i: usize, j: usize, v: f32| {
            cross[i * u + j] = v;
            cross[j * u + i] = v;
        };
        set(&mut audit.cross, 0, 1, 0.5);
        set(&mut audit.cross, 0, 2, 0.95);
        set(&mut audit.cross, 1, 2, 0.95);
        set(&mut audit.cross, 0, 3, 0.9);
        set(&mut audit.cross, 1, 3, 0.9);
        set(&mut audit.cross, 2, 3, 0.9);

        let cfg = OooAuditConfig {
            top_k: 2,
            ooo_cutoff: 0.8,
            cross_cutoff: 0.8,
        };
        let mut scratch = AuditScratch::new();
        let mut result = CurationResult::new(u);
        greedy_curate(&audit, &cfg, &mut scratch, &mut result);

        assert_eq!(result.kept, vec![0, 2], "OOO-descending, duplicate pruned");
        assert_eq!(result.pruned_redundant, vec![1]);
        assert_eq!(result.dropped_uninterpretable, vec![3]);
        assert_eq!(result.unique_feature_count(), 2);
    }

    #[test]
    fn t1_rbf_sim_hand_value() {
        let x = vec![0.0f32, 0.0, 1.0, 1.0];
        let mut sim = vec![0.0f32; 4];
        exemplar_rbf_sim_into(&x, 2, 1.0, &mut sim);
        assert_eq!(sim[0], 1.0);
        assert_eq!(sim[3], 1.0);
        let expected = (-2.0f32).exp();
        assert!((sim[1] - expected).abs() < 1e-6, "e^-2 = {expected}, got {}", sim[1]);
        assert_eq!(sim[1], sim[2], "symmetric");
    }

    // ── T2: synthetic planted bank (G1 correctness gate) ──────────────────

    fn run_planted_bank(seed: u64) -> (BankAudit, CurationResult) {
        let (x, a) = fixtures::planted_cluster_bank(4, 20, 8, 2, 2, seed);
        let n = 4 * 20;
        let u = 4 + 2 + 2;
        let mut sim = vec![0.0f32; n * n];
        exemplar_rbf_sim_into(&x, 8, 1.0, &mut sim);

        let cfg = OooAuditConfig::default(); // K=8, cutoffs 0.8
        let mut scratch = AuditScratch::new();
        let mut audit = BankAudit::new(u, n, cfg.top_k);
        audit_bank_into(&a, &sim, &cfg, &mut scratch, &mut audit);

        let mut result = CurationResult::new(u);
        greedy_curate(&audit, &cfg, &mut scratch, &mut result);
        (audit, result)
    }

    #[test]
    fn t2_planted_bank_recovers_true_feature_count() {
        // Deterministic across seeds: curation recovers the 4 planted
        // clusters exactly (issue acceptance: ±1; aim exact).
        for seed in [7u64, 42, 2026] {
            let (audit, result) = run_planted_bank(seed);

            assert_eq!(
                result.unique_feature_count(),
                4,
                "seed {seed}: kept exactly the 4 planted clusters"
            );
            // All four true units (0..4) kept, in OOO-descending order.
            let mut kept_sorted = result.kept.clone();
            kept_sorted.sort_unstable();
            assert_eq!(kept_sorted, vec![0, 1, 2, 3], "seed {seed}");
            // Duplicates (units 4,5) pruned as redundant.
            let mut pruned_sorted = result.pruned_redundant.clone();
            pruned_sorted.sort_unstable();
            assert_eq!(pruned_sorted, vec![4, 5], "seed {seed}: duplicates pruned");
            // Noise units (6,7) dropped as uninterpretable.
            let mut dropped_sorted = result.dropped_uninterpretable.clone();
            dropped_sorted.sort_unstable();
            assert_eq!(dropped_sorted, vec![6, 7], "seed {seed}: noise dropped");
            // True units are interpretable under the paper's 0.8 bar.
            for unit in 0..4 {
                assert!(
                    audit.ooo[unit] >= 0.8,
                    "seed {seed}: true unit {unit} OOO = {}",
                    audit.ooo[unit]
                );
            }
        }
    }

    #[test]
    fn t2_kept_order_preserves_ooo_ranking() {
        let (audit, result) = run_planted_bank(7);
        let kept = &result.kept;
        for w in 1..kept.len() {
            let prev = audit.ooo[kept[w - 1]];
            let cur = audit.ooo[kept[w]];
            assert!(
                prev > cur || (prev == cur && kept[w - 1] < kept[w]),
                "curated list must stay OOO-descending (ties → lower index)"
            );
        }
    }

    #[test]
    fn t2_audit_is_deterministic() {
        let (a1, r1) = run_planted_bank(42);
        let (a2, r2) = run_planted_bank(42);
        assert_eq!(a1.ooo, a2.ooo);
        assert_eq!(a1.meis, a2.meis);
        assert_eq!(a1.cross, a2.cross);
        assert_eq!(r1.kept, r2.kept);
    }

    // ── T3: defend-wrong PoC on real k-means substrate ────────────────────
    //
    // The kill condition (Issue 759 T3): fit an OVER-COMPLETE codebook
    // (K = 12 centroids over 4 true clusters) with the shipped deterministic
    // k-means (factorized_action::fit_codebook_kmeans_into — no GD), read it
    // as a UTM bank (activation = −Euclidean distance to each centroid, the
    // paper's exact construction), and audit. If the curated count came back
    // ≈ K (no redundancy detected), the gate is hygiene-only → demote.
    // Expected: count ≈ 4 — the audit measures the 4 distinct contents the
    // 12-centroid bank actually carries (the F3 K-selection signal).

    /// Concept-level similarity matrix (the DreamSim role): `hi` within a
    /// planted cluster, `lo` across — constant regardless of micro-position.
    /// Same-concept exemplars sit AT the threshold (not below), so OOO
    /// measures the concept's dataset fraction and Cross-OOO reads exactly
    /// 0.0 for same-concept units (every comparison ties) — the deterministic
    /// form of the paper's chance-level redundancy signal.
    fn concept_block_sim(n: usize, m: usize, hi: f32, lo: f32) -> Vec<f32> {
        let mut s = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                s[i * n + j] = if i / m == j / m { hi } else { lo };
            }
        }
        s
    }

    #[cfg(feature = "factorized_action")]
    #[test]
    fn t3_overcomplete_codebook_audit_detects_redundancy() {
        use crate::factorized_action::fit_codebook_kmeans_into;
        use crate::factorized_action::EffectCodebook;

        const G: usize = 4;
        const M: usize = 30; // OOO = (N−M)/(N−K) = 90/112 ≈ 0.804 ≥ 0.8
        const D: usize = 8;
        const K: usize = 12; // over-complete: 3× the true cluster count
        let n = G * M;

        // Exemplars: one-hot-scaled centers + noise (same shape as T2).
        let (x, _) = fixtures::planted_cluster_bank(G, M, D, 0, 0, 99);
        let patches: Vec<&[f32]> = (0..n).map(|i| &x[i * D..i * D + D]).collect();

        // Fit the over-complete codebook on the shipped k-means substrate.
        let mut codebook: EffectCodebook<K, D> = EffectCodebook::zeroed();
        fit_codebook_kmeans_into(&patches, K, 7, 20, &mut codebook);
        let centroids = &codebook.centroids;

        // UTM bank: unit z's activation on exemplar i = −dist(x_i, c_z).
        let mut a = vec![0.0f32; K * n];
        for z in 0..K {
            for i in 0..n {
                let mut acc = 0.0f32;
                for c in 0..D {
                    let dd = x[i * D + c] - centroids[z][c];
                    acc += dd * dd;
                }
                a[z * n + i] = -acc.sqrt();
            }
        }

        // Concept-level similarity — the DreamSim role (see the
        // metric-granularity law test below for the geometric counter-arm).
        let sim = concept_block_sim(n, M, 0.95, 0.05);

        let cfg = OooAuditConfig {
            top_k: 8,
            ..Default::default()
        };
        let mut scratch = AuditScratch::new();
        let mut audit = BankAudit::new(K, n, cfg.top_k);
        audit_bank_into(&a, &sim, &cfg, &mut scratch, &mut audit);
        let mut result = CurationResult::new(K);
        greedy_curate(&audit, &cfg, &mut scratch, &mut result);

        let count = result.unique_feature_count();
        // Redundancy DETECTED: 12-centroid bank curates to well under 12.
        assert!(
            count < 8,
            "kill condition: over-complete K=12 must not pass as 12 distinct features (got {count})"
        );
        // And it recovers the planted structure: ≈ the 4 true clusters
        // (±1 for k-means local optima at the cluster boundary).
        assert!(
            (3..=5).contains(&count),
            "curated count {count} should match the 4 true clusters ±1"
        );
        // At least one centroid pair was pruned as redundant — the measured
        // redundancy this PoC exists to demonstrate.
        assert!(
            !result.pruned_redundant.is_empty(),
            "over-complete codebook must produce at least one pruned-redundant centroid"
        );
    }

    #[cfg(feature = "factorized_action")]
    #[test]
    fn t3_exact_codebook_audit_recovers_cluster_count() {
        use crate::factorized_action::EffectCodebook;

        const G: usize = 4;
        const M: usize = 30;
        const D: usize = 8;
        const K: usize = 4; // exact: one centroid per true cluster
        let n = G * M;

        let (x, _) = fixtures::planted_cluster_bank(G, M, D, 0, 0, 99);
        let patches: Vec<&[f32]> = (0..n).map(|i| &x[i * D..i * D + D]).collect();
        let mut codebook: EffectCodebook<K, D> = EffectCodebook::zeroed();
        crate::factorized_action::fit_codebook_kmeans_into(&patches, K, 7, 20, &mut codebook);
        let centroids = &codebook.centroids;

        let mut a = vec![0.0f32; K * n];
        for z in 0..K {
            for i in 0..n {
                let mut acc = 0.0f32;
                for c in 0..D {
                    let dd = x[i * D + c] - centroids[z][c];
                    acc += dd * dd;
                }
                a[z * n + i] = -acc.sqrt();
            }
        }
        let sim = concept_block_sim(n, M, 0.95, 0.05);

        let cfg = OooAuditConfig::default();
        let mut scratch = AuditScratch::new();
        let mut audit = BankAudit::new(K, n, cfg.top_k);
        audit_bank_into(&a, &sim, &cfg, &mut scratch, &mut audit);
        let mut result = CurationResult::new(K);
        greedy_curate(&audit, &cfg, &mut scratch, &mut result);

        assert_eq!(
            result.unique_feature_count(),
            G,
            "exact-fit codebook curates to exactly the cluster count (the K-selection elbow anchor)"
        );
    }

    /// The metric-granularity law (measured 2026-09-12, the defend-wrong
    /// PoC's counter-arm): Cross-OOO detects redundancy AT THE SIMILARITY
    /// METRIC'S concept granularity. A geometric RBF (γ=1) resolves the
    /// intra-cluster micro-splits kmeans creates — each centroid's
    /// nearest-exemplar subset is spatially distinct — so the SAME K=12
    /// bank that curates to 4 under concept-level similarity keeps ~11
    /// features under geometric similarity. Neither reading is wrong: the
    /// metric defines the concept. Consumers must choose a metric at their
    /// semantic granularity (span embeddings for rules, latent cosine for
    /// directions — the DreamSim role), or the audit honestly reports
    /// micro-features.
    #[cfg(feature = "factorized_action")]
    #[test]
    fn t3_metric_granularity_law_geometric_rbf_resolves_micro_splits() {
        use crate::factorized_action::fit_codebook_kmeans_into;
        use crate::factorized_action::EffectCodebook;

        const G: usize = 4;
        const M: usize = 30;
        const D: usize = 8;
        const K: usize = 12;
        let n = G * M;

        let (x, _) = fixtures::planted_cluster_bank(G, M, D, 0, 0, 99);
        let patches: Vec<&[f32]> = (0..n).map(|i| &x[i * D..i * D + D]).collect();
        let mut codebook: EffectCodebook<K, D> = EffectCodebook::zeroed();
        fit_codebook_kmeans_into(&patches, K, 7, 20, &mut codebook);
        let centroids = &codebook.centroids;

        let mut a = vec![0.0f32; K * n];
        for z in 0..K {
            for i in 0..n {
                let mut acc = 0.0f32;
                for c in 0..D {
                    let dd = x[i * D + c] - centroids[z][c];
                    acc += dd * dd;
                }
                a[z * n + i] = -acc.sqrt();
            }
        }
        let mut sim = vec![0.0f32; n * n];
        exemplar_rbf_sim_into(&x, D, 1.0, &mut sim);

        let cfg = OooAuditConfig::default();
        let mut scratch = AuditScratch::new();
        let mut audit = BankAudit::new(K, n, cfg.top_k);
        audit_bank_into(&a, &sim, &cfg, &mut scratch, &mut audit);
        let mut result = CurationResult::new(K);
        greedy_curate(&audit, &cfg, &mut scratch, &mut result);

        assert!(
            (8..=12).contains(&result.unique_feature_count()),
            "geometric granularity resolves kmeans micro-splits as distinct features \
             (got {} — the concept-level arm of the same bank curates to 4)",
            result.unique_feature_count()
        );
    }
}
