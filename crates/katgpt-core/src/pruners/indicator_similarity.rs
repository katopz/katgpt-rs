//! Indicator Similarity Matrix — pairwise cosine structure of an
//! [`IndicatorProbeBank`](super::indicator_probe_bank::IndicatorProbeBank)'s
//! direction vectors (Plan 320 Phase 2, Research 301).
//!
//! Paper Fig. 6 finding: the indicators form a shared "misaligned-reasoning"
//! subspace (most pairs in [0.3, 0.7]) with within-category block structure.
//! This matrix makes that structure first-class inspectable and committable.
//!
//! The matrix is computed ONCE at construction (`O(N²·D)`) and stored row-major.
//! `similarity(i, j)` is then `O(1)`. The `cluster` greedy recovery exposes
//! the within-category block structure as a `Vec<Vec<L>>`.

use core::marker::PhantomData;

use crate::pruners::indicator_probe_bank::{IndicatorLabel, IndicatorProbeBank};

/// Cosine-similarity matrix of an `IndicatorProbeBank`'s direction vectors.
///
/// Symmetric, diagonal = 1.0 (within float tolerance). Stored row-major as
/// `cosines[i * n + j]`.
pub struct IndicatorSimilarityMatrix<L: IndicatorLabel> {
    /// `N × N` symmetric matrix of cosines. Stored row-major.
    cosines: Vec<f32>,
    /// Number of indicators.
    n: usize,
    _marker: PhantomData<L>,
}

impl<L: IndicatorLabel, const D: usize> From<&IndicatorProbeBank<L, D>>
    for IndicatorSimilarityMatrix<L>
{
    /// Compute the full `N × N` cosine matrix from a bank.
    ///
    /// `O(N²·D)` — done once at construction; lookups are then `O(1)`.
    #[inline]
    fn from(bank: &IndicatorProbeBank<L, D>) -> Self {
        Self::from_bank(bank)
    }
}

impl<L: IndicatorLabel> IndicatorSimilarityMatrix<L> {
    /// Compute the full `N × N` cosine matrix from a bank's direction vectors.
    pub fn from_bank<const D: usize>(bank: &IndicatorProbeBank<L, D>) -> Self {
        let n = L::COUNT;
        let mut cosines = vec![0.0f32; n * n];
        for i in 0..n {
            let di = bank.direction(i);
            let ni = norm(di);
            cosines[i * n + i] = if ni > 0.0 { 1.0 } else { 0.0 };
            for j in (i + 1)..n {
                let dj = bank.direction(j);
                let nj = norm(dj);
                let c = if ni > 0.0 && nj > 0.0 {
                    crate::simd::simd_dot_f32(di, dj, D) / (ni * nj)
                } else {
                    0.0
                };
                cosines[i * n + j] = c;
                cosines[j * n + i] = c; // symmetric
            }
        }
        Self {
            cosines,
            n,
            _marker: PhantomData,
        }
    }

    /// Number of indicators.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Constant-time lookup of `cosine(direction_i, direction_j)`.
    #[inline]
    pub fn similarity(&self, i: usize, j: usize) -> f32 {
        debug_assert!(i < self.n && j < self.n, "indices out of range");
        self.cosines[i * self.n + j]
    }

    /// Greedy within-category block recovery.
    ///
    /// Group indicators whose pairwise similarity exceeds `tau_intra`, using
    /// complete linkage: merge `Ga, Gb` iff **every** cross-pair has
    /// `sim(i, j) ≥ tau_intra`. This recovers the paper's Fig. 6 tight
    /// within-category blocks while rejecting loose cross-category bridges
    /// (which have at least one sub-threshold cross-pair).
    ///
    /// `tau_inter` is reserved for future inter-group rejection rules; for now
    /// the algorithm uses complete linkage on `tau_intra` only. The two-arg
    /// signature matches the paper's dual-threshold description so callers can
    /// upgrade the algorithm without an API break.
    ///
    /// At each step the densest valid merge (highest `min` cross-pair) is chosen,
    /// giving deterministic output. Merges continue until no valid pair remains.
    pub fn cluster(&self, tau_intra: f32, _tau_inter: f32) -> Vec<Vec<L>> {
        let n = self.n;
        // Start each indicator in its own group (singleton).
        let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        let cross_min = |a: &[usize], b: &[usize]| -> f32 {
            let mut mn = f32::INFINITY;
            for &i in a {
                for &j in b {
                    let s = self.similarity(i, j);
                    if s < mn {
                        mn = s;
                    }
                }
            }
            mn
        };

        // Greedy merge until stable. O(N³) worst case but N is tiny (≤ ~64).
        loop {
            // Pick the densest valid merge (highest min cross-pair ≥ tau_intra).
            // Deterministic: lowest (ga, gb) index pair breaks ties.
            let mut best: Option<(usize, usize)> = None;
            let mut best_min = f32::NEG_INFINITY;
            for ga in 0..groups.len() {
                for gb in (ga + 1)..groups.len() {
                    let mn = cross_min(&groups[ga], &groups[gb]);
                    if mn < tau_intra {
                        continue;
                    }
                    if mn > best_min {
                        best_min = mn;
                        best = Some((ga, gb));
                    }
                }
            }
            match best {
                Some((ga, gb)) => {
                    // Merge gb into ga (preserve lower index for determinism).
                    let mut b_group = groups[gb].clone();
                    groups[ga].append(&mut b_group);
                    groups[ga].sort_unstable();
                    groups.remove(gb);
                }
                None => break,
            }
        }

        // Materialize labels. Sort within-group + sort groups by first index for
        // deterministic output ordering.
        for g in groups.iter_mut() {
            g.sort_unstable();
        }
        groups.sort_by_key(|g| g.first().copied().unwrap_or(usize::MAX));

        groups
            .into_iter()
            .map(|idxs| {
                idxs.into_iter()
                    .map(|i| L::from_u8(i as u8).expect("valid label discriminant"))
                    .collect()
            })
            .collect()
    }
}

impl<L: IndicatorLabel> core::fmt::Debug for IndicatorSimilarityMatrix<L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IndicatorSimilarityMatrix")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

/// L2 norm of an f32 slice.
#[inline]
fn norm(v: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::indicator_probe_bank::IndicatorProbeBank;

    /// Build a bank with planted block structure for cluster tests.
    ///
    /// 6 indicators (A=0..C=2 expanded to 6) split into two within-block pairs:
    ///   block 0 = {0, 1}, block 1 = {2, 3}, block 2 = {4, 5}
    /// We need 6 labels, so use a 6-variant label enum.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(u8)]
    enum SixLabel {
        L0 = 0,
        L1 = 1,
        L2 = 2,
        L3 = 3,
        L4 = 4,
        L5 = 5,
    }
    impl IndicatorLabel for SixLabel {
        fn as_u8(&self) -> u8 {
            *self as u8
        }
        fn from_u8(d: u8) -> Option<Self> {
            match d {
                0 => Some(Self::L0),
                1 => Some(Self::L1),
                2 => Some(Self::L2),
                3 => Some(Self::L3),
                4 => Some(Self::L4),
                5 => Some(Self::L5),
                _ => None,
            }
        }
        const COUNT: usize = 6;
    }

    const D: usize = 8;

    /// Planted directions: 3 blocks of 2 indicators, within-block cosine ≈ 0.7,
    /// cross-block ≈ 0.
    ///
    /// For directions `[1, a]` and `[a, 1]`, cosine = 2a/(1+a²). Solving
    /// 2a/(1+a²) = 0.7 gives a ≈ 0.408. We use a = 0.4 → cosine ≈ 0.69 (> 0.6).
    /// The three blocks use disjoint coordinate axes so cross-block cosines are
    /// exactly 0.
    fn planted_bank() -> IndicatorProbeBank<SixLabel, D> {
        let mut dirs = vec![0.0f32; SixLabel::COUNT * D];
        // Three anchor/partner pairs, each on disjoint 2-dim subspaces.
        let anchors: [[f32; D]; 3] = [
            [1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0],
        ];
        let partners: [[f32; D]; 3] = [
            [0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 0.0, 0.0],
        ];
        for (block, (a, p)) in anchors.iter().zip(partners.iter()).enumerate() {
            // block i → labels 2i, 2i+1
            let idx0 = (2 * block) * D;
            let idx1 = (2 * block + 1) * D;
            dirs[idx0..idx0 + D].copy_from_slice(a);
            dirs[idx1..idx1 + D].copy_from_slice(p);
        }
        let thresholds = vec![0.0f32; SixLabel::COUNT];
        IndicatorProbeBank::new(dirs, thresholds).expect("planted bank shape")
    }

    #[test]
    fn test_from_bank_produces_symmetric_matrix() {
        let bank = planted_bank();
        let m = IndicatorSimilarityMatrix::<SixLabel>::from_bank(&bank);
        for i in 0..SixLabel::COUNT {
            for j in 0..SixLabel::COUNT {
                let a = m.similarity(i, j);
                let b = m.similarity(j, i);
                assert!(
                    (a - b).abs() < 1e-5,
                    "asymmetric: sim({},{})={} vs sim({},{})={}",
                    i,
                    j,
                    a,
                    j,
                    i,
                    b
                );
            }
        }
    }

    #[test]
    fn test_diagonal_is_one() {
        let bank = planted_bank();
        let m = IndicatorSimilarityMatrix::<SixLabel>::from_bank(&bank);
        for i in 0..SixLabel::COUNT {
            assert!(
                (m.similarity(i, i) - 1.0).abs() < 1e-5,
                "diagonal sim({},{}) = {} != 1.0",
                i,
                i,
                m.similarity(i, i)
            );
        }
    }

    #[test]
    fn test_cluster_recovers_planted_blocks() {
        let bank = planted_bank();
        let m = IndicatorSimilarityMatrix::<SixLabel>::from_bank(&bank);

        // Sanity: within-block cosines should be ≈ 0.707, cross-block ≈ 0.
        // (Our planted anchors are exactly orthogonal across blocks.)
        for block in 0..3 {
            let i = 2 * block;
            let j = 2 * block + 1;
            let within = m.similarity(i, j);
            assert!(
                within > 0.6,
                "within-block sim({},{}) = {} should be > 0.6",
                i,
                j,
                within
            );
        }

        // Cluster with tau_intra=0.6, tau_inter=0.6 (no third-group rejection
        // needed since cross-block is ~0). Should recover 3 blocks of 2.
        let clusters = m.cluster(0.6, 0.6);
        assert_eq!(
            clusters.len(),
            3,
            "expected 3 planted blocks, got {}: {:?}",
            clusters.len(),
            clusters
        );
        for c in &clusters {
            assert_eq!(c.len(), 2, "each planted block has 2 indicators");
        }

        // Compute ARI vs planted partition and assert ≥ 0.9.
        let planted = vec![
            vec![SixLabel::L0, SixLabel::L1],
            vec![SixLabel::L2, SixLabel::L3],
            vec![SixLabel::L4, SixLabel::L5],
        ];
        let ari = adjusted_rand_index(&clusters, &planted);
        assert!(
            ari >= 0.9,
            "ARI {} < 0.9 — cluster did not recover planted blocks. clusters={:?} planted={:?}",
            ari,
            clusters,
            planted
        );
    }

    #[test]
    fn test_cluster_returns_single_group_when_all_similar() {
        // All directions identical → all cosines 1.0 → one group.
        let dirs = vec![1.0f32; SixLabel::COUNT * D];
        let thresholds = vec![0.0f32; SixLabel::COUNT];
        let bank = IndicatorProbeBank::<SixLabel, D>::new(dirs, thresholds).unwrap();
        let m = IndicatorSimilarityMatrix::<SixLabel>::from_bank(&bank);
        let clusters = m.cluster(0.6, 0.6);
        assert_eq!(clusters.len(), 1, "all-similar → one group");
        assert_eq!(clusters[0].len(), SixLabel::COUNT);
    }

    #[test]
    fn test_cluster_returns_singletons_when_all_orthogonal() {
        // Orthonormal basis → all off-diagonal cosines 0 → N singletons.
        // 6 labels × D=8: use distinct standard basis vectors.
        let mut dirs = vec![0.0f32; SixLabel::COUNT * D];
        for i in 0..SixLabel::COUNT {
            dirs[i * D + i] = 1.0;
        }
        let thresholds = vec![0.0f32; SixLabel::COUNT];
        let bank = IndicatorProbeBank::<SixLabel, D>::new(dirs, thresholds).unwrap();
        let m = IndicatorSimilarityMatrix::<SixLabel>::from_bank(&bank);
        let clusters = m.cluster(0.6, 0.6);
        assert_eq!(
            clusters.len(),
            SixLabel::COUNT,
            "all-orthogonal → N singletons"
        );
        for c in &clusters {
            assert_eq!(c.len(), 1, "singletons");
        }
    }

    // ---- ARI helper (test-only; small N) ----

    /// Adjusted Rand Index between two partitions of the same label set.
    /// Returns 1.0 for identical, ~0.0 for random, negative for anti-correlated.
    fn adjusted_rand_index<L: IndicatorLabel>(a: &[Vec<L>], b: &[Vec<L>]) -> f64 {
        // Flatten both partitions to a per-label assignment using a contingency
        // table. Map every label to (cluster_id_in_a, cluster_id_in_b).
        let mut all: Vec<L> = Vec::new();
        for g in a {
            all.extend_from_slice(g);
        }
        for g in b {
            all.extend_from_slice(g);
        }
        all.sort_by_key(|l| l.as_u8());
        all.dedup_by_key(|l| l.as_u8());

        let label_to_a = |l: L| -> usize {
            for (ci, g) in a.iter().enumerate() {
                if g.contains(&l) {
                    return ci;
                }
            }
            usize::MAX
        };
        let label_to_b = |l: L| -> usize {
            for (ci, g) in b.iter().enumerate() {
                if g.contains(&l) {
                    return ci;
                }
            }
            usize::MAX
        };

        // Contingency table nij = |cluster_i_in_a ∩ cluster_j_in_b|.
        let na = a.len();
        let nb = b.len();
        let mut nij = vec![0u64; na * nb];
        for &l in &all {
            let ia = label_to_a(l);
            let ib = label_to_b(l);
            if ia != usize::MAX && ib != usize::MAX {
                nij[ia * nb + ib] += 1;
            }
        }
        let sum_comb_nij: f64 = nij.iter().map(|&n| comb2(n)).sum();
        let ai_sums: Vec<u64> = (0..na)
            .map(|i| (0..nb).map(|j| nij[i * nb + j]).sum())
            .collect();
        let bj_sums: Vec<u64> = (0..nb)
            .map(|j| (0..na).map(|i| nij[i * nb + j]).sum())
            .collect();
        let sum_comb_a: f64 = ai_sums.iter().map(|&n| comb2(n)).sum();
        let sum_comb_b: f64 = bj_sums.iter().map(|&n| comb2(n)).sum();
        let total: u64 = all.len() as u64;
        let expected = sum_comb_a * sum_comb_b / comb2(total);
        let max_index = 0.5 * (sum_comb_a + sum_comb_b);
        if (max_index - expected).abs() < 1e-12 {
            return 1.0;
        }
        (sum_comb_nij - expected) / (max_index - expected)
    }

    /// C(n, 2) as f64.
    #[inline]
    fn comb2(n: u64) -> f64 {
        if n < 2 {
            0.0
        } else {
            let n = n as f64;
            n * (n - 1.0) / 2.0
        }
    }
}
