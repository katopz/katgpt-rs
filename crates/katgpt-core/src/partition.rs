//! Min-max DP partition of an ordered sequence under a pairwise-distance
//! oracle (Plan 616 — promoted verbatim from riir-infer `twt::partition`,
//! Issue 022 T2.1, katgpt-rs Research 594).
//!
//! Objective (TWT §3.1): `min_P (m, max_j S[s_j,e_j])` s.t.
//! `S[s_j,e_j] ≤ ε` — pass 1 minimizes the block count under the ε
//! constraint, pass 2 tie-breaks on the worst-case intra-block discrepancy.
//! Blocks are CONTIGUOUS (depth is ordered) and half-open `[start, end)`.
//!
//! # DRY cross-link (Research 594 generalize-or-neighbor — T0.1 NEIGHBOR)
//!
//! `crate::ugc_schedule::dp_partition(profile, k)` ships a contiguous
//! K-block DP with a SUM-of-costs edge oracle at FIXED k (paper §4.4.2 Eq
//! 39). This module answers a different objective (min-MAX worst case
//! under an ε constraint, k FREE) over a generic distance matrix. The two
//! are NEIGHBORS, deliberately: a generalized form would change every
//! `dp_partition` call-site signature (the promotion criterion was
//! call-site byte-identity, and a promotion must not move a shipped
//! consumer), and the UGC recurrence's value is being exactly the paper's
//! shape. Each documents the other; neither may drift silently.
//!
//! # Determinism
//!
//! Every comparison is a total-order f32 compare. Argmin keeps the first
//! index on ties (index-ordered), so identical inputs give bit-identical
//! partitions on every platform. The fixture gate
//! (`tests/minmax_partition_fixture.rs`) pins an externally measured
//! S-matrix's m table as a cross-repo known answer.
//!
//! Min-max DP partition under a pairwise-distance ε oracle (Plan 616 — the
//! TWT phase-collapse lane's partition primitive, promoted verbatim from
//! riir-infer `twt::partition`, Issue 022 T2.1 / Research 594 row 2).
//! NEIGHBOR of `ugc_schedule::dp_partition` (min-SUM fixed-k, paper Eq 39)
//! — the generalize-vs-neighbor decision and the cross-link live in the
//! module doc. Opt-in per the no-default-consumer rule; first consumer is
//! riir-infer's twt lane (re-exporting since the promotion commit).

/// Errors returned by the min-max partition primitives. Hand-rolled
/// `Display` + `Error` impls (the house convention — katgpt-core carries
/// no `thiserror` dep).
#[derive(Clone, Debug, PartialEq)]
pub enum PartitionError {
    /// `eps` was negative or non-finite; the DP requires `eps ≥ 0` and
    /// finite (a NaN ε would read every block feasible via `> eps` being
    /// false).
    InvalidEps(f32),
    /// The typed partition's `types` slice did not carry one flag per
    /// layer.
    TypesLengthMismatch { got: usize, want: usize },
}

impl std::fmt::Display for PartitionError {
    #[cold]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidEps(eps) => write!(f, "invalid ε {eps}: must be finite and >= 0"),
            Self::TypesLengthMismatch { got, want } => {
                write!(
                    f,
                    "types length {got} != S size {want} — the typed partition needs one flag per layer"
                )
            }
        }
    }
}

impl std::error::Error for PartitionError {}

/// One contiguous block of layers, half-open `[start, end)` — `end` is
/// EXCLUSIVE (Rust range convention); a block covers layers
/// `start ..= end-1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub start: usize,
    pub end: usize,
}

impl Block {
    /// Number of layers in the block.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// A finalized L×L symmetric discrepancy matrix (row-major, diagonal 0,
/// upper triangle computed and mirrored — symmetry holds BY CONSTRUCTION,
/// the fixture gate asserts it anyway).
///
/// Promoted with the partition DP (Plan 616): the DP's only read pattern
/// is `get(q, p)` over `i ≤ p < q < j`, and construction validates
/// finiteness eagerly — a NaN distance would read FEASIBLE inside the DP
/// (`NaN > eps` is false), so non-finite inputs are refused at
/// construction, never pooled.
#[derive(Debug, Clone)]
pub struct SMatrix {
    n: usize,
    data: Vec<f32>,
}

impl SMatrix {
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.n + j]
    }

    /// Synthetic construction for gates and perf probes — not a
    /// production path; the diagonal is forced to 0, the fn is only read
    /// for `i < j` in fixed row-major order, and NaN/Inf inputs panic
    /// (the DP requires a finite oracle; `minmax_partition` would misread
    /// NaN as feasible).
    pub fn from_fn(n: usize, mut f: impl FnMut(usize, usize) -> f32) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let v = f(i, j);
                assert!(v.is_finite(), "SMatrix::from_fn: non-finite at ({i},{j})");
                data[i * n + j] = v;
                data[j * n + i] = v;
            }
        }
        Self { n, data }
    }

    /// Production construction from a fully computed row-major mirrored
    /// matrix (the streaming builder's finalize path, and fixture loads).
    /// Validates the same finiteness contract as [`Self::from_fn`] —
    /// eagerly, O(n²), so a NaN can never reach the DP.
    pub fn from_parts(n: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            n * n,
            "SMatrix::from_parts: data must be n×n row-major"
        );
        assert!(
            data.iter().all(|v| v.is_finite()),
            "SMatrix::from_parts: non-finite distance — the DP requires a finite oracle"
        );
        Self { n, data }
    }

    /// Upper-triangle iteration (i < j).
    pub fn entries(&self) -> impl Iterator<Item = (usize, usize, f32)> + '_ {
        let n = self.n;
        (0..n).flat_map(move |i| ((i + 1)..n).map(move |j| (i, j, self.data[i * n + j])))
    }
}

/// Streaming dot+norm accumulator over `(a, b)` state pairs.
///
/// `distance()` finalizes `1 - cos` in `[0, 2]`. The zero-norm /
/// non-finite case returns `1.0` (maximally distant — the conservative
/// choice of the cosine_distance discipline: no information → treat as
/// different, never as similar; a merge decision made on a fabricated 0.0
/// would collapse layers that were never observed to agree).
///
/// Follows the reduction DISCIPLINE of single-pass dot + squared norms
/// accumulated in **f64**: no allocation, overflow-safe for f32 inputs,
/// conservative zero-norm. Promoted with the partition DP (Plan 616).
///
/// # Determinism
///
/// f64 addition order is part of the result: callers must feed pairs in a
/// canonical order (rows ascending, forwards in corpus order). Identical
/// feeds are bit-identical cross-platform.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PairCosineAccum {
    dot: f64,
    na: f64,
    nb: f64,
    positions: u64,
}

impl PairCosineAccum {
    pub const fn new() -> Self {
        Self {
            dot: 0.0,
            na: 0.0,
            nb: 0.0,
            positions: 0,
        }
    }

    /// Fold one `(a, b)` state pair (one calibration position).
    ///
    /// Mismatched lengths fold the common prefix (the cosine_distance
    /// discipline's `min`); builders that assert equality upstream make
    /// this arm defensive only.
    #[inline]
    pub fn add(&mut self, a: &[f32], b: &[f32]) {
        debug_assert_eq!(a.len(), b.len(), "pair state lengths");
        let len = a.len().min(b.len());
        for k in 0..len {
            let x = a[k] as f64;
            let y = b[k] as f64;
            self.dot += x * y;
            self.na += x * x;
            self.nb += y * y;
        }
        self.positions += 1;
    }

    /// Calibration positions folded so far.
    pub fn positions(&self) -> u64 {
        self.positions
    }

    /// Finalize `1 - cos` in `[0, 2]`; `1.0` when either pooled side is
    /// zero-norm or the reduction went non-finite (conservative).
    pub fn distance(&self) -> f32 {
        let denom = (self.na * self.nb).sqrt();
        if denom == 0.0 || !denom.is_finite() {
            return 1.0;
        }
        let cos = (self.dot / denom).clamp(-1.0, 1.0);
        if cos.is_nan() {
            return 1.0;
        }
        (1.0 - cos) as f32
    }
}

/// `max S[q][p]` over `i ≤ p < q < j` — the worst intra-block discrepancy
/// for the block `[i, j)`. Returned as a full `n×n` row-major table
/// (`[i*n+j]`, `j ≥ i`; diagonal 0). O(n²) time via the descending-i
/// column-prefix recurrence, O(n²) memory.
fn worst_table(s: &SMatrix) -> Vec<f32> {
    let n = s.n();
    let mut w = vec![0.0f32; n * n];
    for i in (0..n).rev() {
        // `run` = max over q in [i+1, j] of S[q][i] — a column prefix,
        // extended in O(1) as j grows.
        let mut run = 0.0f32;
        for j in (i + 1)..n {
            let v = s.get(j, i);
            run = if v > run { v } else { run };
            let below = w[(i + 1) * n + j];
            w[i * n + j] = if run > below { run } else { below };
        }
    }
    w
}

/// The two-pass min-max DP. Returns the count-optimal, worst-case-minimal
/// contiguous partition. Feasibility monotonicity (worst(i,j) is
/// non-decreasing as the block grows leftward) lets each candidate scan
/// stop at the first infeasible boundary.
pub fn minmax_partition(s: &SMatrix, eps: f32) -> Result<Vec<Block>, PartitionError> {
    partition_dp(s, eps, None)
}

/// [`minmax_partition`] under the TYPE constraint: a block may only span
/// layers of ONE type, because layers of different types carry different
/// tensor sets and shapes — there is no operator to average across the
/// boundary. This is the merge-feasible partition: every multi-layer
/// block it emits is a REAL merge candidate (homogeneous), which the
/// unconstrained DP only produced by accident.
///
/// `types[i]` = true for sliding-window/recurrent layers, false for
/// full-attention (the same convention [`forced_min_blocks`] documents).
/// Monotone feasibility carries over: the block `[i, j]` grows leftward
/// in the scans, so the first `types[i] != types[j]` boundary ends the
/// scan (every deeper `i` keeps the mismatched layer `i` inside).
pub fn minmax_partition_typed(
    s: &SMatrix,
    eps: f32,
    types: &[bool],
) -> Result<Vec<Block>, PartitionError> {
    if types.len() != s.n() {
        return Err(PartitionError::TypesLengthMismatch {
            got: types.len(),
            want: s.n(),
        });
    }
    partition_dp(s, eps, Some(types))
}

/// The two-pass min-max DP core. `types = None` is the unconstrained
/// original; `Some(types)` restricts every block to one type (see
/// [`minmax_partition_typed`]).
fn partition_dp(
    s: &SMatrix,
    eps: f32,
    types: Option<&[bool]>,
) -> Result<Vec<Block>, PartitionError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(PartitionError::InvalidEps(eps));
    }
    let n = s.n();
    if n == 0 {
        return Ok(Vec::new());
    }
    let worst = worst_table(s);
    // The per-scan feasibility guard: a type-constrained scan stops at the
    // first mismatched layer (monotone — see minmax_partition_typed).
    let type_ok = |i: usize, j: usize| -> bool {
        match types {
            None => true,
            Some(t) => t[i] == t[j],
        }
    };

    const INF: usize = usize::MAX;
    // Pass 1 — count[j]: minimum blocks covering [0, j).
    let mut count = vec![INF; n];
    for j in 0..n {
        let mut i = j + 1;
        while i > 0 {
            i -= 1;
            if !type_ok(i, j) || worst[i * n + j] > eps {
                break; // feasibility is monotone in i — everything further left is worse
            }
            let prev = if i == 0 { 0 } else { count[i - 1] };
            if prev == INF {
                continue;
            }
            if prev + 1 < count[j] {
                count[j] = prev + 1;
            }
        }
    }
    debug_assert_ne!(count[n - 1], INF, "eps >= 0: singletons always feasible");

    // Pass 2 — worst_pref[j]: the minimum worst-case discrepancy over
    // count-optimal partitions of [0, j).
    let mut worst_pref = vec![f32::INFINITY; n];
    for j in 0..n {
        let target = count[j] - 1;
        let mut i = j + 1;
        while i > 0 {
            i -= 1;
            if !type_ok(i, j) || worst[i * n + j] > eps {
                break;
            }
            let prev_count = if i == 0 { 0 } else { count[i - 1] };
            if prev_count != target {
                continue;
            }
            let prev_worst = if i == 0 { 0.0 } else { worst_pref[i - 1] };
            let cand = if prev_worst > worst[i * n + j] {
                prev_worst
            } else {
                worst[i * n + j]
            };
            if cand < worst_pref[j] {
                worst_pref[j] = cand;
            }
        }
    }

    // Reconstruct: walk boundaries backward, picking the first boundary
    // achieving the stored optimum (index-ordered argmin — deterministic).
    let mut blocks = Vec::with_capacity(count[n - 1]);
    let mut j = n - 1;
    loop {
        let target = count[j] - 1;
        let target_worst = worst_pref[j];
        let mut i = j + 1;
        let mut chosen = 0usize;
        while i > 0 {
            i -= 1;
            if !type_ok(i, j) || worst[i * n + j] > eps {
                break;
            }
            let prev_count = if i == 0 { 0 } else { count[i - 1] };
            if prev_count != target {
                continue;
            }
            let prev_worst = if i == 0 { 0.0 } else { worst_pref[i - 1] };
            let cand = if prev_worst > worst[i * n + j] {
                prev_worst
            } else {
                worst[i * n + j]
            };
            if cand == target_worst {
                chosen = i;
                break;
            }
        }
        blocks.push(Block {
            start: chosen,
            end: j + 1,
        });
        if chosen == 0 {
            break;
        }
        j = chosen - 1;
    }
    blocks.reverse();
    Ok(blocks)
}

/// Worst intra-block discrepancy of a CONCRETE partition (the G1
/// post-condition arm + the brute-force comparator's objective).
pub fn partition_worst(s: &SMatrix, blocks: &[Block]) -> f32 {
    let mut worst = 0.0f32;
    for b in blocks {
        for p in b.start..b.end {
            for q in (p + 1)..b.end {
                let v = s.get(q, p);
                if v > worst {
                    worst = v;
                }
            }
        }
    }
    worst
}

/// Brute-force lexicographic optimum `(count, worst)` over ALL 2^(L-1)
/// contiguous partitions. Exponential — gate instrument only (n ≤ 14).
pub fn brute_force_optimal(s: &SMatrix, eps: f32) -> (usize, f32) {
    let n = s.n();
    assert!(n <= 14, "brute force is 2^(n-1) — gate instrument only");
    let mut best: Option<(usize, f32)> = None;
    for mask in 0u32..(1u32 << (n - 1)) {
        // bit c set ⇒ a cut before layer c+1.
        let mut blocks: Vec<Block> = Vec::with_capacity(n);
        let mut start = 0usize;
        for c in 0..(n - 1) {
            if mask & (1 << c) != 0 {
                blocks.push(Block { start, end: c + 1 });
                start = c + 1;
            }
        }
        blocks.push(Block { start, end: n });
        let mut feasible = true;
        let mut worst = 0.0f32;
        'blk: for b in &blocks {
            for p in b.start..b.end {
                for q in (p + 1)..b.end {
                    let v = s.get(q, p);
                    if v > eps {
                        feasible = false;
                        break 'blk;
                    }
                    if v > worst {
                        worst = v;
                    }
                }
            }
        }
        if !feasible {
            continue;
        }
        let cand = (blocks.len(), worst);
        let better = match best {
            None => true,
            Some((bc, bw)) => bc > cand.0 || (bc == cand.0 && bw > cand.1),
        };
        if better {
            best = Some(cand);
        }
    }
    best.expect("eps >= 0: singleton partition always feasible")
}

/// The typed twin of [`brute_force_optimal`] — the G1 comparator for
/// [`minmax_partition_typed`]. Same enumeration, feasibility adds the
/// same-type requirement per block.
pub fn brute_force_optimal_typed(s: &SMatrix, eps: f32, types: &[bool]) -> (usize, f32) {
    let n = s.n();
    assert!(n <= 14, "brute force is 2^(n-1) — gate instrument only");
    assert_eq!(types.len(), n, "one flag per layer");
    let mut best: Option<(usize, f32)> = None;
    for mask in 0u32..(1u32 << (n - 1)) {
        let mut blocks: Vec<Block> = Vec::with_capacity(n);
        let mut start = 0usize;
        for c in 0..(n - 1) {
            if mask & (1 << c) != 0 {
                blocks.push(Block { start, end: c + 1 });
                start = c + 1;
            }
        }
        blocks.push(Block { start, end: n });
        let mut feasible = true;
        let mut worst = 0.0f32;
        'blk: for b in &blocks {
            for p in b.start..b.end {
                for q in (p + 1)..b.end {
                    let v = s.get(q, p);
                    if v > eps || types[p] != types[q] {
                        feasible = false;
                        break 'blk;
                    }
                    if v > worst {
                        worst = v;
                    }
                }
            }
        }
        if !feasible {
            continue;
        }
        let cand = (blocks.len(), worst);
        let better = match best {
            None => true,
            Some((bc, bw)) => bc > cand.0 || (bc == cand.0 && bw > cand.1),
        };
        if better {
            best = Some(cand);
        }
    }
    best.expect("eps >= 0: singleton partition always feasible")
}

/// The number of blocks the TYPE-SPLIT forces: maximal runs of the same
/// layer type. `types[k]` = true for sliding-window layers.
/// Hard-infeasible pairs (distinct RoPE thetas, GDN vs attention) never
/// merge, so runs are the ACHIEVABLE floor — any kill-rule-style
/// feasibility measure reads against this, never against raw L (a
/// constraint-forced block count is structure, not absence-of-phase).
pub fn forced_min_blocks(types: &[bool]) -> usize {
    let mut runs = 0usize;
    let mut prev: Option<bool> = None;
    for &t in types {
        if prev != Some(t) {
            runs += 1;
        }
        prev = Some(t);
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s_of(n: usize, f: impl FnMut(usize, usize) -> f32) -> SMatrix {
        SMatrix::from_fn(n, f)
    }

    /// Deterministic LCG draws — total-order compares all the way, so
    /// exact-equality assertions are sound (the original
    /// `twt::synth::Lcg`'s constants; the promoted tests stay
    /// self-contained).
    fn lcg_draw(seed: u64) -> impl FnMut() -> f32 {
        let mut st = seed;
        move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    #[test]
    fn dp_recovers_one_planted_block() {
        // Layers 2,3,4 identical (distance 0); everything else ≈ 1.
        let s = s_of(6, |i, j| {
            let grp = |l: usize| matches!(l, 2..=4);
            if grp(i) && grp(j) { 0.0 } else { 1.0 }
        });
        let p = minmax_partition(&s, 0.1).unwrap();
        assert_eq!(
            p,
            vec![
                Block { start: 0, end: 1 },
                Block { start: 1, end: 2 },
                Block { start: 2, end: 5 },
                Block { start: 5, end: 6 },
            ]
        );
    }

    #[test]
    fn count_and_worst_match_brute_force_on_random_matrices() {
        // Deterministic LCG "random" S matrices — total-order compares all
        // the way, so exact agreement is the assertion.
        let mut draw = lcg_draw(0x243F_6A88_85A3_08D3);
        for n in [2usize, 5, 8, 11] {
            let s = s_of(n, |i, j| if i == j { 0.0 } else { draw() * 1.5 });
            for eps in [0.05f32, 0.3, 0.7, 1.2] {
                let p = minmax_partition(&s, eps).unwrap();
                let got = (p.len(), partition_worst(&s, &p));
                let want = brute_force_optimal(&s, eps);
                assert_eq!(got, want, "n={n} eps={eps}");
                for b in &p {
                    for x in b.start..b.end {
                        for y in (x + 1)..b.end {
                            assert!(s.get(y, x) <= eps, "constraint violated n={n} eps={eps}");
                        }
                    }
                }
            }
        }
    }

    /// A 7-point grid mirroring the lane's pre-registered sweep — the
    /// monotonicity property needs a fixed ascending ladder, not the
    /// lane's own pinned constants (those stay lane-side with the kill
    /// rule).
    const MONOTONE_EPS_GRID: [f32; 7] = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.20];

    #[test]
    fn m_monotone_non_increasing_in_eps() {
        let mut draw = lcg_draw(0xDEAD_BEEF_CAFE_F00D);
        let s = s_of(12, |i, j| if i == j { 0.0 } else { draw() });
        let mut prev = usize::MAX;
        for eps in MONOTONE_EPS_GRID {
            let m = minmax_partition(&s, eps).unwrap().len();
            assert!(m <= prev, "m grew at eps={eps}");
            prev = m;
        }
    }

    #[test]
    fn invalid_eps_refused() {
        let s = s_of(3, |_, _| 0.5);
        assert!(matches!(
            minmax_partition(&s, -0.1),
            Err(PartitionError::InvalidEps(_))
        ));
        assert!(matches!(
            minmax_partition(&s, f32::NAN),
            Err(PartitionError::InvalidEps(_))
        ));
        assert!(matches!(
            minmax_partition(&s, f32::INFINITY),
            Err(PartitionError::InvalidEps(_))
        ));
    }

    #[test]
    fn typed_partition_matches_typed_brute_force_and_enforces_types() {
        // Alternating types + identical layer clones across types: the
        // unconstrained DP merges everything, the typed DP may not —
        // exact agreement with the typed brute force is the assertion.
        let mut draw = lcg_draw(0x0B0B_5EED_1D1E_CAFE);
        for n in [2usize, 5, 8, 11] {
            let s = s_of(n, |i, j| if i == j { 0.0 } else { draw() * 1.5 });
            let types: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
            for eps in [0.05f32, 0.3, 0.7, 1.2] {
                let p = minmax_partition_typed(&s, eps, &types).unwrap();
                let got = (p.len(), partition_worst(&s, &p));
                let want = brute_force_optimal_typed(&s, eps, &types);
                assert_eq!(got, want, "typed n={n} eps={eps}");
                for b in &p {
                    for x in b.start..b.end {
                        assert_eq!(types[x], types[b.start], "mixed-type block n={n} eps={eps}");
                        for y in (x + 1)..b.end {
                            assert!(s.get(y, x) <= eps);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn typed_partition_grows_no_fewer_blocks_than_unconstrained() {
        // The type constraint only REMOVES feasible partitions, so m is
        // pointwise ≥ the unconstrained m at every ε.
        let mut draw = lcg_draw(0x1234_ABCD_5678_EF90);
        let n = 12usize;
        let s = s_of(n, |i, j| if i == j { 0.0 } else { draw() });
        let types: Vec<bool> = (0..n).map(|i| i % 4 != 3).collect();
        for eps in MONOTONE_EPS_GRID {
            let un = minmax_partition(&s, eps).unwrap().len();
            let ty = minmax_partition_typed(&s, eps, &types).unwrap().len();
            assert!(ty >= un, "typed m {ty} < unconstrained {un} at eps={eps}");
        }
    }

    #[test]
    fn typed_partition_m_monotone_non_increasing_in_eps() {
        let mut draw = lcg_draw(0xFEED_FACE_DADA_5501);
        let n = 12usize;
        let s = s_of(n, |i, j| if i == j { 0.0 } else { draw() });
        let types: Vec<bool> = (0..n).map(|i| i % 4 != 3).collect();
        let mut prev = usize::MAX;
        for eps in MONOTONE_EPS_GRID {
            let m = minmax_partition_typed(&s, eps, &types).unwrap().len();
            assert!(m <= prev, "typed m grew at eps={eps}");
            prev = m;
        }
    }

    #[test]
    fn typed_partition_refuses_length_mismatch() {
        let s = s_of(3, |_, _| 0.5);
        assert!(minmax_partition_typed(&s, 0.5, &[true, false]).is_err());
    }

    #[test]
    fn typed_partition_bonsai_rhythm_structure() {
        // The qwen35 interval-4 rhythm (3 DeltaNet runs + 1 attention) at
        // generous ε: the typed DP cannot cross types, so every attention
        // layer is its own singleton block — m = 16 singletons + merged
        // GDN triples, strictly less than L.
        let n = 16usize;
        let types: Vec<bool> = (0..n).map(|i| (i + 1) % 4 != 0).collect();
        let s = s_of(n, |i, j| {
            if i == j { 0.0 } else { 0.01 } // distances don't matter: the
            // type constraint, not ε, is what forces the cuts here
        });
        let p = minmax_partition_typed(&s, 0.1, &types).unwrap();
        for b in &p {
            for x in b.start..b.end {
                assert_eq!(types[x], types[b.start]);
            }
        }
        let attn_singletons = p.iter().filter(|b| !types[b.start] && b.len() == 1).count();
        assert_eq!(attn_singletons, 4, "every attention layer a singleton");
        assert!(p.len() < n, "merges happened: m {} < {n}", p.len());
    }

    #[test]
    fn forced_min_blocks_counts_runs() {
        // laya's G-S-S pattern over 7 layers: G S S G S S G → 5 runs.
        let types = [false, true, true, false, true, true, false];
        assert_eq!(forced_min_blocks(&types), 5);
        assert_eq!(forced_min_blocks(&[true; 4]), 1);
        assert_eq!(forced_min_blocks(&[]), 0);
    }

    #[test]
    fn pair_cosine_accum_basics() {
        let mut acc = PairCosineAccum::new();
        let v = [1.0f32, -2.0, 3.0, 0.5];
        for _ in 0..7 {
            acc.add(&v, &v);
        }
        // dot == na == nb exactly (same bytes both sides) → cos 1.0 exactly.
        assert_eq!(acc.distance(), 0.0);
        assert_eq!(acc.positions(), 7);
    }

    #[test]
    fn pair_cosine_accum_overflow_safe_via_f64() {
        // The cosine overflow class: 1e20 squares to 1e40, which overflows
        // f32 but is exact in f64 — a pure-f32 reduction yields inf/inf =
        // NaN, the f64 accumulator stays well-defined.
        let mut acc = PairCosineAccum::new();
        let a = [1e20f32, 0.0];
        let b = [2e20f32, 0.0];
        acc.add(&a, &b);
        assert_eq!(acc.distance(), 0.0);
    }

    #[test]
    fn pair_cosine_accum_orthogonal_opposite_zero() {
        let mut acc = PairCosineAccum::new();
        acc.add(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((acc.distance() - 1.0).abs() < 1e-6);

        let mut acc = PairCosineAccum::new();
        acc.add(&[1.0, 0.0], &[-1.0, 0.0]);
        assert!((acc.distance() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn pair_cosine_accum_conservative_one() {
        let mut acc = PairCosineAccum::new();
        acc.add(&[0.0, 0.0], &[1.0, 1.0]);
        assert_eq!(acc.distance(), 1.0);

        let acc = PairCosineAccum::new();
        assert_eq!(acc.distance(), 1.0, "empty accumulator is conservative");

        let mut acc = PairCosineAccum::new();
        acc.add(&[f32::NAN, 1.0], &[1.0, 1.0]);
        assert_eq!(acc.distance(), 1.0, "non-finite input is conservative");
    }

    #[test]
    fn smatrix_from_parts_validates_finiteness() {
        assert!(
            std::panic::catch_unwind(|| {
                let mut data = vec![0.0f32; 9];
                data[4] = f32::NAN;
                SMatrix::from_parts(3, data)
            })
            .is_err(),
            "a NaN distance must be refused at construction"
        );
    }
}
