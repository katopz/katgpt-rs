//! PRI / CDG / TaR — the paper's §6 evaluation metrics as pure functions.
//!
//! All three are computed from corpora of [`PrimitiveTransitionGraph`]s (no
//! per-call allocation beyond the result containers — they live in the warm
//! tier, not the decode hot path). Latent-to-raw bridges between these
//! metrics and per-NPC embeddings live in [`crate::bridge`].
//!
//! ## What's here vs `riir-ai`
//!
//! | Metric | Here (katgpt-rs) | riir-ai (private) |
//! |--------|-----------------|-------------------|
//! | PRI    | Full per-primitive score | — |
//! | CDG    | EMA of success-at-extrapolation | Actual eval harness |
//! | TaR    | **Public proxy**: Jaccard over motif multisets | Real TaR via `AnchorProfile.translate_priorities()` |
//!
//! The TaR exposed here is a *modelless proxy* — the real metric requires the
//! private cross-game transfer machinery in riir-ai. The proxy is enough for
//! the G1 GOAT gate (latency) and for relative-comparison benchmarks; G3
//! (correlation with real transfer) needs the riir-ai hookup in Phase 4.

use ahash::AHashMap;

use super::{PrimitiveKind, PrimitiveTransitionGraph};

// ── Primitive Reuse Index (PRI) ────────────────────────────────────────
//
// Hot-path optimization (Issue 035, `.contexts/optimization.md`). The
// primitive id space is bounded to `[0, 512)` by `PrimitiveKind::to_u32`
// (256 `UserDefined` + 256 `Composite`), and the corpus carries a small number
// of distinct `task_family_id`s (5 in the GOAT bench, typically a handful).
// We exploit both facts to replace the old nested
// `HashMap<PrimitiveKind, HashSet<u32>>` (≈4ms / 1K-trace corpus on std's
// SipHash) with a dense primitive×family **bit matrix** + a rolling-tag
// per-PTG dedup. Per-node work drops to one indexed array write; the only
// remaining hash work is the small unique-family pre-pass and the final
// scores map (both `AHashMap`). Measured: <100µs / 1K-trace corpus.

/// Size of the dense primitive id space — `0..256` user + `256..512` composite.
///
/// Kept in sync with `PrimitiveKind::{USER_DEFINED_MAX, COMPOSITE_MAX}`. If
/// those ever widen, this constant must widen too.
const PRIM_SPACE: usize = PrimitiveKind::COMPOSITE_MAX as usize; // 512

/// Per-primitive Primitive Reuse Index scores.
///
/// For each primitive, `PRI(p) = (distinct task families containing p) / (total task families)`.
/// Computed across an entire corpus of PTGs. Higher = more reused across task
/// families ⇒ more "general-purpose" (paper §6.1).
#[derive(Clone, Debug)]
pub struct PriScores(pub AHashMap<PrimitiveKind, f32>);

impl PriScores {
    /// Lookup the PRI for a primitive. Returns `0.0` if not present (the
    /// primitive was never observed in the corpus).
    #[inline]
    #[must_use]
    pub fn get(&self, p: PrimitiveKind) -> f32 {
        self.0.get(&p).copied().unwrap_or(0.0)
    }
}

/// Compute [`PriScores`] over a corpus.
///
/// **Complexity**: `O(N + P·F/64)` where `N = total nodes across all PTGs`,
/// `P = 512` (fixed primitive space), and `F = distinct task families`. The
/// per-node work is a single indexed array write into a primitive×family bit
/// matrix; no hashing on the hot path. GOAT G1 target `< 100µs` per 1K-trace
/// corpus (Issue 035).
///
/// # Algorithm
///
/// 1. **Unique-family pre-pass.** Collect distinct `task_family_id`s into a
///    small `AHashMap` and assign each a dense bit index `0..F`. (`F` is tiny
///    in practice — 5 in the GOAT bench — so the matrix is one or two words
///    wide per primitive.)
/// 2. **Bit matrix.** Allocate `PRIM_SPACE × ⌈F/64⌉` `u64`s, zero-init. Bit
///    `(prim_idx, family_idx)` is set iff primitive `prim_idx` appears in at
///    least one PTG of family `family_idx`.
/// 3. **Per-PTG dedup via rolling tag.** A stack `[u32; PRIM_SPACE]` tag array
///    + a wrapping generation counter lets us dedupe primitives within one
///    PTG without allocating (or clearing) a `HashSet` per PTG — touched
///    entries are detected by `tag[i] == cur_gen`.
/// 4. **Popcount per primitive.** Final PRI = `popcount(row) / F`.
///
/// # Arguments
///
/// - `corpus` — slice of PTGs to aggregate over.
///
/// # Returns
///
/// For each primitive in the corpus, `score = (count of distinct task families
/// containing it) / (total task families in corpus)`. An empty corpus yields
/// an empty map.
#[inline]
#[must_use]
pub fn compute_pri(corpus: &[PrimitiveTransitionGraph]) -> PriScores {
    // ── Phase 1: unique task families → dense bit indices ─────────────────
    // Small map (5 in the bench, ≤ corpus size in general). aHash keeps the
    // unavoidable hash work cheap.
    let mut family_to_bit: AHashMap<u32, u32> = AHashMap::with_capacity(64);
    for ptg in corpus {
        // `entry`-free fast path: most PTGs share a family with a neighbour.
        if !family_to_bit.contains_key(&ptg.task_family_id) {
            let idx = family_to_bit.len() as u32;
            family_to_bit.insert(ptg.task_family_id, idx);
        }
    }
    let n_families = family_to_bit.len();
    if n_families == 0 {
        return PriScores(AHashMap::new());
    }
    let total = n_families as f32;

    // ── Phase 2: primitive×family bit matrix ─────────────────────────────
    // One contiguous `Vec<u64>` so a primitive's row is a contiguous slice —
    // cache-friendly popcount at the end. `words_per_row` is 1 for ≤ 64
    // families (the common case), giving a 4KB matrix (512 × 8B).
    let words_per_row = (n_families + u64::BITS as usize - 1) / u64::BITS as usize;
    let mut bits: Vec<u64> = vec![0u64; PRIM_SPACE * words_per_row];

    // ── Phase 3: per-PTG dedup via rolling generation tag ─────────────────
    // `[u32; 512]` = 2KB on the stack. We touch only the slots a PTG visits
    // (~8 entries); the rest stay cold. The array is never cleared — `cur_gen`
    // monotonically increases and we treat any slot whose tag lags `cur_gen`
    // as "not seen this PTG". Wrap-around at `u32::MAX` is handled by clearing
    // the array once per `u32::MAX` PTGs (well above any realistic corpus).
    let mut seen_tag: [u32; PRIM_SPACE] = [0u32; PRIM_SPACE];
    let mut cur_gen: u32 = 0;

    for ptg in corpus {
        cur_gen = cur_gen.wrapping_add(1);
        // Generation 0 is reserved as "uninitialized" — skip it.
        if cur_gen == 0 {
            seen_tag.fill(0);
            cur_gen = 1;
        }

        // Look up the bit index once per PTG (aHash lookup, small map).
        let fam_bit = family_to_bit[&ptg.task_family_id] as usize;
        let word_idx = fam_bit / u64::BITS as usize;
        let bit_mask = 1u64 << (fam_bit % u64::BITS as usize);

        // Mark each distinct primitive this PTG uses. Same primitive twice in
        // one PTG ⇒ second occurrence sees `seen_tag[i] == cur_gen` and is
        // skipped (counts once per family, per the paper's definition).
        for node in &ptg.nodes {
            // `to_u32` already clamps to `[0, PRIM_SPACE)`, so the index is
            // always in-bounds for the matrix and tag array.
            let prim_idx = node.primitive.to_u32() as usize;
            // Branch-free would be `seen_tag[prim_idx] != cur_gen` then set;
            // the branch is correctly predicted (almost always taken for the
            // first hit) and the alternative would force a redundant write on
            // every node.
            if seen_tag[prim_idx] != cur_gen {
                seen_tag[prim_idx] = cur_gen;
                bits[prim_idx * words_per_row + word_idx] |= bit_mask;
            }
        }
    }

    // ── Phase 4: popcount per primitive → PRI scores ──────────────────────
    // Walk every primitive slot once. Most rows are all-zero (the primitive
    // never appeared) — skip them entirely so we don't pollute the scores map.
    let mut scores: AHashMap<PrimitiveKind, f32> =
        AHashMap::with_capacity(bits.len() / 8);
    for prim_idx in 0..PRIM_SPACE {
        let row = &bits[prim_idx * words_per_row..(prim_idx + 1) * words_per_row];
        let mut count: u32 = 0;
        for &word in row {
            count += word.count_ones();
        }
        if count > 0 {
            scores.insert(PrimitiveKind::from_u32(prim_idx as u32), count as f32 / total);
        }
    }
    PriScores(scores)
}

// ── Compositional Depth Generalization (CDG) ──────────────────────────────

/// Compositional Depth Generalization scalar per NPC.
///
/// `ema_success_at_extrapolation` is an EMA of `success_rate` observed at
/// depths strictly greater than the max training depth seen so far.
/// `max_train_depth_seen` is updated when the train corpus grows.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CdgScore {
    /// EMA of success rate at depths > max(train_depths).
    pub ema_success_at_extrapolation: f32,
    /// Maximum training depth observed (depth of deepest train PTG).
    pub max_train_depth_seen: u32,
}

/// EMA decay factor for the success-rate accumulator.
pub const CDG_ALPHA: f32 = 0.3;

/// Compute one step of CDG.
///
/// The CDG rule: only update the EMA if `test_depth > max(train_depths)`. If
/// `test_depth ≤ max(train_depths)` (interpolation, not extrapolation),
/// return `prev` unchanged. The first extrapolation observation (when `prev`
/// is `None` or `ema == 0`) initializes the EMA to the current `success_rate`.
///
/// # Arguments
///
/// - `train_depths` — depths (e.g. PTG node counts) of training instances.
/// - `test_depth` — depth of the just-evaluated test instance.
/// - `success_rate` — `[0,1]` success rate at `test_depth`.
/// - `prev` — previous [`CdgScore`] (use `None` for first observation).
#[inline]
#[must_use]
pub fn compute_cdg(
    train_depths: &[u32],
    test_depth: u32,
    success_rate: f32,
    prev: Option<&CdgScore>,
) -> CdgScore {
    let max_train = train_depths.iter().copied().max().unwrap_or(0);
    let base = prev.copied().unwrap_or(CdgScore {
        ema_success_at_extrapolation: 0.0,
        max_train_depth_seen: max_train,
    });

    // Only update if we're extrapolating beyond the train frontier.
    if test_depth <= base.max_train_depth_seen {
        return base;
    }

    // First-ever extrapolation ⇒ initialize the EMA to the observation.
    // Otherwise EMA update: `ema = α·new + (1−α)·ema`.
    let new_ema = if base.ema_success_at_extrapolation == 0.0 {
        success_rate.clamp(0.0, 1.0)
    } else {
        CDG_ALPHA * success_rate.clamp(0.0, 1.0)
            + (1.0 - CDG_ALPHA) * base.ema_success_at_extrapolation
    };

    CdgScore {
        ema_success_at_extrapolation: new_ema,
        max_train_depth_seen: base.max_train_depth_seen,
    }
}

// ── Transfer as Recomposition (TaR) — modelless proxy ─────────────────────

/// Compute the **proxy TaR score** between a baseline and a perturbed corpus.
///
/// Returns the Jaccard similarity over the **multiset** of motif subgraph
/// hashes in each corpus:
///
/// ```text
/// TaR = |M_base ∩ M_perturbed| / |M_base ∪ M_perturbed|
/// ```
///
/// where `M_X` is the multiset of canonical subgraph hashes discovered in
/// corpus `X` by a single round of motif enumeration (1-, 2-, and 3-node
/// chain motifs — see [`crate::motif`]). Range `[0, 1]`:
/// - `1.0` — every motif in baseline reappears with the same multiplicity in
///   perturbed (perfect recomposition).
/// - `0.0` — no motif overlap.
///
/// **Modelless proxy for the real TaR** (paper §6.3). The *real* TaR requires
/// `AnchorProfile.translate_priorities()` (riir-ai private IP) to score how
/// well the transfer mechanism preserves solver behavior across perturbation.
/// The proxy here scores only *structural motif overlap* — Phase 4 wires G3
/// (correlation between this proxy and measured transfer acceleration) when
/// the riir-ai benchmark traces become available.
#[inline]
#[must_use]
pub fn compute_tar_score(
    baseline_ptgs: &[PrimitiveTransitionGraph],
    perturbed_ptgs: &[PrimitiveTransitionGraph],
) -> f32 {
    let m_base = motif_multiset(baseline_ptgs);
    let m_pert = motif_multiset(perturbed_ptgs);
    jaccard_multiset(&m_base, &m_pert)
}

/// Build the multiset of all canonical subgraph hashes for a corpus.
///
/// Used by [`compute_tar_score`] — also reusable for any other comparison
/// (clustering, dedup, etc.). Public for that reason. Returns `AHashMap` for
/// speed on the warm tier (G3 path); callers that only `.iter()` / `.get()` /
/// `.len()` are unaffected by the choice of hasher.
#[inline]
#[must_use]
pub fn motif_multiset(corpus: &[PrimitiveTransitionGraph]) -> AHashMap<[u8; 32], u32> {
    let mut out: AHashMap<[u8; 32], u32> = AHashMap::new();
    for ptg in corpus {
        for hash in crate::closure::motif::enumerate_subgraph_hashes(ptg) {
            *out.entry(hash).or_insert(0) += 1;
        }
    }
    out
}

/// Jaccard similarity over the multiset of hashes.
///
/// For each key present in either map, the multiset intersection counts the
/// `min(count_a, count_b)` copies; the union counts `max(count_a, count_b)`.
/// Returns `0.0` if both maps are empty (no division by zero).
#[inline]
fn jaccard_multiset(
    a: &AHashMap<[u8; 32], u32>,
    b: &AHashMap<[u8; 32], u32>,
) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let mut inter: u64 = 0;
    let mut union: u64 = 0;
    // Walk the smaller map for intersection, the larger for union extras.
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    for (k, &c_small) in small {
        let c_large = large.get(k).copied().unwrap_or(0);
        inter += c_small.min(c_large) as u64;
        union += c_small.max(c_large) as u64;
    }
    // Add entries only in `large`.
    for (k, &c_large) in large {
        if !small.contains_key(k) {
            union += c_large as u64;
        }
    }
    if union == 0 {
        0.0
    } else {
        (inter as f32) / (union as f32)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{OperatorKind, PrimitiveKind, PtgRecorder};

    fn build_ptg(task_family: u32, primitives: &[u32]) -> PrimitiveTransitionGraph {
        let mut rec = PtgRecorder::new(task_family);
        let mut prev: Option<u32> = None;
        for (i, &p) in primitives.iter().enumerate() {
            let n = rec.enter(PrimitiveKind::UserDefined(p), i as u32, [p as u8; 32]);
            if let Some(p_id) = prev {
                rec.exit(p_id, n, OperatorKind::Sequence);
            }
            prev = Some(n);
        }
        rec.finish()
    }

    /// T3.6 — PRI on a synthetic 3-family / 5-primitive corpus.
    #[test]
    fn pri_on_synthetic_corpus() {
        // Family 0: primitives {0, 1, 2}
        // Family 1: primitives {0, 1, 3}
        // Family 2: primitives {0, 4}
        // Total 3 task families. PRI(0) = 3/3 = 1.0 (in all families).
        // PRI(1) = 2/3 ≈ 0.667. PRI(2) = 1/3 ≈ 0.333. PRI(3) = 1/3.
        // PRI(4) = 1/3.
        let corpus = vec![
            build_ptg(0, &[0, 1, 2]),
            build_ptg(1, &[0, 1, 3]),
            build_ptg(2, &[0, 4]),
        ];
        let scores = compute_pri(&corpus);
        assert!((scores.get(PrimitiveKind::UserDefined(0)) - 1.0).abs() < 1e-6);
        assert!(
            (scores.get(PrimitiveKind::UserDefined(1)) - 2.0 / 3.0).abs() < 1e-6,
            "PRI(1)={}",
            scores.get(PrimitiveKind::UserDefined(1))
        );
        assert!((scores.get(PrimitiveKind::UserDefined(2)) - 1.0 / 3.0).abs() < 1e-6);
        assert!((scores.get(PrimitiveKind::UserDefined(3)) - 1.0 / 3.0).abs() < 1e-6);
        assert!((scores.get(PrimitiveKind::UserDefined(4)) - 1.0 / 3.0).abs() < 1e-6);
        // Primitive never observed ⇒ 0.
        assert_eq!(scores.get(PrimitiveKind::UserDefined(99)), 0.0);
    }

    #[test]
    fn pri_on_empty_corpus_is_empty() {
        let scores = compute_pri(&[]);
        assert!(scores.0.is_empty());
    }

    #[test]
    fn pri_counts_primitive_once_per_family_even_if_repeated() {
        // Same primitive 5 times in one PTG, one family ⇒ PRI = 1/1 = 1.0.
        let corpus = vec![build_ptg(0, &[5, 5, 5, 5, 5])];
        let scores = compute_pri(&corpus);
        assert!((scores.get(PrimitiveKind::UserDefined(5)) - 1.0).abs() < 1e-6);
        assert_eq!(scores.0.len(), 1);
    }

    /// Edge case for the bit-matrix implementation (Issue 035): when the corpus
    /// spans more than 64 distinct task families, each primitive's row is
    /// multiple `u64` words wide. This test forces `words_per_row = 2` and
    /// verifies correctness across the word boundary.
    #[test]
    fn pri_handles_more_than_64_task_families() {
        // 70 task families, each containing primitive 0 in exactly one PTG.
        // Primitive 0 should have PRI = 70/70 = 1.0. Primitive 1 only appears
        // in the first family ⇒ PRI = 1/70.
        let mut corpus = Vec::new();
        for fam in 0..70u32 {
            corpus.push(build_ptg(fam, &[0]));
        }
        corpus.push(build_ptg(0, &[1]));
        let scores = compute_pri(&corpus);
        // 70 distinct task families.
        assert_eq!(scores.0.len(), 2, "only primitives 0 and 1 appeared");
        assert!((scores.get(PrimitiveKind::UserDefined(0)) - 1.0).abs() < 1e-6);
        assert!(
            (scores.get(PrimitiveKind::UserDefined(1)) - 1.0 / 70.0).abs() < 1e-6,
            "PRI(1) = {}",
            scores.get(PrimitiveKind::UserDefined(1))
        );
    }

    /// Edge case for the bit-matrix implementation (Issue 035): rolling-tag
    /// wrap-around. The `cur_gen` counter wraps at `u32::MAX`; we can't easily
    /// trigger that here, but we can confirm a single-family corpus with
    /// sparse primitive usage still scores correctly.
    #[test]
    fn pri_composite_primitive_round_trip_through_bit_matrix() {
        // Composite primitives occupy the `[256, 512)` slice of the matrix.
        // Verify they index correctly.
        let mut rec = PtgRecorder::new(7);
        let _ = rec.enter(PrimitiveKind::UserDefined(0), 0, [0u8; 32]);
        let ptg = rec.finish();
        let scores = compute_pri(&[ptg]);
        assert!((scores.get(PrimitiveKind::UserDefined(0)) - 1.0).abs() < 1e-6);
        // Composite primitive 0 was never observed ⇒ 0.
        assert_eq!(scores.get(PrimitiveKind::Composite(0)), 0.0);
    }

    /// CDG: only updates when test_depth exceeds max train depth.
    #[test]
    fn cdg_updates_only_on_extrapolation() {
        let train = [3u32, 5, 7];
        // Interpolation: test_depth = 4 ≤ 7 ⇒ no update, prev unchanged.
        let prev = CdgScore {
            ema_success_at_extrapolation: 0.5,
            max_train_depth_seen: 7,
        };
        let r = compute_cdg(&train, 4, 0.9, Some(&prev));
        assert_eq!(r, prev, "interpolation must not update EMA");

        // Extrapolation: test_depth = 10 > 7 ⇒ EMA update.
        // First call (prev EMA = 0.5 ≠ 0): new = 0.3*0.9 + 0.7*0.5 = 0.27 + 0.35 = 0.62.
        let r2 = compute_cdg(&train, 10, 0.9, Some(&prev));
        assert!(
            (r2.ema_success_at_extrapolation - 0.62).abs() < 1e-5,
            "got {}",
            r2.ema_success_at_extrapolation
        );

        // First-ever extrapolation (prev = None or ema == 0): initializes.
        let r3 = compute_cdg(&train, 10, 0.8, None);
        assert!((r3.ema_success_at_extrapolation - 0.8).abs() < 1e-6);
        assert_eq!(r3.max_train_depth_seen, 7);
    }

    #[test]
    fn cdg_empty_train_depths_treats_zero_as_max() {
        // No train data ⇒ max_train_depth_seen = 0 ⇒ any test_depth > 0 extrapolates.
        let r = compute_cdg(&[], 5, 0.7, None);
        assert!((r.ema_success_at_extrapolation - 0.7).abs() < 1e-6);
        assert_eq!(r.max_train_depth_seen, 0);
    }

    /// TaR: 100% same motifs ⇒ 1.0; 0% overlap ⇒ 0.0; partial ⇒ fractional.
    #[test]
    fn tar_score_jaccard_behavior() {
        // Same corpus on both sides ⇒ TaR = 1.0.
        let corpus = vec![
            build_ptg(0, &[0, 1, 2]),
            build_ptg(1, &[0, 1, 3]),
        ];
        let t = compute_tar_score(&corpus, &corpus);
        assert!((t - 1.0).abs() < 1e-6, "same corpus ⇒ 1.0, got {t}");

        // Disjoint corpora.
        let a = vec![build_ptg(0, &[0, 1])];
        let b = vec![build_ptg(0, &[3, 4])];
        let t2 = compute_tar_score(&a, &b);
        assert!((t2 - 0.0).abs() < 1e-6, "disjoint ⇒ 0.0, got {t2}");

        // Partial overlap.
        let a2 = vec![build_ptg(0, &[0, 1, 2])];
        let b2 = vec![build_ptg(0, &[0, 1, 2]), build_ptg(0, &[3, 4, 5])];
        let t3 = compute_tar_score(&a2, &b2);
        assert!(t3 > 0.0 && t3 < 1.0, "partial overlap ⇒ fractional, got {t3}");
    }

    #[test]
    fn tar_score_empty_both_sides_is_zero() {
        let t = compute_tar_score(&[], &[]);
        assert!((t - 0.0).abs() < 1e-6);
    }
}
