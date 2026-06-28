//! `MedianTopMAvailability` — the paper's load-bearing community-aggregation
//! rule.
//!
//! See [`crate::alien_sampler`] for the module-level doc and paper citation.
//!
//! Implements [`super::traits::AvailabilityScorer`]`<f32>` by computing the
//! median of the top-`m` cosine similarities between the candidate (treated as
//! a dense `&[f32]` embedding) and a precomputed `community_bank` of
//! community embeddings. This is the dual-encoder availability signal that
//! the paper proves (via ablation in §1.4) is *not* substitutable by a
//! density estimator — the median-of-top-m aggregation is load-bearing.
//!
//! # Top-m partial sort
//! For each candidate we compute `n_bank` cosines, then take the median of the
//! top-`m`. The top-`m` selection uses `select_nth_unstable_by` — an
//! `O(n_bank)` expected-time partial sort that avoids the `O(n log n)` cost
//! of a full sort. The median of the resulting `m` cosines is then taken via
//! a small fixed-size sort on the top-`m` slice.
//!
//! # Storage layout (Issue 002 C1 — SoA flat bank)
//! The bank is stored as a single flat `Vec<f32>` in row-major order
//! `(bank_len, bank_dim)`, not as `Vec<Vec<f32>>`. This makes the bank a
//! single contiguous allocation — L1-resident for the per-candidate sweep
//! (typical bank: 200 items × 16 dim = 12.8 KB, fits in 32 KB L1) — and a
//! prerequisite for SIMD auto-vectorization of the inner dot product.
//!
//! Inverse norms (`1.0 / L2_norm`) are precomputed at construction so the
//! per-cosine hot path uses two multiplies (`dot * inv_cand_norm *
//! inv_bank_norm`) instead of one divide. f32 divides pipeline poorly (~4
//! cycles throughput on NEON) vs multiplies (~2/cycle), and there are
//! `n_candidates × n_bank` normalizations per cycle.
//!
//! # Zero-alloc hot path
//! The [`Self::availability_embedded_with_scratch`] variant takes a
//! caller-owned cosine scratch buffer (`&mut [f32]` of length `>= n_bank`),
//! so the per-candidate hot path performs **no allocation**. The
//! [`AvailabilityScorer`] trait impl uses [`Self::availability_embedded`],
//! which lazily allocates / reuses an internal scratch buffer — convenient
//! for cold paths and tests, but not the recommended hot-path entry point.
//!
//! # Inner dot product + determinism
//! The inner dot product is [`dot_seq`] (sequential `s += a*b`). Issue 002
//! prototyped a 4-accumulator `mul_add` form (`dot_4acc`) for SIMD
//! auto-vectorization, but benchmarks showed it slower than the sequential
//! form without `target-cpu=native` (no autovec, added register pressure), so
//! it was removed. The sequential form preserves strict run-to-run
//! determinism (same input → same output, bit-identical). G3 closure for
//! Plan 311 was instead achieved by parallelizing the NPC loop in the GOAT
//! bench with rayon (see `.benchmarks/311_alien_sampler_goat.md`).
//!
//!
//! # Edge cases
//! - Empty bank → availability = 0.0 (no community signal; candidate is
//!   "neutral" w.r.t. the community). The sampler will z-score this constant
//!   to 0.
//! - `m` larger than bank size → falls back to `m = bank.len()` (effectively
//!   median over the whole bank).
//! - `m == 1` → returns the single top-1 cosine (max similarity).
//! - Zero-norm candidate or bank item → cosine = 0.0 (avoid divide-by-zero).
//! - Candidate longer/shorter than `bank_dim` → truncated to `bank_dim`
//!   (matches the pre-Issue-002 `zip` behavior; preserves bench semantics
//!   where candidates carry extra atom payload beyond the embedding).
//!
//! Reference: Plan 311 (T1.4), Research 293, arXiv:2603.01092 §1.4.
//! Issue 002 (SIMD + GEMM perf optimization, C1/C2/C3).

use super::traits::AvailabilityScorer;

/// Median-of-top-m cosine availability scorer.
///
/// Stores the bank as a single flat `Vec<f32>` (row-major `(bank_len,
/// bank_dim)`) plus precomputed inverse L2 norms. The flat layout is
/// L1-friendly and SIMD-ready (Issue 002 C1); the inverse norms turn the
/// per-cosine normalization into two multiplies instead of a divide (C2).
///
/// Public construction:
/// - [`Self::new`] — accepts `Vec<Vec<f32>>` for back-compat (flattens
///   internally). Used by the GOAT bench unchanged.
/// - [`Self::from_flat_bank`] — accepts a flat row-major `Vec<f32>` directly.
///   Zero-copy for hot-path callers that already maintain a flat bank.
/// - [`Self::push_bank_items`] — incremental append (Issue 002 C3). Appends
///   rows and updates norms incrementally; no full bank rebuild.
///
/// # Determinism
/// Bit-identical across runs for the same `(candidate, bank, m)` — no RNG, no
/// thread-local state. The partial sort is deterministic; the median is
/// deterministic; the dot product is deterministic (same FMA order every
/// time, even though that order differs from the pre-Issue-002 sequential
/// loop — see the module-level R1 relaxation note).
///
/// Reference: Plan 311 (T1.4), Issue 002 (C1/C2/C3).
pub struct MedianTopMAvailability {
    /// Flat row-major bank `(bank_len, bank_dim)`. Empty iff bank is empty.
    bank_flat: Vec<f32>,
    /// Embedding dimension (row stride). `0` iff bank is empty.
    bank_dim: usize,
    /// Raw L2 norms of each bank item (kept for diagnostics + the
    /// `new_precomputes_norms` white-box test; the hot path uses
    /// `bank_inv_norms` instead).
    bank_norms: Vec<f32>,
    /// `1.0 / L2_norm` for each bank item. `0.0` for zero-norm rows
    /// (sentinel: hot path skips those slots, writing cosine = 0.0).
    /// Precomputed at construction / mutation so the hot path is multiply-only.
    bank_inv_norms: Vec<f32>,
    m: usize,
    /// Reusable cosine scratch for the convenience `availability_embedded`
    /// entry point. The zero-alloc hot-path variant
    /// (`availability_embedded_with_scratch`) does not touch this.
    scratch: Vec<f32>,
}

impl MedianTopMAvailability {
    /// Construct from an owned bank + `m`.
    ///
    /// **Validation** (panics — one-time setup):
    /// - All bank items must have the same length (the embedding dimension).
    ///   Mixed-dim banks are a wiring bug.
    /// - Bank items must be finite (no NaN / inf embeddings — they corrupt
    ///   every cosine).
    /// - `m >= 1` (m=0 is meaningless; the median of zero items is undefined).
    ///
    /// Bank norms + inverse norms are precomputed here so the per-candidate
    /// hot path is a single dot + two multiplies per bank item.
    ///
    /// Reference: Plan 311 T1.4, Issue 002 C1.
    #[must_use]
    pub fn new(community_bank: Vec<Vec<f32>>, m: usize) -> Self {
        assert!(
            m >= 1,
            "MedianTopMAvailability::new: m must be >= 1, got {m}"
        );
        let dim = community_bank.first().map(Vec::len).unwrap_or(0);
        for (i, item) in community_bank.iter().enumerate() {
            assert_eq!(
                item.len(),
                dim,
                "MedianTopMAvailability::new: bank item {i} has len {} but bank dim is {dim} (all items must share the embedding dimension)",
                item.len()
            );
            for (j, &v) in item.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "MedianTopMAvailability::new: bank item {i}[{j}] is not finite (got {v})"
                );
            }
        }
        // Flatten to row-major SoA.
        let bank_flat: Vec<f32> = if dim == 0 {
            Vec::new()
        } else {
            let cap = community_bank
                .len()
                .checked_mul(dim)
                .expect("bank size overflow");
            let mut flat = Vec::with_capacity(cap);
            for item in &community_bank {
                flat.extend_from_slice(item);
            }
            flat
        };
        let (bank_norms, bank_inv_norms) = Self::compute_norms(&bank_flat, dim);
        // Cosine scratch sized to the bank; reused by `availability_embedded`.
        let scratch = vec![0.0_f32; community_bank.len()];
        Self {
            bank_flat,
            bank_dim: dim,
            bank_norms,
            bank_inv_norms,
            m,
            scratch,
        }
    }

    /// Construct with paper-default `m = 10`.
    #[must_use]
    pub fn with_paper_default_m(community_bank: Vec<Vec<f32>>) -> Self {
        Self::new(community_bank, 10)
    }

    /// Hot-path constructor: take ownership of an already-flat row-major bank.
    ///
    /// `bank_flat.len()` must be a multiple of `dim`; each consecutive `dim`
    /// floats form one bank item. Skips the `Vec<Vec<f32>>` intermediate
    /// allocation entirely — useful when the caller already maintains a flat
    /// bank (e.g. an `SoA` zone bank in riir-ai's CGSP runtime).
    ///
    /// **Validation** (panics):
    /// - `bank_flat.len() % dim == 0` (else rows are truncated / misaligned).
    /// - `dim == 0` iff `bank_flat` is empty.
    /// - All entries finite.
    /// - `m >= 1`.
    ///
    /// Reference: Issue 002 C1/C3.
    #[must_use]
    pub fn from_flat_bank(bank_flat: Vec<f32>, dim: usize, m: usize) -> Self {
        assert!(
            m >= 1,
            "MedianTopMAvailability::from_flat_bank: m must be >= 1, got {m}"
        );
        if dim == 0 {
            assert!(
                bank_flat.is_empty(),
                "from_flat_bank: dim=0 but bank_flat is non-empty ({} floats)",
                bank_flat.len()
            );
        } else {
            assert_eq!(
                bank_flat.len() % dim,
                0,
                "from_flat_bank: bank_flat.len() ({}) must be a multiple of dim ({dim})",
                bank_flat.len()
            );
            for (j, &v) in bank_flat.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "from_flat_bank: bank_flat[{j}] is not finite (got {v})"
                );
            }
        }
        let bank_len = if dim == 0 { 0 } else { bank_flat.len() / dim };
        let (bank_norms, bank_inv_norms) = Self::compute_norms(&bank_flat, dim);
        let scratch = vec![0.0_f32; bank_len];
        Self {
            bank_flat,
            bank_dim: dim,
            bank_norms,
            bank_inv_norms,
            m,
            scratch,
        }
    }

    /// Incremental append: push new rows into the bank without rebuilding.
    ///
    /// Each item in `items` must have length `== self.bank_dim` (or the bank
    /// must be empty with `bank_dim == 0`, in which case the first push
    /// establishes `bank_dim`). Norms are extended incrementally — O(items ×
    /// dim), not O((bank + items) × dim).
    ///
    /// For the GOAT bench's "append as NPCs emit" pattern, this removes the
    /// periodic full-rebuild cliff (Issue 002 C3). The bench itself is
    /// unchanged (still uses `new` with a clone), but consumers that adopt
    /// this method skip the clone + re-norm cost.
    ///
    /// # Panics
    /// - Items must match the bank's embedding dimension (or establish it on
    ///   first push into an empty bank).
    /// - Items must be finite.
    pub fn push_bank_items(&mut self, items: &[&[f32]]) {
        if items.is_empty() {
            return;
        }
        // Establish dim on first push into an empty bank.
        if self.bank_dim == 0 && self.bank_flat.is_empty() {
            self.bank_dim = items[0].len();
            assert!(
                self.bank_dim > 0,
                "push_bank_items: first item has dim 0 (cannot infer embedding dimension)"
            );
        }
        for (i, item) in items.iter().enumerate() {
            assert_eq!(
                item.len(),
                self.bank_dim,
                "push_bank_items: item {i} has len {} but bank dim is {}",
                item.len(),
                self.bank_dim
            );
            for (j, &v) in item.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "push_bank_items: item {i}[{j}] is not finite (got {v})"
                );
            }
        }
        // Append rows flat.
        let extra = items
            .len()
            .checked_mul(self.bank_dim)
            .expect("bank size overflow");
        self.bank_flat.reserve(extra);
        for item in items {
            self.bank_flat.extend_from_slice(item);
        }
        // Extend norms incrementally.
        for item in items {
            let mut s = 0.0_f32;
            for &v in *item {
                s += v * v;
            }
            let norm = s.sqrt();
            self.bank_norms.push(norm);
            // 0.0 sentinel for zero-norm (hot path skips).
            self.bank_inv_norms
                .push(if norm > 0.0 { norm.recip() } else { 0.0 });
        }
        // Grow the convenience scratch to match.
        self.scratch
            .resize(self.bank_flat.len() / self.bank_dim.max(1), 0.0);
    }

    /// Force a full recompute of the cached norms.
    ///
    /// Only needed if the caller mutated `bank_flat` directly via unsafe /
    /// interior mutability (the public API keeps norms in sync automatically).
    /// Provided for auditability and freeze/thaw restore paths.
    ///
    /// Reference: Issue 002 C3.
    pub fn invalidate_norms(&mut self) {
        let (norms, inv_norms) = Self::compute_norms(&self.bank_flat, self.bank_dim);
        self.bank_norms = norms;
        self.bank_inv_norms = inv_norms;
    }

    /// Compute (raw norms, inverse norms) from a flat bank.
    ///
    /// Inverse norm is `0.0` for zero-norm rows (sentinel — hot path skips
    /// those slots, writing cosine = 0.0). Avoids `inf` from `1.0 / 0.0`.
    #[inline]
    fn compute_norms(bank_flat: &[f32], dim: usize) -> (Vec<f32>, Vec<f32>) {
        if dim == 0 || bank_flat.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let bank_len = bank_flat.len() / dim;
        let mut norms = Vec::with_capacity(bank_len);
        let mut inv_norms = Vec::with_capacity(bank_len);
        for i in 0..bank_len {
            let row = &bank_flat[i * dim..(i + 1) * dim];
            let mut s = 0.0_f32;
            for &v in row {
                s += v * v;
            }
            let norm = s.sqrt();
            norms.push(norm);
            inv_norms.push(if norm > 0.0 { norm.recip() } else { 0.0 });
        }
        (norms, inv_norms)
    }

    /// Flat row-major view of the community bank `(bank_len, bank_dim).
    ///
    /// Issue 002 C1: previously returned `&[Vec<f32>]` (AoS); now returns
    /// the flat SoA slice. Use [`Self::bank_dim`] for the row stride and
    /// [`Self::bank_row`] for per-row access.
    #[inline]
    #[must_use]
    pub fn bank(&self) -> &[f32] {
        &self.bank_flat
    }

    /// Alias for [`Self::bank`] — explicit flat-slice accessor.
    #[inline]
    #[must_use]
    pub fn bank_flat(&self) -> &[f32] {
        &self.bank_flat
    }

    /// Embedding dimension (row stride). `0` for an empty bank.
    #[inline]
    #[must_use]
    pub fn bank_dim(&self) -> usize {
        self.bank_dim
    }

    /// Borrow row `i` of the flat bank. `None` if `i >= bank_len`.
    ///
    /// Zero-allocation view into [`Self::bank_flat`].
    #[inline]
    #[must_use]
    pub fn bank_row(&self, i: usize) -> Option<&[f32]> {
        if self.bank_dim == 0 || i >= self.bank_flat.len() / self.bank_dim {
            return None;
        }
        Some(&self.bank_flat[i * self.bank_dim..(i + 1) * self.bank_dim])
    }

    /// The configured `m` (top-m count).
    #[inline]
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Number of items in the community bank.
    #[inline]
    #[must_use]
    pub fn bank_len(&self) -> usize {
        if self.bank_dim == 0 {
            0
        } else {
            self.bank_flat.len() / self.bank_dim
        }
    }

    /// Embedding dimension (length of each bank item). `0` for an empty bank.
    /// Alias for [`Self::bank_dim`] (back-compat name).
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.bank_dim
    }

    /// Compute median-of-top-m cosine availability for an embedded candidate.
    ///
    /// This is the **convenience** entry point — it uses an internal scratch
    /// buffer (`self.scratch`) and is therefore not allocation-free across
    /// the first call (the buffer is sized at construction; subsequent calls
    /// reuse it). For the hot path, prefer
    /// [`Self::availability_embedded_with_scratch`].
    ///
    /// Returns `0.0` for an empty bank.
    ///
    /// Reference: Plan 311 T1.4.
    #[inline]
    pub fn availability_embedded(&mut self, candidate: &[f32]) -> f32 {
        // Borrow self.scratch mutably for the duration of the call. Safe
        // because we don't re-enter (no recursion through self).
        let scratch = core::mem::take(&mut self.scratch);
        let mut scratch = scratch;
        let out = self.availability_embedded_with_scratch(candidate, &mut scratch);
        self.scratch = scratch;
        out
    }

    /// Zero-alloc hot-path variant: caller owns the cosine scratch buffer.
    ///
    /// `cosine_scratch` must have length `>= self.bank_len()`. It's
    /// overwritten with per-bank-item cosine similarities during the
    /// computation and then partially sorted in place to extract the top-m
    /// and compute the median. The caller can inspect it after the call to
    /// see all cosines (pre-sort).
    ///
    /// # Panics
    /// Debug builds assert `cosine_scratch.len() >= self.bank_len()`. Release
    /// builds trust the caller (hot-path contract).
    ///
    /// Reference: Plan 311 T1.4, Issue 002 C1/C2 (SoA + SIMD dot).
    #[inline]
    pub fn availability_embedded_with_scratch(
        &self,
        candidate: &[f32],
        cosine_scratch: &mut [f32],
    ) -> f32 {
        let dim = self.bank_dim;
        let n_bank = self.bank_len();
        if n_bank == 0 || dim == 0 {
            return 0.0;
        }
        debug_assert_eq!(
            cosine_scratch.len(),
            n_bank,
            "availability_embedded_with_scratch: cosine_scratch must have len == bank_len ({n_bank})"
        );

        // Candidate slice for the dot product. Truncate to bank_dim (matches
        // the pre-Issue-002 `zip` semantics — candidates may carry extra
        // atom payload beyond the embedding, e.g. the Phase 2 bench's 4×16
        // atoms where only the first 16 are the embedding).
        //
        // `dot_seq` uses `zip` so it naturally truncates to the shorter of
        // cand_slice / row — no per-item branch needed for short candidates.
        let cand_slice = if candidate.len() >= dim {
            &candidate[..dim]
        } else {
            candidate
        };

        // Candidate L2 norm (single pass).
        let mut cand_norm_sq = 0.0_f32;
        for &v in cand_slice {
            cand_norm_sq += v * v;
        }
        if cand_norm_sq == 0.0 {
            // Zero-norm candidate has no direction; cosine is undefined.
            // Treat as availability 0 (neutral) and zero the scratch so the
            // downstream median sees a clean constant pool.
            for s in &mut cosine_scratch[..n_bank] {
                *s = 0.0;
            }
            return 0.0;
        }
        // 1 divide amortized across all n_bank items (vs n_bank divides in
        // the pre-Issue-002 `dot / (cand_norm * bank_norm)` form).
        let inv_cand_norm = cand_norm_sq.sqrt().recip();

        // Pass 1: cosine similarity against each bank item.
        // cosine(a, b) = (a · b) * inv_cand_norm * inv_bank_norm
        //
        // Branch-free inner loop (rust-optimize skill rule):
        // - Zero-norm rows have inv_bank_norm = 0.0, and their dot product is
        //   0.0 (zero vector · anything = 0), so `dot * inv_cand_norm * 0.0 = 0.0`
        //   — no explicit zero-norm branch needed; the multiply handles it.
        // - The three-way `zip` lets LLVM elide per-iteration bounds checks
        //   on `cosine_scratch` / `bank_inv_norms` (lengths all == n_bank).
        // - `chunks_exact(dim)` avoids per-iteration index arithmetic; `zip`
        //   inside `dot_seq` handles short candidates.
        for ((row, inv_norm), out) in self
            .bank_flat
            .chunks_exact(dim)
            .zip(self.bank_inv_norms.iter())
            .zip(cosine_scratch.iter_mut())
        {
            let dot = dot_seq(cand_slice, row);
            *out = dot * inv_cand_norm * inv_norm;
        }

        // Pass 2: median of top-m.
        median_of_top_m(&mut cosine_scratch[..n_bank], self.m)
    }

    /// Batch availability scoring: fills `out[i]` with the availability of
    /// `candidates[i]`, reusing a single cosine scratch across all candidates.
    ///
    /// This is the **hot-path** entry point for ranking passes: one scratch
    /// allocation amortized across the whole candidate pool, instead of one
    /// per candidate in the trait path. Pair with
    /// [`super::sampler::AlienSampler::rank_precomputed`] to skip the
    /// per-candidate trait allocation entirely.
    ///
    /// `out` must have length `>= candidates.len()`; `cosine_scratch` must
    /// have length `>= self.bank_len()`. Only the first `candidates.len()`
    /// entries of `out` are written.
    ///
    /// Reference: Plan 311 T4.4.
    #[inline]
    pub fn availability_batch(
        &self,
        candidates: &[Vec<f32>],
        out: &mut [f32],
        cosine_scratch: &mut [f32],
    ) {
        let n_bank = self.bank_len();
        debug_assert_eq!(
            cosine_scratch.len(),
            n_bank,
            "availability_batch: cosine_scratch must have len == bank_len ({n_bank})"
        );
        for (i, cand) in candidates.iter().enumerate() {
            out[i] = self.availability_embedded_with_scratch(cand, cosine_scratch);
        }
    }
}

impl AvailabilityScorer<f32> for MedianTopMAvailability {
    /// Trait-compatible entry point. Allocates a cosine scratch buffer per call.
    ///
    /// This is the **cold-path** convenience entry point: it matches the
    /// `&self` trait signature but performs one `Vec` allocation per call.
    /// Hot-path callers should use [`Self::availability_embedded_with_scratch`]
    /// (zero-alloc, caller-owned scratch) or [`Self::availability_embedded`]
    /// (`&mut self`, reuses internal scratch) instead.
    ///
    /// The trait is `&self`, so we cannot reuse the internal `scratch` field
    /// here without interior mutability (`RefCell` / `UnsafeCell`) — and the
    /// open primitive deliberately avoids those for determinism + audit. The
    /// per-call allocation is the price of trait compatibility.
    fn availability(&self, atoms: &[f32]) -> f32 {
        let mut scratch = vec![0.0_f32; self.bank_len()];
        self.availability_embedded_with_scratch(atoms, &mut scratch)
    }
}

/// Dot product — simple sequential `s += a*b`.
///
/// Benchmark-validated choice (rust-optimize skill: "measure after each
/// change — some optimizations make things worse"). The 4-accumulator
/// unrolled form (`dot_4acc`, Issue 002 C2) was tried and is ~35% SLOWER on
/// this target (M3 Max, generic target without `target-cpu=native`) — it has
/// since been removed; the sequential `zip` form is the only shipped kernel.
/// The simple `zip` form lets LLVM extract ~2× from OoO overlap without the
/// register-pressure cost of 4 accumulators. G3 closure for Plan 311 was
/// instead achieved by rayon-parallelizing the NPC loop in the GOAT bench.
///
/// `a.len()` must be `<= b.len()`; consumes `a.len()` elements.
#[inline]
fn dot_seq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

/// Compute the median of the top-`m` values in `xs` (in place).
///
/// After the call, `xs` is partially sorted (top-`m` at the tail, sorted
/// ascending within the tail). Returns the median of those top-m values.
///
/// - `xs.len() == 0` → returns `0.0` (no data).
/// - `m >= xs.len()` → returns the median of all of `xs`.
/// - `m == 1` → returns `xs.max()`.
/// - Odd top-m count → middle element.
/// - Even top-m count → mean of the two middle elements.
///
/// Uses `select_nth_unstable_by` for `O(n)` expected-time top-m extraction
/// (the load-bearing perf trick per AGENTS.md "Prefer `match` … partial
/// sort").
#[inline]
fn median_of_top_m(xs: &mut [f32], m: usize) -> f32 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    let effective_m = m.min(n);
    if effective_m == 1 {
        // Top-1 = max.
        return xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }

    // Partial sort: place the top-m (largest) at the tail of xs.
    // select_nth_unstable_by(k, cmp) partitions so that xs[k] is in its
    // sorted position, everything before is "less" by cmp, everything after
    // is "greater". We want the top-m at the tail, so we partition at
    // k = n - effective_m with ascending cmp → tail [n-m, n) holds the
    // largest m.
    let k = n - effective_m;
    xs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    let top_m = &mut xs[k..];
    // Sort the top-m slice ascending so we can pick the median index cleanly.
    // `total_cmp` is branch-free (single instruction on most ISAs) vs
    // `partial_cmp().unwrap_or()` which has a branch — rust-optimize skill rule.
    top_m.sort_by(|a, b| a.total_cmp(b));

    // Median of top_m (length = effective_m):
    // - odd:  top_m[effective_m / 2]
    // - even: (top_m[effective_m/2 - 1] + top_m[effective_m/2]) / 2
    let mid = effective_m / 2;
    if effective_m % 2 == 1 {
        top_m[mid]
    } else {
        // Even: average of the two middle elements. Deterministic (no FMA
        // reassociation here; just one add + one multiply).
        (top_m[mid - 1] + top_m[mid]) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction validation ─────────────────────────────────────────────

    #[test]
    fn new_empty_bank_ok() {
        // Empty bank is allowed; availability returns 0.0.
        let s = MedianTopMAvailability::new(vec![], 10);
        assert_eq!(s.bank_len(), 0);
        assert_eq!(s.dim(), 0);
        assert_eq!(s.bank_flat().len(), 0);
    }

    #[test]
    fn new_precomputes_norms() {
        let bank = vec![
            vec![3.0, 4.0], // norm = 5
            vec![1.0, 0.0], // norm = 1
        ];
        let s = MedianTopMAvailability::new(bank, 2);
        assert!((s.bank_norms[0] - 5.0).abs() < 1e-6);
        assert!((s.bank_norms[1] - 1.0).abs() < 1e-6);
        // Inverse norms (Issue 002 C2 hot-path storage).
        assert!((s.bank_inv_norms[0] - 0.2).abs() < 1e-6); // 1/5
        assert!((s.bank_inv_norms[1] - 1.0).abs() < 1e-6); // 1/1
    }

    #[test]
    fn from_flat_bank_matches_new() {
        // The flat constructor must produce identical results to `new`.
        let bank: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let i = i as f32;
                vec![i * 0.1, (i + 1.0) * 0.1, (i + 2.0) * 0.1]
            })
            .collect();
        let dim = 3;
        let mut flat = Vec::new();
        for row in &bank {
            flat.extend_from_slice(row);
        }
        let s_a = MedianTopMAvailability::new(bank.clone(), 4);
        let s_b = MedianTopMAvailability::from_flat_bank(flat, dim, 4);
        assert_eq!(s_a.bank_flat(), s_b.bank_flat());
        assert_eq!(s_a.bank_dim(), s_b.bank_dim());
        assert_eq!(s_a.bank_len(), s_b.bank_len());
        assert_eq!(s_a.bank_norms, s_b.bank_norms);
        assert_eq!(s_a.bank_inv_norms, s_b.bank_inv_norms);

        // Hot path produces identical availability on a sample candidate.
        let cand = [0.5, 0.5, 0.5];
        let mut sa = vec![0.0; s_a.bank_len()];
        let mut sb = vec![0.0; s_b.bank_len()];
        let va = s_a.availability_embedded_with_scratch(&cand, &mut sa);
        let vb = s_b.availability_embedded_with_scratch(&cand, &mut sb);
        assert!(
            (va - vb).abs() < 1e-6,
            "from_flat_bank mismatch: {va} vs {vb}"
        );
    }

    #[test]
    fn push_bank_items_extends_incrementally() {
        // Issue 002 C3: incremental append updates norms without a full rebuild.
        let mut s = MedianTopMAvailability::new(vec![vec![3.0, 4.0]], 2);
        assert_eq!(s.bank_len(), 1);
        assert!((s.bank_norms[0] - 5.0).abs() < 1e-6);

        // Push two more rows.
        s.push_bank_items(&[&[1.0, 0.0], &[0.0, 5.0]]);
        assert_eq!(s.bank_len(), 3);
        assert_eq!(s.bank_dim(), 2);
        assert!((s.bank_norms[1] - 1.0).abs() < 1e-6);
        assert!((s.bank_norms[2] - 5.0).abs() < 1e-6);
        assert!((s.bank_inv_norms[1] - 1.0).abs() < 1e-6);
        assert!((s.bank_inv_norms[2] - 0.2).abs() < 1e-6);

        // Matches a fresh `new` with the same full bank.
        let fresh =
            MedianTopMAvailability::new(vec![vec![3.0, 4.0], vec![1.0, 0.0], vec![0.0, 5.0]], 2);
        assert_eq!(s.bank_flat(), fresh.bank_flat());
        assert_eq!(s.bank_norms, fresh.bank_norms);
        assert_eq!(s.bank_inv_norms, fresh.bank_inv_norms);
    }

    #[test]
    fn push_bank_items_establishes_dim_on_empty_bank() {
        let mut s = MedianTopMAvailability::new(vec![], 3);
        assert_eq!(s.bank_dim(), 0);
        s.push_bank_items(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        assert_eq!(s.bank_dim(), 3);
        assert_eq!(s.bank_len(), 2);
    }

    #[test]
    fn invalidate_norms_after_external_mutation_pattern() {
        // Issue 002 C3: full recompute path. We can't externally mutate
        // bank_flat through the public API, so we reconstruct with `new`
        // then re-invalidate as a sanity check (idempotent).
        let mut s = MedianTopMAvailability::new(vec![vec![3.0, 4.0]], 2);
        let norms_before = s.bank_norms.clone();
        let inv_before = s.bank_inv_norms.clone();
        s.invalidate_norms();
        assert_eq!(s.bank_norms, norms_before);
        assert_eq!(s.bank_inv_norms, inv_before);
    }

    #[test]
    #[should_panic(expected = "m must be >= 1")]
    fn new_rejects_zero_m() {
        let _ = MedianTopMAvailability::new(vec![vec![1.0]], 0);
    }

    #[test]
    #[should_panic(expected = "all items must share the embedding dimension")]
    fn new_rejects_mixed_dim_bank() {
        let _ = MedianTopMAvailability::new(vec![vec![1.0, 2.0], vec![1.0]], 1);
    }

    #[test]
    #[should_panic(expected = "is not finite")]
    fn new_rejects_nan_in_bank() {
        let _ = MedianTopMAvailability::new(vec![vec![f32::NAN]], 1);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of dim")]
    fn from_flat_bank_rejects_misaligned() {
        // 7 floats, dim=3 → not a multiple.
        let _ = MedianTopMAvailability::from_flat_bank(vec![1.0; 7], 3, 2);
    }

    #[test]
    #[should_panic(expected = "dim=0 but bank_flat is non-empty")]
    fn from_flat_bank_rejects_zero_dim_nonempty() {
        let _ = MedianTopMAvailability::from_flat_bank(vec![1.0], 0, 2);
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn empty_bank_returns_zero() {
        let mut s = MedianTopMAvailability::new(vec![], 10);
        let v = s.availability_embedded(&[1.0, 2.0, 3.0]);
        assert!(v.abs() < 1e-6);
    }

    #[test]
    fn zero_norm_candidate_returns_zero() {
        let mut s = MedianTopMAvailability::new(vec![vec![1.0, 0.0]], 1);
        let v = s.availability_embedded(&[0.0, 0.0]);
        assert!(v.abs() < 1e-6);
    }

    #[test]
    fn zero_norm_bank_item_returns_zero_for_that_slot() {
        // Bank has one zero-norm item and one normal item. m=1 picks the max,
        // which should be the normal item's cosine.
        let mut s = MedianTopMAvailability::new(vec![vec![0.0, 0.0], vec![1.0, 0.0]], 1);
        let v = s.availability_embedded(&[1.0, 0.0]);
        // cosine with [1,0] = 1.0, cosine with [0,0] = 0.0. Top-1 = max = 1.0.
        assert!((v - 1.0).abs() < 1e-6);
    }

    // ── Top-m fallback ──────────────────────────────────────────────────────

    #[test]
    fn top1_fallback_returns_max_cosine() {
        // m=1 with bank larger than 1 → returns max cosine.
        let bank = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0], // normalized: [0.707, 0.707]
        ];
        let mut s = MedianTopMAvailability::new(bank, 1);
        // Candidate [1, 0]: cosines = [1.0, 0.0, 0.707]. Top-1 = 1.0.
        let v = s.availability_embedded(&[1.0, 0.0]);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn m_larger_than_bank_falls_back_to_bank_size() {
        // m=100 but bank only has 3 items → effective m = 3.
        let bank = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.6, 0.8], // norm=1
        ];
        let mut s = MedianTopMAvailability::new(bank, 100);
        // Candidate [1, 0]: cosines = [1.0, 0.0, 0.6]. Median of all 3 = 0.6.
        let v = s.availability_embedded(&[1.0, 0.0]);
        assert!((v - 0.6).abs() < 1e-6);
    }

    // ── Paper default m=10 ─────────────────────────────────────────────────

    #[test]
    fn paper_default_m10_bank_of_50_median_of_top10() {
        // Bank of 50 items, m=10. We construct deterministic cosines and
        // verify the median of the top-10.
        //
        // We build a bank where each item has a known cosine to a fixed
        // candidate, then check the median.
        let candidate = vec![1.0, 0.0];
        // Build bank items as unit vectors at known angles so cosines are
        // known. We'll just use [cos θ, sin θ] for θ in [0, π/2).
        // cosine([1,0], [cos θ, sin θ]) = cos θ.
        // We want 50 cosines; pick θ_i = (i / 50) * (π/2).
        let mut bank: Vec<Vec<f32>> = Vec::with_capacity(50);
        let mut cosines: Vec<f32> = Vec::with_capacity(50);
        for i in 0..50 {
            let theta = (i as f32 / 50.0) * (core::f32::consts::PI / 2.0);
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            bank.push(vec![cos_t, sin_t]);
            cosines.push(cos_t); // cosine similarity with [1, 0]
        }
        let mut s = MedianTopMAvailability::new(bank, 10);
        let got = s.availability_embedded(&candidate);
        // Top-10 cosines = the 10 largest values in `cosines`.
        cosines.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let top10 = &cosines[40..]; // 10 largest
        // Median of 10 (even) = average of top10[4] and top10[5].
        let expected = (top10[4] + top10[5]) * 0.5;
        assert!(
            (got - expected).abs() < 1e-5,
            "paper default m=10 median mismatch: got {got}, expected {expected}"
        );
    }

    // ── Invariance ─────────────────────────────────────────────────────────

    #[test]
    fn invariant_to_bank_permutation() {
        // Shuffling the bank should not change the result (median of top-m is
        // order-independent).
        let candidate = vec![1.0, 0.0];
        let bank_a = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.43589], // ~cos 25°
            vec![0.5, 0.86603], // ~cos 60°
            vec![0.0, 1.0],
        ];
        // Permute: move item 0 to the end.
        let mut bank_b = bank_a.clone();
        let first = bank_b.remove(0);
        bank_b.push(first);
        let mut sa = MedianTopMAvailability::new(bank_a, 2);
        let mut sb = MedianTopMAvailability::new(bank_b, 2);
        let va = sa.availability_embedded(&candidate);
        let vb = sb.availability_embedded(&candidate);
        assert!(
            (va - vb).abs() < 1e-6,
            "bank permutation changed result: {va} vs {vb}"
        );
    }

    #[test]
    fn determinism_same_inputs_same_output() {
        let candidate = vec![0.7, 0.7, 0.7];
        let bank = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ];
        let s = MedianTopMAvailability::new(bank, 3);
        let mut scratch1 = vec![0.0; 5];
        let mut scratch2 = vec![0.0; 5];
        let v1 = s.availability_embedded_with_scratch(&candidate, &mut scratch1);
        let v2 = s.availability_embedded_with_scratch(&candidate, &mut scratch2);
        assert!((v1 - v2).abs() < 1e-7);
    }

    // ── Trait impl ─────────────────────────────────────────────────────────

    #[test]
    fn trait_impl_matches_direct_call() {
        use super::super::traits::AvailabilityScorer;
        let bank = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let s = MedianTopMAvailability::new(bank, 1);
        // Trait method (allocates scratch internally).
        let trait_v = s.availability(&[1.0, 0.0]);
        // Direct call with explicit scratch.
        let mut scratch = vec![0.0; 2];
        let direct_v = s.availability_embedded_with_scratch(&[1.0, 0.0], &mut scratch);
        assert!((trait_v - direct_v).abs() < 1e-6);
    }

    // ── bank_row / bank_flat accessors ─────────────────────────────────────

    #[test]
    fn bank_row_returns_correct_slices() {
        let bank = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let s = MedianTopMAvailability::new(bank, 2);
        assert_eq!(s.bank_row(0), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(s.bank_row(1), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(s.bank_row(2), Some(&[7.0, 8.0, 9.0][..]));
        assert_eq!(s.bank_row(3), None);
    }

    #[test]
    fn bank_flat_is_row_major() {
        let bank = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let s = MedianTopMAvailability::new(bank, 1);
        assert_eq!(s.bank_flat(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(s.bank_dim(), 2);
    }

    // ── median_of_top_m unit tests ─────────────────────────────────────────

    #[test]
    fn median_of_top_m_empty_returns_zero() {
        let mut xs: Vec<f32> = vec![];
        assert_eq!(median_of_top_m(&mut xs, 5), 0.0);
    }

    #[test]
    fn median_of_top_m_top1_returns_max() {
        let mut xs = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        assert!((median_of_top_m(&mut xs, 1) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn median_of_top_m_odd_count() {
        // top-3 of [0.1, 0.9, 0.5, 0.3, 0.7] = [0.5, 0.7, 0.9] (sorted).
        // median (odd, 3) = middle = 0.7.
        let mut xs = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        assert!((median_of_top_m(&mut xs, 3) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn median_of_top_m_even_count() {
        // top-4 of [0.1, 0.9, 0.5, 0.3, 0.7, 0.8] = [0.5, 0.7, 0.8, 0.9] (sorted).
        // median (even, 4) = (0.7 + 0.8) / 2 = 0.75.
        let mut xs = vec![0.1, 0.9, 0.5, 0.3, 0.7, 0.8];
        let got = median_of_top_m(&mut xs, 4);
        assert!((got - 0.75).abs() < 1e-6, "even-count median: got {got}");
    }

    #[test]
    fn median_of_top_m_m_larger_than_n_uses_n() {
        // m=100, n=3 → effective m=3 → median of all 3.
        let mut xs = vec![1.0, 2.0, 3.0];
        // sorted: [1, 2, 3], median = 2.
        assert!((median_of_top_m(&mut xs, 100) - 2.0).abs() < 1e-6);
    }
}
