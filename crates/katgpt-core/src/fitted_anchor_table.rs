//! Fitted anchor tables — the shared streaming table-builder substrate
//! (Issue 882 P0 + Issue 883 P0; Research 586 §"one axpy" + Research 587
//! §2.1, the F1 fusion: one substrate, two consumers).
//!
//! Both issues fit **closed-form per-key mean tables on frozen
//! observations** — no gradient, one streaming pass:
//!
//! - **882 (Q-side)**: the anchor `ā` is the mean over a set of observed
//!   vectors (query rows or corpus rows) — a one-row table; the consumer
//!   is `differential_anchor::correct_query` (`q̂ = q − λ·ā`).
//! - **883 (V-side)**: `E_l[s] = mean(V_t − K_t | s_t = s)` per
//!   (layer, token) plus the value-mean twin `E^V_l[s] = mean(V_t | s_t =
//!   s)` — one table per (layer, signal); the consumers are the
//!   token-mean-removed V-quant (P1), the fitted K=V+ retrofit (P2), and
//!   the V-cache-halving reconstruction (P3).
//!
//! # The estimator family (shared laws)
//!
//! - **Streaming sufficient statistics**: per-key `n` + `Σx` (f32 rows —
//!   they ARE the table product), grand `N`, `Σx`, `Σx²` (f64 — the
//!   one-way-ANOVA arithmetic must survive 10⁷–10⁹ observations).
//! - **R² dashboard** (law of total variance, exact from those stats):
//!   `SS_total,d = Σx²_d − (Σx_d)²/N`, `SS_between,d = Σ_keys
//!   S_{k,d}²/n_k − (Σx_d)²/N` — the per-dim token-explained fraction is
//!   `ρ_d = SSB_d/SST_d`, the aggregate is `Σ_d SSB_d / Σ_d SST_d`
//!   (variance-weighted — the quantity quant-error scaling is monotone
//!   in). f32 row sums are promoted to f64 only at finalize; observe is
//!   plain f32 axpy (documented precision envelope: relative accumulation
//!   error grows ~√N·ε on adversarial cancellation — a dashboard
//!   instrument, not a committed-value path).
//! - **James–Stein-flavored shrinkage** (Research 587 §2.1):
//!   `E_λ[s] = n_s/(n_s+λ) · mean_s` — Zipf-tail-safe (rare tokens shrink
//!   toward 0, never steer); λ by direct evaluation on held-out
//!   fixtures, **never GD**; `λ=0` is the plain mean exactly.
//! - **Tail-lump lower bound**: when only a top-K key set is tracked,
//!   untracked observations accumulate into the reserved tail row. The
//!   partition {tracked keys ∪ tail-lump} is complete, so SSB over it is
//!   a **strict lower bound** on true per-key between-variance (lumping
//!   underestimates the tail's internal between-variance, never
//!   overestimates it) while SS_total stays exact over all observations —
//!   an honest `ρ_lower` go/no-go with a measured coverage share.
//!
//! # Memory shape (the P0 calibration budget)
//!
//! `rows × width` f32 sums + u64 counts, allocated once at construction.
//! The 883 gemma-2-2b budget (top-K=8192 tokens, kv_dim=1024, 26 layers,
//! 2 signals) is ~1.7 GiB; `observe` is a single row axpy + 2·width f64
//! grand updates — **alloc-free by construction** (G4; asserted by
//! `bench_886`'s counting allocator over a full calibration pass).
//!
//! Offline report methods (`r_squared`, `sorted_counts_desc`,
//! `coverage_curve`) allocate — they run once per calibration, never in
//! the observe loop.

/// James–Stein shrinkage factor `n/(n+λ)` — the Zipf-tail control shared
/// by every fitted-table consumer. `λ = 0` ⇒ factor exactly 1 (the plain
/// mean; 883's G3 bit-identity class). `const fn` so table builders can
/// precompute schedules at compile time.
#[inline]
#[must_use]
pub const fn shrinkage(n: u64, lambda: f32) -> f32 {
    // n ≥ 0 always; λ < 0 is a caller bug — the divide yields a factor > 1
    // which callers can see in their own validation, never a panic.
    let n_f = n as f32;
    n_f / (n_f + lambda)
}

/// Per-key streaming mean accumulator with one-way-ANOVA sufficient
/// statistics (the R² dashboard's arithmetic) and James–Stein shrinkage
/// at finalize.
///
/// `rows` tracked keys + one reserved **tail row** (index `rows`) that
/// lumps every untracked observation — see the module doc's lower-bound
/// law. Width is uniform (the table product); per-head ρ reads slice the
/// per-dim report at head boundaries.
///
/// One allocation set at construction; `observe*` are alloc-free.
pub struct StreamingMeanTable {
    width: usize,
    counts: Vec<u64>,
    sums: Vec<f32>, // (rows + 1) × width, flat; row `rows` = tail lump
    grand_n: u64,
    grand_sum: Vec<f64>,
    grand_sq: Vec<f64>,
}

impl StreamingMeanTable {
    /// Allocate a `rows`-key table of `width`-wide rows (plus the tail
    /// lump row). `rows == 0` is legal — a tail-only table still yields
    /// exact grand statistics and an `r_squared` of 0 with coverage 0.
    #[must_use]
    pub fn new(rows: usize, width: usize) -> Self {
        let total_rows = rows + 1;
        Self {
            width,
            counts: vec![0; total_rows],
            sums: vec![0.0; total_rows * width],
            grand_n: 0,
            grand_sum: vec![0.0; width],
            grand_sq: vec![0.0; width],
        }
    }

    /// Number of tracked keys (excluding the tail row).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.counts.len() - 1
    }

    /// Row width (the table product's dimensionality).
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Total observations seen (tracked + tail).
    #[must_use]
    pub fn n_total(&self) -> u64 {
        self.grand_n
    }

    /// Observations in row `key` (the tail row reports its lump count).
    #[must_use]
    pub fn count(&self, key: usize) -> u64 {
        self.counts[key.min(self.counts.len() - 1)]
    }

    /// Observe vector `x` under tracked key `key` — one row axpy + grand
    /// updates, alloc-free. Panics in debug if `key >= rows()` or the
    /// width mismatches (a calibration wiring bug must be loud, not a
    /// silent wrong-row accumulate — 883 trap 1).
    pub fn observe(&mut self, key: usize, x: &[f32]) {
        debug_assert!(key < self.rows(), "key {key} outside tracked rows");
        self.observe_row(key, x);
    }

    /// Observe vector `x` under the tail lump (untracked key) — grand
    /// updates + tail row accumulate only.
    pub fn observe_tail(&mut self, x: &[f32]) {
        self.observe_row(self.counts.len() - 1, x);
    }

    fn observe_row(&mut self, row: usize, x: &[f32]) {
        debug_assert_eq!(x.len(), self.width, "observation width mismatch");
        let base = row * self.width;
        let mut nonfinite = false;
        for (j, &v) in x.iter().enumerate() {
            self.sums[base + j] += v;
            let v64 = f64::from(v);
            self.grand_sum[j] += v64;
            self.grand_sq[j] += v64 * v64;
            nonfinite |= !v.is_finite();
        }
        if nonfinite {
            // Poison control: a non-finite input corrupts BOTH the row and
            // the grand stats. Refuse loudly — a calibration pass feeding
            // NaN V/K taps is a tap-point bug (883 trap 1), never data.
            panic!("fitted_anchor_table: non-finite observation");
        }
        self.counts[row] += 1;
        self.grand_n += 1;
    }

    /// Plain per-key mean into caller scratch — zeros when the row has no
    /// observations (an empty key has no direction, not NaN).
    pub fn mean_into(&self, key: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.width);
        let n = self.count(key);
        if n == 0 {
            out.fill(0.0);
            return;
        }
        let inv = 1.0 / n as f32;
        let base = key * self.width;
        for (j, o) in out.iter_mut().enumerate() {
            *o = self.sums[base + j] * inv;
        }
    }

    /// James–Stein-shrunk per-key mean `n/(n+λ)·mean` into caller scratch
    /// (λ=0 ⇒ the plain mean, bit-identical to [`mean_into`]'s value).
    pub fn shrunk_into(&self, key: usize, lambda: f32, out: &mut [f32]) {
        let n = self.count(key);
        if n == 0 {
            out.fill(0.0);
            return;
        }
        if lambda == 0.0 {
            self.mean_into(key, out);
            return;
        }
        let f = shrinkage(n, lambda);
        let base = key * self.width;
        for (j, o) in out.iter_mut().enumerate() {
            *o = self.sums[base + j] * (1.0 / n as f32) * f;
        }
    }

    /// Exact one-way-ANOVA report (the R² dashboard): per-dim and
    /// variance-weighted-aggregate token-explained fractions, computed in
    /// f64 over the accumulated sufficient statistics.
    ///
    /// **Lower-bound semantics** (module doc): between-sums cover tracked
    /// keys + the tail lump, so a partial-tracking calibration reports a
    /// strict lower bound on the true per-key ρ; `coverage` discloses how
    /// much of N the tracked keys hold. Roundoff can push SSB marginally
    /// negative or above SST on degenerate dims — clamped to [0, SST] and
    /// counted in `degenerate_dims` (SST ≈ 0 ⇒ dim contributes nothing).
    #[must_use]
    pub fn r_squared(&self) -> R2Report {
        let width = self.width;
        let n = self.grand_n;
        let mut per_dim = vec![0.0f32; width];
        let mut ssb_dim = vec![0.0f64; width];
        let mut sst_dim = vec![0.0f64; width];
        let mut degenerate = 0usize;

        if n == 0 {
            return R2Report {
                width,
                n: 0,
                tracked_rows: self.rows(),
                tracked_mass: 0.0,
                per_dim,
                ss_between_dim: ssb_dim,
                ss_total_dim: sst_dim,
                ss_between_sum: 0.0,
                ss_total_sum: 0.0,
                aggregate: 0.0,
                degenerate_dims: 0,
                empty: true,
            };
        }

        let nf = n as f64;
        let rows_incl_tail = self.counts.len();

        for d in 0..width {
            let correction = self.grand_sum[d] * self.grand_sum[d] / nf;
            let sst = (self.grand_sq[d] - correction).max(0.0);
            let mut ssb = -correction;
            for row in 0..rows_incl_tail {
                let c = self.counts[row];
                if c > 0 {
                    let s = f64::from(self.sums[row * width + d]);
                    ssb += s * s / c as f64;
                }
            }
            ssb = ssb.clamp(0.0, sst);
            ssb_dim[d] = ssb;
            sst_dim[d] = sst;
            per_dim[d] = if sst > f64::EPSILON * nf.max(1.0) {
                (ssb / sst) as f32
            } else {
                degenerate += 1;
                0.0
            };
        }

        let ssb_sum: f64 = ssb_dim.iter().sum();
        let sst_sum: f64 = sst_dim.iter().sum();
        let tracked_n: u64 = self.counts[..self.rows()].iter().sum();
        R2Report {
            width,
            n,
            tracked_rows: self.rows(),
            tracked_mass: tracked_n as f64 / nf,
            per_dim,
            ss_between_dim: ssb_dim,
            ss_total_dim: sst_dim,
            ss_between_sum: ssb_sum,
            ss_total_sum: sst_sum,
            aggregate: if sst_sum > 0.0 {
                (ssb_sum / sst_sum) as f32
            } else {
                0.0
            },
            degenerate_dims: degenerate,
            empty: false,
        }
    }

    /// Tracked-key observation counts, descending — the Zipf shape read
    /// (offline, allocating). The tail lump is NOT included; its count is
    /// `count(rows())`.
    #[must_use]
    pub fn sorted_counts_desc(&self) -> Vec<u64> {
        let mut c: Vec<u64> = self.counts[..self.rows()].to_vec();
        c.sort_unstable_by(|a, b| b.cmp(a));
        c
    }

    /// Zipf coverage curve: element `k-1` = fraction of ALL observations
    /// held by the top-`k` tracked keys (monotone ↑, ends at
    /// `tracked_mass`). The storage dial for 883's top-K residency
    /// (`P(K) = b_w·L·K·d_v` picks K off this curve).
    #[must_use]
    pub fn coverage_curve(&self) -> Vec<f64> {
        if self.grand_n == 0 {
            return Vec::new();
        }
        let sorted = self.sorted_counts_desc();
        let mut out = Vec::with_capacity(sorted.len());
        let mut acc = 0u64;
        for &c in &sorted {
            acc += c;
            out.push(acc as f64 / self.grand_n as f64);
        }
        out
    }
}

/// Offline R² dashboard report — the 883 P0 go/no-go artifact.
///
/// `aggregate` is the variance-weighted token-explained fraction
/// `Σ_d SSB_d / Σ_d SST_d`; per-head ρ slices `per_dim` (or the exact f64
/// sums) at head boundaries — the P1 quant-error prediction is monotone
/// in the per-head aggregate, not the pooled scalar.
#[derive(Debug, Clone)]
pub struct R2Report {
    pub width: usize,
    pub n: u64,
    pub tracked_rows: usize,
    /// Tracked-key share of all observations (the lower-bound disclosure).
    pub tracked_mass: f64,
    /// Per-dim SSB/SST (0 on degenerate dims).
    pub per_dim: Vec<f32>,
    /// Per-dim between-sums (f64, post-clamp).
    pub ss_between_dim: Vec<f64>,
    /// Per-dim total-sums (f64).
    pub ss_total_dim: Vec<f64>,
    pub ss_between_sum: f64,
    pub ss_total_sum: f64,
    /// Variance-weighted aggregate ρ (0 when empty/degenerate).
    pub aggregate: f32,
    /// Dims with SST ≈ 0 (constant dims — excluded from ρ, counted here).
    pub degenerate_dims: usize,
    /// True when no observations were seen at all (a calibration wiring
    /// bug must read as EMPTY, not as a clean zero — the frontier-report
    /// law).
    pub empty: bool,
}

impl R2Report {
    /// Variance-weighted ρ over a dim slice `[start, end)` — the per-head
    /// (or per-block) aggregate from exact f64 sums.
    #[must_use]
    pub fn aggregate_over(&self, start: usize, end: usize) -> f32 {
        let end = end.min(self.width);
        let start = start.min(end);
        let ssb: f64 = self.ss_between_dim[start..end].iter().sum();
        let sst: f64 = self.ss_total_dim[start..end].iter().sum();
        if sst > 0.0 {
            (ssb / sst) as f32
        } else {
            0.0
        }
    }
}

/// Per-layer triplet of streaming tables (V, K, V−K) — one
/// [`StreamingMeanTable`] per signal, `top_k` tracked token rows + the
/// tail lump per table. The two-fixture shared shape: gemma-2 (26 GQA
/// layers, one triplet per layer) and Kimi-K3 (the 2 MLA layers only —
/// KDA layers carry no KV cache and are documented fixture-class nulls).
pub struct VkLayerTables {
    /// `E^V_l[s] = mean(V_t | s_t = s)` — the value-mean table (P1's
    /// mean-removed-quant input).
    pub v: StreamingMeanTable,
    /// `mean(K_t | s_t = s)` — the K-mean table (the K=V+ baseline read).
    pub k: StreamingMeanTable,
    /// `E_l[s] = mean(V_t − K_t | s_t = s)` — the retrofit table (P2/P3's
    /// fitted residual).
    pub vk: StreamingMeanTable,
}

/// The full layered calibration state: one [`VkLayerTables`] triplet per
/// tapped layer + the token→row map (`u32::MAX` = untracked → tail) built
/// from the corpus frequency pre-pass, plus the owned V−K scratch (so the
/// observe path never borrows caller buffers).
///
/// Promoted from riir-infer's gemma-2 harness at the Kimi-K3 fixture's
/// landing (883 P0, Bench 889): one builder, two fixtures — the layer count
/// and row width are caller facts (`n_layer` = the TAPPED layers only;
/// `layer` in [`observe_layer`](Self::observe_layer) is the table index,
/// which the harness maps onto its model-layer list).
pub struct LayeredVkCalibration {
    /// One triplet per TAPPED layer (harness-indexed, not model-indexed).
    pub layers: Vec<VkLayerTables>,
    /// vocab-size map: token id → tracked row index, or `u32::MAX` (tail).
    pub row_of_token: Vec<u32>,
    /// Corpus frequency counts per token id (the Zipf shape read).
    pub token_counts: Vec<u64>,
    /// Tracked key count (the top-K residency dial).
    pub top_k: usize,
    vk_scratch: Vec<f32>,
}

impl LayeredVkCalibration {
    /// Build tables for `n_layer` tapped layers of `kv_dim`-wide rows,
    /// tracking the `top_k` most frequent tokens of `token_counts` (the
    /// frequency pre-pass output; ties broken by token id for determinism).
    #[must_use]
    pub fn from_counts(
        n_layer: usize,
        kv_dim: usize,
        token_counts: Vec<u64>,
        top_k: usize,
    ) -> Self {
        let vocab = token_counts.len();
        let mut order: Vec<u32> = (0..vocab as u32)
            .filter(|&t| token_counts[t as usize] > 0)
            .collect();
        order.sort_unstable_by(|a, b| {
            token_counts[*b as usize]
                .cmp(&token_counts[*a as usize])
                .then_with(|| a.cmp(b))
        });
        let top_k = top_k.min(order.len());
        let mut row_of_token = vec![u32::MAX; vocab];
        for (row, &tok) in order[..top_k].iter().enumerate() {
            row_of_token[tok as usize] = row as u32;
        }
        let layers = (0..n_layer)
            .map(|_| VkLayerTables {
                v: StreamingMeanTable::new(top_k, kv_dim),
                k: StreamingMeanTable::new(top_k, kv_dim),
                vk: StreamingMeanTable::new(top_k, kv_dim),
            })
            .collect();
        Self {
            layers,
            row_of_token,
            token_counts,
            top_k,
            vk_scratch: vec![0.0; kv_dim],
        }
    }

    /// Observe one token's tap pair for tapped-layer index `layer`.
    /// `k_vec` MUST be the tap-point-law K (where the cache path would
    /// consume it); `v_vec` the corresponding V. The V−K residual is
    /// computed into the owned scratch — the retrofit table's exact future
    /// input. Alloc-free (the substrate's G4 law).
    pub fn observe_layer(&mut self, layer: usize, token: usize, k_vec: &[f32], v_vec: &[f32]) {
        let kvd = self.vk_scratch.len();
        for i in 0..kvd {
            self.vk_scratch[i] = v_vec[i] - k_vec[i];
        }
        let t = &mut self.layers[layer];
        let row = self.row_of_token[token] as usize;
        if row != u32::MAX as usize {
            t.v.observe(row, v_vec);
            t.k.observe(row, k_vec);
            t.vk.observe(row, &self.vk_scratch);
        } else {
            t.v.observe_tail(v_vec);
            t.k.observe_tail(k_vec);
            t.vk.observe_tail(&self.vk_scratch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact ANOVA on a hand-computed fixture: two keys perfectly
    /// determine their rows ⇒ ρ = 1 on every dim and in aggregate.
    #[test]
    fn r_squared_is_one_when_keys_perfectly_determine_values() {
        let mut t = StreamingMeanTable::new(2, 2);
        t.observe(0, &[1.0, 0.0]);
        t.observe(0, &[1.0, 0.0]);
        t.observe(1, &[0.0, 1.0]);
        t.observe(1, &[0.0, 1.0]);
        t.observe(1, &[0.0, 1.0]);
        let r = t.r_squared();
        assert!(!r.empty);
        assert_eq!(r.n, 5);
        assert_eq!(r.tracked_rows, 2);
        assert!((r.tracked_mass - 1.0).abs() < 1e-12);
        // dim0: values 1,1,0,0,0 → SST = 2 − 4/5 = 1.2, SSB = 2 − 0.8 = 1.2
        // dim1: values 0,0,1,1,1 → SST = 3 − 9/5 = 1.2, SSB = 3 − 1.8 = 1.2
        assert!((r.ss_total_dim[0] - 1.2).abs() < 1e-9);
        assert!((r.ss_between_dim[0] - 1.2).abs() < 1e-9);
        assert!((r.ss_total_dim[1] - 1.2).abs() < 1e-9);
        assert!((r.ss_between_dim[1] - 1.2).abs() < 1e-9);
        assert!((r.per_dim[0] - 1.0).abs() < 1e-6);
        assert!((r.per_dim[1] - 1.0).abs() < 1e-6);
        assert!((r.aggregate - 1.0).abs() < 1e-6);
        assert_eq!(r.degenerate_dims, 0);
    }

    /// The null fixture: keys carry no information (identical value
    /// multisets per key) ⇒ ρ = 0 exactly on every dim.
    #[test]
    fn r_squared_is_zero_when_keys_carry_nothing() {
        let mut t = StreamingMeanTable::new(2, 2);
        t.observe(0, &[2.0, 0.0]);
        t.observe(0, &[0.0, 2.0]);
        t.observe(1, &[0.0, 2.0]);
        t.observe(1, &[2.0, 0.0]);
        let r = t.r_squared();
        assert!((r.ss_total_dim[0] - 4.0).abs() < 1e-9);
        assert!(r.ss_between_dim[0].abs() < 1e-9, "SSB dim0 = {}", r.ss_between_dim[0]);
        assert!(r.ss_between_dim[1].abs() < 1e-9);
        assert!(r.aggregate.abs() < 1e-6);
    }

    /// Partial tracking + tail lump ⇒ reported ρ is a strict lower bound
    /// of the full-tracking ρ (the lump underestimates the tail's
    /// between-variance), while SST stays exact.
    #[test]
    fn tail_lump_reports_a_strict_lower_bound() {
        // Three real "keys" with internally-homogeneous value clusters;
        // only key 0 is tracked in the lumped arm.
        let full = {
            let mut t = StreamingMeanTable::new(3, 1);
            t.observe(0, &[10.0]);
            t.observe(0, &[10.0]);
            t.observe(1, &[-10.0]);
            t.observe(1, &[-10.0]);
            t.observe(2, &[-30.0]);
            t.observe(2, &[-30.0]);
            t.r_squared()
        };
        let lumped = {
            let mut t = StreamingMeanTable::new(1, 1);
            t.observe(0, &[10.0]);
            t.observe(0, &[10.0]);
            t.observe_tail(&[-10.0]);
            t.observe_tail(&[-10.0]);
            t.observe_tail(&[-30.0]);
            t.observe_tail(&[-30.0]);
            t.r_squared()
            };
            // Full tracking: three internally-homogeneous keys ⇒ ρ = 1.
            // Lumped: the tail fuses keys {−10,−10} and {−30,−30} into one
            // group of mean −20 ⇒ their internal split is invisible ⇒ ρ
            // strictly below the full-tracking value, above 0, with SST exact.
            assert!((lumped.ss_total_dim[0] - full.ss_total_dim[0]).abs() < 1e-9);
            assert!(lumped.aggregate > 0.0 && lumped.aggregate < full.aggregate);
            assert!(lumped.tracked_mass < 1.0);
    }

    /// Shrinkage law: factor 1 at λ=0, monotone ↓ in λ, 0 as λ→∞; the
    /// shrunk row equals `factor × mean` bit-for-bit at λ=0.
    #[test]
    fn shrinkage_law_and_bit_identity_at_zero() {
        assert_eq!(shrinkage(10, 0.0), 1.0);
        assert!(shrinkage(10, 1.0) < 1.0);
        assert!(shrinkage(10, 100.0) < shrinkage(10, 1.0));
        assert!(shrinkage(10, 1e9) < 1e-3);

        let mut t = StreamingMeanTable::new(1, 3);
        t.observe(0, &[1.0, -2.0, 4.0]);
        t.observe(0, &[3.0, 0.0, 6.0]);
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        t.mean_into(0, &mut a);
        t.shrunk_into(0, 0.0, &mut b);
        assert_eq!(a, b, "λ=0 must be the plain mean bit-identically");
        // n=2, λ=2 ⇒ factor 1/2.
        t.shrunk_into(0, 2.0, &mut b);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((y - 0.5 * x).abs() < 1e-6);
        }
    }

    /// Coverage curve: monotone, ends at tracked_mass, and the empty-key
    /// edge reads as empty data, not NaN.
    #[test]
    fn coverage_curve_shape() {
        let mut t = StreamingMeanTable::new(3, 1);
        t.observe(0, &[1.0]);
        t.observe(0, &[1.0]);
        t.observe(0, &[1.0]);
        t.observe(1, &[1.0]);
        t.observe_tail(&[1.0]);
        let cov = t.coverage_curve();
        assert_eq!(cov.len(), 3);
        assert!((cov[0] - 0.6).abs() < 1e-12);
        assert!((cov[1] - 0.8).abs() < 1e-12);
        assert!((cov[2] - 0.8).abs() < 1e-12, "empty key adds nothing");
        assert!(cov.is_windows_sorted());
        let empty = StreamingMeanTable::new(0, 4);
        assert!(empty.coverage_curve().is_empty());
        assert!(empty.r_squared().empty);
    }

    /// Aggregate-over-slice (the per-head read) uses the exact f64 sums,
    /// not the rounded per-dim f32 ratios.
    #[test]
    fn aggregate_over_slice_matches_manual_anova() {
        let mut t = StreamingMeanTable::new(2, 4);
        // head A (dims 0..2): perfectly key-determined; head B: pure noise.
        t.observe(0, &[1.0, 1.0, 0.0, 5.0]);
        t.observe(0, &[1.0, 1.0, 0.0, 7.0]);
        t.observe(1, &[-1.0, -1.0, 0.0, 5.0]);
        t.observe(1, &[-1.0, -1.0, 0.0, 7.0]);
        let r = t.r_squared();
        assert!((r.aggregate_over(0, 2) - 1.0).abs() < 1e-6);
        assert!(r.aggregate_over(2, 4) < 1e-6);
    }

    /// Non-finite input refuses loudly (a tap-point bug, not data).
    #[test]
    #[should_panic(expected = "non-finite observation")]
    fn nonfinite_observation_panics() {
        let mut t = StreamingMeanTable::new(1, 2);
        t.observe(0, &[0.0, f32::NAN]);
    }

    /// Precision envelope: 10⁵ observations of a constant + alternating
    /// signal keep the f32 row mean within 1e-4 (the documented plain-axpy
    /// envelope — sufficient for a dashboard, never a committed value).
    #[test]
    fn f32_accumulation_envelope_at_1e5_observations() {
        let mut t = StreamingMeanTable::new(1, 2);
        for i in 0..100_000u32 {
            let v = (i % 7) as f32 * 0.25;
            t.observe(0, &[1.0, v]);
        }
        let mut m = [0.0f32; 2];
        t.mean_into(0, &mut m);
        assert!((m[0] - 1.0).abs() < 1e-4);
        let true_mean = 0.25 * (0.0 + 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0) / 7.0;
        assert!((m[1] - true_mean).abs() < 1e-3, "m[1] = {}", m[1]);
    }

    /// Layered builder: frequency-ranked row map (ties by id), untracked
    /// → tail, per-layer table triplets independent, V−K residual exact.
    #[test]
    fn layered_calibration_row_map_and_residual() {
        // vocab of 5; counts make token 4 the most frequent, token 1 next;
        // token 0 never seen → excluded from ranking entirely.
        let counts = vec![0, 7, 0, 0, 9];
        let mut c = LayeredVkCalibration::from_counts(2, 3, counts, 2);
        assert_eq!(c.top_k, 2);
        assert_eq!(c.row_of_token[4], 0, "most frequent → row 0");
        assert_eq!(c.row_of_token[1], 1, "second → row 1 (tie n/a)");
        assert_eq!(c.row_of_token[0], u32::MAX, "never-seen → untracked");
        assert_eq!(c.row_of_token[2], u32::MAX);

        c.observe_layer(0, 4, &[1.0, 2.0, 3.0], &[2.0, 2.0, 1.0]);
        // layer 1 sees the same token — tables are independent per layer.
        c.observe_layer(1, 4, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
        // untracked token 2 → tail rows.
        c.observe_layer(0, 2, &[5.0, 5.0, 5.0], &[5.0, 5.0, 5.0]);

        let r0v = c.layers[0].v.r_squared();
        assert_eq!(r0v.n, 2);
        assert!((r0v.tracked_mass - 0.5).abs() < 1e-12);
        // V−K of the tracked obs = [1,0,−2]; tail = [0,0,0].
        let mut vk_mean = [0.0f32; 3];
        c.layers[0].vk.mean_into(0, &mut vk_mean);
        assert_eq!(vk_mean, [1.0, 0.0, -2.0]);
        let mut k1_mean = [0.0f32; 3];
        c.layers[1].k.mean_into(0, &mut k1_mean);
        assert_eq!(k1_mean, [0.0, 0.0, 0.0]);
    }

    /// Tie-break determinism: equal counts rank by ascending token id.
    #[test]
    fn layered_calibration_ties_break_by_token_id() {
        let counts = vec![3, 3, 3];
        let c = LayeredVkCalibration::from_counts(1, 1, counts, 3);
        assert_eq!(c.row_of_token[0], 0);
        assert_eq!(c.row_of_token[1], 1);
        assert_eq!(c.row_of_token[2], 2);
    }

    trait SortedCheck {
        fn is_windows_sorted(&self) -> bool;
    }
    impl SortedCheck for [f64] {
        fn is_windows_sorted(&self) -> bool {
            self.windows(2).all(|w| w[0] <= w[1] + 1e-12)
        }
    }
}
