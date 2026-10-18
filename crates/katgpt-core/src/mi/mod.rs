//! Mutual-information (MI) bound estimation over **fixed critics** — the
//! modelless `mi_est` layer (Plan 583, Research 521 — MINE arXiv:1801.04062
//! extracted to a zero-training evaluator).
//!
//! # What ships
//!
//! Every variational MI bound (DV, NWJ, InfoNCE-K, JS) is a sample-evaluable
//! functional of critic scores `T(x, y)`. Fix the critic (dot / cosine /
//! frozen BLAKE3-seeded projection — all zero-parameter) and the bounds become
//! a one-pass, zero-alloc **evaluation in nats**, upgradeable by a
//! **permutation test** that is distribution-free and finite-sample exact
//! under H0. No training, no gradient descent, no learned parameters — the
//! modelless mandate holds by construction.
//!
//! ```text
//! mi/mod.rs      this file: MiNats, Critic, MiScratch (buffers + seeded RNG)
//! mi/dv.rs       Donsker–Varadhan bound: plug-in (λ=0), leave-one-out, NWJ (λ=1)
//! mi/bounds.rs   bound ladder from ONE score pass: NWJ / InfoNCE-K / JS + K-ladder
//! mi/perm.rs     permutation test (uniform / circular / block / stratified) + dCor
//! mi/gaussian.rs closed-form Gaussian MI, gated by the shipped sketched_gaussianity
//! mi/ib.rs       frozen-representation Information-Bottleneck ratio + Pareto rank
//! ```
//!
//! # Scoring flow
//!
//! Scores are scratch-resident: [`MiScratch::score_joint`] fills
//! `scratch.joint[i] = T(x_i, y_i)`, [`MiScratch::score_perm`] fills
//! `scratch.perm[i] = T(x_i, y_{σ(i)})` from the scratch's current permutation
//! (or its inverse — the antithetic partner). The bound evaluators in `dv` /
//! `bounds` / `perm` consume `&scratch.joint` / `&scratch.perm` — ONE scoring
//! pass feeds every bound (the plan's "one ScoreMatrix scratch pass").
//!
//! # The honesty contract (read before consuming a number)
//!
//! The module reports **bound VALUES, not I itself**. The gap between the
//! reported value and the true MI is the critic's approximation error — it can
//! be large, and it is invisible in any single number. Therefore every
//! consumer ships the **tuple**, never a bare scalar:
//!
//! 1. **value** — the bound estimate (DV+LOO by default; the plug-in λ=0 form
//!    carries an upward bias of order `critic-dof / 2N` — the "detects MI on
//!    the null" trap, see the calibration curve below);
//! 2. **spread** — between-fold dispersion of the estimator over 8 contiguous
//!    folds. Large spread ⇒ the estimate is not trustworthy (DV variance
//!    explodes at high MI — the tail of `e^T` under the product measure has
//!    divergent moments beyond ρ ≈ 0.577 per dependent dim; SMILE
//!    arXiv:1906.03309 documents the class);
//! 3. **K-ladder / critic_headroom** — how much tighter the bound gets as the
//!    InfoNCE negative count K grows. Large headroom ⇒ the critic family is
//!    still hungry (the reported magnitude is ceiling-limited, not converged);
//! 4. **permutation p** — the only piece that is *calibrated*: under H0 the
//!    p-value is exact for any critic, any statistic, any N (finite-sample
//!    exchangeability). Power, not validity, degrades off the critic's axis.
//!
//! The bilinear blindness of dot/cosine critics is real and pinned by the
//! `Y = X²` non-vacuity control (T2.5): the DV mean term is exactly 0
//! (E[x·x²] = E[x³] = 0) on data with strictly positive MI — while the bound
//! VALUE collapses (−12 nats measured) under the Q-term's e^{x·(x')²} tail.
//! The control pairs that blind-and-collapsed report with a *significant*
//! permutation p (distance-correlation statistic — a characteristic
//! dependence statistic) and a fired Gaussian-arm gate, which is WHY the
//! tuple ships together: each field covers a different failure mode of the
//! others.
//!
//! # Null calibration (T1.4 — the upward plug-in bias, made visible)
//!
//! On ρ = 0 Gaussian data with an informative fixed quadratic critic (3
//! effective dof in 1-D), the DV bias vs N — measured with fixed seeds on
//! this box (2026-08-31;
//! `tests/bench_693_mi_est_goat.rs::g1e_null_calibration_curve` re-derives
//! and re-pins it). The plug-in estimate sits ABOVE the critic's own analytic
//! null bound value by `≲ C·dof/N` (C ≈ 1; LOO removes the dominant
//! self-term; the residue is the log-Jensen gap). Shipped as a recorded
//! curve, not hidden: **a small-N bound value reads high on the null, by
//! construction** — compare against this curve (or against the permutation
//! p) before calling a small positive value "dependence".
//!
//! # Units and representation
//!
//! Everything is **nats** ([`MiNats`]); bits exist only at the presentation
//! edge ([`MiNats::bits`]) — nats/bits mixing is the silent-bug class.
//! Bound math runs in f64 (log-mean-exp accumulation); report fields are f32
//! per plan. The log-mean-exp is max-subtracted — no overflow, no softmax
//! over the batch (the sigmoid-not-softmax house rule composes).
//!
//! # Determinism
//!
//! Permutations are drawn from a BLAKE3-seeded [`fastrand::Rng`] inside
//! [`MiScratch`] — same `(seed, inputs)` ⇒ bit-identical report, run to run.
//! Permutation state, dCor buffers, and stratification tables all live in the
//! scratch: constructed once, **zero allocation in steady state** (G4).
//!
//! # Consumers
//!
//! - third audit axis for the dist-guard family (erank + gaussianity + MI) —
//!   riir-train `edge_lora_dist_guard` (plan 583 T3.4);
//! - information-fidelity probe for quantization/compaction surfaces (T3.5);
//! - the shared DV core for riir-train plan 365 (trained-critic campaign —
//!   SECONDARY track; DRY: it consumes this module, no re-implementation).
//!
//! Opt-in feature `mi_est` (no default consumer yet — the no-default-consumer
//! rule; promotion only after the T3.4/T3.5 consumer GOAT gates pass).
//!
//! mi_est — Modelless Mutual-Information Estimator over fixed critics
//! (Plan 583, Research 521 — MINE arXiv:1801.04062 modelless extraction).
//! DV/NWJ/InfoNCE/JS bound VALUES in nats + LOO bias control + K-ladder
//! tightness + permutation calibration (uniform/circular/block/stratified,
//! dCor non-vacuity control) + a Gaussian closed-form arm gated by the shipped
//! `sketched_gaussianity` + the frozen-representation IB ratio. Opt-in
//! feature `mi_est` — diagnostic surface with no default consumer yet
//! (no-default-consumer rule); consumers: riir-train dist-guard third audit
//! axis (T3.4), quant-fidelity probes (T3.5), riir-train plan 365 DV core.

use blake3::Hasher;

pub mod bounds;
pub mod dv;
pub mod gaussian;
pub mod ib;
pub mod perm;

#[cfg(test)]
pub(crate) mod test_support;

// ─────────────────────────────────────────────────────────────────────────────
// MiNats — nats-only newtype (bits at the presentation edge)
// ─────────────────────────────────────────────────────────────────────────────

/// A mutual-information magnitude in **nats**. Conversion to bits is explicit
/// at the presentation edge — never store or compare mixed units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MiNats(f32);

impl MiNats {
    /// Wrap a value already in nats.
    #[must_use]
    pub fn from_nats(v: f32) -> Self {
        Self(v)
    }

    /// The magnitude in nats.
    #[must_use]
    pub fn nats(self) -> f32 {
        self.0
    }

    /// The magnitude in bits — **presentation edge only**.
    #[must_use]
    pub fn bits(self) -> f32 {
        self.0 * std::f32::consts::LOG2_E
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Critic — the fixed statistics function
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed (never-optimized) critic family. `#[repr(u8)]` — sync/FFI-safe
/// discriminant; the score dispatch is a `match`, no trait objects, no
/// monomorphization bloat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Critic {
    /// `T(x, y) = ⟨x, y⟩` — the raw bilinear critic. Zero-parameter; blind to
    /// any dependence orthogonal to the linear axis (pinned by the `Y = X²`
    /// control).
    Dot = 0,
    /// `T(x, y) = ⟨x/‖x‖, y/‖y‖⟩` — scale-free bilinear.
    Cosine = 1,
    /// `T(x, y) = ⟨R·x, R·y⟩ / √k` with `R` a frozen BLAKE3-seeded Rademacher
    /// ±1 projection (k = min(d, [`FROZEN_PROJ_MAX_K`])). Deterministic for a
    /// given scratch seed; spreads the bilinear axis over k random directions.
    FrozenProj = 2,
}

/// Projection width cap for [`Critic::FrozenProj`].
pub const FROZEN_PROJ_MAX_K: usize = 32;

/// Which permutation table a scored pass reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermSource {
    /// The current `perm_idx` (σ).
    Current,
    /// The inverse table `inv_idx` (σ⁻¹ — the antithetic partner).
    Inverse,
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-pair scoring core (split-borrow parts — pad-resident rows and external
// populations share one implementation)
// ─────────────────────────────────────────────────────────────────────────────

/// Score one pair `(xr, yr)`. The scratch parts (`tmp`, frozen table) are
/// split-borrowed from `MiScratch` so rows may live in ANY buffer (external
/// populations or the scratch's own pad buffers) without aliasing.
#[inline]
fn score_one_parts(
    critic: Critic,
    tmp: &mut [Vec<f32>; 2],
    frozen_k: usize,
    signs: &[i8],
    xr: &[f32],
    yr: &[f32],
) -> f64 {
    let d = xr.len();
    match critic {
        Critic::Dot => f64::from(crate::simd::simd_dot_f32(xr, yr, d)),
        Critic::Cosine => {
            normalize_row_into(xr, &mut tmp[0]);
            normalize_row_into(yr, &mut tmp[1]);
            let (t0, t1) = (&tmp[0][..d], &tmp[1][..d]);
            f64::from(crate::simd::simd_dot_f32(t0, t1, d))
        }
        Critic::FrozenProj => {
            let k = frozen_k.clamp(1, FROZEN_PROJ_MAX_K);
            if tmp[0].len() < k {
                tmp[0].resize(k, 0.0);
            }
            if tmp[1].len() < k {
                tmp[1].resize(k, 0.0);
            }
            project_row_into(xr, signs, k, &mut tmp[0][..k]);
            project_row_into(yr, signs, k, &mut tmp[1][..k]);
            let s = crate::simd::simd_dot_f32(&tmp[0][..k], &tmp[1][..k], k);
            f64::from(s) / (k as f64).sqrt()
        }
    }
}

/// L2-normalize `row` into `buf` (grown if needed).
fn normalize_row_into(row: &[f32], buf: &mut Vec<f32>) {
    let d = row.len();
    if buf.len() < d {
        buf.resize(d, 0.0);
    }
    buf[..d].copy_from_slice(row);
    let mut acc = 0.0f32;
    for v in buf[..d].iter() {
        acc += v * v;
    }
    let inv = 1.0 / acc.sqrt().max(1e-12);
    for v in buf[..d].iter_mut() {
        *v *= inv;
    }
}

/// Frozen Rademacher projection: `out[ki] = Σ_j signs[ki·d + j] · row[j]`
/// (row-major signs, contiguous inner loop). `out` must be `≥ k` long —
/// callers pre-size; the projection-cache path writes row windows.
fn project_row_into(row: &[f32], signs: &[i8], k: usize, out: &mut [f32]) {
    let d = row.len();
    assert!(out.len() >= k, "projection buffer too small");
    for ki in 0..k {
        let row_signs = &signs[ki * d..(ki + 1) * d];
        let mut acc = 0.0f32;
        for (j, &xv) in row.iter().enumerate() {
            acc += row_signs[j] as f32 * xv;
        }
        out[ki] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MiScratch — score buffers + BLAKE3-seeded permutation RNG (zero-alloc steady
// state)
// ─────────────────────────────────────────────────────────────────────────────

/// One scratch per measurement stream: score buffers, permutation state, the
/// seeded RNG, and the frozen projection table. Construct once per
/// (capacity, seed); [`MiScratch::ensure`] grows buffers if a later call needs
/// more — **allocation happens only at construction / growth, never in the
/// steady-state evaluate path** (G4).
///
/// `Clone` is exact (the RNG state is one `u64`, cloned stream-identically —
/// `fastrand::Rng`), so a cloned scratch reproduces the same permutation
/// stream from the same position. Consumers embedding a scratch in their own
/// reusable state (e.g. the riir-train dist-guard axis) need the derive.
#[derive(Clone, Debug)]
pub struct MiScratch {
    /// Joint scores `T(x_i, y_i)` (capacity ≥ n).
    pub joint: Vec<f64>,
    /// Permuted scores `T(x_i, y_{σ(i)})` (capacity ≥ n).
    pub perm: Vec<f64>,
    /// The current permutation σ (capacity ≥ n).
    pub perm_idx: Vec<u32>,
    /// Inverse of the current permutation (antithetic pairing; capacity ≥ n).
    pub inv_idx: Vec<u32>,
    /// Double-centered distance scratch for the dCor statistic
    /// (capacity ≥ n² each).
    pub dist_x: Vec<f32>,
    pub dist_y: Vec<f32>,
    /// Stratified-permutation scratch (counting-sort offsets + items).
    pub strat_offsets: Vec<u32>,
    pub strat_items: Vec<u32>,
    /// Stratified-permutation shuffled copy of `strat_items`.
    pub strat_shuffled: Vec<u32>,
    /// Sort buffer for the Median permutation statistic (capacity ≥ n).
    pub stat_buf: Vec<f64>,
    /// Permutation-null statistic draws (capacity ≥ b) — the G4 alloc-free
    /// substitute for a per-run Vec.
    pub null_buf: Vec<f64>,
    /// Sorted copy of the null draws for percentile extraction (capacity ≥ b).
    pub null_sorted: Vec<f64>,
    /// Row-padded population buffers for cross-dimension scoring (the IB
    /// ratio's v1 path; capacity ≥ n·d_max).
    pub pad_a: Vec<f32>,
    pub pad_b: Vec<f32>,
    /// Cached FrozenProj populations (the many-permutation fast path):
    /// `proj_a`/`proj_b` are `[n × k]` projections of the last
    /// [`MiScratch::project_frozen`] call. Pairing-invariant — project once,
    /// then every permutation pass costs n length-k dots instead of n
    /// re-projections (measured ~25× on the B=128 guard shape).
    pub(crate) proj_a: Vec<f32>,
    pub(crate) proj_b: Vec<f32>,
    /// Row count of the cached projections (0 = empty).
    pub(crate) proj_n: usize,

    pub(crate) rng: fastrand::Rng,
    pub(crate) frozen_k: usize,
    pub(crate) frozen_signs: Vec<i8>,
    pub(crate) tmp: [Vec<f32>; 2],
    pub(crate) seed: u64,
}

/// Serial Dot-critic scoring over an explicit pair table (the wasm32 path
/// and any future no-rayon build).
#[cfg(target_arch = "wasm32")]
fn score_dot_serial(
    x: &[f32],
    y: &[f32],
    idx: Option<&[u32]>,
    out: &mut [f64],
    n: usize,
    d: usize,
) {
    for i in 0..n {
        let j = idx.map_or(i, |s| s[i] as usize);
        out[i] = f64::from(crate::simd::simd_dot_f32(
            &x[i * d..(i + 1) * d],
            &y[j * d..(j + 1) * d],
            d,
        ));
    }
}

/// Parallel-chunk threshold for the Dot-critic score paths: below it the
/// serial loop wins (rayon task overhead), above it the batched per-pair
/// dots amortize across the pool. Also keeps the G4 alloc-check (n = 4096)
/// on the alloc-free serial path.
const DOT_PAR_CHUNK: usize = 4096;

impl MiScratch {
    /// Construct for populations of `n` pairs at dimension `d`, seeded by
    /// `seed`. The RNG stream is `blake3("katgpt-core::mi" ‖ seed)` — the
    /// house BLAKE3-deterministic-table pattern; same seed ⇒ bit-identical
    /// permutation stream and frozen projection.
    #[must_use]
    pub fn new(n: usize, d: usize, seed: u64) -> Self {
        assert!(n > 0 && d > 0, "n and d must be positive");
        let mut h = Hasher::new();
        h.update(b"katgpt-core::mi");
        h.update(&seed.to_le_bytes());
        let digest = h.finalize();
        let mixed = u64::from_le_bytes(digest.as_bytes()[0..8].try_into().expect("8 bytes"));
        let k = FROZEN_PROJ_MAX_K.min(d);
        let mut s = Self {
            joint: vec![0.0; n],
            perm: vec![0.0; n],
            perm_idx: (0..n as u32).collect(),
            inv_idx: vec![0; n],
            dist_x: Vec::new(),
            dist_y: Vec::new(),
            strat_offsets: Vec::new(),
            strat_items: Vec::new(),
            strat_shuffled: Vec::new(),
            stat_buf: Vec::new(),
            null_buf: Vec::new(),
            null_sorted: Vec::new(),
            pad_a: Vec::new(),
            pad_b: Vec::new(),
            proj_a: Vec::new(),
            proj_b: Vec::new(),
            proj_n: 0,
            rng: fastrand::Rng::with_seed(mixed),
            frozen_k: k,
            frozen_signs: Self::build_frozen_signs(seed, d, k),
            tmp: [vec![0.0; d], vec![0.0; d]],
            seed,
        };
        s.next_perm(n); // deterministic initial permutation state
        s
    }

    /// Rademacher ±1 table from the BLAKE3 XOF (block-chained past 256 signs
    /// — the `data_probe::gaussianity` pattern). Layout: row-major `[k × d]`,
    /// entry `(ki, j)` at `ki * d + j`.
    fn build_frozen_signs(seed: u64, d: usize, k: usize) -> Vec<i8> {
        let mut h = Hasher::new();
        h.update(b"katgpt-core::mi::frozenproj");
        h.update(&seed.to_le_bytes());
        h.update(&(d as u64).to_le_bytes());
        let mut block = h.finalize();
        let mut block_idx = 0u64;
        let mut signs = Vec::with_capacity(k * d);
        for slot in 0..k * d {
            if slot > 0 && slot % 256 == 0 {
                let mut hb = Hasher::new();
                hb.update(block.as_bytes());
                hb.update(&block_idx.to_le_bytes());
                block = hb.finalize();
                block_idx += 1;
            }
            let bytes = block.as_bytes();
            let byte = bytes[(slot / 8) % 32];
            let bit = (byte >> (slot % 8)) & 1;
            signs.push(if bit == 1 { 1 } else { -1 });
        }
        signs
    }

    /// Seed the scratch was constructed with.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Grow buffers to `(n, d)` capacity. Idempotent — only allocates when the
    /// current capacity is too small. The score paths call it defensively;
    /// steady-state callers at a fixed size never allocate after construction.
    pub fn ensure(&mut self, n: usize, d: usize) {
        if self.joint.len() < n {
            self.joint.resize(n, 0.0);
        }
        if self.perm.len() < n {
            self.perm.resize(n, 0.0);
        }
        if self.perm_idx.len() < n {
            self.perm_idx.resize(n, 0);
        }
        if self.inv_idx.len() < n {
            self.inv_idx.resize(n, 0);
        }
        if self.stat_buf.len() < n {
            self.stat_buf.resize(n, 0.0);
        }
        if self.pad_a.len() < n * d {
            self.pad_a.resize(n * d, 0.0);
        }
        if self.pad_b.len() < n * d {
            self.pad_b.resize(n * d, 0.0);
        }
        let k = FROZEN_PROJ_MAX_K.min(d);
        if self.frozen_signs.len() < k * d {
            self.frozen_k = k;
            self.frozen_signs = Self::build_frozen_signs(self.seed, d, k);
        } else if self.frozen_k < k {
            self.frozen_k = k;
        }
        for t in &mut self.tmp {
            if t.len() < d {
                t.resize(d, 0.0);
            }
        }
    }

    /// Draw a fresh uniform permutation into `perm_idx` (Fisher–Yates over the
    /// seeded RNG stream).
    pub fn next_perm(&mut self, n: usize) {
        assert!(self.perm_idx.len() >= n, "perm_idx capacity < n");
        for (i, slot) in self.perm_idx.iter_mut().enumerate().take(n) {
            *slot = i as u32;
        }
        for i in (1..n).rev() {
            let j = self.rng.usize(..=i);
            self.perm_idx.swap(i, j);
        }
    }

    /// Fill `inv_idx` with the inverse of the current `perm_idx` (the
    /// antithetic partner σ⁻¹).
    pub fn invert_perm(&mut self, n: usize) {
        assert!(self.perm_idx.len() >= n && self.inv_idx.len() >= n);
        for i in 0..n {
            self.inv_idx[self.perm_idx[i] as usize] = i as u32;
        }
    }

    /// Reset the RNG stream to the construction seed — makes a multi-draw
    /// helper called repeatedly on one scratch bit-deterministic per
    /// (seed, inputs), the same contract `PermTest` reseed gives.
    pub fn reseed(&mut self) {
        let mut h = Hasher::new();
        h.update(b"katgpt-core::mi");
        h.update(&self.seed.to_le_bytes());
        let digest = h.finalize();
        let mixed = u64::from_le_bytes(digest.as_bytes()[0..8].try_into().expect("8 bytes"));
        self.rng = fastrand::Rng::with_seed(mixed);
    }

    // ── scoring (external populations) ────────────────────────────────

    /// Score the identity pairing into `scratch.joint`: `joint[i] = T(x_i, y_i)`.
    /// The Dot critic parallelizes over pair chunks above [`DOT_PAR_CHUNK`]
    /// (disjoint slice borrows, no locks — the chunk kernel is pure).
    pub fn score_joint(&mut self, critic: Critic, x: &[f32], y: &[f32], n: usize, d: usize) {
        self.ensure(n, d);
        if let Critic::Dot = critic
            && n > DOT_PAR_CHUNK
        {
            #[cfg(target_arch = "wasm32")]
            {
                score_dot_serial(x, y, None, &mut self.joint, n, d);
                return;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                let joint = &mut self.joint[..n];
                joint
                    .par_chunks_mut(DOT_PAR_CHUNK)
                    .enumerate()
                    .for_each(|(ci, jc)| {
                        // chunks_exact removes the per-pair bounds checks
                        // on both row slices (rows are contiguous).
                        let x_rows = &x[ci * DOT_PAR_CHUNK * d..];
                        let y_rows = &y[ci * DOT_PAR_CHUNK * d..];
                        for (li, o) in jc.iter_mut().enumerate() {
                            let xr = &x_rows[li * d..(li + 1) * d];
                            let yr = &y_rows[li * d..(li + 1) * d];
                            *o = f64::from(crate::simd::simd_dot_f32(xr, yr, d));
                        }
                    });
                return;
            }
        }
        for i in 0..n {
            let v = {
                let tmp = &mut self.tmp;
                let fk = self.frozen_k;
                let signs = self.frozen_signs.as_slice();
                let xr = &x[i * d..(i + 1) * d];
                let yr = &y[i * d..(i + 1) * d];
                score_one_parts(critic, tmp, fk, signs, xr, yr)
            };
            self.joint[i] = v;
        }
    }

    /// Score the current permutation (or its inverse) into `scratch.perm`:
    /// `perm[i] = T(x_i, y_{σ(i)})`. `inv_idx` is kept in sync.
    pub fn score_perm(
        &mut self,
        critic: Critic,
        x: &[f32],
        y: &[f32],
        n: usize,
        d: usize,
        src: PermSource,
    ) {
        self.ensure(n, d);
        self.invert_perm(n);
        if let Critic::Dot = critic
            && n > DOT_PAR_CHUNK
        {
            #[cfg(target_arch = "wasm32")]
            {
                let idx_ref: &[u32] = match src {
                    PermSource::Current => &self.perm_idx,
                    PermSource::Inverse => &self.inv_idx,
                };
                score_dot_serial(x, y, Some(idx_ref), &mut self.perm, n, d);
                return;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                // Split-borrow: the perm table read and the perm score
                // write are disjoint fields.
                let (perm, idx_ref) = match src {
                    PermSource::Current => (&mut self.perm[..n], &self.perm_idx),
                    PermSource::Inverse => (&mut self.perm[..n], &self.inv_idx),
                };
                perm.par_chunks_mut(DOT_PAR_CHUNK)
                    .enumerate()
                    .for_each(|(ci, pc)| {
                        let x_rows = &x[ci * DOT_PAR_CHUNK * d..];
                        for (li, o) in pc.iter_mut().enumerate() {
                            let i = ci * DOT_PAR_CHUNK + li;
                            let j = idx_ref[i] as usize;
                            let xr = &x_rows[li * d..(li + 1) * d];
                            let yr = &y[j * d..(j + 1) * d];
                            *o = f64::from(crate::simd::simd_dot_f32(xr, yr, d));
                        }
                    });
                return;
            }
        }
        for i in 0..n {
            let j = match src {
                PermSource::Current => self.perm_idx[i] as usize,
                PermSource::Inverse => self.inv_idx[i] as usize,
            };
            let v = {
                let tmp = &mut self.tmp;
                let fk = self.frozen_k;
                let signs = self.frozen_signs.as_slice();
                let xr = &x[i * d..(i + 1) * d];
                let yr = &y[j * d..(j + 1) * d];
                score_one_parts(critic, tmp, fk, signs, xr, yr)
            };
            self.perm[i] = v;
        }
    }

    // ── scoring (FrozenProj cache — the many-permutation fast path) ────

    /// Project BOTH populations once into the FrozenProj cache:
    /// `proj_a[i·k + t] = ⟨R_t, x_i⟩` and `proj_b[j·k + t] = ⟨R_t, y_j⟩` —
    /// the exact [`Critic::FrozenProj`] projection arithmetic
    /// (`project_row_into`), so every subsequent cached score is
    /// **bit-identical** to the uncached scoring path (pinned by test).
    /// Allocation happens here (grow-once per shape), never in the cached
    /// score passes (G4).
    pub fn project_frozen(&mut self, x: &[f32], y: &[f32], n: usize, d: usize) {
        assert!(x.len() >= n * d && y.len() >= n * d, "population too small");
        let k = self.frozen_k.clamp(1, FROZEN_PROJ_MAX_K);
        if self.proj_a.len() < n * k {
            self.proj_a.resize(n * k, 0.0);
        }
        if self.proj_b.len() < n * k {
            self.proj_b.resize(n * k, 0.0);
        }
        let signs = self.frozen_signs.as_slice();
        let (pa, pb) = (&mut self.proj_a, &mut self.proj_b);
        for i in 0..n {
            project_row_into(
                &x[i * d..(i + 1) * d],
                signs,
                k,
                &mut pa[i * k..(i + 1) * k],
            );
            project_row_into(
                &y[i * d..(i + 1) * d],
                signs,
                k,
                &mut pb[i * k..(i + 1) * k],
            );
        }
        self.proj_n = n;
    }

    /// Score the identity pairing from the projection cache (identity to
    /// `score_joint(Critic::FrozenProj, …)` bit-for-bit). Requires a prior
    /// [`MiScratch::project_frozen`] at `≥ n` rows.
    pub fn score_joint_cached(&mut self, n: usize) {
        let k = self.frozen_k.clamp(1, FROZEN_PROJ_MAX_K);
        assert!(
            self.proj_n >= n,
            "project_frozen must run before the cached score pass"
        );
        let scale = (k as f64).sqrt();
        for i in 0..n {
            let s = crate::simd::simd_dot_f32(
                &self.proj_a[i * k..(i + 1) * k],
                &self.proj_b[i * k..(i + 1) * k],
                k,
            );
            self.joint[i] = f64::from(s) / scale;
        }
    }

    /// Score the current permutation (or inverse) from the projection cache
    /// (identity to `score_perm(Critic::FrozenProj, …)` bit-for-bit).
    pub fn score_perm_cached(&mut self, n: usize, src: PermSource) {
        let k = self.frozen_k.clamp(1, FROZEN_PROJ_MAX_K);
        assert!(
            self.proj_n >= n,
            "project_frozen must run before the cached score pass"
        );
        self.invert_perm(n);
        let scale = (k as f64).sqrt();
        for i in 0..n {
            let j = match src {
                PermSource::Current => self.perm_idx[i] as usize,
                PermSource::Inverse => self.inv_idx[i] as usize,
            };
            let s = crate::simd::simd_dot_f32(
                &self.proj_a[i * k..(i + 1) * k],
                &self.proj_b[j * k..(j + 1) * k],
                k,
            );
            self.perm[i] = f64::from(s) / scale;
        }
    }

    // ── scoring (scratch pad buffers — the cross-dimension IB path) ─────────

    /// Score the identity pairing from the scratch's own pad buffers
    /// (`pad_a` × `pad_b`) into `joint`.
    pub fn score_joint_pads(&mut self, critic: Critic, n: usize, dm: usize) {
        self.ensure(n, dm);
        for i in 0..n {
            let v = {
                let tmp = &mut self.tmp;
                let fk = self.frozen_k;
                let signs = self.frozen_signs.as_slice();
                let xr = &self.pad_a[i * dm..(i + 1) * dm];
                let yr = &self.pad_b[i * dm..(i + 1) * dm];
                score_one_parts(critic, tmp, fk, signs, xr, yr)
            };
            self.joint[i] = v;
        }
    }

    /// Score the current permutation (or inverse) from the pad buffers into
    /// `perm`.
    pub fn score_perm_pads(&mut self, critic: Critic, n: usize, dm: usize, src: PermSource) {
        self.ensure(n, dm);
        self.invert_perm(n);
        for i in 0..n {
            let j = match src {
                PermSource::Current => self.perm_idx[i] as usize,
                PermSource::Inverse => self.inv_idx[i] as usize,
            };
            let v = {
                let tmp = &mut self.tmp;
                let fk = self.frozen_k;
                let signs = self.frozen_signs.as_slice();
                let xr = &self.pad_a[i * dm..(i + 1) * dm];
                let yr = &self.pad_b[j * dm..(j + 1) * dm];
                score_one_parts(critic, tmp, fk, signs, xr, yr)
            };
            self.perm[i] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mi::perm::PermTest;

    #[test]
    fn minats_units() {
        let v = MiNats::from_nats(0.693_147_4);
        assert!((v.bits() - 1.0).abs() < 1e-6);
        assert_eq!(MiNats::from_nats(2.0).nats(), 2.0);
    }

    #[test]
    fn scratch_perm_is_a_permutation() {
        let mut s = MiScratch::new(64, 8, 7);
        for _ in 0..16 {
            s.next_perm(64);
            let mut seen = [false; 64];
            for &v in &s.perm_idx[..64] {
                assert!((v as usize) < 64, "out of range");
                assert!(!seen[v as usize], "duplicate index");
                seen[v as usize] = true;
            }
        }
    }

    #[test]
    fn scratch_inverse_perm_inverts() {
        let mut s = MiScratch::new(128, 4, 99);
        s.next_perm(128);
        s.invert_perm(128);
        for i in 0..128 {
            assert_eq!(s.inv_idx[s.perm_idx[i] as usize] as usize, i);
        }
    }

    #[test]
    fn scratch_seeded_determinism() {
        let mut a = MiScratch::new(32, 8, 0xDEAD_BEEF);
        let mut b = MiScratch::new(32, 8, 0xDEAD_BEEF);
        a.next_perm(32);
        b.next_perm(32);
        assert_eq!(a.perm_idx[..32], b.perm_idx[..32]);
        let mut c = MiScratch::new(32, 8, 0xDEAD_BEEC);
        c.next_perm(32);
        assert_ne!(a.perm_idx[..32], c.perm_idx[..32]);
    }

    #[test]
    fn frozenproj_signs_deterministic_and_wellformed() {
        let a = MiScratch::new(8, 16, 42);
        let b = MiScratch::new(8, 16, 42);
        assert_eq!(a.frozen_signs, b.frozen_signs);
        assert_eq!(a.frozen_k, b.frozen_k);
        assert!(a.frozen_signs.iter().all(|&s| s == 1 || s == -1));
        assert_eq!(a.frozen_signs.len(), a.frozen_k * 16);
    }

    #[test]
    fn score_joint_matches_manual_dot() {
        let n = 5;
        let d = 4;
        let x: Vec<f32> = (0..n * d).map(|v| v as f32 * 0.25).collect();
        let y: Vec<f32> = (0..n * d).map(|v| 1.0 - v as f32 * 0.1).collect();
        let mut s = MiScratch::new(n, d, 1);
        s.score_joint(Critic::Dot, &x, &y, n, d);
        for i in 0..n {
            let manual: f32 = (0..d).map(|j| x[i * d + j] * y[i * d + j]).sum();
            assert!(
                (s.joint[i] - f64::from(manual)).abs() < 1e-4,
                "{} vs {manual}",
                s.joint[i]
            );
        }
    }

    #[test]
    fn score_perm_tracks_current_and_inverse_sigma() {
        // Shift permutation σ(i) = i+1: Current reads y_{i+1}, Inverse reads
        // y_{i−1}.
        let n = 6;
        let d = 3;
        let x: Vec<f32> = (0..n * d).map(|v| v as f32).collect();
        let y: Vec<f32> = (0..n * d).map(|v| (v * 7 % 13) as f32).collect();
        let mut s = MiScratch::new(n, d, 3);
        for i in 0..n {
            s.perm_idx[i] = ((i + 1) % n) as u32;
        }
        s.score_perm(Critic::Dot, &x, &y, n, d, PermSource::Current);
        for i in 0..n {
            let j = (i + 1) % n;
            let manual: f32 = (0..d).map(|k| x[i * d + k] * y[j * d + k]).sum();
            assert!((s.perm[i] - f64::from(manual)).abs() < 1e-4);
        }
        s.score_perm(Critic::Dot, &x, &y, n, d, PermSource::Inverse);
        for i in 0..n {
            let j = (i + n - 1) % n;
            let manual: f32 = (0..d).map(|k| x[i * d + k] * y[j * d + k]).sum();
            assert!((s.perm[i] - f64::from(manual)).abs() < 1e-4);
        }
    }

    #[test]
    fn pad_scoring_matches_external_scoring() {
        // Score a same-shaped pair through pads and externally with Dot —
        // identical results.
        let n = 16;
        let dm = 4;
        let mut s = MiScratch::new(n, dm, 21);
        let x: Vec<f32> = (0..n * dm).map(|v| (v % 17) as f32 * 0.5).collect();
        let y: Vec<f32> = (0..n * dm).map(|v| (v % 23) as f32 * 0.25).collect();
        s.pad_a.resize(n * dm, 0.0);
        s.pad_b.resize(n * dm, 0.0);
        s.pad_a[..n * dm].copy_from_slice(&x);
        s.pad_b[..n * dm].copy_from_slice(&y);
        s.score_joint_pads(Critic::Dot, n, dm);
        let pad_scores = s.joint[..n].to_vec();
        s.score_joint(Critic::Dot, &x, &y, n, dm);
        assert_eq!(pad_scores, s.joint[..n]);
    }

    /// The FrozenProj projection cache is BIT-IDENTICAL to the uncached
    /// scoring path — same projection arithmetic, same simd_dot, same scale.
    /// Pinned at a production shape (n=64, d=32, k=32) and a ragged shape
    /// (n=7, d=13, k=13); also covers the paired permutation tables (same
    /// seed ⇒ same σ ⇒ same perm scores).
    #[test]
    fn frozen_cached_scores_are_bit_identical_to_uncached() {
        for (n, d) in [(64usize, 32usize), (7, 13)] {
            let x: Vec<f32> = (0..n * d).map(|v| (v as f32 * 0.173).sin() * 2.0).collect();
            let y: Vec<f32> = (0..n * d).map(|v| (v as f32 * 0.031).cos() * 1.5).collect();
            let mut a = MiScratch::new(n, d, 7);
            let mut b = MiScratch::new(n, d, 7);

            a.score_joint(Critic::FrozenProj, &x, &y, n, d);
            a.next_perm(n);
            a.score_perm(Critic::FrozenProj, &x, &y, n, d, PermSource::Current);

            b.project_frozen(&x, &y, n, d);
            b.score_joint_cached(n);
            b.next_perm(n);
            b.score_perm_cached(n, PermSource::Current);

            for i in 0..n {
                assert_eq!(
                    a.joint[i].to_bits(),
                    b.joint[i].to_bits(),
                    "joint[{i}] @{n}x{d}"
                );
                assert_eq!(
                    a.perm[i].to_bits(),
                    b.perm[i].to_bits(),
                    "perm[{i}] @{n}x{d}"
                );
            }
        }
    }

    /// `PermTest::run_frozen_cached` reproduces `run(FrozenProj)` bit-for-bit
    /// — same reseeded null stream, same statistic values, same exact p.
    #[test]
    fn frozen_cached_perm_test_is_bit_identical_to_run() {
        let n = 64;
        let d = 32;
        let x: Vec<f32> = (0..n * d).map(|v| (v as f32 * 0.091).sin()).collect();
        let y: Vec<f32> = (0..n * d).map(|v| (v as f32 * 0.052).cos()).collect();
        let t = PermTest::new(32, 9);
        let mut s1 = MiScratch::new(n, d, 11);
        let mut s2 = MiScratch::new(n, d, 11);
        let r1 = t.run(Critic::FrozenProj, &x, &y, n, d, None, &mut s1);
        let r2 = t.run_frozen_cached(&x, &y, n, d, &mut s2);
        assert_eq!(r1.observed.to_bits(), r2.observed.to_bits());
        assert_eq!(r1.p.to_bits(), r2.p.to_bits());
        assert_eq!(r1.null_hi95.to_bits(), r2.null_hi95.to_bits());
    }
}
