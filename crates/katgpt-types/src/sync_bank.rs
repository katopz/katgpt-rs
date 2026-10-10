//! SyncBank — online second-order co-activation memory (Issue 930 / Research 615).
//!
//! The modelless extraction of CMM's synchronization memory (arXiv:2610.07907,
//! the CTM sync summary `S_ij ∝ Σ_τ e^{−r_ij(t−τ)} z_τ^i z_τ^j` stripped of
//! training): K timescale banks, each holding a retention-decayed
//! **second-moment** accumulator over a bounded belief vector, read out as the
//! exact mass-normalized covariance. Produces a per-entity *relation* signal
//! (`δ_level` — coupling state; `δ_surprise` — relation change) that no
//! first-order value-surprise kernel can express.
//!
//! # The estimator (three reviewer-verified laws — all load-bearing)
//!
//! **(i) Read C, never raw S.** Raw products track first-order movement: a
//! single dim `z_i` moving with constant coupling changes `z_i·z_j` for every
//! `j`, so `Σ|S|` spikes ≈ D× on pure value change. Only the centered,
//! mass-normalized covariance isolates the *relation*.
//!
//! **(ii) Centering is mass-normalized at readout — never incremental-lagged.**
//! `C = S/mass − (m/mass)⊗(m/mass)` is the exact weighted covariance; the
//! lagged form `(z−m)⊗(z−m)` has products in `(−1,1)` (z = 0.9, m = 0 → 0.81)
//! and breaks the covariance bound `C_ij ∈ [−¼, ¼]` for `z ∈ [0,1]`. This is
//! the A.3 horizon-invariance law: normalize by the **linear weight sum**
//! (`mass`), never its sqrt and never the raw tick count.
//!
//! **(iii) `d` is the STEP SIZE; retention is `λ = 1−d`.** All updates are
//! retention-form — `S ← λ·S + zz`, `m ← λ·m + z`, `mass ← λ·mass + 1`
//! (recursive: no pow, no tick count) — giving `mass → 1/d`, i.e. the
//! 4/16/64-tick effective horizons of the canonical ladder
//! [`SYNC_BANK_LADDER`] (tick/second/minute-class). Mixing the two readings
//! collapses the ladder (d-as-retention puts the "minute" bank at ~1 tick) or
//! wrecks the normalizer (the mass formula belongs to the λ form only).
//!
//! # Shape
//!
//! Storage is the packed upper triangle (`i ≤ j`, `D(D+1)/2` floats per bank)
//! — the covariance is symmetric by construction. Readouts stream over the
//! packed slab and aggregate on the fly: **zero allocation, zero scratch**,
//! deterministic iteration order. Observe cost: `K·D(D+1)/2` FMA + same-order
//! ops (D=8 → 108; D=64 → 6,240).
//!
//! # Boundary law
//!
//! Latent only: state is per-entity local; only the scalar readouts
//! (`δ_level`, `δ_surprise`, `argmax`) may cross any sync boundary — never the
//! `D²`/`D(D+1)/2` slab (the 5-scalar sync law). Observability is
//! limelight-gated at the consumer (HOT/WARM entities carry a SyncBank; COLD
//! entities carry none).

/// The canonical step-size ladder (tick / second / minute class).
///
/// Bank k decays per observation by `λ_k = 1 − d_k`; the effective memory
/// horizon of bank k is `1/d_k` observations (4 / 16 / 64 ticks). Pass this to
/// [`SyncBank::new`] for `K == 3`.
pub const SYNC_BANK_LADDER: [f32; 3] = [0.25, 0.0625, 0.015625];

/// Packed upper-triangle offset for `(i, j)` with `i ≤ j` in a `D×D` row-major
/// triangle (`D(D+1)/2` entries, row-major over `i`).
///
/// Inverse of [`packed_unpack`].
#[inline(always)]
pub fn packed_index(d: usize, i: usize, j: usize) -> usize {
    debug_assert!(i <= j && j < d);
    i * d + j - i * (i + 1) / 2
}

/// Inverse of [`packed_index`]: returns the `(i, j)` pair (`i ≤ j`) for a
/// packed-triangle offset. Consumers decode the `argmax` output of
/// [`SyncBank::relation_level_into`] with this.
pub fn packed_unpack(d: usize, idx: usize) -> (usize, usize) {
    let tri = d * (d + 1) / 2;
    debug_assert!(idx < tri);
    // Row i has (d − i) entries; rows 0..i hold d + (d−1) + … + (d−i+1)
    // = i·d − i(i−1)/2 entries. Linear scan is fine (readout-time only); the
    // `i + 1 < d` bound keeps a malformed offset terminating (debug builds
    // trip the assert above first).
    let mut i = 0usize;
    // row_start(i) = i·d − i(i−1)/2 = i(2d − i + 1)/2 — the second form never
    // subtracts (i = 0 would underflow `i − 1` in usize).
    while i + 1 < d && i * (2 * d - i + 1) / 2 + (d - i) <= idx {
        i += 1;
    }
    let row_start = i * (2 * d - i + 1) / 2;
    (i, i + idx - row_start)
}

/// Online second-order co-activation memory over `D` belief dims in `K`
/// timescale banks.
///
/// Construct with [`SyncBank::new`] (pass [`SYNC_BANK_LADDER`] for `K == 3`),
/// feed each tick's belief with [`SyncBank::observe`], read out with
/// [`SyncBank::relation_level_into`] / [`SyncBank::relation_surprise_into`].
/// The unarmed construction reads zeros until the first observation.
pub struct SyncBank<const D: usize, const K: usize> {
    /// Per-bank raw upper-triangle second-moment sums, one contiguous slab:
    /// bank `k` at `[k·tri .. (k+1)·tri)`. Update: `S ← λ·S + z_i·z_j`.
    ///
    /// Law (i): never read directly — only through the mass-normalized
    /// covariance at [`SyncBank::readout`]-level methods.
    sums: Vec<f32>,
    /// Per-bank weighted mean (`m ← λ·m + z`), one contiguous `K·D` slab.
    means: Vec<f32>,
    /// Per-bank recursive mass (`mass ← λ·mass + 1`) — the A.3 linear
    /// weight sum; `0.0` until the first observation, `→ 1/d_k` after.
    mass: [f32; K],
    /// The step-size ladder this bank was built with (exposed for horizon
    /// labeling at the consumer).
    ladder: [f32; K],
}

impl<const D: usize, const K: usize> SyncBank<D, K> {
    /// Packed upper-triangle length per bank.
    #[inline(always)]
    pub fn tri_len() -> usize {
        D * (D + 1) / 2
    }

    /// Construct with the given step-size ladder (entry `k` drives bank `k`;
    /// horizon `1/d_k`). Panics unless every `d` is strictly inside `(0, 1)`
    /// (the TemporalDerivativeKernel::new posture — refuse loud, never clamp).
    pub fn new(ladder: [f32; K]) -> Self {
        for &d in &ladder {
            assert!(
                d > 0.0 && d < 1.0,
                "SyncBank: ladder steps must be strictly in (0, 1), got {d}"
            );
        }
        Self {
            sums: vec![0.0; K * Self::tri_len()],
            means: vec![0.0; K * D],
            mass: [0.0; K],
            ladder,
        }
    }

    /// The step-size ladder (bank k's decay per observation is `d_k`; memory
    /// horizon `1/d_k`).
    #[inline]
    pub fn ladder(&self) -> &[f32; K] {
        &self.ladder
    }

    /// Per-bank recursive mass (the A.3 normalizer). `0.0` before the first
    /// observation; converges to `1/d_k`.
    #[inline]
    pub fn mass(&self) -> &[f32; K] {
        &self.mass
    }

    /// The raw second-moment slab (`K × tri_len`, bank-major). **Law (i):
    /// never read directly** — exposed for freeze/debug tooling only; the
    /// meaningful quantities are the readouts below.
    #[inline]
    pub fn sums(&self) -> &[f32] {
        &self.sums
    }

    /// The raw weighted-mean slab (`K × D`, bank-major). Debug/freeze tooling;
    /// the normalized mean is `means[k·D + i] / mass[k]`.
    #[inline]
    pub fn means(&self) -> &[f32] {
        &self.means
    }

    /// Whether any observation has landed (mass > 0 in some bank).
    #[inline]
    pub fn observed(&self) -> bool {
        self.mass.iter().any(|&m| m > 0.0)
    }

    /// Feed one observation (the current belief vector) into every bank.
    ///
    /// Retention-form update per bank `k` (`λ_k = 1 − d_k`):
    /// `S ← λ·S + zz` (packed upper triangle), `m ← λ·m + z`,
    /// `mass ← λ·mass + 1`. Allocation-free; `K·D(D+1)/2` FMAs + same-order ops.
    pub fn observe(&mut self, z: &[f32; D]) {
        let tri = Self::tri_len();
        for k in 0..K {
            let lam = 1.0 - self.ladder[k];
            let s = &mut self.sums[k * tri..(k + 1) * tri];
            let m = &mut self.means[k * D..(k + 1) * D];
            for i in 0..D {
                let zi = z[i];
                // row start in the packed layout = i·D − i(i−1)/2, written
                // overflow-free as i(2D − i + 1)/2 (the `i − 1` form would
                // underflow usize at i = 0). Must match `packed_index`.
                let base = i * (2 * D - i + 1) / 2;
                // Contiguous zip-axpy over the row segment — the form LLVM
                // auto-vectorizes (NEON/AVX FMA); bounds-check-free.
                let row = &mut s[base..base + (D - i)];
                for (si, zj) in row.iter_mut().zip(z[i..D].iter()) {
                    *si = si.mul_add(lam, zi * zj);
                }
                m[i] = m[i].mul_add(lam, z[i]);
            }
            self.mass[k] = self.mass[k].mul_add(lam, 1.0);
        }
    }

    /// Covariance entry `(i, j)` (`i ≤ j`) of bank `k` — the mass-normalized
    /// exact form `S_ij/mass − m_i·m_j/mass²` (law ii). Zeros before the first
    /// observation (mass 0), never a division by zero.
    #[inline]
    pub fn cov(&self, k: usize, i: usize, j: usize) -> f32 {
        debug_assert!(k < K && i <= j && j < D);
        let mass = self.mass[k];
        if mass == 0.0 {
            return 0.0;
        }
        let inv = 1.0 / mass;
        let tri = Self::tri_len();
        let idx = packed_index(D, i, j);
        self.sums[k * tri + idx] * inv
            - self.means[k * D + i] * self.means[k * D + j] * inv * inv
    }

    /// `δ_level` readout: per bank, `Σ_ij |C_ij|` (coupling STATE — high on a
    /// tightly-coupled pack, permanently) + `argmax_ij |C_ij|` as the packed
    /// offset (decode with [`packed_unpack`]; the dominant pair for salience).
    ///
    /// Writes `K` entries into each of `level` and `argmax` (debug-asserted).
    /// Zero-alloc, zero-scratch: streams the packed slab once per bank.
    pub fn relation_level_into(&self, level: &mut [f32], argmax: &mut [f32]) {
        debug_assert!(level.len() >= K && argmax.len() >= K);
        let tri = Self::tri_len();
        for k in 0..K {
            let mass = self.mass[k];
            if mass == 0.0 {
                level[k] = 0.0;
                argmax[k] = 0.0;
                continue;
            }
            let inv = 1.0 / mass;
            let inv2 = inv * inv;
            let s = &self.sums[k * tri..(k + 1) * tri];
            let m = &self.means[k * D..(k + 1) * D];
            let mut sum = 0.0f32;
            let mut best = 0.0f32;
            let mut best_idx = 0usize;
            let mut idx = 0usize;
            for i in 0..D {
                let mi = m[i];
                let row_len = D - i;
                for (off, (sv, mj)) in s[idx..idx + row_len]
                    .iter()
                    .zip(m[i..D].iter())
                    .enumerate()
                {
                    let c = sv * inv - mi * mj * inv2;
                    let a = c.abs();
                    sum += a;
                    if a > best {
                        best = a;
                        best_idx = idx + off;
                    }
                }
                idx += row_len;
            }
            level[k] = sum;
            argmax[k] = best_idx as f32;
        }
    }

    /// `δ_surprise` readout: per ADJACENT bank pair (fast − slow, `K−1`
    /// values), `Σ_ij |C^p_ij − C^q_ij|` — the relation CHANGE signal; plus
    /// the max over pairs in the last slot (`K` entries total, debug-asserted).
    ///
    /// Spikes on correlation flips AND decouplings, settles after; blind to
    /// marginal movement by construction once the per-dim means settle (the
    /// G1a arms pin both directions). Zeros before any observation.
    pub fn relation_surprise_into(&self, out: &mut [f32]) {
        debug_assert!(out.len() >= K);
        let tri = Self::tri_len();
        let mut max = 0.0f32;
        for (p, out_p) in out.iter_mut().enumerate().take(K.saturating_sub(1)) {
            let q = p + 1;
            let (mp, mq) = (self.mass[p], self.mass[q]);
            let d = if mp == 0.0 || mq == 0.0 {
                0.0
            } else {
                let ip = 1.0 / mp;
                let ip2 = ip * ip;
                let iq = 1.0 / mq;
                let iq2 = iq * iq;
                let sp = &self.sums[p * tri..(p + 1) * tri];
                let sq = &self.sums[q * tri..(q + 1) * tri];
                let mp_v = &self.means[p * D..(p + 1) * D];
                let mq_v = &self.means[q * D..(q + 1) * D];
                let mut sum = 0.0f32;
                let mut idx = 0usize;
                for i in 0..D {
                    let pi = mp_v[i];
                    let qi = mq_v[i];
                    let row_len = D - i;
                    sum += sp[idx..idx + row_len]
                        .iter()
                        .zip(sq[idx..idx + row_len].iter())
                        .zip(mp_v[i..D].iter())
                        .zip(mq_v[i..D].iter())
                        .map(|(((spv, sqv), pj), qj)| {
                            let cp = spv * ip - pi * pj * ip2;
                            let cq = sqv * iq - qi * qj * iq2;
                            (cp - cq).abs()
                        })
                        .sum::<f32>();
                    idx += row_len;
                }
                sum
            };
            *out_p = d;
            if d > max {
                max = d;
            }
        }
        if K >= 2 {
            out[K - 1] = max;
        } else {
            out[0] = 0.0;
        }
    }

    /// Convenience: the max-over-adjacent-pairs `δ_surprise` scalar (the
    /// detection statistic the G1a gate asserts on).
    #[inline]
    pub fn surprise_max(&self) -> f32 {
        let mut out = [0.0f32; K];
        self.relation_surprise_into(&mut out);
        out[K - 1]
    }

    /// Convenience: the max-over-banks `δ_level` scalar.
    #[inline]
    pub fn level_max(&self) -> f32 {
        let mut level = [0.0f32; K];
        let mut argmax = [0.0f32; K];
        self.relation_level_into(&mut level, &mut argmax);
        level.iter().copied().fold(0.0f32, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::temporal::TemporalDerivativeKernel;

    /// Uniform f32 in `[lo, hi)` from the in-crate XorShift64 (SplitMix64-
    /// decorrelated seeds), bounded to the belief range.
    struct Stream {
        rng: Rng,
    }
    impl Stream {
        fn new(seed: u64) -> Self {
            Self { rng: Rng::new(seed) }
        }
        fn u(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (self.rng.next() >> 40) as f32 / 16_777_216.0 * (hi - lo)
        }
    }

    /// T3a — the A.3 horizon-invariance law. A constant stream must read its
    /// normalized mean EXACTLY at every horizon (the mass normalizer divides
    /// by the linear weight sum, never the count — a count normalizer reads
    /// `c·mass/n ≈ 0.45c` at n=8), while the raw mass tracks the observation
    /// count (strictly increasing, closed-form `(1−λⁿ)/d`, converging to `1/d`).
    #[test]
    fn horizon_invariance_and_mass_tracks_count() {
        const C: f32 = 0.5;
        let mut short = SyncBank::<8, 3>::new(SYNC_BANK_LADDER);
        let mut long = SyncBank::<8, 3>::new(SYNC_BANK_LADDER);
        let z = [C; 8];
        for _ in 0..8 {
            short.observe(&z);
        }
        for _ in 0..64 {
            long.observe(&z);
        }

        let ms = *short.mass();
        let ml = *long.mass();
        for k in 0..3 {
            // normalized mean == c at both horizons (≤ 1e-6 divergence)
            let inv_s = 1.0 / ms[k];
            let inv_l = 1.0 / ml[k];
            for i in 0..8 {
                let ns = short.means()[k * 8 + i] * inv_s;
                let nl = long.means()[k * 8 + i] * inv_l;
                assert!(
                    (ns - C).abs() <= 1e-6 && (nl - C).abs() <= 1e-6,
                    "bank {k} dim {i}: short {ns} long {nl} vs {C}"
                );
                assert!((ns - nl).abs() <= 1e-6, "T vs 8T divergence {ns} vs {nl}");
            }
            // mass tracks the observation count: increasing, closed-form, → 1/d
            let d = SYNC_BANK_LADDER[k];
            let lam = (1.0 - d) as f64;
            let closed = (1.0 - lam.powi(8)) / d as f64;
            assert!(
                (ms[k] as f64 - closed).abs() <= 1e-5,
                "bank {k}: mass(8)={} vs closed form {closed}",
                ms[k]
            );
            assert!(ms[k] < ml[k], "mass must track observation count");
            // the recursion IS the closed form (1 − λⁿ)/d at every n — pin it
            let closed64 = (1.0 - lam.powi(64)) / d as f64;
            assert!(
                (ml[k] as f64 - closed64).abs() <= 1e-4,
                "bank {k}: mass(64)={} vs closed form {closed64}",
                ml[k]
            );
            // convergence toward 1/d — exact for the fast bank at n=64
            // (λ⁶⁴ ≈ 1e-8), loose-scaled for slower banks (λ⁶⁴ = 1.6% of 1/d)
            let target = 1.0 / d;
            assert!(
                (ml[k] - target).abs() <= (target as f64 * lam.powi(64)) as f32 + 1e-3,
                "bank {k}: mass(64)={} should approach 1/d = {target}",
                ml[k]
            );
        }
    }

    /// T3b — exactness: the retention-form accumulator IS the λ-weighted batch
    /// covariance. Compare every packed entry of every bank against an f64
    /// batch reference on a fixed correlated stream (≤ 1e-5).
    #[test]
    fn covariance_matches_batch_weighted_reference() {
        const N: usize = 256;
        let mut st = Stream::new(0x9E37_79B9_7F4A_7C15);
        let mut stream = [[0.0f32; 8]; N];
        for row in stream.iter_mut().take(N) {
            let a = st.u(0.1, 0.9);
            row[0] = a;
            // positive coupling into dim 1, negative into dim 2 (values stay
            // inside [0, 1] — the domain the ¼-bound assumes)
            row[1] = (0.15 + (a - 0.5) * 0.5).clamp(0.05, 0.95);
            row[2] = (0.85 - (a - 0.5) * 0.5).clamp(0.05, 0.95);
            for v in row[3..8].iter_mut() {
                *v = st.u(0.1, 0.9);
            }
        }

        let mut bank = SyncBank::<8, 3>::new(SYNC_BANK_LADDER);
        for row in &stream {
            bank.observe(row);
        }

        let tri = SyncBank::<8, 3>::tri_len();
        for k in 0..3 {
            let lam = (1.0 - SYNC_BANK_LADDER[k]) as f64;
            let ws: f64 = (0..N).map(|t| lam.powi((N - 1 - t) as i32)).sum();
            let mut mu = [0.0f64; 8];
            for (i, mu_i) in mu.iter_mut().enumerate() {
                *mu_i = (0..N)
                    .map(|t| lam.powi((N - 1 - t) as i32) * stream[t][i] as f64)
                    .sum::<f64>()
                    / ws;
            }
            for i in 0..8 {
                for j in i..8 {
                    let c_ref = (0..N)
                        .map(|t| {
                            lam.powi((N - 1 - t) as i32)
                                * (stream[t][i] as f64 - mu[i])
                                * (stream[t][j] as f64 - mu[j])
                        })
                        .sum::<f64>()
                        / ws;
                    let c_bank = bank.cov(k, i, j);
                    assert!(
                        (c_bank as f64 - c_ref).abs() <= 1e-5,
                        "bank {k} ({i},{j}): {c_bank} vs {c_ref}"
                    );
                }
            }
            // and the streaming level readout agrees with a recomputed sum
            let mut level = [0.0f32; 3];
            let mut argmax = [0.0f32; 3];
            bank.relation_level_into(&mut level, &mut argmax);
            let recomputed: f32 = (0..8)
                .map(|i| (i..8).map(|j| bank.cov(k, i, j).abs()).sum::<f32>())
                .sum();
            assert!((level[k] - recomputed).abs() <= 1e-4);
            let (ai, aj) = packed_unpack(8, argmax[k] as usize);
            assert!(ai <= aj && aj < 8);
            assert!(argmax[k] < tri as f32);
        }
    }

    /// One synthetic tick: a rank-1 pack coupling. Every dim shares the
    /// latent `u` signed per dim by `signs[i]` + small independent noise;
    /// changing the sign VECTOR mirrors part of the coupling structure
    /// around 0.5 — cross-group pairs flip covariance sign while every
    /// dim's marginal is HELD (u is symmetric around 0.5). Returns
    /// `(δ_surprise max, Plan-277 value-surprise norm)` for the tick.
    ///
    /// Note the flip must be PARTIAL: negating every sign is a covariance
    /// no-op (cov(s_i·u, s_j·u) = s_i·s_j·var(u) — (−1)(−1) = +1), which is
    /// exactly why the (0,1) sign assert below is load-bearing.
    fn flip_tick(
        signs: &[f32; 8],
        bank: &mut SyncBank<8, 3>,
        tk: &mut TemporalDerivativeKernel<8>,
        st: &mut Stream,
    ) -> (f32, f32) {
        let u = st.u(0.2, 0.8);
        let mut z = [0.5f32; 8];
        for (i, zi) in z.iter_mut().enumerate() {
            let e = st.u(-0.03, 0.03);
            *zi = (0.5 + signs[i] * (u - 0.5) + e).clamp(0.0, 1.0);
        }
        bank.observe(&z);
        let sv = tk.observe(&z);
        let vn = sv.iter().map(|x| x * x).sum::<f32>().sqrt();
        (bank.surprise_max(), vn)
    }

    /// T4 arm (a) — G1a flip arm: the pack's rank-1 coupling flips sign
    /// (+ρ ↔ −ρ on every pair) with marginals HELD. `δ_surprise` must spike
    /// on the flip and settle after; the Plan-277 first-order value-surprise
    /// stays comparatively quiet (its transient is one EMA lag, the relation
    /// signal persists for the slow bank's horizon).
    #[test]
    fn g1a_correlation_flip_spikes_surprise_and_value_surprise_stays_quiet() {
        let mut st = Stream::new(0xDEADBEEF);
        let mut bank = SyncBank::<8, 3>::new(SYNC_BANK_LADDER);
        // Plan-277 baseline: the shipped dual-EMA value-surprise kernel.
        let mut tk = TemporalDerivativeKernel::<8>::new(0.3, 0.03);

        const WARM: usize = 200;
        const BASE_WIN: usize = 128;
        const POST: usize = 260;
        // phase 1: all dims load +u; phase 2: dim 1 (and 3,5,7) load −u —
        // the cross-group pairs (e.g. (0,1)) flip +ρ → −ρ, within-group pairs
        // hold. A whole-vector negation would leave every covariance intact.
        const PLUS: [f32; 8] = [1.0; 8];
        const MINUS: [f32; 8] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

        for _ in 0..WARM {
            let _ = flip_tick(&PLUS, &mut bank, &mut tk, &mut st);
        }
        let mut base = Vec::new();
        for _ in 0..BASE_WIN {
            base.push(flip_tick(&PLUS, &mut bank, &mut tk, &mut st));
        }
        let base_surprise_max = base.iter().map(|(s, _)| *s).fold(0.0f32, f32::max);
        let base_surprise_mean: f32 =
            base.iter().map(|(s, _)| *s).sum::<f32>() / base.len() as f32;
        let base_value = base.iter().map(|(_, v)| *v).fold(0.0f32, f32::max);

        // post-flip window (anti-coupled)
        let mut post = Vec::new();
        for _ in 0..POST {
            post.push(flip_tick(&MINUS, &mut bank, &mut tk, &mut st));
        }
        let peak_surprise = post.iter().map(|(s, _)| *s).fold(0.0f32, f32::max);

        // (0) ground truth: the (0,1) covariance SIGN flips — fast bank reads
        // negative after the transient, and the entry's magnitude survives
        // (Σ|C| alone cannot see this: it dips through zero and returns — the
        // reason the δ split exists).
        let c01_fast = bank.cov(0, 0, 1);
        assert!(
            c01_fast < 0.0,
            "post-flip fast-bank C(0,1) should read negative, got {c01_fast}"
        );

        // (1) the flip must SPIKE above the stationary floor: peak ≥ 3× the
        // baseline MEAN (the max statistic is tail-heavy — comparing peak to
        // peak would measure the noise tail, not the signal).
        assert!(
            peak_surprise >= 3.0 * base_surprise_mean,
            "δ_surprise peak {peak_surprise} < 3× baseline mean {base_surprise_mean} (baseline max {base_surprise_max})"
        );

        // (2) PERSISTENCE is the discriminator: integrate the elevation over
        // the post window. The first-order value-surprise decays with its
        // ~3-tick EMA lag; the relation signal holds for the slow bank's
        // ~64-tick horizon. Compare window integrals above each own baseline.
        let int_surprise: f32 = post
            .iter()
            .map(|(s, _)| (s - base_surprise_mean).max(0.0))
            .sum();
        let int_value: f32 = post.iter().map(|(_, v)| (v - base_value).max(0.0)).sum();
        assert!(
            int_surprise >= 4.0 * int_value.max(1e-6),
            "relation elevation {int_surprise} should dominate value elevation {}",
            int_value.max(1e-6)
        );

        // (3) settle: the last 64 ticks sit back near baseline (≤ 2× mean)
        let tail = &post[post.len() - 64..];
        let tail_mean: f32 = tail.iter().map(|(s, _)| *s).sum::<f32>() / tail.len() as f32;
        assert!(
            tail_mean <= 2.0 * base_surprise_mean,
            "δ_surprise did not settle: tail mean {tail_mean} vs baseline mean {base_surprise_mean}"
        );
    }

    /// T4 arm (b) — negative control: independent marginals step at DIFFERENT
    /// ticks (no co-movement). Once the per-dim means settle (~3/d ticks), the
    /// relation signal must be quiet — marginal movement alone is invisible.
    #[test]
    fn g1b_independent_marginal_steps_stay_quiet_after_settle() {
        let mut st = Stream::new(0x0BADC0DE);
        let mut bank = SyncBank::<8, 3>::new(SYNC_BANK_LADDER);

        const STEP_A: usize = 300;
        const STEP_B: usize = 460;
        const SETTLE: usize = 250; // > 3/d_slow = 192
        const WIN: usize = 128;

        let mut trace: Vec<f32> = Vec::new();
        let (mut z0_base, mut z1_base) = (0.3f32, 0.3f32);
        for t in 0..(STEP_B + SETTLE + WIN) {
            if t == STEP_A {
                z0_base = 0.7;
            }
            if t == STEP_B {
                z1_base = 0.7;
            }
            let mut z = [0.5f32; 8];
            z[0] = (z0_base + st.u(-0.05, 0.05)).clamp(0.0, 1.0);
            z[1] = (z1_base + st.u(-0.05, 0.05)).clamp(0.0, 1.0);
            for dim in z.iter_mut().take(8).skip(2) {
                *dim = st.u(0.1, 0.9);
            }
            bank.observe(&z);
            trace.push(bank.surprise_max());
        }
        let pre: f32 = trace[..STEP_A].iter().copied().fold(0.0f32, f32::max);
        let quiet = &trace[STEP_B + SETTLE..];
        let quiet_max = quiet.iter().copied().fold(0.0f32, f32::max);
        assert!(
            quiet_max <= 2.0 * pre.max(1e-6),
            "relation signal did not stay quiet after independent steps: {quiet_max} vs baseline {}",
            pre.max(1e-6)
        );
    }

    /// Boundary law: the packed index helpers round-trip and tile the triangle.
    #[test]
    fn packed_index_round_trips() {
        for d in [1usize, 2, 3, 8, 64] {
            let tri = d * (d + 1) / 2;
            let mut seen = vec![false; tri];
            for i in 0..d {
                for j in i..d {
                    let idx = packed_index(d, i, j);
                    assert!(idx < tri);
                    assert!(!seen[idx], "duplicate at ({i},{j}) d={d}");
                    seen[idx] = true;
                    assert_eq!(packed_unpack(d, idx), (i, j));
                }
            }
        }
    }

    /// Pre-first-observation readouts are zeros (never NaN from mass 0).
    #[test]
    fn unarmed_readouts_are_zero() {
        let bank = SyncBank::<8, 3>::new(SYNC_BANK_LADDER);
        assert!(!bank.observed());
        assert_eq!(bank.surprise_max(), 0.0);
        assert_eq!(bank.level_max(), 0.0);
        assert_eq!(bank.cov(0, 0, 1), 0.0);
    }

    /// The ladder law is enforced loud (0 < d < 1).
    #[test]
    #[should_panic(expected = "strictly in (0, 1)")]
    fn ladder_rejects_degenerate_steps() {
        let _ = SyncBank::<8, 2>::new([0.25, 0.0]);
    }
}
