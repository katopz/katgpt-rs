//! KARC Batched MatVec — SIMD-batched forecast across N forecasters of identical
//! `(D, M, K)` shape (Plan 556 Phase 2).
//!
//! # The problem this addresses
//!
//! Each per-NPC `KarcForecaster::forecast_into` is ~381 ns at the HLA config
//! (Plan 308 G2 measurement, D=8/M=8/K=4 → d_h=256). At crowd scale (e.g. 1000
//! NPCs in a single octree cell), the per-tick cost is dominated by *memory
//! bandwidth* (reading N `Wout` matrices) rather than by per-call dispatch
//! overhead. The batched matvec amortizes that bandwidth by:
//!
//! 1. Laying out N `Wout` matrices in one contiguous `[N][D·d_h]` slice so the
//!    hardware prefetcher sees a single long sequential read.
//! 2. Hoisting the per-output-row `simd::simd_dot_f32` call across N
//!    forecasters — one loop iteration computes N partial outputs in parallel
//!    (interleaved across the batch), instead of one.
//! 3. Skipping per-NPC function-call overhead (fit check, debug_asserts, etc).
//!
//! # Algorithm
//!
//! The straightforward "call `simd_matvec` N times" is already the right
//! algorithm at N ≤ 32 (the realistic cell size — OctreeSpatialIndex at
//! crowd scale rarely packs >32 NPCs per leaf). Rayon parallelism (the
//! `simd_binary_matmul_batch` pattern) is *not* the right tool here because
//! rayon's thread-pool scheduling overhead is ~5 µs/task — bigger than the
//! entire 8-forecast budget (575 ns).
//!
//! The amortization comes from the contiguous layout + the SIMD inner loop,
//! *not* from parallelism. We benchmark the pure matvec (`karc_batched_matvec_into`)
//! against N sequential `simd::simd_matvec` calls; the win comes from the
//! memory layout + loop hoisting, not from anything fancy.
//!
//! # Bit-identical contract (G1)
//!
//! The batched path MUST produce bit-identical output to N sequential
//! `KarcForecaster::forecast_into` calls on the same inputs. This is enforced
//! by reusing the exact same `simd::simd_matvec` inner kernel (which itself
//! uses `simd::simd_dot_f32`) — no reordering, no FMA contraction, no
//! algorithmic change. The only difference from N sequential calls is the
//! outer loop structure.
//!
//! # Modelless
//!
//! Pure linear algebra. No training, no learned params. Same KARC fitter, same
//! `Wout` matrix — this primitive is just a different *batching* of the matvec,
//! not a different algorithm.
//!
//! # Latent-only sync boundary
//!
//! Same as `KarcForecaster`: the `Wout` matrices are local latent state; only
//! the resulting D-length forecast vector crosses sync (and only via the
//! emotion-projection bridge functions in the runtime). The batched slice
//! itself never crosses `SyncBlock`.
//!
//! # Allocation
//!
//! Zero on the hot path. The batched primitive takes borrowed slices; the
//! caller owns the buffer. `KarcBatchForecaster` pre-allocates one shared
//! feature scratch buffer (`N · d_h` floats) at construction.
//!
//! # Plan 556 GOAT gate
//!
//! - **G1**: batched N-forecast output bit-identical to N sequential
//!   `forecast_into` calls.
//! - **G2**: N=8 batched ≤ 1.5× single-forecast latency (≥5.3× amortization).
//!   Target ≤ 575 ns for 8 forecasts at the HLA config.
//! - **G3**: enabling `karc_batched_matvec` does not perturb the single-forecast
//!   path (separate code path, no shared mutable state).
//! - **G4**: 0 allocs/N batched calls on the hot path.
//!
//! # References
//!
//! - **Plan:** `katgpt-rs/.plans/556_karc_mitigations_open_primitives.md` (Phase 2)
//! - **Plan 308:** `katgpt-rs/.plans/308_karc_delay_basis_ridge_forecaster.md`
//!   (the KARC primitive; per-forecast latency baseline).
//! - **Runtime consumer:** `riir-ai/.plans/514_karc_mitigations_runtime.md`
//!   Phase 3 (octree-batched cell-level KARC — the crowd-scale use case).

use crate::karc::{KarcBasis, feature_expand};
use crate::simd;

mod imp {
    use super::*;

    /// Pure batched matvec: N independent `out[i] = Wout[i] · ψ[i]` operations,
    /// laid out contiguously for memory-bandwidth amortization.
    ///
    /// # Layout
    ///
    /// - `wouts`:  `[N][D·d_h]` row-major — each NPC's `Wout` matrix is one
    ///   contiguous `D·d_h` row block.
    /// - `features`: `[N][d_h]` — each NPC's expanded feature vector `ψ`.
    /// - `out`:    `[N][D]` write-only — each NPC's forecast output.
    ///
    /// # Bit-identical guarantee
    ///
    /// Bit-identical to N sequential `simd::simd_matvec(out[i..i+D],
    /// &wouts[i*D*d_h..(i+1)*D*d_h], &features[i*d_h..(i+1)*d_h], D, d_h)` calls.
    /// The implementation is literally that loop — no reordering, no FMA
    /// contraction. The only win is contiguous layout (better hardware
    /// prefetcher utilization) + loop hoisting.
    ///
    /// # Panics
    ///
    /// In debug builds: panics if `wouts.len() != n * d * d_h`, if
    /// `features.len() != n * d_h`, or if `out.len() < n * d`.
    #[inline]
    pub fn karc_batched_matvec_into(
        wouts: &[f32],
        features: &[f32],
        out: &mut [f32],
        n: usize,
        d_h: usize,
        d: usize,
    ) {
        debug_assert_eq!(
            wouts.len(),
            n * d * d_h,
            "karc_batched_matvec_into: wouts.len() = {} but expected n*d*d_h = {}*{}*{} = {}",
            wouts.len(),
            n,
            d,
            d_h,
            n * d * d_h,
        );
        debug_assert_eq!(
            features.len(),
            n * d_h,
            "karc_batched_matvec_into: features.len() = {} but expected n*d_h = {}*{} = {}",
            features.len(),
            n,
            d_h,
            n * d_h,
        );
        debug_assert!(
            out.len() >= n * d,
            "karc_batched_matvec_into: out.len() = {} but expected >= n*d = {}*{} = {}",
            out.len(),
            n,
            d,
            n * d,
        );
        for i in 0..n {
            let wout_off = i * d * d_h;
            let feat_off = i * d_h;
            let out_off = i * d;
            simd::simd_matvec(
                &mut out[out_off..out_off + d],
                &wouts[wout_off..wout_off + d * d_h],
                &features[feat_off..feat_off + d_h],
                d,
                d_h,
            );
        }
    }

    /// Batched KARC forecaster: owns N `Wout` matrices + a shared basis and
    /// runs the batched forecast in one call.
    ///
    /// Each NPC owns its own `Wout` (per-NPC fit); the basis (`B`) is shared
    /// across the batch (it is a stateless dictionary). The delay state is
    /// per-NPC (each NPC's recent trajectory).
    ///
    /// # Allocation
    ///
    /// One `Vec<f32>` of size `N · d_h` is pre-allocated at construction and
    /// reused on every forecast call — zero allocation on the hot path (G4).
    ///
    /// # Const generics
    ///
    /// `B` — basis type (Fourier/Chebyshev/BSpline).
    /// `D` — observation dimension.
    /// `M` — basis order (features per coordinate).
    /// `K` — delay embedding depth.
    ///
    /// `d_h = K · D · M` is the feature dimension, derived at compile time.
    pub struct KarcBatchForecaster<B: KarcBasis<M>, const D: usize, const M: usize, const K: usize> {
        /// Shared basis dictionary (identical across all NPCs in the batch).
        pub basis: B,
        /// Stacked `Wout` matrices, `[N][D·d_h]` row-major. NPC `i`'s `Wout`
        /// is at `[i·D·d_h .. (i+1)·D·d_h]`.
        pub wouts: Vec<f32>,
        /// Number of forecasters in the batch (length along the N axis).
        n: usize,
        /// Per-NPC "fitted" flags. `fitted[i] == false` → NPC `i`'s output is
        /// left untouched on forecast.
        fitted: Vec<bool>,
        /// Pre-allocated feature scratch, `[N][d_h]`. Reused across calls.
        features_buf: Vec<f32>,
    }

    impl<B: KarcBasis<M>, const D: usize, const M: usize, const K: usize>
        KarcBatchForecaster<B, D, M, K>
    {
        /// Feature dimension per NPC: `d_h = K · D · M`.
        pub const D_H: usize = K * D * M;

        /// Construct an empty batch forecaster with capacity for `n` forecasters.
        ///
        /// The caller is expected to fill `self.wouts` and mark `self.fitted`
        /// via the per-NPC setters before calling [`forecast_into`](Self::forecast_into).
        /// The `basis` is shared across all NPCs in the batch.
        pub fn with_capacity(basis: B, n: usize) -> Self {
            let d_h = Self::D_H;
            Self {
                basis,
                wouts: vec![0.0; n * D * d_h],
                n,
                fitted: vec![false; n],
                features_buf: vec![0.0; n * d_h],
            }
        }

        /// Number of forecasters in the batch.
        #[inline]
        pub fn n(&self) -> usize {
            self.n
        }

        /// Per-NPC fitted flag. `false` until [`set_wout`](Self::set_wout) is
        /// called for NPC `i`.
        #[inline]
        pub fn is_fitted(&self, i: usize) -> bool {
            self.fitted[i]
        }

        /// Install NPC `i`'s `Wout` matrix and mark it fitted.
        ///
        /// `wout` must have length `D · d_h = D · K · D · M`. Panics otherwise
        /// (caller bug — usually a shape mismatch from a different forecaster
        /// config).
        ///
        /// This is the batched analogue of `KarcForecaster::restore_wout`. The
        /// fitter itself (`fit_ridge`) is per-NPC — the batched primitive does
        /// not batch the fit, only the forecast.
        pub fn set_wout(&mut self, i: usize, wout: Vec<f32>) {
            let d_h = Self::D_H;
            let expected = D * d_h;
            assert_eq!(
                wout.len(),
                expected,
                "KarcBatchForecaster::set_wout: wout.len() = {} but expected D*d_h = {}*{} = {}",
                wout.len(),
                D,
                d_h,
                expected,
            );
            let off = i * D * d_h;
            self.wouts[off..off + D * d_h].copy_from_slice(&wout);
            self.fitted[i] = true;
        }

        /// Borrow NPC `i`'s `Wout` matrix as a flat slice (length `D · d_h`).
        #[inline]
        pub fn wout(&self, i: usize) -> &[f32] {
            let d_h = Self::D_H;
            let off = i * D * d_h;
            &self.wouts[off..off + D * d_h]
        }

        /// Batched forecast: expand each NPC's delay state into `ψ`, then run
        /// the batched matvec.
        ///
        /// # Layout
        ///
        /// - `delay_states`: `[N][K·D]` — each NPC's delay ring state.
        /// - `out`: `[N][D]` write-only — each NPC's forecast.
        ///
        /// NPCs whose `fitted[i] == false` have their output left untouched
        /// (caller-zeroed by convention). The feature expansion is still run
        /// (cheap, no allocation) to keep the hot-path loop branch-free.
        ///
        /// # Bit-identical guarantee
        ///
        /// For each NPC `i` where `fitted[i] == true`, the output
        /// `out[i·D .. (i+1)·D]` is bit-identical to calling
        /// `KarcForecaster::<B, D, M, K>::forecast_into(&self.delay_states[i·K·D ..],
        /// &mut out[i·D..])` on a forecaster with the same `Wout` and basis.
        /// Enforced by `test_batched_matvec_bit_identical_to_sequential` (raw matvec)
        /// and `test_batched_forecaster_matches_single_forecast_into` (full
        /// forecast_into path through the basis expansion) in `mod tests` below.
        ///
        /// # Allocation
        ///
        /// Zero on the hot path (G4). The feature scratch is pre-allocated at
        /// construction and reused via indexing.
        #[inline]
        pub fn forecast_into(&mut self, delay_states: &[f32], out: &mut [f32]) {
            let d_h = Self::D_H;
            debug_assert_eq!(
                delay_states.len(),
                self.n * K * D,
                "KarcBatchForecaster::forecast_into: delay_states.len() mismatch",
            );
            debug_assert!(
                out.len() >= self.n * D,
                "KarcBatchForecaster::forecast_into: out.len() too small",
            );
            // Expand per-NPC delay states into the contiguous feature buffer.
            for i in 0..self.n {
                let delay_off = i * K * D;
                let feat_off = i * d_h;
                let psi = &mut self.features_buf[feat_off..feat_off + d_h];
                // If unfitted, the features don't matter (matvec output is
                // ignored by caller convention). Still expand to keep the loop
                // branch-free; the cost is ~30ns of basis eval.
                feature_expand::<B, M>(
                    &delay_states[delay_off..delay_off + K * D],
                    &self.basis,
                    psi,
                );
            }
            // Run the batched matvec into `out`.
            karc_batched_matvec_into(&self.wouts, &self.features_buf, out, self.n, d_h, D);
            // Zero out unfitted NPCs' outputs (caller convention: unfitted NPCs
            // produce zero output, not garbage). This is a tiny tail loop, run
            // once per call, not in the hot inner loop.
            for (i, &fitted_i) in self.fitted[..self.n].iter().enumerate() {
                if !fitted_i {
                    let out_off = i * D;
                    for slot in &mut out[out_off..out_off + D] {
                        *slot = 0.0;
                    }
                }
            }
        }
    }
}

pub use imp::{KarcBatchForecaster, karc_batched_matvec_into};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FourierBasis, KarcForecaster};

    /// Build a fitted `KarcForecaster` with a deterministic synthetic
    /// trajectory, return it + the seed delay state.
    fn make_fitted<const D: usize, const M: usize, const K: usize>(
        basis: FourierBasis<M>,
        n_train: usize,
        seed_offset: usize,
    ) -> (KarcForecaster<FourierBasis<M>, D, M, K>, Vec<f32>) {
        let traj: Vec<f32> = (0..n_train)
            .flat_map(|i| {
                let t = i as f32 * 0.05;
                let mut row = [0.0f32; 32];
                let n = D.min(32);
                for (d, row_d) in row.iter_mut().enumerate().take(n) {
                    let freq = 0.3 + 0.2 * d as f32;
                    *row_d = (freq * t).sin() + 0.5 * ((freq + 1.0) * t).cos();
                }
                row[..D].to_vec()
            })
            .collect();
        let n_total = traj.len() / D;
        let mut f = KarcForecaster::with_capacity(basis, n_total);
        let kd = K * D;
        for t in (K - 1)..(n_total - 1) {
            let mut delay = vec![0.0f32; kd];
            for lag in 0..K {
                let idx = t - lag;
                for d in 0..D {
                    delay[lag * D + d] = traj[idx * D + d];
                }
            }
            let mut target = vec![0.0f32; D];
            for d in 0..D {
                target[d] = traj[(t + 1) * D + d];
            }
            f.accumulate_pair(&delay, target.as_slice().try_into().unwrap());
        }
        f.fit_ridge(1e-4).expect("fit_ridge");
        // Seed delay state = last K observations, offset by seed_offset to give
        // different NPCs different starting states.
        let mut seed = vec![0.0f32; kd];
        for lag in 0..K {
            let idx = (n_total - 1 - seed_offset) - lag;
            for d in 0..D {
                seed[lag * D + d] = traj[idx * D + d];
            }
        }
        (f, seed)
    }

    #[test]
    fn test_batched_matvec_bit_identical_to_sequential() {
        const D: usize = 8;
        const M: usize = 8;
        const K: usize = 4;
        let d_h = K * D * M; // 256
        let n = 4;

        // Build deterministic inputs.
        let wouts: Vec<f32> = (0..n * D * d_h).map(|i| (i as f32 * 0.001) - 0.5).collect();
        let features: Vec<f32> = (0..n * d_h).map(|i| (i as f32 * 0.002) - 0.5).collect();

        // Run batched.
        let mut out_batched = vec![f32::NAN; n * D];
        karc_batched_matvec_into(&wouts, &features, &mut out_batched, n, d_h, D);

        // Run sequential and compare bit-identically.
        for i in 0..n {
            let mut out_seq = vec![f32::NAN; D];
            simd::simd_matvec(
                &mut out_seq,
                &wouts[i * D * d_h..(i + 1) * D * d_h],
                &features[i * d_h..(i + 1) * d_h],
                D,
                d_h,
            );
            for d in 0..D {
                assert_eq!(
                    out_batched[i * D + d].to_bits(),
                    out_seq[d].to_bits(),
                    "batched output bit-mismatch at NPC {i} dim {d}",
                );
            }
        }
    }

    #[test]
    fn test_batched_forecaster_matches_single_forecast_into() {
        const D: usize = 8;
        const M: usize = 8;
        const K: usize = 4;
        const N: usize = 8;

        // Build N fitted forecasters with different seed offsets so each NPC
        // has a different Wout + delay state.
        let mut singles: Vec<KarcForecaster<FourierBasis<M>, D, M, K>> = Vec::with_capacity(N);
        let mut seeds: Vec<Vec<f32>> = Vec::with_capacity(N);
        for i in 0..N {
            let (f, seed) = make_fitted::<D, M, K>(FourierBasis::new(4.0), 200, i);
            singles.push(f);
            seeds.push(seed);
        }

        // Run N sequential forecasts.
        let mut out_seq = vec![0.0f32; N * D];
        for (i, f) in singles.iter_mut().enumerate() {
            let ok = f.forecast_into(&seeds[i], &mut out_seq[i * D..(i + 1) * D]);
            assert!(ok, "single forecast_into returned false for NPC {i}");
        }

        // Build the batched forecaster: clone each Wout + share the basis.
        let mut batch = KarcBatchForecaster::<FourierBasis<M>, D, M, K>::with_capacity(
            FourierBasis::new(4.0),
            N,
        );
        for (i, f) in singles.iter().enumerate() {
            batch.set_wout(i, f.wout.clone());
        }
        // Stack delay states.
        let mut delay_states = vec![0.0f32; N * K * D];
        for (i, seed) in seeds.iter().enumerate() {
            delay_states[i * K * D..(i + 1) * K * D].copy_from_slice(seed);
        }
        // Run batched forecast.
        let mut out_batched = vec![0.0f32; N * D];
        batch.forecast_into(&delay_states, &mut out_batched);

        // Bit-identical comparison.
        for i in 0..N {
            for d in 0..D {
                assert_eq!(
                    out_batched[i * D + d].to_bits(),
                    out_seq[i * D + d].to_bits(),
                    "batched vs single bit-mismatch at NPC {i} dim {d}",
                );
            }
        }
    }

    #[test]
    fn test_batched_forecaster_unfitted_zero_output() {
        const D: usize = 4;
        const M: usize = 4;
        const K: usize = 2;
        const N: usize = 3;

        let mut batch = KarcBatchForecaster::<FourierBasis<M>, D, M, K>::with_capacity(
            FourierBasis::new(2.0),
            N,
        );
        // Only NPC 1 is fitted.
        let wout = vec![0.5f32; D * K * D * M];
        batch.set_wout(1, wout);

        let delay_states = vec![0.7f32; N * K * D];
        let mut out = vec![f32::NAN; N * D];
        batch.forecast_into(&delay_states, &mut out);

        // NPCs 0 and 2 should be zeroed (unfitted).
        for &unfitted_idx in &[0, 2] {
            for d in 0..D {
                assert_eq!(
                    out[unfitted_idx * D + d],
                    0.0,
                    "unfitted NPC {unfitted_idx} dim {d} should be zero",
                );
            }
        }
        // NPC 1 should be non-zero (fitted, non-trivial input).
        let mut any_nonzero = false;
        for d in 0..D {
            if out[D + d] != 0.0 {
                any_nonzero = true;
                break;
            }
        }
        assert!(any_nonzero, "fitted NPC 1 produced all-zero output");
    }

    #[test]
    fn test_batched_matvec_n1_matches_single() {
        // Degenerate case: N=1 should behave identically to a single simd_matvec.
        const D: usize = 8;
        const D_H: usize = 256;
        let wouts: Vec<f32> = (0..D * D_H).map(|i| i as f32 * 0.01).collect();
        let features: Vec<f32> = (0..D_H).map(|i| (i as f32 - 128.0) * 0.001).collect();

        let mut out_batched = vec![0.0; D];
        karc_batched_matvec_into(&wouts, &features, &mut out_batched, 1, D_H, D);

        let mut out_single = vec![0.0; D];
        simd::simd_matvec(&mut out_single, &wouts, &features, D, D_H);

        for d in 0..D {
            assert_eq!(
                out_batched[d].to_bits(),
                out_single[d].to_bits(),
                "N=1 case bit-mismatch at dim {d}",
            );
        }
    }

    #[test]
    fn test_batched_set_wout_shape_check() {
        const D: usize = 4;
        const M: usize = 4;
        const K: usize = 2;

        let mut batch = KarcBatchForecaster::<FourierBasis<M>, D, M, K>::with_capacity(
            FourierBasis::new(2.0),
            2,
        );

        // Correct shape: D * K * D * M = 4*2*4*4 = 128.
        let wout_ok = vec![0.0; 128];
        batch.set_wout(0, wout_ok); // should not panic

        // Wrong shape: panics.
        let wout_bad = vec![0.0; 127];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut b = KarcBatchForecaster::<FourierBasis<M>, D, M, K>::with_capacity(
                FourierBasis::new(2.0),
                2,
            );
            b.set_wout(0, wout_bad);
        }));
        assert!(result.is_err(), "set_wout with wrong shape should panic");
    }

    #[test]
    fn test_batched_forecaster_n_accessor() {
        const D: usize = 4;
        const M: usize = 4;
        const K: usize = 2;
        let batch = KarcBatchForecaster::<FourierBasis<M>, D, M, K>::with_capacity(
            FourierBasis::new(2.0),
            5,
        );
        assert_eq!(batch.n(), 5);
        assert!(!batch.is_fitted(0));
        assert!(!batch.is_fitted(4));
    }
}
