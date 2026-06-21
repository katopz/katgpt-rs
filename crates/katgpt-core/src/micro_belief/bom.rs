//! `BoMSampler` — Best-of-Many single-pass K-hypothesis belief sampling.
//! (Plan 281, Research 248 §2.4. Behind the `bom_sampling` feature.)
//!
//! Injects K Gaussian noise queries at the kernel input site and evaluates
//! the kernel K times against ONE precomputed base activation, producing K
//! diverse next-belief-states in a single batched call. The deterministic
//! [`MicroRecurrentBeliefState::step()`](super::types::MicroRecurrentBeliefState::step)
//! path is unchanged.
//!
//! # The "single batched matvec" (Research 248 §2.4, honest accounting)
//!
//! For Family A the base activation `act[i] = W_s[i]·s + W_x[i]·x + b[i]`
//! (D dot products) is computed ONCE. The K queries then perturb the
//! pre-sigmoid activation elementwise:
//!   `out[k*D + i] = clamp(2·σ(act[i] + queries[k*D + i]) − 1, ±clamp)`
//! The expensive D×D matvec happens once; the K × (D adds + D sigmoids) is
//! elementwise and auto-vectorises. This is the single-pass property.
//!
//! # σ=0 degeneracy
//!
//! With `queries = [0.0; K*D]`, BoM reproduces the deterministic `step()`
//! output exactly (G1.3). BoM is a strict superset of the deterministic path.
//!
//! # Latent vs raw boundary
//!
//! The K hypotheses are LATENT and LOCAL — never synced. Only a caller-chosen
//! selection (mean, or [`BoMSampler::select_best`] winner) projects to synced
//! scalars. Syncing the K-vector distribution would violate the AGENTS.md
//! bandwidth rule.
//!
//! # Object safety
//!
//! [`BoMSampler::select_best`] takes `impl Fn(&[f32]) -> f32`, so this trait is
//! NOT object-safe. BoM is opt-in and used generically (the caller knows the
//! concrete kernel type), so this is acceptable — it is never dispatched
//! through `&dyn BoMSampler`.

use crate::micro_belief::types::MicroRecurrentBeliefState;
use crate::micro_belief::{assume_init_slice, uninit_stack};
use crate::simd::simd_dot_f32;

#[cfg(not(feature = "simd_sigmoid"))]
use crate::simd::fast_sigmoid;

#[cfg(feature = "simd_sigmoid")]
use crate::simd::simd_sigmoid_tanh_clamp_inplace;

// ─────────────────────────────────────────────────────────────────────────────
// NoiseQueryConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Noise query distribution config for [`BoMSampler`]. Versioned as a companion
/// artifact to `MicroRecurrentKernelSnapshot` (own BLAKE3 commitment, see
/// [`commit`](Self::commit)) — not embedded in the kernel snapshot itself, to
/// avoid bumping `SNAPSHOT_VERSION` and breaking Plan 276's G1.5 tests
/// (decision: Plan 281 T0.5).
///
/// σ is per-NPC-class and freeze/thaw-able via the commitment. The paper's
/// σ=0.02 is tuned for DINOv3 features; our [-1,1] HLA space likely needs
/// σ≈0.1–0.5 (G1.2 distinctness test will catch near-identical hypotheses).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct NoiseQueryConfig {
    /// Gaussian noise stddev. Paper default 0.02; needs calibration (R3).
    pub sigma: f32,
    /// Number of hypotheses K. Paper trains K=256, evals K=20; plasma-tier
    /// budget caps at K=8 (1000 NPCs × 20 Hz, µs budget).
    pub k: usize,
    /// How seeds are derived per-NPC vs per-class.
    pub seed_strategy: SeedStrategy,
}

/// Seed strategy for noise query generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum SeedStrategy {
    /// Per-NPC seed derived from a v7 UUID (AGENTS.md). Maximises diversity.
    PerNpc = 0,
    /// Shared per-NPC-class seed. Cheaper, less diverse — same noise pattern
    /// across all NPCs of a class.
    PerClass = 1,
}

impl Default for NoiseQueryConfig {
    fn default() -> Self {
        // G1.2 will likely require raising sigma from 0.02 for [-1,1] space;
        // 0.1 is the conservative starting point that keeps hypotheses
        // distinct without saturating the sigmoid.
        Self {
            sigma: 0.1,
            k: 8,
            seed_strategy: SeedStrategy::PerNpc,
        }
    }
}

impl NoiseQueryConfig {
    /// Builder for sigma.
    #[inline]
    pub fn with_sigma(mut self, sigma: f32) -> Self {
        self.sigma = sigma;
        self
    }

    /// Builder for k.
    #[inline]
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Builder for seed_strategy.
    #[inline]
    pub fn with_seed_strategy(mut self, s: SeedStrategy) -> Self {
        self.seed_strategy = s;
        self
    }

    /// BLAKE3 commitment over `(sigma_le || k_le || seed_strategy_byte)`.
    ///
    /// Companion artifact to `MicroRecurrentKernelSnapshot`
    /// (`super::snapshot::MicroRecurrentKernelSnapshot`) — callers SHOULD embed
    /// this alongside the kernel snapshot commitment in the hot-swap audit
    /// event, so σ/K are freeze/thaw-able and tamper-evident without changing
    /// the kernel snapshot's own commitment scheme (which would bump
    /// `SNAPSHOT_VERSION` and break Plan 276 G1.5 atomicity tests — see Plan
    /// 281 T0.5 for the rationale).
    pub fn commit(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.sigma.to_le_bytes());
        hasher.update(&(self.k as u64).to_le_bytes());
        hasher.update(&[self.seed_strategy as u8]);
        *hasher.finalize().as_bytes()
    }

    /// Verify a stored commitment against this config.
    pub fn verify(&self, stored: &[u8; 32]) -> bool {
        &self.commit() == stored
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BoMSampler trait
// ─────────────────────────────────────────────────────────────────────────────

/// K-hypothesis belief sampling (Research 248, Plan 281). Extends
/// [`MicroRecurrentBeliefState`] with a single batched call that produces K
/// diverse next-belief-states.
///
/// See the module-level docs for the single-pass matvec accounting and the
/// σ=0 degeneracy property.
pub trait BoMSampler: MicroRecurrentBeliefState {
    /// Sample K diverse next-states from `(s_prev, x)` in one batched call.
    ///
    /// `queries` is a `[K * D]` row-major slice where `D = self.dim()`. Each
    /// row `queries[k*D .. (k+1)*D]` is a noise vector `q_k ~ N(0, σ²I)` (the
    /// caller pre-samples it from their RNG of choice; σ lives in `cfg` for
    /// commitment/provenance only — the actual noise values are in `queries`).
    /// Writes K next-states into `out` (`[K * D]`, row-major, caller-allocated).
    ///
    /// # Zero-allocation
    ///
    /// Reads `s_prev`, `x`, `queries`; writes `out`. No allocation.
    ///
    /// # σ=0 degeneracy (G1.3)
    ///
    /// If every element of `queries` is zero, `out[k*D..(k+1)*D]` reproduces
    /// the deterministic `step()` output for all k.
    fn sample_k_states(
        &self,
        s_prev: &[f32],
        x: &[f32],
        queries: &[f32], // [K * D], row-major
        out: &mut [f32], // [K * D], row-major
        cfg: &NoiseQueryConfig,
    );

    /// Select the best hypothesis index by a caller-provided scorer.
    ///
    /// `scorer` is applied to each of the K row-slices
    /// `hypotheses[k*D..(k+1)*D]`; returns the index of the maximum score.
    /// Ties resolve to the lowest index.
    fn select_best(
        &self,
        hypotheses: &[f32], // [K * D]
        scorer: impl Fn(&[f32]) -> f32,
        k: usize,
    ) -> usize;
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic select_best helper (DRY — shared by all impls)
// ─────────────────────────────────────────────────────────────────────────────

/// Free helper backing both [`BoMSampler::select_best`] impls. Iterates K
/// row-slices of width `dim`, tracks the argmax of `scorer`, returns the index.
/// Ties resolve to the lowest index (strict `>` comparison).
#[inline]
fn select_best_generic<S: Fn(&[f32]) -> f32>(
    hypotheses: &[f32],
    scorer: S,
    k: usize,
    dim: usize,
) -> usize {
    debug_assert!(
        hypotheses.len() >= k * dim,
        "select_best: hypotheses too short: need {} have {}",
        k * dim,
        hypotheses.len()
    );
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for k_idx in 0..k {
        let row = &hypotheses[k_idx * dim..(k_idx + 1) * dim];
        let score = scorer(row);
        // Strict `>` → ties keep the earlier (lower) index.
        if score > best_score {
            best_score = score;
            best_idx = k_idx;
        }
    }
    best_idx
}

/// Default scorer factory: max dot-product against a caller direction.
///
/// Returns a closure suitable for [`BoMSampler::select_best`]. Reuses
/// [`simd_dot_f32`]. Callers wanting minimax-over-threat supply their own `Fn`.
pub fn dot_product_scorer<'a>(direction: &'a [f32], dim: usize) -> impl Fn(&[f32]) -> f32 + 'a {
    move |hyp| simd_dot_f32(hyp, &direction[..dim], dim)
}

// ─────────────────────────────────────────────────────────────────────────────
// impl BoMSampler for AttractorKernel (Family A)
// ─────────────────────────────────────────────────────────────────────────────

use crate::micro_belief::attractor::AttractorKernel;

impl BoMSampler for AttractorKernel {
    /// Single-pass K-hypothesis sampling for Family A.
    ///
    /// Computes the base activation `act[i] = W_s[i]·s + W_x[i]·x + b[i]` once
    /// (mirroring [`AttractorKernel::step`] exactly — same chunked-4 outer
    /// loop, same `simd_dot_f32` reductions), then perturbs elementwise for
    /// each of K queries.
    ///
    /// # G1.3 (σ=0 degeneracy)
    ///
    /// With all-zero `queries`, the output is bit-identical to `step()` because
    /// `act[i] + 0.0 == act[i]` (f32 addition with +0.0 is exact) and the
    /// sigmoid/clamp chain is the same deterministic path.
    #[inline]
    fn sample_k_states(
        &self,
        s_prev: &[f32],
        x: &[f32],
        queries: &[f32],
        out: &mut [f32],
        cfg: &NoiseQueryConfig,
    ) {
        let dim = self.dim;
        let clamp = self.clamp;
        let k = cfg.k;

        debug_assert_eq!(s_prev.len(), dim, "s_prev/dim mismatch");
        debug_assert_eq!(x.len(), dim, "x/dim mismatch");
        debug_assert!(
            queries.len() >= k * dim,
            "queries too short: need {} have {}",
            k * dim,
            queries.len()
        );
        debug_assert!(
            out.len() >= k * dim,
            "out too short: need {} have {}",
            k * dim,
            out.len()
        );

        // ── Phase 1: base activation (computed ONCE, bit-identical to step()) ──
        //
        // Stack buffer — supports dim ≤ 1024 (same cap as AttractorKernel::step).
        // Uninit (matches cumprodsum.rs pattern) — act[..dim] is written by
        // the matvec below before the Phase 2 sigmoid reads it.
        let mut act_buf = uninit_stack::<1024>();
        // SAFETY: act[..dim] is fully written by the matvec below before any read.
        let act: &mut [f32] = unsafe { assume_init_slice(&mut act_buf, dim) };

        // Chunked-4 outer loop — mirrors AttractorKernel::step exactly so the
        // dot-product reductions and FMA scheduling produce the same f32
        // intermediates. Four independent dot-pairs hide FMA latency.
        let mut i = 0usize;
        while i + 4 <= dim {
            let ws_r0 = &self.ws[i * dim..(i + 1) * dim];
            let ws_r1 = &self.ws[(i + 1) * dim..(i + 2) * dim];
            let ws_r2 = &self.ws[(i + 2) * dim..(i + 3) * dim];
            let ws_r3 = &self.ws[(i + 3) * dim..(i + 4) * dim];
            let wx_r0 = &self.wx[i * dim..(i + 1) * dim];
            let wx_r1 = &self.wx[(i + 1) * dim..(i + 2) * dim];
            let wx_r2 = &self.wx[(i + 2) * dim..(i + 3) * dim];
            let wx_r3 = &self.wx[(i + 3) * dim..(i + 4) * dim];

            let dot_ws_0 = simd_dot_f32(s_prev, ws_r0, dim);
            let dot_ws_1 = simd_dot_f32(s_prev, ws_r1, dim);
            let dot_ws_2 = simd_dot_f32(s_prev, ws_r2, dim);
            let dot_ws_3 = simd_dot_f32(s_prev, ws_r3, dim);
            let dot_wx_0 = simd_dot_f32(x, wx_r0, dim);
            let dot_wx_1 = simd_dot_f32(x, wx_r1, dim);
            let dot_wx_2 = simd_dot_f32(x, wx_r2, dim);
            let dot_wx_3 = simd_dot_f32(x, wx_r3, dim);

            // Same addition order as step(): (dot_ws + dot_wx) + b.
            act[i] = dot_ws_0 + dot_wx_0 + self.b[i];
            act[i + 1] = dot_ws_1 + dot_wx_1 + self.b[i + 1];
            act[i + 2] = dot_ws_2 + dot_wx_2 + self.b[i + 2];
            act[i + 3] = dot_ws_3 + dot_wx_3 + self.b[i + 3];
            i += 4;
        }
        // Tail: remaining rows (dim mod 4).
        while i < dim {
            let ws_row = &self.ws[i * dim..(i + 1) * dim];
            let wx_row = &self.wx[i * dim..(i + 1) * dim];
            let dot_ws = simd_dot_f32(s_prev, ws_row, dim);
            let dot_wx = simd_dot_f32(x, wx_row, dim);
            act[i] = dot_ws + dot_wx + self.b[i];
            i += 1;
        }

        // ── Phase 2: K elementwise perturbations ──
        //
        // For each hypothesis k, add the noise query to the pre-sigmoid
        // activation, apply 2·σ(·)−1, clamp.
        //
        // Under `simd_sigmoid`: a single fused NEON/AVX2 pass per k replaces
        // the inner dim-loop. The call signature
        //   simd_sigmoid_tanh_clamp_inplace(out, act, queries, clamp)
        // is bit-identical to step()'s call when queries is all-zero (q[i]=0.0,
        // f32 addition is exact) — preserving Plan 281 G1.3.
        //
        // Under the default (scalar) path: per-element fast_sigmoid loop,
        // bit-for-bit unchanged from the pre-simd_sigmoid implementation.
        #[cfg(feature = "simd_sigmoid")]
        for k_idx in 0..k {
            let q_base = k_idx * dim;
            let out_base = k_idx * dim;
            simd_sigmoid_tanh_clamp_inplace(
                &mut out[out_base..out_base + dim],
                &act[..dim],
                &queries[q_base..q_base + dim],
                clamp,
            );
        }
        #[cfg(not(feature = "simd_sigmoid"))]
        for k_idx in 0..k {
            let q_base = k_idx * dim;
            let out_base = k_idx * dim;
            for j in 0..dim {
                let a = act[j] + queries[q_base + j];
                out[out_base + j] = (2.0 * fast_sigmoid(a) - 1.0).clamp(-clamp, clamp);
            }
        }
    }

    #[inline]
    fn select_best(&self, hypotheses: &[f32], scorer: impl Fn(&[f32]) -> f32, k: usize) -> usize {
        select_best_generic(hypotheses, scorer, k, self.dim)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// impl BoMSampler for LeakyIntegrator (Family C / evolve_hla family)
// ─────────────────────────────────────────────────────────────────────────────

use crate::micro_belief::leaky::LeakyIntegrator;

impl BoMSampler for LeakyIntegrator {
    /// Single-pass K-hypothesis sampling for Family C.
    ///
    /// Computes the shared normalization scalars (`total`, `scale`,
    /// `half_total`) once — mirroring [`crate::leaky_core::leaky_step`] — then
    /// perturbs the pre-clamp delta elementwise for each of K queries.
    ///
    /// # G1.3 (σ=0 degeneracy)
    ///
    /// With all-zero `queries`, the delta is `scale * (x[i] − half_total) + 0.0`
    /// = the deterministic delta, so `out` reproduces `step()`. When
    /// `total < 1e-8`, `step()` is a no-op (state unchanged); this impl copies
    /// `s_prev` into every row of `out` to match.
    #[inline]
    fn sample_k_states(
        &self,
        s_prev: &[f32],
        x: &[f32],
        queries: &[f32],
        out: &mut [f32],
        cfg: &NoiseQueryConfig,
    ) {
        let dim = self.dim;
        let k = cfg.k;
        let lr = self.lr;
        let max_delta = self.max_delta;

        debug_assert_eq!(s_prev.len(), dim, "s_prev/dim mismatch");
        debug_assert_eq!(x.len(), dim, "x/dim mismatch");
        debug_assert!(
            queries.len() >= k * dim,
            "queries too short: need {} have {}",
            k * dim,
            queries.len()
        );
        debug_assert!(
            out.len() >= k * dim,
            "out too short: need {} have {}",
            k * dim,
            out.len()
        );

        // Shared normalization mass — identical to LeakyIntegrator::step.
        let total: f32 = x[..dim].iter().copied().sum();

        // Degenerate-input guard (mirrors leaky_core::leaky_step early-return).
        // With zero queries this is a no-op for step(); we write s_prev into
        // every row so sample_k_states matches.
        if total < 1e-8 {
            for k_idx in 0..k {
                out[k_idx * dim..(k_idx + 1) * dim].copy_from_slice(&s_prev[..dim]);
            }
            return;
        }

        let t_min = total.min(1.0);
        let scale = lr * t_min / total;
        let half_total = 0.5 * total;

        // Hoist the k-invariant base delta into a stack buffer (≤ 1024 f32, matches
        // AttractorKernel::step's cap). The K elementwise perturbations then become
        // a single `add + clamp + clamp` per (k, j) — no per-k mul/sub.
        // Precomputing saves K×dim multiplications (256 at the G3 K=8/dim=32 budget).
        let mut base_delta_buf = uninit_stack::<1024>();
        // SAFETY: base_delta[..dim] is fully written by the loop below before
        // any read in the K-iteration phase.
        let base_delta: &mut [f32] = unsafe { assume_init_slice(&mut base_delta_buf, dim) };
        for j in 0..dim {
            base_delta[j] = scale * (x[j] - half_total);
        }

        // K elementwise perturbations of the pre-clamp delta.
        for k_idx in 0..k {
            let q_base = k_idx * dim;
            let out_base = k_idx * dim;
            for j in 0..dim {
                // Base delta (precomputed) + noise perturbation.
                let delta = base_delta[j] + queries[q_base + j];
                let clamped_delta = delta.clamp(-max_delta, max_delta);
                out[out_base + j] = (s_prev[j] + clamped_delta).clamp(-1.0, 1.0);
            }
        }
    }

    #[inline]
    fn select_best(&self, hypotheses: &[f32], scorer: impl Fn(&[f32]) -> f32, k: usize) -> usize {
        select_best_generic(hypotheses, scorer, k, self.dim)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::micro_belief::{AttractorKernel, LeakyIntegrator, MicroRecurrentBeliefState};

    // ── G1.1: determinism (fixed queries → bit-identical out) ──────────────

    #[test]
    fn bom_determinism_fixed_queries() {
        let kernel = AttractorKernel::from_seed(42, 32);
        let dim = 32;
        let k = 8;
        let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(0.3);

        let mut s_prev = vec![0.0f32; dim];
        let input = vec![0.5f32; dim];
        // Step a few times to get a non-trivial state.
        for _ in 0..5 {
            kernel.step(&mut s_prev, &input);
        }

        // Fixed queries (seed=99).
        let mut rng = fastrand::Rng::with_seed(99);
        let queries: Vec<f32> = (0..k * dim)
            .map(|_| (rng.f32() * 2.0 - 1.0) * 0.3)
            .collect();

        let mut out_a = vec![0.0f32; k * dim];
        let mut out_b = vec![0.0f32; k * dim];
        kernel.sample_k_states(&s_prev, &input, &queries, &mut out_a, &cfg);
        kernel.sample_k_states(&s_prev, &input, &queries, &mut out_b, &cfg);

        assert_eq!(
            out_a, out_b,
            "G1.1: same queries must produce bit-identical out"
        );
    }

    // ── G1.2: distinct hypotheses (cosine sim < 0.99) ──────────────────────

    #[test]
    fn bom_distinct_hypotheses() {
        // sigma=0.3 for [-1,1] HLA space (paper's 0.02 is for DINOv3 features
        // and produces near-identical hypotheses — R3 calibration signal).
        let kernel = AttractorKernel::from_seed(42, 32);
        let dim = 32;
        let k = 8;
        let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(0.3);

        let mut s_prev = vec![0.0f32; dim];
        let input = vec![0.5f32; dim];
        for _ in 0..5 {
            kernel.step(&mut s_prev, &input);
        }

        let mut rng = fastrand::Rng::with_seed(42);
        let queries: Vec<f32> = (0..k * dim)
            .map(|_| (rng.f32() * 2.0 - 1.0) * 0.3)
            .collect();
        let mut out = vec![0.0f32; k * dim];
        kernel.sample_k_states(&s_prev, &input, &queries, &mut out, &cfg);

        for a in 0..k {
            for b in (a + 1)..k {
                let row_a = &out[a * dim..(a + 1) * dim];
                let row_b = &out[b * dim..(b + 1) * dim];
                let dot = simd_dot_f32(row_a, row_b, dim);
                let norm_a = simd_dot_f32(row_a, row_a, dim).sqrt();
                let norm_b = simd_dot_f32(row_b, row_b, dim).sqrt();
                let denom = norm_a * norm_b;
                assert!(denom > 1e-12, "zero-norm hypothesis");
                let cos_sim = dot / denom;
                assert!(
                    cos_sim < 0.99,
                    "G1.2: hypotheses {a},{b} near-identical: cos_sim={cos_sim:.6}"
                );
            }
        }
    }

    // ── G1.3: σ=0 degeneracy (zero queries → step() output) ────────────────

    #[test]
    fn bom_sigma_zero_matches_step_attractor() {
        let kernel = AttractorKernel::from_seed(42, 32);
        let dim = 32;
        let k = 8;
        let cfg = NoiseQueryConfig::default().with_k(k);

        let mut s_prev = vec![0.0f32; dim];
        let input = vec![0.5f32; dim];
        for _ in 0..3 {
            kernel.step(&mut s_prev, &input);
        }

        // Deterministic step() on a copy.
        let mut stepped = s_prev.clone();
        kernel.step(&mut stepped, &input);

        // BoM with all-zero queries.
        let queries = vec![0.0f32; k * dim];
        let mut out = vec![0.0f32; k * dim];
        kernel.sample_k_states(&s_prev, &input, &queries, &mut out, &cfg);

        for k_idx in 0..k {
            let row = &out[k_idx * dim..(k_idx + 1) * dim];
            assert_eq!(
                row,
                &stepped[..],
                "G1.3: row {k_idx} must match step() output with zero queries"
            );
        }
    }

    #[test]
    fn bom_sigma_zero_matches_step_leaky() {
        let kernel = LeakyIntegrator::hla_default(32);
        let dim = 32;
        let k = 8;
        let cfg = NoiseQueryConfig::default().with_k(k);

        let mut s_prev = vec![0.0f32; dim];
        let input = vec![0.5f32; dim];
        for _ in 0..3 {
            kernel.step(&mut s_prev, &input);
        }

        let mut stepped = s_prev.clone();
        kernel.step(&mut stepped, &input);

        let queries = vec![0.0f32; k * dim];
        let mut out = vec![0.0f32; k * dim];
        kernel.sample_k_states(&s_prev, &input, &queries, &mut out, &cfg);

        for k_idx in 0..k {
            let row = &out[k_idx * dim..(k_idx + 1) * dim];
            assert_eq!(
                row,
                &stepped[..],
                "G1.3 (leaky): row {k_idx} must match step() output with zero queries"
            );
        }
    }

    #[test]
    fn bom_sigma_zero_matches_step_leaky_zero_total() {
        // Edge case: zero-total input → step() is a no-op; sample_k_states
        // must copy s_prev into every row.
        let kernel = LeakyIntegrator::hla_default(8);
        let dim = 8;
        let k = 4;
        let cfg = NoiseQueryConfig::default().with_k(k);

        let s_prev: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 0.4).collect();
        let input = vec![0.0f32; dim]; // total = 0 → no-op

        let queries = vec![0.0f32; k * dim];
        let mut out = vec![0.0f32; k * dim];
        kernel.sample_k_states(&s_prev, &input, &queries, &mut out, &cfg);

        for k_idx in 0..k {
            let row = &out[k_idx * dim..(k_idx + 1) * dim];
            assert_eq!(row, &s_prev[..], "zero-total row {k_idx} must equal s_prev");
        }
    }

    // ── Bounded state over 100 ticks ───────────────────────────────────────

    #[test]
    fn bom_attractor_state_stays_bounded() {
        let kernel = AttractorKernel::from_seed(42, 32);
        let dim = 32;
        let k = 8;
        let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(0.3);

        let mut s_prev = vec![0.0f32; dim];
        let input = vec![0.5f32; dim];
        let mut rng = fastrand::Rng::with_seed(7);
        let mut queries = vec![0.0f32; k * dim];
        let mut out = vec![0.0f32; k * dim];

        for _ in 0..100 {
            for q in queries.iter_mut() {
                *q = (rng.f32() * 2.0 - 1.0) * 0.3;
            }
            kernel.sample_k_states(&s_prev, &input, &queries, &mut out, &cfg);
            for &v in &out {
                assert!(
                    v > -1.0001 && v < 1.0001,
                    "attractor state out of (-1,1): {v}"
                );
            }
            // Advance using row 0 as the next state.
            s_prev.copy_from_slice(&out[..dim]);
        }
    }

    #[test]
    fn bom_leaky_state_stays_bounded() {
        let kernel = LeakyIntegrator::hla_default(32);
        let dim = 32;
        let k = 8;
        let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(0.3);

        let mut s_prev = vec![0.0f32; dim];
        let input = vec![0.5f32; dim];
        let mut rng = fastrand::Rng::with_seed(7);
        let mut queries = vec![0.0f32; k * dim];
        let mut out = vec![0.0f32; k * dim];

        for _ in 0..100 {
            for q in queries.iter_mut() {
                *q = (rng.f32() * 2.0 - 1.0) * 0.3;
            }
            kernel.sample_k_states(&s_prev, &input, &queries, &mut out, &cfg);
            for &v in &out {
                assert!((-1.0..=1.0).contains(&v), "leaky state out of [-1,1]: {v}");
            }
            s_prev.copy_from_slice(&out[..dim]);
        }
    }

    // ── select_best ────────────────────────────────────────────────────────

    #[test]
    fn select_best_picks_max() {
        let kernel = AttractorKernel::from_seed(42, 4);
        let dim = 4;
        // 3 hypotheses with known scores under dot with [1,0,0,0]:
        //   row 0: dot = 0.1  → score 0.1
        //   row 1: dot = 0.9  → score 0.9  ← max
        //   row 2: dot = 0.5  → score 0.5
        let hypotheses: Vec<f32> = vec![0.1, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0];
        let direction = vec![1.0f32, 0.0, 0.0, 0.0];
        let scorer = dot_product_scorer(&direction, dim);
        let best = kernel.select_best(&hypotheses, scorer, 3);
        assert_eq!(best, 1, "select_best must pick the max-score hypothesis");
    }

    #[test]
    fn select_best_ties_resolve_lowest() {
        let kernel = AttractorKernel::from_seed(42, 4);
        let dim = 4;
        // Two rows with equal score (dot = 0.5), third is lower.
        let hypotheses: Vec<f32> = vec![
            0.5, 0.0, 0.0, 0.0, // score 0.5
            0.0, 0.5, 0.0, 0.0, // score 0.0
            0.5, 0.0, 0.0, 0.0, // score 0.5  (tie with row 0)
        ];
        let direction = vec![1.0f32, 0.0, 0.0, 0.0];
        let scorer = dot_product_scorer(&direction, dim);
        let best = kernel.select_best(&hypotheses, scorer, 3);
        assert_eq!(best, 0, "ties must resolve to the lowest index");
    }

    #[test]
    fn select_best_leaky_works() {
        // Verify select_best also works through the LeakyIntegrator impl.
        let kernel = LeakyIntegrator::hla_default(4);
        let dim = 4;
        let hypotheses: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // max norm
            0.5, 0.5, 0.5, 0.5,
        ];
        let direction = vec![1.0f32; dim];
        let scorer = dot_product_scorer(&direction, dim);
        let best = kernel.select_best(&hypotheses, scorer, 3);
        assert_eq!(best, 1);
    }

    // ── NoiseQueryConfig commitment ────────────────────────────────────────

    #[test]
    fn noise_config_commit_roundtrips() {
        let cfg = NoiseQueryConfig::default().with_sigma(0.3).with_k(16);
        let commit = cfg.commit();
        assert!(cfg.verify(&commit), "freshly-committed config must verify");

        // Mutating sigma must break verification.
        let mut tampered = cfg;
        tampered.sigma = 0.5;
        assert!(!tampered.verify(&commit), "tampered sigma must fail verify");

        // Mutating k must break verification.
        let mut tampered_k = cfg;
        tampered_k.k = 4;
        assert!(!tampered_k.verify(&commit), "tampered k must fail verify");
    }

    #[test]
    fn noise_config_seed_strategy_affects_commit() {
        let base = NoiseQueryConfig::default();
        let per_class = NoiseQueryConfig::default().with_seed_strategy(SeedStrategy::PerClass);
        assert_ne!(
            base.commit(),
            per_class.commit(),
            "different seed_strategy must produce different commitment"
        );
    }

    #[test]
    fn noise_config_default_is_k8_sigma_0p1() {
        let cfg = NoiseQueryConfig::default();
        assert_eq!(cfg.k, 8, "default k=8");
        assert!((cfg.sigma - 0.1).abs() < 1e-6, "default sigma=0.1");
        assert_eq!(cfg.seed_strategy, SeedStrategy::PerNpc);
    }

    #[test]
    fn noise_config_builder_chains() {
        let cfg = NoiseQueryConfig::default()
            .with_sigma(0.5)
            .with_k(16)
            .with_seed_strategy(SeedStrategy::PerClass);
        assert!((cfg.sigma - 0.5).abs() < 1e-6);
        assert_eq!(cfg.k, 16);
        assert_eq!(cfg.seed_strategy, SeedStrategy::PerClass);
    }

    // ── T2.2: coherence over 1000 ticks ───────────────────────────────────

    #[test]
    fn bom_coherence_1000_ticks_bounded_attractor() {
        // Catches Family A divergence (Research 242 R1). The attractor state
        // is 2·σ(·)−1 ∈ (−1,1) by construction, but 1000 ticks with random
        // noise is the stress test for any numerical drift.
        let kernel = AttractorKernel::from_seed(42, 32);
        let dim = 32;
        let k = 8;
        let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(0.3);

        let mut s_prev = vec![0.0f32; dim];
        let input = vec![0.5f32; dim];
        let mut queries = vec![0.0f32; k * dim];
        let mut out = vec![0.0f32; k * dim];
        let mut rng = fastrand::Rng::with_seed(12345);

        for _tick in 0..1000 {
            for q in queries.iter_mut() {
                *q = (rng.f32() * 2.0 - 1.0) * 0.3;
            }
            kernel.sample_k_states(&s_prev, &input, &queries, &mut out, &cfg);
            for &v in &out {
                assert!(
                    v > -1.0001 && v < 1.0001,
                    "attractor diverged after 1000 ticks: {v}"
                );
            }
            // Pick row 0 as the next s_prev (the "selected" hypothesis).
            s_prev.copy_from_slice(&out[..dim]);
        }
    }

    #[test]
    fn bom_coherence_1000_ticks_bounded_leaky() {
        // Family C is always stable by construction (clamp to [-1,1]); this
        // test documents that property holds under BoM perturbation too.
        let kernel = LeakyIntegrator::hla_default(32);
        let dim = 32;
        let k = 8;
        let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(0.3);

        let mut s_prev = vec![0.0f32; dim];
        let input = vec![0.5f32; dim];
        let mut queries = vec![0.0f32; k * dim];
        let mut out = vec![0.0f32; k * dim];
        let mut rng = fastrand::Rng::with_seed(54321);

        for _tick in 0..1000 {
            for q in queries.iter_mut() {
                *q = (rng.f32() * 2.0 - 1.0) * 0.3;
            }
            kernel.sample_k_states(&s_prev, &input, &queries, &mut out, &cfg);
            for &v in &out {
                assert!((-1.0..=1.0).contains(&v), "leaky diverged after 1000 ticks: {v}");
            }
            s_prev.copy_from_slice(&out[..dim]);
        }
    }

    // ── LeakyIntegrator BoM sanity (non-zero queries produce K distinct rows) ──

    #[test]
    fn bom_leaky_distinct_hypotheses() {
        let kernel = LeakyIntegrator::hla_default(8);
        let dim = 8;
        let k = 8;
        let cfg = NoiseQueryConfig::default().with_k(k).with_sigma(0.3);

        let s_prev = vec![0.0f32; dim];
        let input = vec![0.5f32; dim];

        let mut rng = fastrand::Rng::with_seed(42);
        let queries: Vec<f32> = (0..k * dim)
            .map(|_| (rng.f32() * 2.0 - 1.0) * 0.3)
            .collect();
        let mut out = vec![0.0f32; k * dim];
        kernel.sample_k_states(&s_prev, &input, &queries, &mut out, &cfg);

        // At least some pairs should differ.
        let mut any_distinct = false;
        for a in 0..k {
            for b in (a + 1)..k {
                if out[a * dim..(a + 1) * dim] != out[b * dim..(b + 1) * dim] {
                    any_distinct = true;
                    break;
                }
            }
        }
        assert!(
            any_distinct,
            "leaky BoM must produce at least some distinct rows"
        );
    }
}
