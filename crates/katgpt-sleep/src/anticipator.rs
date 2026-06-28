//! Sleep-time anticipator (Plan 334 Phase 1 T1.4).
//!
//! [`SleepTimeAnticipator::anticipate`] is the **sleep-time operator**
//! `S(c) → c'` from the paper. It orchestrates predictability scoring +
//! per-direction budget allocation + c' artifact emission.
//!
//! The concrete per-direction compute (the z_i output) is provided by the
//! consumer via the [`SleepTimeComputeOp`] trait — this module ships only
//! the orchestration. Examples of `SleepTimeComputeOp` impls:
//!
//! - **`IdentityFunctorOp`** (this file): `z_i = c + dir_i`. The simplest
//!   modelless op — what would the NPC "feel" if asked query i? The
//!   direction-projected HLA. Used as the synthetic-test default and the
//!   no-op baseline.
//! - **`latent_functor` extraction** (riir-ai Plan 341): `z_i = extract(c, dir_i)`
//!   using the open `latent_functor` primitive from Plan 303.
//! - **`karc_forecast`** (riir-ai Plan 341): `z_i = karc_forecaster.forecast(c + dir_i)`.
//!
//! # Zero-allocation discipline
//!
//! `anticipate()` necessarily allocates the output `AnticipatedQuerySet`
//! (the c' artifact). The wake-time hot path — `consume()` — is the one
//! that must be zero-alloc; see `consume.rs`.

use crate::predictability::PredictabilityScorer;
use crate::types::{AnticipatedQueryDir, AnticipatedQuerySet, AnticipatedSlot};

/// One sleep-time compute call. Produces `z_i` for direction `i`.
///
/// Implementations MUST be:
/// - **Modelless** — no training, no backprop. Closed-form algebra only.
/// - **Deterministic** — given `(c, dir, budget)`, always returns the same
///   `z_i`. This is what makes the BLAKE3 commitment meaningful.
/// - **Side-effect-free** — except via the `scratch` buffer, which the caller
///   owns and reuses across calls (zero-alloc hot path per AGENTS.md).
pub trait SleepTimeComputeOp<const D: usize> {
    /// Compute `z_i` for direction `i`.
    ///
    /// `budget` is a hint: implementations MAY use it to decide how much
    /// compute to spend (e.g. KARC ridge order, LLM token budget). The
    /// synthetic `IdentityFunctorOp` ignores it.
    fn sleep_compute(
        &self,
        c: &[f32; D],
        dir: &AnticipatedQueryDir<D>,
        budget: u32,
        scratch: &mut SleepTimeScratch<D>,
    ) -> [f32; D];
}

/// Reusable scratch buffer for sleep-time compute.
///
/// Passed in by the caller to keep the per-direction compute path
/// zero-allocation (per AGENTS.md hot-loop rules). Two `[f32; D]` slots so
/// ops that need a temporary (ridge fit, FFT, etc.) don't allocate.
#[derive(Clone, Debug)]
pub struct SleepTimeScratch<const D: usize> {
    /// Primary scratch slot.
    pub buf: [f32; D],
    /// Auxiliary scratch slot (for ops that need a second buffer).
    pub aux: [f32; D],
}

impl<const D: usize> SleepTimeScratch<D> {
    /// Zero-initialised scratch. Caller owns one per thread/NPC and reuses
    /// across all `anticipate()` calls.
    #[inline]
    pub fn new() -> Self {
        Self {
            buf: [0.0; D],
            aux: [0.0; D],
        }
    }
}

impl<const D: usize> Default for SleepTimeScratch<D> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Orchestrates sleep-time compute across K directions, emitting a c'
/// artifact.
///
/// Generic over:
/// - `D` — latent dim.
/// - `K` — catalog size (number of anticipated-query directions).
/// - `Op` — the per-direction compute op (consumer-provided).
/// - `Scorer` — the predictability scorer (consumer-provided).
#[derive(Clone, Debug)]
pub struct SleepTimeAnticipator<const D: usize, const K: usize, Op, Scorer> {
    /// The per-direction compute op. Called once per direction in `anticipate`.
    pub op: Op,
    /// The predictability scorer. Called once per direction in `anticipate`.
    pub scorer: Scorer,
    /// Per-direction sleep-time compute budget (tokens, or any opaque hint).
    pub budgets: [u32; K],
    /// Gate threshold τ. Used by the wake-time `consume()` (passed through
    /// here for symmetry; the anticipator itself does not use it).
    pub tau: f32,
    /// Gate sharpness β. Same caveat as `tau`.
    pub beta: f32,
}

impl<const D: usize, const K: usize, Op, Scorer> SleepTimeAnticipator<D, K, Op, Scorer>
where
    Op: SleepTimeComputeOp<D>,
    Scorer: PredictabilityScorer<D>,
{
    /// Run sleep-time compute. Produces c' for the given `c` and direction set.
    ///
    /// # Allocation
    ///
    /// This method allocates the output `AnticipatedQuerySet` (the c'
    /// artifact). The per-direction compute itself is zero-alloc (caller
    /// provides `scratch`). The wake-time hot path (`consume()`) is the one
    /// that must be zero-alloc; see `consume.rs`.
    pub fn anticipate(
        &self,
        c: &[f32; D],
        dirs: &[AnticipatedQueryDir<D>; K],
        scratch: &mut SleepTimeScratch<D>,
    ) -> AnticipatedQuerySet<D, K> {
        // Build the slots one at a time. `AnticipatedSlot` is not `Copy`
        // (contains `AnticipatedQueryDir` which is not Copy), so we use
        // `from_fn` to construct the initial placeholder, then overwrite.
        // The placeholder `AnticipatedQueryDir::new([0.0; D])` is thrown away.
        let mut slots: [AnticipatedSlot<D>; K] = std::array::from_fn(|_| AnticipatedSlot {
            dir: AnticipatedQueryDir::new([0.0; D]),
            precomputed: [0.0; D],
            predictability: 0.0,
        });
        for i in 0..K {
            let z = self.op.sleep_compute(c, &dirs[i], self.budgets[i], scratch);
            let p = self.scorer.predictability(c, &dirs[i]);
            slots[i] = AnticipatedSlot {
                dir: dirs[i].clone(),
                precomputed: z,
                predictability: p,
            };
        }
        let blake3 = AnticipatedQuerySet::commit_slots(&slots);
        AnticipatedQuerySet {
            slots,
            blake3,
            version: 0,
        }
    }
}

// ── IdentityFunctorOp — the synthetic-test default ──────────────────────────

/// `z_i = c + dir_i` — the simplest modelless sleep-time op.
///
/// "What would the NPC feel if asked query i?" Answer: the current state
/// nudged in the query's direction. Used as the synthetic-test default and
/// the no-op baseline (Phase 2 GOAT gate G1).
///
/// Real consumers (riir-ai Plan 341) swap this for `latent_functor` extraction
/// or `karc_forecast` — both still modelless.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityFunctorOp;

impl<const D: usize> SleepTimeComputeOp<D> for IdentityFunctorOp {
    #[inline]
    fn sleep_compute(
        &self,
        c: &[f32; D],
        dir: &AnticipatedQueryDir<D>,
        _budget: u32,
        _scratch: &mut SleepTimeScratch<D>,
    ) -> [f32; D] {
        // z_i = c + dir_i — closed-form, deterministic, zero-alloc.
        let mut z = [0.0f32; D];
        for j in 0..D {
            z[j] = c[j] + dir.direction[j];
        }
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictability::DotPredictabilityScorer;

    #[test]
    fn anticipate_emits_k_slots() {
        const D: usize = 2;
        const K: usize = 3;
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
            AnticipatedQueryDir::new([1.0, 1.0]),
        ];
        let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
            op: IdentityFunctorOp,
            scorer: DotPredictabilityScorer::default(),
            budgets: [100, 100, 100],
            tau: 0.5,
            beta: 4.0,
        };
        let c = [0.5, 0.5];
        let mut scratch = SleepTimeScratch::new();
        let artifact = anticipator.anticipate(&c, &dirs, &mut scratch);

        assert_eq!(artifact.slots.len(), K);
        // Each slot's direction matches the input.
        for i in 0..K {
            assert_eq!(artifact.slots[i].dir.blake3, dirs[i].blake3);
        }
        // IdentityFunctorOp: z_i = c + dir_i.
        assert!((artifact.slots[0].precomputed[0] - 1.5).abs() < 1e-6);
        assert!((artifact.slots[0].precomputed[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn anticipate_commitment_is_stable_across_calls_with_same_inputs() {
        const D: usize = 2;
        const K: usize = 2;
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
            op: IdentityFunctorOp,
            scorer: DotPredictabilityScorer::default(),
            budgets: [100, 100],
            tau: 0.5,
            beta: 4.0,
        };
        let c = [0.3, 0.7];
        let mut scratch = SleepTimeScratch::new();
        let a1 = anticipator.anticipate(&c, &dirs, &mut scratch);
        let a2 = anticipator.anticipate(&c, &dirs, &mut scratch);
        assert_eq!(
            a1.blake3, a2.blake3,
            "same inputs → same BLAKE3 commitment"
        );
        assert!(a1.verify_commitment(), "commitment verifies");
    }

    #[test]
    fn anticipate_commitment_changes_when_context_changes() {
        const D: usize = 2;
        const K: usize = 2;
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
            op: IdentityFunctorOp,
            scorer: DotPredictabilityScorer::default(),
            budgets: [100, 100],
            tau: 0.5,
            beta: 4.0,
        };
        let mut scratch = SleepTimeScratch::new();
        let a1 = anticipator.anticipate(&[0.3, 0.7], &dirs, &mut scratch);
        let a2 = anticipator.anticipate(&[0.3, 0.8], &dirs, &mut scratch);
        assert_ne!(
            a1.blake3, a2.blake3,
            "different c → different commitment (z_i changed)"
        );
    }

    #[test]
    fn anticipate_predictability_in_unit_interval() {
        const D: usize = 4;
        const K: usize = 2;
        let dirs = [
            AnticipatedQueryDir::new([1.0; D]),
            AnticipatedQueryDir::new([-1.0; D]),
        ];
        let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
            op: IdentityFunctorOp,
            scorer: DotPredictabilityScorer::default(),
            budgets: [100, 100],
            tau: 0.5,
            beta: 4.0,
        };
        let c = [2.0; D];
        let mut scratch = SleepTimeScratch::new();
        let artifact = anticipator.anticipate(&c, &dirs, &mut scratch);
        for slot in &artifact.slots {
            assert!(
                (0.0..=1.0).contains(&slot.predictability),
                "predictability {} out of [0,1]",
                slot.predictability
            );
        }
    }

    #[test]
    fn identity_op_ignores_budget() {
        // IdentityFunctorOp is budget-agnostic; verify the contract by
        // checking that two calls with different budgets produce identical z.
        let op = IdentityFunctorOp;
        let dir = AnticipatedQueryDir::new([1.0, 1.0]);
        let c = [0.0, 0.0];
        let mut scratch = SleepTimeScratch::new();
        let z1 = op.sleep_compute(&c, &dir, 1, &mut scratch);
        let z2 = op.sleep_compute(&c, &dir, 10_000, &mut scratch);
        assert_eq!(z1, z2, "IdentityFunctorOp must ignore budget");
    }
}
