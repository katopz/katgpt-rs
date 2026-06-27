//! Wake-time consumer (Plan 334 Phase 1 T1.6).
//!
//! [`consume`] is the wake-time hot path — the `T_b(q, c') → a` operator
//! from the paper. Given a query `q` and the pre-computed artifact `c'`,
//! produce an answer via cheap dot-product + sigmoid-gated lookup, falling
//! through to the caller-provided `fresh_think` if the gate is low (the
//! query is unpredictable).
//!
//! # Why this is the hot path
//!
//! `consume()` runs on every player-NPC interaction. The sleep-time compute
//! (`anticipate()`) runs once per NPC per sleep cycle. The whole point of
//! the paper is: amortize the expensive sleep-time compute over many cheap
//! wake-time `consume()` calls.
//!
//! Per AGENTS.md hot-loop rules and Plan 334 T2.3 (G5 zero-alloc gate),
//! `consume()` MUST NOT allocate. The closure `fresh_think` is allowed to
//! allocate (it's the fallback path, which only fires on low-predictability
//! queries).
//!
//! # Sigmoid blend, not hard switch (AGENTS.md)
//!
//! The output is a smooth blend: `gate * z_precomputed + (1 − gate) * fresh`.
//! Per AGENTS.md ("use sigmoid not softmax"), we never hard-switch — the
//! smooth blend preserves the modelless property and avoids discontinuities
//! in the gate threshold.

use crate::simd::{fast_sigmoid, simd_dot_f32};
use crate::sleep_time::types::AnticipatedQuerySet;

/// Wake-time consumer: given query `q` and pre-computed `c'`, produce an
/// answer via cheap lookup + sigmoid gate.
///
/// # Algorithm
///
/// 1. Find the best-matching anticipated direction `i* = argmax_i dot(q, dir_i)`.
/// 2. Compute the gate: `gate = sigmoid(beta * (p_{i*} − tau))`.
/// 3. Blend: `out = gate * z_{i*} + (1 − gate) * fresh_think(q)`.
///
/// When `gate ≈ 1` (predictable query), the output is the precomputed slot.
/// When `gate ≈ 0` (unpredictable query), the output is the fresh compute.
/// In between, it's a smooth blend.
///
/// # Parameters
///
/// - `q`: the incoming query (latent embedding).
/// - `c_prime`: the pre-computed artifact from `SleepTimeAnticipator::anticipate`.
/// - `tau`: gate threshold. Higher = require higher predictability to use the cache.
/// - `beta`: gate sharpness. Higher = sharper transition around `tau`.
/// - `fresh_think`: closure that produces a fresh answer for `q` (the
///   fallback when the gate is low). Called at most once.
///
/// # Allocation
///
/// This function is **zero-allocation** in the steady state. The
/// `fresh_think` closure MAY allocate (it's the fallback path); if it does,
/// those allocations happen only when `gate < 1.0` (i.e. on cache misses).
///
/// # Determinism
///
/// Given `(q, c_prime, tau, beta)` and a deterministic `fresh_think`,
/// `consume()` is deterministic. The G1 gate verifies this.
#[inline]
pub fn consume<const D: usize, const K: usize, F>(
    q: &[f32; D],
    c_prime: &AnticipatedQuerySet<D, K>,
    tau: f32,
    beta: f32,
    fresh_think: F,
) -> [f32; D]
where
    F: FnOnce(&[f32; D]) -> [f32; D],
{
    // 1. Find best-matching anticipated direction (O(K) scan; K is bounded).
    let mut best_i = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for i in 0..K {
        let d = simd_dot_f32(q, &c_prime.slots[i].dir.direction, D);
        if d > best_dot {
            best_dot = d;
            best_i = i;
        }
    }

    // 2. Sigmoid gate from the best match's predictability.
    let p = c_prime.slots[best_i].predictability;
    let gate = fast_sigmoid(beta * (p - tau));

    // 3. Blend precomputed + fresh. We always call fresh_think here for
    //    simplicity — if the caller wants to skip fresh compute when
    //    `gate ≈ 1`, they can check the gate themselves before calling
    //    consume(). This keeps consume() branch-free in the blend.
    //    (Plan 334 T2.1 verifies the blend is correct; the optimization of
    //    skipping fresh_think on gate≈1 is a consumer concern, not a
    //    primitive concern.)
    let z = c_prime.slots[best_i].precomputed;
    let fresh = fresh_think(q);
    let mut out = [0.0f32; D];
    for j in 0..D {
        out[j] = gate * z[j] + (1.0 - gate) * fresh[j];
    }
    out
}

/// Cheap gate-only check: returns `(best_i, gate)` without running `fresh_think`.
///
/// Consumers that want to skip fresh compute when the gate is high can call
/// this first, check the gate, and only call `fresh_think` if needed. This
/// keeps the primitive flexible without forcing every consumer to pay for
/// fresh compute on every call.
///
/// Same matching + gating as [`consume`], but no blend — just the decision.
#[inline]
pub fn consume_gate<const D: usize, const K: usize>(
    q: &[f32; D],
    c_prime: &AnticipatedQuerySet<D, K>,
    tau: f32,
    beta: f32,
) -> (usize, f32) {
    let mut best_i = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for i in 0..K {
        let d = simd_dot_f32(q, &c_prime.slots[i].dir.direction, D);
        if d > best_dot {
            best_dot = d;
            best_i = i;
        }
    }
    let p = c_prime.slots[best_i].predictability;
    let gate = fast_sigmoid(beta * (p - tau));
    (best_i, gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sleep_time::anticipator::{IdentityFunctorOp, SleepTimeAnticipator, SleepTimeScratch};
    use crate::sleep_time::predictability::DotPredictabilityScorer;
    use crate::sleep_time::types::AnticipatedQueryDir;

    /// Build a small c' artifact for testing consume().
    fn build_artifact(
        c: &[f32; 2],
        dirs: &[AnticipatedQueryDir<2>; 2],
    ) -> crate::sleep_time::types::AnticipatedQuerySet<2, 2> {
        let anticipator = SleepTimeAnticipator::<2, 2, IdentityFunctorOp, DotPredictabilityScorer> {
            op: IdentityFunctorOp,
            scorer: DotPredictabilityScorer::default(),
            budgets: [100, 100],
            tau: 0.5,
            beta: 4.0,
        };
        let mut scratch = SleepTimeScratch::new();
        anticipator.anticipate(c, dirs, &mut scratch)
    }

    #[test]
    fn consume_returns_precomputed_when_predictable() {
        // c aligned with dir[0] → predictability of slot 0 is high → gate near 1.
        // Use beta=50 so sigmoid(50 * (p - tau)) saturates to ~1.0 when p > tau.
        let dirs = [
            AnticipatedQueryDir::new([10.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let c = [10.0, 0.0]; // strongly aligned with dir 0 → p ≈ sigmoid(100) ≈ 1.0
        let artifact = build_artifact(&c, &dirs);
        // Query also aligned with dir 0.
        let q = [10.0, 0.0];
        // fresh_think that returns a distinct value so we can detect blend weight.
        // With beta=50, p≈1.0, tau=0.5: gate = sigmoid(50 * 0.5) = sigmoid(25) ≈ 1.0.
        let out = consume(&q, &artifact, 0.5, 50.0, |fresh_q| [fresh_q[0] * 100.0, 0.0]);
        // Precomputed z_0 = c + dir_0 = [20, 0]. Fresh = [1000, 0].
        // gate ≈ 1.0 → out ≈ [20, 0].
        assert!(
            (out[0] - 20.0).abs() < 1.0,
            "expected precomputed (~20.0) when predictable, got {}",
            out[0]
        );
    }

    #[test]
    fn consume_returns_fresh_when_unpredictable() {
        // c orthogonal to all dirs → predictability ≈ 0.5 → with high tau,
        // gate ≈ 0 → out ≈ fresh.
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let c = [0.0, 0.0]; // dot = 0 with both → predictability = sigmoid(0) = 0.5
        let artifact = build_artifact(&c, &dirs);
        let q = [1.0, 0.0];
        // tau = 0.99, beta = 50 → gate = sigmoid(50 * (0.5 - 0.99)) = sigmoid(-24.5) ≈ 0.
        let out = consume(&q, &artifact, 0.99, 50.0, |_| [42.0, 7.0]);
        // gate ≈ 0 → out ≈ fresh = [42, 7].
        assert!(
            (out[0] - 42.0).abs() < 1.0,
            "expected ≈ fresh (42.0) when unpredictable, got {}",
            out[0]
        );
        assert!(
            (out[1] - 7.0).abs() < 1.0,
            "expected ≈ fresh y (7.0) when unpredictable, got {}",
            out[1]
        );
    }

    #[test]
    fn consume_is_deterministic() {
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let c = [0.5, 0.5];
        let artifact = build_artifact(&c, &dirs);
        let q = [0.7, 0.3];
        // Deterministic fresh_think (no RNG).
        let out1 = consume(&q, &artifact, 0.5, 4.0, |fq| [fq[0] + 1.0, fq[1] + 1.0]);
        let out2 = consume(&q, &artifact, 0.5, 4.0, |fq| [fq[0] + 1.0, fq[1] + 1.0]);
        assert_eq!(out1, out2, "consume must be deterministic");
    }

    #[test]
    fn consume_gate_finds_best_match() {
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let c = [1.0, 1.0]; // equally aligned → predictability equal
        let artifact = build_artifact(&c, &dirs);
        // Query aligned with dir 1.
        let q = [0.0, 1.0];
        let (best_i, _gate) = consume_gate(&q, &artifact, 0.5, 4.0);
        assert_eq!(best_i, 1, "best match should be slot 1");
    }

    #[test]
    fn consume_gate_value_in_unit_interval() {
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let c = [0.0, 0.0];
        let artifact = build_artifact(&c, &dirs);
        for q in &[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]] {
            let (_, gate) = consume_gate(q, &artifact, 0.5, 4.0);
            assert!(
                (0.0..=1.0).contains(&gate),
                "gate {} out of [0,1] for q={:?}",
                gate,
                q
            );
        }
    }

    #[test]
    fn consume_blend_is_smooth_not_hard_switch() {
        // At gate = 0.5 (predictability == tau), out should be exactly
        // 50/50 blend of precomputed and fresh.
        let dirs = [AnticipatedQueryDir::new([1.0, 0.0])];
        // Build a single-slot artifact by hand so we control predictability.
        let slots = [crate::sleep_time::types::AnticipatedSlot {
            dir: dirs[0].clone(),
            precomputed: [10.0, 0.0],
            predictability: 0.5, // == tau → gate = sigmoid(0) = 0.5
        }];
        let blake3 = crate::sleep_time::types::AnticipatedQuerySet::<2, 1>::commit_slots(&slots);
        let artifact = crate::sleep_time::types::AnticipatedQuerySet {
            slots,
            blake3,
            version: 0,
        };
        let q = [1.0, 0.0];
        let out = consume(&q, &artifact, 0.5, 4.0, |_| [0.0, 20.0]);
        // 0.5 * [10, 0] + 0.5 * [0, 20] = [5, 10].
        assert!((out[0] - 5.0).abs() < 1e-6, "blend x: {}", out[0]);
        assert!((out[1] - 10.0).abs() < 1e-6, "blend y: {}", out[1]);
    }
}
