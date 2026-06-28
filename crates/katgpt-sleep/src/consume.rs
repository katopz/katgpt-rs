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
//!
//! # Match modes (Issue 004)
//!
//! [`ConsumeMatchMode::Direction`] is the default topic-identification match
//! (`argmax_i <q, dir_i>`): forecast-independent, depends only on the static
//! direction catalog. [`ConsumeMatchMode::Precomputed`] is the answer-retrieval
//! match (`argmin_i ||slot_i.precomputed − q||²`): forecast-dependent, and
//! what makes the curiosity inversion's predictability correlation observable
//! through the live consume() path — when the sleep-time forecast was accurate,
//! the nearest precomputed slot is the player's true topic.

use katgpt_types::simd::{fast_sigmoid, simd_dot_f32};
use crate::types::AnticipatedQuerySet;

/// How [`consume_with_match_mode`] / [`consume_gate_with_match_mode`] match a
/// query `q` to an anticipated slot in `c'`.
///
/// Issue 004: the default [`ConsumeMatchMode::Direction`] preserves the original
/// `consume()` behavior (forecast-independent topic identification). The new
/// [`ConsumeMatchMode::Precomputed`] mode matches against the *forecast answers*
/// rather than the catalog directions — this is what makes the curiosity
/// inversion's predictability correlation observable through the live consume()
/// path: when the sleep-time forecast was accurate, the nearest precomputed
/// slot is the player's true topic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsumeMatchMode {
    /// `argmax_i <q, dir_i.direction>` — topic-identification match.
    /// Forecast-independent (depends only on `q` and the static catalog).
    /// This is the current `consume()` behavior and stays the default.
    Direction,
    /// `argmin_i ||slot_i.precomputed − q||²` — answer-retrieval match.
    /// Forecast-dependent. This is what makes the curiosity inversion
    /// observable through the live consume() path: when the forecast was
    /// accurate, the nearest precomputed slot is the player's true topic.
    ///
    /// Compares *squared* distances (no `sqrt`) — argmin is identical, and
    /// it keeps the inner loop branch-free and allocation-free per the
    /// AGENTS.md hot-loop rules.
    Precomputed,
}

/// Pick the best-matching slot index in `c_prime` for `q` under `match_mode`.
///
/// Single source of truth for the match logic shared by both
/// [`consume_with_match_mode`] and [`consume_gate_with_match_mode`]. Kept
/// branch-free and allocation-free; `K` is bounded (catalog size, paper K≤10).
#[inline]
fn find_best_match<const D: usize, const K: usize>(
    q: &[f32; D],
    c_prime: &AnticipatedQuerySet<D, K>,
    match_mode: ConsumeMatchMode,
) -> usize {
    match match_mode {
        ConsumeMatchMode::Direction => {
            // argmax_i <q, dir_i.direction> — forecast-independent topic match.
            let mut best_i = 0usize;
            let mut best_dot = f32::NEG_INFINITY;
            for i in 0..K {
                let d = simd_dot_f32(q, &c_prime.slots[i].dir.direction, D);
                if d > best_dot {
                    best_dot = d;
                    best_i = i;
                }
            }
            best_i
        }
        ConsumeMatchMode::Precomputed => {
            // argmin_i ||slot_i.precomputed − q||² — forecast-dependent
            // answer-retrieval match. No sqrt (argmin identical); branch-free
            // inner accumulation per AGENTS.md hot-loop rules.
            let mut best_i = 0usize;
            let mut best_dist = f32::INFINITY;
            for i in 0..K {
                let mut dist_sq = 0.0f32;
                for (q_val, &pre) in q.iter().zip(c_prime.slots[i].precomputed.iter()).take(D) {
                    let d = pre - q_val;
                    dist_sq += d * d;
                }
                if dist_sq < best_dist {
                    best_dist = dist_sq;
                    best_i = i;
                }
            }
            best_i
        }
    }
}

/// Wake-time consumer with an explicit [`ConsumeMatchMode`] (Issue 004).
///
/// Same gate + blend logic as [`consume`], but the slot match is parameterized
/// by `match_mode`. See [`ConsumeMatchMode`] for the two strategies and their
/// semantics. This is the canonical implementation; [`consume`] delegates here
/// with [`ConsumeMatchMode::Direction`].
///
/// # Algorithm
///
/// 1. Find the best-matching slot `i*` per `match_mode`.
/// 2. Compute the gate: `gate = sigmoid(beta * (p_{i*} − tau))`.
/// 3. Blend: `out = gate * z_{i*} + (1 − gate) * fresh_think(q)`.
///
/// # Allocation
///
/// Zero-allocation in steady state. The `fresh_think` closure MAY allocate
/// (fallback path only).
///
/// # Determinism
///
/// Given `(q, c_prime, tau, beta, match_mode)` and a deterministic
/// `fresh_think`, this function is deterministic.
#[inline]
pub fn consume_with_match_mode<const D: usize, const K: usize, F>(
    q: &[f32; D],
    c_prime: &AnticipatedQuerySet<D, K>,
    tau: f32,
    beta: f32,
    match_mode: ConsumeMatchMode,
    fresh_think: F,
) -> [f32; D]
where
    F: FnOnce(&[f32; D]) -> [f32; D],
{
    // 1. Best-matching slot per match_mode (single source of truth).
    let best_i = find_best_match(q, c_prime, match_mode);

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
///
/// # Match mode
///
/// Delegates to [`consume_with_match_mode`] with [`ConsumeMatchMode::Direction`]
/// — the original forecast-independent topic-identification match. Issue 004
/// preserved this behavior bit-identically.
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
    consume_with_match_mode(q, c_prime, tau, beta, ConsumeMatchMode::Direction, fresh_think)
}

/// Cheap gate-only check with an explicit [`ConsumeMatchMode`] (Issue 004):
/// returns `(best_i, gate)` without running `fresh_think`.
///
/// Same matching + gating as [`consume_with_match_mode`], but no blend — just
/// the decision. This is the canonical implementation; [`consume_gate`]
/// delegates here with [`ConsumeMatchMode::Direction`].
#[inline]
pub fn consume_gate_with_match_mode<const D: usize, const K: usize>(
    q: &[f32; D],
    c_prime: &AnticipatedQuerySet<D, K>,
    tau: f32,
    beta: f32,
    match_mode: ConsumeMatchMode,
) -> (usize, f32) {
    let best_i = find_best_match(q, c_prime, match_mode);
    let p = c_prime.slots[best_i].predictability;
    let gate = fast_sigmoid(beta * (p - tau));
    (best_i, gate)
}

/// Cheap gate-only check: returns `(best_i, gate)` without running `fresh_think`.
///
/// Consumers that want to skip fresh compute when the gate is high can call
/// this first, check the gate, and only call `fresh_think` if needed. This
/// keeps the primitive flexible without forcing every consumer to pay for
/// fresh compute on every call.
///
/// Same matching + gating as [`consume`], but no blend — just the decision.
///
/// Delegates to [`consume_gate_with_match_mode`] with
/// [`ConsumeMatchMode::Direction`] (Issue 004) — bit-identical to the
/// pre-refactor behavior.
#[inline]
pub fn consume_gate<const D: usize, const K: usize>(
    q: &[f32; D],
    c_prime: &AnticipatedQuerySet<D, K>,
    tau: f32,
    beta: f32,
) -> (usize, f32) {
    consume_gate_with_match_mode(q, c_prime, tau, beta, ConsumeMatchMode::Direction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anticipator::{IdentityFunctorOp, SleepTimeAnticipator, SleepTimeScratch};
    use crate::predictability::DotPredictabilityScorer;
    use crate::types::AnticipatedQueryDir;

    /// Build a small c' artifact for testing consume().
    fn build_artifact(
        c: &[f32; 2],
        dirs: &[AnticipatedQueryDir<2>; 2],
    ) -> crate::types::AnticipatedQuerySet<2, 2> {
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

    /// Build a hand-constructed 2-slot artifact with explicit precomputed
    /// answers (Issue 004 match-mode tests need control over `precomputed`,
    /// which `anticipate()` derives deterministically from `c`+`dir`).
    fn build_artifact_with_precomputed(
        dirs: [AnticipatedQueryDir<2>; 2],
        precomputed: [[f32; 2]; 2],
        predictability: [f32; 2],
    ) -> crate::types::AnticipatedQuerySet<2, 2> {
        let slots = [
            crate::types::AnticipatedSlot {
                dir: dirs[0].clone(),
                precomputed: precomputed[0],
                predictability: predictability[0],
            },
            crate::types::AnticipatedSlot {
                dir: dirs[1].clone(),
                precomputed: precomputed[1],
                predictability: predictability[1],
            },
        ];
        let blake3 = crate::types::AnticipatedQuerySet::<2, 2>::commit_slots(&slots);
        crate::types::AnticipatedQuerySet {
            slots,
            blake3,
            version: 0,
        }
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
        let slots = [crate::types::AnticipatedSlot {
            dir: dirs[0].clone(),
            precomputed: [10.0, 0.0],
            predictability: 0.5, // == tau → gate = sigmoid(0) = 0.5
        }];
        let blake3 = crate::types::AnticipatedQuerySet::<2, 1>::commit_slots(&slots);
        let artifact = crate::types::AnticipatedQuerySet {
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

    // ── Issue 004: ConsumeMatchMode tests ─────────────────────────────────────

    #[test]
    fn consume_precomputed_match_finds_nearest_slot() {
        // slot 0.precomputed = [10, 0], slot 1.precomputed = [0, 0].
        // q = [9, 0] is closer to slot 0 by both L2 (||[10,0]-[9,0]||²=1 vs
        // ||[0,0]-[9,0]||²=81) and direction alignment (dirs [1,0]/[0,1],
        // <q,dir_0>=9 > <q,dir_1>=0). Both modes pick slot 0 here — the point
        // is to confirm the Precomputed match path is wired and picks the
        // nearest precomputed answer.
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let artifact = build_artifact_with_precomputed(
            dirs,
            [[10.0, 0.0], [0.0, 0.0]],
            [0.9, 0.5],
        );
        let q = [9.0, 0.0];

        let (best_dir, _) = consume_gate_with_match_mode(&q, &artifact, 0.5, 4.0, ConsumeMatchMode::Direction);
        let (best_pre, _) = consume_gate_with_match_mode(&q, &artifact, 0.5, 4.0, ConsumeMatchMode::Precomputed);

        assert_eq!(best_dir, 0, "Direction mode: q=[9,0] aligns with dir_0=[1,0]");
        assert_eq!(best_pre, 0, "Precomputed mode: q=[9,0] nearest to slot 0 ([10,0])");
    }

    #[test]
    fn consume_precomputed_match_differs_from_direction_when_forecast_varies() {
        // The load-bearing test: construct a c' where Direction and Precomputed
        // modes genuinely disagree, proving the Precomputed path is
        // forecast-dependent (matches the *answers*, not the *catalog*).
        //
        // Setup (Issue 004 spec):
        //   dir_0 = [1, 0],   dir_1 = [0, 1]   (catalog axes)
        //   P_0   = [10, 0]                     (forecast for dir_0)
        //   P_1   = [0, 1]                      (forecast for dir_1)
        //
        // Pick q = [0, -1]. Then:
        //   Direction match:
        //     <q, dir_0> = 0,   <q, dir_1> = -1   → argmax = slot 0 (0 > -1)
        //   Precomputed match (squared L2):
        //     ||P_0 − q||² = ||[10, 1]||² = 100 + 1 = 101
        //     ||P_1 − q||² = ||[0, 2]||²  = 0 + 4   = 4
        //     → argmin = slot 1 (4 < 101)
        //
        // Disagreement: Direction → slot 0, Precomputed → slot 1.
        //
        // Intuition: q is geometrically nearer to P_1=[0,1] (both live near the
        // y-axis), even though q's *direction* leans toward the x-axis
        // (qx=0 > qy=−1). This is exactly the curiosity-inversion observable:
        // the forecast-answer geometry, not the catalog geometry, identifies
        // the player's true topic.
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let artifact = build_artifact_with_precomputed(
            dirs,
            [[10.0, 0.0], [0.0, 1.0]],
            [0.8, 0.8], // equal predictability so the gate value doesn't depend on the match
        );
        let q = [0.0, -1.0];

        let (best_dir, _) = consume_gate_with_match_mode(&q, &artifact, 0.5, 4.0, ConsumeMatchMode::Direction);
        let (best_pre, _) = consume_gate_with_match_mode(&q, &artifact, 0.5, 4.0, ConsumeMatchMode::Precomputed);

        assert_eq!(
            best_dir, 0,
            "Direction: <[0,-1],[1,0]>=0 > <[0,-1],[0,1]>=-1 → slot 0"
        );
        assert_eq!(
            best_pre, 1,
            "Precomputed: ||[10,0]-[0,-1]||²=101 > ||[0,1]-[0,-1]||²=4 → slot 1"
        );
        assert_ne!(
            best_dir, best_pre,
            "Direction and Precomputed modes must disagree on this q (the whole point)"
        );
    }

    #[test]
    fn consume_with_match_mode_preserves_direction_behavior() {
        // The refactor delegates consume() → consume_with_match_mode(Direction).
        // Prove the delegation is bit-identical: same inputs → same outputs,
        // for both the blend variant and the gate-only variant.
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let c = [0.7, 0.3];
        let artifact = build_artifact(&c, &dirs);
        let q = [0.9, 0.1];
        let tau = 0.5;
        let beta = 4.0;
        let fresh = |fq: &[f32; 2]| [fq[0] * 2.0, fq[1] + 0.5];

        let out_legacy = consume(&q, &artifact, tau, beta, fresh);
        let out_mode = consume_with_match_mode(&q, &artifact, tau, beta, ConsumeMatchMode::Direction, fresh);
        assert_eq!(
            out_legacy, out_mode,
            "consume() and consume_with_match_mode(Direction) must be bit-identical"
        );

        let gate_legacy = consume_gate(&q, &artifact, tau, beta);
        let gate_mode = consume_gate_with_match_mode(&q, &artifact, tau, beta, ConsumeMatchMode::Direction);
        assert_eq!(
            gate_legacy, gate_mode,
            "consume_gate() and consume_gate_with_match_mode(Direction) must be bit-identical"
        );
    }

    #[test]
    fn consume_with_match_mode_is_deterministic() {
        // Precomputed mode with a deterministic fresh_think must be reproducible.
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let artifact = build_artifact_with_precomputed(
            dirs,
            [[10.0, 0.0], [0.0, 1.0]],
            [0.7, 0.6],
        );
        let q = [9.1, 0.1];
        let fresh = |fq: &[f32; 2]| [fq[0] + 1.0, fq[1] - 1.0];

        let out1 = consume_with_match_mode(&q, &artifact, 0.5, 4.0, ConsumeMatchMode::Precomputed, fresh);
        let out2 = consume_with_match_mode(&q, &artifact, 0.5, 4.0, ConsumeMatchMode::Precomputed, fresh);
        assert_eq!(out1, out2, "Precomputed-mode consume must be deterministic");
    }

    #[test]
    fn consume_gate_with_match_mode_returns_valid_gate() {
        // For both modes, the gate value must land in [0, 1] (sigmoid range).
        let dirs = [
            AnticipatedQueryDir::new([1.0, 0.0]),
            AnticipatedQueryDir::new([0.0, 1.0]),
        ];
        let artifact = build_artifact_with_precomputed(
            dirs,
            [[10.0, 0.0], [0.0, 1.0]],
            [0.9, 0.1],
        );
        for q in &[[1.0, 0.0], [0.0, 1.0], [9.0, 0.0], [0.0, -1.0], [5.0, 5.0]] {
            for mode in [ConsumeMatchMode::Direction, ConsumeMatchMode::Precomputed] {
                let (_, gate) = consume_gate_with_match_mode(q, &artifact, 0.5, 4.0, mode);
                assert!(
                    (0.0..=1.0).contains(&gate),
                    "gate {} out of [0,1] for q={:?} mode={:?}",
                    gate,
                    q,
                    mode
                );
            }
        }
    }
}
