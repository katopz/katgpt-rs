//! Phase 4 composition tests — verify the five Plan 329 primitives compose
//! cleanly with the four existing systems they're designed to fuse with.
//!
//! - T4.1: `BranchBank` × `arg_protocol` — `BranchLifecycle` IS ARG
//!   `LifecycleState` (type alias); spawn/merge/prune round-trip through ARG
//!   lifecycle; ARG `RedirectTable` provides post-merge continuity.
//! - T4.2: `VerifierGate` × CLR — `should_write_composed` short-circuits on
//!   CLR reject; CLR accept + centroid check compose as upstream→downstream
//!   gates; end-to-end CLR+Verifier+BranchBank write pipeline.
//! - T4.3: `BranchRouter` × `engram` — branches anchored on embeddings
//!   derived from Engram `multi_head_hash` outputs; same-suffix snaps,
//!   different-suffix spawns.
//! - T4.4: `NonInterferenceProjection` × `closure` — PTG motif embeddings
//!   populate `ProceduralRule.direction`; assign + project round-trip.
//!
//! These are **open-primitive** composition tests: they verify the *seams*
//! where two independently-shipped modules meet. CLR (Plan 284) and the
//! riir-ai runtime are NOT in katgpt-core, so T4.2 simulates CLR's
//! `should_write_memory(r_k, S_LP)` as a local closure (the real CLR lives in
//! riir-ai; the composition contract is the `clr_allows: bool` parameter).

#![cfg(test)]

use super::{BranchBank, VerifierGate, WriteDecision};

// ──────────────────────────────────────────────────────────────────────────
// T4.1 — BranchBank × arg_protocol
// ──────────────────────────────────────────────────────────────────────────
//
// When `arg_protocol` is enabled (DEFAULT-ON since Plan 327 Phase 4),
// `BranchLifecycle` is a *type alias* for `crate::arg::LifecycleState`. This
// makes branch lifecycle state committable and redirect-resolvable via the
// ARG `RedirectTable` (Step E). The composition contract:
//
//   spawn  → Active     (= LifecycleState::Active)
//   prune  → Removed    (= LifecycleState::Removed)
//   merge  → source Removed (data moved to target); caller registers
//            RedirectTable redirect for external-reference continuity.
//
// NOTE on plan-vs-code: Plan 329 T4.1 narrative says "merge = source →
// Deprecated + RedirectTable". The *implemented* `BranchBank::merge` moves
// the source's episodic/procedural/failure stores into the target (via
// `append`) then calls `prune(source)` → `Removed`. `Deprecated` would leave
// an empty husk (the data already moved); `Removed` is semantically correct
// for a data-moving merge. The ARG `RedirectTable` composition is a
// *caller-side* continuity pattern: after merge, the caller registers
// `source_label → target_label` so external references (e.g., episodic
// records keyed by old BranchId) still resolve. This test demonstrates both
// halves: the bank's actual merge→Removed behavior AND the caller-side
// RedirectTable composition.
#[cfg(feature = "arg_protocol")]
mod t41_arg_protocol {
    use super::BranchBank;
    use crate::arg::{LifecycleState, RedirectTable};
    use crate::arg::taxonomy::LabelId;
    use crate::branching::types::BranchLifecycle;

    /// Compile-time proof that `BranchLifecycle` IS `LifecycleState` when
    /// `arg_protocol` is on (the type-alias composition). If this compiles,
    /// the two types are literally the same — no conversion needed.
    #[test]
    fn branch_lifecycle_is_arg_lifecycle_state_type_alias() {
        fn _assert_type_identity<T>(_x: T)
        where
            T: std::fmt::Debug + Copy + PartialEq + Default,
        {
        }
        // Both must be callable with the same value — they ARE the same type.
        let active: BranchLifecycle = LifecycleState::Active;
        let also_active: LifecycleState = BranchLifecycle::Active;
        assert_eq!(active, also_active);
        _assert_type_identity::<BranchLifecycle>(active);
        _assert_type_identity::<LifecycleState>(also_active);
    }

    /// Discriminant identity: the four branch lifecycle states map 1:1 to
    /// the four ARG `LifecycleState` variants by `#[repr(u8)]` value.
    #[test]
    fn lifecycle_discriminants_match_arg() {
        assert_eq!(LifecycleState::Active as u8, BranchLifecycle::Active as u8);
        assert_eq!(LifecycleState::Shadow as u8, BranchLifecycle::Shadow as u8);
        assert_eq!(
            LifecycleState::Deprecated as u8,
            BranchLifecycle::Deprecated as u8
        );
        assert_eq!(LifecycleState::Removed as u8, BranchLifecycle::Removed as u8);
    }

    /// `is_routable` / `requires_redirect` semantics are identical between
    /// the branching and ARG views (because they're the same type).
    #[test]
    fn lifecycle_predicates_identical_to_arg() {
        assert!(BranchLifecycle::Active.is_routable());
        assert!(LifecycleState::Active.is_routable());
        assert!(!BranchLifecycle::Shadow.is_routable());
        assert!(!LifecycleState::Shadow.is_routable());

        assert!(BranchLifecycle::Deprecated.requires_redirect());
        assert!(LifecycleState::Deprecated.requires_redirect());
        assert!(BranchLifecycle::Removed.requires_redirect());
        assert!(LifecycleState::Removed.requires_redirect());
        assert!(!BranchLifecycle::Active.requires_redirect());
    }

    /// Spawn → `Active` (= ARG `LifecycleState::Active`). The newly-spawned
    /// branch is routable by the router (the `is_routable()` filter passes).
    #[test]
    fn spawn_yields_active_lifecycle() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let id = bank.spawn(vec![1.0, 0.0, 0.0]).expect("spawn ok");
        let branch = bank.get(id).expect("branch exists");
        assert_eq!(branch.lifecycle, BranchLifecycle::Active);
        assert_eq!(branch.lifecycle, LifecycleState::Active);
        assert!(branch.lifecycle.is_routable());
    }

    /// Prune → `Removed` (= ARG `LifecycleState::Removed`). The pruned branch
    /// requires redirect for future lookups (ARG §3.5 continuity).
    #[test]
    fn prune_yields_removed_lifecycle() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let id = bank.spawn(vec![1.0, 0.0]).expect("spawn ok");
        assert!(bank.prune(id));
        let branch = bank.get(id).expect("slot retained");
        assert_eq!(branch.lifecycle, BranchLifecycle::Removed);
        assert_eq!(branch.lifecycle, LifecycleState::Removed);
        assert!(!branch.lifecycle.is_routable());
        assert!(branch.lifecycle.requires_redirect());
    }

    /// Merge: source → `Removed` (data moved to target); target stays
    /// `Active`. This is the implemented data-moving merge semantics.
    #[test]
    fn merge_source_removed_target_active() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let target = bank.spawn(vec![1.0, 0.0]).expect("target spawn");
        let source = bank.spawn(vec![0.0, 1.0]).expect("source spawn");

        let result = bank.merge(target, source).expect("merge ok");
        assert_eq!(result, target);

        // Target stays Active and routable.
        let tgt = bank.get(target).expect("target exists");
        assert_eq!(tgt.lifecycle, BranchLifecycle::Active);
        assert!(tgt.lifecycle.is_routable());

        // Source is Removed (data moved); requires redirect.
        let src = bank.get(source).expect("source slot retained");
        assert_eq!(src.lifecycle, BranchLifecycle::Removed);
        assert!(src.lifecycle.requires_redirect());
    }

    /// ARG `RedirectTable` composition: after merge, the caller registers
    /// `source_label → target_label` so external references to the old
    /// BranchId resolve to the target. This is the continuity pattern from
    /// ARG §3.5 — the bank doesn't do it internally (it doesn't know about
    /// LabelId), but the types compose cleanly when the caller wires it.
    #[test]
    fn arg_redirect_table_composes_post_merge() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let target = bank.spawn(vec![1.0, 0.0]).expect("target spawn");
        let source = bank.spawn(vec![0.0, 1.0]).expect("source spawn");

        // Merge moves source data into target; source → Removed.
        bank.merge(target, source).expect("merge ok");

        // Caller-side continuity: register redirect in ARG RedirectTable.
        // BranchId.0 is a u32; LabelId::new takes u32 — direct mapping.
        let redirects = RedirectTable::new();
        let source_label = LabelId::new(source.into());
        let target_label = LabelId::new(target.into());
        redirects.insert_redirect(source_label, target_label);

        // External lookup: "where did source go?" → resolves to target.
        let resolved = redirects.redirect(source_label);
        assert_eq!(resolved, Some(target_label));

        // Chain walk: includes the start node, so [source, target] (len 2).
        let chain = redirects.redirect_chain(source_label);
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0], source_label);
        assert_eq!(chain[1], target_label);
    }

    /// Full lifecycle round-trip through ARG types: spawn (Active) →
    /// deprecate-via-redirect (Deprecated) → prune (Removed), verifying
    /// every state is a valid `LifecycleState` and round-trips through
    /// `BranchLifecycle` (same type).
    #[test]
    fn full_lifecycle_round_trips_through_arg() {
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let id = bank.spawn(vec![1.0, 0.0]).expect("spawn");

        // Active (spawn).
        let b = bank.get(id).unwrap();
        let state: LifecycleState = b.lifecycle; // no conversion — same type
        assert_eq!(state, LifecycleState::Active);

        // Manually set Deprecated (simulating a caller-driven soft-retire
        // before prune — the ARG progressive-activation pattern).
        {
            let b = bank.get_mut(id).unwrap();
            b.lifecycle = BranchLifecycle::Deprecated;
        }
        let b = bank.get(id).unwrap();
        assert_eq!(b.lifecycle, LifecycleState::Deprecated);
        assert!(!b.lifecycle.is_routable());
        assert!(b.lifecycle.requires_redirect());

        // Prune → Removed (bank.prune only works on routable branches, so
        // re-activate first to exercise the prune path).
        {
            let b = bank.get_mut(id).unwrap();
            b.lifecycle = BranchLifecycle::Active;
        }
        assert!(bank.prune(id));
        let b = bank.get(id).unwrap();
        assert_eq!(b.lifecycle, LifecycleState::Removed);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T4.2 — VerifierGate × CLR (upstream composition)
// ──────────────────────────────────────────────────────────────────────────
//
// CLR (Plan 284) is the upstream two-sided reward gate:
// `should_write_memory(r_k, S_LP)` returns true iff `r_k > τ_reliable ∧
// S_LP > τ_curiosity`. CLR does NOT live in katgpt-core (it's a riir-ai
// runtime concern); the composition contract is the `clr_allows: bool`
// parameter on `VerifierGate::should_write_composed`.
//
// These tests simulate CLR locally (a closure replicating the two-sided
// gate) and verify the upstream→downstream composition:
//   CLR reject  → VerifierGate Reject (short-circuit, centroid never checked)
//   CLR accept  → VerifierGate applies centroid quarantine downstream
mod t42_clr_composition {
    use super::{VerifierGate, WriteDecision};
    use crate::branching::bank::BranchBank;

    /// Local stand-in for CLR's `should_write_memory(r_k, S_LP)`.
    /// Returns `true` iff both reward and curiosity exceed their CLR
    /// thresholds (the two-sided gate). This mirrors the contract documented
    /// in `verifier.rs` line 4.
    fn clr_should_write_memory(reward: f32, curiosity: f32) -> bool {
        let tau_reliable = 0.5;
        let tau_curiosity = 0.3;
        reward > tau_reliable && curiosity > tau_curiosity
    }

    /// CLR reject short-circuits the VerifierGate: the centroid check is
    /// never reached. This is the upstream-gate-rejects-fast-path contract.
    #[test]
    fn clr_reject_short_circuits_before_centroid_check() {
        let gate = VerifierGate::default();
        // CLR rejects (reward too low).
        let clr_allows = clr_should_write_memory(0.2, 0.9);
        assert!(!clr_allows);
        // Even with a perfect centroid, the composed gate Rejects.
        let decision = gate.should_write_composed(clr_allows, 0.99);
        assert_eq!(decision, WriteDecision::Reject);
    }

    /// CLR accept + on-centroid → Write. The downstream centroid check passes.
    #[test]
    fn clr_accept_plus_on_centroid_writes() {
        let gate = VerifierGate::default();
        let clr_allows = clr_should_write_memory(0.8, 0.5);
        assert!(clr_allows);
        let decision = gate.should_write_composed(clr_allows, 0.9);
        assert_eq!(decision, WriteDecision::Write);
    }

    /// CLR accept + off-centroid → Quarantine. The downstream centroid check
    /// catches a potential contamination that CLR's reward+curiosity gate
    /// alone would have admitted. This is the RIZZ value-add: CLR is
    /// necessary but not sufficient; the centroid quarantine adds a
    /// branch-local non-interference guard.
    #[test]
    fn clr_accept_but_off_centroid_quarantines() {
        let gate = VerifierGate::default();
        let clr_allows = clr_should_write_memory(0.8, 0.5);
        assert!(clr_allows);
        let decision = gate.should_write_composed(clr_allows, 0.3);
        assert_eq!(decision, WriteDecision::Quarantine);
    }

    /// Compositional equivalence: `should_write_composed(clr_allows, sim)`
    /// equals `should_write(r, c, sim)` when `clr_allows` is computed from
    /// the SAME `(r, c)` via the CLR gate. This proves the two API surfaces
    /// agree on the decision — the composition is consistent, not ad hoc.
    #[test]
    fn composed_gate_equivalent_to_direct_gate_under_same_inputs() {
        let gate = VerifierGate::default();
        for (r, c, sim) in [
            (0.8_f32, 0.5_f32, 0.9_f32), // all pass → Write
            (0.2, 0.9, 0.9),             // reward low → Reject
            (0.8, 0.1, 0.9),             // curiosity low → Reject
            (0.8, 0.5, 0.3),             // off-centroid → Quarantine
            (0.5, 0.5, 0.9),             // boundary reward (==) → Reject
            (0.8, 0.3, 0.9),             // boundary curiosity (==) → Reject
        ] {
            let clr_allows = clr_should_write_memory(r, c);
            let composed = gate.should_write_composed(clr_allows, sim);
            let direct = gate.should_write(r, c, sim);
            assert_eq!(
                composed, direct,
                "mismatch at (r={r}, c={c}, sim={sim}): composed={composed:?} direct={direct:?}"
            );
        }
    }

    /// End-to-end pipeline: CLR gate → VerifierGate → BranchBank write.
    /// Only writes that pass BOTH gates reach the branch's episodic store.
    /// This is the full RIZZ §"verifier-gated memory" composition.
    #[test]
    fn clr_plus_verifier_plus_bank_end_to_end() {
        let gate = VerifierGate::default();
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let branch = bank.spawn(vec![1.0, 0.0, 0.0]).expect("spawn");

        // Case 1: CLR + Verifier both pass → write admitted.
        let r = 0.8_f32;
        let c = 0.5_f32;
        let sim = 0.9_f32;
        let clr_ok = clr_should_write_memory(r, c);
        let decision = gate.should_write_composed(clr_ok, sim);
        assert_eq!(decision, WriteDecision::Write);
        assert!(decision.is_write());
        let wrote = bank.write_episodic(branch, vec![1.0, 0.0, 0.0], (), r, None, 0);
        assert!(wrote, "write should succeed when decision is Write");
        assert_eq!(bank.get(branch).unwrap().episodic.len(), 1);

        // Case 2: CLR rejects → write NOT admitted (we check the gate, skip
        // the bank call — the runtime pattern is "ask the gate, then write").
        let r_bad = 0.2_f32;
        let clr_bad = clr_should_write_memory(r_bad, c);
        let decision_bad = gate.should_write_composed(clr_bad, sim);
        assert_eq!(decision_bad, WriteDecision::Reject);
        assert!(decision_bad.is_blocked());
        // Runtime would skip bank.write_episodic here. Episodic count stays 1.
        assert_eq!(bank.get(branch).unwrap().episodic.len(), 1);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T4.3 — BranchRouter × engram
// ──────────────────────────────────────────────────────────────────────────
//
// Engram (Plan 299) produces K_MAX `EngramHash(u64)` addresses from a token
// suffix via `multi_head_hash`. The composition: derive a deterministic
// `spawn_anchor: Vec<f32>` from the K_MAX hashes (one f32 per head), use it
// to spawn a branch, then verify the router snaps to that branch when given
// a query derived from the SAME suffix (and does NOT snap for a different
// suffix).
//
// This is the "branches whose `spawn_anchor` is derived from Engram
// hash-address embeddings" pattern from Plan 329 T4.3.
#[cfg(feature = "engram")]
mod t43_engram_composition {
    use super::BranchBank;
    use crate::branching::router::{BranchRouter, RouteMode};
    use crate::engram::{multi_head_hash, CanonicalId, EngramHash, HashHead, K_MAX};

    /// Build K_MAX deterministic hash heads (distinct seeds + primes) for the
    /// composition fixture. Mirrors the Engram test helper convention.
    fn make_heads(seed_base: u64) -> [HashHead; K_MAX] {
        let mut heads = [HashHead {
            n: 64,
            k: 0,
            modulus: 1_000_000_007,
            seed: seed_base,
        }; K_MAX];
        for (k, h) in heads.iter_mut().enumerate() {
            h.k = k as u8;
            // Distinct prime modulus + distinct seed per head.
            h.modulus = 1_000_000_007_u64.wrapping_add((k as u64) * 2u64);
            h.seed = seed_base.wrapping_add(k as u64 * 7919);
        }
        heads
    }

    /// Derive a deterministic `spawn_anchor` embedding from K_MAX EngramHash
    /// values. Each hash head contributes one f32 dimension. This is the
    /// "hash-address embedding" bridge: Engram addresses → latent anchor.
    ///
    /// We map the full 64-bit `EngramHash(u64)` to `f32` via a `[0, 1]`
    /// fixed-point conversion (using all 64 bits, not just the high bits —
    /// high-bit-only mappings collide for suffixes whose hashes share a
    /// prefix). The router then does dot-product snap on these anchors.
    ///
    /// The anchor is NOT pre-normalized here (the router's snap threshold is
    /// tuned for pre-normalized embeddings); for the composition test we use
    /// a low `tau_snap` so un-normalized anchors still snap on exact-match.
    fn engram_hashes_to_anchor(hashes: &[EngramHash; K_MAX]) -> Vec<f32> {
        hashes
            .iter()
            .map(|h| (h.0 as f64 / u64::MAX as f64) as f32)
            .collect()
    }

    /// Same-suffix snaps: a branch anchored on Engram hashes of suffix S is
    /// found by routing a query derived from the SAME suffix S. The router
    /// returns `Reuse` with the correct branch id.
    #[test]
    fn same_suffix_snaps_to_engram_anchored_branch() {
        let heads = make_heads(42);
        let suffix_a = [CanonicalId(1), CanonicalId(2), CanonicalId(3)];
        let hashes_a = multi_head_hash(&suffix_a, &heads);
        let anchor_a = engram_hashes_to_anchor(&hashes_a);

        let mut bank: BranchBank<()> = BranchBank::new(8);
        let branch = bank.spawn(anchor_a.clone()).expect("spawn");

        // Router with a low snap threshold (un-normalized anchors).
        let router = BranchRouter::new(0.0, 0.40, 0.0);
        // Query from the same suffix → same hashes → same anchor → high dot.
        let result = router.route(&anchor_a, &bank);
        assert_eq!(result.mode, RouteMode::Reuse);
        assert_eq!(result.branch, Some(branch));
    }

    /// Different-suffix does NOT snap: a query derived from a different
    /// suffix produces a different anchor; the router does not falsely match.
    /// With distinct hash heads, different suffixes → different hashes →
    /// different anchors. The router returns `Spawn` (capacity available).
    #[test]
    fn different_suffix_does_not_snap() {
        let heads = make_heads(42);
        let suffix_a = [CanonicalId(1), CanonicalId(2), CanonicalId(3)];
        let suffix_b = [CanonicalId(4), CanonicalId(5), CanonicalId(6)];
        let hashes_a = multi_head_hash(&suffix_a, &heads);
        let hashes_b = multi_head_hash(&suffix_b, &heads);
        let anchor_a = engram_hashes_to_anchor(&hashes_a);
        let anchor_b = engram_hashes_to_anchor(&hashes_b);

        // Sanity: different suffixes → different anchors (at least one dim).
        let any_diff = anchor_a.iter().zip(anchor_b.iter()).any(|(x, y)| x != y);
        assert!(any_diff, "different suffixes must produce different anchors");

        let mut bank: BranchBank<()> = BranchBank::new(8);
        let _branch = bank.spawn(anchor_a).expect("spawn");

        let router = BranchRouter::new(0.92, 0.40, 0.0);
        let result = router.route(&anchor_b, &bank);
        // anchor_b != anchor_a → no dot-product snap → Spawn (capacity avail).
        assert_eq!(result.mode, RouteMode::Spawn);
        assert!(result.branch.is_none());
    }

    /// Engram determinism → router determinism: the same suffix always
    /// produces the same hashes (Engram T1.6 contract), which always produce
    /// the same anchor, which always routes to the same branch. The
    /// composition is deterministic end-to-end.
    #[test]
    fn engram_determinism_implies_router_determinism() {
        let heads = make_heads(99);
        let suffix = [CanonicalId(7), CanonicalId(11), CanonicalId(13)];

        // Hash twice → identical hashes (Engram determinism).
        let h1 = multi_head_hash(&suffix, &heads);
        let h2 = multi_head_hash(&suffix, &heads);
        assert_eq!(h1, h2);

        // Same hashes → same anchor → same route.
        let anchor1 = engram_hashes_to_anchor(&h1);
        let anchor2 = engram_hashes_to_anchor(&h2);
        assert_eq!(anchor1, anchor2);

        let mut bank: BranchBank<()> = BranchBank::new(8);
        let id = bank.spawn(anchor1.clone()).expect("spawn");
        let router = BranchRouter::new(0.0, 0.40, 0.0);
        let r1 = router.route(&anchor1, &bank);
        let r2 = router.route(&anchor2, &bank);
        assert_eq!(r1, r2);
        assert_eq!(r1.branch, Some(id));
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T4.4 — NonInterferenceProjection × closure
// ──────────────────────────────────────────────────────────────────────────
//
// The closure-instrument (Plan 290) mines `Motif`s from
// `PrimitiveTransitionGraph`s. `ptg_to_motif_embedding(&ptg, &dirs)` projects
// a PTG's primitive-frequency feature vector through a `MotifDirections`
// table → a K-dim sigmoid embedding. The composition: use this embedding as
// `ProceduralRule.direction` AND as a `NonInterferenceProjection` direction
// for a branch. Two branches anchored on embeddings from two DIFFERENT PTGs
// measure their interference honestly via the projection.
#[cfg(feature = "closure_instrument")]
mod t44_closure_composition {
    use super::BranchBank;
    use crate::branching::projection::NonInterferenceProjection;
    use crate::branching::types::{BranchId, ProceduralRule};
    use crate::closure::bridge::{ptg_to_motif_embedding, MotifDirections};
    use crate::closure::{PrimitiveKind, PrimitiveTransitionGraph, PtgNode};

    const D: usize = 8;

    /// Build a deterministic, non-zero `MotifDirections` table of shape
    /// (K=D, N=D). We use a circulant pattern so each direction row is
    /// distinct and non-zero (zeros → all-0.5 sigmoid output → zero-magnitude
    /// direction, which `assign_direction` rejects).
    fn make_dirs() -> MotifDirections {
        let mut flat = vec![0.0f32; D * D];
        for k in 0..D {
            for n in 0..D {
                // Circulant: row k is a rotated identity-like basis.
                flat[k * D + n] = if (k + n) % D == 0 { 1.0 } else { 0.05 };
            }
        }
        MotifDirections::from_flat(flat, D, D).expect("shape D*D")
    }

    /// Build a PTG with the given primitive ids (as `UserDefined` nodes).
    fn make_ptg(task_family: u32, primitives: &[u32]) -> PrimitiveTransitionGraph {
        let mut ptg = PrimitiveTransitionGraph::empty(task_family);
        for (i, &p) in primitives.iter().enumerate() {
            ptg.nodes.push(PtgNode {
                primitive: PrimitiveKind::UserDefined(p),
                tick: i as u32,
                blake3_in: None,
            });
        }
        ptg
    }

    /// L2-normalize a vector in place. `assign_direction` requires normalized
    /// directions; the motif embedding is sigmoid-bounded (0,1) and generally
    /// NOT unit-norm.
    fn l2_normalize(v: &mut [f32]) {
        let mut norm_sq = 0.0f32;
        for &x in v.iter() {
            norm_sq += x * x;
        }
        if norm_sq > 0.0 {
            let inv = 1.0 / norm_sq.sqrt();
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
    }

    /// PTG motif embedding → `ProceduralRule.direction`: the embedding
    /// populates the rule's latent direction field directly. This is the
    /// "closure motifs can populate `ProceduralRule.direction` from PTG node
    /// signatures" contract from Plan 329 T4.4.
    #[test]
    fn motif_embedding_populates_procedural_rule_direction() {
        let dirs = make_dirs();
        let ptg = make_ptg(0, &[0, 1, 2, 3]);
        let emb = ptg_to_motif_embedding(&ptg, &dirs);

        // Embedding has the right dimensionality (K = D).
        assert_eq!(emb.len(), D);

        // Populate a ProceduralRule with this direction.
        let rule = ProceduralRule {
            direction: emb.clone(),
            antecedent: [0u8; 32],
            strategy: [1u8; 32],
            helpful: 0,
            harmful: 0,
        };
        assert_eq!(rule.direction.len(), D);
        assert_eq!(rule.net_credit(), 0);

        // Store it in a branch.
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let id = bank.spawn(vec![1.0; D]).expect("spawn");
        {
            let b = bank.get_mut(id).unwrap();
            b.procedural.push(rule);
        }
        assert_eq!(bank.get(id).unwrap().procedural.len(), 1);
        assert_eq!(bank.get(id).unwrap().procedural[0].direction, emb);
    }

    /// Motif embedding → projection direction: the same embedding (normalized)
    /// serves as a `NonInterferenceProjection` direction. `assign_direction`
    /// succeeds; `project(branch, &embedding)` returns a high value (the
    /// branch projects strongly onto its own direction).
    #[test]
    fn motif_embedding_serves_as_projection_direction() {
        let dirs = make_dirs();
        let ptg = make_ptg(0, &[0, 1, 2, 3]);
        let mut emb = ptg_to_motif_embedding(&ptg, &dirs);
        assert_eq!(emb.len(), D);
        l2_normalize(&mut emb);

        let mut proj: NonInterferenceProjection<D> = NonInterferenceProjection::default();
        let branch = BranchId::new(0);
        let result = proj.assign_direction(branch, &emb);
        assert!(result.error.is_none(), "assign should succeed: {:?}", result);

        // Project the embedding onto the branch's own direction → high value.
        let projected = proj.project(branch, &emb).expect("assigned");
        // Since direction == embedding (normalized), dot ≈ |embedding|² = 1.0.
        assert!(
            (projected - 1.0).abs() < 1e-5,
            "self-projection should be ≈1.0, got {projected}"
        );
    }

    /// Two branches anchored on embeddings from two DIFFERENT PTGs: their
    /// interference is measured honestly by the projection.
    ///
    /// NOTE: closure motif embeddings (sigmoid-bounded, always-positive) are
    /// NOT naturally orthogonal — sigmoid(0)=0.5 creates a high baseline, so
    /// even disjoint primitive sets produce embeddings with significant dot
    /// product after normalization. This test verifies the projection reports
    /// that interference *honestly* (whatever it is), not that the embeddings
    /// are orthogonal. The `assign_max_interference` is raised to 1.0 here so
    /// both assignments succeed; we then verify `interference()` matches a
    /// manual dot-product computation.
    #[test]
    fn two_ptgs_measure_interference_honestly() {
        let dirs = make_dirs();
        // PTG A: primitives {0,1,2,3}.
        let ptg_a = make_ptg(0, &[0, 1, 2, 3]);
        // PTG B: primitives {4,5,6,7} — different primitive set.
        let ptg_b = make_ptg(1, &[4, 5, 6, 7]);

        let mut emb_a = ptg_to_motif_embedding(&ptg_a, &dirs);
        let mut emb_b = ptg_to_motif_embedding(&ptg_b, &dirs);
        l2_normalize(&mut emb_a);
        l2_normalize(&mut emb_b);

        // The two embeddings are distinct (different primitive frequencies).
        let any_diff = emb_a.iter().zip(emb_b.iter()).any(|(x, y)| x != y);
        assert!(any_diff, "distinct PTGs must produce distinct embeddings");

        // Manual interference: |dot(emb_a, emb_b)| (both normalized → in [0,1]).
        let manual_interference: f32 = emb_a
            .iter()
            .zip(emb_b.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            manual_interference >= 0.0 && manual_interference <= 1.0 + 1e-6,
            "manual interference in [0,1]: {manual_interference}"
        );

        // Projection with raised max_interference so both assignments succeed
        // (we're measuring interference, not gating on it).
        let mut proj: NonInterferenceProjection<D> =
            NonInterferenceProjection::with_thresholds(64, 1e-6, 1.0);
        let b0 = BranchId::new(0);
        let b1 = BranchId::new(1);
        let r0 = proj.assign_direction(b0, &emb_a);
        let r1 = proj.assign_direction(b1, &emb_b);
        assert!(r0.error.is_none(), "assign b0: {:?}", r0);
        assert!(r1.error.is_none(), "assign b1: {:?}", r1);

        // The projection's interference() must match the manual dot product.
        let measured = proj.interference(b0, b1);
        assert!(
            (measured - manual_interference).abs() < 1e-5,
            "projection interference {measured} must match manual {manual_interference}"
        );

        // Cross-projection: b0's direction projected onto emb_b equals the
        // same dot product (interference is symmetric |dot|).
        let cross = proj.project(b0, &emb_b).expect("assigned");
        assert!(
            (cross - manual_interference).abs() < 1e-5,
            "cross-projection ≈ interference: cross={cross}, manual={manual_interference}"
        );
    }

    /// Full closure→branching pipeline: mine a motif embedding from a PTG,
    /// use it as BOTH a `ProceduralRule.direction` AND a projection
    /// direction, store the rule in a branch, and verify the projection is
    /// consistent with the stored rule's direction.
    #[test]
    fn closure_to_branching_full_pipeline() {
        let dirs = make_dirs();
        let ptg = make_ptg(42, &[0, 1, 0, 2, 1, 3]);
        let mut emb = ptg_to_motif_embedding(&ptg, &dirs);
        assert_eq!(emb.len(), D);
        l2_normalize(&mut emb);

        // 1. Projection direction.
        let mut proj: NonInterferenceProjection<D> = NonInterferenceProjection::default();
        let branch = BranchId::new(0);
        let assign = proj.assign_direction(branch, &emb);
        assert!(assign.error.is_none(), "assign: {:?}", assign);
        assert!(proj.is_non_interfering_with_all(branch));

        // 2. ProceduralRule with the same direction.
        let rule = ProceduralRule {
            direction: emb.clone(),
            antecedent: [2u8; 32],
            strategy: [3u8; 32],
            helpful: 3,
            harmful: 1,
        };
        assert_eq!(rule.net_credit(), 2);

        // 3. Store in a bank branch.
        let mut bank: BranchBank<()> = BranchBank::new(8);
        let id = bank.spawn(vec![0.0; D]).expect("spawn");
        bank.get_mut(id).unwrap().procedural.push(rule);

        // 4. Verify consistency: the stored rule's direction projects ≈1.0
        //    onto the branch's projection direction (they're the same vector).
        let stored_dir = &bank.get(id).unwrap().procedural[0].direction;
        let projected = proj.project(branch, stored_dir).expect("assigned");
        assert!(
            (projected - 1.0).abs() < 1e-5,
            "stored rule direction projects ≈1.0: got {projected}"
        );
    }
}
