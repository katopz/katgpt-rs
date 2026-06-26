//! ASOC (Asynchronous Spectral Operator Cascade) trait shapes.
//!
//! This module ships the GENERIC trait shapes only — **NO** `GpuFuture` import
//! (per the Plan 330 layering correction, 2026-06-26). The `ComposerTick:
//! GpuFuture` impl + `Join3` combinator live in `riir-engine/src/analytic_lattice/asoc.rs`
//! (Phase 1b, separate task), the only layer with both `katgpt-core` AND
//! `riir-gpu-async` in scope.
//!
//! # Why the split
//!
//! `RederiveOp::Fut` has NO bound at the trait level. The
//! `GpuFuture<Output = TransportOperator>` bound is applied at the impl site
//! in `riir-engine`. This keeps the leaf crate (`katgpt-core`) free of the
//! `riir-gpu-async` dependency — adding it here would invert the 5-repo
//! commercial boundary (R311 §6: "Generic math, no game IP" stays in katgpt-rs).
//!
//! # Per-tick cascade flow (the contract these traits support)
//!
//! See Plan 330 § "Cascade param threading" for the mermaid diagram. Per tick:
//!
//! 1. `ComposerCtx::new(tick, zone_hash)` is built once (cheap to clone).
//! 2. `PlasmaDraft::draft(&ctx)` → synchronous `Action` (always returns,
//!    never fails). Output stashed as the non-blocking fallback.
//! 3. `RederiveOp::rederive(&ctx)` (×3: boss / quest / player) → 3 `Fut`s
//!    that may never complete (GPU congestion).
//! 4. The `Fut`s are joined (in riir-engine's `Join3`):
//!    - **Ready** → `compose_chain` + `direction_vector_decode` → fresh action.
//!    - **Pending** → return the stashed plasma draft (non-blocking contract).

// `TransportOperator` is only referenced in doc comments at the trait level
// (the `Fut` associated type has no bound here). It IS used in the
// `#[cfg(test)]` block below. Gate the import accordingly to avoid an
// unused-import warning in non-test builds.
#[cfg(test)]
use crate::analytic_lattice::TransportOperator;

/// Plasma-tier synchronous draft producer.
///
/// Always completes in nanoseconds (a cache read + cheap decode); the ASOC
/// cascade returns its stale output when the hot-tier join returns
/// `Poll::Pending` (GPU congestion).
///
/// The concrete implementation lives in riir-ai (e.g. wraps
/// `riir-games::quest_draft::QuestDraftModel`). katgpt-rs ships only the trait.
///
/// # Contract
///
/// - `draft` MUST NOT block, allocate on the hot path, or fail.
/// - `draft` MUST be deterministic in `(self, ctx)` (same inputs → same action)
///   — this is what makes the stale fallback safe (the bot loop re-runs it
///   every tick per Plan 330 T1b.7, and the G1 determinism gate depends on it).
/// - The `Action` type is CALLER-defined (game IP). katgpt-core keeps it
///   generic.
pub trait PlasmaDraft {
    /// The action type. Caller-defined (game IP).
    type Action;

    /// Produce a synchronous draft action. Must not block, allocate, or fail.
    fn draft(&self, ctx: &ComposerCtx) -> Self::Action;
}

/// Hot-tier transport-operator rederive. Produces a future (associated type
/// `Fut`, no bound at the trait level) that resolves to a [`TransportOperator`]
/// when the work completes. The ASOC cascade joins 3 of these per tick
/// (`C_boss`, `C_quest`, `C_player`).
///
/// # The `Fut` bound (critical for layering)
///
/// `Fut` has **NO** `GpuFuture` bound at the trait level — that would require
/// importing `riir-gpu-async` into katgpt-core, inverting the 5-repo commercial
/// boundary. Instead, the `GpuFuture<Output = TransportOperator>` bound is
/// applied at the **impl site** in `riir-engine` (Phase 1b):
///
/// ```text,ignore
/// // in riir-engine/src/analytic_lattice/asoc.rs
/// impl<Rb: RederiveOp> ComposerTick<..., Rb, ...>
/// where
///     Rb::Fut: GpuFuture<Output = TransportOperator> + Unpin,
/// { /* ... */ }
/// ```
///
/// This keeps katgpt-core leaf-clean while still giving riir-engine a generic
/// trait to slot its `GpuFuture` impls into.
///
/// # Contract
///
/// - `rederive` itself is synchronous (returns the future immediately); only
///   the future's resolution is async.
/// - The returned `Fut` may legitimately never complete (GPU congestion) — the
///   ASOC cascade falls back to the stashed plasma draft in that case.
/// - `rederive` is keyed by `ctx.tick` / `ctx.zone_hash` for cache hits; each
///   impl owns its own entity-state source (see [`ComposerCtx`] shape contract).
pub trait RederiveOp {
    /// The future type. NO `GpuFuture` bound here — applied at the impl site
    /// in riir-engine. Resolves to a [`TransportOperator`] when the work
    /// completes.
    type Fut;

    /// Produce a hot-tier rederive future for this tick. Keyed by `ctx.tick`
    /// and `ctx.zone_hash` (the impl owns its own entity-state source).
    fn rederive(&self, ctx: &ComposerCtx) -> Self::Fut;
}

/// Per-tick composer context — shared read-only state used by both the plasma
/// draft and the hot-tier rederives.
///
/// # Shape contract (Plan 330 T1a.4)
///
/// `ComposerCtx` carries ONLY generic cache-keying + routing fields: `tick`
/// and `zone_hash`. It does NOT carry entity state.
///
/// Each [`RederiveOp`] impl owns its own entity-state source internally (e.g.
/// `BossRederiveOp` holds `Arc<BossSnapshots>` and only uses `ctx.tick` /
/// `ctx.zone_hash` to key its internal cache). This:
///
/// 1. **Avoids bloating the generic ctx struct with game-specific fields**
///    (no game IP leaks to katgpt-core).
/// 2. **Keeps `ctx.clone()` cheap** — needed for the warm-tier reflection
///    queue, where each entry is `(ctx.clone(), draft, fresh)` (see Plan 330
///    § Cascade param threading).
/// 3. **Respects the katgpt-core leaf discipline** — this struct is part of
///    the public leaf API; it cannot mention `BossSnapshots`, `PlayerState`,
///    or any riir-ai type.
///
/// # Why `Copy`
///
/// Two `u64`s (16 bytes). Cloning is cheaper than passing by reference for
/// all hot-path callers. The reflection queue clones it on every divergence
/// event; making it `Copy` removes the `.clone()` boilerplate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComposerCtx {
    /// Logical tick (e.g. Bevy `FixedUpdate` count at 20 Hz). Used to key the
    /// plasma-tier cache and to identify reflection-queue entries.
    pub tick: u64,
    /// Zone hash (BLAKE3 of the zone id, or a stable 64-bit zone key). Used to
    /// key the boss/quest rederive caches (zone-level facts) and to route
    /// reflection events to the correct warm-tier scheduler.
    pub zone_hash: u64,
}

impl ComposerCtx {
    /// Construct a new composer context. Generic — concrete construction from
    /// Bevy resources / game state lives in riir-ai.
    #[inline]
    pub const fn new(tick: u64, zone_hash: u64) -> Self {
        Self { tick, zone_hash }
    }
}

impl Default for ComposerCtx {
    #[inline]
    fn default() -> Self {
        Self::new(0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composer_ctx_new_and_copy() {
        let a = ComposerCtx::new(42, 0xDEAD_BEEF);
        let b = a; // Copy — no .clone() needed
        assert_eq!(a, b);
        assert_eq!(a.tick, 42);
        assert_eq!(a.zone_hash, 0xDEAD_BEEF);
    }

    #[test]
    fn rederive_op_no_bound_is_object_safe_shape() {
        // Compile-only check: a `RederiveOp` impl with a non-`GpuFuture` Fut
        // (here `()` for testing) satisfies the trait. This proves the trait
        // is leaf-clean — no `GpuFuture` import needed to implement it.
        struct DummyRederive;
        impl RederiveOp for DummyRederive {
            type Fut = TransportOperator; // not GpuFuture — but the trait doesn't care
            fn rederive(&self, _ctx: &ComposerCtx) -> Self::Fut {
                TransportOperator::identity(2)
            }
        }

        let ctx = ComposerCtx::new(1, 2);
        let op = DummyRederive;
        let fut = op.rederive(&ctx);
        assert_eq!(fut.k, 2);
    }

    #[test]
    fn plasma_draft_action_type_is_caller_defined() {
        struct StringDraft;
        impl PlasmaDraft for StringDraft {
            type Action = String;
            fn draft(&self, ctx: &ComposerCtx) -> Self::Action {
                format!("tick={} zone={:#x}", ctx.tick, ctx.zone_hash)
            }
        }
        let ctx = ComposerCtx::new(7, 0xBEEF);
        assert_eq!(StringDraft.draft(&ctx), "tick=7 zone=0xbeef");
    }
}
