//! Type definitions for the salience tri-gate primitive.
//!
//! See [`crate::salience`] for the full module doc + paper citation.

/// First-class output of the salience gate. `Silent` is a *decision*, not a
/// default suppression — subscribers observe it through the same channels as
/// `Speak` / `Delegate`.
///
/// Generic over the delegate payload `A`. We bound only `Clone` (no `Eq`)
/// because delegate payloads may carry floats / vectors in caller code; the
/// gate itself never needs `Eq` on `A` to make a decision.
///
/// Reference: Plan 303 (T1.4), Research 281.
#[derive(Clone, Debug, PartialEq)]
pub enum SalienceDecision<A> {
    /// Agent actively chose silence this tick. Emit a [`SilenceToken`] via
    /// the caller's observe channel in Phase 3+.
    Silent,
    /// Speak inline (no async escalation).
    Speak,
    /// Delegate to an async backend. The payload is a typed handoff
    /// (`DelegateToken<A>`) — the caller spawns the task.
    Delegate(A),
}

/// Newtype signaling "this NPC actively chose silence this tick".
///
/// Flow through the same observe channels as `Speak`/`Delegate` so subscribers
/// can distinguish "nothing to say" from "explicitly chose silence". The
/// primitive returns the bare [`SalienceDecision::Silent`] variant in Phase 1;
/// `SilenceToken` construction is wired in Phase 3 (Plan 303 T3.x).
///
/// Reference: Plan 303 (T1.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SilenceToken {
    /// Tick at which silence was chosen.
    pub tick: u64,
}

impl SilenceToken {
    /// Construct a `SilenceToken` for the given tick.
    #[inline]
    pub fn new(tick: u64) -> Self {
        Self { tick }
    }
}

/// Where the async delegate result lands. Open enum — the backend is the
/// caller's concern (AnyRAG gateway / Engram / Cold-tier / etc. live in
/// `riir-neuron-db` and `riir-ai`, not here).
///
/// Reference: Plan 303 (T1.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FoldbackTarget {
    /// Result becomes a new direction in the caller's latent activation state.
    ActivationState = 0,
    /// Result is a hash-addressed pattern in the caller's memory system.
    PatternMemory = 1,
    /// Result routes through an external gateway (caller's network layer).
    ExternalJudge = 2,
    /// Result is a frozen shard in the caller's persistence (Cold) tier.
    ColdTier = 3,
}

/// Typed handoff returned inside [`SalienceDecision::Delegate`].
///
/// The caller (riir-ai runtime, Plan 330) owns the async spawn; this crate
/// only provides the typed payload. `holding_reply_idx` is an index into a
/// caller-provided template table — we just store it, validation is the
/// caller's responsibility (Plan 303 T3.1).
///
/// Reference: Plan 303 (T1.4).
#[derive(Clone, Debug)]
pub struct DelegateToken<A: Clone> {
    /// Caller-supplied payload (prompt, shard ref, judge request, …).
    pub payload: A,
    /// Tick at which the delegate decision was issued.
    pub issued_tick: u64,
    /// Index into the caller's holding-reply template table.
    pub holding_reply_idx: u8,
    /// Where the async result should fold back to.
    pub foldback_target: FoldbackTarget,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_token_constructs_and_carries_tick() {
        let t = SilenceToken::new(42);
        assert_eq!(t.tick, 42);
        // Copy semantics: rebind is cheap.
        let t2 = t;
        assert_eq!(t, t2);
    }

    #[test]
    fn foldback_target_repr_u8_values() {
        // Stable wire encoding — the caller may persist these as u8.
        assert_eq!(FoldbackTarget::ActivationState as u8, 0);
        assert_eq!(FoldbackTarget::PatternMemory as u8, 1);
        assert_eq!(FoldbackTarget::ExternalJudge as u8, 2);
        assert_eq!(FoldbackTarget::ColdTier as u8, 3);
    }

    #[test]
    fn decision_variants_match_without_eq_bound_on_a() {
        // Payload type that is Clone + Debug + PartialEq but NOT Eq (carries f32).
        #[derive(Clone, Debug, PartialEq)]
        struct FloatPayload(f32);

        let silent: SalienceDecision<FloatPayload> = SalienceDecision::Silent;
        let speak: SalienceDecision<FloatPayload> = SalienceDecision::Speak;
        let delegate: SalienceDecision<FloatPayload> =
            SalienceDecision::Delegate(FloatPayload(1.5));

        assert_eq!(silent, SalienceDecision::Silent);
        assert_eq!(speak, SalienceDecision::Speak);
        assert_eq!(delegate, SalienceDecision::Delegate(FloatPayload(1.5)));
        assert_ne!(silent, speak);
    }
}
