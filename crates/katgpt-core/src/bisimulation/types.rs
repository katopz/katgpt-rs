//! Core types for bisimulation operator inference (Plan 324 T1.2–T1.8).
//!
//! All types are `Copy` (where possible), `#[repr(transparent)]` or
//! `#[repr(C)]` for sync-friendly layout, and have no heap allocation in their
//! own field layout — only the owning containers (`Vec`-backed graphs and
//! quotients) allocate, and those allocations are confined to construction
//! time, never the hot path.

use core::fmt;
use core::hash::{Hash, Hasher};

// ─── State identifiers ─────────────────────────────────────────────────────

/// Newtype wrapper around a raw `u32` state identifier.
///
/// `StateId`s are dense, non-negative indices into a `TransitionGraph`'s
/// `states` array (consumer must remap sparse domain ids → dense `StateId`s
/// before constructing a graph). `#[repr(transparent)]` guarantees the same
/// ABI as `u32`, so an array of `StateId` is byte-compatible with `&[u32]`.
///
/// Why a newtype: prevents mixing `StateId` with `StateClassId` (the quotient
/// side) at the type level. Both are `u32` underneath; conflating them in
/// caller code is a class of bug this primitive specifically wants to make
/// unrepresentable.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StateId(pub u32);

impl StateId {
    /// Construct from a `u32`. Convenience method.
    #[inline]
    pub const fn new(v: u32) -> Self {
        Self(v)
    }

    /// Sentinel for "no state". Used internally for sentinel-slot patterns
    /// during partition refinement (Paige-Tarjan splits); never escapes the
    /// algorithm boundary.
    pub const SENTINEL: Self = Self(u32::MAX);

    /// True if this is the sentinel value.
    #[inline]
    pub const fn is_sentinel(self) -> bool {
        self.0 == u32::MAX
    }
}

impl From<u32> for StateId {
    #[inline]
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<StateId> for u32 {
    #[inline]
    fn from(s: StateId) -> u32 {
        s.0
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

// ─── State-class identifiers ───────────────────────────────────────────────

/// Newtype wrapper around a raw `u32` state-class identifier.
///
/// A `StateClassId` is the *quotient-side* counterpart to [`StateId`]: after
/// partition refinement, every `StateId` is mapped to exactly one
/// `StateClassId`. Two states are in the same class iff they are
/// bisimulation-equivalent (their outgoing labeled transitions land in the
/// same class set).
///
/// Distinct from `StateId` at the type level so consumers can't accidentally
/// index a quotient with a raw-state id (or vice versa).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StateClassId(pub u32);

impl StateClassId {
    #[inline]
    pub const fn new(v: u32) -> Self {
        Self(v)
    }

    /// Sentinel — mirrors [`StateId::SENTINEL`].
    pub const SENTINEL: Self = Self(u32::MAX);
}

impl From<u32> for StateClassId {
    #[inline]
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<StateClassId> for u32 {
    #[inline]
    fn from(c: StateClassId) -> u32 {
        c.0
    }
}

impl fmt::Display for StateClassId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "c{}", self.0)
    }
}

// ─── Operator labels ───────────────────────────────────────────────────────

/// Abstract operator tag.
///
/// The NSM paper (arXiv:2508.21501) infers operator labels from the symbolic
/// abstraction stage; this primitive treats them as opaque tags. We ship a
/// small fixed set of common manipulation labels (mirroring the
/// Towers-of-Hanoi demo fixture) plus an `Other(u8)` escape hatch for
/// domain extension without re-compiling this crate.
///
/// `#[repr(u8)]` keeps the discriminant 1 byte; the `Other(u8)` payload
/// variant makes the total enum 2 bytes (disc + payload). `Copy + Eq +
/// Hash + Ord` make it usable as a `Vec` sort key and a `HashMap` key
/// (during cold-tier analysis).
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OperatorLabel {
    /// Pick up the topmost object from a stack (Hanoi demo).
    PickTop = 0,
    /// Place an object onto a non-empty stack (Hanoi demo).
    PlaceOn = 1,
    /// Place an object onto an empty peg (Hanoi demo).
    PlaceOnEmpty = 2,
    /// Generic "no-op" sentinel — sometimes useful for self-loops in
    /// degenerate graphs.
    NoOp = 3,
    /// Domain extension escape hatch. The inner `u8` is caller-defined; this
    /// primitive never assigns `Other` values itself.
    Other(u8) = 255,
}

impl Default for OperatorLabel {
    #[inline]
    fn default() -> Self {
        Self::NoOp
    }
}

impl OperatorLabel {
    /// Stable discriminant for sorting / hashing. The `Other(v)` variant
    /// returns `v + 4` (offset past the fixed variants) so the order is
    /// `PickTop < PlaceOn < PlaceOnEmpty < NoOp < Other(v)`.
    #[inline]
    pub const fn discriminant(self) -> u8 {
        match self {
            Self::PickTop => 0,
            Self::PlaceOn => 1,
            Self::PlaceOnEmpty => 2,
            Self::NoOp => 3,
            // Adding 4 keeps `Other(0)` strictly greater than the named
            // variants (4 > 3), preserving sort stability across enum
            // extensions.
            Self::Other(v) => v.saturating_add(4),
        }
    }
}

// ─── Edge / transition records ─────────────────────────────────────────────

/// A single observed transition in the raw transition graph.
///
/// `#[repr(C)]` so a slice of `Transition` is a contiguous byte array
/// (`Vec<Transition>` is then bit-compatible with `&[u8]` of size
/// `8 * len`). This matters for two reasons:
///
/// 1. Sort + dedup at graph-build time is branch-free on packed records.
/// 2. BLAKE3 commitment on the quotient's edge set can use a contiguous
///    byte slice directly (no per-element serialization overhead).
///
/// All fields are `Copy`. Total size is 12 bytes (4 + 4 + 1 + 3 padding to
/// align to the `u32` of `StateId`). The 3-byte tail pad is unavoidable with
/// a `u8`-typed `op`; if a consumer needs tighter packing they can use a
/// `u32`-backed operator enum at the cost of 3× the edge-set memory.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Transition {
    /// Source state id.
    pub from: StateId,
    /// Destination state id.
    pub to: StateId,
    /// Operator label observed on this transition.
    pub op: OperatorLabel,
}

impl Transition {
    #[inline]
    pub const fn new(from: StateId, to: StateId, op: OperatorLabel) -> Self {
        Self { from, to, op }
    }
}

impl PartialOrd for Transition {
    /// Lexicographic ordering `(from, op, to)` — matches the canonical sort
    /// used by `TransitionGraphBuilder::build()` so adjacency scans for a
    /// given `(from, op)` are contiguous.
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Transition {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // (from, op.discriminant(), to) — op uses discriminant() so the
        // ordering is total even with the `Other(u8)` escape hatch.
        self.from
            .cmp(&other.from)
            .then(self.op.discriminant().cmp(&other.op.discriminant()))
            .then(self.to.cmp(&other.to))
    }
}

/// A single edge in the *quotient* (post-bisimulation) graph.
///
/// `from` / `to` are now class ids (not raw state ids). Same layout as
/// [`Transition`] but with the class-id newtype for type safety.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct QuotientEdge {
    pub from: StateClassId,
    pub to: StateClassId,
    pub op: OperatorLabel,
}

impl QuotientEdge {
    #[inline]
    pub const fn new(from: StateClassId, to: StateClassId, op: OperatorLabel) -> Self {
        Self { from, to, op }
    }
}

impl PartialOrd for QuotientEdge {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QuotientEdge {
    /// Canonical sort key `(from, op.discriminant(), to)`. Matches the
    /// `Transition` ordering so the quotient graph can be byte-compared
    /// against hand-constructed reference quotients in tests.
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.from
            .cmp(&other.from)
            .then(self.op.discriminant().cmp(&other.op.discriminant()))
            .then(self.to.cmp(&other.to))
    }
}

// ─── Hash-by-bytes for commitment helpers ──────────────────────────────────
//
// The `blake3` commitment in `refine.rs` hashes a canonical byte layout
// directly (no per-field serialization). For that we rely on the `#[repr(C)]`
// layout of `Transition` / `QuotientEdge` being stable and contiguous when
// stored in a `Vec`. The helpers below exist for any caller that wants to
// hash the *content* (rather than the structural identity) of a record.

/// Write a `StateId` in canonical little-endian form.
#[inline]
pub fn write_state_id<H: Hasher>(s: StateId, h: &mut H) {
    s.0.to_le_bytes().iter().for_each(|b| b.hash(h));
}

/// Write a `StateClassId` in canonical little-endian form.
#[inline]
pub fn write_class_id<H: Hasher>(c: StateClassId, h: &mut H) {
    c.0.to_le_bytes().iter().for_each(|b| b.hash(h));
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_id_is_u32_abit_compatible() {
        // `#[repr(transparent)]` → `size_of::<StateId>() == size_of::<u32>()`.
        assert_eq!(core::mem::size_of::<StateId>(), core::mem::size_of::<u32>());
        assert_eq!(
            core::mem::align_of::<StateId>(),
            core::mem::align_of::<u32>()
        );
    }

    #[test]
    fn state_class_id_is_u32_abi_compatible() {
        assert_eq!(
            core::mem::size_of::<StateClassId>(),
            core::mem::size_of::<u32>()
        );
    }

    #[test]
    fn operator_label_is_compact() {
        // `#[repr(u8)]` sets discriminant to 1 byte. The `Other(u8)`
        // variant carries a 1-byte payload → total enum size is 2 bytes
        // (disc + payload). Without the payload variant it would be 1 byte.
        assert_eq!(core::mem::size_of::<OperatorLabel>(), 2);
    }

    #[test]
    fn transition_is_twelve_bytes() {
        // `from: u32` (4) + `to: u32` (4) + `op: u8` (1) + 3 bytes tail
        // padding to align the struct to `u32`. Total 12 bytes.
        assert_eq!(core::mem::size_of::<Transition>(), 12);
        assert_eq!(core::mem::size_of::<QuotientEdge>(), 12);
        assert_eq!(core::mem::align_of::<Transition>(), 4);
    }

    #[test]
    fn state_id_sentinel_round_trips() {
        let s = StateId::SENTINEL;
        assert!(s.is_sentinel());
        assert!(!StateId::new(42).is_sentinel());
    }

    #[test]
    fn operator_label_discriminant_is_total_order() {
        // Named variants sort first, `Other(v)` sorts strictly after.
        let labels = [
            OperatorLabel::PickTop,
            OperatorLabel::PlaceOn,
            OperatorLabel::PlaceOnEmpty,
            OperatorLabel::NoOp,
            OperatorLabel::Other(0),
            OperatorLabel::Other(5),
        ];
        for w in labels.windows(2) {
            assert!(
                w[0].discriminant() < w[1].discriminant(),
                "discriminant order broken: {:?} ({}) vs {:?} ({})",
                w[0],
                w[0].discriminant(),
                w[1],
                w[1].discriminant()
            );
        }
    }

    #[test]
    fn transition_lex_order_is_total() {
        let t1 = Transition::new(StateId(0), StateId(1), OperatorLabel::PickTop);
        let t2 = Transition::new(StateId(0), StateId(1), OperatorLabel::PlaceOn);
        let t3 = Transition::new(StateId(0), StateId(2), OperatorLabel::PickTop);
        let t4 = Transition::new(StateId(1), StateId(0), OperatorLabel::PickTop);
        // cmp key: (from, op.discriminant(), to)
        // t1=(0,0,1), t2=(0,1,1), t3=(0,0,2), t4=(1,0,0)
        // Sorted: t1 < t3 < t2 < t4
        assert!(t1 < t3); // (0,0,1) < (0,0,2)
        assert!(t3 < t2); // (0,0,2) < (0,1,1) — op 0 < op 1
        assert!(t2 < t4); // from 0 < from 1
        // Total: t1 < t3 < t2 < t4
    }

    #[test]
    fn quotient_edge_order_matches_transition_order() {
        let q1 = QuotientEdge::new(StateClassId(0), StateClassId(1), OperatorLabel::PickTop);
        let q2 = QuotientEdge::new(StateClassId(0), StateClassId(2), OperatorLabel::PickTop);
        assert!(q1 < q2); // from equal, op equal, to 1 < 2
    }
}
