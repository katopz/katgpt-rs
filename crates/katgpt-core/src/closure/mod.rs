//! Closure-Expansion Instrument (CEI) — Primitive Transition Graphs + motifs + metrics.
//!
//! Implementation of the runtime/data-structure half of Momennejad & Raileanu,
//! "A Compositional Framework for Open-ended Intelligence" (arxiv 2606.15386,
//! Jun 2026). Turns any execution into an observable, committable
//! [`PrimitiveTransitionGraph`] (PTG); discovers recurring subgraphs
//! ([`Motif`]); and exposes the paper's §6 evaluation metrics (PRI / CDG / TaR).
//!
//! - **Plan 290** (`katgpt-rs/.plans/290_closure_expansion_instrument.md`)
//! - **Research 264** (`katgpt-rs/.research/264_Compositional_Open_Ended_Intelligence_Framework.md`)
//! - arxiv: <https://arxiv.org/abs/2606.15386>
//!
//! ## Scope — what lives here
//!
//! - **Raw/syncable**: PTG structure (nodes, edges, root, task_family_id),
//!   commitment hashes. Postcard-serializable, BLAKE3-committable (Plan 280
//!   Merkle-octree compatible).
//! - **Latent/local**: motif embeddings (per-PTG dot-product projections onto
//!   pre-computed direction vectors). Never synced directly.
//! - **Diagnostic scalar**: TaR score — a `[0,1]` Jaccard similarity between
//!   motif multisets. Public proxy for the real TaR (`AnchorProfile.translate_priorities()`,
//!   private IP in riir-ai).
//!
//! ## What this is NOT
//!
//! - No new capability class. The MDL admission gate lives in Plan 215;
//!   [`MotifAdmitter`] only emits a [`GateResult`] Phase 4 can wire.
//! - No NPP training objective (riir-train territory). PTGs are training
//!   targets the trainer can consume.
//! - No game semantics. `PrimitiveKind` reserves 0..=255 for engine use;
//!   game/runtime IDs stay in riir-ai and reference back via opaque `u32`.
//!
//! ## Feature gate
//!
//! Entire module is `#[cfg(feature = "closure_instrument")]`. Zero cost when
//! disabled — every public API vanishes from the build.

pub mod admit;
pub mod bridge;
pub mod metrics;
pub mod motif;
pub mod trace;

// ── Public type re-exports (flat namespace) ────────────────────────────────

pub use admit::{GateResult, MotifAdmitter, RejectionReason};
pub use bridge::{
    DEFAULT_MOTIF_DIRS, MotifDirections, motif_embedding_to_tar_score, ptg_to_motif_embedding,
};
pub use metrics::{CdgScore, PriScores, compute_tar_score};
pub use motif::{
    FixedU32Set, MAX_MOTIF_EDGES, MAX_MOTIF_NODES, Motif, MotifMiner, RING_BUFFER_K,
};
pub use trace::{NodeId, PtgRecorder};

// ── Cold-tier serde + commitment helpers ───────────────────────────────────
// (T1.3 — kept in `mod.rs` next to the data model; the file is small enough
//  that a separate `serde.rs` would just bounce re-exports.)

/// Postcard-encode a [`PrimitiveTransitionGraph`] for cold-tier commitment.
///
/// Bytes are deterministic for a given PTG (postcard is canonical for
/// fixed-shape structs). Used as input to [`commitment`].
#[inline]
pub fn serialize_postcard(ptg: &PrimitiveTransitionGraph) -> Result<Vec<u8>, postcard::Error> {
    postcard::to_allocvec(ptg)
}

/// Decode a [`PrimitiveTransitionGraph`] from postcard bytes.
///
/// Inverse of [`serialize_postcard`]. Round-trip preserves structure exactly
/// (see `tests::serde_roundtrip_preserves_structure`).
#[inline]
pub fn deserialize_postcard(bytes: &[u8]) -> Result<PrimitiveTransitionGraph, postcard::Error> {
    postcard::from_bytes(bytes)
}

/// BLAKE3 commitment of a [`PrimitiveTransitionGraph`].
///
/// Postcard-serializes then hashes — the canonical cold-tier commitment per
/// Plan 280 (Merkle-octree). Bytes are reproducible across runs and machines
/// (no floating point, no pointers, no RNG state).
#[inline]
pub fn commitment(ptg: &PrimitiveTransitionGraph) -> [u8; 32] {
    // Fall back to hashing an empty buffer on serialization failure — never
    // panics. (Serialization of these fixed-shape types cannot fail in
    // practice, but we keep the no-panic contract for the hot path.)
    let bytes = serialize_postcard(ptg).unwrap_or_default();
    blake3::hash(&bytes).into()
}

// ── Data model (Plan 290 §"Data Model", locked in Phase 0) ─────────────────

/// Stable identifier for a primitive in the open katgpt-rs enumeration space.
///
/// Encoded as `u32` so PTGs serialize/replay bit-identically across nodes
/// (raw/syncable per AGENTS.md "Latent vs Raw Space Rules"):
/// - `0..=255`: **engine** primitives (UserDefined). Open to katgpt-rs.
/// - `256..=511`: **composite** primitives (Composite) — admitted by
///   [`MotifAdmitter`] when a recurring motif crosses the MDL gate. The
///   embedded `u32` is the BE-encoded prefix of the motif's BLAKE3 hash.
/// - `512..`: reserved for game/runtime extensions (stay in riir-ai; never
///   leak game semantics into this enum).
///
/// Discriminant layout (not Rust enum discriminants — `PrimitiveKind` is a
/// 2-field tagged union encoded as a single `u32`):
/// ```text
/// | bits 31..9 | bit 8    | bits 7..0 |
/// | payload    | is_comp  | tag/id    |
/// ```
/// `to_u32` / `from_u32` give the canonical wire form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum PrimitiveKind {
    /// Engine primitive — `id` must be `< 256`.
    UserDefined(u32) = 0,
    /// Composite primitive — `blake3_prefix` should be `>= 256` (offset added
    /// by `to_u32`).
    Composite(/* blake3 prefix */ u32) = 256,
}

impl PrimitiveKind {
    /// Lower bound for the engine primitive id space.
    pub const USER_DEFINED_MIN: u32 = 0;
    /// Exclusive upper bound for engine primitive ids.
    pub const USER_DEFINED_MAX: u32 = 256;
    /// Lower bound for composite ids in the u32 wire form.
    pub const COMPOSITE_MIN: u32 = 256;
    /// Exclusive upper bound for composite ids.
    pub const COMPOSITE_MAX: u32 = 512;

    /// Canonical `u32` wire form.
    ///
    /// - `UserDefined(id)` for `id < 256` → `id`.
    /// - `Composite(prefix)` → `256 + (prefix & 0xFF)` (one byte of hash prefix).
    ///
    /// Inverse: [`from_u32`].
    #[inline]
    #[must_use]
    pub fn to_u32(self) -> u32 {
        match self {
            Self::UserDefined(id) => id.min(Self::USER_DEFINED_MAX - 1),
            Self::Composite(prefix) => Self::COMPOSITE_MIN + (prefix & 0xFF),
        }
    }

    /// Recover a [`PrimitiveKind`] from its canonical `u32` wire form.
    ///
    /// Values `< 256` → [`PrimitiveKind::UserDefined`]; values in `256..512`
    /// → [`PrimitiveKind::Composite`] with the lower byte preserved. Values
    /// `>= 512` are clamped into the composite space (defensive — they should
    /// not occur on the wire but never cause a runtime panic).
    #[inline]
    #[must_use]
    pub fn from_u32(v: u32) -> Self {
        if v < Self::COMPOSITE_MIN {
            Self::UserDefined(v)
        } else {
            Self::Composite(v - Self::COMPOSITE_MIN)
        }
    }

    /// `true` iff this is a composite primitive admitted by [`MotifAdmitter`].
    #[inline]
    #[must_use]
    pub fn is_composite(self) -> bool {
        matches!(self, Self::Composite(_))
    }
}

serde_via_u32!(PrimitiveKind);

/// Operator joining two PTG nodes — the "operadic" glue of a primitive graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OperatorKind {
    /// Linear `A → B` (B runs after A completes).
    Sequence = 0,
    /// Conditional dispatch (`A → B | skip`).
    Branch = 1,
    /// Self-composition (A calls A at smaller scale).
    Recurse = 2,
    /// Both branches run and join (`(A ∥ B) → C`).
    ParallelJoin = 3,
}

serde_via_u8!(OperatorKind);

/// A node in a [`PrimitiveTransitionGraph`]: one invocation of one primitive.
///
/// `blake3_in` is the commitment of the input state observed at entry — used
/// for tamper-evident replay (raw/syncable). It is **not** a latent embedding.
///
/// Stored as `Option<[u8; 32]>` (Plan 290 G4 fix, 2026-06-26) so the common
/// production case — a node with no per-node audit commitment, e.g. every
/// node emitted by [`PtgTracedPruner::trace`] — costs 1 byte (postcard `None`
/// tag) instead of 32 bytes of zeros. Callers that need tamper-evidence pass
/// `Some(hash)`; the G4 size target (< 1MB / 10K traces) is met when the
/// corpus is dominated by `None` nodes, which matches the production reality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PtgNode {
    /// Which primitive was invoked.
    pub primitive: PrimitiveKind,
    /// Tick at entry (raw — deterministic ordering).
    pub tick: u32,
    /// Optional BLAKE3 commitment of input state at this node (audit).
    /// `None` in the common production case (no per-node commitment).
    pub blake3_in: Option<[u8; 32]>,
}

/// An edge in a [`PrimitiveTransitionGraph`]: a primitive-to-primitive op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PtgEdge {
    /// How `from` and `to` compose.
    pub op: OperatorKind,
    /// Source node id (index into [`PrimitiveTransitionGraph::nodes`]).
    pub from: u32,
    /// Destination node id.
    pub to: u32,
}

/// A Primitive Transition Graph — directed graph of primitive invocations
/// produced by one execution of a [`crate::traits::ConstraintPruner`] (or any
/// other producer).
///
/// `root` is the index of the entry node; `task_family_id` groups PTGs from
/// the same task family for PRI computation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PrimitiveTransitionGraph {
    /// Nodes in insertion order (id == index).
    pub nodes: Vec<PtgNode>,
    /// Edges in insertion order.
    pub edges: Vec<PtgEdge>,
    /// Index into `nodes` of the entry node.
    pub root: u32,
    /// Which task family produced this PTG (drives PRI).
    pub task_family_id: u32,
}

impl PrimitiveTransitionGraph {
    /// Empty PTG (no nodes, no edges, root = 0). Used as a starting point by
    /// [`PtgRecorder`] and as a "feature disabled" placeholder.
    #[inline]
    #[must_use]
    pub fn empty(task_family_id: u32) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            root: 0,
            task_family_id,
        }
    }
}

// ── Custom serde for PrimitiveKind / OperatorKind ──────────────────────────
// The `serde_via_u32!` / `serde_via_u8!` macros impl Serialize + Deserialize
// via the canonical wire form so PTGs replay bit-identically across machines.

/// Implements `serde::Serialize` + `Deserialize` for a type with a `to_u32`
/// / `from_u32` pair. Keeps the wire form stable independent of Rust enum
/// layout.
macro_rules! serde_via_u32 {
    ($ty:ty) => {
        impl serde::Serialize for $ty {
            #[inline]
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_u32(<$ty>::to_u32(*self))
            }
        }

        impl<'de> serde::Deserialize<'de> for $ty {
            #[inline]
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let v = u32::deserialize(d)?;
                Ok(<$ty>::from_u32(v))
            }
        }
    };
}

/// Same idea, for `u8`-backed enums.
macro_rules! serde_via_u8 {
    ($ty:ty) => {
        impl serde::Serialize for $ty {
            #[inline]
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_u8(*self as u8)
            }
        }

        impl<'de> serde::Deserialize<'de> for $ty {
            #[inline]
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let v = u8::deserialize(d)?;
                Ok(match v {
                    0 => <$ty>::Sequence,
                    1 => <$ty>::Branch,
                    2 => <$ty>::Recurse,
                    _ => <$ty>::ParallelJoin,
                })
            }
        }
    };
}

pub(crate) use serde_via_u32;
pub(crate) use serde_via_u8;

// ── Tests (T1.4 + T1.5) ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_node(id: u32, tick: u32) -> PtgNode {
        PtgNode {
            primitive: PrimitiveKind::UserDefined(id),
            tick,
            blake3_in: Some([id as u8; 32]),
        }
    }

    #[test]
    fn primitive_kind_round_trip_u32() {
        for v in [0u32, 1, 100, 255] {
            let p = PrimitiveKind::from_u32(v);
            assert_eq!(p, PrimitiveKind::UserDefined(v));
            assert_eq!(p.to_u32(), v);
        }
        for v in [256u32, 257, 400, 511] {
            let p = PrimitiveKind::from_u32(v);
            assert!(matches!(p, PrimitiveKind::Composite(_)));
            assert_eq!(p.to_u32(), v);
        }
        assert!(!PrimitiveKind::UserDefined(0).is_composite());
        assert!(PrimitiveKind::Composite(7).is_composite());
    }

    #[test]
    fn serde_roundtrip_preserves_structure() {
        let ptg = PrimitiveTransitionGraph {
            nodes: vec![dummy_node(0, 10), dummy_node(1, 20), dummy_node(2, 30)],
            edges: vec![
                PtgEdge { op: OperatorKind::Sequence, from: 0, to: 1 },
                PtgEdge { op: OperatorKind::Branch, from: 1, to: 2 },
            ],
            root: 0,
            task_family_id: 42,
        };
        let bytes = serialize_postcard(&ptg).expect("serialize");
        let back = deserialize_postcard(&bytes).expect("deserialize");
        assert_eq!(back.nodes.len(), ptg.nodes.len());
        assert_eq!(back.edges.len(), ptg.edges.len());
        assert_eq!(back.root, ptg.root);
        assert_eq!(back.task_family_id, ptg.task_family_id);
        for (a, b) in back.nodes.iter().zip(ptg.nodes.iter()) {
            assert_eq!(a.primitive, b.primitive);
            assert_eq!(a.tick, b.tick);
            assert_eq!(a.blake3_in, b.blake3_in);
        }
        for (a, b) in back.edges.iter().zip(ptg.edges.iter()) {
            assert_eq!(a.op, b.op);
            assert_eq!(a.from, b.from);
            assert_eq!(a.to, b.to);
        }
    }

    #[test]
    fn commitment_is_deterministic() {
        let ptg = PrimitiveTransitionGraph {
            nodes: vec![dummy_node(0, 1)],
            edges: vec![],
            root: 0,
            task_family_id: 1,
        };
        let h1 = commitment(&ptg);
        let h2 = commitment(&ptg);
        assert_eq!(h1, h2, "commitment must be deterministic");
    }

    #[test]
    fn empty_ptg_serializes() {
        let ptg = PrimitiveTransitionGraph::empty(7);
        let bytes = serialize_postcard(&ptg).expect("serialize");
        let back = deserialize_postcard(&bytes).expect("deserialize");
        assert_eq!(back.task_family_id, 7);
        assert!(back.nodes.is_empty());
        assert!(back.edges.is_empty());
    }

    #[test]
    fn deeply_nested_chain_serializes() {
        // 5-level chain: root → n1 → n2 → n3 → n4
        let mut recorder = PtgRecorder::new(99);
        let root = recorder.enter(PrimitiveKind::UserDefined(0), 0, None);
        let mut prev = root;
        for i in 1..=4u32 {
            let cur = recorder.enter(PrimitiveKind::UserDefined(i), i, Some([i as u8; 32]));
            recorder.exit(prev, cur, OperatorKind::Recurse);
            prev = cur;
        }
        let ptg = recorder.finish();
        assert_eq!(ptg.nodes.len(), 5);
        assert_eq!(ptg.edges.len(), 4);
        let bytes = serialize_postcard(&ptg).expect("serialize");
        let back = deserialize_postcard(&bytes).expect("deserialize");
        assert_eq!(back.nodes.len(), 5);
        assert_eq!(back.edges.len(), 4);
        // Round-trip preserves the Option<blake3> field exactly — exercises
        // both the `None` (root) and `Some` (children) paths.
        assert_eq!(back.nodes[0].blake3_in, None);
        assert_eq!(back.nodes[1].blake3_in, Some([1u8; 32]));
    }
}
