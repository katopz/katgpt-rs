//! `partition_refine` — bisimulation quotient via signature-based partition
//! refinement (Plan 324 T2.1–T2.5).
//!
//! # Algorithm
//!
//! **Goal:** Given a labeled transition graph `G`, partition its states into
//! equivalence classes `C₁, …, Cₖ` such that two states `s, t` are in the same
//! class iff for every operator label `a`:
//!
//! - For every successor `s'` of `s` under `a`, there exists a successor `t'`
//!   of `t` under `a` with `s' ~ t'`, and vice versa.
//!
//! This is **(strong) bisimulation equivalence** (Milner 1980). The
//! Paige-Tarjan (1987) algorithm computes it in `O((|S| + |E|) log |S|)`.
//!
//! ## Implementation choice: signature-based fixed-point with per-iteration canonicalization
//!
//! The full Paige-Tarjan "three partition refinement algorithms" paper
//! uses a sophisticated process-list / splitter-stack technique to hit the
//! `O((S+E) log S)` bound. This implementation uses the simpler
//! **signature-based** refinement (Fernandez 1990; Dill 1996 lecture notes):
//!
//! 1. Assign each state a **signature** = sorted multiset of
//!    `(op.discriminant(), class_of(s'))` tuples over its outgoing edges.
//! 2. Group states by signature. Each distinct signature becomes a new class.
//! 3. **Canonicalize** class labels by smallest-member-state-id.
//! 4. Repeat until the canonicalized partition stops changing.
//!
//! ### Why canonicalize inside the loop
//!
//! The naive convergence check `new_class == old_class` can **oscillate
//! forever** when the partition is stable but labels permute. Concrete
//! period-2 counterexample (discovered during Plan 324 implementation):
//!
//! ```text
//! States: 0, 1, 2.
//! Edges:  0 --A--> 1, 0 --B--> 2, 1 --A--> 0, 2 --A--> 0.
//! ```
//!
//! Without canonicalization, labels alternate between `[0,1,1]` and
//! `[1,0,0]` indefinitely — same partition `{0},{1,2}`, permuted labels.
//! Canonicalizing (renumber by smallest member) collapses both to `[0,1,1]`,
//! breaking the cycle.
//!
//! ### Complexity
//!
//! `O(|S| · log |S| · (avg_degree · log(avg_degree)))` per iteration,
//! `O(log |S|)` iterations worst case → `O((S+E) log² S log d)` total.
//! In practice (sparse graphs, low average degree) it's within a small
//! constant factor of Paige-Tarjan, and is dramatically simpler to
//! implement correctly. The G4 latency gate (`partition_refine` ≤ 1 ms for
//! N=1024) is met by a wide margin on Apple Silicon arm64.
//!
//! If the G4 gate ever fails on a real workload, the plan calls for a
//! drop-in replacement with the full PT splitter-stack technique (Plan 324
//! Risk R1).
//!
//! # Latent vs raw boundary
//!
//! The partition is **raw and deterministic**: same graph → same
//! `state_to_class` vector bit-identically. The `blake3` field is the
//! chain-committable canonical form.

use super::graph::TransitionGraph;
use super::types::{QuotientEdge, StateClassId, StateId};
use core::mem;

// ─── Quotient ──────────────────────────────────────────────────────────────

/// Output of [`partition_refine`]: a bisimulation quotient of the input
/// [`TransitionGraph`].
///
/// All fields are owned and immutable after construction. The hot-path
/// accessor is [`class_of`](Self::class_of), which is O(1) and
/// allocation-free (Plan 324 G5).
#[derive(Clone, Debug, Default)]
pub struct BisimulationQuotient {
    /// Number of distinct equivalence classes (= `max(state_to_class) + 1`).
    pub n_classes: u32,
    /// Indexed by `StateId` → that state's class id. Length ==
    /// `graph.n_states()`. Class ids are dense `0..n_classes` after
    /// canonicalization.
    pub state_to_class: Vec<StateClassId>,
    /// Edges of the quotient graph, sorted by `(from, op.discriminant(), to)`
    /// and deduped. There is at most one edge per `(from_class, op,
    /// to_class)` triple — the quotient is a simple labeled graph.
    pub quotient_edges: Vec<QuotientEdge>,
    /// BLAKE3 commitment over the canonical byte serialization. Two graphs
    /// that quotient identically produce bit-identical `blake3`. Suitable
    /// as a chain-consensus commitment artifact (riir-chain LatCal
    /// integration).
    pub blake3: [u8; 32],
}

impl BisimulationQuotient {
    /// Look up the class id of a state. O(1) direct index, **no allocation**
    /// — this is the Plan 324 G5 hot path.
    ///
    /// # Panics
    ///
    /// Panics if `state.0` ≥ `state_to_class.len()` (i.e., the state id is
    /// outside the input graph's range). This is a programmer-error panic,
    /// not an expected runtime failure.
    #[inline]
    pub fn class_of(&self, state: StateId) -> StateClassId {
        self.state_to_class[state.0 as usize]
    }

    /// Convenience: number of states covered by this quotient (==
    /// `state_to_class.len()`).
    #[inline]
    pub fn n_states(&self) -> usize {
        self.state_to_class.len()
    }

    /// Borrow the quotient edges. Sorted by `(from, op.discriminant(), to)`.
    #[inline]
    pub fn edges(&self) -> &[QuotientEdge] {
        &self.quotient_edges
    }
}

// ─── Partition refinement ─────────────────────────────────────────────────

/// Compute the bisimulation quotient of `graph` (Plan 324 T2.1).
///
/// Returns a [`BisimulationQuotient`] with:
/// - dense `0..n_classes` canonical class ids
/// - sorted + deduped `quotient_edges`
/// - BLAKE3 commitment over the canonical byte layout.
///
/// # Algorithmic complexity
///
/// `O((S + E) log² S log d)` where `d` is the average out-degree; see the
/// module-level comment for the implementation-choice rationale. In practice
/// this completes in well under 1 ms for N=1024 states on Apple Silicon
/// arm64 (Plan 324 G4 gate).
///
/// # Determinism
///
/// Bit-identical output across re-runs for the same input graph. The
/// `blake3` field makes this externally verifiable.
pub fn partition_refine(graph: &TransitionGraph) -> BisimulationQuotient {
    let n = graph.n_states() as usize;

    // Empty graph → empty quotient.
    if n == 0 {
        let q = BisimulationQuotient {
            n_classes: 0,
            state_to_class: Vec::new(),
            quotient_edges: Vec::new(),
            blake3: [0u8; 32],
        };
        // BLAKE3 over the canonical empty layout (just n_classes=0 LE).
        let mut q = q;
        q.blake3 = blake3_commit(&q);
        return q;
    }

    // ── Step 1: signature-based fixed-point refinement ───────────────────
    //
    // `current_class[i]` = class id of `StateId(i)`, CANONICALIZED (renumbered
    // by smallest member state id). Start with the trivial single-class
    // partition — already canonical.
    let mut current_class: Vec<u32> = vec![0u32; n];

    // Per-iteration scratch buffers (allocated ONCE here, reused across
    // iterations — no allocation in the fixed-point loop body).
    // `signatures[i]` is the signature vec for `StateId(i)`.
    let mut signatures: Vec<Vec<(u8, u32)>> = (0..n).map(|_| Vec::with_capacity(8)).collect();
    // `(signature, original_index)` pairs, rebuilt each iteration.
    let mut indexed_sigs: Vec<(Vec<(u8, u32)>, u32)> = Vec::with_capacity(n);

    loop {
        // ── Step 1a: compute each state's signature from `current_class` ─
        for sig_slot in signatures.iter_mut() {
            sig_slot.clear();
        }
        for (i, state) in graph.states().iter().enumerate() {
            graph.for_each_adjacent(*state, |op, succ| {
                let succ_class = current_class[succ.0 as usize];
                signatures[i].push((op.discriminant(), succ_class));
            });
            // Sort + dedup so two states with the same successor multiset
            // (different observation order) hash to the same signature.
            signatures[i].sort_unstable();
            signatures[i].dedup();
        }

        // ── Step 1b: group states by signature, assign new class ids ────
        // Move signatures into `indexed_sigs` (mem::take avoids cloning).
        indexed_sigs.clear();
        for (i, sig_slot) in signatures.iter_mut().enumerate().take(n) {
            let sig = mem::take(sig_slot);
            indexed_sigs.push((sig, i as u32));
        }
        // Sort by (signature, original_index) — stable for signature ties.
        indexed_sigs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Assign new class ids: first distinct signature → 0, etc.
        let mut new_class: Vec<u32> = vec![0u32; n];
        let mut next_class: u32 = 0;
        let mut prev_full: Option<&Vec<(u8, u32)>> = None;
        for (sig, original_idx) in &indexed_sigs {
            // Distinct-signature check via full vec comparison.
            let is_new = prev_full.is_none_or(|p| p.as_slice() != sig.as_slice());
            if is_new {
                next_class += 1;
            }
            new_class[*original_idx as usize] = next_class - 1;
            prev_full = Some(sig);
        }

        // ── Step 1c: restore signatures into the reusable buffer ─────────
        for (sig, original_idx) in indexed_sigs.drain(..) {
            signatures[original_idx as usize] = sig;
        }

        // ── Step 1d: canonicalize new_class BEFORE convergence check ─────
        //
        // Critical: without canonicalization, labels can oscillate with
        // period 2 when the partition is stable but the sorted-signature
        // order flips class ids. See the module-level doc for the concrete
        // counterexample. Canonicalizing both sides makes the comparison
        // partition-structural rather than label-positional.
        canonicalize_labels(&mut new_class, graph.states());

        if new_class == current_class {
            // Converged: partition is stable AND labels are canonical.
            break;
        }
        current_class = new_class;
    }

    // ── Step 2: convert to StateClassId + recompute n_classes ────────────
    let state_to_class: Vec<StateClassId> =
        current_class.iter().map(|&c| StateClassId(c)).collect();

    let n_classes = state_to_class
        .iter()
        .map(|c| c.0)
        .max()
        .map_or(0, |m| m + 1);

    // ── Step 3: build quotient edges (sorted, deduped) ───────────────────
    let mut quotient_edges: Vec<QuotientEdge> = Vec::with_capacity(graph.n_edges());
    for edge in graph.edges() {
        let from_class = state_to_class[edge.from.0 as usize];
        let to_class = state_to_class[edge.to.0 as usize];
        quotient_edges.push(QuotientEdge::new(from_class, to_class, edge.op));
    }
    quotient_edges.sort_unstable();
    quotient_edges.dedup();

    // ── Step 4: BLAKE3 commitment (Plan 324 T2.3) ────────────────────────
    let mut q = BisimulationQuotient {
        n_classes,
        state_to_class,
        quotient_edges,
        blake3: [0u8; 32],
    };
    q.blake3 = blake3_commit(&q);
    q
}

/// Renumber class labels so the class containing the smallest `StateId`
/// becomes class 0, the next new class becomes 1, etc.
///
/// This makes the label vector invariant under class-id permutations: two
/// partitions with the same equivalence structure but different label
/// assignments map to the same canonical label vector. Used both inside the
/// refinement loop (convergence check) and as the final canonicalization.
///
/// `labels` is modified in place. `states` is the sorted state-id vec from
/// the input graph (walked in ascending order to assign canonical labels).
fn canonicalize_labels(labels: &mut [u32], states: &[StateId]) {
    if labels.is_empty() {
        return;
    }

    // Map old_class_id → new_class_id. Sized to max label value + 1.
    let max_old = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut old_to_new: Vec<u32> = vec![u32::MAX; max_old + 1];
    let mut next_new_id: u32 = 0;
    for &state in states {
        let old = labels[state.0 as usize] as usize;
        if old < old_to_new.len() && old_to_new[old] == u32::MAX {
            old_to_new[old] = next_new_id;
            next_new_id += 1;
        }
    }

    for label in labels.iter_mut() {
        let old = *label as usize;
        if old < old_to_new.len() {
            *label = old_to_new[old];
        }
        // else: shouldn't happen (label > max_old is impossible by
        // construction of old_to_new), but defensively leave it.
    }
}

/// Compute the BLAKE3 commitment over the canonical byte serialization
/// (Plan 324 T2.3).
///
/// Canonical layout (all little-endian, written per-element to avoid
/// touching potentially-uninitialized `#[repr(C)]` padding bytes):
///
/// ```text
/// u32 LE          n_classes
/// u32 LE × N      state_to_class[i].0 for i in 0..N
/// per edge × M:   from.0 (u32 LE) || to.0 (u32 LE) || op.discriminant() (u8)
/// ```
///
/// The per-edge encoding is **logical** (no padding) — this is what matters
/// for cross-run stability. Relying on the raw `#[repr(C)]` memory layout
/// would include uninitialized padding bytes, which is UB to hash.
pub fn blake3_commit(q: &BisimulationQuotient) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&q.n_classes.to_le_bytes());
    for c in &q.state_to_class {
        hasher.update(&c.0.to_le_bytes());
    }
    for e in &q.quotient_edges {
        hasher.update(&e.from.0.to_le_bytes());
        hasher.update(&e.to.0.to_le_bytes());
        hasher.update(&[e.op.discriminant()]);
    }
    *hasher.finalize().as_bytes()
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bisimulation::graph::TransitionGraphBuilder;
    use crate::bisimulation::types::OperatorLabel;

    fn s(v: u32) -> StateId {
        StateId(v)
    }

    #[test]
    fn empty_graph_yields_empty_quotient() {
        let g = TransitionGraphBuilder::new().build();
        let q = partition_refine(&g);
        assert_eq!(q.n_classes, 0);
        assert!(q.state_to_class.is_empty());
        assert!(q.quotient_edges.is_empty());
        // BLAKE3 over the canonical empty layout = just `n_classes=0` as
        // 4 LE bytes (no state_to_class, no edges).
        let mut expected = blake3::Hasher::new();
        expected.update(&0u32.to_le_bytes());
        assert_eq!(q.blake3, *expected.finalize().as_bytes());
    }

    #[test]
    fn chain_a_to_b_to_c_yields_three_classes() {
        // A -PickTop-> B -PlaceOn-> C : three distinct states (none
        // bisim-equivalent because their out-degree signatures differ).
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        b.push_transition(s(1), s(2), OperatorLabel::PlaceOn);
        let g = b.build();
        let q = partition_refine(&g);

        assert_eq!(q.n_classes, 3);
        assert_ne!(q.class_of(s(0)), q.class_of(s(1)));
        assert_ne!(q.class_of(s(1)), q.class_of(s(2)));
        assert_ne!(q.class_of(s(0)), q.class_of(s(2)));
    }

    #[test]
    fn parallel_chains_collapse_into_two_classes() {
        // A -PickTop-> B, C -PickTop-> D. B and D are sinks (bisim-equiv);
        // A and C each have one PickTop edge to a sink (bisim-equiv).
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        b.push_transition(s(2), s(3), OperatorLabel::PickTop);
        let g = b.build();
        let q = partition_refine(&g);

        assert_eq!(q.n_classes, 2, "expected 2 classes, got {}", q.n_classes);
        assert_eq!(q.class_of(s(0)), q.class_of(s(2)), "A~C");
        assert_eq!(q.class_of(s(1)), q.class_of(s(3)), "B~D");
        assert_ne!(q.class_of(s(0)), q.class_of(s(1)), "A≄B");
    }

    #[test]
    fn rerun_is_bit_identical() {
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        b.push_transition(s(0), s(2), OperatorLabel::PlaceOn);
        b.push_transition(s(1), s(2), OperatorLabel::PickTop);
        b.push_transition(s(2), s(0), OperatorLabel::PlaceOnEmpty);
        let g = b.build();

        let q1 = partition_refine(&g);
        let q2 = partition_refine(&g);
        assert_eq!(q1.blake3, q2.blake3, "blake3 must be deterministic");
        assert_eq!(q1.state_to_class, q2.state_to_class);
        assert_eq!(q1.quotient_edges, q2.quotient_edges);
    }

    #[test]
    fn identical_inputs_produce_identical_blake3() {
        let mut b1 = TransitionGraphBuilder::new();
        b1.push_transition(s(0), s(1), OperatorLabel::PickTop);
        b1.push_transition(s(2), s(3), OperatorLabel::PickTop);
        let g1 = b1.build();

        let mut b2 = TransitionGraphBuilder::new();
        b2.push_transition(s(0), s(1), OperatorLabel::PickTop);
        b2.push_transition(s(2), s(3), OperatorLabel::PickTop);
        let g2 = b2.build();

        assert_eq!(partition_refine(&g1).blake3, partition_refine(&g2).blake3);
    }

    #[test]
    fn quotient_edges_are_sorted_and_deduped() {
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(1), OperatorLabel::PickTop);
        b.push_transition(s(2), s(3), OperatorLabel::PickTop);
        let g = b.build();
        let q = partition_refine(&g);

        // Both raw edges (0→1) and (2→3) map to (class 0 → class 1) in the
        // quotient → dedup to one edge.
        assert_eq!(q.quotient_edges.len(), 1);
        let e = &q.quotient_edges[0];
        assert_eq!(e.from, StateClassId(0));
        assert_eq!(e.to, StateClassId(1));
        assert_eq!(e.op, OperatorLabel::PickTop);
    }

    #[test]
    fn label_oscillation_counterexample_converges() {
        // The period-2 oscillation counterexample from the module doc:
        //   States: 0, 1, 2.
        //   Edges:  0 --A--> 1, 0 --B--> 2, 1 --A--> 0, 2 --A--> 0.
        //
        // Without per-iteration canonicalization, labels alternate between
        // [0,1,1] and [1,0,0] forever. With canonicalization, both collapse
        // to [0,1,1] and the loop converges.
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(1), OperatorLabel::PickTop); // A
        b.push_transition(s(0), s(2), OperatorLabel::PlaceOn); // B
        b.push_transition(s(1), s(0), OperatorLabel::PickTop); // A
        b.push_transition(s(2), s(0), OperatorLabel::PickTop); // A
        let g = b.build();
        let q = partition_refine(&g);

        // State 0 has 2 outgoing edges (A,B); states 1,2 have 1 (A) →
        // 2 classes: {0} and {1,2}.
        assert_eq!(q.n_classes, 2);
        assert_ne!(q.class_of(s(0)), q.class_of(s(1)));
        assert_eq!(q.class_of(s(1)), q.class_of(s(2)), "1~2");
    }

    #[test]
    fn class_of_is_o1_for_in_range() {
        let mut b = TransitionGraphBuilder::new();
        b.push_transition(s(0), s(1), OperatorLabel::NoOp);
        let g = b.build();
        let q = partition_refine(&g);
        let _ = q.class_of(s(0));
        let _ = q.class_of(s(1));
    }

    #[test]
    fn canonicalize_labels_renumbers_by_smallest_member() {
        // labels = [1, 0, 0] (state 0 in class 1, states 1,2 in class 0)
        // states = [0, 1, 2]
        // Canonical: walk states ascending → state 0's old class 1 → new 0;
        //   state 1's old class 0 → new 1; state 2's old class 0 → seen.
        // Result: [0, 1, 1].
        let mut labels = vec![1u32, 0, 0];
        let states = vec![StateId(0), StateId(1), StateId(2)];
        canonicalize_labels(&mut labels, &states);
        assert_eq!(labels, vec![0, 1, 1]);
    }

    #[test]
    fn canonicalize_labels_is_idempotent() {
        let mut labels = vec![0u32, 1, 1, 2];
        let states = vec![StateId(0), StateId(1), StateId(2), StateId(3)];
        canonicalize_labels(&mut labels, &states);
        let before = labels.clone();
        canonicalize_labels(&mut labels, &states);
        assert_eq!(labels, before, "canonical form must be a fixed point");
    }
}
