//! The expression DAG — the algorithmic heart (Research 607 §1).
//!
//! Nodes: `Lookup` leaves (one per referenced `(dimension, container)`
//! utilization, SHARED across specs; one per object for the moved flag) +
//! `Sum` / `Max` internal nodes + `Affine` (`k*x + c`, exact i64) +
//! `Const`. Every node caches its current `i64` value. The leaf-affectance
//! view: a candidate move touches a handful of leaves; recompute runs
//! bottom-up over REACHED nodes only (delta evaluation).
//!
//! Exactness note (why no segment tree): Meta's production solver needs a
//! segment tree for incremental `Sum` because FLOAT accumulation drifts
//! (1e-3-epsilon per apply × 1e4 moves). Integer sums are exact by
//! construction — a plain recompose-over-changed-children is bit-identical
//! to a full recompute (asserted by the delta-property test), so the
//! simpler mechanism is the correct one here.
//!
//! `Max` children are recomputed by scan. The paper's value-sorted child
//! list (O(changed) updates) pays off at 1e5-scale child counts; Phase 1
//! target sizes (containers ≤ ~10⁴ ⇒ child counts ≤ ~10⁴) sit inside the
//! regime where the branch-free scan is simpler and fast enough — noted
//! as the Phase 2 optimization if a consumer needs it.

/// Node handle (index into the arena; insertion order IS topological order
/// because builders always create children before parents).
pub type NodeId = u32;

/// One node in the arena. Children of `Sum`/`Max` live in a flat CSR slab
/// (no per-node allocation).
#[derive(Debug, Clone)]
pub enum ExprNode {
    /// Utilization leaf: value = Σ demands of objects currently assigned
    /// to `container` in `dim`. `slot` indexes the flat
    /// `dim * num_containers + container` space (over referenced dims).
    LeafUtil {
        slot: u32,
    },
    /// Moved-flag leaf: value = 1 iff object `object`'s container differs
    /// from its initial container, else 0.
    LeafMoved {
        object: u32,
    },
    /// Placement leaf (Phase 2): value = 1 iff object `object` is currently
    /// assigned to `container`, else 0. One per distinct `(object,
    /// container)` the Phase 2 specs read (`AssignmentAffinities` entries
    /// and `GroupCount` member×container pairs) — shared across specs, the
    /// util-leaf sharing discipline. Delta construction lives solver-side
    /// (the leaf changes only when the object's container changes).
    LeafAt {
        object: u32,
        container: u32,
    },
    Const {
        value: i64,
    },
    Sum,
    Max,
    Affine {
        k: i64,
        c: i64,
        child: NodeId,
    },
}

/// The DAG arena + connectivity (parents CSR for upward delta propagation).
#[derive(Debug, Clone)]
pub struct Dag {
    nodes: Vec<ExprNode>,
    /// CSR child slab: children of node n are
    /// `child_slab[child_off[n]..child_off[n+1]]`.
    child_slab: Vec<NodeId>,
    child_off: Vec<u32>,
    /// CSR parent slab (built once; upward BFS during delta evaluation).
    parent_slab: Vec<NodeId>,
    parent_off: Vec<u32>,
    /// The folded-objective root.
    pub root: NodeId,
}

/// Scratch for delta evaluation — allocated ONCE per solver and reused
/// (G4: the evaluate hot loop allocates nothing).
#[derive(Debug)]
pub struct DeltaScratch {
    /// Generation stamps so dirty marks don't need clearing between
    /// evaluations. u32 wraps after ~4.29e9 evaluations — the default
    /// eval budget (1e7) is 400× below it; a wrap would only alias a
    /// stale mark, so the guard is `max_evaluations < 2^31` (documented).
    stamp: Vec<u32>,
    generation: u32,
    /// Dirty (reached) nodes for the in-flight evaluation, sorted
    /// ascending (= topological order).
    dirty: Vec<NodeId>,
    /// Candidate values for dirty nodes (sparse overlay indexed by NodeId).
    new_vals: Vec<i64>,
}

impl DeltaScratch {
    /// The candidate root value from the most recent `eval_delta` call.
    pub fn last_new_root(&self, root: NodeId) -> i64 {
        self.new_vals[root as usize]
    }

    /// Was `node` reached (recomputed) by the most recent `eval_delta`?
    /// The `explain` surface reads this to pick overlay-vs-committed
    /// values for nodes the candidate evaluation did not touch.
    pub fn is_reached(&self, node: NodeId) -> bool {
        self.stamp[node as usize] == self.generation
    }

    /// The candidate value of a REACHED node from the most recent
    /// `eval_delta` (undefined for unreached nodes — pair with
    /// [`Self::is_reached`]).
    pub fn new_value(&self, node: NodeId) -> i64 {
        self.new_vals[node as usize]
    }
}

impl Dag {
    /// Empty builder (root set by `finish`).
    pub fn builder() -> DagBuilder {
        DagBuilder::default()
    }

    pub fn node(&self, id: NodeId) -> &ExprNode {
        &self.nodes[id as usize]
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn children(&self, id: NodeId) -> &[NodeId] {
        let s = self.child_off[id as usize] as usize;
        let e = self.child_off[id as usize + 1] as usize;
        &self.child_slab[s..e]
    }

    pub fn parents(&self, id: NodeId) -> &[NodeId] {
        let s = self.parent_off[id as usize] as usize;
        let e = self.parent_off[id as usize + 1] as usize;
        &self.parent_slab[s..e]
    }

    /// Scratch sized for this DAG (all buffers pre-sized to their worst
    /// case so the evaluation hot loop never grows them — G4).
    pub fn scratch(&self) -> DeltaScratch {
        DeltaScratch {
            stamp: vec![0; self.nodes.len()],
            generation: 0,
            dirty: Vec::with_capacity(self.nodes.len()),
            new_vals: vec![0; self.nodes.len()],
        }
    }

    /// Full topological recompute of `values` from the leaves up. The
    /// caller seeds leaf values first (leaves are the lowest ids — this
    /// pass starts at the first internal node). O(nodes).
    pub fn eval_full(&self, values: &mut [i64]) {
        for id in 0..self.nodes.len() {
            let v = match &self.nodes[id] {
                ExprNode::LeafUtil { .. }
                | ExprNode::LeafMoved { .. }
                | ExprNode::LeafAt { .. }
                | ExprNode::Const { .. } => values[id],
                ExprNode::Sum => {
                    let mut sum = 0i64;
                    for &c in self.children(id as NodeId) {
                        sum += values[c as usize];
                    }
                    sum
                }
                ExprNode::Max => {
                    let mut best = i64::MIN;
                    for &c in self.children(id as NodeId) {
                        let v = values[c as usize];
                        if v > best {
                            best = v;
                        }
                    }
                    best
                }
                ExprNode::Affine { k, c, child } => {
                    k.wrapping_mul(values[*child as usize]).wrapping_add(*c)
                }
            };
            values[id] = v;
        }
    }

    /// Candidate evaluation (const): apply `leaf_deltas` (relative to the
    /// CURRENT committed values) and return the would-be root value.
    /// Nothing is committed — `commit` does that. `leaf_deltas` must not
    /// name the same leaf twice (debug-asserted).
    pub fn eval_delta(
        &self,
        values: &[i64],
        leaf_deltas: &[(NodeId, i64)],
        scratch: &mut DeltaScratch,
    ) -> i64 {
        scratch.generation = scratch.generation.wrapping_add(1);
        if scratch.generation == 0 {
            // Wrapped: clear stamps completely so no stale mark can alias.
            scratch.stamp.iter_mut().for_each(|s| *s = 0);
            scratch.generation = 1;
        }
        scratch.dirty.clear();
        let current_gen = scratch.generation;
        // Seed: touched leaves with their new values.
        for &(leaf, delta) in leaf_deltas {
            debug_assert!(matches!(
                self.node(leaf),
                ExprNode::LeafUtil { .. } | ExprNode::LeafMoved { .. } | ExprNode::LeafAt { .. }
            ));
            debug_assert!(
                scratch.stamp[leaf as usize] != current_gen,
                "duplicate leaf delta"
            );
            scratch.stamp[leaf as usize] = current_gen;
            scratch.dirty.push(leaf);
            scratch.new_vals[leaf as usize] = values[leaf as usize].wrapping_add(delta);
        }
        // Upward BFS over parents: collect every reached node exactly once.
        let mut i = 0;
        while i < scratch.dirty.len() {
            let n = scratch.dirty[i];
            for &p in self.parents(n) {
                if scratch.stamp[p as usize] != current_gen {
                    scratch.stamp[p as usize] = current_gen;
                    scratch.dirty.push(p);
                }
            }
            i += 1;
        }
        // Ascending node id == topological order (children are always
        // created before parents), so an unstable sort with the unique id
        // key is fully deterministic.
        scratch.dirty.sort_unstable_by_key(|&n| n);
        // Bottom-up recompute over reached nodes only.
        for &n in &scratch.dirty {
            let v = match self.node(n) {
                ExprNode::LeafUtil { .. }
                | ExprNode::LeafMoved { .. }
                | ExprNode::LeafAt { .. } => scratch.new_vals[n as usize],
                ExprNode::Const { value } => *value,
                ExprNode::Sum => {
                    let mut sum = 0i64;
                    for &c in self.children(n) {
                        sum += self.child_val(c, values, scratch);
                    }
                    sum
                }
                ExprNode::Max => {
                    let mut best = i64::MIN;
                    for &c in self.children(n) {
                        let v = self.child_val(c, values, scratch);
                        if v > best {
                            best = v;
                        }
                    }
                    best
                }
                ExprNode::Affine { k, c, child } => k
                    .wrapping_mul(self.child_val(*child, values, scratch))
                    .wrapping_add(*c),
            };
            scratch.new_vals[n as usize] = v;
        }
        if scratch.dirty.is_empty() {
            // No leaf touched: the root keeps its committed value (the
            // overlay would be stale).
            return values[self.root as usize];
        }
        scratch.new_vals[self.root as usize]
    }

    #[inline]
    fn child_val(&self, c: NodeId, values: &[i64], scratch: &DeltaScratch) -> i64 {
        // A dirty child was recomputed earlier in this pass (topological
        // order); a clean child keeps its committed value.
        if scratch.stamp[c as usize] == scratch.generation {
            scratch.new_vals[c as usize]
        } else {
            values[c as usize]
        }
    }

    /// Commit a completed evaluation: copy the overlay into `values`.
    /// Must follow the `eval_delta` call that produced `scratch`'s current
    /// state, with no other evaluation in between.
    pub fn commit(&self, values: &mut [i64], scratch: &DeltaScratch) {
        for &n in &scratch.dirty {
            values[n as usize] = scratch.new_vals[n as usize];
        }
    }
}

/// Incremental DAG builder. Nodes are pushed children-first by the spec
/// compiler, which makes insertion order topological order.
#[derive(Default)]
pub struct DagBuilder {
    nodes: Vec<ExprNode>,
    child_slab: Vec<NodeId>,
    child_off: Vec<u32>,
}

impl DagBuilder {
    pub fn push_leaf_util(&mut self, slot: u32) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(ExprNode::LeafUtil { slot });
        self.child_off.push(self.child_slab.len() as u32);
        id
    }

    pub fn push_leaf_moved(&mut self, object: u32) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(ExprNode::LeafMoved { object });
        self.child_off.push(self.child_slab.len() as u32);
        id
    }

    /// Phase 2 placement leaf: `1` iff `object` sits in `container`.
    pub fn push_leaf_at(&mut self, object: u32, container: u32) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(ExprNode::LeafAt { object, container });
        self.child_off.push(self.child_slab.len() as u32);
        id
    }

    pub fn push_const(&mut self, value: i64) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(ExprNode::Const { value });
        self.child_off.push(self.child_slab.len() as u32);
        id
    }

    pub fn push_sum(&mut self, children: &[NodeId]) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(ExprNode::Sum);
        self.child_off.push(self.child_slab.len() as u32);
        self.child_slab.extend_from_slice(children);
        id
    }

    pub fn push_max(&mut self, children: &[NodeId]) -> NodeId {
        debug_assert!(!children.is_empty(), "Max of nothing is undefined");
        let id = self.nodes.len() as NodeId;
        self.nodes.push(ExprNode::Max);
        self.child_off.push(self.child_slab.len() as u32);
        self.child_slab.extend_from_slice(children);
        id
    }

    pub fn push_affine(&mut self, k: i64, c: i64, child: NodeId) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(ExprNode::Affine { k, c, child });
        self.child_off.push(self.child_slab.len() as u32);
        self.child_slab.push(child);
        id
    }

    /// Finalize with the root; builds the parent CSR.
    pub fn finish(mut self, root: NodeId) -> Dag {
        self.child_off.push(self.child_slab.len() as u32);
        let n = self.nodes.len();
        // Count incoming edges per child (slab edges only — Affine's child
        // rides the same slab, so children()/parents() stay uniform).
        let mut parent_off = vec![0u32; n + 1];
        for &c in &self.child_slab {
            parent_off[c as usize + 1] += 1;
        }
        for i in 0..n {
            parent_off[i + 1] += parent_off[i];
        }
        let mut parent_slab = vec![0u32; parent_off[n] as usize];
        let mut cursor = parent_off.clone();
        for id in 0..n {
            let kids =
                &self.child_slab[self.child_off[id] as usize..self.child_off[id + 1] as usize];
            for &c in kids {
                let at = cursor[c as usize] as usize;
                parent_slab[at] = id as NodeId;
                cursor[c as usize] += 1;
            }
        }
        Dag {
            nodes: self.nodes,
            child_slab: self.child_slab,
            child_off: self.child_off,
            parent_slab,
            parent_off,
            root,
        }
    }
}
