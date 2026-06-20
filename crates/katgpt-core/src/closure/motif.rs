//! `Motif` + `MotifMiner` — recurring subgraph discovery across recent PTGs.
//!
//! Implements the paper's §4.4 "Discovering Motifs" + §5.2 "wrapped motifs
//! become higher-order primitives" pattern, in a modelless form: a
//! **gSpan-lite** enumeration that walks each recent PTG and hashes every
//! connected subgraph of bounded size into a canonical structural key. The
//! same key observed across many task families ⇒ high PRI ⇒ the
//! [`crate::admit::MotifAdmitter`] may promote it to a composite primitive.
//!
//! **Canonicalization is structural only** — node kinds + edge operator kinds,
//! sorted. Ticks and per-node `blake3_in` commitments are intentionally
//! ignored: two PTGs that exercise the same primitive composition in different
//! orders or with different inputs contribute to the *same* motif. This is
//! the gSpan-lite approximation called out in Plan 290 — full gSpan
//! DFS-lexicographic canonicalization is out of scope.
//!
//! ## Complexity
//!
//! `O(K · N^4)` at `K = RING_BUFFER_K = 1024` traces and `N ≤ 10` nodes per
//! trace ≈ manageable in the warm tier (low-ms range). Rayon-par over K. No
//! locks held across the parallel section — each worker accumulates into a
//! thread-local `std::collections::HashMap` and the results are merged
//! serially afterwards.

use std::collections::HashMap;

use papaya::HashMap as PapayaMap;
use rayon::prelude::*;

use super::PrimitiveTransitionGraph;

/// Upper bound on the number of nodes in a discovered motif.
pub const MAX_MOTIF_NODES: u8 = 4;

/// Upper bound on the number of edges in a discovered motif.
pub const MAX_MOTIF_EDGES: u8 = 4;

/// Ring-buffer slot count for [`MotifMiner::recent_ptgs`]. `1024` is the
/// warm-tier default called out in Plan 290 (≈ `K` in the complexity formula).
pub const RING_BUFFER_K: usize = 1024;

/// Bounded set of `u32`s with a fixed-capacity backing array.
///
/// Used by [`Motif::task_family_ids`] — we only ever care about up to 16
/// task families for a motif (more than that and the PRI saturates anyway).
/// `insert` is no-op once full; duplicates are silently dropped.
#[derive(Clone, Debug)]
pub struct FixedU32Set<const N: usize>(
    /// Slots; `[u32; N]` with explicit `len`.
    [u32; N],
    /// Number of populated slots.
    u8,
);

impl<const N: usize> Default for FixedU32Set<N> {
    #[inline]
    fn default() -> Self {
        Self([0u32; N], 0)
    }
}

impl<const N: usize> FixedU32Set<N> {
    /// New empty set.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `v` if not present and capacity allows. Returns `true` if the
    /// set changed.
    #[inline]
    pub fn insert(&mut self, v: u32) -> bool {
        if self.contains(&v) {
            return false;
        }
        let n = self.1 as usize;
        if n < N {
            self.0[n] = v;
            self.1 += 1;
            true
        } else {
            false
        }
    }

    /// `true` iff `v` is present.
    #[inline]
    #[must_use]
    pub fn contains(&self, v: &u32) -> bool {
        let n = self.1 as usize;
        self.0[..n].iter().any(|x| x == v)
    }

    /// Number of populated slots.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.1 as usize
    }

    /// `true` iff no elements have been inserted.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.1 == 0
    }

    /// Iterate the populated slots.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        let n = self.1 as usize;
        self.0[..n].iter().copied()
    }
}

// Hand-rolled serde — fixed-size array + len byte. Avoids the
// `serde::Serialize` derive generating `Option`-shaped payloads and keeps the
// wire format stable at `4*N + 1` bytes.
impl<const N: usize> serde::Serialize for FixedU32Set<N> {
    #[inline]
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let n = self.1 as usize;
        let mut tup = s.serialize_tuple(1 + n)?;
        tup.serialize_element(&self.1)?;
        for v in &self.0[..n] {
            tup.serialize_element(v)?;
        }
        tup.end()
    }
}

impl<'de, const N: usize> serde::Deserialize<'de> for FixedU32Set<N> {
    #[inline]
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V<const N: usize>;
        impl<'de, const N: usize> serde::de::Visitor<'de> for V<N> {
            type Value = FixedU32Set<N>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a sequence of at most {} u32 values", N)
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut arr = [0u32; N];
                let mut len: u8 = 0;
                if let Some(first) = seq.next_element::<u8>()? {
                    len = first.min(N as u8);
                }
                for slot in arr.iter_mut().take(len as usize) {
                    *slot = seq.next_element()?.unwrap_or(0);
                }
                Ok(FixedU32Set::<N>(arr, len))
            }
        }
        d.deserialize_seq(V::<N>)
    }
}

/// A discovered motif — a recurring connected subgraph.
///
/// `subgraph_hash` is the canonical structural key (BLAKE3 over
/// `(sorted_node_kinds, sorted_edge_kinds)`). It is identical for any PTG
/// that exercises the same primitive composition regardless of ticks or input
/// commitments.
#[derive(Clone, Debug)]
pub struct Motif {
    /// BLAKE3 of `(sorted_node_kinds, sorted_edge_kinds)` — structural key.
    pub subgraph_hash: [u8; 32],
    /// Number of nodes in the motif (≤ [`MAX_MOTIF_NODES`]).
    pub node_count: u8,
    /// Number of edges in the motif (≤ [`MAX_MOTIF_EDGES`]).
    pub edge_count: u8,
    /// How many times this motif's canonical form was observed across all
    /// recent PTGs.
    pub occurrence_count: u32,
    /// Which distinct task families contributed observations.
    pub task_family_ids: FixedU32Set<16>,
}

impl Motif {
    /// Primitive Reuse Index: fraction of `total_task_families` that contain
    /// this motif.
    ///
    /// Clamped to `[0, 1]`. Drives the admission gate ([`crate::admit`]).
    #[inline]
    #[must_use]
    pub fn primitive_reuse_index(&self, total_task_families: u32) -> f32 {
        if total_task_families == 0 {
            return 0.0;
        }
        let r = self.task_family_ids.len() as f32 / total_task_families as f32;
        r.clamp(0.0, 1.0)
    }
}

serde_for_motif!(Motif);

/// Computes the canonical structural key for a subgraph induced by `nodes`
/// (their kinds) and `edges` (their operator kinds).
///
/// Returns a 32-byte BLAKE3 hash. Two subgraphs hash identically iff they have
/// the same multiset of node kinds and edge operator kinds (ticks/blake3_in
/// ignored).
///
/// Inputs are bounded by [`MAX_MOTIF_NODES`] / [`MAX_MOTIF_EDGES`] (= 4 each),
/// so the sort scratch *and* the postcard wire-form both live on the stack —
/// zero heap allocation per call (the previous version cloned into two heap
/// `Vec`s for sort scratch + two more via `postcard::to_allocvec`).
///
/// The postcard-compatible varint encoding is hand-rolled to match
/// `postcard::to_allocvec(&n_buf[..n_len])` byte-for-byte. Verified by the
/// `canonical_hash_matches_postcard` test below. Wire format:
/// - `&[u32]` (seq):  `varint(len) || varint(u32) || varint(u32) || ...`
/// - `&[u8]` (bytes): `varint(len) || raw bytes`  (serde `serialize_bytes`)
#[inline]
fn canonical_hash(node_kinds: &[u32], edge_kinds: &[u8]) -> [u8; 32] {
    debug_assert!(
        node_kinds.len() <= MAX_MOTIF_NODES as usize,
        "canonical_hash node_kinds overflow: {} > {}",
        node_kinds.len(),
        MAX_MOTIF_NODES
    );
    debug_assert!(
        edge_kinds.len() <= MAX_MOTIF_EDGES as usize,
        "canonical_hash edge_kinds overflow: {} > {}",
        edge_kinds.len(),
        MAX_MOTIF_EDGES
    );

    // Stack-backed sort scratch — sized to the motif bounds.
    let mut n_buf = [0u32; MAX_MOTIF_NODES as usize];
    let mut e_buf = [0u8; MAX_MOTIF_EDGES as usize];
    let n_len = node_kinds.len();
    let e_len = edge_kinds.len();
    n_buf[..n_len].copy_from_slice(node_kinds);
    e_buf[..e_len].copy_from_slice(edge_kinds);
    n_buf[..n_len].sort_unstable();
    e_buf[..e_len].sort_unstable();

    // Postcard-compatible varint encoding into stack buffers — eliminates the
    // two heap allocations from `postcard::to_allocvec`.
    //
    // `n_enc` bound: 1 byte length prefix + N varint-encoded u32s (≤ 5 bytes
    // each via LEB128) = 1 + 4*5 = 21 bytes.
    // `e_enc` bound: 1 byte length prefix + N raw bytes = 1 + 4 = 5 bytes
    //   (postcard's `serialize_bytes` shortcut writes raw bytes, not varint-per-element).
    let mut n_enc: [u8; 1 + MAX_MOTIF_NODES as usize * 5] = [0; 1 + 4 * 5];
    let mut e_enc: [u8; 1 + MAX_MOTIF_EDGES as usize] = [0; 1 + 4];
    let mut n_pos = write_varint_u32(&mut n_enc, n_len as u32);
    for &v in &n_buf[..n_len] {
        n_pos += write_varint_u32(&mut n_enc[n_pos..], v);
    }
    let mut e_pos = write_varint_u32(&mut e_enc, e_len as u32);
    e_enc[e_pos..e_pos + e_len].copy_from_slice(&e_buf[..e_len]);
    e_pos += e_len;

    let mut hasher = blake3::Hasher::new();
    hasher.update(&n_enc[..n_pos]);
    hasher.update(&e_enc[..e_pos]);
    hasher.finalize().into()
}

/// Encode `v` as a postcard-compatible LEB128 varint into `out`. Returns the
/// number of bytes written. Mirrors `postcard::varint::varint_u32` byte for
/// byte: LSB first, MSB of each byte is the continuation bit (1 = more bytes).
#[inline]
fn write_varint_u32(out: &mut [u8], mut v: u32) -> usize {
    let mut i = 0;
    loop {
        out[i] = v as u8;
        if v < 0x80 {
            return i + 1;
        }
        out[i] |= 0x80;
        v >>= 7;
        i += 1;
    }
}

/// `MotifMiner` — observes PTGs and runs `mine_batch()` at sleep-cycle
/// boundaries (Plan 107 AutoDreamer consolidation tick).
///
/// Backed by:
/// - A FIFO ring buffer of the last `RING_BUFFER_K` observed PTGs.
/// - A `papaya` lock-free index from motif hash → aggregated [`Motif`].
///
/// `observe()` is single-threaded by design (no `&self` concurrent callers).
/// `mine_batch()` parallelizes the enumeration work via rayon but merges the
/// per-thread accumulators back into the shared index serially to avoid
/// lock contention (per AGENTS.md "Parallelism / Rayon": no locks in
/// parallel sections).
pub struct MotifMiner {
    /// Ring buffer of recent PTGs (FIFO eviction).
    pub recent_ptgs: Vec<PrimitiveTransitionGraph>,
    /// Shared motif index. Written by `mine_batch()` only.
    pub motif_index: PapayaMap<[u8; 32], Motif>,
}

impl MotifMiner {
    /// New empty miner.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            recent_ptgs: Vec::with_capacity(RING_BUFFER_K),
            motif_index: PapayaMap::new(),
        }
    }

    /// Push a PTG; evict the oldest if we've exceeded `RING_BUFFER_K` (FIFO).
    #[inline]
    pub fn observe(&mut self, ptg: PrimitiveTransitionGraph) {
        if self.recent_ptgs.len() >= RING_BUFFER_K {
            // FIFO eviction — drop the front.
            self.recent_ptgs.remove(0);
        }
        self.recent_ptgs.push(ptg);
    }

    /// Enumerate all connected subgraphs of bounded size across all recent
    /// PTGs and merge them into the shared [`MotifMiner::motif_index`].
    ///
    /// Parallelized via rayon over PTGs; each thread accumulates into a local
    /// `std::collections::HashMap` and the merge into `motif_index` happens
    /// serially afterwards. This avoids `papaya` lock contention on the hot
    /// enumeration path.
    ///
    /// # Complexity
    ///
    /// `O(K · N^4)` at K=`RING_BUFFER_K`=1024 traces and `N≤10` nodes/trace
    /// ≈ low-ms range on the warm tier.
    #[inline]
    pub fn mine_batch(&self) -> Vec<Motif> {
        // Each PTG produces a local map (hash → (Motif accumulator)).
        let local_maps: Vec<HashMap<[u8; 32], Motif>> = self
            .recent_ptgs
            .par_iter()
            .map(enumerate_motifs_for_ptg)
            .collect();

        // Serial merge into the shared papaya index.
        let guard = self.motif_index.pin();
        for local in local_maps {
            for (hash, motif) in local {
                let existing_opt: Option<Motif> = guard.get(&hash).cloned();
                match existing_opt {
                    Some(mut existing) => {
                        // Merge: add occurrence counts, union task families.
                        existing.occurrence_count += motif.occurrence_count;
                        for fam in motif.task_family_ids.iter() {
                            existing.task_family_ids.insert(fam);
                        }
                        guard.insert(hash, existing);
                    }
                    None => {
                        guard.insert(hash, motif);
                    }
                }
            }
        }

        // Snapshot the index for callers.
        guard.iter().map(|(_, v)| v.clone()).collect()
    }
}

impl Default for MotifMiner {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Enumerate the **multiset of canonical subgraph hashes** for a single PTG.
///
/// Public so other modules (e.g. [`crate::metrics::compute_tar_score`]) can
/// reuse the same enumeration without re-running the accumulator. Returns
/// one hash per observed subgraph occurrence (so duplicates are intentional
/// — they encode multiplicity for Jaccard-multiset comparisons).
///
/// Covers 1-, 2-, 3-, and 4-node chain motifs (greedy DFS along edges).
#[inline]
#[must_use]
pub fn enumerate_subgraph_hashes(ptg: &PrimitiveTransitionGraph) -> Vec<[u8; 32]> {
    let mut acc: HashMap<[u8; 32], Motif> = HashMap::new();
    enumerate_motifs_into(ptg, &mut acc);
    // Emit one hash per occurrence count.
    let mut out: Vec<[u8; 32]> = Vec::with_capacity(8);
    for (hash, m) in acc {
        for _ in 0..m.occurrence_count {
            out.push(hash);
        }
    }
    out
}

/// Enumerate all connected subgraphs of bounded size within a single PTG,
/// hash each canonical form, and accumulate counts + task family.
///
/// The "subgraph" we extract is the multiset of node kinds + edge operator
/// kinds — we do not preserve the topology itself (gSpan-lite). This trades
/// precision for tractability: O(N^4) instead of exponential.
fn enumerate_motifs_for_ptg(ptg: &PrimitiveTransitionGraph) -> HashMap<[u8; 32], Motif> {
    let mut out: HashMap<[u8; 32], Motif> = HashMap::new();
    enumerate_motifs_into(ptg, &mut out);
    out
}

/// Shared inner routine — fills `out` with motif entries for `ptg`.
/// Split out from [`enumerate_motifs_for_ptg`] so [`enumerate_subgraph_hashes`]
/// can reuse the same logic without round-tripping through the map shape.
fn enumerate_motifs_into(
    ptg: &PrimitiveTransitionGraph,
    out: &mut HashMap<[u8; 32], Motif>,
) {
    let n = ptg.nodes.len();
    if n == 0 {
        return;
    }

    // Per-node primitive kinds (as u32 wire form) — referenced by index.
    let node_kinds: Vec<u32> = ptg.nodes.iter().map(|x| x.primitive.to_u32()).collect();
    // Per-edge (op kind as u8, from_idx, to_idx).
    let edges: Vec<(u8, usize, usize)> = ptg
        .edges
        .iter()
        .map(|e| (e.op as u8, e.from as usize, e.to as usize))
        .collect();

    // Single-node "motifs" — every primitive observed becomes a 1-node motif.
    // (Useful for PRI: which primitives get reused.)
    for &k in &node_kinds {
        let hash = canonical_hash(&[k], &[]);
        let entry = out.entry(hash).or_insert_with(|| Motif {
            subgraph_hash: hash,
            node_count: 1,
            edge_count: 0,
            occurrence_count: 0,
            task_family_ids: FixedU32Set::<16>::new(),
        });
        entry.occurrence_count += 1;
        entry.task_family_ids.insert(ptg.task_family_id);
    }

    // 2-node motifs: every directed edge.
    for &(op, from, to) in &edges {
        let kinds = [node_kinds[from], node_kinds[to]];
        let hash = canonical_hash(&kinds, &[op]);
        let entry = out.entry(hash).or_insert_with(|| Motif {
            subgraph_hash: hash,
            node_count: 2,
            edge_count: 1,
            occurrence_count: 0,
            task_family_ids: FixedU32Set::<16>::new(),
        });
        entry.occurrence_count += 1;
        entry.task_family_ids.insert(ptg.task_family_id);
    }

    // 3-node / 4-node chain motifs: walk along edges up to depth 2 / 3.
    // Greedy: start from each node, BFS through up to 3 edges.
    // (This is an approximation — not all connected subgraphs are found. The
    // canonical hash absorbs ordering, so different walks over the same shape
    // collapse to one entry.)
    for start in 0..n {
        enumerate_chains(start, &node_kinds, &edges, 3, ptg.task_family_id, out);
    }
}

/// Greedy BFS-chain enumeration: from `start`, walk along edges, accumulating
/// (node_kind, edge_op) multisets up to `max_depth` hops. Hashes each prefix
/// ≥ 2 nodes as a motif.
fn enumerate_chains(
    start: usize,
    node_kinds: &[u32],
    edges: &[(u8, usize, usize)],
    max_depth: u8,
    task_family_id: u32,
    out: &mut HashMap<[u8; 32], Motif>,
) {
    // DFS stack: (current_node, visited_set_as_path, edge_ops_along_path).
    // Visited is tracked by the path Vec itself.
    let mut stack: Vec<(usize, Vec<usize>, Vec<u8>)> = Vec::with_capacity(8);
    stack.push((start, vec![start], Vec::new()));

    while let Some((node, path, ops)) = stack.pop() {
        if path.len() >= 2 && path.len() as u8 <= MAX_MOTIF_NODES {
            // Hash this prefix as a motif.
            let kinds: Vec<u32> = path.iter().map(|&i| node_kinds[i]).collect();
            let hash = canonical_hash(&kinds, &ops);
            let entry = out.entry(hash).or_insert_with(|| Motif {
                subgraph_hash: hash,
                node_count: path.len() as u8,
                edge_count: ops.len() as u8,
                occurrence_count: 0,
                task_family_ids: FixedU32Set::<16>::new(),
            });
            entry.occurrence_count += 1;
            entry.task_family_ids.insert(task_family_id);
        }
        if (path.len() as u8) >= MAX_MOTIF_NODES {
            continue;
        }
        if (ops.len() as u8) >= MAX_MOTIF_EDGES {
            continue;
        }
        // Expand: try every outgoing edge from `node` whose target is not on path.
        for &(op, from, to) in edges {
            if from != node {
                continue;
            }
            if path.contains(&to) {
                continue;
            }
            let mut new_path = path.clone();
            new_path.push(to);
            let mut new_ops = ops.clone();
            new_ops.push(op);
            if (new_path.len() as u8) <= max_depth + 1 {
                stack.push((to, new_path, new_ops));
            }
        }
    }
}

// Hand-rolled serde for Motif — keeps the wire form compact and stable.
macro_rules! serde_for_motif {
    ($ty:ty) => {
        impl serde::Serialize for $ty {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeStruct;
                let mut st = s.serialize_struct("Motif", 5)?;
                st.serialize_field("subgraph_hash", &self.subgraph_hash.to_vec())?;
                st.serialize_field("node_count", &self.node_count)?;
                st.serialize_field("edge_count", &self.edge_count)?;
                st.serialize_field("occurrence_count", &self.occurrence_count)?;
                st.serialize_field("task_family_ids", &self.task_family_ids)?;
                st.end()
            }
        }
        impl<'de> serde::Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                #[derive(serde::Deserialize)]
                #[serde(default)]
                struct Shadow {
                    subgraph_hash: Vec<u8>,
                    node_count: u8,
                    edge_count: u8,
                    occurrence_count: u32,
                    task_family_ids: FixedU32Set<16>,
                }
                impl Default for Shadow {
                    fn default() -> Self {
                        Self {
                            subgraph_hash: Vec::new(),
                            node_count: 0,
                            edge_count: 0,
                            occurrence_count: 0,
                            task_family_ids: FixedU32Set::<16>::new(),
                        }
                    }
                }
                let sh = Shadow::deserialize(d)?;
                let mut arr = [0u8; 32];
                let len = sh.subgraph_hash.len().min(32);
                arr[..len].copy_from_slice(&sh.subgraph_hash[..len]);
                Ok(Motif {
                    subgraph_hash: arr,
                    node_count: sh.node_count,
                    edge_count: sh.edge_count,
                    occurrence_count: sh.occurrence_count,
                    task_family_ids: sh.task_family_ids,
                })
            }
        }
    };
}
pub(crate) use serde_for_motif;

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{OperatorKind, PrimitiveKind, PtgRecorder};

    fn make_search_verify_branch_ptg(task_family_id: u32) -> PrimitiveTransitionGraph {
        // Search → Verify → Branch
        let mut rec = PtgRecorder::new(task_family_id);
        let a = rec.enter(PrimitiveKind::UserDefined(0), 0, [0u8; 32]); // Search
        let b = rec.enter(PrimitiveKind::UserDefined(1), 1, [1u8; 32]); // Verify
        let c = rec.enter(PrimitiveKind::UserDefined(2), 2, [2u8; 32]); // Branch
        rec.exit(a, b, OperatorKind::Sequence);
        rec.exit(b, c, OperatorKind::Branch);
        rec.finish()
    }

    #[test]
    fn fixed_u32_set_basic() {
        let mut s = FixedU32Set::<4>::new();
        assert!(s.is_empty());
        assert!(s.insert(1));
        assert!(s.insert(2));
        assert!(!s.insert(1)); // dup
        assert_eq!(s.len(), 2);
        assert!(s.contains(&1));
        assert!(!s.contains(&99));
        // Cap
        assert!(s.insert(3));
        assert!(s.insert(4));
        assert!(!s.insert(5)); // full
        assert_eq!(s.len(), 4);
        let collected: Vec<u32> = s.iter().collect();
        assert_eq!(collected.len(), 4);
    }

    #[test]
    fn primitive_reuse_index_clamps() {
        let mut m = Motif {
            subgraph_hash: [0u8; 32],
            node_count: 1,
            edge_count: 0,
            occurrence_count: 0,
            task_family_ids: FixedU32Set::<16>::new(),
        };
        assert!((m.primitive_reuse_index(0) - 0.0).abs() < 1e-6);
        for v in 0..10 {
            m.task_family_ids.insert(v);
        }
        // 10 inserted but capacity is 16 — count is 10
        let pri = m.primitive_reuse_index(20);
        assert!((pri - 0.5).abs() < 1e-6, "pri={pri}");
        // Clamp >1
        let pri2 = m.primitive_reuse_index(5);
        assert!((pri2 - 1.0).abs() < 1e-6, "pri2={pri2}");
    }

    /// T2.6 — 100 PTGs across 3 task families containing the same motif.
    #[test]
    fn mine_batch_finds_motif_across_families() {
        let mut miner = MotifMiner::new();
        for i in 0..100u32 {
            let fam = i % 3; // 3 distinct families
            miner.observe(make_search_verify_branch_ptg(fam));
        }
        let motifs = miner.mine_batch();
        // Find a 3-node motif (Search→Verify→Branch).
        let found = motifs
            .iter()
            .find(|m| m.node_count == 3)
            .expect("3-node motif should be discovered");
        assert!(found.occurrence_count >= 100, "occ={}", found.occurrence_count);
        assert_eq!(
            found.task_family_ids.len(),
            3,
            "should span all 3 task families"
        );
    }

    /// T2.7 — demotion: motif present in only 1 family but with high
    /// occurrence count ⇒ low PRI.
    #[test]
    fn motif_in_one_family_has_low_pri() {
        let mut miner = MotifMiner::new();
        // 100 PTGs all from the same task family.
        for _ in 0..100u32 {
            miner.observe(make_search_verify_branch_ptg(0));
        }
        let motifs = miner.mine_batch();
        let found = motifs
            .iter()
            .find(|m| m.node_count == 3)
            .expect("3-node motif present");
        // PRI over 3 *possible* task families ⇒ 1/3.
        let pri = found.primitive_reuse_index(3);
        assert!(
            (pri - (1.0 / 3.0)).abs() < 1e-6,
            "pri should be 1/3, got {pri}"
        );
    }

    #[test]
    fn ring_buffer_eviction_is_fifo() {
        let mut miner = MotifMiner::new();
        // Override capacity by stuffing more than RING_BUFFER_K.
        for i in 0..(RING_BUFFER_K + 5) as u32 {
            let mut rec = PtgRecorder::new(i);
            let _ = rec.enter(PrimitiveKind::UserDefined(i), i, [i as u8; 32]);
            miner.observe(rec.finish());
        }
        assert_eq!(miner.recent_ptgs.len(), RING_BUFFER_K);
        // Oldest 5 evicted — the front should now be task_family_id=5.
        assert_eq!(miner.recent_ptgs[0].task_family_id, 5);
        // Latest is the last one pushed.
        let last_id = (RING_BUFFER_K + 4) as u32;
        assert_eq!(miner.recent_ptgs.last().unwrap().task_family_id, last_id);
    }

    /// The hand-rolled varint encoder in `canonical_hash` must produce
    /// byte-for-byte identical BLAKE3 output to the previous
    /// `postcard::to_allocvec`-based path. Covers edge cases: empty slices,
    /// single elements, max-bound sizes, and large u32 values that span the
    /// full 5-byte varint range.
    #[test]
    fn canonical_hash_matches_postcard() {
        let cases: &[(Vec<u32>, Vec<u8>)] = &[
            (vec![], vec![]),
            (vec![1], vec![]),
            (vec![], vec![5]),
            (vec![1, 2], vec![5]),
            (vec![3, 1, 2], vec![10, 20]),
            (vec![100, 200, 300, 400], vec![100, 200, 255, 0]),
            (vec![], vec![1, 2, 3, 4]),
            // Large u32 values that need multi-byte varint encoding.
            (vec![0x1234_5678], vec![0xff]),
            (vec![u32::MAX], vec![0xff, 0x80, 0x40, 0x20]),
            (vec![0x4000, 0x200000, 0x80, 0x40], vec![0]),
        ];
        for (n, e) in cases {
            let hash_new = canonical_hash(n, e);
            // Reference: pre-existing postcard-based path. Sort independently
            // to mirror what the old `canonical_hash` did before hashing.
            let mut n_sorted = n.clone();
            let mut e_sorted = e.clone();
            n_sorted.sort_unstable();
            e_sorted.sort_unstable();
            let mut hasher = blake3::Hasher::new();
            hasher.update(
                &postcard::to_allocvec(&n_sorted[..]).unwrap_or_default(),
            );
            hasher.update(
                &postcard::to_allocvec(&e_sorted[..]).unwrap_or_default(),
            );
            let hash_ref: [u8; 32] = hasher.finalize().into();
            assert_eq!(
                hash_new, hash_ref,
                "canonical_hash drift for n={n:?} e={e:?}\n\
                 new: {hash_new:?}\n\
                 ref: {hash_ref:?}",
            );
        }
    }
}
