//! RadixPrefixTree — the RadixAttention index primitive (Issue 771).
//!
//! A mutable radix tree over token sequences at **page granularity**: node
//! spans are whole token chunks, and each chunk carries one page index per
//! KV layer. The tree is the *index half* of SGLang's RadixAttention — the
//! page pool (allocation, ref-counting, CoW) stays in the caller's paged KV
//! cache ([`katgpt_transformer::PagedKVCache`] via its chunk-page seam). The
//! tree owns **indices, never buffers** — shared pages are never moved, so
//! captured CUDA graphs that bake buffer addresses stay valid (the vLLM
//! capture×prefix-cache corruption class is structurally impossible here).
//!
//! # Serving flow (the contract the API is shaped around)
//!
//! ```text
//! request → match_prefix(tokens)          // locks terminal, zero-alloc
//!         → path_pages_into(hit, &mut out) // chunk-major page indices
//!         → pool.adopt_chunk_pages(seq, …) // refcount++ (request hold)
//!         → prefill only tokens[matched*page_tokens..]
//!         → pool.retain_chunk_pages(…)     // refcount++ (tree hold, NEW chunks only)
//!         → tree.insert(tokens, tables)    // index the request
//!         → tree.unlock(&hit)              // drop the match lock
//! … memory pressure …
//!         → tree.evict_to_budget(|pages| pool.release_chunk_pages(&pages))
//!          // or: policy-driven (greedy-dual-size-class retention, Issue
//!          // katgpt-rs 771 lineage / riir-infer plan 630 T3):
//!          → for LeafSnapshot in tree.leaf_ids() { rank; }
//!          → tree.evict_leaf(victim_id, |pages| pool.release_chunk_pages(&pages))
//! ```
//!
//! **Refcount discipline (safety-critical):** for every chunk page, the pool
//! refcount equals (# live sequences adopting it) + (1 while tree-indexed).
//! The tree never releases a page it still indexes, and eviction only drops
//! the tree's own hold — a live sequence's hold always keeps the page alive.
//! Locks are a *hit-rate* optimization (protect in-flight requests from
//! eviction), never a safety mechanism.
//!
//! # Divergences from SGLang (measured/deliberate)
//!
//! - **Chunk-floor matching:** only whole `page_tokens` chunks are shared;
//!   the trailing partial chunk is re-prefilled (≤ `page_tokens`−1 tokens).
//!   This is what makes CoW unnecessary: a request never writes into a page
//!   another branch reads.
//! - **No per-chunk hash filter:** node spans are compared chunk-wise by
//!   direct `memcmp` (16 × u32 = 64 B). The exact compare is the authority
//!   and must run anyway on a hit; a rolling-hash fast-filter only pays off
//!   for spans of hundreds of chunks, where `memcmp` is still ~100 ns. The
//!   hash+verify pattern lives in [`crate::cache_prune::KvSegmentPool`],
//!   where segments are unanchored and the filter does pay.
//! - **Node-per-request, not edge extension:** each insert adds one child
//!   node for its divergent suffix instead of extending the leaf edge. Same
//!   sharing semantics, finer LRU granularity (turn-level peel), and no
//!   extend-vs-split special case.
//! - **Locks survive splits on the head slot:** splitting keeps the node id
//!   for the matched prefix, so `MatchHit` node ids stay valid; every locker
//!   of a split node matched at most the head's span.
//!
//! # Node arena (Issue 800 C1)
//!
//! Nodes live in a
//! [`GraphStablePool`](katgpt_core::graph_stable_pool::GraphStablePool) —
//! the extracted form of the exact free-list contract this tree shipped
//! inline (`Vec<Node>` + LIFO `free_nodes`): slot ids are stable across any
//! alloc/free sequence, recycled slots are reused LIFO, and a freed slot
//! reads `None` — eviction drops the node value, so token/page buffers are
//! released eagerly instead of parked cleared-but-allocated until the next
//! overwrite. The root is slot 0, allocated at construction, never freed.
//!
//! # Modelless
//!
//! Pure index arithmetic over `u32` tokens and `usize` page indices. No
//! weights, no inference, zero deps beyond std.

use katgpt_core::graph_stable_pool::GraphStablePool;
use std::fmt;

/// Node handle. `0` is the root (empty span, never evicted).
pub type NodeId = usize;

/// Root node id.
pub const ROOT: NodeId = 0;

/// A longest-prefix match result.
///
/// `matched_chunks` is the number of whole chunks whose KV pages are
/// reusable; `partial_chunks` is how many of those live in `node` (the walk
/// always consumes the terminal fully when it did not stop inside it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchHit {
    /// Deepest node the walk reached (the locked terminal; `ROOT` when
    /// nothing matched).
    pub node: NodeId,
    /// Whole chunks matched (floor of token lcp / `page_tokens`).
    pub matched_chunks: usize,
    /// Chunks matched within `node` (its full span unless the walk stopped
    /// inside it). Pages to take from `node` = `partial_chunks`.
    pub partial_chunks: usize,
}

/// Outcome of an [`RadixPrefixTree::insert`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InsertOutcome {
    /// Chunks newly indexed by this insert (the caller must have retained
    /// exactly these chunks' pages on the pool — the tree asserts path
    /// consistency for the pre-existing prefix).
    pub new_chunks: usize,
    /// Whether a node was split at a chunk boundary.
    pub split: bool,
}

/// Tree errors — all caller-contract violations, never internal panics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeError {
    /// `pages.len() != floor(tokens.len()/page_tokens) * n_layers`.
    PagesLen { expected: usize, got: usize },
    /// The supplied page tables disagree with the already-indexed path at
    /// this chunk — the caller passed tables from a different sequence.
    PathMismatch { chunk: usize },
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeError::PagesLen { expected, got } => write!(
                f,
                "pages chunk-major len {got}, expected {expected} (chunks × n_layers)"
            ),
            TreeError::PathMismatch { chunk } => {
                write!(
                    f,
                    "supplied tables diverge from indexed path at chunk {chunk}"
                )
            }
        }
    }
}

impl std::error::Error for TreeError {}

/// One evictable leaf's ranking snapshot, from [`RadixPrefixTree::leaf_ids`].
/// A policy-driven eviction ([`RadixPrefixTree::evict_leaf`]) ranks these —
/// cost-density retention ranks `chunks` (re-prefill cost) against the pages
/// the caller's pool still uniquely holds (freed capacity), with
/// `last_touch` as the recency tiebreak. The snapshot is a momentary read:
/// ids stay valid until the next tree mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafSnapshot {
    /// The leaf's node id (pass to [`RadixPrefixTree::evict_leaf`]).
    pub id: NodeId,
    /// Whole token chunks the leaf spans (≥ 1).
    pub chunks: usize,
    /// The leaf's chunk-major page ids (`chunks × n_layers` entries).
    pub pages: Vec<usize>,
    /// LRU clock at last traversal through this leaf.
    pub last_touch: u64,
}

/// Counters for bench reporting and invariant tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct TreeStats {
    /// `match_prefix` calls.
    pub matches: u64,
    /// Sum of matched chunks over all calls (hit volume).
    pub matched_chunks_total: u64,
    /// `insert` calls that indexed ≥1 new chunk.
    pub inserts: u64,
    /// Node splits performed.
    pub splits: u64,
    /// Nodes evicted.
    pub evictions: u64,
    /// Eviction scans that ended over-budget (all leaves locked/interior).
    pub evict_deadlocks: u64,
}

/// One radix-tree node. Spans are multiples of `page_tokens` (root: empty).
struct RadixNode {
    /// Token span, chunk-aligned, ≥ 1 chunk (except the root).
    tokens: Vec<u32>,
    /// Page indices, chunk-major: `[chunk * n_layers + layer]`.
    pages: Vec<usize>,
    /// Child node ids. Lookup scans by first-chunk equality (a first-token
    /// key is not unique: two chunks may share token 0 and diverge later).
    children: Vec<NodeId>,
    parent: NodeId,
    /// LRU clock at last traversal through this node.
    last_touch: u64,
    /// Live requests whose match terminated here (eviction protection).
    lock: u32,
}

/// The radix prefix tree. See the [module docs](self) for the contract.
pub struct RadixPrefixTree {
    /// Node arena — the Issue-800-C1 [`GraphStablePool`] re-point: slot
    /// occupancy IS liveness (freed slots read `None`; the old
    /// `tokens.is_empty()` zombie marker is structural now). The root is
    /// slot 0, allocated at construction, never freed.
    nodes: GraphStablePool<RadixNode>,
    n_layers: usize,
    /// Tokens per chunk (the pool's page size — `PagedKVCache::PAGE_SIZE`).
    page_tokens: usize,
    /// Upper bound on `held_pages` (chunks × n_layers the tree indexes).
    page_budget: usize,
    held_pages: usize,
    tick: u64,
    /// Reusable scratch for insert's path-integrity check (insert is cold;
    /// match never allocates).
    scratch: Vec<usize>,
    stats: TreeStats,
}

impl RadixPrefixTree {
    /// New tree. `page_tokens` = tokens per chunk (16 for `PagedKVCache`);
    /// `page_budget` = max chunk-pages (chunks × n_layers) held before
    /// eviction is required.
    ///
    /// # Panics
    /// `n_layers == 0` or `page_tokens == 0` (constructor misuse).
    pub fn new(n_layers: usize, page_tokens: usize, page_budget: usize) -> Self {
        assert!(n_layers > 0, "n_layers must be > 0");
        assert!(page_tokens > 0, "page_tokens must be > 0");
        let mut nodes = GraphStablePool::with_capacity(64);
        let root = nodes.alloc(RadixNode {
            tokens: Vec::new(),
            pages: Vec::new(),
            children: Vec::new(),
            parent: ROOT,
            last_touch: 0,
            lock: 0,
        });
        debug_assert_eq!(root, ROOT, "root must occupy slot 0");
        Self {
            nodes,
            n_layers,
            page_tokens,
            page_budget,
            held_pages: 0,
            tick: 0,
            scratch: Vec::new(),
            stats: TreeStats::default(),
        }
    }

    /// Layers per chunk (constructor copy for callers converting tables).
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Tokens per chunk (the pool's page size).
    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    /// Chunk-pages currently held (chunks × n_layers indexed).
    pub fn held_pages(&self) -> usize {
        self.held_pages
    }

    /// Configured budget.
    pub fn page_budget(&self) -> usize {
        self.page_budget
    }

    /// Live node count (root excluded).
    pub fn node_count(&self) -> usize {
        self.nodes.len() - 1
    }

    /// Counters.
    pub fn stats(&self) -> &TreeStats {
        &self.stats
    }

    /// Total locks currently held (0 after balanced match/unlock pairs).
    pub fn total_locks(&self) -> u64 {
        self.nodes.iter().map(|(_, n)| n.lock as u64).sum()
    }

    // ── arena access ───────────────────────────────────────────────────────

    /// Borrow a live node. Referenced ids are live by the tree's own
    /// discipline (locked nodes are never evicted; a split keeps the head in
    /// place; children are unlinked before their slot is freed) — `None`
    /// here would be a tree bug, hence the loud witness.
    fn node(&self, id: NodeId) -> &RadixNode {
        self.nodes
            .get(id)
            .expect("radix node id must reference a live slot")
    }

    fn node_mut(&mut self, id: NodeId) -> &mut RadixNode {
        self.nodes
            .get_mut(id)
            .expect("radix node id must reference a live slot")
    }

    /// Visit every live node's chunk-major page hold (diagnostics /
    /// invariant audits — the pool-free-list disjointness check in the
    /// Issue-771 G1 gate).
    pub fn for_each_held_page(&self, mut f: impl FnMut(&[usize])) {
        for (id, node) in self.nodes.iter() {
            if id != ROOT {
                f(&node.pages);
            }
        }
    }

    // ── walk ───────────────────────────────────────────────────────────────

    /// Walk `tokens` chunk-aligned to the longest indexed prefix.
    /// Returns `(terminal_node, matched_chunks_total, lcp_within_terminal)`.
    ///
    /// `lcp_within_terminal` equals the terminal's full span when the walk
    /// consumed it entirely; when the walk stopped inside it, it is the
    /// partial span (the divergence chunk excluded).
    fn walk(&mut self, tokens: &[u32], touch: bool) -> (NodeId, usize, usize) {
        let pt = self.page_tokens;
        let total_chunks = tokens.len() / pt;
        let mut node = ROOT;
        let mut matched = 0usize;
        let mut lcp = 0usize;
        while matched < total_chunks {
            let Some(child) = self.node(node).children.iter().copied().find(|&c| {
                let span = &self.node(c).tokens;
                span.len() >= pt && span[..pt] == tokens[matched * pt..(matched + 1) * pt]
            }) else {
                break;
            };
            let span_chunks = self.node(child).tokens.len() / pt;
            let avail = total_chunks - matched;
            let mut c = 0usize;
            while c < span_chunks.min(avail)
                && self.node(child).tokens[c * pt..(c + 1) * pt]
                    == tokens[(matched + c) * pt..(matched + c + 1) * pt]
            {
                c += 1;
            }
            if c == 0 {
                // First-chunk equality was the find predicate, so c ≥ 1
                // whenever the child was found; defensive break.
                break;
            }
            matched += c;
            node = child;
            lcp = c;
            if touch {
                self.tick += 1;
                self.node_mut(child).last_touch = self.tick;
            }
            if c < span_chunks {
                break; // diverged inside the child
            }
        }
        (node, matched, lcp)
    }

    // ── match ──────────────────────────────────────────────────────────────

    /// Longest-prefix match. Zero-allocation. Locks the terminal node while
    /// matched > 0 (the caller MUST balance with [`Self::unlock`]).
    pub fn match_prefix(&mut self, tokens: &[u32]) -> MatchHit {
        let (node, matched, lcp) = self.walk(tokens, true);
        self.stats.matches += 1;
        self.stats.matched_chunks_total += matched as u64;
        if matched > 0 {
            self.node_mut(node).lock += 1;
        }
        MatchHit {
            node,
            matched_chunks: matched,
            partial_chunks: if matched == 0 { 0 } else { lcp },
        }
    }

    /// Drop the lock taken by the matching [`Self::match_prefix`] call.
    /// `hit.node` stays valid while locked: a locked node is never evicted
    /// and its ancestors always have children, and a split keeps the head
    /// in the same slot.
    pub fn unlock(&mut self, hit: &MatchHit) {
        if hit.matched_chunks == 0 {
            return;
        }
        let node = self.node_mut(hit.node);
        debug_assert!(node.lock > 0, "unlock on unlocked node {}", hit.node);
        node.lock = node.lock.saturating_sub(1);
    }

    /// Copy the matched path's page indices (chunk-major) into `out`
    /// (caller-owned scratch — zero allocation). `out.len()` after the call
    /// is `hit.matched_chunks * n_layers`.
    pub fn path_pages_into(&self, hit: &MatchHit, out: &mut Vec<usize>) {
        let nl = self.n_layers;
        out.clear();
        out.resize(hit.matched_chunks * nl, 0);
        let mut fill = hit.matched_chunks;
        let mut n = hit.node;
        let mut terminal = true;
        while fill > 0 {
            let node = self.node(n);
            let node_chunks = node.tokens.len() / self.page_tokens;
            let take = if terminal {
                hit.partial_chunks
            } else {
                node_chunks
            };
            debug_assert!(take <= node_chunks, "partial exceeds node span");
            fill -= take;
            let dst = fill * nl;
            let src = take * nl;
            out[dst..dst + src].copy_from_slice(&node.pages[..src]);
            terminal = false;
            n = node.parent;
        }
        debug_assert_eq!(fill, 0, "path did not cover matched chunks");
    }

    // ── insert ─────────────────────────────────────────────────────────────

    /// Index `tokens`'s whole chunks against `pages` (chunk-major, length
    /// `floor(tokens.len()/page_tokens) * n_layers`). The divergent suffix
    /// becomes a new child node; a node whose span the tokens leave midway
    /// is split in place at the chunk boundary.
    ///
    /// The caller retains the NEW chunks' pages on the pool (`matched ..
    /// total`); insert verifies the pre-existing path's tables against
    /// `pages[..matched * n_layers]` and errors on divergence.
    ///
    /// Does NOT touch locks and does NOT evict — call
    /// [`Self::evict_to_budget`] when `held_pages() > page_budget()`.
    pub fn insert(
        &mut self,
        tokens: &[u32],
        pages_chunk_major: &[usize],
    ) -> Result<InsertOutcome, TreeError> {
        let pt = self.page_tokens;
        let nl = self.n_layers;
        let m = tokens.len() / pt;
        if m == 0 {
            return Ok(InsertOutcome::default());
        }
        let expected = m * nl;
        if pages_chunk_major.len() != expected {
            return Err(TreeError::PagesLen {
                expected,
                got: pages_chunk_major.len(),
            });
        }
        self.tick += 1;
        let (node, matched, lcp) = self.walk(tokens, true);

        // Path integrity: the caller's tables for the already-indexed
        // prefix must equal the stored ones (they adopted those pages).
        if matched > 0 {
            let hit = MatchHit {
                node,
                matched_chunks: matched,
                partial_chunks: lcp,
            };
            // mem::take: `path_pages_into` takes &self while we need &mut
            // scratch — swap the buffer out for the duration.
            let mut scratch = std::mem::take(&mut self.scratch);
            self.path_pages_into(&hit, &mut scratch);
            let bad = (0..matched).find(|&chunk| {
                let base = chunk * nl;
                scratch[base..base + nl] != pages_chunk_major[base..base + nl]
            });
            self.scratch = scratch;
            if let Some(chunk) = bad {
                return Err(TreeError::PathMismatch { chunk });
            }
        }

        if matched == m {
            // Fully indexed already (re-insert of a known sequence).
            return Ok(InsertOutcome::default());
        }

        // Split the terminal when the walk stopped inside its span.
        let mut split = false;
        if node != ROOT {
            let span_chunks = self.node(node).tokens.len() / pt;
            debug_assert!(lcp <= span_chunks);
            if lcp < span_chunks {
                self.split_node(node, lcp);
                split = true;
            }
        }

        // Attach the divergent suffix [matched..m) as a new child.
        let new_chunks = m - matched;
        let child_tokens = tokens[matched * pt..m * pt].to_vec();
        let child_pages = pages_chunk_major[matched * nl..].to_vec();
        let id = self.alloc_node(child_tokens, child_pages, node);
        self.node_mut(node).children.push(id);
        self.held_pages += new_chunks * nl;
        self.stats.inserts += 1;
        Ok(InsertOutcome { new_chunks, split })
    }

    /// Split `id` at `at_chunks`: the head keeps `[..at)` tokens/pages, the
    /// lock, the parent, and the slot id; a fresh tail node takes `[at..)`
    /// plus all children. Every locker of `id` matched at most `at` chunks,
    /// so locks correctly stay with the head.
    fn split_node(&mut self, id: NodeId, at_chunks: usize) {
        let pt = self.page_tokens;
        let nl = self.n_layers;
        let tail_tokens = self.node_mut(id).tokens.split_off(at_chunks * pt);
        let tail_pages = self.node_mut(id).pages.split_off(at_chunks * nl);
        let tail_children = std::mem::take(&mut self.node_mut(id).children);
        let tail = self.alloc_node(tail_tokens, tail_pages, id);
        self.node_mut(tail).children = tail_children;
        for i in 0..self.node(tail).children.len() {
            let c = self.node(tail).children[i];
            self.node_mut(c).parent = tail;
        }
        self.node_mut(id).children.push(tail);
        self.stats.splits += 1;
    }

    fn alloc_node(&mut self, tokens: Vec<u32>, pages: Vec<usize>, parent: NodeId) -> NodeId {
        debug_assert_eq!(tokens.len() % self.page_tokens, 0);
        debug_assert_eq!(pages.len(), tokens.len() / self.page_tokens * self.n_layers);
        let node = RadixNode {
            tokens,
            pages,
            children: Vec::new(),
            parent,
            last_touch: self.tick,
            lock: 0,
        };
        self.nodes.alloc(node)
    }

    // ── eviction ───────────────────────────────────────────────────────────

    /// Enumerate the currently evictable leaves (childless, unlocked,
    /// non-root) with the data a retention policy ranks on: span chunks,
    /// page ids, recency. A cold-path snapshot — the leaf set changes on
    /// every eviction/insert, so rank from one snapshot per round and
    /// evict through [`Self::evict_leaf`].
    pub fn leaf_ids(&self) -> Vec<LeafSnapshot> {
        self.nodes
            .iter()
            .filter(|&(id, n)| id != ROOT && n.children.is_empty() && n.lock == 0)
            .map(|(id, n)| LeafSnapshot {
                id,
                chunks: n.tokens.len() / self.page_tokens,
                pages: n.pages.clone(),
                last_touch: n.last_touch,
            })
            .collect()
    }

    /// Evict ONE leaf. Returns `false` — untouched — unless `id` is
    /// currently an evictable leaf (childless, unlocked, live, not root):
    /// interior, locked, stale, and root ids all refuse, so a policy may
    /// act on a slightly stale snapshot without corrupting the tree. On
    /// success `release` receives the node's chunk-major pages (the caller
    /// drops the tree's pool hold) exactly as [`Self::evict_to_budget`]
    /// would, the node is unlinked from its parent and its slot freed.
    pub fn evict_leaf(&mut self, id: NodeId, release: impl FnOnce(&[usize])) -> bool {
        let evictable = self
            .nodes
            .get(id)
            .is_some_and(|n| id != ROOT && n.children.is_empty() && n.lock == 0);
        if !evictable {
            return false;
        }
        // Release the page hold while the node is still borrowed, then
        // take the slot: the pool's free drops the node value, so the
        // token/page buffers are released here instead of parked
        // cleared-but-allocated until the next overwrite.
        let (parent, chunks) = {
            let node = self.node(id);
            release(&node.pages);
            (node.parent, node.tokens.len() / self.page_tokens)
        };
        self.node_mut(parent).children.retain(|&c| c != id);
        drop(self.nodes.free(id));
        self.held_pages -= chunks * self.n_layers;
        self.stats.evictions += 1;
        true
    }

    /// Evict leaf-preferential LRU until `held_pages <= budget` (or every
    /// leaf is locked — reported via `stats.evict_deadlocks`). `release`
    /// receives each evicted node's chunk-major pages; the caller drops the
    /// tree's pool hold for them (`PagedKVCache::release_chunk_pages`).
    ///
    /// Interior nodes are never evicted (they have children); a locked leaf
    /// is skipped; becoming-childless parents are picked up by later scans.
    /// A custom retention policy (cost-density, priority, …) can replace
    /// the victim choice entirely: [`Self::leaf_ids`] + [`Self::evict_leaf`].
    pub fn evict_to_budget(&mut self, mut release: impl FnMut(&[usize])) {
        while self.held_pages > self.page_budget {
            // Slot occupancy is liveness now: freed slots yield nothing from
            // `iter`, and every live non-root node carries a non-empty span,
            // so the candidate set is identical to the old
            // `!tokens.is_empty()` filter.
            let Some(victim) = self
                .nodes
                .iter()
                .filter(|&(id, n)| id != ROOT && n.children.is_empty() && n.lock == 0)
                .min_by_key(|(_, n)| n.last_touch)
                .map(|(id, _)| id)
            else {
                self.stats.evict_deadlocks += 1;
                break;
            };
            // The victim came from the same evictable predicate evict_leaf
            // re-checks, so this cannot refuse; a false here is a defensive
            // deadlock (never loop on a refusing victim).
            if !self.evict_leaf(victim, |pages| release(pages)) {
                self.stats.evict_deadlocks += 1;
                break;
            }
        }
    }

    /// Drop every node, releasing all chunk-page holds through `release`.
    /// Locks are ignored (test/teardown path).
    pub fn clear(&mut self, mut release: impl FnMut(&[usize])) {
        // Free every non-root slot; each node value is dropped (its
        // token/page buffers go with it) after its page hold is released.
        // Freed slots read `None`, so `node_count` (len − 1) reads 0 and
        // future inserts reuse the arena LIFO — highest index first,
        // matching the old `(1..len)` free-list build.
        let slots = self.nodes.capacity();
        for id in 1..slots {
            if let Some(node) = self.nodes.free(id) {
                release(&node.pages);
            }
        }
        self.node_mut(ROOT).children.clear();
        self.held_pages = 0;
    }
}

#[cfg(test)]
mod tests {
    // Property/behavior tests live in `tests/radix_prefix_cache_tests.rs`
    // (integration, `required-features = ["radix_prefix_cache"]`) so they
    // cannot silently compile to a green zero under default features — the
    // Issue-713 cfg-gated-target discipline.
}
