//! `GraphStablePool<T>` — the append-only + free-list + never-invalidate slot
//! pool (Issue 800 Arm C).
//!
//! A DRY extraction of a contract that ships **four times under four names**
//! — an independently-rediscovered substrate, not new machinery. The narrowest
//! common contract: slot-index allocation where **allocating never invalidates
//! the indices of live elements**; a free list recycles slots; the backing
//! store grows by append only (never compacts, never reorders).
//!
//! # The four sites
//!
//! | # | Site | Slot store | Free list | Growth | Stability it actually needs |
//! |---|---|---|---|---|---|
//! | 1 | [`katgpt-kv` `radix_prefix`] `RadixPrefixTree` — `crates/katgpt-kv/src/radix_prefix/mod.rs` (nodes L155, `free_nodes` L157, LIFO pop `alloc_node` L478-484, push on evict L517) — **✅ RE-POINTED 2026-09-16** onto this type (the inline `free_nodes` stack is gone; eviction drops node values instead of parking cleared buffers) | `GraphStablePool<RadixNode>` | (pool free list), LIFO | append | **index** (node ids stable; node values move on growth) — as before |
//! | 2 | `katgpt-transformer` `PagedKVCache` — `crates/katgpt-transformer/src/kv_cache.rs` (`pages` L506, `free_pages` L510, LIFO pop `alloc_page` L550-562, push on rollback/release L709-711 + L765-767) — **⏸ RE-POINT DECLINED 2026-09-16** (contract-matched lineage, kept as-is; verdict below) | `Vec<Vec<f32>>` (heap handle per slot) | `Vec<usize>`, LIFO | append (`push`) | **index**, plus **payload address stability** via the Vec-of-Vec indirection (handles move on growth; the f32 page buffers behind them never do — "the pool never moves pages") |
//! | 3 | `riir-gpu` `Qwen38LaneSet` — `../riir-ai/crates/riir-gpu/src/qwen38_dense_cudarc.rs` (L2813-2853, `enable_lanes` L5096-5168) — **✅ N.A. 2026-09-16**: no free list, no slot churn — the set already IS the lifetime-scoped flat-arena pattern; nothing for `alloc`/`free` to own (struct doc records the verdict) | one flat `CudaSlice<f32>` per layer, `n` **fixed** lane slots via zero-copy slice views | **none** | **whole-set replacement** (drops the captured graphs together with the pointers they bake) | **true address stability** — captured CUDA graphs bake DEVICE POINTERS into the arenas (L2843-2848); achieved by allocate-once pre-allocation, not chunking |
//! | 4 | this crate's `BranchBank<E>` — `src/branching/bank.rs` (`free_slots` L47, LIFO pop L205-210, append L214-222, push on prune L302) — **⏸ RE-POINT DECLINED 2026-09-16** (wire-pinned; verdict below) | `Vec<CognitiveBranch>` by value | `Vec<u32>`, LIFO | append, capped by `max_branches` (caller policy) | **index** (`BranchId` stable) |
//!
//! # The stability verdict (read this before the name misleads you)
//!
//! This type guarantees **INDEX stability, and states it as such**: while an
//! element is live, its slot index never changes — across any sequence of
//! `alloc`/`free` calls, including growth. That is exactly what sites 1 and 4
//! consume, and it is the honest majority shape (site 2's payload guarantee is
//! a property of *its* `T` being a heap handle, see below).
//!
//! This type does **NOT** promise that a `&T` borrowed from `get` survives the
//! next `alloc`: growing the slot array reallocates it and **moves the
//! `Option<T>` slot values in memory**. Hold the index; re-borrow after each
//! alloc. A debug-mode `#[test]` in this module pins the payload-level analog:
//! when `T` is itself a heap indirection (`Vec<_>`, the site-2 recipe —
//! `pages: Vec<Vec<f32>>`), the allocation **behind** each `T` never moves,
//! because growth relocates only the 24-byte handles. That payload-stability
//! is the captured-graph property in its host terms, and it is `T`-shaped,
//! not pool-shaped: the pool never touches a live slot's payload.
//!
//! Site 3 achieves true address stability the other way — flat allocate-once
//! buffers plus whole-set replacement, with the graph cache scoped INSIDE the
//! lane set so a replaced set takes its baked pointers with it. That is a
//! lifetime-scoping discipline, not a layout, which is why the extracted type
//! does **not** adopt chunked/pinned storage: chunking would complicate the
//! API beyond what the two index-stable sites (1, 4) and the handle-stable
//! site (2) need, and site 3's guarantee does not come from chunking anyway.
//! The site-3 re-point maps to: pre-allocate with [`GraphStablePool::with_capacity`],
//! and scope any address-baked artifact to the pool's lifetime (the
//! `Qwen38LaneSet` pattern).
//!
//! # Site deltas (what the follow-up re-points must adapt — wrappers or generics)
//!
//! | Site | Delta vs this type | Adaptation on re-point |
//! |---|---|---|
//! | 1 `radix_prefix` | Nodes carry structured payloads (`tokens`/`pages`/`children` Vecs) and eviction *clears* payloads instead of taking them; free slots are recognized by `tokens.is_empty()` | **✅ DONE 2026-09-16** — node struct stored as `T`; `alloc`/`free` replace the inline pop/push; liveness is pool occupancy (the `tokens.is_empty()` zombie filter is structural now); the arena scans (LRU eviction, `total_locks`, `for_each_held_page`) consume `iter()`
//! | 2 `PagedKVCache` | Refcount layer decides when a page is freed; `alloc_page` **refills** the existing `Vec<f32>` (`fill(0.0)`) instead of replacing the handle — a buffer-recycling micro-opt | **⏸ DECLINED 2026-09-16** — the refill-at-stable-index contract forces the recycled buffers OUT of the pool between free and reuse (a side stash), turning one bookkeeping structure (`free_pages`) into two (pool free-stack + buffer stash); `pages`/`free_pages`/`page_ref_counts` are `pub` and are the measurement instrument of bench_414's legacy replica + root tests; the pool's None-on-freed safety targets a class rollback's table truncation already prevents. Lineage row stands — the pool was distilled FROM this shape; re-pointing is not what makes the claim true |
//! | 3 `Qwen38LaneSet` | No free list; fixed `n`; whole-set replacement invalidating baked pointers | **✅ N.A.** — discipline already implemented in the code (allocate-once + graphs-scoped-inside-the-set); a pool wrapper adds alloc/free nothing calls. Verdict recorded in the struct doc (riir-ai `35108b6a7`) |
//! | 4 `BranchBank` | Fixed `max_branches` cap enforced caller-side (`debug_assert` before append); flat anchor side-cache keyed by slot | **⏸ DECLINED 2026-09-16** — the wire format (`to_bytes`/`from_bytes`) pins slots-by-value with in-band `Removed` zombie slots + the EXPLICIT `free_slots` stack in order; a pool-backed rebuild cannot reproduce that byte-identity without exposing pool internals (the free-stack order is an implementation detail), and the wire feeds neuron-db freeze — a change is a versioned migration, not a refactor. Same free-list machinery verdict stands as the reason the site WAS matched |
//!
//! # API contract
//!
//! - [`alloc`](GraphStablePool::alloc) — pops the free list (**LIFO**, matching
//!   the free-list discipline of all three free-list sites: radix's former
//!   `free_nodes.pop` (now this type's own stack), `PagedKVCache::alloc_page`
//!   `free_pages.pop`, `BranchBank::free_slots.pop`), else appends and returns
//!   the new index. Writing a recycled slot replaces a `None`, so alloc never
//!   drops a stale value.
//! - [`free`](GraphStablePool::free) — takes the value out and returns it
//!   (`Option<T>`); pushes the index onto the free list. **Double-free is a
//!   silent no-op returning `None`** — the `BranchBank::prune` discipline
//!   (quiet `false` on already-removed, bank.rs L287-304), the majority
//!   defensive posture; out-of-range is likewise a silent `None`. A freed
//!   slot reads back `None` from `get`/`get_mut` — no stale-value footgun.
//! - [`get`](GraphStablePool::get) / [`get_mut`](GraphStablePool::get_mut) —
//!   `None` for out-of-range **or freed** indices.
//! - [`len`](GraphStablePool::len) — live count. [`capacity`](GraphStablePool::capacity)
//!   — total slots ever (live + free), the `PagedKVCache::total_pages` /
//!   `BranchBank::branches.len()` semantics; NOT the `Vec` backing capacity.
//!
//! Slots are `Option<T>`: zero-cost when `T` has a pointer niche (every
//! site's `T` qualifies — `Vec` handles, structs containing Vecs, slices) and
//! +1 aligned byte otherwise. This is what makes freed slots distinguishable
//! without a parallel occupancy array (radix uses `tokens.is_empty()`,
//! `PagedKVCache` uses refcounts — same job, per-site shape).
//!
//! # Zero-alloc steady state (G4)
//!
//! `alloc` / `free` / `get` / `get_mut` perform no heap allocation once the
//! working set is warm: recycled-slot writes replace in place, the free list
//! is bounded by the slot count and stops reallocating after warmup. The
//! documented exception is **amortized growth** — appending a slot beyond the
//! current backing capacity reallocates the slot array (and free-list pushes
//! realloc until its capacity covers the peak free depth), exactly the
//! exception every site accepts. An inline gate below pins 1000 churn cycles
//! at zero allocations on a pre-warmed pool.
//!
//! # Modelless + portability
//!
//! Pure index arithmetic over `std` `Vec`/`Option` — no weights, no inference,
//! no `std::arch`, no deps. wasm32-clean by construction (the whole-module
//! check runs in CI lanes).
//!
//! graph_stable_pool — the append-only + free-list + never-invalidate slot
//! pool (Issue 800 Arm C): the DRY extraction of the contract that ships four
//! times under four names — katgpt-kv `radix_prefix` nodes,
//! katgpt-transformer `PagedKVCache` pages, riir-gpu `Qwen38LaneSet` arenas,
//! and this crate's `BranchBank` slots. INDEX-stable allocation (live slot
//! indices never move); see the module docs for the index-vs-address
//! stability verdict and the per-site re-point deltas. Phase 1: type +
//! contract tests only; consumer re-points are incremental follow-ups.

/// The slot pool. See the [module docs](self) for the contract and the
/// four-site comparison.
#[derive(Debug)]
pub struct GraphStablePool<T> {
    /// Slot storage. `Some` = live, `None` = free (recycled or never
    /// written). Indices into this Vec ARE the pool's stable handles.
    slots: Vec<Option<T>>,
    /// Recycled slot indices — LIFO stack (`pop` on alloc, `push` on free),
    /// the shape of all three free-list sites. Bounded by `slots.len()`.
    free: Vec<usize>,
    /// Live count: `slots` entries that are `Some`. Maintained by
    /// `alloc`/`free`; `len + free.len() == slots.len()` is the structural
    /// invariant (debug-asserted).
    live: usize,
}

impl<T> GraphStablePool<T> {
    /// New empty pool. No allocation until the first `alloc`.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    /// New pool with backing capacity for `cap` slots pre-reserved — the
    /// site-3 fixed-capacity posture (pre-allocate everything, then never
    /// churn past it) and the site-4 `max_branches` pre-allocation. Does not
    /// create the slots; `capacity()` still reads 0 until slots are appended.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
            free: Vec::new(),
            live: 0,
        }
    }

    /// Allocate a slot holding `value`; returns its index.
    ///
    /// Reuses the most recently freed slot (LIFO — the free-list discipline
    /// of all three free-list sites), else appends a new slot. The returned
    /// index is stable for the element's lifetime: no subsequent sequence of
    /// `alloc`/`free` calls changes it (INDEX stability — the load-bearing
    /// guarantee; see the module docs for the index-vs-address verdict).
    ///
    /// Zero-alloc in steady state (free list non-empty); amortized-growth
    /// allocation when appending past the backing capacity.
    pub fn alloc(&mut self, value: T) -> usize {
        debug_assert_eq!(
            self.live + self.free.len(),
            self.slots.len(),
            "pool structural invariant broken"
        );
        let idx = match self.free.pop() {
            Some(idx) => {
                // Recycled slot holds None (free() took the value), so this
                // write drops nothing.
                self.slots[idx] = Some(value);
                idx
            }
            None => {
                self.slots.push(Some(value));
                self.slots.len() - 1
            }
        };
        self.live += 1;
        idx
    }

    /// Free the slot at `idx`, returning its value.
    ///
    /// The index is pushed onto the free list and WILL be returned by a
    /// later `alloc` (LIFO). Already-freed (double-free) and out-of-range
    /// indices are **silent no-ops returning `None`** — the
    /// `BranchBank::prune` majority discipline; callers that need loud
    /// failure should match on the `None` return.
    ///
    /// The value is returned rather than dropped so heap-payload `T`s (the
    /// site-2 recipe) can be recycled without a dealloc/realloc round trip.
    pub fn free(&mut self, idx: usize) -> Option<T> {
        debug_assert_eq!(
            self.live + self.free.len(),
            self.slots.len(),
            "pool structural invariant broken"
        );
        if idx >= self.slots.len() {
            return None;
        }
        match self.slots[idx].take() {
            Some(value) => {
                self.free.push(idx);
                self.live -= 1;
                Some(value)
            }
            // Double-free: silent no-op (documented discipline).
            None => None,
        }
    }

    /// Borrow the live value at `idx`; `None` when out-of-range or freed.
    ///
    /// The reference is valid for the borrow's lifetime only — it must not
    /// be held across `alloc` (growth may move the slot values). Hold the
    /// index instead; that is the guarantee this type makes.
    pub fn get(&self, idx: usize) -> Option<&T> {
        self.slots.get(idx).and_then(Option::as_ref)
    }

    /// Mutably borrow the live value at `idx`; `None` when out-of-range or
    /// freed. Same lifetime rule as [`get`](Self::get).
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        self.slots.get_mut(idx).and_then(Option::as_mut)
    }

    /// Iterate live slots as `(index, &value)` pairs in slot order —
    /// ascending index, freed slots never yielded.
    ///
    /// O(total slots): the same shape as the full-arena scans the sites run
    /// today (the radix LRU eviction scan + held-page audit, BranchBank's
    /// active sweeps). Zero-alloc (lazy `filter_map` over the slot slice).
    /// The borrow lives for the loop body only — hold indices, not
    /// references, across any later `alloc` (the same rule as [`get`](Self::get)).
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.as_ref().map(|value| (idx, value)))
    }

    /// Live element count (freed slots excluded).
    pub fn len(&self) -> usize {
        self.live
    }

    /// True when no element is live.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Total slots ever allocated: live + free. Matches
    /// `PagedKVCache::total_pages` / `BranchBank::branches.len()` semantics —
    /// this is NOT the `Vec` backing capacity.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
}

impl<T> Default for GraphStablePool<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C2 (1): alloc→free→alloc reuses the freed slot, LIFO order — the
    /// free-list discipline of all three free-list sites.
    #[test]
    fn iter_yields_live_slots_in_index_order() {
        let mut pool = GraphStablePool::new();
        let a = pool.alloc('a');
        let b = pool.alloc('b');
        let c = pool.alloc('c');
        assert_eq!((a, b, c), (0, 1, 2));

        // A freed slot is skipped, never yielded as None.
        assert_eq!(pool.free(b), Some('b'));
        let got: Vec<(usize, char)> = pool.iter().map(|(i, v)| (i, *v)).collect();
        assert_eq!(got, vec![(0, 'a'), (2, 'c')]);

        // A recycled slot re-enters at its index, order preserved.
        let b2 = pool.alloc('d');
        assert_eq!(b2, b);
        let got: Vec<usize> = pool.iter().map(|(i, _)| i).collect();
        assert_eq!(got, vec![0, 1, 2]);
        let values: Vec<char> = pool.iter().map(|(_, v)| *v).collect();
        assert_eq!(values, vec!['a', 'd', 'c']);
    }

    #[test]
    fn alloc_reuses_freed_slot_lifo() {
        let mut pool = GraphStablePool::new();
        let a = pool.alloc(10u64);
        let b = pool.alloc(20);
        let c = pool.alloc(30);
        assert_eq!((a, b, c), (0, 1, 2));
        assert_eq!(pool.len(), 3);
        assert_eq!(pool.capacity(), 3);

        assert_eq!(pool.free(b), Some(20));
        assert_eq!(pool.alloc(21), b, "freed slot must be reused");

        assert_eq!(pool.free(a), Some(10));
        assert_eq!(pool.free(c), Some(30));
        assert_eq!(pool.alloc(31), c, "LIFO: most recently freed first");
        assert_eq!(pool.alloc(11), a);
        assert_eq!(pool.len(), 3);
        assert_eq!(pool.capacity(), 3, "no growth once slots recycle");
    }

    /// C2 (2): the generalization of bench_762 `g1_address_stability` —
    /// indices (and values) of LIVE elements never change across ≥1000
    /// alloc/free churn cycles. In bench_762 the held pages were the
    /// tree's; here the held elements are the caller's, and the churn is
    /// scratch-slot alloc/free.
    #[test]
    fn live_indices_stable_across_churn() {
        let mut pool = GraphStablePool::new();
        let held: Vec<usize> = (0..16u64).map(|i| pool.alloc(1000 + i)).collect();
        assert!(held.iter().enumerate().all(|(i, &h)| h == i));

        // A single scratch slot churned 1000×: free returns the previous
        // value, alloc returns the same index back (LIFO single-slot stack).
        let mut scratch = pool.alloc(u64::MAX);
        assert_eq!(scratch, 16);
        for cycle in 0..1000u64 {
            let old = pool.free(scratch).expect("scratch must be live");
            let expected = if cycle == 0 { u64::MAX } else { cycle - 1 };
            assert_eq!(old, expected, "value must round-trip through free");
            scratch = pool.alloc(cycle);
            assert_eq!(scratch, 16, "scratch index must never move");
        }

        // Multi-slot churn: rotate four scratch slots in reverse-free order
        // to exercise the LIFO stack under interleaving.
        let scratches: Vec<usize> = (0..4).map(|_| pool.alloc(0u64)).collect();
        for cycle in 0..1000u64 {
            for &s in scratches.iter().rev() {
                assert!(pool.free(s).is_some());
            }
            for (k, &s) in scratches.iter().enumerate() {
                let got = pool.alloc(cycle * 4 + k as u64);
                assert_eq!(got, s, "LIFO must return the same slot set");
            }
        }

        for (i, &idx) in held.iter().enumerate() {
            assert_eq!(idx, i, "held index changed after churn");
            assert_eq!(
                pool.get(idx),
                Some(&(1000 + i as u64)),
                "held value changed"
            );
        }
        assert_eq!(pool.len(), 16 + 1 + 4);
    }

    /// C2 (3): stale-index discipline — freed slots read `None` (no stale
    /// value leak), double-free and out-of-range are silent no-ops returning
    /// `None` (the `BranchBank::prune` majority discipline), and neither
    /// corrupts pool state.
    #[test]
    fn stale_index_discipline_is_silent_noop() {
        let mut pool = GraphStablePool::new();
        let a = pool.alloc(1u64);
        assert_eq!(pool.free(a), Some(1));

        // Freed slot reads None from both accessors.
        assert_eq!(pool.get(a), None);
        assert_eq!(pool.get_mut(a), None);

        // Double-free: silent no-op.
        assert_eq!(pool.free(a), None);
        // Out-of-range: silent no-op.
        assert_eq!(pool.free(999), None);
        assert_eq!(pool.get(999), None);
        assert_eq!(pool.get_mut(999), None);

        // State untouched by both no-ops: the slot stays exactly once on
        // the free list.
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.capacity(), 1);
        assert_eq!(
            pool.free,
            vec![a],
            "free list must hold the slot exactly once"
        );
        assert!(pool.is_empty());
    }

    /// C2 (2b): the free-list-disjointness invariant, generalized from
    /// bench_762 `g1_address_stability` (after eviction churn, no tree-held
    /// page is in the pool's free list). Here: after randomized churn, every
    /// live index is absent from the free list and the structural invariant
    /// holds — the property that keeps a live slot from ever being handed
    /// out twice.
    #[test]
    fn free_list_disjoint_from_live_slots_after_churn() {
        let mut pool = GraphStablePool::new();
        // xorshift64 — deterministic, no dep.
        let mut rng = 0x243F6A8885A308D3u64;
        let mut live: Vec<(usize, u64)> = Vec::new();
        let mut next_val = 0u64;
        for _ in 0..2000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            if rng & 1 == 0 || live.len() < 8 {
                let idx = pool.alloc(next_val);
                live.push((idx, next_val));
                next_val += 1;
            } else {
                let victim = (rng as usize) % live.len();
                let (idx, val) = live.swap_remove(victim);
                assert_eq!(pool.free(idx), Some(val), "freed value must round-trip");
            }
        }

        // The disjointness + structural invariants over the final state.
        assert_eq!(pool.len(), live.len());
        assert_eq!(pool.len() + pool.free.len(), pool.capacity());
        for &(idx, val) in &live {
            assert!(
                !pool.free.contains(&idx),
                "live slot {idx} is in the free list (bench_762 g1 class)"
            );
            assert_eq!(pool.get(idx), Some(&val), "live value corrupted");
        }
        let mut sorted = pool.free.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), pool.free.len(), "duplicate free-list entry");
        assert!(
            pool.free.iter().all(|&i| pool.slots[i].is_none()),
            "free-list entry points at a live slot"
        );
        assert_eq!(
            pool.slots.iter().filter(|s| s.is_some()).count(),
            pool.len(),
            "live count disagrees with Some-slot count"
        );
    }

    /// The payload-stability pin: when `T` is a heap handle (the site-2
    /// recipe — `PagedKVCache` stores `Vec<Vec<f32>>`), pool growth moves
    /// only the slot handles; the allocation BEHIND each live `T` never
    /// moves. This is the captured-graph property at the payload level and
    /// the reason a `T`-indirection is the documented route to
    /// address-stable payloads.
    #[test]
    fn payload_addresses_stable_when_t_is_a_heap_handle() {
        let mut pool = GraphStablePool::new();
        let mut addrs: Vec<(usize, usize)> = Vec::new();
        for i in 0..8usize {
            let idx = pool.alloc(vec![i as u64; 64]);
            addrs.push((idx, pool.get(idx).unwrap().as_ptr() as usize));
        }
        // Grow well past the original backing capacity — the slot array
        // reallocates (handles move), the payloads must not.
        for i in 8..256usize {
            let idx = pool.alloc(vec![i as u64; 64]);
            assert_eq!(pool.get(idx).unwrap().len(), 64);
        }
        for (idx, addr) in addrs {
            assert_eq!(
                pool.get(idx).unwrap().as_ptr() as usize,
                addr,
                "payload of live slot {idx} moved across pool growth"
            );
        }
    }

    /// C2 (4): zero-alloc steady state (G4) — 1000 churn cycles on a
    /// pre-warmed pool perform no heap allocation. Gated on the Issue-741
    /// predicate: the same `any(debug_assertions, alloc_tracking)` gate that
    /// carries the `alloc` module and the test binary's
    /// `TrackingAllocator`, so the counters exist exactly when this test
    /// does (lib.rs `TEST_GLOBAL_ALLOC`).
    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn zero_alloc_steady_state() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};

        let mut pool = GraphStablePool::with_capacity(64);
        // Warmup: create the working set AND prime the free list's backing
        // capacity so the measured loop exercises only the steady state.
        let mut scratches: Vec<usize> = Vec::with_capacity(4);
        for i in 0..32u64 {
            let idx = pool.alloc(i);
            assert_eq!(idx, i as usize);
        }
        for i in 32..36u64 {
            scratches.push(pool.alloc(i));
        }
        for _ in 0..8 {
            // Free in reverse so the LIFO pop order matches the forward
            // alloc order below (free 35..32 → stack top is 32 → pop 32 first).
            for &s in scratches.iter().rev() {
                assert!(pool.free(s).is_some());
            }
            for (k, &s) in scratches.iter().enumerate() {
                assert_eq!(pool.alloc(32 + k as u64), s);
            }
        }

        reset_alloc_stats();
        for cycle in 0..1000u64 {
            for &s in scratches.iter().rev() {
                assert!(pool.free(s).is_some());
            }
            // Read path on a just-freed index (None) and the alloc write.
            for (k, &s) in scratches.iter().enumerate() {
                assert_eq!(pool.get(s), None);
                assert_eq!(pool.alloc(32 + k as u64 + cycle), s);
                assert!(pool.get(s).is_some());
            }
        }
        let (count, _bytes) = get_alloc_stats();
        assert_eq!(count, 0, "steady-state churn allocated {count} times");
    }
}
