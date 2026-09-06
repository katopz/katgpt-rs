//! State-Action Pair Cache for MCTS over Deterministic Inference Actions.
//!
//! Distilled from **UnMaskFork** (Misaki & Akiba, Sakana AI, Feb 2026,
//! [arXiv:2602.04344](https://arxiv.org/abs/2602.04344)) — test-time scaling for
//! masked diffusion via deterministic action branching. This module is the
//! open, modelless, game-IP-free half of the primitive: a lock-free
//! `StateActionCache` keyed on `(blake3::Hash, InferenceAction)` pairs, plus a
//! generic `mcts_search_with_state_action_cache` entry point that
//! consults/inserts the cache at Expand time.
//!
//! # Why this exists (vs the shipped `mcts` substrate)
//!
//! The existing [`crate::mcts`] module is generic over [`crate::traits::GameState`]
//! and caches at the **state** granularity (transposition). It cannot represent
//! the "same state, *different deterministic action*, different transition"
//! axis that UnMaskFork exploits: when transitions are deterministic
//! (`Var[R | s, a] = 0`), the pair `(state, action)` uniquely determines the
//! next state and reward, so revisiting that pair is a zero-cost cache hit.
//! State-only caches lose this because two actions at the same state collide
//! on the state key.
//!
//! # The DeterministicTransition contract (load-bearing)
//!
//! **The caller MUST guarantee that for any fixed `(state, action)`, applying
//! `action` to `state` yields a unique `(next_state, reward)`.** Concretely:
//! the `InferenceActionSpace::apply` function must be a pure function of
//! `(state, action)` — no internal RNG, no thread-local drift, no
//! wall-clock-dependent tie-breaking.
//!
//! Violations cause stale cache hits: a later visit to `(s, a)` reads the
//! `(s', r)` recorded by an earlier visit, even though re-applying `a` to `s`
//! would now produce a different result. The Phase 2 debug-mode re-check
//! (`StateActionCache::verify_determinism`) detects such drift empirically,
//! but the contract itself is trusted in release builds (the re-check is a
//! diagnostic, not a correctness backstop).
//!
//! # Theoretical motivation — UnMaskFork Eq. 1
//!
//! Let `ε_a(z)` be the kernel error of action `a` on state `z`. Then:
//!
//! ```text
//!   Σ_z min_a ε_a(z)  ≤  min_a Σ_z ε_a(z)
//!      (switching)         (single best static)
//! ```
//!
//! The left-hand side is the error of a *state-dependent switching* policy
//! (pick the best action per state); the right-hand side is the error of the
//! best *single static* action applied everywhere. The inequality holds
//! pointwise (the per-state min dominates any fixed choice) and is **strict**
//! whenever no single action dominates in every state — which is exactly the
//! case UnMaskFork's dLLM-unmasking case study exhibits (different remasking
//! strategies win at different mask ratios). The cache is what makes the
//! switching policy budget-cheap: once `(s, a) → (s', r)` is recorded, the
//! switch costs zero NFE.
//!
//! See `.research/386_*.md` §1.4 for the full derivation and the Phase 2
//! property test (`tests/mcts_state_action_cache_eq1.rs`) for an empirical
//! check on a synthetic landscape.
//!
//! # vs state-only transposition (Plan 061 / Plan 388)
//!
//! Neither [`crate`] `TranspositionTable` nor `ProofGoalCache` is deprecated —
//! they remain the right tool when the action axis is irrelevant (single-action
//! search, or when actions are stochastic so `(s, a)` does NOT determine `s'`).
//! This module generalizes both to the `(state, action)` key for the
//! deterministic-transition regime.

use arrayvec::ArrayVec;
use papaya::HashMap as PapayaHashMap;
use std::sync::atomic::AtomicBool;

/// UCB1 exploration constant. Mirrors [`crate::mcts`]'s value (sqrt(2)).
const UCB1_C: f32 = 1.414;

/// Maximum unexpanded actions per node (stack-allocated, no per-node heap).
/// 16 covers typical inference-action spaces (3–6 solver/strategy combos in
/// UnMaskFork's dLLM case; up to ~12 in a multi-shard crowd-NPC analog).
const MAX_UNEXPANDED: usize = 16;

/// Cap on tree nodes, preventing unbounded memory growth. Matches the
/// existing [`crate::mcts`] guard.
const MAX_TREE_SIZE: usize = 10_000;

// ── Action handle ──────────────────────────────────────────────

/// A single discrete action in the MCTS-over-configurations search.
///
/// Generalizes UnMaskFork's `(θ_a, T_a, g_a)` (model, temperature, remasking
/// strategy) to any finite set of inference configurations. The semantics of
/// `config_id` / `strategy_id` are entirely caller-defined: they index into a
/// caller-supplied config table via [`InferenceActionSpace`]. The cache treats
/// this as an opaque 4-byte handle.
///
/// `#[repr(C)]` so the layout is stable across the FFI / freeze-thaw boundary
/// (future riir-chain commitment of cached trajectories). 3 bytes of payload
/// + 1 byte padding = 4 bytes total.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct InferenceAction {
    /// Index into a caller-supplied config table (e.g. which solver /
    /// shard / adapter / temperature bucket).
    pub config_id: u16,
    /// Remasking / sampling strategy enum (caller-defined encoding).
    pub strategy_id: u8,
}

impl InferenceAction {
    /// Convenience constructor.
    #[inline]
    #[must_use]
    pub const fn new(config_id: u16, strategy_id: u8) -> Self {
        Self {
            config_id,
            strategy_id,
        }
    }
}

// ── Cache key ──────────────────────────────────────────────────

/// Cache key: `(state hash, action)`. The `state` hash is BLAKE3 of the
/// caller's state representation (see [`InferenceActionSpace::state_hash`]),
/// NOT the state itself — this keeps the key a fixed 36 bytes regardless of
/// state size.
///
/// Distinct from state-only transposition: two different actions at the same
/// state produce two different keys (and, under the DeterministicTransition
/// contract, two different transitions).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StateActionKey {
    /// BLAKE3 hash of the state.
    pub state: blake3::Hash,
    /// The inference action applied at that state.
    pub action: InferenceAction,
}

// ── Cache ──────────────────────────────────────────────────────
// ── Cache ─────────────────────────────────────────────────

/// Lock-free state-action pair cache.
///
/// Maps `(state_hash, action) → (next_state_hash, reward)`. Once an entry is
/// recorded, revisiting the same `(state, action)` pair is an O(1) lock-free
/// pin (papaya's bucket array), costing zero transition evaluations (NFE in
/// the dLLM framing).
///
/// `R` is the reward type — `f32` for the standard search, but generic to
/// allow richer reward payloads (e.g. `(f32, u32)` for reward + depth).
pub struct StateActionCache<R> {
    inner: PapayaHashMap<StateActionKey, (blake3::Hash, R)>,
    /// When false, `get` always returns `None` and `insert` is a no-op.
    /// Used by the G2 no-cache baseline (Issue 044) to measure the cache's
    /// true benefit without duplicating the search implementation. The flag
    /// is checked with a relaxed atomic load — negligible vs the BLAKE3 hash
    /// that dominates each lookup.
    disabled: AtomicBool,
}

impl<R: Copy> StateActionCache<R> {
    /// Create a cache with papaya's default initial capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: PapayaHashMap::new(),
            disabled: AtomicBool::new(false),
        }
    }

    /// Create a cache with a pre-allocated capacity hint. Useful when the
    /// caller knows the approximate `(state, action)` pair count up front
    /// (e.g. `nfe_budget × avg_actions_per_state`), avoiding rehash churn.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: PapayaHashMap::with_capacity(cap),
            disabled: AtomicBool::new(false),
        }
    }

    /// Create a **disabled** cache: `get` always returns `None`, `insert` is a
    /// no-op. Used by the G2 no-cache baseline (Issue 044) to measure the
    /// cache's true benefit through the same search code path, without
    /// duplicating the search implementation. The capacity is zero because no
    /// entries are ever stored.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            inner: PapayaHashMap::new(),
            disabled: AtomicBool::new(true),
        }
    }

    /// O(1) lock-free lookup. Returns the cached `(next_state_hash, reward)`
    /// for `(state, action)` if present.
    ///
    /// The caller is responsible for honouring the DeterministicTransition
    /// contract (see module docs): a hit is only valid if `apply(state, action)`
    /// still yields the same `next_state`.
    #[inline]
    pub fn get(&self, state: blake3::Hash, action: InferenceAction) -> Option<(blake3::Hash, R)> {
        if self.disabled.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let key = StateActionKey { state, action };
        // `get` acquires a pin; the closure runs under the pin. Returning the
        // copied value out of the pin is fine because (blake3::Hash, R) is Copy.
        self.inner.pin().get(&key).copied()
    }

    /// Insert a cached transition. The caller MUST have observed the transition
    /// deterministically — i.e. `apply(state, action)` was actually evaluated
    /// and produced `next` with reward `reward`, and re-evaluating would
    /// produce the identical result.
    ///
    /// If `(state, action)` already has an entry, it is overwritten with the
    /// new value (the last observation wins; this is correct under the
    /// DeterministicTransition contract since both observations are equal).
    #[inline]
    pub fn insert(
        &self,
        state: blake3::Hash,
        action: InferenceAction,
        next: blake3::Hash,
        reward: R,
    ) {
        if self.disabled.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let key = StateActionKey { state, action };
        self.inner.pin().insert(key, (next, reward));
    }

    /// Number of cached `(state, action)` pairs.
    ///
    /// Note: papaya's `len` is a relaxed estimate (lock-free); it may lag
    /// in-flight inserts by a small amount. For the G5 cache-size-bounded
    /// gate (Phase 3) this is the right granularity — we report the steady-state
    /// count after search completion, not a per-tick exact figure.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the cache is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear all entries. Intended for per-search-scope reset (e.g. between
    /// independent queries that should not share cache state).
    ///
    /// Under the DeterministicTransition contract, NOT clearing between
    /// searches is a valid optimization (revisits across searches are still
    /// hits). Clear when the underlying `InferenceActionSpace` semantics change
    /// in a way that would invalidate old entries (e.g. a different model /
    /// shard set, which changes what `apply` does).
    pub fn clear(&self) {
        // papaya's `clear` takes a guard; `pin()` acquires one implicitly.
        self.inner.pin().clear();
    }

    /// Debug-only determinism re-check. For each `(state, action)` pair in
    /// `samples`, re-applies the action via `space`, hashes the fresh
    /// next-state, and BLAKE3-compares it to the cached next-state hash.
    /// Returns the number of mismatches (0 = contract holds for all samples).
    ///
    /// **This is a diagnostic, not a correctness backstop.** In release builds
    /// the caller's DeterministicTransition contract is trusted (see module
    /// docs). Use this in tests / debug builds to catch contract violations
    /// early (e.g. a bug in `apply` that introduces hidden RNG or thread-local
    /// state).
    ///
    /// The cache stores `(state_hash, action) → (next_hash, reward)`, so it
    /// cannot re-apply actions on its own (BLAKE3 is not invertible). The
    /// caller supplies the concrete states to audit — typically the set of
    /// states the search visited, which the caller already has in hand.
    ///
    /// Skips pairs whose `(state_hash, action)` is not in the cache (a miss
    /// is not a determinism violation, just an un-cached pair).
    #[cfg(debug_assertions)]
    pub fn verify_determinism<S, A>(&self, space: &A, samples: &[(S, InferenceAction)]) -> usize
    where
        S: Clone,
        A: InferenceActionSpace<S>,
    {
        let mut mismatches = 0usize;
        for (state, action) in samples {
            let state_hash = space.state_hash(state);
            let Some((cached_next_hash, _cached_reward)) = self.get(state_hash, *action) else {
                // Not in cache — skip (a miss is not a violation).
                continue;
            };
            let fresh_next = space.apply(state, *action);
            let fresh_next_hash = space.state_hash(&fresh_next);
            if fresh_next_hash != cached_next_hash {
                mismatches += 1;
            }
        }
        mismatches
    }
}

impl<R: Copy> Default for StateActionCache<R> {
    fn default() -> Self {
        Self {
            inner: PapayaHashMap::new(),
            disabled: AtomicBool::new(false),
        }
    }
}

// ── Action space trait ─────────────────────────────────────────

/// The caller-implemented action space over which the search runs.
///
/// This is the modelless, game-IP-free abstraction: the search is agnostic to
/// what `S` is, what the actions mean, or how rewards are computed. The caller
/// wires in concrete semantics (dLLM unmasking step, multi-shard NPC decision,
/// etc.) by implementing these four methods.
///
/// # DeterministicTransition contract
///
/// Implementations MUST guarantee that [`apply`](InferenceActionSpace::apply)
/// is a pure function of `(state, action)`: same inputs → same outputs, every
/// time, on every thread. See the module-level docs for the full contract and
/// the consequences of violation.
pub trait InferenceActionSpace<S> {
    /// The actions available at `state`. The returned slice is borrowed for
    /// the duration of the call only — the search copies the actions it needs
    /// into its stack-allocated node buffers immediately.
    ///
    /// Returning an empty slice marks `state` as terminal-equivalent (the
    /// search will treat it as a leaf even if [`is_terminal`](InferenceActionSpace::is_terminal)
    /// is false).
    fn actions_at(&self, state: &S) -> &[InferenceAction];

    /// Apply `action` to `state`, returning the next state. MUST be
    /// deterministic (pure function of `(state, action)`).
    fn apply(&self, state: &S, action: InferenceAction) -> S;

    /// Terminal reward at `state`, if any. Returning `None` means "not
    /// terminal — keep searching". Returning `Some(r)` freezes the rollout
    /// at this state with reward `r`.
    fn reward(&self, state: &S) -> Option<f32>;

    /// Whether `state` is terminal (no further actions). Terminal states are
    /// never expanded; their reward comes from [`reward`](InferenceActionSpace::reward).
    fn is_terminal(&self, state: &S) -> bool;

    /// BLAKE3 hash of `state`. This is the cache key's state component.
    /// Callers SHOULD hash a canonical byte representation of the state
    /// (two states that are "equal" for search purposes must hash equal).
    fn state_hash(&self, state: &S) -> blake3::Hash;
}

// ── Tree node (index-based, stack-allocated children) ──────────

/// A single node in the search tree. Mirrors the [`crate::mcts::MCTSNode`]
/// design: index-based parent/child links into a flat `Vec`, stack-allocated
/// `ArrayVec` for children/unexpanded (no per-node heap allocation).
///
/// Unlike the game-MCTS node, this node tracks the **action** that produced it
/// (so we can return the best action from the root) and the **state hash**
/// (so we can consult the cache without re-hashing).
struct SearchNode {
    /// Accumulated reward from backpropagation.
    total_reward: f32,
    /// Number of visits through this node.
    visits: usize,
    /// The action that produced this node (`None` for root).
    action_from_parent: Option<InferenceAction>,
    /// Parent node index (`None` for root). Kept for structural parity with
    /// [`crate::mcts::MCTSNode`] and future parent-link backprop paths; the
    /// current search uses an explicit path stack instead, so this field is
    /// written-but-not-read.
    #[allow(dead_code)]
    parent: Option<usize>,
    /// BLAKE3 hash of the state at this node (cached to avoid re-hashing
    /// during selection walks).
    state_hash: blake3::Hash,
    /// Child node indices. Stack-allocated.
    children: ArrayVec<usize, { MAX_UNEXPANDED }>,
    /// Indices of actions not yet expanded into children. Stack-allocated.
    unexpanded: ArrayVec<InferenceAction, { MAX_UNEXPANDED }>,
}

impl SearchNode {
    fn new_root(state_hash: blake3::Hash, actions: &[InferenceAction]) -> Self {
        assert!(
            actions.len() <= MAX_UNEXPANDED,
            "SearchNode::new_root: action_count ({}) exceeds unexpanded capacity ({MAX_UNEXPANDED})",
            actions.len()
        );
        let mut unexpanded = ArrayVec::new();
        // Collect copies (InferenceAction is Copy) — the caller's slice is
        // borrowed only for this call.
        unexpanded.try_extend_from_slice(actions).expect(
            "actions_at returned more than MAX_UNEXPANDED actions; raise the const if a real \
             domain needs more headroom",
        );
        Self {
            total_reward: 0.0,
            visits: 0,
            action_from_parent: None,
            parent: None,
            state_hash,
            children: ArrayVec::new(),
            unexpanded,
        }
    }

    fn new_child(
        state_hash: blake3::Hash,
        action_from_parent: InferenceAction,
        parent: usize,
        actions: &[InferenceAction],
    ) -> Self {
        assert!(
            actions.len() <= MAX_UNEXPANDED,
            "SearchNode::new_child: action_count ({}) exceeds unexpanded capacity ({MAX_UNEXPANDED})",
            actions.len()
        );
        let mut unexpanded = ArrayVec::new();
        unexpanded.try_extend_from_slice(actions).expect(
            "actions_at returned more than MAX_UNEXPANDED actions; raise the const if a real \
             domain needs more headroom",
        );
        Self {
            total_reward: 0.0,
            visits: 0,
            action_from_parent: Some(action_from_parent),
            parent: Some(parent),
            state_hash,
            children: ArrayVec::new(),
            unexpanded,
        }
    }

    #[inline]
    fn is_fully_expanded(&self) -> bool {
        self.unexpanded.is_empty()
    }
}

// ── UCB1 ───────────────────────────────────────────────────────

/// UCB1 score for a child node. Unvisited children return +∞ (always explore
/// first). Mirrors [`crate::mcts`] but takes the cached `ln(parent_visits)`
/// to avoid recomputing the log on every child comparison.
#[inline]
fn ucb1_score(total_reward: f32, visits: usize, ln_parent: f32) -> f32 {
    if visits == 0 {
        return f32::INFINITY;
    }
    let exploit = total_reward / visits as f32;
    let explore = UCB1_C * ln_parent.sqrt() / (visits as f32).sqrt();
    exploit + explore
}

// ── Search scratch (zero-alloc hot path) ───────────────────────

/// Pre-allocated scratch buffers for the search, reused across iterations to
/// keep the hot path allocation-free. Construct once per search via
/// [`SearchScratch::with_capacity`].
///
/// The Phase 3 G4 gate (zero-alloc hot path) verifies that after warmup,
/// `mcts_search_with_state_action_cache` allocates 0 bytes per iteration —
/// this struct is what makes that possible: the action buffer, the selection
/// path stack, and the tree node vector are all pre-sized.
pub struct SearchScratch {
    /// The flat tree node vector. Grown once at search start; nodes are pushed
    /// but the vector is never reallocated mid-search (capacity is sized to
    /// `MAX_TREE_SIZE`).
    nodes: Vec<SearchNode>,
    /// Selection path (root → leaf indices), reused per iteration.
    path: Vec<usize>,
}

impl SearchScratch {
    /// Construct scratch with capacity hints. `node_capacity` should be ≥ the
    /// expected tree size (use `MAX_TREE_SIZE` for the safe default).
    #[must_use]
    pub fn with_capacity(node_capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(node_capacity),
            path: Vec::with_capacity(64),
        }
    }
}

impl Default for SearchScratch {
    fn default() -> Self {
        Self::with_capacity(256)
    }
}

// ── Search entry point ─────────────────────────────────────────

/// Result of a single MCTS search: the best action plus cache statistics.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The best action at the root (highest UCB1-backed visit count), or
    /// `None` if the root had no actions (terminal-equivalent).
    pub best_action: Option<InferenceAction>,
    /// Number of cache hits during this search (Expand-time lookups that
    /// returned a cached transition, skipping the `apply` call).
    pub cache_hits: usize,
    /// Number of cache misses during this search (Expand-time lookups that
    /// fell through to a real `apply` + rollout).
    pub cache_misses: usize,
    /// Total tree nodes at search end.
    pub tree_size: usize,
    /// Total forward steps taken across all rollouts (each `apply` inside
    /// `rollout_to_terminal` counts as one step; cache-hit shortcircuits count
    /// as zero steps because the remaining trajectory was skipped). Used by
    /// the G2 NFE-savings metric: `nfe_saved ≈ cache_hits × (total_rollout_steps / (hits + misses))`.
    pub total_rollout_steps: usize,
}

/// Run MCTS over `space` starting from `root`, consulting `cache` at Expand
/// time. Returns the best action at the root plus cache statistics.
///
/// `budget` is the number of search iterations (each iteration = one
/// Select → Expand → (Cache-hit-or-Rollout) → Backprop cycle). `scratch` is
/// borrowed and reused across iterations for zero-alloc steady state —
/// construct it once per search thread and reuse it across searches to keep
/// the Phase 3 G4 gate (zero-alloc hot path) honest.
///
/// **Note on the `S: Clone` bound:** the search re-walks the selection path
/// from the root each iteration (re-applying actions to reconstruct the leaf
/// state inline, rather than cloning state into every tree node). This needs
/// one `root.clone()` per iteration plus one `apply` per tree level — both
/// cheap for the fixed-size / refcounted state representations every realistic
/// caller uses (dLLM token sequences, NPC belief states, etc.).
///
/// # Algorithm
///
/// 1. **Select**: UCB1 down the tree, re-applying actions to reconstruct the
///    leaf state inline. Records the root→leaf index path.
/// 2. **Expand**: pick an unexpanded action `a` at the leaf.
///    - If `cache.has(leaf_state_hash, a)` → **cache hit**: use the cached
///      `(next_state_hash, reward)` directly, skipping the `apply` + rollout.
///      This is the UnMaskFork budget win.
///    - Else → **cache miss**: call `space.apply(leaf_state, a)`, run a rollout
///      to terminal, and **insert** `(leaf_state_hash, a) → (next_hash, reward)`.
/// 3. **Backpropagate**: walk the recorded path back to the root, accumulating
///    the reward into each node's `(visits, total_reward)`.
///
/// The search is self-contained — it does NOT reuse [`crate::mcts::mcts_search`]
/// (which is parameterized over `GameState` game actions and player IDs, not
/// opaque inference actions).
///
/// # Panics
///
/// Panics if `actions_at` returns more than `MAX_UNEXPANDED` (16) actions at
/// any reachable state. Raise the const if a real domain needs more headroom.
pub fn mcts_search_with_state_action_cache<S, A>(
    space: &A,
    root: &S,
    budget: usize,
    cache: &StateActionCache<f32>,
    scratch: &mut SearchScratch,
) -> SearchResult
where
    S: Clone,
    A: InferenceActionSpace<S>,
{
    // Reset scratch (keep capacity — no realloc).
    scratch.nodes.clear();
    scratch.path.clear();

    let root_hash = space.state_hash(root);
    let root_actions = space.actions_at(root);

    // Terminal-equivalent root: nothing to search.
    if root_actions.is_empty() {
        return SearchResult {
            best_action: None,
            cache_hits: 0,
            cache_misses: 0,
            tree_size: 0,
            total_rollout_steps: 0,
        };
    }

    scratch
        .nodes
        .push(SearchNode::new_root(root_hash, root_actions));

    let mut cache_hits = 0usize;
    let mut cache_misses = 0usize;
    let mut total_rollout_steps = 0usize;

    for _ in 0..budget {
        if scratch.nodes.len() >= MAX_TREE_SIZE {
            break;
        }

        // ── 1. Selection: UCB1 down to a leaf, tracking inline state ──
        // Re-walk from the root each iteration, re-applying actions to
        // reconstruct the leaf state. O(depth) apply calls per iteration —
        // acceptable because (a) depth is small (the domain's transition-graph
        // depth, not the tree size), and (b) it keeps the tree free of
        // per-node state clones (only hashes are stored).
        scratch.path.clear();
        scratch.path.push(0); // root index
        let mut current_state = root.clone();

        let mut leaf_idx = 0usize;
        loop {
            // Terminal or Expand candidate (has unexpanded actions) → stop.
            if space.is_terminal(&current_state) || !scratch.nodes[leaf_idx].is_fully_expanded() {
                break;
            }
            // Fully expanded non-terminal: descend via UCB1.
            let ln_parent = if scratch.nodes[leaf_idx].visits > 0 {
                (scratch.nodes[leaf_idx].visits as f32).ln()
            } else {
                0.0
            };
            let children = &scratch.nodes[leaf_idx].children;
            let best_child = children.iter().copied().max_by(|&a, &b| {
                let sa = ucb1_score(
                    scratch.nodes[a].total_reward,
                    scratch.nodes[a].visits,
                    ln_parent,
                );
                let sb = ucb1_score(
                    scratch.nodes[b].total_reward,
                    scratch.nodes[b].visits,
                    ln_parent,
                );
                // float_order::cmp_for_max (NaN can never win the selection).
                crate::float_order::cmp_for_max(sa, sb)
            });
            match best_child {
                Some(child_idx) => {
                    let action = scratch.nodes[child_idx]
                        .action_from_parent
                        .expect("non-root node must have an action_from_parent");
                    current_state = space.apply(&current_state, action);
                    leaf_idx = child_idx;
                    scratch.path.push(leaf_idx);
                }
                None => break, // no children yet (fresh root, first iteration)
            }
        }

        // ── 2. Expand + (Cache-hit-or-Rollout) ──
        let leaf_hash = scratch.nodes[leaf_idx].state_hash;
        let reward = if space.is_terminal(&current_state) {
            // Terminal leaf: reward comes from the space (or 0 if unset).
            space.reward(&current_state).unwrap_or(0.0)
        } else {
            match scratch.nodes[leaf_idx].unexpanded.pop() {
                // Degenerate leaf: fully expanded, non-terminal, but with no
                // descendable children and no unexpanded actions left. This
                // happens when every action at this state has already been
                // expanded into a terminal child. Fall back to the space's
                // reward (or 0) — no further exploration possible here.
                None => space.reward(&current_state).unwrap_or(0.0),
                // Normal Expand: consult the cache for (leaf_hash, action).
                Some(action) => {
                    let (r, steps) = expand_one(
                        space,
                        &current_state,
                        leaf_hash,
                        leaf_idx,
                        action,
                        cache,
                        scratch,
                        &mut cache_hits,
                        &mut cache_misses,
                    );
                    total_rollout_steps += steps;
                    r
                }
            }
        };

        // ── 3. Backpropagate: walk the recorded path to root ───────
        for &idx in &scratch.path {
            scratch.nodes[idx].visits += 1;
            scratch.nodes[idx].total_reward += reward;
        }
    }

    // ── Pick the best root child by visit count (the standard MCTS action
    //    choice — robust against UCB1 exploration noise). ──
    let best_action = scratch.nodes[0]
        .children
        .iter()
        .copied()
        .max_by_key(|&idx| scratch.nodes[idx].visits)
        .and_then(|idx| scratch.nodes[idx].action_from_parent);

    SearchResult {
        best_action,
        cache_hits,
        cache_misses,
        tree_size: scratch.nodes.len(),
        total_rollout_steps,
    }
}

/// Inner Expand step: consult the cache for `(leaf_hash, action)`, and on a
/// hit short-circuit the rollout; on a miss, apply + rollout + insert. Grows
/// the tree by one child node (a real next-state node on miss; a hash-only
/// leaf on hit). Returns the reward to backpropagate.
#[allow(clippy::too_many_arguments)]
fn expand_one<S, A>(
    space: &A,
    leaf_state: &S,
    leaf_hash: blake3::Hash,
    leaf_idx: usize,
    action: InferenceAction,
    cache: &StateActionCache<f32>,
    scratch: &mut SearchScratch,
    cache_hits: &mut usize,
    cache_misses: &mut usize,
) -> (f32, usize)
where
    S: Clone,
    A: InferenceActionSpace<S>,
{
    if let Some((cached_next_hash, cached_reward)) = cache.get(leaf_hash, action) {
        // ── Cache HIT: skip apply + rollout ──
        *cache_hits += 1;
        // Attach a hash-only leaf child (no actions_at — we don't have the
        // real next state, only its hash). The child's reward is the cached
        // reward; future selection into this child re-runs the cache lookup
        // (still a hit) rather than re-applying. No tree-growth benefit to
        // expanding a cache-hit node here: its children would themselves be
        // cached on revisit.
        let child_idx = scratch.nodes.len();
        scratch.nodes.push(SearchNode {
            total_reward: 0.0,
            visits: 0,
            action_from_parent: Some(action),
            parent: Some(leaf_idx),
            state_hash: cached_next_hash,
            children: ArrayVec::new(),
            unexpanded: ArrayVec::new(),
        });
        scratch.nodes[leaf_idx].children.push(child_idx);
        // Cache hit skips the entire rollout: 0 rollout steps.
        (cached_reward, 0)
    } else {
        // ── Cache MISS: apply + rollout ──
        *cache_misses += 1;
        let next_state = space.apply(leaf_state, action);
        let next_hash = space.state_hash(&next_state);
        let (rollout_reward, rollout_steps) = rollout_to_terminal(space, &next_state, cache);
        // Cache the transition (DeterministicTransition contract: re-applying
        // yields the identical result, so this entry stays valid).
        cache.insert(leaf_hash, action, next_hash, rollout_reward);
        // Grow the tree with the real next state so future iterations can
        // descend into it without a cache lookup.
        let next_actions = if space.is_terminal(&next_state) {
            &[][..]
        } else {
            space.actions_at(&next_state)
        };
        let child_idx = scratch.nodes.len();
        scratch.nodes.push(SearchNode::new_child(
            next_hash,
            action,
            leaf_idx,
            next_actions,
        ));
        scratch.nodes[leaf_idx].children.push(child_idx);
        (rollout_reward, rollout_steps)
    }
}

/// Rollout from `state` to a terminal, using a fixed-action policy (first
/// available action at each step — deterministic, matching the
/// DeterministicTransition contract). Caps rollout depth at a sane bound to
/// prevent infinite loops on pathological non-terminating spaces.
///
/// Consults the cache at each step: a hit returns the cached reward for the
/// remainder of the rollout, saving NFE. Inserts each observed transition.
fn rollout_to_terminal<S, A>(space: &A, state: &S, cache: &StateActionCache<f32>) -> (f32, usize)
where
    S: Clone,
    A: InferenceActionSpace<S>,
{
    const MAX_ROLLOUT_DEPTH: usize = 64;
    let mut current = state.clone();
    let mut current_hash = space.state_hash(state);
    let mut steps = 0usize;

    for _ in 0..MAX_ROLLOUT_DEPTH {
        if let Some(r) = space.reward(&current) {
            return (r, steps);
        }
        if space.is_terminal(&current) {
            return (0.0, steps);
        }
        let actions = space.actions_at(&current);
        if actions.is_empty() {
            return (space.reward(&current).unwrap_or(0.0), steps);
        }
        // First-available-action policy (deterministic). A richer policy
        // (epsilon-greedy, UCB1 within rollout) is a Phase 3+ tuning knob;
        // the cache benefit is independent of the rollout policy.
        let action = actions[0];
        if let Some((_cached_next_hash, cached_reward)) = cache.get(current_hash, action) {
            // Cache hit mid-rollout: return the cached reward for the rest of
            // the trajectory. This is the compounding benefit — one hit can
            // short-circuit the entire remaining rollout. Steps are NOT
            // incremented because the remaining trajectory was skipped.
            return (cached_reward, steps);
        }
        let next = space.apply(&current, action);
        let next_hash = space.state_hash(&next);
        let reward = if space.is_terminal(&next) {
            space.reward(&next).unwrap_or(0.0)
        } else {
            0.0 // intermediate; final reward comes from a future terminal step
        };
        cache.insert(current_hash, action, next_hash, reward);
        current_hash = next_hash;
        current = next;
        steps += 1;
    }
    // Hit the depth cap without a terminal — return the last state's reward
    // (or 0 if none). This is a graceful degradation, not a panic: real domains
    // may have long trajectories and the search still benefits from the
    // partial reward signal.
    (space.reward(&current).unwrap_or(0.0), steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── T1.5 unit tests (the subset that doesn't need a separate test binary) ──

    #[test]
    fn inference_action_is_4_bytes() {
        // The plan requires InferenceAction be 4 bytes (3 payload + 1 pad).
        assert_eq!(
            std::mem::size_of::<InferenceAction>(),
            4,
            "InferenceAction must be 4 bytes (#[repr(C)] u16+u8+pad)"
        );
    }

    #[test]
    fn state_action_key_distinguishes_actions() {
        let state = blake3::hash(b"state-a");
        let a0 = InferenceAction::new(0, 0);
        let a1 = InferenceAction::new(1, 0);
        let k0 = StateActionKey { state, action: a0 };
        let k1 = StateActionKey { state, action: a1 };
        assert_ne!(k0, k1, "different actions at the same state must differ");
    }

    #[test]
    fn state_action_key_distinguishes_states() {
        let sa = blake3::hash(b"state-a");
        let sb = blake3::hash(b"state-b");
        let a = InferenceAction::new(0, 0);
        let ka = StateActionKey {
            state: sa,
            action: a,
        };
        let kb = StateActionKey {
            state: sb,
            action: a,
        };
        assert_ne!(ka, kb, "same action at different states must differ");
    }

    #[test]
    fn cache_round_trip_deterministic() {
        let cache: StateActionCache<f32> = StateActionCache::new();
        let s = blake3::hash(b"state");
        let a = InferenceAction::new(0, 0);
        let next = blake3::hash(b"next");
        cache.insert(s, a, next, 0.75);
        let got = cache.get(s, a).expect("inserted entry must be retrievable");
        assert_eq!(got.0, next);
        assert!((got.1 - 0.75).abs() < 1e-6);
    }

    #[test]
    fn cache_clear_empties() {
        let cache: StateActionCache<f32> = StateActionCache::new();
        let s = blake3::hash(b"state");
        let a = InferenceAction::new(0, 0);
        cache.insert(s, a, blake3::hash(b"next"), 1.0);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(
            cache.get(s, a).is_none(),
            "cleared entry must not be retrievable"
        );
    }

    #[test]
    fn cache_same_state_different_action_distinct_entries() {
        let cache: StateActionCache<f32> = StateActionCache::new();
        let s = blake3::hash(b"shared-state");
        let a0 = InferenceAction::new(0, 0);
        let a1 = InferenceAction::new(1, 0);
        cache.insert(s, a0, blake3::hash(b"next-0"), 0.1);
        cache.insert(s, a1, blake3::hash(b"next-1"), 0.9);
        assert_eq!(cache.len(), 2, "two actions at one state = two entries");
        let g0 = cache.get(s, a0).expect("a0 must hit");
        let g1 = cache.get(s, a1).expect("a1 must hit");
        assert!((g0.1 - 0.1).abs() < 1e-6);
        assert!((g1.1 - 0.9).abs() < 1e-6);
        assert_ne!(
            g0.0, g1.0,
            "different actions must lead to different next states"
        );
    }

    #[test]
    fn cache_disabled_never_stores_or_returns() {
        // Issue 044: the disabled() constructor is the true no-cache baseline.
        // get() always returns None; insert() is a no-op; len() stays 0.
        let cache: StateActionCache<f32> = StateActionCache::disabled();
        let s = blake3::hash(b"state");
        let a = InferenceAction::new(0, 0);
        // Insert should be a no-op.
        cache.insert(s, a, blake3::hash(b"next"), 0.5);
        assert!(cache.is_empty(), "disabled cache must not store entries");
        assert_eq!(cache.len(), 0);
        // Get must always miss.
        assert!(cache.get(s, a).is_none(), "disabled cache must always miss");
    }
}
